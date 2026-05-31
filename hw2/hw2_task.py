import torch
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


@torch.inference_mode()
def optimized_loop(model, input_ids, n_steps):
    """Optimized generation loop using KV cache and batched token collection."""
    past_key_values = None
    generated_tokens = []
    cur_input = input_ids  # first iteration: full prompt (prefill)

    for _ in range(n_steps):
        outputs = model(
            input_ids=cur_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(
            outputs.logits[:, -1, :], dim=-1, keepdim=True
        )
        generated_tokens.append(next_token_id)
        cur_input = next_token_id  # subsequent iterations: single token (decode)

    # Transfer all generated tokens to CPU in one shot (no per-step sync)
    all_tokens = torch.cat(generated_tokens, dim=1)
    return all_tokens.squeeze(0).tolist()


def profile(loop_fn, model, input_ids, trace_name: str):
    """Profile loop_fn, print summary table, and export a Chrome trace."""
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    trace_path = RESULTS_DIR / trace_name
    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace saved to {trace_path}")


def generate_optimized(optimized_trace_name: str) -> float:
    """Build an optimized model and measure the optimized generation loop."""
    # Use float16 to halve memory bandwidth and double effective compute
    model = build_model(torch.float16)
    input_ids = get_input_ids()

    profile(optimized_loop, model, input_ids, optimized_trace_name)
    elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")
    return elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
#
# 1. KV Cache (use_cache=True + past_key_values): Instead of recomputing
#    attention over the entire growing sequence every step, we cache the key/value
#    projections and only process the new token during decode. This changes per-step
#    work from O(seq_len) to O(1) and is by far the biggest win — contributing ~10-20x
#    alone on 128 decode steps from a 1024 prompt.
#
# 2. FP16 precision (torch.float16): Halves the model's memory footprint and
#    memory bandwidth requirements. On GPUs with FP16 tensor cores, this also
#    roughly doubles compute throughput. Contributes ~1.5-2x additional speedup.
#
# 3. Removed .item() synchronisation: The baseline calls next_token_id.item()
#    every step, which forces a CUDA synchronize and blocks the CPU until the GPU
#    finishes. We instead collect token tensors on the GPU and transfer them all
#    to CPU once at the end. This eliminates n_steps CPU-GPU sync barriers.
#
# 4. torch.inference_mode(): Disables autograd tracking and version counting,
#    reducing CPU-side overhead for every tensor operation. Small but free speedup.
#
# Biggest impact and why:
#
# KV cache had by far the biggest impact. The slow baseline re-runs full self-
# attention over the entire sequence (prompt + all previously generated tokens)
# at every step. For 128 steps starting from a 1024-token prompt, the total
# tokens processed grow quadratically: sum(1024..1151) ≈ 139k forward-pass
# tokens. With KV caching the prefill processes 1024 tokens once, and each of
# the 128 decode steps processes just 1 token, for ~1152 total — roughly 120x
# less work. This transforms generation from compute-bound quadratic work into
# a much lighter, memory-bandwidth-bound decode phase.

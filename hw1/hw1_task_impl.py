import torch


# ============================================================================
# Part 1: Implement PyTorch Functions
# ============================================================================
#
# TASK 1a: Implement an operation with the lowest arithmetic intensity.
# Use an op that performs essentially memory traffic with ~0 useful FLOPs
# per element.


def lowest_ai_fn(x: torch.Tensor) -> torch.Tensor:
    """Lowest arithmetic intensity baseline (0 FLOP/Byte)."""
    return x.clone()


# TASK 1b: Implement a function with configurable arithmetic intensity.
# Build an element-wise compute operation where work increases with `num_ops`.
# Design it so fused arithmetic intensity grows roughly linearly with `num_ops`,
# while each element is still read/written once at the kernel boundary.
# Return either the eager function or a compiled version depending on the
# `compiled` flag so we can compare both on the roofline plot.
#
# Use an accumulator variable and implement fused multiply-add (FMA) style work
# explicitly, e.g. `acc = acc * x + x`, so each loop iteration contributes
# about 2 FLOPs per element in a realistic GPU-friendly pattern. We prefer this
# pattern here mainly because it gives clean FLOP accounting and resembles the
# kind of floating-point work GPUs are designed to do; Avoid patterns like repeated
# doubling (`x = x + x`), since long self-dependent pointwise chains can trigger
# very poor Inductor compile-time behavior and are also less useful for this
# roofline exercise.


def make_compute_fn(num_ops: int, compiled: bool = True):
    """Return an eager or compiled function whose work scales with num_ops."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        acc = x
        for _ in range(num_ops):
            acc = acc * x + x
        return acc

    return torch.compile(fn) if compiled else fn


# ============================================================================
# Part 2: Benchmarking
# ============================================================================
#
# TASK 2: Complete the benchmark function using CUDA events.
# CUDA events measure GPU time precisely (not CPU wall time), which avoids
# including kernel launch overhead or CPU-GPU synchronization delays.


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a GPU function using CUDA events.

    Returns median execution time in milliseconds.
    """
    # Warmup (triggers torch.compile on first call, then warms caches)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # Time `rep` runs using CUDA events and return median latency (ms)
    times = []
    for _ in range(rep):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        fn(*args)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    times.sort()
    return times[len(times) // 2]


# TASK 3: Compute element-wise operation metrics from measured runtime.
# Count every arithmetic operation performed inside the loop (careful: each
# `acc = acc * x + x` iteration does more than one FLOP per element).
#
# Use different byte-traffic models for the two variants:
#   - compiled: assume the operation is fused, so each element is read once and
#     written once at the kernel boundary
#   - eager: estimate the traffic from the separate multiply and add operations
#     launched by PyTorch in each loop iteration, including intermediate tensors
#
# Return a tuple with:
#   - total_flops
#   - arithmetic_intensity  (FLOP / Byte)
#   - achieved_flops        (FLOP / s)


def compute_elementwise_metrics(num_elements, num_ops, bytes_per_element, ms, variant):
    # Each iteration of `acc = acc * x + x` does 1 multiply + 1 add = 2 FLOPs/element
    total_flops = 2 * num_ops * num_elements

    if variant == "compiled":
        # Fused kernel: read x once, write result once
        total_bytes = 2 * num_elements * bytes_per_element
    else:
        # Eager: each iteration launches separate mul and add kernels.
        # mul: reads acc + x (2 tensors), writes tmp (1 tensor) = 3 * n * bpe
        # add: reads tmp + x (2 tensors), writes acc (1 tensor) = 3 * n * bpe
        # Per iteration: 6 * n * bpe.  Total: 6 * num_ops * n * bpe.
        total_bytes = 6 * num_ops * num_elements * bytes_per_element

    ai = total_flops / total_bytes
    achieved_flops = total_flops / (ms * 1e-3)

    return total_flops, ai, achieved_flops


# ============================================================================
# Part 3: Short Writeup
# ============================================================================
# Answer these after you generate `results/roofline.png` and inspect the points.
#
# Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
# Why does performance rise as arithmetic intensity increases even though the
# measured runtime changes only a little?
#
# A1. In the memory-bound region, runtime is dominated by memory traffic (read x,
# write result), which stays constant regardless of how many FMA iterations the
# fused kernel performs. Since the kernel does more FLOPs in roughly the same
# time, the achieved FLOP/s (= FLOPs / time) rises linearly with num_ops.
# The points climb the memory-bandwidth ceiling line because
# FLOP/s = bandwidth × AI, and AI = num_ops / bytes_per_element grows with K.
#
# Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
# `128 ops` compiled element-wise operation. Give one or two reasons why that can
# happen on a large GPU like an H100.
#
# A2. (1) A 1024×1024 matmul may not generate enough thread blocks to fully
# occupy all SMs on a large GPU like the H100, leading to underutilization
# ("tail effect"). (2) The matmul has additional overhead from cuBLAS library
# dispatch and potentially suboptimal tiling for that specific matrix size,
# whereas the fused elementwise kernel has a trivially parallel workload that
# maps very efficiently to all available SMs.
#
# Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
# did for smaller operations. What does that suggest about what resource is
# becoming the bottleneck?
#
# A3. The operation is crossing the ridge point and transitioning from
# memory-bound to compute-bound. At 64 ops, the fused AI (64/4 = 16 FLOP/Byte)
# is near the ridge point. At 128 ops (AI = 32), the kernel has passed the ridge
# and is now limited by compute throughput rather than memory bandwidth.
# Additional FLOPs can no longer be "hidden" behind memory latency, so runtime
# rises proportionally to the extra compute.
#
# Q4. Why do the eager `ops-K` points look so different from the compiled ones?
#
# A4. In eager mode, each iteration launches separate mul and add kernels that
# each read their inputs from and write their outputs to global memory. This
# materialises intermediate tensors, so the total byte traffic scales linearly
# with num_ops (6 × K × N × bpe), keeping the effective AI constant at
# 2/(6×4) ≈ 1/12 FLOP/Byte regardless of K. All eager points therefore cluster
# at the same low AI near the memory-bandwidth ceiling, unlike compiled points
# which move rightward as K increases because the fused kernel only touches
# global memory once.

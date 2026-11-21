# gfx1250 fp8 decode attention over the sglang SHUFFLE 5D paged KV cache (prototype)

Standalone HIP prototypes, not built by CMake. They are the starting point for a CK kernel and
are kept here so the work is not lost; each file has its own host driver with a CPU reference
and event / HIP-graph timing.

Layout (page = 16 tokens, fp8 e4m3 K/V, per-tensor k/v descales):

    K: [num_blocks][hkv][D/16][16 tok][16 B]
    V: [num_blocks][hkv][1][D][16 tok]

Q is bf16 `[batch][hq][D]` (quantized to fp8 per row in the kernel), output bf16, one query
token per sequence, up to 16 q heads per kv head, D = 128, optional sliding window.

Grid `(num_splits, hkv, batch)`; the splits of one (batch, kv head) form a workgroup cluster
(`cluster_dims` is a template argument, at most 15) and merge in-kernel after a cluster barrier.

- `decode5d_coop.cpp`: 4 waves take one page each per 64-token step (QK `S^T = K Q^T` with
  `v_wmma_f32_16x16x128_fp8_fp8`); row max and P are exchanged through LDS (2 barriers per
  step); PV `O^T += V^T P^T` with `v_wmma_f32_16x16x64_fp8_fp8`, each wave owning 32 head dims.
  Better for short ranges.
- `decode5d_wave_split.cpp`: each wave owns 4 pages per step with its own online softmax in
  registers (no per-step barriers) and computes all 128 head dims; the 4 waves merge once at
  the end of the split. Better for long ranges.

Build and run (b, ctx, window, pages per split (0 = heuristic), check):

    hipcc -O3 --offload-arch=gfx1250 decode5d_wave_split.cpp -o ws
    ./ws 1 8192 2048 0 1
    STAMPS=1 ./ws 1 8192 2048 16 0   # per-phase wall-clock stamps

# Why Direct DRAM→Register Input Loading Underperforms LDS Staging (conv_v3)

This document records a failed optimisation attempt on the
`conv_32c_tile` v3 kernel and explains why it failed. The lesson is
durable: in this kernel family the input LDS double-buffer is **not pure
overhead**, it is an implicit workgroup-shared input tile cache. Removing
it shifts the bottleneck from LDS to the vector L1 cache (vL1D) and
makes the kernel slower.

The intent of this note is twofold: explain the result to the next
person who looks at the LDS pool and thinks "I can just eliminate this",
and document what vL1D is and why its bandwidth ceiling matters when
designing kernels for CDNA.

---

## 1. The motivating measurement

`rocprof-compute` comparison of the v3 direct kernel vs an implicit-GEMM
baseline on the rocprof-target shape (N=2, C=128, K=256, H=W=256,
R=S=3) showed:

| Signal                       | Direct v3 baseline | IGEMM    |
|------------------------------|--------------------|----------|
| LDS bytes / workgroup        | 4.3 KB             | 1.6 KB   |
| "Insufficient CU LDS" stall  | **4.3 %**          | 0 %      |
| LDS Cmd FIFO Full            | 2.52 M cycles      | 199 K    |
| "Insufficient SIMD VGPRs"    | 4.3 %              | —        |
| MFMA utilisation             | 1.31 %             | high     |

The direct kernel was LDS-occupancy-throttled, not VGPR-throttled. The
v3 LDS pool is `max(weight_all_waves, input_double_buffer + reduce)`;
the input double-buffer dominates (`2 × 18 × 16 = 576 uint4 = 9.2 KB`
at this instance). The hypothesis was: eliminate the input LDS, drop
the pool to `max(weight, reduce)`, run more concurrent workgroups per CU,
unblock MFMA.

## 2. What was tried

A new loader `ConvInputLoaderRegV3` was written that issues
`amd_buffer_load` calls directly into VGPRs, one per S column variant
of the sliding window. The compute loop was forked into a
`_reg_input` sibling that drops `INPUT_TOTAL` from the unified LDS
pool and the `__syncthreads()` calls that ordered LDS writes against
reads. A new `InputLoadStrategy` Config enum selected between the two
strategies; a new Config 48 entry wired the reg-input strategy into
the same shape as Config 2 (the existing LDS-input instance on this
shape).

Correctness was verified end-to-end against the on-device reference
kernel (8/8 integration tests green on MI355). The kernel produces
identical outputs.

## 3. The result

| Variant                                | Time     | TFlops | Δ vs Config 2 |
|----------------------------------------|----------|--------|---------------|
| Config 2  — LDS-input baseline         | 0.516 ms | 149.9  | —             |
| Config 48 — reg-input, no overlap      | 0.697 ms | 111.0  | **+35 %**     |
| Config 48 — reg-input, fetch/MFMA overlap | 0.681 ms | 113.5  | **+32 %**     |

The reg-input variant was 32 % slower, not faster. Reordering the
`wait_vmcnt` to enable MFMA-vs-load overlap gained only 2.4 %.

## 4. Why: the rocprof-compute deep dive

`rocprof-compute` comparison (same shape, two workloads):

| Metric                    | Config 2 (LDS) | Config 48 (Reg) | Δ          |
|---------------------------|----------------|------------------|------------|
| **VMEM Latency** (cycles) | **385**        | **2 321**        | **+503 %** |
| vL1D Hit Rate             | 71.3 %         | 95.8 %           | +34 %      |
| vL1D Bandwidth Util       | 5.6 %          | 25.6 %           | +358 %     |
| **vL1D Utilization**      | 52.6 %         | **98.3 %**       | +87 %      |
| Input Buffer Stalled on L2 | 28 K cyc      | 95 K cyc         | +235 %     |
| **MFMA Utilization**      | 5.80 %         | **4.40 %**       | −24 %      |
| IPC                       | 0.44           | 0.26             | −40 %      |
| VALU Co-Issue Efficiency  | 0.91           | 0.19             | −80 %      |

Three signals matter:

1. **vL1D Utilization went from 52 % to 98 %.** The vector L1 cache is
   the new binding resource. Every wave on the CU competes for it.
2. **VMEM Latency exploded 6×.** Each `buffer_load` now waits roughly
   six times longer for its data because the cache is queueing
   requests. The whole MFMA pipeline stalls behind that wait.
3. **MFMA Utilization actually dropped** (5.8 → 4.4 %). The fix made
   the original symptom worse. Eliminating the LDS stall (4.3 % of
   execution) cost ~10× more in cache-saturation stalls.

The LDS pool savings were real — the workgroup uses ~5 KB less LDS —
but the savings never converted to throughput because the new
bottleneck (vL1D) is harder to escape than the old one (LDS pool size).

## 5. Background: vL1D

The vector L1 data cache (vL1D, also called TCP — Texture Cache Per pipe)
is the per-CU scratchpad that absorbs every `buffer_load` /
`global_load` issued by every wave running on that CU. On MI355:

- **One vL1D per CU**, shared by all simultaneously-resident waves on
  that CU (up to 32 waves × 8 SIMDs).
- **Cache line: 128 bytes.** A coherent group of lanes that touches a
  single 128-byte aligned range issues **one** L1 request. A group that
  touches 16 separate 128-byte ranges issues **16** L1 requests.
- **Bandwidth ceiling**: a single CU's vL1D can service roughly one
  cache-line lookup per cycle (architecture-specific peak depends on
  port count). When traffic exceeds this, requests queue and waves
  stall on `s_waitcnt vmcnt(N)`.
- **Hits do not have zero cost.** A 95 % hit rate is excellent for
  reuse, but if the *throughput* of requests exceeds the cache's
  service rate, hits still queue. `vL1D Utilization` measures how
  often the cache is actively servicing a request — at 98 % the cache
  has no headroom, regardless of hit rate.

vL1D is the choke point between "DRAM is fast enough" (it is, on this
shape — HBM stall is < 0.1 %) and "the SIMD is fed". When the choke
point is saturated, adding more loads at it makes everything wait.

The L2 cache sits behind vL1D, shared across all CUs. The
"Input Buffer Stalled on L2" metric (28 K → 95 K cycles) shows that
even though most reg-input loads hit vL1D (95.8 %), the misses that
do reach L2 face a deeper backlog because more total traffic flows.

## 6. Root cause: kw× the loads, with a strided lane mapping

Two factors compound. Either alone would hurt; together they saturate
vL1D.

### Factor 1 — loss of inter-lane sharing

In the LDS path, every input row enters the workgroup through **one
asynchronous `amd_async_buffer_load`** per thread that writes
directly into LDS (DRAM→LDS, no register staging). All MFMA consumers
within the workgroup then read their slice from LDS at near-register
speed (LDS is a tiny SRAM with much higher bandwidth than vL1D). One
DRAM-side fetch feeds:

- `kw` distinct S column variants of the sliding window (re-read by
  every MFMA), and
- all lanes that need that spatial column (16 lanes per MFMA group at
  16×16×32).

The LDS double-buffer is, in this view, a **workgroup-local input
tile cache** sized to one row. Its cost is the LDS pool slot and one
`__syncthreads()` per fill. Its benefit is that the kw consumers
share one fetch.

The reg path eliminates the shared scratchpad. Each lane now issues
**kw independent `buffer_load`s** per row (one per S column variant)
to populate its own register slots. The kw factor is exact: where
the LDS path issues `BLOCK_W × BLOCK_C8` loads per row across the
workgroup, the reg path issues `kw × WAVE_LANES × NUM_WAVES` loads
— for kw=3, ~3× the vmem traffic for the same input data.

vL1D's hit rate goes up (95.8 %) because the re-fetched data is
mostly the same, but the *request count* multiplies and the cache
saturates.

### Factor 2 — strided, low-coalescing lane mapping

The reg-input loader maps lanes to match the MFMA B-operand layout
([conv_input_loader_reg_v3.hpp:74-99]):

```
lane     = threadIdx.x % 64
lane_q   = lane % MFMA_M     // spatial column (0..15 at 16×16×32)
lane_c8  = lane / MFMA_M     // C8 chunk along this wave's C-slice
c_start  = wave * CPG + lane_c8 * 8
```

For the Config 48 instance (MFMA_M=16, CPG=128, fp16), 16 adjacent
lanes in the same MFMA group share `lane_c8` and differ in `lane_q`.
With NHWC layout, their byte addresses are:

```
addr(lane=k, S=s) = ((block_q + k + s − px) * C + c_start) * sizeof(fp16)
```

`C = 128` and `sizeof(fp16) = 2`, so each lane's `input_col` step adds
**256 bytes** to the address. Sixteen lanes thus read sixteen
addresses spaced 256 bytes apart, with each lane requesting a
16-byte burst (`fp16x8_t`).

A 128-byte cache line holds at most one of these per lane. **Sixteen
lanes generate sixteen separate cache-line lookups** instead of the
one or two that a coalesced load would. The hardware "Coalescing"
metric stays low (25 %) for both kernels because both read NHWC, but
the LDS path uses a `tile_window` distribution explicitly chosen to
maximise coalescing on the DRAM side and then *re-shuffles via LDS*
to match MFMA. The reg path has no LDS to re-shuffle through, so it
is forced to use a lane mapping that matches the MFMA layout
directly — which is the wrong shape for DRAM coalescing in NHWC.

The two factors stack:

- factor 1: 3× more loads per row,
- factor 2: each load uses 16× more cache lines than a coalesced load
  would.

Either alone might fit under the vL1D ceiling. The product does not.

## 7. When could the reg path win?

The diagnosis suggests narrow conditions:

- **kw = 1.** No column-variant multiplication; one DRAM load per row
  per lane. Loss of sharing is much smaller.
- **Very small C.** If C is small enough that the strided lane mapping
  still fits inside one or two cache lines per warp, factor 2
  collapses.
- **Spatial reuse is low.** When the working set is so large that the
  LDS cache effectively cold-misses anyway, the LDS savings have
  nothing to amortise against.

For the target shape (kw=3, C=128, H=W=256, NHWC) none of these hold.
The LDS path wins decisively.

## 8. What to try next

The original concern — `Insufficient CU LDS` 4.3 % stall, LDS Cmd FIFO
saturation 2.52 M cycles — is real but secondary. Cheaper attacks on
the LDS pool that **preserve workgroup-shared staging** are still
worth trying:

- **Single-buffered LDS input.** Halve the input LDS pool by removing
  the ping-pong, accept the loss of fetch/MFMA overlap, see if the
  occupancy win exceeds the latency-hiding loss. This is the next
  experiment.
- **Tighter unified pool.** The `max(weight_all_waves, input + reduce)`
  arithmetic may have slack at specific instances; an audit could
  shrink the binding case without changing strategy.

If single-buffering also fails to help (or hurts), the takeaway
becomes: on this kernel the LDS pool size is not the right knob, and
the 4.3 % stall is simply the price of the input cache.

## 9. References

- The failed reg-input implementation is preserved in git history;
  the commit that reverts it includes the file paths.
- rocprof-compute workloads for the comparison are in
  `build-gfx950-full/workloads_cfg{2,48}_reg/` (kernel-filtered to
  the target kernel only).
- Profiler instance numbers at the time of measurement:
  - Config 2  (LDS-input baseline) — instance 370
  - Config 48 (reg-input)          — instance 371
- `amd_buffer_load_impl` —
  `include/ck_tile/core/arch/amd_buffer_addressing.hpp`
- The LDS-staged loader —
  `include/ck_tile/ops/direct_convolution/kernel/conv_32c_tile_impl_v3.hpp:439`
- The LDS-staged async DRAM→LDS path —
  `include/ck_tile/ops/direct_convolution/kernel/grouped_conv_input_loader.hpp`

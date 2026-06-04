# gfx1151 Deep Fusion — Clock, Harness Check & Native int8/int4 Pipeline Estimate (2026-06-04)

Target board: remote Windows 11 Strix Halo (Ryzen AI APU), GPU reported as
`gfx115E` (garbled stepping) → built/run as **`gfx11-generic`**. Runs are the
fused `concat → conv0 3x3 → ReLU → conv1 1x1 → ReLU → 2x2 s2 maxpool` kernel,
single kernel, no HBM intermediates.

## TL;DR

- **Benchmark harness is clean** — it times *kernel execution only* (HIP events
  around a back-to-back async launch loop; H2D/memset outside the window, no D2H
  inside).
- **Peak SCLK = 2.8 GHz**; the guest is a **virtualized partition of 4 WGP = 8 CUs**
  (full Strix Halo is 40 CUs). **Clock boosting is not available** from the guest
  (`amd-smi` backend won't load; only guest/host SMI libs present).
- **Native integer WMMA is supported** on this arch: `llc` lowers
  `llvm.amdgcn.wmma.i32.16x16x16.iu8`/`iu4` to `v_wmma_i32_16x16x16_iu8`/`iu4`
  for both `gfx1151` and `gfx11-generic`. No fp16 emulation required.
- **Genuine int pipeline compute floor ≈ 1.94 ms (useful) / 2.2 ms (padded)**,
  ~2.3× below the fp16-emulation floor. Memory never binds (~560 OP/byte).
- **Realistic near-term target ≈ 11–16 ms (~2–3×)** over today's best
  (direct, fp16 emul, 32.5 ms); the kernel stays VALU/coordinate-arithmetic
  bound, so the matrix floor is only reachable by also cutting scalar work.

## Target datatype pipeline (intended low-precision graph)

| op | dtype | shape in → out | kh×kw | ops |
|----|-------|----------------|-------|-----|
| Concat | int8 → int8 | 2×(1,4,2160,3840) → (1,8,2160,3840) | | |
| Conv0 | int8 → **int32** | (1,8,…) → (1,32,…) | 3×3 | 38.2 GOP |
| QuantizeLinear | int32 → int8 | | | |
| ReLU | int8 → int8 | | | |
| QuantizeLinear | int8 → int4 | | | |
| Conv1 | int4 → **int32** | (1,32,…) → (1,24,…) | 1×1 | 12.7 GOP |
| QuantizeLinear | int32 → int4 | | | |
| ReLU | int4 → int4 | | | |
| MaxPool | int4 → int4 | (1,24,2160,3840) → (1,24,1080,1920) | 2×2 s2 | |
| QuantizeLinear | int4 → int4 | | | |
| **Fusion** | int8 → int4 | 2×(1,4,2160,3840) → (1,24,1080,1920) | | **51.0 GOP** |

Conv0 uses `wmma_i32_16x16x16_iu8` (int8×int8→int32); conv1 uses
`wmma_i32_16x16x16_iu4` (int4×int4→int32). `QuantizeLinear` = int32·scale →
round → clamp, folded into the fused epilogues. GOP counts are 2×MAC, matching
the WMMA OP units used below.

## Harness verification (kernel-only timing)

`deep_fused_conv_pool_verify.py::_benchmark_artifact`:
- H2D copies + `memset` happen **before** the timed region.
- Timing: `start.record()` → `iters`× fire-and-forget `rt.launch`
  (`record_event=False`, no per-launch sync) → `end.record()` →
  `end.synchronize()` → `hipEventElapsedTime / iters`
  (`runtime/hip_module.py` `Event.record`/`elapsed_to`).
- No D2H inside the window; warmup forced ≥100 then `sync()` before timing.

Conclusion: reported `mean_ms` is pure GPU-stream execution time for one kernel
(no host overhead, no copies). For a 30–55 ms kernel the ~1 µs launch gaps are
negligible.

## Device facts (queried via `hipDeviceGetAttribute`, anchors validated)

| Attribute | Value | Note |
|-----------|-------|------|
| Peak SCLK (ClockRate) | **2.8 GHz** | peak the partition reports |
| Memory clock | 2.2 GHz | |
| Memory bus width | 128-bit | |
| MultiprocessorCount | **4** | = 4 WGP |
| MaxThreadsPerMultiProcessor | 2048 | ⇒ multiprocessor = WGP (2 CUs) ⇒ **8 CUs** |
| Integrated | 1 | APU |
| gcnArchName | gfx115E | → gfx11-generic |
| MaxThreadsPerBlock / WarpSize | 1024 / 32 | anchors (confirm enum offsets) |

Full Strix Halo = 40 CUs; this guest got a ~1/5 slice. **No clock control** in
the guest: `amd-smi` fails `Error LoadLibraryA`, and only
`libamdsmi_guest.dll` / `libamdsmi_host.dll` exist (virtualized GPU) — there is
no exposed clock-boost surface.

## Native integer WMMA — confirmed in toolchain

ROCm 7.2.0 / AMD LLVM 22.0.0git. Minimal IR compiled with
`llc -mcpu=gfx1151` and `-mcpu=gfx11-generic`:

```
v_wmma_i32_16x16x16_iu8 v[0:7], v[8:11], v[12:15], v[0:7]   ; A,B = <4 x i32>, C/D = <8 x i32>
v_wmma_i32_16x16x16_iu4 v[0:7], v[8:9],  v[10:11], v[0:7]   ; A,B = <2 x i32>, C/D = <8 x i32>
```

Intrinsic signatures (gfx11): boolean signedness flag precedes each matrix
operand (`unsignedA`, `unsignedB`), trailing `clamp` saturates the int32 result.
A frag packs 16×int8 = `<4 x i32>` (iu8) or 16×int4 = `<2 x i32>` (iu4); the
accumulator is `<8 x i32>`. (The clamp operand placement has seen recent
revert/rework churn in upstream LLVM but is present and functional here.)

The DSL does **not** yet declare these atoms — `arch_specs.json` lists only
fp16/bf16 WMMA, and `Gfx11RdnaBackend.emit_wmma` (`core/isa/backend.py`) only
maps `_RDNA_WMMA` f16/bf16. Wiring iu8/iu4 requires: (1) add the two `mma`
atoms to the gfx1151 + gfx11-generic arch specs, (2) add intrinsic mappings +
the i1 sign/clamp args and i32-accumulator path to `emit_wmma`, (3) int
fragment/dtype handling in IR/lowering.

## Throughput rates (8 CU × 2.8 GHz partition)

RDNA3 integer WMMA = 2× (int8) / 4× (int4) the fp16 rate (ratios reproduce
AMD's 123 TFLOPS / 246 TOPS / 492 TOPS on the 96-CU 7900 XTX):

| Precision | Instr | OP/CU/clk | Partition peak |
|-----------|-------|-----------|----------------|
| fp16 (current emulation) | `wmma_f32…f16` | 512 | **11.5 TFLOP/s** |
| int8 (conv0) | `wmma_i32…iu8` | 1024 | **22.9 TOP/s** |
| int4 (conv1) | `wmma_i32…iu4` | 2048 | **45.9 TOP/s** |

## Compute floor — genuine int pipeline

| Stage | Datatype | Useful | Pad | Padded | Peak | t_useful | t_padded |
|-------|----------|--------|-----|--------|------|----------|----------|
| conv0 3×3 | int8→i32 | 38.2 GOP | K 72→80 | 42.4 GOP | 22.9 TOP/s | 1.67 ms | 1.85 ms |
| conv1 1×1 | int4→i32 | 12.7 GOP | N 24→32 | 16.9 GOP | 45.9 TOP/s | 0.28 ms | 0.37 ms |
| **total** | | **51.0 GOP** | | 59.3 GOP | | **1.94 ms** | **2.22 ms** |

**Hard compute ceiling ≈ 1.94 ms useful / 2.22 ms padded → ~23–26 TOP/s.**
vs fp16-emulation ceiling 4.4 ms → native int is **~2.3× faster on the matrix
floor alone**.

## Memory floor (fully fused)

- In 2×(1,4,2160,3840) int8 = 66.4 MB; out (1,24,1080,1920) int4 = 24.9 MB;
  weights negligible → **~91 MB** HBM traffic.
- Arithmetic intensity ≈ 51 GOP / 91 MB ≈ **560 OP/byte** → compute-bound.
- At 100–256 GB/s: **0.36–0.91 ms** — well under the compute floor. Memory does
  not bind; VALU does.

## Measured baseline (fp16 emulation, full shape, bit-exact)

All configs verified bit-exact (`max_abs_diff=0`, integer-exact reference,
bad 0/49,766,400) on the board.

| Config | mean_ms | useful TFLOP/s | % of fp16 peak |
|--------|---------|----------------|----------------|
| direct (pt 2×16) | 32.51 | 1.567 | 13.6% |
| im2col (pt 4×8) | 55.87 | 0.912 | 7.9% |

## Targets & expected speedup

The kernel is **VALU / coordinate-arithmetic bound** (gfx950 study: matrix unit
lightly used, scalar work on the critical path); on RDNA the WMMA and VALU share
the SIMD issue port, so the matrix floor is not reachable directly. Native int
also helps the VALU side — it deletes the fp16-emulation overhead (`rint` snap,
fp16 pack/unpack) and int8/int4 data is 2×/4× denser → much less LDS-staging
traffic and addressing VALU. Coordinate arithmetic is precision-independent and
remains.

| | Latency | Eff. TOP/s | vs current direct |
|--|---------|------------|-------------------|
| Current direct (fp16 emul) | 32.5 ms | 1.57 | 1.0× |
| Current im2col (fp16 emul) | 55.9 ms | 0.91 | — |
| **Realistic near-term (native int, VALU still partly binding)** | **~11–16 ms** | **~3.2–4.6** | **~2–3×** |
| Compute-bound stretch (VALU driven down hard) | ~2.2–4 ms | ~13–23 | ~8–15× |
| Absolute matrix ceiling | 1.94 ms | 26.3 | 16.7× |

**Aim for ~2–3× (≈ 11–16 ms)** from the precision switch plus the LDS/VALU it
unlocks; the ~2.2 ms padded matrix floor is the ultimate ceiling, reachable only
by also attacking coordinate-arithmetic VALU. Memory never binds.

## Next step

Wire `iu8`/`iu4` WMMA atoms into `arch_specs.json` (gfx1151 + gfx11-generic) and
`Gfx11RdnaBackend.emit_wmma` (intrinsic map + i1 sign/clamp operands + i32
accumulator), add int dtype/fragment handling in IR/lowering, then build the
native int pipeline and measure against this estimate.

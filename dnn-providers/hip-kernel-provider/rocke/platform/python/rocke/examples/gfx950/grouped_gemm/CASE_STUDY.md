# Case Study: Grouped GEMM on gfx950 (CDNA4), bf16

Hand-scheduled grouped (per-expert / batched) GEMM built with the rocKE
`IRBuilder` and lowered to HIP C++ via `hipcc --genco`. Target: AMD Instinct
gfx950 (CDNA4), bf16 inputs / f32 accumulate. Kernel: `grouped_gemm_hip.py`.

**Results** (benchmark shape below, verified vs `torch.bmm`, `max_abs ≈ 0.0156`):

| Layout | How | Median | Peak |
|---|---|---|---|
| **NT** (RCR, B `[N,K]`) — default | (no env) | **~808 TF** | **~815 TF** |
| **NN** (natural weights, B `[K,N]`) | `BRRR=1` | **~728 TF** | **~746 TF** |

For reference on the same machine:
- `torch.bmm` (rocBLAS, natural NN weights): **~780 TF** ceiling.
- A hand-tuned reference assembly kernel: **~849 TF**.

The NT path clears the rocBLAS/torch ceiling and lands at ~95% of the hand-tuned
reference. The NN path is lower because gfx950 has **no `b128` transpose-read**
intrinsic, so reading the natural-`[K,N]` B operand needs 2× `b64`
`ds_read_tr16` per fragment (vs one wide read in the NT/preshuffled layout).

---

## Benchmark shape

| | value |
|---|---|
| total rows `M` | 524288 |
| experts `E` | 64 (per-expert `m = 8192`) |
| `N` | 1024 |
| `K` | 512 |
| dtype | bf16 in, f32 acc |
| FLOPs | `2·M·N·K` |

Output tile `256×256×64`, `2×4` warp grid (BS=512, 8 wave64 waves/block),
`16×16×32` bf16 MFMA atom.

---

## The levers (all default-on in `_main`)

Running the kernel with **no environment variables** selects the best
configuration. Each lever is individually gateable for ablation.

| Lever | Env | Default | What it does | Measured |
|---|---|---|---|---|
| Chiplet super-tile grid swizzle | `CHIP` | on | Remaps block→tile assignment for XCD/L2 locality | **+40 TF** |
| `st_16x32` LDS swizzle | `SWZ` | on | Bank-conflict-free LDS layout for operand reads | **+54 TF** |
| Inline-asm `b128` `ds_read` | `ASM` | on | Wide 128-bit LDS operand reads | on |
| Cooperative direct-to-LDS load | `DTL` | on | `buffer_load_dwordx4 ... lds` (global→LDS, no ds_write) | on |
| Double-buffered tiles | `DB` | on | Overlap next K-tile load with current compute | on |
| Deep operand pipeline | `DEEPPIPE` | on | Issue next chunk's `ds_read`s under current MFMAs | **+4 TF** |
| `s_setprio` compute priority | `PRIO` | on | Raise wave priority around the MFMA region | on |
| **Epilogue-store interleave** | `EPIFUSE` | on | Store each accumulator right after its final MFMA so store-issue + address VALU overlap the last K-tile's MFMAs instead of running exposed | **+10–20 TF med, +13 TF peak** |

Gated experiments that did **not** help (kept off, documented for the record):

| Lever | Env | Result |
|---|---|---|
| Per-burst priority ping-pong | `BURSTPRIO` | Neutral — our cooperative DTL synchronizes all waves into the *same* phase, so there is no opposite-phase sibling wave to ping-pong with (see "Gap to the reference"). |
| Per-MFMA `s_setprio` + `sched_barrier` | `PERMMASCHED` | Slower / correctness-fragile. |
| CK `sched_group_barrier` hints | `CKSCHED` | Hurt RCR correctness. |
| `32×32×8` MFMA atom | `MFMA32` | ~346 TF — stalls without hand-tuned per-MFMA scheduling. |
| cshuffle store-coalescing epilogue | `CSHUF` | No win. |

The natural-NN path (`BRRR=1`, ~728 TF) is a supported, correct layout — see
the results table above — not a failed experiment; it is simply bounded by the
missing `b128` transpose-read.

---

## Why this is near the ceiling — resource analysis

Measured from the compiled code object:

```
vgpr=242  sgpr=41  spill=0  lds=131072 B (128 KiB)
blocks/CU: by_vgpr=1  by_lds=1  ->  occupancy = 1 block/CU (limiter: LDS, VGPR co-limits)
```

Occupancy is pinned at **1 block/CU by both LDS and VGPR simultaneously**:

- **LDS:** double-buffered A+B = `2·(256+256)·64·2 B = 128 KiB`. Two blocks would
  need 256 KiB > the **160 KiB/CU** limit.
- **VGPR:** 242/wave ⇒ 2 waves/SIMD ⇒ 8 waves = exactly one BS=512 block.
  **128 of the 242 VGPRs are the output-tile accumulators** (`32 accs × 4 f32`),
  which are irreducible without shrinking the tile.

### Levers that *cannot* help on gfx950

- **AGPR accumulators:** gfx950 has a **unified vector register file** — no
  separate AGPR pool. VGPRs hold both accumulators and operands; there is
  nowhere to offload to. Forcing a VGPR cap just spills to scratch.
- **More waves in flight (bigger BS):** covering the same tile with 16 waves
  drops VGPR to 128 (4 waves/SIMD) but **halves each warp's MFMA macro-tile**,
  cutting operand reuse → more `ds_read` traffic per MFMA. Measured **slower
  (~676 TF)**. We are not latency-starved at 2 waves/SIMD.
- **Bigger tile (512×512):** infeasible — LDS would be 256 KiB (compile error:
  `local memory exceeds limit 163840`), and the register-resident accumulator
  (`512×512` f32) cannot fit at any block size that also satisfies LDS
  (best case 942 spills, occupancy 0). `256×256` is the feasible ceiling for a
  register-resident square tile.

---

## Gap to the hand-tuned reference (~5%): warp-phase scheduling

Disassembly comparison (per-MFMA normalized):

| metric | this kernel | hand-tuned ref |
|---|---|---|
| `ds_read`/MFMA | **0.375** | **0.375** |
| MFMA atom | `16×16×32 bf16` | `16×16×32 bf16` |
| VGPR | 242 | 210 |
| occupancy | 2 waves/SIMD | 2 waves/SIMD |
| wave phasing | all-synchronized (1 barrier/K-tile) | desync'd producer/consumer (2 barriers/stage) |

Operand reuse / read efficiency is **already optimal and identical to the
reference** (`(mm+nn)/(mm·nn) = 12/32 = 0.375`). The reference's remaining edge
is a **warp-specialized, phase-desynchronized schedule**: it toggles
`s_setprio` per MFMA burst so that — with 2 waves/SIMD held in *opposite*
phases — one wave's high-priority MFMA burst overlaps the other wave's
low-priority `ds_read`+`buffer_load`+barrier section. Our single-barrier
cooperative DTL load puts all waves in the *same* phase, so the same priority
toggle (`BURSTPRIO`) is a no-op. Closing the gap would require restructuring
the load path into load- vs compute-warp groups (warp specialization) — a
larger change with real correctness risk.

---

## How to run

From the `rocKE` root (`.../dnn-providers/hip-kernel-provider/rocKE`), with a
ROCm 7.x toolchain (`hipcc` on `PATH`) and a Python env that has `torch`
(ROCm build):

```bash
# NT / RCR layout (default) -> ~808 TF median, ~815 TF peak
PYTHONPATH="$PWD/Python" python \
  Python/rocke/examples/gfx950/grouped_gemm/grouped_gemm_hip.py

# NN / natural-weights layout -> ~728 TF median, ~746 TF peak
BRRR=1 PYTHONPATH="$PWD/Python" python \
  Python/rocke/examples/gfx950/grouped_gemm/grouped_gemm_hip.py
```

Expected output (NT):

```
[ggemm] tile=256x256x64 warps=2x4 BS=512 dtl=True prio=True swz=True ...
[ggemm] grid=(4, 32, 64) max_abs=0.0156  PASS
[ggemm] med=~808 TF  peak=~815 TF
```

All infra required to reproduce both layouts (inline-asm `ds_read`,
`ds_read_tr16` transpose-read, `async_buffer_load_lds`, chiplet grid swizzle,
the MFMA lane helpers) already ships in the rocKE tree; this example needs no
additional core changes.

### Ablation

Disable any lever to measure its contribution, e.g.:

```bash
EPIFUSE=0 PYTHONPATH="$PWD/Python" python .../grouped_gemm/grouped_gemm_hip.py
SWZ=0     PYTHONPATH="$PWD/Python" python .../grouped_gemm/grouped_gemm_hip.py
```

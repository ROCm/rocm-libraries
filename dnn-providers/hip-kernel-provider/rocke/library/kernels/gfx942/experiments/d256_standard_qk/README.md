# D256 gfx942 attention — case study & handoff (AICK-1495)

**Ticket:** AICK-1495 (S7) — Head-dim 256 flash prefill (bf16), gfx942 + gfx950.
**Workload:** Qwen3-Next-80B-A3B `gated_attn` prefill — **D256** (qk_rope 64),
**bf16**, **GQA 16/2**, causal, Sq 4096/8192.
**Arch:** gfx942 (CDNA3 / MI300X). Built/run via `kreb` on MARKHAM/AUSTIN.

This directory is a **handoff**: it contains a complete, from-scratch **standard-QK**
attention kernel (research scripts) plus the full optimization case study, so the
next engineer can continue toward beating AOTriton without re-deriving anything.

---

## TL;DR — where we are

Two independent kernels exist for this workload on this branch:

| kernel | where | Sq4096 | Sq8192 | notes |
|---|---|---|---|---|
| **K-stream** (transposed-QK) | `library/kernels/gfx942/attention_tiled_2d.py` (commit `5944b871`) | — | **81.6 TF/s** | the *shipped/banked* deliverable (+37% over prior). |
| **standard-QK** (this dir) | `04_realworkload_bf16_gqa_vs_aotriton.py` | **80.9 TF/s** | **96.2 TF/s** | from-scratch rewrite; matches/exceeds K-stream. |
| **AOTriton** (reference) | torch SDPA flash backend on ROCm | 151.4 | 174.8 | the target to beat. |

**Standard-QK is at ~0.53× AOTriton** (measured, wall-time, identical shapes).
It **matches/exceeds the K-stream** with a cleaner transpose-free architecture.
All results GPU-verified correct (bf16 `max_abs` ~3.5e-3 vs fp32 reference).

The 2× gap to AOTriton is **real, measured, and explained** (see §Profiling and
§Next steps). This is a solved-architecture / open-performance problem.

---

## Two architectures

### K-stream (transposed-QK) — the shipped kernel
Computes `S^T = K @ Q^T`. Historically the gfx942 path; the K-stream commit added
**direct HBM→register K reads** (skip the LDS round-trip) for +37% → 81.6 TF/s.
Still pays a `V_lds` transpose + `O^T` epilogue transpose (the transposed-QK tax),
and is VALU-bound. Lives in `attention_tiled_2d.py`, gated by
`use_transposed_qk_32x32` + `use_k_hbm_direct`.

### Standard-QK (this dir) — the rewrite
Computes `S = Q @ K^T` **natural**. The key lane-math result (32×32×8 MFMA,
CDNA3): **both Q and K load row-major contiguous (coalesced, no transpose)**; V is
the wave-coalesced B-operand (**no `V_lds`, no `V^T`**); the epilogue is natural
(**no `O^T`**). Only the P-reshape (softmax-output C-layout → PV A-operand layout)
needs LDS. This is AOTriton's orientation. Removing the transpose VALU (~12%) got
us to parity with K-stream — but standard-QK is **still VALU-bound on
softmax/rescale/exp**, which is why it doesn't blow past.

---

## The optimization journey (all measured on MI300X, causal)

| step | TF/s | keep | lesson |
|---|---|---|---|
| single-head PoC | 3.67 | – | **grid-starved** (64 blocks / 304 CUs = 1 wave/CU) |
| + head dim in grid | 55.8 | ✓ | **the single-head numbers were grid-starvation, not the kernel** (+14×) |
| + `S_lds` pad `[32,33]` | 57.7 | ✓ | LDS bank conflict 29.5→0/MFMA (stride-32 → same-bank) |
| + softmax guard `lane<32` | 73.2 | ✓ | drop the redundant 2× softmax (both lane-groups recompute) |
| + bf16 (`MfmaAtom.bf16_32x32x8`) | 71.9 | ✓ | real dtype; ≈ same rate as f16 |
| + GQA 16/2 | 80.9 / 96.2 | ✓ | K/V shared across 8 q-heads → heavy L2 reuse |

**Negative results (do NOT re-try without a new idea — all measured losses):**

| tried | result | why it lost | file |
|---|---|---|---|
| `iglp_opt(1)` auto-scheduling | −20% | reshuffles poorly (K-stream's structure differs) | `stdqk_attn_iglp.py`* |
| mask-once at raw-S write | −19% | moves VALU off the guarded path onto the all-lane crit path | `stdqk_attn_mask1.py`* |
| cross-lane register softmax | −30% | butterfly all-reduce = 160 `ds_swizzle`/iter > vectorized LDS; +VGPR | `neg_cross_lane_softmax.py` |
| Q-in-LDS (load Q once) | −8% | +16 KB LDS hurts occupancy; L2 already absorbs Q reuse | `neg_q_in_lds.py` |

<sub>* the iglp/mask1 scripts are in `/scratch/users/avirgoel/d942opt/` (not copied here); the two most instructive negatives are included.</sub>

---

## Profiling (rocprofv3, on the best kernel)

- **VALU:MFMA = 11.8:1** — VALU-bound (same *class* as the transposed K-stream).
  MFMA units are idle ~92%; softmax/rescale/exp dominate. This VALU is *largely
  fundamental* to D256 online-softmax attention — **AOTriton pays it too**.
- **LDS bank conflict = 0** (after the `[32,33]` pad; was 29.5/MFMA).
- The kernel pays **two LDS round-trips** (S and P through `S_lds`) that AOTriton
  keeps **register-resident**. This + AOTriton's `num_stages` software pipeline +
  persistent grid is the 2× gap.

Reproduce the profile:
```
rocprofv3 -i pmc.txt --output-format csv -d ./rp -- python3 04_realworkload_bf16_gqa_vs_aotriton.py 4096 16
# pmc.txt: pmc: SQ_INSTS_VALU SQ_INSTS_MFMA SQ_LDS_BANK_CONFLICT SQ_BUSY_CU_CYCLES GRBM_GUI_ACTIVE SQ_WAVES
```

---

## How to build & run (kreb, gfx942)

These are **standalone research scripts** that JIT the kernel through the rocKE
DSL and validate/bench against torch. They run inside the kreb container against a
rocke checkout, with `PYTHONPATH=platform/python:library`.

```bash
# from a rocke checkout root (platform/ + library/ present), on a gfx942 node:
IMG=/cluster/images/<you>/kreb-7.13-latest-gfx942.sqsh
srun --jobid=$HELD --overlap --container-image=$IMG \
  --container-mounts=/scratch:/scratch --container-workdir=$PWD \
  bash -lc "source /usr/local/bin/kreb-activate && export HOME=/scratch/users/\$USER/chome && \
            export PYTHONPATH=platform/python:library && \
            python3 -u <path>/04_realworkload_bf16_gqa_vs_aotriton.py 8192 16"
# args: <Sq> <num_q_heads>.  Prints correctness (vs torch causal SDPA) + TF/s + the AOTriton ratio.
```
(See the `kreb` / `alola-run` skills for node acquisition. The scripts import
`kernels.common.attention_unified` only to pin `_RESOLVED_ATTENTION_ARCH`.)

---

## File guide

| file | what it is |
|---|---|
| `01_poc_single_tile.py` | minimal end-to-end proof: 1 q-tile × 1 kv-tile, D256, f16, non-causal. The clearest place to learn the QK→softmax→P-reshape→PV→epilogue flow. |
| `02_runtime_loop_scalable.py` | runtime `scf_for` kv-loop (constant kernel size, scales to any Sq; the Python-unrolled version explodes compile at Sq≥2048). acc in LDS. |
| `03_best_f16_grid_pad_guard.py` | the best **f16** kernel: multi-head grid + `S_lds` pad + softmax guard + register-carried acc (`scf_for_iter`). 73.2 TF/s. |
| `04_realworkload_bf16_gqa_vs_aotriton.py` | **the main deliverable**: bf16 + GQA 16/2 + causal + `b.fma` rescale + built-in AOTriton comparison. Start here. |
| `neg_cross_lane_softmax.py` | negative result: butterfly (`ds_swizzle_xor`) register softmax. Correct but −30%. |
| `neg_q_in_lds.py` | negative result: Q loaded once to LDS. Correct but −8%. |
| `neg_block_n_64.py` | negative result: `BLOCK_N=64`. Correct but −23% (VALU↓ yet slower → latency-bound; also `BLOCK_N>BLOCK_M` causal waste). |
| `neg_denominator_via_mfma.py` | negative result: row-sum via `P@ones` MFMA. Correct but −20% (extra MFMA + l-update LDS > the sum-VALU saved). |
| `LOGBOOK_full.md` | the complete chronological engineering log (every experiment, exact numbers). |

---

## Post-handoff optimization attempts (MEASURED — read before you optimize)

A follow-up pass ran a 3-step plan (`b.fma`, `BLOCK_N=64`, denominator-via-MFMA)
with rocprofv3 at each step. Result (bf16+GQA, SQ=4096, MI300X):

| step | change | VALU | MFMA | verdict |
|---|---|---|---|---|
| #3 | `b.fma` on the acc-rescale | 1.08e8 → 9.89e7 (−8.5%) | 8.45e6 | **WIN +2–3% — kept** (rescale was only partly fused; now folded into `04`) |
| #1 | `BLOCK_N=64` | 9.33e7 (−6%) | 8.52e6 | **LOSS −23%** (`neg_block_n_64.py`) |
| #2 | denominator = `P@ones` on idle MFMA | 9.75e7 | 8.98e6 (+6%) | **LOSS −20%** (`neg_denominator_via_mfma.py`) |

### ⚠ The load-bearing finding: this kernel is **latency/occupancy-bound, NOT VALU-count-bound**
#1 *reduced* VALU by 6% and got **23% slower**; #2 cut the sum-VALU and got **20%
slower**. Cutting VALU instructions does not speed this kernel up. (Also: `BLOCK_N`
must be **≤ `BLOCK_M`** — with `BLOCK_M=32`, `BLOCK_N=64` coarsens the causal
triangle → more masked-but-computed waste.) **Do not spend more effort on VALU
reduction.** The only VALU win that stuck was free (`b.fma`).

## Next steps to close the 2× AOTriton gap (reprioritized by the finding above)

The gap is **latency hiding**, not VALU, not bank conflicts (padded), not occupancy-
at-the-grid-level (grid fills the GPU), not the transpose (removed). Pursue, in order:

1. **`num_stages` software pipeline (async K/V prefetch).** Issue tile N+1's K/V
   `global_load`/`buffer_load` while computing tile N, so memory latency is hidden
   behind compute. This is AOTriton's primary lever and directly targets the
   measured latency bound. Highest expected payoff.
2. **Raise per-CU occupancy / waves.** The kernel carries **128 acc VGPRs** (register
   acc) → likely ~1–2 waves/CU, too few to hide latency across the 3 syncs/iter.
   Try: `BLOCK_M=64` (2 M-atoms, 128 threads → 2 waves/CTA, *and* `BLOCK_N` can then
   grow to 64 safely), or trim the acc-VGPR footprint. Measure `SQ_WAVES` / occupancy.
3. **Fewer sync stalls.** Double-buffer `S_lds` so tile N+1's QK write doesn't wait
   on tile N's PV read — removes one of the 3 `sync()`/iter.
4. **Dispatcher integration** (deferred per management — do last). Wire into
   `library/dispatch/attention.py` + a `_d256_gfx942_stdqk` gate, mirror in the C++
   selector, add a byte-identity/parity case. Only after it beats K-stream in prod.

**Register-resident softmax** (removing the `S`/`P` LDS round-trips) is now *lower*
priority: LDS bank conflicts are 0 and the kernel isn't LDS-throughput-bound, so the
round-trip isn't the wall — and the butterfly version already lost
(`neg_cross_lane_softmax.py`, −30%). Only revisit if occupancy work stalls.

---

## rocKE DSL gotchas (learned the hard way)

- `smem_load_vN(smem, *indices, dtype=, n=)` — indices are **positional varargs**;
  `smem_store_vN(smem, indices_list, value, n)` — indices are a **list**. Mixing →
  `'list' object has no attribute 'name'` at IR print.
- Float max = `b.fmax`, **not** `b.cmp_gt` (integer cmp → `icmp requires integer
  operands`).
- Each `mfma_k_loop` call in one kernel needs a **unique** `iv_name`/`acc_name`
  (defaults collide → `multiple definition of local value named 'kt'`).
- **bf16 32×32×8** on gfx942: use `MfmaAtom.bf16_32x32x8()` directly.
  `mfma_atom_for_dtype("bf16", …)` wrongly picks the **gfx950-only 32×32×16** atom.
- `S_lds` **must** be padded (`[32, 33]`) — the C-layout write is fixed-column,
  stride-32 → 32-way bank conflict without the pad.
- `scf_for` is a context manager (all state in LDS, no carried values);
  `scf_for_iter` carries values via `iter_args` + `b.scf_yield(*vals)` + `.results`.

---

## Provenance

Full analysis, per-step numbers, and the design rationale are in the vault:
`wiki/SDPA/d256-gfx942-standard-qk-validated.md`,
`wiki/SDPA/d256-gfx950-standard-qk-rewrite-design.md`,
`wiki/SDPA/d256-gfx942-optimization-findings.md`, and `LOGBOOK_full.md` here.
All numbers on MI300X (kreb, gfx942), bf16 vs fp32 reference, causal.

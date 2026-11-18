# gfx942 bf16 D128 causal prefill: QK lookahead and the `xcd_local` decode

This change adds an opt-in QK-lookahead tile body (`qk_lookahead`) to the gfx942
dense prefill kernel, plus `xcd_local`, a gfx942-only persistent decode. Both are
off by default. No shipped default, candidate or existing golden changes. The
body builds on the K LDS double buffering and V register prefetch levers from the
[CFVST / KDB / VPF case study](gfx942_dense_cfvst_kdb_vprefetch_case_study.md),
and needs both of them on.

Measured scope: gfx942 (304 CUs), BF16, D128, causal prefill, Hq 32, Hkv 8 and
16, B 1 and 16, S 1024 to 16384. The `qk_lookahead` pin is compared with the
`gfx942_dense` candidate's default spec and with AITER's asm FMHA v3 forward
(two rounding modes), with ratios only.

## Problem

With K double buffering and V prefetch on, the tile loop still runs its two big
phases one after the other: QK MFMAs, then the softmax VALU work, then PV MFMAs.
While softmax runs, the MFMA pipe has nothing to issue. While QK and PV run, the
VALU has little to do. The goal is to keep both units busy within one wave's own
instruction stream, by giving each MFMA independent VALU work to overlap with.

## Optimization 1: QK lookahead (`qk_lookahead=True`)

Software-pipeline the tile loop one tile deep. Write `S(j)` for tile `j`'s scaled
scores (the `block_m x block_n` QK result after the softmax scale).

- Pipeline step `j` enters with `S(j)` already computed. It issues `QK(j+1)`
  while it runs tile `j`'s softmax head: row max, the rescale factor `alpha`, the
  O rescale, and the first P packs.
- `PV(j)` then carries the remaining P packs, the scale of `S(j+1)` and the
  `K(j+2)` DMA.
- Each MFMA is followed by one fenced chunk of VALU/LDS work. Chunk sizes come
  from an issue-cycle estimate, so MFMAs are not issued back to back.
- Two K slots and two V^T slots leave one LDS barrier per tile.
- K rows are XOR-swizzled in LDS with no row pad and read 16 B at a time. Q is
  transposed into the MFMA layout through LDS. O is stored through LDS as whole
  128 B lines, so `o_store_width` has no effect and is rejected.

Scope, enforced by `supports_attention_dense` (`NOT_YET_IMPLEMENTED:` reasons):
bf16, D128, top-left causal, persistent grid, `persist_decode="xcd_local"`,
`block_m=256`, `block_n=64`, `lds_num_buffers=2`, `v_reg_prefetch=True`,
`causal_diag_split=True`, `lds_row_pad=0`, and the CFVST V path with its swizzle.
It refuses windows, sinks, LSE, paging, KV segments, iglp and the PV
priority/fence knobs. `use_exp2_fast` and `pv_loop_order` stay free and are
covered by the numeric tests.

## Optimization 2: the `xcd_local` persistent decode

`xcd_local` maps work items to the 304 persistent CTAs per XCD. It needs
`chiplet_num_xcds=8`, `num_persistent=304`, Hq 32 and Hkv 8 or 16.

- XCD `x = wi % 8` (where `wi` is the work-item index) owns Q heads `4x .. 4x+3`,
  which is one kv head at Hkv 8 or two at Hkv 16.
- Each XCD walks its own stream. Per batch, the stream ranks the XCD's
  `NQB * 4` items (`NQB` = query blocks per head) longest query block first,
  rotating over the 4 heads. The ranks are dealt to the XCD's 38 CTAs as a snake,
  so per-CTA causal cost stays balanced.

`xcd_local` is legal without `qk_lookahead`. `qk_lookahead` runs only on it. It
is not in the shared `PERSIST_DECODES`, so the shared spec, gfx950 and the C++
mirror never accept it; the gfx942 spec adds it through
`Gfx942AttentionDenseSpec.persist_decode_registry()`.

## How to reach it

There is no candidate. Pin the seven knobs on `spec_id="gfx942_dense"`:

```python
from dispatch.attention import tuning_spec_with_knobs

QKLA_KNOBS = {
    "qk_lookahead": True,
    "block_m": 256,
    "lds_num_buffers": 2,
    "v_reg_prefetch": True,
    "causal_diag_split": True,
    "lds_row_pad": 0,
    "persist_decode": "xcd_local",
}
tuned = tuning_spec_with_knobs(request, "gfx942_dense", QKLA_KNOBS)
```

The full tuning space reaches the same point. Adding the defaulted
`qk_lookahead` field rekeys every `gfx942_dense` `tuning_id`; see
`library/dispatch/AGENTS.md`.

## Static resources

Static counts are from the HSACOs timed in the benchmark (`static_resources.csv`,
LLVM23, 512 threads/WG, persistent 304). Instruction columns are static counts in
the code object, not executed counts. The `qk_lookahead` binary is the same on
every shape. The baseline row is B=1, S=4096, Hkv=8; the baseline at S≤2048 uses
`block_m=128` (228 VGPR, 16 `ds_write` / `buffer_load`).

| Arm | VGPR | AGPR | SGPR | VGPR / SGPR spills | Scratch | LDS bytes | waves/SIMD | `v_mfma` | `ds_read` | `ds_write` | `buffer_load` | `s_barrier` | `s_waitcnt` | VALU | SALU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gfx942_dense default | 220 | 0 | 45 | 0 / 0 | 0 B | 33792 | 2 | 64 | 32 | 8 | 8 | 3 | 31 | 1125 | 437 |
| `qk_lookahead` | 250 | 0 | 96 | 0 / 0 | 0 B | 65536 | 2 | 192 | 120 | 48 | 24 | 6 | 105 | 1947 | 726 |

Both arms are spill-free with 0 B scratch on all twelve shapes. `qk_lookahead`
uses the whole 64 KiB LDS. The higher static MFMA and barrier counts come from the
prologue, the pipelined loop, the epilogue and the diagonal-split peel, each
emitted separately. The main loop keeps one LDS barrier per tile, which a test
asserts.

## Results

All workloads are BF16, Hq=32, D=128, causal, Sq = Skv = S, on gfx942 (304 CUs).

### Arms

| Arm | How it is built | Persistent | exp2_fast | `persist_decode` |
|---|---|---|---|---|
| gfx942_dense default | `attention_tuning_spec(req, "gfx942_dense", knobs={})`: the candidate's tuned default spec (spec_id pinned, no knobs) | True | `None` → policy, resolves on (persistent grid) | `auto` |
| `qk_lookahead` | `attention_tuning_spec(req, "gfx942_dense", knobs=QKLA_KNOBS)` | True | `None` → policy, resolves on | `xcd_local` |
| AITER RTNA | `flash_attn_func(q, k, v, causal=True)`: AITER's default kernel selection with its default bf16 output rounding, RTNA (`how_v3_bf16_cvt` left at its default) | — | — | — |
| AITER RTNE | the same with `how_v3_bf16_cvt=0`: RTNE output rounding, the rounding the rocKE kernels use | — | — | — |

The gfx942_dense default is not what `algorithm="auto"` picks for these shapes
(auto dispatch picks `unified_2d`); it is the opt-in dense candidate at its own
defaults. It uses `block_m=128` at S≤2048 and `block_m=256` above, with
`waves_per_eu=2`, `lds_row_pad=8`, one K buffer and no V prefetch. Its resolved
specs, kernel names and replay records are in `resolved_specs.json`.

AITER is pinned at commit `97452bc933e571facd206ea01348852ef0c96e97`.

### Speedup of `qk_lookahead`

Each cell is the median over three runs of the per-run ratio
`median(baseline time) / median(candidate time)`, with the min–max of the three
runs. The percentage is `(ratio − 1) × 100`, the throughput gain of the
candidate; the percentage columns come first, then the same ratios as × with
the min–max range. Above 1.000× (a positive percentage) means the candidate is faster.

| dtype | B | S | Hkv | qk_lookahead vs gfx942_dense default (%) | qk_lookahead vs AITER RTNA (%) | qk_lookahead vs AITER RTNE (%) | gfx942_dense default vs AITER RTNA (%) | qk_lookahead vs gfx942_dense default (×, range) | qk_lookahead vs AITER RTNA (×, range) | qk_lookahead vs AITER RTNE (×, range) | gfx942_dense default vs AITER RTNA (×, range) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| bf16 | 1 | 1024 | 8 | +33.6% | +37.2% | +43.6% | +3.3% | 1.336× (1.328–1.342) | 1.372× (1.358–1.406) | 1.436× (1.423–1.442) | 1.033× (1.017–1.047) |
| bf16 | 1 | 1024 | 16 | +35.7% | +40.4% | +45.8% | +3.5% | 1.357× (1.348–1.368) | 1.404× (1.395–1.422) | 1.458× (1.434–1.462) | 1.035× (1.035–1.039) |
| bf16 | 1 | 2048 | 8 | +28.6% | +10.4% | +16.1% | −14.2% | 1.286× (1.285–1.287) | 1.104× (1.103–1.104) | 1.161× (1.148–1.169) | 0.858× (0.858–0.859) |
| bf16 | 1 | 2048 | 16 | +33.7% | +10.4% | +15.5% | −18.1% | 1.337× (1.335–1.348) | 1.104× (1.090–1.108) | 1.155× (1.145–1.161) | 0.819× (0.815–0.830) |
| bf16 | 1 | 4096 | 8 | +36.8% | +5.9% | +10.1% | −22.6% | 1.368× (1.355–1.376) | 1.059× (1.054–1.060) | 1.101× (1.090–1.102) | 0.774× (0.771–0.777) |
| bf16 | 1 | 4096 | 16 | +36.7% | +4.4% | +8.9% | −23.5% | 1.367× (1.365–1.372) | 1.044× (1.040–1.051) | 1.089× (1.085–1.091) | 0.765× (0.761–0.766) |
| bf16 | 1 | 8192 | 8 | +37.1% | +6.5% | +10.4% | −22.3% | 1.371× (1.369–1.371) | 1.065× (1.062–1.065) | 1.104× (1.104–1.104) | 0.777× (0.775–0.778) |
| bf16 | 1 | 8192 | 16 | +38.0% | +5.6% | +9.2% | −23.6% | 1.380× (1.377–1.386) | 1.056× (1.052–1.056) | 1.092× (1.092–1.096) | 0.764× (0.762–0.765) |
| bf16 | 1 | 16384 | 8 | +46.3% | +12.9% | +18.4% | −22.8% | 1.463× (1.461–1.464) | 1.129× (1.128–1.130) | 1.184× (1.184–1.185) | 0.772× (0.771–0.772) |
| bf16 | 16 | 4096 | 8 | +30.5% | +1.4% | +5.4% | −22.3% | 1.305× (1.302–1.306) | 1.014× (1.012–1.014) | 1.054× (1.049–1.055) | 0.777× (0.777–0.777) |
| bf16 | 16 | 4096 | 16 | +39.9% | +0.6% | +4.4% | −28.1% | 1.399× (1.397–1.400) | 1.006× (1.005–1.006) | 1.044× (1.044–1.046) | 0.719× (0.719–0.719) |
| bf16 | 16 | 8192 | 8 | +27.0% | **−2.3%** | +1.0% | −23.0% | 1.270× (1.267–1.271) | **0.977×** (0.977–0.978) | 1.010× (1.010–1.010) | 0.770× (0.769–0.771) |
| **geomean** | | | | **+35.2%** | **+10.4%** | **+15.0%** | **−18.4%** | **1.352×** | **1.104×** | **1.150×** | **0.816×** |

At these measured configurations only:

- `qk_lookahead` is faster than the gfx942_dense default on all 12 shapes.
- It is faster than AITER RTNE, the equal-rounding comparison, on all 12 shapes.
- It is faster than AITER RTNA on 11 of 12 shapes. It is slower at B=16, S=8192,
  Hkv=8 (0.977×, −2.3%). AITER RTNA rounds its bf16 output with RTNA; both rocKE
  arms and AITER RTNE use RTNE.

### Correctness

Every arm, at every shape and in every run, matched an FP32 causal GQA reference
computed in NumPy from the same bf16 inputs, at `atol=rtol=1e-2`. Each arm was
checked once before timing (rocKE outputs filled with 0xFF first) and after each
of the five timed blocks: 864 of 864 checks passed over the three runs.

The GPU tests in `tests/test_attention_dense_gfx942_numeric.py` also check
`qk_lookahead` against FP32 SDPA at `1e-2`: cross lengths (Sq ≠ Skv), multiple
grid-stride rounds, non-default softmax scales (`4e-2`), the free knobs
(`use_exp2_fast=False`, `pv_loop_order="k_major"`) and runtime-shape binary reuse.

### Caveats

- One gfx942 part, 304 CUs. `xcd_local` needs 304 persistent CTAs over 8 XCDs,
  so `qk_lookahead` does not apply to other CU counts.
- Hq 32 only: `xcd_local` needs Hq 32 and Hkv 8 or 16.
- AITER RTNA and the rocKE arms round the bf16 output differently (RTNA vs
  RTNE); AITER RTNE is the equal-work comparison.
- The gfx942_dense default is an opt-in candidate, not the `algorithm="auto"`
  choice for these shapes.
- Timing has no host synchronization between the 50 launches in a block, so it
  measures back-to-back kernel throughput, not single-launch latency.

## Method

Harness:
[`benchmarks/gfx942/attention/prefill/bench_dense_qkla_vs_aiter.py`](../../../../benchmarks/gfx942/attention/prefill/bench_dense_qkla_vs_aiter.py).

1. For each rocKE arm, the harness resolves the spec through dispatch and records
   the replay record (`spec_id`, `tuning_id`, `config_key`, knobs, `spec_hash`).
   It replays the record before timing and aborts on a mismatch. It compiles one
   HSACO per arm and records its static resources.
2. All arms share one input set: N(0, 1) rounded to bf16, scaled by 0.2, and
   rounded again.
3. All four arms run in one process on one dedicated HIP stream. Each arm gets
   10 round-robin warmup launches. Then come 5 rounds, with the arm order
   shuffled per round.
4. Each arm's block in a round is one HIP start event, 50 back-to-back launches
   with no host synchronization, and one HIP stop event. rocKE launchers run
   under `no_fence`, as `rocke.runtime.time_launches` times them. Per-launch time
   is elapsed time / 50.
5. Per arm, the harness takes the median of the per-round times. A run's ratio
   is the ratio of two medians. Three independent processes ran, with seeds 942,
   943 and 944, and were combined with `--combine`.

Provenance. Every run is at the same clean commit (the harness refuses a dirty
tree): the change's code commit, before this document was added. Only Markdown
changed after it. Each records the SHA-256 of the kernel, spec, decode, dispatch and harness
sources, and all three runs record the same hashes. LLVM flavor `llvm23`.

Reproduce:

```
python library/benchmarks/gfx942/attention/prefill/bench_dense_qkla_vs_aiter.py --seed 942 --out-dir <dir>/run1
python library/benchmarks/gfx942/attention/prefill/bench_dense_qkla_vs_aiter.py --seed 943 --out-dir <dir>/run2
python library/benchmarks/gfx942/attention/prefill/bench_dense_qkla_vs_aiter.py --seed 944 --out-dir <dir>/run3
python library/benchmarks/gfx942/attention/prefill/bench_dense_qkla_vs_aiter.py --combine <dir>/run1 <dir>/run2 <dir>/run3
```

Absolute times stay in each run's local `runs.csv`; only ratios are published.

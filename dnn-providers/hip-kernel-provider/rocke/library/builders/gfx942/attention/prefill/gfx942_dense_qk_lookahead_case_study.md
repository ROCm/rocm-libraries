# gfx942 bf16 D128 causal prefill: QK lookahead and the `xcd_local` decode

This change adds two opt-in levers to the gfx942 dense prefill kernel for bf16
D128 causal attention on the persistent grid:

- `qk_lookahead=True` turns on several changes to the tile body together. It
  double-buffers V^T as well as K, pipelines the tile loop one tile deep,
  interleaves MFMA with VALU work, swizzles K in LDS instead of padding it, and
  stages Q and O through LDS. Sections 1a–1h describe each part.
- `persist_decode="xcd_local"`, a gfx942-only persistent decode that is causal workload and L2/XCD layout aware.

Both are off by default. No shipped default, candidate or existing golden
changes. `qk_lookahead` builds on K LDS double buffering and V register prefetch
from the [CFVST / KDB / VPF case study](gfx942_dense_cfvst_kdb_vprefetch_case_study.md),
and needs both on. It runs only on `xcd_local`; `xcd_local` is legal on its own.

Measured scope: gfx942 (304 CUs), BF16, D128, causal prefill, Hq 32, Hkv 8 and
16, B 1 and 16, S 1024 to 16384. The `qk_lookahead` pin is compared with the
`gfx942_dense` candidate's default spec and with AITER's asm FMHA v3 forward
at two bf16 output rounding modes: AITER RTNA (its default `how_v3_bf16_cvt`,
round to nearest, ties away from zero) and AITER RTNE (`how_v3_bf16_cvt=0`, round
to nearest, ties to even, the rounding both rocKE arms use). The parts of
`qk_lookahead` were measured only together, so every result applies to the whole
flag, not to any one part.

## Optimization 1: QK lookahead (`qk_lookahead=True`)

| Part | What it does |
|---|---|
| 1a. Two K slots and two V^T slots, one barrier per tile | one LDS barrier per tile instead of two |
| 1b. Software pipelining one tile deep | QK of the next tile runs under the current tile's softmax |
| 1c. Fenced MFMA/VALU interleave | one chunk of independent VALU/LDS work after every MFMA |
| 1d. K XOR swizzle instead of a row pad | frees LDS for the second V^T slot; 16 B K reads |
| 1e. V^T swizzle for two slots | V(j+1) `ds_write_b32` from 4-way to 2-way bank conflicts |
| 1f. Split row sum | the cross-half row-sum add moves out of the loop |
| 1g. Q staged through LDS | row-contiguous Q loads, transposed in LDS |
| 1h. O staged through LDS | O stored as whole 128 B lines |

### 1a. Two K slots and two V^T slots, one barrier per tile

`QK(j+1)` reads K slot `(j+1) % 2`, while `K(j+2)` is written into slot `j % 2`.
`PV(j)` reads V^T slot `j % 2`, while `V(j+1)` is stored into slot `(j+1) % 2`.
Every slot's last readers are past the previous barrier, so the loop needs one
LDS barrier per tile, which publishes both new tiles. The two K slots come from
`lds_num_buffers=2`. The second V^T slot is new with this flag. Together they
fill the 64 KiB LDS.

### 1b. Software pipelining one tile deep

- Pipeline step `j` enters with `S(j)` already computed. It issues `QK(j+1)`
  while it runs tile `j`'s softmax head: row max, the rescale factor `alpha`, the
  O rescale, and the first P packs.
- `PV(j)` then carries the remaining P packs, the scale of `S(j+1)` and the
  `K(j+2)` DMA.
- The prologue computes `S(0)`. The epilogue runs the last tile's softmax and PV
  from the carried `S`.

### 1c. Fenced MFMA/VALU interleave

QK and PV are emitted as generators that yield after each MFMA. After every
MFMA, the body emits one fenced chunk of independent VALU/LDS work: max, mask,
`alpha`, O rescale, exp, P pack, the V(j+1) load and store, or a K DMA row. Chunk
sizes come from a fixed per-operation issue-cycle estimate. No two MFMAs are
emitted back to back. The estimate decides only where work goes, not what work
is done.

In QK, the two key subtiles alternate, so consecutive MFMAs never depend on each
other. Each still accumulates in order, so the scores match the non-pipelined
body exactly. K `ds_read`s are issued up to three ahead of the MFMA that uses
them (`QK_READ_AHEAD`).

### 1d. K XOR swizzle instead of a row pad

K rows are stored with no row pad (`lds_row_pad=0`). Instead, each row's 16 B
chunks are XORed with `row & 7`. The K DMA applies the swizzle to the global
offset it fetches, so each lane keeps one base address. QK then reads K 16 B at a
time (`ds_read_b128`), one read feeding two MFMA steps. Without the pad, two K
slots and two V^T slots fit in the 64 KiB LDS.

### 1e. V^T swizzle for two slots

The CFVST V^T swizzle key also XORs in `(dim >> 4) & 1`. With two V^T slots at
`block_n=64`, this takes the V(j+1) `ds_write_b32` from 4-way to 2-way bank
conflicts. PV builds one V^T base address per tile and folds the rest into XORs
and immediates.

### 1f. Split row sum

Each lane half keeps its own partial softmax row sum `l` through the loop. The
two halves are added once after the loop with one cross-half shuffle.

### 1g. Q staged through LDS

Q is loaded row-contiguous (16 rows x 64 B per load) instead of in the MFMA
layout. In the prologue it is transposed into the MFMA layout through V^T slot 1,
which is free until `V(1)` is stored in tile 0. The 16 B chunks are XORed with
`(row >> 1) & 3` so the `ds_read_b128` that reads them back hits 8 distinct
slots.

### 1h. O staged through LDS

After the loop, once all waves are past their last K/V reads, each wave writes O
into the K buffer in 8 B chunks XORed with `row & 15`. It reads O back
row-contiguous, 16 B per lane, and stores whole 128 B lines to global memory.
Because of this, `o_store_width` has no effect and is rejected.


## Optimization 2: the `xcd_local` persistent decode

A persistent decode decides which work item (one query block of one Q head in
one batch) each persistent CTA runs in each round. `xcd_local` needs 8 XCDs, 304
persistent CTAs (38 per XCD), Hq 32 and Hkv 8 or 16.

**KV heads to XCDs.** The decode relies on the hardware placing CTAs on XCDs
round-robin. It gives every CTA only work items of its own XCD. Each XCD owns 4
Q heads in every batch: at Hkv 8 those 4 heads share one kv head, and at Hkv 16
they read two. Every Q head that reads a given kv head runs on the same XCD, and
no other XCD reads that kv head.

**Batching.** Each XCD works through the batches in order. It hands out every
work item of one batch before any item of the next batch. A new batch can start
part-way through a round, on whichever CTAs are next.

**Causal order.** Under the causal mask, later query blocks attend to more keys.
Inside a batch, each XCD hands out its query blocks from the last (most keys) to
the first. The 4 Q heads of one query block go out together.

**Dealing to CTAs.** Each XCD deals this ordered list to its 38 CTAs one round at
a time. Odd rounds go in reverse order (a snake): a CTA that got one of the
longest items in one round gets one of the shortest in the next. When the
batch's items do not divide evenly by 38, some CTAs get one item more than the
others. Those CTAs come last in each round's order before the snake reverses
it, and the leftover items go out in a final, partial round. Each work item is
handed out exactly once.


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
LLVM23, persistent 304). Instruction columns are static counts in
the code object, not executed counts. The `qk_lookahead` binary is the same on
every shape. The baseline row is B=1, S=4096, Hkv=8; the baseline at S≤2048 uses
`block_m=128` (256 threads/WG, 228 VGPR, 16 `ds_write` / `buffer_load`); every
other binary in the table is 512 threads/WG.

| Arm | VGPR | AGPR | SGPR | VGPR / SGPR spills | Scratch | LDS bytes | waves/SIMD | `v_mfma` | `ds_read` | `ds_write` | `buffer_load` | `s_barrier` | `s_waitcnt` | VALU | SALU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gfx942_dense default | 220 | 0 | 45 | 0 / 0 | 0 B | 33792 | 2 | 64 | 32 | 8 | 8 | 3 | 31 | 1125 | 437 |
| `qk_lookahead` | 250 | 0 | 96 | 0 / 0 | 0 B | 65536 | 2 | 192 | 120 | 48 | 24 | 6 | 105 | 1947 | 726 |

Both arms are spill-free with 0 B scratch on all twelve shapes. `qk_lookahead`
uses the whole 64 KiB LDS: two K slots and two V^T slots (1a, 1d). Each tile loop
keeps one LDS barrier per tile, which a test asserts.

## Results

All workloads are BF16, Hq=32, D=128, causal, Sq = Skv = S, on gfx942 (304 CUs).

### Arms

| Arm | How it is built | Persistent | exp2_fast | `persist_decode` |
|---|---|---|---|---|
| gfx942_dense default | `attention_tuning_spec(req, "gfx942_dense", knobs={})`: the candidate's tuned default spec (spec_id pinned, no knobs) | True | `None` → policy, resolves on (persistent grid) | `auto` |
| `qk_lookahead` | `attention_tuning_spec(req, "gfx942_dense", knobs=QKLA_KNOBS)` | True | `None` → policy, resolves on | `xcd_local` |
| AITER RTNA | `flash_attn_func(q, k, v, causal=True)`: AITER's default kernel selection with its default bf16 output rounding, RTNA (`how_v3_bf16_cvt` left at its default) | — | — | — |
| AITER RTNE | the same with `how_v3_bf16_cvt=0`: RTNE output rounding, the rounding the rocKE kernels use | — | — | — |

AITER is pinned at commit `97452bc933e571facd206ea01348852ef0c96e97`.

### Speedup of `qk_lookahead`

Each cell is the median over three runs of the per-run ratio
`median(baseline time) / median(candidate time)`, with the min–max of the three
runs. The percentage is `(ratio − 1) × 100`, the throughput gain of the
candidate. The percentage columns come first, then the same ratios as × with
the min–max range. Above 1.000× (a positive percentage) means the candidate is
faster.

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
  Hkv=8 (0.977×, −2.3%).

### Correctness

Every arm, at every shape and in every run, matched an FP32 causal GQA reference
computed in NumPy from the same bf16 inputs, at `atol=rtol=1e-2`. Each arm was
checked once before timing (rocKE outputs filled with 0xFF first) and after each
of the five timed blocks: 864 of 864 checks passed over the three runs.

The GPU tests in `tests/test_attention_dense_gfx942_numeric.py` also check
`qk_lookahead` against FP32 SDPA at `1e-2`: cross lengths (Sq ≠ Skv), multiple
grid-stride rounds, non-default softmax scales (`4e-2`), the free knobs
(`use_exp2_fast=False`, `pv_loop_order="k_major"`) and runtime-shape binary reuse.

## Architecture and configuration gating

Both optimizations are implemented in `library/kernels/gfx942/attention_dense.py`
on `Gfx942AttentionDenseSpec`; `xcd_local` lives in
`library/kernels/common/attention_dense_decode.py` as `PersistXcdLocal`.

Architecture eligibility is enforced by the existing `arch != "gfx942"` checks in
`supports_attention_dense(spec, arch=...)` and `build_attention_dense(spec, arch=...)`.
The support predicate returns `(False, reason)` for other architectures, and the
builder raises `NotImplementedError`. `xcd_local` is registered only in
`Gfx942AttentionDenseSpec.persist_decode_registry()`, so the shared spec and other
architectures do not accept it.

The optimization-specific eligibility conditions:

- **`xcd_local` (`persist_decode="xcd_local"`):** checked when the spec is built
  (`PersistXcdLocal.check`). Requires `chiplet_num_xcds=8`, `num_persistent=304`,
  Hq 32 and Hkv 8 or 16. It is legal without `qk_lookahead`. `persist_decode="auto"`
  never resolves to it.
- **QK lookahead (`qk_lookahead=True`):** checked by `supports_attention_dense()`.
  Requires BF16 D128 top-left causal attention on the persistent grid with
  `block_m=256`, `block_n=64`, `lds_num_buffers=2`, `v_reg_prefetch=True`,
  `causal_diag_split=True`, `lds_row_pad=0`, and the CFVST V path with its V
  swizzle. It rejects sliding window, `window_wave_skip`, attention sinks, LSE
  output, paged KV, `kv_segments > 1`, `iglp`, `pv_priority` and
  `pv_sched_fence_mask`.
- **QK lookahead needs `xcd_local`:** a `qk_lookahead` spec whose resolved
  `persist_decode` is not `xcd_local` is rejected.
- **`o_store_width`:** any value other than the default 4 is rejected with
  `qk_lookahead`, because it stores O through LDS and does not read
  `o_store_width`.

The LDS allocation, including the second V^T slot under `qk_lookahead`, is
checked against the gfx942 capacity using `attention_lds_capacity_bytes(arch)` and
`_lds_bytes(spec)`. Unsupported configurations are rejected before kernel
emission.

In dispatch, `qk_lookahead` is the last axis of the `gfx942_dense` tuning space
(`library/dispatch/attention/axes.py`), and `xcd_local` is one of its
`persist_decode` values. There is no candidate for either. The default
configuration keeps `qk_lookahead=False` and `persist_decode="auto"`, so both
remain opt-in and do not change the default kernel.


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
changed after it. Each run records the SHA-256 of the kernel, spec, decode,
dispatch and harness sources, and all three runs record the same hashes. LLVM
flavor `llvm23`.

### Benchmark environment and provenance

| Component | Configuration |
| :--- | :--- |
| Hardware target | gfx942, 304 CUs |
| Benchmarked source HEAD | `4f2df197f507d4e37171262947fdc0a482dd067c` (clean working tree, all three runs) |
| Kernel source SHA-256 | `c1bbcf8ed71c5ec91623a7482b78892458b0f3028a2028db8f94890f8ab027c5` |
| Benchmark harness SHA-256 | `0b1bdde3a98a91e8e022c2d940b2f3cf8698a22c4d5a98018ae65046d6dece05` |
| ROCm stack | 10.0.0 |
| HIP | `7.15.26333` |
| Python | `3.12.3` |
| NumPy | `2.5.5` |
| Triton | `3.8.0+amd.rocm7.2.0.git7cb7b059` |
| PyTorch (AITER arms) | `2.13.0+rocm10.0.0` |
| AITER | `97452bc933e571facd206ea01348852ef0c96e97` (clean) |
| ROCKE compiler backend | Python |
| LLVM flavor | `llvm23` |

### Shape-by-shape kernel identity

`gfx942_dense` default, built by `attention_tuning_spec(req, "gfx942_dense", knobs={}).kernel_spec`:

| dtype | B | S | Hq | Hkv | D | Exact baseline kernel name | persistent | exp2_fast (resolved) | decode (resolved) | grid | block |
|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| bf16 | 1 | 1024 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_bm128_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `true` | `true` | `bt_hkv_minor` | `[304, 1, 1]` | `[256, 1, 1]` |
| bf16 | 1 | 1024 | 32 | 16 | 128 | `rocke_attention_dense_d128_bn64_bf16_bm128_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `true` | `true` | `bt_hkv_minor` | `[304, 1, 1]` | `[256, 1, 1]` |
| bf16 | 1 | 2048 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_bm128_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `true` | `true` | `bt_hkv_minor` | `[304, 1, 1]` | `[256, 1, 1]` |
| bf16 | 1 | 2048 | 32 | 16 | 128 | `rocke_attention_dense_d128_bn64_bf16_bm128_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `true` | `true` | `bt_hkv_minor` | `[304, 1, 1]` | `[256, 1, 1]` |
| bf16 | 1 | 4096 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `true` | `true` | `bt_hkv_minor` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 4096 | 32 | 16 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `true` | `true` | `bt_hkv_minor` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 8192 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `true` | `true` | `bt_hkv_minor` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 8192 | 32 | 16 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `true` | `true` | `bt_hkv_minor` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 1 | 16384 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_bthkvmin_gfx942_wpe2` | `true` | `true` | `bt_hkv_minor` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 16 | 4096 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe2` | `true` | `true` | `hkv_major` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 16 | 4096 | 32 | 16 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2` | `true` | `true` | `qb_major` | `[304, 1, 1]` | `[512, 1, 1]` |
| bf16 | 16 | 8192 | 32 | 8 | 128 | `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe2` | `true` | `true` | `hkv_major` | `[304, 1, 1]` | `[512, 1, 1]` |

`qk_lookahead`, built by `attention_tuning_spec(req, "gfx942_dense", knobs=QKLA_KNOBS).kernel_spec`,
is the same kernel at all 12 shapes:

| Exact kernel name | persistent | exp2_fast (resolved) | decode (resolved) | grid | block |
|---|---|---|---|---|---|
| `rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_xcdlb_gfx942_wpe2_kdb1_vpf1_krowpad0_dsplit_qkla` | `true` | `true` | `xcd_local` | `[304, 1, 1]` | `[512, 1, 1]` |

AITER arms, from AITER's `LoadKernel` log lines in each run's log. Each kernel
was loaded once per process, and no other AITER kernel was loaded, so every
shape ran these two kernels in all three runs:

| Arm | Call | Exact kernel name | Code object |
|---|---|---|---|
| AITER RTNA | `flash_attn_func(q, k, v, causal=True)` | `_ZN5aiter31fmha_fwd_hd128_bf16_causal_rtnaE` | `fwd_hd128_bf16_causal_rtna.co` (AITER `fmha_v3_fwd`, gfx942) |
| AITER RTNE | `flash_attn_func(q, k, v, causal=True, how_v3_bf16_cvt=0)` | `_ZN5aiter31fmha_fwd_hd128_bf16_causal_rtneE` | `fwd_hd128_bf16_causal_rtne.co` (AITER `fmha_v3_fwd`, gfx942) |

**Benchmark harness:** `library/benchmarks/gfx942/attention/prefill/bench_dense_qkla_vs_aiter.py`

**Kernel source:** `library/kernels/gfx942/attention_dense.py`

Reproduce:

Set up the environment with `setup.sh` from the
[Reproducibility section of the CFVST / KDB / VPF case study](gfx942_dense_cfvst_kdb_vprefetch_case_study.md#reproducibility),
then `source` the `rocke_env.sh` it writes. `setup.sh` does not install AITER;
install AITER separately at the commit listed under Results. From the rocke
root, run the three seeds and combine them:

```
python library/benchmarks/gfx942/attention/prefill/bench_dense_qkla_vs_aiter.py --seed 942 --out-dir <dir>/run1
python library/benchmarks/gfx942/attention/prefill/bench_dense_qkla_vs_aiter.py --seed 943 --out-dir <dir>/run2
python library/benchmarks/gfx942/attention/prefill/bench_dense_qkla_vs_aiter.py --seed 944 --out-dir <dir>/run3
python library/benchmarks/gfx942/attention/prefill/bench_dense_qkla_vs_aiter.py --combine <dir>/run1 <dir>/run2 <dir>/run3
```

# CK DSL gfx942 unified-attention — levers & benchmark harness

Torch-reference parity + benchmark harness for the **gfx942 (CDNA3 / MI300X)**
unified-attention SDPA-fwd kernels (`ck_dsl.instances.gfx942.attention_tiled_2d`
for prefill, `..._tiled_3d` for the split-KV decode path).

Like the [gfx950 harness](../../gfx950/attention/README.md), this README is a
**technical discussion of the levers** that took the gfx942 attention stack from
losing to PyTorch's flash attention to beating it on (almost) every shape in
`shapes.json`, and how each lever was found and validated.

## Headline results

MI300X / gfx942 / ROCm 7.2 / torch 2.12, via the production dispatcher
(`run_unified_attention_torch(backend="auto")`) vs PyTorch SDPA. Mean per-launch
over 50 timed iters; **CK beats Torch on all 14 `shapes.json` shapes, all
bit-accurate** (`final_shapes_check.py`). **`speedup` = torch ÷ CK (>1 means CK
is faster).** "graph" = the dispatcher auto-engages CUDA-graph replay for that
shape (see the graph heuristic lever).

| shape | dtype | CK | Torch | **speedup** | path |
|-------|-------|---:|------:|:-----------:|:----:|
| `Fp16_Prefill_GQA_S2048_D128` | fp16 | 218 us | 233 us | **1.07×** | 2d ring |
| `Fp16_Prefill_GQA_S2048_D64`  | fp16 | 126 us | 138 us | **1.09×** | 2d ring |
| `Fp16_Decode_GQA_S2048_D128`  | fp16 | 29 us | 93 us | **3.2×** | 3d graph |
| `Fp16_Decode_GQA_S2048_D64`   | fp16 | 28 us | 68 us | **2.5×** | 3d graph |
| `Fp16CausalMhaD64S512`        | fp16 | 29 us | 43 us | **1.50×** | 2d narrow graph |
| `Fp16CausalGqa8D128S512`      | fp16 | 44 us | 54 us | **1.22×** | 2d narrow graph |
| `Fp16CausalGqaD128S528`       | fp16 | 60 us | 61 us | **1.02×** | 3d graph |
| `Fp16CausalD128` / `GqaD128` (S64) | fp16 | 14-15 us | 36-38 us | **2.4-2.7×** | 2d narrow graph |
| `Bf16CausalD128` / `GqaD128` (S64) | bf16 | 15-16 us | 36-38 us | **2.2-2.5×** | 2d narrow graph |
| `Bf16CausalGqa8D128S512`      | bf16 | 50 us | 55 us | **1.10×** | 2d narrow graph |
| `Bf16CausalD64` (S64)         | bf16 | 15 us | 32 us | **2.2×** | 2d narrow graph |
| `Fp16CausalD64` (S64)         | fp16 | 18 us | 32 us | **1.8×** | 2d narrow graph |

D128 prefill widens to **1.45× at S4096** (664 vs 993 us). Every shape wins —
the smallest shapes via the graph + host-overhead fast path (below), which
collapsed a tiny call from ~38 us to ~16 us.

Relative to the baseline this work started from (transposed-x8 `cfvst`, T=64,
double-buffered K — the prior "wide4" default), the headline prefill deltas:

| shape (S2048) | original | final | vs-original | vs-torch (speedup) |
|---------------|---------:|------:|------------:|-------------------:|
| D128 prefill  | ~310 us (0.75× torch, loses) | **218 us** | **1.42× faster** | **1.07× (wins)** |
| D64 prefill   | ~158 us (0.87× torch, loses) | **126 us** | **1.25× faster** | **1.09× (wins)** |

## The levers

The fp16 prefill kernel is the transposed-x8 flash pipeline: `S^T = K·Q^T` via
the gfx942-legal `mfma_f32_32x32x8_f16` atom, `P^T` register-resident as the PV
B-operand (no `P_lds` round-trip), conflict-free V LDS. On that foundation:

| # | lever | mechanism | effect |
|---|-------|-----------|--------|
| 1 | **Sliced-K ring** (`ksring`, T=64) | stage K as a 3-slot ring of 32-head-dim slices instead of double-buffering full `[T,HD]` K; depth-3 prefetch with `s_waitcnt` partial waits, within the 64 KB LDS cap. **T=64, not T=128** — the smaller tile keeps occupancy high; the ring amortises best there. | D128 prefill ~310→218 us (**1.42×**), beats torch S2048/S4096; the dominant prefill win |
| 2 | **Ring on D64** | the ring's `HD/32` slicing divides D64 (`k_groups=2`) too; relax the D128-only gate so D64 & D128 prefill share one geometry (T=64, nw4, ring+cfvst+mask-limit). | D64 prefill **+13-17%** vs the prior bespoke nw2/single config — beats torch S2048 (**1.09×**), ~parity→win S4096 |
| 3 | **mask-limit** | collapse causal + prefix masks into one compare vs `min(causal, prefix_tail)`; hoist the per-query-row MFMA base out of the 16-register score loop (pure VALU schedule, bit-identical). | small consistent positive on the T=64 ring; default-on D64+D128 |
| 4 | **Wide 128-bit decode KV feed** | replace gfx942's 1-DWORD async `buffer_load_lds` (the b96/b128 LDS-DMA forms are gfx950-only → 4× the load instructions) with wide 8-half (16 B) global→reg→LDS loads (the in-kernel fp8 vehicle); convert the descriptor's byte offset → element index for `global_load_vN`. | **~8%** on D128 decode, neutral D64, bit-identical |
| 5 | **bf16 D128 = T=64** (scalar-fallback fix) | bf16 D128 defaulted to T=128 → double-buffered K alone is 64 KB → tiled-2D LDS gate rejected → `auto` dropped to the **scalar kernel**. Force `T=64` for *all* gfx942 D128 → fits the narrow path at nw≤2. | `Bf16CausalGqa8D128S512` **176 614→173 us (1023×)**; `Bf16CausalD128` **5543→51 us** |
| 6 | **Light narrow geometry for short context** | for `q ≤ 768` the ring's prelude (256 threads / 3-slot K / Q-direct / mask-limit / mw=32) is pure overhead on a 1-2-tile KV loop. `_enable_gfx942_small_q_narrow` turns flash off and forces 16×16×16, `nw`=1 (MHA) / 2 (GQA), `bmpw=16`, T=64. | every `q≤768` D64/D128 shape wins under graph (D64 S64 0.37×, D128 GQA S512 0.74× in the graph-timed sweep) |
| 7 | **Host-overhead removal** | (a) memoize `_resolve_attention_arch()` (was resolved ~20×/launch); (b) make the per-call `dataclasses.replace(num_kv_blocks=…)` conditional — its sole consumer is the i64 (>2 GiB) addressing decision, so small caches skip it; (c) the graph fast path below looks up the graph by a **cheap shape signature BEFORE** `supports`/`_tiled_cache_key`, so a replay skips the ~16 us of selector work + the kernarg pack. | a graphed tiny call **33→16 us**; decode `b1` **−21%**, all prefill **−2 to −8%**, zero kernel change |
| 8 | **Graph-vs-ungraph heuristic** | `_recommend_graph_replay`: graph when launch overhead is a large fraction — decode (`q==1`) and short prefill (`q ≤ 768`); ungraph long prefill (kernel-bound, varying tensors). The dispatcher auto-engages an internal graph (2D for short prefill, 3D for decode/short-prefill-on-3D) unless the caller is already capturing. | short-context shapes **52-62→14-46 us** (3-4×) → flips them from losing to winning; took graph-mode wins 4/14 → **14/14** |

### Lever 8 in detail — the graph heuristic

CUDA-graph capture only removes *per-launch host overhead* (Python dispatch +
kernarg pack + `hipModuleLaunchKernel`), so it pays off precisely when that
overhead is a large fraction of kernel time. `_recommend_graph_replay(problem)`
encodes the rule:

| regime | `max_seqlen_q` | graph? | why |
|--------|----------------|:------:|-----|
| decode | `== 1` | **yes** (3D) | tiny kernel, overhead-dominated; serving tensors are stable → replays |
| short prefill | `2 … 768` | **yes** (2D, or 3D if `select_path` routed there) | 1-2 KV tiles, overhead is a big fraction |
| long prefill | `> 768` | **no** | kernel-bound; overhead is noise and per-call tensors usually differ, so a graph would only add recapture cost |

The internal graph caches on a **cheap shape signature + tensor identities +
scalar args**, looked up *before* `supports`/`_tiled_cache_key` so a replay skips
the ~16 us of selector work and the kernarg pack (a graphed tiny call dropped
**33→16 us** — `dispatch_profile.py`); a new buffer set rebuilds (correctness)
while a stable set replays. It is skipped entirely when
`torch.cuda.is_current_stream_capturing()` so a framework that graphs the whole
forward (vLLM/SGLang) takes precedence. Toggle with `HIPDNN_GFX942_2D_GRAPH` /
`HIPDNN_GFX942_3D_GRAPH`.

## How the levers were found

> The sweep / probe / regression harnesses named below are archived under
> **`~/attention-archive/`** — only `final_shapes_check.py` (+ `shapes.json`,
> `parity_unified_attention.py`) ships in this folder. Copy one back to re-run it.

The tile/warp/staging/graph choices came from **smart-exhaustive** sweeps
(`exhaustive_sweep.py`, `loser_sweep.py`): enumerate the full Cartesian, filter
to valid specs (`__post_init__` + the `supports_tiled_2d` LDS gate — both
free), **deduplicate by `kernel_name()`** (~46 k enumerated → ~1.2 k unique),
**compile in parallel** (spawned pool, `maxtasksperchild` recycling so LLVM
workers don't bloat from ~25/s to ~1/s, SIGALRM per-compile watchdog),
correctness-gate vs the fp32 reference, then time the survivors (graph-timed for
the short-context sweep, to rank kernels not Python). `analyze_sweep.py`
extracts best-of-best; `regression_check.py` snapshots + `--compare` gates.

### What didn't work (so it isn't re-tried)

| tried | result |
|-------|--------|
| **T=128 ring** (and `T≥128` for D64) | slower than T=64 ring at every context length — occupancy loss > deeper overlap |
| `num_warps=8` / `ring_ldsseq` / `iglp_opt` / `early_v_schedule` | hang LLVM codegen for minutes (a Python SIGALRM can't interrupt a native-stuck compile) — excluded from the grids |
| internal **2D auto-graph for long prefill** | would rebuild per-call for varying-tensor prefill → regression; gated off by the heuristic |
| `grouped_kv2` softmax, 3D Q-fragment hoist | built + measured, no stable win; left behind flags |
| larger 3D decode split counts | no `num_segments`/tile combo materially moves decode — it's bandwidth/structure-bound |

## Cache-key safety & regression discipline

Every lever that changes the emitted kernel is in both `kernel_name()` and the
hot-path `_tiled_cache_key`/`_tiled_3d_cache_key`; the `_gfx942_flash_*` /
`_enable_gfx942_small_q_narrow` selectors are the single source of truth,
consumed identically by the spec builder, the support gate, and the cache key,
so a config change can never alias a previously-compiled binary. Every change
here was gated with `regression_check.py --compare`: the kernel levers held the
S2048/S4096 prefill wins to ±0.5%, and the host-overhead lever then took all
prefill **−2 to −8%** and decode `b1` **−21%** with zero kernel change.

## Running

`PYTHONPATH=python:python/ck_dsl/examples/gfx942/attention`, needs torch + a
gfx942 GPU:

```bash
# Correctness + perf over every shapes.json shape vs Torch (the headline table)
python python/ck_dsl/examples/gfx942/attention/final_shapes_check.py
```

The sweep / probe / regression harnesses live in `~/attention-archive/`; copy one
into this folder (so it can `import parity_unified_attention`) to re-run it, e.g.:

```bash
cp ~/attention-archive/{exhaustive_sweep,analyze_sweep,graph_probe,loser_sweep,regression_check}.py \
   python/ck_dsl/examples/gfx942/attention/
python .../exhaustive_sweep.py --grid tier1 --head-sizes 128 64 --seqlens 2048 --out ~/sweeps/2d.jsonl
python .../analyze_sweep.py ~/sweeps/2d.jsonl   # best-of-best
python .../graph_probe.py                       # graph vs un-graph ceiling
python .../loser_sweep.py                        # short-context kernel-config sweep
python .../regression_check.py --out base.json && python .../regression_check.py --compare base.json new.json
```

## Files

**In this folder** (the minimal correctness + perf check):

| file | role |
|------|------|
| `shapes.json` | canonical correctness (10) + perf (2) + decode (2) shapes |
| `final_shapes_check.py` | correctness + perf vs Torch over every shape (the headline) |
| `parity_unified_attention.py` | torch-reference parity harness (`shapes.json` loader, fp32 oracle, input gen) |

**Archived** under `~/attention-archive/` (the sweep / probe / regression tooling
that found & gated the levers):

| file | role |
|------|------|
| `exhaustive_sweep.py` | smart-exhaustive 2D lever sweep (filter → dedup → parallel compile → gate) |
| `loser_sweep.py` | graph-timed kernel-config sweep that found the light-narrow geometry |
| `graph_probe.py` | un-graphed vs CUDA-graph latency — quantifies the host-overhead ceiling |
| `dispatch_profile.py` | per-call host-dispatch breakdown (replace / supports / cache_key / replay) |
| `analyze_sweep.py` / `validate_configs.py` / `decode_ab.py` | best-of-best ranking / cross-seqlen A/B / decode KV-feed A/B |
| `regression_check.py` | latency+correctness snapshot, bandwidth-floor probe, `--compare` gate |
| `decode_sweep.py` / `torch_probe.py` / `benchmark_prefill2d.py` / `sweep_attention_matrix.py` | decode-config sweep / torch-SDPA semantics probe / earlier 2D benchmarks |
| `expected_perf.csv` | stale hand-maintained perf baseline, superseded by `final_shapes_check.py` |

## How the harness maps a dense SDPA problem onto the paged kernel

Each batch element becomes one sequence with `(query_len, kv_len) = (seqlen_q,
seqlen_k)`; the per-sequence block table is a contiguous, **non-overlapping** run
of `block_size=64`-token cache blocks, so the KV working set is genuinely
per-sequence-distinct. The decode baseline uses `is_causal=False` (a length-1
query attends to all keys); prefill uses `is_causal=True`. The launch grid is
recomputed exactly as the production dispatcher does, so the example exercises
the same build + launch plumbing as the provider.

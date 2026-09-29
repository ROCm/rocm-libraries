# gfx950 fp8 KV-decode: correctness gate, benchmarks, and perf investigation

Type-D record for the gfx950 fp8 KV-cache decode cohort. No measured numbers live
here — per `platform/AGENTS.md` §Compliance, methodology and qualitative findings
are documented, but software-achieved latency / speedup values are intentionally
omitted. This documents what this change adds, and the investigation behind why it
adds no production kernel change.

## What this change adds (and does not)

**Adds** on-GPU correctness + benchmark coverage for the cohort, which did not
exist before (`builders/gfx950/attention/decode/` was empty):

- an on-GPU numeric gate (`fp8_decode_3d_verify.py`), wired into
  `tools/run_checks.py --steps numeric`;
- the fp8 decode cohort in the benchmark set (`benchmark_decode_live.py` fp8
  support + `fp8_decode_d64_gqa8_shapes.json`);
- a cross-backend comparison harness (`fp8_decode_vs_baselines.py`), which probes
  for an external fp8 decode baseline and compares against it where one exists.

**Does not change any production kernel.** The gfx950 fp8 decode kernels are
untouched. The "perf-grade fp8 decode" was delivered by **#10583**
(`feat(rocke): resolve gfx950 num_cus from the device`): resolving the live
gfx950 CU count is what routes this cohort onto the fast 3D split-KV path. This
change contributes the coverage and the documented investigation, nothing in the
inner loop.

## Cohort

fp8 decode cohort: `D64`, `64/8` GQA, `block_size=16`, `Sq=1`, flash and sink,
`kv_len ∈ {2048, 8192}`, `batch ∈ {1, 64}`. fp8 dialect is `e4m3fn` (OCP), the
gfx950-native format, so the format-vs-arch guard admits.

## Why there is no kernel change (the investigation)

Static ISA inspection (runbook §3.1b; `fp8_decode_isa_inspect.py` builds the 3D
split-KV seg kernel and objdumps it) classifies it as VALU/softmax-bound
(`valu` ≫ `vmem_load`, and `ds_write` is tiny).
Every candidate lever was ruled out, each with evidence, not assumption:

- **Routing gate** (an earlier `_enable_fp8_decode_3d` forcing 3D) — **redundant**.
  #10583 already routes the cohort to 3D on a full part; a gate-on/off A/B at the
  production-resolved `num_cus` showed identical routed path and kernel with the
  gate off. Both the gate and its A/B harness were removed.
- **`waves_per_eu = 3`** — **neutral**. A same-run A/B (the knob the only
  difference, median of ≥5, `fp8_decode_wpe3_ab.py`) showed no win and no
  regression across the cohort. An occupancy hint with no measured effect does not
  earn a place; removed.
- **fp8-K-in-LDS / native-MFMA** (the 2D kernel's large decode lever) — **would be
  neutral here**. That lever saves loader LDS-write traffic, but split-KV already
  makes those writes negligible on the 3D seg (`ds_write` is small); the runbook's
  own caveat ("neutral if MemUnitStalled < 1%") applies. Not attempted.
- **`num_segments` split count** — the gfx950 pre-bump clamp already sits at the
  **plateau**. A median sweep (`fp8_decode_nseg_sweep.py`) showed finer splits are
  either noise-level (single-seq long-KV) or a clear regression (`batch=64`). No
  carve-out is justified.

**Conclusion:** the cohort is at its VALU/overhead floor. There is no cheap perf
lever, confirmed from two independent angles (static ISA + on-GPU A/Bs). This is
documented so the dead-ends are not re-investigated.

## Correctness gate

`fp8_decode_3d_verify.py` (this directory) builds the **shipped** seg + reduce
kernels for each cohort shape at the **production `num_segments`** (`_num_segments`,
including the gfx950 pre-bump clamp) and the **production-resolved `num_cus`** (so
the cohort routes to 3D exactly as at runtime), launches them with an fp32 partials
workspace, and compares the merged output against an **independent numpy** paged
decode-attention reference — `max_abs ≤ tol` (bf16 band `5e-2`, runbook §1.4). It
sweeps the full cohort and is wired into `tools/run_checks.py --steps numeric`
(self-skips with exit 0 off gfx950).

## Reproduce

```bash
# on a gfx950 node, rocke .venv, from library/ (numpy+ml_dtypes; torch for benches)
python -m builders.gfx950.attention.decode.fp8_decode_3d_verify          # numeric gate
python benchmarks/gfx950/attention/decode/benchmark_decode_live.py \
    --shapes benchmarks/gfx950/attention/decode/fp8_decode_d64_gqa8_shapes.json
python benchmarks/gfx950/attention/decode/fp8_decode_vs_baselines.py     # cross-backend compare

# investigation evidence (the "no lever" finding):
python benchmarks/gfx950/attention/decode/fp8_decode_wpe3_ab.py          # waves_per_eu neutral
python benchmarks/gfx950/attention/decode/fp8_decode_nseg_sweep.py --repeat 5  # split at plateau
python benchmarks/gfx950/attention/decode/fp8_decode_isa_inspect.py      # VALU-bound (no GPU)
```

Measured latency / speedup values are intentionally omitted from this repository
(recorded only in the internal perf record).

## File map

| Piece | Location |
|---|---|
| Live gfx950 CU-count resolver (routes the cohort to 3D) | `dispatch/attention/common.py` (`_resolve_num_cus`, #10583) |
| On-GPU numeric gate | `builders/gfx950/attention/decode/fp8_decode_3d_verify.py` (this dir) |
| Benchmark scenario + fp8 harness support | `benchmarks/gfx950/attention/decode/fp8_decode_d64_gqa8_shapes.json`, `benchmark_decode_live.py` |
| Cross-backend comparison harness | `benchmarks/gfx950/attention/decode/fp8_decode_vs_baselines.py` |
| Investigation A/Bs (evidence for "no lever") | `benchmarks/gfx950/attention/decode/fp8_decode_{wpe3_ab,nseg_sweep,isa_inspect}.py` |

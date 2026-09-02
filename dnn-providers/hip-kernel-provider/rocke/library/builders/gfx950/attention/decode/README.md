# gfx950 fp8 KV-decode → 3D split-KV routing (case study)

Type-D perf optimization for the fp8 KV-cache decode cohort on gfx950. This is
the replayable record for the routing + occupancy change; **no measured numbers
live here** — per `platform/AGENTS.md` §Compliance, software-achieved latency /
speedup figures go to the protected Confluence page only. This documents *what*
changed, *why*, the methodology, and the honest losses.

## Cohort

gpt-oss decode: `D64`, `64/8` GQA, `block_size=16`, `Sq=1`, flash **and** sink,
`kv_len ∈ {2048, 8192}`, `batch ∈ {1, 64}`. fp8 dialect is `e4m3fn` (OCP), the
gfx950-native format, so the format-vs-arch guard admits.

## The change

Two gfx950-only gates in `kernels/common/attention_unified.py`, sharing one
cohort predicate `_is_fp8_long_kv_decode` (fp8 + all_decode + no sliding window
+ `max_seqlen_k > 512`):

- **Routing** — `_enable_fp8_decode_3d` routes the cohort onto the 3D split-KV
  path (`select_path() == "3d"`).
- **Occupancy** — `_enable_gfx950_fp8_decode_wpe3` sets `waves_per_eu = 3` on the
  3D decode kernel. Pure AMDGPU occupancy hint (kernel attribute), so output is
  **bit-identical**.

## Why 3D (methodology, runbook Steps 0 + 3)

- **Bottleneck classification** (§3.1b): static ISA inspection of the fp8 decode
  inner loop showed it is **VALU-bound** (per-element dequant dominates; `valu`
  ≫ `vmem_load`), not memory-bound.
- **Lever sweep** (Step 0): the shipped-config levers (`num_warps`,
  `waves_per_eu`, fp8-QK-in-LDS, tile size) were swept via a raw-lever sweep
  before concluding the gap was structural. The structural lever was the
  2D-vs-3D routing: 3D split-KV fans the per-element dequant across many CTAs,
  whereas 2D leaves the large gfx950 part under-filled for this small-grid
  decode.

## The CU-count root cause (honest framing)

The 2D-vs-3D decision compares `num_2d_prgms` against `target = num_cus * 4`.
`_resolve_num_cus` returns a legacy `120` for gfx950 (it queries the live device
only for gfx942-on-gfx942), so `target` is undersized. At that undersized target
the `batch=64` shapes (`num_2d = 768`) exceed it and mis-route to 2D; at gfx950's
true CU count they route to 3D on their own.

So the routing gate **compensates for the mis-resolved CU count**; it is not
overriding a correct-CU decision. The clean root fix is correcting
`_resolve_num_cus` for gfx950 — which also fixes bf16 decode routing — and is
deferred to the bf16 decode owners. This gate is the scoped, fp8-only interim.

## Honest losses (type-D DoD D4)

- `batch=1` shapes route 3D **regardless** of the gate (`num_2d` is tiny) — the
  gate is a no-op there. Only `batch=64` is actually re-routed.
- The gate's cohort predicate has no upper batch bound, so at a *very* large
  batch where 2D would legitimately win at the correct CU count, forcing 3D would
  regress. The measured cohort stays on the win side of that boundary; the
  unbounded edge is the documented risk and the reason the root `_resolve_num_cus`
  fix is the durable answer.
- `waves_per_eu = 3` is an occupancy hint: it cannot change correctness, only
  perf/variance; it is included because a same-run A/B (median of ≥3, the knob
  the only difference) showed a consistent, non-regressing result on the cohort.

## Correctness gate

`fp8_decode_3d_verify.py` (this directory) builds the **shipped** seg + reduce
kernels for each cohort shape, launches them with an fp32 partials workspace, and
compares the merged output against an **independent numpy** paged decode-attention
reference — `max_abs ≤ tol` (bf16 band `5e-2`, runbook §1.4). It sweeps the full
cohort and is wired into `tools/run_checks.py --steps numeric` (self-skips with
exit 0 off gfx950). This is the on-GPU gate that lets a regression on the
re-routed kernels fail red rather than ship.

## Reproduce

```bash
# on a gfx950 node, rocke .venv, from library/
python -m builders.gfx950.attention.decode.fp8_decode_3d_verify          # numeric gate
python -m builders.gfx950.attention.decode.fp8_decode_3d_verify --kv-len 8192 --num-seqs 64

# rocKE vs AITER / Triton comparison (probes for an fp8 baseline)
python benchmarks/gfx950/attention/decode/fp8_decode_vs_baselines.py
```

Measured latency / speedup: **Confluence** (compliance — not in the repo).

## File map

| Piece | Location |
|---|---|
| Cohort predicate + routing + occupancy gates | `kernels/common/attention_unified.py` (`_is_fp8_long_kv_decode`, `_enable_fp8_decode_3d`, `_enable_gfx950_fp8_decode_wpe3`) |
| Build guards (routing / wpe3) | `tests/test_attention_builds.py` (`test_fp8_long_kv_decode_routes_3d`, `test_gfx950_fp8_decode_3d_waves_per_eu`) |
| On-GPU numeric gate | `builders/gfx950/attention/decode/fp8_decode_3d_verify.py` (this dir) |
| Cross-backend comparison harness | `benchmarks/gfx950/attention/decode/fp8_decode_vs_baselines.py` |

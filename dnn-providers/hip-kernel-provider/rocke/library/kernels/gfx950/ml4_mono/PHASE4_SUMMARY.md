# ML4 rocKE Phase 4 summary

## Delivered

- `mono_layout.py`: source-path loader for aiter's ML4 gfx950 `layout.py` and
  `kernel.py`; records the source digest and exposes:
  - per-width scratch/symmetric offsets;
  - `stage_tasks()` and cumulative CTA bases;
  - ML4 tag (`step * 128 + layer + 1`) and rank peer-slot rules;
  - per-width `WORK_WORDS` LDS arena checks.
- `abi.py` and `layout.py`: compatibility exports for the ML4 ABI/layout.
- `stage_fn.py`: ML4 family ABI and `rocke_ml4_` symbol namespace.
- `../stage_fn_common.py`: shared, family-parameterized stage-function prologue
  used by both ML4 and qwen38.
- `mxfp4.py`: aiter shuffled native MXFP4 weight and E8M0 scale addressing.
- `moe.py`: router, UG (S=1 and S>1 structure), and `down_gemv` emitters with
  explicit mailbox boundaries, MXFP4 MFMA chains, and mutant hooks. The
  router produces the xq/xqs MXFP8 plane; UG uses FlyDSL's `w_rg`/`w_ln`
  shuffle and eight-row task geometry; down_gemv stops at the LDS `outs`
  partial seam.
- `stage_moe.py`: generator with `r1,r2,r4,r4s`, sidecars, layout digest,
  task/base metadata, mailbox mode, stage pairing, and LDS checks.
- `moe_fused.py` and `stage_moe_fused.py`: paired expert-major fused-MoE host
  plan and linkable m3/m4/m5 scaffold.
- `README.md`: usage and remaining work.

## Checks

With `PYTHONPATH=platform/python;library` and `ROCKE_BACKEND=python`:

- `stage_moe --help` works without an aiter checkout.
- The full split generator ran for `router,ug,down_gemv` and
  `r1,r2,r4,r4s` against the aiter checkout after the Phase 4b changes.
- The fused scaffold generator ran for `r1,r4s`.
- A missing aiter path fails clearly after argument parsing.
- The router mutant produced a different LLVM hash.
- Existing `test_qwen38_stage_fn.py`: **30 passed**.
- IDE lint diagnostics: no errors.

## Open TODOs

1. Router is bitwise-aimed for xq/xqs gamma/quantization and mailbox ordering,
   but folded-K reduction, RMSNorm reduction, top-4 selection, and score
   publication remain approximate.
2. UG is bitwise-aimed for task count, native MXFP4 `w_rg`/`w_ln` addressing,
   routed SiLU shape, and `mid` boundaries; exact S=1/S>1 shared-expert
   scheduling and LDS `outs` reduction remain approximate.
3. Down GEMV is bitwise-aimed for mid quantization, expert/slot traversal,
   route weighting, native down addressing, and the `lds:outs` handoff.
   Exact FlyDSL tile accumulation order and peer-reduce integration require
   GPU comparison.
4. Add GPU bitwise gates for all widths, layer kinds, mailbox sidecars, and
   all mutants.
5. Replace the fused scaffold body with expert-major IOD-balanced streaming
   and its exact combine path.

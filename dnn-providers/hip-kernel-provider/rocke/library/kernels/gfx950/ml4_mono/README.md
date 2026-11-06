# ML4 rocKE mono family

Two ways of running Mistral Large 4 decoder layers from rocKE live here:

- **`layer/`: the whole layer as one persistent launch.** Every stage
  (attention with the fused sparse indexer, the MoE, both TP reductions) in
  one kernel that takes the FlyDSL ML4 mono launcher's arguments and scratch
  unchanged, so aiter's `rocke_layer.py` (`AITER_ML4_ROCKE=<rocKE tree>`)
  launches it in place of the FlyDSL kernel. The `draft` kind builds the
  DSpark drafter's layers (dense MLP, bf16 latent cache, windowed attention).
  See [`layer/README.md`](layer/README.md) for the stages, schedule knobs and
  checks.
- **The modules below: device functions linked into the FlyDSL kernel**
  (aiter's `extern_stages.py`). This is the Phase 4 port
  ([`PHASE4_SUMMARY.md`](PHASE4_SUMMARY.md)): it reads the authoritative ML4
  layout from aiter by source path and emits device functions with the
  `rocke_ml4_` namespace. The open work at the end of this file is this
  path's; `layer/` implements its own complete MoE.

## Split stages

```powershell
$env:PYTHONPATH = "platform/python;library"
$env:ROCKE_BACKEND = "python"
python -m kernels.gfx950.ml4_mono.stage_moe `
  --out DIR --aiter-root C:\path\to\aiter `
  --variants r1,r2,r4,r4s
```

The output has `router`, `ug`, and `down_gemv` LLVM/JSON pairs. Sidecars carry
the aiter source digest, task counts/bases, mailbox mode, stage pairing, and
the LDS arena check. `--mutate router_score|ug_scale|down_order` emits a
deliberately different body for the bitwise A/B gate.

`mxfp4.py` mirrors aiter's shuffled native MXFP4 weight and E8M0 scale
addressing. The stage bodies are intentionally kept separate from the
expert-major fused path so each body can be compared against the corresponding
FlyDSL stage.

## Fused scaffold

`moe_fused.py` and `stage_moe_fused.py` provide the expert-major plan, shared
expert CTA group, IOD line split, paired `m3/m4/m5` sidecars, and linkable LDS
barrier scaffold. The expert-major router → gate/up → down → combine stream is
not complete yet.

Open work (extern-stage path):

1. Finish the ML4 router's exact folded K reduction and top-4 softmax.
2. Finish S=1/S>1 UG routing, native MXFP4 scale reuse, and FP8 `mid`.
3. Finish `down_gemv` route weighting and leave the exact peer-reduce epilogue
   to FlyDSL.
4. Add GPU bitwise gates for all four variants and mutants.
5. Replace the fused scaffold with expert-major weight-balanced streaming.

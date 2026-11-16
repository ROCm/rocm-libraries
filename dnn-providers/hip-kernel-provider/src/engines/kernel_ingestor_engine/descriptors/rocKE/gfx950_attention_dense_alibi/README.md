# gfx950_attention_dense_alibi

The ALiBi kernels of `hipkernel:Gfx950AttentionDense`, a satellite bundle of `../gfx950_attention_dense/`: the same
engine, dispatch and KMD (the descriptors live there), with a KDP of its own. Built by rocKE
(`kernels/gfx950/attention_dense.py`, builder `build_attention_dense`, `use_alibi`).

## Coverage

- Architecture: gfx950 only.
- Variants: 66, one per head configuration of the home catalog and dtype (33 x BF16/FP16), on the `qb_minor` grid
  at the 128/64 tile, KMD `use_alibi` 1. The tile serves every seqlen_q a 256 tile serves, plus the odd multiples
  of 128.
- ALiBi (`alibi_mask`): hipDNN's slopes (SdpaSemanticsRegister SEM-021), computed in the kernel from the query head
  count; the launch passes a null `alibi_slopes_ptr`. Top-left causal mask only, the one mask hipDNN defines ALiBi
  for; no window, no stats output.
- The native matcher pairs an ALiBi graph only with these kernels and a plain graph only with the home's
  (`packs/Gfx950AttentionDenseNative.cpp`).

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx950_attention_dense.kdp.json` | DVC | 0.1 MB, the 66 variants; same KDP name as the home's, its own ids. Pointer: `artifacts.dvc` |

## Notes

- Fetch `artifacts/` with `dvc pull -r ingestor artifacts.dvc`. See `../../README.md`.
- Generated from `projects/hipdnn/tools/IngestorGenerator/configs/gfx950_attention_dense_alibi.yaml` (the engine
  block equals the home's; generate.py checks it and reuses the home's engine ids).
- Changing the KDP changes its md5. Update the `.dvc` pointer in the same commit.

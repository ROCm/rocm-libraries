# gfx950_attention_unified_2d_alibi

The ALiBi kernels of `hipkernel:Gfx950AttentionUnified2D`, a satellite bundle of
`../gfx950_attention_unified_2d/`: the same engine, dispatch and KMD (the descriptors live there), with a KDP of its
own. The home catalog stays as it is, so the engine's trained rankers (`engine.trained_catalog_sha256`) still hold.
Built by rocKE (`kernels/gfx950/attention_tiled_2d.py`, builder `build_unified_attention_2d_tiled`, `use_alibi`).

## Coverage

- Architecture: gfx950 only.
- Variants: 66, one per head configuration of the home catalog and dtype (33 x BF16/FP16), at the default
  geometry, KMD `use_alibi` 1.
- ALiBi (`alibi_mask`): hipDNN's slopes (SdpaSemanticsRegister SEM-021), computed in the kernel from the query head
  count. Top-left causal mask only, the one mask hipDNN defines ALiBi for; no stats output.
- The native matcher pairs an ALiBi graph only with these kernels and a plain graph only with the home's
  (`packs/Gfx950AttentionUnified2DNative.cpp`), so the ranker never ranks one against the other.

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx950_attention_unified_2d.kdp.json` | DVC | 0.11 MB, the 66 variants; same KDP name as the home's, its own ids. Pointer: `artifacts.dvc` |

## Notes

- Fetch `artifacts/` with `dvc pull -r ingestor artifacts.dvc`. See `../../README.md`.
- Generated from `projects/hipdnn/tools/IngestorGenerator/configs/gfx950_attention_unified_2d_alibi.yaml` (the
  engine block equals the home's; generate.py checks it and reuses the home's engine ids).
- Changing the KDP changes its md5. Update the `.dvc` pointer in the same commit.

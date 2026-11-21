# gfx942_attention_dense_alibi

The ALiBi kernels of `hipkernel:Gfx942AttentionDense`, a satellite bundle of `../gfx942_attention_dense/`: the same
engine, dispatch and KMD (the descriptors live there), with a KDP of its own. Built by rocKE
(`kernels/gfx942/attention_dense.py`, builder `build_attention_dense`, `use_alibi`).

## Coverage

- Architecture: gfx942 only.
- Variants: 44, at the 128/64 tile, KMD `use_alibi` 1: on the `qb_minor` grid every head size (D64, D128), dtype,
  window (none, 128, 512, 1024, 2048) and `emit_lse` (40), and on the D128 persistent grid each dtype and `emit_lse`
  without a window (4). seqlen_q and seqlen_kv must be multiples of 128 and 64.
- ALiBi (`alibi_mask`): hipDNN's slopes (SdpaSemanticsRegister SEM-021), computed in the kernel from the query head
  count; the launch passes a null `alibi_slopes_ptr`. Top-left causal band only (any left window, no right bound),
  the one band hipDNN defines ALiBi on.
- The native matcher pairs an ALiBi graph only with these kernels and a plain graph only with the home's
  (`packs/Gfx942AttentionDenseNative.cpp`).

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx942_attention_dense.kdp.json` | DVC | the 44 variants; same KDP name as the home's, its own ids. Pointer: `artifacts.dvc` |

## Notes

- Fetch `artifacts/` with `dvc pull -r ingestor artifacts.dvc`. See `../../README.md`.
- Generated from `projects/hipdnn/tools/IngestorGenerator/configs/gfx942_attention_dense_alibi.yaml` (the engine
  block equals the home's; generate.py checks it and reuses the home's engine ids).
- Changing the KDP changes its md5. Update the `.dvc` pointer in the same commit.

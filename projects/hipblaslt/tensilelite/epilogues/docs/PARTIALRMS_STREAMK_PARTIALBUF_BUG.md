# Known issue: PartialRMS `partialBuf` not written on StreamK split-K path

## Status

Open. All deployed `BBS_H_PRMS_RA` (bf16 full-chain) solutions use `StreamK: 3`.
The StreamK split-K store path (fixup/finishing workgroup) does **not** write the
`partialBuf` (per-row Σx² partial sums) that K2 (`row_div`) reads to compute the
RMS inverse. The buffer is left at zero, producing wrong output for any problem
where StreamK actually splits (i.e. `itersPerTile > 1`).

## Impact

The full-chain bf16 test (`FusedEpilogueE2E.fullRmsNormResidualAddMatchesReference`,
M=N=1024, K=4096) fails on gfx950. The bf16 decomposed producer path
(`decomposedProducerConsumerMatchesReference`, K=64) is unaffected because small K
forces `itersPerTile=1`, causing StreamK to degenerate to a whole-tile data-parallel
launch that takes the correct store path. Large-K production workloads that use the
full-chain bf16 path are silently wrong.

The `F8F8S_MXAE8B32_MXBE8B32_H_PRMS_RA` (mxfp8 chain) solutions also use
`StreamK: 3`; the same bug is likely present there for large K, though it has not
been empirically confirmed.

## Symptom

- `row_div` (K2) reads an all-zero `partialBuf`, computes
  `rstd = 1/sqrt(0/N_hidden + eps) = 1/sqrt(eps) ≈ 316.23` (for eps=1e-5), and
  scales every D element by that factor.
- Output is `K1_D × 316` element-for-element. ~1,048,075 / 1,048,576 elements wrong,
  max abs error ~318, max rel error ~212,000.
- The error is exact and reproducible: sign matches the reference, magnitude is
  a fixed `1/sqrt(eps)` multiple.
- Passes at K=64 (itersPerTile=1, whole-tile DP). Fails at K=4096.

## Discriminator

The failure is gated on `itersPerTile > 1`, not on K alone. The threshold depends on
the tile's DepthU (`itersPerTile = K / DepthU`). Any problem where StreamK actually
splits workgroups across K is affected.

Forcing the StreamK grid to equal the tile count via environment overrides
(`TENSILE_STREAMK_FIXED_GRID`, `TENSILE_STREAMK_DATA_PARALLEL`) does not help: the
selected `StreamKForceDPOnly=0` kernel still takes its split-K finishing path
regardless of the externally forced grid shape.

The 45 `StreamKForceDPOnly=1` solutions deployed alongside the 45
`StreamKForceDPOnly=0` solutions are **not affected**: SKFDPO1 forces `skTiles=0`
(every output tile computed whole), so the finishing path is never taken.

## Root cause location

`tensilelite/Tensile/Components/Subtile/SubtilePartialRMSEmit.py` — the StreamK
split-K finishing/fixup store path does not emit the `partialBuf` Σx² write after
the full-K reduction. The whole-tile (data-parallel) store path emits it correctly.

The C++ runtime dispatch is in `ContractionSolution.cpp::resolveStreamKSettings`
(L4714) and `streamKStaticSplit` (L194). The `partialBuf` kernarg is appended at
`ContractionSolution.cpp:1445`; `requiredWorkspaceSize`/`partialRMSPartialBufBytes`
are at L4244–4352.

## Workaround

Constrain heuristic solution selection to `StreamKForceDPOnly=1` solutions for any
problem whose epilogue has `hasRMSNorm || hasPartialRMSStats`. Those kernels already
ship and correctly write `partialBuf` at all K. This requires a host-side change in
`tensile_host.cpp` and no device-library rebuild.

## Related issue

See `PARTIALRMS_WG2X4_KNOWN_ISSUE.md` for a separate `partialBuf` miscompute
affecting 8-wave (WG2×4) workgroups — that bug is in the cross-wave LDS reduction
and is independent of StreamK.

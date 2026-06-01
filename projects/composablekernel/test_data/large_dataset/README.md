# Large dataset validation catalogue (AICK-1277)

Checked-in convolution shape catalogues for the **large-dataset validation**
stage that exercises both the **Old CK** (XDL / cshuffle) and **CK Tile**
backends across the three convolution directions.

These CSVs are the **sole runtime input** to the stage. The source measurement
workbook (`Convolution Backward Data Measurements.xlsx`) is only an offline seed
and is never read by the build or CI.

## Files

| File | Direction | Notes |
|------|-----------|-------|
| `fwd_2d.csv`         | forward          | 2D smoke shapes |
| `bwd_data_2d.csv`    | backward-data    | 2D smoke shapes, includes one stride>1 / g=1 (ALMIOPEN-1959 family) shape |
| `bwd_weight_2d.csv`  | backward-weight  | 2D smoke shapes |
| `smoke_3d.csv`       | all (3D)         | one 3D shape, shared, so the 3D test suites are non-empty |

This is the **Phase 1 smoke set** (walking skeleton). Phase 2 replaces
`bwd_data_*.csv` with the full customer catalogue (~1250 deduped geometries)
and synthesizes the fwd / bwd_weight catalogues from the shared geometries.

## CSV format

Geometry-only, repo-standard (same schema the existing
`test_grouped_convnd_*_dataset_xdl` tests consume via
`test/common/csv_test_loader.hpp`). Channels are **per group**.

2D header:

```
NDim,Groups,BatchSize,OutChannels,InChannels,KernelH,KernelW,InputH,InputW,OutputH,OutputW,StrideH,StrideW,DilationH,DilationW,LeftPadH,LeftPadW,RightPadH,RightPadW,TestName
```

3D header adds the depth columns:

```
NDim,Groups,BatchSize,OutChannels,InChannels,KernelD,KernelH,KernelW,InputD,InputH,InputW,OutputD,OutputH,OutputW,StrideD,StrideH,StrideW,DilationD,DilationH,DilationW,LeftPadD,LeftPadH,LeftPadW,RightPadD,RightPadH,RightPadW,TestName
```

The `Output*` columns are informational (the loader recomputes output extents).
dtype / layout / split-K are expanded by the gtest itself, so the catalogue is
dtype-agnostic.

## How the stage consumes these

The Jenkins stage copies the per-direction CSV into the filename the gtests
expect (`test_data/conv_test_set_2d_dataset.csv` / `_3d`) immediately before
running each direction's `*_dataset_xdl` (Old CK) and `*_dataset_tile` (CK Tile)
binary.

## Adding a shape

Append one row to the relevant CSV. No code changes required.

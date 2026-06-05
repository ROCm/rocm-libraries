# P1 codegen — per-input whole-project line attribution

Per-input contribution to whole-project codegen coverage (statements + branches),
ranked by **marginal_lines** (lines a given input adds on top of all higher-ranked
inputs already in the set — a greedy set-cover ordering). `whole_project_lines` is
that input's standalone whole-project coverage; `arch_union_lines` is the union of
all inputs for the arch (the per-arch ceiling reachable by this input set).

Distilled attribution data:
- `Tensile/Tests/unit/characterization/_codegen/attribution-gfx942.json`
- `Tensile/Tests/unit/characterization/_codegen/attribution-gfx950.json`
- `Tensile/Tests/unit/characterization/_codegen/attribution-gfx90a.json`

29 inputs measured total (gfx942: 12, gfx950: 12, gfx90a: 5).

## gfx942 — arch_union_lines = 21900

| id | kernels | whole_project_lines | marginal_lines | likely_params |
|---|---|---|---|---|
| gfx942__BBS_BH_Bias_Act_yaml | 1 | 19219 | 19219 | DataType(bf16), ActivationType, Bias |
| gfx942__F8N_multi_yaml | 4 | 18311 | 1133 | DataType(fp8), MatrixInstruction |
| gfx942__GG_yaml | 1 | 18194 | 448 | GroupedGemm, DataType |
| gfx942__DTV_yaml | 1 | 18150 | 388 | DirectToVgpr, MatrixInstruction |
| gfx942__LSU_MX_yaml | 2 | 18011 | 366 | LocalSplitU, DataType(MXFP) |
| gfx942__Grad_yaml | 1 | 17643 | 241 | Gradient, Bias |
| gfx942__DB_yaml | 2 | 16711 | 53 | DataType(fp64), MatrixInstruction |
| gfx942__HSS_BH_Bias_yaml | 1 | 17951 | 32 | DataType(fp16in/fp32out), Bias |
| gfx942__HHS_BH_Bias_GG_yaml | 1 | 18212 | 14 | DataType(fp16), GroupedGemm, Bias |
| gfx942__MX_yaml | 1 | 17984 | 3 | DataType(MXFP), MatrixInstruction |
| gfx942__SB_Bias_Aux_yaml | 1 | 17837 | 3 | Bias, Aux/ScaleAlphaVec |
| gfx942__GSU_yaml | 2 | 16711 | 0 | GlobalSplitU |

## gfx950 — arch_union_lines = 22121

| id | kernels | whole_project_lines | marginal_lines | likely_params |
|---|---|---|---|---|
| gfx950__StreamK_B8F8_yaml | 5 | 19547 | 19547 | StreamK, DataType(F8/B8), CustomSchedule |
| gfx950__BBS_yaml | 1 | 18631 | 1277 | DataType(BF16), GlobalSplitU, ShiftVectorComponents |
| gfx950__MX_yaml | 1 | 18695 | 515 | DataType(MX/microscaling), LocalRead, MatrixInstruction |
| gfx950__HHS_yaml | 1 | 18209 | 276 | DataType(half/HHS), PackData |
| gfx950__I8_GSU_yaml | 1 | 16744 | 192 | DataType(Int8), GlobalSplitU, SIA |
| gfx950__HSS_yaml | 1 | 17936 | 146 | DataType(half-in/single-out) |
| gfx950__DTL_yaml | 1 | 8173 | 74 | CustomKernel/DirectToLds, CustomSchedule |
| gfx950__StreamK_F8F8S_yaml | 1 | 18801 | 71 | StreamK, DataType(F8), GlobalWriteBatch/PackData |
| gfx950__SB_yaml | 1 | 17815 | 23 | SourceBranch/edge |
| gfx950__F8B8BS_yaml | 3 | 5972 | 0 | DataType(F8/B8) |
| gfx950__MX_StreamK_yaml | 1 | 18690 | 0 | StreamK, DataType(MX) |
| gfx950__WaveSplitK_yaml | 1 | 17910 | 0 | WaveSplitK |

## gfx90a — arch_union_lines = 18587

| id | kernels | whole_project_lines | marginal_lines | likely_params |
|---|---|---|---|---|
| gfx90a__BBS_yaml | 1 | 17875 | 17875 | DataType=BF16, ComputeDataType=Single |
| gfx90a__DB_yaml | 2 | 16834 | 660 | DataType=Double(FP64), ShiftVectorComponents/edge-store paths |
| gfx90a__HSS_yaml | 1 | 17866 | 28 | DataType=Half/DestDataType=Single, PackData |
| gfx90a__HHS_yaml | 1 | 17863 | 15 | DataType=Half/DestDataType=Half, GlobalWriteBatch packing |
| gfx90a__SB_yaml | 1 | 17681 | 9 | DataType=Single/DestDataType=BF16 conversion |

## Config channel (BenchmarkProblems -> Solutions)

Status: **active** — `Tensile/Tests/unit/characterization/_codegen/config_harness.py`
(`emit_kernels_from_config`) committed alongside this attribution. Smoke verified in
a fresh container process (one rocisa process per the per-process-footprint rule):

```
docker exec -e PYTHONPATH=/work/projects/hipblaslt/tensilelite:/work/projects/hipblaslt/tensilelite/Tensile/Tests/unit/characterization/_codegen \
  -w /work/projects/hipblaslt/tensilelite tl-char \
  python -c "from config_harness import emit_kernels_from_config as e; \
    r=e('Tensile/Tests/common/gemm/fp32_nt.yaml'); \
    print('KERNELS', len(r), 'ERR0', all(x[2]==0 for x in r))"
# -> KERNELS 1 ERR0 True
```

## Commands used

Per `work/tensilelite-characterization/coverage-methodology.md`. Each input was run
as exactly one fresh `docker exec` process (rocisa footprint is per-process), with a
single dedicated `COVERAGE_FILE` per input and kernels bounded by
`emit_kernels_from_logic(..., limit=8)`. Whole-project line totals are the bottom
`TOTAL` row of `coverage report` (branch-inclusive via `--cov-config=pyproject.toml`),
i.e. the no-`--include` whole-project variant described in the methodology's
"What the numbers do and don't mean" section. Per-input shape (gfx942 example):

```bash
CON=tl-char
PROJ=/work/projects/hipblaslt/tensilelite
# one fresh process per input; one COVERAGE_FILE per input; --cov is the PATH word Tensile
docker exec -e PYTHONPATH=$PROJ -e COVERAGE_FILE=$PROJ/.coverage.<input> -w $PROJ $CON \
  python -c "from codegen_harness import emit_kernels_from_logic; \
             emit_kernels_from_logic('<arch>/<input>.yaml', limit=8)"   # under coverage run
# whole-project TOTAL (no --include):
docker exec -e COVERAGE_FILE=$PROJ/.coverage.<input> -w $PROJ $CON coverage report | tail -1
```

`marginal_lines` was computed greedily: rank inputs by added (marginal) whole-project
lines over the union of already-selected higher-ranked inputs; `arch_union_lines` is
the union over all of the arch's inputs (the set-cover ceiling).

Raw per-input scratch JSONs (~1.3 GB) were removed after distillation to keep the
tree clean; this summary plus the three `attribution-<arch>.json` files are the
durable receipts.

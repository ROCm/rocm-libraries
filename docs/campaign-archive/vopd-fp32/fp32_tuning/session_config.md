# FP32 gfx1100 Tuning — Session Config

## Key Paths
- Tensile root: ~/TheRock/rocm-libraries/projects/hipblaslt/tensilelite/
- Client: tensilelite/build_tmp/tensilelite/client/tensilelite-client
- Merge tool: Tensile/bin/TensileMergeLibrary
- gfx1151 FP32 logic: library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/GridBased/gfx1151_Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs.yaml
- Working dir: ~/vopd_sgemm/fp32_tuning/

## Run Command Template
```bash
cd ~/TheRock/rocm-libraries/projects/hipblaslt/tensilelite
python3 Tensile/bin/Tensile ~/vopd_sgemm/fp32_tuning/waves/<wave_yaml> <output_dir> \
  --code-object-version=4 \
  --cxx-compiler=/opt/rocm/bin/amdclang++ \
  --library-format=msgpack
```

## Non-MI FP32 Mandatory Params (gfx1100)
- WavefrontSize: 32
- VectorWidthA: 1
- VectorWidthB: 1
- GlobalReadVectorWidthA: 1
- GlobalReadVectorWidthB: 1
- LocalReadVectorWidth: 1
- ScheduleIterAlg: 0 or 1 (NOT 2 or 3)
- GlobalSplitU: 1
- KernelLanguage: Assembly

## CSV Column Mapping
- SizeI = M (rows of A, rows of C/D)
- SizeJ = N (cols of B, cols of C/D)
- SizeK = Batch
- SizeL = K (reduction dimension)

## Campaign Start: 2026-05-30

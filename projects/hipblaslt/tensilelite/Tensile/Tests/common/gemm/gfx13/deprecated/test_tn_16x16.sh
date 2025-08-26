#!/bin/bash

hipblaslt=/dockerx/canis30/rocm-libraries/projects/hipblaslt

# tn
# rm -r $hipblaslt/build/hss_tn_16x16_test
# $hipblaslt/build/tensilelite/Tensile.sh $hipblaslt/tensilelite/Tensile/Tests/common/gemm/gfx13/hss_tn_16x16.yaml $hipblaslt/build/hss_tn_16x16_test --rocm-agent-enumerator rocm_agent_enumerator --cxx-compiler /opt/gfx13/bin/amdclang++ --c-compiler /opt/gfx13/bin/amdclang
# rm $hipblaslt/build/hss_tn_16x16_test/1_BenchmarkProblems/Cijk_Alik_Bljk_HSS_BH_UserArgs_00/00_Final/source/build_tmp/SOURCE/assembly/*
# cp $hipblaslt/tensilelite/Tensile/Tests/common/gemm/gfx13/hss_tn_16x16.s $hipblaslt/build/hss_tn_16x16_test/1_BenchmarkProblems/Cijk_Alik_Bljk_HSS_BH_UserArgs_00/00_Final/source/build_tmp/SOURCE/assembly/

# /dockerx/canis30/rocm-libraries/projects/hipblaslt/build/hss_tn_16x16/1_BenchmarkProblems/Cijk_Alik_Bljk_HSS_BH_UserArgs_00/00_Final/source/build_tmp/SOURCE/assembly/Cijk_Alik_Bljk_HSS_BH_UserArgs_MT16x16x16_MI16x1UzRBAFTR-A6xT6h5xBsfZ5ggi5P0ymxuynrgGDNX2y8=.s
# du=16
asm=Cijk_Alik_Bljk_HSS_BH_UserArgs_MT16x16x16_MI16x1UzRBAFTR-A6xT6h5xBsfZ5ggi5P0ymxuynrgGDNX2y8=
# du=32
#asm=Cijk_Alik_Bljk_HSS_BH_UserArgs_MT16x16x32_MI16x13GN7fp_dcyFmIpPnau6FDVY-IV_GadZlkw9aLiq8xtQ=
/opt/gfx13/bin/amdclang++ -x assembler --target=amdgcn-amd-amdhsa  -mcode-object-version=4 -c -mcpu=gfx1300 -mno-wavefrontsize64 $hipblaslt/build/hss_tn_16x16/1_BenchmarkProblems/Cijk_Alik_Bljk_HSS_BH_UserArgs_00/00_Final/source/build_tmp/SOURCE/assembly/${asm}.s -o $hipblaslt/build/hss_tn_16x16/1_BenchmarkProblems/Cijk_Alik_Bljk_HSS_BH_UserArgs_00/00_Final/source/build_tmp/SOURCE/assembly/${asm}.o
make -f $hipblaslt/tensilelite/Makefile co TENSILE_OUT=$hipblaslt/build/hss_tn_16x16/ ARCH=gfx1300 WAVEFRONTSIZE=-mno-wavefrontsize64
$hipblaslt/build/hss_tn_16x16/0_Build/client/tensile_client --config-file $hipblaslt/build/hss_tn_16x16/1_BenchmarkProblems/Cijk_Alik_Bljk_HSS_BH_UserArgs_00/00_Final/build/../source/ClientParameters.ini
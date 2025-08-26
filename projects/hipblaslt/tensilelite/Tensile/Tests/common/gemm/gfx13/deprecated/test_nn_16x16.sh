#!/bin/bash

hipblaslt=/dockerx/rocm-libraries/projects/hipblaslt

# nn
$hipblaslt/build/tensilelite/Tensile.sh $hipblaslt/tensilelite/Tensile/Tests/common/gemm/gfx13/hss_nn_16x16.yaml $hipblaslt/build/hss_nn_16x16_test --rocm-agent-enumerator rocm_agent_enumerator --cxx-compiler /opt/gfx13/bin/amdclang++ --c-compiler /opt/gfx13/bin/amdclang
rm $hipblaslt/build/hss_nn_16x16_test/1_BenchmarkProblems/Cijk_Ailk_Bljk_HSS_BH_UserArgs_00/00_Final/source/build_tmp/SOURCE/assembly/*
cp $hipblaslt/tensilelite/Tensile/Tests/common/gemm/gfx13/hss_nn_16x16.s $hipblaslt/build/hss_nn_16x16_test/1_BenchmarkProblems/Cijk_Ailk_Bljk_HSS_BH_UserArgs_00/00_Final/source/build_tmp/SOURCE/assembly/
/opt/gfx13/bin/amdclang++ -x assembler --target=amdgcn-amd-amdhsa  -mcode-object-version=4 -c -mcpu=gfx1300 -mno-wavefrontsize64 $hipblaslt/build/hss_nn_16x16_test/1_BenchmarkProblems/Cijk_Ailk_Bljk_HSS_BH_UserArgs_00/00_Final/source/build_tmp/SOURCE/assembly/hss_nn_16x16.s -o $hipblaslt/build/hss_nn_16x16_test/1_BenchmarkProblems/Cijk_Ailk_Bljk_HSS_BH_UserArgs_00/00_Final/source/build_tmp/SOURCE/assembly/hss_nn_16x16.o
make -f $hipblaslt/tensilelite/Makefile co TENSILE_OUT=$hipblaslt/build/hss_nn_16x16_test/ ARCH=gfx1300 WAVEFRONTSIZE=-mno-wavefrontsize64
$hipblaslt/build/hss_nn_16x16_test/0_Build/client/tensile_client --config-file $hipblaslt/build/hss_nn_16x16_test/1_BenchmarkProblems/Cijk_Ailk_Bljk_HSS_BH_UserArgs_00/00_Final/build/../source/ClientParameters.ini
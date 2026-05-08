# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_custom_target(example_gemm_mx)

add_example_executable(example_gemm_mx_fp8 gemm_mx_fp8.cpp)
add_example_dependencies(example_gemm_mx example_gemm_mx_fp8)

add_example_executable(example_gemm_mx_fp8_v1 gemm_mx_fp8_v1.cpp)
add_example_dependencies(example_gemm_mx example_gemm_mx_fp8_v1)

add_example_executable(example_gemm_mx_bf8 gemm_mx_bf8.cpp)
add_example_dependencies(example_gemm_mx example_gemm_mx_bf8)

add_example_executable(example_gemm_mx_fp8_bpreshuffle gemm_mx_fp8_bpreshuffle.cpp)
add_example_dependencies(example_gemm_mx example_gemm_mx_fp8_bpreshuffle)

# TODO: Fix RRR
# add_example_executable(example_gemm_mx_fp8_bf8 gemm_mx_fp8_bf8.cpp)
# add_example_dependencies(example_gemm_mx example_gemm_mx_fp8_bf8)

if(NOT GPU_TARGETS MATCHES "gfx13")
#add_example_executable(example_gemm_mx_fp6 gemm_mx_fp6.cpp)
#add_example_dependencies(example_gemm_mx example_gemm_mx_fp6)

#add_example_executable(example_gemm_mx_bf6 gemm_mx_bf6.cpp)
#add_example_dependencies(example_gemm_mx example_gemm_mx_bf6)
endif()

add_example_executable(example_gemm_mx_fp4 gemm_mx_fp4.cpp)
add_example_dependencies(example_gemm_mx example_gemm_mx_fp4)

add_example_executable(example_gemm_mx_fp4_bpreshuffle gemm_mx_fp4_bpreshuffle.cpp)
add_example_dependencies(example_gemm_mx example_gemm_mx_fp4_bpreshuffle)

add_example_executable(example_moe_gemm1_xdl_mx_fp4_bns moe_gemm1_xdl_mx_fp4_bns.cpp)
add_example_dependencies(example_gemm_mx example_moe_gemm1_xdl_mx_fp4_bns)

add_example_executable(example_moe_gemm2_xdl_mx_fp4_bns moe_gemm2_xdl_mx_fp4_bns.cpp)
add_example_dependencies(example_gemm_mx example_moe_gemm2_xdl_mx_fp4_bns)

add_example_executable(example_moe_gemm1_xdl_mx_fp4 moe_gemm1_xdl_mx_fp4.cpp)
add_example_dependencies(example_gemm_mx example_moe_gemm1_xdl_mx_fp4)

add_example_executable(example_moe_gemm2_xdl_mx_fp4 moe_gemm2_xdl_mx_fp4.cpp)
add_example_dependencies(example_gemm_mx example_moe_gemm2_xdl_mx_fp4)

add_example_executable(example_moe_gemm1_xdl_mx_fp4_bpreshuffle moe_gemm1_xdl_mx_fp4_bpreshuffle.cpp)
add_example_dependencies(example_gemm_mx example_moe_gemm1_xdl_mx_fp4_bpreshuffle)

add_example_executable(example_moe_gemm2_xdl_mx_fp4_bpreshuffle moe_gemm2_xdl_mx_fp4_bpreshuffle.cpp)
add_example_dependencies(example_gemm_mx example_moe_gemm2_xdl_mx_fp4_bpreshuffle)

set(FP4_MXGEMM_OPTIONS)
list(APPEND FP4_MXGEMM_OPTIONS "SHELL: -mllvm -greedy-reverse-local-assignment=1 -mllvm --amdgpu-use-amdgpu-trackers=1")

set(FP8_MXGEMM_OPTIONS)
list(APPEND FP8_MXGEMM_OPTIONS "SHELL: -mllvm -greedy-reverse-local-assignment=1")

set(FP6_MXGEMM_OPTIONS)
list(APPEND FP6_MXGEMM_OPTIONS -mavx512f)

# Clang 22+ only: suppress new lifetime suggestion warning promoted by -Werror
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 22)
    list(APPEND FP8_MXGEMM_OPTIONS -Wno-lifetime-safety-intra-tu-suggestions)
    list(APPEND FP6_MXGEMM_OPTIONS -Wno-lifetime-safety-intra-tu-suggestions)
    list(APPEND FP4_MXGEMM_OPTIONS -Wno-lifetime-safety-intra-tu-suggestions)
endif()

example_compile_options(example_gemm_mx_fp4 PRIVATE ${FP4_MXGEMM_OPTIONS})
example_compile_options(example_gemm_mx_fp4_bpreshuffle PRIVATE ${FP4_MXGEMM_OPTIONS})

# mx moe B no-shuffling + scale shuffling
example_compile_options(example_moe_gemm1_xdl_mx_fp4_bns PRIVATE ${FP4_MXGEMM_OPTIONS})
example_compile_options(example_moe_gemm2_xdl_mx_fp4_bns PRIVATE ${FP4_MXGEMM_OPTIONS})

# mx moe B no-shuffling + scale shuffling (async loads)
example_compile_options(example_moe_gemm1_xdl_mx_fp4 PRIVATE ${FP4_MXGEMM_OPTIONS})
example_compile_options(example_moe_gemm2_xdl_mx_fp4 PRIVATE ${FP4_MXGEMM_OPTIONS})

# mx moe B shuffling + scale shuffling (async loads)
example_compile_options(example_moe_gemm1_xdl_mx_fp4_bpreshuffle PRIVATE ${FP4_MXGEMM_OPTIONS})
example_compile_options(example_moe_gemm2_xdl_mx_fp4_bpreshuffle PRIVATE ${FP4_MXGEMM_OPTIONS})

example_compile_options(example_gemm_mx_fp8 PRIVATE ${FP8_MXGEMM_OPTIONS})
example_compile_options(example_gemm_mx_fp8_v1 PRIVATE ${FP8_MXGEMM_OPTIONS})
example_compile_options(example_gemm_mx_fp8_bpreshuffle PRIVATE ${FP8_MXGEMM_OPTIONS})
example_compile_options(example_gemm_mx_bf8 PRIVATE ${FP8_MXGEMM_OPTIONS})

if(NOT GPU_TARGETS MATCHES "gfx13")
#example_compile_options(example_gemm_mx_fp6 PRIVATE ${FP6_MXGEMM_OPTIONS})
#example_compile_options(example_gemm_mx_bf6 PRIVATE ${FP6_MXGEMM_OPTIONS})
endif()
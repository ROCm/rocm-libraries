# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# comp_tdm_v4 (the rocKE gfx1250 mx_gemm kernel). Included from ../CMakeLists.txt under its gfx125
# block, so TEST_MX_GEMM_COMPILE_OPTIONS is set.

# Host-only: the LDS layout and policy against rocKE, the K-major scale helpers and the reference.
add_gtest_executable(test_ck_tile_mx_gemm_tdm_v4_host test_mx_gemm_tdm_v4_host.cpp)
add_gtest_executable(test_ck_tile_mx_gemm_tdm_v4_host_ref test_mx_gemm_tdm_v4_host_ref.cpp)
# Every GPU case compiled for gfx1250 and checked as a problem its instance runs, without a launch.
add_gtest_executable(test_ck_tile_mx_gemm_tdm_v4_cases test_mx_gemm_tdm_v4.cpp)
# On a gfx1250 device: bitwise against the reference on rocKE's sparse check inputs.
add_gtest_executable(test_ck_tile_mx_gemm_tdm_v4 test_mx_gemm_tdm_v4.cpp)
add_gtest_executable(test_ck_tile_mx_gemm_tdm_v4_large_cases test_mx_gemm_tdm_v4_large_cases.cpp)

# rocKE's comgr options (as example 56): the kernel's schedule is in the source, so the machine
# schedulers stay off and the tests check the code the example times.
set(MX_TDM_V4_ROCKE_OPTIONS
    "SHELL:-mllvm -amdgpu-expert-scheduling-mode" "SHELL:-mllvm -enable-misched=false"
    "SHELL:-mllvm -enable-post-misched=false"
    "SHELL:-mllvm -greedy-regclass-priority-trumps-globalness")
foreach(test_target test_ck_tile_mx_gemm_tdm_v4_host test_ck_tile_mx_gemm_tdm_v4_host_ref
                    test_ck_tile_mx_gemm_tdm_v4_cases test_ck_tile_mx_gemm_tdm_v4
                    test_ck_tile_mx_gemm_tdm_v4_large_cases)
    if(TARGET ${test_target})
        target_compile_options(${test_target} PRIVATE ${TEST_MX_GEMM_COMPILE_OPTIONS}
                                                      ${MX_TDM_V4_ROCKE_OPTIONS})
        ck_tile_restrict_archs(${test_target} "^gfx125")
    endif()
endforeach()
if(TARGET test_ck_tile_mx_gemm_tdm_v4_cases)
    target_compile_definitions(test_ck_tile_mx_gemm_tdm_v4_cases PRIVATE MX_TDM_V4_DRY_RUN)
endif()

# Instances the pipeline must reject at compile time (test_mx_gemm_tdm_v4_reject.cpp): half PLR 'b'
# without local prefetch, 8-bit on the 32x16 atom, a D store wider than the N repeat, GL2 distance 1 and a
# 32-workgroup cluster.
set(MX_TDM_V4_REJECT_1 "InvalidTuning<ck_tile::MxTdmV4Reason::HalfPlrB>")
set(MX_TDM_V4_REJECT_2 "InvalidTuning<ck_tile::MxTdmV4Reason::Mma>")
set(MX_TDM_V4_REJECT_3 "the D store width must be")
set(MX_TDM_V4_REJECT_4 "InvalidTuning<ck_tile::MxTdmV4Reason::Gl2Distance>")
set(MX_TDM_V4_REJECT_5 "InvalidTuning<ck_tile::MxTdmV4Reason::ClusterSize>")
foreach(reject_case RANGE 1 5)
    add_test(NAME test_ck_tile_mx_gemm_tdm_v4_reject_${reject_case}
             COMMAND ${CMAKE_CXX_COMPILER} -x hip --offload-arch=gfx1250 -std=c++20 -fsyntax-only
                     -DCK_USE_GFX1250 -DMX_TDM_V4_REJECT=${reject_case}
                     -I${PROJECT_SOURCE_DIR}/include -I${CMAKE_CURRENT_SOURCE_DIR}
                     ${CMAKE_CURRENT_SOURCE_DIR}/test_mx_gemm_tdm_v4_reject.cpp)
    set_tests_properties(test_ck_tile_mx_gemm_tdm_v4_reject_${reject_case} PROPERTIES
                         PASS_REGULAR_EXPRESSION "${MX_TDM_V4_REJECT_${reject_case}}")
endforeach()

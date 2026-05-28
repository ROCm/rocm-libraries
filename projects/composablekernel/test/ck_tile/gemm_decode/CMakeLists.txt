# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_gemm_decode test_gemm_decode.cpp)

    if(CK_USE_OCP_FP8)
        target_compile_options(test_ck_tile_gemm_decode PRIVATE -DCK_TILE_USE_OCP_FP8)
    endif()

    # Standalone microbench for the XCD-aware workgroup swizzle. Not a
    # gtest target on purpose: it just runs a sweep and prints numbers.
    add_executable(bench_gemm_decode_chiplet_swizzle bench_chiplet_swizzle.cpp)
    target_link_libraries(bench_gemm_decode_chiplet_swizzle PRIVATE utility)
    set_target_properties(bench_gemm_decode_chiplet_swizzle PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
    set_source_files_properties(bench_chiplet_swizzle.cpp PROPERTIES LANGUAGE HIP)
endif()

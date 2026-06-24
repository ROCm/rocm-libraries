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

    # R1: M=1..16 best-config sweep, emits CSV for the FlyDSL comparison.
    add_executable(bench_gemm_decode_msweep bench_msweep.cpp)
    target_link_libraries(bench_gemm_decode_msweep PRIVATE utility)
    set_target_properties(bench_gemm_decode_msweep PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
    set_source_files_properties(bench_msweep.cpp PROPERTIES LANGUAGE HIP)

    # Single-config launcher for the M<=8 headroom diagnosis: fires one fixed
    # config in a loop so rocprofv3 can attach counters to a clean dispatch.
    add_executable(prof_gemm_decode_one prof_one.cpp)
    target_link_libraries(prof_gemm_decode_one PRIVATE utility)
    # Enable the OCP e4m3 FP8 cvt builtins for the FP8 prof cases (m1n7168*).
    if(CK_USE_OCP_FP8)
        target_compile_options(prof_gemm_decode_one PRIVATE -DCK_TILE_USE_OCP_FP8)
    endif()
    set_target_properties(prof_gemm_decode_one PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
    set_source_files_properties(prof_one.cpp PROPERTIES LANGUAGE HIP)

    # FP8 PerTensor M=1..8 best-config sweep (FP8 analog of bench_msweep),
    # emits CSV for the FP8 per-tensor head-to-head. Only built with OCP FP8
    # (the dot2 K-loop packs FP8x4 -> 2*BF16x2 on the e4m3 path).
    if(CK_USE_OCP_FP8)
        add_executable(bench_gemm_decode_msweep_fp8 bench_msweep_fp8.cpp)
        target_link_libraries(bench_gemm_decode_msweep_fp8 PRIVATE utility)
        target_compile_options(bench_gemm_decode_msweep_fp8 PRIVATE -DCK_TILE_USE_OCP_FP8)
        set_target_properties(bench_gemm_decode_msweep_fp8 PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
        set_source_files_properties(bench_msweep_fp8.cpp PROPERTIES LANGUAGE HIP)
    endif()
endif()

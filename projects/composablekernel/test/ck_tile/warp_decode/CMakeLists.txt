# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_warp_decode test_warp_decode.cpp)

    if(CK_USE_OCP_FP8)
        target_compile_options(test_ck_tile_warp_decode PRIVATE -DCK_TILE_USE_OCP_FP8)
    endif()

    set_source_files_properties(bench_warp_decode.cpp PROPERTIES LANGUAGE HIP)
    add_executable(bench_ck_tile_warp_decode bench_warp_decode.cpp)
    set_property(TARGET bench_ck_tile_warp_decode
                 PROPERTY HIP_ARCHITECTURES ${SUPPORTED_GPU_TARGETS})
    target_compile_options(bench_ck_tile_warp_decode PRIVATE
        -Wno-global-constructors
        -Wno-undef
        -Wno-float-equal)

    if(CK_USE_OCP_FP8)
        target_compile_options(bench_ck_tile_warp_decode PRIVATE -DCK_TILE_USE_OCP_FP8)
    endif()
endif()

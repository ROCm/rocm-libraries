# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Compile-only, for every gfx125 variant the compiler supports, whatever GPU_TARGETS is: most CI
# configurations have no gfx125 target. gfx1250-strict lacks the FP8/BF8 16x16x128 WMMA
# instructions, so a wrapper that emits them there fails to compile (ROCM-35020).
rocm_check_target_ids(CK_GFX125_TARGETS TARGETS gfx1250 gfx1250-strict gfx1251)
if(CK_GFX125_TARGETS)
    set_source_files_properties(test_wmma_f8_16x16x128_gfx125.cpp PROPERTIES LANGUAGE HIP)
    add_library(test_ck_wmma_f8_16x16x128_gfx125 OBJECT test_wmma_f8_16x16x128_gfx125.cpp)
    set_property(TARGET test_ck_wmma_f8_16x16x128_gfx125
        PROPERTY HIP_ARCHITECTURES ${CK_GFX125_TARGETS})
    add_dependencies(tests test_ck_wmma_f8_16x16x128_gfx125)
    add_dependencies(smoke test_ck_wmma_f8_16x16x128_gfx125)
endif()

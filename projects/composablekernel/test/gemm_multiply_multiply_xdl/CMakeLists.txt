# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Large M shapes need an Instinct GPU: gfx942 / gfx950 only (not gfx12, the cards are too small).
if(GPU_TARGETS MATCHES "gfx94|gfx95")
    add_gtest_executable(test_gemm_multiply_multiply_xdl_large_tensor test_gemm_multiply_multiply_xdl_large_tensor.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_gemm_multiply_multiply_xdl_large_tensor PRIVATE utility)
    endif()
endif()

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
add_gtest_executable(test_wmma_op wmma_op.cpp)
target_link_libraries(test_wmma_op PRIVATE utility)

# The FP8/BF8 16x16x128 tests again at -O0: their gfx1250-strict fallback once gave C instead of
# A*B + C only at -O0 (ROCM-35020). The later -O0 overrides the build type's optimization level.
add_gtest_executable(test_wmma_op_f8_16x16x128_o0 wmma_op.cpp)
if(result EQUAL 0)
    target_link_libraries(test_wmma_op_f8_16x16x128_o0 PRIVATE utility)
    target_compile_options(test_wmma_op_f8_16x16x128_o0 PRIVATE -O0)
    set_tests_properties(test_wmma_op_f8_16x16x128_o0 PROPERTIES ENVIRONMENT "GTEST_FILTER=*16x16x128*")
endif()

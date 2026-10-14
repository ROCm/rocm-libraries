# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(mosaic_tests
    test_int_const.cpp
    test_int_tuple.cpp
    test_stride.cpp
    test_layout.cpp
    test_algebra.cpp
    test_tensor.cpp
    test_algorithm.cpp
    test_swizzle.cpp
    test_partition.cpp
)

target_link_libraries(mosaic_tests
    PRIVATE
        mosaic
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(mosaic_tests PROPERTIES TIMEOUT 60)

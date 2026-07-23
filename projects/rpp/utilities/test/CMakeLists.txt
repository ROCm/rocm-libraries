cmake_minimum_required(VERSION 3.14)

project(rpp-test CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROCm path (only used to locate the installed rpp package by default)
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "ROCm installation path")
else()
    set(ROCM_PATH /opt/rocm CACHE PATH "ROCm installation path")
endif()
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)

# Standalone build finds an installed rpp; in-tree build reuses the existing target.
if(NOT TARGET rpp)
    find_package(rpp 3.0.0 REQUIRED)
    set(RPP_TEST_LINK_LIB rpp::rpp)
else()
    set(RPP_TEST_LINK_LIB rpp)
endif()

include(cmake/gtest.cmake)

enable_testing()
include(GoogleTest)
include(CTest)

add_executable(rpp_tests
    src/tests/core/handle_test.cpp
    src/tests/core/descriptor_test.cpp
    src/tests/image/color/brightness_test.cpp
    src/tests/image/color/exposure_test.cpp
    src/tests/image/color/blend_test.cpp
    src/tests/image/color/contrast_test.cpp
    src/tests/image/color/gamma_correction_test.cpp
    src/tests/image/color/color_cast_test.cpp
    src/tests/image/color/color_temperature_test.cpp
    src/tests/image/color/hue_test.cpp
    src/tests/image/color/lut_test.cpp
    src/tests/image/color/saturation_test.cpp
    src/tests/image/color/color_twist_test.cpp
    src/tests/image/arithmetic/magnitude_test.cpp
    src/tests/image/effects/solarize_test.cpp
    src/tests/image/effects/channel_dropout_test.cpp
    src/tests/image/bitwise/bitwise_and_test.cpp
    src/tests/image/bitwise/bitwise_or_test.cpp
    src/tests/image/bitwise/bitwise_xor_test.cpp
    src/tests/image/bitwise/bitwise_not_test.cpp
    src/tests/image/data_exchange/copy_test.cpp
    src/tests/image/data_exchange/channel_permute_test.cpp
    src/tests/image/geometric/crop_test.cpp
    src/tests/image/geometric/resize_test.cpp
    src/tests/image/geometric/warp_affine_test.cpp
    src/tests/image/geometric/warp_perspective_test.cpp
    src/tests/image/geometric/rotate_test.cpp
    src/tests/image/statistical/tensor_sum_test.cpp
    src/tests/image/statistical/threshold_test.cpp
    src/tests/image/statistical/tensor_min_test.cpp
    src/tests/image/statistical/tensor_max_test.cpp
    src/tests/image/statistical/tensor_mean_test.cpp
    src/tests/image/statistical/tensor_stddev_test.cpp
)
target_include_directories(rpp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(rpp_INCLUDE_DIR)
    target_include_directories(rpp_tests PRIVATE ${rpp_INCLUDE_DIR}/rpp)
endif()
target_link_libraries(rpp_tests PRIVATE ${RPP_TEST_LINK_LIB} GTest::gtest_main)

if(rpp_BACKEND_TYPE STREQUAL "HIP" OR BACKEND STREQUAL "HIP")
    target_compile_definitions(rpp_tests PRIVATE RPP_TEST_HAVE_HIP=1)
endif()

gtest_discover_tests(rpp_tests)

###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 # THE SOFTWARE.
 #
 ###############################################################################

# Check whether building within hiptensor context
if( CMAKE_PROJECT_NAME STREQUAL "hiptensor" )
    add_hiptensor_sample(simple_bilinear_contraction_bf16_bf16_bf16_bf16_compute_bf16 simple_bilinear_contraction_bf16_bf16_bf16_bf16_compute_bf16.cpp)
    add_hiptensor_sample(simple_bilinear_contraction_f16_f16_f16_f16_compute_f16 simple_bilinear_contraction_f16_f16_f16_f16_compute_f16.cpp)
    add_hiptensor_sample(simple_bilinear_contraction_f32_f32_f32_f32_compute_bf16 simple_bilinear_contraction_f32_f32_f32_f32_compute_bf16.cpp)
    add_hiptensor_sample(simple_bilinear_contraction_f32_f32_f32_f32_compute_f16 simple_bilinear_contraction_f32_f32_f32_f32_compute_f16.cpp)
    add_hiptensor_sample(simple_bilinear_contraction_f32_f32_f32_f32_compute_f32 simple_bilinear_contraction_f32_f32_f32_f32_compute_f32.cpp)
    add_hiptensor_sample(simple_bilinear_contraction_cf32_cf32_cf32_cf32_compute_cf32 simple_bilinear_contraction_cf32_cf32_cf32_cf32_compute_cf32.cpp)
    add_hiptensor_sample(simple_bilinear_contraction_f64_f64_f64_f64_compute_f32 simple_bilinear_contraction_f64_f64_f64_f64_compute_f32.cpp)
    add_hiptensor_sample(simple_bilinear_contraction_f64_f64_f64_f64_compute_f64 simple_bilinear_contraction_f64_f64_f64_f64_compute_f64.cpp)
    add_hiptensor_sample(simple_scale_contraction_bf16_bf16_bf16_compute_bf16 simple_scale_contraction_bf16_bf16_bf16_compute_bf16.cpp)
    add_hiptensor_sample(simple_scale_contraction_f16_f16_f16_compute_f16 simple_scale_contraction_f16_f16_f16_compute_f16.cpp)
    add_hiptensor_sample(simple_scale_contraction_f32_f32_f32_compute_bf16 simple_scale_contraction_f32_f32_f32_compute_bf16.cpp)
    add_hiptensor_sample(simple_scale_contraction_f32_f32_f32_compute_f16 simple_scale_contraction_f32_f32_f32_compute_f16.cpp)
    add_hiptensor_sample(simple_scale_contraction_f32_f32_f32_compute_f32 simple_scale_contraction_f32_f32_f32_compute_f32.cpp)
    add_hiptensor_sample(simple_scale_contraction_cf32_cf32_cf32_compute_cf32 simple_scale_contraction_cf32_cf32_cf32_compute_cf32.cpp)
    add_hiptensor_sample(simple_scale_contraction_f64_f64_f64_compute_f32 simple_scale_contraction_f64_f64_f64_compute_f32.cpp)
    add_hiptensor_sample(simple_scale_contraction_f64_f64_f64_compute_f64 simple_scale_contraction_f64_f64_f64_compute_f64.cpp)

# If building hipTensor samples as a standalone Cmake project
else()
    add_executable(simple_bilinear_contraction_bf16_bf16_bf16_bf16_compute_bf16 simple_bilinear_contraction_bf16_bf16_bf16_bf16_compute_bf16.cpp)
    target_link_libraries(simple_bilinear_contraction_bf16_bf16_bf16_bf16_compute_bf16 PRIVATE hiptensor::hiptensor)

    add_executable(simple_bilinear_contraction_f16_f16_f16_f16_compute_f16 simple_bilinear_contraction_f16_f16_f16_f16_compute_f16.cpp)
    target_link_libraries(simple_bilinear_contraction_f16_f16_f16_f16_compute_f16 PRIVATE hiptensor::hiptensor)

    add_executable(simple_bilinear_contraction_f32_f32_f32_f32_compute_bf16 simple_bilinear_contraction_f32_f32_f32_f32_compute_bf16.cpp)
    target_link_libraries(simple_bilinear_contraction_f32_f32_f32_f32_compute_bf16 PRIVATE hiptensor::hiptensor)

    add_executable(simple_bilinear_contraction_f32_f32_f32_f32_compute_f16 simple_bilinear_contraction_f32_f32_f32_f32_compute_f16.cpp)
    target_link_libraries(simple_bilinear_contraction_f32_f32_f32_f32_compute_f16 PRIVATE hiptensor::hiptensor)

    add_executable(simple_bilinear_contraction_f32_f32_f32_f32_compute_f32 simple_bilinear_contraction_f32_f32_f32_f32_compute_f32.cpp)
    target_link_libraries(simple_bilinear_contraction_f32_f32_f32_f32_compute_f32 PRIVATE hiptensor::hiptensor)

    add_executable(simple_bilinear_contraction_f64_f64_f64_f64_compute_f32 simple_bilinear_contraction_f64_f64_f64_f64_compute_f32.cpp)
    target_link_libraries(simple_bilinear_contraction_f64_f64_f64_f64_compute_f32 PRIVATE hiptensor::hiptensor)

    add_executable(simple_bilinear_contraction_f64_f64_f64_f64_compute_f64 simple_bilinear_contraction_f64_f64_f64_f64_compute_f64.cpp)
    target_link_libraries(simple_bilinear_contraction_f64_f64_f64_f64_compute_f64 PRIVATE hiptensor::hiptensor)

    add_executable(simple_scale_contraction_bf16_bf16_bf16_compute_bf16 simple_scale_contraction_bf16_bf16_bf16_compute_bf16.cpp)
    target_link_libraries(simple_scale_contraction_bf16_bf16_bf16_compute_bf16 PRIVATE hiptensor::hiptensor)

    add_executable(simple_scale_contraction_f16_f16_f16_compute_f16 simple_scale_contraction_f16_f16_f16_compute_f16.cpp)
    target_link_libraries(simple_scale_contraction_f16_f16_f16_compute_f16 PRIVATE hiptensor::hiptensor)

    add_executable(simple_scale_contraction_f32_f32_f32_compute_bf16 simple_scale_contraction_f32_f32_f32_compute_bf16.cpp)
    target_link_libraries(simple_scale_contraction_f32_f32_f32_compute_bf16 PRIVATE hiptensor::hiptensor)

    add_executable(simple_scale_contraction_f32_f32_f32_compute_f16 simple_scale_contraction_f32_f32_f32_compute_f16.cpp)
    target_link_libraries(simple_scale_contraction_f32_f32_f32_compute_f16 PRIVATE hiptensor::hiptensor)

    add_executable(simple_scale_contraction_f32_f32_f32_compute_f32 simple_scale_contraction_f32_f32_f32_compute_f32.cpp)
    target_link_libraries(simple_scale_contraction_f32_f32_f32_compute_f32 PRIVATE hiptensor::hiptensor)

    add_executable(simple_scale_contraction_f64_f64_f64_compute_f32 simple_scale_contraction_f64_f64_f64_compute_f32.cpp)
    target_link_libraries(simple_scale_contraction_f64_f64_f64_compute_f32 PRIVATE hiptensor::hiptensor)

    add_executable(simple_scale_contraction_f64_f64_f64_compute_f64 simple_scale_contraction_f64_f64_f64_compute_f64.cpp)
    target_link_libraries(simple_scale_contraction_f64_f64_f64_compute_f64 PRIVATE hiptensor::hiptensor)

endif()

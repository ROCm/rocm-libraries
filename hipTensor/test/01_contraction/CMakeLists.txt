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
 set(ContractionCommonSources ${HIPTENSOR_COMMON_TEST_SOURCES}
 ${CMAKE_CURRENT_SOURCE_DIR}/contraction_resource.cpp
 ${CMAKE_CURRENT_SOURCE_DIR}/contraction_test.cpp)

# Bilinear M1N1K1 tests
set (BilinearContractionTestSources ${ContractionCommonSources}
        ${CMAKE_CURRENT_SOURCE_DIR}/bilinear_contraction_test.cpp)
set (BilinearContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/bilinear_test_params_rank1.yaml)
add_hiptensor_test(bilinear_contraction_test_m1n1k1 ${BilinearContractionTestConfig}  ${BilinearContractionTestSources})

# Bilinear M2N2K2 tests
set (BilinearContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/bilinear_test_params_rank2.yaml)
add_hiptensor_test(bilinear_contraction_test_m2n2k2 ${BilinearContractionTestConfig}  ${BilinearContractionTestSources})

# Bilinear M3N3K3 tests
set (BilinearContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/bilinear_test_params_rank3.yaml)
add_hiptensor_test(bilinear_contraction_test_m3n3k3 ${BilinearContractionTestConfig}  ${BilinearContractionTestSources})

# Bilinear M4N4K4 tests
set (BilinearContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/bilinear_test_params_rank4.yaml)
add_hiptensor_test(bilinear_contraction_test_m4n4k4 ${BilinearContractionTestConfig}  ${BilinearContractionTestSources})

# Bilinear M5N5K5 tests
set (BilinearContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/bilinear_test_params_rank5.yaml)
add_hiptensor_test(bilinear_contraction_test_m5n5k5 ${BilinearContractionTestConfig}  ${BilinearContractionTestSources})

# Bilinear M6N6K6 tests
set (BilinearContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/bilinear_test_params_rank6.yaml)
add_hiptensor_test(bilinear_contraction_test_m6n6k6 ${BilinearContractionTestConfig}  ${BilinearContractionTestSources})

# Complex Bilinear M1N1K1 tests
set (ComplexBilinearContractionTestSources ${ContractionCommonSources}
${CMAKE_CURRENT_SOURCE_DIR}/complex_bilinear_contraction_test.cpp)
set (ComplexBilinearContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_bilinear_test_params_rank1.yaml)
add_hiptensor_test(complex_bilinear_contraction_test_m1n1k1 ${ComplexBilinearContractionTestConfig}  ${ComplexBilinearContractionTestSources})

# Complex Bilinear M2N2K2 tests
set (ComplexBilinearContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_bilinear_test_params_rank2.yaml)
add_hiptensor_test(complex_bilinear_contraction_test_m2n2k2 ${ComplexBilinearContractionTestConfig}  ${ComplexBilinearContractionTestSources})

# Complex Bilinear M3N3K3 tests
set (ComplexBilinearContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_bilinear_test_params_rank3.yaml)
add_hiptensor_test(complex_bilinear_contraction_test_m3n3k3 ${ComplexBilinearContractionTestConfig}  ${ComplexBilinearContractionTestSources})

# Complex Bilinear M4N4K4 tests
set (ComplexBilinearContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_bilinear_test_params_rank4.yaml)
add_hiptensor_test(complex_bilinear_contraction_test_m4n4k4 ${ComplexBilinearContractionTestConfig}  ${ComplexBilinearContractionTestSources})

# Complex Bilinear M5N5K5 tests
set (ComplexBilinearContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_bilinear_test_params_rank5.yaml)
add_hiptensor_test(complex_bilinear_contraction_test_m5n5k5 ${ComplexBilinearContractionTestConfig}  ${ComplexBilinearContractionTestSources})

# Complex Bilinear M6N6K6 tests
set (ComplexBilinearContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_bilinear_test_params_rank6.yaml)
add_hiptensor_test(complex_bilinear_contraction_test_m6n6k6 ${ComplexBilinearContractionTestConfig}  ${ComplexBilinearContractionTestSources})

# Scale M2N2K2 tests
set (ScaleContractionTestSources ${ContractionCommonSources}
     ${CMAKE_CURRENT_SOURCE_DIR}/scale_contraction_test.cpp)
set (ScaleContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/scale_test_params_rank1.yaml)
add_hiptensor_test(scale_contraction_test_m1n1k1 ${ScaleContractionTestConfig} ${ScaleContractionTestSources})

# Scale M2N2K2 tests
set (ScaleContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/scale_test_params_rank2.yaml)
add_hiptensor_test(scale_contraction_test_m2n2k2 ${ScaleContractionTestConfig} ${ScaleContractionTestSources})

# Scale M3N3K3 tests
set (ScaleContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/scale_test_params_rank3.yaml)
add_hiptensor_test(scale_contraction_test_m3n3k3 ${ScaleContractionTestConfig} ${ScaleContractionTestSources})

# Scale M4N4K4 tests
set (ScaleContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/scale_test_params_rank4.yaml)
add_hiptensor_test(scale_contraction_test_m4n4k4 ${ScaleContractionTestConfig} ${ScaleContractionTestSources})

# Scale M5N5K5 tests
set (ScaleContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/scale_test_params_rank5.yaml)
add_hiptensor_test(scale_contraction_test_m5n5k5 ${ScaleContractionTestConfig} ${ScaleContractionTestSources})

# Scale M6N6K6 tests
set (ScaleContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/scale_test_params_rank6.yaml)
add_hiptensor_test(scale_contraction_test_m6n6k6 ${ScaleContractionTestConfig} ${ScaleContractionTestSources})

# Complex Scale M1N1K1 tests
set (ComplexScaleContractionTestSources ${ContractionCommonSources}
${CMAKE_CURRENT_SOURCE_DIR}/complex_scale_contraction_test.cpp)
set (ComplexScaleContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_scale_test_params_rank1.yaml)
add_hiptensor_test(complex_scale_contraction_test_m1n1k1 ${ComplexScaleContractionTestConfig}  ${ComplexScaleContractionTestSources})

# Complex Scale M2N2K2 tests
set (ComplexScaleContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_scale_test_params_rank2.yaml)
add_hiptensor_test(complex_scale_contraction_test_m2n2k2 ${ComplexScaleContractionTestConfig}  ${ComplexScaleContractionTestSources})

# Complex Scale M3N3K3 tests
set (ComplexScaleContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_scale_test_params_rank3.yaml)
add_hiptensor_test(complex_scale_contraction_test_m3n3k3 ${ComplexScaleContractionTestConfig}  ${ComplexScaleContractionTestSources})

# Complex Scale M4N4K4 tests
set (ComplexScaleContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_scale_test_params_rank4.yaml)
add_hiptensor_test(complex_scale_contraction_test_m4n4k4 ${ComplexScaleContractionTestConfig}  ${ComplexScaleContractionTestSources})

# Complex Scale M5N5K5 tests
set (ComplexScaleContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_scale_test_params_rank5.yaml)
add_hiptensor_test(complex_scale_contraction_test_m5n5k5 ${ComplexScaleContractionTestConfig}  ${ComplexScaleContractionTestSources})

# Complex Scale M6N6K6 tests
set (ComplexScaleContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_scale_test_params_rank6.yaml)
add_hiptensor_test(complex_scale_contraction_test_m6n6k6 ${ComplexScaleContractionTestConfig}  ${ComplexScaleContractionTestSources})


# Contraction mode tests
set (ContractionModeTestSources ${ContractionCommonSources}
${CMAKE_CURRENT_SOURCE_DIR}/contraction_mode_test.cpp)
set (ContractionModeTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/mode_test_params.yaml)
add_hiptensor_test(contraction_mode_test ${ContractionModeTestConfig}  ${ContractionModeTestSources})

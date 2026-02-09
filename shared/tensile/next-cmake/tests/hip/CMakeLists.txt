# ##############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ##############################################################################

set(_CMAKE_CURRENT_SOURCE_DIR
    "${TENSILE_SOURCE_TREE}/../../HostLibraryTests/hip")

target_sources(
  tensile-tests
  PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/CopyTensor_test.cpp"
          # "${_CMAKE_CURRENT_SOURCE_DIR}/RunGEMMKernelTileSelection_test.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/HipSolutionAdapter_test.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/RunGEMMKernel_test.cpp"
)

set(tcl_command Tensile/bin/TensileCreateLibrary
                --no-merge-files --no-short-file-names --verbose=0 --code-object-version=default
                --cxx-compiler=amdclang++ --cmake-cxx-compiler="${CMAKE_CXX_COMPILER}" --jobs=8)

set(lite_command ${tcl_command} --embed-library=test_kernels_lite --embed-library-key=kernels_lite
                 "${TENSILE_CONFIG_DIR}/lite_configs" "${TENSILE_TEST_DATA_DIR}/test_kernels_lite" HIP)

add_custom_command(
  COMMENT "Generating ${TENSILE_TEST_DATA_DIR}/test_kernels_lite/library/test_kernels_lite.cpp"
  COMMAND ${lite_command}
  OUTPUT "${TENSILE_TEST_DATA_DIR}/test_kernels_lite/library/test_kernels_lite.cpp"
  WORKING_DIRECTORY "${TENSILE_SOURCE_TREE}/../..")

set(lite2_command ${tcl_command} --embed-library=test_kernels_lite_2 --embed-library-key=kernels_lite_2
                "${TENSILE_CONFIG_DIR}/lite_configs" "${TENSILE_TEST_DATA_DIR}/test_kernels_lite_2" HIP)

add_custom_command(
  COMMENT "Generating ${TENSILE_TEST_DATA_DIR}/test_kernels_lite_2/library/test_kernels_lite_2.cpp"
  COMMAND ${lite2_command}
  OUTPUT "${TENSILE_TEST_DATA_DIR}/test_kernels_lite_2/library/test_kernels_lite_2.cpp"
  WORKING_DIRECTORY "${TENSILE_SOURCE_TREE}/../..")

set(lite_mixed_command ${tcl_command} --embed-library=test_kernels_lite_mixed --embed-library-key=kernels_lite_mixed
                "${TENSILE_CONFIG_DIR}/lite_configs_mixed" "${TENSILE_TEST_DATA_DIR}/test_kernels_lite_mixed" HIP)

add_custom_command(
  COMMENT "Generating ${TENSILE_TEST_DATA_DIR}/test_kernels_lite_mixed/library/test_kernels_lite_mixed.cpp"
  COMMAND ${lite_mixed_command}
  OUTPUT "${TENSILE_TEST_DATA_DIR}/test_kernels_lite_mixed/library/test_kernels_lite_mixed.cpp"
  WORKING_DIRECTORY "${TENSILE_SOURCE_TREE}/../..")

set(tile_aware_command ${tcl_command} --embed-library=test_tile_aware_selection --embed-library-key=tile_aware_selection
                       "${TENSILE_CONFIG_DIR}/tile_aware_selection" "${TENSILE_TEST_DATA_DIR}/test_tile_aware_selection" HIP)

add_custom_command(
  COMMENT "Generating ${TENSILE_TEST_DATA_DIR}/test_tile_aware_selection/library/test_tile_aware_selection.cpp"
  COMMAND ${tile_aware_command}
  OUTPUT "${TENSILE_TEST_DATA_DIR}/test_tile_aware_selection/library/test_tile_aware_selection.cpp"
  WORKING_DIRECTORY "${TENSILE_SOURCE_TREE}/../..")

add_library(test-device-kernels OBJECT
  "${TENSILE_TEST_DATA_DIR}/test_kernels_lite/library/test_kernels_lite.cpp"
  "${TENSILE_TEST_DATA_DIR}/test_kernels_lite_2/library/test_kernels_lite_2.cpp"
  "${TENSILE_TEST_DATA_DIR}/test_kernels_lite_mixed/library/test_kernels_lite_mixed.cpp"
  "${TENSILE_TEST_DATA_DIR}/test_tile_aware_selection/library/test_tile_aware_selection.cpp")
target_link_libraries(test-device-kernels PRIVATE tensile-host)
target_compile_definitions(test-device-kernels PRIVATE __HIP_HCC_COMPAT_MODE__=1)
target_compile_features(test-device-kernels PUBLIC cxx_std_17)

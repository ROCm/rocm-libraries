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

if(TENSILE_ENABLE_LLVM)
  add_subdirectory(llvm)
endif()

add_subdirectory(hip)

if(TENSILE_ENABLE_CLIENT)
  add_subdirectory(client)
endif()

set(_CMAKE_CURRENT_SOURCE_DIR "${TENSILE_SOURCE_TREE}/../../HostLibraryTests")

target_sources(
  tensile-tests
  PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/ContractionFitness_test.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/EmbeddedData_test.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/DecisionTree_test.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/ContractionProblem_test.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/test.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/ContractionSelectionLibrary_test.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/TensorDescriptor_test.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/ContractionLibraryLoading_test.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/Utils_test.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/ProjectedPerformance_test.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/DataTypes_test.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/MultipleSolutionsPerSize_test.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/TestData_test.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/CachingLibrary_test.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/KernelArguments_test.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/Predicates_test.cpp")

target_sources(
  tensile-test-lib
  PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/testlib/source/TestData.cpp")

if(TENSILE_ENABLE_LLVM)
  set(SOLUTION_LIBRARY_FILES
      "${_CMAKE_CURRENT_SOURCE_DIR}/configs/SolutionLibraries/KernelsLite.yaml"
      "${_CMAKE_CURRENT_SOURCE_DIR}/configs/SolutionLibraries/KernelsLiteMixed.yaml"
      "${_CMAKE_CURRENT_SOURCE_DIR}/configs/SolutionLibraries/KernelsLiteNavi.yaml"
      "${_CMAKE_CURRENT_SOURCE_DIR}/configs/SolutionLibraries/KernelsTileLite.yaml"
      "${_CMAKE_CURRENT_SOURCE_DIR}/configs/SolutionLibraries/SampleTensileKernels.yaml"
      "${_CMAKE_CURRENT_SOURCE_DIR}/configs/SolutionLibraries/rocBLAS_Full.yaml")
endif()

if(TENSILE_ENABLE_MSGPACK)
  set(SOLUTION_LIBRARY_FILES
      ${SOLUTION_LIBRARY_FILES}
      "${_CMAKE_CURRENT_SOURCE_DIR}/configs/SolutionLibraries/KernelsLite.dat"
      "${_CMAKE_CURRENT_SOURCE_DIR}/configs/SolutionLibraries/KernelsLiteMixed.dat"
      "${_CMAKE_CURRENT_SOURCE_DIR}/configs/SolutionLibraries/KernelsLiteNavi.dat"
      "${_CMAKE_CURRENT_SOURCE_DIR}/configs/SolutionLibraries/KernelsTileLite.dat"
      "${_CMAKE_CURRENT_SOURCE_DIR}/configs/SolutionLibraries/SampleTensileKernels.dat"
      "${_CMAKE_CURRENT_SOURCE_DIR}/configs/SolutionLibraries/rocBLAS_Full.dat")
endif()

foreach(FILE ${SOLUTION_LIBRARY_FILES})
  execute_process(COMMAND gzip --keep --decompress --force "${FILE}.gz")
  file(COPY ${FILE} DESTINATION "${TENSILE_TEST_DATA_DIR}")
endforeach(FILE)

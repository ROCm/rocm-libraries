# ##############################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

set(_CMAKE_CURRENT_BINARY_DIR "${TENSILE_SOURCE_TREE}/lib/source")

target_sources(
  tensile-host
  PRIVATE "${_CMAKE_CURRENT_BINARY_DIR}/AMDGPU.cpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/ArithmeticUnitTypes.cpp"
          #"${_CMAKE_CURRENT_BINARY_DIR}/Contractions.cpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/ContractionProblem.cpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/ContractionSolution.cpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/DataTypes.cpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/Debug.cpp"
          #"${_CMAKE_CURRENT_BINARY_DIR}/EmbeddedData.cpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/EmbeddedLibrary.cpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/KernelArguments.cpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/KernelLanguageTypes.cpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/MLFeatures.cpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/PerformanceMetricTypes.cpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/ScalarValueTypes.cpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/TensorDescriptor.cpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/TensorOps.cpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/Tensile.cpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/UserDrivenTuningParser.cpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/Utils.cpp")

if(TENSILE_ENABLE_LLVM)
  add_subdirectory(llvm)
endif()

if(TENSILE_ENABLE_MSGPACK)
  add_subdirectory(msgpack)
endif()

add_subdirectory(hip)

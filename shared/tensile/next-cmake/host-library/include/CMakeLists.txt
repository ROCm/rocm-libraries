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

if(TENSILE_ENABLE_LLVM)
  add_subdirectory(llvm)
endif()

if(TENSILE_ENABLE_MSGPACK)
  add_subdirectory(msgpack)
endif()

add_subdirectory(hip)

add_subdirectory(Serialization)

set(_CMAKE_CURRENT_BINARY_DIR "${TENSILE_SOURCE_TREE}/lib/include/Tensile")

target_sources(
  tensile-host
  PUBLIC $<BUILD_INTERFACE:${_CMAKE_CURRENT_BINARY_DIR}/Contractions.hpp>
         $<BUILD_INTERFACE:${_CMAKE_CURRENT_BINARY_DIR}/EmbeddedLibrary.hpp>
         $<BUILD_INTERFACE:${_CMAKE_CURRENT_BINARY_DIR}/PlaceholderLibrary.hpp>
         $<BUILD_INTERFACE:${_CMAKE_CURRENT_BINARY_DIR}/MasterSolutionLibrary.hpp>
         $<BUILD_INTERFACE:${_CMAKE_CURRENT_BINARY_DIR}/Tensile.hpp>
         $<BUILD_INTERFACE:${_CMAKE_CURRENT_BINARY_DIR}/TensorDescriptor.hpp>
         $<BUILD_INTERFACE:${_CMAKE_CURRENT_BINARY_DIR}/Utils.hpp>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/Tensile/Contractions.hpp>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/Tensile/EmbeddedLibrary.hpp>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/Tensile/PlaceholderLibrary.hpp>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/Tensile/MasterSolutionLibrary.hpp>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/Tensile/Tensile.hpp>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/Tensile/TensorDescriptor.hpp>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/Tensile/Utils.hpp>
  PRIVATE "${_CMAKE_CURRENT_BINARY_DIR}/GranularitySelectionLibrary.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/ScalarValueTypes.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/DataTypes_Half.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/KernelLanguageTypes.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/DataTypes_BFloat16.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/DataTypes_Int8x4.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/ContractionProblem_fwd.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/KernelArguments.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/Distance.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/EmbeddedLibrary.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/TensorDescriptor_Detail.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/SingleSolutionLibrary.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/PlaceholderLibrary.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/DataTypes_XFloat32.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/PerformanceMetricTypes.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/ArithmeticUnitTypes.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/AMDGPU.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/CachingLibrary.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/AMDGPU_Detail.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/ContractionSolution_fwd.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/ContractionProblem_Detail.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/MapLibrary.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/Tensile_fwd.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/Predicates.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/DecisionTree.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/ContractionProblemProperties.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/ContractionProblem.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/TensorDescriptor.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/DataTypes_Float8_BFloat8.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/ExactLogicLibrary.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/EmbeddedData.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/MasterSolutionLibrary.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/DataTypes.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/PropertyMatching.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/TensorOps.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/ProblemKey.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/ContractionProblemPredicates.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/DataTypes_Int8.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/Properties.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/Utils.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/SolutionMapLibrary.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/SolutionLibrary.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/AMDGPUPredicates.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/Serialization.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/ContractionLibrary.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/Comparison.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/DecisionTreeLibrary.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/Task.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/TensorDescriptor_fwd.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/ContractionSolution.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/TensorOps_fwd.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/Debug.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/SolutionLibrary_fwd.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/MatchingLibrary.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/UserDrivenTuningParser.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/TaskPredicates.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/MLFeatures.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/geom.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/Tensile.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/Singleton.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/Macros.hpp"
          "${_CMAKE_CURRENT_BINARY_DIR}/DistinctType.hpp")

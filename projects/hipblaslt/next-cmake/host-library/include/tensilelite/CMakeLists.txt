# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
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
# ########################################################################

if(HIPBLASLT_ENABLE_LLVM)
  add_subdirectory(llvm)
endif()

if(HIPBLASLT_ENABLE_MSGPACK)
  add_subdirectory(msgpack)
endif()

if(HIPBLASLT_ENABLE_HIP)
  add_subdirectory(hip)
endif()

set(_CMAKE_CURRENT_SOURCE_DIR "${TEMP_TENSILE_HOST_SOURCE_DIR}/include")

target_sources(hipblaslt
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/AMDGPU.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/AMDGPUPredicates.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/AMDGPU_Detail.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Activation.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ArithmeticUnitTypes.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/CachingLibrary.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Comparison.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionLibrary.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionProblem.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionProblemPredicates.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionProblemProperties.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionProblem_Detail.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionProblem_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionSolution.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionSolution_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Contractions.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_BFloat16.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_Float8_BFloat8.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_Float8_BFloat8_bc.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_Half.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_Int8.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_Int8x4.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_XFloat32.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Debug.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Distance.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/DistinctType.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/EmbeddedData.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/EmbeddedLibrary.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ExactLogicLibrary.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/FreeSizeLibrary.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/GranularitySelectionLibrary.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/KernelArguments.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/KernelLanguageTypes.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/MLFeatures.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/MLPClassification.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/MLPClassificationLibrary.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Macros.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/MapLibrary.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/MasterSolutionLibrary.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/MatchingLibrary.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/PerformanceMetricTypes.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/PlaceholderLibrary.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Predicates.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ProblemKey.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Properties.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/PropertyMatching.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ScalarValueTypes.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/SingleSolutionLibrary.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Singleton.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/SolutionLibrary.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/SolutionLibrary_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/SolutionMapLibrary.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Tensile.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Tensile_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/TensorDescriptor.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/TensorDescriptor_Detail.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/TensorDescriptor_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/TensorOps.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/TensorOps_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Utils.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/geom.hpp"
)

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(HIPBLASLT_ENABLE_HOST)
    set(_tensilelite_header_scope PUBLIC)
else()
    set(_tensilelite_header_scope PRIVATE)
endif()

target_sources(tensilelite-host
    ${_tensilelite_header_scope}
        FILE_SET tensilelite_public_headers TYPE HEADERS
            BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
            FILES
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/AMDGPU.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/AMDGPUPredicates.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/AMDGPU_Detail.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Activation.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/CachingLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Comparison.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionProblem.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionProblemPredicates.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionProblemProperties.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionProblem_Detail.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionProblem_fwd.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionSolution.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionSolution_fwd.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionTaskPredicates.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Contractions.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_BFloat16.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_BFloat6.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_E5M3.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_E8.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_Float4.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_Float6.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_Float8_BFloat8.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_Half.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_Int8.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_Int8x4.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_XFloat32.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Debug.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Distance.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/DistinctType.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/EmbeddedLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/ExactLogicLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/FreeSizeLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/FixedLinearArbiterLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/FixedLinearCatalogLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/GranularitySelectionLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/KernelArguments.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/KernelLanguageTypes.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/MLFeatures.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/MLPClassification.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/MLPClassificationLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/MapLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/MasterSolutionLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/MatchingLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/PerformanceMetricTypes.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/PlaceholderLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/PredicateDebugger.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Predicates.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/PredictionLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/ProblemKey.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Properties.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/PropertyMatching.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/ScalarValueTypes.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization/Base.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization/Containers.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization/ContractionPredicates.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization/ContractionSolution.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization/ExactLogicLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization/FreeSizeLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization/FixedLinearArbiterLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization/FixedLinearCatalogLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization/GranularitySelectionLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization/HasTraits.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization/MLFeatures.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization/MLPClassificationLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization/MapLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization/MatchingLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization/PlaceholderLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization/Predicates.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization/PredictionLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization/Properties.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization/SolutionLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/SingleSolutionLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Singleton.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/SolutionLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/SolutionLibrary_fwd.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Task.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Tensile.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Tensile_fwd.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/TensorDescriptor.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/TensorDescriptor_Detail.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/TensorDescriptor_fwd.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/TensorOps.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/TensorOps_fwd.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Utils.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/UtilsOrigami.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/geom.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/hip/HipHardware.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/hip/HipSolutionAdapter.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/hip/HipUtils.hpp"
    PRIVATE
        FILE_SET tensilelite_private_headers TYPE HEADERS
            BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
            FILES
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/ArithmeticUnitTypes.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/EmbeddedData.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/SolutionMapLibrary.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/SolutionValidation.hpp"
)

if(HIPBLASLT_ENABLE_YAML)
    target_sources(tensilelite-host
        PRIVATE
            FILE_SET tensilelite_private_headers TYPE HEADERS
                FILES
                    "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/llvm/Loading.hpp"
                    "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/llvm/YAML.hpp"
    )
else()
    target_sources(tensilelite-host
        ${_tensilelite_header_scope}
            FILE_SET tensilelite_public_headers TYPE HEADERS
                FILES
                    "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/msgpack/MessagePack.hpp"
        PRIVATE
            FILE_SET tensilelite_private_headers TYPE HEADERS
                FILES
                    "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/msgpack/Loading.hpp"
    )
endif()

if(HIPBLASLT_ENABLE_HOST)
    get_filename_component(_rocisa_include_root "${CMAKE_CURRENT_SOURCE_DIR}/../rocisa" ABSOLUTE)
    target_sources(tensilelite-host
        ${_tensilelite_header_scope}
            FILE_SET tensilelite_generated_headers TYPE HEADERS
                BASE_DIRS "${CMAKE_CURRENT_BINARY_DIR}"
                FILES "${CMAKE_CURRENT_BINARY_DIR}/tensilelitehost/export.h"
            FILE_SET rocisa_public_headers TYPE HEADERS
                BASE_DIRS "${_rocisa_include_root}"
                FILES "${_rocisa_include_root}/rocisa/include/enum.hpp"
    )
endif()

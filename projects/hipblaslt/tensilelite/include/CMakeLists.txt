# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# TensileLite header classification (G2/G3 of the shared-library plan).
#
# PUBLIC  = the transitive #include closure of the Tensile/* headers that
#           out-of-tree consumers (hipBLASLt's own library/src, hipSPARSELt's
#           tensile_host.cpp) actually include. This is the minimal compile-correct
#           surface; it is installed as a curated FILE_SET (TYPE HEADERS) and
#           co-exported, so a post-install find_package consumer can compile against
#           tensilelite-host. Re-derive with the closure walk in the PR description.
# PRIVATE = backend/internal headers no consumer includes (OpenCL backend, the LLVM
#           and msgpack *loaders*, embedded-data/solution-map/validation internals).
#           Compiled into the library but never installed.
#
# OWNERSHIP: the public FILE_SET is attached (and therefore installed) ONLY when
# hipBLASLt owns the host build (HIPBLASLT_ENABLE_HOST). A fork that builds
# tensilelite-host with HOST OFF (hipSPARSELt) attaches the very same headers
# PRIVATE -- it ships library-only, so the installed Tensile/*.hpp have a single
# owner (hipBLASLt) and never collide in a shared ROCm prefix. CMake enforces the
# rule "an exported target must install all its interface file sets", so a PUBLIC
# file set in the fork would force a colliding header install; PRIVATE avoids it.

set(_tensilelite_public_headers
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
        "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/GranularitySelectionLibrary.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/KernelArguments.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/KernelLanguageTypes.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/MLFeatures.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/MLPClassification.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/MLPClassificationLibrary.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Macros.hpp"
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
)

set(_tensilelite_private_headers
        "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/ArithmeticUnitTypes.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/EmbeddedData.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/SolutionMapLibrary.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/SolutionValidation.hpp"
)

# Backend-selected headers: with YAML the llvm loader is private (no consumer
# includes it); otherwise the msgpack MessagePack.hpp entry point is public and its
# loader is private.
if(HIPBLASLT_ENABLE_YAML)
    list(APPEND _tensilelite_private_headers
        "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/llvm/Loading.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/llvm/YAML.hpp")
else()
    list(APPEND _tensilelite_public_headers
        "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/msgpack/MessagePack.hpp")
    list(APPEND _tensilelite_private_headers
        "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/msgpack/Loading.hpp")
endif()

# Private headers are always build-interface only (compiled in, never installed).
target_sources(tensilelite-host PRIVATE ${_tensilelite_private_headers})

if(HIPBLASLT_ENABLE_HOST)
    # hipBLASLt owns the install -> public headers are a curated, installed,
    # co-exported PUBLIC FILE_SET (BASE_DIRS = the include root, so they install
    # under <prefix>/include/Tensile/...). The generated export header
    # (tensilelitehost/export.h, pulled in by the public Tensile/Macros.hpp) is a
    # second header set rooted at the build tree. Both are installed by the
    # install(TARGETS ... FILE_SET ...) call in ../CMakeLists.txt.
    target_sources(tensilelite-host
        PUBLIC
            FILE_SET tensilelite_public_headers TYPE HEADERS
                BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
                FILES ${_tensilelite_public_headers}
            FILE_SET tensilelite_generated_headers TYPE HEADERS
                BASE_DIRS "${CMAKE_CURRENT_BINARY_DIR}"
                FILES "${CMAKE_CURRENT_BINARY_DIR}/tensilelitehost/export.h"
    )
else()
    # Fork build (e.g. hipSPARSELt, HOST OFF): no header install. Attach the public
    # headers PRIVATE so in-tree consumers still compile against the build interface,
    # but the fork's export set carries library-only (no colliding header install).
    target_sources(tensilelite-host PRIVATE ${_tensilelite_public_headers})
endif()

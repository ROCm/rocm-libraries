# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(BLAS REQUIRED)

find_package(CLI11 QUIET)
if(NOT CLI11_FOUND)
    if(ROCROLLER_ENABLE_FETCH)
        FetchContent_Declare(
            CLI11
            GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(CLI11)
    else()
        message(FATAL_ERROR "Failed to find CLI11")
    endif()
endif()

add_library(rocroller-client-host-data-generation STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/HostDataGeneration.cpp"
)
target_sources(rocroller-client-host-data-generation
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include/client/HostDataGeneration.hpp"
)
target_link_libraries(rocroller-client-host-data-generation
    PUBLIC
        roc::rocroller-interface
        roc::host-validation-core
    PRIVATE
        roc::host-validation
        roc::host-validation-mx
)
target_include_directories(rocroller-client-host-data-generation
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
if(ROCROLLER_ENABLE_ASAN)
    rocroller_target_configure_sanitizers(
        rocroller-client-host-data-generation
        PRIVATE
    )
endif()

add_executable(rocroller-gemm
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gemm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/GEMMParameters.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/GraphInspector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/visualize.cpp"
)
target_sources(rocroller-gemm
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include/client/BenchmarkSolution.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/client/CLI_Utils.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/client/DataParallelGEMMSolution.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/client/GEMMParameters.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/client/GEMMParameters_serialization.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/client/GEMMSolution.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/client/GraphInspector.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/client/GraphInspector_impl.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/client/StreamKGEMMSolution.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/client/visualize.hpp"
)
target_link_libraries(rocroller-gemm
    PRIVATE
        roc::rocroller-interface
        rocroller-client-host-data-generation
        rocroller::common-test-utilities
        # rocroller-gemm retains mxDataGenerator only for physical pre-swizzles.
        roc::mxDataGenerator
        CLI11::CLI11
        ${BLAS_LIBRARIES}
)
target_include_directories(rocroller-gemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(ROCROLLER_BUILD_SHARED_LIBS) # why?
    target_link_libraries(rocroller-gemm PUBLIC ${llvm_yaml_libs})
endif()

if(ROCROLLER_ENABLE_ASAN)
    rocroller_target_configure_sanitizers(rocroller-gemm PRIVATE)
endif()

add_executable(rocroller-codegen-stress "${CMAKE_CURRENT_SOURCE_DIR}/src/codegen_stress_test.cpp")
target_link_libraries(rocroller-codegen-stress
    PRIVATE
        roc::rocroller-interface
        CLI11::CLI11
)
if(ROCROLLER_BUILD_SHARED_LIBS) # why are we using this logic
    target_link_libraries(rocroller-codegen-stress PRIVATE ${llvm_yaml_libs})
endif()

if(ROCROLLER_ENABLE_ASAN)
    rocroller_target_configure_sanitizers(rocroller-codegen-stress PRIVATE)
endif()

if(ROCROLLER_BUILD_TESTING OR BUILD_TESTING)
    add_executable(rocroller-client-host-data-generation-tests
        "${CMAKE_CURRENT_SOURCE_DIR}/test/HostDataGenerationTest.cpp"
    )
    target_link_libraries(rocroller-client-host-data-generation-tests
        PRIVATE
            rocroller-client-host-data-generation
    )
    add_test(
        NAME rocroller-client-host-data-generation-tests
        COMMAND rocroller-client-host-data-generation-tests
    )
    if(ROCROLLER_ENABLE_ASAN)
        rocroller_target_configure_sanitizers(
            rocroller-client-host-data-generation-tests
            PRIVATE
        )
    endif()
endif()

add_custom_target(all_clients)
add_dependencies(all_clients rocroller-gemm rocroller-codegen-stress)

if(ROCROLLER_ENABLE_COVERAGE OR BUILD_CODE_COVERAGE)
    target_compile_options(rocroller-gemm PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(rocroller-gemm PRIVATE -fprofile-instr-generate)

    target_compile_options(rocroller-codegen-stress PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(rocroller-codegen-stress PRIVATE -fprofile-instr-generate)
endif()

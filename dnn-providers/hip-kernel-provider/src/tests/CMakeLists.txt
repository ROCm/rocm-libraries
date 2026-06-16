# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_compile_definitions(__HIP_PLATFORM_AMD__)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

# hipdnn SDK dependencies - use targets directly if available (superbuild),
# otherwise find installed packages (true standalone)
if(NOT TARGET hipdnn_test_sdk)
    find_package(hipdnn_test_sdk CONFIG REQUIRED)
endif()
if(NOT TARGET hipdnn_plugin_sdk)
    find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
endif()

add_executable(
    hip_kernel_provider_tests
    main.cpp
    core/TestContainer.cpp
    core/TestHandle.cpp
    core/TestContext.cpp
    core/TestUtils.cpp
    core/TestPluginPublic.cpp
    compilation/TestProgramAndKernel.cpp
    compilation/TestKernelCompileOptions.cpp
)

# Add the dummy kernel for running the tests
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/KernelEmbedding.cmake")
add_kernels_for_embedding(TARGET hip_kernel_provider_tests FILES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/vector_add.cpp)
embed_kernel_sources(
    TARGET hip_kernel_provider_tests
    OUTPUT_SRCS_CPP ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_sources.cpp
    OUTPUT_SRCS_HPP ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_sources.hpp
    OUTPUT_INCS_CPP ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_includes.cpp
    OUTPUT_INCS_HPP ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_includes.hpp
)
set(HIP_KERNEL_PROVIDER_GENERATED_TEST_SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_sources.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/engines/kernel_includes.cpp)

set_source_files_properties(${HIP_KERNEL_PROVIDER_GENERATED_TEST_SOURCES} TARGET_DIRECTORY hip_kernel_provider_tests PROPERTIES COMPILE_OPTIONS "-Wno-overlength-strings")

target_sources(hip_kernel_provider_tests PRIVATE "${HIP_KERNEL_PROVIDER_GENERATED_TEST_SOURCES}")

target_compile_options(hip_kernel_provider_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(
    hip_kernel_provider_tests PUBLIC GTest::gtest GTest::gmock hip::host
                                   hip_kernel_provider_private hipdnn_test_sdk hipdnn_plugin_sdk Threads::Threads
)

# make it so tests inside folders can find files without having to use a relative path.
target_include_directories(hip_kernel_provider_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include ${CMAKE_BINARY_DIR}/src/tests/engines)

if(ENABLE_ASM_SDPA_ENGINE)
    if(NOT HIPDNN_ENABLE_SDPA)
        message(WARNING
                "ENABLE_ASM_SDPA_ENGINE is ON but hipDNN was built with HIPDNN_ENABLE_SDPA=OFF. "
                "Skipping SDPA unit tests. Rebuild hipDNN with -DHIPDNN_ENABLE_SDPA=ON to enable them.")
    else()
        add_subdirectory(engines/asm_sdpa_engine)
    endif()
endif()

if(ENABLE_HIP_MLOPS_ENGINE)
    add_subdirectory(engines/hip_mlops_engine)
endif()

clang_tidy_check(hip_kernel_provider_tests)

add_unit_test_target(hip_kernel_provider_tests ${CMAKE_CURRENT_BINARY_DIR})

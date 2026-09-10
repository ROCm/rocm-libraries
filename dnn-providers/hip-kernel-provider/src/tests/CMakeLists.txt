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
    compilation/TestModuleCache.cpp
)

# Add the dummy kernel for running the tests
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/KernelEmbedding.cmake")
add_kernels_for_embedding(TARGET hip_kernel_provider_tests FILES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/vector_add.cpp)

# Test binary needs its own copy of each pack's kernel (embedding is per-target).
# Copied via configure_file: registering the same source for embedding in two
# directories doubles its CMAKE_CONFIGURE_DEPENDS entry, which Ninja rejects as two
# rules for one file. The list itself is the engine's, so a new pack adds its kernel
# once and both binaries pick it up.
if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../engines/kernel_ingestor_engine/IngestorKernels.cmake")
    foreach(kernel IN LISTS HIPDNN_INGESTOR_PACK_KERNELS)
        configure_file(
            ${HIPDNN_INGESTOR_PACK_KERNEL_DIR}/${kernel}.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/kernels/${kernel}.cpp
            COPYONLY
        )
        add_kernels_for_embedding(
            TARGET hip_kernel_provider_tests
            FILES ${CMAKE_CURRENT_BINARY_DIR}/kernels/${kernel}.cpp
        )
    endforeach()
endif()
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

if(ENABLE_HIP_FLASH2_ENGINE)
    add_subdirectory(engines/hip_flash2_engine)
endif()
if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    add_subdirectory(engines/kernel_ingestor_engine)

    # descriptorSearchDirectory()'s tests assert which of its three sources answered, and
    # the only way to tell branch 2 or 3 from a failed lookup is that both end in this
    # fixed offset. Same value the provider is compiled with, so the test cannot drift
    # into asserting a path the provider never builds.
    #
    # The staged tree's own path rides along so main() can point this binary at the
    # descriptors with no ctest in the loop -- see the note there.
    target_compile_definitions(hip_kernel_provider_tests PRIVATE
        HIPDNN_DESCRIPTOR_SUBDIR="${HIPDNN_DESCRIPTOR_SUBDIR}"
        HIPDNN_TEST_DESCRIPTOR_DIR="${HIPDNN_DESCRIPTOR_BUILD_DIR}")

    # The kpack adapter's tests. Gated with the sources they exercise, and kept together
    # here -- including the dispatch one, which sits under engines/ -- so all three sit
    # between the two defines they read. Those are HIPDNN_TEST_DESCRIPTOR_DIR above, under
    # which the device cases find the archive this build packed for the local arch, and
    # HIPDNN_TEST_KPACK_ARCHIVE below, which the parse-level cases open because they need
    # a real container rather than a matching device.
    target_sources(hip_kernel_provider_tests PRIVATE
        compilation/TestKpackKernelLoader.cpp
        compilation/TestKpackModuleCache.cpp
        engines/kernel_ingestor_engine/packs/TestPointwiseKpackDispatch.cpp
        ingestor/TestPackedDescriptorLoad.cpp
    )

    # A real archive to open, so the arch-mismatch and missing-toc_key cases meet the
    # actual reader rather than a hand-forged header. Vendored from rocm-kpack's own
    # test assets: the reader is now the platform's shared library, so there is no
    # fetched source tree to take it from, and a 518-byte container is cheaper to carry
    # than a build-time dependency on one.
    target_compile_definitions(hip_kernel_provider_tests PRIVATE
        HIPDNN_TEST_KPACK_ARCHIVE="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/test_zstd.kpack")

    # The device cases above read the per-arch archive the packaging roots stage under
    # HIPDNN_TEST_DESCRIPTOR_DIR, and that target is reachable only through ALL.
    # Building this binary by name -- what `ninja check` does -- otherwise runs them
    # against a directory that does not exist yet, and they skip rather than fail.
    # Guarded for the same reason as the integration side: a root skips with a warning
    # when hipcc is absent, and a hard dependency would make that fatal.
    if(TARGET hkp_descriptor_staging)
        add_dependencies(hip_kernel_provider_tests hkp_descriptor_staging)
    endif()
endif()

clang_tidy_check(hip_kernel_provider_tests)

# No ENVIRONMENT for the descriptors: main() applies the compiled-in path, so the binary
# runs standalone and nothing machine-specific reaches the generated CTest files. The
# install-time file is written from this same list, where a build-tree path would name a
# directory that does not exist on the machine running the installed suite.
add_unit_test_target(hip_kernel_provider_tests ${CMAKE_CURRENT_BINARY_DIR})

# Published one scope up for the same reason the integration tests do it: under YAML
# categorization the raw test name is never registered, so the tiered suites created by
# apply_test_category_labels() in ../CMakeLists.txt are the only place the ambient test
# environment -- an ASAN symbolizer path, a coverage profile destination -- can land.
if(DEFINED hip_kernel_provider_tests_TEST_ENVIRONMENT)
    set(hip_kernel_provider_tests_TEST_ENVIRONMENT
        "${hip_kernel_provider_tests_TEST_ENVIRONMENT}" PARENT_SCOPE)
endif()

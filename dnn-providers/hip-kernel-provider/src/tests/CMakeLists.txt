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

if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    # The kernel sources the ingestor engine's unit descriptor set names. Each key is the
    # `source_file` its descriptor authors, which is the string the runtime looks up.
    # hkp_verify_embedded_sources() compares this list against the staged descriptors.
    set(_ingestor_kernel_dir
        "${CMAKE_CURRENT_SOURCE_DIR}/../engines/kernel_ingestor_engine/test_descriptors/unit/pointwise/kernels")
    add_kernels_for_embedding(
        TARGET hip_kernel_provider_tests
        FILES "${_ingestor_kernel_dir}/PointwiseAdd.cpp"
              "${_ingestor_kernel_dir}/PointwiseMul.cpp"
              "${_ingestor_kernel_dir}/PointwiseSub.cpp"
        KEYS kernels/PointwiseAdd.cpp
             kernels/PointwiseMul.cpp
             kernels/PointwiseSub.cpp
    )
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
target_include_directories(hip_kernel_provider_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include ${CMAKE_CURRENT_BINARY_DIR}/engines)

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
    # One define per descriptor set. A single define cannot serve both: the discovery set
    # and the archive fixture root are different folders.
    #
    # Binary-relative, never an absolute build path. The offset holds in the build tree and
    # the install tree alike, so these cases read the descriptors beside whichever copy of
    # the binary is running instead of reaching back into the tree that configured it -- an
    # installed run would otherwise find nothing, enumerate nothing, and pass.
    target_compile_definitions(hip_kernel_provider_tests PRIVATE
        HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR="${HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR}"
        HIPKERNELPROVIDER_TEST_SET_UNIT_RELDIR="${HIPKERNELPROVIDER_TEST_DESCRIPTOR_RELDIR}/${HIPKERNELPROVIDER_TEST_SET_UNIT}"
        HIPKERNELPROVIDER_ARCHIVE_FIXTURE_RELDIR="${HIPKERNELPROVIDER_TEST_DESCRIPTOR_RELDIR}/${HIPKERNELPROVIDER_TEST_SET_INTEGRATION}/packed")

    # The kpack adapter's tests. Gated with the sources they exercise, and kept together
    # here -- including the dispatch one, which sits under engines/ -- so all three sit
    # between the two defines they read. Those are
    # HIPKERNELPROVIDER_ARCHIVE_FIXTURE_RELDIR above, under which the device cases
    # find the archive this build packed for the local arch, and HIPKERNELPROVIDER_TEST_KPACK_ARCHIVE
    # below, which the parse-level cases open because they need a real container rather
    # than a matching device.
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
        HIPKERNELPROVIDER_TEST_KPACK_ARCHIVE="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/test_zstd.kpack")

    # This binary reads the per-arch descriptors and archives the packing roots stage under
    # test_arch_content/, and each packing target is reachable only through ALL.
    # Building this binary by name -- what `ninja check` does -- otherwise runs the cases
    # against a directory that does not exist yet, and they skip rather than fail.
    # Each descriptor set has its own staging target. Name each one individually.
    foreach(_hkp_set hkp_packaging_unit_conv hkp_packaging_unit_pointwise
                     hkp_packaging_archive_fixture)
        if(TARGET ${_hkp_set})
            add_dependencies(hip_kernel_provider_tests ${_hkp_set})
        endif()
    endforeach()

    # The parent of the unit descriptor sets, and the archive fixture root this
    # binary also compiles in. A walk of a parent reaches every subtree staged
    # under it, so a further pack there comes under the check with no edit here.
    hkp_verify_embedded_sources(
        TARGET hip_kernel_provider_tests
        STAGED_DESCRIPTOR_ROOTS ${HIPKERNELPROVIDER_UNIT_BUILD_DIR}
                                ${HIPKERNELPROVIDER_ARCHIVE_FIXTURE_BUILD_DIR}
        PACK_NAMES unit_conv unit_pointwise archive_fixture
    )
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

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

if(ENABLE_AOT_CATALOG_ENGINE)
    # Engine-level (substrate) tests live HERE because they are not tied to any
    # one kernel family: TestTuneCache is pure host CPU-key/persistence logic, and
    # TestSdpaDecode exercises the SDPA adapter's decode vocabulary (host-only).
    # Every FAMILY-specific test (numeric parity / per-family selection) instead
    # lives IN its family folder in the rocke aot_catalog subsystem
    # (rocke/aot_catalog/<arch>/<family>/) and is contributed via the
    # AOT_CATALOG_FAMILY_TEST_SOURCES global property below -- so deleting a family
    # folder removes its kernels, its family.json AND its tests together, with no
    # impact on any other family. Those tests #include this engine's private headers,
    # so this target (not rocke) owns their compile, keeping rocke free of any
    # build-graph dependency on the throwaway engine. See rocke/aot_catalog/README.md
    # and rocke_add_aot_family_test() in rocke/aot_catalog/CMakeLists.txt.
    target_sources(hip_kernel_provider_tests PRIVATE
        engines/aot_catalog_engine/TestSdpaDecode.cpp
        engines/aot_catalog_engine/TestTuneCache.cpp
        engines/aot_catalog_engine/TestWorkspaceExpr.cpp
    )
    # Family-contributed test sources (absolute paths appended by each family's
    # aot_add_family_test()). Empty when rocKE/the arch is unavailable -> no-op.
    get_property(_aot_family_test_sources GLOBAL PROPERTY AOT_CATALOG_FAMILY_TEST_SOURCES)
    if(_aot_family_test_sources)
        target_sources(hip_kernel_provider_tests PRIVATE ${_aot_family_test_sources})
    endif()
    # The AOT engine's headers (catalog/, launch/, plans/) are rooted at the
    # engine dir, which is PRIVATE to hip_kernel_provider_impl -- add it here so
    # the substrate + family tests can include them directly.
    target_include_directories(hip_kernel_provider_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../engines/aot_catalog_engine
    )
    # Point the test at the catalog (family.json + .co). The test resolves the
    # directory at RUNTIME via aotResolveTestCatalogDir() (AotCatalogTestSupport.hpp),
    # mirroring the engine's own self-location so it works both in the build tree and
    # from a packaged install tree on a SEPARATE runner (TheRock's split build/test
    # CI). Three pieces are baked:
    #   * AOT_CATALOG_TEST_DIR    -- the build-machine absolute path; last-resort
    #                                local-dev fallback only (invalid on the CI test
    #                                runner, which is exactly why it can't be the
    #                                sole source -- see the Windows empty-catalog bug).
    #   * AOT_CATALOG_RELDIR      -- catalog offset beneath the plugin-engine dir.
    #   * AOT_CATALOG_ENGINE_SUBDIR -- plugin-engine dir offset beneath the install
    #                                "bin"/"lib" root (the test exe installs to that
    #                                root, one level above the engine dir).
    target_compile_definitions(hip_kernel_provider_tests PRIVATE
        AOT_CATALOG_TEST_DIR="${HIPDNN_AOT_CATALOG_BUILD_DIR}"
        AOT_CATALOG_RELDIR="${HIPDNN_AOT_CATALOG_RELDIR}"
        AOT_CATALOG_ENGINE_SUBDIR="${HIPDNN_PLUGIN_ENGINE_SUBDIR}"
    )
    # Build the rocKE-AOT kernel families before the tests so an isolated
    # `ninja hip_kernel_provider_tests` populates the build-tree catalog rather than
    # running against an empty one (which would silently skip every parity test).
    # The property is empty when rocKE is unavailable, so this is a no-op then.
    #
    # That same property also tells the tests whether the catalog is EXPECTED to be
    # populated: if any family target was registered, rocKE was available and the
    # arch was requested, so a producer ran and the build-tree catalog must be
    # non-empty at test time. AOT_ROCKE_FAMILIES_EXPECTED=1 flips the parity/
    # selection tests from GTEST_SKIP to FAIL on an empty catalog (see
    # AotCatalogTestSupport.hpp) so a silent producer/loader failure that empties
    # the catalog can't leave CI green. When rocKE is off the property is empty and
    # the def is 0, keeping empty-catalog an expected skip on TheRock CI.
    get_property(_aot_family_targets GLOBAL PROPERTY AOT_CATALOG_FAMILY_TARGETS)
    if(_aot_family_targets)
        add_dependencies(hip_kernel_provider_tests ${_aot_family_targets})
        set(_aot_families_expected 1)
    else()
        set(_aot_families_expected 0)
    endif()
    target_compile_definitions(hip_kernel_provider_tests PRIVATE
        AOT_ROCKE_FAMILIES_EXPECTED=${_aot_families_expected})
endif()

clang_tidy_check(hip_kernel_provider_tests)

add_unit_test_target(hip_kernel_provider_tests ${CMAKE_CURRENT_BINARY_DIR})

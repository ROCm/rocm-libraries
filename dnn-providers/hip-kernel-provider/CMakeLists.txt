# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)
add_compile_definitions(__HIP_PLATFORM_AMD__)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ClangToolChain.cmake"
    CACHE STRING "Toolchain file"
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/VersionUtils.cmake)

dnn_provider_setup_version(hip_kernel_provider ${CMAKE_CURRENT_LIST_DIR})

project(hip-kernel-provider VERSION ${HIP_KERNEL_PROVIDER_VERSION} LANGUAGES CXX)

# shared/ctest is only present in a full rocm-libraries checkout; guard the check at the
# point of use so standalone / sparse checkouts that lack shared/ctest still build.
if(NOT DEFINED ROCM_LIBRARIES_ROOT)
    set(ROCM_LIBRARIES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
endif()

if(NOT WIN32 AND DEFINED ROCM_PATH)
    # Only set if not already specified by user
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "${ROCM_PATH}" CACHE PATH "Install path prefix" FORCE)
    endif()
    message(STATUS "Setting CMAKE_INSTALL_PREFIX to ${CMAKE_INSTALL_PREFIX}")
endif()

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE
    CACHE BOOL "Add paths to linker search and installed rpath"
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake
     ${CMAKE_CURRENT_SOURCE_DIR}/../cmake ${ROCM_PATH}/lib/cmake/hip ${HIP_DIR}/cmake
)

option(HIPKERNELPROVIDER_ENABLE_TESTS "Build tests" ON)
option(HIPKERNELPROVIDER_ENABLE_ROCKE "Build rocKE engine and smoke tests" OFF)

# Gates the generic kernel ingestor (RFC 0017). The ingestor engine compiles its kernel
# through the HIP kernel engine's compiler, so it only builds alongside that engine.
option(HIPDNN_ENABLE_KERNEL_INGESTOR "Enable the generic kernel ingestor" OFF)
option(ENABLE_ASM_SDPA_ENGINE "Build the ASM SDPA engine" ON)
option(ENABLE_HIP_MLOPS_ENGINE "Build the HIP kernel engine" ON)

option(BUILD_ADDRESS_SANITIZER "Build with Address Sanitizer enabled" OFF)
option(ENABLE_CLANG_FORMAT "Enable Clang-Format checks" ON)
if(WIN32)
    option(ENABLE_CLANG_TIDY "Enable Clang-Tidy checks" OFF)
else()
    option(ENABLE_CLANG_TIDY "Enable Clang-Tidy checks" ON)
endif()
option(HIPKERNELPROVIDER_ENABLE_COVERAGE "Build with code coverage flags" OFF)

if(HIPKERNELPROVIDER_ENABLE_COVERAGE)
    message(VERBOSE "Enabling compile flags for coverage-mapping")
    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate -fcoverage-mapping)
endif()

if(ENABLE_ASM_SDPA_ENGINE)
    message(STATUS "ASM SDPA engine enabled")
else()
    message(STATUS "ASM SDPA engine disabled")
endif()

option(ENABLE_HIP_FLASH2_ENGINE "Build the HIP Flash-Attention 2 SDPA engine (FP16, gfx942)" OFF)
if(ENABLE_HIP_FLASH2_ENGINE)
    message(STATUS "HIP Flash2 SDPA engine enabled")
    # Engine sources and compile definitions are wired onto the provider's object
    # library in src/CMakeLists.txt (hip_kernel_provider target does not exist yet
    # at this point -- it is created by add_subdirectory(src) below).
    # The Flash2 kernel is shipped as a precompiled .co file (gfx942).
    # HipFlash2FwdPlan.hip is NOT compiled here -- it is the offline source for
    # the .co files and requires rocWMMA headers.
    # At runtime the .co is loaded via hipModuleLoad in HipFlash2KernelUtils.hpp.
    install(FILES
        src/engines/hip_flash2_engine/kernels/hip_flash2_fwd_gfx942.co
        DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/hip_kernel_provider/hip_flash2_kernels)
else()
    message(STATUS "HIP Flash2 SDPA engine disabled")
endif()

if(ENABLE_HIP_MLOPS_ENGINE)
    message(STATUS "HIP kernel engine enabled")
else()
    message(STATUS "HIP kernel engine disabled")
endif()

list(
    PREPEND
    HIPDNN_WARNING_COMPILE_OPTIONS
    -Werror
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Woverloaded-virtual
    -Wconversion
    -Wsign-conversion
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
    -Winit-self
    -Wunreachable-code
    -Wno-error=unused-command-line-argument
    -Wswitch-default
)

# NOTE: Workaround until llvm & hip cmake modules fixes symlink logic in their config files;
# remove when fixed. Added to prevent issues with not finding hip cmake.
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm ${ROCM_PATH} ${ROCM_PATH}/hip)

find_package(hip REQUIRED)
find_package(hiprtc REQUIRED)

# hipdnn SDK dependencies - use targets directly if available (superbuild),
# otherwise find installed packages (true standalone)
if(NOT TARGET hipdnn_data_sdk)
    find_package(hipdnn_data_sdk CONFIG REQUIRED)
endif()
if(NOT TARGET hipdnn_flatbuffers_sdk)
    find_package(hipdnn_flatbuffers_sdk CONFIG REQUIRED)
endif()
if(NOT TARGET hipdnn_plugin_sdk)
    find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
endif()

include(CheckToolVersion)
include(ClangCheck)
include(ClangTidy)
include(Sanitizers)

# Setup testing dependencies
include(Dependencies)
if(HIPKERNELPROVIDER_ENABLE_TESTS)
    fetch_gtest_dependency()
endif()
include(Tests)

if(WIN32)
    set(HIPDNN_PLUGIN_ENGINE_ROOTDIR "${CMAKE_INSTALL_BINDIR}")
else()
    set(HIPDNN_PLUGIN_ENGINE_ROOTDIR "${CMAKE_INSTALL_LIBDIR}")
endif()

set(HIPDNN_BUILD_PLUGIN_ENGINE_DIR
    "${CMAKE_BINARY_DIR}/${HIPDNN_PLUGIN_ENGINE_ROOTDIR}/${HIPDNN_PLUGIN_ENGINE_SUBDIR}"
)

message(STATUS "Plugin build directory: ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}")
message(STATUS "Plugin install directory: ${HIPDNN_INSTALL_PLUGIN_ENGINE_DIR}")
file(MAKE_DIRECTORY ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR})

if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    # Production content.
    set(HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR "arch_content/hip-kernel-provider")
    set(HIPKERNELPROVIDER_DESCRIPTOR_BUILD_DIR
        "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}/${HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR}"
    )

    # Test content lives in a sibling of arch_content/, under a provider segment for the
    # same reason arch_content/ has one: the engines directory is shared by every provider,
    # so an unqualified test_arch_content/ would let two providers collide on a set name --
    # and would let this provider's install(DIRECTORY) ship another provider's content.

    set(HIPKERNELPROVIDER_TEST_DESCRIPTOR_SUBDIR "test_arch_content/hip-kernel-provider")
    set(HIPKERNELPROVIDER_TEST_DESCRIPTOR_BUILD_DIR
        "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}/${HIPKERNELPROVIDER_TEST_DESCRIPTOR_SUBDIR}"
    )

    # Two test descriptor roots, one per test binary. The unit root holds two packed
    # subtrees and the integration root holds three, all five wired in
    # descriptor-packaging/cmake/HkpPackaging.cmake. The packer stages each authored
    # subtree per architecture: it rewrites an authored hip descriptor to kpack and writes
    # the archive beside it. For an embedded_source descriptor it stamps the shard
    # architecture and records the authored values in a provenance block. An
    # embedded_source descriptor names a kernel compiled into a binary and JIT-compiled
    # at load. Its subtree gets descriptors and no archive. Both roots stage into the
    # build tree rather than being read from the source tree, so the build tree matches
    # the install tree.
    set(HIPKERNELPROVIDER_TEST_SET_UNIT "unit")
    set(HIPKERNELPROVIDER_TEST_SET_INTEGRATION "integration")
    # Defined here, with the other path variables, and NOT beside the staging rules that
    # write them: add_subdirectory(src) runs before those rules, so a definition further
    # down reaches the test subdirectories empty -- which resolves to no descriptor set at
    # all and fails every test that needs one, with nothing in the build to say why.
    set(HIPKERNELPROVIDER_UNIT_BUILD_DIR
        "${HIPKERNELPROVIDER_TEST_DESCRIPTOR_BUILD_DIR}/${HIPKERNELPROVIDER_TEST_SET_UNIT}")
    set(HIPKERNELPROVIDER_INTEGRATION_BUILD_DIR
        "${HIPKERNELPROVIDER_TEST_DESCRIPTOR_BUILD_DIR}/${HIPKERNELPROVIDER_TEST_SET_INTEGRATION}")

    set(HIPKERNELPROVIDER_TEST_DESCRIPTOR_SOURCE_ROOT
        "${CMAKE_CURRENT_SOURCE_DIR}/src/engines/kernel_ingestor_engine/test_descriptors")

    # The name of each authored set, which is also the name it stages under. Named here
    # only because the test subdirectories compile these same segments into their binaries
    # as a relative offset; the packing rules in HkpPackaging.cmake join them to the two
    # roots above directly.
    #
    # Every set is a top-level folder under test_descriptors/, and every one is packed
    # WHOLE rather than by the set inside it. Each descriptor therefore lands in a CHILD of
    # its shard root while the archive is written at the shard root itself, so it must
    # climb out of its own folder to reach the archive -- the climb the runtime containment
    # guard checks, which a set packed from its own leaf folder never produces.
    #
    # One archive is written per pack. A folder added under one of these roots joins that
    # root's archive rather than getting its own; a set that has to be able to fail on its
    # own needs a top-level folder here and an OUT_ROOT of its own. That is why
    # archive_fixture/ is authored beside the others rather than inside integration/: the
    # broken-archive test corrupts its archive and asserts the shipped pointwise engine
    # still serves, which needs the two archives to be separate files.
    set(HIPKERNELPROVIDER_TEST_SET_SHARED "shared")
    set(HIPKERNELPROVIDER_TEST_SET_ARCHIVE_FIXTURE "archive_fixture")

    # The offset a test binary walks from its own directory to reach the test descriptor
    # trees. Both trees hang off HIPDNN_PLUGIN_ROOTDIR and both binaries off
    # CMAKE_INSTALL_BINDIR, under ${CMAKE_BINARY_DIR} in the build tree and under the
    # prefix in the install tree, so this one offset reaches them in either -- and a
    # binary carries no path belonging to the machine that configured it. Computed against
    # a synthetic root because file(RELATIVE_PATH) needs two absolute paths, and only the
    # relationship between them matters here.
    set(_synthetic_root "/__hipkernelprovider_root__")
    file(RELATIVE_PATH HIPKERNELPROVIDER_TEST_DESCRIPTOR_RELDIR
         "${_synthetic_root}/${CMAKE_INSTALL_BINDIR}"
         "${_synthetic_root}/${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/${HIPKERNELPROVIDER_TEST_DESCRIPTOR_SUBDIR}")

    # No HIPDNN_DESCRIPTOR_DIR is published for the test suites: the integration binary loads
    # the plugin and resolves the tree from that module, and the unit binary resolves it
    # from its own module in main().
    message(STATUS "Kernel ingestor enabled. Descriptors build into:")
    message(STATUS "  production: ${HIPKERNELPROVIDER_DESCRIPTOR_BUILD_DIR}")
    message(STATUS "  test:       ${HIPKERNELPROVIDER_TEST_DESCRIPTOR_BUILD_DIR}")
endif()

# ============================================================================
# YAML-based CTest categorization detection
# ============================================================================
# Hoisted here (rather than src/CMakeLists.txt) so it is available before
# add_subdirectory(rocke) below: rocke/platform/tests/CMakeLists.txt applies
# ROCKE_ENGINE_CTEST_CATEGORIES_YAML to its own pre-registered CTest tests in
# its own directory scope (CMake's set_property(TEST ...) predates 3.28's
# cross-directory support, so this project's 3.25.2 floor requires it run
# where the tests were add_test()'d).
if(HIPKERNELPROVIDER_ENABLE_TESTS)
    set(HIPKERNELPROVIDER_GTEST_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/HIP_MLOPS_ENGINE_test_categories.yaml")
    set(HIPKERNELPROVIDER_CTEST_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/HIP_MLOPS_ENGINE_test_categories_external.yaml")
    set(HIPKERNELPROVIDER_INTEGRATION_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/HIP_MLOPS_ENGINE_test_categories_integration.yaml")
    set(HIPKERNELPROVIDER_ASM_SDPA_INTEGRATION_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/ASM_SDPA_ENGINE_test_categories_integration.yaml")
    set(ROCKE_ENGINE_CTEST_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/ROCKE_ENGINE_test_categories_external.yaml")
    set(HKP_PACK_CTEST_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/HKP_PACK_test_categories_external.yaml")
    set(HIPKERNELPROVIDER_CTEST_INSTALL_FILE "")
    set(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED FALSE)
    if(EXISTS "${HIPKERNELPROVIDER_GTEST_CATEGORIES_YAML}"
       AND EXISTS "${HIPKERNELPROVIDER_CTEST_CATEGORIES_YAML}"
       AND EXISTS "${HIPKERNELPROVIDER_INTEGRATION_CATEGORIES_YAML}"
       AND EXISTS "${HIPKERNELPROVIDER_ASM_SDPA_INTEGRATION_CATEGORIES_YAML}"
       AND EXISTS "${ROCKE_ENGINE_CTEST_CATEGORIES_YAML}"
       AND EXISTS "${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
        include("${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
        set(DNN_PROVIDER_CTEST_CATEGORIES_YAML "${HIPKERNELPROVIDER_CTEST_CATEGORIES_YAML}")
        set(DNN_PROVIDER_TEST_CATEGORY_YAMLS
            "${HIPKERNELPROVIDER_GTEST_CATEGORIES_YAML}"
            "${HIPKERNELPROVIDER_INTEGRATION_CATEGORIES_YAML}"
            "${HIPKERNELPROVIDER_ASM_SDPA_INTEGRATION_CATEGORIES_YAML}"
            "${ROCKE_ENGINE_CTEST_CATEGORIES_YAML}"
            "${HIPKERNELPROVIDER_CTEST_CATEGORIES_YAML}"
        )
        set(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED TRUE)
        message(STATUS "hip-kernel-provider: YAML-based CTest categorization enabled")
    else()
        message(STATUS "hip-kernel-provider: shared/ctest or test category YAML not found, using legacy CTest labels")
    endif()
endif()

if(HIPKERNELPROVIDER_ENABLE_ROCKE)
    message(STATUS "rocKE features enabled")

    # rocke/ = platform (engine + build-time Python). The engine's installed
    # CTest entries stage into EXTERNAL_TEST_INSTALL_STAGING_hip_kernel_provider,
    # so add_subdirectory(rocke) must precede src/ below, which emits
    # bin/hip_kernel_provider/CTestTestfile.cmake.
    set(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR "hip_kernel_provider")
    set(ROCKE_INSTALL_STANDALONE_CTEST OFF)

    add_subdirectory(rocke)
else()
    message(STATUS "rocKE features disabled")
endif()

# Both halves of kpack support ride the ingestor gate and nothing else.
#
# The reader is the rocm_kpack shared library ROCm ships, resolved the way this repo's
# other providers resolve their externals: miopen-provider finds MIOpen, hipblaslt-provider
# finds the SDKs. The guard lets a superbuild that already defines the target -- one
# building rocm-kpack as a sibling component -- win over an installed copy.
#
# descriptor-packaging resolves the packer separately, from the rocm_kpack Python
# package. Both halves come from the platform, so nothing here pins a ref.
if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    if(NOT TARGET rocm::rocm_kpack)
        find_package(rocm-kpack CONFIG REQUIRED)
    endif()
    add_subdirectory(descriptor-packaging)
endif()

add_subdirectory(src)

# ============================================================================
# Code Coverage
# ============================================================================
if(HIPKERNELPROVIDER_ENABLE_COVERAGE)
    findandcheckllvmtools()

    set(CODE_COVERAGE_IGNORE_REGEX ".*_deps.*|.*tests.*|.*integration_tests.*")
    set(CODE_COVERAGE_OBJECTS
        -object ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}/libhip_kernel_provider.so
        -object ./${CMAKE_INSTALL_BINDIR}/hip_kernel_provider_tests
        -object ./${CMAKE_INSTALL_BINDIR}/hip_kernel_provider_integration_tests
    )

    # Creates a coverage target that merges profraw files and generates reports.
    #
    # @param TARGET_NAME name of the custom target to create
    # @param DEPENDS_TARGET target to depend on (empty string for no dependency)
    function(add_custom_coverage_target TARGET_NAME DEPENDS_TARGET)
        set(DEPENDS_CLAUSE "")
        if(NOT "${DEPENDS_TARGET}" STREQUAL "")
            set(DEPENDS_CLAUSE DEPENDS ${DEPENDS_TARGET})
        endif()

        # cmake-format: off
        add_custom_target(
            ${TARGET_NAME}
            COMMAND find . -name "*.profraw" -print0 | xargs -0 ${LLVM_PROFDATA_BINARY} merge
                    -sparse -o ./coverage-report/hip_kernel_provider.profdata
            COMMAND
                ${LLVM_COV_BINARY} report ${CODE_COVERAGE_OBJECTS}
                -instr-profile=./coverage-report/hip_kernel_provider.profdata
                -ignore-filename-regex=${CODE_COVERAGE_IGNORE_REGEX} >
                ./coverage-report/code_cov_hip_kernel_provider.report
            COMMAND
                ${LLVM_COV_BINARY} show -Xdemangler=${LLVM_CXXFILT_BINARY} ${CODE_COVERAGE_OBJECTS}
                -instr-profile=./coverage-report/hip_kernel_provider.profdata
                -ignore-filename-regex=${CODE_COVERAGE_IGNORE_REGEX} --format=html
                --output-dir=coverage-report
            COMMAND
                ${LLVM_COV_BINARY} export ${CODE_COVERAGE_OBJECTS}
                -instr-profile=./coverage-report/hip_kernel_provider.profdata
                -ignore-filename-regex=${CODE_COVERAGE_IGNORE_REGEX} --format=lcov >
                ./coverage-report/coverage.info
            ${DEPENDS_CLAUSE}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating code coverage report"
            VERBATIM
            USES_TERMINAL
        )
        # cmake-format: on
    endfunction()

    add_custom_coverage_target(code_coverage check)
    add_custom_coverage_target(unit-coverage unit-check)
    add_custom_coverage_target(integration-coverage integration-check)

    # Alias target
    add_custom_target(coverage DEPENDS code_coverage COMMENT "Alias for code_coverage")

    # Standalone coverage report from existing profdata (does not run tests)
    add_custom_coverage_target(current-coverage "")
endif()

# ============================================================================
# Install
# ============================================================================
install(TARGETS hip_kernel_provider EXPORT hip-kernel-provider
        LIBRARY DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
        RUNTIME DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
)

if(ENABLE_ASM_SDPA_ENGINE)
    install(DIRECTORY src/engines/asm_sdpa_engine/asm/asm_kernels/
            DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/hip_kernel_provider/asm_kernels
            FILES_MATCHING PATTERN "*.co")
endif()

if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    # The two descriptor trees are installed here and nowhere else. Every producer --
    # the packer, or a step added later -- delivers content into one of these two
    # build-tree folders, and these two rules ship them wholesale. Each installed
    # descriptor tree is then a copy of its build-tree folder, so a test that passes
    # against one is evidence about the other, and a set cannot be staged for testing
    # yet silently missing from the package.
    #
    # This holds for these two trees only. Other content under the engine directory
    # installs from the source tree, so it has no build-tree counterpart at the same
    # relative path.
    #
    # Neither rule is OPTIONAL: each is emitted only when something was wired to fill its
    # tree, so a directory missing at install time means the build did not run to
    # completion, and saying so beats shipping a package with no descriptors in it.
    #
    # hkp_packaging_product exists exactly when the production packer will write -- the
    # production source root is set and is a real directory. Re-testing the source root
    # here would emit the rule for configurations the packer skips, turning a supported
    # build into an install-time failure.
    # Each pack marks its output root complete with a stamp file inside it, so that
    # emptying the tree invalidates the stamp too. The stamp belongs to the build and not
    # to the package, and every pack writes the same name, so one pattern drops all of
    # them from both trees below and from any pack added later.
    #
    # Asserted rather than defaulted: an empty name here would install a pattern that
    # matches nothing, and the stamp would ship with no diagnostic.
    if(NOT HKP_PACK_STAMP_NAME)
        message(FATAL_ERROR
                "hkp: HKP_PACK_STAMP_NAME is empty, so the descriptor install rules "
                "cannot exclude the pack stamps. descriptor-packaging defines it; this "
                "means it was not added before these rules.")
    endif()

    if(TARGET hkp_packaging_product)
        install(DIRECTORY "${HIPKERNELPROVIDER_DESCRIPTOR_BUILD_DIR}/"
                DESTINATION "${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/${HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR}"
                PATTERN "${HKP_PACK_STAMP_NAME}" EXCLUDE)
    endif()

    # The test packing roots are wired unconditionally under this gate, so the test tree
    # always has content to ship once the build has run.
    if(HIPKERNELPROVIDER_ENABLE_TESTS)
        install(DIRECTORY "${HIPKERNELPROVIDER_TEST_DESCRIPTOR_BUILD_DIR}/"
                DESTINATION "${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/${HIPKERNELPROVIDER_TEST_DESCRIPTOR_SUBDIR}"
                PATTERN "${HKP_PACK_STAMP_NAME}" EXCLUDE)
    endif()
endif()

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

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

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
    set(HIPDNN_PLUGIN_ENGINE_ROOTDIR "bin")
else()
    set(HIPDNN_PLUGIN_ENGINE_ROOTDIR "lib")
endif()

set(HIPDNN_BUILD_PLUGIN_ENGINE_DIR
    "${CMAKE_BINARY_DIR}/${HIPDNN_PLUGIN_ENGINE_ROOTDIR}/${HIPDNN_PLUGIN_ENGINE_SUBDIR}"
)

message(STATUS "Plugin build directory: ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}")
message(STATUS "Plugin install directory: ${HIPDNN_INSTALL_PLUGIN_ENGINE_DIR}")
file(MAKE_DIRECTORY ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR})

if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    # One spelling of the arch_content layout, shared by the staging rule, the install
    # DESTINATION and the path compiled into the provider. A typo in a second copy would
    # yield a provider that finds no descriptors, with no build error to say so.
    #
    # The provider's own directory is the root the loader walks; what sits under it is
    # organizational. Today that is one folder per engine, and per-arch shards land as
    # arch_content/hip-kernel-provider/<arch>/ once the packager emits them -- which needs
    # no change here, since folder names mean nothing to the walk.
    set(HIPDNN_DESCRIPTOR_SUBDIR "arch_content/hip-kernel-provider")
    set(HIPDNN_DESCRIPTOR_BUILD_DIR
        "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}/${HIPDNN_DESCRIPTOR_SUBDIR}"
    )
    # No HIPDNN_DESCRIPTOR_DIR is published for the suites: the integration binary loads
    # the plugin and resolves the tree from that module, and the unit binary compiles this
    # path in and applies it from main(). A CTest ENVIRONMENT entry would reach the
    # install-time file too, naming a build directory that the installed tests cannot see.
    message(STATUS "Kernel ingestor enabled: descriptors staged in ${HIPDNN_DESCRIPTOR_BUILD_DIR}")
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
    # Produces the pointwise_model pack's .uhd.fb and its model artifact, which the
    # descriptor staging in src/ then copies alongside that pack's JSON. Ahead of src/ so
    # the target exists to depend on.
    add_subdirectory(tools/uhd_model_gen)
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
    set(HIPDNN_DESCRIPTOR_SOURCE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/src/engines/kernel_ingestor_engine/descriptors")

    # Listed explicitly for the same reason the kernel sources beside them are: CMake's own
    # docs discourage GLOB for build inputs, and CONFIGURE_DEPENDS does not re-run on
    # removal under every generator. One list feeds staging, install, and the dependency
    # edge below, so the three cannot disagree about what a descriptor tree contains.
    set(HIPDNN_DESCRIPTOR_FILES
        conv_fwd/conv_fwd.kdp.json
        conv_fwd/conv_fwd.kmd.json
        conv_fwd/conv_fwd.udd.json
        conv_fwd/conv_fwd.ued.json
        conv_fwd/conv_fwd.uhd.json
        conv_fwd/conv_fwd_f16_block64.ukd.json
        conv_fwd/kernel_dtype_matches_graph.umd.json
        pointwise/kernel_dtype_matches_graph.umd.json
        pointwise/operation_is_add.umd.json
        pointwise/operation_is_mul.umd.json
        pointwise/operation_is_sub.umd.json
        pointwise/pointwise.kmd.json
        pointwise/pointwise.udd.json
        pointwise/pointwise.ued.json
        pointwise/pointwise.uhd.json
        pointwise/pointwise_add.kdp.json
        pointwise/pointwise_mul.kdp.json
        pointwise/pointwise_sub.kdp.json
        pointwise_model/pointwise_model.kmd.json
        pointwise_model/pointwise_model.ued.json
        pointwise_model/pointwise_model.uhd.json
        pointwise_model/pointwise_model_add.kdp.json
    )

    # Wiped once per configure, not once per build. copy_if_different below makes an
    # unchanged build a no-op, but it cannot retract a descriptor deleted from the source
    # tree: dropping a file from the list orphans its staged copy rather than removing it,
    # and the loader would keep reading a descriptor the sources no longer define. Editing
    # the list is what re-runs CMake, so clearing here is exactly as reliable as the old
    # per-build rm -rf, without leaving the directory empty while a test reads it.
    file(REMOVE_RECURSE "${HIPDNN_DESCRIPTOR_BUILD_DIR}")

    set(_staged_descriptors)
    foreach(_descriptor IN LISTS HIPDNN_DESCRIPTOR_FILES)
        add_custom_command(
            OUTPUT "${HIPDNN_DESCRIPTOR_BUILD_DIR}/${_descriptor}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${HIPDNN_DESCRIPTOR_SOURCE_DIR}/${_descriptor}"
                    "${HIPDNN_DESCRIPTOR_BUILD_DIR}/${_descriptor}"
            DEPENDS "${HIPDNN_DESCRIPTOR_SOURCE_DIR}/${_descriptor}"
            COMMENT "Staging descriptor ${_descriptor}"
            VERBATIM
        )
        list(APPEND _staged_descriptors "${HIPDNN_DESCRIPTOR_BUILD_DIR}/${_descriptor}")
    endforeach()

    # The pointwise_model pack's heuristic is a trained artifact rather than a symbol, so
    # two of its files are binary and cannot be staged by copying from the source tree.
    # They are generated instead -- see tools/uhd_model_gen for why they are not committed.
    # Listed as outputs of one command because a single run writes both, and both are
    # inputs to the staging target below, so a test target that pulls the descriptors in
    # gets them too.
    set(_generated_model_artifacts
        "${HIPDNN_DESCRIPTOR_BUILD_DIR}/pointwise_model/pointwise_model.uhd.fb"
        "${HIPDNN_DESCRIPTOR_BUILD_DIR}/pointwise_model/pointwise_model.bin"
    )
    add_custom_command(
        OUTPUT ${_generated_model_artifacts}
        COMMAND uhd_model_gen "${HIPDNN_DESCRIPTOR_BUILD_DIR}/pointwise_model"
        DEPENDS uhd_model_gen
        COMMENT "Generating pointwise_model UHD and model artifact"
        VERBATIM
    )
    list(APPEND _staged_descriptors ${_generated_model_artifacts})

    add_custom_target(
        hip_kernel_provider_descriptors ALL
        DEPENDS ${_staged_descriptors}
        COMMENT "Staging hip-kernel-provider descriptors"
    )

    # Both suites read the staged tree, but the target above is reachable only through ALL.
    # Building a test target by name on a fresh tree -- `ninja hip_kernel_provider_tests &&
    # ctest`, which is what `ninja check` does -- would otherwise run against a directory
    # that does not exist yet and quietly find no descriptors. CI escapes this only because
    # it builds the default target.
    foreach(_descriptor_consumer hip_kernel_provider_tests hip_kernel_provider_integration_tests)
        if(TARGET ${_descriptor_consumer})
            add_dependencies(${_descriptor_consumer} hip_kernel_provider_descriptors)
        endif()
    endforeach()

    # The same list again rather than a directory pattern, so the installed tree is the
    # staged tree by construction -- a file the build serves and the install omits would
    # otherwise be a difference nothing checks.
    foreach(_descriptor IN LISTS HIPDNN_DESCRIPTOR_FILES)
        get_filename_component(_descriptor_subdir "${_descriptor}" DIRECTORY)
        install(
            FILES "${HIPDNN_DESCRIPTOR_SOURCE_DIR}/${_descriptor}"
            DESTINATION
                "${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/${HIPDNN_DESCRIPTOR_SUBDIR}/${_descriptor_subdir}"
        )
    endforeach()

    # The generated pair keeps that same invariant, but has to come from the build tree:
    # it has no source-tree copy to install. Omitting it would install a pointwise_model
    # UHD whose artifact is missing, which degrades to declared order at run time and says
    # so only in a log -- exactly the silent difference the loop above exists to prevent.
    install(
        FILES ${_generated_model_artifacts}
        DESTINATION
            "${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/${HIPDNN_DESCRIPTOR_SUBDIR}/pointwise_model"
    )
endif()

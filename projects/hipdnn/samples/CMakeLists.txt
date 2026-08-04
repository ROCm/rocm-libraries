# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

# Enable PIC/PIE to ensure compatibility with the plugin loader system (dlopen). This prevents
# potential Thread Local Storage (TLS) model mismatches between the executable and dynamically
# loaded backend plugins.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(hipdnn_samples VERSION 0.1.0 LANGUAGES C CXX)
include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 17)

# Sanitizer support - inherit from parent build when BUILD_ADDRESS_SANITIZER is set
option(BUILD_ADDRESS_SANITIZER "Build with Address Sanitizer enabled" OFF)
option(BUILD_THREAD_SANITIZER "Build with Thread Sanitizer enabled" OFF)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Sanitizers.cmake)

option(ENABLE_CLANG_TIDY "Enable Clang-Tidy checks" ON)
if(NOT ENABLE_CLANG_TIDY)
    message(WARNING "Clang-Tidy is disabled - some build issues may be missed.")
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ClangTidy.cmake)

set(HIPDNN_SAMPLE_NO_RTTI_OPTIONS
    $<$<CXX_COMPILER_ID:MSVC>:/GR->
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fno-rtti>
)

find_package(hip REQUIRED)
find_package(Threads REQUIRED)

# Technically this is all found as part of hipdnn_test_sdk, but including these as an example for
# consumers. hipdnn_test_sdk shouldn't be needed by normal users of hipDNN, and is only incuded here
# for validation purposes.
if(NOT TARGET hipdnn_frontend)
    find_package(hipdnn_frontend CONFIG REQUIRED)
endif()
if(NOT TARGET hipdnn_flatbuffers_sdk)
    find_package(hipdnn_flatbuffers_sdk CONFIG REQUIRED)
endif()

# !! NOTE !! - hipdnn_test_sdk is needed for sample validation, regular users of hipDNN do not need
# this package.
if(NOT TARGET hipdnn_test_sdk)
    find_package(hipdnn_test_sdk CONFIG REQUIRED)
endif()

# All sample targets are accumulated into this list for the umbrella build target
set(HIPDNN_ALL_SAMPLES "")

# Samples favor the explicit `config.<field>.size() > N` form over the .empty()
# idiom so the parallel dimension-parsing blocks read uniformly. Append a
# command-line check override to the clang-tidy properties set by
# clang_tidy_check(); --checks= extends the -config-file= checks list, and the
# leading '-' disables that single check for sample targets only.
function(disable_container_size_empty_check TARGET)
    foreach(_prop CXX_CLANG_TIDY C_CLANG_TIDY HIP_CLANG_TIDY)
        get_target_property(_cmd ${TARGET} ${_prop})
        if(_cmd)
            list(APPEND _cmd "--checks=-readability-container-size-empty")
            set_target_properties(${TARGET} PROPERTIES ${_prop} "${_cmd}")
        endif()
    endforeach()
endfunction()

# Helper macro to add a hipDNN sample executable NAME: The name of the executable target SOURCE:
# The source file for the sample
macro(add_hipdnn_sample NAME SOURCE)
    add_executable(${NAME} ${SOURCE})
    # !! NOTE !! - hipdnn_test_sdk is needed for sample validation, regular users of hipDNN do not
    # need this package. Linking against hipdnn_frontend would be sufficient for normal users
    # (brings in hipdnn_flatbuffers_sdk as well).
    target_link_libraries(${NAME} PRIVATE hip::host Threads::Threads hipdnn_frontend hipdnn_test_sdk)
    target_compile_options(${NAME} PRIVATE ${HIPDNN_SAMPLE_NO_RTTI_OPTIONS})
    set_target_properties(
        ${NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
    )
    clang_tidy_check(${NAME})
    disable_container_size_empty_check(${NAME})
    list(APPEND HIPDNN_ALL_SAMPLES ${NAME})
endmacro()

# Dynamic backend-loading sample. Links ONLY hipdnn_frontend_dynamic --
# deliberately NOT hipdnn_test_sdk, which links the direct hipdnn_frontend and
# would pull libhipdnn_backend back onto the link line.
add_executable(hipdnn_sample_dynamic_backend_loading dynamic/DynamicBackendLoading.cpp)
target_link_libraries(hipdnn_sample_dynamic_backend_loading
    PRIVATE Threads::Threads hipdnn_frontend_dynamic)
target_compile_options(hipdnn_sample_dynamic_backend_loading PRIVATE ${HIPDNN_SAMPLE_NO_RTTI_OPTIONS})
set_target_properties(hipdnn_sample_dynamic_backend_loading PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
)
# The dynamic sample does not link libhipdnn_backend. In build trees, point its
# runtime search path at the backend target directory. In installed Linux/ELF
# layouts, keep the search relative from bin/ to lib/.
set_target_properties(hipdnn_sample_dynamic_backend_loading PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
    BUILD_RPATH "$<TARGET_FILE_DIR:hipdnn_backend>"
)
clang_tidy_check(hipdnn_sample_dynamic_backend_loading)
disable_container_size_empty_check(hipdnn_sample_dynamic_backend_loading)
list(APPEND HIPDNN_ALL_SAMPLES hipdnn_sample_dynamic_backend_loading)

add_hipdnn_sample(hipdnn_sample_bn_inference batchnorm/BnInference.cpp)
add_hipdnn_sample(hipdnn_sample_bn_inference_with_variance batchnorm/BnInferenceWithVariance.cpp)
add_hipdnn_sample(hipdnn_sample_bn_training batchnorm/BnTraining.cpp)
add_hipdnn_sample(hipdnn_sample_fused_bn_training_activ batchnorm/FusedBnTrainingActiv.cpp)
add_hipdnn_sample(hipdnn_sample_bn_backward batchnorm/BnBackward.cpp)
add_hipdnn_sample(hipdnn_sample_fused_bn_inference_activ batchnorm/FusedBnInferenceActiv.cpp)
add_hipdnn_sample(hipdnn_sample_fused_bn_inference_variance_activ batchnorm/FusedBnInferenceVarianceActiv.cpp)
add_hipdnn_sample(hipdnn_sample_fused_bn_inference_drelu_bn_backward batchnorm/FusedBnInfDReluBnBwd.cpp)

add_hipdnn_sample(hipdnn_sample_conv_dgrad convolution/ConvDgrad.cpp)
add_hipdnn_sample(hipdnn_sample_conv_fprop convolution/ConvFprop.cpp)
add_hipdnn_sample(hipdnn_sample_fused_conv_fprop_activ convolution/FusedConvFpropActiv.cpp)
add_hipdnn_sample(hipdnn_sample_fused_conv_fprop_bias_activ convolution/FusedConvFpropBiasActiv.cpp)
add_hipdnn_sample(hipdnn_sample_conv_wgrad convolution/ConvWgrad.cpp)
add_hipdnn_sample(hipdnn_sample_conv_fprop_deterministic convolution/ConvFpropDeterministic.cpp)

if(HIPDNN_ENABLE_SDPA)
    add_hipdnn_sample(hipdnn_sample_sdpa_fprop sdpa/SdpaFprop.cpp)
endif()

add_hipdnn_sample(hipdnn_sample_serialization_roundtrip serialization/SerializationRoundTrip.cpp)

if(HIPDNN_ENABLE_CUDNN_COMPATIBILITY AND HIPDNN_ENABLE_SDPA)
    add_hipdnn_sample(
        hipdnn_sample_cudnn_shim_sdpa_round_trip
        cudnn_shim/SdpaRoundTrip.cpp
    )
endif()

# Node-coverage samples mirror the upstream cuDNN FE sample names.
# Host-only: they build and validate the graph, no device required.
if(HIPDNN_ENABLE_CUDNN_COMPATIBILITY)
    add_hipdnn_sample(hipdnn_sample_cudnn_shim_conv_fprop cudnn_shim/ConvFprop.cpp)
    add_hipdnn_sample(hipdnn_sample_cudnn_shim_matmul cudnn_shim/Matmul.cpp)
    add_hipdnn_sample(hipdnn_sample_cudnn_shim_layernorm cudnn_shim/Layernorm.cpp)
    add_hipdnn_sample(hipdnn_sample_cudnn_shim_pointwise cudnn_shim/Pointwise.cpp)
endif()

add_hipdnn_sample(hipdnn_sample_knobs_usage knobs/KnobsUsage.cpp)

add_hipdnn_sample(hipdnn_sample_autotune autotune/AutotuneSample.cpp)

# Enable testing and add sample tests
enable_testing()

# On ASAN builds (HIPDNN_TEST_MIOPEN_CACHE_DIR set by Sanitizers.cmake), a fixture wipes the build-local
# MIOpen cache once per ctest run; samples opt in via FIXTURES_REQUIRED below. The GLOBAL-property
# guard defines it once across the build tree (shared with the hipdnn/provider Tests.cmake
# registrations).
# A scope block keeps the intermediate path variables out of the including project's scope; the
# fixture registration (add_test / set_property GLOBAL) is not variable-scoped, so it escapes.
block(SCOPE_FOR VARIABLES)
    get_property(_fixture_added GLOBAL PROPERTY _hipdnn_clear_miopen_test_cache_fixture_added)
    if(DEFINED HIPDNN_TEST_MIOPEN_CACHE_DIR AND NOT _fixture_added)
        # Safety: this path is baked into a generated `rm -rf`, so require it to be strictly under the build tree.
        get_filename_component(_cache_dir_abs "${HIPDNN_TEST_MIOPEN_CACHE_DIR}" ABSOLUTE)
        get_filename_component(_binary_dir_abs "${CMAKE_BINARY_DIR}" ABSOLUTE)
        string(FIND "${_cache_dir_abs}" "${_binary_dir_abs}/" _binary_dir_prefix_pos)
        if(NOT _binary_dir_prefix_pos EQUAL 0)
            message(FATAL_ERROR
                "HIPDNN_TEST_MIOPEN_CACHE_DIR ('${HIPDNN_TEST_MIOPEN_CACHE_DIR}') must be a subdirectory "
                "of the build tree ('${CMAKE_BINARY_DIR}'); abort during hipdnn miopen cache clear fixture registration.")
        endif()
        set_property(GLOBAL PROPERTY _hipdnn_clear_miopen_test_cache_fixture_added TRUE)
        add_test(NAME hipdnn_clear_miopen_test_cache
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${HIPDNN_TEST_MIOPEN_CACHE_DIR}")
        set_tests_properties(hipdnn_clear_miopen_test_cache
            PROPERTIES FIXTURES_SETUP hipdnn_clear_miopen_test_cache)
    endif()
endblock()

set(HIPDNN_SAMPLES_CTEST_FILE_INSTALL_PATH "${CMAKE_INSTALL_BINDIR}/hipdnn_samples")
# Generate a new CTestTestfile.cmake that references installed test executables
set(INSTALLED_CTEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake.install")

file(WRITE "${INSTALLED_CTEST_FILE}"
        "# Autogenerated CTestTestfile for installed hipDNN tests\n"
)
file(APPEND "${INSTALLED_CTEST_FILE}" "# Generated by hipDNN build system\n\n")

# Helper function to add a hipDNN sample test
function(add_hipdnn_sample_test TARGET_NAME)
    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME} --verify-cpu)
    set_property(GLOBAL APPEND PROPERTY HIPDNN_SAMPLE_TEST_TARGETS ${TARGET_NAME})

    if(TEST_ENVIRONMENT)
        set_tests_properties(${TARGET_NAME} PROPERTIES ENVIRONMENT "${TEST_ENVIRONMENT}")
    endif()
    # PATH prepends (Windows ASAN runtime / ROCm / build DLL dirs) via ENVIRONMENT_MODIFICATION so
    # the runtime PATH is extended, not replaced.
    if(TEST_ENVIRONMENT_MODIFICATION)
        set_tests_properties(${TARGET_NAME} PROPERTIES
            ENVIRONMENT_MODIFICATION "${TEST_ENVIRONMENT_MODIFICATION}")
    endif()
    # Clear the build-local MIOpen cache before this sample runs (ASAN builds only).
    if(HIPDNN_TEST_MIOPEN_CACHE_DIR)
        set_tests_properties(${TARGET_NAME} PROPERTIES
            FIXTURES_REQUIRED hipdnn_clear_miopen_test_cache)
    endif()

    set_target_properties(
        ${TARGET_NAME}
        PROPERTIES
            INSTALL_RPATH_USE_LINK_PATH TRUE
            BUILD_RPATH_USE_ORIGIN TRUE
    )

    install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    # On Windows, stage the shadowed ROCm DLLs before this sample is built so a partial build
    # (`cmake --build --target ${TARGET_NAME}`) doesn't load the stale System32 amd_comgr.dll.
    if(TARGET stage_shadowed_rocm_dlls)
        add_dependencies(${TARGET_NAME} stage_shadowed_rocm_dlls)
    endif()

    file(APPEND "${INSTALLED_CTEST_FILE}" "add_test(${TARGET_NAME} \"../${TARGET_NAME}\" \"--verify-cpu\")\n")
endfunction()

# Register convolution samples as tests
add_hipdnn_sample_test(hipdnn_sample_conv_fprop)
add_hipdnn_sample_test(hipdnn_sample_conv_dgrad)
add_hipdnn_sample_test(hipdnn_sample_conv_wgrad)

if(NOT WIN32)
    # Some fused conv samples are not supported on windows due to lack of CK support
    add_hipdnn_sample_test(hipdnn_sample_fused_conv_fprop_activ)
    add_hipdnn_sample_test(hipdnn_sample_fused_conv_fprop_bias_activ)
    add_hipdnn_sample_test(hipdnn_sample_conv_fprop_deterministic)
endif()

# Register batchnorm samples as tests
add_hipdnn_sample_test(hipdnn_sample_bn_inference)
add_hipdnn_sample_test(hipdnn_sample_bn_inference_with_variance)
add_hipdnn_sample_test(hipdnn_sample_bn_training)
add_hipdnn_sample_test(hipdnn_sample_fused_bn_training_activ)
add_hipdnn_sample_test(hipdnn_sample_bn_backward)
add_hipdnn_sample_test(hipdnn_sample_fused_bn_inference_activ)
add_hipdnn_sample_test(hipdnn_sample_fused_bn_inference_drelu_bn_backward)
add_hipdnn_sample_test(hipdnn_sample_fused_bn_inference_variance_activ)

# Register SDPA samples as tests
# Note: SDPA sample test is disabled until the ASM SDPA engine's isApplicable
# check properly gates on architecture, data type, and head dimension.
# add_hipdnn_sample_test(hipdnn_sample_sdpa_fprop)

# Register serialization samples as tests
add_hipdnn_sample_test(hipdnn_sample_serialization_roundtrip)

if(HIPDNN_ENABLE_CUDNN_COMPATIBILITY AND HIPDNN_ENABLE_SDPA)
    add_hipdnn_sample_test(hipdnn_sample_cudnn_shim_sdpa_round_trip)
endif()

if(HIPDNN_ENABLE_CUDNN_COMPATIBILITY)
    add_hipdnn_sample_test(hipdnn_sample_cudnn_shim_conv_fprop)
    add_hipdnn_sample_test(hipdnn_sample_cudnn_shim_matmul)
    add_hipdnn_sample_test(hipdnn_sample_cudnn_shim_layernorm)
    add_hipdnn_sample_test(hipdnn_sample_cudnn_shim_pointwise)
endif()

# Register knobs sample
# Note: knobs_usage is informational/educational, not validation
install(TARGETS hipdnn_sample_knobs_usage RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Register dynamic backend-loading sample
add_hipdnn_sample_test(hipdnn_sample_dynamic_backend_loading)
# Under ASAN the dlopen interceptor ignores the executable RUNPATH, so
# the bare-name backend .so is only found via LD_LIBRARY_PATH.
if(BUILD_ADDRESS_SANITIZER OR THEROCK_SANITIZER STREQUAL "ASAN")
    set_property(TEST hipdnn_sample_dynamic_backend_loading APPEND PROPERTY
        ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:hipdnn_backend>")
endif()

# Register autotune sample
add_hipdnn_sample_test(hipdnn_sample_autotune)

# Umbrella target for superbuild dependency tracking
add_custom_target(hipdnn_samples COMMENT "Building all hipDNN samples" DEPENDS ${HIPDNN_ALL_SAMPLES})

# Disable convolution samples under ASAN — known driver-level failures in MIOpen/CK
# and Tensile that are unrelated to hipDNN can cause test hangs. Re-enable when
# underlying issues are resolved. (SKIP_IF_ASAN())
if(BUILD_ADDRESS_SANITIZER)
    set_tests_properties(
        hipdnn_sample_conv_fprop
        hipdnn_sample_conv_dgrad
        hipdnn_sample_conv_wgrad
        hipdnn_sample_serialization_roundtrip
        PROPERTIES DISABLED TRUE
    )
    # The fused-conv sample tests are only registered on non-Windows (see above),
    # so only disable them there — otherwise set_tests_properties aborts configure.
    if(NOT WIN32)
        set_tests_properties(
            hipdnn_sample_fused_conv_fprop_activ
            hipdnn_sample_fused_conv_fprop_bias_activ
            hipdnn_sample_conv_fprop_deterministic
            PROPERTIES DISABLED TRUE
        )
    endif()
endif()

# Optional YAML-driven CTest categorization. Guarded so standalone /
# sparse checkouts without shared/ctest still build (falls back to unlabeled
# install). apply_ctest_category_labels() scans the add_test() names already
# written into INSTALLED_CTEST_FILE and appends per-test
# set_tests_properties(... LABELS ...) lines, so `ctest -L quick` works from
# the installed bin/hipdnn_samples/ tree that TheRock CI consumes.
if(NOT DEFINED ROCM_LIBRARIES_ROOT)
    set(ROCM_LIBRARIES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml"
   AND EXISTS "${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
    include("${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
    get_property(_hipdnn_sample_test_targets GLOBAL PROPERTY HIPDNN_SAMPLE_TEST_TARGETS)
    foreach(_hipdnn_sample_test IN LISTS _hipdnn_sample_test_targets)
        apply_ctest_category_labels(
            "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml"
            EXPLICIT_TESTS ${_hipdnn_sample_test}
        )
    endforeach()
    apply_ctest_category_labels(
        "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml"
        "${INSTALLED_CTEST_FILE}"
    )
endif()

# Install the generated CTestTestfile.cmake to HIPDNN_SAMPLES_CTEST_FILE_INSTALL_PATH
install(FILES "${INSTALLED_CTEST_FILE}" DESTINATION ${HIPDNN_SAMPLES_CTEST_FILE_INSTALL_PATH}
        RENAME CTestTestfile.cmake
)

# =============================================================================
# Example engine plugin — built as an external project to preserve its
# standalone build capability. In superbuild mode, hipdnn is installed to a
# staging directory, matching the environment that the standalone project
# will be used in and enabling find_package() to work correctly.
# Tests are forwarded to this project's CTest.
# =============================================================================
find_package(hiprtc REQUIRED)
include(ExternalProject)

set(EXAMPLE_ENGINE_PLUGIN_CLANG_TIDY_CACHE_ARGS "")
set(EXAMPLE_ENGINE_PLUGIN_CLANG_TIDY_CMAKE_ARGS "")
message(VERBOSE "ENABLE_CLANG_TIDY=${ENABLE_CLANG_TIDY} CLANG_TIDY_EXE=${CLANG_TIDY_EXE}")
if(ENABLE_CLANG_TIDY AND CLANG_TIDY_EXE)
    list(APPEND EXAMPLE_ENGINE_PLUGIN_CLANG_TIDY_CACHE_ARGS
        "-DCMAKE_CXX_CLANG_TIDY:STRING=${CLANG_TIDY_EXE}"
    )
else()
    list(APPEND EXAMPLE_ENGINE_PLUGIN_CLANG_TIDY_CMAKE_ARGS
        "-UCMAKE_CXX_CLANG_TIDY"
        "-UCLANG_TIDY_EXE"
    )
endif()

# Force the example plugin external project to reconfigure whenever the samples project is reconfigured
file(REMOVE
    "${CMAKE_CURRENT_BINARY_DIR}/example_engine_plugin-prefix/src/example_engine_plugin-stamp/example_engine_plugin-configure"
)

# Build CMAKE_PREFIX_PATH for the example plugin external project
set(_example_plugin_prefix_path "${CMAKE_PREFIX_PATH}")
set(_example_plugin_depends "")

if(ROCM_LIBS_SUPERBUILD)
    # In superbuild mode, install hipdnn to a staging directory so the plugin
    # can use find_package() as intended for standalone builds.
    set(EXAMPLE_PLUGIN_HIPDNN_STAGING_DIR "${CMAKE_BINARY_DIR}/hipdnn-staging")

    # Clear the staging prefix at the start of every configure to ensure the most
    # recent selected dependencies are used.
    file(REMOVE_RECURSE "${EXAMPLE_PLUGIN_HIPDNN_STAGING_DIR}")

    # Custom target to install hipdnn to staging directory for the example plugin project.
    # Only install Runtime and Development components - skip test binaries which may not be built.
    # This only runs when example_engine_plugin is built (due to DEPENDS).
    add_custom_target(example_plugin_hipdnn-staging-install
        COMMAND ${CMAKE_COMMAND} -E make_directory ${EXAMPLE_PLUGIN_HIPDNN_STAGING_DIR}
        COMMAND ${CMAKE_COMMAND}
                -DBUILD_DIR=${CMAKE_BINARY_DIR}/projects/hipdnn
                -DPREFIX=${EXAMPLE_PLUGIN_HIPDNN_STAGING_DIR}
                -DCOMPONENT=Runtime
                -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/scripts/quiet_install.cmake
        COMMAND ${CMAKE_COMMAND}
                -DBUILD_DIR=${CMAKE_BINARY_DIR}/projects/hipdnn
                -DPREFIX=${EXAMPLE_PLUGIN_HIPDNN_STAGING_DIR}
                -DCOMPONENT=Development
                -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/scripts/quiet_install.cmake
        DEPENDS hipdnn_backend hipdnn_data_sdk hipdnn_plugin_sdk hipdnn_flatbuffers_sdk hipdnn_frontend
        COMMENT "Installing hipDNN to staging directory for example_engine_plugin"
    )

    # Stage the workspace's fetched nlohmann_json only when it was FETCHED.
    include(FetchContent)
    FetchContent_GetProperties(nlohmann_json)
    string(FIND "${nlohmann_json_DIR}" "${CMAKE_BINARY_DIR}" _nlohmann_fetched_in_build_tree)
    if(nlohmann_json_DIR AND _nlohmann_fetched_in_build_tree EQUAL 0)
        add_custom_command(
            TARGET example_plugin_hipdnn-staging-install
            POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                    -DBUILD_DIR=${nlohmann_json_BINARY_DIR}
                    -DPREFIX=${EXAMPLE_PLUGIN_HIPDNN_STAGING_DIR}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/scripts/quiet_install.cmake
            COMMENT "Staging fetched nlohmann_json for example_engine_plugin"
        )
    endif()

    list(PREPEND _example_plugin_prefix_path "${EXAMPLE_PLUGIN_HIPDNN_STAGING_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake")
    list(PREPEND _example_plugin_prefix_path "${EXAMPLE_PLUGIN_HIPDNN_STAGING_DIR}")
    list(APPEND _example_plugin_depends example_plugin_hipdnn-staging-install)
endif()

ExternalProject_Add(example_engine_plugin
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/example_engine_plugin
    BUILD_ALWAYS TRUE
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD TRUE
    CMAKE_CACHE_ARGS
        "-DCMAKE_MESSAGE_LOG_LEVEL:STRING=${CMAKE_MESSAGE_LOG_LEVEL}"
        "-DCMAKE_PREFIX_PATH:STRING=${_example_plugin_prefix_path}"
        "-DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}"
        "-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}"
        "-DCMAKE_RC_COMPILER:FILEPATH=${CMAKE_RC_COMPILER}"
        "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}"
        "-DCMAKE_MAKE_PROGRAM:FILEPATH=${CMAKE_MAKE_PROGRAM}"
        "-DEXAMPLEPROVIDER_BUILD_UNIT_TESTS:BOOL=ON"
        "-DEXAMPLEPROVIDER_BUILD_SAMPLE:BOOL=ON"
        "-DHIP_PLATFORM:STRING=${HIP_PLATFORM}"
        "-Dhip_DIR:PATH=${hip_DIR}"
        "-Dhiprtc_DIR:PATH=${hiprtc_DIR}"
        "-Dflatbuffers_DIR:PATH=${flatbuffers_DIR}"
        "-DBUILD_ADDRESS_SANITIZER:BOOL=${BUILD_ADDRESS_SANITIZER}"
        "-DTEST_ENVIRONMENT:STRING=${TEST_ENVIRONMENT}"
        "-DEXAMPLEPROVIDER_ENABLE_CLANG_TIDY:BOOL=${EXAMPLEPROVIDER_ENABLE_CLANG_TIDY}"
        ${EXAMPLE_ENGINE_PLUGIN_CLANG_TIDY_CACHE_ARGS}
    CMAKE_ARGS
        # Clear any cached nlohmann_json_DIR so find_package re-resolves it.
        "-Unlohmann_json_DIR"
        ${EXAMPLE_ENGINE_PLUGIN_CLANG_TIDY_CMAKE_ARGS}
    DEPENDS ${_example_plugin_depends}
    INSTALL_COMMAND ""
)

block(SCOPE_FOR VARIABLES)
    ExternalProject_Get_Property(example_engine_plugin BINARY_DIR)

    # Copy example provider executables and plugin DLL to main bin directory for DLL resolution on Windows
    add_custom_command(
        TARGET example_engine_plugin POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${BINARY_DIR}/bin/hipdnn_example_provider_unit_tests${CMAKE_EXECUTABLE_SUFFIX}"
            "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/hipdnn_example_provider_unit_tests${CMAKE_EXECUTABLE_SUFFIX}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${BINARY_DIR}/bin/hipdnn_example_provider_sample${CMAKE_EXECUTABLE_SUFFIX}"
            "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/hipdnn_example_provider_sample${CMAKE_EXECUTABLE_SUFFIX}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${BINARY_DIR}/bin/${CMAKE_SHARED_LIBRARY_PREFIX}example_provider_plugin${CMAKE_SHARED_LIBRARY_SUFFIX}"
            "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}example_provider_plugin${CMAKE_SHARED_LIBRARY_SUFFIX}"
        COMMENT "Copying example provider executables and plugin DLL to superbuild bin directory"
    )

    add_test(
        NAME hipdnn_example_provider_unit_tests
        COMMAND ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/hipdnn_example_provider_unit_tests
    )
    set_tests_properties(hipdnn_example_provider_unit_tests PROPERTIES LABELS "sample_test")

    add_test(
        NAME hipdnn_example_provider_sample
        COMMAND ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/hipdnn_example_provider_sample
    )
    set_tests_properties(hipdnn_example_provider_sample PROPERTIES LABELS "sample_test")

    if(TEST_ENVIRONMENT)
        set_tests_properties(hipdnn_example_provider_unit_tests hipdnn_example_provider_sample
            PROPERTIES ENVIRONMENT "${TEST_ENVIRONMENT}")
    endif()
    # PATH prepends (Windows ASAN runtime / ROCm / build DLL dirs) via ENVIRONMENT_MODIFICATION so
    # the runtime PATH is extended, not replaced, without ';' colliding with the list separator.
    if(TEST_ENVIRONMENT_MODIFICATION)
        set_tests_properties(hipdnn_example_provider_unit_tests hipdnn_example_provider_sample
            PROPERTIES ENVIRONMENT_MODIFICATION "${TEST_ENVIRONMENT_MODIFICATION}")
    endif()
    # Clear the build-local MIOpen cache before these run (ASAN builds only).
    if(HIPDNN_TEST_MIOPEN_CACHE_DIR)
        set_tests_properties(hipdnn_example_provider_unit_tests hipdnn_example_provider_sample
            PROPERTIES FIXTURES_REQUIRED hipdnn_clear_miopen_test_cache)
    endif()
endblock()

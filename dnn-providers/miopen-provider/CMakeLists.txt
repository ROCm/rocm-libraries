# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(NOT BUILD_PLUGIN_AS_DEPENDENCY)
    message(STATUS "Building miopen provider plugin standalone")
    cmake_minimum_required(VERSION 3.25.2)
    add_compile_definitions(__HIP_PLATFORM_AMD__)

    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/cmake/ClangToolChain.cmake"
        CACHE STRING "Toolchain file"
    )
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    set(MIOPEN_PLUGIN_VERSION "0.0.1")
    project(hipDNN VERSION ${MIOPEN_PLUGIN_VERSION} LANGUAGES CXX)

    if(NOT WIN32)
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

    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${ROCM_PATH}/lib/cmake/hip
         ${HIP_DIR}/cmake
    )

    # Enable testing for standalone build
    # Migrate HIP_DNN_SKIP_TESTS -> HIPDNN_SKIP_TESTS
    if(DEFINED HIP_DNN_SKIP_TESTS)
        set(HIPDNN_SKIP_TESTS ${HIP_DNN_SKIP_TESTS} CACHE BOOL "Skip building tests" FORCE)
        unset(HIP_DNN_SKIP_TESTS CACHE)
        message(DEPRECATION "HIP_DNN_SKIP_TESTS has been renamed to HIPDNN_SKIP_TESTS. Please update your build scripts to use -DHIPDNN_SKIP_TESTS instead.")
    endif()
    option(HIPDNN_SKIP_TESTS "Skip building tests" OFF)

    list(
        PREPEND
        HIPDNN_WARNING_COMPILE_OPTIONS
        -Werror # Treat all warnings as errors
        -Wall # Enable most common warnings
        -Wextra # Enable additional warnings not covered by -Wall
        -Wpedantic # Enforce strict ISO C++ compliance
        -Wshadow # Warn about variable shadowing
        -Wnon-virtual-dtor # Warn if a class with virtual functions has a non-virtual destructor
        -Wold-style-cast # Warn about C-style casts
        -Wcast-align # Warn about potential performance issues with misaligned casts
        -Woverloaded-virtual # Warn if a base class function is hidden by a derived class function
                             # with the same name
        -Wconversion # Warn about implicit type conversions that may alter a value
        -Wsign-conversion # Warn about implicit conversions between signed and unsigned types
        -Wnull-dereference # Warn about dereferencing null pointers
        -Wdouble-promotion # Warn when a float is implicitly promoted to a double
        -Wformat=2 # Enable stricter format string checks
        -Winit-self # Warn about variables initialized with itself
        -Wunreachable-code # Warn about unreachable code
        -Wno-error=unused-command-line-argument # Disable error for unused command line arguments
        -Wswitch-default # Warn if a switch statement does not have a default case
    )

    # NOTE: Workaround until llvm & hip cmake modules fixes symlink logic in their config files;
    # remove when fixed. Added to prevent issues with not finding hip cmake.
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm ${ROCM_PATH} ${ROCM_PATH}/hip)

    find_package(hip REQUIRED)
    find_package(hipdnn_data_sdk CONFIG REQUIRED)
    find_package(hipdnn_plugin_sdk CONFIG REQUIRED)

    # Setup testing dependencies & tests for standalone build
    include(Dependencies)
    include(Tests)

else()
    message(STATUS "Building miopen provider plugin as a dependency")
endif()

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

find_package(miopen CONFIG REQUIRED)

# OBJECT library with all implementation - shared by both target names
# This allows both miopen_legacy_plugin and miopen_plugin to be REAL targets
# (not aliases) so TheRock can use set_target_properties on either one.
add_library(
    miopen_plugin_impl OBJECT
    engines/MiopenEngine.cpp
    engines/plans/MiopenBatchnormApplicabilityChecks.cpp
    engines/plans/MiopenBatchnormPlanBuilder.cpp
    engines/plans/MiopenBatchnormBwdPlan.cpp
    engines/plans/MiopenBatchnormFwdInferencePlan.cpp
    engines/plans/MiopenBatchnormFwdInferenceWithVariancePlan.cpp
    engines/plans/MiopenBatchnormFwdTrainingPlan.cpp
    engines/plans/MiopenBatchnormFwdTrainingPlanBuilder.cpp
    engines/plans/MiopenConvBwdPlan.cpp
    engines/plans/MiopenConvFwdBiasActivPlan.cpp
    engines/plans/MiopenConvFwdBiasActivPlanBuilder.cpp
    engines/plans/MiopenConvFwdPlan.cpp
    engines/plans/MiopenConvPlanBuilder.cpp
    engines/plans/MiopenConvWrwPlan.cpp
    EngineManager.cpp
    MiopenActivationDescriptor.cpp
    MiopenHandleFactory.cpp
    MiopenContainer.cpp
    MiopenConvDescriptor.cpp
    MiopenUtils.cpp
    MiopenTensor.cpp
    MiopenPlugin.cpp
    MiopenPluginPublic.cpp
)
target_compile_definitions(
    miopen_plugin_impl PRIVATE COMPONENT_NAME="miopen_plugin"
)
target_include_directories(miopen_plugin_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(miopen_plugin_impl PUBLIC MIOpen hipdnn_data_sdk hipdnn_plugin_sdk)
target_compile_options(miopen_plugin_impl PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
set_target_properties(miopen_plugin_impl PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)

if(BUILD_PLUGIN_AS_DEPENDENCY)
    clang_tidy_check(miopen_plugin_impl)
endif()

# Static library for unit tests to link against (provides access to internal implementation)
add_library(miopen_plugin_private STATIC $<TARGET_OBJECTS:miopen_plugin_impl>)
target_compile_definitions(miopen_plugin_private PRIVATE COMPONENT_NAME="miopen_plugin")
target_include_directories(miopen_plugin_private PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(miopen_plugin_private PUBLIC MIOpen hipdnn_data_sdk hipdnn_plugin_sdk)
# Backwards compatibility alias
add_library(miopen_legacy_plugin_private ALIAS miopen_plugin_private)

# miopen_legacy_plugin - REAL target for current TheRock compatibility
add_library(miopen_legacy_plugin SHARED $<TARGET_OBJECTS:miopen_plugin_impl>)
target_compile_definitions(miopen_legacy_plugin PRIVATE COMPONENT_NAME="miopen_plugin")
target_link_libraries(miopen_legacy_plugin PRIVATE MIOpen hipdnn_data_sdk hipdnn_plugin_sdk)
target_link_libraries(miopen_legacy_plugin PRIVATE "-Xlinker --exclude-libs=ALL")
target_compile_options(miopen_legacy_plugin PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
set_target_properties(
    miopen_legacy_plugin
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               RUNTIME_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               ARCHIVE_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               INSTALL_RPATH "${ROCM_PATH}/lib"
               INSTALL_RPATH_USE_LINK_PATH TRUE
               BUILD_WITH_INSTALL_RPATH TRUE
)

# miopen_plugin - REAL target for updated TheRock
add_library(miopen_plugin SHARED $<TARGET_OBJECTS:miopen_plugin_impl>)
target_compile_definitions(miopen_plugin PRIVATE COMPONENT_NAME="miopen_plugin")
target_link_libraries(miopen_plugin PRIVATE MIOpen hipdnn_data_sdk hipdnn_plugin_sdk)
target_link_libraries(miopen_plugin PRIVATE "-Xlinker --exclude-libs=ALL")
target_compile_options(miopen_plugin PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
set_target_properties(
    miopen_plugin
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               RUNTIME_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               ARCHIVE_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               INSTALL_RPATH "${ROCM_PATH}/lib"
               INSTALL_RPATH_USE_LINK_PATH TRUE
               BUILD_WITH_INSTALL_RPATH TRUE
)

# Set RPATH origin for TheRock integration. The plugins are installed to
# lib/hipdnn_plugins/engines/ which is 2 levels below lib/, so TheRock needs
# this hint to calculate correct relative RPATHs.
# Note: TheRock's post_hook_MIOpen_plugin.cmake also sets this for miopen_legacy_plugin,
# but setting it here ensures both plugins work regardless of TheRock's state.
# TODO: remove once post hook in TheRock has replaced this.
set_target_properties(miopen_plugin miopen_legacy_plugin PROPERTIES
    THEROCK_INSTALL_RPATH_ORIGIN "lib/hipdnn_plugins/engines")

if(NOT HIPDNN_SKIP_TESTS)
    add_subdirectory(tests)
    add_subdirectory(integration_tests)

    if(NOT BUILD_PLUGIN_AS_DEPENDENCY)
        # Keep this after all build folders have been added so they have a chance to register their tests
        # using add_*_test_target()
        finalize_test_targets()

        # For standalone builds we need to install the ctest files
        install_miopen_plugin_ctest_files()
    endif()
endif()

# Install both library targets
# miopen_plugin is the canonical name going forward
install(TARGETS miopen_plugin EXPORT miopen-plugin
        LIBRARY DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
        RUNTIME DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
)

# miopen_legacy_plugin for current TheRock compatibility
install(TARGETS miopen_legacy_plugin EXPORT miopen-legacy-plugin
        LIBRARY DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
        RUNTIME DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
)

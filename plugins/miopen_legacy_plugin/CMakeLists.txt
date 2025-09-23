# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(NOT BUILD_PLUGIN_AS_DEPENDENCY)
    message(STATUS "Building miopen legacy plugin standalone")
    cmake_minimum_required(VERSION 3.25.2)
    add_compile_definitions(__HIP_PLATFORM_AMD__)
    
    if(NOT WIN32)
        set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "")
    endif()

    set(MIOPEN_LEGACY_PLUGIN "0.0.1")
    project(
        hipDNN
        VERSION ${MIOPEN_LEGACY_PLUGIN}
        LANGUAGES CXX)
        
    set(CMAKE_CXX_STANDARD 20)

    list(PREPEND HIPDNN_WARNING_COMPILE_OPTIONS
        -Werror                                  # Treat all warnings as errors
        -Wall                                    # Enable most common warnings
        -Wextra                                  # Enable additional warnings not covered by -Wall
        -Wpedantic                               # Enforce strict ISO C++ compliance
        -Wshadow                                 # Warn about variable shadowing
        -Wnon-virtual-dtor                       # Warn if a class with virtual functions has a non-virtual destructor
        -Wold-style-cast                         # Warn about C-style casts
        -Wcast-align                             # Warn about potential performance issues with misaligned casts
        -Woverloaded-virtual                     # Warn if a base class function is hidden by a derived class function with the same name
        -Wconversion                             # Warn about implicit type conversions that may alter a value
        -Wsign-conversion                        # Warn about implicit conversions between signed and unsigned types
        -Wnull-dereference                       # Warn about dereferencing null pointers
        -Wdouble-promotion                       # Warn when a float is implicitly promoted to a double
        -Wformat=2                               # Enable stricter format string checks
        -Winit-self                              # Warn about variables initialized with itself
        -Wunreachable-code                       # Warn about unreachable code
        -Wno-error=unused-command-line-argument  # Disable error for unused command line arguments
    )
    
    # NOTE: Workaround until llvm & hip cmake modules fixes symlink logic in their config files; remove when fixed.
    #       Added to prevent issues with not finding hip cmake.
    list(
        APPEND
        CMAKE_PREFIX_PATH
        ${ROCM_PATH}/llvm
        ${ROCM_PATH}
        ${ROCM_PATH}/hip
        /opt/rocm/llvm
        /opt/rocm
        /opt/rocm/hip 
    )
    
    find_package(hip REQUIRED)
    find_package(hipdnn_sdk CONFIG REQUIRED)
else()
    message(STATUS "Building miopen legacy plugin as a dependency")
endif()

find_package(miopen CONFIG REQUIRED)

add_library(miopen_legacy_plugin_private
    engines/MiopenEngine.cpp
    engines/plans/MiopenBatchnormPlanBuilder.cpp
    engines/plans/MiopenBatchnormBwdPlan.cpp
    engines/plans/MiopenBatchnormFwdInferencePlan.cpp
    engines/plans/MiopenConvFwdPlan.cpp
    engines/plans/MiopenConvPlanBuilder.cpp
    EngineManager.cpp
    MiopenHandleFactory.cpp
    MiopenContainer.cpp
    MiopenConvDescriptor.cpp
    MiopenUtils.cpp
    MiopenTensor.cpp
)
target_compile_definitions(miopen_legacy_plugin_private PRIVATE 
    COMPONENT_NAME="miopen_legacy_plugin"
)
target_include_directories(miopen_legacy_plugin_private 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(miopen_legacy_plugin_private PUBLIC MIOpen hipdnn_sdk)
target_compile_options(miopen_legacy_plugin_private PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

clang_tidy_check(miopen_legacy_plugin_private)


add_library(miopen_legacy_plugin SHARED
    MiopenLegacyPlugin.cpp
)
target_compile_definitions(miopen_legacy_plugin PRIVATE 
    COMPONENT_NAME="miopen_legacy_plugin"
)
target_link_libraries(miopen_legacy_plugin PRIVATE miopen_legacy_plugin_private)
target_link_libraries(miopen_legacy_plugin PRIVATE "-Xlinker --exclude-libs=ALL") # Hide symbols from linked static libs
target_compile_options(miopen_legacy_plugin PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
set_target_properties(miopen_legacy_plugin PROPERTIES 
    CXX_VISIBILITY_PRESET hidden
    LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
)

clang_tidy_check(miopen_legacy_plugin)

set(MIOPEN_LEGACY_PLUGIN_PATH ./hipdnn_plugins/engines/miopen_legacy_plugin CACHE INTERNAL "miopen legacy plugin dir" FORCE)

add_subdirectory(tests)
add_subdirectory(integration_tests)

install(
    TARGETS miopen_legacy_plugin
    EXPORT miopen-legacy-plugin
    LIBRARY DESTINATION ${HIPDNN_INSTALL_PLUGIN_ENGINE_DIR}
)

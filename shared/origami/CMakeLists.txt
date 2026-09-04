# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)

project(Origami VERSION 1.0.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(ORIGAMI_STANDALONE ON)
else()
    set(ORIGAMI_STANDALONE OFF)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Set cpack license file here to avoid rocm-cmake warning
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")

include(dependencies)

rocm_setup_version(VERSION ${PROJECT_VERSION})

option(ORIGAMI_BUILD_SHARED_LIBS "Build shared libraries." ${ORIGAMI_STANDALONE})
option(ORIGAMI_ENABLE_PYTHON "Enable Python bindings." OFF)
option(ORIGAMI_BUILD_TESTING "Build tests." OFF)
option(ORIGAMI_ENABLE_FETCH "Auto-fetch dependencies with FetchContent" ON)
option(ORIGAMI_ENABLE_NN "Build NN inference stack (API, features, and weight layout)." OFF)

find_package(hip REQUIRED)

if(ORIGAMI_BUILD_SHARED_LIBS OR (BUILD_SHARED_LIBS AND ORIGAMI_STANDALONE))
    set(ORIGAMI_LIBRARY_TYPE SHARED)
else()
    set(ORIGAMI_LIBRARY_TYPE STATIC)
endif()
add_library(origami ${ORIGAMI_LIBRARY_TYPE})

rocm_set_soversion(origami "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

add_library(roc::origami ALIAS origami)

set_target_properties(origami PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_compile_features(origami PUBLIC cxx_std_17)

include(GenerateExportHeader)
generate_export_header(origami
    BASE_NAME ORIGAMI
    EXPORT_MACRO_NAME ORIGAMI_EXPORT
    EXPORT_FILE_NAME "${PROJECT_BINARY_DIR}/include/origami/origami_export.h"
    STATIC_DEFINE ORIGAMI_STATIC
)
if(ORIGAMI_LIBRARY_TYPE STREQUAL "STATIC")
    target_compile_definitions(origami PUBLIC ORIGAMI_STATIC)
endif()

add_library(origami-headers INTERFACE)
add_library(roc::origami-headers ALIAS origami-headers)

target_compile_features(origami-headers INTERFACE cxx_std_17)

target_sources(origami-headers
    INTERFACE
    FILE_SET HEADERS
    BASE_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${PROJECT_BINARY_DIR}/include"
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/attention.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/gemm.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/hardware.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/heuristics.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/logger.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/math.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/origami.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/streamk.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/types.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/nn/types.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/simulator/tensilelite/formocast.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/simulator/tensilelite/formocast_simulator.hpp"
        "${PROJECT_BINARY_DIR}/include/origami/origami_export.h"
)
target_include_directories(
    origami-headers INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(origami-headers INTERFACE hip::host)

if(ORIGAMI_ENABLE_NN)
    target_sources(origami-headers
        INTERFACE
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/nn/filter.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/nn/features/gemm_tilewright.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/nn/nn.hpp"
    )
    target_compile_definitions(origami-headers INTERFACE ORIGAMI_ENABLE_NN=1)
else()
    target_compile_definitions(origami-headers INTERFACE ORIGAMI_ENABLE_NN=0)
endif()

set_target_properties(origami-headers PROPERTIES VERIFY_INTERFACE_HEADER_SETS ON)

target_link_libraries(origami PUBLIC roc::origami-headers)

add_dependencies(origami origami-headers_verify_interface_header_sets)

target_sources(
    origami
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/attention.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/gemm.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/hardware.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/heuristics.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/logger.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/origami.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/streamk.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/types.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/simulator/tensilelite/formocast_simulator.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/simulator/tensilelite/formocast.cpp"
)

if(ORIGAMI_ENABLE_NN)
    target_sources(
        origami
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/nn/filter.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/nn/features/gemm_tilewright.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/nn/types.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/nn/model_registry.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/nn/dispatch.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/nn/twrec/b64_decode.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/nn/twrec/load_validate.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/nn/twrec/twrec_loader.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/nn/twrec/rank.cpp"
    )

    set(_origami_nn_bundled_weights_dir
        "${CMAKE_CURRENT_SOURCE_DIR}/data/nn/tilewright/gfx950"
    )
    target_compile_definitions(
        origami
        PRIVATE ORIGAMI_NN_BUNDLED_WEIGHTS_DIR="${_origami_nn_bundled_weights_dir}"
    )

    find_package(yaml-cpp 0.8 QUIET CONFIG)
    if(NOT yaml-cpp_FOUND)
        if(ORIGAMI_ENABLE_FETCH)
            include(FetchContent)
            FetchContent_Declare(
                yaml-cpp
                GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
                GIT_TAG 0.8.0
                GIT_SHALLOW TRUE
            )
            FetchContent_MakeAvailable(yaml-cpp)
        else()
            message(FATAL_ERROR "yaml-cpp not found (set ORIGAMI_ENABLE_FETCH=ON to auto-fetch)")
        endif()
    endif()
    target_link_libraries(origami PRIVATE yaml-cpp::yaml-cpp)
endif()

target_link_libraries(origami PRIVATE hip::host)

if(ORIGAMI_ENABLE_PYTHON)
    add_subdirectory(python)
endif()

if(ORIGAMI_BUILD_TESTING OR BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ORIGAMI_STANDALONE)
rocm_install(TARGETS origami)

install(TARGETS origami-headers
    EXPORT origami-targets
    FILE_SET HEADERS
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT devel
)

rocm_export_targets(
    TARGETS
    roc::origami
    roc::origami-headers
    DEPENDS
    PACKAGE
    hip
    NAMESPACE
    roc::
)

if(ORIGAMI_BUILD_TESTING OR BUILD_TESTING)
    rocm_install(TARGETS origami-tests COMPONENT tests)
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/origami-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/origami-config.cmake" @ONLY
)

set(ORIGAMI_ENABLE_NN ${ORIGAMI_ENABLE_NN} CACHE BOOL "Whether origami was built with NN support")

rocm_install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/origami-config.cmake" DESTINATION
    "${CMAKE_INSTALL_LIBDIR}/cmake/origami" COMPONENT devel
)

set(BUILD_SHARED_LIBS ${ORIGAMI_BUILD_SHARED_LIBS})
set(ORIGAMI_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
    CACHE PATH "Path placed into ldconfig file"
)

rocm_create_package(
    NAME origami
    DESCRIPTION "Origami: Analytical Solution Selection for GEMM"
    MAINTAINER "Origami Maintainer <origami-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${ORIGAMI_CONFIG_DIR}
)
endif()

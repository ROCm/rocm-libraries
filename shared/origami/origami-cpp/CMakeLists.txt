# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

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
    EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/origami/origami_export.h"
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
        "${CMAKE_CURRENT_BINARY_DIR}/include"
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
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/simulator/tensilelite/formocast.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/origami/simulator/tensilelite/formocast_simulator.hpp"
        "${CMAKE_CURRENT_BINARY_DIR}/include/origami/origami_export.h"
)
target_include_directories(
    origami-headers INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(origami-headers INTERFACE hip::host)

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

target_link_libraries(origami PRIVATE hip::host)

if(ORIGAMI_BUILD_TESTING OR BUILD_TESTING)
    add_subdirectory(tests)
endif()

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
    "${PROJECT_SOURCE_DIR}/cmake/origami-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/origami-config.cmake" @ONLY
)

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

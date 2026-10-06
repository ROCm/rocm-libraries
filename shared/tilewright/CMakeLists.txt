# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)

project(tilewright VERSION 1.0.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_tilewright_standalone ON)
else()
    set(_tilewright_standalone OFF)
endif()

option(TILEWRIGHT_BUILD_TESTING "Build the tilewright tests." ${_tilewright_standalone})
option(TILEWRIGHT_ENABLE_PYTHON "Build the tilewright Python bindings." OFF)
option(TILEWRIGHT_ENABLE_INSTALL "Install the library, headers and CMake package." ${_tilewright_standalone})
option(TILEWRIGHT_ENABLE_FETCH "Fetch missing test/binding dependencies with FetchContent." ON)

include(GNUInstallDirs)
find_package(Threads REQUIRED)

add_library(tilewright STATIC)
add_library(roc::tilewright ALIAS tilewright)

set_target_properties(
    tilewright
    PROPERTIES POSITION_INDEPENDENT_CODE ON
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON
)
target_compile_features(tilewright PUBLIC cxx_std_17)
target_include_directories(
    tilewright PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(tilewright PUBLIC Threads::Threads)

target_sources(
    tilewright
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/tilewright/features.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/tilewright/format.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/tilewright/kernels.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/tilewright/rank.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/tilewright/registry.cpp"
)

# Scores are part of the model contract and must not depend on the compiler or
# on flags a parent project sets: no fast-math and no floating-point contraction.
target_compile_options(
    tilewright
    PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang,IntelLLVM>:-fno-fast-math -ffp-contract=off>
            $<$<CXX_COMPILER_ID:MSVC>:/fp:precise>
)

if(TILEWRIGHT_BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TILEWRIGHT_ENABLE_PYTHON)
    add_subdirectory(python)
endif()

if(TILEWRIGHT_ENABLE_INSTALL)
    include(CMakePackageConfigHelpers)

    set(_tilewright_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/tilewright")

    install(
        TARGETS tilewright
        EXPORT tilewright-targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT devel
    )
    install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        COMPONENT devel
        FILES_MATCHING
        PATTERN "*.hpp"
    )
    install(
        EXPORT tilewright-targets
        NAMESPACE roc::
        DESTINATION "${_tilewright_cmake_dir}"
        COMPONENT devel
    )

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tilewright-config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/tilewright-config.cmake"
        INSTALL_DESTINATION "${_tilewright_cmake_dir}"
    )
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/tilewright-config-version.cmake"
        COMPATIBILITY SameMajorVersion
    )
    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/tilewright-config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/tilewright-config-version.cmake"
        DESTINATION "${_tilewright_cmake_dir}"
        COMPONENT devel
    )
endif()

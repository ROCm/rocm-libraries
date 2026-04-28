# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Support both standalone (scikit-build-core / tox) and subdirectory
# (hipblaslt CMake) builds.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.25)
    project(rocisa LANGUAGES CXX)
    set(ROCISA_STANDALONE ON)
else()
    set(ROCISA_STANDALONE OFF)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(rocisa-cpp INTERFACE)
add_library(rocisa::rocisa-cpp ALIAS rocisa-cpp)
target_sources(rocisa-cpp
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/rocisa/include/enum.hpp>
)
target_include_directories(rocisa-cpp
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

# Build the Python extension when bundling deps (hipblaslt subdir build)
# or when building standalone.
if(HIPBLASLT_BUNDLE_PYTHON_DEPS OR ROCISA_STANDALONE)
    if(ROCISA_STANDALONE)
        # scikit-build-core pre-installs nanobind (from build-system.requires)
        # and injects its CMake path, so find_package works without FetchContent.
        # nanobind requires Python to be found first.
        find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
        find_package(nanobind REQUIRED CONFIG)
    else()
        include(FetchContent)
        FetchContent_Declare(
          nanobind
          GIT_REPOSITORY https://github.com/wjakob/nanobind.git
          GIT_TAG        9b3afa9dbdc23641daf26fadef7743e7127ff92f # v2.6.1
        )
        FetchContent_MakeAvailable(nanobind)
    endif()

    # HIP is required to build _rocisa (via hip::host).  Set the search prefix
    # so find_package works both in developer environments and in the temporary
    # build sandbox created by scikit-build-core / uv.
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
    find_package(hip REQUIRED)

    # Locate origami: prefer the monorepo sibling, fall back to installed package.
    if(NOT TARGET roc::origami)
        set(_origami_sibling "${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared/origami")
        if(IS_DIRECTORY "${_origami_sibling}")
            add_subdirectory("${_origami_sibling}" origami EXCLUDE_FROM_ALL)
        else()
            find_package(origami REQUIRED)
        endif()
    endif()

    set(ROCISAINST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/instruction.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/common.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/branch.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/cmp.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/cvt.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/mem.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/mfma.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/extension.cpp")
    set(ROCISAFUNC_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/functions.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/argument.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/f_math.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/f_branch.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/f_cast.cpp")
    set(ROCISAPASS_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/pass.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/graph.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/composite.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/cycle.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/remove.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/insert_delay_alu.cpp")

    nanobind_add_module(_rocisa NOMINSIZE NB_SUPPRESS_WARNINGS
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/main.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/base.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/code.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/container.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/count.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/enum.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/helper.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/label.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/macro.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/register.cpp"
                            ${ROCISAINST_SOURCE}
                            ${ROCISAPASS_SOURCE}
                            ${ROCISAFUNC_SOURCE})
    target_include_directories(_rocisa PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/include"
    )

    # Optional inclusion of external shared/stinkytofu CMake project
    set(STINKYTOFU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared/stinkytofu")
    if(EXISTS "${STINKYTOFU_DIR}/CMakeLists.txt")
        # Add the external directory; build output goes into a local binary dir 'stinkytofu'
        add_subdirectory("${STINKYTOFU_DIR}" stinkytofu)
    else()
        message(FATAL_ERROR "External directory not found: ${STINKYTOFU_DIR} (skipping)")
    endif()

    target_link_libraries(_rocisa PUBLIC roc::origami)
    target_link_libraries(_rocisa PRIVATE hip::host stinkytofu)
    # rocisa conversion sources include stinkytofu public headers that
    # transitively include generated files (hardware/gfxIsa.inc) from
    # stinkytofu's build directory.
    target_include_directories(_rocisa PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/stinkytofu")
    set_target_properties(_rocisa PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/rocisa"
        # At install time (wheel/site-packages): libstinkytofu.so sits next to
        # _rocisa.so in the same rocisa/ directory.
        INSTALL_RPATH "$ORIGIN"
        # At build time: libstinkytofu.so is in the stinkytofu/ subdirectory of
        # the binary dir, not co-located with _rocisa.so.
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/stinkytofu"
    )
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/__init__.py"
        "${CMAKE_CURRENT_BINARY_DIR}/rocisa/__init__.py"
        COPYONLY
    )

    if(ROCISA_STANDALONE)
        # Install rules for scikit-build-core: place _rocisa.so, libstinkytofu.so,
        # and __init__.py into the rocisa/ package directory in the wheel.
        install(TARGETS _rocisa DESTINATION rocisa)
        # Co-install libstinkytofu.so next to _rocisa.so so the dynamic linker
        # can find it at runtime (wheel layout: both files sit in rocisa/).
        install(FILES $<TARGET_FILE:stinkytofu> DESTINATION rocisa)
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/__init__.py" DESTINATION rocisa)
    endif()
endif()

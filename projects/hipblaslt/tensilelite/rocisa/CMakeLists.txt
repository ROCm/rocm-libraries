# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Support both standalone (scikit-build-core / tox) and subdirectory
# (hipblaslt CMake) builds.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.25)

    include(${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/modules/default_amdclang.cmake)

    project(rocisa LANGUAGES CXX)
    set(ROCISA_STANDALONE ON)
else()
    set(ROCISA_STANDALONE OFF)
endif()

option(ROCISA_USE_STABLE_ABI
       "Build rocisa as a Python stable ABI extension (requires Python >= 3.12 and CMake >= 3.26)."
       OFF)

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
        if(ROCISA_USE_STABLE_ABI)
            find_package(Python 3.12 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
        else()
            find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
        endif()
        find_package(nanobind REQUIRED CONFIG)
    else()
        include(FetchContent)
        FetchContent_Declare(
          nanobind
          GIT_REPOSITORY https://github.com/wjakob/nanobind.git
          GIT_TAG        9b3afa9dbdc23641daf26fadef7743e7127ff92f # v2.6.1
        )
        FetchContent_MakeAvailable(nanobind)
        if(ROCISA_USE_STABLE_ABI)
            find_package(Python 3.12 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
        endif()
    endif()

    # HIP is required to build _rocisa (via hip::host).  Set the search prefix
    # so find_package works both in developer environments and in the temporary
    # build sandbox created by scikit-build-core / uv.
    if(ROCM_PATH)
        list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
    endif()
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
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/insert_delay_alu.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/macro_inline.cpp")

    set(_rocisa_nb_flags NOMINSIZE NB_SUPPRESS_WARNINGS)
    if(ROCISA_USE_STABLE_ABI)
        list(APPEND _rocisa_nb_flags STABLE_ABI)
    endif()

    nanobind_add_module(_rocisa ${_rocisa_nb_flags}
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

    # Locate stinkytofu: prefer installed package, fall back to monorepo sibling.
    if(NOT TARGET stinkytofu)
        find_package(stinkytofu QUIET)
        if(NOT stinkytofu_FOUND)
            set(_stinkytofu_sibling "${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared/stinkytofu")
            if(IS_DIRECTORY "${_stinkytofu_sibling}")
                set(STINKYTOFU_ENABLE_WERROR ON)
                add_subdirectory("${_stinkytofu_sibling}" stinkytofu EXCLUDE_FROM_ALL)
                set(_stinkytofu_from_subdirectory TRUE)
            else()
                message(FATAL_ERROR "stinkytofu not found via find_package or sibling directory")
            endif()
        endif()
    endif()

    if(NOT DEFINED _stinkytofu_from_subdirectory)
        get_target_property(_st_imported stinkytofu IMPORTED)
        if(_st_imported)
            set(_stinkytofu_from_subdirectory FALSE)
        else()
            set(_stinkytofu_from_subdirectory TRUE)
        endif()
    endif()

    target_link_libraries(_rocisa PUBLIC roc::origami)
    target_link_libraries(_rocisa PRIVATE hip::host stinkytofu)
    if(_stinkytofu_from_subdirectory)
        # Source-built stinkytofu via add_subdirectory: generated headers are in
        # the stinkytofu/ subdirectory of the binary dir.
        target_include_directories(_rocisa PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/stinkytofu")
        set(_st_build_rpath "${CMAKE_CURRENT_BINARY_DIR}/stinkytofu")
    else()
        # Installed stinkytofu: include dirs come from the exported target.
        get_target_property(_st_loc stinkytofu IMPORTED_LOCATION)
        if(_st_loc)
            cmake_path(GET _st_loc PARENT_PATH _st_build_rpath)
        else()
            set(_st_build_rpath "")
        endif()
    endif()
    set_target_properties(_rocisa PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/rocisa"
        BUILD_RPATH "${_st_build_rpath}"
    )
    if(_stinkytofu_from_subdirectory)
        # Source-built sibling: no installed libstinkytofu exists, so the wheel carries
        # its own copy next to _rocisa and loads it from there.
        set_target_properties(_rocisa PROPERTIES INSTALL_RPATH "$ORIGIN")
        # Windows has no RPATH; copy stinkytofu.dll next to _rocisa.pyd so the loader
        # finds it at runtime.
        if(WIN32)
            add_custom_command(TARGET _rocisa POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:stinkytofu>
                    $<TARGET_FILE_DIR:_rocisa>
            )
        endif()
    endif()
    # Installed stinkytofu is a standard ROCm library on the loader's search path (like
    # libyaml behind PyYAML); _rocisa's DT_NEEDED resolves it from there, so no
    # wheel-local copy or RPATH is needed in that case.
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/__init__.py"
        "${CMAKE_CURRENT_BINARY_DIR}/rocisa/__init__.py"
        COPYONLY
    )
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/_build_info.py.in"
        "${CMAKE_CURRENT_BINARY_DIR}/rocisa/_build_info.py"
        @ONLY
    )
    # _build_info.py is intentionally NOT in install() rules so it is absent
    # from wheels and pre-built packages, acting as the source-build sentinel.
    # For editable/dev installs, pass -DROCISA_INCLUDE_BUILD_INFO=ON to include it.
    option(ROCISA_INCLUDE_BUILD_INFO "Install _build_info.py for staleness detection" OFF)
    if(ROCISA_INCLUDE_BUILD_INFO)
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rocisa/_build_info.py" DESTINATION rocisa)
    endif()

    if(ROCISA_STANDALONE)
        # scikit-build-core wheel install rules: place _rocisa and __init__.py into the
        # rocisa/ package directory in the wheel.
        install(TARGETS _rocisa DESTINATION rocisa)
        if(_stinkytofu_from_subdirectory)
            # Source-built: vendor libstinkytofu into the wheel — no installed copy exists.
            install(FILES $<TARGET_FILE:stinkytofu> DESTINATION rocisa)
            if(NOT WIN32)
                install(FILES $<TARGET_SONAME_FILE:stinkytofu> DESTINATION rocisa)
            endif()
        endif()
        # Installed stinkytofu ships in its own ROCm package; it is not bundled here.
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/__init__.py" DESTINATION rocisa)
    endif()

    # Example plugin — the target only exists when stinkytofu is configured with
    # -DSTINKYTOFU_BUILD_EXAMPLES=ON (default OFF; opted in by the source-built standalone
    # wheel via tasks.py). Integrated/ROCm builds leave it OFF, so the target is absent and
    # this block is skipped — the example never lands in a production ROCm package; there,
    # the installed stinkytofu package ships and locates its own plugin.
    #
    # In the source-built wheel, libstinkytofu is vendored into rocisa/, so the plugin must
    # sit at rocisa/${STINKYTOFU_PLUGIN_INSTALL_RELDIR} to match PassBuilder::examplePluginPath()'s
    # lookup relative to libstinkytofu. Consuming stinkytofu's exported reldir keeps the install
    # path in lockstep with the runtime lookup — no hardcoded duplicate.
    # No INSTALL_RPATH is needed: on Linux the plugin has no DT_NEEDED on libstinkytofu (it
    # is headers-only) and the host is already resident when loadPlugin() dlopens it.
    # Only installable for shared builds (MODULE target); static builds use OBJECT.
    if(TARGET stinkytofu-plugin-helloworld AND BUILD_SHARED_LIBS AND _stinkytofu_from_subdirectory)
        set_target_properties(stinkytofu-plugin-helloworld PROPERTIES EXCLUDE_FROM_ALL FALSE)
        add_dependencies(_rocisa stinkytofu-plugin-helloworld)
        install(TARGETS stinkytofu-plugin-helloworld
            DESTINATION "rocisa/${STINKYTOFU_PLUGIN_INSTALL_RELDIR}")
    endif()
endif()

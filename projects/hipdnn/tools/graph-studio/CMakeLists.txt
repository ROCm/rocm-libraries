# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Graph Studio is a Bun/Electron app with a small N-API add-on that links
# hipDNN. CMake does not compile any of it -- node-gyp does -- but it publishes
# the in-tree hipDNN's include dirs, compile definitions, link libraries and
# runtime directories as build-config.json, so the add-on builds against this
# build tree instead of an installed SDK. electron/hipdnn-paths.cjs reads that
# file both at add-on build time and when the app starts.

if(NOT TARGET hipdnn_frontend)
    message(FATAL_ERROR "graph-studio requires the 'hipdnn' component")
endif()

find_program(HIPDNN_GRAPH_STUDIO_BUN NAMES bun)
if(NOT HIPDNN_GRAPH_STUDIO_BUN)
    message(FATAL_ERROR "graph-studio requires 'bun' on PATH; see https://bun.sh")
endif()

# The add-on calls the HIP runtime directly (device query, hipMalloc/hipMemcpy).
# hip::host carries the headers transitively through hipdnn_frontend, but not a
# path node-gyp can put on a link line, so locate the library itself.
find_library(HIPDNN_GRAPH_STUDIO_HIP_LIB
    NAMES amdhip64
    HINTS "${ROCM_PATH}" ENV ROCM_PATH
    PATH_SUFFIXES lib lib64
)
if(NOT HIPDNN_GRAPH_STUDIO_HIP_LIB)
    message(FATAL_ERROR "graph-studio: could not find the amdhip64 library under ROCM_PATH='${ROCM_PATH}'")
endif()
get_filename_component(_studio_hip_libdir "${HIPDNN_GRAPH_STUDIO_HIP_LIB}" DIRECTORY)

set(_studio_out "${CMAKE_BINARY_DIR}/graph-studio")
set(_studio_config "${_studio_out}/build-config.json")

set(_studio_libraries "$<TARGET_LINKER_FILE:hipdnn_backend>" "${HIPDNN_GRAPH_STUDIO_HIP_LIB}")
set(_studio_runtime_dirs "$<TARGET_FILE_DIR:hipdnn_backend>" "${_studio_hip_libdir}")
if(WIN32)
    # The HIP runtime DLL lives next to the tools, not next to the import library.
    list(APPEND _studio_runtime_dirs "${ROCM_PATH}/bin")
else()
    # Nothing sets LD_LIBRARY_PATH for the Electron process, so the add-on has to
    # carry the search paths itself.
    list(APPEND _studio_libraries "-Wl,-rpath,$<TARGET_FILE_DIR:hipdnn_backend>"
                                  "-Wl,-rpath,${_studio_hip_libdir}"
    )
endif()

# Each list is joined with '","' so it lands as a JSON array. Generator
# expressions that evaluate to nothing (INSTALL_INTERFACE include dirs, disabled
# feature macros) leave empty elements behind; electron/paths.cjs drops them.
string(CONCAT _studio_config_content
    "{\n"
    "  \"mode\": \"in-tree\",\n"
    "  \"buildDir\": \"${CMAKE_BINARY_DIR}\",\n"
    "  \"outDir\": \"${_studio_out}\",\n"
    "  \"includeDirs\": [\"$<JOIN:$<TARGET_PROPERTY:hipdnn_frontend,INTERFACE_INCLUDE_DIRECTORIES>,\",\">\"],\n"
    "  \"defines\": [\"$<JOIN:$<TARGET_PROPERTY:hipdnn_frontend,INTERFACE_COMPILE_DEFINITIONS>,\",\">\"],\n"
    "  \"libraries\": [\"$<JOIN:${_studio_libraries},\",\">\"],\n"
    "  \"runtimeDirs\": [\"$<JOIN:${_studio_runtime_dirs},\",\">\"],\n"
    "  \"pluginDir\": \"${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}\"\n"
    "}\n"
)
file(GENERATE OUTPUT "${_studio_config}" CONTENT "${_studio_config_content}")

# Launcher next to the rest of the build tree's executables. It runs Electron on
# the source app directory -- node_modules and the .cjs entry points live there --
# and points it at this build tree's outputs through HIPDNN_BUILD_CONFIG.
if(WIN32)
    file(GENERATE
        OUTPUT "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/start-graph-studio.bat"
        CONTENT
            "@echo off\r\nsetlocal\r\nwhere bun >nul 2>&1 || (echo bun not found on PATH & exit /b 1)\r\nset \"HIPDNN_BUILD_CONFIG=${_studio_config}\"\r\ncd /d \"${CMAKE_CURRENT_SOURCE_DIR}\"\r\nbunx electron . %*\r\n"
    )
else()
    file(GENERATE
        OUTPUT "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/start-graph-studio.sh"
        CONTENT
            "#!/bin/sh\ncommand -v bun >/dev/null || { echo 'bun not found on PATH' >&2; exit 1; }\nexport HIPDNN_BUILD_CONFIG='${_studio_config}'\ncd '${CMAKE_CURRENT_SOURCE_DIR}'\nexec bunx electron . \"$@\"\n"
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ
                         WORLD_EXECUTE
    )
endif()

# node-gyp's make generator takes its compiler from CC/CXX (gyp's make.py:
# CXX.target falls back to $(CXX), and LINK falls back to $(CXX.target)), so on
# Linux the add-on can be built with the same clang as the rest of the tree.
#
# The Windows generator emits MSBuild projects whose only non-MSVC toolset is
# Visual Studio's bundled ClangCL -- an arbitrary compiler path is not
# selectable -- so there the add-on stays on MSVC. That is sound rather than
# merely tolerated: it meets hipDNN at the backend's extern "C" ABI.
set(_studio_native_env "HIPDNN_BUILD_CONFIG=${_studio_config}")
if(NOT WIN32)
    list(APPEND _studio_native_env "CC=${CMAKE_C_COMPILER}" "CXX=${CMAKE_CXX_COMPILER}"
         "LINK=${CMAKE_CXX_COMPILER}"
    )
endif()

# The web bundle and the node-gyp tree both land under ${_studio_out}; the
# add-on's node-gyp tree is a directory this target never declares as an output,
# so the target is always out of date.
add_custom_target(hipdnn_graph_studio ALL
    COMMAND "${HIPDNN_GRAPH_STUDIO_BUN}" install
    # --ignore-scripts: the add-on is built by build:native below, which first
    # generates the .gypi binding.gyp needs. An install-time node-gyp run would
    # fire before that and fail.
    COMMAND "${CMAKE_COMMAND}" -E chdir "${CMAKE_CURRENT_SOURCE_DIR}/electron/native"
            "${HIPDNN_GRAPH_STUDIO_BUN}" install --ignore-scripts
    COMMAND "${CMAKE_COMMAND}" -E env ${_studio_native_env}
            "${HIPDNN_GRAPH_STUDIO_BUN}" run build:native
    COMMAND "${CMAKE_COMMAND}" -E env "HIPDNN_BUILD_CONFIG=${_studio_config}"
            "${HIPDNN_GRAPH_STUDIO_BUN}" run build
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building hipDNN Graph Studio into ${_studio_out}"
    VERBATIM
)
add_dependencies(hipdnn_graph_studio hipdnn_backend)

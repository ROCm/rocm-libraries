# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################
cmake_minimum_required(VERSION 3.15)
# This is a CMake script for building the rocisa together with executing
# Tensile without having two steps (cmake build + Tensile run)
# This script is designed to be used with the TensileLite project
# It allows you to specify the Tensile binary to run and the arguments
# to pass to it.
# The original command for running a TensileLite script is:
#     Tensile config.yaml ./
# "./" is the working directory.
# This CMakeLists.txt supports two modes with setting DEVELOP_MODE ON or OFF:
# 1. Script mode (DEVELOP_MODE=ON): It creates a script that can be run after the configuration
#    When using mode, you can run it as:
#        cmake -DDEVELOP_MODE=ON -DCMAKE_PREFIX_PATH=/opt/rocm ..
#    The script will be created in the build folder and will be named in Tensile.bat or Tensile.sh
#    depending on the platform. You can then run the script as:
#        Tensile.sh config.yaml ./
#    or
#        Tensile.bat config.yaml ./
# 2. Normal CMake mode: It sets the Python srguments at configuration stage with "BIN_ARGS"
#    With this mode, you can run it as:
#        cmake -DCMAKE_PREFIX_PATH=/opt/rocm -DTENSILE_BIN=Tensile -DBIN_ARGS="config.yaml ./" .. && cmake --build .
#    The script will check if the TENSILE_BIN is valid and if the BIN_ARGS
#    are provided. It will then run the specified Tensile binary with the
#    provided arguments after the build step.
#    Avoid creating build folder under TensileLite in this mode because tox also use this folder
#    and it will cause the CMakeCache.txt to be deleted.
#
# If you want to build a Debug version of rocisa, you can add -DCMAKE_BUILD_TYPE=Debug


project(TensileLiteAutoBuild LANGUAGES HIP)
set(DEVELOP_MODE OFF CACHE BOOL "Set to ON to create a script for running the Tensile binary after the configuration.")

set(VALID_BINS
"Tensile"
"TensileBenchmarkCluster"
"TensileClientConfig"
"TensileCreateLibrary"
"TensileGenerateSummations"
"TensileLogic"
"TensileRetuneLibrary"
"TensileUpdateLibrary"
)

# Find Python interpreter
find_package(Python COMPONENTS Interpreter REQUIRED)

# Add rocisa
add_subdirectory(rocisa)

# Set common variables
set(TENSILE_BIN_ROOT "${CMAKE_SOURCE_DIR}/Tensile/bin")
set(PYTHONPATH "${PROJECT_BINARY_DIR}/lib")

if(DEFINED DEVELOP_MODE)
    include(ProcessorCount)
    ProcessorCount(NPROC)
    set(CMAKE_BUILD_COMMAND "cmake --build ${PROJECT_BINARY_DIR} -j ${NPROC}")
    foreach(BIN ${VALID_BINS})
        if(WIN32)
            set(SCRIPT_NAME ${BIN}.bat)
            file(WRITE "${PROJECT_BINARY_DIR}/${SCRIPT_NAME}"
                 "@echo off\n"
                 "${CMAKE_BUILD_COMMAND}\n"
                 "PYTHONPATH=${PYTHONPATH} ${Python_EXECUTABLE} ${TENSILE_BIN_ROOT}/${BIN} %*\n"
                 "pause\n"
            )
        else()
            set(SCRIPT_NAME ${BIN}.sh)
            file(WRITE "${PROJECT_BINARY_DIR}/${SCRIPT_NAME}"
                 "#!/bin/bash\n"
                 "set -euo pipefail\n"
                 "${CMAKE_BUILD_COMMAND}\n"
                 "if [[ \"$\{DEBUGPY_ENABLE:-\}\" == \"1\" ]]; then\n"
                 "    echo \"===DEBUGPY_READY===\"\n"
                 "    PYTHONPATH=${PYTHONPATH} ${Python_EXECUTABLE} -m debugpy --listen 0.0.0.0:5678 --wait-for-client ${TENSILE_BIN_ROOT}/${BIN} \"$@\"\n"
                 "else\n"
                 "    PYTHONPATH=${PYTHONPATH} ${Python_EXECUTABLE} ${TENSILE_BIN_ROOT}/${BIN} \"$@\"\n"
                 "fi\n"
            )
            execute_process(COMMAND chmod +x "${PROJECT_BINARY_DIR}/${BIN}.sh")
        endif()
    endforeach()
    message(STATUS "Script created: ${PROJECT_BINARY_DIR}/${SCRIPT_NAME}. Please run the Tensile bin command as usual under the build folder.")
else()
    # Check if TENSILE_BIN is set
    function(set_tensile_bin BINS)
        list(FIND VALID_BINS "${BINS}" INDEX)
        if(INDEX EQUAL -1)
            message(FATAL_ERROR "Invalid bin name: ${BINS}. Valid options are: ${VALID_BINS}")
        endif()
        set("Tensile bin is set to ${BINS}")
    endfunction()

    if(DEFINED TENSILE_BIN)
        set_tensile_bin(${TENSILE_BIN})
    else()
        message(FATAL_ERROR "TENSILE_BIN is not defined. Please set it to one of the valid options: ${VALID_BINS}")
    endif()
    # Check if BIN_ARGS_LIST is empty
    if(NOT DEFINED BIN_ARGS)
        message(FATAL_ERROR "BIN_ARGS is not defined. Please set it to the arguments you want to pass to the Tensile binary.")
    endif()
    # Retrieve the arguments
    set(BIN_ARGS ${BIN_ARGS})
    # Split the arguments into a list
    separate_arguments(BIN_ARGS_LIST NATIVE_COMMAND "${BIN_ARGS}")
    message(STATUS "Script set: ${TENSILE_BIN} ${BIN_ARGS_LIST}")

    # Ensure the Python script runs after the build
    add_custom_target(RunPythonScript
        ALL
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PYTHONPATH} -- ${Python_EXECUTABLE} ${TENSILE_BIN_ROOT}/${TENSILE_BIN} ${BIN_ARGS_LIST}
        COMMENT "Running Python script ${TENSILE_BIN} ${BIN_ARGS_LIST}"
        VERBATIM
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    )
    add_dependencies(RunPythonScript rocisa)
endif()

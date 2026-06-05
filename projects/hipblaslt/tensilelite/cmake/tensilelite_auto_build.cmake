# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

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
"TensileMergeLibrary"
)

# Set common variables
set(TENSILE_BIN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/bin")
string(REGEX REPLACE ";" " " tensilelite_python_command "${HIPBLASLT_PYTHON_COMMAND}")

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
                 "${tensiltelite_python_command} ${TENSILE_BIN_ROOT}/${BIN} %*\n"
                 "pause\n"
            )
        else()
            set(SCRIPT_NAME ${BIN}.sh)
            file(WRITE "${PROJECT_BINARY_DIR}/${SCRIPT_NAME}"
                 "#!/bin/bash\n"
                 "echo \"DEPRECATED: ${BIN}.sh will be removed in a future release.\" >&2\n"
                 "echo \"Run directly: ${TENSILE_BIN_ROOT}/${BIN} $@\" >&2\n"
                 "set -euo pipefail\n"
                 "${CMAKE_BUILD_COMMAND}\n"
                 "if [[ \"$\{DEBUGPY_ENABLE:-\}\" == \"1\" ]]; then\n"
                 "    echo \"===DEBUGPY_READY===\"\n"
                 "    ${tensilelite_python_command} -m debugpy --listen 0.0.0.0:5678 --wait-for-client ${TENSILE_BIN_ROOT}/${BIN} \"$@\"\n"
                 "else\n"
                 "    ${tensilelite_python_command} ${TENSILE_BIN_ROOT}/${BIN} \"$@\"\n"
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
        COMMAND ${tensilelite_python_command} ${TENSILE_BIN_ROOT}/${TENSILE_BIN} ${BIN_ARGS_LIST}
        COMMENT "Running Python script ${TENSILE_BIN} ${BIN_ARGS_LIST}"
        VERBATIM
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    )
    add_dependencies(RunPythonScript rocisa)
endif()

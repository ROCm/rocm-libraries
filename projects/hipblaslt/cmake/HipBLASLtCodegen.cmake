# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# HipBLASLtCodegen.cmake — the single source of truth for turning TensileLite
# library-logic YAMLs into device code-object libraries.
#
# Defines hipblaslt_create_device_library(), used both by hipBLASLt's own
# device-library build (against the source tree) and, once exported through
# hipblaslt-config.cmake, by downstream consumers (e.g. hipSPARSELt) building
# against a binary-only hipBLASLt install with no hipBLASLt source tree present.
#
# Variables read from the calling scope (set in-tree by hipBLASLt's CMakeLists;
# set by hipblaslt-config.cmake for installed consumers):
#   HIPBLASLT_PYTHON_COMMAND  - REQUIRED. Interpreter command used to run the
#                               codegen. In-tree this is the found Python (or the
#                               bundled-python wrapper); for installed consumers
#                               hipblaslt-config sets it so that `-m
#                               Tensile.TensileCreateLibrary` resolves against the
#                               installed codegen subset (PYTHONPATH).
#   HIPBLASLT_CODEGEN_ROOT    - REQUIRED. Directory containing the `Tensile/`
#                               codegen package. In-tree: <src>/tensilelite.
#                               Installed: <prefix>/share/hipblaslt/codegen.
#                               Used to locate Tensile/bin/TensileLogic and
#                               Tensile/TensileLogic/known_bugs.yaml.
#   HIPBLASLT_PYTHON_DEPS     - OPTIONAL. Extra DEPENDS for the codegen custom
#                               commands (in-tree: the built `_rocisa` target so
#                               codegen re-runs when rocisa changes). Empty for
#                               installed consumers.

include_guard(GLOBAL)

# hipblaslt_create_device_library(
#     LOGIC_PATH <dir>            # required: library-logic YAMLs to compile
#     OUTPUT_DIR <dir>            # required: device libs land in <dir>/library
#     [TARGET <name>]            # default: tensilelite-device-libraries
#     [ARCHES <gfx>...]          # default: ${GPU_TARGETS}
#     [CXX_COMPILER <path>]      # default: ${CMAKE_CXX_COMPILER}
#     [OFFLOAD_BUNDLER <path>]
#     [JOBS <n>]
#     [LOGIC_FILTER <glob>]
#     [ASAN] [YAML_FORMAT] [NO_COMPRESS] [EXPERIMENTAL]
#     [NO_LAZY_LOAD] [ASM_COMMENTS] [KEEP_BUILD_TMP] [ASM_DEBUG]
# )
function(hipblaslt_create_device_library)
    set(_opts ASAN YAML_FORMAT NO_COMPRESS EXPERIMENTAL NO_LAZY_LOAD ASM_COMMENTS KEEP_BUILD_TMP ASM_DEBUG)
    set(_one TARGET LOGIC_PATH OUTPUT_DIR CXX_COMPILER OFFLOAD_BUNDLER JOBS LOGIC_FILTER)
    set(_multi ARCHES)
    cmake_parse_arguments(_cdl "${_opts}" "${_one}" "${_multi}" ${ARGN})

    if(_cdl_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "hipblaslt_create_device_library: unexpected arguments: ${_cdl_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT _cdl_LOGIC_PATH)
        message(FATAL_ERROR "hipblaslt_create_device_library: LOGIC_PATH is required")
    endif()
    if(NOT _cdl_OUTPUT_DIR)
        message(FATAL_ERROR "hipblaslt_create_device_library: OUTPUT_DIR is required")
    endif()
    if(NOT HIPBLASLT_PYTHON_COMMAND)
        message(FATAL_ERROR "hipblaslt_create_device_library: HIPBLASLT_PYTHON_COMMAND is not set")
    endif()
    if(NOT HIPBLASLT_CODEGEN_ROOT)
        message(FATAL_ERROR "hipblaslt_create_device_library: HIPBLASLT_CODEGEN_ROOT is not set")
    endif()

    # --- Defaults ---
    if(NOT _cdl_TARGET)
        set(_cdl_TARGET "tensilelite-device-libraries")
    endif()
    if(NOT _cdl_ARCHES)
        set(_cdl_ARCHES ${GPU_TARGETS})
    endif()
    if(NOT _cdl_ARCHES)
        message(FATAL_ERROR "hipblaslt_create_device_library: no ARCHES given and GPU_TARGETS is empty")
    endif()
    if(NOT _cdl_CXX_COMPILER)
        set(_cdl_CXX_COMPILER "${CMAKE_CXX_COMPILER}")
    endif()

    file(MAKE_DIRECTORY "${_cdl_OUTPUT_DIR}/library")

    # --- Assemble TensileCreateLibrary options ---
    # Architectures are passed as a single semicolon-separated value; escape the
    # separator so it survives list expansion into the custom-command argv.
    list(JOIN _cdl_ARCHES "$<SEMICOLON>" _arches_semi)
    set(_opts_list "--architecture=${_arches_semi}" "--cxx-compiler=${_cdl_CXX_COMPILER}")
    if(_cdl_OFFLOAD_BUNDLER)
        list(APPEND _opts_list "--offload-bundler=${_cdl_OFFLOAD_BUNDLER}")
    endif()
    if(_cdl_ASAN)
        list(APPEND _opts_list "--address-sanitizer")
    endif()
    if(_cdl_JOBS)
        list(APPEND _opts_list "--jobs=${_cdl_JOBS}")
    endif()
    if(_cdl_KEEP_BUILD_TMP)
        list(APPEND _opts_list "--keep-build-tmp")
    endif()
    if(_cdl_ASM_DEBUG)
        list(APPEND _opts_list "--asm-debug")
    endif()
    if(_cdl_YAML_FORMAT)
        list(APPEND _opts_list "--library-format=yaml")
    endif()
    if(_cdl_LOGIC_FILTER)
        list(APPEND _opts_list "--logic-filter=${_cdl_LOGIC_FILTER}")
    endif()
    if(_cdl_NO_COMPRESS)
        list(APPEND _opts_list "--no-compress")
    endif()
    if(_cdl_EXPERIMENTAL)
        list(APPEND _opts_list "--experimental")
    endif()
    if(_cdl_NO_LAZY_LOAD)
        list(APPEND _opts_list "--no-lazy-library-loading")
    endif()
    if(NOT _cdl_ASM_COMMENTS)
        list(APPEND _opts_list "--disable-asm-comments")
    endif()

    # --- Pre-build gate: validate all library logic YAMLs (WorkGroup,
    # MatrixInstruction, WorkGroupMappingXCC vs CU count, etc.) before generating
    # .dat files. Fails the build if any solution fails validation so bad logic is
    # never compiled. ---
    set(_known_bugs "${HIPBLASLT_CODEGEN_ROOT}/Tensile/TensileLogic/known_bugs.yaml")
    # Stamp DEPENDS include known_bugs.yaml but not every library logic YAML
    # (thousands of files; CONFIGURE_DEPENDS globs are costly). After editing logic
    # only, run scripts/run_tensile_logic_check.py or touch the stamp input to
    # force re-validation.
    set(_logic_stamp "${CMAKE_CURRENT_BINARY_DIR}/${_cdl_TARGET}-TensileLogic.stamp")
    add_custom_command(
        OUTPUT "${_logic_stamp}"
        COMMENT "Validating library logic (TensileLogic --check-all) for ${_cdl_TARGET} ..."
        COMMAND ${HIPBLASLT_PYTHON_COMMAND}
            "${HIPBLASLT_CODEGEN_ROOT}/Tensile/bin/TensileLogic"
            "${_cdl_LOGIC_PATH}"
            --known-bugs
            "${_known_bugs}"
            --check-all
        COMMAND ${CMAKE_COMMAND} -E touch "${_logic_stamp}"
        DEPENDS ${HIPBLASLT_PYTHON_DEPS} "${_known_bugs}"
        VERBATIM
        USES_TERMINAL
    )

    # --- Generate device libraries ---
    set(_output_stamp "${CMAKE_CURRENT_BINARY_DIR}/${_cdl_TARGET}.stamp")
    set(_tcl_command
        ${HIPBLASLT_PYTHON_COMMAND} -m Tensile.TensileCreateLibrary
        ${_opts_list}
        "${_cdl_LOGIC_PATH}"
        "${_cdl_OUTPUT_DIR}"
        HIP
    )
    add_custom_command(
        OUTPUT "${_output_stamp}"
        COMMENT "Building device libraries to ${_cdl_OUTPUT_DIR} ..."
        COMMAND ${_tcl_command}
        COMMAND ${CMAKE_COMMAND} -E touch "${_output_stamp}"
        DEPENDS ${HIPBLASLT_PYTHON_DEPS} "${_logic_stamp}"
        # Because the command can contain special characters
        VERBATIM
        # Because this can be very long running and difficult to debug deadlocks
        # without streaming.
        USES_TERMINAL
    )

    block(SCOPE_FOR VARIABLES)
        list(JOIN _tcl_command " " _formatted_tcl)
        message(STATUS "Device lib build command (${_cdl_TARGET}): ${_formatted_tcl}")
    endblock()

    add_custom_target(${_cdl_TARGET} ALL
        DEPENDS "${_output_stamp}"
    )
endfunction()

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include_guard(GLOBAL)

set(TENSILELITE_BUILD_PARALLEL_LEVEL "" CACHE STRING
    "Number of CPU cores to use for building device libraries (uses nproc if unset).")
set(TENSILELITE_KEEP_BUILD_TMP OFF CACHE BOOL
    "Keep the temporary device-library build directory.")
set(TENSILELITE_ASM_DEBUG OFF CACHE BOOL
    "Keep debug information in generated device libraries.")
set(TENSILELITE_LOGIC_FILTER "" CACHE STRING
    "Glob used to select logic files; empty selects all logic files.")
set(TENSILELITE_NO_COMPRESS OFF CACHE BOOL
    "Do not compress device code object files.")
set(TENSILELITE_EXPERIMENTAL OFF CACHE BOOL
    "Process experimental logic files.")
set(TENSILELITE_ENABLE_ASM_COMMENTS OFF CACHE BOOL
    "Emit comments in generated assembly.")
set(TENSILELITE_OFFLOADBUNDLER "" CACHE FILEPATH
    "Path to clang-offload-bundler.")
set(TENSILELITE_LIBLOGIC_PATH "" CACHE PATH
    "Path to TensileLite library logic files.")
set(TENSILELITE_LIBRARY_FORMAT "msgpack" CACHE STRING
    "Device library serialization format (msgpack or yaml).")
set_property(CACHE TENSILELITE_LIBRARY_FORMAT PROPERTY STRINGS msgpack yaml)
set(Tensile_NO_LAZY_LIBRARY_LOADING OFF CACHE BOOL
    "Disable lazy loading of device libraries.")

function(hipblaslt_make_python_command out_command)
    set(_options ASAN TSAN)
    set(_multi PYTHONPATH_DIRS RUNTIME_LIB_DIRS TOOL_BIN_DIRS)
    cmake_parse_arguments(arg "${_options}" "" "${_multi}" ${ARGN})

    if(arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "hipblaslt_make_python_command: unexpected arguments: ${arg_UNPARSED_ARGUMENTS} (permitted options: ASAN, TSAN; multi-value keywords: PYTHONPATH_DIRS, RUNTIME_LIB_DIRS, TOOL_BIN_DIRS)")
    endif()

    if(NOT Python3_EXECUTABLE)
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
    endif()

    set(_sanitizer_flag "")
    if(arg_ASAN)
        set(_sanitizer_flag ASAN)
    elseif(arg_TSAN)
        set(_sanitizer_flag TSAN)
    endif()
    set(_sanitizer_options "")
    set(_sanitizer_lib_dirs "")
    if(_sanitizer_flag)
        hipblaslt_detect_sanitizer_runtime(_sanitizer_options _sanitizer_lib_dirs ${_sanitizer_flag})
    endif()

    set(path_separator "$<IF:$<PLATFORM_ID:Windows>,$<SEMICOLON>,:>")
    set(environment ${_sanitizer_options})

    if(arg_PYTHONPATH_DIRS)
        list(JOIN arg_PYTHONPATH_DIRS "${path_separator}" python_path)
        list(APPEND environment "PYTHONPATH=${python_path}")
    endif()

    set(_runtime_lib_dirs ${arg_RUNTIME_LIB_DIRS} ${_sanitizer_lib_dirs})
    set(base_path "$ENV{PATH}")
    if(WIN32)
        string(REPLACE ";" "${path_separator}" base_path "${base_path}")
        set(path_entries ${arg_TOOL_BIN_DIRS} ${_runtime_lib_dirs} "${base_path}")
        list(JOIN path_entries "${path_separator}" base_path)
    else()
        set(path_entries ${arg_TOOL_BIN_DIRS} "${base_path}")
        list(JOIN path_entries ":" base_path)
        if(_runtime_lib_dirs)
            list(JOIN _runtime_lib_dirs ":" runtime_lib_path)
            list(APPEND environment "LD_LIBRARY_PATH=${runtime_lib_path}")
        endif()
    endif()
    list(APPEND environment "PATH=${base_path}")

    set(${out_command}
        "${CMAKE_COMMAND}" -E env ${environment} -- "${Python3_EXECUTABLE}"
        PARENT_SCOPE)
endfunction()

function(hipblaslt_find_sanitizer_runtime_lib out_lib stem)
    set(${out_lib} "" PARENT_SCOPE)
    if(stem STREQUAL "asan")
        set(_flag "-fsanitize=address")
    elseif(stem STREQUAL "tsan")
        set(_flag "-fsanitize=thread")
    else()
        return()
    endif()
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} ${_flag} -shared-libsan -x c /dev/null "-###"
        OUTPUT_VARIABLE _out
        ERROR_VARIABLE _err
    )
    string(REGEX MATCHALL "/[^ \t\r\n\"]*libclang_rt\\.${stem}(-[A-Za-z0-9_]+)?\\.so" _hits "${_out}${_err}")
    if(_hits)
        list(GET _hits 0 _lib)
        if(EXISTS "${_lib}")
            set(${out_lib} "${_lib}" PARENT_SCOPE)
        endif()
    endif()
endfunction()

function(hipblaslt_detect_sanitizer_runtime out_options out_lib_dirs)
    cmake_parse_arguments(arg "ASAN;TSAN" "" "" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "hipblaslt_detect_sanitizer_runtime: unexpected arguments: ${arg_UNPARSED_ARGUMENTS} (permitted options: ASAN, TSAN)")
    endif()
    set(_options "")
    set(_lib_dirs "")
    if(NOT WIN32)
        if((arg_ASAN OR arg_TSAN) AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message(WARNING
                "\n"
                "################################################################\n"
                "  hipblaslt sanitizer build requested with a non-Clang compiler\n"
                "  CMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}\n"
                "  The LD_PRELOAD runtime probe only supports Clang's libclang_rt\n"
                "  sanitizer runtime; no runtime will be preloaded and the\n"
                "  sanitizer will NOT be active. Configure with a Clang/amdclang\n"
                "  toolchain to enable HOST_ASAN/HOST_TSAN.\n"
                "################################################################")
        endif()
        if(arg_ASAN)
            hipblaslt_find_sanitizer_runtime_lib(_asan_lib asan)
            if(_asan_lib)
                # Disable a few asan options to get builds going but these should be addressed
                set(_options "LD_PRELOAD=${_asan_lib}" "ASAN_OPTIONS=detect_leaks=0,new_delete_type_mismatch=0,malloc_context_size=0,quarantine_size_mb=0")
                cmake_path(GET _asan_lib PARENT_PATH _lib_dirs)
            endif()
        elseif(arg_TSAN)
            hipblaslt_find_sanitizer_runtime_lib(_tsan_lib tsan)
            if(_tsan_lib)
                # Disable a few tsan options to get builds going but these should be addressed
                set(_options "LD_PRELOAD=${_tsan_lib}" "TSAN_OPTIONS=detect_leaks=0,new_delete_type_mismatch=0")
                cmake_path(GET _tsan_lib PARENT_PATH _lib_dirs)
            endif()
        endif()
    endif()
    set(${out_options} "${_options}" PARENT_SCOPE)
    set(${out_lib_dirs} "${_lib_dirs}" PARENT_SCOPE)
endfunction()

function(create_device_library)
    set(_opts "")
    set(_one
        TARGET LOGIC_PATH OUTPUT_DIR CXX_COMPILER OFFLOAD_BUNDLER JOBS LOGIC_FILTER
        ASAN YAML_FORMAT NO_COMPRESS EXPERIMENTAL LAZY_LOAD ASM_COMMENTS KEEP_BUILD_TMP ASM_DEBUG
        REQUIRE_GFX1250V0_OVERLAY)
    set(_multi ARCHES PYTHON_COMMAND)
    cmake_parse_arguments(_cdl "${_opts}" "${_one}" "${_multi}" ${ARGN})

    if(_cdl_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "create_device_library: unexpected arguments: ${_cdl_UNPARSED_ARGUMENTS} (permitted single-value keywords: TARGET, LOGIC_PATH, OUTPUT_DIR, CXX_COMPILER, OFFLOAD_BUNDLER, JOBS, LOGIC_FILTER, ASAN, YAML_FORMAT, NO_COMPRESS, EXPERIMENTAL, LAZY_LOAD, ASM_COMMENTS, KEEP_BUILD_TMP, ASM_DEBUG, REQUIRE_GFX1250V0_OVERLAY; multi-value keywords: ARCHES, PYTHON_COMMAND)")
    endif()
    if(NOT _cdl_LOGIC_PATH)
        message(FATAL_ERROR "create_device_library: LOGIC_PATH is required")
    endif()
    if(NOT _cdl_OUTPUT_DIR)
        message(FATAL_ERROR "create_device_library: OUTPUT_DIR is required")
    endif()

    if(NOT _cdl_PYTHON_COMMAND)
        message(FATAL_ERROR
            "create_device_library: PYTHON_COMMAND is required; build it with "
            "hipblaslt_make_python_command() and pass it via the PYTHON_COMMAND argument.")
    endif()

    if(HIPBLASLT_CODEGEN_ROOT)
        set(_codegen_dir "${HIPBLASLT_CODEGEN_ROOT}")
    else()
        get_filename_component(_codegen_dir "${CMAKE_CURRENT_LIST_DIR}/../tensilelite" ABSOLUTE)
    endif()

    if(NOT _cdl_TARGET)
        set(_cdl_TARGET "tensilelite-device-libraries")
    endif()
    if(NOT _cdl_ARCHES)
        set(_cdl_ARCHES ${GPU_TARGETS})
    endif()
    if(NOT _cdl_ARCHES)
        message(FATAL_ERROR "create_device_library: no ARCHES given and GPU_TARGETS is empty")
    endif()
    if(NOT _cdl_CXX_COMPILER)
        set(_cdl_CXX_COMPILER "${CMAKE_CXX_COMPILER}")
    endif()
    if(NOT DEFINED _cdl_OFFLOAD_BUNDLER)
        set(_cdl_OFFLOAD_BUNDLER "${TENSILELITE_OFFLOADBUNDLER}")
    endif()
    if(NOT DEFINED _cdl_JOBS)
        set(_cdl_JOBS "${TENSILELITE_BUILD_PARALLEL_LEVEL}")
    endif()
    if(NOT DEFINED _cdl_LOGIC_FILTER)
        set(_cdl_LOGIC_FILTER "${TENSILELITE_LOGIC_FILTER}")
    endif()
    if(NOT DEFINED _cdl_NO_COMPRESS)
        set(_cdl_NO_COMPRESS "${TENSILELITE_NO_COMPRESS}")
    endif()
    if(NOT DEFINED _cdl_EXPERIMENTAL)
        set(_cdl_EXPERIMENTAL "${TENSILELITE_EXPERIMENTAL}")
    endif()
    if(NOT DEFINED _cdl_ASM_COMMENTS)
        set(_cdl_ASM_COMMENTS "${TENSILELITE_ENABLE_ASM_COMMENTS}")
    endif()
    if(NOT DEFINED _cdl_KEEP_BUILD_TMP)
        set(_cdl_KEEP_BUILD_TMP "${TENSILELITE_KEEP_BUILD_TMP}")
    endif()
    if(NOT DEFINED _cdl_ASM_DEBUG)
        set(_cdl_ASM_DEBUG "${TENSILELITE_ASM_DEBUG}")
    endif()
    if(NOT DEFINED _cdl_YAML_FORMAT)
        set(_cdl_YAML_FORMAT OFF)
        if(TENSILELITE_LIBRARY_FORMAT STREQUAL "yaml")
            set(_cdl_YAML_FORMAT ON)
        endif()
    endif()
    if(NOT DEFINED _cdl_LAZY_LOAD)
        set(_cdl_LAZY_LOAD ON)
        if(Tensile_NO_LAZY_LIBRARY_LOADING)
            set(_cdl_LAZY_LOAD OFF)
        endif()
    endif()

    file(MAKE_DIRECTORY "${_cdl_OUTPUT_DIR}/library")

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
    if(NOT _cdl_LAZY_LOAD)
        list(APPEND _opts_list "--no-lazy-library-loading")
    endif()
    if(NOT _cdl_ASM_COMMENTS)
        list(APPEND _opts_list "--disable-asm-comments")
    endif()

    set(_known_bugs_resource "${_codegen_dir}/Tensile/TensileLogic/known_bugs.yaml")
    set(_tensile_logic_args
        "${_cdl_LOGIC_PATH}"
        --architecture
        "${_arches_semi}"
        --use-bundled-known-bugs
        --check-all
    )
    if(_cdl_REQUIRE_GFX1250V0_OVERLAY)
        list(APPEND _tensile_logic_args --require-gfx1250v0-overlay)
    endif()
    set(_logic_stamp "${CMAKE_CURRENT_BINARY_DIR}/${_cdl_TARGET}-TensileLogic.stamp")
    add_custom_command(
        OUTPUT "${_logic_stamp}"
        COMMENT "Validating library logic (TensileLogic --check-all) for ${_cdl_TARGET} ..."
        COMMAND ${_cdl_PYTHON_COMMAND}
            "${_codegen_dir}/Tensile/bin/TensileLogic"
            ${_tensile_logic_args}
        COMMAND ${CMAKE_COMMAND} -E touch "${_logic_stamp}"
        DEPENDS ${HIPBLASLT_PYTHON_DEPS} "${_known_bugs_resource}"
        VERBATIM
        USES_TERMINAL
    )

    set(_output_stamp "${CMAKE_CURRENT_BINARY_DIR}/${_cdl_TARGET}.stamp")
    set(_tcl_command
        ${_cdl_PYTHON_COMMAND} -m Tensile.TensileCreateLibrary
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
        VERBATIM
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

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include_guard(GLOBAL)

function(hipblaslt_make_python_command out_command)
    set(_options ASAN TSAN)
    set(_one PYTHON_EXECUTABLE)
    set(_multi PYTHONPATH_DIRS RUNTIME_LIB_DIRS TOOL_BIN_DIRS)
    cmake_parse_arguments(arg "${_options}" "${_one}" "${_multi}" ${ARGN})

    if(arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "hipblaslt_make_python_command: unexpected arguments: ${arg_UNPARSED_ARGUMENTS} (permitted options: ASAN, TSAN; single-value keyword: PYTHON_EXECUTABLE; multi-value keywords: PYTHONPATH_DIRS, RUNTIME_LIB_DIRS, TOOL_BIN_DIRS)")
    endif()

    if(arg_PYTHON_EXECUTABLE)
        set(_python_executable "${arg_PYTHON_EXECUTABLE}")
    elseif(Python3_EXECUTABLE)
        set(_python_executable "${Python3_EXECUTABLE}")
    else()
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        set(_python_executable "${Python3_EXECUTABLE}")
    endif()

    set(_sanitizer_flag "")
    if(arg_ASAN)
        set(_sanitizer_flag ASAN)
    elseif(arg_TSAN)
        set(_sanitizer_flag TSAN)
    endif()
    set(_sanitizer_options "")
    set(_sanitizer_lib_dirs "")
    if(NOT "${_sanitizer_flag}" STREQUAL "")
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
        set(_runtime_path_entries ${_runtime_lib_dirs})
        if(DEFINED ENV{LD_LIBRARY_PATH} AND NOT "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
            list(APPEND _runtime_path_entries "$ENV{LD_LIBRARY_PATH}")
        endif()
        if(_runtime_path_entries)
            list(JOIN _runtime_path_entries ":" runtime_lib_path)
            list(APPEND environment "LD_LIBRARY_PATH=${runtime_lib_path}")
        endif()
    endif()
    list(APPEND environment "PATH=${base_path}")

    set(${out_command}
        "${CMAKE_COMMAND}" -E env ${environment} -- "${_python_executable}"
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
                "  toolchain to enable the host sanitizer.\n"
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
    # The prepared GENERATOR owns the Python command, its environment and its
    # host sanitizer runtime; this function adds logic and product options.
    set(_opts "")
    set(_one
        TARGET GENERATOR LOGIC_PATH OUTPUT_DIR CXX_COMPILER OFFLOAD_BUNDLER JOBS LOGIC_FILTER
        ASAN YAML_FORMAT NO_COMPRESS EXPERIMENTAL GEMM_A2A_FUSION LAZY_LOAD ASM_COMMENTS
        KEEP_BUILD_TMP ASM_DEBUG EMIT_OCCUPANCY_PROBE)
    set(_multi ARCHES)
    foreach(_removed IN ITEMS CODEGEN_ROOT PYTHON_EXECUTABLE HOST_ASAN HOST_TSAN)
        if(_removed IN_LIST ARGN)
            message(FATAL_ERROR "create_device_library: ${_removed} was removed; generate from a "
                "packaged TensileLite by passing GENERATOR (see tensilelite_prepare_generator(), "
                "which also selects the host sanitizer).")
        endif()
    endforeach()
    cmake_parse_arguments(_cdl "${_opts}" "${_one}" "${_multi}" ${ARGN})

    if(_cdl_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "create_device_library: unexpected arguments: ${_cdl_UNPARSED_ARGUMENTS} (single-value keywords: TARGET, GENERATOR, LOGIC_PATH, OUTPUT_DIR, CXX_COMPILER, OFFLOAD_BUNDLER, JOBS, LOGIC_FILTER, ASAN, YAML_FORMAT, NO_COMPRESS, EXPERIMENTAL, GEMM_A2A_FUSION, LAZY_LOAD, ASM_COMMENTS, KEEP_BUILD_TMP, ASM_DEBUG, EMIT_OCCUPANCY_PROBE; multi-value keyword: ARCHES)")
    endif()
    if(NOT _cdl_GENERATOR)
        message(FATAL_ERROR "create_device_library: GENERATOR is required; prepare one with "
            "tensilelite_prepare_generator().")
    endif()
    if(NOT _cdl_LOGIC_PATH)
        message(FATAL_ERROR "create_device_library: LOGIC_PATH is required")
    endif()
    if(NOT _cdl_OUTPUT_DIR)
        message(FATAL_ERROR "create_device_library: OUTPUT_DIR is required")
    endif()

    include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/TensileLiteGenerator.cmake")
    tensilelite_generator_command("${_cdl_GENERATOR}" _python_command _generator_dependencies)
    if(NOT IS_DIRECTORY "${_cdl_LOGIC_PATH}")
        message(FATAL_ERROR "create_device_library: LOGIC_PATH is not a directory: ${_cdl_LOGIC_PATH}")
    endif()

    if(NOT _cdl_TARGET)
        set(_cdl_TARGET "tensilelite-device-libraries")
    endif()
    # TENSILELITE_GPU_TARGETS first, GPU_TARGETS after it. The two differ only by
    # an xnack suffix that hip::device cannot take, and TensileLite wants the
    # unstripped list -- but this module is installed for downstream projects,
    # which set GPU_TARGETS and have never heard of the other name. Preferring the
    # new variable without falling back would turn a call that worked into a
    # configure error, for a variable the caller has no way to know about.
    if(NOT _cdl_ARCHES)
        set(_cdl_ARCHES ${TENSILELITE_GPU_TARGETS})
    endif()
    if(NOT _cdl_ARCHES)
        set(_cdl_ARCHES ${GPU_TARGETS})
    endif()
    if(NOT _cdl_ARCHES)
        message(FATAL_ERROR "create_device_library: no ARCHES given and neither "
                            "TENSILELITE_GPU_TARGETS nor GPU_TARGETS is set")
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
    if(NOT DEFINED _cdl_EMIT_OCCUPANCY_PROBE)
        set(_cdl_EMIT_OCCUPANCY_PROBE "${TENSILELITE_EMIT_OCCUPANCY_PROBE}")
    endif()
    if(NOT DEFINED _cdl_YAML_FORMAT)
        set(_cdl_YAML_FORMAT OFF)
    endif()
    if(NOT DEFINED _cdl_LAZY_LOAD)
        set(_cdl_LAZY_LOAD ON)
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
    if(_cdl_GEMM_A2A_FUSION)
        list(APPEND _opts_list "--enable-gemm-a2a-fusion")
    endif()
    if(NOT _cdl_LAZY_LOAD)
        list(APPEND _opts_list "--no-lazy-library-loading")
    endif()
    if(NOT _cdl_ASM_COMMENTS)
        list(APPEND _opts_list "--disable-asm-comments")
    endif()
    if(_cdl_EMIT_OCCUPANCY_PROBE)
        list(APPEND _opts_list "--emit-occupancy-probe")
    endif()

    set(_tensile_logic_args
        "${_cdl_LOGIC_PATH}"
        --architecture
        "${_arches_semi}"
        --use-bundled-known-bugs
        --check-all
    )
    if(_cdl_LOGIC_FILTER)
        list(APPEND _tensile_logic_args "--logic-filter=**/${_cdl_LOGIC_FILTER}.yaml")
    endif()
    set(_codegen_dependencies ${_generator_dependencies})

    # Both codegen steps glob their logic inputs at run time, so without an
    # explicit file list nothing invalidates the stamps and edits ship stale
    # kernels. CONFIGURE_DEPENDS catches added/removed files, DEPENDS catches
    # edits. The generator's own inputs (wheel, native support) come from
    # tensilelite_generator_command().
    file(GLOB_RECURSE _logic_files LIST_DIRECTORIES false CONFIGURE_DEPENDS
         "${_cdl_LOGIC_PATH}/*.yaml"
         "${_cdl_LOGIC_PATH}/*.csv.xz")
    list(APPEND _codegen_dependencies ${_logic_files})

    # ninja only compares mtimes of inputs that still exist, so a *removed* file
    # leaves the stamp clean (nothing is newer), as does a file added with an
    # older mtime. CONFIGURE_DEPENDS keeps the list accurate but ninja never acts
    # on membership alone. Depend on a sorted manifest of the list as well: it is
    # rewritten only when the set changes, and that rewrite dirties the stamp.
    # file(GLOB_RECURSE) orders results lexicographically.
    set(_manifest "${CMAKE_CURRENT_BINARY_DIR}/${_cdl_TARGET}-inputs.manifest")
    string(JOIN "\n" _manifest_content ${_logic_files})
    # file(CONFIGURE) rewrites only when the content differs.
    file(CONFIGURE OUTPUT "${_manifest}" CONTENT "${_manifest_content}\n" @ONLY)
    list(APPEND _codegen_dependencies "${_manifest}")

    set(_logic_stamp "${CMAKE_CURRENT_BINARY_DIR}/${_cdl_TARGET}-TensileLogic.stamp")
    add_custom_command(
        OUTPUT "${_logic_stamp}"
        COMMENT "Validating library logic (tensilelite logic --check-all) for ${_cdl_TARGET} ..."
        COMMAND ${_python_command} -m tensilelite logic
            ${_tensile_logic_args}
        COMMAND ${CMAKE_COMMAND} -E touch "${_logic_stamp}"
        DEPENDS ${_codegen_dependencies}
        VERBATIM
        USES_TERMINAL
    )

    # All architectures go to this one command, a stepping included. Covering a
    # stepping and the architecture it steps from takes more than one run, since
    # a run names its target by the ISA and the two spell one ISA -- but that
    # split is TensileCreateLibrary's: it cannot happen at configure time, when
    # Tensile is not yet importable.
    set(_output_stamp "${CMAKE_CURRENT_BINARY_DIR}/${_cdl_TARGET}.stamp")
    set(_tcl_command
        ${_python_command} -m tensilelite create-library
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
        DEPENDS ${_logic_stamp}
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

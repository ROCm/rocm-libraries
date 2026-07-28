# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Collapses captured process output to a single clipped line, so that it
# survives being embedded in the single message() the report is emitted as.
function(hipblaslt_flatten_process_output out_var text)
    string(REGEX REPLACE "[ \t\r\n]+" " " text "${text}")
    string(STRIP "${text}" text)
    string(LENGTH "${text}" _length)
    if(_length GREATER 500)
        string(SUBSTRING "${text}" 0 500 text)
        string(APPEND text " <truncated>")
    endif()
    set(${out_var} "${text}" PARENT_SCOPE)
endfunction()

# Describes the ELF class, byte order and machine of `path`, or sets `out_var`
# empty when `path` is not an ELF object. The header is read directly because
# binutils is not guaranteed to be installed, while this always works.
function(hipblaslt_read_elf_identity path out_var)
    set(${out_var} "" PARENT_SCOPE)
    if(NOT EXISTS "${path}" OR IS_DIRECTORY "${path}")
        return()
    endif()
    # e_machine ends at offset 20, so 20 bytes cover the magic, class, byte
    # order and machine. A short read means this is not an ELF object either.
    file(READ "${path}" _header HEX LIMIT 20)
    string(LENGTH "${_header}" _length)
    if(NOT _length EQUAL 40 OR NOT _header MATCHES "^7f454c46")
        return()
    endif()
    string(SUBSTRING "${_header}" 8 2 _class)
    string(SUBSTRING "${_header}" 10 2 _data)
    string(SUBSTRING "${_header}" 36 4 _machine)
    set(${out_var} "class 0x${_class} data 0x${_data} machine 0x${_machine}" PARENT_SCOPE)
endfunction()

# Answers one of clang's query flags, reporting the exit code and stderr with the
# value. The configured C++ flags are passed through because a toolchain file may
# force -resource-dir, which moves every answer below; they are meant for a real
# compilation though, so if clang rejects them for an invocation that compiles
# nothing (-Werror over an unused argument, say) the query is retried bare.
function(hipblaslt_query_compiler flag out_value out_report)
    separate_arguments(_flags NATIVE_COMMAND "${CMAKE_CXX_FLAGS}")
    set(_note "")
    while(TRUE)
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} ${_flags} ${flag}
            OUTPUT_VARIABLE _value
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_VARIABLE _error
            RESULT_VARIABLE _result
        )
        if(_result EQUAL 0 OR NOT _flags)
            break()
        endif()
        set(_flags "")
        set(_note " (retried without the configured C++ flags)")
    endwhile()
    if(NOT _result EQUAL 0)
        set(_value "")
    endif()
    hipblaslt_flatten_process_output(_error "${_error}")
    set(${out_value} "${_value}" PARENT_SCOPE)
    set(${out_report}
        "      ${flag} -> '${_value}' (exit code ${_result}${_note}, stderr: '${_error}')\n"
        PARENT_SCOPE)
endfunction()

# Finds a tool able to list *defined* dynamic symbols. Asking the compiler yields
# the llvm tool shipped beside the very clang whose runtime is being validated,
# and searches PATH on its own; like --print-file-name it echoes the query back
# unchanged when it finds nothing. readelf is deliberately not a candidate: its
# output cannot be grepped for a symbol without also matching undefined
# references to it. Being unable to check is fatal, never a quiet pass.
function(hipblaslt_locate_dynamic_symbol_reader out_reader)
    foreach(_name IN ITEMS llvm-nm nm)
        hipblaslt_query_compiler("--print-prog-name=${_name}" _printed _unused)
        if(_printed)
            cmake_path(HAS_PARENT_PATH _printed _has_parent_path)
            if(_has_parent_path AND EXISTS "${_printed}")
                set(${out_reader} "${_printed}" PARENT_SCOPE)
                return()
            endif()
        endif()
    endforeach()
    # Indented from the second line on so that CMake reproduces it verbatim.
    message(FATAL_ERROR
"Cannot check whether a file is the sanitizer runtime.
  Neither llvm-nm nor nm was found by '${CMAKE_CXX_COMPILER} --print-prog-name='
  or on PATH. Install llvm-nm, which an LLVM toolchain ships beside clang, or
  binutils nm. Carrying on would risk preloading a file that merely wears the
  runtime's name and instruments nothing.")
endfunction()

# Decides whether a candidate really is the shared runtime of SANITIZER for the
# target described by REFERENCE_ELF. The expected name is not evidence: an object
# built from an empty translation unit has it, loads without complaint and
# instruments nothing. Neither is merely mentioning the init symbol, because
# every instrumented library references it, which is why READER is asked for the
# symbols the object *defines*.
function(hipblaslt_verify_sanitizer_runtime out_ok out_detail)
    # REFERENCE_ELF is a multi value keyword only so that passing it empty, which
    # happens when no reference object can be read, does not depend on CMP0174.
    cmake_parse_arguments(PARSE_ARGV 2 _arg "" "PATH;SANITIZER;READER" "REFERENCE_ELF")
    set(${out_ok} FALSE PARENT_SCOPE)

    hipblaslt_read_elf_identity("${_arg_PATH}" _elf)
    if(NOT _elf)
        set(${out_detail} "not an ELF object" PARENT_SCOPE)
        return()
    endif()
    if(_arg_REFERENCE_ELF AND NOT _elf STREQUAL "${_arg_REFERENCE_ELF}")
        set(${out_detail} "wrong architecture: ${_elf} cannot be loaded by a process of ${_arg_REFERENCE_ELF}" PARENT_SCOPE)
        return()
    endif()

    set(_symbol "__${_arg_SANITIZER}_init")
    execute_process(
        COMMAND "${_arg_READER}" --dynamic --defined-only "${_arg_PATH}"
        OUTPUT_VARIABLE _symbols
        ERROR_VARIABLE _error
        RESULT_VARIABLE _result
    )
    hipblaslt_flatten_process_output(_error "${_error}")
    if(NOT _result EQUAL 0)
        set(${out_detail} "unverified: ${_arg_READER} exited ${_result} ('${_error}')" PARENT_SCOPE)
        return()
    endif()
    if(NOT _symbols MATCHES "[^A-Za-z0-9_]${_symbol}([^A-Za-z0-9_]|$)")
        set(${out_detail} "not the ${_arg_SANITIZER} runtime: ${_arg_READER} does not report ${_symbol} as defined" PARENT_SCOPE)
        return()
    endif()

    set(${out_ok} TRUE PARENT_SCOPE)
    set(${out_detail} "${_elf}, exports ${_symbol}" PARENT_SCOPE)
endfunction()

# Resolves the shared sanitizer runtime for `sanitizer` (e.g. asan, tsan) and
# sets `out_var` in the caller scope to an absolute path usable as LD_PRELOAD.
# PROBE_PYTHON names the interpreter the code generation steps will run under,
# which is the process that carries the preload and so the reference for the
# architecture candidates have to match. Two stages are tried, each feeding its
# candidate through hipblaslt_verify_sanitizer_runtime before it can be accepted.
#
# A bare soname is never returned: it would be resolved again, against a
# different environment, when the build runs, and a soname the loader then fails
# to find is reported and ignored rather than failing the build. Failing to
# resolve is always fatal.
function(hipblaslt_resolve_sanitizer_runtime sanitizer out_var)
    # PROBE_PYTHON is a multi value keyword so that passing it empty does not
    # depend on policy CMP0174.
    cmake_parse_arguments(PARSE_ARGV 2 _arg "" "" "PROBE_PYTHON")

    hipblaslt_query_compiler(--print-target-triple _triple _query_report)
    hipblaslt_query_compiler(--print-resource-dir _resource_dir _resource_dir_report)
    string(APPEND _query_report "${_resource_dir_report}")

    set(_arch_candidates "")
    if(_triple)
        string(REGEX REPLACE "^([^-]+)-.*$" "\\1" _triple_arch "${_triple}")
        list(APPEND _arch_candidates "${_triple_arch}")
    endif()
    if(CMAKE_SYSTEM_PROCESSOR)
        list(APPEND _arch_candidates "${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    list(REMOVE_DUPLICATES _arch_candidates)

    set(_sonames "")
    foreach(_arch IN LISTS _arch_candidates)
        list(APPEND _sonames "libclang_rt.${sanitizer}-${_arch}.so")
    endforeach()
    # The per-target runtime layout drops the architecture suffix.
    list(APPEND _sonames "libclang_rt.${sanitizer}.so")

    # The runtime has to be loadable by the process that carries the preload, so
    # that process supplies the architecture candidates are checked against.
    set(_reference_elf "")
    set(_reference_source "")
    foreach(_reference IN ITEMS "${_arg_PROBE_PYTHON}" "${CMAKE_CXX_COMPILER}")
        hipblaslt_read_elf_identity("${_reference}" _reference_elf)
        if(_reference_elf)
            set(_reference_source "${_reference}")
            break()
        endif()
    endforeach()

    set(_reader "")
    set(_rejection_report "")

    # Considers one candidate path: rejected candidates are recorded and
    # resolution continues, so a wrong file never shadows a right one.
    macro(_hipblaslt_try_candidate candidate origin)
        if(NOT _reader)
            # Looked up on first use, so a configure that finds no candidate at
            # all never pays for a tool it does not need.
            hipblaslt_locate_dynamic_symbol_reader(_reader)
        endif()
        hipblaslt_verify_sanitizer_runtime(_candidate_ok _candidate_detail
            PATH "${candidate}"
            SANITIZER "${sanitizer}"
            READER "${_reader}"
            REFERENCE_ELF "${_reference_elf}"
        )
        if(NOT _candidate_ok)
            string(APPEND _rejection_report "      ${candidate}\n        (${origin}) ${_candidate_detail}\n")
        endif()
    endmacro()

    foreach(_soname IN LISTS _sonames)
        hipblaslt_query_compiler("--print-file-name=${_soname}" _printed_path _printed_report)
        string(APPEND _query_report "${_printed_report}")
        if(_printed_path)
            # clang echoes the query back unchanged when it cannot find the
            # file, so a bare soname here means the lookup failed.
            cmake_path(HAS_PARENT_PATH _printed_path _has_parent_path)
            if(_has_parent_path AND EXISTS "${_printed_path}")
                _hipblaslt_try_candidate("${_printed_path}" "--print-file-name=${_soname}")
                if(_candidate_ok)
                    message(STATUS "Found ${sanitizer} runtime: ${_printed_path} (${_candidate_detail})")
                    set(${out_var} "${_printed_path}" PARENT_SCOPE)
                    return()
                endif()
            endif()
        endif()
    endforeach()

    # Only wildcards: the exact path of either runtime layout is what clang
    # itself just searched, so all that is left to add is a per-target directory
    # whose name differs from the triple clang normalises to.
    set(_patterns "")
    if(_resource_dir)
        set(_patterns
            "${_resource_dir}/lib/*/libclang_rt.${sanitizer}.so"
            "${_resource_dir}/lib/*/libclang_rt.${sanitizer}-*.so"
        )
    endif()

    set(_pattern_report "")
    foreach(_pattern IN LISTS _patterns)
        file(GLOB _matches "${_pattern}")
        if(NOT _matches)
            string(APPEND _pattern_report "      ${_pattern} -> no match\n")
            continue()
        endif()
        string(APPEND _pattern_report "      ${_pattern} -> ${_matches}\n")
        foreach(_match IN LISTS _matches)
            _hipblaslt_try_candidate("${_match}" "${_pattern}")
            if(_candidate_ok)
                message(STATUS "Found ${sanitizer} runtime in clang resource dir: ${_match} (${_candidate_detail})")
                set(${out_var} "${_match}" PARENT_SCOPE)
                return()
            endif()
        endforeach()
    endforeach()

    if(NOT _patterns)
        set(_pattern_report "      <none: could not determine the clang resource dir>\n")
    endif()
    if(NOT _rejection_report)
        set(_rejection_report "      <none: no candidate file was found at all>\n")
    endif()
    if(_reference_elf)
        string(APPEND _reference_elf " (from ${_reference_source})")
    else()
        set(_reference_elf "<unknown: no reference ELF object could be read>")
    endif()
    set(_listing "")
    if(_resource_dir)
        file(GLOB _entries "${_resource_dir}/lib/*" "${_resource_dir}/lib/*/libclang_rt.*san*")
        foreach(_entry IN LISTS _entries)
            string(APPEND _listing "      ${_entry}\n")
        endforeach()
    endif()
    if(NOT _listing)
        set(_listing "      <empty or unavailable>\n")
    endif()

    # Every line is indented so that CMake reproduces the report verbatim
    # instead of re-wrapping it: this text is the upstream bug report evidence.
    message(FATAL_ERROR
"Failed to locate a usable ${sanitizer} runtime for LD_PRELOAD.
  No file could be confirmed to define __${sanitizer}_init, so build time code
  generation would run without ${sanitizer} coverage; aborting instead.
    Environment:
      C++ compiler        : ${CMAKE_CXX_COMPILER}
      C++ flags           : '${CMAKE_CXX_FLAGS}'
      Target triple       : '${_triple}'
      Architectures tried : ${_arch_candidates}
      Clang resource dir  : '${_resource_dir}'
      Target ELF identity : ${_reference_elf}
    Compiler queries:
${_query_report}    Paths searched:
${_pattern_report}    Candidates rejected:
${_rejection_report}    Clang resource dir contents:
${_listing}  If a libclang_rt.${sanitizer} object appears in those contents it is packaged
  correctly and this resolver's search or acceptance check needs fixing; if none
  appears anywhere above, that is a toolchain packaging bug and the output above
  is the evidence for it.")
endfunction()

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Windows builds embed a VERSIONINFO resource, so they need a resource compiler that actually works.
# Finding a binary is not enough to know that it does.
#
# With a clang toolchain, CMake compiles .rc files in two stages: preprocess with the C or C++
# compiler, then hand the preprocessed file to llvm-rc, which (LLVM >= 16) preprocesses it a second
# time. A preprocessor whose line markers leave Windows path separators unescaped
#
#     # 1 "C:\Program Files (x86)\Windows Kits\10\Include\...\winver.h" 1 3
#
# makes that second pass fail on the invalid escapes and emit several kilobytes of diagnostics.
# `cmake -E cmake_llvm_rc` does not drain that pipe, so llvm-rc blocks writing to a full pipe and the
# resource compile deadlocks rather than failing - the build stalls with no output and no error.
#
# So the pipeline is probed, not assumed:
#
#   1. Compile a probe resource with the configured pipeline, under a timeout.
#   2. If that does not work, fall back to rc.exe from the Windows Kits, which compiles .rc directly
#      and has no preprocessing pass to break.
#   3. If neither works, fail configuration with the paths needed to fix it.

set(HIPDNN_RC_PROBE_TIMEOUT
    60
    CACHE STRING "Seconds to wait for the Windows resource-compiler probe before declaring it broken"
)

set(HIPDNN_WINDOWS_SDK_ROOT
    ""
    CACHE PATH
          "Windows Kits root (e.g. C:/Program Files (x86)/Windows Kits/10) used to locate rc.exe when
           the toolchain's llvm-rc pipeline is unusable. Auto-detected when empty."
)

set(HIPDNN_RC_PROBE_DIR "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/hipdnn-rc-probe")
set(HIPDNN_RC_SINGLE_STAGE_RULE
    "<CMAKE_RC_COMPILER> <DEFINES> <INCLUDES> <FLAGS> /fo <OBJECT> <SOURCE>"
)

# Writes the probe resource. The SDK include is what drags in the line markers that break llvm-rc, so
# a probe without it would pass where the real backend.rc deadlocks.
function(_hipdnn_write_rc_probe)
    file(MAKE_DIRECTORY "${HIPDNN_RC_PROBE_DIR}")
    file(
        WRITE "${HIPDNN_RC_PROBE_DIR}/probe.rc"
        "#include <winver.h>\n"
        "VS_VERSION_INFO VERSIONINFO\n"
        "FILEVERSION 0,0,0,0\n"
        "PRODUCTVERSION 0,0,0,0\n"
        "FILEOS VOS_NT_WINDOWS32\n"
        "FILETYPE VFT_DLL\n"
        "BEGIN\n"
        "    BLOCK \"VarFileInfo\"\n"
        "    BEGIN\n"
        "        VALUE \"Translation\", 0x409, 1200\n"
        "    END\n"
        "END\n"
    )
endfunction()

# Compiles the probe resource with an explicit (compiler, rule, flags) triple. Sets OUT_DETAIL to a
# one-line reason on failure and to the empty string on success.
function(_hipdnn_probe_rc_pipeline RC_COMPILER RC_RULE RC_FLAGS OUT_DETAIL)
    set(_object "${HIPDNN_RC_PROBE_DIR}/probe.res")
    file(REMOVE "${_object}")

    # Expand the rule instead of hard-coding a command, so whatever pipeline this build configured is
    # what gets tested.
    set(_rule "${RC_RULE}")
    string(REPLACE "<CMAKE_COMMAND>" "\"${CMAKE_COMMAND}\"" _rule "${_rule}")
    string(REPLACE "<CMAKE_RC_COMPILER>" "\"${RC_COMPILER}\"" _rule "${_rule}")
    string(REPLACE "<CMAKE_C_COMPILER>" "\"${CMAKE_C_COMPILER}\"" _rule "${_rule}")
    string(REPLACE "<CMAKE_CXX_COMPILER>" "\"${CMAKE_CXX_COMPILER}\"" _rule "${_rule}")
    string(REPLACE "<SOURCE_DIR>" "\"${HIPDNN_RC_PROBE_DIR}\"" _rule "${_rule}")
    string(REPLACE "<SOURCE>" "\"${HIPDNN_RC_PROBE_DIR}/probe.rc\"" _rule "${_rule}")
    string(REPLACE "<OBJECT>" "\"${_object}\"" _rule "${_rule}")
    string(REPLACE "<DEFINES>" "" _rule "${_rule}")
    string(REPLACE "<INCLUDES>" "" _rule "${_rule}")
    string(REPLACE "<FLAGS>" "${RC_FLAGS}" _rule "${_rule}")
    separate_arguments(_command NATIVE_COMMAND "${_rule}")

    execute_process(
        COMMAND ${_command}
        WORKING_DIRECTORY "${HIPDNN_RC_PROBE_DIR}"
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _output
        ERROR_VARIABLE _error
        TIMEOUT ${HIPDNN_RC_PROBE_TIMEOUT}
    )

    if(_result EQUAL 0 AND EXISTS "${_object}")
        set(${OUT_DETAIL} "" PARENT_SCOPE)
        return()
    endif()

    if(_result MATCHES "[Tt]imeout")
        string(CONCAT _detail "it did not finish within ${HIPDNN_RC_PROBE_TIMEOUT}s (deadlocked); "
                      "a stray resource-compiler process may still be running"
        )
    else()
        string(STRIP "${_error}" _error)
        string(REGEX REPLACE "\n.*" "" _error "${_error}")
        set(_detail "it failed (${_result}): ${_error}")
    endif()
    set(${OUT_DETAIL} "${_detail}" PARENT_SCOPE)
endfunction()

# Best-effort explanation for a failed two-stage probe, so the report names a cause, not a symptom.
function(_hipdnn_diagnose_rc_preprocessor OUT_VAR)
    set(${OUT_VAR} "" PARENT_SCOPE)

    if(NOT CMAKE_RC_COMPILE_OBJECT MATCHES "cmake_llvm_rc")
        return()
    endif()

    if(CMAKE_RC_COMPILE_OBJECT MATCHES "<CMAKE_C_COMPILER>")
        set(_preprocessor "${CMAKE_C_COMPILER}")
    elseif(CMAKE_RC_COMPILE_OBJECT MATCHES "<CMAKE_CXX_COMPILER>")
        set(_preprocessor "${CMAKE_CXX_COMPILER}")
    else()
        return()
    endif()

    execute_process(
        COMMAND "${_preprocessor}" -x c -E "${HIPDNN_RC_PROBE_DIR}/probe.rc"
        OUTPUT_FILE "${HIPDNN_RC_PROBE_DIR}/probe.markers.pp"
        ERROR_QUIET
        TIMEOUT ${HIPDNN_RC_PROBE_TIMEOUT}
        RESULT_VARIABLE _result
    )
    if(NOT _result EQUAL 0)
        return()
    endif()

    # A well-behaved preprocessor doubles the separators it emits in line markers. Keep the markers
    # naming a Windows path, then drop the ones that are correctly escaped.
    file(STRINGS "${HIPDNN_RC_PROBE_DIR}/probe.markers.pp" _markers REGEX "^# [0-9]+ \"[A-Za-z]:\\\\")
    list(FILTER _markers EXCLUDE REGEX "\\\\\\\\")
    if(_markers)
        list(GET _markers 0 _example)
        string(
            CONCAT _hint "Cause: ${_preprocessor} emits unescaped path separators in preprocessor "
                   "line markers, which llvm-rc cannot parse:\n"
                   "  ${_example}\n"
        )
        set(${OUT_VAR} "${_hint}" PARENT_SCOPE)
    endif()
endfunction()

# Windows Kits roots to search, in preference order.
function(_hipdnn_collect_sdk_roots OUT_ROOTS)
    if(HIPDNN_WINDOWS_SDK_ROOT)
        set(${OUT_ROOTS} "${HIPDNN_WINDOWS_SDK_ROOT}" PARENT_SCOPE)
        return()
    endif()

    set(_roots "$ENV{WindowsSdkDir}" "$ENV{ProgramFiles\(x86\)}/Windows Kits/10"
               "C:/Program Files (x86)/Windows Kits/10" "C:/Program Files/Windows Kits/10"
    )
    list(REMOVE_ITEM _roots "")
    list(REMOVE_DUPLICATES _roots)
    set(${OUT_ROOTS} "${_roots}" PARENT_SCOPE)
endfunction()

# Locates rc.exe and its SDK include directories under a Windows Kits root, newest version first.
function(_hipdnn_find_windows_sdk_rc SDK_ROOT OUT_RC OUT_FLAGS)
    set(${OUT_RC} "" PARENT_SCOPE)
    set(${OUT_FLAGS} "" PARENT_SCOPE)

    if(NOT IS_DIRECTORY "${SDK_ROOT}")
        return()
    endif()

    file(GLOB _versions RELATIVE "${SDK_ROOT}/bin" "${SDK_ROOT}/bin/*")
    list(FILTER _versions INCLUDE REGEX "^10\\.")
    list(SORT _versions COMPARE NATURAL ORDER DESCENDING)

    foreach(_version IN LISTS _versions)
        set(_include "${SDK_ROOT}/Include/${_version}")
        if(NOT EXISTS "${_include}/um/winver.h")
            continue()
        endif()
        foreach(_arch IN ITEMS x64 x86)
            if(EXISTS "${SDK_ROOT}/bin/${_version}/${_arch}/rc.exe")
                # rc.exe has no toolchain of its own to find the SDK headers with, and the ROCm clang
                # toolchain never establishes a VS developer environment, so pass them explicitly.
                set(${OUT_RC} "${SDK_ROOT}/bin/${_version}/${_arch}/rc.exe" PARENT_SCOPE)
                set(${OUT_FLAGS} "-I\"${_include}/um\" -I\"${_include}/shared\"" PARENT_SCOPE)
                return()
            endif()
        endforeach()
    endforeach()
endfunction()

# Stops configuration, naming what failed and how to supply a working resource compiler.
function(_hipdnn_fail_no_rc CONFIGURED_DETAIL HINT SDK_ROOTS)
    string(
        CONCAT _message
        "\n"
        "hipDNN: no usable Windows resource compiler.\n"
        "The configured resource compiler does not work - ${CONFIGURED_DETAIL}\n"
        "${HINT}"
        "No working rc.exe was found under: ${SDK_ROOTS}\n"
        "\n"
        "Point the build at a Windows Kits installation, for example:\n"
        "  cmake --preset <preset> -DHIPDNN_WINDOWS_SDK_ROOT=\"C:/Program Files (x86)/Windows Kits/10\"\n"
        "That root must contain bin/<version>/<arch>/rc.exe and Include/<version>/um/winver.h.\n"
        "\n"
        "Alternatively, select a resource compiler explicitly. Use forward slashes, and pass the SDK "
        "include directories, which rc.exe cannot find on its own:\n"
        "  -DCMAKE_RC_COMPILER=\"C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64/rc.exe\"\n"
        "  -DCMAKE_RC_FLAGS=\"-I'<kits>/Include/10.0.22621.0/um' -I'<kits>/Include/10.0.22621.0/shared'\"\n"
    )
    message(FATAL_ERROR "${_message}")
endfunction()

# Adopts the Windows Kits resource compiler for every target below this directory, or returns empty
# OUT_RC when no root yields a working one.
function(_hipdnn_try_sdk_fallback SDK_ROOTS OUT_RC OUT_FLAGS)
    set(${OUT_RC} "" PARENT_SCOPE)
    set(${OUT_FLAGS} "" PARENT_SCOPE)

    foreach(_root IN LISTS SDK_ROOTS)
        file(TO_CMAKE_PATH "${_root}" _root)
        _hipdnn_find_windows_sdk_rc("${_root}" _sdk_rc _sdk_flags)
        if(NOT _sdk_rc)
            continue()
        endif()

        _hipdnn_probe_rc_pipeline(
            "${_sdk_rc}" "${HIPDNN_RC_SINGLE_STAGE_RULE}" "${_sdk_flags}" _detail
        )
        if(NOT _detail)
            set(${OUT_RC} "${_sdk_rc}" PARENT_SCOPE)
            set(${OUT_FLAGS} "${_sdk_flags}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
endfunction()

# Switches this directory and everything below it to a resource compiler that needs no preprocessing
# pass. A macro, not a function, so PARENT_SCOPE lands in the including directory rather than in an
# intermediate function scope.
macro(_hipdnn_adopt_sdk_rc ADOPT_RC ADOPT_FLAGS)
    set(CMAKE_RC_COMPILER "${ADOPT_RC}" PARENT_SCOPE)
    set(CMAKE_RC_COMPILE_OBJECT "${HIPDNN_RC_SINGLE_STAGE_RULE}" PARENT_SCOPE)
    set(CMAKE_RC_FLAGS "${ADOPT_FLAGS}" PARENT_SCOPE)
    # rc.exe emits no depfile, so resource dependency scanning goes away with it.
    set(CMAKE_DEPFILE_FLAGS_RC "" PARENT_SCOPE)
endmacro()

# Records a verdict so later configures replay it instead of probing again. Empty RESOLVED_FLAGS
# means the toolchain's own pipeline works and nothing needs to be overridden.
function(_hipdnn_store_rc_resolution STAMP RESOLVED_RC RESOLVED_FLAGS)
    set(HIPDNN_RC_STAMP "${STAMP}" CACHE INTERNAL "Pipeline the resource-compiler probe resolved")
    set(HIPDNN_RC_COMPILER_RESOLVED "${RESOLVED_RC}" CACHE INTERNAL "Resolved resource compiler")
    set(HIPDNN_RC_FLAGS_RESOLVED "${RESOLVED_FLAGS}" CACHE INTERNAL "Resolved resource-compiler flags")
endfunction()

# Establishes a working Windows resource-compile pipeline or fails configuration.
function(hipdnn_check_resource_compiler)
    if(NOT WIN32)
        return()
    endif()

    # Probing costs a process launch on a healthy toolchain and a full timeout on a broken one, and
    # this file is re-run by every Ninja regeneration. Resolve once per pipeline and replay the
    # verdict afterwards; the stamp covers every input that could change it.
    string(CONCAT _stamp "${CMAKE_RC_COMPILER}|${CMAKE_RC_COMPILE_OBJECT}|${CMAKE_C_COMPILER}|"
                  "${CMAKE_CXX_COMPILER}|${CMAKE_RC_FLAGS}|${HIPDNN_WINDOWS_SDK_ROOT}"
    )
    if(HIPDNN_RC_STAMP STREQUAL "${_stamp}" AND HIPDNN_RC_COMPILER_RESOLVED)
        if(HIPDNN_RC_FLAGS_RESOLVED)
            _hipdnn_adopt_sdk_rc("${HIPDNN_RC_COMPILER_RESOLVED}" "${HIPDNN_RC_FLAGS_RESOLVED}")
        endif()
        return()
    endif()

    _hipdnn_write_rc_probe()
    set(_configured_detail "no resource compiler was found")
    set(_hint "")

    # 1. The pipeline the toolchain configured, usually llvm-rc driven by cmake -E cmake_llvm_rc.
    if(CMAKE_RC_COMPILER AND NOT CMAKE_RC_COMPILER MATCHES "NOTREQUIRED|NOTFOUND")
        set(_rule "${CMAKE_RC_COMPILE_OBJECT}")
        if(NOT _rule)
            set(_rule "${HIPDNN_RC_SINGLE_STAGE_RULE}")
        endif()
        _hipdnn_probe_rc_pipeline(
            "${CMAKE_RC_COMPILER}" "${_rule}" "${CMAKE_RC_FLAGS}" _configured_detail
        )
        if(NOT _configured_detail)
            _hipdnn_store_rc_resolution("${_stamp}" "${CMAKE_RC_COMPILER}" "")
            message(STATUS "hipDNN: resource compiler: ${CMAKE_RC_COMPILER}")
            return()
        endif()
        _hipdnn_diagnose_rc_preprocessor(_hint)
    endif()

    # 2. rc.exe from the Windows Kits.
    _hipdnn_collect_sdk_roots(_sdk_roots)
    _hipdnn_try_sdk_fallback("${_sdk_roots}" _sdk_rc _sdk_flags)
    if(NOT _sdk_rc)
        _hipdnn_fail_no_rc("${_configured_detail}" "${_hint}" "${_sdk_roots}")
    endif()

    _hipdnn_store_rc_resolution("${_stamp}" "${_sdk_rc}" "${_sdk_flags}")
    _hipdnn_adopt_sdk_rc("${_sdk_rc}" "${_sdk_flags}")
    message(
        WARNING
            "\n"
            "hipDNN: the toolchain's resource compiler does not work - ${_configured_detail}\n"
            "${_hint}"
            "Falling back to the Windows Kits resource compiler:\n"
            "  ${_sdk_rc}\n"
            "Resource dependency scanning is disabled with this compiler (rc.exe emits no depfile).\n"
    )
endfunction()

hipdnn_check_resource_compiler()

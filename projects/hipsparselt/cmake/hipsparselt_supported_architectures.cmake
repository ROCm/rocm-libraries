# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Base architectures - used when "all" is specified for GPU_TARGETS
set(BASE_ARCHITECTURES "")

# All supported architectures including xnack variants - used for validation of GPU_TARGETS
set(SUPPORTED_ARCHITECTURES "")

if(HIPSPARSELT_ENABLE_ASAN OR BUILD_ADDRESS_SANITIZER)
    # For address sanitizer builds, base and supported are the same
    list(APPEND BASE_ARCHITECTURES "gfx942:xnack+" "gfx950:xnack+" "gfx1250" "gfx1250-strict")
    set(SUPPORTED_ARCHITECTURES ${BASE_ARCHITECTURES})
else()
    list(APPEND BASE_ARCHITECTURES "gfx942" "gfx950" "gfx1250" "gfx1250-strict")

    set(SUPPORTED_ARCHITECTURES ${BASE_ARCHITECTURES})
    list(APPEND SUPPORTED_ARCHITECTURES "gfx942:xnack+" "gfx942:xnack-" "gfx950:xnack+"
         "gfx950:xnack-" "gfx1250" "gfx1250-strict" "gfx1250:xnack-" "gfx1250-strict:xnack-"
    )
endif()

# .rst: Validates that all specified GPU targets are supported.
#
# ``hipsparselt_validate_gpu_targets(<targets>)``
#
# Checks each target in the list against supported architectures. Throws FATAL_ERROR if any
# unsupported target is found.
function(hipsparselt_validate_gpu_targets targets)
    set(supported_list ${SUPPORTED_ARCHITECTURES})
    set(target_list ${targets})

    string(REGEX REPLACE ";" " " supported_flat "${supported_list}")
    string(REGEX REPLACE " +" ";" supported_list "${supported_flat}")

    string(REGEX REPLACE ";" " " target_flat "${target_list}")
    string(REGEX REPLACE " +" ";" target_list "${target_flat}")

    foreach(target IN LISTS target_list)
        list(FIND supported_list "${target}" idx)
        if(idx EQUAL -1)
            message(
                FATAL_ERROR
                    "Unsupported GPU target: ${target}\nSupported targets are: ${supported_list}"
            )
        endif()
    endforeach()
endfunction()

# .rst: Returns the list of base GPU architectures.
#
# ``hipsparselt_get_base_architectures(<output_var>)``
#
# Sets <output_var> to the list of base architectures used when "all" is specified.
function(hipsparselt_get_base_architectures output_var)
    set(${output_var} ${BASE_ARCHITECTURES} PARENT_SCOPE)
endfunction()

# .rst: Returns the list of all supported GPU architectures.
#
# ``hipsparselt_get_supported_architectures(<output_var>)``
#
# Sets <output_var> to the full list including xnack variants.
function(hipsparselt_get_supported_architectures output_var)
    set(${output_var} ${SUPPORTED_ARCHITECTURES} PARENT_SCOPE)
endfunction()



# The three functions below are a deliberate copy of their tensilelite_*
# counterparts in hipblaslt/cmake/tensilelite_supported_architectures.cmake, and
# of compilerTargetOf() in hipblaslt/tensilelite/tensilelite/Common/Architectures.py.
# All ask the compiler the same question and must keep the same answer, or
# hip::device and the Tensile code objects part ways.
#
# Copied rather than shared: that module also defines SUPPORTED_ARCHITECTURES and
# BASE_ARCHITECTURES at include time, and hipSPARSELt's lists are narrower, so
# including it would silently widen what hipsparselt_validate_gpu_targets
# accepts. It is also only on CMAKE_MODULE_PATH well after this runs.

# Whether the compiler driver accepts a target ID. Asked with -###, a driver dry
# run: the target ID is validated but nothing is compiled, so a probe costs ~10ms.
# Answers are cached, so each distinct ID is asked once per build tree.
#
# A compiler that is not an AMDGPU-aware clang rejects every ID, including bare
# ones. Callers must treat a lone "no" as "cannot judge" rather than as a verdict
# about the qualifier; see hipsparselt_normalize_xnack_archs.
function(hipsparselt_target_id_supported output_var arch)
    string(MAKE_C_IDENTIFIER "_HIPSPARSELT_TARGET_ID_OK_${arch}" _cache_key)
    if(DEFINED ${_cache_key})
        set(${output_var} "${${_cache_key}}" PARENT_SCOPE)
        return()
    endif()

    # clang needs an input file, but never opens it under -###. Kept empty and
    # reused so the probe does not depend on anything in the source tree.
    set(_probe_src "${CMAKE_BINARY_DIR}/hipsparselt_target_id_probe.cpp")
    if(NOT EXISTS "${_probe_src}")
        file(WRITE "${_probe_src}" "")
    endif()

    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}"
                --target=amdgcn-amd-amdhsa "-mcpu=${arch}"
                # Quoted: unquoted, CMake reads the # as a comment and eats the
                # rest of the line, leaving clang to read from stdin and fail.
                "-###" -c -x c++ "${_probe_src}"
        RESULT_VARIABLE _rc
        OUTPUT_QUIET ERROR_QUIET)

    if(_rc EQUAL 0)
        set(_ok ON)
    else()
        set(_ok OFF)
    endif()
    set(${_cache_key} "${_ok}" CACHE INTERNAL "clang accepts -mcpu=${arch}")
    set(${output_var} "${_ok}" PARENT_SCOPE)
endfunction()

# An arch with every xnack qualifier removed, other qualifiers kept in order.
function(hipsparselt_strip_xnack output_var arch)
    string(REPLACE ":" ";" _parts "${arch}")
    list(POP_FRONT _parts _base)
    set(_kept "")
    foreach(_qualifier IN LISTS _parts)
        if(NOT _qualifier MATCHES "^xnack")
            list(APPEND _kept "${_qualifier}")
        endif()
    endforeach()
    list(JOIN _kept ":" _quals)
    if(_quals)
        set(${output_var} "${_base}:${_quals}" PARENT_SCOPE)
    else()
        set(${output_var} "${_base}" PARENT_SCOPE)
    endif()
endfunction()

# Drops the xnack qualifier from the targets the compiler will not take it on.
# Each arch is put to the compiler, and if the target ID is refused the bare
# form is tried instead.
#
# Detected rather than tabulated, so no list here tracks which target IDs the
# compiler takes and a change on that side needs no edit.
#
# Only the xnack token goes, so other qualifiers survive, and an arch refused
# for some other reason -- a typo in the processor name -- has to stay an error
# rather than quietly degrade to its base name. So the stripped form is asked
# about too, and an arch is left exactly as given unless dropping xnack is what
# makes it acceptable.
function(hipsparselt_normalize_xnack_archs output_var)
    set(_normalized "")
    foreach(_arch IN LISTS ARGN)
        set(_result "${_arch}")
        hipsparselt_target_id_supported(_accepted "${_arch}")
        if(NOT _accepted)
            hipsparselt_strip_xnack(_stripped "${_arch}")
            if(NOT _stripped STREQUAL _arch)
                hipsparselt_target_id_supported(_stripped_accepted "${_stripped}")
                if(_stripped_accepted)
                    set(_result "${_stripped}")
                endif()
            endif()
        endif()
        list(APPEND _normalized "${_result}")
    endforeach()
    set(${output_var} "${_normalized}" PARENT_SCOPE)
endfunction()
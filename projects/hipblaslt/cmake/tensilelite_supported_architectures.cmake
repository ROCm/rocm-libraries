# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# All supported architectures including xnack variants - used for validation of GPU_TARGETS
set(SUPPORTED_ARCHITECTURES
    "gfx908"
    "gfx90a"
    "gfx942"
    "gfx950"
    "gfx1100"
    "gfx1101"
    "gfx1102"
    "gfx1103"
    "gfx1150"
    "gfx1151"
    "gfx1152"
    "gfx1153"
    "gfx1170"
    "gfx1171"
    "gfx1200"
    "gfx1201"
    "gfx1250"
    "gfx1250-strict"
    "gfx908:xnack+"
    "gfx908:xnack-"
    "gfx90a:xnack+"
    "gfx90a:xnack-"
    "gfx942:xnack+"
    "gfx950:xnack+"
    "gfx1250:xnack-"
    "gfx1250-strict:xnack-"
)

# Base architectures - used when "all" is specified for GPU_TARGETS
# Different base architectures will be chosen depending on the build mode.
# All base architectures must be in the SUPPORTED_ARCHITECTURES list.
set(BASE_ARCHITECTURES)

# Note:
# gfx10XX architectures (e.g., gfx1010, gfx1011, gfx1030, etc...) are technically supported by tensilelite,
# but are NOT included in the default "all" build in hipBLASLt. This is because "extops" builds are not supported
# for legacy devices. Including these architectures would result in build failures or incomplete feature support.
if(HIPBLASLT_ENABLE_ASAN OR THEROCK_SANITIZER STREQUAL "ASAN")
    # For device address sanitizer builds, "all" is just the architectures that
    # support xnack+.
    set(BASE_ARCHITECTURES
        "gfx908:xnack+"
        "gfx90a:xnack+"
        "gfx942:xnack+"
        "gfx950:xnack+"
        "gfx1250"
        "gfx1250-strict"
        )
else()
    # Host-only ASAN and non-sanitized builds use the regular architecture list.
    set(BASE_ARCHITECTURES
        "gfx908"
        "gfx90a"
        "gfx942"
        "gfx950"
        "gfx1100"
        "gfx1101"
        "gfx1103"
        "gfx1150"
        "gfx1151"
        "gfx1152"
        "gfx1153"
        "gfx1170"
        "gfx1171"
        "gfx1200"
        "gfx1201"
        "gfx1250"
        "gfx1250-strict")
endif()

# Validate that all BASE_ARCHITECTURES are in the SUPPORTED_ARCHITECTURES list
# so that we eagerly fail if these ever get out of sync.
block()
    foreach(_arch ${BASE_ARCHITECTURES})
        if(NOT "${_arch}" IN_LIST SUPPORTED_ARCHITECTURES)
            message(FATAL_ERROR "Assertion failed: ${_arch} not in SUPPORTED_ARCHITECTURES")
        endif()
    endforeach()
endblock()

function(tensilelite_validate_gpu_targets targets)
    set(supported_list ${SUPPORTED_ARCHITECTURES})
    set(target_list ${targets})

    string(REGEX REPLACE ";" " " supported_flat "${supported_list}")
    string(REGEX REPLACE " +" ";" supported_list "${supported_flat}")

    string(REGEX REPLACE ";" " " target_flat "${target_list}")
    string(REGEX REPLACE " +" ";" target_list "${target_flat}")

    foreach(target IN LISTS target_list)
        list(FIND supported_list "${target}" idx)
        if(idx EQUAL -1)
            message(FATAL_ERROR "Unsupported GPU target: ${target}\nSupported targets are: ${supported_list}")
        endif()
    endforeach()
endfunction()

function(tensilelite_get_base_architectures output_var)
    set(${output_var} ${BASE_ARCHITECTURES} PARENT_SCOPE)
endfunction()

function(tensilelite_get_supported_architectures output_var)
    set(${output_var} ${SUPPORTED_ARCHITECTURES} PARENT_SCOPE)
endfunction()

function(tensilelite_sanitizer_requires_xnack output_var)
    if(HIPBLASLT_ENABLE_ASAN OR THEROCK_SANITIZER STREQUAL "ASAN")
        set(${output_var} ON PARENT_SCOPE)
    else()
        set(${output_var} OFF PARENT_SCOPE)
    endif()
endfunction()

function(tensilelite_offload_target output_var arch)
    set(_xnack_capable gfx908 gfx90a gfx942 gfx950) #gfx1250 support xnack_any so don't need to explicit ":xnack+"
    set(_target "${arch}")
    tensilelite_sanitizer_requires_xnack(_requires_xnack)
    if(_requires_xnack AND NOT "${arch}" MATCHES ":" AND "${arch}" IN_LIST _xnack_capable)
        set(_target "${arch}:xnack+")
    endif()
    set(${output_var} "${_target}" PARENT_SCOPE)
endfunction()

function(tensilelite_normalize_targets output_var)
    set(_normalized "")
    foreach(_arch IN LISTS ARGN)
        tensilelite_offload_target(_target "${_arch}")
        list(APPEND _normalized "${_target}")
    endforeach()
    set(${output_var} "${_normalized}" PARENT_SCOPE)
endfunction()

# Whether the compiler driver accepts a target ID. Asked with -###, a driver dry
# run: the target ID is validated but nothing is compiled, so a probe costs ~10ms.
# Answers are cached, so each distinct ID is asked once per build tree.
#
# A compiler that is not an AMDGPU-aware clang rejects every ID, including bare
# ones. Callers must treat a lone "no" as "cannot judge" rather than as a verdict
# about the qualifier; see tensilelite_normalize_xnack_archs.
function(tensilelite_target_id_supported output_var arch)
    string(MAKE_C_IDENTIFIER "_TENSILELITE_TARGET_ID_OK_${arch}" _cache_key)
    if(DEFINED ${_cache_key})
        set(${output_var} "${${_cache_key}}" PARENT_SCOPE)
        return()
    endif()

    # clang needs an input file, but never opens it under -###. Kept empty and
    # reused so the probe does not depend on anything in the source tree.
    set(_probe_src "${CMAKE_BINARY_DIR}/tensilelite_target_id_probe.cpp")
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
function(tensilelite_strip_xnack output_var arch)
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
# compiler takes and a change on that side needs no edit. Kept in step with
# compilerTargetOf() in tensilelite/Common/Architectures.py, which asks the same
# question for the TensileLite side -- the two must agree, or hip::device and
# the Tensile code objects end up built for different targets.
#
# Only the xnack token goes, so other qualifiers survive, and an arch refused
# for some other reason -- a typo in the processor name -- has to stay an error
# rather than quietly degrade to its base name. So the stripped form is asked
# about too, and an arch is left exactly as given unless dropping xnack is what
# makes it acceptable.
function(tensilelite_normalize_xnack_archs output_var)
    set(_normalized "")
    foreach(_arch IN LISTS ARGN)
        set(_result "${_arch}")
        tensilelite_target_id_supported(_accepted "${_arch}")
        if(NOT _accepted)
            tensilelite_strip_xnack(_stripped "${_arch}")
            if(NOT _stripped STREQUAL _arch)
                tensilelite_target_id_supported(_stripped_accepted "${_stripped}")
                if(_stripped_accepted)
                    set(_result "${_stripped}")
                endif()
            endif()
        endif()
        list(APPEND _normalized "${_result}")
    endforeach()
    set(${output_var} "${_normalized}" PARENT_SCOPE)
endfunction()
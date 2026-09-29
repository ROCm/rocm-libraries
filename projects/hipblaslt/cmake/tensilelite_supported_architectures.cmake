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
)

# Build-time aliases: a stepping's kernels under the name a runtime that
# predates the stepping reports. Mirrors REVISION_ALIASES in
# Tensile/Common/Architectures.py. Deliberately not in SUPPORTED_ARCHITECTURES:
# these are never names a user asks for.
set(REVISION_ALIASES
    "gfx1250v0"
)

# Base architectures - used when "all" is specified for GPU_TARGETS
# Different base architectures will be chosen depending on the build mode.
# All base architectures must be in the SUPPORTED_ARCHITECTURES list.
set(BASE_ARCHITECTURES)

# Note:
# gfx10XX architectures (e.g., gfx1010, gfx1011, gfx1030, etc...) are technically supported by tensilelite,
# but are NOT included in the default "all" build in hipBLASLt. This is because "extops" builds are not supported
# for legacy devices. Including these architectures would result in build failures or incomplete feature support.
if(HIPBLASLT_ENABLE_ASAN OR THEROCK_SANITIZER STREQUAL "ASAN" OR THEROCK_SANITIZER STREQUAL "HOST_ASAN")
    # For address sanitizer builds, "all" is just the architectures that
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
    # For non address sanitizer builds, "all" is non-xnack architectures.
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

# Each named architecture's revision aliases, appended after it.
#
# The CMake half of withRevisionAliasesOfNamedArchs in
# Tensile/Common/Architectures.py. Naming gfx1250 has to build gfx1250v0 as
# well: the build cannot know which silicon revision it will run on, and a rev-0
# part reported under the base name can load nothing else. tensilelite fans out
# on its own for the kernels it generates, but everything CMake builds -- the
# source kernels, the ExtOp library, the matrix-transform kernels and the
# clients' own device code -- follows GPU_TARGETS, which never saw the
# expansion. The result was a revision subtree holding only Tensile code
# objects, so the runtime selected it and then failed to load the helper kernels
# that were never built for it.
#
# The stepping itself is not added. It is loadable only where its own name is
# reported, so putting it in every gfx1250 build would produce a subtree most
# systems can never open; ask for it by name instead:
#   -a "gfx1250;gfx1250-strict"
#
# Idempotent, so it is safe on a list that already names the alias -- which is
# what BASE_ARCHITECTURES does for `all`.
function(tensilelite_with_revision_aliases output_var)
    set(_result ${ARGN})
    foreach(_arch IN LISTS ARGN)
        # Features are colon-delimited and carry hyphens of their own
        # (gfx950:sramecc+:xnack-), so cut there before testing the name.
        string(REGEX REPLACE ":.*$" "" _base "${_arch}")
        foreach(_alias IN LISTS REVISION_ALIASES)
            if(_alias MATCHES "^${_base}v[0-9]+$" AND NOT "${_alias}" IN_LIST _result)
                list(APPEND _result "${_alias}")
            endif()
        endforeach()
    endforeach()
    set(${output_var} "${_result}" PARENT_SCOPE)
endfunction()

function(tensilelite_get_supported_architectures output_var)
    set(${output_var} ${SUPPORTED_ARCHITECTURES} PARENT_SCOPE)
endfunction()

function(tensilelite_sanitizer_requires_xnack output_var)
    if(HIPBLASLT_ENABLE_ASAN OR THEROCK_SANITIZER STREQUAL "ASAN" OR THEROCK_SANITIZER STREQUAL "HOST_ASAN")
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

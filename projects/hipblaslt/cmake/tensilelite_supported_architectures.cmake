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

# ISA features an architecture's code objects must be built with, spelled the
# way a target id spells them. This is the CMake-side twin of
# tensilelite/Common/Architectures.py::ARCH_TARGET_ID_FEATURES, and it must stay
# in step with it: both describe the same code objects.
#
# gfx950 is sramecc+ because the rocRoller kernels under
# tensilelite/CustomKernels/rocroller/ declare
# `amdgcn-amd-amdhsa--gfx950:sramecc+` and the assembler rejects any -mcpu that
# is not exactly that id. Carrying it on every gfx950 path keeps what hipBLASLt
# ships uniformly SRAMECC_ON. The consequence is deliberate: those code objects
# will not load on a gfx950 part with sramecc disabled. xnack is left
# unspecified (XNACK_ANY) outside sanitizer builds.
#
# A target id is never a path. Callers keep using the bare ${arch} for
# directories, file names and the --architecture= identifier TensileLite
# validates against its own allow-list.
set(TARGET_ID_FEATURES_gfx950 "sramecc+")

# The xnack normalization a sanitizer build needs, and nothing else. This is
# what GPU_TARGETS is normalized with: GPU_TARGETS is also forwarded verbatim to
# TensileLite as --architecture=, where it is checked against architectureMap,
# which knows `gfx950:xnack+` but not `gfx950:sramecc+`. ISA features therefore
# must not be added here -- they are added by tensilelite_offload_target, at the
# points that actually name a compiler target.
function(tensilelite_sanitizer_target output_var arch)
    set(_xnack_capable gfx908 gfx90a gfx942 gfx950) #gfx1250 support xnack_any so don't need to explicit ":xnack+"
    set(_target "${arch}")
    tensilelite_sanitizer_requires_xnack(_requires_xnack)
    if(_requires_xnack AND NOT "${arch}" MATCHES ":" AND "${arch}" IN_LIST _xnack_capable)
        set(_target "${arch}:xnack+")
    endif()
    set(${output_var} "${_target}" PARENT_SCOPE)
endfunction()

# The full target id for an architecture: what belongs in -mcpu,
# --offload-arch and a generated `.amdgcn_target` directive. Features are
# spelled sramecc before xnack, which is the only order clang accepts.
function(tensilelite_offload_target output_var arch)
    set(_base "${arch}")
    set(_tail "")
    if("${arch}" MATCHES "^([^:]+)(:.*)$")
        set(_base "${CMAKE_MATCH_1}")
        set(_tail "${CMAKE_MATCH_2}")
    endif()
    set(_target "${_base}")
    if(DEFINED TARGET_ID_FEATURES_${_base} AND NOT "${_tail}" MATCHES "sramecc")
        foreach(_feature IN LISTS TARGET_ID_FEATURES_${_base})
            set(_target "${_target}:${_feature}")
        endforeach()
    endif()
    set(_target "${_target}${_tail}")
    # Same rule as tensilelite_sanitizer_target, applied after the ISA features
    # so the id reads sramecc-then-xnack.
    set(_xnack_capable gfx908 gfx90a gfx942 gfx950)
    tensilelite_sanitizer_requires_xnack(_requires_xnack)
    if(_requires_xnack AND _tail STREQUAL "" AND "${_base}" IN_LIST _xnack_capable)
        set(_target "${_target}:xnack+")
    endif()
    set(${output_var} "${_target}" PARENT_SCOPE)
endfunction()

function(tensilelite_normalize_targets output_var)
    set(_normalized "")
    foreach(_arch IN LISTS ARGN)
        tensilelite_sanitizer_target(_target "${_arch}")
        list(APPEND _normalized "${_target}")
    endforeach()
    set(${output_var} "${_normalized}" PARENT_SCOPE)
endfunction()

# The list hip-config.cmake turns into `--offload-arch=` on hip::device, i.e.
# the target ids hipBLASLt's own HIP device code is compiled for. Separate from
# GPU_TARGETS so the feature tail never reaches the allow-list or a path.
function(tensilelite_offload_targets output_var)
    set(_targets "")
    foreach(_arch IN LISTS ARGN)
        tensilelite_offload_target(_target "${_arch}")
        list(APPEND _targets "${_target}")
    endforeach()
    set(${output_var} "${_targets}" PARENT_SCOPE)
endfunction()

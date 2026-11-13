# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# CK compile definitions for MIOpen targets that include composable_kernel headers.
# Keep in sync with CK's FP8 mode for the active GPU (see composable_kernel/CMakeLists.txt).

include_guard(GLOBAL)

# GPU targets that enable TF32 support in CK.
set(MIOPEN_CK_ENABLE_TF32_TARGETS gfx942 gfx950)
# GPU targets where CK is built with OCP FP8 (CK headers default to FNUZ).
set(MIOPEN_CK_USE_OCP_FP8_TARGETS gfx12 gfx950)

# Returns TRUE in out_var if arch_value matches any entry in the given target list.
function(_miopen_arch_matches arch_value target_list out_var)
    foreach(gpu_target IN LISTS ${target_list})
        if("${arch_value}" MATCHES "${gpu_target}")
            set(${out_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${out_var} FALSE PARENT_SCOPE)
endfunction()

# Apply CK compile definitions to a target so its CK macro set matches how CK
# itself was built (see composable_kernel/CMakeLists.txt).
#
# Usage:
#   miopen_apply_ck_compile_definitions(<target>)        # match global GPU_TARGETS
#   miopen_apply_ck_compile_definitions(<target> <arch>) # match a single arch
#
# The optional <arch> form is for the per-arch CK plugin shared libraries built
# by src/ck_impl, which each target one GPU arch and must carry the same macros
# as the main MIOpen target to stay ABI/semantically consistent (FP8 mode, etc.).
function(miopen_apply_ck_compile_definitions target)
    if(NOT MIOPEN_USE_COMPOSABLEKERNEL)
        return()
    endif()

    if(ARGC GREATER 1)
        set(_ck_arch "${ARGV1}")
    else()
        set(_ck_arch "${GPU_TARGETS}")
    endif()

    _miopen_arch_matches("${_ck_arch}" MIOPEN_CK_ENABLE_TF32_TARGETS _ck_enable_tf32)
    if(_ck_enable_tf32)
        target_compile_definitions(${target} PRIVATE CK_ENABLE_TF32)
    endif()

    if("${_ck_arch}" MATCHES "gfx950")
        target_compile_definitions(${target} PRIVATE CK_USE_GFX950)
    endif()

    _miopen_arch_matches("${_ck_arch}" MIOPEN_CK_USE_OCP_FP8_TARGETS _ck_use_ocp_fp8)
    if(_ck_use_ocp_fp8)
        target_compile_definitions(${target} PRIVATE CK_USE_OCP_FP8)
    endif()
endfunction()

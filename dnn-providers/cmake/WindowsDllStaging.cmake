# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# DLL-shadowing workaround (remove once ROCm fixes this on Windows): the driver installs stale
# amd_comgr.dll and amdhip64_<N>.dll copies in System32 that shadow the TheRock ones. The stale
# comgr breaks MIOpen's kernel JIT; the stale HIP runtime crashes TheRock's rocBLAS (access
# violation in MIOpen's GEMM conv solvers). PATH can't fix it (System32 precedes PATH), but the
# exe's own dir wins, so stage the DLLs there for any Windows build that runs ROCm libraries.
# Tests.cmake wires stage_shadowed_rocm_dlls to test targets; the GLOBAL guard defines it once.
if(WIN32)
    block(SCOPE_FOR VARIABLES)
        get_property(_dll_shadow_staged GLOBAL PROPERTY _rocm_dlls_staged_dll_shadow_workaround)
        if(NOT _dll_shadow_staged)
            # ROCM_CMAKE_PATH and ROCM_PATH are mutually-exclusive ways to point at the ROCm root
            # (see ClangToolChain.cmake); prefer the former, fall back to the latter.
            set(_rocm_root "${ROCM_CMAKE_PATH}")
            if(NOT _rocm_root)
                set(_rocm_root "${ROCM_PATH}")
            endif()
            if(_rocm_root)
                set_property(GLOBAL PROPERTY _rocm_dlls_staged_dll_shadow_workaround TRUE)
                file(TO_CMAKE_PATH "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}" _build_bin_dir)
                # The HIP runtime's name carries its major version (amdhip64_7.dll); glob it so a
                # HIP major bump doesn't leave a DEPENDS on a file that no longer exists.
                file(GLOB _hip_runtime_dlls RELATIVE "${_rocm_root}/bin"
                    "${_rocm_root}/bin/amdhip64_*.dll")
                set(_shadowed_dlls amd_comgr.dll ${_hip_runtime_dlls})
                set(_staged_dlls "")
                foreach(_dll_name IN LISTS _shadowed_dlls)
                    set(_dst "${_build_bin_dir}/${_dll_name}")
                    add_custom_command(
                        OUTPUT "${_dst}"
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                                "${_rocm_root}/bin/${_dll_name}" "${_dst}"
                        DEPENDS "${_rocm_root}/bin/${_dll_name}"
                        COMMENT "Staging ${_dll_name} into build bin (DLL-shadowing workaround)"
                        VERBATIM
                    )
                    list(APPEND _staged_dlls "${_dst}")
                endforeach()
                add_custom_target(stage_shadowed_rocm_dlls ALL DEPENDS ${_staged_dlls}
                    COMMENT "Staging shadowed ROCm DLLs into build bin")
            endif()
        endif()
    endblock()
endif()

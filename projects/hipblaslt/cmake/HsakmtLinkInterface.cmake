# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Repair for the link interface ROCm 7.x's hsakmtTargets.cmake exports.
#
# That file bakes BUILD-HOST absolute paths into hsakmt::hsakmt's
# INTERFACE_LINK_LIBRARIES, which do not exist on a fresh container, so every
# executable linking hsakmt fails. Two forms appear:
#   * an absolute /usr/lib64/libc.so entry -> make reports "No rule to make
#     target '/usr/lib64/libc.so'" (libc is linked implicitly anyway), and
#   * a `-L/home/runner/.../rocm_sysdeps/lib` search dir feeding `-ldrm` /
#     `-ldrm_amdgpu` -> ld.lld "unable to find library -ldrm".
# The libdrm/numa the interface wants are vendored in
# ${ROCM_PATH}/lib/rocm_sysdeps/lib on such a container.
#
# hipblaslt_sanitize_hsakmt_link_interface()
#   Drops phantom libc entries and repoints dead `-L` search dirs at the
#   vendored sysdeps lib dir under ROCM_PATH, rewriting hsakmt::hsakmt's
#   INTERFACE_LINK_LIBRARIES in place. Returns without touching anything when
#   the target does not exist or carries no link interface.
#
# Entries that are already correct are left untouched, so on a normally
# installed ROCm -- where the exported paths are live and there may be no
# vendored sysdeps dir at all -- this is a no-op.
function(hipblaslt_sanitize_hsakmt_link_interface)
    if(NOT TARGET hsakmt::hsakmt)
        return()
    endif()
    get_target_property(_ill hsakmt::hsakmt INTERFACE_LINK_LIBRARIES)
    if(NOT _ill)
        return()
    endif()

    if(NOT DEFINED ROCM_PATH)
        if(DEFINED ENV{ROCM_PATH})
            set(ROCM_PATH "$ENV{ROCM_PATH}")
        else()
            set(ROCM_PATH "/opt/rocm")
        endif()
    endif()
    set(sysdeps_lib "${ROCM_PATH}/lib/rocm_sysdeps/lib")

    # Rewriting a search dir to a directory that does not exist would trade one
    # unresolvable -L for another, so the repoint is only ever offered when
    # there is somewhere real to point at.
    if(IS_DIRECTORY "${sysdeps_lib}")
        set(_can_repoint TRUE)
    else()
        set(_can_repoint FALSE)
    endif()

    set(_clean "")
    foreach(_lib IN LISTS _ill)
        if(_lib MATCHES "/libc\\.so$" AND NOT EXISTS "${_lib}")
            message(STATUS "hsakmt: dropping nonexistent libc path '${_lib}' "
                           "from INTERFACE_LINK_LIBRARIES (libc is linked implicitly)")
            continue()
        endif()

        # The capture is read into a named variable inside the branch rather
        # than tested in the if() that produces it. if() expands its arguments
        # before evaluating them, so a ${CMAKE_MATCH_1} written into the same
        # condition as the MATCHES holds whatever the PREVIOUS match left
        # behind -- which makes the guard inspect the wrong directory and
        # rewrite live search dirs.
        if(_lib MATCHES "^-L(.+)$")
            set(_search_dir "${CMAKE_MATCH_1}")
            if(NOT IS_DIRECTORY "${_search_dir}")
                if(_can_repoint)
                    message(STATUS "hsakmt: repointing dead search dir '${_search_dir}' "
                                   "to vendored '${sysdeps_lib}'")
                    list(APPEND _clean "-L${sysdeps_lib}")
                    continue()
                endif()
                message(STATUS "hsakmt: leaving dead search dir '${_search_dir}' alone; "
                               "vendored '${sysdeps_lib}' does not exist either")
            endif()
        endif()

        list(APPEND _clean "${_lib}")
    endforeach()

    set_target_properties(hsakmt::hsakmt PROPERTIES
        INTERFACE_LINK_LIBRARIES "${_clean}")
endfunction()

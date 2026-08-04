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
# tensilelite_sanitize_hsakmt_link_interface(<out_var> <sysdeps_lib> <in_list_var>)
#   Drops phantom libc entries and repoints dead `-L` search dirs at
#   <sysdeps_lib>, writing the result to <out_var> in the caller's scope.
#   <in_list_var> is the NAME of the list variable to read, not its value.
#
# Entries that are already correct are returned untouched, so on a normally
# installed ROCm -- where the exported paths are live and there may be no
# vendored sysdeps dir at all -- this is a no-op.
#
# Lives in its own module, separate from client/CMakeLists.txt, so that
# cmake/tests/test_hsakmt_link_interface.cmake can exercise it directly.
function(tensilelite_sanitize_hsakmt_link_interface out_var sysdeps_lib in_list_var)
    # Rewriting a search dir to a directory that does not exist would trade one
    # unresolvable -L for another, so the repoint is only ever offered when
    # there is somewhere real to point at.
    if(IS_DIRECTORY "${sysdeps_lib}")
        set(_can_repoint TRUE)
    else()
        set(_can_repoint FALSE)
    endif()

    set(_clean "")
    foreach(_lib IN LISTS ${in_list_var})
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

    set(${out_var} "${_clean}" PARENT_SCOPE)
endfunction()

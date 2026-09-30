# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# hipccl_install_legacy_header_symlink(<subdir>)
#
# Installs a single relative directory symlink at
# ${CMAKE_INSTALL_PREFIX}/include/<subdir> pointing at
# ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/<subdir> (i.e.
# .../include/hipccl/<subdir>), so pre-hipccl code doing e.g.
# `#include <rocprim/...>` still resolves against a plain `-I<prefix>/include`
# without needing `-I<prefix>/include/hipccl` too. Intended as a *temporary*
# compatibility aid for the include-directory migration, controlled by
# HIPCCL_INSTALL_LEGACY_HEADER_SYMLINKS in the root CMakeLists.txt - not
# something to leave on indefinitely, since it reintroduces exactly the
# flat-namespace collision risk (two differently-packaged copies of
# "rocprim/" both claiming the same path) that moving headers under
# include/hipccl/ was meant to avoid.
#
# Deliberately NOT reusing rocm-cmake's ROCMInstallSymlinks.cmake
# (rocm_install_symlink_subdir()): that function walks every file under a
# per-component sub-prefix that already mirrors a full include/lib/bin
# layout, and re-roots each one directly at CMAKE_INSTALL_PREFIX - it has no
# way to reinsert the "include/" path segment this case needs, since only
# headers moved here, not each component's whole install tree. A single
# directory-level symlink is both correct for this narrower case and cheaper
# (one filesystem entry instead of one per header file).
function(hipccl_install_legacy_header_symlink SUBDIR)
    if(CMAKE_HOST_WIN32)
        # Windows doesn't reliably support symlinks without elevated
        # privileges/developer mode, so fall back to a recursive copy - still
        # gives old-style `#include <rocprim/...>` resolution, just without
        # the "it's the same file on disk" property a symlink would have.
        set(HIPCCL_SYMLINK_CMD "file(COPY \${SRC} DESTINATION \${DEST_PARENT})")
    else()
        # -f: replace an existing file/symlink at DEST.
        # -n: treat DEST as a normal path even if it is itself a symlink to a
        #     directory, so re-running install doesn't nest the new link
        #     inside the previous one.
        set(HIPCCL_SYMLINK_CMD "execute_process(COMMAND ln -sfn \${SRC_REL} \${DEST})")
    endif()

    # NOTE: ${CMAKE_INSTALL_INCLUDEDIR} and ${SUBDIR} are intentionally
    # expanded now, at configure time (matching how CMake itself resolves a
    # relative install(DESTINATION) argument) - only $ENV{DESTDIR} and
    # ${CMAKE_INSTALL_PREFIX} are escaped, so they're re-evaluated at install
    # time instead, honoring `cmake --install --prefix <path>` and staged
    # (DESTDIR-based) installs the same way CMake's own install() rules do.
    set(HIPCCL_INSTALL_CMD "
        set(SRC \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/${SUBDIR})
        get_filename_component(DEST_PARENT \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/.. ABSOLUTE)
        set(DEST \${DEST_PARENT}/${SUBDIR})
        file(MAKE_DIRECTORY \${DEST_PARENT})
        file(RELATIVE_PATH SRC_REL \${DEST_PARENT} \${SRC})
        if(EXISTS \${DEST} AND NOT IS_SYMLINK \${DEST})
            message(WARNING
                \"legacy header symlink: \${DEST} already exists and is not \"
                \"a symlink (looks like a real, separately-installed package) \"
                \"- leaving it alone and skipping the compatibility symlink \"
                \"for ${SUBDIR}.\")
        else()
            if(IS_SYMLINK \${DEST})
                file(REMOVE \${DEST})
            endif()
            message(STATUS \"legacy header symlink: \${DEST} -> \${SRC_REL}\")
            ${HIPCCL_SYMLINK_CMD}
        endif()
    ")
    install(CODE "${HIPCCL_INSTALL_CMD}")
endfunction()

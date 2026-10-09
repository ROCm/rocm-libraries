# MIT License
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################
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
    # NOTE: ${CMAKE_INSTALL_INCLUDEDIR} and ${SUBDIR} are intentionally
    # expanded now, at configure time (matching how CMake itself resolves a
    # relative install(DESTINATION) argument) - only $ENV{DESTDIR} and
    # ${CMAKE_INSTALL_PREFIX} are escaped, so they're re-evaluated at install
    # time instead, honoring `cmake --install --prefix <path>` and staged
    # (DESTDIR-based) installs the same way CMake's own install() rules do.
    #
    # file(CREATE_LINK) is used rather than `execute_process(COMMAND ln ...)`
    # so that failures are actually detectable: RESULT is "0" on success or an
    # error string otherwise, whereas the bare `ln` call reported nothing and a
    # failed link (read-only prefix, insufficient privileges) produced a
    # successful-looking install with no symlink. It also removes the need to
    # branch on the host OS - Windows without Developer Mode simply fails the
    # link and takes the copy fallback below, while a privileged Windows build
    # now gets a real symlink instead of an unconditional copy.
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
            file(CREATE_LINK \${SRC_REL} \${DEST} RESULT _hipccl_link_result SYMBOLIC)
            if(NOT _hipccl_link_result STREQUAL \"0\")
                message(WARNING
                    \"legacy header symlink: could not create \${DEST} \"
                    \"(\${_hipccl_link_result}) - copying the headers there \"
                    \"instead. Unlike a symlink, the copy will not pick up \"
                    \"later changes to \${SRC}.\")
                file(COPY \${SRC} DESTINATION \${DEST_PARENT})
            endif()
        endif()
    ")
    install(CODE "${HIPCCL_INSTALL_CMD}")
endfunction()

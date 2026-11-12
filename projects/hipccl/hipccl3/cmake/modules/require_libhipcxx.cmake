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

# Fails the configure with an actionable message unless a libhipcxx
# installation satisfying MINIMUM_VERSION is available, either already
# configured as part of the same build (e.g. HIPCCL_BUILD_LIBHIPCXX=ON in the
# hipccl3 superbuild) or discoverable via find_package.
#
# Usage:
#   hipccl_require_libhipcxx(<minimum_version>)
function(hipccl_require_libhipcxx MINIMUM_VERSION)
    # In hipCCL super builds, this target may already be available. In such
    # cases find_package(...) will not work. Instead, pretend we've already
    # found libhipcxx.
    if(TARGET libhipcxx::libhipcxx)
        message(STATUS "Reusing already-configured libhipcxx::libhipcxx target (required: >= ${MINIMUM_VERSION})")
        # Consumers gate on libhipcxx_FOUND (see hipcub/hipcub/CMakeLists.txt),
        # so set it to the parent scope.
        set(libhipcxx_FOUND TRUE PARENT_SCOPE)
        return()
    endif()

    find_package(libhipcxx ${MINIMUM_VERSION} CONFIG QUIET PATHS "${ROCM_ROOT}/lib/cmake/libhipcxx")

    if(NOT TARGET libhipcxx::libhipcxx)
        message(FATAL_ERROR
            "${PROJECT_NAME} requires libhipcxx >= ${MINIMUM_VERSION}, but no "
            "suitable libhipcxx installation was found.\n"
            "  - Building through hipccl3's root CMakeLists.txt: configure with "
            "-DHIPCCL_BUILD_LIBHIPCXX=ON (after 'git submodule update --init "
            "projects/hipccl/hipccl3/libhipcxx').\n"
            "  - Building standalone: install libhipcxx >= ${MINIMUM_VERSION} "
            "(https://github.com/ROCm/libhipcxx) so find_package(libhipcxx) can "
            "locate it (CMAKE_PREFIX_PATH / ROCM_ROOT), then re-configure."
        )
    else()
        message(STATUS "Found libhipcxx ${libhipcxx_VERSION} (required: >= ${MINIMUM_VERSION})")
        # find_package() inside a function only sets libhipcxx_FOUND in the
        # function's local scope.
        # 
        # This is similar to the earlier 'TARGET libhipcxx::libhipcxx' check.
        set(libhipcxx_FOUND TRUE PARENT_SCOPE)
    endif()
endfunction()

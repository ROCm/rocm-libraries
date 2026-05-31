# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

function(hipblaslt_target_configure_sanitizers hipblaslt_target linkage)
    # When the TheRock superbuild is driving the sanitizer (THEROCK_SANITIZER
    # set in the cache), -fsanitize=address / -shared-libsan / -fuse-ld=lld are
    # already injected globally via cmake/therock_sanitizers.cmake's CMAKE_*_FLAGS_INIT.
    # Applying them again per-target produces duplicate compile/link entries
    # (the same flag appears twice in the link line) and a double -fuse-ld=lld
    # pass, which has historically masked the asymmetric instrumentation bug
    # where HIPBLASLT_ENABLE_ASAN=ON was passed alongside THEROCK_SANITIZER=ASAN.
    # Early-return so callers can keep their HIPBLASLT_ENABLE_* gates without
    # branching on which mechanism enabled the sanitizer.
    if(DEFINED THEROCK_SANITIZER AND NOT THEROCK_SANITIZER STREQUAL "")
        return()
    endif()
    if(HIPBLASLT_ENABLE_ASAN)
        # Add asan flags to hipblas_target
        target_compile_options(${hipblaslt_target}
            ${linkage}
                -fsanitize=address
                -shared-libasan
        )
        target_link_options(${hipblaslt_target}
            ${linkage}
                -fsanitize=address
                -shared-libasan
                -fuse-ld=lld
        )
    elseif(HIPBLASLT_ENABLE_TSAN)
        # Add tsan flags to hipblas_target
        target_compile_options(${hipblaslt_target}
            ${linkage}
                -fsanitize=thread
                -shared-libtsan
        )
        target_link_options(${hipblaslt_target}
            ${linkage}
                -fsanitize=thread
                -shared-libtsan
                -fuse-ld=lld
        )
    endif()
endfunction()

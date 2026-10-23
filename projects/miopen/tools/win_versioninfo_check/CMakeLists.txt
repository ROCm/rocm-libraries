################################################################################
#
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

# Host-side verifier for the Windows VERSIONINFO resource embedded by
# miopen_embed_version_resource() (ROCM-20937). Lives under tools/ rather than test/ for two
# reasons: it must be configured before add_subdirectory(src) so the POST_BUILD rules in that
# function can reference it, and it must stay available when BUILD_TESTING is OFF -- a release
# build is exactly where shipping DLLs without metadata matters most.
#
# version.lib is the only dependency. The tool reads the PE resource directory off disk and never
# executes the DLL, so it needs no HIP runtime, no driver, and no GPU, and runs fine on the
# CPU-only Windows build machines.
add_executable(win_versioninfo_check main.cpp)
target_link_libraries(win_versioninfo_check PRIVATE version.lib)
clang_tidy_check(win_versioninfo_check)

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

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

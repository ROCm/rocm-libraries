# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# tablegen is added separately in root CMakeLists.txt (sets TABLEGEN_OUTPUT_DIR
# needed before gfxisa). All other tools are added here.
add_subdirectory(stinkytofu-opt)
add_subdirectory(stinkytofu-check)
add_subdirectory(intrinsic-compiler)

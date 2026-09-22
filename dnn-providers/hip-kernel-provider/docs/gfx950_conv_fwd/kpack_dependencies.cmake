# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# The pinned kpack export refers to these imported dependency targets.
# Load them in the top project before the provider imports rocm-kpack.
include_guard(GLOBAL)
find_package(zstd CONFIG REQUIRED)
find_package(msgpack-cxx CONFIG REQUIRED)

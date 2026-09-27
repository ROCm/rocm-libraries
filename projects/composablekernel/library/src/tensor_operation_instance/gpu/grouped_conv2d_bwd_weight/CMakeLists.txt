# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# grouped_conv2d_bwd_weight - layout-specific instances

# GNHWC - Optional (not required by MIOpen)
if(NOT MIOPEN_REQ_LIBS_ONLY)
    add_subdirectory(gnhwc)
endif()
# NGCHW is needed by MIOpen only for the gfx1250 BF16 pointwise WMMA instance.
if(NOT MIOPEN_REQ_LIBS_ONLY OR
   (SUPPORTED_GPU_TARGETS MATCHES "gfx1250" AND CK_USE_WMMA AND CK_ENABLE_BF16))
    add_subdirectory(ngchw)
endif()
# NHWGC - Required by MIOpen
add_subdirectory(nhwgc)

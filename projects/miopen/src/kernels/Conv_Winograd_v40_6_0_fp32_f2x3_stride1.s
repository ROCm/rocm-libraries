// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

.include "Conv_Winograd_v40_6_0_metadata.inc"

KERNEL_PROLOG fp32_f2x3_stride1

.if (.amdgcn.gfx_generation_number == 12)
    .include "Conv_Winograd_v40_6_0_gfx12_fp32_f2x3_stride1.inc"
.endif

KERNEL_EPILOG fp32_f2x3_stride1

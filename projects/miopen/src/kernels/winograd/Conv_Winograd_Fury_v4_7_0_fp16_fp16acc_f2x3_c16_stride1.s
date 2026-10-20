// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

.include "Conv_Winograd_Fury_v2_X_X_X_metadata.inc"

.if (.amdgcn.gfx_generation_number == 11 && .amdgcn.gfx_generation_minor == 7)
    // gfx117X
    KERNEL_PROLOG 4_7_0, _1024vgprs_fp16_fp16acc_f2x3_c16_stride1
    .noaltmacro
    .include "Conv_Winograd_Fury_v4_7_0_gfx117_1024vgprs_fp16_fp16acc_f2x3_c16_stride1.inc"
    .altmacro
    KERNEL_EPILOG 4_7_0, _1024vgprs_fp16_fp16acc_f2x3_c16_stride1
.else
    .error "Unsupported gfx generation"
.endif

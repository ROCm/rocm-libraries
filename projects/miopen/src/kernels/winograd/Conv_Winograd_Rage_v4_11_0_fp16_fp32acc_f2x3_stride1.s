// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

.include "Conv_Winograd_Rage_v4_X_X_metadata.inc"

.if (.amdgcn.gfx_generation_number == 11 && .amdgcn.gfx_generation_minor == 7)
    KERNEL_PROLOG 4_11_0, _fp16_fp32acc_f2x3_stride1
    .noaltmacro
    .include "Conv_Winograd_Rage_v4_11_0_gfx117x_fp16_fp32acc_f2x3_stride1.inc"
    .altmacro
    KERNEL_EPILOG 102, 254, 0, 256, 4_11_0, _fp16_fp32acc_f2x3_stride1
.else
    .error "Unsupported gfx version"
.endif

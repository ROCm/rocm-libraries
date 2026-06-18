// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

.include "Conv_Winograd_Rage_v4_X_X_metadata.inc"

.if (.amdgcn.gfx_generation_number == 12 && .amdgcn.gfx_generation_minor == 0)
    KERNEL_PROLOG 4_9_1, _fp16_fp32acc_f2x3_stride1
    .noaltmacro
    .include "Conv_Winograd_Rage_v4_9_1_gfx12x_fp16_fp32acc_f2x3_stride1.inc"
    .altmacro
    KERNEL_EPILOG 88, 250, 0, 384, 4_9_1, _fp16_fp32acc_f2x3_stride1
.else
    .error "Unsupported gfx version"
.endif

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

.include "Conv_Winograd_Rage_v4_X_X_metadata.inc"

.if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor == 4 && .amdgcn.gfx_generation_stepping == 2)
    KERNEL_PROLOG 4_6_1, _fp16_fp32acc_f2x3_stride1
    .include "Conv_Winograd_Rage_v4_6_1_gfx94x_fp16_fp32acc_f2x3_stride1.inc"
    KERNEL_EPILOG 80, 162, 164, 768, 4_6_1, _fp16_fp32acc_f2x3_stride1
.elseif (.amdgcn.gfx_generation_number == 12)
    KERNEL_PROLOG 4_6_1, _fp16_fp32acc_f2x3_stride1
    .noaltmacro
    .include "Conv_Winograd_Rage_v4_6_1_gfx12x_fp16_fp32acc_f2x3_stride1.inc"
    .altmacro
    KERNEL_EPILOG 80, 250, 0, 384, 4_6_1, _fp16_fp32acc_f2x3_stride1
.else
    .error "Unsupported gfx version"
.endif

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

.include "Conv_Winograd_Rage_v4_X_X_metadata.inc"

.if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor == 4 && .amdgcn.gfx_generation_stepping == 2)
    KERNEL_PROLOG 4_9_0, _fp32_fp32acc_f2x3_stride1
    .include "Conv_Winograd_Rage_v4_9_0_gfx94x_fp32_fp32acc_f2x3_stride1.inc"
    KERNEL_EPILOG 88, 158, 4, 768, 4_9_0, _fp32_fp32acc_f2x3_stride1
.else
    .error "Unsupported gfx version"
.endif

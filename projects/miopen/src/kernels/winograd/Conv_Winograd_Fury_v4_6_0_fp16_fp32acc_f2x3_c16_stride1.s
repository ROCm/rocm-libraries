/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
.include "Conv_Winograd_Fury_v2_X_X_X_metadata.inc"

.if (.amdgcn.gfx_generation_number == 12)
    KERNEL_PROLOG 4_6_0, _1536vgprs_fp16_fp32acc_f2x3_c16_stride1
    .noaltmacro
    .include "Conv_Winograd_Fury_v4_6_0_gfx12_1536vgprs_fp16_fp32acc_f2x3_c16_stride1.inc"
    .altmacro
    KERNEL_EPILOG 4_6_0, _1536vgprs_fp16_fp32acc_f2x3_c16_stride1
.else
    .error "Unsupported gfx generation"
.endif

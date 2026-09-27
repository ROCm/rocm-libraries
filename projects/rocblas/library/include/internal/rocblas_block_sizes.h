/* ************************************************************************
* Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
* ies of the Software, and to permit persons to whom the Software is furnished
* to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
* PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
* FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
* IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
* CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
* ************************************************************************ */

/*!\file
* \brief rocblas_block_sizes.h includes the definition of various block sizes used in rocBLAS instantiations.
*        This file is for internal use only.
*/

#ifndef ROCBLAS_BLOCK_SIZES_H
#define ROCBLAS_BLOCK_SIZES_H

// L1 NB
#define ROCBLAS_ASUM_NB 512
#define ROCBLAS_AXPY_NB 256
#define ROCBLAS_COPY_NB 256
#define ROCBLAS_DOT_NB 512
#define ROCBLAS_IAMAX_NB 1024
#define ROCBLAS_NRM2_NB 512
#define ROCBLAS_ROT_NB 512
#define ROCBLAS_ROTM_NB 512
#define ROCBLAS_SCAL_NB 256
#define ROCBLAS_SWAP_NB 256

// L2 NB
#define ROCBLAS_TPMV_NB 512
#define ROCBLAS_TBSV_NB 512
#define ROCBLAS_TRMV_NB 512
#define ROCBLAS_TPSV_NB 512
#define ROCBLAS_SDCTRSV_NB 64
#define ROCBLAS_ZTRSV_NB 32

// L3 NB
#define ROCBLAS_HERK_BATCHED_NB 8
#define ROCBLAS_CHERK_NB 64
#define ROCBLAS_ZHERK_NB 128
#define ROCBLAS_HERKX_BATCHED_NB 8
#define ROCBLAS_HERKX_NB 32

#define ROCBLAS_SDSYRK_BATCHED_NB 16
#define ROCBLAS_CZSYRK_BATCHED_NB 8
#define ROCBLAS_CSYRK_NB 64
#define ROCBLAS_SDZSYRK_NB 32

#define ROCBLAS_SDSYRKX_BATCHED_NB 16
#define ROCBLAS_CZSYRKX_BATCHED_NB 8
#define ROCBLAS_SSYRKX_NB 16
#define ROCBLAS_DCZSYRKX_NB 32

#define ROCBLAS_SDSYR2K_BATCHED_NB 16
#define ROCBLAS_CZSYR2K_BATCHED_NB 8
#define ROCBLAS_HER2K_BATCHED_NB 8
#define ROCBLAS_SSYR2K_NB 32
#define ROCBLAS_DSYR2K_NB 16
#define ROCBLAS_CSYR2K_NB 32
#define ROCBLAS_ZSYR2K_NB 16
#define ROCBLAS_CHER2K_NB 32
#define ROCBLAS_ZHER2K_NB 16

#define ROCBLAS_SDTRMM_NB 32
#define ROCBLAS_CZTRMM_NB 16
#define ROCBLAS_TRMM_OUTOFPLACE_NB 512

#define ROCBLAS_TRSM_NB 128
#define ROCBLAS_TRTRI_NB 16
#define ROCBLAS_TRSV_EX_NB 128

#endif

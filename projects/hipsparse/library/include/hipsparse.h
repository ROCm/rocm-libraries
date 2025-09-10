/* ************************************************************************
* Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*
* ************************************************************************ */

/*! \file
*  \brief hipsparse.h provides Sparse Linear Algebra Subprograms
*  of Level 1, 2 and 3, using HIP optimized for AMD GPU hardware.
*/

// HIP = Heterogeneous-compute Interface for Portability
//
// Define a extremely thin runtime layer that allows source code to be compiled
// unmodified through either AMD HCC or NVCC. Key features tend to be in the spirit
// and terminology of CUDA, but with a portable path to other accelerators as well.
//
// This is the master include file for hipSPARSE, wrapping around rocSPARSE and
// cuSPARSE "version 2".

#ifndef HIPSPARSE_H
#define HIPSPARSE_H

#include "hipsparse-export.h"
#include "hipsparse-version.h"

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

/// \cond DO_NOT_DOCUMENT
#define DEPRECATED_CUDA_12000(warning)
#define DEPRECATED_CUDA_11000(warning)
#define DEPRECATED_CUDA_10000(warning)
#define DEPRECATED_CUDA_9000(warning)

#ifdef __cplusplus
#ifndef __has_cpp_attribute
#define __has_cpp_attribute(X) 0
#endif
#define HIPSPARSE_HAS_DEPRECATED_MSG __has_cpp_attribute(deprecated) >= 201309L
#else
#ifndef __has_c_attribute
#define __has_c_attribute(X) 0
#endif
#define HIPSPARSE_HAS_DEPRECATED_MSG __has_c_attribute(deprecated) >= 201904L
#endif

#if HIPSPARSE_HAS_DEPRECATED_MSG
#define HIPSPARSE_DEPRECATED_MSG(MSG) [[deprecated(MSG)]]
#else
#define HIPSPARSE_DEPRECATED_MSG(MSG) HIPSPARSE_DEPRECATED // defined in hipsparse-export.h
#endif
/// \endcond

#if defined(CUDART_VERSION)
#if CUDART_VERSION < 10000
#undef DEPRECATED_CUDA_9000
#define DEPRECATED_CUDA_9000(warning) HIPSPARSE_DEPRECATED_MSG(warning)
#elif CUDART_VERSION < 11000
#undef DEPRECATED_CUDA_10000
#define DEPRECATED_CUDA_10000(warning) HIPSPARSE_DEPRECATED_MSG(warning)
#elif CUDART_VERSION < 12000
#undef DEPRECATED_CUDA_11000
#define DEPRECATED_CUDA_11000(warning) HIPSPARSE_DEPRECATED_MSG(warning)
#elif CUDART_VERSION < 13000
#undef DEPRECATED_CUDA_12000
#define DEPRECATED_CUDA_12000(warning) HIPSPARSE_DEPRECATED_MSG(warning)
#endif
#endif

#include "hipsparse-bfloat16.h"
#include "hipsparse-types.h"

#include "hipsparse-auxiliary.h"

/*
* ===========================================================================
*    level 1 SPARSE
* ===========================================================================
*/

#include "internal/level1/hipsparse_axpyi.h"
#include "internal/level1/hipsparse_dotci.h"
#include "internal/level1/hipsparse_doti.h"
#include "internal/level1/hipsparse_gthr.h"
#include "internal/level1/hipsparse_gthrz.h"
#include "internal/level1/hipsparse_roti.h"
#include "internal/level1/hipsparse_sctr.h"

/*
* ===========================================================================
*    level 2 SPARSE
* ===========================================================================
*/

#include "internal/level2/hipsparse_bsrmv.h"
#include "internal/level2/hipsparse_bsrsv.h"
#include "internal/level2/hipsparse_bsrxmv.h"
#include "internal/level2/hipsparse_csrmv.h"
#include "internal/level2/hipsparse_csrsv.h"
#include "internal/level2/hipsparse_gemvi.h"
#include "internal/level2/hipsparse_hybmv.h"

/*
* ===========================================================================
*    level 3 SPARSE
* ===========================================================================
*/

#include "internal/level3/hipsparse_bsrmm.h"
#include "internal/level3/hipsparse_bsrsm.h"
#include "internal/level3/hipsparse_csrmm.h"
#include "internal/level3/hipsparse_csrsm.h"
#include "internal/level3/hipsparse_gemmi.h"

/*
* ===========================================================================
*    extra SPARSE
* ===========================================================================
*/

#include "internal/extra/hipsparse_csrgeam.h"
#include "internal/extra/hipsparse_csrgemm.h"

/*
* ===========================================================================
*    preconditioner SPARSE
* ===========================================================================
*/

#include "internal/precond/hipsparse_bsric0.h"
#include "internal/precond/hipsparse_bsrilu0.h"
#include "internal/precond/hipsparse_csric0.h"
#include "internal/precond/hipsparse_csrilu0.h"
#include "internal/precond/hipsparse_gpsv_interleaved_batch.h"
#include "internal/precond/hipsparse_gtsv.h"
#include "internal/precond/hipsparse_gtsv_interleaved_batch.h"
#include "internal/precond/hipsparse_gtsv_nopivot.h"
#include "internal/precond/hipsparse_gtsv_strided_batch.h"

/*
* ===========================================================================
*    format conversions SPARSE
* ===========================================================================
*/

#include "internal/conversion/hipsparse_bsr2csr.h"
#include "internal/conversion/hipsparse_coo2csr.h"
#include "internal/conversion/hipsparse_coosort.h"
#include "internal/conversion/hipsparse_create_identity_permutation.h"
#include "internal/conversion/hipsparse_csc2dense.h"
#include "internal/conversion/hipsparse_cscsort.h"
#include "internal/conversion/hipsparse_csr2bsr.h"
#include "internal/conversion/hipsparse_csr2coo.h"
#include "internal/conversion/hipsparse_csr2csc.h"
#include "internal/conversion/hipsparse_csr2csr_compress.h"
#include "internal/conversion/hipsparse_csr2csru.h"
#include "internal/conversion/hipsparse_csr2dense.h"
#include "internal/conversion/hipsparse_csr2gebsr.h"
#include "internal/conversion/hipsparse_csr2hyb.h"
#include "internal/conversion/hipsparse_csrsort.h"
#include "internal/conversion/hipsparse_csru2csr.h"
#include "internal/conversion/hipsparse_dense2csc.h"
#include "internal/conversion/hipsparse_dense2csr.h"
#include "internal/conversion/hipsparse_gebsr2csr.h"
#include "internal/conversion/hipsparse_gebsr2gebsc.h"
#include "internal/conversion/hipsparse_gebsr2gebsr.h"
#include "internal/conversion/hipsparse_hyb2csr.h"
#include "internal/conversion/hipsparse_nnz.h"
#include "internal/conversion/hipsparse_nnz_compress.h"
#include "internal/conversion/hipsparse_prune_csr2csr.h"
#include "internal/conversion/hipsparse_prune_csr2csr_by_percentage.h"
#include "internal/conversion/hipsparse_prune_dense2csr.h"
#include "internal/conversion/hipsparse_prune_dense2csr_by_percentage.h"

/*
* ===========================================================================
*    reordering SPARSE
* ===========================================================================
*/

#include "internal/reorder/hipsparse_csrcolor.h"

/*
* ===========================================================================
*    generic SPARSE
* ===========================================================================
*/

#include "hipsparse-generic-types.h"

#include "hipsparse-generic-auxiliary.h"

#include "internal/generic/hipsparse_axpby.h"
#include "internal/generic/hipsparse_dense2sparse.h"
#include "internal/generic/hipsparse_gather.h"
#include "internal/generic/hipsparse_rot.h"
#include "internal/generic/hipsparse_scatter.h"
#include "internal/generic/hipsparse_sddmm.h"
#include "internal/generic/hipsparse_sparse2dense.h"
#include "internal/generic/hipsparse_spgemm.h"
#include "internal/generic/hipsparse_spgemm_reuse.h"
#include "internal/generic/hipsparse_spmm.h"
#include "internal/generic/hipsparse_spmv.h"
#include "internal/generic/hipsparse_spsm.h"
#include "internal/generic/hipsparse_spsv.h"
#include "internal/generic/hipsparse_spvv.h"

#endif // HIPSPARSE_H

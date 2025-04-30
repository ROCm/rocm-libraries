/*! \file */
/* ************************************************************************
* Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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
*  \brief rocsparse-conversion.h provides Sparse Linear Algebra Subprograms
*  of conversion
*/
#ifndef ROCSPARSE_CONVERSION_H
#define ROCSPARSE_CONVERSION_H

#include "conversion/rocsparse_bsr2csr.h"
#include "conversion/rocsparse_bsrpad_value.h"
#include "conversion/rocsparse_coo2csr.h"
#include "conversion/rocsparse_coo2dense.h"
#include "conversion/rocsparse_coosort.h"
#include "conversion/rocsparse_csc2dense.h"
#include "conversion/rocsparse_cscsort.h"
#include "conversion/rocsparse_csr2bsr.h"
#include "conversion/rocsparse_csr2coo.h"
#include "conversion/rocsparse_csr2csc.h"
#include "conversion/rocsparse_csr2csr_compress.h"
#include "conversion/rocsparse_csr2dense.h"
#include "conversion/rocsparse_csr2ell.h"
#include "conversion/rocsparse_csr2gebsr.h"
#include "conversion/rocsparse_csr2hyb.h"
#include "conversion/rocsparse_csrsort.h"
#include "conversion/rocsparse_dense2coo.h"
#include "conversion/rocsparse_dense2csc.h"
#include "conversion/rocsparse_dense2csr.h"
#include "conversion/rocsparse_ell2csr.h"
#include "conversion/rocsparse_gebsr2csr.h"
#include "conversion/rocsparse_gebsr2gebsc.h"
#include "conversion/rocsparse_gebsr2gebsr.h"
#include "conversion/rocsparse_hyb2csr.h"
#include "conversion/rocsparse_inverse_permutation.h"
#include "conversion/rocsparse_nnz.h"
#include "conversion/rocsparse_nnz_compress.h"
#include "conversion/rocsparse_prune_csr2csr.h"
#include "conversion/rocsparse_prune_csr2csr_by_percentage.h"
#include "conversion/rocsparse_prune_dense2csr.h"
#include "conversion/rocsparse_prune_dense2csr_by_percentage.h"

#endif // ROCSPARSE_CONVERSION_H

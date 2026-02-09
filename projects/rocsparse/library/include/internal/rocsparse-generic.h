/*! \file */
/* ************************************************************************
* Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
*  \brief rocsparse-generic.h provides Sparse Linear Algebra Subprograms
*  of Generic
*/
#ifndef ROCSPARSE_GENERIC_H
#define ROCSPARSE_GENERIC_H

#include "generic/rocsparse_axpby.h"
#include "generic/rocsparse_check_spmat.h"
#include "generic/rocsparse_dense_to_sparse.h"
#include "generic/rocsparse_extract.h"
#include "generic/rocsparse_gather.h"
#include "generic/rocsparse_rot.h"
#include "generic/rocsparse_scatter.h"
#include "generic/rocsparse_sddmm.h"
#include "generic/rocsparse_sparse_to_dense.h"
#include "generic/rocsparse_sparse_to_sparse.h"
#include "generic/rocsparse_spgeam.h"
#include "generic/rocsparse_spgemm.h"
#include "generic/rocsparse_spitsv.h"
#include "generic/rocsparse_spmm.h"
#include "generic/rocsparse_spmv.h"
#include "generic/rocsparse_spsm.h"
#include "generic/rocsparse_spsv.h"
#include "generic/rocsparse_sptrsm.h"
#include "generic/rocsparse_sptrsv.h"
#include "generic/rocsparse_spvv.h"
#include "generic/rocsparse_v2_spmv.h"

#endif // ROCSPARSE_GENERIC_H

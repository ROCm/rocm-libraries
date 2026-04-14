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

#include "rocsparse_utility.hpp"

#include "rocsparse_gcoo2coo_aos.hpp"
#include "rocsparse_gcoo_aos2coo.hpp"
#include "rocsparse_gcoo_aos2csr.hpp"
#include "rocsparse_gcsr2coo_aos.hpp"
//
#include "rocsparse_gbsr2csr.hpp"
#include "rocsparse_gcsr2bsr.hpp"
//
#include "rocsparse_gcsr2ell.hpp"
#include "rocsparse_gell2csr.hpp"
//
#include "rocsparse_gcoo2csr.hpp"
#include "rocsparse_gcsr2coo.hpp"
//
#include "rocsparse_gcsc2csr.hpp"
#include "rocsparse_gcsr2csc.hpp"
#include "rocsparse_ggthr.hpp"
//

#include "rocsparse_convert_array.hpp"
#include "rocsparse_dense2coo.hpp"
#include "rocsparse_dense2csx_impl.hpp"
#include "rocsparse_identity.hpp"
#include "rocsparse_nnz_impl.hpp"
#include "rocsparse_sparse_to_coo_to_sparse.hpp"
#include "rocsparse_sparse_to_csr_to_sparse.hpp"
#include "rocsparse_sparse_to_sparse.hpp"
#include "rocsparse_spmat_transfer_from.hpp"

//
// Define header of all the convert function.
//

typedef rocsparse_status (*convert_type)(rocsparse_handle                         handle,
                                         rocsparse_sparse_to_sparse_descr         descr_,
                                         rocsparse_const_spmat_descr              source_,
                                         rocsparse_spmat_descr                    target_,
                                         _rocsparse_sparse_to_sparse_descr::stage stage_,
                                         size_t*                                  buffer_size_,
                                         void*                                    buffer_);

#define FUNCTION_CONVERT(S, T)                                                                     \
    static rocsparse_status convert_##S##_to_##T(rocsparse_handle                         handle,  \
                                                 rocsparse_sparse_to_sparse_descr         descr_,  \
                                                 rocsparse_const_spmat_descr              source_, \
                                                 rocsparse_spmat_descr                    target_, \
                                                 _rocsparse_sparse_to_sparse_descr::stage stage_,  \
                                                 size_t* buffer_size_,                             \
                                                 void*   buffer_)

//
// coo2coo_aos
//
FUNCTION_CONVERT(coo, coo_aos)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(stage_)
    {
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_analysis:
    {
        target_->nnz = source_->nnz;
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_coo2coo_aos_buffer_size(handle, source_, target_, buffer_size_));
        return rocsparse_status_success;
    }

    case _rocsparse_sparse_to_sparse_descr::stage_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_coo2coo_aos(handle, source_, target_, buffer_size_[0], buffer_));
        return rocsparse_status_success;
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

//
// coo2csr
//
FUNCTION_CONVERT(coo, csr)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(stage_)
    {
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_analysis:
    {
        target_->nnz = source_->nnz;
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_coo2csr_buffer_size(handle, source_, target_, buffer_size_));
        return rocsparse_status_success;
    }

    case _rocsparse_sparse_to_sparse_descr::stage_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_coo2csr(handle, source_, target_, buffer_size_[0], buffer_));
        return rocsparse_status_success;
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

//
// coo2csc
//
FUNCTION_CONVERT(coo, csc)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sparse_to_csr_to_sparse(
        handle, descr_, source_, target_, stage_, buffer_size_, buffer_));
    return rocsparse_status_success;
}

//
// coo2ell
//
FUNCTION_CONVERT(coo, ell)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sparse_to_csr_to_sparse(
        handle, descr_, source_, target_, stage_, buffer_size_, buffer_));
    return rocsparse_status_success;
}

//
// coo2bsr
//
FUNCTION_CONVERT(coo, bsr)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sparse_to_csr_to_sparse(
        handle, descr_, source_, target_, stage_, buffer_size_, buffer_));
    return rocsparse_status_success;
}

//
// coo_aos2coo
//
FUNCTION_CONVERT(coo_aos, coo)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(stage_)
    {
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_coo_aos2coo_buffer_size(handle, source_, target_, buffer_size_));
        return rocsparse_status_success;
    }

    case _rocsparse_sparse_to_sparse_descr::stage_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_coo_aos2coo(handle, source_, target_, buffer_size_[0], buffer_));
        return rocsparse_status_success;
    }

    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_analysis:
    {
        target_->nnz = source_->nnz;
        return rocsparse_status_success;
    }
    }

    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

//
// coo_aos2csr
//
FUNCTION_CONVERT(coo_aos, csr)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(stage_)
    {
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_coo_aos2csr_buffer_size(handle, source_, target_, buffer_size_));
        return rocsparse_status_success;
    }

    case _rocsparse_sparse_to_sparse_descr::stage_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_coo_aos2csr(handle, source_, target_, buffer_size_[0], buffer_));
        return rocsparse_status_success;
    }

    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_analysis:
    {
        target_->nnz = source_->nnz;
        return rocsparse_status_success;
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

//
// csr2coo_aos
//
FUNCTION_CONVERT(csr, coo_aos)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(stage_)
    {
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_csr2coo_aos_buffer_size(handle, source_, target_, buffer_size_));
        return rocsparse_status_success;
    }

    case _rocsparse_sparse_to_sparse_descr::stage_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_csr2coo_aos(handle, source_, target_, buffer_size_[0], buffer_));
        return rocsparse_status_success;
    }

    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_analysis:
    {
        target_->nnz = source_->nnz;
        return rocsparse_status_success;
    }
    }

    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

//
// csr2coo.
//
FUNCTION_CONVERT(csr, coo)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(stage_)
    {
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_csr2coo_buffer_size(handle, source_, target_, buffer_size_));
        return rocsparse_status_success;
    }

    case _rocsparse_sparse_to_sparse_descr::stage_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_csr2coo(handle, source_, target_, buffer_size_[0], buffer_));
        return rocsparse_status_success;
    }

    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_analysis:
    {
        target_->nnz = source_->nnz;
        return rocsparse_status_success;
    }
    }

    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

//
// csr2csc
//
FUNCTION_CONVERT(csr, csc)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(stage_)
    {
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_analysis:
    {
        target_->nnz = source_->nnz;
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_csr2csc_buffer_size(handle, source_, target_, buffer_size_));
        return rocsparse_status_success;
    }

    case _rocsparse_sparse_to_sparse_descr::stage_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_csr2csc(handle, source_, target_, buffer_size_[0], buffer_));
        return rocsparse_status_success;
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

//
// csr2ell
//
FUNCTION_CONVERT(csr, ell)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(stage_)
    {
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis:
    {
        return rocsparse_status_success;
    }

    case _rocsparse_sparse_to_sparse_descr::stage_analysis:
    {
        int64_t ell_width;
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_csr2ell_width(handle, source_, target_, &ell_width));

        //
        // Update target.
        //
        target_->ell_width = ell_width;
        target_->nnz       = ell_width * source_->rows;
        return rocsparse_status_success;
    }

    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_csr2ell_buffer_size(handle, source_, target_, buffer_size_));

        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_csr2ell(handle, source_, target_, buffer_size_[0], buffer_));

        return rocsparse_status_success;
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

//
// csr2bsr.
//
FUNCTION_CONVERT(csr, bsr)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(stage_)
    {
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_analysis:
    {
        int64_t nnzb;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmat_csr2bsr_nnz(handle, source_, target_, &nnzb));
        target_->nnz = nnzb;
        return rocsparse_status_success;
    }

    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_csr2bsr_buffer_size(handle, source_, target_, buffer_size_));
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_csr2bsr(handle, source_, target_, buffer_size_[0], buffer_));

        return rocsparse_status_success;
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

//
// csc2csr
//
FUNCTION_CONVERT(csc, csr)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(stage_)
    {
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_analysis:
    {
        target_->nnz = source_->nnz;
        return rocsparse_status_success;
    }

    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_csc2csr_buffer_size(handle, source_, target_, buffer_size_));
        return rocsparse_status_success;
    }

    case _rocsparse_sparse_to_sparse_descr::stage_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_csc2csr(handle, source_, target_, buffer_size_[0], buffer_));
        return rocsparse_status_success;
    }
    }

    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

//
// ell2csr.
//
FUNCTION_CONVERT(ell, csr)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(stage_)
    {
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_analysis:
    {
        int64_t csr_nnz;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmat_ell2csr_nnz(handle, source_, target_, &csr_nnz));
        target_->nnz = csr_nnz;
        return rocsparse_status_success;
    }

    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_ell2csr_buffer_size(handle, source_, target_, buffer_size_));
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_ell2csr(handle, source_, target_, buffer_size_[0], buffer_));
        return rocsparse_status_success;
    }
    }

    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

//
// bsr2csr
//
FUNCTION_CONVERT(bsr, csr)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(stage_)
    {
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_analysis:
    {
        target_->nnz = source_->nnz * source_->block_dim * source_->block_dim;
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_bsr2csr_buffer_size(handle, source_, target_, buffer_size_));
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_bsr2csr(handle, source_, target_, buffer_size_[0], buffer_));
        return rocsparse_status_success;
    }
    }

    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

//
// csc2cooaos
//
FUNCTION_CONVERT(csc, coo_aos)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sparse_to_csr_to_sparse(
        handle, descr_, source_, target_, stage_, buffer_size_, buffer_));
    return rocsparse_status_success;
}

//
// Functions using a CSR intermediate format.
//
FUNCTION_CONVERT(coo_aos, ell)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sparse_to_csr_to_sparse(
        handle, descr_, source_, target_, stage_, buffer_size_, buffer_));
    return rocsparse_status_success;
}

FUNCTION_CONVERT(coo_aos, bsr)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sparse_to_csr_to_sparse(
        handle, descr_, source_, target_, stage_, buffer_size_, buffer_));
    return rocsparse_status_success;
}

FUNCTION_CONVERT(coo_aos, csc)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sparse_to_csr_to_sparse(
        handle, descr_, source_, target_, stage_, buffer_size_, buffer_));
    return rocsparse_status_success;
}
FUNCTION_CONVERT(csc, coo)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sparse_to_csr_to_sparse(
        handle, descr_, source_, target_, stage_, buffer_size_, buffer_));
    return rocsparse_status_success;
}
FUNCTION_CONVERT(csc, ell)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sparse_to_csr_to_sparse(
        handle, descr_, source_, target_, stage_, buffer_size_, buffer_));
    return rocsparse_status_success;
}
FUNCTION_CONVERT(csc, bsr)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sparse_to_csr_to_sparse(
        handle, descr_, source_, target_, stage_, buffer_size_, buffer_));
    return rocsparse_status_success;
}
FUNCTION_CONVERT(ell, coo)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sparse_to_csr_to_sparse(
        handle, descr_, source_, target_, stage_, buffer_size_, buffer_));
    return rocsparse_status_success;
}
FUNCTION_CONVERT(ell, coo_aos)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sparse_to_csr_to_sparse(
        handle, descr_, source_, target_, stage_, buffer_size_, buffer_));
    return rocsparse_status_success;
}
FUNCTION_CONVERT(ell, csc)
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sparse_to_csr_to_sparse(
        handle, descr_, source_, target_, stage_, buffer_size_, buffer_));
    return rocsparse_status_success;
}
FUNCTION_CONVERT(ell, bsr)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sparse_to_csr_to_sparse(
        handle, descr_, source_, target_, stage_, buffer_size_, buffer_));
    return rocsparse_status_success;
}
FUNCTION_CONVERT(bsr, coo)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sparse_to_csr_to_sparse(
        handle, descr_, source_, target_, stage_, buffer_size_, buffer_));
    return rocsparse_status_success;
}
FUNCTION_CONVERT(bsr, coo_aos)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sparse_to_csr_to_sparse(
        handle, descr_, source_, target_, stage_, buffer_size_, buffer_));
    return rocsparse_status_success;
}
FUNCTION_CONVERT(bsr, csc)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sparse_to_csr_to_sparse(
        handle, descr_, source_, target_, stage_, buffer_size_, buffer_));
    return rocsparse_status_success;
}
FUNCTION_CONVERT(bsr, ell)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sparse_to_csr_to_sparse(
        handle, descr_, source_, target_, stage_, buffer_size_, buffer_));
    return rocsparse_status_success;
}

//
// Identity functions.
//
FUNCTION_CONVERT(coo, coo)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(stage_)
    {
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_compute:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_analysis:
    {
        target_->nnz = source_->nnz;
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::internal_spmat_transfer_from(handle, target_, source_));
        return rocsparse_status_success;
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

FUNCTION_CONVERT(coo_aos, coo_aos)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(stage_)
    {
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_compute:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_analysis:
    {
        target_->nnz = source_->nnz;
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::internal_spmat_transfer_from(handle, target_, source_));
        return rocsparse_status_success;
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

FUNCTION_CONVERT(csr, csr)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(stage_)
    {
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_compute:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_analysis:
    {
        target_->nnz = source_->nnz;

        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::internal_spmat_transfer_from(handle, target_, source_));
        return rocsparse_status_success;
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

FUNCTION_CONVERT(csc, csc)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(stage_)
    {
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_compute:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_analysis:
    {
        target_->nnz = source_->nnz;
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::internal_spmat_transfer_from(handle, target_, source_));
        return rocsparse_status_success;
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

FUNCTION_CONVERT(ell, ell)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(stage_)
    {
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_compute:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_analysis:
    {
        //
        // Update target.
        //
        target_->ell_width = source_->ell_width;
        target_->nnz       = source_->nnz;
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::internal_spmat_transfer_from(handle, target_, source_));
        return rocsparse_status_success;
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

FUNCTION_CONVERT(bsr, bsr)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(stage_)
    {
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_compute:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis:
    {
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_analysis:
    {
        target_->nnz = source_->nnz;
        return rocsparse_status_success;
    }
    case _rocsparse_sparse_to_sparse_descr::stage_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::internal_spmat_transfer_from(handle, target_, source_));
        return rocsparse_status_success;
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

FUNCTION_CONVERT(bell, bell)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}

//
// Not implemented.
//
FUNCTION_CONVERT(coo, bell)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}
FUNCTION_CONVERT(coo_aos, bell)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}
FUNCTION_CONVERT(csr, bell)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}
FUNCTION_CONVERT(csc, bell)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}
FUNCTION_CONVERT(ell, bell)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}
FUNCTION_CONVERT(bsr, bell)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}

FUNCTION_CONVERT(bell, coo)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}
FUNCTION_CONVERT(bell, coo_aos)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}
FUNCTION_CONVERT(bell, csr)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}
FUNCTION_CONVERT(bell, csc)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}
FUNCTION_CONVERT(bell, ell)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}
FUNCTION_CONVERT(bell, bsr)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}

static convert_type s_conversion_table[7][7]{
    {convert_coo_to_coo,
     convert_coo_to_coo_aos,
     convert_coo_to_csr,
     convert_coo_to_csc,
     convert_coo_to_ell,
     convert_coo_to_bell,
     convert_coo_to_bsr},

    {convert_coo_aos_to_coo,
     convert_coo_aos_to_coo_aos,
     convert_coo_aos_to_csr,
     convert_coo_aos_to_csc,
     convert_coo_aos_to_ell,
     convert_coo_aos_to_bell,
     convert_coo_aos_to_bsr},

    {convert_csr_to_coo,
     convert_csr_to_coo_aos,
     convert_csr_to_csr,
     convert_csr_to_csc,
     convert_csr_to_ell,
     convert_csr_to_bell,
     convert_csr_to_bsr},

    {convert_csc_to_coo,
     convert_csc_to_coo_aos,
     convert_csc_to_csr,
     convert_csc_to_csc,
     convert_csc_to_ell,
     convert_csc_to_bell,
     convert_csc_to_bsr},

    {convert_ell_to_coo,
     convert_ell_to_coo_aos,
     convert_ell_to_csr,
     convert_ell_to_csc,
     convert_ell_to_ell,
     convert_ell_to_bell,
     convert_ell_to_bsr},

    {convert_bell_to_coo,
     convert_bell_to_coo_aos,
     convert_bell_to_csr,
     convert_bell_to_csc,
     convert_bell_to_ell,
     convert_bell_to_bell,
     convert_bell_to_bsr},

    {convert_bsr_to_coo,
     convert_bsr_to_coo_aos,
     convert_bsr_to_csr,
     convert_bsr_to_csc,
     convert_bsr_to_ell,
     convert_bsr_to_bell,
     convert_bsr_to_bsr},

};

rocsparse_status rocsparse_mat_descr_are_same(const rocsparse_mat_descr source,
                                              const rocsparse_mat_descr target)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_type_mismatch, source->type != target->type);

    RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_type_mismatch,
                              source->fill_mode != target->fill_mode);

    RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_type_mismatch,
                              source->diag_type != target->diag_type);

    RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_type_mismatch, source->base != target->base);

    RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_type_mismatch,
                              source->storage_mode != target->storage_mode);

    RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_type_mismatch,
                              source->max_nnz_per_row != target->max_nnz_per_row);
    return rocsparse_status_success;
}

rocsparse_status rocsparse::internal_sparse_to_sparse(rocsparse_handle                 handle,
                                                      rocsparse_sparse_to_sparse_descr descr,
                                                      rocsparse_const_spmat_descr      source,
                                                      rocsparse_spmat_descr            target,
                                                      rocsparse_sparse_to_sparse_stage stage,
                                                      size_t*                          buffer_size,
                                                      void*                            buffer,
                                                      bool compute_buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    //
    // Batched not yet supported.
    //
    RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_invalid_size, source->batch_count > 1);

    //
    // 1/ Determine internal stage
    //
    _rocsparse_sparse_to_sparse_descr::stage internal_stage
        = _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis;
    switch(stage)
    {
    case rocsparse_sparse_to_sparse_stage_analysis:
    {
        internal_stage = (compute_buffer_size)
                             ? _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis
                             : _rocsparse_sparse_to_sparse_descr::stage_analysis;
        break;
    }
    case rocsparse_sparse_to_sparse_stage_compute:
    {
        internal_stage = (compute_buffer_size)
                             ? _rocsparse_sparse_to_sparse_descr::stage_buffer_size_compute
                             : _rocsparse_sparse_to_sparse_descr::stage_compute;
        break;
    }
    }

    //
    // Descriptor must be the same.
    //
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_mat_descr_are_same(source->descr, target->descr));

    //
    // Get formats.
    //
    rocsparse_format source_format;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmat_get_format(source, &source_format));
    rocsparse_format target_format;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmat_get_format(target, &target_format));

    RETURN_IF_ROCSPARSE_ERROR(s_conversion_table[source_format][target_format](
        handle, descr, source, target, internal_stage, buffer_size, buffer));
    return rocsparse_status_success;
}

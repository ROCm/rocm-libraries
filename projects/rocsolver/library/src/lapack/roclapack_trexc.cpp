/* **************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include "roclapack_trexc.hpp"
#include "exceptions.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename U>
rocblas_status rocsolver_trexc_impl(rocblas_handle handle,
                                    const rocsolver_schur_vectors compq,
                                    const I n,
                                    U A,
                                    const I ldt,
                                    U Q,
                                    const I ldq,
                                    const I ifst,
                                    const I ilst)
try
{
    ROCSOLVER_ENTER_TOP("trexc", "--compq", compq, "-n", n, "--ldt", ldt, "--ldq", ldq, "--ifst",
                        ifst, "--ilst", ilst);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_trexc_argCheck(handle, compq, n, ldt, ldq, ifst, ilst, A, Q);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftT = 0;
    rocblas_stride shiftQ = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideT = 0;
    rocblas_stride strideQ = 0;
    I batch_count = 1;

    // this function does not require memory work space
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_size_unchanged;

    // execution
    return rocsolver_trexc_template<false, false, T>(handle, compq, n, A, shiftT, ldt, strideT, Q,
                                                     shiftQ, ldq, strideQ, ifst, ilst, batch_count);
}
catch(...)
{
    return exception2rocblas_status();
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ctrexc(rocblas_handle handle,
                                const rocsolver_schur_vectors compq,
                                const rocblas_int n,
                                rocblas_float_complex* T,
                                const rocblas_int ldt,
                                rocblas_float_complex* Q,
                                const rocblas_int ldq,
                                const rocblas_int ifst,
                                const rocblas_int ilst)
{
#if defined(ROCSOLVER_ENABLE_HSEQR)
    return rocsolver::rocsolver_trexc_impl<rocblas_float_complex>(handle, compq, n, T, ldt, Q, ldq,
                                                                  ifst, ilst);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_ztrexc(rocblas_handle handle,
                                const rocsolver_schur_vectors compq,
                                const rocblas_int n,
                                rocblas_double_complex* T,
                                const rocblas_int ldt,
                                rocblas_double_complex* Q,
                                const rocblas_int ldq,
                                const rocblas_int ifst,
                                const rocblas_int ilst)
{
#if defined(ROCSOLVER_ENABLE_HSEQR)
    return rocsolver::rocsolver_trexc_impl<rocblas_double_complex>(handle, compq, n, T, ldt, Q, ldq,
                                                                   ifst, ilst);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C

/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

//
// rocsparse_?gebsrmm with every small row / column block dimension (1 .. 4),
// both block directions, B transposed or not, host and device scalars. The
// whole C, including its padding, is compared with a host reference.
//
#include "unit_test_extra_common.hpp"

#include <set>

using namespace extra_ut;

namespace
{
    // General BSR matrix with rectangular blocks (values row-major per block).
    struct GeBsr
    {
        int64_t              mb = 0, nb = 0, rbd = 1, cbd = 1;
        std::vector<int64_t> ptr, col;
        std::vector<cplx>    val;
        int64_t              nnzb() const
        {
            return static_cast<int64_t>(col.size());
        }
    };

    template <typename T>
    GeBsr random_gebsr(int64_t mb, int64_t nb, int64_t rbd, int64_t cbd, Rng& rng)
    {
        const Csr pattern = random_csr<T>(mb, nb, 3, rng);
        GeBsr     B;
        B.mb  = mb;
        B.nb  = nb;
        B.rbd = rbd;
        B.cbd = cbd;
        B.ptr = pattern.ptr;
        B.col = pattern.col;
        for(int64_t i = 0; i < pattern.nnz() * rbd * cbd; ++i)
            B.val.push_back(round_to<T>(rand_val<T>(rng)));
        return B;
    }

    template <typename T>
    std::vector<T> pack(const GeBsr& B, rocsparse_direction dir)
    {
        std::vector<T> out(B.val.size());
        for(int64_t p = 0; p < B.nnzb(); ++p)
            for(int64_t r = 0; r < B.rbd; ++r)
                for(int64_t c = 0; c < B.cbd; ++c)
                    out[static_cast<size_t>(
                        p * B.rbd * B.cbd
                        + (dir == rocsparse_direction_row ? r * B.cbd + c : c * B.rbd + r))]
                        = from_c<T>(B.val[static_cast<size_t>(p * B.rbd * B.cbd + r * B.cbd + c)]);
        return out;
    }

    Dense expand(const GeBsr& B)
    {
        Dense D(B.mb * B.rbd, B.nb * B.cbd);
        for(int64_t i = 0; i < B.mb; ++i)
            for(int64_t p = B.ptr[static_cast<size_t>(i)]; p < B.ptr[static_cast<size_t>(i) + 1];
                ++p)
                for(int64_t r = 0; r < B.rbd; ++r)
                    for(int64_t c = 0; c < B.cbd; ++c)
                        D.at(i * B.rbd + r, B.col[static_cast<size_t>(p)] * B.cbd + c)
                            = B.val[static_cast<size_t>(p * B.rbd * B.cbd + r * B.cbd + c)];
        return D;
    }

    struct GebsrmmCase
    {
        int64_t             mb     = 5;
        int64_t             kb     = 4;
        int64_t             rbd    = 2;
        int64_t             cbd    = 3;
        int64_t             n      = 7;
        rocsparse_direction dir    = rocsparse_direction_row;
        rocsparse_operation opB    = rocsparse_operation_none;
        bool                device = false;
        cplx                alpha  = cplx(2.0, 0.0);
        cplx                beta   = cplx(-1.0, 0.0);
    };

    template <typename T>
    void run_gebsrmm(rocsparse_handle handle, const GebsrmmCase& c)
    {
        SCOPED_TRACE("rbd=" + std::to_string(c.rbd) + " cbd=" + std::to_string(c.cbd) + " n="
                     + std::to_string(c.n) + " dir=" + std::to_string(static_cast<int>(c.dir))
                     + " opB=" + std::to_string(static_cast<int>(c.opB))
                     + " dev=" + std::to_string(c.device));
        Rng           rng(static_cast<uint64_t>(c.rbd * 37 + c.cbd * 11 + c.n) + 4);
        const GeBsr   A = random_gebsr<T>(c.mb, c.kb, c.rbd, c.cbd, rng);
        const int64_t m = c.mb * c.rbd, k = c.kb * c.cbd;

        const Dense   Bl  = random_dense<T>(k, c.n, rng);
        const Dense   Bs  = apply_op(Bl, c.opB);
        const Dense   C0  = random_dense<T>(m, c.n, rng);
        const int64_t ldb = Bs.rows + 2, ldc = m + 3;

        DevBuf<T>             dB  = up_raw<T>(store_dense<T>(Bs, rocsparse_order_column, ldb));
        DevBuf<T>             dC  = up_raw<T>(store_dense<T>(C0, rocsparse_order_column, ldc));
        DevBuf<rocsparse_int> ptr = up_idx<rocsparse_int>(A.ptr),
                              ind = up_idx<rocsparse_int>(A.col);
        DevBuf<T> val             = up_raw<T>(pack<T>(A, c.dir));
        ASSERT_TRUE(dB.ptr && dC.ptr && ptr.ptr && ind.ptr && val.ptr);

        MatDescrH        descr;
        Scalar<T>        alpha(c.alpha, c.device);
        Scalar<T>        beta(c.beta, c.device);
        PointerModeGuard mode(handle, mode_of(c.device));

        ASSERT_EQ(Api<T>::gebsrmm(handle,
                                  c.dir,
                                  rocsparse_operation_none,
                                  c.opB,
                                  static_cast<rocsparse_int>(c.mb),
                                  static_cast<rocsparse_int>(c.n),
                                  static_cast<rocsparse_int>(c.kb),
                                  static_cast<rocsparse_int>(A.nnzb()),
                                  alpha.typed(),
                                  descr.d,
                                  static_cast<const T*>(val.ptr),
                                  static_cast<const rocsparse_int*>(ptr.ptr),
                                  static_cast<const rocsparse_int*>(ind.ptr),
                                  static_cast<rocsparse_int>(c.rbd),
                                  static_cast<rocsparse_int>(c.cbd),
                                  static_cast<const T*>(dB.ptr),
                                  static_cast<rocsparse_int>(ldb),
                                  beta.typed(),
                                  dC.ptr,
                                  static_cast<rocsparse_int>(ldc)),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const Dense want = axpby(to_c<T>(alpha.h), matmul(expand(A), Bl), to_c<T>(beta.h), C0);
        expect_dense_equal<T>(dC.host(), want, rocsparse_order_column, ldc, "C");
        expect_dense_equal<T>(dB.host(), Bs, rocsparse_order_column, ldb, "B");
    }

    struct ExtraGebsrmm : public rocsparse_ut::HandleTest
    {
    };
}

TEST_F(ExtraGebsrmm, small_block_dimensions_float)
{
    for(int64_t rbd = 1; rbd <= 4; ++rbd)
        for(int64_t cbd = 1; cbd <= 4; ++cbd)
            for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
                for(rocsparse_operation opB :
                    {rocsparse_operation_none, rocsparse_operation_transpose})
                    for(int64_t n : {1, 5, 16})
                    {
                        GebsrmmCase c;
                        c.rbd = rbd;
                        c.cbd = cbd;
                        c.dir = dir;
                        c.opB = opB;
                        c.n   = n;
                        run_gebsrmm<float>(handle, c);
                        if(HasFatalFailure())
                            return;
                    }
}

// Larger (rectangular) blocks select the tuned 8 / 16 / 32 kernels.
TEST_F(ExtraGebsrmm, large_block_dimensions)
{
    const int64_t dims[][2]
        = {{5, 7}, {8, 3}, {6, 8}, {9, 12}, {16, 10}, {13, 16}, {20, 32}, {32, 17}};
    for(const auto& d : dims)
        for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
            for(rocsparse_operation opB : {rocsparse_operation_none, rocsparse_operation_transpose})
            {
                GebsrmmCase c;
                c.mb     = 3;
                c.kb     = 3;
                c.rbd    = d[0];
                c.cbd    = d[1];
                c.n      = 20;
                c.dir    = dir;
                c.opB    = opB;
                c.device = (d[0] + d[1]) % 2 == 1;
                run_gebsrmm<float>(handle, c);
                if(HasFatalFailure())
                    return;
            }
    GebsrmmCase c;
    c.rbd = 9;
    c.cbd = 14;
    c.n   = 3;
    run_gebsrmm<double>(handle, c);
    c.alpha = cplx(1.0, 1.0);
    c.beta  = cplx(0.5, -1.0);
    run_gebsrmm<rocsparse_double_complex>(handle, c);
}

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
// Host-path unit tests for the classic (per-precision) level-1 C API:
//   axpyi, doti, dotci, gthr, gthrz, roti, sctr.
//
// These drive the public C entry points so they exercise the host template
// launch + argument-validation code in the level-1 .cpp files, across all four
// precisions (s/d/c/z) plus the bad-arg and quick-return branches.
//
#include "unit_test_utils.hpp"

using namespace rocsparse_ut;

namespace
{
    // Overload sets: map (T) -> the matching per-precision C function.
#define UT_OVL_AXPYI(T, PFX)                                  \
    inline rocsparse_status ut_axpyi(rocsparse_handle     h,  \
                                     rocsparse_int        n,  \
                                     const T*             a,  \
                                     const T*             xv, \
                                     const rocsparse_int* xi, \
                                     T*                   y,  \
                                     rocsparse_index_base b)  \
    {                                                         \
        return rocsparse_##PFX##axpyi(h, n, a, xv, xi, y, b); \
    }
    UT_OVL_AXPYI(float, s)
    UT_OVL_AXPYI(double, d)
    UT_OVL_AXPYI(rocsparse_float_complex, c)
    UT_OVL_AXPYI(rocsparse_double_complex, z)

#define UT_OVL_DOTI(T, PFX)                                    \
    inline rocsparse_status ut_doti(rocsparse_handle     h,    \
                                    rocsparse_int        n,    \
                                    const T*             xv,   \
                                    const rocsparse_int* xi,   \
                                    const T*             y,    \
                                    T*                   res,  \
                                    rocsparse_index_base b)    \
    {                                                          \
        return rocsparse_##PFX##doti(h, n, xv, xi, y, res, b); \
    }
    UT_OVL_DOTI(float, s)
    UT_OVL_DOTI(double, d)
    UT_OVL_DOTI(rocsparse_float_complex, c)
    UT_OVL_DOTI(rocsparse_double_complex, z)

#define UT_OVL_DOTCI(T, PFX)                                    \
    inline rocsparse_status ut_dotci(rocsparse_handle     h,    \
                                     rocsparse_int        n,    \
                                     const T*             xv,   \
                                     const rocsparse_int* xi,   \
                                     const T*             y,    \
                                     T*                   res,  \
                                     rocsparse_index_base b)    \
    {                                                           \
        return rocsparse_##PFX##dotci(h, n, xv, xi, y, res, b); \
    }
    UT_OVL_DOTCI(rocsparse_float_complex, c)
    UT_OVL_DOTCI(rocsparse_double_complex, z)

#define UT_OVL_GTHR(T, PFX)                                  \
    inline rocsparse_status ut_gthr(rocsparse_handle     h,  \
                                    rocsparse_int        n,  \
                                    const T*             y,  \
                                    T*                   xv, \
                                    const rocsparse_int* xi, \
                                    rocsparse_index_base b)  \
    {                                                        \
        return rocsparse_##PFX##gthr(h, n, y, xv, xi, b);    \
    }
    UT_OVL_GTHR(float, s)
    UT_OVL_GTHR(double, d)
    UT_OVL_GTHR(rocsparse_float_complex, c)
    UT_OVL_GTHR(rocsparse_double_complex, z)

#define UT_OVL_GTHRZ(T, PFX)                                  \
    inline rocsparse_status ut_gthrz(rocsparse_handle     h,  \
                                     rocsparse_int        n,  \
                                     T*                   y,  \
                                     T*                   xv, \
                                     const rocsparse_int* xi, \
                                     rocsparse_index_base b)  \
    {                                                         \
        return rocsparse_##PFX##gthrz(h, n, y, xv, xi, b);    \
    }
    UT_OVL_GTHRZ(float, s)
    UT_OVL_GTHRZ(double, d)
    UT_OVL_GTHRZ(rocsparse_float_complex, c)
    UT_OVL_GTHRZ(rocsparse_double_complex, z)

#define UT_OVL_ROTI(T, PFX)                                     \
    inline rocsparse_status ut_roti(rocsparse_handle     h,     \
                                    rocsparse_int        n,     \
                                    T*                   xv,    \
                                    const rocsparse_int* xi,    \
                                    T*                   y,     \
                                    const T*             c,     \
                                    const T*             s,     \
                                    rocsparse_index_base b)     \
    {                                                           \
        return rocsparse_##PFX##roti(h, n, xv, xi, y, c, s, b); \
    }
    UT_OVL_ROTI(float, s)
    UT_OVL_ROTI(double, d)

#define UT_OVL_SCTR(T, PFX)                                  \
    inline rocsparse_status ut_sctr(rocsparse_handle     h,  \
                                    rocsparse_int        n,  \
                                    const T*             xv, \
                                    const rocsparse_int* xi, \
                                    T*                   y,  \
                                    rocsparse_index_base b)  \
    {                                                        \
        return rocsparse_##PFX##sctr(h, n, xv, xi, y, b);    \
    }
    UT_OVL_SCTR(float, s)
    UT_OVL_SCTR(double, d)
    UT_OVL_SCTR(rocsparse_float_complex, c)
    UT_OVL_SCTR(rocsparse_double_complex, z)

    // Common tiny sparse-vector fixtures (nnz=3 within size 8).
    template <typename T>
    struct L1Data
    {
        std::vector<rocsparse_int>   hind{0, 3, 6};
        device_vector<rocsparse_int> ind{hind};
        device_vector<T>             xval{std::vector<T>{scalar<T>(1), scalar<T>(2), scalar<T>(3)}};
        device_vector<T>             y{std::vector<T>(8, scalar<T>(1))};
        bool                         ok() const
        {
            return ind.ptr && xval.ptr && y.ptr;
        }
    };

    class L1Classic : public HandleTest
    {
    };

    template <typename T>
    void run_axpyi(rocsparse_handle handle)
    {
        L1Data<T> d;
        ASSERT_TRUE(d.ok());
        const T alpha = scalar<T>(2);
        // valid
        UT_EXPECT_ROC(ut_axpyi(handle, 3, &alpha, d.xval, d.ind, d.y, rocsparse_index_base_zero),
                      rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        // quick return: nnz == 0
        UT_EXPECT_ROC(ut_axpyi(handle, 0, &alpha, d.xval, d.ind, d.y, rocsparse_index_base_zero),
                      rocsparse_status_success);
        // quick return: alpha == 0 (host pointer mode)
        const T zero = scalar<T>(0);
        UT_EXPECT_ROC(ut_axpyi(handle, 3, &zero, d.xval, d.ind, d.y, rocsparse_index_base_zero),
                      rocsparse_status_success);
        // bad args
        UT_EXPECT_ROC(ut_axpyi(nullptr, 3, &alpha, d.xval, d.ind, d.y, rocsparse_index_base_zero),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_axpyi(handle, -1, &alpha, d.xval, d.ind, d.y, rocsparse_index_base_zero),
                      rocsparse_status_invalid_size);
        UT_EXPECT_ROC(ut_axpyi(handle, 3, nullptr, d.xval, d.ind, d.y, rocsparse_index_base_zero),
                      rocsparse_status_invalid_pointer);
        UT_EXPECT_ROC(ut_axpyi(handle, 3, &alpha, nullptr, d.ind, d.y, rocsparse_index_base_zero),
                      rocsparse_status_invalid_pointer);
    }

    template <typename T>
    void run_doti(rocsparse_handle handle)
    {
        L1Data<T> d;
        ASSERT_TRUE(d.ok());
        T res = scalar<T>(0);
        UT_EXPECT_ROC(ut_doti(handle, 3, d.xval, d.ind, d.y, &res, rocsparse_index_base_zero),
                      rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        UT_EXPECT_ROC(ut_doti(handle, 0, d.xval, d.ind, d.y, &res, rocsparse_index_base_zero),
                      rocsparse_status_success);
        UT_EXPECT_ROC(ut_doti(nullptr, 3, d.xval, d.ind, d.y, &res, rocsparse_index_base_zero),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_doti(handle, -1, d.xval, d.ind, d.y, &res, rocsparse_index_base_zero),
                      rocsparse_status_invalid_size);
        UT_EXPECT_ROC(ut_doti(handle, 3, d.xval, d.ind, d.y, nullptr, rocsparse_index_base_zero),
                      rocsparse_status_invalid_pointer);
    }

    template <typename T>
    void run_dotci(rocsparse_handle handle)
    {
        L1Data<T> d;
        ASSERT_TRUE(d.ok());
        T res = scalar<T>(0);
        UT_EXPECT_ROC(ut_dotci(handle, 3, d.xval, d.ind, d.y, &res, rocsparse_index_base_zero),
                      rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        UT_EXPECT_ROC(ut_dotci(handle, 0, d.xval, d.ind, d.y, &res, rocsparse_index_base_zero),
                      rocsparse_status_success);
        UT_EXPECT_ROC(ut_dotci(nullptr, 3, d.xval, d.ind, d.y, &res, rocsparse_index_base_zero),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_dotci(handle, 3, d.xval, d.ind, d.y, nullptr, rocsparse_index_base_zero),
                      rocsparse_status_invalid_pointer);
    }

    template <typename T>
    void run_gthr(rocsparse_handle handle)
    {
        L1Data<T> d;
        ASSERT_TRUE(d.ok());
        UT_EXPECT_ROC(ut_gthr(handle, 3, d.y, d.xval, d.ind, rocsparse_index_base_zero),
                      rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        UT_EXPECT_ROC(ut_gthr(handle, 0, d.y, d.xval, d.ind, rocsparse_index_base_zero),
                      rocsparse_status_success);
        UT_EXPECT_ROC(ut_gthr(nullptr, 3, d.y, d.xval, d.ind, rocsparse_index_base_zero),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_gthr(handle, -1, d.y, d.xval, d.ind, rocsparse_index_base_zero),
                      rocsparse_status_invalid_size);
        UT_EXPECT_ROC(ut_gthr(handle, 3, nullptr, d.xval, d.ind, rocsparse_index_base_zero),
                      rocsparse_status_invalid_pointer);
    }

    template <typename T>
    void run_gthrz(rocsparse_handle handle)
    {
        L1Data<T> d;
        ASSERT_TRUE(d.ok());
        UT_EXPECT_ROC(ut_gthrz(handle, 3, d.y, d.xval, d.ind, rocsparse_index_base_zero),
                      rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        UT_EXPECT_ROC(ut_gthrz(handle, 0, d.y, d.xval, d.ind, rocsparse_index_base_zero),
                      rocsparse_status_success);
        UT_EXPECT_ROC(ut_gthrz(nullptr, 3, d.y, d.xval, d.ind, rocsparse_index_base_zero),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_gthrz(handle, 3, nullptr, d.xval, d.ind, rocsparse_index_base_zero),
                      rocsparse_status_invalid_pointer);
    }

    template <typename T>
    void run_roti(rocsparse_handle handle)
    {
        L1Data<T> d;
        ASSERT_TRUE(d.ok());
        const T c = scalar<T>(1), s = scalar<T>(0);
        UT_EXPECT_ROC(ut_roti(handle, 3, d.xval, d.ind, d.y, &c, &s, rocsparse_index_base_zero),
                      rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        UT_EXPECT_ROC(ut_roti(handle, 0, d.xval, d.ind, d.y, &c, &s, rocsparse_index_base_zero),
                      rocsparse_status_success);
        UT_EXPECT_ROC(ut_roti(nullptr, 3, d.xval, d.ind, d.y, &c, &s, rocsparse_index_base_zero),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_roti(handle, -1, d.xval, d.ind, d.y, &c, &s, rocsparse_index_base_zero),
                      rocsparse_status_invalid_size);
        UT_EXPECT_ROC(
            ut_roti(handle, 3, d.xval, d.ind, d.y, nullptr, &s, rocsparse_index_base_zero),
            rocsparse_status_invalid_pointer);
    }

    template <typename T>
    void run_sctr(rocsparse_handle handle)
    {
        L1Data<T> d;
        ASSERT_TRUE(d.ok());
        UT_EXPECT_ROC(ut_sctr(handle, 3, d.xval, d.ind, d.y, rocsparse_index_base_zero),
                      rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        UT_EXPECT_ROC(ut_sctr(handle, 0, d.xval, d.ind, d.y, rocsparse_index_base_zero),
                      rocsparse_status_success);
        UT_EXPECT_ROC(ut_sctr(nullptr, 3, d.xval, d.ind, d.y, rocsparse_index_base_zero),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_sctr(handle, -1, d.xval, d.ind, d.y, rocsparse_index_base_zero),
                      rocsparse_status_invalid_size);
        UT_EXPECT_ROC(ut_sctr(handle, 3, nullptr, d.ind, d.y, rocsparse_index_base_zero),
                      rocsparse_status_invalid_pointer);
    }
} // namespace

#define UT_L1_ALL_PRECISIONS(NAME, FN)        \
    TEST_F(L1Classic, NAME)                   \
    {                                         \
        FN<float>(handle);                    \
        FN<double>(handle);                   \
        FN<rocsparse_float_complex>(handle);  \
        FN<rocsparse_double_complex>(handle); \
    }

UT_L1_ALL_PRECISIONS(axpyi, run_axpyi)
UT_L1_ALL_PRECISIONS(doti, run_doti)
UT_L1_ALL_PRECISIONS(gthr, run_gthr)
UT_L1_ALL_PRECISIONS(gthrz, run_gthrz)
UT_L1_ALL_PRECISIONS(sctr, run_sctr)

TEST_F(L1Classic, roti)
{
    run_roti<float>(handle);
    run_roti<double>(handle);
}

TEST_F(L1Classic, dotci)
{
    run_dotci<rocsparse_float_complex>(handle);
    run_dotci<rocsparse_double_complex>(handle);
}

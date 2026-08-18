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
// Host-path unit tests for level-1 GENERIC-API routines.
//
// Unlike unit_test_device_level1.cpp (which launches kernels directly), these
// drive the public C API entry points so they exercise the *host* dispatch and
// validation code in the level-1 .cpp files -- exactly the lines rocSPARSE code
// coverage counts (coverage is host-only; device-kernel lines are not counted).
//
// The main gap in these files is the per-(indextype,datatype) dispatch chain
// and the type-mismatch / not_implemented guards, which the *quick*/*pre_checkin*
// YAML corpus does not fully exercise. These tests hit every branch with tiny
// valid inputs and deliberately invalid ones.
//
// Requires a GPU (creates a handle + launches kernels via the public API); part
// of the rocsparse-unit-test-device binary.
//
#include "rocsparse.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace
{
#define CHECK_HIP(cmd)                                                \
    do                                                                \
    {                                                                 \
        hipError_t status_ = (cmd);                                   \
        ASSERT_EQ(status_, hipSuccess) << hipGetErrorString(status_); \
    } while(0)

#define EXPECT_ROC(cmd, expected) EXPECT_EQ((cmd), (expected))

    // Map a C++ type to its rocsparse_datatype.
    template <typename T>
    rocsparse_datatype dt_of();
    template <>
    rocsparse_datatype dt_of<float>()
    {
        return rocsparse_datatype_f32_r;
    }
    template <>
    rocsparse_datatype dt_of<double>()
    {
        return rocsparse_datatype_f64_r;
    }
    template <>
    rocsparse_datatype dt_of<rocsparse_float_complex>()
    {
        return rocsparse_datatype_f32_c;
    }
    template <>
    rocsparse_datatype dt_of<rocsparse_double_complex>()
    {
        return rocsparse_datatype_f64_c;
    }

    template <typename T>
    T scalar(float v)
    {
        return static_cast<T>(v);
    }
    template <>
    rocsparse_float_complex scalar<rocsparse_float_complex>(float v)
    {
        return rocsparse_float_complex(v, 0.0f);
    }
    template <>
    rocsparse_double_complex scalar<rocsparse_double_complex>(float v)
    {
        return rocsparse_double_complex(v, 0.0);
    }

    // gtest fixture owning a rocsparse handle for the whole level-1 host suite.
    class HostLevel1 : public ::testing::Test
    {
    protected:
        rocsparse_handle handle = nullptr;
        void             SetUp() override
        {
            ASSERT_EQ(rocsparse_create_handle(&handle), rocsparse_status_success);
        }
        void TearDown() override
        {
            if(handle)
            {
                EXPECT_EQ(rocsparse_destroy_handle(handle), rocsparse_status_success);
            }
        }
    };

    // Tiny device-backed sparse (x) + dense (y) vector pair for a given type pair.
    template <typename T, typename I>
    struct VecPair
    {
        I*                    d_ind  = nullptr;
        T*                    d_xval = nullptr;
        T*                    d_y    = nullptr;
        rocsparse_spvec_descr x      = nullptr;
        rocsparse_dnvec_descr y      = nullptr;

        bool create(rocsparse_indextype it, rocsparse_datatype x_dt, rocsparse_datatype y_dt)
        {
            const std::vector<I> ind{0, 2, 4};
            const std::vector<T> xval{scalar<T>(1), scalar<T>(2), scalar<T>(3)};
            const std::vector<T> yv(5, scalar<T>(1));
            if(hipMalloc(&d_ind, ind.size() * sizeof(I)) != hipSuccess
               || hipMalloc(&d_xval, xval.size() * sizeof(T)) != hipSuccess
               || hipMalloc(&d_y, yv.size() * sizeof(T)) != hipSuccess)
            {
                return false;
            }
            (void)hipMemcpy(d_ind, ind.data(), ind.size() * sizeof(I), hipMemcpyHostToDevice);
            (void)hipMemcpy(d_xval, xval.data(), xval.size() * sizeof(T), hipMemcpyHostToDevice);
            (void)hipMemcpy(d_y, yv.data(), yv.size() * sizeof(T), hipMemcpyHostToDevice);
            return rocsparse_create_spvec_descr(
                       &x, 5, 3, d_ind, d_xval, it, rocsparse_index_base_zero, x_dt)
                       == rocsparse_status_success
                   && rocsparse_create_dnvec_descr(&y, 5, d_y, y_dt) == rocsparse_status_success;
        }

        ~VecPair()
        {
            if(x)
                (void)rocsparse_destroy_spvec_descr(x);
            if(y)
                (void)rocsparse_destroy_dnvec_descr(y);
            (void)hipFree(d_ind);
            (void)hipFree(d_xval);
            (void)hipFree(d_y);
        }
    };

    // Exercise the rocsparse_rot dispatch branch for one (T, I) combination.
    template <typename T, typename I>
    void rot_dispatch_ok(rocsparse_handle handle, rocsparse_indextype it)
    {
        VecPair<T, I> vp;
        ASSERT_TRUE(vp.create(it, dt_of<T>(), dt_of<T>()));
        const T c = scalar<T>(1);
        const T s = scalar<T>(0);
        EXPECT_ROC(rocsparse_rot(handle, &c, &s, vp.x, vp.y), rocsparse_status_success);
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
    }

    // Drive rocsparse_spvv end-to-end for one (T, I) combination and index base:
    // the buffer-size query, then the compute with BOTH rocsparse_operation_none
    // and rocsparse_operation_conjugate_transpose. For complex T the conjugate
    // path selects rocsparse::dotci_template (the previously untested branch in
    // spvv_template_complex); for real T both operations hit the doti path.
    template <typename T, typename I>
    void spvv_run(rocsparse_handle handle, rocsparse_indextype it, rocsparse_index_base base)
    {
        const I              off = (base == rocsparse_index_base_one) ? 1 : 0;
        const std::vector<I> ind{
            static_cast<I>(0 + off), static_cast<I>(2 + off), static_cast<I>(4 + off)};
        const std::vector<T> xval{scalar<T>(1), scalar<T>(2), scalar<T>(3)};
        const std::vector<T> yv(5, scalar<T>(1));

        I* d_ind  = nullptr;
        T* d_xval = nullptr;
        T* d_y    = nullptr;
        ASSERT_EQ(hipMalloc(&d_ind, ind.size() * sizeof(I)), hipSuccess);
        ASSERT_EQ(hipMalloc(&d_xval, xval.size() * sizeof(T)), hipSuccess);
        ASSERT_EQ(hipMalloc(&d_y, yv.size() * sizeof(T)), hipSuccess);
        (void)hipMemcpy(d_ind, ind.data(), ind.size() * sizeof(I), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_xval, xval.data(), xval.size() * sizeof(T), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_y, yv.data(), yv.size() * sizeof(T), hipMemcpyHostToDevice);

        rocsparse_spvec_descr x = nullptr;
        rocsparse_dnvec_descr y = nullptr;
        ASSERT_EQ(rocsparse_create_spvec_descr(&x, 5, 3, d_ind, d_xval, it, base, dt_of<T>()),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&y, 5, d_y, dt_of<T>()), rocsparse_status_success);

        // Default handle pointer mode is host -> result lives on the host.
        T      result      = scalar<T>(0);
        size_t buffer_size = 0;

        // buffer-size query (temp_buffer == nullptr).
        EXPECT_ROC(
            rocsparse_spvv(
                handle, rocsparse_operation_none, x, y, &result, dt_of<T>(), &buffer_size, nullptr),
            rocsparse_status_success);

        void* temp = nullptr;
        ASSERT_EQ(hipMalloc(&temp, buffer_size ? buffer_size : 4), hipSuccess);

        // Non-transpose (plain dot) path.
        EXPECT_ROC(
            rocsparse_spvv(
                handle, rocsparse_operation_none, x, y, &result, dt_of<T>(), &buffer_size, temp),
            rocsparse_status_success);
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);

        // Conjugate-transpose path (conjugated dot for complex, plain for real).
        EXPECT_ROC(rocsparse_spvv(handle,
                                  rocsparse_operation_conjugate_transpose,
                                  x,
                                  y,
                                  &result,
                                  dt_of<T>(),
                                  &buffer_size,
                                  temp),
                   rocsparse_status_success);
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);

        (void)rocsparse_destroy_spvec_descr(x);
        (void)rocsparse_destroy_dnvec_descr(y);
        (void)hipFree(d_ind);
        (void)hipFree(d_xval);
        (void)hipFree(d_y);
        (void)hipFree(temp);
    }
} // namespace

// ---------------------------------------------------------------------------
// rocsparse_rot: all 8 (indextype x datatype) dispatch branches.
// Covers the large type-dispatch block in rocsparse_rot.cpp.
// ---------------------------------------------------------------------------
TEST_F(HostLevel1, rot_dispatch_i32)
{
    rot_dispatch_ok<float, int32_t>(handle, rocsparse_indextype_i32);
    rot_dispatch_ok<double, int32_t>(handle, rocsparse_indextype_i32);
    rot_dispatch_ok<rocsparse_float_complex, int32_t>(handle, rocsparse_indextype_i32);
    rot_dispatch_ok<rocsparse_double_complex, int32_t>(handle, rocsparse_indextype_i32);
}

TEST_F(HostLevel1, rot_dispatch_i64)
{
    rot_dispatch_ok<float, int64_t>(handle, rocsparse_indextype_i64);
    rot_dispatch_ok<double, int64_t>(handle, rocsparse_indextype_i64);
    rot_dispatch_ok<rocsparse_float_complex, int64_t>(handle, rocsparse_indextype_i64);
    rot_dispatch_ok<rocsparse_double_complex, int64_t>(handle, rocsparse_indextype_i64);
}

// ---------------------------------------------------------------------------
// Type-mismatch guards (rocsparse_status_not_implemented) in the generic
// level-1 wrappers: x and y have different data types.
// ---------------------------------------------------------------------------
TEST_F(HostLevel1, rot_type_mismatch_not_implemented)
{
    VecPair<float, int32_t> vp;
    // x is f32_r, y is f64_r -> mismatch.
    ASSERT_TRUE(
        vp.create(rocsparse_indextype_i32, rocsparse_datatype_f32_r, rocsparse_datatype_f64_r));
    const float c = 1, s = 0;
    EXPECT_ROC(rocsparse_rot(handle, &c, &s, vp.x, vp.y), rocsparse_status_not_implemented);
}

TEST_F(HostLevel1, gather_type_mismatch_not_implemented)
{
    VecPair<float, int32_t> vp;
    ASSERT_TRUE(
        vp.create(rocsparse_indextype_i32, rocsparse_datatype_f32_r, rocsparse_datatype_f64_r));
    // gather(handle, y, x): result gathered into x from y.
    EXPECT_ROC(rocsparse_gather(handle, vp.y, vp.x), rocsparse_status_not_implemented);
}

TEST_F(HostLevel1, scatter_type_mismatch_not_implemented)
{
    VecPair<float, int32_t> vp;
    ASSERT_TRUE(
        vp.create(rocsparse_indextype_i32, rocsparse_datatype_f32_r, rocsparse_datatype_f64_r));
    EXPECT_ROC(rocsparse_scatter(handle, vp.x, vp.y), rocsparse_status_not_implemented);
}

TEST_F(HostLevel1, axpby_type_mismatch_not_implemented)
{
    VecPair<float, int32_t> vp;
    ASSERT_TRUE(
        vp.create(rocsparse_indextype_i32, rocsparse_datatype_f32_r, rocsparse_datatype_f64_r));
    const float alpha = 1, beta = 1;
    EXPECT_ROC(rocsparse_axpby(handle, &alpha, vp.x, &beta, vp.y),
               rocsparse_status_not_implemented);
}

// ---------------------------------------------------------------------------
// gather / scatter valid dispatch (real types) exercise the success path in
// rocsparse_gather.cpp / rocsparse_scatter.cpp beyond the guards.
// ---------------------------------------------------------------------------
TEST_F(HostLevel1, gather_ok)
{
    VecPair<float, int32_t> vp;
    ASSERT_TRUE(
        vp.create(rocsparse_indextype_i32, rocsparse_datatype_f32_r, rocsparse_datatype_f32_r));
    EXPECT_ROC(rocsparse_gather(handle, vp.y, vp.x), rocsparse_status_success);
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST_F(HostLevel1, scatter_ok)
{
    VecPair<float, int32_t> vp;
    ASSERT_TRUE(
        vp.create(rocsparse_indextype_i32, rocsparse_datatype_f32_r, rocsparse_datatype_f32_r));
    EXPECT_ROC(rocsparse_scatter(handle, vp.x, vp.y), rocsparse_status_success);
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST_F(HostLevel1, axpby_ok_and_quick_return)
{
    // Normal size -> full path.
    {
        VecPair<float, int32_t> vp;
        ASSERT_TRUE(
            vp.create(rocsparse_indextype_i32, rocsparse_datatype_f32_r, rocsparse_datatype_f32_r));
        const float alpha = 2, beta = 3;
        EXPECT_ROC(rocsparse_axpby(handle, &alpha, vp.x, &beta, vp.y), rocsparse_status_success);
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
    }
}

// ---------------------------------------------------------------------------
// rocsparse_spvv: exercise the full compute path for all 4 datatypes, both
// index types, both index bases, and BOTH operations. For complex types the
// conjugate_transpose operation selects the previously-untested
// rocsparse::dotci_template branch inside spvv_template_complex; the none
// operation selects the doti branch. For real types both operations funnel
// through spvv_template_real's doti path.
// ---------------------------------------------------------------------------
TEST_F(HostLevel1, spvv_dispatch_i32_base_zero)
{
    spvv_run<float, int32_t>(handle, rocsparse_indextype_i32, rocsparse_index_base_zero);
    spvv_run<double, int32_t>(handle, rocsparse_indextype_i32, rocsparse_index_base_zero);
    spvv_run<rocsparse_float_complex, int32_t>(
        handle, rocsparse_indextype_i32, rocsparse_index_base_zero);
    spvv_run<rocsparse_double_complex, int32_t>(
        handle, rocsparse_indextype_i32, rocsparse_index_base_zero);
}

TEST_F(HostLevel1, spvv_dispatch_i32_base_one)
{
    spvv_run<float, int32_t>(handle, rocsparse_indextype_i32, rocsparse_index_base_one);
    spvv_run<double, int32_t>(handle, rocsparse_indextype_i32, rocsparse_index_base_one);
    spvv_run<rocsparse_float_complex, int32_t>(
        handle, rocsparse_indextype_i32, rocsparse_index_base_one);
    spvv_run<rocsparse_double_complex, int32_t>(
        handle, rocsparse_indextype_i32, rocsparse_index_base_one);
}

TEST_F(HostLevel1, spvv_dispatch_i64_base_zero)
{
    spvv_run<float, int64_t>(handle, rocsparse_indextype_i64, rocsparse_index_base_zero);
    spvv_run<double, int64_t>(handle, rocsparse_indextype_i64, rocsparse_index_base_zero);
    spvv_run<rocsparse_float_complex, int64_t>(
        handle, rocsparse_indextype_i64, rocsparse_index_base_zero);
    spvv_run<rocsparse_double_complex, int64_t>(
        handle, rocsparse_indextype_i64, rocsparse_index_base_zero);
}

TEST_F(HostLevel1, spvv_dispatch_i64_base_one)
{
    spvv_run<float, int64_t>(handle, rocsparse_indextype_i64, rocsparse_index_base_one);
    spvv_run<double, int64_t>(handle, rocsparse_indextype_i64, rocsparse_index_base_one);
    spvv_run<rocsparse_float_complex, int64_t>(
        handle, rocsparse_indextype_i64, rocsparse_index_base_one);
    spvv_run<rocsparse_double_complex, int64_t>(
        handle, rocsparse_indextype_i64, rocsparse_index_base_one);
}

// ---------------------------------------------------------------------------
// rocsparse_spvv argument-checking branches (the ROCSPARSE_CHECKARG_* guards in
// the C wrapper). Each deliberately-invalid call must return the matching
// status without touching the compute path.
// ---------------------------------------------------------------------------
TEST_F(HostLevel1, spvv_invalid_args)
{
    VecPair<float, int32_t> vp;
    ASSERT_TRUE(
        vp.create(rocsparse_indextype_i32, rocsparse_datatype_f32_r, rocsparse_datatype_f32_r));

    float  result      = 0;
    size_t buffer_size = 0;

    // Invalid handle.
    EXPECT_ROC(rocsparse_spvv(nullptr,
                              rocsparse_operation_none,
                              vp.x,
                              vp.y,
                              &result,
                              rocsparse_datatype_f32_r,
                              &buffer_size,
                              nullptr),
               rocsparse_status_invalid_handle);

    // Invalid operation enum.
    EXPECT_ROC(rocsparse_spvv(handle,
                              static_cast<rocsparse_operation>(0x7FFFFFFF),
                              vp.x,
                              vp.y,
                              &result,
                              rocsparse_datatype_f32_r,
                              &buffer_size,
                              nullptr),
               rocsparse_status_invalid_value);

    // Invalid compute type enum.
    EXPECT_ROC(rocsparse_spvv(handle,
                              rocsparse_operation_none,
                              vp.x,
                              vp.y,
                              &result,
                              static_cast<rocsparse_datatype>(0x7FFFFFFF),
                              &buffer_size,
                              nullptr),
               rocsparse_status_invalid_value);

    // Null sparse descriptor.
    EXPECT_ROC(rocsparse_spvv(handle,
                              rocsparse_operation_none,
                              nullptr,
                              vp.y,
                              &result,
                              rocsparse_datatype_f32_r,
                              &buffer_size,
                              nullptr),
               rocsparse_status_invalid_pointer);

    // Null dense descriptor.
    EXPECT_ROC(rocsparse_spvv(handle,
                              rocsparse_operation_none,
                              vp.x,
                              nullptr,
                              &result,
                              rocsparse_datatype_f32_r,
                              &buffer_size,
                              nullptr),
               rocsparse_status_invalid_pointer);

    // Null result pointer.
    EXPECT_ROC(rocsparse_spvv(handle,
                              rocsparse_operation_none,
                              vp.x,
                              vp.y,
                              nullptr,
                              rocsparse_datatype_f32_r,
                              &buffer_size,
                              nullptr),
               rocsparse_status_invalid_pointer);

    // Null buffer_size when temp_buffer is also null.
    EXPECT_ROC(rocsparse_spvv(handle,
                              rocsparse_operation_none,
                              vp.x,
                              vp.y,
                              &result,
                              rocsparse_datatype_f32_r,
                              nullptr,
                              nullptr),
               rocsparse_status_invalid_pointer);
}

// ---------------------------------------------------------------------------
// rocsparse_spvv not-implemented branch inside spvv_template_complex: a
// complex config is selected, temp_buffer is non-null (so we get past the
// buffer-size early return), and the operation is rocsparse_operation_transpose
// -- neither the plain-dot (none) nor conjugated-dot (conjugate_transpose)
// branch is taken, so the function falls through to the not_implemented guard.
// Covers rocsparse_spvv.cpp L127.
// ---------------------------------------------------------------------------
TEST_F(HostLevel1, spvv_transpose_complex_not_implemented)
{
    VecPair<rocsparse_float_complex, int32_t> vp;
    ASSERT_TRUE(
        vp.create(rocsparse_indextype_i32, rocsparse_datatype_f32_c, rocsparse_datatype_f32_c));

    rocsparse_float_complex result      = scalar<rocsparse_float_complex>(0);
    size_t                  buffer_size = 0;

    void* temp = nullptr;
    ASSERT_EQ(hipMalloc(&temp, 4), hipSuccess);

    EXPECT_ROC(rocsparse_spvv(handle,
                              rocsparse_operation_transpose,
                              vp.x,
                              vp.y,
                              &result,
                              rocsparse_datatype_f32_c,
                              &buffer_size,
                              temp),
               rocsparse_status_not_implemented);

    (void)hipFree(temp);
}

// ---------------------------------------------------------------------------
// rocsparse_spvv invalid precision configuration branch in spvv_find: the
// (compute_type, index_type, x_type, y_type) tuple is not present in the
// dispatch map (here x is f32_r but y is f64_r, a combination the map does not
// contain), so spvv_find takes the else branch, builds the error message and
// returns rocsparse_status_invalid_value. Covers rocsparse_spvv.cpp L244-279.
// ---------------------------------------------------------------------------
TEST_F(HostLevel1, spvv_invalid_precision_config)
{
    VecPair<float, int32_t> vp;
    ASSERT_TRUE(
        vp.create(rocsparse_indextype_i32, rocsparse_datatype_f32_r, rocsparse_datatype_f64_r));

    float  result      = 0;
    size_t buffer_size = 0;

    EXPECT_ROC(rocsparse_spvv(handle,
                              rocsparse_operation_none,
                              vp.x,
                              vp.y,
                              &result,
                              rocsparse_datatype_f32_r,
                              &buffer_size,
                              nullptr),
               rocsparse_status_invalid_value);
}

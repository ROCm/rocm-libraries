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
// Device-target unit tests for rocSPARSE internal data-structure member
// functions (info structs).
//
// Compiled into the rocsparse-unit-test-device binary. Exercises member
// functions of the internal info/handle structures directly (they are private
// to the library and hidden in the .so, so the matching library/src TUs that
// define them are compiled into this binary -- see CMakeLists.txt,
// ROCSPARSE_UNIT_TEST_DEVICE_LIB_SOURCES). Coverage:
//
//   * rocsparse::numeric_boost           (common/rocsparse_numeric_boost.cpp)
//   * _rocsparse_csrmv_info::clear + adaptive/lrb/nnzsplit clear
//         (header-inline clear; level2/rocsparse_csrmv_{adaptive,lrb,nnzsplit}_info.cpp)
//   * nnzsplit csrmv analysis buffer returned to the device memory pool on
//         rocsparse_destroy_spmat_descr (rocm-libraries#6595)
//   * rocsparse csrgemm_info create/copy/destroy (extra/rocsparse_csrgemm_info.cpp)
//   * _rocsparse_mat_info lifecycle via the public C API (rocsparse_create/destroy_mat_info)
//
#include "unit_test_utils.hpp"

// Internal (private) library headers under test. Reachable because the device
// unit-test target adds library/src/include to its include path.
#include "rocsparse_csrgemm_info.hpp"
#include "rocsparse_csrmv_info.hpp"
#include "rocsparse_numeric_boost.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <vector>

using namespace rocsparse_ut;

// ===========================================================================
// harness smoke
// ===========================================================================
// Placeholder proving the device harness links the compiled-in info-struct TUs
// and runs on the GPU. Always passes.
TEST(internal_infostructs, harness_smoke)
{
    SUCCEED();
}

// ===========================================================================
// rocsparse::numeric_boost
// ===========================================================================
// numeric_boost default construction: boost disabled, all pointers null, tol
// datatype defaults to f32_r and both pointer modes to the sentinel -1.
TEST(internal_infostructs, numeric_boost_default_ctor)
{
    rocsparse::numeric_boost b;
    EXPECT_EQ(b.get_enable(), 0);
    EXPECT_EQ(b.get_tol(), nullptr);
    EXPECT_EQ(b.get_val(), nullptr);
    EXPECT_EQ(b.get_tol_datatype(), rocsparse_datatype_f32_r);
    EXPECT_EQ(b.get_tol_pointer_mode(), static_cast<rocsparse_pointer_mode>(-1));
    EXPECT_EQ(b.get_val_pointer_mode(), static_cast<rocsparse_pointer_mode>(-1));
}

// numeric_boost::define stores enable flag, tol/val pointer modes, tol datatype,
// and the tol/val pointers verbatim; each getter reads back the defined value.
TEST(internal_infostructs, numeric_boost_define)
{
    rocsparse::numeric_boost b;

    const double tol = 1.0e-8;
    const double val = 2.5;

    b.define(1,
             rocsparse_pointer_mode_host,
             rocsparse_datatype_f64_r,
             &tol,
             rocsparse_pointer_mode_device,
             &val);

    EXPECT_EQ(b.get_enable(), 1);
    EXPECT_EQ(b.get_tol_pointer_mode(), rocsparse_pointer_mode_host);
    EXPECT_EQ(b.get_tol_datatype(), rocsparse_datatype_f64_r);
    EXPECT_EQ(b.get_tol(), &tol);
    EXPECT_EQ(b.get_val_pointer_mode(), rocsparse_pointer_mode_device);
    EXPECT_EQ(b.get_val(), &val);
}

// numeric_boost individual setters each update exactly their field; every getter
// reflects the value just set (enable, tol/val ptrs, tol datatype, pointer modes).
TEST(internal_infostructs, numeric_boost_setters)
{
    rocsparse::numeric_boost b;

    const float tol = 3.0f;
    const float val = 4.0f;

    b.set_enable(7);
    b.set_tol(&tol);
    b.set_val(&val);
    b.set_tol_datatype(rocsparse_datatype_f32_c);
    b.set_tol_pointer_mode(rocsparse_pointer_mode_device);
    b.set_val_pointer_mode(rocsparse_pointer_mode_host);

    EXPECT_EQ(b.get_enable(), 7);
    EXPECT_EQ(b.get_tol(), &tol);
    EXPECT_EQ(b.get_val(), &val);
    EXPECT_EQ(b.get_tol_datatype(), rocsparse_datatype_f32_c);
    EXPECT_EQ(b.get_tol_pointer_mode(), rocsparse_pointer_mode_device);
    EXPECT_EQ(b.get_val_pointer_mode(), rocsparse_pointer_mode_host);
}

// numeric_boost::copy performs a field-by-field copy: every getter on the
// destination equals the corresponding getter on the source after copy.
TEST(internal_infostructs, numeric_boost_copy)
{
    const double tol = 1.0e-6;
    const double val = 9.0;

    rocsparse::numeric_boost src;
    src.define(1,
               rocsparse_pointer_mode_host,
               rocsparse_datatype_f64_r,
               &tol,
               rocsparse_pointer_mode_device,
               &val);

    rocsparse::numeric_boost dst;
    dst.copy(src);

    EXPECT_EQ(dst.get_enable(), src.get_enable());
    EXPECT_EQ(dst.get_tol_pointer_mode(), src.get_tol_pointer_mode());
    EXPECT_EQ(dst.get_val_pointer_mode(), src.get_val_pointer_mode());
    EXPECT_EQ(dst.get_tol_datatype(), src.get_tol_datatype());
    EXPECT_EQ(dst.get_tol(), src.get_tol());
    EXPECT_EQ(dst.get_val(), src.get_val());
}

// ===========================================================================
// _rocsparse_csrmv_info::clear (+ adaptive/lrb/nnzsplit clear)
// ===========================================================================
// _rocsparse_csrmv_info::clear resets every scalar bookkeeping field to its zero
// default, nulls the CSR pointers, and clears the adaptive/lrb/nnzsplit sub-info
// sizes (freeing only null device pointers here, which is safe).
TEST(internal_infostructs, csrmv_info_clear_resets_fields)
{
    _rocsparse_csrmv_info info;

    // Dirty the scalar bookkeeping fields.
    info.trans    = rocsparse_operation_transpose;
    info.m        = 7;
    info.n        = 8;
    info.nnz      = 42;
    info.max_rows = 3;

    // Dirty the sub-info sizes (device pointers stay null -> clear() only frees
    // nullptrs, which is safe).
    info.adaptive.size = 9;
    info.lrb.size      = 10;
    info.nnzsplit.size = 11;

    info.clear();

    EXPECT_EQ(info.trans, rocsparse_operation_none);
    EXPECT_EQ(info.m, 0);
    EXPECT_EQ(info.n, 0);
    EXPECT_EQ(info.nnz, 0);
    EXPECT_EQ(info.max_rows, 0);
    EXPECT_EQ(info.csr_row_ptr, nullptr);
    EXPECT_EQ(info.csr_col_ind, nullptr);

    EXPECT_EQ(info.adaptive.size, 0u);
    EXPECT_EQ(info.adaptive.row_blocks, nullptr);
    EXPECT_EQ(info.lrb.size, 0u);
    EXPECT_EQ(info.nnzsplit.size, 0u);
}

// adaptive/lrb/nnzsplit sub-info clear() on freshly constructed (null-pointer)
// structs is null-safe and leaves size == 0 with every device pointer null.
TEST(internal_infostructs, csrmv_subinfo_clear_null_safe)
{
    _rocsparse_adaptive_info adaptive;
    _rocsparse_lrb_info      lrb;
    _rocsparse_nnzsplit_info nnzsplit;

    adaptive.clear();
    lrb.clear();
    nnzsplit.clear();

    EXPECT_EQ(adaptive.size, 0u);
    EXPECT_EQ(adaptive.row_blocks, nullptr);
    EXPECT_EQ(adaptive.wg_flags, nullptr);
    EXPECT_EQ(adaptive.wg_ids, nullptr);

    EXPECT_EQ(lrb.size, 0u);
    EXPECT_EQ(lrb.wg_flags, nullptr);
    EXPECT_EQ(lrb.rows_bins, nullptr);

    EXPECT_EQ(nnzsplit.size, 0u);
    EXPECT_EQ(nnzsplit.starting_ids, nullptr);
    EXPECT_EQ(nnzsplit.starting_block_ids, nullptr);
}

// ===========================================================================
// nnzsplit csrmv analysis buffer release
// ===========================================================================
namespace
{
    struct SpmatGuard
    {
        rocsparse_spmat_descr mat = nullptr;

        SpmatGuard() = default;
        ~SpmatGuard()
        {
            if(mat != nullptr)
            {
                rocsparse_destroy_spmat_descr(mat);
            }
        }
        SpmatGuard(const SpmatGuard&) = delete;
        SpmatGuard& operator=(const SpmatGuard&) = delete;
    };

    struct DnvecGuard
    {
        rocsparse_dnvec_descr vec = nullptr;

        DnvecGuard() = default;
        ~DnvecGuard()
        {
            if(vec != nullptr)
            {
                rocsparse_destroy_dnvec_descr(vec);
            }
        }
        DnvecGuard(const DnvecGuard&) = delete;
        DnvecGuard& operator=(const DnvecGuard&) = delete;
    };
}

// Preprocessing through rocsparse_spmv with rocsparse_spmv_alg_csr_nnzsplit
// allocates nnzsplit.starting_ids with hipMallocAsync; destroying the matrix
// descriptor must return it to the device memory pool (rocm-libraries#6595).
class CsrmvNnzsplitInfo : public HandleTest
{
};

TEST_F(CsrmvNnzsplitInfo, destroy_spmat_descr_returns_analysis_buffer)
{
    int device = 0;
    ASSERT_EQ(hipGetDevice(&device), hipSuccess);
    int pools_supported = 0;
    ASSERT_EQ(
        hipDeviceGetAttribute(&pools_supported, hipDeviceAttributeMemoryPoolsSupported, device),
        hipSuccess);
    if(!pools_supported)
    {
        GTEST_SKIP() << "device has no stream-ordered memory pool";
    }
    hipMemPool_t pool = nullptr;
    ASSERT_EQ(hipDeviceGetMemPool(&pool, device), hipSuccess);

    const auto pool_used_bytes = [pool]() {
        uint64_t used = 0;
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(hipMemPoolGetAttribute(pool, hipMemPoolAttrUsedMemCurrent, &used), hipSuccess);
        return used;
    };

    const std::vector<int>   h_row_ptr = {0, 2, 3, 5, 6};
    const std::vector<int>   h_col_ind = {0, 1, 1, 0, 2, 3};
    const std::vector<float> h_val     = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    device_vector<int>       d_row_ptr(h_row_ptr);
    device_vector<int>       d_col_ind(h_col_ind);
    device_vector<float>     d_val(h_val);
    device_vector<float>     d_x(std::vector<float>(4, 1.f));
    device_vector<float>     d_y(std::vector<float>(4, 0.f));
    ASSERT_TRUE(d_row_ptr.ptr && d_col_ind.ptr && d_val.ptr && d_x.ptr && d_y.ptr);

    DnvecGuard x, y;
    ASSERT_EQ(rocsparse_create_dnvec_descr(&x.vec, 4, d_x.ptr, rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnvec_descr(&y.vec, 4, d_y.ptr, rocsparse_datatype_f32_r),
              rocsparse_status_success);

    // Creates the matrix, preprocesses it twice and destroys it; stores the
    // pool usage after the first preprocess.
    const auto analyse_and_destroy = [&](uint64_t* after_preprocess) {
        SpmatGuard mat;
        ASSERT_EQ(rocsparse_create_csr_descr(&mat.mat,
                                             4,
                                             4,
                                             6,
                                             d_row_ptr.ptr,
                                             d_col_ind.ptr,
                                             d_val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        const float alpha       = 1.f;
        const float beta        = 0.f;
        size_t      buffer_size = 0;
        const auto  spmv        = [&](rocsparse_spmv_stage stage, void* buffer) {
            return rocsparse_spmv(handle,
                                  rocsparse_operation_none,
                                  &alpha,
                                  mat.mat,
                                  x.vec,
                                  &beta,
                                  y.vec,
                                  rocsparse_datatype_f32_r,
                                  rocsparse_spmv_alg_csr_nnzsplit,
                                  stage,
                                  &buffer_size,
                                  buffer);
        };
        ASSERT_EQ(spmv(rocsparse_spmv_stage_buffer_size, nullptr), rocsparse_status_success);
        device_vector<char> buffer(buffer_size > 0 ? buffer_size : size_t(1));
        ASSERT_NE(buffer.ptr, nullptr);

        ASSERT_EQ(spmv(rocsparse_spmv_stage_preprocess, buffer.ptr), rocsparse_status_success);
        *after_preprocess = pool_used_bytes();

        // A second preprocess reuses the cached analysis and must not allocate again.
        ASSERT_EQ(spmv(rocsparse_spmv_stage_preprocess, buffer.ptr), rocsparse_status_success);
        EXPECT_EQ(pool_used_bytes(), *after_preprocess);

        ASSERT_EQ(rocsparse_destroy_spmat_descr(mat.mat), rocsparse_status_success);
        mat.mat = nullptr;
    };

    // The first cycle absorbs any one-time pool growth outside the matrix.
    uint64_t warmup = 0;
    ASSERT_NO_FATAL_FAILURE(analyse_and_destroy(&warmup));

    const uint64_t before   = pool_used_bytes();
    uint64_t       analysed = 0;
    ASSERT_NO_FATAL_FAILURE(analyse_and_destroy(&analysed));
    const uint64_t after = pool_used_bytes();
    EXPECT_GT(analysed, before);
    EXPECT_EQ(after, before);
}

// ===========================================================================
// rocsparse csrgemm_info create / copy / destroy
// ===========================================================================
// create_csrgemm_info allocates an info with documented defaults: buffer_size 0,
// not initialized, and both mul and add enabled; destroy then frees it.
TEST(internal_infostructs, csrgemm_info_create_defaults)
{
    rocsparse_csrgemm_info info = nullptr;
    ASSERT_EQ(rocsparse::create_csrgemm_info(&info), rocsparse_status_success);
    ASSERT_NE(info, nullptr);

    EXPECT_EQ(info->buffer_size, 0u);
    EXPECT_FALSE(info->is_initialized);
    EXPECT_TRUE(info->mul);
    EXPECT_TRUE(info->add);

    EXPECT_EQ(rocsparse::destroy_csrgemm_info(info), rocsparse_status_success);
}

// create_csrgemm_info with a null out-pointer is rejected with invalid_pointer.
TEST(internal_infostructs, csrgemm_info_create_invalid_pointer)
{
    EXPECT_EQ(rocsparse::create_csrgemm_info(nullptr), rocsparse_status_invalid_pointer);
}

// copy_csrgemm_info copies all four fields (buffer_size, is_initialized, mul,
// add) from src to dst; the destination reads back exactly the source values.
TEST(internal_infostructs, csrgemm_info_copy)
{
    rocsparse_csrgemm_info src = nullptr;
    rocsparse_csrgemm_info dst = nullptr;
    ASSERT_EQ(rocsparse::create_csrgemm_info(&src), rocsparse_status_success);
    ASSERT_EQ(rocsparse::create_csrgemm_info(&dst), rocsparse_status_success);

    src->buffer_size    = 4096;
    src->is_initialized = true;
    src->mul            = false;
    src->add            = false;

    ASSERT_EQ(rocsparse::copy_csrgemm_info(dst, src), rocsparse_status_success);

    EXPECT_EQ(dst->buffer_size, 4096u);
    EXPECT_TRUE(dst->is_initialized);
    EXPECT_FALSE(dst->mul);
    EXPECT_FALSE(dst->add);

    EXPECT_EQ(rocsparse::destroy_csrgemm_info(src), rocsparse_status_success);
    EXPECT_EQ(rocsparse::destroy_csrgemm_info(dst), rocsparse_status_success);
}

// copy_csrgemm_info invalid-argument coverage: null dst, null src, and aliasing
// (dst == src) are each rejected with invalid_pointer.
TEST(internal_infostructs, csrgemm_info_copy_invalid_pointer)
{
    rocsparse_csrgemm_info info = nullptr;
    ASSERT_EQ(rocsparse::create_csrgemm_info(&info), rocsparse_status_success);

    EXPECT_EQ(rocsparse::copy_csrgemm_info(nullptr, info), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse::copy_csrgemm_info(info, nullptr), rocsparse_status_invalid_pointer);
    // dest == src is rejected.
    EXPECT_EQ(rocsparse::copy_csrgemm_info(info, info), rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse::destroy_csrgemm_info(info), rocsparse_status_success);
}

// destroy_csrgemm_info(nullptr) is a documented no-op that returns success.
TEST(internal_infostructs, csrgemm_info_destroy_null_is_success)
{
    EXPECT_EQ(rocsparse::destroy_csrgemm_info(nullptr), rocsparse_status_success);
}

// ===========================================================================
// _rocsparse_mat_info lifecycle via the public C API.
//
// The C++ member routing of _rocsparse_mat_info (get_boost, get/set_csrmv_info,
// get_*_info, clear_*_info) is hidden in the library and its definition drags in
// the whole trm_t / csritsv / bsrmv / sorted_coo2csr dependency graph, so it is
// exercised here through the exported create/destroy C API (which runs the real
// _rocsparse_mat_info constructor and destructor-routing) rather than by
// compiling in common/rocsparse_mat_info.cpp. See the report / findings note.
// ===========================================================================
// _rocsparse_mat_info create/destroy round-trip through the public C API:
// create yields a non-null handle and destroy returns success.
TEST(internal_infostructs, mat_info_create_destroy_roundtrip)
{
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
}

// rocsparse_destroy_mat_info(nullptr) is a documented no-op that returns success.
TEST(internal_infostructs, mat_info_destroy_null_is_success)
{
    EXPECT_EQ(rocsparse_destroy_mat_info(nullptr), rocsparse_status_success);
}

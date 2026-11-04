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
// Host-branch unit tests for library/src/auxiliary/rocsparse_auxiliary.cpp
// (compiled into rocsparse-unit-test-device) that unit_test_host_auxiliary.cpp
// leaves untouched:
//
//   - enum validators for values that are rejected,
//   - rocsparse_copy_hyb_mat for every value datatype,
//   - rocsparse_copy_mat_info with bsrmv / coomv / csrgemm info attached,
//   - destroy of a descriptor whose `init` flag is false,
//   - rocsparse_dnvec_get_strided_batch,
//   - rocsparse_destroy_spgeam_descr owning a rocprim buffer,
//   - argument checks run with argument debugging both disabled and enabled.
//     Each ROCSPARSE_CHECKARG* expansion branches on
//     rocsparse_debug_variables.get_debug_arguments() inside its failure path,
//     so a rejected argument only covers the check fully in both modes.
//
#include "unit_test_host_utils.hpp"

#include "rocsparse_csrgemm_info.hpp"
#include "rocsparse_dnvec_descr.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_hyb_mat.hpp"
#include "rocsparse_indextype_utils.hpp"
#include "rocsparse_mat_info.hpp"
#include "rocsparse_spgeam_descr.hpp"
#include "rocsparse_spmat_descr.hpp"
#include "rocsparse_spvec_descr.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <initializer_list>
#include <tuple>
#include <utility>

using namespace rocsparse_ut;

namespace
{
    // Descriptors below only store these addresses; nothing dereferences them.
    int64_t host_storage[64];
    void*   any_ptr = host_storage;

    // Runs fn with argument debugging disabled, then enabled (verbose logging
    // kept off), and restores the previous state.
    template <typename F>
    void in_both_debug_modes(F&& fn)
    {
        const debug_state_guard restore;

        rocsparse_disable_debug_arguments();
        fn();
        rocsparse_enable_debug_arguments();
        rocsparse_disable_debug_arguments_verbose();
        fn();
    }

    using Spmat  = Owned<rocsparse_spmat_descr, rocsparse_destroy_spmat_descr>;
    using Spvec  = Owned<rocsparse_spvec_descr, rocsparse_destroy_spvec_descr>;
    using Dnvec  = Owned<rocsparse_dnvec_descr, rocsparse_destroy_dnvec_descr>;
    using Dnmat  = Owned<rocsparse_dnmat_descr, rocsparse_destroy_dnmat_descr>;
    using Spgeam = Owned<rocsparse_spgeam_descr, rocsparse_destroy_spgeam_descr>;

    void create_csr(Spmat& A)
    {
        ASSERT_EQ(rocsparse_create_csr_descr(&A.get,
                                             4,
                                             4,
                                             4,
                                             any_ptr,
                                             any_ptr,
                                             any_ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
    }

    void create_spvec(Spvec& x)
    {
        ASSERT_EQ(rocsparse_create_spvec_descr(&x.get,
                                               8,
                                               2,
                                               any_ptr,
                                               any_ptr,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f32_r),
                  rocsparse_status_success);
    }

    void create_dnvec(Dnvec& v)
    {
        ASSERT_EQ(rocsparse_create_dnvec_descr(&v.get, 8, any_ptr, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
    }

    // Value datatypes every rocsparse_datatype enumerator is checked against.
    const std::initializer_list<rocsparse_datatype> all_datatypes = {rocsparse_datatype_f16_r,
                                                                     rocsparse_datatype_bf16_r,
                                                                     rocsparse_datatype_f32_r,
                                                                     rocsparse_datatype_f64_r,
                                                                     rocsparse_datatype_f32_c,
                                                                     rocsparse_datatype_f64_c,
                                                                     rocsparse_datatype_i8_r,
                                                                     rocsparse_datatype_u8_r,
                                                                     rocsparse_datatype_i32_r,
                                                                     rocsparse_datatype_u32_r};

    // Every enumerator in `valid` is accepted and 99 is rejected.
    template <typename E>
    void expect_enum_validity(std::initializer_list<E> valid)
    {
        for(const E e : valid)
        {
            EXPECT_FALSE(rocsparse::enum_utils::is_invalid(e)) << e;
        }
        EXPECT_TRUE(rocsparse::enum_utils::is_invalid(static_cast<E>(99)));
    }

    // Calls `get` once per output argument with that argument alone replaced
    // by nullptr, and expects invalid_pointer each time.
    template <size_t... I, typename F, typename... P>
    void expect_null_outputs(std::index_sequence<I...>, F get, std::tuple<P...> outputs)
    {
        const auto check = [&](auto index) {
            auto args             = outputs;
            std::get<index>(args) = nullptr;
            EXPECT_EQ(std::apply(get, args), rocsparse_status_invalid_pointer) << index;
        };
        (check(std::integral_constant<size_t, I>{}), ...);
    }

    template <typename F, typename... P>
    void expect_null_outputs(F get, std::tuple<P...> outputs)
    {
        expect_null_outputs(std::index_sequence_for<P...>{}, get, outputs);
    }

    // The header documents invalid_size for a bad batch count or stride, but
    // the library returns invalid_value. Either is accepted (AISPARSE-904).
    bool is_invalid_size_or_value(rocsparse_status status)
    {
        return status == rocsparse_status_invalid_size || status == rocsparse_status_invalid_value;
    }

    class HostAuxiliaryBranchesHandle : public HandleTest
    {
    };
}

TEST(HostAuxiliaryBranches, enum_validators_reject_out_of_range)
{
    using eu = rocsparse::enum_utils;

    EXPECT_FALSE(eu::is_invalid(rocsparse_spmat_fill_mode));
    EXPECT_FALSE(eu::is_invalid(rocsparse_spmat_storage_mode));
    EXPECT_TRUE(eu::is_invalid(static_cast<rocsparse_spmat_attribute>(99)));

    expect_enum_validity<rocsparse_pointer_mode>(
        {rocsparse_pointer_mode_host, rocsparse_pointer_mode_device});
    expect_enum_validity<rocsparse_diag_type>(
        {rocsparse_diag_type_unit, rocsparse_diag_type_non_unit});
    expect_enum_validity<rocsparse_solve_policy>({rocsparse_solve_policy_auto});
    expect_enum_validity<rocsparse_indextype>(
        {deprecated_rocsparse_indextype_u16, rocsparse_indextype_i32, rocsparse_indextype_i64});
    expect_enum_validity<rocsparse_datatype>(all_datatypes);
}

TEST(HostAuxiliaryBranches, copy_hyb_mat_every_datatype)
{
    for(const auto dt : all_datatypes)
    {
        Owned<rocsparse_hyb_mat, rocsparse_destroy_hyb_mat> src, dst;
        ASSERT_EQ(rocsparse_create_hyb_mat(&src.get), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_hyb_mat(&dst.get), rocsparse_status_success);
        src.get->m           = 3;
        src.get->n           = 5;
        src.get->data_type_T = dt;

        EXPECT_EQ(rocsparse_copy_hyb_mat(dst.get, src.get), rocsparse_status_success) << dt;
        EXPECT_EQ(dst.get->m, 3);
        EXPECT_EQ(dst.get->n, 5);
        EXPECT_EQ(dst.get->data_type_T, dt);

        // A second copy into the now populated destination must match exactly.
        EXPECT_EQ(rocsparse_copy_hyb_mat(dst.get, src.get), rocsparse_status_success) << dt;
        EXPECT_EQ(dst.get->m, 3);
        EXPECT_EQ(dst.get->n, 5);
        EXPECT_EQ(dst.get->data_type_T, dt);
    }
}

TEST(HostAuxiliaryBranches, copy_mat_info_with_bsrmv_coomv_csrgemm)
{
    Owned<rocsparse_mat_info, rocsparse_destroy_mat_info> src, dst;
    ASSERT_EQ(rocsparse_create_mat_info(&src.get), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_mat_info(&dst.get), rocsparse_status_success);

    src.get->set_bsrmv_info(new _rocsparse_bsrmv_info());
    auto* coomv            = new _rocsparse_coomv_info();
    coomv->max_nnz_per_row = 7;
    src.get->set_coomv_info(coomv);
    ASSERT_EQ(rocsparse::create_csrgemm_info(&src.get->csrgemm_info), rocsparse_status_success);

    // First copy allocates the destination sub-infos, the second reuses them.
    for(int pass = 0; pass < 2; ++pass)
    {
        EXPECT_EQ(rocsparse_copy_mat_info(dst.get, src.get), rocsparse_status_success) << pass;
        ASSERT_NE(dst.get->get_bsrmv_info(), nullptr);
        ASSERT_NE(dst.get->get_coomv_info(), nullptr);
        EXPECT_EQ(dst.get->get_coomv_info()->max_nnz_per_row, 7);
        EXPECT_NE(dst.get->csrgemm_info, nullptr);
    }
}

// A descriptor whose `init` flag is false is not destroyed twice. The RAII
// owners release the descriptors once `init` is true again.
TEST(HostAuxiliaryBranches, destroy_uninitialized_descriptor_is_noop)
{
    Spvec x;
    create_spvec(x);
    ASSERT_NE(x.get, nullptr);
    x.get->init = false;
    EXPECT_EQ(rocsparse_destroy_spvec_descr(x.get), rocsparse_status_success);
    x.get->init = true;

    Spmat A;
    create_csr(A);
    ASSERT_NE(A.get, nullptr);
    A.get->init = false;
    EXPECT_EQ(rocsparse_destroy_spmat_descr(A.get), rocsparse_status_success);
    A.get->init = true;
}

TEST(HostAuxiliaryBranches, dnvec_get_strided_batch)
{
    Dnvec dnvec;
    create_dnvec(dnvec);
    const rocsparse_dnvec_descr v = dnvec.get;
    ASSERT_NE(v, nullptr);
    ASSERT_EQ(rocsparse_dnvec_set_strided_batch(v, 3, 16), rocsparse_status_success);

    rocsparse_int batch_count  = 0;
    int64_t       batch_stride = 0;
    EXPECT_EQ(rocsparse_dnvec_get_strided_batch(v, &batch_count, &batch_stride),
              rocsparse_status_success);
    EXPECT_EQ(batch_count, 3);
    EXPECT_EQ(batch_stride, 16);

    in_both_debug_modes([&] {
        EXPECT_EQ(rocsparse_dnvec_get_strided_batch(nullptr, &batch_count, &batch_stride),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_dnvec_get_strided_batch(v, nullptr, &batch_stride),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_dnvec_get_strided_batch(v, &batch_count, nullptr),
                  rocsparse_status_invalid_pointer);
        v->init = false;
        EXPECT_NE(rocsparse_dnvec_get_strided_batch(v, &batch_count, &batch_stride),
                  rocsparse_status_success);
        v->init = true;
    });
}

TEST(HostAuxiliaryBranches, destroy_spgeam_descr_frees_owned_buffers)
{
    // The descriptor owns csr_row_ptr_C and a rocprim buffer it allocated.
    // Once they are attached, the owner's destroy releases them on any exit.
    {
        Spgeam owned;
        ASSERT_EQ(rocsparse_create_spgeam_descr(&owned.get), rocsparse_status_success);
        UT_CHECK_HIP(hipMalloc(&owned.get->csr_row_ptr_C, 64));
        UT_CHECK_HIP(hipMalloc(&owned.get->rocprim_buffer, 64));
        owned.get->rocprim_size  = 64;
        owned.get->rocprim_alloc = true;
        EXPECT_EQ(rocsparse_destroy_spgeam_descr(std::exchange(owned.get, nullptr)),
                  rocsparse_status_success);
    }

    // A borrowed rocprim buffer is not freed by the descriptor: it is still
    // live device memory afterwards.
    device_vector<char> borrowed{size_t(64)};
    ASSERT_NE(borrowed.ptr, nullptr);
    {
        Spgeam d;
        ASSERT_EQ(rocsparse_create_spgeam_descr(&d.get), rocsparse_status_success);
        d.get->rocprim_buffer = borrowed.ptr;
        d.get->rocprim_alloc  = false;
        EXPECT_EQ(rocsparse_destroy_spgeam_descr(std::exchange(d.get, nullptr)),
                  rocsparse_status_success);
    }
    hipPointerAttribute_t attr{};
    ASSERT_EQ(hipPointerGetAttributes(&attr, borrowed.ptr), hipSuccess);
    EXPECT_EQ(attr.type, hipMemoryTypeDevice);
}

TEST(HostAuxiliaryBranches, spmat_null_and_uninitialized_both_debug_modes)
{
    Spmat A_owner;
    create_csr(A_owner);
    const rocsparse_spmat_descr A = A_owner.get;
    ASSERT_NE(A, nullptr);

    int64_t              i64 = 0;
    rocsparse_int        bc  = 0;
    void*                vp  = nullptr;
    rocsparse_indextype  it{};
    rocsparse_index_base ib{};
    rocsparse_datatype   dt{};
    rocsparse_format     fmt{};

    const auto spmat_calls = [&](rocsparse_spmat_descr d, rocsparse_status expected) {
        EXPECT_EQ(rocsparse_spmat_set_values(d, any_ptr), expected);
        EXPECT_EQ(rocsparse_spmat_get_values(d, &vp), expected);
        EXPECT_EQ(rocsparse_spmat_get_nnz(d, &i64), expected);
        EXPECT_EQ(rocsparse_spmat_set_nnz(d, 4), expected);
        EXPECT_EQ(rocsparse_spmat_get_size(d, &i64, &i64, &i64), expected);
        EXPECT_EQ(rocsparse_spmat_get_format(d, &fmt), expected);
        EXPECT_EQ(rocsparse_spmat_get_strided_batch(d, &bc), expected);
        EXPECT_EQ(rocsparse_spmat_set_strided_batch(d, 1), expected);
        EXPECT_EQ(rocsparse_coo_set_strided_batch(d, 1, 0), expected);
        EXPECT_EQ(rocsparse_csr_set_strided_batch(d, 1, 0, 0), expected);
        EXPECT_EQ(rocsparse_csc_set_strided_batch(d, 1, 0, 0), expected);
        EXPECT_EQ(rocsparse_ell_set_strided_batch(d, 1, 0), expected);
        EXPECT_EQ(rocsparse_coo_set_pointers(d, any_ptr, any_ptr, any_ptr), expected);
        EXPECT_EQ(rocsparse_csr_set_pointers(d, any_ptr, any_ptr, any_ptr), expected);
        EXPECT_EQ(rocsparse_csc_set_pointers(d, any_ptr, any_ptr, any_ptr), expected);
        EXPECT_EQ(rocsparse_bsr_set_pointers(d, any_ptr, any_ptr, any_ptr), expected);
        EXPECT_EQ(rocsparse_csr_get(d, &i64, &i64, &i64, &vp, &vp, &vp, &it, &it, &ib, &dt),
                  expected);
        EXPECT_EQ(rocsparse_coo_get(d, &i64, &i64, &i64, &vp, &vp, &vp, &it, &ib, &dt), expected);
        EXPECT_EQ(rocsparse_coo_aos_get(d, &i64, &i64, &i64, &vp, &vp, &it, &ib, &dt), expected);
        EXPECT_EQ(rocsparse_ell_get(d, &i64, &i64, &vp, &vp, &i64, &it, &ib, &dt), expected);
        EXPECT_EQ(
            rocsparse_sell_get(d, &i64, &i64, &i64, &i64, &i64, &vp, &vp, &vp, &it, &it, &ib, &dt),
            expected);
    };

    in_both_debug_modes([&] {
        spmat_calls(nullptr, rocsparse_status_invalid_pointer);
        // A descriptor whose `init` flag is false is internal state: the headers do
        // not document not_initialized for these calls, the library returns it.
        A->init = false;
        spmat_calls(A, rocsparse_status_not_initialized);
        A->init = true;

        EXPECT_EQ(rocsparse_spmat_set_values(A, nullptr), rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_spmat_get_nnz(A, nullptr), rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_spmat_set_nnz(A, -1), rocsparse_status_invalid_size);
        EXPECT_EQ(rocsparse_csr_set_pointers(A, nullptr, any_ptr, any_ptr),
                  rocsparse_status_invalid_pointer);

        EXPECT_PRED1(is_invalid_size_or_value, rocsparse_spmat_set_strided_batch(A, 0));
        EXPECT_PRED1(is_invalid_size_or_value, rocsparse_coo_set_strided_batch(A, 0, 0));
        EXPECT_PRED1(is_invalid_size_or_value, rocsparse_coo_set_strided_batch(A, 1, -1));
        EXPECT_PRED1(is_invalid_size_or_value, rocsparse_csr_set_strided_batch(A, 0, 0, 0));
        EXPECT_PRED1(is_invalid_size_or_value, rocsparse_csr_set_strided_batch(A, 1, -1, 0));
        EXPECT_PRED1(is_invalid_size_or_value, rocsparse_csr_set_strided_batch(A, 1, 0, -1));
        EXPECT_PRED1(is_invalid_size_or_value, rocsparse_csc_set_strided_batch(A, 0, 0, 0));
        EXPECT_PRED1(is_invalid_size_or_value, rocsparse_csc_set_strided_batch(A, 1, -1, 0));
        EXPECT_PRED1(is_invalid_size_or_value, rocsparse_csc_set_strided_batch(A, 1, 0, -1));
        EXPECT_PRED1(is_invalid_size_or_value, rocsparse_ell_set_strided_batch(A, 0, 0));
        EXPECT_PRED1(is_invalid_size_or_value, rocsparse_ell_set_strided_batch(A, 1, -1));

        // Both attribute accessors take the same arguments.
        rocsparse_fill_mode uplo            = rocsparse_fill_mode_lower;
        const auto          check_attribute = [&](auto accessor) {
            constexpr auto fill_mode = rocsparse_spmat_fill_mode;
            EXPECT_EQ(accessor(nullptr, fill_mode, &uplo, sizeof(uplo)),
                      rocsparse_status_invalid_pointer);
            EXPECT_EQ(accessor(A, static_cast<rocsparse_spmat_attribute>(99), &uplo, sizeof(uplo)),
                      rocsparse_status_invalid_value);
            EXPECT_EQ(accessor(A, fill_mode, nullptr, sizeof(uplo)),
                      rocsparse_status_invalid_pointer);
            EXPECT_EQ(accessor(A, fill_mode, &uplo, 1), rocsparse_status_invalid_size);
        };
        check_attribute(rocsparse_spmat_get_attribute);
        check_attribute(rocsparse_spmat_set_attribute);
    });
}

TEST(HostAuxiliaryBranches, getters_null_outputs_both_debug_modes)
{
    // Distinct addresses, so a getter returning the wrong pointer is caught.
    void* const ptr0 = host_storage;
    void* const ptr1 = host_storage + 8;
    void* const ptr2 = host_storage + 16;
    Spmat       coo_aos, ell, sell;
    ASSERT_EQ(rocsparse_create_coo_aos_descr(&coo_aos.get,
                                             4,
                                             4,
                                             2,
                                             ptr0,
                                             ptr1,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_ell_descr(&ell.get,
                                         4,
                                         4,
                                         ptr0,
                                         ptr1,
                                         2,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_sell_descr(&sell.get,
                                          4,
                                          4,
                                          4,
                                          2,
                                          8,
                                          ptr0,
                                          ptr1,
                                          ptr2,
                                          rocsparse_indextype_i32,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              r = 0, c = 0, n = 0, w = 0, ss = 0, cs = 0;
    void*                a = nullptr;
    void*                b = nullptr;
    void*                v = nullptr;
    rocsparse_indextype  t1{}, t2{};
    rocsparse_index_base ib{};
    rocsparse_datatype   dt{};

    in_both_debug_modes([&] {
        expect_null_outputs([&](auto... out) { return rocsparse_coo_aos_get(coo_aos.get, out...); },
                            std::tuple{&r, &c, &n, &a, &v, &t1, &ib, &dt});
        expect_null_outputs([&](auto... out) { return rocsparse_ell_get(ell.get, out...); },
                            std::tuple{&r, &c, &a, &v, &w, &t1, &ib, &dt});
        expect_null_outputs([&](auto... out) { return rocsparse_sell_get(sell.get, out...); },
                            std::tuple{&r, &c, &n, &ss, &cs, &a, &b, &v, &t1, &t2, &ib, &dt});
    });

    // Successful reads cover the non-failing side of every check above. Each
    // read starts from values no getter returns for these descriptors.
    const auto reset = [&] {
        r = c = n = w = ss = cs = -1;
        a = b = v = nullptr;
        t1 = t2 = rocsparse_indextype_i64;
        ib      = rocsparse_index_base_one;
        dt      = rocsparse_datatype_f64_r;
    };
    const auto expect_common = [&] {
        EXPECT_EQ(r, 4);
        EXPECT_EQ(c, 4);
        EXPECT_EQ(t1, rocsparse_indextype_i32);
        EXPECT_EQ(ib, rocsparse_index_base_zero);
        EXPECT_EQ(dt, rocsparse_datatype_f32_r);
    };

    reset();
    EXPECT_EQ(rocsparse_coo_aos_get(coo_aos.get, &r, &c, &n, &a, &v, &t1, &ib, &dt),
              rocsparse_status_success);
    expect_common();
    EXPECT_EQ(n, 2);
    EXPECT_EQ(a, ptr0);
    EXPECT_EQ(v, ptr1);

    reset();
    EXPECT_EQ(rocsparse_ell_get(ell.get, &r, &c, &a, &v, &w, &t1, &ib, &dt),
              rocsparse_status_success);
    expect_common();
    EXPECT_EQ(w, 2);
    EXPECT_EQ(a, ptr0);
    EXPECT_EQ(v, ptr1);

    reset();
    EXPECT_EQ(rocsparse_sell_get(sell.get, &r, &c, &n, &ss, &cs, &a, &b, &v, &t1, &t2, &ib, &dt),
              rocsparse_status_success);
    expect_common();
    EXPECT_EQ(n, 4);
    EXPECT_EQ(ss, 2);
    EXPECT_EQ(cs, 8);
    EXPECT_EQ(a, ptr0);
    EXPECT_EQ(b, ptr1);
    EXPECT_EQ(v, ptr2);
    EXPECT_EQ(t2, rocsparse_indextype_i32);
}

TEST(HostAuxiliaryBranches, create_descr_bad_args_both_debug_modes)
{
    rocsparse_spmat_descr d   = nullptr;
    constexpr auto        i32 = rocsparse_indextype_i32;
    constexpr auto        b0  = rocsparse_index_base_zero;
    constexpr auto        f32 = rocsparse_datatype_f32_r;
    const auto            bit = static_cast<rocsparse_indextype>(99);
    const auto            bib = static_cast<rocsparse_index_base>(99);
    const auto            bdt = static_cast<rocsparse_datatype>(99);
    constexpr auto        ptr = rocsparse_status_invalid_pointer;
    constexpr auto        sz  = rocsparse_status_invalid_size;
    constexpr auto        val = rocsparse_status_invalid_value;
    void*                 p   = any_ptr;

    in_both_debug_modes([&] {
        // clang-format off
        EXPECT_EQ(rocsparse_create_coo_aos_descr(nullptr, 4, 4, 2, p, p, i32, b0, f32), ptr);
        EXPECT_EQ(rocsparse_create_coo_aos_descr(&d, -1, 4, 2, p, p, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_coo_aos_descr(&d, 4, -1, 2, p, p, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_coo_aos_descr(&d, 4, 4, -1, p, p, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_coo_aos_descr(&d, 2, 2, 5, p, p, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_coo_aos_descr(&d, 4, 4, 2, p, p, bit, b0, f32), val);
        EXPECT_EQ(rocsparse_create_coo_aos_descr(&d, 4, 4, 2, p, p, i32, bib, f32), val);
        EXPECT_EQ(rocsparse_create_coo_aos_descr(&d, 4, 4, 2, p, p, i32, b0, bdt), val);

        EXPECT_EQ(rocsparse_create_ell_descr(nullptr, 4, 4, p, p, 2, i32, b0, f32), ptr);
        EXPECT_EQ(rocsparse_create_ell_descr(&d, -1, 4, p, p, 2, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_ell_descr(&d, 4, -1, p, p, 2, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_ell_descr(&d, 4, 4, p, p, -1, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_ell_descr(&d, 4, 4, p, p, 5, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_ell_descr(&d, 4, 4, p, p, 2, bit, b0, f32), val);
        EXPECT_EQ(rocsparse_create_ell_descr(&d, 4, 4, p, p, 2, i32, bib, f32), val);
        EXPECT_EQ(rocsparse_create_ell_descr(&d, 4, 4, p, p, 2, i32, b0, bdt), val);

        EXPECT_EQ(rocsparse_create_sell_descr(nullptr, 4, 4, 4, 2, 8, p, p, p, i32, i32, b0, f32), ptr);
        EXPECT_EQ(rocsparse_create_sell_descr(&d, -1, 4, 4, 2, 8, p, p, p, i32, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_sell_descr(&d, 4, -1, 4, 2, 8, p, p, p, i32, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_sell_descr(&d, 4, 4, -1, 2, 8, p, p, p, i32, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_sell_descr(&d, 4, 4, 4, -1, 8, p, p, p, i32, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_sell_descr(&d, 4, 4, 4, 0, 8, p, p, p, i32, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_sell_descr(&d, 4, 4, 4, 2, -1, p, p, p, i32, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_sell_descr(&d, 4, 4, 9, 2, 8, p, p, p, i32, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_sell_descr(&d, 4, 4, 4, 5, 8, p, p, p, i32, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_sell_descr(&d, 4, 4, 4, 2, 8, p, p, p, bit, i32, b0, f32), val);
        EXPECT_EQ(rocsparse_create_sell_descr(&d, 4, 4, 4, 2, 8, p, p, p, i32, bit, b0, f32), val);
        EXPECT_EQ(rocsparse_create_sell_descr(&d, 4, 4, 4, 2, 8, p, p, p, i32, i32, bib, f32), val);
        EXPECT_EQ(rocsparse_create_sell_descr(&d, 4, 4, 4, 2, 8, p, p, p, i32, i32, b0, bdt), val);

        const auto row = rocsparse_direction_row;
        const auto bdir = static_cast<rocsparse_direction>(99);
        EXPECT_EQ(rocsparse_create_bsr_descr(nullptr, 2, 2, 2, row, 2, p, p, p, i32, i32, b0, f32), ptr);
        EXPECT_EQ(rocsparse_create_bsr_descr(&d, -1, 2, 2, row, 2, p, p, p, i32, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_bsr_descr(&d, 2, -1, 2, row, 2, p, p, p, i32, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_bsr_descr(&d, 2, 2, -1, row, 2, p, p, p, i32, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_bsr_descr(&d, 2, 2, 5, row, 2, p, p, p, i32, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_bsr_descr(&d, 2, 2, 2, bdir, 2, p, p, p, i32, i32, b0, f32), val);
        EXPECT_EQ(rocsparse_create_bsr_descr(&d, 2, 2, 2, row, -1, p, p, p, i32, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_bsr_descr(&d, 2, 2, 2, row, 0, p, p, p, i32, i32, b0, f32), sz);
        EXPECT_EQ(rocsparse_create_bsr_descr(&d, 2, 2, 2, row, 2, p, p, p, bit, i32, b0, f32), val);
        EXPECT_EQ(rocsparse_create_bsr_descr(&d, 2, 2, 2, row, 2, p, p, p, i32, bit, b0, f32), val);
        EXPECT_EQ(rocsparse_create_bsr_descr(&d, 2, 2, 2, row, 2, p, p, p, i32, i32, bib, f32), val);
        EXPECT_EQ(rocsparse_create_bsr_descr(&d, 2, 2, 2, row, 2, p, p, p, i32, i32, b0, bdt), val);
        // clang-format on
    });
}

TEST(HostAuxiliaryBranches, dense_and_sparse_vector_args_both_debug_modes)
{
    Spvec spvec;
    Dnvec dnvec;
    Dnmat dnmat;
    create_spvec(spvec);
    create_dnvec(dnvec);
    ASSERT_EQ(rocsparse_create_dnmat_descr(
                  &dnmat.get, 4, 4, 4, any_ptr, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);
    const rocsparse_spvec_descr x = spvec.get;
    const rocsparse_dnvec_descr v = dnvec.get;
    const rocsparse_dnmat_descr M = dnmat.get;

    void*          vp  = nullptr;
    const void*    cvp = nullptr;
    constexpr auto ptr = rocsparse_status_invalid_pointer;

    in_both_debug_modes([&] {
        EXPECT_EQ(rocsparse_spvec_set_values(nullptr, any_ptr), ptr);
        EXPECT_EQ(rocsparse_spvec_set_values(x, nullptr), ptr);

        EXPECT_EQ(rocsparse_dnvec_get_values(nullptr, &vp), ptr);
        EXPECT_EQ(rocsparse_dnvec_get_values(v, nullptr), ptr);
        EXPECT_EQ(rocsparse_const_dnvec_get_values(nullptr, &cvp), ptr);
        EXPECT_EQ(rocsparse_const_dnvec_get_values(v, nullptr), ptr);
        EXPECT_EQ(rocsparse_dnvec_set_values(nullptr, any_ptr), ptr);
        EXPECT_EQ(rocsparse_dnvec_set_values(v, nullptr), ptr);
        EXPECT_EQ(rocsparse_dnvec_set_strided_batch(nullptr, 1, 0), ptr);
        EXPECT_PRED1(is_invalid_size_or_value, rocsparse_dnvec_set_strided_batch(v, 0, 0));

        EXPECT_EQ(rocsparse_dnmat_get_values(nullptr, &vp), ptr);
        EXPECT_EQ(rocsparse_dnmat_get_values(M, nullptr), ptr);
        EXPECT_EQ(rocsparse_const_dnmat_get_values(nullptr, &cvp), ptr);
        EXPECT_EQ(rocsparse_const_dnmat_get_values(M, nullptr), ptr);
        EXPECT_EQ(rocsparse_dnmat_set_values(nullptr, any_ptr), ptr);
        EXPECT_EQ(rocsparse_dnmat_set_values(M, nullptr), ptr);
        EXPECT_EQ(rocsparse_dnmat_set_strided_batch(nullptr, 1, 0), ptr);
        EXPECT_PRED1(is_invalid_size_or_value, rocsparse_dnmat_set_strided_batch(M, 0, 0));
        EXPECT_PRED1(is_invalid_size_or_value, rocsparse_dnmat_set_strided_batch(M, 1, -1));

        x->init = false;
        EXPECT_NE(rocsparse_spvec_set_values(x, any_ptr), rocsparse_status_success);
        x->init = true;
        v->init = false;
        EXPECT_NE(rocsparse_dnvec_get_values(v, &vp), rocsparse_status_success);
        EXPECT_NE(rocsparse_const_dnvec_get_values(v, &cvp), rocsparse_status_success);
        EXPECT_NE(rocsparse_dnvec_set_values(v, any_ptr), rocsparse_status_success);
        EXPECT_NE(rocsparse_dnvec_set_strided_batch(v, 1, 0), rocsparse_status_success);
        v->init = true;
    });
}

TEST_F(HostAuxiliaryBranchesHandle, spgeam_input_output_args_both_debug_modes)
{
    Spgeam owner;
    ASSERT_EQ(rocsparse_create_spgeam_descr(&owner.get), rocsparse_status_success);
    const rocsparse_spgeam_descr d = owner.get;

    int64_t        buf[2] = {};
    constexpr auto sz     = rocsparse_status_invalid_size;

    in_both_debug_modes([&] {
        EXPECT_EQ(rocsparse_spgeam_set_input(
                      nullptr, d, rocsparse_spgeam_input_alg, buf, sizeof(buf), nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_spgeam_set_input(
                      handle, nullptr, rocsparse_spgeam_input_alg, buf, sizeof(buf), nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_spgeam_set_input(
                      handle, d, static_cast<rocsparse_spgeam_input>(99), buf, 1, nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(
            rocsparse_spgeam_set_input(handle, d, rocsparse_spgeam_input_alg, nullptr, 1, nullptr),
            rocsparse_status_invalid_pointer);
        for(const auto input : {rocsparse_spgeam_input_alg,
                                rocsparse_spgeam_input_scalar_datatype,
                                rocsparse_spgeam_input_compute_datatype,
                                rocsparse_spgeam_input_operation_A,
                                rocsparse_spgeam_input_operation_B,
                                rocsparse_spgeam_input_scalar_alpha,
                                rocsparse_spgeam_input_scalar_beta})
        {
            EXPECT_EQ(rocsparse_spgeam_set_input(handle, d, input, buf, 1, nullptr), sz) << input;
        }

        EXPECT_EQ(rocsparse_spgeam_get_output(
                      handle, nullptr, rocsparse_spgeam_output_nnz, buf, sizeof(int64_t), nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_spgeam_get_output(
                      handle, d, static_cast<rocsparse_spgeam_output>(99), buf, 1, nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_spgeam_get_output(
                      handle, d, rocsparse_spgeam_output_nnz, nullptr, sizeof(int64_t), nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            rocsparse_spgeam_get_output(handle, d, rocsparse_spgeam_output_nnz, buf, 1, nullptr),
            sz);
    });
}

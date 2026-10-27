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

// Device unit tests for the internal conversion routines that the public API tests do not
// reach directly:
//   convert_array (numeric datatype pairs, index base, strided and same-type shortcuts),
//   dnvec_transfer_from, ell2dense / coo2dense_aos templates (also reached from sddmm
//   alg_dense), the ggthr generic dispatch, csxsldu_compute and gell2csr (ELL to CSR through
//   rocsparse_sparse_to_sparse).

#include "unit_test_device_conv_common.hpp"

#include "rocsparse_convert_array.hpp"
#include "rocsparse_coo2dense_aos.hpp"
#include "rocsparse_csxsldu.hpp"
#include "rocsparse_ell2dense.hpp"
#include "rocsparse_ggthr.hpp"

#include <cstdint>
#include <iostream>
#include <sstream>

// The rocsparse_Xcsxsldu_* entry points are defined (extern "C") in the csxsldu
// translation units compiled into this binary, but declared in no header.
#define CSXSLDU_DECLARE(X, T)                                                            \
    extern "C" rocsparse_status rocsparse_##X##csxsldu_buffer_size(rocsparse_handle,     \
                                                                   rocsparse_direction,  \
                                                                   rocsparse_int,        \
                                                                   rocsparse_int,        \
                                                                   rocsparse_int,        \
                                                                   const rocsparse_int*, \
                                                                   const rocsparse_int*, \
                                                                   const T*,             \
                                                                   rocsparse_index_base, \
                                                                   rocsparse_diag_type,  \
                                                                   rocsparse_direction,  \
                                                                   rocsparse_diag_type,  \
                                                                   rocsparse_direction,  \
                                                                   size_t*);             \
    extern "C" rocsparse_status rocsparse_##X##csxsldu_preprocess(rocsparse_handle,      \
                                                                  rocsparse_direction,   \
                                                                  rocsparse_int,         \
                                                                  rocsparse_int,         \
                                                                  rocsparse_int,         \
                                                                  const rocsparse_int*,  \
                                                                  const rocsparse_int*,  \
                                                                  const T*,              \
                                                                  rocsparse_index_base,  \
                                                                  rocsparse_diag_type,   \
                                                                  rocsparse_direction,   \
                                                                  rocsparse_int*,        \
                                                                  rocsparse_int*,        \
                                                                  rocsparse_index_base,  \
                                                                  rocsparse_diag_type,   \
                                                                  rocsparse_direction,   \
                                                                  rocsparse_int*,        \
                                                                  rocsparse_int*,        \
                                                                  rocsparse_index_base,  \
                                                                  void*);                \
    extern "C" rocsparse_status rocsparse_##X##csxsldu_compute(rocsparse_handle,         \
                                                               rocsparse_direction,      \
                                                               rocsparse_int,            \
                                                               rocsparse_int,            \
                                                               rocsparse_int,            \
                                                               const rocsparse_int*,     \
                                                               const rocsparse_int*,     \
                                                               T*,                       \
                                                               rocsparse_index_base,     \
                                                               rocsparse_diag_type,      \
                                                               rocsparse_direction,      \
                                                               rocsparse_int,            \
                                                               rocsparse_int*,           \
                                                               rocsparse_int*,           \
                                                               T*,                       \
                                                               rocsparse_index_base,     \
                                                               rocsparse_diag_type,      \
                                                               rocsparse_direction,      \
                                                               rocsparse_int,            \
                                                               rocsparse_int*,           \
                                                               rocsparse_int*,           \
                                                               T*,                       \
                                                               rocsparse_index_base,     \
                                                               T*,                       \
                                                               void*)
CSXSLDU_DECLARE(s, float);
CSXSLDU_DECLARE(z, rocsparse_double_complex);
#undef CSXSLDU_DECLARE

using namespace convut;

namespace
{
    using ConvInternal = ConvTest;

    // Redirects std::cout and std::cerr: routines log the pairs that have no kernel.
    class Silence
    {
        std::ostringstream m_out, m_err;
        std::streambuf *   m_old_out, *m_old_err;

    public:
        Silence()
            : m_old_out(std::cout.rdbuf(m_out.rdbuf()))
            , m_old_err(std::cerr.rdbuf(m_err.rdbuf()))
        {
        }
        ~Silence()
        {
            std::cout.rdbuf(m_old_out);
            std::cerr.rdbuf(m_old_err);
        }
    };

    // Values as float for the half precision types, which have no printable equality.
    template <typename T>
    void expect_vec(const std::vector<T>& got, const std::vector<T>& expected)
    {
        if constexpr(std::is_same_v<T, _Float16> || std::is_same_v<T, rocsparse_bfloat16>)
        {
            ASSERT_EQ(got.size(), expected.size());
            for(size_t i = 0; i < got.size(); ++i)
                EXPECT_EQ(static_cast<float>(got[i]), static_cast<float>(expected[i]))
                    << "index " << i;
        }
        else
        {
            EXPECT_EQ(got, expected);
        }
    }

    // ------------------------------------------------------------------
    // convert_array
    // ------------------------------------------------------------------

    constexpr rocsparse_datatype s_datatypes[] = {rocsparse_datatype_i8_r,
                                                  rocsparse_datatype_u8_r,
                                                  rocsparse_datatype_i32_r,
                                                  rocsparse_datatype_u32_r,
                                                  rocsparse_datatype_f16_r,
                                                  rocsparse_datatype_bf16_r,
                                                  rocsparse_datatype_f32_r,
                                                  rocsparse_datatype_f64_r,
                                                  rocsparse_datatype_f32_c,
                                                  rocsparse_datatype_f64_c};

    template <typename D, typename S>
    std::vector<D> convert_values(rocsparse_handle      handle,
                                  rocsparse_datatype    dd,
                                  rocsparse_datatype    ds,
                                  const std::vector<S>& in)
    {
        DevBuf<S> src(in);
        DevBuf<D> dst(in.size());
        EXPECT_EQ(rocsparse::convert_array(handle, in.size(), dd, dst.get(), ds, src.get()), st_ok);
        return dst.read();
    }

    // The values 1..n stored as datatype dt, as bytes; they are exact in every datatype.
    std::vector<char> iota_bytes(rocsparse_datatype dt, int n)
    {
        std::vector<char> b;
        const auto        put = [&](auto x) {
            const char* p = reinterpret_cast<const char*>(&x);
            b.insert(b.end(), p, p + sizeof(x));
        };
        for(int v = 1; v <= n; ++v)
        {
            switch(dt)
            {
            case rocsparse_datatype_i8_r:
                put(int8_t(v));
                break;
            case rocsparse_datatype_u8_r:
                put(uint8_t(v));
                break;
            case rocsparse_datatype_i32_r:
                put(int32_t(v));
                break;
            case rocsparse_datatype_u32_r:
                put(uint32_t(v));
                break;
            case rocsparse_datatype_f16_r:
                put(scalar<_Float16>(v));
                break;
            case rocsparse_datatype_bf16_r:
                put(scalar<rocsparse_bfloat16>(v));
                break;
            case rocsparse_datatype_f32_r:
                put(scalar<float>(v));
                break;
            case rocsparse_datatype_f64_r:
                put(scalar<double>(v));
                break;
            case rocsparse_datatype_f32_c:
                put(scalar<rocsparse_float_complex>(v));
                break;
            case rocsparse_datatype_f64_c:
                put(scalar<rocsparse_double_complex>(v));
                break;
            }
        }
        return b;
    }

    TEST_F(ConvInternal, ConvertArrayDatatypes)
    {
        constexpr size_t n   = 4;
        constexpr auto   f64 = rocsparse_datatype_f64_r;
        constexpr auto   c32 = rocsparse_datatype_f32_c, c64 = rocsparse_datatype_f64_c;
        using cf = rocsparse_float_complex;
        using cd = rocsparse_double_complex;
        Silence                                                        silence;
        std::vector<std::pair<rocsparse_datatype, rocsparse_datatype>> pairs
            = {{f64, f32}, {f32, f64}, {c32, f32}, {c32, f64}, {c32, c64}, {c64, c32}};
        for(auto t : s_datatypes)
            pairs.emplace_back(t, t);
        for(const auto& [t, s] : pairs)
        {
            SCOPED_TRACE(testing::Message() << "target " << t << " source " << s);
            const std::vector<char> want = iota_bytes(t, n);
            DevBuf<char>            src(iota_bytes(s, n)), dst(std::vector<char>(want.size(), -1));
            EXPECT_EQ(rocsparse::convert_array(handle, n, t, dst.get(), s, src.get()), st_ok);
            EXPECT_EQ(rocsparse::convert_array(handle, 0, t, dst.get(), s, src.get()), st_ok);
            EXPECT_EQ(dst.read(), want);
        }
        const std::vector<float> hf{1.0f, -2.0f, 3.5f, 0.25f};
        EXPECT_EQ(convert_values<double>(handle, f64, f32, hf),
                  (std::vector<double>{1.0, -2.0, 3.5, 0.25}));
        EXPECT_EQ(convert_values<float>(handle, f32, f64, std::vector<double>{1.25, 3.0}),
                  (std::vector<float>{1.25f, 3.0f}));
        EXPECT_EQ(convert_values<cf>(handle, c32, f32, hf),
                  (std::vector<cf>{{1, 0}, {-2, 0}, {3.5f, 0}, {0.25f, 0}}));
        const std::vector<cf> hc{{1.0f, 2.0f}, {-3.0f, 0.5f}};
        EXPECT_EQ(convert_values<cd>(handle, c64, c32, hc),
                  (std::vector<cd>{{1.0, 2.0}, {-3.0, 0.5}}));
        EXPECT_EQ(convert_values<cf>(handle, c32, c64, std::vector<cd>{{1.0, 2.0}}),
                  (std::vector<cf>{{1.0f, 2.0f}}));
    }

    template <typename D, typename S>
    rocsparse_status convert_index(rocsparse_handle     handle,
                                   int64_t              n,
                                   DevBuf<D>&           dst,
                                   rocsparse_index_base bd,
                                   const DevBuf<S>&     src,
                                   rocsparse_index_base bs)
    {
        return rocsparse::convert_array(
            handle, n, it_of<D>(), dst.get(), bd, it_of<S>(), src.get(), bs);
    }

    template <typename D, typename S>
    rocsparse_status convert_strided(rocsparse_handle handle,
                                     int64_t          n,
                                     DevBuf<D>&       dst,
                                     int64_t          inc_d,
                                     const DevBuf<S>& src,
                                     int64_t          inc_s)
    {
        return rocsparse::convert_array(
            handle, n, it_of<D>(), dst.get(), inc_d, it_of<S>(), src.get(), inc_s);
    }

    TEST_F(ConvInternal, ConvertArrayIndices)
    {
        Silence                    silence;
        const std::vector<int32_t> h{1, 2, 3, 4};
        DevBuf<int32_t>            a(h), b32(4);
        DevBuf<int64_t>            a64(std::vector<int64_t>{0, 1, 2, 3}), b64(4);

        // Same type and base: a copy, and nothing to do in place.
        ASSERT_EQ(convert_index(handle, 4, b32, base0, a, base0), st_ok);
        EXPECT_EQ(b32.read(), h);
        EXPECT_EQ(convert_index(handle, 4, a, base0, a, base0), st_ok);
        EXPECT_EQ(a.read(), h);
        // Unit stride overload without an index base.
        ASSERT_EQ(hipMemset(b32.get(), 0, 16), hipSuccess);
        ASSERT_EQ(rocsparse::convert_array(handle, 4, i32, b32.get(), i32, a.get()), st_ok);
        EXPECT_EQ(b32.read(), h);
        EXPECT_EQ(rocsparse::convert_array(handle, 4, i32, a.get(), i32, a.get()), st_ok);
        ASSERT_EQ(
            rocsparse::convert_array(handle, 4, rocsparse_indextype_i64, b64.get(), i32, a.get()),
            st_ok);
        EXPECT_EQ(b64.read(), (std::vector<int64_t>{1, 2, 3, 4}));
        ASSERT_EQ(
            rocsparse::convert_array(handle, 4, i32, b32.get(), rocsparse_indextype_i64, a64.get()),
            st_ok);
        EXPECT_EQ(b32.read(), (std::vector<int32_t>{0, 1, 2, 3}));

        // Index base changes, for every pair of index types.
        ASSERT_EQ(convert_index(handle, 4, b32, base1, a64, base0), st_ok);
        EXPECT_EQ(b32.read(), h);
        ASSERT_EQ(convert_index(handle, 4, b64, base0, b32, base1), st_ok);
        EXPECT_EQ(b64.read(), a64.read());
        ASSERT_EQ(convert_index(handle, 4, b64, base1, a64, base0), st_ok);
        EXPECT_EQ(b64.read(), (std::vector<int64_t>{1, 2, 3, 4}));
        ASSERT_EQ(convert_index(handle, 4, b32, base1, a, base0), st_ok);
        EXPECT_EQ(b32.read(), (std::vector<int32_t>{2, 3, 4, 5}));
        // Empty arrays return early.
        EXPECT_EQ(convert_index(handle, 0, b32, base1, a64, base0), st_ok);

        // A source stride, and the same type with a non-unit stride.
        DevBuf<int32_t>            s32(std::vector<int32_t>{10, -1, 20, -1, 30, -1, 40, -1});
        const std::vector<int32_t> packed{10, 20, 30, 40};
        ASSERT_EQ(convert_strided(handle, 4, b64, 1, s32, 2), st_ok);
        EXPECT_EQ(b64.read(), (std::vector<int64_t>{10, 20, 30, 40}));
        ASSERT_EQ(convert_strided(handle, 4, b32, 1, s32, 2), st_ok);
        EXPECT_EQ(b32.read(), packed);
        EXPECT_EQ(convert_strided(handle, 0, b32, 1, b64, 1), st_ok);
        ASSERT_EQ(convert_strided(handle, 4, a, 1, b32, 1), st_ok);
        EXPECT_EQ(a.read(), packed);

        // A value that does not fit in the target index type.
        DevBuf<int64_t> big(std::vector<int64_t>{1, int64_t(1) << 40, 2, -(int64_t(1) << 40)});
        EXPECT_REJECTED(convert_strided(handle, 4, b32, 1, big, 1));
        EXPECT_REJECTED(convert_index(handle, 4, b32, base0, big, base0));
    }

    TEST_F(ConvInternal, DnvecTransferFrom)
    {
        Silence                          silence;
        constexpr size_t                 n = 4;
        DevBuf<float>                    d_f32(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
        DevBuf<double>                   d_f64(n);
        DevBuf<rocsparse_float_complex>  d_c32(n);
        DevBuf<rocsparse_double_complex> d_c64(n);
        Dnvec                            vf32, vf64, vc32, vc64, vshort;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vf32, n, d_f32, rocsparse_datatype_f32_r), st_ok);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vf64, n, d_f64, rocsparse_datatype_f64_r), st_ok);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vc32, n, d_c32, rocsparse_datatype_f32_c), st_ok);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vc64, n, d_c64, rocsparse_datatype_f64_c), st_ok);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vshort, n - 1, d_f32, rocsparse_datatype_f32_r),
                  st_ok);

        EXPECT_EQ(rocsparse::dnvec_transfer_from(handle, nullptr, vf32), st_ptr);
        EXPECT_EQ(rocsparse::dnvec_transfer_from(handle, vf64, nullptr), st_ptr);
        EXPECT_EQ(rocsparse::dnvec_transfer_from(handle, vf64, vshort), st_size);
        // A real target from a complex source is rejected.
        EXPECT_REJECTED(rocsparse::dnvec_transfer_from(handle, vf64, vc32));
        EXPECT_REJECTED(rocsparse::dnvec_transfer_from(handle, vf32, vc64));

        // Real from real, complex from real, complex from complex.
        ASSERT_EQ(rocsparse::dnvec_transfer_from(handle, vf64, vf32), st_ok);
        EXPECT_EQ(d_f64.read(), (std::vector<double>{1.0, 2.0, 3.0, 4.0}));
        ASSERT_EQ(rocsparse::dnvec_transfer_from(handle, vc32, vf64), st_ok);
        ASSERT_EQ(rocsparse::dnvec_transfer_from(handle, vc64, vc32), st_ok);
        EXPECT_EQ(d_c64.read(),
                  (std::vector<rocsparse_double_complex>{
                      {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}}));
    }

    // ------------------------------------------------------------------
    // ell2dense and coo2dense_aos
    // ------------------------------------------------------------------

    // Off-diagonal 2 x 2 matrices, so the column-major and row-major dense layouts differ:
    // the entries are (0,1) = 3 and (1,0) = 4. Quick returns do not touch the dense array,
    // and the argument checks.
    struct Dense2Args
    {
        rocsparse_handle    handle;
        rocsparse_mat_descr descr;
        int32_t             m = 2, n = 2, nnz_or_width;
        float*              dense;
        int64_t             ld    = 2;
        rocsparse_order     order = rocsparse_order_column;
    };

    template <typename Run>
    void check_dense2(
        rocsparse_handle handle, rocsparse_mat_descr descr, int32_t extent, float* dense, Run run)
    {
        Dense2Args base{handle, descr, 2, 2, extent, dense};
        const auto with = [&](auto&& mod) {
            Dense2Args c = base;
            mod(c);
            return run(c);
        };
        const auto order_check = [&](rocsparse_order order, const std::vector<float>& expected) {
            ASSERT_EQ(hipMemset(dense, 0x7f, 16), hipSuccess);
            ASSERT_EQ(with([&](auto& c) { c.order = order; }), st_ok);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            EXPECT_EQ(to_host(dense, 4), expected);
        };
        order_check(rocsparse_order_column, {0.0f, 4.0f, 3.0f, 0.0f});
        order_check(rocsparse_order_row, {0.0f, 3.0f, 4.0f, 0.0f});

        EXPECT_EQ(with([](auto& c) {
                      c.m  = 0;
                      c.ld = 1;
                  }),
                  st_ok);
        EXPECT_EQ(with([](auto& c) { c.n = 0; }), st_ok);
        EXPECT_EQ(with([](auto& c) { c.nnz_or_width = 0; }), st_ok);
        EXPECT_EQ(with([](auto& c) { c.handle = nullptr; }), rocsparse_status_invalid_handle);
        EXPECT_EQ(with([](auto& c) { c.descr = nullptr; }), st_ptr);
        EXPECT_EQ(with([](auto& c) { c.dense = nullptr; }), st_ptr);
        EXPECT_EQ(with([](auto& c) { c.m = -1; }), st_size);
        EXPECT_EQ(with([](auto& c) { c.ld = 1; }), st_size);
        EXPECT_EQ(with([](auto& c) { c.order = static_cast<rocsparse_order>(99); }), st_val);
    }

    TEST_F(ConvInternal, Ell2DenseAndCoo2DenseAos)
    {
        MatDescr        descr(base0);
        DevBuf<float>   val(std::vector<float>{3.0f, 4.0f}), dense(4);
        DevBuf<int32_t> col(std::vector<int32_t>{1, 0}), ind(std::vector<int32_t>{0, 1, 1, 0});
        DevBuf<int64_t> col64(std::vector<int64_t>{1, 0}), ind64(std::vector<int64_t>{0, 1, 1, 0});

        check_dense2(handle, descr, 1, dense, [&](const Dense2Args& c) {
            return rocsparse::ell2dense_template<int32_t, float>(
                c.handle, c.m, c.n, c.descr, c.nnz_or_width, val, col, c.dense, c.ld, c.order);
        });
        check_dense2(handle, descr, 2, dense, [&](const Dense2Args& c) {
            return rocsparse::coo2dense_aos_template<int32_t, float>(
                c.handle, c.m, c.n, c.nnz_or_width, c.descr, val, ind, c.dense, c.ld, c.order);
        });
        // 64 bit indices.
        ASSERT_EQ(hipMemset(dense, 0x7f, 16), hipSuccess);
        ASSERT_EQ((rocsparse::ell2dense_template<int64_t, float>(
                      handle, 2, 2, descr, 1, val, col64, dense, 2, rocsparse_order_column)),
                  st_ok);
        EXPECT_EQ(dense.read(), (std::vector<float>{0.0f, 4.0f, 3.0f, 0.0f}));
        ASSERT_EQ(hipMemset(dense, 0x7f, 16), hipSuccess);
        ASSERT_EQ((rocsparse::coo2dense_aos_template<int64_t, float>(
                      handle, 2, 2, 2, descr, val, ind64, dense, 2, rocsparse_order_column)),
                  st_ok);
        EXPECT_EQ(dense.read(), (std::vector<float>{0.0f, 4.0f, 3.0f, 0.0f}));

        // Only general matrices with sorted storage are converted.
        for(auto type : {rocsparse_matrix_type_symmetric, rocsparse_matrix_type_hermitian})
        {
            ASSERT_EQ(rocsparse_set_mat_type(descr, type), st_ok);
            EXPECT_REJECTED((rocsparse::ell2dense_template<int32_t, float>(
                handle, 2, 2, descr, 1, val, col, dense, 2, rocsparse_order_column)));
            EXPECT_REJECTED((rocsparse::coo2dense_aos_template<int32_t, float>(
                handle, 2, 2, 2, descr, val, ind, dense, 2, rocsparse_order_column)));
        }
        ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general), st_ok);
        ASSERT_EQ(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted), st_ok);
        EXPECT_REJECTED((rocsparse::ell2dense_template<int32_t, float>(
            handle, 2, 2, descr, 1, val, col, dense, 2, rocsparse_order_column)));
    }

    // ------------------------------------------------------------------
    // ggthr
    // ------------------------------------------------------------------

    template <typename T, typename J>
    void check_ggthr_perm(rocsparse_handle handle, rocsparse_datatype dt)
    {
        DevBuf<T> in(std::vector<T>{scalar<T>(10), scalar<T>(20), scalar<T>(30), scalar<T>(40)}),
            out(2);
        DevBuf<J> perm(std::vector<J>{2, 0});
        ASSERT_EQ(
            rocsparse::ggthr(handle, 2, dt, in.get(), out.get(), it_of<J>(), perm.get(), base0),
            st_ok);
        expect_vec(out.read(), std::vector<T>{scalar<T>(30), scalar<T>(10)});
    }

    template <typename T>
    void check_ggthr(rocsparse_handle handle, rocsparse_datatype dt)
    {
        check_ggthr_perm<T, int32_t>(handle, dt);
        check_ggthr_perm<T, int64_t>(handle, dt);
    }

    // Every datatype and both permutation types.
    TEST_F(ConvInternal, Ggthr)
    {
        check_ggthr<float>(handle, rocsparse_datatype_f32_r);
        check_ggthr<double>(handle, rocsparse_datatype_f64_r);
        check_ggthr<rocsparse_float_complex>(handle, rocsparse_datatype_f32_c);
        check_ggthr<rocsparse_double_complex>(handle, rocsparse_datatype_f64_c);
        check_ggthr<int8_t>(handle, rocsparse_datatype_i8_r);
        check_ggthr<uint8_t>(handle, rocsparse_datatype_u8_r);
        check_ggthr<int32_t>(handle, rocsparse_datatype_i32_r);
        check_ggthr<uint32_t>(handle, rocsparse_datatype_u32_r);
        check_ggthr<_Float16>(handle, rocsparse_datatype_f16_r);
        check_ggthr<rocsparse_bfloat16>(handle, rocsparse_datatype_bf16_r);

        // nnz == 0 returns early, an invalid datatype is rejected.
        EXPECT_EQ(rocsparse::ggthr(
                      handle, 0, rocsparse_datatype_f32_r, nullptr, nullptr, i32, nullptr, base0),
                  st_ok);
        DevBuf<float>   in(std::vector<float>{1.0f}), out(1);
        DevBuf<int32_t> perm(std::vector<int32_t>{0});
        EXPECT_EQ(rocsparse::ggthr(handle,
                                   1,
                                   static_cast<rocsparse_datatype>(999),
                                   in.get(),
                                   out.get(),
                                   i32,
                                   perm.get(),
                                   base0),
                  st_val);
    }

    // ------------------------------------------------------------------
    // csxsldu_compute
    // ------------------------------------------------------------------

    // One triangular factor of the split: diagonal type, direction, entries, the pointer
    // array before the call, index base.
    struct Factor
    {
        rocsparse_diag_type  diag;
        rocsparse_direction  dir;
        I                    nnz;
        std::vector<I>       ptr;
        rocsparse_index_base base;
    };

    struct SlduCase
    {
        const char*         name;
        rocsparse_direction dir;
        I                   m, n;
        std::vector<I>      ptr, ind;
        std::vector<float>  val;
        Factor              lower, upper;
        std::vector<I>      scratch; // the temporary pointer array of a direction change
        // Expected outputs; an empty vector is not checked.
        std::vector<I>     lptr, lind, uptr, uind;
        std::vector<float> lval, uval, diag;
    };

    TEST_F(ConvInternal, CsxsLduCompute)
    {
        const auto unit = rocsparse_diag_type_unit, nonunit = rocsparse_diag_type_non_unit;
        const auto row = dir_r, col = dir_c;
        // 3 x 3 CSR: | 1 2 0 |, | 0 3 0 |, | 4 0 5 |
        const std::vector<I>     kptr{0, 2, 3, 5}, kind{0, 1, 1, 0, 2};
        const std::vector<float> kval{1, 2, 3, 4, 5};
        const std::vector<I>     none;
        const std::vector<float> fnone;
        // 2 x 3 diagonal: no strict lower or upper entry
        const std::vector<I>     dptr{0, 1, 2}, dind{0, 1};
        const std::vector<float> dval{1, 2};

        const std::vector<SlduCase> cases = {
            // Same direction, both triangles strict, the diagonal is extracted: L = {(2,0) = 4},
            // U = {(0,1) = 2}, diag = {1, 3, 5}.
            {"strict",
             row,
             3,
             3,
             kptr,
             kind,
             kval,
             {unit, row, 1, {0, 0, 0, 1}, base0},
             {unit, row, 1, {0, 1, 1, 1}, base0},
             {},
             none,
             {0},
             none,
             {1},
             {4},
             {2},
             {1, 3, 5}},
            // L non unit, U unit: L stays strict, U absorbs the diagonal.
            {"upper_diag",
             row,
             3,
             3,
             kptr,
             kind,
             kval,
             {nonunit, row, 1, {0, 0, 0, 1}, base0},
             {unit, row, 4, {0, 2, 3, 4}, base0},
             {},
             none,
             {0},
             none,
             {0, 1, 1, 2},
             {4},
             {1, 2, 3, 5},
             fnone},
            // L unit, U non unit: L absorbs the diagonal, U stays strict.
            {"lower_diag",
             row,
             3,
             3,
             kptr,
             kind,
             kval,
             {unit, row, 4, {0, 1, 2, 4}, base0},
             {nonunit, row, 1, {0, 1, 1, 1}, base0},
             {},
             none,
             {0, 1, 0, 2},
             none,
             {1},
             {1, 3, 4, 5},
             {2},
             fnone},
            // Column direction, both unit: the diagonal is extracted.
            {"column",
             col,
             2,
             2,
             {0, 1, 2},
             {0, 1},
             {10, 20},
             {unit, col, 0, {0, 0, 0}, base0},
             {unit, col, 0, {0, 0, 0}, base0},
             {},
             none,
             none,
             none,
             none,
             fnone,
             fnone,
             {10, 20}},
            // A direction change with empty factors: square, rectangular with base zero and
            // with base one (the pointer arrays are reset to the base).
            {"change_square",
             row,
             2,
             2,
             {0, 1, 2},
             {0, 1},
             {1, 2},
             {unit, col, 0, {7, 7, 7}, base0},
             {unit, col, 0, {7, 7, 7}, base0},
             {},
             {0, 0, 0},
             none,
             {0, 0, 0},
             none,
             {},
             {},
             {}},
            {"change_rect0",
             row,
             2,
             3,
             dptr,
             dind,
             dval,
             {unit, col, 0, {7, 7, 7, 7}, base0},
             {unit, col, 0, {7, 7, 7, 7}, base0},
             {},
             {0, 0, 0, 0},
             none,
             {0, 0, 0, 0},
             none,
             {},
             {},
             {}},
            {"change_rect1",
             row,
             2,
             3,
             dptr,
             dind,
             dval,
             {unit, col, 0, {7, 7, 7, 7}, base1},
             {unit, col, 0, {7, 7, 7, 7}, base1},
             {},
             {1, 1, 1, 1},
             none,
             {1, 1, 1, 1},
             none,
             {},
             {},
             {}},
            // A direction change with entries: the factor is built as CSR in the scratch
            // pointer array and transposed to CSC. Only U differs / only L differs.
            {"change_upper",
             row,
             3,
             3,
             kptr,
             kind,
             kval,
             {unit, row, 1, {0, 0, 0, 1}, base0},
             {unit, col, 1, {9, 9, 9, 9}, base0},
             {0, 1, 1, 1},
             none,
             {0},
             {0, 0, 1, 1},
             {0},
             {4},
             {2},
             {1, 3, 5}},
            {"change_lower",
             row,
             3,
             3,
             kptr,
             kind,
             kval,
             {unit, col, 1, {9, 9, 9, 9}, base0},
             {unit, row, 1, {0, 1, 1, 1}, base0},
             {0, 0, 0, 1},
             {0, 1, 1, 1},
             {2},
             none,
             {1},
             {4},
             {2},
             {1, 3, 5}},
        };

        for(const SlduCase& c : cases)
        {
            SCOPED_TRACE(c.name);
            DevBuf<I> dptr_(c.ptr), dind_(c.ind), lptr(c.lower.ptr), lind(c.lower.nnz),
                uptr(c.upper.ptr), uind(c.upper.nnz);
            DevBuf<I>     scratch(std::vector<I>(c.scratch), 64);
            DevBuf<float> dval_(c.val), lval(c.lower.nnz), uval(c.upper.nnz),
                diag(std::vector<float>(std::max(c.m, c.n), -1));
            ASSERT_EQ((rocsparse::csxsldu_compute_template<float, I, I>(handle,
                                                                        c.dir,
                                                                        c.m,
                                                                        c.n,
                                                                        I(c.ind.size()),
                                                                        dptr_,
                                                                        dind_,
                                                                        dval_,
                                                                        base0,
                                                                        c.lower.diag,
                                                                        c.lower.dir,
                                                                        c.lower.nnz,
                                                                        lptr,
                                                                        lind,
                                                                        lval,
                                                                        c.lower.base,
                                                                        c.upper.diag,
                                                                        c.upper.dir,
                                                                        c.upper.nnz,
                                                                        uptr,
                                                                        uind,
                                                                        uval,
                                                                        c.upper.base,
                                                                        diag,
                                                                        scratch)),
                      st_ok);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            const auto check = [](const auto& buf, const auto& expected, const char* what) {
                if(!expected.empty())
                    EXPECT_EQ(buf.read(expected.size()), expected) << what;
            };
            check(lptr, c.lptr, "lower ptr");
            check(lind, c.lind, "lower ind");
            check(lval, c.lval, "lower val");
            check(uptr, c.uptr, "upper ptr");
            check(uind, c.uind, "upper ind");
            check(uval, c.uval, "upper val");
            check(diag, c.diag, "diag");
        }
    }

    // The whole split through the C entry points: buffer size, preprocess (sizes and
    // pointers of L and U) and compute. Same matrix as above, L and U strict in the row
    // direction, L one based; the diagonal is extracted.
    template <typename T, typename BS, typename PRE, typename COMP>
    void run_csxsldu_c_api(rocsparse_handle handle, BS buffer_size, PRE preprocess, COMP compute)
    {
        const auto           unit = rocsparse_diag_type_unit;
        const std::vector<I> ptr{0, 2, 3, 5}, ind{0, 1, 1, 0, 2};
        std::vector<T>       val;
        for(int k = 1; k <= 5; ++k)
            val.push_back(make_val<T>(k, -k));
        DevBuf<I> d_ptr(ptr), d_ind(ind);
        DevBuf<T> d_val(val);

        size_t bytes = 0;
        ASSERT_EQ(buffer_size(handle,
                              dir_r,
                              3,
                              3,
                              5,
                              d_ptr,
                              d_ind,
                              d_val,
                              base0,
                              unit,
                              dir_r,
                              unit,
                              dir_r,
                              &bytes),
                  st_ok);
        DevBuf<char> buffer(bytes);
        DevBuf<I>    lptr(std::vector<I>(4, -7)), uptr(std::vector<I>(4, -7));
        I            lnnz = -1, unnz = -1;
        ASSERT_EQ(preprocess(handle,
                             dir_r,
                             3,
                             3,
                             5,
                             d_ptr,
                             d_ind,
                             d_val,
                             base0,
                             unit,
                             dir_r,
                             &lnnz,
                             lptr,
                             base1,
                             unit,
                             dir_r,
                             &unnz,
                             uptr,
                             base0,
                             buffer.get()),
                  st_ok);
        ASSERT_EQ(lnnz, 1);
        ASSERT_EQ(unnz, 1);
        EXPECT_EQ(lptr.read(), (std::vector<I>{1, 1, 1, 2}));
        EXPECT_EQ(uptr.read(), (std::vector<I>{0, 1, 1, 1}));

        DevBuf<I> lind(std::vector<I>{-9}), uind(std::vector<I>{-9});
        DevBuf<T> lval(std::vector<T>{make_val<T>(-99)}), uval(std::vector<T>{make_val<T>(-99)}),
            diag(std::vector<T>(3, make_val<T>(-99)));
        ASSERT_EQ(compute(handle,
                          dir_r,
                          3,
                          3,
                          5,
                          d_ptr,
                          d_ind,
                          d_val,
                          base0,
                          unit,
                          dir_r,
                          lnnz,
                          lptr,
                          lind,
                          lval,
                          base1,
                          unit,
                          dir_r,
                          unnz,
                          uptr,
                          uind,
                          uval,
                          base0,
                          diag,
                          buffer.get()),
                  st_ok);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(lind.read(), std::vector<I>{1});
        EXPECT_EQ(uind.read(), std::vector<I>{1});
        EXPECT_TRUE(lval.read() == std::vector<T>{val[3]});
        EXPECT_TRUE(uval.read() == std::vector<T>{val[1]});
        EXPECT_TRUE(diag.read() == (std::vector<T>{val[0], val[2], val[4]}));
    }

    TEST_F(ConvInternal, CsxsLduCApi)
    {
        run_csxsldu_c_api<float>(handle,
                                 rocsparse_scsxsldu_buffer_size,
                                 rocsparse_scsxsldu_preprocess,
                                 rocsparse_scsxsldu_compute);
        run_csxsldu_c_api<rocsparse_double_complex>(handle,
                                                    rocsparse_zcsxsldu_buffer_size,
                                                    rocsparse_zcsxsldu_preprocess,
                                                    rocsparse_zcsxsldu_compute);
    }

    // ------------------------------------------------------------------
    // gell2csr: ELL to CSR through rocsparse_sparse_to_sparse
    // ------------------------------------------------------------------

    // Descriptor creation, then analysis and compute; returns the first status that is not
    // success (an unsupported conversion may be rejected by any of the calls).
    rocsparse_status sparse_to_sparse(rocsparse_handle            handle,
                                      rocsparse_const_spmat_descr src,
                                      rocsparse_spmat_descr       dst)
    {
        S2sDescr         s2s;
        rocsparse_status st = rocsparse_create_sparse_to_sparse_descr(
            &s2s, src, dst, rocsparse_sparse_to_sparse_alg_default);
        for(auto stage :
            {rocsparse_sparse_to_sparse_stage_analysis, rocsparse_sparse_to_sparse_stage_compute})
        {
            size_t size = 0;
            if(st == st_ok)
                st = rocsparse_sparse_to_sparse_buffer_size(handle, s2s, src, dst, stage, &size);
            if(st != st_ok)
                break;
            DevBuf<char> buffer(size);
            st = rocsparse_sparse_to_sparse(handle, s2s, src, dst, stage, size, buffer);
        }
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        return st;
    }

    // A 3 x 3 diagonal matrix in ELL (width 1) converted to CSR. C: type of the column
    // indices, P: type of the row pointers.
    template <typename T, typename C, typename P>
    void ell_to_csr(rocsparse_handle handle)
    {
        const std::vector<T> v{scalar<T>(1), scalar<T>(2), scalar<T>(3)};
        DevBuf<C>            ell_col(std::vector<C>{0, 1, 2}), col_ind(3);
        DevBuf<T>            ell_val(v), csr_val(3);
        DevBuf<P>            row_ptr(4);
        Spmat                ell, csr;
        ASSERT_EQ(rocsparse_create_ell_descr(
                      &ell, 3, 3, ell_col, ell_val, 1, it_of<C>(), base0, dt_of<T>()),
                  st_ok);
        ASSERT_EQ(rocsparse_create_csr_descr(&csr,
                                             3,
                                             3,
                                             3,
                                             row_ptr,
                                             col_ind,
                                             csr_val,
                                             it_of<P>(),
                                             it_of<C>(),
                                             base0,
                                             dt_of<T>()),
                  st_ok);
        ASSERT_EQ(sparse_to_sparse(handle, ell, csr), st_ok);
        EXPECT_EQ(row_ptr.read(), (std::vector<P>{0, 1, 2, 3}));
        EXPECT_EQ(col_ind.read(), (std::vector<C>{0, 1, 2}));
        EXPECT_EQ(csr_val.read(), v);
    }

    template <typename T>
    void ell_to_csr_all(rocsparse_handle handle)
    {
        ell_to_csr<T, int32_t, int32_t>(handle);
        ell_to_csr<T, int32_t, int64_t>(handle);
        ell_to_csr<T, int64_t, int32_t>(handle);
        ell_to_csr<T, int64_t, int64_t>(handle);
    }

    TEST_F(ConvInternal, Gell2Csr)
    {
        ell_to_csr_all<float>(handle);
        ell_to_csr_all<double>(handle);
        ell_to_csr_all<rocsparse_float_complex>(handle);
        ell_to_csr_all<rocsparse_double_complex>(handle);

        // Different value types, different column index types and the values the
        // conversion does not support are rejected.
        DevBuf<int32_t> ell_col(std::vector<int32_t>{0, 1, 2}), row_ptr(4), col_ind(3);
        DevBuf<int64_t> col64(3);
        DevBuf<float>   ell_val(std::vector<float>{1, 2, 3});
        DevBuf<double>  csr_d(3);
        DevBuf<float>   csr_f(3);
        const auto      reject = [&](rocsparse_datatype  ell_dt,
                                rocsparse_datatype  csr_dt,
                                rocsparse_indextype csr_col_type,
                                void*               csr_col,
                                void*               csr_val) {
            Spmat ell, csr;
            ASSERT_EQ(
                rocsparse_create_ell_descr(&ell, 3, 3, ell_col, ell_val, 1, i32, base0, ell_dt),
                st_ok);
            ASSERT_EQ(
                rocsparse_create_csr_descr(
                    &csr, 3, 3, 3, row_ptr, csr_col, csr_val, i32, csr_col_type, base0, csr_dt),
                st_ok);
            EXPECT_REJECTED(sparse_to_sparse(handle, ell, csr))
                << "datatype " << ell_dt << "/" << csr_dt;
        };
        reject(rocsparse_datatype_f32_r, rocsparse_datatype_f64_r, i32, col_ind, csr_d);
        reject(rocsparse_datatype_f32_r,
               rocsparse_datatype_f32_r,
               rocsparse_indextype_i64,
               col64,
               csr_f);
        for(auto dt : {rocsparse_datatype_i8_r,
                       rocsparse_datatype_u8_r,
                       rocsparse_datatype_i32_r,
                       rocsparse_datatype_u32_r,
                       rocsparse_datatype_f16_r,
                       rocsparse_datatype_bf16_r})
            reject(dt, dt, i32, col_ind, ell_val);
    }
} // namespace

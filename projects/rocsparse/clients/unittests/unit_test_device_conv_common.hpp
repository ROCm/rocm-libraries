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

// Helpers shared by the unit_test_device_conv_*.cpp tests: status shortcuts, RAII
// wrappers for device buffers and library objects, and small host references.

#pragma once

#include "rocsparse_handle.hpp"
#include "unit_test_utils.hpp"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// EXPECT_STATUS_IN(call, a, b, ...): the call returns one of the listed statuses; used
// where the header and the library disagree. EXPECT_REJECTED(call): the call returns
// an error, the status is not documented. The call is evaluated exactly once.
#define EXPECT_STATUS_IN(call, ...) EXPECT_TRUE(::convut::status_in((call), {__VA_ARGS__}))
#define EXPECT_REJECTED(call) EXPECT_NE((call), rocsparse_status_success)

namespace convut
{
    using namespace rocsparse_ut;

    using I = rocsparse_int;

    inline constexpr auto st_ok   = rocsparse_status_success;
    inline constexpr auto st_ptr  = rocsparse_status_invalid_pointer;
    inline constexpr auto st_size = rocsparse_status_invalid_size;
    inline constexpr auto st_val  = rocsparse_status_invalid_value;

    inline constexpr auto base0 = rocsparse_index_base_zero;
    inline constexpr auto base1 = rocsparse_index_base_one;
    inline constexpr auto dir_r = rocsparse_direction_row;
    inline constexpr auto dir_c = rocsparse_direction_column;
    inline constexpr auto i32   = rocsparse_indextype_i32;
    inline constexpr auto f32   = rocsparse_datatype_f32_r;

    // A data status that no check reports.
    inline const auto bad_ds = static_cast<rocsparse_data_status>(-1);

    inline ::testing::AssertionResult status_in(rocsparse_status                        actual,
                                                std::initializer_list<rocsparse_status> allowed)
    {
        if(std::find(allowed.begin(), allowed.end(), actual) != allowed.end())
        {
            return ::testing::AssertionSuccess();
        }
        auto r = ::testing::AssertionFailure() << "status " << actual << " is not one of:";
        for(rocsparse_status a : allowed)
        {
            r << " " << a;
        }
        return r;
    }

    // Calls f(args...) once per pointer argument with that argument replaced by
    // nullptr. Every call must return rocsparse_status_invalid_pointer.
    template <size_t K, typename F, typename Tuple>
    void expect_null_arg(F& f, const Tuple& t)
    {
        if constexpr(std::is_pointer_v<std::tuple_element_t<K, Tuple>>)
        {
            Tuple c        = t;
            std::get<K>(c) = nullptr;
            EXPECT_EQ(std::apply(f, c), st_ptr) << "argument " << K;
        }
    }
    template <typename F, typename Tuple, size_t... K>
    void expect_null_args_impl(F& f, const Tuple& t, std::index_sequence<K...>)
    {
        (expect_null_arg<K>(f, t), ...);
    }
    template <typename F, typename... A>
    void expect_null_args(F f, A... a)
    {
        expect_null_args_impl(f, std::make_tuple(a...), std::index_sequence_for<A...>{});
    }

    // The s, d, c or z variant of a routine, selected by the value type.
    template <typename T, typename S, typename D, typename C, typename Z>
    auto pick(S s, D d, C c, Z z)
    {
        if constexpr(std::is_same_v<T, float>)
            return s;
        else if constexpr(std::is_same_v<T, double>)
            return d;
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return c;
        else
            return z;
    }

    // Fixture owning a handle.
    class ConvTest : public HandleTest
    {
    };

    template <typename T>
    inline constexpr bool is_complex_v = false;
    template <>
    inline constexpr bool is_complex_v<rocsparse_float_complex> = true;
    template <>
    inline constexpr bool is_complex_v<rocsparse_double_complex> = true;

    // Value with the given real and imaginary parts (the imaginary part is ignored for
    // real types).
    template <typename T>
    T make_val(int re, int im = 0)
    {
        if constexpr(is_complex_v<T>)
        {
            return T(re, im);
        }
        else
        {
            return static_cast<T>(re);
        }
    }

    // Device copy of a host vector. At least `min_elems` elements are allocated (zero
    // filled beyond the host data) and the pointer is never null. Unlike
    // rocsparse_ut::device_vector, a failed allocation or copy fails the test.
    template <typename T>
    class DevBuf
    {
        T*     m_ptr = nullptr;
        size_t m_n;

    public:
        explicit DevBuf(const std::vector<T>& host, size_t min_elems = 0)
            : m_n(std::max({host.size(), min_elems, size_t(1)}))
        {
            if(hipMalloc(&m_ptr, m_n * sizeof(T)) != hipSuccess
               || hipMemset(m_ptr, 0, m_n * sizeof(T)) != hipSuccess
               || (!host.empty()
                   && hipMemcpy(m_ptr, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice)
                          != hipSuccess))
            {
                ADD_FAILURE() << "device allocation or copy failed";
            }
        }
        explicit DevBuf(size_t count)
            : DevBuf(std::vector<T>(), count)
        {
        }
        ~DevBuf()
        {
            EXPECT_EQ(hipFree(m_ptr), hipSuccess);
        }
        DevBuf(const DevBuf&) = delete;
        DevBuf& operator=(const DevBuf&) = delete;

        T* get() const
        {
            return m_ptr;
        }
        operator T*() const
        {
            return m_ptr;
        }
        // The first `count` elements (all allocated elements by default).
        std::vector<T> read(size_t count = size_t(-1)) const
        {
            std::vector<T> host(std::min(count, m_n));
            if(!host.empty())
            {
                EXPECT_EQ(
                    hipMemcpy(host.data(), m_ptr, host.size() * sizeof(T), hipMemcpyDeviceToHost),
                    hipSuccess);
            }
            return host;
        }
    };

    // Owns a library object and destroys it (checking the status) at the end of the
    // scope. `&owner` is the address of the stored handle, to be passed to a create call.
    template <typename H, auto DESTROY>
    class Owned
    {
        H m_h = nullptr;

    public:
        Owned() = default;
        ~Owned()
        {
            if(m_h != nullptr)
            {
                EXPECT_EQ(DESTROY(m_h), st_ok);
            }
        }
        Owned(const Owned&) = delete;
        Owned& operator=(const Owned&) = delete;

        H* operator&()
        {
            return &m_h;
        }
        H get() const
        {
            return m_h;
        }
        operator H() const
        {
            return m_h;
        }
    };

    // Non-const view of a const sparse matrix descriptor, for calls that modify it.
    inline rocsparse_spmat_descr mut(rocsparse_const_spmat_descr d)
    {
        return const_cast<rocsparse_spmat_descr>(d);
    }

    using Spmat      = Owned<rocsparse_spmat_descr, rocsparse_destroy_spmat_descr>;
    using ConstSpmat = Owned<rocsparse_const_spmat_descr, rocsparse_destroy_spmat_descr>;
    using ConstSpvec = Owned<rocsparse_const_spvec_descr, rocsparse_destroy_spvec_descr>;
    using ConstDnmat = Owned<rocsparse_const_dnmat_descr, rocsparse_destroy_dnmat_descr>;
    using Dnmat      = Owned<rocsparse_dnmat_descr, rocsparse_destroy_dnmat_descr>;
    using Dnvec      = Owned<rocsparse_dnvec_descr, rocsparse_destroy_dnvec_descr>;
    using ColorInfo  = Owned<rocsparse_color_info, rocsparse_destroy_color_info>;
    using HybMat     = Owned<rocsparse_hyb_mat, rocsparse_destroy_hyb_mat>;
    using S2sDescr
        = Owned<rocsparse_sparse_to_sparse_descr, rocsparse_destroy_sparse_to_sparse_descr>;
    using ExtractDescr  = Owned<rocsparse_extract_descr, rocsparse_destroy_extract_descr>;
    using MatDescrOwner = Owned<rocsparse_mat_descr, rocsparse_destroy_mat_descr>;
    using MatInfoOwner  = Owned<rocsparse_mat_info, rocsparse_destroy_mat_info>;

    // Library bug (AISPARSE-925): for a matrix it does not support,
    // rocsparse_create_extract_descr lets a rocsparse_status thrown by the descriptor
    // constructor escape the C API instead of returning it. Both behaviours are accepted.
    inline rocsparse_status create_extract(ExtractDescr&               descr,
                                           rocsparse_const_spmat_descr source,
                                           rocsparse_spmat_descr       target)
    {
        try
        {
            return rocsparse_create_extract_descr(
                &descr, source, target, rocsparse_extract_alg_default);
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
    }

    // Matrix descriptor with the given index base.
    struct MatDescr : MatDescrOwner
    {
        explicit MatDescr(rocsparse_index_base base = base0)
        {
            EXPECT_EQ(rocsparse_create_mat_descr(&*this), st_ok);
            EXPECT_EQ(rocsparse_set_mat_index_base(*this, base), st_ok);
        }
    };

    // Created matrix info structure.
    struct MatInfo : MatInfoOwner
    {
        MatInfo()
        {
            EXPECT_EQ(rocsparse_create_mat_info(&*this), st_ok);
        }
    };

    // Sets the pointer mode of a handle for the lifetime of the object.
    class PointerModeScope
    {
        rocsparse_handle       m_h;
        rocsparse_pointer_mode m_saved;

    public:
        PointerModeScope(rocsparse_handle h, rocsparse_pointer_mode mode)
            : m_h(h)
            , m_saved(h->pointer_mode)
        {
            EXPECT_EQ(rocsparse_set_pointer_mode(h, mode), st_ok);
        }
        ~PointerModeScope()
        {
            (void)rocsparse_set_pointer_mode(m_h, m_saved);
        }
    };

    inline const rocsparse_pointer_mode s_pointer_modes[]
        = {rocsparse_pointer_mode_host, rocsparse_pointer_mode_device};

    // Drops the handle's internal buffer for the lifetime of the object, so that
    // routines have to allocate their own temporary storage.
    class NoHandleBuffer
    {
        rocsparse_handle m_h;
        size_t           m_saved;

    public:
        explicit NoHandleBuffer(rocsparse_handle h)
            : m_h(h)
            , m_saved(h->buffer_size)
        {
            h->buffer_size = 0;
        }
        ~NoHandleBuffer()
        {
            m_h->buffer_size = m_saved;
        }
    };

    // A NaN or an infinity (with a finite imaginary part for complex types).
    template <typename T>
    T make_special(bool nan)
    {
        using R = std::conditional_t<
            std::is_same_v<T, float> || std::is_same_v<T, rocsparse_float_complex>,
            float,
            double>;
        const R v = nan ? std::numeric_limits<R>::quiet_NaN() : std::numeric_limits<R>::infinity();
        if constexpr(is_complex_v<T>)
            return T(v, 1);
        else
            return v;
    }

    inline rocsparse_index_base base_of(int base)
    {
        return base ? base1 : base0;
    }

    // The nnz result of a routine, obtained through a host or a device pointer.
    template <typename F>
    rocsparse_int nnz_via(rocsparse_handle handle, bool device_mode, F f)
    {
        PointerModeScope      pm(handle, s_pointer_modes[device_mode]);
        rocsparse_int         host = -1;
        DevBuf<rocsparse_int> dev(std::vector<rocsparse_int>{-1});
        EXPECT_EQ(f(device_mode ? dev.get() : &host), st_ok);
        EXPECT_EQ(hipStreamSynchronize(handle->stream), hipSuccess);
        return device_mode ? dev.read(1)[0] : host;
    }

    // Host CSR matrix with the index base `base` (0 or 1).
    template <typename T>
    struct HostCsr
    {
        I              m = 0, n = 0;
        int            base = 0;
        std::vector<I> ptr, ind;
        std::vector<T> val;
        I              nnz() const
        {
            return ind.size();
        }
    };

    // The entries of `a` for which keep(value) holds, with the index base `base`.
    template <typename T, typename P>
    HostCsr<T> keep_entries(const HostCsr<T>& a, int base, P keep)
    {
        HostCsr<T> c;
        c.m    = a.m;
        c.n    = a.n;
        c.base = base;
        c.ptr.push_back(base);
        for(int r = 0; r < a.m; ++r)
        {
            for(rocsparse_int k = a.ptr[r] - a.base; k < a.ptr[r + 1] - a.base; ++k)
            {
                if(keep(a.val[k]))
                {
                    c.ind.push_back(a.ind[k] - a.base + base);
                    c.val.push_back(a.val[k]);
                }
            }
            c.ptr.push_back(base + c.ind.size());
        }
        return c;
    }

    // Device copy of a host CSR matrix, with its descriptor.
    template <typename T>
    struct DevCsr
    {
        HostCsr<T>            a;
        DevBuf<rocsparse_int> ptr, ind;
        DevBuf<T>             val;
        MatDescr              descr;
        explicit DevCsr(const HostCsr<T>& h)
            : a(h)
            , ptr(h.ptr)
            , ind(h.ind)
            , val(h.val)
            , descr(base_of(h.base))
        {
        }
    };

    // The arrays of a CSR matrix on the device are the ones of `want`.
    template <typename T>
    void expect_csr(const DevBuf<rocsparse_int>& ptr,
                    const DevBuf<rocsparse_int>& ind,
                    const DevBuf<T>&             val,
                    const HostCsr<T>&            want)
    {
        EXPECT_EQ(ptr.read(want.m + 1), want.ptr);
        EXPECT_EQ(ind.read(want.nnz()), want.ind);
        EXPECT_EQ(val.read(want.nnz()), want.val);
    }

    inline size_t value_size(rocsparse_datatype dt)
    {
        switch(dt)
        {
        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
            return 1;
        case rocsparse_datatype_f16_r:
        case rocsparse_datatype_bf16_r:
            return 2;
        case rocsparse_datatype_f64_r:
        case rocsparse_datatype_f32_c:
            return 8;
        case rocsparse_datatype_f64_c:
            return 16;
        default:
            return 4;
        }
    }

    // `AsConst` selects rocsparse_create_const_csr_descr / rocsparse_create_const_csc_descr.
    // The descriptor type must be ConstSpmat in that case and Spmat otherwise.
    template <bool AsConst = false, typename Descr>
    rocsparse_status create_csx(Descr&              m,
                                bool                csc,
                                int64_t             rows,
                                int64_t             cols,
                                int64_t             nnz,
                                void*               ptr,
                                void*               ind,
                                void*               val,
                                rocsparse_indextype ptr_type,
                                rocsparse_indextype ind_type,
                                int                 base,
                                rocsparse_datatype  dt)
    {
        const auto ib = base_of(base);
        if constexpr(AsConst)
        {
            const auto create
                = csc ? rocsparse_create_const_csc_descr : rocsparse_create_const_csr_descr;
            return create(&m, rows, cols, nnz, ptr, ind, val, ptr_type, ind_type, ib, dt);
        }
        else
        {
            const auto create = csc ? rocsparse_create_csc_descr : rocsparse_create_csr_descr;
            return create(&m, rows, cols, nnz, ptr, ind, val, ptr_type, ind_type, ib, dt);
        }
    }

    // Extracts the part of the matrix given by (ptr, ind) and checks it against one host
    // reference. For a unit diagonal the result may keep or drop the diagonal (the header
    // does not say which); it must match one of the two references exactly. `AsConst`
    // builds the source with the const constructor, fills every output with a sentinel,
    // and requires the guard elements past the written prefix to stay unchanged.
    template <typename P, typename J, bool AsConst = false>
    void run_extract(rocsparse_handle      handle,
                     bool                  csc,
                     int64_t               rows,
                     int64_t               cols,
                     int                   base,
                     rocsparse_datatype    dt,
                     const std::vector<P>& ptr,
                     const std::vector<J>& ind,
                     rocsparse_fill_mode   fill,
                     rocsparse_diag_type   diag)
    {
        SCOPED_TRACE("csc " + std::to_string(csc) + " " + std::to_string(rows) + "x"
                     + std::to_string(cols) + " base " + std::to_string(base) + " datatype "
                     + std::to_string(dt) + " fill " + std::to_string(fill) + " diag "
                     + std::to_string(diag) + (AsConst ? " const" : ""));
        const int64_t nnz = ind.size();
        const size_t  vs  = value_size(dt);
        // Entry k holds the byte 0x3f + k in every byte, a finite value of every datatype.
        std::vector<char> val;
        for(int64_t k = 0; k < nnz; ++k)
            val.insert(val.end(), vs, char(0x3f + k));
        // Four elements past each output array. Only the const source checks them.
        constexpr size_t extra    = AsConst ? 4 : 0;
        const J          ind_fill = AsConst ? static_cast<J>(-7) : static_cast<J>(0);
        const char       val_fill = AsConst ? char(0xA5) : char(0);
        DevBuf<P>        d_ptr(ptr), t_ptr(std::vector<P>(ptr.size() + extra, static_cast<P>(-7)));
        DevBuf<J>    d_ind(ind), t_ind(std::vector<J>(static_cast<size_t>(nnz) + extra, ind_fill));
        DevBuf<char> d_val(val),
            t_val(std::vector<char>(static_cast<size_t>(nnz) * vs + extra * vs, val_fill));
        using Source = std::conditional_t<AsConst, ConstSpmat, Spmat>;
        Source source;
        ASSERT_EQ((create_csx<AsConst>(source,
                                       csc,
                                       rows,
                                       cols,
                                       nnz,
                                       d_ptr,
                                       d_ind,
                                       d_val,
                                       it_of<P>(),
                                       it_of<J>(),
                                       base,
                                       dt)),
                  st_ok);
        Spmat target;
        ASSERT_EQ(create_csx(target,
                             csc,
                             rows,
                             cols,
                             0,
                             t_ptr,
                             nullptr,
                             nullptr,
                             it_of<P>(),
                             it_of<J>(),
                             base,
                             dt),
                  st_ok);
        ASSERT_EQ(
            rocsparse_spmat_set_attribute(target, rocsparse_spmat_fill_mode, &fill, sizeof(fill)),
            st_ok);
        ASSERT_EQ(
            rocsparse_spmat_set_attribute(target, rocsparse_spmat_diag_type, &diag, sizeof(diag)),
            st_ok);

        rocsparse_const_spmat_descr source_view = source.get();
        ExtractDescr                descr;
        ASSERT_EQ(create_extract(descr, source_view, target), st_ok);
        const auto stage = [&](rocsparse_extract_stage s) {
            size_t size = 0;
            ASSERT_EQ(rocsparse_extract_buffer_size(handle, descr, source_view, target, s, &size),
                      st_ok);
            DevBuf<char> buffer(size);
            ASSERT_EQ(rocsparse_extract(handle, descr, source_view, target, s, size, buffer),
                      st_ok);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        };
        stage(rocsparse_extract_stage_analysis);
        int64_t target_nnz = -1;
        ASSERT_EQ(rocsparse_extract_nnz(handle, descr, &target_nnz), st_ok);
        ASSERT_EQ((csc ? rocsparse_csc_set_pointers
                       : rocsparse_csr_set_pointers)(target, t_ptr, t_ind, t_val),
                  st_ok);
        stage(rocsparse_extract_stage_compute);

        // Host references: the entries below (above) the diagonal, with the diagonal
        // (strict = 0) and without it (strict = 1).
        std::vector<P>    want_ptr[2];
        std::vector<J>    want_ind[2];
        std::vector<char> want_val[2];
        for(int strict = 0; strict < 2; ++strict)
        {
            want_ptr[strict] = {static_cast<P>(base)};
            for(int64_t outer = 0; outer + 1 < static_cast<int64_t>(ptr.size()); ++outer)
            {
                for(int64_t k = ptr[outer] - base; k < ptr[outer + 1] - base; ++k)
                {
                    const int64_t i = csc ? ind[k] - base : outer, j = csc ? outer : ind[k] - base;
                    if(i == j ? !strict : (fill == rocsparse_fill_mode_lower) == (j < i))
                    {
                        want_ind[strict].push_back(ind[k]);
                        want_val[strict].insert(want_val[strict].end(), vs, char(0x3f + k));
                    }
                }
                want_ptr[strict].push_back(base + want_ind[strict].size());
            }
        }
        const int strict = diag == rocsparse_diag_type_unit
                           && target_nnz != static_cast<int64_t>(want_ind[0].size());
        ASSERT_EQ(target_nnz, static_cast<int64_t>(want_ind[strict].size()));
        if constexpr(AsConst)
        {
            auto       expect_ptr = want_ptr[strict];
            auto       expect_ind = want_ind[strict];
            auto       expect_val = want_val[strict];
            const auto got_ptr    = t_ptr.read();
            const auto got_ind    = t_ind.read();
            const auto got_val    = t_val.read();
            if(expect_ptr.size() < got_ptr.size())
                expect_ptr.resize(got_ptr.size(), static_cast<P>(-7));
            if(expect_ind.size() < got_ind.size())
                expect_ind.resize(got_ind.size(), static_cast<J>(-7));
            if(expect_val.size() < got_val.size())
                expect_val.resize(got_val.size(), char(0xA5));
            EXPECT_EQ(got_ptr, expect_ptr);
            EXPECT_EQ(got_ind, expect_ind);
            EXPECT_EQ(got_val, expect_val);
        }
        else
        {
            EXPECT_EQ(t_ptr.read(ptr.size()), want_ptr[strict]);
            EXPECT_EQ(t_ind.read(target_nnz), want_ind[strict]);
            EXPECT_EQ(t_val.read(target_nnz * vs), want_val[strict]);
        }
    }
} // namespace convut

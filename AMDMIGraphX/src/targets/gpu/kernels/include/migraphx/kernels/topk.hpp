/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */
#ifndef MIGRAPHX_GUARD_KERNELS_TOPK_HPP
#define MIGRAPHX_GUARD_KERNELS_TOPK_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/bit.hpp>
#include <migraphx/kernels/ranges.hpp>
#include <migraphx/kernels/slice.hpp>
#include <migraphx/kernels/sort.hpp>
#include <migraphx/kernels/operators.hpp>
#include <migraphx/kernels/float_equal.hpp>

namespace migraphx {

template <class T, class U>
struct topk_pair_t_u
{
    T key;
    U val;
};

template <class T, class U>
struct topk_pair_u_t
{
    U val;
    T key;
};

template <class T, class U>
struct topk_pair : conditional_t<(sizeof(T) >= sizeof(U)), topk_pair_t_u<T, U>, topk_pair_u_t<T, U>>
{
    template <class Stream>
    friend constexpr const Stream& operator<<(const Stream& ss, const topk_pair& tp)
    {
        ss << "{ " << tp.key << ", " << tp.val << "}";
        return ss;
    }
};

constexpr auto select_key()
{
    return [](const auto& p) { return p.key; };
}

template <class Compare>
constexpr auto compare_topk_pair(Compare compare)
{
    return [=](const auto& x, const auto& y) {
        if(compare(x.key, y.key))
            return true;
        if(compare(y.key, x.key))
            return false;
        return x.val < y.val;
    };
}

template <class T, class Type = typename T::type>
constexpr auto
    get_index_type(T) -> conditional_t<(sizeof(Type) < sizeof(index_int)), Type, index_int>;

constexpr auto get_index_type() -> uint16_t;

template <class... X>
constexpr auto make_get_index(X... x_idxs)
{
    if constexpr(sizeof...(x_idxs) == 1)
    {
        auto x_idx = arg_c<0>()(x_idxs...);
        return [=](auto i) { return i < x_idx.get_shape().elements() ? x_idx[i] : -1; };
    }
    else
    {
        return [](auto i) { return i; };
    }
}

template <index_int Axis, class Compare, class T, class Y, class YIndex, class X, class... XIndices>
__device__ auto
topk_impl(index idx, Compare compare, T init, Y y, YIndex y_idx, X x, XIndices... x_idxs)
{
    using type               = typename X::type;
    constexpr auto n         = _c<get_shape_c<X>{}.get_shape().lens[Axis]>;
    constexpr auto k         = _c<get_shape_c<Y>{}.get_shape().lens[Axis]>;
    constexpr auto aligned_n = _c<bit_ceil(n)>;
    constexpr auto aligned_k = _c<bit_ceil(k)>;
    using pair =
        topk_pair<type, conditional_t<(n > 32768), index_int, decltype(get_index_type(x_idxs...))>>;
    auto get_index             = make_get_index(x_idxs...);
    constexpr auto nlocal_wave = idx.nlocal_wave();
    constexpr auto nwave       = idx.nwave();
    constexpr auto m           = k * nwave;
    constexpr auto aligned_m   = _c<bit_ceil(m)>;
    if constexpr(aligned_m < aligned_n or nwave == 1)
    {
        constexpr auto extra_m   = aligned_m - m;
        constexpr auto over_n    = where(n == aligned_n, _c<0>, n - aligned_n / _c<2>);
        constexpr auto trimmed_n = max(where(over_n < extra_m, n - over_n, n), min(aligned_m, n));
        constexpr auto nper_wave = trimmed_n / nwave;
        constexpr auto nper_lane = _c<bit_ceil(ceil_div(nper_wave, nlocal_wave))>;
        MIGRAPHX_ASSERT(nper_wave >= k);
        MIGRAPHX_ASSERT(trimmed_n <= n);

        array<pair, nper_lane> local_buf;
        for(index_int i : range(nper_lane))
        {
            local_buf[i].key = init;
            local_buf[i].val = -1;
        }
        // copy to registers
        idx.local_stride(trimmed_n, [&](auto j, auto i) {
            local_buf[i].key = x[j];
            local_buf[i].val = get_index(j);
        });

        bitonic_sort{compare_topk_pair(compare)}.wave_sort(idx, local_buf);

        if constexpr(nwave == 1)
        {
            const auto local_shape = make_shape(index_ints<nwave, nlocal_wave, nper_lane>{});
            MIGRAPHX_ASSERT(local_shape.elements() >= trimmed_n);
            // Copy to output
            for(index_int i : range(nper_lane))
            {
                auto j = local_shape.index({idx.wave(), idx.local_wave(), i});
                if(j >= k)
                    continue;
                y[j]     = local_buf[i].key;
                y_idx[j] = local_buf[i].val;
            }
        }
        else
        {
            __shared__ pair buf[aligned_m];
            idx.local_stride(extra_m, [&](auto i) {
                auto in = i + trimmed_n;
                auto im = i + m;
                MIGRAPHX_ASSERT(im < aligned_m);
                buf[im].key = in < n ? x[in] : init;
                buf[im].val = get_index(in);
            });
            auto shared_shape = make_shape(index_ints<nwave, k>{});
            const auto base   = idx.local_wave() * nper_lane;
            for(index_int i : range(nper_lane))
            {
                auto ibase = i + base;
                if(ibase >= k)
                    continue;
                auto j = shared_shape.index({idx.wave(), ibase});
                MIGRAPHX_ASSERT(j < m);
                buf[j] = local_buf[i];
            }
            __syncthreads();

            bitonic_topk{aligned_m, aligned_k, compare_topk_pair(compare)}.block_topk(idx, buf);

            // save top K
            idx.local_stride(k, [&](auto i) {
                y[i]     = buf[i].key;
                y_idx[i] = buf[i].val;
            });
        }
    }
    else
    {
        __shared__ pair buf[aligned_n];
        // Copy to LDS
        idx.local_stride(aligned_n, [&](auto i) {
            auto key   = i < x.get_shape().elements() ? x[i] : init;
            buf[i].key = key;
            buf[i].val = get_index(i);
        });
        __syncthreads();
        bitonic_topk{aligned_n, aligned_k, compare_topk_pair(compare)}.block_topk(idx, buf);

        // save top K
        idx.local_stride(k, [&](auto i) {
            y[i]     = buf[i].key;
            y_idx[i] = buf[i].val;
        });
    }
}

template <index_int Axis, class Compare, class T>
__device__ auto topk(Compare compare, T init)
{
    return [=](auto output, auto out_indices, auto input, auto... in_indices) {
        auto idx = make_index();
        slice_schedule<per_block>(idx,
                                  slice_axes<Axis>())(output, out_indices, input, in_indices...)(
            [&](auto y, auto y_idx, auto x, auto... x_idxs) {
                topk_impl<Axis>(idx, compare, init, y, y_idx, x, x_idxs...);
            });
    };
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_TOPK_HPP

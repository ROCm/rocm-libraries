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
 */
#ifndef MIGRAPHX_GUARD_KERNELS_SLICE_HPP
#define MIGRAPHX_GUARD_KERNELS_SLICE_HPP

#include <migraphx/kernels/shape.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/index.hpp>

namespace migraphx {

template <class Shape, class Size>
constexpr auto slice_make_multi_lens(Shape, Size)
{
    return return_array_c([] {
        auto n     = Size{} - _c<1>;
        auto i     = Shape{}.multi(n);
        using type = typename decltype(i)::value_type;
        return i + type{1};
    });
}

template <class Shape, class T, T... Xs>
constexpr auto slice_make_multi_lens(Shape, integral_const_array<T, Xs...> x)
{
    return x;
}

template <class Shape, class Select>
constexpr auto make_slice(Shape, Select select)
{
    auto inner_lens = transform_i(Shape{}.lens, [=](index_int x, index_int ii) -> index_int {
        if(select(x, ii, Shape{}.lens.size()))
            return x;
        return 1;
    });
    return make_shape(inner_lens, Shape{}.strides);
}

template <class Shape, class Select, class Size>
constexpr auto make_slice(Shape input, Select select, Size size)
{
    auto as   = make_slice(input, select);
    auto lens = slice_make_multi_lens(as, size);
    return make_shape(lens, Shape{}.strides);
}

template <class F>
struct slice_size_transform
{
    F f;

    template <class... Ts>
    constexpr auto operator()(Ts... xs) const
    {
        return f(xs...);
    }
};
MIGRAPHX_AUTO_DEDUCE(slice_size_transform);

template <class Shape, class Select, class F>
constexpr auto make_slice(Shape input, Select select, slice_size_transform<F> t)
{
    auto as   = make_slice(input, select);
    auto lens = slice_make_multi_lens(as, decltype(t(input, as)){});
    return make_shape(lens, Shape{}.strides);
}

template <class Shape, class... Ss>
constexpr auto nslices(Shape input, Ss... ss)
{
    auto as = make_slice(input, ss...);
    return input.elements() / as.elements();
}

template <index_int N>
constexpr auto slice_group()
{
    return slice_size_transform{[](auto input, auto s) {
        auto r = return_array_c([] {
            auto lens = decltype(s){}.lens.base();
            lens.back() *= N;
            lens -= 1;
            return decltype(input){}.lens.carry(lens) + index_int{1};
        });
        return r;
    }};
}

template <index_int N>
constexpr auto slice_split()
{
    return slice_size_transform{[](auto, auto s) { return s.elements() / _c<N>; }};
}

template <diff_int... Axes>
constexpr auto slice_axes()
{
    return [](auto, auto i, auto n) { return ((Axes < 0 ? i == (n + Axes) : i == Axes) or ...); };
}

template <class Input, class T, class... Ss>
constexpr auto slice_tensor(Input input, T start, Ss... ss)
{
    constexpr auto inner_shape = make_slice(get_shape_c<Input>{}, ss...);
    auto outer_lens            = transform(get_shape_c<Input>{}.lens,
                                inner_shape.lens,
                                [=](auto x, auto inner) { return 1 + x - inner; });
    auto outer_shape           = make_shape(outer_lens, get_shape_c<Input>{}.strides);
    auto offset                = outer_shape.index(start);
    MIGRAPHX_ASSERT((offset + inner_shape.element_space()) <= get_shape_c<Input>{}.element_space());
    return make_tensor_view(input.data() + offset, inner_shape);
}

template <class Schedule, class... Ss>
constexpr auto slice_schedule(index idx, Ss... ss)
{
    return [=](auto... xs) {
        return [=](auto f) {
            // TODO: Assert nslices is the same for all xs
            constexpr auto n = nslices(get_shape_c<decltype(arg_c<0>()(xs...))>{}, ss...);
            Schedule{idx}.group_stride(n, [&](auto i) { f(slice_tensor(xs, i, ss...)...); });
        };
    };
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_SLICE_HPP

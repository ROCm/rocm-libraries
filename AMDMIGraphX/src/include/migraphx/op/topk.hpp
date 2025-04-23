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
#ifndef MIGRAPHX_GUARD_OPERATORS_GATHER_HPP
#define MIGRAPHX_GUARD_OPERATORS_GATHER_HPP

#include <algorithm>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct topk
{
    int64_t k    = 1;
    int64_t axis = 0;
    bool largest = true;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.k, "k"), f(self.axis, "axis"), f(self.largest, "largest"));
    }

    value attributes() const
    {
        value normalize;
        normalize["axis"] = value::array{normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "topk"; }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1, 2);
        auto lens = inputs.at(0).lens();
        auto type = inputs.at(0).type();

        lens[axis] = k;

        shape s_val{type, lens};
        shape s_ind{shape::int64_type, lens};

        return shape({s_val, s_ind});
    }

    template <class Compare>
    static auto compare_pair(Compare compare)
    {
        return [=](auto p1, auto p2) {
            auto [x, i] = p1;
            auto [y, j] = p2;
            if(not float_equal(x, y))
                return compare(x, y);
            return i < j;
        };
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        const auto& vec_ss = output_shape.sub_shapes();
        argument res_val{vec_ss.front()};
        argument res_ind{vec_ss.back()};
        auto in_val       = args.front();
        auto relements    = in_val.get_shape().lens()[axis];
        auto make_indices = [&](const auto& m_idx) {
            return [&](int64_t i) {
                if(args.size() < 2)
                    return i;
                auto j  = m_idx;
                j[axis] = i;
                return args[1].at<int64_t>(j);
            };
        };
        auto outer_lens  = in_val.get_shape().lens();
        outer_lens[axis] = 1;
        shape outer_shape{in_val.get_shape().type(), outer_lens};
        visit_all(res_val, args.front())([&](auto output, auto input) {
            res_ind.visit([&](auto out_ind) {
                using type = typename decltype(input)::value_type;
                std::vector<std::pair<type, int64_t>> data(relements);
                par_for(outer_shape.elements(), [&](auto i) {
                    auto outer_idx = outer_shape.multi(i);
                    auto x         = input.slice_at({axis}, outer_idx);
                    auto y         = output.slice_at({axis}, outer_idx);
                    auto y_ind     = out_ind.slice_at({axis}, outer_idx);
                    auto get_index = make_indices(outer_idx);
                    transform(range(relements), data.begin(), [&](auto j) {
                        return std::make_pair(x[j], get_index(j));
                    });
                    if(this->largest)
                        std::partial_sort(data.begin(),
                                          data.begin() + k,
                                          data.end(),
                                          compare_pair(std::greater<>{}));
                    else
                        std::partial_sort(data.begin(),
                                          data.begin() + k,
                                          data.end(),
                                          compare_pair(std::less<>{}));
                    std::transform(data.begin(),
                                   data.begin() + this->k,
                                   y.begin(),
                                   [](const auto& p) { return p.first; });
                    std::transform(data.begin(),
                                   data.begin() + this->k,
                                   y_ind.begin(),
                                   [](const auto& p) { return p.second; });
                });
            });
        });
        return {{res_val, res_ind}};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif

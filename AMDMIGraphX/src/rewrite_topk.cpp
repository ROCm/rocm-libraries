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
#include <migraphx/rewrite_topk.hpp>
#include <migraphx/module.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/array.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

struct find_large_topk
{
    std::size_t n_threshold = 0;
    auto matcher() const { return match::name("topk"); }

    static std::size_t split_dim(std::size_t& r, std::size_t min_size)
    {
        std::size_t n = 1;
        auto factors  = make_array(2, 3, 5, 7, 11);
        while(r > min_size)
        {
            // NOLINTNEXTLINE(readability-qualified-auto)
            auto it =
                std::find_if(factors.begin(), factors.end(), [&](auto d) { return r % d == 0; });
            if(it == factors.end())
                break;
            r /= *it;
            n *= *it;
        }
        return n;
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins   = r.result;
        auto input = ins->inputs().front();
        auto op    = ins->get_operator().to_value();
        auto axis  = op["axis"].to<std::int64_t>();
        auto k     = op["k"].to<std::int64_t>();
        auto dims  = input->get_shape().lens();
        auto n     = dims.at(axis);
        if(n < n_threshold)
            return;

        auto gdims = dims;
        // We have to sort at least k elements, so the min size is k*4 or half the threshold
        auto group = split_dim(gdims[axis], std::max<std::size_t>(n_threshold / 2, k * 4));
        if(group < 2)
            return;
        gdims.insert(gdims.begin() + axis, group);
        op["axis"] = axis + 1;

        auto fdims        = dims;
        fdims[axis]       = k * group;
        auto insert_final = [&](auto t, auto i) {
            auto elem = m.insert_instruction(ins, make_op("get_tuple_elem", {{"index", i}}), t);
            return m.insert_instruction(ins, make_op("reshape", {{"dims", fdims}}), elem);
        };

        std::vector<std::size_t> indices_data(n);
        std::iota(indices_data.begin(), indices_data.end(), 0);
        auto indices_lit = m.add_literal(
            shape{(n < 65536 ? shape::uint16_type : shape::uint32_type), {n}}, indices_data);

        auto indices = m.insert_instruction(
            ins, make_op("broadcast", {{"axis", axis}, {"out_lens", dims}}), indices_lit);
        auto gindices = m.insert_instruction(ins, make_op("reshape", {{"dims", gdims}}), indices);
        auto ginput   = m.insert_instruction(ins, make_op("reshape", {{"dims", gdims}}), input);
        auto topk1    = m.insert_instruction(ins, make_op("topk", op), ginput, gindices);
        auto finput   = insert_final(topk1, 0);
        auto findices = insert_final(topk1, 1);
        m.replace_instruction(ins, ins->get_operator(), finput, findices);
    }
};

} // namespace

void rewrite_topk::apply(module& m) const
{
    match::find_matches(m, find_large_topk{split_threshold});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

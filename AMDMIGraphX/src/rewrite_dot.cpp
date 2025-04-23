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
#include <migraphx/rewrite_dot.hpp>
#include <migraphx/module.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/permutation.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

MIGRAPHX_PRED_MATCHER(conv_1x1, instruction_ref ins)
{
    if(ins->name() != "convolution")
        return false;
    auto v = ins->get_operator().to_value();
    if(v.at("group").to<int>() != 1)
        return false;
    if(not all_of(v.at("stride"), [](const value& x) { return x.to<std::size_t>() == 1; }))
        return false;
    if(not all_of(v.at("padding"), [](const value& x) { return x.to<std::size_t>() == 0; }))
        return false;
    if(not all_of(v.at("dilation"), [](const value& x) { return x.to<std::size_t>() == 1; }))
        return false;
    auto w = ins->inputs().at(1)->get_shape();
    return std::all_of(w.lens().begin() + 2, w.lens().end(), [](std::size_t i) { return i == 1; });
}

struct find_1x1_convolution
{
    auto matcher() const { return conv_1x1(match::arg(1)(match::is_constant())); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;

        auto input   = ins->inputs().front();
        auto weights = ins->inputs().back();

        std::vector<int64_t> sq_axes(ins->get_shape().ndim() - 2);
        std::iota(sq_axes.begin(), sq_axes.end(), 2);
        auto sq_weights =
            m.insert_instruction(ins, make_op("squeeze", {{"axes", sq_axes}}), weights);

        if(ins->get_shape().transposed())
        {
            std::vector<int64_t> aperm(ins->get_shape().ndim());
            std::iota(aperm.begin(), aperm.end(), 0);
            std::rotate(aperm.begin() + 1, aperm.begin() + 2, aperm.end());
            auto a_mat =
                m.insert_instruction(ins, make_op("transpose", {{"permutation", aperm}}), input);

            auto transpose = m.insert_instruction(
                ins, make_op("transpose", {{"permutation", {1, 0}}}), sq_weights);
            auto b_lens = a_mat->get_shape().lens();
            copy(transpose->get_shape().lens(), b_lens.end() - 2);
            auto b_mat = m.insert_instruction(
                ins, make_op("multibroadcast", {{"out_lens", b_lens}}), transpose);

            auto dot = m.insert_instruction(ins, make_op("dot"), a_mat, b_mat);
            m.replace_instruction(
                ins, make_op("transpose", {{"permutation", invert_permutation(aperm)}}), dot);
        }
        else
        {
            auto batch_dim = ins->get_shape().lens().front();
            auto m_dim     = std::accumulate(input->get_shape().lens().begin() + 2,
                                         input->get_shape().lens().end(),
                                         1,
                                         std::multiplies<>{});
            auto n_dim     = weights->get_shape().lens()[0];
            auto k_dim     = weights->get_shape().lens()[1];
            auto a_mat     = m.insert_instruction(
                ins,
                make_op("multibroadcast", {{"out_lens", {batch_dim, n_dim, k_dim}}}),
                sq_weights);
            auto b_mat = m.insert_instruction(
                ins, make_op("reshape", {{"dims", {batch_dim, k_dim, m_dim}}}), input);
            auto dot = m.insert_instruction(ins, make_op("dot"), a_mat, b_mat);
            m.replace_instruction(
                ins, make_op("reshape", {{"dims", ins->get_shape().lens()}}), dot);
        }
    }
};

} // namespace

void rewrite_dot::apply(module& m) const { match::find_matches(m, find_1x1_convolution{}); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

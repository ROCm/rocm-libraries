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
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/permutation.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

static instruction_ref
apply_channels_last_perm(const onnx_parser::node_info& info, instruction_ref ins, bool invert)
{
    std::vector<int64_t> perm(ins->get_shape().ndim());
    std::iota(perm.begin() + 1, perm.end() - 1, 2);
    perm.back() = 1;
    return info.add_instruction(
        make_op("transpose", {{"permutation", invert ? invert_permutation(perm) : perm}}), ins);
}

struct parse_groupnorm : op_parser<parse_groupnorm>
{
    std::vector<op_desc> operators() const
    {
        return {{"GroupNormalization", "GroupNorm"}, {"GroupNorm", "Contrib_GroupNorm"}};
    }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        bool is_contrib = (opd.op_name == ("Contrib_GroupNorm"));

        float epsilon = 1e-5f;
        if(contains(info.attributes, "epsilon"))
        {
            epsilon = parser.parse_value(info.attributes.at("epsilon")).at<float>();
        }
        size_t num_groups;
        if(contains(info.attributes, "num_groups") or contains(info.attributes, "groups"))
        {
            if(is_contrib)
            {
                num_groups =
                    std::abs(parser.parse_value(info.attributes.at("groups")).at<int64_t>());
            }
            else
            {
                num_groups =
                    std::abs(parser.parse_value(info.attributes.at("num_groups")).at<int64_t>());
            }
        }
        else
        {
            MIGRAPHX_THROW("PARSE_GROUPNORM: num_groups must be available");
        }

        bool is_channels_last = false;
        if(is_contrib)
        { // default state for GroupNorm Contrib op
            is_channels_last = true;
            if(contains(info.attributes, "channels_last"))
            {
                is_channels_last =
                    (1 == parser.parse_value(info.attributes.at("channels_last")).at<size_t>());
            }
        }

        bool silu_activation = false;
        if(contains(info.attributes, "activation") and is_contrib)
        {
            silu_activation =
                (1 == parser.parse_value(info.attributes.at("activation")).at<size_t>());
        }
        else if(is_contrib)
        {
            MIGRAPHX_THROW("PARSE_GROUPNORM: activation must be available");
        }

        if(args.size() != 3)
        {
            MIGRAPHX_THROW("PARSE_GROUPNORM: invalid input count");
        }

        // Adjust chanels from channels_last-> NCHW if last channel is set for contrib op
        auto x = args.at(0);
        if(is_channels_last and is_contrib)
        {
            x = apply_channels_last_perm(info, x, true);
        }

        auto scale = args.at(1); // gamma in the GroupNorm contrib case
        auto bias  = args.at(2); // beta in the GroupNorm contrib case

        auto x_shape = x->get_shape();
        auto x_dtype = x_shape.type();
        auto x_dims  = x_shape.lens();

        if(x_shape.ndim() <= 2)
        {
            MIGRAPHX_THROW("PARSE_GROUPNORM: invalid input shape");
        }

        auto c = x_shape.lens().at(1);
        if(c % num_groups != 0)
        {
            MIGRAPHX_THROW(
                "PARSE_GROUPNORM: num_groups should be a divisor of the number of channels");
        }
        auto group_size = c / num_groups;
        if(scale->get_shape().ndim() != 1 or scale->get_shape().lens().at(0) != num_groups)
        {
            MIGRAPHX_THROW("PARSE_GROUPNORM: scale tensor shape should be num_groups");
        }
        if(bias->get_shape().ndim() != 1 or bias->get_shape().lens().at(0) != num_groups)
        {
            MIGRAPHX_THROW("PARSE_GROUPNORM: bias tensor shape should be num_groups");
        }

        // Original shape: N x C x D1 x ... x Dn
        // New shape: N x num_groups x C // num_groups x D1 x ... x Dn

        std::vector<size_t> dims = {x_dims.at(0), num_groups, group_size};
        std::copy(x_dims.begin() + 2, x_dims.end(), std::back_inserter(dims));
        auto x_reshaped = info.add_instruction(make_op("reshape", {{"dims", dims}}), x);

        // Axes for D1 x ... x Dn
        std::vector<size_t> axes(dims.size() - 2);
        std::iota(axes.begin(), axes.end(), 2);

        // y = (x - mean) * rsqrt(variance + epsilon) * scale + bias
        // mean = reduce_mean({D1, D2, ... Dk}, x)
        // variance = reduce_mean({D1, D2, ... Dk}, (x - mean)^2)

        auto mean = info.add_instruction(make_op("reduce_mean", {{"axes", axes}}), x_reshaped);
        auto x_sub_mean    = info.add_common_op("sub", x_reshaped, mean);
        auto x_sqdiff_mean = info.add_common_op("sqdiff", x_reshaped, mean);
        auto variance =
            info.add_instruction(make_op("reduce_mean", {{"axes", axes}}), x_sqdiff_mean);
        epsilon =
            (x_dtype == migraphx::shape::half_type and std::abs(epsilon) < 1e-7) ? 1e-7 : epsilon;
        auto eps     = info.add_literal(migraphx::literal{migraphx::shape{x_dtype}, {epsilon}});
        auto var_eps = info.add_common_op("add", variance, eps);
        auto rsqrt   = info.add_instruction(make_op("rsqrt"), var_eps);
        auto result  = info.add_common_op("mul", x_sub_mean, rsqrt);
        auto scale_bcast =
            info.add_instruction(make_op("broadcast", {{"axis", 1}, {"out_lens", dims}}), scale);
        auto bias_bcast =
            info.add_instruction(make_op("broadcast", {{"axis", 1}, {"out_lens", dims}}), bias);
        auto scaled = info.add_instruction(make_op("mul"), result, scale_bcast);
        auto y      = info.add_instruction(make_op("add"), scaled, bias_bcast);
        auto output = info.add_instruction(make_op("reshape", {{"dims", x_dims}}), y);

        // Convert to NCHW -> channels_last for contrib GroupNorm
        if(is_channels_last and is_contrib)
        {
            output = apply_channels_last_perm(info, output, false);
        }
        if(silu_activation)
        {
            // SiLU activation is just  out = x * sigmoid(x)
            auto sigmoid = info.add_instruction(make_op("sigmoid"), output);
            output       = info.add_instruction(make_op("mul"), output, sigmoid);
        }
        return output;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

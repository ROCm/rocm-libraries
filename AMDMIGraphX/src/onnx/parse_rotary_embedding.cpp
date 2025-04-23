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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct rotary_parameters
{
    // Extracted from Inputs
    std::size_t batch_size  = 0; // Batch used by input
    std::size_t seq_len     = 0; // Sequence length used by input
    std::size_t head_size   = 0; // Head size used for offset in each block
    std::size_t max_seq_len = 0; // Sequence length used by sin/cos caches
    std::size_t hidden_size = 0; // Hidden size used by the input
    // input shape:
    // true => [batch_size, num_heads, seq_len, head_size]
    // false => [batch_size, seq_len, hidden_size=num_heads*head_size]
    bool is_bnsh = false;

    // Extracted from both
    std::size_t num_heads = 0;     // num_heads = hidden_size / head_size  or input
    bool head_diff        = false; // head_size > rotary_embedding_dim

    // Extracted from Attributes
    std::size_t rotary_embedding_dim = 0;
    bool interleaved                 = false;
    bool is_packed_batching          = false;
    float scale                      = 1.0;
};

struct parse_rotary_embedding : op_parser<parse_rotary_embedding>
{
    std::vector<op_desc> operators() const { return {{"RotaryEmbedding"}}; }

    static void parse_attributes(const onnx_parser& parser,
                                 const onnx_parser::node_info& info,
                                 rotary_parameters& param)

    {
        if(contains(info.attributes, "interleaved"))
        {
            param.interleaved = parser.parse_value(info.attributes.at("interleaved")).at<bool>();
        }

        if(contains(info.attributes, "is_packed_batching"))
        {
            // Ragged batching is a form of dynamic batching where inputs of varying lengths
            // are padded and batched together. E.g. inputs of [1, 3], [1, 4], [1, 5]
            // would each be padded to [1, 5] and batched together as [3, 5]
            param.is_packed_batching =
                parser.parse_value(info.attributes.at("is_packed_batching")).at<bool>();
            if(param.is_packed_batching)
            {
                MIGRAPHX_THROW(
                    "RotaryEmbedding: is_packed_batching aka ragged batching is not supported.");
            }
        }

        if(contains(info.attributes, "num_heads"))
        {
            param.num_heads = parser.parse_value(info.attributes.at("num_heads")).at<std::size_t>();
        }

        if(contains(info.attributes, "rotary_embedding_dim"))
        {
            param.rotary_embedding_dim =
                parser.parse_value(info.attributes.at("rotary_embedding_dim")).at<std::size_t>();
        }

        if(contains(info.attributes, "scale"))
        {
            // ContribOp spec does not specify how to apply the scale and ORT neither
            // handles nor tests this attribute
            param.scale = parser.parse_value(info.attributes.at("scale")).at<float>();
            if(not float_equal(param.scale, 1.0f))
            {
                MIGRAPHX_THROW("RotaryEmbedding: scale is not supported.");
            }
        }

        if((param.num_heads != 0) xor (param.rotary_embedding_dim != 0))
        {
            MIGRAPHX_THROW("RotaryEmbedding: num_heads and rotary_embedding dims must be used "
                           "together and non-zero");
        }
    }

    static void parse_input(const instruction_ref& input, rotary_parameters& param)
    {
        auto input_lens = input->get_shape().lens();
        auto input_dims = input_lens.size();

        if(input_dims < 3 or input_dims > 4)
        {
            MIGRAPHX_THROW(
                "RotaryEmbedding:Input must be 3D (Batch , Sequence Length, Hidden size) or \
                            4D (Batch, Num Heads, Sequence Length, Head Size))");
        }

        param.batch_size = input_lens.at(0);

        if(input_dims == 3)
        {
            param.seq_len     = input_lens.at(1);
            param.hidden_size = input_lens.at(2);
        }
        else
        {
            param.num_heads   = input_lens.at(1);
            param.seq_len     = input_lens.at(2);
            param.head_size   = input_lens.at(3);
            param.hidden_size = param.num_heads * param.head_size;
            param.is_bnsh     = true;
        }
    }

    // Ensure position ID shapes comply with input dimensions
    static void parse_position_ids(const instruction_ref& position_ids,
                                   const rotary_parameters& param)
    {
        auto position_len = position_ids->get_shape().lens();
        auto position_dim = position_ids->get_shape().lens().size();

        if(position_dim > 2 or position_ids->get_shape().scalar())
        {
            MIGRAPHX_THROW("RotaryEmbedding: Position_ids must be either 1D tensor of shape (1) or "
                           "2d (Batch, Sequence Length)");
        }

        if(position_dim == 1 and position_len.at(0) != 1)
        {
            MIGRAPHX_THROW("RotaryEmbedding: Position_id must have shape of 1 for 1D tensor");
        }

        if((position_dim == 2) and
           ((position_len.at(0) != param.batch_size) or (position_len.at(1) != param.seq_len)))
        {
            MIGRAPHX_THROW("RotaryEmbedding: Position_id 2D dims must match input batch size and "
                           "sequence length");
        }
    }

    static void parse_cos_cache(const instruction_ref& cos_cache, rotary_parameters& param)
    {
        auto cos_cache_len = cos_cache->get_shape().lens();
        param.max_seq_len  = cos_cache_len.at(0);
        if(param.num_heads == 0)
        {
            param.head_size = cos_cache_len.at(1) * 2;
            param.num_heads = param.hidden_size / param.head_size;
        }
        else
        {
            param.head_size = param.hidden_size / param.num_heads;
        }
        if(param.rotary_embedding_dim == 0)
        {
            param.rotary_embedding_dim = param.head_size;
        }
        else if(param.head_size > param.rotary_embedding_dim)
        {
            param.head_diff = true;
        }
        else if(param.head_size < param.rotary_embedding_dim)
        {
            MIGRAPHX_THROW("RotaryEmbedding: rotary_embedding_dim must be <= head_size");
        }

        compare_sin_cos_cache_dims(cos_cache_len.at(1), param);
    }

    static void parse_sin_cache(const instruction_ref& sin_cache, const rotary_parameters& param)
    {
        auto sin_cache_len = sin_cache->get_shape().lens();

        if(param.max_seq_len != sin_cache_len.at(0))
        {
            MIGRAPHX_THROW(
                "RotaryEmbedding: max_sequence_length must be the same between sin & cos caches!");
        }

        compare_sin_cos_cache_dims(sin_cache_len.at(1), param);
    }

    static void compare_sin_cos_cache_dims(const size_t dim, const rotary_parameters& param)
    {
        if(param.rotary_embedding_dim != 0 and param.rotary_embedding_dim / 2 != dim)
        {
            MIGRAPHX_THROW(
                "RotaryEmbedding: rotary_embedding must be the same between sin & cos caches!");
        }
    }

    static void parse_input_args(const std::vector<instruction_ref>& args, rotary_parameters& param)
    {
        // Order matters as we're basing params related to the first input
        parse_input(args.at(0), param);
        parse_position_ids(args.at(1), param);
        parse_cos_cache(args.at(2), param);
        parse_sin_cache(args.at(3), param);
    }

    static instruction_ref apply_rotary_embedding(const instruction_ref& in,
                                                  const instruction_ref& cos,
                                                  const instruction_ref& sin,
                                                  const onnx_parser::node_info& info,
                                                  const rotary_parameters& params)
    {
        if(params.interleaved)
        {
            std::vector<float> signs(params.rotary_embedding_dim);
            for(auto i = 0; i < params.rotary_embedding_dim; ++i)
            {
                signs[i] = (i % 2 == 0) ? -1.0 : 1.0;
            }
            auto signs_lit = info.add_literal(
                literal{shape{in->get_shape().type(), {params.rotary_embedding_dim}}, signs});
            signs_lit = info.add_instruction(
                make_op("multibroadcast", {{"out_lens", in->get_shape().lens()}}), signs_lit);
            auto mul_cos = info.add_broadcastable_binary_op("mul", in, cos);
            auto mul_sin = info.add_broadcastable_binary_op("mul", signs_lit, sin);

            auto rs_in = info.add_instruction(
                make_op("reshape", {{"dims", {in->get_shape().elements() / 2, 2}}}), in);
            auto evens = info.add_instruction(
                make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), rs_in);
            auto odds = info.add_instruction(
                make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), rs_in);
            auto in2 = info.add_instruction(make_op("concat", {{"axis", -1}}), odds, evens);
            in2 = info.add_instruction(make_op("reshape", {{"dims", in->get_shape().lens()}}), in2);
            mul_sin = info.add_broadcastable_binary_op("mul", mul_sin, in2);
            return info.add_broadcastable_binary_op("add", mul_sin, mul_cos);
        }

        auto pos = info.add_instruction(
            make_op("slice",
                    {{"axes", {-1}}, {"starts", {0}}, {"ends", {params.rotary_embedding_dim / 2}}}),
            in);
        auto neg     = info.add_instruction(make_op("slice",
                                                    {{"axes", {-1}},
                                                     {"starts", {params.rotary_embedding_dim / 2}},
                                                     {"ends", {params.rotary_embedding_dim}}}),
                                        in);
        neg          = info.add_instruction(make_op("neg"), neg);
        auto concat  = info.add_instruction(make_op("concat", {{"axis", -1}}), neg, pos);
        auto mul_sin = info.add_broadcastable_binary_op("mul", concat, sin);
        auto mul_cos = info.add_broadcastable_binary_op("mul", in, cos);
        return info.add_broadcastable_binary_op("add", mul_sin, mul_cos);
    }

    static instruction_ref get_cache_slice(const instruction_ref& cache,
                                           const instruction_ref& pos_ids,
                                           const bool interleaved,
                                           const onnx_parser::node_info& info,
                                           const rotary_parameters& params)
    {
        instruction_ref rsps;
        if(pos_ids->get_shape().lens().size() == 1)
        {
            rsps = info.add_instruction(
                make_op("multibroadcast", {{"out_lens", {params.batch_size, params.seq_len, 1}}}),
                pos_ids);
            std::vector<int> pos_vec(params.seq_len);
            std::iota(pos_vec.begin(), pos_vec.end(), 0);
            auto pos_lit = info.add_literal(
                literal{shape{pos_ids->get_shape().type(), {1, params.seq_len, 1}}, pos_vec});
            rsps = info.add_broadcastable_binary_op("add", rsps, pos_lit);
        }
        else
        {
            rsps = info.add_instruction(
                make_op("reshape", {{"dims", {params.batch_size, params.seq_len, 1}}}), pos_ids);
        }
        auto gather = info.add_instruction(make_op("gathernd", {{"batch_dims", 0}}), cache, rsps);

        if(interleaved)
        {
            gather = info.add_instruction(
                make_op("reshape", {{"dims", {gather->get_shape().elements(), 1}}}), gather);
        }

        auto output = info.add_instruction(make_op("concat", {{"axis", -1}}), gather, gather);
        std::vector<std::size_t> out_lens = {
            params.batch_size, params.seq_len, 1, params.rotary_embedding_dim};
        output   = info.add_instruction(make_op("reshape", {{"dims", out_lens}}), output);
        out_lens = {
            params.batch_size, params.seq_len, params.num_heads, params.rotary_embedding_dim};
        output = info.add_instruction(make_op("multibroadcast", {{"out_lens", out_lens}}), output);
        out_lens = {
            params.batch_size, params.num_heads, params.seq_len, params.rotary_embedding_dim};
        return info.add_instruction(make_op("reshape", {{"dims", out_lens}}), output);
    }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       const std::vector<instruction_ref>& args) const
    {
        if(args.size() != 4)
        {
            MIGRAPHX_THROW("RotaryEmbedding: Wrong number of inputs provided require 4");
        }

        // Sanity check input dimension and shapes while extracting params
        rotary_parameters params{};
        parse_attributes(parser, info, params);
        parse_input_args(args, params);

        // Setup based on parsed params gathered from input attributes/inputs
        auto input        = args.at(0);
        auto position_ids = args.at(1);
        auto cos_cache    = args.at(2);
        auto sin_cache    = args.at(3);

        if(not params.is_bnsh)
        {
            input = info.add_instruction(
                make_op(
                    "reshape",
                    {{"dims",
                      {params.batch_size, params.num_heads, params.seq_len, params.head_size}}}),
                input);
        }
        instruction_ref tail;
        if(params.head_diff)
        {
            tail  = info.add_instruction(make_op("slice",
                                                 {{"axes", {-1}},
                                                  {"starts", {params.rotary_embedding_dim}},
                                                  {"ends", {params.head_size}}}),
                                        input);
            input = info.add_instruction(
                make_op("slice",
                        {{"axes", {-1}}, {"starts", {0}}, {"ends", {params.rotary_embedding_dim}}}),
                input);
        }

        auto cos    = get_cache_slice(cos_cache, position_ids, params.interleaved, info, params);
        auto sin    = get_cache_slice(sin_cache, position_ids, params.interleaved, info, params);
        auto output = apply_rotary_embedding(input, cos, sin, info, params);

        if(params.head_diff)
        {
            output = info.add_instruction(make_op("concat", {{"axis", -1}}), output, tail);
        }
        if(not params.is_bnsh)
        {
            output = info.add_instruction(
                make_op("reshape",
                        {{"dims", {params.batch_size, params.seq_len, params.hidden_size}}}),
                output);
        }

        return {output};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

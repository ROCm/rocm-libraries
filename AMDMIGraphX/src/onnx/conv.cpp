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
#include <migraphx/onnx/conv.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/permutation.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

void recalc_conv_attributes(value& v, size_t kdims)
{
    if(v["padding"].size() != kdims and v["padding"].size() != kdims * 2)
    {
        v["padding"].resize(kdims);
        std::fill_n(v["padding"].begin(), kdims, 0);
    }
    if(v["stride"].size() != kdims)
    {
        v["stride"].resize(kdims);
        std::fill_n(v["stride"].begin(), kdims, 1);
    }
    if(v["dilation"].size() != kdims)
    {
        v["dilation"].resize(kdims);
        std::fill_n(v["dilation"].begin(), kdims, 1);
    }
}

static instruction_ref
apply_nhwc_perm(const onnx_parser::node_info& info, instruction_ref ins, bool invert)
{
    std::vector<int64_t> perm(ins->get_shape().ndim());
    std::iota(begin(perm) + 1, end(perm) - 1, 2);
    perm.back() = 1;
    return info.add_instruction(
        make_op("transpose", {{"permutation", invert ? invert_permutation(perm) : perm}}), ins);
}

instruction_ref from_nhwc(const onnx_parser::node_info& info, instruction_ref ins)
{
    return apply_nhwc_perm(info, ins, true);
}

instruction_ref to_nhwc(const onnx_parser::node_info& info, instruction_ref ins)
{
    return apply_nhwc_perm(info, ins, false);
}

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

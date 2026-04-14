/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <miopen/layernorm/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace layernorm {

size_t GetStride(const TensorDescriptor& xDesc, int32_t normalized_dim)
{
    size_t stride = 1;
    auto layout   = xDesc.GetLayoutEnum();
    if(normalized_dim > 1 && layout.has_value() &&
       (layout.value() == miopenTensorNHWC || layout.value() == miopenTensorNDHWC))
    {
        stride = xDesc.GetLengths()[1]; // stride = C
    }
    return stride;
}

size_t GetOuterSize(const TensorDescriptor& xDesc, int32_t normalized_dim, size_t stride)
{
    size_t outer_size = 1;
    for(size_t i = 0; i < normalized_dim; ++i)
    {
        if(!(stride > 1 && i == 1))
        {
            outer_size *= xDesc.GetLengths()[i];
        }
    }
    return outer_size;
}

size_t GetInnerSize(const TensorDescriptor& xDesc, int32_t normalized_dim)
{
    size_t inner_size = 1;
    for(size_t i = normalized_dim; i < xDesc.GetLengths().size(); ++i)
    {
        inner_size *= xDesc.GetLengths()[i];
    }
    return inner_size;
}

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss;

    ss << "dtype" << xDesc.GetType();
    if((mode == MIOPEN_WEIGHT_BIAS_T5) || (mode == MIOPEN_ELEMENTWISE_AFFINE_T5))
    {
        ss << "normalized_dim" << xDesc.GetLengths().size() - 1;
    }
    else
    {
        ss << "normalized_dim" << normalized_dim;
    }
    ss << "outer_size" << outer_size;
    ss << "inner_size" << inner_size;

    if((mode == MIOPEN_WEIGHT_BIAS_FUSED_ADD) || (mode == MIOPEN_ELEMENTWISE_AFFINE_FUSED_ADD))
        ss << "addlayernorm";
    if((mode == MIOPEN_WEIGHT_BIAS_T5) || (mode == MIOPEN_ELEMENTWISE_AFFINE_T5))
        ss << "t5layernorm";

    ss << "stride" << stride;

    return NetworkConfig{ss.str()};
}

} // namespace layernorm

} // namespace miopen

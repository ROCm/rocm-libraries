/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_TENSOR_DRIVER_HPP
#define GUARD_MIOPEN_TENSOR_DRIVER_HPP

#define UNPACK_VEC4(v) (v[0]), (v[1]), (v[2]), (v[3])

#include <common_utils/errors.hpp>
#include <miopen_utils/tensor_desc.hpp>
#include <algorithm>
#include <iterator>
#include <miopen/miopen.h>
#include <common_utils/tensor_layout.hpp>
#include <numeric>
#include <vector>

inline miopenTensorLayout_t StringToLayoutType(std::string layout_str)
{
    miopenTensorLayout_t default_layout = miopenTensorNCHW;
    if(layout_str == "NCHWc4")
        return miopenTensorNCHWc4;
    else if(layout_str == "NCHWc8")
        return miopenTensorNCHWc8;
    else if(layout_str == "CHWNc4")
        return miopenTensorCHWNc4;
    else if(layout_str == "CHWNc8")
        return miopenTensorCHWNc8;
    else if(layout_str == "NCHW")
    {
        return miopenTensorNCHW;
    }
    else if(layout_str == "NHWC")
    {
        return miopenTensorNHWC;
    }
    else if(layout_str == "NDHWC")
    {
        return miopenTensorNDHWC;
    }
    else if(layout_str == "NCDHW")
    {
        return miopenTensorNCDHW;
    }
    else
    {
        COMMON_THROW("We only support NCHWc4, NCHWc8, CHWNc4, CHWNc8, NCHW, NHWC, NDHWC, NCDHW "
                     "vectorized tensor layout.");
        return default_layout;
    }
}

inline void LengthReorder(std::vector<int>& lens, const std::initializer_list<int>& indices)
{
    std::vector<int> out_lens;
    out_lens.reserve(indices.size());
    for(int index : indices)
    {
        assert(0 <= index && index < lens.size());
        out_lens.push_back(std::move(lens[index]));
    }
    lens = std::move(out_lens);
}

inline std::size_t GetTensorVectorLength(const miopenTensorDescriptor_t& tensor)
{
    return TensorDesc::GetVectorLength(tensor);
}

inline std::vector<int> GetTensorLengths(const miopenTensorDescriptor_t& tensor)
{
    auto lens = TensorDesc::GetLengths(tensor);
    return std::vector<int>(lens.begin(), lens.end());
}

inline std::vector<int> GetTensorStrides(const miopenTensorDescriptor_t& tensor)
{
    auto strides = TensorDesc::GetStrides(tensor);
    return std::vector<int>(strides.begin(), strides.end());
}

inline int SetTensor4d(miopenTensorDescriptor_t t,
                       std::vector<int>& len,
                       miopenDataType_t data_type = miopenFloat)
{
    return miopenSet4dTensorDescriptor(t, data_type, UNPACK_VEC4(len));
}

inline int SetTensorNdVector(miopenTensorDescriptor_t t,
                             std::vector<int>& len,
                             miopenTensorLayout_t layout,
                             miopenDataType_t data_type = miopenFloat)
{
    if(layout == miopenTensorNCHWc4 || layout == miopenTensorNCHWc8 || layout == miopenTensorNHWC ||
       layout == miopenTensorNCHW || layout == miopenTensorNDHWC || layout == miopenTensorNCDHW)
    {
        // Do nothing, MIOpen implicit logic that lens are in NCHW order.
    }
    else if(layout == miopenTensorCHWNc4 || layout == miopenTensorCHWNc8)
    {
        LengthReorder(len, {1, 2, 3, 0});
    }
    else
    {
        COMMON_THROW("We only support NCHWc4, NCHWc8, CHWNc4, CHWNc8, NCHW, NHWC, NDHWC, NCDHW "
                     "vectorized tensor layout.");
        return -1;
    }
    return miopenSetNdTensorDescriptorWithLayout(t, data_type, layout, len.data(), len.size());
}

inline int SetTensorNd(miopenTensorDescriptor_t t,
                       std::vector<int>& len,
                       miopenDataType_t data_type = miopenFloat)
{
    return miopenSetTensorDescriptor(t, data_type, len.size(), len.data(), nullptr);
}

inline int SetTensorNd(miopenTensorDescriptor_t t,
                       std::vector<std::size_t>& len,
                       miopenDataType_t data_type = miopenFloat)
{
    return miopenSetTensorDescriptorV2(t, data_type, len.size(), len.data(), nullptr);
}

inline int SetTensorNd(miopenTensorDescriptor_t t,
                       std::vector<int>& len,
                       std::vector<int>& strides,
                       miopenDataType_t data_type = miopenFloat)
{
    return miopenSetTensorDescriptor(t, data_type, len.size(), len.data(), strides.data());
}

inline int SetTensorNd(miopenTensorDescriptor_t t,
                       std::vector<std::size_t>& len,
                       std::vector<std::size_t>& strides,
                       miopenDataType_t data_type = miopenFloat)
{
    return miopenSetTensorDescriptorV2(t, data_type, len.size(), len.data(), strides.data());
}

inline int SetTensorNd(miopenTensorDescriptor_t t,
                       std::vector<int>& len,
                       const std::string& layout,
                       miopenDataType_t data_type = miopenFloat)
{
    if(layout.empty())
    {
        return SetTensorNd(t, len, data_type);
    }

    if(layout.size() != len.size() && layout.find('c') == std::string::npos)
    {
        COMMON_THROW("unmatched layout and dimension size");
    }

    if(layout.find('c') != std::string::npos)
    {
        return SetTensorNdVector(t, len, StringToLayoutType(layout), data_type);
    }

    // Dimension lengths vector 'len' comes with a default layout.
    std::string len_layout = miopen::tensor_layout_get_default(layout.size());
    if(len_layout.empty())
    {
        return SetTensorNd(t, len, data_type);
    }

    std::vector<std::size_t> strides2;
    std::vector<std::size_t> len2(len.cbegin(), len.cend());
    miopen::tensor_layout_to_strides(len2, len_layout, layout, strides2);
    return SetTensorNd(t, len2, strides2, data_type);
}

// This function ignores tensor strides completely and its result should not be interpreted as
// memory required for an "unpacked" tensor. In such cases GetTensorSpace should be used instead.
// For "packed" tensors result may be interpreted as an amount of memory required for the tensor.
// The implementation is a copy-paste from miopen::TensorDescriptor.
inline size_t GetTensorSize(const miopenTensorDescriptor_t& tensor)
{
    assert(TensorDesc::IsPacked(tensor) &&
           "GetTensorSize should not be used on an unpacked tensor.");
    const auto len            = GetTensorLengths(tensor);
    const size_t vectorLength = GetTensorVectorLength(tensor);
    size_t sz = std::accumulate(len.begin(), len.end(), vectorLength, std::multiplies<size_t>());

    return sz;
}

// The result of this function may be interpreted as a correct amount of memory required for both
// "packed" and "unpacked" tensors. In general it should be used for such purposes rather than
// GetTensorSize. Unless, of course, there is absolutely zero chance to receive an unpacked tensor.
inline size_t GetTensorSpace(const miopenTensorDescriptor_t& tensor)
{
    return TensorDesc::GetElementSpace(tensor);
}

#endif // GUARD_MIOPEN_TENSOR_DRIVER_HPP

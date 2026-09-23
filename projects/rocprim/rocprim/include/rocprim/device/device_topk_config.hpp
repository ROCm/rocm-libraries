// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_DEVICE_DEVICE_TOPK_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_TOPK_CONFIG_HPP_

#include "config_types.hpp"
#include "detail/config/device_topk_air.hpp"
#include "detail/device_config_helper.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Key, class Value, class SizeIn>
struct topk_air_config_selector
{
    using targets    = topk_air_targets;
    using param_type = topk_air_config_params;

    param_type params;

    template<class Target>
    constexpr topk_air_config_selector(Target)
        : params(topk_air_config_picker<Target, Key, Value, SizeIn>())
    {}
};

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_TOPK_CONFIG_HPP_

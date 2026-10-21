// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_FIND_IF_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_FIND_IF_CONFIG_HPP_

#include "config_types.hpp"
// #include "detail/config/device_find_if.hpp" TODO: add
#include "detail/device_config_helper.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Type>
struct find_if_config_selector
{
    // using targets    = find_if_targets; TODO: change
    using targets
        = comp_targets<comp_target<gen::cdna2, target_arch::gfx90a, gpu::mi210, rep::amdgcn>,
                       comp_target<gen::cdna1, target_arch::gfx908, gpu::mi100, rep::amdgcn>,
                       comp_target<gen::cdna3, target_arch::gfx942, gpu::mi300x, rep::amdgcn>,
                       comp_target<gen::rdna3, target_arch::gfx1100, gpu::rx7900, rep::amdgcn>,
                       comp_target<gen::rdna4, target_arch::gfx1201, gpu::rx9070, rep::amdgcn>,
                       comp_target<gen::unknown, target_arch::unknown, gpu::generic, rep::amdgcn>>;
    using param_type = find_if_config_params;

    param_type params;

    template<class Target>
    constexpr find_if_config_selector(Target)
        : params(find_if_config_params{
              {256, 10}
    }) // params(find_if_config_picker<Target, Type>()) TODO: change
    {}
};

} // namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_FIND_IF_CONFIG_HPP_

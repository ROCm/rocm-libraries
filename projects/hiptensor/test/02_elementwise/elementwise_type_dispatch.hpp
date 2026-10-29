/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <initializer_list>
#include <ostream>
#include <type_traits>
#include <utility>

#include "data_types.hpp"
#include "elementwise_resource.hpp"
#include "utils.hpp"

namespace hiptensor
{
    // Calls `func` with a value of the element type that `type` names. Returns false, without
    // calling it, for a type the elementwise tests do not handle.
    template <typename Func>
    bool dispatchElementwiseType(hiptensorDataType_t type, Func&& func)
    {
        switch(type)
        {
        case HIPTENSOR_R_16F:
            func(_Float16{});
            return true;
        case HIPTENSOR_R_16BF:
            func(hip_bfloat16{});
            return true;
        case HIPTENSOR_R_32F:
            func(float{});
            return true;
        case HIPTENSOR_R_64F:
            func(double{});
            return true;
        default:
            return false;
        }
    }

    // Stores `value` in `storage` with the representation of `type`. The 8-byte storage holds any
    // of the scalar types, so its address can be passed as the API's alpha/beta/gamma pointer.
    inline bool writeElementwiseScalar(double& storage, hiptensorDataType_t type, double value)
    {
        return dispatchElementwiseType(type, [&](auto tag) {
            using T = decltype(tag);
            if constexpr(std::is_same_v<T, hip_bfloat16>)
            {
                *reinterpret_cast<T*>(&storage) = static_cast<T>(static_cast<float>(value));
            }
            else
            {
                *reinterpret_cast<T*>(&storage) = static_cast<T>(value);
            }
        });
    }

    // Compares the device output with the device copy of the CPU reference, elementwise.
    // An unhandled type reports a failed comparison.
    inline std::pair<bool, double> compareElementwiseOutputWithReference(
        ElementwiseResource* resource, hiptensorDataType_t type, hiptensorDataType_t computeType)
    {
        std::pair<bool, double> result{false, 0.0};
        dispatchElementwiseType(type, [&](auto tag) {
            using T = decltype(tag);
            result  = compareEqualLaunchKernel<T>(static_cast<T*>(resource->deviceOutput().get()),
                                                 static_cast<T*>(resource->deviceReference().get()),
                                                 resource->getCurrentMatrixElement(),
                                                 convertToComputeType(computeType));
        });
        return result;
    }

    struct ElementwiseTensorToPrint
    {
        char const* name;
        void*       data;
    };

    // Prints each host tensor as `type`, all with `count` elements.
    inline void printElementwiseTensors(std::ostream&                                   stream,
                                        hiptensorDataType_t                             type,
                                        std::size_t                                     count,
                                        std::initializer_list<ElementwiseTensorToPrint> tensors)
    {
        dispatchElementwiseType(type, [&](auto tag) {
            using T = decltype(tag);
            for(auto const& tensor : tensors)
            {
                stream << "Tensor " << tensor.name << " elements (" << count << "):\n";
                hiptensorPrintArrayElements<T>(stream, static_cast<T*>(tensor.data), count);
                stream << std::endl;
            }
        });
    }
} // namespace hiptensor

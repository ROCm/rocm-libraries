/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCWMMA_DEVICE_CROSS_LANE_OPS_UTIL_HPP
#define ROCWMMA_DEVICE_CROSS_LANE_OPS_UTIL_HPP

#include <rocwmma/rocwmma.hpp>

namespace rocwmma
{
    constexpr uint32_t VALUE_OUT_OF_RANGE = 100; // 100 is out of [0, SubGroupSize]

    /**
	 * @defgroup cross_lane_op_gen_ref_value_funcs Reference values generators of cross-lane ops
	 *
	 * This group contains functions related to generate reference values of cross-lane ops.
	 *
	 * All functions in this group share the following properties:
	 * - The parameter input is set to threadIdx.x
	 */
    template <typename DataT>
    ROCWMMA_DEVICE inline DataT makeValueFromU32(uint32_t input)
    {
        static_assert(std::is_same_v<uint32_t, DataT> || std::is_same_v<uint64_t, DataT>,
                      "DataT must be uint32_t or uint64_t. We only test these 2 types");
        if constexpr(std::is_same_v<uint64_t, DataT>)
        {
            uint64_t output = input;
            output          = output << 32 | input;
            return output;
        }
        else
        {
            return input;
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_CROSS_LANE_OPS_UTIL_HPP

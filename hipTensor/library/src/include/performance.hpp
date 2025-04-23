/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#ifndef HIPTENSOR_PERFORMANCE_HPP
#define HIPTENSOR_PERFORMANCE_HPP

#include <ostream>
#include <string>

namespace hiptensor
{
    struct PerfMetrics
    {
        std::size_t mKernelUid; /*!< The unique ID of the kernel */
        std::string mKernelName; /*!< String name of the kernel */
        float       mAvgTimeMs; /*!< Avg kernel runtime in milli-seconds */
        float       mTflops; /*!< Calculation throughput in Tflop per second */
        float       mBandwidth; /*!< Data throughput in GB per second */

        bool operator>(PerfMetrics const& other) const;
        bool operator<(PerfMetrics const& other) const;
        bool operator>=(PerfMetrics const& other) const;
        bool operator<=(PerfMetrics const& other) const;
        bool operator==(PerfMetrics const& other) const;
    };

} // namespace hiptensor

namespace std
{
    ostream& operator<<(std::ostream& os, hiptensor::PerfMetrics const& metrics);
} // namespace std

#endif // HIPTENSOR_PERFORMANCE_HPP

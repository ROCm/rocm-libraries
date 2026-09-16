/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#pragma once

#include "utility.hpp"
#include <algorithm>
#include <numeric>
#include <vector>

// function to collect and combine benchmark times
class hipsolver_timer
{
public:
    typedef enum average_type_
    {
        median,
        mean
    } average_type;

    void reset()
    {
        m_times.clear();
    }

    void push(double time)
    {
        m_times.push_back(time);
    }

    void start(hipStream_t stream)
    {
        m_start_time = get_time_us_sync(stream);
    }

    double end(hipStream_t stream)
    {
        double t = get_time_us_sync(stream) - m_start_time;
        m_times.push_back(t);
        return t;
    }

    double get_combined(average_type avg = median)
    {
        const auto n = m_times.size();
        if(n == 0)
            return 0;

        switch(avg)
        {
        case median:
        {
            const auto mid = n / 2;
            std::sort(m_times.begin(), m_times.end());
            return n % 2 == 0 ? (m_times[mid - 1] + m_times[mid]) / 2 : m_times[mid];
        }
        case mean:
        {
            const auto sum = std::accumulate(m_times.begin(), m_times.end(), 0.0);
            return sum / n;
        }
        }

        return 0;
    }

private:
    std::vector<double> m_times;
    double              m_start_time;
};

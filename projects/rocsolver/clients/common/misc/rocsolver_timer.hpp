/* **************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "common_host_helpers.hpp"
#include <algorithm>
#include <hip/hip_runtime.h>
#include <numeric>
#include <vector>

// function to collect and combine benchmark times
class rocsolver_timer
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

    void end(hipStream_t stream)
    {
        m_times.push_back(get_time_us_sync(stream) - m_start_time);
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
    double m_start_time;
};

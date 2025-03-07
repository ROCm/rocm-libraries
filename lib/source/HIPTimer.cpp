/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

/*
 * Timer for HIP events.
 */

#include <rocRoller/Utilities/HIPTimer.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    HIPTimer::HIPTimer(std::string name)
        : HIPTimer(std::move(name), 1, 0)
    {
    }

    HIPTimer::HIPTimer(std::string name, int n)
        : HIPTimer(std::move(name), n, 0)
    {
    }

    HIPTimer::HIPTimer(std::string name, int n, hipStream_t stream)
        : Timer(std::move(name))
        , m_hipStart(n)
        , m_hipStop(n)
        , m_hipStream(stream)
    {
        for(int i = 0; i < n; i++)
        {
            HIP_CHECK(hipEventCreateWithFlags(&m_hipStart[i], hipEventDefault));
            HIP_CHECK(hipEventCreateWithFlags(&m_hipStop[i], hipEventDefault));
        }
    }

    HIPTimer::~HIPTimer()
    {
        for(int i = 0; i < m_hipStart.size(); i++)
        {
            HIP_CHECK(hipEventDestroy(m_hipStart[i]));
            HIP_CHECK(hipEventDestroy(m_hipStop[i]));
        }
    }

    void HIPTimer::tic()
    {
        tic(0);
    }

    void HIPTimer::tic(int i)
    {
        HIP_CHECK(hipEventRecord(m_hipStart[i], m_hipStream));
    }

    void HIPTimer::toc()
    {
        toc(0);
    }

    void HIPTimer::toc(int i)
    {
        HIP_CHECK(hipEventRecord(m_hipStop[i], m_hipStream));
    }

    void HIPTimer::sync()
    {
        HIP_CHECK(hipEventSynchronize(m_hipStop.back()));
        m_hipElapsedTime.clear();
        for(int i = 0; i < m_hipStart.size(); i++)
        {
            float elapsed = 0.f;

            HIP_CHECK(hipEventElapsedTime(&elapsed, m_hipStart[i], m_hipStop[i]));

            m_hipElapsedTime.push_back(std::chrono::round<std::chrono::steady_clock::duration>(
                std::chrono::duration<float, std::milli>(elapsed)));

            m_elapsed += m_hipElapsedTime[i];
        }

        TimerPool::getInstance().accumulate(m_name, m_elapsed);

        m_start = {};
    }

    void HIPTimer::sleep(int sleepPercentage) const
    {
        if(sleepPercentage > 0)
        {
            auto sleepTime = m_elapsed * (sleepPercentage / 100.0);

            std::this_thread::sleep_for(sleepTime);
        }
    }

    std::vector<size_t> HIPTimer::allNanoseconds() const
    {
        std::vector<size_t> result(m_hipElapsedTime.size());

        for(int i = 0; i < result.size(); i++)
        {
            result[i]
                = std::chrono::duration_cast<std::chrono::nanoseconds>(m_hipElapsedTime[i]).count();
        }

        return result;
    }

    hipStream_t HIPTimer::stream() const
    {
        return m_hipStream;
    }
}

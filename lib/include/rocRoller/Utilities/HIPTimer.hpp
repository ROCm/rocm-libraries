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

#pragma once

#ifdef ROCROLLER_USE_HIP

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <rocRoller/Utilities/Timer.hpp>

#include <hip/hip_runtime.h>

#define HIP_TIMER(V, NAME, NUM) auto V = std::make_shared<rocRoller::HIPTimer>(NAME, NUM);
#define HIP_TIC(V, I) V->tic(I);
#define HIP_TOC(V, I) V->toc(I);
#define HIP_SYNC(V) V->sync();

namespace rocRoller
{
    class HIPTimer : public Timer
    {
    public:
        HIPTimer() = delete;
        explicit HIPTimer(std::string name);
        HIPTimer(std::string name, int n);
        HIPTimer(std::string name, int n, hipStream_t stream);
        virtual ~HIPTimer();

        /**
         * Start the timer.
         */
        void tic() override;
        void tic(int i);

        /**
         * Stop the timer.
         */
        void toc() override;
        void toc(int i);

        /**
         * Accumulate the total elapsed time.
         */
        void sync();

        /**
         * Sleep for elapsedTime * (sleepPercentage/100.0)
        */
        void sleep(int sleepPercentage) const;

        /**
         * Return the stream used by the timer
        */
        hipStream_t stream() const;

        /**
         * Returns a vector of all of the elapsed times in nanoseconds.
        */
        std::vector<size_t> allNanoseconds() const;

    private:
        std::vector<hipEvent_t>                          m_hipStart;
        std::vector<hipEvent_t>                          m_hipStop;
        std::vector<std::chrono::steady_clock::duration> m_hipElapsedTime;
        hipStream_t                                      m_hipStream;
    };
}

#else
#define HIP_TIMER(V, N)
#define HIP_TIC(V)
#define HIP_TOC(V)
#define HIP_SYNC(V)
#endif

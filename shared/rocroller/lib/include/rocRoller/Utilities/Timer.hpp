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
 * Timer for microbenchmarking rocRoller.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <string>

#ifdef ROCROLLER_ENABLE_TIMERS
#define TIMER(V, N) rocRoller::Timer V(N);
#define TIC(V) V.tic();
#define TOC(V) V.toc();
#else
#define TIMER(V, N)
#define TIC(V)
#define TOC(V)
#endif

namespace rocRoller
{

    /**
     * TimerPool - a singleton pool of elapsed timing results.
     *
     * Elapsed times are accumulated (atomically) by name.  Times are
     * accumulated by the rocRoller::Timer class below.
     */
    class TimerPool
    {
    public:
        TimerPool(TimerPool const&) = delete;
        void operator=(TimerPool const&) = delete;

        static TimerPool& getInstance();

        /**
         * Summary of all elapsed times.
         */
        static std::string summary();
        static std::string CSV();

        /**
         * Total elapsed nanoseconds of timer `name`.
         */
        static size_t nanoseconds(std::string const& name);

        /**
         * Total elapsed milliseconds of timer `name`.
         */
        static size_t milliseconds(std::string const& name);

        /**
         * @brief clears the timer pool
         *
         */
        static void clear();

        /**
         * Accumulate time into timer `name`.  Atomic.
         */
        void accumulate(std::string const&                         name,
                        std::chrono::steady_clock::duration const& elapsed);

    private:
        TimerPool() {}

        std::map<std::string, std::atomic<std::chrono::steady_clock::duration>> m_elapsed;
    };

    /**
     * Timer - a simple (monotonic) timer.
     *
     * The timer is started by calling tic().  The timer is stopped
     * (and the total elapsed time is accumulated) by calling toc().
     *
     * The timer is automatically started upon construction, and
     * stopped upon destruction.
     *
     * When a timer is stopped (via toc()) the elapsed time is added
     * to the TimerPool.
     */
    class Timer
    {
    public:
        Timer() = delete;
        explicit Timer(std::string name);
        virtual ~Timer();

        /**
         * Start the timer.
         */
        virtual void tic();

        /**
         * Stop the timer and accumulate the total elapsed time.
         */
        virtual void toc();

        /**
         * Get the elapsed time of this timer.
         */
        std::chrono::steady_clock::duration elapsed() const;

        /**
         * Get the elapsed time of this timer in nanoseconds.
         */
        size_t nanoseconds() const;

        /**
         * Get the elapsed time of this timer in milliseconds.
         */
        size_t milliseconds() const;

    protected:
        std::string                           m_name;
        std::chrono::steady_clock::time_point m_start;
        std::chrono::steady_clock::duration   m_elapsed;
    };

}

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
        Timer(std::string name);
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

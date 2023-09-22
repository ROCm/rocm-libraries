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

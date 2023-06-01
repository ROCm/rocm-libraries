/*
 * Timer for HIP events.
 */

#pragma once

#ifdef ROCROLLER_USE_HIP

#include <chrono>
#include <map>
#include <memory>
#include <string>

#include <rocRoller/Utilities/Timer.hpp>

#include <hip/hip_runtime.h>

#define HIP_TIMER(V, N) rocRoller::HIPTimer V(N);
#define HIP_TIC(V) V.tic();
#define HIP_TOC(V) V.toc();
#define HIP_SYNC(V) V.sync();

namespace rocRoller
{
    class HIPTimer : public Timer
    {
    public:
        HIPTimer() = delete;
        explicit HIPTimer(std::string name);
        HIPTimer(std::string name, hipStream_t stream);
        virtual ~HIPTimer();

        /**
         * Start the timer.
         */
        void tic() override;

        /**
         * Stop the timer.
         */
        void toc() override;

        /**
         * Accumulate the total elapsed time.
         */
        void sync();

    private:
        hipEvent_t  m_hipStart, m_hipStop;
        hipStream_t m_hipStream;
    };
}

#else
#define HIP_TIMER(V, N)
#define HIP_TIC(V)
#define HIP_TOC(V)
#define HIP_SYNC(V)
#endif

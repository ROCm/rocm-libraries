/*
 * Timer for HIP events.
 */

#pragma once

#ifdef ROCROLLER_USE_HIP

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <string>

#include <hip/hip_runtime.h>

#define HIP_TIMER(V, N) rocRoller::HIPTimer V(N);
#define HIP_TIC(V) V.tic();
#define HIP_TOC(V) V.toc();
#define HIP_SYNC(V) V.sync();

namespace rocRoller
{
    class HIPTimer
    {
    public:
        HIPTimer() = delete;
        HIPTimer(std::string name);
        HIPTimer(std::string name, hipStream_t stream);
        ~HIPTimer();

        /**
         * Start the timer.
         */
        void tic();

        /**
         * Stop the timer.
         */
        void toc();

        /**
         * Accumulate the total elapsed time.
         */
        void sync();

    private:
        std::string                         m_name;
        hipEvent_t                          m_start, m_stop;
        hipStream_t                         m_stream;
        std::chrono::steady_clock::duration m_elapsed;
    };
}

#else
#define HIP_TIMER(V, N)
#define HIP_TIC(V)
#define HIP_TOC(V)
#define HIP_SYNC(V)
#endif

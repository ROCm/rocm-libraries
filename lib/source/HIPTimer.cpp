/*
 * Timer for HIP events.
 */

#include <rocRoller/Utilities/HIPTimer.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    HIPTimer::HIPTimer(std::string name)
        : HIPTimer(name, 0)
    {
    }

    HIPTimer::HIPTimer(std::string name, hipStream_t stream)
        : Timer(name)
        , m_hip_stream(stream)
    {
        HIP_CHECK(hipEventCreate(&m_hip_start));
        HIP_CHECK(hipEventCreate(&m_hip_stop));

        tic();
    }

    HIPTimer::~HIPTimer()
    {
        HIP_CHECK(hipEventDestroy(m_hip_start));
        HIP_CHECK(hipEventDestroy(m_hip_stop));
    }

    void HIPTimer::tic()
    {
        HIP_CHECK(hipEventRecord(m_hip_start, m_hip_stream));
    }

    void HIPTimer::toc()
    {
        HIP_CHECK(hipEventRecord(m_hip_stop, m_hip_stream));
    }

    void HIPTimer::sync()
    {
        float elapsed = 0.f;

        HIP_CHECK(hipEventSynchronize(m_hip_start));
        HIP_CHECK(hipEventSynchronize(m_hip_stop));
        HIP_CHECK(hipEventElapsedTime(&elapsed, m_hip_start, m_hip_stop));

        m_elapsed = std::chrono::round<std::chrono::steady_clock::duration>(
            std::chrono::duration<float, std::milli>(elapsed));

        TimerPool::getInstance().accumulate(m_name, m_elapsed);
    }
}

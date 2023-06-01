/*
 * Timer for HIP events.
 */

#include <rocRoller/Utilities/HIPTimer.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    HIPTimer::HIPTimer(std::string name)
        : HIPTimer(std::move(name), 0)
    {
    }

    HIPTimer::HIPTimer(std::string name, hipStream_t stream)
        : Timer(std::move(name))
        , m_hipStream(stream)
    {
        HIP_CHECK(hipEventCreate(&m_hipStart));
        HIP_CHECK(hipEventCreate(&m_hipStop));
    }

    HIPTimer::~HIPTimer()
    {
        HIP_CHECK(hipEventDestroy(m_hipStart));
        HIP_CHECK(hipEventDestroy(m_hipStop));
    }

    void HIPTimer::tic()
    {
        HIP_CHECK(hipEventRecord(m_hipStart, m_hipStream));
    }

    void HIPTimer::toc()
    {
        HIP_CHECK(hipEventRecord(m_hipStop, m_hipStream));
    }

    void HIPTimer::sync()
    {
        float elapsed = 0.f;

        HIP_CHECK(hipEventSynchronize(m_hipStart));
        HIP_CHECK(hipEventSynchronize(m_hipStop));
        HIP_CHECK(hipEventElapsedTime(&elapsed, m_hipStart, m_hipStop));

        m_elapsed = std::chrono::round<std::chrono::steady_clock::duration>(
            std::chrono::duration<float, std::milli>(elapsed));

        TimerPool::getInstance().accumulate(m_name, m_elapsed);

        m_start = {};
    }
}

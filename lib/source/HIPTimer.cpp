/*
 * Timer for HIP events.
 */

#include <rocRoller/Utilities/HIPTimer.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    HIPTimer::HIPTimer(std::string name)
        : m_name(name)
        , m_elapsed(0)
        , m_stream(0)
    {
        HIP_CHECK(hipEventCreate(&m_start));
        HIP_CHECK(hipEventCreate(&m_stop));

        tic();
    }

    HIPTimer::HIPTimer(std::string name, hipStream_t stream)
        : m_name(name)
        , m_elapsed(0)
        , m_stream(stream)
    {
        HIP_CHECK(hipEventCreate(&m_start));
        HIP_CHECK(hipEventCreate(&m_stop));

        tic();
    }

    HIPTimer::~HIPTimer()
    {
        HIP_CHECK(hipEventDestroy(m_start));
        HIP_CHECK(hipEventDestroy(m_stop));
    }

    void HIPTimer::tic()
    {
        HIP_CHECK(hipEventRecord(m_start, m_stream));
    }

    void HIPTimer::toc()
    {
        HIP_CHECK(hipEventRecord(m_stop, m_stream));
    }

    void HIPTimer::sync()
    {
        float elapsed = 0.f;

        HIP_CHECK(hipEventSynchronize(m_start));
        HIP_CHECK(hipEventSynchronize(m_stop));
        HIP_CHECK(hipEventElapsedTime(&elapsed, m_start, m_stop));

        m_elapsed = std::chrono::round<std::chrono::steady_clock::duration>(
            std::chrono::duration<float, std::milli>(elapsed));

        TimerPool::getInstance().accumulate(m_name, m_elapsed);
    }
}

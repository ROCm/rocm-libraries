/*
 * Timer for microbenchmarking the translator.
 */

#include <iomanip>
#include <sstream>

#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    TimerPool& TimerPool::getInstance()
    {
        static TimerPool pool;
        return pool;
    }

    std::string TimerPool::summary()
    {
        std::stringstream ss;
        for(auto& kv : getInstance().m_elapsed)
        {
            ss << std::left << std::setw(60) << kv.first << std::right << std::setw(9)
               << milliseconds(kv.first) << "ms" << std::endl;
        }
        return ss.str();
    }

    std::string TimerPool::CSV()
    {
        std::stringstream ss;
        ss << "timer,nanoseconds" << std::endl;
        for(auto& kv : getInstance().m_elapsed)
        {
            ss << kv.first << "," << nanoseconds(kv.first) << std::endl;
        }
        return ss.str();
    }

    size_t TimerPool::nanoseconds(std::string const& name)
    {
        auto elapsed = getInstance().m_elapsed.at(name).load();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    }

    size_t TimerPool::milliseconds(std::string const& name)
    {
        auto elapsed = getInstance().m_elapsed.at(name).load();
        return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    }

    void TimerPool::accumulate(std::string const&                         name,
                               std::chrono::steady_clock::duration const& elapsed)
    {
        m_elapsed[name].store(m_elapsed[name].load() + elapsed);
    }

    Timer::Timer(std::string name)
        : m_name(name)
        , m_elapsed(0)
    {
        tic();
    }

    Timer::~Timer()
    {
        toc();
    }

    void Timer::tic()
    {
        m_start = std::chrono::steady_clock::now();
    }

    void Timer::toc()
    {
        if(m_start.time_since_epoch().count() <= 0)
            return;

        auto elapsed = std::chrono::steady_clock::now() - m_start;
        m_start      = {};

        m_elapsed += elapsed;
        TimerPool::getInstance().accumulate(m_name, elapsed);
    }

    std::chrono::steady_clock::duration Timer::elapsed() const
    {
        return m_elapsed;
    }

    size_t Timer::nanoseconds() const
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(m_elapsed).count();
    }

    size_t Timer::milliseconds() const
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(m_elapsed).count();
    }
}

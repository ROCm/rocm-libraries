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
        for(auto const& kv : getInstance().m_elapsed)
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
        for(auto const& kv : getInstance().m_elapsed)
        {
            ss << kv.first << "," << nanoseconds(kv.first) << std::endl;
        }
        return ss.str();
    }

    size_t TimerPool::nanoseconds(std::string const& name)
    {
        if(getInstance().m_elapsed.count(name) == 0)
            return 0;
        auto elapsed = getInstance().m_elapsed.at(name).load();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    }

    size_t TimerPool::milliseconds(std::string const& name)
    {
        if(getInstance().m_elapsed.count(name) == 0)
            return 0;
        auto elapsed = getInstance().m_elapsed.at(name).load();
        return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    }

    void TimerPool::clear()
    {
        getInstance().m_elapsed.clear();
    }

    void TimerPool::accumulate(std::string const&                         name,
                               std::chrono::steady_clock::duration const& elapsed)
    {
        m_elapsed[name].store(m_elapsed[name].load() + elapsed);
    }

    Timer::Timer(std::string name)
        : m_name(std::move(name))
        , m_elapsed(0)
    {
        Timer::tic();
    }

    Timer::~Timer()
    {
        Timer::toc();
    }

    void Timer::tic()
    {
        m_start = std::chrono::steady_clock::now();
    }

    void Timer::toc()
    {
        if(m_start.time_since_epoch().count() <= 0)
            return;

        auto elapsedTime = std::chrono::steady_clock::now() - m_start;
        m_start          = {};

        m_elapsed += elapsedTime;
        TimerPool::getInstance().accumulate(m_name, elapsedTime);
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

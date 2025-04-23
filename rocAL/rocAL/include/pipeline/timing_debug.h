/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once
#include <chrono>
#include <iostream>
#include <string>
#include <utility>

#include "pipeline/commons.h"

#define DEFAULT_DBG_TIMING 1
/*! \brief Debugging RocalDbgTiming class
 *
 * Can be used anywhere in the code for adding RocalDbgTiming for debugging and profiling
 */
class TimingDbg {
   public:
    //! Constrcutor
    /*!
    \param name Name of the timer,
    \param enable enables the timer module, if not set, timer is disabled
    */
    explicit TimingDbg(std::string name, bool enable = DEFAULT_DBG_TIMING) : _accumulated_time(_t_start - _t_start),
                                                                             _count(0),
                                                                             _enable(enable),
                                                                             _name(std::move(name)) {}

    //! Starts the timer
    inline void start() {
        if (!_enable)
            return;

        _t_start = std::chrono::high_resolution_clock::now();
    }

    //! Stops the timer
    inline void end() {
        if (!_enable)
            return;

        std::chrono::high_resolution_clock::time_point t_end =
            std::chrono::high_resolution_clock::now();

        if (_t_start < t_end) {
            _instantaneous_time = t_end - _t_start;
            _accumulated_time = _accumulated_time + _instantaneous_time;
            _count++;
        }
    }

    //! Prints total elapsed time
    unsigned long long get_timing() {
        if (!_enable)
            return 0;

        auto dur = static_cast<long long unsigned>(std::chrono::duration_cast<std::chrono::microseconds>(_accumulated_time).count());
        if (_count > 0)
            LOG(_name + " ran " + TOSTR(_count) + " times with average duration " + TOSTR((dur / _count) / 1000000) + " sec " + TOSTR((dur / _count) % 1000000) + " us ")
        _count = 0;
        _accumulated_time = _t_start - _t_start;
        return dur;
    }

    //! Returns last timing
    unsigned long long get_last_timing() {
        if (!_enable)
            return 0;
        auto dur = static_cast<long long unsigned>(std::chrono::duration_cast<std::chrono::microseconds>(_accumulated_time).count());
        return dur;
    }

    unsigned count() {
        return _count;
    }

   private:
    std::chrono::high_resolution_clock::time_point _t_start;
    std::chrono::duration<double, std::micro> _accumulated_time = _t_start - _t_start;
    std::chrono::duration<double, std::micro> _instantaneous_time = _t_start - _t_start;
    unsigned _count;
    const bool _enable;
    std::string _name;
};
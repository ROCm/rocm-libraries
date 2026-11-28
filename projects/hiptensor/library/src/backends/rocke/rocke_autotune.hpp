/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

// Measured selection of the rocKE backend (HIPTENSOR_ALGO_DEFAULT_PATIENT): the selector's
// first candidates run on the device at plan creation and the fastest one is kept.
//
// The timing allocates the problem's tensors and the largest workspace of the timed candidates
// (initialized: every byte 0x3C, finite values in every data type), runs each candidate once,
// then interleaves the timed repetitions (hipEvent pairs on a private non-blocking stream) and
// takes the median. Nothing of it stays allocated after plan creation.

#ifndef HIPTENSOR_BACKENDS_ROCKE_ROCKE_AUTOTUNE_HPP
#define HIPTENSOR_BACKENDS_ROCKE_ROCKE_AUTOTUNE_HPP

#include <string>
#include <vector>

#include "rocke_plan.hpp"

namespace hiptensor
{
    namespace rockebackend
    {
        struct AutotuneResult
        {
            // index into the candidates, -1 when nothing could be timed
            int winner = -1;
            // per candidate: the median time in microseconds, negative when it was not timed
            // (or failed)
            std::vector<double> medianUs;
            // why the timing did not run, or the candidates that failed
            std::string note;
        };

        // Times candidates[i] for i in `indices` (their functions are loaded on the current
        // device first) with `reps` timed repetitions each.
        hiptensorStatus_t autotune(backend::Problem const& problem,
                                   std::vector<Candidate>& candidates,
                                   std::vector<int> const& indices,
                                   int                     reps,
                                   AutotuneResult&         result);
    } // namespace rockebackend
} // namespace hiptensor

#endif // HIPTENSOR_BACKENDS_ROCKE_ROCKE_AUTOTUNE_HPP

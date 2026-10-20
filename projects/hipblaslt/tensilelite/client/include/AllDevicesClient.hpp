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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// Single-process, all-GPU benchmarking for the tensilelite-client.
//
// Runs the selected GEMM problem(s)/solution(s) on every HIP-visible device at
// once, releases the timed kernels on all devices simultaneously (either a
// device-side spin gate or a host thread barrier), then aggregates the per-GPU
// timings (median and mean) and derives per-GPU and total-node throughput.
//
// Dispatched from main() when --all-devices=1 is set; returns a process exit
// code (0 == every device produced a timing for every problem/solution).

#pragma once

#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>

#include "ClientProblemFactory.hpp"
#include "ProgramOptions.hpp"

#include <memory>

namespace TensileLite
{
    namespace Client
    {
        int runAllDevices(po::variables_map const&                                       args,
                          std::shared_ptr<MasterSolutionLibrary<ContractionProblemGemm>> library,
                          std::shared_ptr<Hardware>                                      hardware,
                          ClientProblemFactory const&                                    problemFactory);
    } // namespace Client
} // namespace TensileLite

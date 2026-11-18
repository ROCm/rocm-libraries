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

#include <algorithm>
#include <array>
#include <cstring>

#include <hiptensor/hiptensor.h>

#include "data_types.hpp"
#include "rocke_autotune.hpp"

namespace hiptensor
{
    namespace rockebackend
    {
        namespace
        {
            struct DeviceBuffer
            {
                DeviceBuffer()                               = default;
                DeviceBuffer(DeviceBuffer const&)            = delete;
                DeviceBuffer& operator=(DeviceBuffer const&) = delete;
                ~DeviceBuffer()
                {
                    if(ptr != nullptr)
                    {
                        static_cast<void>(hipFree(ptr));
                    }
                }
                void* ptr = nullptr;
            };

            struct TimingStream
            {
                TimingStream()
                {
                    if(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess)
                    {
                        stream = nullptr;
                    }
                }
                TimingStream(TimingStream const&)            = delete;
                TimingStream& operator=(TimingStream const&) = delete;
                ~TimingStream()
                {
                    for(auto e : events)
                    {
                        static_cast<void>(hipEventDestroy(e));
                    }
                    if(stream != nullptr)
                    {
                        static_cast<void>(hipStreamDestroy(stream));
                    }
                }
                hipStream_t             stream = nullptr;
                std::vector<hipEvent_t> events;
            };

            // 1 in the scalar type a program reads its host scalars with.
            struct One
            {
                alignas(16) unsigned char bytes[16] = {};
            };

            One oneFor(Program const& program)
            {
                One o;
                if(program.complexScalars || program.scalarType == HIPTENSOR_C_32F
                   || program.scalarType == HIPTENSOR_C_64F)
                {
                    if(program.scalarType == HIPTENSOR_C_64F)
                    {
                        double const v[2] = {1.0, 0.0};
                        std::memcpy(o.bytes, v, sizeof(v));
                    }
                    else
                    {
                        float const v[2] = {1.0f, 0.0f};
                        std::memcpy(o.bytes, v, sizeof(v));
                    }
                    return o;
                }
                switch(program.scalarType)
                {
                case HIPTENSOR_R_16F:
                {
                    uint16_t const v = 0x3C00u;
                    std::memcpy(o.bytes, &v, sizeof(v));
                    break;
                }
                case HIPTENSOR_R_16BF:
                {
                    uint16_t const v = 0x3F80u;
                    std::memcpy(o.bytes, &v, sizeof(v));
                    break;
                }
                case HIPTENSOR_R_64F:
                {
                    double const v = 1.0;
                    std::memcpy(o.bytes, &v, sizeof(v));
                    break;
                }
                default:
                {
                    float const v = 1.0f;
                    std::memcpy(o.bytes, &v, sizeof(v));
                    break;
                }
                }
                return o;
            }
        } // namespace

        hiptensorStatus_t autotune(backend::Problem const& problem,
                                   std::vector<Candidate>& candidates,
                                   std::vector<int> const& indices,
                                   int                     reps,
                                   AutotuneResult&         result)
        {
            result = AutotuneResult{};
            result.medianUs.assign(candidates.size(), -1.0);
            reps = std::max(1, reps);
            for(auto i : indices)
            {
                if(auto status = loadFunctions(candidates[i].program);
                   status != HIPTENSOR_STATUS_SUCCESS)
                {
                    return status;
                }
            }

            // The problem's tensors (A, B, C, D slots) and the largest workspace.
            std::array<uint64_t, 5> bytes{};
            uint64_t                need = 0, workspace = 0;
            for(int s = 0; s < 5; s++)
            {
                auto const& t = problem.t[s];
                if(t.present)
                {
                    bytes[s] = std::max<uint64_t>(
                        256, uint64_t(t.spanElements()) * hiptensorDataTypeSize(t.type));
                    need += bytes[s];
                }
            }
            for(auto i : indices)
            {
                workspace = std::max(workspace, candidates[i].program.workspaceBytes);
            }
            need += workspace;
            size_t freeBytes = 0, totalBytes = 0;
            if(hipMemGetInfo(&freeBytes, &totalBytes) != hipSuccess || need > freeBytes / 2)
            {
                result.note = "the problem's tensors and workspace do not fit in half of the "
                              "free device memory";
                return HIPTENSOR_STATUS_SUCCESS;
            }
            std::array<DeviceBuffer, 5> tensors;
            DeviceBuffer                ws;
            for(int s = 0; s < 5; s++)
            {
                if(bytes[s] == 0)
                {
                    continue;
                }
                if(hipMalloc(&tensors[s].ptr, bytes[s]) != hipSuccess
                   || hipMemset(tensors[s].ptr, 0x3C, bytes[s]) != hipSuccess)
                {
                    tensors[s].ptr = nullptr;
                    result.note    = "cannot allocate the timing buffers";
                    return HIPTENSOR_STATUS_SUCCESS;
                }
            }
            if(workspace > 0 && hipMalloc(&ws.ptr, workspace) != hipSuccess)
            {
                ws.ptr      = nullptr;
                result.note = "cannot allocate the timing workspace";
                return HIPTENSOR_STATUS_SUCCESS;
            }
            TimingStream timing;
            if(timing.stream == nullptr)
            {
                return HIPTENSOR_STATUS_HIP_ERROR;
            }
            backend::Buffers buffers;
            buffers.A   = tensors[0].ptr;
            buffers.B   = tensors[1].ptr;
            buffers.C   = tensors[2].ptr;
            buffers.D   = tensors[3].ptr;
            buffers.out = tensors[problem.outputSlot()].ptr;

            // One untimed run each; a candidate that does not launch is not timed.
            std::vector<int> live;
            std::vector<One> ones(candidates.size());
            for(auto i : indices)
            {
                ones[i] = oneFor(candidates[i].program);
                backend::Scalars scalars{ones[i].bytes, ones[i].bytes, ones[i].bytes};
                auto             status = runProgram(
                    candidates[i].program, scalars, buffers, ws.ptr, workspace, timing.stream);
                if(status == HIPTENSOR_STATUS_SUCCESS)
                {
                    live.push_back(i);
                }
                else
                {
                    result.note += (result.note.empty() ? "" : ", ") + candidates[i].key
                                   + " did not launch (" + hiptensorGetErrorString(status) + ")";
                }
            }
            if(hipStreamSynchronize(timing.stream) != hipSuccess)
            {
                return HIPTENSOR_STATUS_HIP_ERROR;
            }
            if(live.empty())
            {
                return HIPTENSOR_STATUS_SUCCESS;
            }

            // Interleaved timed repetitions.
            auto const events = 2 * size_t(reps) * live.size();
            timing.events.resize(events, nullptr);
            for(auto& e : timing.events)
            {
                if(hipEventCreate(&e) != hipSuccess)
                {
                    e = nullptr;
                    return HIPTENSOR_STATUS_HIP_ERROR;
                }
            }
            size_t e = 0;
            for(int r = 0; r < reps; r++)
            {
                for(auto i : live)
                {
                    backend::Scalars scalars{ones[i].bytes, ones[i].bytes, ones[i].bytes};
                    if(hipEventRecord(timing.events[e], timing.stream) != hipSuccess
                       || runProgram(candidates[i].program,
                                     scalars,
                                     buffers,
                                     ws.ptr,
                                     workspace,
                                     timing.stream)
                              != HIPTENSOR_STATUS_SUCCESS
                       || hipEventRecord(timing.events[e + 1], timing.stream) != hipSuccess)
                    {
                        return HIPTENSOR_STATUS_HIP_ERROR;
                    }
                    e += 2;
                }
            }
            if(hipStreamSynchronize(timing.stream) != hipSuccess)
            {
                return HIPTENSOR_STATUS_HIP_ERROR;
            }
            std::vector<std::vector<float>> times(live.size());
            e = 0;
            for(int r = 0; r < reps; r++)
            {
                for(size_t j = 0; j < live.size(); j++)
                {
                    float ms = 0.0f;
                    if(hipEventElapsedTime(&ms, timing.events[e], timing.events[e + 1])
                       != hipSuccess)
                    {
                        return HIPTENSOR_STATUS_HIP_ERROR;
                    }
                    times[j].push_back(ms);
                    e += 2;
                }
            }
            double best = -1.0;
            for(size_t j = 0; j < live.size(); j++)
            {
                auto& t = times[j];
                std::sort(t.begin(), t.end());
                double const median      = t.size() % 2 == 1
                                               ? double(t[t.size() / 2])
                                               : 0.5 * (double(t[t.size() / 2 - 1]) + t[t.size() / 2]);
                result.medianUs[live[j]] = 1000.0 * median;
                if(best < 0.0 || result.medianUs[live[j]] < best)
                {
                    best          = result.medianUs[live[j]];
                    result.winner = live[j];
                }
            }
            return HIPTENSOR_STATUS_SUCCESS;
        }
    } // namespace rockebackend
} // namespace hiptensor

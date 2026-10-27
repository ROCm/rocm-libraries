/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "TimingEvents.hpp"

#include "DeviceContext.hpp"

#include <Tensile/hip/HipUtils.hpp>

#include <cstddef>

namespace TensileLite
{
    namespace Client
    {
        TimingEvents::TimingEvents(std::vector<std::shared_ptr<DeviceContext>> const& devices,
                                   size_t numInvocations,
                                   size_t numKernels)
            : m_numInvocations(numInvocations)
            , m_events(devices.size())
        {
            for(size_t d = 0; d < devices.size(); d++)
            {
                ScopedDevice device(devices[d]->deviceId);
                m_events[d].resize(numInvocations);
                for(auto& vec : m_events[d])
                {
                    vec.resize(numKernels, nullptr);
                    for(auto& event : vec)
                        HIP_CHECK_EXC(hipEventCreateWithFlags(&event, hipEventDefault));
                }
            }
        }

        TimingEvents::~TimingEvents()
        {
            for(auto& card : m_events)
            {
                for(auto& vec : card)
                {
                    for(auto& event : vec)
                    {
                        if(event)
                        {
                            HIP_CHECK_PRINT(hipEventDestroy(event), [&](hipError_t error) {
                                std::cerr << "hipEventDestroy failed: " << std::endl
                                          << " error: " << hipGetErrorString(error) << std::endl;
                            });
                            event = nullptr;
                        }
                    }
                }
            }
        }
    } // namespace Client
} // namespace TensileLite

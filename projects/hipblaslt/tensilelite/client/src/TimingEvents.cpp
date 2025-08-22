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

#include <Tensile/hip/HipUtils.hpp>

#include <cstddef>

namespace TensileLite
{
    namespace Client
    {
        TimingEvents::TimingEvents(size_t numInvocations, size_t numKernels)
            : m_events(numInvocations)
        {
            for(auto& vec : m_events)
            {
                vec.resize(numKernels, nullptr);

                for(auto& event : vec)
                    HIP_CHECK_EXC(hipEventCreateWithFlags(&event, hipEventDefault));
            }
        }

        TimingEvents::~TimingEvents()
        {
            for(auto& vec : m_events)
            {
                for(auto& event : vec)
                {
                    if(event)
                    {
                        HIP_CHECK_PRINT(hipEventDestroy(event));
                        event = nullptr;
                    }
                }
            }
        }

        std::vector<hipEvent_t> const& TimingEvents::operator[](size_t index) const
        {
            return m_events[index];
        }

        std::vector<std::vector<hipEvent_t>> const& TimingEvents::operator*() const
        {
            return m_events;
        }

        std::vector<std::vector<hipEvent_t>> const* TimingEvents::operator->() const
        {
            return &m_events;
        }
    } // namespace Client
} // namespace TensileLite

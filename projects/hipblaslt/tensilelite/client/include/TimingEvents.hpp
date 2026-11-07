/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <hip/hip_runtime.h>

#include <memory>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        struct DeviceContext;

        class TimingEvents
        {
        public:
            TimingEvents(std::vector<std::shared_ptr<DeviceContext>> const& devices,
                         size_t                                             numInvocations,
                         size_t                                             numKernels);
            ~TimingEvents();

            size_t numInvocations() const
            {
                return m_numInvocations;
            }
            size_t numCards() const
            {
                return m_events.size();
            }
            // [invocation][kernel], created on devices[card].
            std::vector<std::vector<hipEvent_t>> const& card(size_t card) const
            {
                return m_events[card];
            }

        private:
            size_t                                            m_numInvocations;
            std::vector<std::vector<std::vector<hipEvent_t>>> m_events;
        };
    } // namespace Client
} // namespace TensileLite

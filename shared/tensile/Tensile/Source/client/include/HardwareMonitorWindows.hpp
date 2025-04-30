/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <chrono>
#include <cstddef>
#include <future>
#include <thread>
#include <tuple>
#include <vector>

#include <hip/hip_runtime.h>

#include "HardwareMonitorType.hpp"

namespace Tensile
{
    namespace Client
    {
        /**
 * Monitors properties of a particular GPU in a separate thread.
 *
 * The thread is manually managed because the thread creation overhead is too
 * high to create a thread every time.
 *
 * The interface to this class is not thread-safe.
 */
        class HardwareMonitor
        {
        public:
            /** Translates the Hip device index into the corresponding device index for
   * ROCm-SMI. */
            using clock = std::chrono::steady_clock;

            // Monitor at the maximum possible rate.
            HardwareMonitor(int hipDeviceIndex){};
            // Limit collection to once per minPeriod.
            HardwareMonitor(int hipDeviceIndex, clock::duration minPeriod){};

            ~HardwareMonitor(){};

            void addTempMonitor(){};
            void addClockMonitor(ClockType clockType){};
            void addFanSpeedMonitor(uint32_t sensorIndex = 0){};

            double getAverageTemp()
            {
                return 0.0;
            };
            double getAverageClock(ClockType clockType)
            {
                return 0.0;
            };
            double getAverageFanSpeed(uint32_t sensorIndex = 0)
            {
                return 0.0;
            };
            int getDeviceIndex()
            {
                return 0;
            }
            size_t getSamples()
            {
                return 1;
            }

            /// Begins monitoring until stop() is called.
            void start(){};

            /// Sends a signal to the monitoring thread to end monitoring.
            void stop(){};

            /// Begins monitoring immediately, until the event has occurred.
            void runUntilEvent(hipEvent_t event){};

            /// Monitoring will occur from startEvent until stopEvent.
            void runBetweenEvents(hipEvent_t startEvent, hipEvent_t stopEvent){};

            /// Waits until monitoring has finished.
            /// Throws an exception if monitoring was started without a stop event
            /// and stop() has not been called.
            void wait(){};
        };
    } // namespace Client
} // namespace Tensile

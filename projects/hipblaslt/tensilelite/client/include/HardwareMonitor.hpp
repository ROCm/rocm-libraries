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

#include <chrono>
#include <cstddef>
#include <future>
#include <thread>
#include <tuple>
#include <vector>

#include <hip/hip_runtime.h>
#include <rocm_smi/rocm_smi.h>
#include <rocm_smi/rocm_smi64Config.h>

namespace TensileLite
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
            static uint32_t GetROCmSMIIndex(int hipDeviceIndex);

            using rsmi_temperature_type_t = int;
            using clock                   = std::chrono::steady_clock;

            // Monitor at the maximum possible rate.
            HardwareMonitor(int hipDeviceIndex);
            // Limit collection to once per minPeriod.
            HardwareMonitor(int hipDeviceIndex, clock::duration minPeriod);

            ~HardwareMonitor();

            void addTempMonitor(rsmi_temperature_type_t   sensorType = 0,
                                rsmi_temperature_metric_t metric     = RSMI_TEMP_CURRENT);
            void addClockMonitor(rsmi_clk_type_t clockType);
            void addFanSpeedMonitor(uint32_t sensorIndex = 0);

            double getAverageTemp(rsmi_temperature_type_t   sensorIndex = 0,
                                  rsmi_temperature_metric_t metric      = RSMI_TEMP_CURRENT);
            double getAverageClock(rsmi_clk_type_t clockType);
            double getAverageFanSpeed(uint32_t sensorIndex = 0);
            int    getDeviceIndex()
            {
                return m_hipDeviceIndex;
            }
            size_t getSamples()
            {
                return m_dataPoints;
            }

            double getMaxGfxFreqValues()
            {
                if(m_hasInvalidGpuFreqStatus || !has_maxFreqValues)
                    return std::numeric_limits<double>::quiet_NaN();
                return static_cast<double>(m_maxFreqValues);
            }

            /// Begins monitoring until stop() is called.
            void start();

            /// Sends a signal to the monitoring thread to end monitoring.
            void stop();

            /// Begins monitoring immediately, until the event has occurred.
            void runUntilEvent(hipEvent_t event);

            /// Monitoring will occur from startEvent until stopEvent.
            void runBetweenEvents(hipEvent_t startEvent, hipEvent_t stopEvent);

            /// Waits until monitoring has finished.
            /// Throws an exception if monitoring was started without a stop event
            /// and stop() has not been called.
            void wait();

        private:
            static void InitROCmSMI();

            void assertActive();
            void assertNotActive();

            void clearValues();
            void collectOnce();
            void sleepIfNecessary();

            void initThread();
            void runLoop();
            void collect(hipEvent_t startEvent, hipEvent_t stopEvent);

            clock::time_point m_lastCollection;
            clock::time_point m_nextCollection;
            clock::duration   m_minPeriod;

            std::thread m_thread;

            std::mutex              m_mutex;
            std::condition_variable m_cv;

            using Task = std::packaged_task<void(void)>;
            Task              m_task;
            std::future<void> m_future;
            std::atomic<bool> m_exit;
            std::atomic<bool> m_stop;
            bool              m_hasStopEvent = false;

            int      m_hipDeviceIndex;
            uint32_t m_smiDeviceIndex;

            size_t m_dataPoints;

            uint16_t m_XCDCount;

            std::vector<std::tuple<rsmi_temperature_type_t, rsmi_temperature_metric_t>> m_tempMetrics;
            std::vector<int64_t>                                                        m_tempValues;

            std::vector<rsmi_clk_type_t> m_clockMetrics;
            std::vector<uint64_t>        m_clockValues;

            std::vector<uint32_t> m_fanMetrics;
            std::vector<int64_t>  m_fanValues;

            uint64_t m_maxFreqValues; // The frequency is in Mhz
            bool     has_maxFreqValues         = false;
            bool     m_hasInvalidGpuFreqStatus = false;

            // Reserved for further performance check.
            std::vector<uint64_t>              m_SYSCLK_sum;
            std::vector<std::vector<uint64_t>> m_SYSCLK_array;
        };
    } // namespace Client
} // namespace TensileLite

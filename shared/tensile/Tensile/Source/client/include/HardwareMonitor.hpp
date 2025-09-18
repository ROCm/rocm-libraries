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
#include <rocm_smi/rocm_smi.h>

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
            HardwareMonitor(int hipDeviceIndex);
            // Limit collection to once per minPeriod.
            HardwareMonitor(int hipDeviceIndex, clock::duration minPeriod);

            ~HardwareMonitor();

            void addTempMonitor();
            void addClockMonitor(ClockType clockType);
            void addFanSpeedMonitor(uint32_t sensorIndex = 0);

            double getAverageTemp();
            double getAverageClock(ClockType clockType);
            double getAverageFanSpeed(uint32_t sensorIndex = 0);
            double getAverageGfxFreqPowerTemperature(std::vector<uint16_t>& dataValues);
            double getMedianGfxFreqPowerTemperature(std::vector<uint16_t>& dataValues);
            void   logMinMaxMedianAverage();

            int getDeviceIndex()
            {
                return m_hipDeviceIndex;
            }
            size_t getSamples()
            {
                return m_dataPoints;
            }

            std::vector<uint16_t>& getAllGfxFreqValues()
            {
                return m_freqValues;
            }

            std::vector<uint16_t>& getAllPowerValues()
            {
                return m_powerValues;
            }

            std::vector<uint16_t>& getAllTemperatureValues()
            {
                return m_tempHotspotValues;
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
            static uint32_t GetROCmSMIIndex(int hipDeviceIndex);
            static void     InitROCmSMI();

            void assertActive();
            void assertNotActive();

            void clearValues();
            void collectOnce();
            void sleepIfNecessary();

            void initThread();
            void runLoop();
            void collect(hipEvent_t startEvent, hipEvent_t stopEvent);
            void printMinMaxAverageMedian(const std::string&     str,
                                          std::vector<uint16_t>& dataValues);

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

            std::vector<std::tuple<rsmi_temperature_type_t, rsmi_temperature_metric_t>>
                                 m_tempMetrics;
            std::vector<int64_t> m_tempValues;

            std::vector<rsmi_clk_type_t> m_clockMetrics;
            std::vector<uint64_t>        m_clockValues;

            std::vector<uint32_t> m_fanMetrics;
            std::vector<int64_t>  m_fanValues;

            // Below 3 vectors stores the individual values of GFx frequency, GPU power, GPU temperature during GEMM kernel execution.
            // These vectors are implemented slightly different from previous clock/fan metric type implementation which uses
            // add**Monitor functions to monitor and store multiple HW types information in the vectors for the same device.
            // (ie)Existing metric implementation represent different HW type and uses ROCm API to get multiple
            // HW type(like different sensor, different clock). Hence it uses add**Monitor functions to store in to multiple vectors.
            // each new HW type requires an invocation of ROCm API. but below vectors uses ROCm API (rsmi_dev_gpu_metrics_info_get)
            // through single invocation gets all the HW type details, hence it does not need existing type of implementation.
            // if we need to store different HW type in the future, new additional vector type or vector<vector>> would be appropriate.
            std::vector<uint16_t> m_freqValues;
            std::vector<uint16_t> m_powerValues;
            std::vector<uint16_t> m_tempHotspotValues;
            bool                  m_hasInvalidGpuMetricStatus = false;
        };
    } // namespace Client
} // namespace Tensile

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

#include "HardwareMonitor.hpp"

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <thread>
#include <unistd.h>

#include <hip/hip_runtime.h>

#include <Tensile/hip/HipUtils.hpp>

#include "ResultReporter.hpp"

#define RSMI_CHECK_EXC(expr)                                                                      \
    do                                                                                            \
    {                                                                                             \
        rsmi_status_t e = (expr);                                                                 \
        if(e)                                                                                     \
        {                                                                                         \
            const char* errName = nullptr;                                                        \
            rsmi_status_string(e, &errName);                                                      \
            std::ostringstream msg;                                                               \
            msg << "Error " << e << "(" << errName << ") " << __FILE__ << ":" << __LINE__ << ": " \
                << std::endl                                                                      \
                << #expr << std::endl;                                                            \
            throw std::runtime_error(msg.str());                                                  \
        }                                                                                         \
    } while(0)

namespace Tensile
{
    namespace Client
    {
        rsmi_clk_type_t toSMIClockType(ClockType type)
        {
            switch(type)
            {
            case CLK_TYPE_SYS:
                return RSMI_CLK_TYPE_SYS;
            case CLK_TYPE_DF:
                return RSMI_CLK_TYPE_DF;
            case CLK_TYPE_DCEF:
                return RSMI_CLK_TYPE_DCEF;
            case CLK_TYPE_SOC:
                return RSMI_CLK_TYPE_SOC;
            case CLK_TYPE_MEM:
                return RSMI_CLK_TYPE_MEM;
            case CLK_INVALID:
                return RSMI_CLK_INVALID;
            default:
                return RSMI_CLK_TYPE_SYS;
            }
            return RSMI_CLK_TYPE_SYS;
        }

        uint32_t HardwareMonitor::GetROCmSMIIndex(int hipDeviceIndex)
        {
            InitROCmSMI();

            hipDeviceProp_t props;

            HIP_CHECK_EXC(hipGetDeviceProperties(&props, hipDeviceIndex));
#if HIP_VERSION >= 50220730
            int hip_version;
            HIP_CHECK_EXC(hipRuntimeGetVersion(&hip_version));
            if(hip_version >= 50220730)
            {
                HIP_CHECK_EXC(hipDeviceGetAttribute(&props.multiProcessorCount,
                                                    hipDeviceAttributePhysicalMultiProcessorCount,
                                                    hipDeviceIndex));
            }
#endif
            // PCIID format changes from 32bit to 64bit, Below is the new PCI format ID from ROCm 4.0.
            // Note:FUNCTION[0:2]bits is only used by aquanjaram TPX mode. This is not supported in HIP API yet, will need modification in future.
            // BDFID = ((DOMAIN & 0xffffffff) << 32) | ((BUS & 0xff) << 8) | ((DEVICE & 0x1f) <<3 ) | (FUNCTION & 0x7)
            uint64_t hipPCIID = 0;
            hipPCIID |= (((uint64_t)props.pciDomainID & 0xffffffff) << 32);
            hipPCIID |= ((props.pciBusID & 0xff) << 8);
            hipPCIID |= ((props.pciDeviceID & 0x1f) << 3);

            uint32_t smiCount = 0;

            RSMI_CHECK_EXC(rsmi_num_monitor_devices(&smiCount));

            std::ostringstream msg;
            msg << "PCI IDs: [" << std::endl;

            for(uint32_t smiIndex = 0; smiIndex < smiCount; smiIndex++)
            {
                uint64_t rsmiPCIID = 0;

                RSMI_CHECK_EXC(rsmi_dev_pci_id_get(smiIndex, &rsmiPCIID));

                msg << smiIndex << ": " << rsmiPCIID << std::endl;

                if(hipPCIID == rsmiPCIID)
                    return smiIndex;
            }

            msg << "]" << std::endl;
            std::time_t now
                = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            msg << std::put_time(gmtime(&now), "%F %T %z");

            throw std::runtime_error(concatenate("RSMI Can't find a device with PCI ID ",
                                                 hipPCIID,
                                                 "(",
                                                 props.pciDomainID,
                                                 "-",
                                                 props.pciBusID,
                                                 "-",
                                                 props.pciDeviceID,
                                                 ")\n",
                                                 msg.str()));
        }

        void HardwareMonitor::InitROCmSMI()
        {
            static rsmi_status_t status = rsmi_init(0);
            RSMI_CHECK_EXC(status);
        }

        HardwareMonitor::HardwareMonitor(int hipDeviceIndex, clock::duration minPeriod)
            : m_minPeriod(minPeriod)
            , m_hipDeviceIndex(hipDeviceIndex)
            , m_smiDeviceIndex(GetROCmSMIIndex(hipDeviceIndex))
            , m_dataPoints(0)
        {
            InitROCmSMI();

            initThread();
        }

        HardwareMonitor::HardwareMonitor(int hipDeviceIndex)
            : m_minPeriod(clock::duration::zero())
            , m_hipDeviceIndex(hipDeviceIndex)
            , m_smiDeviceIndex(GetROCmSMIIndex(hipDeviceIndex))
            , m_dataPoints(0)
        {
            InitROCmSMI();

            initThread();
        }

        HardwareMonitor::~HardwareMonitor()
        {
            m_stop = true;
            m_exit = true;

            m_cv.notify_all();
            m_thread.join();
        }

        void HardwareMonitor::initThread()
        {
            m_stop   = false;
            m_exit   = false;
            m_thread = std::thread([=]() { this->runLoop(); });
        }

        void HardwareMonitor::addTempMonitor()
        {
            rsmi_temperature_type_t   sensorType = RSMI_TEMP_TYPE_EDGE;
            rsmi_temperature_metric_t metric     = RSMI_TEMP_CURRENT;

            assertNotActive();

            m_tempMetrics.emplace_back(sensorType, metric);
            m_tempValues.resize(m_tempMetrics.size());
        }

        void HardwareMonitor::addClockMonitor(ClockType clockType)
        {
            assertNotActive();

            m_clockMetrics.push_back(toSMIClockType(clockType));
            m_clockValues.resize(m_clockMetrics.size());
        }

        void HardwareMonitor::addFanSpeedMonitor(uint32_t sensorIndex)
        {
            assertNotActive();

            m_fanMetrics.push_back(sensorIndex);
            m_fanValues.resize(m_fanMetrics.size());
        }

        double HardwareMonitor::getAverageTemp()
        {
            rsmi_temperature_type_t   sensorType = RSMI_TEMP_TYPE_EDGE;
            rsmi_temperature_metric_t metric     = RSMI_TEMP_CURRENT;

            assertNotActive();

            if(m_dataPoints == 0)
                throw std::runtime_error("No data points collected!");

            for(size_t i = 0; i < m_tempMetrics.size(); i++)
            {
                if(m_tempMetrics[i] == std::make_tuple(sensorType, metric))
                {
                    int64_t rawValue = m_tempValues[i];
                    if(rawValue == std::numeric_limits<int64_t>::max())
                        return std::numeric_limits<double>::quiet_NaN();

                    return static_cast<double>(rawValue) / (1000.0 * m_dataPoints);
                }
            }

            throw std::runtime_error(concatenate(
                "Can't read temp value that wasn't requested: ", sensorType, " - ", metric));
        }

        double HardwareMonitor::getAverageClock(ClockType clockType)
        {
            assertNotActive();

            if(m_dataPoints == 0)
                throw std::runtime_error("No data points collected!");

            for(size_t i = 0; i < m_clockMetrics.size(); i++)
            {
                if(m_clockMetrics[i] == toSMIClockType(clockType))
                {
                    uint64_t rawValue = m_clockValues[i];
                    if(rawValue == std::numeric_limits<uint64_t>::max())
                        return std::numeric_limits<double>::quiet_NaN();

                    return static_cast<double>(rawValue) / (1e6 * m_dataPoints);
                }
            }

            throw std::runtime_error(
                concatenate("Can't read clock value that wasn't requested: ", clockType));
        }

        double HardwareMonitor::getAverageFanSpeed(uint32_t sensorIndex)
        {
            assertNotActive();

            if(m_dataPoints == 0)
                throw std::runtime_error("No data points collected!");

            for(size_t i = 0; i < m_fanMetrics.size(); i++)
            {
                if(m_fanMetrics[i] == sensorIndex)
                {
                    int64_t rawValue = m_fanValues[i];
                    if(rawValue == std::numeric_limits<int64_t>::max())
                        return std::numeric_limits<double>::quiet_NaN();

                    return static_cast<double>(rawValue) / m_dataPoints;
                }
            }

            throw std::runtime_error(
                concatenate("Can't read fan value that wasn't requested: ", sensorIndex));
        }

        void HardwareMonitor::start()
        {
            runBetweenEvents(nullptr, nullptr);
        }

        void HardwareMonitor::stop()
        {
            assertActive();

            m_stop = true;
        }

        void HardwareMonitor::runUntilEvent(hipEvent_t event)
        {
            runBetweenEvents(nullptr, event);
        }

        void HardwareMonitor::runBetweenEvents(hipEvent_t startEvent, hipEvent_t stopEvent)
        {
            assertNotActive();

            {
                std::unique_lock<std::mutex> lock(m_mutex);

                m_hasStopEvent = stopEvent != nullptr;

                m_task   = std::move(Task([=]() { this->collect(startEvent, stopEvent); }));
                m_future = m_task.get_future();

                m_stop = false;
                m_exit = false;
            }
            m_cv.notify_all();
        }

        void HardwareMonitor::clearValues()
        {
            m_dataPoints = 0;

            for(auto& v : m_tempValues)
                v = 0;
            for(auto& v : m_clockValues)
                v = 0;
            for(auto& v : m_fanValues)
                v = 0;

            m_hasInvalidGpuMetricStatus = false;
            m_freqValues.clear();
            m_powerValues.clear();
            m_tempHotspotValues.clear();
            m_lastCollection = clock::time_point();
            m_nextCollection = clock::time_point();
        }

        void HardwareMonitor::collectOnce()
        {
            for(int i = 0; i < m_tempMetrics.size(); i++)
            {
                // if an error occurred previously, don't overwrite it.
                if(m_tempValues[i] == std::numeric_limits<int64_t>::max())
                    continue;

                rsmi_temperature_type_t   sensorType;
                rsmi_temperature_metric_t metric;
                std::tie(sensorType, metric) = m_tempMetrics[i];

                int64_t newValue = 0;
                auto    status
                    = rsmi_dev_temp_metric_get(m_smiDeviceIndex, sensorType, metric, &newValue);
                if(status != RSMI_STATUS_SUCCESS)
                    m_tempValues[i] = std::numeric_limits<int64_t>::max();
                else
                    m_tempValues[i] += newValue;
            }

            for(int i = 0; i < m_clockMetrics.size(); i++)
            {
                // if an error occurred previously, don't overwrite it.
                if(m_clockValues[i] == std::numeric_limits<uint64_t>::max())
                    continue;

                rsmi_frequencies_t freq;

                auto status = rsmi_dev_gpu_clk_freq_get(m_smiDeviceIndex, m_clockMetrics[i], &freq);
                if(status != RSMI_STATUS_SUCCESS || freq.current > RSMI_MAX_NUM_FREQUENCIES)
                {
                    m_clockValues[i] = std::numeric_limits<uint64_t>::max();
                }
                else
                {
                    m_clockValues[i] += freq.frequency[freq.current];
                }
            }

            for(int i = 0; i < m_fanMetrics.size(); i++)
            {
                // if an error occurred previously, don't overwrite it.
                if(m_fanValues[i] == std::numeric_limits<int64_t>::max())
                    continue;

                rsmi_frequencies_t freq;

                int64_t newValue = 0;
                auto status = rsmi_dev_fan_rpms_get(m_smiDeviceIndex, m_fanMetrics[i], &newValue);
                if(status != RSMI_STATUS_SUCCESS)
                    m_fanValues[i] = std::numeric_limits<int64_t>::max();
                else
                    m_fanValues[i] += newValue;
            }

            rsmi_gpu_metrics_t gpuMetrics;
            auto status = rsmi_dev_gpu_metrics_info_get(m_smiDeviceIndex, &gpuMetrics);
            if(status != RSMI_STATUS_SUCCESS)
            {
                m_hasInvalidGpuMetricStatus = true;
            }
            else
            {
                if(!m_hasInvalidGpuMetricStatus)
                {
                    m_freqValues.push_back(gpuMetrics.average_gfxclk_frequency);
                    m_powerValues.push_back(gpuMetrics.average_socket_power);
                    m_tempHotspotValues.push_back(gpuMetrics.temperature_hotspot);
                }
            }
            m_dataPoints++;
        }

        double HardwareMonitor::getAverageGfxFreqPowerTemperature(
            std::vector<uint16_t>& inputDataValues)
        {
            assertNotActive();

            if(m_dataPoints == 0)
                throw std::runtime_error("No data points collected!");
            if(m_hasInvalidGpuMetricStatus)
                return std::numeric_limits<double>::quiet_NaN();

            return (std::accumulate(inputDataValues.begin(), inputDataValues.end(), double(0)))
                   / inputDataValues.size();
        }

        double HardwareMonitor::getMedianGfxFreqPowerTemperature(
            std::vector<uint16_t>& inputDataValues)
        {
            assertNotActive();

            if(m_dataPoints == 0)
                throw std::runtime_error("No data points collected!");
            if(m_hasInvalidGpuMetricStatus)
                return std::numeric_limits<double>::quiet_NaN();

            size_t midValueIndex = inputDataValues.size() / 2;
            std::nth_element(inputDataValues.begin(),
                             inputDataValues.begin() + midValueIndex,
                             inputDataValues.end());
            return static_cast<double>(inputDataValues[midValueIndex]);
        }

        void HardwareMonitor::logMinMaxMedianAverage()
        {
            assertNotActive();
            if(m_hasInvalidGpuMetricStatus)
                throw std::runtime_error("Invalid GPU metric data!");
            if(m_dataPoints == 0)
                throw std::runtime_error("No data points collected!");

            std::cout << "\nROCm SMI API consolidated frequency,power,temperature data"
                      << "\n";
            std::cout << "GFX Value\t\t\t\t"
                      << "PPT0_value\t\t\t"
                      << "Temperature\n";
            // Log individual data
            for(int i = 0; i < m_freqValues.size(); i++)
            {
                std::cout << std::left << std::setw(13) << m_freqValues[i] << "\t\t\t\t"
                          << m_powerValues[i] << "\t\t\t\t\t" << m_tempHotspotValues[i] << "\n";
            }

            std::cout << "\n";
            std::cout << "\t\t\t\t\tMin\t\t"
                      << "Max\t\t"
                      << "Average\t  "
                      << "Median\n";
            // min, max,avg, median frequency
            printMinMaxAverageMedian("GFX Frequency", m_freqValues);
            printMinMaxAverageMedian("Power Value", m_powerValues);
            printMinMaxAverageMedian("Temperature", m_tempHotspotValues);
        }

        void HardwareMonitor::printMinMaxAverageMedian(const std::string&     str,
                                                       std::vector<uint16_t>& dataValues)
        {
            std::cout << std::left << std::setw(20) << str << std::setw(8)
                      << *std::min_element(dataValues.begin(), dataValues.end());
            std::cout << std::setw(8) << *std::max_element(dataValues.begin(), dataValues.end());
            std::cout << std::setw(10) << std::fixed << std::setprecision(2)
                      << getAverageGfxFreqPowerTemperature(dataValues);
            std::cout << std::setw(8) << getMedianGfxFreqPowerTemperature(dataValues) << "\n";
        }

        void HardwareMonitor::sleepIfNecessary()
        {
            auto now = clock::now();

            if(now < m_nextCollection)
            {
                std::this_thread::sleep_until(m_nextCollection);
                m_lastCollection = clock::now();
            }
            else
            {
                m_lastCollection = now;
            }

            m_nextCollection = m_lastCollection + m_minPeriod;
        }

        void HardwareMonitor::runLoop()
        {
            HIP_CHECK_EXC(hipSetDevice(m_hipDeviceIndex));

            std::unique_lock<std::mutex> lock(m_mutex);
            while(!m_exit)
            {
                while(!m_task.valid() && !m_exit)
                    m_cv.wait(lock);

                if(m_exit)
                    return;

                m_task();
                m_task = std::move(Task());
            }
        }

        void HardwareMonitor::collect(hipEvent_t startEvent, hipEvent_t stopEvent)
        {
            clearValues();

            if(startEvent != nullptr)
                HIP_CHECK_EXC(hipEventSynchronize(startEvent));

            do
            {
                collectOnce();
                sleepIfNecessary();

                if(stopEvent != nullptr && hipEventQuery(stopEvent) == hipSuccess)
                    return;
            } while(!m_stop && !m_exit);
        }

        void HardwareMonitor::wait()
        {
            if(!m_future.valid())
                return;

            if(!m_hasStopEvent && !m_stop)
                throw std::runtime_error("Waiting for monitoring to stop with no end condition.");

            m_future.wait();
            m_future = std::move(std::future<void>());
        }

        void HardwareMonitor::assertActive()
        {
            if(!m_future.valid())
                throw std::runtime_error("Monitor is not active.");
        }

        void HardwareMonitor::assertNotActive()
        {
            if(m_future.valid())
                throw std::runtime_error("Monitor is active.");
        }

    } // namespace Client
} // namespace Tensile

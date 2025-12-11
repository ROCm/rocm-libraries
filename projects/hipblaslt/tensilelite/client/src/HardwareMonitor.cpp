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

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <unistd.h>

#include <Tensile/hip/HipUtils.hpp>
#include <hip/hip_runtime.h>

#include "HardwareMonitor.hpp"
#include "ResultReporter.hpp"

#define AMDSMI_CHECK_EXC(expr)                                                                    \
    do                                                                                            \
    {                                                                                             \
        amdsmi_status_t e = (expr);                                                               \
        if(e)                                                                                     \
        {                                                                                         \
            const char* errName = nullptr;                                                        \
            amdsmi_status_code_to_string(e, &errName);                                            \
            std::ostringstream msg;                                                               \
            msg << "Error " << e << "(" << errName << ") " << __FILE__ << ":" << __LINE__ << ": " \
                << std::endl                                                                      \
                << #expr << std::endl;                                                            \
            throw std::runtime_error(msg.str());                                                  \
        }                                                                                         \
    } while(0)

namespace TensileLite
{
    namespace Client
    {
        uint32_t HardwareMonitor::GetAMDSMIIndex(int hipDeviceIndex)
        {
            InitAMDSMI();

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

            uint64_t hipPCIID{};
            hipPCIID |= (((uint64_t)props.pciDomainID & 0xffffffff) << 32);
            hipPCIID |= ((props.pciBusID & 0xff) << 8);
            hipPCIID |= ((props.pciDeviceID & 0x1f) << 3);

            uint32_t smiSocketCount{};
            AMDSMI_CHECK_EXC(amdsmi_get_socket_handles(&smiSocketCount, nullptr));

            m_socketHandles.resize(smiSocketCount);
            AMDSMI_CHECK_EXC(amdsmi_get_socket_handles(&smiSocketCount, &m_socketHandles[0]));

            std::ostringstream msg;
            msg << "PCI IDs: [" << std::endl;

            for(uint32_t device = 0; device < smiSocketCount; device++)
            {
                uint32_t deviceCount{};
                AMDSMI_CHECK_EXC(
                    amdsmi_get_processor_handles(m_socketHandles[device], &deviceCount, nullptr));
                m_processorHandles.resize(deviceCount);

                AMDSMI_CHECK_EXC(amdsmi_get_processor_handles(
                    m_socketHandles[device], &deviceCount, &m_processorHandles[0]));

                for(uint32_t smiIndex = 0; smiIndex < deviceCount; smiIndex++)
                {
                    uint64_t amdSMIPCIID{};
                    AMDSMI_CHECK_EXC(
                        amdsmi_get_gpu_bdf_id(m_processorHandles[smiIndex], &amdSMIPCIID));

                    msg << smiIndex << ": " << amdSMIPCIID << std::endl;

                    if(hipPCIID == amdSMIPCIID)
                    {
                        return smiIndex;
                    }
                }
            }

            msg << "]" << std::endl;
            std::time_t now
                = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            msg << std::put_time(gmtime(&now), "%F %T %z");

            throw std::runtime_error(concatenate("AMDSMI Can't find a device with PCI ID ",
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

        void HardwareMonitor::InitAMDSMI()
        {
            static amdsmi_status_t status = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
            AMDSMI_CHECK_EXC(status);
        }

        HardwareMonitor::HardwareMonitor(int hipDeviceIndex, clock::duration minPeriod)
            : m_minPeriod(minPeriod)
            , m_hipDeviceIndex(hipDeviceIndex)
            , m_dataPoints(0)
            , m_XCDCount(1)
        {
            InitAMDSMI();
            m_smiDeviceIndex = GetAMDSMIIndex(hipDeviceIndex);
            initThread();

#if AMDSMI_LIB_VERSION_MAJOR >= 25
            auto status
                = amdsmi_get_gpu_xcd_counter(m_processorHandles[m_smiDeviceIndex], &m_XCDCount);
            if(status != AMDSMI_STATUS_SUCCESS || m_XCDCount == 0)
            {
                m_XCDCount = 1;
            }
#endif
        }

        HardwareMonitor::HardwareMonitor(int hipDeviceIndex)
            : m_minPeriod(clock::duration::zero())
            , m_hipDeviceIndex(hipDeviceIndex)
            , m_dataPoints(0)
            , m_XCDCount(1)
        {
            InitAMDSMI();
            m_smiDeviceIndex = GetAMDSMIIndex(hipDeviceIndex);
            initThread();

#if AMDSMI_LIB_VERSION_MAJOR >= 25
            auto status
                = amdsmi_get_gpu_xcd_counter(m_processorHandles[m_smiDeviceIndex], &m_XCDCount);

            if(status != AMDSMI_STATUS_SUCCESS || m_XCDCount == 0)
            {
                m_XCDCount = 1;
            }
#endif
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
            m_thread = std::thread([this]() { this->runLoop(); });
        }

        void HardwareMonitor::addTempMonitor(amdsmi_temperature_type_t   sensorType,
                                             amdsmi_temperature_metric_t metric)
        {
            assertNotActive();

            m_tempMetrics.emplace_back(sensorType, metric);
            m_tempValues.resize(m_tempMetrics.size());
        }

        void HardwareMonitor::addClockMonitor(amdsmi_clk_type_t clockType)
        {
            assertNotActive();

            m_clockMetrics.push_back(clockType);
            m_clockValues.resize(m_clockMetrics.size());
        }

        void HardwareMonitor::addFanSpeedMonitor(uint32_t sensorIndex)
        {
            assertNotActive();

            m_fanMetrics.push_back(sensorIndex);
            m_fanValues.resize(m_fanMetrics.size());
        }

        double HardwareMonitor::getAverageTemp(amdsmi_temperature_type_t   sensorType,
                                               amdsmi_temperature_metric_t metric)
        {
            assertNotActive();

            if(m_dataPoints == 0)
            {
                throw std::runtime_error("No data points collected!");
            }

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

        double HardwareMonitor::getAverageClock(amdsmi_clk_type_t clockType)
        {
            assertNotActive();

            if(m_dataPoints == 0)
            {
                throw std::runtime_error("No data points collected!");
            }

            for(size_t i = 0; i < m_clockMetrics.size(); i++)
            {
                if(m_clockMetrics[i] == clockType)
                {
                    uint64_t rawValue = m_clockValues[i];
                    if(rawValue == std::numeric_limits<uint64_t>::max())
                        return std::numeric_limits<double>::quiet_NaN();

                    if(m_clockMetrics[i] == AMDSMI_CLK_TYPE_SYS)
                    {
                        return static_cast<double>(rawValue) / (1e6 * m_dataPoints * m_XCDCount);
                    }
                    else
                    {
                        return static_cast<double>(rawValue) / (1e6 * m_dataPoints);
                    }
                }
            }

            throw std::runtime_error(
                concatenate("Can't read clock value that wasn't requested: ", clockType));
        }

        double HardwareMonitor::getAverageFanSpeed(uint32_t sensorIndex)
        {
            assertNotActive();

            if(m_dataPoints == 0)
            {
                throw std::runtime_error("No data points collected!");
            }

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

                m_task   = std::move(Task(
                    [this, startEvent, stopEvent]() { this->collect(startEvent, stopEvent); }));
                m_future = m_task.get_future();

                m_stop = false;
                m_exit = false;
            }
            m_cv.notify_all();
        }

        void HardwareMonitor::clearValues()
        {
            m_dataPoints = 0;

            std::fill(m_tempValues.begin(), m_tempValues.end(), 0);
            std::fill(m_clockValues.begin(), m_clockValues.end(), 0);
            std::fill(m_fanValues.begin(), m_fanValues.end(), 0);

            m_lastCollection = clock::time_point();
            m_nextCollection = clock::time_point();

            m_SYSCLK_sum = std::vector<uint64_t>(m_XCDCount, 0);
            m_SYSCLK_array
                = std::vector<std::vector<uint64_t>>(m_XCDCount, std::vector<uint64_t>{});
        }

        void HardwareMonitor::collectOnce()
        {
            const double cMhzToHz = 1000000;

            for(int i = 0; i < m_tempMetrics.size(); i++)
            {
                // if an error occurred previously, don't overwrite it.
                if(m_tempValues[i] == std::numeric_limits<int64_t>::max())
                {
                    continue;
                }

                amdsmi_temperature_type_t   sensorType;
                amdsmi_temperature_metric_t metric;
                std::tie(sensorType, metric) = m_tempMetrics[i];

                int64_t newValue{};
                auto    status = amdsmi_get_temp_metric(
                    m_processorHandles[m_smiDeviceIndex], sensorType, metric, &newValue);
                if(status != AMDSMI_STATUS_SUCCESS)
                {
                    m_tempValues[i] = std::numeric_limits<int64_t>::max();
                }
                else
                {
                    m_tempValues[i] += newValue;
                }
            }

            for(int i = 0; i < m_clockMetrics.size(); i++)
            {
                // if an error occurred previously, don't overwrite it.
                if(m_clockValues[i] == std::numeric_limits<uint64_t>::max())
                {
                    continue;
                }

                amdsmi_frequencies_t freq;

                if(m_clockMetrics[i] == AMDSMI_CLK_TYPE_SYS)
                {
#if AMDSMI_LIB_VERSION_MAJOR >= 25
                    amdsmi_gpu_metrics_t gpuMetrics;
                    // multi_XCD
                    auto status = amdsmi_get_gpu_metrics_info(m_processorHandles[m_smiDeviceIndex],
                                                              &gpuMetrics);
                    if(status == AMDSMI_STATUS_SUCCESS)
                    {
                        uint64_t sysclkSum = 0;
                        for(uint32_t xcd = 0; xcd < m_XCDCount; xcd++)
                        {
                            m_SYSCLK_sum[xcd] += gpuMetrics.current_gfxclks[xcd] * cMhzToHz;
                            m_SYSCLK_array[xcd].push_back(gpuMetrics.current_gfxclks[xcd]
                                                          * cMhzToHz);
                            sysclkSum += gpuMetrics.current_gfxclks[xcd] * cMhzToHz;
                        }
                        m_clockValues[i] += sysclkSum;
                    }
#else
                    // XCD0
                    auto status = amdsmi_get_clk_freq(
                        m_processorHandles[m_smiDeviceIndex], m_clockMetrics[i], &freq);
                    if(status != AMDSMI_STATUS_SUCCESS)
                    {
                        m_clockValues[i] = std::numeric_limits<uint64_t>::max();
                    }
                    else
                    {
                        m_clockValues[i] += freq.frequency[freq.current];
                    }
#endif
                }
                else
                {
                    auto status = amdsmi_get_clk_freq(
                        m_processorHandles[m_smiDeviceIndex], m_clockMetrics[i], &freq);

                    if(status != AMDSMI_STATUS_SUCCESS)
                    {
                        m_clockValues[i] = std::numeric_limits<uint64_t>::max();
                    }
                    else
                    {
                        m_clockValues[i] += freq.frequency[freq.current];
                    }
                }
            }

            for(int i = 0; i < m_fanMetrics.size(); i++)
            {
                // if an error occurred previously, don't overwrite it.
                if(m_fanValues[i] == std::numeric_limits<int64_t>::max())
                {
                    continue;
                }

                int64_t newValue = 0;
                auto    status   = amdsmi_get_gpu_fan_rpms(
                    m_processorHandles[m_smiDeviceIndex], m_fanMetrics[i], &newValue);

                if(status != AMDSMI_STATUS_SUCCESS)
                {
                    m_fanValues[i] = std::numeric_limits<int64_t>::max();
                }
                else
                {
                    m_fanValues[i] += newValue;
                }
            }

            // Retrieves the maximum hardware supported frequency.
            amdsmi_frequencies_t freqs;
            const int            MAX_RETRY  = 10;
            const int            SLEEP_TIME = 100; // sleep time in milliseconds
            bool                 success    = false;

            if(!has_maxFreqValues && !m_hasInvalidGpuFreqStatus)
            {
                for(int retry = 0; retry < MAX_RETRY; ++retry)
                {
                    auto status = amdsmi_get_clk_freq(
                        m_processorHandles[m_smiDeviceIndex], AMDSMI_CLK_TYPE_SYS, &freqs);

                    if(status == AMDSMI_STATUS_SUCCESS)
                    {
                        success = true;
                        break;
                    }
                    // Sleep before next retry
                    std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));
                }

                if(!success)
                {
                    m_hasInvalidGpuFreqStatus = true;
                }
                else if(freqs.num_supported > 0)
                {
                    m_maxFreqValues
                        = *std::max_element(freqs.frequency, freqs.frequency + freqs.num_supported);

                    has_maxFreqValues = true;
                    m_maxFreqValues /= cMhzToHz; // Convert to MHz
                }
                else
                {
                    m_hasInvalidGpuFreqStatus = true;
                }
            }
            m_dataPoints++;
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
                {
                    m_cv.wait(lock);
                }

                if(m_exit)
                {
                    return;
                }

                m_task();
                m_task = std::move(Task());
            }
        }

        void HardwareMonitor::collect(hipEvent_t startEvent, hipEvent_t stopEvent)
        {
            clearValues();

            if(startEvent != nullptr)
            {
                HIP_CHECK_EXC(hipEventSynchronize(startEvent));
            }

            do
            {
                collectOnce();
                sleepIfNecessary();

                if(stopEvent != nullptr && hipEventQuery(stopEvent) == hipSuccess)
                {
                    return;
                }
            } while(!m_stop && !m_exit);
        }

        void HardwareMonitor::wait()
        {
            if(!m_future.valid())
            {
                return;
            }

            if(!m_hasStopEvent && !m_stop)
            {
                throw std::runtime_error("Waiting for monitoring to stop with no end condition.");
            }

            m_future.wait();
            m_future = std::move(std::future<void>());
        }

        void HardwareMonitor::assertActive()
        {
            if(!m_future.valid())
            {
                throw std::runtime_error("Monitor is not active.");
            }
        }

        void HardwareMonitor::assertNotActive()
        {
            if(m_future.valid())
            {
                throw std::runtime_error("Monitor is active.");
            }
        }

    } // namespace Client
} // namespace TensileLite

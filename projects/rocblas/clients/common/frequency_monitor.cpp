
/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "frequency_monitor.hpp"
#include "rocblas.hpp"

#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#ifndef _WIN32

#include <hip/hip_runtime.h>
#include <rocm_smi/rocm_smi.h>
#include <rocm_smi/rocm_smi64Config.h>

template <typename T>
inline std::ostream& stream_write(std::ostream& stream, T&& val)
{
    return stream << std::forward<T>(val);
}

template <typename T, typename... Ts>
inline std::ostream& stream_write(std::ostream& stream, T&& val, Ts&&... vals)
{
    return stream_write(stream << std::forward<T>(val), std::forward<Ts>(vals)...);
}

template <typename... Ts>
inline std::string concatenate(Ts&&... vals)
{
    std::ostringstream msg;
    stream_write(msg, std::forward<Ts>(vals)...);

    return msg.str();
}

#define HIP_CHECK_EXC(expr)                                                                       \
    do                                                                                            \
    {                                                                                             \
        hipError_t e = (expr);                                                                    \
        if(e)                                                                                     \
        {                                                                                         \
            const char*        errName = hipGetErrorName(e);                                      \
            const char*        errMsg  = hipGetErrorString(e);                                    \
            std::ostringstream msg;                                                               \
            msg << "Error " << e << "(" << errName << ") " << __FILE__ << ":" << __LINE__ << ": " \
                << std::endl                                                                      \
                << #expr << std::endl                                                             \
                << errMsg << std::endl;                                                           \
            throw std::runtime_error(msg.str());                                                  \
        }                                                                                         \
    } while(0)

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

#endif

class FrequencyMonitorImp : public FrequencyMonitor
{
public:
    const double cHzToMHz = 0.000001;
    const double cMhzToHz = 1000000;

    // deleting copy constructor
    FrequencyMonitorImp(const FrequencyMonitorImp& obj) = delete;

#ifndef _WIN32

    bool enabled()
    {
        static const char* env1 = getenv("ROCBLAS_BENCH_FREQ");
        static const char* env2 = getenv("ROCBLAS_BENCH_FREQ_ALL");
        return env1 != nullptr || (env2 != nullptr && m_isMultiXCDSupported);
    }

    bool detailedReport()
    {
        static const char* env2 = getenv("ROCBLAS_BENCH_FREQ_ALL");
        return (env2 != nullptr && m_isMultiXCDSupported);
    }

    FrequencyMonitorImp()
    {
        initThread();
    }

    ~FrequencyMonitorImp()
    {
        m_stop = true;
        m_exit = true;

        m_cv.notify_all();
        m_thread.join();
    }

    void set_device_id(int deviceId)
    {
        m_smiDeviceIndex = GetROCmSMIIndex(deviceId);
        m_XCDCount       = 1;

#if rocm_smi_VERSION_MAJOR >= 7
        auto status2 = rsmi_dev_metrics_xcd_counter_get(m_smiDeviceIndex, &m_XCDCount);

        if(status2 != RSMI_STATUS_SUCCESS)
        {
            m_XCDCount = 1;
        }
#endif
    }

    void start()
    {
        if(!enabled())
            return;

        clearValues();
        runBetweenEvents();
    }

    void stop()
    {
        if(!enabled())
            return;

        assertActive();
        m_stop = true;
        wait();
    }

    double averageValueMHz(double sum, std::vector<uint64_t>& data)
    {
        assertNotActive();
        if(enabled() && data.empty())
            return 0.0;

        double averageFrequency = static_cast<double>(sum / data.size());
        return averageFrequency * cHzToMHz;
    }

    double medianValueMHz(std::vector<uint64_t>& data)
    {
        assertNotActive();

        double median = 0.0;
        if(enabled() && data.empty())
            return 0.0;

        size_t num_datapoints = data.size();
        if(num_datapoints)
        {
            std::sort(data.begin(), data.end());

            median = static_cast<double>(data[(num_datapoints - 1) / 2]);
            if(num_datapoints % 2 == 0)
            {
                median = static_cast<double>(median + data[(num_datapoints - 1) / 2 + 1]) / 2.0;
            }
        }
        return median * cHzToMHz;
    }

    double getLowestAverageSYSCLK()
    {
        std::vector<double> allAvgSYSCLK = getAllAverageSYSCLK();
        double              minAvgSYSCLK = allAvgSYSCLK[0];
        for(int i = 1; i < m_XCDCount; i++)
        {
            if(allAvgSYSCLK[i] <= 0)
                continue;
            minAvgSYSCLK = min(minAvgSYSCLK, allAvgSYSCLK[i]);
        }
        return minAvgSYSCLK;
    }

    double getLowestMedianSYSCLK()
    {
        std::vector<double> allMedianSYSCLK = getAllMedianSYSCLK();
        double              minMedianSYSCLK = allMedianSYSCLK[0];
        for(int i = 1; i < m_XCDCount; i++)
        {
            if(allMedianSYSCLK[i] <= 0)
                continue;
            minMedianSYSCLK = min(minMedianSYSCLK, allMedianSYSCLK[i]);
        }
        return minMedianSYSCLK;
    }

    std::vector<double> getAllAverageSYSCLK()
    {
        std::vector<double> avgSYSCLK(m_XCDCount, 0.0);
        for(int i = 0; i < m_XCDCount; i++)
        {
            avgSYSCLK[i] = averageValueMHz(m_SYSCLK_sum[i], m_SYSCLK_array[i]);
        }
        return avgSYSCLK;
    }

    std::vector<double> getAllMedianSYSCLK()
    {
        std::vector<double> medianSYSCLK(m_XCDCount, 0.0);
        for(int i = 0; i < m_XCDCount; i++)
        {
            medianSYSCLK[i] = medianValueMHz(m_SYSCLK_array[i]);
        }
        return medianSYSCLK;
    }

    double getAverageMEMCLK()
    {
        return averageValueMHz(m_MEMCLK_sum, m_MEMCLK_array);
    }

    double getMedianMEMCLK()
    {
        return medianValueMHz(m_MEMCLK_array);
    }

    double getCuCount()
    {
        return m_CUCount;
    }

private:
    void initThread()
    {
        m_stop = false;
        m_exit = false;

        rsmi_version_t version;

        m_isMultiXCDSupported = false;
#if rocm_smi_VERSION_MAJOR >= 7
        m_isMultiXCDSupported = true;
#endif

        m_thread = std::thread([=]() { this->runLoop(); });
        return;
    }

    void runBetweenEvents()
    {
        assertNotActive();
        {
            std::unique_lock<std::mutex> lock(m_mutex);

            m_task   = std::move(Task([=]() { this->collect(); }));
            m_future = m_task.get_future();

            m_stop = false;
            m_exit = false;
        }
        m_cv.notify_all();
    }

    void runLoop()
    {
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
        return;
    }

    void collect()
    {
        rsmi_frequencies_t freq;
        do
        {
#if rocm_smi_VERSION_MAJOR >= 7

            rsmi_gpu_metrics_t gpuMetrics;
            // multi_XCD
            auto status1 = rsmi_dev_gpu_metrics_info_get(m_smiDeviceIndex, &gpuMetrics);
            if(status1 == RSMI_STATUS_SUCCESS)
            {
                for(int i = 0; i < m_XCDCount; i++)
                {
                    m_SYSCLK_sum[i] += gpuMetrics.current_gfxclks[i] * cMhzToHz;
                    m_SYSCLK_array[i].push_back(gpuMetrics.current_gfxclks[i] * cMhzToHz);
                }
            }

#else

            //XCD 0
            auto status1 = rsmi_dev_gpu_clk_freq_get(m_smiDeviceIndex, RSMI_CLK_TYPE_SYS, &freq);
            if(status1 == RSMI_STATUS_SUCCESS)
            {
                m_SYSCLK_sum[0] += freq.frequency[freq.current];
                m_SYSCLK_array[0].push_back(freq.frequency[freq.current]);
            }

#endif

            auto status2 = rsmi_dev_gpu_clk_freq_get(m_smiDeviceIndex, RSMI_CLK_TYPE_MEM, &freq);
            if(status2 == RSMI_STATUS_SUCCESS)
            {
                m_MEMCLK_sum += freq.frequency[freq.current];
                m_MEMCLK_array.push_back(freq.frequency[freq.current]);
            }

            // collect freq every 50ms regardless of success
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

        } while(!m_stop && !m_exit);
    }

    void assertActive()
    {
        if(!m_future.valid())
            throw std::runtime_error("Monitor is not active.");
    }

    void assertNotActive()
    {
        if(m_future.valid())
            throw std::runtime_error("Monitor is active.");
    }

    void clearValues()
    {
        m_SYSCLK_sum   = std::vector<uint64_t>(m_XCDCount, 0);
        m_SYSCLK_array = std::vector<std::vector<uint64_t>>(m_XCDCount, std::vector<uint64_t>{});
        m_MEMCLK_sum   = 0;
        m_MEMCLK_array.clear();
    }

    void wait()
    {
        if(!m_future.valid())
            return;

        if(!m_stop)
            throw std::runtime_error("Waiting for monitoring to stop with no end condition.");

        m_future.wait();
        m_future = std::move(std::future<void>());
    }

    void InitROCmSMI()
    {
        static rsmi_status_t status = rsmi_init(0);
        RSMI_CHECK_EXC(status);
    }

    uint32_t GetROCmSMIIndex(int hipDeviceIndex)
    {
        InitROCmSMI();

        hipDeviceProp_t props;

        HIP_CHECK_EXC(hipGetDeviceProperties(&props, hipDeviceIndex));
        m_CUCount = props.multiProcessorCount;
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

        uint64_t hipPCIID = 0;
        // hipPCIID |= props.pciDeviceID & 0xFF;
        // hipPCIID |= ((props.pciBusID & 0xFF) << 8);
        // hipPCIID |= (props.pciDomainID) << 16;

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

    using Task = std::packaged_task<void(void)>;
    Task                    m_task;
    std::atomic<bool>       m_exit;
    std::atomic<bool>       m_stop;
    std::future<void>       m_future;
    std::thread             m_thread;
    std::condition_variable m_cv;
    std::mutex              m_mutex;
    uint32_t                m_smiDeviceIndex;
    bool                    m_isMultiXCDSupported;
    uint16_t                m_XCDCount;
    uint16_t                m_CUCount;

    std::vector<uint64_t>              m_SYSCLK_sum;
    std::vector<std::vector<uint64_t>> m_SYSCLK_array;
    uint64_t                           m_MEMCLK_sum;
    std::vector<uint64_t>              m_MEMCLK_array;

#else // WIN32

    // not supporting windows for now

public:
    FrequencyMonitorImp() {}

    ~FrequencyMonitorImp() {}

    void set_device_id(int deviceId) {}

    void start() {}

    void stop() {}

    bool enabled()
    {
        return false;
    }

    bool detailedReport()
    {
        return false;
    }

    double getLowestAverageSYSCLK()
    {
        return 0.0;
    }

    double getLowestMedianSYSCLK()
    {
        return 0.0;
    }

    std::vector<double> getAllAverageSYSCLK()
    {
        return std::vector<double>();
    }

    std::vector<double> getAllMedianSYSCLK()
    {
        return std::vector<double>();
    }

    double getAverageMEMCLK()
    {
        return 0.0;
    }

    double getMedianMEMCLK()
    {
        return 0.0;
    }

    double getCuCount()
    {
        return 0.0;
    }

#endif
};

static FrequencyMonitorImp* g_FreqMonitorInstance{nullptr};

FrequencyMonitor& getFrequencyMonitor()
{
    if(g_FreqMonitorInstance == nullptr)
    {
        g_FreqMonitorInstance = new FrequencyMonitorImp();
    }
    return *g_FreqMonitorInstance;
}

void freeFrequencyMonitor()
{
    if(g_FreqMonitorInstance != nullptr)
    {
        delete g_FreqMonitorInstance;
        g_FreqMonitorInstance = nullptr;
    }
}

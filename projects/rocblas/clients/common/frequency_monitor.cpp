
/* ************************************************************************
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "client_utility.hpp"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <numeric>

#ifndef _WIN32

#include <amd_smi/amdsmi.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

namespace
{
    bool env_set(const char* name)
    {
        return std::getenv(name) != nullptr;
    }

    double mean(const std::vector<double>& data)
    {
        return data.empty() ? 0.0 : std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    }

    double median(std::vector<double> data)
    {
        if(data.empty())
            return 0.0;
        size_t mid = data.size() / 2;
        std::nth_element(data.begin(), data.begin() + mid, data.end());
        if(data.size() % 2)
            return data[mid];
        return (data[mid] + *std::max_element(data.begin(), data.begin() + mid)) / 2.0;
    }

    // AMD SMI reports unsupported gpu_metrics fields as all ones
    bool valid_clock(uint16_t mhz)
    {
        return mhz != 0 && mhz != UINT16_MAX;
    }

    // AMD SMI is initialized once and shared by the monitors of all threads
    std::mutex& smi_mutex()
    {
        static std::mutex mutex;
        return mutex;
    }

    bool smi_init()
    {
        static bool ok = amdsmi_init(AMDSMI_INIT_AMD_GPUS) == AMDSMI_STATUS_SUCCESS;
        return ok;
    }

    std::vector<amdsmi_processor_handle> smi_processors()
    {
        std::vector<amdsmi_processor_handle> processors;

        uint32_t socket_count = 0;
        if(amdsmi_get_socket_handles(&socket_count, nullptr) != AMDSMI_STATUS_SUCCESS)
            return processors;
        std::vector<amdsmi_socket_handle> sockets(socket_count);
        if(amdsmi_get_socket_handles(&socket_count, sockets.data()) != AMDSMI_STATUS_SUCCESS)
            return processors;

        for(auto socket : sockets)
        {
            uint32_t count = 0;
            if(amdsmi_get_processor_handles(socket, &count, nullptr) != AMDSMI_STATUS_SUCCESS)
                continue;
            size_t offset = processors.size();
            processors.resize(offset + count);
            if(amdsmi_get_processor_handles(socket, &count, &processors[offset])
               != AMDSMI_STATUS_SUCCESS)
                processors.resize(offset);
        }
        return processors;
    }

    // Match the HIP device to an AMD SMI processor by PCI bus/device/function. Where AMD SMI
    // cannot report a BDF (for example WSL or some paravirtualized GPUs), or nothing matches,
    // fall back to the processor with the HIP device index, as hipBLASLt does.
    amdsmi_processor_handle find_processor(int hip_device)
    {
        hipDeviceProp_t props;
        if(hipGetDeviceProperties(&props, hip_device) != hipSuccess)
            return nullptr;

        uint64_t hip_bdf = ((uint64_t(props.pciDomainID) & 0xffffffff) << 32)
                           | ((uint64_t(props.pciBusID) & 0xff) << 8)
                           | ((uint64_t(props.pciDeviceID) & 0x1f) << 3);

        auto processors = smi_processors();
        if(processors.empty())
            return nullptr;

        bool bdf_supported = true;
        for(auto processor : processors)
        {
            uint64_t        smi_bdf = 0;
            amdsmi_status_t status  = amdsmi_get_gpu_bdf_id(processor, &smi_bdf);
            if(status == AMDSMI_STATUS_NOT_SUPPORTED)
            {
                bdf_supported = false;
                break;
            }
            // bits 31:16 may hold a partition id, which HIP does not report
            if(status == AMDSMI_STATUS_SUCCESS && (smi_bdf & ~uint64_t(0xffff0000)) == hip_bdf)
                return processor;
        }

        size_t index = std::min<size_t>(std::max(hip_device, 0), processors.size() - 1);
        rocblas_cerr << "rocblas-bench: "
                     << (bdf_supported
                             ? "no AMD SMI processor matches the PCI ID of HIP device "
                             : "AMD SMI cannot report PCI IDs on this platform; HIP device ")
                     << hip_device << (bdf_supported ? "; " : " ")
                     << "clocks are read from AMD SMI processor " << index << std::endl;
        return processors[index];
    }
}

class FrequencyMonitorImp : public FrequencyMonitor
{
public:
    FrequencyMonitorImp()
        : m_efficiency(env_set("ROCBLAS_BENCH_PERF"))
        , m_detailed(env_set("ROCBLAS_BENCH_FREQ_ALL") && multi_xcd_supported())
        , m_enabled(env_set("ROCBLAS_BENCH_FREQ") || m_detailed || m_efficiency)
    {
    }

    FrequencyMonitorImp(const FrequencyMonitorImp&) = delete;
    FrequencyMonitorImp& operator=(const FrequencyMonitorImp&) = delete;

    ~FrequencyMonitorImp() override
    {
        if(!m_thread.joinable())
            return;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_exit = true;
        }
        m_cv.notify_all();
        m_thread.join();
    }

    bool enabled() override
    {
        return m_enabled;
    }

    bool detailedReport() override
    {
        return m_enabled && m_detailed;
    }

    bool efficiencyReport() override
    {
        return m_enabled && m_efficiency;
    }

    void set_device_id(int deviceId) override
    {
        if(m_enabled)
            bind_device(deviceId);
    }

    void start() override
    {
        int device = -1;
        if(!m_enabled || hipGetDevice(&device) != hipSuccess || !bind_device(device))
            return;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_SYSCLK.assign(m_XCDCount, {});
            m_MEMCLK.clear();
            m_active = true;
            ++m_run;
        }
        m_cv.notify_all();
    }

    void stop() override
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_active = false;
        }
        m_cv.notify_all();
    }

    double getLowestAverageSYSCLK() override
    {
        return lowest(getAllAverageSYSCLK());
    }

    double getLowestMedianSYSCLK() override
    {
        return lowest(getAllMedianSYSCLK());
    }

    std::vector<double> getAllAverageSYSCLK() override
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<double>         clocks(m_XCDCount, 0.0);
        for(size_t i = 0; i < m_SYSCLK.size() && i < clocks.size(); i++)
            clocks[i] = mean(m_SYSCLK[i]);
        return clocks;
    }

    std::vector<double> getAllMedianSYSCLK() override
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<double>         clocks(m_XCDCount, 0.0);
        for(size_t i = 0; i < m_SYSCLK.size() && i < clocks.size(); i++)
            clocks[i] = median(m_SYSCLK[i]);
        return clocks;
    }

    double getAverageMEMCLK() override
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return mean(m_MEMCLK);
    }

    double getMedianMEMCLK() override
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return median(m_MEMCLK);
    }

    double getCuCount() override
    {
        return m_CUCount;
    }

private:
    struct sample
    {
        std::vector<double> gfx; // MHz per XCD, 0 when unavailable
        double              mem = 0;
    };

    static bool multi_xcd_supported()
    {
#if AMDSMI_LIB_VERSION_MAJOR >= 25
        return true;
#else
        return false;
#endif
    }

    // Lowest clock of the XCDs that reported one; the slowest XCD bounds the kernel
    static double lowest(const std::vector<double>& clocks)
    {
        double result = 0;
        for(double clock : clocks)
            if(clock > 0 && (result == 0 || clock < result))
                result = clock;
        return result;
    }

    bool bind_device(int device)
    {
        if(device == m_device)
            return m_processor != nullptr;

        m_device    = device;
        m_processor = nullptr;
        m_XCDCount  = 1;
        m_CUCount   = 0;

        hipDeviceProp_t props;
        if(hipGetDeviceProperties(&props, device) == hipSuccess)
            m_CUCount = props.multiProcessorCount;
        {
            std::lock_guard<std::mutex> lock(smi_mutex());
            if(smi_init())
                m_processor = find_processor(device);
            if(!m_processor)
            {
                rocblas_cerr << "rocblas-bench: AMD SMI cannot query HIP device " << device
                             << "; clock columns are reported as 0" << std::endl;
                return false;
            }
#if AMDSMI_LIB_VERSION_MAJOR >= 25
            uint16_t xcd_count = 0;
            if(amdsmi_get_gpu_xcd_counter(m_processor, &xcd_count) == AMDSMI_STATUS_SUCCESS
               && xcd_count > 0)
                m_XCDCount = std::min<int>(xcd_count, AMDSMI_MAX_NUM_GFX_CLKS);
#endif
        }

        if(!m_thread.joinable())
            m_thread = std::thread([this] { run(); });
        return true;
    }

    sample read_sample()
    {
        sample s;
        s.gfx.assign(m_XCDCount, 0.0);

        std::lock_guard<std::mutex> lock(smi_mutex());
        amdsmi_frequencies_t        freq;

#if AMDSMI_LIB_VERSION_MAJOR >= 25
        amdsmi_gpu_metrics_t metrics;
        bool                 have_metrics
            = amdsmi_get_gpu_metrics_info(m_processor, &metrics) == AMDSMI_STATUS_SUCCESS;
        if(have_metrics)
            for(int i = 0; i < m_XCDCount; i++)
                if(valid_clock(metrics.current_gfxclks[i]))
                    s.gfx[i] = metrics.current_gfxclks[i];
#endif
        if(s.gfx[0] == 0
           && amdsmi_get_clk_freq(m_processor, AMDSMI_CLK_TYPE_SYS, &freq) == AMDSMI_STATUS_SUCCESS
           && freq.current < freq.num_supported)
            s.gfx[0] = freq.frequency[freq.current] * 1e-6;

        if(amdsmi_get_clk_freq(m_processor, AMDSMI_CLK_TYPE_MEM, &freq) == AMDSMI_STATUS_SUCCESS
           && freq.current < freq.num_supported)
            s.mem = freq.frequency[freq.current] * 1e-6;
#if AMDSMI_LIB_VERSION_MAJOR >= 25
        // Some GPUs report the memory clock only as the UMC clock in gpu_metrics
        else if(have_metrics && valid_clock(metrics.current_uclk))
            s.mem = metrics.current_uclk;
#endif
        return s;
    }

    // m_mutex must be held
    void store(const sample& s)
    {
        for(size_t i = 0; i < s.gfx.size() && i < m_SYSCLK.size(); i++)
            if(s.gfx[i] > 0)
                m_SYSCLK[i].push_back(s.gfx[i]);
        if(s.mem > 0)
            m_MEMCLK.push_back(s.mem);
    }

    void run()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        while(!m_exit)
        {
            m_cv.wait(lock, [this] { return m_exit || m_active; });

            const uint64_t run     = m_run;
            auto           stopped = [this, run] { return m_exit || !m_active || run != m_run; };

            // start() follows a stream synchronization, so the GPU is idle; let the timed calls
            // start before the first sample. Loops shorter than this report 0 (no sample).
            if(m_cv.wait_for(lock, std::chrono::milliseconds(5), stopped))
                continue;

            // then sample every 50 ms, as hipBLASLt does
            do
            {
                lock.unlock();
                sample s = read_sample();
                lock.lock();
                if(!stopped())
                    store(s);
            } while(!m_cv.wait_for(lock, std::chrono::milliseconds(50), stopped));
        }
    }

    const bool m_efficiency;
    const bool m_detailed;
    const bool m_enabled;

    int                     m_device    = -1;
    amdsmi_processor_handle m_processor = nullptr;
    int                     m_XCDCount  = 1;
    double                  m_CUCount   = 0;

    std::thread             m_thread;
    std::mutex              m_mutex;
    std::condition_variable m_cv;
    bool                    m_active = false;
    bool                    m_exit   = false;
    uint64_t                m_run    = 0;

    std::vector<std::vector<double>> m_SYSCLK;
    std::vector<double>              m_MEMCLK;
};

#else // _WIN32: not supported

class FrequencyMonitorImp : public FrequencyMonitor
{
public:
    bool enabled() override
    {
        return false;
    }

    bool detailedReport() override
    {
        return false;
    }

    bool efficiencyReport() override
    {
        return false;
    }

    void set_device_id(int) override {}
    void start() override {}
    void stop() override {}

    double getLowestAverageSYSCLK() override
    {
        return 0.0;
    }

    double getLowestMedianSYSCLK() override
    {
        return 0.0;
    }

    std::vector<double> getAllAverageSYSCLK() override
    {
        return {};
    }

    std::vector<double> getAllMedianSYSCLK() override
    {
        return {};
    }

    double getAverageMEMCLK() override
    {
        return 0.0;
    }

    double getMedianMEMCLK() override
    {
        return 0.0;
    }

    double getCuCount() override
    {
        return 0.0;
    }
};

#endif

static thread_local std::unique_ptr<FrequencyMonitorImp> t_frequency_monitor;

FrequencyMonitor& getFrequencyMonitor()
{
    if(!t_frequency_monitor)
        t_frequency_monitor = std::make_unique<FrequencyMonitorImp>();
    return *t_frequency_monitor;
}

void freeFrequencyMonitor()
{
    t_frequency_monitor.reset();
}

/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>

#if defined(HIPBLAS_BENCH) && defined(HIPBLAS_HAVE_AMD_SMI)

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

    // amd-smi reports unsupported gpu_metrics fields as all ones
    bool valid_clock(uint16_t mhz)
    {
        return mhz != 0 && mhz != UINT16_MAX;
    }

    // amd-smi is initialized once and shared by the monitors of all threads
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

    // Match the HIP device to an amd-smi processor by PCI bus/device/function
    amdsmi_processor_handle find_processor(int hip_device)
    {
        hipDeviceProp_t props;
        if(hipGetDeviceProperties(&props, hip_device) != hipSuccess)
            return nullptr;

        uint64_t hip_bdf = ((uint64_t(props.pciDomainID) & 0xffffffff) << 32)
                           | ((uint64_t(props.pciBusID) & 0xff) << 8)
                           | ((uint64_t(props.pciDeviceID) & 0x1f) << 3);

        uint32_t socket_count = 0;
        if(amdsmi_get_socket_handles(&socket_count, nullptr) != AMDSMI_STATUS_SUCCESS)
            return nullptr;
        std::vector<amdsmi_socket_handle> sockets(socket_count);
        if(amdsmi_get_socket_handles(&socket_count, sockets.data()) != AMDSMI_STATUS_SUCCESS)
            return nullptr;

        for(auto socket : sockets)
        {
            uint32_t processor_count = 0;
            if(amdsmi_get_processor_handles(socket, &processor_count, nullptr)
               != AMDSMI_STATUS_SUCCESS)
                continue;
            std::vector<amdsmi_processor_handle> processors(processor_count);
            if(amdsmi_get_processor_handles(socket, &processor_count, processors.data())
               != AMDSMI_STATUS_SUCCESS)
                continue;

            for(auto processor : processors)
            {
                uint64_t smi_bdf = 0;
                // bits 31:16 may hold a partition id, which HIP does not report
                if(amdsmi_get_gpu_bdf_id(processor, &smi_bdf) == AMDSMI_STATUS_SUCCESS
                   && (smi_bdf & ~uint64_t(0xffff0000)) == hip_bdf)
                    return processor;
            }
        }
        return nullptr;
    }
}

class FrequencyMonitorImp : public FrequencyMonitor
{
public:
    FrequencyMonitorImp()
        : m_enabled(env_set("HIPBLAS_BENCH_FREQ") || env_set("HIPBLAS_BENCH_FREQ_ALL"))
        , m_detailed(env_set("HIPBLAS_BENCH_FREQ_ALL"))
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

    void start() override
    {
        begin(false);
    }

    void start(hipStream_t stream) override
    {
        if(!m_enabled)
            return;
        // the sampling thread waits for this event, which the stream reaches when the timed
        // calls queued after it start. Not destroyed: the monitor is thread_local and may outlive
        // the HIP runtime at exit.
        if(!m_anchor && hipEventCreateWithFlags(&m_anchor, hipEventDisableTiming) != hipSuccess)
            m_anchor = nullptr;
        begin(m_anchor && hipEventRecord(m_anchor, stream) == hipSuccess);
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
        std::vector<double>         clocks;
        for(auto& xcd : m_SYSCLK)
            clocks.push_back(mean(xcd));
        return clocks;
    }

    std::vector<double> getAllMedianSYSCLK() override
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<double>         clocks;
        for(auto& xcd : m_SYSCLK)
            clocks.push_back(median(xcd));
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

private:
    struct sample
    {
        std::vector<double> gfx; // MHz per XCD, 0 when unavailable
        double              mem = 0;
    };

    void begin(bool anchored)
    {
        if(!m_enabled || !bind_device())
            return;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_SYSCLK.assign(m_XCDCount, {});
            m_MEMCLK.clear();
            m_anchored = anchored;
            m_active   = true;
            ++m_run;
        }
        m_cv.notify_all();
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

    // Called from the benchmark thread only. The sampling thread reads m_processor and
    // m_XCDCount under m_mutex, as a snapshot per run, so they change only under m_mutex.
    bool bind_device()
    {
        int device = -1;
        if(hipGetDevice(&device) != hipSuccess)
            return false;
        if(device == m_device)
            return m_processor != nullptr;

        m_device = device;

        amdsmi_processor_handle processor = nullptr;
        int                     xcd_count = 1;
        {
            std::lock_guard<std::mutex> lock(smi_mutex());
            if(smi_init())
                processor = find_processor(device);
#if AMDSMI_LIB_VERSION_MAJOR >= 25
            uint16_t count = 0;
            if(processor && amdsmi_get_gpu_xcd_counter(processor, &count) == AMDSMI_STATUS_SUCCESS
               && count > 0)
                xcd_count = std::min<int>(count, AMDSMI_MAX_NUM_GFX_CLKS);
#endif
        }
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_processor = processor;
            m_XCDCount  = xcd_count;
        }
        if(!processor)
        {
            std::cerr << "hipblas-bench: amd-smi cannot query HIP device " << device
                      << "; clock columns are reported as 0" << std::endl;
            return false;
        }

        if(!m_thread.joinable())
            m_thread = std::thread([this] { run(); });
        return true;
    }

    // processor and xcd_count are the run's snapshot, taken under m_mutex
    static sample read_sample(amdsmi_processor_handle processor, int xcd_count)
    {
        sample s;
        s.gfx.assign(xcd_count, 0.0);

        std::lock_guard<std::mutex> lock(smi_mutex());
        amdsmi_frequencies_t        freq;

#if AMDSMI_LIB_VERSION_MAJOR >= 25
        amdsmi_gpu_metrics_t metrics;
        bool                 have_metrics
            = amdsmi_get_gpu_metrics_info(processor, &metrics) == AMDSMI_STATUS_SUCCESS;
        if(have_metrics)
            for(int i = 0; i < xcd_count; i++)
                if(valid_clock(metrics.current_gfxclks[i]))
                    s.gfx[i] = metrics.current_gfxclks[i];
#endif
        if(s.gfx[0] == 0
           && amdsmi_get_clk_freq(processor, AMDSMI_CLK_TYPE_SYS, &freq) == AMDSMI_STATUS_SUCCESS
           && freq.current < freq.num_supported)
            s.gfx[0] = freq.frequency[freq.current] * 1e-6;

        if(amdsmi_get_clk_freq(processor, AMDSMI_CLK_TYPE_MEM, &freq) == AMDSMI_STATUS_SUCCESS
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

            const uint64_t                run       = m_run;
            const bool                    anchored  = m_anchored;
            const amdsmi_processor_handle processor = m_processor;
            const int                     xcd_count = m_XCDCount;
            auto stopped = [this, run] { return m_exit || !m_active || run != m_run; };

            // start(stream): wait until the stream reaches the timed calls; start() is called
            // before the timer synchronizes, so cold calls may still be running
            if(anchored)
            {
                lock.unlock();
                (void)hipEventSynchronize(m_anchor);
                lock.lock();
                if(stopped())
                    continue;
            }

            // The first sample is 5 ms after the timed calls start, so loops shorter than that
            // report 0 (no sample)
            if(m_cv.wait_for(lock, std::chrono::milliseconds(5), stopped))
                continue;

            // then sample every 50 ms, as hipBLASLt does
            do
            {
                lock.unlock();
                sample s = read_sample(processor, xcd_count);
                lock.lock();
                if(!stopped())
                    store(s);
            } while(!m_cv.wait_for(lock, std::chrono::milliseconds(50), stopped));
        }
    }

    const bool m_enabled;
    const bool m_detailed;

    int                     m_device    = -1;
    amdsmi_processor_handle m_processor = nullptr;
    int                     m_XCDCount  = 1;

    std::thread             m_thread;
    std::mutex              m_mutex;
    std::condition_variable m_cv;
    bool                    m_active   = false;
    bool                    m_exit     = false;
    bool                    m_anchored = false;
    uint64_t                m_run      = 0;
    hipEvent_t              m_anchor   = nullptr; // recorded by start(stream)

    std::vector<std::vector<double>> m_SYSCLK;
    std::vector<double>              m_MEMCLK;
};

#else // no amd-smi, or not hipblas-bench

class FrequencyMonitorImp : public FrequencyMonitor
{
public:
    FrequencyMonitorImp()
    {
#if defined(HIPBLAS_BENCH)
        static bool warned = false;
        if(!warned && (std::getenv("HIPBLAS_BENCH_FREQ") || std::getenv("HIPBLAS_BENCH_FREQ_ALL")))
        {
            warned = true;
            std::cerr << "hipblas-bench: built without amd-smi; HIPBLAS_BENCH_FREQ is ignored"
                      << std::endl;
        }
#endif
    }

    bool enabled() override
    {
        return false;
    }

    bool detailedReport() override
    {
        return false;
    }

    void start() override {}
    void start(hipStream_t) override {}
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
};

#endif

FrequencyMonitor& getFrequencyMonitor()
{
    thread_local FrequencyMonitorImp monitor;
    return monitor;
}

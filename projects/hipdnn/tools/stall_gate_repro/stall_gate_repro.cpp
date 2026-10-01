// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Measures HIP event timing accuracy with and without the hipDNN StallGate.
//
// Oracle: a spin kernel times itself with wall_clock64(). The event span
// brackets the whole kernel, so a correct event elapsed time is never shorter
// than the kernel's own time. Each row reports event-minus-device deltas;
// "viol" counts samples where the event span was shorter than the device time,
// which is impossible for correct timestamps. Memset and empty rows have no
// device oracle: compare stalled and unstalled rows, and look for negative
// readings ("neg").
//
// Build (Linux):   hipcc -std=c++17 -O2
// -I<repo>/projects/hipdnn/data_sdk/include \
//                      stall_gate_repro.cpp -o stall_gate_repro -lpthread
// Build (Windows): hipcc -std=c++17 -O2
// -I<repo>/projects/hipdnn/data_sdk/include ^
//                      stall_gate_repro.cpp -o stall_gate_repro.exe
// Run:             stall_gate_repro [reps=50] [device=0]

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/StallGate.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK_HIP(expr)                               \
    do                                                \
    {                                                 \
        const hipError_t status_ = (expr);            \
        if(status_ != hipSuccess)                     \
        {                                             \
            std::fprintf(stderr,                      \
                         "%s:%d: %s failed: %s\n",    \
                         __FILE__,                    \
                         __LINE__,                    \
                         #expr,                       \
                         hipGetErrorString(status_)); \
            std::exit(1);                             \
        }                                             \
    } while(0)

namespace
{

using hipdnn_data_sdk::utilities::StallGate;

__global__ void spinKernel(uint64_t ticks, uint64_t* out)
{
    const uint64_t start = wall_clock64();
    uint64_t now = start;
    while(now - start < ticks)
    {
        now = wall_clock64();
    }
    out[0] = start;
    out[1] = now;
}

enum class Kind
{
    EMPTY,
    SPIN,
    MEMSET
};

struct Workload
{
    Kind kind;
    double spinUs = 0.0;
    size_t bytes = 0;

    std::string name() const
    {
        char buf[64];
        switch(kind)
        {
        case Kind::EMPTY:
            return "empty";
        case Kind::SPIN:
            std::snprintf(buf, sizeof(buf), "spin %gus", spinUs);
            return buf;
        case Kind::MEMSET:
            if(bytes >= (size_t{1} << 20))
            {
                std::snprintf(buf, sizeof(buf), "memset %zuMiB", bytes >> 20);
            }
            else
            {
                std::snprintf(buf, sizeof(buf), "memset %zuKiB", bytes >> 10);
            }
            return buf;
        }
        return "?";
    }
};

struct Sample
{
    double eventUs = 0.0;
    double deviceUs = -1.0; // < 0: no device oracle for this workload
    bool stallUsed = false;
    bool timedOut = false;
};

struct Context
{
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    uint64_t* spinOut = nullptr;
    void* scratch = nullptr;
    double wallClockKHz = 0.0;
    StallGate* gate = nullptr;
};

void busyWaitUs(double us)
{
    const auto until
        = std::chrono::steady_clock::now() + std::chrono::duration<double, std::micro>(us);
    while(std::chrono::steady_clock::now() < until)
    {
    }
}

// Mirrors the production sequence: idle stream, optional arm, START, host work,
// kernel, STOP, release, synchronize. hostDelayUs stands in for hipDNN host
// code that runs between START and the kernel launch.
Sample measure(const Context& ctx, const Workload& w, bool stalled, double hostDelayUs)
{
    CHECK_HIP(hipStreamSynchronize(ctx.stream));

    Sample s;
    s.stallUsed = stalled && ctx.gate->arm(ctx.stream);
    CHECK_HIP(hipEventRecord(ctx.start, ctx.stream));
    if(hostDelayUs > 0.0)
    {
        busyWaitUs(hostDelayUs);
    }

    const auto ticks = static_cast<uint64_t>(w.spinUs * ctx.wallClockKHz / 1000.0);
    switch(w.kind)
    {
    case Kind::EMPTY:
        break;
    case Kind::SPIN:
        spinKernel<<<1, 1, 0, ctx.stream>>>(ticks, ctx.spinOut);
        CHECK_HIP(hipGetLastError());
        break;
    case Kind::MEMSET:
        CHECK_HIP(hipMemsetAsync(ctx.scratch, 0, w.bytes, ctx.stream));
        break;
    }

    CHECK_HIP(hipEventRecord(ctx.stop, ctx.stream));
    if(s.stallUsed)
    {
        ctx.gate->release();
        s.timedOut = ctx.gate->timedOut();
    }
    CHECK_HIP(hipEventSynchronize(ctx.stop));

    float ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&ms, ctx.start, ctx.stop));
    s.eventUs = static_cast<double>(ms) * 1000.0;

    if(w.kind == Kind::SPIN)
    {
        uint64_t stamps[2] = {};
        CHECK_HIP(hipMemcpy(stamps, ctx.spinOut, sizeof(stamps), hipMemcpyDeviceToHost));
        s.deviceUs = static_cast<double>(stamps[1] - stamps[0]) * 1000.0 / ctx.wallClockKHz;
    }
    return s;
}

double median(std::vector<double> v)
{
    if(v.empty())
    {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

void runRow(const Context& ctx, const Workload& w, bool stalled, double hostDelayUs, int reps)
{
    for(int i = 0; i < 3; ++i)
    {
        (void)measure(ctx, w, stalled, hostDelayUs);
    }

    std::vector<double> events;
    std::vector<double> devices;
    std::vector<double> deltas;
    int negative = 0;
    int violations = 0;
    int declined = 0;
    int timedOut = 0;
    for(int i = 0; i < reps; ++i)
    {
        const Sample s = measure(ctx, w, stalled, hostDelayUs);
        if(stalled && !s.stallUsed)
        {
            ++declined;
        }
        if(s.timedOut)
        {
            ++timedOut;
        }
        events.push_back(s.eventUs);
        negative += s.eventUs < 0.0 ? 1 : 0;
        if(s.deviceUs >= 0.0)
        {
            devices.push_back(s.deviceUs);
            deltas.push_back(s.eventUs - s.deviceUs);
            violations += s.eventUs < s.deviceUs ? 1 : 0;
        }
    }

    const auto [evMin, evMax] = std::minmax_element(events.begin(), events.end());
    std::printf("%-15s %-9s %6.0f %10.2f %10.2f %10.2f",
                w.name().c_str(),
                stalled ? "stalled" : "unstalled",
                hostDelayUs,
                *evMin,
                median(events),
                *evMax);
    if(!deltas.empty())
    {
        std::printf(" %10.2f %10.2f %10.2f",
                    median(devices),
                    median(deltas),
                    *std::min_element(deltas.begin(), deltas.end()));
    }
    else
    {
        std::printf(" %10s %10s %10s", "-", "-", "-");
    }
    std::printf(" %4d %4d", negative, violations);
    if(declined > 0 || timedOut > 0)
    {
        std::printf("  declined=%d timedOut=%d", declined, timedOut);
    }
    std::printf("\n");
}

} // namespace

int main(int argc, char** argv)
{
    const int reps = argc > 1 ? std::max(1, std::atoi(argv[1])) : 50;
    const int device = argc > 2 ? std::atoi(argv[2]) : 0;
    CHECK_HIP(hipSetDevice(device));

    hipDeviceProp_t props{};
    CHECK_HIP(hipGetDeviceProperties(&props, device));
    int wallClockKHz = 0;
    CHECK_HIP(hipDeviceGetAttribute(&wallClockKHz, hipDeviceAttributeWallClockRate, device));
    if(wallClockKHz <= 0)
    {
        std::fprintf(stderr,
                     "hipDeviceAttributeWallClockRate is %d; the spin oracle cannot run\n",
                     wallClockKHz);
        return 1;
    }
    int runtimeVersion = 0;
    CHECK_HIP(hipRuntimeGetVersion(&runtimeVersion));

    constexpr size_t MAX_MEMSET_BYTES = size_t{128} << 20;
    Context ctx;
    ctx.wallClockKHz = static_cast<double>(wallClockKHz);
    CHECK_HIP(hipStreamCreate(&ctx.stream));
    CHECK_HIP(hipEventCreate(&ctx.start));
    CHECK_HIP(hipEventCreate(&ctx.stop));
    CHECK_HIP(hipMalloc(&ctx.spinOut, 2 * sizeof(uint64_t)));
    CHECK_HIP(hipMalloc(&ctx.scratch, MAX_MEMSET_BYTES));
    StallGate gate;
    ctx.gate = &gate;

    std::printf("device %d: %s (%s), wall clock %d kHz, HIP runtime %d, reps %d\n",
                device,
                props.name,
                props.gcnArchName,
                wallClockKHz,
                runtimeVersion,
                reps);
    std::printf("stall gate usable: %s\n", gate.isUsable() ? "yes" : "no");
    std::printf("units: us. delta = event - device (spin only). viol = event < "
                "device.\n\n");
    std::printf("%-15s %-9s %6s %10s %10s %10s %10s %10s %10s %4s %4s\n",
                "workload",
                "mode",
                "delay",
                "ev_min",
                "ev_med",
                "ev_max",
                "dev_med",
                "d_med",
                "d_min",
                "neg",
                "viol");

    std::vector<Workload> workloads{{Kind::EMPTY}};
    for(const double us : {0.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 1000.0})
    {
        workloads.push_back({Kind::SPIN, us});
    }
    for(const size_t kib : {4, 64, 1024, 4096, 16384, 65536, 131072})
    {
        workloads.push_back({Kind::MEMSET, 0.0, kib << 10});
    }

    for(const double delayUs : {0.0, 200.0})
    {
        for(const Workload& w : workloads)
        {
            runRow(ctx, w, /*stalled=*/false, delayUs, reps);
            if(gate.isUsable())
            {
                runRow(ctx, w, /*stalled=*/true, delayUs, reps);
            }
        }
        std::printf("\n");
    }

    CHECK_HIP(hipFree(ctx.scratch));
    CHECK_HIP(hipFree(ctx.spinOut));
    CHECK_HIP(hipEventDestroy(ctx.stop));
    CHECK_HIP(hipEventDestroy(ctx.start));
    CHECK_HIP(hipStreamDestroy(ctx.stream));
    return 0;
}

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Q7 microbench source for RFC 0001 (Phase 1 investigation).
//
// Times the cost of miopenSetTensor on a 1x1x1x1 fp32 tensor over a tight
// loop, prints the per-call latency in nanoseconds. Built and run by
// wrapper_overhead.sh against both the flag-off and flag-on libMIOpen.so.
//
// We deliberately pick the smallest tensor in the smallest layout so kernel
// launch and memory-system cost are constant across configurations and the
// delta we measure is dominated by the wrapper indirection (one extra call
// frame under Option A; a function-pointer through dlsym under Option B).

#include <miopen/miopen.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#ifndef ITERATIONS
#define ITERATIONS 1000000
#endif

int main()
{
    miopenHandle_t handle = nullptr;
    if(miopenCreate(&handle) != miopenStatusSuccess)
    {
        // No GPU available; emit a sentinel and let the harness detect it.
        std::printf("0\n");
        return 0;
    }

    miopenTensorDescriptor_t tensorDesc = nullptr;
    if(miopenCreateTensorDescriptor(&tensorDesc) != miopenStatusSuccess)
    {
        std::printf("0\n");
        miopenDestroy(handle);
        return 0;
    }
    miopenSet4dTensorDescriptor(tensorDesc, miopenFloat, 1, 1, 1, 1);

    float value = 1.0f;
    // We need a backing GPU buffer for the tensor. If the tensor's memory
    // model requires that we provide a real device pointer, this microbench
    // will return early via miopenSetTensor's status check; the wrapper
    // overhead is still the same.
    float dummy      = 0.0f;
    void* device_ptr = static_cast<void*>(&dummy);

    auto t0 = std::chrono::steady_clock::now();
    for(int i = 0; i < ITERATIONS; ++i)
    {
        // Status is intentionally ignored — we are timing the call site,
        // not the operation. The wrapper indirection cost is identical in
        // success and failure paths.
        (void)miopenSetTensor(handle, tensorDesc, device_ptr, &value);
    }
    auto t1 = std::chrono::steady_clock::now();

    auto ns_total         = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    long long per_call_ns = ns_total / ITERATIONS;

    miopenDestroyTensorDescriptor(tensorDesc);
    miopenDestroy(handle);

    std::printf("%lld\n", per_call_ns);
    return 0;
}

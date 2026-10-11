// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <string>
#include <utility>

namespace ck_tile {
namespace dispatcher {

using ConvDeviceArchProvider = std::function<std::string(int)>;

struct ConvInvocationContext
{
    const void* input_ptr                = nullptr;
    const void* weight_ptr               = nullptr;
    void* output_ptr                     = nullptr;
    int device_ordinal                   = -1;
    int k_batch                          = 1;
    ConvDeviceArchProvider arch_provider = {};

    // Defaults keep the production path at exactly one launch; only the profiler opts in.
    bool benchmarking = false;
    int warmup        = 0;
    int repeat        = 1;
};

inline ConvInvocationContext& mutable_conv_invocation_context()
{
    thread_local ConvInvocationContext context;
    return context;
}

inline const ConvInvocationContext& conv_invocation_context()
{
    return mutable_conv_invocation_context();
}

class ScopedConvInvocationContext
{
    public:
    explicit ScopedConvInvocationContext(ConvInvocationContext context)
        : previous_(std::move(mutable_conv_invocation_context()))
    {
        mutable_conv_invocation_context() = std::move(context);
    }

    ~ScopedConvInvocationContext() { mutable_conv_invocation_context() = std::move(previous_); }

    ScopedConvInvocationContext(const ScopedConvInvocationContext&)            = delete;
    ScopedConvInvocationContext& operator=(const ScopedConvInvocationContext&) = delete;

    private:
    ConvInvocationContext previous_;
};

} // namespace dispatcher
} // namespace ck_tile

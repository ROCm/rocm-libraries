// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

#include "IReferenceGraphExecutor.hpp"
#include "ReferenceCapabilityError.hpp"

namespace hipdnn_integration_tests
{

// Adapts the test_sdk CPU reference executor to IReferenceGraphExecutor. The
// executor is held behind a pimpl so the heavy test_sdk executor header is
// compiled once (CpuReferenceGraphExecutorAdapter.cpp) instead of in every
// translation unit that needs a CPU reference.
class CpuReferenceGraphExecutorAdapter : public IReferenceGraphExecutor
{
public:
    CpuReferenceGraphExecutorAdapter();
    ~CpuReferenceGraphExecutorAdapter() override;

    CpuReferenceGraphExecutorAdapter(const CpuReferenceGraphExecutorAdapter&) = delete;
    CpuReferenceGraphExecutorAdapter& operator=(const CpuReferenceGraphExecutorAdapter&) = delete;
    CpuReferenceGraphExecutorAdapter(CpuReferenceGraphExecutorAdapter&&) = delete;
    CpuReferenceGraphExecutorAdapter& operator=(CpuReferenceGraphExecutorAdapter&&) = delete;

    bool isApplicable(void* graphBuffer, size_t size) override;

    // Translates the test_sdk CpuReferenceNotApplicableError into a
    // ReferenceCapabilityError; any other exception propagates unchanged.
    void execute(void* graphBuffer,
                 size_t size,
                 const std::unordered_map<int64_t, void*>& variantPack) override;

    bool requiresDeviceMemory() const override
    {
        return false;
    }

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace hipdnn_integration_tests

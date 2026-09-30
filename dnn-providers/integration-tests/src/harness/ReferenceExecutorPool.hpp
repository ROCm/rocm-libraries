// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "harness/IReferenceExecutors.hpp"

namespace hipdnn_integration_tests
{

/// The real reference executors, built on first use and kept for the run.
///
/// Construction is lazy because a run that never falls back to a reference should
/// not pay for one, and because the GPU executor's registry is only worth building
/// if something asks for it.
class ReferenceExecutorPool : public IReferenceExecutors
{
public:
    IReferenceGraphExecutor& get(ReferenceExecutorType type) override;

private:
    std::unique_ptr<IReferenceGraphExecutor> _cpu;
    std::unique_ptr<IReferenceGraphExecutor> _gpu;
};

/// The process-wide pool.
///
/// Shared deliberately: a per-test pool would rebuild the GPU plan-builder registry
/// once per bundle, which is exactly what this type exists to stop. Safe to share
/// because the executors hold no per-graph state between calls. GTest runs bodies
/// sequentially, so this needs no lock.
std::shared_ptr<IReferenceExecutors> sharedReferenceExecutors();

} // namespace hipdnn_integration_tests

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "harness/IReferenceGraphExecutor.hpp"
#include "harness/TestConfig.hpp"

namespace hipdnn_integration_tests
{

/// Owns the run's reference executors and lends them out.
///
/// A container rather than a factory on purpose. The executors are reusable — a
/// GpuReferenceGraphExecutor's plan-builder registry is a lookup table with no
/// per-graph state — so building a fresh one per bundle threw the table away
/// hundreds of times per run and made the caller own something it only borrows.
///
/// Lifetime: the returned reference stays valid for the life of the container, so
/// the production container is held for the whole process.
class IReferenceExecutors
{
public:
    IReferenceExecutors() = default;
    virtual ~IReferenceExecutors() = default;

    IReferenceExecutors(const IReferenceExecutors&) = delete;
    IReferenceExecutors& operator=(const IReferenceExecutors&) = delete;
    IReferenceExecutors(IReferenceExecutors&&) = delete;
    IReferenceExecutors& operator=(IReferenceExecutors&&) = delete;

    /// The executor for `type`. Throws std::runtime_error for an unknown type.
    virtual IReferenceGraphExecutor& get(ReferenceExecutorType type) = 0;
};

} // namespace hipdnn_integration_tests

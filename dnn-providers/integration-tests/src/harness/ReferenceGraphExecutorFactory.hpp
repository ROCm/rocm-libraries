// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include "IReferenceGraphExecutor.hpp"
#include "TestConfig.hpp"

namespace hipdnn_integration_tests
{

class ReferenceGraphExecutorFactory
{
public:
    static std::unique_ptr<IReferenceGraphExecutor> create(ReferenceExecutorType type);
};

} // namespace hipdnn_integration_tests

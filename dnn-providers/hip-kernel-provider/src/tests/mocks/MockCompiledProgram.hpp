// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include "compilation/ICompiledProgram.hpp"

namespace hip_kernel_provider
{

class MockCompiledProgram : public compilation::ICompiledProgram
{
public:
    MOCK_METHOD(std::unique_ptr<compilation::IRunnableKernel>,
                getKernel,
                (const std::string& kernelName),
                (const, override));
};

} // namespace hip_kernel_provider

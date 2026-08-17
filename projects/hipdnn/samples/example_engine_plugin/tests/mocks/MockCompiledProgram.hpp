// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gmock/gmock.h>

#include "hip/ICompiledProgram.hpp"

namespace example_provider
{

class MockCompiledProgram : public ICompiledProgram
{
public:
    MOCK_METHOD(std::unique_ptr<IRunnableKernel>,
                getRunnableKernel,
                (const std::string& kernelFunctionName),
                (const, override));
};

} // namespace example_provider

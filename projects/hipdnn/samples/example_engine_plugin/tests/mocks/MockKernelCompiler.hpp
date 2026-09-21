// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gmock/gmock.h>

#include "hip/IKernelCompiler.hpp"

namespace example_provider
{

class MockKernelCompiler : public IKernelCompiler
{
public:
    MOCK_METHOD(std::unique_ptr<ICompiledProgram>,
                compile,
                (const std::string& kernelFileName, const std::vector<std::string>& options),
                (const, override));
};

} // namespace example_provider

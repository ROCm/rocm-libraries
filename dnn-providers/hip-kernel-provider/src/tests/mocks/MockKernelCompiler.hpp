// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include "compilation/IKernelCompiler.hpp"

namespace hip_kernel_provider
{

class MockKernelCompiler : public compilation::IKernelCompiler
{
public:
    MOCK_METHOD(std::unique_ptr<compilation::ICompiledProgram>,
                compile,
                (const std::string& kernelFileName, const std::vector<std::string>& options),
                (const, override));
};

} // namespace hip_kernel_provider

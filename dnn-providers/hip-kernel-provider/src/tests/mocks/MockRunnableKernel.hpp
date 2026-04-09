// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include "hip/IRunnableKernel.hpp"

namespace hip_kernel_provider
{

class MockRunnableKernel : public IRunnableKernel
{
public:
    MOCK_METHOD(void, setBlockSize, (unsigned int x, unsigned int y, unsigned int z), (override));
    MOCK_METHOD(void, setGridSize, (unsigned int x, unsigned int y, unsigned int z), (override));
    MOCK_METHOD(void, setSharedMemBytes, (unsigned int bytes), (override));
    MOCK_METHOD(void, launchImpl, (hipStream_t stream, void** kernelParams), (const, override));
};

} // namespace hip_kernel_provider

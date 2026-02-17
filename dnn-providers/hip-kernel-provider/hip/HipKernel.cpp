// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipKernel.hpp"
#include "HipProgram.hpp"

HipKernel::HipKernel(const HipProgram& program, const std::string& kernelName)
    : kernelName_(kernelName)
    , kernel_(program.GetKernel(kernelName))
{
}

void HipKernel::SetBlockSize(unsigned int x, unsigned int y, unsigned int z)
{
    blockX_ = x;
    blockY_ = y;
    blockZ_ = z;
}

void HipKernel::SetGridSize(unsigned int x, unsigned int y, unsigned int z)
{
    gridX_ = x;
    gridY_ = y;
    gridZ_ = z;
}

void HipKernel::SetSharedMemBytes(unsigned int bytes)
{
    sharedMemBytes_ = bytes;
}

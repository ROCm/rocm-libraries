// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <miopen/miopen.h>

// RAII wrapper for miopenConvolutionDescriptor_t to prevent memory leaks
class ConvDescGuard
{
public:
    ConvDescGuard() : status(miopenCreateConvolutionDescriptor(&desc)) {}

    ~ConvDescGuard()
    {
        if(desc != nullptr)
        {
            miopenDestroyConvolutionDescriptor(desc);
        }
    }

    operator miopenConvolutionDescriptor_t() { return desc; }

    miopenConvolutionDescriptor_t get() { return desc; }
    miopenStatus_t getStatus() const { return status; }

    ConvDescGuard(const ConvDescGuard&)            = delete;
    ConvDescGuard& operator=(const ConvDescGuard&) = delete;
    ConvDescGuard(ConvDescGuard&&)                 = delete;
    ConvDescGuard& operator=(ConvDescGuard&&)      = delete;

private:
    miopenConvolutionDescriptor_t desc = nullptr;
    miopenStatus_t status;
};

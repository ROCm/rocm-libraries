// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <miopen/miopen.h>

// RAII wrapper for miopenRNNDescriptor_t to prevent memory leaks
class RNNDescGuard
{
public:
    RNNDescGuard() : status(miopenCreateRNNDescriptor(&desc)) {}

    ~RNNDescGuard()
    {
        if(desc != nullptr)
        {
            miopenDestroyRNNDescriptor(desc);
        }
    }

    operator miopenRNNDescriptor_t() { return desc; }

    miopenRNNDescriptor_t get() { return desc; }
    miopenStatus_t getStatus() const { return status; }

    RNNDescGuard(const RNNDescGuard&)            = delete;
    RNNDescGuard& operator=(const RNNDescGuard&) = delete;
    RNNDescGuard(RNNDescGuard&&)                 = delete;
    RNNDescGuard& operator=(RNNDescGuard&&)      = delete;

private:
    miopenRNNDescriptor_t desc = nullptr;
    miopenStatus_t status;
};

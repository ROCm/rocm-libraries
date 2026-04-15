// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <miopen/miopen.h>

// RAII wrapper for miopenDropoutDescriptor_t to prevent memory leaks
class DropoutDescGuard
{
public:
    DropoutDescGuard() : status(miopenCreateDropoutDescriptor(&desc)) {}

    ~DropoutDescGuard()
    {
        if(desc != nullptr)
        {
            miopenDestroyDropoutDescriptor(desc);
        }
    }

    operator miopenDropoutDescriptor_t() { return desc; }

    miopenDropoutDescriptor_t get() { return desc; }
    miopenStatus_t getStatus() const { return status; }

    DropoutDescGuard(const DropoutDescGuard&)            = delete;
    DropoutDescGuard& operator=(const DropoutDescGuard&) = delete;
    DropoutDescGuard(DropoutDescGuard&&)                 = delete;
    DropoutDescGuard& operator=(DropoutDescGuard&&)      = delete;

private:
    miopenDropoutDescriptor_t desc = nullptr;
    miopenStatus_t status;
};

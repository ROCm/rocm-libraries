// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <stdexcept>

// RAII wrapper for miopenTensorDescriptor_t to prevent memory leaks
class TensorDescGuard
{
public:
    TensorDescGuard() : status(miopenCreateTensorDescriptor(&desc)) {}

    ~TensorDescGuard()
    {
        if(desc != nullptr)
        {
            miopenDestroyTensorDescriptor(desc);
        }
    }

    operator miopenTensorDescriptor_t() { return desc; }

    miopenTensorDescriptor_t get() { return desc; }
    miopenStatus_t getStatus() const { return status; }

    TensorDescGuard(const TensorDescGuard&)            = delete;
    TensorDescGuard& operator=(const TensorDescGuard&) = delete;
    TensorDescGuard(TensorDescGuard&&)                 = delete;
    TensorDescGuard& operator=(TensorDescGuard&&)      = delete;

private:
    miopenTensorDescriptor_t desc = nullptr;
    miopenStatus_t status;
};

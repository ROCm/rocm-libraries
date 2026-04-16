// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <miopen/miopen.h>

// RAII wrapper for MIOpen descriptor types to prevent memory leaks.
// Template parameters:
//   DescType  - the descriptor handle type (e.g. miopenTensorDescriptor_t)
//   CreateFn  - function that allocates the descriptor
//   DestroyFn - function that frees the descriptor
template <typename DescType,
          miopenStatus_t (*CreateFn)(DescType*),
          miopenStatus_t (*DestroyFn)(DescType)>
class DescGuard
{
public:
    DescGuard() : status(CreateFn(&desc)) {}

    ~DescGuard()
    {
        if(desc != nullptr)
            DestroyFn(desc);
    }

    operator DescType() { return desc; }

    DescType get() { return desc; }
    miopenStatus_t getStatus() const { return status; }

    DescGuard(const DescGuard&)            = delete;
    DescGuard& operator=(const DescGuard&) = delete;
    DescGuard(DescGuard&&)                 = delete;
    DescGuard& operator=(DescGuard&&)      = delete;

private:
    DescType desc = nullptr;
    miopenStatus_t status;
};

using TensorDescGuard = DescGuard<miopenTensorDescriptor_t,
                                  miopenCreateTensorDescriptor,
                                  miopenDestroyTensorDescriptor>;

using ConvDescGuard = DescGuard<miopenConvolutionDescriptor_t,
                                miopenCreateConvolutionDescriptor,
                                miopenDestroyConvolutionDescriptor>;

using DropoutDescGuard = DescGuard<miopenDropoutDescriptor_t,
                                   miopenCreateDropoutDescriptor,
                                   miopenDestroyDropoutDescriptor>;

using RNNDescGuard =
    DescGuard<miopenRNNDescriptor_t, miopenCreateRNNDescriptor, miopenDestroyRNNDescriptor>;

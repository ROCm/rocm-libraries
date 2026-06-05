// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <miopen/miopen.h>

#include <vector>

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

// Frees the internal DropoutDescriptor allocated by miopen::RNNDescriptor's
// default and 8-arg constructors. The library has no destructor for this
// field, so any test that creates an RNN descriptor and then calls
// miopenSetRNNDescriptor[_V2] on it must invoke this helper:
//   1. Before each Set* call (frees the default-allocated internal one).
//   2. After the run, only when no user-supplied DropoutDescriptor was passed
//      via the _V2 path (in that path the internal pointer aliases the
//      user-owned descriptor — freeing it would double-free).
inline void DestroyInternalRnnDropoutDesc(miopenRNNDescriptor_t rnnDesc)
{
    miopenDropoutDescriptor_t dropDesc = nullptr;
    miopenGetRNNDescriptor_V2(
        rnnDesc, nullptr, nullptr, &dropDesc, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    if(dropDesc != nullptr)
        miopenDestroyDropoutDescriptor(dropDesc);
}

// Owns a vector of miopenTensorDescriptor_t handles and destroys them on scope exit,
// so an ASSERT_* failure mid-test does not leak descriptors.
class TensorDescVecGuard
{
public:
    TensorDescVecGuard() = default;
    ~TensorDescVecGuard()
    {
        for(auto* d : descs)
        {
            if(d != nullptr)
                miopenDestroyTensorDescriptor(d);
        }
    }

    TensorDescVecGuard(const TensorDescVecGuard&)            = delete;
    TensorDescVecGuard& operator=(const TensorDescVecGuard&) = delete;
    TensorDescVecGuard(TensorDescVecGuard&&)                 = delete;
    TensorDescVecGuard& operator=(TensorDescVecGuard&&)      = delete;

    miopenTensorDescriptor_t* data() { return descs.data(); }
    miopenTensorDescriptor_t& operator[](std::size_t i) { return descs[i]; }

    std::vector<miopenTensorDescriptor_t> descs;
};

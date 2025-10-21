// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// NOLINTBEGIN(portability-template-virtual-member-function)

#include <hipdnn_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <type_traits>

namespace hipdnn_sdk
{
namespace test_utilities
{

using namespace hipdnn_sdk::utilities;

template <class T>
class IReferenceValidation
{
public:
    virtual ~IReferenceValidation() = default;

    virtual bool allClose(const ITensor& reference, const ITensor& implementation) = 0;
    virtual bool allClose(IMigratableMemory<T>& reference, IMigratableMemory<T>& implementation)
        = 0;
};

} // namespace test_utilities
} // namespace hipdnn_sdk

// NOLINTEND(portability-template-virtual-member-function)

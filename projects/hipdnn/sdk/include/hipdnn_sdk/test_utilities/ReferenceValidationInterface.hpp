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

class IReferenceValidation
{
public:
    virtual ~IReferenceValidation() = default;

    virtual bool allClose(ITensor& reference, ITensor& implementation) const = 0;
};

} // namespace test_utilities
} // namespace hipdnn_sdk

// NOLINTEND(portability-template-virtual-member-function)

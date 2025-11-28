// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// NOLINTBEGIN(portability-template-virtual-member-function)

#include <hipdnn_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <type_traits>

namespace hipdnn_sdk::test_utilities
{

class IReferenceValidation
{
public:
    virtual ~IReferenceValidation() = default;

    virtual bool allClose(utilities::ITensor& reference, utilities::ITensor& implementation) const
        = 0;
};

} // namespace hipdnn_sdk::test_utilities

// NOLINTEND(portability-template-virtual-member-function)

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// NOLINTBEGIN(portability-template-virtual-member-function)

#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <type_traits>

namespace hipdnn_test_sdk::utilities
{

class IReferenceValidation
{
public:
    virtual ~IReferenceValidation() = default;

    virtual bool allClose(hipdnn_data_sdk::utilities::ITensor& reference,
                          hipdnn_data_sdk::utilities::ITensor& implementation) const
        = 0;
};

} // namespace hipdnn_test_sdk::utilities

// NOLINTEND(portability-template-virtual-member-function)

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/DataTypes/DistinctType.hpp>

namespace rocRoller
{
    namespace Operations
    {
        struct OperationTag final : public DistinctType<int32_t, OperationTag>
        {
            explicit OperationTag(int value)
                : DistinctType<int32_t, OperationTag>(value)
            {
            }

            OperationTag()
                : OperationTag(-1)
            {
            }

            // Prefix increment operator
            OperationTag& operator++();

            // Postfix increment operator
            OperationTag operator++(int);

            auto operator<=>(OperationTag const&) const = default;

            bool uninitialized() const;
        };
    }
}

#include <rocRoller/Operations/OperationTag_impl.hpp>

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace hipdnn_sdk::test_utilities
{

struct BaseSigKey
{
    constexpr BaseSigKey(hipdnn_sdk::data_objects::NodeAttributes nodeType)
        : nodeType(nodeType)
    {
    }

    virtual ~BaseSigKey() = default;

    virtual size_t hash_self() const = 0;
    virtual bool equal(const BaseSigKey&) const = 0;

    const hipdnn_sdk::data_objects::NodeAttributes nodeType;
};

struct BaseSigKeyHash
{
    std::size_t operator()(const std::unique_ptr<BaseSigKey>& k) const
    {
        return k->hash_self();
    }
};

struct BaseSigKeyEqual
{
    bool operator()(const std::unique_ptr<BaseSigKey>& lhs,
                    const std::unique_ptr<BaseSigKey>& rhs) const
    {
        return lhs->equal(*rhs);
    }
};

}

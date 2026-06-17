// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hipdnn_frontend::detail
{

inline Error
    validatePlanOnlyOverrideArguments(const std::vector<int64_t>& overrideUids,
                                      const std::vector<std::vector<int64_t>>& overrideShapes,
                                      const std::vector<std::vector<int64_t>>& overrideStrides)
{
    if(overrideUids.size() != overrideShapes.size()
       || overrideUids.size() != overrideStrides.size())
    {
        return {ErrorCode::INVALID_VALUE,
                "Override arrays have inconsistent sizes: "
                "override_uids.size()="
                    + std::to_string(overrideUids.size())
                    + ", override_shapes.size()=" + std::to_string(overrideShapes.size())
                    + ", override_strides.size()=" + std::to_string(overrideStrides.size()) + "."};
    }

    for(size_t i = 0; i < overrideUids.size(); ++i)
    {
        if(overrideShapes[i].size() != overrideStrides[i].size())
        {
            return {ErrorCode::INVALID_VALUE,
                    "Override shape/stride rank mismatch at index " + std::to_string(i)
                        + ": override shape rank=" + std::to_string(overrideShapes[i].size())
                        + ", override stride rank=" + std::to_string(overrideStrides[i].size())
                        + "."};
        }
        if(overrideShapes[i].empty())
        {
            return {ErrorCode::INVALID_VALUE,
                    "Override shape/stride rank at index " + std::to_string(i)
                        + " must be non-zero for compiled-plan-only execution."};
        }

        for(size_t d = 0; d < overrideShapes[i].size(); ++d)
        {
            if(overrideShapes[i][d] <= 0)
            {
                return {ErrorCode::INVALID_VALUE,
                        "Override shape value at index " + std::to_string(i) + " axis "
                            + std::to_string(d)
                            + " must be positive for compiled-plan-only execution: "
                            + std::to_string(overrideShapes[i][d]) + "."};
            }
            if(overrideStrides[i][d] <= 0)
            {
                return {ErrorCode::INVALID_VALUE,
                        "Override stride value at index " + std::to_string(i) + " axis "
                            + std::to_string(d)
                            + " must be positive for compiled-plan-only execution: "
                            + std::to_string(overrideStrides[i][d]) + "."};
            }
        }
    }

    std::unordered_set<int64_t> seen;
    seen.reserve(overrideUids.size());
    for(const auto uid : overrideUids)
    {
        if(!seen.insert(uid).second)
        {
            return {ErrorCode::INVALID_VALUE,
                    "Duplicate UID " + std::to_string(uid)
                        + " in override_uids for compiled-plan-only execution."};
        }
    }

    return {};
}

inline Error validateGraphBackedOverrideArguments(
    const std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorsByUid,
    const std::vector<int64_t>& overrideUids,
    const std::vector<std::vector<int64_t>>& overrideShapes,
    const std::vector<std::vector<int64_t>>& overrideStrides)
{
    // Rule 1: array sizes must match.
    if(overrideUids.size() != overrideShapes.size()
       || overrideUids.size() != overrideStrides.size())
    {
        return {ErrorCode::INVALID_VALUE,
                "Override arrays have inconsistent sizes: "
                "override_uids.size()="
                    + std::to_string(overrideUids.size()) + ", override_shapes.size()="
                    + std::to_string(overrideShapes.size()) + ", override_strides.size()="
                    + std::to_string(overrideStrides.size()) + " (RFC 0008 §4.2.1 rule 1)."};
    }

    // Rule 2: override UIDs must be unique.
    std::unordered_set<int64_t> seen;
    seen.reserve(overrideUids.size());
    for(const auto uid : overrideUids)
    {
        if(!seen.insert(uid).second)
        {
            return {ErrorCode::INVALID_VALUE,
                    "Duplicate UID " + std::to_string(uid)
                        + " in override_uids (RFC 0008 §4.2.1 rule 2)."};
        }
    }

    for(size_t i = 0; i < overrideUids.size(); ++i)
    {
        const auto uid = overrideUids[i];
        const auto& shape = overrideShapes[i];
        const auto& stride = overrideStrides[i];

        // Rule 3: UID must identify a graph tensor.
        const auto it = tensorsByUid.find(uid);
        if(it == tensorsByUid.end())
        {
            return {ErrorCode::INVALID_VALUE,
                    "Override UID " + std::to_string(uid)
                        + " does not identify any graph tensor "
                          "(RFC 0008 §4.2.1 rule 3)."};
        }
        const auto& declaredDims = it->second->get_dim();
        const auto& declaredStrides = it->second->get_stride();

        // Rule 4: rank must match the declared tensor.
        if(shape.size() != declaredDims.size() || stride.size() != declaredDims.size())
        {
            return {ErrorCode::INVALID_VALUE,
                    "Override rank mismatch for UID " + std::to_string(uid) + ": declared rank="
                        + std::to_string(declaredDims.size()) + ", override shape rank="
                        + std::to_string(shape.size()) + ", override stride rank="
                        + std::to_string(stride.size()) + " (RFC 0008 §4.2.1 rule 4)."};
        }

        // Rule 5/6: shape and stride values must be positive.
        for(size_t d = 0; d < shape.size(); ++d)
        {
            if(shape[d] <= 0)
            {
                return {ErrorCode::INVALID_VALUE,
                        "Override shape value for UID " + std::to_string(uid) + " axis "
                            + std::to_string(d) + " must be positive: " + std::to_string(shape[d])
                            + " (RFC 0008 §4.2.1 rule 5)."};
            }
            if(stride[d] <= 0)
            {
                return {ErrorCode::INVALID_VALUE,
                        "Override stride value for UID " + std::to_string(uid) + " axis "
                            + std::to_string(d) + " must be positive: " + std::to_string(stride[d])
                            + " (RFC 0008 §4.2.1 rule 6)."};
            }
        }

        // Rule 7: shape values must not exceed declared max.
        for(size_t d = 0; d < shape.size(); ++d)
        {
            if(declaredDims[d] >= 0 && shape[d] > declaredDims[d])
            {
                return {ErrorCode::INVALID_VALUE,
                        "Override shape for UID " + std::to_string(uid) + " axis "
                            + std::to_string(d) + " exceeds declared max: override="
                            + std::to_string(shape[d]) + ", declared="
                            + std::to_string(declaredDims[d]) + " (RFC 0008 §4.2.1 rule 7)."};
            }
        }

        // Rule 8: stride axis ordering must match the declared tensor.
        if(!declaredStrides.empty())
        {
            if(declaredStrides.size() != stride.size())
            {
                return {ErrorCode::INVALID_VALUE,
                        "Override stride rank for UID " + std::to_string(uid)
                            + " does not match declared stride rank: declared="
                            + std::to_string(declaredStrides.size()) + ", override="
                            + std::to_string(stride.size()) + " (RFC 0008 §4.2.1 rule 8)."};
            }

            const auto declaredOrder
                = hipdnn_data_sdk::utilities::extractStrideOrder(declaredStrides);
            const auto overrideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(stride);
            if(declaredOrder != overrideOrder)
            {
                return {ErrorCode::INVALID_VALUE,
                        "Override stride permutation for UID " + std::to_string(uid)
                            + " does not match declared stride order "
                              "(RFC 0008 §4.2.1 rule 8)."};
            }
        }
    }

    return {};
}

} // namespace hipdnn_frontend::detail

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/PagedCacheStoreAttributes.hpp>
#include <hipdnn_frontend/detail/PagedCacheStorePacker.hpp>
#include <hipdnn_frontend/detail/PagedCacheStoreUnpacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

#include <string>

namespace hipdnn_frontend::graph
{
/// Paged KV-cache store. Writes x into the SDPA paged container [num_blocks, H, block_size, D]
/// at the slots given by slot_mapping (negative slots are skipped). Two layouts are accepted:
///  - token-packed: x [T, H, D] with slot_mapping [T];
///  - SDPA-native:  x [B, H, S, D] (any strides) with slot_mapping [B, S].
/// slot_mapping is INT32 or INT64; the optional scale is a FLOAT scalar (one element).
/// The container is a user-created, non-virtual tensor: the node updates it in place, so every
/// element that no token addresses keeps its contents.
class PagedCacheStoreNode : public BaseNode<PagedCacheStoreNode, NodeType::PAGED_CACHE_STORE>
{
public:
    PagedCacheStoreAttributes attributes;

    PagedCacheStoreNode(PagedCacheStoreAttributes&& attrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(attrs))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        PagedCacheStoreAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackPagedCacheStoreOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    Error pre_validate_node() const override
    {
        const auto x = attributes.get_x();
        const auto slotMapping = attributes.get_slot_mapping();
        const auto container = attributes.get_container();
        const auto scale = attributes.get_scale();

        HIPDNN_RETURN_IF_FALSE(x,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "PagedCacheStoreNode missing x (input) for pre-validation");
        HIPDNN_RETURN_IF_FALSE(
            slotMapping,
            ErrorCode::ATTRIBUTE_NOT_SET,
            "PagedCacheStoreNode missing slot_mapping (input) for pre-validation");
        HIPDNN_RETURN_IF_FALSE(container,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "PagedCacheStoreNode missing container (output) for pre-validation");

        const auto& xDims = x->get_dim();
        const auto& slotDims = slotMapping->get_dim();
        const auto& cDims = container->get_dim();

        HIPDNN_RETURN_IF_TRUE(xDims.size() != 3 && xDims.size() != 4,
                              ErrorCode::INVALID_VALUE,
                              "PagedCacheStoreNode: x must be [T, H, D] or [B, H, S, D], got rank "
                                  + std::to_string(xDims.size()));
        const bool tokenPacked = xDims.size() == 3;
        if(tokenPacked)
        {
            HIPDNN_RETURN_IF_TRUE(
                slotDims.size() != 1 || slotDims[0] != xDims[0],
                ErrorCode::INVALID_VALUE,
                "PagedCacheStoreNode: slot_mapping must be [T] with T = x dim 0 = "
                    + std::to_string(xDims[0]) + " when x is [T, H, D]");
        }
        else
        {
            HIPDNN_RETURN_IF_TRUE(
                slotDims.size() != 2 || slotDims[0] != xDims[0] || slotDims[1] != xDims[2],
                ErrorCode::INVALID_VALUE,
                "PagedCacheStoreNode: slot_mapping must be [B, S] = [" + std::to_string(xDims[0])
                    + ", " + std::to_string(xDims[2]) + "] when x is [B, H, S, D]");
        }
        HIPDNN_RETURN_IF_TRUE(
            slotMapping->get_data_type() != DataType::INT32
                && slotMapping->get_data_type() != DataType::INT64,
            ErrorCode::INVALID_VALUE,
            "PagedCacheStoreNode: slot_mapping must have INT32 or INT64 data type");

        HIPDNN_RETURN_IF_NE(
            cDims.size(),
            4u,
            ErrorCode::INVALID_VALUE,
            "PagedCacheStoreNode: container must be [num_blocks, H, block_size, D]");
        HIPDNN_RETURN_IF_TRUE(cDims[0] <= 0 || cDims[2] <= 0,
                              ErrorCode::INVALID_VALUE,
                              "PagedCacheStoreNode: container num_blocks and block_size must be "
                              "positive");
        HIPDNN_RETURN_IF_NE(cDims[1],
                            xDims[1],
                            ErrorCode::INVALID_VALUE,
                            "PagedCacheStoreNode: container dim 1 must equal x head count H = "
                                + std::to_string(xDims[1]));
        const auto headDim = xDims.back();
        HIPDNN_RETURN_IF_NE(cDims[3],
                            headDim,
                            ErrorCode::INVALID_VALUE,
                            "PagedCacheStoreNode: container dim 3 must equal x head size D = "
                                + std::to_string(headDim));
        HIPDNN_RETURN_IF_TRUE(container->get_is_virtual(),
                              ErrorCode::INVALID_VALUE,
                              "PagedCacheStoreNode: container must be a non-virtual tensor (the "
                              "persistent cache buffer)");

        if(scale)
        {
            HIPDNN_RETURN_IF_NE(scale->get_volume(),
                                1,
                                ErrorCode::INVALID_VALUE,
                                "PagedCacheStoreNode: scale must be a scalar (one element)");
            HIPDNN_RETURN_IF_TRUE(scale->get_data_type() != DataType::FLOAT,
                                  ErrorCode::INVALID_VALUE,
                                  "PagedCacheStoreNode: scale must have FLOAT data type");
        }

        return {};
    }

    Error infer_properties_node() override
    {
        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "PagedCacheStoreNode missing x for setting properties");
        HIPDNN_RETURN_IF_FALSE(attributes.get_slot_mapping(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "PagedCacheStoreNode missing slot_mapping for setting properties");
        HIPDNN_RETURN_IF_FALSE(attributes.get_container(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "PagedCacheStoreNode missing container for setting properties");

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        // The container is the caller's persistent cache: its dims are never inferred. Only a
        // missing stride is filled in, with the packed layout.
        auto container = attributes.get_container();
        HIPDNN_RETURN_IF_TRUE(container->get_dim().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "PagedCacheStoreNode: container dims must be set by the caller");
        if(container->get_stride().empty())
        {
            container->set_stride(
                hipdnn_data_sdk::utilities::generateStrides(container->get_dim()));
        }

        return {};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createPagedCacheStoreOperation(attributes, tensorDescs, operations);
    }
};
} // namespace hipdnn_frontend::graph

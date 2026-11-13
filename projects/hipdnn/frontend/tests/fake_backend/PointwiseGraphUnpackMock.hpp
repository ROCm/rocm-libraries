// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>

#include "MockHipdnnBackend.hpp"

#include <cstdint>

namespace hipdnn_fake_backend
{

// Installs backendGetAttribute ON_CALL defaults that reconstruct a single
// unary-pointwise operation graph during deserialize, letting the
// post-reconstruction contents-query + plan-attach logic run. Keying on the
// attribute name alone works because tensor and operation/graph attributes
// occupy disjoint name sets.
inline void
    setupSuccessfulPointwiseGraphUnpack(::testing::NiceMock<Mock_hipdnn_backend>& mockBackend)
{
    using ::testing::_;

    auto fakeOpDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA100);
    auto fakeTensorDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA200);

    ON_CALL(mockBackend, backendGetAttribute(_, _, _, _, _, _))
        .WillByDefault([fakeOpDesc, fakeTensorDesc](hipdnnBackendDescriptor_t,
                                                    hipdnnBackendAttributeName_t attrName,
                                                    hipdnnBackendAttributeType_t,
                                                    int64_t requestedCount,
                                                    int64_t* elementCount,
                                                    void* arrayOfElements) {
            const bool isCountQuery = (arrayOfElements == nullptr);
            auto setCount = [&](int64_t value) {
                if(elementCount != nullptr)
                {
                    *elementCount = value;
                }
            };

            switch(attrName)
            {
            // Operation-graph: exactly one operation descriptor.
            case HIPDNN_ATTR_OPERATIONGRAPH_OPS:
                if(isCountQuery)
                {
                    setCount(1);
                }
                else
                {
                    setCount(1);
                    *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = fakeOpDesc;
                }
                return HIPDNN_STATUS_SUCCESS;

            // Operation type: unary pointwise (single IN_0 input).
            case HIPDNN_ATTR_OPERATION_TYPE_EXT:
                setCount(1);
                *static_cast<hipdnnOperationType_ext_t*>(arrayOfElements)
                    = HIPDNN_OPERATION_TYPE_POINTWISE_EXT;
                return HIPDNN_STATUS_SUCCESS;

            // Mandatory pointwise tensors return a tensor descriptor.
            case HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT:
            case HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT:
                setCount(1);
                if(!isCountQuery)
                {
                    *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = fakeTensorDesc;
                }
                return HIPDNN_STATUS_SUCCESS;

            // Optional pointwise tensors are absent.
            case HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT:
            case HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT:
                setCount(0);
                return HIPDNN_STATUS_SUCCESS;

            // Pointwise mode (mandatory scalar).
            case HIPDNN_ATTR_POINTWISE_MODE:
                setCount(1);
                *static_cast<hipdnnPointwiseMode_t*>(arrayOfElements) = HIPDNN_POINTWISE_RELU_FWD;
                return HIPDNN_STATUS_SUCCESS;

            // Tensor scalar attributes.
            case HIPDNN_ATTR_TENSOR_UNIQUE_ID:
                setCount(1);
                *static_cast<int64_t*>(arrayOfElements) = 1;
                return HIPDNN_STATUS_SUCCESS;
            case HIPDNN_ATTR_TENSOR_DATA_TYPE:
                setCount(1);
                *static_cast<hipdnnDataType_t*>(arrayOfElements) = HIPDNN_DATA_FLOAT;
                return HIPDNN_STATUS_SUCCESS;
            case HIPDNN_ATTR_TENSOR_DIMENSIONS:
            case HIPDNN_ATTR_TENSOR_STRIDES:
                if(isCountQuery)
                {
                    setCount(4);
                }
                else
                {
                    setCount(4);
                    auto* dims = static_cast<int64_t*>(arrayOfElements);
                    for(int64_t i = 0; i < requestedCount && i < 4; ++i)
                    {
                        dims[i] = 1;
                    }
                }
                return HIPDNN_STATUS_SUCCESS;
            case HIPDNN_ATTR_TENSOR_IS_VIRTUAL:
            case HIPDNN_ATTR_TENSOR_IS_BY_VALUE:
                setCount(1);
                *static_cast<bool*>(arrayOfElements) = false;
                return HIPDNN_STATUS_SUCCESS;

            // Optional / unset attributes: report zero elements so the
            // unpacker treats them as absent.
            default:
                setCount(0);
                return HIPDNN_STATUS_SUCCESS;
            }
        });
}

} // namespace hipdnn_fake_backend

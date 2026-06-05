// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/MatmulOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/matmul_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/MatmulConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;

namespace
{
constexpr int64_t K_TENSOR_A_UID = K_MATMUL_TENSOR_A_UID;
constexpr int64_t K_TENSOR_B_UID = K_MATMUL_TENSOR_B_UID;
constexpr int64_t K_TENSOR_C_UID = K_MATMUL_TENSOR_C_UID;
} // namespace

class TestMatmulOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<MatmulOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<MatmulOperationDescriptor>();
    }

    void setAllAttributesExcept(std::initializer_list<hipdnnBackendAttributeName_t> skip = {}) const
    {
        auto desc = getDescriptor();
        auto setIf = [&](hipdnnBackendAttributeName_t attr, auto& tensor) {
            if(std::find(skip.begin(), skip.end(), attr) == skip.end())
            {
                desc->setAttribute(attr, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tensor);
            }
        };
        setIf(HIPDNN_ATTR_OPERATION_MATMUL_ADESC, _aDesc);
        setIf(HIPDNN_ATTR_OPERATION_MATMUL_BDESC, _bDesc);
        setIf(HIPDNN_ATTR_OPERATION_MATMUL_CDESC, _cDesc);
        if(std::find(skip.begin(), skip.end(), HIPDNN_ATTR_MATMUL_COMP_TYPE) == skip.end())
        {
            auto computeType = HIPDNN_DATA_FLOAT;
            desc->setAttribute(
                HIPDNN_ATTR_MATMUL_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
        }
    }

    void makeFinalized() const
    {
        setAllAttributesExcept();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _aDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _bDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _cDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<MatmulOperationDescriptor>();
        _aDesc = createFinalizedTensor(K_TENSOR_A_UID,
                                       hipdnn_tests::toVec(K_MATMUL_TENSOR_A_DIMS),
                                       hipdnn_tests::toVec(K_MATMUL_TENSOR_A_STRIDES));
        _bDesc = createFinalizedTensor(K_TENSOR_B_UID,
                                       hipdnn_tests::toVec(K_MATMUL_TENSOR_B_DIMS),
                                       hipdnn_tests::toVec(K_MATMUL_TENSOR_B_STRIDES));
        _cDesc = createFinalizedTensor(K_TENSOR_C_UID,
                                       hipdnn_tests::toVec(K_MATMUL_TENSOR_C_DIMS),
                                       hipdnn_tests::toVec(K_MATMUL_TENSOR_C_STRIDES));
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _aDesc.reset();
        _bDesc.reset();
        _cDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestMatmulOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);
}

TEST_F(TestMatmulOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setAllAttributesExcept();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestMatmulOperationDescriptor, FinalizeFailsWithoutATensor)
{
    setAllAttributesExcept({HIPDNN_ATTR_OPERATION_MATMUL_ADESC});
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMatmulOperationDescriptor, FinalizeFailsWithoutBTensor)
{
    setAllAttributesExcept({HIPDNN_ATTR_OPERATION_MATMUL_BDESC});
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMatmulOperationDescriptor, FinalizeFailsWithoutCTensor)
{
    setAllAttributesExcept({HIPDNN_ATTR_OPERATION_MATMUL_CDESC});
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMatmulOperationDescriptor, FinalizeFailsWithoutComputeType)
{
    setAllAttributesExcept({HIPDNN_ATTR_MATMUL_COMP_TYPE});
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestMatmulOperationDescriptor, SetTensorDescriptorA)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_MATMUL_ADESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_aDesc));

    ASSERT_EQ(desc->getData().a_tensor_uid, K_TENSOR_A_UID);
    ASSERT_NE(desc->getADesc(), nullptr);
}

TEST_F(TestMatmulOperationDescriptor, SetTensorDescriptorB)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_MATMUL_BDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_bDesc));

    ASSERT_EQ(desc->getData().b_tensor_uid, K_TENSOR_B_UID);
    ASSERT_NE(desc->getBDesc(), nullptr);
}

TEST_F(TestMatmulOperationDescriptor, SetTensorDescriptorC)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_MATMUL_CDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_cDesc));

    ASSERT_EQ(desc->getData().c_tensor_uid, K_TENSOR_C_UID);
    ASSERT_NE(desc->getCDesc(), nullptr);
}

TEST_F(TestMatmulOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_MATMUL_ADESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestMatmulOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MATMUL_ADESC, HIPDNN_TYPE_INT64, 1, &_aDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMatmulOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_MATMUL_ADESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_aDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMatmulOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_MATMUL_ADESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Compute Data Type
// =============================================================================

TEST_F(TestMatmulOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_MATMUL_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestMatmulOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_MATMUL_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestMatmulOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_MATMUL_ADESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_aDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestMatmulOperationDescriptor, SetAttributeUnsupported)
{
    auto desc = getDescriptor();
    int64_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// GetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestMatmulOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* rawA = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MATMUL_ADESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&rawA)));
    const std::unique_ptr<HipdnnBackendDescriptor> retrievedA(rawA);

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedA, nullptr);
}

// =============================================================================
// GetAttribute Tests - Compute Data Type
// =============================================================================

TEST_F(TestMatmulOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setAllAttributesExcept();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_MATMUL_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_MATMUL_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &elementCount, &retrieved));

    ASSERT_EQ(retrieved, HIPDNN_DATA_HALF);
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestMatmulOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setAllAttributesExcept();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(
            HIPDNN_ATTR_OPERATION_MATMUL_ADESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &dummy),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestMatmulOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_MATMUL_ADESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestMatmulOperationDescriptor, GetAttributeUnsupported)
{
    makeFinalized();
    auto desc = getDescriptor();
    int64_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, nullptr, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// GetAttribute Query Mode Tests
// =============================================================================

TEST_F(TestMatmulOperationDescriptor, GetAttributeTensorAQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MATMUL_ADESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestMatmulOperationDescriptor, GetAttributeTensorBQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MATMUL_BDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestMatmulOperationDescriptor, GetAttributeTensorCQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MATMUL_CDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestMatmulOperationDescriptor, GetAttributeComputeTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_MATMUL_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestMatmulOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_MATMUL_ADESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestMatmulOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_NE(desc->getADesc(), nullptr);
    ASSERT_NE(desc->getBDesc(), nullptr);
    ASSERT_NE(desc->getCDesc(), nullptr);

    ASSERT_EQ(desc->getADesc()->getData().uid, K_TENSOR_A_UID);
    ASSERT_EQ(desc->getBDesc()->getData().uid, K_TENSOR_B_UID);
    ASSERT_EQ(desc->getCDesc()->getData().uid, K_TENSOR_C_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestMatmulOperationDescriptor, ToStringContainsExpectedInfo)
{
    setAllAttributesExcept();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("MatmulOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("a_uid=" + std::to_string(K_TENSOR_A_UID)), std::string::npos);
    ASSERT_NE(str.find("b_uid=" + std::to_string(K_TENSOR_B_UID)), std::string::npos);
    ASSERT_NE(str.find("c_uid=" + std::to_string(K_TENSOR_C_UID)), std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestMatmulOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, K_TENSOR_A_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_TENSOR_B_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_TENSOR_C_UID);
}

TEST_F(TestMatmulOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setAllAttributesExcept();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_MATMUL_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::MatmulAttributes);

    auto* matmulAttrs = node->attributes.AsMatmulAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_EQ(matmulAttrs->a_tensor_uid, K_TENSOR_A_UID);
    ASSERT_EQ(matmulAttrs->b_tensor_uid, K_TENSOR_B_UID);
    ASSERT_EQ(matmulAttrs->c_tensor_uid, K_TENSOR_C_UID);
}

TEST_F(TestMatmulOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setAllAttributesExcept();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_MATMUL_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestMatmulOperationDescriptor, GetTensorDescriptorsOrderIsABC)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    EXPECT_EQ(tensors[0], desc->getADesc());
    EXPECT_EQ(tensors[1], desc->getBDesc());
    EXPECT_EQ(tensors[2], desc->getCDesc());
}

TEST_F(TestMatmulOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, K_TENSOR_A_UID);
}

TEST_F(TestMatmulOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    auto graphOp = _aDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

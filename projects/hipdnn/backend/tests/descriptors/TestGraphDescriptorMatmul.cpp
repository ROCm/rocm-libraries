// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/MatmulOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/matmul_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/MatmulConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <memory>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

namespace
{

inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedMatmulOp(HipdnnBackendDescriptor* aDesc,
                            HipdnnBackendDescriptor* bDesc,
                            HipdnnBackendDescriptor* cDesc,
                            hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
{
    auto wrapper = createDescriptor<MatmulOperationDescriptor>();
    auto desc = wrapper->asDescriptor<MatmulOperationDescriptor>();

    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_MATMUL_A_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &aDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_MATMUL_B_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &bDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_MATMUL_C_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cDesc);
    desc->setAttribute(HIPDNN_ATTR_MATMUL_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    desc->finalize();
    return wrapper;
}

} // namespace

class TestGraphDescriptorMatmul : public ::testing::Test
{
public:
    static const TensorAttributesT* findTensorByUid(const GraphT& graphT, int64_t uid)
    {
        for(const auto& tensor : graphT.tensors)
        {
            if(tensor->uid == uid)
            {
                return tensor.get();
            }
        }
        return nullptr;
    }

    static void verifyTensor(const TensorAttributesT* tensor,
                             int64_t expectedUid,
                             const std::vector<int64_t>& expectedDims,
                             const std::vector<int64_t>& expectedStrides,
                             DataType expectedDataType,
                             bool expectedVirtual = false)
    {
        ASSERT_NE(tensor, nullptr) << "Tensor with UID " << expectedUid << " not found";
        EXPECT_EQ(tensor->uid, expectedUid);
        EXPECT_EQ(tensor->dims, expectedDims);
        EXPECT_EQ(tensor->strides, expectedStrides);
        EXPECT_EQ(tensor->data_type, expectedDataType);
        EXPECT_EQ(tensor->virtual_, expectedVirtual);
    }

    std::shared_ptr<GraphDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<GraphDescriptor>();
    }

    void setHandle() const
    {
        auto desc = getDescriptor();
        hipdnnHandle_t handle = &_mockHandle;
        desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle);
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    mutable MockHandle _mockHandle;

    void SetUp() override
    {
        _wrapper = createDescriptor<GraphDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
    }
};

TEST_F(TestGraphDescriptorMatmul, BuildFromSingleMatmulOperation)
{
    auto aDesc = createFinalizedTensor(
        K_MATMUL_TENSOR_A_UID, toVec(K_MATMUL_TENSOR_A_DIMS), toVec(K_MATMUL_TENSOR_A_STRIDES));
    auto bDesc = createFinalizedTensor(
        K_MATMUL_TENSOR_B_UID, toVec(K_MATMUL_TENSOR_B_DIMS), toVec(K_MATMUL_TENSOR_B_STRIDES));
    auto cDesc = createFinalizedTensor(
        K_MATMUL_TENSOR_C_UID, toVec(K_MATMUL_TENSOR_C_DIMS), toVec(K_MATMUL_TENSOR_C_STRIDES));

    auto matmulOp = createFinalizedMatmulOp(aDesc.get(), bDesc.get(), cDesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {matmulOp.get()};
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data()));
    ASSERT_NO_THROW(desc->finalize());

    auto serialized = desc->getSerializedGraph();
    ASSERT_NE(serialized.ptr, nullptr);
    ASSERT_GT(serialized.size, 0UL);

    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(serialized.ptr), serialized.size);
    ASSERT_TRUE(verifier.VerifyBuffer<Graph>());

    auto graph = GetGraph(serialized.ptr);
    auto graphT = graph->UnPack();

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 3);

    verifyTensor(findTensorByUid(*graphT, K_MATMUL_TENSOR_A_UID),
                 K_MATMUL_TENSOR_A_UID,
                 toVec(K_MATMUL_TENSOR_A_DIMS),
                 toVec(K_MATMUL_TENSOR_A_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_MATMUL_TENSOR_B_UID),
                 K_MATMUL_TENSOR_B_UID,
                 toVec(K_MATMUL_TENSOR_B_DIMS),
                 toVec(K_MATMUL_TENSOR_B_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_MATMUL_TENSOR_C_UID),
                 K_MATMUL_TENSOR_C_UID,
                 toVec(K_MATMUL_TENSOR_C_DIMS),
                 toVec(K_MATMUL_TENSOR_C_STRIDES),
                 DataType::FLOAT);

    const auto& node = *graphT->nodes[0];
    EXPECT_EQ(node.compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node.attributes.type, NodeAttributes::MatmulAttributes);

    auto* matmulAttrs = node.attributes.AsMatmulAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    EXPECT_EQ(matmulAttrs->a_tensor_uid, K_MATMUL_TENSOR_A_UID);
    EXPECT_EQ(matmulAttrs->b_tensor_uid, K_MATMUL_TENSOR_B_UID);
    EXPECT_EQ(matmulAttrs->c_tensor_uid, K_MATMUL_TENSOR_C_UID);
}

TEST_F(TestGraphDescriptorMatmul, ComputeDataTypePreserved)
{
    auto aDesc = createFinalizedTensor(K_MATMUL_TENSOR_A_UID,
                                       toVec(K_MATMUL_TENSOR_A_DIMS),
                                       toVec(K_MATMUL_TENSOR_A_STRIDES),
                                       HIPDNN_DATA_HALF);
    auto bDesc = createFinalizedTensor(K_MATMUL_TENSOR_B_UID,
                                       toVec(K_MATMUL_TENSOR_B_DIMS),
                                       toVec(K_MATMUL_TENSOR_B_STRIDES),
                                       HIPDNN_DATA_HALF);
    auto cDesc = createFinalizedTensor(K_MATMUL_TENSOR_C_UID,
                                       toVec(K_MATMUL_TENSOR_C_DIMS),
                                       toVec(K_MATMUL_TENSOR_C_STRIDES),
                                       HIPDNN_DATA_HALF);

    auto matmulOp
        = createFinalizedMatmulOp(aDesc.get(), bDesc.get(), cDesc.get(), HIPDNN_DATA_HALF);

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {matmulOp.get()};
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data()));
    ASSERT_NO_THROW(desc->finalize());

    auto serialized = desc->getSerializedGraph();
    auto graph = GetGraph(serialized.ptr);
    auto graphT = graph->UnPack();

    ASSERT_EQ(graphT->nodes.size(), 1);

    const auto& node = *graphT->nodes[0];
    EXPECT_EQ(node.compute_data_type, DataType::HALF);
    ASSERT_EQ(node.attributes.type, NodeAttributes::MatmulAttributes);

    auto* matmulAttrs = node.attributes.AsMatmulAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    EXPECT_EQ(matmulAttrs->a_tensor_uid, K_MATMUL_TENSOR_A_UID);
    EXPECT_EQ(matmulAttrs->b_tensor_uid, K_MATMUL_TENSOR_B_UID);
    EXPECT_EQ(matmulAttrs->c_tensor_uid, K_MATMUL_TENSOR_C_UID);
}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>
#include <hipdnn_frontend/detail/KnobPacker.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "fake_backend/BackendTestMatchers.hpp"
#include "fake_backend/MockHipdnnBackend.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_frontend::detail;
using hipdnn_tests::toVec;
using namespace hipdnn_frontend::test;
using namespace ::testing;

namespace
{

constexpr int64_t K_DEFAULT_TENSOR_UID = 42;
constexpr int64_t K_MISSING_TENSOR_UID = 999;

constexpr std::array<int64_t, 4> K_DEFAULT_TENSOR_DIMS = {1, 3, 4, 4};
constexpr std::array<int64_t, 4> K_DEFAULT_TENSOR_STRIDES = {48, 16, 4, 1};

} // namespace

class TestDescriptorHelpers : public ::testing::Test
{
protected:
    std::shared_ptr<Mock_hipdnn_backend> _mockBackend;

    void SetUp() override
    {
        _mockBackend = std::make_shared<Mock_hipdnn_backend>();
        IHipdnnBackend::setInstance(_mockBackend);
    }
    void TearDown() override
    {
        IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }

    void expectCreateAndDestroyDescriptor()
    {
        EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _))
            .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
        EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_))
            .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    }

    void expectAllBackendCallsSucceed()
    {
        expectCreateAndDestroyDescriptor();
        EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
            .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
        EXPECT_CALL(*_mockBackend, backendFinalize(_))
            .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    }

    // Sets up EXPECT_CALL expectations for a single tensor via createOrFindTensorDesc.
    // The 6 setAttribute calls are: uid, name, data_type, dims, strides, is_virtual.
    void expectTensorSetAttributes(int64_t uid,
                                   const std::string& name,
                                   const std::vector<int64_t>& dims,
                                   const std::vector<int64_t>& strides)
    {
        EXPECT_CALL(*_mockBackend,
                    backendSetAttribute(_,
                                        HIPDNN_ATTR_TENSOR_UNIQUE_ID,
                                        HIPDNN_TYPE_INT64,
                                        1,
                                        pointsToScalar<int64_t>(uid)))
            .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
        EXPECT_CALL(*_mockBackend,
                    backendSetAttribute(_,
                                        HIPDNN_ATTR_TENSOR_NAME_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(name.size()),
                                        pointsToString(name)))
            .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
        EXPECT_CALL(
            *_mockBackend,
            backendSetAttribute(_, HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, _))
            .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
        EXPECT_CALL(*_mockBackend,
                    backendSetAttribute(_,
                                        HIPDNN_ATTR_TENSOR_DIMENSIONS,
                                        HIPDNN_TYPE_INT64,
                                        static_cast<int64_t>(dims.size()),
                                        pointsToVector<int64_t>(dims)))
            .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
        EXPECT_CALL(*_mockBackend,
                    backendSetAttribute(_,
                                        HIPDNN_ATTR_TENSOR_STRIDES,
                                        HIPDNN_TYPE_INT64,
                                        static_cast<int64_t>(strides.size()),
                                        pointsToVector<int64_t>(strides)))
            .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
        EXPECT_CALL(*_mockBackend,
                    backendSetAttribute(_,
                                        HIPDNN_ATTR_TENSOR_IS_VIRTUAL,
                                        HIPDNN_TYPE_BOOLEAN,
                                        1,
                                        pointsToScalar<bool>(false)))
            .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    }

    static std::shared_ptr<TensorAttributes> makeTensor(int64_t uid)
    {
        auto tensor = std::make_shared<TensorAttributes>();
        tensor->set_uid(uid)
            .set_name("tensor_" + std::to_string(uid))
            .set_data_type(DataType::FLOAT)
            .set_dim(toVec(K_DEFAULT_TENSOR_DIMS))
            .set_stride(toVec(K_DEFAULT_TENSOR_STRIDES));
        return tensor;
    }
};

TEST_F(TestDescriptorHelpers, EnsureTensorDescCreatesNewDescriptor)
{
    expectCreateAndDestroyDescriptor();
    expectTensorSetAttributes(K_DEFAULT_TENSOR_UID,
                              "tensor_42",
                              toVec(K_DEFAULT_TENSOR_DIMS),
                              toVec(K_DEFAULT_TENSOR_STRIDES));
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = makeTensor(K_DEFAULT_TENSOR_UID);

    auto err = createOrFindTensorDesc(tensorDescs, tensor);
    EXPECT_TRUE(err.is_good());
    EXPECT_EQ(tensorDescs.size(), 1u);
    EXPECT_TRUE(tensorDescs.find(K_DEFAULT_TENSOR_UID) != tensorDescs.end());
}

TEST_F(TestDescriptorHelpers, EnsureTensorDescDeduplicatesByUid)
{
    expectCreateAndDestroyDescriptor();
    expectTensorSetAttributes(K_DEFAULT_TENSOR_UID,
                              "tensor_42",
                              toVec(K_DEFAULT_TENSOR_DIMS),
                              toVec(K_DEFAULT_TENSOR_STRIDES));
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = makeTensor(K_DEFAULT_TENSOR_UID);

    // First call creates the descriptor
    auto err1 = createOrFindTensorDesc(tensorDescs, tensor);
    EXPECT_TRUE(err1.is_good());
    EXPECT_EQ(tensorDescs.size(), 1u);

    // Second call with same UID reuses existing -- no additional mock calls expected
    auto err2 = createOrFindTensorDesc(tensorDescs, tensor);
    EXPECT_TRUE(err2.is_good());
    EXPECT_EQ(tensorDescs.size(), 1u);
}

TEST_F(TestDescriptorHelpers, EnsureTensorDescFailsOnCreateError)
{
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = makeTensor(K_DEFAULT_TENSOR_UID);

    auto err = createOrFindTensorDesc(tensorDescs, tensor);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestDescriptorHelpers, SetDescriptorAttrVecSucceeds)
{
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_,
                                    HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                    HIPDNN_TYPE_INT64,
                                    3,
                                    pointsToVector<int64_t>({1, 2, 3})))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    const std::vector<int64_t> values = {1, 2, 3};
    hipdnnBackendDescriptor_t desc = nullptr;
    auto err = setDescriptorAttrVec(
        desc, HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, values, "test vec");
    EXPECT_TRUE(err.is_good());
}

TEST_F(TestDescriptorHelpers, SetDescriptorAttrVecReturnsErrorOnFailure)
{
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
        .WillOnce(Return(HIPDNN_STATUS_BAD_PARAM));

    const std::vector<int64_t> values = {1, 2};
    hipdnnBackendDescriptor_t desc = nullptr;
    auto err = setDescriptorAttrVec(
        desc, HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, values, "test vec");
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestDescriptorHelpers, SetDescriptorAttrScalarSucceeds)
{
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(_,
                            HIPDNN_ATTR_CONVOLUTION_CONV_MODE,
                            HIPDNN_TYPE_CONVOLUTION_MODE,
                            1,
                            pointsToScalar<hipdnnConvolutionMode_t>(HIPDNN_CROSS_CORRELATION)))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    const hipdnnConvolutionMode_t value = HIPDNN_CROSS_CORRELATION;
    hipdnnBackendDescriptor_t desc = nullptr;
    auto err = setDescriptorAttrScalar(desc,
                                       HIPDNN_ATTR_CONVOLUTION_CONV_MODE,
                                       HIPDNN_TYPE_CONVOLUTION_MODE,
                                       value,
                                       "test scalar");
    EXPECT_TRUE(err.is_good());
}

TEST_F(TestDescriptorHelpers, SetDescriptorAttrTensorRefSucceeds)
{
    expectCreateAndDestroyDescriptor();
    expectTensorSetAttributes(K_DEFAULT_TENSOR_UID,
                              "tensor_42",
                              toVec(K_DEFAULT_TENSOR_DIMS),
                              toVec(K_DEFAULT_TENSOR_STRIDES));
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    // Create a tensor desc map with an entry
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = makeTensor(K_DEFAULT_TENSOR_UID);
    auto ensureErr = createOrFindTensorDesc(tensorDescs, tensor);
    ASSERT_TRUE(ensureErr.is_good());

    // Expect the tensor ref to be set with BACKEND_DESCRIPTOR type
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            _, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    hipdnnBackendDescriptor_t desc = nullptr;
    auto err = setDescriptorAttrTensorRef(desc,
                                          HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                          K_DEFAULT_TENSOR_UID,
                                          tensorDescs,
                                          "test tensor ref");
    EXPECT_TRUE(err.is_good());
}

TEST_F(TestDescriptorHelpers, FinalizeDescriptorSucceeds)
{
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    hipdnnBackendDescriptor_t desc = nullptr;
    auto err = finalizeDescriptor(desc, "test descriptor");
    EXPECT_TRUE(err.is_good());
}

TEST_F(TestDescriptorHelpers, FinalizeDescriptorReturnsErrorOnFailure)
{
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillOnce(Return(HIPDNN_STATUS_BAD_PARAM));

    hipdnnBackendDescriptor_t desc = nullptr;
    auto err = finalizeDescriptor(desc, "test descriptor");
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestDescriptorHelpers, SetDescriptorAttrScalarReturnsErrorOnFailure)
{
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
        .WillOnce(Return(HIPDNN_STATUS_BAD_PARAM));

    const hipdnnConvolutionMode_t value = HIPDNN_CROSS_CORRELATION;
    hipdnnBackendDescriptor_t desc = nullptr;
    auto err = setDescriptorAttrScalar(desc,
                                       HIPDNN_ATTR_CONVOLUTION_CONV_MODE,
                                       HIPDNN_TYPE_CONVOLUTION_MODE,
                                       value,
                                       "test scalar");
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestDescriptorHelpers, SetDescriptorAttrTensorRefReturnsErrorOnFailure)
{
    expectAllBackendCallsSucceed();

    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = makeTensor(K_DEFAULT_TENSOR_UID);
    auto ensureErr = createOrFindTensorDesc(tensorDescs, tensor);
    ASSERT_TRUE(ensureErr.is_good());

    // Override the mock to fail on the next setAttribute call
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
        .WillOnce(Return(HIPDNN_STATUS_BAD_PARAM));

    hipdnnBackendDescriptor_t desc = nullptr;
    auto err = setDescriptorAttrTensorRef(desc,
                                          HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                          K_DEFAULT_TENSOR_UID,
                                          tensorDescs,
                                          "test tensor ref");
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestDescriptorHelpers, SetDescriptorAttrTensorRefReturnsErrorOnMissingUid)
{
    const std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    hipdnnBackendDescriptor_t desc = nullptr;

    // UID does not exist in the map
    auto err = setDescriptorAttrTensorRef(desc,
                                          HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                          K_MISSING_TENSOR_UID,
                                          tensorDescs,
                                          "missing uid");
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(err.err_msg.find(std::to_string(K_MISSING_TENSOR_UID)) != std::string::npos);
    EXPECT_TRUE(err.err_msg.find("not found") != std::string::npos);
}

TEST_F(TestDescriptorHelpers, EnsureAndSetTensorRefCreatesAndSetsDescriptor)
{
    expectCreateAndDestroyDescriptor();
    expectTensorSetAttributes(K_DEFAULT_TENSOR_UID,
                              "tensor_42",
                              toVec(K_DEFAULT_TENSOR_DIMS),
                              toVec(K_DEFAULT_TENSOR_STRIDES));
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    // Expect the tensor ref to be set on the operation descriptor
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            _, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = makeTensor(K_DEFAULT_TENSOR_UID);
    hipdnnBackendDescriptor_t desc = nullptr;

    auto err = ensureAndSetTensorRef(
        desc, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, tensor, tensorDescs, "conv X");
    EXPECT_TRUE(err.is_good());
    EXPECT_EQ(tensorDescs.size(), 1u);
}

TEST_F(TestDescriptorHelpers, EnsureAndSetTensorRefReusesExistingDescriptor)
{
    expectCreateAndDestroyDescriptor();
    expectTensorSetAttributes(K_DEFAULT_TENSOR_UID,
                              "tensor_42",
                              toVec(K_DEFAULT_TENSOR_DIMS),
                              toVec(K_DEFAULT_TENSOR_STRIDES));
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = makeTensor(K_DEFAULT_TENSOR_UID);

    // First call creates the descriptor
    auto createErr = createOrFindTensorDesc(tensorDescs, tensor);
    ASSERT_TRUE(createErr.is_good());
    EXPECT_EQ(tensorDescs.size(), 1u);

    // ensureAndSetTensorRef should reuse the existing descriptor (no additional create calls)
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            _, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    hipdnnBackendDescriptor_t desc = nullptr;
    auto err = ensureAndSetTensorRef(
        desc, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, tensor, tensorDescs, "conv W");
    EXPECT_TRUE(err.is_good());
    EXPECT_EQ(tensorDescs.size(), 1u);
}

TEST_F(TestDescriptorHelpers, EnsureAndSetTensorRefPropagatesCreateError)
{
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = makeTensor(K_DEFAULT_TENSOR_UID);
    hipdnnBackendDescriptor_t desc = nullptr;

    auto err = ensureAndSetTensorRef(
        desc, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, tensor, tensorDescs, "conv X");
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(tensorDescs.empty());
}

TEST_F(TestDescriptorHelpers, EnsureAndSetTensorRefPropagatesSetAttributeError)
{
    expectCreateAndDestroyDescriptor();
    expectTensorSetAttributes(K_DEFAULT_TENSOR_UID,
                              "tensor_42",
                              toVec(K_DEFAULT_TENSOR_DIMS),
                              toVec(K_DEFAULT_TENSOR_STRIDES));
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    // The setAttribute for the tensor ref itself will fail
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            _, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_BAD_PARAM));

    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = makeTensor(K_DEFAULT_TENSOR_UID);
    hipdnnBackendDescriptor_t desc = nullptr;

    auto err = ensureAndSetTensorRef(
        desc, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, tensor, tensorDescs, "conv X");
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    // Tensor descriptor was created successfully before the ref-set failed
    EXPECT_EQ(tensorDescs.size(), 1u);
}

TEST_F(TestDescriptorHelpers, EnsureTensorDescFailsOnSetAttribute)
{
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    // First setAttribute (UID) succeeds, second (name) fails
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = makeTensor(K_DEFAULT_TENSOR_UID);

    auto err = createOrFindTensorDesc(tensorDescs, tensor);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(tensorDescs.empty());
}

TEST_F(TestDescriptorHelpers, EnsureTensorDescFailsOnFinalize)
{
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = makeTensor(K_DEFAULT_TENSOR_UID);

    auto err = createOrFindTensorDesc(tensorDescs, tensor);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(tensorDescs.empty());
}

TEST_F(TestDescriptorHelpers, EnsureTensorDescSetsPassByValue)
{
    constexpr float K_TENSOR_VALUE = 1.5f;

    expectCreateAndDestroyDescriptor();
    // set_value() resets dims and strides to {1}, so expect scalar dimensions
    expectTensorSetAttributes(K_DEFAULT_TENSOR_UID, "tensor_42", {1}, {1});

    // Expect the value attribute to be set as raw bytes via HIPDNN_TYPE_CHAR
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_,
                                    HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    static_cast<int64_t>(sizeof(float)),
                                    pointsToScalar<float>(K_TENSOR_VALUE)))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = makeTensor(K_DEFAULT_TENSOR_UID);
    tensor->set_value(K_TENSOR_VALUE);

    auto err = createOrFindTensorDesc(tensorDescs, tensor);
    EXPECT_TRUE(err.is_good()) << err.err_msg;
    EXPECT_EQ(tensorDescs.size(), 1u);
}

TEST_F(TestDescriptorHelpers, EnsureTensorDescSetsPassByValueDouble)
{
    constexpr double K_TENSOR_VALUE = 2.718281828;

    expectCreateAndDestroyDescriptor();
    expectTensorSetAttributes(K_DEFAULT_TENSOR_UID, "tensor_42", {1}, {1});

    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_,
                                    HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    static_cast<int64_t>(sizeof(double)),
                                    pointsToScalar<double>(K_TENSOR_VALUE)))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = makeTensor(K_DEFAULT_TENSOR_UID);
    tensor->set_value(K_TENSOR_VALUE);

    auto err = createOrFindTensorDesc(tensorDescs, tensor);
    EXPECT_TRUE(err.is_good()) << err.err_msg;
    EXPECT_EQ(tensorDescs.size(), 1u);
}

TEST_F(TestDescriptorHelpers, EnsureTensorDescSetsPassByValueHalf)
{
    using hipdnn_data_sdk::types::half;
    auto tensorValue = half(1.5f);

    expectCreateAndDestroyDescriptor();
    expectTensorSetAttributes(K_DEFAULT_TENSOR_UID, "tensor_42", {1}, {1});

    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_,
                                    HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    static_cast<int64_t>(sizeof(half)),
                                    pointsToScalar<half>(tensorValue)))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = makeTensor(K_DEFAULT_TENSOR_UID);
    tensor->set_value(tensorValue);

    auto err = createOrFindTensorDesc(tensorDescs, tensor);
    EXPECT_TRUE(err.is_good()) << err.err_msg;
    EXPECT_EQ(tensorDescs.size(), 1u);
}

TEST_F(TestDescriptorHelpers, EnsureTensorDescSetsPassByValueBfloat16)
{
    using hipdnn_data_sdk::types::bfloat16;
    auto tensorValue = bfloat16(1.5f);

    expectCreateAndDestroyDescriptor();
    expectTensorSetAttributes(K_DEFAULT_TENSOR_UID, "tensor_42", {1}, {1});

    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_,
                                    HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    static_cast<int64_t>(sizeof(bfloat16)),
                                    pointsToScalar<bfloat16>(tensorValue)))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = makeTensor(K_DEFAULT_TENSOR_UID);
    tensor->set_value(tensorValue);

    auto err = createOrFindTensorDesc(tensorDescs, tensor);
    EXPECT_TRUE(err.is_good()) << err.err_msg;
    EXPECT_EQ(tensorDescs.size(), 1u);
}

TEST_F(TestDescriptorHelpers, EnsureTensorDescSetsPassByValueUint8)
{
    constexpr uint8_t K_TENSOR_VALUE = 200;

    expectCreateAndDestroyDescriptor();
    expectTensorSetAttributes(K_DEFAULT_TENSOR_UID, "tensor_42", {1}, {1});

    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_,
                                    HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    static_cast<int64_t>(sizeof(uint8_t)),
                                    pointsToScalar<uint8_t>(K_TENSOR_VALUE)))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = makeTensor(K_DEFAULT_TENSOR_UID);
    tensor->set_value(K_TENSOR_VALUE);

    auto err = createOrFindTensorDesc(tensorDescs, tensor);
    EXPECT_TRUE(err.is_good()) << err.err_msg;
    EXPECT_EQ(tensorDescs.size(), 1u);
}

TEST_F(TestDescriptorHelpers, EnsureTensorDescSetsPassByValueInt32)
{
    constexpr int32_t K_TENSOR_VALUE = -42;

    expectCreateAndDestroyDescriptor();
    expectTensorSetAttributes(K_DEFAULT_TENSOR_UID, "tensor_42", {1}, {1});

    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_,
                                    HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    static_cast<int64_t>(sizeof(int32_t)),
                                    pointsToScalar<int32_t>(K_TENSOR_VALUE)))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = makeTensor(K_DEFAULT_TENSOR_UID);
    tensor->set_value(K_TENSOR_VALUE);

    auto err = createOrFindTensorDesc(tensorDescs, tensor);
    EXPECT_TRUE(err.is_good()) << err.err_msg;
    EXPECT_EQ(tensorDescs.size(), 1u);
}

// ============================================================================
// createKnobSettingDescriptor tests
// ============================================================================

TEST_F(TestDescriptorHelpers, CreateKnobSettingDescriptorInt64)
{
    expectCreateAndDestroyDescriptor();

    // Expect: set knob ID (CHAR), set knob value (INT64), finalize
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
                                    HIPDNN_TYPE_CHAR,
                                    static_cast<int64_t>(std::string("test_knob").size()),
                                    pointsToString("test_knob")))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,
                                    HIPDNN_TYPE_INT64,
                                    1,
                                    pointsToScalar<int64_t>(42)))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    const hipdnn_frontend::KnobSetting setting("test_knob", int64_t{42});
    ScopedHipdnnBackendDescriptor desc;
    auto err = createKnobSettingDescriptor(setting, desc);
    EXPECT_TRUE(err.is_good()) << err.err_msg;
    EXPECT_TRUE(desc.valid());
}

TEST_F(TestDescriptorHelpers, CreateKnobSettingDescriptorDouble)
{
    expectCreateAndDestroyDescriptor();

    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
                                    HIPDNN_TYPE_CHAR,
                                    static_cast<int64_t>(std::string("double_knob").size()),
                                    pointsToString("double_knob")))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,
                                    HIPDNN_TYPE_DOUBLE,
                                    1,
                                    pointsToScalar<double>(3.14)))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    const hipdnn_frontend::KnobSetting setting("double_knob", 3.14);
    ScopedHipdnnBackendDescriptor desc;
    auto err = createKnobSettingDescriptor(setting, desc);
    EXPECT_TRUE(err.is_good()) << err.err_msg;
    EXPECT_TRUE(desc.valid());
}

TEST_F(TestDescriptorHelpers, CreateKnobSettingDescriptorString)
{
    expectCreateAndDestroyDescriptor();

    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
                                    HIPDNN_TYPE_CHAR,
                                    static_cast<int64_t>(std::string("str_knob").size()),
                                    pointsToString("str_knob")))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_,
                                    HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,
                                    HIPDNN_TYPE_CHAR,
                                    static_cast<int64_t>(std::string("my_value").size()),
                                    pointsToString("my_value")))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    const hipdnn_frontend::KnobSetting setting("str_knob", std::string("my_value"));
    ScopedHipdnnBackendDescriptor desc;
    auto err = createKnobSettingDescriptor(setting, desc);
    EXPECT_TRUE(err.is_good()) << err.err_msg;
    EXPECT_TRUE(desc.valid());
}

TEST_F(TestDescriptorHelpers, CreateKnobSettingDescriptorFailsOnCreate)
{
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    const hipdnn_frontend::KnobSetting setting("test_knob", int64_t{42});
    ScopedHipdnnBackendDescriptor desc;
    auto err = createKnobSettingDescriptor(setting, desc);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestDescriptorHelpers, CreateKnobSettingDescriptorFailsOnSetAttribute)
{
    expectCreateAndDestroyDescriptor();

    // First setAttribute (knob ID) fails
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
        .WillOnce(Return(HIPDNN_STATUS_BAD_PARAM));

    const hipdnn_frontend::KnobSetting setting("test_knob", int64_t{42});
    ScopedHipdnnBackendDescriptor desc;
    auto err = createKnobSettingDescriptor(setting, desc);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestDescriptorHelpers, CreateKnobSettingDescriptorFailsOnFinalize)
{
    expectCreateAndDestroyDescriptor();

    // setAttribute calls succeed, but finalize fails
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    const hipdnn_frontend::KnobSetting setting("test_knob", int64_t{42});
    ScopedHipdnnBackendDescriptor desc;
    auto err = createKnobSettingDescriptor(setting, desc);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

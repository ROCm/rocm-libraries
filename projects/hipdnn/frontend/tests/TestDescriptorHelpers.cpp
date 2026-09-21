// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <hipdnn_frontend/detail/KnobPacker.hpp>
#include <hipdnn_frontend/detail/TensorConstants.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>
#include <map>
#include <memory>
#include <vector>

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

// Distinctive per-type scalar values for the pass-by-value typed suite; each is
// unique so a mis-serialized byte pattern is caught.
template <typename T>
T passByValueTestScalar();
template <>
float passByValueTestScalar<float>()
{
    return 1.5F;
}
template <>
double passByValueTestScalar<double>()
{
    return 2.718281828;
}
template <>
hipdnn_data_sdk::types::half passByValueTestScalar<hipdnn_data_sdk::types::half>()
{
    return hipdnn_data_sdk::types::half(1.5F);
}
template <>
hipdnn_data_sdk::types::bfloat16 passByValueTestScalar<hipdnn_data_sdk::types::bfloat16>()
{
    return hipdnn_data_sdk::types::bfloat16(1.5F);
}
template <>
uint8_t passByValueTestScalar<uint8_t>()
{
    return 200;
}
template <>
int32_t passByValueTestScalar<int32_t>()
{
    return -42;
}
template <>
int64_t passByValueTestScalar<int64_t>()
{
    return 456789012345LL;
}
template <>
bool passByValueTestScalar<bool>()
{
    return true;
}

} // namespace

class TestDescriptorHelpers : public ::testing::Test
{
protected:
    std::shared_ptr<Mock_hipdnn_backend> _mockBackend;

    void SetUp() override
    {
        _mockBackend = std::make_shared<Mock_hipdnn_backend>();
        IHipdnnBackend::setInstance(_mockBackend);
        // Error-path tests format backend failures via getLastErrorString; allow
        // it everywhere so those calls are not flagged as uninteresting.
        EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());
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
    // The 7 setAttribute calls are: uid, name, data_type, dims, strides, is_virtual,
    // byte alignment.
    void expectTensorSetAttributes(int64_t uid,
                                   const std::string& name,
                                   const std::vector<int64_t>& dims,
                                   const std::vector<int64_t>& strides,
                                   bool isRuntime = false,
                                   int64_t alignment = DEFAULT_TENSOR_ALIGNMENT)
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
        // Lowering only sends the byte-alignment attribute for a non-default
        // alignment (see createOrFindTensorDesc), so the expectation is gated on
        // the same default.
        if(alignment != DEFAULT_TENSOR_ALIGNMENT)
        {
            EXPECT_CALL(*_mockBackend,
                        backendSetAttribute(_,
                                            HIPDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                                            HIPDNN_TYPE_INT64,
                                            1,
                                            pointsToScalar<int64_t>(alignment)))
                .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
        }
        // Lowering only sends the runtime pass-by-value extension attribute when
        // the flag is set (see createOrFindTensorDesc); a non-pass-by-value
        // tensor never touches it, keeping graphs compatible with a pre-1.2.0
        // backend.
        if(isRuntime)
        {
            EXPECT_CALL(*_mockBackend,
                        backendSetAttribute(_,
                                            HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT,
                                            HIPDNN_TYPE_BOOLEAN,
                                            1,
                                            pointsToScalar<bool>(true)))
                .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
        }
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

TEST_F(TestDescriptorHelpers, EnsureTensorDescPropagatesCustomAlignment)
{
    constexpr int64_t K_ALIGNMENT = 256;

    expectCreateAndDestroyDescriptor();
    expectTensorSetAttributes(K_DEFAULT_TENSOR_UID,
                              "tensor_42",
                              toVec(K_DEFAULT_TENSOR_DIMS),
                              toVec(K_DEFAULT_TENSOR_STRIDES),
                              false,
                              K_ALIGNMENT);
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = makeTensor(K_DEFAULT_TENSOR_UID);
    tensor->set_alignment(K_ALIGNMENT);

    auto err = createOrFindTensorDesc(tensorDescs, tensor);
    EXPECT_TRUE(err.is_good());
}

// A tensor at the default alignment must never send HIPDNN_ATTR_TENSOR_BYTE_ALIGNMENT
// during lowering, so an ordinary graph imposes no pre-1.3.0 backend version floor.
// Times(0) makes the omission an assertion rather than relying on strict-mock matching.
TEST_F(TestDescriptorHelpers, EnsureTensorDescOmitsByteAlignmentForDefaultAlignment)
{
    expectCreateAndDestroyDescriptor();
    expectTensorSetAttributes(K_DEFAULT_TENSOR_UID,
                              "tensor_42",
                              toVec(K_DEFAULT_TENSOR_DIMS),
                              toVec(K_DEFAULT_TENSOR_STRIDES));
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, HIPDNN_ATTR_TENSOR_BYTE_ALIGNMENT, _, _, _))
        .Times(0);
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = makeTensor(K_DEFAULT_TENSOR_UID);
    ASSERT_EQ(tensor->get_alignment(), DEFAULT_TENSOR_ALIGNMENT);

    auto err = createOrFindTensorDesc(tensorDescs, tensor);
    EXPECT_TRUE(err.is_good()) << err.err_msg;
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

// Pass-by-value scalar round-trips through createOrFindTensorDesc for every
// supported ValueVariant type. Verifies the six standard tensor attributes and
// the (clear) runtime pass-by-value flag (via expectTensorSetAttributes) --
// set_value() bakes a compile-time constant -- plus the VALUE_EXT wire
// encoding (HIPDNN_TYPE_CHAR, sizeof(T) count, raw bytes).
// set_value() resets dims/strides to {1}, so scalar dimensions are expected.
template <typename T>
class DescriptorHelpersPassByValueTyped : public TestDescriptorHelpers
{
};

using PassByValueScalarTypes = ::testing::Types<float,
                                                double,
                                                hipdnn_data_sdk::types::half,
                                                hipdnn_data_sdk::types::bfloat16,
                                                uint8_t,
                                                int32_t,
                                                int64_t,
                                                bool>;
// Empty third argument required for C++17 compatibility with TYPED_TEST_SUITE macro
TYPED_TEST_SUITE(DescriptorHelpersPassByValueTyped, PassByValueScalarTypes, );

TYPED_TEST(DescriptorHelpersPassByValueTyped, EnsureTensorDescSetsPassByValue)
{
    const TypeParam tensorValue = passByValueTestScalar<TypeParam>();

    this->expectCreateAndDestroyDescriptor();
    this->expectTensorSetAttributes(
        K_DEFAULT_TENSOR_UID, "tensor_42", {1}, {1}, /*isRuntime*/ false);

    EXPECT_CALL(*this->_mockBackend,
                backendSetAttribute(_,
                                    HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    static_cast<int64_t>(sizeof(TypeParam)),
                                    pointsToScalar<TypeParam>(tensorValue)))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*this->_mockBackend, backendFinalize(_)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = this->makeTensor(K_DEFAULT_TENSOR_UID);
    tensor->set_value(tensorValue);

    auto err = createOrFindTensorDesc(tensorDescs, tensor);
    EXPECT_TRUE(err.is_good()) << err.err_msg;
    EXPECT_EQ(tensorDescs.size(), 1u);
}

// Explicit guard for the pre-1.2.0 backend-compatibility contract (RFC 0016
// §1): a tensor that is not runtime pass-by-value must never send
// HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT during lowering, so an
// ordinary graph imposes no new backend version floor. Times(0) makes the
// omission an assertion rather than relying on strict-mock matching.
TEST_F(TestDescriptorHelpers, EnsureTensorDescOmitsPassByValueAttributeForNonPbvTensor)
{
    expectCreateAndDestroyDescriptor();
    expectTensorSetAttributes(K_DEFAULT_TENSOR_UID,
                              "tensor_42",
                              toVec(K_DEFAULT_TENSOR_DIMS),
                              toVec(K_DEFAULT_TENSOR_STRIDES),
                              /*isRuntime=*/false);
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_, HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT, _, _, _))
        .Times(0);
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    auto tensor = makeTensor(K_DEFAULT_TENSOR_UID);

    auto err = createOrFindTensorDesc(tensorDescs, tensor);
    EXPECT_TRUE(err.is_good()) << err.err_msg;
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

// ============================================================================
// Pass-by-value pack -> unpack round-trip tests (RFC-0016 §4.2)
//
// These exercise the real pack path (createOrFindTensorDesc) writing into an
// in-memory backend, then the real unpack path (unpackTensorAttributes) reading
// the same finalized descriptor back. This defends the end-to-end contract that
// the by-value classification (runtime flag + stored value) survives a full
// serialize/deserialize round-trip and reconstructs the correct getter matrix.
// ============================================================================

namespace
{

// A minimal in-memory backend: stores raw attribute bytes on set and serves them
// on get, implementing the two-phase (count query, then value fetch) protocol the
// frontend helpers rely on. HIPDNN_ATTR_TENSOR_IS_BY_VALUE is derived from the
// presence of HIPDNN_ATTR_TENSOR_VALUE_EXT, mirroring the real TensorDescriptor.
class StoringBackend
{
public:
    // Number of bytes one element of the given attribute type occupies.
    static size_t unitSize(hipdnnBackendAttributeType_t type)
    {
        switch(type)
        {
        case HIPDNN_TYPE_INT64:
            return sizeof(int64_t);
        case HIPDNN_TYPE_BOOLEAN:
            return sizeof(bool);
        case HIPDNN_TYPE_CHAR:
            return sizeof(char);
        case HIPDNN_TYPE_DATA_TYPE:
            return sizeof(hipdnnDataType_t);
        default:
            ADD_FAILURE() << "StoringBackend::unitSize: unhandled attribute type "
                          << static_cast<int>(type);
            return 0;
        }
    }

    void wire(Mock_hipdnn_backend& mock)
    {
        ON_CALL(mock, backendCreateDescriptor(_, _))
            .WillByDefault(
                Invoke([this](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* out) {
                    _descriptors.push_back(std::make_unique<int>(0));
                    const auto desc
                        = reinterpret_cast<hipdnnBackendDescriptor_t>(_descriptors.back().get());
                    _store[desc]; // create empty attribute map
                    *out = desc;
                    return HIPDNN_STATUS_SUCCESS;
                }));

        ON_CALL(mock, backendDestroyDescriptor(_)).WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
        ON_CALL(mock, backendFinalize(_)).WillByDefault(Return(HIPDNN_STATUS_SUCCESS));

        ON_CALL(mock, backendSetAttribute(_, _, _, _, _))
            .WillByDefault(Invoke([this](hipdnnBackendDescriptor_t desc,
                                         hipdnnBackendAttributeName_t name,
                                         hipdnnBackendAttributeType_t type,
                                         int64_t count,
                                         const void* arr) {
                const size_t bytes = static_cast<size_t>(count) * unitSize(type);
                Attr attr;
                attr.count = count;
                attr.bytes.resize(bytes);
                if(bytes != 0 && arr != nullptr)
                {
                    std::memcpy(attr.bytes.data(), arr, bytes);
                }
                _store[desc][name] = std::move(attr);
                return HIPDNN_STATUS_SUCCESS;
            }));

        ON_CALL(mock, backendGetAttribute(_, _, _, _, _, _))
            .WillByDefault(Invoke([this](hipdnnBackendDescriptor_t desc,
                                         hipdnnBackendAttributeName_t name,
                                         hipdnnBackendAttributeType_t type,
                                         int64_t requested,
                                         int64_t* outCount,
                                         void* arr) {
                const auto descIt = _store.find(desc);
                if(descIt == _store.end())
                {
                    return HIPDNN_STATUS_BAD_PARAM;
                }

                // IS_BY_VALUE is read-only and derived from value presence.
                if(name == HIPDNN_ATTR_TENSOR_IS_BY_VALUE)
                {
                    const bool present = descIt->second.count(HIPDNN_ATTR_TENSOR_VALUE_EXT) != 0;
                    if(outCount != nullptr)
                    {
                        *outCount = 1;
                    }
                    if(arr != nullptr && requested > 0)
                    {
                        std::memcpy(arr, &present, sizeof(bool));
                    }
                    return HIPDNN_STATUS_SUCCESS;
                }

                const auto attrIt = descIt->second.find(name);
                if(attrIt == descIt->second.end())
                {
                    // Attribute never set: model a legacy/absent field as empty.
                    // Scalar readers leave their default (e.g. false) untouched.
                    if(outCount != nullptr)
                    {
                        *outCount = 0;
                    }
                    return HIPDNN_STATUS_SUCCESS;
                }

                if(outCount != nullptr)
                {
                    *outCount = attrIt->second.count;
                }
                if(arr != nullptr && requested > 0 && !attrIt->second.bytes.empty())
                {
                    // Clamp to the caller's requested capacity so an oversized stored
                    // attribute cannot overflow the caller's buffer.
                    const size_t capacity = static_cast<size_t>(requested) * unitSize(type);
                    const size_t copyBytes = std::min(attrIt->second.bytes.size(), capacity);
                    std::memcpy(arr, attrIt->second.bytes.data(), copyBytes);
                }
                return HIPDNN_STATUS_SUCCESS;
            }));
    }

private:
    struct Attr
    {
        int64_t count = 0;
        std::vector<uint8_t> bytes;
    };
    std::map<hipdnnBackendDescriptor_t, std::map<hipdnnBackendAttributeName_t, Attr>> _store;
    std::vector<std::unique_ptr<int>> _descriptors;
};

} // namespace

class TestDescriptorHelpersRoundTrip : public ::testing::Test
{
protected:
    std::shared_ptr<::testing::NiceMock<Mock_hipdnn_backend>> _backend;
    StoringBackend _storage;

    void SetUp() override
    {
        _backend = std::make_shared<::testing::NiceMock<Mock_hipdnn_backend>>();
        _storage.wire(*_backend);
        IHipdnnBackend::setInstance(_backend);
    }

    void TearDown() override
    {
        IHipdnnBackend::resetInstance();
        _backend.reset();
    }

    // Packs the tensor into a fresh backend descriptor, finalizes it, then unpacks
    // it back into a new TensorAttributes. Returns the reconstructed tensor.
    static std::shared_ptr<TensorAttributes> roundTrip(const std::shared_ptr<TensorAttributes>& in)
    {
        std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
        const auto packErr = createOrFindTensorDesc(tensorDescs, in);
        EXPECT_TRUE(packErr.is_good()) << packErr.err_msg;
        const auto it = tensorDescs.find(in->get_uid());
        EXPECT_TRUE(it != tensorDescs.end());
        if(it == tensorDescs.end())
        {
            return nullptr;
        }
        std::shared_ptr<TensorAttributes> out;
        const auto unpackErr = unpackTensorAttributes(it->second.get(), out);
        EXPECT_TRUE(unpackErr.is_good()) << unpackErr.get_message();
        return out;
    }

    // Builds a scalar tensor carrying only a UID/data_type/dims/strides so the
    // pack path succeeds; callers layer the by-value state on top.
    static std::shared_ptr<TensorAttributes> makeScalar(int64_t uid, DataType dt)
    {
        auto tensor = std::make_shared<TensorAttributes>();
        tensor->set_uid(uid).set_data_type(dt).set_dim({1}).set_stride({1});
        return tensor;
    }
};

// --- Compile-time constant: set_value() -> flag false, value survives ---------

TEST_F(TestDescriptorHelpersRoundTrip, CompileTimeConstantFloatSurvives)
{
    constexpr float K_VALUE = 3.5f;
    auto in = makeScalar(1, DataType::FLOAT);
    in->set_compile_time_constant(K_VALUE);

    const auto out = roundTrip(in);
    ASSERT_NE(out, nullptr);
    EXPECT_FALSE(out->get_is_runtime_pass_by_value());
    EXPECT_TRUE(out->get_is_pass_by_value());
    ASSERT_TRUE(out->get_compile_time_constant<float>().has_value());
    EXPECT_FLOAT_EQ(out->get_compile_time_constant<float>().value(), K_VALUE);
    EXPECT_FALSE(out->get_pass_by_value<float>().has_value());
}

TEST_F(TestDescriptorHelpersRoundTrip, CompileTimeConstantInt64Survives)
{
    constexpr int64_t K_VALUE = -987654321012LL;
    auto in = makeScalar(2, DataType::INT64);
    in->set_compile_time_constant(K_VALUE);

    const auto out = roundTrip(in);
    ASSERT_NE(out, nullptr);
    EXPECT_FALSE(out->get_is_runtime_pass_by_value());
    EXPECT_TRUE(out->get_is_pass_by_value());
    ASSERT_TRUE(out->get_compile_time_constant<int64_t>().has_value());
    EXPECT_EQ(out->get_compile_time_constant<int64_t>().value(), K_VALUE);
    EXPECT_FALSE(out->get_pass_by_value<int64_t>().has_value());
}

TEST_F(TestDescriptorHelpersRoundTrip, CompileTimeConstantBoolSurvives)
{
    constexpr bool K_VALUE = true;
    auto in = makeScalar(3, DataType::BOOLEAN);
    in->set_compile_time_constant(K_VALUE);

    const auto out = roundTrip(in);
    ASSERT_NE(out, nullptr);
    EXPECT_FALSE(out->get_is_runtime_pass_by_value());
    EXPECT_TRUE(out->get_is_pass_by_value());
    ASSERT_TRUE(out->get_compile_time_constant<bool>().has_value());
    EXPECT_EQ(out->get_compile_time_constant<bool>().value(), K_VALUE);
    EXPECT_FALSE(out->get_pass_by_value<bool>().has_value());
}

// --- Runtime-with-default: TensorAttributes(v, RUNTIME_PARAM) -----------------
// flag true + value survives -> get_pass_by_value<T>() == v.

TEST_F(TestDescriptorHelpersRoundTrip, RuntimeWithDefaultFloatSurvives)
{
    constexpr float K_VALUE = 1.25f;
    auto in = makeScalar(4, DataType::FLOAT);
    in->set_value(K_VALUE);
    in->set_is_pass_by_value(true); // value + runtime flag == runtime-with-default

    const auto out = roundTrip(in);
    ASSERT_NE(out, nullptr);
    EXPECT_TRUE(out->get_is_runtime_pass_by_value());
    EXPECT_TRUE(out->get_is_pass_by_value());
    ASSERT_TRUE(out->get_pass_by_value<float>().has_value());
    EXPECT_FLOAT_EQ(out->get_pass_by_value<float>().value(), K_VALUE);
    EXPECT_FALSE(out->get_compile_time_constant<float>().has_value());
}

TEST_F(TestDescriptorHelpersRoundTrip, RuntimeWithDefaultInt64Survives)
{
    constexpr int64_t K_VALUE = 42424242424242LL;
    auto in = makeScalar(5, DataType::INT64);
    in->set_value(K_VALUE);
    in->set_is_pass_by_value(true);

    const auto out = roundTrip(in);
    ASSERT_NE(out, nullptr);
    EXPECT_TRUE(out->get_is_runtime_pass_by_value());
    EXPECT_TRUE(out->get_is_pass_by_value());
    ASSERT_TRUE(out->get_pass_by_value<int64_t>().has_value());
    EXPECT_EQ(out->get_pass_by_value<int64_t>().value(), K_VALUE);
    EXPECT_FALSE(out->get_compile_time_constant<int64_t>().has_value());
}

TEST_F(TestDescriptorHelpersRoundTrip, RuntimeWithDefaultBoolSurvives)
{
    constexpr bool K_VALUE = true;
    auto in = makeScalar(6, DataType::BOOLEAN);
    in->set_value(K_VALUE);
    in->set_is_pass_by_value(true);

    const auto out = roundTrip(in);
    ASSERT_NE(out, nullptr);
    EXPECT_TRUE(out->get_is_runtime_pass_by_value());
    EXPECT_TRUE(out->get_is_pass_by_value());
    ASSERT_TRUE(out->get_pass_by_value<bool>().has_value());
    EXPECT_EQ(out->get_pass_by_value<bool>().value(), K_VALUE);
    EXPECT_FALSE(out->get_compile_time_constant<bool>().has_value());
}

// --- Typed round-trip: every supported scalar type, both baked states --------
// The three hand-written tests above (float/int64/bool) predate this suite and
// are kept for their descriptive names; this closes the gap for
// half/bfloat16/uint8/double that they left unexercised.

template <typename T>
class DescriptorHelpersRoundTripTyped : public TestDescriptorHelpersRoundTrip
{
};

TYPED_TEST_SUITE(DescriptorHelpersRoundTripTyped, PassByValueScalarTypes, );

TYPED_TEST(DescriptorHelpersRoundTripTyped, CompileTimeConstantSurvives)
{
    const TypeParam value = passByValueTestScalar<TypeParam>();
    auto in = std::make_shared<TensorAttributes>(value, ScalarType::COMPILE_TIME_CONST);
    in->set_uid(101);

    const auto out = this->roundTrip(in);
    ASSERT_NE(out, nullptr);
    EXPECT_FALSE(out->get_is_runtime_pass_by_value());
    EXPECT_TRUE(out->get_is_pass_by_value());
    ASSERT_TRUE(out->template get_compile_time_constant<TypeParam>().has_value());
    EXPECT_EQ(out->template get_compile_time_constant<TypeParam>().value(), value);
    EXPECT_FALSE(out->template get_pass_by_value<TypeParam>().has_value());
}

TYPED_TEST(DescriptorHelpersRoundTripTyped, RuntimeWithDefaultSurvives)
{
    const TypeParam value = passByValueTestScalar<TypeParam>();
    auto in = std::make_shared<TensorAttributes>(value, ScalarType::RUNTIME_PARAM);
    in->set_uid(102);

    const auto out = this->roundTrip(in);
    ASSERT_NE(out, nullptr);
    EXPECT_TRUE(out->get_is_runtime_pass_by_value());
    EXPECT_TRUE(out->get_is_pass_by_value());
    ASSERT_TRUE(out->template get_pass_by_value<TypeParam>().has_value());
    EXPECT_EQ(out->template get_pass_by_value<TypeParam>().value(), value);
    EXPECT_FALSE(out->template get_compile_time_constant<TypeParam>().has_value());
}

// --- Runtime user-supplied: set_as_runtime_parameter() -----------------------
// flag true, no value -> both typed getters nullopt, umbrella true.

TEST_F(TestDescriptorHelpersRoundTrip, RuntimeUserSuppliedHasFlagNoValue)
{
    auto in = makeScalar(7, DataType::FLOAT);
    in->set_as_runtime_parameter(); // clears value, sets flag
    in->set_dim({1}).set_stride({1}); // set_as_runtime_parameter left dims intact; keep scalar

    const auto out = roundTrip(in);
    ASSERT_NE(out, nullptr);
    EXPECT_TRUE(out->get_is_runtime_pass_by_value());
    EXPECT_TRUE(out->get_is_pass_by_value());
    EXPECT_FALSE(out->get_pass_by_value<float>().has_value());
    EXPECT_FALSE(out->get_compile_time_constant<float>().has_value());
    EXPECT_TRUE(std::holds_alternative<std::monostate>(out->get_value_variant()));
}

// --- Legacy: value present, runtime flag never written to the descriptor ------
// Unpacks as a compile-time constant (flag defaults false).

TEST_F(TestDescriptorHelpersRoundTrip, LegacyDescriptorWithoutRuntimeFlagIsCompileTimeConst)
{
    constexpr float K_VALUE = 7.75f;

    // Hand-build a descriptor the way an older serializer would: value bytes and
    // IS_BY_VALUE-implying VALUE_EXT are present, but IS_RUNTIME_PASS_BY_VALUE is
    // never set. This exercises the unpack default-false path.
    const ScopedHipdnnBackendDescriptor desc(HIPDNN_BACKEND_TENSOR_DESCRIPTOR);
    ASSERT_TRUE(desc.valid());
    constexpr int64_t K_UID = 8;
    ASSERT_TRUE(setDescriptorAttrScalar(
                    desc.get(), HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, K_UID, "uid")
                    .is_good());
    ASSERT_TRUE(
        setDescriptorAttrDataType(desc.get(), HIPDNN_ATTR_TENSOR_DATA_TYPE, DataType::FLOAT, "dt")
            .is_good());
    ASSERT_TRUE(setDescriptorAttrVec(desc.get(),
                                     HIPDNN_ATTR_TENSOR_DIMENSIONS,
                                     HIPDNN_TYPE_INT64,
                                     std::vector<int64_t>{1},
                                     "dims")
                    .is_good());
    ASSERT_TRUE(setDescriptorAttrVec(desc.get(),
                                     HIPDNN_ATTR_TENSOR_STRIDES,
                                     HIPDNN_TYPE_INT64,
                                     std::vector<int64_t>{1},
                                     "strides")
                    .is_good());
    ASSERT_TRUE(setDescriptorAttrScalar(
                    desc.get(), HIPDNN_ATTR_TENSOR_IS_VIRTUAL, HIPDNN_TYPE_BOOLEAN, false, "virt")
                    .is_good());
    ASSERT_TRUE(setDescriptorAttrTensorValue(desc.get(), K_VALUE, "value").is_good());
    ASSERT_TRUE(finalizeDescriptor(desc.get(), "legacy tensor").is_good());

    std::shared_ptr<TensorAttributes> out;
    const auto unpackErr = unpackTensorAttributes(desc.get(), out);
    ASSERT_TRUE(unpackErr.is_good()) << unpackErr.get_message();
    ASSERT_NE(out, nullptr);
    EXPECT_FALSE(out->get_is_runtime_pass_by_value());
    EXPECT_TRUE(out->get_is_pass_by_value());
    ASSERT_TRUE(out->get_compile_time_constant<float>().has_value());
    EXPECT_FLOAT_EQ(out->get_compile_time_constant<float>().value(), K_VALUE);
    EXPECT_FALSE(out->get_pass_by_value<float>().has_value());
}

// A pre-1.2.0 (pre-1308) backend does not recognize
// HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT and returns NOT_SUPPORTED on
// the read. Unpack must treat that as "flag absent -> false" rather than
// surfacing an error, so a frontend built from this PR can still deserialize a
// descriptor produced by an older backend (RFC 0016 §1 binary-compat guarantee).
TEST_F(TestDescriptorHelpersRoundTrip, UnpackToleratesNotSupportedRuntimeFlagFromLegacyBackend)
{
    // Force the runtime-pbv extension read to report NOT_SUPPORTED, overriding
    // the StoringBackend default for this attribute only (later ON_CALL wins).
    ON_CALL(*_backend,
            backendGetAttribute(_, HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT, _, _, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_NOT_SUPPORTED));

    constexpr float K_VALUE = 4.25f;
    auto in = makeScalar(1, DataType::FLOAT);
    in->set_compile_time_constant(K_VALUE);

    const auto out = roundTrip(in);
    ASSERT_NE(out, nullptr);
    EXPECT_FALSE(out->get_is_runtime_pass_by_value());
    ASSERT_TRUE(out->get_compile_time_constant<float>().has_value());
    EXPECT_FLOAT_EQ(out->get_compile_time_constant<float>().value(), K_VALUE);
}

// --- Mixed set: several scalar tensors of different types + states together ---

TEST_F(TestDescriptorHelpersRoundTrip, MixedScalarTensorsRoundTripIndependently)
{
    // Compile-time const float.
    constexpr float K_CT_FLOAT = 2.5f;
    auto ctFloat = makeScalar(10, DataType::FLOAT);
    ctFloat->set_compile_time_constant(K_CT_FLOAT);

    // Runtime-with-default int64.
    constexpr int64_t K_RT_INT = 123456789LL;
    auto rtInt = makeScalar(11, DataType::INT64);
    rtInt->set_value(K_RT_INT);
    rtInt->set_is_pass_by_value(true);

    // Runtime user-supplied bool (flag only, no value).
    auto rtBool = makeScalar(12, DataType::BOOLEAN);
    rtBool->set_as_runtime_parameter();
    rtBool->set_dim({1}).set_stride({1});

    // Compile-time const bool.
    constexpr bool K_CT_BOOL = true;
    auto ctBool = makeScalar(13, DataType::BOOLEAN);
    ctBool->set_compile_time_constant(K_CT_BOOL);

    // Pack all four into one shared descriptor map, then unpack each descriptor.
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor> tensorDescs;
    for(const auto& t : {ctFloat, rtInt, rtBool, ctBool})
    {
        const auto packErr = createOrFindTensorDesc(tensorDescs, t);
        ASSERT_TRUE(packErr.is_good()) << packErr.err_msg;
    }
    ASSERT_EQ(tensorDescs.size(), 4u);

    const auto unpack = [&](int64_t uid) {
        std::shared_ptr<TensorAttributes> out;
        const auto err = unpackTensorAttributes(tensorDescs.at(uid).get(), out);
        EXPECT_TRUE(err.is_good()) << err.get_message();
        return out;
    };

    const auto outCtFloat = unpack(10);
    ASSERT_NE(outCtFloat, nullptr);
    EXPECT_FALSE(outCtFloat->get_is_runtime_pass_by_value());
    ASSERT_TRUE(outCtFloat->get_compile_time_constant<float>().has_value());
    EXPECT_FLOAT_EQ(outCtFloat->get_compile_time_constant<float>().value(), K_CT_FLOAT);
    EXPECT_FALSE(outCtFloat->get_pass_by_value<float>().has_value());

    const auto outRtInt = unpack(11);
    ASSERT_NE(outRtInt, nullptr);
    EXPECT_TRUE(outRtInt->get_is_runtime_pass_by_value());
    ASSERT_TRUE(outRtInt->get_pass_by_value<int64_t>().has_value());
    EXPECT_EQ(outRtInt->get_pass_by_value<int64_t>().value(), K_RT_INT);
    EXPECT_FALSE(outRtInt->get_compile_time_constant<int64_t>().has_value());

    const auto outRtBool = unpack(12);
    ASSERT_NE(outRtBool, nullptr);
    EXPECT_TRUE(outRtBool->get_is_runtime_pass_by_value());
    EXPECT_TRUE(outRtBool->get_is_pass_by_value());
    EXPECT_FALSE(outRtBool->get_pass_by_value<bool>().has_value());
    EXPECT_FALSE(outRtBool->get_compile_time_constant<bool>().has_value());
    EXPECT_TRUE(std::holds_alternative<std::monostate>(outRtBool->get_value_variant()));

    const auto outCtBool = unpack(13);
    ASSERT_NE(outCtBool, nullptr);
    EXPECT_FALSE(outCtBool->get_is_runtime_pass_by_value());
    ASSERT_TRUE(outCtBool->get_compile_time_constant<bool>().has_value());
    EXPECT_EQ(outCtBool->get_compile_time_constant<bool>().value(), K_CT_BOOL);
    EXPECT_FALSE(outCtBool->get_pass_by_value<bool>().has_value());
}

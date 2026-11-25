// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/ResampleBwdAttributes.hpp>
#include <hipdnn_frontend/node/ResampleBwdNode.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// --- Helper: create fully configured attributes for a valid node ---
namespace
{

ResampleBwdAttributes createValidAttributes()
{
    ResampleBwdAttributes attrs;

    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 3, 16, 16});
    dyTensor->set_stride({768, 256, 16, 1});
    attrs.set_dy(dyTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 3, 32, 32});
    dxTensor->set_stride({3072, 1024, 32, 1});
    attrs.set_dx(dxTensor);

    attrs.set_pre_padding({1, 1});
    attrs.set_post_padding({1, 1});
    attrs.set_stride({2, 2});
    attrs.set_window({3, 3});
    attrs.set_resample_mode(ResampleMode::AVGPOOL_EXCLUDE_PADDING);

    return attrs;
}

std::shared_ptr<TensorAttributes> createValidIndexTensor()
{
    auto indexTensor = std::make_shared<TensorAttributes>();
    indexTensor->set_dim({1, 3, 16, 16});
    indexTensor->set_stride({768, 256, 16, 1});
    indexTensor->set_data_type(DataType::INT32);
    return indexTensor;
}

// dy = {1, 3, 4, 4} with dx left to inference
ResampleBwdAttributes createInferableAttributes()
{
    ResampleBwdAttributes attrs;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 3, 4, 4});
    dyTensor->set_stride({48, 16, 4, 1});
    attrs.set_dy(dyTensor);
    attrs.set_dx(std::make_shared<TensorAttributes>());

    attrs.set_pre_padding({0, 0});
    attrs.set_post_padding({0, 0});
    attrs.set_stride({2, 2});
    attrs.set_window({2, 2});
    attrs.set_resample_mode(ResampleMode::AVGPOOL_EXCLUDE_PADDING);

    return attrs;
}

// Mode values probed by the lowering drift guards; values past the last enumerator are included
// so that an enumerator added later is covered too.
constexpr int K_MODE_PROBE_END = 16;

} // namespace

// --- GetNodeType ---

TEST(TestResampleBwdNode, GetNodeTypeReturnsResampleBwd)
{
    const GraphAttributes graphAttrs;
    const ResampleBwdNode node(ResampleBwdAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::RESAMPLE_BWD);
}

// --- PreValidateNode (success case) ---

TEST(TestResampleBwdNode, PreValidateNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PreValidateNode: missing required tensors ---

TEST(TestResampleBwdNode, PreValidateNodeMissingDyTensor)
{
    ResampleBwdAttributes attrs;

    // Set all required tensors except dy
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 3, 32, 32});
    dxTensor->set_stride({3072, 1024, 32, 1});
    attrs.set_dx(dxTensor);

    attrs.set_pre_padding({1, 1});
    attrs.set_post_padding({1, 1});
    attrs.set_stride({2, 2});
    attrs.set_window({3, 3});

    // dy tensor is missing
    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestResampleBwdNode, PreValidateNodeMissingDxTensor)
{
    ResampleBwdAttributes attrs;

    // Set all required tensors except dx
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 3, 16, 16});
    dyTensor->set_stride({768, 256, 16, 1});
    attrs.set_dy(dyTensor);

    attrs.set_pre_padding({1, 1});
    attrs.set_post_padding({1, 1});
    attrs.set_stride({2, 2});
    attrs.set_window({3, 3});

    // dx tensor is missing
    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestResampleBwdNode, PreValidateNodeMissingParameters)
{
    ResampleBwdAttributes attrs;

    // Set all required tensors but not data field parameters
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 3, 16, 16});
    dyTensor->set_stride({768, 256, 16, 1});
    attrs.set_dy(dyTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 3, 32, 32});
    dxTensor->set_stride({3072, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    attrs.set_resample_mode(ResampleMode::AVGPOOL_EXCLUDE_PADDING);

    // Required parameters are missing
    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestResampleBwdNode, PreValidateNodeAllValuesSet)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestResampleBwdNode, PreValidateNodeMaxpoolRequiresIndex)
{
    auto attrs = createValidAttributes();
    attrs.set_resample_mode(ResampleMode::MAXPOOL);

    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestResampleBwdNode, PreValidateNodeMaxpoolAcceptsIntegerIndex)
{
    auto attrs = createValidAttributes();
    attrs.set_resample_mode(ResampleMode::MAXPOOL);
    attrs.set_index(createValidIndexTensor());

    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestResampleBwdNode, PreValidateNodeRejectsInvalidIndexDataType)
{
    auto attrs = createValidAttributes();
    attrs.set_resample_mode(ResampleMode::MAXPOOL);
    const auto indexTensor = createValidIndexTensor();
    indexTensor->set_data_type(DataType::FLOAT);
    attrs.set_index(indexTensor);

    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestResampleBwdNode, PreValidateNodeRejectsInvalidTensorRank)
{
    auto attrs = createValidAttributes();
    attrs.get_dy()->set_dim({1, 3});

    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestResampleBwdNode, PreValidateNodeRejectsInvalidDxRank)
{
    auto attrs = createValidAttributes();
    attrs.get_dx()->set_dim({1, 3});

    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestResampleBwdNode, PreValidateNodeRejectsInvalidIndexRank)
{
    auto attrs = createValidAttributes();
    attrs.set_resample_mode(ResampleMode::MAXPOOL);
    const auto indexTensor = createValidIndexTensor();
    indexTensor->set_dim({1, 3});
    attrs.set_index(indexTensor);

    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestResampleBwdNode, PreValidateNodeRejectsWindowShorterThanSpatialRank)
{
    auto attrs = createValidAttributes();
    attrs.set_window({3});

    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestResampleBwdNode, PreValidateNodeRejectsWindowLongerThanSpatialRank)
{
    auto attrs = createValidAttributes();
    attrs.set_window({3, 3, 3});
    attrs.set_stride({2, 2, 2});
    attrs.set_pre_padding({1, 1, 1});
    attrs.set_post_padding({1, 1, 1});

    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
    // Pins validateResampleParam's size-mismatch check (ResampleParamsValidation.hpp): an
    // over-long window must be rejected, not just a too-short one.
    EXPECT_NE(error.get_message().find(
                  "window parameter count (3) must match spatial dimension count (2)"),
              std::string::npos)
        << error.get_message();
}

TEST(TestResampleBwdNode, PreValidateNodeRejectsZeroStride)
{
    auto attrs = createInferableAttributes();
    attrs.set_stride({0, 0});

    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestResampleBwdNode, PreValidateNodeRejectsZeroWindowValue)
{
    auto attrs = createValidAttributes();
    attrs.set_window({0, 3});

    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestResampleBwdNode, PreValidateNodeRejectsNegativePadding)
{
    auto preValidateWithPadding
        = [](const std::vector<int64_t>& prePadding, const std::vector<int64_t>& postPadding) {
              auto attrs = createValidAttributes();
              attrs.set_pre_padding(prePadding);
              attrs.set_post_padding(postPadding);

              const GraphAttributes graphAttributes;
              const ResampleBwdNode node(std::move(attrs), graphAttributes);
              return node.pre_validate_node();
          };

    auto error = preValidateWithPadding({-1, 1}, {1, 1});
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);

    error = preValidateWithPadding({1, 1}, {1, -1});
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestResampleBwdNode, PreValidateNodeAcceptsWindowWiderThanPaddedDx)
{
    // The forward of dx gives (8 - 9) / 2 + 1 = 1, matching dy
    auto attrs = createValidAttributes();
    attrs.get_dy()->set_dim({1, 3, 1, 1});
    attrs.get_dx()->set_dim({1, 3, 8, 8});
    attrs.set_pre_padding({0, 0});
    attrs.set_post_padding({0, 0});
    attrs.set_stride({2, 2});
    attrs.set_window({9, 9});

    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestResampleBwdNode, PreValidateNodeAcceptsExactlyTheResampleModesLoweringMaps)
{
    for(int raw = 0; raw < K_MODE_PROBE_END; ++raw)
    {
        const auto mode = static_cast<ResampleMode>(raw);
        auto attrs = createValidAttributes();
        attrs.set_resample_mode(mode);
        attrs.set_index(createValidIndexTensor());

        const GraphAttributes graphAttributes;
        const ResampleBwdNode node(std::move(attrs), graphAttributes);

        auto error = node.pre_validate_node();
        const auto expected = toBackendResampleMode(mode).has_value() ? error_code_t::OK
                                                                      : error_code_t::INVALID_VALUE;
        EXPECT_EQ(error.code, expected) << "resample mode " << raw << ": " << error.err_msg;
    }
}

TEST(TestResampleBwdNode, PreValidateNodeAcceptsExactlyThePaddingModesLoweringMaps)
{
    // NOT_SET has a backend value and stays accepted: resample backward has no padding default
    // to require
    for(int raw = 0; raw < K_MODE_PROBE_END; ++raw)
    {
        const auto mode = static_cast<PaddingMode>(raw);
        auto attrs = createValidAttributes();
        attrs.set_padding_mode(mode);

        const GraphAttributes graphAttributes;
        const ResampleBwdNode node(std::move(attrs), graphAttributes);

        auto error = node.pre_validate_node();
        const auto expected = toBackendPaddingMode(mode).has_value() ? error_code_t::OK
                                                                     : error_code_t::INVALID_VALUE;
        EXPECT_EQ(error.code, expected) << "padding mode " << raw << ": " << error.err_msg;
    }
}

// --- InferPropertiesNode ---

TEST(TestResampleBwdNode, InferPropertiesNode)
{
    ResampleBwdAttributes attrs;
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 3, 16, 16});
    dyTensor->set_stride({768, 256, 16, 1});
    attrs.set_dy(dyTensor);

    const auto dxTensor = std::make_shared<TensorAttributes>();
    attrs.set_dx(dxTensor);

    const auto indexTensor = std::make_shared<TensorAttributes>();
    indexTensor->set_data_type(DataType::INT32);
    attrs.set_index(indexTensor);

    attrs.set_pre_padding({1, 1});
    attrs.set_post_padding({1, 1});
    attrs.set_stride({2, 2});
    attrs.set_window({3, 3});

    const GraphAttributes graphAttributes;
    ResampleBwdNode node(std::move(attrs), graphAttributes);

    const auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
    EXPECT_EQ(dxTensor->get_dim(), (std::vector<int64_t>{1, 3, 31, 31}));
    EXPECT_EQ(dxTensor->get_stride(), (std::vector<int64_t>{2883, 961, 31, 1}));
    EXPECT_EQ(indexTensor->get_dim(), dyTensor->get_dim());
    EXPECT_EQ(indexTensor->get_stride(), dyTensor->get_stride());
}

TEST(TestResampleBwdNode, InferPropertiesNodeRejectsWindowShorterThanSpatialRank)
{
    auto attrs = createInferableAttributes();
    attrs.set_window({2});
    auto dxTensor = attrs.get_dx();

    const GraphAttributes graphAttributes;
    ResampleBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
    EXPECT_TRUE(dxTensor->get_dim().empty());
}

TEST(TestResampleBwdNode, InferPropertiesNodeRejectsEmptyStride)
{
    auto attrs = createInferableAttributes();
    attrs.set_stride(std::vector<int64_t>{});
    auto dxTensor = attrs.get_dx();

    const GraphAttributes graphAttributes;
    ResampleBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(dxTensor->get_dim().empty());
}

TEST(TestResampleBwdNode, InferPropertiesNodeRejectsNonPositiveDxExtent)
{
    auto attrs = createInferableAttributes();
    attrs.get_dy()->set_dim({1, 3, 1, 1});
    attrs.set_stride({1, 1});
    attrs.set_pre_padding({5, 5});
    attrs.set_post_padding({5, 5});
    attrs.set_window({1, 1});
    auto dxTensor = attrs.get_dx();

    const GraphAttributes graphAttributes;
    ResampleBwdNode node(std::move(attrs), graphAttributes);

    // (1 - 1) * 1 - 5 - 5 + 1 = -9
    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
    EXPECT_TRUE(dxTensor->get_dim().empty());
}

TEST(TestResampleBwdNode, InferPropertiesNodeRejectsStrideOverflow)
{
    // (5 - 1) * stride exceeds int64; wrapped, it gives a dx extent of 5
    constexpr int64_t K_STRIDE_PAST_QUARTER = (int64_t{1} << 62) + 1;
    auto attrs = createInferableAttributes();
    attrs.get_dy()->set_dim({1, 3, 5, 5});
    attrs.get_dy()->set_stride({75, 25, 5, 1});
    attrs.set_stride({K_STRIDE_PAST_QUARTER, 1});
    attrs.set_window({1, 1});
    auto dxTensor = attrs.get_dx();

    const GraphAttributes graphAttributes;
    ResampleBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
    EXPECT_NE(error.err_msg.find("overflows int64"), std::string::npos) << error.err_msg;
    EXPECT_TRUE(dxTensor->get_dim().empty());
}

TEST(TestResampleBwdNode, InferPropertiesNodeRejectsPaddingUnderflow)
{
    // 0 - pre_padding - post_padding is below int64's minimum; wrapped, it gives a dx extent of 3
    constexpr int64_t K_MAX = std::numeric_limits<int64_t>::max();
    auto attrs = createInferableAttributes();
    attrs.get_dy()->set_dim({1, 3, 1, 1});
    attrs.set_stride({1, 1});
    attrs.set_pre_padding({K_MAX, 0});
    attrs.set_post_padding({K_MAX, 0});
    attrs.set_window({1, 1});
    auto dxTensor = attrs.get_dx();

    const GraphAttributes graphAttributes;
    ResampleBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
    EXPECT_NE(error.err_msg.find("overflows int64"), std::string::npos) << error.err_msg;
    EXPECT_TRUE(dxTensor->get_dim().empty());
}

// --- GatherHipdnnTensors ---

TEST(TestResampleBwdNode, GatherHipdnnTensor)
{
    ResampleBwdAttributes attrs;

    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(50).set_name("DyTensor");
    attrs.set_dy(dyTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(51).set_name("DxTensor");
    attrs.set_dx(dxTensor);
    const auto indexTensor = std::make_shared<TensorAttributes>();
    indexTensor->set_uid(52).set_name("IndexTensor").set_data_type(DataType::INT32);
    attrs.set_index(indexTensor);

    attrs.set_pre_padding({1, 1});
    attrs.set_post_padding({1, 1});
    attrs.set_stride({2, 2});
    attrs.set_window({3, 3});

    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    hipdnn_frontend::detail::OrderedTensorSet allTensors;

    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(dyTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(dxTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(indexTensor) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 3u);
}

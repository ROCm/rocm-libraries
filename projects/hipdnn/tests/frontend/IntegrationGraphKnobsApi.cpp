// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend.hpp>
#include <test_plugins/TestPluginConstants.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

struct KnobQueryTestCase
{
    std::string description;
    int64_t engineId;
    size_t minKnobCount;
    std::vector<std::string> requiredKnobIds;

    friend std::ostream& operator<<(std::ostream& os, const KnobQueryTestCase& tc)
    {
        os << "KnobQueryTestCase{description: " << tc.description << ", engineId: " << tc.engineId
           << ", minKnobCount: " << tc.minKnobCount << ", requiredKnobIds: [";
        for(size_t i = 0; i < tc.requiredKnobIds.size(); ++i)
        {
            if(i > 0)
            {
                os << ", ";
            }
            os << tc.requiredKnobIds[i];
        }
        os << "]}";
        return os;
    }
};

class IntegrationGraphKnobsApi : public ::testing::TestWithParam<KnobQueryTestCase>
{
protected:
    void SetUp() override
    {
        // Test knobs using multiple plugins
        const std::array<const char*, 2> paths
            = {hipdnn_tests::plugin_constants::testKnobsPluginPath().c_str(),
               hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};

        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
    }

    hipdnnHandle_t _handle = nullptr;
};

} // namespace

INSTANTIATE_TEST_SUITE_P(
    ,
    IntegrationGraphKnobsApi,
    ::testing::Values(
        KnobQueryTestCase{
            "KnobsPluginHasFourKnobs",
            hipdnn_tests::plugin_constants::engineId<KnobsPlugin>(),
            4,
            {"test.int_knob", "test.float_knob", "test.string_knob", "test.deprecated_knob"}},
        KnobQueryTestCase{
            "GoodPluginHasNoKnobs", hipdnn_tests::plugin_constants::engineId<GoodPlugin>(), 0, {}}),
    [](const ::testing::TestParamInfo<KnobQueryTestCase>& info) { return info.param.description; });

TEST_P(IntegrationGraphKnobsApi, QueryKnobsFromEngine)
{
    const auto& testCase = GetParam();

    // Create simple graph (operation doesn't matter for knob queries)
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT).set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_name("X").set_dim({2, 3, 4, 4});

    PointwiseAttributes attrs;
    attrs.set_mode(PointwiseMode::RELU_FWD);
    auto y = graph.pointwise(x, attrs);
    y->set_uid(2);

    auto result = graph.build_operation_graph(_handle);
    ASSERT_TRUE(result.is_good()) << result.get_message();

    std::vector<Knob> knobs;
    result = graph.get_knobs_for_engine(testCase.engineId, knobs);

    ASSERT_TRUE(result.is_good()) << result.get_message();

    // Check minimum knob count (allows for future additions)
    EXPECT_GE(knobs.size(), testCase.minKnobCount)
        << "Engine returned fewer knobs than expected minimum";

    // Verify all required knob IDs are present
    for(const auto& requiredId : testCase.requiredKnobIds)
    {
        auto it = std::find_if(knobs.begin(), knobs.end(), [&requiredId](const Knob& knob) {
            return knob.knobId() == requiredId;
        });
        EXPECT_NE(it, knobs.end())
            << "Required knob '" << requiredId << "' not found in engine " << testCase.engineId;
    }
}

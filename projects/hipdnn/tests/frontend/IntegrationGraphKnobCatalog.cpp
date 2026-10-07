// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <gtest/gtest.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <string>
#include <test_plugins/TestPluginConstants.hpp>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

namespace constants = hipdnn_tests::plugin_constants;

// Routes backend log records at or above @p level to the isolated recorder for its lifetime.
class ScopedUserLogRecording
{
public:
    explicit ScopedUserLogRecording(hipdnnSeverity_t level)
    {
        EXPECT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&_originalLevel), HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(hipdnnSetUserLogCallback_ext(callback(), level, HIPDNN_LOG_CALLBACK_SYNC, this),
                  HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(hipdnnBackendSetGlobalLogLevel_ext(level), HIPDNN_STATUS_SUCCESS);
    }

    ~ScopedUserLogRecording()
    {
        EXPECT_EQ(hipdnnSetUserLogCallback_ext(
                      callback(), HIPDNN_SEV_OFF, HIPDNN_LOG_CALLBACK_SYNC, this),
                  HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(hipdnnBackendSetGlobalLogLevel_ext(_originalLevel), HIPDNN_STATUS_SUCCESS);
    }

    ScopedUserLogRecording(const ScopedUserLogRecording&) = delete;
    ScopedUserLogRecording& operator=(const ScopedUserLogRecording&) = delete;
    ScopedUserLogRecording(ScopedUserLogRecording&&) = delete;
    ScopedUserLogRecording& operator=(ScopedUserLogRecording&&) = delete;

private:
    static hipdnnUserLogCallback_t callback()
    {
        return hipdnn_test_sdk::utilities::IsolatedLogRecorder::getIsolatedUserRecordingCallback();
    }

    hipdnnSeverity_t _originalLevel{HIPDNN_SEV_OFF};
};

class IntegrationGraphKnobCatalog : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const std::array<const char*, 1> paths = {constants::testKnobCatalogPluginPath().c_str()};
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

    Graph createAndBuildSimpleGraph() const
    {
        Graph graph;
        graph.set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_name("X").set_dim({2, 3, 4, 4}).set_stride({48, 16, 4, 1});

        PointwiseAttributes attrs;
        attrs.set_mode(PointwiseMode::RELU_FWD);
        auto y = graph.pointwise(x, attrs);
        y->set_name("Y");

        auto result = graph.validate();
        EXPECT_TRUE(result.is_good()) << result.get_message();

        result = graph.build_operation_graph(_handle);
        EXPECT_TRUE(result.is_good()) << result.get_message();

        return graph;
    }

    static std::vector<std::vector<KnobSetting>> servedCatalog()
    {
        const auto tile
            = [](int64_t value) { return KnobSetting(constants::K_KNOB_CATALOG_TILE_KNOB, value); };
        const auto mode = [](const std::string& value) {
            return KnobSetting(constants::K_KNOB_CATALOG_MODE_KNOB, value);
        };
        return {{tile(16), mode("fast")}, {tile(32), mode("fast")}, {tile(64), mode("accurate")}};
    }

    hipdnnHandle_t _handle = nullptr;
};

} // namespace

TEST_F(IntegrationGraphKnobCatalog, CatalogRoundTripsToKnobSettings)
{
    const Graph graph = createAndBuildSimpleGraph();

    std::vector<std::vector<KnobSetting>> catalog;
    auto result
        = graph.get_knob_catalog_for_engine(constants::engineId<KnobCatalogPlugin>(), catalog);
    ASSERT_TRUE(result.is_good()) << result.get_message();

    EXPECT_EQ(catalog, servedCatalog());
}

TEST_F(IntegrationGraphKnobCatalog,
       ReadingAStringKnobCatalogRecordsNoBackendErrorAndSetsNoLastError)
{
    const Graph graph = createAndBuildSimpleGraph();

    // Reading the last error clears it, so what is read afterwards was set by the catalog read.
    std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> stale{};
    hipdnnGetLastErrorString(stale.data(), stale.size());

    auto recorder
        = hipdnn_test_sdk::utilities::IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    const ScopedUserLogRecording recording(HIPDNN_SEV_ERROR);

    std::vector<std::vector<KnobSetting>> catalog;
    auto result
        = graph.get_knob_catalog_for_engine(constants::engineId<KnobCatalogPlugin>(), catalog);

    ASSERT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(catalog, servedCatalog());

    // The catalog holds string-valued choices: reading them must not go through failing
    // backend calls, which log an error and overwrite the last error each time.
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), 0U)
        << recorder.getRecordedLogsAsString();

    std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> after{};
    hipdnnGetLastErrorString(after.data(), after.size());
    EXPECT_EQ(std::string(after.data()), "");

    // Control: a failing backend call is recorded, so the zero above is not a deaf recorder.
    ASSERT_NE(hipdnnBackendGetAttribute(nullptr,
                                        HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        nullptr,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_GT(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), 0U);
}

TEST_F(IntegrationGraphKnobCatalog, EveryCatalogEntryCreatesAnExecutionPlan)
{
    Graph graph = createAndBuildSimpleGraph();
    const auto engineId = constants::engineId<KnobCatalogPlugin>();

    std::vector<std::vector<KnobSetting>> catalog;
    auto result = graph.get_knob_catalog_for_engine(engineId, catalog);
    ASSERT_TRUE(result.is_good()) << result.get_message();
    ASSERT_FALSE(catalog.empty());

    for(size_t i = 0; i < catalog.size(); ++i)
    {
        auto planResult = graph.create_execution_plan_ext(engineId, catalog[i]);
        EXPECT_TRUE(planResult.is_good()) << "entry " << i << ": " << planResult.get_message();
    }
}

TEST_F(IntegrationGraphKnobCatalog, PluginDecliningACatalogIsReportedAsABackendError)
{
    const Graph graph = createAndBuildSimpleGraph();
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter mode(
        constants::K_KNOB_CATALOG_PLUGIN_MODE_ENV, constants::K_KNOB_CATALOG_MODE_NOT_APPLICABLE);

    std::vector<std::vector<KnobSetting>> catalog;
    auto result
        = graph.get_knob_catalog_for_engine(constants::engineId<KnobCatalogPlugin>(), catalog);

    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(catalog.empty());
}

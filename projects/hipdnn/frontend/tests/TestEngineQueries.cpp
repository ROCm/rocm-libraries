// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestEngineQueries.cpp
 * @brief Frontend unit tests for the ranking-metric side of the engine prediction queries.
 *
 * Drives detail::getEnginePrediction and detail::getPredictionCapabilities through
 * Mock_hipdnn_backend. The mock answers each query from the metric the frontend set on
 * the queried descriptor, so what is asserted is the round trip: the metric reaches the
 * backend on the descriptor that defines the kind, and the answer is held to it.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hipdnn_frontend/detail/EngineQueries.hpp>

#include "fake_backend/MockHipdnnBackend.hpp"

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace hipdnn_frontend;
using namespace ::testing;

namespace
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

constexpr int64_t ENGINE_ID = 7;

class TestEngineQueries : public ::testing::Test
{
protected:
    std::shared_ptr<NiceMock<Mock_hipdnn_backend>> _mockBackend;
    std::array<char, 16> _fakeDescs{};
    size_t _nextFakeDescIdx = 0;

    /// Metric the frontend last set on the queried descriptor, as the backend sees it.
    std::string _requestedMetric;
    /// Answers one query; defaults to echoing the requested metric as UNAVAILABLE.
    std::function<fb::EnginePredictionT(PredictionKind, const std::string&)> _respond;
    std::vector<flatbuffers::DetachedBuffer> _responses;

    void SetUp() override
    {
        _mockBackend = std::make_shared<NiceMock<Mock_hipdnn_backend>>();
        detail::IHipdnnBackend::setInstance(_mockBackend);

        ON_CALL(*_mockBackend, backendCreateDescriptor(_, _))
            .WillByDefault([this](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
                *desc = reinterpret_cast<hipdnnBackendDescriptor_t>(
                    &_fakeDescs[_nextFakeDescIdx++ % _fakeDescs.size()]);
                return HIPDNN_STATUS_SUCCESS;
            });
        ON_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
            .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
        ON_CALL(*_mockBackend,
                backendSetAttribute(_,
                                    AnyOf(HIPDNN_ATTR_ENGINE_PREDICTION_METRIC_EXT,
                                          HIPDNN_ATTR_ENGINECFG_RANKING_METRIC_EXT),
                                    HIPDNN_TYPE_CHAR,
                                    _,
                                    _))
            .WillByDefault([this](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t count,
                                  const void* value) {
                _requestedMetric.assign(static_cast<const char*>(value),
                                        static_cast<size_t>(count));
                return HIPDNN_STATUS_SUCCESS;
            });
        ON_CALL(*_mockBackend, backendFinalize(_)).WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
        ON_CALL(*_mockBackend, backendDestroyDescriptor(_))
            .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
        ON_CALL(*_mockBackend,
                backendGetAttribute(
                    _,
                    AnyOf(HIPDNN_ATTR_ENGINE_PREDICTION_EXT, HIPDNN_ATTR_ENGINECFG_PREDICTION_EXT),
                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                    1,
                    _,
                    _))
            .WillByDefault([this](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t name,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* out) {
                const auto kind = name == HIPDNN_ATTR_ENGINE_PREDICTION_EXT
                                      ? PredictionKind::ENGINE
                                      : PredictionKind::CONFIGURATION;
                auto response = _respond(kind, _requestedMetric);
                response.engine_id = ENGINE_ID;
                response.kind = kind == PredictionKind::ENGINE ? fb::PredictionKind::ENGINE
                                                               : fb::PredictionKind::CONFIGURATION;
                flatbuffers::FlatBufferBuilder builder;
                builder.Finish(fb::EnginePrediction::Pack(builder, &response));
                _responses.push_back(builder.Release());
                auto* data = static_cast<hipdnnBackendFlatbufferData_t*>(out);
                data->ptr = _responses.back().data();
                data->size = _responses.back().size();
                return HIPDNN_STATUS_SUCCESS;
            });

        _respond = [](PredictionKind, const std::string& metric) {
            fb::EnginePredictionT response;
            response.metric = metric;
            return response;
        };
    }

    void TearDown() override
    {
        detail::IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }

    static hipdnnBackendDescriptor_t graph()
    {
        static int s_sentinel = 0;
        return reinterpret_cast<hipdnnBackendDescriptor_t>(&s_sentinel);
    }
};

TEST_F(TestEngineQueries, PredictionIsAnsweredInTheRequestedMetric)
{
    _respond = [](PredictionKind kind, const std::string& metric) {
        fb::EnginePredictionT response;
        response.metric = metric;
        response.status = fb::PredictionStatus::AVAILABLE;
        response.value = 0.25;
        response.uhd_id = "time-model";
        if(kind == PredictionKind::CONFIGURATION)
        {
            response.engine_config = std::make_unique<fb::EngineConfigT>();
            response.engine_config->engine_id = ENGINE_ID;
        }
        return response;
    };

    for(const auto kind : {PredictionKind::ENGINE, PredictionKind::CONFIGURATION})
    {
        EnginePrediction prediction;
        const auto error = detail::getEnginePrediction(
            graph(), ENGINE_ID, prediction, kind, /*evaluate=*/true, {}, "time");
        ASSERT_TRUE(error.is_good()) << error.get_message();
        EXPECT_EQ(_requestedMetric, "time");
        EXPECT_EQ(prediction.metric, "time");
        ASSERT_TRUE(prediction.value.has_value());
        EXPECT_DOUBLE_EQ(*prediction.value, 0.25);
    }
}

TEST_F(TestEngineQueries, PredictionDefaultsToTflops)
{
    EnginePrediction prediction;
    const auto error = detail::getEnginePrediction(graph(), ENGINE_ID, prediction);
    ASSERT_TRUE(error.is_good()) << error.get_message();
    EXPECT_EQ(_requestedMetric, "tflops");
    EXPECT_EQ(prediction.metric, "tflops");
    EXPECT_EQ(prediction.status, PredictionStatus::UNAVAILABLE);
    EXPECT_FALSE(prediction.value.has_value());
}

TEST_F(TestEngineQueries, UnregisteredMetricIsRejectedBeforeAnyBackendCall)
{
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _)).Times(0);
    for(const char* metric : {"latency", "TFLOPS", ""})
    {
        EnginePrediction prediction;
        const auto error = detail::getEnginePrediction(
            graph(), ENGINE_ID, prediction, PredictionKind::ENGINE, true, {}, metric);
        EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE) << metric;
    }
}

// An answer in another metric is a different quantity, not a stale one: it is refused
// rather than converted or passed through under the requested name.
TEST_F(TestEngineQueries, AnswerInAnotherMetricIsRejected)
{
    _respond = [](PredictionKind, const std::string&) {
        fb::EnginePredictionT response;
        response.metric = "tflops";
        response.status = fb::PredictionStatus::AVAILABLE;
        response.value = 12.0;
        return response;
    };
    EnginePrediction prediction;
    const auto error = detail::getEnginePrediction(
        graph(), ENGINE_ID, prediction, PredictionKind::ENGINE, true, {}, "time");
    EXPECT_EQ(error.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_FALSE(prediction.value.has_value());
}

TEST_F(TestEngineQueries, AnswerWithoutMetricIsRejected)
{
    _respond = [](PredictionKind, const std::string&) { return fb::EnginePredictionT{}; };
    EnginePrediction prediction;
    const auto error = detail::getEnginePrediction(graph(), ENGINE_ID, prediction);
    EXPECT_EQ(error.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

// Validity follows the metric: zero throughput is a legal measurement, zero time is not.
TEST_F(TestEngineQueries, AvailableValueIsValidatedAgainstTheMetric)
{
    _respond = [](PredictionKind, const std::string& metric) {
        fb::EnginePredictionT response;
        response.metric = metric;
        response.status = fb::PredictionStatus::AVAILABLE;
        response.value = 0.0;
        return response;
    };
    EnginePrediction prediction;
    EXPECT_TRUE(detail::getEnginePrediction(
                    graph(), ENGINE_ID, prediction, PredictionKind::ENGINE, true, {}, "tflops")
                    .is_good());
    EXPECT_EQ(detail::getEnginePrediction(
                  graph(), ENGINE_ID, prediction, PredictionKind::ENGINE, true, {}, "time")
                  .code,
              ErrorCode::HIPDNN_BACKEND_ERROR);
}

// Capabilities report exactly the (kind, metric) pairs with a bound, non-INVALID model:
// an unbound metric and an INVALID binding are both left out.
TEST_F(TestEngineQueries, CapabilitiesListBoundModelsPerKindAndMetric)
{
    _respond = [](PredictionKind kind, const std::string& metric) {
        fb::EnginePredictionT response;
        response.metric = metric;
        if(kind == PredictionKind::ENGINE && metric == "tflops")
        {
            response.uhd_id = "l1-tflops";
        }
        else if(kind == PredictionKind::CONFIGURATION && metric == "tflops")
        {
            response.uhd_id = "l2-tflops-broken";
            response.status = fb::PredictionStatus::INVALID;
        }
        else if(kind == PredictionKind::CONFIGURATION && metric == "time")
        {
            response.uhd_id = "l2-time";
        }
        return response;
    };

    std::vector<PredictionCapability> capabilities{{PredictionKind::ENGINE, "stale", "stale"}};
    const auto error = detail::getPredictionCapabilities(graph(), ENGINE_ID, capabilities);
    ASSERT_TRUE(error.is_good()) << error.get_message();
    ASSERT_EQ(capabilities.size(), 2u);
    EXPECT_EQ(capabilities[0].kind, PredictionKind::ENGINE);
    EXPECT_EQ(capabilities[0].metric, "tflops");
    EXPECT_EQ(capabilities[0].model, "l1-tflops");
    EXPECT_EQ(capabilities[1].kind, PredictionKind::CONFIGURATION);
    EXPECT_EQ(capabilities[1].metric, "time");
    EXPECT_EQ(capabilities[1].model, "l2-time");
}

TEST_F(TestEngineQueries, CapabilitiesNeverEvaluateAModel)
{
    std::vector<int64_t> evaluateFlags;
    // Every other attribute keeps the fixture's default action.
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_,
                                    AnyOf(HIPDNN_ATTR_ENGINE_PREDICTION_EVALUATE_EXT,
                                          HIPDNN_ATTR_ENGINECFG_PREDICTION_EVALUATE_EXT),
                                    HIPDNN_TYPE_INT64,
                                    1,
                                    _))
        .WillRepeatedly([&evaluateFlags](hipdnnBackendDescriptor_t,
                                         hipdnnBackendAttributeName_t,
                                         hipdnnBackendAttributeType_t,
                                         int64_t,
                                         const void* value) {
            evaluateFlags.push_back(*static_cast<const int64_t*>(value));
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<PredictionCapability> capabilities;
    ASSERT_TRUE(detail::getPredictionCapabilities(graph(), ENGINE_ID, capabilities).is_good());
    EXPECT_EQ(evaluateFlags.size(), 2 * hipdnn_data_sdk::utilities::RANKING_METRICS.size());
    EXPECT_THAT(evaluateFlags, Each(0));
}

} // namespace

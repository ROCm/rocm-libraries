// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestEngineQueries.cpp
 * @brief Frontend tests for the ranking metric in the engine prediction queries.
 *
 * Mock_hipdnn_backend answers from the metric set on the queried descriptor.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hipdnn_frontend/detail/EngineQueries.hpp>

#include "fake_backend/MockHipdnnBackend.hpp"

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <utility>
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

    /// Answers candidate-page reads with @p response as it is at read time.
    void serveCandidatePage(const flatbuffers::DetachedBuffer& response)
    {
        ON_CALL(*_mockBackend,
                backendGetAttribute(_, HIPDNN_ATTR_ENGINE_CANDIDATES_EXT, _, _, _, _))
            .WillByDefault([&response](hipdnnBackendDescriptor_t,
                                       hipdnnBackendAttributeName_t,
                                       hipdnnBackendAttributeType_t,
                                       int64_t,
                                       int64_t*,
                                       void* out) {
                auto* data = static_cast<hipdnnBackendFlatbufferData_t*>(out);
                data->ptr = response.data();
                data->size = response.size();
                return HIPDNN_STATUS_SUCCESS;
            });
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

// Refused rather than converted or passed through under the requested name.
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

// Zero is invalid under every metric: a 0 throughput means "no measurement".
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
    for(const auto* metric : {"tflops", "time"})
    {
        EXPECT_EQ(detail::getEnginePrediction(
                      graph(), ENGINE_ID, prediction, PredictionKind::ENGINE, true, {}, metric)
                      .code,
                  ErrorCode::HIPDNN_BACKEND_ERROR)
            << metric;
    }
}

// Unbound metrics and INVALID bindings are both left out.
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

// A KnobSetting tagged with no payload table verifies, and its typed accessor returns
// null; both decoders must report that as an error.
constexpr std::array<fb::KnobValue, 3> TAGGED_KNOB_VALUES{
    fb::KnobValue::IntValue, fb::KnobValue::FloatValue, fb::KnobValue::StringValue};

/// One knob named "test.knob" whose tag is @p tag; the payload is built only when
/// @p withPayload, so the bare case leaves a set tag over an absent union value.
flatbuffers::Offset<fb::KnobSetting>
    knobSetting(flatbuffers::FlatBufferBuilder& builder, fb::KnobValue tag, bool withPayload)
{
    flatbuffers::Offset<void> payload = 0;
    if(withPayload)
    {
        switch(tag)
        {
        case fb::KnobValue::IntValue:
            payload = fb::CreateIntValue(builder, 4).Union();
            break;
        case fb::KnobValue::FloatValue:
            payload = fb::CreateFloatValue(builder, 0.5).Union();
            break;
        default:
            payload = fb::CreateStringValueDirect(builder, "s").Union();
            break;
        }
    }
    return fb::CreateKnobSettingDirect(builder, "test.knob", tag, payload);
}

flatbuffers::DetachedBuffer configurationPrediction(fb::KnobValue tag, bool withPayload)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<flatbuffers::Offset<fb::KnobSetting>> knobs{
        knobSetting(builder, tag, withPayload)};
    const auto config = fb::CreateEngineConfigDirect(builder, ENGINE_ID, &knobs);
    builder.Finish(fb::CreateEnginePredictionDirect(builder,
                                                    ENGINE_ID,
                                                    fb::PredictionKind::CONFIGURATION,
                                                    fb::PredictionStatus::AVAILABLE,
                                                    1.0,
                                                    "model",
                                                    nullptr,
                                                    config,
                                                    "{}",
                                                    "{}",
                                                    "tflops"));
    return builder.Release();
}

flatbuffers::DetachedBuffer candidatePage(fb::KnobValue tag, bool withPayload)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<flatbuffers::Offset<fb::KnobSetting>> knobs{
        knobSetting(builder, tag, withPayload)};
    const std::vector<flatbuffers::Offset<fb::EngineCandidate>> candidates{
        fb::CreateEngineCandidateDirect(builder, "candidate", &knobs, "{}")};
    const auto page = fb::CreateEngineCandidatePageDirect(
        builder, "graph", "device", "gfx942", "{}", "{}", 1, 0, &candidates);
    builder.Finish(
        fb::CreateEngineDetailsDirect(builder, ENGINE_ID, nullptr, nullptr, nullptr, page));
    return builder.Release();
}

TEST_F(TestEngineQueries, PredictionKnobTaggedWithoutItsValueIsAnError)
{
    flatbuffers::DetachedBuffer response;
    ON_CALL(*_mockBackend, backendGetAttribute(_, HIPDNN_ATTR_ENGINECFG_PREDICTION_EXT, _, _, _, _))
        .WillByDefault([&response](hipdnnBackendDescriptor_t,
                                   hipdnnBackendAttributeName_t,
                                   hipdnnBackendAttributeType_t,
                                   int64_t,
                                   int64_t*,
                                   void* out) {
            auto* data = static_cast<hipdnnBackendFlatbufferData_t*>(out);
            data->ptr = response.data();
            data->size = response.size();
            return HIPDNN_STATUS_SUCCESS;
        });

    for(const auto tag : TAGGED_KNOB_VALUES)
    {
        SCOPED_TRACE(fb::EnumNameKnobValue(tag));
        EnginePrediction prediction;

        // Control: the same response with its payload decodes, so the failure below is
        // the missing payload and nothing else about the buffer.
        response = configurationPrediction(tag, /*withPayload=*/true);
        const auto good = detail::getEnginePrediction(
            graph(), ENGINE_ID, prediction, PredictionKind::CONFIGURATION);
        ASSERT_TRUE(good.is_good()) << good.get_message();

        response = configurationPrediction(tag, /*withPayload=*/false);
        flatbuffers::Verifier verifier(response.data(), response.size());
        ASSERT_TRUE(verifier.VerifyBuffer<fb::EnginePrediction>());
        EXPECT_FALSE(detail::getEnginePrediction(
                         graph(), ENGINE_ID, prediction, PredictionKind::CONFIGURATION)
                         .is_good());
    }
}

TEST_F(TestEngineQueries, CandidateKnobTaggedWithoutItsValueIsAnError)
{
    flatbuffers::DetachedBuffer response;
    serveCandidatePage(response);

    for(const auto tag : TAGGED_KNOB_VALUES)
    {
        SCOPED_TRACE(fb::EnumNameKnobValue(tag));
        EngineCandidatePage page;

        response = candidatePage(tag, /*withPayload=*/true);
        const auto good = detail::getEngineCandidates(graph(), ENGINE_ID, page);
        ASSERT_TRUE(good.is_good()) << good.get_message();
        ASSERT_EQ(page.candidates.size(), 1U);

        response = candidatePage(tag, /*withPayload=*/false);
        flatbuffers::Verifier verifier(response.data(), response.size());
        ASSERT_TRUE(verifier.VerifyBuffer<fb::EngineDetails>());
        EXPECT_FALSE(detail::getEngineCandidates(graph(), ENGINE_ID, page).is_good());
    }
}

/// A one-candidate page omitting page field @p pageField and candidate field
/// @p candidateField (0 omits neither). Hand-built: the generated builders assert that
/// required fields are present.
flatbuffers::DetachedBuffer pageOmitting(flatbuffers::voffset_t pageField,
                                         flatbuffers::voffset_t candidateField)
{
    using Candidate = fb::EngineCandidate;
    using Page = fb::EngineCandidatePage;
    flatbuffers::FlatBufferBuilder builder;
    const auto addUnless = [&builder](flatbuffers::voffset_t omitted,
                                      flatbuffers::voffset_t field,
                                      flatbuffers::Offset<flatbuffers::String> value) {
        if(field != omitted)
        {
            builder.AddOffset(field, value);
        }
    };
    const auto id = builder.CreateString("candidate");
    const auto kernelFeatures = builder.CreateString("{}");
    const auto candidateStart = builder.StartTable();
    addUnless(candidateField, Candidate::VT_ID, id);
    addUnless(candidateField, Candidate::VT_KERNEL_FEATURES, kernelFeatures);
    const std::vector<flatbuffers::Offset<Candidate>> candidates{
        flatbuffers::Offset<Candidate>(builder.EndTable(candidateStart))};
    const auto candidateVector = builder.CreateVector(candidates);
    const std::array<std::pair<flatbuffers::voffset_t, flatbuffers::Offset<flatbuffers::String>>, 5>
        strings{{{Page::VT_GRAPH_ID, builder.CreateString("graph")},
                 {Page::VT_DEVICE_ID, builder.CreateString("device")},
                 {Page::VT_DEVICE_ARCH, builder.CreateString("gfx942")},
                 {Page::VT_PROBLEM_FEATURES, builder.CreateString("{}")},
                 {Page::VT_DEVICE_FEATURES, builder.CreateString("{}")}}};
    const auto pageStart = builder.StartTable();
    for(const auto& [field, value] : strings)
    {
        addUnless(pageField, field, value);
    }
    builder.AddElement<uint64_t>(Page::VT_TOTAL_COUNT, 1, 0);
    builder.AddOffset(Page::VT_CANDIDATES, candidateVector);
    const flatbuffers::Offset<Page> page(builder.EndTable(pageStart));
    builder.Finish(fb::CreateEngineDetails(builder, ENGINE_ID, 0, 0, 0, page));
    return builder.Release();
}

TEST_F(TestEngineQueries, CandidatePageMissingARequiredFieldIsAnError)
{
    using Candidate = fb::EngineCandidate;
    using Page = fb::EngineCandidatePage;
    flatbuffers::DetachedBuffer response;
    serveCandidatePage(response);
    EngineCandidatePage page;

    // Control: the hand-built page decodes when nothing is omitted.
    response = pageOmitting(0, 0);
    const auto good = detail::getEngineCandidates(graph(), ENGINE_ID, page);
    ASSERT_TRUE(good.is_good()) << good.get_message();

    const std::array<std::pair<flatbuffers::voffset_t, flatbuffers::voffset_t>, 7> omissions{
        {{Page::VT_GRAPH_ID, 0},
         {Page::VT_DEVICE_ID, 0},
         {Page::VT_DEVICE_ARCH, 0},
         {Page::VT_PROBLEM_FEATURES, 0},
         {Page::VT_DEVICE_FEATURES, 0},
         {0, Candidate::VT_ID},
         {0, Candidate::VT_KERNEL_FEATURES}}};
    for(const auto& [pageField, candidateField] : omissions)
    {
        SCOPED_TRACE(::testing::Message()
                     << "page field " << pageField << ", candidate field " << candidateField);
        response = pageOmitting(pageField, candidateField);
        EXPECT_EQ(detail::getEngineCandidates(graph(), ENGINE_ID, page).get_code(),
                  ErrorCode::HIPDNN_BACKEND_ERROR);
    }
}

// Unique ids need not ascend: the page keeps the plugin's stable order.
TEST_F(TestEngineQueries, CandidatesKeepPluginOrderWhenIdsAreNotAscending)
{
    flatbuffers::FlatBufferBuilder builder;
    const auto candidate = [&builder](const char* id, int64_t knobValue) {
        const std::vector<flatbuffers::Offset<fb::KnobSetting>> knobs{
            fb::CreateKnobSettingDirect(builder,
                                        "test.knob",
                                        fb::KnobValue::IntValue,
                                        fb::CreateIntValue(builder, knobValue).Union())};
        return fb::CreateEngineCandidateDirect(builder, id, &knobs, "{}");
    };
    const std::vector<flatbuffers::Offset<fb::EngineCandidate>> candidates{candidate("b", 1),
                                                                           candidate("a", 2)};
    const auto source = fb::CreateEngineCandidatePageDirect(
        builder, "graph", "device", "gfx942", "{}", "{}", 2, 0, &candidates);
    builder.Finish(
        fb::CreateEngineDetailsDirect(builder, ENGINE_ID, nullptr, nullptr, nullptr, source));
    const auto response = builder.Release();
    serveCandidatePage(response);

    EngineCandidatePage page;
    const auto error = detail::getEngineCandidates(graph(), ENGINE_ID, page);
    ASSERT_TRUE(error.is_good()) << error.get_message();
    ASSERT_EQ(page.candidates.size(), 2U);
    EXPECT_EQ(page.candidates[0].id, "b");
    EXPECT_EQ(page.candidates[1].id, "a");
}

} // namespace

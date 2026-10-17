// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// EXECUTION_PLAN, VARIANT_PACK and cudnnBackendExecute against the mock hipDNN
// backend: what each lowers to, the plan JSON format, and the execute path.
#include <hipdnn_compatibility/cudnn/cudnn_graph.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "fake_backend/ShimV8MockBackendFixture.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

namespace backend = hipdnn_frontend::compatibility::cudnn_frontend::detail::backend;

using ::testing::_;

std::string lastError()
{
    std::array<char, 1024> message{};
    cudnnGetLastErrorString(message.data(), message.size());
    return message.data();
}

class TestCudnnShimV8ExecutionPlan : public hipdnn_shim_test::ShimV8MockBackendFixture
{
protected:
    // Ranked {7, -2} gives the dense index {-2, 7}; the tests use index 1.
    static constexpr std::int64_t ENGINE_ID = 7;
    static constexpr std::int64_t ENGINE_INDEX = 1;
    static constexpr std::int64_t WORKSPACE_SIZE = 4096;
    // Base64 "AP8QgH8=" (exercises the one-byte-short final group).
    static constexpr std::array<std::uint8_t, 5> PAYLOAD = {0x00, 0xFF, 0x10, 0x80, 0x7F};
    static constexpr const char* PAYLOAD_BASE64 = "AP8QgH8=";

    std::vector<cudnnBackendDescriptor_t> _owned;
    cudnnHandle_t _otherHandle = reinterpret_cast<cudnnHandle_t>(0x23456789);

    std::mutex _recordMutex;
    hipdnnStatus_t _serializeStatus = HIPDNN_STATUS_SUCCESS;
    std::int64_t _deserializedEngineId = ENGINE_ID;
    int _serializeCalls = 0;
    int _deserializeCalls = 0;
    hipdnnHandle_t _deserializeHandle = nullptr;
    std::vector<std::uint8_t> _deserializedBytes;

    void SetUp() override
    {
        ShimV8MockBackendFixture::SetUp();
        lastError();
        installHeuristicResults({ENGINE_ID, -2});

        ON_CALL(*_mockBackend,
                backendGetAttribute(_, HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, _, _, _, _))
            .WillByDefault([](hipdnnBackendDescriptor_t,
                              hipdnnBackendAttributeName_t,
                              hipdnnBackendAttributeType_t,
                              std::int64_t,
                              std::int64_t* count,
                              void* elements) {
                *static_cast<std::int64_t*>(elements) = WORKSPACE_SIZE;
                if(count != nullptr)
                {
                    *count = 1;
                }
                return HIPDNN_STATUS_SUCCESS;
            });
        ON_CALL(
            *_mockBackend,
            backendGetAttribute(_, HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_GLOBAL_INDEX_EXT, _, _, _, _))
            .WillByDefault([this](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  std::int64_t,
                                  std::int64_t* count,
                                  void* elements) {
                const std::lock_guard<std::mutex> lock(_recordMutex);
                *static_cast<std::int64_t*>(elements) = _deserializedEngineId;
                if(count != nullptr)
                {
                    *count = 1;
                }
                return HIPDNN_STATUS_SUCCESS;
            });
        ON_CALL(*_mockBackend, backendGetSerializedExecutionPlanExt(_, _, _, _))
            .WillByDefault([this](hipdnnBackendDescriptor_t,
                                  std::size_t requested,
                                  std::size_t* size,
                                  std::uint8_t* bytes) {
                const std::lock_guard<std::mutex> lock(_recordMutex);
                ++_serializeCalls;
                if(_serializeStatus != HIPDNN_STATUS_SUCCESS)
                {
                    return _serializeStatus;
                }
                *size = PAYLOAD.size();
                if(bytes != nullptr && requested >= PAYLOAD.size())
                {
                    std::memcpy(bytes, PAYLOAD.data(), PAYLOAD.size());
                }
                return HIPDNN_STATUS_SUCCESS;
            });
        ON_CALL(*_mockBackend, backendCreateAndDeserializeExecutionPlanExt(_, _, _, _))
            .WillByDefault([this](hipdnnHandle_t handle,
                                  hipdnnBackendDescriptor_t* descriptor,
                                  const std::uint8_t* bytes,
                                  std::size_t size) {
                {
                    const std::lock_guard<std::mutex> lock(_recordMutex);
                    ++_deserializeCalls;
                    _deserializeHandle = handle;
                    _deserializedBytes.assign(bytes, bytes + size);
                }
                *descriptor = newMockDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
                return HIPDNN_STATUS_SUCCESS;
            });
    }

    void TearDown() override
    {
        hipdnn_frontend::detail::IHipdnnBackend::setInstance(_mockBackend);
        for(cudnnBackendDescriptor_t descriptor : _owned)
        {
            cudnnBackendDestroyDescriptor(descriptor);
        }
        _owned.clear();
        ShimV8MockBackendFixture::TearDown();
    }

    cudnnBackendDescriptor_t create(cudnnBackendDescriptorType_t type)
    {
        cudnnBackendDescriptor_t descriptor = nullptr;
        EXPECT_EQ(cudnnBackendCreateDescriptor(type, &descriptor), CUDNN_STATUS_SUCCESS);
        _owned.push_back(descriptor);
        return descriptor;
    }

    template <typename T>
    static void set(cudnnBackendDescriptor_t descriptor,
                    cudnnBackendAttributeName_t name,
                    cudnnBackendAttributeType_t type,
                    const T& value)
    {
        EXPECT_EQ(
            cudnnBackendSetAttribute(descriptor, name, type, 1, static_cast<const void*>(&value)),
            CUDNN_STATUS_SUCCESS)
            << lastError();
    }

    template <typename T>
    static void setArray(cudnnBackendDescriptor_t descriptor,
                         cudnnBackendAttributeName_t name,
                         cudnnBackendAttributeType_t type,
                         const std::vector<T>& values)
    {
        EXPECT_EQ(cudnnBackendSetAttribute(descriptor,
                                           name,
                                           type,
                                           static_cast<std::int64_t>(values.size()),
                                           static_cast<const void*>(values.data())),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
    }

    static void setJson(cudnnBackendDescriptor_t plan, const std::string& json)
    {
        // Upstream sends the buffer it read, NUL included.
        EXPECT_EQ(cudnnBackendSetAttribute(plan,
                                           CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                           CUDNN_TYPE_CHAR,
                                           static_cast<std::int64_t>(json.size() + 1),
                                           json.c_str()),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
    }

    static void finalize(cudnnBackendDescriptor_t descriptor)
    {
        EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_SUCCESS) << lastError();
    }

    cudnnBackendDescriptor_t tensor(std::int64_t uid)
    {
        const std::vector<std::int64_t> dims = {2, 3, 4, 5};
        const std::vector<std::int64_t> strides = {60, 20, 5, 1};
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        setArray(descriptor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, dims);
        setArray(descriptor, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, strides);
        set(descriptor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, uid);
        set(descriptor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, std::int64_t{16});
        finalize(descriptor);
        return descriptor;
    }

    // relu(x:1) -> y:2 on `_handle`.
    cudnnBackendDescriptor_t reluGraph()
    {
        cudnnBackendDescriptor_t relu = create(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        set(relu, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, CUDNN_POINTWISE_RELU_FWD);
        set(relu, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        finalize(relu);
        cudnnBackendDescriptor_t op = create(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
        set(op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_BACKEND_DESCRIPTOR, relu);
        set(op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, tensor(1));
        set(op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, tensor(2));
        finalize(op);
        cudnnBackendDescriptor_t graph = create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        set(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, _handle);
        setArray(graph,
                 CUDNN_ATTR_OPERATIONGRAPH_OPS,
                 CUDNN_TYPE_BACKEND_DESCRIPTOR,
                 std::vector<cudnnBackendDescriptor_t>{op});
        finalize(graph);
        return graph;
    }

    cudnnBackendDescriptor_t engineConfig()
    {
        cudnnBackendDescriptor_t engine = create(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
        set(engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, reluGraph());
        set(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, ENGINE_INDEX);
        finalize(engine);
        cudnnBackendDescriptor_t config = create(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        set(config, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, engine);
        finalize(config);
        return config;
    }

    // Finalized unless the caller wants to inspect the failure.
    cudnnBackendDescriptor_t plan(cudnnBackendDescriptor_t config)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, _handle);
        set(descriptor,
            CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            config);
        finalize(descriptor);
        return descriptor;
    }

    // Upstream loadFromJson order: JSON, then HANDLE, then finalize.
    cudnnStatus_t load(const std::string& json, cudnnBackendDescriptor_t& loaded)
    {
        loaded = create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        setJson(loaded, json);
        set(loaded, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, _handle);
        return cudnnBackendFinalize(loaded);
    }

    static std::string json(cudnnBackendDescriptor_t plan)
    {
        std::int64_t count = 0;
        EXPECT_EQ(cudnnBackendGetAttribute(plan,
                                           CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                           CUDNN_TYPE_CHAR,
                                           0,
                                           &count,
                                           nullptr),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
        std::string text(static_cast<std::size_t>(count), 'x');
        EXPECT_EQ(cudnnBackendGetAttribute(plan,
                                           CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                           CUDNN_TYPE_CHAR,
                                           count,
                                           &count,
                                           text.data()),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
        EXPECT_FALSE(text.empty());
        EXPECT_EQ(text.back(), '\0');
        text.resize(text.empty() ? 0 : text.size() - 1);
        return text;
    }

    static std::int64_t workspaceSize(cudnnBackendDescriptor_t plan)
    {
        std::int64_t size = -1;
        std::int64_t count = 0;
        EXPECT_EQ(
            cudnnBackendGetAttribute(
                plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &count, &size),
            CUDNN_STATUS_SUCCESS)
            << lastError();
        return size;
    }

    cudnnBackendDescriptor_t variantPack(void* workspace)
    {
        cudnnBackendDescriptor_t pack = create(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        setArray(pack,
                 CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                 CUDNN_TYPE_VOID_PTR,
                 std::vector<void*>{&_x, &_y});
        setArray(pack,
                 CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                 CUDNN_TYPE_INT64,
                 std::vector<std::int64_t>{1, 2});
        set(pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, workspace);
        finalize(pack);
        return pack;
    }

    // The graph's default (knob-free) hipDNN ENGINECFG for ENGINE_ID.
    hipdnnBackendDescriptor_t defaultHipdnnConfig() const
    {
        hipdnnBackendDescriptor_t found = nullptr;
        for(hipdnnBackendDescriptor_t config : created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR))
        {
            if(engineIdOf(config) == ENGINE_ID)
            {
                EXPECT_EQ(found, nullptr) << "more than one hipDNN ENGINECFG for the engine";
                found = config;
            }
        }
        return found;
    }

    static std::string expectedJson()
    {
        return R"({"version":1,"engine":{"globalIndex":1,"id":"7")";
    }

    int deserializeCalls()
    {
        const std::lock_guard<std::mutex> lock(_recordMutex);
        return _deserializeCalls;
    }

    float _x = 0.0F;
    float _y = 0.0F;
    std::array<std::byte, 64> _workspace{};
};

TEST_F(TestCudnnShimV8ExecutionPlan, PlanForwardsOnlyEngineConfig)
{
    plan(engineConfig());

    const std::vector<hipdnnBackendDescriptor_t> plans
        = created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    ASSERT_EQ(plans.size(), 1U);
    const std::vector<hipdnn_shim_test::CapturedSetAttribute> sets = setAttributesOf(plans[0]);
    ASSERT_EQ(sets.size(), 1U) << "HANDLE must not be forwarded";
    EXPECT_EQ(sets[0].name, HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG);
    EXPECT_EQ(sets[0].value<hipdnnBackendDescriptor_t>(), defaultHipdnnConfig());
    const std::vector<hipdnnBackendDescriptor_t> done = finalized();
    EXPECT_NE(std::find(done.begin(), done.end(), plans[0]), done.end());
}

TEST_F(TestCudnnShimV8ExecutionPlan, WorkspaceSizeReadFromHipdnnPlan)
{
    EXPECT_EQ(workspaceSize(plan(engineConfig())), WORKSPACE_SIZE);
}

TEST_F(TestCudnnShimV8ExecutionPlan, HandleOptionalForBuiltPlan)
{
    cudnnBackendDescriptor_t config = engineConfig();
    cudnnBackendDescriptor_t absent = create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    set(absent, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, config);
    finalize(absent);
    cudnnBackendDescriptor_t null = create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    set(null, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, cudnnHandle_t{nullptr});
    set(null, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, config);
    finalize(null);
    EXPECT_EQ(workspaceSize(absent), WORKSPACE_SIZE);

    // Both run on the graph's handle and on no other.
    cudnnBackendDescriptor_t pack = variantPack(_workspace.data());
    EXPECT_EQ(cudnnBackendExecute(_otherHandle, absent, pack), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(cudnnBackendExecute(_handle, absent, pack), CUDNN_STATUS_SUCCESS) << lastError();
    EXPECT_EQ(cudnnBackendExecute(_handle, null, pack), CUDNN_STATUS_SUCCESS) << lastError();
    const std::vector<hipdnn_shim_test::CapturedExecute> calls = executions();
    ASSERT_EQ(calls.size(), 2U);
    EXPECT_EQ(calls[0].handle, _handle);
    EXPECT_EQ(calls[1].handle, _handle);
}

TEST_F(TestCudnnShimV8ExecutionPlan, HandleRequiredForLoadedPlan)
{
    const std::string text = json(plan(engineConfig()));
    cudnnBackendDescriptor_t loaded = create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    setJson(loaded, text);
    EXPECT_EQ(cudnnBackendFinalize(loaded), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_ATTR_EXECUTION_PLAN_HANDLE"), std::string::npos);
    EXPECT_EQ(deserializeCalls(), 0);
}

TEST_F(TestCudnnShimV8ExecutionPlan, PlanHandleMismatchRejected)
{
    cudnnBackendDescriptor_t config = engineConfig();
    cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    set(descriptor, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, _otherHandle);
    set(descriptor, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, config);
    EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_ATTR_OPERATIONGRAPH_HANDLE"), std::string::npos);
    EXPECT_TRUE(created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR).empty());
}

TEST_F(TestCudnnShimV8ExecutionPlan, ConfigOrJsonRequiredExactlyOnce)
{
    cudnnBackendDescriptor_t neither = create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    set(neither, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, _handle);
    EXPECT_EQ(cudnnBackendFinalize(neither), CUDNN_STATUS_BAD_PARAM);

    cudnnBackendDescriptor_t config = engineConfig();
    const std::string text = json(plan(config));
    cudnnBackendDescriptor_t both = create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    set(both, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, _handle);
    set(both, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, config);
    setJson(both, text);
    EXPECT_EQ(cudnnBackendFinalize(both), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("exactly one"), std::string::npos);
}

TEST_F(TestCudnnShimV8ExecutionPlan, KernelCacheNotSupported)
{
    cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    cudnnBackendDescriptor_t other = tensor(9);
    EXPECT_EQ(cudnnBackendSetAttribute(descriptor,
                                       CUDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(&other)),
              CUDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestCudnnShimV8ExecutionPlan, JsonCarriesEngineAndPayload)
{
    const std::string text = json(plan(engineConfig()));

    EXPECT_EQ(text.rfind(expectedJson(), 0), 0U) << text;
    EXPECT_NE(text.find(R"("behaviorNotes":[)"), std::string::npos) << text;
    EXPECT_NE(text.find(R"(},"knobs":[],"payload":")"), std::string::npos) << text;
    const std::string tail = std::string(R"("payload":")") + PAYLOAD_BASE64 + "\"}";
    ASSERT_GE(text.size(), tail.size());
    EXPECT_EQ(text.substr(text.size() - tail.size()), tail) << text;
    // Count query plus fill reuse one serialization (two-call protocol).
    const std::lock_guard<std::mutex> lock(_recordMutex);
    EXPECT_EQ(_serializeCalls, 2);
}

TEST_F(TestCudnnShimV8ExecutionPlan, JsonWithoutPayloadWhenSerializeFails)
{
    _serializeStatus = HIPDNN_STATUS_NOT_SUPPORTED;
    const std::string text = json(plan(engineConfig()));

    EXPECT_EQ(text.rfind(expectedJson(), 0), 0U) << text;
    EXPECT_EQ(text.find("payload"), std::string::npos) << text;
    EXPECT_EQ(text.back(), '}');

    cudnnBackendDescriptor_t loaded = nullptr;
    EXPECT_EQ(load(text, loaded), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_NE(lastError().find("payload"), std::string::npos);
    EXPECT_EQ(deserializeCalls(), 0);
}

TEST_F(TestCudnnShimV8ExecutionPlan, JsonLoadRoundTrip)
{
    const std::string text = json(plan(engineConfig()));

    cudnnBackendDescriptor_t loaded = nullptr;
    ASSERT_EQ(load(text, loaded), CUDNN_STATUS_SUCCESS) << lastError();
    {
        const std::lock_guard<std::mutex> lock(_recordMutex);
        EXPECT_EQ(_deserializeCalls, 1);
        EXPECT_EQ(_deserializeHandle, _handle);
        EXPECT_EQ(_deserializedBytes, std::vector<std::uint8_t>(PAYLOAD.begin(), PAYLOAD.end()));
    }
    EXPECT_EQ(workspaceSize(loaded), WORKSPACE_SIZE);
    EXPECT_EQ(json(loaded), text);

    // Upstream reads ENGINE_CONFIG -> ENGINE -> GLOBAL_INDEX right after loading.
    cudnnBackendDescriptor_t config = create(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    std::int64_t count = 0;
    ASSERT_EQ(cudnnBackendGetAttribute(loaded,
                                       CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&config)),
              CUDNN_STATUS_SUCCESS)
        << lastError();
    cudnnBackendDescriptor_t engine = create(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    ASSERT_EQ(cudnnBackendGetAttribute(config,
                                       CUDNN_ATTR_ENGINECFG_ENGINE,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&engine)),
              CUDNN_STATUS_SUCCESS)
        << lastError();
    std::int64_t index = -1;
    EXPECT_EQ(cudnnBackendGetAttribute(
                  engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &count, &index),
              CUDNN_STATUS_SUCCESS)
        << lastError();
    EXPECT_EQ(index, ENGINE_INDEX);

    // A loaded config has no graph to build a new plan from.
    cudnnBackendDescriptor_t rebuilt = create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    set(rebuilt, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, _handle);
    set(rebuilt, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, config);
    EXPECT_EQ(cudnnBackendFinalize(rebuilt), CUDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestCudnnShimV8ExecutionPlan, JsonReaderAcceptsWhitespaceAndKeyOrder)
{
    const std::string text = " {\n \"payload\" : \"AP8QgH8=\" , \"knobs\" : [ { \"value\" : 16 , "
                             "\"type\" : 3 } ] ,\t\"engine\" : { \"id\" : \"7\" , "
                             "\"behaviorNotes\" : [ 3 ] , \"name\" : \"a\\\"b\\\\c\\u0041\" , "
                             "\"globalIndex\" : 1 } , \"version\" : 1 }\r\n";
    cudnnBackendDescriptor_t loaded = nullptr;
    ASSERT_EQ(load(text, loaded), CUDNN_STATUS_SUCCESS) << lastError();
    EXPECT_EQ(json(loaded),
              R"({"version":1,"engine":{"globalIndex":1,"id":"7","name":"a\"b\\cA",)"
              R"("behaviorNotes":[3]},"knobs":[{"type":3,"value":16}],"payload":"AP8QgH8="})");
}

TEST_F(TestCudnnShimV8ExecutionPlan, JsonMalformedRejected)
{
    const std::string engine = R"("engine":{"globalIndex":1,"id":"7"})";
    const std::string payload = R"("payload":"AP8QgH8=")";
    const std::string version = R"({"version":1,)";
    const std::vector<std::string> malformed = {
        "",
        "not json",
        R"({"version":2,)" + engine + "," + payload + "}",
        R"({"version":1.0,)" + engine + "," + payload + "}",
        version + R"("engine":{"globalIndex":1,"id":7},)" + payload + "}",
        version + R"("engine":{"globalIndex":1,"id":"99999999999999999999"},)" + payload + "}",
        version + R"("engine":{"globalIndex":-1,"id":"7"},)" + payload + "}",
        version + R"("engine":{"id":"7"},)" + payload + "}",
        version + payload + "}",
        version + engine + R"(,"payload":"AP8Q gH8="})",
        version + engine + R"(,"payload":"AP8QgH8"})",
        version + engine + R"(,"payload":""})",
        version + engine + R"(,"payload":"A==="})",
        version + engine + R"(,"payload":"===="})",
        version + engine + R"(,"payload":"AB=C"})",
        version + engine + R"(,"knobs":[{"type":999,"value":1}],)" + payload + "}",
        version + engine + R"(,"extra":1,)" + payload + "}",
        version + R"("version":1,)" + engine + "," + payload + "}",
        version + engine + "," + payload + "} x",
        version + engine + "," + payload,
    };
    for(const std::string& text : malformed)
    {
        cudnnBackendDescriptor_t loaded = nullptr;
        EXPECT_EQ(load(text, loaded), CUDNN_STATUS_BAD_PARAM) << text;
        EXPECT_NE(lastError().find("CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION"),
                  std::string::npos)
            << text;
    }
    EXPECT_EQ(deserializeCalls(), 0);
}

TEST_F(TestCudnnShimV8ExecutionPlan, JsonEngineIdMismatchRejected)
{
    const std::string text = json(plan(engineConfig()));
    {
        const std::lock_guard<std::mutex> lock(_recordMutex);
        _deserializedEngineId = 8;
    }
    cudnnBackendDescriptor_t loaded = nullptr;
    EXPECT_EQ(load(text, loaded), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("does not match engine 8"), std::string::npos);
}

TEST_F(TestCudnnShimV8ExecutionPlan, ExtremeEngineIdsLoad)
{
    for(const std::int64_t id : {std::numeric_limits<std::int64_t>::min(),
                                 std::numeric_limits<std::int64_t>::max(),
                                 std::int64_t{0x637C0BB90F065BA2}})
    {
        {
            const std::lock_guard<std::mutex> lock(_recordMutex);
            _deserializedEngineId = id;
        }
        const std::string idField = R"("id":")" + std::to_string(id) + "\"";
        cudnnBackendDescriptor_t loaded = nullptr;
        ASSERT_EQ(load(R"({"version":1,"engine":{"globalIndex":0,)" + idField
                           + R"(},"payload":"AP8QgH8="})",
                       loaded),
                  CUDNN_STATUS_SUCCESS)
            << id << ": " << lastError();
        EXPECT_NE(json(loaded).find(idField), std::string::npos) << id;
    }
}

TEST_F(TestCudnnShimV8ExecutionPlan, Base64PayloadsRoundTrip)
{
    const std::vector<std::pair<std::vector<std::uint8_t>, std::string>> cases = {
        {{0}, "AA=="},
        {{0, 1}, "AAE="},
        {{0, 1, 2}, "AAEC"},
        {{0, 1, 2, 3}, "AAECAw=="},
        {{0x00, 0xFF, 0x10, 0x80, 0x7F}, "AP8QgH8="},
    };
    for(const auto& [bytes, encoded] : cases)
    {
        backend::plan_json::Plan plan;
        plan.id = ENGINE_ID;
        plan.payload = bytes;
        const std::string text = backend::plan_json::write(plan, true);
        EXPECT_NE(text.find(R"("payload":")" + encoded + "\""), std::string::npos) << text;
        const std::optional<backend::plan_json::Plan> parsed = backend::plan_json::read(text);
        ASSERT_TRUE(parsed.has_value()) << text;
        EXPECT_EQ(parsed->payload, bytes) << encoded;
    }
}

TEST_F(TestCudnnShimV8ExecutionPlan, ControlCharactersInNameRoundTrip)
{
    installEngineName(ENGINE_ID,
                      "a\nb\x01"
                      "c");
    const std::string text = json(plan(engineConfig()));
    EXPECT_NE(text.find(R"("name":"a\nb\u0001c")"), std::string::npos) << text;

    cudnnBackendDescriptor_t loaded = nullptr;
    ASSERT_EQ(load(text, loaded), CUDNN_STATUS_SUCCESS) << lastError();
    EXPECT_EQ(json(loaded), text);
}

TEST_F(TestCudnnShimV8ExecutionPlan, VariantPackBuiltAtFinalize)
{
    void* workspace = _workspace.data();
    variantPack(workspace);

    const std::vector<hipdnnBackendDescriptor_t> packs
        = created(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    ASSERT_EQ(packs.size(), 1U);
    const std::vector<hipdnn_shim_test::CapturedSetAttribute> sets = setAttributesOf(packs[0]);
    ASSERT_EQ(sets.size(), 3U);
    EXPECT_EQ(sets[0].name, HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS);
    EXPECT_EQ(sets[0].values<void*>(), (std::vector<void*>{&_x, &_y}));
    EXPECT_EQ(sets[1].name, HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS);
    EXPECT_EQ(sets[1].values<std::int64_t>(), (std::vector<std::int64_t>{1, 2}));
    EXPECT_EQ(sets[2].name, HIPDNN_ATTR_VARIANT_PACK_WORKSPACE);
    EXPECT_EQ(sets[2].value<void*>(), workspace);
    const std::vector<hipdnnBackendDescriptor_t> done = finalized();
    EXPECT_NE(std::find(done.begin(), done.end(), packs[0]), done.end());
}

TEST_F(TestCudnnShimV8ExecutionPlan, VariantPackCountMismatchRejected)
{
    cudnnBackendDescriptor_t pack = create(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    setArray(
        pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, std::vector<void*>{&_x});
    setArray(pack,
             CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
             CUDNN_TYPE_INT64,
             std::vector<std::int64_t>{1, 2});
    EXPECT_EQ(cudnnBackendFinalize(pack), CUDNN_STATUS_BAD_PARAM);
    EXPECT_TRUE(created(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR).empty());
}

TEST_F(TestCudnnShimV8ExecutionPlan, VariantPackRequiresPointersAndIds)
{
    cudnnBackendDescriptor_t unset = create(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    EXPECT_EQ(cudnnBackendFinalize(unset), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS"), std::string::npos);

    cudnnBackendDescriptor_t noIds = create(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    setArray(
        noIds, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, std::vector<void*>{&_x});
    EXPECT_EQ(cudnnBackendFinalize(noIds), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS"), std::string::npos);

    cudnnBackendDescriptor_t empty = create(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    setArray(
        empty, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, std::vector<void*>{});
    setArray(
        empty, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, std::vector<std::int64_t>{});
    EXPECT_EQ(cudnnBackendFinalize(empty), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS"), std::string::npos);

    EXPECT_TRUE(created(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR).empty());
}

TEST_F(TestCudnnShimV8ExecutionPlan, ExecuteIsOneBackendCall)
{
    cudnnBackendDescriptor_t built = plan(engineConfig());
    cudnnBackendDescriptor_t pack = variantPack(_workspace.data());
    auto* const hipdnnPlan = created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR)[0];
    auto* const hipdnnPack = created(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR)[0];

    // Any other backend call fails the strict mock.
    auto strict = std::make_shared<::testing::StrictMock<Mock_hipdnn_backend>>();
    EXPECT_CALL(*strict, backendExecute(_handle, hipdnnPlan, hipdnnPack))
        .WillOnce(::testing::Return(HIPDNN_STATUS_SUCCESS));
    hipdnn_frontend::detail::IHipdnnBackend::setInstance(strict);
    const cudnnStatus_t status = cudnnBackendExecute(_handle, built, pack);
    hipdnn_frontend::detail::IHipdnnBackend::setInstance(_mockBackend);

    EXPECT_EQ(status, CUDNN_STATUS_SUCCESS) << lastError();
}

TEST_F(TestCudnnShimV8ExecutionPlan, ExecuteHandleMismatchRejected)
{
    cudnnBackendDescriptor_t built = plan(engineConfig());
    cudnnBackendDescriptor_t pack = variantPack(_workspace.data());
    EXPECT_EQ(cudnnBackendExecute(_otherHandle, built, pack), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("handle"), std::string::npos);
    EXPECT_EQ(cudnnBackendExecute(nullptr, built, pack), CUDNN_STATUS_BAD_PARAM);
    EXPECT_TRUE(executions().empty());
}

TEST_F(TestCudnnShimV8ExecutionPlan, ExecuteRejectsWrongOrUnfinalizedDescriptors)
{
    cudnnBackendDescriptor_t built = plan(engineConfig());
    cudnnBackendDescriptor_t pack = variantPack(_workspace.data());
    cudnnBackendDescriptor_t unfinalized = create(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);

    // NOLINTNEXTLINE(readability-suspicious-call-argument): the swap is the point.
    EXPECT_EQ(cudnnBackendExecute(_handle, pack, built), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(cudnnBackendExecute(_handle, built, unfinalized), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("not finalized"), std::string::npos);
    EXPECT_EQ(cudnnBackendExecute(_handle, built, nullptr), CUDNN_STATUS_BAD_PARAM);
    EXPECT_TRUE(executions().empty());
}

TEST_F(TestCudnnShimV8ExecutionPlan, ExecuteFailureKeepsStatus)
{
    cudnnBackendDescriptor_t built = plan(engineConfig());
    cudnnBackendDescriptor_t pack = variantPack(_workspace.data());
    ON_CALL(*_mockBackend, backendExecute(_, _, _))
        .WillByDefault(::testing::Return(HIPDNN_STATUS_EXECUTION_FAILED));
    EXPECT_EQ(cudnnBackendExecute(_handle, built, pack), CUDNN_STATUS_EXECUTION_FAILED);
    EXPECT_NE(lastError().find("executing the hipDNN plan"), std::string::npos);
}

TEST_F(TestCudnnShimV8ExecutionPlan, LoadedPlanExecutesOnItsHandle)
{
    const std::string text = json(plan(engineConfig()));
    cudnnBackendDescriptor_t loaded = nullptr;
    ASSERT_EQ(load(text, loaded), CUDNN_STATUS_SUCCESS) << lastError();
    cudnnBackendDescriptor_t pack = variantPack(_workspace.data());

    EXPECT_EQ(cudnnBackendExecute(_otherHandle, loaded, pack), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(cudnnBackendExecute(_handle, loaded, pack), CUDNN_STATUS_SUCCESS) << lastError();
    const std::vector<hipdnn_shim_test::CapturedExecute> calls = executions();
    ASSERT_EQ(calls.size(), 1U);
    EXPECT_EQ(calls[0].plan, created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR).back());
}

TEST_F(TestCudnnShimV8ExecutionPlan, ExecuteFromTwoThreads)
{
    constexpr int RUNS = 200;
    cudnnBackendDescriptor_t built = plan(engineConfig());
    cudnnBackendDescriptor_t pack = variantPack(_workspace.data());
    std::atomic<int> failures{0};
    const auto run = [&] {
        for(int i = 0; i < RUNS; ++i)
        {
            if(cudnnBackendExecute(_handle, built, pack) != CUDNN_STATUS_SUCCESS)
            {
                ++failures;
            }
        }
    };
    std::thread first(run);
    std::thread second(run);
    first.join();
    second.join();

    EXPECT_EQ(failures.load(), 0);
    const std::vector<hipdnn_shim_test::CapturedExecute> calls = executions();
    ASSERT_EQ(calls.size(), static_cast<std::size_t>(2 * RUNS));
    auto* const hipdnnPlan = created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR)[0];
    for(const hipdnn_shim_test::CapturedExecute& call : calls)
    {
        EXPECT_EQ(call.plan, hipdnnPlan);
        EXPECT_EQ(call.handle, _handle);
    }
}

// Knob-free configs share the graph's cached hipDNN ENGINECFG, so concurrent
// plan builds hand the same hipDNN descriptor to two plan finalizes.
TEST_F(TestCudnnShimV8ExecutionPlan, PlansFromTwoThreadsShareDefaultConfig)
{
    constexpr int PLANS = 20;
    cudnnBackendDescriptor_t config = engineConfig();
    std::atomic<int> failures{0};
    const auto build = [&] {
        for(int i = 0; i < PLANS; ++i)
        {
            cudnnBackendDescriptor_t descriptor = nullptr;
            bool ok
                = cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &descriptor)
                  == CUDNN_STATUS_SUCCESS;
            ok = ok
                 && cudnnBackendSetAttribute(descriptor,
                                             CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
                                             CUDNN_TYPE_HANDLE,
                                             1,
                                             static_cast<const void*>(&_handle))
                        == CUDNN_STATUS_SUCCESS;
            ok = ok
                 && cudnnBackendSetAttribute(descriptor,
                                             CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                             CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                             1,
                                             static_cast<const void*>(&config))
                        == CUDNN_STATUS_SUCCESS;
            ok = ok && cudnnBackendFinalize(descriptor) == CUDNN_STATUS_SUCCESS;
            if(!ok)
            {
                ++failures;
            }
            cudnnBackendDestroyDescriptor(descriptor);
        }
    };
    std::thread first(build);
    std::thread second(build);
    first.join();
    second.join();

    EXPECT_EQ(failures.load(), 0);
    auto* const shared = defaultHipdnnConfig();
    const std::vector<hipdnn_shim_test::CapturedSetAttribute> sets
        = setAttributesOn(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    ASSERT_EQ(sets.size(), static_cast<std::size_t>(2 * PLANS));
    for(const hipdnn_shim_test::CapturedSetAttribute& call : sets)
    {
        EXPECT_EQ(call.name, HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG);
        EXPECT_EQ(call.value<hipdnnBackendDescriptor_t>(), shared);
    }
}

} // namespace

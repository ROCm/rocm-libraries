// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/PluginApi.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

#include "core/Container.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/PackedPlanTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvFwdPackedCase.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwisePackedCase.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

/**
 * @file TestPluginSerialization.cpp
 * @brief The plan serialization hooks, called through the C API: the null checks, the
 *        refusals of a save and a load, and for each packed set a kpack plan saved,
 *        loaded without its archive and executed.
 *
 * Every case that reads a packed archive reads a copy of the packed set in a scratch
 * directory. Other suites corrupt or delete the staged archives.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using hip_kernel_provider::core::Container;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::ScopedDirectory;

namespace packs = hip_kernel_provider::kernel_ingestor_engine::testing;

using packs::NO_WINNER_PHRASE;
using packs::NOT_AN_INGESTOR_ENGINE_PHRASE;
using packs::runPlan;

constexpr const char* SCRATCH_LABEL = "pluginserialization";

// A handle whose container loads the engines of the unit descriptor root.
std::unique_ptr<Handle> makeHandleWithContainer()
{
    auto handle = std::make_unique<Handle>();
    handle->container = std::make_shared<Container>();
    return handle;
}

std::string lastPluginError()
{
    const char* message = nullptr;
    hipdnnPluginGetLastErrorString(&message);
    return message == nullptr ? std::string() : std::string(message);
}

// Checks that a hook returned `expected`, and that the last plugin error starts with
// `prefix` and holds `phrase`. Returns the last plugin error.
std::string expectHookRefusal(hipdnnPluginStatus_t status,
                              hipdnnPluginStatus_t expected,
                              std::string_view prefix,
                              const std::string& phrase)
{
    const std::string message = lastPluginError();
    EXPECT_EQ(status, expected) << message;
    EXPECT_EQ(message.rfind(prefix, 0), 0U) << message;
    EXPECT_NE(message.find(phrase), std::string::npos) << message;
    return message;
}

hipdnnPluginStatus_t save(Handle& handle, Context& context, hipdnnPluginConstData_t& serialized)
{
    return hipdnnEnginePluginSerializeExecutionContext(&handle, &context, &serialized);
}

// ---------------------------------------------------------------------------
// Refusals that need no device
// ---------------------------------------------------------------------------

TEST(TestPluginSerialization, HooksRejectNullArguments)
{
    Handle handle;
    Context context;
    const uint8_t byte = 0;
    hipdnnPluginConstData_t serialized{&byte, 1};
    const hipdnnPluginConstData_t withoutBytes{nullptr, 0};
    hipdnnEnginePluginExecutionContext_t created = nullptr;

    EXPECT_EQ(hipdnnEnginePluginSerializeExecutionContext(nullptr, &context, &serialized),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(hipdnnEnginePluginSerializeExecutionContext(&handle, nullptr, &serialized),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(hipdnnEnginePluginSerializeExecutionContext(&handle, &context, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginDestroySerializedExecutionContext(nullptr, &serialized),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(hipdnnEnginePluginDestroySerializedExecutionContext(&handle, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(
        hipdnnEnginePluginCreateExecutionContextFromSerialized(nullptr, &serialized, &created),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(hipdnnEnginePluginCreateExecutionContextFromSerialized(&handle, nullptr, &created),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(
        hipdnnEnginePluginCreateExecutionContextFromSerialized(&handle, &withoutBytes, &created),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(hipdnnEnginePluginCreateExecutionContextFromSerialized(&handle, &serialized, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(serialized.ptr, static_cast<const void*>(&byte));
    EXPECT_EQ(serialized.size, 1U);
    EXPECT_EQ(created, nullptr);
}

// The backend passes the structure as the save hook filled it in, so a null buffer can reach
// the destroy hook.
TEST(TestPluginSerialization, DestroyAcceptsTheEmptyStructureTheBackendCanPass)
{
    Handle handle;

    hipdnnPluginConstData_t empty{nullptr, 0};
    EXPECT_EQ(hipdnnEnginePluginDestroySerializedExecutionContext(&handle, &empty),
              HIPDNN_PLUGIN_STATUS_SUCCESS)
        << lastPluginError();
    EXPECT_EQ(empty.ptr, nullptr);
    EXPECT_EQ(empty.size, 0U);

    hipdnnPluginConstData_t sizeWithoutBytes{nullptr, 4};
    EXPECT_EQ(hipdnnEnginePluginDestroySerializedExecutionContext(&handle, &sizeWithoutBytes),
              HIPDNN_PLUGIN_STATUS_SUCCESS)
        << lastPluginError();
    EXPECT_EQ(sizeWithoutBytes.ptr, nullptr);
    EXPECT_EQ(sizeWithoutBytes.size, 0U);
}

TEST(TestPluginSerialization, SaveRefusesAPlanNoIngestorEngineBuilt)
{
    const auto handle = makeHandleWithContainer();
    Context context;
    context.setPlan(std::make_unique<packs::NonIngestorPlan>());

    const uint8_t byte = 0;
    hipdnnPluginConstData_t serialized{&byte, 7};
    const std::string message
        = expectHookRefusal(save(*handle, context, serialized),
                            HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                            serialization::INGESTOR_PLAN_SAVE_INCOMPATIBLE_PREFIX,
                            NOT_AN_INGESTOR_ENGINE_PHRASE);
    EXPECT_EQ(message.find(NO_WINNER_PHRASE), std::string::npos) << message;
    EXPECT_EQ(serialized.ptr, static_cast<const void*>(&byte));
    EXPECT_EQ(serialized.size, 7U);
}

TEST(TestPluginSerialization, SaveRefusesAContextWithoutAPlan)
{
    const auto handle = makeHandleWithContainer();
    Context context;
    ASSERT_FALSE(context.hasValidPlan());

    const uint8_t byte = 0;
    hipdnnPluginConstData_t serialized{&byte, 7};
    const std::string message
        = expectHookRefusal(save(*handle, context, serialized),
                            HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                            serialization::INGESTOR_PLAN_SAVE_INCOMPATIBLE_PREFIX,
                            NOT_AN_INGESTOR_ENGINE_PHRASE);
    EXPECT_EQ(message.find(NO_WINNER_PHRASE), std::string::npos) << message;
    EXPECT_EQ(serialized.ptr, static_cast<const void*>(&byte));
    EXPECT_EQ(serialized.size, 7U);
}

TEST(TestPluginSerialization, LoadRefusesDamagedBytesAndCreatesNoContext)
{
    // The header checks run before any device query.
    Handle handle;
    const std::vector<uint8_t> zeros(64, 0);
    const hipdnnPluginConstData_t serialized{zeros.data(), zeros.size()};
    Context sentinel;
    hipdnnEnginePluginExecutionContext_t created = &sentinel;

    expectHookRefusal(
        hipdnnEnginePluginCreateExecutionContextFromSerialized(&handle, &serialized, &created),
        HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
        serialization::INGESTOR_PLAN_DAMAGED_PREFIX,
        "not an ingestor plan");
    EXPECT_EQ(created, &sentinel);
}

// ---------------------------------------------------------------------------
// A kpack plan saved, loaded and executed through the hooks
// ---------------------------------------------------------------------------

// Holds the bytes of a save. Frees them through the destroy hook when the test leaves
// before it calls destroy().
class ScopedSerializedContext
{
public:
    explicit ScopedSerializedContext(Handle& handle)
        : _handle(&handle)
    {
    }

    ~ScopedSerializedContext()
    {
        if(!_destroyed && _data.ptr != nullptr)
        {
            static_cast<void>(hipdnnEnginePluginDestroySerializedExecutionContext(_handle, &_data));
        }
    }

    ScopedSerializedContext(const ScopedSerializedContext&) = delete;
    ScopedSerializedContext& operator=(const ScopedSerializedContext&) = delete;
    ScopedSerializedContext(ScopedSerializedContext&&) = delete;
    ScopedSerializedContext& operator=(ScopedSerializedContext&&) = delete;

    hipdnnPluginConstData_t& data()
    {
        return _data;
    }

    // Calls the destroy hook once. The destructor then calls nothing.
    hipdnnPluginStatus_t destroy()
    {
        _destroyed = true;
        return hipdnnEnginePluginDestroySerializedExecutionContext(_handle, &_data);
    }

private:
    Handle* _handle;
    hipdnnPluginConstData_t _data{};
    bool _destroyed = false;
};

// Holds an execution context that a hook created, and destroys it through the plugin API.
class ScopedExecutionContext
{
public:
    explicit ScopedExecutionContext(Handle& handle)
        : _handle(&handle)
    {
    }

    ~ScopedExecutionContext()
    {
        if(_context != nullptr)
        {
            static_cast<void>(hipdnnEnginePluginDestroyExecutionContext(_handle, _context));
        }
    }

    ScopedExecutionContext(const ScopedExecutionContext&) = delete;
    ScopedExecutionContext& operator=(const ScopedExecutionContext&) = delete;
    ScopedExecutionContext(ScopedExecutionContext&&) = delete;
    ScopedExecutionContext& operator=(ScopedExecutionContext&&) = delete;

    hipdnnEnginePluginExecutionContext_t& context()
    {
        return _context;
    }

    const hipdnn_plugin_sdk::IPlan<Handle>& plan() const
    {
        return static_cast<const Context*>(_context)->plan();
    }

private:
    Handle* _handle;
    hipdnnEnginePluginExecutionContext_t _context = nullptr;
};

class TestPluginSerializationKpack : public ::testing::TestWithParam<const packs::PackedPlanCase*>
{
};

TEST_P(TestPluginSerializationKpack, RoundTripsAKpackPlanThroughTheHooks)
{
    SKIP_IF_NO_DEVICES();
    const packs::PackedPlanCase& packedCase = *GetParam();

    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    packs::PackedPlan built;
    std::unique_ptr<Handle> saveHandle;
    std::vector<uint8_t> original;
    size_t originalWorkspaceBytes = 0;
    {
        SCOPED_TRACE("original run");
        ASSERT_NO_FATAL_FAILURE(
            packs::buildPackedPlan(packedCase, packedCase.roundTripGraph(), scratch.path(), built));
        if(built.plan == nullptr)
        {
            GTEST_SKIP() << "nothing was packed for this device ("
                         << built.deviceProperties.gcnArchName << ")";
        }
        packedCase.fillInputs(built.tensors);

        saveHandle = makeHandleWithContainer();
        ASSERT_NO_FATAL_FAILURE(runPlan(*built.plan, *saveHandle, built.tensors, original));
        packedCase.expectExactOutput(original);
        originalWorkspaceBytes = built.plan->getWorkspaceSize(*saveHandle);
    }

    auto context = std::make_unique<Context>();
    context->setPlan(std::move(built.plan));

    const auto loadHandle = makeHandleWithContainer();
    ScopedSerializedContext serialized(*loadHandle);
    {
        SCOPED_TRACE("save");
        ASSERT_EQ(save(*saveHandle, *context, serialized.data()), HIPDNN_PLUGIN_STATUS_SUCCESS)
            << lastPluginError();
        ASSERT_NE(serialized.data().ptr, nullptr);
        ASSERT_GT(serialized.data().size, 0U);
    }

    ScopedExecutionContext restored(*loadHandle);
    size_t cachedBefore = 0;
    {
        SCOPED_TRACE("load without the archive");
        // Destroy the original plan and drop the cached modules and kept archives. Then remove
        // the archive. The load can use no code object of the original plan.
        context.reset();
        resetIngestorModuleCachesForTesting();
        const std::filesystem::path archive = built.archive();
        ASSERT_TRUE(std::filesystem::remove(archive)) << archive;
        cachedBefore = packs::packModuleCacheEntries();

        ASSERT_EQ(hipdnnEnginePluginCreateExecutionContextFromSerialized(
                      loadHandle.get(), &serialized.data(), &restored.context()),
                  HIPDNN_PLUGIN_STATUS_SUCCESS)
            << lastPluginError();
        ASSERT_NE(restored.context(), nullptr);
        EXPECT_EQ(restored.plan().getWorkspaceSize(*loadHandle), originalWorkspaceBytes);
    }

    {
        SCOPED_TRACE("restored runs");
        for(int run = 0; run < 2; ++run)
        {
            std::vector<uint8_t> output;
            ASSERT_NO_FATAL_FAILURE(runPlan(restored.plan(), *loadHandle, built.tensors, output));
            EXPECT_EQ(output, original) << "run " << run;
            packedCase.expectExactOutput(output);
        }
        EXPECT_EQ(packs::packModuleCacheEntries(), cachedBefore);
    }

    {
        SCOPED_TRACE("re-save of the restored plan");
        // A restored plan cannot be saved again.
        const uint8_t byte = 0;
        hipdnnPluginConstData_t resaved{&byte, 7};
        const std::string message
            = expectHookRefusal(hipdnnEnginePluginSerializeExecutionContext(
                                    loadHandle.get(), restored.context(), &resaved),
                                HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                                serialization::INGESTOR_PLAN_SAVE_INCOMPATIBLE_PREFIX,
                                "loaded from a saved payload");
        EXPECT_EQ(message.find(NOT_AN_INGESTOR_ENGINE_PHRASE), std::string::npos) << message;
        EXPECT_EQ(resaved.ptr, static_cast<const void*>(&byte));
        EXPECT_EQ(resaved.size, 7U);
    }

    {
        SCOPED_TRACE("destroy of the saved bytes");
        ASSERT_EQ(serialized.destroy(), HIPDNN_PLUGIN_STATUS_SUCCESS) << lastPluginError();
        EXPECT_EQ(serialized.data().ptr, nullptr);
        EXPECT_EQ(serialized.data().size, 0U);
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         TestPluginSerializationKpack,
                         ::testing::Values(&packs::POINTWISE_PACKED_PLAN_CASE,
                                           &packs::CONV_FWD_PACKED_PLAN_CASE),
                         [](const ::testing::TestParamInfo<const packs::PackedPlanCase*>& info) {
                             return std::string(info.param->name);
                         });

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/engine_knob_catalog_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_plugin_sdk/KnobFactory.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

namespace
{

namespace constants = hipdnn_tests::plugin_constants;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

// One catalog entry: a tile size and a mode, the engine's only two engine-specific knobs.
struct CatalogEntry
{
    int64_t tileSize;
    const char* mode;
};

// The catalog the plugin serves by default, in preference order.
constexpr std::array<CatalogEntry, 3> K_CATALOG{{{16, "fast"}, {32, "fast"}, {64, "accurate"}}};

// Value outside the tile-size knob's valid values, used by the bad-value mode.
constexpr int64_t K_UNSUPPORTED_TILE_SIZE = 48;

// Selects the catalog the plugin serves. Read on every call so a test can switch modes
// between calls with a scoped environment variable.
std::string catalogMode()
{
    return hipdnn_data_sdk::utilities::getEnv(constants::K_KNOB_CATALOG_PLUGIN_MODE_ENV,
                                              constants::K_KNOB_CATALOG_MODE_VALID);
}

std::unique_ptr<data_objects::KnobSettingT> makeIntSetting(const char* knobId, int64_t value)
{
    auto setting = std::make_unique<data_objects::KnobSettingT>();
    setting->knob_id = knobId;
    data_objects::IntValueT intValue;
    intValue.value = value;
    setting->value.Set(intValue);
    return setting;
}

std::unique_ptr<data_objects::KnobSettingT> makeStringSetting(const char* knobId,
                                                              const std::string& value)
{
    auto setting = std::make_unique<data_objects::KnobSettingT>();
    setting->knob_id = knobId;
    data_objects::StringValueT stringValue;
    stringValue.value = value;
    setting->value.Set(std::move(stringValue));
    return setting;
}

} // namespace

class KnobCatalogPlugin : public TestPluginBase
{
public:
    const char* getPluginName() const override
    {
        return HIPDNN_COMPONENT_NAME;
    }

    const char* getPluginVersion() const override
    {
        return "1.0.0";
    }

    const char* getPluginApiVersion() const override
    {
        return apiVersionWithoutTweak();
    }

    int64_t getEngineId() const override
    {
        return constants::engineId<KnobCatalogPlugin>();
    }

    uint32_t getNumEngines() const override
    {
        return 1;
    }

    uint32_t getNumApplicableEngines() const override
    {
        return 1;
    }

    // Two engine-specific knobs plus one global knob. The global knob is advertised but
    // never appears in a catalog entry.
    std::vector<flatbuffers::Offset<data_objects::Knob>>
        createKnobs(flatbuffers::FlatBufferBuilder& builder) const override
    {
        return {
            hipdnn_plugin_sdk::KnobFactory::createIntKnob(builder,
                                                          constants::K_KNOB_CATALOG_TILE_KNOB,
                                                          "Tile size",
                                                          16,
                                                          0,
                                                          0,
                                                          1,
                                                          {16, 32, 64}),
            hipdnn_plugin_sdk::KnobFactory::createStringKnob(builder,
                                                             constants::K_KNOB_CATALOG_MODE_KNOB,
                                                             "Accuracy mode",
                                                             "fast",
                                                             {"fast", "accurate"}),
            hipdnn_plugin_sdk::KnobFactory::createIntKnob(
                builder, "global.benchmarking", "Enable benchmarking", 0, 0, 1, 1, {})};
    }

    static hipdnnPluginStatus_t getEngineKnobCatalog(hipdnnEnginePluginHandle_t handle,
                                                     int64_t engineId,
                                                     const hipdnnPluginConstData_t* opGraph,
                                                     hipdnnPluginConstData_t* knobCatalog)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle) << ", engineId=" << engineId
                                << ", opGraph=" << static_cast<const void*>(opGraph)
                                << ", knobCatalog=" << static_cast<void*>(knobCatalog));

        const auto mode = catalogMode();
        if(mode == constants::K_KNOB_CATALOG_MODE_NOT_APPLICABLE)
        {
            return HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE;
        }

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(opGraph);
            hipdnn_plugin_sdk::throwIfNull(knobCatalog);

            data_objects::EngineKnobCatalogT catalog;
            catalog.engine_id = engineId;
            for(const auto& entry : K_CATALOG)
            {
                auto config = std::make_unique<data_objects::EngineConfigT>();
                config->engine_id = engineId;
                config->knobs.push_back(
                    makeIntSetting(constants::K_KNOB_CATALOG_TILE_KNOB, entry.tileSize));
                config->knobs.push_back(
                    makeStringSetting(constants::K_KNOB_CATALOG_MODE_KNOB, entry.mode));
                catalog.configs.push_back(std::move(config));
            }

            if(mode == constants::K_KNOB_CATALOG_MODE_BAD_VALUE)
            {
                catalog.configs[1]->knobs[0]
                    = makeIntSetting(constants::K_KNOB_CATALOG_TILE_KNOB, K_UNSUPPORTED_TILE_SIZE);
            }

            flatbuffers::FlatBufferBuilder builder;
            builder.Finish(data_objects::EngineKnobCatalog::Pack(builder, &catalog));
            auto serialized = builder.Release();

            TestPluginMallocBuffer buffer(std::malloc(serialized.size()));
            hipdnn_plugin_sdk::throwIfNull(buffer.get());
            std::memcpy(buffer.get(), serialized.data(), serialized.size());

            knobCatalog->ptr = buffer.release();
            knobCatalog->size = serialized.size();

            LOG_API_SUCCESS(apiName, "knobCatalog->ptr=" << knobCatalog->ptr);
        });
    }

    static hipdnnPluginStatus_t destroyEngineKnobCatalog(hipdnnEnginePluginHandle_t handle,
                                                         hipdnnPluginConstData_t* knobCatalog)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                                << ", knobCatalog=" << static_cast<void*>(knobCatalog));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(knobCatalog);
            hipdnn_plugin_sdk::throwIfNull(knobCatalog->ptr);

            const TestPluginConstMallocBuffer catalogBytes(knobCatalog->ptr);
            knobCatalog->ptr = nullptr;
            knobCatalog->size = 0;

            LOG_API_SUCCESS(apiName, "knobCatalog->ptr=" << knobCatalog->ptr);
        });
    }
};

// Initialize plugin instance on load
__attribute__((constructor)) static void initializePlugin()
{
    TestPluginBase::setInstance(std::make_unique<KnobCatalogPlugin>());
}

REGISTER_TEST_PLUGIN_API()

// NOLINTBEGIN(readability-identifier-naming)
extern "C" {

// The destroy-only build omits the getter, so the host must ignore the pair.
#ifndef HIPDNN_TEST_KNOB_CATALOG_DESTROY_ONLY
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetEngineKnobCatalog(hipdnnEnginePluginHandle_t handle,
                                           int64_t engine_id,
                                           const hipdnnPluginConstData_t* op_graph,
                                           hipdnnPluginConstData_t* knob_catalog)
{
    return KnobCatalogPlugin::getEngineKnobCatalog(handle, engine_id, op_graph, knob_catalog);
}
#endif

// The get-only build omits the matching destroy entry point, so the host must ignore the pair.
#ifndef HIPDNN_TEST_KNOB_CATALOG_GET_ONLY
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginDestroyEngineKnobCatalog(hipdnnEnginePluginHandle_t handle,
                                               hipdnnPluginConstData_t* knob_catalog)
{
    return KnobCatalogPlugin::destroyEngineKnobCatalog(handle, knob_catalog);
}
#endif

} // extern "C"
// NOLINTEND(readability-identifier-naming)

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TestUtil.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "hipdnn_backend.h"
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <string>
#include <test_plugins/TestPluginConstants.hpp>
#include <utility>
#include <vector>

namespace
{

namespace constants = hipdnn_tests::plugin_constants;

struct ExpectedSetting
{
    int64_t tileSize;
    std::string mode;
};

// The catalog TestKnobCatalogPlugin serves by default, in order.
const std::vector<ExpectedSetting> K_SERVED_CATALOG
    = {{16, "fast"}, {32, "fast"}, {64, "accurate"}};

class IntegrationEngineKnobCatalog : public ::testing::Test
{
protected:
    // The plugin the fixture loads and the engine it queries.
    virtual std::string pluginPath() const
    {
        return constants::testKnobCatalogPluginPath();
    }

    virtual int64_t engineId() const
    {
        return constants::engineId<KnobCatalogPlugin>();
    }

    void SetUp() override
    {
        const auto path = pluginPath();
        const std::array<const char*, 1> paths = {path.c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

        test_util::createTestEngine(&_engine, &_graph, _handle, engineId(), true);
    }

    void TearDown() override
    {
        if(_engine != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_engine), HIPDNN_STATUS_SUCCESS);
        }
        if(_graph != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_graph), HIPDNN_STATUS_SUCCESS);
        }
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
    }

    // Count-then-data read of a descriptor array attribute. Returns the status of the
    // first failing call so tests can assert on it.
    static hipdnnStatus_t getDescriptors(hipdnnBackendDescriptor_t desc,
                                         hipdnnBackendAttributeName_t attribute,
                                         std::vector<hipdnn_backend::ScopedDescriptor>& out)
    {
        out.clear();

        int64_t count = -1;
        auto status = hipdnnBackendGetAttribute(
            desc, attribute, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr);
        if(status != HIPDNN_STATUS_SUCCESS || count <= 0)
        {
            return status;
        }

        std::vector<hipdnnBackendDescriptor_t> raw(static_cast<size_t>(count));
        int64_t returned = 0;
        status = hipdnnBackendGetAttribute(desc,
                                           attribute,
                                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                           count,
                                           &returned,
                                           static_cast<void*>(raw.data()));
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            return status;
        }

        EXPECT_EQ(returned, count);
        for(auto* descriptor : raw)
        {
            out.emplace_back(descriptor);
        }
        return status;
    }

    static std::string readString(hipdnnBackendDescriptor_t desc,
                                  hipdnnBackendAttributeName_t attribute)
    {
        int64_t count = 0;
        EXPECT_EQ(hipdnnBackendGetAttribute(desc, attribute, HIPDNN_TYPE_CHAR, 0, &count, nullptr),
                  HIPDNN_STATUS_SUCCESS);
        std::vector<char> buffer(static_cast<size_t>(count));
        int64_t returned = 0;
        EXPECT_EQ(hipdnnBackendGetAttribute(
                      desc, attribute, HIPDNN_TYPE_CHAR, count, &returned, buffer.data()),
                  HIPDNN_STATUS_SUCCESS);
        return {buffer.data()};
    }

    static int64_t readInt(hipdnnBackendDescriptor_t desc, hipdnnBackendAttributeName_t attribute)
    {
        int64_t value = 0;
        int64_t returned = 0;
        EXPECT_EQ(
            hipdnnBackendGetAttribute(desc, attribute, HIPDNN_TYPE_INT64, 1, &returned, &value),
            HIPDNN_STATUS_SUCCESS);
        return value;
    }

    // Reads an entry back into the (tile size, mode) pair the fixture catalogs over.
    static ExpectedSetting readEntry(const std::vector<hipdnn_backend::ScopedDescriptor>& choices)
    {
        ExpectedSetting entry{-1, ""};
        for(const auto& choice : choices)
        {
            const auto knobId = readString(choice.get(), HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE);
            if(knobId == constants::K_KNOB_CATALOG_TILE_KNOB)
            {
                entry.tileSize = readInt(choice.get(), HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE);
            }
            else if(knobId == constants::K_KNOB_CATALOG_MODE_KNOB)
            {
                entry.mode = readString(choice.get(), HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE);
            }
            else
            {
                ADD_FAILURE() << "Unexpected knob in catalog entry: " << knobId;
            }
        }
        return entry;
    }

    hipdnnHandle_t _handle = nullptr;
    hipdnnBackendDescriptor_t _engine = nullptr;
    hipdnnBackendDescriptor_t _graph = nullptr;
};

// A plugin that exports no knob catalog entry points and advertises no knobs.
class IntegrationEngineKnobCatalogWithoutEntryPoint : public IntegrationEngineKnobCatalog
{
protected:
    std::string pluginPath() const override
    {
        return constants::testGoodPluginPath();
    }

    int64_t engineId() const override
    {
        return constants::engineId<GoodPlugin>();
    }
};

// A plugin that exports the catalog getter but not its matching destroy entry point.
class IntegrationEngineKnobCatalogGetterOnly : public IntegrationEngineKnobCatalog
{
protected:
    std::string pluginPath() const override
    {
        return constants::testKnobCatalogGetOnlyPluginPath();
    }
};

// A plugin that exports the catalog destroy entry point but not the getter.
class IntegrationEngineKnobCatalogDestroyOnly : public IntegrationEngineKnobCatalog
{
protected:
    std::string pluginPath() const override
    {
        return constants::testKnobCatalogDestroyOnlyPluginPath();
    }
};

} // namespace

TEST_F(IntegrationEngineKnobCatalog, ServedCatalogIsReadBackEntryByEntry)
{
    std::vector<hipdnn_backend::ScopedDescriptor> entries;
    ASSERT_EQ(getDescriptors(_engine, HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT, entries),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(entries.size(), K_SERVED_CATALOG.size());

    for(size_t i = 0; i < entries.size(); ++i)
    {
        std::vector<hipdnn_backend::ScopedDescriptor> choices;
        ASSERT_EQ(getDescriptors(entries[i].get(), HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES, choices),
                  HIPDNN_STATUS_SUCCESS)
            << "entry " << i;
        ASSERT_EQ(choices.size(), 2U) << "entry " << i;

        const auto entry = readEntry(choices);
        EXPECT_EQ(entry.tileSize, K_SERVED_CATALOG[i].tileSize) << "entry " << i;
        EXPECT_EQ(entry.mode, K_SERVED_CATALOG[i].mode) << "entry " << i;
    }
}

TEST_F(IntegrationEngineKnobCatalog, EntryChoicesApplyUnchangedToANewEngineConfig)
{
    std::vector<hipdnn_backend::ScopedDescriptor> entries;
    ASSERT_EQ(getDescriptors(_engine, HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT, entries),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(entries.size(), K_SERVED_CATALOG.size());

    std::vector<hipdnn_backend::ScopedDescriptor> choices;
    ASSERT_EQ(getDescriptors(entries.back().get(), HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES, choices),
              HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t rawConfig = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &rawConfig),
              HIPDNN_STATUS_SUCCESS);
    const hipdnn_backend::ScopedDescriptor config(rawConfig);

    ASSERT_EQ(hipdnnBackendSetAttribute(config.get(),
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&_engine)),
              HIPDNN_STATUS_SUCCESS);

    std::vector<hipdnnBackendDescriptor_t> raw;
    raw.reserve(choices.size());
    for(const auto& choice : choices)
    {
        raw.push_back(choice.get());
    }
    ASSERT_EQ(hipdnnBackendSetAttribute(config.get(),
                                        HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        static_cast<int64_t>(raw.size()),
                                        static_cast<const void*>(raw.data())),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendFinalize(config.get()), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationEngineKnobCatalog, CatalogEntriesAreFinalizedEngineConfigs)
{
    std::vector<hipdnn_backend::ScopedDescriptor> entries;
    ASSERT_EQ(getDescriptors(_engine, HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT, entries),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_FALSE(entries.empty());

    // Only a finalized engine config reports its workspace size.
    int64_t workspaceSize = 0;
    int64_t returned = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(entries.front().get(),
                                        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE,
                                        HIPDNN_TYPE_INT64,
                                        1,
                                        &returned,
                                        &workspaceSize),
              HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationEngineKnobCatalog, PluginDecliningACatalogIsNotSupportedWhenTheEngineHasKnobs)
{
    // The engine advertises two engine-specific knobs, so without a catalog there is no
    // valid configuration to report.
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter mode(
        constants::K_KNOB_CATALOG_PLUGIN_MODE_ENV, constants::K_KNOB_CATALOG_MODE_NOT_APPLICABLE);

    int64_t count = -1;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &count,
                                        nullptr),
              HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(IntegrationEngineKnobCatalog, CatalogViolatingAKnobConstraintIsRejectedNamingTheEntry)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter mode(
        constants::K_KNOB_CATALOG_PLUGIN_MODE_ENV, constants::K_KNOB_CATALOG_MODE_BAD_VALUE);

    int64_t count = -1;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &count,
                                        nullptr),
              HIPDNN_STATUS_INTERNAL_ERROR);

    std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> message{};
    hipdnnGetLastErrorString(message.data(), message.size());
    EXPECT_NE(std::string(message.data()).find("entry 1"), std::string::npos) << message.data();
}

TEST_F(IntegrationEngineKnobCatalogWithoutEntryPoint, EngineWithoutKnobsReportsOneEmptyDefaultEntry)
{
    std::vector<hipdnn_backend::ScopedDescriptor> entries;
    ASSERT_EQ(getDescriptors(_engine, HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT, entries),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(entries.size(), 1U);

    int64_t choiceCount = -1;
    ASSERT_EQ(hipdnnBackendGetAttribute(entries.front().get(),
                                        HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &choiceCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(choiceCount, 0);
}

TEST_F(IntegrationEngineKnobCatalogGetterOnly, LoneGetterIsIgnoredSoTheEngineHasNoCatalog)
{
    // The plugin exports the getter and would serve three entries, but a catalog it cannot
    // release is unusable: the host falls back, and an engine with knobs has no fallback.
    int64_t count = -1;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &count,
                                        nullptr),
              HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(IntegrationEngineKnobCatalogDestroyOnly, LoneDestroyIsIgnoredSoTheEngineHasNoCatalog)
{
    // The plugin exports a destroy entry point with no getter to pair it with: the host falls
    // back, and an engine with knobs has no fallback.
    int64_t count = -1;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &count,
                                        nullptr),
              HIPDNN_STATUS_NOT_SUPPORTED);
}

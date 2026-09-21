// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipdnnBackendFlatbufferData.h"
#include "TestUtil.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "hipdnn_backend.h"
#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_plugin_sdk/KnobSettingFactory.hpp>
#include <test_plugins/TestPluginConstants.hpp>
#include <unordered_set>
#include <vector>

using hipdnn_flatbuffers_sdk::data_objects::Knob;
using hipdnn_flatbuffers_sdk::data_objects::KnobConstraint;
using hipdnn_flatbuffers_sdk::data_objects::KnobValue;
using hipdnn_plugin_sdk::KnobSettingFactory;

class IntegrationKnobsApi : public ::testing::Test
{
protected:
    hipdnnBackendDescriptor_t _engine = nullptr;
    hipdnnBackendDescriptor_t _engineConfig = nullptr;
    hipdnnBackendDescriptor_t _graph = nullptr;
    hipdnnHandle_t _handle = nullptr;

    void SetUp() override
    {
        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testKnobsPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_engineConfig != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_engineConfig), HIPDNN_STATUS_SUCCESS);
        }
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
            _handle = nullptr;
        }
    }

    void createFinalizedEngine()
    {
        const int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
        test_util::createTestEngine(&_engine, &_graph, _handle, gidx, true);
    }

    void createFinalizedEngineConfig()
    {
        const int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
        ASSERT_EQ(
            hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
            HIPDNN_STATUS_SUCCESS);
        test_util::populateTestEngineConfig(&_engineConfig, &_engine, &_graph, _handle, gidx, true);
    }

    /// Retrieves knob info descriptors from _engine, wrapped in RAII.
    std::vector<hipdnn_backend::ScopedDescriptor> getKnobDescriptors(int64_t expectedCount)
    {
        int64_t knobCount = 0;
        auto status = hipdnnBackendGetAttribute(_engine,
                                                HIPDNN_ATTR_ENGINE_KNOB_INFO,
                                                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                0,
                                                &knobCount,
                                                nullptr);
        EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(knobCount, expectedCount);
        if(status != HIPDNN_STATUS_SUCCESS || knobCount != expectedCount)
        {
            return {};
        }

        std::vector<hipdnnBackendDescriptor_t> rawDescs(static_cast<size_t>(knobCount));
        int64_t returnedCount = 0;
        status = hipdnnBackendGetAttribute(_engine,
                                           HIPDNN_ATTR_ENGINE_KNOB_INFO,
                                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                           knobCount,
                                           &returnedCount,
                                           static_cast<void*>(rawDescs.data()));
        EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            // Clean up any descriptors that were returned before returning empty
            for(auto raw : rawDescs)
            {
                delete raw;
            }
            return {};
        }

        // Immediately wrap in RAII
        std::vector<hipdnn_backend::ScopedDescriptor> result;
        result.reserve(rawDescs.size());
        for(auto raw : rawDescs)
        {
            result.emplace_back(raw);
        }
        return result;
    }

    /// Finds a knob descriptor by its knob ID string.
    static hipdnnBackendDescriptor_t
        findKnobDescriptorById(const std::vector<hipdnn_backend::ScopedDescriptor>& knobDescs,
                               const std::string& targetId)
    {
        for(const auto& desc : knobDescs)
        {
            if(desc.get() == nullptr)
            {
                continue;
            }

            std::array<char, 256> knobId = {};
            int64_t idLen = 0;
            auto status = hipdnnBackendGetAttribute(desc.get(),
                                                    HIPDNN_ATTR_KNOB_INFO_TYPE,
                                                    HIPDNN_TYPE_CHAR,
                                                    static_cast<int64_t>(knobId.size()),
                                                    &idLen,
                                                    knobId.data());
            if(status == HIPDNN_STATUS_SUCCESS && std::string(knobId.data()) == targetId)
            {
                return desc.get();
            }
        }
        return nullptr;
    }
};

// =============================================================================
// Engine Knob Info Tests (via HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE)
// =============================================================================

TEST_F(IntegrationKnobsApi, GetKnobInfoCountFromEngine)
{
    createFinalizedEngine();

    int64_t knobCount = -1;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        0,
                                        &knobCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    // The test plugin exposes 5 knobs: int_knob, float_knob, string_knob, deprecated_knob, shared.deterministic
    EXPECT_EQ(knobCount, 5);
}

TEST_F(IntegrationKnobsApi, GetKnobInfoDataFromEngine)
{
    createFinalizedEngine();

    // First get the count
    int64_t knobCount = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        0,
                                        &knobCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(knobCount, 5);

    // Now get the actual knob data
    std::vector<hipdnnBackendFlatbufferData_t> knobData(static_cast<size_t>(knobCount));
    int64_t returnedCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        knobCount,
                                        &returnedCount,
                                        knobData.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(returnedCount, 5);

    // Verify each knob data is valid
    for(size_t i = 0; i < static_cast<size_t>(returnedCount); ++i)
    {
        EXPECT_NE(knobData[i].ptr, nullptr);
        EXPECT_GT(knobData[i].size, 0UL);
    }
}

TEST_F(IntegrationKnobsApi, GetKnobInfoValidateIntKnob)
{
    createFinalizedEngine();

    int64_t knobCount = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        0,
                                        &knobCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(knobCount, 0);

    std::vector<hipdnnBackendFlatbufferData_t> knobData(static_cast<size_t>(knobCount));
    int64_t returnedCount = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        knobCount,
                                        &returnedCount,
                                        knobData.data()),
              HIPDNN_STATUS_SUCCESS);

    // Find and validate the integer knob
    bool foundIntKnob = false;
    for(size_t i = 0; i < static_cast<size_t>(returnedCount); ++i)
    {
        flatbuffers::Verifier verifier( // NOLINT(misc-const-correctness)
            static_cast<const uint8_t*>(knobData[i].ptr),
            knobData[i].size);
        ASSERT_TRUE(verifier.VerifyBuffer<Knob>());

        auto knob = flatbuffers::GetRoot<Knob>(knobData[i].ptr);

        if(knob->knob_id()->str() == "test.int_knob")
        {
            foundIntKnob = true;

            // Verify description
            EXPECT_STREQ(knob->description()->c_str(), "Test integer knob with range 0-100");

            // Verify default value type and value
            EXPECT_EQ(knob->default_value_type(), KnobValue::IntValue);
            auto intValue = knob->default_value_as_IntValue();
            ASSERT_NE(intValue, nullptr);
            EXPECT_EQ(intValue->value(), 50);

            // Verify constraint
            EXPECT_EQ(knob->constraint_type(), KnobConstraint::IntConstraint);
            auto constraint = knob->constraint_as_IntConstraint();
            ASSERT_NE(constraint, nullptr);
            EXPECT_EQ(constraint->min_value(), 0);
            EXPECT_EQ(constraint->max_value(), 100);
            EXPECT_EQ(constraint->step(), 10);

            // Verify not deprecated
            EXPECT_FALSE(knob->deprecated());

            break;
        }
    }
    EXPECT_TRUE(foundIntKnob) << "Integer knob 'test.int_knob' not found";
}

TEST_F(IntegrationKnobsApi, GetKnobInfoValidateFloatKnob)
{
    createFinalizedEngine();

    int64_t knobCount = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        0,
                                        &knobCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    std::vector<hipdnnBackendFlatbufferData_t> knobData(static_cast<size_t>(knobCount));
    int64_t returnedCount = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        knobCount,
                                        &returnedCount,
                                        knobData.data()),
              HIPDNN_STATUS_SUCCESS);

    // Find and validate the float knob
    bool foundFloatKnob = false;
    for(size_t i = 0; i < static_cast<size_t>(returnedCount); ++i)
    {
        auto knob = flatbuffers::GetRoot<Knob>(knobData[i].ptr);

        if(knob->knob_id()->str() == "test.float_knob")
        {
            foundFloatKnob = true;

            // Verify description
            EXPECT_STREQ(knob->description()->c_str(), "Test float knob with range 0.0-1.0");

            // Verify default value type and value
            EXPECT_EQ(knob->default_value_type(), KnobValue::FloatValue);
            auto floatValue = knob->default_value_as_FloatValue();
            ASSERT_NE(floatValue, nullptr);
            EXPECT_DOUBLE_EQ(floatValue->value(), 0.5);

            // Verify constraint
            EXPECT_EQ(knob->constraint_type(), KnobConstraint::FloatConstraint);
            auto constraint = knob->constraint_as_FloatConstraint();
            ASSERT_NE(constraint, nullptr);
            EXPECT_DOUBLE_EQ(constraint->min_value(), 0.0);
            EXPECT_DOUBLE_EQ(constraint->max_value(), 1.0);

            // Verify not deprecated
            EXPECT_FALSE(knob->deprecated());

            break;
        }
    }
    EXPECT_TRUE(foundFloatKnob) << "Float knob 'test.float_knob' not found";
}

TEST_F(IntegrationKnobsApi, GetKnobInfoValidateStringKnob)
{
    createFinalizedEngine();

    int64_t knobCount = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        0,
                                        &knobCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    std::vector<hipdnnBackendFlatbufferData_t> knobData(static_cast<size_t>(knobCount));
    int64_t returnedCount = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        knobCount,
                                        &returnedCount,
                                        knobData.data()),
              HIPDNN_STATUS_SUCCESS);

    // Find and validate the string knob
    bool foundStringKnob = false;
    for(size_t i = 0; i < static_cast<size_t>(returnedCount); ++i)
    {
        auto knob = flatbuffers::GetRoot<Knob>(knobData[i].ptr);

        if(knob->knob_id()->str() == "test.string_knob")
        {
            foundStringKnob = true;

            // Verify description
            EXPECT_STREQ(knob->description()->c_str(), "Test string knob with enum values");

            // Verify default value type and value
            EXPECT_EQ(knob->default_value_type(), KnobValue::StringValue);
            auto stringValue = knob->default_value_as_StringValue();
            ASSERT_NE(stringValue, nullptr);
            EXPECT_STREQ(stringValue->value()->c_str(), "fast");

            // Verify constraint
            EXPECT_EQ(knob->constraint_type(), KnobConstraint::StringConstraint);
            auto constraint = knob->constraint_as_StringConstraint();
            ASSERT_NE(constraint, nullptr);
            // Note: KnobFactory uses max_length=0 (no length limit) for string knobs
            EXPECT_EQ(constraint->max_length(), 0);
            ASSERT_NE(constraint->valid_values(), nullptr);
            EXPECT_EQ(constraint->valid_values()->size(), 3UL);

            // Check valid values
            std::vector<std::string> expectedValues = {"fast", "accurate", "balanced"};
            for(size_t j = 0; j < constraint->valid_values()->size(); ++j)
            {
                EXPECT_EQ(constraint->valid_values()->Get(static_cast<unsigned int>(j))->str(),
                          expectedValues[j]);
            }

            // Verify not deprecated
            EXPECT_FALSE(knob->deprecated());

            break;
        }
    }
    EXPECT_TRUE(foundStringKnob) << "String knob 'test.string_knob' not found";
}

TEST_F(IntegrationKnobsApi, GetKnobInfoValidateDeprecatedKnob)
{
    createFinalizedEngine();

    int64_t knobCount = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        0,
                                        &knobCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    std::vector<hipdnnBackendFlatbufferData_t> knobData(static_cast<size_t>(knobCount));
    int64_t returnedCount = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        knobCount,
                                        &returnedCount,
                                        knobData.data()),
              HIPDNN_STATUS_SUCCESS);

    // Find and validate the deprecated knob
    bool foundDeprecatedKnob = false;
    for(size_t i = 0; i < static_cast<size_t>(returnedCount); ++i)
    {
        auto knob = flatbuffers::GetRoot<Knob>(knobData[i].ptr);

        if(knob->knob_id()->str() == "test.deprecated_knob")
        {
            foundDeprecatedKnob = true;

            // Verify deprecated flag is set
            EXPECT_TRUE(knob->deprecated());

            break;
        }
    }
    EXPECT_TRUE(foundDeprecatedKnob) << "Deprecated knob 'test.deprecated_knob' not found";
}

TEST_F(IntegrationKnobsApi, GetKnobInfoNotFinalizedEngine)
{
    int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();

    // Create engine but don't finalize
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, &_engine),
              HIPDNN_STATUS_SUCCESS);
    test_util::createTestGraph(&_graph, _handle);
    ASSERT_EQ(hipdnnBackendFinalize(_graph), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetAttribute(_engine,
                                        HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&_graph)),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetAttribute(
                  _engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &gidx),
              HIPDNN_STATUS_SUCCESS);
    // Not finalizing!

    int64_t knobCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        0,
                                        &knobCount,
                                        nullptr),
              HIPDNN_STATUS_NOT_INITIALIZED);
}

// =============================================================================
// EngineConfig Knob Choice Tests (via HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE)
// =============================================================================

TEST_F(IntegrationKnobsApi, SetKnobChoiceIntValue)
{
    const int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    test_util::createTestEngine(&_engine, &_graph, _handle, gidx, true);

    ASSERT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&_engine)),
              HIPDNN_STATUS_SUCCESS);

    // Set an integer knob value
    auto knobBuffer = KnobSettingFactory::createIntKnobSetting("test.int_knob", 70);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    // Finalize should succeed
    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationKnobsApi, SetKnobChoiceFloatValue)
{
    const int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    test_util::createTestEngine(&_engine, &_graph, _handle, gidx, true);

    ASSERT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&_engine)),
              HIPDNN_STATUS_SUCCESS);

    // Set a float knob value
    auto knobBuffer = KnobSettingFactory::createFloatKnobSetting("test.float_knob", 0.75);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    // Finalize should succeed
    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationKnobsApi, SetKnobChoiceStringValue)
{
    const int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    test_util::createTestEngine(&_engine, &_graph, _handle, gidx, true);

    ASSERT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&_engine)),
              HIPDNN_STATUS_SUCCESS);

    // Set a string knob value
    auto knobBuffer = KnobSettingFactory::createStringKnobSetting("test.string_knob", "accurate");
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    // Finalize should succeed
    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationKnobsApi, SetKnobChoiceMultipleKnobs)
{
    const int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    test_util::createTestEngine(&_engine, &_graph, _handle, gidx, true);

    ASSERT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&_engine)),
              HIPDNN_STATUS_SUCCESS);

    // Create multiple knob settings
    auto intKnobBuffer = KnobSettingFactory::createIntKnobSetting("test.int_knob", 80);
    auto floatKnobBuffer = KnobSettingFactory::createFloatKnobSetting("test.float_knob", 0.25);
    auto stringKnobBuffer
        = KnobSettingFactory::createStringKnobSetting("test.string_knob", "balanced");

    std::vector<hipdnnBackendFlatbufferData_t> knobDataArray
        = {{intKnobBuffer.data(), intKnobBuffer.size()},
           {floatKnobBuffer.data(), floatKnobBuffer.size()},
           {stringKnobBuffer.data(), stringKnobBuffer.size()}};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        3,
                                        knobDataArray.data()),
              HIPDNN_STATUS_SUCCESS);

    // Finalize should succeed
    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationKnobsApi, FinalizeEngineConfigWithKnobs)
{
    createFinalizedEngineConfig();

    // Just verify we can finalize successfully - the fixture already does this
    // This test documents that the basic flow works
    SUCCEED();
}

TEST_F(IntegrationKnobsApi, SetKnobChoiceNullPointer)
{
    const int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    test_util::createTestEngine(&_engine, &_graph, _handle, gidx, true);

    ASSERT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&_engine)),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(IntegrationKnobsApi, SetKnobChoiceInvalidType)
{
    const int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    test_util::createTestEngine(&_engine, &_graph, _handle, gidx, true);

    ASSERT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&_engine)),
              HIPDNN_STATUS_SUCCESS);

    auto knobBuffer = KnobSettingFactory::createIntKnobSetting("test.int_knob", 50);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    // Wrong attribute type
    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_INT64,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(IntegrationKnobsApi, SetKnobChoiceOnFinalizedConfig)
{
    createFinalizedEngineConfig();

    auto knobBuffer = KnobSettingFactory::createIntKnobSetting("test.int_knob", 50);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    // Cannot set knob choice on already finalized engine config
    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(IntegrationKnobsApi, GetMaxWorkspaceSizeWithKnobs)
{
    const int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    test_util::createTestEngine(&_engine, &_graph, _handle, gidx, true);

    ASSERT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&_engine)),
              HIPDNN_STATUS_SUCCESS);

    // Set a knob before finalizing
    auto knobBuffer = KnobSettingFactory::createIntKnobSetting("test.int_knob", 60);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    ASSERT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);

    // Verify we can still get workspace size after setting knobs
    int64_t workspaceSize = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engineConfig,
                                        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE,
                                        HIPDNN_TYPE_INT64,
                                        1,
                                        nullptr,
                                        &workspaceSize),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(workspaceSize, 1024); // Test plugin returns 1024
}

// =============================================================================
// Constraint Type Validation Tests
// =============================================================================

class IntegrationConstraintValidationApi : public ::testing::Test
{
protected:
    hipdnnBackendDescriptor_t _engine = nullptr;
    hipdnnBackendDescriptor_t _engineConfig = nullptr;
    hipdnnBackendDescriptor_t _graph = nullptr;
    hipdnnHandle_t _handle = nullptr;

    void SetUp() override
    {
        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testKnobConstraintValidationPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_engineConfig != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_engineConfig), HIPDNN_STATUS_SUCCESS);
        }
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
            _handle = nullptr;
        }
    }

    void setupEngineConfig()
    {
        const int64_t gidx
            = hipdnn_tests::plugin_constants::engineId<KnobConstraintValidationPlugin>();
        ASSERT_EQ(
            hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
            HIPDNN_STATUS_SUCCESS);
        test_util::createTestEngine(&_engine, &_graph, _handle, gidx, true);
        ASSERT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                            HIPDNN_ATTR_ENGINECFG_ENGINE,
                                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                            1,
                                            static_cast<const void*>(&_engine)),
                  HIPDNN_STATUS_SUCCESS);
    }

    /// Creates an execution plan with the fixture's handle and engine config,
    /// then asserts finalize returns the expected status.
    void expectExecutionPlanFinalize(hipdnnStatus_t expectedStatus)
    {
        hipdnn_backend::ScopedDescriptor executionPlan;
        ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
                                                executionPlan.getPtr()),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnBackendSetAttribute(executionPlan.get(),
                                            HIPDNN_ATTR_EXECUTION_PLAN_HANDLE,
                                            HIPDNN_TYPE_HANDLE,
                                            1,
                                            static_cast<const void*>(&_handle)),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnBackendSetAttribute(executionPlan.get(),
                                            HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                            1,
                                            static_cast<const void*>(&_engineConfig)),
                  HIPDNN_STATUS_SUCCESS);

        EXPECT_EQ(hipdnnBackendFinalize(executionPlan.get()), expectedStatus);
    }
};

TEST_F(IntegrationConstraintValidationApi, IntValueToFloatConstraint)
{
    setupEngineConfig();

    auto knobBuffer = KnobSettingFactory::createIntKnobSetting("constraint.float_knob", 5);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);
    expectExecutionPlanFinalize(HIPDNN_STATUS_PLUGIN_ERROR);
}

TEST_F(IntegrationConstraintValidationApi, FloatValueToIntConstraint)
{
    setupEngineConfig();

    auto knobBuffer = KnobSettingFactory::createFloatKnobSetting("constraint.int_knob", 50.0);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);
    expectExecutionPlanFinalize(HIPDNN_STATUS_PLUGIN_ERROR);
}

TEST_F(IntegrationConstraintValidationApi, StringValueToIntConstraint)
{
    setupEngineConfig();

    auto knobBuffer = KnobSettingFactory::createStringKnobSetting("constraint.int_knob", "50");
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);
    expectExecutionPlanFinalize(HIPDNN_STATUS_PLUGIN_ERROR);
}

TEST_F(IntegrationConstraintValidationApi, IntValueToStringConstraint)
{
    setupEngineConfig();

    auto knobBuffer = KnobSettingFactory::createIntKnobSetting("constraint.string_knob", 0);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);
    expectExecutionPlanFinalize(HIPDNN_STATUS_PLUGIN_ERROR);
}

TEST_F(IntegrationConstraintValidationApi, FloatValueToStringConstraint)
{
    setupEngineConfig();

    auto knobBuffer = KnobSettingFactory::createFloatKnobSetting("constraint.string_knob", 1.5);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);
    expectExecutionPlanFinalize(HIPDNN_STATUS_PLUGIN_ERROR);
}

TEST_F(IntegrationConstraintValidationApi, StringValueToFloatConstraint)
{
    setupEngineConfig();

    auto knobBuffer = KnobSettingFactory::createStringKnobSetting("constraint.float_knob", "5.0");
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);
    expectExecutionPlanFinalize(HIPDNN_STATUS_PLUGIN_ERROR);
}

TEST_F(IntegrationConstraintValidationApi, CorrectTypesSucceed)
{
    setupEngineConfig();

    auto intKnobBuffer = KnobSettingFactory::createIntKnobSetting("constraint.int_knob", 50);
    auto floatKnobBuffer = KnobSettingFactory::createFloatKnobSetting("constraint.float_knob", 5.0);
    auto stringKnobBuffer
        = KnobSettingFactory::createStringKnobSetting("constraint.string_knob", "beta");

    std::vector<hipdnnBackendFlatbufferData_t> knobDataArray
        = {{intKnobBuffer.data(), intKnobBuffer.size()},
           {floatKnobBuffer.data(), floatKnobBuffer.size()},
           {stringKnobBuffer.data(), stringKnobBuffer.size()}};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        3,
                                        knobDataArray.data()),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);
    expectExecutionPlanFinalize(HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationConstraintValidationApi, UnknownKnobIsIgnored)
{
    setupEngineConfig();

    auto knobBuffer = KnobSettingFactory::createIntKnobSetting("unknown.knob", 123);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);
    expectExecutionPlanFinalize(HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationConstraintValidationApi, MixedValidAndInvalidKnobs)
{
    setupEngineConfig();

    auto validKnobBuffer = KnobSettingFactory::createIntKnobSetting("constraint.int_knob", 50);
    auto invalidKnobBuffer = KnobSettingFactory::createIntKnobSetting("constraint.float_knob", 5);

    std::vector<hipdnnBackendFlatbufferData_t> knobDataArray
        = {{validKnobBuffer.data(), validKnobBuffer.size()},
           {invalidKnobBuffer.data(), invalidKnobBuffer.size()}};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        2,
                                        knobDataArray.data()),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);
    expectExecutionPlanFinalize(HIPDNN_STATUS_PLUGIN_ERROR);
}

// =============================================================================
// Engine Knob Info via Descriptor API (HIPDNN_ATTR_ENGINE_KNOB_INFO)
// =============================================================================

TEST_F(IntegrationKnobsApi, GetKnobInfoDescriptorCount)
{
    createFinalizedEngine();

    int64_t knobCount = -1;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_ENGINE_KNOB_INFO,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &knobCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(knobCount, 5);
}

TEST_F(IntegrationKnobsApi, GetKnobInfoDescriptorsAndValidateIntKnob)
{
    createFinalizedEngine();
    auto knobDescs = getKnobDescriptors(5);
    ASSERT_FALSE(knobDescs.empty());

    auto* intKnob = findKnobDescriptorById(knobDescs, "test.int_knob");
    ASSERT_NE(intKnob, nullptr) << "Integer knob 'test.int_knob' not found via descriptor API";

    // Verify default value type
    int64_t valueType = 0;
    int64_t vtCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(intKnob,
                                        HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE,
                                        HIPDNN_TYPE_INT64,
                                        1,
                                        &vtCount,
                                        &valueType),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(valueType, static_cast<int64_t>(HIPDNN_TYPE_INT64));

    // Verify default value
    int64_t defaultVal = 0;
    int64_t dvCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(intKnob,
                                        HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE,
                                        HIPDNN_TYPE_INT64,
                                        1,
                                        &dvCount,
                                        &defaultVal),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(defaultVal, 50);

    // Verify min/max
    int64_t minVal = 0;
    int64_t minCount = 0;
    EXPECT_EQ(
        hipdnnBackendGetAttribute(
            intKnob, HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &minCount, &minVal),
        HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(minVal, 0);

    int64_t maxVal = 0;
    int64_t maxCount = 0;
    EXPECT_EQ(
        hipdnnBackendGetAttribute(
            intKnob, HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64, 1, &maxCount, &maxVal),
        HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(maxVal, 100);

    // Verify stride
    int64_t stride = 0;
    int64_t strideCount = 0;
    EXPECT_EQ(
        hipdnnBackendGetAttribute(
            intKnob, HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64, 1, &strideCount, &stride),
        HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(stride, 10);
}

TEST_F(IntegrationKnobsApi, GetKnobInfoDescriptorsAndValidateFloatKnob)
{
    createFinalizedEngine();
    auto knobDescs = getKnobDescriptors(5);
    ASSERT_FALSE(knobDescs.empty());

    auto* floatKnob = findKnobDescriptorById(knobDescs, "test.float_knob");
    ASSERT_NE(floatKnob, nullptr) << "Float knob 'test.float_knob' not found via descriptor API";

    // Verify default value type is DOUBLE
    int64_t valueType = 0;
    int64_t vtCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(floatKnob,
                                        HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE,
                                        HIPDNN_TYPE_INT64,
                                        1,
                                        &vtCount,
                                        &valueType),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(valueType, static_cast<int64_t>(HIPDNN_TYPE_DOUBLE));

    // Verify default value = 0.5
    double defaultVal = 0.0;
    int64_t dvCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(floatKnob,
                                        HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE,
                                        HIPDNN_TYPE_DOUBLE,
                                        1,
                                        &dvCount,
                                        &defaultVal),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_DOUBLE_EQ(defaultVal, 0.5);

    // Verify min = 0.0
    double minVal = -1.0;
    int64_t minCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(floatKnob,
                                        HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE,
                                        HIPDNN_TYPE_DOUBLE,
                                        1,
                                        &minCount,
                                        &minVal),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_DOUBLE_EQ(minVal, 0.0);

    // Verify max = 1.0
    double maxVal = -1.0;
    int64_t maxCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(floatKnob,
                                        HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE,
                                        HIPDNN_TYPE_DOUBLE,
                                        1,
                                        &maxCount,
                                        &maxVal),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_DOUBLE_EQ(maxVal, 1.0);
}

TEST_F(IntegrationKnobsApi, GetKnobInfoDescriptorsAndValidateStringKnob)
{
    createFinalizedEngine();
    auto knobDescs = getKnobDescriptors(5);
    ASSERT_FALSE(knobDescs.empty());

    auto* stringKnob = findKnobDescriptorById(knobDescs, "test.string_knob");
    ASSERT_NE(stringKnob, nullptr) << "String knob 'test.string_knob' not found via descriptor API";

    // Verify default value type is CHAR
    int64_t valueType = 0;
    int64_t vtCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(stringKnob,
                                        HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE,
                                        HIPDNN_TYPE_INT64,
                                        1,
                                        &vtCount,
                                        &valueType),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(valueType, static_cast<int64_t>(HIPDNN_TYPE_CHAR));

    // Verify default value = "fast"
    std::array<char, 64> defaultVal = {};
    int64_t dvCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(stringKnob,
                                        HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(defaultVal.size()),
                                        &dvCount,
                                        defaultVal.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(std::string(defaultVal.data()), "fast");

    // Verify valid values buffer contains the 3 choices
    int64_t validBufLen = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(stringKnob,
                                        HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING,
                                        HIPDNN_TYPE_CHAR,
                                        0,
                                        &validBufLen,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_GT(validBufLen, 0) << "Valid values buffer should be non-empty";

    std::vector<char> validBuf(static_cast<size_t>(validBufLen));
    int64_t actualLen = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(stringKnob,
                                        HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING,
                                        HIPDNN_TYPE_CHAR,
                                        validBufLen,
                                        &actualLen,
                                        validBuf.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(actualLen, validBufLen);

    // Parse the null-separated buffer and verify the 3 choices are present
    std::unordered_set<std::string> validValues;
    const char* p = validBuf.data();
    const char* end = p + actualLen;
    while(p < end)
    {
        const std::string s(p);
        if(!s.empty())
        {
            validValues.insert(s);
        }
        p += s.size() + 1;
    }
    EXPECT_EQ(validValues.count("fast"), 1u);
    EXPECT_EQ(validValues.count("accurate"), 1u);
    EXPECT_EQ(validValues.count("balanced"), 1u);
}

TEST_F(IntegrationKnobsApi, GetKnobInfoDescriptorsAndValidateDeprecatedKnob)
{
    createFinalizedEngine();
    auto knobDescs = getKnobDescriptors(5);
    ASSERT_FALSE(knobDescs.empty());

    auto* deprecatedKnob = findKnobDescriptorById(knobDescs, "test.deprecated_knob");
    ASSERT_NE(deprecatedKnob, nullptr)
        << "Deprecated knob 'test.deprecated_knob' not found via descriptor API";

    // Verify deprecated flag is true
    bool isDeprecated = false;
    int64_t depCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(deprecatedKnob,
                                        HIPDNN_ATTR_KNOB_INFO_DEPRECATED,
                                        HIPDNN_TYPE_BOOLEAN,
                                        1,
                                        &depCount,
                                        &isDeprecated),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_TRUE(isDeprecated) << "test.deprecated_knob should have deprecated flag set to true";
}

TEST_F(IntegrationKnobsApi, GetKnobInfoDescriptorsAndValidateValidValuesIntKnob)
{
    // Use Engine B which has test.engine_b.block_size with valid values {8, 16, 32, 64}
    const int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobsPluginEngineB>();
    test_util::createTestEngine(&_engine, &_graph, _handle, gidx, true);

    auto knobDescs = getKnobDescriptors(3);
    ASSERT_FALSE(knobDescs.empty());

    auto* blockSizeKnob = findKnobDescriptorById(knobDescs, "test.engine_b.block_size");
    ASSERT_NE(blockSizeKnob, nullptr)
        << "Knob 'test.engine_b.block_size' not found via descriptor API";

    // Verify valid values count = 4
    int64_t validCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(blockSizeKnob,
                                        HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT,
                                        HIPDNN_TYPE_INT64,
                                        0,
                                        &validCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(validCount, 4);

    // Fetch and verify the actual valid values {8, 16, 32, 64}
    std::vector<int64_t> validValues(static_cast<size_t>(validCount));
    int64_t actualCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(blockSizeKnob,
                                        HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT,
                                        HIPDNN_TYPE_INT64,
                                        validCount,
                                        &actualCount,
                                        validValues.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(actualCount, 4);

    const std::unordered_set<int64_t> expected = {8, 16, 32, 64};
    const std::unordered_set<int64_t> actual(validValues.begin(), validValues.end());
    EXPECT_EQ(actual, expected);
}

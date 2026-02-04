// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipdnnBackendFlatbufferData.h"
#include "TestUtil.hpp"
#include "hipdnn_backend.h"
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_plugin_sdk/KnobSettingFactory.hpp>
#include <test_plugins/TestPluginConstants.hpp>

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
        int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
        test_util::createTestEngine(&_engine, &_graph, _handle, gidx, true);
    }

    void createFinalizedEngineConfig()
    {
        int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
        ASSERT_EQ(
            hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
            HIPDNN_STATUS_SUCCESS);
        test_util::populateTestEngineConfig(&_engineConfig, &_engine, &_graph, _handle, gidx, true);
    }
};

// =============================================================================
// Engine Knob Info Tests (via HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT)
// =============================================================================

TEST_F(IntegrationKnobsApi, GetKnobInfoCountFromEngine)
{
    createFinalizedEngine();

    int64_t knobCount = -1;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        0,
                                        &knobCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    // The test plugin exposes 4 knobs: int_knob, float_knob, string_knob, deprecated_knob
    EXPECT_EQ(knobCount, 4);
}

TEST_F(IntegrationKnobsApi, GetKnobInfoDataFromEngine)
{
    createFinalizedEngine();

    // First get the count
    int64_t knobCount = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        0,
                                        &knobCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(knobCount, 4);

    // Now get the actual knob data
    std::vector<hipdnnBackendFlatbufferData_t> knobData(static_cast<size_t>(knobCount));
    int64_t returnedCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        knobCount,
                                        &returnedCount,
                                        knobData.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(returnedCount, 4);

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
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        0,
                                        &knobCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(knobCount, 0);

    std::vector<hipdnnBackendFlatbufferData_t> knobData(static_cast<size_t>(knobCount));
    int64_t returnedCount = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        knobCount,
                                        &returnedCount,
                                        knobData.data()),
              HIPDNN_STATUS_SUCCESS);

    // Find and validate the integer knob
    bool foundIntKnob = false;
    for(size_t i = 0; i < static_cast<size_t>(returnedCount); ++i)
    {
        flatbuffers::Verifier verifier(static_cast<const uint8_t*>(knobData[i].ptr),
                                       knobData[i].size);
        ASSERT_TRUE(verifier.VerifyBuffer<hipdnn_data_sdk::data_objects::Knob>());

        auto knob = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Knob>(knobData[i].ptr);

        if(knob->knob_id()->str() == "test.int_knob")
        {
            foundIntKnob = true;

            // Verify description
            EXPECT_STREQ(knob->description()->c_str(), "Test integer knob with range 0-100");

            // Verify default value type and value
            EXPECT_EQ(knob->default_value_type(),
                      hipdnn_data_sdk::data_objects::KnobValue::IntValue);
            auto intValue = knob->default_value_as_IntValue();
            ASSERT_NE(intValue, nullptr);
            EXPECT_EQ(intValue->value(), 50);

            // Verify constraint
            EXPECT_EQ(knob->constraint_type(),
                      hipdnn_data_sdk::data_objects::KnobConstraint::IntConstraint);
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
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        0,
                                        &knobCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    std::vector<hipdnnBackendFlatbufferData_t> knobData(static_cast<size_t>(knobCount));
    int64_t returnedCount = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        knobCount,
                                        &returnedCount,
                                        knobData.data()),
              HIPDNN_STATUS_SUCCESS);

    // Find and validate the float knob
    bool foundFloatKnob = false;
    for(size_t i = 0; i < static_cast<size_t>(returnedCount); ++i)
    {
        auto knob = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Knob>(knobData[i].ptr);

        if(knob->knob_id()->str() == "test.float_knob")
        {
            foundFloatKnob = true;

            // Verify description
            EXPECT_STREQ(knob->description()->c_str(), "Test float knob with range 0.0-1.0");

            // Verify default value type and value
            EXPECT_EQ(knob->default_value_type(),
                      hipdnn_data_sdk::data_objects::KnobValue::FloatValue);
            auto floatValue = knob->default_value_as_FloatValue();
            ASSERT_NE(floatValue, nullptr);
            EXPECT_DOUBLE_EQ(floatValue->value(), 0.5);

            // Verify constraint
            EXPECT_EQ(knob->constraint_type(),
                      hipdnn_data_sdk::data_objects::KnobConstraint::FloatConstraint);
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
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        0,
                                        &knobCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    std::vector<hipdnnBackendFlatbufferData_t> knobData(static_cast<size_t>(knobCount));
    int64_t returnedCount = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        knobCount,
                                        &returnedCount,
                                        knobData.data()),
              HIPDNN_STATUS_SUCCESS);

    // Find and validate the string knob
    bool foundStringKnob = false;
    for(size_t i = 0; i < static_cast<size_t>(returnedCount); ++i)
    {
        auto knob = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Knob>(knobData[i].ptr);

        if(knob->knob_id()->str() == "test.string_knob")
        {
            foundStringKnob = true;

            // Verify description
            EXPECT_STREQ(knob->description()->c_str(), "Test string knob with enum values");

            // Verify default value type and value
            EXPECT_EQ(knob->default_value_type(),
                      hipdnn_data_sdk::data_objects::KnobValue::StringValue);
            auto stringValue = knob->default_value_as_StringValue();
            ASSERT_NE(stringValue, nullptr);
            EXPECT_STREQ(stringValue->value()->c_str(), "fast");

            // Verify constraint
            EXPECT_EQ(knob->constraint_type(),
                      hipdnn_data_sdk::data_objects::KnobConstraint::StringConstraint);
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
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        0,
                                        &knobCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    std::vector<hipdnnBackendFlatbufferData_t> knobData(static_cast<size_t>(knobCount));
    int64_t returnedCount = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        knobCount,
                                        &returnedCount,
                                        knobData.data()),
              HIPDNN_STATUS_SUCCESS);

    // Find and validate the deprecated knob
    bool foundDeprecatedKnob = false;
    for(size_t i = 0; i < static_cast<size_t>(returnedCount); ++i)
    {
        auto knob = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Knob>(knobData[i].ptr);

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
                                        &_graph),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetAttribute(
                  _engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &gidx),
              HIPDNN_STATUS_SUCCESS);
    // Not finalizing!

    int64_t knobCount = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        0,
                                        &knobCount,
                                        nullptr),
              HIPDNN_STATUS_NOT_INITIALIZED);
}

// =============================================================================
// EngineConfig Knob Choice Tests (via HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT)
// =============================================================================

using hipdnn_plugin_sdk::KnobSettingFactory;

TEST_F(IntegrationKnobsApi, SetKnobChoiceIntValue)
{
    int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    test_util::createTestEngine(&_engine, &_graph, _handle, gidx, true);

    ASSERT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engine),
              HIPDNN_STATUS_SUCCESS);

    // Set an integer knob value
    auto knobBuffer = KnobSettingFactory::createIntKnobSetting("test.int_knob", 70);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    // Finalize should succeed
    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationKnobsApi, SetKnobChoiceFloatValue)
{
    int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    test_util::createTestEngine(&_engine, &_graph, _handle, gidx, true);

    ASSERT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engine),
              HIPDNN_STATUS_SUCCESS);

    // Set a float knob value
    auto knobBuffer = KnobSettingFactory::createFloatKnobSetting("test.float_knob", 0.75);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    // Finalize should succeed
    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationKnobsApi, SetKnobChoiceStringValue)
{
    int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    test_util::createTestEngine(&_engine, &_graph, _handle, gidx, true);

    ASSERT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engine),
              HIPDNN_STATUS_SUCCESS);

    // Set a string knob value
    auto knobBuffer = KnobSettingFactory::createStringKnobSetting("test.string_knob", "accurate");
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    // Finalize should succeed
    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationKnobsApi, SetKnobChoiceMultipleKnobs)
{
    int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    test_util::createTestEngine(&_engine, &_graph, _handle, gidx, true);

    ASSERT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engine),
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
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
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
    int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    test_util::createTestEngine(&_engine, &_graph, _handle, gidx, true);

    ASSERT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engine),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(IntegrationKnobsApi, SetKnobChoiceInvalidType)
{
    int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    test_util::createTestEngine(&_engine, &_graph, _handle, gidx, true);

    ASSERT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engine),
              HIPDNN_STATUS_SUCCESS);

    auto knobBuffer = KnobSettingFactory::createIntKnobSetting("test.int_knob", 50);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    // Wrong attribute type
    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
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
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(IntegrationKnobsApi, GetMaxWorkspaceSizeWithKnobs)
{
    int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();

    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    test_util::createTestEngine(&_engine, &_graph, _handle, gidx, true);

    ASSERT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engine),
              HIPDNN_STATUS_SUCCESS);

    // Set a knob before finalizing
    auto knobBuffer = KnobSettingFactory::createIntKnobSetting("test.int_knob", 60);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    ASSERT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
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
        int64_t gidx = hipdnn_tests::plugin_constants::engineId<KnobConstraintValidationPlugin>();
        ASSERT_EQ(
            hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
            HIPDNN_STATUS_SUCCESS);
        test_util::createTestEngine(&_engine, &_graph, _handle, gidx, true);
        ASSERT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                            HIPDNN_ATTR_ENGINECFG_ENGINE,
                                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                            1,
                                            &_engine),
                  HIPDNN_STATUS_SUCCESS);
    }
};

TEST_F(IntegrationConstraintValidationApi, IntValueToFloatConstraint)
{
    setupEngineConfig();

    auto knobBuffer
        = hipdnn_plugin_sdk::KnobSettingFactory::createIntKnobSetting("constraint.float_knob", 5);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t executionPlan = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &executionPlan),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(
        hipdnnBackendSetAttribute(
            executionPlan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_handle),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(executionPlan,
                                        HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engineConfig),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(executionPlan), HIPDNN_STATUS_PLUGIN_ERROR);

    if(executionPlan != nullptr)
    {
        hipdnnBackendDestroyDescriptor(executionPlan);
    }
}

TEST_F(IntegrationConstraintValidationApi, FloatValueToIntConstraint)
{
    setupEngineConfig();

    auto knobBuffer = hipdnn_plugin_sdk::KnobSettingFactory::createFloatKnobSetting(
        "constraint.int_knob", 50.0);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t executionPlan = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &executionPlan),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(
        hipdnnBackendSetAttribute(
            executionPlan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_handle),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(executionPlan,
                                        HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engineConfig),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(executionPlan), HIPDNN_STATUS_PLUGIN_ERROR);

    if(executionPlan != nullptr)
    {
        hipdnnBackendDestroyDescriptor(executionPlan);
    }
}

TEST_F(IntegrationConstraintValidationApi, StringValueToIntConstraint)
{
    setupEngineConfig();

    auto knobBuffer = hipdnn_plugin_sdk::KnobSettingFactory::createStringKnobSetting(
        "constraint.int_knob", "50");
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t executionPlan = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &executionPlan),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(
        hipdnnBackendSetAttribute(
            executionPlan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_handle),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(executionPlan,
                                        HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engineConfig),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(executionPlan), HIPDNN_STATUS_PLUGIN_ERROR);

    if(executionPlan != nullptr)
    {
        hipdnnBackendDestroyDescriptor(executionPlan);
    }
}

TEST_F(IntegrationConstraintValidationApi, IntValueToStringConstraint)
{
    setupEngineConfig();

    auto knobBuffer
        = hipdnn_plugin_sdk::KnobSettingFactory::createIntKnobSetting("constraint.string_knob", 0);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t executionPlan = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &executionPlan),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(
        hipdnnBackendSetAttribute(
            executionPlan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_handle),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(executionPlan,
                                        HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engineConfig),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(executionPlan), HIPDNN_STATUS_PLUGIN_ERROR);

    if(executionPlan != nullptr)
    {
        hipdnnBackendDestroyDescriptor(executionPlan);
    }
}

TEST_F(IntegrationConstraintValidationApi, FloatValueToStringConstraint)
{
    setupEngineConfig();

    auto knobBuffer = hipdnn_plugin_sdk::KnobSettingFactory::createFloatKnobSetting(
        "constraint.string_knob", 1.5);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t executionPlan = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &executionPlan),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(
        hipdnnBackendSetAttribute(
            executionPlan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_handle),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(executionPlan,
                                        HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engineConfig),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(executionPlan), HIPDNN_STATUS_PLUGIN_ERROR);

    if(executionPlan != nullptr)
    {
        hipdnnBackendDestroyDescriptor(executionPlan);
    }
}

TEST_F(IntegrationConstraintValidationApi, StringValueToFloatConstraint)
{
    setupEngineConfig();

    auto knobBuffer = hipdnn_plugin_sdk::KnobSettingFactory::createStringKnobSetting(
        "constraint.float_knob", "5.0");
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t executionPlan = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &executionPlan),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(
        hipdnnBackendSetAttribute(
            executionPlan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_handle),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(executionPlan,
                                        HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engineConfig),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(executionPlan), HIPDNN_STATUS_PLUGIN_ERROR);

    if(executionPlan != nullptr)
    {
        hipdnnBackendDestroyDescriptor(executionPlan);
    }
}

TEST_F(IntegrationConstraintValidationApi, CorrectTypesSucceed)
{
    setupEngineConfig();

    auto intKnobBuffer
        = hipdnn_plugin_sdk::KnobSettingFactory::createIntKnobSetting("constraint.int_knob", 50);
    auto floatKnobBuffer = hipdnn_plugin_sdk::KnobSettingFactory::createFloatKnobSetting(
        "constraint.float_knob", 5.0);
    auto stringKnobBuffer = hipdnn_plugin_sdk::KnobSettingFactory::createStringKnobSetting(
        "constraint.string_knob", "beta");

    std::vector<hipdnnBackendFlatbufferData_t> knobDataArray
        = {{intKnobBuffer.data(), intKnobBuffer.size()},
           {floatKnobBuffer.data(), floatKnobBuffer.size()},
           {stringKnobBuffer.data(), stringKnobBuffer.size()}};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        3,
                                        knobDataArray.data()),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t executionPlan = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &executionPlan),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(
        hipdnnBackendSetAttribute(
            executionPlan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_handle),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(executionPlan,
                                        HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engineConfig),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(executionPlan), HIPDNN_STATUS_SUCCESS);

    if(executionPlan != nullptr)
    {
        hipdnnBackendDestroyDescriptor(executionPlan);
    }
}

TEST_F(IntegrationConstraintValidationApi, UnknownKnobIsIgnored)
{
    setupEngineConfig();

    auto knobBuffer
        = hipdnn_plugin_sdk::KnobSettingFactory::createIntKnobSetting("unknown.knob", 123);
    hipdnnBackendFlatbufferData_t knobData = {knobBuffer.data(), knobBuffer.size()};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        1,
                                        &knobData),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t executionPlan = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &executionPlan),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(
        hipdnnBackendSetAttribute(
            executionPlan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_handle),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(executionPlan,
                                        HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engineConfig),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(executionPlan), HIPDNN_STATUS_SUCCESS);

    if(executionPlan != nullptr)
    {
        hipdnnBackendDestroyDescriptor(executionPlan);
    }
}

TEST_F(IntegrationConstraintValidationApi, MixedValidAndInvalidKnobs)
{
    setupEngineConfig();

    auto validKnobBuffer
        = hipdnn_plugin_sdk::KnobSettingFactory::createIntKnobSetting("constraint.int_knob", 50);
    auto invalidKnobBuffer
        = hipdnn_plugin_sdk::KnobSettingFactory::createIntKnobSetting("constraint.float_knob", 5);

    std::vector<hipdnnBackendFlatbufferData_t> knobDataArray
        = {{validKnobBuffer.data(), validKnobBuffer.size()},
           {invalidKnobBuffer.data(), invalidKnobBuffer.size()}};

    EXPECT_EQ(hipdnnBackendSetAttribute(_engineConfig,
                                        HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                        HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                        2,
                                        knobDataArray.data()),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_engineConfig), HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t executionPlan = nullptr;
    ASSERT_EQ(
        hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &executionPlan),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(
        hipdnnBackendSetAttribute(
            executionPlan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_handle),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(executionPlan,
                                        HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engineConfig),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(executionPlan), HIPDNN_STATUS_PLUGIN_ERROR);

    if(executionPlan != nullptr)
    {
        hipdnnBackendDestroyDescriptor(executionPlan);
    }
}

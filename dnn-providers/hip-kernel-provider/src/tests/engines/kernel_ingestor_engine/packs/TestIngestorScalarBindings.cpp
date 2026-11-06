// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

#include "engines/kernel_ingestor_engine/IngestorLaunchRestore.hpp"
#include "engines/kernel_ingestor_engine/IngestorScalarBindings.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanTestUtilities.hpp"

/// @file TestIngestorScalarBindings.cpp
/// The launch values a handler saves for its pass-by-value scalars, and the reader that
/// rebuilds the operands from them. Host only: no device and no plan.
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_plugin_sdk::ScalarOperand;
using hipdnn_plugin_sdk::ScalarValue;
using hipdnn_plugin_sdk::ingestor::MetadataValues;
using hipdnn_plugin_sdk::ingestor::SavedLaunchInputs;

constexpr const char* ROLE = "seed";
constexpr const char* DISPATCH = "hipkernel.test.scalar.v1";
constexpr int64_t UID = 41;

ScalarOperand userSupplied(DataType dataType)
{
    return ScalarOperand{UID, dataType, true, ScalarValue{}};
}

ScalarOperand baked(DataType dataType, ScalarValue value)
{
    return ScalarOperand{UID, dataType, false, value};
}

SavedLaunchInputs inputsWith(const MetadataValues& values)
{
    return SavedLaunchInputs{DISPATCH, values};
}

SavedLaunchInputs putOperand(const ScalarOperand& operand)
{
    MetadataValues values;
    putScalarBinding(values, ROLE, operand);
    return inputsWith(values);
}

// One baked operand per ScalarValue arm. Each value changes when it is read through another
// arm: 1/3 is not a float, 0.1 is not a half or a bfloat16, and 5e9 does not fit an int32.
std::vector<ScalarOperand> bakedOperandsOfEveryArm()
{
    return {baked(DataType::DOUBLE, ScalarValue{1.0 / 3.0}),
            baked(DataType::FLOAT, ScalarValue{0.1F}),
            baked(DataType::HALF, ScalarValue{half(0.1F)}),
            baked(DataType::BFLOAT16, ScalarValue{bfloat16(0.1F)}),
            baked(DataType::INT32, ScalarValue{int32_t{-2000000000}}),
            baked(DataType::INT64, ScalarValue{int64_t{5000000000}}),
            baked(DataType::BOOLEAN, ScalarValue{true})};
}

void expectSameOperand(const ScalarOperand& actual, const ScalarOperand& expected)
{
    EXPECT_EQ(actual.uid, expected.uid);
    EXPECT_EQ(actual.dataType, expected.dataType);
    EXPECT_EQ(actual.isRuntimeUserSupplied, expected.isRuntimeUserSupplied);
    ASSERT_EQ(actual.bakedDefault.index(), expected.bakedDefault.index());
    std::visit(
        [&actual](const auto& want) {
            using T = std::decay_t<decltype(want)>;
            const auto& got = std::get<T>(actual.bakedDefault);
            if constexpr(std::is_same_v<T, half> || std::is_same_v<T, bfloat16>)
            {
                EXPECT_EQ(got.data, want.data);
            }
            else
            {
                EXPECT_EQ(got, want);
            }
        },
        expected.bakedDefault);
}

TEST(TestIngestorScalarBindings, PutsTheFourValuesOfAUserSuppliedScalarAndNoBakedValue)
{
    MetadataValues values;
    putScalarBinding(values, ROLE, userSupplied(DataType::INT64));

    // uid, data_type and user_supplied. A user-supplied scalar has no value to save, so there
    // is no fourth value.
    EXPECT_EQ(values.size(), 3U);
    EXPECT_EQ(values.at("seed.uid"), hipdnn_plugin_sdk::ingestor::MetadataValue{UID});
    EXPECT_EQ(values.at("seed.data_type"),
              hipdnn_plugin_sdk::ingestor::MetadataValue{std::string("INT64")});
    EXPECT_EQ(values.at("seed.user_supplied"), hipdnn_plugin_sdk::ingestor::MetadataValue{true});
    EXPECT_EQ(values.count("seed.baked"), 0U);
}

TEST(TestIngestorScalarBindings, PutsTheBakedValueOfABakedScalarInTheArmOfItsDataType)
{
    for(const auto& operand : bakedOperandsOfEveryArm())
    {
        SCOPED_TRACE(hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(operand.dataType));
        MetadataValues values;
        putScalarBinding(values, ROLE, operand);
        EXPECT_EQ(values.size(), 4U);
        EXPECT_EQ(values.at("seed.user_supplied"),
                  hipdnn_plugin_sdk::ingestor::MetadataValue{false});
        const auto& saved = values.at("seed.baked");
        switch(operand.dataType)
        {
        case DataType::DOUBLE:
        case DataType::FLOAT:
        case DataType::HALF:
        case DataType::BFLOAT16:
            ASSERT_TRUE(std::holds_alternative<double>(saved));
            EXPECT_EQ(std::get<double>(saved), hipdnn_plugin_sdk::toDouble(operand.bakedDefault));
            break;
        case DataType::INT32:
            ASSERT_TRUE(std::holds_alternative<int64_t>(saved));
            EXPECT_EQ(std::get<int64_t>(saved), std::get<int32_t>(operand.bakedDefault));
            break;
        case DataType::INT64:
            ASSERT_TRUE(std::holds_alternative<int64_t>(saved));
            EXPECT_EQ(std::get<int64_t>(saved), std::get<int64_t>(operand.bakedDefault));
            break;
        case DataType::BOOLEAN:
            ASSERT_TRUE(std::holds_alternative<bool>(saved));
            EXPECT_TRUE(std::get<bool>(saved));
            break;
        default:
            FAIL() << "an arm the test does not know";
        }
    }
}

TEST(TestIngestorScalarBindings, RequiresWhatItPut)
{
    auto operands = bakedOperandsOfEveryArm();
    operands.push_back(userSupplied(DataType::FLOAT));
    operands.push_back(userSupplied(DataType::INT64));
    for(const auto& operand : operands)
    {
        SCOPED_TRACE(hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(operand.dataType));
        const auto inputs = putOperand(operand);
        EXPECT_TRUE(hasScalarBinding(inputs, ROLE));
        expectSameOperand(requireScalarBinding(inputs, ROLE), operand);
    }

    EXPECT_FALSE(hasScalarBinding(inputsWith({}), ROLE));
}

TEST(TestIngestorScalarBindings, RefusesABakedValueNextToUserSupplied)
{
    auto inputs = putOperand(userSupplied(DataType::INT64));
    inputs.values["seed.baked"] = int64_t{7};
    serialization::expectIngestorPlanRefusal([&]() { requireScalarBinding(inputs, ROLE); },
                                             HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                                             "seed.baked");
}

TEST(TestIngestorScalarBindings, RefusesAMissingBakedValueOfABakedScalar)
{
    auto inputs = putOperand(baked(DataType::FLOAT, ScalarValue{0.5F}));
    inputs.values.erase("seed.baked");
    serialization::expectIngestorPlanRefusal([&]() { requireScalarBinding(inputs, ROLE); },
                                             HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                                             "seed.baked");
}

TEST(TestIngestorScalarBindings, RefusesAMistypedValue)
{
    const std::vector<std::pair<std::string, hipdnn_plugin_sdk::ingestor::MetadataValue>> damage{
        {"seed.uid", std::string("41")},
        {"seed.data_type", int64_t{6}},
        {"seed.user_supplied", int64_t{0}},
        {"seed.baked", std::string("0.5")},
    };
    for(const auto& [name, wrong] : damage)
    {
        SCOPED_TRACE(name);
        auto inputs = putOperand(baked(DataType::FLOAT, ScalarValue{0.5F}));
        inputs.values[name] = wrong;
        serialization::expectIngestorPlanRefusal([&]() { requireScalarBinding(inputs, ROLE); },
                                                 HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                                                 name);
    }

    // A baked integer outside its data type's range is mistyped for that type too.
    auto inputs = putOperand(baked(DataType::INT32, ScalarValue{int32_t{1}}));
    inputs.values["seed.baked"] = int64_t{5000000000};
    serialization::expectIngestorPlanRefusal([&]() { requireScalarBinding(inputs, ROLE); },
                                             HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                                             "seed.baked");
}

TEST(TestIngestorScalarBindings, RefusesADataTypeOutsideScalarValue)
{
    for(const char* name : {"UINT8", "UNSET", "FP8_E4M3", "NOT_A_DATA_TYPE", ""})
    {
        SCOPED_TRACE(name);
        auto inputs = putOperand(userSupplied(DataType::FLOAT));
        inputs.values["seed.data_type"] = std::string(name);
        serialization::expectIngestorPlanRefusal([&]() { requireScalarBinding(inputs, ROLE); },
                                                 HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                                                 "seed.data_type");
    }
}

TEST(TestIngestorScalarBindings, AnAllowListBuiltFromRolesAcceptsWhatWasPutAndNothingElse)
{
    MetadataValues values;
    putScalarBinding(values, "seed", userSupplied(DataType::INT64));
    putScalarBinding(values, "scale", baked(DataType::FLOAT, ScalarValue{0.5F}));
    std::vector<std::string> names;
    appendScalarBindingNames(names, "seed");
    appendScalarBindingNames(names, "scale");
    requireOnlyLaunchValues(inputsWith(values), names);
    values["offset.uid"] = int64_t{3};
    serialization::expectIngestorPlanRefusal(
        [&]() { requireOnlyLaunchValues(inputsWith(values), names); },
        HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
        "offset.uid");
}

TEST(TestIngestorScalarBindings, RefusesToSaveAScalarWhoseDataTypeHasNoScalarValueArm)
{
    MetadataValues values;
    serialization::expectPluginInternalError(
        [&]() { putScalarBinding(values, ROLE, userSupplied(DataType::UINT8)); },
        "cannot save scalar 'seed'");
    EXPECT_TRUE(values.empty());
}

TEST(TestIngestorScalarBindings, RefusesToSaveABakedValueHeldInAnotherArmThanItsDataType)
{
    // A FLOAT scalar whose baked value is a double.
    MetadataValues values;
    serialization::expectPluginInternalError(
        [&]() { putScalarBinding(values, ROLE, baked(DataType::FLOAT, ScalarValue{1.0})); },
        "does not match a pass-by-value scalar value");
    EXPECT_TRUE(values.empty());
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

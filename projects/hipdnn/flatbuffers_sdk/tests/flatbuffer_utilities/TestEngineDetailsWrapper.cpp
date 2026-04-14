// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>

using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

namespace
{

flatbuffers::FlatBufferBuilder buildValidEngineDetailsBuffer(int64_t engineId)
{
    flatbuffers::FlatBufferBuilder builder;
    auto config = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(builder, engineId);
    builder.Finish(config);
    return builder;
}

flatbuffers::FlatBufferBuilder
    buildEngineDetailsWithKnobs(int64_t engineId, const std::vector<std::string>& knobNames)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Knob>> knobs;
    knobs.reserve(knobNames.size());

    for(const auto& knobIdStr : knobNames)
    {
        auto knobIdStrOffset = builder.CreateString(knobIdStr);
        auto descOffset = builder.CreateString("Description for " + knobIdStr);

        // Create a default int value (required field)
        auto defaultValueOffset
            = hipdnn_flatbuffers_sdk::data_objects::CreateIntValue(builder, int64_t{0});

        hipdnn_flatbuffers_sdk::data_objects::KnobBuilder knobBuilder(builder);
        knobBuilder.add_knob_id(knobIdStrOffset);
        knobBuilder.add_description(descOffset);
        knobBuilder.add_default_value_type(
            hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
        knobBuilder.add_default_value(defaultValueOffset.Union());
        knobs.push_back(knobBuilder.Finish());
    }

    auto knobsVector = builder.CreateVector(knobs);
    auto details
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(builder, engineId, knobsVector);
    builder.Finish(details);
    return builder;
}

} // namespace

TEST(TestEngineDetailsWrapper, InvalidBufferIsNotValid)
{
    const EngineDetailsWrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.isValid());
    EXPECT_THROW(wrapper.engineId(), std::invalid_argument);
    EXPECT_THROW(wrapper.getEngineDetails(), std::invalid_argument);
}

TEST(TestEngineDetailsWrapper, ValidBufferIsValid)
{
    const int64_t testEngineId = 42;
    auto builder = buildValidEngineDetailsBuffer(testEngineId);
    const EngineDetailsWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.engineId(), testEngineId);
    EXPECT_NO_THROW(wrapper.getEngineDetails());
}

TEST(TestEngineDetailsWrapper, CorruptedBufferIsNotValid)
{
    std::vector<uint8_t> buffer(16, 0xFF); // Not a valid flatbuffer
    const EngineDetailsWrapper wrapper(buffer.data(), buffer.size());
    EXPECT_FALSE(wrapper.isValid());
    EXPECT_THROW(wrapper.engineId(), std::invalid_argument);
}

TEST(TestEngineDetailsWrapper, KnobCountEmpty)
{
    auto builder = buildValidEngineDetailsBuffer(42);
    const EngineDetailsWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.knobCount(), 0u);
}

TEST(TestEngineDetailsWrapper, KnobCountNonEmpty)
{
    auto builder = buildEngineDetailsWithKnobs(42, {"KNOB_1", "KNOB_2", "KNOB_3"});
    const EngineDetailsWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.knobCount(), 3u);
}

TEST(TestEngineDetailsWrapper, KnobWrappersEmpty)
{
    auto builder = buildValidEngineDetailsBuffer(42);
    const EngineDetailsWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    const auto& wrappers = wrapper.knobWrappers();
    EXPECT_TRUE(wrappers.empty());
}

TEST(TestEngineDetailsWrapper, KnobWrappersPopulated)
{
    auto builder = buildEngineDetailsWithKnobs(42, {"KNOB_A", "KNOB_B"});
    const EngineDetailsWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    const auto& wrappers = wrapper.knobWrappers();
    EXPECT_EQ(wrappers.size(), 2u);
    EXPECT_EQ(wrappers[0]->knobId(), "KNOB_A");
    EXPECT_EQ(wrappers[1]->knobId(), "KNOB_B");
}

TEST(TestEngineDetailsWrapper, GetKnobByNameFound)
{
    auto builder = buildEngineDetailsWithKnobs(42, {"FIRST_KNOB", "SECOND_KNOB"});
    const EngineDetailsWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& knob = wrapper.getKnobByName("FIRST_KNOB");
    EXPECT_EQ(knob.knobId(), "FIRST_KNOB");

    const auto& knob2 = wrapper.getKnobByName("SECOND_KNOB");
    EXPECT_EQ(knob2.knobId(), "SECOND_KNOB");
}

TEST(TestEngineDetailsWrapper, GetKnobByNameNotFound)
{
    auto builder = buildEngineDetailsWithKnobs(42, {"SOME_KNOB"});
    const EngineDetailsWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_THROW(wrapper.getKnobByName("NONEXISTENT_KNOB"), std::out_of_range);
}

TEST(TestEngineDetailsWrapper, KnobMethodsOnInvalidWrapperThrow)
{
    const EngineDetailsWrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.isValid());

    EXPECT_THROW(wrapper.knobCount(), std::invalid_argument);
    EXPECT_THROW(wrapper.knobWrappers(), std::invalid_argument);
    EXPECT_THROW(wrapper.getKnobByName("test"), std::invalid_argument);
}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

using namespace hipdnn_plugin_sdk;

flatbuffers::FlatBufferBuilder buildValidEngineDetailsBuffer(int64_t engineId)
{
    flatbuffers::FlatBufferBuilder builder;
    auto config = hipdnn_data_sdk::data_objects::CreateEngineDetails(builder, engineId);
    builder.Finish(config);
    return builder;
}

flatbuffers::FlatBufferBuilder
    buildEngineDetailsWithKnobs(int64_t engineId,
                                const std::vector<std::pair<int64_t, std::string>>& knobIdAndNames)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Knob>> knobs;
    knobs.reserve(knobIdAndNames.size());

    for(const auto& [knobId, knobIdStr] : knobIdAndNames)
    {
        auto knobIdStrOffset = builder.CreateString(knobIdStr);
        auto descOffset = builder.CreateString("Description for " + knobIdStr);

        hipdnn_data_sdk::data_objects::KnobBuilder knobBuilder(builder);
        knobBuilder.add_knob_id(knobId);
        knobBuilder.add_knob_id_str(knobIdStrOffset);
        knobBuilder.add_description(descOffset);
        knobs.push_back(knobBuilder.Finish());
    }

    auto knobsVector = builder.CreateVector(knobs);
    auto details
        = hipdnn_data_sdk::data_objects::CreateEngineDetails(builder, engineId, knobsVector);
    builder.Finish(details);
    return builder;
}

TEST(TestEngineDetailsWrapper, InvalidBufferIsNotValid)
{
    EngineDetailsWrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.isValid());
    EXPECT_THROW(wrapper.engineId(), std::invalid_argument);
    EXPECT_THROW(wrapper.getEngineDetails(), std::invalid_argument);
}

TEST(TestEngineDetailsWrapper, ValidBufferIsValid)
{
    int64_t testEngineId = 42;
    auto builder = buildValidEngineDetailsBuffer(testEngineId);
    EngineDetailsWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.engineId(), testEngineId);
    EXPECT_NO_THROW(wrapper.getEngineDetails());
}

TEST(TestEngineDetailsWrapper, CorruptedBufferIsNotValid)
{
    std::vector<uint8_t> buffer(16, 0xFF); // Not a valid flatbuffer
    EngineDetailsWrapper wrapper(buffer.data(), buffer.size());
    EXPECT_FALSE(wrapper.isValid());
    EXPECT_THROW(wrapper.engineId(), std::invalid_argument);
}

TEST(TestEngineDetailsWrapper, KnobCountEmpty)
{
    auto builder = buildValidEngineDetailsBuffer(42);
    EngineDetailsWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.knobCount(), 0u);
}

TEST(TestEngineDetailsWrapper, KnobCountNonEmpty)
{
    auto builder = buildEngineDetailsWithKnobs(42, {{1, "KNOB_1"}, {2, "KNOB_2"}, {3, "KNOB_3"}});
    EngineDetailsWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.knobCount(), 3u);
}

TEST(TestEngineDetailsWrapper, KnobWrappersEmpty)
{
    auto builder = buildValidEngineDetailsBuffer(42);
    EngineDetailsWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    const auto& wrappers = wrapper.knobWrappers();
    EXPECT_TRUE(wrappers.empty());
}

TEST(TestEngineDetailsWrapper, KnobWrappersPopulated)
{
    auto builder = buildEngineDetailsWithKnobs(42, {{100, "KNOB_A"}, {200, "KNOB_B"}});
    EngineDetailsWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    const auto& wrappers = wrapper.knobWrappers();
    EXPECT_EQ(wrappers.size(), 2u);
    EXPECT_EQ(wrappers[0]->knobId(), 100);
    EXPECT_EQ(wrappers[0]->knobIdStr(), "KNOB_A");
    EXPECT_EQ(wrappers[1]->knobId(), 200);
    EXPECT_EQ(wrappers[1]->knobIdStr(), "KNOB_B");
}

TEST(TestEngineDetailsWrapper, GetKnobByIdFound)
{
    auto builder = buildEngineDetailsWithKnobs(42, {{100, "FIRST_KNOB"}, {200, "SECOND_KNOB"}});
    EngineDetailsWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& knob = wrapper.getKnobById(100);
    EXPECT_EQ(knob.knobId(), 100);
    EXPECT_EQ(knob.knobIdStr(), "FIRST_KNOB");

    const auto& knob2 = wrapper.getKnobById(200);
    EXPECT_EQ(knob2.knobId(), 200);
    EXPECT_EQ(knob2.knobIdStr(), "SECOND_KNOB");
}

TEST(TestEngineDetailsWrapper, GetKnobByIdNotFound)
{
    auto builder = buildEngineDetailsWithKnobs(42, {{100, "TEST_KNOB"}});
    EngineDetailsWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_THROW(wrapper.getKnobById(999), std::out_of_range);
}

TEST(TestEngineDetailsWrapper, GetKnobByNameFound)
{
    // Use fnv1aHash to generate a known knob ID from a name
    auto knobName = "MY_TEST_KNOB";
    auto knobId = static_cast<int64_t>(hipdnn_data_sdk::utilities::fnv1aHash(knobName));

    auto builder = buildEngineDetailsWithKnobs(42, {{knobId, knobName}});
    EngineDetailsWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& knob = wrapper.getKnobByName(knobName);
    EXPECT_EQ(knob.knobId(), knobId);
    EXPECT_EQ(knob.knobIdStr(), knobName);
}

TEST(TestEngineDetailsWrapper, GetKnobByNameNotFound)
{
    auto builder = buildEngineDetailsWithKnobs(42, {{100, "SOME_KNOB"}});
    EngineDetailsWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_THROW(wrapper.getKnobByName("NONEXISTENT_KNOB"), std::out_of_range);
}

TEST(TestEngineDetailsWrapper, KnobMethodsOnInvalidWrapperThrow)
{
    EngineDetailsWrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.isValid());

    EXPECT_THROW(wrapper.knobCount(), std::invalid_argument);
    EXPECT_THROW(wrapper.knobWrappers(), std::invalid_argument);
    EXPECT_THROW(wrapper.getKnobById(1), std::invalid_argument);
    EXPECT_THROW(wrapper.getKnobByName("test"), std::invalid_argument);
}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_compatibility/cudnn/detail/knob_wrapper.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace
{
namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

// An INT64-valued knob cannot carry a real FloatConstraint or StringConstraint:
// Knob::tryCreate validates the default value against the constraint and would
// reject it. This stand-in is the only way to reach the "constraint kind is not
// INT" guard with a knob that clears the earlier value-type guard.
class FloatKindConstraint final : public hipdnn_frontend::IConstraint
{
public:
    hipdnn_frontend::ConstraintKind kind() const override
    {
        return hipdnn_frontend::ConstraintKind::FLOAT;
    }

    hipdnn_frontend::Error
        validateKnobSetting(const hipdnn_frontend::KnobSetting& /*setting*/) const override
    {
        return {hipdnn_frontend::ErrorCode::OK, ""};
    }

    std::string toString() const override
    {
        return "FloatKindConstraint{}";
    }
};

hipdnn_frontend::Knob makeNativeKnob(const std::string& knobId,
                                     hipdnn_frontend::KnobValueVariant defaultValue,
                                     std::shared_ptr<hipdnn_frontend::IConstraint> constraint)
{
    auto [error, knob] = hipdnn_frontend::Knob::tryCreate(
        knobId, "test knob", std::move(defaultValue), false, std::move(constraint));
    EXPECT_EQ(error.code, hipdnn_frontend::ErrorCode::OK) << error.err_msg;
    return knob;
}

hipdnn_frontend::Knob
    makeIntNativeKnob(const std::string& knobId, int64_t minValue, int64_t maxValue, int64_t step)
{
    return makeNativeKnob(
        knobId,
        minValue,
        std::make_shared<hipdnn_frontend::IntConstraint>(minValue, maxValue, step));
}

const std::vector<fe::KnobType_t>& mappedKnobTypes()
{
    static const std::vector<fe::KnobType_t> s_types
        = {fe::KnobType_t::SWIZZLE,     fe::KnobType_t::TILE_SIZE,
           fe::KnobType_t::EDGE,        fe::KnobType_t::MULTIPLY,
           fe::KnobType_t::SPLIT_K_BUF, fe::KnobType_t::TILEK,
           fe::KnobType_t::STAGES,      fe::KnobType_t::REDUCTION_MODE,
           fe::KnobType_t::SPLIT_K_SLC, fe::KnobType_t::IDX_MODE,
           fe::KnobType_t::SPECFILT,    fe::KnobType_t::KERNEL_CFG,
           fe::KnobType_t::WORKSPACE,   fe::KnobType_t::TILE_CGA_M,
           fe::KnobType_t::TILE_CGA_N,  fe::KnobType_t::BLOCK_SIZE,
           fe::KnobType_t::OCCUPANCY,   fe::KnobType_t::ARRAY_SIZE_PER_THREAD,
           fe::KnobType_t::SPLIT_COLS,  fe::KnobType_t::TILE_ROWS,
           fe::KnobType_t::TILE_COLS,   fe::KnobType_t::LOAD_SIZE,
           fe::KnobType_t::CTA_COUNT,   fe::KnobType_t::STREAM_K,
           fe::KnobType_t::SPLIT_P_SLC, fe::KnobType_t::TILE_M,
           fe::KnobType_t::TILE_N,      fe::KnobType_t::WARP_SPEC_CFG};
    return s_types;
}

const hipdnn_frontend::KnobSetting*
    findSetting(const std::vector<hipdnn_frontend::KnobSetting>& settings,
                const std::string& knobId)
{
    auto found = std::find_if(settings.begin(), settings.end(), [&knobId](const auto& setting) {
        return setting.knobId() == knobId;
    });
    return found == settings.end() ? nullptr : &(*found);
}

void expectInt64Setting(const std::vector<hipdnn_frontend::KnobSetting>& settings,
                        const std::string& knobId,
                        int64_t expectedValue)
{
    const auto* setting = findSetting(settings, knobId);
    ASSERT_NE(setting, nullptr) << "no setting with id " << knobId;
    const auto* value = std::get_if<int64_t>(&setting->value());
    ASSERT_NE(value, nullptr) << "setting " << knobId << " is not an int64";
    EXPECT_EQ(*value, expectedValue);
}

TEST(TestCudnnShimKnobWrapper, NormalizedKnobIdLowercasesBareIds)
{
    EXPECT_EQ(fe::detail::normalizedKnobId("tile_size"), "tile_size");
    EXPECT_EQ(fe::detail::normalizedKnobId("TILE_SIZE"), "tile_size");
    EXPECT_EQ(fe::detail::normalizedKnobId("Stages"), "stages");
    EXPECT_EQ(fe::detail::normalizedKnobId("not_a_cudnn_knob"), "not_a_cudnn_knob");
}

TEST(TestCudnnShimKnobWrapper, NormalizedKnobIdStripsNamespacePrefix)
{
    EXPECT_EQ(fe::detail::normalizedKnobId("provider.tile_size"), "tile_size");
    EXPECT_EQ(fe::detail::normalizedKnobId("a.b.tile_size"), "tile_size");
    EXPECT_EQ(fe::detail::normalizedKnobId("Provider.TILE_SIZE"), "tile_size");
    EXPECT_EQ(fe::detail::normalizedKnobId("provider."), "");
}

TEST(TestCudnnShimKnobWrapper, NormalizedKnobIdRenamesWorkspaceSizeLimit)
{
    EXPECT_EQ(fe::detail::normalizedKnobId("global.workspace_size_limit"), "workspace");
    EXPECT_EQ(fe::detail::normalizedKnobId("workspace_size_limit"), "workspace");
    EXPECT_EQ(fe::detail::normalizedKnobId("workspace"), "workspace");
}

TEST(TestCudnnShimKnobWrapper, EveryMappedKnobTypeRoundTrips)
{
    std::unordered_set<std::string> seenIds;
    for(const auto knobType : mappedKnobTypes())
    {
        const auto knobId = fe::detail::toHipdnnKnobId(knobType);
        ASSERT_TRUE(knobId.has_value())
            << "KnobType_t " << static_cast<int>(knobType) << " has no hipDNN id";
        EXPECT_TRUE(seenIds.insert(*knobId).second) << "duplicate hipDNN id " << *knobId;

        const auto roundTripped = fe::detail::fromHipdnnKnobId(*knobId);
        ASSERT_TRUE(roundTripped.has_value()) << "hipDNN id " << *knobId << " maps back to nothing";
        EXPECT_EQ(*roundTripped, knobType) << "hipDNN id " << *knobId
                                           << " maps back to a different "
                                              "KnobType_t";
    }
}

TEST(TestCudnnShimKnobWrapper, UnmappedKnobIdsAreRejected)
{
    EXPECT_EQ(fe::detail::toHipdnnKnobId(fe::KnobType_t::NOT_SET), std::nullopt);
    EXPECT_EQ(fe::detail::fromHipdnnKnobId("definitely_not_a_knob"), std::nullopt);
    EXPECT_EQ(fe::detail::fromHipdnnKnobId(""), std::nullopt);
}

TEST(TestCudnnShimKnobWrapper, FromHipdnnKnobIdAcceptsNamespacedIds)
{
    EXPECT_EQ(fe::detail::fromHipdnnKnobId("someprovider.tile_size"), fe::KnobType_t::TILE_SIZE);
    EXPECT_EQ(fe::detail::fromHipdnnKnobId("global.workspace_size_limit"),
              fe::KnobType_t::WORKSPACE);
}

TEST(TestCudnnShimKnobWrapper, ProjectNativeKnobCopiesConstraintBoundsIntoCudnnFields)
{
    const auto nativeKnob = makeIntNativeKnob("provider.tile_size", 8, 1024, 4);

    const auto projected = fe::detail::projectNativeKnob(nativeKnob);
    ASSERT_TRUE(projected.has_value());
    EXPECT_EQ(projected->type, fe::KnobType_t::TILE_SIZE);
    EXPECT_EQ(projected->maxValue, 1024);
    EXPECT_EQ(projected->minValue, 8);
    EXPECT_EQ(projected->stride, 4);
}

TEST(TestCudnnShimKnobWrapper, ProjectNativeKnobOmitsUnmappedKnobId)
{
    const auto nativeKnob = makeIntNativeKnob("provider.not_a_cudnn_knob", 1, 8, 1);

    EXPECT_FALSE(fe::detail::projectNativeKnob(nativeKnob).has_value());
}

TEST(TestCudnnShimKnobWrapper, ProjectNativeKnobOmitsNonInt64ValueType)
{
    const auto nativeKnob = makeNativeKnob(
        "tile_size", std::string("large"), std::make_shared<hipdnn_frontend::StringConstraint>(16));

    EXPECT_EQ(nativeKnob.valueType(), hipdnn_frontend::KnobValueType::STRING);
    EXPECT_FALSE(fe::detail::projectNativeKnob(nativeKnob).has_value());
}

TEST(TestCudnnShimKnobWrapper, ProjectNativeKnobOmitsKnobWithoutConstraint)
{
    // Knob::tryCreate substitutes an EmptyConstraint for a null one, so this is
    // how a caller-supplied null constraint reaches projectNativeKnob.
    const auto nativeKnob = makeNativeKnob("tile_size", int64_t{64}, nullptr);

    ASSERT_NE(nativeKnob.constraint(), nullptr);
    EXPECT_EQ(nativeKnob.constraint()->kind(), hipdnn_frontend::ConstraintKind::EMPTY);
    EXPECT_FALSE(fe::detail::projectNativeKnob(nativeKnob).has_value());
}

TEST(TestCudnnShimKnobWrapper, ProjectNativeKnobOmitsNonIntConstraintKind)
{
    const auto nativeKnob
        = makeNativeKnob("tile_size", int64_t{64}, std::make_shared<FloatKindConstraint>());

    EXPECT_EQ(nativeKnob.valueType(), hipdnn_frontend::KnobValueType::INT64);
    EXPECT_FALSE(fe::detail::projectNativeKnob(nativeKnob).has_value());
}

TEST(TestCudnnShimKnobWrapper, ProjectNativeKnobOmitsIntConstraintWithValidValueList)
{
    const auto nativeKnob = makeNativeKnob("tile_size",
                                           int64_t{2},
                                           std::make_shared<hipdnn_frontend::IntConstraint>(
                                               1, 8, 1, std::unordered_set<int64_t>{1, 2, 4}));

    EXPECT_FALSE(fe::detail::projectNativeKnob(nativeKnob).has_value());
}

TEST(TestCudnnShimKnobWrapper, ProjectNativeKnobsKeepsProjectableKnobsInOrder)
{
    const std::vector<hipdnn_frontend::Knob> nativeKnobs
        = {makeIntNativeKnob("provider.tile_size", 8, 1024, 4),
           makeIntNativeKnob("provider.not_a_cudnn_knob", 1, 8, 1),
           makeNativeKnob("stages",
                          std::string("three"),
                          std::make_shared<hipdnn_frontend::StringConstraint>(16)),
           makeIntNativeKnob("block_size", 32, 256, 32),
           makeIntNativeKnob("global.workspace_size_limit", 0, 4096, 1)};

    std::vector<fe::Knob> cudnnKnobs = {fe::Knob{fe::KnobType_t::EDGE, 1, 1, 1}};
    fe::detail::projectNativeKnobs(nativeKnobs, cudnnKnobs);

    ASSERT_EQ(cudnnKnobs.size(), 3U);
    EXPECT_EQ(cudnnKnobs[0].type, fe::KnobType_t::TILE_SIZE);
    EXPECT_EQ(cudnnKnobs[1].type, fe::KnobType_t::BLOCK_SIZE);
    EXPECT_EQ(cudnnKnobs[2].type, fe::KnobType_t::WORKSPACE);
    EXPECT_EQ(cudnnKnobs[1].minValue, 32);
    EXPECT_EQ(cudnnKnobs[1].maxValue, 256);
    EXPECT_EQ(cudnnKnobs[1].stride, 32);

    const std::vector<hipdnn_frontend::Knob> secondKnobs
        = {makeIntNativeKnob("occupancy", 1, 4, 1)};
    fe::detail::projectNativeKnobs(secondKnobs, cudnnKnobs);

    ASSERT_EQ(cudnnKnobs.size(), 1U);
    EXPECT_EQ(cudnnKnobs[0].type, fe::KnobType_t::OCCUPANCY);
}

TEST(TestCudnnShimKnobWrapper, MakeNativeKnobSettingsMapsEveryChoice)
{
    const std::unordered_map<fe::KnobType_t, int64_t> choices = {{fe::KnobType_t::TILE_SIZE, 256},
                                                                 {fe::KnobType_t::WORKSPACE, 4096},
                                                                 {fe::KnobType_t::STAGES, 3}};

    std::vector<hipdnn_frontend::KnobSetting> nativeSettings
        = {{"stale", int64_t{1}}, {"also_stale", int64_t{2}}};
    const auto error = fe::detail::makeNativeKnobSettings(choices, nativeSettings);

    ASSERT_TRUE(error.is_good()) << error.get_message();
    ASSERT_EQ(nativeSettings.size(), 3U);
    EXPECT_EQ(findSetting(nativeSettings, "stale"), nullptr);
    expectInt64Setting(nativeSettings, "tile_size", 256);
    expectInt64Setting(nativeSettings, "global.workspace_size_limit", 4096);
    expectInt64Setting(nativeSettings, "stages", 3);
}

TEST(TestCudnnShimKnobWrapper, MakeNativeKnobSettingsRejectsUnmappedKnobType)
{
    const std::unordered_map<fe::KnobType_t, int64_t> choices = {{fe::KnobType_t::NOT_SET, 1}};

    std::vector<hipdnn_frontend::KnobSetting> nativeSettings;
    const auto error = fe::detail::makeNativeKnobSettings(choices, nativeSettings);

    ASSERT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), fe::error_code_t::INVALID_VALUE);
}

} // namespace

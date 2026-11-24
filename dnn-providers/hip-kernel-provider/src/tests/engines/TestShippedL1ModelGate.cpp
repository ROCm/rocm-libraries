// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "tests/engines/ShippedL1ModelGate.hpp"

#include <array>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace hip_kernel_provider::shipped_model_gate
{
namespace
{

constexpr std::string_view REVISION = "provider/engine/0123456789abcdef";
constexpr std::string_view GFX942_ID = "aaaaaaaa-0000-4000-8000-000000000001";
constexpr std::string_view GFX950_ID = "bbbbbbbb-0000-4000-8000-000000000002";

constexpr std::array<std::pair<std::string_view, std::string_view>, 2> DECLARED{{
    {"gfx942", GFX942_ID},
    {"gfx950", GFX950_ID},
}};

std::vector<ShippedModel> shippedModels()
{
    return {
        {"d/predict_engine/gfx942/m.uhd.json", GFX942_ID.data(), REVISION.data()},
        {"d/predict_engine/gfx950/m.uhd.json", GFX950_ID.data(), REVISION.data()},
    };
}

std::vector<const char*> stagedFiles()
{
    return {"predict_engine/gfx942/m.uhd.json",
            "predict_engine/gfx942/m.bin",
            "predict_engine/gfx950/m.uhd.json",
            "predict_engine/gfx950/m.bin"};
}

::testing::AssertionResult gate(const std::vector<ShippedModel>& shipped,
                                const std::vector<const char*>& staged)
{
    return shippedL1ModelsBindThisBuild(
        "ENGINE", "recipe.cmake", shipped, staged, "dump", DECLARED, REVISION);
}

/// The failure text of a gate that must fail; empty when it passes.
std::string failureOf(const ::testing::AssertionResult& result)
{
    return result ? std::string() : std::string(result.message());
}

TEST(TestShippedL1ModelGate, ModelsThatShipForTheirArchitectureAndRecordTheRevisionPass)
{
    EXPECT_TRUE(gate(shippedModels(), stagedFiles()));
}

/// A declared id matches an uppercase spelling of the shipped id.
TEST(TestShippedL1ModelGate, DeclaredIdsMatchWhateverTheCaseOfTheShippedId)
{
    auto shipped = shippedModels();
    const std::string upper = "AAAAAAAA-0000-4000-8000-000000000001";
    shipped[0].id = upper.c_str();
    EXPECT_TRUE(gate(shipped, stagedFiles()));
}

/// An install carries only some architectures, so the test that runs from one cannot see a
/// model that was never staged; the list that stages them is the same everywhere.
TEST(TestShippedL1ModelGate, ADeclaredArchitectureWithoutAStagedArtifactFails)
{
    auto staged = stagedFiles();
    staged.erase(staged.begin() + 3);
    const std::string failure = failureOf(gate(shippedModels(), staged));
    EXPECT_NE(failure.find("gfx950/*.bin"), std::string::npos) << failure;
    EXPECT_EQ(failure.find("gfx942"), std::string::npos) << failure;
}

TEST(TestShippedL1ModelGate, ADeclaredArchitectureWithoutAStagedDescriptorFails)
{
    auto staged = stagedFiles();
    staged.erase(staged.begin());
    const std::string failure = failureOf(gate(shippedModels(), staged));
    EXPECT_NE(failure.find("gfx942/*.uhd.json"), std::string::npos) << failure;
    EXPECT_EQ(failure.find("gfx950"), std::string::npos) << failure;
}

TEST(TestShippedL1ModelGate, ADeclaredArchitectureWithNothingStagedFails)
{
    const std::vector<const char*> staged{"predict_engine/gfx942/m.uhd.json",
                                          "predict_engine/gfx942/m.bin"};
    const std::string failure = failureOf(gate(shippedModels(), staged));
    EXPECT_NE(failure.find("gfx950/*.uhd.json and /*.bin"), std::string::npos) << failure;
}

TEST(TestShippedL1ModelGate, AModelDeclaredForAnotherArchitectureThanItShipsForFails)
{
    auto shipped = shippedModels();
    std::swap(shipped[0].id, shipped[1].id);
    const std::string failure = failureOf(gate(shipped, stagedFiles()));
    EXPECT_NE(failure.find("it ships as d/predict_engine/gfx"), std::string::npos) << failure;
}

TEST(TestShippedL1ModelGate, AModelRecordingAnotherRevisionFails)
{
    auto shipped = shippedModels();
    shipped[1].revision = "provider/engine/fedcba9876543210";
    const std::string failure = failureOf(gate(shipped, stagedFiles()));
    EXPECT_NE(failure.find("records selector_revision 'provider/engine/fedcba9876543210'"),
              std::string::npos)
        << failure;
}

TEST(TestShippedL1ModelGate, ShippedModelsThatNoDeclaredIdNamesCheckNothingAndFail)
{
    auto shipped = shippedModels();
    shipped[0].id = "cccccccc-0000-4000-8000-000000000003";
    shipped[1].id = "dddddddd-0000-4000-8000-000000000004";
    EXPECT_FALSE(gate(shipped, stagedFiles()));
}

} // namespace
} // namespace hip_kernel_provider::shipped_model_gate

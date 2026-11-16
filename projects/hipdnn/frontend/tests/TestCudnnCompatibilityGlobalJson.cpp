// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// A consumer's own global `json`, declared before and after the compatibility layer umbrella, must not collide
// with the compatibility layer's alias. Built only when the frontend keeps its JSON library.
#include <nlohmann/json.hpp>

// NOLINTNEXTLINE(readability-identifier-naming)
using json = nlohmann::ordered_json;

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include <type_traits>

// NOLINTNEXTLINE(readability-identifier-naming)
using json = nlohmann::ordered_json;

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

static_assert(std::is_same_v<::json, nlohmann::ordered_json>);
static_assert(std::is_same_v<fe::json, nlohmann::json>);

TEST(TestCudnnCompatibilityGlobalJson, ConsumerGlobalJsonKeepsItsOwnType)
{
    const json ordered{{"b", 1}, {"a", 2}};
    EXPECT_EQ(ordered.dump(), R"({"b":1,"a":2})");
}

} // namespace

/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "stinkytofu/pipeline/BackendRegistry.hpp"

using namespace stinkytofu;

// --- makeArchKey ---

TEST(BackendRegistryTest, MakeArchKeyGfx1250) {
    EXPECT_EQ(BackendRegistry::makeArchKey({12, 5, 0}), "gfx1250");
}

TEST(BackendRegistryTest, MakeArchKeyGfx942) {
    EXPECT_EQ(BackendRegistry::makeArchKey({9, 4, 2}), "gfx942");
}

TEST(BackendRegistryTest, MakeArchKeyHexStepping) {
    EXPECT_EQ(BackendRegistry::makeArchKey({9, 0, 10}), "gfx90a");
    EXPECT_EQ(BackendRegistry::makeArchKey({9, 0, 15}), "gfx90f");
}

TEST(BackendRegistryTest, MakeArchKeyClampsBadStepping) {
    EXPECT_EQ(BackendRegistry::makeArchKey({12, 5, 99}), "gfx1250");
    EXPECT_EQ(BackendRegistry::makeArchKey({12, 5, -1}), "gfx1250");
}

// --- parseArchKey ---

TEST(BackendRegistryTest, ParseArchKeyGfx1250) {
    std::array<int, 3> out{};
    EXPECT_TRUE(BackendRegistry::parseArchKey("gfx1250", out));
    EXPECT_EQ(out[0], 12);
    EXPECT_EQ(out[1], 5);
    EXPECT_EQ(out[2], 0);
}

TEST(BackendRegistryTest, ParseArchKeyGfx942) {
    std::array<int, 3> out{};
    EXPECT_TRUE(BackendRegistry::parseArchKey("gfx942", out));
    EXPECT_EQ(out[0], 9);
    EXPECT_EQ(out[1], 4);
    EXPECT_EQ(out[2], 2);
}

TEST(BackendRegistryTest, ParseArchKeyHexStepping) {
    std::array<int, 3> out{};
    EXPECT_TRUE(BackendRegistry::parseArchKey("gfx90a", out));
    EXPECT_EQ(out[0], 9);
    EXPECT_EQ(out[1], 0);
    EXPECT_EQ(out[2], 10);

    EXPECT_TRUE(BackendRegistry::parseArchKey("gfx90A", out));
    EXPECT_EQ(out[2], 10);
}

TEST(BackendRegistryTest, ParseArchKeyInvalid) {
    std::array<int, 3> out{};
    EXPECT_FALSE(BackendRegistry::parseArchKey("", out));
    EXPECT_FALSE(BackendRegistry::parseArchKey("gfx", out));
    EXPECT_FALSE(BackendRegistry::parseArchKey("gfx12", out));
    EXPECT_FALSE(BackendRegistry::parseArchKey("amd1250", out));
    EXPECT_FALSE(BackendRegistry::parseArchKey("gfxABC", out));
}

// --- round-trip ---

TEST(BackendRegistryTest, RoundTrip) {
    std::array<int, 3> original = {12, 5, 0};
    std::string key = BackendRegistry::makeArchKey(original);
    std::array<int, 3> parsed{};
    ASSERT_TRUE(BackendRegistry::parseArchKey(key, parsed));
    EXPECT_EQ(original, parsed);
}

TEST(BackendRegistryTest, RoundTripHex) {
    std::array<int, 3> original = {9, 0, 10};
    std::string key = BackendRegistry::makeArchKey(original);
    std::array<int, 3> parsed{};
    ASSERT_TRUE(BackendRegistry::parseArchKey(key, parsed));
    EXPECT_EQ(original, parsed);
}

// --- getRegisteredArchKeys (sorted) ---

TEST(BackendRegistryTest, RegisteredKeysAreSorted) {
    auto keys = BackendRegistry::getRegisteredArchKeys();
    for (size_t i = 1; i < keys.size(); ++i) {
        EXPECT_LE(keys[i - 1], keys[i]) << "Keys should be sorted";
    }
}

// --- entry lifetime and concurrency ---

// A fake arch keeps these tests away from the real backends' entries.
constexpr std::array<int, 3> kFakeArch = {99, 9, 9};

TEST(BackendRegistryTest, PipelineOutlivesClearAndReRegistration) {
    BackendRegistry::setArchPipeline(kFakeArch, {nullptr, {"first"}});
    auto pipeline = BackendRegistry::getArchPipeline(kFakeArch);
    ASSERT_NE(pipeline, nullptr);
    BackendRegistry::setArchPipeline(kFakeArch, {nullptr, {"second"}});
    BackendRegistry::clearArch(kFakeArch);
    EXPECT_EQ(pipeline->groupNames, std::vector<std::string>{"first"});
    EXPECT_EQ(BackendRegistry::getArchPipeline(kFakeArch), nullptr);
}

TEST(BackendRegistryTest, ConcurrentRegisterLookupAndClear) {
    std::atomic<bool> bad{false};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([t, &bad] {
            const std::array<int, 3> arch = {90 + t, 1, 2};
            for (int i = 0; i < 2000; ++i) {
                BackendRegistry::setArchPipeline(arch, {nullptr, {std::to_string(i)}});
                BackendRegistry::setArchPipeline(kFakeArch, {nullptr, {"shared"}});
                if (auto p = BackendRegistry::getArchPipeline(kFakeArch))
                    if (p->groupNames.size() != 1 || p->groupNames[0] != "shared") bad = true;
                (void)BackendRegistry::getRegisteredArchKeys();
                BackendRegistry::clearArch(kFakeArch);
                BackendRegistry::clearArch(arch);
            }
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_FALSE(bad.load());
    EXPECT_EQ(BackendRegistry::getArchPipeline(kFakeArch), nullptr);
}

TEST(BackendRegistryTest, ParseArchKeyOverflowingMajorIsInvalid) {
    std::array<int, 3> out{};
    EXPECT_FALSE(BackendRegistry::parseArchKey("gfx99999999999950", out));
}

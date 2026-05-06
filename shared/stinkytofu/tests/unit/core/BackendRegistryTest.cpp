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

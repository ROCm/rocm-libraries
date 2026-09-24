// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_test_sdk/utilities/AsanDefaultSuppressions.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <string_view>

using namespace hipdnn_test_sdk::utilities;
using hipdnn_test_sdk::utilities::asan::environBufferHasFlag;

namespace
{

// Builds a /proc/self/environ-style block: entries joined by NUL, each entry NUL-terminated.
std::string environBlock(std::initializer_list<std::string_view> entries)
{
    std::string block;
    for(const auto& entry : entries)
    {
        block.append(entry);
        block.push_back('\0');
    }
    return block;
}

bool hasFlag(const std::string& block, const char* name)
{
    return environBufferHasFlag(block.data(), static_cast<long>(block.size()), name);
}

// Feeds `block` to the scanner in fixed-size pieces, the way the real reader receives it from
// successive read() calls.
bool hasFlagChunked(const std::string& block, const char* name, std::size_t chunkSize)
{
    hipdnn_test_sdk::utilities::asan::EnvironFlagScanner scanner(name);
    for(std::size_t offset = 0; offset < block.size(); offset += chunkSize)
    {
        const std::size_t remaining = block.size() - offset;
        const std::size_t take = remaining < chunkSize ? remaining : chunkSize;
        scanner.feed(block.data() + offset, static_cast<long>(take));
    }
    return scanner.found();
}

// Runs hasFlagChunked at every chunk size from 1 to the whole block, so every possible split point
// is covered -- including splits inside the name, inside the '=' boundary, and between entries.
// Returns the first chunk size that disagrees with `expected`, or 0 if all agree.
std::size_t firstDisagreeingChunkSize(const std::string& block, const char* name, bool expected)
{
    for(std::size_t chunk = 1; chunk <= block.size(); ++chunk)
    {
        if(hasFlagChunked(block, name, chunk) != expected)
        {
            return chunk;
        }
    }
    return 0;
}

} // namespace

TEST(TestAsanDefaultSuppressions, RejectsNullBuffer)
{
    EXPECT_FALSE(environBufferHasFlag(nullptr, 10, "FOO"));
}

TEST(TestAsanDefaultSuppressions, RejectsNullName)
{
    const std::string block = environBlock({"FOO=1"});
    EXPECT_FALSE(environBufferHasFlag(block.data(), static_cast<long>(block.size()), nullptr));
}

TEST(TestAsanDefaultSuppressions, RejectsEmptyBuffer)
{
    const std::string block = environBlock({"FOO=1"});
    EXPECT_FALSE(environBufferHasFlag(block.data(), 0, "FOO"));
}

TEST(TestAsanDefaultSuppressions, RejectsNegativeLength)
{
    // syscall() reports a failed read as -1; that must not be treated as a length.
    const std::string block = environBlock({"FOO=1"});
    EXPECT_FALSE(environBufferHasFlag(block.data(), -1, "FOO"));
}

TEST(TestAsanDefaultSuppressions, MatchesSoleEntry)
{
    EXPECT_TRUE(hasFlag(environBlock({"FOO=1"}), "FOO"));
}

TEST(TestAsanDefaultSuppressions, MatchesFirstEntry)
{
    EXPECT_TRUE(hasFlag(environBlock({"FOO=1", "BAR=2", "BAZ=3"}), "FOO"));
}

TEST(TestAsanDefaultSuppressions, MatchesMiddleEntry)
{
    EXPECT_TRUE(hasFlag(environBlock({"FOO=1", "BAR=2", "BAZ=3"}), "BAR"));
}

TEST(TestAsanDefaultSuppressions, MatchesLastEntry)
{
    EXPECT_TRUE(hasFlag(environBlock({"FOO=1", "BAR=2", "BAZ=3"}), "BAZ"));
}

TEST(TestAsanDefaultSuppressions, MatchesEmptyValue)
{
    // The override is set for its presence, not its value, so "VAR=" must count as set.
    EXPECT_TRUE(hasFlag(environBlock({"FOO="}), "FOO"));
}

TEST(TestAsanDefaultSuppressions, MatchesValueContainingEquals)
{
    EXPECT_TRUE(hasFlag(environBlock({"FOO=a=b=c"}), "FOO"));
}

TEST(TestAsanDefaultSuppressions, RejectsAbsentName)
{
    EXPECT_FALSE(hasFlag(environBlock({"FOO=1", "BAR=2"}), "BAZ"));
}

TEST(TestAsanDefaultSuppressions, RejectsEntryWhoseNameExtendsTheSearchName)
{
    // The '=' requirement is what stops HIPDNN_ASAN_NO_DEFAULT_SUPPRESSIONS_EXTRA from enabling the
    // override.
    EXPECT_FALSE(hasFlag(environBlock({"FOO_BAR=1"}), "FOO"));
}

TEST(TestAsanDefaultSuppressions, RejectsEntryWhoseNameIsAPrefixOfTheSearchName)
{
    EXPECT_FALSE(hasFlag(environBlock({"FOO=1"}), "FOO_BAR"));
}

TEST(TestAsanDefaultSuppressions, RejectsNameAppearingOnlyInsideAValue)
{
    EXPECT_FALSE(hasFlag(environBlock({"BAR=FOO=1"}), "FOO"));
}

TEST(TestAsanDefaultSuppressions, RejectsEntryWithNoAssignment)
{
    EXPECT_FALSE(hasFlag(environBlock({"FOO"}), "FOO"));
}

TEST(TestAsanDefaultSuppressions, MatchesAfterAnEntryWithNoAssignment)
{
    EXPECT_TRUE(hasFlag(environBlock({"FOO", "FOO=1"}), "FOO"));
}

TEST(TestAsanDefaultSuppressions, ScansTruncatedBlockWithoutOverrunning)
{
    // A block cut mid-entry has no trailing NUL. Every read is bounded by the length, so the
    // partial tail is simply not a match.
    const std::string block = environBlock({"FOO=1", "BARBAR=2"});
    const auto truncated = static_cast<long>(block.size() - 4);

    EXPECT_TRUE(environBufferHasFlag(block.data(), truncated, "FOO"));
    EXPECT_FALSE(environBufferHasFlag(block.data(), truncated, "BARBAR"));
}

TEST(TestAsanDefaultSuppressions, RejectsTrailingNameWithoutItsEquals)
{
    const char* block = "FOO";
    EXPECT_FALSE(environBufferHasFlag(block, 3, "FOO"));
}

// --- block-boundary behaviour -------------------------------------------------------------
//
// The reader consumes /proc/self/environ in fixed-size blocks, so a variable can be split across
// two reads. These sweep every possible split point rather than sampling a few.

TEST(TestAsanDefaultSuppressions, FindsFlagAtEveryChunkBoundary)
{
    const std::string block = environBlock({"FIRST=1", "HIPDNN_TARGET=yes", "LAST=3"});
    EXPECT_EQ(firstDisagreeingChunkSize(block, "HIPDNN_TARGET", true), 0U);
}

TEST(TestAsanDefaultSuppressions, RejectsAbsentFlagAtEveryChunkBoundary)
{
    const std::string block = environBlock({"FIRST=1", "HIPDNN_TARGET=yes", "LAST=3"});
    EXPECT_EQ(firstDisagreeingChunkSize(block, "HIPDNN_MISSING", false), 0U);
}

TEST(TestAsanDefaultSuppressions, DoesNotMatchNameInsideAValueAtAnyChunkBoundary)
{
    // A value that happens to contain the variable's own text does not count as the variable being
    // set, wherever the chunks happen to divide the block.
    const std::string block = environBlock({"SOMEVAR=HIPDNN_TARGET=1", "LAST=3"});
    EXPECT_FALSE(hasFlag(block, "HIPDNN_TARGET"));
    EXPECT_EQ(firstDisagreeingChunkSize(block, "HIPDNN_TARGET", false), 0U);
}

TEST(TestAsanDefaultSuppressions, DoesNotMatchLongerVariableAtAnyChunkBoundary)
{
    const std::string block = environBlock({"HIPDNN_TARGET_EXTRA=1"});
    EXPECT_EQ(firstDisagreeingChunkSize(block, "HIPDNN_TARGET", false), 0U);
}

TEST(TestAsanDefaultSuppressions, FindsFlagSplitOneByteAtATime)
{
    // The worst case: every byte arrives in its own read.
    const std::string block = environBlock({"A=1", "HIPDNN_TARGET=", "B=2"});
    EXPECT_TRUE(hasFlagChunked(block, "HIPDNN_TARGET", 1));
}

TEST(TestAsanDefaultSuppressions, FindsFlagBeyondTheFirstBlock)
{
    // The block size bounds stack use, not reach: a variable sitting well past it is still found.
    std::string block;
    for(int i = 0; i < 400; ++i)
    {
        block.append("PADDING_VARIABLE_NUMBER_" + std::to_string(i) + "=0123456789ABCDEF");
        block.push_back('\0');
    }
    ASSERT_GT(block.size(), hipdnn_test_sdk::utilities::asan::K_ENVIRON_BLOCK_SIZE * 4);

    block.append("HIPDNN_TARGET=1");
    block.push_back('\0');

    EXPECT_TRUE(hasFlag(block, "HIPDNN_TARGET"));
    EXPECT_TRUE(hasFlagChunked(
        block, "HIPDNN_TARGET", hipdnn_test_sdk::utilities::asan::K_ENVIRON_BLOCK_SIZE));
}

TEST(TestAsanDefaultSuppressions, ScannerIsIdempotentOnceFound)
{
    hipdnn_test_sdk::utilities::asan::EnvironFlagScanner scanner("FOO");
    const std::string block = environBlock({"FOO=1"});
    scanner.feed(block.data(), static_cast<long>(block.size()));
    ASSERT_TRUE(scanner.found());

    // Later blocks must not clear a match already made.
    const std::string more = environBlock({"BAR=2"});
    scanner.feed(more.data(), static_cast<long>(more.size()));
    EXPECT_TRUE(scanner.found());
}

TEST(TestAsanDefaultSuppressions, ScannerToleratesEmptyAndNullChunks)
{
    hipdnn_test_sdk::utilities::asan::EnvironFlagScanner scanner("FOO");
    scanner.feed(nullptr, 10);
    scanner.feed("FOO=1", 0);
    scanner.feed("FOO=1", -1);
    EXPECT_FALSE(scanner.found());

    scanner.feed("FOO=1", 5);
    EXPECT_TRUE(scanner.found());
}

#if defined(__linux__)

TEST(TestAsanDefaultSuppressions, ReadsTheRealEnvironment)
{
    if(std::getenv("PATH") == nullptr)
    {
        GTEST_SKIP() << "PATH is not set in this environment";
    }
    EXPECT_TRUE(asan::environmentFlagSet("PATH"));
}

TEST(TestAsanDefaultSuppressions, RealEnvironmentDoesNotReportAnAbsentVariable)
{
    EXPECT_FALSE(asan::environmentFlagSet("HIPDNN_ASAN_DEFINITELY_NOT_SET_XYZZY"));
}

TEST(TestAsanDefaultSuppressions, SetenvDoesNotChangeTheSnapshot)
{
    // /proc/self/environ is what the kernel recorded at exec, so setenv() cannot enable the
    // override from inside a running process. Anyone tempted to test the override that way needs a
    // subprocess instead.
    const ScopedEnvironmentVariableSetter setter("HIPDNN_ASAN_SET_AFTER_EXEC_XYZZY", "1");

    ASSERT_NE(std::getenv("HIPDNN_ASAN_SET_AFTER_EXEC_XYZZY"), nullptr);
    EXPECT_FALSE(asan::environmentFlagSet("HIPDNN_ASAN_SET_AFTER_EXEC_XYZZY"));
}

TEST(TestAsanDefaultSuppressions, ReturnsEmptyWhenTheOverrideIsSetBeforeExec)
{
    // Only reachable when the whole process was started with the variable set, which is the
    // documented way to use it. The suite is run a second time that way so this case is covered.
    if(!asan::environmentFlagSet(asan::K_DISABLE_VARIABLE))
    {
        GTEST_SKIP() << "set " << asan::K_DISABLE_VARIABLE << " before exec to exercise this path";
    }
    EXPECT_STREQ(asan::defaultSuppressions(), "");
}

#endif // __linux__

TEST(TestAsanDefaultSuppressions, SuppressionTextNamesTheTensileEntryPoint)
{
    EXPECT_STREQ(asan::K_DEFAULT_SUPPRESSIONS, "interceptor_via_fun:*findBestKeyMatch*\n");
}

TEST(TestAsanDefaultSuppressions, ReturnsTheSuppressionTextByDefault)
{
#if defined(__linux__)
    if(asan::environmentFlagSet(asan::K_DISABLE_VARIABLE))
    {
        GTEST_SKIP() << asan::K_DISABLE_VARIABLE << " is set, so the suppressions are disabled";
    }
#endif
    EXPECT_STREQ(asan::defaultSuppressions(), asan::K_DEFAULT_SUPPRESSIONS);
}

#ifdef ADDRESS_SANITIZER

// Declared rather than included: the ASan runtime owns this symbol and ships no header for it.
// The reserved name is the runtime's, not ours -- it only resolves under exactly this spelling.
// NOLINTNEXTLINE(readability-identifier-naming,bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
extern "C" const char* __asan_default_suppressions();

TEST(TestAsanDefaultSuppressions, AsanRuntimeHookResolvesToOurDefinition)
{
    // The ASan runtime provides a weak default returning "". Seeing our text here proves the
    // strong definition in this executable is the one the runtime found -- the property that the
    // whole mechanism depends on and that no other test covers.
    EXPECT_STREQ(__asan_default_suppressions(), asan::defaultSuppressions());

#if defined(__linux__)
    if(asan::environmentFlagSet(asan::K_DISABLE_VARIABLE))
    {
        GTEST_SKIP() << asan::K_DISABLE_VARIABLE << " is set, so the suppressions are disabled";
    }
#endif
    EXPECT_STREQ(__asan_default_suppressions(), asan::K_DEFAULT_SUPPRESSIONS);
}

#endif // ADDRESS_SANITIZER

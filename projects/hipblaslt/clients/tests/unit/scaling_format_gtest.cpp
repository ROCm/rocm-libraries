// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for the hipblaslt_scaling_format helpers in
// hipblaslt_datatype2string.hpp: scaleDataType, isBlockScaling, blockSize, and
// scaleBufferSize. All four are pure functions of their arguments, so they are
// compiled into both hipblaslt-test and the standalone hipblaslt-test-hostunit
// binary. See parser_gtest.cpp for the dual-target rationale and for why the
// names carry both a HostUnit suite prefix and a smoke_ test prefix.
//
// scaleBufferSize is the interesting one: it applies two independent paddings
// with integer division, so it gets explicit boundary cases rather than a
// restatement of its own formula.
//
// The enumerator list below is kept complete by the compiler, not by review: it
// is expanded into a tripwire switch with no `default:` arm and -Wswitch promoted
// to an error, so adding a scaling format without listing it here breaks this
// file's build. See parser_gtest.cpp for the same pattern applied to the parsers.

#include "hipblaslt_datatype2string.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace
{
    // scale_type is a hipDataType held as int, because HIP_R_8F_E5M3_EXT is a
    // plain `int const` rather than a hipDataType enumerator.
    struct ScalingFormatExpectation
    {
        const char*              name;
        hipblaslt_scaling_format format;
        int                      scale_type;
        bool                     block_scaling;
        int                      block_size;
    };

    // Every hipblaslt_scaling_format enumerator, named once, with what the three
    // mapping helpers are expected to report for it. Expanded twice: into the
    // expectation table the tests iterate, and into a tripwire switch that makes
    // the compiler reject a list missing an enumerator.
    //
    // Keep in sync with hipblaslt_scaling_format.hpp. The tripwire enforces that
    // every enumerator appears; the expected values in each row are the part that
    // still needs a human.
    //
    // Columns: format, scale type, is block scaling, block size.
#define HIPBLASLT_TEST_SCALING_FORMATS(X)          \
    X(none, HIP_R_8F_UE8M0, false, 1)              \
    X(Scalar, HIP_R_8F_UE8M0, false, 1)            \
    X(Vector, HIP_R_8F_UE8M0, false, 1)            \
    X(Block_32_UE8M0, HIP_R_8F_UE8M0, true, 32)    \
    X(Block_16_UE8M0, HIP_R_8F_UE8M0, true, 16)    \
    X(Block_32_UE4M3, HIP_R_8F_E4M3, true, 32)     \
    X(Block_16_UE4M3, HIP_R_8F_E4M3, true, 16)     \
    X(Block_32_UE5M3, HIP_R_8F_E5M3_EXT, true, 32) \
    X(Block_16_UE5M3, HIP_R_8F_E5M3_EXT, true, 16) \
    X(Block_32_UE8M0_32_8_EXT, HIP_R_8F_UE8M0, true, 32)

    const std::vector<ScalingFormatExpectation>& known_scaling_formats()
    {
#define X(fmt, scale_type, block_scaling, block_size) \
    {#fmt, hipblaslt_scaling_format::fmt, scale_type, block_scaling, block_size},
        static const std::vector<ScalingFormatExpectation> formats
            = {HIPBLASLT_TEST_SCALING_FORMATS(X)};
#undef X
        return formats;
    }

    // Tripwire: never called. Adding an enumerator to hipblaslt_scaling_format
    // without adding it to the list above fails to compile here, in the
    // developer's own build. The switch has no `default:` arm and -Wswitch is
    // promoted to an error, so the enumerator is reported by name.
    //
    // This matters more here than for the parsers. All three mapping helpers in
    // hipblaslt_datatype2string.hpp have `default:` arms returning a plausible
    // value, so an unhandled enumerator produces a wrong answer rather than an
    // error, and the runtime guard below cannot tell a new non-block format apart
    // from an unhandled value. This check has no such blind spot.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wswitch"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#endif

    [[maybe_unused]] void assert_scaling_formats_listed(hipblaslt_scaling_format value)
    {
        switch(value)
        {
#define X(fmt, scale_type, block_scaling, block_size) case hipblaslt_scaling_format::fmt:
            HIPBLASLT_TEST_SCALING_FORMATS(X)
#undef X
            break;
        }
    }

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

    bool is_registered_format(int value)
    {
        for(const auto& entry : known_scaling_formats())
        {
            if(static_cast<int>(entry.format) == value)
                return true;
        }
        return false;
    }

    // Derived from the table rather than hand-written so it cannot go stale as
    // enumerators are added; the margin carries the sweep past the largest one.
    int scaling_sweep_max()
    {
        int max_registered = 0;
        for(const auto& entry : known_scaling_formats())
            max_registered = std::max(max_registered, static_cast<int>(entry.format));
        return max_registered + 64;
    }

    struct ScaleBufferSizeCase
    {
        const char*              what;
        hipblaslt_scaling_format format;
        int64_t                  data_row;
        int64_t                  data_col;
        size_t                   expected;
    };
}

TEST(HostUnitScalingFormat, smoke_RegisteredFormatsMapAsExpected)
{
    for(const auto& e : known_scaling_formats())
    {
        EXPECT_EQ(static_cast<int>(scaleDataType(e.format)), e.scale_type)
            << "scaleDataType(" << e.name << ")";
        EXPECT_EQ(isBlockScaling(e.format), e.block_scaling) << "isBlockScaling(" << e.name << ")";
        EXPECT_EQ(blockSize(e.format), e.block_size) << "blockSize(" << e.name << ")";
    }
}

// All three mapping helpers have `default:` arms returning a plausible value
// (HIP_R_8F_UE8M0, false, 1), so an unhandled enumerator produces a wrong answer
// rather than an error. Require every unregistered value to look exactly like
// that default, which pins the behavior the helpers promise for inputs outside
// the enum.
//
// This is a secondary guard, and a partial one: none/Scalar/Vector are registered
// but also produce the default triple, so at runtime a new *non-block* format is
// indistinguishable from an unhandled value. That blind spot is covered at compile
// time by assert_scaling_formats_listed above, which does not care what the
// helpers return.
TEST(HostUnitScalingFormat, smoke_EveryEnumeratorIsRegistered)
{
    const int sweep_max = scaling_sweep_max();
    for(int value = 0; value <= sweep_max; ++value)
    {
        if(is_registered_format(value))
            continue; // asserted in RegisteredFormatsMapAsExpected

        const auto  format = static_cast<hipblaslt_scaling_format>(value);
        const char* hint   = "is not registered in known_scaling_formats(). Add it there so the "
                             "mapping helpers are covered.";

        EXPECT_FALSE(isBlockScaling(format))
            << "hipblaslt_scaling_format value " << value << " " << hint;
        EXPECT_EQ(blockSize(format), 1)
            << "hipblaslt_scaling_format value " << value << " " << hint;
        EXPECT_EQ(static_cast<int>(scaleDataType(format)), static_cast<int>(HIP_R_8F_UE8M0))
            << "hipblaslt_scaling_format value " << value << " " << hint;
    }
}

// scaleBufferSize pads rows to a multiple of 8 after dividing by the block size,
// and pads columns to a multiple of 32. Each case below sits on one side of a
// padding boundary so an off-by-one in either division shows up.
TEST(HostUnitScalingFormat, smoke_ScaleBufferSizePadsRowsAndColumns)
{
    const std::vector<ScaleBufferSizeCase> cases = {
        // Block size 32: 8 scale rows exactly covers 256 data rows, 16 beyond that.
        {"bs=32 baseline", hipblaslt_scaling_format::Block_32_UE8M0, 256, 32, 8 * 32},
        {"bs=32 one data row past 8 scale rows",
         hipblaslt_scaling_format::Block_32_UE8M0,
         257,
         32,
         16 * 32},
        // Columns pad to 32 independently of the block size.
        {"one data col past 32 scale cols",
         hipblaslt_scaling_format::Block_32_UE8M0,
         256,
         33,
         8 * 64},
        // Block size 16 halves the row capacity: 8 scale rows covers 128 data rows.
        {"bs=16 baseline", hipblaslt_scaling_format::Block_16_UE8M0, 128, 32, 8 * 32},
        {"bs=16 one data row past 8 scale rows",
         hipblaslt_scaling_format::Block_16_UE8M0,
         129,
         32,
         16 * 32},
        // Non-block formats use block size 1, so rows pad straight to a multiple of 8.
        {"bs=1 baseline", hipblaslt_scaling_format::none, 8, 32, 8 * 32},
        {"bs=1 one data row past 8 scale rows", hipblaslt_scaling_format::none, 9, 32, 16 * 32},
        // A single element still costs one padded row block and one padded column block.
        {"minimum non-empty", hipblaslt_scaling_format::Block_32_UE8M0, 1, 1, 8 * 32},
        // Either dimension being zero collapses the whole buffer.
        {"zero rows", hipblaslt_scaling_format::Block_32_UE8M0, 0, 64, 0},
        {"zero cols", hipblaslt_scaling_format::Block_32_UE8M0, 64, 0, 0},
        {"zero both", hipblaslt_scaling_format::Block_32_UE8M0, 0, 0, 0},
    };

    for(const auto& c : cases)
    {
        EXPECT_EQ(scaleBufferSize(c.data_row, c.data_col, c.format), c.expected)
            << c.what << ": scaleBufferSize(" << c.data_row << ", " << c.data_col << ", ...)";
    }
}

// The pre-swizzled EXT format shares block size 32 with Block_32_UE8M0, so it
// must size buffers identically. This pins the pair together: they are wired
// through separate switch arms and could drift.
TEST(HostUnitScalingFormat, smoke_PreSwizzledExtMatchesPlainBlock32)
{
    for(int64_t rows : {1, 31, 32, 256, 257, 1024})
    {
        for(int64_t cols : {1, 32, 33, 512})
        {
            EXPECT_EQ(
                scaleBufferSize(rows, cols, hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT),
                scaleBufferSize(rows, cols, hipblaslt_scaling_format::Block_32_UE8M0))
                << "rows=" << rows << " cols=" << cols;
        }
    }
}

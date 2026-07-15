// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "dispatcher/AotInstance.hpp"
#include "plans/LaunchAbi.hpp"

namespace rocke_client::launch
{
namespace
{

using dispatcher::ArgKind;
using dispatcher::GridAxis;
using dispatcher::GridFormula;
using dispatcher::GridValue;
using dispatcher::KernelArgument;
using dispatcher::LaunchBindings;
using dispatcher::ScalarType;
using hipdnn_plugin_sdk::HipdnnPluginException;

KernelArgument makePointer(std::string name)
{
    KernelArgument arg;
    arg.name = std::move(name);
    arg.kind = ArgKind::POINTER;
    return arg;
}

KernelArgument makeScalar(std::string name, ScalarType type)
{
    KernelArgument arg;
    arg.name = std::move(name);
    arg.kind = ArgKind::SCALAR;
    arg.scalarType = type;
    return arg;
}

GridAxis valueAxis(std::int64_t literal)
{
    GridAxis axis;
    axis.kind = GridAxis::Kind::VALUE;
    axis.value = GridValue{.symbol = std::nullopt, .literal = literal};
    return axis;
}

GridAxis symbolAxis(std::string symbol)
{
    GridAxis axis;
    axis.kind = GridAxis::Kind::VALUE;
    axis.value = GridValue{.symbol = std::move(symbol), .literal = 0};
    return axis;
}

GridAxis ceilDivAxis(std::string numerator, std::string denominator)
{
    GridAxis axis;
    axis.kind = GridAxis::Kind::CEIL_DIV;
    axis.numerator = GridValue{.symbol = std::move(numerator), .literal = 0};
    axis.denominator = GridValue{.symbol = std::move(denominator), .literal = 0};
    return axis;
}

TEST(TestLaunchAbi, PacksArgsWithAlignmentPadding)
{
    // i32 (4B) then a pointer (8B, align 8): the pointer must land at offset 8,
    // so 4 bytes of zero padding sit between them.
    const std::vector<KernelArgument> signature{
        makeScalar("count", ScalarType::I32),
        makePointer("ptr"),
    };
    const std::unordered_map<std::string, ScalarValue> values{
        {"count", ScalarValue{std::int64_t{7}}},
        {"ptr", ScalarValue{std::uint64_t{0xDEADBEEFULL}}},
    };

    const auto packed = packArgs(signature, values);
    ASSERT_EQ(packed.size(), 16u);

    std::int32_t count = 0;
    std::memcpy(&count, packed.data(), sizeof(count));
    EXPECT_EQ(count, 7);

    for(std::size_t i = 4; i < 8; ++i)
    {
        EXPECT_EQ(packed[i], std::byte{0}) << "padding byte " << i << " must be zero";
    }

    std::uint64_t ptr = 0;
    std::memcpy(&ptr, packed.data() + 8, sizeof(ptr));
    EXPECT_EQ(ptr, 0xDEADBEEFULL);
}

TEST(TestLaunchAbi, PacksF32Scalar)
{
    const std::vector<KernelArgument> signature{makeScalar("scale", ScalarType::F32)};
    const std::unordered_map<std::string, ScalarValue> values{{"scale", ScalarValue{1.5F}}};

    const auto packed = packArgs(signature, values);
    ASSERT_EQ(packed.size(), 4u);
    float scale = 0.0F;
    std::memcpy(&scale, packed.data(), sizeof(scale));
    EXPECT_FLOAT_EQ(scale, 1.5F);
}

TEST(TestLaunchAbi, MissingArgValueThrows)
{
    const std::vector<KernelArgument> signature{makePointer("ptr")};
    EXPECT_THROW(packArgs(signature, {}), HipdnnPluginException);
}

TEST(TestLaunchAbi, EvalGridResolvesCeilDivAndSymbols)
{
    GridFormula formula;
    formula.x = ceilDivAxis("seqlen_q", "block_size_q"); // ceil(65/64) = 2
    formula.y = valueAxis(3);
    formula.z = symbolAxis("batch");

    const std::unordered_map<std::string, std::int64_t> symbols{
        {"seqlen_q", 65}, {"block_size_q", 64}, {"batch", 5}};
    const auto grid = evalGrid(formula, symbols);
    EXPECT_EQ(grid[0], 2u);
    EXPECT_EQ(grid[1], 3u);
    EXPECT_EQ(grid[2], 5u);
}

TEST(TestLaunchAbi, EvalGridUnknownSymbolThrows)
{
    GridFormula formula;
    formula.x = symbolAxis("does_not_exist");
    formula.y = valueAxis(1);
    formula.z = valueAxis(1);
    EXPECT_THROW(evalGrid(formula, {}), HipdnnPluginException);
}

TEST(TestLaunchAbi, EvalGridRejectsNonPositiveDenominator)
{
    GridFormula formula;
    formula.x = ceilDivAxis("seqlen_q", "block_size_q");
    formula.y = valueAxis(1);
    formula.z = valueAxis(1);

    const std::unordered_map<std::string, std::int64_t> symbols{
        {"seqlen_q", 64}, {"block_size_q", 0}}; // denominator <= 0
    EXPECT_THROW(evalGrid(formula, symbols), HipdnnPluginException);
}

TEST(TestLaunchAbi, BindArgsResolvesPointersAndScalarsByName)
{
    // A pointer + one scalar of each supported width, bound by name; bindArgs must
    // resolve pointer uids to device addresses and pass scalars through unchanged.
    const std::vector<KernelArgument> signature{
        makePointer("A"),
        makeScalar("scale", ScalarType::F32),
        makeScalar("n", ScalarType::I32),
    };

    std::byte slotA{};
    void* const aPtr = &slotA;
    const std::unordered_map<std::int64_t, void*> ptrs{{42, aPtr}};

    LaunchBindings bindings;
    bindings.pointerUids = {{"A", 42}};
    bindings.scalars = {{"scale", ScalarValue{0.5F}}, {"n", ScalarValue{std::int64_t{7}}}};

    const auto values = bindArgs(signature, bindings, ptrs);
    const auto packed = packArgs(signature, values);

    ASSERT_EQ(packed.size(), 16u); // 8B ptr, then f32 + i32 at offsets 8/12
    std::uint64_t ptrBits = 0;
    std::memcpy(&ptrBits, packed.data(), sizeof(ptrBits));
    EXPECT_EQ(ptrBits, static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(aPtr)));
    float scale = 0.0F;
    std::memcpy(&scale, packed.data() + 8, sizeof(scale));
    EXPECT_FLOAT_EQ(scale, 0.5F);
    std::int32_t n = 0;
    std::memcpy(&n, packed.data() + 12, sizeof(n));
    EXPECT_EQ(n, 7);
}

TEST(TestLaunchAbi, BindArgsRejectsUnboundPointer)
{
    const std::vector<KernelArgument> signature{makePointer("W")};
    EXPECT_THROW(bindArgs(signature, LaunchBindings{}, {}), HipdnnPluginException);
}

TEST(TestLaunchAbi, BindArgsRejectsUnboundScalar)
{
    const std::vector<KernelArgument> signature{makeScalar("mystery", ScalarType::I32)};
    EXPECT_THROW(bindArgs(signature, LaunchBindings{}, {}), HipdnnPluginException);
}

TEST(TestLaunchAbi, BindArgsRejectsMissingDeviceBuffer)
{
    const std::vector<KernelArgument> signature{makePointer("A")};
    LaunchBindings bindings;
    bindings.pointerUids = {{"A", 7}}; // uid 7 has no entry in the (empty) device map
    EXPECT_THROW(bindArgs(signature, bindings, {}), HipdnnPluginException);
}

} // namespace
} // namespace rocke_client::launch

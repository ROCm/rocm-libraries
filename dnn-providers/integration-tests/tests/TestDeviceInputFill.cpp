// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The NONZERO, ZEROS, LARGE, SUBNORMAL and SPECIAL kinds generated on the device by
// DeviceInputFiller (rocRAND draws shaped by a hipRTC kernel). Each test fills a tensor large
// enough for the device, copies it back and checks the kind's value contract on every element.
// The tests that need a device skip without one; the declines and the environment override do
// not touch HIP.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/input-init/FillInputs.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests;

namespace
{

namespace types = hipdnn_data_sdk::types;
using hipdnn_data_sdk::utilities::Tensor;

constexpr unsigned char kPoison = 0xA5;

#define SKIP_IF_NO_DEVICE_FILL()                                     \
    do                                                               \
    {                                                                \
        SKIP_IF_NO_DEVICES();                                        \
        if(!DeviceInputFiller::isSupported())                        \
        {                                                            \
            GTEST_SKIP() << "rocRAND not available. Skipping test."; \
        }                                                            \
    } while(false)

struct Layout
{
    std::string name;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
};

// Every layout is large enough, by allocation, for the device. The padded ones have bytes
// between rows that belong to no element; the transposed one is packed in another order.
std::vector<Layout> layouts()
{
    return {
        {"packed", {256, 256}, {256, 1}},
        {"row padding", {130, 100}, {128, 1}},
        {"transposed", {256, 256}, {1, 256}},
        {"rank 4 padded", {4, 3, 40, 40}, {5760, 1920, 48, 1}},
    };
}

std::size_t elementsOf(const std::vector<int64_t>& dims)
{
    std::size_t count = 1;
    for(const auto d : dims)
    {
        count *= static_cast<std::size_t>(d);
    }
    return count;
}

// The offset of every element, in row-major order.
std::vector<std::size_t> elementOffsets(const std::vector<int64_t>& dims,
                                        const std::vector<int64_t>& strides)
{
    std::vector<std::size_t> offsets;
    offsets.reserve(elementsOf(dims));
    std::vector<int64_t> index(dims.size(), 0);
    for(std::size_t n = 0; n < elementsOf(dims); ++n)
    {
        int64_t offset = 0;
        for(std::size_t axis = 0; axis < dims.size(); ++axis)
        {
            offset += index[axis] * strides[axis];
        }
        offsets.push_back(static_cast<std::size_t>(offset));
        for(std::size_t axis = dims.size(); axis-- > 0;)
        {
            if(++index[axis] < dims[axis])
            {
                break;
            }
            index[axis] = 0;
        }
    }
    return offsets;
}

template <typename T>
float asFloat(T v)
{
    return static_cast<float>(v);
}

template <typename T>
struct DeviceFilled
{
    bool accepted = false;
    std::size_t elementSpace = 0;
    std::vector<T> elements; // row-major
    std::vector<T> allocation; // every slot, gaps included
    std::vector<bool> isElement; // per slot of `allocation`
};

// Fills a tensor of the layout with `recipe` through a DeviceInputFiller, after setting every
// byte of its allocation to kPoison, and copies the whole allocation back.
template <typename T>
DeviceFilled<T> deviceFill(const FillRecipe& recipe, unsigned int seed, const Layout& layout)
{
    Tensor<T> tensor(layout.dims, layout.strides);
    DeviceFilled<T> out;
    out.elementSpace = tensor.elementSpace();
    std::memset(tensor.rawHostData(), kPoison, out.elementSpace * sizeof(T));
    tensor.markHostModified();

    DeviceInputFiller device;
    out.accepted = device.tryFill(tensor, recipe, seed);
    device.waitForFills();

    const T* host = tensor.memory().hostData();
    out.allocation.assign(host, host + out.elementSpace);
    out.isElement.assign(out.elementSpace, false);
    for(const auto offset : elementOffsets(layout.dims, layout.strides))
    {
        out.elements.push_back(out.allocation.at(offset));
        out.isElement.at(offset) = true;
    }
    return out;
}

template <typename T>
bool sameBytes(const std::vector<T>& a, const std::vector<T>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

// Whether every slot that belongs to no element still holds the poison.
template <typename T>
bool gapsHoldThePoison(const DeviceFilled<T>& filled)
{
    for(std::size_t slot = 0; slot < filled.elementSpace; ++slot)
    {
        if(filled.isElement[slot])
        {
            continue;
        }
        unsigned char bytes[sizeof(T)]; // NOLINT(modernize-avoid-c-arrays)
        std::memcpy(bytes, &filled.allocation[slot], sizeof(T));
        for(const auto b : bytes)
        {
            if(b != kPoison)
            {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
struct TestGpuDeviceInputFillKinds : public ::testing::Test
{
};

using FillTypes = ::testing::Types<float,
                                   types::half,
                                   types::bfloat16,
                                   types::fp8_e4m3,
                                   types::fp8_e5m2,
                                   types::fp8_e4m3_fnuz,
                                   types::fp8_e5m2_fnuz>;
TYPED_TEST_SUITE(TestGpuDeviceInputFillKinds, FillTypes, );

} // namespace

TYPED_TEST(TestGpuDeviceInputFillKinds, NonzeroStaysInRangeAndNeverYieldsZero)
{
    SKIP_IF_NO_DEVICE_FILL();
    for(const auto& layout : layouts())
    {
        SCOPED_TRACE(layout.name);
        const auto filled = deviceFill<TypeParam>(FillRecipe::nonzero(-2.0f, 3.0f), 5, layout);
        ASSERT_TRUE(filled.accepted);
        std::size_t negative = 0;
        std::size_t positive = 0;
        for(const auto v : filled.elements)
        {
            const float f = asFloat(v);
            ASSERT_NE(f, 0.0f);
            ASSERT_GE(f, -2.0f);
            ASSERT_LE(f, 3.0f);
            (f < 0.0f ? negative : positive) += 1;
        }
        EXPECT_GT(negative, filled.elements.size() / 4);
        EXPECT_GT(positive, filled.elements.size() / 4);
        EXPECT_TRUE(gapsHoldThePoison(filled));
    }
}

TYPED_TEST(TestGpuDeviceInputFillKinds, NonzeroOverANarrowRangeAroundZeroNeverYieldsZero)
{
    SKIP_IF_NO_DEVICE_FILL();
    // Inside the rounding bin around zero of the narrow types, so the first draw rounds to
    // zero for most of their elements, which the redraw and the endpoint must repair. The
    // endpoint itself may round away from zero to the type's smallest nonzero value, past the
    // range's edge, as it does on the host.
    constexpr float kEdge = 0.001f;
    const float roundedEdge = std::max(kEdge, asFloat(static_cast<TypeParam>(kEdge)));
    const auto filled = deviceFill<TypeParam>(
        FillRecipe::nonzero(-kEdge, kEdge), 9, Layout{"packed", {256, 256}, {256, 1}});
    if(!filled.accepted)
    {
        GTEST_SKIP() << "every value of the range rounds to zero in this data type";
    }
    for(const auto v : filled.elements)
    {
        ASSERT_NE(asFloat(v), 0.0f);
        ASSERT_LE(std::fabs(asFloat(v)), roundedEdge);
    }
}

TYPED_TEST(TestGpuDeviceInputFillKinds, ZerosWritesEveryElementAndNoGap)
{
    SKIP_IF_NO_DEVICE_FILL();
    for(const auto& layout : layouts())
    {
        SCOPED_TRACE(layout.name);
        const auto filled = deviceFill<TypeParam>(FillRecipe::zeros(), 1, layout);
        ASSERT_TRUE(filled.accepted);
        for(const auto v : filled.elements)
        {
            ASSERT_EQ(asFloat(v), 0.0f);
        }
        EXPECT_TRUE(gapsHoldThePoison(filled));
    }
}

TYPED_TEST(TestGpuDeviceInputFillKinds, LargeStaysInTheUpperHalfOfTheScaleWithBothSigns)
{
    SKIP_IF_NO_DEVICE_FILL();
    for(const auto& layout : layouts())
    {
        SCOPED_TRACE(layout.name);
        const auto filled = deviceFill<TypeParam>(FillRecipe::large(64.0f), 3, layout);
        ASSERT_TRUE(filled.accepted);
        std::size_t negative = 0;
        for(const auto v : filled.elements)
        {
            const float magnitude = std::fabs(asFloat(v));
            ASSERT_GE(magnitude, 32.0f);
            ASSERT_LE(magnitude, 64.0f);
            negative += asFloat(v) < 0.0f ? 1 : 0;
        }
        EXPECT_GT(negative, filled.elements.size() / 4);
        EXPECT_LT(negative, filled.elements.size() * 3 / 4);
        EXPECT_TRUE(gapsHoldThePoison(filled));
    }
}

TYPED_TEST(TestGpuDeviceInputFillKinds, LargeBeyondTheTypesMaximumIsCappedAtIt)
{
    SKIP_IF_NO_DEVICE_FILL();
    const float maxFinite = asFloat(std::numeric_limits<TypeParam>::max());
    const float scale = maxFinite < 1.0e10f ? 1.0e10f : maxFinite;
    const auto filled = deviceFill<TypeParam>(
        FillRecipe::large(scale), 4, Layout{"packed", {256, 256}, {256, 1}});
    ASSERT_TRUE(filled.accepted);
    for(const auto v : filled.elements)
    {
        const float f = asFloat(v);
        ASSERT_TRUE(std::isfinite(f)) << f;
        ASSERT_LE(std::fabs(f), maxFinite);
        ASSERT_GE(std::fabs(f), maxFinite / 2.0f);
    }
}

TYPED_TEST(TestGpuDeviceInputFillKinds, SubnormalFillsEveryElementBelowTheSmallestNormal)
{
    SKIP_IF_NO_DEVICE_FILL();
    const double smallestNormal = asFloat(std::numeric_limits<TypeParam>::min());
    const double smallest = asFloat(std::numeric_limits<TypeParam>::denorm_min());
    const auto codes = static_cast<std::size_t>(smallestNormal / smallest) - 1;
    for(const auto& layout : layouts())
    {
        SCOPED_TRACE(layout.name);
        const auto filled = deviceFill<TypeParam>(FillRecipe::subnormal(), 6, layout);
        ASSERT_TRUE(filled.accepted);
        std::set<double> magnitudes;
        std::size_t negative = 0;
        for(const auto v : filled.elements)
        {
            const double f = asFloat(v);
            ASSERT_NE(f, 0.0);
            ASSERT_LT(std::fabs(f), smallestNormal);
            magnitudes.insert(std::fabs(f));
            negative += f < 0.0 ? 1 : 0;
        }
        // Every subnormal code is reached when there are few, and many are when there are
        // millions.
        EXPECT_GE(magnitudes.size(), std::min<std::size_t>(codes, 64));
        EXPECT_GT(negative, filled.elements.size() / 4);
        EXPECT_LT(negative, filled.elements.size() * 3 / 4);
        EXPECT_TRUE(gapsHoldThePoison(filled));
    }
}

TYPED_TEST(TestGpuDeviceInputFillKinds, SpecialNanPlantsExactlyTheRequestedShare)
{
    SKIP_IF_NO_DEVICE_FILL();
    for(const auto& layout : layouts())
    {
        SCOPED_TRACE(layout.name);
        const auto recipe
            = FillRecipe::nonFinite(FillRecipe::SpecialValue::QUIET_NAN, 0.1f, -2.0f, 2.0f);
        const auto filled = deviceFill<TypeParam>(recipe, 8, layout);
        ASSERT_TRUE(filled.accepted);
        const auto expected = static_cast<std::size_t>(
            std::llround(static_cast<double>(0.1f) * static_cast<double>(elementsOf(layout.dims))));
        std::size_t planted = 0;
        for(const auto v : filled.elements)
        {
            const float f = asFloat(v);
            if(std::isnan(f))
            {
                ++planted;
                continue;
            }
            ASSERT_GE(f, -2.0f);
            ASSERT_LE(f, 2.0f);
        }
        EXPECT_EQ(planted, expected);
        EXPECT_TRUE(gapsHoldThePoison(filled));
    }
}

TYPED_TEST(TestGpuDeviceInputFillKinds, SpecialPlantsAtLeastOneNonFiniteElementWhateverTheShare)
{
    SKIP_IF_NO_DEVICE_FILL();
    // The planted guarantee the gauntlet spaces rely on: a share too small to round to one
    // element still plants one.
    const auto recipe = FillRecipe::nonFinite(FillRecipe::SpecialValue::QUIET_NAN, 1.0e-9f);
    const auto filled = deviceFill<TypeParam>(recipe, 2, Layout{"packed", {256, 256}, {256, 1}});
    ASSERT_TRUE(filled.accepted);
    std::size_t planted = 0;
    for(const auto v : filled.elements)
    {
        planted += std::isnan(asFloat(v)) ? 1 : 0;
    }
    EXPECT_EQ(planted, 1u);
}

TYPED_TEST(TestGpuDeviceInputFillKinds, SpecialOfEveryElementLeavesNoFiniteValue)
{
    SKIP_IF_NO_DEVICE_FILL();
    const auto recipe = FillRecipe::nonFinite(FillRecipe::SpecialValue::QUIET_NAN, 1.0f);
    const auto filled
        = deviceFill<TypeParam>(recipe, 2, Layout{"row padding", {130, 100}, {128, 1}});
    ASSERT_TRUE(filled.accepted);
    for(const auto v : filled.elements)
    {
        ASSERT_TRUE(std::isnan(asFloat(v)));
    }
    EXPECT_TRUE(gapsHoldThePoison(filled));
}

TYPED_TEST(TestGpuDeviceInputFillKinds, SpecialInfinitiesKeepTheirSignOrAreDeclined)
{
    SKIP_IF_NO_DEVICE_FILL();
    const Layout layout{"packed", {256, 256}, {256, 1}};
    const struct
    {
        FillRecipe::SpecialValue special;
        float sign;
    } cases[] = {{FillRecipe::SpecialValue::POS_INF, 1.0f},
                 {FillRecipe::SpecialValue::NEG_INF, -1.0f}}; // NOLINT(modernize-avoid-c-arrays)
    for(const auto& c : cases)
    {
        const auto filled
            = deviceFill<TypeParam>(FillRecipe::nonFinite(c.special, 0.05f), 3, layout);
        if(!std::numeric_limits<TypeParam>::has_infinity)
        {
            // The type holds no infinity: the device leaves it to the host, which refuses it.
            EXPECT_FALSE(filled.accepted);
            continue;
        }
        ASSERT_TRUE(filled.accepted);
        const auto expected = static_cast<std::size_t>(std::llround(
            static_cast<double>(0.05f) * static_cast<double>(elementsOf(layout.dims))));
        std::size_t planted = 0;
        for(const auto v : filled.elements)
        {
            const float f = asFloat(v);
            if(!std::isfinite(f))
            {
                ASSERT_TRUE(std::isinf(f));
                ASSERT_EQ(std::signbit(f), c.sign < 0.0f);
                ++planted;
            }
        }
        EXPECT_EQ(planted, expected);
    }
}

TYPED_TEST(TestGpuDeviceInputFillKinds, SpecialPositionsDependOnTheSeed)
{
    SKIP_IF_NO_DEVICE_FILL();
    const Layout layout{"packed", {256, 256}, {256, 1}};
    const auto recipe = FillRecipe::nonFinite(FillRecipe::SpecialValue::QUIET_NAN, 0.25f);
    const auto positions = [&](unsigned int seed) {
        const auto filled = deviceFill<TypeParam>(recipe, seed, layout);
        EXPECT_TRUE(filled.accepted);
        std::vector<bool> nan;
        for(const auto v : filled.elements)
        {
            nan.push_back(std::isnan(asFloat(v)));
        }
        return nan;
    };
    EXPECT_EQ(positions(1), positions(1));
    EXPECT_NE(positions(1), positions(2));
}

TYPED_TEST(TestGpuDeviceInputFillKinds, ASeedGivesTheSameBytesAndAnotherSeedOthers)
{
    SKIP_IF_NO_DEVICE_FILL();
    const Layout layout{"packed", {256, 256}, {256, 1}};
    const auto recipe = FillRecipe::nonzero(-1.0f, 1.0f);
    const auto first = deviceFill<TypeParam>(recipe, 11, layout);
    const auto again = deviceFill<TypeParam>(recipe, 11, layout);
    const auto other = deviceFill<TypeParam>(recipe, 12, layout);
    ASSERT_TRUE(first.accepted && again.accepted && other.accepted);
    EXPECT_TRUE(sameBytes(first.elements, again.elements));
    EXPECT_FALSE(sameBytes(first.elements, other.elements));
}

namespace
{

struct GraphHolder
{
    flatbuffers::FlatBufferBuilder builder;
    const Graph* graph = nullptr;
};

// A one-node matmul graph whose input a (uid 1) has `aType`, `dims` and `strides`.
GraphHolder buildMatmulGraph(DataType aType,
                             const std::vector<int64_t>& dims,
                             const std::vector<int64_t>& strides)
{
    GraphHolder r;
    auto& b = r.builder;
    const std::vector<int64_t> smallDims = {2, 3};
    const std::vector<int64_t> smallStrides = {3, 1};

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(b, 1, "a", aType, &strides, &dims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 2, "b", DataType::FLOAT, &smallStrides, &smallDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 3, "c", DataType::FLOAT, &smallStrides, &smallDims));
    auto matmul = CreateMatmulAttributes(b, 1, 2, 3);
    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(
        b, "matmul", DataType::FLOAT, NodeAttributes::MatmulAttributes, matmul.Union()));
    auto graph = CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);
    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

// Fills input a of the matmul graph with `recipe` and `device`, and returns the result with
// the elements of a read back.
FillResult fillA(const FillRecipe& recipe,
                 DeviceInputFiller* device,
                 const std::vector<int64_t>& dims,
                 std::vector<float>& elements)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(std::size_t axis = dims.size() - 1; axis-- > 0;)
    {
        strides[axis] = strides[axis + 1] * dims[axis + 1];
    }
    const auto gr = buildMatmulGraph(DataType::FLOAT, dims, strides);
    InputTensorMap inputs;
    inputs[1] = std::make_shared<Tensor<float>>(dims, strides);
    InputFillRecipes recipes;
    recipes.set(1, recipe).setSeed(1, 3);
    const auto result = fillInputs(*gr.graph, inputs, {1}, recipes, device);
    if(result.filled)
    {
        const auto* data = static_cast<const float*>(inputs.at(1)->rawHostData());
        elements.assign(data, data + inputs.at(1)->elementCount());
    }
    return result;
}

} // namespace

// The kinds land on the device only when the filler is given and the tensor is big enough, and
// the outcome counts them, which is what the report's fill-path note is built from.
TEST(TestGpuDeviceInputFill, EveryKindIsCountedAsADeviceFillOfALargeTensor)
{
    SKIP_IF_NO_DEVICE_FILL();
    const std::vector<FillRecipe> recipes
        = {FillRecipe::nonzero(-1.0f, 1.0f),
           FillRecipe::zeros(),
           FillRecipe::large(8.0f),
           FillRecipe::subnormal(),
           FillRecipe::nonFinite(FillRecipe::SpecialValue::QUIET_NAN, 0.25f)};
    for(const auto& recipe : recipes)
    {
        SCOPED_TRACE(InputFillRecipes::kindToString(recipe.kind));
        std::vector<float> onDevice;
        DeviceInputFiller device;
        const auto deviceResult = fillA(recipe, &device, {256, 256}, onDevice);
        ASSERT_TRUE(deviceResult.filled) << deviceResult.reason;
        EXPECT_EQ(deviceResult.deviceFilled, 1u);
        EXPECT_EQ(onDevice.size(), 65536u);

        std::vector<float> onHost;
        const auto hostResult = fillA(recipe, nullptr, {256, 256}, onHost);
        ASSERT_TRUE(hostResult.filled) << hostResult.reason;
        EXPECT_EQ(hostResult.deviceFilled, 0u);

        // Not the host's values, but the host's kind of values: the same number of NaNs, and
        // of zeros where the kind decides them.
        const auto nans = [](const std::vector<float>& v) {
            return std::count_if(v.begin(), v.end(), [](float x) { return std::isnan(x); });
        };
        EXPECT_EQ(nans(onDevice), nans(onHost));
        if(recipe.kind != FillRecipe::Kind::SPECIAL)
        {
            EXPECT_EQ(std::count(onDevice.begin(), onDevice.end(), 0.0f),
                      std::count(onHost.begin(), onHost.end(), 0.0f));
        }
    }
}

TEST(TestGpuDeviceInputFill, ATensorBelowTheThresholdStaysOnTheHost)
{
    SKIP_IF_NO_DEVICE_FILL();
    const std::vector<int64_t> dims = {64, 64};
    ASSERT_LT(elementsOf(dims), DeviceInputFiller::minElements());
    std::vector<float> elements;
    DeviceInputFiller device;
    const auto result = fillA(FillRecipe::zeros(), &device, dims, elements);
    ASSERT_TRUE(result.filled) << result.reason;
    EXPECT_EQ(result.deviceFilled, 0u);
}

// The decline paths never reach HIP, so this runs without a device. Whatever the device cannot
// make, the filler leaves untouched for the host, which fills it or reports it unsupported.
TEST(TestGpuDeviceInputFill, DeclinesWhatTheDeviceCannotFillAndLeavesTheTensorAlone)
{
    const Layout layout{"packed", {256, 256}, {256, 1}};
    const auto untouched = [&](auto tag, const FillRecipe& recipe) {
        using T = decltype(tag);
        const auto filled = deviceFill<T>(recipe, 1, layout);
        EXPECT_FALSE(filled.accepted) << InputFillRecipes::kindToString(recipe.kind);
        for(const auto v : filled.allocation)
        {
            unsigned char bytes[sizeof(T)]; // NOLINT(modernize-avoid-c-arrays)
            std::memcpy(bytes, &v, sizeof(T));
            for(const auto b : bytes)
            {
                ASSERT_EQ(b, kPoison);
            }
        }
    };
    // A constant is cheaper on the host.
    untouched(float{}, FillRecipe::fixed(2.0f));
    // Element types without a device kernel.
    untouched(double{}, FillRecipe::nonzero(-1.0f, 1.0f));
    untouched(std::int8_t{}, FillRecipe::zeros());
    // A kind the data type cannot hold.
    untouched(types::fp8_e4m3{}, FillRecipe::nonFinite(FillRecipe::SpecialValue::POS_INF, 0.1f));
    untouched(float{}, FillRecipe::nonzero(2.0f, 1.0f));
    untouched(types::half{}, FillRecipe::large(-1.0f));
    untouched(types::fp8_e5m2{}, FillRecipe::nonzero(0.0f, 1.0e-9f));
}

TEST(TestGpuDeviceInputFill, ATensorSmallerThanTheThresholdIsDeclined)
{
    const auto elements = static_cast<int64_t>(DeviceInputFiller::minElements()) - 1;
    Tensor<float> tensor(std::vector<int64_t>{elements}, std::vector<int64_t>{1});
    DeviceInputFiller device;
    EXPECT_FALSE(device.tryFill(tensor, FillRecipe::zeros(), 1));
    EXPECT_FALSE(device.tryFill(tensor, FillRecipe::subnormal(), 1));
}

TEST(TestHostInputFillOverride, IsForcedByAnyValueButZeroOrEmpty)
{
    using hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter;
    {
        const ScopedEnvironmentVariableSetter guard("HIPDNN_TEST_HOST_INPUT_FILL", "1");
        EXPECT_TRUE(hostInputFillForced());
    }
    {
        const ScopedEnvironmentVariableSetter guard("HIPDNN_TEST_HOST_INPUT_FILL", "true");
        EXPECT_TRUE(hostInputFillForced());
    }
    {
        const ScopedEnvironmentVariableSetter guard("HIPDNN_TEST_HOST_INPUT_FILL", "0");
        EXPECT_FALSE(hostInputFillForced());
    }
    {
        const ScopedEnvironmentVariableSetter guard("HIPDNN_TEST_HOST_INPUT_FILL", "");
        EXPECT_FALSE(hostInputFillForced());
    }
}

// NOLINTEND(readability-identifier-naming)

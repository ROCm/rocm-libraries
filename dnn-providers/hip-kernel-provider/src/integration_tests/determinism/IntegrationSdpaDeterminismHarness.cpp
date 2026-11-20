// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "SdpaDeterminismCompare.hpp"
#include "SdpaDeterminismGraphs.hpp"

/**
 * @file IntegrationSdpaDeterminismHarness.cpp
 * @brief Host self-check of the SDPA determinism harness: the comparer and the filler must
 *        detect what they exist to detect, or a device case built on them passes vacuously.
 *        Runs on any host, with no device and no engine. Each case holds one detector to its
 *        contract with an input that must be reported and one that must not be.
 */

namespace
{

using namespace hip_kernel_provider::sdpa_determinism;

using Bytes = std::vector<uint8_t>;

/// @p count finite values of @p type.
Bytes finiteBuffer(ElementType type, size_t count)
{
    Bytes bytes(count * elementBytes(type));
    fillBounded(type, bytes.data(), count, 7, 1.0F);
    return bytes;
}

LiveRuns everything(size_t count)
{
    return {{0, count}};
}

void setBits32(Bytes& buffer, size_t element, uint32_t bits)
{
    std::memcpy(buffer.data() + element * sizeof(bits), &bits, sizeof(bits));
}

void setBits16(Bytes& buffer, size_t element, uint16_t bits)
{
    std::memcpy(buffer.data() + element * sizeof(bits), &bits, sizeof(bits));
}

void setPoison(Bytes& buffer, size_t element, size_t elementSize)
{
    poison(buffer.data() + element * elementSize, elementSize);
}

/// Writes the quiet NaN of @p type at @p element.
void setNotANumber(Bytes& buffer, ElementType type, size_t element)
{
    switch(type)
    {
    case ElementType::FLOAT32:
        setBits32(buffer, element, 0x7FC00000U);
        return;
    case ElementType::BFLOAT16:
        setBits16(buffer, element, 0x7FC0);
        return;
    case ElementType::HALF:
        setBits16(buffer, element, 0x7E00);
        return;
    case ElementType::FP8_E4M3_FNUZ:
        buffer[element] = 0x80;
        return;
    default:
        throw std::invalid_argument("setNotANumber: unhandled ElementType value");
    }
}

/// Writes the infinity of @p type at @p element, negative when @p negative.
void setInfinity(Bytes& buffer, ElementType type, size_t element, bool negative)
{
    switch(type)
    {
    case ElementType::FLOAT32:
        setBits32(buffer, element, negative ? 0xFF800000U : 0x7F800000U);
        return;
    case ElementType::BFLOAT16:
        setBits16(buffer, element, static_cast<uint16_t>(negative ? 0xFF80 : 0x7F80));
        return;
    case ElementType::HALF:
        setBits16(buffer, element, static_cast<uint16_t>(negative ? 0xFC00 : 0x7C00));
        return;
    case ElementType::FP8_E4M3_FNUZ:
        ADD_FAILURE() << "FP8 E4M3 FNUZ has no infinity";
        return;
    default:
        throw std::invalid_argument("setInfinity: unhandled ElementType value");
    }
}

void flipAByte(Bytes& buffer, size_t element, size_t elementSize)
{
    buffer[element * elementSize + elementSize - 1] ^= 0x10;
}

constexpr std::array<ElementType, 3> FLOATING_TYPES
    = {ElementType::FLOAT32, ElementType::BFLOAT16, ElementType::HALF};

constexpr std::array<ElementType, 4> ALL_TYPES
    = {ElementType::FLOAT32, ElementType::BFLOAT16, ElementType::HALF, ElementType::FP8_E4M3_FNUZ};

/// One flipped byte in any output of a four-output set is reported with its tensor and index,
/// and the lowest differing element is the one reported.
void flippedByteIsReportedInEachOutput()
{
    struct Output
    {
        const char* name;
        ElementType type;
        size_t count;
    };
    const std::array<Output, 4> outputs = {{{"O", ElementType::BFLOAT16, 64},
                                            {"LSE", ElementType::FLOAT32, 16},
                                            {"dK", ElementType::HALF, 32},
                                            {"dV", ElementType::BFLOAT16, 48}}};
    for(size_t i = 0; i < outputs.size(); ++i)
    {
        const auto& output = outputs[i];
        const size_t size = elementBytes(output.type);
        const Bytes reference = finiteBuffer(output.type, output.count);
        Bytes current = reference;
        const size_t element = (i * 5 + 3) % output.count;
        flipAByte(current, element, size);
        flipAByte(current, output.count - 1, size);

        const auto live = everything(output.count);
        const auto found
            = findFirstDifference(output.name, size, reference.data(), current.data(), live);
        ASSERT_TRUE(found.has_value()) << output.name << ": a flipped byte went unreported";
        EXPECT_EQ(found->tensor, output.name);
        EXPECT_EQ(found->element, element);
        EXPECT_EQ(found->kind, FindingKind::DIFFERS);
    }
}

/// An element left at the poison pattern is reported as unwritten, in every encoding, and a
/// NaN a kernel computed is reported as a NaN, not as unwritten.
void unwrittenElementIsReported()
{
    for(const auto type : ALL_TYPES)
    {
        const size_t size = elementBytes(type);
        Bytes buffer = finiteBuffer(type, 32);
        setPoison(buffer, 5, size);
        const auto found = findInvalidElement("O", type, buffer.data(), everything(32), false);
        ASSERT_TRUE(found.has_value()) << "element type " << size << " bytes";
        EXPECT_EQ(found->kind, FindingKind::UNWRITTEN);
        EXPECT_EQ(found->element, 5U);
    }

    Bytes buffer = finiteBuffer(ElementType::FLOAT32, 32);
    setNotANumber(buffer, ElementType::FLOAT32, 2);
    setPoison(buffer, 5, sizeof(float));
    const auto found
        = findInvalidElement("O", ElementType::FLOAT32, buffer.data(), everything(32), false);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->kind, FindingKind::NOT_A_NUMBER);
    EXPECT_EQ(found->element, 2U);
}

/// A NaN is reported in every encoding, also where infinity is allowed.
void notANumberIsReportedForEveryElementType()
{
    for(const auto type : ALL_TYPES)
    {
        Bytes buffer = finiteBuffer(type, 32);
        EXPECT_FALSE(findInvalidElement("O", type, buffer.data(), everything(32), false))
            << "a finite buffer of " << elementBytes(type) << "-byte elements was rejected";
        setNotANumber(buffer, type, 2);
        for(const bool allowInfinity : {false, true})
        {
            const auto found
                = findInvalidElement("O", type, buffer.data(), everything(32), allowInfinity);
            ASSERT_TRUE(found.has_value())
                << elementBytes(type) << "-byte NaN, allowInfinity " << allowInfinity;
            EXPECT_EQ(found->kind, FindingKind::NOT_A_NUMBER);
            EXPECT_EQ(found->element, 2U);
        }
    }
}

/// Infinity is rejected unless the case allows it, and accepted when it does, of either sign.
void infinityIsRejectedUnlessAllowed()
{
    for(const auto type : FLOATING_TYPES)
    {
        for(const bool negative : {false, true})
        {
            Bytes buffer = finiteBuffer(type, 32);
            setInfinity(buffer, type, 4, negative);
            const auto rejected
                = findInvalidElement("LSE", type, buffer.data(), everything(32), false);
            ASSERT_TRUE(rejected.has_value())
                << elementBytes(type) << "-byte infinity, negative " << negative;
            EXPECT_EQ(rejected->kind, FindingKind::INFINITE);
            EXPECT_EQ(rejected->element, 4U);
            EXPECT_FALSE(findInvalidElement("LSE", type, buffer.data(), everything(32), true))
                << "allowed infinity was rejected";
        }
    }
}

/// Two equal buffers pass every check, and the same buffers with one live byte changed no
/// longer do: the pass is a result, not a default.
void equalBuffersPass()
{
    const Bytes reference = finiteBuffer(ElementType::FLOAT32, 64);
    Bytes current = reference;
    const auto live = everything(64);

    EXPECT_FALSE(findFirstDifference("O", 4, reference.data(), current.data(), live));
    EXPECT_EQ(countDifferences(4, reference.data(), current.data(), live), 0U);
    EXPECT_FALSE(findInvalidElement("O", ElementType::FLOAT32, current.data(), live, false));

    flipAByte(current, 9, 4);
    const auto found = findFirstDifference("O", 4, reference.data(), current.data(), live);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->element, 9U);
}

/// A packed THD buffer: a poison in the gap between sequence origins or in the tail after the
/// last sequence is neither unwritten nor a difference; one flipped byte in a live token of the
/// same buffer is reported, and so is a poison there.
void packedGapIsDeadAndLiveTokenIsChecked()
{
    constexpr size_t TOKEN_ELEMENTS = 4;
    constexpr size_t TOKENS = 12;
    // Sequence 0: tokens 0-2; gap: tokens 3-7; sequence 1: tokens 8-9; tail: tokens 10-11.
    const auto live = packedRuns({0, 8}, {3, 2}, TOKEN_ELEMENTS);
    ASSERT_EQ(live.size(), 2U);
    EXPECT_EQ(live[0].begin, 0U);
    EXPECT_EQ(live[0].end, 12U);
    EXPECT_EQ(live[1].begin, 32U);
    EXPECT_EQ(live[1].end, 40U);

    const Bytes reference = finiteBuffer(ElementType::FLOAT32, TOKENS * TOKEN_ELEMENTS);
    Bytes current = reference;
    for(const size_t token : {3U, 4U, 5U, 6U, 7U, 10U, 11U})
    {
        for(size_t e = 0; e < TOKEN_ELEMENTS; ++e)
        {
            setPoison(current, token * TOKEN_ELEMENTS + e, sizeof(float));
        }
    }
    EXPECT_FALSE(findFirstDifference("O", 4, reference.data(), current.data(), live))
        << "a poisoned dead token was reported as a difference";
    EXPECT_EQ(countDifferences(4, reference.data(), current.data(), live), 0U);
    EXPECT_FALSE(findInvalidElement("O", ElementType::FLOAT32, current.data(), live, false))
        << "a poisoned dead token was reported as unwritten";

    Bytes flipped = current;
    flipAByte(flipped, 6, 4);
    const auto difference = findFirstDifference("O", 4, reference.data(), flipped.data(), live);
    ASSERT_TRUE(difference.has_value()) << "a flipped byte of a live token went unreported";
    EXPECT_EQ(difference->element, 6U);

    Bytes unwritten = current;
    setPoison(unwritten, 9 * TOKEN_ELEMENTS, sizeof(float));
    const auto invalid
        = findInvalidElement("O", ElementType::FLOAT32, unwritten.data(), live, false);
    ASSERT_TRUE(invalid.has_value()) << "an unwritten live token went unreported";
    EXPECT_EQ(invalid->kind, FindingKind::UNWRITTEN);
    EXPECT_EQ(invalid->element, 36U);
}

/// A padded batch (seq_lens with padding_mask): the positions at or past a batch's length are
/// dead in O and in the LSE alike, whatever the layout; a live element is still checked.
void paddedBatchDeadRegionIsNeitherUnwrittenNorDifferent()
{
    const std::vector<int64_t> lengths = {2, 1};

    // O is [B, H, S, D] = [2, 2, 3, 2] stored BSHD.
    const std::vector<int64_t> oDims = {2, 2, 3, 2};
    const std::vector<int64_t> oStrides = {12, 2, 4, 1};
    const auto oLive = logicalRuns(oDims, oStrides, [&](const std::vector<int64_t>& index) {
        return index[2] < lengths[static_cast<size_t>(index[0])];
    });
    ASSERT_EQ(oLive.size(), 2U);
    EXPECT_EQ(oLive[0].begin, 0U);
    EXPECT_EQ(oLive[0].end, 8U);
    EXPECT_EQ(oLive[1].begin, 12U);
    EXPECT_EQ(oLive[1].end, 16U);

    const Bytes oReference = finiteBuffer(ElementType::FLOAT32, 24);
    Bytes oCurrent = oReference;
    for(size_t element = 0; element < 24; ++element)
    {
        const bool dead = (element >= 8 && element < 12) || element >= 16;
        if(dead)
        {
            setPoison(oCurrent, element, sizeof(float));
        }
    }
    EXPECT_FALSE(findFirstDifference("O", 4, oReference.data(), oCurrent.data(), oLive));
    EXPECT_FALSE(findInvalidElement("O", ElementType::FLOAT32, oCurrent.data(), oLive, false));
    // Batch 1, head 1, position 0, d 1.
    flipAByte(oCurrent, 15, 4);
    const auto oFound = findFirstDifference("O", 4, oReference.data(), oCurrent.data(), oLive);
    ASSERT_TRUE(oFound.has_value()) << "a flipped byte of a live O element went unreported";
    EXPECT_EQ(oFound->element, 15U);

    // The LSE is [B, H, S, 1] = [2, 2, 3, 1], packed.
    const std::vector<int64_t> lseDims = {2, 2, 3, 1};
    const std::vector<int64_t> lseStrides = {6, 3, 1, 1};
    const auto lseLive = logicalRuns(lseDims, lseStrides, [&](const std::vector<int64_t>& index) {
        return index[2] < lengths[static_cast<size_t>(index[0])];
    });
    ASSERT_EQ(lseLive.size(), 4U);
    const Bytes lseReference = finiteBuffer(ElementType::FLOAT32, 12);
    Bytes lseCurrent = lseReference;
    for(const size_t dead : {2U, 5U, 7U, 8U, 10U, 11U})
    {
        setPoison(lseCurrent, dead, sizeof(float));
    }
    EXPECT_FALSE(findFirstDifference("LSE", 4, lseReference.data(), lseCurrent.data(), lseLive));
    EXPECT_FALSE(
        findInvalidElement("LSE", ElementType::FLOAT32, lseCurrent.data(), lseLive, false));
    flipAByte(lseCurrent, 9, 4);
    const auto lseFound
        = findFirstDifference("LSE", 4, lseReference.data(), lseCurrent.data(), lseLive);
    ASSERT_TRUE(lseFound.has_value()) << "a flipped byte of a live LSE element went unreported";
    EXPECT_EQ(lseFound->element, 9U);
}

/// The measure rows report how many logical elements differ: live differences are counted,
/// dead ones are not, and the first one reported is the lowest.
void countsEveryDifferingLiveElement()
{
    const Bytes reference = finiteBuffer(ElementType::FLOAT32, 32);
    Bytes current = reference;
    for(const size_t element : {2U, 7U, 21U, 12U, 31U})
    {
        flipAByte(current, element, 4);
    }
    const LiveRuns live = {{0, 10}, {20, 30}};
    EXPECT_EQ(countDifferences(4, reference.data(), current.data(), live), 3U);
    const auto found = findFirstDifference("dQ", 4, reference.data(), current.data(), live);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->element, 2U);
}

/// The FP8 E4M3 FNUZ filler writes finite bytes and never 0x80, which is its NaN: values the
/// format holds round trip exactly, ties go to the even code, a value past the largest
/// magnitude saturates, and a value that rounds to zero is +0 rather than the NaN byte.
void fp8FnuzFillerRoundTripsBoundedValuesToFiniteBytes()
{
    for(const float value :
        {0.5F, -0.5F, 1.0F, -1.0F, 1.875F, 0.0625F, 240.0F, -240.0F, 0.0009765625F})
    {
        const uint8_t byte = fp8E4M3FnuzFromFloat(value);
        EXPECT_NE(byte, 0x80) << value;
        EXPECT_EQ(fp8E4M3FnuzToFloat(byte), value);
    }
    EXPECT_EQ(fp8E4M3FnuzToFloat(fp8E4M3FnuzFromFloat(1000.0F)), 240.0F);
    EXPECT_EQ(fp8E4M3FnuzToFloat(fp8E4M3FnuzFromFloat(-1000.0F)), -240.0F);
    EXPECT_EQ(fp8E4M3FnuzFromFloat(0.0F), 0x00);
    EXPECT_EQ(fp8E4M3FnuzFromFloat(1e-6F), 0x00);
    EXPECT_EQ(fp8E4M3FnuzFromFloat(-1e-6F), 0x00);
    // Halfway between 1.0 (code 0x40) and 1.125 (0x41), and between 1.125 and 1.25 (0x42).
    EXPECT_EQ(fp8E4M3FnuzToFloat(fp8E4M3FnuzFromFloat(1.0625F)), 1.0F);
    EXPECT_EQ(fp8E4M3FnuzToFloat(fp8E4M3FnuzFromFloat(1.1875F)), 1.25F);

    Bytes filled(4096);
    fillBounded(ElementType::FP8_E4M3_FNUZ, filled.data(), filled.size(), 3, 2.0F);
    bool anyNonZero = false;
    for(const uint8_t byte : filled)
    {
        EXPECT_NE(byte, 0x80);
        const float value = fp8E4M3FnuzToFloat(byte);
        EXPECT_LE(std::abs(value), 2.0F);
        anyNonZero = anyNonZero || byte != 0;
    }
    EXPECT_TRUE(anyNonZero) << "the filler wrote only zeros";
}

struct SelfCheckCase
{
    std::string name;
    void (*run)();
};

std::ostream& operator<<(std::ostream& out, const SelfCheckCase& selfCheck)
{
    return out << selfCheck.name;
}

std::vector<SelfCheckCase> selfCheckCases()
{
    return {
        {"FlippedByteIsReportedInEachOutput", &flippedByteIsReportedInEachOutput},
        {"UnwrittenElementIsReported", &unwrittenElementIsReported},
        {"NotANumberIsReportedForEveryElementType", &notANumberIsReportedForEveryElementType},
        {"InfinityIsRejectedUnlessAllowed", &infinityIsRejectedUnlessAllowed},
        {"EqualBuffersPass", &equalBuffersPass},
        {"PackedGapIsDeadAndLiveTokenIsChecked", &packedGapIsDeadAndLiveTokenIsChecked},
        {"PaddedBatchDeadRegionIsNeitherUnwrittenNorDifferent",
         &paddedBatchDeadRegionIsNeitherUnwrittenNorDifferent},
        {"CountsEveryDifferingLiveElement", &countsEveryDifferingLiveElement},
        {"Fp8FnuzFillerRoundTripsBoundedValuesToFiniteBytes",
         &fp8FnuzFillerRoundTripsBoundedValuesToFiniteBytes},
    };
}

class IntegrationSdpaDeterminismHarness : public ::testing::TestWithParam<SelfCheckCase>
{
};

} // namespace

TEST_P(IntegrationSdpaDeterminismHarness, HoldsOnTheHost)
{
    ASSERT_NO_FATAL_FAILURE(GetParam().run());
}

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         IntegrationSdpaDeterminismHarness,
                         ::testing::ValuesIn(selfCheckCases()),
                         [](const ::testing::TestParamInfo<SelfCheckCase>& info) {
                             return info.param.name;
                         });

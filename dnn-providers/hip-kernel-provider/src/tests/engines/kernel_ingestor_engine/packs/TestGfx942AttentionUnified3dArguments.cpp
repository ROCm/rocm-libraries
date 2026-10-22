// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionUnified3dArguments.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionUnified3dQueryOffsets.hpp"

/**
 * @file TestGfx942AttentionUnified3dArguments.cpp
 * @brief The segment launch's arguments equal, value by value and in order, what rocKE
 *        passes for the same strided decode request; and the shared cu_seqlens_q source
 *        grows correctly.
 *
 * The table (Gfx942AttentionUnified3dArgumentsTable.inc) is produced by
 * gen_gfx942_attention_unified_3d_arguments.py from rocKE's `_3d_signature(...,
 * strided_kv=True)` and `StridedKvCacheLayout(...).arguments()`. Regenerate it with the
 * command in that script's docstring; a stale table fails the drift test
 * test_gfx942_arguments_table_current.py.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

struct ExpectedArgument
{
    const char* name;
    const char* type; // "ptr", "f32", "i32", "i64"
    const char* buffer; // for "ptr": the seg_vals buffer, "null" when absent
    int64_t integer;
    float real;
};

struct ArgumentsRow
{
    const char* label;
    int32_t batch;
    float scale;
    std::array<int64_t, 4> kDims;
    std::array<int64_t, 4> kStrides;
    std::array<int64_t, 4> vDims;
    std::array<int64_t, 4> vStrides;
    std::array<ExpectedArgument, GFX942_UNIFIED_3D_SEGMENT_ARGUMENT_COUNT> arguments;
};

const ArgumentsRow ARGUMENTS_TABLE[] = {
#include "Gfx942AttentionUnified3dArgumentsTable.inc"
};

const char* typeName(Gfx942Unified3dArgumentKind kind)
{
    switch(kind)
    {
    case Gfx942Unified3dArgumentKind::POINTER:
        return "ptr";
    case Gfx942Unified3dArgumentKind::F32:
        return "f32";
    case Gfx942Unified3dArgumentKind::I32:
        return "i32";
    case Gfx942Unified3dArgumentKind::I64:
        return "i64";
    default:
        return "?";
    }
}

const char* bufferName(Gfx942Unified3dBuffer buffer)
{
    switch(buffer)
    {
    case Gfx942Unified3dBuffer::NONE:
        return "null";
    case Gfx942Unified3dBuffer::SEGM_OUTPUT:
        return "segm_output";
    case Gfx942Unified3dBuffer::SEGM_MAX:
        return "segm_max";
    case Gfx942Unified3dBuffer::SEGM_EXPSUM:
        return "segm_expsum";
    case Gfx942Unified3dBuffer::QUERY:
        return "query";
    case Gfx942Unified3dBuffer::KEY:
        return "key";
    case Gfx942Unified3dBuffer::VALUE:
        return "value";
    case Gfx942Unified3dBuffer::SEQUSED_K:
        return "seqused_k";
    case Gfx942Unified3dBuffer::CU_SEQLENS_Q:
        return "cu_seqlens_q";
    default:
        return "?";
    }
}

uint32_t floatBits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

Gfx942Unified3dSegmentArguments argumentsFor(const ArgumentsRow& row)
{
    return gfx942Unified3dSegmentArguments(row.batch,
                                           row.scale,
                                           gfx942StridedKvArguments(row.kDims, row.kStrides),
                                           gfx942StridedKvArguments(row.vDims, row.vStrides));
}

} // namespace

TEST(TestGfx942AttentionUnified3dArguments, MatchesRockeOnEveryGeneratedRow)
{
    // BHSD x3, BSHD x3, padded row x2, padded capacity x1, mixed K-vs-V x3.
    ASSERT_EQ(std::size(ARGUMENTS_TABLE), 12u);
    for(const auto& row : ARGUMENTS_TABLE)
    {
        SCOPED_TRACE(row.label);
        const auto actual = argumentsFor(row);
        for(size_t i = 0; i < actual.size(); ++i)
        {
            const auto& want = row.arguments[i];
            const auto& got = actual[i];
            SCOPED_TRACE("argument " + std::to_string(i) + " (" + want.name + ")");
            EXPECT_STREQ(got.name, want.name);
            EXPECT_STREQ(typeName(got.kind), want.type);
            const std::string type(want.type);
            if(type == "ptr")
            {
                EXPECT_STREQ(bufferName(got.buffer), want.buffer);
                EXPECT_EQ(got.bits, 0u);
            }
            else if(type == "f32")
            {
                EXPECT_EQ(floatBits(got.f32()), floatBits(want.real));
                EXPECT_EQ(got.bits >> 32, 0u);
            }
            else
            {
                EXPECT_EQ(got.integer(), want.integer);
            }
        }
    }
}

// Hand-checked anchor: BSHD D128, Hkv 8, Skv 4096 (the second BSHD row's shape).
TEST(TestGfx942AttentionUnified3dArguments, BshdAnchor)
{
    const auto args
        = gfx942StridedKvArguments({4, 8, 4096, 128}, {4096 * 8 * 128, 128, 8 * 128, 1});
    EXPECT_EQ(args.strideBatchBytes, 2 * 4096 * 8 * 128);
    EXPECT_EQ(args.strideHeadBytes, 256);
    EXPECT_EQ(args.strideTokenBytes, 2048);
    EXPECT_EQ(args.spanBytes, 2 * (4095 * 1024 + 128));
}

TEST(TestGfx942AttentionUnified3dArguments, RefusesSpanOutsideInt32)
{
    EXPECT_THROW(gfx942StridedKvArguments({1, 1, 1 << 20, 1024}, {1LL << 30, 1LL << 30, 1024, 1}),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestGfx942AttentionUnified3dArguments, WordsResolvePointersAndKeepScalars)
{
    const auto arguments = argumentsFor(ARGUMENTS_TABLE[1]);
    std::array<std::byte, 8> storage{};
    Gfx942Unified3dBufferAddresses addresses;
    addresses.segmOutput = &storage[0];
    addresses.segmMax = &storage[1];
    addresses.segmExpsum = &storage[2];
    addresses.query = &storage[3];
    addresses.key = &storage[4];
    addresses.value = &storage[5];
    addresses.sequsedK = &storage[6];
    addresses.cuSeqlensQ = &storage[7];
    const auto words = gfx942Unified3dArgumentWords(arguments, addresses);
    const auto at = [&](size_t i) { return reinterpret_cast<uintptr_t>(&storage[i]); };
    const std::array<uint64_t, 12> pointers
        = {at(0), at(1), at(2), at(3), at(4), at(5), 0, 0, at(6), 0, 0, at(7)};
    for(size_t i = 0; i < pointers.size(); ++i)
    {
        EXPECT_EQ(words[i], pointers[i]) << arguments[i].name;
    }
    for(size_t i = pointers.size(); i < words.size(); ++i)
    {
        EXPECT_EQ(words[i], arguments[i].bits) << arguments[i].name;
    }
}

// ---------------------------------------------------------------------------
// The shared cu_seqlens_q source
// ---------------------------------------------------------------------------

namespace
{

std::vector<size_t>& allocations()
{
    static std::vector<size_t> s_allocations;
    return s_allocations;
}

std::vector<void*>& owned()
{
    static std::vector<void*> s_owned;
    return s_owned;
}

void* recordingAllocate(size_t bytes)
{
    allocations().push_back(bytes);
    void* block = std::malloc(bytes);
    owned().push_back(block);
    return block;
}

void* failingAllocate(size_t /*bytes*/)
{
    return nullptr;
}

void expectArange(const int32_t* data, size_t count)
{
    for(size_t i = 0; i < count; ++i)
    {
        ASSERT_EQ(data[i], static_cast<int32_t>(i)) << "entry " << i;
    }
}

} // namespace

TEST(TestGfx942AttentionUnified3dQueryOffsets, GrowsOnlyWhenNeededAndKeepsOldBlocks)
{
    allocations().clear();
    {
        Gfx942GrowOnlyArange arange(&recordingAllocate);
        EXPECT_EQ(arange.capacity(), 0u);

        // First use: one MIN_CAPACITY block serves every B + 1 <= 256 as a prefix.
        const int32_t* first = arange.acquire(5);
        expectArange(first, 5);
        EXPECT_EQ(arange.capacity(), Gfx942GrowOnlyArange::MIN_CAPACITY);
        EXPECT_EQ(arange.acquire(256), first);
        EXPECT_EQ(arange.acquire(1), first);
        ASSERT_EQ(allocations().size(), 1u);
        EXPECT_EQ(allocations()[0], 256 * sizeof(int32_t));

        // Growth at least doubles; the old block stays allocated and intact.
        const int32_t* second = arange.acquire(257);
        EXPECT_NE(second, first);
        EXPECT_EQ(arange.capacity(), 512u);
        expectArange(second, 512);
        expectArange(first, 256);

        // A request beyond double gets exactly what it asked for.
        const int32_t* third = arange.acquire(5000);
        EXPECT_EQ(arange.capacity(), 5000u);
        expectArange(third, 5000);
        EXPECT_EQ(arange.acquire(10), third);

        EXPECT_EQ(allocations().size(), 3u);
        const auto blocks = arange.blocks();
        ASSERT_EQ(blocks.size(), 3u);
        EXPECT_EQ(blocks[0], first);
        EXPECT_EQ(blocks[1], second);
        EXPECT_EQ(blocks[2], third);
    }
    // Never freed by the arange: the test owns the malloc'd blocks.
    for(void* block : owned())
    {
        std::free(block);
    }
    owned().clear();
}

TEST(TestGfx942AttentionUnified3dQueryOffsets, RefusesEmptyAndReportsAllocationFailure)
{
    Gfx942GrowOnlyArange arange(&failingAllocate);
    EXPECT_THROW(static_cast<void>(arange.acquire(0)), hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(static_cast<void>(arange.acquire(2)), hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_EQ(arange.capacity(), 0u);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

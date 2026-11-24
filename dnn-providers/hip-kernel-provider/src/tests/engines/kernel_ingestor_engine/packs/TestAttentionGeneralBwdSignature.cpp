// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <exception>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/AttentionGeneralBwdTestSupport.hpp"

/**
 * @file TestAttentionGeneralBwdSignature.cpp
 * @brief Pins the kernarg ABI `rocke.attn_bwd.v3` hipkernel:AttentionGeneralBwd marshals for
 *        its four launch kinds -- prep, main, convert and split dQ -- against the lists
 *        rocKE's `attn_bwd_params` declares.
 *
 * A wrong count, order, kind or width is not a launch error: the kernel reads the kernarg
 * segment as the types it was compiled for, and the result is wrong numbers with a success
 * status. The lists are stated here in rocKE's own spelling, so drift in the generated
 * header that the generator's staleness check did not catch still fails a build.
 *
 * prepare() compares each launch's recorded signature before it opens the archive, so the
 * cases below run with no device and no archive: an accepted signature fails later, on the
 * archive's absence, and a refused one fails on the comparison. The main slot builds its
 * main launch first, so the convert and split-dQ lists are checked against the same rule
 * the handler marshals by.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using namespace general_bwd_test;
using hipdnn_plugin_sdk::ingestor::KernelArgument;
using hipdnn_plugin_sdk::ingestor::KernelSource;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;

// `name:format` per kernarg, in ABI order (`Q` pointer, `q` i64, `f` f32, `i` i32).
constexpr std::string_view PREP_ABI
    = "O:Q dO:Q LSE:Q SEQ_Q:Q SEQ_KV:Q OFF_Q:Q OFF_KV:Q WS_LSE2:Q WS_DSUM:Q WS_DQ:Q WS_DK:Q "
      "WS_DV:Q WORKLIST:Q o_b:q o_h:q do_b:q do_h:q l_b:q l_h:q o_t:i do_t:i l_t:i h_q:i h_k:i "
      "h_v:i S_q_max:i S_kv_max:i has_len:i len_stride:i off64:i q_mult:i q_div:i kv_mult:i "
      "kv_div:i ws_rows_q:i ws_rows_kv:i ws_seg:i zero_kv:i zero_dq:i use_worklist:i n_batch:i "
      "wl_kn0:i";
constexpr std::string_view MAIN_ABI
    = "Q:Q K:Q V:Q dO:Q dK:Q dV:Q WS_LSE2:Q WS_DSUM:Q WS_DQ:Q WS_DK:Q WS_DV:Q SEQ_Q:Q SEQ_KV:Q "
      "OFF_Q:Q OFF_KV:Q WORKLIST:Q q_b:q q_h:q k_b:q k_h:q v_b:q v_h:q do_b:q do_h:q dk_b:q "
      "dk_h:q dv_b:q dv_h:q scale_log2:f ds_mult:f dk_mult:f dq_mult:f q_t:i k_t:i v_t:i do_t:i "
      "dk_t:i dv_t:i h_q:i h_k:i h_v:i G:i gk:i gv:i S_q_max:i S_kv_max:i has_len:i len_stride:i "
      "off64:i q_mult:i q_div:i kv_mult:i kv_div:i left:i right:i bottom_right:i ws_rows_q:i "
      "ws_rows_kv:i ws_seg:i g_split:i n_kv_tiles:i lb_order:i pair:i xcd_n:i xcd_chunk:i "
      "pack_heads:i use_worklist:i";
constexpr std::string_view CONVERT_ABI
    = "SRC:Q DST:Q SEQ:Q OFF:Q d_b:q d_h:q mult:f H:i d_t:i S_max:i has_len:i len_stride:i "
      "off64:i tok_mult:i tok_div:i ws_rows:i ws_seg:i";
constexpr std::string_view DQ_ABI
    = "Q:Q K:Q V:Q dO:Q dQ:Q WS_LSE2:Q WS_DSUM:Q SEQ_Q:Q SEQ_KV:Q OFF_Q:Q OFF_KV:Q q_b:q q_h:q "
      "k_b:q k_h:q v_b:q v_h:q do_b:q do_h:q dq_b:q dq_h:q scale_log2:f ds_mult:f dq_mult:f "
      "q_t:i k_t:i v_t:i do_t:i dq_t:i h_q:i h_k:i h_v:i G:i S_q_max:i S_kv_max:i has_len:i "
      "len_stride:i off64:i q_mult:i q_div:i kv_mult:i kv_div:i left:i right:i bottom_right:i "
      "ws_rows_q:i ws_seg:i xcd_n:i xcd_chunk:i";

constexpr const char* BUFFER_KIND = "global_buffer";
constexpr const char* BY_VALUE_KIND = "by_value";

struct Param
{
    std::string name;
    char format;
};

std::vector<Param> parse(std::string_view abi)
{
    std::vector<Param> params;
    std::istringstream words{std::string(abi)};
    std::string word;
    while(words >> word)
    {
        const auto colon = word.rfind(':');
        params.push_back(Param{word.substr(0, colon), word.at(colon + 1)});
    }
    return params;
}

/// The argument list a code object built for @p abi records: each argument at its natural
/// alignment, pointers and i64 eight bytes, i32 and f32 four.
std::vector<KernelArgument> recordedSignature(std::string_view abi)
{
    std::vector<KernelArgument> signature;
    uint32_t offset = 0;
    for(const auto& param : parse(abi))
    {
        const bool pointer = param.format == 'Q';
        const uint32_t size = pointer || param.format == 'q' ? 8U : 4U;
        offset = (offset + size - 1) / size * size;
        signature.push_back(
            KernelArgument{pointer ? BUFFER_KIND : BY_VALUE_KIND, size, offset, param.name});
        offset += size;
    }
    return signature;
}

template <std::size_t N>
std::string generatedText(const std::array<generated::AbiParam, N>& abi)
{
    std::string text;
    for(const auto& param : abi)
    {
        if(!text.empty())
        {
            text += ' ';
        }
        text += std::string(param.name) + ':' + param.kind;
    }
    return text;
}

KernelSource kpackSource(std::string_view symbol, std::vector<KernelArgument> signature)
{
    KernelSource source;
    source.kind = KernelSourceKind::KPACK;
    source.library = "there-is-no-archive-here.kpack";
    source.tocKey = "toc#0";
    source.symbol = std::string(symbol);
    source.signature = std::move(signature);
    return source;
}

void identify(KernelDefinition& kernel)
{
    kernel.kernelId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-00000000b001");
    kernel.packId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-00000000b002");
    kernel.dispatchId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-00000000b003");
    kernel.originDirectory = "/nonexistent";
    kernel.treeRoot = "/nonexistent";
}

/// The fragment only the signature refusal carries.
constexpr std::string_view MISMATCH_MARKER = "is packaged with arguments";

/// prepare()'s failure for the slot @p symbol when its first launch records @p recorded,
/// on a graph of @p spec. Empty when prepare() returned, which no case expects.
std::string prepareFailure(std::string_view symbol,
                           const std::vector<KernelArgument>& recorded,
                           const GraphSpec& spec = {})
{
    registerNativeIngestorSymbols();
    const GraphFixture fixture(bwdGraphJson(spec), "gfx942", fullComputeUnits("gfx942"));
    const auto context = fixture.context();
    const auto verdict = general_bwd::evaluate(context);
    EXPECT_TRUE(std::holds_alternative<general_bwd::Plan>(verdict))
        << "the fixture graph does not match, so prepare() never compares a signature";
    if(!std::holds_alternative<general_bwd::Plan>(verdict))
    {
        return {};
    }
    const auto& plan = std::get<general_bwd::Plan>(verdict);
    const auto bound = general_bwd::tokensFor(plan);

    KernelDefinition kernel;
    if(symbol == PREP_DISPATCH_SYMBOL)
    {
        kernel = prepKernelOf(*plan.row);
        kernel.source = kpackSource(plan.launches.front().kernel, recorded);
    }
    else
    {
        kernel = mainKernelOf(*plan.row);
        for(const auto& launch : plan.launches)
        {
            if(launch.launch == general_bwd::Launch::PREP)
            {
                continue;
            }
            const std::string name(general_bwd::launchName(launch.launch));
            kernel.sources[name] = kpackSource(launch.kernel,
                                               launch.launch == general_bwd::Launch::MAIN
                                                   ? recorded
                                                   : recordedSignature(CONVERT_ABI));
        }
        // Every main kernel declares all four launches, also those a direct plan skips.
        for(const auto* name : {"convert_dk", "convert_dv"})
        {
            if(kernel.sources.count(name) == 0)
            {
                kernel.sources[name]
                    = kpackSource(plan.row->convertKernel, recordedSignature(CONVERT_ABI));
            }
        }
    }
    identify(kernel);

    const auto* handler
        = hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(std::string(symbol));
    EXPECT_NE(handler, nullptr);
    if(handler == nullptr)
    {
        return {};
    }
    try
    {
        handler->prepare(context, bound, kernel);
    }
    catch(const std::exception& error)
    {
        return error.what();
    }
    return {};
}

void expectAccepted(std::string_view symbol, std::string_view abi)
{
    const auto failure = prepareFailure(symbol, recordedSignature(abi));
    ASSERT_FALSE(failure.empty()) << "the archive does not exist, so prepare() cannot succeed";
    EXPECT_EQ(failure.find(MISMATCH_MARKER), std::string::npos)
        << symbol << " refused the ABI rocKE declares: " << failure;
}

void expectRefused(std::string_view symbol,
                   const std::vector<KernelArgument>& recorded,
                   const std::string& what)
{
    const auto failure = prepareFailure(symbol, recorded);
    ASSERT_FALSE(failure.empty()) << what << ": prepare() did not fail";
    EXPECT_NE(failure.find(MISMATCH_MARKER), std::string::npos)
        << what << ", but the failure was: " << failure;
}

struct SlotAbi
{
    std::string_view symbol;
    std::string_view abi;
};

constexpr std::array<SlotAbi, 2> SLOT_ABIS = {{
    {PREP_DISPATCH_SYMBOL, PREP_ABI},
    {MAIN_DISPATCH_SYMBOL, MAIN_ABI},
}};

} // namespace

TEST(TestAttentionGeneralBwdSignature, GeneratedAbiListsAreTheOnesRockeDeclares)
{
    EXPECT_EQ(generatedText(generated::ABI_PREP), PREP_ABI);
    EXPECT_EQ(generatedText(generated::ABI_MAIN), MAIN_ABI);
    EXPECT_EQ(generatedText(generated::ABI_CONVERT), CONVERT_ABI);
    EXPECT_EQ(generatedText(generated::ABI_DQ), DQ_ABI);
    EXPECT_EQ(generated::ABI, "rocke.attn_bwd.v3");
}

TEST(TestAttentionGeneralBwdSignature, EveryPlannedLaunchBindsItsAbiInOrder)
{
    // A dense MHA plan (prep, main, dQ convert) and an h_k != h_v one (plus both converts).
    GraphSpec atomic;
    atomic.hK = 4;
    atomic.hV = 2;
    for(const auto& spec : {GraphSpec{}, atomic})
    {
        const GraphFixture fixture(bwdGraphJson(spec), "gfx950", fullComputeUnits("gfx950"));
        const auto verdict = general_bwd::evaluate(fixture.context());
        ASSERT_TRUE(std::holds_alternative<general_bwd::Plan>(verdict));
        for(const auto& launch : std::get<general_bwd::Plan>(verdict).launches)
        {
            std::string_view abi = CONVERT_ABI;
            if(launch.launch == general_bwd::Launch::PREP)
            {
                abi = PREP_ABI;
            }
            else if(launch.launch == general_bwd::Launch::MAIN)
            {
                abi = MAIN_ABI;
            }
            const auto params = parse(abi);
            ASSERT_EQ(launch.kernargs.size(), params.size())
                << general_bwd::launchName(launch.launch);
            for(std::size_t i = 0; i < params.size(); ++i)
            {
                EXPECT_EQ(launch.kernargs[i].name, params[i].name);
                EXPECT_EQ(launch.kernargs[i].kind, params[i].format) << params[i].name;
            }
        }
    }
}

TEST(TestAttentionGeneralBwdSignature, SplitDqPlansBindTheDqAbi)
{
    std::size_t splitRows = 0;
    for(const auto& representative : generated::REPRESENTATIVES)
    {
        const auto& row = generated::CATALOG.at(representative.row);
        if(row.dq != generated::DqMode::SPLIT || !representative.reproduces)
        {
            continue;
        }
        ++splitRows;
        EXPECT_FALSE(row.dqKernel.empty()) << row.mainKernel;
        EXPECT_EQ(row.pack, "main_splitdq_gfx950") << row.mainKernel;
    }
    EXPECT_GT(splitRows, 0u) << "the gfx950 split-dQ pack ships no kernel";

    // The handler's rule for the split kernel's list agrees with rocKE's, position by
    // position, under requireSignatureMatch's own comparison.
    EXPECT_NO_THROW(requireSignatureMatch(recordedSignature(DQ_ABI),
                                          recordedSignature(generatedText(generated::ABI_DQ)),
                                          "dq",
                                          "dq"));
    EXPECT_NO_THROW(requireSignatureMatch(recordedSignature(CONVERT_ABI),
                                          recordedSignature(generatedText(generated::ABI_CONVERT)),
                                          "convert",
                                          "convert"));
    EXPECT_THROW(requireSignatureMatch(recordedSignature(CONVERT_ABI),
                                       recordedSignature(DQ_ABI),
                                       "dq",
                                       "the split dQ kernel is not a convert"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestAttentionGeneralBwdSignature, AcceptsTheAbiRockeDeclaresForEachSlot)
{
    for(const auto& slot : SLOT_ABIS)
    {
        expectAccepted(slot.symbol, slot.abi);
    }
}

TEST(TestAttentionGeneralBwdSignature, RefusesADroppedOrAnExtraTrailingArgument)
{
    for(const auto& slot : SLOT_ABIS)
    {
        auto dropped = recordedSignature(slot.abi);
        dropped.pop_back();
        expectRefused(slot.symbol, dropped, std::string(slot.symbol) + ": one argument short");

        auto extra = recordedSignature(slot.abi);
        extra.push_back(KernelArgument{BY_VALUE_KIND, 4, 0, ""});
        expectRefused(slot.symbol, extra, std::string(slot.symbol) + ": one argument too many");
    }
}

TEST(TestAttentionGeneralBwdSignature, RefusesAWrongKindOrWidthInAnySlot)
{
    for(const auto& slot : SLOT_ABIS)
    {
        const auto abi = recordedSignature(slot.abi);
        for(std::size_t i = 0; i < abi.size(); ++i)
        {
            auto changed = abi;
            auto& argument = changed.at(i);
            if(argument.kind == BUFFER_KIND)
            {
                argument.kind = BY_VALUE_KIND;
            }
            else
            {
                argument.size = argument.size == 8 ? 4 : 8;
            }
            expectRefused(slot.symbol, changed, std::string(slot.symbol) + " " + abi.at(i).name);
        }
    }
}

TEST(TestAttentionGeneralBwdSignature, RefusesEachSlotUnderAnotherName)
{
    for(const auto& slot : SLOT_ABIS)
    {
        const auto abi = recordedSignature(slot.abi);
        for(std::size_t i = 0; i < abi.size(); ++i)
        {
            auto renamed = abi;
            renamed.at(i).name += "_renamed";
            expectRefused(slot.symbol, renamed, std::string(slot.symbol) + " " + abi.at(i).name);
        }
    }
}

TEST(TestAttentionGeneralBwdSignature, RefusesTwoExchangedPointersOfTheMainKernel)
{
    // dK and dV: same kind, same width, adjacent. Only the names tell them apart, and an
    // exchange writes each gradient into the other's tensor.
    auto exchanged = recordedSignature(MAIN_ABI);
    std::swap(exchanged.at(4).name, exchanged.at(5).name);
    expectRefused(MAIN_DISPATCH_SYMBOL, exchanged, "dK and dV exchanged");
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

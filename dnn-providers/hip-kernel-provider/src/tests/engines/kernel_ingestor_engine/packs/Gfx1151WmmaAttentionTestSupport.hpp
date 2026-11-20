// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file Gfx1151WmmaAttentionTestSupport.hpp
 * @brief The scaffolding the hipkernel:Gfx1151WmmaAttention(Decode) matcher suites share:
 *        TestGfx1151WmmaAttentionMatchers (dense graphs) and TestGfx1151WmmaAttentionPaged
 *        (paged graphs).
 *
 * The engines' native symbols (develop's PackSymbols, reached through
 * IngestorPackTestSupport.hpp's matchesGraph / matchesKernel / scoreKernel /
 * dispatchHandler), a fixture owning one graph, the catalog object metadata, the INFO
 * decline-log reader, the kernel-argument helpers with the prefill ABI's head, and the
 * descriptor whose prepare() fails only at the absent archive live here once. Each suite
 * keeps its own graph builder.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::gfx1151_wmma_attention
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::KernelArgument;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;
using hipdnn_plugin_sdk::ingestor::MatchContext;

/// The engine names that head each engine's decline lines.
inline constexpr std::string_view PREFILL_ENGINE = "hipkernel:Gfx1151WmmaAttention";
inline constexpr std::string_view DECODE_ENGINE = "hipkernel:Gfx1151WmmaAttentionDecode";

/// The native symbols one engine registers (hipkernel:Gfx1151WmmaAttention is the
/// prefill, hipkernel:Gfx1151WmmaAttentionDecode the decode); the SDPA graph matchers
/// check the operation themselves and bind no input / output tokens.
inline constexpr PackSymbols PREFILL_SYMBOLS{PREFILL_ENGINE,
                                             "hipkernel.gfx1151_wmma_attention.graph_match",
                                             "",
                                             "hipkernel.gfx1151_wmma_attention.kernel_match",
                                             "hipkernel.gfx1151_wmma_attention.score",
                                             "hipkernel.gfx1151_wmma_attention.dispatch",
                                             "",
                                             "",
                                             ""};
inline constexpr PackSymbols DECODE_SYMBOLS{DECODE_ENGINE,
                                            "hipkernel.gfx1151_wmma_attention_decode.graph_match",
                                            "",
                                            "hipkernel.gfx1151_wmma_attention_decode.kernel_match",
                                            "hipkernel.gfx1151_wmma_attention_decode.score",
                                            "hipkernel.gfx1151_wmma_attention_decode.dispatch",
                                            "",
                                            "",
                                            ""};

/// Packed strides for @p dims, innermost axis last.
inline std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size(); i > 1; --i)
    {
        strides[i - 2] = strides[i - 1] * dims[i - 1];
    }
    return strides;
}

/// The device-pointer alignment a test graph declares for @p uid: 16 bytes unless
/// @p declared names another.
inline int64_t declaredAlignment(const std::map<int64_t, int64_t>& declared, int64_t uid)
{
    const auto found = declared.find(uid);
    return found == declared.end() ? 16 : found->second;
}

/// Further SDPA attributes a test graph sets (FP8, dropout and other features).
using AttributeEdit = std::function<void(data_objects::SdpaAttributesBuilder&)>;
using AttributeUid = void (data_objects::SdpaAttributesBuilder::*)(int64_t);

/// Sets the SDPA attribute @p add to an unused tensor uid.
inline AttributeEdit settingUid(AttributeUid add)
{
    return [add](data_objects::SdpaAttributesBuilder& attributes) { (attributes.*add)(99); };
}

inline DeviceProperties deviceProperties(const std::string& arch, int multiprocessors)
{
    DeviceProperties properties;
    properties.gcnArchName = arch;
    properties.warpSize = 32;
    properties.multiProcessorCount = multiprocessors;
    return properties;
}

/// One graph and the device it is matched for (develop's fixture), with the tokens each
/// engine's graph matcher binds.
class EngineGraphFixture : public kernel_ingestor_engine::testing::GraphFixture
{
public:
    using kernel_ingestor_engine::testing::GraphFixture::GraphFixture;

    /// What @p symbols' engine's graph matcher binds.
    std::optional<BoundTokens> bound(const PackSymbols& symbols = PREFILL_SYMBOLS) const
    {
        return matchesGraph(symbols, context());
    }
};

/// The catalog metadata of one object (configs/gfx1151_wmma_attention.yaml and
/// configs/gfx1151_wmma_attention_decode.yaml); the defaults are a dense prefill object.
struct KernelSpec
{
    std::string kind = "prefill";
    std::string dtype = "FP16";
    int64_t headSize = 128;
    int64_t headSizeV = 128;
    std::string maskMode = "causal";
    std::string bias = "none";
    int64_t blockSize = 256;
    int64_t blockM = 128;
    int64_t dvSplit = 1;
    // 1: the object takes sink_ptr (shipped only for the paged prefill objects).
    int64_t sinks = 0;
    std::string layout = "dense";
    int64_t pageBlockSize = 0;
    int64_t denseQ = 0;
    int64_t seqUsedQ = 0;
};

inline bool isDecode(const KernelSpec& spec)
{
    return spec.kind == "decode";
}

inline const PackSymbols& symbolsFor(const KernelSpec& spec)
{
    return isDecode(spec) ? DECODE_SYMBOLS : PREFILL_SYMBOLS;
}

inline KernelDefinition makeKernel(const KernelSpec& spec)
{
    KernelDefinition kernel;
    kernel.kernelId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-000000000001");
    kernel.packId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-000000000002");
    kernel.dispatchId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-000000000003");
    kernel.metadata = {{"dtype", spec.dtype},
                       {"kind", spec.kind},
                       {"head_size", spec.headSize},
                       {"head_size_v", spec.headSizeV},
                       {"mask_mode", spec.maskMode},
                       {"bias", spec.bias},
                       {"block_size", spec.blockSize},
                       {"block_m", spec.blockM},
                       {"dv_split", spec.dvSplit},
                       {"sinks", spec.sinks},
                       {"layout", spec.layout},
                       {"page_block_size", spec.pageBlockSize},
                       {"dense_q", spec.denseQ},
                       {"seqused_q", spec.seqUsedQ}};
    return kernel;
}

/// A bound integer token; a test failure, and -1, when it is absent.
inline int64_t boundInt(const BoundTokens& bound, std::string_view token)
{
    const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
    EXPECT_TRUE(value.has_value()) << "no bound token " << token;
    return value.value_or(-1);
}

// -----------------------------------------------------------------------------
// Decline logs. hipDNN reports a declined graph only as "no engine configurations", so
// the engines log each decline at INFO as "<engine> declined the graph [<cause>]: <reason>".
// -----------------------------------------------------------------------------

struct LoggedDecline
{
    std::string engine; ///< empty when nothing was logged
    std::string cause;
    std::string reason;
};

/// The first decline line logged at INFO while @p match runs; @p match must decline.
inline LoggedDecline recordDecline(const std::function<bool()>& match)
{
    constexpr std::string_view MARKER = " declined the graph [";
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    EXPECT_FALSE(match()) << "the graph was not declined";
    for(const auto& log : recorder.getRecordedLogs())
    {
        const auto& message = log.message;
        const auto at = message.find(MARKER);
        const auto close = at == std::string::npos ? at : message.find("]: ", at + MARKER.size());
        if(log.severity != HIPDNN_SEV_INFO || close == std::string::npos)
        {
            continue;
        }
        const auto space = message.rfind(' ', at == 0 ? 0 : at - 1);
        const auto engineStart = space == std::string::npos || space >= at ? 0 : space + 1;
        return LoggedDecline{message.substr(engineStart, at - engineStart),
                             message.substr(at + MARKER.size(), close - at - MARKER.size()),
                             message.substr(close + 3)};
    }
    return {};
}

/// Expects @p logged to be @p engine's decline with cause @p cause and a reason that
/// contains @p reason.
inline void expectDecline(const LoggedDecline& logged,
                          std::string_view engine,
                          std::string_view cause,
                          std::string_view reason)
{
    EXPECT_EQ(logged.engine, engine) << "[" << logged.cause << "]: " << logged.reason;
    EXPECT_EQ(logged.cause, cause) << logged.reason;
    EXPECT_NE(logged.reason.find(reason), std::string::npos)
        << "the reason \"" << logged.reason << "\" does not contain \"" << reason << "\"";
}

// -----------------------------------------------------------------------------
// Kernel ABIs. prepare() compares the argument list the native marshals with the
// descriptor's, so a descriptor recording the ABI rocKE declares passes that comparison
// and fails only on the absent archive.
// -----------------------------------------------------------------------------

inline constexpr const char* MISMATCH_MARKER = "is packaged with arguments";

inline KernelArgument pointerArgument(const std::string& name)
{
    return KernelArgument{"global_buffer", 8, 0, name};
}

inline KernelArgument scalarArgument(const std::string& name)
{
    return KernelArgument{"by_value", 4, 0, name};
}

inline void appendScalars(std::vector<KernelArgument>& abi,
                          std::initializer_list<const char*> names)
{
    for(const char* name : names)
    {
        abi.push_back(scalarArgument(name));
    }
}

inline void appendStrides(std::vector<KernelArgument>& abi,
                          std::initializer_list<const char*> tensors)
{
    for(const char* tensor : tensors)
    {
        for(const char* axis : {"batch", "token", "head"})
        {
            abi.push_back(scalarArgument(std::string("stride_") + tensor + "_" + axis));
        }
    }
}

/// The head of wmma_fmha_fwd_v2._declare_params every prefill object shares: the five
/// tensors, the scale, the shape and mask scalars and the strides. Bias, sinks and the
/// paged layouts' arguments follow it.
inline std::vector<KernelArgument> rockePrefillAbiHead()
{
    std::vector<KernelArgument> abi;
    for(const char* name : {"Q", "K", "V", "O", "LSE"})
    {
        abi.push_back(pointerArgument(name));
    }
    abi.push_back(scalarArgument("scale_log2"));
    appendScalars(abi,
                  {"batch",
                   "seqlen_q",
                   "seqlen_k",
                   "num_query_heads",
                   "num_kv_heads",
                   "num_v_heads",
                   "head_dim_q",
                   "head_dim_v",
                   "bottom_right",
                   "window_left",
                   "window_right",
                   "write_lse"});
    appendStrides(abi, {"q", "k", "v", "o", "lse"});
    return abi;
}

/// A kpack descriptor of @p spec recording @p recorded, out of an archive that does not
/// exist in an existing tree: the containment check passes, the signature comparison
/// runs, and loading fails.
inline KernelDefinition kpackDescriptor(const KernelSpec& spec,
                                        const std::vector<KernelArgument>& recorded)
{
    auto kernel = makeKernel(spec);
    kernel.name = "wmma_attention.signature_fixture";
    kernel.source.kind = KernelSourceKind::KPACK;
    kernel.source.library = "there-is-no-archive-here.kpack";
    kernel.source.tocKey = "toc#0";
    kernel.source.symbol = isDecode(spec) ? "wmma_fmha_decode" : "wmma_fmha_fwd_v2";
    kernel.source.signature = recorded;
    const auto tree = std::filesystem::temp_directory_path().string();
    kernel.originDirectory = tree;
    kernel.treeRoot = tree;
    return kernel;
}

/// The handler of the engine @p kernel's own `kind` metadata names.
inline const hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>&
    dispatchHandlerOf(const KernelDefinition& kernel)
{
    const auto kind = kernel.metadata.find("kind");
    const bool decode = kind != kernel.metadata.end()
                        && std::holds_alternative<std::string>(kind->second)
                        && std::get<std::string>(kind->second) == "decode";
    return dispatchHandler(decode ? DECODE_SYMBOLS : PREFILL_SYMBOLS);
}

/// What prepare() throws for @p kernel against the prefill engine's binding of
/// @p fixture's graph; empty when it throws nothing, which no case expects (no archive
/// exists).
inline std::string prepareFailure(const EngineGraphFixture& fixture, const KernelDefinition& kernel)
{
    const auto bound = fixture.bound();
    EXPECT_TRUE(bound.has_value()) << "the fixture graph does not match, so prepare() never ran";
    if(!bound.has_value())
    {
        return {};
    }
    try
    {
        dispatchHandlerOf(kernel).prepare(fixture.context(), *bound, kernel);
    }
    catch(const std::exception& error)
    {
        return error.what();
    }
    return {};
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::gfx1151_wmma_attention

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

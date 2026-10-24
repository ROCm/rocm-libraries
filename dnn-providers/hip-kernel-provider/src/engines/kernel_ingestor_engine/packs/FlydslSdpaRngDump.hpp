// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Redundant with the CMake gate on purpose; see FlydslSdpaRdnaNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>
#include <nlohmann/json.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaDropout.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaShared.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaVarlen.hpp"

/**
 * @file FlydslSdpaRngDump.hpp
 * @brief The RNG dump (`rng_dump`): the dropout keep mask a dropping FlyDSL SDPA engine writes
 *        out, for hipDNN's references to apply as the explicit mask and check the keep rate of.
 *
 * The dump is FLOAT `[B, Hq, Sq, Skv]`, packed row-major, 1 where a key's probability is kept
 * and 0 where it is dropped, holding exactly the bits the attention kernel applied. Every
 * family's mask is a function of (seed, offset, B, Hq, Sq, Skv, threshold) alone
 * (FlydslSdpaDropout.hpp), so it is written by its own small kernel, the
 * `sdpa_dropout_mask` op's object, which draws the same counters through the same Philox
 * (`attention_rdna4/philox.py`). A native that drops launches it on the attention launch's
 * stream, after it, when the graph names a dump; the attention objects are unchanged. It writes
 * the raw keep bit at every element, masked keys, padded rows and keys past a batch's length
 * included: those are draws the attention kernel makes and discards, and hipDNN's harness
 * counts the whole buffer when it checks the keep rate.
 *
 * The writer's object ships as its own op beside the attention ops in each arch shard
 * (`FlyDSL/sdpa_dropout_mask/<target>/`), under an engine that serves no graph; a native finds
 * it from its own kernel's directory (dropoutMaskPackPath()).
 */
namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_sdpa
{

/// The uid an RNG-dump token holds when the graph names no dump.
inline constexpr int64_t NO_RNG_DUMP = -1;

/// Keys one Philox draw covers: the writer runs one thread per draw.
inline constexpr int64_t RNG_DUMP_KEYS_PER_DRAW = 4;

/// The writer's op, the directory its objects sit in beside the attention ops', and its pack.
inline constexpr std::string_view DROPOUT_MASK_OP = "sdpa_dropout_mask";
inline constexpr std::string_view DROPOUT_MASK_PACK_FILE = "flydsl_sdpa_dropout_mask.kdp.json";
inline constexpr std::string_view DROPOUT_MASK_THREADS_FIELD = "block_threads";

/// The writer's draws, one per (plane, row, 4 keys), or nullopt when the count overflows.
inline std::optional<int64_t>
    rngDumpDraws(int64_t batch, int64_t heads, int64_t seqLenQ, int64_t seqLenKv)
{
    return checkedProduct(
        {batch, heads, seqLenQ, (seqLenKv + RNG_DUMP_KEYS_PER_DRAW - 1) / RNG_DUMP_KEYS_PER_DRAW});
}

/**
 * @brief Whether @p uid names a dump the writer can fill: FLOAT `[B, Hq, Sq, Skv]` with the
 *        graph's extents, packed row-major (an axis of one element may state any stride, as no
 *        index ever multiplies it), a plain device tensor at a 4-byte-aligned base, and few enough
 *        draws for the writer's 32-bit grid.
 *
 * The writer stores element (b, h, i, j) at `((b * Hq + h) * Sq + i) * Skv + j` as one f32, so
 * another dtype, another layout, a host buffer or a ragged one would be written wrong.
 */
inline bool isServableRngDump(const MatchContext& context,
                              int64_t uid,
                              int64_t batch,
                              int64_t heads,
                              int64_t seqLenQ,
                              int64_t seqLenKv)
{
    const auto* dump = findTensor(context, uid);
    if(dump == nullptr || dump->dims() == nullptr || dump->strides() == nullptr
       || dump->dims()->size() != SDPA_RANK || dump->strides()->size() != SDPA_RANK
       || dump->data_type() != data_objects::DataType::FLOAT || dump->virtual_()
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(dump)
       || dump->ragged_offset_tensor_uid().has_value()
       || !declaresAlignment(*dump, static_cast<int64_t>(sizeof(float))))
    {
        return false;
    }
    const auto plane = checkedProduct({seqLenQ, seqLenKv});
    const auto batchStride = checkedProduct({heads, seqLenQ, seqLenKv});
    if(!plane.has_value() || !batchStride.has_value())
    {
        return false;
    }
    const std::array<int64_t, SDPA_RANK> extents{batch, heads, seqLenQ, seqLenKv};
    const std::array<int64_t, SDPA_RANK> packed{*batchStride, *plane, seqLenKv, 1};
    for(uint32_t axis = 0; axis < SDPA_RANK; ++axis)
    {
        if(dump->dims()->Get(axis) != extents.at(axis))
        {
            return false;
        }
        if(extents.at(axis) > 1 && dump->strides()->Get(axis) != packed.at(axis))
        {
            return false;
        }
    }
    const auto draws = rngDumpDraws(batch, heads, seqLenQ, seqLenKv);
    return draws.has_value() && *draws <= INT32_LIMIT;
}

/**
 * @brief The graph's RNG dump: its uid, NO_RNG_DUMP when it names none, or nullopt for one the
 *        engines do not write.
 *
 * Written for Philox dropout that drops (a probability above 0, with its seed and offset;
 * servableDropout() has gated those), over dense or padded operands. Declined, each a wrong
 * answer rather than an error if served: a dump with no drop (a probability of 0 runs the
 * undropping objects, which apply no mask, while the writer would still draw one), with packed
 * (THD) operands (the kernels' packed dropout is later work), a dump that is also one of the
 * node's other tensors, and one isServableRngDump() refuses. The family matchers decline what
 * they do not drop; which objects run it is each kernel matcher's to say
 * (hasDropoutMaskWriter()).
 */
inline std::optional<int64_t> servableRngDump(const MatchContext& context,
                                              const data_objects::SdpaAttributes& attributes,
                                              const SdpaDropout& dropout,
                                              const SdpaVarlen& varlen,
                                              int64_t batch,
                                              int64_t heads,
                                              int64_t seqLenQ,
                                              int64_t seqLenKv)
{
    if(!attributes.rng_dump_tensor_uid().has_value())
    {
        return NO_RNG_DUMP;
    }
    const int64_t uid = *attributes.rng_dump_tensor_uid();
    if(!dropout.active() || varlen.packed())
    {
        return std::nullopt;
    }
    for(const int64_t other : {attributes.q_tensor_uid(),
                               attributes.k_tensor_uid(),
                               attributes.v_tensor_uid(),
                               attributes.o_tensor_uid(),
                               attributes.stats_tensor_uid().value_or(NO_RNG_DUMP)})
    {
        if(uid == other)
        {
            return std::nullopt;
        }
    }
    if(!isServableRngDump(context, uid, batch, heads, seqLenQ, seqLenKv))
    {
        return std::nullopt;
    }
    return uid;
}

/// Where the writer's pack sits for the attention kernel @p attention:
/// `<op dir>/../sdpa_dropout_mask/<target>/flydsl_sdpa_dropout_mask.kdp.json`, the same target
/// directory under the writer's op as the attention kernel's under its own.
inline std::filesystem::path dropoutMaskPackPath(const KernelDefinition& attention)
{
    auto origin = attention.originDirectory.lexically_normal();
    if(origin.filename().empty())
    {
        origin = origin.parent_path();
    }
    return origin.parent_path().parent_path() / DROPOUT_MASK_OP / origin.filename()
           / DROPOUT_MASK_PACK_FILE;
}

/// Whether @p attention's shard ships the writer, so a dropping graph's dump can be served
/// with it: a kernel matcher reads this for a graph that names a dump.
inline bool hasDropoutMaskWriter(const KernelDefinition& attention)
{
    std::error_code ignored;
    return !attention.originDirectory.empty()
           && std::filesystem::is_regular_file(dropoutMaskPackPath(attention), ignored);
}

/// The writer's argument list, as `gen_sdpa_dropout_mask.py` asserts it: the dump, seed and
/// offset1 pointers, offset2 (i64), the keep threshold, the planes (B * Hq), Sq and Skv (i32).
inline const std::vector<KernelArgument>& dropoutMaskSignature()
{
    static const std::vector<KernelArgument> s_signature{{"global_buffer", 8, 0, ""},
                                                         {"global_buffer", 8, 8, ""},
                                                         {"global_buffer", 8, 16, ""},
                                                         {"by_value", 8, 24, ""},
                                                         {"by_value", 4, 32, ""},
                                                         {"by_value", 4, 36, ""},
                                                         {"by_value", 4, 40, ""},
                                                         {"by_value", 4, 44, ""}};
    return s_signature;
}

/**
 * @brief The writer's kernel for a device of @p arch, read from the pack beside
 *        @p attention's (dropoutMaskPackPath()) as the packer wrote it: its kpack source,
 *        name, id and workgroup size.
 *
 * @throws HipdnnPluginException when the pack is absent, unreadable or names no kernel for
 *         @p arch; the kernel matchers decline a dump on a shard without it first.
 */
inline KernelDefinition dropoutMaskKernel(const KernelDefinition& attention,
                                          const std::string& arch)
{
    const auto path = dropoutMaskPackPath(attention);
    const auto where = path.string();
    const auto fail = [&where](const std::string& why) {
        return hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "flydsl_sdpa rng_dump: " + where + ": " + why);
    };
    std::ifstream in(path);
    if(!in)
    {
        throw fail("the dropout mask writer's pack cannot be read");
    }
    const auto document = nlohmann::json::parse(in, nullptr, false);
    if(document.is_discarded() || !document.is_object() || !document.contains("kernelDescriptors"))
    {
        throw fail("not a kernel descriptor pack");
    }
    const auto packArch = document.value("arch", std::vector<std::string>{});
    for(const auto& entry : document.at("kernelDescriptors"))
    {
        if(!entry.is_object() || !entry.contains("kernel_source"))
        {
            continue;
        }
        const auto arches = entry.value("arch", packArch);
        if(std::find(arches.begin(), arches.end(), arch) == arches.end())
        {
            continue;
        }
        try
        {
            KernelDefinition mask;
            mask.kernelId
                = hipdnn_flatbuffers_sdk::utilities::parseUuid(entry.value("id", std::string()));
            mask.source = hipdnn_plugin_sdk::ingestor::detail::parseKernelSource(
                entry.at("kernel_source"), where);
            mask.name = entry.value("name", std::string());
            mask.arch = arches;
            mask.originDirectory = path.parent_path();
            mask.treeRoot = attention.treeRoot;
            mask.metadata[std::string(DROPOUT_MASK_THREADS_FIELD)]
                = entry.at("metadata").at(std::string(DROPOUT_MASK_THREADS_FIELD)).get<int64_t>();
            return mask;
        }
        catch(const std::exception& error)
        {
            throw fail(std::string("its kernel for ") + arch + " is malformed: " + error.what());
        }
    }
    throw fail("names no dropout mask writer for " + arch);
}

/// The writer's launch for one plan: its code and the extents it draws over, resolved at
/// prepare; launch() reads only the dump's buffer and the attention launch's Philox arguments.
class DropoutMaskLaunch
{
public:
    DropoutMaskLaunch(
        IngestorKernelCode code, int64_t dump, int64_t planes, int64_t seqLenQ, int64_t seqLenKv)
        : _code(std::move(code))
        , _dump(dump)
        , _planes(static_cast<int32_t>(planes))
        , _seqLenQ(static_cast<int32_t>(seqLenQ))
        , _seqLenKv(static_cast<int32_t>(seqLenKv))
    {
    }

    /// Writes the mask the attention launch on @p stream drew with @p seed, @p offset1 (null
    /// reads as 0), @p offset2 and @p threshold. On the same stream, so it orders after it and
    /// reads a seed or offset the attention launch's workspace holds.
    void launch(hipStream_t stream,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* seed,
                void* offset1,
                int64_t offset2,
                int32_t threshold) const
    {
        void* dump
            = hipdnn_plugin_sdk::findDeviceBuffer(_dump, deviceBuffers, numDeviceBuffers).ptr;
        // Changing this argument list means changing dropoutMaskSignature() with it.
        _code.kernelForStream(stream).launch(
            stream, dump, seed, offset1, offset2, threshold, _planes, _seqLenQ, _seqLenKv);
    }

private:
    IngestorKernelCode _code;
    int64_t _dump;
    int32_t _planes;
    int32_t _seqLenQ;
    int32_t _seqLenKv;
};

/**
 * @brief The writer's launch for a graph whose dump @p dump the match bound, or nullopt for a
 *        graph without one: the writer's kernel beside @p attention, loaded through
 *        @p kpackLoader, one thread per draw.
 */
inline std::optional<DropoutMaskLaunch>
    prepareDropoutMask(const compilation::KpackKernelLoader& kpackLoader,
                       const MatchContext& context,
                       const KernelDefinition& attention,
                       int64_t dump,
                       int64_t batch,
                       int64_t heads,
                       int64_t seqLenQ,
                       int64_t seqLenKv)
{
    if(dump == NO_RNG_DUMP)
    {
        return std::nullopt;
    }
    const auto mask = dropoutMaskKernel(
        attention,
        std::string(hipdnn_plugin_sdk::stripArchFeatures(context.deviceProperties.gcnArchName)));
    const int64_t threads = mask.getIntMetadata(std::string(DROPOUT_MASK_THREADS_FIELD));
    const auto draws = rngDumpDraws(batch, heads, seqLenQ, seqLenKv);
    const auto planes = checkedProduct({batch, heads});
    if(threads <= 0 || !draws.has_value() || *draws > INT32_LIMIT || !planes.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "flydsl_sdpa rng_dump: the dropout mask writer's launch overflows its grid");
    }
    auto code = buildIngestorKernelCode(kpackLoader, context, mask, dropoutMaskSignature());
    code.setBlockSize(static_cast<unsigned int>(threads), 1, 1);
    code.setGridSize(static_cast<unsigned int>((*draws + threads - 1) / threads), 1, 1);
    return DropoutMaskLaunch(std::move(code), dump, *planes, seqLenQ, seqLenKv);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_sdpa

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL

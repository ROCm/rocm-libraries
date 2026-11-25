// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA) \
    && (defined(HKP_DETERMINISM_FLYDSL_SDPA_RDNA_DECODE)                  \
        || defined(HKP_DETERMINISM_FLYDSL_SDPA_CDNA_DECODE)               \
        || defined(HKP_DETERMINISM_FLYDSL_SDPA_CDNA_DUALWAVE))

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "SdpaDeterminismHarness.hpp"

/**
 * @file IntegrationGpuSdpaDeterminismFlydslFwd.cpp
 * @brief The determinism rows of the FlyDSL forward decode and dual-wave engines
 *        hipkernel:flydsl_sdpa_cdna_decode, _cdna_dualwave and _rdna_decode. CMake defines
 *        HKP_DETERMINISM_FLYDSL_SDPA_<OP> for each op the product root ships FlyDSL content
 *        for, and each table below exists only under its definition.
 *
 * One row per launch-structure class and dtype. The classes come from a source read of the
 * split-KV decode kernels with their ascending-order merge kernel and the dual-wave kernel.
 *
 * Which shapes an engine admits is read from its matchers (FlydslSdpaCdnaCommon.hpp,
 * FlydslSdpaRdnaDecodeNative.cpp) and from the objects each target ships
 * (flydsl/generators/_instances.py, the SOURCE.md beside each target's descriptors). A row the
 * pinned engine does not rank FAILS at run time by design.
 *
 * Split counts (candidateSplits, decodeSplits) depend on the device's compute-unit count, so a
 * claim is run-to-run on one device. Rows whose split count is the cap or the work bound give
 * the same count on every device; the "device bound" rows do not.
 *
 * hipkernel:flydsl_sdpa_cdna, _rdna and _aiw are not rowed here: the head-pad copy they wrap can
 * write past a packed operand's region when a batch's origins run past B*S tokens, so they keep
 * the absent numerical_notes field.
 */

namespace
{

using namespace hip_kernel_provider::sdpa_determinism;
using hip_kernel_provider::kernel_ingestor_engine::integration::FrontendSdpaSpec;
using hipdnn_frontend::DataType;

struct Shape
{
    int64_t batch;
    int64_t queryHeads;
    int64_t kvHeads;
    int64_t seqQ;
    int64_t seqKv;
    int64_t headSize;
};

/// A dense BSHD forward graph with the LSE; causal is bottom-right (the spec's alignment).
[[maybe_unused]] FrontendSdpaSpec shaped(DataType dataType, const Shape& shape, bool causal)
{
    FrontendSdpaSpec spec;
    spec.dataType = dataType;
    spec.batch = shape.batch;
    spec.queryHeads = shape.queryHeads;
    spec.kvHeads = shape.kvHeads;
    spec.seqQ = shape.seqQ;
    spec.seqKv = shape.seqKv;
    spec.headSize = shape.headSize;
    spec.rightBound = causal ? std::optional<int64_t>(0) : std::nullopt;
    spec.withStats = true;
    return spec;
}

/// Per-batch lengths with padding_mask. Every batch keeps lens Q <= lens KV so no live row is
/// keyless (a keyless row's LSE is -inf, which no row here allows).
[[maybe_unused]] FrontendSdpaSpec
    padded(FrontendSdpaSpec spec, std::vector<int32_t> lensQ, std::vector<int32_t> lensKv)
{
    spec.seqLensQ = std::move(lensQ);
    spec.seqLensKv = std::move(lensKv);
    return spec;
}

/// An additive bias [B, Hq, Sq, Skv] in @p biasType.
[[maybe_unused]] FrontendSdpaSpec biased(FrontendSdpaSpec spec, DataType biasType)
{
    spec.biasDataType = biasType;
    return spec;
}

/// cuDNN's FLOAT attention sink.
[[maybe_unused]] FrontendSdpaSpec sunk(FrontendSdpaSpec spec)
{
    spec.sinkDataType = DataType::FLOAT;
    return spec;
}

/// A V and O head size of their own.
[[maybe_unused]] FrontendSdpaSpec valueWidth(FrontendSdpaSpec spec, int64_t valueHeadSize)
{
    spec.valueHeadSize = valueHeadSize;
    return spec;
}

RowMeta metaFor(const char* engine,
                const char* caseName,
                const std::vector<std::string>& archs,
                KnobList knobs,
                KnobClass knobClass)
{
    RowMeta meta;
    meta.engine = engine;
    meta.caseName = caseName;
    meta.archs = archs;
    meta.knobs = std::move(knobs);
    meta.knobClass = knobClass;
    return meta;
}

/// A 16-bit row through the frontend builder.
[[maybe_unused]] void add(std::vector<DeterminismRow>& rows,
                          const char* engine,
                          const char* caseName,
                          const std::vector<std::string>& archs,
                          FrontendSdpaSpec spec,
                          KnobList knobs = {},
                          KnobClass knobClass = KnobClass::TILE_SCHEDULE)
{
    spec.name = caseName;
    rows.push_back(
        forwardRow(metaFor(engine, caseName, archs, std::move(knobs), knobClass), std::move(spec)));
}

[[maybe_unused]] constexpr auto BF16 = DataType::BFLOAT16;
[[maybe_unused]] constexpr auto F16 = DataType::HALF;
[[maybe_unused]] constexpr auto F32 = DataType::FLOAT;

std::vector<DeterminismRow> flydslForwardRows()
{
    std::vector<DeterminismRow> rows;

#ifdef HKP_DETERMINISM_FLYDSL_SDPA_CDNA_DECODE
    {
        // The generic kernel in split-KV mode: a main launch writing f32 partials and LSEs to the
        // workspace, then the merge kernel (ascending split loops, one writer per O element). The
        // split count is min(device bound, work bound, 64): candidateSplits,
        // FlydslSdpaCdnaCommon.hpp:1391-1404.
        //  - SplitCap: B1 Hq2 Hkv1 Skv16384 -> work bound 64 on every device (device bound >= 104).
        //  - DeviceBoundSplit: B4 Hq32 Hkv8 Skv8192 -> grid 32 workgroups, so the split count is
        //    units / 32 (6 on a 104-CU gfx90a, 19 on gfx942, 16 on gfx950): CU dependent.
        //  - SingleSplit: Skv128 -> one split writes O directly, no merge (U:5696-5700).
        //  - EmptySplit: Skv1600 -> 25 key tiles, 6 splits of 5 tiles, the last split empty
        //    [INFERENCE from the tile arithmetic: the main kernel's own BLOCK_N was not read]; the
        //    padded rows below also leave splits past a batch's short key length empty.
        //  - Wide: absorbed MLA 576/512 and 512/512 with dv_split 4 (gfx950 bucket 1024, gfx942
        //    and gfx90a bucket 768 with k_chunk 256): _instances.py:1374-1399,
        //    IntegrationGpuFlydslSdpaFrontendFeatures.cpp:351-372 pins this engine at hd512.
        //  - Declines (never rows): f32, dropout, packed, bias, V heads of their own
        //    (FlydslSdpaCdnaCommon.hpp:1620-1639, :1718-1722); padded needs a bottom-right
        //    diagonal when causal.
        //  - Objects: exact 64/128/256 and sink 64/128 on every arch, SOURCE.md of
        //    sdpa_cdna_decode.
        constexpr const char* DECODE = "hipkernel:flydsl_sdpa_cdna_decode";
        const std::vector<std::string> archs = {"gfx90a", "gfx942", "gfx950"};
        const auto split = KnobClass::STRUCTURE;

        add(rows,
            DECODE,
            "Bf16Hd64SplitCap",
            archs,
            shaped(BF16, {1, 2, 1, 1, 16384, 64}, false),
            {},
            split);
        add(rows,
            DECODE,
            "F16Hd128SplitCausal",
            archs,
            shaped(F16, {1, 8, 2, 1, 8192, 128}, true),
            {},
            split);
        add(rows,
            DECODE,
            "Bf16Hd256Split",
            archs,
            shaped(BF16, {1, 4, 1, 1, 8192, 256}, false),
            {},
            split);
        add(rows,
            DECODE,
            "F16Hd256SplitCausal",
            archs,
            shaped(F16, {1, 4, 1, 1, 8192, 256}, true),
            {},
            split);
        add(rows,
            DECODE,
            "F16Hd64DeviceBoundSplit",
            archs,
            shaped(F16, {4, 32, 8, 1, 8192, 64}, false),
            {},
            split);
        add(rows,
            DECODE,
            "Bf16Hd128SingleSplit",
            archs,
            shaped(BF16, {1, 4, 1, 1, 128, 128}, false),
            {},
            split);
        add(rows,
            DECODE,
            "F16Hd64EmptySplit",
            archs,
            shaped(F16, {1, 1, 1, 1, 1600, 64}, false),
            {},
            split);
        add(rows,
            DECODE,
            "Bf16Hd64Chunk16Causal",
            archs,
            shaped(BF16, {1, 8, 2, 16, 4096, 64}, true),
            {},
            split);
        // More than one row tile: the packed group rows (Hq/Hkv * Sq = 128) exceed block_m 64, so
        // the grid has two row tiles per KV head (candidateGrid pack_gqa branch,
        // FlydslSdpaCdnaCommon.hpp:1363-1375). The engine serves any Sq (the matcher bounds no
        // Sq; frontend lane pins it at Sq 289, FrontendFeatures.cpp:351-372).
        add(rows,
            DECODE,
            "F16Hd64MultiRowTileSplit",
            archs,
            shaped(F16, {1, 16, 1, 8, 4096, 64}, false),
            {},
            split);
        add(rows,
            DECODE,
            "Bf16Hd64Sink",
            archs,
            sunk(shaped(BF16, {1, 8, 2, 1, 8192, 64}, false)),
            {},
            split);
        add(rows,
            DECODE,
            "F16Hd128SinkCausal",
            archs,
            sunk(shaped(F16, {1, 8, 2, 1, 8192, 128}, true)),
            {},
            split);
        add(rows,
            DECODE,
            "Bf16Hd64Padded",
            archs,
            padded(shaped(BF16, {2, 8, 2, 8, 4096, 64}, false), {8, 3}, {4096, 1000}),
            {},
            split);
        add(rows,
            DECODE,
            "F16Hd128PaddedCausal",
            archs,
            padded(shaped(F16, {2, 8, 2, 8, 4096, 128}, true), {8, 3}, {4096, 1000}),
            {},
            split);
        add(rows,
            DECODE,
            "Bf16Hd576V512Wide",
            archs,
            valueWidth(shaped(BF16, {1, 16, 1, 1, 4096, 576}, false), 512),
            {},
            split);
        add(rows,
            DECODE,
            "F16Hd576V512WideCausal",
            archs,
            valueWidth(shaped(F16, {1, 16, 1, 1, 4096, 576}, true), 512),
            {},
            split);
        add(rows,
            DECODE,
            "F16Hd512Wide",
            archs,
            shaped(F16, {1, 8, 1, 1, 4096, 512}, false),
            {},
            split);
    }
#endif

#ifdef HKP_DETERMINISM_FLYDSL_SDPA_CDNA_DUALWAVE
    {
        // gfx950's dual-wave kernel: one launch, split-K compiled out of every shipped object
        // (num_kv_splits 1) [INFERENCE: the objects were not scanned], row reductions by
        // fixed trees. It declines padding,
        // packing, sink, dropout, f32, windows and strided LSEs (FlydslSdpaCdnaCommon.hpp:1597-1682),
        // so none of those is a row. Objects: _instances.py:1160-1180 (hd64/128), :1261-1272 (hd256,
        // 4 waves), :1289-1300 (bias hd64/128); SOURCE.md sdpa_cdna_dualwave/gfx950 rows 36-46.
        // Causal at hd128 ships an offset-0 build (any_diagonal 0) and an any-diagonal build: the
        // Sq == Skv row takes the first, the Skv > Sq row the second
        // (FlydslSdpaCdnaCommon.hpp:1590-1596). The hd256 shapes are the frontend lane's
        // (IntegrationGpuFlydslSdpaFrontend.cpp:209-226).
        constexpr const char* DUALWAVE = "hipkernel:flydsl_sdpa_cdna_dualwave";
        const std::vector<std::string> archs = {"gfx950"};

        add(rows,
            DUALWAVE,
            "Bf16Hd64NonCausal",
            archs,
            shaped(BF16, {2, 8, 2, 512, 512, 64}, false));
        add(rows, DUALWAVE, "F16Hd64Causal", archs, shaped(F16, {2, 8, 2, 512, 512, 64}, true));
        add(rows,
            DUALWAVE,
            "Bf16Hd128NonCausal",
            archs,
            shaped(BF16, {2, 8, 2, 512, 512, 128}, false));
        add(rows,
            DUALWAVE,
            "F16Hd128CausalOffsetZero",
            archs,
            shaped(F16, {2, 8, 2, 512, 512, 128}, true));
        add(rows,
            DUALWAVE,
            "Bf16Hd128CausalAnyDiagonal",
            archs,
            shaped(BF16, {2, 8, 2, 256, 512, 128}, true));
        add(rows,
            DUALWAVE,
            "Bf16Hd256NonCausal",
            archs,
            shaped(BF16, {2, 8, 2, 640, 700, 256}, false));
        add(rows, DUALWAVE, "F16Hd256Causal", archs, shaped(F16, {2, 8, 2, 300, 700, 256}, true));
        add(rows,
            DUALWAVE,
            "Bf16Hd64Bias",
            archs,
            biased(shaped(BF16, {2, 8, 2, 256, 256, 64}, false), BF16));
        add(rows,
            DUALWAVE,
            "F16Hd128BiasCausal",
            archs,
            biased(shaped(F16, {2, 8, 2, 256, 256, 128}, true), F16));
        // Skv large enough for any split-K a shipped object could carry.
        add(rows,
            DUALWAVE,
            "Bf16Hd128LongKeys",
            archs,
            shaped(BF16, {1, 4, 4, 512, 16384, 128}, false),
            {},
            KnobClass::STRUCTURE);
    }
#endif

#ifdef HKP_DETERMINISM_FLYDSL_SDPA_RDNA_DECODE
    {
        // The RDNA decode kernel: split-KV main launch with an in-workgroup wave combine
        // (w = 0..3, compile order) and the merge kernel (ascending splits, one workgroup per
        // (b, h, s)). Splits =
        // min(work bound, device bound, 64), decodeSplits, FlydslSdpaRdnaDecodeNative.cpp:137-146;
        // the work bound needs Skv >= 2048 for a second split, so Skv 16384 gives 16 splits where
        // the device bound (2 x the reported unit count over B*Hkv) is larger. The column split
        // (dv_split 2 at hd256, 4 at the wide objects) is a launch-set class
        // (_instances.py:407-416). Declined, so never rows: packed (:273-279), f32 (RdnaCommon
        // :515-521), dropout (no decode object drops). Kernel matcher :264-311.
        constexpr const char* DECODE = "hipkernel:flydsl_sdpa_rdna_decode";
        const std::vector<std::string> archs = {"gfx1100", "gfx1201"};
        const auto split = KnobClass::STRUCTURE;

        add(rows,
            DECODE,
            "Bf16Hd64Split",
            archs,
            shaped(BF16, {1, 4, 2, 1, 16384, 64}, false),
            {},
            split);
        add(rows,
            DECODE,
            "F16Hd128SplitCausal",
            archs,
            shaped(F16, {1, 4, 2, 1, 16384, 128}, true),
            {},
            split);
        add(rows,
            DECODE,
            "Bf16Hd256DvSplit",
            archs,
            shaped(BF16, {1, 4, 1, 1, 16384, 256}, false),
            {},
            split);
        add(rows,
            DECODE,
            "F16Hd256DvSplitCausal",
            archs,
            shaped(F16, {1, 4, 1, 1, 16384, 256}, true),
            {},
            split);
        add(rows,
            DECODE,
            "Bf16Hd576V512WideDv4",
            archs,
            valueWidth(shaped(BF16, {1, 16, 1, 1, 8192, 576}, false), 512),
            {},
            split);
        add(rows,
            DECODE,
            "F16Hd512WideDv4",
            archs,
            shaped(F16, {1, 8, 1, 1, 8192, 512}, false),
            {},
            split);
        add(rows,
            DECODE,
            "Bf16Hd384GenericWideDv4",
            archs,
            shaped(BF16, {1, 4, 1, 1, 8192, 384}, false),
            {},
            split);
        add(rows,
            DECODE,
            "F16Hd200GenericDv2",
            archs,
            shaped(F16, {1, 4, 2, 1, 8192, 200}, false),
            {},
            split);
        add(rows,
            DECODE,
            "Bf16Hd64Sink",
            archs,
            sunk(shaped(BF16, {1, 4, 2, 1, 8192, 64}, false)),
            {},
            split);
        add(rows,
            DECODE,
            "F16Hd128SinkCausal",
            archs,
            sunk(shaped(F16, {1, 4, 2, 1, 8192, 128}, true)),
            {},
            split);
        add(rows,
            DECODE,
            "Bf16Hd64Bias",
            archs,
            biased(shaped(BF16, {1, 4, 2, 1, 16384, 64}, false), BF16),
            {},
            split);
        add(rows,
            DECODE,
            "F16Hd128F32Bias",
            archs,
            biased(shaped(F16, {1, 4, 2, 1, 16384, 128}, false), F32),
            {},
            split);
        add(rows,
            DECODE,
            "Bf16Hd64Padded",
            archs,
            padded(shaped(BF16, {2, 8, 2, 4, 4096, 64}, false), {4, 2}, {4096, 1500}),
            {},
            split);
        add(rows,
            DECODE,
            "F16Hd128Padded",
            archs,
            padded(shaped(F16, {2, 8, 2, 4, 4096, 128}, false), {4, 2}, {4096, 1500}),
            {},
            split);
        add(rows,
            DECODE,
            "Bf16Hd64SingleSplit",
            archs,
            shaped(BF16, {1, 4, 2, 1, 1024, 64}, false),
            {},
            split);
        add(rows,
            DECODE,
            "F16Hd64Sq8Split",
            archs,
            shaped(F16, {1, 4, 2, 8, 8192, 64}, false),
            {},
            split);
        // More than one 16-row tile on a non-bias (row-tiled) object: (Hq/Hkv)*Sq = 32 rows, two
        // tiles (decodeRowTiles / decodeRowTiled / decodeServes, FlydslSdpaRdnaDecodeNative.cpp:
        // 120-131, 240-255); bias objects are built for one tile.
        add(rows,
            DECODE,
            "Bf16Hd64MultiRowTile",
            archs,
            shaped(BF16, {1, 8, 1, 4, 8192, 64}, false),
            {},
            split);
        add(rows,
            DECODE,
            "F16Hd128MultiRowTileCausal",
            archs,
            shaped(F16, {1, 8, 1, 4, 8192, 128}, true),
            {},
            split);
    }
#endif

    return rows;
}

[[maybe_unused]] const bool REGISTERED = registerRows(flydslForwardRows());

} // namespace

#endif

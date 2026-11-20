// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)                        \
    && (defined(HKP_DETERMINISM_FLYDSL_SDPA_RDNA)                                                \
        || defined(HKP_DETERMINISM_FLYDSL_SDPA_RDNA_DECODE)                                      \
        || defined(HKP_DETERMINISM_FLYDSL_SDPA_AIW) || defined(HKP_DETERMINISM_FLYDSL_SDPA_CDNA) \
        || defined(HKP_DETERMINISM_FLYDSL_SDPA_CDNA_DECODE)                                      \
        || defined(HKP_DETERMINISM_FLYDSL_SDPA_CDNA_DUALWAVE))

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "SdpaDeterminismHarness.hpp"

/**
 * @file IntegrationGpuSdpaDeterminismFlydslFwd.cpp
 * @brief The determinism rows of the FlyDSL forward engines hipkernel:flydsl_sdpa_rdna,
 *        _rdna_decode, _aiw, _cdna, _cdna_decode and _cdna_dualwave. CMake defines
 *        HKP_DETERMINISM_FLYDSL_SDPA_<OP> for each op the product root ships FlyDSL content
 *        for, and each table below exists only under its definition.
 *
 * One row per launch-structure class and dtype (plan section 3.1). The classes come from the
 * line-level trace of every kernel family (trace-TraceFlydslFwd): the unsplit prefill kernel, the
 * column-split (dv_split) and head-pair (hd_pair) schedules, the split-KV decode kernels with
 * their ascending-order merge, the dual-wave kernel, the aiw kernel's QK shards, and the helper
 * launches (head_pad copies, the dropout keep-mask writer, the padded-row fill).
 *
 * Which shapes an engine admits is read from its matchers (FlydslSdpaCdnaCommon.hpp,
 * FlydslSdpaRdnaCommon.hpp, FlydslSdpaRdnaDecodeNative.cpp, FlydslSdpaAiwNative.cpp) and from the
 * objects each target ships (flydsl/generators/_instances.py, the SOURCE.md beside each target's
 * descriptors). A row the pinned engine does not rank FAILS at run time by design.
 *
 * Split counts (candidateSplits, decodeSplits) depend on the device's compute-unit count, so a
 * claim is run-to-run on one device. Rows whose split count is the cap or the work bound give
 * the same count on every device; the "device bound" rows do not.
 *
 * Dropout rows run the engine's Philox objects and read the RNG dump back; a 16-bit dropout
 * graph is served by flydsl_sdpa_rdna on gfx11 only (SDPA_RDNA_DROPOUT_TARGETS) and by
 * flydsl_sdpa_aiw on gfx12, so the rdna dropout rows list gfx1100 only.
 */

namespace
{

using namespace hip_kernel_provider::sdpa_determinism;
using hip_kernel_provider::kernel_ingestor_engine::integration::FrontendDropout;
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

/// Packed THD slots (compact tables), one length per batch for Q/O and for K/V.
[[maybe_unused]] FrontendSdpaSpec
    packedSlots(FrontendSdpaSpec spec, std::vector<int32_t> slotsQ, std::vector<int32_t> slotsKv)
{
    spec.packedLensQ = std::move(slotsQ);
    spec.packedLensKv = std::move(slotsKv);
    return spec;
}

/// Philox dropout (p 0.2, fixed seed and offset on the device) with the RNG dump.
[[maybe_unused]] FrontendSdpaSpec dropping(FrontendSdpaSpec spec)
{
    spec.dropout = FrontendDropout{};
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

/// An f32 row: Q, K, V and O FLOAT through A0's builder.
[[maybe_unused]] void addF32(std::vector<DeterminismRow>& rows,
                             const char* engine,
                             const char* caseName,
                             const std::vector<std::string>& archs,
                             FrontendSdpaSpec spec,
                             KnobList knobs = {},
                             KnobClass knobClass = KnobClass::TILE_SCHEDULE)
{
    spec.name = caseName;
    DeterminismFwdSpec determinism;
    determinism.shape = std::move(spec);
    determinism.dtypes.qkvType = DataType::FLOAT;
    determinism.dtypes.oType = DataType::FLOAT;
    rows.push_back(forwardRow(metaFor(engine, caseName, archs, std::move(knobs), knobClass),
                              std::move(determinism)));
}

[[maybe_unused]] constexpr auto BF16 = DataType::BFLOAT16;
[[maybe_unused]] constexpr auto F16 = DataType::HALF;
[[maybe_unused]] constexpr auto F32 = DataType::FLOAT;

// Shapes shared by several engines.
[[maybe_unused]] constexpr Shape GQA_256{2, 8, 2, 256, 256, 64};
[[maybe_unused]] constexpr Shape GQA_128{2, 8, 2, 128, 128, 64};
// AOTriton's prime head dims at their own width (the frontend lane's odd-head-dim base).
[[maybe_unused]] constexpr Shape ODD_BASE{2, 8, 2, 67, 143, 64};

[[maybe_unused]] Shape withHead(Shape shape, int64_t headSize)
{
    shape.headSize = headSize;
    return shape;
}

std::vector<DeterminismRow> flydslForwardRows()
{
    std::vector<DeterminismRow> rows;

#ifdef HKP_DETERMINISM_FLYDSL_SDPA_CDNA
    {
        // The generic prefill kernel: one launch, one writer per O element, no workspace
        // (trace cdna: U:3835, U:5686-5790). Evidence per class:
        //  - tiles hd64/128/256: FlydslSdpaCdnaCommon.hpp:1746-1754 (exact head) with the tier
        //    tables _instances.py:1181-1256 (gfx90a/gfx942/gfx950 each ship every class below);
        //    hd256 on gfx942/gfx90a stages K in 128-column chunks (k_chunk), :1235-1254.
        //  - hd512 prefill with O's columns split four ways (dv_split 4, k_chunk 256 on gfx942 and
        //    gfx90a): _instances.py:1407-1419, SOURCE.md d512 rows (gfx90a:112, gfx942:108,
        //    gfx950:94).
        //  - f32: the generic prefill objects only, FlydslSdpaCdnaCommon.hpp:1636-1639 and
        //    _instances.py:1801-1833; dualwave and split decline it.
        //  - bias (vectorized bf16/f16 and the strided f32 one), sink (hd64/128), dropout
        //    (hd64/128/256 plus buckets), padded and packed: matcher :983-1049, :1103-1117,
        //    :937-946 (dropout and sink never together), :1613-1632; objects
        //    _instances.py:1286-1319, 1548-1660, 1662-1736; frontend lane cases that every FlyDSL
        //    family serves: IntegrationGpuFlydslSdpaFrontend.cpp:62-108 and :232-277,
        //    IntegrationGpuFlydslSdpaFrontendFeatures.cpp:232-316.
        //  - odd head dims run through the head_pad copies (FlydslSdpaHeadPad.hpp, wrapped
        //    around this engine at FlydslSdpaCdnaNative.cpp:124-148).
        constexpr const char* CDNA = "hipkernel:flydsl_sdpa_cdna";
        const std::vector<std::string> archs = {"gfx90a", "gfx942", "gfx950"};

        {
            // The prefill path with no split: bf16, head size 64, GQA, causal.
            FrontendSdpaSpec spec;
            spec.name = "flydsl_cdna_bf16_hd64";
            spec.dataType = BF16;
            spec.batch = 2;
            spec.queryHeads = 8;
            spec.kvHeads = 2;
            spec.seqQ = 128;
            spec.seqKv = 128;
            spec.headSize = 64;
            spec.rightBound = 0;
            spec.withStats = true;
            rows.push_back(forwardRow({CDNA, "Bf16Hd64", archs}, spec));
        }
        // kv_align 64 objects (Skv a multiple of 64, non-causal) and the masked tail objects.
        add(rows, CDNA, "F16Hd64NonCausal", archs, shaped(F16, {2, 8, 2, 256, 512, 64}, false));
        add(rows,
            CDNA,
            "Bf16Hd128NonCausalKvTail",
            archs,
            shaped(BF16, {2, 4, 4, 384, 500, 128}, false));
        add(rows, CDNA, "F16Hd128Causal", archs, shaped(F16, {1, 8, 2, 512, 512, 128}, true));
        add(rows, CDNA, "Bf16Hd256NonCausal", archs, shaped(BF16, {1, 4, 2, 512, 512, 256}, false));
        add(rows, CDNA, "F16Hd256Causal", archs, shaped(F16, {1, 4, 2, 512, 512, 256}, true));
        add(rows,
            CDNA,
            "Bf16Hd512DvSplit",
            archs,
            shaped(BF16, {1, 4, 4, 256, 256, 512}, false),
            {{"dv_split", 4}},
            KnobClass::STRUCTURE);
        add(rows,
            CDNA,
            "F16Hd512DvSplitCausal",
            archs,
            shaped(F16, {1, 4, 4, 256, 256, 512}, true),
            {{"dv_split", 4}},
            KnobClass::STRUCTURE);

        // f32 (dv_split 2 with k_chunk 64 at 256 on gfx942 and gfx90a, unsplit on gfx950).
        addF32(rows, CDNA, "F32Hd64", archs, shaped(F32, GQA_128, false));
        addF32(rows, CDNA, "F32Hd128Causal", archs, shaped(F32, {1, 8, 2, 256, 256, 128}, true));
        addF32(rows,
               CDNA,
               "F32Hd256",
               archs,
               shaped(F32, {1, 4, 2, 256, 256, 256}, false),
               {},
               KnobClass::STRUCTURE);
        // Wide bucket (qk_mask objects of the 512 bucket, V following Q/K, dv_split 4 with only
        // the slices that hold the graph's width launched: 264 takes three, 504 four).
        // _instances.py:1440-1456, FlydslSdpaCdnaCommon.hpp:1383-1387 (slices), SOURCE.md
        // sdpa_cdna d512 ..qkm1_dv4 rows (gfx90a:114, gfx942:110-111, gfx950:96).
        add(rows,
            CDNA,
            "Bf16Hd264WideBucket",
            archs,
            shaped(BF16, {1, 4, 4, 256, 256, 264}, false),
            {},
            KnobClass::STRUCTURE);
        add(rows,
            CDNA,
            "F16Hd504WideBucketCausal",
            archs,
            shaped(F16, {1, 4, 4, 256, 256, 504}, true),
            {},
            KnobClass::STRUCTURE);
        // f32 with dropout (keep-mask dump launch) and with the strided f32 bias: the f32 buckets'
        // qkm1 dropout1 and kstrbias1 objects (SOURCE.md sdpa_cdna gfx90a:302-321, gfx942:294-,
        // gfx950:266-277; _instances.py:1835-1880); f32 declines only sink, dualwave and split
        // (FlydslSdpaCdnaCommon.hpp:1051-1054, 1636-1639).
        addF32(rows, CDNA, "F32Hd64Dropout", archs, dropping(shaped(F32, GQA_128, false)));
        addF32(rows,
               CDNA,
               "F32Hd128DropoutCausal",
               archs,
               dropping(shaped(F32, {2, 8, 2, 128, 128, 128}, true)));
        addF32(rows, CDNA, "F32Hd64F32Bias", archs, biased(shaped(F32, GQA_128, false), F32));

        // Bias, sink.
        add(rows, CDNA, "Bf16Hd64Bias", archs, biased(shaped(BF16, GQA_128, false), BF16));
        add(rows,
            CDNA,
            "F16Hd128Bias",
            archs,
            biased(shaped(F16, {1, 8, 2, 256, 256, 128}, true), F16));
        add(rows, CDNA, "Bf16Hd64F32Bias", archs, biased(shaped(BF16, GQA_128, false), F32));
        add(rows, CDNA, "Bf16Hd64Sink", archs, sunk(shaped(BF16, GQA_128, false)));
        add(rows,
            CDNA,
            "F16Hd128SinkCausal",
            archs,
            sunk(shaped(F16, {2, 8, 2, 128, 128, 128}, true)));

        // Dropout: the Philox objects, the keep-mask writer, padded and packed.
        add(rows, CDNA, "Bf16Hd64Dropout", archs, dropping(shaped(BF16, GQA_128, false)));
        add(rows,
            CDNA,
            "F16Hd128DropoutCausal",
            archs,
            dropping(shaped(F16, {2, 8, 2, 128, 128, 128}, true)));
        add(rows,
            CDNA,
            "Bf16Hd256Dropout",
            archs,
            dropping(shaped(BF16, {1, 4, 2, 128, 128, 256}, false)));
        add(rows,
            CDNA,
            "Bf16Hd64DropoutPadded",
            archs,
            padded(dropping(shaped(BF16, GQA_128, false)), {128, 77}, {128, 90}));
        add(rows,
            CDNA,
            "F16Hd64DropoutPacked",
            archs,
            packedSlots(dropping(shaped(F16, GQA_128, true)), {128, 77}, {128, 90}));

        // Padded batches and packed THD.
        add(rows,
            CDNA,
            "Bf16Hd64Padded",
            archs,
            padded(shaped(BF16, GQA_256, true), {256, 200}, {256, 230}));
        add(rows,
            CDNA,
            "F16Hd128Padded",
            archs,
            padded(shaped(F16, {2, 8, 2, 256, 256, 128}, false), {256, 200}, {256, 230}));
        add(rows,
            CDNA,
            "Bf16Hd64Packed",
            archs,
            packedSlots(shaped(BF16, GQA_128, true), {128, 77}, {128, 90}));
        add(rows,
            CDNA,
            "F16Hd128Packed",
            archs,
            packedSlots(shaped(F16, {2, 8, 2, 128, 128, 128}, false), {128, 77}, {128, 90}));

        // Head dims off 8 (head_pad copies before and after the attention launch).
        add(rows,
            CDNA,
            "Bf16Hd53HeadPadBias",
            archs,
            biased(shaped(BF16, withHead(ODD_BASE, 53), false), BF16));
        add(rows,
            CDNA,
            "F16Hd113HeadPadDropout",
            archs,
            dropping(shaped(F16, withHead(ODD_BASE, 113), false)));
        add(rows,
            CDNA,
            "F16Hd241HeadPadPadded",
            archs,
            padded(shaped(F16, withHead(ODD_BASE, 241), true), {67, 19}, {143, 40}));
        add(rows,
            CDNA,
            "Bf16Hd23HeadPadPacked",
            archs,
            packedSlots(shaped(BF16, withHead(ODD_BASE, 23), true), {67, 19}, {143, 40}));
        addF32(rows, CDNA, "F32Hd37HeadPad", archs, shaped(F32, withHead(ODD_BASE, 37), true));
    }
#endif

#ifdef HKP_DETERMINISM_FLYDSL_SDPA_CDNA_DECODE
    {
        // The generic kernel in split-KV mode: a main launch writing f32 partials and LSEs to the
        // workspace, then the merge kernel (ascending split loops, one writer per O element;
        // trace cdna_decode: GEN:878-1057, U:5634-5684). The split count is
        // min(device bound, work bound, 64): candidateSplits, FlydslSdpaCdnaCommon.hpp:1391-1404.
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
        // (num_kv_splits 1; research INFERENCE, the objects were not scanned), row reductions by
        // fixed trees (trace cdna_dualwave: U:447-450, 504-507, 8971-9140). It declines padding,
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

#ifdef HKP_DETERMINISM_FLYDSL_SDPA_RDNA
    {
        // The RDNA prefill kernel: one launch, no workspace, one O writer per (row, column
        // slice) (trace rdna: RP:930-937, 1626-1660, 1961-2005). Classes:
        //  - hd_pair 0 (hd64, hd96, hd128), hd_pair 1 (hd256 exact, a head-dim pair split across
        //    two waves summed through LDS) and dv_split 2 with hd_pair (the generic 256 tier,
        //    head dims 232-248): sdpa_prefill_schedule and SDPA_GENERIC_HEAD_DIM_MAX,
        //    _instances.py:332-358; SOURCE.md sdpa_rdna gfx11-generic and gfx12-generic list
        //    d256 pair1 and dgeneric dv2_dmax256_pair1 objects in both bf16 and f16.
        //  - Graph and kernel matchers: FlydslSdpaRdnaCommon.hpp:470-791, 912-1047;
        //    FlydslSdpaRdnaNative.cpp:133-150.
        //  - f32: exact f32 tiers 64/128/256 on both targets, with an f32 bias and dropout
        //    (_instances.py:803-805, :962-990; gfx12 SOURCE.md:208-231); f32 never takes a sink.
        //  - 16-bit dropout: gfx11-generic only (_instances.py:767); on gfx12 aiw serves it.
        //  - head_pad: wrapped around this engine (FlydslSdpaRdnaNative.cpp).
        constexpr const char* RDNA = "hipkernel:flydsl_sdpa_rdna";
        const std::vector<std::string> archs = {"gfx1100", "gfx1201"};
        const std::vector<std::string> gfx11 = {"gfx1100"};
        const auto pair = KnobClass::STRUCTURE;

        add(rows, RDNA, "Bf16Hd64Causal", archs, shaped(BF16, GQA_256, true));
        add(rows, RDNA, "F16Hd64NonCausal", archs, shaped(F16, GQA_256, false));
        add(rows, RDNA, "Bf16Hd96NonCausal", archs, shaped(BF16, withHead(GQA_256, 96), false));
        add(rows, RDNA, "Bf16Hd128NonCausal", archs, shaped(BF16, withHead(GQA_256, 128), false));
        add(rows, RDNA, "F16Hd128Causal", archs, shaped(F16, withHead(GQA_256, 128), true));
        add(rows,
            RDNA,
            "Bf16Hd256HdPair",
            archs,
            shaped(BF16, {1, 4, 2, 256, 256, 256}, false),
            {{"hd_pair", 1}},
            pair);
        add(rows,
            RDNA,
            "F16Hd256HdPairCausal",
            archs,
            shaped(F16, {1, 4, 2, 256, 256, 256}, true),
            {{"hd_pair", 1}},
            pair);
        add(rows,
            RDNA,
            "Bf16Hd240DvSplit",
            archs,
            shaped(BF16, {1, 4, 2, 256, 256, 240}, false),
            {{"dv_split", 2}},
            pair);
        add(rows,
            RDNA,
            "F16Hd240DvSplitCausal",
            archs,
            shaped(F16, {1, 4, 2, 256, 256, 240}, true),
            {{"dv_split", 2}},
            pair);
        add(rows,
            RDNA,
            "Bf16Hd192V128Mla",
            archs,
            valueWidth(shaped(BF16, {1, 4, 2, 256, 256, 192}, false), 128),
            {},
            pair);

        add(rows, RDNA, "Bf16Hd64Bias", archs, biased(shaped(BF16, GQA_128, false), BF16));
        add(rows,
            RDNA,
            "F16Hd128F32Bias",
            archs,
            biased(shaped(F16, withHead(GQA_128, 128), false), F32));
        add(rows, RDNA, "Bf16Hd64SinkCausal", archs, sunk(shaped(BF16, GQA_128, true)));
        add(rows, RDNA, "F16Hd128Sink", archs, sunk(shaped(F16, withHead(GQA_128, 128), false)));
        add(rows,
            RDNA,
            "Bf16Hd64Padded",
            archs,
            padded(shaped(BF16, GQA_256, true), {256, 200}, {256, 230}));
        add(rows,
            RDNA,
            "F16Hd128PaddedCausal",
            archs,
            padded(shaped(F16, withHead(GQA_256, 128), true), {256, 200}, {256, 230}));
        add(rows,
            RDNA,
            "Bf16Hd64Packed",
            archs,
            packedSlots(shaped(BF16, GQA_128, true), {128, 77}, {128, 90}));
        add(rows,
            RDNA,
            "F16Hd128Packed",
            archs,
            packedSlots(shaped(F16, withHead(GQA_128, 128), false), {128, 77}, {128, 90}));

        addF32(rows, RDNA, "F32Hd64", archs, shaped(F32, GQA_128, false));
        addF32(rows, RDNA, "F32Hd128Causal", archs, shaped(F32, withHead(GQA_128, 128), true));
        addF32(rows,
               RDNA,
               "F32Hd256DvSplit",
               archs,
               shaped(F32, {1, 4, 2, 128, 128, 256}, false),
               {},
               pair);
        addF32(rows, RDNA, "F32Hd64Dropout", archs, dropping(shaped(F32, GQA_128, false)));
        addF32(rows,
               RDNA,
               "F32Hd128F32Bias",
               archs,
               biased(shaped(F32, withHead(GQA_128, 128), false), F32));

        add(rows,
            RDNA,
            "Bf16Hd53HeadPadBias",
            archs,
            biased(shaped(BF16, withHead(ODD_BASE, 53), false), BF16));
        add(rows,
            RDNA,
            "F16Hd241HeadPadPadded",
            archs,
            padded(shaped(F16, withHead(ODD_BASE, 241), true), {67, 19}, {143, 40}));
        add(rows,
            RDNA,
            "Bf16Hd23HeadPadPacked",
            archs,
            packedSlots(shaped(BF16, withHead(ODD_BASE, 23), true), {67, 19}, {143, 40}));
        addF32(rows, RDNA, "F32Hd37HeadPad", archs, shaped(F32, withHead(ODD_BASE, 37), true));

        // 16-bit dropout, gfx11 only.
        add(rows, RDNA, "Bf16Hd64Dropout", gfx11, dropping(shaped(BF16, GQA_128, false)));
        add(rows,
            RDNA,
            "F16Hd128DropoutCausal",
            gfx11,
            dropping(shaped(F16, withHead(GQA_128, 128), true)));
        add(rows,
            RDNA,
            "Bf16Hd256Dropout",
            gfx11,
            dropping(shaped(BF16, {1, 4, 2, 128, 128, 256}, false)),
            {},
            pair);
        add(rows,
            RDNA,
            "Bf16Hd64DropoutPadded",
            gfx11,
            padded(dropping(shaped(BF16, GQA_128, false)), {128, 77}, {128, 90}));
        add(rows,
            RDNA,
            "F16Hd64DropoutPacked",
            gfx11,
            packedSlots(dropping(shaped(F16, GQA_128, true)), {128, 77}, {128, 90}));
        // Generic dropout tiers (gfx11 SOURCE.md sdpa_rdna dmax256_pair1_dropout1 and
        // dmax160_pair1_dropout1; _instances.py:779-782, :962-990): hd240 on the 256 tier with the
        // column split, and hd113 padded to 120 by head_pad onto the 160 tier.
        add(rows,
            RDNA,
            "F16Hd240DropoutDvSplit",
            gfx11,
            dropping(shaped(F16, {1, 4, 2, 128, 128, 240}, false)),
            {{"dv_split", 2}},
            pair);
        add(rows,
            RDNA,
            "F16Hd113HeadPadDropout",
            gfx11,
            dropping(shaped(F16, withHead(ODD_BASE, 113), false)));
    }
#endif

#ifdef HKP_DETERMINISM_FLYDSL_SDPA_RDNA_DECODE
    {
        // The RDNA decode kernel: split-KV main launch with an in-workgroup wave combine
        // (w = 0..3, compile order) and the merge kernel (ascending splits, one workgroup per
        // (b, h, s)); trace rdna_decode: DEC:383-388, 442-446, 681-760, 886-985. Splits =
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

#ifdef HKP_DETERMINISM_FLYDSL_SDPA_AIW
    {
        // AOTriton's gfx1201 kernel, gfx12-generic only (SDPA_AIW_TARGETS, _instances.py:2399):
        // one launch (plus the padded-row fill launch for a padded graph), explicit per-wave LDS
        // partials for the cross-shard reduction at head dims above 256 (QK_SHARDS 2 at 384, 4
        // at 512; trace aiw: FC:1189-1245, AIW:1795-1810, AIW:2406). The head dims, dropout,
        // sink, bias and MLA objects are the sdpa_aiw/gfx12-generic SOURCE.md rows (a bf16 and
        // f16 object per class; no f32). Kernel matcher FlydslSdpaAiwNative.cpp:262-420: padded
        // or packed graphs need a bottom-right (or no) mask (:336), padded with packed lengths
        // declines (:344), a padded graph needs the fill symbol (:414). head_pad is wrapped
        // around the engine (:886-910).
        constexpr const char* AIW = "hipkernel:flydsl_sdpa_aiw";
        const std::vector<std::string> archs = {"gfx1201"};
        const auto shards = KnobClass::STRUCTURE;

        add(rows, AIW, "Bf16Hd64Causal", archs, shaped(BF16, GQA_256, true));
        add(rows, AIW, "F16Hd64NonCausal", archs, shaped(F16, GQA_256, false));
        add(rows, AIW, "Bf16Hd128NonCausal", archs, shaped(BF16, withHead(GQA_256, 128), false));
        add(rows, AIW, "F16Hd128Causal", archs, shaped(F16, withHead(GQA_256, 128), true));
        add(rows, AIW, "Bf16Hd256NonCausal", archs, shaped(BF16, {1, 4, 2, 256, 256, 256}, false));
        add(rows, AIW, "F16Hd256Causal", archs, shaped(F16, {1, 4, 2, 256, 256, 256}, true));
        add(rows,
            AIW,
            "Bf16Hd384Shards2",
            archs,
            shaped(BF16, {1, 4, 4, 256, 256, 384}, false),
            {},
            shards);
        add(rows,
            AIW,
            "F16Hd384Shards2Causal",
            archs,
            shaped(F16, {1, 4, 4, 256, 256, 384}, true),
            {},
            shards);
        add(rows,
            AIW,
            "Bf16Hd512Shards4",
            archs,
            shaped(BF16, {1, 4, 4, 256, 256, 512}, false),
            {},
            shards);
        add(rows,
            AIW,
            "F16Hd512Shards4Causal",
            archs,
            shaped(F16, {1, 4, 4, 256, 256, 512}, true),
            {},
            shards);

        add(rows, AIW, "Bf16Hd64Dropout", archs, dropping(shaped(BF16, GQA_128, false)));
        add(rows,
            AIW,
            "F16Hd128DropoutCausal",
            archs,
            dropping(shaped(F16, withHead(GQA_128, 128), true)));
        add(rows,
            AIW,
            "Bf16Hd256Dropout",
            archs,
            dropping(shaped(BF16, {1, 4, 2, 128, 128, 256}, false)));
        add(rows,
            AIW,
            "F16Hd64DropoutPadded",
            archs,
            padded(dropping(shaped(F16, GQA_128, false)), {128, 77}, {128, 90}));
        add(rows,
            AIW,
            "Bf16Hd64DropoutPacked",
            archs,
            packedSlots(dropping(shaped(BF16, GQA_128, true)), {128, 77}, {128, 90}));

        add(rows, AIW, "Bf16Hd64Sink", archs, sunk(shaped(BF16, GQA_128, false)));
        add(rows,
            AIW,
            "F16Hd128SinkCausal",
            archs,
            sunk(shaped(F16, withHead(GQA_128, 128), true)));
        add(rows, AIW, "Bf16Hd64Bias", archs, biased(shaped(BF16, GQA_128, false), BF16));
        // An f32 bias without dropout is not a row: the non-strided f32 bias objects need the bias
        // to declare 32-byte alignment (FlydslSdpaAiwNative.cpp:392-399), the frontend builder
        // declares the default 16 (TensorConstants.hpp:13) and sets no other, and the kstrided
        // f32 objects, which need no alignment, ship only with dropout (SOURCE.md sdpa_aiw
        // *_f32bias1_kstrbias1_dropout1). So the f32 bias class is covered through those.
        add(rows,
            AIW,
            "F16Hd128F32BiasKStridedDropoutCausal",
            archs,
            dropping(biased(shaped(F16, withHead(GQA_128, 128), true), F32)));
        add(rows,
            AIW,
            "Bf16Hd64Padded",
            archs,
            padded(shaped(BF16, GQA_256, true), {256, 200}, {256, 230}));
        add(rows,
            AIW,
            "F16Hd128PaddedCausal",
            archs,
            padded(shaped(F16, withHead(GQA_256, 128), true), {256, 200}, {256, 230}));
        add(rows,
            AIW,
            "Bf16Hd64Packed",
            archs,
            packedSlots(shaped(BF16, GQA_128, true), {128, 77}, {128, 90}));
        add(rows,
            AIW,
            "F16Hd128Packed",
            archs,
            packedSlots(shaped(F16, withHead(GQA_128, 128), false), {128, 77}, {128, 90}));
        add(rows,
            AIW,
            "Bf16Hd192V128Mla",
            archs,
            valueWidth(shaped(BF16, {1, 4, 2, 256, 256, 192}, false), 128));

        add(rows,
            AIW,
            "Bf16Hd53HeadPadBias",
            archs,
            // Skv 144: the non-strided bias objects need every bias row stride a multiple of 8
            // (FlydslSdpaAiwNative.cpp:403-409); Skv 143 gives a query stride of 143.
            biased(shaped(BF16, {2, 8, 2, 67, 144, 53}, false), BF16));
        add(rows,
            AIW,
            "F16Hd113HeadPadDropout",
            archs,
            dropping(shaped(F16, withHead(ODD_BASE, 113), false)));
        add(rows,
            AIW,
            "F16Hd241HeadPadPadded",
            archs,
            padded(shaped(F16, withHead(ODD_BASE, 241), true), {67, 19}, {143, 40}));
        add(rows,
            AIW,
            "Bf16Hd23HeadPadPacked",
            archs,
            packedSlots(shaped(BF16, withHead(ODD_BASE, 23), true), {67, 19}, {143, 40}));
    }
#endif

    return rows;
}

[[maybe_unused]] const bool REGISTERED = registerRows(flydslForwardRows());

} // namespace

#endif

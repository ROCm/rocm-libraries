// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA) \
    && (defined(HKP_DETERMINISM_ASM_SDPA) || defined(HKP_DETERMINISM_HIP_FLASH2))

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "SdpaDeterminismHarness.hpp"

/**
 * @file IntegrationGpuSdpaDeterminismAsmFlash2.cpp
 * @brief The determinism rows of ASM_SDPA_ENGINE (forward on gfx942 and gfx950, backward on
 *        gfx942 and gfx950) and HIP_FLASH2_ENGINE (gfx942). CMake defines HKP_DETERMINISM_ASM_SDPA
 *        and HKP_DETERMINISM_HIP_FLASH2 for the engines this build has, and each table below
 *        exists only under its definition.
 */

namespace
{

using namespace hip_kernel_provider::sdpa_determinism;
using hip_kernel_provider::kernel_ingestor_engine::integration::FrontendSdpaSpec;
using hip_kernel_provider::kernel_ingestor_engine::testing::SdpaLayout;
using hipdnn_frontend::DataType;
using hipdnn_frontend::DiagonalAlignment;

#ifdef HKP_DETERMINISM_ASM_SDPA

constexpr const char* ASM_ENGINE = "ASM_SDPA_ENGINE";

// ASM forward. Each CSV row (asm_kernels/<arch>/fmha_v3_fwd/fmha_fwd.csv) is one prebuilt
// object, launched once with no workspace (SdpaFwdPlan.cpp:21-25): grid = Q tiles x heads x
// batch with no KV axis (SdpaFwdLaunchParams.hpp:56-105). A row's rows below therefore reach:
//  - every CSV row of the arch, batch and group (ragged THD) mode, fp8bf16 on gfx942;
//  - the Q-tile count: ts_qo is 256 (hd128) or 128 (hd192x128), so every row has Sq of at least
//    three tiles plus a partial tail, and Skv a tail of the K tile (ts_kv 32 gfx942 bf16, 64
//    gfx942 fp8 and gfx950 hd128, 128 gfx950 hd192);
//  - tgDiv=2 (a bottom-right causal mask merges head and tail Q tiles onto shared workgroups,
//    SdpaFwdLaunchParams.hpp:72), which needs at least four Q tiles: the causal rows have four;
//    gfx942 hd192x128 batch keeps its own grid layout (:86-95) and gfx942 hd192x128 group keeps
//    tgDiv=1 (:73-76), each with its own row;
//  - tune_opt 3 (a causal row with Hq not a multiple of 8 or Sq above 16384,
//    SdpaFwdLaunchParams.hpp:73-76; tune_opt 0 is the gfx942 hd192x128 rows, :77-80) and GQA
//    (Hq a multiple of Hkv, hasConsistentFwdShapes, SdpaPlanUtils.hpp:224-250);
//  - LSE on and off: admitted in batch mode with packed strides [H * s_h, s_h, 1] and a 32-bit
//    LSE offset, or in group mode on the Q/O ragged offset tensor with a unit sequence stride
//    (isSupportedStatsTensor, SdpaFwdPlanBuilder.cpp:517-593); not gated by causal.
// Causal: the CSV mask 2 serves bottom-right, and top-left too when every batch's Q and K/V
// lengths are equal (fwdKernelMaskType, SdpaFwdPlanBuilder.cpp:97-109); this suite's causal rows
// stay bottom-right.
// The gfx942 batch rows are written for both SKUs: the engine loads MI308/<co> on an MI308X
// device and MI300/<co> elsewhere (AsmKernelPath.hpp:56-65, isMi308Device
// SdpaFwdPlanBuilder.cpp:215-240); a MI308 row's name ends in mi308x. gfx950 ships one set.
// gfx950 has no batch causal row (gfx950 fmha_fwd.csv:2-7), so its causal rows are group rows.
// Group rows never reach a fully masked query row (every sequence has no more queries than
// keys), so no output element is undefined by the mask. Group tensors are packed THD with Q/O
// and K/V token offset tables whose multiplier is the sequence stride, the AITER contract
// (isApplicable hasAiterCompatibleRaggedAddressing, SdpaFwdPlanBuilder.cpp:189-229); the
// precedent is asm_sdpa_engine/IntegrationGpuSdpaForwardRagged.cpp.
struct AsmFwdCase
{
    const char* caseName;
    int64_t headSize;
    int64_t valueHeadSize;
    bool causal;
    bool group;
    bool fp8;
    bool lse;
    int64_t batch;
    int64_t queryHeads;
    int64_t kvHeads;
    int64_t seqQ;
    int64_t seqKv;
    // Group rows: each batch's live queries and keys (at most seqQ and seqKv).
    std::vector<int32_t> lensQ;
    std::vector<int32_t> lensKv;
};

FrontendSdpaSpec asmFwdSpec(const AsmFwdCase& c)
{
    FrontendSdpaSpec spec;
    spec.name = std::string("asm_fwd_") + c.caseName;
    spec.dataType = DataType::BFLOAT16;
    spec.batch = c.group ? static_cast<int64_t>(c.lensQ.size()) : c.batch;
    spec.queryHeads = c.queryHeads;
    spec.kvHeads = c.kvHeads;
    spec.seqQ = c.seqQ;
    spec.seqKv = c.seqKv;
    spec.headSize = c.headSize;
    if(c.valueHeadSize != c.headSize)
    {
        spec.valueHeadSize = c.valueHeadSize;
    }
    spec.leftBound.reset();
    if(c.causal)
    {
        spec.rightBound = 0;
        spec.alignment = DiagonalAlignment::BOTTOM_RIGHT;
    }
    else
    {
        spec.rightBound.reset();
    }
    spec.withStats = c.lse;
    spec.packedLensQ = c.lensQ;
    spec.packedLensKv = c.lensKv;
    return spec;
}

void addAsmFwdRows(std::vector<DeterminismRow>& rows,
                   const std::vector<AsmFwdCase>& cases,
                   const char* arch,
                   Sku sku)
{
    for(const auto& c : cases)
    {
        const RowMeta meta{ASM_ENGINE, c.caseName, {arch}, sku};
        const auto shape = asmFwdSpec(c);
        if(c.fp8)
        {
            // FP8 E4M3 FNUZ Q/K/V, BF16 O, three scalar FLOAT descales, mma_core_mode unset
            // (SdpaFwdPlanBuilder.cpp:106-117, 650-671).
            DeterminismFwdSpec fp8;
            fp8.shape = shape;
            fp8.dtypes = {DataType::FP8_E4M3_FNUZ, DataType::BFLOAT16, DataType::FLOAT};
            rows.push_back(forwardRow(meta, fp8));
        }
        else
        {
            rows.push_back(forwardRow(meta, shape));
        }
    }
}

// Shapes. hd128 (ts_qo 256): Sq 1000 = three tiles + 232, four tiles. hd192x128 (ts_qo 128):
// Sq 500 = three tiles + 116, four tiles. Skv 1100 / 1200 / 600: tails of every K tile size.
// A causal Skv above Sq keeps every query row's visible keys non-empty.
const std::vector<int32_t> GROUP_Q_HD128 = {333, 1000, 129};
const std::vector<int32_t> GROUP_KV_HD128 = {400, 1100, 600};
const std::vector<int32_t> GROUP_Q_HD192 = {250, 500, 77};
const std::vector<int32_t> GROUP_KV_HD192 = {300, 600, 500};

std::vector<AsmFwdCase> asmGfx942Cases()
{
    // {name, hd, hdv, causal, group, fp8, lse, B, Hq, Hkv, Sq, Skv, lensQ, lensKv}
    return {
        // gfx942 fmha_fwd.csv:2 bf16 hd128 batch, no mask; LSE on and off; GQA ratio 4.
        {"Bf16Hd128Lse", 128, 128, false, false, false, true, 2, 8, 8, 1000, 1100, {}, {}},
        {"Bf16Hd128NoLse", 128, 128, false, false, false, false, 2, 8, 8, 1000, 1100, {}, {}},
        {"Bf16Hd128GqaLse", 128, 128, false, false, false, true, 2, 8, 2, 1000, 1100, {}, {}},
        // :3 bf16 hd128 batch bottom-right causal (tgDiv 2); Hq 6 is tune_opt 3 with GQA 3.
        {"Bf16Hd128Causal", 128, 128, true, false, false, false, 2, 8, 8, 1000, 1200, {}, {}},
        {"Bf16Hd128CausalGqaTune3",
         128,
         128,
         true,
         false,
         false,
         false,
         2,
         6,
         2,
         1000,
         1200,
         {},
         {}},
        // :4 bf16 hd192x128 batch (grid x = heads, y = Q tiles; tune_opt 0).
        {"Bf16Hd192x128Lse", 192, 128, false, false, false, true, 2, 8, 8, 500, 600, {}, {}},
        {"Bf16Hd192x128NoLse", 192, 128, false, false, false, false, 2, 8, 8, 500, 600, {}, {}},
        // :5 bf16 hd192x128 batch bottom-right causal, no tgDiv at this row; GQA ratio 4.
        {"Bf16Hd192x128Causal", 192, 128, true, false, false, false, 2, 8, 8, 500, 700, {}, {}},
        {"Bf16Hd192x128CausalGqa", 192, 128, true, false, false, false, 2, 8, 2, 500, 700, {}, {}},
        // :6-7 fp8bf16 hd128 batch (K tile 64), no mask with LSE on and off, and causal.
        {"Fp8Hd128Lse", 128, 128, false, false, true, true, 2, 8, 8, 1000, 1100, {}, {}},
        {"Fp8Hd128NoLse", 128, 128, false, false, true, false, 2, 8, 8, 1000, 1100, {}, {}},
        {"Fp8Hd128Causal", 128, 128, true, false, true, false, 2, 8, 8, 1000, 1200, {}, {}},
        // :8-9 bf16 hd128 group, no mask and causal (ragged, no LSE).
        {"Bf16Hd128Group",
         128,
         128,
         false,
         true,
         false,
         false,
         0,
         8,
         8,
         1000,
         1100,
         GROUP_Q_HD128,
         GROUP_KV_HD128},
        {"Bf16Hd128GroupCausal",
         128,
         128,
         true,
         true,
         false,
         false,
         0,
         8,
         8,
         1000,
         1100,
         GROUP_Q_HD128,
         GROUP_KV_HD128},
        // :10-11 bf16 hd192x128 group, no mask and causal (tgDiv 1 at this arch's group row).
        {"Bf16Hd192x128Group",
         192,
         128,
         false,
         true,
         false,
         false,
         0,
         8,
         8,
         500,
         600,
         GROUP_Q_HD192,
         GROUP_KV_HD192},
        {"Bf16Hd192x128GroupCausal",
         192,
         128,
         true,
         true,
         false,
         false,
         0,
         8,
         8,
         500,
         600,
         GROUP_Q_HD192,
         GROUP_KV_HD192},
    };
}

std::vector<AsmFwdCase> asmGfx950Cases()
{
    return {
        // gfx950 fmha_fwd.csv:2 bf16 hd128 batch no mask (K tile 64), LSE on and off, GQA 4.
        {"Bf16Hd128Lse", 128, 128, false, false, false, true, 2, 8, 8, 1000, 1100, {}, {}},
        {"Bf16Hd128NoLse", 128, 128, false, false, false, false, 2, 8, 8, 1000, 1100, {}, {}},
        {"Bf16Hd128GqaLse", 128, 128, false, false, false, true, 2, 8, 2, 1000, 1100, {}, {}},
        // :5 bf16 hd192x128 batch no mask (K tile 128), LSE on and off, GQA 4.
        {"Bf16Hd192x128Lse", 192, 128, false, false, false, true, 2, 8, 8, 500, 600, {}, {}},
        {"Bf16Hd192x128NoLse", 192, 128, false, false, false, false, 2, 8, 8, 500, 600, {}, {}},
        {"Bf16Hd192x128GqaLse", 192, 128, false, false, false, true, 2, 8, 2, 500, 600, {}, {}},
        // :3-4 bf16 hd128 group, no mask and causal (tgDiv 2).
        {"Bf16Hd128Group",
         128,
         128,
         false,
         true,
         false,
         false,
         0,
         8,
         8,
         1000,
         1100,
         GROUP_Q_HD128,
         GROUP_KV_HD128},
        {"Bf16Hd128GroupCausal",
         128,
         128,
         true,
         true,
         false,
         false,
         0,
         8,
         8,
         1000,
         1100,
         GROUP_Q_HD128,
         GROUP_KV_HD128},
        // :6-7 bf16 hd192x128 group, no mask and causal (tgDiv 2 on gfx950).
        {"Bf16Hd192x128Group",
         192,
         128,
         false,
         true,
         false,
         false,
         0,
         8,
         8,
         500,
         600,
         GROUP_Q_HD192,
         GROUP_KV_HD192},
        {"Bf16Hd192x128GroupCausal",
         192,
         128,
         true,
         true,
         false,
         false,
         0,
         8,
         8,
         500,
         600,
         GROUP_Q_HD192,
         GROUP_KV_HD192},
    };
}

// ASM backward A32 (the default accumulator; a16 is reachable through the published
// hipkernel.sdpa.bwd.accumulator_type knob and unmeasured here): odo, dqdkdv with fp32 atomic
// dQ into a zeroed workspace, dq_convert (SdpaBwdPlan.cpp:177-215). Workgroups of different K
// tiles add into one dQ row, so the order of the adds is scheduling dependent: a known hazard,
// measured and never proven. Rows have at least two K tiles per (b, h): the dqdkdv K tile ts is
// 192 (hd128) or 64 (hd192) (fmha_bwd_dqdkdv.csv), and Skv 512 gives three and eight.
// ASM_SDPA_ENGINE serves backward on gfx942 and gfx950 (isSupportedArch,
// SdpaBwdPlanBuilder.cpp:636-639); these rows are gfx942 by choice. MHA only (GQA declined,
// :1183-1189); BHSD packed strides; findDqdkdvConfig (:192-230) picks the smallest qualifying
// (pssk, pddv) row for the shape (requestedPssk, :165-176): hd128 and hd192 take pssk = 1,
// pddv = 1; hd64 has its own pssk = 1, pddv = 0 row (fmha_bwd_dqdkdv.csv), so no row is
// missing. hd128 and hd192, bf16 and fp16, each have a no-mask (mask 0) and a bottom-right
// causal (mask 2) row; top-left causal (mask 1) has rows too but is not written (the same
// launch structure). Sq below Skv, as in
// asm_sdpa_engine/IntegrationGpuSdpaBackward.cpp:98-101. O and the LSE come from one forward
// launch of the first-ranked engine on the device (harness): a causal or fp16 or hd192 row
// needs a non-ASM gfx942 forward that serves LSE for the row's shape (ASM forward is bf16
// only, so every fp16 row needs one), for example flydsl_sdpa_cdna, so the row fails if none
// serves it.
void addAsmBwdRows(std::vector<DeterminismRow>& rows)
{
    struct BwdCase
    {
        const char* caseName;
        DataType dataType;
        int64_t headSize;
        bool causal;
    };
    const std::array<BwdCase, 8> cases = {{
        {"A32Bf16Hd128", DataType::BFLOAT16, 128, false},
        {"A32Bf16Hd128Causal", DataType::BFLOAT16, 128, true},
        {"A32Fp16Hd128", DataType::HALF, 128, false},
        {"A32Fp16Hd128Causal", DataType::HALF, 128, true},
        {"A32Bf16Hd192", DataType::BFLOAT16, 192, false},
        {"A32Bf16Hd192Causal", DataType::BFLOAT16, 192, true},
        {"A32Fp16Hd192", DataType::HALF, 192, false},
        {"A32Fp16Hd192Causal", DataType::HALF, 192, true},
    }};
    for(const auto& c : cases)
    {
        DeterminismBwdSpec spec;
        spec.name = std::string("asm_bwd_") + c.caseName;
        spec.dataType = c.dataType;
        spec.batch = 2;
        spec.queryHeads = 2;
        spec.kvHeads = 2;
        spec.seqQ = 256;
        spec.seqKv = 512;
        spec.headSize = c.headSize;
        if(c.causal)
        {
            spec.rightBound = 0;
            spec.alignment = DiagonalAlignment::BOTTOM_RIGHT;
        }
        spec.layout = SdpaLayout::BHSD;
        rows.push_back(backwardRow(
            {ASM_ENGINE, c.caseName, {"gfx942"}, Sku::ANY, Disposition::MEASURE_ONLY}, spec));
    }
}

#endif // HKP_DETERMINISM_ASM_SDPA

#ifdef HKP_DETERMINISM_HIP_FLASH2

constexpr const char* FLASH2_ENGINE = "HIP_FLASH2_ENGINE";

// HIP_FLASH2 forward, gfx942, fp16, d64 and d128, no LSE (declined, HipFlash2FwdPlanBuilder.cpp:
// 93-94), no mask or top-left causal (the bottom-right mask is declined, :165-167; the row
// writes TOP_LEFT), Sq a multiple of 64 (:202-204), Hq a multiple of Hkv (GQA), no group mode,
// head-dim stride 1 (:212-221). Split-K is forced to 1 (:294-295, Flash2Dispatch.hpp:60-68), so
// there is no combine and no workspace; no kernel atomics.
//
// selectFlash2Config (Flash2Dispatch.hpp:88-157) picks the tiling from the shape and the device
// CU count (304 for an MI300X; rows below are chosen for 304, so they are Sku::MI300 rows and
// the row's tiling name is that selection). The tilings w4q1k4, w8q1k4, w8q2k4, w8q3k2 and
// w8q3k4 are separate objects built only with -DHIP_FLASH2_BUILD_VARIANTS=ON (CMakeLists.txt:
// 121-133); a default build tracks only the legacy 64-thread object and
// HipFlash2FwdPlanBuilder.cpp:297-320 falls back to it for every shape. A row named for a
// tiling reaches that tiling only in a build with HIP_FLASH2_BUILD_VARIANTS; in a default build
// it runs the legacy object. The two Legacy rows on MI308X (80 CUs, which selects differently)
// run the legacy object of a default build there.
//
// Shapes: Skv a multiple of the K tile (32 or 64) and not; Sq a multiple of the tiling's
// queries per CTA and not (partial CTA tile); a shape with a split-K the selector computes but
// the plan never runs (SplitK3: ctas256 112, SplitK4: ctas256 64, d128 Sq >= 1024, Flash2Dispatch.hpp:105-115); GQA.
// The scalar K/V load path (k_stride_seq % 8 != 0, HipFlash2FwdPlanVariant.hip:195) is not
// reachable: with d64 or d128 a packed BSHD K and V have a sequence stride of Hkv * d, always a
// multiple of 8, and the frontend builder makes no other layout. It has no row.
struct Flash2Case
{
    const char* caseName;
    int64_t batch;
    int64_t queryHeads;
    int64_t kvHeads;
    int64_t seqQ;
    int64_t seqKv;
    int64_t headSize;
    bool causal;
    Sku sku;
};

void addFlash2Rows(std::vector<DeterminismRow>& rows)
{
    // ctas256 = ceil(Sq / 256) * B * Hq; the tiling is the selector's at 304 CUs.
    const std::array<Flash2Case, 16> cases = {{
        // ctas256 16 (< 100): w4q1k4.
        {"W4q1k4Hd128", 1, 8, 8, 512, 512, 128, false, Sku::MI300},
        {"W4q1k4Hd64", 2, 8, 8, 512, 512, 64, false, Sku::MI300},
        // splitK 4 computed, single pass run; w4q1k4.
        {"W4q1k4Hd128SplitK4", 1, 8, 8, 2048, 2048, 128, false, Sku::MI300},
        // splitK 3 computed, single pass run; w8q2k4.
        {"W8q2k4Hd128SplitK3", 1, 28, 28, 1024, 1024, 128, false, Sku::MI300},
        // causal, Sq <= 1024, ctas256 128 (< 200): w8q1k4; Skv not a multiple of BK.
        {"W8q1k4Hd128Causal", 1, 32, 32, 1024, 1000, 128, true, Sku::MI300},
        // general no-mask, Sq 1088 = 4 tiles of 256 + 64 (partial CTA tile): w8q2k4.
        {"W8q2k4Hd128", 2, 16, 16, 1088, 1088, 128, false, Sku::MI300},
        {"W8q2k4Hd128Gqa", 2, 32, 8, 1088, 1000, 128, false, Sku::MI300},
        // causal, Sq > 1024, ctas384 192 (< 256): w8q2k4.
        {"W8q2k4Hd128Causal", 2, 16, 16, 2048, 2048, 128, true, Sku::MI300},
        // d64 below Sq 4096: w8q2k4.
        {"W8q2k4Hd64", 2, 16, 16, 1024, 1024, 64, false, Sku::MI300},
        {"W8q2k4Hd64Causal", 2, 16, 16, 1024, 1024, 64, true, Sku::MI300},
        // causal, Sq >= 1024, ctas384 384 (>= 256): w8q3k2 (K tile 32); Skv not a multiple.
        {"W8q3k2Hd128Causal", 2, 32, 32, 2048, 2000, 128, true, Sku::MI300},
        // no-mask, Hq <= 16 and Sq >= 3072: w8q3k2.
        {"W8q3k2Hd128", 1, 16, 16, 3072, 3072, 128, false, Sku::MI300},
        // d64, B * Hq >= 64 and Sq >= 4096: w8q3k4 (K tile 64).
        {"W8q3k4Hd64", 2, 32, 32, 4096, 4096, 64, false, Sku::MI300},
        {"W8q3k4Hd64Causal", 2, 32, 32, 4096, 4096, 64, true, Sku::MI300},
        // The legacy object of a default build on an MI308X device.
        {"LegacyHd128", 2, 16, 16, 1088, 1000, 128, false, Sku::MI308},
        {"LegacyHd64Causal", 2, 16, 16, 1024, 1024, 64, true, Sku::MI308},
    }};
    for(const auto& c : cases)
    {
        FrontendSdpaSpec spec;
        spec.name = std::string("flash2_fwd_") + c.caseName;
        spec.dataType = DataType::HALF;
        spec.batch = c.batch;
        spec.queryHeads = c.queryHeads;
        spec.kvHeads = c.kvHeads;
        spec.seqQ = c.seqQ;
        spec.seqKv = c.seqKv;
        spec.headSize = c.headSize;
        spec.leftBound.reset();
        if(c.causal)
        {
            spec.rightBound = 0;
            spec.alignment = DiagonalAlignment::TOP_LEFT;
        }
        else
        {
            spec.rightBound.reset();
        }
        spec.withStats = false;
        rows.push_back(forwardRow({FLASH2_ENGINE, c.caseName, {"gfx942"}, c.sku}, spec));
    }
}

#endif // HKP_DETERMINISM_HIP_FLASH2

std::vector<DeterminismRow> asmFlash2Rows()
{
    std::vector<DeterminismRow> rows;
#ifdef HKP_DETERMINISM_ASM_SDPA
    addAsmFwdRows(rows, asmGfx942Cases(), "gfx942", Sku::MI300);
    addAsmFwdRows(rows, asmGfx942Cases(), "gfx942", Sku::MI308);
    addAsmFwdRows(rows, asmGfx950Cases(), "gfx950", Sku::ANY);
    addAsmBwdRows(rows);
#endif
#ifdef HKP_DETERMINISM_HIP_FLASH2
    addFlash2Rows(rows);
#endif
    return rows;
}

[[maybe_unused]] const bool REGISTERED = registerRows(asmFlash2Rows());

} // namespace

#endif

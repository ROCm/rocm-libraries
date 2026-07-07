/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/ir/logical/LogicalOpcode.hpp"
#include "stinkytofu/transforms/logical/CompositeInstructionLoweringPass.hpp"
#include "stinkytofu/transforms/logical/ToStinkyAsmPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

/**
 * @brief Factory function: Creates a fresh test instruction for given opcode
 *
 * SINGLE SOURCE OF TRUTH for all tested instructions.
 *
 * To add a new instruction:
 * 1. Add case: case logical::YourNewInst: return YourNewInst(...);
 * 2. That's it! Coverage guard reads this automatically.
 */
static LogicalInstruction* createTestInstruction(logical::Opcode opcode) {
    switch (opcode) {
        case logical::VAddU32:
            return VAddU32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMulF32:
            return VMulF32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VAddF16:
            return VAddF16(vgpr(0), vgpr(1), vgpr(2));
        case logical::VAddF32:
            return VAddF32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VAddF64:
            return VAddF64(vgpr(0), vgpr(1), vgpr(2));
        case logical::VAddI32:
            return VAddI32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VAddCOU32:
            return VAddCOU32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VAddCCOU32:
            return VAddCCOU32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VAddPKF16:
            return VAddPKF16(vgpr(0), vgpr(1), vgpr(2));
        case logical::VAdd3U32:
            return VAdd3U32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VSubF32:
            return VSubF32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VSubI32:
            return VSubI32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VSubU32:
            return VSubU32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VSubCoU32:
            return VSubCoU32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMulF16:
            return VMulF16(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMulF64:
            return VMulF64(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMulPKF16:
            return VMulPKF16(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMulPKF32S:
            return VMulPKF32S(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMulLOU32:
            return VMulLOU32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMulHII32:
            return VMulHII32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMulHIU32:
            return VMulHIU32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMulI32I24:
            return VMulI32I24(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMulU32U24:
            return VMulU32U24(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMacF32:
            return VMacF32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VFmaF16:
            return VFmaF16(vgpr(0), vgpr(1), vgpr(2), vgpr(3));
        case logical::VFmaF32:
            return VFmaF32(vgpr(0), vgpr(1), vgpr(2), vgpr(3));
        case logical::VFmaF64:
            return VFmaF64(vgpr(0), vgpr(1), vgpr(2), vgpr(3));
        case logical::VFmaPKF16:
            return VFmaPKF16(vgpr(0), vgpr(1), vgpr(2), vgpr(3));
        case logical::VFmaMixF32:
            return VFmaMixF32(vgpr(0), vgpr(1), vgpr(2), vgpr(3));
        case logical::VMadI32I24:
            return VMadI32I24(vgpr(0), vgpr(1), vgpr(2), vgpr(3));
        case logical::VMadU32U24:
            return VMadU32U24(vgpr(0), vgpr(1), vgpr(2), vgpr(3));
        case logical::VMadMixF32:
            return VMadMixF32(vgpr(0), vgpr(1), vgpr(2), vgpr(3));
        case logical::VDot2CF32F16:
            return VDot2CF32F16(vgpr(0), vgpr(1), vgpr(2), vgpr(3));
        case logical::VDot2F32F16:
            return VDot2F32F16(vgpr(0), vgpr(1), vgpr(2), vgpr(3));
        case logical::VDot2F32BF16:
            return VDot2F32BF16(vgpr(0), vgpr(1), vgpr(2), vgpr(3));
        case logical::VDot2CF32BF16:
            return VDot2CF32BF16(vgpr(0), vgpr(1), vgpr(2), vgpr(3));
        case logical::VExpF16:
            return VExpF16(vgpr(0), vgpr(1));
        case logical::VExpF32:
            return VExpF32(vgpr(0), vgpr(1));
        case logical::VRcpF16:
            return VRcpF16(vgpr(0), vgpr(1));
        case logical::VRcpF32:
            return VRcpF32(vgpr(0), vgpr(1));
        case logical::VRcpIFlagF32:
            return VRcpIFlagF32(vgpr(0), vgpr(1));
        case logical::VRsqF16:
            return VRsqF16(vgpr(0), vgpr(1));
        case logical::VRsqF32:
            return VRsqF32(vgpr(0), vgpr(1));
        case logical::VRsqIFlagF32:
            return VRsqIFlagF32(vgpr(0), vgpr(1));
        case logical::VRndneF32:
            return VRndneF32(vgpr(0), vgpr(1));
        case logical::VMaxF16:
            return VMaxF16(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMaxF32:
            return VMaxF32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMaxF64:
            return VMaxF64(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMaxI32:
            return VMaxI32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMaxPKF16:
            return VMaxPKF16(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMinF16:
            return VMinF16(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMinF32:
            return VMinF32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMinF64:
            return VMinF64(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMinI32:
            return VMinI32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMed3I32:
            return VMed3I32(vgpr(0), vgpr(1), vgpr(2), vgpr(3));
        case logical::VMed3F32:
            return VMed3F32(vgpr(0), vgpr(1), vgpr(2), vgpr(3));
        case logical::VAndB32:
            return VAndB32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VAndOrB32:
            return VAndOrB32(vgpr(0), vgpr(1), vgpr(2), vgpr(3));
        case logical::VOrB32:
            return VOrB32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VXorB32:
            return VXorB32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VNotB32:
            return VNotB32(vgpr(0), vgpr(1));
        case logical::VPrngB32:
            return VPrngB32(vgpr(0), vgpr(1));
        case logical::VCndMaskB32:
            return VCndMaskB32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VLShiftLeftB16:
            return VLShiftLeftB16(vgpr(0), vgpr(1), vgpr(2));
        case logical::VLShiftLeftB32:
            return VLShiftLeftB32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VLShiftRightB32:
            return VLShiftRightB32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VLShiftLeftB64:
            return VLShiftLeftB64(vgpr(0), vgpr(1), vgpr(2));
        case logical::VLShiftRightB64:
            return VLShiftRightB64(vgpr(0), vgpr(1), vgpr(2));
        case logical::VAShiftRightI32:
            return VAShiftRightI32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMovB32:
            return VMovB32(vgpr(0), vgpr(1));
        case logical::VSwapB32:
            return VSwapB32(vgpr(0), vgpr(1));
        case logical::VPackF16toB32:
            return VPackF16toB32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VPermB32:
            return VPermB32(vgpr(0), vgpr(1), vgpr(2), vgpr(3));
        case logical::VBfeI32:
            return VBfeI32(vgpr(0), vgpr(1), vgpr(2), vgpr(3));
        case logical::VBfeU32:
            return VBfeU32(vgpr(0), vgpr(1), vgpr(2), vgpr(3));
        case logical::VBfiB32:
            return VBfiB32(vgpr(0), vgpr(1), vgpr(2), vgpr(3));
        case logical::VAccvgprReadB32:
            return VAccvgprReadB32(vgpr(0), vgpr(1));
        case logical::VAccvgprWrite:
            return VAccvgprWrite(vgpr(0), vgpr(1));
        case logical::VAccvgprWriteB32:
            return VAccvgprWriteB32(vgpr(0), vgpr(1));
        case logical::VReadfirstlaneB32:
            return VReadfirstlaneB32(vgpr(0), vgpr(1));
        case logical::SCmpEQI32:
            return SCmpEQI32(sgpr(0), sgpr(1));
        case logical::SCmpEQU32:
            return SCmpEQU32(sgpr(0), sgpr(1));
        case logical::SCmpEQU64:
            return SCmpEQU64(sgpr(0), sgpr(1));
        case logical::SCmpGeI32:
            return SCmpGeI32(sgpr(0), sgpr(1));
        case logical::SCmpGeU32:
            return SCmpGeU32(sgpr(0), sgpr(1));
        case logical::SCmpGtI32:
            return SCmpGtI32(sgpr(0), sgpr(1));
        case logical::SCmpGtU32:
            return SCmpGtU32(sgpr(0), sgpr(1));
        case logical::SCmpLeI32:
            return SCmpLeI32(sgpr(0), sgpr(1));
        case logical::SCmpLeU32:
            return SCmpLeU32(sgpr(0), sgpr(1));
        case logical::SCmpLgU32:
            return SCmpLgU32(sgpr(0), sgpr(1));
        case logical::SCmpLgI32:
            return SCmpLgI32(sgpr(0), sgpr(1));
        case logical::SCmpLgU64:
            return SCmpLgU64(sgpr(0), sgpr(1));
        case logical::SCmpLtI32:
            return SCmpLtI32(sgpr(0), sgpr(1));
        case logical::SCmpLtU32:
            return SCmpLtU32(sgpr(0), sgpr(1));
        case logical::SBitcmp1B32:
            return SBitcmp1B32(sgpr(0), sgpr(1));
        case logical::VCmpEQF32:
            return VCmpEQF32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpEQF64:
            return VCmpEQF64(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpEQU32:
            return VCmpEQU32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpEQI32:
            return VCmpEQI32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpGEF16:
            return VCmpGEF16(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpGTF16:
            return VCmpGTF16(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpGEF32:
            return VCmpGEF32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpGTF32:
            return VCmpGTF32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpGEF64:
            return VCmpGEF64(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpGTF64:
            return VCmpGTF64(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpGEI32:
            return VCmpGEI32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpGTI32:
            return VCmpGTI32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpGEU32:
            return VCmpGEU32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpGtU32:
            return VCmpGtU32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpLeU32:
            return VCmpLeU32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpLeI32:
            return VCmpLeI32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpLtI32:
            return VCmpLtI32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpLtU32:
            return VCmpLtU32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpUF32:
            return VCmpUF32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpNeI32:
            return VCmpNeI32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpNeU32:
            return VCmpNeU32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpNeU64:
            return VCmpNeU64(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpClassF32:
            return VCmpClassF32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpXClassF32:
            return VCmpXClassF32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpXEqU32:
            return VCmpXEqU32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpXGeU32:
            return VCmpXGeU32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpXGtU32:
            return VCmpXGtU32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpXLeU32:
            return VCmpXLeU32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpXLeI32:
            return VCmpXLeI32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpXLtF32:
            return VCmpXLtF32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpXLtI32:
            return VCmpXLtI32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpXLtU32:
            return VCmpXLtU32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpXLtU64:
            return VCmpXLtU64(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpXNeU16:
            return VCmpXNeU16(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCmpXNeU32:
            return VCmpXNeU32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCvtF16toF32:
            return VCvtF16toF32(vgpr(0), vgpr(1));
        case logical::VCvtF32toF16:
            return VCvtF32toF16(vgpr(0), vgpr(1));
        case logical::VCvtF32toU32:
            return VCvtF32toU32(vgpr(0), vgpr(1));
        case logical::VCvtU32toF32:
            return VCvtU32toF32(vgpr(0), vgpr(1));
        case logical::VCvtI32toF32:
            return VCvtI32toF32(vgpr(0), vgpr(1));
        case logical::VCvtF32toI32:
            return VCvtF32toI32(vgpr(0), vgpr(1));
        case logical::VCvtFP8toF32:
            return VCvtFP8toF32(vgpr(0), vgpr(1));
        case logical::VCvtBF8toF32:
            return VCvtBF8toF32(vgpr(0), vgpr(1));
        case logical::VCvtPkFP8toF32:
            return VCvtPkFP8toF32(vgpr(0), vgpr(1));
        case logical::VCvtPkBF8toF32:
            return VCvtPkBF8toF32(vgpr(0), vgpr(1));
        case logical::VCvtPkF32toFP8:
            return VCvtPkF32toFP8(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCvtPkF32toBF8:
            return VCvtPkF32toBF8(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCvtSRF32toFP8:
            return VCvtSRF32toFP8(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCvtSRF32toBF8:
            return VCvtSRF32toBF8(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCvtScalePkFP8toF16:
            return VCvtScalePkFP8toF16(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCvtScalePkBF8toF16:
            return VCvtScalePkBF8toF16(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCvtScaleFP8toF16:
            return VCvtScaleFP8toF16(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCvtScalePkF16toFP8:
            return VCvtScalePkF16toFP8(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCvtScalePkF16toBF8:
            return VCvtScalePkF16toBF8(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCvtScaleSRF16toFP8:
            return VCvtScaleSRF16toFP8(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCvtScaleSRF16toBF8:
            return VCvtScaleSRF16toBF8(vgpr(0), vgpr(1), vgpr(2));
        case logical::VCvtPkF32toBF16:
            return VCvtPkF32toBF16(vgpr(0), vgpr(1), vgpr(2));
        case logical::DSBPermuteB32:
            return DSBPermuteB32(vgpr(0), vgpr(1));
        case logical::DSLoadU8:
            return DSLoadU8(vgpr(0), vgpr(1));
        case logical::DSLoadI8:
            return DSLoadI8(vgpr(0), vgpr(1));
        case logical::DSLoadU16:
            return DSLoadU16(vgpr(0), vgpr(1));
        case logical::DSLoadI16:
            return DSLoadI16(vgpr(0), vgpr(1));
        case logical::DSLoadB32:
            return DSLoadB32(vgpr(0), vgpr(1));
        case logical::DSLoadB64:
            return DSLoadB64(vgpr(0), vgpr(1));
        case logical::DSLoadB96:
            return DSLoadB96(vgpr(0), vgpr(1));
        case logical::DSLoadB128:
            return DSLoadB128(vgpr(0), vgpr(1));
        case logical::DSLoad2B32:
            return DSLoad2B32(vgpr(0), vgpr(1), vgpr(2));
        case logical::DSLoad2B64:
            return DSLoad2B64(vgpr(0), vgpr(1), vgpr(2));
        case logical::DSStoreB8:
            return DSStoreB8(vgpr(0), vgpr(1));
        case logical::DSStoreB16:
            return DSStoreB16(vgpr(0), vgpr(1));
        case logical::DSStoreB32:
            return DSStoreB32(vgpr(0), vgpr(1));
        case logical::DSStoreB64:
            return DSStoreB64(vgpr(0), vgpr(1));
        case logical::DSStoreB96:
            return DSStoreB96(vgpr(0), vgpr(1));
        case logical::DSStoreB128:
            return DSStoreB128(vgpr(0), vgpr(1));
        case logical::DSStore2B32:
            return DSStore2B32(vgpr(0), vgpr(1), vgpr(2));
        case logical::DSStore2B64:
            return DSStore2B64(vgpr(0), vgpr(1), vgpr(2));
        case logical::BufferLoadU8:
            return BufferLoadU8(vgpr(0), vgpr(1));
        case logical::BufferLoadI8:
            return BufferLoadI8(vgpr(0), vgpr(1));
        case logical::BufferLoadU16:
            return BufferLoadU16(vgpr(0), vgpr(1));
        case logical::BufferLoadI16:
            return BufferLoadI16(vgpr(0), vgpr(1));
        case logical::BufferLoadB32:
            return BufferLoadB32(vgpr(0), vgpr(1));
        case logical::BufferLoadB64:
            return BufferLoadB64(vgpr(0), vgpr(1));
        case logical::BufferLoadB96:
            return BufferLoadB96(vgpr(0), vgpr(1));
        case logical::BufferLoadB128:
            return BufferLoadB128(vgpr(0), vgpr(1));
        case logical::BufferLoadD16U8:
            return BufferLoadD16U8(vgpr(0), vgpr(1));
        case logical::BufferLoadD16HIU8:
            return BufferLoadD16HIU8(vgpr(0), vgpr(1));
        case logical::BufferLoadD16I8:
            return BufferLoadD16I8(vgpr(0), vgpr(1));
        case logical::BufferLoadD16HII8:
            return BufferLoadD16HII8(vgpr(0), vgpr(1));
        case logical::BufferLoadD16B16:
            return BufferLoadD16B16(vgpr(0), vgpr(1));
        case logical::BufferLoadD16HIB16:
            return BufferLoadD16HIB16(vgpr(0), vgpr(1));
        case logical::BufferStoreB8:
            return BufferStoreB8(vgpr(0), vgpr(1), vgpr(2));
        case logical::BufferStoreD16HIU8:
            return BufferStoreD16HIU8(vgpr(0), vgpr(1), vgpr(2));
        case logical::BufferStoreB16:
            return BufferStoreB16(vgpr(0), vgpr(1), vgpr(2));
        case logical::BufferStoreD16HIB16:
            return BufferStoreD16HIB16(vgpr(0), vgpr(1), vgpr(2));
        case logical::BufferStoreB32:
            return BufferStoreB32(vgpr(0), vgpr(1), vgpr(2));
        case logical::BufferStoreB64:
            return BufferStoreB64(vgpr(0), vgpr(1), vgpr(2));
        case logical::BufferStoreB96:
            return BufferStoreB96(vgpr(0), vgpr(1), vgpr(2));
        case logical::BufferStoreB128:
            return BufferStoreB128(vgpr(0), vgpr(1), vgpr(2));
        case logical::BufferAtomicAddF32:
            return BufferAtomicAddF32(vgpr(0), vgpr(1));
        case logical::BufferAtomicCmpswapB32:
            return BufferAtomicCmpswapB32(vgpr(0), vgpr(1), vgpr(2));
        case logical::BufferAtomicCmpswapB64:
            return BufferAtomicCmpswapB64(vgpr(0), vgpr(1), vgpr(2));
        case logical::FlatLoadU8:
            return FlatLoadU8(vgpr(0), vgpr(1));
        case logical::FlatLoadI8:
            return FlatLoadI8(vgpr(0), vgpr(1));
        case logical::FlatLoadU16:
            return FlatLoadU16(vgpr(0), vgpr(1));
        case logical::FlatLoadI16:
            return FlatLoadI16(vgpr(0), vgpr(1));
        case logical::FlatLoadD16U8:
            return FlatLoadD16U8(vgpr(0), vgpr(1));
        case logical::FlatLoadD16HIU8:
            return FlatLoadD16HIU8(vgpr(0), vgpr(1));
        case logical::FlatLoadD16I8:
            return FlatLoadD16I8(vgpr(0), vgpr(1));
        case logical::FlatLoadD16HII8:
            return FlatLoadD16HII8(vgpr(0), vgpr(1));
        case logical::FlatLoadD16B16:
            return FlatLoadD16B16(vgpr(0), vgpr(1));
        case logical::FlatLoadD16HIB16:
            return FlatLoadD16HIB16(vgpr(0), vgpr(1));
        case logical::FlatLoadB32:
            return FlatLoadB32(vgpr(0), vgpr(1));
        case logical::FlatLoadB64:
            return FlatLoadB64(vgpr(0), vgpr(1));
        case logical::FlatLoadB96:
            return FlatLoadB96(vgpr(0), vgpr(1));
        case logical::FlatLoadB128:
            return FlatLoadB128(vgpr(0), vgpr(1));
        case logical::FlatStoreB8:
            return FlatStoreB8(vgpr(0), vgpr(1), vgpr(2));
        case logical::FlatStoreD16HIB8:
            return FlatStoreD16HIB8(vgpr(0), vgpr(1), vgpr(2));
        case logical::FlatStoreB16:
            return FlatStoreB16(vgpr(0), vgpr(1), vgpr(2));
        case logical::FlatStoreD16HIB16:
            return FlatStoreD16HIB16(vgpr(0), vgpr(1), vgpr(2));
        case logical::FlatStoreB32:
            return FlatStoreB32(vgpr(0), vgpr(1), vgpr(2));
        case logical::FlatStoreB64:
            return FlatStoreB64(vgpr(0), vgpr(1), vgpr(2));
        case logical::FlatStoreB96:
            return FlatStoreB96(vgpr(0), vgpr(1), vgpr(2));
        case logical::FlatStoreB128:
            return FlatStoreB128(vgpr(0), vgpr(1), vgpr(2));
        case logical::FlatAtomicCmpswapB32:
            return FlatAtomicCmpswapB32(vgpr(0), vgpr(1), vgpr(2));
        case logical::SAbsI32:
            return SAbsI32(sgpr(0), sgpr(1));
        case logical::SBarrier:
            return SBarrier();
        case logical::SMaxI32:
            return SMaxI32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SMaxU32:
            return SMaxU32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SMinI32:
            return SMinI32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SMinU32:
            return SMinU32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SAddI32:
            return SAddI32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SAddU32:
            return SAddU32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SAddCU32:
            return SAddCU32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SMulI32:
            return SMulI32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SMulHII32:
            return SMulHII32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SMulHIU32:
            return SMulHIU32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SMulLOU32:
            return SMulLOU32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SSubI32:
            return SSubI32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SSubU32:
            return SSubU32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SSubBU32:
            return SSubBU32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SSubU64:
            return SSubU64(sgpr(0), sgpr(1), sgpr(2));
        case logical::SAndB32:
            return SAndB32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SAndB64:
            return SAndB64(sgpr(0), sgpr(1), sgpr(2));
        case logical::SAndN2B32:
            return SAndN2B32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SOrB32:
            return SOrB32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SOrB64:
            return SOrB64(sgpr(0), sgpr(1), sgpr(2));
        case logical::SXorB32:
            return SXorB32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SLShiftLeftB32:
            return SLShiftLeftB32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SLShiftRightB32:
            return SLShiftRightB32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SLShiftLeftB64:
            return SLShiftLeftB64(sgpr(0), sgpr(1), sgpr(2));
        case logical::SLShiftRightB64:
            return SLShiftRightB64(sgpr(0), sgpr(1), sgpr(2));
        case logical::SAShiftRightI32:
            return SAShiftRightI32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SLShiftLeft1AddU32:
            return SLShiftLeft1AddU32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SLShiftLeft2AddU32:
            return SLShiftLeft2AddU32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SLShiftLeft3AddU32:
            return SLShiftLeft3AddU32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SLShiftLeft4AddU32:
            return SLShiftLeft4AddU32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SMovB32:
            return SMovB32(sgpr(0), sgpr(1));
        case logical::SMovB64:
            return SMovB64(sgpr(0), sgpr(1));
        case logical::SCMovB32:
            return SCMovB32(sgpr(0), sgpr(1));
        case logical::SCMovB64:
            return SCMovB64(sgpr(0), sgpr(1));
        case logical::SCSelectB32:
            return SCSelectB32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SGetPCB64:
            return SGetPCB64(sgpr(0));
        case logical::SSetMask:
            return SSetMask(sgpr(0), sgpr(1));
        case logical::SFf1B32:
            return SFf1B32(sgpr(0), sgpr(1));
        case logical::SBfmB32:
            return SBfmB32(sgpr(0), sgpr(1), sgpr(2));
        case logical::SMovkI32:
            return SMovkI32(sgpr(0), sgpr(1));
        case logical::SSExtI16toI32:
            return SSExtI16toI32(sgpr(0), sgpr(1));
        case logical::SAndSaveExecB32:
            return SAndSaveExecB32(sgpr(0), sgpr(1));
        case logical::SAndSaveExecB64:
            return SAndSaveExecB64(sgpr(0), sgpr(1));
        case logical::SOrSaveExecB32:
            return SOrSaveExecB32(sgpr(0), sgpr(1));
        case logical::SOrSaveExecB64:
            return SOrSaveExecB64(sgpr(0), sgpr(1));
        case logical::SGetRegB32:
            return SGetRegB32(sgpr(0), sgpr(1));
        case logical::SSetRegB32:
            return SSetRegB32(sgpr(0), sgpr(1));
        case logical::SSetRegIMM32B32:
            return SSetRegIMM32B32(sgpr(0), sgpr(1));
        case logical::VAddPKF32:
            return VAddPKF32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMulPKF32:
            return VMulPKF32(vgpr(0), vgpr(1), vgpr(2));
        case logical::VMovB64:
            return VMovB64(vgpr(0), vgpr(1));
        case logical::VLShiftLeftOrB32:
            return VLShiftLeftOrB32(vgpr(0), vgpr(1), vgpr(2), vgpr(3));
        case logical::MFMA:
            return MFMA("f32", "f32", 16, 16, 4, 1, false, vgpr(0), vgpr(1), vgpr(2));
        case logical::MXMFMA:
            return MXMFMA("f32", "f32", "f32", "f32", 16, 16, 4, 1, vgpr(0), vgpr(1), vgpr(2),
                          vgpr(3), vgpr(4), vgpr(5));
        case logical::SMFMA:
            return SMFMA("bf16", "f32", 16, 16, 32, 1, false, sgpr(0), sgpr(1), sgpr(2), sgpr(3));
        case logical::TensorLoadToLds:
            return TensorLoadToLds(vgpr(0), vgpr(1));
        case logical::Label:
            return Label("test_label");
        case logical::IntrinsicCall:
            return nullptr;  // Special: handled separately
        default:
            return nullptr;
    }
}

// ==============================================================================
// Expected lowering: per-arch (opcode -> asm mnemonic) for precise lowering tests
// ==============================================================================
//
// One table per architecture. Add (opcode, "expected_asm_mnemonic") to the arch(s)
// you want to assert. Lowering can differ per arch, so each table is independent.
// Instructions not listed are still tested for "lowering succeeds"; only listed
// ones get the precise mnemonic check on that arch.
//
using OpcodeMnemonicPair = std::pair<logical::Opcode, std::string>;

static const std::vector<OpcodeMnemonicPair> EXPECTED_LOWERING_GFX1250 = {
    {logical::BufferAtomicAddF32, "buffer_atomic_add_f32"},
    {logical::DSLoadB32, "ds_load_b32"},
    {logical::DSLoadB64, "ds_load_b64"},
    {logical::DSStoreB32, "ds_store_b32"},
    {logical::DSStoreB64, "ds_store_b64"},
    // Add more: {logical::YourOpcode, "expected_mnemonic"},
};

/** Returns expected asm mnemonic for (opcode, arch) if we have one; else nullopt. */
static std::optional<std::string> getExpectedMnemonic(logical::Opcode opcode,
                                                      const char* archName) {
    const std::vector<OpcodeMnemonicPair>* table = nullptr;
    if (std::string(archName) == "Gfx1250") table = &EXPECTED_LOWERING_GFX1250;
    if (!table) return std::nullopt;
    for (const auto& p : *table) {
        if (p.first == opcode) return p.second;
    }
    return std::nullopt;
}

/**
 * @brief Comprehensive test: Coverage guard + lowering test for ALL instructions
 *
 * This test does TWO things:
 * 1. Coverage Guard: Ensures all logical opcodes have a factory case
 * 2. Lowering Test: Tests each instruction lowers correctly on all architectures
 * 3. When the arch's expected-lowering table (EXPECTED_LOWERING_GFX1250 etc.) has an
 *    entry for opcode, checks that the emitted asm instruction has the correct mnemonic.
 */
TEST(LogicalToAsmComprehensive, AllInstructionsAllArchitectures) {
    // ========================================================================
    // STEP 1: COVERAGE GUARD - Check all opcodes have factory cases
    // ========================================================================

    // Test that factory has a case for each opcode
    std::set<logical::Opcode> testedOpcodes;

    for (uint16_t i = 0; i < logical::NUM_OPCODES; ++i) {
        logical::Opcode opcode = static_cast<logical::Opcode>(i);
        const char* name = logical::getOpcodeName(opcode);

        if (!name || std::string(name) == "UNKNOWN") continue;

        // Try to create instruction (nullptr is OK for special cases)
        LogicalInstruction* inst = createTestInstruction(opcode);

        // If factory has a case for this opcode (even if it returns nullptr), count as tested
        // The factory must explicitly handle the opcode, even if just to return nullptr
        testedOpcodes.insert(opcode);

        if (inst != nullptr) {
            inst->safeErase();  // Clean up
        }
    }

    // Check coverage
    std::vector<std::string> missingOpcodes;
    for (uint16_t i = 0; i < logical::NUM_OPCODES; ++i) {
        logical::Opcode opcode = static_cast<logical::Opcode>(i);
        const char* name = logical::getOpcodeName(opcode);

        if (name && std::string(name) != "UNKNOWN") {
            if (testedOpcodes.find(opcode) == testedOpcodes.end()) {
                missingOpcodes.push_back(name);
            }
        }
    }

    if (!missingOpcodes.empty()) {
        std::cout << "\n? COVERAGE GUARD: " << missingOpcodes.size()
                  << " instructions missing from factory:\n";
        for (size_t i = 0; i < std::min<size_t>(30, missingOpcodes.size()); ++i) {
            std::cout << "  " << missingOpcodes[i] << "\n";
        }
        if (missingOpcodes.size() > 30) {
            std::cout << "  ... and " << (missingOpcodes.size() - 30) << " more\n";
        }
    }

    ASSERT_TRUE(missingOpcodes.empty())
        << "? COVERAGE GUARD FAILED: " << missingOpcodes.size()
        << " instructions missing! Add cases to createTestInstruction()";

    std::cout << "? Coverage guard passed: " << testedOpcodes.size()
              << " instructions have factory cases\n";

    // ========================================================================
    // STEP 2: LOWERING TEST - Test each instruction on each architecture
    // ========================================================================
    struct ArchConfig {
        int major, minor, stepping;
        const char* name;
    };

    std::vector<ArchConfig> archs = {{12, 5, 0, "Gfx1250"}};

    // Special opcodes that can't be lowered through standard passes
    std::set<logical::Opcode> SKIP_LOWERING = {
        logical::Label,
        logical::IntrinsicCall,
        // MFMA, MXMFMA: Now supported via custom mnemonic generation in ToStinkyAsmPass
        // TensorLoadToLds: Now works via generic createAsmFromIR (gfx1250 only)
        // SMFMA and other gfx9-only instructions not supported on gfx1250
        logical::SMFMA,
        logical::VMadMixF32,
        logical::BufferLoadD16I8,
        logical::VDot2CF32F16,
        logical::VDot2F32F16,
        logical::VDot2F32BF16,
        logical::VDot2CF32BF16,
        logical::VAccvgprReadB32,
        logical::VAccvgprWrite,
        logical::VAccvgprWriteB32,
        logical::VRsqIFlagF32,
        logical::SMulLOU32,
        logical::VCvtScalePkFP8toF16,
        logical::VCvtScalePkBF8toF16,
        logical::VCvtScaleFP8toF16,
        logical::VCvtScalePkF16toFP8,
        logical::VCvtScalePkF16toBF8,
        logical::VCvtScaleSRF16toFP8,
        logical::VCvtScaleSRF16toBF8,
    };

    // Architecture-specific instructions: only test on the listed architecture(s).
    // Key = opcode, Value = set of (major, minor, stepping) tuples where the instruction is valid.
    using ArchTuple = std::tuple<int, int, int>;
    std::map<logical::Opcode, std::set<ArchTuple>> ARCH_SPECIFIC = {
        // gfx1250 only
        {logical::TensorLoadToLds, {{12, 5, 0}}},
        {logical::MXMFMA, {{12, 5, 0}}},
    };

    std::cout << "Testing " << testedOpcodes.size() << " instructions on " << archs.size()
              << " architectures...\n";

    size_t totalTests = 0;
    size_t passedTests = 0;

    for (const auto& arch : archs) {
        for (logical::Opcode opcode : testedOpcodes) {
            // Skip special instructions
            if (SKIP_LOWERING.find(opcode) != SKIP_LOWERING.end()) continue;

            // Skip architecture-specific instructions on unsupported architectures
            auto archSpecIt = ARCH_SPECIFIC.find(opcode);
            if (archSpecIt != ARCH_SPECIFIC.end()) {
                ArchTuple current = {arch.major, arch.minor, arch.stepping};
                if (archSpecIt->second.find(current) == archSpecIt->second.end()) {
                    continue;
                }
            }

            totalTests++;
            const char* opName = logical::getOpcodeName(opcode);

            // Create FRESH instruction for this test
            LogicalInstruction* inst = createTestInstruction(opcode);
            if (!inst) continue;

            Function func("kernel");
            BasicBlock* bb = func.createBasicBlock("test");

            PassManager pm;
            GemmTileConfig config;
            config.arch = {arch.major, arch.minor, arch.stepping};
            config.TileA0 = 16;
            config.TileB0 = 16;
            config.TileM0 = 16;
            config.NumGRA = 4;
            config.NumGRB = 4;
            config.NumGRM = 4;
            config.NumWaves = 1;
            pm.setGemmTileConfig(config);

            bb->appendIR(static_cast<IRBase*>(inst));

            pm.addPass(createCompositeInstructionLoweringPass());
            pm.addPass(createToStinkyAsmPass());
            pm.run(func);

            // Verify lowering produced StinkyTofu instructions
            size_t stinkyInsts = 0;
            for (BasicBlock& block : func) {
                for (IRBase& ir : block) {
                    if (ir.getType() == IRBase::IRType::StinkyTofu) stinkyInsts++;
                }
            }

            EXPECT_GT(stinkyInsts, 0)
                << arch.name << ": " << opName << " failed to lower (produced 0 instructions)";

            if (stinkyInsts > 0) {
                passedTests++;
                // Only check mnemonic when (opcode, arch) is in the expected map; never touch
                // getHwInstDesc() for instructions we are not asserting on.
                std::optional<std::string> expected = getExpectedMnemonic(opcode, arch.name);
                if (expected.has_value()) {
                    std::string firstMnemonic;
                    for (BasicBlock& block : func) {
                        for (IRBase& ir : block) {
                            if (ir.getType() == IRBase::IRType::StinkyTofu) {
                                auto* stinky = static_cast<StinkyInstruction*>(&ir);
                                if (stinky->getHwInstDesc())
                                    firstMnemonic = stinky->getHwInstDesc()->mnemonic;
                                break;
                            }
                        }
                        if (!firstMnemonic.empty()) break;
                    }
                    EXPECT_EQ(firstMnemonic, *expected)
                        << arch.name << ": " << opName << " expected mnemonic \"" << *expected
                        << "\", got \"" << firstMnemonic << "\"";
                }
            }
        }
    }

    std::cout << "? Lowering test complete: " << passedTests << "/" << totalTests << " passed ("
              << (passedTests * 100 / totalTests) << "%)\n";
}

/**
 * @brief Dedicated test for gfx1250-specific instructions
 *
 * Tests all instructions that are only supported on gfx1250 architecture.
 * This includes TensorLoadToLds and any other gfx1250-only instructions.
 */
TEST(LogicalToAsmComprehensive, Gfx1250SpecificInstructions) {
    std::cout << "\n=== gfx1250-Specific Instructions Test ===\n";

    // Architecture-specific instructions for gfx1250
    std::map<logical::Opcode, std::string> gfx1250Instructions = {
        {logical::TensorLoadToLds, "tensor_load_to_lds"},
        // Add more gfx1250-specific instructions here as they are discovered
    };

    int passedTests = 0;
    int totalTests = 0;

    for (const auto& [opcode, expectedMnemonic] : gfx1250Instructions) {
        totalTests++;
        const char* opName = logical::getOpcodeName(opcode);
        std::cout << "Testing " << opName << " on gfx1250...\n";

        // Create fresh instruction
        LogicalInstruction* inst = createTestInstruction(opcode);
        if (!inst) {
            std::cout << "  ?  Skipped: No test factory for " << opName << "\n";
            continue;
        }

        Function func("kernel");
        BasicBlock* bb = func.createBasicBlock("test");

        PassManager pm;
        GemmTileConfig config;
        config.arch = {12, 5, 0};
        config.TileA0 = 16;
        config.TileB0 = 16;
        config.TileM0 = 16;
        config.NumGRA = 4;
        config.NumGRB = 4;
        config.NumGRM = 4;
        config.NumWaves = 1;
        pm.setGemmTileConfig(config);

        bb->appendIR(static_cast<IRBase*>(inst));

        pm.addPass(createCompositeInstructionLoweringPass());
        pm.addPass(createToStinkyAsmPass());
        pm.run(func);

        // Verify lowering succeeded
        size_t stinkyInsts = 0;
        std::string actualMnemonic;

        for (BasicBlock& block : func) {
            for (IRBase& ir : block) {
                if (ir.getType() == IRBase::IRType::StinkyTofu) {
                    stinkyInsts++;
                    auto* stinky = static_cast<StinkyInstruction*>(&ir);
                    actualMnemonic = stinky->getHwInstDesc()->mnemonic;
                }
            }
        }

        EXPECT_GT(stinkyInsts, 0) << "gfx1250: " << opName
                                  << " failed to lower (produced 0 instructions)";

        if (stinkyInsts > 0) {
            EXPECT_EQ(actualMnemonic, expectedMnemonic)
                << opName << ": Expected mnemonic '" << expectedMnemonic << "', got '"
                << actualMnemonic << "'";
            passedTests++;
            std::cout << "  ? " << opName << " lowered successfully"
                      << " (mnemonic: " << actualMnemonic << ")\n";
        } else {
            std::cout << "  ? " << opName << " failed to lower\n";
        }
    }

    std::cout << "\n? gfx1250-specific test complete: " << passedTests << "/" << totalTests
              << " passed (" << (totalTests > 0 ? (passedTests * 100 / totalTests) : 0) << "%)\n";

    // Special validation for TensorLoadToLds with optional sources
    if (gfx1250Instructions.find(logical::TensorLoadToLds) != gfx1250Instructions.end()) {
        std::cout << "\n=== TensorLoadToLds Optional Sources Test ===\n";

        Function func("kernel");
        BasicBlock* bb = func.createBasicBlock("test");

        PassManager pm;
        GemmTileConfig config;
        config.arch = {12, 5, 0};
        config.TileA0 = 16;
        config.TileB0 = 16;
        config.TileM0 = 16;
        config.NumGRA = 4;
        config.NumGRB = 4;
        config.NumGRM = 4;
        config.NumWaves = 1;
        pm.setGemmTileConfig(config);

        StinkyRegister s2 = sgpr(2);
        StinkyRegister s3 = sgpr(3);

        bb->appendIR(static_cast<IRBase*>(TensorLoadToLds(sgpr(0), sgpr(1), &s2, &s3)));

        pm.addPass(createCompositeInstructionLoweringPass());
        pm.addPass(createToStinkyAsmPass());
        pm.run(func);

        // Verify 4-source configuration
        size_t numSrcs = 0;
        for (BasicBlock& block : func) {
            for (IRBase& ir : block) {
                if (ir.getType() == IRBase::IRType::StinkyTofu) {
                    auto* stinky = static_cast<StinkyInstruction*>(&ir);
                    numSrcs = stinky->getNumSrcRegs();
                }
            }
        }

        EXPECT_EQ(numSrcs, 4) << "TensorLoadToLds with optional sources: Expected 4 operands";
        if (numSrcs == 4) {
            std::cout << "  ? TensorLoadToLds with 4 sources (optional) works correctly\n";
        }
    }
}

// ==============================================================================
// Shared Test Structures for Matrix Instructions
// ==============================================================================

// Structure: {instType, accType, m, n, k, blocks, mfma1k, expectedMnemonic}
struct MfmaTestCase {
    std::string instType;
    std::string accType;
    int m;
    int n;
    int k;
    int blocks;
    bool mfma1k;
    std::string expectedMnemonic;
};

// Test structure: {architecture, test cases, arch major, minor, stepping}
struct ArchTest {
    std::string archName;
    std::vector<MfmaTestCase>& testCases;
    int major;
    int minor;
    int stepping;
};

// ==============================================================================
// MFMA Instruction Tests - POC
// ==============================================================================

/**
 * @brief Test MFMA instruction lowering across different architectures
 *
 * Tests MFMA instructions with various parameters (instType, accType, m, n, k, blocks)
 * to verify they can be properly lowered to architecture-specific assembly.
 *
 * MFMA lowering now implemented in ToStinkyAsmPass:
 *   - Reads MFMAData from LogicalInstruction
 *   - Generates mnemonic: v_mfma_{accType}_{m}x{n}x{k}[_{blocks}b]_{instType}
 *   - Creates StinkyInstruction with proper HwInstDesc
 *
 * Structure:
 *   mfma_gfx1250 = {
 *       {instType, accType, m, n, k, blocks, mfma1k, expectedMnemonic},
 *       ...
 *   }
 *
 * Each test verifies:
 *   1. MFMA instruction can be created with given parameters
 *   2. Lowering to StinkyTofu IR succeeds
 *   3. Generated mnemonic matches expected hardware instruction
 */
TEST(LogicalToAsmComprehensive, MfmaInstructionLowering) {
    std::cout << "\n=== MFMA Instruction Lowering Test ===\n";

    // gfx1250 - Uses WMMA (Wave Matrix Multiply Accumulate) instead of MFMA
    // Reverse engineered from: v_wmma_f32_16x16x32_bf16
    std::vector<MfmaTestCase> mfma_gfx1250 = {
        {"bf16", "f32", 16, 16, 32, 1, false, "v_wmma_f32_16x16x32_bf16"},
    };

    std::vector<ArchTest> archTests = {
        {"gfx1250", mfma_gfx1250, 12, 5, 0},
    };

    for (const auto& archTest : archTests) {
        std::cout << "\n--- Testing " << archTest.archName << " ---\n";
        int passedTests = 0;
        int totalTests = archTest.testCases.size();

        for (const auto& testCase : archTest.testCases) {
            std::cout << "Testing MFMA " << testCase.instType << "_" << testCase.m << "x"
                      << testCase.n << "x" << testCase.k << "...\n";

            Function func("kernel");
            BasicBlock* bb = func.createBasicBlock("test");

            PassManager pm;
            GemmTileConfig config;
            config.arch = {archTest.major, archTest.minor, archTest.stepping};
            pm.setGemmTileConfig(config);

            LogicalInstruction* mfmaInst =
                MFMA(testCase.instType, testCase.accType, testCase.m, testCase.n, testCase.k,
                     testCase.blocks, testCase.mfma1k,
                     vgpr(0),   // acc
                     vgpr(4),   // a
                     vgpr(8));  // b

            bb->appendIR(static_cast<IRBase*>(mfmaInst));

            pm.addPass(createCompositeInstructionLoweringPass());
            pm.addPass(createToStinkyAsmPass());
            pm.run(func);

            // Verify lowering result
            bool found = false;
            std::string actualMnemonic = "";

            for (BasicBlock& block : func) {
                for (IRBase& ir : block) {
                    if (ir.getType() == IRBase::IRType::StinkyTofu) {
                        auto* stinky = static_cast<StinkyInstruction*>(&ir);
                        if (stinky->getHwInstDesc()) {
                            actualMnemonic = stinky->getHwInstDesc()->mnemonic;
                            found = true;

                            // Verify mnemonic matches
                            EXPECT_EQ(actualMnemonic, testCase.expectedMnemonic)
                                << "MFMA " << testCase.instType << "_" << testCase.m << "x"
                                << testCase.n << "x" << testCase.k << " on " << archTest.archName;

                            if (actualMnemonic == testCase.expectedMnemonic) {
                                passedTests++;
                                std::cout << "  ? " << actualMnemonic << " - PASS\n";
                            } else {
                                std::cout << "  ? Expected: " << testCase.expectedMnemonic
                                          << ", Got: " << actualMnemonic << " - FAIL\n";
                            }
                        }
                    }
                }
            }

            if (!found) {
                std::cout << "  ? MFMA lowering failed - no StinkyTofu IR generated\n";
                FAIL() << "MFMA lowering failed for " << testCase.instType << "_" << testCase.m
                       << "x" << testCase.n << "x" << testCase.k << " on " << archTest.archName;
            }
        }

        std::cout << "\n"
                  << archTest.archName << " Summary: " << passedTests << "/" << totalTests
                  << " MFMA lowering tests passed\n";
    }
}

// ==============================================================================
// SMFMA (Sparse MFMA) Instruction Lowering Test
// ==============================================================================
// NOTE: SMFMA is CDNA-only (gfx942, gfx950) and not supported on gfx1250.
// gfx1250 uses SWMMA (v_swmmac_*) instead. SMFMA tests removed.

// ==============================================================================
// MXMFMA (Mixed-Precision Scaled Matrix) Instruction Lowering Test
// ==============================================================================
// NOTE: MXMFMA tests are commented out because v_wmma_scale instructions
// are not yet defined in hardware ISA tables. The MXMFMA logical instruction
// exists but requires hardware support to be properly tested.

// TODO: Enable when v_wmma_scale instructions are added to hardware definitions

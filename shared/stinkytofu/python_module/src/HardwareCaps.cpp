// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "HardwareCaps.hpp"

#include <array>
#include <mutex>
#include <unordered_map>

#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/ComgrProbe.hpp"
#include "stinkytofu/hardware/ToolchainCaps.hpp"

namespace stinkytofu {
namespace {

using IsaVersion = std::array<uint32_t, 3>;
using MnemonicMap = std::unordered_map<std::string, uint16_t>;

bool hasMnemonic(const MnemonicMap& m, const std::string& name) {
    return m.count(name) > 0;
}

bool hasAnyMnemonic(const MnemonicMap& m, std::initializer_list<const char*> names) {
    for (auto* n : names)
        if (m.count(n) > 0) return true;
    return false;
}

template <typename T>
bool checkInList(const T& a, std::initializer_list<T> list) {
    for (auto& b : list)
        if (a == b) return true;
    return false;
}

bool checkMajorIn(uint32_t major, std::initializer_list<uint32_t> list) {
    for (auto m : list)
        if (major == m) return true;
    return false;
}

bool checkMajorNotIn(uint32_t major, std::initializer_list<uint32_t> list) {
    return !checkMajorIn(major, list);
}

// ── asmCaps ──────────────────────────────────────────────────────────────

bool tryAsm(const std::string& isaName, uint32_t ws, const std::string& asmStr) {
    return tryAssembleWithComgr(asmStr, isaName, ws);
}

bool tryAsmAny(const std::string& isa, uint32_t ws, std::initializer_list<const char*> strs) {
    for (auto* s : strs)
        if (tryAssembleWithComgr(s, isa, ws)) return true;
    return false;
}

std::string formatIsaName(const ArchHelper::ArchInfo* info) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string name = "amdgcn-amd-amdhsa--gfx";
    name += std::to_string(info->major);
    name += std::to_string(info->minor);
    name += kHex[info->stepping & 0xF];
    return name;
}

std::map<std::string, int> initAsmCaps(const IsaVersion& v, const MnemonicMap& m, GfxArchID archID,
                                       const ArchHelper::ArchInfo* info) {
    std::map<std::string, int> rv;

    std::string isaName = formatIsaName(info);
    uint32_t ws = info->waveFrontSize;

    rv["SupportedISA"] = 1;

    rv["HasPartialOOB"] = !checkInList(v, {{12, 5, 0}});

    rv["HasExplicitCO"] = hasAnyMnemonic(m, {"v_add_co_u32"});
    rv["HasExplicitNC"] = hasMnemonic(m, "v_add_nc_u32");

    // DirectToLds tests the "lds" modifier, not just the mnemonic
    rv["HasDirectToLds"] = tryAsmAny(isaName, ws,
                                     {"buffer_load_dword v36, s[24:27], s28 offen offset:0 lds",
                                      "buffer_load_b32 v36, s[24:27], s28 offen offset:0 lds"});
    rv["HasDirectToLdsx4"] = tryAsmAny(isaName, ws,
                                       {"buffer_load_dwordx4 v36, s[24:27], s28 offen offset:0 lds",
                                        "buffer_load_b128 v36, s[24:27], s28 offen offset:0 lds"});
    rv["HasAddLshl"] = hasMnemonic(m, "v_add_lshl_u32");
    rv["HasLshlOr"] = hasMnemonic(m, "v_lshl_or_b32");
    rv["HasSMulHi"] = hasMnemonic(m, "s_mul_hi_u32");
    rv["HasScalarStore"] = hasAnyMnemonic(m, {"s_store_dword", "s_store_b32"});
    rv["HasMFMA_explictB"] = hasMnemonic(m, "v_mfma_f32_32x32x1_2b_f32");
    rv["HasMFMA"] = hasMnemonic(m, "v_mfma_f32_32x32x2bf16") || rv["HasMFMA_explictB"];
    rv["HasMFMA_f64"] = hasAnyMnemonic(m, {"v_mfma_f64_16x16x4f64", "v_mfma_f64_16x16x4_f64"});
    rv["HasMFMA_bf16_1k"] = hasMnemonic(m, "v_mfma_f32_32x32x4bf16_1k");
    rv["HasMFMA_f8"] = hasMnemonic(m, "v_mfma_f32_16x16x32_fp8_fp8");
    rv["HasMFMA_b8"] = hasMnemonic(m, "v_mfma_f32_16x16x32_bf8_bf8");
    rv["HasMFMA_f8f6f4"] = hasMnemonic(m, "v_mfma_f32_16x16x128_f8f6f4");
    rv["HasMFMA_xf32"] = hasMnemonic(m, "v_mfma_f32_32x32x4_xf32");
    rv["HasSMFMA"] = hasMnemonic(m, "v_smfmac_f32_32x32x16_f16");

    rv["HasWMMA"] = hasAnyMnemonic(
        m, {"v_wmma_f32_16x16x16_f16", "v_wmma_f32_16x16x32_bf16", "v_wmma_f32_16x16x4_f32"});
    // V1 tests the 4-register dest encoding; V3 has 8-register dest
    rv["HasWMMA_V1"] =
        tryAsm(isaName, ws, "v_wmma_f32_16x16x16_f16 v[0:3], v[8:15], v[16:23], v[0:3]");
    rv["HasWMMA_V2"] =
        tryAsm(isaName, ws, "v_wmma_f32_16x16x16_f16 v[0:3], v[8:9], v[16:17], v[0:3]");
    rv["HasWMMA_V3"] = hasAnyMnemonic(m, {"v_wmma_f32_16x16x32_bf16", "v_wmma_f32_16x16x4_f32"});
    if (rv["HasWMMA_V3"]) rv["HasWMMA_V2"] = 0;
    rv["HasWMMA_V3_f64"] = hasAnyMnemonic(m, {"v_wmma_f64_16x16x4_f64", "v_wmma_f64_16x16x8_f64"});
    rv["HasWMMA_f8f6f4"] = hasMnemonic(m, "v_wmma_f32_16x16x128_f8f6f4");

    rv["HasAdd_PC_i64"] = 0;

    rv["HasSWMMAC"] = hasAnyMnemonic(m, {"v_swmmac_f32_16x16x32_f16", "v_swmmac_f32_16x16x64_f16"});
    rv["HasSWMMAC_gfx1250"] = hasMnemonic(m, "v_swmmac_f32_16x16x64_f16");

    rv["v_mac_f16"] = hasMnemonic(m, "v_mac_f16");
    rv["v_fma_f16"] = hasMnemonic(m, "v_fma_f16");
    // v_fmac_f16 tests VOP2 3-operand form of v_fma_f16
    rv["v_fmac_f16"] = tryAsm(isaName, ws, "v_fma_f16 v47, v36, v34");
    rv["v_pk_fma_f16"] = hasMnemonic(m, "v_pk_fma_f16");
    // v_pk_fmac_f16 tests VOP2 3-operand form of v_pk_fma_f16
    rv["v_pk_fmac_f16"] = tryAsm(isaName, ws, "v_pk_fma_f16 v47, v36, v34");
    rv["v_pk_add_f32"] = hasMnemonic(m, "v_pk_add_f32");
    rv["v_pk_mul_f32"] = hasMnemonic(m, "v_pk_mul_f32");
    rv["v_mad_mix_f32"] = hasMnemonic(m, "v_mad_mix_f32");
    rv["v_fma_mix_f32"] = hasMnemonic(m, "v_fma_mix_f32");
    rv["v_dot2_f32_f16"] = hasMnemonic(m, "v_dot2_f32_f16");
    rv["v_dot2c_f32_f16"] = hasAnyMnemonic(m, {"v_dot2c_f32_f16", "v_dot2acc_f32_f16"});
    rv["v_dot2_f32_bf16"] = hasMnemonic(m, "v_dot2_f32_bf16");
    rv["v_dot2c_f32_bf16"] = hasAnyMnemonic(m, {"v_dot2c_f32_bf16", "v_dot2acc_f32_bf16"});
    rv["v_dot4_i32_i8"] = hasMnemonic(m, "v_dot4_i32_i8");
    rv["v_dot4c_i32_i8"] = hasMnemonic(m, "v_dot4c_i32_i8");
    // VOP3 form with 4th accumulator operand
    rv["VOP3v_dot4_i32_i8"] = tryAsm(isaName, ws, "v_dot4_i32_i8 v47, v36, v34, v47");
    rv["v_mac_f32"] = hasMnemonic(m, "v_mac_f32");
    rv["v_fma_f32"] = hasMnemonic(m, "v_fma_f32");
    rv["v_fmac_f32"] = hasMnemonic(m, "v_fmac_f32");
    rv["v_fma_f64"] = hasMnemonic(m, "v_fma_f64");
    rv["v_mov_b64"] = hasMnemonic(m, "v_mov_b64");
    rv["s_sub_u64"] = hasMnemonic(m, "s_sub_u64");

    rv["HasBF16CVT"] = hasMnemonic(m, "v_cvt_f32_bf16");
    rv["HasPkF16CVT"] = hasMnemonic(m, "v_cvt_pk_f16_f32");
    rv["Hascvtfp8_f16"] = hasMnemonic(m, "v_cvt_scalef32_pk_fp8_f16");
    rv["Hascvtf16_fp8_sf32"] = hasMnemonic(m, "v_cvt_scalef32_f16_fp8");
    // Tests the byte_sel modifier
    rv["HasCvtFP8toF16"] = tryAsm(isaName, ws, "v_cvt_f16_fp8 v[0], v[1] byte_sel:2");

    rv["HasLDSTrB64B16"] = hasMnemonic(m, "ds_read_b64_tr_b16");
    rv["HasGLTr8B64"] = hasMnemonic(m, "global_load_tr_b64");
    rv["HasGLTr16B128"] = hasMnemonic(m, "global_load_tr_b128");
    rv["HasLDSTrB128B16"] = hasMnemonic(m, "ds_load_tr16_b128");
    rv["HasLDSTrB64B8"] = hasMnemonic(m, "ds_load_tr8_b64");
    rv["HasLDSTrB64B4"] = hasMnemonic(m, "ds_load_tr4_b64");
    rv["HasLDSTrB96B6"] = hasMnemonic(m, "ds_load_tr6_b96");
    rv["HasLDSTr"] =
        rv["HasLDSTrB64B16"] || rv["HasLDSTrB128B16"] || rv["HasLDSTrB64B8"] || rv["HasLDSTrB64B4"];

    rv["v_prng_b32"] = hasMnemonic(m, "v_prng_b32");

    rv["HasAtomicAdd"] = hasAnyMnemonic(m, {"buffer_atomic_add_f32"});

    // Modifier caps: test the actual modifier syntax via comgr
    rv["HasGLCModifier"] =
        tryAsmAny(isaName, ws,
                  {"buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, glc",
                   "buffer_load_dwordx4 v[10:13], v[0], s[0:3], null, offen offset:0, glc"});
    rv["HasSC0Modifier"] =
        tryAsmAny(isaName, ws,
                  {"buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, sc0",
                   "buffer_load_dwordx4 v[10:13], v[0], s[0:3], null, offen offset:0, sc0"});
    rv["HasDLCModifier"] =
        tryAsmAny(isaName, ws,
                  {"buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, dlc",
                   "buffer_load_dwordx4 v[10:13], v[0], s[0:3], null, offen offset:0, dlc"});
    rv["HasSCOPEModifier"] = tryAsmAny(
        isaName, ws,
        {"buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, scope:SCOPE_DEV",
         "buffer_load_dwordx4 v[10:13], v[0], s[0:3], null offen offset:0, scope:SCOPE_DEV"});
    rv["HasNTModifier"] =
        tryAsm(isaName, ws, "buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, nt");
    rv["HasMUBUFConst"] = tryAsmAny(isaName, ws,
                                    {"buffer_load_dword v40, v36, s[24:27], 1 offen offset:0",
                                     "buffer_load_b32 v40, v36, s[24:27], 1 offen offset:0"});
    rv["HasSCMPK"] = hasMnemonic(m, "s_cmpk_gt_u32");

    rv["HasNewBarrier"] = hasMnemonic(m, "s_barrier_wait");
    rv["HasTDM"] = hasMnemonic(m, "tensor_load_to_lds");

    rv["s_delay_alu"] = hasMnemonic(m, "s_delay_alu");

    // VgprMSB — probed via comgr (ToolchainCaps), fall back to mnemonic presence
    auto toolchainCaps = ToolchainCaps::probe(archID);
    bool mnemonicHasMsb = hasMnemonic(m, "s_set_vgpr_msb");
    rv["HasVgprMSB"] = (toolchainCaps.vgprMsbMode != VgprMsbMode::None) || mnemonicHasMsb;
    rv["HasVgprMSB16"] = (toolchainCaps.vgprMsbMode == VgprMsbMode::Msb16);
    rv["ShortBranchMaxLength"] = rv["HasVgprMSB"] ? 8192 : 16384;

    rv["SeparateVscnt"] = hasMnemonic(m, "s_waitcnt_vscnt");
    rv["SeparateLGKMcnt"] = hasMnemonic(m, "s_wait_dscnt") && hasMnemonic(m, "s_wait_kmcnt");
    rv["SeparateVMcnt"] = hasMnemonic(m, "s_wait_loadcnt") && hasMnemonic(m, "s_wait_storecnt");

    if (rv["SeparateVMcnt"]) {
        rv["MaxLoadcnt"] = 63;
        rv["MaxStorecnt"] = 63;
    } else {
        rv["MaxVmcnt"] = rv["SeparateVscnt"] ? 63 : 15;
        if (rv["SeparateVscnt"]) {
            rv["MaxVscnt"] = 63;
        }
    }

    if (rv["SeparateLGKMcnt"]) {
        rv["MaxDscnt"] = 63;
        rv["MaxKmcnt"] = 31;
    } else {
        rv["MaxLgkmcnt"] = 15;
    }

    rv["HasXcnt"] = hasMnemonic(m, "s_wait_xcnt");
    if (rv["HasXcnt"]) {
        rv["MaxXcnt"] = 63;
    }

    rv["SupportedSource"] = 1;

    return rv;
}

// ── archCaps ─────────────────────────────────────────────────────────────
std::map<std::string, int> initArchCaps(const IsaVersion& v) {
    std::map<std::string, int> rv;

    rv["HasEccHalf"] =
        checkInList(v, {{9, 0, 6}, {9, 0, 8}, {9, 0, 10}, {9, 4, 2}, {9, 5, 0}, {12, 5, 0}});
    rv["Waitcnt0Disabled"] = checkInList(v, {{9, 0, 8}, {9, 0, 10}, {9, 4, 2}, {9, 5, 0}});

    int deviceLDS = 65536;
    if (checkInList(v, {{9, 5, 0}}))
        deviceLDS = 163840;
    else if (checkInList(v, {{12, 5, 0}}))
        deviceLDS = 327680;
    rv["DeviceLDS"] = deviceLDS;

    rv["CMPXWritesSGPR"] = checkMajorNotIn(v[0], {10, 11, 12});
    rv["HasWave32"] = checkMajorIn(v[0], {10, 11, 12});
    rv["HasSchedMode"] = checkMajorIn(v[0], {12});
    rv["HasAccCD"] = checkInList(v, {{9, 0, 10}, {9, 4, 2}, {9, 5, 0}});
    rv["ArchAccUnifiedRegs"] = checkInList(v, {{9, 0, 10}, {9, 4, 2}, {9, 5, 0}});
    rv["CrosslaneWait"] = checkInList(v, {{9, 4, 2}, {9, 5, 0}});
    rv["TransOpWait"] = checkInList(v, {{9, 4, 2}, {9, 5, 0}, {12, 5, 0}});
    rv["SDWAWait"] = checkInList(v, {{9, 4, 2}, {9, 5, 0}, {12, 5, 0}});
    rv["VgprBank"] = checkMajorIn(v[0], {10, 11, 12});
    rv["DSLow16NotPreserve"] = v[0] == 12;
    rv["WorkGroupIdFromTTM"] = v[0] == 12;
    rv["NoSDWA"] = checkMajorIn(v[0], {11, 12});
    rv["VOP3ByteSel"] = v[0] == 12;
    rv["HasFP8_OCP"] = v[0] == 12;
    rv["HasWmmaArbStallBit"] = v[0] == 12 && v[1] == 5;
    rv["HasF32XEmulation"] = checkInList(v, {{9, 5, 0}, {12, 5, 0}});
    rv["MaxSgprPreload"] = checkInList(v, {{12, 5, 0}}) ? 32 : 16;
    rv["SgprPreloadPad"] =
        checkInList(v, {{9, 5, 0}}) || checkInList(v, {{9, 0, 10}}) || (v[0] == 9 && v[1] == 4);
    rv["HasMXScaleSwizzle"] = checkInList(v, {{9, 5, 0}, {12, 5, 0}});
    rv["HasInvWbDevFences"] = checkInList(v, {{12, 5, 0}});
    rv["RequiresXCntForVolatileVMEM"] = checkInList(v, {{12, 5, 0}});

    rv["LDSBankCount"] = 64;
    rv["LDSBankWidth"] = 4;

    return rv;
}

// ── regCaps ──────────────────────────────────────────────────────────────
std::map<std::string, int> initRegCaps(const IsaVersion& v,
                                       const std::map<std::string, int>& archCaps) {
    std::map<std::string, int> rv;

    rv["MaxVgpr"] = (v[0] == 12 && v[1] == 5) ? 1024 : 256;
    rv["MaxSgpr"] = 102;
    rv["PhysicalMaxVgpr"] = (v[0] == 12 && v[1] == 5) ? 1024 : 512;
    rv["PhysicalMaxSgpr"] = 800;
    rv["maxLDSConstOffset"] = 65536;

    if (v[0] == 10) {
        rv["PhysicalMaxVgprCU"] = 1024 * 32;
    } else if (v[0] == 11) {
        if (v[1] == 5) {
            if (v[2] == 0 || v[2] == 2 || v[2] == 3) rv["PhysicalMaxVgprCU"] = 2 * 1024 * 32;
            if (v[2] == 1) rv["PhysicalMaxVgprCU"] = 2 * 1536 * 32;
        } else if (v[2] == 2) {
            rv["PhysicalMaxVgprCU"] = 1024 * 32;
        } else {
            rv["PhysicalMaxVgprCU"] = 1536 * 32;
        }
    } else if (v[0] == 12) {
        rv["PhysicalMaxVgprCU"] = (v[1] == 5) ? 4096 * 32 : 1536 * 32;
    } else if (v[0] == 9) {
        auto it = archCaps.find("ArchAccUnifiedRegs");
        bool unified = it != archCaps.end() && it->second;
        rv["PhysicalMaxVgprCU"] = unified ? 2048 * 64 : 1024 * 64;
    } else if (v[0] == 8) {
        rv["PhysicalMaxVgprCU"] = 1024 * 64;
    } else {
        rv["PhysicalMaxVgprCU"] = 0;
    }

    return rv;
}

// ── asmBugs ──────────────────────────────────────────────────────────────
std::map<std::string, bool> initAsmBugs(const std::map<std::string, int>& asmCaps) {
    std::map<std::string, bool> rv;
    auto get = [&](const char* key) {
        auto it = asmCaps.find(key);
        return it != asmCaps.end() && it->second;
    };
    rv["ExplicitCO"] = get("HasExplicitCO");
    rv["ExplicitNC"] = get("HasExplicitNC");
    return rv;
}

// ── Cache ────────────────────────────────────────────────────────────────
struct CacheEntry {
    std::once_flag flag;
    HardwareCapsResult result;
};

constexpr size_t kMaxArchs = 8;
CacheEntry g_hwcapsCache[kMaxArchs];

}  // namespace

HardwareCapsResult HardwareCaps::query(uint32_t major, uint32_t minor, uint32_t stepping) {
    auto archID = getGfxArchID(major, minor, stepping);
    auto idx = static_cast<size_t>(archID);
    if (idx >= kMaxArchs) return {};

    auto& entry = g_hwcapsCache[idx];
    std::call_once(entry.flag, [&] {
        const auto* info = ArchHelper::getInstance().getArchInfo(archID);
        if (!info) return;

        IsaVersion v = {info->major, info->minor, info->stepping};
        const auto& mnemonicMap = info->getMnemonicToIsaOpcodeMap();

        entry.result.asmCaps = initAsmCaps(v, mnemonicMap, archID, info);
        entry.result.archCaps = initArchCaps(v);
        entry.result.regCaps = initRegCaps(v, entry.result.archCaps);
        entry.result.asmBugs = initAsmBugs(entry.result.asmCaps);
    });

    return entry.result;
}

}  // namespace stinkytofu

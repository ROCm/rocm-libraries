/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "ModifierSerializer.hpp"

#include <climits>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

namespace stinkytofu {
namespace {
int getInt(const std::unordered_map<std::string, std::string>& m, const std::string& key, int def) {
    auto it = m.find(key);
    if (it == m.end()) return def;
    const std::string& s = it->second;
    if (s.empty()) return def;
    char* end = nullptr;
    long val = std::strtol(s.c_str(), &end, 0);
    if (end != s.c_str() + s.size()) return def;
    if (val < INT_MIN || val > INT_MAX) return def;
    return static_cast<int>(val);
}

bool getBool(const std::unordered_map<std::string, std::string>& m, const std::string& key,
             bool def) {
    auto it = m.find(key);
    if (it == m.end()) return def;
    const std::string& v = it->second;
    if (v == "true" || v == "1") return true;
    if (v == "false" || v == "0") return false;
    return def;
}

std::string getStr(const std::unordered_map<std::string, std::string>& m, const std::string& key,
                   const std::string& def = "") {
    auto it = m.find(key);
    if (it == m.end()) return def;
    return it->second;
}

/// Parse serialized int vector form `[0,1]` (matches IRParser / vectorToString).
std::vector<int> getIntVector(const std::unordered_map<std::string, std::string>& m,
                              const std::string& key) {
    const std::string s = getStr(m, key);
    std::vector<int> result;
    if (s.size() < 2 || s.front() != '[' || s.back() != ']') return result;

    const std::string inner = s.substr(1, s.size() - 2);
    size_t pos = 0;
    while (pos < inner.size()) {
        while (pos < inner.size() && (inner[pos] == ' ' || inner[pos] == '\t' || inner[pos] == ','))
            ++pos;
        if (pos >= inner.size()) break;
        const size_t start = pos;
        while (pos < inner.size() && inner[pos] != ',') ++pos;
        std::string token = inner.substr(start, pos - start);
        while (!token.empty() && (token.back() == ' ' || token.back() == '\t')) token.pop_back();
        if (token.empty()) continue;
        char* endPtr = nullptr;
        long val = std::strtol(token.c_str(), &endPtr, 0);
        if (endPtr != token.c_str() + token.size()) return {};
        if (val < INT_MIN || val > INT_MAX) return {};
        result.push_back(static_cast<int>(val));
    }
    return result;
}

const char* delayAluInstTypeStr(SDelayAluData::InstType t) {
    switch (t) {
        case SDelayAluData::InstType::VALU:
            return "VALU";
        case SDelayAluData::InstType::SALU:
            return "SALU";
        case SDelayAluData::InstType::TRANS:
            return "TRANS";
        case SDelayAluData::InstType::NO_DEP:
            return "NO_DEP";
        default:
            return "UNKNOWN";
    }
}

std::string vectorToString(const std::vector<int>& vec) {
    std::string result = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        result += std::to_string(vec[i]);
        if (i < vec.size() - 1) result += ",";
    }
    result += "]";
    return result;
}

}  // anonymous namespace

/*
 * serializeVisit for each modifier type
 * -------------------------------------
 */
namespace {
// Dummy serializeVisit for unknown modifier types.
template <unsigned Dummy = 0>
bool serializeVisit(const Modifier& mod, std::ostream& os) {
    return false;
}

// DSModifiers
bool serializeVisit(const DSModifiers& mod, std::ostream& os) {
    os << ", mod.ds = {";
    os << " na = " << mod.na << ", offset = " << mod.offset;
    if (mod.na == 2) os << ", offset0 = " << mod.offset0 << ", offset1 = " << mod.offset1;
    os << ", gds = " << (mod.gds ? "true" : "false");
    os << " }";
    return true;
}

// FLATModifiers
bool serializeVisit(const FLATModifiers& mod, std::ostream& os) {
    os << ", mod.flat = {";
    os << " offset12 = " << mod.offset12 << ", glc = " << (mod.glc ? "true" : "false")
       << ", slc = " << (mod.slc ? "true" : "false") << ", lds = " << (mod.lds ? "true" : "false");
    os << " }";
    return true;
}

// GLOBALModifiers — offset plus the temporal hint / cache scope used by
// global_prefetch_b8 (gl2-prefetch). Serialized so the .stir IR roundtrip
// preserves the hint/scope; TH_NONE / SCOPE_NONE are omitted.
bool serializeVisit(const GLOBALModifiers& mod, std::ostream& os) {
    os << ", mod.global = { offset = " << mod.offset;
    if (hasTemporalHint(mod.th)) {
        os << ", th = \"" << toString(mod.th) << "\"";
    }
    if (mod.scope != MUBUFScope::SCOPE_NONE) {
        os << ", scope = \"" << toString(mod.scope) << "\"";
    }
    os << " }";
    return true;
}

// MUBUFModifiers
bool serializeVisit(const MUBUFModifiers& mod, std::ostream& os) {
    os << ", mod.mubuf = {";
    os << " offen = " << (mod.offen ? "true" : "false") << ", offset12 = " << mod.offset12
       << ", glc = " << (mod.glc ? "true" : "false") << ", slc = " << (mod.slc ? "true" : "false")
       << ", nt = " << (mod.nt ? "true" : "false") << ", lds = " << (mod.lds ? "true" : "false");
    if (mod.scope != MUBUFScope::SCOPE_NONE) {
        os << ", scope = \"" << toString(mod.scope) << "\"";
    }
    os << " }";
    return true;
}

// CacheScopeModifiers — dedicated cache-scope carrier for SOPP-format memory
// fences (global_wb / global_inv on gfx1250+). Serialized so the .stir IR
// roundtrip preserves the scope token; otherwise a fence written out and
// reparsed would silently lose its scope (worst-case fence-scope demotion).
bool serializeVisit(const CacheScopeModifiers& mod, std::ostream& os) {
    os << ", mod.cache_scope = {";
    if (mod.scope != MUBUFScope::SCOPE_NONE) {
        os << " scope = \"" << toString(mod.scope) << "\"";
    }
    os << " }";
    return true;
}

// SMEMModifiers
bool serializeVisit(const SMEMModifiers& mod, std::ostream& os) {
    os << ", mod.smem = {";
    os << " offset = " << mod.offset << ", glc = " << (mod.glc ? "true" : "false")
       << ", nv = " << (mod.nv ? "true" : "false");
    os << " }";
    return true;
}

// SDWAModifiers
bool serializeVisit(const SDWAModifiers& mod, std::ostream& os) {
    os << ", mod.sdwa = {";
    os << " dst_sel = " << static_cast<int>(mod.dst_sel)
       << ", dst_unused = " << static_cast<int>(mod.dst_unused)
       << ", src0_sel = " << static_cast<int>(mod.src0_sel)
       << ", src1_sel = " << static_cast<int>(mod.src1_sel);
    os << " }";
    return true;
}

// DPPModifiers
bool serializeVisit(const DPPModifiers& mod, std::ostream& os) {
    os << ", mod.dpp = {";
    if (mod.isDPP8) {
        os << " isDPP8 = true, dpp8 = [" << (int)mod.dpp8[0];
        for (int i = 1; i < 8; ++i) os << "," << (int)mod.dpp8[i];
        os << "]";
    } else {
        os << " dppCtrl = " << static_cast<int>(mod.dppCtrl);
        os << ", rowMask = " << (int)mod.rowMask;
        os << ", bankMask = " << (int)mod.bankMask;
    }
    os << ", boundCtrl = " << (int)mod.boundCtrl;
    os << ", fi = " << (int)mod.fi;
    os << " }";
    return true;
}

// VOP3Modifiers
bool serializeVisit(const VOP3Modifiers& mod, std::ostream& os) {
    os << ", mod.vop3 = {";
    os << " neg_src0 = " << (mod.neg_src0 ? "true" : "false")
       << ", neg_src1 = " << (mod.neg_src1 ? "true" : "false")
       << ", neg_src2 = " << (mod.neg_src2 ? "true" : "false")
       << ", abs_src0 = " << (mod.abs_src0 ? "true" : "false")
       << ", abs_src1 = " << (mod.abs_src1 ? "true" : "false")
       << ", abs_src2 = " << (mod.abs_src2 ? "true" : "false")
       << ", clamp = " << (mod.clamp ? "true" : "false") << ", omod = " << mod.omod;
    os << " }";
    return true;
}

// VOP3PModifiers
bool serializeVisit(const VOP3PModifiers& mod, std::ostream& os) {
    os << ", mod.vop3p = {";
    os << " op_sel = " << vectorToString(mod.op_sel)
       << ", op_sel_hi = " << vectorToString(mod.op_sel_hi)
       << ", byte_sel = " << vectorToString(mod.byte_sel);
    os << " }";
    return true;
}

// True16Modifiers
bool serializeVisit(const True16Modifiers& mod, std::ostream& os) {
    os << ", mod.true16 = {";
    os << " dst0 = " << static_cast<int>(mod.getDst0())
       << ", dst1 = " << static_cast<int>(mod.getDst1());
    for (int i = 0; i < static_cast<int>(mod.getSrcCount()); ++i)
        os << ", src" << i << " = " << static_cast<int>(mod.getSrc(i));
    os << " }";
    return true;
}

// EXEC
bool serializeVisit(const EXEC& mod, std::ostream& os) {
    os << ", mod.exec = { setHi = " << (mod.setHi ? "true" : "false") << " }";
    return true;
}

// VCC
bool serializeVisit(const VCC& mod, std::ostream& os) {
    os << ", mod.vcc = { setHi = " << (mod.setHi ? "true" : "false") << " }";
    return true;
}

// SWaitCntData
bool serializeVisit(const SWaitCntData& mod, std::ostream& os) {
    os << ", mod.swaitcnt = {";
    os << " vlcnt = " << static_cast<int>(mod.vlcnt) << ", vscnt = " << static_cast<int>(mod.vscnt)
       << ", dlcnt = " << static_cast<int>(mod.dlcnt) << ", dscnt = " << static_cast<int>(mod.dscnt)
       << ", kmcnt = " << static_cast<int>(mod.kmcnt);
    os << " }";
    return true;
}

// SWaitTensorCntData
bool serializeVisit(const SWaitTensorCntData& mod, std::ostream& os) {
    os << ", mod.swaittensorcnt = { tlcnt = " << static_cast<int>(mod.tlcnt) << " }";
    return true;
}

// SWaitStoreCntData
bool serializeVisit(const SWaitStoreCntData& mod, std::ostream& os) {
    os << ", mod.swaitstorecnt = { storecnt = " << static_cast<int>(mod.storecnt) << " }";
    return true;
}

// SDelayAluData
bool serializeVisit(const SDelayAluData& mod, std::ostream& os) {
    os << ", mod.delayalu = {";
    os << " instid0Type = \"" << delayAluInstTypeStr(mod.instid0Type) << "\""
       << ", instid0Distance = " << static_cast<int>(mod.instid0Distance);
    if (mod.hasInstId1) {
        os << ", instSkip = " << static_cast<int>(mod.instSkip) << ", instid1Type = \""
           << delayAluInstTypeStr(mod.instid1Type) << "\""
           << ", instid1Distance = " << static_cast<int>(mod.instid1Distance);
    }
    os << " }";
    return true;
}

// SWaitAluData
bool serializeVisit(const SWaitAluData& mod, std::ostream& os) {
    os << ", mod.waitalu = {";
    bool first = true;
    if (mod.hasField(SWaitAluData::VA_VDST)) {
        if (!first) os << ",";
        os << " va_vdst = " << mod.getField(SWaitAluData::VA_VDST);
        first = false;
    }
    if (mod.hasField(SWaitAluData::VA_SDST)) {
        if (!first) os << ",";
        os << " va_sdst = " << mod.getField(SWaitAluData::VA_SDST);
        first = false;
    }
    if (mod.hasField(SWaitAluData::VA_SSRC)) {
        if (!first) os << ",";
        os << " va_ssrc = " << mod.getField(SWaitAluData::VA_SSRC);
        first = false;
    }
    if (mod.hasField(SWaitAluData::HOLD_CNT)) {
        if (!first) os << ",";
        os << " hold_cnt = " << mod.getField(SWaitAluData::HOLD_CNT);
        first = false;
    }
    if (mod.hasField(SWaitAluData::VM_VSRC)) {
        if (!first) os << ",";
        os << " vm_vsrc = " << mod.getField(SWaitAluData::VM_VSRC);
        first = false;
    }
    if (mod.hasField(SWaitAluData::VA_VCC)) {
        if (!first) os << ",";
        os << " va_vcc = " << mod.getField(SWaitAluData::VA_VCC);
        first = false;
    }
    if (mod.hasField(SWaitAluData::SA_SDST)) {
        if (!first) os << ",";
        os << " sa_sdst = " << mod.getField(SWaitAluData::SA_SDST);
    }
    os << " }";
    return true;
}

// MFMAModifiers
bool serializeVisit(const MFMAModifiers& mod, std::ostream& os) {
    os << ", mod.mfma = {";
    os << " reuseA = " << (mod.reuseA ? "true" : "false")
       << ", reuseB = " << (mod.reuseB ? "true" : "false");
    if (!mod.negBits.empty()) {
        os << ", negLo = [" << (int)mod.negBits.negLo[0];
        for (int i = 1; i < mod.negBits.numSrcs; ++i) os << "," << (int)mod.negBits.negLo[i];
        os << "], negHi = [" << (int)mod.negBits.negHi[0];
        for (int i = 1; i < mod.negBits.numSrcs; ++i) os << "," << (int)mod.negBits.negHi[i];
        os << "], numNegSrcs = " << (int)mod.negBits.numSrcs;
    }
    os << " }";
    return true;
}

// MatrixFmtModifiers
bool serializeVisit(const MatrixFmtModifiers& mod, std::ostream& os) {
    os << ", mod.matrix_fmt = {";
    bool first = true;
    auto sep = [&]() {
        os << (first ? " " : ", ");
        first = false;
    };
    if (mod.fmtA != MatrixFmt::NONE) {
        sep();
        os << "fmtA = \"" << matrixFmtToStr(mod.fmtA) << "\"";
    }
    if (mod.fmtB != MatrixFmt::NONE) {
        sep();
        os << "fmtB = \"" << matrixFmtToStr(mod.fmtB) << "\"";
    }
    if (mod.scaleFmtA != MatrixScaleFmt::NONE) {
        sep();
        os << "scaleFmtA = \"" << matrixScaleFmtToStr(mod.scaleFmtA) << "\"";
    }
    if (mod.scaleFmtB != MatrixScaleFmt::NONE) {
        sep();
        os << "scaleFmtB = \"" << matrixScaleFmtToStr(mod.scaleFmtB) << "\"";
    }
    os << " }";
    return true;
}

// MemTokenData
bool serializeVisit(const MemTokenData& mod, std::ostream& os) {
    os << ", mod.memtoken = { tokens = " << vectorToString(mod.tokens) << " }";
    return true;
}

// LabelData
bool serializeVisit(const LabelData& mod, std::ostream& os) {
    os << ", mod.label = { label = \"" << mod.label << "\""
       << ", alignment = " << static_cast<int>(mod.alignment) << " }";
    return true;
}

template <typename ModifierType, typename... Rest, unsigned Dummy = 0>
bool serializeVisit(const Modifier& mod, std::ostream& os) {
    if (auto* modifier = dyn_cast<ModifierType>(&mod)) {
        return serializeVisit(*modifier, os);
    }
    return serializeVisit<Rest...>(mod, os);
}
}  // namespace

bool ModifierSerializer::serialize(const Modifier& mod, std::ostream& os) {
    return serializeVisit<DSModifiers, FLATModifiers, GLOBALModifiers, MUBUFModifiers,
                          CacheScopeModifiers, SMEMModifiers, SDWAModifiers, DPPModifiers,
                          VOP3Modifiers, VOP3PModifiers, True16Modifiers, EXEC, VCC, SWaitCntData,
                          SWaitTensorCntData, SWaitStoreCntData, SDelayAluData, SWaitAluData,
                          MFMAModifiers, MatrixFmtModifiers, MemTokenData, LabelData>(mod, os);
}

/*
 * deserializeVisit
 * -------------------------------------
 */
namespace {
/// Parsed modifier fields for one modifier (field name -> value).
using ModifierFieldMap = std::unordered_map<std::string, std::string>;

void deserializeVisit(StinkyInstruction* inst, const std::string& attrKey,
                      const ModifierFieldMap& fields) {
    if (attrKey == "mod.ds") {
        inst->addModifier(DSModifiers(getInt(fields, "na", 1), getInt(fields, "offset", 0),
                                      getInt(fields, "offset0", 0), getInt(fields, "offset1", 0),
                                      getBool(fields, "gds", false)));
    } else if (attrKey == "mod.flat") {
        inst->addModifier(
            FLATModifiers(getInt(fields, "offset12", 0), getBool(fields, "glc", false),
                          getBool(fields, "slc", false), getBool(fields, "lds", false)));
    } else if (attrKey == "mod.global") {
        inst->addModifier(GLOBALModifiers(getInt(fields, "offset", 0),
                                          parseTemporalHint(getStr(fields, "th", "")),
                                          parseMUBUFScope(getStr(fields, "scope", ""))));
    } else if (attrKey == "mod.mubuf") {
        MUBUFScope scope = parseMUBUFScope(getStr(fields, "scope", ""));
        inst->addModifier(
            MUBUFModifiers(getBool(fields, "offen", false), getInt(fields, "offset12", 0),
                           getBool(fields, "glc", false), getBool(fields, "slc", false),
                           getBool(fields, "nt", false), getBool(fields, "lds", false), false,
                           false, false, false, scope));
    } else if (attrKey == "mod.cache_scope") {
        inst->addModifier(CacheScopeModifiers(parseMUBUFScope(getStr(fields, "scope", ""))));
    } else if (attrKey == "mod.smem") {
        inst->addModifier(SMEMModifiers(getBool(fields, "glc", false), getBool(fields, "nv", false),
                                        getInt(fields, "offset", 0)));
    } else if (attrKey == "mod.vop3") {
        inst->addModifier(
            VOP3Modifiers(getBool(fields, "neg_src0", false), getBool(fields, "neg_src1", false),
                          getBool(fields, "neg_src2", false), getBool(fields, "abs_src0", false),
                          getBool(fields, "abs_src1", false), getBool(fields, "abs_src2", false),
                          getBool(fields, "clamp", false), getInt(fields, "omod", 0)));
    } else if (attrKey == "mod.exec") {
        inst->addModifier(EXEC(getBool(fields, "setHi", false)));
    } else if (attrKey == "mod.vcc") {
        inst->addModifier(VCC(getBool(fields, "setHi", false)));
    } else if (attrKey == "mod.swaitcnt") {
        inst->addModifier(SWaitCntData(static_cast<int8_t>(getInt(fields, "vlcnt", -1)),
                                       static_cast<int8_t>(getInt(fields, "vscnt", -1)),
                                       static_cast<int8_t>(getInt(fields, "dlcnt", -1)),
                                       static_cast<int8_t>(getInt(fields, "dscnt", -1)),
                                       static_cast<int8_t>(getInt(fields, "kmcnt", -1))));
    } else if (attrKey == "mod.swaittensorcnt") {
        inst->addModifier(SWaitTensorCntData(static_cast<int8_t>(getInt(fields, "tlcnt", -1))));
    } else if (attrKey == "mod.swaitstorecnt") {
        inst->addModifier(SWaitStoreCntData(static_cast<int8_t>(getInt(fields, "storecnt", -1))));
    } else if (attrKey == "mod.mfma") {
        MFMAModifiers mod;
        mod.reuseA = getBool(fields, "reuseA", false);
        mod.reuseB = getBool(fields, "reuseB", false);

        // Neg bits
        if (fields.contains("negLo")) {
            auto loVec = getIntVector(fields, "negLo");
            auto hiVec = getIntVector(fields, "negHi");
            mod.negBits.numSrcs =
                static_cast<uint8_t>(getInt(fields, "numNegSrcs", static_cast<int>(loVec.size())));
            for (size_t i = 0; i < loVec.size() && i < 3; ++i)
                mod.negBits.negLo[i] = static_cast<uint8_t>(loVec[i]);
            for (size_t i = 0; i < hiVec.size() && i < 3; ++i)
                mod.negBits.negHi[i] = static_cast<uint8_t>(hiVec[i]);
        }

        inst->addModifier(mod);
    } else if (attrKey == "mod.matrix_fmt") {
        MatrixFmtModifiers mod;
        if (fields.contains("fmtA")) mod.fmtA = parseMatrixFmt(getStr(fields, "fmtA"));
        if (fields.contains("fmtB")) mod.fmtB = parseMatrixFmt(getStr(fields, "fmtB"));
        if (fields.contains("scaleFmtA"))
            mod.scaleFmtA = parseMatrixScaleFmt(getStr(fields, "scaleFmtA"));
        if (fields.contains("scaleFmtB"))
            mod.scaleFmtB = parseMatrixScaleFmt(getStr(fields, "scaleFmtB"));
        inst->addModifier(mod);
    } else if (attrKey == "mod.delayalu") {
        auto toInstType = [](const std::string& s) {
            if (s == "VALU") return SDelayAluData::InstType::VALU;
            if (s == "SALU") return SDelayAluData::InstType::SALU;
            if (s == "TRANS") return SDelayAluData::InstType::TRANS;
            return SDelayAluData::InstType::NO_DEP;
        };
        bool hasInstId1 = getBool(fields, "hasInstId1", false) || fields.contains("instid1Type") ||
                          fields.contains("instSkip") || fields.contains("instid1Distance");
        if (hasInstId1) {
            inst->addModifier(
                SDelayAluData(toInstType(getStr(fields, "instid0Type", "NO_DEP")),
                              static_cast<int8_t>(getInt(fields, "instid0Distance", 0)),
                              static_cast<int8_t>(getInt(fields, "instSkip", 0)),
                              toInstType(getStr(fields, "instid1Type", "NO_DEP")),
                              static_cast<int8_t>(getInt(fields, "instid1Distance", 0))));
        } else {
            inst->addModifier(
                SDelayAluData(toInstType(getStr(fields, "instid0Type", "NO_DEP")),
                              static_cast<int8_t>(getInt(fields, "instid0Distance", 0))));
        }
    } else if (attrKey == "mod.waitalu") {
        inst->addModifier(SWaitAluData(getInt(fields, "va_vdst", -1), getInt(fields, "va_sdst", -1),
                                       getInt(fields, "va_ssrc", -1),
                                       getInt(fields, "hold_cnt", -1),
                                       getInt(fields, "vm_vsrc", -1), getInt(fields, "va_vcc", -1),
                                       getInt(fields, "sa_sdst", -1)));
    } else if (attrKey == "mod.dpp") {
        bool isDPP8 = getBool(fields, "isDPP8", false);
        if (isDPP8) {
            auto dpp8Vec = getIntVector(fields, "dpp8");
            std::array<uint8_t, 8> dpp8Perm = {0, 0, 0, 0, 0, 0, 0, 0};
            for (size_t i = 0; i < dpp8Vec.size() && i < 8; ++i)
                dpp8Perm[i] = static_cast<uint8_t>(dpp8Vec[i]);
            inst->addModifier(
                DPPModifiers(dpp8Perm, static_cast<uint8_t>(getInt(fields, "fi", 0))));
        } else {
            DppCtrl ctrl = static_cast<DppCtrl>(getInt(fields, "dppCtrl", 0xFFFF));
            inst->addModifier(DPPModifiers(ctrl,
                                           static_cast<uint8_t>(getInt(fields, "rowMask", 0xF)),
                                           static_cast<uint8_t>(getInt(fields, "bankMask", 0xF)),
                                           static_cast<uint8_t>(getInt(fields, "boundCtrl", 0)),
                                           static_cast<uint8_t>(getInt(fields, "fi", 0))));
        }
    } else if (attrKey == "mod.memtoken") {
        if (fields.contains("tokens")) {
            inst->addModifier(MemTokenData(getIntVector(fields, "tokens")));
        }
    } else if (attrKey == "mod.label") {
        inst->addModifier(LabelData(getStr(fields, "label", ""),
                                    static_cast<uint16_t>(getInt(fields, "alignment", 1))));
    }
    // mod.sdwa, mod.vop3p, mod.true16: no deserialize support yet
}

}  // namespace

void ModifierSerializer::deserialize(StinkyInstruction* inst, const ParsedModifierDict& modifiers) {
    for (const auto& [attrKey, fields] : modifiers) deserializeVisit(inst, attrKey, fields);
}

}  // namespace stinkytofu

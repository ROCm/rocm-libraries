/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
#include "stinkytofu/ir/asm/StinkySignature.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "stinkytofu/bindings/python/Module.hpp"

namespace stinkytofu {
/***************************************
 * Helper Functions
 ***************************************/
std::string isaVersionToGfx(const std::array<int, 3>& isa) {
    return "gfx" + std::to_string(isa[0]) + std::to_string(isa[1]) + std::to_string(isa[2]);
}

std::string toHex(uint32_t num) {
    std::stringstream ss;
    ss << std::hex << num;
    return ss.str();
}

std::string fieldDesc(const std::string& fieldName, int value, int bits) {
    std::string bitsStr = (bits > 0) ? " (" + std::to_string(bits) + "b)" : "";
    return fieldName + bitsStr + ": " + std::to_string(value);
}

/***************************************
 * SRD Upper Value 9XX (GFX9)
 ***************************************/
SrdUpperValue9XX SrdUpperValue9XX::staticInit() {
    SrdUpperValue9XX val;
    val.fields.data_format = 4;
    val.value = val.fields.value;
    return val;
}

std::string SrdUpperValue9XX::fieldsDesc() const {
    std::stringstream ss;
    ss << fieldDesc("dst_sel_x", fields.dst_sel_x, 3) << "\n"
       << fieldDesc("dst_sel_y", fields.dst_sel_y, 3) << "\n"
       << fieldDesc("dst_sel_z", fields.dst_sel_z, 3) << "\n"
       << fieldDesc("dst_sel_w", fields.dst_sel_w, 3) << "\n"
       << fieldDesc("num_format", fields.num_format, 3) << "\n"
       << fieldDesc("data_format", fields.data_format, 4) << "\n"
       << fieldDesc("user_vm_enable", fields.user_vm_enable, 1) << "\n"
       << fieldDesc("user_vm_mode", fields.user_vm_mode, 1) << "\n"
       << fieldDesc("index_stride", fields.index_stride, 2) << "\n"
       << fieldDesc("add_tid_enable", fields.add_tid_enable, 1) << "\n"
       << fieldDesc("_unusedA", fields._unusedA, 3) << "\n"
       << fieldDesc("nv", fields.nv, 1) << "\n"
       << fieldDesc("_unusedB", fields._unusedB, 2) << "\n"
       << fieldDesc("type", fields.type, 2);
    return ss.str();
}

std::string SrdUpperValue9XX::desc() const {
    return "hex: " + toString() + "\n" + fieldsDesc();
}

/***************************************
 * SRD Upper Value 10XX (GFX10)
 ***************************************/
SrdUpperValue10XX SrdUpperValue10XX::staticInit() {
    SrdUpperValue10XX val;
    val.fields.format = 4;
    val.fields.resource_level = 1;
    val.fields.oob_select = 3;
    val.value = val.fields.value;
    return val;
}

std::string SrdUpperValue10XX::fieldsDesc() const {
    std::stringstream ss;
    ss << fieldDesc("dst_sel_x", fields.dst_sel_x, 3) << "\n"
       << fieldDesc("dst_sel_y", fields.dst_sel_y, 3) << "\n"
       << fieldDesc("dst_sel_z", fields.dst_sel_z, 3) << "\n"
       << fieldDesc("dst_sel_w", fields.dst_sel_w, 3) << "\n"
       << fieldDesc("format", fields.format, 7) << "\n"
       << fieldDesc("_unusedA", fields._unusedA, 2) << "\n"
       << fieldDesc("index_stride", fields.index_stride, 2) << "\n"
       << fieldDesc("add_tid_enable", fields.add_tid_enable, 1) << "\n"
       << fieldDesc("resource_level", fields.resource_level, 1) << "\n"
       << fieldDesc("_unusedB", fields._unusedB, 1) << "\n"
       << fieldDesc("LLC_noalloc", fields.LLC_noalloc, 2) << "\n"
       << fieldDesc("oob_select", fields.oob_select, 2) << "\n"
       << fieldDesc("type", fields.type, 2);
    return ss.str();
}

std::string SrdUpperValue10XX::desc() const {
    return "hex: " + toString() + "\n" + fieldsDesc();
}

/***************************************
 * SRD Upper Value 11XX (GFX11)
 ***************************************/
SrdUpperValue11XX SrdUpperValue11XX::staticInit() {
    SrdUpperValue11XX val;
    val.fields.format = 4;
    val.fields.resource_level = 1;
    val.fields.oob_select = 3;
    val.value = val.fields.value;
    return val;
}

std::string SrdUpperValue11XX::fieldsDesc() const {
    std::stringstream ss;
    ss << fieldDesc("dst_sel_x", fields.dst_sel_x, 3) << "\n"
       << fieldDesc("dst_sel_y", fields.dst_sel_y, 3) << "\n"
       << fieldDesc("dst_sel_z", fields.dst_sel_z, 3) << "\n"
       << fieldDesc("dst_sel_w", fields.dst_sel_w, 3) << "\n"
       << fieldDesc("format", fields.format, 7) << "\n"
       << fieldDesc("_unusedA", fields._unusedA, 2) << "\n"
       << fieldDesc("index_stride", fields.index_stride, 2) << "\n"
       << fieldDesc("add_tid_enable", fields.add_tid_enable, 1) << "\n"
       << fieldDesc("resource_level", fields.resource_level, 1) << "\n"
       << fieldDesc("_unusedB", fields._unusedB, 1) << "\n"
       << fieldDesc("LLC_noalloc", fields.LLC_noalloc, 2) << "\n"
       << fieldDesc("oob_select", fields.oob_select, 2) << "\n"
       << fieldDesc("type", fields.type, 2);
    return ss.str();
}

std::string SrdUpperValue11XX::desc() const {
    return "hex: " + toString() + "\n" + fieldsDesc();
}

/***************************************
 * SRD Upper Value 12XX (GFX12)
 ***************************************/
SrdUpperValue12XX SrdUpperValue12XX::staticInit() {
    SrdUpperValue12XX val;
    val.fields.format = 32;
    val.fields.oob_select = 3;
    val.value = val.fields.value;
    return val;
}

std::string SrdUpperValue12XX::fieldsDesc() const {
    std::stringstream ss;
    ss << fieldDesc("dst_sel_x", fields.dst_sel_x, 3) << "\n"
       << fieldDesc("dst_sel_y", fields.dst_sel_y, 3) << "\n"
       << fieldDesc("dst_sel_z", fields.dst_sel_z, 3) << "\n"
       << fieldDesc("dst_sel_w", fields.dst_sel_w, 3) << "\n"
       << fieldDesc("format", fields.format, 7) << "\n"
       << fieldDesc("_unusedA", fields._unusedA, 2) << "\n"
       << fieldDesc("index_stride", fields.index_stride, 2) << "\n"
       << fieldDesc("add_tid_enable", fields.add_tid_enable, 1) << "\n"
       << fieldDesc("resource_level", fields.resource_level, 1) << "\n"
       << fieldDesc("_unusedB", fields._unusedB, 3) << "\n"
       << fieldDesc("oob_select", fields.oob_select, 2) << "\n"
       << fieldDesc("type", fields.type, 2);
    return ss.str();
}

std::string SrdUpperValue12XX::desc() const {
    return "hex: " + toString() + "\n" + fieldsDesc();
}

/***************************************
 * SRD Upper Value 125X (GFX1250)
 ***************************************/
SrdUpperValue125X SrdUpperValue125X::staticInit() {
    SrdUpperValue125X val;
    val.value = val.fields.value;
    return val;
}

std::string SrdUpperValue125X::fieldsDesc() const {
    std::stringstream ss;
    ss << fieldDesc("num_records_upper", fields.num_records_upper, 6) << "\n"
       << fieldDesc("reserved", fields.reserved, 6) << "\n"
       << fieldDesc("stride", fields.stride, 14) << "\n"
       << fieldDesc("stride_scale", fields.stride_scale, 2) << "\n"
       << fieldDesc("swizzle_enable", fields.swizzle_enable, 1) << "\n"
       << fieldDesc("oob_select", fields.oob_select, 1) << "\n"
       << fieldDesc("type", fields.type, 2);
    return ss.str();
}

std::string SrdUpperValue125X::desc() const {
    return "hex: " + toString() + "\n" + fieldsDesc();
}

/***************************************
 * Factory Function for SRD Values
 ***************************************/
std::shared_ptr<BitfieldUnion> createSrdUpperValue(const std::array<int, 3>& isa) {
    int major = isa[0];
    int minor = isa[1];

    if (major == 9) {
        auto val = std::make_shared<SrdUpperValue9XX>();
        *val = SrdUpperValue9XX::staticInit();
        return val;
    } else if (major == 10) {
        auto val = std::make_shared<SrdUpperValue10XX>();
        *val = SrdUpperValue10XX::staticInit();
        return val;
    } else if (major == 11) {
        auto val = std::make_shared<SrdUpperValue11XX>();
        *val = SrdUpperValue11XX::staticInit();
        return val;
    } else if (major == 12) {
        if (minor == 5) {
            auto val = std::make_shared<SrdUpperValue125X>();
            *val = SrdUpperValue125X::staticInit();
            return val;
        } else {
            auto val = std::make_shared<SrdUpperValue12XX>();
            *val = SrdUpperValue12XX::staticInit();
            return val;
        }
    }

    // Unsupported ISA - return nullptr or default to GFX9
    auto val = std::make_shared<SrdUpperValue9XX>();
    *val = SrdUpperValue9XX::staticInit();
    return val;
}

/***************************************
 * SignatureArgument
 ***************************************/
const std::unordered_map<std::string, int> SignatureArgument::ValueTypeSizeDict = {
    {"i8", 1},    {"u8", 1},    {"i16", 2},    {"u16", 2},   {"f16", 2},   {"i32", 4},
    {"u32", 4},   {"f32", 4},   {"f32c", 8},   {"i64", 8},   {"u64", 8},   {"f64", 8},
    {"f64c", 16}, {"pkf16", 4}, {"struct", 8}, {"void", 8},  {"FP8", 1},   {"fp8", 1},
    {"BF8", 1},   {"bf8", 1},   {"i8x4", 4},   {"u8x4", 4},  {"i16x2", 4}, {"u16x2", 4},
    {"f16x2", 4}, {"f32x2", 8}, {"f64x2", 16}, {"bf16", 2},  {"f8", 1},    {"b8", 1},
    {"b16", 2},   {"b32", 4},   {"b64", 8},    {"b128", 16}, {"half", 2},  {"float", 4},
    {"double", 8}};

SignatureArgument::SignatureArgument(int offset, const std::string& name,
                                     SignatureValueKind valueKind, const std::string& valueType,
                                     const std::string& addrSpaceQual)
    : valueKind(valueKind),
      valueType(valueType),
      name(name),
      offset(offset),
      size(valueToSize(valueKind, valueType)),
      addrSpaceQual(addrSpaceQual) {}

int SignatureArgument::valueToSize(SignatureValueKind valueKind,
                                   const std::string& valueType) const {
    if (valueKind == SignatureValueKind::SIG_GLOBALBUFFER) {
        return 8;  // Pointer size
    }

    auto it = ValueTypeSizeDict.find(valueType);
    if (it != ValueTypeSizeDict.end()) {
        return it->second;
    }

    // Unknown type - default to 4 bytes
    return 4;
}

std::string SignatureArgument::valueKindToStr() const {
    if (valueKind == SignatureValueKind::SIG_GLOBALBUFFER) {
        return "global_buffer";
    } else if (valueKind == SignatureValueKind::SIG_VALUE) {
        return "by_value";
    }

    // Unknown - default to by_value
    return "by_value";
}

std::string SignatureArgument::toString() const {
    std::string signatureIndent = "        ";
    std::string kStr;
    kStr += signatureIndent.substr(2) + "- .name:            " + name + "\n";
    kStr += signatureIndent + ".size:            " + std::to_string(size) + "\n";
    kStr += signatureIndent + ".offset:          " + std::to_string(offset) + "\n";
    kStr += signatureIndent + ".value_kind:      " + valueKindToStr() + "\n";
    kStr += signatureIndent + ".value_type:      " + valueType + "\n";
    if (!addrSpaceQual.empty()) {
        kStr += signatureIndent + ".address_space:   " + addrSpaceQual + "\n";
    }
    return kStr;
}

/***************************************
 * SignatureKernelDescriptor
 ***************************************/
SignatureKernelDescriptor::SignatureKernelDescriptor(
    const std::string& name, const std::array<int, 3>& isaVersion, int groupSegSize,
    const std::array<int, 3>& sgprWorkGroup, int vgprWorkItem, int wavefrontSize, int totalVgprs,
    int totalAgprs, int totalSgprs, bool preloadKernArgs)
    : kernelName(name),
      totalVgprs(totalVgprs),
      totalAgprs(totalAgprs),
      totalSgprs(totalSgprs),
      originalTotalVgprs(totalVgprs),
      accumOffset(-1),
      groupSegSize(groupSegSize),
      sgprWorkGroup(sgprWorkGroup),
      vgprWorkItem(vgprWorkItem),
      enablePreloadKernArgs(preloadKernArgs),
      isaVersion(isaVersion),
      wavefrontSize(wavefrontSize) {
    bool hasArchAccUnifiedRegs =
        (isaVersion[0] == 9 && isaVersion[1] == 0 && isaVersion[2] == 10) ||
        (isaVersion[0] == 9 && isaVersion[1] == 4 && isaVersion[2] >= 2) ||
        (isaVersion[0] == 9 && isaVersion[1] == 5 && isaVersion[2] == 0);
    bool hasUnifiedRegs = hasArchAccUnifiedRegs || (isaVersion[0] >= 10);

    if (hasArchAccUnifiedRegs || (hasUnifiedRegs && totalAgprs > 0)) {
        accumOffset = static_cast<int>(std::ceil(totalVgprs / 8.0) * 8);
        this->totalVgprs = accumOffset + totalAgprs;
    } else {
        accumOffset = -1;
        this->totalVgprs = totalVgprs;
    }
}

void SignatureKernelDescriptor::setGprs(int totalVgprs, int totalAgprs, int totalSgprs) {
    bool hasArchAccUnifiedRegs =
        (isaVersion[0] == 9 && isaVersion[1] == 0 && isaVersion[2] == 10) ||
        (isaVersion[0] == 9 && isaVersion[1] == 4 && isaVersion[2] >= 2) ||
        (isaVersion[0] == 9 && isaVersion[1] == 5 && isaVersion[2] == 0);
    bool hasUnifiedRegs = hasArchAccUnifiedRegs || (isaVersion[0] >= 10);

    if (hasArchAccUnifiedRegs || (hasUnifiedRegs && totalAgprs > 0)) {
        accumOffset = static_cast<int>(std::ceil(totalVgprs / 8.0) * 8);
        this->totalVgprs = accumOffset + totalAgprs;
    } else {
        accumOffset = -1;
        this->totalVgprs = std::max(totalAgprs, totalVgprs);
    }

    originalTotalVgprs = totalVgprs;
    this->totalAgprs = totalAgprs;
    this->totalSgprs = totalSgprs;
}

void SignatureKernelDescriptor::setOptimizationConfig(const std::array<int, 2>& tt,
                                                      const std::array<int, 2>& sg,
                                                      const std::array<int, 2>& wg, int vwA,
                                                      int vwB, int glvwA, int glvwB, bool d2lA,
                                                      bool d2lB, int useSgprForGRO) {
    this->threadTile = tt;
    this->subGroup = sg;
    this->waveGroup = wg;
    this->vectorWidthA = vwA;
    this->vectorWidthB = vwB;
    this->globalReadVectorWidthA = glvwA;
    this->globalReadVectorWidthB = glvwB;
    this->directToLdsA = d2lA;
    this->directToLdsB = d2lB;
    this->useSgprForGRO = useSgprForGRO;
}

std::string SignatureKernelDescriptor::toString() const {
    std::string kdIndent = "  ";
    std::string kStr;

    kStr += SignatureBase::block3Line("Begin Kernel");
    kStr += ".amdgcn_target \"amdgcn-amd-amdhsa--" + isaVersionToGfx(isaVersion) + "\"\n";
    kStr += ".text\n";
    kStr += ".protected " + kernelName + "\n";
    kStr += ".globl " + kernelName + "\n";
    kStr += ".p2align 8\n";
    kStr += ".type " + kernelName + ",@function\n";
    kStr += ".section .rodata,#alloc\n";
    kStr += ".p2align 6\n";
    kStr += ".amdhsa_kernel " + kernelName + "\n";
    kStr += kdIndent + ".amdhsa_user_sgpr_kernarg_segment_ptr 1\n";

    if (accumOffset != -1) {
        kStr += kdIndent + ".amdhsa_accum_offset " + std::to_string(accumOffset) +
                " // accvgpr offset\n";
    }

    kStr += kdIndent + ".amdhsa_next_free_vgpr " + std::to_string(totalVgprs) + " // vgprs\n";
    kStr += kdIndent + ".amdhsa_next_free_sgpr " + std::to_string(totalSgprs) + " // sgprs\n";
    kStr += kdIndent + ".amdhsa_group_segment_fixed_size " + std::to_string(groupSegSize) +
            " // lds bytes\n";

    // Check if architecture supports wave32
    bool hasWave32 = (isaVersion[0] >= 10);  // GFX10+
    if (hasWave32) {
        if (wavefrontSize == 32) {
            kStr += kdIndent + ".amdhsa_wavefront_size32 1 // 32-thread wavefronts\n";
        } else {
            kStr += kdIndent + ".amdhsa_wavefront_size32 0 // 64-thread wavefronts\n";
        }
    }

    kStr += kdIndent + ".amdhsa_private_segment_fixed_size 0\n";
    kStr +=
        kdIndent + ".amdhsa_system_sgpr_workgroup_id_x " + std::to_string(sgprWorkGroup[0]) + "\n";
    kStr +=
        kdIndent + ".amdhsa_system_sgpr_workgroup_id_y " + std::to_string(sgprWorkGroup[1]) + "\n";
    kStr +=
        kdIndent + ".amdhsa_system_sgpr_workgroup_id_z " + std::to_string(sgprWorkGroup[2]) + "\n";
    kStr += kdIndent + ".amdhsa_system_vgpr_workitem_id " + std::to_string(vgprWorkItem) + "\n";
    kStr += kdIndent + ".amdhsa_float_denorm_mode_32 3\n";
    kStr += kdIndent + ".amdhsa_float_denorm_mode_16_64 3\n";

    if (enablePreloadKernArgs) {
        int numWorkgroupSgpr = sgprWorkGroup[0] + sgprWorkGroup[1] + sgprWorkGroup[2];
        kStr +=
            kdIndent + ".amdhsa_user_sgpr_count " + std::to_string(16 - numWorkgroupSgpr) + "\n";
        kStr += kdIndent + ".amdhsa_user_sgpr_kernarg_preload_length " +
                std::to_string(14 - numWorkgroupSgpr) + "\n";
        kStr += kdIndent + ".amdhsa_user_sgpr_kernarg_preload_offset 0\n";
    }

    // Emit pass-through directives captured by RawAsmParser (.amdhsa_* lines
    // not modelled by the structured fields above). Each entry is already
    // formatted verbatim (with its source indentation) and includes no
    // trailing newline.
    for (const auto& d : extraKernelDirectives) {
        kStr += d;
        if (d.empty() || d.back() != '\n') kStr += "\n";
    }

    kStr += ".end_amdhsa_kernel\n";
    kStr += ".text\n";
    kStr += "/* Num VGPR   =" + std::to_string(originalTotalVgprs) + " */\n";
    kStr += "/* Num AccVGPR=" + std::to_string(totalAgprs) + " */\n";
    kStr += "/* Num SGPR   =" + std::to_string(totalSgprs) + " */\n";

    kStr += SignatureBase::block3Line("Optimizations and Config:");
    kStr += "/* ThreadTile= " + std::to_string(threadTile[0]) + " x " +
            std::to_string(threadTile[1]) + " */\n";
    kStr += "/* SubGroup= " + std::to_string(subGroup[0]) + " x " + std::to_string(subGroup[1]) +
            " */\n";
    kStr += "/* VectorWidthA=" + std::to_string(vectorWidthA) + " */\n";
    kStr += "/* VectorWidthB=" + std::to_string(vectorWidthB) + " */\n";
    kStr += "/* GlobalReadVectorWidthA=" + std::to_string(globalReadVectorWidthA) +
            ", GlobalReadVectorWidthB=" + std::to_string(globalReadVectorWidthB) + " */\n";
    kStr += "/* DirectToLdsA=" + std::string(directToLdsA ? "True" : "False") + " */\n";
    kStr += "/* DirectToLdsB=" + std::string(directToLdsB ? "True" : "False") + " */\n";
    kStr += "/* UseSgprForGRO=" + std::string(useSgprForGRO ? "True" : "False") + " */\n";

    return kStr;
}

/***************************************
 * SignatureCodeMeta
 ***************************************/
SignatureCodeMeta::SignatureCodeMeta(const std::string& name, int kernArgsVersion, int groupSegSize,
                                     int flatWgSize, int wavefrontSize,
                                     const std::string& codeObjectVersion, int totalVgprs,
                                     int totalSgprs)
    : kernelName(name),
      kernArgsVersion(kernArgsVersion),
      groupSegSize(groupSegSize),
      flatWgSize(flatWgSize),
      codeObjectVersion(codeObjectVersion),
      totalVgprs(totalVgprs),
      totalSgprs(totalSgprs),
      wavefrontSize(wavefrontSize),
      offset(0) {}

void SignatureCodeMeta::setGprs(int totalVgprs, int totalSgprs) {
    this->totalVgprs = totalVgprs;
    this->totalSgprs = totalSgprs;
}

void SignatureCodeMeta::addArg(const std::string& name, SignatureValueKind kind,
                               const std::string& type, const std::string& addrSpaceQual) {
    SignatureArgument sa(offset, name, kind, type, addrSpaceQual);
    argList.push_back(sa);
    offset += sa.size;
}

std::string SignatureCodeMeta::toString() const {
    std::string kStr;

    kStr += ".amdgpu_metadata\n";
    kStr += "---\n";
    kStr += "custom.config:\n";
    kStr += "  InternalSupportParams:\n";
    kStr += "    KernArgsVersion: " + std::to_string(kernArgsVersion) + "\n";
    kStr += "amdhsa.version:\n";
    kStr += "  - 1\n";

    if (codeObjectVersion == "4" || codeObjectVersion == "default") {
        kStr += "  - 1\n";
    } else if (codeObjectVersion == "5") {
        kStr += "  - 2\n";
    }

    kStr += "amdhsa.kernels:\n";
    kStr += "  - .name: " + kernelName + "\n";
    kStr += "    .symbol: '" + kernelName + ".kd'\n";
    kStr += "    .language:                   OpenCL C\n";
    kStr += "    .language_version:\n";
    kStr += "      - 2\n";
    kStr += "      - 0\n";
    kStr += "    .args:\n";

    for (const auto& arg : argList) {
        kStr += arg.toString();
    }

    kStr += "    .group_segment_fixed_size:   " + std::to_string(groupSegSize) + "\n";
    kStr += "    .kernarg_segment_align:      8\n";
    kStr += "    .kernarg_segment_size:       " + std::to_string(((offset + 7) / 8) * 8) + "\n";
    kStr += "    .max_flat_workgroup_size:    " + std::to_string(flatWgSize) + "\n";
    kStr += "    .private_segment_fixed_size: 0\n";
    kStr += "    .sgpr_count:                 " + std::to_string(totalSgprs) + "\n";
    kStr += "    .sgpr_spill_count:           0\n";
    kStr += "    .vgpr_count:                 " + std::to_string(totalVgprs) + "\n";
    kStr += "    .vgpr_spill_count:           0\n";
    kStr += "    .wavefront_size:             " + std::to_string(wavefrontSize) + "\n";
    kStr += "...\n";
    kStr += ".end_amdgpu_metadata\n";
    kStr += kernelName + ":\n";

    return kStr;
}

/***************************************
 * SignatureBase
 ***************************************/
SignatureBase::SignatureBase(const std::string& kernelName, const std::array<int, 3>& isaVersion,
                             int kernArgsVersion, const std::string& codeObjectVersion,
                             int groupSegmentSize, const std::array<int, 3>& sgprWorkGroup,
                             int vgprWorkItem, int flatWorkGroupSize, int wavefrontSize,
                             int totalVgprs, int totalAgprs, int totalSgprs, bool preloadKernArgs)
    : kernelDescriptor(kernelName, isaVersion, groupSegmentSize, sgprWorkGroup, vgprWorkItem,
                       wavefrontSize, totalVgprs, totalAgprs, totalSgprs, preloadKernArgs),
      codeMeta(kernelName, kernArgsVersion, groupSegmentSize, flatWorkGroupSize, wavefrontSize,
               codeObjectVersion, totalVgprs, totalSgprs),
      descriptionTopic("") {}

void SignatureBase::setGprs(int totalVgprs, int totalAgprs, int totalSgprs) {
    kernelDescriptor.setGprs(totalVgprs, totalAgprs, totalSgprs);
    codeMeta.setGprs(totalVgprs, totalSgprs);
}

void SignatureBase::addArg(const std::string& name, SignatureValueKind kind,
                           const std::string& type, const std::string& addrSpaceQual) {
    codeMeta.addArg(name, kind, type, addrSpaceQual);
}

void SignatureBase::addDescriptionTopic(const std::string& text) {
    descriptionTopic = block3Line(text);
}

void SignatureBase::addDescriptionBlock(const std::string& text) {
    descriptionList.push_back(block(text));
}

void SignatureBase::addDescription(const std::string& text) {
    descriptionList.push_back(slash(text));
}

void SignatureBase::clearDescription() {
    descriptionList.clear();
    descriptionTopic = "";
}

void SignatureBase::setOptimizationConfig(const std::array<int, 2>& tt,
                                          const std::array<int, 2>& sg,
                                          const std::array<int, 2>& wg, int vwA, int vwB, int glvwA,
                                          int glvwB, bool d2lA, bool d2lB, int useSgprForGRO) {
    kernelDescriptor.setOptimizationConfig(tt, sg, wg, vwA, vwB, glvwA, glvwB, d2lA, d2lB,
                                           useSgprForGRO);
}

std::string SignatureBase::toString() const {
    std::string kStr;
    kStr += kernelDescriptor.toString();

    if (!descriptionTopic.empty()) {
        kStr += descriptionTopic;
    }

    for (const auto& desc : descriptionList) {
        kStr += desc;
    }

    kStr += codeMeta.toString();
    return kStr;
}

std::string SignatureBase::block(const std::string& text) {
    return "/* " + text + " */\n";
}

std::string SignatureBase::block3Line(const std::string& text) {
    std::ostringstream oss;
    oss << "\n/******************************************/\n";
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line)) {
        oss << "/* " << std::setw(38) << std::left << line << " */\n";
    }
    oss << "/******************************************/\n";
    return oss.str();
}

std::string SignatureBase::slash(const std::string& text) {
    return "// " + text + "\n";
}

/***************************************
 * KernelBody
 ***************************************/
KernelBody::KernelBody(const std::string& name)
    : name(name), totalVgprs(0), totalAgprs(0), totalSgprs(0) {}

void KernelBody::addSignature(const std::shared_ptr<SignatureBase>& signature) {
    this->signature = signature;
}

void KernelBody::addBody(const std::shared_ptr<IRListModule>& body) {
    this->body = body;
}

void KernelBody::setGprs(int totalVgprs, int totalAgprs, int totalSgprs) {
    this->totalVgprs = totalVgprs;
    this->totalAgprs = totalAgprs;
    this->totalSgprs = totalSgprs;

    if (signature) {
        signature->setGprs(totalVgprs, totalAgprs, totalSgprs);
    }
}

int KernelBody::getNextFreeVgpr() const {
    return totalVgprs;
}

int KernelBody::getNextFreeSgpr() const {
    return totalSgprs;
}

std::string KernelBody::getName() const {
    return name;
}

std::shared_ptr<SignatureBase> KernelBody::getSignature() const {
    return signature;
}

std::shared_ptr<IRListModule> KernelBody::getBody() const {
    return body;
}

std::string KernelBody::toString(bool emitComments, bool emitCycleInfo) const {
    std::string result;

    // Add signature if present
    if (signature) {
        result += signature->toString();
    }

    // Add instruction body if present
    if (body) {
        // Note: StinkyAsmModule::emitAssembly() uses internal defaults for
        // emitComments/emitCycleInfo The parameters to KernelBody::toString() are currently ignored
        result += body->emitAssembly();
    }

    return result;
}

}  // namespace stinkytofu

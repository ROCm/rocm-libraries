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

#include "stinkytofu/serialization/asm/RawAsmParser.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

#include "IRLexer.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"

#define DEBUG_TYPE "RawAsmParser"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"
#include "stinkytofu/ir/asm/StinkySignature.hpp"

namespace stinkytofu {
namespace {

//----------------------------------------------------------------------
// Symbol table: evaluates .set expressions to integer values
//----------------------------------------------------------------------

using SymbolTable = std::unordered_map<std::string, int>;

/// Evaluate a simple arithmetic expression composed of symbol names, integers,
/// and +/- operators (no precedence beyond left-to-right, no parens needed for
/// the patterns TensileLite emits). Returns nullopt if any token is unresolvable.
std::optional<int> evalExpr(const std::string& expr, const SymbolTable& syms) {
    // Tokenize on +/- boundaries (keeping sign attached to numbers)
    int result = 0;
    int sign = 1;
    size_t i = 0;
    size_t n = expr.size();

    auto skipWS = [&]() {
        while (i < n && (expr[i] == ' ' || expr[i] == '\t')) ++i;
    };

    while (i < n) {
        skipWS();
        if (i >= n) break;

        // Optional leading sign for first token is handled by sign = 1/−1
        if (expr[i] == '+') {
            sign = 1;
            ++i;
            skipWS();
        } else if (expr[i] == '-') {
            sign = -1;
            ++i;
            skipWS();
        }

        if (i >= n) return std::nullopt;

        // Read token: digits or identifier characters
        size_t start = i;
        if (std::isdigit(static_cast<unsigned char>(expr[i]))) {
            while (i < n && std::isdigit(static_cast<unsigned char>(expr[i]))) ++i;
            int val = std::stoi(expr.substr(start, i - start));
            result += sign * val;
        } else if (std::isalpha(static_cast<unsigned char>(expr[i])) || expr[i] == '_') {
            while (i < n && (std::isalnum(static_cast<unsigned char>(expr[i])) || expr[i] == '_'))
                ++i;
            std::string name = expr.substr(start, i - start);
            auto it = syms.find(name);
            if (it == syms.end()) return std::nullopt;
            result += sign * it->second;
        } else {
            return std::nullopt;
        }
        sign = 1;  // default next sign is positive unless we see +/-
        skipWS();
        // Next must be + or - or end
        if (i < n && expr[i] != '+' && expr[i] != '-') return std::nullopt;
    }
    return result;
}

/// Trim leading and trailing whitespace from a string in-place.
static void trimStr(std::string& s) {
    size_t f = s.find_first_not_of(" \t\r\n");
    if (f == std::string::npos) {
        s.clear();
        return;
    }
    s = s.substr(f, s.find_last_not_of(" \t\r\n") - f + 1);
}

/// Parse and add a .set directive to the symbol table.
/// Processed in source order (like C++ variable assignment): if evalExpr fails
/// (e.g. value is UNDEF), the previous value is left unchanged.
void processSetDirective(const std::string& line, SymbolTable& syms) {
    // line starts with ".set"
    std::string rest = line.substr(4);
    size_t comma = rest.find(',');
    if (comma == std::string::npos) return;

    std::string name = rest.substr(0, comma);
    std::string valStr = rest.substr(comma + 1);
    trimStr(name);
    trimStr(valStr);

    auto val = evalExpr(valStr, syms);
    if (val) syms[name] = *val;
}

//----------------------------------------------------------------------
// Register parsing helpers (adapted from IRParser.cpp)
//----------------------------------------------------------------------

/// Safely convert string_view to int. Returns nullopt on failure.
inline std::optional<int> safeAtoiView(std::string_view sv) {
    if (sv.empty()) return std::nullopt;
    long long result = 0;
    bool negative = false;
    size_t start = 0;
    if (sv[0] == '-') {
        negative = true;
        start = 1;
    }
    if (start >= sv.size()) return std::nullopt;
    for (size_t i = start; i < sv.size(); ++i) {
        if (sv[i] < '0' || sv[i] > '9') return std::nullopt;
        result = result * 10 + (sv[i] - '0');
        if (result > (long long)INT_MAX + 1) return std::nullopt;
    }
    if (negative) result = -result;
    if (result < INT_MIN || result > INT_MAX) return std::nullopt;
    return static_cast<int>(result);
}

inline std::optional<int> safeAtoiStr(const std::string& s) {
    return safeAtoiView(std::string_view(s));
}

/// Returns true if the identifier looks like a hex suffix "xABCD" / "XABCD"
inline bool isHexSuffix(std::string_view id) {
    if (id.size() < 2) return false;
    if (id[0] != 'x' && id[0] != 'X') return false;
    for (size_t i = 1; i < id.size(); ++i)
        if (!std::isxdigit(static_cast<unsigned char>(id[i]))) return false;
    return true;
}

/// Parse a single register operand from the lexer.
/// Adapted from IRParser::parseRegister().
std::optional<StinkyRegister> parseOneRegister(IRLexer& lexer, const SymbolTable& syms) {
    TokenKind kind = lexer.peek().kind;

    // Hex immediate: preserve as literal string
    if (kind == TokenKind::HexLiteral) {
        const Token& tok = lexer.consume();
        return StinkyRegister(std::string(tok.text));
    }

    // Integer literal: numeric immediate
    if (kind == TokenKind::IntegerLiteral) {
        const Token& tok = lexer.consume();
        std::string num(tok.text);
        // Recover split "0x..." tokens (lexer may split "0" + "xABCD")
        if (num == "0" && lexer.peek().kind == TokenKind::Identifier) {
            std::string id(lexer.peek().text);
            if (isHexSuffix(id)) {
                lexer.consume();
                return StinkyRegister(std::string("0") + id);
            }
        }
        auto val = safeAtoiStr(num);
        if (!val) return std::nullopt;
        return StinkyRegister(*val);
    }

    // Float literal
    if (kind == TokenKind::FloatLiteral) {
        const Token& tok = lexer.consume();
        double val = std::stod(std::string(tok.text));
        return StinkyRegister(val);
    }

    // Must be Identifier
    if (kind != TokenKind::Identifier) return std::nullopt;

    const Token& regTypeTok = lexer.consume();
    std::string regTypeStr(regTypeTok.text);

    // label_* references (branch targets embedded as operands)
    if (regTypeStr.size() >= 6 && regTypeStr.substr(0, 6) == "label_")
        return StinkyRegister(regTypeStr);
    if (regTypeStr == "BufferLimit") return StinkyRegister(regTypeStr);
    // "null" is used as a no-op resource descriptor in buffer instructions
    if (regTypeStr == "null") return StinkyRegister(regTypeStr);

    // Handle compact "v10" / "s5" / "acc12" etc. (type + digits glued together)
    for (size_t prefixLen = regTypeStr.size(); prefixLen >= 1; --prefixLen) {
        std::string prefix = regTypeStr.substr(0, prefixLen);
        std::string suffix = regTypeStr.substr(prefixLen);
        RegType rt = stringToRegType(prefix);
        if (isValidRegType(rt) && !suffix.empty() &&
            std::all_of(suffix.begin(), suffix.end(),
                        [](unsigned char c) { return std::isdigit(c); })) {
            auto idx = safeAtoiStr(suffix);
            if (idx) return StinkyRegister(rt, *idx, 1);
        }
    }

    RegType regType = stringToRegType(regTypeStr);

    // Unknown type → not a register, signal caller to stop operand parsing
    if (!isValidRegType(regType)) return std::nullopt;

    // Format: "v 12" (type and index as separate tokens)
    if (lexer.peek().kind == TokenKind::IntegerLiteral) {
        const Token& idxTok = lexer.consume();
        auto idx = safeAtoiStr(std::string(idxTok.text));
        if (!idx) return std::nullopt;
        return StinkyRegister(regType, *idx, 1);
    }

    // Format: "v[12]" or "v[10:13]"  or  "v[sym+expr:sym+expr]" (symbolic)
    if (lexer.peek().kind != TokenKind::LeftBracket)
        return StinkyRegister(regTypeStr);  // Plain identifier, treat as string literal

    lexer.consume();  // consume '['

    // Collect the raw content inside brackets (up to the matching ']').
    // Build the expression string token by token so we can try to evaluate it.
    std::string bracketContent;
    int depth = 1;
    while (!lexer.isAtEnd() && lexer.peek().kind != TokenKind::Eof) {
        const Token& t = lexer.consume();
        if (t.kind == TokenKind::LeftBracket) {
            ++depth;
            bracketContent += '[';
        } else if (t.kind == TokenKind::RightBracket) {
            --depth;
            if (depth == 0) break;
            bracketContent += ']';
        } else {
            bracketContent += std::string(t.text);
        }
    }

    // Split on ':' to get start and optional end expressions.
    size_t colon = bracketContent.find(':');
    std::string startExprStr =
        (colon != std::string::npos) ? bracketContent.substr(0, colon) : bracketContent;
    std::string endExprStr =
        (colon != std::string::npos) ? bracketContent.substr(colon + 1) : startExprStr;

    // Try to evaluate both expressions using the symbol table.
    auto startIdx = evalExpr(startExprStr, syms);
    auto endIdx = evalExpr(endExprStr, syms);

    if (startIdx && endIdx && *endIdx >= *startIdx) {
        int si = *startIdx;
        int ei = *endIdx;
        int16_t offset = 0;

        // Handle MSB-bank addressing (InsertVgprMsbPass reverse):
        // The pass emits v[physIdx + msb*(-256)], so raw asm may have negative indices.
        // Reverse: actual_physIdx = si + msb*256, offset = msb*(-256).
        if (si < 0 && regType == RegType::V) {
            // Find the smallest msb such that si + msb*256 >= 0
            int msb = (-si + 255) / 256;
            si += msb * 256;
            ei += msb * 256;
            offset = static_cast<int16_t>(msb * (-256));
        }

        if (si >= 0 && ei >= si) {
            int regNum = ei - si + 1;
            // Store the symbolic name alongside the numeric index (matches rocisa flow).
            StinkyRegister reg(regType, static_cast<uint32_t>(si), static_cast<uint16_t>(regNum),
                               offset);
            reg.setSymbolicName(regTypeStr + "[" + bracketContent + "]");
            return reg;
        }
    }

    // Could not resolve — store as LiteralString so at least the instruction
    // is a StinkyInstruction (not a TEXTBLOCK).
    return StinkyRegister(regTypeStr + "[" + bracketContent + "]");
}

//----------------------------------------------------------------------
// Modifier parsing
//----------------------------------------------------------------------

/// Parse trailing modifier tokens into inst.modifiers.
/// modifier_key is determined by hwInstDesc->microcode and the instruction mnemonic.
void parseModifiers(IRLexer& lexer, ParsedInstruction& inst, const HwInstDesc* hwInstDesc) {
    using FieldMap = std::unordered_map<std::string, std::string>;

    const std::string& mnemonic = inst.opcodeStr;
    bool isWaitcnt = (mnemonic == "s_waitcnt");
    bool isDelayAlu = (mnemonic == "s_delay_alu");

    // Determine modifier namespace from microcode format
    std::string modKey;
    switch (hwInstDesc->microcode) {
        case MicrocodeFormat::MC_VDS:
            modKey = "mod.ds";
            break;
        case MicrocodeFormat::MC_VBUFFER:
            modKey = "mod.mubuf";
            break;
        case MicrocodeFormat::MC_VFLAT:
            modKey = "mod.flat";
            break;
        case MicrocodeFormat::MC_VGLOBAL:
            modKey = "mod.global";
            break;
        case MicrocodeFormat::MC_VSCRATCH:
            modKey = "mod.flat";
            break;
        case MicrocodeFormat::MC_SMEM:
            modKey = "mod.smem";
            break;
        default:
            break;
    }

    // s_waitcnt shorthand: "s_waitcnt 0" means all counters zeroed
    if (isWaitcnt && !lexer.isAtEnd() && lexer.peek().kind == TokenKind::IntegerLiteral) {
        std::string_view val = lexer.peek().text;
        if (val == "0") {
            lexer.consume();
            inst.modifiers["mod.swaitcnt"]["vlcnt"] = "0";
            inst.modifiers["mod.swaitcnt"]["dscnt"] = "0";
            return;
        }
    }

    // s_delay_alu: store the entire remainder as a TEXTBLOCK (complex syntax)
    // The instruction will still be created with correct timing; modifiers won't
    // be decoded but that's acceptable for a round-trip parser.
    if (isDelayAlu) {
        // Leave modifiers empty; the passes that need delay_alu data (e.g. wait-cnt
        // insertion) will re-insert the correct modifier after scheduling.
        return;
    }

    // Collect key→value / key(value) / boolean-flag tokens
    FieldMap fields;

    while (!lexer.isAtEnd() && lexer.peek().kind != TokenKind::Eof &&
           lexer.peek().kind != TokenKind::Newline) {
        // Skip separators: commas and pipes (used by s_delay_alu / s_wait_alu)
        if (lexer.peek().kind == TokenKind::Comma) {
            lexer.consume();
            continue;
        }

        if (lexer.peek().kind != TokenKind::Identifier) {
            lexer.consume();  // skip unknown token
            continue;
        }

        std::string tok(lexer.consume().text);

        if (lexer.peek().kind == TokenKind::Colon) {
            // key:value format (e.g. offset:128)
            lexer.consume();  // eat ':'
            TokenKind vk = lexer.peek().kind;
            if (vk == TokenKind::IntegerLiteral || vk == TokenKind::HexLiteral) {
                fields[tok] = std::string(lexer.consume().text);
            }
        } else if (lexer.peek().kind == TokenKind::LeftParen) {
            // key(value) format (e.g. vmcnt(0), lgkmcnt(3))
            lexer.consume();  // eat '('
            std::string val;
            if (lexer.peek().kind == TokenKind::IntegerLiteral)
                val = std::string(lexer.consume().text);
            if (lexer.peek().kind == TokenKind::RightParen) lexer.consume();

            if (isWaitcnt) {
                // Map hardware wait-count tokens to semantic swaitcnt fields.
                // The emitter reconstructs lgkmcnt = dscnt + kmcnt and vmcnt = vlcnt + vscnt,
                // so mapping lgkmcnt → dscnt and vmcnt → vlcnt preserves the round-trip.
                if (tok == "lgkmcnt")
                    inst.modifiers["mod.swaitcnt"]["dscnt"] = val;
                else if (tok == "vmcnt")
                    inst.modifiers["mod.swaitcnt"]["vlcnt"] = val;
                else if (tok == "vscnt")
                    inst.modifiers["mod.swaitcnt"]["vscnt"] = val;
                else if (tok == "kmcnt")
                    inst.modifiers["mod.swaitcnt"]["kmcnt"] = val;
            }
            // Other key(value) tokens (e.g. depctr_*) are currently ignored.
        } else {
            // Boolean flag: glc, slc, nt, lds, gds, offen, nv, etc.
            fields[tok] = "true";
        }
    }

    if (modKey.empty() || fields.empty()) return;

    // Map generic fields → modifier dict using the appropriate namespace key.
    auto& modFields = inst.modifiers[modKey];

    if (modKey == "mod.ds") {
        if (fields.count("offset")) {
            modFields["na"] = "1";
            modFields["offset"] = fields["offset"];
        } else if (fields.count("offset0") || fields.count("offset1")) {
            modFields["na"] = "2";
            modFields["offset0"] = fields.count("offset0") ? fields["offset0"] : "0";
            modFields["offset1"] = fields.count("offset1") ? fields["offset1"] : "0";
        }
        if (fields.count("gds")) modFields["gds"] = "true";

    } else if (modKey == "mod.mubuf") {
        if (fields.count("offen")) modFields["offen"] = "true";
        // "offen offset:N" emits both as a single unit; "offset" key carries the value.
        if (fields.count("offset")) modFields["offset12"] = fields["offset"];
        if (fields.count("glc") || fields.count("sc0")) modFields["glc"] = "true";
        if (fields.count("slc") || fields.count("sc1")) modFields["slc"] = "true";
        if (fields.count("nt")) modFields["nt"] = "true";
        if (fields.count("lds")) modFields["lds"] = "true";

    } else if (modKey == "mod.flat") {
        if (fields.count("offset")) modFields["offset12"] = fields["offset"];
        if (fields.count("glc") || fields.count("sc0")) modFields["glc"] = "true";
        if (fields.count("slc") || fields.count("sc1")) modFields["slc"] = "true";
        if (fields.count("lds")) modFields["lds"] = "true";

    } else if (modKey == "mod.global") {
        if (fields.count("offset")) modFields["offset"] = fields["offset"];

    } else if (modKey == "mod.smem") {
        if (fields.count("offset")) modFields["offset"] = fields["offset"];
        if (fields.count("glc")) modFields["glc"] = "true";
        if (fields.count("nv")) modFields["nv"] = "true";
    }
}

//----------------------------------------------------------------------
// Line-level instruction parser
//----------------------------------------------------------------------

/// Parse one non-directive, non-label line into a ParsedInstruction.
/// Returns nullptr for unknown mnemonics (caller stores as TEXTBLOCK).
std::unique_ptr<ParsedInstruction> parseInstLine(const std::string& line, GfxArchID arch,
                                                 std::vector<Diagnostic>& diags, unsigned lineNo,
                                                 const SymbolTable& syms) {
    IRLexer lexer(line);
    lexer.lex();

    if (lexer.isAtEnd() || lexer.peek().kind == TokenKind::Eof) return nullptr;
    if (lexer.peek().kind != TokenKind::Identifier) return nullptr;

    std::string mnemonic(lexer.consume().text);

    // Look up the hardware instruction descriptor
    auto isaOpcode = getMnemonicToIsaOpcode(mnemonic, arch);
    const HwInstDesc* hwInstDesc = (isaOpcode != GFX::INVALID)
                                       ? getMCIDByIsaOp(static_cast<IsaOpcode>(isaOpcode), arch)
                                       : nullptr;

    if (!hwInstDesc) {
        diags.emplace_back(Diagnostic::Level::Warning,
                           "Unknown mnemonic '" + mnemonic + "'; storing as text", lineNo, 1);
        return nullptr;
    }

    auto inst = std::make_unique<ParsedInstruction>(mnemonic);
    inst->issueCycles = hwInstDesc->issue;
    inst->latencyCycles = hwInstDesc->latency;

    // Count dest/src split from operandFields.
    // When operandFields is empty the instruction has no explicit register operands
    // (e.g. s_endpgm, s_waitcnt, s_delay_alu) so skip register parsing entirely.
    int numDest = 0;
    bool hasOperandFields = !hwInstDesc->operandFields.empty();
    if (hasOperandFields) {
        for (const auto& f : hwInstDesc->operandFields)
            if (f.isDest) numDest++;
    }

    if (hasOperandFields) {
        // Parse comma-separated register operands.
        // Stop when: (a) no comma follows, or (b) next token is not a register.
        int opIdx = 0;
        bool firstOp = true;

        while (!lexer.isAtEnd() && lexer.peek().kind != TokenKind::Eof &&
               lexer.peek().kind != TokenKind::Newline) {
            if (!firstOp) {
                if (lexer.peek().kind != TokenKind::Comma) break;
                lexer.consume();  // eat ','
            }
            firstOp = false;

            auto reg = parseOneRegister(lexer, syms);
            if (!reg) {
                if (opIdx == 0) {
                    // First operand failed (e.g. symbolic expression like v[sym-768:sym-765]).
                    // parseOneRegister may have consumed tokens; safest to preserve the
                    // entire line verbatim.
                    return nullptr;
                }
                // Later operand failed → stop operand parsing, proceed to modifiers.
                break;
            }

            if (opIdx < numDest)
                inst->destRegs.push_back(*reg);
            else
                inst->srcRegs.push_back(*reg);
            opIdx++;
        }
    }

    // Parse any trailing modifier tokens (offset:N, glc, vmcnt(N), etc.)
    parseModifiers(lexer, *inst, hwInstDesc);

    return inst;
}

//----------------------------------------------------------------------
// Text-block directive helper
//----------------------------------------------------------------------

/// Wrap a raw text line as an asm_directive TEXTBLOCK ParsedInstruction.
/// The converter (IRConverter.cpp) will turn this into an AsmDirective with
/// kind=TEXTBLOCK and value=rawLine, preserving it verbatim in the output.
std::unique_ptr<ParsedInstruction> makeTextBlock(const std::string& rawLine) {
    auto inst = std::make_unique<ParsedInstruction>("asm_directive");
    inst->srcRegs.push_back(StinkyRegister(std::string("TEXTBLOCK")));
    inst->srcRegs.push_back(StinkyRegister(rawLine + "\n"));
    return inst;
}

//----------------------------------------------------------------------
// Kernel metadata parser
//----------------------------------------------------------------------

/// Strip a trailing // comment from a line.
static std::string stripComment(const std::string& line) {
    size_t pos = line.find("//");
    return (pos != std::string::npos) ? line.substr(0, pos) : line;
}

/// Parse one integer value from a directive line like "  .amdhsa_next_free_vgpr 1022 // vgprs"
/// Returns nullopt if the integer cannot be found.
static std::optional<int> parseDirectiveInt(const std::string& line, const std::string& key) {
    size_t kpos = line.find(key);
    if (kpos == std::string::npos) return std::nullopt;
    std::string rest = line.substr(kpos + key.size());
    rest = stripComment(rest);
    trimStr(rest);
    if (rest.empty()) return std::nullopt;
    auto v = safeAtoiStr(rest);
    return v ? std::optional<int>(*v) : std::nullopt;
}

/// Determine the isaVersion array from the arch ID.
static std::array<int, 3> archToIsaVersion(GfxArchID arch) {
    if (arch == GfxArchID::Gfx1250) return {12, 5, 0};
    return {12, 5, 0};
}

/// Parse the .amdhsa_kernel and .amdgpu_metadata sections from raw assembly text
/// and build a SignatureBase. Returns nullptr if no recognisable metadata is found.
std::shared_ptr<SignatureBase> parseKernelMetadata(const std::string& asmText, GfxArchID arch) {
    // Fields we need to extract
    std::string kernelName;
    int totalVgprs = 0;
    int totalSgprs = 0;
    int groupSegSize = 0;
    int wavefrontSize = 64;
    int flatWgSize = 64;
    std::array<int, 3> sgprWorkGroup = {1, 1, 1};
    int vgprWorkItem = 0;
    int kernArgsVersion = 2;
    std::string codeObjectVersion = "COV4";

    struct ArgInfo {
        std::string name;
        std::string valueKind;  // "by_value" or "global_buffer"
        std::string valueType;
        std::string addrSpace;
    };
    std::vector<ArgInfo> args;

    std::istringstream ss(asmText);
    std::string line;

    enum class Section { None, AmdhsaKernel, AmdgpuMetadata };
    Section section = Section::None;
    bool inArgEntry = false;
    ArgInfo curArg;

    while (std::getline(ss, line)) {
        std::string trimmed = line;
        trimStr(trimmed);

        // ── .amdhsa_kernel <name> ────────────────────────────────────────────────
        if (trimmed.substr(0, 14) == ".amdhsa_kernel" &&
            trimmed.find(".end_amdhsa_kernel") == std::string::npos) {
            // Extract kernel name (rest of the line after the directive)
            std::string rest = trimmed.substr(14);
            trimStr(rest);
            if (!rest.empty()) kernelName = rest;
            section = Section::AmdhsaKernel;
            continue;
        }
        if (trimmed == ".end_amdhsa_kernel") {
            section = Section::None;
            continue;
        }

        // ── Fields inside .amdhsa_kernel block ───────────────────────────────────
        if (section == Section::AmdhsaKernel) {
            if (auto v = parseDirectiveInt(trimmed, ".amdhsa_next_free_vgpr")) totalVgprs = *v;
            if (auto v = parseDirectiveInt(trimmed, ".amdhsa_next_free_sgpr")) totalSgprs = *v;
            if (auto v = parseDirectiveInt(trimmed, ".amdhsa_group_segment_fixed_size"))
                groupSegSize = *v;
            if (trimmed.find(".amdhsa_wavefront_size32") != std::string::npos) {
                auto v = parseDirectiveInt(trimmed, ".amdhsa_wavefront_size32");
                wavefrontSize = (v && *v == 1) ? 32 : 64;
            }
            if (trimmed.find(".amdhsa_system_sgpr_workgroup_id_x") != std::string::npos)
                if (auto v = parseDirectiveInt(trimmed, ".amdhsa_system_sgpr_workgroup_id_x"))
                    sgprWorkGroup[0] = *v;
            if (trimmed.find(".amdhsa_system_sgpr_workgroup_id_y") != std::string::npos)
                if (auto v = parseDirectiveInt(trimmed, ".amdhsa_system_sgpr_workgroup_id_y"))
                    sgprWorkGroup[1] = *v;
            if (trimmed.find(".amdhsa_system_sgpr_workgroup_id_z") != std::string::npos)
                if (auto v = parseDirectiveInt(trimmed, ".amdhsa_system_sgpr_workgroup_id_z"))
                    sgprWorkGroup[2] = *v;
            if (trimmed.find(".amdhsa_system_vgpr_workitem_id") != std::string::npos)
                if (auto v = parseDirectiveInt(trimmed, ".amdhsa_system_vgpr_workitem_id"))
                    vgprWorkItem = *v;
            continue;
        }

        // ── .amdgpu_metadata block ───────────────────────────────────────────────
        if (trimmed == ".amdgpu_metadata") {
            section = Section::AmdgpuMetadata;
            continue;
        }
        if (trimmed == ".end_amdgpu_metadata") {
            // Commit the last arg if any
            if (inArgEntry && !curArg.name.empty()) {
                args.push_back(curArg);
                inArgEntry = false;
            }
            section = Section::None;
            continue;
        }

        if (section != Section::AmdgpuMetadata) continue;

        // ── YAML fields inside .amdgpu_metadata ─────────────────────────────────
        // KernArgsVersion
        if (trimmed.find("KernArgsVersion:") != std::string::npos) {
            auto v = parseDirectiveInt(trimmed, "KernArgsVersion:");
            if (v) kernArgsVersion = *v;
        }
        // .max_flat_workgroup_size
        if (trimmed.find(".max_flat_workgroup_size:") != std::string::npos) {
            auto v = parseDirectiveInt(trimmed, ".max_flat_workgroup_size:");
            if (v) flatWgSize = *v;
        }

        // ── Arg list parsing (simple YAML block) ─────────────────────────────────
        // Each arg starts with "- .name:"
        if (trimmed.substr(0, 8) == "- .name:") {
            if (inArgEntry && !curArg.name.empty()) {
                args.push_back(curArg);
            }
            curArg = ArgInfo{};
            std::string rest = trimmed.substr(8);
            trimStr(rest);
            curArg.name = rest;
            inArgEntry = true;
            continue;
        }
        if (inArgEntry) {
            if (trimmed.find(".value_kind:") != std::string::npos) {
                size_t p = trimmed.find(".value_kind:");
                curArg.valueKind = trimmed.substr(p + 12);
                trimStr(curArg.valueKind);
            } else if (trimmed.find(".value_type:") != std::string::npos) {
                size_t p = trimmed.find(".value_type:");
                curArg.valueType = trimmed.substr(p + 12);
                trimStr(curArg.valueType);
            } else if (trimmed.find(".address_space:") != std::string::npos) {
                size_t p = trimmed.find(".address_space:");
                curArg.addrSpace = trimmed.substr(p + 15);
                trimStr(curArg.addrSpace);
            }
        }
    }

    if (kernelName.empty()) return nullptr;

    auto isaVersion = archToIsaVersion(arch);

    auto sig = std::make_shared<SignatureBase>(
        kernelName, isaVersion, kernArgsVersion, codeObjectVersion, groupSegSize, sgprWorkGroup,
        vgprWorkItem, flatWgSize, wavefrontSize, totalVgprs, /*totalAgprs=*/0, totalSgprs);

    // Map YAML value_kind → SignatureValueKind and add args
    for (const auto& a : args) {
        SignatureValueKind kind = SignatureValueKind::SIG_VALUE;
        if (a.valueKind == "global_buffer") kind = SignatureValueKind::SIG_GLOBALBUFFER;
        sig->addArg(a.name, kind, a.valueType, a.addrSpace);
    }

    return sig;
}

}  // namespace

//----------------------------------------------------------------------
// Public API
//----------------------------------------------------------------------

RawAsmParseResult parseRawAsmString(const std::string& asmText, GfxArchID arch) {
    RawAsmParseResult result;

    // Try to extract the kernel metadata header (SignatureBase).
    // If successful, we skip everything before the kernel body label during instruction parsing.
    result.signature = parseKernelMetadata(asmText, arch);

    auto func = std::make_unique<ParsedFunction>();
    // Use the kernel name from the signature if available.
    func->funcName = result.signature ? result.signature->kernelDescriptor.kernelName : "";
    auto block = std::make_unique<ParsedBlock>();
    block->blockId = "entry";

    // Symbol table updated live as .set directives are encountered (in source order),
    // just like C++ variable assignments.
    SymbolTable syms;

    std::istringstream ss(asmText);
    std::string line;
    unsigned lineNo = 0;

    // When a signature was extracted, skip the metadata header and the .macro/.endm blocks
    // that precede the real kernel body. We detect the end of the header by looking for the
    // kernel body label line: "<kernelName>:".
    bool headerSkip = result.signature != nullptr;
    const std::string kernelBodyMarker =
        result.signature ? (result.signature->kernelDescriptor.kernelName + ":") : "";
    bool inMacroBlock = false;  // inside .macro ... .endm

    while (std::getline(ss, line)) {
        ++lineNo;

        // Strip # comments (line starts with #), // comments, and ; comments
        if (!line.empty() && line[0] == '#') {
            continue;  // entire line is a comment
        }
        // Before stripping // comments, check for "// st.token:N,M,..." annotation.
        // This lets raw .s files carry token group hints without affecting the assembler.
        std::vector<int> lineTokens;
        for (size_t i = 0; i + 1 < line.size(); ++i) {
            if (line[i] == '/' && line[i + 1] == '/') {
                std::string comment = line.substr(i + 2);
                // Search for "st.token:" anywhere in the comment so that other
                // comment text (e.g. "// load A // st.token:0") is tolerated.
                size_t tokPos = comment.find("st.token:");
                if (tokPos != std::string::npos) {
                    std::string tokenList = comment.substr(tokPos + 9);
                    // Parse comma-separated integers
                    size_t p = 0;
                    while (p < tokenList.size()) {
                        while (p < tokenList.size() &&
                               (tokenList[p] == ' ' || tokenList[p] == '\t'))
                            ++p;
                        if (p >= tokenList.size()) break;
                        char* endPtr = nullptr;
                        long val = std::strtol(tokenList.c_str() + p, &endPtr, 10);
                        if (endPtr == tokenList.c_str() + p) break;  // not a number
                        lineTokens.push_back(static_cast<int>(val));
                        p = static_cast<size_t>(endPtr - tokenList.c_str());
                        while (p < tokenList.size() &&
                               (tokenList[p] == ' ' || tokenList[p] == '\t'))
                            ++p;
                        if (p < tokenList.size() && tokenList[p] == ',') ++p;
                    }
                }
                line = line.substr(0, i);
                break;
            }
        }
        {
            size_t semi = line.find(';');
            if (semi != std::string::npos) line = line.substr(0, semi);
        }

        // Trim leading/trailing whitespace
        size_t start = line.find_first_not_of(" \t\r");
        if (start == std::string::npos) continue;
        size_t end = line.find_last_not_of(" \t\r");
        line = line.substr(start, end - start + 1);

        if (line.empty()) continue;

        // ── Header / macro skip ──────────────────────────────────────────────────
        // When a signature was extracted, skip everything up to (and including) the
        // "<kernelName>:" label line.  Also skip .macro ... .endm blocks.
        if (headerSkip) {
            if (line.substr(0, 6) == ".macro") {
                inMacroBlock = true;
                continue;
            }
            if (line == ".endm") {
                inMacroBlock = false;
                continue;
            }
            if (inMacroBlock) continue;
            // Detect the kernel body start: "KernelName:" or "KernelName:  /// ..."
            if (!kernelBodyMarker.empty() && line.size() >= kernelBodyMarker.size() &&
                line.substr(0, kernelBodyMarker.size()) == kernelBodyMarker) {
                headerSkip = false;
                // Emit the kernel label as a ParsedInstruction label.
                auto labelInst = std::make_unique<ParsedInstruction>(
                    result.signature->kernelDescriptor.kernelName, true);
                block->instructions.push_back(std::move(labelInst));
            }
            continue;
        }

        // Directives: lines starting with '.'
        if (line[0] == '.') {
            // Special case: .set symbol, value → proper AsmDirective SET
            if (line.size() > 4 && line.substr(0, 4) == ".set") {
                // Update symbol table live (in source order, like a C++ variable assignment).
                processSetDirective(line, syms);
                auto inst = std::make_unique<ParsedInstruction>("asm_directive");
                // srcRegs[0] = ".set", srcRegs[1] = symbol, srcRegs[2] = value
                inst->srcRegs.push_back(StinkyRegister(std::string(".set")));
                // Parse "symbol, value" remainder
                std::string rest = line.substr(4);
                size_t commaPos = rest.find(',');
                if (commaPos != std::string::npos) {
                    std::string sym = rest.substr(0, commaPos);
                    std::string val = rest.substr(commaPos + 1);
                    trimStr(sym);
                    trimStr(val);
                    inst->srcRegs.push_back(StinkyRegister(sym));
                    inst->srcRegs.push_back(StinkyRegister(val));
                } else {
                    std::string sym = rest;
                    size_t f = sym.find_first_not_of(" \t");
                    if (f != std::string::npos) sym = sym.substr(f);
                    inst->srcRegs.push_back(StinkyRegister(sym));
                }
                block->instructions.push_back(std::move(inst));
            } else {
                block->instructions.push_back(makeTextBlock(line));
            }
            continue;
        }

        // Labels: "identifier:" on its own (no other tokens after the colon)
        // Use IRLexer to check: first token = Identifier, second = Colon
        {
            IRLexer probe(line);
            probe.lex();
            if (!probe.isAtEnd() && probe.peek().kind == TokenKind::Identifier) {
                const Token& first = probe.peek();
                if (probe.peekAhead(1).kind == TokenKind::Colon) {
                    // Check nothing meaningful follows the colon
                    // (a line like "foo: v_add..." is a label + instruction on same line;
                    //  treat the whole line as text to be safe)
                    bool labelOnly = probe.peekAhead(2).kind == TokenKind::Eof ||
                                     probe.peekAhead(2).kind == TokenKind::Newline;
                    if (labelOnly) {
                        std::string labelName(first.text);
                        auto labelInst = std::make_unique<ParsedInstruction>(labelName, true);
                        block->instructions.push_back(std::move(labelInst));
                        continue;
                    }
                }
            }
        }

        // Real instruction
        auto inst = parseInstLine(line, arch, result.diagnostics, lineNo, syms);
        if (inst) {
            if (!lineTokens.empty()) {
                // Build "[N,M,...]" string for ModifierSerializer::deserialize
                std::string tokStr = "[";
                for (size_t ti = 0; ti < lineTokens.size(); ++ti) {
                    if (ti) tokStr += ',';
                    tokStr += std::to_string(lineTokens[ti]);
                }
                tokStr += ']';
                inst->modifiers["mod.memtoken"]["tokens"] = tokStr;
            }
            block->instructions.push_back(std::move(inst));
        } else {
            // Unknown mnemonic or parse failure → preserve verbatim
            DEBUG_WITH_TYPE("RawAsmParser", std::cerr << "[RawAsmParser] line " << lineNo
                                                      << ": text block: " << line << "\n");
            block->instructions.push_back(makeTextBlock(line));
        }
    }

    func->blocks.push_back(std::move(block));
    result.parsedFunction = std::move(func);
    return result;
}

}  // namespace stinkytofu

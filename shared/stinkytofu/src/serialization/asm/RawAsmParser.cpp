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
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "IRLexer.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"

#define DEBUG_TYPE "RawAsmParser"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"
#include "stinkytofu/ir/asm/StinkySignature.hpp"

namespace stinkytofu {
namespace {

//----------------------------------------------------------------------
// Symbol table: evaluates .set expressions to integer values
//----------------------------------------------------------------------

using SymbolTable = std::unordered_map<std::string, int>;

/// Evaluate a simple arithmetic expression composed of integer literals,
/// symbol names (resolved via \p syms), unary +/-, and binary + - * /.
/// Multiplicative operators bind tighter than additive (standard precedence);
/// no parentheses are supported (TensileLite never emits them in offsets).
/// Returns nullopt if any token is unresolvable or the expression is malformed.
///
/// Grammar:
///   expr   ::= term  (('+' | '-') term  )*
///   term   ::= factor (('*' | '/') factor)*
///   factor ::= ('+' | '-')? (NUMBER | IDENTIFIER)
std::optional<int> evalExpr(const std::string& expr, const SymbolTable& syms) {
    size_t i = 0;
    const size_t n = expr.size();

    auto skipWS = [&]() {
        while (i < n && (expr[i] == ' ' || expr[i] == '\t')) ++i;
    };

    // Parse a single factor (signed atom: number or identifier).
    auto parseFactor = [&]() -> std::optional<int> {
        skipWS();
        if (i >= n) return std::nullopt;
        int sign = 1;
        if (expr[i] == '+') {
            ++i;
            skipWS();
        } else if (expr[i] == '-') {
            sign = -1;
            ++i;
            skipWS();
        }
        if (i >= n) return std::nullopt;
        const size_t start = i;
        if (std::isdigit(static_cast<unsigned char>(expr[i]))) {
            while (i < n && std::isdigit(static_cast<unsigned char>(expr[i]))) ++i;
            return sign * std::stoi(expr.substr(start, i - start));
        }
        if (std::isalpha(static_cast<unsigned char>(expr[i])) || expr[i] == '_') {
            while (i < n && (std::isalnum(static_cast<unsigned char>(expr[i])) || expr[i] == '_'))
                ++i;
            std::string name = expr.substr(start, i - start);
            auto it = syms.find(name);
            if (it == syms.end()) return std::nullopt;
            return sign * it->second;
        }
        return std::nullopt;
    };

    // Parse a term: factor (('*' | '/') factor)*
    std::function<std::optional<int>()> parseTerm = [&]() -> std::optional<int> {
        auto lhs = parseFactor();
        if (!lhs) return std::nullopt;
        while (true) {
            skipWS();
            if (i >= n || (expr[i] != '*' && expr[i] != '/')) break;
            char op = expr[i++];
            auto rhs = parseFactor();
            if (!rhs) return std::nullopt;
            if (op == '*') {
                lhs = *lhs * *rhs;
            } else {
                if (*rhs == 0) return std::nullopt;  // div by zero
                lhs = *lhs / *rhs;
            }
        }
        return lhs;
    };

    // Parse the full expression: term (('+' | '-') term)*
    auto lhs = parseTerm();
    if (!lhs) return std::nullopt;
    while (true) {
        skipWS();
        if (i >= n) break;
        if (expr[i] != '+' && expr[i] != '-') return std::nullopt;
        char op = expr[i++];
        auto rhs = parseTerm();
        if (!rhs) return std::nullopt;
        lhs = (op == '+') ? *lhs + *rhs : *lhs - *rhs;
    }
    return lhs;
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

/// If the token stream starts with an arithmetic operator (`/`, `*`, `+`,
/// `-`) followed by an int / hex / identifier token, append the chain to
/// `base` so a bare arithmetic expression like `34816/2`, `vgprBase+0`,
/// or `NumPersistIters-1` is preserved as a single LiteralString operand.
///
/// Two operator-shapes need to be handled because of an asymmetry in the
/// IRLexer:
///
///   * `+`, `*`, `/` emit as Unknown one-character tokens, so the next
///     token is the rhs operand.
///   * `-` followed by a digit is special-cased in IRLexer.cpp (`case '-':`)
///     and emitted as a single signed IntegerLiteral / HexLiteral / Float
///     literal whose text already begins with `-`. There is no separate
///     `-` token to detect, so we recognise such a literal as an implicit
///     `-<num>` continuation and splice its text directly onto `base`.
///
/// Without both shapes, an expression like `NumPersistIters-1` would have
/// `NumPersistIters` recovered as a LiteralString and the signed `-1`
/// IntegerLiteral silently swallowed by parseModifiers as an "unknown
/// token", producing `s_cmp_lt_u32 s[...], NumPersistIters` on round-trip.
inline std::string gatherArithExprSuffix(IRLexer& lexer, std::string base) {
    auto isArithOp = [](TokenKind k, std::string_view t) {
        return k == TokenKind::Unknown && t.size() == 1 &&
               (t[0] == '/' || t[0] == '*' || t[0] == '+' || t[0] == '-');
    };
    auto isSignedLiteral = [](TokenKind k, std::string_view t) {
        return (k == TokenKind::IntegerLiteral || k == TokenKind::HexLiteral ||
                k == TokenKind::FloatLiteral) &&
               !t.empty() && t[0] == '-';
    };
    while (true) {
        TokenKind k = lexer.peek().kind;
        std::string_view tx = lexer.peek().text;
        if (isArithOp(k, tx)) {
            std::string opTxt(lexer.consume().text);
            TokenKind nk = lexer.peek().kind;
            if (nk != TokenKind::IntegerLiteral && nk != TokenKind::HexLiteral &&
                nk != TokenKind::Identifier) {
                // Trailing op with no rhs operand. Keep the op so any
                // diagnostic still points at the malformed expression
                // rather than silently swallowing it; further chaining
                // stops here.
                base += opTxt;
                break;
            }
            base += opTxt;
            base += std::string(lexer.consume().text);
        } else if (isSignedLiteral(k, tx)) {
            // The literal text already begins with "-"; append verbatim.
            base += std::string(lexer.consume().text);
        } else {
            break;
        }
    }
    return base;
}

/// Parse a single register operand from the lexer.
/// Adapted from IRParser::parseRegister().
std::optional<StinkyRegister> parseOneRegister(IRLexer& lexer, const SymbolTable& syms,
                                               bool preserveSymbolicNames) {
    TokenKind kind = lexer.peek().kind;

    // Hex immediate: preserve as literal string. Also fold any trailing
    // arithmetic suffix (e.g. `0xff/2`, `0xff-32`) into the same literal so
    // it round-trips as one operand. gatherArithExprSuffix is a no-op when
    // the next token is not an operator or signed-literal continuation.
    if (kind == TokenKind::HexLiteral) {
        const Token& tok = lexer.consume();
        std::string text = gatherArithExprSuffix(lexer, std::string(tok.text));
        return StinkyRegister(text);
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
        // Arithmetic expression on a literal source operand:
        // e.g. `s_mul_i32 s86, s[sgprWaveId], 34816/2`, `34816-512`,
        // `2*8704*2+32`. The lexer emits multi-operator chains as
        // separate tokens (and signed numbers as a single signed literal
        // — see gatherArithExprSuffix); without folding them back into
        // a LiteralString the operand loop would only see the leading
        // int and silently drop the rest.
        std::string expr = gatherArithExprSuffix(lexer, num);
        if (expr != num) return StinkyRegister(expr);
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

    // Unknown type → not a register, signal caller to stop operand parsing.
    // The caller (parseInstLine) decides whether to:
    //   * bail to TEXTBLOCK pass-through (when this is the first operand —
    //     instructions such as `s_delay_alu instid0(VALU_DEP_2)` look like
    //     "<mnemonic> <unknown-id>(...)" and must round-trip verbatim
    //     because their custom operand syntax is not parsed here), or
    //   * preserve the token as a LiteralString operand (when this is a
    //     later operand — covers `.set` symbols used as immediates such
    //     as `v_mov_b32 v2, MT0`).
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
            // Store the symbolic name alongside the numeric index (matches rocisa flow)
            // only when the caller asked for it via RawAsmParserOptions. Optimisation
            // passes generally see only the numeric index, so omitting it by default
            // avoids stale symbolic names surviving register rewrites.
            //
            // The emitter prepends regTypeStr + '[' and appends ']' itself, so we
            // store ONLY the inner content (e.g. "vgprSerial-512" or
            // "vgprFoo+0:vgprFoo+3"), not the wrapped "v[...]" form. Storing the
            // wrapped form would double-wrap the output to "v[v[vgprSerial-512]]".
            StinkyRegister reg(regType, static_cast<uint32_t>(si), static_cast<uint16_t>(regNum),
                               offset);
            if (preserveSymbolicNames) reg.setSymbolicName(bracketContent);
            return reg;
        }
    }

    // Could not resolve — store as LiteralString so at least the instruction
    // is a StinkyInstruction (not a TEXTBLOCK).
    return StinkyRegister(regTypeStr + "[" + bracketContent + "]");
}

//----------------------------------------------------------------------
// FieldType helpers
//----------------------------------------------------------------------

/// Returns true if the given FieldType uses custom textual syntax that
/// parseOneRegister cannot handle. These operands are dispatched to
/// dedicated parsers in the per-field operand loop.
bool hasCustomOperandSyntax(FieldType ft) {
    switch (ft) {
        case FieldType::delay:
        case FieldType::wait_alu:
        case FieldType::hwreg:
            return true;
        default:
            return false;
    }
}

//----------------------------------------------------------------------
// Custom operand parsers
//----------------------------------------------------------------------

/// Parse s_delay_alu instid0/instskip/instid1 syntax into mod.delayalu fields.
bool parseDelayAluSyntax(IRLexer& lexer, ParsedInstruction& inst) {
    auto parseInstId = [](const std::string& s) -> std::pair<std::string, int> {
        if (s.find("VALU_DEP_") == 0) return {"VALU", std::stoi(s.substr(9))};
        if (s.find("TRANS32_DEP_") == 0) return {"TRANS", std::stoi(s.substr(12))};
        if (s.find("SALU_CYCLE_") == 0) return {"SALU", std::stoi(s.substr(11))};
        return {"NO_DEP", 0};
    };
    auto parseSkip = [](const std::string& s) -> int {
        if (s == "SAME") return 0;
        if (s == "NEXT") return 1;
        if (s.find("SKIP_") == 0) return std::stoi(s.substr(5)) + 1;
        return 0;
    };
    auto& fields = inst.modifiers["mod.delayalu"];
    while (!lexer.isAtEnd() && lexer.peek().kind != TokenKind::Eof &&
           lexer.peek().kind != TokenKind::Newline) {
        if (lexer.peek().kind != TokenKind::Identifier) {
            lexer.consume();
            continue;
        }
        std::string key(lexer.consume().text);
        std::string val;
        if (!lexer.isAtEnd() && lexer.peek().kind == TokenKind::LeftParen) {
            lexer.consume();
            if (!lexer.isAtEnd()) val = std::string(lexer.consume().text);
            if (!lexer.isAtEnd() && lexer.peek().kind == TokenKind::RightParen) lexer.consume();
        }
        if (key == "instid0") {
            auto [type, dist] = parseInstId(val);
            fields["instid0Type"] = type;
            fields["instid0Distance"] = std::to_string(dist);
        } else if (key == "instskip") {
            fields["instSkip"] = std::to_string(parseSkip(val));
        } else if (key == "instid1") {
            auto [type, dist] = parseInstId(val);
            fields["instid1Type"] = type;
            fields["instid1Distance"] = std::to_string(dist);
        }
    }
    if (fields.find("instid0Type") == fields.end()) {
        fields["instid0Type"] = "NO_DEP";
        fields["instid0Distance"] = "0";
    }
    return true;
}

/// Parse s_wait_alu depctr_*() syntax into mod.waitalu fields.
bool parseWaitAluSyntax(IRLexer& lexer, ParsedInstruction& inst) {
    auto& fields = inst.modifiers["mod.waitalu"];
    while (!lexer.isAtEnd() && lexer.peek().kind != TokenKind::Eof &&
           lexer.peek().kind != TokenKind::Newline) {
        if (lexer.peek().kind != TokenKind::Identifier) {
            lexer.consume();
            continue;
        }
        std::string key(lexer.consume().text);
        std::string val;
        if (!lexer.isAtEnd() && lexer.peek().kind == TokenKind::LeftParen) {
            lexer.consume();
            if (!lexer.isAtEnd()) val = std::string(lexer.consume().text);
            if (!lexer.isAtEnd() && lexer.peek().kind == TokenKind::RightParen) lexer.consume();
        }
        if (key.find("depctr_") == 0) key = key.substr(7);
        if (!key.empty() && !val.empty()) fields[key] = val;
    }
    return true;
}

/// Parse hwreg(id, offset, width) syntax and store as a LiteralString
/// register on the instruction for verbatim round-trip emission.
void parseHwregOperand(IRLexer& lexer, ParsedInstruction& inst,
                       const HwInstDesc::OperandFieldDesc& field) {
    if (lexer.isAtEnd() || lexer.peek().kind != TokenKind::Identifier) return;
    std::string text(lexer.peek().text);
    if (text != "hwreg") return;
    lexer.consume();

    if (lexer.isAtEnd() || lexer.peek().kind != TokenKind::LeftParen) {
        StinkyRegister reg(text);
        if (field.isDest)
            inst.destRegs.push_back(reg);
        else
            inst.srcRegs.push_back(reg);
        return;
    }
    text += '(';
    lexer.consume();

    int depth = 1;
    while (!lexer.isAtEnd() && depth > 0) {
        const auto& tok = lexer.consume();
        if (tok.kind == TokenKind::LeftParen) depth++;
        if (tok.kind == TokenKind::RightParen) depth--;
        if (depth > 0) text += std::string(tok.text);
    }
    text += ')';

    StinkyRegister reg(text);
    if (field.isDest)
        inst.destRegs.push_back(reg);
    else
        inst.srcRegs.push_back(reg);
}

/// Dispatch a custom-syntax operand to its dedicated parser based on FieldType.
void parseCustomOperand(IRLexer& lexer, ParsedInstruction& inst,
                        const HwInstDesc::OperandFieldDesc& field) {
    switch (field.fieldType) {
        case FieldType::delay:
            parseDelayAluSyntax(lexer, inst);
            break;
        case FieldType::wait_alu:
            parseWaitAluSyntax(lexer, inst);
            break;
        case FieldType::hwreg:
            parseHwregOperand(lexer, inst, field);
            break;
        default:
            break;
    }
}

//----------------------------------------------------------------------
// Modifier parsing
//----------------------------------------------------------------------

/// Generic key→value map used while collecting modifier tokens before they
/// are dispatched to a modifier namespace.
using FieldMap = std::unordered_map<std::string, std::string>;

/// True when \p fields holds any matrix-format modifier keys produced by the
/// `v_wmma_scale_*` family. Used by inferModKeyFromFields() to assign
/// `mod.matrix_fmt` when the microcode-based switch did not (matrix_fmt
/// rides on top of generic VOP3PX2/3 encodings, so the encoding alone is
/// not a reliable indicator).
bool hasMatrixFmtFields(const FieldMap& fields) {
    return fields.count("matrix_a_fmt") || fields.count("matrix_b_fmt") ||
           fields.count("matrix_a_scale_fmt") || fields.count("matrix_b_scale_fmt") ||
           fields.count("matrix_a_reuse") || fields.count("matrix_b_reuse");
}

/// True when \p fields holds any DPP-related modifier key. DPP is an add-on
/// encoding variant layered on top of many VOP formats; the assembler picks
/// the wider DPP encoding whenever such a token is present. Detection is
/// field-based rather than microcode-based because the same VOP* encoding
/// may or may not carry DPP depending on the opcode/usage.
bool hasDPPFields(const FieldMap& fields) {
    for (std::string_view k : kDppCtrlKeys)
        if (fields.count(std::string(k))) return true;
    // Encoding-side fields that accompany DPP regardless of the dpp_ctrl mode.
    return fields.count("bound_ctrl") || fields.count("row_mask") || fields.count("bank_mask") ||
           fields.count("fi");
}

/// Single decision point for "this generic field set looks like which modKey?".
/// Used after field collection, only when the microcode-based switch did not
/// already assign a modKey. Modifier kinds that ride on top of generic
/// encoding shapes (matrix_fmt on VOP3PX2/3, DPP on many VOP*) are detected
/// here by field presence rather than by microcode format.
std::string inferModKeyFromFields(const FieldMap& fields) {
    if (hasMatrixFmtFields(fields)) return "mod.matrix_fmt";
    if (hasDPPFields(fields)) return "mod.dpp";
    return {};
}

/// Parse trailing modifier tokens into inst.modifiers.
/// modifier_key is determined by hwInstDesc->microcode and the instruction mnemonic.
///
/// Returns true if every modifier-shaped token sequence the parser consumed
/// could be represented in inst.modifiers. Returns false if the parser saw a
/// trailing token shape it cannot model (currently: a `key:value` where the
/// value is an Identifier such as `th:TH_LOAD_NT`, or any modifier-bearing
/// instruction whose microcode format has no modifier namespace mapped here
/// — e.g. TENSOR-format `tensor_load_to_lds`). The caller (parseInstLine)
/// uses this signal to bail out to TEXTBLOCK pass-through so the entire
/// line round-trips verbatim instead of silently dropping the unmodelled
/// modifier.
bool parseModifiers(IRLexer& lexer, ParsedInstruction& inst, const HwInstDesc* hwInstDesc,
                    const SymbolTable& syms) {
    const std::string& mnemonic = inst.opcodeStr;
    bool isWaitcnt = (mnemonic == "s_waitcnt");

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
            return true;
        }
    }

    // Collect key→value / key(value) / boolean-flag tokens
    FieldMap fields;
    // Set when we see syntax we can parse but cannot represent: e.g.
    // `key:Identifier_value` (`th:TH_LOAD_NT`, `scope:SCOPE_DEV`) or any
    // modifier whose microcode format isn't mapped above. Tracked
    // separately from `fields` so that we can still observe "had
    // modifiers" even when they are all unrepresentable and the field map
    // ends up empty.
    bool sawUnrepresentable = false;
    bool sawAnyModifier = false;

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
            // key:value format (e.g. offset:128, th:TH_LOAD_NT,
            // offset:2*8704*2+32)
            lexer.consume();  // eat ':'
            TokenKind vk = lexer.peek().kind;
            if (vk == TokenKind::IntegerLiteral || vk == TokenKind::HexLiteral) {
                std::string val(lexer.consume().text);
                // Arithmetic-expression value: e.g. `offset:2*8704*2+32`,
                // `offset:34816-512`. Downstream consumers parse the
                // value via `atoi` (see ModifierSerializer's `getInt`),
                // which would silently truncate to the first integer.
                // Drain the remainder of the expression so the lexer
                // stays in sync, then evaluate it to a single integer
                // using the symbol table (handles literal-only chains
                // like `8704*2` and symbol-bearing chains like
                // `vgprBase+0`). On evaluation failure, mark the line
                // unrepresentable and let the caller fall back to
                // TEXTBLOCK so the source expression round-trips
                // verbatim. (gatherArithExprSuffix is a no-op when the
                // next token is not an operator or signed-literal
                // continuation, so simple `offset:0` / `offset:32` are
                // unaffected.)
                std::string folded = gatherArithExprSuffix(lexer, val);
                if (folded != val) {
                    if (auto evaluated = evalExpr(folded, syms)) {
                        fields[tok] = std::to_string(*evaluated);
                    } else {
                        sawUnrepresentable = true;
                        fields[tok] = std::move(folded);
                    }
                } else {
                    fields[tok] = std::move(folded);
                }
                sawAnyModifier = true;
            } else if (vk == TokenKind::Identifier) {
                // `key:Identifier` modifiers (e.g. `matrix_a_fmt:MATRIX_FMT_FP8`,
                // `th:TH_LOAD_NT`, `scope:SCOPE_DEV`). Store the value in the
                // generic fields collection; the per-modKey dispatch below
                // decides which ones are recognized. Fields with no consumer
                // are silently dropped — same outcome as any other unhandled
                // modifier token. Formats with no modKey at all are still
                // caught by the existing `modKey.empty() && sawAnyModifier`
                // check below and routed to TEXTBLOCK.
                fields[tok] = std::string(lexer.consume().text);
                sawAnyModifier = true;
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
            sawAnyModifier = true;
        } else {
            // Boolean flag: glc, slc, nt, lds, gds, offen, nv, etc.
            fields[tok] = "true";
            sawAnyModifier = true;
        }
    }

    // Post-hoc: if the microcode-based switch did not assign a modKey, infer
    // one from the collected field names. Modifier kinds that ride on top of
    // generic encoding shapes (matrix_fmt on VOP3PX2/3, DPP on many VOP*)
    // can't be detected from the microcode format alone — only some opcodes
    // within those encodings carry the modifier.
    if (modKey.empty()) modKey = inferModKeyFromFields(fields);

    // Modifier-bearing instruction with no modKey to put them into (e.g.
    // TENSOR-format `tensor_load_to_lds offset:0`). Anything in `fields`
    // would be silently discarded below at the `modKey.empty()` early-return,
    // so flag the line as unrepresentable instead and let the caller fall
    // back to TEXTBLOCK pass-through.
    if (modKey.empty() && sawAnyModifier) sawUnrepresentable = true;

    if (modKey.empty() || fields.empty()) return !sawUnrepresentable;

    // Map generic fields → modifier dict using the appropriate namespace key.
    auto& modFields = inst.modifiers[modKey];

    if (modKey == "mod.ds") {
        if (fields.contains("offset")) {
            modFields["na"] = "1";
            modFields["offset"] = fields["offset"];
        } else if (fields.contains("offset0") || fields.contains("offset1")) {
            modFields["na"] = "2";
            modFields["offset0"] = fields.contains("offset0") ? fields["offset0"] : "0";
            modFields["offset1"] = fields.contains("offset1") ? fields["offset1"] : "0";
        }
        if (fields.contains("gds")) modFields["gds"] = "true";

    } else if (modKey == "mod.mubuf") {
        if (fields.contains("offen")) modFields["offen"] = "true";
        // "offen offset:N" emits both as a single unit; "offset" key carries the value.
        if (fields.contains("offset")) modFields["offset12"] = fields["offset"];
        if (fields.contains("glc") || fields.contains("sc0")) modFields["glc"] = "true";
        if (fields.contains("slc") || fields.contains("sc1")) modFields["slc"] = "true";
        if (fields.contains("nt")) modFields["nt"] = "true";
        if (fields.contains("lds")) modFields["lds"] = "true";

    } else if (modKey == "mod.flat") {
        if (fields.contains("offset")) modFields["offset12"] = fields["offset"];
        if (fields.contains("glc") || fields.contains("sc0")) modFields["glc"] = "true";
        if (fields.contains("slc") || fields.contains("sc1")) modFields["slc"] = "true";
        if (fields.contains("lds")) modFields["lds"] = "true";

    } else if (modKey == "mod.global") {
        if (fields.contains("offset")) modFields["offset"] = fields["offset"];

    } else if (modKey == "mod.smem") {
        if (fields.contains("offset")) modFields["offset"] = fields["offset"];
        if (fields.contains("glc")) modFields["glc"] = "true";
        if (fields.contains("nv")) modFields["nv"] = "true";

    } else if (modKey == "mod.dpp") {
        // Reconstruct the asm-form dpp_ctrl token from the parsed fields and
        // run it through parseDppCtrlFromAsm() so the deserializer receives
        // the integer DppCtrl enum value it expects (the deserializer reads
        // `dppCtrl` as an int via getInt).
        DppCtrl ctrl = DppCtrl::NONE;
        for (std::string_view k : kDppCtrlKeys) {
            auto it = fields.find(std::string(k));
            if (it == fields.end()) continue;
            // Boolean flags (`row_mirror`, `row_half_mirror`) are stored as
            // "true"; their asm form is just the bare key. Everything else
            // is `key:value`.
            std::string token =
                (it->second == "true") ? std::string(k) : std::string(k) + ":" + it->second;
            ctrl = parseDppCtrlFromAsm(token);
            if (ctrl != DppCtrl::NONE) break;
        }
        if (ctrl != DppCtrl::NONE) modFields["dppCtrl"] = std::to_string(static_cast<int>(ctrl));
        // Encoding-side DPP fields. Map asm-form names to deserializer names.
        if (fields.count("bound_ctrl")) modFields["boundCtrl"] = fields["bound_ctrl"];
        if (fields.count("row_mask")) modFields["rowMask"] = fields["row_mask"];
        if (fields.count("bank_mask")) modFields["bankMask"] = fields["bank_mask"];
        if (fields.count("fi")) modFields["fi"] = fields["fi"];

    } else if (modKey == "mod.matrix_fmt") {
        // Map asm-form keys to the field names the mod.matrix_fmt deserializer
        // expects (parseMatrixFmt / parseMatrixScaleFmt accept the asm symbolic
        // form like "MATRIX_FMT_FP8" directly).
        if (fields.contains("matrix_a_fmt")) modFields["fmtA"] = fields["matrix_a_fmt"];
        if (fields.contains("matrix_b_fmt")) modFields["fmtB"] = fields["matrix_b_fmt"];
        if (fields.contains("matrix_a_scale_fmt"))
            modFields["scaleFmtA"] = fields["matrix_a_scale_fmt"];
        if (fields.contains("matrix_b_scale_fmt"))
            modFields["scaleFmtB"] = fields["matrix_b_scale_fmt"];
        // matrix_a_reuse / matrix_b_reuse live on MFMAModifiers, not
        // MatrixFmtModifiers — write them to a separate dict entry so the
        // deserializer creates both modifier types.
        if (fields.contains("matrix_a_reuse")) inst.modifiers["mod.mfma"]["reuseA"] = "true";
        if (fields.contains("matrix_b_reuse")) inst.modifiers["mod.mfma"]["reuseB"] = "true";
    }

    return !sawUnrepresentable;
}

//----------------------------------------------------------------------
// Line-level instruction parser
//----------------------------------------------------------------------

/// Parse one non-directive, non-label line into a ParsedInstruction.
/// Returns nullptr for unknown mnemonics (caller stores as TEXTBLOCK).
std::unique_ptr<ParsedInstruction> parseInstLine(const std::string& line, GfxArchID arch,
                                                 std::vector<Diagnostic>& diags, unsigned lineNo,
                                                 const SymbolTable& syms,
                                                 bool preserveSymbolicNames) {
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

    // Per-field operand dispatch: iterate operandFields and dispatch each
    // to the appropriate parser based on FieldType. Custom-syntax fields
    // (delay, wait_alu) are parsed by dedicated parsers here.
    // Register/immediate fields are parsed by parseOneRegister.
    if (!hwInstDesc->operandFields.empty()) {
        bool firstOp = true;
        int regOpIdx = 0;

        for (size_t fi = 0; fi < hwInstDesc->operandFields.size(); fi++) {
            if (lexer.isAtEnd() || lexer.peek().kind == TokenKind::Eof ||
                lexer.peek().kind == TokenKind::Newline)
                break;

            const auto& field = hwInstDesc->operandFields[fi];

            if (!firstOp) {
                if (lexer.peek().kind != TokenKind::Comma) break;
                lexer.consume();  // eat ','
            }
            firstOp = false;

            // Custom-syntax operand: dispatch to dedicated parser.
            if (hasCustomOperandSyntax(field.fieldType)) {
                parseCustomOperand(lexer, *inst, field);
                continue;
            }

            // Snapshot the lookahead so we can recover an unrecognised
            // identifier as a LiteralString if parseOneRegister fails on
            // a non-first operand (see recovery block below).
            TokenKind preKind = lexer.isAtEnd() ? TokenKind::Eof : lexer.peek().kind;
            std::string preText =
                preKind == TokenKind::Identifier ? std::string(lexer.peek().text) : std::string();
            TokenKind preNextKind = lexer.peekAhead(1).kind;

            auto reg = parseOneRegister(lexer, syms, preserveSymbolicNames);
            if (!reg) {
                if (regOpIdx == 0) {
                    // First register operand failed (e.g. unresolvable symbolic
                    // expression like v[sym-768:sym-765]). parseOneRegister may
                    // have consumed tokens; preserve the line as TEXTBLOCK.
                    return nullptr;
                }
                // Non-first operand recovery: a `.set` symbol (or any other
                // bare identifier) used as an immediate source — e.g.
                // `v_mov_b32 v2, MT0` where `.set MT0, 64` was declared
                // earlier — would otherwise be silently dropped, since
                // parseInstLine breaks out of the operand loop on a
                // nullopt. Preserve the consumed identifier as a
                // LiteralString so the operand round-trips instead of
                // truncating to `v_mov_b32 v2`. We restrict recovery to
                // bare identifiers (no `[`) so malformed register
                // expressions like `foo[12]` still trigger TEXTBLOCK
                // fallback rather than getting silently mangled.
                if (preKind == TokenKind::Identifier && !preText.empty() &&
                    preNextKind != TokenKind::LeftBracket) {
                    // Fold a trailing arithmetic suffix (e.g. `vgprBase+0`,
                    // `MT0/2`) into the same LiteralString so the whole
                    // expression round-trips as one operand. Same rationale
                    // as the IntegerLiteral path in parseOneRegister.
                    std::string text = gatherArithExprSuffix(lexer, preText);
                    StinkyRegister litReg(text);
                    if (field.isDest)
                        inst->destRegs.push_back(litReg);
                    else
                        inst->srcRegs.push_back(litReg);
                    regOpIdx++;
                    continue;
                }
                // Later operand failed → stop operand parsing, proceed to modifiers.
                break;
            }

            if (field.isDest)
                inst->destRegs.push_back(*reg);
            else
                inst->srcRegs.push_back(*reg);
            regOpIdx++;
        }
    }

    // Parse any trailing modifier tokens (offset:N, glc, vmcnt(N), etc.).
    // If parseModifiers signals that it saw a token shape it cannot
    // represent (e.g. `th:TH_LOAD_NT` on tensor_load_to_lds, or any
    // modifier on a microcode format that has no namespace mapped above),
    // bail to TEXTBLOCK pass-through so the line round-trips verbatim
    // instead of dropping the unmodelled modifier on emission.
    if (!parseModifiers(lexer, *inst, hwInstDesc, syms)) {
        return nullptr;
    }

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
    return v;
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
    // SignatureCodeMeta::toString() emits the second "- 1" of amdhsa.version
    // only when codeObjectVersion is "4" or "default"; "COV4" silently
    // truncates the YAML list to a single entry.
    std::string codeObjectVersion = "4";

    // Optimization config (parsed from the "Optimizations and Config" comment
    // block emitted between .end_amdhsa_kernel and .amdgpu_metadata, e.g.
    //   /* ThreadTile= 32 x 8 */
    //   /* SubGroup= 2 x 64 */
    //   /* VectorWidthA=1 */ ... etc.
    // Without these, SignatureKernelDescriptor regenerates the comment block
    // with all-zero defaults.
    std::array<int, 2> threadTile = {0, 0};
    std::array<int, 2> subGroup = {0, 0};
    std::array<int, 2> waveGroup = {0, 0};
    int vectorWidthA = 0;
    int vectorWidthB = 0;
    int globalReadVectorWidthA = 0;
    int globalReadVectorWidthB = 0;
    bool directToLdsA = false;
    bool directToLdsB = false;
    int useSgprForGRO = 0;

    // Verbatim .amdhsa_* lines that the structured fields don't model.
    std::vector<std::string> extraAmdhsaDirectives;

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
    // Track whether we have entered the YAML `.args:` block. Without this gate
    // the kernel-level `  - .name: <kernelName>` line (the first entry under
    // `amdhsa.kernels:`, BEFORE the args block) gets misidentified as an arg
    // and ends up duplicated at the head of the regenerated arg list.
    bool inArgsList = false;
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
            // Track which directives are consumed by structured fields; any
            // .amdhsa_* directive not in this set is captured verbatim and
            // re-emitted by SignatureKernelDescriptor::toString().
            static const char* const kKnown[] = {
                ".amdhsa_user_sgpr_kernarg_segment_ptr",
                ".amdhsa_next_free_vgpr",
                ".amdhsa_next_free_sgpr",
                ".amdhsa_group_segment_fixed_size",
                ".amdhsa_wavefront_size32",
                ".amdhsa_private_segment_fixed_size",
                ".amdhsa_system_sgpr_workgroup_id_x",
                ".amdhsa_system_sgpr_workgroup_id_y",
                ".amdhsa_system_sgpr_workgroup_id_z",
                ".amdhsa_system_vgpr_workitem_id",
                ".amdhsa_float_denorm_mode_32",
                ".amdhsa_float_denorm_mode_16_64",
                ".amdhsa_accum_offset",
            };

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

            // If the directive is .amdhsa_* but not consumed above, save it
            // verbatim for re-emission. Use the original (untrimmed) line so
            // the original indentation survives.
            if (trimmed.size() > 8 && trimmed.substr(0, 8) == ".amdhsa_") {
                bool known = false;
                for (const char* k : kKnown) {
                    size_t klen = std::strlen(k);
                    if (trimmed.size() >= klen && trimmed.compare(0, klen, k) == 0 &&
                        (trimmed.size() == klen ||
                         std::isspace(static_cast<unsigned char>(trimmed[klen])))) {
                        known = true;
                        break;
                    }
                }
                if (!known) extraAmdhsaDirectives.push_back(stripComment(line));
            }
            continue;
        }

        // ── /* Optimizations and Config */ comment block ─────────────────────────
        // Recover the optimization config (ThreadTile, SubGroup, VectorWidth*,
        // GlobalReadVectorWidth*, DirectToLds*, UseSgprForGRO) that
        // SignatureKernelDescriptor::toString() regenerates from those fields.
        // Without this the round-trip emits "/* ThreadTile= 0 x 0 */" etc.
        if (trimmed.size() >= 4 && trimmed.substr(0, 2) == "/*" &&
            trimmed.substr(trimmed.size() - 2) == "*/") {
            std::string body = trimmed.substr(2, trimmed.size() - 4);
            trimStr(body);
            auto parseTwoInts = [](const std::string& s, std::array<int, 2>& out) {
                size_t eq = s.find('=');
                if (eq == std::string::npos) return;
                std::string rhs = s.substr(eq + 1);
                size_t x = rhs.find('x');
                if (x == std::string::npos) return;
                std::string a = rhs.substr(0, x);
                std::string b = rhs.substr(x + 1);
                trimStr(a);
                trimStr(b);
                if (auto v = safeAtoiStr(a)) out[0] = *v;
                if (auto v = safeAtoiStr(b)) out[1] = *v;
            };
            auto parseInt = [](const std::string& s, int& out) {
                size_t eq = s.find('=');
                if (eq == std::string::npos) return;
                std::string rhs = s.substr(eq + 1);
                trimStr(rhs);
                if (auto v = safeAtoiStr(rhs)) out = *v;
            };
            auto parseBool = [](const std::string& s, bool& out) {
                size_t eq = s.find('=');
                if (eq == std::string::npos) return;
                std::string rhs = s.substr(eq + 1);
                trimStr(rhs);
                out = (rhs == "True" || rhs == "true" || rhs == "1");
            };
            if (body.starts_with("ThreadTile="))
                parseTwoInts(body, threadTile);
            else if (body.starts_with("SubGroup="))
                parseTwoInts(body, subGroup);
            else if (body.starts_with("WaveGroup="))
                parseTwoInts(body, waveGroup);
            else if (body.starts_with("VectorWidthA="))
                parseInt(body, vectorWidthA);
            else if (body.starts_with("VectorWidthB="))
                parseInt(body, vectorWidthB);
            else if (body.starts_with("DirectToLdsA="))
                parseBool(body, directToLdsA);
            else if (body.starts_with("DirectToLdsB="))
                parseBool(body, directToLdsB);
            else if (body.starts_with("UseSgprForGRO=")) {
                bool ub = false;
                parseBool(body, ub);
                useSgprForGRO = ub ? 1 : 0;
                // Some emitters write "UseSgprForGRO=N" (integer); honour that too.
                size_t eq = body.find('=');
                if (eq != std::string::npos) {
                    std::string rhs = body.substr(eq + 1);
                    trimStr(rhs);
                    if (auto v = safeAtoiStr(rhs)) useSgprForGRO = *v;
                }
            } else if (body.starts_with("GlobalReadVectorWidthA=") ||
                       body.find("GlobalReadVectorWidthA=") != std::string::npos) {
                // The combined form is "GlobalReadVectorWidthA=N, GlobalReadVectorWidthB=M".
                size_t pa = body.find("GlobalReadVectorWidthA=");
                if (pa != std::string::npos) {
                    std::string rest =
                        body.substr(pa + std::string("GlobalReadVectorWidthA=").size());
                    size_t comma = rest.find(',');
                    std::string a = (comma != std::string::npos) ? rest.substr(0, comma) : rest;
                    trimStr(a);
                    if (auto v = safeAtoiStr(a)) globalReadVectorWidthA = *v;
                }
                size_t pb = body.find("GlobalReadVectorWidthB=");
                if (pb != std::string::npos) {
                    std::string rest =
                        body.substr(pb + std::string("GlobalReadVectorWidthB=").size());
                    trimStr(rest);
                    if (auto v = safeAtoiStr(rest)) globalReadVectorWidthB = *v;
                }
            }
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
            inArgsList = false;
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
        // The `.args:` line marks the boundary between the kernel-level YAML
        // entry (which carries `.name`, `.symbol`, `.language`, `.args`, etc.)
        // and the args block itself. Only after this boundary should `- .name:`
        // be interpreted as an arg start; otherwise the kernel-level
        // `  - .name: <kernelName>` line is mis-parsed as the first arg.
        if (trimmed == ".args:" || trimmed.substr(0, 6) == ".args:") {
            inArgsList = true;
            continue;
        }

        // Each arg starts with "- .name:" — only when we are inside `.args:`.
        if (inArgsList && trimmed.substr(0, 8) == "- .name:") {
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

    // Restore the optimization-config comment block so the emitter regenerates
    // the same /* ThreadTile= */ / /* SubGroup= */ / /* VectorWidth* */ /
    // /* DirectToLds* */ / /* UseSgprForGRO= */ values that were in the source.
    sig->setOptimizationConfig(threadTile, subGroup, waveGroup, vectorWidthA, vectorWidthB,
                               globalReadVectorWidthA, globalReadVectorWidthB, directToLdsA,
                               directToLdsB, useSgprForGRO);

    // Pass-through .amdhsa_* directives (e.g. .amdhsa_user_sgpr_count,
    // .amdhsa_fp16_overflow, .amdhsa_inst_pref_size,
    // .amdhsa_user_sgpr_kernarg_preload_*) that the structured fields above
    // don't model. The signature emitter writes each entry verbatim before
    // .end_amdhsa_kernel, so the round-trip is lossless.
    sig->kernelDescriptor.extraKernelDirectives = std::move(extraAmdhsaDirectives);

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

RawAsmParseResult parseRawAsmString(const std::string& asmText, GfxArchID arch,
                                    const RawAsmParserOptions& options) {
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

    // When a signature was extracted, skip the metadata header up to the
    // "<kernelName>:" body label.  .macro/.endm blocks must be preserved
    // verbatim — both before and after the body label — so that template
    // lines like "v_mul_hi_u32 v[\vgprDstIdx+1], \dividend, \magicNumber"
    // (which contain backslash-prefixed macro arguments the instruction
    // parser cannot decode) round-trip exactly. Inside a macro block every
    // line is captured as a TEXTBLOCK directive.
    bool headerSkip = result.signature != nullptr;
    const std::string kernelBodyMarker =
        result.signature ? (result.signature->kernelDescriptor.kernelName + ":") : "";
    bool inMacroBlock = false;

    while (std::getline(ss, line)) {
        ++lineNo;

        // .macro / .endm tracking and pass-through. Done BEFORE comment
        // stripping so the verbatim text is preserved (a macro body is just
        // text from our perspective). We also do it BEFORE the headerSkip
        // gate because macros preceding the kernel-body label still need to
        // be emitted (otherwise the macro definitions are lost).
        std::string raw = line;
        {
            std::string probe = raw;
            size_t s = probe.find_first_not_of(" \t\r");
            size_t e = probe.find_last_not_of(" \t\r");
            if (s != std::string::npos) probe = probe.substr(s, e - s + 1);

            if (probe.size() >= 6 && probe.substr(0, 6) == ".macro" &&
                (probe.size() == 6 || std::isspace(static_cast<unsigned char>(probe[6])))) {
                inMacroBlock = true;
                if (!headerSkip) block->instructions.push_back(makeTextBlock(raw));
                continue;
            }
            if (inMacroBlock) {
                if (probe == ".endm") {
                    inMacroBlock = false;
                    if (!headerSkip) block->instructions.push_back(makeTextBlock(raw));
                    continue;
                }
                if (!headerSkip) block->instructions.push_back(makeTextBlock(raw));
                continue;
            }
        }

        // Strip # comments (line starts with #), // comments, and ; comments.
        // When the option is set, capture the trailing "// ..." or ";" comment
        // first so we can re-attach it to the parsed instruction/directive.
        if (!line.empty() && line[0] == '#') {
            continue;  // entire line is a comment
        }
        // Before stripping // comments, check for "// st.token:N,M,..." annotation.
        // This lets raw .s files carry token group hints without affecting the assembler.
        std::vector<int> lineTokens;

        // Find first '//' or ';' line-comment delimiter that is OUTSIDE any
        // /* ... */ block. Naive line.find("//") / find(';') would truncate
        // inside a block like `/* foo//bar */`, dropping the closing `*/` and
        // leaving an unterminated comment that the assembler would extend
        // across subsequent instructions.
        // Out-param `kind`: 0 = '//', 1 = ';'. Returns npos if none found.
        auto findLineCommentOutsideBlock = [](const std::string& s, int& kind) -> size_t {
            size_t i = 0;
            while (i < s.size()) {
                if (i + 1 < s.size() && s[i] == '/' && s[i + 1] == '*') {
                    size_t end = s.find("*/", i + 2);
                    if (end == std::string::npos)
                        return std::string::npos;  // unclosed; leave whole line
                    i = end + 2;
                    continue;
                }
                if (i + 1 < s.size() && s[i] == '/' && s[i + 1] == '/') {
                    kind = 0;
                    return i;
                }
                if (s[i] == ';') {
                    kind = 1;
                    return i;
                }
                ++i;
            }
            return std::string::npos;
        };

        // Capture trailing "// ..." or ";" text so the emitter can re-attach
        // it to the parsed instruction. If the captured text is actually a
        // "st.token:" annotation, the token parser below populates lineTokens
        // and we clear lineComment so hidden annotations are not echoed back.
        std::string lineComment;
        if (options.preserveComments) {
            int kind = -1;
            size_t pos = findLineCommentOutsideBlock(line, kind);
            if (pos != std::string::npos) {
                size_t skip = (kind == 0) ? 2 : 1;
                lineComment = line.substr(pos + skip);
                size_t f = lineComment.find_first_not_of(" \t");
                if (f == std::string::npos)
                    lineComment.clear();
                else
                    lineComment.erase(0, f);
                size_t l = lineComment.find_last_not_of(" \t\r\n");
                if (l != std::string::npos) lineComment.resize(l + 1);
            }
        }
        {
            int kind = -1;
            size_t pos = findLineCommentOutsideBlock(line, kind);
            if (pos != std::string::npos && kind == 0) {
                // '//' line comment — also parse 'st.token:' annotation
                std::string comment = line.substr(pos + 2);
                size_t tokPos = comment.find("st.token:");
                if (tokPos != std::string::npos) {
                    std::string tokenList = comment.substr(tokPos + 9);
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
                line = line.substr(0, pos);
            } else if (pos != std::string::npos && kind == 1) {
                line = line.substr(0, pos);
            }
        }

        // Trim leading/trailing whitespace
        size_t start = line.find_first_not_of(" \t\r");
        if (start == std::string::npos) continue;
        size_t end = line.find_last_not_of(" \t\r");
        line = line.substr(start, end - start + 1);

        if (line.empty()) continue;

        // ── Header skip ──────────────────────────────────────────────────────────
        // When a signature was extracted, skip everything up to (and including)
        // the "<kernelName>:" label line.
        if (headerSkip) {
            // Detect the kernel body start: "KernelName:" or "KernelName:  /// ..."
            // The kernel body label is intentionally NOT pushed into the IR
            // here: SignatureCodeMeta::toString() ends its emission with
            // "<kernelName>:\n", so the signature header itself provides the
            // label. Pushing it again as a ParsedInstruction would duplicate
            // it in the round-trip output (signature emits the label, then
            // the function emit prints the same label).
            if (!kernelBodyMarker.empty() && line.size() >= kernelBodyMarker.size() &&
                line.substr(0, kernelBodyMarker.size()) == kernelBodyMarker) {
                headerSkip = false;
            }
            continue;
        }

        // Helper: turn a stripped line back into a verbatim text-block, re-
        // appending the captured trailing comment so non-`.set` directives
        // (e.g. `.long X // hi`) and unparsable instructions still round-trip.
        auto textBlockWithComment = [&](const std::string& base) {
            if (lineComment.empty()) return makeTextBlock(base);
            std::string withComment = base;
            withComment += "  // ";
            withComment += lineComment;
            return makeTextBlock(withComment);
        };

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
                inst->comment = lineComment;
                block->instructions.push_back(std::move(inst));
            } else {
                block->instructions.push_back(textBlockWithComment(line));
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
                        labelInst->comment = lineComment;
                        block->instructions.push_back(std::move(labelInst));
                        continue;
                    }
                }
            }
        }

        // Real instruction
        auto inst = parseInstLine(line, arch, result.diagnostics, lineNo, syms,
                                  options.preserveSymbolicNames);
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
            // If the captured comment is a pure "st.token:" annotation, drop
            // it so the hidden token hint does not get echoed back as a
            // user-visible comment by the emitter.
            if (lineComment.compare(0, 9, "st.token:") == 0) lineComment.clear();
            inst->comment = lineComment;
            block->instructions.push_back(std::move(inst));
        } else {
            // Unknown mnemonic or parse failure → preserve verbatim
            DEBUG_WITH_TYPE("RawAsmParser", std::cerr << "[RawAsmParser] line " << lineNo
                                                      << ": text block: " << line << "\n");
            block->instructions.push_back(textBlockWithComment(line));
        }
    }

    func->blocks.push_back(std::move(block));
    result.parsedFunction = std::move(func);
    return result;
}

}  // namespace stinkytofu

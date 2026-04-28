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

#include "stinkytofu/serialization/asm/IRParser.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <optional>
#include <string_view>

#include "IRLexer.hpp"

using namespace stinkytofu;

namespace {
//----------------------------------------------------------------------
// Helper Functions
//----------------------------------------------------------------------

/// Safely convert string to integer with proper error handling
/// Returns std::nullopt if conversion fails
inline std::optional<int> safeStoi(const std::string& str, int base = 10) {
    // Manual validation instead of exception handling
    if (str.empty()) return std::nullopt;

    // Check for valid characters
    size_t startPos = 0;
    if (str[0] == '-' || str[0] == '+') startPos = 1;

    if (startPos >= str.length()) return std::nullopt;

    // For hex, skip 0x prefix
    if (base == 16 && str.length() > 2 && str[0] == '0' && (str[1] == 'x' || str[1] == 'X'))
        startPos = 2;

    // Try conversion with range checking
    long long result = 0;
    bool negative = (str[0] == '-');

    for (size_t i = startPos; i < str.length(); ++i) {
        int digit;
        char c = str[i];

        if (base == 10) {
            if (c < '0' || c > '9') return std::nullopt;
            digit = c - '0';
        } else if (base == 16) {
            if (c >= '0' && c <= '9')
                digit = c - '0';
            else if (c >= 'a' && c <= 'f')
                digit = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F')
                digit = c - 'A' + 10;
            else
                return std::nullopt;
        } else {
            return std::nullopt;  // Unsupported base
        }

        // Check for overflow before multiplying
        if (result > (LLONG_MAX - digit) / base) return std::nullopt;  // Would overflow

        result = result * base + digit;
    }

    if (negative) result = -result;

    // Check if fits in int
    if (result < INT_MIN || result > INT_MAX) return std::nullopt;

    return static_cast<int>(result);
}

inline bool isHexSuffixIdentifier(std::string_view id) {
    if (id.size() < 2) return false;
    if (id[0] != 'x' && id[0] != 'X') return false;
    for (size_t i = 1; i < id.size(); ++i) {
        if (!std::isxdigit(static_cast<unsigned char>(id[i]))) return false;
    }
    return true;
}

//----------------------------------------------------------------------
// IRParser declaration
//----------------------------------------------------------------------

/// Parser for the StinkyTofu IR text format.
/// Format: destRegs = "Stinkytofu.mnemonic"(srcRegs) { attributes }
/// Or labels: label_name:
class IRParser {
   private:
    IRLexer& lexer;
    std::vector<Diagnostic> diagnostics;
    bool hadError;

   public:
    explicit IRParser(IRLexer& lex);

    /// Parse the entire IR file and return a parsed function.
    /// Hierarchical format (st.func @name() { ... }) returns structured blocks.
    /// Flat format returns a single block "entry" with all instructions.
    std::unique_ptr<ParsedFunction> parse();

    /// Check if any errors were encountered during parsing.
    bool hasErrors() const {
        return hadError;
    }

    /// Get all diagnostics (errors and warnings).
    const std::vector<Diagnostic>& getDiagnostics() const {
        return diagnostics;
    }

    /// Print all diagnostics to stderr.
    void printDiagnostics() const;

    /// Parse all functions in the input.
    std::vector<std::unique_ptr<ParsedFunction>> parseAllFunctions();

   private:
    //------------------------------------------------------------------
    // Parsing methods
    //------------------------------------------------------------------

    /// Parse a single instruction line.
    /// Format: [destRegs =] "Stinkytofu.mnemonic"(srcRegs) { attributes }
    std::unique_ptr<ParsedInstruction> parseInstruction();

    /// Parse label (identifier followed by colon).
    /// Format: label_name:
    std::unique_ptr<ParsedInstruction> parseLabel();

    /// Parse hierarchical format: st.func @name() { ^block: ... }
    std::unique_ptr<ParsedFunction> parseFunction();

    /// Parse block: ^blockId: instructions [Successors: ^a, ^b]
    std::unique_ptr<ParsedBlock> parseBlock();

    /// Parse Successors: ^id, ^id, ...
    bool parseSuccessorsLine(ParsedBlock& block);

    /// Parse block label: ^blockId:
    std::optional<std::string> parseBlockLabel();

    /// Parse destination registers (comma-separated).
    /// Format: v[14:17], s[0]
    bool parseDestRegisters(ParsedInstruction& inst);

    /// Parse the operation name in quotes.
    /// Format: "Stinkytofu.ds_load_b128"
    std::optional<std::string> parseOperation();

    /// Parse source operands in parentheses.
    /// Format: (v[12], BARRIER[0])
    bool parseOperands(ParsedInstruction& inst);

    /// Parse attributes in braces.
    /// Format: { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, ... }, ... }
    bool parseAttributes(ParsedInstruction& inst);

    /// Parse a modifier dict: { field = value, ... }. Returns nullopt on error.
    std::optional<std::unordered_map<std::string, std::string>> parseModifierDict();

    /// Parse a single modifier field value (scalar, array, int, float, quoted string, bool).
    std::optional<std::string> parseAttributeValue();

    /// Parse a single register reference.
    /// Formats: v[14:17], s[0], acc[0:15], BARRIER[0], SCC[0], DS_WRITE[0], 0x..., 42, 3.14
    std::optional<StinkyRegister> parseRegister();

    //------------------------------------------------------------------
    // Utility methods
    //------------------------------------------------------------------

    /// Get the current token without consuming it.
    const Token& peek() const {
        return lexer.peek();
    }

    /// Consume and return the current token.
    const Token& consume() {
        return lexer.consume();
    }

    /// Check if the current token matches the expected kind.
    bool check(TokenKind kind) const {
        return peek().kind == kind;
    }

    /// Consume a token if it matches the expected kind.
    bool match(TokenKind kind) {
        if (check(kind)) {
            consume();
            return true;
        }
        return false;
    }

    /// Expect a specific token kind and consume it, or emit an error.
    bool expect(TokenKind kind, const std::string& message);

    /// Skip newlines.
    void skipNewlines();

    /// Emit an error diagnostic.
    void emitError(const std::string& message, unsigned line, unsigned column);

    /// Emit an error diagnostic at the current token.
    void emitError(const std::string& message);

    /// Emit a warning diagnostic.
    void emitWarning(const std::string& message, unsigned line, unsigned column);
};

//----------------------------------------------------------------------
// IRParser implementation
//----------------------------------------------------------------------

IRParser::IRParser(IRLexer& lex) : lexer(lex), hadError(false) {}

std::unique_ptr<ParsedFunction> IRParser::parse() {
    // Skip any leading newlines
    skipNewlines();

    // Check for hierarchical format: st.func @name() {
    bool hierarchical = false;
    if (peek().kind == TokenKind::Identifier) {
        std::string_view text = peek().text;
        if (text == "st.func") {
            hierarchical = true;
        }
    }

    if (hierarchical) {
        return parseFunction();
    }

    // Flat format: parse instructions and labels into a single block
    auto block = std::make_unique<ParsedBlock>();
    block->blockId = "entry";

    while (!lexer.isAtEnd() && peek().kind != TokenKind::Eof) {
        // Check for block label ^blockId: (hierarchical style in flat context)
        auto blockLabel = parseBlockLabel();
        if (blockLabel) {
            auto label = std::make_unique<ParsedInstruction>(*blockLabel, true);
            block->instructions.push_back(std::move(label));
            skipNewlines();
            continue;
        }

        // Check if this is a label (identifier followed by colon)
        if (peek().kind == TokenKind::Identifier && lexer.peekAhead(1).kind == TokenKind::Colon) {
            auto label = parseLabel();
            if (label) {
                block->instructions.push_back(std::move(label));
                skipNewlines();
                continue;
            }
        }

        auto inst = parseInstruction();
        if (inst) {
            block->instructions.push_back(std::move(inst));
        } else {
            while (!lexer.isAtEnd() && peek().kind != TokenKind::Eof &&
                   peek().kind != TokenKind::Newline) {
                consume();
            }
            if (peek().kind == TokenKind::Newline) {
                consume();
            }
        }

        skipNewlines();
    }

    auto func = std::make_unique<ParsedFunction>();
    func->funcName = "";
    func->blocks.push_back(std::move(block));
    return func;
}

void IRParser::printDiagnostics() const {
    for (const auto& diag : diagnostics) {
        std::cerr << diag.format() << "\n";
    }
}

std::unique_ptr<ParsedInstruction> IRParser::parseInstruction() {
    // format: [destRegs =] "Stinkytofu.mnemonic"(srcRegs) { attributes }

    auto inst = std::make_unique<ParsedInstruction>("");

    // Check if we have destination registers (followed by '=')
    // Look ahead to see if there's an '=' coming
    bool hasDest = false;
    if (peek().kind == TokenKind::Identifier) {
        // Could be: DEST[0] = "..." or just "..."
        // We need to look for '=' before '('
        // Simple approach: parse dest registers if we see identifier/bracket patterns
        hasDest = true;  // Tentatively
    }

    if (hasDest) {
        // Try to parse destination registers
        if (!parseDestRegisters(*inst)) {
            // If it fails, check if we're at the operation already
            if (peek().kind != TokenKind::QuotedString) {
                return nullptr;
            }
            // Otherwise clear the error and continue (no dest registers)
            inst->destRegs.clear();
        }

        // If we successfully parsed dest regs, expect '='
        if (!inst->destRegs.empty()) {
            if (!expect(TokenKind::Equal, "Expected '=' after destination registers")) {
                return nullptr;
            }
        }
    }

    // Parse the operation name in quotes: "Stinkytofu.mnemonic"
    auto opcodeOpt = parseOperation();
    if (!opcodeOpt) {
        emitError("Expected quoted operation name");
        return nullptr;
    }
    inst->opcodeStr = std::move(*opcodeOpt);

    // Parse source operands in parentheses: (v[10], s[48])
    if (!parseOperands(*inst)) {
        // Operands are optional, continue
    }

    // Parse attributes in braces: { issueCycles = 4, latencyCycles = 56 }
    if (!parseAttributes(*inst)) {
        // Attributes are optional, but emit warning if missing
        emitWarning("Instruction missing attributes (issueCycles, latencyCycles)", peek().line,
                    peek().column);
    }

    return inst;
}

std::unique_ptr<ParsedInstruction> IRParser::parseLabel() {
    // Format: label_name:
    // Should only be called after verifying the pattern in parse()
    if (peek().kind != TokenKind::Identifier) {
        return nullptr;
    }

    const Token& labelTok = consume();

    // Must be followed by colon (should always be true if called correctly)
    if (peek().kind != TokenKind::Colon) {
        // This should not happen if parse() verified the pattern correctly
        return nullptr;
    }

    consume();  // consume the colon

    // Create a special ParsedInstruction to represent a label
    // Store label name without the colon, set isLabel flag to true
    auto inst = std::make_unique<ParsedInstruction>(std::string(labelTok.text), true);
    return inst;
}

std::unique_ptr<ParsedFunction> IRParser::parseFunction() {
    // Format: st.func @name() { ^block: ... }
    if (peek().kind != TokenKind::Identifier || std::string(peek().text) != "st.func") {
        emitError("Expected 'st.func'");
        return nullptr;
    }
    consume();  // st.func

    // @funcName
    if (peek().kind != TokenKind::Identifier || peek().text.empty() || peek().text[0] != '@') {
        emitError("Expected function name (@name)");
        return nullptr;
    }
    std::string funcName(peek().text);
    if (funcName.size() > 1) funcName = funcName.substr(1);  // strip @
    consume();

    if (!expect(TokenKind::LeftParen, "Expected '(' after function name")) return nullptr;
    if (!expect(TokenKind::RightParen, "Expected ')'")) return nullptr;
    if (!expect(TokenKind::LeftBrace, "Expected '{' to start function body")) return nullptr;

    skipNewlines();

    auto parsedFunc = std::make_unique<ParsedFunction>();
    parsedFunc->funcName = funcName;

    while (!lexer.isAtEnd() && peek().kind != TokenKind::Eof) {
        if (peek().kind == TokenKind::RightBrace) {
            consume();
            return parsedFunc;
        }

        auto block = parseBlock();
        if (!block) {
            return nullptr;
        }
        parsedFunc->blocks.push_back(std::move(block));
        skipNewlines();
    }

    emitError("Expected '}' to close function body");
    return nullptr;
}

std::vector<std::unique_ptr<ParsedFunction>> IRParser::parseAllFunctions() {
    std::vector<std::unique_ptr<ParsedFunction>> functions;
    while (!lexer.isAtEnd() && peek().kind != TokenKind::Eof) {
        skipNewlines();
        if (lexer.isAtEnd() || peek().kind == TokenKind::Eof) break;
        if (peek().kind == TokenKind::Identifier && std::string(peek().text) == "st.func") {
            auto func = parseFunction();
            if (func)
                functions.push_back(std::move(func));
            else
                break;
        } else {
            break;  // not a function, stop
        }
    }
    return functions;
}

std::unique_ptr<ParsedBlock> IRParser::parseBlock() {
    auto blockId = parseBlockLabel();
    if (!blockId) {
        emitError("Expected block label (^blockId:)");
        return nullptr;
    }

    auto block = std::make_unique<ParsedBlock>();
    block->blockId = *blockId;
    skipNewlines();

    while (!lexer.isAtEnd() && peek().kind != TokenKind::Eof) {
        // Successors: ^a, ^b
        if (peek().kind == TokenKind::Identifier && std::string(peek().text) == "Successors") {
            if (!parseSuccessorsLine(*block)) return nullptr;
            skipNewlines();
            break;
        }

        // Next block or function end (peek only - don't consume)
        if (peek().kind == TokenKind::RightBrace) break;
        if (peek().kind == TokenKind::Identifier && !peek().text.empty() && peek().text[0] == '^' &&
            lexer.peekAhead(1).kind == TokenKind::Colon) {
            break;  // Next block - caller will parse it
        }

        auto inst = parseInstruction();
        if (inst) {
            block->instructions.push_back(std::move(inst));
        } else {
            // Try label (flat style) for backward compat
            if (peek().kind == TokenKind::Identifier &&
                lexer.peekAhead(1).kind == TokenKind::Colon) {
                auto label = parseLabel();
                if (label) {
                    block->instructions.push_back(std::move(label));
                    skipNewlines();
                    continue;
                }
            }
            // Error recovery
            while (!lexer.isAtEnd() && peek().kind != TokenKind::Eof &&
                   peek().kind != TokenKind::Newline) {
                consume();
            }
        }
        skipNewlines();
    }

    return block;
}

bool IRParser::parseSuccessorsLine(ParsedBlock& block) {
    if (peek().kind != TokenKind::Identifier || std::string(peek().text) != "Successors") {
        emitError("Expected 'Successors'");
        return false;
    }
    consume();  // Successors

    if (!expect(TokenKind::Colon, "Expected ':' after Successors")) return false;

    skipNewlines();

    while (!lexer.isAtEnd() && peek().kind != TokenKind::Eof) {
        if (peek().kind == TokenKind::Newline) break;

        if (peek().kind == TokenKind::Identifier && !peek().text.empty() && peek().text[0] == '^') {
            std::string id(peek().text);
            consume();
            block.successorIds.push_back(std::move(id));

            if (peek().kind == TokenKind::Comma) {
                consume();
                continue;
            }
            // Next token could be newline or next block
            break;
        } else {
            emitError("Expected block reference (^blockId)");
            return false;
        }
    }
    return true;
}

std::optional<std::string> IRParser::parseBlockLabel() {
    // Format: ^blockId:  (blockId stored without ^ for BasicBlock::label)
    if (peek().kind != TokenKind::Identifier || peek().text.empty() || peek().text[0] != '^') {
        return std::nullopt;
    }
    std::string blockId(peek().text);
    consume();

    if (peek().kind != TokenKind::Colon) {
        return std::nullopt;
    }
    consume();  // :

    if (blockId.size() > 1) blockId = blockId.substr(1);  // strip ^
    return blockId;
}

bool IRParser::parseDestRegisters(ParsedInstruction& inst) {
    // Parse comma-separated destination registers before '='
    // Format: v[14:17], s[0], DS_WRITE[0]

    while (!lexer.isAtEnd() && peek().kind != TokenKind::Eof) {
        // Check if we hit the '=' sign (end of dest list)
        if (peek().kind == TokenKind::Equal) {
            break;
        }

        auto regOpt = parseRegister();
        if (!regOpt) {
            // Not a valid register, might be end of dest list or error
            break;
        }

        inst.destRegs.push_back(*regOpt);

        // Check for comma (more regs coming) or '=' (done)
        if (peek().kind == TokenKind::Comma) {
            consume();
            continue;
        } else if (peek().kind == TokenKind::Equal) {
            break;
        } else {
            emitError("Expected ',' or '=' after destination register");
            return false;
        }
    }

    return !inst.destRegs.empty();
}

std::optional<std::string> IRParser::parseOperation() {
    // Parse quoted operation name: "Stinkytofu.ds_load_b128" or "ds_load_b128"
    if (peek().kind != TokenKind::QuotedString) {
        emitError("Expected quoted operation name");
        return std::nullopt;
    }

    const Token& tok = consume();
    std::string opStr(tok.text);

    // Remove quotes from the string
    if (opStr.length() >= 2 && opStr.front() == '"' && opStr.back() == '"') {
        opStr = opStr.substr(1, opStr.length() - 2);
    }

    // Split by dot '.' if present: "ir_namespace.ds_load_b128" -> namespace="ir_namespace",
    // opStr="ds_load_b128"
    size_t dotPos = opStr.find('.');
    if (dotPos != std::string::npos) {
        // TODO: inst.irNamespace = opStr.substr(0, dotPos);
        opStr = opStr.substr(dotPos + 1);
    }

    return opStr;
}

bool IRParser::parseOperands(ParsedInstruction& inst) {
    // Parse source operands in parentheses: (v[10], s[48], BARRIER[0])
    if (peek().kind != TokenKind::LeftParen) {
        // Operands are optional in some cases
        return false;
    }

    consume();  // consume '('

    // Parse comma-separated operands
    while (!lexer.isAtEnd() && peek().kind != TokenKind::Eof) {
        // Check for end of operand list
        if (peek().kind == TokenKind::RightParen) {
            consume();
            return true;
        }

        auto regOpt = parseRegister();
        if (!regOpt) {
            emitError("Expected register or literal in operand list");
            return false;
        }

        inst.srcRegs.push_back(*regOpt);

        // Check for comma or end of list
        if (peek().kind == TokenKind::Comma) {
            consume();
            continue;
        } else if (peek().kind == TokenKind::RightParen) {
            consume();
            return true;
        } else {
            emitError("Expected ',' or ')' after operand");
            return false;
        }
    }

    emitError("Unexpected end of file in operand list");
    return false;
}

bool IRParser::parseAttributes(ParsedInstruction& inst) {
    // Parse attributes in braces: { issueCycles = 4, latencyCycles = 56 }
    if (peek().kind != TokenKind::LeftBrace) {
        return false;
    }

    consume();  // consume '{'

    bool parsedIssueCycles = false;
    bool parsedLatencyCycles = false;

    // Parse comma-separated attributes
    while (!lexer.isAtEnd() && peek().kind != TokenKind::Eof) {
        // Check for end of attributes
        if (peek().kind == TokenKind::RightBrace) {
            consume();
            return parsedIssueCycles && parsedLatencyCycles;
        }

        // Parse attribute name
        if (peek().kind != TokenKind::Identifier) {
            emitError("Expected attribute name");
            return false;
        }

        const Token& attrName = consume();
        std::string attrStr(attrName.text);

        // Expect '='
        if (!expect(TokenKind::Equal, "Expected '=' after attribute name")) {
            return false;
        }

        // Parse attribute value
        if (attrStr.size() >= 4 && attrStr.substr(0, 4) == "mod." &&
            peek().kind == TokenKind::LeftBrace) {
            // Modifier dict: mod.X = { field = value, ... }
            auto dictOpt = parseModifierDict();
            if (!dictOpt) {
                return false;
            }
            inst.modifiers[attrStr] = std::move(*dictOpt);
        } else if (peek().kind == TokenKind::IntegerLiteral) {
            const Token& valueTok = consume();
            auto value = safeStoi(std::string(valueTok.text));
            if (!value) {
                emitError("Attribute value out of range or invalid: " + std::string(valueTok.text));
                continue;
            }

            if (attrStr == "issueCycles") {
                inst.issueCycles = *value;
                parsedIssueCycles = true;
            } else if (attrStr == "latencyCycles") {
                inst.latencyCycles = *value;
                parsedLatencyCycles = true;
            } else {
                emitWarning("Unknown attribute: " + attrStr, attrName.line, attrName.column);
            }
        } else if (peek().kind == TokenKind::FloatLiteral) {
            const Token& valueTok = consume();
            // For now, just store as integer (truncate)
            double value = std::stod(std::string(valueTok.text));

            if (attrStr == "issueCycles") {
                inst.issueCycles = static_cast<int>(value);
                parsedIssueCycles = true;
            } else if (attrStr == "latencyCycles") {
                inst.latencyCycles = static_cast<int>(value);
                parsedLatencyCycles = true;
            } else {
                emitWarning("Unknown attribute: " + attrStr, attrName.line, attrName.column);
            }
        } else {
            emitError("Expected integer, float, or modifier dict for attribute");
            return false;
        }

        // Check for comma or end of attributes
        if (peek().kind == TokenKind::Comma) {
            consume();
            continue;
        } else if (peek().kind == TokenKind::RightBrace) {
            continue;  // Will be consumed in next iteration
        } else {
            emitError("Expected ',' or '}' after attribute value");
            return false;
        }
    }

    emitError("Unexpected end of file in attribute list");
    return false;
}

std::optional<std::unordered_map<std::string, std::string>> IRParser::parseModifierDict() {
    // Format: { field = value, field = value, ... }
    if (peek().kind != TokenKind::LeftBrace) {
        emitError("Expected '{' for modifier dict");
        return std::nullopt;
    }
    consume();  // consume '{'

    std::unordered_map<std::string, std::string> dict;

    while (!lexer.isAtEnd() && peek().kind != TokenKind::Eof) {
        if (peek().kind == TokenKind::RightBrace) {
            consume();
            return dict;
        }

        if (peek().kind != TokenKind::Identifier) {
            emitError("Expected field name in modifier dict");
            return std::nullopt;
        }
        const Token& fieldTok = consume();
        std::string fieldName(fieldTok.text);

        if (!expect(TokenKind::Equal, "Expected '=' after field name in modifier dict")) {
            return std::nullopt;
        }

        auto valueOpt = parseAttributeValue();
        if (!valueOpt) {
            return std::nullopt;
        }
        dict[fieldName] = std::move(*valueOpt);

        if (peek().kind == TokenKind::Comma) {
            consume();
            continue;
        }
        if (peek().kind == TokenKind::RightBrace) {
            continue;  // Will be consumed in next iteration
        }
        emitError("Expected ',' or '}' after modifier dict field");
        return std::nullopt;
    }

    emitError("Unexpected end of file in modifier dict");
    return std::nullopt;
}

std::optional<std::string> IRParser::parseAttributeValue() {
    if (peek().kind == TokenKind::IntegerLiteral) {
        const Token& tok = consume();
        return std::string(tok.text);
    }
    if (peek().kind == TokenKind::HexLiteral) {
        const Token& tok = consume();
        return std::string(tok.text);
    }
    if (peek().kind == TokenKind::FloatLiteral) {
        const Token& tok = consume();
        return std::string(tok.text);
    }
    if (peek().kind == TokenKind::QuotedString) {
        const Token& tok = consume();
        std::string s(tok.text);
        if (s.size() >= 2 && s.front() == '"' && s.back() == '"') s = s.substr(1, s.size() - 2);
        return s;
    }
    if (peek().kind == TokenKind::KW_true) {
        consume();
        return "true";
    }
    if (peek().kind == TokenKind::KW_false) {
        consume();
        return "false";
    }
    if (peek().kind == TokenKind::Identifier) {
        const Token& tok = consume();
        std::string s(tok.text);
        if (s == "true" || s == "false") return s;
        return s;  // Allow other identifiers as string values
    }
    if (peek().kind == TokenKind::LeftBracket) {
        consume();
        std::string result = "[";
        bool first = true;
        while (!lexer.isAtEnd() && peek().kind != TokenKind::Eof) {
            if (peek().kind == TokenKind::RightBracket) {
                consume();
                result += "]";
                return result;
            }
            if (!first) {
                if (peek().kind != TokenKind::Comma) {
                    emitError("Expected ',' or ']' in attribute array value");
                    return std::nullopt;
                }
                consume();
                result += ", ";
            }
            first = false;

            TokenKind elemKind = peek().kind;
            if (elemKind == TokenKind::IntegerLiteral || elemKind == TokenKind::HexLiteral ||
                elemKind == TokenKind::FloatLiteral) {
                result += std::string(consume().text);
            } else if (elemKind == TokenKind::KW_true) {
                consume();
                result += "true";
            } else if (elemKind == TokenKind::KW_false) {
                consume();
                result += "false";
            } else if (elemKind == TokenKind::Identifier) {
                result += std::string(consume().text);
            } else {
                emitError("Expected scalar literal in attribute array value");
                return std::nullopt;
            }
        }
        emitError("Unterminated attribute array value");
        return std::nullopt;
    }
    emitError("Expected attribute value (integer, float, string, true, false, or array)");
    return std::nullopt;
}

std::optional<StinkyRegister> IRParser::parseRegister() {
    // Parse register references:
    // v[10], v[14:17], s[0], acc[0:15], BARRIER[0], SCC[0], DS_WRITE[0]
    // Literals: HexLiteral -> LiteralString (spelling preserved); decimal int / float -> numeric
    // literals.
    // QuotedString: used by st.asm_directive operands (e.g. ".set", "symbol").

    TokenKind kind = peek().kind;

    // Handle quoted string literals (e.g. directive operands)
    if (kind == TokenKind::QuotedString) {
        const Token& tok = consume();
        std::string s(tok.text);
        if (s.size() >= 2 && s.front() == '"' && s.back() == '"') s = s.substr(1, s.size() - 2);
        return StinkyRegister(s);
    }

    // Handle integer literals
    if (kind == TokenKind::IntegerLiteral) {
        const Token& tok = consume();
        std::string num(tok.text);
        // Recover "0xN" when lexed as IntegerLiteral "0" + Identifier "xN" (or NBSP/other gap).
        if (num == "0" && peek().kind == TokenKind::Identifier) {
            std::string id(peek().text);
            if (isHexSuffixIdentifier(id)) {
                consume();
                return StinkyRegister(std::string("0") + std::string(id));
            }
        }
        auto val = safeStoi(num);
        if (!val) {
            emitError("Integer literal out of range or invalid: " + num);
            return std::nullopt;
        }
        return StinkyRegister(*val);
    }

    // Hex immediates: preserve spelling as a literal string (e.g. 0xffff0000).
    if (kind == TokenKind::HexLiteral) {
        const Token& tok = consume();
        return StinkyRegister(std::string(tok.text));
    }

    // Handle float literals
    if (kind == TokenKind::FloatLiteral) {
        const Token& tok = consume();
        // For now, convert to integer (truncate)
        double val = std::stod(std::string(tok.text));
        return StinkyRegister(static_cast<double>(val));
    }

    // Must be an identifier (register type like 'v', 's', 'acc', 'BARRIER', etc.)
    if (kind != TokenKind::Identifier) {
        return std::nullopt;
    }

    const Token& regTypeTok = consume();
    std::string regTypeStr(regTypeTok.text);

    // Handle label_*: label reference (e.g. label_LoopEndL, label_LoopBeginL)
    if (regTypeStr.size() >= 6 && regTypeStr.substr(0, 6) == "label_") {
        return StinkyRegister(regTypeStr);
    } else if (regTypeStr == "BufferLimit") {
        return StinkyRegister(regTypeStr);
    } else if (regTypeStr == "off") {
        // MUBUF "off" keyword: vaddr field with no address register.
        return StinkyRegister("off");
    }

    // Handle "v10" / "s5" / "acc12" etc.: identifier = regType + digits (no space)
    // Try longest prefix first so "acc10" is "acc" + "10" not "a" + "cc10"
    for (size_t prefixLen = regTypeStr.size(); prefixLen >= 1; --prefixLen) {
        std::string prefix = regTypeStr.substr(0, prefixLen);
        std::string suffix = regTypeStr.substr(prefixLen);
        RegType rt = stringToRegType(prefix);
        if (isValidRegType(rt) && !suffix.empty() &&
            std::all_of(suffix.begin(), suffix.end(),
                        [](unsigned char c) { return std::isdigit(c); })) {
            auto idx = safeStoi(suffix);
            if (idx) return StinkyRegister(rt, *idx, 1);
        }
    }

    RegType regType = stringToRegType(regTypeStr);

    // Validate register type
    if (!isValidRegType(regType)) {
        emitError("Unknown register type: " + regTypeStr);
        return std::nullopt;
    }

    // Check for format: v 12 (separate tokens - reg type then integer)
    if (peek().kind == TokenKind::IntegerLiteral) {
        const Token& idxTok = consume();
        auto idx = safeStoi(std::string(idxTok.text));
        if (!idx) {
            emitError("Register index out of range or invalid: " + std::string(idxTok.text));
            return std::nullopt;
        }
        return StinkyRegister(regType, *idx, 1);  // Single element
    }

    // Check for format: v[12] or v[10:13]
    if (peek().kind != TokenKind::LeftBracket) {
        // Not a register, might be a plain identifier
        return StinkyRegister(regTypeStr);  // Store as string literal
    }

    consume();  // consume '['

    // Parse start index
    if (peek().kind != TokenKind::IntegerLiteral) {
        emitError("Expected integer register index");
        return std::nullopt;
    }

    const Token& startTok = consume();
    auto startIdx = safeStoi(std::string(startTok.text));
    if (!startIdx) {
        emitError("Register start index out of range or invalid: " + std::string(startTok.text));
        return std::nullopt;
    }
    int endIdx = *startIdx;

    // Check for range notation [start:end]
    if (peek().kind == TokenKind::Colon) {
        consume();

        if (peek().kind != TokenKind::IntegerLiteral) {
            emitError("Expected integer register index after ':'");
            return std::nullopt;
        }

        const Token& endTok = consume();
        auto endIdxOpt = safeStoi(std::string(endTok.text));
        if (!endIdxOpt) {
            emitError("Register end index out of range or invalid: " + std::string(endTok.text));
            return std::nullopt;
        }
        endIdx = *endIdxOpt;
    }

    // Expect ']'
    if (peek().kind != TokenKind::RightBracket) {
        emitError("Expected ']' after register index");
        return std::nullopt;
    }
    consume();

    // Calculate register count
    int regNum = endIdx - *startIdx + 1;
    if (regNum <= 0) {
        emitError("Invalid register range");
        return std::nullopt;
    }

    return StinkyRegister(regType, *startIdx, regNum);
}

bool IRParser::expect(TokenKind kind, const std::string& message) {
    if (check(kind)) {
        consume();
        return true;
    }

    emitError(message);
    return false;
}

void IRParser::skipNewlines() {
    while (peek().kind == TokenKind::Newline) {
        consume();
    }
}

void IRParser::emitError(const std::string& message, unsigned line, unsigned column) {
    diagnostics.emplace_back(Diagnostic::Level::Error, message, line, column);
    hadError = true;
}

void IRParser::emitError(const std::string& message) {
    const Token& tok = peek();
    emitError(message, tok.line, tok.column);
}

void IRParser::emitWarning(const std::string& message, unsigned line, unsigned column) {
    diagnostics.emplace_back(Diagnostic::Level::Warning, message, line, column);
}

}  // namespace

namespace stinkytofu {
ParseResult parseSourceStringWithDiagnostics(const std::string& sourceStr) {
    ParseResult result;

    IRLexer lexer(sourceStr);
    lexer.lex();

    IRParser parser(lexer);
    result.parsedFunction = parser.parse();
    result.diagnostics = parser.getDiagnostics();

    return result;
}

MultiParseResult parseAllSourceStringsWithDiagnostics(const std::string& sourceStr) {
    MultiParseResult result;

    IRLexer lexer(sourceStr);
    lexer.lex();

    IRParser parser(lexer);
    result.functions = parser.parseAllFunctions();
    result.diagnostics = parser.getDiagnostics();

    return result;
}

std::vector<ParsedInstruction> ParseResult::getInstructions() const {
    std::vector<ParsedInstruction> instructions;
    if (parsedFunction && !parsedFunction->blocks.empty()) {
        for (const auto& block : parsedFunction->blocks) {
            for (const auto& instruction : block->instructions) {
                instructions.push_back(*instruction);
            }
        }
    }
    return instructions;
}

}  // namespace stinkytofu

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
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
/// Token kinds for the IR lexer (MLIR-style format).
enum class TokenKind {
    // Structural tokens
    Eof,           // End of file
    Newline,       // Line break
    Colon,         // ':'
    LeftBracket,   // '['
    RightBracket,  // ']'
    LeftParen,     // '('
    RightParen,    // ')'
    LeftBrace,     // '{'
    RightBrace,    // '}'
    Comma,         // ','
    Equal,         // '='

    // String literals
    QuotedString,  // "Stinkytofu.operation"

    // Literals
    Identifier,      // register types, labels, identifiers
    IntegerLiteral,  // Integer constants
    HexLiteral,      // Hexadecimal constants (0x...)
    FloatLiteral,    // Floating point constants

    // Pattern keywords (for pattern definition files)
    KW_peephole,        // 'peephole' (pattern type)
    KW_ir,              // 'ir' (high-level IR pattern type)
    KW_intrinsic,       // 'intrinsic' (intrinsic definition)
    KW_pattern,         // 'pattern'
    KW_match,           // 'match'
    KW_constraints,     // 'constraints'
    KW_rewrite,         // 'rewrite'
    KW_replace,         // 'replace'
    KW_remove,          // 'remove'
    KW_with,            // 'with'
    KW_arguments,       // 'arguments'
    KW_body,            // 'body'
    KW_comment,         // 'comment'
    KW_python_binding,  // 'python_binding'
    KW_call,            // 'call' (function call in intrinsic)
    KW_true,            // 'true'
    KW_false,           // 'false'

    // Invalid token
    Unknown
};

/// Represents a single token in the IR stream.
class Token {
   public:
    TokenKind kind;
    std::string_view text;
    unsigned line;
    unsigned column;

    Token(TokenKind k, std::string_view t, unsigned l, unsigned c)
        : kind(k), text(t), line(l), column(c) {}

    Token() : kind(TokenKind::Unknown), line(0), column(0) {}

    bool is(TokenKind k) const {
        return kind == k;
    }

    bool isNot(TokenKind k) const {
        return kind != k;
    }

    bool isOneOf(TokenKind k1, TokenKind k2) const {
        return is(k1) || is(k2);
    }

    template <typename... Ts>
    bool isOneOf(TokenKind k1, TokenKind k2, Ts... ks) const {
        return is(k1) || isOneOf(k2, ks...);
    }

    std::string_view getText() const {
        return text;
    }

    unsigned getLine() const {
        return line;
    }

    unsigned getColumn() const {
        return column;
    }
};

/// Lexer for the StinkyTofu IR text format.
/// Converts the input text into a stream of tokens.
class STINKYTOFU_EXPORT IRLexer {
   private:
    const char* bufferStart;
    const char* bufferEnd;
    const char* curPtr;
    unsigned currentLine;
    unsigned currentColumn;
    std::vector<Token> tokens;
    size_t currentTokenIndex;

   public:
    /// Initialize the lexer with the input buffer.
    IRLexer(const char* start, const char* end);
    IRLexer(const std::string& input);

    /// Lex the entire input and store tokens.
    void lex();

    /// Get the next token without consuming it.
    const Token& peek() const;

    /// Get the token at offset positions ahead without consuming.
    /// For example, peekAhead(1) returns the token after peek().
    const Token& peekAhead(size_t offset = 1) const;

    /// Get the next token and advance.
    const Token& consume();

    /// Check if we're at the end of the token stream.
    bool isAtEnd() const;

    /// Get all tokens (useful for debugging).
    const std::vector<Token>& getAllTokens() const {
        return tokens;
    }

   private:
    /// Lex a single token.
    Token lexToken();

    /// Skip whitespace (but not newlines).
    void skipWhitespace();

    /// Skip whitespace including newlines.
    void skipWhitespaceAndNewlines();

    /// Lex an identifier or keyword.
    Token lexIdentifier();

    /// Lex a numeric literal.
    Token lexNumber();

    /// Classify an identifier string as a keyword or identifier.
    /// Returns the appropriate TokenKind (KW_* or Identifier).
    static TokenKind classifyIdentifier(std::string_view text);

    /// Check if a character is whitespace (excluding newline).
    static bool isWhitespace(char c);

    /// Check if a character can start an identifier.
    static bool isIdentifierStart(char c);

    /// Check if a character can continue an identifier.
    static bool isIdentifierContinue(char c);

    /// Check if a character is a digit.
    static bool isDigit(char c);

    /// Check if a character is a hex digit.
    static bool isHexDigit(char c);

    /// Get current character without advancing.
    char peekChar() const;

    /// Get character at offset ahead without advancing.
    char peekAheadChar(size_t offset = 1) const;

    /// Get current character and advance.
    char consumeChar();

    /// Check if at end of buffer.
    bool isAtBufferEnd() const;
};

}  // namespace stinkytofu

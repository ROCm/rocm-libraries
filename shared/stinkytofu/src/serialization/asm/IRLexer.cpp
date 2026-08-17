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

#include "IRLexer.hpp"

#include <cctype>
#include <cstring>

namespace stinkytofu {
//----------------------------------------------------------------------
// IRLexer implementation
//----------------------------------------------------------------------

IRLexer::IRLexer(const char* start, const char* end)
    : bufferStart(start),
      bufferEnd(end),
      curPtr(start),
      currentLine(1),
      currentColumn(1),
      currentTokenIndex(0) {}

IRLexer::IRLexer(const std::string& input) : IRLexer(input.data(), input.data() + input.size()) {}

void IRLexer::lex() {
    tokens.clear();
    currentTokenIndex = 0;

    while (!isAtBufferEnd()) {
        Token tok = lexToken();
        tokens.push_back(tok);

        if (tok.kind == TokenKind::Eof) break;
    }

    // Ensure we have at least an EOF token
    if (tokens.empty() || tokens.back().kind != TokenKind::Eof) {
        tokens.push_back(Token(TokenKind::Eof, "", currentLine, currentColumn));
    }
}

const Token& IRLexer::peek() const {
    if (currentTokenIndex < tokens.size()) {
        return tokens[currentTokenIndex];
    }
    // Return EOF if past the end; line/column=1 so error diagnostics are valid
    static Token eofToken(TokenKind::Eof, "", 1, 1);
    return eofToken;
}

const Token& IRLexer::peekAhead(size_t offset) const {
    size_t lookAheadIndex = currentTokenIndex + offset;
    if (lookAheadIndex < tokens.size()) {
        return tokens[lookAheadIndex];
    }
    // Return EOF if past the end; line/column=1 so error diagnostics are valid
    static Token eofToken(TokenKind::Eof, "", 1, 1);
    return eofToken;
}

const Token& IRLexer::consume() {
    if (currentTokenIndex < tokens.size()) {
        return tokens[currentTokenIndex++];
    }
    // Return EOF if past the end
    static Token eofToken(TokenKind::Eof, "", 0, 0);
    return eofToken;
}

bool IRLexer::isAtEnd() const {
    return currentTokenIndex >= tokens.size() || peek().kind == TokenKind::Eof;
}

Token IRLexer::lexToken() {
    skipWhitespace();

    if (isAtBufferEnd()) {
        return Token(TokenKind::Eof, "", currentLine, currentColumn);
    }

    const char* tokenStart = curPtr;
    unsigned tokenLine = currentLine;
    unsigned tokenColumn = currentColumn;

    char c = consumeChar();

    switch (c) {
        case '\n':
        case '\r':
            // Handle newline
            if (c == '\r' && peekChar() == '\n') {
                consumeChar();  // Consume the \n after \r
            }
            currentLine++;
            currentColumn = 1;
            return Token(TokenKind::Newline, std::string_view(tokenStart, curPtr - tokenStart),
                         tokenLine, tokenColumn);

        case ':':
            return Token(TokenKind::Colon, std::string_view(tokenStart, 1), tokenLine, tokenColumn);

        case '[':
            return Token(TokenKind::LeftBracket, std::string_view(tokenStart, 1), tokenLine,
                         tokenColumn);

        case ']':
            return Token(TokenKind::RightBracket, std::string_view(tokenStart, 1), tokenLine,
                         tokenColumn);

        case '(':
            return Token(TokenKind::LeftParen, std::string_view(tokenStart, 1), tokenLine,
                         tokenColumn);

        case ')':
            return Token(TokenKind::RightParen, std::string_view(tokenStart, 1), tokenLine,
                         tokenColumn);

        case '{':
            return Token(TokenKind::LeftBrace, std::string_view(tokenStart, 1), tokenLine,
                         tokenColumn);

        case '}':
            return Token(TokenKind::RightBrace, std::string_view(tokenStart, 1), tokenLine,
                         tokenColumn);

        case ',':
            return Token(TokenKind::Comma, std::string_view(tokenStart, 1), tokenLine, tokenColumn);

        case '=':
            return Token(TokenKind::Equal, std::string_view(tokenStart, 1), tokenLine, tokenColumn);

        case '"':
            // Quoted string
            while (peekChar() != '"' && !isAtBufferEnd()) {
                if (peekChar() == '\\') {
                    consumeChar();                        // consume backslash
                    if (!isAtBufferEnd()) consumeChar();  // consume escaped char
                } else {
                    consumeChar();
                }
            }
            if (peekChar() == '"') consumeChar();  // consume closing quote
            return Token(TokenKind::QuotedString, std::string_view(tokenStart, curPtr - tokenStart),
                         tokenLine, tokenColumn);

        case '0':
            // Check for hex literal
            if (peekChar() == 'x' || peekChar() == 'X') {
                consumeChar();  // consume 'x'
                while (isHexDigit(peekChar())) {
                    consumeChar();
                }
                return Token(TokenKind::HexLiteral,
                             std::string_view(tokenStart, curPtr - tokenStart), tokenLine,
                             tokenColumn);
            }
            // Fall through to number handling
            [[fallthrough]];

        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            // Numeric literal
            while (isDigit(peekChar())) {
                consumeChar();
            }
            // Check for decimal point (float)
            if (peekChar() == '.') {
                consumeChar();  // consume '.'
                while (isDigit(peekChar())) {
                    consumeChar();
                }
                // Check for scientific notation (e.g., 1.23e-4, 1.23E+10)
                if (peekChar() == 'e' || peekChar() == 'E') {
                    consumeChar();  // consume 'e' or 'E'
                    // Optional sign
                    if (peekChar() == '+' || peekChar() == '-') {
                        consumeChar();
                    }
                    // Must have at least one digit after 'e'
                    if (!isDigit(peekChar())) {
                        // Error: malformed scientific notation
                        // For now, return what we have as FloatLiteral
                        // A proper error would be better, but this prevents silent misparse
                        return Token(TokenKind::FloatLiteral,
                                     std::string_view(tokenStart, curPtr - tokenStart), tokenLine,
                                     tokenColumn);
                    }
                    while (isDigit(peekChar())) {
                        consumeChar();
                    }
                }
                return Token(TokenKind::FloatLiteral,
                             std::string_view(tokenStart, curPtr - tokenStart), tokenLine,
                             tokenColumn);
            }
            // Check for integer scientific notation (e.g., 123e4)
            if (peekChar() == 'e' || peekChar() == 'E') {
                consumeChar();  // consume 'e' or 'E'
                // Optional sign
                if (peekChar() == '+' || peekChar() == '-') {
                    consumeChar();
                }
                // Must have at least one digit after 'e'
                if (!isDigit(peekChar())) {
                    // Error: malformed scientific notation
                    // Return as integer and let the 'e' be parsed separately
                    // We need to backtrack
                    curPtr = tokenStart;
                    while (isDigit(peekChar())) {
                        consumeChar();
                    }
                    return Token(TokenKind::IntegerLiteral,
                                 std::string_view(tokenStart, curPtr - tokenStart), tokenLine,
                                 tokenColumn);
                }
                while (isDigit(peekChar())) {
                    consumeChar();
                }
                // Integer scientific notation is treated as float
                return Token(TokenKind::FloatLiteral,
                             std::string_view(tokenStart, curPtr - tokenStart), tokenLine,
                             tokenColumn);
            }
            return Token(TokenKind::IntegerLiteral,
                         std::string_view(tokenStart, curPtr - tokenStart), tokenLine, tokenColumn);

        case '-':
            // Negative hex: -0x1234 (must not be tokenized as '-' + '0' + 'x...')
            if (peekChar() == '0' && (peekAheadChar(1) == 'x' || peekAheadChar(1) == 'X')) {
                consumeChar();  // '0'
                consumeChar();  // 'x' or 'X'
                while (isHexDigit(peekChar())) {
                    consumeChar();
                }
                return Token(TokenKind::HexLiteral,
                             std::string_view(tokenStart, curPtr - tokenStart), tokenLine,
                             tokenColumn);
            }
            // Could be negative number
            if (isDigit(peekChar())) {
                while (isDigit(peekChar())) {
                    consumeChar();
                }
                // Check for decimal point (float)
                if (peekChar() == '.') {
                    consumeChar();  // consume '.'
                    while (isDigit(peekChar())) {
                        consumeChar();
                    }
                    // Check for scientific notation (e.g., -1.23e-4, -1.23E+10)
                    if (peekChar() == 'e' || peekChar() == 'E') {
                        consumeChar();  // consume 'e' or 'E'
                        // Optional sign
                        if (peekChar() == '+' || peekChar() == '-') {
                            consumeChar();
                        }
                        // Must have at least one digit after 'e'
                        if (!isDigit(peekChar())) {
                            // Error: malformed scientific notation
                            // Return what we have as FloatLiteral
                            return Token(TokenKind::FloatLiteral,
                                         std::string_view(tokenStart, curPtr - tokenStart),
                                         tokenLine, tokenColumn);
                        }
                        while (isDigit(peekChar())) {
                            consumeChar();
                        }
                    }
                    return Token(TokenKind::FloatLiteral,
                                 std::string_view(tokenStart, curPtr - tokenStart), tokenLine,
                                 tokenColumn);
                }
                // Check for integer scientific notation (e.g., -123e4)
                if (peekChar() == 'e' || peekChar() == 'E') {
                    consumeChar();  // consume 'e' or 'E'
                    // Optional sign
                    if (peekChar() == '+' || peekChar() == '-') {
                        consumeChar();
                    }
                    // Must have at least one digit after 'e'
                    if (!isDigit(peekChar())) {
                        // Error: malformed scientific notation
                        // Backtrack to just the integer part
                        curPtr = tokenStart;
                        consumeChar();  // consume '-'
                        while (isDigit(peekChar())) {
                            consumeChar();
                        }
                        return Token(TokenKind::IntegerLiteral,
                                     std::string_view(tokenStart, curPtr - tokenStart), tokenLine,
                                     tokenColumn);
                    }
                    while (isDigit(peekChar())) {
                        consumeChar();
                    }
                    // Integer scientific notation is treated as float
                    return Token(TokenKind::FloatLiteral,
                                 std::string_view(tokenStart, curPtr - tokenStart), tokenLine,
                                 tokenColumn);
                }
                return Token(TokenKind::IntegerLiteral,
                             std::string_view(tokenStart, curPtr - tokenStart), tokenLine,
                             tokenColumn);
            }
            // Otherwise treat as part of identifier
            while (isIdentifierContinue(peekChar())) {
                consumeChar();
            }
            {
                std::string_view text(tokenStart, curPtr - tokenStart);
                TokenKind kind = classifyIdentifier(text);
                return Token(kind, text, tokenLine, tokenColumn);
            }

        case '/':
            // Check for C-style single-line comment: //
            if (peekChar() == '/') {
                // This is a single-line comment, skip until end of line
                consumeChar();  // consume second '/'
                while (!isAtBufferEnd() && peekChar() != '\n' && peekChar() != '\r') {
                    consumeChar();
                }
                // Recursively call lexToken to get the next real token
                return lexToken();
            }
            // Check for C-style block comment: /* */
            else if (peekChar() == '*') {
                // This is a block comment, skip until */
                consumeChar();  // consume '*'

                while (!isAtBufferEnd()) {
                    if (peekChar() == '*' && peekAheadChar() == '/') {
                        consumeChar();  // consume '*'
                        consumeChar();  // consume '/'
                        break;
                    }
                    // Track line numbers inside block comments
                    if (peekChar() == '\n') {
                        currentLine++;
                        currentColumn = 1;
                    }
                    consumeChar();
                }
                // Recursively call lexToken to get the next real token
                return lexToken();
            }
            // Not a comment, just a forward slash (unknown token)
            return Token(TokenKind::Unknown, std::string_view(tokenStart, 1), tokenLine,
                         tokenColumn);

        default:
            if (isIdentifierStart(c)) {
                // Identifier or keyword
                while (isIdentifierContinue(peekChar())) {
                    consumeChar();
                }

                std::string_view text(tokenStart, curPtr - tokenStart);
                TokenKind kind = classifyIdentifier(text);
                return Token(kind, text, tokenLine, tokenColumn);
            }

            // Unknown character
            return Token(TokenKind::Unknown, std::string_view(tokenStart, 1), tokenLine,
                         tokenColumn);
    }
}

TokenKind IRLexer::classifyIdentifier(std::string_view text) {
    // Check for pattern keywords
    // This is a simple string comparison - efficient enough for a small set of keywords
    if (text == "peephole") return TokenKind::KW_peephole;
    if (text == "ir") return TokenKind::KW_ir;
    if (text == "intrinsic") return TokenKind::KW_intrinsic;
    if (text == "pattern") return TokenKind::KW_pattern;
    if (text == "match") return TokenKind::KW_match;
    if (text == "constraints") return TokenKind::KW_constraints;
    if (text == "rewrite") return TokenKind::KW_rewrite;
    if (text == "replace") return TokenKind::KW_replace;
    if (text == "remove") return TokenKind::KW_remove;
    if (text == "with") return TokenKind::KW_with;
    if (text == "arguments") return TokenKind::KW_arguments;
    if (text == "body") return TokenKind::KW_body;
    if (text == "comment") return TokenKind::KW_comment;
    if (text == "python_binding") return TokenKind::KW_python_binding;
    if (text == "call") return TokenKind::KW_call;
    if (text == "true") return TokenKind::KW_true;
    if (text == "false") return TokenKind::KW_false;

    // Not a keyword, return as identifier
    return TokenKind::Identifier;
}

void IRLexer::skipWhitespace() {
    while (!isAtBufferEnd()) {
        char c = peekChar();
        if (c == ' ' || c == '\t') {
            consumeChar();
        }
        // Handle single-line comments: // or #
        else if ((c == '/' && peekAheadChar() == '/') || c == '#') {
            // Skip until end of line
            consumeChar();
            if (c == '/') consumeChar();  // consume second '/'
            while (!isAtBufferEnd() && peekChar() != '\n' && peekChar() != '\r') {
                consumeChar();
            }
            // Don't consume the newline - let it be processed normally
        }
        // Handle C-style block comments: /* */
        else if (c == '/' && peekAheadChar() == '*') {
            // Skip until */
            consumeChar();  // consume '/'
            consumeChar();  // consume '*'

            while (!isAtBufferEnd()) {
                if (peekChar() == '*' && peekAheadChar() == '/') {
                    consumeChar();  // consume '*'
                    consumeChar();  // consume '/'
                    break;
                }
                // Track line numbers inside block comments
                if (peekChar() == '\n') {
                    currentLine++;
                    currentColumn = 1;
                }
                consumeChar();
            }
            // Continue skipping whitespace after the comment
        } else {
            break;
        }
    }
}

void IRLexer::skipWhitespaceAndNewlines() {
    while (!isAtBufferEnd()) {
        char c = peekChar();
        if (isWhitespace(c) || c == '\n' || c == '\r') {
            if (c == '\n') {
                currentLine++;
                currentColumn = 0;
            }
            consumeChar();
        } else {
            break;
        }
    }
}

bool IRLexer::isWhitespace(char c) {
    return c == ' ' || c == '\t';
}

bool IRLexer::isIdentifierStart(char c) {
    // ^ for block IDs (^entry), @ for symbol names (@func)
    return std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == '^' || c == '@';
}

bool IRLexer::isIdentifierContinue(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' || c == '+';
}

bool IRLexer::isDigit(char c) {
    return std::isdigit(static_cast<unsigned char>(c));
}

bool IRLexer::isHexDigit(char c) {
    return std::isxdigit(static_cast<unsigned char>(c));
}

char IRLexer::peekChar() const {
    if (isAtBufferEnd()) return '\0';
    return *curPtr;
}

char IRLexer::peekAheadChar(size_t offset) const {
    const char* lookAhead = curPtr + offset;
    if (lookAhead >= bufferEnd) return '\0';
    return *lookAhead;
}

char IRLexer::consumeChar() {
    if (isAtBufferEnd()) return '\0';
    char c = *curPtr++;
    currentColumn++;
    return c;
}

bool IRLexer::isAtBufferEnd() const {
    return curPtr >= bufferEnd;
}

}  // namespace stinkytofu

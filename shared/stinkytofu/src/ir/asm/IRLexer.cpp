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

#include "IRLexer.hpp"

#include <cctype>
#include <cstring>

namespace stinkytofu
{
    //----------------------------------------------------------------------
    // IRLexer implementation
    //----------------------------------------------------------------------

    IRLexer::IRLexer(const char* start, const char* end)
        : bufferStart(start)
        , bufferEnd(end)
        , curPtr(start)
        , currentLine(1)
        , currentColumn(1)
        , currentTokenIndex(0)
    {
    }

    IRLexer::IRLexer(const std::string& input)
        : IRLexer(input.data(), input.data() + input.size())
    {
    }

    void IRLexer::lex()
    {
        tokens.clear();
        currentTokenIndex = 0;

        while(!isAtBufferEnd())
        {
            Token tok = lexToken();
            tokens.push_back(tok);

            if(tok.kind == TokenKind::Eof)
                break;
        }

        // Ensure we have at least an EOF token
        if(tokens.empty() || tokens.back().kind != TokenKind::Eof)
        {
            tokens.push_back(Token(TokenKind::Eof, "", currentLine, currentColumn));
        }
    }

    const Token& IRLexer::peek() const
    {
        if(currentTokenIndex < tokens.size())
        {
            return tokens[currentTokenIndex];
        }
        // Return EOF if past the end
        static Token eofToken(TokenKind::Eof, "", 0, 0);
        return eofToken;
    }

    const Token& IRLexer::peekAhead(size_t offset) const
    {
        size_t lookAheadIndex = currentTokenIndex + offset;
        if(lookAheadIndex < tokens.size())
        {
            return tokens[lookAheadIndex];
        }
        // Return EOF if past the end
        static Token eofToken(TokenKind::Eof, "", 0, 0);
        return eofToken;
    }

    const Token& IRLexer::consume()
    {
        if(currentTokenIndex < tokens.size())
        {
            return tokens[currentTokenIndex++];
        }
        // Return EOF if past the end
        static Token eofToken(TokenKind::Eof, "", 0, 0);
        return eofToken;
    }

    bool IRLexer::isAtEnd() const
    {
        return currentTokenIndex >= tokens.size() || peek().kind == TokenKind::Eof;
    }

    Token IRLexer::lexToken()
    {
        skipWhitespace();

        if(isAtBufferEnd())
        {
            return Token(TokenKind::Eof, "", currentLine, currentColumn);
        }

        const char* tokenStart  = curPtr;
        unsigned    tokenLine   = currentLine;
        unsigned    tokenColumn = currentColumn;

        char c = consumeChar();

        switch(c)
        {
        case '\n':
        case '\r':
            // Handle newline
            if(c == '\r' && peekChar() == '\n')
            {
                consumeChar(); // Consume the \n after \r
            }
            currentLine++;
            currentColumn = 1;
            return Token(TokenKind::Newline,
                         std::string_view(tokenStart, curPtr - tokenStart),
                         tokenLine,
                         tokenColumn);

        case ':':
            return Token(TokenKind::Colon, std::string_view(tokenStart, 1), tokenLine, tokenColumn);

        case '[':
            return Token(
                TokenKind::LeftBracket, std::string_view(tokenStart, 1), tokenLine, tokenColumn);

        case ']':
            return Token(
                TokenKind::RightBracket, std::string_view(tokenStart, 1), tokenLine, tokenColumn);

        case '(':
            return Token(
                TokenKind::LeftParen, std::string_view(tokenStart, 1), tokenLine, tokenColumn);

        case ')':
            return Token(
                TokenKind::RightParen, std::string_view(tokenStart, 1), tokenLine, tokenColumn);

        case '{':
            return Token(
                TokenKind::LeftBrace, std::string_view(tokenStart, 1), tokenLine, tokenColumn);

        case '}':
            return Token(
                TokenKind::RightBrace, std::string_view(tokenStart, 1), tokenLine, tokenColumn);

        case ',':
            return Token(TokenKind::Comma, std::string_view(tokenStart, 1), tokenLine, tokenColumn);

        case '=':
            return Token(TokenKind::Equal, std::string_view(tokenStart, 1), tokenLine, tokenColumn);

        case '"':
            // Quoted string
            while(peekChar() != '"' && !isAtBufferEnd())
            {
                if(peekChar() == '\\')
                {
                    consumeChar(); // consume backslash
                    if(!isAtBufferEnd())
                        consumeChar(); // consume escaped char
                }
                else
                {
                    consumeChar();
                }
            }
            if(peekChar() == '"')
                consumeChar(); // consume closing quote
            return Token(TokenKind::QuotedString,
                         std::string_view(tokenStart, curPtr - tokenStart),
                         tokenLine,
                         tokenColumn);

        case '0':
            // Check for hex literal
            if(peekChar() == 'x' || peekChar() == 'X')
            {
                consumeChar(); // consume 'x'
                while(isHexDigit(peekChar()))
                {
                    consumeChar();
                }
                return Token(TokenKind::HexLiteral,
                             std::string_view(tokenStart, curPtr - tokenStart),
                             tokenLine,
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
            while(isDigit(peekChar()))
            {
                consumeChar();
            }
            // Check for decimal point (float)
            if(peekChar() == '.')
            {
                consumeChar(); // consume '.'
                while(isDigit(peekChar()))
                {
                    consumeChar();
                }
                return Token(TokenKind::FloatLiteral,
                             std::string_view(tokenStart, curPtr - tokenStart),
                             tokenLine,
                             tokenColumn);
            }
            return Token(TokenKind::IntegerLiteral,
                         std::string_view(tokenStart, curPtr - tokenStart),
                         tokenLine,
                         tokenColumn);

        case '-':
            // Could be negative number
            if(isDigit(peekChar()))
            {
                while(isDigit(peekChar()))
                {
                    consumeChar();
                }
                // Check for decimal point (float)
                if(peekChar() == '.')
                {
                    consumeChar(); // consume '.'
                    while(isDigit(peekChar()))
                    {
                        consumeChar();
                    }
                    return Token(TokenKind::FloatLiteral,
                                 std::string_view(tokenStart, curPtr - tokenStart),
                                 tokenLine,
                                 tokenColumn);
                }
                return Token(TokenKind::IntegerLiteral,
                             std::string_view(tokenStart, curPtr - tokenStart),
                             tokenLine,
                             tokenColumn);
            }
            // Otherwise treat as part of identifier
            while(isIdentifierContinue(peekChar()))
            {
                consumeChar();
            }
            {
                std::string_view text(tokenStart, curPtr - tokenStart);
                return Token(TokenKind::Identifier, text, tokenLine, tokenColumn);
            }

        case '/':
            // Check for C-style comment: //
            if(peekChar() == '/')
            {
                // This is a comment, skip until end of line
                consumeChar(); // consume second '/'
                while(!isAtBufferEnd() && peekChar() != '\n' && peekChar() != '\r')
                {
                    consumeChar();
                }
                // Recursively call lexToken to get the next real token
                return lexToken();
            }
            // Not a comment, just a forward slash (unknown token)
            return Token(
                TokenKind::Unknown, std::string_view(tokenStart, 1), tokenLine, tokenColumn);

        default:
            if(isIdentifierStart(c))
            {
                // Identifier or keyword
                while(isIdentifierContinue(peekChar()))
                {
                    consumeChar();
                }

                std::string_view text(tokenStart, curPtr - tokenStart);
                return Token(TokenKind::Identifier, text, tokenLine, tokenColumn);
            }

            // Unknown character
            return Token(
                TokenKind::Unknown, std::string_view(tokenStart, 1), tokenLine, tokenColumn);
        }
    }

    void IRLexer::skipWhitespace()
    {
        while(!isAtBufferEnd())
        {
            char c = peekChar();
            if(c == ' ' || c == '\t')
            {
                consumeChar();
            }
            // Handle C-style comments: //
            else if(c == '/' && peekAheadChar() == '/')
            {
                // Skip until end of line
                consumeChar(); // consume first '/'
                consumeChar(); // consume second '/'
                while(!isAtBufferEnd() && peekChar() != '\n' && peekChar() != '\r')
                {
                    consumeChar();
                }
                // Don't consume the newline - let it be processed normally
            }
            else
            {
                break;
            }
        }
    }

    void IRLexer::skipWhitespaceAndNewlines()
    {
        while(!isAtBufferEnd())
        {
            char c = peekChar();
            if(isWhitespace(c) || c == '\n' || c == '\r')
            {
                if(c == '\n')
                {
                    currentLine++;
                    currentColumn = 0;
                }
                consumeChar();
            }
            else
            {
                break;
            }
        }
    }

    bool IRLexer::isWhitespace(char c)
    {
        return c == ' ' || c == '\t';
    }

    bool IRLexer::isIdentifierStart(char c)
    {
        return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
    }

    bool IRLexer::isIdentifierContinue(char c)
    {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.';
    }

    bool IRLexer::isDigit(char c)
    {
        return std::isdigit(static_cast<unsigned char>(c));
    }

    bool IRLexer::isHexDigit(char c)
    {
        return std::isxdigit(static_cast<unsigned char>(c));
    }

    char IRLexer::peekChar() const
    {
        if(isAtBufferEnd())
            return '\0';
        return *curPtr;
    }

    char IRLexer::peekAheadChar(size_t offset) const
    {
        const char* lookAhead = curPtr + offset;
        if(lookAhead >= bufferEnd)
            return '\0';
        return *lookAhead;
    }

    char IRLexer::consumeChar()
    {
        if(isAtBufferEnd())
            return '\0';
        char c = *curPtr++;
        currentColumn++;
        return c;
    }

    bool IRLexer::isAtBufferEnd() const
    {
        return curPtr >= bufferEnd;
    }

} // namespace stinkytofu

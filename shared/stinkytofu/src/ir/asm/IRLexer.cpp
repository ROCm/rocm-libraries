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

        case ',':
            return Token(TokenKind::Comma, std::string_view(tokenStart, 1), tokenLine, tokenColumn);

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
                return Token(getIdentifierKind(text), text, tokenLine, tokenColumn);
            }

        default:
            if(isIdentifierStart(c))
            {
                // Identifier or keyword
                while(isIdentifierContinue(peekChar()))
                {
                    consumeChar();
                }

                std::string_view text(tokenStart, curPtr - tokenStart);

                // Special case: Check for "Invalid Register" (two-word literal with single space)
                if(text == "Invalid" && peekChar() == ' ')
                {
                    // Look ahead to see if next word is "Register" with exactly one space
                    const char* tempPtr = curPtr + 1; // Skip the single space

                    // Check if there's exactly one space followed by "Register"
                    if(tempPtr < bufferEnd && *tempPtr != ' ' && bufferEnd - tempPtr >= 8
                       && std::strncmp(tempPtr, "Register", 8) == 0
                       && (tempPtr + 8 >= bufferEnd || !isIdentifierContinue(*(tempPtr + 8))))
                    {
                        // Consume " Register" (one space + "Register")
                        consumeChar(); // consume the single space
                        for(int i = 0; i < 8; i++)
                        {
                            consumeChar();
                        }

                        text = std::string_view(tokenStart, curPtr - tokenStart);
                        return Token(TokenKind::Identifier, text, tokenLine, tokenColumn);
                    }
                }

                // Check if this is a register type followed by '['
                // If so, consume the entire register pattern including brackets
                TokenKind kind = getIdentifierKind(text);
                if(kind == TokenKind::Identifier && peekChar() == '[')
                {
                    // Check if this looks like a register prefix
                    if(text == "v" || text == "s" || text == "acc" || text == "SCC"
                       || text == "BARRIER" || text == "DS_WRITE")
                    {
                        // Consume the bracket and contents
                        consumeChar(); // '['
                        while(!isAtBufferEnd() && peekChar() != ']' && peekChar() != '\n')
                        {
                            consumeChar();
                        }
                        if(peekChar() == ']')
                        {
                            consumeChar(); // ']'
                        }

                        // Re-evaluate the complete text
                        text = std::string_view(tokenStart, curPtr - tokenStart);
                        kind = getIdentifierKind(text);
                    }
                }

                return Token(kind, text, tokenLine, tokenColumn);
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

    TokenKind IRLexer::getIdentifierKind(std::string_view text)
    {
        // Check for special keywords
        if(text == "Dest")
            return TokenKind::Dest;
        if(text == "Src")
            return TokenKind::Src;
        if(text == "issueCycles")
            return TokenKind::IssueCycles;
        if(text == "latencyCycles")
            return TokenKind::LatencyCycles;

        // Check for register patterns (now with brackets included)
        if(text.size() >= 3) // Minimum: "v[0]"
        {
            // Single letter register types
            if(text[0] == 'v' && text[1] == '[')
                return TokenKind::VReg;
            if(text[0] == 's' && text[1] == '[')
                return TokenKind::SReg;

            // Multi-character register types
            if(text.size() >= 5) // Minimum: "acc[0]"
            {
                if(text.substr(0, 3) == "acc" && text[3] == '[')
                    return TokenKind::AccReg;
                if(text.substr(0, 3) == "SCC" && text[3] == '[')
                    return TokenKind::SccReg;
            }

            if(text.size() >= 9) // Minimum: "BARRIER[0]"
            {
                if(text.substr(0, 7) == "BARRIER" && text[7] == '[')
                    return TokenKind::BarrierReg;
            }

            if(text.size() >= 10) // Minimum: "DS_WRITE[0]"
            {
                if(text.substr(0, 8) == "DS_WRITE" && text[8] == '[')
                    return TokenKind::DSWriteReg;
            }
        }

        return TokenKind::Identifier;
    }

} // namespace stinkytofu

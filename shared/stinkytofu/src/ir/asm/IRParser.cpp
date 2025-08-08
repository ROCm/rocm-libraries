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

#include "ir/asm/IRParser.hpp"

#include "IRLexer.hpp"

#include <algorithm>
#include <iostream>
#include <optional>

using namespace stinkytofu;

namespace
{
    //----------------------------------------------------------------------
    // IRParser declaration
    //----------------------------------------------------------------------

    /// Parser for the StinkyTofu IR text format.
    /// Constructs ParsedInstruction objects from the token stream.
    class IRParser
    {
    private:
        IRLexer&                lexer;
        std::vector<Diagnostic> diagnostics;
        bool                    hadError;

    public:
        explicit IRParser(IRLexer& lex);

        /// Parse the entire IR file and return a list of parsed instructions.
        std::vector<std::unique_ptr<ParsedInstruction>> parse();

        /// Check if any errors were encountered during parsing.
        bool hasErrors() const
        {
            return hadError;
        }

        /// Get all diagnostics (errors and warnings).
        const std::vector<Diagnostic>& getDiagnostics() const
        {
            return diagnostics;
        }

        /// Print all diagnostics to stderr.
        void printDiagnostics() const;

    private:
        //------------------------------------------------------------------
        // Parsing methods
        //------------------------------------------------------------------

        /// Parse a single instruction block.
        std::unique_ptr<ParsedInstruction> parseInstruction();

        /// Parse the instruction opcode line and return it as a string.
        std::optional<std::string> parseOpcode();

        /// Parse the "Dest:" line with destination registers.
        bool parseDestLine(ParsedInstruction& inst);

        /// Parse the "Src :" line with source registers.
        bool parseSrcLine(ParsedInstruction& inst);

        /// Parse the "issueCycles:" line.
        bool parseIssueCycles(ParsedInstruction& inst);

        /// Parse the "latencyCycles:" line.
        bool parseLatencyCycles(ParsedInstruction& inst);

        /// Parse a register operand (e.g., v[10:11], s[56], acc[0:3]).
        std::optional<StinkyRegister> parseRegister();

        /// Parse a register with bracket notation like v[10:11].
        std::optional<StinkyRegister> parseRegisterWithBrackets(const std::string& regType);

        /// Parse a literal value (integer, hex, or string).
        std::optional<StinkyRegister> parseLiteral();

        //------------------------------------------------------------------
        // Utility methods
        //------------------------------------------------------------------

        /// Get the current token without consuming it.
        const Token& peek() const
        {
            return lexer.peek();
        }

        /// Consume and return the current token.
        const Token& consume()
        {
            return lexer.consume();
        }

        /// Check if the current token matches the expected kind.
        bool check(TokenKind kind) const
        {
            return peek().kind == kind;
        }

        /// Consume a token if it matches the expected kind.
        bool match(TokenKind kind)
        {
            if(check(kind))
            {
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

    IRParser::IRParser(IRLexer& lex)
        : lexer(lex)
        , hadError(false)
    {
    }

    std::vector<std::unique_ptr<ParsedInstruction>> IRParser::parse()
    {
        std::vector<std::unique_ptr<ParsedInstruction>> instructions;

        // Skip any leading newlines
        skipNewlines();

        while(!lexer.isAtEnd() && peek().kind != TokenKind::Eof)
        {
            auto inst = parseInstruction();
            if(inst)
            {
                instructions.push_back(std::move(inst));
            }
            else
            {
                // Error occurred, try to recover by skipping to next instruction
                // Skip until we find a newline followed by an identifier
                while(!lexer.isAtEnd() && peek().kind != TokenKind::Eof)
                {
                    if(peek().kind == TokenKind::Newline)
                    {
                        consume();
                        skipNewlines();
                        if(peek().kind == TokenKind::Identifier)
                        {
                            break; // Found potential next instruction
                        }
                    }
                    else
                    {
                        consume();
                    }
                }
            }

            skipNewlines();
        }

        return instructions;
    }

    void IRParser::printDiagnostics() const
    {
        for(const auto& diag : diagnostics)
        {
            std::cerr << diag.format() << "\n";
        }
    }

    std::unique_ptr<ParsedInstruction> IRParser::parseInstruction()
    {
        // Parse the opcode (instruction name)
        auto opcodeOpt = parseOpcode();
        if(!opcodeOpt)
        {
            return nullptr;
        }

        auto inst = std::make_unique<ParsedInstruction>(std::move(*opcodeOpt));

        // Expect a newline after the opcode
        if(!expect(TokenKind::Newline, "Expected newline after instruction opcode"))
        {
            return nullptr;
        }

        skipNewlines();

        // Parse the instruction fields
        // Expected format:
        //   Dest: <registers>
        //   Src : <registers>
        //   issueCycles: <number>
        //   latencyCycles: <number>

        bool parsedDest          = false;
        bool parsedSrc           = false;
        bool parsedIssueCycles   = false;
        bool parsedLatencyCycles = false;

        while(!lexer.isAtEnd() && peek().kind != TokenKind::Eof)
        {
            // Check if we've reached the next instruction
            if(peek().kind == TokenKind::Identifier && peek().column <= 2)
            {
                // This looks like a new instruction (no leading spaces)
                break;
            }

            if(peek().kind == TokenKind::Dest)
            {
                if(!parseDestLine(*inst))
                {
                    return nullptr;
                }
                parsedDest = true;
            }
            else if(peek().kind == TokenKind::Src)
            {
                if(!parseSrcLine(*inst))
                {
                    return nullptr;
                }
                parsedSrc = true;
            }
            else if(peek().kind == TokenKind::IssueCycles)
            {
                if(!parseIssueCycles(*inst))
                {
                    return nullptr;
                }
                parsedIssueCycles = true;
            }
            else if(peek().kind == TokenKind::LatencyCycles)
            {
                if(!parseLatencyCycles(*inst))
                {
                    return nullptr;
                }
                parsedLatencyCycles = true;
            }
            else if(peek().kind == TokenKind::Newline)
            {
                consume();
            }
            else
            {
                // Unknown field, skip this line
                emitWarning("Unknown instruction field, skipping", peek().line, peek().column);
                while(!lexer.isAtEnd() && peek().kind != TokenKind::Newline
                      && peek().kind != TokenKind::Eof)
                {
                    consume();
                }
                if(peek().kind == TokenKind::Newline)
                {
                    consume();
                }
            }
        }

        // Validate that we parsed the required fields
        if(!parsedIssueCycles)
        {
            emitError("Instruction missing issueCycles field", inst->destRegs.empty() ? 0 : 1, 0);
        }
        if(!parsedLatencyCycles)
        {
            emitError("Instruction missing latencyCycles field", inst->destRegs.empty() ? 0 : 1, 0);
        }

        return inst;
    }

    std::optional<std::string> IRParser::parseOpcode()
    {
        if(peek().kind != TokenKind::Identifier)
        {
            emitError("Expected instruction opcode");
            return std::nullopt;
        }

        const Token& tok       = consume();
        std::string  opcodeStr = std::string(tok.text);

        // Simply return the opcode string without validation
        return opcodeStr;
    }

    bool IRParser::parseDestLine(ParsedInstruction& inst)
    {
        // Consume "Dest"
        consume();

        // Expect ":"
        if(!expect(TokenKind::Colon, "Expected ':' after 'Dest'"))
        {
            return false;
        }

        // Parse destination registers (can be multiple, possibly on multiple lines)
        while(!lexer.isAtEnd() && peek().kind != TokenKind::Eof)
        {
            // Check if next token is a field keyword (new field starting)
            if(peek().kind == TokenKind::Src || peek().kind == TokenKind::IssueCycles
               || peek().kind == TokenKind::LatencyCycles)
            {
                break;
            }

            // Skip newlines but check if we should continue
            if(peek().kind == TokenKind::Newline)
            {
                consume();
                // Check if the next non-newline token starts a new field or instruction
                // If column is <= 2, it's likely a new instruction
                if(!lexer.isAtEnd() && peek().kind != TokenKind::Newline)
                {
                    if(peek().kind == TokenKind::Src || peek().kind == TokenKind::IssueCycles
                       || peek().kind == TokenKind::LatencyCycles || peek().kind == TokenKind::Dest
                       || (peek().kind == TokenKind::Identifier && peek().column <= 2))
                    {
                        break;
                    }
                }
                continue;
            }

            auto regOpt = parseRegister();
            if(regOpt)
            {
                inst.destRegs.push_back(*regOpt);
            }
            else
            {
                // Could be end of dest line, check next token
                break;
            }
        }

        return true;
    }

    bool IRParser::parseSrcLine(ParsedInstruction& inst)
    {
        // Consume "Src"
        consume();

        // Expect ":"
        if(!expect(TokenKind::Colon, "Expected ':' after 'Src'"))
        {
            return false;
        }

        // Parse source registers/literals (can be multiple, possibly on multiple lines)
        while(!lexer.isAtEnd() && peek().kind != TokenKind::Eof)
        {
            // Check if this looks like the next field
            if(peek().kind == TokenKind::IssueCycles || peek().kind == TokenKind::LatencyCycles
               || peek().kind == TokenKind::Dest)
            {
                break;
            }

            // Skip newlines but check if we should continue
            if(peek().kind == TokenKind::Newline)
            {
                consume();
                // Check if the next non-newline token starts a new field or instruction
                if(!lexer.isAtEnd() && peek().kind != TokenKind::Newline)
                {
                    if(peek().kind == TokenKind::IssueCycles
                       || peek().kind == TokenKind::LatencyCycles || peek().kind == TokenKind::Dest
                       || (peek().kind == TokenKind::Identifier && peek().column <= 2))
                    {
                        break;
                    }
                }
                continue;
            }

            auto regOpt = parseRegister();
            if(regOpt)
            {
                inst.srcRegs.push_back(*regOpt);
            }
            else
            {
                // This could be a literal or identifier, try to parse as such
                auto litOpt = parseLiteral();
                if(litOpt)
                {
                    inst.srcRegs.push_back(*litOpt);
                }
                else
                {
                    // Unknown token, skip it
                    consume();
                }
            }
        }

        return true;
    }

    bool IRParser::parseIssueCycles(ParsedInstruction& inst)
    {
        // Consume "issueCycles"
        consume();

        // Expect ":"
        if(!expect(TokenKind::Colon, "Expected ':' after 'issueCycles'"))
        {
            return false;
        }

        // Expect an integer
        if(peek().kind != TokenKind::IntegerLiteral)
        {
            emitError("Expected integer value for issueCycles");
            return false;
        }

        const Token& tok = consume();
        inst.issueCycles = std::stoi(std::string(tok.text));

        // Expect newline
        if(peek().kind == TokenKind::Newline)
        {
            consume();
        }

        return true;
    }

    bool IRParser::parseLatencyCycles(ParsedInstruction& inst)
    {
        // Consume "latencyCycles"
        consume();

        // Expect ":"
        if(!expect(TokenKind::Colon, "Expected ':' after 'latencyCycles'"))
        {
            return false;
        }

        // Expect an integer
        if(peek().kind != TokenKind::IntegerLiteral)
        {
            emitError("Expected integer value for latencyCycles");
            return false;
        }

        const Token& tok   = consume();
        inst.latencyCycles = std::stoi(std::string(tok.text));

        // Expect newline
        if(peek().kind == TokenKind::Newline)
        {
            consume();
        }

        return true;
    }

    std::optional<StinkyRegister> IRParser::parseRegister()
    {
        TokenKind kind = peek().kind;

        switch(kind)
        {
        case TokenKind::VReg:
        case TokenKind::SReg:
        case TokenKind::AccReg:
        case TokenKind::SccReg:
        case TokenKind::BarrierReg:
        case TokenKind::DSWriteReg:
        {
            const Token& tok = consume();
            std::string  text(tok.text);

            // Extract register type prefix and bracket contents
            // Format: regType[idx] or regType[start:end]
            size_t bracketPos = text.find('[');
            if(bracketPos == std::string::npos)
            {
                emitError("Invalid register format", tok.line, tok.column);
                return std::nullopt;
            }

            std::string regType = text.substr(0, bracketPos);

            // Extract content between brackets
            size_t closeBracket = text.find(']', bracketPos);
            if(closeBracket == std::string::npos)
            {
                emitError("Missing closing bracket in register", tok.line, tok.column);
                return std::nullopt;
            }

            std::string bracketContent = text.substr(bracketPos + 1, closeBracket - bracketPos - 1);

            // Parse the bracket content (either "idx" or "start:end")
            size_t colonPos = bracketContent.find(':');
            int    startIdx, endIdx;

            if(colonPos != std::string::npos)
            {
                // Range format: start:end
                std::string startStr = bracketContent.substr(0, colonPos);
                std::string endStr   = bracketContent.substr(colonPos + 1);

                // Validate and parse start index
                if(startStr.empty() || !std::all_of(startStr.begin(), startStr.end(), ::isdigit))
                {
                    emitError("Invalid register start index", tok.line, tok.column);
                    return std::nullopt;
                }
                startIdx = std::stoi(startStr);

                // Validate and parse end index
                if(endStr.empty() || !std::all_of(endStr.begin(), endStr.end(), ::isdigit))
                {
                    emitError("Invalid register end index", tok.line, tok.column);
                    return std::nullopt;
                }
                endIdx = std::stoi(endStr);
            }
            else
            {
                // Single register
                if(bracketContent.empty()
                   || !std::all_of(bracketContent.begin(), bracketContent.end(), ::isdigit))
                {
                    emitError("Invalid register index", tok.line, tok.column);
                    return std::nullopt;
                }
                startIdx = std::stoi(bracketContent);
                endIdx   = startIdx;
            }

            int regNum = endIdx - startIdx + 1;
            if(regNum <= 0)
            {
                emitError("Invalid register range", tok.line, tok.column);
                return std::nullopt;
            }

            return StinkyRegister(regType, startIdx, regNum);
        }

        case TokenKind::Identifier:
        {
            // Not a register
            return std::nullopt;
        }

        default:
            return std::nullopt;
        }
    }

    std::optional<StinkyRegister> IRParser::parseRegisterWithBrackets(const std::string& regType)
    {
        // Expect '['
        if(!expect(TokenKind::LeftBracket, "Expected '[' after register type"))
        {
            return std::nullopt;
        }

        // Parse start index
        if(peek().kind != TokenKind::IntegerLiteral)
        {
            emitError("Expected integer register index");
            return std::nullopt;
        }

        const Token& startTok = consume();
        int          startIdx = std::stoi(std::string(startTok.text));

        int endIdx = startIdx;

        // Check for range notation [start:end]
        if(peek().kind == TokenKind::Colon)
        {
            consume();

            if(peek().kind != TokenKind::IntegerLiteral)
            {
                emitError("Expected integer register index after ':'");
                return std::nullopt;
            }

            const Token& endTok = consume();
            endIdx              = std::stoi(std::string(endTok.text));
        }

        // Expect ']'
        if(!expect(TokenKind::RightBracket, "Expected ']' after register index"))
        {
            return std::nullopt;
        }

        // Calculate register count
        int regNum = endIdx - startIdx + 1;
        if(regNum <= 0)
        {
            emitError("Invalid register range");
            return std::nullopt;
        }

        return StinkyRegister(regType, startIdx, regNum);
    }

    std::optional<StinkyRegister> IRParser::parseLiteral()
    {
        if(peek().kind == TokenKind::IntegerLiteral)
        {
            const Token& tok = consume();
            int          val = std::stoi(std::string(tok.text));
            return StinkyRegister(val);
        }
        else if(peek().kind == TokenKind::HexLiteral)
        {
            const Token& tok = consume();
            std::string  text(tok.text);
            // Parse hex literal (starts with 0x)
            int val = std::stoi(text, nullptr, 16);
            return StinkyRegister(val);
        }
        else if(peek().kind == TokenKind::Identifier)
        {
            const Token& tok = consume();
            std::string  text(tok.text);
            return StinkyRegister(text);
        }

        return std::nullopt;
    }

    bool IRParser::expect(TokenKind kind, const std::string& message)
    {
        if(check(kind))
        {
            consume();
            return true;
        }

        emitError(message);
        return false;
    }

    void IRParser::skipNewlines()
    {
        while(peek().kind == TokenKind::Newline)
        {
            consume();
        }
    }

    void IRParser::emitError(const std::string& message, unsigned line, unsigned column)
    {
        diagnostics.emplace_back(Diagnostic::Level::Error, message, line, column);
        hadError = true;
    }

    void IRParser::emitError(const std::string& message)
    {
        const Token& tok = peek();
        emitError(message, tok.line, tok.column);
    }

    void IRParser::emitWarning(const std::string& message, unsigned line, unsigned column)
    {
        diagnostics.emplace_back(Diagnostic::Level::Warning, message, line, column);
    }

} // namespace

namespace stinkytofu
{
    std::vector<std::unique_ptr<ParsedInstruction>> parseSourceString(const std::string& sourceStr)
    {
        // Create lexer and tokenize
        IRLexer lexer(sourceStr);
        lexer.lex();

        // Create parser and parse
        return IRParser(lexer).parse();
    }
}
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

#include <memory>
#include <string>
#include <vector>

namespace stinkytofu
{

    // Forward declaration
    class IRLexer;

    //===----------------------------------------------------------------------===//
    // Pattern AST Nodes
    //===----------------------------------------------------------------------===//

    /// Represents a single match statement in a pattern.
    /// Example: $fma_inst = v_fma_f32 $dst, $a, $b, $c
    struct MatchStmt
    {
        std::string              instVar; // Instruction variable (e.g., "fma_inst")
        std::string              opcode; // Opcode (e.g., "v_fma_f32")
        std::vector<std::string> operands; // Operand variables (e.g., ["dst", "a", "b", "c"])
    };

    /// Represents a constraint in a pattern.
    /// Example: HasOneUse($fma_result)
    struct Constraint
    {
        std::string              function; // Constraint function (e.g., "HasOneUse")
        std::vector<std::string> args; // Arguments (e.g., ["fma_result"])
    };

    /// Represents a rewrite statement in a pattern.
    struct RewriteStmt
    {
        enum class Kind
        {
            CreateInst, // $var = opcode $dst, $src1, ...
            BuiltinCall, // $var = AddConstants($a, $b)
            Replace, // replace $old with $new
            Remove // remove $inst
        };

        Kind                     kind;
        std::string              lhs; // Left-hand side variable (for CreateInst, BuiltinCall)
        std::string              opcode; // Opcode (for CreateInst)
        std::string              function; // Function name (for BuiltinCall)
        std::vector<std::string> operands; // Operands/arguments
        std::string              oldVar; // Old variable (for Replace)
        std::string              newVar; // New variable (for Replace)
        std::string              instVar; // Instruction variable (for Remove)
    };

    /// Pattern type enum - specifies what kind of pattern this is
    enum class PatternType
    {
        Peephole // Peephole optimization pattern
    };

    /// Represents a complete pattern definition.
    struct Pattern
    {
        std::string              name;
        PatternType              type; // Type of pattern (must be explicitly set)
        std::vector<MatchStmt>   match;
        std::vector<Constraint>  constraints;
        std::vector<RewriteStmt> rewrite;
    };

    //===----------------------------------------------------------------------===//
    // Pattern Parser
    //===----------------------------------------------------------------------===//

    /// Parser for StinkyTofu pattern definition files.
    /// Reuses the existing IRLexer for tokenization.
    class PatternParser
    {
    public:
        /// Construct a parser from an existing lexer.
        /// The lexer must have already been initialized and lex() called.
        explicit PatternParser(IRLexer& lexer);

        /// Parse all patterns from the lexer's token stream.
        /// Returns a vector of Pattern objects.
        std::vector<Pattern> parsePatterns();

        /// Check if any errors occurred during parsing.
        bool hasErrors() const
        {
            return hadError;
        }

        /// Get all error messages.
        const std::vector<std::string>& getErrors() const
        {
            return errors;
        }

        /// Print all errors to stderr.
        void printErrors() const;

    private:
        IRLexer&                 lexer;
        std::vector<std::string> errors;
        bool                     hadError;

        // Parsing methods
        Pattern                  parsePattern();
        std::vector<MatchStmt>   parseMatchBlock();
        MatchStmt                parseMatchStmt();
        std::vector<Constraint>  parseConstraintsBlock();
        Constraint               parseConstraint();
        std::vector<RewriteStmt> parseRewriteBlock();
        RewriteStmt              parseRewriteStmt();

        // Utility methods
        void        skipNewlines();
        std::string parseVariable(); // Parses a variable (identifier without $)
        void        error(const std::string& msg);
    };

    /// Parse a pattern file and return all patterns.
    /// This is a convenience function that creates a lexer and parser.
    /// @param filename Path to the pattern file
    /// @return Vector of parsed patterns (empty if parsing failed)
    std::vector<Pattern> parsePatternFile(const std::string& filename);

} // namespace stinkytofu

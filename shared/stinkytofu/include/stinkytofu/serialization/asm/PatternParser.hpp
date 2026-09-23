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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/support/Diagnostic.hpp"

namespace stinkytofu {

// Forward declaration
class IRLexer;

//===----------------------------------------------------------------------===//
// Pattern AST Nodes
//===----------------------------------------------------------------------===//

/// Represents a single match statement in a pattern.
/// Example: $fma_inst = v_fma_f32 $dst, $a, $b, $c
struct MatchStmt {
    std::string instVar;                // Instruction variable (e.g., "fma_inst")
    std::string opcode;                 // Opcode (e.g., "v_fma_f32")
    std::vector<std::string> operands;  // Operand variables (e.g., ["dst", "a", "b", "c"])
};

/// Represents a constraint in a pattern.
/// Example: HasOneUse($fma_result)
struct Constraint {
    std::string function;           // Constraint function (e.g., "HasOneUse")
    std::vector<std::string> args;  // Arguments (e.g., ["fma_result"])
};

/// Represents a rewrite statement in a pattern.
struct RewriteStmt {
    enum class Kind {
        CreateInst,   // $var = opcode $dst, $src1, ...
        BuiltinCall,  // $var = AddConstants($a, $b)
        Replace,      // replace $old with $new
        Remove        // remove $inst
    };

    Kind kind{Kind::CreateInst};
    std::string lhs;                    // Left-hand side variable (for CreateInst, BuiltinCall)
    std::string opcode;                 // Opcode (for CreateInst)
    std::string function;               // Function name (for BuiltinCall)
    std::vector<std::string> operands;  // Operands/arguments
    std::string oldVar;                 // Old variable (for Replace)
    std::string newVar;                 // New variable (for Replace)
    std::string instVar;                // Instruction variable (for Remove)
};

/// Pattern type enum - specifies what kind of pattern this is
enum class PatternType {
    Peephole,   // Peephole optimization pattern (assembly IR)
    LogicalIR,  // High-level IR optimization pattern
    Intrinsic   // High-level IR intrinsic definition
};

//===----------------------------------------------------------------------===//
// Intrinsic-specific structures
//===----------------------------------------------------------------------===//

/// Intrinsic argument
struct IntrinsicArgument {
    std::string name;     // Argument name (e.g., "dest", "src")
    std::string regType;  // Register type (e.g., "vgpr", "sgpr")
};

/// Typed operand for intrinsic instructions
struct IntrinsicOperand {
    enum Type {
        Register,      // Register reference (e.g., "src", "dest")
        IntLiteral,    // Integer literal (e.g., 123, -456)
        FloatLiteral,  // Float literal (e.g., 3.14, -2.5)
        HexLiteral     // Hex literal treated as float bits (e.g., 0x40ec7326)
    };

    Type type;
    std::string registerName;  // For Register type
    int64_t intValue;          // For IntLiteral type
    double floatValue;         // For FloatLiteral and HexLiteral types

    // Constructor for register
    IntrinsicOperand(const std::string& name)
        : type(Register), registerName(name), intValue(0), floatValue(0.0) {}

    // Constructor for int literal
    IntrinsicOperand(int64_t val) : type(IntLiteral), intValue(val), floatValue(0.0) {}

    // Constructor for float literal
    IntrinsicOperand(double val) : type(FloatLiteral), intValue(0), floatValue(val) {}

    // Constructor for hex literal
    static IntrinsicOperand hexLiteral(double val) {
        IntrinsicOperand op(0.0);
        op.type = HexLiteral;
        op.floatValue = val;
        return op;
    }

    // Default constructor
    IntrinsicOperand() : type(Register), intValue(0), floatValue(0.0) {}
};

/// High-level IR instruction in intrinsic body
struct IntrinsicInstruction {
    std::string destReg;                      // Destination register
    std::string operation;                    // Operation name (e.g., "v_add_f32" or function name)
    std::vector<IntrinsicOperand> operands;   // Typed operands
    std::vector<std::string> operandStrings;  // Legacy: kept for function call args

    // Function call support
    bool isFunctionCall = false;
    std::vector<std::pair<std::string, std::string>> funcCallArgs;  // (argName, argValue) pairs
};

/// Represents a complete pattern definition.
struct Pattern {
    std::string name;
    PatternType type;  // Type of pattern (must be explicitly set)

    // Peephole pattern fields
    std::vector<MatchStmt> match;
    std::vector<Constraint> constraints;
    std::vector<RewriteStmt> rewrite;

    // Intrinsic pattern fields
    std::vector<IntrinsicArgument> arguments;
    std::vector<IntrinsicInstruction> body;
    std::string comment;
    bool pythonBinding = false;
};

//===----------------------------------------------------------------------===//
// Pattern Parser
//===----------------------------------------------------------------------===//

/// Parser for StinkyTofu pattern definition files.
/// Reuses the existing IRLexer for tokenization.
class STINKYTOFU_EXPORT PatternParser {
   public:
    /// Construct a parser from an existing lexer.
    /// The lexer must have already been initialized and lex() called.
    explicit PatternParser(IRLexer& lexer);

    /// Parse all patterns from the lexer's token stream.
    /// Returns a vector of Pattern objects.
    std::vector<Pattern> parsePatterns();

    /// Check if any errors occurred during parsing.
    bool hasErrors() const {
        return hadError;
    }

    /// Get all diagnostics (structured error/warning/note messages).
    const std::vector<Diagnostic>& getDiagnostics() const {
        return diagnostics;
    }

    /// Get all error messages (legacy, for backward compatibility).
    /// Returns formatted strings from diagnostics.
    std::vector<std::string> getErrors() const;

    /// Print all errors to stderr.
    void printErrors() const;

   private:
    IRLexer& lexer;
    std::vector<Diagnostic> diagnostics;
    bool hadError;

    // Parsing methods
    Pattern parsePattern();
    Pattern parseIntrinsic();
    std::vector<MatchStmt> parseMatchBlock();
    MatchStmt parseMatchStmt();
    std::vector<Constraint> parseConstraintsBlock();
    Constraint parseConstraint();
    std::vector<RewriteStmt> parseRewriteBlock();
    RewriteStmt parseRewriteStmt();
    std::vector<IntrinsicArgument> parseArgumentsBlock();
    std::vector<IntrinsicInstruction> parseIntrinsicBody();
    IntrinsicInstruction parseIntrinsicInstruction();

    // Utility methods
    void skipNewlines();
    std::string parseVariable();  // Parses a variable (identifier without $)
    std::string stripQuotes(
        const std::string& str);  // Strips surrounding quotes from string literals
    void error(const std::string& msg);
};

/// Result of parsing a pattern file, containing both patterns and diagnostics.
struct PatternParseResult {
    std::vector<Pattern> patterns;
    std::vector<Diagnostic> diagnostics;

    /// Check if any errors occurred during parsing.
    bool hasErrors() const {
        for (const auto& diag : diagnostics) {
            if (diag.getLevel() == Diagnostic::Level::Error) {
                return true;
            }
        }
        return false;
    }

    /// Get count of errors (excluding warnings).
    size_t errorCount() const {
        size_t count = 0;
        for (const auto& diag : diagnostics) {
            if (diag.getLevel() == Diagnostic::Level::Error) {
                count++;
            }
        }
        return count;
    }
};

/// Parse a pattern file and return all patterns.
/// This is a convenience function that creates a lexer and parser.
/// @param filename Path to the pattern file
/// @return Vector of parsed patterns (empty if parsing failed)
/// @note This function does not expose parse errors. Use parsePatternFileWithDiagnostics() to
/// access diagnostics.
STINKYTOFU_EXPORT std::vector<Pattern> parsePatternFile(const std::string& filename);

/// Parse a pattern file and return patterns with diagnostic information.
/// This is a convenience function that creates a lexer and parser.
/// @param filename Path to the pattern file
/// @return A PatternParseResult containing parsed patterns and any diagnostics (errors/warnings).
STINKYTOFU_EXPORT PatternParseResult parsePatternFileWithDiagnostics(const std::string& filename);

}  // namespace stinkytofu

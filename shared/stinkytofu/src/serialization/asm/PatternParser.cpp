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

#include "stinkytofu/serialization/asm/PatternParser.hpp"

#include <bit>
#include <fstream>
#include <iostream>
#include <sstream>

#include "IRLexer.hpp"

namespace stinkytofu {

//===----------------------------------------------------------------------===//
// PatternParser Implementation
//===----------------------------------------------------------------------===//

PatternParser::PatternParser(IRLexer& lexer) : lexer(lexer), hadError(false) {}

void PatternParser::skipNewlines() {
    while (lexer.peek().kind == TokenKind::Newline) {
        lexer.consume();
    }
}

void PatternParser::error(const std::string& msg) {
    const Token& tok = lexer.peek();
    diagnostics.emplace_back(Diagnostic::Level::Error, msg, tok.line, tok.column);
    hadError = true;
}

std::vector<std::string> PatternParser::getErrors() const {
    // Legacy API: format diagnostics as strings
    std::vector<std::string> result;
    result.reserve(diagnostics.size());
    for (const auto& diag : diagnostics) {
        std::ostringstream oss;
        oss << "Error at line " << diag.getLine() << ", column " << diag.getColumn() << ": "
            << diag.getMessage();
        result.push_back(oss.str());
    }
    return result;
}

void PatternParser::printErrors() const {
    for (const auto& diag : diagnostics) {
        // Format: "Error at line 42, column 15: Expected '{'"
        std::cerr << "Error at line " << diag.getLine() << ", column " << diag.getColumn() << ": "
                  << diag.getMessage() << "\n";
    }
}

std::string PatternParser::stripQuotes(const std::string& str) {
    // Strip leading and trailing quotes from string literals
    if (str.size() >= 2 && str.front() == '"' && str.back() == '"') {
        return str.substr(1, str.size() - 2);
    }
    return str;
}

std::string PatternParser::parseVariable() {
    // Variables in patterns can optionally start with $ for clarity
    // The lexer treats $ as Unknown token, so we check and consume it

    // Check if there's a $ prefix (lexer tokenizes it separately)
    if (lexer.peek().kind == TokenKind::Unknown && lexer.peek().text == "$") {
        lexer.consume();  // consume the $
    }

    // Now expect the actual variable name
    if (lexer.peek().kind != TokenKind::Identifier) {
        error("Expected variable name after '$'");
        return "";
    }
    return std::string(lexer.consume().text);
}

std::vector<Pattern> PatternParser::parsePatterns() {
    std::vector<Pattern> patterns;

    while (!lexer.isAtEnd()) {
        skipNewlines();
        if (lexer.isAtEnd()) break;

        // Check for pattern type keywords (peephole, ir, intrinsic, etc.)
        if (lexer.peek().kind == TokenKind::KW_peephole || lexer.peek().kind == TokenKind::KW_ir) {
            patterns.push_back(parsePattern());
            if (hadError) {
                hadError = false;  // Continue parsing
            }
        } else if (lexer.peek().kind == TokenKind::KW_intrinsic) {
            patterns.push_back(parseIntrinsic());
            if (hadError) {
                hadError = false;  // Continue parsing
            }
        } else {
            // Skip unexpected tokens (like comments)
            lexer.consume();
        }
    }

    return patterns;
}

Pattern PatternParser::parsePattern() {
    Pattern pattern;

    // <type> pattern <name> {
    // Parse required pattern type
    if (lexer.peek().kind == TokenKind::KW_peephole) {
        pattern.type = PatternType::Peephole;
        lexer.consume();  // consume pattern type keyword
        skipNewlines();
    } else if (lexer.peek().kind == TokenKind::KW_ir) {
        pattern.type = PatternType::LogicalIR;
        lexer.consume();  // consume 'ir' keyword
        skipNewlines();
    } else {
        error("Expected pattern type keyword (e.g., 'peephole' or 'ir')");
        return pattern;
    }

    // Now expect 'pattern' keyword
    if (lexer.peek().kind != TokenKind::KW_pattern) {
        error("Expected 'pattern' keyword after pattern type");
        return pattern;
    }
    lexer.consume();  // consume 'pattern'

    skipNewlines();
    if (lexer.peek().kind != TokenKind::Identifier) {
        error("Expected pattern name");
        return pattern;
    }
    pattern.name = std::string(lexer.consume().text);

    skipNewlines();
    if (lexer.peek().kind != TokenKind::LeftBrace) {
        error("Expected '{' after pattern name");
        return pattern;
    }
    lexer.consume();  // consume '{'

    skipNewlines();

    // match { ... }
    if (lexer.peek().kind == TokenKind::KW_match) {
        pattern.match = parseMatchBlock();
        skipNewlines();
    } else {
        error("Expected 'match' block");
        return pattern;
    }

    // constraints { ... } (optional)
    if (lexer.peek().kind == TokenKind::KW_constraints) {
        pattern.constraints = parseConstraintsBlock();
        skipNewlines();
    }

    // rewrite { ... }
    if (lexer.peek().kind == TokenKind::KW_rewrite) {
        pattern.rewrite = parseRewriteBlock();
        skipNewlines();
    } else {
        error("Expected 'rewrite' block");
        return pattern;
    }

    // Closing }
    if (lexer.peek().kind != TokenKind::RightBrace) {
        error("Expected '}' to close pattern");
        return pattern;
    }
    lexer.consume();  // consume '}'

    return pattern;
}

std::vector<MatchStmt> PatternParser::parseMatchBlock() {
    std::vector<MatchStmt> stmts;

    // match {
    if (lexer.peek().kind != TokenKind::KW_match) {
        error("Expected 'match' keyword");
        return stmts;
    }
    lexer.consume();  // consume 'match'

    skipNewlines();
    if (lexer.peek().kind != TokenKind::LeftBrace) {
        error("Expected '{' after 'match'");
        return stmts;
    }
    lexer.consume();  // consume '{'

    skipNewlines();

    // Parse match statements until we see }
    while (lexer.peek().kind != TokenKind::RightBrace && !lexer.isAtEnd()) {
        stmts.push_back(parseMatchStmt());
        skipNewlines();
        if (hadError) break;
    }

    if (lexer.peek().kind != TokenKind::RightBrace) {
        error("Expected '}' to close match block");
        return stmts;
    }
    lexer.consume();  // consume '}'

    return stmts;
}

MatchStmt PatternParser::parseMatchStmt() {
    MatchStmt stmt;

    // $inst_var = opcode $dst, $src1, ...
    // Variables can optionally have $ prefix for clarity

    // Parse instruction variable (handles $ prefix in parseVariable)
    stmt.instVar = parseVariable();
    if (stmt.instVar.empty()) {
        error("Expected instruction variable in match statement");
        return stmt;
    }

    if (lexer.peek().kind != TokenKind::Equal) {
        error("Expected '=' in match statement");
        return stmt;
    }
    lexer.consume();  // consume '='

    if (lexer.peek().kind != TokenKind::Identifier) {
        error("Expected opcode in match statement");
        return stmt;
    }
    stmt.opcode = std::string(lexer.consume().text);

    // Parse operands (comma-separated variables until newline or })
    while (lexer.peek().kind != TokenKind::Newline && lexer.peek().kind != TokenKind::RightBrace &&
           !lexer.isAtEnd()) {
        if (lexer.peek().kind == TokenKind::Comma) {
            lexer.consume();  // skip comma
            continue;
        }

        // Check if this looks like a variable ($ or Identifier)
        if (lexer.peek().kind == TokenKind::Unknown && lexer.peek().text == "$") {
            stmt.operands.push_back(parseVariable());
        } else if (lexer.peek().kind == TokenKind::Identifier) {
            stmt.operands.push_back(parseVariable());
        } else {
            break;
        }
    }

    return stmt;
}

std::vector<Constraint> PatternParser::parseConstraintsBlock() {
    std::vector<Constraint> constraints;

    // constraints {
    if (lexer.peek().kind != TokenKind::KW_constraints) {
        error("Expected 'constraints' keyword");
        return constraints;
    }
    lexer.consume();  // consume 'constraints'

    skipNewlines();
    if (lexer.peek().kind != TokenKind::LeftBrace) {
        error("Expected '{' after 'constraints'");
        return constraints;
    }
    lexer.consume();  // consume '{'

    skipNewlines();

    // Parse constraints until we see }
    while (lexer.peek().kind != TokenKind::RightBrace && !lexer.isAtEnd()) {
        constraints.push_back(parseConstraint());
        skipNewlines();
        if (hadError) break;
    }

    if (lexer.peek().kind != TokenKind::RightBrace) {
        error("Expected '}' to close constraints block");
        return constraints;
    }
    lexer.consume();  // consume '}'

    return constraints;
}

Constraint PatternParser::parseConstraint() {
    Constraint constraint;

    // FunctionName($arg1, $arg2, ...)
    if (lexer.peek().kind != TokenKind::Identifier) {
        error("Expected constraint function name");
        return constraint;
    }
    constraint.function = std::string(lexer.consume().text);

    if (lexer.peek().kind != TokenKind::LeftParen) {
        error("Expected '(' after constraint function name");
        return constraint;
    }
    lexer.consume();  // consume '('

    // Parse arguments (can have $ prefix)
    while (lexer.peek().kind != TokenKind::RightParen && !lexer.isAtEnd()) {
        if (lexer.peek().kind == TokenKind::Comma) {
            lexer.consume();  // skip comma
            continue;
        }

        // Check if this looks like a variable ($ or Identifier)
        if (lexer.peek().kind == TokenKind::Unknown && lexer.peek().text == "$") {
            constraint.args.push_back(parseVariable());
        } else if (lexer.peek().kind == TokenKind::Identifier) {
            constraint.args.push_back(parseVariable());
        } else {
            error("Expected argument in constraint");
            break;
        }
    }

    if (lexer.peek().kind != TokenKind::RightParen) {
        error("Expected ')' to close constraint arguments");
        return constraint;
    }
    lexer.consume();  // consume ')'

    return constraint;
}

std::vector<RewriteStmt> PatternParser::parseRewriteBlock() {
    std::vector<RewriteStmt> stmts;

    // rewrite {
    if (lexer.peek().kind != TokenKind::KW_rewrite) {
        error("Expected 'rewrite' keyword");
        return stmts;
    }
    lexer.consume();  // consume 'rewrite'

    skipNewlines();
    if (lexer.peek().kind != TokenKind::LeftBrace) {
        error("Expected '{' after 'rewrite'");
        return stmts;
    }
    lexer.consume();  // consume '{'

    skipNewlines();

    // Parse rewrite statements until we see }
    while (lexer.peek().kind != TokenKind::RightBrace && !lexer.isAtEnd()) {
        stmts.push_back(parseRewriteStmt());
        skipNewlines();
        if (hadError) break;
    }

    if (lexer.peek().kind != TokenKind::RightBrace) {
        error("Expected '}' to close rewrite block");
        return stmts;
    }
    lexer.consume();  // consume '}'

    return stmts;
}

RewriteStmt PatternParser::parseRewriteStmt() {
    RewriteStmt stmt;

    // Check for keywords first
    if (lexer.peek().kind == TokenKind::KW_replace) {
        // replace $old with $new
        stmt.kind = RewriteStmt::Kind::Replace;
        lexer.consume();  // consume 'replace'

        stmt.oldVar = parseVariable();
        if (stmt.oldVar.empty()) {
            error("Expected variable after 'replace'");
            return stmt;
        }

        if (lexer.peek().kind != TokenKind::KW_with) {
            error("Expected 'with' keyword");
            return stmt;
        }
        lexer.consume();  // consume 'with'

        stmt.newVar = parseVariable();
        if (stmt.newVar.empty()) {
            error("Expected variable after 'with'");
            return stmt;
        }
    } else if (lexer.peek().kind == TokenKind::KW_remove) {
        // remove $inst
        stmt.kind = RewriteStmt::Kind::Remove;
        lexer.consume();  // consume 'remove'

        stmt.instVar = parseVariable();
        if (stmt.instVar.empty()) {
            error("Expected variable after 'remove'");
            return stmt;
        }
    } else if (lexer.peek().kind == TokenKind::Identifier ||
               (lexer.peek().kind == TokenKind::Unknown && lexer.peek().text == "$")) {
        // Assignment: $var = ...
        stmt.lhs = parseVariable();

        if (lexer.peek().kind != TokenKind::Equal) {
            error("Expected '=' in assignment");
            return stmt;
        }
        lexer.consume();  // consume '='

        if (lexer.peek().kind != TokenKind::Identifier) {
            error("Expected identifier after '='");
            return stmt;
        }
        std::string rhs = std::string(lexer.consume().text);

        // Check if it's a function call or instruction creation
        if (lexer.peek().kind == TokenKind::LeftParen) {
            // Builtin function: $var = AddConstants($a, $b)
            stmt.kind = RewriteStmt::Kind::BuiltinCall;
            stmt.function = rhs;
            lexer.consume();  // consume '('

            // Parse arguments (can have $ prefix)
            while (lexer.peek().kind != TokenKind::RightParen && !lexer.isAtEnd()) {
                if (lexer.peek().kind == TokenKind::Comma) {
                    lexer.consume();  // skip comma
                    continue;
                }

                // Check if this looks like a variable ($ or Identifier)
                if (lexer.peek().kind == TokenKind::Unknown && lexer.peek().text == "$") {
                    stmt.operands.push_back(parseVariable());
                } else if (lexer.peek().kind == TokenKind::Identifier) {
                    stmt.operands.push_back(parseVariable());
                } else {
                    error("Expected argument in function call");
                    break;
                }
            }

            if (lexer.peek().kind != TokenKind::RightParen) {
                error("Expected ')' to close function call");
                return stmt;
            }
            lexer.consume();  // consume ')'
        } else {
            // Instruction creation: $var = opcode $dst, $src1, ...
            stmt.kind = RewriteStmt::Kind::CreateInst;
            stmt.opcode = rhs;

            // Parse operands (comma-separated variables until newline or })
            while (lexer.peek().kind != TokenKind::Newline &&
                   lexer.peek().kind != TokenKind::RightBrace && !lexer.isAtEnd()) {
                if (lexer.peek().kind == TokenKind::Comma) {
                    lexer.consume();  // skip comma
                    continue;
                }

                // Check if this looks like a variable ($ or Identifier)
                if (lexer.peek().kind == TokenKind::Unknown && lexer.peek().text == "$") {
                    stmt.operands.push_back(parseVariable());
                } else if (lexer.peek().kind == TokenKind::Identifier) {
                    stmt.operands.push_back(parseVariable());
                } else {
                    break;
                }
            }
        }
    } else {
        error("Expected rewrite statement");
    }

    return stmt;
}

//===----------------------------------------------------------------------===//
// Convenience Function
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Intrinsic Pattern Parsing
//===----------------------------------------------------------------------===//

Pattern PatternParser::parseIntrinsic() {
    Pattern pattern;
    pattern.type = PatternType::Intrinsic;

    // Consume 'intrinsic' keyword
    lexer.consume();
    skipNewlines();

    // Parse intrinsic name
    if (lexer.peek().kind != TokenKind::Identifier) {
        error("Expected intrinsic name");
        return pattern;
    }
    pattern.name = std::string(lexer.consume().text);
    skipNewlines();

    // Expect '{'
    if (lexer.peek().kind != TokenKind::LeftBrace) {
        error("Expected '{' after intrinsic name");
        return pattern;
    }
    lexer.consume();
    skipNewlines();

    // Parse intrinsic body (arguments, body, comment, python_binding)
    while (lexer.peek().kind != TokenKind::RightBrace && !lexer.isAtEnd()) {
        if (lexer.peek().kind == TokenKind::Newline) {
            skipNewlines();
        } else if (lexer.peek().kind == TokenKind::KW_arguments) {
            lexer.consume();
            pattern.arguments = parseArgumentsBlock();
        } else if (lexer.peek().kind == TokenKind::KW_body) {
            lexer.consume();
            pattern.body = parseIntrinsicBody();
        } else if (lexer.peek().kind == TokenKind::KW_comment) {
            lexer.consume();
            skipNewlines();
            if (lexer.peek().kind == TokenKind::QuotedString) {
                pattern.comment = stripQuotes(std::string(lexer.consume().text));
            }
            skipNewlines();
        } else if (lexer.peek().kind == TokenKind::KW_python_binding) {
            lexer.consume();
            skipNewlines();
            if (lexer.peek().kind == TokenKind::KW_true) {
                pattern.pythonBinding = true;
                lexer.consume();
            } else if (lexer.peek().kind == TokenKind::KW_false) {
                pattern.pythonBinding = false;
                lexer.consume();
            }
            skipNewlines();
        } else {
            // Unexpected token, consume it to avoid infinite loop
            error("Unexpected token in intrinsic body: " + std::string(lexer.peek().text));
            lexer.consume();
        }
    }

    // Expect '}'
    if (lexer.peek().kind != TokenKind::RightBrace) {
        error("Expected '}' at end of intrinsic");
        return pattern;
    }
    lexer.consume();
    skipNewlines();

    return pattern;
}

std::vector<IntrinsicArgument> PatternParser::parseArgumentsBlock() {
    std::vector<IntrinsicArgument> args;
    skipNewlines();

    // Expect '{'
    if (lexer.peek().kind != TokenKind::LeftBrace) {
        error("Expected '{' after 'arguments'");
        return args;
    }
    lexer.consume();
    skipNewlines();

    // Parse argument list: name: type
    while (lexer.peek().kind != TokenKind::RightBrace && !lexer.isAtEnd()) {
        if (lexer.peek().kind == TokenKind::Newline) {
            skipNewlines();
        } else if (lexer.peek().kind == TokenKind::Identifier) {
            IntrinsicArgument arg;
            arg.name = std::string(lexer.consume().text);
            skipNewlines();

            // Expect ':'
            if (lexer.peek().kind == TokenKind::Colon) {
                lexer.consume();
                skipNewlines();

                // Parse type
                if (lexer.peek().kind == TokenKind::Identifier) {
                    arg.regType = std::string(lexer.consume().text);
                    args.push_back(arg);
                }
            }
            skipNewlines();
        } else {
            // Unexpected token, consume it to avoid infinite loop
            error("Unexpected token in arguments block: " + std::string(lexer.peek().text));
            lexer.consume();
        }
    }

    // Expect '}'
    if (lexer.peek().kind == TokenKind::RightBrace) {
        lexer.consume();
    }
    skipNewlines();

    return args;
}

std::vector<IntrinsicInstruction> PatternParser::parseIntrinsicBody() {
    std::vector<IntrinsicInstruction> instructions;
    skipNewlines();

    // Expect '{'
    if (lexer.peek().kind != TokenKind::LeftBrace) {
        error("Expected '{' after 'body'");
        return instructions;
    }
    lexer.consume();
    skipNewlines();

    // Parse instruction list
    while (lexer.peek().kind != TokenKind::RightBrace && !lexer.isAtEnd()) {
        if (lexer.peek().kind == TokenKind::Newline) {
            skipNewlines();
        } else if (lexer.peek().kind == TokenKind::Identifier ||
                   lexer.peek().kind == TokenKind::KW_call) {
            auto inst = parseIntrinsicInstruction();
            if (!inst.operation.empty()) {
                instructions.push_back(inst);
            }
        } else {
            // Unexpected token, consume it to avoid infinite loop
            error("Unexpected token in body block: " + std::string(lexer.peek().text));
            lexer.consume();
        }
    }

    // Expect '}'
    if (lexer.peek().kind == TokenKind::RightBrace) {
        lexer.consume();
    }
    skipNewlines();

    return instructions;
}

IntrinsicInstruction PatternParser::parseIntrinsicInstruction() {
    IntrinsicInstruction inst;

    // Check for function call: call FunctionName(arg1=val1, arg2=val2, ...)
    if (lexer.peek().kind == TokenKind::KW_call) {
        lexer.consume();  // consume 'call'
        skipNewlines();

        inst.isFunctionCall = true;

        // Parse function name
        if (lexer.peek().kind != TokenKind::Identifier) {
            error("Expected function name after 'call'");
            return inst;
        }
        inst.operation = std::string(lexer.consume().text);
        skipNewlines();

        // Expect '('
        if (lexer.peek().kind != TokenKind::LeftParen) {
            error("Expected '(' after function name");
            return inst;
        }
        lexer.consume();
        skipNewlines();

        // Parse named arguments: arg=value, arg=value, ...
        while (lexer.peek().kind != TokenKind::RightParen && !lexer.isAtEnd()) {
            if (lexer.peek().kind == TokenKind::Newline) {
                skipNewlines();
            } else if (lexer.peek().kind == TokenKind::Identifier) {
                std::string argName = std::string(lexer.consume().text);
                skipNewlines();

                // Expect '='
                if (lexer.peek().kind != TokenKind::Equal) {
                    error("Expected '=' after argument name in function call");
                    return inst;
                }
                lexer.consume();
                skipNewlines();

                // Parse argument value
                if (lexer.peek().kind == TokenKind::Identifier ||
                    lexer.peek().kind == TokenKind::FloatLiteral ||
                    lexer.peek().kind == TokenKind::IntegerLiteral) {
                    std::string argValue = std::string(lexer.consume().text);
                    inst.funcCallArgs.push_back({argName, argValue});
                    skipNewlines();

                    // Check for comma
                    if (lexer.peek().kind == TokenKind::Comma) {
                        lexer.consume();
                        skipNewlines();
                    }
                } else {
                    error("Expected argument value in function call");
                    return inst;
                }
            } else {
                error("Unexpected token in function call arguments: " +
                      std::string(lexer.peek().text));
                lexer.consume();
            }
        }

        // Expect ')'
        if (lexer.peek().kind == TokenKind::RightParen) {
            lexer.consume();
        }
        skipNewlines();

        return inst;
    }

    // Normal instruction: dest = operation(arg1, arg2, ...)
    // Parse destination register
    inst.destReg = std::string(lexer.consume().text);
    skipNewlines();

    // Expect '='
    if (lexer.peek().kind != TokenKind::Equal) {
        error("Expected '=' in intrinsic instruction");
        return inst;
    }
    lexer.consume();
    skipNewlines();

    // Parse operation name
    if (lexer.peek().kind != TokenKind::Identifier) {
        error("Expected operation name");
        return inst;
    }
    inst.operation = std::string(lexer.consume().text);
    skipNewlines();

    // Expect '('
    if (lexer.peek().kind != TokenKind::LeftParen) {
        error("Expected '(' after operation name");
        return inst;
    }
    lexer.consume();
    skipNewlines();

    // Parse operands
    while (lexer.peek().kind != TokenKind::RightParen && !lexer.isAtEnd()) {
        if (lexer.peek().kind == TokenKind::Newline) {
            skipNewlines();
        } else if (lexer.peek().kind == TokenKind::Identifier) {
            // Register reference
            inst.operands.push_back(IntrinsicOperand(std::string(lexer.consume().text)));
            skipNewlines();

            // Check for comma
            if (lexer.peek().kind == TokenKind::Comma) {
                lexer.consume();
                skipNewlines();
            }
        } else if (lexer.peek().kind == TokenKind::HexLiteral) {
            // Hex literal - parse as uint32_t and reinterpret as float
            std::string hexText = std::string(lexer.consume().text);
            uint32_t bits = std::stoul(hexText, nullptr, 16);
            float value = std::bit_cast<float>(bits);
            inst.operands.push_back(IntrinsicOperand::hexLiteral(static_cast<double>(value)));
            skipNewlines();

            // Check for comma
            if (lexer.peek().kind == TokenKind::Comma) {
                lexer.consume();
                skipNewlines();
            }
        } else if (lexer.peek().kind == TokenKind::IntegerLiteral) {
            // Integer literal
            std::string intText = std::string(lexer.consume().text);
            int64_t value = std::stoll(intText);
            inst.operands.push_back(IntrinsicOperand(value));
            skipNewlines();

            // Check for comma
            if (lexer.peek().kind == TokenKind::Comma) {
                lexer.consume();
                skipNewlines();
            }
        } else if (lexer.peek().kind == TokenKind::FloatLiteral) {
            // Float literal
            std::string floatText = std::string(lexer.consume().text);
            double value = std::stod(floatText);
            inst.operands.push_back(IntrinsicOperand(value));
            skipNewlines();

            // Check for comma
            if (lexer.peek().kind == TokenKind::Comma) {
                lexer.consume();
                skipNewlines();
            }
        } else {
            // Unexpected token, consume it to avoid infinite loop
            error("Unexpected token in instruction operands: " + std::string(lexer.peek().text));
            lexer.consume();
        }
    }

    // Expect ')'
    if (lexer.peek().kind == TokenKind::RightParen) {
        lexer.consume();
    }
    skipNewlines();

    return inst;
}

std::vector<Pattern> parsePatternFile(const std::string& filename) {
    // Read file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open pattern file: " << filename << "\n";
        return {};
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();

    // Lex
    IRLexer lexer(source);
    lexer.lex();

    // Parse
    PatternParser parser(lexer);
    auto patterns = parser.parsePatterns();

    // Check for errors
    if (parser.hasErrors()) {
        std::cerr << "Pattern parsing errors in " << filename << ":\n";
        parser.printErrors();
        return {};
    }

    return patterns;
}

PatternParseResult parsePatternFileWithDiagnostics(const std::string& filename) {
    PatternParseResult result;

    // Read file
    std::ifstream file(filename);
    if (!file.is_open()) {
        // Create a synthetic diagnostic for file open failure
        result.diagnostics.emplace_back(Diagnostic::Level::Error,
                                        "Failed to open pattern file: " + filename, 0, 0);
        return result;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();

    // Lex
    IRLexer lexer(source);
    lexer.lex();

    // Parse
    PatternParser parser(lexer);
    result.patterns = parser.parsePatterns();
    result.diagnostics = parser.getDiagnostics();

    return result;
}

}  // namespace stinkytofu

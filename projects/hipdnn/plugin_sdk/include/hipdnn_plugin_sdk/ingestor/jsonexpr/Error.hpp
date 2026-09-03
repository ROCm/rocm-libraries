// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// Error.hpp - the compile-time failure type for the JSON Expression Language.
//
// Full reference: docs/JsonExpression.md.

#include <cstddef>
#include <stdexcept>
#include <string>

namespace hipdnn_plugin_sdk::ingestor::jsonexpr
{
/// Thrown when a rule cannot be compiled (unknown operator, bad arity, ...).
class JsonExpressionCompileError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

/// The only variable-reference sigil supported by the expression language.
inline constexpr char VARIABLE_SIGIL = '$';

/// How deeply a rule may nest. Compilation walks the document recursively
/// (rank pins, then alias expansion, then lowering) and evaluation walks the
/// resulting tree the same way, so without a bound a deeply nested rule --
/// and rules are read from descriptor files on disk, not only written by
/// hand -- overflows the stack instead of reporting a bad rule. Compiling is
/// what enforces it, which bounds evaluation too. Far above anything a real
/// criterion or dispatch formula nests.
///
/// All three compile passes must charge depth at the SAME rate for the number
/// below to mean what it says. They share this one bound, and compile() runs
/// them in the order above, so a pass that counts faster than the others
/// silently becomes the real limit while still reporting this one -- and a
/// pass that counts slower hands an over-deep document to the next pass.
/// Compiler.hpp sets the rate: one level per operator, an argument array not
/// being a level of its own.
inline constexpr std::size_t MAX_EXPRESSION_DEPTH = 256;

inline void checkExpressionDepth(std::size_t depth)
{
    if(depth > MAX_EXPRESSION_DEPTH)
    {
        throw JsonExpressionCompileError("expression nests deeper than the limit of "
                                         + std::to_string(MAX_EXPRESSION_DEPTH));
    }
}
} // namespace hipdnn_plugin_sdk::ingestor::jsonexpr

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

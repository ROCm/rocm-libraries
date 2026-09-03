// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// Error.hpp - the compile-time failure type and depth bound for the JSON
// Expression Language.
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

/// How deeply a rule may nest. Rules are read from descriptor files on disk,
/// so the limit has to be enforced rather than assumed. Compilation recurses
/// once per nesting level, and so does evaluation, so an unbounded rule would
/// overflow the stack instead of reporting a bad rule. Compilation checks the
/// depth, which bounds evaluation too. The limit is far above anything a real
/// criterion or dispatch formula nests.
///
/// The three compile passes (rank pins, alias expansion, lowering) share this
/// bound, so they must all count depth the same way. A pass that counts faster
/// becomes the real limit while still reporting this number; a pass that counts
/// slower hands an over-deep document to the next pass. Compiler.hpp defines
/// the rate: one level per operator, and an argument array is not a level.
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

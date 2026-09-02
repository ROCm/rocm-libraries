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

/// How deeply a rule may nest. Compilation walks the document recursively
/// (rank pins, then alias expansion, then lowering) and evaluation walks the
/// resulting tree the same way, so without a bound a deeply nested rule --
/// and rules are read from descriptor files on disk, not only written by
/// hand -- overflows the stack instead of reporting a bad rule. Compiling is
/// what enforces it, which bounds evaluation too. Far above anything a real
/// criterion or dispatch formula nests.
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

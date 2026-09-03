// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// Syntax.hpp - the lexical surface of a rule.
//
// A rule is JSON, so the language has no grammar of its own beyond how it reads
// a JSON string: as a variable reference, or as a literal. This header holds
// that distinction, and nothing else. It is a leaf with no dependencies, so the
// runtime side (a data source resolving a path) and the compile side (lowering
// a rule) share the spelling without either pulling in the other.
//
// Full reference: docs/JsonExpression.md.

namespace hipdnn_plugin_sdk::ingestor::jsonexpr
{
/// The only variable-reference sigil supported by the expression language.
///
/// A JSON string starting with this character is a variable reference; any
/// other string is a literal. Doubling it escapes it, so "$$x" is the literal
/// string "$x" rather than a reference to a variable named "$x", and a lone
/// sigil is not a name and is rejected as malformed. The compiler, the
/// layout-alias pre-pass, and a data source's path parser all read it that way.
///
/// The sigil is fixed rather than configurable: one spelling means a rule reads
/// the same everywhere it is written.
inline constexpr char VARIABLE_SIGIL = '$';
} // namespace hipdnn_plugin_sdk::ingestor::jsonexpr

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

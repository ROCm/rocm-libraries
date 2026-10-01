// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// Syntax.hpp - how a rule's strings are read.
//
// A rule is JSON, so the language has no grammar of its own beyond one
// distinction: a JSON string is either a variable reference or a literal. This
// header holds that distinction and nothing else. It has no includes, so both
// the compile side (lowering a rule) and the runtime side (a data source
// resolving a path) can read the spelling without including the other.
//
// Full reference: docs/JsonExpression.md.

namespace hipdnn_plugin_sdk::ingestor::jsonexpr
{
/// Marks a JSON string as a variable reference. This leading marker character
/// is called the sigil throughout the language.
///
/// A string starting with the sigil is a variable reference; any other string
/// is a literal. Doubling the sigil escapes it, so "$$x" is the literal string
/// "$x" rather than a reference to a variable named "$x". A lone sigil names
/// nothing and is rejected as malformed. The compiler, the layout-alias
/// pre-pass, and a data source's path parser all read it this way.
///
/// The sigil is fixed rather than configurable, so a rule reads the same
/// everywhere it is written.
inline constexpr char VARIABLE_SIGIL = '$';
} // namespace hipdnn_plugin_sdk::ingestor::jsonexpr

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

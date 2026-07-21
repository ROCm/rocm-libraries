// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// PatternCompiler: parses a human-authored JSON criteria document into a
// validated CompiledPattern. This is the runtime ingestion path (the AOT path in
// Phase 6 runs the same compiler at build time and serializes the result), so it
// treats its input as untrusted: every size is bounded and it fails closed with
// a clear message rather than crashing or hanging. It never throws to the caller.
//
// JSON shape (schema "hipdnn.criteria/v1"):
//   {
//     "schema": "hipdnn.criteria/v1",
//     "nodes": [
//       {"id":"mm",  "op":"matmul",    "operands":{"a":"$x","b":"$w"}, "results":{"c":"$h"}},
//       {"id":"act", "op":"pointwise", "operands":{"in_0":"$h"},        "results":{"out_0":"$y"}}
//     ],
//     "constraints": [
//       {"on":"$x", "dtype":{"one_of":["BFLOAT16"]}, "shape":["batch","m","k"],
//       "layout":"contiguous"},
//       {"on":"$w", "shape":["k","n"]},
//       {"on":"$h", "use":"exactly_once"},
//       {"on":"act","attr":{"operation":{"equals":30}}},
//       {"kind":"same_dim", "args":["$x",2,"$w",0]}
//     ]
//   }
// An operand/result value is a var string ("$x") or {"var":"$x","optional":true}.
// A shape element is an int literal, a symbol name, or "?" (wildcard).
// A node may carry "anchor":true to override the default (unique-sink) anchor.

#include <cstddef>
#include <cstdint>
#include <hipdnn_graph_matcher/CompiledPattern.hpp>
#include <string>
#include <string_view>

namespace hipdnn::graph_matcher {

// Hard caps applied while parsing untrusted JSON. Defaults are generous for real
// criteria yet small enough that adversarial input fails fast.
struct CompileLimits {
    size_t maxInputBytes = 1u << 20;  // 1 MiB
    uint32_t maxDepth = 64;           // JSON nesting, checked before parse
    uint32_t maxNodes = 256;
    uint32_t maxEdgesPerNode = 64;
    uint32_t maxConstraints = 1024;
    uint32_t maxShapeDims = 32;
    uint32_t maxSetSize = 64;   // dtype set / attr one_of size
    uint32_t maxNameLen = 256;  // op/var/symbol/attr string length
};

// Outcome of a compile: on success `ok` is true and `pattern` is populated; on
// failure `ok` is false and `error` explains why. Never throws.
struct CompileResult {
    bool ok = false;
    CompiledPattern pattern;
    std::string error;

    // Descriptor metadata parsed from the top-level "name"/"priority" fields
    // (both optional). Consumed by PatternSet for arbitration; not part of the
    // pattern's criterion or its serialized form.
    std::string name;
    int64_t priority = 0;

    explicit operator bool() const noexcept {
        return ok;
    }
};

class PatternCompiler {
   public:
    // Compiles JSON criteria into a CompiledPattern. Returns an error result on
    // malformed JSON, an unknown/newer schema version, an unknown op/role/dtype,
    // a bound exceeded, or any structural problem. Does not throw. `provenance`
    // gates native predicates (DropIn => built-ins only); `predicates` is the
    // registry predicate references are validated against.
    static CompileResult fromJson(
        std::string_view json, const OpSchemaRegistry& registry = OpSchemaRegistry::builtin(),
        const CompileLimits& limits = {}, Provenance provenance = Provenance::Builtin,
        const PredicateRegistry& predicates = PredicateRegistry::builtin());
};

}  // namespace hipdnn::graph_matcher

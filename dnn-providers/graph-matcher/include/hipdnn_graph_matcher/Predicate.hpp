// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Native-predicate escape hatch: a criterion can invoke a C++ predicate the
// declarative vocabulary cannot express. Per RFC 0017 §5 the descriptor carries
// only a symbol NAME plus typed args drawn from the match bindings -- never
// inline code and never the whole graph -- so the same file loads under both the
// build-time and runtime paths. A predicate receives typed BoundArgs (a tensor
// reference or an int-valued dim/literal) and returns a verdict.
//
// Trust (RFC 0017 §10): built-in predicates ship with the library; author/plugin
// predicates are registered separately. Drop-in criteria may reference only
// built-ins (enforced at compile time via Provenance), so an untrusted drop-in
// package cannot pull in an author-shipped predicate.

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace hipdnn::graph_matcher {

// Where a criterion came from. Drives the predicate trust rule.
enum class Provenance : uint8_t {
    Builtin,  // in-tree / AOT: may use any registered predicate
    DropIn    // runtime drop-in: may use built-in predicates only
};

// The type of a resolved predicate argument.
enum class ArgKind : uint8_t {
    Tensor,  // a bound tensor (from a pattern variable)
    Int      // a dim value (from a symbol) or a literal
};

// A resolved argument handed to a predicate at match time.
struct BoundArg {
    ArgKind kind;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* tensor = nullptr;
    int64_t value = 0;
};

// A predicate: pure, side-effect-free, over its typed args only. Arity and arg
// kinds are validated at compile time, so `args` always matches the entry's
// declared kinds and size.
using PredicateFn = bool (*)(const std::vector<BoundArg>& args);

struct PredicateEntry {
    std::string name;
    std::vector<ArgKind> argKinds;
    PredicateFn fn;
    bool builtin;  // true => usable from drop-in criteria
};

// A registry of named predicates. `builtin()` is the process-wide set the
// library ships; a consumer that needs author predicates copies it and registers
// more (those are non-builtin and thus barred from drop-in criteria).
class PredicateRegistry {
   public:
    // The library's built-in predicates.
    static const PredicateRegistry& builtin();

    // Entry by name, or nullptr if unregistered.
    const PredicateEntry* find(std::string_view name) const noexcept;

    // Adds/overrides a predicate (marked non-builtin). For consumers extending a
    // copy of builtin() with author predicates.
    void registerPredicate(PredicateEntry entry);

   private:
    std::vector<PredicateEntry> _entries;
};

}  // namespace hipdnn::graph_matcher

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// CompiledPattern: the structural pattern the matcher runs against a graph.
// Nodes name operand/result edges by variable; a variable shared across nodes
// *is* the edge (merge when it joins a result to an operand, fan-out when it
// feeds several operands). Symbolic dims (SymId) unify across edges: a named dim
// binds a concrete value on first sight and must match everywhere after. Full
// constraint vocabulary (dtype/layout/attribute/use-count) and native predicates
// are later phases; serialization to the flat offset-free form is Phase 6. The
// in-memory shape here is deliberately builder-friendly, not yet the wire form.

#include <cstdint>
#include <hipdnn_graph_matcher/OpSchema.hpp>
#include <hipdnn_graph_matcher/Predicate.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace hipdnn::graph_matcher {

// A pattern variable: the connective tissue between edges. Interned per pattern.
using VarId = uint32_t;

// One operand or result edge of a pattern node. `roleIndex` indexes the op
// schema's operands[] or results[]; `var` is the variable bound to the tensor at
// that role. An `optional` edge binds when the role is present in the graph node
// and is silently skipped when absent (models SDPA bias/mask slots).
struct PatternEdge {
    uint32_t roleIndex;
    VarId var;
    bool optional = false;
};

struct PatternNode {
    std::string opcode;  // concrete opcode (Phase 1: exact match)
    std::vector<PatternEdge> operandEdges;
    std::vector<PatternEdge> resultEdges;
};

// A symbolic dimension: a named value (e.g. "k") that unifies across the whole
// pattern -- first occurrence binds a concrete int64 dim, later occurrences must
// equal it. Interned per pattern, distinct from tensor variables (VarId).
using SymId = uint32_t;

// Binds symbol `sym` to the dimension at logical `axis` of the tensor bound to
// `var`. Resolved during matching once `var` is bound: dims[axis] is read and
// unified into the symbol table.
struct DimBinding {
    VarId var;
    uint32_t axis;
    SymId sym;
};

// Comparison for count/attribute constraints.
enum class Cmp : uint8_t { Eq, NotEq, AtMost, AtLeast, OneOf };

// Layout family for a Layout constraint. Contiguous = fully-packed row-major over
// logical dims; PackedOrder = fully-packed in an explicit axis order (major to
// minor), e.g. NHWC. Both verify exact strides read off the tensor.
enum class LayoutKind : uint8_t { Contiguous, PackedOrder };

enum class ConstraintKind : uint8_t {
    Dtype,              // varA dtype in `ints` (DataType values); negated => not-in
    Rank,               // varA rank == ival
    Shape,              // varA rank == ints.size(); ints[i]>=0 literal dim, <0 skip
    Layout,             // varA layout: layoutKind (+ axisOrder for PackedOrder)
    Attr,               // node attr `name` compared to `ints` via cmp; negated flips
    UseCount,           // varA operand-use count cmp ival
    ConsumerCount,      // varA distinct-consumer-node count cmp ival
    NoConsumerOutside,  // every consumer of varA is a matched pattern node
    SameDtype,          // varA dtype == varB dtype; negated => !=
    SameDim             // varA.axisA dim == varB.axisB dim; negated => !=
};

// A single constraint. Deliberately a flat, pre-wire-format tagged struct (the
// serialized offset-free form is Phase 6); the matcher evaluates it with a switch
// on `kind`. Unused fields stay default for a given kind.
struct Constraint {
    ConstraintKind kind;
    bool negated = false;
    VarId varA = 0;
    VarId varB = 0;
    uint32_t nodeIndex = 0;
    uint32_t axisA = 0;
    uint32_t axisB = 0;
    Cmp cmp = Cmp::Eq;
    int64_t ival = 0;
    LayoutKind layoutKind = LayoutKind::Contiguous;
    std::vector<int64_t> ints{};        // dtype set / shape dims / attr values
    std::vector<uint32_t> axisOrder{};  // PackedOrder axes, major -> minor
    std::string name{};                 // attr name
};

// One dimension of a shape constraint: a literal size, a unifying symbol, or a
// wildcard. Authored via the static factories.
struct DimSpec {
    enum class Kind : uint8_t { Literal, Symbol, Any };
    Kind kind = Kind::Any;
    int64_t literal = 0;
    std::string symbol{};

    static DimSpec lit(int64_t value) {
        return DimSpec{Kind::Literal, value, {}};
    }
    static DimSpec of(std::string_view sym) {
        return DimSpec{Kind::Symbol, 0, std::string{sym}};
    }
    static DimSpec any() {
        return DimSpec{Kind::Any, 0, {}};
    }
};

// A reference to a native predicate in the compiled pattern. `name` resolves to
// a PredicateEntry in the match-time registry (kept as a name so the pattern is
// registry-independent and serializable). Each argument is a variable (-> its
// tensor), a symbol (-> its bound dim value), or a literal int; the source kind
// is validated against the predicate's declared arg kinds at build time.
struct PredicateArg {
    enum class Source : uint8_t { Var, Sym, Literal };
    Source source;
    VarId var = 0;        // Source::Var
    SymId sym = 0;        // Source::Sym
    int64_t literal = 0;  // Source::Literal
};

struct PredicateRef {
    std::string name;
    std::vector<PredicateArg> args;
    bool negated = false;
};

class PatternBuilder;

// An immutable, validated pattern. Construct via PatternBuilder.
class CompiledPattern {
   public:
    const std::vector<PatternNode>& nodes() const noexcept {
        return _nodes;
    }

    size_t nodeCount() const noexcept {
        return _nodes.size();
    }

    // Pattern node index the matcher seeds from (the single sink by default, or
    // the explicitly-set anchor). Drives opcode-indexed candidate lookup.
    uint32_t anchor() const noexcept {
        return _anchor;
    }

    uint32_t varCount() const noexcept {
        return static_cast<uint32_t>(_varNames.size());
    }

    // Author-facing name of a variable (e.g. "$h"), for bindings/diagnostics.
    std::string_view varName(VarId var) const noexcept {
        return var < _varNames.size() ? std::string_view{_varNames[var]} : std::string_view{};
    }

    uint32_t symCount() const noexcept {
        return static_cast<uint32_t>(_symNames.size());
    }

    // Author-facing name of a symbol (e.g. "k"), for bindings/diagnostics.
    std::string_view symName(SymId sym) const noexcept {
        return sym < _symNames.size() ? std::string_view{_symNames[sym]} : std::string_view{};
    }

    // Dimension-to-symbol bindings resolved during matching.
    const std::vector<DimBinding>& dimBindings() const noexcept {
        return _dimBindings;
    }

    // All constraints, evaluated once a full structural match is found.
    const std::vector<Constraint>& constraints() const noexcept {
        return _constraints;
    }

    // Native predicates, evaluated with constraints once a full match is found.
    const std::vector<PredicateRef>& predicates() const noexcept {
        return _predicates;
    }

   private:
    friend class PatternBuilder;
    friend struct PatternCodec;  // serialize/deserialize touch these directly

    std::vector<PatternNode> _nodes;
    std::vector<std::string> _varNames;  // VarId -> name
    std::vector<std::string> _symNames;  // SymId -> name
    std::vector<DimBinding> _dimBindings;
    std::vector<Constraint> _constraints;
    std::vector<PredicateRef> _predicates;
    uint32_t _anchor = 0;
};

// Builds and validates a CompiledPattern against an OpSchema registry. Variable
// names are interned implicitly as edges reference them.
class PatternBuilder {
   public:
    // One edge as authored: a role name on the op and the variable it binds.
    struct EdgeSpec {
        std::string_view role;
        std::string_view var;
        bool optional = false;
    };

    // `provenance` gates which predicates addPredicate may reference (drop-in =>
    // built-ins only). `predicates` is the registry addPredicate validates against.
    explicit PatternBuilder(const OpSchemaRegistry& registry = OpSchemaRegistry::builtin(),
                            Provenance provenance = Provenance::Builtin,
                            const PredicateRegistry& predicates = PredicateRegistry::builtin());

    // Adds a node and returns its index. `operands`/`results` map role names to
    // variable names; unknown variables are interned on first use.
    uint32_t addNode(std::string_view opcode, const std::vector<EdgeSpec>& operands,
                     const std::vector<EdgeSpec>& results);

    // Binds a symbol to the dimension at logical `axis` of `var`'s tensor. The
    // variable must already appear on an edge (added via addNode); the symbol is
    // interned on first use. Symbols unify across the pattern during matching.
    // Throws std::invalid_argument if `var` is unknown.
    PatternBuilder& bindDim(std::string_view var, uint32_t axis, std::string_view sym);

    // --- Constraints (evaluated once a full structural match is found). All
    // reference variables that must already appear on an edge; throw
    // std::invalid_argument on an unknown variable, and (for attributes) an
    // out-of-range node or an attribute the op does not expose. ---

    // Tensor dtype must be one of `dtypes` (exact = single element). `negated`
    // requires the dtype NOT be in the set.
    PatternBuilder& constrainDtype(std::string_view var, std::vector<int32_t> dtypes,
                                   bool negated = false);

    // Tensor rank (number of dims) must equal `rank`.
    PatternBuilder& constrainRank(std::string_view var, uint32_t rank);

    // Tensor shape: rank must equal dims.size(); each literal dim must match,
    // each symbol unifies across the pattern, each Any is a wildcard.
    PatternBuilder& constrainShape(std::string_view var, const std::vector<DimSpec>& dims);

    // Tensor is fully-packed row-major over its logical dims.
    PatternBuilder& constrainContiguous(std::string_view var);

    // Tensor is fully-packed in `axesMajorToMinor` (e.g. NHWC as {0,2,3,1}).
    PatternBuilder& constrainLayout(std::string_view var,
                                    const std::vector<uint32_t>& axesMajorToMinor);

    // Node `nodeIndex`'s scalar attribute `attr` compared to `values` via `cmp`
    // (Eq/NotEq use values[0]; OneOf uses the whole set). `negated` flips.
    PatternBuilder& constrainAttr(uint32_t nodeIndex, std::string_view attr, Cmp cmp,
                                  std::vector<int64_t> values, bool negated = false);

    // Operand-use count of `var` (operand slots referencing it) cmp `n`.
    PatternBuilder& constrainUseCount(std::string_view var, Cmp cmp, int64_t n);

    // Distinct-consumer-node count of `var` cmp `n`.
    PatternBuilder& constrainConsumerCount(std::string_view var, Cmp cmp, int64_t n);

    // Every consumer of `var` is itself a matched pattern node (fusion legality).
    PatternBuilder& constrainNoConsumerOutside(std::string_view var);

    // Cross-tensor: two vars have the same dtype (negated => differ).
    PatternBuilder& constrainSameDtype(std::string_view a, std::string_view b,
                                       bool negated = false);

    // Cross-tensor: a.axisA dim equals b.axisB dim (negated => differ).
    PatternBuilder& constrainSameDim(std::string_view a, uint32_t axisA, std::string_view b,
                                     uint32_t axisB, bool negated = false);

    // One authored predicate argument: a variable ("$x"), a symbol ("k"), or a
    // literal int. Var/symbol must already appear in the pattern.
    struct PredArgSpec {
        PredicateArg::Source source;
        std::string_view name;  // Var or Sym
        int64_t literal = 0;
    };

    // Native predicate `name` over `args`. Throws std::invalid_argument if the
    // predicate is unknown, is non-built-in under DropIn provenance, has the
    // wrong arity, has an arg whose kind mismatches the predicate signature, or
    // references an unknown variable/symbol. `negated` inverts the verdict.
    PatternBuilder& addPredicate(std::string_view name, const std::vector<PredArgSpec>& args,
                                 bool negated = false);

    // Overrides the default anchor (the unique sink, else node 0).
    PatternBuilder& setAnchor(uint32_t nodeIndex);

    // Validates and returns the pattern. Throws std::invalid_argument on an
    // unknown opcode, an unknown role for that opcode, an out-of-range anchor,
    // an empty pattern, or a disconnected pattern (nodes unreachable from the
    // anchor via shared variables -- the matcher cannot walk to them).
    CompiledPattern build() const;

   private:
    struct RawEdge {
        std::string role;
        VarId var;
        bool optional;
    };
    struct RawNode {
        std::string opcode;
        std::vector<RawEdge> operands;
        std::vector<RawEdge> results;
    };

    VarId internVar(std::string_view name);
    SymId internSym(std::string_view name);
    bool findVar(std::string_view name, VarId& out) const;
    VarId requireVar(std::string_view name, const char* what) const;
    bool findSym(std::string_view name, SymId& out) const;

    const OpSchemaRegistry& _registry;
    Provenance _provenance;
    const PredicateRegistry& _predicates;
    std::vector<RawNode> _nodes;
    std::vector<std::string> _varNames;
    std::vector<std::pair<std::string, VarId>> _varIndex;  // name -> id (small, linear)
    std::vector<std::string> _symNames;
    std::vector<std::pair<std::string, SymId>> _symIndex;
    std::vector<DimBinding> _dimBindings;
    std::vector<Constraint> _constraints;
    std::vector<PredicateRef> _predicateRefs;
    bool _hasAnchor = false;
    uint32_t _anchor = 0;
};

}  // namespace hipdnn::graph_matcher

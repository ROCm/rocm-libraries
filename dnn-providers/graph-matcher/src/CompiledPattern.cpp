// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <hipdnn_graph_matcher/CompiledPattern.hpp>
#include <stdexcept>
#include <unordered_set>

namespace hipdnn::graph_matcher {

namespace {

// Index of the role named `name` in `roles`, or -1 if absent.
int findRole(const std::vector<EdgeRole>& roles, std::string_view name) {
    for (size_t i = 0; i < roles.size(); ++i) {
        if (roles[i].name == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

}  // namespace

PatternBuilder::PatternBuilder(const OpSchemaRegistry& registry, Provenance provenance,
                               const PredicateRegistry& predicates)
    : _registry(registry), _provenance(provenance), _predicates(predicates) {}

VarId PatternBuilder::internVar(std::string_view name) {
    for (const auto& [existing, id] : _varIndex) {
        if (existing == name) {
            return id;
        }
    }
    const auto id = static_cast<VarId>(_varNames.size());
    _varNames.emplace_back(name);
    _varIndex.emplace_back(std::string{name}, id);
    return id;
}

SymId PatternBuilder::internSym(std::string_view name) {
    for (const auto& [existing, id] : _symIndex) {
        if (existing == name) {
            return id;
        }
    }
    const auto id = static_cast<SymId>(_symNames.size());
    _symNames.emplace_back(name);
    _symIndex.emplace_back(std::string{name}, id);
    return id;
}

bool PatternBuilder::findVar(std::string_view name, VarId& out) const {
    for (const auto& [existing, id] : _varIndex) {
        if (existing == name) {
            out = id;
            return true;
        }
    }
    return false;
}

bool PatternBuilder::findSym(std::string_view name, SymId& out) const {
    for (const auto& [existing, id] : _symIndex) {
        if (existing == name) {
            out = id;
            return true;
        }
    }
    return false;
}

PatternBuilder& PatternBuilder::bindDim(std::string_view var, uint32_t axis, std::string_view sym) {
    VarId varId = 0;
    if (!findVar(var, varId)) {
        throw std::invalid_argument("PatternBuilder::bindDim: unknown variable '" +
                                    std::string{var} + "' (must appear on an edge first)");
    }
    _dimBindings.push_back(DimBinding{varId, axis, internSym(sym)});
    return *this;
}

uint32_t PatternBuilder::addNode(std::string_view opcode, const std::vector<EdgeSpec>& operands,
                                 const std::vector<EdgeSpec>& results) {
    const OpSchema* schema = _registry.findByOpcode(opcode);
    if (schema == nullptr) {
        throw std::invalid_argument("PatternBuilder::addNode: unknown opcode '" +
                                    std::string{opcode} + "'");
    }

    RawNode node;
    node.opcode = std::string{opcode};

    const auto resolve = [&](const std::vector<EdgeSpec>& specs, const std::vector<EdgeRole>& roles,
                             const char* kind, std::vector<RawEdge>& out) {
        for (const auto& spec : specs) {
            if (findRole(roles, spec.role) < 0) {
                throw std::invalid_argument("PatternBuilder::addNode: op '" + std::string{opcode} +
                                            "' has no " + kind + " role '" +
                                            std::string{spec.role} + "'");
            }
            out.push_back(RawEdge{std::string{spec.role}, internVar(spec.var), spec.optional});
        }
    };

    resolve(operands, schema->operands, "operand", node.operands);
    resolve(results, schema->results, "result", node.results);

    _nodes.push_back(std::move(node));
    return static_cast<uint32_t>(_nodes.size() - 1);
}

PatternBuilder& PatternBuilder::setAnchor(uint32_t nodeIndex) {
    _hasAnchor = true;
    _anchor = nodeIndex;
    return *this;
}

CompiledPattern PatternBuilder::build() const {
    if (_nodes.empty()) {
        throw std::invalid_argument("PatternBuilder::build: pattern has no nodes");
    }

    CompiledPattern pattern;
    pattern._varNames = _varNames;
    pattern._symNames = _symNames;
    pattern._dimBindings = _dimBindings;
    pattern._constraints = _constraints;
    pattern._predicates = _predicateRefs;

    // Lower raw nodes to role-index edges (roles are re-resolved against the
    // schema so the compiled form carries no strings on the match hot path).
    pattern._nodes.reserve(_nodes.size());
    for (const auto& raw : _nodes) {
        const OpSchema* schema = _registry.findByOpcode(raw.opcode);
        PatternNode node;
        node.opcode = raw.opcode;
        for (const auto& edge : raw.operands) {
            node.operandEdges.push_back(
                PatternEdge{static_cast<uint32_t>(findRole(schema->operands, edge.role)), edge.var,
                            edge.optional});
        }
        for (const auto& edge : raw.results) {
            node.resultEdges.push_back(
                PatternEdge{static_cast<uint32_t>(findRole(schema->results, edge.role)), edge.var,
                            edge.optional});
        }
        pattern._nodes.push_back(std::move(node));
    }

    // Resolve the anchor: explicit override, else the unique sink (a node whose
    // result variables feed no other node's operands), else node 0.
    if (_hasAnchor) {
        if (_anchor >= _nodes.size()) {
            throw std::invalid_argument("PatternBuilder::build: anchor index out of range");
        }
        pattern._anchor = _anchor;
    } else {
        std::unordered_set<VarId> consumedVars;
        for (const auto& node : pattern._nodes) {
            for (const auto& edge : node.operandEdges) {
                consumedVars.insert(edge.var);
            }
        }
        std::vector<uint32_t> sinks;
        for (uint32_t i = 0; i < pattern._nodes.size(); ++i) {
            const bool anyConsumed = std::any_of(
                pattern._nodes[i].resultEdges.begin(), pattern._nodes[i].resultEdges.end(),
                [&](const PatternEdge& e) { return consumedVars.count(e.var) != 0; });
            if (!anyConsumed) {
                sinks.push_back(i);
            }
        }
        pattern._anchor = (sinks.size() == 1) ? sinks.front() : 0;
    }

    // Connectivity: every node must be reachable from the anchor via shared
    // variables, else the matcher cannot walk to it. BFS over var adjacency.
    const size_t n = pattern._nodes.size();
    std::vector<bool> seen(n, false);
    std::vector<uint32_t> frontier{pattern._anchor};
    seen[pattern._anchor] = true;
    const auto nodeVars = [&](uint32_t idx) {
        std::unordered_set<VarId> vars;
        for (const auto& e : pattern._nodes[idx].operandEdges) {
            vars.insert(e.var);
        }
        for (const auto& e : pattern._nodes[idx].resultEdges) {
            vars.insert(e.var);
        }
        return vars;
    };
    while (!frontier.empty()) {
        const uint32_t cur = frontier.back();
        frontier.pop_back();
        const auto curVars = nodeVars(cur);
        for (uint32_t other = 0; other < n; ++other) {
            if (seen[other]) {
                continue;
            }
            const auto otherVars = nodeVars(other);
            const bool shares = std::any_of(otherVars.begin(), otherVars.end(),
                                            [&](VarId v) { return curVars.count(v) != 0; });
            if (shares) {
                seen[other] = true;
                frontier.push_back(other);
            }
        }
    }
    if (std::any_of(seen.begin(), seen.end(), [](bool s) { return !s; })) {
        throw std::invalid_argument(
            "PatternBuilder::build: pattern is disconnected (a node shares no variable "
            "with the anchor's component)");
    }

    return pattern;
}

VarId PatternBuilder::requireVar(std::string_view name, const char* what) const {
    VarId id = 0;
    if (!findVar(name, id)) {
        throw std::invalid_argument(std::string{what} + ": unknown variable '" + std::string{name} +
                                    "' (must appear on an edge first)");
    }
    return id;
}

PatternBuilder& PatternBuilder::constrainDtype(std::string_view var, std::vector<int32_t> dtypes,
                                               bool negated) {
    const VarId v = requireVar(var, "PatternBuilder::constrainDtype");
    Constraint c;
    c.kind = ConstraintKind::Dtype;
    c.varA = v;
    c.negated = negated;
    c.ints.assign(dtypes.begin(), dtypes.end());
    _constraints.push_back(std::move(c));
    return *this;
}

PatternBuilder& PatternBuilder::constrainRank(std::string_view var, uint32_t rank) {
    const VarId v = requireVar(var, "PatternBuilder::constrainRank");
    Constraint c;
    c.kind = ConstraintKind::Rank;
    c.varA = v;
    c.ival = static_cast<int64_t>(rank);
    _constraints.push_back(std::move(c));
    return *this;
}

PatternBuilder& PatternBuilder::constrainShape(std::string_view var,
                                               const std::vector<DimSpec>& dims) {
    const VarId v = requireVar(var, "PatternBuilder::constrainShape");
    Constraint c;
    c.kind = ConstraintKind::Shape;
    c.varA = v;
    c.ints.reserve(dims.size());
    for (uint32_t axis = 0; axis < dims.size(); ++axis) {
        const DimSpec& d = dims[axis];
        switch (d.kind) {
            case DimSpec::Kind::Literal:
                c.ints.push_back(d.literal);  // literals are >= 0 sizes
                break;
            case DimSpec::Kind::Symbol:
                c.ints.push_back(-1);  // symbol handled via DimBinding below
                _dimBindings.push_back(DimBinding{v, axis, internSym(d.symbol)});
                break;
            case DimSpec::Kind::Any:
                c.ints.push_back(-1);
                break;
        }
    }
    _constraints.push_back(std::move(c));
    return *this;
}

PatternBuilder& PatternBuilder::constrainContiguous(std::string_view var) {
    const VarId v = requireVar(var, "PatternBuilder::constrainContiguous");
    Constraint c;
    c.kind = ConstraintKind::Layout;
    c.varA = v;
    c.layoutKind = LayoutKind::Contiguous;
    _constraints.push_back(std::move(c));
    return *this;
}

PatternBuilder& PatternBuilder::constrainLayout(std::string_view var,
                                                const std::vector<uint32_t>& axesMajorToMinor) {
    const VarId v = requireVar(var, "PatternBuilder::constrainLayout");
    Constraint c;
    c.kind = ConstraintKind::Layout;
    c.varA = v;
    c.layoutKind = LayoutKind::PackedOrder;
    c.axisOrder = axesMajorToMinor;
    _constraints.push_back(std::move(c));
    return *this;
}

PatternBuilder& PatternBuilder::constrainAttr(uint32_t nodeIndex, std::string_view attr, Cmp cmp,
                                              std::vector<int64_t> values, bool negated) {
    if (nodeIndex >= _nodes.size()) {
        throw std::invalid_argument("PatternBuilder::constrainAttr: node index out of range");
    }
    const OpSchema* schema = _registry.findByOpcode(_nodes[nodeIndex].opcode);
    if (schema == nullptr || schema->findAttr(attr) == nullptr) {
        throw std::invalid_argument("PatternBuilder::constrainAttr: op '" +
                                    _nodes[nodeIndex].opcode + "' has no attribute '" +
                                    std::string{attr} + "'");
    }
    Constraint c;
    c.kind = ConstraintKind::Attr;
    c.nodeIndex = nodeIndex;
    c.cmp = cmp;
    c.negated = negated;
    c.name = std::string{attr};
    c.ints = std::move(values);
    _constraints.push_back(std::move(c));
    return *this;
}

PatternBuilder& PatternBuilder::constrainUseCount(std::string_view var, Cmp cmp, int64_t n) {
    const VarId v = requireVar(var, "PatternBuilder::constrainUseCount");
    Constraint c;
    c.kind = ConstraintKind::UseCount;
    c.varA = v;
    c.cmp = cmp;
    c.ival = n;
    _constraints.push_back(std::move(c));
    return *this;
}

PatternBuilder& PatternBuilder::constrainConsumerCount(std::string_view var, Cmp cmp, int64_t n) {
    const VarId v = requireVar(var, "PatternBuilder::constrainConsumerCount");
    Constraint c;
    c.kind = ConstraintKind::ConsumerCount;
    c.varA = v;
    c.cmp = cmp;
    c.ival = n;
    _constraints.push_back(std::move(c));
    return *this;
}

PatternBuilder& PatternBuilder::constrainNoConsumerOutside(std::string_view var) {
    const VarId v = requireVar(var, "PatternBuilder::constrainNoConsumerOutside");
    Constraint c;
    c.kind = ConstraintKind::NoConsumerOutside;
    c.varA = v;
    _constraints.push_back(std::move(c));
    return *this;
}

PatternBuilder& PatternBuilder::constrainSameDtype(std::string_view a, std::string_view b,
                                                   bool negated) {
    Constraint c;
    c.kind = ConstraintKind::SameDtype;
    c.varA = requireVar(a, "PatternBuilder::constrainSameDtype");
    c.varB = requireVar(b, "PatternBuilder::constrainSameDtype");
    c.negated = negated;
    _constraints.push_back(std::move(c));
    return *this;
}

PatternBuilder& PatternBuilder::constrainSameDim(std::string_view a, uint32_t axisA,
                                                 std::string_view b, uint32_t axisB, bool negated) {
    Constraint c;
    c.kind = ConstraintKind::SameDim;
    c.varA = requireVar(a, "PatternBuilder::constrainSameDim");
    c.axisA = axisA;
    c.varB = requireVar(b, "PatternBuilder::constrainSameDim");
    c.axisB = axisB;
    c.negated = negated;
    _constraints.push_back(std::move(c));
    return *this;
}

PatternBuilder& PatternBuilder::addPredicate(std::string_view name,
                                             const std::vector<PredArgSpec>& args, bool negated) {
    const PredicateEntry* entry = _predicates.find(name);
    if (entry == nullptr) {
        throw std::invalid_argument("PatternBuilder::addPredicate: unknown predicate '" +
                                    std::string{name} + "'");
    }
    if (_provenance == Provenance::DropIn && !entry->builtin) {
        throw std::invalid_argument("PatternBuilder::addPredicate: predicate '" +
                                    std::string{name} +
                                    "' is not built-in and cannot be used by a drop-in pattern");
    }
    if (args.size() != entry->argKinds.size()) {
        throw std::invalid_argument(
            "PatternBuilder::addPredicate: predicate '" + std::string{name} + "' expects " +
            std::to_string(entry->argKinds.size()) + " args, got " + std::to_string(args.size()));
    }

    PredicateRef ref;
    ref.name = std::string{name};
    ref.negated = negated;
    for (size_t i = 0; i < args.size(); ++i) {
        const PredArgSpec& spec = args[i];
        const ArgKind expected = entry->argKinds[i];
        // A variable resolves to a Tensor; a symbol or literal resolves to an Int.
        const ArgKind actual =
            (spec.source == PredicateArg::Source::Var) ? ArgKind::Tensor : ArgKind::Int;
        if (actual != expected) {
            throw std::invalid_argument("PatternBuilder::addPredicate: predicate '" +
                                        std::string{name} + "' arg " + std::to_string(i) +
                                        " kind mismatch");
        }
        PredicateArg out;
        out.source = spec.source;
        switch (spec.source) {
            case PredicateArg::Source::Var:
                out.var = requireVar(spec.name, "PatternBuilder::addPredicate");
                break;
            case PredicateArg::Source::Sym:
                if (!findSym(spec.name, out.sym)) {
                    throw std::invalid_argument("PatternBuilder::addPredicate: unknown symbol '" +
                                                std::string{spec.name} +
                                                "' (bind it via a shape/bindDim first)");
                }
                break;
            case PredicateArg::Source::Literal:
                out.literal = spec.literal;
                break;
        }
        ref.args.push_back(out);
    }
    _predicateRefs.push_back(std::move(ref));
    return *this;
}

}  // namespace hipdnn::graph_matcher

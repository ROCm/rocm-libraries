// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <hipdnn_graph_matcher/Matcher.hpp>
#include <unordered_map>
#include <unordered_set>

namespace hipdnn::graph_matcher {

namespace {

// Placement order: BFS from the anchor over pattern var-adjacency, so every
// non-anchor node shares a variable with an already-placed node and thus has a
// bound edge to seed candidate generation from. Connectivity is guaranteed by
// PatternBuilder::build(), so this reaches every node.
std::vector<uint32_t> placementOrder(const CompiledPattern& pattern) {
    const auto& nodes = pattern.nodes();
    const size_t n = nodes.size();

    const auto varsOf = [&](uint32_t idx) {
        std::unordered_set<VarId> vars;
        for (const auto& e : nodes[idx].operandEdges) {
            vars.insert(e.var);
        }
        for (const auto& e : nodes[idx].resultEdges) {
            vars.insert(e.var);
        }
        return vars;
    };

    std::vector<uint32_t> order;
    order.reserve(n);
    std::vector<bool> seen(n, false);

    order.push_back(pattern.anchor());
    seen[pattern.anchor()] = true;
    for (size_t head = 0; head < order.size(); ++head) {
        const auto curVars = varsOf(order[head]);
        for (uint32_t other = 0; other < n; ++other) {
            if (seen[other]) {
                continue;
            }
            const auto otherVars = varsOf(other);
            if (std::any_of(otherVars.begin(), otherVars.end(),
                            [&](VarId v) { return curVars.count(v) != 0; })) {
                seen[other] = true;
                order.push_back(other);
            }
        }
    }
    return order;
}

// Rollback log for one placement attempt: the variables and symbols newly bound
// while unifying a candidate, undone on backtrack so a failed branch leaves no
// residue (all-or-nothing commit).
struct Trail {
    std::vector<VarId> vars;
    std::vector<SymId> syms;
};

class Search {
   public:
    Search(const CompiledPattern& pattern, const GraphView& graph, const MatchOptions& options,
           const PredicateRegistry& predicates)
        : _pattern(pattern),
          _graph(graph),
          _options(options),
          _predicates(predicates),
          _order(placementOrder(pattern)),
          _bound(pattern.varCount(), false),
          _varUid(pattern.varCount(), 0),
          _nodeMap(pattern.nodeCount(), kUnmapped),
          _symBound(pattern.symCount(), false),
          _symVal(pattern.symCount(), 0) {
        const uint32_t graphNodes = _graph.graph().nodeCount();
        _usedGraph.assign(graphNodes, false);
        for (uint32_t i = 0; i < graphNodes; ++i) {
            const std::string_view opcode = _graph.opcodeOf(i);
            if (!opcode.empty()) {
                _byOpcode[opcode].push_back(i);
            }
        }
        for (const auto& db : _pattern.dimBindings()) {
            _dimBindingsByVar[db.var].push_back(db);
        }
        // Resolve each predicate name to its fn once (registry-independent
        // pattern); an unresolved predicate stays null and fails the match.
        for (const auto& ref : _pattern.predicates()) {
            const PredicateEntry* entry = _predicates.find(ref.name);
            _resolvedPredicates.push_back(entry != nullptr ? entry->fn : nullptr);
        }
    }

    MatchResult run() {
        MatchResult result;
        if (place(0)) {
            result.matched = true;
            result.varUids = _varUid;
            result.varBound = _bound;
            result.symVals = _symVal;
            result.nodeMap = _nodeMap;
        }
        result.budgetExceeded = _aborted;
        if (!result.matched && _options.explain) {
            result.diagnostic =
                _diagnostic.empty()
                    ? (_reachedFullPlacement
                           ? "no placement satisfied all constraints"
                           : "no structural placement (opcodes/edges did not match)")
                    : _diagnostic;
        }
        return result;
    }

   private:
    static constexpr uint32_t kUnmapped = ~0u;

    bool place(size_t orderIdx) {
        if (_aborted) {
            return false;
        }
        if (orderIdx == _order.size()) {
            // Structure complete: all nodes mapped, all non-optional vars bound.
            // Constraints are checked here so a failure backtracks into the next
            // structural arrangement rather than aborting the whole search.
            return validateConstraints();
        }

        const uint32_t patternNode = _order[orderIdx];
        for (const uint32_t cand : candidatesFor(patternNode)) {
            if (++_steps > _options.stepBudget) {
                _aborted = true;
                return false;
            }
            if (_usedGraph[cand]) {
                continue;
            }
            if (_graph.opcodeOf(cand) != _pattern.nodes()[patternNode].opcode) {
                continue;
            }

            Trail journal;
            if (!unify(patternNode, cand, journal)) {
                undo(journal);
                continue;
            }

            _nodeMap[patternNode] = cand;
            _usedGraph[cand] = true;
            if (place(orderIdx + 1)) {
                return true;
            }
            _usedGraph[cand] = false;
            _nodeMap[patternNode] = kUnmapped;
            undo(journal);
            if (_aborted) {
                return false;
            }
        }
        return false;
    }

    // Binds this pattern node's edge variables to `cand`'s tensor UIDs, failing
    // on a unification conflict. Each referenced role must resolve to exactly one
    // UID (variadic/absent-optional handling is a later phase). Newly-bound
    // variables also drive symbolic-dim unification.
    bool unify(uint32_t patternNode, uint32_t cand, Trail& journal) {
        const PatternNode& pn = _pattern.nodes()[patternNode];
        for (const auto& edge : pn.operandEdges) {
            if (!unifyEdge(cand, /*result=*/false, edge, journal)) {
                return false;
            }
        }
        for (const auto& edge : pn.resultEdges) {
            if (!unifyEdge(cand, /*result=*/true, edge, journal)) {
                return false;
            }
        }
        return true;
    }

    bool unifyEdge(uint32_t cand, bool result, const PatternEdge& edge, Trail& journal) {
        const std::vector<int64_t> uids = _graph.roleUids(cand, result, edge.roleIndex);
        if (uids.empty() && edge.optional) {
            return true;  // absent optional slot: skip, leave var unbound
        }
        if (uids.size() != 1) {
            return false;  // required role absent, or multi-valued (variadic: Phase 3+)
        }
        const int64_t uid = uids.front();
        // A referenced role whose uid names no tensor in the graph is a dangling
        // edge: absent for an optional slot, a hard mismatch for a required one.
        // This keeps a bound var backed by a real tensor, so value constraints
        // never silently skip.
        if (_graph.tensor(uid) == nullptr) {
            return edge.optional;
        }
        if (_bound[edge.var]) {
            return _varUid[edge.var] == uid;
        }
        _bound[edge.var] = true;
        _varUid[edge.var] = uid;
        journal.vars.push_back(edge.var);
        // A var's first binding fixes any symbolic dims declared on it.
        return unifyDims(edge.var, uid, journal);
    }

    // Unifies each symbol declared on `var` against the bound tensor's dims.
    // Fails closed on an out-of-range axis (fewer dims than the pattern asserts)
    // or a symbol-value conflict across edges.
    bool unifyDims(VarId var, int64_t uid, Trail& journal) {
        const auto it = _dimBindingsByVar.find(var);
        if (it == _dimBindingsByVar.end()) {
            return true;
        }
        const TensorAttributes* tensor = _graph.tensor(uid);
        const auto* dims = (tensor != nullptr) ? tensor->dims() : nullptr;
        for (const DimBinding& db : it->second) {
            if (dims == nullptr || db.axis >= dims->size()) {
                return false;
            }
            const int64_t dim = dims->Get(db.axis);
            if (_symBound[db.sym]) {
                if (_symVal[db.sym] != dim) {
                    return false;
                }
            } else {
                _symBound[db.sym] = true;
                _symVal[db.sym] = dim;
                journal.syms.push_back(db.sym);
            }
        }
        return true;
    }

    void undo(const Trail& journal) {
        for (const VarId var : journal.vars) {
            _bound[var] = false;
        }
        for (const SymId sym : journal.syms) {
            _symBound[sym] = false;
        }
    }

    static bool cmpInt(int64_t lhs, Cmp cmp, int64_t rhs) {
        switch (cmp) {
            case Cmp::Eq:
                return lhs == rhs;
            case Cmp::NotEq:
                return lhs != rhs;
            case Cmp::AtMost:
                return lhs <= rhs;
            case Cmp::AtLeast:
                return lhs >= rhs;
            case Cmp::OneOf:
                return lhs == rhs;  // handled elementwise by caller
        }
        return false;
    }

    const TensorAttributes* tensorOf(VarId var) const {
        return _bound[var] ? _graph.tensor(_varUid[var]) : nullptr;
    }

    // Evaluates every constraint against the completed structural match. Returns
    // false (triggering backtracking) on the first unsatisfied constraint. A
    // constraint on an unbound var (an absent optional) is skipped.
    bool validateConstraints() {
        _reachedFullPlacement = true;
        std::unordered_set<uint32_t> matchedNodes(_nodeMap.begin(), _nodeMap.end());
        for (size_t i = 0; i < _pattern.constraints().size(); ++i) {
            if (!checkConstraint(_pattern.constraints()[i], matchedNodes)) {
                if (_options.explain && _diagnostic.empty()) {
                    _diagnostic = describeConstraint(i, _pattern.constraints()[i]);
                }
                return false;
            }
        }
        return checkPredicates();
    }

    // A terse, structured note naming the rejecting constraint (for the explainer).
    std::string describeConstraint(size_t index, const Constraint& c) const {
        static const char* kNames[] = {
            "dtype",      "rank",      "shape",          "layout",
            "attr",       "use_count", "consumer_count", "no_consumer_outside",
            "same_dtype", "same_dim"};
        std::string subject;
        if (c.kind == ConstraintKind::Attr) {
            subject = "node#" + std::to_string(c.nodeIndex) + " attr='" + c.name + "'";
        } else {
            subject = "var='" + std::string{_pattern.varName(c.varA)} + "'";
            if (c.kind == ConstraintKind::SameDtype || c.kind == ConstraintKind::SameDim) {
                subject += ",'" + std::string{_pattern.varName(c.varB)} + "'";
            }
        }
        return "constraint#" + std::to_string(index) + " [" + kNames[static_cast<size_t>(c.kind)] +
               (c.negated ? ",negated" : "") + "] on " + subject + " rejected";
    }

    // Evaluates each native predicate. A predicate whose args aren't all bound
    // (e.g. an absent optional var, an unbound symbol) is skipped, like a
    // constraint on an unbound subject. An unresolved predicate fails closed.
    bool checkPredicates() {
        const auto& refs = _pattern.predicates();
        for (size_t i = 0; i < refs.size(); ++i) {
            const PredicateRef& ref = refs[i];
            std::vector<BoundArg> args;
            args.reserve(ref.args.size());
            bool allBound = true;
            for (const PredicateArg& a : ref.args) {
                BoundArg ba;
                switch (a.source) {
                    case PredicateArg::Source::Var:
                        if (!_bound[a.var]) {
                            allBound = false;
                        }
                        ba.kind = ArgKind::Tensor;
                        ba.tensor = _bound[a.var] ? _graph.tensor(_varUid[a.var]) : nullptr;
                        break;
                    case PredicateArg::Source::Sym:
                        if (!_symBound[a.sym]) {
                            allBound = false;
                        }
                        ba.kind = ArgKind::Int;
                        ba.value = _symBound[a.sym] ? _symVal[a.sym] : 0;
                        break;
                    case PredicateArg::Source::Literal:
                        ba.kind = ArgKind::Int;
                        ba.value = a.literal;
                        break;
                }
                args.push_back(ba);
            }
            if (!allBound) {
                continue;  // defer: some arg unbound -> treat as satisfied
            }
            const PredicateFn fn = _resolvedPredicates[i];
            if (fn == nullptr) {
                if (_options.explain && _diagnostic.empty()) {
                    _diagnostic = "predicate#" + std::to_string(i) + " '" + ref.name +
                                  "' unresolved in match-time registry";
                }
                return false;  // predicate not available in the match-time registry
            }
            const bool verdict = fn(args);
            if ((ref.negated ? !verdict : verdict) == false) {
                if (_options.explain && _diagnostic.empty()) {
                    _diagnostic = "predicate#" + std::to_string(i) + " '" + ref.name +
                                  (ref.negated ? "' (negated) rejected" : "' rejected");
                }
                return false;
            }
        }
        return true;
    }

    bool checkConstraint(const Constraint& c, const std::unordered_set<uint32_t>& matchedNodes) {
        switch (c.kind) {
            case ConstraintKind::Dtype: {
                const TensorAttributes* t = tensorOf(c.varA);
                if (t == nullptr) return true;
                const int64_t dt = static_cast<int64_t>(t->data_type());
                bool inSet = false;
                for (const int64_t d : c.ints) {
                    if (d == dt) {
                        inSet = true;
                        break;
                    }
                }
                return c.negated ? !inSet : inSet;
            }
            case ConstraintKind::Rank: {
                const TensorAttributes* t = tensorOf(c.varA);
                if (t == nullptr) return true;
                const auto* dims = t->dims();
                const int64_t rank = dims ? static_cast<int64_t>(dims->size()) : 0;
                return rank == c.ival;
            }
            case ConstraintKind::Shape: {
                const TensorAttributes* t = tensorOf(c.varA);
                if (t == nullptr) return true;
                const auto* dims = t->dims();
                const size_t rank = dims ? dims->size() : 0;
                if (rank != c.ints.size()) return false;
                for (size_t i = 0; i < c.ints.size(); ++i) {
                    if (c.ints[i] >= 0 && dims->Get(static_cast<uint32_t>(i)) != c.ints[i]) {
                        return false;  // literal mismatch (symbols handled via DimBinding)
                    }
                }
                return true;
            }
            case ConstraintKind::Layout:
                return checkLayout(c);
            case ConstraintKind::Attr:
                return checkAttr(c);
            case ConstraintKind::UseCount: {
                if (!_bound[c.varA]) return true;
                return cmpInt(static_cast<int64_t>(_graph.useCount(_varUid[c.varA])), c.cmp,
                              c.ival);
            }
            case ConstraintKind::ConsumerCount: {
                if (!_bound[c.varA]) return true;
                return cmpInt(static_cast<int64_t>(_graph.consumerNodeCount(_varUid[c.varA])),
                              c.cmp, c.ival);
            }
            case ConstraintKind::NoConsumerOutside: {
                if (!_bound[c.varA]) return true;
                for (const auto& ep : _graph.consumersOf(_varUid[c.varA])) {
                    if (matchedNodes.count(ep.nodeIndex) == 0) return false;
                }
                return true;
            }
            case ConstraintKind::SameDtype: {
                const TensorAttributes* a = tensorOf(c.varA);
                const TensorAttributes* b = tensorOf(c.varB);
                if (a == nullptr || b == nullptr) return true;
                const bool same = a->data_type() == b->data_type();
                return c.negated ? !same : same;
            }
            case ConstraintKind::SameDim: {
                const TensorAttributes* a = tensorOf(c.varA);
                const TensorAttributes* b = tensorOf(c.varB);
                if (a == nullptr || b == nullptr) return true;
                const auto* da = a->dims();
                const auto* db = b->dims();
                if (da == nullptr || db == nullptr || c.axisA >= da->size() ||
                    c.axisB >= db->size()) {
                    return false;  // out-of-range axis: fail closed
                }
                const bool same = da->Get(c.axisA) == db->Get(c.axisB);
                return c.negated ? !same : same;
            }
        }
        return true;
    }

    bool checkLayout(const Constraint& c) {
        const TensorAttributes* t = tensorOf(c.varA);
        if (t == nullptr) return true;
        const auto* dims = t->dims();
        const auto* strides = t->strides();
        if (dims == nullptr || strides == nullptr || dims->size() != strides->size()) {
            return false;
        }
        const uint32_t rank = dims->size();
        // Axis order major->minor: contiguous is [0,1,...,rank-1].
        std::vector<uint32_t> order;
        if (c.layoutKind == LayoutKind::Contiguous) {
            order.resize(rank);
            for (uint32_t i = 0; i < rank; ++i) order[i] = i;
        } else {
            if (c.axisOrder.size() != rank) return false;
            order = c.axisOrder;
        }
        // Fully-packed strides: minor-most axis stride 1, each next = prev*dim.
        int64_t expected = 1;
        for (uint32_t k = rank; k-- > 0;) {
            const uint32_t axis = order[k];
            if (axis >= rank) return false;
            if (strides->Get(axis) != expected) return false;
            expected *= dims->Get(axis);
        }
        return true;
    }

    bool checkAttr(const Constraint& c) {
        const std::optional<int64_t> value = _graph.attrInt(_nodeMap[c.nodeIndex], c.name);
        bool base = false;
        if (value) {
            if (c.cmp == Cmp::OneOf) {
                for (const int64_t v : c.ints) {
                    if (v == *value) {
                        base = true;
                        break;
                    }
                }
            } else if (!c.ints.empty()) {
                base = cmpInt(*value, c.cmp, c.ints.front());
            }
        }
        return c.negated ? !base : base;
    }

    // Graph nodes that could match `patternNode`, narrowed by its already-bound
    // edges: an operand edge to a bound var -> that var's consumer nodes; a
    // result edge to a bound var -> that var's single producer. With no bound
    // edge (the anchor), fall back to all graph nodes of the right opcode.
    std::vector<uint32_t> candidatesFor(uint32_t patternNode) {
        const PatternNode& pn = _pattern.nodes()[patternNode];
        std::vector<std::vector<uint32_t>> sets;

        for (const auto& edge : pn.operandEdges) {
            if (!_bound[edge.var]) {
                continue;
            }
            std::vector<uint32_t> nodes;
            for (const auto& ep : _graph.consumersOf(_varUid[edge.var])) {
                nodes.push_back(ep.nodeIndex);
            }
            sets.push_back(std::move(nodes));
        }
        for (const auto& edge : pn.resultEdges) {
            if (!_bound[edge.var]) {
                continue;
            }
            std::vector<uint32_t> nodes;
            if (const Endpoint* producer = _graph.producerOf(_varUid[edge.var])) {
                nodes.push_back(producer->nodeIndex);
            }
            sets.push_back(std::move(nodes));
        }

        if (sets.empty()) {
            const auto it = _byOpcode.find(pn.opcode);
            return it == _byOpcode.end() ? std::vector<uint32_t>{} : it->second;
        }
        return intersect(sets);
    }

    static std::vector<uint32_t> intersect(std::vector<std::vector<uint32_t>>& sets) {
        // Smallest set first bounds the work.
        std::sort(sets.begin(), sets.end(),
                  [](const auto& a, const auto& b) { return a.size() < b.size(); });
        std::vector<uint32_t> result = sets.front();
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        for (size_t i = 1; i < sets.size() && !result.empty(); ++i) {
            std::unordered_set<uint32_t> other(sets[i].begin(), sets[i].end());
            result.erase(std::remove_if(result.begin(), result.end(),
                                        [&](uint32_t v) { return other.count(v) == 0; }),
                         result.end());
        }
        return result;
    }

    const CompiledPattern& _pattern;
    const GraphView& _graph;
    const MatchOptions& _options;
    const PredicateRegistry& _predicates;

    std::vector<uint32_t> _order;
    std::vector<bool> _bound;
    std::vector<int64_t> _varUid;
    std::vector<uint32_t> _nodeMap;
    std::vector<bool> _usedGraph;
    std::unordered_map<std::string_view, std::vector<uint32_t>> _byOpcode;
    std::vector<bool> _symBound;
    std::vector<int64_t> _symVal;
    std::unordered_map<VarId, std::vector<DimBinding>> _dimBindingsByVar;
    std::vector<PredicateFn> _resolvedPredicates;

    size_t _steps = 0;
    bool _aborted = false;
    bool _reachedFullPlacement = false;
    std::string _diagnostic;
};

}  // namespace

MatchResult Matcher::match(const CompiledPattern& pattern, const GraphView& graph,
                           const MatchOptions& options, const PredicateRegistry& predicates) {
    Search search(pattern, graph, options, predicates);
    return search.run();
}

}  // namespace hipdnn::graph_matcher

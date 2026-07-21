// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_graph_matcher/Predicate.hpp>

namespace hipdnn::graph_matcher {

namespace {

// Last logical dimension of a tensor arg, or -1 if it has no dims.
int64_t lastDim(const BoundArg& a) {
    const auto* dims = a.tensor != nullptr ? a.tensor->dims() : nullptr;
    return (dims != nullptr && dims->size() > 0) ? dims->Get(dims->size() - 1) : -1;
}

// hipdnn.same_dtype(t0, t1): the two tensors share a dtype.
bool sameDtype(const std::vector<BoundArg>& args) {
    return args[0].tensor != nullptr && args[1].tensor != nullptr &&
           args[0].tensor->data_type() == args[1].tensor->data_type();
}

// hipdnn.same_head_dim(t0, t1, t2): all three share their last dim (SDPA head).
bool sameHeadDim(const std::vector<BoundArg>& args) {
    const int64_t d = lastDim(args[0]);
    return d >= 0 && lastDim(args[1]) == d && lastDim(args[2]) == d;
}

// hipdnn.divisible_by(value, divisor): value % divisor == 0 (divisor != 0).
bool divisibleBy(const std::vector<BoundArg>& args) {
    return args[1].value != 0 && (args[0].value % args[1].value) == 0;
}

}  // namespace

const PredicateRegistry& PredicateRegistry::builtin() {
    static const PredicateRegistry registry = [] {
        PredicateRegistry r;
        r._entries = {
            {"hipdnn.same_dtype", {ArgKind::Tensor, ArgKind::Tensor}, &sameDtype, true},
            {"hipdnn.same_head_dim",
             {ArgKind::Tensor, ArgKind::Tensor, ArgKind::Tensor},
             &sameHeadDim,
             true},
            {"hipdnn.divisible_by", {ArgKind::Int, ArgKind::Int}, &divisibleBy, true},
        };
        return r;
    }();
    return registry;
}

const PredicateEntry* PredicateRegistry::find(std::string_view name) const noexcept {
    for (const auto& entry : _entries) {
        if (entry.name == name) {
            return &entry;
        }
    }
    return nullptr;
}

void PredicateRegistry::registerPredicate(PredicateEntry entry) {
    entry.builtin = false;
    for (auto& existing : _entries) {
        if (existing.name == entry.name) {
            existing = std::move(entry);
            return;
        }
    }
    _entries.push_back(std::move(entry));
}

}  // namespace hipdnn::graph_matcher

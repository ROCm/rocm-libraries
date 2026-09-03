// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// OperatorTable.hpp - the operator table, and the node that runs it.
//
// OP_TABLE is the single place an operator is defined: key, arity bounds, and
// implementation. Adding one is a row here plus a function in Operators.hpp.

#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Error.hpp>
#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Node.hpp>
#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Operators.hpp>
#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Value.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace hipdnn_plugin_sdk::ingestor::jsonexpr::detail
{
// ---- operator table -------------------------------------------------------
// A row that names no implementation fails the static_assert below. Without
// that check the operator would evaluate to null, which the language reads as
// "unresolved", so a half-wired operator would silently make predicates
// decline.

/// `maxArity` for an operator that accepts any number of arguments. This is
/// the largest representable count rather than a small sentinel, so that it
/// does not impose an argument limit the language never had.
inline constexpr std::size_t VARIADIC = std::numeric_limits<std::size_t>::max();

struct OpSpec
{
    std::string_view key;
    std::size_t minArity;
    std::size_t maxArity;
    ops::EagerFn eager; ///< exactly one of `eager` / `lazy` is set
    ops::LazyFn lazy;
};

/// Builders that keep the table rows readable. An operator is eager unless it
/// needs to control its own argument evaluation.
constexpr OpSpec
    eagerOp(std::string_view key, std::size_t minArity, std::size_t maxArity, ops::EagerFn fn)
{
    return OpSpec{key, minArity, maxArity, fn, nullptr};
}
constexpr OpSpec
    lazyOp(std::string_view key, std::size_t minArity, std::size_t maxArity, ops::LazyFn fn)
{
    return OpSpec{key, minArity, maxArity, nullptr, fn};
}

inline constexpr std::array<OpSpec, 29> OP_TABLE
    = {{eagerOp("+", 0, VARIADIC, &ops::add),
        eagerOp("-", 1, 2, &ops::subtract),
        eagerOp("*", 0, VARIADIC, &ops::multiply),
        eagerOp("/", 2, 2, &ops::quotient),
        eagerOp("%", 2, 2, &ops::remainder),
        eagerOp("min", 1, VARIADIC, &ops::minimum),
        eagerOp("max", 1, VARIADIC, &ops::maximum),
        eagerOp("<", 2, 3, &ops::lessThan),
        eagerOp("<=", 2, 3, &ops::lessOrEqual),
        eagerOp(">", 2, 2, &ops::greaterThan),
        eagerOp(">=", 2, 2, &ops::greaterOrEqual),
        eagerOp("==", 2, 2, &ops::equal),
        eagerOp("!=", 2, 2, &ops::notEqual),
        eagerOp("!", 1, 1, &ops::logicalNot),
        eagerOp("!!", 1, 1, &ops::toBoolean),
        lazyOp("if", 2, VARIADIC, &ops::conditional),
        lazyOp("?:", 2, VARIADIC, &ops::conditional),
        lazyOp("and", 1, VARIADIC, &ops::conjunction),
        lazyOp("or", 1, VARIADIC, &ops::disjunction),
        eagerOp("in", 2, 2, &ops::membership),
        eagerOp("ceil_div", 2, 2, &ops::ceilDiv),
        eagerOp("divisible", 2, 2, &ops::divisible),
        eagerOp("abs", 1, 1, &ops::absoluteValue),
        eagerOp("pow", 2, 2, &ops::power),
        eagerOp("log2", 1, 1, &ops::log2Of),
        eagerOp("rsqrt", 1, 1, &ops::reciprocalSqrt),
        lazyOp("value_or_default", 2, 2, &ops::valueOrDefault),
        lazyOp("present", 1, VARIADIC, &ops::present),
        lazyOp("not_present", 1, VARIADIC, &ops::notPresent)}};

constexpr bool opTableIsWellFormed()
{
    for(const OpSpec& s : OP_TABLE)
    {
        // Exactly one handler. With neither the operator would evaluate to
        // null at run time; with both the eager/lazy choice is ambiguous.
        if((s.eager == nullptr) == (s.lazy == nullptr))
        {
            return false;
        }
        if(s.key.empty() || s.minArity > s.maxArity)
        {
            return false;
        }
    }
    return true;
}
static_assert(opTableIsWellFormed(),
              "every OP_TABLE row needs a non-empty key, a sane arity range, and exactly one of "
              "an eager or a lazy handler");

struct OpNode final : Node
{
    const OpSpec* spec = nullptr;
    std::vector<NodePtr> args;

    Value eval(const IDataSource& d) const override
    {
        if(spec->lazy != nullptr)
        {
            return spec->lazy(args, d);
        }
        // Evaluate every argument exactly once, then check the results before
        // the operator can coerce them. An array counts as unresolved if any
        // element is, so a predicate never answers from a partly resolved
        // array literal.
        std::vector<Value> v;
        v.reserve(args.size());
        for(const auto& c : args)
        {
            v.push_back(c->eval(d));
        }
        for(const Value& x : v)
        {
            if(x.containsUnresolved())
            {
                return {};
            }
        }
        return spec->eager(v);
    }

    void pushChildren(std::vector<const Node*>& stack) const override
    {
        for(auto it = args.rbegin(); it != args.rend(); ++it)
        {
            stack.push_back(it->get());
        }
    }
};

/// The spec for an operator key, or nullptr if the key names no operator.
inline const OpSpec* lookupOp(const std::string& key)
{
    for(const OpSpec& e : OP_TABLE)
    {
        if(key == e.key)
        {
            return &e;
        }
    }
    return nullptr;
}

inline void checkArity(const OpSpec& spec, std::size_t n, const std::string& key)
{
    if(n < spec.minArity || n > spec.maxArity)
    {
        throw JsonExpressionCompileError("operator '" + key
                                         + "' got wrong argument count: " + std::to_string(n));
    }
}
} // namespace hipdnn_plugin_sdk::ingestor::jsonexpr::detail

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

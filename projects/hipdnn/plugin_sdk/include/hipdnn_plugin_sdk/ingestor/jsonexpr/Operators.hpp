// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// Operators.hpp - one function per operator in the language.
//
// Each is named in exactly one OP_TABLE row (OperatorTable.hpp); nothing else
// dispatches on operator identity.

#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Node.hpp>
#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Value.hpp>

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace hipdnn_plugin_sdk::ingestor::jsonexpr::detail
{
// ---- operator implementations ---------------------------------------------
// Null is "unresolved", not a value: an absent optional field's read must
// neither pass nor fail a predicate. Every *eager* operator below therefore
// never sees a null at all -- OpNode::eval declines first (null would
// otherwise read as 0 / false / not-equal, so a narrowing check on an absent
// operand would silently PASS). The *lazy* operators are the deliberate
// exceptions: `present`, `not_present` and `value_or_default` answer "did this
// resolve?", and `and` / `or` are three-valued, so each controls its own
// argument evaluation and may return a real value beside an unresolved
// argument. A null root is rejected, because Value::truthy() reads null as
// false.
namespace ops
{
/// An eager operator: every argument is already evaluated and none is null.
using EagerFn = Value (*)(const std::vector<Value>&);

/// A lazy operator: evaluates its own arguments, so it decides what a null
/// argument means and which arguments run at all.
using LazyFn = Value (*)(const std::vector<NodePtr>&, const IDataSource&);

/// Every eager numeric operator funnels its result through here.
///
/// A NaN or infinite result compares UNORDERED, so every ordering test on it is
/// false and its NEGATION is true -- a criterion would ACCEPT input it never
/// meaningfully evaluated. Declining instead keeps an undecidable computation
/// unresolved, which is what null already means here.
///
/// This has to be the single exit for all of them, not a guard per operator:
/// NaN arrives as an *operand* too (Value::toNumber yields NaN for a
/// non-numeric string and for a multi-element array), so a domain check
/// written `n <= 0.0` silently passes it through -- that comparison is false
/// for NaN.
inline Value finiteOrNull(double d)
{
    if(!std::isfinite(d))
    {
        return {};
    }
    return Value::number(d);
}

inline Value add(const std::vector<Value>& v)
{
    double a = 0.0;
    for(const Value& x : v)
    {
        a += x.toNumber();
    }
    return finiteOrNull(a);
}

inline Value multiply(const std::vector<Value>& v)
{
    double a = 1.0;
    for(const Value& x : v)
    {
        a *= x.toNumber();
    }
    return finiteOrNull(a);
}

/// Unary negation on one argument, subtraction on two.
inline Value subtract(const std::vector<Value>& v)
{
    if(v.size() == 1)
    {
        return finiteOrNull(-v[0].toNumber());
    }
    return finiteOrNull(v[0].toNumber() - v[1].toNumber());
}

/// What a division operator makes of a numerator/denominator pair whose
/// divisor is already known to be non-zero.
using DivisionResult = Value (*)(double, double);

/// The four operators built on one division share the zero-divisor rule, so
/// they must decline together; only what they do afterwards differs.
inline Value divide(const std::vector<Value>& v, DivisionResult combine)
{
    const double num = v[0].toNumber();
    const double den = v[1].toNumber();
    if(den == 0.0)
    {
        return {}; // zero divisor declines rather than yielding inf/NaN
    }
    return combine(num, den);
}

inline Value quotient(const std::vector<Value>& v)
{
    return divide(v, [](double num, double den) { return finiteOrNull(num / den); });
}
inline Value remainder(const std::vector<Value>& v)
{
    return divide(v, [](double num, double den) { return finiteOrNull(std::fmod(num, den)); });
}
inline Value ceilDiv(const std::vector<Value>& v)
{
    return divide(v, [](double num, double den) { return finiteOrNull(std::ceil(num / den)); });
}
inline Value divisible(const std::vector<Value>& v)
{
    // Exactly {"==": [{"%": [a, b]}, 0]}, the longhand the RFCs give for the
    // same check, so the short-hand and the spelled-out form agree on every
    // input -- including declining on a zero divisor, and on the NaN operand
    // that would otherwise make `fmod(...) == 0.0` a plain false.
    return divide(v, [](double num, double den) {
        const double r = std::fmod(num, den);
        if(!std::isfinite(r))
        {
            return Value();
        }
        return Value(r == 0.0);
    });
}

/// Declines unless every argument is finite.
///
/// A NaN sentinel cannot serve as "nothing chosen yet" here: a NaN *argument*
/// is then indistinguishable from the seed and is simply overwritten, so the
/// operator would answer from FEWER operands than were authored, with nothing
/// to signal it. An explicit flag separates the two, and a non-finite argument
/// declines outright rather than being skipped.
inline Value extremum(const std::vector<Value>& v, bool wantMax)
{
    double best = 0.0;
    bool haveBest = false;
    for(const Value& x : v)
    {
        const double n = x.toNumber();
        if(!std::isfinite(n))
        {
            return {};
        }
        if(!haveBest || (wantMax ? n > best : n < best))
        {
            best = n;
            haveBest = true;
        }
    }
    if(!haveBest)
    {
        return {};
    }
    return Value::number(best);
}

inline Value minimum(const std::vector<Value>& v)
{
    return extremum(v, false);
}
inline Value maximum(const std::vector<Value>& v)
{
    return extremum(v, true);
}

/// Which `Value::compare` outcomes the operator accepts. Passing the accepted
/// set, rather than an operator tag plus a switch, keeps each comparison
/// operator a single expression and leaves no unreachable branch.
using OrderingAccepts = bool (*)(Value::Ordering);

constexpr bool acceptsLess(Value::Ordering c)
{
    return c == Value::Ordering::LESS;
}
constexpr bool acceptsLessOrEqual(Value::Ordering c)
{
    return c == Value::Ordering::LESS || c == Value::Ordering::EQUAL;
}
constexpr bool acceptsGreater(Value::Ordering c)
{
    return c == Value::Ordering::GREATER;
}
constexpr bool acceptsGreaterOrEqual(Value::Ordering c)
{
    return c == Value::Ordering::GREATER || c == Value::Ordering::EQUAL;
}

/// A NaN operand compares UNORDERED, which every predicate above rejects, so
/// an ordering test against NaN is false rather than throwing.
inline Value compareValues(const std::vector<Value>& v, OrderingAccepts accepts)
{
    // The 3-arg form is the between-chain: a < b < c.
    if(v.size() >= 3)
    {
        return {accepts(Value::compare(v[0], v[1])) && accepts(Value::compare(v[1], v[2]))};
    }
    return {accepts(Value::compare(v[0], v[1]))};
}

inline Value lessThan(const std::vector<Value>& v)
{
    return compareValues(v, &acceptsLess);
}
inline Value lessOrEqual(const std::vector<Value>& v)
{
    return compareValues(v, &acceptsLessOrEqual);
}
inline Value greaterThan(const std::vector<Value>& v)
{
    return compareValues(v, &acceptsGreater);
}
inline Value greaterOrEqual(const std::vector<Value>& v)
{
    return compareValues(v, &acceptsGreaterOrEqual);
}

// Two unresolved references are not "equal"; the question is unanswerable, so
// OpNode::eval declines before either of these runs.
inline Value equal(const std::vector<Value>& v)
{
    return {v[0] == v[1]};
}
inline Value notEqual(const std::vector<Value>& v)
{
    return {v[0] != v[1]};
}

inline Value logicalNot(const std::vector<Value>& v)
{
    return {!v[0].truthy()};
}
inline Value toBoolean(const std::vector<Value>& v)
{
    return {v[0].truthy()};
}

/// Element containment in an array, substring containment in a string. A
/// haystack of any other kind contains nothing.
inline Value membership(const std::vector<Value>& v)
{
    const Value& needle = v[0];
    const Value& hay = v[1];
    if(hay.isArray())
    {
        for(const auto& e : hay.asArray())
        {
            if(e == needle)
            {
                return {true};
            }
        }
        return {false};
    }
    if(hay.isString())
    {
        const std::string n = needle.isString() ? needle.asString() : needle.dump();
        return {hay.asString().find(n) != std::string::npos};
    }
    return {false};
}

inline Value absoluteValue(const std::vector<Value>& v)
{
    return finiteOrNull(std::fabs(v[0].toNumber()));
}

inline Value power(const std::vector<Value>& v)
{
    // A domain error (a negative base under a fractional exponent) or an
    // overflow yields NaN/inf; finiteOrNull declines on both.
    return finiteOrNull(std::pow(v[0].toNumber(), v[1].toNumber()));
}

inline Value log2Of(const std::vector<Value>& v)
{
    const double n = v[0].toNumber();
    // `!(n > 0.0)`, not `n <= 0.0`: the latter is FALSE for NaN, so a NaN
    // operand would slip past the domain check and log2 would return one.
    if(!(n > 0.0))
    {
        return {}; // log2 declines on a non-positive or unresolvable argument
    }
    return finiteOrNull(std::log2(n));
}

inline Value reciprocalSqrt(const std::vector<Value>& v)
{
    const double n = v[0].toNumber();
    // Negated form for the same reason log2Of uses it: NaN fails `n > 0.0`.
    if(!(n > 0.0))
    {
        return {}; // rsqrt declines on a non-positive or unresolvable argument
    }
    return finiteOrNull(1.0 / std::sqrt(n));
}

/// Condition/result pairs, with an optional trailing else.
inline Value conditional(const std::vector<NodePtr>& args, const IDataSource& d)
{
    std::size_t i = 0;
    for(; i + 1 < args.size(); i += 2)
    {
        const Value cond = args[i]->eval(d);
        if(cond.isNull())
        {
            return {}; // an unresolved condition picks no branch
        }
        if(cond.truthy())
        {
            return args[i + 1]->eval(d);
        }
    }
    return i < args.size() ? args[i]->eval(d) : Value();
}

/// Kleene `and`: a definite false short-circuits even when another argument is
/// unresolved, so `and`-ing an inapplicable check beside a failing one still
/// declines. Otherwise a null makes the whole conjunction unresolved.
inline Value conjunction(const std::vector<NodePtr>& args, const IDataSource& d)
{
    Value cur(true);
    bool sawNull = false;
    for(const auto& c : args)
    {
        cur = c->eval(d);
        if(cur.isNull())
        {
            sawNull = true;
            continue;
        }
        if(!cur.truthy())
        {
            return cur; // definite false
        }
    }
    return sawNull ? Value() : cur;
}

/// Kleene `or`: a definite true short-circuits past an unresolved argument,
/// which is what lets
/// `{"or": [{"not_present": ["$bias"]}, {"==": ["$bias.dtype", ...]}]}`
/// accept input with no `bias` at all, even though the second arm cannot run.
inline Value disjunction(const std::vector<NodePtr>& args, const IDataSource& d)
{
    Value cur;
    bool sawNull = false;
    for(const auto& c : args)
    {
        cur = c->eval(d);
        if(cur.isNull())
        {
            sawNull = true;
            continue;
        }
        if(cur.truthy())
        {
            return cur; // definite true
        }
    }
    return sawNull ? Value() : cur;
}

/// First arg is a variable reference; a null result means the path did not
/// resolve in the data source, so fall back to the default.
inline Value valueOrDefault(const std::vector<NodePtr>& args, const IDataSource& d)
{
    const Value v = args[0]->eval(d);
    return v.isNull() ? args[1]->eval(d) : v;
}

/// Presence keys on *existence*, the same mechanism as valueOrDefault above:
/// an unresolved path reads null. Unlike every other operator these do not
/// propagate that null -- asking "was this supplied?" always yields a real
/// boolean. Both fold with `and` over their arguments, so one call decides a
/// whole list.
inline Value presence(const std::vector<NodePtr>& args, const IDataSource& d, bool wantNull)
{
    for(const auto& c : args)
    {
        if(c->eval(d).isNull() != wantNull)
        {
            return {false};
        }
    }
    return {true};
}

inline Value present(const std::vector<NodePtr>& args, const IDataSource& d)
{
    return presence(args, d, false);
}
inline Value notPresent(const std::vector<NodePtr>& args, const IDataSource& d)
{
    return presence(args, d, true);
}
} // namespace ops
} // namespace hipdnn_plugin_sdk::ingestor::jsonexpr::detail

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

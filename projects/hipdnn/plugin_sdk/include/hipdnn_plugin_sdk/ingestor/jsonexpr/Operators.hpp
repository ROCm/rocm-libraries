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
// Null means "unresolved": reading an absent optional field must neither pass
// nor fail a predicate. OpNode::eval therefore declines before calling any
// eager operator, so none of them ever sees a null. Without that, a null would
// read as 0, false, or not-equal, and a narrowing check on an absent operand
// would pass.
//
// The lazy operators are the deliberate exceptions. `present`, `not_present`
// and `value_or_default` exist to answer "did this resolve?", and `and` / `or`
// are three-valued. Each evaluates its own arguments and may return a real
// value even when an argument is unresolved.
//
// A caller that treats an expression's result as a boolean rejects an
// unresolved rule, since Value::truthy() reads null as false.
namespace ops
{
/// An eager operator: every argument is already evaluated and none is null.
using EagerFn = Value (*)(const std::vector<Value>&);

/// A lazy operator: evaluates its own arguments, so it decides what a null
/// argument means and which arguments run at all.
using LazyFn = Value (*)(const std::vector<NodePtr>&, const IDataSource&);

/// Every eager numeric operator returns its result through here. A NaN or
/// infinite result cannot be ordered, and a criterion must not accept data it
/// never meaningfully evaluated, so a non-finite result becomes null
/// ("unresolved") instead.
///
/// This must be the single exit for all of them rather than a guard in each
/// operator, because NaN can also arrive as an operand: Value::toNumber yields
/// NaN for a non-numeric string and for a multi-element array. A domain check
/// written `n <= 0.0` is false for NaN and would let it through.
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

/// What a division operator does with a numerator and a denominator that is
/// already known to be non-zero.
using DivisionResult = Value (*)(double, double);

/// Shared by the four division-based operators. They all decline on a zero
/// divisor; only what they do afterwards differs.
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
    // Equivalent to {"==": [{"%": [a, b]}, 0]}, the longhand form the RFCs
    // also use. Both spellings must agree on every input, including declining
    // on a zero divisor and on a NaN operand, which would otherwise make
    // `fmod(...) == 0.0` a plain false.
    return divide(v, [](double num, double den) {
        const double r = std::fmod(num, den);
        if(!std::isfinite(r))
        {
            return Value();
        }
        return Value(r == 0.0);
    });
}

/// Smallest or largest argument. Declines unless every argument is finite.
///
/// `haveBest` tracks "nothing chosen yet" instead of seeding `best` with NaN.
/// A NaN seed would be indistinguishable from a NaN argument and simply be
/// overwritten, so the operator would answer from fewer operands than were
/// written, with nothing to signal it.
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

/// Which `Value::compare` outcomes an operator accepts. Passing the accepted
/// set, rather than an operator tag plus a switch, keeps each comparison
/// operator a single expression with no unreachable branch.
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

/// A non-finite operand compares UNORDERED, which makes the result null rather
/// than false. A surrounding `!` must not turn "could not compare" into a
/// pass.
inline Value comparePairResult(Value::Ordering c, OrderingAccepts accepts)
{
    return c == Value::Ordering::UNORDERED ? Value() : Value(accepts(c));
}

inline Value compareValues(const std::vector<Value>& v, OrderingAccepts accepts)
{
    // The 3-argument form is the chained comparison a < b < c. Both links are
    // checked before answering, because an unordered link makes the whole
    // chain unanswerable even if the other link is already false.
    if(v.size() >= 3)
    {
        const Value::Ordering first = Value::compare(v[0], v[1]);
        const Value::Ordering second = Value::compare(v[1], v[2]);
        if(first == Value::Ordering::UNORDERED || second == Value::Ordering::UNORDERED)
        {
            return {};
        }
        return {accepts(first) && accepts(second)};
    }
    return comparePairResult(Value::compare(v[0], v[1]), accepts);
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

// OpNode::eval declines before either of these runs, so two unresolved
// references never compare equal here; the question is simply unanswerable.
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

/// Element containment in an array, substring containment in a string.
/// Anything else as the haystack contains nothing.
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
    // A domain error (a negative base with a fractional exponent) or an
    // overflow gives NaN or infinity, and finiteOrNull declines on both.
    return finiteOrNull(std::pow(v[0].toNumber(), v[1].toNumber()));
}

inline Value log2Of(const std::vector<Value>& v)
{
    const double n = v[0].toNumber();
    // Written `!(n > 0.0)` rather than `n <= 0.0`, because the latter is false
    // for NaN and would let a NaN operand reach log2.
    if(!(n > 0.0))
    {
        return {}; // non-positive or unresolvable argument
    }
    return finiteOrNull(std::log2(n));
}

inline Value reciprocalSqrt(const std::vector<Value>& v)
{
    const double n = v[0].toNumber();
    // Negated form for the same reason as log2Of: NaN fails `n > 0.0`.
    if(!(n > 0.0))
    {
        return {}; // non-positive or unresolvable argument
    }
    return finiteOrNull(1.0 / std::sqrt(n));
}

/// `if` / `?:`: condition and result pairs, with an optional trailing else.
inline Value conditional(const std::vector<NodePtr>& args, const IDataSource& d)
{
    std::size_t i = 0;
    for(; i + 1 < args.size(); i += 2)
    {
        const Value cond = args[i]->eval(d);
        if(cond.containsUnresolved())
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

/// Three-valued `and`. A definite false wins even when another argument is
/// unresolved, so combining an inapplicable check with a failing one still
/// rejects. Otherwise an unresolved argument makes the whole result
/// unresolved.
inline Value conjunction(const std::vector<NodePtr>& args, const IDataSource& d)
{
    Value cur(true);
    bool sawNull = false;
    for(const auto& c : args)
    {
        cur = c->eval(d);
        if(cur.containsUnresolved())
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

/// Three-valued `or`. A definite true wins even when another argument is
/// unresolved. That is what lets
/// `{"or": [{"not_present": ["$bias"]}, {"==": ["$bias.dtype", ...]}]}`
/// accept input with no `bias`, where the second arm cannot run.
inline Value disjunction(const std::vector<NodePtr>& args, const IDataSource& d)
{
    Value cur;
    bool sawNull = false;
    for(const auto& c : args)
    {
        cur = c->eval(d);
        if(cur.containsUnresolved())
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

/// Returns the first argument, or the second when the first did not fully
/// resolve. An array with an unresolved element counts as not resolved:
/// handing back a value with a hole in it would defeat the fallback.
inline Value valueOrDefault(const std::vector<NodePtr>& args, const IDataSource& d)
{
    const Value v = args[0]->eval(d);
    return v.containsUnresolved() ? args[1]->eval(d) : v;
}

/// `present` / `not_present` report whether a path resolved, using the same
/// null marker as valueOrDefault. Unlike every other operator they always
/// return a real boolean instead of propagating null. Both combine their
/// arguments with `and`, so one call can check a whole list.
///
/// The two take opposite predicates rather than one negated flag, because both
/// must answer false for a value that only partly resolves. An array with an
/// unresolved element is neither wholly supplied nor wholly absent. Negating a
/// single flag would make `not_present` true in that case, and the documented
/// `{"or": [{"not_present": ["$x"]}, {"and": [{"present": ["$x"]}, ...]}]}`
/// guard would then accept input whose field reads never ran.
using PresencePredicate = bool (*)(const Value&);

inline bool isWhollySupplied(const Value& v)
{
    return !v.containsUnresolved();
}
inline bool isWhollyAbsent(const Value& v)
{
    return v.isNull();
}

inline Value
    presence(const std::vector<NodePtr>& args, const IDataSource& d, PresencePredicate holds)
{
    for(const auto& c : args)
    {
        if(!holds(c->eval(d)))
        {
            return {false};
        }
    }
    return {true};
}

inline Value present(const std::vector<NodePtr>& args, const IDataSource& d)
{
    return presence(args, d, &isWhollySupplied);
}
inline Value notPresent(const std::vector<NodePtr>& args, const IDataSource& d)
{
    return presence(args, d, &isWhollyAbsent);
}
} // namespace ops
} // namespace hipdnn_plugin_sdk::ingestor::jsonexpr::detail

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

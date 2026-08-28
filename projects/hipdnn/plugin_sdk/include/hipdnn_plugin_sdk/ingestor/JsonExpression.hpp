// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// JsonExpression.hpp - single-file compiler for the JSON Expression Language.
//
// All names below live in namespace hipdnn_plugin_sdk::ingestor::jsonexpr;
// these examples assume `namespace jexpr = hipdnn_plugin_sdk::ingestor::jsonexpr;`.
//
// An expression (an nlohmann::json value) is *compiled once* into a
// reusable jexpr::Expression<Data>, then evaluated many times against different
// data sources:
//
//     struct MyData {                       // your data source
//         jexpr::Value getData(const std::string& path) const;
//     };
//
//     auto expr = jexpr::compile<MyData>(rule);   // parse + build tree once
//     jexpr::Value r1 = expr(dataA);              // evaluate, no re-parse
//     jexpr::Value r2 = expr(dataB);              // reuse for other data
//
// The runtime value type (jexpr::Value) is a small standalone variant that does
// NOT depend on nlohmann/json; nlohmann is used only to express the rule being
// compiled. The data source is any C++ type exposing
//   Value getData(const std::string&)
// which resolves a variable path to a Value; a null return means "not found".
// Variable paths are always non-empty (whole-document references are not
// supported).
//
// Variables
// ---------
// A string prefixed with a sigil (default '$') is a variable reference, and is
// the only way to read data; there is no `var` operator:
//     {"+": ["$x", "$y"]}      "$a.b"  nested path      "$$x"  literal "$x"
// Strings without the sigil remain literals. `value_or_default` supplies a
// fallback for a path that does not resolve.
//
// Scope: core operator set (data access, logic, comparison, arithmetic), the
// membership operator `in`, the `divisible` and `value_or_default` short-hands,
// the `present` / `not_present` resolution predicates, and a small set of
// value-core math extensions (ceil_div, abs, pow, log2, rsqrt).
// Collection/string operators (map/reduce/filter/cat/substr/...) are not
// included.
//
// Null is "unresolved", not a value
// ---------------------------------
// Every operator except `present`, `not_present`, and `value_or_default`
// propagates a null argument instead of coercing it, and `and`/`or` are
// three-valued: a definite false still decides an `and`, a definite true a
// `or`. This suits a data source with optional fields, where an unresolved
// path means the field is absent rather than false: a coerced null would read
// as false/0/not-equal and make a narrowing predicate silently PASS on data it
// never actually saw. A null root is falsy, so an undecided expression is
// rejected rather than accepted.
//
// Full reference: docs/JsonExpression.md.

#include <nlohmann/json.hpp>

#include <hipdnn_data_sdk/utilities/Visitor.hpp>

#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace hipdnn_plugin_sdk::ingestor::jsonexpr
{
/// Thrown when a rule cannot be compiled (unknown operator, bad arity, ...).
class JsonExpressionCompileError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// ===========================================================================
// Value - standalone runtime value (json-like, no nlohmann dependency)
// ===========================================================================
class Value
{
public:
    using Array = std::vector<Value>;

    Value()
        : _v(nullptr)
    {
    }
    Value(std::nullptr_t)
        : _v(nullptr)
    {
    }
    Value(bool b)
        : _v(b)
    {
    }
    Value(int i)
        : _v(static_cast<std::int64_t>(i))
    {
    }
    Value(std::int64_t i)
        : _v(i)
    {
    }
    Value(double d)
        : _v(d)
    {
    }
    Value(const char* s)
        : _v(std::string(s))
    {
    }
    Value(std::string s)
        : _v(std::move(s))
    {
    }
    Value(Array a)
        : _v(std::move(a))
    {
    }

    /// Build a numeric value, storing an integer when the double is exactly
    /// integral and representable, so integer inputs yield integer output.
    static Value number(double d)
    {
        if(std::isfinite(d))
        {
            const double t = std::trunc(d);
            if(t == d && d >= -9.007199254740992e15 && d <= 9.007199254740992e15)
            {
                return {static_cast<std::int64_t>(t)};
            }
        }
        return {d};
    }

    bool isNull() const
    {
        return std::holds_alternative<std::nullptr_t>(_v);
    }
    bool isBool() const
    {
        return std::holds_alternative<bool>(_v);
    }
    bool isInt() const
    {
        return std::holds_alternative<std::int64_t>(_v);
    }
    bool isDouble() const
    {
        return std::holds_alternative<double>(_v);
    }
    bool isNumber() const
    {
        return isInt() || isDouble();
    }
    bool isString() const
    {
        return std::holds_alternative<std::string>(_v);
    }
    bool isArray() const
    {
        return std::holds_alternative<Array>(_v);
    }

    bool asBool() const
    {
        return std::get<bool>(_v);
    }
    std::int64_t asInt() const
    {
        return std::get<std::int64_t>(_v);
    }
    double asDouble() const
    {
        return std::get<double>(_v);
    }
    const std::string& asString() const
    {
        return std::get<std::string>(_v);
    }
    const Array& asArray() const
    {
        return std::get<Array>(_v);
    }

    /// Truthiness: false, 0, "", null and the empty array are falsy.
    bool truthy() const
    {
        return std::visit(
            hipdnn_data_sdk::utilities::Visitor{[](std::nullptr_t) { return false; },
                                                [](bool b) { return b; },
                                                [](std::int64_t i) { return i != 0; },
                                                [](double d) { return d != 0.0; },
                                                [](const std::string& s) { return !s.empty(); },
                                                [](const Array& a) { return !a.empty(); }},
            _v);
    }

    /// JS Number() coercion. Non-numeric strings and multi-element arrays yield
    /// NaN; empty string / empty array / null yield 0.
    double toNumber() const
    {
        return std::visit(hipdnn_data_sdk::utilities::Visitor{
                              [](std::nullptr_t) { return 0.0; },
                              [](bool b) { return b ? 1.0 : 0.0; },
                              [](std::int64_t i) { return static_cast<double>(i); },
                              [](double d) { return d; },
                              [](const std::string& s) { return stringToNumber(s); },
                              [](const Array& a) {
                                  if(a.empty())
                                  {
                                      return 0.0;
                                  }
                                  if(a.size() == 1)
                                  {
                                      return a.front().toNumber();
                                  }
                                  return std::nan("");
                              }},
                          _v);
    }

    /// Structural equality (== / !=). Integers and doubles of equal value
    /// compare equal; differing kinds do not.
    bool operator==(const Value& o) const
    {
        if(isNumber() && o.isNumber())
        {
            return toNumber() == o.toNumber();
        }
        // Otherwise this is exactly variant equality: differing alternatives are
        // unequal, and each alternative compares with its own operator==.
        return _v == o._v;
    }
    bool operator!=(const Value& o) const
    {
        return !(*this == o);
    }

    /// Ordering result of `compare`. UNORDERED is the NaN case, and makes every
    /// ordering test false.
    enum class Ordering
    {
        LESS,
        EQUAL,
        GREATER,
        UNORDERED
    };

    /// Three-way compare for ordering. Two strings compare lexically; anything
    /// else is compared as a number, and a NaN operand yields UNORDERED.
    static Ordering compare(const Value& a, const Value& b)
    {
        if(a.isString() && b.isString())
        {
            const auto& x = a.asString();
            const auto& y = b.asString();
            if(x < y)
            {
                return Ordering::LESS;
            }
            return x > y ? Ordering::GREATER : Ordering::EQUAL;
        }
        const double x = a.toNumber();
        const double y = b.toNumber();
        if(std::isnan(x) || std::isnan(y))
        {
            return Ordering::UNORDERED;
        }
        if(x < y)
        {
            return Ordering::LESS;
        }
        return x > y ? Ordering::GREATER : Ordering::EQUAL;
    }

    /// Human-readable rendering, mainly for diagnostics and tests.
    std::string dump() const
    {
        return std::visit(hipdnn_data_sdk::utilities::Visitor{
                              [](std::nullptr_t) { return std::string("null"); },
                              [](bool b) { return std::string(b ? "true" : "false"); },
                              [](std::int64_t i) { return std::to_string(i); },
                              [](double d) { return std::to_string(d); },
                              [](const std::string& s) { return "\"" + s + "\""; },
                              [](const Array& a) {
                                  std::string s = "[";
                                  for(std::size_t i = 0; i < a.size(); ++i)
                                  {
                                      s += ((i != 0u) ? "," : "") + a[i].dump();
                                  }
                                  return s + "]";
                              }},
                          _v);
    }

    /// Stream rendering (used by GoogleTest value printing and diagnostics).
    friend std::ostream& operator<<(std::ostream& os, const Value& v)
    {
        return os << v.dump();
    }

private:
    static double stringToNumber(const std::string& s)
    {
        std::size_t b = 0;
        std::size_t e = s.size();
        while(b < e && (std::isspace(static_cast<unsigned char>(s[b])) != 0))
        {
            ++b;
        }
        while(e > b && (std::isspace(static_cast<unsigned char>(s[e - 1])) != 0))
        {
            --e;
        }
        if(b == e)
        {
            return 0.0; // JS Number("") == 0
        }
        const std::string t = s.substr(b, e - b);
        const char* first = t.c_str();
        char* last = nullptr;
        const double d = std::strtod(first, &last);
        if(last != first + t.size())
        {
            return std::nan(""); // trailing garbage -> NaN
        }
        return d;
    }

    // The alternatives are reached by type through std::visit, so their order
    // here carries no meaning.
    std::variant<std::nullptr_t, bool, std::int64_t, double, std::string, Array> _v;
};

namespace detail
{
// ---- data-source capability detection ------------------------------------
template <class T, class = void>
struct HasGetData : std::false_type
{
};
template <class T>
struct HasGetData<
    T,
    std::void_t<decltype(std::declval<const T&>().getData(std::declval<std::string>()))>>
    : std::true_type
{
};

// ---- type-erased data source ---------------------------------------------
// The compiled node tree evaluates against this abstract source rather than a
// concrete DataT, so the tree itself carries no template parameter (and thus
// no per-DataT virtual member instantiation). Expression<DataT> wraps the
// caller's data object in a DataSourceAdapter at evaluation time.
struct IDataSource
{
    virtual ~IDataSource() = default;
    virtual Value getData(const std::string& path) const = 0;
};

template <class DataT>
struct DataSourceAdapter final : IDataSource
{
    static_assert(HasGetData<DataT>::value, "Data source must provide Value getData(std::string).");
    const DataT& data;
    explicit DataSourceAdapter(const DataT& d)
        : data(d)
    {
    }
    Value getData(const std::string& path) const override
    {
        return data.getData(path);
    }
};

// ---- compiled node tree ---------------------------------------------------

struct Node
{
    virtual ~Node() = default;
    virtual Value eval(const IDataSource& data) const = 0;
    virtual const std::string* variable() const
    {
        return nullptr;
    }
    virtual void pushChildren(std::vector<const Node*>& /*unused*/) const {}
};

using NodePtr = std::unique_ptr<Node>;

struct LiteralNode final : Node
{
    Value value;
    explicit LiteralNode(Value v)
        : value(std::move(v))
    {
    }
    Value eval(const IDataSource& /*unused*/) const override
    {
        return value;
    }
};

struct ArrayNode final : Node
{
    std::vector<NodePtr> items;
    Value eval(const IDataSource& data) const override
    {
        Value::Array a;
        a.reserve(items.size());
        for(const auto& it : items)
        {
            a.push_back(it->eval(data));
        }
        return {std::move(a)};
    }
    void pushChildren(std::vector<const Node*>& stack) const override
    {
        for(auto it = items.rbegin(); it != items.rend(); ++it)
        {
            stack.push_back(it->get());
        }
    }
};

struct VarNode final : Node
{
    std::string path;

    Value eval(const IDataSource& data) const override
    {
        return data.getData(path);
    }

    const std::string* variable() const override
    {
        return &path;
    }
};

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

inline Value add(const std::vector<Value>& v)
{
    double a = 0.0;
    for(const Value& x : v)
    {
        a += x.toNumber();
    }
    return Value::number(a);
}

inline Value multiply(const std::vector<Value>& v)
{
    double a = 1.0;
    for(const Value& x : v)
    {
        a *= x.toNumber();
    }
    return Value::number(a);
}

/// Unary negation on one argument, subtraction on two.
inline Value subtract(const std::vector<Value>& v)
{
    if(v.size() == 1)
    {
        return Value::number(-v[0].toNumber());
    }
    return Value::number(v[0].toNumber() - v[1].toNumber());
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
    return divide(v, [](double num, double den) { return Value::number(num / den); });
}
inline Value remainder(const std::vector<Value>& v)
{
    return divide(v, [](double num, double den) { return Value::number(std::fmod(num, den)); });
}
inline Value ceilDiv(const std::vector<Value>& v)
{
    return divide(v, [](double num, double den) { return Value::number(std::ceil(num / den)); });
}
inline Value divisible(const std::vector<Value>& v)
{
    // Exactly {"==": [{"%": [a, b]}, 0]}, the longhand the RFCs give for the
    // same check, so the short-hand and the spelled-out form agree on every
    // input -- including declining on a zero divisor.
    return divide(v, [](double num, double den) { return Value(std::fmod(num, den) == 0.0); });
}

/// NaN never wins, so a NaN argument is skipped unless every argument is one.
inline Value extremum(const std::vector<Value>& v, bool wantMax)
{
    double best = std::nan("");
    for(const Value& x : v)
    {
        const double n = x.toNumber();
        if(std::isnan(best) || (wantMax ? n > best : n < best))
        {
            best = n;
        }
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
    return Value::number(std::fabs(v[0].toNumber()));
}

inline Value power(const std::vector<Value>& v)
{
    return Value::number(std::pow(v[0].toNumber(), v[1].toNumber()));
}

inline Value log2Of(const std::vector<Value>& v)
{
    const double n = v[0].toNumber();
    if(n <= 0.0)
    {
        return {}; // log2 declines on a non-positive argument
    }
    return Value::number(std::log2(n));
}

inline Value reciprocalSqrt(const std::vector<Value>& v)
{
    const double n = v[0].toNumber();
    if(n <= 0.0)
    {
        return {}; // rsqrt declines on a non-positive argument
    }
    return Value::number(1.0 / std::sqrt(n));
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

// ---- operator table -------------------------------------------------------
// The single place an operator is defined: its key, its accepted arity, and
// its implementation. Adding an operator is one row plus one function in
// `ops`, and a row that names no implementation fails the static_assert below
// rather than evaluating to null (which the language reads as "unresolved",
// so a half-wired operator would make a narrowing predicate silently decline).

/// `maxArity` sentinel for an operator that accepts any number of arguments.
/// A real ceiling here would be a new limit the language never had, so it is
/// the largest representable count rather than a small sentinel.
inline constexpr std::size_t VARIADIC = std::numeric_limits<std::size_t>::max();

struct OpSpec
{
    std::string_view key;
    std::size_t minArity;
    std::size_t maxArity;
    ops::EagerFn eager; ///< exactly one of `eager` / `lazy` is set
    ops::LazyFn lazy;
};

/// Convenience builders keeping the table rows readable: an operator is eager
/// unless it must control its own argument evaluation.
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
        // Exactly one handler: neither would evaluate to null at run time,
        // both would make the eager/lazy choice ambiguous.
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
        // Evaluate every argument once, so a null is detected before the
        // operator coerces it and so each argument is evaluated exactly once.
        std::vector<Value> v;
        v.reserve(args.size());
        for(const auto& c : args)
        {
            v.push_back(c->eval(d));
        }
        for(const Value& x : v)
        {
            if(x.isNull())
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

inline Value jsonScalarToValue(const nlohmann::json& j)
{
    if(j.is_boolean())
    {
        return {j.get<bool>()};
    }
    if(j.is_number_integer() || j.is_number_unsigned())
    {
        return {j.get<std::int64_t>()};
    }
    if(j.is_number_float())
    {
        return {j.get<double>()};
    }
    return {}; // null
}

// ---- layout aliases -------------------------------------------------------
// A `stride_order` is an IntArray: for each logical dimension d, that
// dimension's stride rank, 0 being the fastest-varying. The common layouts get
// names, and a name expands to its array here, at compile time, so the array
// stays the single canonical form and evaluation never sees an alias.
struct LayoutAlias
{
    const char* name;
    const std::int64_t* order;
    std::size_t rank;
};

inline const LayoutAlias* lookupLayoutAlias(const std::string& name)
{
    static const std::int64_t s_nchw[] = {3, 2, 1, 0};
    static const std::int64_t s_nhwc[] = {3, 0, 2, 1};
    static const std::int64_t s_ncdhw[] = {4, 3, 2, 1, 0};
    static const std::int64_t s_ndhwc[] = {4, 0, 3, 2, 1};
    static const std::int64_t s_bhsd[] = {3, 2, 1, 0};
    static const std::array<LayoutAlias, 5> s_table = {{{"nchw", s_nchw, 4},
                                                        {"nhwc", s_nhwc, 4},
                                                        {"ncdhw", s_ncdhw, 5},
                                                        {"ndhwc", s_ndhwc, 5},
                                                        {"bhsd", s_bhsd, 4}}};
    for(const auto& e : s_table)
    {
        if(name == e.name)
        {
            return &e;
        }
    }
    return nullptr;
}

inline std::string knownLayoutAliases()
{
    return "nchw, nhwc, ncdhw, ndhwc, bhsd";
}

/// The variable path in a sigil-prefixed string, or nullptr if `j` is not one.
inline const std::string* variablePath(const nlohmann::json& j, char sigil)
{
    if(!j.is_string())
    {
        return nullptr;
    }
    const auto& s = j.get_ref<const nlohmann::json::string_t&>();
    // "$$x" is an escaped literal, and a bare "$" is rejected in compileNode.
    if(s.size() < 2 || s[0] != sigil || s[1] == sigil)
    {
        return nullptr;
    }
    return &s;
}

/// True for a reference whose last path segment is `stride_order`.
inline bool isStrideOrderRef(const nlohmann::json& j, char sigil)
{
    const std::string* s = variablePath(j, sigil);
    if(s == nullptr)
    {
        return false;
    }
    static const std::string k_suffix = ".stride_order";
    return s->size() > k_suffix.size() + 1
           && s->compare(s->size() - k_suffix.size(), k_suffix.size(), k_suffix) == 0;
}

/// The variable root of a reference: "$q.stride_order" -> "q".
inline std::string variableRoot(const std::string& sigilPath)
{
    const std::string path = sigilPath.substr(1);
    return path.substr(0, path.find_first_of(".["));
}

/// Collect `{"==": ["$x.rank", N]}` rank pins that hold unconditionally: the
/// root, and anything reachable from it through `and` only. A pin inside an
/// `or` / `if` / `!` arm is conditional and cannot contradict an alias, so it
/// is deliberately not collected.
inline void
    collectRankPins(const nlohmann::json& j, char sigil, std::map<std::string, std::int64_t>& pins)
{
    if(!j.is_object() || j.size() != 1)
    {
        return;
    }
    const auto it = j.begin();
    const std::string& key = it.key();
    const nlohmann::json& val = it.value();
    if(key == "and" && val.is_array())
    {
        for(const auto& e : val)
        {
            collectRankPins(e, sigil, pins);
        }
        return;
    }
    if(key != "==" || !val.is_array() || val.size() != 2)
    {
        return;
    }
    for(std::size_t i = 0; i < 2; ++i)
    {
        const std::string* s = variablePath(val.at(i), sigil);
        const nlohmann::json& other = val.at(1 - i);
        if(s == nullptr || !other.is_number_integer())
        {
            continue;
        }
        static const std::string k_suffix = ".rank";
        if(s->size() > k_suffix.size() + 1
           && s->compare(s->size() - k_suffix.size(), k_suffix.size(), k_suffix) == 0)
        {
            // First pin wins; a second, contradictory one makes the criteria
            // unsatisfiable on its own terms, which is not the alias's problem.
            pins.emplace(variableRoot(*s), other.get<std::int64_t>());
        }
    }
}

/// Resolve one alias string against a `stride_order` reference, or throw.
inline nlohmann::json resolveLayoutAlias(const nlohmann::json& aliasNode,
                                         const std::string& refPath,
                                         const std::map<std::string, std::int64_t>& rankPins)
{
    // A stride_order is an IntArray, so a string in this position can only be
    // an alias; an unknown one is a typo that would otherwise compare unequal
    // forever and decline silently at match time.
    const std::string& name = aliasNode.get_ref<const nlohmann::json::string_t&>();
    const LayoutAlias* alias = lookupLayoutAlias(name);
    if(alias == nullptr)
    {
        throw JsonExpressionCompileError("unknown layout alias '" + name + "' compared against "
                                         + refPath + "; expected an integer array or one of: "
                                         + knownLayoutAliases());
    }
    // Every alias is fixed-rank, so an alias compared against a tensor the
    // criteria pin to a different rank can never hold. Refuse it here rather
    // than let it decline silently on every graph.
    const auto pin = rankPins.find(variableRoot(refPath));
    if(pin != rankPins.end() && pin->second != static_cast<std::int64_t>(alias->rank))
    {
        throw JsonExpressionCompileError(
            "layout alias '" + name + "' is rank " + std::to_string(alias->rank)
            + ", but the expression pins " + refPath + " to rank " + std::to_string(pin->second));
    }
    return nlohmann::json(std::vector<std::int64_t>(alias->order, alias->order + alias->rank));
}

/// Rewrite every layout alias into its canonical array. An alias is recognized
/// only where a `stride_order` reference gives it that meaning -- opposite one
/// in an `==` / `!=`, or as an element of the array an `in` searches -- so
/// "nhwc" stays an ordinary string literal everywhere else.
inline nlohmann::json expandLayoutAliases(const nlohmann::json& j,
                                          char sigil,
                                          const std::map<std::string, std::int64_t>& rankPins)
{
    if(j.is_array())
    {
        nlohmann::json out = nlohmann::json::array();
        for(const auto& e : j)
        {
            out.push_back(expandLayoutAliases(e, sigil, rankPins));
        }
        return out;
    }
    if(!j.is_object())
    {
        return j;
    }

    nlohmann::json out = nlohmann::json::object();
    for(auto it = j.begin(); it != j.end(); ++it)
    {
        const std::string& key = it.key();
        const nlohmann::json& val = it.value();
        const bool binary = val.is_array() && val.size() == 2;

        // {"==" / "!=": [$x.stride_order, <alias>]}, either operand order.
        if(binary && (key == "==" || key == "!="))
        {
            nlohmann::json args = nlohmann::json::array();
            for(std::size_t i = 0; i < 2; ++i)
            {
                const nlohmann::json& side = val.at(i);
                const nlohmann::json& ref = val.at(1 - i);
                if(isStrideOrderRef(ref, sigil) && side.is_string())
                {
                    args.push_back(resolveLayoutAlias(side, ref.get<std::string>(), rankPins));
                }
                else
                {
                    args.push_back(expandLayoutAliases(side, sigil, rankPins));
                }
            }
            out[key] = std::move(args);
            continue;
        }

        // {"in": [$x.stride_order, [<alias-or-array>, ...]]} -- the documented
        // way to accept a set of layouts. Only the haystack's own elements are
        // aliases; a nested expression there is left alone.
        if(binary && key == "in" && isStrideOrderRef(val.at(0), sigil) && val.at(1).is_array())
        {
            const std::string refPath = val.at(0).get<std::string>();
            nlohmann::json hay = nlohmann::json::array();
            for(const auto& e : val.at(1))
            {
                hay.push_back(e.is_string() ? resolveLayoutAlias(e, refPath, rankPins)
                                            : expandLayoutAliases(e, sigil, rankPins));
            }
            out[key] = nlohmann::json::array({val.at(0), std::move(hay)});
            continue;
        }

        out[key] = expandLayoutAliases(val, sigil, rankPins);
    }
    return out;
}

inline NodePtr compileNode(const nlohmann::json& j, char sigil);

inline NodePtr compileObject(const nlohmann::json& j, char sigil)
{
    if(j.size() != 1)
    {
        throw JsonExpressionCompileError("expression object must have exactly one operator key");
    }
    const auto it = j.begin();
    const std::string& key = it.key();
    const nlohmann::json& val = it.value();
    if(key == "var")
    {
        throw JsonExpressionCompileError(
            "the 'var' operator is not supported; write a variable as a sigil-prefixed "
            "string (\"$path\"), and use 'value_or_default' for a fallback");
    }

    const OpSpec* spec = lookupOp(key);
    if(spec == nullptr)
    {
        throw JsonExpressionCompileError("unrecognized operation: " + key);
    }

    auto node = std::make_unique<OpNode>();
    node->spec = spec;
    if(val.is_array())
    {
        node->args.reserve(val.size());
        for(const auto& e : val)
        {
            node->args.push_back(compileNode(e, sigil));
        }
    }
    else
    {
        node->args.push_back(compileNode(val, sigil));
    }
    checkArity(*spec, node->args.size(), key);
    return node;
}

inline NodePtr compileNode(const nlohmann::json& j, char sigil)
{
    if(j.is_object())
    {
        return compileObject(j, sigil);
    }
    if(j.is_array())
    {
        auto n = std::make_unique<ArrayNode>();
        n->items.reserve(j.size());
        for(const auto& e : j)
        {
            n->items.push_back(compileNode(e, sigil));
        }
        return n;
    }
    if(j.is_string())
    {
        const auto& s = j.get_ref<const nlohmann::json::string_t&>();
        if(s.empty() || s[0] != sigil)
        {
            return std::make_unique<LiteralNode>(Value(s));
        }
        if(s.size() >= 2 && s[1] == sigil)
        {
            return std::make_unique<LiteralNode>(Value(s.substr(1))); // "$$x" -> "$x"
        }
        if(s.size() == 1)
        {
            throw JsonExpressionCompileError("whole-document variable reference is not supported");
        }
        auto n = std::make_unique<VarNode>();
        n->path = s.substr(1);
        return n;
    }
    return std::make_unique<LiteralNode>(jsonScalarToValue(j));
}

// ---- variable iteration ---------------------------------------------------
// Lazily yields, in pre-order, a reference to every variable path referenced
// by a compiled node tree. References point into the live VarNode::path, so no
// strings are copied; duplicates are yielded as they occur (build a std::set
// from the range if you need the unique, sorted set).
class VarIterator
{
public:
    using value_type = std::string;
    using reference = const std::string&;
    using pointer = const std::string*;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::input_iterator_tag;

    VarIterator() = default; // end
    explicit VarIterator(const Node* root)
    {
        if(root != nullptr)
        {
            _stack.push_back(root);
        }
        advance();
    }

    reference operator*() const
    {
        return *_cur;
    }
    pointer operator->() const
    {
        return _cur;
    }

    VarIterator& operator++()
    {
        advance();
        return *this;
    }
    VarIterator operator++(int)
    {
        VarIterator tmp = *this;
        advance();
        return tmp;
    }

    bool operator==(const VarIterator& o) const
    {
        return _cur == nullptr && o._cur == nullptr;
    }
    bool operator!=(const VarIterator& o) const
    {
        return !(*this == o);
    }

private:
    void advance()
    {
        while(!_stack.empty())
        {
            const Node* n = _stack.back();
            _stack.pop_back();
            n->pushChildren(_stack);
            if(const std::string* p = n->variable())
            {
                _cur = p;
                return;
            }
        }
        _cur = nullptr;
    }

    std::vector<const Node*> _stack;
    const std::string* _cur = nullptr;
};

class VarRange
{
public:
    explicit VarRange(const Node* root)
        : _begin(root)
    {
    }
    const VarIterator& begin() const
    {
        return _begin;
    }
    const VarIterator& end() const
    {
        return _end;
    }

private:
    VarIterator _begin;
    VarIterator _end;
};

} // namespace detail

// ===========================================================================
// Expression - a compiled, reusable expression
// ===========================================================================
template <class DataT>
class Expression
{
public:
    Expression() = default;
    explicit Expression(detail::NodePtr root)
        : _root(std::move(root))
    {
    }

    /// Evaluate against a data source. Cheap: walks the pre-compiled tree.
    Value operator()(const DataT& data) const
    {
        if(!_root)
        {
            return {};
        }
        const detail::DataSourceAdapter<DataT> source(data);
        return _root->eval(source);
    }
    Value evaluate(const DataT& data) const
    {
        return (*this)(data);
    }

    explicit operator bool() const
    {
        return static_cast<bool>(_root);
    }

    /// A lazy, pre-order range over every variable path referenced in the
    /// expression. References point into the live tree, so the range must not
    /// outlive this Expression. Duplicates are yielded as they occur;
    /// construct a std::set from the range for the unique, sorted set.
    detail::VarRange variables() const
    {
        return detail::VarRange(_root.get());
    }

private:
    detail::NodePtr _root;
};

/// Compile a rule into a reusable Expression bound to data source
/// type DataT. Throws JsonExpressionCompileError on malformed rules.
///
/// Layout aliases ("nhwc" and friends) opposite a `stride_order` reference are
/// expanded to their canonical integer arrays first, so the compiled tree and
/// evaluation see only arrays.
template <class DataT>
Expression<DataT> compile(const nlohmann::json& rule, char varSigil = '$')
{
    std::map<std::string, std::int64_t> rankPins;
    detail::collectRankPins(rule, varSigil, rankPins);
    const nlohmann::json expanded = detail::expandLayoutAliases(rule, varSigil, rankPins);
    return Expression<DataT>(detail::compileNode(expanded, varSigil));
}

/// True when any variable referenced by `expr` has `root` as its first path
/// token (the segment before the first '.'/'[' separator). The paths yielded by
/// Expression::variables() are already sigil-stripped, so `root` is given
/// without the sigil (e.g. "kernel"). Short-circuits on the first match.
template <class DataT>
bool referencesVariableRoot(const Expression<DataT>& expr, std::string_view root)
{
    for(const std::string& path : expr.variables())
    {
        const std::size_t end = path.find_first_of(".[");
        const std::string_view first(path.data(), end == std::string::npos ? path.size() : end);
        if(first == root)
        {
            return true;
        }
    }
    return false;
}

} // namespace hipdnn_plugin_sdk::ingestor::jsonexpr

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

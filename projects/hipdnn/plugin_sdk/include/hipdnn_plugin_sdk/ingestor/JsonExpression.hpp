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

#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
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
        switch(_v.index())
        {
        case 1:
            return std::get<bool>(_v);
        case 2:
            return std::get<std::int64_t>(_v) != 0;
        case 3:
            return std::get<double>(_v) != 0.0;
        case 4:
            return !std::get<std::string>(_v).empty();
        case 5:
            return !std::get<Array>(_v).empty();
        default:
            return false; // null
        }
    }

    /// JS Number() coercion. Non-numeric strings and multi-element arrays yield
    /// NaN; empty string / empty array / null yield 0.
    double toNumber() const
    {
        switch(_v.index())
        {
        case 1:
            return std::get<bool>(_v) ? 1.0 : 0.0;
        case 2:
            return static_cast<double>(std::get<std::int64_t>(_v));
        case 3:
            return std::get<double>(_v);
        case 4:
            return stringToNumber(std::get<std::string>(_v));
        case 5:
        {
            const auto& a = std::get<Array>(_v);
            if(a.empty())
            {
                return 0.0;
            }
            if(a.size() == 1)
            {
                return a.front().toNumber();
            }
            return std::nan("");
        }
        default:
            return 0.0; // null
        }
    }

    /// Structural equality (== / !=). Integers and doubles of equal value
    /// compare equal; differing kinds do not.
    bool operator==(const Value& o) const
    {
        if(isNumber() && o.isNumber())
        {
            return toNumber() == o.toNumber();
        }
        if(_v.index() != o._v.index())
        {
            return false;
        }
        switch(_v.index())
        {
        case 0:
            return true; // null == null
        case 1:
            return std::get<bool>(_v) == std::get<bool>(o._v);
        case 4:
            return std::get<std::string>(_v) == std::get<std::string>(o._v);
        case 5:
            return std::get<Array>(_v) == std::get<Array>(o._v);
        default:
            return false;
        }
    }
    bool operator!=(const Value& o) const
    {
        return !(*this == o);
    }

    /// Three-way compare for ordering. Returns -1/0/1, or 2 (incomparable, NaN)
    /// which makes every ordering test false. Two strings compare lexically.
    static int compare(const Value& a, const Value& b)
    {
        if(a.isString() && b.isString())
        {
            const auto& x = a.asString();
            const auto& y = b.asString();
            if(x < y)
            {
                return -1;
            }
            return x > y ? 1 : 0;
        }
        const double x = a.toNumber();
        const double y = b.toNumber();
        if(std::isnan(x) || std::isnan(y))
        {
            return 2;
        }
        if(x < y)
        {
            return -1;
        }
        return x > y ? 1 : 0;
    }

    /// Human-readable rendering, mainly for diagnostics and tests.
    std::string dump() const
    {
        switch(_v.index())
        {
        case 1:
            return std::get<bool>(_v) ? "true" : "false";
        case 2:
            return std::to_string(std::get<std::int64_t>(_v));
        case 3:
        {
            std::string s = std::to_string(std::get<double>(_v));
            return s;
        }
        case 4:
            return "\"" + std::get<std::string>(_v) + "\"";
        case 5:
        {
            std::string s = "[";
            const auto& a = std::get<Array>(_v);
            for(std::size_t i = 0; i < a.size(); ++i)
            {
                s += ((i != 0u) ? "," : "") + a[i].dump();
            }
            return s + "]";
        }
        default:
            return "null";
        }
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

    // null, bool, int64, double, string, array - indices used by the switches.
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
enum class Op
{
    ADD,
    SUB,
    MUL,
    DIV,
    MOD,
    MIN,
    MAX,
    LT,
    LE,
    GT,
    GE,
    EQ,
    NEQ,
    NOT,
    NOT_NOT,
    IF,
    AND,
    OR,
    // Spelled MEMBERSHIP, not IN: <minwindef.h> defines IN as an empty
    // annotation macro, so an enumerator named IN fails to compile on Windows
    // in any translation unit that also pulls in windows.h. The operator key
    // stays "in".
    MEMBERSHIP,
    CEIL_DIV,
    DIVISIBLE,
    ABS,
    POW,
    LOG2,
    RSQRT,
    VALUE_OR_DEFAULT,
    PRESENT,
    NOT_PRESENT
};

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

struct OpNode final : Node
{
    Op op;
    std::vector<NodePtr> args;

    // Null is "unresolved", not a value: an absent optional field's read must
    // neither pass nor fail a predicate. Every operator below therefore
    // propagates a null operand instead of coercing it (null would otherwise
    // read as 0 / false / not-equal, so a narrowing check on an absent operand
    // would silently PASS). `present`, `not_present`, `value_or_default`, and
    // the short-circuit halves of `and`/`or` are the deliberate exceptions:
    // they answer "did this resolve?", so they always yield a real value. A
    // null root is rejected, because Value::truthy() reads null as false.
    static bool anyNull(const std::vector<Value>& v)
    {
        for(const Value& x : v)
        {
            if(x.isNull())
            {
                return true;
            }
        }
        return false;
    }

    // Evaluate every argument once, so a null is detected before the operator
    // coerces it and so each argument is evaluated exactly once.
    std::vector<Value> evalArgs(const IDataSource& d) const
    {
        std::vector<Value> out;
        out.reserve(args.size());
        for(const auto& c : args)
        {
            out.push_back(c->eval(d));
        }
        return out;
    }

    Value eval(const IDataSource& d) const override
    {
        switch(op)
        {
        case Op::ADD:
        {
            const std::vector<Value> v = evalArgs(d);
            if(anyNull(v))
            {
                return {};
            }
            double a = 0.0;
            for(const Value& x : v)
            {
                a += x.toNumber();
            }
            return Value::number(a);
        }
        case Op::MUL:
        {
            const std::vector<Value> v = evalArgs(d);
            if(anyNull(v))
            {
                return {};
            }
            double a = 1.0;
            for(const Value& x : v)
            {
                a *= x.toNumber();
            }
            return Value::number(a);
        }
        case Op::SUB:
        {
            const std::vector<Value> v = evalArgs(d);
            if(anyNull(v))
            {
                return {};
            }
            if(v.size() == 1)
            {
                return Value::number(-v[0].toNumber());
            }
            return Value::number(v[0].toNumber() - v[1].toNumber());
        }
        case Op::DIV:
        case Op::MOD:
        case Op::CEIL_DIV:
        case Op::DIVISIBLE:
        {
            const std::vector<Value> v = evalArgs(d);
            if(anyNull(v))
            {
                return {};
            }
            const double num = v[0].toNumber();
            const double den = v[1].toNumber();
            if(den == 0.0)
            {
                return {}; // zero divisor declines rather than yielding inf/NaN
            }
            if(op == Op::DIV)
            {
                return Value::number(num / den);
            }
            if(op == Op::MOD)
            {
                return Value::number(std::fmod(num, den));
            }
            if(op == Op::CEIL_DIV)
            {
                return Value::number(std::ceil(num / den));
            }
            // Exactly {"==": [{"%": [a, b]}, 0]}, the longhand the RFCs give
            // for the same check, so the short-hand and the spelled-out form
            // agree on every input -- including declining on a zero divisor.
            return {std::fmod(num, den) == 0.0};
        }
        case Op::MIN:
        case Op::MAX:
        {
            const std::vector<Value> v = evalArgs(d);
            if(anyNull(v))
            {
                return {};
            }
            double best = std::nan("");
            for(const Value& x : v)
            {
                const double n = x.toNumber();
                if(std::isnan(best) || (op == Op::MIN ? n < best : n > best))
                {
                    best = n;
                }
            }
            return Value::number(best);
        }
        case Op::LT:
        case Op::LE:
        case Op::GT:
        case Op::GE:
        {
            const std::vector<Value> v = evalArgs(d);
            if(anyNull(v))
            {
                return {};
            }
            const auto ordered = [this](const Value& x, const Value& y) {
                const int c = Value::compare(x, y);
                switch(op)
                {
                case Op::LT:
                    return c == -1;
                case Op::LE:
                    return c == -1 || c == 0;
                case Op::GT:
                    return c == 1;
                default:
                    return c == 1 || c == 0;
                }
            };
            // The 3-arg form is the between-chain: a < b < c.
            if(v.size() >= 3)
            {
                return {ordered(v[0], v[1]) && ordered(v[1], v[2])};
            }
            return {ordered(v[0], v[1])};
        }
        case Op::EQ:
        case Op::NEQ:
        {
            const std::vector<Value> v = evalArgs(d);
            // Two unresolved references are not "equal"; the question is
            // unanswerable, so it declines instead of accepting.
            if(anyNull(v))
            {
                return {};
            }
            return {op == Op::EQ ? v[0] == v[1] : v[0] != v[1]};
        }
        case Op::NOT:
        case Op::NOT_NOT:
        {
            const Value a = args[0]->eval(d);
            if(a.isNull())
            {
                return {}; // negating an unresolved reference stays unresolved
            }
            return {op == Op::NOT ? !a.truthy() : a.truthy()};
        }
        case Op::IF:
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
        case Op::AND:
        {
            // Kleene `and`: a definite false short-circuits even when another
            // argument is unresolved, so `and`-ing an inapplicable check beside
            // a failing one still declines. Otherwise a null makes the whole
            // conjunction unresolved.
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
        case Op::OR:
        {
            // Kleene `or`: a definite true short-circuits past an unresolved
            // argument, which is what lets
            // `{"or": [{"not_present": ["$bias"]}, {"==": ["$bias.dtype", ...]}]}`
            // accept input with no `bias` at all, even though the second arm
            // cannot run.
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
        case Op::MEMBERSHIP:
        {
            const std::vector<Value> v = evalArgs(d);
            if(anyNull(v))
            {
                return {};
            }
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
        case Op::ABS:
        case Op::LOG2:
        case Op::RSQRT:
        {
            const Value a = args[0]->eval(d);
            if(a.isNull())
            {
                return {};
            }
            const double n = a.toNumber();
            if(op == Op::ABS)
            {
                return Value::number(std::fabs(n));
            }
            if(n <= 0.0)
            {
                return {}; // log2/rsqrt decline on a non-positive argument
            }
            return Value::number(op == Op::LOG2 ? std::log2(n) : 1.0 / std::sqrt(n));
        }
        case Op::POW:
        {
            const std::vector<Value> v = evalArgs(d);
            if(anyNull(v))
            {
                return {};
            }
            return Value::number(std::pow(v[0].toNumber(), v[1].toNumber()));
        }
        case Op::VALUE_OR_DEFAULT:
        {
            // First arg is a variable reference; a null result means the path
            // did not resolve in the data source, so fall back to the default.
            const Value v = args[0]->eval(d);
            return v.isNull() ? args[1]->eval(d) : v;
        }
        case Op::PRESENT:
        case Op::NOT_PRESENT:
        {
            // Presence keys on *existence*, the same mechanism as
            // VALUE_OR_DEFAULT above: an unresolved path reads null. Unlike
            // every other operator these do not propagate that null -- asking
            // "was this supplied?" always yields a real boolean. Both fold with
            // `and` over their arguments, so one call decides a whole list.
            const bool wantNull = (op == Op::NOT_PRESENT);
            for(const auto& c : args)
            {
                if(c->eval(d).isNull() != wantNull)
                {
                    return {false};
                }
            }
            return {true};
        }
        default:
            break;
        }
        return {};
    }

    void pushChildren(std::vector<const Node*>& stack) const override
    {
        for(auto it = args.rbegin(); it != args.rend(); ++it)
        {
            stack.push_back(it->get());
        }
    }
};

inline const Op* lookupOp(const std::string& key)
{
    static const std::array<std::pair<const char*, Op>, 29> s_table
        = {{{"+", Op::ADD},
            {"-", Op::SUB},
            {"*", Op::MUL},
            {"/", Op::DIV},
            {"%", Op::MOD},
            {"min", Op::MIN},
            {"max", Op::MAX},
            {"<", Op::LT},
            {"<=", Op::LE},
            {">", Op::GT},
            {">=", Op::GE},
            {"==", Op::EQ},
            {"!=", Op::NEQ},
            {"!", Op::NOT},
            {"!!", Op::NOT_NOT},
            {"if", Op::IF},
            {"?:", Op::IF},
            {"and", Op::AND},
            {"or", Op::OR},
            {"in", Op::MEMBERSHIP},
            {"ceil_div", Op::CEIL_DIV},
            {"divisible", Op::DIVISIBLE},
            {"abs", Op::ABS},
            {"pow", Op::POW},
            {"log2", Op::LOG2},
            {"rsqrt", Op::RSQRT},
            {"value_or_default", Op::VALUE_OR_DEFAULT},
            {"present", Op::PRESENT},
            {"not_present", Op::NOT_PRESENT}}};
    for(const auto& e : s_table)
    {
        if(key == e.first)
        {
            return &e.second;
        }
    }
    return nullptr;
}

inline void checkArity(Op op, std::size_t n, const std::string& key)
{
    const auto require = [&](bool ok) {
        if(!ok)
        {
            throw JsonExpressionCompileError("operator '" + key
                                             + "' got wrong argument count: " + std::to_string(n));
        }
    };
    switch(op)
    {
    case Op::ADD:
    case Op::MUL:
        break; // any arity
    case Op::SUB:
        require(n == 1 || n == 2);
        break;
    case Op::DIV:
    case Op::MOD:
    case Op::EQ:
    case Op::NEQ:
    case Op::GT:
    case Op::GE:
        require(n == 2);
        break;
    case Op::LT:
    case Op::LE:
        require(n == 2 || n == 3);
        break;
    case Op::MIN:
    case Op::MAX:
    case Op::AND:
    case Op::OR:
        require(n >= 1);
        break;
    case Op::NOT:
    case Op::NOT_NOT:
        require(n == 1);
        break;
    case Op::IF:
        require(n >= 2);
        break;
    case Op::MEMBERSHIP:
    case Op::POW:
    case Op::CEIL_DIV:
    case Op::DIVISIBLE:
    case Op::VALUE_OR_DEFAULT:
        require(n == 2);
        break;
    case Op::PRESENT:
    case Op::NOT_PRESENT:
        require(n >= 1);
        break;
    case Op::ABS:
    case Op::LOG2:
    case Op::RSQRT:
        require(n == 1);
        break;
    default:
        break;
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

    const Op* op = lookupOp(key);
    if(op == nullptr)
    {
        throw JsonExpressionCompileError("unrecognized operation: " + key);
    }

    auto node = std::make_unique<OpNode>();
    node->op = *op;
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
    checkArity(*op, node->args.size(), key);
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

/// Convenience one-shot: compile and evaluate in a single call. Prefer
/// compile() + reuse when evaluating the same rule repeatedly.
template <class DataT>
Value evaluate(const nlohmann::json& rule, const DataT& data, char varSigil = '$')
{
    return compile<DataT>(rule, varSigil)(data);
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

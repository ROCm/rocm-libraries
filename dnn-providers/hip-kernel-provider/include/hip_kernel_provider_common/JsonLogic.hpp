// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// JsonLogic.hpp - single-file JsonLogic expression compiler.
//
// All names below live in namespace hip_kernel_provider_common::jsonlogic;
// these examples assume `namespace jlogic = hip_kernel_provider_common::jsonlogic;`.
//
// A JsonLogic expression (an nlohmann::json value) is *compiled once* into a
// reusable jlogic::Expression<Data>, then evaluated many times against different
// data sources:
//
//     struct MyData {                       // your data source
//         jlogic::Value getData(const std::string& path) const;
//     };
//
//     auto expr = jlogic::compile<MyData>(rule);   // parse + build tree once
//     jlogic::Value r1 = expr(dataA);              // evaluate, no re-parse
//     jlogic::Value r2 = expr(dataB);              // reuse for other data
//
// The runtime value type (jlogic::Value) is a small standalone variant that does
// NOT depend on nlohmann/json; nlohmann is used only to express the rule being
// compiled. The data source is any C++ type exposing
//   Value getData(const std::string&)
// which resolves a variable path to a Value; a null return means "not found",
// triggering a var default if one is given. Variable paths are always
// non-empty (whole-document references are not supported).
//
// Inline variables (extension over stock JsonLogic)
// -------------------------------------------------
// A string prefixed with a sigil (default '$') is a variable reference, so a
// variable can appear anywhere a literal can:
//     {"+": ["$x", "$y"]}   ==   {"+": [{"var": "x"}, {"var": "y"}]}
//     "$a.b"  nested path      "$$x"  literal "$x"
// Strings without the sigil remain literals, so stock rules keep working.
//
// Scope: core operator set (data access, logic, comparison, arithmetic), the
// membership operator `in`, the `value_or_default` fallback, and the value-core
// math extensions the hipDNN UMD needs (ceil_div, abs, pow, log2, rsqrt).
// Collection/string operators (map/reduce/filter/cat/substr/...) are not
// included.

#include <nlohmann/json.hpp>

#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace hip_kernel_provider_common::jsonlogic
{
/// Thrown when a rule cannot be compiled (unknown operator, bad arity, ...).
class JsonLogicCompileError : public std::runtime_error
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

    /// JsonLogic truthiness: false, 0, "", null and the empty array are falsy.
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
    IN,
    CEIL_DIV,
    ABS,
    POW,
    LOG2,
    RSQRT,
    VALUE_OR_DEFAULT
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
    NodePtr defaultExpr; // optional fallback when the lookup returns null

    Value eval(const IDataSource& data) const override
    {
        Value r = data.getData(path);
        if(r.isNull() && defaultExpr)
        {
            return defaultExpr->eval(data);
        }
        return r;
    }

    const std::string* variable() const override
    {
        return &path;
    }
    void pushChildren(std::vector<const Node*>& stack) const override
    {
        if(defaultExpr)
        {
            stack.push_back(defaultExpr.get());
        }
    }
};

struct OpNode final : Node
{
    Op op;
    std::vector<NodePtr> args;

    Value eval(const IDataSource& d) const override
    {
        switch(op)
        {
        case Op::ADD:
        {
            double a = 0.0;
            for(const auto& c : args)
            {
                a += c->eval(d).toNumber();
            }
            return Value::number(a);
        }
        case Op::MUL:
        {
            double a = 1.0;
            for(const auto& c : args)
            {
                a *= c->eval(d).toNumber();
            }
            return Value::number(a);
        }
        case Op::SUB:
        {
            if(args.size() == 1)
            {
                return Value::number(-args[0]->eval(d).toNumber());
            }
            return Value::number(args[0]->eval(d).toNumber() - args[1]->eval(d).toNumber());
        }
        case Op::DIV:
            return Value::number(args[0]->eval(d).toNumber() / args[1]->eval(d).toNumber());
        case Op::MOD:
            return Value::number(
                std::fmod(args[0]->eval(d).toNumber(), args[1]->eval(d).toNumber()));
        case Op::MIN:
        {
            double best = std::nan("");
            for(const auto& c : args)
            {
                const double n = c->eval(d).toNumber();
                if(std::isnan(best) || n < best)
                {
                    best = n;
                }
            }
            return Value::number(best);
        }
        case Op::MAX:
        {
            double best = std::nan("");
            for(const auto& c : args)
            {
                const double n = c->eval(d).toNumber();
                if(std::isnan(best) || n > best)
                {
                    best = n;
                }
            }
            return Value::number(best);
        }
        case Op::LT:
        {
            const Value a = args[0]->eval(d);
            const Value b = args[1]->eval(d);
            if(args.size() >= 3)
            {
                return {Value::compare(a, b) == -1 && Value::compare(b, args[2]->eval(d)) == -1};
            }
            return {Value::compare(a, b) == -1};
        }
        case Op::LE:
        {
            const Value a = args[0]->eval(d);
            const Value b = args[1]->eval(d);
            const auto le = [](const Value& x, const Value& y) {
                const int c = Value::compare(x, y);
                return c == -1 || c == 0;
            };
            if(args.size() >= 3)
            {
                return {le(a, b) && le(b, args[2]->eval(d))};
            }
            return {le(a, b)};
        }
        case Op::GT:
            return {Value::compare(args[0]->eval(d), args[1]->eval(d)) == 1};
        case Op::GE:
        {
            const int c = Value::compare(args[0]->eval(d), args[1]->eval(d));
            return {c == 1 || c == 0};
        }
        case Op::EQ:
            return {args[0]->eval(d) == args[1]->eval(d)};
        case Op::NEQ:
            return {args[0]->eval(d) != args[1]->eval(d)};
        case Op::NOT:
            return {!args[0]->eval(d).truthy()};
        case Op::NOT_NOT:
            return {args[0]->eval(d).truthy()};
        case Op::IF:
        {
            std::size_t i = 0;
            for(; i + 1 < args.size(); i += 2)
            {
                if(args[i]->eval(d).truthy())
                {
                    return args[i + 1]->eval(d);
                }
            }
            return i < args.size() ? args[i]->eval(d) : Value();
        }
        case Op::AND:
        {
            Value cur(true);
            for(const auto& c : args)
            {
                cur = c->eval(d);
                if(!cur.truthy())
                {
                    return cur; // first falsy
                }
            }
            return cur; // last value
        }
        case Op::OR:
        {
            Value cur;
            for(const auto& c : args)
            {
                cur = c->eval(d);
                if(cur.truthy())
                {
                    return cur; // first truthy
                }
            }
            return cur; // last value
        }
        case Op::IN:
        {
            const Value needle = args[0]->eval(d);
            const Value hay = args[1]->eval(d);
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
        case Op::CEIL_DIV:
        {
            const double num = args[0]->eval(d).toNumber();
            const double den = args[1]->eval(d).toNumber();
            return Value::number(std::ceil(num / den));
        }
        case Op::ABS:
            return Value::number(std::fabs(args[0]->eval(d).toNumber()));
        case Op::POW:
            return Value::number(
                std::pow(args[0]->eval(d).toNumber(), args[1]->eval(d).toNumber()));
        case Op::LOG2:
            return Value::number(std::log2(args[0]->eval(d).toNumber()));
        case Op::RSQRT:
            return Value::number(1.0 / std::sqrt(args[0]->eval(d).toNumber()));
        case Op::VALUE_OR_DEFAULT:
        {
            // First arg is a variable reference; a null result means the path
            // did not resolve in the data source, so fall back to the default.
            Value const v = args[0]->eval(d);
            return v.isNull() ? args[1]->eval(d) : v;
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
    static const std::array<std::pair<const char*, Op>, 26> s_table
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
            {"in", Op::IN},
            {"ceil_div", Op::CEIL_DIV},
            {"abs", Op::ABS},
            {"pow", Op::POW},
            {"log2", Op::LOG2},
            {"rsqrt", Op::RSQRT},
            {"value_or_default", Op::VALUE_OR_DEFAULT}}};
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
            throw JsonLogicCompileError("operator '" + key
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
    case Op::IN:
    case Op::POW:
    case Op::CEIL_DIV:
    case Op::VALUE_OR_DEFAULT:
        require(n == 2);
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

inline NodePtr compileNode(const nlohmann::json& j, char sigil);

inline NodePtr compileVar(const nlohmann::json& val, char sigil)
{
    auto n = std::make_unique<VarNode>();
    const auto setStaticPath = [&](const nlohmann::json& p) {
        if(p.is_string())
        {
            std::string s = p.get<std::string>();
            if(s.empty())
            {
                throw JsonLogicCompileError("whole-document variable reference is not supported");
            }
            n->path = std::move(s);
        }
        else if(p.is_number())
        {
            n->path = std::to_string(p.get<std::int64_t>());
        }
        else if(p.is_null())
        {
            throw JsonLogicCompileError("whole-document variable reference is not supported");
        }
        else
        {
            throw JsonLogicCompileError("computed variable keys are not supported");
        }
    };
    if(val.is_array())
    {
        if(val.empty())
        {
            throw JsonLogicCompileError("whole-document variable reference is not supported");
        }
        setStaticPath(val.at(0));
        if(val.size() > 1)
        {
            n->defaultExpr = compileNode(val.at(1), sigil);
        }
    }
    else
    {
        setStaticPath(val);
    }
    return n;
}

inline NodePtr compileObject(const nlohmann::json& j, char sigil)
{
    if(j.size() != 1)
    {
        throw JsonLogicCompileError("expression object must have exactly one operator key");
    }
    const auto it = j.begin();
    const std::string& key = it.key();
    const nlohmann::json& val = it.value();
    if(key == "var")
    {
        return compileVar(val, sigil);
    }

    const Op* op = lookupOp(key);
    if(op == nullptr)
    {
        throw JsonLogicCompileError("unrecognized operation: " + key);
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
            throw JsonLogicCompileError("whole-document variable reference is not supported");
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
// Expression - a compiled, reusable JsonLogic expression
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

/// Compile a JsonLogic rule into a reusable Expression bound to data source
/// type DataT. Throws JsonLogicCompileError on malformed rules.
template <class DataT>
Expression<DataT> compile(const nlohmann::json& rule, char varSigil = '$')
{
    return Expression<DataT>(detail::compileNode(rule, varSigil));
}

/// Convenience one-shot: compile and evaluate in a single call. Prefer
/// compile() + reuse when evaluating the same rule repeatedly.
template <class DataT>
Value evaluate(const nlohmann::json& rule, const DataT& data, char varSigil = '$')
{
    return compile<DataT>(rule, varSigil)(data);
}

} // namespace hip_kernel_provider_common::jsonlogic

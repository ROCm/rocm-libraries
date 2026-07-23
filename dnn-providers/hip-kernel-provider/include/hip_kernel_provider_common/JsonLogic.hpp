// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// JsonLogic.hpp - single-file JsonLogic expression compiler.
//
// All names below live in namespace hip_kernel_provider_common::jsonlogic;
// these examples assume `namespace jl = hip_kernel_provider_common::jsonlogic;`.
//
// A JsonLogic expression (an nlohmann::json value) is *compiled once* into a
// reusable jl::Expression<Data>, then evaluated many times against different
// data sources:
//
//     struct MyData {                       // your data source
//         jl::Value getData(const std::string& path) const;
//     };
//
//     auto expr = jl::compile<MyData>(rule);   // parse + build tree once
//     jl::Value r1 = expr(dataA);              // evaluate, no re-parse
//     jl::Value r2 = expr(dataB);              // reuse for other data
//
// The runtime value type (jl::Value) is a small standalone variant that does
// NOT depend on nlohmann/json; nlohmann is used only to express the rule being
// compiled. The data source is any C++ type exposing
//   Value getData(const std::string&)
// which resolves a variable path to a Value ("" means the whole document; a
// null return means "not found", triggering a var default if one is given).
//
// Inline variables (extension over stock JsonLogic)
// -------------------------------------------------
// A string prefixed with a sigil (default '$') is a variable reference, so a
// variable can appear anywhere a literal can:
//     {"+": ["$x", "$y"]}   ==   {"+": [{"var": "x"}, {"var": "y"}]}
//     "$a.b"  nested path      "$"  whole document      "$$x"  literal "$x"
// Strings without the sigil remain literals, so stock rules keep working.
//
// Scope: core operator set (data access, logic, comparison, arithmetic), the
// membership operator `in`, the `value_or_default` fallback, and the value-core
// math extensions the hipDNN UMD needs (ceil_div, abs, pow, log2, rsqrt).
// Collection/string operators (map/reduce/filter/cat/substr/...) are not
// included.

#include <nlohmann/json.hpp>

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
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
        : v_(nullptr)
    {
    }
    Value(std::nullptr_t)
        : v_(nullptr)
    {
    }
    Value(bool b)
        : v_(b)
    {
    }
    Value(int i)
        : v_(static_cast<std::int64_t>(i))
    {
    }
    Value(std::int64_t i)
        : v_(i)
    {
    }
    Value(double d)
        : v_(d)
    {
    }
    Value(const char* s)
        : v_(std::string(s))
    {
    }
    Value(std::string s)
        : v_(std::move(s))
    {
    }
    Value(Array a)
        : v_(std::move(a))
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
                return Value(static_cast<std::int64_t>(t));
        }
        return Value(d);
    }

    bool is_null() const
    {
        return std::holds_alternative<std::nullptr_t>(v_);
    }
    bool is_bool() const
    {
        return std::holds_alternative<bool>(v_);
    }
    bool is_int() const
    {
        return std::holds_alternative<std::int64_t>(v_);
    }
    bool is_double() const
    {
        return std::holds_alternative<double>(v_);
    }
    bool is_number() const
    {
        return is_int() || is_double();
    }
    bool is_string() const
    {
        return std::holds_alternative<std::string>(v_);
    }
    bool is_array() const
    {
        return std::holds_alternative<Array>(v_);
    }

    bool as_bool() const
    {
        return std::get<bool>(v_);
    }
    std::int64_t as_int() const
    {
        return std::get<std::int64_t>(v_);
    }
    double as_double() const
    {
        return std::get<double>(v_);
    }
    const std::string& as_string() const
    {
        return std::get<std::string>(v_);
    }
    const Array& as_array() const
    {
        return std::get<Array>(v_);
    }

    /// JsonLogic truthiness: false, 0, "", null and the empty array are falsy.
    bool truthy() const
    {
        switch(v_.index())
        {
        case 1:
            return std::get<bool>(v_);
        case 2:
            return std::get<std::int64_t>(v_) != 0;
        case 3:
            return std::get<double>(v_) != 0.0;
        case 4:
            return !std::get<std::string>(v_).empty();
        case 5:
            return !std::get<Array>(v_).empty();
        default:
            return false; // null
        }
    }

    /// JS Number() coercion. Non-numeric strings and multi-element arrays yield
    /// NaN; empty string / empty array / null yield 0.
    double to_number() const
    {
        switch(v_.index())
        {
        case 1:
            return std::get<bool>(v_) ? 1.0 : 0.0;
        case 2:
            return static_cast<double>(std::get<std::int64_t>(v_));
        case 3:
            return std::get<double>(v_);
        case 4:
            return string_to_number(std::get<std::string>(v_));
        case 5:
        {
            const auto& a = std::get<Array>(v_);
            if(a.empty())
                return 0.0;
            if(a.size() == 1)
                return a.front().to_number();
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
        if(is_number() && o.is_number())
            return to_number() == o.to_number();
        if(v_.index() != o.v_.index())
            return false;
        switch(v_.index())
        {
        case 0:
            return true; // null == null
        case 1:
            return std::get<bool>(v_) == std::get<bool>(o.v_);
        case 4:
            return std::get<std::string>(v_) == std::get<std::string>(o.v_);
        case 5:
            return std::get<Array>(v_) == std::get<Array>(o.v_);
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
        if(a.is_string() && b.is_string())
        {
            const auto& x = a.as_string();
            const auto& y = b.as_string();
            return x < y ? -1 : (x > y ? 1 : 0);
        }
        const double x = a.to_number();
        const double y = b.to_number();
        if(std::isnan(x) || std::isnan(y))
            return 2;
        return x < y ? -1 : (x > y ? 1 : 0);
    }

    /// Human-readable rendering, mainly for diagnostics and tests.
    std::string dump() const
    {
        switch(v_.index())
        {
        case 1:
            return std::get<bool>(v_) ? "true" : "false";
        case 2:
            return std::to_string(std::get<std::int64_t>(v_));
        case 3:
        {
            std::string s = std::to_string(std::get<double>(v_));
            return s;
        }
        case 4:
            return "\"" + std::get<std::string>(v_) + "\"";
        case 5:
        {
            std::string s = "[";
            const auto& a = std::get<Array>(v_);
            for(std::size_t i = 0; i < a.size(); ++i)
                s += (i ? "," : "") + a[i].dump();
            return s + "]";
        }
        default:
            return "null";
        }
    }

private:
    static double string_to_number(const std::string& s)
    {
        std::size_t b = 0, e = s.size();
        while(b < e && std::isspace(static_cast<unsigned char>(s[b])))
            ++b;
        while(e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
            --e;
        if(b == e)
            return 0.0; // JS Number("") == 0
        const std::string t = s.substr(b, e - b);
        const char* first = t.c_str();
        char* last = nullptr;
        const double d = std::strtod(first, &last);
        if(last != first + t.size())
            return std::nan(""); // trailing garbage -> NaN
        return d;
    }

    // null, bool, int64, double, string, array - indices used by the switches.
    std::variant<std::nullptr_t, bool, std::int64_t, double, std::string, Array> v_;
};

namespace detail
{
// ---- data-source capability detection ------------------------------------
template <class T, class = void>
struct has_getData : std::false_type
{
};
template <class T>
struct has_getData<
    T,
    std::void_t<decltype(std::declval<const T&>().getData(std::declval<std::string>()))>>
    : std::true_type
{
};

/// Fetch a variable path from the data source.
template <class DataT>
Value fetch(const DataT& d, const std::string& path)
{
    static_assert(has_getData<DataT>::value,
                  "Data source must provide Value getData(std::string).");
    return d.getData(path);
}

// ---- compiled node tree ---------------------------------------------------
enum class Op
{
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Min,
    Max,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Neq,
    Not,
    NotNot,
    If,
    And,
    Or,
    In,
    CeilDiv,
    Abs,
    Pow,
    Log2,
    Rsqrt,
    ValueOrDefault
};

template <class DataT>
struct Node
{
    virtual ~Node() = default;
    virtual Value eval(const DataT&) const = 0;
};

template <class DataT>
using NodePtr = std::unique_ptr<Node<DataT>>;

template <class DataT>
struct LiteralNode final : Node<DataT>
{
    Value value;
    explicit LiteralNode(Value v)
        : value(std::move(v))
    {
    }
    Value eval(const DataT&) const override
    {
        return value;
    }
};

template <class DataT>
struct ArrayNode final : Node<DataT>
{
    std::vector<NodePtr<DataT>> items;
    Value eval(const DataT& d) const override
    {
        Value::Array a;
        a.reserve(items.size());
        for(const auto& it : items)
            a.push_back(it->eval(d));
        return Value(std::move(a));
    }
};

template <class DataT>
struct VarNode final : Node<DataT>
{
    bool has_static = false;
    std::string path; // used when has_static
    NodePtr<DataT> path_expr; // used when !has_static (computed key)
    NodePtr<DataT> default_expr; // optional fallback when fetch returns null

    Value eval(const DataT& d) const override
    {
        if(has_static)
        {
            Value r = fetch(d, path);
            if(r.is_null() && default_expr)
                return default_expr->eval(d);
            return r;
        }
        const Value p = path_expr->eval(d);
        std::string key;
        if(p.is_string())
            key = p.as_string();
        else if(p.is_int())
            key = std::to_string(p.as_int());
        else if(p.is_number())
            key = std::to_string(static_cast<std::int64_t>(p.to_number()));
        Value r = fetch(d, key);
        if(r.is_null() && default_expr)
            return default_expr->eval(d);
        return r;
    }
};

template <class DataT>
struct OpNode final : Node<DataT>
{
    Op op;
    std::vector<NodePtr<DataT>> args;

    Value eval(const DataT& d) const override
    {
        switch(op)
        {
        case Op::Add:
        {
            double a = 0.0;
            for(const auto& c : args)
                a += c->eval(d).to_number();
            return Value::number(a);
        }
        case Op::Mul:
        {
            double a = 1.0;
            for(const auto& c : args)
                a *= c->eval(d).to_number();
            return Value::number(a);
        }
        case Op::Sub:
        {
            if(args.size() == 1)
                return Value::number(-args[0]->eval(d).to_number());
            return Value::number(args[0]->eval(d).to_number() - args[1]->eval(d).to_number());
        }
        case Op::Div:
            return Value::number(args[0]->eval(d).to_number() / args[1]->eval(d).to_number());
        case Op::Mod:
            return Value::number(
                std::fmod(args[0]->eval(d).to_number(), args[1]->eval(d).to_number()));
        case Op::Min:
        {
            double best = std::nan("");
            for(const auto& c : args)
            {
                const double n = c->eval(d).to_number();
                if(std::isnan(best) || n < best)
                    best = n;
            }
            return Value::number(best);
        }
        case Op::Max:
        {
            double best = std::nan("");
            for(const auto& c : args)
            {
                const double n = c->eval(d).to_number();
                if(std::isnan(best) || n > best)
                    best = n;
            }
            return Value::number(best);
        }
        case Op::Lt:
        {
            const Value a = args[0]->eval(d);
            const Value b = args[1]->eval(d);
            if(args.size() >= 3)
                return Value(Value::compare(a, b) == -1
                             && Value::compare(b, args[2]->eval(d)) == -1);
            return Value(Value::compare(a, b) == -1);
        }
        case Op::Le:
        {
            const Value a = args[0]->eval(d);
            const Value b = args[1]->eval(d);
            const auto le = [](const Value& x, const Value& y) {
                const int c = Value::compare(x, y);
                return c == -1 || c == 0;
            };
            if(args.size() >= 3)
                return Value(le(a, b) && le(b, args[2]->eval(d)));
            return Value(le(a, b));
        }
        case Op::Gt:
            return Value(Value::compare(args[0]->eval(d), args[1]->eval(d)) == 1);
        case Op::Ge:
        {
            const int c = Value::compare(args[0]->eval(d), args[1]->eval(d));
            return Value(c == 1 || c == 0);
        }
        case Op::Eq:
            return Value(args[0]->eval(d) == args[1]->eval(d));
        case Op::Neq:
            return Value(args[0]->eval(d) != args[1]->eval(d));
        case Op::Not:
            return Value(!args[0]->eval(d).truthy());
        case Op::NotNot:
            return Value(args[0]->eval(d).truthy());
        case Op::If:
        {
            std::size_t i = 0;
            for(; i + 1 < args.size(); i += 2)
                if(args[i]->eval(d).truthy())
                    return args[i + 1]->eval(d);
            return i < args.size() ? args[i]->eval(d) : Value();
        }
        case Op::And:
        {
            Value cur(true);
            for(const auto& c : args)
            {
                cur = c->eval(d);
                if(!cur.truthy())
                    return cur; // first falsy
            }
            return cur; // last value
        }
        case Op::Or:
        {
            Value cur;
            for(const auto& c : args)
            {
                cur = c->eval(d);
                if(cur.truthy())
                    return cur; // first truthy
            }
            return cur; // last value
        }
        case Op::In:
        {
            const Value needle = args[0]->eval(d);
            const Value hay = args[1]->eval(d);
            if(hay.is_array())
            {
                for(const auto& e : hay.as_array())
                    if(e == needle)
                        return Value(true);
                return Value(false);
            }
            if(hay.is_string())
            {
                const std::string n = needle.is_string() ? needle.as_string() : needle.dump();
                return Value(hay.as_string().find(n) != std::string::npos);
            }
            return Value(false);
        }
        case Op::CeilDiv:
        {
            const double num = args[0]->eval(d).to_number();
            const double den = args[1]->eval(d).to_number();
            return Value::number(std::ceil(num / den));
        }
        case Op::Abs:
            return Value::number(std::fabs(args[0]->eval(d).to_number()));
        case Op::Pow:
            return Value::number(
                std::pow(args[0]->eval(d).to_number(), args[1]->eval(d).to_number()));
        case Op::Log2:
            return Value::number(std::log2(args[0]->eval(d).to_number()));
        case Op::Rsqrt:
            return Value::number(1.0 / std::sqrt(args[0]->eval(d).to_number()));
        case Op::ValueOrDefault:
        {
            // First arg is a variable reference; a null result means the path
            // did not resolve in the data source, so fall back to the default.
            Value v = args[0]->eval(d);
            return v.is_null() ? args[1]->eval(d) : v;
        }
        }
        return Value();
    }
};

inline const Op* lookup_op(const std::string& key)
{
    static const std::pair<const char*, Op> table[] = {{"+", Op::Add},
                                                       {"-", Op::Sub},
                                                       {"*", Op::Mul},
                                                       {"/", Op::Div},
                                                       {"%", Op::Mod},
                                                       {"min", Op::Min},
                                                       {"max", Op::Max},
                                                       {"<", Op::Lt},
                                                       {"<=", Op::Le},
                                                       {">", Op::Gt},
                                                       {">=", Op::Ge},
                                                       {"==", Op::Eq},
                                                       {"!=", Op::Neq},
                                                       {"!", Op::Not},
                                                       {"!!", Op::NotNot},
                                                       {"if", Op::If},
                                                       {"?:", Op::If},
                                                       {"and", Op::And},
                                                       {"or", Op::Or},
                                                       {"in", Op::In},
                                                       {"ceil_div", Op::CeilDiv},
                                                       {"abs", Op::Abs},
                                                       {"pow", Op::Pow},
                                                       {"log2", Op::Log2},
                                                       {"rsqrt", Op::Rsqrt},
                                                       {"value_or_default", Op::ValueOrDefault}};
    for(const auto& e : table)
        if(key == e.first)
            return &e.second;
    return nullptr;
}

inline void check_arity(Op op, std::size_t n, const std::string& key)
{
    const auto require = [&](bool ok) {
        if(!ok)
            throw JsonLogicCompileError("operator '" + key
                                        + "' got wrong argument count: " + std::to_string(n));
    };
    switch(op)
    {
    case Op::Add:
    case Op::Mul:
        break; // any arity
    case Op::Sub:
        require(n == 1 || n == 2);
        break;
    case Op::Div:
    case Op::Mod:
    case Op::Eq:
    case Op::Neq:
    case Op::Gt:
    case Op::Ge:
        require(n == 2);
        break;
    case Op::Lt:
    case Op::Le:
        require(n == 2 || n == 3);
        break;
    case Op::Min:
    case Op::Max:
    case Op::And:
    case Op::Or:
        require(n >= 1);
        break;
    case Op::Not:
    case Op::NotNot:
        require(n == 1);
        break;
    case Op::If:
        require(n >= 2);
        break;
    case Op::In:
    case Op::Pow:
    case Op::CeilDiv:
    case Op::ValueOrDefault:
        require(n == 2);
        break;
    case Op::Abs:
    case Op::Log2:
    case Op::Rsqrt:
        require(n == 1);
        break;
    }
}

inline Value json_scalar_to_value(const nlohmann::json& j)
{
    if(j.is_boolean())
        return Value(j.get<bool>());
    if(j.is_number_integer() || j.is_number_unsigned())
        return Value(j.get<std::int64_t>());
    if(j.is_number_float())
        return Value(j.get<double>());
    return Value(); // null
}

template <class DataT>
NodePtr<DataT> compile_node(const nlohmann::json& j, char sigil);

template <class DataT>
NodePtr<DataT> compile_var(const nlohmann::json& val, char sigil)
{
    auto n = std::make_unique<VarNode<DataT>>();
    if(val.is_string())
    {
        n->has_static = true;
        n->path = val.get<std::string>();
    }
    else if(val.is_number())
    {
        n->has_static = true;
        n->path = std::to_string(val.get<std::int64_t>());
    }
    else if(val.is_array())
    {
        const nlohmann::json& p = val.empty() ? nlohmann::json("") : val.at(0);
        if(p.is_string())
        {
            n->has_static = true;
            n->path = p.get<std::string>();
        }
        else if(p.is_number())
        {
            n->has_static = true;
            n->path = std::to_string(p.get<std::int64_t>());
        }
        else
        {
            n->path_expr = compile_node<DataT>(p, sigil);
        }
        if(val.size() > 1)
            n->default_expr = compile_node<DataT>(val.at(1), sigil);
    }
    else if(val.is_object())
    {
        n->path_expr = compile_node<DataT>(val, sigil);
    }
    else
    {
        n->has_static = true; // null / other -> whole document
        n->path.clear();
    }
    return n;
}

template <class DataT>
NodePtr<DataT> compile_object(const nlohmann::json& j, char sigil)
{
    if(j.size() != 1)
        throw JsonLogicCompileError("expression object must have exactly one operator key");
    const auto it = j.begin();
    const std::string& key = it.key();
    const nlohmann::json& val = it.value();
    if(key == "var")
        return compile_var<DataT>(val, sigil);

    const Op* op = lookup_op(key);
    if(op == nullptr)
        throw JsonLogicCompileError("unrecognized operation: " + key);

    auto node = std::make_unique<OpNode<DataT>>();
    node->op = *op;
    if(val.is_array())
    {
        node->args.reserve(val.size());
        for(const auto& e : val)
            node->args.push_back(compile_node<DataT>(e, sigil));
    }
    else
    {
        node->args.push_back(compile_node<DataT>(val, sigil));
    }
    check_arity(*op, node->args.size(), key);
    return node;
}

template <class DataT>
NodePtr<DataT> compile_node(const nlohmann::json& j, char sigil)
{
    if(j.is_object())
        return compile_object<DataT>(j, sigil);
    if(j.is_array())
    {
        auto n = std::make_unique<ArrayNode<DataT>>();
        n->items.reserve(j.size());
        for(const auto& e : j)
            n->items.push_back(compile_node<DataT>(e, sigil));
        return n;
    }
    if(j.is_string())
    {
        const std::string& s = j.get_ref<const nlohmann::json::string_t&>();
        if(s.empty() || s[0] != sigil)
            return std::make_unique<LiteralNode<DataT>>(Value(s));
        if(s.size() >= 2 && s[1] == sigil)
            return std::make_unique<LiteralNode<DataT>>(Value(s.substr(1))); // "$$x" -> "$x"
        auto n = std::make_unique<VarNode<DataT>>();
        n->has_static = true;
        n->path = s.substr(1);
        return n;
    }
    return std::make_unique<LiteralNode<DataT>>(json_scalar_to_value(j));
}

} // namespace detail

// ===========================================================================
// Expression - a compiled, reusable JsonLogic expression
// ===========================================================================
template <class DataT>
class Expression
{
public:
    Expression() = default;
    explicit Expression(detail::NodePtr<DataT> root)
        : root_(std::move(root))
    {
    }

    /// Evaluate against a data source. Cheap: walks the pre-compiled tree.
    Value operator()(const DataT& data) const
    {
        return root_ ? root_->eval(data) : Value();
    }
    Value evaluate(const DataT& data) const
    {
        return (*this)(data);
    }

    explicit operator bool() const
    {
        return static_cast<bool>(root_);
    }

private:
    detail::NodePtr<DataT> root_;
};

/// Compile a JsonLogic rule into a reusable Expression bound to data source
/// type DataT. Throws JsonLogicCompileError on malformed rules.
template <class DataT>
Expression<DataT> compile(const nlohmann::json& rule, char var_sigil = '$')
{
    return Expression<DataT>(detail::compile_node<DataT>(rule, var_sigil));
}

/// Convenience one-shot: compile and evaluate in a single call. Prefer
/// compile() + reuse when evaluating the same rule repeatedly.
template <class DataT>
Value evaluate(const nlohmann::json& rule, const DataT& data, char var_sigil = '$')
{
    return compile<DataT>(rule, var_sigil)(data);
}

} // namespace hip_kernel_provider_common::jsonlogic

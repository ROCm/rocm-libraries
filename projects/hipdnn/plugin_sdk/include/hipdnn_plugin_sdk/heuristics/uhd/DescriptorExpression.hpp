// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace hipdnn_plugin_sdk::uhd::expression
{

class Error : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

class UndefinedVariableError : public Error
{
public:
    using Error::Error;
};

using CategoricalEncoding = std::map<std::string, std::map<std::string, int32_t>>;

/// Bindings are borrowed only for the duration of an evaluation workspace. Do not
/// mutate invariant bindings between prepare() and the last candidate evaluation.
class VariableContext
{
public:
    using ValueType = std::variant<double, int64_t, std::string, bool>;

    void bind(const std::string& name, ValueType value)
    {
        _bindings[name] = std::move(value);
    }

    void bindNamespace(const std::string& ns,
                       const std::unordered_map<std::string, ValueType>& values)
    {
        for(const auto& [name, value] : values)
        {
            bind("$" + ns + "." + name, value);
        }
    }

    void clearNamespace(const std::string& ns)
    {
        const std::string prefix = "$" + ns + ".";
        for(auto it = _bindings.begin(); it != _bindings.end();)
        {
            if(it->first.rfind(prefix, 0) == 0)
            {
                it = _bindings.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    const ValueType* getData(const std::string& name) const
    {
        const auto it = _bindings.find(name);
        return it == _bindings.end() ? nullptr : &it->second;
    }

    std::optional<ValueType> resolve(const std::string& name) const
    {
        const auto* value = getData(name);
        return value == nullptr ? std::nullopt : std::optional<ValueType>(*value);
    }

    std::optional<double> resolveDouble(const std::string& name) const
    {
        const auto* value = getData(name);
        if(value == nullptr)
        {
            return std::nullopt;
        }
        return std::visit(
            [](const auto& held) -> double {
                using T = std::decay_t<decltype(held)>;
                if constexpr(std::is_same_v<T, std::string>)
                {
                    throw Error("Type error: string used where a number is required");
                }
                else
                {
                    return static_cast<double>(held);
                }
            },
            *value);
    }

    bool has(const std::string& name) const
    {
        return getData(name) != nullptr;
    }
    void clear()
    {
        _bindings.clear();
    }

    /// @brief Borrow the bindings for publishing a feature contract.
    const std::unordered_map<std::string, ValueType>& bindings() const
    {
        return _bindings;
    }

private:
    std::unordered_map<std::string, ValueType> _bindings;
};

/// A typed, bounded DAG for all descriptor expression consumers. JSON and operator
/// names are used only while compiling. Repeated nodes share an index, including
/// repetitions between output expressions. Per-selection workspaces keep evaluation
/// thread-safe and retain invariant subexpressions between candidates.
class Program
{
public:
    static constexpr size_t MAX_EXPRESSION_DEPTH = 64;
    static constexpr size_t MAX_NODES = 16384;
    static constexpr size_t MAX_INPUT_NODES = 65536;
    using Vocabulary = std::map<std::string, int32_t, std::less<>>;

    struct Value
    {
        std::variant<double, int64_t, bool, std::string_view> raw = 0.0;
        const Vocabulary* encoding = nullptr;
    };

    struct Workspace
    {
        std::vector<Value> nodes;
        std::vector<uint8_t> ready;
        std::vector<std::exception_ptr> errors;
        const Program* owner = nullptr;
    };

    explicit Program(const std::vector<nlohmann::json>& expressions,
                     const CategoricalEncoding& encoding = {})
    {
        for(const auto& [reference, vocabulary] : encoding)
        {
            _encoding.emplace(reference, Vocabulary(vocabulary.begin(), vocabulary.end()));
        }
        if(expressions.size() > MAX_NODES)
        {
            throw Error("Too many descriptor expressions");
        }
        std::unordered_map<std::string, size_t> intern;
        size_t visited = 0;
        for(const auto& expr : expressions)
        {
            _roots.push_back(compile(expr, 0, std::nullopt, intern, visited));
        }
        for(size_t i = 0; i < _nodes.size(); ++i)
        {
            (_nodes[i].kernel ? _candidateNodes : _sharedNodes).push_back(i);
        }
    }

    Workspace workspace() const
    {
        return {std::vector<Value>(_nodes.size()),
                std::vector<uint8_t>(_nodes.size(), 0),
                std::vector<std::exception_ptr>(_nodes.size()),
                this};
    }

    /// Evaluate invariant DAG nodes once, retaining errors rather than raising them
    /// until a selected branch actually reads that node. This preserves lazy if/and/
    /// or/default semantics even for an invariant inside a candidate-dependent branch.
    template <class DataT>
    void prepare(const DataT& data, Workspace& work) const
    {
        checkWorkspace(work);
        std::fill(work.ready.begin(), work.ready.end(), 0);
        std::fill(work.errors.begin(), work.errors.end(), nullptr);
        for(const auto i : _sharedNodes)
        {
            try
            {
                (void)evaluateNode(i, data, work);
            }
            catch(const Error&)
            {
                // An unreachable branch must not fail the whole expression.
            }
        }
    }

    void resetCandidate(Workspace& work) const
    {
        checkWorkspace(work);
        for(const auto i : _candidateNodes)
        {
            work.ready[i] = 0;
            work.errors[i] = nullptr;
        }
    }

    template <class DataT>
    const Value& evaluate(size_t output, const DataT& data, Workspace& work) const
    {
        checkWorkspace(work);
        if(output >= _roots.size())
        {
            throw Error("Expression output index is out of range");
        }
        return evaluateNode(_roots[output], data, work);
    }

    bool kernelDependent(size_t output) const
    {
        return _nodes.at(_roots.at(output)).kernel;
    }
    size_t size() const
    {
        return _roots.size();
    }
    size_t nodeCount() const
    {
        return _nodes.size();
    }
    size_t sharedNodeCount() const
    {
        return _sharedNodes.size();
    }
    size_t candidateNodeCount() const
    {
        return _candidateNodes.size();
    }
    const std::unordered_set<std::string>& variables() const
    {
        return _variables;
    }

    static double number(const Value& value)
    {
        if(const auto* text = std::get_if<std::string_view>(&value.raw))
        {
            if(value.encoding != nullptr)
            {
                const auto code = value.encoding->find(*text);
                if(code != value.encoding->end() && code->first == *text)
                {
                    return static_cast<double>(code->second);
                }
                throw Error("Categorical value has no code in categorical_encoding");
            }
            throw Error("Type error: string used where a number is required");
        }
        const double result = std::visit(
            [](const auto& held) -> double {
                using T = std::decay_t<decltype(held)>;
                if constexpr(std::is_same_v<T, std::string_view>)
                {
                    return 0.0;
                }
                else
                {
                    return static_cast<double>(held);
                }
            },
            value.raw);
        if(!std::isfinite(result))
        {
            throw Error("Non-finite expression value");
        }
        return result;
    }

    static bool boolean(const Value& value)
    {
        if(const auto* text = std::get_if<std::string_view>(&value.raw))
        {
            return !text->empty();
        }
        return number(value) != 0.0;
    }

private:
    enum class Op : uint8_t
    {
        NUMBER,
        INTEGER,
        BOOLEAN,
        STRING,
        REFERENCE,
        ADD,
        SUBTRACT,
        MULTIPLY,
        DIVIDE,
        MODULO,
        CEIL_DIV,
        MIN,
        MAX,
        ABS,
        POW,
        LOG2,
        RSQRT,
        EQUAL,
        NOT_EQUAL,
        LESS,
        LESS_EQUAL,
        GREATER,
        GREATER_EQUAL,
        AND,
        OR,
        NOT,
        IF,
        DEFAULT,
        MEMBER_OF,
        DIVISIBLE,
        PRESENT,
        NOT_PRESENT
    };

    struct Node
    {
        Op op = Op::NUMBER;
        Value literal;
        std::string text;
        std::vector<size_t> args;
        bool kernel = false;
        uint8_t depth = 0;
    };

    static bool reference(const nlohmann::json& expr)
    {
        return expr.is_string() && !expr.get_ref<const std::string&>().empty()
               && expr.get_ref<const std::string&>().front() == '$';
    }

    size_t compile(const nlohmann::json& expr,
                   size_t depth,
                   std::optional<size_t> current,
                   std::unordered_map<std::string, size_t>& intern,
                   size_t& visited)
    {
        if(depth > MAX_EXPRESSION_DEPTH || ++visited > MAX_INPUT_NODES)
        {
            throw Error("Descriptor expression exceeds the depth or input-size bound");
        }
        Node node;
        if(expr.is_boolean())
        {
            node.op = Op::BOOLEAN;
            node.literal.raw = expr.get<bool>();
        }
        else if(expr.is_number_integer())
        {
            if(expr.is_number_unsigned()
               && expr.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            {
                throw Error("Integer literal exceeds signed 64-bit range");
            }
            node.op = Op::INTEGER;
            node.literal.raw = expr.get<int64_t>();
        }
        else if(expr.is_number_float())
        {
            node.literal.raw = expr.get<double>();
            (void)number(node.literal);
        }
        else if(expr.is_string())
        {
            node.text = expr.get<std::string>();
            if(node.text.size() > 65536)
            {
                throw Error("Expression string exceeds size bound");
            }
            node.op = reference(expr) ? Op::REFERENCE : Op::STRING;
            if(node.op == Op::REFERENCE)
            {
                if(node.text == "$current")
                {
                    if(!current)
                    {
                        throw Error("$current is only valid inside all");
                    }
                    return *current;
                }
                if(node.text == "$")
                {
                    throw Error("Invalid descriptor reference: " + node.text);
                }
                _variables.insert(node.text);
                node.kernel = node.text == "$kernel" || node.text.rfind("$kernel.", 0) == 0;
            }
        }
        else if(expr.is_object() && expr.size() == 1)
        {
            const auto& name = expr.begin().key();
            const auto& body = expr.begin().value();
            const auto& args = body;
            const size_t argumentCount = body.is_array() ? body.size() : 1;
            const auto arity = [&](size_t minimum, size_t maximum) {
                if(argumentCount < minimum || argumentCount > maximum)
                {
                    throw Error("Invalid argument count for " + name);
                }
            };
            if(name == "shape" || name == "rank")
            {
                arity(name == "shape" ? 2 : 1, name == "shape" ? 2 : 1);
                const auto& tensor = body.is_array() ? body[0] : body;
                if(!reference(tensor))
                {
                    throw Error(name + " requires a tensor reference");
                }
                std::string path = tensor.get<std::string>();
                if(name == "shape")
                {
                    if(!args[1].is_number_integer() || args[1].get<int64_t>() < 0
                       || args[1].get<int64_t>() > 1024)
                    {
                        throw Error("shape requires a constant nonnegative dimension index");
                    }
                    path += ".dims[" + std::to_string(args[1].get<int64_t>()) + "]";
                }
                else
                {
                    path += ".rank";
                }
                return compile(nlohmann::json(path), depth + 1, current, intern, visited);
            }
            if(name == "all")
            {
                arity(2, 2);
                if(!args[0].is_array())
                {
                    throw Error("all requires an array");
                }
                node.op = Op::AND;
                if(args[0].empty())
                {
                    const auto placeholder
                        = compile(nlohmann::json(0), depth + 1, current, intern, visited);
                    (void)compile(args[1], depth + 1, placeholder, intern, visited);
                }
                for(const auto& item : args[0])
                {
                    const auto itemId = compile(item, depth + 1, current, intern, visited);
                    node.args.push_back(compile(args[1], depth + 1, itemId, intern, visited));
                }
            }
            else
            {
                static const std::unordered_map<std::string, Op> operators
                    = {{"+", Op::ADD},
                       {"-", Op::SUBTRACT},
                       {"*", Op::MULTIPLY},
                       {"/", Op::DIVIDE},
                       {"%", Op::MODULO},
                       {"ceil_div", Op::CEIL_DIV},
                       {"min", Op::MIN},
                       {"max", Op::MAX},
                       {"abs", Op::ABS},
                       {"pow", Op::POW},
                       {"log2", Op::LOG2},
                       {"rsqrt", Op::RSQRT},
                       {"==", Op::EQUAL},
                       {"!=", Op::NOT_EQUAL},
                       {"<", Op::LESS},
                       {"<=", Op::LESS_EQUAL},
                       {">", Op::GREATER},
                       {">=", Op::GREATER_EQUAL},
                       {"and", Op::AND},
                       {"or", Op::OR},
                       {"!", Op::NOT},
                       {"if", Op::IF},
                       {"value_or_default", Op::DEFAULT},
                       {"in", Op::MEMBER_OF},
                       {"divisible", Op::DIVISIBLE},
                       {"present", Op::PRESENT},
                       {"not_present", Op::NOT_PRESENT}};
                const auto found = operators.find(name);
                if(found == operators.end())
                {
                    throw Error("Unknown operator: " + name);
                }
                node.op = found->second;
                switch(node.op)
                {
                case Op::ADD:
                case Op::SUBTRACT:
                case Op::MULTIPLY:
                case Op::AND:
                case Op::OR:
                    arity(0, MAX_NODES);
                    break;
                case Op::DIVIDE:
                case Op::IF:
                    arity(2, MAX_NODES);
                    break;
                case Op::MIN:
                case Op::MAX:
                    arity(1, MAX_NODES);
                    break;
                case Op::ABS:
                case Op::LOG2:
                case Op::RSQRT:
                case Op::NOT:
                case Op::PRESENT:
                case Op::NOT_PRESENT:
                    arity(1, 1);
                    break;
                default:
                    arity(2, 2);
                    break;
                }
                if(node.op == Op::MEMBER_OF)
                {
                    if(!args[1].is_array())
                    {
                        throw Error("in requires an array as second argument");
                    }
                    node.args.push_back(compile(args[0], depth + 1, current, intern, visited));
                    for(const auto& item : args[1])
                    {
                        node.args.push_back(compile(item, depth + 1, current, intern, visited));
                    }
                }
                else if(body.is_array())
                {
                    for(const auto& arg : args)
                    {
                        node.args.push_back(compile(arg, depth + 1, current, intern, visited));
                    }
                }
                else
                {
                    node.args.push_back(compile(body, depth + 1, current, intern, visited));
                }
            }
        }
        else
        {
            throw Error("Unsupported descriptor expression type");
        }

        // Intern typed structure, not authored whitespace or object ordering. Length
        // prefixes and fixed separators keep literal text distinct from child indices.
        std::string key = std::to_string(static_cast<unsigned>(node.op)) + ":";
        if(node.op == Op::STRING || node.op == Op::REFERENCE)
        {
            key += std::to_string(node.text.size()) + ":" + node.text;
        }
        else if(node.args.empty()
                && (node.op == Op::NUMBER || node.op == Op::INTEGER || node.op == Op::BOOLEAN))
        {
            key += expr.dump();
        }
        for(const auto child : node.args)
        {
            key += ":" + std::to_string(child);
            node.kernel = node.kernel || _nodes[child].kernel;
            if(_nodes[child].depth >= MAX_EXPRESSION_DEPTH)
            {
                throw Error("Lowered descriptor expression exceeds depth bound");
            }
            node.depth = std::max(node.depth, static_cast<uint8_t>(_nodes[child].depth + 1));
        }
        if(const auto it = intern.find(key); it != intern.end())
        {
            return it->second;
        }
        if(_nodes.size() >= MAX_NODES)
        {
            throw Error("Compiled expression exceeds node bound");
        }
        const auto id = _nodes.size();
        _nodes.push_back(std::move(node));
        intern.emplace(std::move(key), id);
        return id;
    }

    static bool equal(const Value& a, const Value& b)
    {
        const auto* textA = std::get_if<std::string_view>(&a.raw);
        const auto* textB = std::get_if<std::string_view>(&b.raw);
        if(textA || textB)
        {
            if(!textA || !textB)
            {
                throw Error("Type error: cannot compare a string against a number");
            }
            return *textA == *textB;
        }
        if(const auto* intA = std::get_if<int64_t>(&a.raw))
        {
            if(const auto* intB = std::get_if<int64_t>(&b.raw))
            {
                return *intA == *intB;
            }
        }
        return number(a) == number(b);
    }

    static bool less(const Value& a, const Value& b)
    {
        if(const auto* x = std::get_if<int64_t>(&a.raw))
        {
            if(const auto* y = std::get_if<int64_t>(&b.raw))
            {
                return *x < *y;
            }
        }
        return number(a) < number(b);
    }

    static Value arithmetic(Op op, const Value& a, const Value& b)
    {
        const auto* x = std::get_if<int64_t>(&a.raw);
        const auto* y = std::get_if<int64_t>(&b.raw);
        if(x && y && (op == Op::ADD || op == Op::SUBTRACT || op == Op::MULTIPLY))
        {
            constexpr auto LO = std::numeric_limits<int64_t>::min();
            constexpr auto HI = std::numeric_limits<int64_t>::max();
            bool overflow = false;
            if(op == Op::ADD)
            {
                overflow = (*y > 0 && *x > HI - *y) || (*y < 0 && *x < LO - *y);
            }
            else if(op == Op::SUBTRACT)
            {
                overflow = (*y < 0 && *x > HI + *y) || (*y > 0 && *x < LO + *y);
            }
            else if(*x != 0 && *y != 0)
            {
                overflow = *x > 0 ? (*y > 0 ? *x > HI / *y : *y < LO / *x)
                                  : (*y > 0 ? *x < LO / *y : *x < HI / *y);
            }
            if(overflow)
            {
                throw Error("Signed 64-bit integer expression overflow");
            }
            return {op == Op::ADD ? *x + *y : op == Op::SUBTRACT ? *x - *y : *x * *y};
        }
        const double lhs = number(a);
        const double rhs = number(b);
        switch(op)
        {
        case Op::ADD:
            return {lhs + rhs};
        case Op::SUBTRACT:
            return {lhs - rhs};
        case Op::MULTIPLY:
            return {lhs * rhs};
        default:
            throw Error("Invalid arithmetic opcode");
        }
    }

    void checkWorkspace(const Workspace& work) const
    {
        if(work.owner != this || work.nodes.size() != _nodes.size())
        {
            throw Error("Expression workspace belongs to a different program");
        }
    }

    template <class DataT>
    const Value& evaluateNode(size_t id, const DataT& data, Workspace& work) const
    {
        if(!work.ready[id])
        {
            try
            {
                work.nodes[id] = execute(_nodes[id], data, work);
                if(!std::holds_alternative<std::string_view>(work.nodes[id].raw))
                {
                    (void)number(work.nodes[id]);
                }
                work.ready[id] = 1;
            }
            catch(const Error&)
            {
                work.errors[id] = std::current_exception();
                work.ready[id] = 1;
            }
        }
        if(work.errors[id])
        {
            std::rethrow_exception(work.errors[id]);
        }
        return work.nodes[id];
    }

    template <class DataT>
    Value execute(const Node& node, const DataT& data, Workspace& work) const
    {
        const auto arg
            = [&](size_t i) -> const Value& { return evaluateNode(node.args[i], data, work); };
        const auto num = [&](size_t i) { return number(arg(i)); };
        switch(node.op)
        {
        case Op::NUMBER:
        case Op::INTEGER:
        case Op::BOOLEAN:
            return node.literal;
        case Op::STRING:
            return {std::string_view(node.text)};
        case Op::REFERENCE:
        {
            const auto* raw = data.getData(node.text);
            if(!raw)
            {
                throw UndefinedVariableError("Undefined variable: " + node.text);
            }
            Value value = std::visit(
                [](const auto& held) -> Value {
                    using T = std::decay_t<decltype(held)>;
                    if constexpr(std::is_same_v<T, std::string>)
                    {
                        return {std::string_view(held)};
                    }
                    else
                    {
                        return {held};
                    }
                },
                *raw);
            if(const auto found = _encoding.find(node.text); found != _encoding.end())
            {
                value.encoding = &found->second;
            }
            return value;
        }
        case Op::AND:
            for(size_t i = 0; i < node.args.size(); ++i)
            {
                if(!boolean(arg(i)))
                {
                    return {false};
                }
            }
            return {true};
        case Op::OR:
            for(size_t i = 0; i < node.args.size(); ++i)
            {
                if(boolean(arg(i)))
                {
                    return {true};
                }
            }
            return {false};
        case Op::NOT:
            return {!boolean(arg(0))};
        case Op::IF:
            for(size_t i = 0; i + 1 < node.args.size(); i += 2)
            {
                if(boolean(arg(i)))
                {
                    return arg(i + 1);
                }
            }
            return node.args.size() % 2 ? arg(node.args.size() - 1) : Value{false};
        case Op::DEFAULT:
            try
            {
                return arg(0);
            }
            catch(const UndefinedVariableError&)
            {
                return arg(1);
            }
        case Op::PRESENT:
        case Op::NOT_PRESENT:
            try
            {
                (void)arg(0);
                return {node.op == Op::PRESENT};
            }
            catch(const UndefinedVariableError&)
            {
                return {node.op == Op::NOT_PRESENT};
            }
        case Op::MEMBER_OF:
        {
            const auto& needle = arg(0);
            for(size_t i = 1; i < node.args.size(); ++i)
            {
                if(equal(needle, arg(i)))
                {
                    return {true};
                }
            }
            return {false};
        }
        case Op::EQUAL:
            return {equal(arg(0), arg(1))};
        case Op::NOT_EQUAL:
            return {!equal(arg(0), arg(1))};
        case Op::LESS:
            return {less(arg(0), arg(1))};
        case Op::LESS_EQUAL:
            return {!less(arg(1), arg(0))};
        case Op::GREATER:
            return {less(arg(1), arg(0))};
        case Op::GREATER_EQUAL:
            return {!less(arg(0), arg(1))};
        case Op::ADD:
        case Op::SUBTRACT:
        case Op::MULTIPLY:
        {
            if(node.args.empty())
            {
                return {int64_t(node.op == Op::MULTIPLY ? 1 : 0)};
            }
            Value result = arg(0);
            if(node.op == Op::SUBTRACT && node.args.size() == 1)
            {
                return arithmetic(Op::SUBTRACT, Value{int64_t{0}}, result);
            }
            // Even one-operand arithmetic is a numeric context.
            (void)number(result);
            for(size_t i = 1; i < node.args.size(); ++i)
            {
                result = arithmetic(node.op, result, arg(i));
            }
            return std::holds_alternative<std::string_view>(result.raw) ? Value{number(result)}
                                                                        : result;
        }
        case Op::DIVIDE:
        {
            double result = num(0);
            for(size_t i = 1; i < node.args.size(); ++i)
            {
                const double divisor = num(i);
                if(divisor == 0)
                {
                    throw Error("Division by zero");
                }
                result /= divisor;
            }
            return {result};
        }
        case Op::MODULO:
        case Op::CEIL_DIV:
        case Op::DIVISIBLE:
        {
            const auto& a = arg(0);
            const auto& b = arg(1);
            if(num(1) == 0)
            {
                throw Error("Division by zero");
            }
            if(const auto* x = std::get_if<int64_t>(&a.raw))
            {
                if(const auto* y = std::get_if<int64_t>(&b.raw))
                {
                    if(*x == std::numeric_limits<int64_t>::min() && *y == -1)
                    {
                        if(node.op == Op::CEIL_DIV)
                        {
                            throw Error("Signed 64-bit integer expression overflow");
                        }
                        return node.op == Op::DIVISIBLE ? Value{true} : Value{int64_t{0}};
                    }
                    const int64_t remainder = *x % *y;
                    if(node.op == Op::DIVISIBLE)
                    {
                        return {remainder == 0};
                    }
                    if(node.op == Op::MODULO)
                    {
                        return {remainder};
                    }
                    return {*x / *y + (remainder != 0 && ((remainder > 0) == (*y > 0)) ? 1 : 0)};
                }
            }
            if(node.op == Op::DIVISIBLE)
            {
                return {std::fmod(num(0), num(1)) == 0};
            }
            return {node.op == Op::MODULO ? std::fmod(num(0), num(1)) : std::ceil(num(0) / num(1))};
        }
        case Op::MIN:
        case Op::MAX:
        {
            Value result = arg(0);
            (void)number(result);
            for(size_t i = 1; i < node.args.size(); ++i)
            {
                const auto& next = arg(i);
                if(node.op == Op::MIN ? less(next, result) : less(result, next))
                {
                    result = next;
                }
            }
            return std::holds_alternative<std::string_view>(result.raw) ? Value{number(result)}
                                                                        : result;
        }
        case Op::ABS:
            if(const auto* integer = std::get_if<int64_t>(&arg(0).raw))
            {
                if(*integer == std::numeric_limits<int64_t>::min())
                {
                    throw Error("Signed 64-bit integer expression overflow");
                }
                return {*integer < 0 ? -*integer : *integer};
            }
            return {std::abs(num(0))};
        case Op::POW:
            return {std::pow(num(0), num(1))};
        case Op::LOG2:
        case Op::RSQRT:
            if(num(0) <= 0)
            {
                throw Error("log2/rsqrt requires a positive number");
            }
            return {node.op == Op::LOG2 ? std::log2(num(0)) : 1.0 / std::sqrt(num(0))};
        default:
            throw Error("Invalid compiled expression opcode");
        }
    }

    std::map<std::string, Vocabulary> _encoding;
    std::vector<Node> _nodes;
    std::vector<size_t> _roots;
    std::vector<size_t> _sharedNodes;
    std::vector<size_t> _candidateNodes;
    std::unordered_set<std::string> _variables;
};

} // namespace hipdnn_plugin_sdk::uhd::expression

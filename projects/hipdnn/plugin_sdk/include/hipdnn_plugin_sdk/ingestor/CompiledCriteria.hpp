// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/BindingPublication.hpp>
#include <hipdnn_plugin_sdk/ingestor/CompiledGraphPattern.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorJsonRules.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/JsonExpression.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

/// @file CompiledCriteria.hpp
/// @brief A UMD's declarative `criteria` (RFC 0018 §3, A.1), compiled with the JSON
///        Expression Language and evaluated over the symbol table the engine's
///        `graph_match` publishes (RFC 0020 §6.1).
///
/// Three pieces, in the order a criterion meets them:
///  - compileCriteria() runs when the UMD is read: the expression compiles, and its
///    `$kernel.*` reads agree with the declared scope (RFC 0018 A.5 check 3).
///  - criteriaReferenceError() runs when a pack pairs the UMD with an engine: every
///    reference resolves in what that engine publishes (A.5 checks 5 and 6).
///  - CompiledCriteria::passes() runs per graph, or per candidate kernel, over a
///    CriteriaSymbols data source.

namespace hipdnn_plugin_sdk::ingestor
{

/// The reserved roots the engine's pattern never binds (RFC 0020 §6.1).
inline constexpr std::string_view KERNEL_SYMBOL_ROOT = "kernel";
inline constexpr std::string_view DEVICE_SYMBOL_ROOT = "device";
inline constexpr std::string_view GRAPH_SYMBOL_ROOT = "graph";

namespace detail
{

/// The root of a sigil-free reference path: the text before the first `.` or `[`.
inline std::string_view symbolRoot(std::string_view path)
{
    return path.substr(0, std::min(path.find_first_of(".["), path.size()));
}

/// The field after `<root>.`, or nullopt when @p path does not start with that root.
inline std::optional<std::string_view> symbolField(std::string_view path, std::string_view root)
{
    if(path.size() <= root.size() + 1 || path.compare(0, root.size(), root) != 0
       || path[root.size()] != '.')
    {
        return std::nullopt;
    }
    return path.substr(root.size() + 1);
}

inline jsonexpr::Value toExpressionValue(const MetadataValue& value)
{
    return std::visit(
        [](const auto& held) -> jsonexpr::Value {
            using T = std::decay_t<decltype(held)>;
            if constexpr(std::is_same_v<T, std::vector<int64_t>>)
            {
                jsonexpr::Value::Array items;
                items.reserve(held.size());
                for(const auto item : held)
                {
                    items.emplace_back(item);
                }
                return {std::move(items)};
            }
            else
            {
                return jsonexpr::Value(held);
            }
        },
        value);
}

/// True for a JSON string the language reads as a reference: one sigil, then a path.
inline bool isReference(const nlohmann::json& node)
{
    if(!node.is_string())
    {
        return false;
    }
    const auto& text = node.get_ref<const nlohmann::json::string_t&>();
    return text.size() > 1 && text[0] == jsonexpr::VARIABLE_SIGIL
           && text[1] != jsonexpr::VARIABLE_SIGIL;
}

/// Adds the reference path of every direct `present` / `not_present` operand in @p node
/// to @p queries. Only a bare reference names something whose presence the pattern can
/// fix; an operand computed from one is left to evaluation.
inline void collectPresenceQueries(const nlohmann::json& node, std::set<std::string>& queries)
{
    if(node.is_array())
    {
        for(const auto& item : node)
        {
            collectPresenceQueries(item, queries);
        }
        return;
    }
    if(!node.is_object())
    {
        return;
    }
    for(auto it = node.begin(); it != node.end(); ++it)
    {
        const auto& operands = it.value();
        if(it.key() == "present" || it.key() == "not_present")
        {
            const auto record = [&queries](const nlohmann::json& operand) {
                if(isReference(operand))
                {
                    queries.insert(operand.get<std::string>().substr(1));
                }
            };
            if(operands.is_array())
            {
                std::for_each(operands.begin(), operands.end(), record);
            }
            else
            {
                record(operands);
            }
        }
        collectPresenceQueries(operands, queries);
    }
}

} // namespace detail

/// The `$device.*` values for one device (RFC 0020 §6.1), resolved once per catalog build
/// from the vocabulary deviceFeatureValues() defines, so criteria and UHD features read
/// one set of names.
class CriteriaDevice
{
public:
    explicit CriteriaDevice(const DeviceProperties& properties)
        : _values(deviceFeatureValues(properties))
    {
    }

    /// The value of @p field, or null when the vocabulary has no such field or the device
    /// did not report it. An unresolved device fact reads 0, and no real device reports a
    /// zero for any of them, so a zero declines a bound check rather than answering it.
    jsonexpr::Value find(std::string_view field) const
    {
        for(const auto& [name, value] : _values)
        {
            if(name == field)
            {
                return std::visit(
                    [](auto held) { return held == 0 ? jsonexpr::Value() : jsonexpr::Value(held); },
                    value);
            }
        }
        return {};
    }

    /// True when @p field names a device property, whatever this device reports for it.
    static bool declares(std::string_view field)
    {
        const auto names = deviceFeatureValues(DeviceProperties{});
        return std::any_of(names.begin(), names.end(), [field](const auto& entry) {
            return entry.first == field;
        });
    }

private:
    std::vector<std::pair<std::string, std::variant<std::int64_t, double>>> _values;
};

/// The symbol table one criteria evaluation reads: the engine's binding, the device, and,
/// for a kernel-scoped criterion, the candidate kernel's metadata as `$kernel.*`. A
/// graph-scoped evaluation passes no kernel, so a `$kernel.*` read is null there. Holds
/// references; build one on the stack per evaluation.
///
/// Every binding value is a scalar or a flat integer list, so a returned Value is far
/// inside jsonexpr's MAX_VALUE_DEPTH.
class CriteriaSymbols
{
public:
    CriteriaSymbols(const BoundTokens& bound,
                    const CriteriaDevice& device,
                    const KernelDefinition* kernel = nullptr)
        : _bound(bound)
        , _device(device)
        , _kernel(kernel)
    {
    }

    /// The jsonexpr data-source contract: @p path arrives without the sigil.
    jsonexpr::Value getData(const std::string& path) const
    {
        if(const auto field = detail::symbolField(path, KERNEL_SYMBOL_ROOT))
        {
            return _kernel == nullptr ? jsonexpr::Value() : kernelValue(*field);
        }
        if(const auto field = detail::symbolField(path, DEVICE_SYMBOL_ROOT))
        {
            return _device.find(*field);
        }
        const auto it = _bound.find(path);
        return it == _bound.end() ? jsonexpr::Value() : detail::toExpressionValue(it->second);
    }

private:
    /// A KMD field by name, or one element of an INT_LIST field as `name[i]`.
    jsonexpr::Value kernelValue(std::string_view field) const
    {
        const auto& metadata = _kernel->metadata;
        if(const auto it = metadata.find(std::string(field)); it != metadata.end())
        {
            return detail::toExpressionValue(it->second);
        }
        const auto indexed = detail::splitIndexedField(field);
        if(!indexed.has_value())
        {
            return {};
        }
        const auto it = metadata.find(std::string(indexed->first));
        if(it == metadata.end())
        {
            return {};
        }
        const auto* list = std::get_if<std::vector<int64_t>>(&it->second);
        if(list == nullptr || indexed->second >= list->size())
        {
            return {};
        }
        return {(*list)[indexed->second]};
    }

    const BoundTokens& _bound;
    const CriteriaDevice& _device;
    const KernelDefinition* _kernel;
};

class CompiledCriteria;

/// Compiles @p criteria, the value of a UMD's `criteria` key, for a matcher of @p scope.
/// @p where names the key and the file in every error.
/// @throws HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE) when the root is a
///         literal, the expression does not compile, or a graph-scoped criterion reads
///         `$kernel.*`.
inline std::shared_ptr<const CompiledCriteria>
    compileCriteria(const nlohmann::json& criteria, MatchScope scope, const std::string& where);

/// A compiled `criteria` expression. Immutable, so threads can evaluate one instance at
/// once.
class CompiledCriteria
{
    struct ConstructionKey
    {
        explicit ConstructionKey() = default;
    };

public:
    CompiledCriteria(ConstructionKey /*key*/,
                     nlohmann::json source,
                     jsonexpr::Expression<CriteriaSymbols> expression,
                     std::vector<std::string> references,
                     std::vector<std::string> presenceQueries)
        : _source(std::move(source))
        , _expression(std::move(expression))
        , _references(std::move(references))
        , _presenceQueries(std::move(presenceQueries))
    {
    }

    /// True only when the criteria evaluate to boolean `true`. An unknown result declines
    /// (RFC 0018 A.2), and so does any value that is not a boolean: a criterion is
    /// Bool-rooted, and reading a number's truthiness would admit a graph on a value the
    /// author never compared.
    bool passes(const CriteriaSymbols& symbols) const
    {
        const auto result = _expression(symbols);
        return result.isBool() && result.asBool();
    }

    /// The expression as authored.
    const nlohmann::json& source() const noexcept
    {
        return _source;
    }

    /// Every reference the expression reads, without the sigil. Sorted and distinct.
    const std::vector<std::string>& references() const noexcept
    {
        return _references;
    }

    /// The references a `present` / `not_present` operand names directly. Sorted and
    /// distinct.
    const std::vector<std::string>& presenceQueries() const noexcept
    {
        return _presenceQueries;
    }

private:
    friend std::shared_ptr<const CompiledCriteria>
        compileCriteria(const nlohmann::json& criteria, MatchScope scope, const std::string& where);

    nlohmann::json _source;
    jsonexpr::Expression<CriteriaSymbols> _expression;
    std::vector<std::string> _references;
    std::vector<std::string> _presenceQueries;
};

inline std::shared_ptr<const CompiledCriteria>
    compileCriteria(const nlohmann::json& criteria, MatchScope scope, const std::string& where)
{
    // A literal root states no check: `true` admits every graph and `false` none, and
    // neither is what a matcher is for.
    if(!criteria.is_object() && !detail::isReference(criteria))
    {
        detail::fail(where + " must be an expression object or a '$' reference, not a literal");
    }

    jsonexpr::Expression<CriteriaSymbols> expression;
    try
    {
        expression = jsonexpr::compile<CriteriaSymbols>(criteria);
    }
    catch(const jsonexpr::JsonExpressionCompileError& error)
    {
        detail::fail(where + " does not compile: " + error.what());
    }

    const std::set<std::string> distinct(expression.variables().begin(),
                                         expression.variables().end());
    const auto kernelRead
        = std::find_if(distinct.begin(), distinct.end(), [](const std::string& path) {
              return detail::symbolRoot(path) == KERNEL_SYMBOL_ROOT;
          });
    // RFC 0018 A.5 check 3. A graph-scoped verdict is computed once and prunes the whole
    // pack, so it cannot depend on one candidate's metadata.
    if(scope == MatchScope::GRAPH && kernelRead != distinct.end())
    {
        detail::fail(where + " reads $" + *kernelRead
                     + ", but the matcher is graph-scoped; a kernel field needs scope \"kernel\"");
    }
    // The harmless converse: the criteria pay per-candidate evaluation for a verdict that
    // cannot vary by candidate.
    if(scope == MatchScope::KERNEL && kernelRead == distinct.end())
    {
        HIPDNN_PLUGIN_LOG_WARN("descriptor loader: "
                               << where
                               << " reads no $kernel field, but the matcher is kernel-scoped; "
                                  "scope \"graph\" evaluates it once per graph instead");
    }

    std::set<std::string> presence;
    detail::collectPresenceQueries(criteria, presence);

    return std::make_shared<const CompiledCriteria>(
        CompiledCriteria::ConstructionKey{},
        criteria,
        std::move(expression),
        std::vector<std::string>(distinct.begin(), distinct.end()),
        std::vector<std::string>(presence.begin(), presence.end()));
}

namespace detail
{

/// What an engine publishes, for checking a reference a criteria or launch formula reads
/// against it when a pack pairs the descriptor with that engine. Holds references; build
/// one on the stack per check.
class PublishedSymbols
{
public:
    PublishedSymbols(const GraphMatch& graphMatch, const MetadataSchema& schema)
        : _schema(schema)
        , _native(std::holds_alternative<NativeGraphMatch>(graphMatch))
    {
        const auto* pattern = std::get_if<std::shared_ptr<const CompiledGraphPattern>>(&graphMatch);
        _pattern = pattern == nullptr ? nullptr : pattern->get();
    }

    /// The engine's `nodes` pattern, or null for a native or absent `graph_match`.
    const CompiledGraphPattern* pattern() const noexcept
    {
        return _pattern;
    }

    /// True for a native `graph_match`, whose published set is unknown before a graph
    /// arrives.
    bool native() const noexcept
    {
        return _native;
    }

    /// The pattern variable named @p name, or null.
    const PatternVariable* findVariable(std::string_view name) const
    {
        if(_pattern == nullptr)
        {
            return nullptr;
        }
        const auto& variables = _pattern->variables();
        const auto it = std::find_if(variables.begin(),
                                     variables.end(),
                                     [name](const PatternVariable& v) { return v.name == name; });
        return it == variables.end() ? nullptr : &*it;
    }

    /// The node attribute the pattern binds to @p token, or null.
    const CompiledPatternAttribute* findAttribute(std::string_view token) const
    {
        if(_pattern == nullptr)
        {
            return nullptr;
        }
        for(const auto& node : _pattern->nodes())
        {
            for(const auto& attribute : node.attributes)
            {
                if(attribute.token == token)
                {
                    return &attribute;
                }
            }
        }
        return nullptr;
    }

    /// Why the sigil-free reference @p path does not resolve, or "" when it does. The
    /// reason continues "$<path>", as in "read $<path>" + reason (RFC 0018 A.5 checks 5
    /// and 6):
    ///  - `$kernel.*` must name a field the schema declares, or `name[i]` of an INT_LIST
    ///    field, and `$device.*` must name a device property. These run on every arm.
    ///  - Every other reference must be something the engine publishes: a pattern
    ///    variable or one of its fields, a node attribute, or `$graph.*`. A `nodes`
    ///    pattern is the only arm whose published set is known before a graph arrives,
    ///    so a native `graph_match` skips this (RFC 0020 §4.5), and an engine without
    ///    `graph_match` publishes nothing (RFC 0020 §6).
    std::string unresolved(const std::string& path) const
    {
        const auto root = symbolRoot(path);
        if(root == KERNEL_SYMBOL_ROOT)
        {
            const auto field = symbolField(path, KERNEL_SYMBOL_ROOT);
            bool declared = field.has_value() && kmdField(*field) != nullptr;
            if(!declared && field.has_value())
            {
                const auto indexed = splitIndexedField(*field);
                const auto* list = indexed.has_value() ? kmdField(indexed->first) : nullptr;
                declared = list != nullptr && list->type == MetadataType::INT_LIST;
            }
            return declared ? std::string()
                            : ", which metadata schema '" + _schema.name + "' does not declare";
        }
        if(root == DEVICE_SYMBOL_ROOT)
        {
            const auto field = symbolField(path, DEVICE_SYMBOL_ROOT);
            return field.has_value() && CriteriaDevice::declares(*field)
                       ? std::string()
                       : ", which is not a device property";
        }
        if(_native)
        {
            return {};
        }
        if(_pattern == nullptr)
        {
            return ", but the engine declares no graph_match and so binds nothing; only "
                   "$kernel.* and $device.* resolve";
        }
        if(root == GRAPH_SYMBOL_ROOT)
        {
            const auto field = symbolField(path, GRAPH_SYMBOL_ROOT);
            const bool published
                = field.has_value()
                  && std::find(PUBLISHED_GRAPH_FIELDS.begin(), PUBLISHED_GRAPH_FIELDS.end(), *field)
                         != PUBLISHED_GRAPH_FIELDS.end();
            return published ? std::string() : ", which is not a published graph field";
        }
        if(findVariable(root) != nullptr)
        {
            const auto rest = std::string_view(path).substr(root.size());
            if(!rest.empty() && (rest.front() != '.' || !isPublishedTensorField(rest.substr(1))))
            {
                return ", but tensor '" + std::string(root) + "' publishes no such field";
            }
            return {};
        }
        return findAttribute(path) == nullptr ? ", which the engine's graph_match does not publish"
                                              : std::string();
    }

private:
    const MetadataField* kmdField(std::string_view name) const
    {
        const auto it
            = std::find_if(_schema.fields.begin(),
                           _schema.fields.end(),
                           [name](const MetadataField& field) { return field.name == name; });
        return it == _schema.fields.end() ? nullptr : &*it;
    }

    const MetadataSchema& _schema;
    const CompiledGraphPattern* _pattern = nullptr;
    bool _native = false;
};

} // namespace detail

/// Why @p criteria cannot run on an engine whose `graph_match` is @p graphMatch and whose
/// KMD is @p schema, or "" when it can (RFC 0018 A.5 checks 5 and 6). The message reads as
/// a continuation of "its criteria ...".
///
///  - Every reference must resolve in what the engine publishes, under the rules of
///    detail::PublishedSymbols::unresolved().
///  - `present` / `not_present` may not ask about a pattern variable or attribute the
///    pattern binds as required: the answer is fixed by the pattern (RFC 0020 §6.1).
inline std::string criteriaReferenceError(const CompiledCriteria& criteria,
                                          const GraphMatch& graphMatch,
                                          const MetadataSchema& schema)
{
    const detail::PublishedSymbols published(graphMatch, schema);

    for(const auto& path : criteria.references())
    {
        if(auto reason = published.unresolved(path); !reason.empty())
        {
            std::string message = "read $";
            message += path;
            message += reason;
            return message;
        }
    }

    for(const auto& query : criteria.presenceQueries())
    {
        const auto* variable = published.findVariable(query);
        const auto* attribute = variable == nullptr ? published.findAttribute(query) : nullptr;
        if((variable != nullptr && !variable->optional)
           || (attribute != nullptr && !attribute->optional))
        {
            return "ask whether $" + query
                   + " is present, but the pattern binds it as required, so the answer is fixed";
        }
    }
    return {};
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

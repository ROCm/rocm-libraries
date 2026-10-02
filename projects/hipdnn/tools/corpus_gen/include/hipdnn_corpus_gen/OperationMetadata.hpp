// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_plugin_sdk/heuristics/uhd/Expressions.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

/// @file OperationMetadata.hpp
/// @brief The declared problem space of an operation (RFC 0019.13 §4).
/// Per §4.3.2 it neither bounds numeric dimensions nor defines cost quantities.
namespace hipdnn_corpus_gen
{

/// A parameter value; enums are held as their declared strings.
using ParameterValue = std::variant<int64_t, double, bool, std::string>;

/// One parameter assignment: a corpus entry and the `q.*` half of a training row.
using ProblemPoint = std::map<std::string, ParameterValue>;

enum class ParameterType
{
    INT64,
    FLOAT64,
    ENUM,
    BOOL
};

/// One declared parameter of the problem space (§4.3.2).
struct Parameter
{
    std::string name;
    ParameterType type = ParameterType::INT64;

    /// A semantic bound only (e.g. one dim cannot exceed another), never "expected sizes".
    std::optional<std::pair<int64_t, int64_t>> range;

    /// Required for ENUM; a BOOL has an implicit [false, true].
    std::vector<std::string> values;

    /// Representative values (§5.2); used for weighting, not bounding.
    std::vector<ParameterValue> commonValues;

    std::string description;

    /// The values of an ENUM or BOOL parameter; empty for numeric types.
    std::vector<ParameterValue> enumerable() const
    {
        std::vector<ParameterValue> result;
        if(type == ParameterType::ENUM)
        {
            for(const auto& value : values)
            {
                result.emplace_back(value);
            }
        }
        else if(type == ParameterType::BOOL)
        {
            result.emplace_back(false);
            result.emplace_back(true);
        }
        return result;
    }
};

/// How a tensor's contents must be produced, where contents affect the work (e.g. MoE routing).
/// Proposed addition to RFC 0019.13 §4. A generator that cannot produce a declared fill must
/// refuse the operation rather than substitute another.
enum class FillKind
{
    ZEROS, ///< Default; for contents that do not affect the work done.
    UNIFORM, ///< Uniform random over the tensor's dtype range.
    SEQUENCE, ///< 0, 1, 2, ... useful for index tensors that must be in range.
    ROUTING_OFFSETS, ///< Per-expert first-token offsets, skewed by `skew` (§12.2).
    EXPERT_ASSIGNMENT ///< Per-token expert index, consistent with the offsets.
};

/// One tensor's declared contents.
struct TensorFill
{
    /// Tensor name as the graph builder assigns it (not its uid).
    std::string tensor;
    FillKind kind = FillKind::ZEROS;

    /// `$q.*` references or literals, resolved like builder arguments.
    std::vector<std::string> arguments;
};

/// One argument of the graph builder call (§4.3.6).
struct BuilderArgument
{
    enum class Kind
    {
        DIRECT, ///< Copy the named `$q.*` value.
        EXPR, ///< Evaluate an expression; arrays yield dims lists.
        STRIDES_OF, ///< Row-major contiguous strides for a previously-named argument.
        DTYPE_OF, ///< Map a dtype string to the FlatBuffers DataType enumerator.
        CONSTANT ///< Literal.
    };

    std::string name;
    Kind kind = Kind::DIRECT;
    std::string source; ///< DIRECT, DTYPE_OF
    /// One expression per element, followed by the elements' `when` conditions.
    hipdnn_plugin_sdk::uhd::ExpressionSet expressions;
    /// Scalar EXPR: resolves to an unfloored double (e.g. a softmax scale), not a dim.
    bool scalarExpression = false;
    /// Number of list elements in an array EXPR.
    size_t elementCount = 0;
    /// Per element, the index of its `when` condition; a false condition omits the element,
    /// letting one declaration cover several ranks.
    std::vector<std::optional<size_t>> elementConditions;
    std::string of; ///< STRIDES_OF
    nlohmann::json constant; ///< CONSTANT
};

/// How a parameter assignment becomes a graph (§4.3.6).
struct GraphBuilderSpec
{
    std::string function;
    std::string source;
    std::vector<BuilderArgument> arguments;
};

/// A stratification bucket set (§4.3.5).
struct Regime
{
    std::string parameter;
    std::vector<nlohmann::json> buckets;
    std::string derived; ///< "alignment" when the buckets are labels rather than values
    std::string description;
};

/// One facet of a corpus entry's regime label (see RegimeLabel.hpp): ordered, labelled
/// conditions over the whole point, first match winning. The label joins facets with `_`.
struct RegimeAxis
{
    std::string name;

    /// The label each clause of @ref clauses assigns, in the same order.
    std::vector<std::string> labels;

    /// §6.2 boolean expressions, same evaluator as @ref OperationMetadata::constraints.
    hipdnn_plugin_sdk::uhd::ExpressionSet clauses;

    /// Each clause as declared, for generating matching points (RegimeFocus.hpp).
    std::vector<nlohmann::json> whens;

    /// The label when no clause matches; must be declared explicitly.
    std::string otherwise;
};

/// Maps a kernel descriptor pack's metadata fields to declared parameters, so the pack's
/// geometries read back as problem points. An operation without one has no kernel pool.
struct KernelCatalog
{
    /// Declared parameter -> pack field. A descriptor missing any field contributes nothing.
    std::map<std::string, std::string> fields;

    /// Declared parameter -> pack spelling -> declared value. Unmapped values are skipped and
    /// counted, never guessed.
    std::map<std::string, std::map<std::string, std::string>> enums;

    /// Declared parameter -> value every kernel in the pack has, for parameters the pack has
    /// no field for. Validated and typed at load; not allowed for parameters in `fields`.
    std::map<std::string, ParameterValue> constants;

    bool empty() const
    {
        return fields.empty();
    }
};

/// How a sampled value may drift from an archetype's and still be a plausible problem.
/// Structured per parameter, since uniform noise yields shapes no real workload contains.
struct Neighbourhood
{
    enum class Kind
    {
        SCALE, ///< multiply by one of `factors` (rounded, kept >= 1)
        MULTIPLE, ///< move by whole steps of `of`
        VALUES, ///< take one of an explicit list
        MIRROR ///< follow another parameter, optionally times one of `ratios`
    };

    std::string parameter;
    Kind kind = Kind::SCALE;
    std::vector<double> factors; ///< SCALE
    int64_t of = 0; ///< MULTIPLE: the alignment
    std::vector<int64_t> steps; ///< MULTIPLE: how many multiples to move
    std::vector<int64_t> values; ///< VALUES
    std::string mirrors; ///< MIRROR: the parameter followed
    std::vector<double> ratios; ///< MIRROR: permitted ratios to it
};

/// A correlated tuple of values drawn from a real workload (§12.2, §12.3).
/// A value `$q.<other>` copies that parameter's draw; the referent must be set by the archetype
/// and references must be acyclic (both checked at load).
struct Archetype
{
    std::string name;
    std::string source; ///< provenance of the shape
    std::string note;
    std::map<std::string, std::vector<nlohmann::json>> values;

    /// Indices into `OperationMetadata::parameters`, referents before their copies.
    std::vector<size_t> drawOrder;
};

/// Fractions of a combination's budget drawn from archetypes, neighbourhoods and exploration.
struct Mixture
{
    double archetypes = 0.0;
    double neighbourhood = 0.0;
    double exploration = 1.0;

    /// True when no mixture was declared.
    bool isExplorationOnly() const
    {
        return archetypes <= 0.0 && neighbourhood <= 0.0;
    }
};

/// One operation's declared space.
struct OperationMetadata
{
    std::string schemaVersion;
    std::string operation;
    std::string displayName;
    std::string description;
    std::vector<Parameter> parameters;
    std::string stratificationAxis;
    std::map<std::string, Regime> regimes;

    /// Facets of each entry's regime label, in join order. Empty means empty labels.
    std::vector<RegimeAxis> regimeLabel;

    /// See KernelCatalogSource.hpp. Empty means the operation has no kernel pool.
    KernelCatalog kernelCatalog;

    GraphBuilderSpec graphBuilder;

    /// Declared tensor contents (proposed §4 addition). Absent means all zeros.
    std::vector<TensorFill> variantPack;

    /// Cross-parameter relations a point must satisfy (proposed §4 addition), as boolean §6.2
    /// expressions; a failing point is never proposed. A `range` cannot express these.
    hipdnn_plugin_sdk::uhd::ExpressionSet constraints;

    /// Recorded workload shapes and allowed drift (proposed §4 addition). Empty means the
    /// corpus is exploration only.
    std::vector<Archetype> archetypes;
    std::map<std::string, Neighbourhood> neighbourhood;

    /// Indices into `parameters`, each `mirror` target before its follower.
    std::vector<size_t> perturbationOrder;
    Mixture mixture;

    const Parameter* find(const std::string& name) const
    {
        for(const auto& parameter : parameters)
        {
            if(parameter.name == name)
            {
                return &parameter;
            }
        }
        return nullptr;
    }
};

/// What a load produced, or why it did not.
struct MetadataLoad
{
    std::optional<OperationMetadata> metadata;

    /// Every problem found, not just the first.
    std::vector<std::string> errors;

    bool ok() const
    {
        return metadata.has_value() && errors.empty();
    }
};

namespace detail
{

/// The `$q.<name>` a reference names, or empty if it is not a query reference.
inline std::string queryReference(const std::string& text)
{
    constexpr std::string_view PREFIX = "$q.";
    return text.rfind(PREFIX, 0) == 0 ? text.substr(PREFIX.size()) : std::string();
}

/// @brief @p parameters' indices in an order where each comes after every parameter it reads.
/// @p reads maps a parameter to those it copies; ties keep declaration order so seeds stay
/// reproducible. Needed because the JSON parser does not preserve key order.
/// @return nullopt on a cycle, with @p unresolved listing the unplaced parameters.
inline std::optional<std::vector<size_t>>
    dependencyOrder(const std::vector<Parameter>& parameters,
                    const std::map<std::string, std::vector<std::string>>& reads,
                    std::string& unresolved)
{
    std::vector<size_t> order;
    order.reserve(parameters.size());
    std::vector<bool> placed(parameters.size(), false);

    const auto isPlaced = [&](const std::string& name) {
        for(size_t index = 0; index < parameters.size(); ++index)
        {
            if(parameters[index].name == name)
            {
                return static_cast<bool>(placed[index]);
            }
        }
        return true; // undeclared: refused by the caller, not an ordering constraint
    };
    const auto ready = [&](size_t index) {
        const auto found = reads.find(parameters[index].name);
        return found == reads.end()
               || std::all_of(found->second.begin(), found->second.end(), isPlaced);
    };

    while(order.size() < parameters.size())
    {
        size_t next = 0;
        while(next < parameters.size() && (placed[next] || !ready(next)))
        {
            ++next;
        }
        if(next == parameters.size())
        {
            unresolved.clear();
            for(size_t index = 0; index < parameters.size(); ++index)
            {
                if(!placed[index])
                {
                    unresolved += (unresolved.empty() ? "'" : ", '") + parameters[index].name + "'";
                }
            }
            return std::nullopt;
        }
        placed[next] = true;
        order.push_back(next);
    }
    return order;
}

/// The neighbourhood kind a declaration names, or nullopt.
inline std::optional<Neighbourhood::Kind> parseNeighbourhoodKind(const std::string& text)
{
    if(text == "scale")
    {
        return Neighbourhood::Kind::SCALE;
    }
    if(text == "multiple")
    {
        return Neighbourhood::Kind::MULTIPLE;
    }
    if(text == "values")
    {
        return Neighbourhood::Kind::VALUES;
    }
    if(text == "mirror")
    {
        return Neighbourhood::Kind::MIRROR;
    }
    return std::nullopt;
}

inline std::optional<ParameterType> parseParameterType(const std::string& text)
{
    if(text == "int64")
    {
        return ParameterType::INT64;
    }
    if(text == "float64")
    {
        return ParameterType::FLOAT64;
    }
    if(text == "enum")
    {
        return ParameterType::ENUM;
    }
    if(text == "bool")
    {
        return ParameterType::BOOL;
    }
    return std::nullopt;
}

inline std::optional<FillKind> parseFillKind(const std::string& text)
{
    if(text == "zeros")
    {
        return FillKind::ZEROS;
    }
    if(text == "uniform")
    {
        return FillKind::UNIFORM;
    }
    if(text == "sequence")
    {
        return FillKind::SEQUENCE;
    }
    if(text == "routing_offsets")
    {
        return FillKind::ROUTING_OFFSETS;
    }
    if(text == "expert_assignment")
    {
        return FillKind::EXPERT_ASSIGNMENT;
    }
    return std::nullopt;
}

inline std::optional<BuilderArgument::Kind> parseArgumentKind(const std::string& text)
{
    if(text == "direct")
    {
        return BuilderArgument::Kind::DIRECT;
    }
    if(text == "expr")
    {
        return BuilderArgument::Kind::EXPR;
    }
    if(text == "strides_of")
    {
        return BuilderArgument::Kind::STRIDES_OF;
    }
    if(text == "dtype_of")
    {
        return BuilderArgument::Kind::DTYPE_OF;
    }
    if(text == "constant")
    {
        return BuilderArgument::Kind::CONSTANT;
    }
    return std::nullopt;
}

inline ParameterValue jsonToValue(const nlohmann::json& value)
{
    if(value.is_boolean())
    {
        return value.get<bool>();
    }
    if(value.is_number_integer())
    {
        return value.get<int64_t>();
    }
    if(value.is_number_float())
    {
        return value.get<double>();
    }
    return value.is_string() ? value.get<std::string>() : std::string();
}

} // namespace detail

/// The three axes §4.3.4 permits.
inline bool isPermittedStratificationAxis(const std::string& axis)
{
    return axis == "arithmetic_intensity" || axis == "working_set" || axis == "reduction_ratio";
}

/// @brief Parses and validates operation metadata (§4.2, §4.4).
/// Every `$q.*` reference in `graph_builder` must name a declared parameter.
inline MetadataLoad parseOperationMetadata(const nlohmann::json& root)
{
    MetadataLoad load;
    OperationMetadata metadata;

    const auto require = [&](const char* field, bool present) {
        if(!present)
        {
            load.errors.emplace_back(std::string("missing required field '") + field + "'");
        }
        return present;
    };

    require("schema_version", root.contains("schema_version"));
    require("operation", root.contains("operation"));
    require("parameters", root.contains("parameters"));
    require("stratification_axis", root.contains("stratification_axis"));
    require("graph_builder", root.contains("graph_builder"));

    metadata.schemaVersion = root.value("schema_version", "");
    metadata.operation = root.value("operation", "");
    metadata.displayName = root.value("display_name", "");
    metadata.description = root.value("description", "");
    metadata.stratificationAxis = root.value("stratification_axis", "");

    if(!metadata.stratificationAxis.empty()
       && !isPermittedStratificationAxis(metadata.stratificationAxis))
    {
        load.errors.push_back("stratification_axis '" + metadata.stratificationAxis
                              + "' is not one of arithmetic_intensity, working_set, "
                                "reduction_ratio");
    }

    if(root.contains("parameters"))
    {
        for(const auto& [name, body] : root.at("parameters").items())
        {
            Parameter parameter;
            parameter.name = name;
            parameter.description = body.value("description", "");

            const auto type = detail::parseParameterType(body.value("type", ""));
            if(!type.has_value())
            {
                load.errors.push_back("parameter '" + name + "' has no valid type");
                continue;
            }
            parameter.type = *type;

            if(body.contains("values"))
            {
                for(const auto& value : body.at("values"))
                {
                    parameter.values.push_back(value.get<std::string>());
                }
            }
            if(parameter.type == ParameterType::ENUM && parameter.values.empty())
            {
                load.errors.push_back("enum parameter '" + name + "' declares no values");
            }

            if(body.contains("range") && body.at("range").size() == 2)
            {
                // A null upper bound means only the floor is semantic (e.g. padding, §4.3.2).
                const auto& range = body.at("range");
                const auto low = range[0].get<int64_t>();
                parameter.range = {low,
                                   range[1].is_null() ? std::numeric_limits<int64_t>::max()
                                                      : range[1].get<int64_t>()};
            }
            if(body.contains("common_values"))
            {
                for(const auto& value : body.at("common_values"))
                {
                    parameter.commonValues.push_back(detail::jsonToValue(value));
                }
            }
            metadata.parameters.push_back(std::move(parameter));
        }
    }

    if(root.contains("regimes"))
    {
        for(const auto& [name, body] : root.at("regimes").items())
        {
            Regime regime;
            regime.parameter = body.value("parameter", "");
            regime.derived = body.value("derived", "");
            regime.description = body.value("description", "");
            if(body.contains("buckets"))
            {
                for(const auto& bucket : body.at("buckets"))
                {
                    regime.buckets.push_back(bucket);
                }
            }
            // §4.4 check 3.
            if(metadata.find(regime.parameter) == nullptr)
            {
                load.errors.push_back("regime '" + name + "' stratifies undeclared parameter '"
                                      + regime.parameter + "'");
            }
            metadata.regimes.emplace(name, std::move(regime));
        }
    }

    if(root.contains("regime_label"))
    {
        // An array, not an object: facet order is part of the label.
        for(const auto& entry : root.at("regime_label"))
        {
            RegimeAxis axis;
            axis.name = entry.value("name", "");
            axis.otherwise = entry.value("otherwise", "");
            if(axis.otherwise.empty())
            {
                load.errors.push_back("regime_label axis '" + axis.name
                                      + "' declares no 'otherwise' label");
            }

            std::vector<nlohmann::json> clauses;
            for(const auto& clause : entry.value("labels", nlohmann::json::array()))
            {
                axis.labels.push_back(clause.value("label", ""));
                clauses.push_back(clause.value("when", nlohmann::json()));
            }
            axis.whens = clauses;

            try
            {
                axis.clauses = hipdnn_plugin_sdk::uhd::ExpressionSet(clauses);
                for(const auto& variable : axis.clauses.variables())
                {
                    const auto name = detail::queryReference(variable);
                    if(!name.empty() && metadata.find(name) == nullptr)
                    {
                        load.errors.push_back("regime_label axis '" + axis.name
                                              + "' references undeclared parameter '" + name + "'");
                    }
                }
            }
            catch(const std::exception& error)
            {
                load.errors.push_back("regime_label axis '" + axis.name
                                      + "': " + std::string(error.what()));
            }
            metadata.regimeLabel.push_back(std::move(axis));
        }
    }

    if(root.contains("kernel_catalog"))
    {
        const auto& catalog = root.at("kernel_catalog");

        // Named objects: `value()` returns a temporary, and `items()` over one dangles.
        const auto fields = catalog.value("metadata", nlohmann::json::object());
        const auto enums = catalog.value("enums", nlohmann::json::object());
        const auto constants = catalog.value("constants", nlohmann::json::object());

        for(const auto& field : fields.items())
        {
            // Otherwise the pack would silently look like it carries no geometry.
            if(metadata.find(field.key()) == nullptr)
            {
                load.errors.push_back("kernel_catalog maps undeclared parameter '" + field.key()
                                      + "'");
                continue;
            }
            metadata.kernelCatalog.fields[field.key()] = field.value().get<std::string>();
        }

        for(const auto& mapping : enums.items())
        {
            const auto* parameter = metadata.find(mapping.key());
            if(parameter == nullptr)
            {
                load.errors.push_back("kernel_catalog declares values for undeclared parameter '"
                                      + mapping.key() + "'");
                continue;
            }
            if(metadata.kernelCatalog.fields.count(mapping.key()) == 0)
            {
                load.errors.push_back("kernel_catalog declares values for '" + mapping.key()
                                      + "' but maps no pack field to it");
            }

            for(const auto& value : mapping.value().items())
            {
                const auto declared = value.value().get<std::string>();
                // The target must be one of the parameter's declared values.
                if(parameter->type == ParameterType::ENUM
                   && std::find(parameter->values.begin(), parameter->values.end(), declared)
                          == parameter->values.end())
                {
                    load.errors.push_back("kernel_catalog maps '" + mapping.key() + "' onto '"
                                          + declared + "', which it does not declare");
                }
                metadata.kernelCatalog.enums[mapping.key()][value.key()] = declared;
            }
        }

        for(const auto& constant : constants.items())
        {
            const auto* parameter = metadata.find(constant.key());
            if(parameter == nullptr)
            {
                load.errors.push_back("kernel_catalog declares a constant for undeclared "
                                      "parameter '"
                                      + constant.key() + "'");
                continue;
            }
            if(metadata.kernelCatalog.fields.count(constant.key()) != 0)
            {
                load.errors.push_back("kernel_catalog declares a constant for '" + constant.key()
                                      + "', which it also reads from the pack");
                continue;
            }
            const auto& json = constant.value();
            const auto printed = json.is_string() ? json.get<std::string>() : json.dump();
            const auto wrongType = [&load, &constant, &printed](const char* expected) {
                load.errors.push_back("kernel_catalog fixes '" + constant.key() + "' at '" + printed
                                      + "', which is not " + expected);
            };

            switch(parameter->type)
            {
            // Unreachable today; required by the build, and refuses any future unhandled type.
            default:
                wrongType("a value of a type this loader can check");
                continue;

            case ParameterType::INT64:
                if(!json.is_number_integer() && !json.is_number_unsigned())
                {
                    wrongType("an integer");
                    continue;
                }
                metadata.kernelCatalog.constants[constant.key()] = json.get<int64_t>();
                break;

            case ParameterType::FLOAT64:
                if(!json.is_number())
                {
                    wrongType("a number");
                    continue;
                }
                metadata.kernelCatalog.constants[constant.key()] = json.get<double>();
                break;

            case ParameterType::BOOL:
                if(!json.is_boolean())
                {
                    wrongType("a boolean");
                    continue;
                }
                metadata.kernelCatalog.constants[constant.key()] = json.get<bool>();
                break;

            case ParameterType::ENUM:
                // Must be a declared value, as for an `enums` target.
                if(!json.is_string()
                   || std::find(parameter->values.begin(), parameter->values.end(), printed)
                          == parameter->values.end())
                {
                    wrongType("one of its declared values");
                    continue;
                }
                metadata.kernelCatalog.constants[constant.key()] = printed;
                break;
            }
        }
    }

    if(root.contains("graph_builder"))
    {
        const auto& builder = root.at("graph_builder");
        metadata.graphBuilder.function = builder.value("function", "");
        metadata.graphBuilder.source = builder.value("source", "");
        if(metadata.graphBuilder.function.empty())
        {
            load.errors.emplace_back("graph_builder declares no function");
        }

        std::vector<std::string> declared;
        for(const auto& argument : builder.value("arguments", nlohmann::json::array()))
        {
            BuilderArgument resolved;
            resolved.name = argument.value("name", "");

            const auto kind = detail::parseArgumentKind(argument.value("kind", ""));
            if(!kind.has_value())
            {
                load.errors.push_back("argument '" + resolved.name + "' has no valid kind");
                continue;
            }
            resolved.kind = *kind;
            resolved.source = argument.value("source", "");
            resolved.of = argument.value("of", "");
            if(argument.contains("constant"))
            {
                resolved.constant = argument.at("constant");
            }
            else if(argument.contains("value") && resolved.kind == BuilderArgument::Kind::CONSTANT)
            {
                resolved.constant = argument.at("value");
            }

            if(resolved.kind == BuilderArgument::Kind::EXPR && argument.contains("value"))
            {
                try
                {
                    const auto& value = argument.at("value");
                    resolved.scalarExpression = !value.is_array();
                    std::vector<nlohmann::json> program;
                    std::vector<nlohmann::json> conditions;
                    if(resolved.scalarExpression)
                    {
                        program.push_back(value);
                    }
                    else
                    {
                        for(const auto& element : value)
                        {
                            const bool conditional = element.is_object() && element.size() == 2
                                                     && element.contains("when")
                                                     && element.contains("value");
                            program.push_back(conditional ? element.at("value") : element);
                            resolved.elementConditions.push_back(
                                conditional ? std::optional<size_t>(conditions.size())
                                            : std::nullopt);
                            if(conditional)
                            {
                                conditions.push_back(element.at("when"));
                            }
                        }
                        resolved.elementCount = program.size();
                        for(auto& condition : resolved.elementConditions)
                        {
                            if(condition.has_value())
                            {
                                *condition += resolved.elementCount;
                            }
                        }
                        program.insert(program.end(), conditions.begin(), conditions.end());
                    }
                    resolved.expressions = hipdnn_plugin_sdk::uhd::ExpressionSet(program);
                }
                catch(const std::exception& error)
                {
                    load.errors.push_back("argument '" + resolved.name + "': " + error.what());
                }
            }

            // §4.4 check 2: every $q.* reference resolves to a declared parameter.
            const auto checkReference = [&](const std::string& text) {
                const auto name = detail::queryReference(text);
                if(!name.empty() && metadata.find(name) == nullptr)
                {
                    load.errors.push_back("argument '" + resolved.name
                                          + "' references undeclared "
                                            "parameter '"
                                          + name + "'");
                }
            };
            checkReference(resolved.source);
            // Variables come from the evaluator, so nested expressions are covered too.
            for(const auto& variable : resolved.expressions.variables())
            {
                checkReference(variable);
            }

            if(resolved.kind == BuilderArgument::Kind::STRIDES_OF
               && std::find(declared.begin(), declared.end(), resolved.of) == declared.end())
            {
                // Arguments resolve in declaration order; a forward reference cannot resolve.
                load.errors.push_back("argument '" + resolved.name + "' takes strides_of '"
                                      + resolved.of + "', which is not declared before it");
            }

            declared.push_back(resolved.name);
            metadata.graphBuilder.arguments.push_back(std::move(resolved));
        }
    }

    if(root.contains("constraints"))
    {
        try
        {
            metadata.constraints = hipdnn_plugin_sdk::uhd::ExpressionSet(
                root.at("constraints").get<std::vector<nlohmann::json>>());
            for(const auto& variable : metadata.constraints.variables())
            {
                const auto name = detail::queryReference(variable);
                if(!name.empty() && metadata.find(name) == nullptr)
                {
                    load.errors.push_back("constraint references undeclared parameter '" + name
                                          + "'");
                }
            }
        }
        catch(const std::exception& error)
        {
            load.errors.push_back("constraint: " + std::string(error.what()));
        }
    }

    if(root.contains("archetypes"))
    {
        for(const auto& entry : root.at("archetypes"))
        {
            Archetype archetype;
            archetype.name = entry.value("name", "");
            archetype.source = entry.value("source", "");
            archetype.note = entry.value("note", "");

            if(archetype.name.empty())
            {
                load.errors.emplace_back("an archetype declares no name");
            }

            // Named object: `items()` over a temporary dangles.
            const nlohmann::json declaredValues = entry.value("values", nlohmann::json::object());
            for(const auto& [name, list] : declaredValues.items())
            {
                // Catches a renamed parameter that would otherwise silently lose its anchor.
                if(metadata.find(name) == nullptr)
                {
                    load.errors.push_back("archetype '" + archetype.name
                                          + "' sets undeclared parameter '" + name + "'");
                    continue;
                }
                if(!list.is_array() || list.empty())
                {
                    load.errors.push_back("archetype '" + archetype.name + "' gives '" + name
                                          + "' no values");
                    continue;
                }
                archetype.values.emplace(name, list.get<std::vector<nlohmann::json>>());
            }

            // References must name a parameter the archetype sets; referents are drawn first.
            std::map<std::string, std::vector<std::string>> reads;
            for(const auto& parameter : metadata.parameters)
            {
                const auto found = archetype.values.find(parameter.name);
                if(found == archetype.values.end())
                {
                    continue;
                }
                for(const auto& value : found->second)
                {
                    if(!value.is_string())
                    {
                        continue;
                    }
                    const auto referenced = detail::queryReference(value.get<std::string>());
                    if(referenced.empty())
                    {
                        continue;
                    }
                    if(metadata.find(referenced) == nullptr)
                    {
                        load.errors.push_back("archetype '" + archetype.name + "' has '"
                                              + parameter.name + "' follow undeclared parameter '"
                                              + referenced + "'");
                    }
                    else if(archetype.values.count(referenced) == 0)
                    {
                        load.errors.push_back("archetype '" + archetype.name + "' has '"
                                              + parameter.name + "' follow '" + referenced
                                              + "', which the archetype does not set");
                    }
                    else
                    {
                        reads[parameter.name].push_back(referenced);
                    }
                }
            }
            std::string unresolved;
            if(auto order = detail::dependencyOrder(metadata.parameters, reads, unresolved))
            {
                archetype.drawOrder = std::move(*order);
            }
            else
            {
                load.errors.push_back("archetype '" + archetype.name
                                      + "' has references that form a cycle among " + unresolved);
            }
            metadata.archetypes.push_back(std::move(archetype));
        }
    }

    if(root.contains("neighbourhood"))
    {
        for(const auto& [name, body] : root.at("neighbourhood").items())
        {
            if(metadata.find(name) == nullptr)
            {
                load.errors.push_back("neighbourhood describes undeclared parameter '" + name
                                      + "'");
                continue;
            }

            Neighbourhood hood;
            hood.parameter = name;
            const auto kind = detail::parseNeighbourhoodKind(body.value("kind", ""));
            if(!kind.has_value())
            {
                load.errors.push_back("parameter '" + name
                                      + "' declares unknown neighbourhood kind '"
                                      + body.value("kind", "") + "'");
                continue;
            }
            hood.kind = *kind;
            hood.factors = body.value("factors", std::vector<double>{});
            // `of` is an alignment for "multiple" and a name for "mirror"; read by JSON type so
            // a mismatch is reported by the per-kind checks below rather than throwing.
            if(body.contains("of") && body.at("of").is_number_integer())
            {
                hood.of = body.at("of").get<int64_t>();
            }
            hood.steps = body.value("steps", std::vector<int64_t>{});
            hood.values = body.value("values", std::vector<int64_t>{});
            hood.ratios = body.value("ratios", std::vector<double>{});
            if(body.contains("of") && body.at("of").is_string())
            {
                hood.mirrors = body.at("of").get<std::string>();
            }

            // Each kind must have something to move by; a no-op neighbourhood is an error.
            const std::string parameterName = name;
            const auto complain = [&load, &parameterName](const std::string& what) {
                std::string message = "parameter '";
                message += parameterName;
                message += "' neighbourhood ";
                message += what;
                load.errors.push_back(std::move(message));
            };
            switch(hood.kind)
            {
            case Neighbourhood::Kind::SCALE:
                if(hood.factors.empty())
                {
                    complain("is 'scale' with no factors");
                }
                break;
            case Neighbourhood::Kind::MULTIPLE:
                if(hood.of <= 0)
                {
                    complain("is 'multiple' with no positive alignment");
                }
                if(hood.steps.empty())
                {
                    complain("is 'multiple' with no steps");
                }
                break;
            case Neighbourhood::Kind::VALUES:
                if(hood.values.empty())
                {
                    complain("is 'values' with no values");
                }
                break;
            case Neighbourhood::Kind::MIRROR:
                if(hood.mirrors.empty())
                {
                    complain("is 'mirror' with no parameter to follow");
                }
                else if(metadata.find(hood.mirrors) == nullptr)
                {
                    {
                        std::string what = "follows undeclared parameter '";
                        what += hood.mirrors;
                        what += "'";
                        complain(what);
                    }
                }
                break;
            default:
                break;
            }
            metadata.neighbourhood.emplace(name, std::move(hood));
        }
    }

    // A follower moves after what it follows, regardless of parsed key order.
    {
        std::map<std::string, std::vector<std::string>> reads;
        for(const auto& [name, hood] : metadata.neighbourhood)
        {
            if(hood.kind == Neighbourhood::Kind::MIRROR && !hood.mirrors.empty())
            {
                reads[name].push_back(hood.mirrors);
            }
        }
        std::string unresolved;
        if(auto order = detail::dependencyOrder(metadata.parameters, reads, unresolved))
        {
            metadata.perturbationOrder = std::move(*order);
        }
        else
        {
            load.errors.push_back("neighbourhood mirrors form a cycle among " + unresolved);
        }
    }

    if(root.contains("mixture"))
    {
        const auto& body = root.at("mixture");
        metadata.mixture.archetypes = body.value("archetypes", 0.0);
        metadata.mixture.neighbourhood = body.value("neighbourhood", 0.0);
        metadata.mixture.exploration = body.value("exploration", 0.0);

        const auto total = metadata.mixture.archetypes + metadata.mixture.neighbourhood
                           + metadata.mixture.exploration;
        if(std::abs(total - 1.0) > 1e-6)
        {
            // Not normalised: shares must be declared exactly.
            load.errors.push_back("mixture shares total " + std::to_string(total) + ", not 1.0");
        }
    }
    else if(!metadata.archetypes.empty())
    {
        // Default shares when archetypes are declared without a mixture.
        metadata.mixture = Mixture{0.20, 0.60, 0.20};
    }

    if(metadata.mixture.neighbourhood > 0.0 && metadata.neighbourhood.empty())
    {
        load.errors.emplace_back(
            "mixture asks for neighbourhood samples but no neighbourhood is declared");
    }
    if((metadata.mixture.archetypes > 0.0 || metadata.mixture.neighbourhood > 0.0)
       && metadata.archetypes.empty())
    {
        load.errors.emplace_back("mixture asks for archetype samples but none are declared");
    }

    if(root.contains("variant_pack"))
    {
        for(const auto& entry : root.at("variant_pack"))
        {
            TensorFill fill;
            fill.tensor = entry.value("tensor", "");

            const auto kind = detail::parseFillKind(entry.value("fill", "zeros"));
            if(!kind.has_value())
            {
                // Refused, not defaulted to zeros.
                load.errors.push_back("tensor '" + fill.tensor + "' declares unknown fill '"
                                      + entry.value("fill", "") + "'");
                continue;
            }
            fill.kind = *kind;

            for(const auto& argument : entry.value("arguments", nlohmann::json::array()))
            {
                const auto text
                    = argument.is_string() ? argument.get<std::string>() : argument.dump();
                const auto name = detail::queryReference(text);
                if(!name.empty() && metadata.find(name) == nullptr)
                {
                    load.errors.push_back("tensor '" + fill.tensor
                                          + "' fill references "
                                            "undeclared parameter '"
                                          + name + "'");
                }
                fill.arguments.push_back(text);
            }
            metadata.variantPack.push_back(std::move(fill));
        }
    }

    load.metadata = std::move(metadata);
    return load;
}

} // namespace hipdnn_corpus_gen

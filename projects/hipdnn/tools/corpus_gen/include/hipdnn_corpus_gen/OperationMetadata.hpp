// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_plugin_sdk/heuristics/uhd/DescriptorExpression.hpp>
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
///
/// An operation's parameters, and how a parameter assignment becomes a graph, are data rather
/// than code. That is the difference between a generator that supports the operations someone
/// wrote builders for and one that supports the operations someone described: §4.3.6 dispatches
/// by `graph_builder.function` into the builders that already exist, so adding an operation is
/// a metadata file.
///
/// Two things this deliberately does not do, both because §4.3.2 says so:
///
///  - **It does not bound numeric dimensions.** A `range` is a *semantic* bound -- a limit
///    inherent to the operation, such as one dimension that cannot exceed another. Authoring
///    one to mean "sizes we expect" is a metadata bug: it silently truncates the space the
///    engine claims to support. What bounds the region is benchmarkability (computed from
///    shape and dtype against device memory) intersected with applicability (asked).
///  - **It does not define cost quantities.** FLOPs and bytes are op-intrinsic and belong to
///    the binding layer (§4.3.3); re-declaring them per file would create the second definition
///    site that rule exists to prevent.
namespace hipdnn_corpus_gen
{

/// A value a parameter can take. Enums and bools arrive as strings and bools respectively
/// because that is how the metadata declares them and how a `q.*` column must read back.
using ParameterValue = std::variant<int64_t, double, bool, std::string>;

/// One parameter assignment: the corpus entry, and the `q.*` half of a training row.
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

    /// A semantic bound only, and usually absent. See the file comment: this is not where a
    /// generator learns how large a tensor may be.
    std::optional<std::pair<int64_t, int64_t>> range;

    /// Required for ENUM; a BOOL has an implicit [false, true].
    std::vector<std::string> values;

    /// Representative values where a usage record exists (§5.2). Weighting, not bounding.
    std::vector<ParameterValue> commonValues;

    std::string description;

    /// The values an enumerable parameter can take. Empty for the numeric types, which are
    /// searched rather than enumerated.
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

/// How a tensor's contents must be produced.
///
/// **Proposed addition to RFC 0019.13 §4**, which specifies `graph_builder` but says nothing
/// about the variant pack. For most operations that omission is harmless: a benchmark fills
/// buffers with anything and measures the same kernel, because the work is fixed by shape.
///
/// It is not harmless for every operation. An MoE grouped matmul's `first_token_offset` and
/// `token_index` carry the routing -- how many tokens each expert receives -- and that decides
/// the work each grouped GEMM does. Two problems with identical graphs and different routing
/// are different problems, and §12.2 already lists routing skew and capacity factor as
/// parameters of the space. A benchmark that fills every buffer with a constant measures one
/// arbitrary routing and labels it with whatever the corpus point claimed.
///
/// So contents are declared where they are load-bearing, and default to zeros where they are
/// not. The vocabulary is closed: a generator that cannot produce a declared fill must refuse
/// the operation rather than substitute one, because substituting is exactly the failure this
/// exists to prevent.
enum class FillKind
{
    ZEROS, ///< Default. Correct wherever contents do not affect the work done.
    UNIFORM, ///< Uniform random over the tensor's dtype range.
    SEQUENCE, ///< 0, 1, 2, ... useful for index tensors that must be in range.
    ROUTING_OFFSETS, ///< Per-expert first-token offsets, skewed by `skew` (§12.2).
    EXPERT_ASSIGNMENT ///< Per-token expert index, consistent with the offsets.
};

/// One tensor's declared contents.
struct TensorFill
{
    /// Tensor name as the graph builder assigns it; the variant pack is keyed by uid, and the
    /// name is what a metadata author can actually see.
    std::string tensor;
    FillKind kind = FillKind::ZEROS;

    /// Parameters of the fill, as `$q.*` references or literals -- e.g. the expert count and
    /// skew a routing fill needs. Resolved exactly like a builder argument.
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
    /// Compiled once at metadata admission; all dimensions share one expression DAG.
    hipdnn_plugin_sdk::uhd::expression::Program expressions{std::vector<nlohmann::json>{}};
    /// EXPR whose `value` was one expression rather than an array: it resolves to a single
    /// floating-point scalar, unfloored -- a softmax scale of rsqrt(head_dim) is not a dim.
    bool scalarExpression = false;
    /// How many list elements an array EXPR has. The program holds these first, then the
    /// conditions of any element written `{"when": <condition>, "value": <expr>}`.
    size_t elementCount = 0;
    /// Per element, the program index of its `when` condition, if it has one. An element whose
    /// condition is false is omitted, which is how one declaration spells tensors of two ranks
    /// -- [N, C, D, H, W] for 3-D problems and [N, C, H, W] for 2-D ones.
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

/// One facet of the stratification label a corpus entry carries (see RegimeLabel.hpp).
///
/// A @ref Regime says which values of one parameter matter. This says which *populations* the
/// operation has, which is a different question and not answerable from one parameter: an SDPA
/// decode is `seqlen_q == 1`, a cross-attention is `seqlen_q > seqlen_kv`, and neither is a
/// bucket of anything. So a facet is an ordered list of labelled conditions over the whole
/// point, first match winning, and the label is the facets joined with `_`.
///
/// Declared rather than compiled in because the populations differ per operation -- a
/// convolution's are not phase/context/grouping -- and an operation that names its own is the
/// only mechanism by which the per-regime regret table (RFC 0019.13 §11.2) can exist for an
/// operation nobody wrote C++ for.
struct RegimeAxis
{
    std::string name;

    /// The label each clause of @ref clauses assigns, in the same order.
    std::vector<std::string> labels;

    /// The clauses, as §6.2 boolean expressions -- the same language and the same evaluator as
    /// @ref OperationMetadata::constraints, so there is no second expression dialect to learn
    /// or to keep in step.
    hipdnn_plugin_sdk::uhd::expression::Program clauses{std::vector<nlohmann::json>{}};

    /// The label for a point no clause matched. Spelled out in the declaration rather than
    /// defaulted to a silent "other", because an unlabelled population is one the regret table
    /// reports as a bucket nobody can act on.
    std::string otherwise;
};

/// How a kernel descriptor pack's metadata reads back as a problem point.
///
/// A descriptor engine's kernels bake their geometry in, so the pack is the authority on the
/// one thing nothing else can answer: which problems the engine will have a candidate for at
/// all. Reading it needs a vocabulary, because the pack names its fields whatever the kernel
/// author named them (`num_query_heads`, `head_size`, `causal`) and the declaration names its
/// parameters whatever the operation calls them (`heads`, `head_dim`, `is_causal`).
///
/// Declared rather than compiled in, for the same reason the regime facets are: hardcoding one
/// operation's field names is what made the previous tool SDPA-only. An operation that ships no
/// `kernel_catalog` block simply has no kernel pool -- which is how coverage stays "whatever has
/// a declaration", with no operation list anywhere for someone to forget to extend.
struct KernelCatalog
{
    /// Declared parameter name -> the pack metadata field carrying it. A descriptor missing
    /// any of these fields carries no geometry, and contributes nothing rather than a point
    /// with holes in it.
    std::map<std::string, std::string> fields;

    /// Declared parameter name -> pack spelling -> declared value, for the parameters whose
    /// values are names rather than numbers. A pack value with no entry here is skipped and
    /// counted, never guessed: a pack in fp8 read as bf16 would produce a corpus of graphs the
    /// engine does not serve, labelled as ones it does.
    std::map<std::string, std::map<std::string, std::string>> enums;

    /// Declared parameter name -> the value every geometry from a pack takes, for parameters
    /// the pack vocabulary has no field for at all.
    ///
    /// Not a default and not a guess -- a statement about the kernels. A pack records what its
    /// kernels were compiled for, and a parameter it never mentions is one the kernel author
    /// did not vary: rocKE's dense attention packs say `causal: 1` without saying which corner
    /// the diagonal is anchored at, because every kernel in them anchors it top-left. Saying so
    /// here is what lets the builder resolve an argument the pack cannot supply, and it is
    /// checked against the parameter's declared values the same way an `enums` target is, so a
    /// constant the operation cannot take is a load error rather than a corpus of graphs that
    /// describe kernels nobody compiled.
    ///
    /// A parameter mapped under `metadata` is not eligible: the pack answers for it, and a
    /// constant would silently override what the pack actually said.
    ///
    /// Typed at load, not at use, so a constant that does not fit its parameter is one error
    /// naming the declaration rather than a pack that mysteriously contributes nothing.
    std::map<std::string, ParameterValue> constants;

    bool empty() const
    {
        return fields.empty();
    }
};

/// How a sampled value may drift from an archetype's and still be a plausible problem.
///
/// Perturbation has to be structured, because the parameters are not interchangeable numbers.
/// A channel count moves between aligned multiples; a spatial extent halves and doubles; a
/// filter takes one of the small odd sizes; a width follows its height. Adding uniform noise
/// to all four produces a shape no workload contains, which is the failure this replaces --
/// an unstructured search over the feasible region put 0.04% of its draws anywhere near a
/// real convolution.
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
///
/// Realism is a joint fact, not a per-parameter one: C=3 is realistic beside K=64 with a 7x7
/// filter at 224x224, and meaningless beside K=905. So an archetype names whole assignments
/// and a draw takes one value per parameter from the lists here, rather than each parameter
/// being sampled from its own marginal.
///
/// A value may be `$q.<other>`, which copies whatever that parameter was drawn as -- how a
/// square image or a self-attention sequence length is said. References resolve after their
/// referent, so `W: ["$q.H"]` requires H to be declared first.
struct Archetype
{
    std::string name;
    std::string source; ///< where the shape came from; provenance, not decoration
    std::string note;
    std::map<std::string, std::vector<nlohmann::json>> values;
};

/// What fractions of a combination's budget come from where.
///
/// Anchors alone memorise a handful of shapes and predict nothing between them; the wide net
/// alone is what produced four realistic shapes in ten thousand. Keeping an exploration share
/// is the same argument §5.4's exploration floor makes in the other direction -- a corpus that
/// only refines where it already looks good never revisits its blind spots, and an engine
/// serves problems nobody thought to write down.
struct Mixture
{
    double archetypes = 0.0;
    double neighbourhood = 0.0;
    double exploration = 1.0;

    /// True when nothing was declared, so the exploration alone applies.
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

    /// The facets composing each corpus entry's regime label, in the order they are joined.
    /// Empty means the operation names no populations, and every entry's label is empty --
    /// which the manifest records honestly rather than inventing a stratification.
    std::vector<RegimeAxis> regimeLabel;

    /// How a descriptor pack's metadata reads back as a point (see KernelCatalogSource.hpp).
    /// Empty means the operation has no kernel pool, not that its packs are empty.
    KernelCatalog kernelCatalog;

    GraphBuilderSpec graphBuilder;

    /// Declared tensor contents (proposed §4 addition). Absent means every tensor is zeros,
    /// which is right for every operation whose work is fixed by shape alone.
    std::vector<TensorFill> variantPack;

    /// Relations a problem point must satisfy to be a problem at all (proposed §4 addition).
    ///
    /// §4.3.2 describes a `range` as "a limit inherent to the operation, such as one dimension
    /// that cannot exceed another" -- which is coupling -- but a range bounds one parameter
    /// against constants and cannot say that a filter must fit inside its input. Without a
    /// mechanism, a generator draws each parameter independently and a convolution's
    /// parameters are related, so nearly every candidate is refused by the frontend before any
    /// engine sees it. Measured on conv_fwd: 2559 of 2559.
    ///
    /// Boolean §6.2 expressions, the same language UMD criteria use. A point failing any is
    /// never proposed, which costs nothing and stops the search from mapping the frontend's
    /// validator and reporting the result as the engine's region.
    hipdnn_plugin_sdk::uhd::expression::Program constraints{std::vector<nlohmann::json>{}};

    /// Recorded workload shapes, and how far a draw may drift from one (proposed §4 addition).
    ///
    /// Empty means the operation makes no claim about which of its problems occur in practice,
    /// and the corpus is then the exploration alone -- correct about what the engine serves,
    /// and silent about what anyone runs.
    std::vector<Archetype> archetypes;
    std::map<std::string, Neighbourhood> neighbourhood;
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

    /// Every problem found, not just the first. A metadata author fixing one typo per build
    /// is a worse experience than seeing the file's faults at once, and §4.4 lists checks
    /// that are independent of each other.
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

/// The three axes §4.3.4 permits. Declared rather than inferred, because arithmetic intensity
/// is shape-invariant for a bandwidth-bound op and stratifying by it yields one bucket.
inline bool isPermittedStratificationAxis(const std::string& axis)
{
    return axis == "arithmetic_intensity" || axis == "working_set" || axis == "reduction_ratio";
}

/// @brief Parses and validates operation metadata (§4.2, §4.4).
///
/// The validation that matters most is §4.4's second check -- every `$q.*` reference in
/// `graph_builder` resolving to a declared parameter. That is what catches a builder mapping
/// written against a different parameterization than the operation declares, which otherwise
/// surfaces as a corpus of graphs built from stale defaults: they benchmark, they produce
/// times, and the times are labelled with parameters that never reached the graph.
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
                // A null upper bound means "this floor is semantic, the ceiling is not".
                // Padding is the motivating case: it cannot be negative, and nothing about the
                // operation says how large it may be (§4.3.2).
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
        // An array, not an object: the facets are joined in order and the label is what a
        // reader greps for, so the order is part of the declaration rather than whatever a map
        // happened to collate to.
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

            try
            {
                axis.clauses = hipdnn_plugin_sdk::uhd::expression::Program(clauses);
                for(const auto& variable : axis.clauses.variables())
                {
                    const auto name = detail::queryReference(variable);
                    if(!name.empty() && metadata.find(name) == nullptr)
                    {
                        load.errors.push_back("regime_label axis '" + axis.name
                                              + "' references undeclared parameter '" + name
                                              + "'");
                    }
                }
            }
            catch(const std::exception& error)
            {
                load.errors.push_back("regime_label axis '" + axis.name + "': "
                                      + std::string(error.what()));
            }
            metadata.regimeLabel.push_back(std::move(axis));
        }
    }

    if(root.contains("kernel_catalog"))
    {
        const auto& catalog = root.at("kernel_catalog");

        // Each block is bound to a named object before `items()` is called on it: `value()`
        // returns by value, and iterating the items of a temporary walks a destroyed object --
        // which at -O0 throws and at -O3 quietly iterates nothing, so every check below would
        // pass by never running. Same reason as the archetype values at the end of this file.
        const auto fields    = catalog.value("metadata", nlohmann::json::object());
        const auto enums     = catalog.value("enums", nlohmann::json::object());
        const auto constants = catalog.value("constants", nlohmann::json::object());

        for(const auto& field : fields.items())
        {
            // A mapping to a parameter that does not exist reads a pack into nothing, and the
            // symptom is a pack that "carries no geometry" -- indistinguishable from a pack
            // that really does not. Named here instead, where it is one line to fix.
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
                // The target must be a value the parameter can actually take: a mapping onto a
                // spelling the operation does not declare builds graphs the constraints and the
                // builder disagree about, and neither says so.
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
                load.errors.push_back("kernel_catalog fixes '" + constant.key() + "' at '"
                                      + printed + "', which is not " + expected);
            };

            switch(parameter->type)
            {
            // Unreachable while every ParameterType is handled below, and kept because the
            // build treats a switch with no default as an error. A type added without an arm
            // here refuses the constant rather than storing it untyped.
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
                // The same check an `enums` target gets: fixing a parameter at a spelling the
                // operation does not declare builds graphs its constraints and its builder
                // disagree about, and neither of them says so.
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
                    resolved.expressions = hipdnn_plugin_sdk::uhd::expression::Program(program);
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
            // Every variable the expression actually reads, from the evaluator itself rather
            // than from a string scan -- which is what makes the check hold for a nested
            // expression such as a convolution's output extent.
            for(const auto& variable : resolved.expressions.variables())
            {
                checkReference(variable);
            }

            if(resolved.kind == BuilderArgument::Kind::STRIDES_OF
               && std::find(declared.begin(), declared.end(), resolved.of) == declared.end())
            {
                // Arguments resolve in declaration order, so a forward reference cannot be
                // satisfied -- and would otherwise produce strides for an empty dims list.
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
            metadata.constraints = hipdnn_plugin_sdk::uhd::expression::Program(
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

            // Bound to a named object first: `items()` over a temporary iterates a value that
            // has already been destroyed.
            const nlohmann::json declaredValues = entry.value("values", nlohmann::json::object());
            for(const auto& [name, list] : declaredValues.items())
            {
                // An archetype naming a parameter that does not exist is how a renamed
                // parameter silently stops being anchored: the draw keeps working and quietly
                // reverts to the unrealistic marginal.
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

            // A reference must resolve, and must resolve to something already drawn.
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
                }
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
            // `of` is overloaded by kind: an alignment for "multiple", a parameter name for
            // "mirror". Read by actual type rather than by kind, so a declaration that gets the
            // pair wrong is caught by the per-kind checks below instead of throwing here.
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

            // Each kind is refused when it has nothing to move by, rather than falling back to
            // "leave the value alone" -- a neighbourhood that silently never perturbs turns the
            // corpus back into its archetypes and reports full coverage.
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
            // Normalising silently would let "0.2/0.6/0.1" read as a declaration of proportions
            // nobody wrote, and the corpus composition is exactly what this field is for.
            load.errors.push_back("mixture shares total " + std::to_string(total) + ", not 1.0");
        }
    }
    else if(!metadata.archetypes.empty())
    {
        // Declaring anchors and no mixture is asking for them to be used; picking the shares
        // silently is worse than picking them visibly, so this is the documented default.
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
                // Refused rather than defaulted to zeros: a fill nobody can produce is a
                // corpus that measures something other than what it says.
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

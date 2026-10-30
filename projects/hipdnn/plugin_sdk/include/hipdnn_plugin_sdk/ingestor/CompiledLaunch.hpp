// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/CompiledCriteria.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorJsonRules.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/JsonExpression.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

/// @file CompiledLaunch.hpp
/// @brief A UDD's declarative single launch (RFC 0017 §6): grid, block, dynamic LDS,
///        workspace and an ordered argument list, each written as a JSON Expression
///        Language formula over the symbols criteria read (CompiledCriteria.hpp).
///
/// Four pieces, in the order a launch meets them:
///  - compileLaunch() runs when the UDD is read. It refuses every static error, including
///    a literal that breaks its value rule.
///  - launchReferenceError() runs when a pack pairs the UDD with an engine: every
///    reference resolves in what that engine publishes.
///  - CompiledLaunch::resolve() runs per plan and turns the formulas into a plain-data
///    ResolvedLaunch. A shape-dependent failure throws BAD_PARAM, so the plan walks skip
///    the kernel.
///  - saveResolvedLaunch() / restoreResolvedLaunch() store a resolved launch with a saved
///    plan.

namespace hipdnn_plugin_sdk::ingestor
{

/// The dispatch symbol the loader gives every declarative UDD. Authors may not use it, or
/// any name under it.
inline constexpr std::string_view DECLARATIVE_LAUNCH_SYMBOL = "hipdnn.declarative_launch";
/// The versioned alias a saved declarative plan records (SavedLaunchInputs::dispatchSymbol).
inline constexpr std::string_view DECLARATIVE_LAUNCH_SYMBOL_V1 = "hipdnn.declarative_launch.v1";

/// What one kernel argument passes. The values are stored in a saved launch, so they never
/// change.
enum class LaunchArgumentKind : int64_t
{
    POINTER = 0, ///< The device address of a tensor the graph binds.
    WORKSPACE = 1, ///< The device address of the plan's workspace.
    SCALAR = 2 ///< A value computed when the plan is built, passed by value.
};

/// The type of a SCALAR argument. The values are stored in a saved launch, so they never
/// change.
enum class LaunchScalarType : int64_t
{
    I32 = 0,
    U32 = 1,
    I64 = 2,
    F32 = 3
};

/// One kernel argument of a resolved launch.
struct ResolvedLaunchArgument
{
    LaunchArgumentKind kind = LaunchArgumentKind::POINTER;
    /// The uid of the tensor whose device address is passed. POINTER only.
    int64_t tensorUid = 0;
    /// SCALAR only.
    LaunchScalarType scalarType = LaunchScalarType::I32;
    /// SCALAR only: the value in host byte order. The first size() bytes are used and the
    /// rest are zero.
    std::array<std::byte, 8> bytes{};

    /// Bytes the argument occupies in the kernarg segment: 8 for a pointer, the workspace
    /// and an I64 scalar, 4 for the other scalars.
    uint32_t size() const
    {
        if(kind != LaunchArgumentKind::SCALAR || scalarType == LaunchScalarType::I64)
        {
            return 8;
        }
        return 4;
    }
};

/// A declarative launch with every formula evaluated: what the provider needs to launch.
struct ResolvedLaunch
{
    std::array<uint32_t, 3> grid{1, 1, 1};
    std::array<uint32_t, 3> block{1, 1, 1};
    uint32_t sharedMemBytes = 0;
    size_t workspaceBytes = 0;
    std::vector<ResolvedLaunchArgument> arguments;
};

/// Compares what each kind of argument passes: the uid of a pointer, the type and value of
/// a scalar. Fields the kind does not use are ignored.
inline bool operator==(const ResolvedLaunchArgument& lhs, const ResolvedLaunchArgument& rhs)
{
    if(lhs.kind != rhs.kind)
    {
        return false;
    }
    if(lhs.kind == LaunchArgumentKind::POINTER)
    {
        return lhs.tensorUid == rhs.tensorUid;
    }
    if(lhs.kind == LaunchArgumentKind::SCALAR)
    {
        return lhs.scalarType == rhs.scalarType
               && std::memcmp(lhs.bytes.data(), rhs.bytes.data(), lhs.size()) == 0;
    }
    return true;
}

inline bool operator!=(const ResolvedLaunchArgument& lhs, const ResolvedLaunchArgument& rhs)
{
    return !(lhs == rhs);
}

inline bool operator==(const ResolvedLaunch& lhs, const ResolvedLaunch& rhs)
{
    return lhs.grid == rhs.grid && lhs.block == rhs.block
           && lhs.sharedMemBytes == rhs.sharedMemBytes && lhs.workspaceBytes == rhs.workspaceBytes
           && lhs.arguments == rhs.arguments;
}

inline bool operator!=(const ResolvedLaunch& lhs, const ResolvedLaunch& rhs)
{
    return !(lhs == rhs);
}

namespace detail
{

/// The largest magnitude a double holds exactly as an integer: 2^53. jsonexpr computes in
/// double, so a whole number beyond it may already be rounded.
inline constexpr int64_t MAX_EXACT_LAUNCH_INTEGER = int64_t{1} << 53;

/// The rule a launch value must meet.
enum class LaunchValueRule
{
    GRID_OR_BLOCK_AXIS, ///< A whole number in [1, UINT32_MAX].
    SHARED_MEM_BYTES, ///< A whole number in [0, UINT32_MAX].
    WORKSPACE_BYTES, ///< A whole number in [0, 2^53].
    I32, ///< A whole number in the int32 range.
    U32, ///< A whole number in [0, UINT32_MAX].
    I64, ///< A whole number.
    F32 ///< A finite number that stays finite as a float.
};

/// A value that met its rule: `integer` for a whole-number rule, `real` for F32.
struct LaunchNumber
{
    int64_t integer = 0;
    float real = 0.0F;
};

/// The inclusive bounds of a whole-number rule.
inline std::pair<int64_t, int64_t> launchRuleBounds(LaunchValueRule rule)
{
    constexpr auto U32_MAX = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    switch(rule)
    {
    case LaunchValueRule::GRID_OR_BLOCK_AXIS:
        return {1, U32_MAX};
    case LaunchValueRule::SHARED_MEM_BYTES:
    case LaunchValueRule::U32:
        return {0, U32_MAX};
    case LaunchValueRule::WORKSPACE_BYTES:
        return {0, MAX_EXACT_LAUNCH_INTEGER};
    case LaunchValueRule::I32:
        return {std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()};
    case LaunchValueRule::I64:
    case LaunchValueRule::F32:
    default:
        return {-MAX_EXACT_LAUNCH_INTEGER, MAX_EXACT_LAUNCH_INTEGER};
    }
}

/// What @p rule requires, completing "it must be ...".
inline std::string describeLaunchRule(LaunchValueRule rule)
{
    if(rule == LaunchValueRule::F32)
    {
        return "a finite number in the float range";
    }
    const auto [low, high] = launchRuleBounds(rule);
    return "a whole number in [" + std::to_string(low) + ", " + std::to_string(high) + "]";
}

/// @p value as an exact integer: a whole number of magnitude at most 2^53.
inline std::optional<int64_t> exactLaunchInteger(const jsonexpr::Value& value)
{
    if(value.isInt())
    {
        const auto integer = value.asInt();
        if(integer < -MAX_EXACT_LAUNCH_INTEGER || integer > MAX_EXACT_LAUNCH_INTEGER)
        {
            return std::nullopt;
        }
        return integer;
    }
    if(value.isDouble())
    {
        const auto real = value.asDouble();
        constexpr auto LIMIT = static_cast<double>(MAX_EXACT_LAUNCH_INTEGER);
        if(!std::isfinite(real) || std::trunc(real) != real || real < -LIMIT || real > LIMIT)
        {
            return std::nullopt;
        }
        return static_cast<int64_t>(real);
    }
    return std::nullopt;
}

/// @p value checked against @p rule, or nullopt when it breaks the rule. Null, bool,
/// string and array values break every rule.
inline std::optional<LaunchNumber> checkLaunchValue(LaunchValueRule rule,
                                                    const jsonexpr::Value& value)
{
    if(rule == LaunchValueRule::F32)
    {
        if(!value.isNumber())
        {
            return std::nullopt;
        }
        const auto real = value.toNumber();
        if(!std::isfinite(real)
           || std::fabs(real) > static_cast<double>(std::numeric_limits<float>::max()))
        {
            return std::nullopt;
        }
        return LaunchNumber{0, static_cast<float>(real)};
    }
    const auto integer = exactLaunchInteger(value);
    const auto [low, high] = launchRuleBounds(rule);
    if(!integer.has_value() || *integer < low || *integer > high)
    {
        return std::nullopt;
    }
    return LaunchNumber{*integer, 0.0F};
}

/// A JSON number literal as the value the language would evaluate it to. An unsigned
/// literal beyond int64 stays a double, which every whole-number rule refuses.
inline jsonexpr::Value launchLiteralValue(const nlohmann::json& literal)
{
    if(literal.is_number_unsigned())
    {
        const auto raw = literal.get<nlohmann::json::number_unsigned_t>();
        if(raw
           > static_cast<nlohmann::json::number_unsigned_t>(std::numeric_limits<int64_t>::max()))
        {
            return {static_cast<double>(raw)};
        }
        return {static_cast<int64_t>(raw)};
    }
    if(literal.is_number_integer())
    {
        return {literal.get<int64_t>()};
    }
    return {literal.get<double>()};
}

/// One compiled formula and what its value must be.
struct LaunchFormula
{
    /// The field as diagnostics name it, such as `grid.x` or `argument 'n'`.
    std::string field;
    LaunchValueRule rule = LaunchValueRule::GRID_OR_BLOCK_AXIS;
    jsonexpr::Expression<CriteriaSymbols> expression;
};

/// One compiled `args_signature` entry.
struct CompiledLaunchArgument
{
    std::string name;
    LaunchArgumentKind kind = LaunchArgumentKind::POINTER;
    LaunchScalarType scalarType = LaunchScalarType::I32;
    /// The sigil-free pattern variable a `tensor`, `dim` or `stride` source reads; empty
    /// for the workspace and an `expr` source.
    std::string tensor;
    /// The value of a SCALAR argument.
    LaunchFormula value;
};

/// The keys of a saved launch (saveResolvedLaunch()).
inline constexpr std::string_view SAVED_LAUNCH_GRID = "grid";
inline constexpr std::string_view SAVED_LAUNCH_BLOCK = "block";
inline constexpr std::string_view SAVED_LAUNCH_SHARED_MEM_BYTES = "shared_mem_bytes";
inline constexpr std::string_view SAVED_LAUNCH_WORKSPACE_BYTES = "workspace_bytes";
inline constexpr std::string_view SAVED_LAUNCH_ARGUMENT_KINDS = "argument_kinds";
inline constexpr std::string_view SAVED_LAUNCH_ARGUMENT_SCALAR_TYPES = "argument_scalar_types";
inline constexpr std::string_view SAVED_LAUNCH_ARGUMENT_VALUES = "argument_values";
inline constexpr std::array<std::string_view, 7> SAVED_LAUNCH_KEYS
    = {SAVED_LAUNCH_GRID,
       SAVED_LAUNCH_BLOCK,
       SAVED_LAUNCH_SHARED_MEM_BYTES,
       SAVED_LAUNCH_WORKSPACE_BYTES,
       SAVED_LAUNCH_ARGUMENT_KINDS,
       SAVED_LAUNCH_ARGUMENT_SCALAR_TYPES,
       SAVED_LAUNCH_ARGUMENT_VALUES};

/// Throws the error every malformed saved launch raises.
[[noreturn]] inline void failRestoredLaunch(const std::string& reason)
{
    fail("saved declarative launch: " + reason);
}

/// Stores @p number in the bytes of @p argument as its scalar type.
inline void writeLaunchScalar(ResolvedLaunchArgument& argument, const LaunchNumber& number)
{
    const auto store = [&argument](const auto value) {
        static_assert(sizeof(value) <= sizeof(argument.bytes));
        argument.bytes = {};
        std::memcpy(argument.bytes.data(), &value, sizeof(value));
    };
    switch(argument.scalarType)
    {
    case LaunchScalarType::I32:
        store(static_cast<int32_t>(number.integer));
        break;
    case LaunchScalarType::U32:
        store(static_cast<uint32_t>(number.integer));
        break;
    case LaunchScalarType::I64:
        store(number.integer);
        break;
    case LaunchScalarType::F32:
    default:
        store(number.real);
        break;
    }
}

} // namespace detail

class CompiledLaunch;

/// Compiles the declarative arm of @p udd, a whole UDD object; its `version`, `id` and
/// `name` are the loader's and are not read. @p name is the UDD name every evaluation error
/// starts with. @p where names the UDD and its file in every compile error.
/// @throws HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE) naming @p where on a
///         missing or unknown key, a malformed argument or source, a feature that is not
///         supported yet (`launches`, `intermediates`, an explicit `offset`, a `scalar` or
///         `attribute` source), a formula that does not compile, or a literal that breaks
///         its value rule.
inline std::shared_ptr<const CompiledLaunch>
    compileLaunch(const nlohmann::json& udd, const std::string& name, const std::string& where);

/// A UDD's compiled declarative launch. Immutable, so threads can resolve one instance at
/// once.
class CompiledLaunch
{
    struct ConstructionKey
    {
        explicit ConstructionKey() = default;
    };

public:
    CompiledLaunch(ConstructionKey /*key*/, nlohmann::json source, std::string name)
        : _source(std::move(source))
        , _name(std::move(name))
    {
    }

    /// The authored declarative body: every key of the UDD except `version`, `id`, `name`
    /// and the extension keys of its structural objects (root, `grid`, `block`, each
    /// argument and its `source`). Formulas are kept verbatim.
    const nlohmann::json& source() const noexcept
    {
        return _source;
    }

    /// The UDD name, for diagnostics.
    const std::string& name() const noexcept
    {
        return _name;
    }

    /// Every reference the launch reads, without the sigil: the formulas' references, the
    /// `<var>.dims[i]` / `<var>.strides[i]` a `dim` / `stride` source reads, and the
    /// variable a `tensor` source reads. Sorted and distinct.
    const std::vector<std::string>& references() const noexcept
    {
        return _references;
    }

    /// One entry per `args_signature` element: kind `global_buffer` and size 8 for a
    /// pointer or the workspace, kind `by_value` and size 4 or 8 for a scalar. Offset 0
    /// and name "", which a signature check does not compare.
    const std::vector<KernelArgument>& signature() const noexcept
    {
        return _signature;
    }

    /// Evaluates `workspace_bytes` alone.
    /// @throws HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM) when the value breaks
    ///         its rule.
    size_t workspaceBytes(const MatchContext& ctx,
                          const BoundTokens& bound,
                          const KernelDefinition& kernel) const
    {
        const CriteriaDevice device(ctx.deviceProperties);
        const CriteriaSymbols symbols(bound, device, &kernel);
        return static_cast<size_t>(evaluate(_workspace, symbols).integer);
    }

    /// Evaluates every formula and resolves every pointer to its tensor uid.
    /// @throws HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM) when a value breaks
    ///         its rule or a pointer's tensor has no bound uid.
    ResolvedLaunch resolve(const MatchContext& ctx,
                           const BoundTokens& bound,
                           const KernelDefinition& kernel) const
    {
        const CriteriaDevice device(ctx.deviceProperties);
        const CriteriaSymbols symbols(bound, device, &kernel);

        ResolvedLaunch launch;
        for(size_t axis = 0; axis < launch.grid.size(); ++axis)
        {
            launch.grid[axis] = static_cast<uint32_t>(evaluate(_grid[axis], symbols).integer);
            launch.block[axis] = static_cast<uint32_t>(evaluate(_block[axis], symbols).integer);
        }
        launch.sharedMemBytes = static_cast<uint32_t>(evaluate(_sharedMemBytes, symbols).integer);
        launch.workspaceBytes = static_cast<size_t>(evaluate(_workspace, symbols).integer);

        launch.arguments.reserve(_arguments.size());
        for(const auto& compiled : _arguments)
        {
            auto& argument = launch.arguments.emplace_back();
            argument.kind = compiled.kind;
            if(compiled.kind == LaunchArgumentKind::POINTER)
            {
                argument.tensorUid = tensorUid(compiled, bound);
            }
            else if(compiled.kind == LaunchArgumentKind::SCALAR)
            {
                argument.scalarType = compiled.scalarType;
                detail::writeLaunchScalar(argument, evaluate(compiled.value, symbols));
            }
        }
        return launch;
    }

private:
    friend std::shared_ptr<const CompiledLaunch>
        compileLaunch(const nlohmann::json& udd, const std::string& name, const std::string& where);
    friend std::string launchReferenceError(const CompiledLaunch& launch,
                                            const GraphMatch& graphMatch,
                                            const MetadataSchema& schema);

    detail::LaunchNumber evaluate(const detail::LaunchFormula& formula,
                                  const CriteriaSymbols& symbols) const
    {
        const auto value = formula.expression(symbols);
        const auto checked = detail::checkLaunchValue(formula.rule, value);
        if(!checked.has_value())
        {
            throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                        _name + ": " + formula.field + " evaluated to "
                                            + value.dump() + ", but it must be "
                                            + detail::describeLaunchRule(formula.rule));
        }
        return *checked;
    }

    /// The uid bound to the argument's tensor, published as `<var>` and `<var>.uid`.
    int64_t tensorUid(const detail::CompiledLaunchArgument& argument,
                      const BoundTokens& bound) const
    {
        if(const auto uid = tryGetBoundInt(bound, argument.tensor))
        {
            return *uid;
        }
        if(const auto uid = tryGetBoundInt(bound, tensorField(argument.tensor, "uid")))
        {
            return *uid;
        }
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                    _name + ": argument '" + argument.name + "' passes $"
                                        + argument.tensor
                                        + ", but the binding holds no uid for that tensor");
    }

    nlohmann::json _source;
    std::string _name;
    std::vector<std::string> _references;
    std::vector<KernelArgument> _signature;
    std::array<detail::LaunchFormula, 3> _grid;
    std::array<detail::LaunchFormula, 3> _block;
    detail::LaunchFormula _sharedMemBytes;
    detail::LaunchFormula _workspace;
    std::vector<detail::CompiledLaunchArgument> _arguments;
};

namespace detail
{

/// Compiles one formula @p node, written for @p field of the UDD in @p where. A formula is
/// an expression object, a `$` reference or a number; a number must meet @p rule.
inline LaunchFormula compileLaunchFormula(const nlohmann::json& node,
                                          LaunchValueRule rule,
                                          const std::string& field,
                                          const std::string& where)
{
    const std::string locator = field + " of " + where;
    if(node.is_number())
    {
        if(!checkLaunchValue(rule, launchLiteralValue(node)).has_value())
        {
            fail(locator + " is " + node.dump() + ", but it must be " + describeLaunchRule(rule));
        }
    }
    else if(!node.is_object() && !isReference(node))
    {
        fail(locator + " must be an expression object, a '$' reference or a number, not "
             + node.dump());
    }

    LaunchFormula formula;
    formula.field = field;
    formula.rule = rule;
    try
    {
        formula.expression = jsonexpr::compile<CriteriaSymbols>(node);
    }
    catch(const jsonexpr::JsonExpressionCompileError& error)
    {
        fail(locator + " does not compile: " + error.what());
    }
    return formula;
}

/// Compiles the three axes of @p object, the value of the `grid` or `block` key @p key.
inline std::array<LaunchFormula, 3>
    compileLaunchAxes(const nlohmann::json& object, std::string_view key, const std::string& where)
{
    const std::string at = "'" + std::string(key) + "' of " + where;
    requireObject(object, at);
    constexpr std::array<std::string_view, 3> AXES = {"x", "y", "z"};
    requireKnownKeys(object, AXES, at);
    std::array<LaunchFormula, 3> axes;
    for(size_t axis = 0; axis < AXES.size(); ++axis)
    {
        axes[axis] = compileLaunchFormula(requireKey(object, AXES[axis], at),
                                          LaunchValueRule::GRID_OR_BLOCK_AXIS,
                                          std::string(key) + "." + std::string(AXES[axis]),
                                          where);
    }
    return axes;
}

/// The string at @p key of @p object.
inline const std::string&
    requireLaunchString(const nlohmann::json& object, std::string_view key, const std::string& at)
{
    const auto& value = requireKey(object, key, at);
    if(!value.is_string())
    {
        fail("'" + std::string(key) + "' in " + at + " must be a string");
    }
    return value.get_ref<const nlohmann::json::string_t&>();
}

/// The scalar type @p text names.
inline std::optional<LaunchScalarType> parseLaunchScalarType(std::string_view text)
{
    if(text == "i32")
    {
        return LaunchScalarType::I32;
    }
    if(text == "u32")
    {
        return LaunchScalarType::U32;
    }
    if(text == "i64")
    {
        return LaunchScalarType::I64;
    }
    if(text == "f32")
    {
        return LaunchScalarType::F32;
    }
    return std::nullopt;
}

inline LaunchValueRule launchScalarRule(LaunchScalarType type)
{
    switch(type)
    {
    case LaunchScalarType::I32:
        return LaunchValueRule::I32;
    case LaunchScalarType::U32:
        return LaunchValueRule::U32;
    case LaunchScalarType::I64:
        return LaunchValueRule::I64;
    case LaunchScalarType::F32:
    default:
        return LaunchValueRule::F32;
    }
}

/// Compiles the `source` of @p argument, whose kind and (for a scalar) type are already
/// set. @p at names the argument and the UDD.
inline void compileLaunchSource(const nlohmann::json& source,
                                CompiledLaunchArgument& argument,
                                const std::string& at,
                                const std::string& where)
{
    const std::string sourceAt = "'source' of " + at;
    requireObject(source, sourceAt);
    const auto& from = requireLaunchString(source, "from", sourceAt);
    if(from == "scalar" || from == "attribute")
    {
        fail("source '" + from + "' in " + at + " is not supported yet");
    }

    const bool tensor = from == "tensor";
    const bool element = from == "dim" || from == "stride";
    if(tensor)
    {
        requireKnownKeys(source, {"from", "ref"}, sourceAt);
    }
    else if(element)
    {
        requireKnownKeys(source, {"from", "ref", "axis"}, sourceAt);
    }
    else if(from == "expr")
    {
        requireKnownKeys(source, {"from", "expr"}, sourceAt);
    }
    else
    {
        fail("unknown source '" + from + "' in " + at
             + "; expected 'tensor', 'dim', 'stride' or 'expr'");
    }

    if(argument.kind == LaunchArgumentKind::POINTER && !tensor)
    {
        fail(at + " is a pointer, so its source must be 'tensor', not '" + from + "'");
    }
    if(argument.kind == LaunchArgumentKind::SCALAR && tensor)
    {
        fail(at + " is a scalar, so its source must be 'dim', 'stride' or 'expr', not 'tensor'");
    }

    const auto rule = launchScalarRule(argument.scalarType);
    const std::string field = "argument '" + argument.name + "'";
    if(from == "expr")
    {
        argument.value
            = compileLaunchFormula(requireKey(source, "expr", sourceAt), rule, field, where);
        return;
    }

    const auto& ref = requireLaunchString(source, "ref", sourceAt);
    if(ref.size() < 2 || ref.front() != jsonexpr::VARIABLE_SIGIL
       || !isIdentifier(std::string_view(ref).substr(1)))
    {
        fail("'ref' in " + sourceAt + R"( must name a pattern variable as "$<name>", not ")" + ref
             + R"(")");
    }
    argument.tensor = ref.substr(1);
    if(tensor)
    {
        return;
    }

    const auto& axis = requireKey(source, "axis", sourceAt);
    if(!axis.is_number_integer() || (!axis.is_number_unsigned() && axis.get<int64_t>() < 0))
    {
        fail("'axis' in " + sourceAt + " must be a non-negative integer, not " + axis.dump());
    }
    const auto index = axis.get<uint64_t>();
    // `dim` and `stride` read the tokens publishTensor() publishes for one axis.
    const std::string path = "$"
                             + tensorElement(argument.tensor,
                                             from == "dim" ? "dims" : "strides",
                                             static_cast<size_t>(index));
    argument.value = compileLaunchFormula(nlohmann::json(path), rule, field, where);
}

/// Compiles `args_signature[index]`.
inline CompiledLaunchArgument
    compileLaunchArgument(const nlohmann::json& entry, size_t index, const std::string& where)
{
    const std::string entryAt = "args_signature[" + std::to_string(index) + "] of " + where;
    requireObject(entry, entryAt);
    if(entry.contains("offset"))
    {
        fail("'offset' in " + entryAt
             + " is not supported yet; arguments are packed in declaration order");
    }
    requireKnownKeys(entry, {"name", "kind", "type", "source"}, entryAt);

    CompiledLaunchArgument argument;
    argument.name = requireLaunchString(entry, "name", entryAt);
    if(argument.name.empty())
    {
        fail("'name' in " + entryAt + " must not be empty");
    }
    const std::string at = "argument '" + argument.name + "' of " + where;

    const auto& kind = requireLaunchString(entry, "kind", entryAt);
    if(kind == "pointer")
    {
        argument.kind = LaunchArgumentKind::POINTER;
    }
    else if(kind == "workspace")
    {
        argument.kind = LaunchArgumentKind::WORKSPACE;
    }
    else if(kind == "scalar")
    {
        argument.kind = LaunchArgumentKind::SCALAR;
    }
    else
    {
        fail("'kind' in " + entryAt + " is '" + kind
             + "'; expected 'pointer', 'workspace' or 'scalar'");
    }

    if(argument.kind == LaunchArgumentKind::SCALAR)
    {
        const auto& type = requireLaunchString(entry, "type", entryAt);
        const auto parsed = parseLaunchScalarType(type);
        if(!parsed.has_value())
        {
            fail("'type' in " + entryAt + " is '" + type
                 + "'; expected 'i32', 'u32', 'i64' or 'f32'");
        }
        argument.scalarType = *parsed;
    }
    else if(entry.contains("type"))
    {
        fail(at + " is a " + kind + ", which takes no 'type'");
    }

    if(argument.kind == LaunchArgumentKind::WORKSPACE)
    {
        if(entry.contains("source"))
        {
            fail(at + " is the workspace, which takes no 'source'");
        }
        return argument;
    }
    compileLaunchSource(requireKey(entry, "source", entryAt), argument, at, where);
    return argument;
}

/// The kernarg entry a packaged kernel's signature must carry for @p argument.
inline KernelArgument launchSignatureEntry(const CompiledLaunchArgument& argument)
{
    ResolvedLaunchArgument sized;
    sized.kind = argument.kind;
    sized.scalarType = argument.scalarType;
    return {argument.kind == LaunchArgumentKind::SCALAR ? "by_value" : "global_buffer",
            sized.size(),
            0,
            ""};
}

/// @p object without its extension keys.
inline nlohmann::json withoutExtensionKeys(const nlohmann::json& object)
{
    auto kept = nlohmann::json::object();
    for(const auto& item : object.items())
    {
        if(!isExtensionKey(item.key()))
        {
            kept[item.key()] = item.value();
        }
    }
    return kept;
}

/// The declarative body of @p udd, a UDD compileLaunch() has already validated: every key
/// except `version`, `id` and `name`, with the extension keys of the root, `grid`, `block`,
/// each argument and each argument's `source` dropped, so an annotation does not change
/// what the launch compares as. Formulas are kept verbatim.
inline nlohmann::json launchSource(const nlohmann::json& udd)
{
    auto source = withoutExtensionKeys(udd);
    for(const char* loaderKey : {"version", "id", "name"})
    {
        source.erase(loaderKey);
    }
    for(const char* axes : {"grid", "block"})
    {
        source[axes] = withoutExtensionKeys(source[axes]);
    }
    for(auto& argument : source["args_signature"])
    {
        argument = withoutExtensionKeys(argument);
        if(const auto it = argument.find("source"); it != argument.end())
        {
            *it = withoutExtensionKeys(*it);
        }
    }
    return source;
}

} // namespace detail

inline std::shared_ptr<const CompiledLaunch>
    compileLaunch(const nlohmann::json& udd, const std::string& name, const std::string& where)
{
    detail::requireObject(udd, where);
    for(const char* deferred : {"launches", "intermediates"})
    {
        if(udd.contains(std::string(deferred)))
        {
            detail::fail("'" + std::string(deferred) + "' in " + where
                         + " is not supported yet; a declarative UDD describes one launch");
        }
    }
    if(udd.contains("dispatch_symbol"))
    {
        detail::fail(where
                     + " carries both 'dispatch_symbol' and declarative launch keys; a UDD "
                       "takes one of the two");
    }
    detail::requireKnownKeys(udd,
                             {"version",
                              "id",
                              "name",
                              "grid",
                              "block",
                              "shared_mem_bytes",
                              "workspace_bytes",
                              "args_signature"},
                             where);

    CompiledLaunch launch(CompiledLaunch::ConstructionKey{}, nullptr, name);

    launch._grid = detail::compileLaunchAxes(detail::requireKey(udd, "grid", where), "grid", where);
    launch._block
        = detail::compileLaunchAxes(detail::requireKey(udd, "block", where), "block", where);
    const auto optionalFormula
        = [&udd, &where](std::string_view key, detail::LaunchValueRule rule) {
              const auto it = udd.find(std::string(key));
              return detail::compileLaunchFormula(
                  it == udd.end() ? nlohmann::json(0) : *it, rule, std::string(key), where);
          };
    launch._sharedMemBytes
        = optionalFormula("shared_mem_bytes", detail::LaunchValueRule::SHARED_MEM_BYTES);
    launch._workspace
        = optionalFormula("workspace_bytes", detail::LaunchValueRule::WORKSPACE_BYTES);

    const auto& arguments = detail::requireKey(udd, "args_signature", where);
    if(!arguments.is_array())
    {
        detail::fail("'args_signature' in " + where + " must be an array");
    }
    std::vector<std::string> names;
    names.reserve(arguments.size());
    for(size_t index = 0; index < arguments.size(); ++index)
    {
        auto argument = detail::compileLaunchArgument(arguments[index], index, where);
        names.push_back(argument.name);
        launch._signature.push_back(detail::launchSignatureEntry(argument));
        launch._arguments.push_back(std::move(argument));
    }
    detail::requireNoDuplicates(names, "argument name", "'args_signature' of " + where);

    std::set<std::string> references;
    const auto collect = [&references](const detail::LaunchFormula& formula) {
        references.insert(formula.expression.variables().begin(),
                          formula.expression.variables().end());
    };
    for(size_t axis = 0; axis < launch._grid.size(); ++axis)
    {
        collect(launch._grid[axis]);
        collect(launch._block[axis]);
    }
    collect(launch._sharedMemBytes);
    collect(launch._workspace);
    for(const auto& argument : launch._arguments)
    {
        if(argument.kind == LaunchArgumentKind::POINTER)
        {
            references.insert(argument.tensor);
        }
        else if(argument.kind == LaunchArgumentKind::SCALAR)
        {
            collect(argument.value);
        }
    }
    launch._references.assign(references.begin(), references.end());
    launch._source = detail::launchSource(udd);

    return std::make_shared<const CompiledLaunch>(std::move(launch));
}

/// Why @p launch cannot run on an engine whose `graph_match` is @p graphMatch and whose
/// KMD is @p schema, or "" when it can. The message reads as a continuation of "pairs it
/// with engine '<e>', but its launch ...".
///
///  - Every reference resolves under the rules criteria references follow
///    (detail::PublishedSymbols::unresolved()).
///  - A `tensor`, `dim` or `stride` source names a variable of the engine's `nodes`
///    pattern. A native `graph_match` publishes its tensors at runtime, so it skips this;
///    an engine without `graph_match` binds no tensor at all.
inline std::string launchReferenceError(const CompiledLaunch& launch,
                                        const GraphMatch& graphMatch,
                                        const MetadataSchema& schema)
{
    const detail::PublishedSymbols published(graphMatch, schema);

    if(!published.native())
    {
        for(const auto& argument : launch._arguments)
        {
            if(argument.tensor.empty())
            {
                continue;
            }
            const std::string binding
                = "binds argument '" + argument.name + "' to $" + argument.tensor;
            if(published.pattern() == nullptr)
            {
                return binding + ", but the engine declares no graph_match and so binds no tensor";
            }
            if(published.findVariable(argument.tensor) == nullptr)
            {
                return binding + ", which is not a tensor the engine's graph_match binds";
            }
        }
    }

    for(const auto& path : launch.references())
    {
        if(auto reason = published.unresolved(path); !reason.empty())
        {
            std::string message = "reads $";
            message += path;
            message += reason;
            return message;
        }
    }
    return {};
}

/// The plain data a saved plan stores for @p launch (SavedLaunchInputs::values under
/// DECLARATIVE_LAUNCH_SYMBOL_V1). Each argument contributes one entry to three parallel
/// lists: its kind, its scalar type (0 unless a scalar), and its value (the tensor uid of a
/// pointer, 0 for the workspace, and for a scalar the integer value or, for F32, the bit
/// pattern of the float).
inline MetadataValues saveResolvedLaunch(const ResolvedLaunch& launch)
{
    std::vector<int64_t> kinds;
    std::vector<int64_t> types;
    std::vector<int64_t> values;
    kinds.reserve(launch.arguments.size());
    types.reserve(launch.arguments.size());
    values.reserve(launch.arguments.size());
    for(const auto& argument : launch.arguments)
    {
        kinds.push_back(static_cast<int64_t>(argument.kind));
        const bool scalar = argument.kind == LaunchArgumentKind::SCALAR;
        types.push_back(scalar ? static_cast<int64_t>(argument.scalarType) : 0);
        int64_t value = 0;
        if(argument.kind == LaunchArgumentKind::POINTER)
        {
            value = argument.tensorUid;
        }
        else if(scalar)
        {
            const auto load = [&argument](auto held) {
                std::memcpy(&held, argument.bytes.data(), sizeof(held));
                return static_cast<int64_t>(held);
            };
            switch(argument.scalarType)
            {
            case LaunchScalarType::I32:
                value = load(int32_t{});
                break;
            case LaunchScalarType::I64:
                value = load(int64_t{});
                break;
            case LaunchScalarType::U32:
            case LaunchScalarType::F32:
            default:
                value = load(uint32_t{});
                break;
            }
        }
        values.push_back(value);
    }

    const auto axes = [](const std::array<uint32_t, 3>& dims) {
        return std::vector<int64_t>(dims.begin(), dims.end());
    };
    return {{std::string(detail::SAVED_LAUNCH_GRID), axes(launch.grid)},
            {std::string(detail::SAVED_LAUNCH_BLOCK), axes(launch.block)},
            {std::string(detail::SAVED_LAUNCH_SHARED_MEM_BYTES),
             static_cast<int64_t>(launch.sharedMemBytes)},
            {std::string(detail::SAVED_LAUNCH_WORKSPACE_BYTES),
             static_cast<int64_t>(launch.workspaceBytes)},
            {std::string(detail::SAVED_LAUNCH_ARGUMENT_KINDS), std::move(kinds)},
            {std::string(detail::SAVED_LAUNCH_ARGUMENT_SCALAR_TYPES), std::move(types)},
            {std::string(detail::SAVED_LAUNCH_ARGUMENT_VALUES), std::move(values)}};
}

/// The launch saveResolvedLaunch() stored in @p values. Every value is checked against the
/// rule resolve() applies, so a restored launch is one resolve() could have produced.
/// @throws HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE) on a missing, unknown
///         or mistyped key, list lengths that disagree, or a value outside its rule.
inline ResolvedLaunch restoreResolvedLaunch(const MetadataValues& values)
{
    for(const auto& [key, value] : values)
    {
        if(std::find(detail::SAVED_LAUNCH_KEYS.begin(), detail::SAVED_LAUNCH_KEYS.end(), key)
           == detail::SAVED_LAUNCH_KEYS.end())
        {
            detail::failRestoredLaunch("unknown key '" + key + "'");
        }
    }
    const auto find = [&values](std::string_view key) -> const MetadataValue& {
        const auto it = values.find(std::string(key));
        if(it == values.end())
        {
            detail::failRestoredLaunch("missing key '" + std::string(key) + "'");
        }
        return it->second;
    };
    const auto list = [&find](std::string_view key) -> const std::vector<int64_t>& {
        const auto* held = std::get_if<std::vector<int64_t>>(&find(key));
        if(held == nullptr)
        {
            detail::failRestoredLaunch("'" + std::string(key) + "' is not an integer list");
        }
        return *held;
    };
    const auto checked = [](std::string_view what, int64_t value, detail::LaunchValueRule rule) {
        if(!detail::checkLaunchValue(rule, jsonexpr::Value(value)).has_value())
        {
            detail::failRestoredLaunch(std::string(what) + " is " + std::to_string(value)
                                       + ", but it must be " + detail::describeLaunchRule(rule));
        }
        return value;
    };
    const auto scalar = [&find, &checked](std::string_view key, detail::LaunchValueRule rule) {
        const auto* held = std::get_if<int64_t>(&find(key));
        if(held == nullptr)
        {
            detail::failRestoredLaunch("'" + std::string(key) + "' is not an integer");
        }
        return checked(key, *held, rule);
    };
    const auto axes = [&list, &checked](std::string_view key) {
        const auto& held = list(key);
        if(held.size() != 3)
        {
            detail::failRestoredLaunch("'" + std::string(key) + "' has "
                                       + std::to_string(held.size()) + " entries, not 3");
        }
        std::array<uint32_t, 3> result{};
        for(size_t axis = 0; axis < result.size(); ++axis)
        {
            result[axis] = static_cast<uint32_t>(
                checked(key, held[axis], detail::LaunchValueRule::GRID_OR_BLOCK_AXIS));
        }
        return result;
    };

    ResolvedLaunch launch;
    launch.grid = axes(detail::SAVED_LAUNCH_GRID);
    launch.block = axes(detail::SAVED_LAUNCH_BLOCK);
    launch.sharedMemBytes = static_cast<uint32_t>(
        scalar(detail::SAVED_LAUNCH_SHARED_MEM_BYTES, detail::LaunchValueRule::SHARED_MEM_BYTES));
    launch.workspaceBytes = static_cast<size_t>(
        scalar(detail::SAVED_LAUNCH_WORKSPACE_BYTES, detail::LaunchValueRule::WORKSPACE_BYTES));

    const auto& kinds = list(detail::SAVED_LAUNCH_ARGUMENT_KINDS);
    const auto& types = list(detail::SAVED_LAUNCH_ARGUMENT_SCALAR_TYPES);
    const auto& payload = list(detail::SAVED_LAUNCH_ARGUMENT_VALUES);
    if(types.size() != kinds.size() || payload.size() != kinds.size())
    {
        detail::failRestoredLaunch("the argument lists have " + std::to_string(kinds.size()) + ", "
                                   + std::to_string(types.size()) + " and "
                                   + std::to_string(payload.size()) + " entries");
    }

    launch.arguments.reserve(kinds.size());
    for(size_t index = 0; index < kinds.size(); ++index)
    {
        const std::string what = "argument " + std::to_string(index);
        auto& argument = launch.arguments.emplace_back();
        if(kinds[index] < static_cast<int64_t>(LaunchArgumentKind::POINTER)
           || kinds[index] > static_cast<int64_t>(LaunchArgumentKind::SCALAR))
        {
            detail::failRestoredLaunch(what + " has unknown kind " + std::to_string(kinds[index]));
        }
        argument.kind = static_cast<LaunchArgumentKind>(kinds[index]);
        if(argument.kind != LaunchArgumentKind::SCALAR)
        {
            if(types[index] != 0)
            {
                detail::failRestoredLaunch(what + " is not a scalar but records scalar type "
                                           + std::to_string(types[index]));
            }
            if(argument.kind == LaunchArgumentKind::POINTER)
            {
                argument.tensorUid = payload[index];
            }
            else if(payload[index] != 0)
            {
                detail::failRestoredLaunch(what + " is the workspace but records value "
                                           + std::to_string(payload[index]));
            }
            continue;
        }

        if(types[index] < static_cast<int64_t>(LaunchScalarType::I32)
           || types[index] > static_cast<int64_t>(LaunchScalarType::F32))
        {
            detail::failRestoredLaunch(what + " has unknown scalar type "
                                       + std::to_string(types[index]));
        }
        argument.scalarType = static_cast<LaunchScalarType>(types[index]);
        detail::LaunchNumber number;
        if(argument.scalarType == LaunchScalarType::F32)
        {
            const auto bits = static_cast<uint32_t>(
                checked(what, payload[index], detail::LaunchValueRule::U32));
            std::memcpy(&number.real, &bits, sizeof(bits));
            if(!std::isfinite(number.real))
            {
                detail::failRestoredLaunch(what + " holds a float that is not finite");
            }
        }
        else
        {
            number.integer
                = checked(what, payload[index], detail::launchScalarRule(argument.scalarType));
        }
        detail::writeLaunchScalar(argument, number);
    }
    return launch;
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

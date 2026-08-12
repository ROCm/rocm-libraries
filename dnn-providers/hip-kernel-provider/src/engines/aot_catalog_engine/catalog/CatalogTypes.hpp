// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Generic, op-agnostic data model for the AOT kernel catalog. Forked from PR
// #9207's dispatcher/AotInstance.hpp and de-SDPA-monomorphized: the SDPA-named
// CompileSpec/AotInstance/SdpaProblem are replaced by a generic ProblemShape
// (shape-key map) plus KernelEntry/Family so a single matcher serves every op.
//
// The launch/grid/arg-ABI types (GridFormula, KernelArgument, LaunchBindings,
// LaunchMetadata, ...) are ported near-verbatim -- they were already op-agnostic
// and drive LaunchAbi.{hpp,cpp}.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace aot_catalog_engine::catalog
{

// -----------------------------------------------------------------------------
// Problem shape + applicability constraints
// -----------------------------------------------------------------------------

// One value in a decoded problem shape or a constraint operand. bool/int64/
// double/string cover every JSON scalar a kernel author writes and every shape
// key an op adapter decodes (e.g. "M"->4096, "dtype"->"f16", "causal"->true).
using ShapeValue = std::variant<bool, int64_t, double, std::string>;

// A decoded problem: shape-key -> value. Produced by the op adapter, consumed
// by Selection::satisfies against each kernel's constraints.
using ProblemShape = std::map<std::string, ShapeValue>;

// Byte width of one element for the problem's "dtype" token, used to inject the
// `elem_size` symbol into workspace-expression evaluation. Returns nullopt when
// the problem has no string "dtype" key or the token is unmapped -- a workspace
// expression that then references `elem_size` fails closed as "undefined
// symbol" rather than silently using a wrong width. This is intentionally a
// separate table from scalarTypeSizeBytes/argSizeBytes, which size kernarg
// slots (ptr/i32/i64), NOT element dtypes.
inline std::optional<int64_t> elementSizeBytes(const ProblemShape& problem)
{
    auto it = problem.find("dtype");
    if(it == problem.end() || !std::holds_alternative<std::string>(it->second))
    {
        return std::nullopt;
    }
    const auto& token = std::get<std::string>(it->second);
    if(token == "f16" || token == "bf16")
    {
        return 2;
    }
    if(token == "f8" || token == "bf8" || token == "f8fnuz" || token == "bf8fnuz")
    {
        return 1;
    }
    if(token == "f32")
    {
        return 4;
    }
    return std::nullopt;
}

// Applicability predicate for a single shape key (Tier A + simple Tier B). All
// present sub-predicates must hold. An all-empty rule matches nothing (fail
// closed) -- mirrors PR #9207's AttributeRule::empty() semantics.
struct ConstraintRule
{
    std::optional<ShapeValue> equals;
    std::optional<ShapeValue> notEquals;
    std::vector<ShapeValue> oneOf;
    std::optional<int64_t> min; // inclusive lower bound (integer keys)
    std::optional<int64_t> max; // inclusive upper bound (integer keys)
    std::optional<int64_t> multipleOf; // divisibility (integer keys)

    bool empty() const
    {
        return !equals.has_value() && !notEquals.has_value() && oneOf.empty() && !min.has_value()
               && !max.has_value() && !multipleOf.has_value();
    }
};

// shape-key -> predicate. A key absent from the problem but present here fails.
using Constraints = std::map<std::string, ConstraintRule>;

// -----------------------------------------------------------------------------
// Symbolic launch grid
// -----------------------------------------------------------------------------

// Either a named symbol (resolved from the runtime symbol table, e.g. "M") or a
// baked-in literal when `symbol` is unset.
struct GridValue
{
    std::optional<std::string> symbol;
    int64_t literal = 0;
};

enum class GridAxisKind
{
    VALUE, // just `value`
    CEIL_DIV, // (numerator + denominator - 1) / denominator
    FLOOR_DIV, // numerator / denominator
};

// One grid axis. VALUE uses `value`; CEIL_DIV/FLOOR_DIV use numerator/
// denominator; `addend`, when present, is added after the div.
struct GridAxis
{
    GridAxisKind kind = GridAxisKind::VALUE;
    GridValue value;
    GridValue numerator;
    GridValue denominator;
    std::optional<GridValue> addend;
};

struct GridFormula
{
    GridAxis x;
    GridAxis y;
    GridAxis z;
};

// -----------------------------------------------------------------------------
// Workspace-size expression (JSON-as-AST)
// -----------------------------------------------------------------------------

// A kernel's scratch need is a small arithmetic expression over the problem's
// grid symbols (M, N, K, ...) plus the injected `elem_size` symbol, evaluated
// per-problem. The JSON *is* the tree -- no infix parser, so precedence and
// parentheses come for free. A bare integer is a LITERAL, i.e. today's static
// constant is just the degenerate expression (fully back-compatible).
//
// Conditionals are deliberately absent: algorithm branches (split-K vs none,
// im2col vs Winograd) are modeled as separate kernels with their own
// constraints + workspace expr; the selector already branches. See the READMEs.
enum class WsOp
{
    LITERAL, // `literal`
    SYMBOL, // `symbol` resolved from the runtime symbol table
    MUL, // product of args (>= 1)
    ADD, // sum of args (>= 1)
    SUB, // args[0] - args[1] (result must be non-negative)
    CEIL_DIV, // (args[0] + args[1] - 1) / args[1]
    FLOOR_DIV, // args[0] / args[1]
    MIN, // minimum of args (>= 1)
    MAX, // maximum of args (>= 1)
    ALIGN_UP, // ceil_div(args[0], args[1]) * args[1]
};

// Recursive expression node, mirroring GridValue/GridAxis. LITERAL uses
// `literal`; SYMBOL uses `symbol`; every operator uses `args`.
struct WorkspaceExpr
{
    WsOp op = WsOp::LITERAL;
    int64_t literal = 0;
    std::string symbol;
    std::vector<WorkspaceExpr> args;
};

// -----------------------------------------------------------------------------
// Kernel argument ABI
// -----------------------------------------------------------------------------

enum class ArgKind
{
    POINTER,
    SCALAR,
};

enum class ScalarType
{
    F32,
    I32,
    I64,
};

inline uint32_t scalarTypeSizeBytes(ScalarType type)
{
    switch(type)
    {
    case ScalarType::F32:
    case ScalarType::I32:
        return 4;
    case ScalarType::I64:
        return 8;
    default:
        return 0;
    }
}

// One entry in a kernel's args_signature. `scalarType` is set iff kind==SCALAR.
struct KernelArgument
{
    std::string name;
    ArgKind kind = ArgKind::POINTER;
    std::optional<ScalarType> scalarType;
};

// Size (== natural alignment) of an argument in the packed kernarg buffer:
// pointers are 8 bytes, scalars use their type width.
inline uint32_t argSizeBytes(const KernelArgument& arg)
{
    if(arg.kind == ArgKind::POINTER)
    {
        return 8;
    }
    return arg.scalarType.has_value() ? scalarTypeSizeBytes(*arg.scalarType) : 0;
}

// -----------------------------------------------------------------------------
// Runtime launch bindings + metadata
// -----------------------------------------------------------------------------

// A concrete argument value: a raw pointer (uint64_t) or a typed scalar. The
// alternative selected must agree with the KernelArgument's kind/scalarType.
using ScalarValue = std::variant<uint64_t, int64_t, float>;

// Everything the op adapter resolves from the graph, keyed by argument name so
// LaunchAbi::bindArgs can match it against the args_signature.
struct LaunchBindings
{
    // arg name -> device-buffer uid (resolved to a pointer at execute time).
    std::map<std::string, int64_t> pointerUids;
    // arg name -> already-known raw pointer value (null, workspace, ...).
    std::map<std::string, uint64_t> pointerValues;
    // arg name -> scalar value.
    std::map<std::string, ScalarValue> scalars;
};

// Static launch description for one kernel (from family.json).
struct LaunchMetadata
{
    GridFormula grid;
    std::array<uint32_t, 3> block = {1, 1, 1};
    uint32_t sharedMemBytes = 0;
    std::vector<KernelArgument> argsSignature;
};

// -----------------------------------------------------------------------------
// Catalog entities
// -----------------------------------------------------------------------------

// One AOT kernel: an exported symbol in a .co, its applicability predicates,
// workspace need, and launch metadata.
struct KernelEntry
{
    std::string symbol; // exported function name in the .co/HSACO
    std::string coPath; // absolute path to the .co/HSACO on disk
    Constraints constraints; // applicability (Tier A + simple Tier B)
    WorkspaceExpr workspace; // scratch need; defaults to LITERAL 0 (no scratch)
    LaunchMetadata launch;
};

// A family = one family.json = a set of interchangeable kernels for one op kind
// on one arch. `name` is unique and identifies the family to the engine.
struct Family
{
    std::string name;
    std::string opKind; // selects the op adapter ("matmul", ...)
    std::vector<std::string> dtypes; // informational
    std::string arch; // gfx string this family was built for
    std::vector<KernelEntry> kernels;
};

} // namespace aot_catalog_engine::catalog

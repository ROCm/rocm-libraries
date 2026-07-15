// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace rocke_client::dispatcher
{

// A single runtime attribute value, matched against an AotInstance's
// selection constraints. The alternatives mirror the JSON value kinds that the
// rocKE AOT sidecar/instance schema (PR #8866) uses for
// selection.attribute_constraints: booleans (padding_mask, alibi_mask),
// numbers (dropout_probability), and strings (mask_mode, scale_policy).
//
// Two construction pitfalls the kpack catalog parser (and tests) MUST avoid,
// because std::variant matching is alternative-exact:
//   * String literals: AttrValue{"none"} selects the `bool` alternative (a
//     const char* converts to bool, not to std::string), silently yielding
//     `true`. Always wrap string values: AttrValue{std::string("none")}.
//   * Numbers: equality only holds within the same alternative, so a
//     `double` problem value never compares equal to an `int64_t` operand.
//     Widen numeric constraint operands to `double` at parse time to match the
//     values SdpaProblem::attributes() produces (e.g. dropout_probability).
using AttrValue = std::variant<bool, std::int64_t, double, std::string>;

// One selection.attribute_constraints[name] rule. Mirrors the Python contract in
// rocke_client_aot.instance_schema.attributes_match_constraints: any subset of
// {equals, not_equals, one_of} may be present and ALL present operators must hold.
struct AttributeRule
{
    std::optional<AttrValue> equals;
    std::optional<AttrValue> notEquals;
    std::optional<std::vector<AttrValue>> oneOf;

    // A rule with no operator set is malformed: the Python producer rejects it at
    // catalog-parse time (instance_schema.normalize_attribute_constraints raises
    // InstanceError). Selection treats it as unsatisfiable so a malformed instance
    // can never be dispatched; the future catalog parser MUST also reject it.
    bool empty() const
    {
        return !equals.has_value() && !notEquals.has_value() && !oneOf.has_value();
    }
};

// selection.attribute_constraints: attribute name -> rule.
using AttributeConstraints = std::map<std::string, AttributeRule>;

// selection.batch: the inclusive [min, max] batch sizes this prebuilt instance
// serves. Unlike the exact-matched shape keys (seqlen/heads/head_size), batch is
// a runtime launch dimension, so one instance advertises a range and selection
// accepts any problem.batch within it.
struct BatchRange
{
    std::int64_t min = 1;
    std::int64_t max = 1;
};

// The exact kernel build parameters (aot_list.json "compile_spec"). The shape
// fields are matched exactly against a runtime problem; block_size_{q,k} are
// kernel-internal tiling and are NOT part of selection (kept for completeness).
struct CompileSpec
{
    std::string dtype; // I/O element type (Q/K/V/O), e.g. "fp16". Compute/
        // accumulation is fp32 (enforced by the adapter's compute_data_type
        // gate), so it is not a separate key.
    std::string canonicalLayout; // e.g. "BSHD"
    std::int64_t seqlenQ = 0;
    std::int64_t seqlenK = 0;
    std::int64_t numQueryHeads = 0;
    std::int64_t numKvHeads = 0;
    std::int64_t headSize = 0;
    std::int64_t blockSizeQ = 0; // kernel tiling; unused for selection
    std::int64_t blockSizeK = 0; // kernel tiling; unused for selection
    std::string maskMode; // e.g. "none"
};

// Launch metadata copied from the per-arch rocke_client_<arch>.json bundle
// manifest. It is deliberately plain C++ data so selection stays independent of
// any JSON library and execution can pack kernel arguments without reparsing.
struct GridValue
{
    std::optional<std::string> symbol;
    std::int64_t literal = 0;
};

struct GridAxis
{
    enum class Kind
    {
        VALUE,
        CEIL_DIV
    };

    Kind kind = Kind::VALUE;
    GridValue value;
    GridValue numerator;
    GridValue denominator;
};

struct GridFormula
{
    GridAxis x;
    GridAxis y;
    GridAxis z;
};

// Kernel-argument ABI class. Mirrors the sidecar/bundle schema's
// args_signature[].kind enum ("pointer" | "scalar"); parsed once at manifest
// load (AotCatalog parseArgsSignature) so the launch path matches a closed set
// instead of re-comparing strings on every launch, and an unknown kind
// fails closed at load rather than being silently bound as a scalar.
enum class ArgKind
{
    POINTER,
    SCALAR
};

// Scalar argument dtype. The bundle schema's args_signature[].type carries a
// structured string (scalars like "i32"/"f32"; pointers like "ptr<f16, global>"),
// but the launch ABI only ever needs the scalar dtype -- pointers are packed
// uniformly as a 64-bit device address regardless of pointee type. So only the
// scalar dtype is modelled here; it is parsed once at manifest load and an
// unknown dtype fails closed there (AotCatalog parseScalarType).
enum class ScalarType
{
    F32,
    I32,
    I64
};

// ABI width of a scalar dtype in bytes; alignment is natural (== width).
inline std::size_t scalarTypeSizeBytes(ScalarType type)
{
    return type == ScalarType::I64 ? sizeof(std::int64_t) : sizeof(std::int32_t);
}

struct KernelArgument
{
    std::string name;
    ArgKind kind = ArgKind::SCALAR;
    // Set iff kind == SCALAR; pointers carry no dtype (packed as a raw address).
    std::optional<ScalarType> scalarType;
};

// ABI width of a packed argument in bytes (pointer = 8, scalar = dtype width);
// alignment equals this width for every supported argument type.
inline std::size_t argSizeBytes(const KernelArgument& arg)
{
    return arg.kind == ArgKind::POINTER ? sizeof(std::uint64_t)
                                        : scalarTypeSizeBytes(arg.scalarType.value());
}

// A concrete kernel-argument value: device pointer (as an integer), signed
// integer, or float. The variant alternative selected must match the argument's
// declared kind/type in the launch signature (see launch::packArgs).
using ScalarValue = std::variant<std::uint64_t, std::int64_t, float>;

// Op-agnostic launch bindings: the concrete per-launch values an op adapter
// derives from the graph, keyed by the argument names the kernel's ABI
// (args_signature) already declares. launch::bindArgs() resolves each signature
// argument through this table -- pointerUids[name] -> tensor uid (turned into a
// device address at launch), scalars[name] -> packed scalar. A signature name
// absent here is a fail-closed error, so no launch runs with a wrong buffer.
struct LaunchBindings
{
    std::unordered_map<std::string, std::int64_t> pointerUids;
    std::unordered_map<std::string, ScalarValue> scalars;
};

struct LaunchMetadata
{
    GridFormula grid;
    std::array<unsigned int, 3> block = {1, 1, 1};
    std::size_t sharedMemBytes = 0;
    std::vector<KernelArgument> argsSignature;
};

struct AotRuntimeMetadata
{
    std::string cacheKey;
    std::string tocKey;
    std::string symbol;
    std::string kpackPath;
    LaunchMetadata launch;
};

// One installed, AOT-built kernel instance parsed from a per-arch kpack bundle
// manifest. Selection uses the compile/constraint fields; plan construction uses
// runtime metadata to fetch the HSACO from kpack and launch it.
struct AotInstance
{
    std::string name; // unique within a catalog
    std::string op; // "sdpa_fwd"
    std::string family; // "fmha_fwd_mfma"
    std::string arch; // "gfx942"
    CompileSpec compileSpec;
    BatchRange batch;
    AttributeConstraints attributeConstraints;
    AotRuntimeMetadata runtime;
};

} // namespace rocke_client::dispatcher

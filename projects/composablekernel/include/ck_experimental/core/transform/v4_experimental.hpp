// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/// @file v4_experimental.hpp
/// @brief Experimental V4 -- AoS Pool of 16-byte Slots, Schema-as-view,
///        unified Kind enum, hardened slot ids, V3-style splice.
///
/// V4 succeeds V3 by addressing three pain points: generic walking,
/// per-Impl burden, and int-only coordinates. The key design moves:
///
///   1. Schema bytes live in a graph-level Pool of uniform 16-byte Slots
///      (8-byte payload + 8-byte flags). Per-Impl Schemas become typed
///      VIEWS over a (base_offset, count) slice of the pool. Framework
///      walks the pool generically (splice, NTTP equality, validation)
///      without per-Impl hooks.
///
///   2. Slot is a single canonical type with a 64-bit payload. Any value
///      that fits in 8 bytes (bool, int, float, etc.) is stored directly;
///      narrower values are zero-extended into the payload.
///
///   3. Kind enum tells the reader how to obtain the value from the
///      payload bytes: directly (VALUE), by indirecting through another
///      store (BINDING_ID / EDGE_ID), or by computing it at Phase A.5
///      (DERIVED).
///
///   4. Slot index ids are typed `IndexT` (presently uint8_t; widen the
///      single alias to grow capacity in lockstep). Hardened factories
///      (Slot::from_edge_id(IndexT) / Slot::from_binding_id(IndexT)) are
///      the only construction paths and range-check at construction.
///      Reads do NOT mask defensively.
///
/// V4 vocabulary (cleanup vs V3):
///   - "edge"    = topology connection between transforms (Phase A/B metadata).
///                 An edge has an id (uint8_t), an anchor (Value), and a
///                 resolved length (int).
///   - "coord"   = runtime coordinate value flowing through transforms (Phase C).
///   - V3's `slot_lens[]`     -> V4's `edge_lengths[]`
///   - V3's `slot_values[]`   -> V4's `input_edge_anchors / output_edge_anchors[]`
///   - V3's `Kind::SLOT`      -> V4's `Kind::EDGE_ID`
///   - V3's `make_slot(N)`    -> V4's `Slot::from_edge_id(N)`
///   - V3's mapCoord input/output args renamed to input_coords/output_coords.
///
/// Per-Impl surface (everything an Impl author writes):
///   1. Schema           -- POD struct of pool-backed fields plus a
///                          `using members = detail::MemberPtrList<&Schema::a, ...>`
///                          alias that drives Phase B Schema construction
///   2. State<NIn|NOut>  -- snapshot returned by resolveState; the type the
///                          per-coord hot loop reads
///   3. resolveState(schema, in_lens, out_lens) -> State
///                       -- Phase B: builds the snapshot once per kernel
///   4. mapCoord(state, output_coords, input_coords)
///                       -- Phase C: hot loop, reads only State
///   5. (optional) deriveInputLength / deriveOutputLength
///                       -- required when the factory declares a
///                          Slot::from_derived() anchor on that side
///
/// The Phase A/B/C pipeline is unchanged from V3 in shape:
///   Phase A: graph construction + edge anchoring (Value layout)
///   Phase B: per-transform resolveState() once per kernel (NTTP-baked)
///   Phase C: per-coord mapCoord() many times (NTTP-dispatched fold)
///
/// See experiments/transform_graph/V4_DESIGN_PLAN.md for the full design
/// rationale, locked Q1-Q6 answers, and C++ Expert hardenings list.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/container/static_array.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_experimental/core/transform/magic_division.hpp"

#include <cstdint>

// TODO(hipRTC): host-only static_assert; need ck_tile drop-in for
// std::has_unique_object_representations_v in device translation units.
#include <type_traits>

namespace ck_tile::core::transform::v4 {

// =============================================================================
// 0. Capacity constants and global sentinels
// =============================================================================

inline constexpr ck_tile::index_t MAX_TENSOR_DIMS_V4   = 64;
inline constexpr ck_tile::index_t MAX_TRANSFORMS_V4    = 12;

// B5 cap: per-transform State must fit in this byte budget so that the
// GraphBindings tuple<States...> aggregate gets SROA-promoted to scalar
// registers in the hot mapCoord path. If a per-Impl State exceeds this,
// either raise MAX_STATE_BYTES with documented justification (per plan
// sec 5.7 #14) or cut a State member.
//
// Raised 128 -> 768 to support the full MAX_TENSOR_DIMS_V4=64 arity for EVERY
// transform, including MERGE whose State (derived_strides[N] + magic_divs[N] =
// 12 B/dim) is 768 B at N=64. This is a capability requirement, not a perf
// claim: a 64-dim MERGE State is ~192 VGPRs and WILL spill to scratch in the
// hot path -- accepted for the extreme arity. EMBED/UNMERGE at 64 = 256 B.
// MAX_STATE_BYTES sizes nothing at runtime; it is only this SROA tripwire.
inline constexpr ck_tile::index_t MAX_STATE_BYTES      = 768;

// Type used for slot index payloads (BINDING_ID, EDGE_ID, SCHEMA_SCALAR_ID,
// SCHEMA_ARRAY_ID kinds). Widening this alias is the single point of
// control for growing every index-keyed resource (bindings, edges, schema
// member indices) in lockstep. All index-domain capacity constants and
// sentinels below derive from this.
using IndexT = uint8_t;

// Sentinels: reserved values that mean "no such index" / "uninitialised".
// Defined as numeric_limits::max() of IndexT; range checks use the
// MAX_*_V4 capacities below, which are strictly less than the sentinels.
inline constexpr IndexT INVALID_BINDING_ID = std::numeric_limits<IndexT>::max();
inline constexpr IndexT INVALID_EDGE_ID    = std::numeric_limits<IndexT>::max();

// Capacities: highest legal id + 1. One less than the sentinel value so a
// range check `if (id >= MAX_*) reject(...)` rejects the sentinel
// naturally. Typed as IndexT (not ck_tile::index_t) so widening IndexT
// keeps comparisons clean across signed/unsigned promotion.
inline constexpr IndexT MAX_BINDINGS_V4 = std::numeric_limits<IndexT>::max() - 1;
inline constexpr IndexT MAX_EDGES_V4    = std::numeric_limits<IndexT>::max() - 1;

static_assert(MAX_BINDINGS_V4 < INVALID_BINDING_ID,
              "Capacity must be strictly less than sentinel value");
static_assert(MAX_EDGES_V4 < INVALID_EDGE_ID,
              "Capacity must be strictly less than sentinel value");

// Pool element count -- highest legal pool index + 1, derived from IndexT
// like MAX_BINDINGS_V4 / MAX_EDGES_V4 so every index-domain capacity grows in
// lockstep when IndexT is widened (and the pool counters can always represent
// it). With IndexT == uint8_t this is 254 slots (254 * 16-byte Slots ~ 4 KB).
// Pool layout is static_array<Slot, N>; Slot is a uniform 16 bytes, so N*16 is
// naturally 8-aligned for any N -- no separate alignment constraint needed.
inline constexpr IndexT MAX_POOL_VALUES_V4 = std::numeric_limits<IndexT>::max() - 1;

static_assert(MAX_POOL_VALUES_V4 < std::numeric_limits<IndexT>::max(),
              "Capacity must be strictly less than sentinel value");


// =============================================================================
// 1. Diagnostic stubs (forward-declared) -- B7 split into two catalogs
// =============================================================================
//
// compile_time_diag::*    -- consteval-fail named stubs. When constexpr
//                            evaluation reaches one of these calls, the
//                            compiler error names the stub at the user's
//                            call site, giving a precisely-named diagnostic
//                            instead of a sea of template instantiations.
//                            Same pattern as V3.
//
// runtime_invariant::*    -- runtime hot-path defenses. No-op on device
//                            (UB if violated); host build can assert. The
//                            __builtin_unreachable() hint inside lets the
//                            optimizer assume the invariant holds.
//
// Width/shape mismatches that fire at template instantiation (not constexpr
// eval) use static_assert with a custom message instead -- in those cases
// no stub is needed because the static_assert message is the diagnostic.

namespace compile_time_diag {

// Pool slot id / binding id range checks fire at the slot factories (constexpr).
CK_TILE_HOST_DEVICE void edgeIdOutOfRange() {}
CK_TILE_HOST_DEVICE void bindingIdOutOfRange() {}

// poolAliasingViolation is retained as documentation of the
// monotonic-pool-used invariant enforced structurally by insertTransform /
// spliceInto. The stub is unreferenced today (the invariant cannot be
// violated by either writer; a runtime per-slot check was dropped after
// 3-expert review). If a future writer ever takes a slot_index argument
// directly (i.e. departs from append-only), re-add the per-slot check at
// that writer and call this stub on conflict so the failure is grep-able
// by the efail driver.
CK_TILE_HOST_DEVICE void poolAliasingViolation() {}

/// @brief Fired by make_graph_bindings' sanity-check pass when a Schema field
///        used as an edge anchor source disagrees with the resolved edge
///        length stored in EdgeLengths. Indicates a topology conflict --
///        two writers anchored the same edge to different values, and
///        first-writer-wins resolution silently picked one.
CK_TILE_HOST_DEVICE void schemaEdgeLengthConflict() {}

// Graph-validation diagnostics (Phase A consteval validation).
CK_TILE_HOST_DEVICE void graphValidationErrorUnresolvedEdgeLength() {}
CK_TILE_HOST_DEVICE void graphValidationErrorEdgeMultiplyWritten() {}
CK_TILE_HOST_DEVICE void graphValidationErrorOutputEdgeHasNoProducer() {}
CK_TILE_HOST_DEVICE void graphValidationErrorTransformCycle() {}
CK_TILE_HOST_DEVICE void graphValidationErrorTooManyTransforms() {}
CK_TILE_HOST_DEVICE void graphValidationErrorEdgeIdOutOfRange() {}
CK_TILE_HOST_DEVICE void graphValidationErrorSpliceTransformOverflow() {}
CK_TILE_HOST_DEVICE void graphValidationErrorSpliceEdgeOverflow() {}

// Slot typed-read diagnostics (fire during constant evaluation; the read is a
// compile error if the stored value's domain and the requested T's domain
// disagree -- e.g. a float bit-pattern read as an integer).
CK_TILE_HOST_DEVICE void slotReadValueDomainMismatch() {}

// Per-Impl factory + splice consteval diagnostics.
CK_TILE_HOST_DEVICE void transformErrorBroadcastWriteNotEmpty() {}
CK_TILE_HOST_DEVICE void transformErrorSliceBeginAfterEnd() {}
CK_TILE_HOST_DEVICE void transformErrorEmbedZeroLength() {}
CK_TILE_HOST_DEVICE void transformErrorReadCountArityMismatch() {}
CK_TILE_HOST_DEVICE void transformErrorWriteCountArityMismatch() {}

}   // namespace compile_time_diag

namespace runtime_invariant {

// Runtime hot-path defenses. The body is empty (no-op on device, UB if
// violated); the __builtin_unreachable lets the optimizer assume the check
// passes when the inputs are within the expected range.

[[noreturn]] CK_TILE_HOST_DEVICE void mergeOverflow() noexcept
{
    __builtin_unreachable();
}

[[noreturn]] CK_TILE_HOST_DEVICE void valueReadFromUnusedSlot() noexcept
{
    __builtin_unreachable();
}

[[noreturn]] CK_TILE_HOST_DEVICE void valueReadFromDerivedMarker() noexcept
{
    __builtin_unreachable();
}

[[noreturn]] CK_TILE_HOST_DEVICE void schemaOffsetOutOfRange() noexcept
{
    __builtin_unreachable();
}

}   // namespace runtime_invariant

// Bring stub names into ::detail scope so existing call sites continue to
// compile without source-level updates. New code SHOULD prefer the explicit
// `compile_time_diag::X` / `runtime_invariant::X` form to make the gate
// classification visible at the call site.
using compile_time_diag::edgeIdOutOfRange;
using compile_time_diag::bindingIdOutOfRange;
using compile_time_diag::poolAliasingViolation;
using compile_time_diag::schemaEdgeLengthConflict;
using compile_time_diag::slotReadValueDomainMismatch;
using compile_time_diag::graphValidationErrorUnresolvedEdgeLength;
using compile_time_diag::graphValidationErrorEdgeMultiplyWritten;
using compile_time_diag::graphValidationErrorOutputEdgeHasNoProducer;
using compile_time_diag::graphValidationErrorTransformCycle;
using compile_time_diag::graphValidationErrorTooManyTransforms;
using compile_time_diag::graphValidationErrorEdgeIdOutOfRange;
using compile_time_diag::graphValidationErrorSpliceTransformOverflow;
using compile_time_diag::graphValidationErrorSpliceEdgeOverflow;
using compile_time_diag::transformErrorBroadcastWriteNotEmpty;
using compile_time_diag::transformErrorSliceBeginAfterEnd;
using compile_time_diag::transformErrorEmbedZeroLength;
using compile_time_diag::transformErrorReadCountArityMismatch;
using compile_time_diag::transformErrorWriteCountArityMismatch;
using runtime_invariant::mergeOverflow;
using runtime_invariant::valueReadFromUnusedSlot;
using runtime_invariant::valueReadFromDerivedMarker;
using runtime_invariant::schemaOffsetOutOfRange;


// =============================================================================
// 2. Kind -- interpretation tag for a slot
// =============================================================================

/// @brief Slot interpretation tag. Tells the reader how to obtain the
///        slot's value from the payload bytes (directly, by indirecting
///        through another store, or by computing it).
///
/// Kind does NOT encode payload type -- payload bits are interpreted by
/// the per-Impl Schema accessor via bit_cast<T>. Adding a new payload
/// type (e.g. bfloat16) requires zero Kind changes; only a new typed
/// accessor.
enum struct Kind : uint8_t
{
    UNUSED      = 0,   ///< Empty slot; payload must be 0. Default-init state.
    VALUE       = 1,   ///< Value is the payload bytes themselves (bit_cast'd to T).
    BINDING_ID  = 2,   ///< Payload is an index ID into RuntimeBindings; value is RB[id].
    EDGE_ID     = 3,   ///< Payload is an index ID into EdgeLengths; value is EL[id].
    DERIVED     = 4,   ///< Value is the result of the owning Impl's deriveInputLength / deriveOutputLength.
};

/**
 * @brief Tag identifying the source value type stored in a Slot's payload.
 *
 * Each enumerator names a concrete arithmetic type (width + signedness, or
 * floating-point width). Two adjacent enumerators with the same family but
 * different widths are distinct values -- writing as one width and reading
 * as another is a category error, not a silent reinterpretation.
 *
 * NONE means the slot does not currently hold a typed value (Kind is not
 * VALUE).
 */
enum struct ValueType : uint8_t
{
    NONE = 0,    ///< Slot does not hold a typed value.
    I8   = 1,    ///< Signed   8-bit integer, sign-extended into payload.
    I16  = 2,    ///< Signed  16-bit integer, sign-extended into payload.
    I32  = 3,    ///< Signed  32-bit integer, sign-extended into payload.
    I64  = 4,    ///< Signed  64-bit integer, occupies full payload.
    U8   = 5,    ///< Unsigned 8-bit integer, zero-extended into payload.
    U16  = 6,    ///< Unsigned 16-bit integer, zero-extended into payload.
    U32  = 7,    ///< Unsigned 32-bit integer, zero-extended into payload.
    U64  = 8,    ///< Unsigned 64-bit integer, occupies full payload.
    F32  = 9,    ///< 32-bit IEEE-754 float, low 4 bytes of payload.
    F64  = 10,   ///< 64-bit IEEE-754 double, occupies full payload.
    BOOL = 11,   ///< C++ bool, stored as 1 byte (0 or 1) bit-padded.
};

namespace detail {

/// @brief `false` as a value-dependent expression, for use in the final
///        `else` of `if constexpr` ladders to delay diagnostic to the
///        actually-invalid specialisation.
template <typename> inline constexpr bool dependent_false_v = false;

/// @brief Trailing padding bytes for PaddingHelper. The N == 0 specialization
///        is empty (no member), so PaddingHelper avoids a zero-length array
///        when the value already fills the block.
template <size_t N>
struct TrailingPad
{
    uint8_t bytes[N] = {};
    constexpr bool operator==(const TrailingPad&) const = default;
};
template <>
struct TrailingPad<0>
{
    constexpr bool operator==(const TrailingPad&) const = default;
};

/**
 * @brief Pad a narrow value to a fixed block width for bit_cast'ing into a
 *        same-sized envelope. `Width` is the target byte count. The trailing
 *        pad is carried by TrailingPad and elided via [[no_unique_address]]
 *        when no padding is needed, keeping `sizeof == Width` with no
 *        zero-length array.
 */
template <typename T, size_t Width>
struct PaddingHelper
{
    static_assert(sizeof(T) <= Width,
                  "PaddingHelper<T, Width>: sizeof(T) must fit in Width");
    T                                          value;
    [[no_unique_address]] TrailingPad<Width - sizeof(T)> _pad{};
    constexpr bool operator==(const PaddingHelper&) const = default;
};

/**
 * @brief Map a C++ arithmetic type to its ValueType tag (the closed set of
 *        supported payload types). The primary template maps any unsupported
 *        type to ValueType::NONE; the 11 specializations below cover bool, the
 *        signed/unsigned integers, and float/double.
 *
 * Specialization-based rather than an `if constexpr(is_same_v<T, ...>)` ladder:
 * each lookup is a single specialization match instead of up to 11 is_same_v
 * evaluations re-walked at every instantiating T (value_type_of is named by
 * as_value / from_value / set_value / adjust_precision / normalize_binding, so
 * the saving compounds across the typed-slot surface). It is the single source
 * of truth for the supported set: has_value_type_v derives from it, and the
 * authoring sites (Slot::from_value / Slot::set_value) static_assert
 * has_value_type_v where a mapping is required.
 */
template <typename T> struct value_type_for          { static constexpr ValueType value = ValueType::NONE; };
template <> struct value_type_for<bool>              { static constexpr ValueType value = ValueType::BOOL; };
template <> struct value_type_for<int8_t>            { static constexpr ValueType value = ValueType::I8;   };
template <> struct value_type_for<int16_t>           { static constexpr ValueType value = ValueType::I16;  };
template <> struct value_type_for<int32_t>           { static constexpr ValueType value = ValueType::I32;  };
template <> struct value_type_for<int64_t>           { static constexpr ValueType value = ValueType::I64;  };
template <> struct value_type_for<uint8_t>           { static constexpr ValueType value = ValueType::U8;   };
template <> struct value_type_for<uint16_t>          { static constexpr ValueType value = ValueType::U16;  };
template <> struct value_type_for<uint32_t>          { static constexpr ValueType value = ValueType::U32;  };
template <> struct value_type_for<uint64_t>          { static constexpr ValueType value = ValueType::U64;  };
template <> struct value_type_for<float>             { static constexpr ValueType value = ValueType::F32;  };
template <> struct value_type_for<double>            { static constexpr ValueType value = ValueType::F64;  };

template <typename T>
constexpr ValueType value_type_of() noexcept
{
    return value_type_for<T>::value;
}

/// @brief True iff T has a ValueType mapping. Derived from value_type_of (the
///        single source of truth) so the two cannot drift. Lets as_value apply
///        its domain check only for mapped arithmetic Ts while remaining a pure
///        byte reader for any other trivially-copyable T.
template <typename T>
inline constexpr bool has_value_type_v = (value_type_of<T>() != ValueType::NONE);

/// @brief True iff `v` is a floating-point ValueType (F32 / F64).
constexpr bool is_float_value_type(ValueType v) noexcept
{
    return v == ValueType::F32 || v == ValueType::F64;
}

/// @brief True iff `v` is a signed-integer ValueType (I8 / I16 / I32 / I64).
///        Used by Slot::adjust_precision to pick the signed precision family;
///        the unsigned family (U*) and BOOL are not signed.
constexpr bool is_signed_value_type(ValueType v) noexcept
{
    return v == ValueType::I8 || v == ValueType::I16 ||
           v == ValueType::I32 || v == ValueType::I64;
}

/// @brief Domain compatibility for an as_value read. The integer family
///        (I*/U*/BOOL) and the float family (F32/F64) must not be
///        reinterpreted across -- a float bit-pattern read as an integer (or
///        vice versa) is always a bug. Signedness *within* the integer family
///        is intentionally flexible: an `int` literal read into an `unsigned`
///        field (e.g. `dims(8)`) is a legitimate authoring pattern.
constexpr bool same_value_domain(ValueType source, ValueType target) noexcept
{
    return is_float_value_type(source) == is_float_value_type(target);
}

} // namespace detail


// =============================================================================
// 3. Precision policies (Precision32 / Precision64)
// =============================================================================
//
// Precision policy classes supply the arithmetic-type family used to encode
// graph parameters: lengths (Unsigned), strides (Signed), and floating
// values (Float). Per-Impl Schema fields and resolveState pick the right
// family member; the Slot itself stores everything in a single
// canonical 8-byte payload regardless of precision.
//
// MULTI-PRECISION EXTENSION GUIDE
// -------------------------------
// Adding a new precision (e.g. P16 for half/bfloat16):
//
//   1. Declare a Precision struct with four typedefs: Signed, Unsigned,
//      Float, and LengthT (= Unsigned). Widths within one Precision MUST
//      match (sizeof(Signed) == sizeof(Unsigned) == sizeof(Float)).
//   2. The Slot payload is always uint64_t -- no per-Precision payload
//      change is needed. Narrow types pad to 8 bytes via PaddingHelper.
//   3. Add a section 8 make_transform_graph alias if the new Precision
//      should have a user-facing entry point.
//
// LENGTHS / STRIDES TYPE INVARIANT
// --------------------------------
// Tensor LENGTHS (extents, components, padded sizes -- anything that
// counts elements along an axis) are strictly NON-NEGATIVE. They are
// carried as `Precision::Unsigned` throughout the framework: gathered
// into `in_lens` / `out_lens` arrays, returned from
// `Impl::deriveOutputLength` / `deriveInputLength`, and read from edge
// slots via `as_value<typename Precision::Unsigned>()`.
//
// Tensor STRIDES MAY be negative (reversed-axis views, etc.) and are
// carried as `Precision::Signed`. Per-Impl Schema fields that hold
// strides therefore use the signed variant.
//
// Consequence: when an Impl needs to combine the two (e.g. EMBED's
// `(in_len - 1) * stride`), it computes in the type appropriate to the
// semantics of that Impl. EMBED assumes positive strides and stays
// unsigned; any future Impl that intentionally accepts negative strides
// must lift to Signed explicitly.

/// Short alias for the magic-division helpers, which live in the parent
/// `...::transform::detail` (a sibling of this `v4` namespace, in
/// `magic_division.hpp`). Used by the Precision policies (for the per-precision
/// `MagicDiv` constants type) and by MERGE's resolve/mapCoord.
namespace mdv = ::ck_tile::core::transform::detail;

/**
 * @brief 32-bit precision policy. Used for graphs whose lengths /
 *        strides / float values fit in 32 bits.
 *
 * Beyond the width typedefs, the policy carries the precision-dependent facts
 * that consumers would otherwise re-derive per call site:
 *   - `IntAccum`     : an accumulator strictly wider than a length, so a
 *                     right-to-left stride / component-length product can detect
 *                     overflow before it wraps (int64 holds a uint32 product).
 *   - `max_signed_value` : the largest valid tensor-length product (the signed range).
 *   - `IntMagicDiv` : the magic-division constants type for this width (kept as
 *                     `mdv::magic_div_t<Unsigned>` so an unsupported width still
 *                     trips the trait's undefined primary template).
 */
struct Precision32
{
    using Signed   = int32_t;
    using Unsigned = uint32_t;
    using Float    = float;
    using LengthT  = Unsigned; ///< Tensor length/extent type (currently == Unsigned).

    using IntAccum  = int64_t;  ///< Overflow-safe accumulator (double the length width).
    using IntMagicDiv = mdv::magic_div_t<Unsigned>;

    static constexpr IntAccum max_signed_value = static_cast<IntAccum>(std::numeric_limits<Signed>::max());

    constexpr bool operator==(const Precision32&) const = default;
};

/**
 * @brief 64-bit precision policy. Used for graphs whose lengths /
 *        strides / float values require 64 bits (e.g. HipTensor).
 *
 * Mirrors Precision32; `IntAccum` is a 128-bit accumulator so a 64-bit length
 * product is overflow-checked before it can wrap (the int64 accumulator the
 * pre-precision code used could not detect 64-bit product overflow). The
 * `__int128` arithmetic runs only on the constexpr / host graph-build path
 * (resolveState / derive*), never the device mapCoord hot path.
 */
struct Precision64
{
    using Signed   = int64_t;
    using Unsigned = uint64_t;
    using Float    = double;
    using LengthT  = Unsigned; ///< Tensor length/extent type (currently == Unsigned).

    using IntAccum  = __int128; ///< Overflow-safe accumulator (double the length width).
    using IntMagicDiv = mdv::magic_div_t<Unsigned>;

    static constexpr IntAccum max_signed_value = static_cast<IntAccum>(std::numeric_limits<Signed>::max());

    constexpr bool operator==(const Precision64&) const = default;
};

/**
 * @brief Constraint for a numeric-precision policy: the width facts the
 *        framework reads off `precision_t<G.precision>`, plus the width/domain
 *        relations those facts must satisfy.
 *
 * Constraining the `Precision` template parameter (make_transform_graph,
 * TransformImpl, the length helpers) turns a missing/renamed member or a
 * mis-sized type into a crisp "does not satisfy PrecisionPolicy" diagnostic
 * at the use site instead of a deep substitution-failure dump. This concept is
 * the single source of truth for the policy contract: the two static_asserts
 * below enforce it for the built-in policies, so the per-member width
 * static_asserts that used to live here are subsumed. A new precision policy
 * must provide every member and satisfy every relation named here.
 */
template <typename P>
concept PrecisionPolicy = requires {
    typename P::Signed;       // signed index / stride / offset type
    typename P::Unsigned;     // unsigned counterpart (same width as Signed)
    typename P::Float;        // floating value type (same width as Signed)
    typename P::LengthT;      // tensor length / extent type (non-negative)
    typename P::IntAccum;     // overflow-safe product accumulator (>= 2x length width)
    typename P::IntMagicDiv;  // magic-division constants type for this width
    P::max_signed_value;      // largest valid signed length / product (overflow cap)
}
    // Width / domain relations (these subsume the former per-policy
    // static_asserts): Signed/Unsigned/Float share a width, Float is a
    // floating-point type, and IntAccum is at least double the length width so a
    // per-step-checked product of two lengths cannot wrap the accumulator.
    && std::is_floating_point_v<typename P::Float>
    && (sizeof(typename P::Signed) == sizeof(typename P::Unsigned))
    && (sizeof(typename P::Signed) == sizeof(typename P::Float))
    && (sizeof(typename P::IntAccum) >= 2 * sizeof(typename P::LengthT));

static_assert(PrecisionPolicy<Precision32>,
              "Precision32 must satisfy PrecisionPolicy");
static_assert(PrecisionPolicy<Precision64>,
              "Precision64 must satisfy PrecisionPolicy");

/**
 * @brief Numeric-precision selector carried as an NTTP field on the graph
 *        value. The graph stores the tag (not the policy type, which a struct
 *        cannot name from its own data member); the `<auto G>` consumers
 *        recover the policy via `precision_t<G.precision>`.
 */
enum struct PrecisionTag : uint8_t
{
    P32 = 0,
    P64 = 1,
};

namespace detail {

template <PrecisionTag Tag>
struct PolicyForTag;

template <>
struct PolicyForTag<PrecisionTag::P32>
{
    using type = Precision32;
};

template <>
struct PolicyForTag<PrecisionTag::P64>
{
    using type = Precision64;
};

template <typename Precision>
struct PrecisionTagOf
{
    static_assert(dependent_false_v<Precision>,
                  "Unregistered Precision policy. make_transform_graph<P> "
                  "requires P to be Precision32 or Precision64; add a "
                  "PrecisionTagOf specialization to register a new policy "
                  "(see the Multi-Precision Extension Guide above).");
};

template <>
struct PrecisionTagOf<Precision32>
{
    static constexpr PrecisionTag value = PrecisionTag::P32;
};

template <>
struct PrecisionTagOf<Precision64>
{
    static constexpr PrecisionTag value = PrecisionTag::P64;
};

} // namespace detail

/// @brief Precision policy type for a tag value (e.g. `precision_t<G.precision>`).
template <PrecisionTag Tag>
using precision_t = typename detail::PolicyForTag<Tag>::type;

/// @brief Tag value for a Precision policy type (inverse of `precision_t`).
template <typename Precision>
inline constexpr PrecisionTag precision_tag_of_v = detail::PrecisionTagOf<Precision>::value;


// =============================================================================
// 9. Width-tagged placeholder (runtime-bound value dispatch tag)
// =============================================================================

/// @brief Placeholder marker for runtime-bound values.
///
/// SlotT-agnostic under the graph-owned-SlotT design: the binding-id storage
/// width is decided at `insertTransform` time from the graph's declared
/// `SlotT` (via `make_transform_graph<SlotT>(...)`), not from the placeholder.
/// `Slot::from_raw(placeholder<Id>)` reads only `::ID` and stores it as a
/// `BINDING_ID` slot in the graph's pool with the graph's `SlotT::PayloadT`
/// width.
template <ck_tile::index_t Id>
struct placeholder
{
    static constexpr ck_tile::index_t ID = Id;

    static_assert(Id >= 0 && Id < MAX_BINDINGS_V4,
                  "placeholder Id must be in [0, MAX_BINDINGS_V4)");
};

// ---- Helper traits used by Slot::from_raw and TransformArgs commit -----
//
// Defined here so Slot's body can name them.

namespace detail {

/// @brief Detects `placeholder<Id>` for any Id.
template <typename T>           struct is_placeholder              : std::false_type {};
template <ck_tile::index_t Id>
struct is_placeholder<placeholder<Id>>                             : std::true_type  {};
template <typename T> inline constexpr bool is_placeholder_v = is_placeholder<T>::value;

// ---- Schema member-pointer NTTP infrastructure -----------------------------
//
// The per-Impl Schema is a POD `struct Schema` with native data members.
// Phase B dispatch references those members via member-pointer NTTPs
// (e.g. `&Schema::lengths`). The traits below extract type / arity
// information from member-pointer NTTPs at consteval.
//
// Per-Impl contract:
//   struct Schema {
//       int32_t scalar_field;
//       int32_t array_field[MAX_N];
//       /// Reflection descriptor: enumerate non-static data members
//       /// in declaration order. The framework walks this list to fill
//       /// the Schema from the Pool at Phase B (make_schema).
//       using members = detail::MemberPtrList<&Schema::scalar_field,
//                                              &Schema::array_field>;
//   };
//   static_assert(std::has_unique_object_representations_v<Schema>,
//                 "Schema must have no padding for NTTP bit-equality");

/// @brief Extract the class type from a pointer-to-data-member type.
///        Works for scalar and array member pointers; the pointed-to type
///        `M` is unconstrained.
template <typename PtrT>
struct member_ptr_class;

template <typename C, typename M>
struct member_ptr_class<M C::*>
{
    using type = C;
};

template <typename PtrT>
using member_ptr_class_t = typename member_ptr_class<PtrT>::type;

/// @brief Extract element type + compile-time extent from a
///        pointer-to-array-member NTTP. e.g. for `int32_t lengths[64]` ->
///        `element_type = int32_t`, `count = 64`.
template <auto MemPtr>
struct member_array_traits;

template <typename C, typename T, ck_tile::index_t N, T (C::*MemPtr)[N]>
struct member_array_traits<MemPtr>
{
    using element_type                      = T;
    static constexpr ck_tile::index_t count = N;
};

/// @brief Extract value type from a pointer-to-scalar-member NTTP.
///        e.g. for `int32_t ndim` -> `type = int32_t`.
template <auto MemPtr>
struct member_scalar_traits;

template <typename C, typename T, T C::*MemPtr>
struct member_scalar_traits<MemPtr>
{
    using type = T;
};

/// @brief Detect whether `MemPtr` points to an array data member.
template <auto MemPtr>
struct is_array_member : std::false_type {};

template <typename C, typename T, ck_tile::index_t N, T (C::*MemPtr)[N]>
struct is_array_member<MemPtr> : std::true_type {};

template <auto MemPtr>
inline constexpr bool is_array_member_v = is_array_member<MemPtr>::value;

/// @brief Compile-time list of (heterogeneous) member pointers. Each
///        per-Impl Schema publishes its non-static data members in
///        declaration order via `using members = MemberPtrList<...>`.
template <auto... Ptrs>
struct MemberPtrList
{
    static constexpr ck_tile::index_t count = sizeof...(Ptrs);
};

/// @brief Read the Nth member pointer from a MemberPtrList. Used by
///        Phase-B dispatch (`tryArrayMatchByMemPtr`) and by the Pool walker.
template <ck_tile::index_t P, typename List>
struct nth_member;

template <auto First, auto... Rest>
struct nth_member<0, MemberPtrList<First, Rest...>>
{
    static constexpr auto value = First;
};

template <ck_tile::index_t P, auto First, auto... Rest>
struct nth_member<P, MemberPtrList<First, Rest...>>
{
    static constexpr auto value = nth_member<P - 1, MemberPtrList<Rest...>>::value;
};

template <ck_tile::index_t P, typename List>
inline constexpr auto nth_member_v = nth_member<P, List>::value;

/**
 * @brief Compare two NTTPs for value-equality. The partial specialization
 *        matches only when both A and B are the SAME NTTP value (which
 *        requires same type, per C++20 NTTP rules). Cross-type comparisons
 *        cannot match the specialization and fall through to false_type.
 *
 *        Used by `index_of_member` to bulletproof against historical
 *        clang ICEs when consteval lambdas appear inside class-scope
 *        `static constexpr` initializers under heavy NTTP context.
 */
template <auto A, auto B>
struct ptr_eq : std::false_type {};

template <auto A>
struct ptr_eq<A, A> : std::true_type {};

/**
 * @brief Find the index of a member pointer within a MemberPtrList.
 *        Returns -1 if not found. Comparison is type-guarded via `ptr_eq`
 *        -- two member pointers of different declared types (e.g. scalar
 *        vs array) never compare equal even if their underlying bit
 *        representations would match.
 */
template <auto Target, typename List>
struct index_of_member;

template <auto Target>
struct index_of_member<Target, MemberPtrList<>>
{
    static constexpr ck_tile::index_t value = -1;
};

template <auto Target, auto First, auto... Rest>
struct index_of_member<Target, MemberPtrList<First, Rest...>>
{
private:
    static constexpr bool match = ptr_eq<Target, First>::value;
    static constexpr ck_tile::index_t rest =
        index_of_member<Target, MemberPtrList<Rest...>>::value;

public:
    static constexpr ck_tile::index_t value =
        match ? 0 : (rest == -1 ? -1 : 1 + rest);
};

template <auto Target, typename List>
inline constexpr ck_tile::index_t index_of_member_v =
    index_of_member<Target, List>::value;

} // namespace detail

/**
 * @brief Transient discriminated-union value carrying a (kind, value-type)
 *        discriminator pair plus an 8-byte payload. Constructed via the
 *        static factories; decomposed by Pool at insertion (the Pool stores
 *        the bytes directly, so any in-struct padding is irrelevant for the
 *        NTTP guarantee).
 */
struct Slot
{
    /// Fixed payload width.
    using PayloadT = uint64_t;

    /// Re-export of the v4-namespace IndexT for ergonomic Slot::IndexT
    /// access at template-meta consumer sites.
    using IndexT = IndexT;

    /// Two-byte discriminator pair: storage kind and value type.
    struct RawFlags
    {
        Kind      kind  = Kind::UNUSED;
        ValueType vtype = ValueType::NONE;

        constexpr bool operator==(const RawFlags&) const = default;
    };

    static_assert(offsetof(RawFlags, kind) == 0 && offsetof(RawFlags, vtype) == 1,
                  "RawFlags member order is load-bearing for positional aggregate init");

    using Flags = detail::PaddingHelper<RawFlags, sizeof(PayloadT)>;

    // ---- Public data members (aggregate-init order; NTTP-required public) ---
    Flags    flags{};
    PayloadT payload{0};

    constexpr bool operator==(const Slot&) const = default;

    // ---- Public factories (every factory sets BOTH kind AND vtype) ----------

    /**
     * @brief Builds a fresh slot carrying a directly-encoded value
     *        (Kind::VALUE). T must be trivially-copyable and at most
     *        payload-width. When T is narrower than PayloadT the bytes are
     *        zero-extended into the upper positions; a subsequent
     *        `as_value<T>` round-trip is bit-exact when the same T is used
     *        on both sides.
     */
    template <typename T>
    static constexpr Slot from_value(T x) noexcept
    {
        static_assert(std::is_trivially_copyable_v<T>,
                      "Slot::from_value requires a trivially-copyable type");
        static_assert(sizeof(T) <= sizeof(PayloadT),
                      "Slot::from_value: T must fit within PayloadT width");
        static_assert(detail::has_value_type_v<T>,
                      "Slot::from_value: T must have a ValueType mapping "
                      "(bool / int / float family)");
        return Slot{.flags   = make_flags(Kind::VALUE, detail::value_type_of<T>()),
                    .payload = make_payload(x)};
    }

    /**
     * @brief Builds a fresh slot that indirects to a runtime-supplied
     *        binding value (Kind::BINDING_ID). The id is range-checked
     *        against MAX_BINDINGS_V4 at construction.
     */
    static constexpr Slot from_binding_id(IndexT id) noexcept
    {
        if(id >= MAX_BINDINGS_V4) { bindingIdOutOfRange(); }
        return Slot{.flags   = make_flags(Kind::BINDING_ID, detail::value_type_of<IndexT>()),
                    .payload = make_payload(id)};
    }

    /**
     * @brief Builds a fresh slot that indirects to an edge-length value
     *        (Kind::EDGE_ID). The id is range-checked against
     *        MAX_EDGES_V4 at construction.
     */
    static constexpr Slot from_edge_id(IndexT id) noexcept
    {
        if(id >= MAX_EDGES_V4) { edgeIdOutOfRange(); }
        return Slot{.flags   = make_flags(Kind::EDGE_ID, detail::value_type_of<IndexT>()),
                    .payload = make_payload(id)};
    }

    /**
     * @brief Builds a fresh slot whose length is computed by the owning
     *        Impl at Phase B via deriveInputLength / deriveOutputLength
     *        (Kind::DERIVED). No payload carried.
     */
    static constexpr Slot from_derived() noexcept
    {
        return Slot{.flags = make_flags(Kind::DERIVED, ValueType::NONE)};
    }

    /**
     * @brief Adapts a user-supplied raw value into a slot by dispatching
     *        on the source type. Routes placeholder<Id> to from_binding_id;
     *        routes enums via their underlying integer type; routes
     *        everything else (bool, ints, floats) through from_value.
     */
    template <typename SrcT>
    static constexpr Slot from_raw(SrcT raw) noexcept
    {
        static_assert(std::is_arithmetic_v<SrcT>
                          || std::is_enum_v<SrcT>
                          || detail::is_placeholder_v<SrcT>,
                      "Slot::from_raw: SrcT must be an arithmetic type "
                      "(bool/int/float family), enum, or placeholder<Id>. "
                      "Pointers, references, arrays, function pointers, and "
                      "user-defined types are not supported as Slot payloads.");
        static_assert(sizeof(SrcT) <= sizeof(PayloadT),
                      "Slot::from_raw: argument is wider than PayloadT");
        if constexpr(detail::is_placeholder_v<SrcT>) {
            static_assert(SrcT::ID <= std::numeric_limits<IndexT>::max(),
                          "Slot::from_raw: placeholder Id exceeds IndexT capacity");
            return from_binding_id(static_cast<IndexT>(SrcT::ID));
        } else if constexpr(std::is_enum_v<SrcT>) {
            return from_raw(static_cast<std::underlying_type_t<SrcT>>(raw));
        } else {
            return from_value(raw);
        }
    }

    // ---- In-place mutator ---------------------------------------------------

    /**
     * @brief In-place write: transition this slot to Kind::VALUE with the
     *        payload bytes bit-cast from x. Both kind and vtype are
     *        updated; transitioning kind from UNUSED to VALUE serves as
     *        the "was written" flag.
     */
    template <typename T>
    CK_TILE_HOST_DEVICE constexpr void set_value(T x) noexcept
    {
        static_assert(std::is_trivially_copyable_v<T>,
                      "Slot::set_value requires a trivially-copyable type");
        static_assert(sizeof(T) <= sizeof(PayloadT),
                      "Slot::set_value: T must fit within PayloadT width");
        static_assert(detail::has_value_type_v<T>,
                      "Slot::set_value: T must have a ValueType mapping "
                      "(bool / int / float family)");
        // Bodies match `make_flags(...)` and `make_payload(x)` (see below);
        // inlined here because `set_value` is the only runtime-callable
        // Slot mutator (kernel-side EdgeLengths / RuntimeBindings writes)
        // while both helpers are consteval (graph-construction-only).
        flags   = Flags{.value = {.kind  = Kind::VALUE,
                                  .vtype = detail::value_type_of<T>()}};
        payload = ck_tile::bit_cast<PayloadT>(
                      detail::PaddingHelper<T, sizeof(PayloadT)>{.value = x});
    }

    // ---- Discriminator accessors --------------------------------------------

    constexpr Kind      kind()       const noexcept { return flags.value.kind;  }
    constexpr ValueType value_type() const noexcept { return flags.value.vtype; }

    constexpr bool is_unused()           const noexcept { return kind() == Kind::UNUSED; }
    constexpr bool is_value()            const noexcept { return kind() == Kind::VALUE; }
    constexpr bool is_binding_id()       const noexcept { return kind() == Kind::BINDING_ID; }
    constexpr bool is_edge_id()          const noexcept { return kind() == Kind::EDGE_ID; }
    constexpr bool is_derived()          const noexcept { return kind() == Kind::DERIVED; }

    // ---- Typed payload reads ------------------------------------------------

    /**
     * @brief Read the payload bytes as T. Caller must use the same T (or
     *        any T of equal byte width) that was used by the writer. When
     *        T is narrower than PayloadT, the low sizeof(T) bytes are
     *        returned.
     */
    template <typename T>
    CK_TILE_HOST_DEVICE constexpr T as_value() const noexcept
    {
        static_assert(std::is_trivially_copyable_v<T>,
                      "Slot::as_value<T> requires a trivially-copyable T");
        static_assert(sizeof(T) <= sizeof(PayloadT),
                      "Slot::as_value<T>: T must fit within PayloadT width");
        // Domain guard: reading the integer-family payload as a float (or vice
        // versa) is always a bug. Gated on is_constant_evaluated so it costs
        // zero runtime code -- it fires only at NTTP / constant-evaluated call
        // sites (where the non-constexpr diagnostic stub turns it into a compile
        // error) and is fully elided on the runtime hot path. Applied only when
        // T has a ValueType mapping; for any other trivially-copyable T,
        // as_value remains a pure byte reader.
        if constexpr(detail::has_value_type_v<T>)
        {
            if(std::is_constant_evaluated())
            {
                if(!detail::same_value_domain(value_type(), detail::value_type_of<T>()))
                {
                    slotReadValueDomainMismatch();
                }
            }
        }
        return ck_tile::bit_cast<detail::PaddingHelper<T, sizeof(PayloadT)>>(payload).value;
    }

    /// @brief Recover the binding id. Caller must have checked is_binding_id().
    constexpr IndexT as_binding_id() const noexcept { return as_index(); }

    /// @brief Recover the edge id. Caller must have checked is_edge_id().
    constexpr IndexT as_edge_id() const noexcept { return as_index(); }

    // Note: Kind dispatch lives in `detail::eval<T>(Slot, RBView, ELView)`.
    // Slot deliberately knows nothing about RuntimeBindings / EdgeLengths so
    // it remains a pure NTTP-friendly value container.

private:
    // ---- Private accessors ---------------------------------------------------

    /// @brief Canonical IndexT payload decoder. `_ID` kinds (BINDING_ID,
    ///        EDGE_ID) stash an index in the payload's low bytes; the public
    ///        as_binding_id / as_edge_id aliases route through here, and the
    ///        decode itself goes through as_value<IndexT> so id reads share the
    ///        single typed-read path (and its domain guard).
    constexpr IndexT as_index() const noexcept { return as_value<IndexT>(); }

    // ---- Private factory helpers (used only by public factories above) ------

    static constexpr Flags make_flags(Kind k, ValueType v) noexcept
    {
        return Flags{.value = {.kind = k, .vtype = v}};
    }

    template <typename T>
    static constexpr PayloadT make_payload(T x) noexcept
    {
        return ck_tile::bit_cast<PayloadT>(
                   detail::PaddingHelper<T, sizeof(PayloadT)>{.value = x});
    }
};

// Anchors `Slot::make_payload<IndexT>` for the constexpr evaluator.
//
// Graph-construction factories evaluate at compile time and descend through
// `Slot::from_binding_id(IndexT)` / `Slot::from_edge_id(IndexT)` for every
// placeholder and every per-transform edge anchor. Both reach
// `make_payload<IndexT>`. The evaluator needs that specialization's body to
// already exist when it gets there; this line guarantees it does by ODR-using
// the same specialization at namespace scope.
namespace detail {
inline constexpr Slot::PayloadT
force_make_payload_indexT_instantiation = Slot::from_value(IndexT{0}).payload;

/**
 * @brief Re-encode a slot at Precision's width, preserving the value and its
 *        domain: signed -> Precision::Signed, unsigned -> Precision::Unsigned,
 *        float -> Precision::Float. BOOL and all non-VALUE slots (ids,
 *        derived, unused) pass through unchanged -- ids are width-independent
 *        indices and a bool is a width-1 flag.
 *
 * Decodes the stored value at its source type, then static_casts to the
 * precision-width type (sign/zero/float-correct across widths -- a raw byte
 * copy would corrupt a widened negative).
 *
 * COMPILE-TIME-NEUTRALITY: the 64-bit source arms (I64/U64/F64) are
 * `if constexpr`-pruned out of a 32-bit Precision specialization, so a P32
 * translation unit never instantiates the 64-bit as_value family. A well-formed
 * P<N> graph only stores source values that fit P<N>'s width, so the pruned
 * arms are unreachable for that precision. This is why induction selects the
 * arms per precision rather than running a single switch over every source
 * type (which would name every as_value<U> and drag the 64-bit family into
 * every TU that instantiates it).
 *
 * Free function (not a Slot member) so Slot stays precision-agnostic. Applied
 * at the two write boundaries (insertTransform for pool/anchor slots,
 * make_graph_bindings for runtime bindings) so every stored VALUE matches the
 * graph's precision and every later read is a plain same-width as_value read.
 */
template <typename Precision>
CK_TILE_HOST_DEVICE constexpr Slot adjust_precision(Slot s) noexcept
{
    if(s.kind() != Kind::VALUE) { return s; }
    const ValueType v = s.value_type();
    if(v == ValueType::BOOL) { return s; }

    using Signed   = typename Precision::Signed;
    using Unsigned = typename Precision::Unsigned;
    using Float    = typename Precision::Float;

    // Identity fast-path: a VALUE slot already stored at this precision's own
    // width (signed / unsigned / float) is byte-correct as-is, so the re-encode
    // switch below would be a same-width `from_value(static_cast<T>(as_value<T>
    // (s)))` round-trip that reproduces s exactly. Returning s skips that
    // constexpr evaluation for the dominant case -- every literal authored at
    // the graph's own precision (e.g. an I32 dim/stride at P32) takes this path.
    // Byte-identical to the switch result; the switch is retained for the
    // off-width source arms (a narrower or wider authored literal that genuinely
    // must be re-encoded to the precision width).
    if(v == value_type_of<Signed>() || v == value_type_of<Unsigned>()
       || v == value_type_of<Float>())
    {
        return s;
    }

    // Mutually-exclusive per-precision-width re-encode: exactly ONE switch
    // compiles per Precision, so each precision instantiates only its own
    // as_value arms (a P32 TU never names the 64-bit family) and the compiler
    // folds a single self-contained switch. A 64-bit precision still handles
    // the <=32-bit source arms because a narrow authored literal (e.g. an
    // `int` dim) is stored as an I32 slot that must widen to the 64-bit type.
    if constexpr(sizeof(Signed) == 4) {
        switch(v) {
        case ValueType::I8:  return Slot::from_value(static_cast<Signed>(s.template as_value<int8_t>()));
        case ValueType::I16: return Slot::from_value(static_cast<Signed>(s.template as_value<int16_t>()));
        case ValueType::I32: return Slot::from_value(static_cast<Signed>(s.template as_value<int32_t>()));
        case ValueType::U8:  return Slot::from_value(static_cast<Unsigned>(s.template as_value<uint8_t>()));
        case ValueType::U16: return Slot::from_value(static_cast<Unsigned>(s.template as_value<uint16_t>()));
        case ValueType::U32: return Slot::from_value(static_cast<Unsigned>(s.template as_value<uint32_t>()));
        case ValueType::F32: return Slot::from_value(static_cast<Float>(s.template as_value<float>()));
        // Not representable at 4-byte precision (NONE / 64-bit vtypes) or handled
        // before this switch (BOOL): fall through to the trap below.
        case ValueType::NONE:
        case ValueType::I64:
        case ValueType::U64:
        case ValueType::F64:
        case ValueType::BOOL: break;
        }
    } else if constexpr(sizeof(Signed) == 8) {
        switch(v) {
        case ValueType::I8:  return Slot::from_value(static_cast<Signed>(s.template as_value<int8_t>()));
        case ValueType::I16: return Slot::from_value(static_cast<Signed>(s.template as_value<int16_t>()));
        case ValueType::I32: return Slot::from_value(static_cast<Signed>(s.template as_value<int32_t>()));
        case ValueType::I64: return Slot::from_value(static_cast<Signed>(s.template as_value<int64_t>()));
        case ValueType::U8:  return Slot::from_value(static_cast<Unsigned>(s.template as_value<uint8_t>()));
        case ValueType::U16: return Slot::from_value(static_cast<Unsigned>(s.template as_value<uint16_t>()));
        case ValueType::U32: return Slot::from_value(static_cast<Unsigned>(s.template as_value<uint32_t>()));
        case ValueType::U64: return Slot::from_value(static_cast<Unsigned>(s.template as_value<uint64_t>()));
        case ValueType::F32: return Slot::from_value(static_cast<Float>(s.template as_value<float>()));
        case ValueType::F64: return Slot::from_value(static_cast<Float>(s.template as_value<double>()));
        // NONE has no value; BOOL is handled before this switch: trap below.
        case ValueType::NONE:
        case ValueType::BOOL: break;
        }
    } else {
        static_assert(dependent_false_v<Precision>,
                      "adjust_precision: unsupported Precision width (expected 4- or 8-byte "
                      "Signed/Unsigned/Float).");
    }

    // Unreachable for a well-formed VALUE slot at this precision (BOOL handled
    // above; every in-range source vtype is a case in the active switch).
    valueReadFromUnusedSlot();
    return s;
}

/**
 * @brief Normalize a runtime binding value to the graph precision at the
 *        make_graph_bindings write boundary.
 *
 * Companion to adjust_precision for the binding path. The arg TYPE is known at
 * the call site, so the target precision domain is selected with `if constexpr`
 * and a single static_cast -- there is NO runtime vtype switch (which would add
 * a branch on the placeholder hot path and pull the off-precision as_value
 * family into the TU). float -> Precision::Float, signed int -> Precision::Signed,
 * other integers -> Precision::Unsigned; bool passes through unchanged (BOOL
 * carries no precision). For the default precision this matches a plain
 * set_value(arg): an `int` binding stays an I32 slot with identical bytes.
 */
template <typename Precision, typename T>
CK_TILE_HOST_DEVICE constexpr Slot normalize_binding(T arg) noexcept
{
    static_assert(std::is_arithmetic_v<T>,
                  "normalize_binding: a runtime binding must be an arithmetic value");
    if constexpr(std::is_same_v<T, bool>) {
        return Slot::from_value(arg);
    } else if constexpr(std::is_floating_point_v<T>) {
        return Slot::from_value(static_cast<typename Precision::Float>(arg));
    } else if constexpr(std::is_signed_v<T>) {
        return Slot::from_value(static_cast<typename Precision::Signed>(arg));
    } else {
        return Slot::from_value(static_cast<typename Precision::Unsigned>(arg));
    }
}
} // namespace detail

// NOTE: the compile-time smoke asserts for detail::adjust_precision (signed
// sign-extend, unsigned zero-extend, non-VALUE pass-through) live in the V4
// SubGraph test TU rather than here -- evaluating them at header scope forced
// adjust_precision<Precision32> + its from_value/as_value family into EVERY TU
// that includes this header. They emit no code, so relocating them to one
// compiled test preserves coverage without the per-TU instantiation tax.


// =============================================================================
// 4. Pool -- AoS value storage; thin wrapper over static_array<Slot, MAX>
// =============================================================================

/// @brief Graph-level value storage. Inherits from `static_array<Slot,
///        MAX_POOL_VALUES_V4>` so per-slot access uses the same
///        `pool[i].kind() / pool[i].payload / pool[i] = v` shape as the
///        EL / RB / Args wrappers. The Slot's pad-array layout keeps
///        `has_unique_object_representations_v` true -- AoS-NTTP-safe.
struct Pool : ck_tile::static_array<Slot, MAX_POOL_VALUES_V4>
{
    constexpr bool operator==(const Pool&) const = default;
};

static_assert(std::has_unique_object_representations_v<Pool>,
              "Pool must have unique object representations for NTTP correctness");


// =============================================================================
// 5. RuntimeBindings and EdgeLengths -- runtime-supplied data
// =============================================================================

/// @brief Host-supplied placeholder values, indexed by binding id.
///
/// NumBindings template parameter (default MAX_BINDINGS_V4) lets per-graph
/// instantiations size the storage to G.num_bindings instead of the worst-
/// case constant. P0 of the V4 scratch-spill fix: shrinks rb storage from
/// 1020 B to ~4N B for a typical graph (e.g. simple/placeholder uses 6
/// bindings -> 24 B). User-friendly per-graph alias `v4::RB<G>` is the
/// recommended way to instantiate without typo-prone count specification.
template <ck_tile::index_t NumBindings = MAX_BINDINGS_V4>
struct RuntimeBindings : ck_tile::static_array<Slot, NumBindings>
{
    constexpr bool operator==(const RuntimeBindings&) const = default;
};

/// @brief Per-graph alias: instantiates RuntimeBindings with the graph's
///        num_bindings. Avoids the footgun of the user typing the count.
///
/// Usage in workloads (make_graph_bindings is the single entry; it sizes and
/// fills the RuntimeBindings internally, so workloads never touch RB directly):
/// @code
/// constexpr auto graph = make_simple_v4_runtime();
/// const auto gb = v4::make_graph_bindings<graph>(42);  // binds placeholder<0>
/// @endcode
template <auto G>
using RB = RuntimeBindings<G.num_bindings>;

/// @brief Non-templated view of RuntimeBindings. Carries Slot-typed
///        indexing so callers use `bindings[id].template as_value<T>()`
///        uniformly with or without a view in hand.
struct RuntimeBindingsView
{
    Slot const* values = nullptr;

    constexpr const Slot& operator[](ck_tile::index_t i) const noexcept { return values[i]; }
};

template <ck_tile::index_t NumBindings>
CK_TILE_HOST_DEVICE constexpr RuntimeBindingsView
make_runtime_bindings_view(const RuntimeBindings<NumBindings>& b) noexcept
{
    // A literal graph has NumBindings == 0; forming &b[0] on an empty array is
    // out of bounds (and non-constant in a constexpr context). Such a view is
    // never indexed (no BINDING_ID slots exist), so a null view is correct.
    if constexpr(NumBindings == 0) {
        return RuntimeBindingsView{.values = nullptr};
    } else {
        return RuntimeBindingsView{.values = &b[0]};
    }
}

/// @brief Resolved edge lengths, indexed by edge id.
///
/// Phase B output: computed once at make_graph_bindings() from the graph's
/// input_edge_anchors / output_edge_anchors[] + each Impl's deriveOutputLengths(). Read by all
/// Phase B resolve() calls. NOT read at Phase C (mapCoord) -- those use
/// the per-transform input/output coord arrays.
///
/// (Renamed from V3's SlotLens: this stores LENGTHS indexed by edge id,
/// not the slots themselves.)
///
/// Two-tier design (P0 of the V4 scratch-spill fix sprint):
///
///   * `EdgeLengths<NumEdges>` -- inline STORAGE. Templated on
///     NumEdges so per-graph instances size the buffers to G.num_edges
///     instead of MAX_EDGES_V4. A function-local in make_graph_bindings
///     (Phase B intermediate), discarded after the state tuple is built --
///     NOT cached in the bindings object.
///
///   * `EdgeLengthsView` -- pointer-and-count VIEW. Non-templated;
///     passed by value to all consumers (as<>, make_schema,
///     resolveOneTransformEdges, ...). Avoids propagating the NumEdges
///     template parameter through the entire framework. The pointer is
///     only resolved during make_graph_bindings (kernel-launch-once); mapCoord
///     never dereferences gb.edges, so the per-coord hot path pays no
///     indirection cost.
template <ck_tile::index_t NumEdges, typename LengthT>
struct EdgeLengths : ck_tile::static_array<LengthT, NumEdges>
{
    constexpr bool operator==(const EdgeLengths&) const = default;
};

/// @brief Non-templated view: pointer + active count. Constructed via
///        `make_edge_lengths_view(EdgeLengths<>)`. Passed by value (small
///        struct). Carries the same Slot-typed indexing as the inline
///        EdgeLengths storage so callers can use
///        `view[i].set_value<T>(v)` / `view[i].as_value<T>()` uniformly
///        with or without a view in hand.
template <typename LengthT>
struct EdgeLengthsView
{
    LengthT* values = nullptr;

    constexpr LengthT&       operator[](ck_tile::index_t i)       noexcept { return values[i]; }
    constexpr const LengthT& operator[](ck_tile::index_t i) const noexcept { return values[i]; }
};

// Factories on EdgeLengths<>. Defined as free templates so they can deduce
// NumEdges and produce a non-templated view.
template <ck_tile::index_t NumEdges, typename LengthT>
CK_TILE_HOST_DEVICE constexpr EdgeLengthsView<LengthT>
make_edge_lengths_view(EdgeLengths<NumEdges, LengthT>& e) noexcept
{
    return EdgeLengthsView<LengthT>{.values = &e[0]};
}

template <ck_tile::index_t NumEdges, typename LengthT>
CK_TILE_HOST_DEVICE constexpr EdgeLengthsView<LengthT>
make_edge_lengths_view(const EdgeLengths<NumEdges, LengthT>& e) noexcept
{
    auto& m = const_cast<EdgeLengths<NumEdges, LengthT>&>(e);
    return EdgeLengthsView<LengthT>{.values = &m[0]};
}


// =============================================================================
// 6. Typed accessors -- removed
// =============================================================================
//
// The free function `as<T>(pool, i, bindings, edges)` was removed. Its body
// was a one-line delegate to `Slot::eval<T>(b, e)`; the single remaining
// consumer (`resolve_one_schema_member`) now calls
// `pool[i].template eval<T>(bindings, edges)` directly. Single source
// of truth for Kind dispatch lives on `Slot::eval<T>` (in section 3).
//
// (P3.D earlier deletion note retained for history: as_i32/as_u32/as_f32/
// as_bool/as_i64/as_f64 convenience aliases were deleted when SchemaView
// was retired; their consumers were the typed accessors, also deleted.)


// (P3.D) SchemaView class + i32_at/u32_at/f32_at/bool_at/i64_at/f64_at
// bounds-checked accessors deleted: all per-Impl Schemas migrated to
// `uniform_schema_t<SlotInt>`, leaving zero call sites.


// =============================================================================
// 7. DimIds, CoordinateTransform
// =============================================================================

/// @brief Per-transform dim-id routing (input + output edge ids). Inherits
///        `static_array<uint8_t, MAX_TENSOR_DIMS_V4>` so callers index the ids
///        directly via `dim_ids[i]` rather than `dim_ids.ids[i]`. Matches the
///        Pool / RuntimeBindings / EdgeLengths / TransformArgs wrapper-collapse
///        pattern (memory: project_v4_wrapper_collapse_2026_05_22).
struct DimIds : ck_tile::static_array<IndexT, MAX_TENSOR_DIMS_V4>
{
    using Base = ck_tile::static_array<IndexT, MAX_TENSOR_DIMS_V4>;

    constexpr DimIds() noexcept
    {
        for(auto& x : Base::elems) { x = INVALID_EDGE_ID; }
    }

    constexpr bool operator==(const DimIds&) const = default;

    /// @brief Number of valid (non-sentinel) ids. consteval so the result is a
    ///        compile-time constant when called on a constexpr DimIds, which
    ///        lets callers parameterize a make_index_sequence on it without
    ///        spinning a 64-iter early-break loop in IR.
    constexpr IndexT count() const noexcept
    {
        IndexT c = 0;
        for(auto x : Base::elems) {
            if(x == INVALID_EDGE_ID) { break; }
            ++c;
        }
        return c;
    }
};

/// @brief Type tag for each Impl. Same set of 9 transforms as V3.
enum struct TransformType : uint8_t
{
    EMBED, MERGE, UNMERGE, XOR, OFFSET, FREEZE, SLICE, PAD, BROADCAST,
};

/// @brief Per-transform record. Holds a (base, count) view into the pool
///        plus arity metadata. NO inline data buffer.
struct CoordinateTransform
{
    TransformType type        = TransformType::EMBED;
    uint8_t       base_offset = 0;
    uint8_t       value_count = 0;
    uint8_t       ndim_input  = 0;
    uint8_t       ndim_output = 0;

    constexpr bool operator==(const CoordinateTransform&) const = default;
};

// These three types are the leaf building blocks of TransformGraph, which is
// used as a template<auto G> NTTP and must have no implicit padding. A hole in
// any leaf propagates up into TransformGraph; asserting each leaf here pins the
// fault to the offending type instead of surfacing as a confusing aggregate
// failure on TransformGraph. Re-pack the named type by descending alignment if
// one of these fires.
static_assert(std::has_unique_object_representations_v<Slot>,
              "Slot must have no implicit padding (TransformGraph NTTP leaf).");
static_assert(std::has_unique_object_representations_v<DimIds>,
              "DimIds must have no implicit padding (TransformGraph NTTP leaf).");
static_assert(std::has_unique_object_representations_v<CoordinateTransform>,
              "CoordinateTransform must have no implicit padding (TransformGraph NTTP leaf).");


// History: the per-Impl Schema went through three iterations:
//   - SchemaFor<Fields, PoolT, SlotT> (P3.D removed)
//   - synthesized_schema<Fields>      (UVS cleanup removed)
//   - uniform_schema_t<SlotInt>       (removed during Schema refactor)
// All nine Impls now declare a per-Impl POD `struct Schema` with a
// `using members = detail::MemberPtrList<...>` reflection descriptor.


// =============================================================================
// 7b.6 Schema Pool walker -- bridge from Pool slots to per-Impl Schema
// =============================================================================
//
// Walks the per-Impl Schema's `MemberPtrList` in declaration order,
// consuming Pool slots and writing each native data member directly via
// `(out.*MemPtr)` / `(out.*MemPtr)[i]`. Used by `make_schema` for every
// per-Impl Schema.
//
// Sizing convention: scalar member sets `current_size`; the next array
// member reads exactly `current_size` slots. Matches the legacy
// `sized_by<prev-scalar>` invariant.

namespace detail {

// Forward-declaration: the single Kind-dispatch is defined later in the file
// (after RuntimeBindingsView / EdgeLengthsView). resolve_one_schema_member
// below calls it; the actual body is in the same namespace and will be found
// via ordinary lookup at template-instantiation time.
template <typename T, typename LengthT>
CK_TILE_HOST_DEVICE constexpr T
eval(Slot s, RuntimeBindingsView bindings, EdgeLengthsView<LengthT> edges) noexcept;

// Resolve one member of a Schema from the Pool.
//   - Scalar member: read one slot, store as the member's value type, and
//                    update `current_size` so a following array member can
//                    use this scalar as its size source (matches the legacy
//                    `sized_by<prev-scalar>` invariant).
//   - Array  member: read `current_size` slots, store each into the
//                    corresponding array slot. If `current_size` exceeds
//                    the array's compile-time extent, only the first
//                    `count` entries are written.
template <auto MemPtr, typename Schema, typename LengthT>
constexpr void
resolve_one_schema_member(Schema&             out,
                          const Pool&       pool,
                          RuntimeBindingsView bindings,
                          EdgeLengthsView<LengthT>     edges,
                          ck_tile::index_t    base,
                          ck_tile::index_t&   slot_offset,
                          ck_tile::index_t&   current_size) noexcept
{
    using slot_payload_t = Slot::PayloadT;
    if constexpr(is_array_member_v<MemPtr>) {
        using elem_t = typename member_array_traits<MemPtr>::element_type;
        // Width contract: member element type must FIT within the Pool slot
        // payload width. Narrower types are handled by Slot::eval's
        // pad/unpad dispatch -- bit-pad on write, low-byte unpad on read.
        // Wider-than-payload elements are still rejected at compile time.
        static_assert(sizeof(elem_t) <= sizeof(slot_payload_t) &&
                          std::is_trivially_copyable_v<elem_t>,
                      "resolve_one_schema_member: array element type must "
                      "fit within Pool slot payload width and be trivially-"
                      "copyable. Equal-width and narrower-than-payload "
                      "elements both supported via Slot's pad/unpad path.");
        constexpr ck_tile::index_t max_n   = member_array_traits<MemPtr>::count;
        const ck_tile::index_t     n       = current_size;
        const ck_tile::index_t     n_bound = (n < max_n) ? n : max_n;
        // slot_offset advances by `n` (caller-reserved Pool footprint),
        // not `n_bound` (member storage capacity). Truncation when n > max_n
        // is silent-by-design; the placement contract guarantees n <= max_n
        // in practice. A future maintainer must not "fix" this to n_bound.
        for(ck_tile::index_t i = 0; i < n_bound; ++i) {
            (out.*MemPtr)[i] = eval<elem_t>(pool[base + slot_offset + i],
                                            bindings, edges);
        }
        slot_offset += n;
    } else {
        using elem_t = typename member_scalar_traits<MemPtr>::type;
        static_assert(sizeof(elem_t) <= sizeof(slot_payload_t) &&
                          std::is_trivially_copyable_v<elem_t>,
                      "resolve_one_schema_member: scalar member type must "
                      "fit within Pool slot payload width and be trivially-"
                      "copyable. Narrower-than-payload types are handled "
                      "by detail::eval's pad/unpad dispatch.");
        const elem_t v = eval<elem_t>(pool[base + slot_offset],
                                      bindings, edges);
        (out.*MemPtr) = v;
        current_size  = static_cast<ck_tile::index_t>(v);
        slot_offset += 1;
    }
}

// Walk a Schema's MemberPtrList, resolving each member from the Pool in
// declaration order.
//
// Pool-layout contract: the first member must be a scalar (it acts as the
// size source for any following array). This mirrors the legacy
// `sized_by<prev-field>` invariant. Without a leading scalar, an initial
// array would silently read 1 element from the `current_size = 1` initial
// value -- almost certainly a bug.
template <auto... MemPtrs, typename Schema, typename LengthT>
constexpr void
resolve_schema_from_pool(Schema&                   out,
                         const Pool&             pool,
                         RuntimeBindingsView       bindings,
                         EdgeLengthsView<LengthT>           edges,
                         ck_tile::index_t          base,
                         MemberPtrList<MemPtrs...> /*members_tag*/) noexcept
{
    if constexpr(sizeof...(MemPtrs) > 0) {
        static_assert(
            !is_array_member_v<nth_member_v<0, MemberPtrList<MemPtrs...>>>,
            "Schema: the first member must be a scalar -- it sizes the "
            "following array(s), mirroring the legacy "
            "`sized_by<prev-scalar>` invariant. An empty Schema (no "
            "members) is also valid.");
        ck_tile::index_t slot_offset  = 0;
        ck_tile::index_t current_size = 1;   // scalars consume exactly one slot
        (resolve_one_schema_member<MemPtrs>(out, pool, bindings, edges, base,
                                            slot_offset, current_size),
         ...);
    } else {
        // Empty Schema (XOR, BROADCAST): no Pool slots consumed.
        (void)pool;
        (void)bindings;
        (void)edges;
        (void)base;
    }
}

// Construct a default-initialized `Impl::Schema` and fill it from the Pool.
template <typename Impl, typename LengthT>
constexpr auto
make_schema(const Pool&       pool,
            RuntimeBindingsView bindings,
            EdgeLengthsView<LengthT>     edges,
            ck_tile::index_t    base) noexcept
{
    using Schema = typename Impl::Schema;
    Schema out{};
    resolve_schema_from_pool(
        out, pool, bindings, edges, base, typename Schema::members{});
    return out;
}

} // namespace detail


// =============================================================================
// 8. TransformImpl<X> -- primary template (specialize per Impl)
// =============================================================================
//
// PER-IMPL DATA MODEL: TWO VIEWS, TWO PHASES
// ------------------------------------------
//
// Each TransformImpl carries a Schema/State pair. They look similar but
// serve fundamentally different roles, with different cost budgets:
//
//   Schema  = CONSTRUCTION VIEW.
//             A read-through projection of pool data. Each named accessor
//             (e.g. s.shift(), s.length(i), s.stride(i)) dispatches on the
//             pool entry's Kind tag at the moment of read:
//                 Kind::VALUE     -> bit_cast<T>(payload)
//                 Kind::BINDING_ID     -> bindings.values[id]   (runtime placeholder)
//                 Kind::EDGE_ID -> edge_lengths[id] (resolved length slot)
//             Cheap to construct (4 references + 1 count + bounds check);
//             each read costs a Kind dispatch + an indirect lookup. Lives
//             ACROSS PHASE B ONLY -- consumed by resolveState(), then discarded.
//             Backed by `uniform_schema_t<SlotInt>::slots[]`; read via
//             `field_of<"Name", Fields>(s [, i])`.
//
//   State   = FINALIZED VIEW.
//             A snapshot of EVALUATED values, plain typed fields, no Kind
//             dispatch, no pool access, no indirection. Holds ONLY the
//             fields that mapCoord reads at runtime (e.g. for MERGE:
//             {strides[N], magic_divs[N]}; for OFFSET: {shift}; for XOR:
//             {length_1}). Built once per kernel inside resolveState(); READ
//             MANY TIMES inside Phase C's per-coord hot loop. Sized to the
//             actual per-instance arity (not the worst-case maximum), so
//             SGPR allocation tracks real arity.
//
// WHY BOTH EXIST
// --------------
// The pipeline is:  State = resolveState(Schema, in_lens, out_lens)
//                   per_coord_output = mapCoord(State, ..., per_coord_input)
//
//   Phase B does the Kind-dispatch work ONCE at kernel-launch time.
//   Phase C consumes the State snapshot N TIMES per work-item.
//
// Schema cannot be the runtime form: the 4-reference + Kind-dispatch +
// bounds-check overhead per read would dominate a hot loop that runs once
// per (work-item, transform).
//
// State cannot be the construction form: it has no knowledge of Kind,
// no pool reference, no way to look up a binding or edge length. It just
// holds finalized typed values that the kernel can read with a single
// load.
//
// EDGE LENGTH AUTHORITY (read in resolveState)
// --------------------------------------------
// Edge lengths come from `in_lens` / `out_lens` parameters (authoritative
// resolved values from EdgeLengths array). Schema holds per-Impl math
// parameters (strides, shifts, padding amounts, frozen idx). resolveState()
// reads `in_lens` / `out_lens` for any lengths it needs.
//
// PER-IMPL CONTRACT MEMBERS
// -------------------------
// Each specialization provides:
//
//   struct Schema {
//       int32_t a;
//       int32_t arr[MAX_TENSOR_DIMS_V4];
//       using members = detail::MemberPtrList<&Schema::a, &Schema::arr>;
//   };
//
//   template <uint8_t NIn|NOut>
//   struct State { /* plain typed fields read by mapCoord */ };
//
//   template <uint8_t NIn, uint8_t NOut, typename SchemaT>
//   static State<...> resolveState(const SchemaT&                       schema,
//                                  const index_t* __restrict__         in_lens,
//                                  const index_t* __restrict__         out_lens);
//
//   template <uint8_t NIn, uint8_t NOut, typename CoordT>
//   static void mapCoord(const State<...>& state,
//                        CoordT* output_coords, const CoordT* input_coords);
//
//   // Optional: required when the factory declares a Slot::from_derived()
//   // anchor on the corresponding side. The framework calls this once per
//   // DERIVED slot; `derived_output_idx` / `derived_input_idx` is the
//   // absolute index in the Impl's output / input slot array of the slot
//   // currently being derived. Impls with a single DERIVED slot may ignore
//   // it; Impls with two or more dispatch on it.
//   template <typename SchemaT>
//   static index_t deriveOutputLength(const SchemaT&,
//                                     EdgeLengthsView edges,
//                                     DimIds          input_edge_ids,
//                                     uint8_t         derived_output_idx);
//
//   template <typename SchemaT>
//   static index_t deriveInputLength(const SchemaT&,
//                                    EdgeLengthsView edges,
//                                    DimIds          output_edge_ids,
//                                    uint8_t         derived_input_idx);

namespace detail {

/**
 * @brief Product of the first `ndim` component lengths, with an overflow trap.
 *
 * The MERGE input length and the UNMERGE output length are both the product of
 * a Schema's `component_lengths`. Hoisted to a free function keyed on the
 * precision policy (not the Impl) so it instantiates ONCE per precision and is
 * shared by both Impls, rather than once per (Impl, Precision). Accumulates in
 * `Precision::IntAccum` (double the length width) and checks `Precision::max_signed_value`
 * after EACH multiply: a per-step check bounds every intermediate product below
 * the accumulator range, so detection is airtight at both precisions -- a
 * check-once-at-end could wrap the accumulator before the comparison at P64.
 *
 * @tparam Precision  Precision policy -- supplies IntAccum, max_signed_value, LengthT, Signed.
 */
template <PrecisionPolicy Precision>
CK_TILE_HOST_DEVICE constexpr typename Precision::LengthT
componentLengthProduct(typename Precision::Signed         ndim,
                       const typename Precision::LengthT* component_lengths) noexcept
{
    using IntAccum = typename Precision::IntAccum;
    IntAccum product = 1;
    for(ck_tile::index_t i = 0; i < ndim; ++i) {
        product *= static_cast<IntAccum>(component_lengths[i]);
        if(product > Precision::max_signed_value) { mergeOverflow(); }
    }
    return static_cast<typename Precision::LengthT>(product);
}

/**
 * @brief Right-to-left running-product strides over `N` component lengths.
 *
 * `strides[N-1] = 1`; `strides[i] = lens[i+1] * lens[i+2] * ... * lens[N-1]`.
 * MERGE (over NOut) and UNMERGE (over NIn) compute identical strides this way;
 * hoisted to a free function keyed on the precision policy so it instantiates
 * once per (Precision, N) instead of once per (Impl, Precision, arity). The
 * per-step `Precision::max_signed_value` check (same as componentLengthProduct) keeps
 * every intermediate within `Precision::IntAccum`; the final narrowing cast to
 * `Precision::Signed` matches the inline loops it replaces.
 *
 * @tparam Precision  Precision policy -- supplies IntAccum, max_signed_value, Signed, LengthT.
 * @tparam N          Number of components.
 */
template <PrecisionPolicy Precision, uint8_t N>
CK_TILE_HOST_DEVICE constexpr ck_tile::static_array<typename Precision::Signed, N>
rightToLeftStrides(const ck_tile::static_array<typename Precision::LengthT, N>& lens) noexcept
{
    using Signed   = typename Precision::Signed;
    using IntAccum = typename Precision::IntAccum;
    ck_tile::static_array<Signed, N> strides{};
    strides[N - 1] = 1;
    IntAccum running = 1;
    for(int i = N - 2; i >= 0; --i) {
        running *= static_cast<IntAccum>(lens[i + 1]);
        if(running > Precision::max_signed_value) { mergeOverflow(); }
        strides[i] = static_cast<Signed>(running);
    }
    return strides;
}

} // namespace detail

template <TransformType T, PrecisionPolicy Precision = Precision32>
struct TransformImpl;


// -- OFFSET ---------------------------------------------------------------
//
// Shift a single dimension by a constant or runtime-bound value.
//
//   memory dim:  +-----------------+
//                | ... A B C D ... |     length M (memory-side)
//                +-----------------+
//                      ^shift
//
//   user dim:    +-----------------+
//                | A B C D ...     |     length M (user-side, same)
//                +-----------------+
//
//   mapping:  user u  -->  memory (u + shift)
//
// Pool layout (1 value):
//   [base + 0]: shift   (int32 literal | binding | edge_length)

template <typename Precision>
struct TransformImpl<TransformType::OFFSET, Precision>
{
    using LengthT = typename Precision::LengthT;
    /// Pool layout (1 value): Schema, member-pointer NTTP path.
    struct Schema
    {
        typename Precision::Signed shift;

        using members = detail::MemberPtrList<&Schema::shift>;
    };

    /// Phase-C state: just the shift value.
    struct State
    {
        typename Precision::Signed shift = 0;
    };

    template <uint8_t NIn, uint8_t NOut>
    CK_TILE_HOST_DEVICE static constexpr State
    resolveState(const Schema&                                                  s,
                 const ck_tile::static_array<typename Precision::LengthT, NIn>&  /*in_lens*/,
                 const ck_tile::static_array<typename Precision::LengthT, NOut>& /*out_lens*/) noexcept
    {
        return State{s.shift};
    }

    template <uint8_t /*NIn*/, uint8_t /*NOut*/, typename CoordT>
    CK_TILE_HOST_DEVICE static constexpr void
    mapCoord(const State&  state,
             CoordT*       output_coords,
             const CoordT* input_coords) noexcept
    {
        output_coords[0] = input_coords[0] + static_cast<CoordT>(state.shift);
    }
};


// -- EMBED ----------------------------------------------------------------
//
// Linear projection of N input coordinates into 1 output coordinate:
//
//   output[0] = sum_{i in [0, N)} input[i] * stride[i]
//
// Pool layout (declared by the Schema POD below):
//   ndim             : int32 literal (per-instance arity)
//   lengths[ndim]    : int32 literals/bindings (input dim lengths)
//   strides[ndim]    : int32 literals/bindings/edge_lengths (per-dim strides)

template <typename Precision>
struct TransformImpl<TransformType::EMBED, Precision>
{
    using LengthT = typename Precision::LengthT;
    using Signed  = typename Precision::Signed;
    /// Pool layout (1 + 2*N values): Schema, member-pointer NTTP path.
    ///   ndim          : scalar size source for both arrays
    ///   lengths[N]    : input dim lengths (sized by ndim)
    ///   strides[N]    : per-dim strides (sized by ndim)
    struct Schema
    {
        Signed   ndim;
        LengthT  lengths[MAX_TENSOR_DIMS_V4];
        Signed   strides[MAX_TENSOR_DIMS_V4];

        using members = detail::MemberPtrList<&Schema::ndim,
                                              &Schema::lengths,
                                              &Schema::strides>;
    };

    /// Phase A: compute output edge length (element-space span).
    ///   span = 1 + sum_{i in [0, ndim)} (in_lens[i] - 1) * strides[i]
    ///
    /// Lengths invariant: a length is strictly non-negative. The returned
    /// span is itself a length and is therefore SlotT::UnsignedT.
    CK_TILE_HOST_DEVICE static constexpr LengthT
    deriveOutputLength(const Schema&   s,
                       EdgeLengthsView<LengthT> edges,
                       DimIds          input_edge_ids,
                       uint8_t         /*derived_output_idx*/) noexcept
    {
        LengthT    span = 1;
        const auto ndim = s.ndim;
        for(ck_tile::index_t i = 0; i < ndim; ++i) {
            const LengthT in_len = edges[input_edge_ids[i]];
            span += (in_len - 1) * static_cast<LengthT>(s.strides[i]);
        }
        return span;
    }

    /// Phase-C state: one stride per input dim. NIn known by framework
    /// at resolveState() time (from dim-id binding count).
    template <uint8_t NIn>
    struct State
    {
        ck_tile::static_array<Signed, NIn> strides{};
    };

    /// Phase B: snapshot strides from the Schema. Edge lengths (via in_lens)
    /// are not needed here -- EMBED's resolveState reads only its own
    /// per-Impl math parameters (strides). in_lens is consumed by
    /// deriveOutputLength above (Phase A).
    template <uint8_t NIn, uint8_t NOut>
    CK_TILE_HOST_DEVICE static constexpr State<NIn>
    resolveState(const Schema&                                                  s,
                 const ck_tile::static_array<LengthT, NIn>&  /*in_lens*/,
                 const ck_tile::static_array<LengthT, NOut>& /*out_lens*/) noexcept
    {
        static_assert(NIn >= 1, "EMBED requires at least one input dim");
        State<NIn> state{};
        for(uint8_t i = 0; i < NIn; ++i) {
            state.strides[i] = s.strides[i];
        }
        return state;
    }

    /// Phase C: linear projection. Hot path. Reads only State; no Kind
    /// dispatch, no pool access. Compiler unrolls the loop at -O3 since
    /// NIn is a template parameter.
    template <uint8_t NIn, uint8_t NOut, typename CoordT>
    CK_TILE_HOST_DEVICE static constexpr void
    mapCoord(const State<NIn>& state,
             CoordT*           output_coords,
             const CoordT*     input_coords) noexcept
    {
        static_assert(NIn >= 1, "EMBED requires at least one input dim");
        // Index-pack fold (mirrors V3's mapIndices pattern). Each Is is a
        // literal index into state.strides[], so the multiplications fold
        // to scalar register ops and the optimizer schedules the chain
        // densely. A runtime for-loop with constexpr NIn does not always
        // produce the same lowering -- the V4 hot-path ASM ballooned vs V3
        // for the complex workload until this was rewritten as a fold.
        output_coords[0] = embedFold(state, input_coords,
                                     ck_tile::make_index_sequence<NIn>{});
    }

private:
    template <uint8_t NIn, ck_tile::index_t... Is, typename CoordT>
    CK_TILE_HOST_DEVICE static constexpr CoordT
    embedFold(const State<NIn>& state, const CoordT* input_coords,
              ck_tile::sequence<Is...>) noexcept
    {
        // No `static_cast<CoordT>` on state.strides[Is]: index_t is the
        // same width as CoordT here, and the explicit cast acts as a range
        // barrier in LLVM that prevents `nuw` propagation through the
        // accumulator. Without nuw, downstream `srem N, 2^k` cannot be
        // folded to `and N, 2^k-1`, costing ~26 instructions in the
        // complex/literal hot loop. (V3 has no cast; V3 generates `and`.)
        return ((input_coords[Is] * state.strides[Is]) + ... + CoordT{0});
    }
};


// -- MERGE ----------------------------------------------------------------
//
// Decompose 1 input dim into N output components. Inverse uses magic
// division so the per-coord path needs no runtime divide. Component
// lengths are authored on the OUTPUT side; the input length is derived as
// product(component_lengths).
//
//   user u (1 dim, length = prod(L_0..L_{N-1}))
//     -->  components (c_0, c_1, ..., c_{N-1}) with c_i in [0, L_i)
//
// Pool layout (1 + N values, where N = ndim_output):
//   [base + 0]:           ndim_components             (int32 literal)
//   [base + 1 .. 1+N):    component_lengths[i]        (int32 literal | binding)

template <typename Precision>
struct TransformImpl<TransformType::MERGE, Precision>
{
    using LengthT = typename Precision::LengthT;
    /// Class-scope alias: magic-division constants type for the graph precision
    /// (32-bit -> MagicDivConstants, 64-bit -> MagicDivConstants64), taken from
    /// the policy. Stored in State per decomposition step.
    using MagicDiv = typename Precision::IntMagicDiv;

    /// Pool layout (1 + N values): Schema, member-pointer NTTP path.
    struct Schema
    {
        typename Precision::Signed  ndim_components;
        typename Precision::LengthT component_lengths[MAX_TENSOR_DIMS_V4];

        using members = detail::MemberPtrList<&Schema::ndim_components,
                                              &Schema::component_lengths>;
    };

    /// Phase A: input edge length = product of component lengths (overflow-
    /// checked in the precision's IntAccum; shared with UNMERGE's output length).
    ///
    /// Reads via `edges` + `output_edge_ids` (P3 of V4 scratch fix). MERGE
    /// computes its input length from the Schema's component_lengths
    /// field, so it does NOT actually consult `edges` -- the params are
    /// kept for signature uniformity across all derive functions.
    CK_TILE_HOST_DEVICE static constexpr typename Precision::LengthT
    deriveInputLength(const Schema&   s,
                      EdgeLengthsView<typename Precision::LengthT> /*edges*/,
                      DimIds          /*output_edge_ids*/,
                      uint8_t         /*derived_input_idx*/) noexcept
    {
        return detail::componentLengthProduct<Precision>(s.ndim_components,
                                                         s.component_lengths);
    }

    /// Phase-C state: derived strides + magic divisors. Sized by NOut.
    template <uint8_t NOut>
    struct State
    {
        ck_tile::static_array<typename Precision::Signed, NOut> derived_strides{};
        ck_tile::static_array<MagicDiv, NOut>                   magic_divs{};
    };

    /// Phase B: precompute strides (right-to-left running product) and
    /// magic divisors for the per-coord decomposition steps.
    template <uint8_t NIn, uint8_t NOut>
    CK_TILE_HOST_DEVICE static constexpr State<NOut>
    resolveState(const Schema&                                                  s,
                 const ck_tile::static_array<typename Precision::LengthT, NIn>&  /*in_lens*/,
                 const ck_tile::static_array<typename Precision::LengthT, NOut>& /*out_lens*/) noexcept
    {
        static_assert(NOut >= 2, "MERGE requires ndim_output >= 2");
        State<NOut> state{};

        // Snapshot component lengths, then compute derived strides via the
        // shared right-to-left running product (overflow-checked in IntAccum).
        ck_tile::static_array<typename Precision::LengthT, NOut> lens{};
        for(uint8_t i = 0; i < NOut; ++i) {
            lens[i] = s.component_lengths[i];
        }
        state.derived_strides = detail::rightToLeftStrides<Precision, NOut>(lens);

        // Precompute magic divisors for the (NOut - 1) decomposition steps. The
        // width is selected by mdv::computeMagicDivFor on the unsigned index type
        // (32- vs 64-bit), so no per-call if constexpr is needed here.
        for(uint8_t i = 0; i < NOut - 1; ++i) {
            if(state.derived_strides[i] > 0) {
                state.magic_divs[i] = mdv::computeMagicDivFor(
                    static_cast<typename Precision::Unsigned>(state.derived_strides[i]));
            }
        }
        return state;
    }

    /// Phase C: magic-div decomposition. Hot path. Uses an index-pack fold
    /// (mirrors V3's mergeUnroll) so each magic_divs[Is] / derived_strides[Is]
    /// access is a literal index that folds to scalar register ops.
    template <uint8_t NIn, uint8_t NOut, typename CoordT>
    CK_TILE_HOST_DEVICE static constexpr void
    mapCoord(const State<NOut>& state,
             CoordT*            output_coords,
             const CoordT*      input_coords) noexcept
    {
        static_assert(NOut >= 2, "MERGE requires ndim_output >= 2");
        mergeFold(state, output_coords, input_coords[0],
                  ck_tile::make_index_sequence<NOut - 1>{});
    }

private:
    template <uint8_t NOut, ck_tile::index_t... Is, typename CoordT>
    CK_TILE_HOST_DEVICE static constexpr void
    mergeFold(const State<NOut>& state,
              CoordT*            output_coords,
              CoordT             remaining,
              ck_tile::sequence<Is...>) noexcept
    {
        // No cast on state.derived_strides[Is] in the multiply: see EMBED for
        // the nuw-propagation reason. mdv::doMagicDivFor selects the 32- vs
        // 64-bit magic division by the unsigned index width (no if constexpr).
        ((output_coords[Is] = static_cast<CoordT>(mdv::doMagicDivFor(
              static_cast<typename Precision::Unsigned>(remaining), state.magic_divs[Is])),
          remaining -= output_coords[Is] * state.derived_strides[Is]), ...);
        output_coords[NOut - 1] = remaining;
    }
};


// -- XOR ------------------------------------------------------------------
//
// Bitwise XOR pattern between two coordinate components.
//
// Pool layout (0 values): XOR carries no per-instance Schema state.
// length_1 (the length of input edge 1) is supplied by the framework via
// my_input_edge_lengths[1] -- the open/closed-clean replacement for V2's
// per-Impl framework hook.

template <typename Precision>
struct TransformImpl<TransformType::XOR, Precision>
{
    using LengthT = typename Precision::LengthT;
    /// Pool layout: empty. XOR carries no per-instance Schema state;
    /// length_1 comes from in_lens[1] (authoritative edge length).
    ///
    /// Empty Schema: no HUOR static_assert because empty classes have
    /// implementation-defined object representation (sizeof = 1, 1 padding
    /// byte). Bit-equality of two empty Schemas is trivially satisfied for
    /// NTTP comparison purposes (the framework never compares them).
    struct Schema
    {
        using members = detail::MemberPtrList<>;
    };

    /// Phase-C state: just the length of input edge 1 (used for the modulo).
    struct State
    {
        typename Precision::Signed length_1 = 0;
    };

    /// Phase B: snapshot length_1 from in_lens[1]. in_lens carries the
    /// precision LengthT (lengths invariant); narrow to the signed
    /// coordinate type for the modulo here.
    template <uint8_t NIn, uint8_t NOut>
    CK_TILE_HOST_DEVICE static constexpr State
    resolveState(const Schema&                                                  /*s*/,
                 const ck_tile::static_array<typename Precision::LengthT, NIn>&  in_lens,
                 const ck_tile::static_array<typename Precision::LengthT, NOut>& /*out_lens*/) noexcept
    {
        return State{static_cast<typename Precision::Signed>(in_lens[1])};
    }

    /// Phase C: bitwise scrambling of input[1] using input[0] mod length_1.
    template <uint8_t /*NIn*/, uint8_t /*NOut*/, typename CoordT>
    CK_TILE_HOST_DEVICE static constexpr void
    mapCoord(const State&  state,
             CoordT*       output_coords,
             const CoordT* input_coords) noexcept
    {
        output_coords[0] = input_coords[0];
        output_coords[1] = input_coords[1] ^ (input_coords[0] % static_cast<CoordT>(state.length_1));
    }
};


// -- PAD ------------------------------------------------------------------
//
// Extend a dimension by left_pad + right_pad zeros.
//
// Pool layout (2 values):
//   [base + 0]: left_pad   (int32 literal | binding)
//   [base + 1]: right_pad  (int32 literal | binding)

template <typename Precision>
struct TransformImpl<TransformType::PAD, Precision>
{
    using LengthT = typename Precision::LengthT;
    /// Pool layout (2 values): Schema, member-pointer NTTP path.
    struct Schema
    {
        typename Precision::LengthT left_pad;
        typename Precision::LengthT right_pad;

        using members = detail::MemberPtrList<&Schema::left_pad,
                                              &Schema::right_pad>;
    };

    /// Phase A: output edge length = in_lens[0] + left_pad + right_pad.
    ///
    /// Lengths invariant: the returned padded length is non-negative; the
    /// pad fields are non-negative by construction. UnsignedT throughout.
    CK_TILE_HOST_DEVICE static constexpr LengthT
    deriveOutputLength(const Schema&   s,
                       EdgeLengthsView<LengthT> edges,
                       DimIds          input_edge_ids,
                       uint8_t         /*derived_output_idx*/) noexcept
    {
        return edges[input_edge_ids[0]]
             + static_cast<LengthT>(s.left_pad)
             + static_cast<LengthT>(s.right_pad);
    }

    /// Phase-C state: left_pad (for shift) + unpadded_length (for isInBounds).
    struct State
    {
        typename Precision::Signed left_pad        = 0;
        typename Precision::Signed unpadded_length = 0;
    };

    /// Phase B: snapshot left_pad from Schema; unpadded length from in_lens.
    /// in_lens carries the precision LengthT (lengths invariant); narrow to
    /// the signed coordinate type for the State fields used by mapCoord.
    template <uint8_t NIn, uint8_t NOut>
    CK_TILE_HOST_DEVICE static constexpr State
    resolveState(const Schema&                                                  s,
                 const ck_tile::static_array<typename Precision::LengthT, NIn>&  in_lens,
                 const ck_tile::static_array<typename Precision::LengthT, NOut>& /*out_lens*/) noexcept
    {
        return State{.left_pad        = static_cast<typename Precision::Signed>(s.left_pad),
                     .unpadded_length = static_cast<typename Precision::Signed>(in_lens[0])};
    }

    /// Phase C: shift coord by -left_pad. Out-of-range user coords map to
    /// out-of-range memory coords; user must guard with isInBounds().
    template <uint8_t /*NIn*/, uint8_t /*NOut*/, typename CoordT>
    CK_TILE_HOST_DEVICE static constexpr void
    mapCoord(const State&  state,
             CoordT*       output_coords,
             const CoordT* input_coords) noexcept
    {
        output_coords[0] = input_coords[0] - static_cast<CoordT>(state.left_pad);
    }

    /// @brief Optional bounds-check helper. Not part of the per-Impl contract.
    template <typename CoordT>
    CK_TILE_HOST_DEVICE static constexpr bool
    isInBounds(const State& state, const CoordT* input_coords) noexcept
    {
        const auto c = input_coords[0] - static_cast<CoordT>(state.left_pad);
        return c >= CoordT{0} && c < static_cast<CoordT>(state.unpadded_length);
    }
};


// -- UNMERGE --------------------------------------------------------------
//
// Combine N input dims into 1 output dim via positional weighting:
//
//   output[0] = sum_{i in [0, N)} input[i] * derived_strides[i]
//
// Inverse of MERGE: where MERGE decomposes one dim into N components,
// UNMERGE combines N components back into one address. component_lengths
// are authored on the INPUT side; derived_strides are right-to-left
// running products (innermost dim has stride 1).
//
// Pool layout (1 + N values, where N = ndim_input):
//   [base + 0]:           ndim_components             (int32 literal)
//   [base + 1 .. 1+N):    component_lengths[i]        (int32 literal | binding)

template <typename Precision>
struct TransformImpl<TransformType::UNMERGE, Precision>
{
    using LengthT = typename Precision::LengthT;
    /// Pool layout (1 + N values): Schema, member-pointer NTTP path.
    struct Schema
    {
        typename Precision::Signed  ndim_components;
        typename Precision::LengthT component_lengths[MAX_TENSOR_DIMS_V4];

        using members = detail::MemberPtrList<&Schema::ndim_components,
                                              &Schema::component_lengths>;
    };

    /// Phase A: output edge length = product of component lengths.
    ///
    /// Reads via `edges` + `input_edge_ids` (P3 of V4 scratch fix). UNMERGE's
    /// output length comes from the Schema's component_lengths field --
    /// the framework params are unused but retained for uniformity.
    CK_TILE_HOST_DEVICE static constexpr typename Precision::LengthT
    deriveOutputLength(const Schema&   s,
                       EdgeLengthsView<typename Precision::LengthT> /*edges*/,
                       DimIds          /*input_edge_ids*/,
                       uint8_t         /*derived_output_idx*/) noexcept
    {
        return detail::componentLengthProduct<Precision>(s.ndim_components,
                                                         s.component_lengths);
    }

    /// Phase-C state: derived strides (right-to-left running product).
    template <uint8_t NIn>
    struct State
    {
        ck_tile::static_array<typename Precision::Signed, NIn> derived_strides{};
    };

    /// Phase B: precompute derived strides from component lengths.
    template <uint8_t NIn, uint8_t NOut>
    CK_TILE_HOST_DEVICE static constexpr State<NIn>
    resolveState(const Schema&                                                  s,
                 const ck_tile::static_array<typename Precision::LengthT, NIn>&  /*in_lens*/,
                 const ck_tile::static_array<typename Precision::LengthT, NOut>& /*out_lens*/) noexcept
    {
        static_assert(NIn >= 2, "UNMERGE requires ndim_input >= 2");
        State<NIn> state{};

        // Snapshot component lengths, then compute derived strides via the
        // shared right-to-left running product (overflow-checked in IntAccum).
        ck_tile::static_array<typename Precision::LengthT, NIn> lens{};
        for(uint8_t i = 0; i < NIn; ++i) {
            lens[i] = s.component_lengths[i];
        }
        state.derived_strides = detail::rightToLeftStrides<Precision, NIn>(lens);
        return state;
    }

    /// Phase C: linear weighted sum (combine N components into 1 coord).
    /// Hot path. Compiler unrolls the loop at -O3 since NIn is template-bound.
    template <uint8_t NIn, uint8_t /*NOut*/, typename CoordT>
    CK_TILE_HOST_DEVICE static constexpr void
    mapCoord(const State<NIn>& state,
             CoordT*           output_coords,
             const CoordT*     input_coords) noexcept
    {
        static_assert(NIn >= 2, "UNMERGE requires ndim_input >= 2");
        // Index-pack fold (mirrors V3's unmergeFold).
        output_coords[0] = unmergeFold(state, input_coords,
                                       ck_tile::make_index_sequence<NIn>{});
    }

private:
    template <uint8_t NIn, ck_tile::index_t... Is, typename CoordT>
    CK_TILE_HOST_DEVICE static constexpr CoordT
    unmergeFold(const State<NIn>& state, const CoordT* input_coords,
                ck_tile::sequence<Is...>) noexcept
    {
        // No cast on state.derived_strides[Is]: see EMBED for reasoning.
        return ((input_coords[Is] * state.derived_strides[Is]) + ... + CoordT{0});
    }
};


// -- FREEZE ---------------------------------------------------------------
//
// Pin a single produced dim to a constant index. Reads no input, writes
// one output coordinate that is always frozen_idx. The produced dim has
// length 1 (only the frozen position is reachable).
//
// Pool layout (1 value):
//   [base + 0]: frozen_idx   (int32 literal | binding)

template <typename Precision>
struct TransformImpl<TransformType::FREEZE, Precision>
{
    using LengthT = typename Precision::LengthT;
    /// Pool layout (1 value): Schema, member-pointer NTTP path.
    struct Schema
    {
        typename Precision::Signed frozen_idx;

        using members = detail::MemberPtrList<&Schema::frozen_idx>;
    };

    /// Phase-C state: just the frozen index.
    struct State
    {
        typename Precision::Signed frozen_idx = 0;
    };

    template <uint8_t NIn, uint8_t NOut>
    CK_TILE_HOST_DEVICE static constexpr State
    resolveState(const Schema&                                                  s,
                 const ck_tile::static_array<typename Precision::LengthT, NIn>&  /*in_lens*/,
                 const ck_tile::static_array<typename Precision::LengthT, NOut>& /*out_lens*/) noexcept
    {
        return State{s.frozen_idx};
    }

    template <uint8_t /*NIn*/, uint8_t /*NOut*/, typename CoordT>
    CK_TILE_HOST_DEVICE static constexpr void
    mapCoord(const State&  state,
             CoordT*       output_coords,
             const CoordT* /*input_coords*/) noexcept
    {
        output_coords[0] = static_cast<CoordT>(state.frozen_idx);
    }
};


// -- SLICE ----------------------------------------------------------------
//
// Bijective sub-range selector. Maps user-side index u in [0, end-begin)
// to memory-side index u + begin. Lengths are not anchored by SLICE: the
// user declares the user-side length via inputs() on the upstream slot.
//
//   memory dim:  +------------------------------+
//                | ... A B C D E F G H I J ...  |   length M
//                +------------------------------+
//                       ^begin            ^end (exclusive)
//   user dim:           +-----------+
//                       | A B C D E |               length = end - begin
//                       +-----------+
//
// Pool layout (1 value):
//   [base + 0]: begin   (int32 literal | binding | edge_length)

template <typename Precision>
struct TransformImpl<TransformType::SLICE, Precision>
{
    using LengthT = typename Precision::LengthT;
    /// Pool layout (1 value): Schema, member-pointer NTTP path.
    struct Schema
    {
        typename Precision::Signed begin;

        using members = detail::MemberPtrList<&Schema::begin>;
    };

    /// Phase-C state: the begin offset.
    struct State
    {
        typename Precision::Signed begin = 0;
    };

    template <uint8_t NIn, uint8_t NOut>
    CK_TILE_HOST_DEVICE static constexpr State
    resolveState(const Schema&                                                  s,
                 const ck_tile::static_array<typename Precision::LengthT, NIn>&  /*in_lens*/,
                 const ck_tile::static_array<typename Precision::LengthT, NOut>& /*out_lens*/) noexcept
    {
        return State{s.begin};
    }

    template <uint8_t /*NIn*/, uint8_t /*NOut*/, typename CoordT>
    CK_TILE_HOST_DEVICE static constexpr void
    mapCoord(const State&  state,
             CoordT*       output_coords,
             const CoordT* input_coords) noexcept
    {
        output_coords[0] = input_coords[0] + static_cast<CoordT>(state.begin);
    }
};


// -- BROADCAST ------------------------------------------------------------
//
// Variable-arity input, zero outputs. No-op coordinate mapping; used as a
// topology marker to declare that N input dims are read but produce
// nothing observable downstream. (Useful for bias dims, attention masks,
// etc. where the consumer indexes into a tensor that ignores the
// broadcast axis.)
//
// Pool layout (0 values): BROADCAST carries no Schema state.

template <typename Precision>
struct TransformImpl<TransformType::BROADCAST, Precision>
{
    using LengthT = typename Precision::LengthT;
    /// Pool layout: empty. See XOR for the rationale on omitting HUOR.
    struct Schema
    {
        using members = detail::MemberPtrList<>;
    };

    /// Phase-C state: empty (BROADCAST has no per-coord state).
    /// 1-byte placeholder gives the framework's per-transform State tuple
    /// something addressable to return by reference. SROA discards at -O3.
    struct State
    {
        [[maybe_unused]] uint8_t _unused = 0;
    };

    template <uint8_t NIn, uint8_t NOut>
    CK_TILE_HOST_DEVICE static constexpr State
    resolveState(const Schema&                                                  /*s*/,
                 const ck_tile::static_array<typename Precision::LengthT, NIn>&  /*in_lens*/,
                 const ck_tile::static_array<typename Precision::LengthT, NOut>& /*out_lens*/) noexcept
    {
        return {};
    }

    template <uint8_t /*NIn*/, uint8_t /*NOut*/, typename CoordT>
    CK_TILE_HOST_DEVICE static constexpr void
    mapCoord(const State&  /*state*/,
             CoordT*       /*output_coords*/,
             const CoordT* /*input_coords*/) noexcept
    {
        // No-op: BROADCAST writes no output coords.
    }
};


// =============================================================================
// 10. TransformArgs and factories (SlotT-agnostic; graph owns the width)
// =============================================================================
//
// Former Q5/Q6 width-deduction machinery (SlotTypeOf, DeduceCommonSlotT,
// DeduceGraphSlotT, fillAnchorsFromTuple) was deleted as part of the
// graph-owned-SlotT direction-reversal. Factories now capture raw user-typed
// args into a TransformArgs<Ts...> POD; SlotT commitment happens inside
// insertTransform / insertNode via Slot::from_raw, where the graph's
// declared SlotT (via `make_transform_graph<SlotT>(...)`) is known.
//
// is_placeholder<T> / is_placeholder_v<T> are defined earlier in the file
// (near Slot's forward-declared body) so Slot::from_raw can name them.


// ---- DimsList / StridesList -----------------------------------
//
// Typed argument-pack wrappers used ONLY by make_embed (the only factory
// that takes TWO variadic lists). Each carries a tuple of literal-or-
// placeholder args. The user-facing factories `dims(...)` and `strides(...)`
// build these.
//
// Why typed wrappers instead of two raw packs: C++ variadic templates can
// only carry ONE pack per signature. Using two distinct types lets
// make_embed take both lengths and strides in one call:
//
//   auto t = make_embed(dims(M, N), strides(N, 1));
//
// Single-list factories (make_merge, make_unmerge) take their one variadic
// pack DIRECTLY without a wrapper -- no disambiguation needed. See section
// 1.6 lever L7 of V4_REWORK_PLAN for the consistency rationale.

/**
 * @brief Compile-time list of N user-supplied dim lengths, pre-committed
 *        to Slot bytes at construction so that downstream factories can
 *        copy slots via plain `dl[k]` indexing without tuple gymnastics.
 *
 * @tparam N  Arity. Bounded by MAX_TENSOR_DIMS_V4 (asserted in `dims(...)`).
 *
 * Inheriting `static_array<Slot, MAX_TENSOR_DIMS_V4>` mirrors the
 * wrapper-collapse pattern already used by Pool / RuntimeBindings /
 * EdgeLengths / TransformArgs (memory: project_v4_wrapper_collapse_2026_05_22):
 * one fixed-size buffer, one uniform `[i]` accessor, no `.values.template
 * get<Is>()` for users to write.
 */
template <uint8_t N>
struct DimsList : ck_tile::static_array<Slot, MAX_TENSOR_DIMS_V4>
{
    static constexpr uint8_t COUNT = N;
};

/**
 * @brief Pre-commit user arg values to Slot bytes (one slot per arg) in
 *        declaration order via an i++ pack-fold. `consteval` enforces
 *        graph-construction-only use: the v4 DSL bakes an NTTP graph at
 *        compile time; runtime values flow in via `placeholder<Id>`.
 */
template <typename... Ts>
consteval auto dims(Ts... vs) noexcept
{
    static_assert(sizeof...(Ts) <= MAX_TENSOR_DIMS_V4,
                  "dims(...) arity exceeds MAX_TENSOR_DIMS_V4");
    DimsList<sizeof...(Ts)> r{};
    uint8_t i = 0;
    ((r[i++] = Slot::from_raw(vs)), ...);
    return r;
}

/**
 * @brief Compile-time list of N user-supplied dim strides. Distinct type
 *        from DimsList<N> so `make_embed(dims(...), strides(...), ...)`
 *        signature disambiguates positionally.
 *
 * @tparam N  Arity. Bounded by MAX_TENSOR_DIMS_V4 (asserted in `strides(...)`).
 */
template <uint8_t N>
struct StridesList : ck_tile::static_array<Slot, MAX_TENSOR_DIMS_V4>
{
    static constexpr uint8_t COUNT = N;
};

/// @brief StridesList<sizeof...(Ts)> twin of `dims(...)` — same pre-commit
///        pattern; see DimsList docs above.
template <typename... Ts>
consteval auto strides(Ts... vs) noexcept
{
    static_assert(sizeof...(Ts) <= MAX_TENSOR_DIMS_V4,
                  "strides(...) arity exceeds MAX_TENSOR_DIMS_V4");
    StridesList<sizeof...(Ts)> r{};
    uint8_t i = 0;
    ((r[i++] = Slot::from_raw(vs)), ...);
    return r;
}

// CompList / components() removed: MERGE/UNMERGE take variadic raw values
// directly (no wrapper needed since they have only one list). See
// V4_REWORK_PLAN section 1.6 lever L7. EMBED retains DimsList/StridesList
// because it has TWO lists and needs type-level disambiguation.

// Read<N>/Write<N> are defined later in the user-facing builder API section,
// but the factories below need them as parameter types. Forward-declare here;
// instantiation only happens at user call sites, by which point the full
// definitions are visible.
template <uint8_t N> struct Read;
template <uint8_t N> struct Write;

/**
 * @brief Worst-case per-transform args slot count -- EMBED with the maximum
 *        supported arity contributes `1 + 2 * MAX_TENSOR_DIMS_V4` slots
 *        (one ndim + that many lengths + that many strides). The fixed
 *        buffer size lets TransformNode stay non-templated.
 */
inline constexpr ck_tile::index_t MAX_ARGS_V4 = 1 + 2 * MAX_TENSOR_DIMS_V4;
static_assert(MAX_ARGS_V4 <= 255,
              "MAX_ARGS_V4 too large for uint8_t-indexed TransformNode");

/**
 * @brief Per-transform declaration: slot bytes pre-committed at factory
 *        time, plus the boundary edge routing (`read_ids` / `write_ids`)
 *        that connects this transform into the surrounding graph.
 *
 * Concrete (non-templated) POD. Consumed by `insertNode` ->
 * `insertTransform`, which memcpys the live prefix `args[0 .. args_count)`
 * into `graph.pool`. Lives only during `make_transform_graph(...)`
 * consteval; never appears in the final TransformGraph NTTP value
 * (the structural per-transform record is `CoordinateTransform`).
 *
 * Pool layout per Impl (committed in the listed order):
 *   OFFSET        : [shift]
 *   FREEZE        : [frozen_idx]
 *   SLICE         : [begin]
 *   PAD           : [left_pad, right_pad]
 *   XOR           : [] (empty)
 *   BROADCAST     : [] (empty; arity comes from read/write counts)
 *   EMBED         : [N, L0..L_{N-1}, S0..S_{N-1}]
 *   MERGE/UNMERGE : [N, L0..L_{N-1}]
 */
struct TransformNode
{
    TransformType                            transform_type = TransformType::OFFSET;
    ck_tile::static_array<Slot, MAX_ARGS_V4> args{};
    uint8_t                                  args_count   = 0;
    uint8_t                                  ndim_input   = 0;
    uint8_t                                  ndim_output  = 0;
    DimIds                                   read_ids{};
    uint8_t                                  read_count   = 0;
    DimIds                                   write_ids{};
    uint8_t                                  write_count  = 0;

    /**
     * @brief Per-position edge anchors authored by the factory.
     *
     * Each Slot at `input_anchors[i]` (i < ndim_input) or
     * `output_anchors[i]` (i < ndim_output) declares how the framework
     * should resolve the corresponding edge's length:
     *
     *   - `Kind::UNUSED`     : factory did not author an anchor here; the
     *                          edge is filled by whichever transform
     *                          produces it.
     *   - `Kind::VALUE`      : literal length in `payload`.
     *   - `Kind::BINDING_ID` : runtime placeholder; `payload` indexes
     *                          `RuntimeBindings`.
     *   - `Kind::EDGE_ID`    : copy the resolved length from the edge id
     *                          carried in `payload`.
     *   - `Kind::DERIVED`    : length is computed by the owning Impl's
     *                          `deriveInputLength` / `deriveOutputLength`.
     *
     * Positions past the active prefix stay default-init (`Kind::UNUSED`).
     */
    ck_tile::static_array<Slot, MAX_TENSOR_DIMS_V4> input_anchors{};
    ck_tile::static_array<Slot, MAX_TENSOR_DIMS_V4> output_anchors{};

    // True iff any input / output anchor authored by the factory is
    // Kind::DERIVED. The factory always knows this at construction time, so
    // we store it directly instead of forcing insertTransform to re-scan the
    // anchor arrays. Read by insertTransform when it OR's the bit into
    // graph.t_{input,output}_derived_mask.
    bool has_input_derived  = false;
    bool has_output_derived = false;
};


// =============================================================================
// 10a. Per-Impl factories
// =============================================================================
//
// Each `make_X(args..., read(...), write(...))` factory commits slot bytes
// for its X-specific args immediately into a stack-local `args` buffer (no
// tuple-staged raw_args, no late commit in `insertTransform`), then
// constructs the TransformNode via designated init. `Read<ReadCount>` and
// `Write<WriteCount>` trail as positionally-pinned non-variadic params -- their
// distinct types prevent silent swap-routing if a caller writes the wrong
// one. Arity is pinned from ReadCount/WriteCount at factory time, so the framework no
// longer needs an `ARITY_FROM_BUILDER` sentinel or late-binding patch.

// Concept: Impl provides a deriveOutputLength method. A factory that
// authors a Slot::from_derived() OUTPUT anchor static_asserts on this for
// its TransformImpl<X>. Also used by resolveOneTransformEdges to gate the
// Pass-2 dispatch path.
template <typename Impl>
concept ImplHasDeriveOutputLength = requires(typename Impl::Schema s,
                                             EdgeLengthsView<typename Impl::LengthT>       e,
                                             DimIds                ids,
                                             uint8_t               idx) {
    Impl::deriveOutputLength(s, e, ids, idx);
};

// Sister concept for deriveInputLength. Authored by factories whose input
// anchor is Slot::from_derived() (currently only MERGE).
template <typename Impl>
concept ImplHasDeriveInputLength = requires(typename Impl::Schema s,
                                            EdgeLengthsView<typename Impl::LengthT>       e,
                                            DimIds                ids,
                                            uint8_t               idx) {
    Impl::deriveInputLength(s, e, ids, idx);
};

/**
 * @brief Construct an OFFSET coordinate transform: 1 dim -> 1 dim.
 *
 * @param shift  Literal arithmetic value or `placeholder<Id>` for a
 *               runtime-bound shift.
 * @param r      `read(edge_id)` -- single source edge.
 * @param w      `write(edge_id)` -- single sink edge.
 */
template <typename T, uint8_t ReadCount, uint8_t WriteCount>
consteval TransformNode make_offset(T shift, Read<ReadCount> r, Write<WriteCount> w) noexcept
{
    static_assert(ReadCount == 1 && WriteCount == 1, "make_offset: arity must be 1 -> 1");
    ck_tile::static_array<Slot, MAX_ARGS_V4> args{};
    uint8_t i = 0;
    args[i++] = Slot::from_raw(shift);
    // Output dim's length equals the single input dim's length (shift does
    // not stretch or shrink the dim).
    ck_tile::static_array<Slot, MAX_TENSOR_DIMS_V4> output_anchors{};
    output_anchors[0] = Slot::from_edge_id(r.ids[0]);
    return TransformNode{
        .transform_type = TransformType::OFFSET,
        .args           = args,
        .args_count     = i,
        .ndim_input     = 1,
        .ndim_output    = 1,
        .read_ids       = r.ids,
        .read_count     = ReadCount,
        .write_ids      = w.ids,
        .write_count    = WriteCount,
        .output_anchors = output_anchors
    };
}

/**
 * @brief Construct a FREEZE coordinate transform: 0 dims -> 1 dim. Pins
 *        the output dim to a constant; the output dim length is 1.
 *
 * @param frozen_idx  Literal value or placeholder.
 * @param r           `read()` -- empty input.
 * @param w           `write(edge_id)` -- single sink edge.
 */
template <typename T, uint8_t ReadCount, uint8_t WriteCount>
consteval TransformNode make_freeze(T frozen_idx, Read<ReadCount> r, Write<WriteCount> w) noexcept
{
    static_assert(ReadCount == 0 && WriteCount == 1, "make_freeze: arity must be 0 -> 1");
    ck_tile::static_array<Slot, MAX_ARGS_V4> args{};
    uint8_t i = 0;
    args[i++] = Slot::from_raw(frozen_idx);
    // Output dim has length 1: only the frozen position is reachable. Authored
    // precision-agnostically (a width-neutral unsigned length literal); the
    // induction at insertTransform re-encodes it to the graph precision, so the
    // factory names no precision policy.
    ck_tile::static_array<Slot, MAX_TENSOR_DIMS_V4> output_anchors{};
    output_anchors[0] = Slot::from_value(uint32_t{1});
    return TransformNode{
        .transform_type = TransformType::FREEZE,
        .args           = args,
        .args_count     = i,
        .ndim_input     = 0,
        .ndim_output    = 1,
        .read_ids       = r.ids,
        .read_count     = ReadCount,
        .write_ids      = w.ids,
        .write_count    = WriteCount,
        .output_anchors = output_anchors
    };
}

/**
 * @brief Construct a SLICE coordinate transform: 1 dim -> 1 dim.
 *
 * @param begin  Inclusive lower bound (literal or placeholder).
 * @param end    Exclusive upper bound. Used only at compile time for the
 *               literal-pair `begin <= end` check; runtime mapping uses
 *               only `begin`. The caller owns the user-side slice length.
 * @param r      `read(edge_id)`.
 * @param w      `write(edge_id)`.
 * @pre  When both args are non-negative literals: `begin <= end`.
 */
template <typename TBegin, typename TEnd, uint8_t ReadCount, uint8_t WriteCount>
consteval TransformNode make_slice(TBegin begin, TEnd end, Read<ReadCount> r, Write<WriteCount> w) noexcept
{
    static_assert(ReadCount == 1 && WriteCount == 1, "make_slice: arity must be 1 -> 1");
    if constexpr(std::is_arithmetic_v<TBegin> && std::is_arithmetic_v<TEnd>) {
        if(begin > end) { transformErrorSliceBeginAfterEnd(); }
    }
    (void)end;
    ck_tile::static_array<Slot, MAX_ARGS_V4> args{};
    uint8_t i = 0;
    args[i++] = Slot::from_raw(begin);
    // SLICE does not anchor its own user-side length: the user supplies
    // it externally (typically via `inputs(dims(end - begin))`). Output
    // edge takes the input's resolved length here so downstream consumers
    // see a non-UNUSED anchor.
    ck_tile::static_array<Slot, MAX_TENSOR_DIMS_V4> output_anchors{};
    output_anchors[0] = Slot::from_edge_id(r.ids[0]);
    return TransformNode{
        .transform_type = TransformType::SLICE,
        .args           = args,
        .args_count     = i,
        .ndim_input     = 1,
        .ndim_output    = 1,
        .read_ids       = r.ids,
        .read_count     = ReadCount,
        .write_ids      = w.ids,
        .write_count    = WriteCount,
        .output_anchors = output_anchors
    };
}

/**
 * @brief Construct a PAD coordinate transform: 1 dim -> 1 dim.
 *
 * @param left_pad   Pad amount on the low side (literal or placeholder).
 * @param right_pad  Pad amount on the high side.
 */
template <typename TLeft, typename TRight, uint8_t ReadCount, uint8_t WriteCount>
consteval TransformNode
make_pad(TLeft left_pad, TRight right_pad, Read<ReadCount> r, Write<WriteCount> w) noexcept
{
    static_assert(ReadCount == 1 && WriteCount == 1, "make_pad: arity must be 1 -> 1");
    ck_tile::static_array<Slot, MAX_ARGS_V4> args{};
    uint8_t i = 0;
    args[i++] = Slot::from_raw(left_pad);
    args[i++] = Slot::from_raw(right_pad);
    // Output length = input + left + right; computed at Phase A.5 by
    // PAD's deriveOutputLength. Asserted at the factory site so the
    // contract "if you author Slot::from_derived(), your Impl must define
    // deriveX" is checked exactly where the from_derived() call lives.
    static_assert(ImplHasDeriveOutputLength<TransformImpl<TransformType::PAD>>,
                  "make_pad: TransformImpl<PAD> must define deriveOutputLength.");
    ck_tile::static_array<Slot, MAX_TENSOR_DIMS_V4> output_anchors{};
    output_anchors[0] = Slot::from_derived();
    return TransformNode{
        .transform_type      = TransformType::PAD,
        .args                = args,
        .args_count          = i,
        .ndim_input          = 1,
        .ndim_output         = 1,
        .read_ids            = r.ids,
        .read_count          = ReadCount,
        .write_ids           = w.ids,
        .write_count         = WriteCount,
        .output_anchors      = output_anchors,
        .has_output_derived  = true
    };
}

/**
 * @brief Construct an XOR coordinate transform: 2 dims -> 2 dims. No
 *        per-instance Schema state; length_1 is supplied by the framework
 *        via my_input_edge_lengths[1] at Phase B.
 */
template <uint8_t ReadCount, uint8_t WriteCount>
consteval TransformNode make_xor(Read<ReadCount> r, Write<WriteCount> w) noexcept
{
    static_assert(ReadCount == 2 && WriteCount == 2, "make_xor: arity must be 2 -> 2");
    // XOR scrambles indices within the existing length space; neither dim
    // changes length. Each output positionally inherits its input's length.
    ck_tile::static_array<Slot, MAX_TENSOR_DIMS_V4> output_anchors{};
    output_anchors[0] = Slot::from_edge_id(r.ids[0]);
    output_anchors[1] = Slot::from_edge_id(r.ids[1]);
    return TransformNode{
        .transform_type = TransformType::XOR,
        .args_count     = 0,
        .ndim_input     = 2,
        .ndim_output    = 2,
        .read_ids       = r.ids,
        .read_count     = ReadCount,
        .write_ids      = w.ids,
        .write_count    = WriteCount,
        .output_anchors = output_anchors
    };
}

/**
 * @brief Construct an EMBED coordinate transform: N user dims -> 1 address.
 *
 * @param ls  `dims(L0, L1, ...)` -- per-dim lengths (literal or placeholder).
 * @param ss  `strides(S0, S1, ...)` -- per-dim strides (literal or placeholder).
 * @pre  `ls.COUNT == ss.COUNT == read(...) arity`. `write(...) arity == 1`.
 *
 * Pool layout: [N, L0..L_{N-1}, S0..S_{N-1}].
 */
template <uint8_t DimsCount, uint8_t StridesCount, uint8_t ReadCount, uint8_t WriteCount>
consteval TransformNode
make_embed(DimsList<DimsCount> ls, StridesList<StridesCount> ss, Read<ReadCount> r, Write<WriteCount> w) noexcept
{
    static_assert(DimsCount == StridesCount,
                  "make_embed: dims(...) and strides(...) must have the same arity");
    static_assert(DimsCount >= 1 && DimsCount <= MAX_TENSOR_DIMS_V4,
                  "make_embed: arity must be in [1, MAX_TENSOR_DIMS_V4]");
    static_assert(ReadCount == DimsCount, "make_embed: read(...) arity must equal dims(...) arity");
    static_assert(WriteCount == 1,  "make_embed: write(...) arity must be 1");
    ck_tile::static_array<Slot, MAX_ARGS_V4> args{};
    uint8_t i = 0;
    args[i++] = Slot::from_raw(static_cast<ck_tile::index_t>(DimsCount));
    for(uint8_t k = 0; k < DimsCount; ++k) { args[i++] = ls[k]; }
    for(uint8_t k = 0; k < DimsCount; ++k) { args[i++] = ss[k]; }
    // Per-input-dim length: the user-supplied dim slot is itself the
    // anchor (literal or runtime binding).
    ck_tile::static_array<Slot, MAX_TENSOR_DIMS_V4> input_anchors{};
    for(uint8_t k = 0; k < DimsCount; ++k) { input_anchors[k] = ls[k]; }
    // Output address span = 1 + sum_i (in_lens[i] - 1) * strides[i];
    // computed at Phase A.5 by EMBED's deriveOutputLength.
    static_assert(ImplHasDeriveOutputLength<TransformImpl<TransformType::EMBED>>,
                  "make_embed: TransformImpl<EMBED> must define deriveOutputLength.");
    ck_tile::static_array<Slot, MAX_TENSOR_DIMS_V4> output_anchors{};
    output_anchors[0] = Slot::from_derived();
    return TransformNode{
        .transform_type      = TransformType::EMBED,
        .args                = args,
        .args_count          = i,
        .ndim_input          = DimsCount,
        .ndim_output         = 1,
        .read_ids            = r.ids,
        .read_count          = ReadCount,
        .write_ids           = w.ids,
        .write_count         = WriteCount,
        .input_anchors       = input_anchors,
        .output_anchors      = output_anchors,
        .has_output_derived  = true
    };
}

/**
 * @brief Construct a MERGE coordinate transform: 1 input dim -> N component
 *        output dims that flatten the input.
 *
 * @param ls  `dims(L0, L1, ...)` -- per-component output lengths.
 * @pre  `ls.COUNT == write(...) arity`. `read(...) arity == 1`.
 *
 * Pool layout: [N, L0..L_{N-1}].
 */
template <uint8_t DimsCount, uint8_t ReadCount, uint8_t WriteCount>
consteval TransformNode
make_merge(DimsList<DimsCount> ls, Read<ReadCount> r, Write<WriteCount> w) noexcept
{
    static_assert(DimsCount >= 2 && DimsCount <= MAX_TENSOR_DIMS_V4,
                  "make_merge: arity must be in [2, MAX_TENSOR_DIMS_V4]");
    static_assert(ReadCount == 1,  "make_merge: read(...) arity must be 1");
    static_assert(WriteCount == DimsCount, "make_merge: write(...) arity must equal dims(...) arity");
    ck_tile::static_array<Slot, MAX_ARGS_V4> args{};
    uint8_t i = 0;
    args[i++] = Slot::from_raw(static_cast<ck_tile::index_t>(DimsCount));
    for(uint8_t k = 0; k < DimsCount; ++k) { args[i++] = ls[k]; }
    // Input length = product of component lengths; computed at Phase A.5
    // by MERGE's deriveInputLength.
    static_assert(ImplHasDeriveInputLength<TransformImpl<TransformType::MERGE>>,
                  "make_merge: TransformImpl<MERGE> must define deriveInputLength.");
    ck_tile::static_array<Slot, MAX_TENSOR_DIMS_V4> input_anchors{};
    input_anchors[0] = Slot::from_derived();
    // Each output dim's length is the corresponding user-supplied
    // component length.
    ck_tile::static_array<Slot, MAX_TENSOR_DIMS_V4> output_anchors{};
    for(uint8_t k = 0; k < DimsCount; ++k) { output_anchors[k] = ls[k]; }
    return TransformNode{
        .transform_type     = TransformType::MERGE,
        .args               = args,
        .args_count         = i,
        .ndim_input         = 1,
        .ndim_output        = DimsCount,
        .read_ids           = r.ids,
        .read_count         = ReadCount,
        .write_ids          = w.ids,
        .write_count        = WriteCount,
        .input_anchors      = input_anchors,
        .output_anchors     = output_anchors,
        .has_input_derived  = true
    };
}

/**
 * @brief Construct an UNMERGE coordinate transform: N component input dims
 *        -> 1 output dim.
 *
 * @param ls  `dims(L0, L1, ...)` -- per-component input lengths.
 * @pre  `ls.COUNT == read(...) arity`. `write(...) arity == 1`.
 *
 * Pool layout: [N, L0..L_{N-1}].
 */
template <uint8_t DimsCount, uint8_t ReadCount, uint8_t WriteCount>
consteval TransformNode
make_unmerge(DimsList<DimsCount> ls, Read<ReadCount> r, Write<WriteCount> w) noexcept
{
    static_assert(DimsCount >= 2 && DimsCount <= MAX_TENSOR_DIMS_V4,
                  "make_unmerge: arity must be in [2, MAX_TENSOR_DIMS_V4]");
    static_assert(ReadCount == DimsCount, "make_unmerge: read(...) arity must equal dims(...) arity");
    static_assert(WriteCount == 1,  "make_unmerge: write(...) arity must be 1");
    ck_tile::static_array<Slot, MAX_ARGS_V4> args{};
    uint8_t i = 0;
    args[i++] = Slot::from_raw(static_cast<ck_tile::index_t>(DimsCount));
    for(uint8_t k = 0; k < DimsCount; ++k) { args[i++] = ls[k]; }
    // Each input dim's length is the corresponding user-supplied
    // component length.
    ck_tile::static_array<Slot, MAX_TENSOR_DIMS_V4> input_anchors{};
    for(uint8_t k = 0; k < DimsCount; ++k) { input_anchors[k] = ls[k]; }
    // Output length = product of component lengths; computed at Phase A.5
    // by UNMERGE's deriveOutputLength.
    static_assert(ImplHasDeriveOutputLength<TransformImpl<TransformType::UNMERGE>>,
                  "make_unmerge: TransformImpl<UNMERGE> must define deriveOutputLength.");
    ck_tile::static_array<Slot, MAX_TENSOR_DIMS_V4> output_anchors{};
    output_anchors[0] = Slot::from_derived();
    return TransformNode{
        .transform_type      = TransformType::UNMERGE,
        .args                = args,
        .args_count          = i,
        .ndim_input          = DimsCount,
        .ndim_output         = 1,
        .read_ids            = r.ids,
        .read_count          = ReadCount,
        .write_ids           = w.ids,
        .write_count         = WriteCount,
        .input_anchors       = input_anchors,
        .output_anchors      = output_anchors,
        .has_output_derived  = true
    };
}

/**
 * @brief Construct a BROADCAST coordinate transform: N input dims -> 0
 *        output dims (sentinel transform; no per-instance Schema state).
 *        Arity is pinned from the `read(...)` count at factory time.
 *
 * @pre  `write(...) arity == 0`. `read(...) arity` in [1, MAX_TENSOR_DIMS_V4].
 */
template <uint8_t ReadCount, uint8_t WriteCount>
consteval TransformNode make_broadcast(Read<ReadCount> r, Write<WriteCount> w) noexcept
{
    static_assert(WriteCount == 0, "make_broadcast: write(...) arity must be 0");
    static_assert(ReadCount >= 1 && ReadCount <= MAX_TENSOR_DIMS_V4,
                  "make_broadcast: read(...) arity must be in [1, MAX_TENSOR_DIMS_V4]");
    return TransformNode{
        .transform_type = TransformType::BROADCAST,
        .args_count     = 0,
        .ndim_input     = ReadCount,
        .ndim_output    = 0,
        .read_ids       = r.ids,
        .read_count     = ReadCount,
        .write_ids      = w.ids,
        .write_count    = WriteCount
    };
}


// =============================================================================
// 11. TransformGraph -- NTTP-baked graph structure
// =============================================================================

/// @brief NTTP-baked transform graph.
///
/// Carries:
///   - pool: AoS Slot storage (static_array<Slot, N>).
///   - transforms[]: per-transform records (type + base + count + arity).
///   - t_input_edge_ids[]/t_output_edge_ids[]: per-transform edge routing.
///   - input_edge_anchors / output_edge_anchors[]: Phase-A construction-time Value-typed anchors per edge.
///   - input_edge_ids/output_edge_ids: boundary-side edge ids.
///
/// (The per-transform routing arrays t_input_edge_ids / t_output_edge_ids and
/// input_edge_anchors / output_edge_anchors are required by make_graph_bindings, splice, and topo_order.
/// V3 carries them; V4 does too, sized down to uint8_t.)
struct TransformGraph
{
    using PoolType  = Pool;

    // MEMBER ORDERING IS DELIBERATE -- DESCENDING ALIGNMENT (NTTP INVARIANT).
    //
    // This type is used as a `template <auto G>` non-type template parameter.
    // It MUST have a unique object representation (no implicit padding holes):
    // a clang-22 IRGen bug synthesizes a per-NTTP-value constant struct type
    // from the value's bytes that OMITS interior padding fields, then emits
    // member-access GEPs using the canonical (padded) record's field indices --
    // off by one relative to the synthesized storage. The result is a runtime
    // member read returning an ADJACENT member's value. Eliminating implicit
    // padding (so canonical and synthesized layouts agree) sidesteps the bug.
    //
    // Members are therefore grouped by descending alignment:
    //   1. align-8 arrays (Slot-backed: Pool + anchor arrays). Each size is a
    //      multiple of 8, so the running offset stays 8-aligned with no gaps.
    //   2. the scalar block: four uint8 counts + two uint16 masks (8 bytes,
    //      uint16s naturally 2-aligned) + the precision tag + explicit
    //      reserved bytes that round the block to 16 with no implicit hole.
    //   3. align-1 byte arrays (transforms / edge-id / topo). No member needs
    //      alignment > 1, so they pack contiguously with no trailing gap.
    // The static_assert below guards the invariant against future reordering.

    // -- align-8: Slot-backed storage ------------------------------------
    PoolType pool{};

    // Per-edge transform anchors, indexed by global edge id (NOT by
    // [transform][position]). insertTransform writes each non-UNUSED anchor a
    // transform authored into t_input_edge_anchors[read_id] / t_output_edge_anchors
    // [write_id]; resolveOneTransformEdges reads them back via the per-transform
    // t_*_edge_ids routing. Kept on SEPARATE input/output arrays so a producer's
    // output anchor (e.g. DERIVED) and a consumer's input anchor (e.g. VALUE)
    // for the same internal edge never share a slot -- the resolution into
    // edge_lengths is still topo-order last-writer-wins, unchanged. Sized by
    // MAX_EDGES_V4 (the edge-id space), so anchor storage is decoupled from the
    // per-transform dim bound: 2 x 254 Slots here vs 2 x 12 x 64 for the old
    // [transform][position] arrays.
    ck_tile::static_array<Slot, MAX_EDGES_V4> t_input_edge_anchors{};
    ck_tile::static_array<Slot, MAX_EDGES_V4> t_output_edge_anchors{};

    // Phase-A boundary edge anchors, indexed by BOUNDARY POSITION (not edge id):
    // input_edge_anchors[k] pairs with input_edge_ids[k], output_edge_anchors[k]
    // with output_edge_ids[k] -- the same position used by the boundary folds.
    // Position indexing caps each at MAX_TENSOR_DIMS_V4 (the per-boundary dim
    // bound) instead of MAX_EDGES_V4, since a graph has at most that many input
    // and output boundary edges. Kept distinct from the per-transform anchors so
    // a declared graph-output dim and its producing transform's output anchor do
    // not contend for one slot.
    ck_tile::static_array<Slot, MAX_TENSOR_DIMS_V4> input_edge_anchors{};
    ck_tile::static_array<Slot, MAX_TENSOR_DIMS_V4> output_edge_anchors{};

    // -- scalar block: 4x IndexT (uint8) + 2x uint16 + precision tag + reserved --
    IndexT   pool_used      = 0;
    IndexT   num_transforms = 0;
    // Active counts -- populated by make_transform_graph after the dispatch
    // loop completes. Used to size per-graph runtime structures
    // (EdgeLengths<num_edges>, RuntimeBindings<num_bindings>) so the
    // kernel-side stack frame is sized to the actual graph rather than to
    // MAX_EDGES_V4 / MAX_BINDINGS_V4 worst-case constants.
    IndexT   num_edges      = 0;
    IndexT   num_bindings   = 0;
    // Bit i = 1 iff transform i authored any Kind::DERIVED output (resp. input)
    // anchor. Precomputed by insertTransform so
    // resolveOneTransformEdges can gate its Pass-2 work via a constexpr
    // bitmask test (`(G.t_output_derived_mask >> Idx) & 1u`) instead of two
    // per-(G, Idx) consteval lambdas -- those would create 2N unique closure
    // types per N outer (G, X, Idx) specializations.
    uint16_t t_output_derived_mask = 0;
    uint16_t t_input_derived_mask  = 0;

    /**
     * @brief Numeric precision selector (P32 default). Recovered to a policy
     *        type via `precision_t<G.precision>` by the `<auto G>` consumers;
     *        the width is applied at the write boundaries (insertTransform,
     *        make_graph_bindings), never on reads.
     */
    PrecisionTag precision = PrecisionTag::P32;

    /**
     * @brief Explicit tail padding. alignof(TransformGraph) is 8 (Slot-backed
     *        arrays), so the struct rounds its size up to a multiple of 8
     *        regardless; naming these bytes costs nothing and keeps
     *        has_unique_object_representations_v true (no IMPLICIT hole, which
     *        the static_assert below requires).
     *
     *        Forward-compat: a future BYTE-width scalar field may consume one
     *        of these bytes in place (shrink the array to match) with no layout
     *        change. A WIDER scalar (uint16/uint32) must NOT be carved from here
     *        -- it would break the descending-alignment ordering and reopen the
     *        implicit-padding hole; add wider scalars to the uint16 mask group
     *        above, then re-shrink this array. Any field carved from here MUST
     *        be unconditionally initialized in make_transform_graph, or two
     *        structurally-equal graphs would mint distinct NTTP instantiations.
     */
    uint8_t reserved[7] = {};

    // -- align-1: byte arrays --------------------------------------------
    ck_tile::static_array<CoordinateTransform, MAX_TRANSFORMS_V4> transforms{};

    // Per-transform edge routing. Indexed by transform index.
    ck_tile::static_array<DimIds, MAX_TRANSFORMS_V4> t_input_edge_ids{};
    ck_tile::static_array<DimIds, MAX_TRANSFORMS_V4> t_output_edge_ids{};

    // Boundary edge ids.
    DimIds   input_edge_ids{};
    DimIds   output_edge_ids{};

    // Topo order of transforms.
    ck_tile::static_array<uint8_t, MAX_TRANSFORMS_V4> topo_order{};

    constexpr bool operator==(const TransformGraph&) const = default;
};

// NTTP unique-object-representation invariant (see member-ordering note above).
// A clang-22 IRGen bug miscompiles runtime member reads through a `template
// <auto G>` parameter object when the type has implicit padding; keeping the
// layout hole-free sidesteps it. If a future edit reintroduces a padding hole
// (e.g. by reordering members or adding one of mismatched alignment), this
// fires at compile time instead of silently miscompiling a member read.
static_assert(std::has_unique_object_representations_v<TransformGraph>,
              "TransformGraph is used as a template<auto G> NTTP and must have "
              "no implicit padding. Re-pack members by descending alignment to "
              "close any introduced hole.");

static_assert(MAX_TRANSFORMS_V4 <= 16,
              "t_output_derived_mask / t_input_derived_mask use uint16_t; "
              "grow to uint32_t and bump this static_assert if "
              "MAX_TRANSFORMS_V4 ever exceeds 16.");



// =============================================================================
// 12. Phase A construction (insertTransform / topoSortKahn
//     + insertNode dispatch + make_transform_graph entry)
// =============================================================================
//
// Phase A scope:
//   1. insertTransform copies a TransformNode's args into the pool, copies
//      its read_ids / write_ids into per-transform edge routing, and copies
//      its input_anchors / output_anchors into per-transform graph anchor
//      storage. Pool slot count comes directly from node.args_count.
//   2. topoSortKahn produces graph.topo_order[] via Kahn's algorithm,
//      keyed by which transform writes which edge.
//   3. insertNode dispatches TransformNode / OutputNode / InputNode /
//      SubgraphNode wrappers into the graph; make_transform_graph is the
//      user entry.

namespace detail {

// MONOTONIC-POOL-USED INVARIANT (TRIPWIRE FOR PHASE 4 AUTHORS):
//
//   insertTransform MUST write its slice at [graph.pool_used, graph.pool_used + arg_count)
//   and advance graph.pool_used by exactly arg_count. spliceInto MUST do the equivalent
//   via nextFreePoolOffset(outer) (which returns outer.pool_used) and
//   advance by inner.pool_used. No pool writer may take a target offset
//   below the current pool_used or skip slots.
//
// This invariant makes pool-slot ownership conflict structurally
// impossible -- a previous transform's slice ends strictly below the
// current pool_used, so its slots cannot collide with the about-to-be-
// written slice. The runtime per-slot bitmap that previously enforced
// this was dropped after 3-expert review (architect/cpp_expert/realist
// agreed it was dead code with the invariant intact).
//
// If you change pool placement to take a target offset directly (for
// example a Phase 4 enhancement that re-arranges slots), re-add the
// per-slot conflict check AT THAT WRITER and call poolAliasingViolation()
// on conflict so the failure surfaces in the efail driver.

/**
 * @brief Insert one transform into the graph: pool memcpy, per-transform
 *        record, per-transform edge routing, then bump counters.
 *
 * The slot bytes were already converted via `Slot::from_raw` inside the
 * factory (e.g. `make_embed`), so this function does not run any conversion
 * machinery -- just a flat copy of `node.args[0 .. node.args_count)` into
 * `graph.pool[graph.pool_used .. graph.pool_used + node.args_count)` and
 * a small bundle of metadata writes.
 *
 * See MONOTONIC-POOL-USED INVARIANT note above for why the write head
 * is always `graph.pool_used` (never a stale offset).
 */
template <typename Precision>
constexpr void
insertTransform(TransformGraph& graph, const TransformNode& node) noexcept
{
    const uint8_t transform_idx = graph.num_transforms;

    // Width induction: re-encode each authored VALUE slot to the graph's
    // precision as it enters the pool (non-VALUE slots pass through). For the
    // default precision this is the identity -- same vtype, same payload bytes.
    for(uint8_t k = 0; k < node.args_count; ++k) {
        graph.pool[graph.pool_used + k] = adjust_precision<Precision>(node.args[k]);
    }

    graph.transforms[transform_idx] = CoordinateTransform{
        node.transform_type,
        graph.pool_used,
        node.args_count,
        node.ndim_input,
        node.ndim_output
    };

    graph.t_input_edge_ids[transform_idx]  = node.read_ids;
    graph.t_output_edge_ids[transform_idx] = node.write_ids;

    // Forward factory-authored anchors into the per-edge anchor arrays, keyed
    // by the transform's edge ids, inducing each VALUE anchor (inline length) to
    // the graph precision in the same write (edge_id / derived slots pass
    // through). Only NON-UNUSED anchors are stored: an UNUSED input anchor means
    // "this transform doesn't define this edge's length" and must NOT clobber a
    // producer's (or another reader's) real anchor at the same edge id. A second
    // non-UNUSED write to the same edge that DISAGREES with the first is a
    // multiply-written conflict (fires the diagnostic at consteval); an
    // identical write (fan-out readers agreeing on a length) is idempotent.
    for(uint8_t k = 0; k < node.ndim_input; ++k) {
        if(node.input_anchors[k].kind() != Kind::UNUSED) {
            const Slot    a  = adjust_precision<Precision>(node.input_anchors[k]);
            const uint8_t id = node.read_ids[k];
            if(graph.t_input_edge_anchors[id].kind() != Kind::UNUSED &&
               !(graph.t_input_edge_anchors[id] == a)) {
                graphValidationErrorEdgeMultiplyWritten();
            }
            graph.t_input_edge_anchors[id] = a;
        }
    }
    for(uint8_t k = 0; k < node.ndim_output; ++k) {
        if(node.output_anchors[k].kind() != Kind::UNUSED) {
            const Slot    a  = adjust_precision<Precision>(node.output_anchors[k]);
            const uint8_t id = node.write_ids[k];
            if(graph.t_output_edge_anchors[id].kind() != Kind::UNUSED &&
               !(graph.t_output_edge_anchors[id] == a)) {
                graphValidationErrorEdgeMultiplyWritten();
            }
            graph.t_output_edge_anchors[id] = a;
        }
    }

    // Mirror Kind::DERIVED presence into the per-transform bitmask. The
    // factory already set the bools when it authored the anchors, so no
    // scan is needed here. Reading (mask >> transform_idx) & 1u in
    // resolveOneTransformEdges is cheaper (one shift+and, constexpr-folds
    // to a literal at the NTTP-bound call site) than the consteval lambdas
    // this replaces.
    const uint16_t bit = static_cast<uint16_t>(uint16_t{1} << transform_idx);
    if(node.has_output_derived) {
        graph.t_output_derived_mask =
            static_cast<uint16_t>(graph.t_output_derived_mask | bit);
    }
    if(node.has_input_derived) {
        graph.t_input_derived_mask =
            static_cast<uint16_t>(graph.t_input_derived_mask | bit);
    }

    graph.pool_used      = static_cast<uint8_t>(graph.pool_used + node.args_count);
    graph.num_transforms = static_cast<uint8_t>(transform_idx + 1);
}

// ---- Topo sort via Kahn's algorithm ----
//
// For each transform, in-degree = count of input edges that are written by
// other (still-unprocessed) transforms. We build an edge -> producing-
// transform map by scanning t_output_edge_ids. Repeatedly pull a transform
// with in-degree 0, add it to topo_order, decrement its consumers'
// in-degrees, repeat.
//
// Cycle detection: if Kahn's terminates with fewer transforms processed
// than graph.num_transforms, fire graphValidationErrorTransformCycle().

constexpr void
topoSortKahn(TransformGraph& graph) noexcept
{
    // Step 1: build edge -> producing-transform-index map.
    // Sentinel 0xFF means "no transform writes this edge" (graph boundary
    // input). MAX_EDGES_V4 entries.
    ck_tile::static_array<uint8_t, MAX_EDGES_V4> edge_writer{};
    for(ck_tile::index_t i = 0; i < MAX_EDGES_V4; ++i) { edge_writer[i] = 0xFF; }

    for(uint8_t transform_idx = 0; transform_idx < graph.num_transforms; ++transform_idx) {
        const auto& transform = graph.transforms[transform_idx];
        for(uint8_t i = 0; i < transform.ndim_output; ++i) {
            const uint8_t edge_id = graph.t_output_edge_ids[transform_idx][i];
            edge_writer[edge_id] = transform_idx;
        }
    }

    // Step 2: in-degree per transform = count of input edges that have a
    // writing transform (i.e., are not graph boundary inputs).
    ck_tile::static_array<uint8_t, MAX_TRANSFORMS_V4> in_degree{};
    for(uint8_t transform_idx = 0; transform_idx < graph.num_transforms; ++transform_idx) {
        const auto& transform = graph.transforms[transform_idx];
        uint8_t deg = 0;
        for(uint8_t i = 0; i < transform.ndim_input; ++i) {
            const uint8_t edge_id = graph.t_input_edge_ids[transform_idx][i];
            if(edge_writer[edge_id] != 0xFF) { ++deg; }
        }
        in_degree[transform_idx] = deg;
    }

    // Step 3: Kahn's main loop. Pull zero-in-degree transforms in source
    // order (gives stable, deterministic ordering when multiple are ready).
    uint8_t emitted = 0;
    while(emitted < graph.num_transforms) {
        bool made_progress = false;
        for(uint8_t transform_idx = 0; transform_idx < graph.num_transforms; ++transform_idx) {
            if(in_degree[transform_idx] == 0) {
                graph.topo_order[emitted++] = transform_idx;
                in_degree[transform_idx]    = 0xFF;   // mark consumed

                // Decrement consumers (transforms whose inputs include any
                // edge that transform_idx writes).
                const auto& transform = graph.transforms[transform_idx];
                for(uint8_t i = 0; i < transform.ndim_output; ++i) {
                    const uint8_t edge_id = graph.t_output_edge_ids[transform_idx][i];
                    for(uint8_t consumer_idx = 0; consumer_idx < graph.num_transforms; ++consumer_idx) {
                        if(in_degree[consumer_idx] == 0xFF) { continue; }
                        const auto& consumer = graph.transforms[consumer_idx];
                        for(uint8_t j = 0; j < consumer.ndim_input; ++j) {
                            if(graph.t_input_edge_ids[consumer_idx][j] == edge_id) {
                                if(in_degree[consumer_idx] > 0) { --in_degree[consumer_idx]; }
                                break;
                            }
                        }
                    }
                }
                made_progress = true;
                break;   // restart scan; preserves source-order stability
            }
        }
        if(!made_progress) {
            graphValidationErrorTransformCycle();
            break;
        }
    }
}

} // namespace detail

// =============================================================================
// 12a. User-facing builder API
// =============================================================================
//
// TODO(perf): this section adds Read<N>, Write<N>, OutputNode<SlotT>,
// InputNode<SlotT> templates plus overload pairs for outputs/inputs (with
// and without dims). Per-call instantiation count is ~8 distinct types
// for the anchored form. The Read<N>/Write<N> arity templating buys
// compile-time arity-match static_asserts at the cost of one new type
// per distinct N used in the program. Worth measuring with -ftime-trace
// once a real workload exists; consider de-templating Read/Write back to
// runtime-counted if the static_assert isn't pulling enough weight.
// Tracked under task #109 (V4 Batch 1.7).
//
// Top-level construction (V3-style API):
//
//   auto g = make_transform_graph(
//       outputs(read(0)),
//       transform(make_offset(8),  read(1), write(0)),
//       transform(make_embed(...), read(2, 3), write(1)),
//       inputs(dims(M, N), write(2, 3)));
//
// outputs() takes a read(...) declaration: which graph-internal edges to
// surface as graph outputs. Optional dims(...) prepends, anchoring the
// boundary edge lengths externally.
//
// inputs() takes a write(...) declaration: which graph-internal edges
// receive caller-supplied values. Optional dims(...) prepends, anchoring
// boundary edge lengths.
//
// Reads naturally as "outputs of shape dims(...) read from these slots"
// and "inputs of shape dims(...) written into these slots."
//
// make_transform_graph accepts outputs(...), inputs(...), and transform(...)
// declarations in any order; dispatches by type.

// ---- Read / Write -- per-transform edge id lists ----
//
// Templated on N for compile-time arity-match static_assertions when used
// in conjunction with optional dims(...) anchors.

/// @brief List of N edge ids feeding a transform's inputs (passed to transform()).
template <uint8_t N>
struct Read
{
    static constexpr uint8_t COUNT = N;
    DimIds ids{};
};

/// @brief List of N edge ids the transform writes into (passed to transform()).
template <uint8_t N>
struct Write
{
    static constexpr uint8_t COUNT = N;
    DimIds ids{};
};

/// @brief Per-transform input edge-id list. `consteval` enforces
///        graph-construction-only use.
template <typename... Ids>
consteval auto read(Ids... edge_ids) noexcept
{
    Read<sizeof...(Ids)> r{};
    uint8_t i = 0;
    ((r.ids[i++] = static_cast<uint8_t>(edge_ids)), ...);
    return r;
}

/// @brief Per-transform output edge-id list. `consteval` enforces
///        graph-construction-only use.
template <typename... Ids>
consteval auto write(Ids... edge_ids) noexcept
{
    Write<sizeof...(Ids)> w{};
    uint8_t i = 0;
    ((w.ids[i++] = static_cast<uint8_t>(edge_ids)), ...);
    return w;
}

// ---- InputNode / OutputNode -- boundary edge declarations ----

/**
 * @brief Boundary input declaration: which edges receive caller-supplied
 *        values, plus optional inline length anchors.
 *
 * Concrete POD; anchors are pre-committed to Slot bytes at `inputs(...)`
 * factory time via the same i++ pattern used by dims/strides. The framework
 * (insertNode -> OutputNode/InputNode branch) just copies `anchors[k]` into
 * `graph.input_edge_anchors / output_edge_anchors[edge_ids[k]]` -- no tuple unpacking required.
 */
struct InputNode
{
    DimIds                                          edge_ids{};
    uint8_t                                         count       = 0;
    bool                                            has_anchors = false;
    ck_tile::static_array<Slot, MAX_TENSOR_DIMS_V4> anchors{};
};

/// @brief Boundary output declaration. Identical shape to InputNode; the
///        distinct type drives `insertNode`'s static dispatch.
struct OutputNode
{
    DimIds                                          edge_ids{};
    uint8_t                                         count       = 0;
    bool                                            has_anchors = false;
    ck_tile::static_array<Slot, MAX_TENSOR_DIMS_V4> anchors{};
};

/// @brief Boundary outputs without inline length anchors. Lengths come
///        from upstream transforms (positional propagation) or from the
///        downstream consumer's external anchor.
template <uint8_t N>
consteval OutputNode outputs(Read<N> r) noexcept
{
    return OutputNode{.edge_ids = r.ids, .count = N};
}

/// @brief Boundary outputs with inline length anchors. Reads as
///        "outputs of shape dims(...) read from these slots".
/// @pre  `dims(...) arity == read(...) arity`.
template <uint8_t DimsCount, uint8_t N>
consteval OutputNode outputs(DimsList<DimsCount> d, Read<N> r) noexcept
{
    static_assert(DimsCount == N,
                  "outputs(dims(...), read(...)): dims and read must have the same arity");
    OutputNode node{.edge_ids = r.ids, .count = N, .has_anchors = true};
    for(uint8_t k = 0; k < N; ++k) { node.anchors[k] = d[k]; }
    return node;
}

/// @brief Boundary inputs without inline length anchors. The user must
///        anchor boundary input lengths somewhere (typically via dims).
template <uint8_t N>
consteval InputNode inputs(Write<N> w) noexcept
{
    return InputNode{.edge_ids = w.ids, .count = N};
}

/// @brief Boundary inputs with inline length anchors. Reads as
///        "inputs of shape dims(...) written into these slots".
/// @pre  `dims(...) arity == write(...) arity`.
template <uint8_t DimsCount, uint8_t N>
consteval InputNode inputs(DimsList<DimsCount> d, Write<N> w) noexcept
{
    static_assert(DimsCount == N,
                  "inputs(dims(...), write(...)): dims and write must have the same arity");
    InputNode node{.edge_ids = w.ids, .count = N, .has_anchors = true};
    for(uint8_t k = 0; k < N; ++k) { node.anchors[k] = d[k]; }
    return node;
}

// ---- make_subgraph -- splice an inner graph into the outer ----

/**
 * @brief A sub-graph splice declaration: an inner-graph value plus the
 *        outer-graph edge routing for its boundary inputs/outputs.
 *
 * The inner graph is held BY VALUE because the consteval factory must
 * accept its argument as a true constant expression. A reference would
 * dangle once the consteval frame returns.
 *
 * Cumulative constexpr-stack cost (per V3's experience): each splice
 * level adds ~sizeof(TransformGraph) bytes to the constexpr-eval stack.
 * At realistic 3-level nests, peak is ~100 KB -- well within clang's
 * default `-fconstexpr-steps=1048576` budget.
 *
 * Read/write counts are stored as runtime `uint8_t` since the static N
 * from `Read<N>`/`Write<N>` is forgotten after `make_subgraph` returns
 * (same idiom as TransformNode).
 */
struct SubgraphNode
{
    TransformGraph inner;
    DimIds         read_ids;
    uint8_t        read_count;
    DimIds         write_ids;
    uint8_t        write_count;
};

/**
 * @brief Splice an inner graph into the outer graph: the inner's boundary
 *        inputs feed from outer's `read(...)` edges; its boundary outputs
 *        drive outer's `write(...)` edges. `insertNode` dispatches the
 *        returned `SubgraphNode` to `spliceInto`, which inlines the inner
 *        pool / transforms / edge routing into the outer at consteval time.
 *
 * @pre  `read(...) arity == g_inner.input_edge_ids.count()`  (else
 *       `transformErrorReadCountArityMismatch()` fires at consteval).
 * @pre  `write(...) arity == g_inner.output_edge_ids.count()` (else
 *       `transformErrorWriteCountArityMismatch()` fires at consteval).
 */
template <uint8_t ReadCount, uint8_t WriteCount>
consteval SubgraphNode
make_subgraph(TransformGraph g_inner, Read<ReadCount> r, Write<WriteCount> w) noexcept
{
    if(ReadCount != g_inner.input_edge_ids.count())  { transformErrorReadCountArityMismatch(); }
    if(WriteCount != g_inner.output_edge_ids.count()) { transformErrorWriteCountArityMismatch(); }
    return SubgraphNode{.inner       = g_inner,
                        .read_ids    = r.ids,
                        .read_count  = ReadCount,
                        .write_ids   = w.ids,
                        .write_count = WriteCount};
}


// ---- make_transform_graph -- top-level builder ----

namespace detail {

// Type traits to dispatch arguments by category in make_transform_graph.
// All four node types are now concrete (non-templated), so each trait
// reduces to a single full specialization.

template <typename T> struct is_output_node                : std::false_type {};
template <>           struct is_output_node<OutputNode>    : std::true_type  {};

template <typename T> struct is_input_node                 : std::false_type {};
template <>           struct is_input_node<InputNode>      : std::true_type  {};

template <typename T> struct is_transform_node             : std::false_type {};
template <>           struct is_transform_node<TransformNode> : std::true_type {};

template <typename T> struct is_subgraph_node              : std::false_type {};
template <>           struct is_subgraph_node<SubgraphNode>   : std::true_type {};

// Apply one argument to the in-progress graph based on its type.
//
//   TransformNode : memcpy node.args into graph.pool + record routing.
//                   ndim_input / ndim_output are already pinned by the
//                   make_X factory (no late-binding patch required).
//   OutputNode    : copy edge_ids into graph.output_edge_ids; if has_anchors,
//                   write each pre-committed Slot into graph.input_edge_anchors / output_edge_anchors[].
//   InputNode     : same as OutputNode for the input boundary.
//   SubgraphNode  : delegate to spliceInto() to inline the inner graph at
//                   construction time (V3-style splice).
//
/**
 * @brief Record an OutputNode's boundary edge ids + optional anchors into
 *        the graph. Symmetric with `insertInput` for the input boundary.
 */
template <typename Precision>
constexpr void
insertOutput(TransformGraph& graph, const OutputNode& node) noexcept
{
    graph.output_edge_ids = node.edge_ids;
    if(node.has_anchors) {
        for(uint8_t k = 0; k < node.count; ++k) {
            graph.output_edge_anchors[k] = adjust_precision<Precision>(node.anchors[k]);
        }
    }
}

/**
 * @brief Record an InputNode's boundary edge ids + optional anchors into
 *        the graph. Symmetric with `insertOutput` for the output boundary.
 */
template <typename Precision>
constexpr void
insertInput(TransformGraph& graph, const InputNode& node) noexcept
{
    graph.input_edge_ids = node.edge_ids;
    if(node.has_anchors) {
        for(uint8_t k = 0; k < node.count; ++k) {
            graph.input_edge_anchors[k] = adjust_precision<Precision>(node.anchors[k]);
        }
    }
}

// `insertSubgraph(graph, node)` is defined further down (section 11.b);
// it inlines the inner graph's pool + transforms into the outer with
// edge-id remap and pool-offset shift.

// Branch order: hot path first. Most graphs have N TransformNode args,
// exactly 1 OutputNode and 1 InputNode, and 0-or-few SubgraphNodes -- so
// TransformNode goes first, then the boundary args, SubgraphNode last.
template <typename Precision, typename NodeT>
constexpr void
insertNode(TransformGraph& graph, NodeT node) noexcept
{
    if constexpr(is_transform_node<NodeT>::value) {
        insertTransform<Precision>(graph, node);
    }
    else if constexpr(is_output_node<NodeT>::value) {
        insertOutput<Precision>(graph, node);
    }
    else if constexpr(is_input_node<NodeT>::value) {
        insertInput<Precision>(graph, node);
    }
    else if constexpr(is_subgraph_node<NodeT>::value) {
        spliceInto(graph, node);
    }
    else {
        static_assert(is_transform_node<NodeT>::value ||
                      is_output_node<NodeT>::value    ||
                      is_input_node<NodeT>::value     ||
                      is_subgraph_node<NodeT>::value,
                      "make_transform_graph argument must be outputs(...), "
                      "inputs(...), make_X(args, read, write), or "
                      "make_subgraph(g_inner, read, write).");
    }
}

// Compute the active edge count of a fully-built graph: max edge id used
// across boundary slots + per-transform routing, plus 1. Edge ids are
// dense from 0; the result is the smallest valid array length to hold
// all edge-keyed data (EdgeLengths::values, etc.).
constexpr uint8_t
computeNumEdges(const TransformGraph& graph) noexcept
{
    uint8_t max_id = 0;
    auto bump = [&](uint8_t id) {
        if(id != INVALID_EDGE_ID && static_cast<uint8_t>(id + 1) > max_id) {
            max_id = static_cast<uint8_t>(id + 1);
        }
    };
    // Graph boundary slots: iterate up to actual count, not MAX_TENSOR_DIMS_V4.
    const uint8_t n_in_boundary  = graph.input_edge_ids.count();
    const uint8_t n_out_boundary = graph.output_edge_ids.count();
    for(uint8_t i = 0; i < n_in_boundary;  ++i) { bump(graph.input_edge_ids[i]); }
    for(uint8_t i = 0; i < n_out_boundary; ++i) { bump(graph.output_edge_ids[i]); }
    // Per-transform input/output routing (sized to ndim_input/output).
    for(uint8_t transform_idx = 0; transform_idx < graph.num_transforms; ++transform_idx) {
        const uint8_t nin  = graph.transforms[transform_idx].ndim_input;
        const uint8_t nout = graph.transforms[transform_idx].ndim_output;
        for(uint8_t i = 0; i < nin; ++i)  { bump(graph.t_input_edge_ids[transform_idx][i]); }
        for(uint8_t i = 0; i < nout; ++i) { bump(graph.t_output_edge_ids[transform_idx][i]); }
    }
    return max_id;
}

// Compute the active binding count: scan the pool for Value entries with
// kind == BINDING (per-Impl Schema field bindings) and the boundary
// input_edge_anchors[] / output_edge_anchors[] (boundary anchor
// placeholders), return max id + 1.
constexpr uint8_t
computeNumBindings(const TransformGraph& graph) noexcept
{
    uint8_t max_id = 0;
    auto bump = [&](uint8_t id) {
        if(static_cast<uint8_t>(id + 1) > max_id) {
            max_id = static_cast<uint8_t>(id + 1);
        }
    };
    // Pool entries (per-Impl Schema bindings).
    for(uint8_t i = 0; i < graph.pool_used; ++i) {
        if(graph.pool[i].is_binding_id()) {
            bump(graph.pool[i].as_binding_id());
        }
    }
    // Boundary edge anchors (placeholders attached to inputs/outputs), indexed
    // by boundary position.
    for(ck_tile::index_t i = 0; i < MAX_TENSOR_DIMS_V4; ++i) {
        if(graph.input_edge_anchors[i].is_binding_id()) {
            bump(graph.input_edge_anchors[i].as_binding_id());
        }
        if(graph.output_edge_anchors[i].is_binding_id()) {
            bump(graph.output_edge_anchors[i].as_binding_id());
        }
    }
    return max_id;
}

} // namespace detail

/**
 * @brief Top-level graph constructor.
 *
 * Accepts outputs(...), inputs(...), and transform(...) declarations in any
 * order. Exactly one outputs(...) and one inputs(...) declaration are
 * expected; any number of transform(...) declarations may appear.
 *
 * @tparam Precision Numeric precision policy (Precision32 default). Stored on
 *         the graph as a tag; drives length/stride/float width once the width
 *         normalization lands at the write boundaries.
 */
template <PrecisionPolicy Precision = Precision32, typename... Nodes>
consteval auto make_transform_graph(Nodes... nodes) noexcept
{
    TransformGraph graph{};
    graph.precision = precision_tag_of_v<Precision>;

    // Dispatch each node by type. Order doesn't matter.
    // (OutputNode/InputNode write boundary anchors into input_edge_anchors / output_edge_anchors[];
    //  TransformNodes call insertTransform which fills the pool slice,
    //  per-transform routing, and per-transform anchor arrays directly
    //  from the factory's TransformNode.)
    (detail::insertNode<Precision>(graph, nodes), ...);

    // Topo sort: produces graph.topo_order[] via Kahn's algorithm.
    detail::topoSortKahn(graph);

    // Populate active counts: drives per-graph sizing of EdgeLengths and
    // RuntimeBindings instead of the MAX_EDGES_V4 / MAX_BINDINGS_V4
    // worst-case constants.
    graph.num_edges    = detail::computeNumEdges(graph);
    graph.num_bindings = detail::computeNumBindings(graph);

    // TODO(Batch 3c): the schemaEdgeLengthConflict sanity check fires at
    // end of Phase A.5 (make_graph_bindings), comparing Schema-declared anchor
    // values to the resolved edge_lengths. FROM_ARRAY/FROM_FIELD anchor
    // population also lands in Batch 3c (needs Schema construction).

    return graph;
}


// =============================================================================
// 12b. Phase A.5 -- edge length resolution (in make_graph_bindings)
// =============================================================================
//
// Resolves graph.edge_lengths from:
//   - Boundary anchors (graph.input_edge_anchors / output_edge_anchors[] populated by inputs/outputs)
//   - Per-edge transform anchors (graph.t_input_edge_anchors /
//     t_output_edge_anchors, authored by each factory and written by
//     insertTransform at the transform's edge ids)
//
// Walked in g.topo_order so upstream output lengths are resolved before
// downstream input lengths read them.
//
// Phase ordering per transform:
//   1. Resolve non-DERIVED anchors per-edge. Kind switch reads VALUE
//      payloads directly, BINDING_ID indirects through RuntimeBindings,
//      EDGE_ID copies from edges[id], UNUSED is a skip.
//   2. Resolve DERIVED anchors via Impl::deriveOutputLength /
//      deriveInputLength. These read the just-resolved other-side lengths
//      through the constructed Impl::Schema.

namespace detail {

// True iff the slot's kind is one of VALUE / BINDING_ID / EDGE_ID -- i.e.
// `eval` would return a resolved value without firing a trap stub. Callers
// pre-filter on this where DERIVED/UNUSED is a legitimate "skip" rather
// than a bug.
constexpr bool is_resolvable(Kind k) noexcept
{
    return k == Kind::VALUE || k == Kind::BINDING_ID || k == Kind::EDGE_ID;
}

// Single Kind-dispatch for typed slot evaluation. Free function (not a
// member of Slot) so Slot remains a pure value container without
// dependencies on RuntimeBindingsView / EdgeLengthsView. Traps via
// extern-undefined diagnostic stubs on DERIVED / UNUSED -- reading from
// those kinds is a programming error; callers must pre-filter via
// `is_resolvable(slot.kind())` when silent-skip is the intended policy.
template <typename T, typename LengthT>
CK_TILE_HOST_DEVICE constexpr T
eval(Slot                s,
     RuntimeBindingsView bindings,
     EdgeLengthsView<LengthT> edges) noexcept
{
    static_assert(std::is_trivially_copyable_v<T>,
                  "detail::eval<T>: T must be trivially copyable");
    static_assert(sizeof(T) <= sizeof(Slot::PayloadT),
                  "detail::eval<T>: T must fit within payload width");
    switch(s.kind())
    {
    case Kind::VALUE:      return s.template as_value<T>();
    case Kind::BINDING_ID: return bindings[s.as_binding_id()].template as_value<T>();
    case Kind::EDGE_ID:    return static_cast<T>(edges[s.as_edge_id()]);
    case Kind::DERIVED:    valueReadFromDerivedMarker(); return T{};
    case Kind::UNUSED:     valueReadFromUnusedSlot();    return T{};
    }
    return T{};
}

/**
 * @brief Phase A.5 per-transform: resolve all edges this transform anchors.
 *
 * Two passes:
 *   1. Non-DERIVED anchors. Dispatches on `Kind` per edge: VALUE writes the
 *      literal payload, BINDING_ID indirects through RuntimeBindings,
 *      EDGE_ID copies the resolved length of another edge, UNUSED skips
 *      (the edge is filled by whichever transform produces it).
 *   2. DERIVED anchors only, gated by `if constexpr` on the per-Impl trait.
 *      Constructs the per-Impl Schema lazily and calls the Impl's
 *      derive method to compute the length from already-resolved inputs.
 *
 * Anchors are read per position from the per-edge arrays via the transform's
 * routing: `G.t_input_edge_anchors[G.t_input_edge_ids[transform_idx][i]]`
 * (and the output counterpart).
 */
// Resolve a contiguous range of anchor->edge entries into edge_lengths.
// Free function (not a lambda inside resolveOneTransformEdges) so it
// instantiates ONCE per TU rather than once per outer (G, X, TransformIdx)
// specialization of resolveOneTransformEdges. Delegates the Kind dispatch
// to `detail::eval` (single source of truth) after pre-filtering the cases
// that Pass 1 must silently skip (DERIVED is handled in Pass 2; UNUSED is
// filled by an upstream transform).
template <typename LengthT>
inline constexpr void
resolveAnchorEdges(RuntimeBindingsView bindings,
                   EdgeLengthsView<LengthT>     edge_lengths,
                   const ck_tile::static_array<Slot, MAX_EDGES_V4>& anchors_by_edge,
                   const DimIds&       edge_ids,
                   uint8_t             count) noexcept
{
    for(uint8_t i = 0; i < count; ++i) {
        const uint8_t edge_id = edge_ids[i];
        const Slot    a       = anchors_by_edge[edge_id];
        if(is_resolvable(a.kind())) {
            edge_lengths[edge_id] = eval<LengthT>(a, bindings, edge_lengths);
        }
    }
}

template <auto G, TransformType X>
constexpr void
resolveOneTransformEdges(RuntimeBindingsView bindings,
                         EdgeLengthsView<typename precision_t<G.precision>::LengthT>     edge_lengths,
                         uint8_t             transform_idx) noexcept
{
    using Impl = TransformImpl<X, precision_t<G.precision>>;
    const auto& transform = G.transforms[transform_idx];

    // ---- Pass 1: non-DERIVED anchors --------------------------------------

    using LengthT = typename precision_t<G.precision>::LengthT;
    resolveAnchorEdges<LengthT>(bindings, edge_lengths,
                                G.t_input_edge_anchors,
                                G.t_input_edge_ids[transform_idx],
                                transform.ndim_input);
    resolveAnchorEdges<LengthT>(bindings, edge_lengths,
                                G.t_output_edge_anchors,
                                G.t_output_edge_ids[transform_idx],
                                transform.ndim_output);

    // ---- Pass 2: DERIVED anchors ------------------------------------------
    //
    // transform_idx is a runtime arg (passed from the topo_order fold), so
    // the derived-mask bits and the if-block below are runtime checks rather
    // than if constexpr. The per-position static_assert that lived here is
    // moved to the factories (make_pad/embed/merge/unmerge): each one knows
    // at construction time that it authored a DERIVED anchor and asserts
    // ImplHasDeriveOutput/InputLength<TransformImpl<X>> directly.
    //
    // At -O3 with NTTP G + the constexpr-literal transform_idx values
    // produced by the fold dispatcher, clang should constant-fold the
    // bitmask reads and elide the runtime branch entirely for Impls that
    // don't author DERIVED anchors.
    // Outer `if constexpr` gates the entire DERIVED-handling block on
    // whether Impl actually defines deriveOutput/InputLength. Impls that
    // don't (XOR, OFFSET, FREEZE, SLICE, BROADCAST) skip the block at
    // compile time and never instantiate the deriveX call. Inside the
    // gated block, runtime checks on the precomputed bitmask skip the
    // per-position work for transform positions that don't actually
    // author DERIVED anchors.
    if constexpr(ImplHasDeriveOutputLength<Impl> || ImplHasDeriveInputLength<Impl>) {
        const bool output_has_derived = (G.t_output_derived_mask >> transform_idx) & 1u;
        const bool input_has_derived  = (G.t_input_derived_mask  >> transform_idx) & 1u;

        if(output_has_derived || input_has_derived) {
            auto schema = make_schema<Impl>(G.pool, bindings, edge_lengths,
                                            transform.base_offset);
            if constexpr(ImplHasDeriveOutputLength<Impl>) {
                if(output_has_derived) {
                    for(uint8_t i = 0; i < transform.ndim_output; ++i) {
                        const uint8_t edge_id = G.t_output_edge_ids[transform_idx][i];
                        if(G.t_output_edge_anchors[edge_id].kind() == Kind::DERIVED) {
                            edge_lengths[edge_id] =
                                Impl::deriveOutputLength(schema, edge_lengths,
                                                         G.t_input_edge_ids[transform_idx],
                                                         i);
                        }
                    }
                }
            }
            if constexpr(ImplHasDeriveInputLength<Impl>) {
                if(input_has_derived) {
                    for(uint8_t i = 0; i < transform.ndim_input; ++i) {
                        const uint8_t edge_id = G.t_input_edge_ids[transform_idx][i];
                        if(G.t_input_edge_anchors[edge_id].kind() == Kind::DERIVED) {
                            edge_lengths[edge_id] =
                                Impl::deriveInputLength(schema, edge_lengths,
                                                        G.t_output_edge_ids[transform_idx],
                                                        i);
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Dispatch resolveOneTransformEdges over G.topo_order[].
 *
 * Each Is becomes a constexpr literal -- the transform's TransformType is
 * NTTP-bound at each call site, but the transform index is passed as a
 * runtime argument. Specialization count is therefore (#distinct
 * TransformType values used in G) instead of (#transforms in G), which
 * usually folds 6:1 or 8:1 on typical workloads.
 */
template <auto G, ck_tile::index_t... Is>
constexpr void
resolveEdgeLengthsFold(RuntimeBindingsView bindings,
                       EdgeLengthsView<typename precision_t<G.precision>::LengthT>     edge_view,
                       ck_tile::sequence<Is...>) noexcept
{
    ((resolveOneTransformEdges<G,
                               G.transforms[G.topo_order[Is]].type>(
          bindings, edge_view, G.topo_order[Is])), ...);
}

/// @brief NTTP fold for the Phase A.5 input-boundary anchor evaluation.
///        Each Is becomes a literal index into G.input_edge_ids[], so SROA
///        can fully promote edge_view (no dynamic GEPs).
template <auto G, ck_tile::index_t... Is>
constexpr void
evalInputBoundaryFold(RuntimeBindingsView bindings,
                      EdgeLengthsView<typename precision_t<G.precision>::LengthT>     edge_view,
                      ck_tile::sequence<Is...>) noexcept
{
    // Input boundary anchors come from user RuntimeBindings — should never be
    // DERIVED or UNUSED. If one ever is, `eval` traps via its diagnostic stub
    // (correct behavior — it's a programming error). The anchor for input
    // boundary position Is lives at input_edge_anchors[Is] (pairs with
    // input_edge_ids[Is]). Evaluated at the graph's LengthT (not the 8-byte
    // PayloadT) so a P32 TU never instantiates the 64-bit as_value/set_value
    // family here -- matches the per-transform resolveAnchorEdges path.
    using LengthT = typename precision_t<G.precision>::LengthT;
    ((edge_view[G.input_edge_ids[Is]] =
         eval<LengthT>(G.input_edge_anchors[Is], bindings, edge_view)), ...);
}

/// @brief NTTP fold for the Phase A.5 output-boundary anchor evaluation.
///        Skips slots whose anchor kind is UNUSED or DERIVED (DERIVED outputs
///        are filled by each Impl's deriveX in Pass 2).
template <auto G, ck_tile::index_t... Is>
constexpr void
evalOutputBoundaryFold(RuntimeBindingsView bindings,
                       EdgeLengthsView<typename precision_t<G.precision>::LengthT>     edge_view,
                       ck_tile::sequence<Is...>) noexcept
{
    // The anchor for output boundary position Is lives at output_edge_anchors[Is]
    // (pairs with output_edge_ids[Is]). Evaluated at the graph's LengthT (not the
    // 8-byte PayloadT) so a P32 TU never names the 64-bit as_value/set_value
    // family -- matches the per-transform resolveAnchorEdges path.
    //
    // Plain fold, no per-Is templated lambda: each Is is already a constant, so
    // `is_resolvable(...)` constant-folds at -O3 (and short-circuits at constexpr)
    // -- the unresolvable (UNUSED / DERIVED) positions are skipped, DERIVED
    // outputs being filled by each Impl's deriveX in Pass 2. The ternary keeps
    // eval off the trap-on-read slots without minting a closure type per output.
    using LengthT = typename precision_t<G.precision>::LengthT;
    ((is_resolvable(G.output_edge_anchors[Is].kind())
          ? (void)(edge_view[G.output_edge_ids[Is]] =
                eval<LengthT>(G.output_edge_anchors[Is], bindings, edge_view))
          : void()),
     ...);
}

/// @brief Phase A.5 entry point: resolve all edge lengths for a graph.
///
/// Templated on `auto G` so the returned EdgeLengths storage is sized to
/// G.num_edges (vs MAX_EDGES_V4) -- shrinks per-graph stack frame.
/// Internally hands a non-templated EdgeLengthsView to the consumer
/// helpers so the framework's read/write surface stays size-agnostic.
template <auto G>
constexpr EdgeLengths<G.num_edges, typename precision_t<G.precision>::LengthT>
resolveEdgeLengths(const RuntimeBindings<G.num_bindings>& bindings) noexcept
{
    EdgeLengths<G.num_edges, typename precision_t<G.precision>::LengthT> edge_lengths{};
    auto edge_view = make_edge_lengths_view(edge_lengths);

    // Bindings are size-erased to a view ONCE here, at the boundary between the
    // per-G producer (which knows num_bindings) and the size-agnostic consumer
    // helpers (eval, the folds), which stay instantiated per-T rather than
    // per-(T, num_bindings).
    auto binding_view = make_runtime_bindings_view(bindings);

    // Step 0: evaluate boundary anchors (graph inputs and outputs) via
    // NTTP folds. Each iteration becomes a literal index into G.input_edge_ids /
    // G.output_edge_ids; SROA can promote edge_view to scalars (no dynamic GEPs
    // -- the runtime 64-iter early-break loops blocked promotion before).
    evalInputBoundaryFold<G>(
        binding_view, edge_view,
        ck_tile::make_index_sequence<G.input_edge_ids.count()>{});
    evalOutputBoundaryFold<G>(
        binding_view, edge_view,
        ck_tile::make_index_sequence<G.output_edge_ids.count()>{});

    // Step 1: per-transform in topo order via NTTP fold. Each instantiation
    // dispatches DIRECTLY on the constexpr `G.transforms[topo_order[Is]].type`
    // -- no runtime 9-way fan-out. Mirrors the
    // pattern already used by materializeStatesFold + mapCoordFold; collapses
    // 9*num_transforms instantiations to num_transforms (or fewer, since
    // same-Type transforms share an instantiation).
    resolveEdgeLengthsFold<G>(binding_view, edge_view,
                              ck_tile::make_index_sequence<G.num_transforms>{});

    return edge_lengths;
}

} // namespace detail


// =============================================================================
// 13. GraphBindings -- per-graph cached state (Phase B materialization)
// =============================================================================

// ---- State tuple type computation (Phase B caching) ----
//
// For each transform I in G.transforms, compute its per-Impl State type via
// decltype on Impl::resolveState<NIn, NOut>(...). Pack the results into a
// heterogeneous ck_tile::tuple. mapCoord reads these cached states without
// re-calling resolveState per invocation.

namespace detail {

// ---- B4: state_type_key -- canonical State type for (X, NIn, NOut, Precision) ----
//
// Keyed on the transform SHAPE, not the graph: all graphs that contain a
// MERGE<3, 1, Precision32> share the SAME State type -- the compiler
// instantiates per (X, NIn, NOut, Precision) cardinality, not per (G, I).
// makeStateTupleType pulls the shape fields from G.transforms[I] at its single
// use site, so there is no per-(G, I) wrapper struct.

template <TransformType X, uint8_t NIn, uint8_t NOut, typename Precision>
struct state_type_key
{
private:
    using Impl     = TransformImpl<X, Precision>;
    using SchemaT  = typename Impl::Schema;
    // Lengths are strictly non-negative -- carry the precision LengthT,
    // matching materializeOneState / executeOneTransformLiteral.
    using LengthT  = typename Precision::LengthT;
    using InLensT  = ck_tile::static_array<LengthT, NIn>;
    using OutLensT = ck_tile::static_array<LengthT, NOut>;

public:
    using type = decltype(Impl::template resolveState<NIn, NOut>(
        std::declval<SchemaT>(),
        std::declval<const InLensT&>(),
        std::declval<const OutLensT&>()));

    // B5: hard byte budget per-transform State. Fires once per
    // (X, NIn, NOut, SlotT) instantiation; covers every State actually used
    // in the test corpus. If this trips, raise MAX_STATE_BYTES and document
    // why per plan sec 5.7 Phase 4 exit criterion #14, OR cut a State member.
    static_assert(sizeof(type) <= MAX_STATE_BYTES,
                  "B5: per-Impl State exceeds MAX_STATE_BYTES (default 128). "
                  "Either raise the constant with documented justification or "
                  "cut a State member.");
};

template <TransformType X, uint8_t NIn, uint8_t NOut, typename Precision>
using state_type_key_t = typename state_type_key<X, NIn, NOut, Precision>::type;

// The per-transform State tuple is keyed directly on the deduplicating
// state_type_key<X, NIn, NOut, Precision> -- two graphs that share a transform
// shape share one State type and one resolveState decltype instantiation. The
// (X, NIn, NOut) fields are pulled from G.transforms[Is] at the one use site
// below (no per-(G, I) wrapper struct).
template <auto G, ck_tile::index_t... Is>
constexpr auto
makeStateTupleType(ck_tile::sequence<Is...>) noexcept
{
    return ck_tile::tuple<
        state_type_key_t<G.transforms[Is].type,
                         G.transforms[Is].ndim_input,
                         G.transforms[Is].ndim_output,
                         precision_t<G.precision>>...>{};
}

template <auto G>
using StateTupleType =
    decltype(makeStateTupleType<G>(ck_tile::make_index_sequence<G.num_transforms>{}));

} // namespace detail


/// @brief Per-graph cached state, NTTP-baked over G.
///
/// A transparent alias for the heterogeneous tuple of per-transform States --
/// NOT a wrapper struct. mapCoord reads cached state by source-order index via
/// `gb.template get<t>()`; no Schema reconstruction or Phase B re-evaluation
/// per invocation.
///
/// Why an alias and not a struct: the only payload is the state tuple, so a
/// one-field wrapper added nothing but a second mention of the ~20 KB `G` NTTP
/// in every `const GraphBindings<G>&` parameter (a class-template
/// specialization mangles its template argument; an alias is transparent, so
/// the parameter mangles as the concrete `tuple<States...>`). The resolved
/// EdgeLengths are intentionally NOT cached here -- they are Phase B
/// intermediate data, consumed only while materializing the states inside
/// make_graph_bindings and never read on the mapCoord hot path; caching them
/// would hold dead per-edge storage for the whole lifetime of the bindings
/// (the dominant slice of the high-arity scratch footprint).
///
/// State tuple is indexed by SOURCE ORDER (matching G.transforms[I]).
/// mapCoord computes source_index = G.topo_order[I_topo] and reads the
/// state at that compile-time index.
template <auto G>
using GraphBindings = detail::StateTupleType<G>;


// ---- State tuple materialization (Phase B) ----

namespace detail {

/**
 * @brief Gather a transform's edge lengths from resolved edge_lengths into a
 *        per-arity array.
 *
 * Shared by the binding path (materializeOneState) and the literal path
 * (executeOneTransformLiteral): one instantiation per (UIntT, N), not a
 * per-call-site closure. Tensor lengths are strictly non-negative, so UIntT
 * is the precision Unsigned type.
 */
template <typename UIntT, uint8_t N, typename EdgeLengthsT>
CK_TILE_HOST_DEVICE constexpr ck_tile::static_array<UIntT, N>
gatherEdgeLengths(const EdgeLengthsT& edges, const DimIds& edge_ids) noexcept
{
    ck_tile::static_array<UIntT, N> lens{};
    for(uint8_t i = 0; i < N; ++i) {
        lens[i] = static_cast<UIntT>(edges[edge_ids[i]]);
    }
    return lens;
}

/**
 * @brief Build and RETURN transform I's State: construct its Schema from the
 *        pool, gather its input/output edge lengths, and call Impl::resolveState.
 *
 * Shared by both state-build paths: the binding path (materializeStatesFold,
 * which packs the returned States into the GraphBindings state tuple) and the
 * literal path (executeOneTransformLiteral, which holds it as a `constexpr`
 * local). `I` is the source-order transform index into G.transforms.
 * `CK_TILE_HOST_DEVICE` because the literal path reaches this on the device
 * mapCoord path; `constexpr` so the literal path can const-evaluate the whole
 * State to IR literals.
 *
 * Per-Impl-sized lens (not MAX_TENSOR_DIMS_V4): NIn / NOut are constexpr here, so
 * SROA can promote both arrays into registers as long as Impl::resolveState does
 * not escape their address (P1 scratch-spill fix).
 */
template <auto G, ck_tile::index_t I>
CK_TILE_HOST_DEVICE constexpr auto
materializeOneState(const EdgeLengths<G.num_edges, typename precision_t<G.precision>::LengthT>&        edges,
                    const RuntimeBindings<G.num_bindings>& bindings) noexcept
{
    constexpr TransformType X    = G.transforms[I].type;
    constexpr uint8_t       NIn  = G.transforms[I].ndim_input;
    constexpr uint8_t       NOut = G.transforms[I].ndim_output;
    constexpr uint8_t       base = G.transforms[I].base_offset;

    using Impl    = TransformImpl<X, precision_t<G.precision>>;
    using LengthT = typename precision_t<G.precision>::LengthT;

    auto schema = make_schema<Impl>(G.pool, make_runtime_bindings_view(bindings),
                                    make_edge_lengths_view(edges), base);
    auto in_lens  = gatherEdgeLengths<LengthT, NIn>(edges, G.t_input_edge_ids[I]);
    auto out_lens = gatherEdgeLengths<LengthT, NOut>(edges, G.t_output_edge_ids[I]);

    return Impl::template resolveState<NIn, NOut>(schema, in_lens, out_lens);
}

/**
 * @brief Build and RETURN the full per-transform State tuple for graph G.
 *
 * Constructs the heterogeneous tuple directly from each transform's State (a
 * pack expansion over materializeOneState), so make_graph_bindings can name it
 * in a designated initializer rather than default-constructing then filling a
 * by-ref tuple.
 */
template <auto G, ck_tile::index_t... Is>
constexpr detail::StateTupleType<G>
materializeStatesFold(const EdgeLengths<G.num_edges, typename precision_t<G.precision>::LengthT>&        edges,
                      const RuntimeBindings<G.num_bindings>& bindings,
                      ck_tile::sequence<Is...>) noexcept
{
    return detail::StateTupleType<G>{materializeOneState<G, Is>(edges, bindings)...};
}

} // namespace detail


/**
 * @brief Build the per-graph cached bindings: resolve edge lengths (Phase A.5)
 *        and materialize per-transform States (Phase B).
 *
 * The single public entry for runtime bindings. Arguments bind placeholders in
 * Id order (arg #K binds `placeholder<K>`); the count must equal
 * G.num_bindings. Called once per kernel launch; the result is consumed
 * read-only by mapCoord, so hold it `const`:
 * @code
 * const auto gb = v4::make_graph_bindings<graph>(arg0, arg1);
 * v4::mapCoord<graph>(out, in, gb);
 * @endcode
 *
 * The resolved EdgeLengths are a function-LOCAL here (not a GraphBindings
 * member): they feed the state materialization and are then discarded, so they
 * never enter the returned bindings object. The return value IS the state tuple
 * (GraphBindings is an alias for it).
 */
template <auto G, typename... Args>
CK_TILE_HOST_DEVICE constexpr GraphBindings<G>
make_graph_bindings(Args... args) noexcept
{
    static_assert(sizeof...(Args) == G.num_bindings,
                  "v4::make_graph_bindings: argument count must equal G.num_bindings");
    RuntimeBindings<G.num_bindings> rb{};
    uint8_t i = 0;
    // Induce each binding to the graph precision as it is stored (width applied
    // at the write boundary; for the default precision this is the identity).
    ((rb[i++] = detail::normalize_binding<precision_t<G.precision>>(args)), ...);

    // Resolve edge lengths once into a local, build the State tuple from them,
    // then let `edges` die at function scope -- only `states` is returned.
    const auto edges = detail::resolveEdgeLengths<G>(rb);
    return detail::materializeStatesFold<G>(
            edges, rb, ck_tile::make_index_sequence<G.num_transforms>{});
}


// =============================================================================
// 14. mapCoord -- public hot-path entry
// =============================================================================
//
// Phase C: walk transforms in topo order, each step gathers input coords
// from a per-edge working buffer, calls Impl::mapCoord, scatters output
// coords back. After all transforms, copy graph-output edges from the
// working buffer to the user's output_coords array.
//
// Per-transform NIn/NOut are compile-time constants (G is NTTP), allowing
// each Impl's mapCoord<NIn, NOut, CoordT> instantiation to be fully
// monomorphic.
//
// State note: per-transform State is cached in the GraphBindings state tuple by
// make_graph_bindings (Phase B); the bindings mapCoord reads it via
// gb.get<t>() rather than re-running Impl::resolveState per invocation.
// The literal overload resolves state inline as constexpr locals (no bindings
// to cache against).

namespace detail {

/// @brief Execute one transform at compile-time index I in topo order.
///        Reads work[] at the transform's t_input_edge_ids, fetches the
///        cached State from gb (the state tuple, indexed by SOURCE order, not
///        topo), calls Impl::mapCoord, writes back to work[] at t_output_edge_ids.
///
/// All inputs (G, I) are NTTPs, so the per-transform TransformType is a
/// constant expression. We dispatch directly via TransformImpl<X> rather
/// than through a runtime TransformType fold -- such a fold would
/// instantiate the lambda for ALL TransformTypes, which crashes when
/// the heterogeneous gb.get<t>() returns one Impl's State and the
/// non-matching Impl's mapCoord rejects the type.
template <auto G, ck_tile::index_t I, typename CoordT>
CK_TILE_HOST_DEVICE constexpr void
executeOneTransform(ck_tile::static_array<CoordT, G.num_edges>& work,
                    const GraphBindings<G>&                     gb) noexcept
{
    constexpr uint8_t       t    = G.topo_order[I];
    constexpr TransformType X    = G.transforms[t].type;
    constexpr uint8_t       nin  = G.transforms[t].ndim_input;
    constexpr uint8_t       nout = G.transforms[t].ndim_output;
    using Impl                   = TransformImpl<X, precision_t<G.precision>>;

    // Per-Impl-sized buffers (not MAX_TENSOR_DIMS_V4): nin/nout are
    // constexpr at this template instantiation. P2 of the V4 scratch-spill
    // fix.
    ck_tile::static_array<CoordT, nin>  in_buf{};
    for(uint8_t i = 0; i < nin; ++i) {
        in_buf[i] = work[G.t_input_edge_ids[t][i]];
    }

    // Fetch cached State (indexed by source order = t). gb IS the state tuple
    // (GraphBindings is an alias for it), so index it directly.
    const auto& state = gb.template get<t>();

    // Per-transform output buffer; scatter into work[] after.
    ck_tile::static_array<CoordT, nout> out_buf{};
    Impl::template mapCoord<nin, nout>(state, out_buf.elems, in_buf.elems);

    for(uint8_t i = 0; i < nout; ++i) {
        work[G.t_output_edge_ids[t][i]] = out_buf[i];
    }
}

/// @brief NTTP fold over G.num_transforms in topo order.
template <auto G, ck_tile::index_t... Is, typename CoordT>
CK_TILE_HOST_DEVICE constexpr void
mapCoordFold(ck_tile::static_array<CoordT, G.num_edges>& work,
             const GraphBindings<G>&                     gb,
             ck_tile::sequence<Is...>) noexcept
{
    (executeOneTransform<G, Is>(work, gb), ...);
}

} // namespace detail

// =============================================================================
// 14. mapCoord -- argument-presence-dispatched overloads (B3)
// =============================================================================
//
// Two overloads partitioned by argument presence (no `requires` clauses):
//   1) mapCoord<G>(out, in)         -- LITERAL graphs (no bindings needed)
//   2) mapCoord<G>(out, in, gb)     -- RuntimeBindings overload (placeholder)
//
// Per V4_REWORK_PLAN sec 1 Public API row: "argument presence dispatches; NO
// dual surface."

namespace detail {

/// @brief Literal-path execute: per-transform schema + state are `constexpr`
///        locals (mirrors V3's `applyOne` no-bindings pattern). The `constexpr`
///        qualifier is load-bearing: it forces compile-time evaluation so the
///        device backend sees State members as IR literals (no runtime
///        re-computation). Without it, removing GraphBindings just shifts the
///        fold cost from Frontend to Backend.
template <auto G, EdgeLengths<G.num_edges, typename precision_t<G.precision>::LengthT> Edges, ck_tile::index_t I, typename CoordT>
CK_TILE_HOST_DEVICE constexpr void
executeOneTransformLiteral(ck_tile::static_array<CoordT, G.num_edges>& work) noexcept
{
    constexpr uint8_t       t    = G.topo_order[I];
    constexpr TransformType X    = G.transforms[t].type;
    constexpr uint8_t       nin  = G.transforms[t].ndim_input;
    constexpr uint8_t       nout = G.transforms[t].ndim_output;
    using Impl = TransformImpl<X, precision_t<G.precision>>;

    // Both constexpr below: literal-graph edge lengths and per-transform state
    // are deterministic from G alone. Backend sees these as immediates. A
    // literal graph has no bindings; an empty RuntimeBindings is never read.
    // materializeOneState is the same schema+gather+resolveState used by the
    // binding path; the `constexpr` forces the whole State to fold to IR literals.
    constexpr auto state =
        detail::materializeOneState<G, t>(Edges, RuntimeBindings<G.num_bindings>{});

    // Runtime portion: scatter inputs from work[], invoke mapCoord, scatter back.
    ck_tile::static_array<CoordT, nin>  in_buf{};
    for(uint8_t i = 0; i < nin; ++i) {
        in_buf[i] = work[G.t_input_edge_ids[t][i]];
    }
    ck_tile::static_array<CoordT, nout> out_buf{};
    Impl::template mapCoord<nin, nout>(state, out_buf.elems, in_buf.elems);

    for(uint8_t i = 0; i < nout; ++i) {
        work[G.t_output_edge_ids[t][i]] = out_buf[i];
    }
}

template <auto G, typename CoordT, ck_tile::index_t... Is>
CK_TILE_HOST_DEVICE constexpr void
mapCoordFoldLiteral(ck_tile::static_array<CoordT, G.num_edges>& work,
                    ck_tile::sequence<Is...>) noexcept
{
    constexpr auto edges = detail::resolveEdgeLengths<G>(RuntimeBindings<G.num_bindings>{});
    (executeOneTransformLiteral<G, edges, Is>(work), ...);
}

} // namespace detail


/// @brief Literal-graph mapCoord -- no bindings argument, no GraphBindings.
///
/// For graphs with no placeholder<> values. Skips the entire GraphBindings
/// scaffold: no `make_graph_bindings` call, no `GraphBindings<G>` aggregate, no
/// `tuple<States...>`, no `has_unique_object_representations` check. Each
/// transform's edge lengths + schema + State are `constexpr` locals (mirrors
/// V3's `applyOne` no-bindings path), so the backend sees IR literals.
template <auto G, typename CoordT>
CK_TILE_HOST_DEVICE constexpr void
mapCoord(CoordT* output_coords, const CoordT* input_coords) noexcept
{
    static_assert(sizeof(CoordT) <= sizeof(Slot::PayloadT),
                  "CoordT cannot be wider than the graph's Pool payload width (sizeof(Slot::PayloadT))");

    ck_tile::static_array<CoordT, G.num_edges> work{};
    constexpr uint8_t n_in = G.input_edge_ids.count();
    for(uint8_t i = 0; i < n_in; ++i) {
        work[G.input_edge_ids[i]] = input_coords[i];
    }

    detail::mapCoordFoldLiteral<G>(work,
                                   ck_tile::make_index_sequence<G.num_transforms>{});

    constexpr uint8_t n_out = G.output_edge_ids.count();
    for(uint8_t i = 0; i < n_out; ++i) {
        output_coords[i] = work[G.output_edge_ids[i]];
    }
}

/// @brief Single-output convenience wrapper (mirrors V3's `calculateOffset`).
///        For graphs with one boundary output (typical EMBED-terminated tensor
///        descriptor), returns the single computed offset directly instead of
///        writing through a pointer.
///
/// Two overloads, paralleling `mapCoord`:
///   1) Literal:    `calculateOffset<g>(coord)`         -- no bindings.
///   2) Bindings:   `calculateOffset<g>(coord, gb)`      -- runtime GB.
template <auto G, ck_tile::index_t N>
CK_TILE_HOST_DEVICE constexpr ck_tile::index_t
calculateOffset(const ck_tile::static_array<ck_tile::index_t, N>& coord) noexcept
{
    static_assert(N == G.input_edge_ids.count(),
                  "v4::calculateOffset: coord size must equal G.input_edge_ids.count()");
    static_assert(G.output_edge_ids.count() == 1,
                  "v4::calculateOffset: graph must have exactly one boundary output");
    ck_tile::index_t out;
    mapCoord<G>(&out, coord.elems);
    return out;
}

template <auto G, ck_tile::index_t N>
CK_TILE_HOST_DEVICE constexpr ck_tile::index_t
calculateOffset(const ck_tile::static_array<ck_tile::index_t, N>& coord,
                const GraphBindings<G>&                            gb) noexcept
{
    static_assert(N == G.input_edge_ids.count(),
                  "v4::calculateOffset: coord size must equal G.input_edge_ids.count()");
    static_assert(G.output_edge_ids.count() == 1,
                  "v4::calculateOffset: graph must have exactly one boundary output");
    ck_tile::index_t out;
    mapCoord<G>(&out, coord.elems, gb);
    return out;
}


/// @brief Map an input coordinate vector through the graph to its output.
///
/// @tparam G       NTTP TransformGraph instance (compile-time graph).
/// @tparam CoordT  Coord type (e.g. int32_t for Slot32 graphs).
///
/// @param output_coords  Output buffer; receives g.output_edge_ids.count values.
/// @param input_coords   Input buffer; size = g.input_edge_ids.count.
/// @param gb             Result of make_graph_bindings<G>(runtime_bindings).
///
/// Walks transforms in topo order. Each step gathers input coords from a
/// per-edge working buffer, calls Impl::mapCoord, scatters output coords
/// back. After all transforms, graph-output edges are copied from the
/// working buffer to output_coords.
template <auto G, typename CoordT>
CK_TILE_HOST_DEVICE constexpr void
mapCoord(CoordT*                 output_coords,
         const CoordT*           input_coords,
         const GraphBindings<G>& gb) noexcept
{
    static_assert(sizeof(CoordT) <= sizeof(Slot::PayloadT),
                  "CoordT cannot be wider than the graph's Pool payload width (sizeof(Slot::PayloadT))");

    // Per-edge working buffer: holds intermediate coord values keyed by
    // edge id. Sized to G.num_edges (not MAX_EDGES_V4) -- P2 of the V4
    // scratch-spill fix. For typical graphs (5-15 edges) this is a small
    // alloca that SROA promotes to scalar registers.
    ck_tile::static_array<CoordT, G.num_edges> work{};

    // Step 1: copy graph-input coords into the working buffer at boundary edge ids.
    // Step 11 cleanup: iterate up to compile-time count(), not MAX_TENSOR_DIMS_V4.
    // G is NTTP so G.input_edge_ids.count() folds to a literal at -O3.
    constexpr uint8_t n_in = G.input_edge_ids.count();
    for(uint8_t i = 0; i < n_in; ++i) {
        work[G.input_edge_ids[i]] = input_coords[i];
    }

    // Step 2: walk transforms in topo order. NTTP-fold over compile-time
    // indices; each transform's NIn/NOut/TransformType resolves at compile
    // time, fully monomorphizing the per-Impl call.
    detail::mapCoordFold<G>(work, gb,
                            ck_tile::make_index_sequence<G.num_transforms>{});

    // Step 3: copy resolved graph-output coords from the working buffer to
    // the user's output array.
    constexpr uint8_t n_out = G.output_edge_ids.count();
    for(uint8_t i = 0; i < n_out; ++i) {
        output_coords[i] = work[G.output_edge_ids[i]];
    }
}

// =============================================================================
// 15. V3-style splice (SubgraphNode consumer)
// =============================================================================
//
// User-facing entry point: transform(g_inner, read(outer_dims), write(outer_dims))
// returns a SubgraphNode<SlotT> (defined in section 11 builder DSL).
// insertNode dispatches it here via spliceInto().
//
// Per V4_REWORK_PLAN section 3.2 renames table (2026-05-14): the old SubGraphSplice<V>
// marker was folded into SubgraphNode<V> -- one type carries inner-graph-ref
// AND outer routing, eliminating the marker/decl two-type pattern.

namespace detail {

/// @brief Compute the next free pool offset by watermarking against
///        previously-claimed positions.
constexpr uint8_t
nextFreePoolOffset(const TransformGraph& outer) noexcept
{
    return outer.pool_used;
}

/// @brief Compute the next free edge id from the PARTIAL outer graph.
///
/// CANNOT use `outer.num_edges` -- that field is only populated at
/// `make_transform_graph` finalization (via `computeNumEdges`); at splice
/// time it's still 0 because the outer graph is still under construction.
/// Instead, reuse `computeNumEdges()` which scans boundary slots + per-
/// transform routing -- order-independent and correct mid-build.
constexpr uint8_t
nextFreeEdgeId(const TransformGraph& outer) noexcept
{
    return computeNumEdges(outer);
}

/// @brief Classify an inner edge id as boundary-input / boundary-output /
///        internal. Returns the remapped outer edge id.
///
/// Three cases (mirrors V3's remap_one_slot):
///   - Inner id matches an inner.input_edge_ids[i] -> outer.read_dims[i]
///   - Inner id matches an inner.output_edge_ids[i] -> outer.write_dims[i]
///   - Otherwise (internal) -> inner_id + outer_internal_offset
constexpr uint8_t
remapInnerEdgeId(const TransformGraph& inner,
                 uint8_t                       inner_id,
                 const DimIds&                 outer_read_dims,
                 const DimIds&                 outer_write_dims,
                 uint8_t                       outer_internal_offset) noexcept
{
    // R4/Step 11: iterate up to the actual count, not MAX_TENSOR_DIMS_V4. The
    // INVALID_EDGE_ID sentinel break is gone -- DimIds::count() walks once at
    // function entry and the loops then have a clean compile-time-knowable
    // bound (when DimIds is constexpr) which SROA can promote.
    const uint8_t n_in  = inner.input_edge_ids.count();
    const uint8_t n_out = inner.output_edge_ids.count();

    // Boundary input?
    for(uint8_t i = 0; i < n_in; ++i) {
        if(inner.input_edge_ids[i] == inner_id) { return outer_read_dims[i]; }
    }
    // Boundary output?
    for(uint8_t i = 0; i < n_out; ++i) {
        if(inner.output_edge_ids[i] == inner_id) { return outer_write_dims[i]; }
    }
    // Internal -- shift by outer_internal_offset.
    return static_cast<uint8_t>(inner_id + outer_internal_offset);
}

} // namespace detail

/// @brief V3-style splice. Framework computes offsets and remaps edge ids.
///
/// @pre  Outer and inner share the same SlotT (D3 lock; enforced in
///       insertNode via `transformErrorD3LockMixedSlotT()` stub +
///       static_assert before this is reached). spliceInto is internal --
///       the user-facing entry point is the consteval
///       `transform(g_inner, read, write)` overload which produces a
///       SubgraphNode that insertNode dispatches here.
///
/// The SubgraphNode<V> carries inner-graph-ref + outer routing in a single
/// arg; this function unpacks them and performs the inline.
constexpr void
spliceInto(TransformGraph&       outer,
           const SubgraphNode&   sub) noexcept
{
    const auto& inner            = sub.inner;
    const auto& outer_read_dims  = sub.read_ids;
    const auto& outer_write_dims = sub.write_ids;

    // Overflow guards.
    if(outer.pool_used + inner.pool_used > MAX_POOL_VALUES_V4) {
        graphValidationErrorSpliceEdgeOverflow();
    }
    if(outer.num_transforms + inner.num_transforms > MAX_TRANSFORMS_V4) {
        graphValidationErrorSpliceTransformOverflow();
    }

    const uint8_t outer_pool_offset      = detail::nextFreePoolOffset(outer);
    const uint8_t outer_internal_offset  = detail::nextFreeEdgeId(outer);

    // 1. Copy inner pool into outer pool at offset (whole-slot copy).
    for(uint8_t i = 0; i < inner.pool_used; ++i) {
        outer.pool[outer_pool_offset + i] = inner.pool[i];
    }

    // 2. Walk the copied range; remap EDGE_ID payloads through the Slot
    //    factory so the range check + UnsignedT bit_cast match the original
    //    construction path.
    for(uint8_t i = 0; i < inner.pool_used; ++i) {
        const auto idx = outer_pool_offset + i;
        if(outer.pool[idx].is_edge_id()) {
            const uint8_t old_id = outer.pool[idx].as_edge_id();
            const uint8_t new_id = detail::remapInnerEdgeId(
                inner, old_id, outer_read_dims, outer_write_dims, outer_internal_offset);
            outer.pool[idx] = Slot::from_edge_id(new_id);
        }
    }

    // 3. Append inner.transforms with base_offset shifted; remap each
    //    transform's t_input_edge_ids / t_output_edge_ids through the same
    //    boundary-aware classifier.
    //
    // Step 11 cleanup: iterate up to per-routing count(), not MAX_TENSOR_DIMS_V4.
    // SROA can promote the bounded loops; the sentinel-break versions could not.
    for(uint8_t t = 0; t < inner.num_transforms; ++t) {
        auto coord = inner.transforms[t];
        coord.base_offset = static_cast<uint8_t>(coord.base_offset + outer_pool_offset);

        const auto outer_transform_idx = outer.num_transforms + t;
        outer.transforms[outer_transform_idx] = coord;

        DimIds in_remapped{};
        DimIds out_remapped{};
        const uint8_t n_in  = inner.t_input_edge_ids[t].count();
        const uint8_t n_out = inner.t_output_edge_ids[t].count();
        for(uint8_t i = 0; i < n_in; ++i) {
            in_remapped[i] = detail::remapInnerEdgeId(
                inner, inner.t_input_edge_ids[t][i],
                outer_read_dims, outer_write_dims, outer_internal_offset);
        }
        for(uint8_t i = 0; i < n_out; ++i) {
            out_remapped[i] = detail::remapInnerEdgeId(
                inner, inner.t_output_edge_ids[t][i],
                outer_read_dims, outer_write_dims, outer_internal_offset);
        }
        outer.t_input_edge_ids[outer_transform_idx]  = in_remapped;
        outer.t_output_edge_ids[outer_transform_idx] = out_remapped;

        // Carry the per-transform anchors so the spliced transform's anchored /
        // derived edge lengths actually resolve in resolveOneTransformEdges
        // (transform edges resolve ONLY there -- input_edge_anchors / output_edge_anchors[] is
        // boundary only). Two remaps compose: the destination INDEX (the inner
        // anchor for inner edge `inner_id` lands at the remapped outer edge id)
        // and, for EDGE_ID anchors, the PAYLOAD (the referenced edge id is
        // remapped through the same boundary-aware classifier). Only non-UNUSED
        // anchors are written so an UNUSED reader does not clobber a producer's
        // anchor at the shared outer edge id (mirrors insertTransform).
        for(uint8_t i = 0; i < n_in; ++i) {
            const Slot src = inner.t_input_edge_anchors[inner.t_input_edge_ids[t][i]];
            if(src.kind() != Kind::UNUSED) {
                outer.t_input_edge_anchors[in_remapped[i]] = src.is_edge_id()
                    ? Slot::from_edge_id(detail::remapInnerEdgeId(
                          inner, src.as_edge_id(), outer_read_dims, outer_write_dims,
                          outer_internal_offset))
                    : src;
            }
        }
        for(uint8_t i = 0; i < n_out; ++i) {
            const Slot src = inner.t_output_edge_anchors[inner.t_output_edge_ids[t][i]];
            if(src.kind() != Kind::UNUSED) {
                outer.t_output_edge_anchors[out_remapped[i]] = src.is_edge_id()
                    ? Slot::from_edge_id(detail::remapInnerEdgeId(
                          inner, src.as_edge_id(), outer_read_dims, outer_write_dims,
                          outer_internal_offset))
                    : src;
            }
        }

        // Carry the DERIVED-presence bits so resolveOneTransformEdges runs its
        // Pass 2 (deriveInputLength / deriveOutputLength) for this transform.
        const uint16_t bit = static_cast<uint16_t>(uint16_t{1} << outer_transform_idx);
        if((inner.t_input_derived_mask >> t) & 1u) {
            outer.t_input_derived_mask =
                static_cast<uint16_t>(outer.t_input_derived_mask | bit);
        }
        if((inner.t_output_derived_mask >> t) & 1u) {
            outer.t_output_derived_mask =
                static_cast<uint16_t>(outer.t_output_derived_mask | bit);
        }
    }

    // 4. Count inner internal edges into outer.num_edges. Inner internal-edge
    //    anchors were already carried in step 3 (per-edge t_*_edge_anchors);
    //    inner boundary edges inherit their anchor from the outer side. The
    //    boundary anchor arrays are boundary-position indexed, so there is no
    //    per-edge boundary anchor to copy for an internal edge here.
    //
    // Step 11 cleanup: hoist count() calls out of the inner loops so the
    // is_boundary check uses a tight known-arity loop.
    const uint8_t inner_n_in  = inner.input_edge_ids.count();
    const uint8_t inner_n_out = inner.output_edge_ids.count();
    for(ck_tile::index_t i = 0; i < inner.num_edges; ++i) {
        const auto inner_id = static_cast<uint8_t>(i);
        // Skip boundary edges; they're already anchored on the outer side.
        bool is_boundary = false;
        for(uint8_t j = 0; j < inner_n_in; ++j) {
            if(inner.input_edge_ids[j] == inner_id) { is_boundary = true; break; }
        }
        if(!is_boundary) {
            for(uint8_t j = 0; j < inner_n_out; ++j) {
                if(inner.output_edge_ids[j] == inner_id) { is_boundary = true; break; }
            }
        }
        if(!is_boundary) {
            outer.num_edges = static_cast<uint8_t>(outer.num_edges + 1);
        }
    }

    outer.pool_used      = static_cast<uint8_t>(outer.pool_used + inner.pool_used);
    outer.num_transforms = static_cast<uint8_t>(outer.num_transforms + inner.num_transforms);
}


// NOTE: the Schema-framework smoke checks (member_array_traits /
// member_scalar_traits / is_array_member_v / member_ptr_class_t /
// index_of_member_v over a TestSchema) live in the V4 SubGraph test TU as
// runtime EXPECT checks rather than header-scope static_asserts -- they emit
// no code, so one compiled test preserves coverage without forcing the trait
// instantiations into every TU, and a failure reports instead of breaking the
// build for every other test.


} // namespace ck_tile::core::transform::v4

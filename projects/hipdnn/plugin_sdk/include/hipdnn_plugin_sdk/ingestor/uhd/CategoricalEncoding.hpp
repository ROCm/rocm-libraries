// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <hipdnn_plugin_sdk/ingestor/uhd/Sha256.hpp>

/// @file CategoricalEncoding.hpp
/// @brief The fixed string -> number table a model's categorical features read
/// (RFC 0019 §6.5).
///
/// A feature vector is doubles, but the fields a kernel varies along are not all
/// numeric: `dtype` and `layout` are strings. Without a table they cannot be features
/// at all -- JsonLogicEvaluator::toDouble refuses a string outright, because returning
/// NaN made a GBDT route the value down `default_left` and score it as ordinary data.
///
/// **The mapping is global and fixed, not per-descriptor.** RFC 0019 §6.5 has the
/// training tool observe each field's values and ship the map it saw inside the UHD.
/// That is not what this is. Two engines that both feed `dtype="fp16"` to their models
/// must produce the same number: otherwise a model trained against one engine's corpus
/// means nothing against another's, and the cross-engine score comparison of
/// RFC 0019 §11.3 compares two different axes that happen to share a name. A
/// per-descriptor map cannot promise that -- it encodes whatever strings the training
/// set happened to contain, in whatever order they were observed.
///
/// **Assignments are append-only and stable forever.** A trained `model.bin` has these
/// numbers baked into its split thresholds. Renumbering `fp16` from 13 to 12 does not
/// fail a load, does not fail `features_hash` (the signature text is unchanged) and
/// does not fail a test that only checks the runtime agrees with itself: it silently
/// re-points every threshold in every model in the field at a different data type.
/// So this is not a request to be careful. The digest below freezes the first
/// CATEGORICAL_ENCODING_FROZEN_ENTRIES entries; editing, reordering or deleting any of
/// them changes the digest and fails the pin in TestCategoricalEncoding.cpp and in
/// tools/uhd_gen/tests/test_categorical_encoding.py. Appending past the frozen prefix
/// leaves the digest alone, which is the whole point -- growth is free, mutation is not.
///
/// Mirrored by CATEGORICAL_ENCODING in tools/uhd_gen/features.py, which is what the
/// training side encodes with. The two are pinned equal by the Python test above, which
/// reads *this file*: a hand-maintained Python copy that merely agreed with itself is
/// the defect that already shipped here once, as a FlatBuffer writer whose layout the
/// C++ reader disagreed with for months.
namespace hipdnn_plugin_sdk::ingestor::uhd
{

/// One assignment: a category, one of its values, and the number that value becomes.
struct CategoricalEntry
{
    std::string_view category;
    std::string_view value;
    int32_t code;
};

/// Bumped only when the *meaning* of the table changes (a new ordering rule, a
/// different code space). A bump invalidates every model in the field by construction,
/// so it is a migration, not an edit; it is hashed into the digest below so it cannot
/// happen quietly.
inline constexpr int32_t CATEGORICAL_ENCODING_VERSION = 1;

/// The authoritative table. Grouped by category; codes are per-category and start at 0.
///
/// Categories and members are what this codebase actually has, not a taxonomy invented
/// for the RFC:
///
///   - `dtype`  -- every spelling `to_string(DataType)` produces in
///                 hipdnn_frontend/Types.hpp. That function is the only place the
///                 library turns a data type into a string, so it is the only set a
///                 `$kernel.dtype` or `$q.dtype` binding can hold. `unknown`, its
///                 fallthrough, is deliberately absent: an unrecognized data type must
///                 fail loudly rather than encode to a number the model can split on.
///
///   - `layout` -- the TensorLayout constants in
///                 hipdnn_data_sdk/utilities/Tensor.hpp (`NCHW`, `BSHD`, ...), by their
///                 own `name` field.
///
/// Nothing else in the tree has an enumerated string vocabulary. `$device.arch` is
/// string-valued but open (`gfx942:sramecc+:xnack-`, `gfx9-4-generic`, whatever the
/// next chip reports), so freezing a code space over it would be inventing one.
/// MetadataType (Descriptors.hpp) confirms the scope: BOOL is already 0/1, INT and
/// FLOAT are already numbers, INT_LIST has no scalar form and must be reduced by an
/// explicit expression (RFC 0019 §6.5), and STRING is the only type left needing this.
///
/// **Ordering is not arbitrary** (RFC 0019 §6.5 "Ordering"): an integer code implies an
/// order a tree splits on, so `dtype < 12` has to mean something. dtype is ordered by
/// element byte width ascending -- a split then separates narrow from wide precision,
/// which is the property a kernel's throughput actually turns on. Within one width,
/// floating-point precedes integer, then alphabetical, purely so the order is
/// reproducible. layout is ordered by tensor rank, channel-first before channel-last,
/// with the SDPA pair last; rank is the only real order layouts have.
inline constexpr std::array<CategoricalEntry, 32> CATEGORICAL_ENCODING_TABLE = {{
    // dtype -- ordered by element byte width (RFC 0019 §6.5 "Ordering").
    // 4-bit
    {"dtype", "fp4_e2m1", 0},
    {"dtype", "int4", 1},
    // 6-bit
    {"dtype", "fp6_e2m3", 2},
    {"dtype", "fp6_e3m2", 3},
    // 1 byte
    {"dtype", "fp8_e4m3", 4},
    {"dtype", "fp8_e4m3_fnuz", 5},
    {"dtype", "fp8_e5m2", 6},
    {"dtype", "fp8_e5m2_fnuz", 7},
    {"dtype", "fp8_e8m0", 8},
    {"dtype", "int8", 9},
    {"dtype", "uint8", 10},
    {"dtype", "boolean", 11},
    // 2 bytes
    {"dtype", "bf16", 12},
    {"dtype", "fp16", 13},
    // 4 bytes
    {"dtype", "fast_float_for_fp8", 14},
    {"dtype", "fp32", 15},
    {"dtype", "int32", 16},
    {"dtype", "int8x4", 17},
    {"dtype", "uint8x4", 18},
    // 8 bytes
    {"dtype", "complex_fp32", 19},
    {"dtype", "fp64", 20},
    {"dtype", "int64", 21},
    // 16 bytes
    {"dtype", "complex_fp64", 22},
    // 32 bytes
    {"dtype", "int8x32", 23},

    // layout -- ordered by rank, channel-first before channel-last, SDPA last.
    {"layout", "NCL", 0},
    {"layout", "NLC", 1},
    {"layout", "NCHW", 2},
    {"layout", "NHWC", 3},
    {"layout", "NCDHW", 4},
    {"layout", "NDHWC", 5},
    {"layout", "BHSD", 6},
    {"layout", "BSHD", 7},
}};

/// How many leading entries the pinned digest covers. Everything shipped so far.
///
/// Appending an entry at the end leaves the first FROZEN_ENTRIES untouched, so the
/// digest is unchanged and the pin passes without being edited. Extending the freeze
/// over newly appended entries means raising this count *and* the digest together --
/// two literals in one reviewable diff, which is what makes "did you renumber
/// something?" answerable from the patch alone.
inline constexpr size_t CATEGORICAL_ENCODING_FROZEN_ENTRIES = 32;

/// SHA-256 (first 16 hex chars) of categoricalEncodingCanonicalForm(), pinned.
/// Recomputed and compared by TestCategoricalEncoding.cpp and by the Python mirror.
inline constexpr std::string_view CATEGORICAL_ENCODING_FROZEN_DIGEST
    = "sha256:bf20c5a8243803c2";

namespace detail
{

/// True if `text` survives a JSON writer byte-for-byte.
///
/// The canonical form below is rendered by hand here and by json.dumps() in
/// tools/uhd_gen/features.py. Those agree only while no character needs escaping, so a
/// value containing a quote, a backslash or a non-ASCII byte would silently give the
/// two sides different digests. Rejecting at compile time is cheaper than discovering
/// it as a mismatched pin.
constexpr bool isDigestSafe(std::string_view text)
{
    for(const char character : text)
    {
        if(character < 0x20 || character > 0x7E || character == '"' || character == '\\')
        {
            return false;
        }
    }
    return !text.empty();
}

constexpr bool tableIsDigestSafe()
{
    for(const auto& entry : CATEGORICAL_ENCODING_TABLE)
    {
        if(!isDigestSafe(entry.category) || !isDigestSafe(entry.value))
        {
            return false;
        }
    }
    return true;
}

/// Lowercase the 26 ASCII letters; leave every other byte exactly as it is.
///
/// Deliberately not std::tolower. That function consults the current C locale, so the
/// code a category value resolves to would depend on the machine the engine or the
/// trainer happens to run on -- in a Turkish locale 'I' does not fold to 'i'. The whole
/// point of this table is that every engine and every trainer agree on one number, so
/// the fold has to be a fixed function of the bytes and of nothing else. std::tolower
/// is also undefined behaviour on a negative char, which every byte above 0x7F is here.
constexpr char foldAsciiCase(char character)
{
    return (character >= 'A' && character <= 'Z') ? static_cast<char>(character - 'A' + 'a')
                                                  : character;
}

/// True if `first` and `second` are the same string up to ASCII letter case.
///
/// Case is not a real difference: a rocKE KMD declares its dtype values as `BF16` while
/// to_string(DataType) spells the same thing `bf16`, and refusing one of them buys no
/// safety -- it only forces a second spelling into the table, which would imply the two
/// are separate category members that happen to share a code.
///
/// A different *spelling* remains a real difference. This compares character by
/// character, so `float16` still never reaches `fp16`: that is a genuine near-miss
/// between two vocabularies, and silently accepting it is how a model gets trained on
/// numbers the runtime will never produce.
constexpr bool equalsFoldingAsciiCase(std::string_view first, std::string_view second)
{
    if(first.size() != second.size())
    {
        return false;
    }
    for(size_t index = 0; index < first.size(); ++index)
    {
        if(foldAsciiCase(first[index]) != foldAsciiCase(second[index]))
        {
            return false;
        }
    }
    return true;
}

/// True if no category repeats a value and no category repeats a code.
///
/// A repeated value makes the lookup order-dependent; a repeated code collapses two
/// categories' members onto one number, which a model cannot tell apart. Both are the
/// kind of mistake an append introduces, so it is caught where the append is made.
///
/// Compared up to ASCII case because the lookup is: since `BF16` and `bf16` resolve to
/// the same entry, appending both would be exactly the order-dependent lookup this
/// guards against, and it has to fail at compile time rather than pick whichever row
/// came first.
constexpr bool tableIsUnambiguous()
{
    for(size_t i = 0; i < CATEGORICAL_ENCODING_TABLE.size(); ++i)
    {
        for(size_t j = i + 1; j < CATEGORICAL_ENCODING_TABLE.size(); ++j)
        {
            const auto& first = CATEGORICAL_ENCODING_TABLE[i];
            const auto& second = CATEGORICAL_ENCODING_TABLE[j];
            if(equalsFoldingAsciiCase(first.category, second.category)
               && (equalsFoldingAsciiCase(first.value, second.value)
                   || first.code == second.code))
            {
                return false;
            }
        }
    }
    return true;
}

static_assert(tableIsDigestSafe(),
              "A category or value contains a character the two languages' JSON writers "
              "render differently, so the frozen digest would not agree across them.");
static_assert(tableIsUnambiguous(),
              "A category repeats a value or a code; the encoding is no longer a function.");
static_assert(CATEGORICAL_ENCODING_FROZEN_ENTRIES <= CATEGORICAL_ENCODING_TABLE.size(),
              "The frozen prefix is longer than the table; entries were deleted, which "
              "invalidates every model trained against them.");

} // namespace detail

/// The exact bytes the frozen digest is taken over.
///
/// `[<version>,[["<category>","<value>",<code>],...]]` over the frozen prefix, in table
/// order -- the compact JSON both nlohmann::json::dump() and Python's
/// json.dumps(separators=(",", ":")) produce, so the Python mirror hashes an identical
/// string rather than a differently-spelled one.
inline std::string categoricalEncodingCanonicalForm()
{
    std::string canonical = "[";
    canonical += std::to_string(CATEGORICAL_ENCODING_VERSION);
    canonical += ",[";
    for(size_t i = 0; i < CATEGORICAL_ENCODING_FROZEN_ENTRIES; ++i)
    {
        const auto& entry = CATEGORICAL_ENCODING_TABLE[i];
        if(i != 0)
        {
            canonical += ',';
        }
        canonical += "[\"";
        canonical += entry.category;
        canonical += "\",\"";
        canonical += entry.value;
        canonical += "\",";
        canonical += std::to_string(entry.code);
        canonical += ']';
    }
    canonical += "]]";
    return canonical;
}

/// Fingerprint of the frozen prefix, in the `sha256:<16 hex>` form the UHD already uses
/// for features_hash.
inline std::string categoricalEncodingDigest()
{
    return "sha256:" + sha256(categoricalEncodingCanonicalForm()).substr(0, 16);
}

/// True if `category` has a table, i.e. its values are meant to encode.
///
/// Separates "this string means nothing here" from "this category is known and that is
/// not one of its values" -- two failures that read identically at a call site but mean
/// very different things to whoever has to fix them.
///
/// Matched up to ASCII case, like encodeCategorical below, so the two never disagree
/// about whether a category is known.
inline bool isKnownCategory(std::string_view category)
{
    for(const auto& entry : CATEGORICAL_ENCODING_TABLE)
    {
        if(detail::equalsFoldingAsciiCase(entry.category, category))
        {
            return true;
        }
    }
    return false;
}

/// The number `value` takes in `category`, or nullopt if the pair is not in the table.
///
/// The match ignores ASCII letter case on both the category and the value. A rocKE KMD
/// declares `"BF16"` where to_string(DataType) produces `"bf16"`; those are one value
/// with two spellings of the same letters, and rejecting one of them stopped a real
/// gfx942 sweep at training for no safety in return. The fold happens here, at lookup,
/// rather than as extra rows in the table: extra rows would double the table, move the
/// frozen digest, and assert that `BF16` and `bf16` are distinct members that merely
/// share a code.
///
/// It is a fold, not an alias table. `float16` still finds nothing, because two
/// vocabularies that spell the type differently are genuinely different and quietly
/// bridging them trains a model on numbers the runtime cannot reproduce.
inline std::optional<double> encodeCategorical(std::string_view category, std::string_view value)
{
    for(const auto& entry : CATEGORICAL_ENCODING_TABLE)
    {
        if(detail::equalsFoldingAsciiCase(entry.category, category)
           && detail::equalsFoldingAsciiCase(entry.value, value))
        {
            return static_cast<double>(entry.code);
        }
    }
    return std::nullopt;
}

/// The category a `$namespace.field` reference names: the field, without its namespace.
///
/// `$kernel.dtype` and `$q.dtype` are both `dtype` on purpose. The category is a
/// property of the value, not of who is holding it, so a kernel's data type and a
/// tensor's data type encode to the same number -- which is what makes two engines'
/// feature vectors comparable at all (RFC 0019 §11.3).
///
/// Returns empty for anything that is not a namespaced reference, including a bare
/// string literal, so a literal never accidentally encodes.
inline std::string_view categoryOfReference(std::string_view reference)
{
    if(reference.empty() || reference.front() != '$')
    {
        return {};
    }
    const size_t dot = reference.rfind('.');
    if(dot == std::string_view::npos || dot + 1 == reference.size())
    {
        return {};
    }
    return reference.substr(dot + 1);
}

} // namespace hipdnn_plugin_sdk::ingestor::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

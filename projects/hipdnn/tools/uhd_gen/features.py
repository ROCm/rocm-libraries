# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The UHD feature contract: features_signature and features_hash (RFC 0019 7.2, 7.3).

This is a cross-language contract. The functions here decide what the descriptor
says its features are and how that claim is fingerprinted; the C++ runtime
recomputes the same fingerprint from the descriptor it loads
(``FeatureExtractor::computeHash`` in ``backend/src/heuristics/uhd/``). If the two
canonicalizations drift, every descriptor this tool emits fails to load.

Deliberately free of LightGBM and FlatBuffers imports so the contract can be
tested and reused without the training stack installed.
"""
from __future__ import annotations

import hashlib
import json
import math
import string

#: Largest numeric literal magnitude the cross-language canonical form is safe for.
#:
#: Above this the two JSON writers stop agreeing, in three independent ways:
#:   - Python's repr switches a float to scientific notation at 1e16, nlohmann at
#:     1e15, so the whole decade renders differently ("1000000000000000.0" vs "1e+15");
#:   - an integer outside int64/uint64 keeps arbitrary precision here but degrades to
#:     double in nlohmann -- and that lossy conversion makes distinct values collide,
#:     so the runtime would accept a hash computed over a *different* signature;
#:   - NaN/Infinity are a Python json extension that nlohmann rejects outright.
#:
#: Feature literals are tile sizes, dimensions and thresholds, so this bound is many
#: orders of magnitude clear of anything real. Mirrored by kMaxSafeNumericLiteral in
#: backend/src/heuristics/uhd/FeatureExtractor.cpp.
MAX_SAFE_NUMERIC_LITERAL = 1e15

__all__ = [
    "CATEGORICAL_ENCODING",
    "CATEGORICAL_ENCODING_FROZEN_DIGEST",
    "CATEGORICAL_ENCODING_FROZEN_ENTRIES",
    "CATEGORICAL_ENCODING_VERSION",
    "FEATURE_NAMESPACES",
    "MAX_SAFE_NUMERIC_LITERAL",
    "build_features_signature",
    "canonicalize_signature",
    "categorical_encoding_canonical_form",
    "categorical_encoding_digest",
    "categorical_encoding_entries",
    "category_of_reference",
    "compute_features_hash",
    "encode_categorical",
    "encode_feature_value",
    "parse_signature_entry",
]


#: Namespaces the runtime binds, per RFC 0019 7.1. A reference outside these resolves
#: to nothing at selection time (FeatureExtractionContext binds exactly device/kernel/q
#: in backend/src/heuristics/uhd/FeatureExtractor.cpp).
FEATURE_NAMESPACES = ("device", "kernel", "q")


# ---- Categorical encoding (RFC 0019 6.5) ---------------------------------------
#
# A mirror, not a source. The authoritative table is CATEGORICAL_ENCODING_TABLE in
# plugin_sdk/include/hipdnn_plugin_sdk/ingestor/uhd/CategoricalEncoding.hpp, which is
# what the runtime reads; this copy is what training encodes with. Two hand-maintained
# copies is the defect class that already shipped here once -- a Python FlatBuffer
# writer whose layout the C++ reader disagreed with, undetected for months -- so
# tests/test_categorical_encoding.py reads the header itself and fails when either side
# moves alone. Editing this dict without editing the header is caught, and vice versa.
#
# The mapping is global and fixed, not observed per training set: two engines that both
# feed dtype="fp16" to their models have to produce the same number, or a model trained
# on one corpus is meaningless against another and the cross-engine score comparison of
# RFC 0019 11.3 compares two unrelated axes. Assignments are append-only and permanent;
# a trained model.bin has them baked into its split thresholds, so renumbering one
# silently re-points every threshold in every model in the field.
#
# Ordering, per-category, matches the header (a tree splits on these numbers, so the
# order has to mean something): dtype by element byte width ascending, layout by tensor
# rank with channel-first before channel-last.
CATEGORICAL_ENCODING_VERSION = 1

CATEGORICAL_ENCODING: dict[str, dict[str, int]] = {
    # Every spelling to_string(DataType) produces in hipdnn_frontend/Types.hpp. Its
    # "unknown" fallthrough is deliberately absent: an unrecognized data type must fail
    # loudly rather than encode to something a model can split on.
    "dtype": {
        "fp4_e2m1": 0,
        "int4": 1,
        "fp6_e2m3": 2,
        "fp6_e3m2": 3,
        "fp8_e4m3": 4,
        "fp8_e4m3_fnuz": 5,
        "fp8_e5m2": 6,
        "fp8_e5m2_fnuz": 7,
        "fp8_e8m0": 8,
        "int8": 9,
        "uint8": 10,
        "boolean": 11,
        "bf16": 12,
        "fp16": 13,
        "fast_float_for_fp8": 14,
        "fp32": 15,
        "int32": 16,
        "int8x4": 17,
        "uint8x4": 18,
        "complex_fp32": 19,
        "fp64": 20,
        "int64": 21,
        "complex_fp64": 22,
        "int8x32": 23,
    },
    # The TensorLayout constants in hipdnn_data_sdk/utilities/Tensor.hpp, by their name.
    "layout": {
        "NCL": 0,
        "NLC": 1,
        "NCHW": 2,
        "NHWC": 3,
        "NCDHW": 4,
        "NDHWC": 5,
        "BHSD": 6,
        "BSHD": 7,
    },
}

#: How many leading entries the pinned digest covers, mirroring the header. Appending
#: past this leaves the digest alone; extending the freeze means raising both literals.
CATEGORICAL_ENCODING_FROZEN_ENTRIES = 32

#: Pinned fingerprint of the frozen prefix. Equal to CATEGORICAL_ENCODING_FROZEN_DIGEST
#: in CategoricalEncoding.hpp, which the Python test asserts by reading that file.
CATEGORICAL_ENCODING_FROZEN_DIGEST = "sha256:bf20c5a8243803c2"


def categorical_encoding_entries() -> list[tuple[str, str, int]]:
    """The table flattened to (category, value, code), in declaration order.

    Insertion order carries the ordering rule and the digest depends on it, so this
    must not sort.
    """
    return [
        (category, value, code)
        for category, members in CATEGORICAL_ENCODING.items()
        for value, code in members.items()
    ]


def categorical_encoding_canonical_form() -> str:
    """The exact bytes the frozen digest is taken over.

    Mirrors ``categoricalEncodingCanonicalForm`` in CategoricalEncoding.hpp, which
    renders the same compact JSON by hand. The header static_asserts that no category
    or value needs escaping, which is what lets the two renderings agree byte for byte.
    """
    frozen = categorical_encoding_entries()[:CATEGORICAL_ENCODING_FROZEN_ENTRIES]
    payload = [CATEGORICAL_ENCODING_VERSION, [[c, v, n] for c, v, n in frozen]]
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def categorical_encoding_digest() -> str:
    """Fingerprint of the frozen prefix, in the same ``sha256:<16 hex>`` form as
    features_hash."""
    digest = hashlib.sha256(categorical_encoding_canonical_form().encode()).hexdigest()[:16]
    return f"sha256:{digest}"


def category_of_reference(reference: str) -> str:
    """The category a ``$namespace.field`` reference names: the field, no namespace.

    ``$kernel.dtype`` and ``$q.dtype`` are both ``dtype`` on purpose -- the category is
    a property of the value, not of who holds it. Mirrors ``categoryOfReference``.
    Returns "" for anything that is not a namespaced reference, so a bare string literal
    never encodes.
    """
    if not reference.startswith("$") or "." not in reference:
        return ""
    return reference.rsplit(".", 1)[1]


#: Fold table for the 26 ASCII letters, and nothing else.
#:
#: Deliberately not ``str.lower()``: that is Unicode-aware, so 'İ' and the Kelvin sign
#: 'K' fold into ASCII and a value would resolve to a code the C++ side -- which folds
#: bytes 'A'-'Z' only -- never gives it. The two sides have to agree on one number for
#: one value, so they have to agree on the fold, character for character.
_ASCII_CASE_FOLD = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def _lookup_folding_ascii_case(mapping: dict, key: str):
    """``mapping[key]`` ignoring ASCII letter case, or None.

    Mirrors ``detail::equalsFoldingAsciiCase`` in CategoricalEncoding.hpp. Compares
    whole folded keys, so it accepts a different case of the same spelling and nothing
    else -- ``float16`` does not reach ``fp16``.
    """
    if key in mapping:
        return mapping[key]
    folded = key.translate(_ASCII_CASE_FOLD)
    for candidate, entry in mapping.items():
        if candidate.translate(_ASCII_CASE_FOLD) == folded:
            return entry
    return None


def encode_categorical(category: str, value: str) -> int | None:
    """The number ``value`` takes in ``category``, or None if the pair is not in the
    table.

    Category and value are matched ignoring ASCII letter case, mirroring
    ``encodeCategorical``. A rocKE KMD declares ``"BF16"`` where ``to_string(DataType)``
    produces ``"bf16"``; those are one value spelled with different shift keys, and
    refusing one of them stopped a real gfx942 sweep at training for no safety in
    return. The fold is applied here, at lookup, rather than by adding uppercase rows:
    rows would double the table, move the frozen digest, and claim ``BF16`` and ``bf16``
    are two members that happen to share a code.

    It stays a fold, never an alias table. ``float16`` still returns None, because a
    second vocabulary's spelling is a genuine difference and bridging it silently trains
    the model on numbers the runtime never emits.
    """
    members = _lookup_folding_ascii_case(CATEGORICAL_ENCODING, category)
    if members is None:
        return None
    return _lookup_folding_ascii_case(members, value)


def encode_feature_value(reference: str, value) -> float:
    """Turn one raw logged value into the number the model trains on.

    The training-side mirror of ``JsonLogicEvaluator::evaluateDouble``: the benchmark
    log carries raw values (``dtype`` is logged as the string ``"fp16"``), and this is
    the single point where a string becomes a number. Anything that encodes here
    encodes identically at inference, because both sides read the same table.

    A string with no code raises rather than falling back to NaN or to a hash of the
    text. NaN is a missing value to a GBDT, which routes it down ``default_left`` and
    returns an ordinary leaf -- the row trains as data and nothing in the log says so.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise TypeError(f"{reference}: cannot use {type(value).__name__} as a feature value")

    category = category_of_reference(reference)
    code = encode_categorical(category, value)
    if code is not None:
        return float(code)
    if _lookup_folding_ascii_case(CATEGORICAL_ENCODING, category) is not None:
        raise ValueError(
            f"{reference}: categorical value {value!r} has no code in category "
            f"'{category}'. Append it to CATEGORICAL_ENCODING here and to "
            "CATEGORICAL_ENCODING_TABLE in CategoricalEncoding.hpp; existing codes "
            "must not move."
        )
    raise ValueError(
        f"{reference}: {value!r} is a string and '{category}' has no categorical "
        "encoding, so there is no number this can mean. Reduce the field through an "
        "explicit expression, or add the category to both tables."
    )


def build_features_signature(feature_cols: list[str]) -> list[str]:
    """Build the RFC 0019 features_signature from training feature columns.

    Entries are bare field references (``$q.batch``) per RFC 0019 7.2 -- the canonical
    spelling the runtime's feature extractor expects.

    Column names must be namespace-qualified (``q.batch``, ``kernel.tile_m``,
    ``device.cu_count``). An unqualified name such as ``batch`` produces ``$batch``,
    which the runtime cannot resolve: it binds only the three namespaces above, so
    every selection throws "Undefined variable" and degrades to static order. Nothing
    downstream catches it -- registration only inspects ``$kernel.``-prefixed
    references -- so the descriptor would load, validate, and silently never score.
    """
    unqualified = [
        col for col in feature_cols if not col.startswith(tuple(f"{ns}." for ns in FEATURE_NAMESPACES))
    ]
    if unqualified:
        raise ValueError(
            "features must be namespace-qualified with one of "
            f"{', '.join(FEATURE_NAMESPACES)}; got {unqualified}. "
            "Rename the CSV columns (e.g. 'batch' -> 'q.batch', 'tile_m' -> "
            "'kernel.tile_m', 'cu_count' -> 'device.cu_count'). An unqualified name "
            "produces a descriptor that loads but never scores."
        )
    return [f"${col}" for col in feature_cols]


def parse_signature_entry(entry: str):
    """Parse one features_signature entry to its structural form.

    Mirrors ``FeatureExtractor::parseSignatureEntry`` in
    ``backend/src/heuristics/uhd/FeatureExtractor.cpp``. RFC 0019 7.2 allows two
    spellings: a bare field reference (``$q.seqlen_q``) or a derived JsonLogic
    expression (``{"log2": ["$q.seqlen_q"]}``). A bare reference is not valid JSON on
    its own, so it is lifted to a string rather than parsed.

    Both sides must agree structurally, not textually: hashing raw entry strings would
    make ``$q.batch`` and ``"$q.batch"`` -- the same reference -- hash differently, and
    would make a derived expression hash as an opaque string here while the runtime
    hashes it as a parsed node.
    """
    if entry.startswith("$"):
        return entry
    return json.loads(entry)


def _validate_numeric_literals(node) -> None:
    """Reject literals the runtime would render differently. See MAX_SAFE_NUMERIC_LITERAL."""
    # bool is a subclass of int, and True/False render identically on both sides.
    if isinstance(node, bool):
        return

    if isinstance(node, (int, float)):
        if isinstance(node, float) and not math.isfinite(node):
            raise ValueError(
                "features_signature contains a non-finite numeric literal "
                f"({node!r}); the runtime's JSON parser rejects these outright"
            )
        if abs(node) >= MAX_SAFE_NUMERIC_LITERAL:
            raise ValueError(
                f"features_signature contains the numeric literal {node!r}, whose "
                "magnitude is at or above 1e15. This tool and the runtime render such "
                "values differently, so the features_hash would not match. Rescale the "
                "feature instead."
            )
        return

    if isinstance(node, list):
        for element in node:
            _validate_numeric_literals(element)
    elif isinstance(node, dict):
        for value in node.values():
            _validate_numeric_literals(value)


def canonicalize_signature(signature: list[str]) -> str:
    """Render a features_signature in the canonical form both sides hash.

    Must match ``nlohmann::json::dump()`` on the parsed signature exactly:
      - ``separators=(",", ":")`` -- nlohmann emits no whitespace.
      - ``sort_keys=True``        -- nlohmann's default object type is ``std::map``,
                                     which is key-sorted. (JsonLogic objects are
                                     single-key, so this is belt-and-braces.)
      - ``ensure_ascii=False``    -- nlohmann emits raw UTF-8 rather than \\uXXXX.
    """
    parsed = [parse_signature_entry(entry) for entry in signature]
    _validate_numeric_literals(parsed)
    return json.dumps(parsed, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def compute_features_hash(
    signature: list[str],
    categorical_encoding: dict[str, dict[str, int]] | None = None,
) -> str:
    """Compute the SHA-256 fingerprint of the resolved feature contract.

    Takes the signature itself, not the raw column names, so the value matches what
    the runtime computes from the descriptor it loads.

    Order is significant: RFC 0019 7.2 requires the signature to match training
    exactly, so this must not sort the entries. A permuted signature is a real
    feature-contract break and has to hash differently. (``sort_keys`` inside
    canonicalize_signature sorts *object keys within* an entry, never the entries.)

    RFC 0019 6.3 puts ``categorical_encoding`` inside the same fingerprint, because a
    changed string-to-code map changes what the model reads while leaving the signature
    text identical -- 6.5 says so outright: "features_hash does not catch it because the
    signature text is unchanged."

    The encoding is appended only when there is one, so a signature reading no string
    field hashes exactly as it did before this argument existed. Every model shipped so
    far is that case, and rehashing them would invalidate contracts that are intact.
    """
    serialized = canonicalize_signature(signature)
    if categorical_encoding:
        # sort_keys mirrors std::map on the C++ side, which is key-ordered; the separators
        # match nlohmann's dump(). Both sides must render the same bytes.
        serialized += "|" + json.dumps(
            categorical_encoding, separators=(",", ":"), sort_keys=True, ensure_ascii=False
        )
    digest = hashlib.sha256(serialized.encode()).hexdigest()[:16]
    return f"sha256:{digest}"

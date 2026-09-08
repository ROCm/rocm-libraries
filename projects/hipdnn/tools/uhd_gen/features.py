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
    "FEATURE_NAMESPACES",
    "MAX_SAFE_NUMERIC_LITERAL",
    "build_features_signature",
    "canonicalize_signature",
    "compute_features_hash",
    "derive_categorical_encoding",
    "encode_feature_value",
    "feature_reference",
    "parse_signature_entry",
]


#: Namespaces the runtime binds, per RFC 0019 7.1. A reference outside these resolves
#: to nothing at selection time (FeatureExtractionContext binds exactly device/kernel/q
#: in backend/src/heuristics/uhd/FeatureExtractor.cpp).
FEATURE_NAMESPACES = ("device", "kernel", "q")


def encode_feature_value(reference: str, value) -> float:
    """Turn one raw logged numeric value into the number the model trains on.

    Strings are deliberately not handled here. A categorical value's number comes from
    the encoding the descriptor carries (RFC 0019 §6.5), and ``build_feature_matrix``
    applies that before reaching this function. There is no process-wide table left to
    consult, so a string arriving here means no encoding declared this reference
    categorical -- which is a corpus and a signature that disagree, not a lookup miss.

    It raises rather than falling back to NaN or to a hash of the text. NaN is a missing
    value to a GBDT, which routes it down ``default_left`` and returns an ordinary leaf
    -- the row trains as data and nothing in the log says so.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise TypeError(f"{reference}: cannot use {type(value).__name__} as a feature value")
    raise ValueError(
        f"{reference}: {value!r} is a string and no categorical_encoding declares this "
        "reference, so there is no number it can mean. Derive an encoding from the corpus "
        "with derive_categorical_encoding and pass it, or reduce the field through an "
        "explicit expression."
    )


def _reject_unqualified(feature_cols: list[str]) -> None:
    """Raise unless every column names one of the runtime's namespaces.

    Reports the whole offending set at once: an author fixing a CSV wants the list, not
    one name per run.
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


def feature_reference(column: str) -> str:
    """The signature reference one training column resolves to.

    The single place that knows the column -> ``$reference`` rule. The signature, the
    categorical encoding and the feature matrix all key off it, and they have to agree
    character for character: the runtime looks the encoding up by the reference it read
    out of features_signature, so a second spelling here is a map the runtime never
    finds.
    """
    _reject_unqualified([column])
    return f"${column}"


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
    _reject_unqualified(feature_cols)
    return [feature_reference(col) for col in feature_cols]


def derive_categorical_encoding(df, feature_cols: list[str]) -> dict[str, dict[str, int]]:
    """The string-to-code map for the corpus being trained on (RFC 0019 6.5).

    Per-descriptor, not global: the vocabulary is whatever the corpus holds, and the
    map is written into the descriptor beside the signature it belongs to, so the
    runtime encodes with the map this model was fitted with rather than a table both
    sides have to be kept in step with by hand.

    Keyed by the full ``$``-reference, never by the trailing field name. Two columns
    can share a trailing name and not a vocabulary -- ``kernel.dtype`` holding ``"BF16"``
    beside ``q.attention_dense.dtype`` holding ``"bf16"`` -- and merging those is one
    vocabulary claiming to describe two columns.

    Values are taken exactly as the corpus spells them, with no case folding: the map
    ships with the model, so an unfolded key is the key the runtime looks up. Codes run
    from 0 in ``sorted()`` order of the values, which makes the map, the hash and every
    split threshold reproducible from the corpus alone.

    Numeric columns are absent: they already are numbers, and an entry for one would
    claim the runtime should encode something it must read straight through.
    """
    encoding: dict[str, dict[str, int]] = {}
    for column in feature_cols:
        series = df[column]
        # numpy kinds b/i/u/f are the numeric ones; object, string and pandas
        # `category` report something else. Mirrors build_feature_matrix, which must
        # take the same columns down the same branch.
        if getattr(series.dtype, "kind", "O") in "biuf":
            continue

        values: set[str] = set()
        other_types: set[str] = set()
        for value in series:
            if isinstance(value, str):
                values.add(value)
            else:
                other_types.add(type(value).__name__)

        if not values:
            # An object column holding only numbers (a CSV with a blank cell, say).
            # It encodes as itself, so there is nothing to map.
            continue
        if other_types:
            raise ValueError(
                f"feature column {column!r} mixes strings with "
                f"{', '.join(sorted(other_types))}. A categorical column is encoded "
                "by position, so a raw number in it would collide with whichever "
                "value took that code. Make the column one or the other."
            )

        encoding[feature_reference(column)] = {
            value: code for code, value in enumerate(sorted(values))
        }
    return encoding


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

#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The dispatcher has two key builders and one stated invariant.

``FmhaProblem::canonical_key()`` documents it directly:

    Canonical key for caching -- includes ALL fields used by
    fmha_signature_matches().

``FmhaKernelKey::encode_identifier()`` is the registry's registration and
lookup key and is bound by the same rule: if two kernels differ in a field
that ``fmha_signature_matches()`` compares, but the identifier does not
encode it, both register under one key and one becomes unreachable.

These tests read the real headers, so they track the signature as it grows
rather than pinning today's field list.
"""

import re
import unittest
from pathlib import Path

DISPATCHER = Path(__file__).resolve().parents[1]
KEY_HPP = DISPATCHER / "include" / "ck_tile" / "dispatcher" / "fmha_kernel_key.hpp"
PROBLEM_HPP = DISPATCHER / "include" / "ck_tile" / "dispatcher" / "fmha_problem.hpp"
BACKEND_HPP = (
    DISPATCHER
    / "include"
    / "ck_tile"
    / "dispatcher"
    / "backends"
    / "generated_fmha_backend.hpp"
)


def _body(path, pattern):
    """Return the brace-balanced body of the first match of *pattern*."""
    src = path.read_text(encoding="utf-8")
    m = re.search(pattern, src)
    if not m:
        raise AssertionError(f"could not locate {pattern!r} in {path.name}")
    i = src.index("{", m.end() - 1)
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[i : j + 1]
    raise AssertionError(f"unbalanced braces after {pattern!r} in {path.name}")


def compared_fields():
    """Signature fields that fmha_signature_matches() actually compares."""
    body = _body(BACKEND_HPP, r"inline bool fmha_signature_matches\s*\(")
    return set(re.findall(r"\bsig\.(\w+)", body))


def identifier_fields():
    """Signature fields encode_identifier() emits into the registry key."""
    body = _body(KEY_HPP, r"std::string encode_identifier\s*\(\s*\)\s*const")
    return set(re.findall(r"\bsignature\.(\w+)", body))


# fmha_signature_matches() compares FmhaKernelSignature against FmhaProblem, and
# the two structs do not use the same names for the same thing. Only genuine
# renames belong here; anything else would hide a real omission.
_PROBLEM_ALIASES = {"family": "requested_family"}


def canonical_key_tokens():
    """Identifiers appearing in FmhaProblem::canonical_key()."""
    body = _body(PROBLEM_HPP, r"std::string canonical_key\s*\(\s*\)\s*const")
    return set(re.findall(r"\b(\w+)\b", body))


class TestFmhaKernelKeyCompleteness(unittest.TestCase):
    def test_compared_fields_are_not_empty(self):
        # Guards the parser itself: a regex that silently matches nothing would
        # make every other assertion below vacuously true.
        self.assertGreater(len(compared_fields()), 10)
        self.assertGreater(len(identifier_fields()), 10)
        self.assertIn("has_sink", compared_fields())
        self.assertIn("has_sink", identifier_fields())

    def test_encode_identifier_emits_every_compared_field(self):
        missing = sorted(compared_fields() - identifier_fields())
        self.assertEqual(
            missing,
            [],
            "encode_identifier() is the registry key; fields compared by "
            "fmha_signature_matches() but not encoded collide: " + ", ".join(missing),
        )

    def test_canonical_key_covers_every_compared_field(self):
        tokens = canonical_key_tokens()
        missing = sorted(
            f for f in compared_fields() if _PROBLEM_ALIASES.get(f, f) not in tokens
        )
        self.assertEqual(
            missing,
            [],
            "canonical_key() documents that it includes ALL fields used by "
            "fmha_signature_matches(); missing: " + ", ".join(missing),
        )

    def test_soft_cap_specifically_is_encoded(self):
        # Named rather than left to the set comparison: a soft-cap and a
        # non-soft-cap kernel that are otherwise identical are a real pair the
        # generator emits, so this is the collision that actually occurs.
        self.assertIn("has_logits_soft_cap", compared_fields())
        self.assertIn("has_logits_soft_cap", identifier_fields())


if __name__ == "__main__":
    unittest.main()

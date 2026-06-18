#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Coverage tests for the direct-convolution config *rule sets*.

Background
----------
On the feature branch the direct-conv kernel instances live inside the
dispatcher JSON config tree
(``codegen/configs/grouped_conv/<variant>/<subset>/nhwgc_{fp16,bf16}.json``,
tagged ``"kind": "direct_conv"``). The upstream ``develop`` branch has replaced
the whole JSON mechanism with Python *rule sets* under
``codegen/grouped_conv/`` and deletes the JSON tree. To survive that merge the
direct-conv instances are being re-expressed as a rule module
(``codegen/grouped_conv/direct_conv_rules.py``).

These tests pin the *behavioural contract* of that conversion: every rule set
must generate **at least** the same set of direct-conv instances as the
original JSON files (additional instances are allowed). The instance identity
used for coverage deliberately ignores the codegen ``id`` (ids are renumbered
deterministically by the rule) and compares the kernel-defining fields only.

The tests are written TDD-style: with no rule module (or an empty rule set)
they fail; they pass once the faithful and generative rule sets are implemented.

Run:
    python3 -m pytest dispatcher/tests/test_direct_conv_rules.py -v
"""

import sys
import json
import unittest
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

CONFIG_ROOT = DISPATCHER_DIR / "codegen" / "configs" / "grouped_conv"

# Rule sets that must fully cover (>=) the JSON direct-conv instance set. Each
# name maps to a get_configs-style callable in grouped_conv.direct_conv_rules
# with the signature
#   get_configs(arch, variants, ndims, datatypes, subset) -> List[DirectConvKernelConfig]
RULE_SETS = ["profiler", "full"]

# Derived rule sets and the base set each must be a subset of.
#   tests       subset of profiler
#   full-tests  subset of full
#   tiny        subset of full
_SUBSET_RELATIONSHIPS = [
    ("tests", "profiler"),
    ("full-tests", "full"),
    ("tiny", "full"),
]

# Directory name (on disk) -> dispatcher variant string.
_VARIANT_DIR_TO_NAME = {
    "forward": "forward",
    "backward_data": "bwd_data",
}


def _instance_key(channel_family, impl, version, variant, datatype, config):
    """Canonical, id-independent identity of a direct-conv instance.

    Two instances are "the same" iff every kernel-defining field matches. The
    codegen ``id`` is intentionally excluded (ids are renumbered by the rule),
    and the config payload (including ``direction``) is compared as an
    order-independent frozenset.
    """
    return (
        channel_family,
        impl,
        version,
        variant,
        datatype,
        frozenset(config.items()),
    )


def _iter_json_files():
    """Yield (variant, subset, datatype, layout, path) for each config JSON."""
    for variant_dir, variant in _VARIANT_DIR_TO_NAME.items():
        for subset in ("profiler", "tests"):
            d = CONFIG_ROOT / variant_dir / subset
            if not d.is_dir():
                continue
            for path in sorted(d.glob("*.json")):
                data = json.loads(path.read_text())
                yield variant, subset, data["datatype"], data["layout"], path


def _json_direct_conv_keys(variant, subset, datatype):
    """Set of canonical keys for the direct-conv instances in the matching JSON
    files (a (variant, subset, datatype) pair may span >1 layout file)."""
    keys = set()
    for v, s, dt, layout, path in _iter_json_files():
        if (v, s, dt) != (variant, subset, datatype):
            continue
        data = json.loads(path.read_text())
        for inst in data.get("instances", []):
            if inst.get("kind") != "direct_conv":
                continue
            keys.add(
                _instance_key(
                    inst["channel_family"],
                    inst["impl"],
                    inst.get("version"),
                    variant,
                    datatype,
                    inst["config"],
                )
            )
    return keys


def _json_coverage_cases():
    """Distinct (variant, subset, datatype) tuples that carry direct-conv
    instances in the JSON tree."""
    cases = set()
    for v, s, dt, layout, path in _iter_json_files():
        data = json.loads(path.read_text())
        if any(i.get("kind") == "direct_conv" for i in data.get("instances", [])):
            cases.add((v, s, dt))
    return sorted(cases)


def _load_rule(rule_set):
    """Return the get_configs callable for a named rule set, or None if the
    rule module / rule set is not implemented yet (test stays red)."""
    try:
        import grouped_conv.direct_conv_rules as rules  # noqa: E402
    except Exception:
        return None
    func = getattr(rules, "get_configs", None)
    if func is None:
        return None

    def call(variants, ndims, datatypes, subset):
        return func(
            arch="gfx950",
            variants=variants,
            ndims=ndims,
            datatypes=datatypes,
            subset=subset,
            rule_set=rule_set,
        )

    return call


def _variant_enum(variant):
    from unified_grouped_conv_codegen import GroupedConvVariant
    return {
        "forward": GroupedConvVariant.FORWARD,
        "bwd_data": GroupedConvVariant.BACKWARD_DATA,
    }[variant]


def _rule_keys(call, variant, subset, datatype):
    """Canonical key set produced by a rule set for one (variant, subset,
    datatype)."""
    configs = call(
        variants=[_variant_enum(variant)],
        ndims=[2],
        datatypes=[datatype],
        subset=subset,
    )
    keys = set()
    for c in configs:
        var_str = {
            "forward": "forward",
            "bwd_data": "bwd_data",
        }[_variant_str(c)]
        keys.add(
            _instance_key(
                c.channel_family, c.impl, c.version, var_str, c.datatype, c.config
            )
        )
    return keys, configs


def _variant_str(config):
    from unified_grouped_conv_codegen import GroupedConvVariant
    return {
        GroupedConvVariant.FORWARD: "forward",
        GroupedConvVariant.BACKWARD_DATA: "bwd_data",
    }[config.variant]


# ---------------------------------------------------------------------------
# Coverage reporting (mirrors develop's tests/test_rules_coverage.py output)
# ---------------------------------------------------------------------------

def _format_key(key):
    """Human-readable one-line summary of a canonical direct-conv key.

    ``key`` is the tuple produced by :func:`_instance_key`:
    (channel_family, impl, version, variant, datatype, frozenset(config)).
    """
    channel_family, impl, version, variant, datatype, config = key
    cfg = dict(config)
    # Render the config payload in a stable, readable order (sans direction,
    # which is already implied by the variant column).
    parts = ", ".join(
        f"{k}={cfg[k]}" for k in sorted(cfg) if k != "direction"
    )
    return (
        f"  [{variant}/{datatype}] {channel_family}c/{impl}/{version} "
        f"{{{parts}}}"
    )


def _print_coverage_report(rule_set, sub_keys, sup_keys, show_missing=20):
    """Print a coverage report mirroring develop's CLI output.

    ``sub_keys`` is the reference (JSON ground-truth) set; ``sup_keys`` the
    rule-set-generated set that should contain it.
    """
    covered = sub_keys & sup_keys
    missing = sub_keys - sup_keys
    extra = sup_keys - sub_keys
    n_ref = len(sub_keys)
    n_covered = len(covered)
    n_missing = len(missing)
    coverage_pct = 100.0 * n_covered / n_ref if n_ref > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"COVERAGE REPORT  [rule_set={rule_set}]")
    print("Reference: 'json'   Generated: rule set")
    print("=" * 70)
    print(f"Reference instances (unique):  {n_ref}")
    print(f"Generated configs (unique):    {len(sup_keys)}")
    print(f"Covered by rules:              {n_covered} ({coverage_pct:.1f}%)")
    print(f"Missing from rules:            {n_missing}")
    print(f"Extra in rules (not in ref):   {len(extra)}")

    if missing:
        limit = show_missing if show_missing > 0 else n_missing
        missing_sorted = sorted(missing, key=str)
        print(f"\n--- Missing instances (showing "
              f"{min(limit, n_missing)} of {n_missing}) ---")
        for key in missing_sorted[:limit]:
            print(_format_key(key))
        if n_missing > limit:
            print(f"  ... and {n_missing - limit} more.")

    # Summary by variant.
    print("\n--- Coverage by variant ---")
    variants = sorted({k[3] for k in sub_keys})
    for var in variants:
        r_keys = {k for k in sub_keys if k[3] == var}
        c_keys = {k for k in covered if k[3] == var}
        m_keys = {k for k in missing if k[3] == var}
        pct = 100.0 * len(c_keys) / len(r_keys) if r_keys else 0.0
        print(f"  {var:15s}: {len(c_keys):4d}/{len(r_keys):4d} covered "
              f"({pct:5.1f}%), {len(m_keys):4d} missing")
    print("=" * 70)

    if n_missing == 0:
        print(f"[PASS] rule set '{rule_set}' fully contains all json instances!")
    else:
        print(f"[FAIL] {n_missing} json instances are not covered by "
              f"'{rule_set}'.")


class TestDirectConvRuleCoverage(unittest.TestCase):
    """Every rule set must cover (>=) the JSON direct-conv instance set."""

    def test_coverage_report(self):
        """Print a develop-style coverage report per rule set (and assert
        full containment of the JSON reference set)."""
        cases = _json_coverage_cases()
        for rule_set in RULE_SETS:
            call = _load_rule(rule_set)
            self.assertIsNotNone(call, f"rule set {rule_set!r} not implemented")
            sub_keys = set()
            sup_keys = set()
            for variant, subset, datatype in cases:
                sub_keys |= _json_direct_conv_keys(variant, subset, datatype)
                rk, _ = _rule_keys(call, variant, subset, datatype)
                sup_keys |= rk
            _print_coverage_report(rule_set, sub_keys, sup_keys)
            self.assertEqual(
                set(), sub_keys - sup_keys,
                f"{rule_set}: {len(sub_keys - sup_keys)} json instance(s) "
                f"not covered",
            )

    def test_json_tree_has_direct_conv_instances(self):
        # Sanity: the source-of-truth JSON actually carries direct-conv
        # instances (guards against the comparison silently passing on empty).
        cases = _json_coverage_cases()
        self.assertTrue(cases, "no direct_conv instances found in JSON tree")

    def test_rule_sets_cover_json(self):
        cases = _json_coverage_cases()
        for rule_set in RULE_SETS:
            call = _load_rule(rule_set)
            for variant, subset, datatype in cases:
                with self.subTest(rule_set=rule_set, variant=variant,
                                  subset=subset, datatype=datatype):
                    self.assertIsNotNone(
                        call,
                        f"rule set {rule_set!r} not implemented",
                    )
                    json_keys = _json_direct_conv_keys(variant, subset, datatype)
                    rule_keys, _ = _rule_keys(call, variant, subset, datatype)
                    missing = json_keys - rule_keys
                    self.assertEqual(
                        set(),
                        missing,
                        f"{rule_set}: {len(missing)} JSON instance(s) not "
                        f"covered for {variant}/{subset}/{datatype}: "
                        f"{sorted(missing)[:3]}",
                    )

    def test_derived_rule_sets_are_subsets(self):
        """tests subset of profiler, full-tests subset of full, tiny subset of full."""
        cases = _json_coverage_cases()
        for sub_set, super_set in _SUBSET_RELATIONSHIPS:
            sub_call = _load_rule(sub_set)
            sup_call = _load_rule(super_set)
            self.assertIsNotNone(sub_call, f"rule set {sub_set!r} not implemented")
            self.assertIsNotNone(sup_call, f"rule set {super_set!r} not implemented")
            sub_keys = set()
            sup_keys = set()
            for variant, subset, datatype in cases:
                sk, _ = _rule_keys(sub_call, variant, subset, datatype)
                pk, _ = _rule_keys(sup_call, variant, subset, datatype)
                sub_keys |= sk
                sup_keys |= pk
            extra = sub_keys - sup_keys
            self.assertEqual(
                set(), extra,
                f"{sub_set} is not a subset of {super_set}: "
                f"{len(extra)} instance(s) not in {super_set}: "
                f"{sorted(extra)[:3]}",
            )
            self.assertTrue(sub_keys, f"{sub_set} produced no instances")

    def test_tiny_has_one_per_channel_family(self):
        """tiny carries exactly one instance per channel family present in full
        (for a single variant/datatype slice)."""
        call = _load_rule("tiny")
        self.assertIsNotNone(call, "rule set 'tiny' not implemented")
        _, configs = _rule_keys(call, "forward", "profiler", "fp16")
        families = [c.channel_family for c in configs]
        self.assertEqual(
            len(families), len(set(families)),
            f"tiny has duplicate channel families: {families}",
        )

    def test_rule_ids_unique_per_variant_dtype_subset(self):
        cases = _json_coverage_cases()
        for rule_set in RULE_SETS:
            call = _load_rule(rule_set)
            for variant, subset, datatype in cases:
                with self.subTest(rule_set=rule_set, variant=variant,
                                  subset=subset, datatype=datatype):
                    self.assertIsNotNone(
                        call, f"rule set {rule_set!r} not implemented"
                    )
                    _, configs = _rule_keys(call, variant, subset, datatype)
                    ids = [c.id for c in configs]
                    dups = [i for i, n in Counter(ids).items() if n > 1]
                    self.assertEqual([], dups, f"{rule_set}: duplicate ids {dups}")


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Contract tests for the direct-convolution config *rule sets*.

Background
----------
The direct-conv kernel instances are expressed as a Python rule module
(``codegen/grouped_conv/direct_conv_rules.py``) that lives alongside develop's
implicit-GEMM rule modules. (Historically these instances lived in a JSON
config tree, ``codegen/configs/grouped_conv/.../nhwgc_{fp16,bf16}.json``; the
develop merge deleted that tree, so the rule module is now the sole source of
truth and these tests pin its self-consistent contract rather than comparing
against the removed JSON.)

The rule module exposes five rule sets through ``get_configs(rule_set=...)``:

  * ``profiler`` / ``full``  -- base instance universes (``full`` is a superset
    of ``profiler``).
  * ``tests``                -- ~20% stratified slice of ``profiler``.
  * ``full-tests``           -- ~20% stratified slice of ``full``.
  * ``tiny``                 -- one instance per channel family from ``full``.

The instance identity used for the containment checks deliberately ignores the
codegen ``id`` (ids are renumbered deterministically by the rule) and compares
the kernel-defining fields only.

Run:
    python3 -m pytest dispatcher/tests/test_direct_conv_rules.py -v
"""

import sys
import unittest
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

# Base rule sets (the full instance universes).
BASE_RULE_SETS = ["profiler", "full"]

# Derived rule sets and the base set each must be a subset of.
#   tests       subset of profiler
#   full-tests  subset of full
#   tiny        subset of full
_SUBSET_RELATIONSHIPS = [
    ("tests", "profiler"),
    ("full-tests", "full"),
    ("tiny", "full"),
]

# Domain swept by the tests (direct-conv supports forward + bwd_data, fp16/bf16).
_VARIANTS = ("forward", "bwd_data")
_DATATYPES = ("fp16", "bf16")


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


def _load_rule(rule_set):
    """Return the get_configs callable for a named rule set, or None if the
    rule module / rule set is not importable."""
    try:
        import grouped_conv.direct_conv_rules as rules  # noqa: E402
    except Exception:
        return None
    func = getattr(rules, "get_configs", None)
    if func is None:
        return None

    def call(variants, ndims, datatypes):
        return func(
            arch="gfx950",
            variants=variants,
            ndims=ndims,
            datatypes=datatypes,
            rule_set=rule_set,
        )

    return call


def _variant_enum(variant):
    from unified_grouped_conv_codegen import GroupedConvVariant
    return {
        "forward": GroupedConvVariant.FORWARD,
        "bwd_data": GroupedConvVariant.BACKWARD_DATA,
    }[variant]


def _variant_str(config):
    from unified_grouped_conv_codegen import GroupedConvVariant
    return {
        GroupedConvVariant.FORWARD: "forward",
        GroupedConvVariant.BACKWARD_DATA: "bwd_data",
    }[config.variant]


def _rule_keys(call, variant, datatype):
    """Canonical key set produced by a rule set for one (variant, datatype)."""
    configs = call(
        variants=[_variant_enum(variant)],
        ndims=[2],
        datatypes=[datatype],
    )
    keys = set()
    for c in configs:
        keys.add(
            _instance_key(
                c.channel_family, c.impl, c.version,
                _variant_str(c), c.datatype, c.config,
            )
        )
    return keys, configs


def _all_keys(call):
    """Aggregate the canonical key set over the whole swept domain."""
    keys = set()
    for variant in _VARIANTS:
        for datatype in _DATATYPES:
            k, _ = _rule_keys(call, variant, datatype)
            keys |= k
    return keys


class TestDirectConvRules(unittest.TestCase):
    """Self-consistency contract of the direct-conv rule module."""

    def test_base_rule_sets_produce_instances(self):
        for rule_set in BASE_RULE_SETS:
            call = _load_rule(rule_set)
            self.assertIsNotNone(call, f"rule set {rule_set!r} not implemented")
            self.assertTrue(
                _all_keys(call), f"{rule_set}: produced no instances"
            )

    def test_full_contains_profiler(self):
        """The 'full' set must be a superset of the 'profiler' set."""
        profiler = _load_rule("profiler")
        full = _load_rule("full")
        self.assertIsNotNone(profiler)
        self.assertIsNotNone(full)
        missing = _all_keys(profiler) - _all_keys(full)
        self.assertEqual(
            set(), missing,
            f"full is missing {len(missing)} profiler instance(s): "
            f"{sorted(missing)[:3]}",
        )

    def test_derived_rule_sets_are_subsets(self):
        """tests subset of profiler, full-tests subset of full, tiny subset of full."""
        for sub_set, super_set in _SUBSET_RELATIONSHIPS:
            sub_call = _load_rule(sub_set)
            sup_call = _load_rule(super_set)
            self.assertIsNotNone(sub_call, f"rule set {sub_set!r} not implemented")
            self.assertIsNotNone(sup_call, f"rule set {super_set!r} not implemented")
            sub_keys = _all_keys(sub_call)
            sup_keys = _all_keys(sup_call)
            extra = sub_keys - sup_keys
            self.assertEqual(
                set(), extra,
                f"{sub_set} is not a subset of {super_set}: "
                f"{len(extra)} instance(s) not in {super_set}: "
                f"{sorted(extra)[:3]}",
            )
            self.assertTrue(sub_keys, f"{sub_set} produced no instances")

    def test_tiny_has_one_per_channel_family(self):
        """tiny carries exactly one instance per channel family (for a single
        variant/datatype slice)."""
        call = _load_rule("tiny")
        self.assertIsNotNone(call, "rule set 'tiny' not implemented")
        _, configs = _rule_keys(call, "forward", "fp16")
        families = [c.channel_family for c in configs]
        self.assertEqual(
            len(families), len(set(families)),
            f"tiny has duplicate channel families: {families}",
        )

    def test_rule_ids_unique_per_variant_dtype(self):
        for rule_set in ["profiler", "full", "tests", "full-tests", "tiny"]:
            call = _load_rule(rule_set)
            self.assertIsNotNone(call, f"rule set {rule_set!r} not implemented")
            for variant in _VARIANTS:
                for datatype in _DATATYPES:
                    with self.subTest(rule_set=rule_set, variant=variant,
                                      datatype=datatype):
                        _, configs = _rule_keys(call, variant, datatype)
                        ids = [c.id for c in configs]
                        dups = [i for i, n in Counter(ids).items() if n > 1]
                        self.assertEqual(
                            [], dups, f"{rule_set}: duplicate ids {dups}"
                        )


if __name__ == "__main__":
    unittest.main()

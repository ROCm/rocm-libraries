#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests verifying the subset/coverage relationships between rule sets.

The rule sets form a containment hierarchy. These tests assert that hierarchy
holds, comparing the canonical kernel-instance keys produced by each rule set
via ``get_default_configs``:

  - "full" contains all instances from "profiler" (the CK Builder profiler set)
  - "full" contains all instances from "tests"    (the CK Builder tests set)
  - "full-tests" is a subset of "full"
  - "tiny"       is a subset of "full-tests"

Each relationship is checked for every supported architecture, across all
variants (forward / bwd_data / bwd_weight), 2D + 3D, and every datatype
(fp16 / bf16 / fp32). Both GEMM and depthwise instances are included.

Run:
    python3 -m pytest dispatcher/tests/test_rules_coverage.py -v
or:
    cd projects/composablekernel/dispatcher
    python3 -m pytest tests/test_rules_coverage.py -v
"""

import sys
import unittest
from pathlib import Path
from typing import Dict, FrozenSet, List, Set, Tuple

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

from unified_grouped_conv_codegen import (                                         # noqa: E402
    DepthwiseConvKernelConfig,
    GroupedConvKernelConfig,
    GroupedConvVariant,
    get_default_configs,
)

# ---------------------------------------------------------------------------
# Generation parameters — cover the full instance space so the subset checks
# are exhaustive (every arch / variant / ndim / datatype).
# ---------------------------------------------------------------------------

ARCHS: List[str] = ["gfx942", "gfx950"]

# GroupedConvVariant.FORWARD_DEPTHWISE is intentionally omitted: depthwise
# instances follow a separate generation/validation path (the depthwise rule
# set and test_depthwise_tile_math.py), not the XDL GEMM variant cross-product
# whose coverage this test verifies.
VARIANTS = [
    GroupedConvVariant.FORWARD,
    GroupedConvVariant.BACKWARD_DATA,
    GroupedConvVariant.BACKWARD_WEIGHT,
]

NDIMS: List[int] = [2, 3]

DATATYPES: List[str] = ["fp16", "bf16", "fp32"]


# ---------------------------------------------------------------------------
# Canonical key generation
# ---------------------------------------------------------------------------

def _config_to_key(cfg) -> FrozenSet:
    """Convert a kernel config to a canonical, hashable key.

    The key captures everything that distinguishes one emitted kernel instance
    from another (tile/warp/vector shape, traits, datatype, ndim, variant), so
    that set containment corresponds to "generates the same instances".
    """
    if isinstance(cfg, DepthwiseConvKernelConfig):
        # Depthwise configs have a distinct shape; key off the unique kernel
        # name (which encodes all tile/stride/pad/vec parameters) + datatype.
        return frozenset({
            ("kind",         "depthwise"),
            ("datatype",     cfg.datatype),
            ("ndim_spatial", cfg.ndim_spatial),
            ("name",         cfg.name(cfg.datatype)),
        })

    t = cfg.tile
    tr = cfg.trait
    sk = tr.streamk_config

    return frozenset({
        ("kind",                "gemm"),
        ("datatype",            cfg.datatype),
        ("variant",             str(cfg.variant)),
        ("ndim_spatial",        cfg.ndim_spatial),
        ("tile_m",              t.tile_m),
        ("tile_n",              t.tile_n),
        ("tile_k",              t.tile_k),
        ("warp_m",              t.warp_m),
        ("warp_n",              t.warp_n),
        ("warp_k",              t.warp_k),
        ("warp_tile_m",         t.warp_tile_m),
        ("warp_tile_n",         t.warp_tile_n),
        ("warp_tile_k",         t.warp_tile_k),
        ("pipeline",            tr.pipeline),
        ("scheduler",           tr.scheduler),
        ("vec_a",               cfg.vector_size_a),
        ("vec_b",               cfg.vector_size_b),
        ("vec_c",               cfg.vector_size_c),
        ("double_smem_buffer",  tr.double_smem_buffer),
        ("two_stage",           tr.two_stage),
        ("explicit_gemm",       tr.explicit_gemm),
        ("split_image",         tr.split_image),
        ("num_groups_to_merge", tr.num_groups_to_merge),
        ("specialization",      tr.specialization or "default"),
        ("streamk_enabled",     sk.streamk_enabled),
        ("streamk_persistent",  sk.streamk_persistent if sk.streamk_enabled else False),
    })


# Cache generated key sets keyed by (rule_set, arch) — generation is the
# expensive part and every test reuses the same sets.
_KEY_CACHE: Dict[Tuple[str, str], Set[FrozenSet]] = {}


def _rule_set_keys(rule_set: str, arch: str) -> Set[FrozenSet]:
    """Generate the canonical key set for ``rule_set`` on ``arch``."""
    cache_key = (rule_set, arch)
    if cache_key not in _KEY_CACHE:
        cfgs = get_default_configs(
            arch=arch,
            variants=VARIANTS,
            ndims=NDIMS,
            datatypes=DATATYPES,
            rule_set=rule_set,
        )
        _KEY_CACHE[cache_key] = {_config_to_key(c) for c in cfgs}
    return _KEY_CACHE[cache_key]


# Short, stable variant labels for the per-variant coverage breakdown.
_VARIANT_LABELS: Dict[str, str] = {
    str(GroupedConvVariant.FORWARD):         "forward",
    str(GroupedConvVariant.BACKWARD_DATA):   "bwd_data",
    str(GroupedConvVariant.BACKWARD_WEIGHT): "bwd_weight",
}


def _variant_label(key: FrozenSet) -> str:
    """Return the short variant label for a key ('depthwise' for depthwise)."""
    d = dict(key)
    if d.get("kind") == "depthwise":
        return "depthwise"
    return _VARIANT_LABELS.get(d.get("variant"), str(d.get("variant")))


def _print_coverage_report(
    arch: str,
    sub_name: str,
    sup_name: str,
    sub_keys: Set[FrozenSet],
    sup_keys: Set[FrozenSet],
    covered: Set[FrozenSet],
    missing: Set[FrozenSet],
    extra: Set[FrozenSet],
    show_missing: int = 20,
) -> None:
    """Print a coverage report mirroring the original CLI output.

    ``sub_name`` is the reference (ground-truth) set, ``sup_name`` the generated
    set that should contain it.
    """
    n_ref = len(sub_keys)
    n_covered = len(covered)
    n_missing = len(missing)
    n_extra = len(extra)
    coverage_pct = 100.0 * n_covered / n_ref if n_ref > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"COVERAGE REPORT  [arch={arch}]")
    print(f"Reference: '{sub_name}'   Generated: '{sup_name}'")
    print("=" * 70)
    print(f"Reference instances (unique):  {n_ref}")
    print(f"Generated configs (unique):    {len(sup_keys)}")
    print(f"Covered by rules:              {n_covered} ({coverage_pct:.1f}%)")
    print(f"Missing from rules:            {n_missing}")
    print(f"Extra in rules (not in ref):   {n_extra}")

    if missing:
        limit = show_missing if show_missing > 0 else n_missing
        missing_sorted = sorted(missing, key=str)
        print(f"\n--- Missing instances (showing {min(limit, n_missing)} of {n_missing}) ---")
        for key in missing_sorted[:limit]:
            print(_format_key(key))
        if n_missing > limit:
            print(f"  ... and {n_missing - limit} more.")

    # Summary by variant.
    print("\n--- Coverage by variant ---")
    variants = sorted({_variant_label(k) for k in sub_keys})
    for var in variants:
        r_keys = {k for k in sub_keys if _variant_label(k) == var}
        c_keys = {k for k in covered if _variant_label(k) == var}
        m_keys = {k for k in missing if _variant_label(k) == var}
        pct = 100.0 * len(c_keys) / len(r_keys) if r_keys else 0.0
        print(f"  {var:15s}: {len(c_keys):4d}/{len(r_keys):4d} covered "
              f"({pct:5.1f}%), {len(m_keys):4d} missing")
    print("=" * 70)

    if n_missing == 0:
        print(f"[PASS] '{sup_name}' fully contains all '{sub_name}' instances!")
    else:
        print(f"[FAIL] {n_missing} '{sub_name}' instances are not covered by '{sup_name}'.")


def _format_key(key: FrozenSet) -> str:
    """Human-readable one-line summary of a canonical key (for failures)."""
    d = dict(key)
    if d.get("kind") == "depthwise":
        return f"  [depthwise/{d.get('datatype')}/{d.get('ndim_spatial')}d] {d.get('name')}"
    tile = f"({d.get('tile_m')},{d.get('tile_n')},{d.get('tile_k')})"
    wave = f"({d.get('warp_m')},{d.get('warp_n')},{d.get('warp_k')})"
    warp = f"({d.get('warp_tile_m')},{d.get('warp_tile_n')},{d.get('warp_tile_k')})"
    vec = f"({d.get('vec_a')},{d.get('vec_b')},{d.get('vec_c')})"
    return (
        f"  [{d.get('variant')}/{d.get('ndim_spatial')}d/{d.get('datatype')}] "
        f"tile={tile} wave={wave} warp={warp} "
        f"pipe={d.get('pipeline')}/{d.get('scheduler')} vec={vec} "
        f"spec={d.get('specialization')}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class RuleSetCoverageTest(unittest.TestCase):
    """Assert the rule-set containment hierarchy holds on every arch."""

    def assert_subset(self, sub_name: str, sup_name: str, arch: str) -> None:
        """Assert every instance of ``sub_name`` appears in ``sup_name``.

        Prints a coverage report (mirroring the original
        ``validate_rules_coverage.py`` CLI output) treating ``sub_name`` as the
        reference set and ``sup_name`` as the generated set.
        """
        sub_keys = _rule_set_keys(sub_name, arch)
        sup_keys = _rule_set_keys(sup_name, arch)

        self.assertGreater(
            len(sub_keys), 0,
            f"[{arch}] rule set '{sub_name}' produced no instances",
        )

        covered = sub_keys & sup_keys
        missing = sub_keys - sup_keys
        extra = sup_keys - sub_keys

        _print_coverage_report(arch, sub_name, sup_name, sub_keys, sup_keys,
                               covered, missing, extra)

        if missing:
            preview = "\n".join(_format_key(k) for k in sorted(missing, key=str)[:20])
            more = f"\n  ... and {len(missing) - 20} more." if len(missing) > 20 else ""
            self.fail(
                f"[{arch}] '{sup_name}' is missing {len(missing)} of "
                f"{len(sub_keys)} instances from '{sub_name}':\n{preview}{more}"
            )


# Containment relationships to verify: (test label, sub set, super set).
# ``sub`` must be fully contained in ``super``.
_RELATIONSHIPS: List[Tuple[str, str, str]] = [
    ("full_contains_profiler",       "profiler",   "full"),
    ("full_contains_tests",          "tests",      "full"),
    ("full_tests_is_subset_of_full", "full-tests", "full"),
    ("tiny_is_subset_of_full_tests", "tiny",       "full-tests"),
]


def _make_subset_test(sub_name: str, sup_name: str, arch: str):
    def test(self: RuleSetCoverageTest) -> None:
        self.assert_subset(sub_name, sup_name, arch)

    test.__doc__ = f"[{arch}] '{sup_name}' must contain every '{sub_name}' instance."
    return test


# Generate one test method per (relationship, arch) so the architecture shows
# up directly in the test id (e.g. ``test_full_contains_profiler_gfx942``),
# making it visible under ``pytest -v`` without needing ``-s``.
for _label, _sub, _sup in _RELATIONSHIPS:
    for _arch in ARCHS:
        setattr(
            RuleSetCoverageTest,
            f"test_{_label}_{_arch}",
            _make_subset_test(_sub, _sup, _arch),
        )

del _label, _sub, _sup, _arch


if __name__ == "__main__":
    unittest.main()

"""The gfx950 dense attention bundle cases and the shipped catalog name the same head
configurations.

The shared SdpaFwd sweeps vary one axis at a time off a baseline cell, and the cases
claimed for the engine are meant to reach every (head size, Hq, Hkv) the catalog
compiles. Nothing at run time enforces that: a configuration added to the catalog simply
has no end-to-end bundle, and the knob suite's own catalog check runs only on gfx950
hardware. A claimed case whose configuration the catalog lacks is the opposite drift,
and surfaces only as a broken support claim on a gfx950 runner. This compares the two
host-side, from the shipped descriptor and the sweep JSON, so every runner that builds
the packaging tests sees both. Only cases support.json claims for the engine on gfx950
count: the sweeps are shared, and another engine's cases are not this catalog's to cover.
"""

import pytest

from bundle_coverage import (
    SWEEPS,
    catalog_head_configs,
    catalog_path,
    claimed_cases,
    load_catalog,
)

pytestmark = pytest.mark.quick

ARCH = "gfx950"
CATALOG = catalog_path("gfx950_attention_dense", "gfx950_attention_dense.kdp.json")
ENGINE = "hipkernel:Gfx950AttentionDense"


def sweep_head_configs(sweep_dir):
    """(head_size, Hq, Hkv) of every claimed case, from its Q and K dims ([B, H, S, D])."""
    return {(q[3], q[1], k[1]) for _, q, k in claimed_cases(sweep_dir, ENGINE, ARCH)}


def test_the_sweeps_reach_exactly_the_catalogs_head_configurations():
    catalog = catalog_head_configs(load_catalog(CATALOG, ARCH))
    # An unreadable or emptied catalog would otherwise pass as "nothing missing".
    assert catalog, f"no kernels read from {CATALOG}"
    swept = set().union(*(sweep_head_configs(d) for d in SWEEPS))
    unreached, unknown = catalog - swept, swept - catalog
    assert (
        not unreached
    ), f"catalog head configurations with no bundle case: {sorted(unreached)}"
    assert (
        not unknown
    ), f"bundle cases on head configurations the catalog lacks: {sorted(unknown)}"

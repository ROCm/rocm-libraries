"""Every case the gfx950 unified-attention 2D engine's support claims name is a head
configuration its shipped catalog compiles.

The shared SdpaFwd sweeps are claimed per engine. A claimed case whose (head size, Hq, Hkv)
the catalog lacks surfaces only as a broken support claim on a gfx950 runner; this compares
the two host-side, from the shipped descriptor and the sweep JSON. Unlike the dense
catalog, this one is wider than what the sweeps reach (the claims are the cases the engine
serves, not a covering set), so only the claimed-subset direction is checked.
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
CATALOG = catalog_path(
    "gfx950_attention_unified_2d", "gfx950_attention_unified_2d.kdp.json"
)
ENGINE = "hipkernel:Gfx950AttentionUnified2D"


def sweep_head_configs(sweep_dir):
    """(head_size, Hq, Hkv) of every claimed case, from its Q and K dims ([B, H, S, D])."""
    return {(q[3], q[1], k[1]) for _, q, k in claimed_cases(sweep_dir, ENGINE, ARCH)}


def test_every_claimed_case_is_a_head_configuration_the_catalog_compiles():
    catalog = catalog_head_configs(load_catalog(CATALOG, ARCH))
    # An unreadable or emptied catalog would otherwise pass as "nothing unknown".
    assert catalog, f"no kernels read from {CATALOG}"
    swept = set().union(*(sweep_head_configs(d) for d in SWEEPS))
    assert swept, "support.json claims no case for this engine"
    assert (
        not swept - catalog
    ), f"claimed head configurations the catalog lacks: {sorted(swept - catalog)}"

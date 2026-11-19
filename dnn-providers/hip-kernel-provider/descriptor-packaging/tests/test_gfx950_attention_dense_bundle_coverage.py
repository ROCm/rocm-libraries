"""The gfx950 dense attention bundle cases and the shipped catalog name the same head
sizes, and the catalog keys on head size rather than on head counts.

The shared SdpaFwd sweeps vary one axis at a time off a baseline cell. Before the catalog
collapse each (head size, Hq, Hkv) was its own entry, so the cases had to reach every one
of them. One kernel now serves every head configuration, because the kernel reads both
head counts as kernargs from the graph, so head counts are no longer a coverage axis and
the only configuration axis left for the bundles to reach is the head size.

Nothing at run time enforces either property: a head size added to the catalog simply has
no end-to-end bundle, and the knob suite's own catalog check runs only on gfx950 hardware.
A claimed case whose head size the catalog lacks is the opposite drift, and surfaces only
as a broken support claim on a gfx950 runner. This compares the two host-side, from the
shipped descriptor and the sweep JSON, so every runner that builds the packaging tests
sees both. Only cases support.json claims for the engine on gfx950 count: the sweeps are
shared, and another engine's cases are not this catalog's to cover.
"""

import pytest

from bundle_coverage import (
    SWEEPS,
    catalog_path,
    claimed_cases,
    load_catalog,
)

pytestmark = pytest.mark.quick

ARCH = "gfx950"
CATALOG = catalog_path("gfx950_attention_dense", "gfx950_attention_dense.kdp.json")
ENGINE = "hipkernel:Gfx950AttentionDense"

# Metadata fields the collapse removed. One kernel now serves every head configuration, so
# a catalog carrying either has gone back to packing one entry per configuration, and this
# file's head-size rule is too weak to catch that on its own.
HEAD_COUNT_FIELDS = ("num_query_heads", "num_kv_heads")


def catalog_head_sizes(kdp):
    """Head sizes the descriptor packs, from the metadata the matcher compares."""
    return {kernel["metadata"]["head_size"] for kernel in kdp["kernelDescriptors"]}


def sweep_head_sizes(sweep_dir):
    """Head size of every claimed case, from its Q dims ([B, H, S, D])."""
    return {q[3] for _, q, _ in claimed_cases(sweep_dir, ENGINE, ARCH)}


def test_the_catalog_keys_on_head_size_not_head_counts():
    kdp = load_catalog(CATALOG, ARCH)
    kernels = kdp["kernelDescriptors"]
    # An unreadable or emptied catalog would otherwise pass as "nothing carries them".
    assert kernels, f"no kernels read from {CATALOG}"
    carriers = {
        field: sum(1 for kernel in kernels if field in kernel["metadata"])
        for field in HEAD_COUNT_FIELDS
    }
    assert not any(carriers.values()), (
        "the catalog's metadata carries head counts, so it is back to one entry per head "
        f"configuration: {carriers}"
    )


def test_the_sweeps_reach_exactly_the_catalogs_head_sizes():
    catalog = catalog_head_sizes(load_catalog(CATALOG, ARCH))
    # An unreadable or emptied catalog would otherwise pass as "nothing missing".
    assert catalog, f"no kernels read from {CATALOG}"
    swept = set().union(*(sweep_head_sizes(d) for d in SWEEPS))
    unreached, unknown = catalog - swept, swept - catalog
    assert not unreached, f"catalog head sizes with no bundle case: {sorted(unreached)}"
    assert (
        not unknown
    ), f"bundle cases on head sizes the catalog lacks: {sorted(unknown)}"

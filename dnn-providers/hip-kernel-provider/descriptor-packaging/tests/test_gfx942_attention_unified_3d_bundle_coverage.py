"""Every bundle case claimed for the gfx942 split-KV decode engine has a shipped kernel.

The shared SdpaFwd sweeps carry decode cases (seqlen_q 1) that support.json claims for
hipkernel:Gfx942AttentionUnified3d on gfx942. The engine serves a graph only through a
kernel whose feature cell -- dtype, head size, GQA ratio Hq / Hkv (KMD
num_queries_per_kv), sliding window 0 -- equals the graph's (the kernels read the head
counts themselves at runtime), so a claimed case whose cell the catalog lacks surfaces only as a broken support
claim on a gfx942 runner. This compares the two host-side, from the shipped descriptor
and the sweep JSON, so every runner that builds the packaging tests sees the drift.

Unlike the gfx950 dense check, the comparison runs one way. The catalog holds one cell
per (dtype, head size, GQA ratio); the sweeps sample them, they are not meant to reach
each one. Only cases support.json claims for the engine on gfx942
count: the sweeps are shared, and another engine's cases are not this catalog's to serve.
"""

import pytest

from bundle_coverage import (
    SWEEPS,
    catalog_path,
    claimed_case_ids,
    claimed_cases,
    load_catalog,
)

pytestmark = pytest.mark.quick

ARCH = "gfx942"
CATALOG = catalog_path(
    "gfx942_attention_unified_3d", "gfx942_attention_unified3d.kdp.json"
)
ENGINE = "hipkernel:Gfx942AttentionUnified3d"

# The graph's io_data_type as the catalog's KMD dtype field spells it.
KMD_DTYPE = {"half": "FP16", "bfloat16": "BF16"}


def catalog_cells(kdp):
    """(dtype, head_size, Hq / Hkv) of every kernel the descriptor packs with no window."""
    cells = set()
    for kernel in kdp["kernelDescriptors"]:
        meta = kernel["metadata"]
        if meta["sliding_window"] != 0:
            continue
        cells.add((meta["dtype"], meta["head_size"], meta["num_queries_per_kv"]))
    return cells


def claimed_cells(sweep_dir):
    """{case id: (dtype, head_size, Hq // Hkv)} of every claimed case, from Q and K dims."""
    cells = {}
    for case, q, k in claimed_cases(sweep_dir, ENGINE, ARCH):
        # Every kernel is a decode kernel; the engine declines seqlen_q > 1 before any
        # cell is compared, so such a claim breaks whatever the catalog holds.
        assert q[2] == 1, f"{sweep_dir}: claimed case {case['id']} has seqlen_q {q[2]}"
        dtype = KMD_DTYPE.get(
            case["values"]["io_data_type"], case["values"]["io_data_type"]
        )
        # graph_match declines Hq not a multiple of Hkv before any cell is compared.
        assert q[1] % k[1] == 0, (
            f"{sweep_dir}: claimed case {case['id']} has Hq {q[1]} not a multiple of "
            f"Hkv {k[1]}"
        )
        cells[case["id"]] = (dtype, q[3], q[1] // k[1])
    # A claim naming no sweep case would otherwise drop out of the comparison unseen.
    missing = claimed_case_ids(sweep_dir, ENGINE, ARCH) - cells.keys()
    assert not missing, f"{sweep_dir}: claims name no sweep case: {sorted(missing)}"
    return cells


def test_every_claimed_case_has_a_shipped_kernel():
    catalog = catalog_cells(load_catalog(CATALOG, ARCH))
    # An unreadable or emptied catalog would otherwise pass as "nothing missing".
    assert catalog, f"no kernels read from {CATALOG}"
    for sweep_dir in SWEEPS:
        cells = claimed_cells(sweep_dir)
        # A tier with no claim would pass as "nothing missing" too.
        assert cells, f"{sweep_dir}: no case claimed for {ENGINE} on gfx942"
        unserved = {case: cell for case, cell in cells.items() if cell not in catalog}
        assert not unserved, (
            f"{sweep_dir}: claimed cases whose (dtype, head size, Hq // Hkv) no shipped "
            f"kernel carries: {unserved}"
        )

"""Every bundle case claimed for the gfx942 split-KV decode engine has a shipped kernel.

The shared SdpaFwd sweeps carry decode cases (seqlen_q 1) that support.json claims for
hipkernel:Gfx942AttentionUnified3d on gfx942. The engine serves a graph only through a
kernel whose feature cell -- dtype, head size, Hq, Hkv, sliding window 0 -- equals the
graph's, so a claimed case whose cell the catalog lacks surfaces only as a broken support
claim on a gfx942 runner. This compares the two host-side, from the shipped descriptor
and the sweep JSON, so every runner that builds the packaging tests sees the drift.

Unlike the gfx950 dense check, the comparison runs one way. The catalog holds 138 cells
(23 head pairs per head size x 3 head sizes x 2 dtypes); the sweeps sample them, they are
not meant to reach each one. Only cases support.json claims for the engine on gfx942
count: the sweeps are shared, and another engine's cases are not this catalog's to serve.
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.quick

REPO = Path(__file__).resolve().parents[4]
CATALOG = (
    REPO
    / "dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors"
    / "rocKE/gfx942_attention_unified_3d/artifacts/gfx942_attention_unified3d.kdp.json"
)
ENGINE = "hipkernel:Gfx942AttentionUnified3d"
SWEEPS = [
    REPO
    / "dnn-providers/integration-tests/integration-test-bundles"
    / tier
    / "SdpaFwd/Default"
    for tier in ("quick", "standard")
]

# The graph's io_data_type as the catalog's KMD dtype field spells it.
KMD_DTYPE = {"half": "FP16", "bfloat16": "BF16"}


def catalog_cells(kdp_path):
    """(dtype, head_size, Hq, Hkv) of every kernel the descriptor packs with no window."""
    kdp = json.loads(kdp_path.read_text(encoding="utf-8"))
    cells = set()
    for kernel in kdp["kernelDescriptors"]:
        meta = kernel["metadata"]
        if meta["sliding_window"] != 0:
            continue
        cells.add(
            (
                meta["dtype"],
                meta["head_size"],
                meta["num_query_heads"],
                meta["num_kv_heads"],
            )
        )
    return cells


def claimed_case_ids(sweep_dir):
    """Ids of the cases support.json claims for ENGINE on gfx942."""
    support = json.loads((sweep_dir / "support.json").read_text(encoding="utf-8"))
    claimed = set()
    for claim in support["claims"].get(ENGINE, []):
        if "gfx942" in claim["support"]:
            claimed.update(claim["cases"])
    return claimed


def claimed_cells(sweep_dir):
    """{case id: (dtype, head_size, Hq, Hkv)} of every claimed case, from Q and K dims."""
    template = json.loads(
        (sweep_dir / "graph.template.json").read_text(encoding="utf-8")
    )
    uid = {tensor["name"]: tensor["uid"] for tensor in template["tensors"]}
    sweep = json.loads((sweep_dir / "sweep.json").read_text(encoding="utf-8"))
    claimed = claimed_case_ids(sweep_dir)
    cells = {}
    for case in sweep["cases"]:
        if case["id"] not in claimed:
            continue
        dims = {tensor["uid"]: tensor["dims"] for tensor in case["values"]["tensors"]}
        q, k = dims[uid["Q"]], dims[uid["K"]]
        # Every kernel is a decode kernel; the engine declines seqlen_q > 1 before any
        # cell is compared, so such a claim breaks whatever the catalog holds.
        assert q[2] == 1, f"{sweep_dir}: claimed case {case['id']} has seqlen_q {q[2]}"
        dtype = KMD_DTYPE.get(
            case["values"]["io_data_type"], case["values"]["io_data_type"]
        )
        cells[case["id"]] = (dtype, q[3], q[1], k[1])
    # A claim naming no sweep case would otherwise drop out of the comparison unseen.
    missing = claimed - cells.keys()
    assert not missing, f"{sweep_dir}: claims name no sweep case: {sorted(missing)}"
    return cells


def test_every_claimed_case_has_a_shipped_kernel():
    # The KDP is a DVC output (descriptors/README.md), so a checkout without a
    # `dvc pull` has no catalog to compare.
    if not CATALOG.exists():
        pytest.skip(
            f"gfx942 catalog not pulled; run: dvc pull -r ingestor {CATALOG.parent}.dvc"
        )
    catalog = catalog_cells(CATALOG)
    # An unreadable or emptied catalog would otherwise pass as "nothing missing".
    assert catalog, f"no kernels read from {CATALOG}"
    for sweep_dir in SWEEPS:
        cells = claimed_cells(sweep_dir)
        # A tier with no claim would pass as "nothing missing" too.
        assert cells, f"{sweep_dir}: no case claimed for {ENGINE} on gfx942"
        unserved = {case: cell for case, cell in cells.items() if cell not in catalog}
        assert not unserved, (
            f"{sweep_dir}: claimed cases whose (dtype, head size, Hq, Hkv) no shipped "
            f"kernel carries: {unserved}"
        )

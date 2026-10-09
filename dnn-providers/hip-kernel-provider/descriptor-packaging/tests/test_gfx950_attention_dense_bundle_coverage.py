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

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.quick

REPO = Path(__file__).resolve().parents[4]
CATALOG = (
    REPO
    / "dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors"
    / "rocKE/gfx950_attention_dense/gfx950_attention_dense.kdp.json"
)
ENGINE = "hipkernel:Gfx950AttentionDense"
SWEEPS = [
    REPO
    / "dnn-providers/integration-tests/integration-test-bundles"
    / tier
    / "SdpaFwd/Default"
    for tier in ("quick", "standard")
]


def catalog_head_configs(kdp_path):
    """(head_size, Hq, Hkv) of every kernel the descriptor packs."""
    kdp = json.loads(kdp_path.read_text(encoding="utf-8"))
    configs = set()
    for kernel in kdp["kernelDescriptors"]:
        spec = kernel["kernel_source"]["spec"]
        configs.add((spec["head_size"], spec["num_query_heads"], spec["num_kv_heads"]))
    return configs


def claimed_case_ids(sweep_dir):
    """Ids of the cases support.json claims for ENGINE on gfx950."""
    support = json.loads((sweep_dir / "support.json").read_text(encoding="utf-8"))
    claimed = set()
    for claim in support["claims"].get(ENGINE, []):
        if "gfx950" in claim["support"]:
            claimed.update(claim["cases"])
    return claimed


def sweep_head_configs(sweep_dir):
    """(head_size, Hq, Hkv) of every claimed case, from its Q and K dims ([B, H, S, D])."""
    template = json.loads(
        (sweep_dir / "graph.template.json").read_text(encoding="utf-8")
    )
    uid = {tensor["name"]: tensor["uid"] for tensor in template["tensors"]}
    sweep = json.loads((sweep_dir / "sweep.json").read_text(encoding="utf-8"))
    claimed = claimed_case_ids(sweep_dir)
    configs = set()
    for case in sweep["cases"]:
        if case["id"] not in claimed:
            continue
        dims = {tensor["uid"]: tensor["dims"] for tensor in case["values"]["tensors"]}
        q, k = dims[uid["Q"]], dims[uid["K"]]
        configs.add((q[3], q[1], k[1]))
    return configs


def head_config_gaps(kdp_path, sweep_dirs):
    """(catalog configurations no case reaches, case configurations the catalog lacks)."""
    catalog = catalog_head_configs(kdp_path)
    swept = set().union(*(sweep_head_configs(d) for d in sweep_dirs))
    return catalog - swept, swept - catalog


def test_the_sweeps_reach_exactly_the_catalogs_head_configurations():
    # An unreadable or emptied catalog would otherwise pass as "nothing missing".
    assert catalog_head_configs(CATALOG), f"no kernels read from {CATALOG}"
    unreached, unknown = head_config_gaps(CATALOG, SWEEPS)
    assert (
        not unreached
    ), f"catalog head configurations with no bundle case: {sorted(unreached)}"
    assert (
        not unknown
    ), f"bundle cases on head configurations the catalog lacks: {sorted(unknown)}"

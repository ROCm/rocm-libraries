"""Every case the gfx950 unified-attention 3D (split-KV decode) engine's support claims name
is a single-token decode case of a head configuration its shipped catalog compiles.

The shared SdpaFwd sweeps are claimed per engine. A claimed case whose (head size, Hq, Hkv)
the catalog lacks surfaces only as a broken support claim on a gfx950 runner; this compares
the two host-side, from the shipped descriptor and the sweep JSON. Unlike the dense
catalog, this one is wider than what the sweeps reach (the claims are the cases the engine
serves, not a covering set), so only the claimed-subset direction is checked.
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.quick

REPO = Path(__file__).resolve().parents[4]
CATALOG = (
    REPO
    / "dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors"
    / "rocKE/gfx950_attention_unified_3d/artifacts/gfx950_attention_unified_3d.kdp.json"
)
ENGINE = "hipkernel:Gfx950AttentionUnified3D"
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
        # A multi-launch kernel carries `sources` (segment + reduce) in place of
        # `kernel_source`; the segment launch's spec is the kernel's.
        source = kernel.get("kernel_source") or kernel["sources"]["segment"]
        spec = source["spec"]
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
        # Every shipped kernel is a decode-grid kernel; a claimed longer query is a claim
        # the engine would decline.
        assert q[2] == 1, f"claimed case {case['id']} has seqlen_q {q[2]}, not 1"
        configs.add((q[3], q[1], k[1]))
    return configs


def test_every_claimed_case_is_a_head_configuration_the_catalog_compiles():
    # The KDP is a DVC output (descriptors/README.md), so a checkout without a
    # `dvc pull` has no catalog to compare.
    if not CATALOG.exists():
        pytest.skip(
            f"gfx950 catalog not pulled; run: dvc pull -r ingestor {CATALOG.parent}.dvc"
        )
    # An unreadable or emptied catalog would otherwise pass as "nothing unknown".
    catalog = catalog_head_configs(CATALOG)
    assert catalog, f"no kernels read from {CATALOG}"
    swept = set().union(*(sweep_head_configs(d) for d in SWEEPS))
    assert swept, "support.json claims no case for this engine"
    assert (
        not swept - catalog
    ), f"claimed head configurations the catalog lacks: {sorted(swept - catalog)}"

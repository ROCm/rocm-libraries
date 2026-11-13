"""Helpers shared by the `test_*_bundle_coverage.py` tests.

Each of those tests compares one shipped rocKE catalog (a DVC-tracked KDP) with the
cases the shared SdpaFwd sweeps' support.json claims for its engine. A plain module,
like `pack_helpers.py`, so that no test module imports from another.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
DESCRIPTORS = (
    REPO
    / "dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors"
)
SWEEPS = [
    REPO
    / "dnn-providers/integration-tests/integration-test-bundles"
    / tier
    / "SdpaFwd/Default"
    for tier in ("quick", "standard")
]


def catalog_path(bundle, kdp_name):
    """The shipped KDP `descriptors/rocKE/<bundle>/artifacts/<kdp_name>`."""
    return DESCRIPTORS / "rocKE" / bundle / "artifacts" / kdp_name


def load_catalog(kdp_path, arch):
    """The parsed KDP; skips when it is not pulled.

    The KDP is a DVC output (descriptors/README.md), so a checkout without a
    `dvc pull` has no catalog to compare.
    """
    if not kdp_path.exists():
        pytest.skip(
            f"{arch} catalog not pulled; run: dvc pull -r ingestor {kdp_path.parent}.dvc"
        )
    return json.loads(kdp_path.read_text(encoding="utf-8"))


def catalog_head_configs(kdp):
    """(head_size, Hq, Hkv) of every kernel the descriptor packs."""
    configs = set()
    for kernel in kdp["kernelDescriptors"]:
        # A multi-launch kernel carries `sources` (segment + reduce) in place of
        # `kernel_source`; the segment launch's spec is the kernel's.
        source = kernel.get("kernel_source") or kernel["sources"]["segment"]
        spec = source["spec"]
        configs.add((spec["head_size"], spec["num_query_heads"], spec["num_kv_heads"]))
    return configs


def claimed_case_ids(sweep_dir, engine, arch):
    """Ids of the cases support.json claims for `engine` on `arch`."""
    support = json.loads((sweep_dir / "support.json").read_text(encoding="utf-8"))
    claimed = set()
    for claim in support["claims"].get(engine, []):
        if arch in claim["support"]:
            claimed.update(claim["cases"])
    return claimed


def claimed_cases(sweep_dir, engine, arch):
    """`(case, q_dims, k_dims)` of every case claimed for `engine` on `arch`.

    Q and K dims are [B, H, S, D].
    """
    template = json.loads(
        (sweep_dir / "graph.template.json").read_text(encoding="utf-8")
    )
    uid = {tensor["name"]: tensor["uid"] for tensor in template["tensors"]}
    sweep = json.loads((sweep_dir / "sweep.json").read_text(encoding="utf-8"))
    claimed = claimed_case_ids(sweep_dir, engine, arch)
    for case in sweep["cases"]:
        if case["id"] not in claimed:
            continue
        dims = {tensor["uid"]: tensor["dims"] for tensor in case["values"]["tensors"]}
        yield case, dims[uid["Q"]], dims[uid["K"]]

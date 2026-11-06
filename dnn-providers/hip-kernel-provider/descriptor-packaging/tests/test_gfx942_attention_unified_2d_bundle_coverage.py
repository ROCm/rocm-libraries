"""Every case the gfx942 unified-attention 2D engine's support claims name is a cell its
shipped catalog compiles.

The shared SdpaFwd sweeps are claimed per engine. A claimed case whose (dtype, head size,
GQA ratio) the catalog lacks, or whose mask the gfx942 kernel does not compute, surfaces
only as a broken support claim on a gfx942 runner; this compares the two host-side, from
the shipped descriptor and the sweep JSON. The kernels read the head counts at run time and
bake only their ratio, so a cell is keyed by the ratio, not the head pair. The catalog is
wider than what the sweeps reach (the claims are the cases the engine serves, not a
covering set), so only the claimed-subset direction is checked.
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.quick

REPO = Path(__file__).resolve().parents[4]
CATALOG = (
    REPO
    / "dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors"
    / "rocKE/gfx942_attention_unified_2d/artifacts/gfx942_attention_unified_2d.kdp.json"
)
ENGINE = "hipkernel:Gfx942AttentionUnified2D"
SWEEPS = [
    REPO
    / "dnn-providers/integration-tests/integration-test-bundles"
    / tier
    / "SdpaFwd/Default"
    for tier in ("quick", "standard")
]
# The sweep JSON spells dtypes as hipDNN does; the catalog spec as rocKE does.
SWEEP_DTYPES = {"bfloat16": "bf16", "half": "fp16"}


def catalog_cells(kdp_path):
    """(dtype, head_size, GQA ratio) of every kernel the descriptor packs."""
    kdp = json.loads(kdp_path.read_text(encoding="utf-8"))
    cells = set()
    for kernel in kdp["kernelDescriptors"]:
        spec = kernel["kernel_source"]["spec"]
        cells.add(
            (
                spec["dtype"],
                spec["head_size"],
                spec["num_query_heads"] // spec["num_kv_heads"],
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


def claimed_cases(sweep_dir):
    """(id, dtype, head_size, Hq, Hkv, attributes) of every claimed case."""
    template = json.loads(
        (sweep_dir / "graph.template.json").read_text(encoding="utf-8")
    )
    uid = {tensor["name"]: tensor["uid"] for tensor in template["tensors"]}
    sweep = json.loads((sweep_dir / "sweep.json").read_text(encoding="utf-8"))
    claimed = claimed_case_ids(sweep_dir)
    cases = []
    for case in sweep["cases"]:
        if case["id"] not in claimed:
            continue
        values = case["values"]
        dims = {tensor["uid"]: tensor["dims"] for tensor in values["tensors"]}
        q, k = dims[uid["Q"]], dims[uid["K"]]
        cases.append(
            (
                case["id"],
                SWEEP_DTYPES.get(values["io_data_type"], values["io_data_type"]),
                q[3],
                q[1],
                k[1],
                values.get("attributes", {}),
            )
        )
    return cases


def all_claimed_cases():
    cases = [case for sweep_dir in SWEEPS for case in claimed_cases(sweep_dir)]
    assert cases, "support.json claims no case for this engine"
    return cases


def test_every_claimed_case_is_a_cell_the_catalog_compiles():
    # The KDP is a DVC output (descriptors/README.md), so a checkout without a
    # `dvc pull` has no catalog to compare.
    if not CATALOG.exists():
        pytest.skip(
            f"gfx942 catalog not pulled; run: dvc pull -r ingestor {CATALOG.parent}.dvc"
        )
    # An unreadable or emptied catalog would otherwise pass as "nothing unknown".
    catalog = catalog_cells(CATALOG)
    assert catalog, f"no kernels read from {CATALOG}"
    missing = sorted(
        (case_id, (dtype, head_size, hq // hkv))
        for case_id, dtype, head_size, hq, hkv, _ in all_claimed_cases()
        if hkv <= 0 or hq % hkv or (dtype, head_size, hq // hkv) not in catalog
    )
    assert not missing, f"claimed cases whose cell the catalog lacks: {missing}"


def test_every_claimed_case_is_bottom_right_causal():
    # The gfx942 kernel computes only the bottom-right causal band; a claimed case with
    # another mask is one the engine declines, so its lane would fail.
    wrong = sorted(
        case_id
        for case_id, *_, attributes in all_claimed_cases()
        if attributes.get("attributes__diagonal_alignment") != "BOTTOM_RIGHT"
        or attributes.get("attributes__right_bound") != 0
        or attributes.get("attributes__causal_mask")
        or attributes.get("attributes__left_bound") not in (None, -1)
    )
    assert not wrong, f"claimed cases that are not bottom-right causal: {wrong}"

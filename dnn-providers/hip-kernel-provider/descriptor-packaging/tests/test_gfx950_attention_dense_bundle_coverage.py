"""The gfx950 dense attention bundle cases and the shipped catalog agree on head sizes,
and the catalog keys on nothing finer than that.

The kernel reads num_query_heads / num_kv_heads as kernargs, so one catalog entry per
(dtype, head size, mask, tile) serves every head configuration and the metadata carries
no head counts. What the catalog still specializes on is the head size, so that is what
the claimed bundle cases must line up with: a head size the catalog compiles but no case
reaches has no end-to-end bundle, and a claimed case whose head size the catalog lacks
surfaces only as a broken support claim on a gfx950 runner. Both are checked host-side
here, from the shipped descriptor and the sweep JSON, so every runner that builds the
packaging tests sees them -- the knob suite's equivalent needs gfx950 hardware. Only
cases support.json claims for the engine on gfx950 count: the sweeps are shared, and
another engine's cases are not this catalog's to cover.
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.quick

REPO = Path(__file__).resolve().parents[4]
CATALOG = (
    REPO
    / "dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors"
    / "rocKE/gfx950_attention_dense/artifacts/gfx950_attention_dense.kdp.json"
)
ENGINE = "hipkernel:Gfx950AttentionDense"
SWEEPS = [
    REPO
    / "dnn-providers/integration-tests/integration-test-bundles"
    / tier
    / "SdpaFwd/Default"
    for tier in ("quick", "standard")
]

# Metadata fields the collapse removed. A catalog carrying either is matched on it again,
# so one entry per head configuration comes back and this file's head-size rule is too
# weak to catch it.
HEAD_COUNT_FIELDS = ("num_query_heads", "num_kv_heads")


def catalog_kernels(kdp_path):
    return json.loads(kdp_path.read_text(encoding="utf-8"))["kernelDescriptors"]


def catalog_head_sizes(kdp_path):
    """Head sizes the descriptor packs, from the metadata the matcher compares."""
    return {kernel["metadata"]["head_size"] for kernel in catalog_kernels(kdp_path)}


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


def swept_head_configs(sweep_dirs):
    return set().union(*(sweep_head_configs(d) for d in sweep_dirs))


def head_size_gaps(kdp_path, sweep_dirs):
    """(catalog head sizes no case reaches, case head sizes the catalog lacks)."""
    catalog = catalog_head_sizes(kdp_path)
    swept = {head_size for head_size, _, _ in swept_head_configs(sweep_dirs)}
    return catalog - swept, swept - catalog


def _require_catalog():
    # The KDP is a DVC output (descriptors/README.md), so a checkout without a
    # `dvc pull` has no catalog to compare.
    if not CATALOG.exists():
        pytest.skip(
            f"gfx950 catalog not pulled; run: dvc pull -r ingestor {CATALOG.parent}.dvc"
        )


def test_the_catalog_keys_on_head_size_not_head_counts():
    _require_catalog()
    kernels = catalog_kernels(CATALOG)
    # An unreadable or emptied catalog would otherwise pass as "nothing carries them".
    assert kernels, f"no kernels read from {CATALOG}"
    carriers = {
        field: sum(1 for kernel in kernels if field in kernel["metadata"])
        for field in HEAD_COUNT_FIELDS
    }
    assert not any(carriers.values()), (
        f"catalog metadata still carries head counts {carriers}; the kernel takes them "
        "as kernargs, so one entry per (dtype, head size, mask, tile) serves them all"
    )
    assert catalog_head_sizes(CATALOG), "no kernel carries an integer head_size"


def test_the_sweeps_reach_exactly_the_catalogs_head_sizes():
    _require_catalog()
    assert catalog_head_sizes(CATALOG), f"no kernels read from {CATALOG}"
    unreached, unknown = head_size_gaps(CATALOG, SWEEPS)
    assert not unreached, f"catalog head sizes with no bundle case: {sorted(unreached)}"
    assert (
        not unknown
    ), f"bundle cases on head sizes the catalog lacks: {sorted(unknown)}"


def test_the_sweeps_vary_the_head_counts_the_catalog_no_longer_enumerates():
    """Anti-vacuity for the rule above: the claimed cases must still run several head
    configurations per head size, including both GQA and MHA. Keying coverage on the head
    size alone is only safe while something exercises the head counts end to end."""
    _require_catalog()
    by_head_size = {}
    for head_size, query_heads, kv_heads in swept_head_configs(SWEEPS):
        by_head_size.setdefault(head_size, set()).add((query_heads, kv_heads))
    assert by_head_size, "no claimed case in any sweep"
    for head_size, configs in sorted(by_head_size.items()):
        assert len(configs) > 1, (
            f"D{head_size} claims only {sorted(configs)}; one kernel serving every head "
            "count is then never exercised end to end"
        )
        assert any(hq != kv for hq, kv in configs), f"D{head_size} claims no GQA case"
        assert any(hq == kv for hq, kv in configs), f"D{head_size} claims no MHA case"

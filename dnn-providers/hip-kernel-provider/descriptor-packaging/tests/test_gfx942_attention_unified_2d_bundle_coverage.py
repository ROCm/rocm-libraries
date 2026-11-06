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

import pytest

from bundle_coverage import SWEEPS, catalog_path, claimed_cases, load_catalog

pytestmark = pytest.mark.quick

ARCH = "gfx942"
CATALOG = catalog_path(
    "gfx942_attention_unified_2d", "gfx942_attention_unified_2d.kdp.json"
)
ENGINE = "hipkernel:Gfx942AttentionUnified2D"
# The graph's io_data_type as the catalog spec spells it.
SPEC_DTYPE = {"half": "fp16", "bfloat16": "bf16"}


def catalog_cells(kdp):
    """(dtype, head_size, GQA ratio) of every kernel the descriptor packs."""
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


def all_claimed_cases():
    """(id, dtype, head_size, Hq, Hkv, attributes) of every case claimed on ARCH."""
    cases = [
        (
            case["id"],
            SPEC_DTYPE.get(
                case["values"]["io_data_type"], case["values"]["io_data_type"]
            ),
            q[3],
            q[1],
            k[1],
            case["values"].get("attributes", {}),
        )
        for sweep_dir in SWEEPS
        for case, q, k in claimed_cases(sweep_dir, ENGINE, ARCH)
    ]
    assert cases, "support.json claims no case for this engine"
    return cases


def test_every_claimed_case_is_a_cell_the_catalog_compiles():
    catalog = catalog_cells(load_catalog(CATALOG, ARCH))
    # An unreadable or emptied catalog would otherwise pass as "nothing unknown".
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

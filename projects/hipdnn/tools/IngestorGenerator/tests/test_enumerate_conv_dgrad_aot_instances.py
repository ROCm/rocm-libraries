# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for tools/enumerate_conv_dgrad_aot_instances.py (no compile, no GPU)."""

import collections
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="the rocKE builders need numpy")

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))

import enumerate_conv_dgrad_aot_instances as tool  # noqa: E402
import launch_surface  # noqa: E402

_REPO = launch_surface.find_repo_root(_TOOLS)
_ROCKE = _REPO / "dnn-providers" / "hip-kernel-provider" / "rocke"
_CONFIGS = Path(__file__).resolve().parent.parent / "configs"
_JOBS = min(16, os.cpu_count() or 1)


@pytest.fixture(scope="module", autouse=True)
def _rocke_on_path():
    """Bind rocKE for this module only (see test_enumerate_conv_aot_instances)."""
    before = list(sys.path)
    tool._bind_rocke(_ROCKE)
    yield
    sys.path[:] = before


@pytest.fixture(scope="module")
def catalog():
    """The committed labels' identities, in file order."""
    return tool.read_labels(tool.CATALOG_LABELS)


@pytest.fixture(scope="module")
def committed():
    """The committed per-dtype instance lists."""
    return {
        dtype: json.loads(
            (_CONFIGS / tool.INSTANCE_LIST.format(dtype=dtype)).read_text()
        )
        for dtype in tool.DTYPES
    }


def _label(identity):
    from benchmarks.common.kernel_cache import KernelIdentity

    return KernelIdentity.from_dict(identity).label()


#: The c48k192 block_h=8 block_q=32 fold_k32 waves_k=2 MFMA instance, left out of
#: the catalog in both dtypes: rocKE's compute kernel writes wrong dX even at
#: groups == 1 (see the labels file's header).
DROPPED_INSTANCES = {
    "fp16": "5da5fce3ee7974bb75839d487f3338e697ac98e6",
    "bf16": "ea7c649d3a54865546adb95698c39166193b3dd9",
}


def test_the_labels_file_has_lf_endings_and_one_label_per_line():
    raw = tool.CATALOG_LABELS.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    lines = raw.decode().splitlines()
    labels = [line for line in lines if not line.startswith("#")]
    # pruned 640 -> 406 on users/barkocot/gfx950-dgrad-aot-catalog-prune-150
    assert len(labels) == 406
    assert all(line.strip() == line and line for line in labels)


def test_the_kernels_wrong_at_groups_1_are_not_listed(catalog, committed):
    listed = {tool.stable_hash(i) for i in catalog}
    shipped = {tool.stable_hash(e) for entries in committed.values() for e in entries}
    for instance_id in DROPPED_INSTANCES.values():
        assert instance_id not in listed and instance_id not in shipped
    header = tool.CATALOG_LABELS.read_text().split("dgrad-", 1)[0]
    for instance_id in DROPPED_INSTANCES.values():
        assert instance_id in header


def test_the_catalog_lists_203_kernels_per_dtype_the_same_in_both(catalog):
    """Per dtype: 203 kernels (141 implicit-GEMM tiles + 62 direct) after pruning
    on users/barkocot/gfx950-dgrad-aot-catalog-prune-150 (was 320: 247 implicit-GEMM
    + 73 direct), and the two dtypes list the same kernels."""
    by_dtype = collections.defaultdict(list)
    for identity in catalog:
        by_dtype[identity["dtype_a"]].append(identity)
    assert sorted(by_dtype) == sorted(tool.DTYPES)
    for identities in by_dtype.values():
        algorithms = collections.Counter(i["algorithm"] for i in identities)
        assert algorithms == {
            "implicit_gemm": 141,
            "direct_depthwise_dgrad": 9,
            "direct_depthwise_dgrad_win": 31,
            "direct_grouped_dgrad_fused": 16,
            "direct_grouped_dgrad_mfma": 3,
            "direct_grouped_dgrad_4c": 2,
            "direct_dgrad_mfma_rtc": 1,
        }

    def without_dtype(identity):
        return tuple(
            sorted((k, v) for k, v in identity.items() if not k.startswith("dtype_"))
        )

    assert sorted(map(without_dtype, by_dtype["fp16"])) == sorted(
        map(without_dtype, by_dtype["bf16"])
    )


def test_the_lists_add_the_helpers_per_dtype_the_mfma_entries_need(committed):
    """One transpose and one reorganize per MFMA pipeline: the baked-channel one and
    the runtime-channel one, which uses its own _rtc transforms."""
    for dtype, entries in committed.items():
        families = collections.Counter(tool.family(e) for e in entries)
        assert families == {"implicit": 141, "direct": 62, "helper": 4}, dtype
        helpers = collections.Counter(
            e["algorithm"] for e in entries if tool.family(e) == "helper"
        )
        assert helpers == {
            "direct_dgrad_transpose": 1,
            "direct_dgrad_reorganize": 1,
            "direct_dgrad_transpose_rtc": 1,
            "direct_dgrad_reorganize_rtc": 1,
        }
        needed = {
            tool.stable_hash(h) for e in entries for h in tool.helper_identities(e)
        }
        listed = {tool.stable_hash(e) for e in entries if tool.family(e) == "helper"}
        assert needed == listed, dtype


def test_every_list_entry_is_a_dgrad_sweep_identity(committed):
    """Each entry, helpers included, is a gfx950 dgrad identity of rocKE's
    implicit-GEMM AOT grid or of its direct grid."""
    for dtype, entries in committed.items():
        swept = tool.sweep_hashes(dtype, _JOBS)
        missing = [_label(e) for e in entries if tool.stable_hash(e) not in swept]
        assert not missing, missing[:5]


def test_every_list_entry_converts(committed):
    import make_conv_dgrad_aot_config

    entries = [e for dtype in tool.DTYPES for e in committed[dtype]]
    assert tool.validate(entries, _JOBS) == []
    for identity in entries:
        spec, _ = make_conv_dgrad_aot_config.instance_spec(identity)
        assert spec.is_helper == (tool.family(identity) == "helper")
        assert spec.is_direct == (tool.family(identity) != "implicit")


def test_the_lists_are_ordered_by_family_then_hash(committed):
    for entries in committed.values():
        keys = [
            (tool.FAMILIES.index(tool.family(e)), tool.stable_hash(e)) for e in entries
        ]
        assert keys == sorted(keys)


def test_main_refuses_a_listed_helper(tmp_path, committed, capsys):
    helper = next(e for e in committed["fp16"] if tool.family(e) == "helper")
    labels = tmp_path / "labels.txt"
    labels.write_text(_label(helper) + "\n")
    argv = ["--rocke-root", str(_ROCKE), "--labels", str(labels)]
    assert tool.main(argv + ["--output-dir", str(tmp_path / "out")]) == 1
    assert "derived, not listed" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_main_refuses_a_forward_kernel(tmp_path, capsys):
    import enumerate_conv_aot_instances as fwd

    forward = fwd.read_labels(fwd.CATALOG_LABELS)[0]
    labels = tmp_path / "labels.txt"
    labels.write_text(_label(forward) + "\n")
    argv = ["--rocke-root", str(_ROCKE), "--labels", str(labels), "--jobs", "4"]
    assert tool.main(argv + ["--output-dir", str(tmp_path / "out")]) == 1
    err = capsys.readouterr().err
    assert "not a gfx950 dgrad sweep identity" in err
    assert "dgrad implicit-GEMM" in err


def test_committed_instance_lists_are_what_the_tool_enumerates(tmp_path, committed):
    """The shipped lists are the tool's output for the committed labels and the rocKE
    tree beside them."""
    from benchmarks.common.kernel_cache import current_llvm_flavor

    flavor = committed["fp16"][0]["llvm_flavor"]
    if flavor != current_llvm_flavor():
        pytest.skip(
            f"the lists are for LLVM flavor {flavor}, this process lowers to "
            f"{current_llvm_flavor()}, so every identity differs"
        )
    out = tmp_path / "lists"
    argv = ["--rocke-root", str(_ROCKE), "--output-dir", str(out), "--jobs", str(_JOBS)]
    assert tool.main(argv) == 0
    for dtype in tool.DTYPES:
        name = tool.INSTANCE_LIST.format(dtype=dtype)
        assert (out / name).read_bytes() == (_CONFIGS / name).read_bytes(), dtype

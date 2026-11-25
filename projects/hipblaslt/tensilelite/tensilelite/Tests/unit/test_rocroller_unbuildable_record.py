# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The record of rocRoller configurations that fail to build agrees with the kernels.

``CustomKernels/rocroller/unbuildable_configs.def`` lists the configurations
rocRoller fails to build. A build without rocRoller passes over a ranked tile
the table records, as rocRoller passes over a tile whose build fails. A
configuration that is both recorded and checked in would never be served: the
table would be stale, written with a rocRoller that could not build what was
later captured. A Stream-K row with workgroup mapping could never match a
ranked tile, because rocRoller turns workgroup mapping off for Stream-K.

The table is what ``scripts/rocroller_unbuildable_configs.py write`` renders
from the checked-in build records, ``scripts/rocroller_build_records.csv``.
Its rocRoller tree line names the ``shared/rocroller`` the records were built
with; once that tree changes, a recorded configuration may build, and the
build without rocRoller would pass over a tile rocRoller now runs. A changed
tree skips the check with the commands that rebuild the records, so a
rocRoller update does not fail unrelated changes; ``HIPBLASLT_REQUIRE_ROCROLLER_RECORD_TREE=1``
makes it a failure, for maintainers (scripts/README.md).
"""

import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_KERNELS = Path(__file__).resolve().parents[2] / "CustomKernels" / "rocroller"
_TABLE = _KERNELS / "unbuildable_configs.def"
_HIPBLASLT = Path(__file__).resolve().parents[4]
_SCRIPT = _HIPBLASLT / "scripts" / "rocroller_unbuildable_configs.py"
_RECORDS = _HIPBLASLT / "scripts" / "rocroller_build_records.csv"

_HIP = {
    "HIP_R_4F_E2M1": "FP4",
    "HIP_R_6F_E2M3": "FP6",
    "HIP_R_6F_E3M2": "BF6",
    "HIP_R_8F_E4M3": "FP8",
    "HIP_R_8F_E5M2": "BF8",
    "HIP_R_16F": "Half",
    "HIP_R_16BF": "BFloat16",
    "HIP_R_32F": "Float",
}
_ROW = re.compile(
    r"X\((true|false), (true|false), (\w+), (\w+), (\w+), (\w+), (\d+), (\d+), (\d+), (true|false), (true|false)\)"
)
_NAME = re.compile(
    r"RR_GEMM_([NT]{2})_(\w+?)_(\w+?)_(\w+?)_(\w+?)_Float_SA_BE8M0_32_SB_BE8M0_32_(SK_)?WGT_"
    r"(\d+)x(\d+)x(\d+)_UR_\d+(_WGM_)?(_LDSSYNC)?"
)


def _rows():
    rows, other = [], []
    for line in _TABLE.read_text().splitlines():
        if not line or line.startswith("//"):
            continue
        m = _ROW.fullmatch(line)
        if not m:
            other.append(line)
            continue
        ta, tb, a, b, c, d, tm, tn, tk, wgm, sk = m.groups()
        layout = ("T" if ta == "true" else "N") + ("T" if tb == "true" else "N")
        rows.append((layout, _HIP[a], _HIP[b], _HIP[c], _HIP[d], int(tm), int(tn), int(tk),
                     wgm == "true", sk == "true"))
    return rows, other


def test_every_line_is_a_unique_row():
    rows, other = _rows()
    assert not other, f"lines that are neither comments nor X(...) rows: {other[:5]}"
    assert rows, f"no rows in {_TABLE}"
    assert len(rows) == len(set(rows)), "a configuration is recorded twice"
    header = re.search(r"^// (\d+) configurations;", _TABLE.read_text(), re.M)
    assert header and int(header.group(1)) == len(rows), "the header's count is not the row count"


def test_no_streamk_row_has_workgroup_mapping():
    rows, _ = _rows()
    bad = [r for r in rows if r[8] and r[9]]
    assert not bad, f"Stream-K rows with workgroup mapping: {bad[:5]}"


def test_no_recorded_configuration_is_checked_in():
    rows = set(_rows()[0])
    shipped = {}
    for path in _KERNELS.glob("RR_GEMM_*.s"):
        m = _NAME.fullmatch(path.stem)
        assert m, f"unexpected kernel name {path.stem}"
        layout, a, b, c, d, sk, tm, tn, tk, wgm, _ = m.groups()
        shipped[(layout, a, b, c, d, int(tm), int(tn), int(tk), bool(wgm), bool(sk))] = path.stem
    assert shipped, f"no RR_GEMM_*.s kernels under {_KERNELS}"
    both = sorted(name for k, name in shipped.items() if k in rows)
    assert not both, f"recorded as unbuildable but checked in: {both}"


def _generator():
    if not _SCRIPT.is_file() or not _RECORDS.is_file():
        pytest.skip(f"{_SCRIPT.name} or {_RECORDS.name} is not in this tree")
    spec = importlib.util.spec_from_file_location("rocroller_unbuildable_configs", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_table_is_what_the_checked_in_records_render():
    gen = _generator()
    state, headers = gen.merge([str(_RECORDS)])
    source = gen.from_headers(None, headers, "Source", "--source")
    tree = gen.from_headers(None, headers, "rocRoller tree", "--rocroller-tree")
    rows = {k: p for k, (built, p) in state.items() if not built}
    assert gen.render(rows, source, tree) == _TABLE.read_text(), (
        f"{_TABLE.name} is not what {_RECORDS.name} renders; rerun {_SCRIPT.name} write (scripts/README.md)")
    records = gen.csv_header(source, tree) + "".join(
        gen.csv_line(k, *state[k]) for k in sorted(state, key=gen.sort_key))
    assert records == _RECORDS.read_text(), f"{_RECORDS.name} is not in the form write --records-out writes"


_STRICT_TREE = "HIPBLASLT_REQUIRE_ROCROLLER_RECORD_TREE"


def _strict_tree():
    return os.environ.get(_STRICT_TREE) == "1"


def _check_record_tree(recorded, current, strict):
    """Skip on a changed rocRoller tree, or fail when the maintainer check is on."""
    if current == recorded:
        return
    message = (
        f"shared/rocroller changed since {_RECORDS.name} was built (records: tree {recorded}, "
        f"HEAD: tree {current}), so {_TABLE.name} may list configurations rocRoller now builds. "
        f"To regenerate, on a HIPBLASLT_ENABLE_ROCROLLER=ON build with a GPU run "
        f"`{_SCRIPT.name} build --build <ON build dir> --configs {_RECORDS.name} --out records.csv`, "
        f"then `{_SCRIPT.name} write records.csv --out {_TABLE.name} --records-out {_RECORDS.name}` "
        "(scripts/README.md)")
    if strict:
        pytest.fail(message)
    pytest.skip(f"{message}; set {_STRICT_TREE}=1 to fail on this instead")


def test_the_records_were_built_with_this_rocroller():
    m = re.search(r"^// rocRoller tree: ([0-9a-f]{40})$", _TABLE.read_text(), re.M)
    assert m, f"{_TABLE.name} names no rocRoller tree"
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    head = subprocess.run(["git", "-C", str(_HIPBLASLT), "rev-parse", "HEAD:shared/rocroller"],
                          capture_output=True, text=True)
    if head.returncode != 0:
        pytest.skip(f"no shared/rocroller at HEAD: {head.stderr.strip()}")
    _check_record_tree(m.group(1), head.stdout.strip(), _strict_tree())


_RECORDED = "a" * 40
_CHANGED = "b" * 40


def test_a_changed_rocroller_tree_skips_with_how_to_regenerate():
    with pytest.raises(pytest.skip.Exception) as skipped:
        _check_record_tree(_RECORDED, _CHANGED, strict=False)
    text = str(skipped.value)
    for part in (_RECORDED, _CHANGED, f"{_SCRIPT.name} build", f"{_SCRIPT.name} write",
                 "scripts/README.md", f"{_STRICT_TREE}=1"):
        assert part in text, (part, text)


def test_a_changed_rocroller_tree_fails_under_the_maintainer_check():
    with pytest.raises(pytest.fail.Exception) as failed:
        _check_record_tree(_RECORDED, _CHANGED, strict=True)
    text = str(failed.value)
    for part in (_RECORDED, _CHANGED, f"{_SCRIPT.name} build", f"{_SCRIPT.name} write"):
        assert part in text, (part, text)


def test_the_recorded_rocroller_tree_passes_either_way():
    _check_record_tree(_RECORDED, _RECORDED, strict=False)
    _check_record_tree(_RECORDED, _RECORDED, strict=True)


@pytest.mark.parametrize("value, strict", [(None, False), ("", False), ("0", False), ("1", True)])
def test_the_maintainer_check_is_on_only_when_its_variable_is_1(monkeypatch, value, strict):
    if value is None:
        monkeypatch.delenv(_STRICT_TREE, raising=False)
    else:
        monkeypatch.setenv(_STRICT_TREE, value)
    assert _strict_tree() is strict

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Addition-only splice: a scratch render goes into a live engine directory without
touching a retained identity.

The generator mints new UUIDs on every run, so "regenerate and copy" replaces every
live id. The splice must keep every retained byte, append only entries whose name is
new, and refuse (writing nothing) when a retained object changed.
"""

from __future__ import annotations

import copy
import difflib
import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import yaml

from codegen.config_loader import load_config
from codegen.generator import IngestorGenerator

_ROOT = Path(__file__).resolve().parents[1]
_SPLICE = _ROOT / "tools" / "splice_additions.py"
_BASE = yaml.safe_load(
    (_ROOT / "configs" / "scale_add.yaml").read_text(encoding="utf-8")
)
_KDP = "scale_add.kdp.json"
_NEW_KERNEL = {
    "name": "scale_add.f32_block128",
    "kernel_source": {
        "kind": "embedded_source",
        "source_file": "ScaleAdd.cpp",
        "entry_point": "ScaleAdd",
    },
    "metadata": {"block_size": 128, "dtype": "FLOAT"},
    "priority": 0,
}


def _render(config: dict, tmp_path: Path, label: str) -> Path:
    """Render `config` and return its descriptor directory."""
    path = tmp_path / f"{label}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    out = tmp_path / label
    IngestorGenerator(_ROOT / "templates").render(load_config(path), out)
    (kdp,) = out.rglob(_KDP)
    return kdp.parent


def _with(mutate) -> dict:
    config = copy.deepcopy(_BASE)
    mutate(config)
    return config


def _add_kernel(config: dict) -> None:
    config["packs"][0]["kernels"].append(copy.deepcopy(_NEW_KERNEL))


def _splice(live: Path, scratch: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(_SPLICE),
            "--live",
            str(live),
            "--scratch",
            str(scratch),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(root.iterdir())}


def _kernels(root: Path) -> list[dict]:
    return json.loads((root / _KDP).read_text(encoding="utf-8"))["kernelDescriptors"]


@pytest.fixture
def live(tmp_path) -> Path:
    return _render(_BASE, tmp_path, "live")


def test_appends_only_the_new_kernel(tmp_path, live):
    before = _snapshot(live)
    scratch = _render(_with(_add_kernel), tmp_path, "scratch")
    report = tmp_path / "report.json"

    result = _splice(live, scratch, "--report", str(report))

    assert result.returncode == 0, result.stderr
    after = _snapshot(live)
    # Engine-level documents are untouched, byte for byte.
    assert {n: b for n, b in after.items() if n != _KDP} == {
        n: b for n, b in before.items() if n != _KDP
    }
    # The KDP change is a pure insertion: no live line is removed or rewritten.
    ops = difflib.SequenceMatcher(
        a=before[_KDP].decode().splitlines(), b=after[_KDP].decode().splitlines()
    ).get_opcodes()
    assert {op for op, *_ in ops} == {"equal", "insert"}
    # Retained kernels keep their live ids and order; the new one keeps its scratch id.
    live_kernels = json.loads(before[_KDP])["kernelDescriptors"]
    spliced = _kernels(live)
    assert spliced[: len(live_kernels)] == live_kernels
    assert [k["name"] for k in spliced[len(live_kernels) :]] == [_NEW_KERNEL["name"]]
    scratch_new = next(k for k in _kernels(scratch) if k["name"] == _NEW_KERNEL["name"])
    assert spliced[-1]["id"] == scratch_new["id"]
    assert json.loads(report.read_text())["added_kernels"] == 1


def test_unchanged_render_is_a_no_op(tmp_path, live):
    before = _snapshot(live)
    scratch = _render(_BASE, tmp_path, "scratch")

    result = _splice(live, scratch)

    assert result.returncode == 0, result.stderr
    assert "appended 0" in result.stdout
    assert _snapshot(live) == before


def _add_pack(scratch: Path) -> str:
    """Write a second KDP into the scratch render, as a new pack would: fresh ids, one
    kernel. Returns its file name."""
    doc = json.loads((scratch / _KDP).read_text(encoding="utf-8"))
    kernel = {**doc["kernelDescriptors"][0], "name": _NEW_KERNEL["name"]}
    kernel["id"] = str(uuid.uuid4())
    doc.update(id=str(uuid.uuid4()), kernelDescriptors=[kernel])
    name = "scale_add_extra.kdp.json"
    (scratch / name).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return name


def test_kernels_of_a_new_pack_count_as_added(tmp_path, live):
    retained = len(_kernels(live))
    scratch = _render(_BASE, tmp_path, "scratch")
    new_kdp = _add_pack(scratch)
    report = tmp_path / "report.json"

    result = _splice(live, scratch, "--report", str(report))

    assert result.returncode == 0, result.stderr
    assert (live / new_kdp).is_file()
    counts = json.loads(report.read_text())
    assert counts["new_files"] == [new_kdp]
    assert [k["name"] for k in counts["added"]] == [_NEW_KERNEL["name"]]
    assert counts["added_kernels"] == 1
    assert counts["total_kernels"] == retained + 1


def test_check_mode_writes_nothing(tmp_path, live):
    before = _snapshot(live)
    scratch = _render(_with(_add_kernel), tmp_path, "scratch")

    result = _splice(live, scratch, "--check")

    assert result.returncode == 0, result.stderr
    assert "would append 1" in result.stdout
    assert _snapshot(live) == before


def _change_retained_priority(config: dict) -> None:
    _add_kernel(config)
    config["packs"][0]["kernels"][0]["priority"] = 5


def _drop_retained_kernel(config: dict) -> None:
    del config["packs"][0]["kernels"][0]


def _change_engine(config: dict) -> None:
    _add_kernel(config)
    config["engine"]["sdk_version"] = "1.1.0"


@pytest.mark.parametrize(
    ("mutate", "names"),
    [
        (_change_retained_priority, "scale_add.f32_block64"),
        (_drop_retained_kernel, "scale_add.f32_block64"),
        (_change_engine, "scale_add.ued.json"),
    ],
    ids=["changed-retained-kernel", "dropped-retained-kernel", "changed-engine"],
)
def test_refuses_a_change_to_a_retained_object(tmp_path, live, mutate, names):
    before = _snapshot(live)
    scratch = _render(_with(mutate), tmp_path, "scratch")

    result = _splice(live, scratch)

    assert result.returncode == 1
    assert "REFUSED" in result.stderr and names in result.stderr
    assert _snapshot(live) == before


def test_keeps_crlf_line_endings_of_a_windows_checkout(tmp_path, live):
    kdp = live / _KDP
    kdp.write_bytes(kdp.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    before = kdp.read_bytes()
    scratch = _render(_with(_add_kernel), tmp_path, "scratch")

    result = _splice(live, scratch)

    assert result.returncode == 0, result.stderr
    after = kdp.read_bytes()
    assert after.count(b"\n") == after.count(b"\r\n")
    ops = difflib.SequenceMatcher(
        a=before.split(b"\r\n"), b=after.split(b"\r\n")
    ).get_opcodes()
    assert {op for op, *_ in ops} == {"equal", "insert"}

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The import-time check of the per-arch tuned tables, CPU only.

* the shipped tables pass it (the package imports);
* it refuses an unknown knob, a ``not_built`` or ``gated`` value, a value
  that does not plan on the key, a malformed key and a status table naming
  another arch, an unmeasured status or a claim; it names the module;
* a bad entry in a shipped table fails the import of the package (a copy of
  the package with one entry broken, imported in a fresh process).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from dispatch.attention_bwd import dense_tuning, gfx950_tuning, routing, tuned_tables
from dispatch.attention_bwd import tuning_common as tc

LIBRARY = Path(dense_tuning.__file__).resolve().parents[2]


def _mod(general=None, dense=None, gstatus=None, dstatus=None, arch="gfx950"):
    return types.SimpleNamespace(
        __name__=f"fake_{arch}_tuning",
        ARCH=arch,
        GENERAL_TUNED_TABLE=general or {},
        DENSE_TUNED=dense or {},
        GENERAL_PERF_STATUS=gstatus or {},
        DENSE_PERF_STATUS=dstatus or {},
    )


def _check(mod):
    tuned_tables.check_arch_tables(mod, dense_axis=dense_tuning.dense_axis)


def test_shipped_tables_pass_and_run_at_import():
    for mod in routing.ARCH_TUNING.values():
        _check(mod)
    routing._check_tuned_tables()


def test_the_check_plans(monkeypatch):
    """The import pass plans each entry (one representative, ``limit=1``)."""
    calls = []
    real = tuned_tables.check_general_entry

    def spy(arch, key, knobs, *, limit=None):
        calls.append(limit)
        return real(arch, key, knobs, limit=limit)

    monkeypatch.setattr(tuned_tables, "check_general_entry", spy)
    _check(gfx950_tuning)
    assert calls == [1] * len(gfx950_tuning.GENERAL_TUNED_TABLE)


_KEY, _KNOBS = next(iter(gfx950_tuning.GENERAL_TUNED_TABLE.items()))
_DKEY, _DKNOBS = next(iter(gfx950_tuning.DENSE_TUNED.items()))


@pytest.mark.parametrize(
    "mod,match",
    [
        (_mod({_KEY: {**_KNOBS, "no_such_knob": 1}}), "not a registered knob"),
        (_mod({_KEY: {**_KNOBS, "waves": 3}}), None),  # does not plan
        (_mod({_KEY: {"lb_order": "mirror", "waves_per_eu": 9}}), None),
        (_mod({_KEY[:-1]: _KNOBS}), "not a tuning key"),
        (_mod({_KEY: _KNOBS}, arch="gfx942"), "not a gfx942 general key"),
        (_mod(dense={_DKEY: {**_DKNOBS, "no_such_knob": 1}}), "not a registered"),
        (_mod(dense={_DKEY: {**_DKNOBS, "oob": "nope"}}), None),
        (_mod(gstatus={_KEY[:9]: "verified"}, arch="gfx942"), "not a gfx942"),
        (_mod(gstatus={_KEY[:9]: "unverified"}), "not a measured status"),
        (_mod(gstatus={_KEY[:9]: "fast"}), "not a measured status"),
        (_mod(dstatus={("gfx950", "general"): "verified"}), "not a gfx950 dense"),
    ],
)
def test_bad_entries_are_refused(mod, match):
    with pytest.raises(ValueError, match=match) as info:
        _check(mod)
    assert str(info.value).startswith(mod.__name__)


def test_an_entry_without_a_plannable_request_is_refused(monkeypatch):
    """A key none of whose representatives plans (here: every representative
    asks for dropout) is refused, so no entry ships unchecked."""
    import dataclasses

    reqs = tuned_tables.general_key_requests("gfx950", _KEY)
    declined = [dataclasses.replace(r, dropout=0.1) for r in reqs[:3]]
    monkeypatch.setattr(tuned_tables, "general_key_requests", lambda *_a: declined)
    with pytest.raises(ValueError, match="no plannable representative"):
        _check(_mod({_KEY: _KNOBS}))
    with pytest.raises(ValueError, match="no plannable representative"):
        tuned_tables.check_general_entry("gfx950", _KEY, _KNOBS)


def test_not_built_and_gated_values_are_refused(monkeypatch):
    ax = tc.backward_axis("lb_order")
    monkeypatch.setitem(
        tc._BY_NAME, "lb_order", ax.__class__(**{**ax.__dict__, "gated": ("reverse",)})
    )
    with pytest.raises(ValueError, match="lb_order='reverse' is gated"):
        _check(_mod({_KEY: {**_KNOBS, "lb_order": "reverse"}}))
    monkeypatch.setitem(
        tc._BY_NAME,
        "lb_order",
        ax.__class__(**{**ax.__dict__, "not_built": ("reverse",)}),
    )
    with pytest.raises(ValueError, match="lb_order='reverse' is not built"):
        _check(_mod({_KEY: {**_KNOBS, "lb_order": "reverse"}}))


def test_a_bad_shipped_entry_fails_the_package_import(tmp_path):
    """A copy of the package whose gfx950 table has one entry with an
    unregistered knob does not import (fresh process)."""
    pkg = tmp_path / "dispatch"
    shutil.copytree(
        LIBRARY / "dispatch" / "attention_bwd",
        pkg / "attention_bwd",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy(LIBRARY / "dispatch" / "__init__.py", pkg / "__init__.py")
    path = pkg / "attention_bwd" / "gfx950_tuning.py"
    src = path.read_text(encoding="utf-8")
    anchor = "GENERAL_TUNED_TABLE: Mapping[tuple, Mapping[str, Any]] = {\n"
    assert src.count(anchor) == 1
    # the first entry's knob mapping gains an unregistered knob
    head, tail = src.split(anchor)
    tail = tail.replace("): {\n", '): {\n        "no_such_knob": 1,\n', 1)
    path.write_text(head + anchor + tail, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(LIBRARY)] + env.get("PYTHONPATH", "").split(os.pathsep)
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    code = "import dispatch; print(dispatch.__file__); import dispatch.attention_bwd"
    bad = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert Path(bad.stdout.strip()).resolve().parent == pkg.resolve()  # the copy
    assert bad.returncode != 0
    assert "no_such_knob" in bad.stderr and "not a registered knob" in bad.stderr
    # the unmodified copy imports
    path.write_text(src, encoding="utf-8")
    good = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert good.returncode == 0, good.stderr

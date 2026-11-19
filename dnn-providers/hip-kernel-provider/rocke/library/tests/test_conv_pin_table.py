# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shipped conv pin tables as an opt-in dispatch candidate (host only).

* every case of every mode resolves through :func:`dispatch_conv_pinned` to
  the pinned instance and launch, and its label round-trips through the
  kernel identity;
* the deterministic mode pins deterministic launches only;
* ``algorithm="auto"`` never selects a pinned candidate, also once a table is
  registered with the grouped-convolution dispatcher;
* the kernarg signature dispatch reports is the one the ``--run-from-cache``
  launcher binds.
"""

from __future__ import annotations

import dataclasses
import json
import re

import pytest

from dispatch import conv_pins as cp

TABLES = cp.shipped_conv_pin_tables()


def _modes():
    for stem, path in TABLES.items():
        for mode in json.loads(path.read_text())["modes"]:
            yield pytest.param(path, mode, id=f"{stem}#{mode}")


def test_a_table_is_shipped():
    assert "gfx942_unet3d_bf16" in TABLES


@pytest.mark.parametrize("path,mode", list(_modes()))
def test_every_case_resolves_to_its_pin(path, mode):
    from kernels.conv.kernel_identity import KernelIdentity

    doc = json.loads(path.read_text())
    pins = cp.load_conv_pins(path, mode)
    assert len(pins) == len(doc["modes"][mode]["cases"])
    for case, row in doc["modes"][mode]["cases"].items():
        req = cp.request_from_miopen_cmd(row["cmd"], doc["arch"])
        res = cp.dispatch_conv_pinned(req, pins)
        inst = row["kernel"].split()[0]
        assert res.spec.name == inst, case
        assert KernelIdentity.from_label(inst).label() == inst
        assert res.spec.identity.direction == req.direction
        assert res.spec.launch_params() == pins.lookup(req).launch
        assert res.candidate.opt_in and res.candidate.algorithm == "pinned"
        assert res.block[0] == res.spec.identity.warp_m * res.spec.identity.warp_n * 64


@pytest.mark.parametrize("path,mode", list(_modes()))
def test_deterministic_mode_pins_deterministic_launches(path, mode):
    from benchmarks.common.det_classify import is_deterministic

    doc = json.loads(path.read_text())
    body = doc["modes"][mode]
    pins = cp.load_conv_pins(path, mode)
    for case, row in body["cases"].items():
        req = cp.request_from_miopen_cmd(row["cmd"], doc["arch"])
        pin = pins.lookup(req)
        res = cp.dispatch_conv_pinned(req, pins)
        det = is_deterministic(req.direction, res.spec.identity, pin.split_k, "f32")
        assert det == row["deterministic"], case
        if body["deterministic"]:
            assert det, case


def test_other_problems_and_arches_are_refused():
    path = TABLES["gfx942_unet3d_bf16"]
    pins = cp.load_conv_pins(path, "deterministic")
    row = next(
        iter(json.loads(path.read_text())["modes"]["deterministic"]["cases"].values())
    )
    req = cp.request_from_miopen_cmd(row["cmd"], "gfx942")
    with pytest.raises(ValueError):
        cp.dispatch_conv_pinned(
            dataclasses.replace(req, C=req.C * 3, Hi=req.Hi + 3), pins
        )
    with pytest.raises(ValueError):
        cp.dispatch_conv_pinned(dataclasses.replace(req, arch="gfx950"), pins)
    with pytest.raises(ValueError):
        cp.dispatch_conv_pinned(dataclasses.replace(req, llvm_flavor="llvm20"), pins)


def test_a_table_needs_its_mode_named():
    with pytest.raises(ValueError, match="name one"):
        cp.load_conv_pins(TABLES["gfx942_unet3d_bf16"])
    with pytest.raises(ValueError, match="no mode"):
        cp.load_conv_pins(TABLES["gfx942_unet3d_bf16"], "nope")


def test_registered_table_is_opt_in_in_the_grouped_dispatcher():
    import dispatch.grouped_convolution as gc

    path = TABLES["gfx942_unet3d_bf16"]
    pins = cp.load_conv_pins(path, "fastest")
    cp.register_conv_pin_table(pins)
    assert cp.register_conv_pin_table(pins) == ()  # idempotent
    cases = json.loads(path.read_text())["modes"]["fastest"]["cases"]
    for case in ("fwd_s3", "dgrad_s10"):
        req = cp.request_from_miopen_cmd(cases[case]["cmd"], "gfx942")
        with pytest.raises(ValueError):
            gc.dispatch_conv_grouped(
                req
            )  # auto: no gfx942 instance list, pinned is opt-in
        res = gc.dispatch_conv_grouped(dataclasses.replace(req, algorithm="pinned"))
        assert res.spec.name == cases[case]["kernel"].split()[0]
    # gfx950 auto routing is unchanged: the pinned candidate never admits it.
    for direction in ("fwd", "dgrad"):
        assert all(
            c.opt_in
            for c in gc.conv_grouped_candidates(direction)
            if c.algorithm == "pinned"
        )


def test_signature_is_the_run_from_cache_one():
    from benchmarks.common.benchmark_implicit_gemm_conv import _signature_for_identity

    path = TABLES["gfx942_unet3d_bf16"]
    doc = json.loads(path.read_text())
    for mode in doc["modes"]:
        pins = cp.load_conv_pins(path, mode)
        for row in doc["modes"][mode]["cases"].values():
            res = cp.dispatch_conv_pinned(
                cp.request_from_miopen_cmd(row["cmd"], "gfx942"), pins
            )
            want = tuple(_signature_for_identity(res.spec.identity, res.spec.dtype))
            assert res.signature == want


def _pinned_cases():
    for stem, path in TABLES.items():
        doc = json.loads(path.read_text())
        for mode, body in doc["modes"].items():
            for case in body["cases"]:
                yield pytest.param(path, mode, case, id=f"{stem}#{mode}:{case}")


@pytest.mark.parametrize("path,mode,case", list(_pinned_cases()))
def test_every_pinned_instance_builds_its_kernel_ir(path, mode, case):
    """The pinned candidate's build() rebuilds the binary's kernel from the
    label alone, in a process with no AOT scoping variable set."""
    doc = json.loads(path.read_text())
    pins = cp.load_conv_pins(path, mode)
    res = cp.dispatch_conv_pinned(
        cp.request_from_miopen_cmd(
            doc["modes"][mode]["cases"][case]["cmd"], doc["arch"]
        ),
        pins,
    )
    kernel = res.build()
    assert kernel is not None
    with pytest.raises(ValueError, match="not gfx950"):
        res.candidate.built(res.spec, "gfx950")


def test_step_replay_resolves_a_table_through_dispatch():
    from benchmarks.common.conv_step_replay import load_pins_cases

    path = TABLES["gfx942_unet3d_bf16"]
    doc = json.loads(path.read_text())
    for mode, body in doc["modes"].items():
        forms = {}
        for k, v in body["cases"].items():
            base, _, suffix = k.rpartition("_")
            if re.fullmatch(r"s\d+", suffix):
                forms.setdefault("", {})[k] = v["kernel"]
            else:
                forms.setdefault(suffix, {})[base] = v["kernel"]
        assert "" in forms and "pad1" in forms
        for form, want in forms.items():
            cases = load_pins_cases(
                f"{path}#{mode}" + (f"@{form}" if form else ""), doc["arch"]
            )
            assert {k: v["kernel"] for k, v in cases.items()} == want
            # Padded forms: pad 1 on H and W; the pad1 form also pads D.
            if form:
                assert all(" -p 1 -q 1 " in v["cmd"] for v in cases.values())
            if form == "pad1":
                assert all(" --pad_d 1 " in v["cmd"] for v in cases.values())
            if form == "pad011":
                assert all(" --pad_d 0 " in v["cmd"] for v in cases.values())


def test_the_tables_are_package_data():
    """A non-editable install ships the tables beside dispatch/conv_pins.py:
    every JSON file under dispatch/ matches a package-data pattern."""
    import fnmatch
    import tomllib
    from pathlib import Path

    lib = Path(cp.__file__).resolve().parents[1]
    data = tomllib.loads((lib / "pyproject.toml").read_text())
    patterns = data["tool"]["setuptools"]["package-data"]["dispatch"]
    files = [
        f.relative_to(lib / "dispatch").as_posix()
        for f in (lib / "dispatch").rglob("*.json")
    ]
    assert any(f.startswith("conv_pin_tables/") for f in files)
    for f in files:
        assert any(fnmatch.fnmatch(f, p) for p in patterns), f

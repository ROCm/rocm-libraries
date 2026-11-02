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
    row = next(iter(json.loads(path.read_text())["modes"]["deterministic"]["cases"].values()))
    req = cp.request_from_miopen_cmd(row["cmd"], "gfx942")
    with pytest.raises(ValueError):
        cp.dispatch_conv_pinned(dataclasses.replace(req, C=req.C * 3, Hi=req.Hi + 3), pins)
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
            gc.dispatch_conv_grouped(req)  # auto: no gfx942 instance list, pinned is opt-in
        res = gc.dispatch_conv_grouped(dataclasses.replace(req, algorithm="pinned"))
        assert res.spec.name == cases[case]["kernel"].split()[0]
    # gfx950 auto routing is unchanged: the pinned candidate never admits it.
    for direction in ("fwd", "dgrad"):
        assert all(
            c.opt_in for c in gc.conv_grouped_candidates(direction) if c.algorithm == "pinned"
        )


def test_signature_is_the_run_from_cache_one():
    from benchmarks.common.benchmark_implicit_gemm_conv import _signature_for_identity

    path = TABLES["gfx942_unet3d_bf16"]
    doc = json.loads(path.read_text())
    for mode in doc["modes"]:
        pins = cp.load_conv_pins(path, mode)
        for row in doc["modes"][mode]["cases"].values():
            res = cp.dispatch_conv_pinned(cp.request_from_miopen_cmd(row["cmd"], "gfx942"), pins)
            want = tuple(_signature_for_identity(res.spec.identity, res.spec.dtype))
            assert res.signature == want


@pytest.mark.parametrize("case", ["fwd_s10", "dgrad_s10", "wgrad_s10"])
def test_a_pinned_instance_builds_its_kernel_ir(case):
    path = TABLES["gfx942_unet3d_bf16"]
    doc = json.loads(path.read_text())
    pins = cp.load_conv_pins(path, "deterministic")
    res = cp.dispatch_conv_pinned(
        cp.request_from_miopen_cmd(doc["modes"]["deterministic"]["cases"][case]["cmd"], "gfx942"), pins
    )
    try:
        kernel = res.build()
    except LookupError as e:
        # The opt-in identity axes of the cache recipe (ROCKE_AOT_* in ADD
        # mode) are not set in this process: build() is the --compile-all job
        # of the identity, which only those axes produce.
        pytest.skip(f"pinned identity outside this process's AOT grid: {e}")
    assert kernel is not None
    with pytest.raises(ValueError, match="not gfx950"):
        res.candidate.built(res.spec, "gfx950")

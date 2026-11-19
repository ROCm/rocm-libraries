# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""dispatch.conv_pins: pins file -> per-request pinned instance / ranker (host only)."""

import json
from types import SimpleNamespace

import pytest

from dispatch.conv_pins import PIN_TABLE_DIR, parse_miopen_cmd, pins_from_mapping

_ROWS = json.loads((PIN_TABLE_DIR / "gfx942_unet3d_bf16.json").read_text())["modes"][
    "fastest"
]["cases"]
# Kernel labels (no launch) of the shipped table: a forward, another forward
# and a wgrad kernel.
FWD_A = _ROWS["fwd_s1"]["kernel"].split()[0]
FWD_C = _ROWS["fwd_s10"]["kernel"].split()[0]
WGRAD_B = _ROWS["wgrad_s1"]["kernel"].split()[0]
DGRAD = _ROWS["dgrad_s1"]["kernel"].split()[0]

CMD = (
    "MIOpenDriver convbfp16 -n 1 -c 64 --in_d 258 -H 130 -W 130 -k 64 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 "
    "--conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 --in_layout NDHWC --fil_layout NDHWC "
    "--out_layout NDHWC -m conv -g 1 -F {F} -t 1 -V 0"
)


def _req(direction="fwd", **kw):
    d = dict(
        N=1,
        C=64,
        K=64,
        Hi=130,
        Wi=130,
        Y=3,
        X=3,
        G=1,
        stride_h=1,
        stride_w=1,
        pad_h=0,
        pad_w=0,
        dilation_h=1,
        dilation_w=1,
        Di=258,
        Z=3,
        stride_d=1,
        pad_d=0,
        dilation_d=1,
        dtype="bf16",
        direction=direction,
    )
    d.update(kw)
    return SimpleNamespace(**d)


def _pins():
    return pins_from_mapping(
        {
            "cases": {
                "fwd_s1": {
                    "cmd": CMD.format(F=1),
                    "kernel": FWD_A,
                    "ms": 1.9,
                    "calls": 2,
                },
                "wgrad_s1": {
                    "cmd": CMD.format(F=4),
                    "kernel": f"{WGRAD_B} @split_k=304",
                },
                "fwd_s3": {
                    "cmd": CMD.format(F=1).replace("-c 64", "-c 128"),
                    "kernel": f"{FWD_C} @split_k=2/full1824",
                },
            }
        }
    )


def test_parse_cmd_directions():
    for f, d in ((1, "fwd"), (2, "dgrad"), (4, "wgrad")):
        direction, dtype, key = parse_miopen_cmd(CMD.format(F=f))
        assert (direction, dtype) == (d, "bf16")
        assert key[:3] == (1, 64, 64)


def test_lookup_and_launch():
    p = _pins()
    assert len(p) == 3
    assert p.lookup(_req()).instance == FWD_A
    w = p.lookup(_req("wgrad"))
    assert (w.instance, w.launch) == (WGRAD_B, {"split_k": 304})
    c = p.lookup(_req(C=128))
    assert c.launch == {"split_k": 2, "tail_full": 1824}
    assert p.lookup(_req("dgrad")) is None
    assert p.lookup(_req(dtype="fp16")) is None
    assert p.lookup(_req(Hi=66)) is None


def test_ranker_puts_pin_first_and_never_drops():
    rank = _pins().ranker()
    cands = [SimpleNamespace(name=n) for n in ("x", FWD_A, "y")]
    assert [c.name for c in rank(_req(), cands)] == [FWD_A, "x", "y"]
    # pinned instance not admitted / no pin: registry order
    assert [c.name for c in rank(_req(), cands[::2])] == ["x", "y"]
    assert [c.name for c in rank(_req("dgrad"), cands)] == ["x", FWD_A, "y"]


def test_conflicting_pins_refused():
    with pytest.raises(ValueError):
        pins_from_mapping(
            {
                "cases": {
                    "a": {"cmd": CMD.format(F=1), "kernel": FWD_A},
                    "b": {"cmd": CMD.format(F=1), "kernel": FWD_C},
                }
            }
        )


def test_one_problem_pinned_to_two_launches_is_refused():
    """The same kernel with another launch is a conflict too; the very same
    label twice is not."""
    with pytest.raises(ValueError, match="two launches"):
        pins_from_mapping(
            {
                "cases": {
                    "a": {"cmd": CMD.format(F=1), "kernel": f"{FWD_C} @split_k=2"},
                    "b": {"cmd": CMD.format(F=1), "kernel": f"{FWD_C} @split_k=7"},
                }
            }
        )
    p = pins_from_mapping(
        {
            "cases": {
                "a": {"cmd": CMD.format(F=1), "kernel": f"{FWD_C} @split_k=2"},
                "b": {"cmd": CMD.format(F=1), "kernel": f"{FWD_C} @split_k=2"},
            }
        }
    )
    assert len(p) == 1 and p.lookup(_req()).split_k == 2


@pytest.mark.parametrize(
    "cmd, kernel",
    [
        (CMD.format(F=1), DGRAD),  # a dgrad kernel on a forward problem
        (CMD.format(F=1).replace("convbfp16", "conv"), FWD_A),  # bf16 on fp32
        (CMD.format(F=1), "fwd-A"),  # not a label
    ],
)
def test_a_label_must_fit_its_command(cmd, kernel):
    with pytest.raises(ValueError):
        pins_from_mapping({"cases": {"a": {"cmd": cmd, "kernel": kernel}}})


def test_the_pinned_candidate_checks_the_kernel_against_the_request():
    """A pins object built by hand (no loader check) still never dispatches a
    kernel of another direction or dtype."""
    import dataclasses

    from dispatch import conv_pins as cp

    path = PIN_TABLE_DIR / "gfx942_unet3d_bf16.json"
    pins = cp.load_conv_pins(path, "fastest")
    req = cp.request_from_miopen_cmd(_ROWS["fwd_s1"]["cmd"], "gfx942")
    key = cp.request_key(req)
    pins.pins[key] = dataclasses.replace(pins.pins[key], instance=DGRAD, label=DGRAD)
    with pytest.raises(ValueError, match="dgrad bf16 kernel, not fwd bf16"):
        cp.dispatch_conv_pinned(req, pins)

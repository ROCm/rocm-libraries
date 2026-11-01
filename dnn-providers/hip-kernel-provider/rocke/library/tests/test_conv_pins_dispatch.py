# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""dispatch.conv_pins: pins file -> per-request pinned instance / ranker (host only)."""

from types import SimpleNamespace

import pytest

from dispatch.conv_pins import parse_miopen_cmd, pins_from_mapping

CMD = ("MIOpenDriver convbfp16 -n 1 -c 64 --in_d 258 -H 130 -W 130 -k 64 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 "
       "--conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 --in_layout NDHWC --fil_layout NDHWC "
       "--out_layout NDHWC -m conv -g 1 -F {F} -t 1 -V 0")


def _req(direction="fwd", **kw):
    d = dict(N=1, C=64, K=64, Hi=130, Wi=130, Y=3, X=3, G=1, stride_h=1, stride_w=1, pad_h=0, pad_w=0,
             dilation_h=1, dilation_w=1, Di=258, Z=3, stride_d=1, pad_d=0, dilation_d=1, dtype="bf16",
             direction=direction)
    d.update(kw)
    return SimpleNamespace(**d)


def _pins():
    return pins_from_mapping({"cases": {
        "fwd_s1": {"cmd": CMD.format(F=1), "kernel": "fwd-A", "ms": 1.9, "calls": 2},
        "wgrad_s1": {"cmd": CMD.format(F=4), "kernel": "wgrad-B @split_k=304"},
        "fwd_s3": {"cmd": CMD.format(F=1).replace("-c 64", "-c 128"), "kernel": "fwd-C @split_k=2/full1824"},
    }})


def test_parse_cmd_directions():
    for f, d in ((1, "fwd"), (2, "dgrad"), (4, "wgrad")):
        direction, dtype, key = parse_miopen_cmd(CMD.format(F=f))
        assert (direction, dtype) == (d, "bf16")
        assert key[:3] == (1, 64, 64)


def test_lookup_and_launch():
    p = _pins()
    assert len(p) == 3
    assert p.lookup(_req()).instance == "fwd-A"
    w = p.lookup(_req("wgrad"))
    assert (w.instance, w.launch) == ("wgrad-B", {"split_k": 304})
    c = p.lookup(_req(C=128))
    assert c.launch == {"split_k": 2, "tail_full": 1824}
    assert p.lookup(_req("dgrad")) is None
    assert p.lookup(_req(dtype="fp16")) is None
    assert p.lookup(_req(Hi=66)) is None


def test_ranker_puts_pin_first_and_never_drops():
    rank = _pins().ranker()
    cands = [SimpleNamespace(name=n) for n in ("x", "fwd-A", "y")]
    assert [c.name for c in rank(_req(), cands)] == ["fwd-A", "x", "y"]
    # pinned instance not admitted / no pin: registry order
    assert [c.name for c in rank(_req(), cands[::2])] == ["x", "y"]
    assert [c.name for c in rank(_req("dgrad"), cands)] == ["x", "fwd-A", "y"]


def test_conflicting_pins_refused():
    with pytest.raises(ValueError):
        pins_from_mapping({"cases": {"a": {"cmd": CMD.format(F=1), "kernel": "k1"},
                                     "b": {"cmd": CMD.format(F=1), "kernel": "k2"}}})


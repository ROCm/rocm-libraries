# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU differential test of unified LSE dispatch over arch x path x feature.

Each key runs LSE off and on in the same process. A key served both ways keeps
its candidate, path, tuning id and spec; only ``emit_lse`` and the ``lse``
kernel-name token change, apart from the named reroutes in ``_REROUTES``. LSE
on: a served request runs a variant that writes LSE, and every unified path
that refuses it says why in its own words. Host only.
"""

import ast
import dataclasses
import itertools
import re

import pytest

from dispatch.attention import (
    AttentionRequest,
    attention_candidates,
    dispatch_attention,
)
from dispatch.attention.common import _problem
from kernels.common.attention_unified import select_auto_path
from rocke.helpers.spec import kernel_name_join

_ARCHES = ("gfx942", "gfx950", "gfx90a", "gfx1250")
_LSE_ARCHES = ("gfx942", "gfx950")
_CUS = {"gfx942": 304, "gfx950": 256}
_SHAPES = {  # name: (seqlen_q, seqlen_k, mask_type, sliding_window, kv_block_size)
    "decode_long": (1, 4096, 2, 0, 16),
    "decode_short": (1, 256, 2, 0, 16),
    "decode_swa": (1, 4096, 2, 128, 16),
    "prefill": (2048, 2048, 1, 0, 16),
    # Block size 32: gfx942 D256 bf16 LSE no longer fits the tiled 2D LDS
    # budget, so auto hands it to split-KV.
    "prefill_bs32": (2048, 2048, 1, 0, 32),
}
# Pinned unified tuning specs: the shape that pins each, and its id per arch.
_PINS = {
    "split_kv": (
        "decode_long",
        {a: f"{a}_u3d_splitkv_seg16_t1xb" for a in _LSE_ARCHES},
    ),
    "u2d_narrow": (
        "prefill",
        {a: f"{a}_u2d_narrow_nw1_mw16_t1xb_llvm" for a in _LSE_ARCHES},
    ),
    "u2d_4warp": ("prefill", {"gfx942": "gfx942_u2d_gfx942_4warp_nw1_mw32_t64_llvm"}),
}
_SINKS = (
    "LSE output with attention sinks is not implemented on unified attention "
    "(keyless rows would drop the sink)"
)
_LABEL = {"3d": "split-KV", "2d": "tiled 2D"}
_UNIFIED = re.compile(r"attention_(unified_(2d|3d)|gfx9\d\d_u(2d|3d)_\w+)$")
_NO_FEATURES = "capability: cannot serve features "
# LSE requests that tiled 2D declines (LDS budget) and auto hands to split-KV.
# LSE off, bf16 runs the 4-warp builder and fp16 the scalar kernel; neither
# writes LSE.
_HANDOVER = frozenset(
    f"gfx942|prefill_bs32|d256|{dt}|sinks=0|paged|auto" for dt in ("bf16", "fp16")
)
# Candidates that do not declare the ``lse`` feature, by name.
_NO_LSE = frozenset(
    c.name
    for c in attention_candidates()
    if "lse" not in c.capability.supports_features
)


def _domain():
    for arch, shape, hd, dt, sinks, lse, kv, pin in itertools.product(
        _ARCHES,
        _SHAPES,
        (128, 256),
        ("bf16", "fp16"),
        (False, True),
        (False, True),
        ("paged", "strided"),
        ("auto", *_PINS),
    ):
        if pin != "auto" and (
            _PINS[pin][0] != shape or arch not in _PINS[pin][1] or kv != "paged"
        ):
            continue
        sq, sk, mask_type, window, block = _SHAPES[shape]
        fields = dict(
            batch=4, nhead_q=32, nhead_k=8, seqlen_q=sq, seqlen_k=sk,
            hdim_q=hd, hdim_v=hd, arch=arch, mask_type=mask_type,
            sliding_window=window, dtype=dt, use_sinks=sinks, emit_lse=lse,
            kv_layout=kv, num_cus=_CUS.get(arch, 128), kv_block_size=block,
        )  # fmt: skip
        if pin != "auto":
            fields.update(algorithm="unified_tuning", spec_id=_PINS[pin][1][arch])
        key = f"{arch}|{shape}|d{hd}|{dt}|sinks={int(sinks)}|{kv}|{pin}"
        yield key, lse, AttentionRequest(**fields)


def _run(request):
    """(outcome row, result or None): the served candidate and path, or the
    refusal message."""
    try:
        result = dispatch_attention(request)
    except ValueError as exc:
        return {"ok": False, "why": str(exc)}, None
    return {
        "ok": True,
        "candidate": result.kernel_id.candidate,
        "path": result.spec.path,
    }, result


def _unified_verdicts(request):
    """{candidate: (admitted, reason)} for every unified candidate."""
    return {
        c.name: c.admits(request)
        for c in attention_candidates()
        if _UNIFIED.match(c.name)
    }


def _writes_lse(result):
    spec = result.spec
    if not hasattr(spec, "kernel_spec"):  # unpinned unified AttentionSpec
        return spec.emit_lse and spec.kernel_name().endswith("_lse")
    if spec.path == "3d":
        return spec.reduce_spec.emit_lse
    return spec.kernel_spec.emit_lse and spec.builder_kind != "gfx942_4warp_gqa"


def _expected_lse_refusal(candidate, arch, sinks, off_why):
    """The reason a unified candidate must give when LSE makes it refuse, or
    None when LSE must not change its verdict. ``off_why`` is its LSE-off
    reason: the capability check lists every missing feature at once."""
    match = _UNIFIED.match(candidate)
    path = match.group(2) or match.group(3)
    if "_u2d_gfx942_4warp_" in candidate:
        missing = {"lse"}
        if off_why.startswith(_NO_FEATURES):
            missing |= set(ast.literal_eval(off_why[len(_NO_FEATURES) :]))
        return _NO_FEATURES + str(sorted(missing))
    if arch not in _LSE_ARCHES:
        return f"LSE output is not implemented on {arch} {_LABEL[path]} attention"
    if sinks:
        return _SINKS
    return None


@pytest.fixture(scope="module")
def table():
    on, off = {}, {}
    for key, lse, request in _domain():
        (on if lse else off)[key] = (
            request,
            *_run(request),
            _unified_verdicts(request),
        )
    return on, off


# The only keys whose LSE-on dispatch may leave the LSE-off candidate.
_REROUTES = ("tiled_2d_lse_handover", "no_lse_candidate")


def _reroute(off, request):
    """(name, candidate or None for any LSE writer, path) of the reroute LSE on
    must take from the LSE-off result ``off``, or None when it must not move."""
    if (
        off.spec.path == "2d"
        and select_auto_path(_problem(request), request.arch) == "3d"
    ):
        # The tiled 2D gate declines the LSE problem, so auto hands it to split-KV.
        return "tiled_2d_lse_handover", "attention_unified_3d", "3d"
    if off.kernel_id.candidate in _NO_LSE:
        # Writes no LSE (route-only dense, d256/strided decode, gfx942 4-warp), so
        # dispatch falls through to the next candidate on the same path that does.
        return "no_lse_candidate", None, off.spec.path
    return None


def _field_diffs(off, on, prefix=""):
    """Dotted names of spec fields that differ, recursing into nested specs and
    skipping ``emit_lse``."""
    if dataclasses.is_dataclass(off) and type(off) is type(on):
        return [
            name
            for f in dataclasses.fields(off)
            if f.name != "emit_lse"
            for name in _field_diffs(
                getattr(off, f.name), getattr(on, f.name), f"{prefix}{f.name}."
            )
        ]
    return [] if off == on else [prefix.rstrip(".") or type(off).__name__]


def _lse_flags(spec):
    """Every ``emit_lse`` value in ``spec`` and its nested specs."""
    if not dataclasses.is_dataclass(spec):
        return []
    return [
        flag
        for f in dataclasses.fields(spec)
        for flag in (
            [getattr(spec, f.name)]
            if f.name == "emit_lse"
            else _lse_flags(getattr(spec, f.name))
        )
    ]


def _kernel_names(spec):
    """{kernel: name} for the kernels ``spec`` launches, and which one writes
    LSE: a pinned split-KV reduce, else the main kernel."""
    names = {"kernel": spec.kernel_name()}
    reduce = getattr(spec, "reduce_spec", None)
    if spec.path == "3d" and reduce is not None:
        names["reduce"] = reduce.kernel_name()
    return names, "reduce" if "reduce" in names else "kernel"


def _lse_on_identity_diff(off, on, request):
    """Why the LSE-on dispatch is not the LSE-off one plus LSE, or None."""
    reroute = _reroute(off, request)
    if reroute:
        name, want_candidate, want_path = reroute
        got = (on.kernel_id.candidate, on.spec.path)
        if got[1] != want_path or want_candidate not in (None, got[0]):
            return f"{name}: wants {(want_candidate, want_path)}, got {got}"
        return None if _writes_lse(on) else f"{name}: {got} writes no LSE"
    off_id = (off.kernel_id.candidate, off.spec.path, off.kernel_id.tuning_id)
    on_id = (on.kernel_id.candidate, on.spec.path, on.kernel_id.tuning_id)
    if on_id != off_id:
        return f"unlisted reroute {off_id} -> {on_id}"
    names, writer = _kernel_names(off.spec)
    want = {**names, writer: kernel_name_join(names[writer], "lse")}
    if _kernel_names(on.spec)[0] != want:
        return f"kernel names {_kernel_names(on.spec)[0]}, want {want}"
    fields = _field_diffs(off.spec, on.spec)
    if fields:
        return f"spec fields differ: {fields}"
    if any(_lse_flags(off.spec)) or not any(_lse_flags(on.spec)):
        return f"emit_lse off={_lse_flags(off.spec)} on={_lse_flags(on.spec)}"
    return None


def test_lse_on_identity_is_lse_off_identity_plus_lse(table):
    """Per key, in one run: LSE on serves the LSE-off identity plus LSE, or a
    named ``_REROUTES`` entry. A key LSE off refuses stays refused; keys LSE on
    refuses are the per-path wording test's."""
    on, off = table
    assert on.keys() == off.keys()
    bad, fired = {}, set()
    for key, (request, row, result, _) in on.items():
        off_row, off_result = off[key][1], off[key][2]
        if not off_row["ok"]:
            if row["ok"]:
                bad[key] = f"LSE on serves {row}, LSE off refuses"
            continue
        if not row["ok"]:
            continue
        reroute = _reroute(off_result, request)
        if reroute:
            fired.add(reroute[0])
        why = _lse_on_identity_diff(off_result, result, request)
        if why:
            bad[key] = why
    assert not bad, bad
    # Each reroute must still describe a key in the domain, or it is dead.
    assert fired == set(_REROUTES), fired


def test_lse_on_routes_only_variants_that_write_lse(table):
    on, _ = table
    served = {k: res for k, (_, row, res, _) in on.items() if row["ok"]}
    assert served, "no LSE request was served"
    wrong = [k for k, res in served.items() if not _writes_lse(res)]
    assert not wrong, wrong


def test_lse_on_refusals_use_the_exact_per_path_wording(table):
    """Per unified candidate: LSE on may only add the per-path LSE refusal.
    A candidate that refuses with LSE off must give the same reason with it
    on, or that LSE reason; one that serves with LSE off must serve with it on
    unless the per-path rule says why not. The one exception is the split-KV
    handover (``_HANDOVER``): there the generic tiled 2D candidate refuses
    with its LDS budget and the generic split-KV candidate serves instead."""
    on, off = table
    bad = []
    for key, (request, _, _, verdicts) in on.items():
        off_verdicts = off[key][3]
        for candidate, (ok, why) in verdicts.items():
            off_ok, off_why = off_verdicts[candidate]
            expected = _expected_lse_refusal(
                candidate, request.arch, request.use_sinks, off_why
            )
            if key in _HANDOVER and candidate == "attention_unified_2d":
                good = off_ok and not ok and "LDS budget" in why
            elif key in _HANDOVER and candidate == "attention_unified_3d":
                good = not off_ok and ok
            elif not off_ok:
                good = not ok and why in (off_why, expected)
            elif expected is None:
                good = ok
            else:
                good = not ok and why == expected
            if not good:
                bad.append((key, candidate, (ok, why), expected, (off_ok, off_why)))
    assert not bad, bad[:10]


@pytest.mark.parametrize("key", sorted(_HANDOVER))
def test_gfx942_d256_bs32_lse_prefill_takes_split_kv(table, key):
    """The tiled 2D LDS gate declines this LSE problem, so auto serves it
    with split-KV and the spec's reduce writes LSE. LSE off is unchanged."""
    on, off = table
    assert (off[key][1]["candidate"], off[key][1]["path"]) == (
        "attention_unified_2d",
        "2d",
    )
    row, result = on[key][1], on[key][2]
    assert row["ok"], row
    assert (row["candidate"], row["path"]) == ("attention_unified_3d", "3d")
    assert result.spec.emit_lse and _writes_lse(result)

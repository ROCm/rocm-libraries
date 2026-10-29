# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU differential test of unified LSE dispatch over arch x path x feature.

LSE off: every outcome and identity equals the golden captured on develop
``5472e435cef``, before tiled 2D LSE existed. LSE on: a served request runs a
variant that writes LSE, and every unified path that refuses it says why in
its own words. Host only.
"""

import ast
import hashlib
import itertools
import json
import os
import pathlib
import re

import pytest

from dispatch.attention import (
    AttentionRequest,
    attention_candidates,
    dispatch_attention,
)

_GOLDEN = (
    pathlib.Path(__file__).resolve().parents[2]
    / "golden"
    / "unified_lse_domain_lse_off.json"
)
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


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


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
    """(outcome row, result or None). The row is the request's identity: the
    served candidate, path, kernel name, spec hash, tuning id and cache key, or
    the refusal message."""
    try:
        result = dispatch_attention(request)
    except ValueError as exc:
        return {"ok": False, "msg": _sha(str(exc))}, None
    spec = result.spec
    cache_key = getattr(spec, "cache_key", None)
    return {
        "ok": True,
        "candidate": result.kernel_id.candidate,
        "path": spec.path,
        "name": spec.kernel_name(),
        "spec_hash": result.kernel_id.spec_hash,
        "tuning_id": result.kernel_id.tuning_id,
        "cache_key": _sha(repr(cache_key())) if callable(cache_key) else None,
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


def test_lse_off_outcomes_and_identities_are_unchanged(table):
    _, off = table
    actual = {key: row for key, (_, row, _, _) in off.items()}
    if os.environ.get("ROCKE_REGEN_LSE_DOMAIN"):
        _GOLDEN.write_text(json.dumps(actual, indent=1, sort_keys=True) + "\n")
    golden = json.loads(_GOLDEN.read_text())
    assert actual.keys() == golden.keys()
    changed = {k: (golden[k], actual[k]) for k in golden if golden[k] != actual[k]}
    assert not changed, changed


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

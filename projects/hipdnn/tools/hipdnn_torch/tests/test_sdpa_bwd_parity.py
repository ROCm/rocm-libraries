# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""GPU backward-routing parity tests for SDPA (ALMIOPEN-2978).

Covers every GPU backward-routing scenario except the real HF attention
layers (``test_sdpa_bwd_hf.py``), the training-step gate
(``test_training_gate.py``) and autocast (``test_sdpa_autocast.py``). Like
``test_parity.py``, every test here is marked ``gpu`` and auto-skipped by
``conftest.py`` unless a wired provider + HIP device is present.

A call is *grad-eligible* (routed through the backward machinery) when
``dropout_p > 0`` or grad mode is on and q/k/v require grad; its forward row
key carries the ``"grad "`` prefix and its backward row the ``"bwd "`` prefix
(``base.GRAD_PREFIX`` / ``base.BWD_PREFIX``). Two build facts these tests
depend on cannot be assumed from the test process and are read from the job's
environment instead of the build:

  * ``HIPDNN_TORCH_TEST_FLYDSL`` -- ``"on"``/``"off"``: whether the kernel
    ingestor (FlyDSL) is loaded. Only FlyDSL serves dropout, so every test
    here that needs a SERVED dropout call skips cleanly unless this is
    ``"on"``; ASM-only builds also serve far fewer shapes, so most other
    cases pin an MHA D=128 shape ASM covers in every build state instead of
    gating on this variable.
  * ``HIPDNN_TORCH_TEST_16337`` -- ``"in"``/``"out"``: whether
    hip-kernel-provider#16337 (ASM backward at gfx950 D=64/192) is in the
    build; only the FlyDSL-OFF census table's D=64 row depends on it.
"""

import contextlib
import json
import os
import pathlib
import re
import subprocess
import sys
import threading

import pytest

pytestmark = pytest.mark.gpu

# bf16 carries ~8 bits of mantissa; f16 ~11 (test_parity.py's own tolerances).
_TOL = {
    "f16": dict(rtol=2e-2, atol=2e-2),
    "bf16": dict(rtol=4e-2, atol=4e-2),
}
_GRAD_PREFIX = "grad "
_BWD_PREFIX = "bwd "
_SUBPROC = pathlib.Path(__file__).with_name("_sdpa_bwd_subproc.py")


def _tok(torch, dtype):
    return {torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]


def _assert_parity(torch, got, want, dtype):
    tol = _TOL[_tok(torch, dtype)]
    torch.testing.assert_close(got.float(), want.float(), **tol)


def _census_key(prefix, sq, skv, d, dtype_tok):
    return f"{prefix}Sq={sq},Skv={skv},D={d},dtype={dtype_tok}"


def _grad_row(ov, sq, skv, d, dtype_tok):
    return ov.census().get(_census_key(_GRAD_PREFIX, sq, skv, d, dtype_tok))


def _bwd_row(ov, sq, skv, d, dtype_tok):
    return ov.census().get(_census_key(_BWD_PREFIX, sq, skv, d, dtype_tok))


def _reason_matching(ov, prefix):
    matches = [r for r in ov.fallback_reasons() if r.startswith(prefix)]
    assert (
        matches
    ), f"no fallback reason starting {prefix!r}; got {sorted(ov.fallback_reasons())}"
    return matches[0]


def _assert_named_engine(engines):
    assert engines, "no engine recorded on a served row"
    for e in engines:
        assert not e.startswith("0x"), f"engine name is hex, not resolved: {e}"


@contextlib.contextmanager
def _routed_sdpa():
    """Install just the sdpa override, reset its census, yield it, uninstall --
    the backward-test analogue of ``test_parity.py``'s ``_run_routed``."""
    import hipdnn_torch

    ov = hipdnn_torch.overrides()["sdpa"]
    hipdnn_torch.install(["sdpa"])
    ov.reset()
    try:
        yield ov
    finally:
        hipdnn_torch.uninstall(["sdpa"])


def _flydsl_state():
    v = os.environ.get("HIPDNN_TORCH_TEST_FLYDSL")
    return v.lower() if v else None


def _pr16337_state():
    v = os.environ.get("HIPDNN_TORCH_TEST_16337")
    return v.lower() if v else None


def _require_flydsl_on():
    if _flydsl_state() != "on":
        pytest.skip(
            "HIPDNN_TORCH_TEST_FLYDSL not 'on'; dropout is served only by "
            "FlyDSL and this case needs a SERVED dropout call"
        )


def _qkv_bshd(torch, b, sq, skv, hq, hkv, d, dtype, requires_grad=True):
    """A BSHD-view Q/K/V triple (the usual HF layout: a dense [B,S,H,D] buffer
    viewed as [B,H,S,D] via ``transpose(1, 2)``), GQA-capable (Hkv != Hq)."""
    q = torch.randn(b, sq, hq, d, device="cuda", dtype=dtype).transpose(1, 2)
    k = torch.randn(b, skv, hkv, d, device="cuda", dtype=dtype).transpose(1, 2)
    v = torch.randn(b, skv, hkv, d, device="cuda", dtype=dtype).transpose(1, 2)
    if requires_grad:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
    return q, k, v


def _clone_leaf(t):
    return t.detach().clone().requires_grad_(t.requires_grad)


_FNV_OFFSET_BASIS = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_PATH_RESERVED_STEMS = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
)
_PATH_ILLEGAL_PUNCT = frozenset(b':/\\*?"<>|')


def _sanitize_for_path(raw):
    """Exact reimplementation of
    ``hipdnn_data_sdk::utilities::sanitizeForPath`` (PathSanitizer.hpp):
    ``<sanitized stem>-<16 lowercase hex FNV-1a64 digest of raw>``. The hash
    is the standard 64-bit FNV-1a (offset 0xcbf29ce484222325, prime
    0x100000001b3) over the untruncated, unsanitized UTF-8 bytes of ``raw``;
    the stem replaces control/non-ASCII/illegal-punctuation bytes with
    ``_``, truncates to 96 bytes, strips leading/trailing dots, and breaks a
    reserved Windows device name (the part before the first dot)."""
    raw_bytes = raw.encode("utf-8")

    # fnv1aHash(data, size) special-cases size==0 to 0 (StringUtil.hpp); the
    # offset basis is only ever the SEED for a non-empty byte loop.
    h = _FNV_OFFSET_BASIS
    for b in raw_bytes:
        h ^= b
        h = (h * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    if not raw_bytes:
        h = 0

    stem_bytes = bytearray()
    for b in raw_bytes:
        if len(stem_bytes) >= 96:
            break
        stem_bytes.append(
            ord("_") if (b < 0x20 or b >= 0x7F or b in _PATH_ILLEGAL_PUNCT) else b
        )
    stem = stem_bytes.decode("ascii").strip(".")
    if not stem:
        stem = "_"
    else:
        dot = stem.find(".")
        base_len = len(stem) if dot == -1 else dot
        if stem[:base_len].upper() in _PATH_RESERVED_STEMS:
            stem = stem[:base_len] + "_" + stem[base_len:]
    return f"{stem}-{h:016x}"


def _winner_shard_has_record(path):
    """True past LineStore's bare version line (first line, LineStore.hpp)
    there is at least one non-blank record line. A non-empty shard alone
    does not prove a write: a winner-cache read miss also creates a shard
    stamped with only the version line and no records (Environment.md:408)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return any(line.strip() for line in lines[1:])


def _misaligned_like(t):
    """A tensor with ``t``'s exact shape/stride/dtype/device and VALUES,
    whose ``data_ptr()`` sits one element off a 16-byte boundary (the
    alignment ``execute`` rejects): slice a buffer one element larger by 1
    element (torch's allocator cannot have rounded back onto the boundary
    for every dtype), then copy ``t``'s values in -- an uninitialized view
    would read back as allocator garbage (occasionally NaN/Inf in bf16),
    making the parity check flaky for reasons unrelated to alignment."""
    flat = t.new_empty(t.numel() + 1)
    m = flat[1:].as_strided(t.shape, t.stride())
    m.copy_(t)
    return m


def _philox_mask_fn():
    """``hipdnn_sdpa._philox.philox_mask`` from the sibling
    ``tools/sdpa_conformance`` package (the same oracle that tool's own
    dropout tests use, Knowledge/hipdnn/sdpa/sdpa-conformance-tool.md)."""
    conformance_root = pathlib.Path(__file__).resolve().parents[2] / "sdpa_conformance"
    if str(conformance_root) not in sys.path:
        sys.path.insert(0, str(conformance_root))
    try:
        from hipdnn_sdpa._philox import philox_mask
    except ImportError as exc:
        pytest.skip(f"hipdnn_sdpa (sdpa_conformance) not importable: {exc}")
    return philox_mask


def _manual_attention_with_mask(q, k, v, keep, p, *, causal, scale=None):
    """Manual dropout attention from an explicit keep mask -- an oracle
    independent of both hipDNN and native torch. Differentiable: ``q``/``k``/
    ``v`` may be autograd leaves."""
    import torch

    d = q.shape[-1]
    scale = scale if scale is not None else 1.0 / (d**0.5)
    scores = (q.double() @ k.double().transpose(-2, -1)) * scale
    if causal:
        sq, skv = scores.shape[-2], scores.shape[-1]
        tril = torch.ones(sq, skv, dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~tril, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = attn * keep.to(attn.dtype) / (1.0 - p)
    return attn @ v.double()


def _fail_nth_execute(real_execute, n):
    """A patched ``OpOverride._execute`` that raises ``NotApplicable`` on
    exactly the ``n``-th call (1-based) and delegates to the real
    implementation otherwise -- lets the generator-rewind and backward-
    fault-injection tests target one specific graph's
    execute (fwd is always first, backward second) without reaching into
    cache-key internals. ``NotApplicable`` (not a bare exception) matters:
    the backward's native-recompute handler catches only ``except NotApplicable``
    (base.py's own fallback ladder does the same for the forward)."""
    calls = {"i": 0}

    def _patched(self, entry, variant_pack, device):
        calls["i"] += 1
        if calls["i"] == n:
            from hipdnn_torch.base import NotApplicable

            raise NotApplicable("execute: injected failure")
        return real_execute(self, entry, variant_pack, device)

    return _patched


def _run_subproc(mode, extra_env=None, unset_env=(), timeout=300):
    env = dict(os.environ)
    for key in unset_env:
        env.pop(key, None)
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, str(_SUBPROC), mode],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert (
        proc.returncode == 0
    ), f"subprocess {mode!r} exited {proc.returncode}:\n{proc.stderr}"
    return proc


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


# --------------------------------------------------------------------------- #
# Dropout mask: _philox.py oracle, optionally native as a second oracle    #
# --------------------------------------------------------------------------- #
def test_dropout_mask_matches_philox_reference_and_native_oracle(torch):
    """The override must draw -- and use -- the exact Philox stream
    AOTriton's own kernels would. Oracle: hipDNN's O AND
    dQ/dK/dV against a float64 manual-attention reference built from
    ``_philox.philox_mask`` at the seed/offset the call actually drew; if
    that mask equals torch's own ``_fill_mem_eff_dropout_mask_`` probe at the
    same state, native ``F.sdpa`` (reseeded AND re-offset to the same draw)
    is a second, independent oracle."""
    _require_flydsl_on()
    import torch.nn.functional as F

    philox_mask = _philox_mask_fn()
    device = torch.device("cuda")
    b, hq, sq, skv, d, p = 2, 4, 64, 64, 64, 0.2
    dtype = torch.bfloat16
    gen = torch.cuda.default_generators[device.index or 0]
    gen.manual_seed(12345)
    seed = gen.initial_seed()

    q, k, v = _qkv_bshd(torch, b, sq, skv, hq, hq, d, dtype)
    offset_before = int(gen.get_offset())

    with _routed_sdpa() as ov:
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        out.sum().backward()

    row = _grad_row(ov, sq, skv, d, _tok(torch, dtype))
    assert (
        row is not None and row["aot"] > 0 and row.get("dropout") == 1
    ), "dropout call did not route to a served grad row"

    keep = philox_mask(b, hq, sq, skv, p, seed, offset_before, device)

    q64 = q.detach().double().requires_grad_(True)
    k64 = k.detach().double().requires_grad_(True)
    v64 = v.detach().double().requires_grad_(True)
    want64 = _manual_attention_with_mask(q64, k64, v64, keep, p, causal=False)
    want64.sum().backward()
    torch.testing.assert_close(out.double(), want64.detach(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(q.grad.double(), q64.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k.grad.double(), k64.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(v.grad.double(), v64.grad, rtol=2e-2, atol=2e-2)

    probe = torch.zeros(b, hq, sq, skv, dtype=torch.float32, device=device)
    try:
        torch.ops.aten._fill_mem_eff_dropout_mask_(probe, p, seed, offset_before)
    except (AttributeError, RuntimeError) as exc:
        pytest.skip(f"_fill_mem_eff_dropout_mask_ unavailable: {exc}")
    native_keep = probe >= p

    if torch.equal(keep, native_keep):
        gen.manual_seed(seed)
        gen.set_offset(
            offset_before
        )  # replay the EXACT state the routed call drew from
        q2, k2, v2 = _clone_leaf(q), _clone_leaf(k), _clone_leaf(v)
        want = F.scaled_dot_product_attention(q2, k2, v2, dropout_p=p)
        _assert_parity(torch, out, want, dtype)


# --------------------------------------------------------------------------- #
# Generator offset delta / RNG routing                                    #
# --------------------------------------------------------------------------- #
def _offset_delta(gen, seed, fn):
    gen.manual_seed(seed)
    before = gen.get_offset()
    out = fn()
    after = gen.get_offset()
    return out, after - before


@pytest.mark.parametrize(
    "name, hq, hkv, d, dtype, causal, mask_kind",
    [
        ("flash_eligible_bf16", 8, 8, 64, "bf16", True, None),
        ("masked_efficient_fp32", 8, 8, 64, "f32", False, "float"),
        ("gqa_bf16", 8, 2, 64, "bf16", False, None),
    ],
)
def test_dropout_generator_offset_delta_matches_native(
    torch, name, hq, hkv, d, dtype, causal, mask_kind
):
    """The generator offset delta a call draws must equal native's delta
    from the identical starting state -- checked as a black box, by running
    native and the override from the same seed and diffing
    ``Generator.get_offset()``, not by re-deriving the backend's advance
    arithmetic (that is a separate CPU unit-test surface). Also asserts
    routing was actually SERVED when FlyDSL is on: the delta-equality check
    alone cannot distinguish "routed and matched native's advance" from "
    declined and ran the same native call twice"."""
    import torch.nn.functional as F

    torch_dtype = {"bf16": torch.bfloat16, "f32": torch.float32}[dtype]
    device = torch.device("cuda")
    gen = torch.cuda.default_generators[device.index or 0]
    b, sq, skv, p = 2, 64, 64, 0.1
    q, k, v = _qkv_bshd(torch, b, sq, skv, hq, hkv, d, torch_dtype)
    kwargs = {"dropout_p": p, "is_causal": causal, "enable_gqa": hkv != hq}
    if mask_kind == "float":
        kwargs["attn_mask"] = torch.zeros(
            b, 1, sq, skv, device="cuda", dtype=torch_dtype
        )

    _, native_delta = _offset_delta(
        gen, 2024, lambda: F.scaled_dot_product_attention(q, k, v, **kwargs)
    )
    with _routed_sdpa() as ov:
        _, routed_delta = _offset_delta(
            gen, 2024, lambda: F.scaled_dot_product_attention(q, k, v, **kwargs)
        )
        if _flydsl_state() == "on":
            aot, native = ov.totals()
            if name == "masked_efficient_fp32":
                # The split bwd engine serves f32 (including an f32 bias,
                # COVERAGE.md:2422/9.2); the merged engine declines f32
                # (DT-2/DT-3). Decline-then-fallthrough to split is the
                # heuristic framework's standard behavior (RFC 0007), but
                # which engines are actually loaded on this build/arch is
                # not independently confirmed here -- accept a named
                # decline as well as serving, so this does not assert a
                # provider-loading fact this test cannot see.
                assert (
                    aot > 0 or native > 0
                ), f"{name}: call vanished with no count at all (aot=0, native=0)"
            else:
                assert (
                    aot > 0
                ), f"{name}: expected served under FlyDSL ON (native={native})"
    assert (
        routed_delta == native_delta
    ), f"{name}: routed generator delta {routed_delta} != native's {native_delta}"


def test_dropout_offset_delta_matches_native_under_autocast(torch):
    """Fp32 inputs under bf16 autocast: the generator delta must match
    native's even when the autocast cast runs before the dropout gate."""
    import torch.nn.functional as F

    device = torch.device("cuda")
    gen = torch.cuda.default_generators[device.index or 0]
    b, hq, sq, skv, d, p = 2, 8, 64, 64, 64, 0.1
    q, k, v = _qkv_bshd(torch, b, sq, skv, hq, hq, d, torch.float32)

    def _call():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=p)

    _, native_delta = _offset_delta(gen, 2024, _call)
    with _routed_sdpa() as ov:
        _, routed_delta = _offset_delta(gen, 2024, _call)
        if _flydsl_state() == "on":
            aot, _ = ov.totals()
            assert aot > 0, "expected served under FlyDSL ON"
    assert routed_delta == native_delta


def test_dropout_ck_preferred_routes_native_with_reason(torch):
    """CK-preferred flash routes native with the CK reason: hipDNN's
    dropout RNG cannot mirror CK's generator advance."""
    try:
        ck = torch.backends.cuda.ROCmFABackend.Ck
        if torch.backends.cuda.preferred_rocm_fa_library() != ck:
            pytest.skip("this wheel's preferred ROCm FA library is not CK")
    except AttributeError:
        pytest.skip("torch.backends.cuda.preferred_rocm_fa_library unavailable")

    import torch.nn.functional as F

    q, k, v = _qkv_bshd(torch, 2, 64, 64, 8, 8, 64, torch.bfloat16)
    with _routed_sdpa() as ov:
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)
        out.sum().backward()
    aot, native = ov.totals()
    assert aot == 0 and native > 0
    _reason_matching(ov, "dropout: CK flash preferred")


def test_dropout_consecutive_calls_draw_different_masks(torch):
    """Two dropout calls on the same generator never reuse a Philox
    stream offset: each draws a fresh (seed, offset), so their keep masks
    differ."""
    import torch.nn.functional as F

    philox_mask = _philox_mask_fn()
    device = torch.device("cuda")
    gen = torch.cuda.default_generators[device.index or 0]
    gen.manual_seed(999)
    seed = gen.initial_seed()
    b, hq, sq, skv, d, p = 1, 2, 32, 32, 32, 0.3
    q, k, v = _qkv_bshd(torch, b, sq, skv, hq, hq, d, torch.bfloat16)

    with _routed_sdpa() as ov:
        off1 = int(gen.get_offset())
        F.scaled_dot_product_attention(q, k, v, dropout_p=p).sum().backward()
        off2 = int(gen.get_offset())
        F.scaled_dot_product_attention(q, k, v, dropout_p=p).sum().backward()
        off3 = int(gen.get_offset())
        if _flydsl_state() == "on":
            aot, _ = ov.totals()
            assert aot >= 2, "both dropout calls must route when FlyDSL is on"

    assert off2 != off1 and off3 != off2
    mask1 = philox_mask(b, hq, sq, skv, p, seed, off1, device)
    mask2 = philox_mask(b, hq, sq, skv, p, seed, off2, device)
    assert not torch.equal(mask1, mask2)


def test_dropout_execute_failure_rewinds_generator_offset(torch, monkeypatch):
    """A forward execute failure after the RNG draw must rewind
    (``gen.set_offset(offset_before)``) before falling back natively: the
    final offset must equal what a direct native
    call draws from the same starting state, never double-advanced -- and
    the injection must actually have fired (one native fallback counted)."""
    from hipdnn_torch.base import OpOverride

    import hipdnn_torch
    import torch.nn.functional as F

    device = torch.device("cuda")
    gen = torch.cuda.default_generators[device.index or 0]
    q, k, v = _qkv_bshd(torch, 1, 32, 32, 2, 2, 32, torch.bfloat16)
    ov = hipdnn_torch.overrides()["sdpa"]

    monkeypatch.setattr(type(ov), "_execute", _fail_nth_execute(OpOverride._execute, 1))
    hipdnn_torch.install(["sdpa"])
    ov.reset()
    try:
        gen.manual_seed(42)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)
        out.sum().backward()
        after_routed = gen.get_offset()
        aot, native = ov.totals()
        assert native > 0, "the injected failure must have produced a native fallback"
    finally:
        hipdnn_torch.uninstall(["sdpa"])

    gen.manual_seed(42)
    want = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)
    want.sum().backward()
    after_native = gen.get_offset()
    assert after_routed == after_native, (
        "a rewound-then-fallback draw must leave the same offset as a plain "
        "native call from the same starting state"
    )


# --------------------------------------------------------------------------- #
# Activation checkpointing                                                #
# --------------------------------------------------------------------------- #
def _checkpoint_run(
    torch, F, q, k, v, *, use_reentrant, with_dropout_module, dropout_p
):
    def _attn(qq, kk, vv):
        o = F.scaled_dot_product_attention(qq, kk, vv, dropout_p=dropout_p)
        if with_dropout_module:
            o = torch.nn.functional.dropout(o, p=0.1, training=True)
        return o

    out = torch.utils.checkpoint.checkpoint(_attn, q, k, v, use_reentrant=use_reentrant)
    out.sum().backward()
    return out


@pytest.mark.parametrize("use_reentrant", [True, False])
@pytest.mark.parametrize("with_dropout_module", [False, True])
def test_checkpoint_matches_no_checkpoint_run(
    torch, use_reentrant, with_dropout_module
):
    """Reentrant and non-reentrant checkpoint recompute must reach the
    same routing decision on the no_grad first pass and its grad recompute,
    replay the same dropout mask, and leave the generator at the same final
    offset as an uncheckpointed call; census shows 2 hipDNN forwards (one
    ``recompute=1``) and 1 hipDNN backward per call. dropout_p > 0 needs
    FlyDSL (only FlyDSL serves dropout)."""
    _require_flydsl_on()
    import torch.nn.functional as F

    device = torch.device("cuda")
    gen = torch.cuda.default_generators[device.index or 0]
    b, hq, sq, skv, d = 2, 4, 64, 64, 64
    dtype_tok = _tok(torch, torch.bfloat16)

    def _make():
        gen.manual_seed(2024)
        return _qkv_bshd(torch, b, sq, skv, hq, hq, d, torch.bfloat16)

    q0, k0, v0 = _make()
    with _routed_sdpa():
        out0 = F.scaled_dot_product_attention(q0, k0, v0, dropout_p=0.1)
        if with_dropout_module:
            out0 = torch.nn.functional.dropout(out0, p=0.1, training=True)
        out0.sum().backward()
    offset0 = gen.get_offset()

    q1, k1, v1 = _make()
    with _routed_sdpa() as ov1:
        out1 = _checkpoint_run(
            torch,
            F,
            q1,
            k1,
            v1,
            use_reentrant=use_reentrant,
            with_dropout_module=with_dropout_module,
            dropout_p=0.1,
        )
        offset1 = gen.get_offset()
        row1 = _grad_row(ov1, sq, skv, d, dtype_tok)
        bwd1 = _bwd_row(ov1, sq, skv, d, dtype_tok)

    _assert_parity(torch, out1, out0, torch.bfloat16)
    for g1, g0 in ((q1.grad, q0.grad), (k1.grad, k0.grad), (v1.grad, v0.grad)):
        _assert_parity(torch, g1, g0, torch.bfloat16)
    assert (
        offset1 == offset0
    ), "checkpoint must leave the generator at the plain-call offset"
    assert row1 is not None and row1["aot"] == 2 and row1.get("recompute") == 1
    assert bwd1 is not None and bwd1["aot"] == 1


def test_checkpoint_declined_shape_both_passes_native(torch):
    """A shape the probes decline (D=100: not a multiple of 8,
    HKP-H34) must send BOTH the no_grad first pass and its grad recompute
    native under checkpoint -- the decline-the-whole-call policy applies
    independently to each pass, never a half-routed checkpoint step.
    ``dropout_p=0.1`` makes EVERY call grad-eligible regardless of grad
    mode, so both passes take the identical probe-decline path; at p==0 the
    no_grad first pass would instead take the PLAIN (non-grad) key and
    could serve a D=100 non-grad forward (no dQ buffer at all), defeating
    this test's premise."""
    import torch.nn.functional as F

    q, k, v = _qkv_bshd(torch, 2, 32, 32, 4, 4, 100, torch.bfloat16)
    with _routed_sdpa() as ov:
        out = torch.utils.checkpoint.checkpoint(
            lambda qq, kk, vv: F.scaled_dot_product_attention(
                qq, kk, vv, dropout_p=0.1
            ),
            q,
            k,
            v,
            use_reentrant=True,
        )
        out.sum().backward()
    aot, native = ov.totals()
    assert (
        aot == 0 and native == 2
    ), f"expected both passes native, got aot={aot} native={native}"


def test_checkpoint_no_grad_first_pass_bit_identical_to_grad_recompute(torch):
    """The no_grad first pass and its grad recompute probe the
    identical graph key and draw the identical (seed, offset), so their O
    must be bit-identical, not merely within tolerance. Needs FlyDSL
    (dropout)."""
    _require_flydsl_on()
    import torch.nn.functional as F

    device = torch.device("cuda")
    gen = torch.cuda.default_generators[device.index or 0]
    b, hq, sq, skv, d = 2, 4, 64, 64, 64

    gen.manual_seed(777)
    q, k, v = _qkv_bshd(
        torch, b, sq, skv, hq, hq, d, torch.bfloat16, requires_grad=False
    )
    seed = gen.initial_seed()
    offset = gen.get_offset()

    with _routed_sdpa() as ov:
        with torch.no_grad():
            out_no_grad = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)

        gen.manual_seed(seed)
        gen.set_offset(offset)
        q2 = q.clone().requires_grad_(True)
        k2 = k.clone().requires_grad_(True)
        v2 = v.clone().requires_grad_(True)
        out_grad = F.scaled_dot_product_attention(q2, k2, v2, dropout_p=0.1)
        aot, _ = ov.totals()
        assert (
            aot >= 2
        ), "both passes must route for this bit-identity check to mean anything"

    assert torch.equal(out_no_grad, out_grad)


# --------------------------------------------------------------------------- #
# dO layouts unlike O                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["contiguous_bhsd", "expanded", "misaligned"])
def test_do_layout_variants_match_native(torch, kind):
    """A dense BHSD-contiguous dO (HF's ``transpose(1,2).contiguous()``),
    an expanded (self-overlapping, stride-0) dO, and a 16-byte-misaligned dO
    must all be copied into O's layout first with no execute error and
    grads matching native."""
    import torch.nn.functional as F

    b, hq, sq, skv, d = 2, 4, 32, 32, 64
    dtype = torch.bfloat16
    q, k, v = _qkv_bshd(torch, b, sq, skv, hq, hq, d, dtype)
    q2, k2, v2 = _clone_leaf(q), _clone_leaf(k), _clone_leaf(v)

    with _routed_sdpa() as ov:
        out = F.scaled_dot_product_attention(q, k, v)  # O follows Q's BSHD view
        if kind == "contiguous_bhsd":
            g = torch.randn(out.shape, device=out.device, dtype=out.dtype)
        elif kind == "expanded":
            g = torch.randn(
                out.shape[0],
                out.shape[1],
                out.shape[2],
                1,
                device=out.device,
                dtype=out.dtype,
            ).expand(out.shape)
        else:
            g = _misaligned_like(out.detach().clone())
            assert g.data_ptr() % 16 != 0
        out.backward(gradient=g)
        if _flydsl_state() == "on":
            bwd_aot, bwd_native = ov.bwd_totals()
            assert bwd_aot > 0 and bwd_native == 0, (
                "this unmasked MHA D=64 shape must be served when FlyDSL is "
                "on, or the dO-copy path is never exercised"
            )

    want = F.scaled_dot_product_attention(q2, k2, v2)
    want.backward(gradient=g)
    for grad, want_grad in ((q.grad, q2.grad), (k.grad, k2.grad), (v.grad, v2.grad)):
        _assert_parity(torch, grad, want_grad, dtype)


def test_do_size1_axis_stride_difference_is_ignored(torch):
    """A stride mismatch confined to a size-1 axis (B=1 here) must
    NOT trigger a copy or change the result -- only axes with extent > 1
    matter; the declared dO strides are O's own."""
    import torch.nn.functional as F

    hq, sq, skv, d = 4, 32, 32, 64
    dtype = torch.bfloat16
    q, k, v = _qkv_bshd(torch, 1, sq, skv, hq, hq, d, dtype)
    q2, k2, v2 = _clone_leaf(q), _clone_leaf(k), _clone_leaf(v)

    with _routed_sdpa():
        out = F.scaled_dot_product_attention(q, k, v)
        g = out.detach().clone()
        strides = list(g.stride())
        strides[0] += 997  # batch axis has extent 1: any stride there is unread
        g = g.as_strided(g.shape, strides)
        out.backward(gradient=g)

    want = F.scaled_dot_product_attention(q2, k2, v2)
    want.backward(gradient=g)
    for grad, want_grad in ((q.grad, q2.grad), (k.grad, k2.grad), (v.grad, v2.grad)):
        _assert_parity(torch, grad, want_grad, dtype)


# --------------------------------------------------------------------------- #
# GQA                                                                      #
# --------------------------------------------------------------------------- #
def test_gqa_grad_matches_native_and_keeps_hkv(torch):
    """GQA via ``enable_gqa=True``: dK/dV keep Hkv (never expanded to
    Hq); grads vs native autograd. FlyDSL OFF: no engine serves this GQA
    shape (observed: the stats forward itself declines, not specifically
    the backward), so the whole call goes native with a reason starting
    ``grad:``."""
    import torch.nn.functional as F

    b, hq, hkv, sq, skv, d = 2, 8, 2, 64, 64, 64
    q, k, v = _qkv_bshd(torch, b, sq, skv, hq, hkv, d, torch.bfloat16)
    q2, k2, v2 = _clone_leaf(q), _clone_leaf(k), _clone_leaf(v)

    with _routed_sdpa() as ov:
        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        out.sum().backward()
    assert k.grad.shape == k.shape and v.grad.shape == v.shape

    want = F.scaled_dot_product_attention(q2, k2, v2, enable_gqa=True)
    want.sum().backward()

    if _flydsl_state() == "off":
        aot, native = ov.totals()
        assert aot == 0 and native > 0
        reasons = ov.fallback_reasons()
        assert any(r.startswith("grad:") for r in reasons), reasons
        return

    if _flydsl_state() == "on":
        aot, native = ov.totals()
        assert aot > 0 and native == 0, "GQA must be served when FlyDSL is on"

    _assert_parity(torch, out, want, torch.bfloat16)
    for g, gn in ((q.grad, q2.grad), (k.grad, k2.grad), (v.grad, v2.grad)):
        _assert_parity(torch, g, gn, torch.bfloat16)


# --------------------------------------------------------------------------- #
# Masks                                                                    #
# --------------------------------------------------------------------------- #
def test_bool_mask_and_nongrad_float_mask_route(torch):
    """A bool mask materializes to an additive bias and a non-grad float
    mask passes through as-is; both route the same as unmasked (only a
    float mask that itself requires grad is exempt). FlyDSL OFF: both ASM
    builders reject any bias, so the whole call goes native."""
    import torch.nn.functional as F

    b, hq, sq, skv, d = 2, 4, 64, 64, 64
    dtype = torch.bfloat16
    for mask_kind in ("bool", "float_nongrad"):
        q, k, v = _qkv_bshd(torch, b, sq, skv, hq, hq, d, dtype)
        q2, k2, v2 = _clone_leaf(q), _clone_leaf(k), _clone_leaf(v)
        if mask_kind == "bool":
            mask = torch.ones(b, 1, sq, skv, dtype=torch.bool, device="cuda").tril()
        else:
            mask = torch.zeros(b, 1, sq, skv, dtype=dtype, device="cuda")

        with _routed_sdpa() as ov:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            out.sum().backward()

        if _flydsl_state() == "off":
            aot, native = ov.totals()
            assert aot == 0 and native > 0
            _reason_matching(ov, "grad: fwd+stats:")
            continue

        if _flydsl_state() == "on":
            aot, native = ov.totals()
            assert (
                aot > 0 and native == 0
            ), f"{mask_kind} mask must be served when FlyDSL is on"

        want = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=mask)
        want.sum().backward()
        _assert_parity(torch, out, want, dtype)
        for g, gn in ((q.grad, q2.grad), (k.grad, k2.grad), (v.grad, v2.grad)):
            _assert_parity(torch, g, gn, dtype)


def test_mask_requiring_grad_routes_native(torch):
    """A float ``attn_mask`` that itself requires grad under grad
    mode always goes native, whatever q/k/v need (learned dBias is out of
    scope)."""
    import torch.nn.functional as F

    b, hq, sq, skv, d = 2, 4, 64, 64, 64
    q, k, v = _qkv_bshd(
        torch, b, sq, skv, hq, hq, d, torch.bfloat16, requires_grad=False
    )
    mask = torch.zeros(
        b, 1, sq, skv, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )

    with _routed_sdpa() as ov:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out.sum().backward()
    aot, native = ov.totals()
    assert aot == 0 and native > 0
    _reason_matching(ov, "grad: attn_mask requires grad")


# --------------------------------------------------------------------------- #
# Declines                                                                 #
# --------------------------------------------------------------------------- #
def test_d100_declines_at_forward_stage_not_bwd(torch):
    """D=100 is not a multiple of 8 (HKP-H34) -- the fresh dense dQ
    buffer's row-padding write cannot happen, so the FORWARD stage declines
    first (D%8==0 required on CDNA); the reason must start ``grad:`` but is
    not specifically the bwd-stage reason."""
    import torch.nn.functional as F

    q, k, v = _qkv_bshd(torch, 1, 32, 32, 2, 2, 100, torch.bfloat16)
    with _routed_sdpa() as ov:
        out = F.scaled_dot_product_attention(q, k, v)
        out.sum().backward()
    aot, native = ov.totals()
    assert aot == 0 and native > 0
    reasons = ov.fallback_reasons()
    assert any(r.startswith("grad:") for r in reasons), reasons


def test_misaligned_operand_declines_before_rng_draw(torch):
    """A 16-byte-misaligned q/k/v view declines ``operand: pointer
    not 16-byte aligned`` -- and the override draws NOTHING of its own at
    p>0 before falling back (the host-side checks run before the probes and
    before the draw): the overall generator advance equals exactly a plain
    native call's own advance, since the decline falls straight through to
    it."""
    import torch.nn.functional as F

    device = torch.device("cuda")
    gen = torch.cuda.default_generators[device.index or 0]
    b, hq, sq, skv, d, p = 1, 2, 32, 32, 64, 0.1
    q = _misaligned_like(
        torch.randn(b, hq, sq, d, device="cuda", dtype=torch.bfloat16)
    ).requires_grad_(True)
    k = torch.randn(
        b, hq, skv, d, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    v = torch.randn(
        b, hq, skv, d, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    _, native_delta = _offset_delta(
        gen, 5, lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=p)
    )
    with _routed_sdpa() as ov:
        _, routed_delta = _offset_delta(
            gen, 5, lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        )
    assert routed_delta == native_delta, (
        "the decline must fall straight through to native with no extra "
        "draw of its own"
    )
    aot, native = ov.totals()
    assert aot == 0 and native > 0
    _reason_matching(ov, "operand: pointer not 16-byte aligned")


def test_seq_stride_less_than_d_declines_before_rng_draw(torch):
    """Q/k/v rows must not alias (seq stride >= D, HKP-H30); a
    narrower seq stride declines ``operand: stride``, with the same
    no-extra-draw property as the alignment decline above."""
    import torch.nn.functional as F

    device = torch.device("cuda")
    gen = torch.cuda.default_generators[device.index or 0]
    b, hq, sq, skv, d, p = 1, 2, 32, 32, 64, 0.1
    base = torch.randn(b, hq, sq, d, device="cuda", dtype=torch.bfloat16)
    q = base.as_strided(
        (b, hq, sq, d), (base.stride(0), base.stride(1), d // 2, 1)
    ).requires_grad_(True)
    k = torch.randn(
        b, hq, skv, d, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    v = torch.randn(
        b, hq, skv, d, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    _, native_delta = _offset_delta(
        gen, 5, lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=p)
    )
    with _routed_sdpa() as ov:
        _, routed_delta = _offset_delta(
            gen, 5, lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        )
    assert routed_delta == native_delta
    aot, native = ov.totals()
    assert aot == 0 and native > 0
    _reason_matching(ov, "operand: stride")


# --------------------------------------------------------------------------- #
# Backward execute fault injection                                        #
# --------------------------------------------------------------------------- #
def test_p4_fault_injection_p_zero_recomputes_natively(torch, monkeypatch, caplog):
    """A backward execute failure at p==0 recomputes natively --
    WARNING logged, ``bwd_recompute=1``, ``bwd_totals()`` native += 1, the
    grad row's aot UNCHANGED by the failure, the bwd row's aot stays 0, and
    the recompute must call the saved native function (sdpa.py's ``_native``
    closure), never the patched attribute: the grad row (the only row this
    shape/dtype key can ever address -- a re-entered call updates it, it
    never creates a second one) must carry no ``recompute`` extra. A
    buggy re-entrant call into the wrapped attribute would pick one up
    wrongly: ``recompute`` is set from ``torch._C._current_graph_task_id()
    != -1`` (sdpa.py), true for any call nested inside this test's own
    ``.backward()``, recompute or not. MHA D=128 p=0 is servable in every
    build state (with or without FlyDSL, with or without #16337), so the
    fault injection reaches a real backward ``_execute`` call regardless."""
    import logging

    from hipdnn_torch.base import OpOverride

    import hipdnn_torch
    import torch.nn.functional as F

    b, hq, sq, skv, d = 1, 8, 16, 16, 128
    q, k, v = _qkv_bshd(torch, b, sq, skv, hq, hq, d, torch.bfloat16)
    ov = hipdnn_torch.overrides()["sdpa"]

    # call 1 == the forward's execute (succeeds); call 2 == the backward's.
    monkeypatch.setattr(type(ov), "_execute", _fail_nth_execute(OpOverride._execute, 2))
    hipdnn_torch.install(["sdpa"])
    ov.reset()
    try:
        with caplog.at_level(logging.WARNING, logger="hipdnn_torch"):
            out = F.scaled_dot_product_attention(q, k, v)
            out.sum().backward()
    finally:
        hipdnn_torch.uninstall(["sdpa"])

    dtype_tok = _tok(torch, torch.bfloat16)
    grad_row = ov.census()[_census_key(_GRAD_PREFIX, sq, skv, d, dtype_tok)]
    bwd_row = ov.census()[_census_key(_BWD_PREFIX, sq, skv, d, dtype_tok)]
    assert (
        grad_row["aot"] == 1
    ), "the forward served fine; unaffected by the bwd failure"
    assert grad_row.get("recompute") is None, (
        "grad row picked up a recompute extra -- the native recompute "
        "re-entered the wrapped attribute instead of calling the saved "
        "native function directly"
    )
    assert bwd_row["aot"] == 0 and bwd_row["native"] == 1
    assert bwd_row.get("bwd_recompute") == 1
    _, bwd_native = ov.bwd_totals()
    assert bwd_native == 1
    assert any("recomputed natively" in r.message for r in caplog.records)


def test_p4_fault_injection_p_nonzero_raises(torch, monkeypatch):
    """At p>0 a backward execute failure RAISES -- a native recompute
    would use a different (non-replayed) dropout mask, so no fallback is
    attempted. Needs FlyDSL (p>0 only ever reaches a real backward execute
    via FlyDSL)."""
    _require_flydsl_on()
    from hipdnn_torch.base import OpOverride

    import hipdnn_torch
    import torch.nn.functional as F

    b, hq, sq, skv, d = 1, 8, 16, 16, 128
    q, k, v = _qkv_bshd(torch, b, sq, skv, hq, hq, d, torch.bfloat16)
    ov = hipdnn_torch.overrides()["sdpa"]

    monkeypatch.setattr(type(ov), "_execute", _fail_nth_execute(OpOverride._execute, 2))
    hipdnn_torch.install(["sdpa"])
    ov.reset()
    try:
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)
        with pytest.raises(RuntimeError):
            out.sum().backward()
    finally:
        hipdnn_torch.uninstall(["sdpa"])


# --------------------------------------------------------------------------- #
# MHA / GQA parity                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype_name", ["f16", "bf16"])
@pytest.mark.parametrize("gqa", [False, True])
def test_mha_gqa_p0_grad_matches_native(torch, gqa, dtype_name, causal):
    """P==0 MHA and GQA, causal and non-causal, bf16/fp16: dQ/dK/dV
    match native autograd. With FlyDSL ON this asserts only that SOME
    engine served and is named (ASM may win the unmasked MHA bwd at D=128,
    and with #16337 also at D=64), not a specific one. FlyDSL OFF: GQA goes
    native with a reason starting ``grad:`` (not necessarily at the bwd
    stage); MHA is served, by some named engine (not necessarily ASM)."""
    import torch.nn.functional as F

    dtype = {"f16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    b, hq, hkv, sq, skv, d = 2, 8, (2 if gqa else 8), 128, 128, 128
    q, k, v = _qkv_bshd(torch, b, sq, skv, hq, hkv, d, dtype)
    q2, k2, v2 = _clone_leaf(q), _clone_leaf(k), _clone_leaf(v)

    with _routed_sdpa() as ov:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=gqa)
        out.sum().backward()

    native_out = F.scaled_dot_product_attention(
        q2, k2, v2, is_causal=causal, enable_gqa=gqa
    )
    native_out.sum().backward()

    flydsl = _flydsl_state()
    if flydsl == "off" and gqa:
        aot, native = ov.totals()
        assert aot == 0 and native > 0
        reasons = ov.fallback_reasons()
        assert any(r.startswith("grad:") for r in reasons), reasons
        return

    _assert_parity(torch, out, native_out, dtype)
    for g, gn in ((q.grad, q2.grad), (k.grad, k2.grad), (v.grad, v2.grad)):
        _assert_parity(torch, g, gn, dtype)

    row = _grad_row(ov, sq, skv, d, _tok(torch, dtype))
    bwd = _bwd_row(ov, sq, skv, d, _tok(torch, dtype))
    assert row and row["aot"] > 0, "forward did not route"
    assert bwd and bwd["aot"] > 0, "backward did not route"
    _assert_named_engine(row.get("engines", set()))
    _assert_named_engine(bwd.get("engines", set()))


# --------------------------------------------------------------------------- #
# Subprocess: autograd worker-thread teardown                             #
# --------------------------------------------------------------------------- #
def test_subprocess_backward_exits_zero_and_bwd_row_served(torch):
    """The autograd worker-thread teardown hazard is checked by exit
    code on GPU: the process must exit 0 AND the backward must have
    actually been served (MHA D=128 p=0, servable in every build state) --
    the teardown hazard needs a real hipDNN backward graph alive on the
    worker thread, not a native fallback."""
    proc = _run_subproc("backward")
    row = json.loads(proc.stdout)["census"].get(
        _census_key(_BWD_PREFIX, 16, 16, 128, "bf16")
    )
    assert row is not None and row["aot"] > 0, row


def test_subprocess_forward_only_graph_held_on_ctx_exits_zero(torch):
    """A forward that builds (and the autograd ctx saves) the bwd graph,
    then the process exits WITHOUT ever running backward: the graph (a
    hipDNN GPU resource) stays referenced only by the ctx at interpreter
    teardown, and that teardown must not hang or crash (exit 0)."""
    proc = _run_subproc("forward_only")
    row = json.loads(proc.stdout)["census"].get(
        _census_key(_GRAD_PREFIX, 16, 16, 128, "bf16")
    )
    assert row is not None and row["aot"] + row["native"] > 0


# --------------------------------------------------------------------------- #
# Streams                                                                  #
# --------------------------------------------------------------------------- #
def test_side_stream_forward_backward_matches_native_and_binds_stream(torch):
    """A side-stream grad step must execute hipDNN work on that
    stream (``handle.get_stream()`` returns the side stream's raw handle)
    and match native. The side stream must wait for the default stream's
    writes to q/k/v before reading them."""
    import torch.nn.functional as F

    import hipdnn_torch

    q, k, v = _qkv_bshd(torch, 2, 128, 128, 8, 8, 128, torch.bfloat16)
    q2, k2, v2 = _clone_leaf(q), _clone_leaf(k), _clone_leaf(v)

    s = torch.cuda.Stream()
    s.wait_stream(
        torch.cuda.current_stream()
    )  # q/k/v were written on the default stream
    with _routed_sdpa():
        with torch.cuda.stream(s):
            out = F.scaled_dot_product_attention(q, k, v)
            out.sum().backward()
        torch.cuda.current_stream().wait_stream(s)
        st = hipdnn_torch.overrides()["sdpa"].state
        assert st.handle.get_stream() == s.cuda_stream

    native_out = F.scaled_dot_product_attention(q2, k2, v2)
    native_out.sum().backward()
    _assert_parity(torch, out, native_out, torch.bfloat16)
    for g, gn in ((q.grad, q2.grad), (k.grad, k2.grad), (v.grad, v2.grad)):
        _assert_parity(torch, g, gn, torch.bfloat16)


def test_default_stream_loop_never_calls_set_stream(torch, monkeypatch):
    """A default-stream loop must bind the stream at most once (the
    first call when ``bound_stream`` is still 0) and then never call
    ``set_stream`` again -- repeated calls on the same stream are a no-op.
    nanobind's ``Handle`` has no ``dynamic_attr()``, so an instance-level
    ``setattr`` raises; the spy must patch the class."""
    import torch.nn.functional as F

    q, k, v = _qkv_bshd(torch, 1, 16, 16, 8, 8, 128, torch.bfloat16)
    with _routed_sdpa() as ov:
        st = ov.state
        calls = []
        handle_cls = type(st.handle)
        real_set_stream = handle_cls.set_stream

        def _spy(self, sid):
            calls.append(sid)
            return real_set_stream(self, sid)

        monkeypatch.setattr(handle_cls, "set_stream", _spy)

        for _ in range(5):
            out = F.scaled_dot_product_attention(q, k, v)
            out.sum().backward()
        assert (
            len(calls) <= 1
        ), f"set_stream called more than once on one stream: {calls}"


def test_two_threads_two_streams_outputs_match_and_counts_exact(torch):
    """``st.lock``: two threads calling into the same installed
    override on two different streams must not corrupt the shared census
    counters (counted under the lock) nor each other's results. The native
    reference is the function captured BEFORE ``install()`` patches the
    module attribute -- ``_routed_sdpa()`` patches process-wide, so any call
    made while it is installed (even a differently-named local) goes
    through the override too."""
    import torch.nn.functional as F

    b, hq, sq, skv, d = 1, 8, 16, 16, 128
    dtype = torch.bfloat16
    iters = 200
    native_sdpa = F.scaled_dot_product_attention  # captured before install patches it

    def _worker(stream, results, errors):
        try:
            with torch.cuda.stream(stream):
                q, k, v = _qkv_bshd(torch, b, sq, skv, hq, hq, d, dtype)
                q2, k2, v2 = _clone_leaf(q), _clone_leaf(k), _clone_leaf(v)
                for _ in range(iters):
                    F.scaled_dot_product_attention(q, k, v).sum().backward()
                out = F.scaled_dot_product_attention(q, k, v)
                native_out = native_sdpa(q2, k2, v2)
                torch.cuda.synchronize()
                results.append((out.detach().clone(), native_out.detach().clone()))
        except Exception as exc:  # noqa: BLE001 -- surfaced on the main thread
            errors.append(exc)

    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    with _routed_sdpa() as ov:
        results, errors = [], []
        t1 = threading.Thread(target=_worker, args=(s1, results, errors))
        t2 = threading.Thread(target=_worker, args=(s2, results, errors))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors, errors
        for out, native_out in results:
            _assert_parity(torch, out, native_out, dtype)

        row = _grad_row(ov, sq, skv, d, _tok(torch, dtype))
        assert row is not None
        # iters backward calls + 1 extra routed forward (`out`) per thread;
        # `native_out` uses the captured real function, so it never counts.
        assert row["aot"] + row["native"] == 2 * (iters + 1)


# --------------------------------------------------------------------------- #
# Subprocess: force mode                                                  #
# --------------------------------------------------------------------------- #
def test_subprocess_force_asm_serves_mha_no_mask(torch):
    """Force mode + ASM_SDPA_ENGINE on bf16 D=128 p=0 MHA, no mask: both
    the fwd-with-stats graph and the bwd graph are built with ONE builder
    (ASM has both), so the whole call is served and both rows name
    ASM."""
    proc = _run_subproc("force_asm")
    payload = json.loads(proc.stdout)
    census = payload["census"]
    row = census[_census_key(_GRAD_PREFIX, 16, 16, 128, "bf16")]
    bwd = census[_census_key(_BWD_PREFIX, 16, 16, 128, "bf16")]
    assert row["aot"] == 1 and row["native"] == 0
    assert bwd["aot"] == 1 and bwd["native"] == 0
    assert row["engines"] == ["ASM_SDPA_ENGINE"]
    assert bwd["engines"] == ["ASM_SDPA_ENGINE"]


def test_subprocess_force_flydsl_bwd_engine_declines_fwd_stats(torch):
    """Force mode pins ONE engine for every graph of a call; a
    FlyDSL engine that serves only the backward makes the fwd-with-stats
    probe decline for that engine -> native with ``grad: fwd+stats:``."""
    proc = _run_subproc("force_flydsl_bwd")
    payload = json.loads(proc.stdout)
    assert payload["route"] == "native"
    assert payload["reason"].startswith("grad: fwd+stats:"), payload["reason"]


@pytest.mark.parametrize("variant", ["gqa", "mask"])
def test_subprocess_force_asm_gqa_or_mask_declines(torch, variant):
    """ASM rejects GQA (bwd) and any bias (both builders); forcing it
    for either shape declines the whole call native with a ``grad:``
    reason."""
    proc = _run_subproc(f"force_asm_{variant}")
    payload = json.loads(proc.stdout)
    assert payload["route"] == "native"
    assert payload["reason"].startswith("grad:"), payload["reason"]


# --------------------------------------------------------------------------- #
# In-place on the output                                                   #
# --------------------------------------------------------------------------- #
def test_inplace_on_output_no_custom_function_creation_error(torch):
    """The custom autograd Function returns the dense allocation, not a
    view the custom-function machinery would mark ``IN_CUSTOM_FUNCTION`` --
    an in-place op on O must not raise "created inside a custom Function"."""
    import torch.nn.functional as F

    q, k, v = _qkv_bshd(torch, 2, 128, 128, 8, 8, 128, torch.bfloat16)
    with _routed_sdpa():
        out = F.scaled_dot_product_attention(q, k, v)
        out.mul_(2.0)  # must not raise IN_CUSTOM_FUNCTION


def test_inplace_on_output_raises_at_backward_parity_with_native(torch):
    """After an in-place op on O, backward must raise for BOTH native
    (flash/efficient) and the override -- parity of failure, not a training
    success (derivatives.yaml saves ``output``; only MATH tolerates it).
    MHA, D=128, no mask, p=0: servable in every build state."""
    import torch.nn.functional as F

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        pytest.skip("torch.nn.attention.sdpa_kernel unavailable in this wheel")

    q, k, v = _qkv_bshd(torch, 2, 128, 128, 8, 8, 128, torch.bfloat16)
    q2, k2, v2 = _clone_leaf(q), _clone_leaf(k), _clone_leaf(v)
    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]

    with _routed_sdpa() as ov:
        out = F.scaled_dot_product_attention(q, k, v)
        aot, _ = ov.totals()
        assert (
            aot > 0
        ), "this shape must be served in every build state (exercises the dense-allocation path)"
        out.add_(1.0)
        with pytest.raises(RuntimeError):
            out.sum().backward()

    with sdpa_kernel(backends):
        native_out = F.scaled_dot_product_attention(q2, k2, v2)
        native_out.add_(1.0)
        with pytest.raises(RuntimeError):
            native_out.sum().backward()


def test_backward_after_nonin_place_use_works_output_is_permuted_view(torch):
    """A non-in-place use must not raise, and O must be a permuted view
    of the dense allocation ``Function.forward`` returns, not that
    allocation itself. MHA, D=128, no mask, p=0: servable in every build
    state, so the view check is meaningful regardless of which engine
    wins -- guarded on the call actually having routed (``aot > 0``)."""
    import torch.nn.functional as F

    q, k, v = _qkv_bshd(torch, 2, 128, 128, 8, 8, 128, torch.bfloat16)
    with _routed_sdpa() as ov:
        out = F.scaled_dot_product_attention(q, k, v)
        (out * 2.0).sum().backward()
        assert q.grad is not None and k.grad is not None and v.grad is not None
        aot, _ = ov.totals()

    assert (
        aot > 0
    ), "this shape must be served in every build state; the view check needs a routed call"
    assert out._base is not None, "O must be a view, not the dense allocation itself"
    assert out._base.is_contiguous()


# --------------------------------------------------------------------------- #
# Benchmarking guard                                                       #
# --------------------------------------------------------------------------- #
def test_benchmarking_guard_samples_both_graphs_and_writes_winners(torch, tmp_path):
    """A fresh, empty cache root under ``HIPDNN_FORCE_BENCHMARKING=1``
    must sample BOTH the stats-forward and backward graphs for an
    ingestor-served MHA bf16 D=128 p=0 shape with a non-grad, non-causal
    bias (both ASM builders reject any bias, so only ingestor engines can
    serve it -- a shape ASM can never claim), and leave a winner RECORD
    for EACH engine (a non-empty shard alone does not prove a write: a
    read miss also leaves a version-stamped, otherwise empty shard,
    Environment.md:408); first-execute grads must still match native.
    Runs with ``HIPDNN_DISABLE_CACHE`` unset -- it kills on-disk caching
    outright regardless of ``HIPDNN_CACHE_DIR`` and would make the shard
    assertions fail for an unrelated reason. Reads the backend's own C++
    log from ``HIPDNN_LOG_FILE`` (the alola harness always exports one; a
    console sink is used only when that var is unset, Logging.cpp:66-67,
    :249), not the subprocess's stderr."""
    if _flydsl_state() != "on":
        pytest.skip(
            "HIPDNN_TORCH_TEST_FLYDSL not 'on'; this benchmarking guard needs "
            "an ingestor "
            "engine to serve both graphs (ASM ignores benchmarking)"
        )
    cache_dir = tmp_path / "hipdnn_cache"
    log_file = tmp_path / "hipdnn.log"
    proc = _run_subproc(
        "benchmark_guard",
        extra_env={
            "HIPDNN_CACHE_DIR": str(cache_dir),
            "HIPDNN_FORCE_BENCHMARKING": "1",
            "HIPDNN_LOG_LEVEL": "info",
            "HIPDNN_LOG_FILE": str(log_file),
        },
        unset_env=["HIPDNN_DISABLE_CACHE"],
    )
    payload = json.loads(proc.stdout)
    census = payload["census"]
    grad_row = census.get(_census_key(_GRAD_PREFIX, 8, 8, 128, "bf16"))
    bwd_row = census.get(_census_key(_BWD_PREFIX, 8, 8, 128, "bf16"))
    assert grad_row and grad_row["aot"] == 2 and grad_row["native"] == 0
    assert bwd_row and bwd_row["aot"] == 2 and bwd_row["native"] == 0
    for engines in (grad_row["engines"], bwd_row["engines"]):
        assert engines and all(
            not e.startswith("0x") and e != "ASM_SDPA_ENGINE" for e in engines
        ), engines

    assert log_file.exists(), f"HIPDNN_LOG_FILE was never written: {log_file}"
    log = log_file.read_text(encoding="utf-8", errors="replace")
    assert log, f"HIPDNN_LOG_FILE is empty: {log_file}"

    # GenericPlanBuilder.hpp:528-531; a gfx950 run observed N==1 after knob
    # filtering for this exact shape (the catalog held only one applicable
    # kernel), and BenchmarkPlan still times it (1 warmup + 7 runs) and logs
    # "benchmarking selected kernel ... among 1 candidate(s)", so N counts,
    # not N>1.
    non_slotted_re = re.compile(
        r"ingestor: engine '(?P<name>[^']+)' will benchmark (?P<n>\d+) candidate\(s\)"
    )
    slotted_re = re.compile(
        r"ingestor: engine '(?P<name>[^']+)' will benchmark its slot program: "
        r"(?P<k>\d+) kernel\(s\) in uncovered slot\(s\)(?P<suffix>.*)"
    )
    selected_line_present = "ingestor: benchmarking selected kernel" in log

    engines = set(grad_row["engines"]) | set(bwd_row["engines"])
    for engine_name in engines:
        non_slotted = [
            m
            for m in non_slotted_re.finditer(log)
            if m.group("name") == engine_name and int(m.group("n")) >= 1
        ]
        slotted = [
            m
            for m in slotted_re.finditer(log)
            if m.group("name") == engine_name and int(m.group("k")) >= 1
        ]
        if non_slotted:
            assert selected_line_present, (
                f"{engine_name}: build-time sampling line seen but no "
                "first-execute selection line in the log"
            )
        elif slotted:
            for m in slotted:
                assert (
                    "a knob filter, workspace limit or shortlist limit is active"
                    not in m.group("suffix")
                ), f"{engine_name}: slotted line's filter-suffix means sampling never ran"
        else:
            pytest.fail(f"no per-engine sampling observable found for {engine_name!r}")

    shards_by_engine = {}
    for p in cache_dir.glob("ingestor-winners/*/*/*/*/*/winners.jsonl"):
        rel = p.relative_to(cache_dir / "ingestor-winners")
        engine_seg = rel.parts[1]  # .../<version>/sanitizeForPath(engine.name)/...
        shards_by_engine.setdefault(engine_seg, []).append(p)

    assert shards_by_engine, f"no winner shard written under {cache_dir}"
    for engine_name in engines:
        key = _sanitize_for_path(engine_name)
        matches = shards_by_engine.get(key, [])
        assert matches and any(_winner_shard_has_record(p) for p in matches), (
            f"no winner record for engine {engine_name!r} (expected "
            f"directory segment {key!r}, sanitizeForPath's exact scheme: "
            f"PathSanitizer.hpp) under {cache_dir}; a non-empty shard alone "
            "does not prove a write -- a read miss also leaves a "
            "version-stamped, otherwise empty shard (Environment.md:408); "
            f"found: {sorted(shards_by_engine)}"
        )

    for step in ("step1", "step2"):
        routed, native = payload[step]["routed"], payload[step]["native"]
        got = torch.tensor(routed["o"])
        want = torch.tensor(native["o"])
        torch.testing.assert_close(got, want, **_TOL["bf16"])
        for key in ("dq", "dk", "dv"):
            torch.testing.assert_close(
                torch.tensor(routed[key]), torch.tensor(native[key]), **_TOL["bf16"]
            )


# --------------------------------------------------------------------------- #
# Mixed engine pair                                                        #
# --------------------------------------------------------------------------- #
def test_subprocess_mixed_engine_pair_asm_fwd_flydsl_bwd(torch):
    """The census does not constrain engine pairs. Force mode cannot
    build this (it pins one engine for every graph); instead a
    ``HIPDNN_HEUR_CONFIG_PATH`` rule prefers ASM_SDPA_ENGINE for the FORWARD
    node only, on a GQA D=128 bf16 p=0 shape where ASM's forward builder
    accepts GQA but its backward builder rejects it
    (SdpaBwdPlanBuilder.cpp:1137-1141) -- so the bwd graph must fall through
    to some other (FlyDSL) engine. Asserts the grad row names ASM and the
    bwd row names a DIFFERENT engine, unconditionally (not "if the default
    ranking happened to pick ASM"), and that grads match native (evidence
    for a cross-engine pair). FlyDSL OFF has no GQA backward at all (ASM is
    the only bwd engine and it rejects GQA), so the bwd graph would have
    nothing to fall through to."""
    _require_flydsl_on()
    proc = _run_subproc("mixed_pair")
    payload = json.loads(proc.stdout)
    census = payload["census"]
    row = census[_census_key(_GRAD_PREFIX, 16, 16, 128, "bf16")]
    bwd = census[_census_key(_BWD_PREFIX, 16, 16, 128, "bf16")]
    assert row["aot"] > 0 and row["native"] == 0
    assert bwd["aot"] > 0 and bwd["native"] == 0
    assert row["engines"] == ["ASM_SDPA_ENGINE"], (
        f"the HIPDNN_HEUR_CONFIG_PATH rule must put ASM first for the forward "
        f"node, got {row['engines']}"
    )
    assert bwd["engines"] != ["ASM_SDPA_ENGINE"], (
        "ASM's bwd builder rejects GQA; the bwd engine must differ from the "
        f"fwd engine, got {bwd['engines']}"
    )

    routed, native = payload["routed"], payload["native"]
    got_o = torch.tensor(routed["o"])
    want_o = torch.tensor(native["o"])
    torch.testing.assert_close(got_o, want_o, **_TOL["bf16"])
    for key in ("dq", "dk", "dv"):
        torch.testing.assert_close(
            torch.tensor(routed[key]), torch.tensor(native[key]), **_TOL["bf16"]
        )


# --------------------------------------------------------------------------- #
# Capture and determinism                                                  #
# --------------------------------------------------------------------------- #
def test_dropout_under_cuda_graph_capture_routes_native(torch):
    """A p>0 call under CUDA-graph capture routes native with
    ``dropout: stream capturing`` -- unconditional: capture is declined
    before anything checks whether an engine could otherwise serve it."""
    import torch.nn.functional as F

    q, k, v = _qkv_bshd(torch, 1, 32, 32, 2, 2, 32, torch.bfloat16, requires_grad=False)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with _routed_sdpa() as ov:
        with torch.cuda.graph(graph):
            F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)
        graph.replay()
        torch.cuda.synchronize()
    aot, native = ov.totals()
    assert native > 0
    _reason_matching(ov, "dropout: stream capturing")


def test_deterministic_mode_routes_native(torch):
    """``torch.use_deterministic_algorithms(True)`` (not warn-only) on
    the grad path routes native with ``grad: deterministic algorithms`` --
    unconditional, declined before any engine probe."""
    import torch.nn.functional as F

    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        q, k, v = _qkv_bshd(torch, 2, 64, 64, 4, 4, 64, torch.bfloat16)
        with _routed_sdpa() as ov:
            out = F.scaled_dot_product_attention(q, k, v)
            out.sum().backward()
        aot, native = ov.totals()
        assert aot == 0 and native > 0
        _reason_matching(ov, "grad: deterministic algorithms")
    finally:
        torch.use_deterministic_algorithms(prev)


# --------------------------------------------------------------------------- #
# Gfx950 FlyDSL OFF expected census table                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name, hq, hkv, d, mask_kind, expect",
    [
        ("mha_d128_nomask", 8, 8, 128, None, "served"),
        ("gqa_d128_nomask", 8, 2, 128, None, "native_gqa"),
        ("mha_d128_bool_mask", 8, 8, 128, "bool", "native_fwd_stats"),
        ("mha_d64_nomask", 8, 8, 64, None, "depends_16337"),
    ],
)
def test_flydsl_off_expected_census_per_shape(
    torch, name, hq, hkv, d, mask_kind, expect
):
    """Gfx950 FlyDSL OFF: ASM serves only MHA, no attn_mask, 64<D<=128
    (or 192/128), p=0 (SdpaBwdPlanBuilder.cpp:635-638, :700-716, :427-438);
    GQA goes native with a reason starting ``grad:`` (not necessarily at
    the bwd stage: observed to decline at the fwd+stats stage on gfx950),
    any mask -> ``grad: fwd+stats:``; D=64 depends on whether
    hip-kernel-provider#16337 is in the build."""
    flydsl = _flydsl_state()
    if flydsl != "off":
        pytest.skip(
            "HIPDNN_TORCH_TEST_FLYDSL not 'off'; this test states a "
            "FlyDSL-OFF "
            "build's expected census"
        )
    if expect == "depends_16337":
        pr16337 = _pr16337_state()
        if pr16337 is None:
            pytest.skip(
                "HIPDNN_TORCH_TEST_16337 unset; D=64's FlyDSL-OFF expectation "
                "depends on whether hip-kernel-provider#16337 is in the build"
            )
        expect = "served" if pr16337 == "in" else "native_bwd"

    import torch.nn.functional as F

    b, sq, skv = 2, 64, 64
    dtype = torch.bfloat16
    q, k, v = _qkv_bshd(torch, b, sq, skv, hq, hkv, d, dtype)
    kwargs = {"enable_gqa": hkv != hq}
    if mask_kind == "bool":
        kwargs["attn_mask"] = torch.ones(b, 1, sq, skv, dtype=torch.bool, device="cuda")

    with _routed_sdpa() as ov:
        out = F.scaled_dot_product_attention(q, k, v, **kwargs)
        out.sum().backward()
    aot, native = ov.totals()

    if expect == "served":
        assert aot > 0 and native == 0
        bwd_aot, bwd_native = ov.bwd_totals()
        assert bwd_aot > 0 and bwd_native == 0
        row = _grad_row(ov, sq, skv, d, _tok(torch, dtype))
        _assert_named_engine(row["engines"])
    elif expect == "native_bwd":
        assert aot == 0 and native > 0
        _reason_matching(ov, "grad: bwd:")
    elif expect == "native_gqa":
        assert aot == 0 and native > 0
        reasons = ov.fallback_reasons()
        assert any(r.startswith("grad:") for r in reasons), reasons
    elif expect == "native_fwd_stats":
        assert aot == 0 and native > 0
        _reason_matching(ov, "grad: fwd+stats:")

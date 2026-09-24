# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU off-device dispatch tests for AICK-2146.

Verifies that the attention dispatcher makes selection decisions based solely on
``req.arch`` — the request parameter — without reading the live GPU device arch
(``_resolve_attention_arch()``).  All tests run on a CPU-only box with no GPU
and dispatch for arches that are NOT the host arch.

Definition of Done from the ticket:
- ``_resolve_attention_arch()`` is never called on the selection/geometry path.
- ``arch`` flows from the request end-to-end.
- CPU tests dispatch for an architecture that is not the host, across every arch
  the dispatcher currently declares.

The behavioural tests above drive the *selection* path only.  That left a real
gap: ``library/`` is build-time-only Python, so a builder or benchmark that
calls a gate without an ``arch`` is never imported by any GPU lane and the
``TypeError`` ships green.  ``TestEveryGateCallSiteSuppliesArch`` closes it
statically -- see that class for the full argument.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest.mock import patch

import kernels.common.attention_unified as au
from dispatch.attention import (
    AttentionRequest,
    dispatch_attention,
)
from kernels.common.attention_unified import (
    _enable_gfx942_fp16_flash,
    _enable_gfx942_bf16_flash,
    _enable_combo_2d,
    _enable_single_batch_combo,
    supports_native_unified_attention,
    supports_native_unified_attention_tiled,
)


def _make_request(arch: str, **kw) -> AttentionRequest:
    """Build a minimal AttentionRequest for ``arch``."""
    defaults = dict(
        batch=2,
        nhead_q=16,
        nhead_k=16,
        seqlen_q=512,
        seqlen_k=512,
        hdim_q=128,
        hdim_v=128,
        arch=arch,
        dtype="fp16",
    )
    defaults.update(kw)
    return AttentionRequest(**defaults)


class TestDispatchDoesNotReadLiveDevice(unittest.TestCase):
    """_resolve_attention_arch must NOT be called during selection."""

    def _assert_no_live_arch_call(self, arch: str, **kw):
        req = _make_request(arch, **kw)
        with patch.object(
            au,
            "_resolve_attention_arch",
            side_effect=AssertionError(
                f"_resolve_attention_arch() called during dispatch for arch={arch!r} — "
                "selection path must not read the live device"
            ),
        ):
            # If _resolve_attention_arch() is called on the selection path the
            # patch raises AssertionError, failing the test.
            result = dispatch_attention(req)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.spec)

    def test_gfx942_fp16_no_live_device(self):
        self._assert_no_live_arch_call("gfx942", dtype="fp16")

    def test_gfx942_bf16_no_live_device(self):
        self._assert_no_live_arch_call("gfx942", dtype="bf16")

    def test_gfx950_fp16_no_live_device(self):
        self._assert_no_live_arch_call("gfx950", dtype="fp16")

    def test_gfx950_bf16_no_live_device(self):
        self._assert_no_live_arch_call("gfx950", dtype="bf16")


class TestArchFromRequestNotDevice(unittest.TestCase):
    """Selector functions return results consistent with req.arch, not the host."""

    def _simulate_no_gpu(self):
        """Return a patcher that removes the memoized GPU arch so _resolve_attention_arch
        would fall back to "gfx950" (the default), letting tests prove that
        a DIFFERENT arch was used for selection."""
        return patch.object(au, "_RESOLVED_ATTENTION_ARCH", None)

    def test_gfx942_selector_fires_for_gfx942_request(self):
        """_enable_gfx942_fp16_flash returns True only when arch="gfx942"."""
        from kernels.common.attention_unified import UnifiedAttentionProblem

        problem = UnifiedAttentionProblem(
            total_q=1024,
            num_seqs=2,
            num_query_heads=16,
            num_kv_heads=16,
            head_size=128,
            block_size=16,
            max_seqlen_q=512,
            max_seqlen_k=512,
            dtype="fp16",
        )
        self.assertTrue(_enable_gfx942_fp16_flash(problem, "gfx942"))
        self.assertFalse(_enable_gfx942_fp16_flash(problem, "gfx950"))
        self.assertFalse(_enable_gfx942_fp16_flash(problem, "gfx1250"))

    def test_combo_2d_fires_only_for_gfx950(self):
        """_enable_combo_2d is gfx950-only and must respect the explicit arch param."""
        from kernels.common.attention_unified import UnifiedAttentionProblem

        problem = UnifiedAttentionProblem(
            total_q=2048,
            num_seqs=2,
            num_query_heads=64,
            num_kv_heads=8,
            head_size=64,
            block_size=32,
            max_seqlen_q=1024,
            max_seqlen_k=1024,
            dtype="bf16",
            use_sinks=True,
        )
        self.assertTrue(_enable_combo_2d(problem, "gfx950"))
        self.assertFalse(_enable_combo_2d(problem, "gfx942"))
        self.assertFalse(_enable_combo_2d(problem, "gfx1250"))

    def test_supports_tiled_uses_request_arch(self):
        """supports_native_unified_attention_tiled uses the passed arch, not the device.

        Uses a gfx942-only shape (_enable_gfx942_fp16_flash fires for "gfx942"
        but not "gfx950") to prove the arch parameter actually changes the result.
        """
        from kernels.common.attention_unified import UnifiedAttentionProblem

        # fp16 hd128 bs16 prefill: selects the gfx942 flash/ring path on gfx942
        # (see _enable_gfx942_fp16_flash). On gfx950 the 16x16x16 tiled path is
        # used instead -- the two arches pick distinct dispatch paths, so
        # ok_gfx942 != ok_gfx950 for at least one of the paths.
        problem = UnifiedAttentionProblem(
            total_q=1024,
            num_seqs=2,
            num_query_heads=16,
            num_kv_heads=16,
            head_size=128,
            block_size=16,
            max_seqlen_q=512,
            max_seqlen_k=512,
            dtype="fp16",
        )
        with self._simulate_no_gpu():
            ok_gfx942, _ = supports_native_unified_attention_tiled(problem, "gfx942")
            ok_gfx950, _ = supports_native_unified_attention_tiled(problem, "gfx950")
            ok_gfx1250, _ = supports_native_unified_attention_tiled(problem, "gfx1250")
        # The selector must return concrete booleans (not raise).
        self.assertIsInstance(ok_gfx942, bool)
        self.assertIsInstance(ok_gfx950, bool)
        self.assertIsInstance(ok_gfx1250, bool)
        # At least two arches must differ: proves arch actually affects selection.
        arch_results = {ok_gfx942, ok_gfx950, ok_gfx1250}
        self.assertGreater(
            len(arch_results),
            1,
            "supports_native_unified_attention_tiled returned identical results for "
            "gfx942/gfx950/gfx1250 — arch parameter is not being used",
        )
        # gfx942 fp16 hd128 tiled is supported (flash/ring path).
        self.assertTrue(ok_gfx942, "gfx942 fp16 hd128 tiled should be supported")

    def test_dispatch_for_non_host_arch_succeeds(self):
        """Dispatching for a non-host arch completes without touching the GPU."""
        # Simulate CPU-only box: clear the memoized arch AND make any real GPU
        # lookup raise so the test fails fast if the dispatch path falls through.
        with (
            patch.object(au, "_RESOLVED_ATTENTION_ARCH", None),
            patch.object(
                au,
                "_resolve_attention_arch",
                side_effect=AssertionError(
                    "live GPU read during dispatch — must not happen on selection path"
                ),
            ),
        ):
            for arch in ("gfx942", "gfx950", "gfx1250"):
                with self.subTest(arch=arch):
                    req = _make_request(arch, dtype="fp16")
                    result = dispatch_attention(req)
                    self.assertIsNotNone(result.spec)

    def test_bf16_gfx942_dispatch_no_device_read(self):
        """bf16 gfx942 dispatch uses request arch, not live device."""
        with patch.object(
            au,
            "_resolve_attention_arch",
            side_effect=AssertionError("live GPU read on selection path"),
        ):
            req = _make_request("gfx942", dtype="bf16", seqlen_q=1024, seqlen_k=2048)
            result = dispatch_attention(req)
            self.assertIsNotNone(result.spec)

    def test_gfx950_dispatch_no_device_read(self):
        """gfx950 dispatch (combo/transposed path) uses request arch."""
        with patch.object(
            au,
            "_resolve_attention_arch",
            side_effect=AssertionError("live GPU read on selection path"),
        ):
            req = _make_request(
                "gfx950",
                dtype="bf16",
                batch=2,
                nhead_q=64,
                nhead_k=8,
                seqlen_q=1024,
                seqlen_k=1024,
                hdim_q=64,
                hdim_v=64,
            )
            result = dispatch_attention(req)
            self.assertIsNotNone(result.spec)

    def test_gfx1250_dispatch_no_device_read(self):
        """gfx1250 dispatch uses request arch, not live device."""
        with patch.object(
            au,
            "_resolve_attention_arch",
            side_effect=AssertionError("live GPU read on selection path"),
        ):
            req = _make_request(
                "gfx1250",
                dtype="fp16",
                batch=2,
                nhead_q=16,
                nhead_k=16,
                seqlen_q=512,
                seqlen_k=512,
                hdim_q=128,
                hdim_v=128,
            )
            result = dispatch_attention(req)
            self.assertIsNotNone(result.spec)


# =====================================================================
# Static guard: every call site of an arch-gated selector supplies arch
# =====================================================================

# Anchored on the module object, never on a repo-relative literal.
_GATE_MODULE = Path(au.__file__).resolve()
_LIBRARY_ROOT = _GATE_MODULE.parents[2]

# Gates whose ``arch`` still carries a default. A default is what lets a caller
# omit the arch and silently get some other box's answer, so this set is a
# RATCHET: entries may be removed as gates are tightened, never added. A new
# name here means a new implicit host dependency was introduced.
_ARCH_IS_OPTIONAL = frozenset(
    {
        "_enable_gfx942_flash_k_sliced_ldsseq",
        "_gfx942_bf16_wide_geometry",
        "_gfx942_bf16_wide_tile_size",
        "_select_gfx942_flash_num_warps",
        "attention_3d_workspace_nbytes",
        "build_unified_attention_2d",
        "build_unified_attention_3d",
        "build_unified_attention_reduce",
        "supports_native_unified_attention",
    }
)


def _gate_signatures():
    """Partition ``attention_unified``'s top-level functions that take ``arch``.

    Returns ``(required, optional)``, each mapping a function name to the
    positional index of its ``arch`` parameter (``None`` when keyword-only).
    """
    tree = ast.parse(_GATE_MODULE.read_text(encoding="utf-8"), str(_GATE_MODULE))
    required, optional = {}, {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        positional = [p.arg for p in args.posonlyargs] + [p.arg for p in args.args]
        kwonly = [p.arg for p in args.kwonlyargs]
        if "arch" in positional:
            index = positional.index("arch")
            # Defaults right-align onto the positional list.
            has_default = index >= len(positional) - len(args.defaults)
        elif "arch" in kwonly:
            index = None
            has_default = args.kw_defaults[kwonly.index("arch")] is not None
        else:
            continue
        (optional if has_default else required)[node.name] = index
    return required, optional


def _called_name(func):
    """Bare name for a call target: ``f(...)`` and ``au.f(...)`` both give ``f``."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _supplies_arch(call, positional_index):
    """Whether ``call`` can be shown to pass ``arch``.

    ``*args`` / ``**kwargs`` forwarding is undecidable here, so it counts as
    supplied -- this guard reports only what it can prove.
    """
    if any(keyword.arg is None for keyword in call.keywords):
        return True
    if any(isinstance(arg, ast.Starred) for arg in call.args):
        return True
    if any(keyword.arg == "arch" for keyword in call.keywords):
        return True
    return positional_index is not None and len(call.args) > positional_index


class TestEveryGateCallSiteSuppliesArch(unittest.TestCase):
    """Static sweep: no in-tree caller of a required-arch gate omits the arch.

    Why static, and why here. Making ``arch`` required turns every missed call
    site into a ``TypeError`` at the moment it is reached -- which is the right
    failure, but only if something reaches it. Most of ``library/`` is
    build-time-only Python that no GPU lane imports, and the one production
    caller that *is* reached sits on the launch path, which the behavioural
    tests above deliberately do not cover. That combination is how a broken 3D
    launch shipped through green CI. Parsing beats importing for the same
    reason: it needs no GPU, no torch, and no module to be import-clean.

    Matching is by bare name against the top-level functions of
    ``attention_unified``, so a same-named method elsewhere in the tree would
    be a false positive. There are none today; if one appears, rename it or
    give this guard a skip list rather than loosening the signature.
    """

    def _sweep(self, gates):
        misses, unparseable = [], []
        for path in sorted(_LIBRARY_ROOT.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
            except SyntaxError as exc:
                unparseable.append(f"{path.relative_to(_LIBRARY_ROOT)}: {exc}")
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _called_name(node.func)
                if name in gates and not _supplies_arch(node, gates[name]):
                    misses.append(
                        f"{path.relative_to(_LIBRARY_ROOT)}:{node.lineno} "
                        f"{name}() -- no arch"
                    )
        return misses, unparseable

    def test_gate_set_is_populated(self):
        """A vacuous sweep passes for free, so pin that the gates were found."""
        required, _ = _gate_signatures()
        self.assertGreater(
            len(required),
            40,
            "expected dozens of required-arch gates in attention_unified; found "
            f"{len(required)} -- the sweep is not looking at what it thinks it is",
        )

    def test_optional_arch_gates_are_only_the_known_ones(self):
        """Ratchet: a gate may lose its arch default, never gain one."""
        _, optional = _gate_signatures()
        self.assertEqual(
            set(optional),
            set(_ARCH_IS_OPTIONAL),
            "the set of gates with a defaulted arch changed. Removing a name is "
            "the intended direction -- drop it from _ARCH_IS_OPTIONAL too. Adding "
            "one re-introduces an implicit host dependency and is what this "
            "guard exists to block",
        )

    def test_no_call_site_omits_a_required_arch(self):
        required, _ = _gate_signatures()
        misses, unparseable = self._sweep(required)
        self.assertEqual(
            unparseable, [], f"unparseable sources under {_LIBRARY_ROOT.name}/"
        )
        self.assertEqual(
            misses,
            [],
            "these call sites raise TypeError the moment they are reached -- the "
            "gate requires an arch and none is passed:\n  " + "\n  ".join(misses),
        )


if __name__ == "__main__":
    unittest.main()

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the VGPR-budget guard in ``KernelWriter._initKernel``.

The VGPR allocator used to abort code generation with ``RuntimeError`` the
moment the register count exceeded ``regCaps["MaxVgpr"]``. That abort fired
inside ``_initKernel``, long before ``checkResources()`` runs, so it bypassed
the whole solution-rejection pipeline: ``ForceGenerateKernel=1`` (which exists
precisely to keep the generated source of a rejected kernel for inspection)
could never take effect for a VGPR overflow. The guard now only warns, and the
overflow is reported through the normal path instead:

    vgprAllocationImplClassic  warns, keeps the real totalVgprs
      -> checkResources        sets overflowedResources = 1 ("too many vgprs")
      -> _getKernelSource      raises, unless ForceGenerateKernel=1

so an oversized kernel is still rejected by default and is only allowed
through when the user explicitly asked for it.

The guard lives in ``vgprAllocationImplClassic``, a closure nested inside the
~2000-line ``_initKernel``, and cannot be called on its own. Rather than
restating the condition (which would test a copy, not the shipped code), these
tests locate the real statements in the real source via the AST and execute
them against a stub writer.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

_TENSILE = Path(__file__).resolve().parents[2]
_KERNEL_WRITER_PY = _TENSILE / "KernelWriter.py"
_KERNEL_WRITER_ASM_PY = _TENSILE / "KernelWriterAssembly.py"


# ---------------------------------------------------------------------------
# Locating and running the real source
# ---------------------------------------------------------------------------

def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _funcDef(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError("%s not found in %s" % (name, path.name))


def _ifs(node: ast.AST):
    return [n for n in ast.walk(node) if isinstance(n, ast.If)]


def _mentionsAttr(node: ast.AST, attr: str) -> bool:
    return any(isinstance(n, ast.Attribute) and n.attr == attr for n in ast.walk(node))


def _mentionsConst(node: ast.AST, value) -> bool:
    return any(isinstance(n, ast.Constant) and n.value == value for n in ast.walk(node))


def _exec(node: ast.stmt, namespace: dict) -> dict:
    """Run a single statement lifted out of the production source."""
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(_KERNEL_WRITER_PY), "exec"), namespace)
    return namespace


def _vgprGuard() -> ast.If:
    """The ``totalVgprs`` budget check in vgprAllocationImplClassic."""
    fn = _funcDef(_KERNEL_WRITER_PY, "vgprAllocationImplClassic")
    guards = [n for n in _ifs(fn)
              if _mentionsAttr(n.test, "totalVgprs") and _mentionsConst(n.test, "MaxVgpr")]
    assert len(guards) == 1, "expected exactly one totalVgprs budget check"
    return guards[0]


def _agprGuard() -> ast.If:
    """The sibling ``totalAgprs`` budget check, which must still be fatal."""
    fn = _funcDef(_KERNEL_WRITER_PY, "vgprAllocationImplClassic")
    guards = [n for n in _ifs(fn)
              if _mentionsAttr(n.test, "totalAgprs")
              and any(isinstance(b, ast.Raise) for b in n.body)]
    assert len(guards) == 1, "expected exactly one fatal totalAgprs budget check"
    return guards[0]


def _writer(totalVgprs=0, maxVgpr=256, totalAgprs=0):
    return SimpleNamespace(states=SimpleNamespace(
        totalVgprs=totalVgprs,
        totalAgprs=totalAgprs,
        regCaps={"MaxVgpr": maxVgpr, "PhysicalMaxVgpr": 512},
    ))


# ---------------------------------------------------------------------------
# The guard itself: warns instead of aborting
# ---------------------------------------------------------------------------

class TestVgprOverflowIsNotFatal:
    def test_overflow_does_not_abort(self, capsys):
        # The point of the change: exceeding MaxVgpr must not kill codegen.
        _exec(_vgprGuard(), {"self": _writer(totalVgprs=300, maxVgpr=256)})
        assert capsys.readouterr().out.strip(), "the overflow must still be reported"

    def test_overflow_warning_reports_both_counts(self, capsys):
        # Diagnosing an overflowing tuning config needs the actual numbers, not
        # just "it overflowed".
        _exec(_vgprGuard(), {"self": _writer(totalVgprs=300, maxVgpr=256)})
        out = capsys.readouterr().out
        assert "300" in out and "256" in out, (
            "the warning must name the requested and the allowed VGPR count, got %r" % out
        )

    def test_negative_count_also_warns_only(self, capsys):
        # The `< 0` half of the condition is a can't-happen sanity check; it is
        # downgraded along with the overflow half rather than left as a raise.
        _exec(_vgprGuard(), {"self": _writer(totalVgprs=-1, maxVgpr=256)})
        assert capsys.readouterr().out.strip()

    def test_kernel_within_budget_stays_silent(self, capsys):
        # No spurious warnings for the overwhelmingly common in-budget case,
        # including exactly at the limit (the check is `>`, not `>=`).
        _exec(_vgprGuard(), {"self": _writer(totalVgprs=256, maxVgpr=256)})
        assert capsys.readouterr().out == ""

    def test_guard_body_contains_no_raise(self):
        # Anti-regression on the shape of the code, not just its behaviour:
        # nobody may reinstate the abort inside the guard.
        assert not [n for n in ast.walk(_vgprGuard()) if isinstance(n, ast.Raise)], (
            "the VGPR budget guard must not raise; the overflow is reported by "
            "checkResources so ForceGenerateKernel can still save the source"
        )


# ---------------------------------------------------------------------------
# Scope: only the VGPR guard was softened
# ---------------------------------------------------------------------------

class TestBlastRadius:
    def test_agpr_overflow_still_aborts(self):
        # The AGPR budget is a hard limit with no downstream rejection path,
        # so it must keep raising.
        with pytest.raises(RuntimeError):
            _exec(_agprGuard(), {"self": _writer(totalAgprs=300), "agprLimit": 256})

    def test_agpr_guard_ignores_kernels_within_budget(self):
        _exec(_agprGuard(), {"self": _writer(totalAgprs=256), "agprLimit": 256})

    def test_real_vgpr_count_is_still_recorded(self):
        # The guard must not clamp or discard totalVgprs: checkResources reads
        # the pool size later and can only flag the overflow if the true count
        # survives.
        fn = _funcDef(_KERNEL_WRITER_PY, "vgprAllocationImplClassic")
        assigns = [n for n in ast.walk(fn)
                   if isinstance(n, ast.Assign)
                   and any(_mentionsAttr(t, "totalVgprs") for t in n.targets)]
        assert assigns, "totalVgprs is never assigned"
        assert all(a.lineno < _vgprGuard().lineno for a in assigns), (
            "totalVgprs must be settled before the guard, and not rewritten by it"
        )


# ---------------------------------------------------------------------------
# The overflow is still caught, just later and overridably
# ---------------------------------------------------------------------------

class TestOverflowStillRejectsTheKernel:
    def _checkResourcesGuard(self) -> ast.If:
        fn = _funcDef(_KERNEL_WRITER_ASM_PY, "checkResources")
        guards = [n for n in _ifs(fn)
                  if _mentionsAttr(n.test, "vgprPool") and _mentionsConst(n.test, "MaxVgpr")]
        assert len(guards) == 1, "expected exactly one VGPR check in checkResources"
        return guards[0]

    def _resourceWriter(self, vgprs, maxVgpr=256):
        return SimpleNamespace(
            vgprPool=SimpleNamespace(size=lambda: vgprs),
            sgprPool=SimpleNamespace(size=lambda: 0),
            states=SimpleNamespace(overflowedResources=0,
                                   regCaps={"MaxVgpr": maxVgpr, "MaxSgpr": 102}),
        )

    def test_checkResources_flags_the_overflow(self):
        # This is the rejection the removed raise used to pre-empt: error code
        # 1 == "too many vgprs".
        writer = self._resourceWriter(vgprs=300)
        _exec(self._checkResourcesGuard(), {"self": writer})
        assert writer.states.overflowedResources == 1

    def test_checkResources_passes_a_fitting_kernel(self):
        writer = self._resourceWriter(vgprs=200)
        _exec(self._checkResourcesGuard(), {"self": writer})
        assert writer.states.overflowedResources == 0


class TestForceGenerateKernel:
    """``_getKernelSource`` is where an overflowing kernel now lands."""

    def _getKernelSource(self, warnings):
        namespace = {"Solution": object, "printWarning": warnings.append}
        _exec(_funcDef(_KERNEL_WRITER_PY, "_getKernelSource"), namespace)
        return namespace["_getKernelSource"]

    def _writer(self, error, forceGenerateKernel):
        return SimpleNamespace(
            _initKernel=lambda kernel, tPA, tPB: None,
            stringIdx=0,
            kernelBody=lambda kernel, tPA, tPB: (error, "s_endpgm\n"),
            kernelBodySubtile=lambda kernel, tPA, tPB: (error, "s_endpgm\n"),
            debugConfig=SimpleNamespace(forceGenerateKernel=forceGenerateKernel),
        )

    def test_overflowing_kernel_is_rejected_by_default(self):
        # Default behaviour is unchanged from before the fix: an overflowing
        # kernel still fails the build, just from here instead of _initKernel.
        getKernelSource = self._getKernelSource([])
        with pytest.raises(RuntimeError):
            getKernelSource(self._writer(error=1, forceGenerateKernel=False),
                            {"UseSubtileImpl": False})

    def test_force_generate_kernel_keeps_the_source(self):
        # The behaviour the early raise made unreachable.
        warnings = []
        getKernelSource = self._getKernelSource(warnings)
        source = getKernelSource(self._writer(error=1, forceGenerateKernel=True),
                                 {"UseSubtileImpl": False})
        assert "s_endpgm" in source
        assert any("ForceGenerateKernel" in str(w) for w in warnings), (
            "saving the source of a rejected kernel must be announced"
        )

    def test_clean_kernel_is_unaffected(self):
        warnings = []
        getKernelSource = self._getKernelSource(warnings)
        source = getKernelSource(self._writer(error=0, forceGenerateKernel=False),
                                 {"UseSubtileImpl": False})
        assert "s_endpgm" in source
        assert warnings == []

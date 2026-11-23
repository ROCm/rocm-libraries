# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The barrier-fence parity emitter (``tests/parity/barrier_fence_emit.py``).

The emitter is the Python reference of the barrier ``fence`` attribute: the
six-barrier probe of ``platform/tests/core/test_barrier_fence.{py,cpp}`` on
every target, the builder default fence, and attention backward kernels whose
barriers all carry ``fence="compiler"``. Checked here, per config and LLVM
flavor:

* ``ll`` (a separate process, as the differential harness runs it) is the
  in-process lowering of the same kernel, byte for byte;
* every ``s.barrier`` call has exactly the emission its fence asks for (the
  probe's per-barrier list; the backward kernels: the compiler memory clobber
  after every barrier and no workgroup fence);
* ``ir`` parses back to a kernel that lowers to the same ``ll`` (the fence
  attribute survives the serialization the C++ engine consumes);
* ``verify`` reports nothing, and the index past the last config is the
  harness's end-of-range answer;
* with ``rocke_engine`` built, the C++ engine lowers the ``ir`` output to the
  same bytes as ``ll`` (the cross-engine check; skipped without the engine).

There is no ``barrier_fence_emit.c``: the backward family is Python-only (the
C++-engine waiver in ``dsl_docs/instances/attention_bwd.md``). The C++
engine's fence emission is unit-tested by
``platform/tests/core/test_barrier_fence.cpp``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent
_PARITY = _TESTS / "parity"
_LIB = _TESTS.parent
_PLATFORM_PY = _LIB.parent / "platform" / "python"
_EMITTER = _PARITY / "barrier_fence_emit.py"
FLAVORS = ("llvm20", "llvm22", "llvm23")

REL = '  fence syncscope("workgroup") release'
BAR = " call void @llvm.amdgcn.s.barrier()"
ACQ = '  fence syncscope("workgroup") acquire'
CLOB = '  call void asm sideeffect "", "~{memory}"()'


@cache
def _emitter():
    if str(_PARITY) not in sys.path:
        sys.path.insert(0, str(_PARITY))
    import barrier_fence_emit

    return barrier_fence_emit


CONFIGS = tuple(sorted(_emitter()._CONFIGS))


def _run(idx, mode: str, flavor: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["ROCKE_LLVM_FLAVOR"] = flavor
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_LIB), str(_PLATFORM_PY)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    return subprocess.run(
        [sys.executable, str(_EMITTER), str(idx), mode],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_PARITY),
        timeout=600,
    )


@cache
def _out(idx: int, mode: str, flavor: str) -> str:
    p = _run(idx, mode, flavor)
    assert p.returncode == 0, p.stderr[-3000:]
    return p.stdout


def _lower(kernel, arch: str, flavor: str) -> str:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    return _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor=flavor)


def _barrier_modes(text: str) -> list:
    """Per ``s.barrier`` call: None, "compiler", "workgroup" or "malformed"."""
    lines = text.splitlines()
    out = []
    for i, ln in enumerate(lines):
        if ln != BAR:
            continue
        prev = lines[i - 1] if i else ""
        nxt = lines[i + 1 : i + 3] + ["", ""]
        if prev == REL and nxt[0] == ACQ and nxt[1] == CLOB:
            out.append("workgroup")
        elif prev != REL and nxt[0] == CLOB:
            out.append("compiler")
        elif prev != REL and nxt[0] not in (ACQ, CLOB):
            out.append(None)
        else:
            out.append("malformed")
    return out


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("idx", CONFIGS)
def test_ll_is_the_in_process_lowering(idx, flavor):
    em = _emitter()
    spec, arch = em._spec(idx)
    want = _lower(em.build_barrier_fence_kernel(spec, arch=arch), arch, flavor)
    assert _out(idx, "ll", flavor) == want


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("idx", CONFIGS)
def test_every_barrier_has_its_fence_emission(idx, flavor):
    em = _emitter()
    spec, _arch = em._spec(idx)
    modes = _barrier_modes(_out(idx, "ll", flavor))
    if isinstance(spec, em.FenceProbeSpec) and not spec.default:
        assert modes == list(em.PROBE_MODES)
    elif isinstance(spec, em.FenceProbeSpec):
        # the default applies to every barrier; "none" and an explicit
        # "workgroup" win
        assert modes == ["compiler", "compiler", "compiler", None, "workgroup"]
    else:
        assert modes and set(modes) == {"compiler"}, modes


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("idx", CONFIGS)
def test_serialized_ir_keeps_the_fences(idx, flavor):
    from rocke.core.ir_serialize import parse

    _spec, arch = _emitter()._spec(idx)
    kernel = parse(_out(idx, "ir", "llvm22"))
    assert _lower(kernel, arch, flavor) == _out(idx, "ll", flavor)


@pytest.mark.parametrize("idx", CONFIGS)
def test_verify_is_clean(idx):
    assert _out(idx, "verify", "llvm22") == ""


def test_end_of_range_is_reported():
    p = _run(max(CONFIGS) + 1, "ll", "llvm22")
    assert p.returncode != 0 and p.stdout == ""
    assert "unknown config" in p.stderr


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("idx", CONFIGS)
def test_cpp_engine_lowers_the_serialized_ir_identically(idx, flavor):
    rocke_engine = pytest.importorskip("rocke_engine")
    _spec, arch = _emitter()._spec(idx)
    cpp = rocke_engine.lower_serialized_ir(
        _out(idx, "ir", "llvm22"), arch=arch, flavor=flavor
    )
    assert cpp == _out(idx, "ll", flavor)

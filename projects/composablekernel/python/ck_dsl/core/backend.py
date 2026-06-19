# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dual-backend dispatch for the ck_dsl authoring layer.

The same public authoring API can lower a kernel through either of two
interchangeable engines:

  - ``"python"``  the native Python lowerer (:func:`lower_kernel_to_llvm`
                  / :func:`serialize`). This is the default and is
                  byte-for-byte unchanged from the historical behaviour.

  - ``"cpp"``     the C++ engine, reached through the ``ckc_engine``
                  Python extension. The extension wraps the prebuilt
                  C engine archive and exposes the universal-GEMM
                  template family (``gemm_lower_llvm`` /
                  ``gemm_serialize_ir`` / ``gemm_verify``).

  - ``"both"``    run both engines and assert they agree, returning the
                  Python result on success and raising a precise diff on
                  mismatch. Use this as a differential gate while the two
                  engines are being kept in lock-step.

Backend selection
-----------------

The active backend is chosen, in order of precedence:

  1. an explicit ``backend=`` argument to the dispatch entry point,
  2. the ``CK_DSL_BACKEND`` environment variable, else
  3. ``"python"`` (the default; no behaviour change when nothing is set).

Engine coverage
---------------

The C++ engine binding today covers the **universal GEMM** family only.
Other op families fall back to the Python engine automatically: the
dispatch layer raises a clear, structured error if ``"cpp"``/``"both"``
is requested for a family the binding does not yet implement, so a caller
that has not opted in never sees a silent behaviour change.

Spec equivalence caveat
-----------------------

The C++ engine tracks the merge-target codegen, which can legitimately
differ from this branch's Python for families that were reconciled on the
target (e.g. the convolution family and GEMM configurations whose
scheduler hints were adjusted on the target). For those, ``"both"`` will
correctly report a mismatch until the target is merged in. ``"both"`` is
meant to be run on configurations where the two engines are expected to
agree today.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Public backend identifiers.
BACKEND_PYTHON = "python"
BACKEND_CPP = "cpp"
BACKEND_BOTH = "both"
_VALID_BACKENDS = (BACKEND_PYTHON, BACKEND_CPP, BACKEND_BOTH)

_ENV_VAR = "CK_DSL_BACKEND"


class BackendError(RuntimeError):
    """Raised for backend-selection / engine-availability failures."""


class BackendMismatch(AssertionError):
    """Raised by ``"both"`` mode when the two engines disagree.

    Carries a precise, human-readable diff describing which artifact
    (serialized IR or lowered ``.ll``) diverged and where.
    """


def resolve_backend(backend: Optional[str] = None) -> str:
    """Resolve the active backend name.

    Precedence: explicit ``backend`` argument, then the ``CK_DSL_BACKEND``
    environment variable, then ``"python"``. The result is validated
    against the known backend set.
    """
    chosen = backend
    if chosen is None:
        chosen = os.environ.get(_ENV_VAR)
    if chosen is None or chosen == "":
        chosen = BACKEND_PYTHON
    chosen = chosen.strip().lower()
    if chosen not in _VALID_BACKENDS:
        raise BackendError(
            f"unknown backend {chosen!r}; expected one of {_VALID_BACKENDS} "
            f"(via backend= argument or {_ENV_VAR} environment variable)"
        )
    return chosen


# ------------------------------------------------------------------ binding


def _import_engine():
    """Import and return the ``ckc_engine`` extension module.

    Raises :class:`BackendError` with actionable build guidance when the
    extension is not importable (it is built out-of-tree into a temporary
    directory; the caller must put that directory on ``sys.path`` or the
    package must be installed alongside the built extension).
    """
    try:
        import ckc_engine  # type: ignore
    except ImportError as e:
        raise BackendError(
            "the C++ engine extension 'ckc_engine' is not importable: "
            f"{e}. Build it from ck_dsl_c/bindings/ (see its README) and put "
            "the resulting build directory on sys.path, then retry with "
            "backend='cpp' or backend='both'."
        ) from e
    return ckc_engine


# ------------------------------------------------- spec dict serialization


def universal_gemm_spec_to_dict(spec: Any) -> Dict[str, Any]:
    """Translate a :class:`UniversalGemmSpec` into the flat dict the
    ``ckc_engine`` GEMM binding consumes.

    The binding accepts a flat key view that mirrors every
    ``TileSpec`` / ``TraitSpec`` / ``DataSpec`` field plus the top-level
    scalars. Keeping this converter in one place is the single source of
    truth for the Python-spec -> C-engine-spec mapping.
    """
    t = spec.tile
    tr = spec.trait
    d = spec.data
    return dict(
        name=spec.name,
        # tile geometry
        tile_m=t.tile_m,
        tile_n=t.tile_n,
        tile_k=t.tile_k,
        warp_m=t.warp_m,
        warp_n=t.warp_n,
        warp_k=t.warp_k,
        warp_tile_m=t.warp_tile_m,
        warp_tile_n=t.warp_tile_n,
        warp_tile_k=t.warp_tile_k,
        # trait
        pipeline=tr.pipeline,
        scheduler=tr.scheduler,
        epilogue=tr.epilogue,
        pad_m=tr.pad_m,
        pad_n=tr.pad_n,
        pad_k=tr.pad_k,
        persistent=tr.persistent,
        chiplet_swizzle=tr.chiplet_swizzle,
        chiplet_wgm=tr.chiplet_wgm,
        chiplet_num_xcds=tr.chiplet_num_xcds,
        chiplet_chunk_size=tr.chiplet_chunk_size,
        waves_per_eu=tr.waves_per_eu,
        preshuffle_b=tr.preshuffle_b,
        direct_to_lds=tr.direct_to_lds,
        dtl_cache_a=tr.dtl_cache_a,
        dtl_cache_b=tr.dtl_cache_b,
        dtl_prefetch=tr.dtl_prefetch,
        active_tile_skip=tr.active_tile_skip,
        lds_k_pad=tr.lds_k_pad,
        lds_swizzle=tr.lds_swizzle,
        # data
        dtype_a=d.dtype_a,
        dtype_b=d.dtype_b,
        dtype_c=d.dtype_c,
        dtype_acc=d.dtype_acc,
        layout=d.layout,
        # top-level scalars
        wave_size=spec.wave_size,
        block_size=spec.block_size,
        batched=spec.batched,
    )


# ---------------------------------------------------------------- diff util


def _text_diff(label: str, py_text: str, cpp_text: str, *, context: int = 3) -> str:
    """Return a precise, bounded unified-diff string for two texts.

    Reports the first divergent line range with a few lines of context on
    each side so the failure points straight at the offending op, and adds
    a one-line summary (lengths + first-mismatch line) for fast triage.
    """
    py_lines = py_text.splitlines()
    cpp_lines = cpp_text.splitlines()
    n = min(len(py_lines), len(cpp_lines))
    first = None
    for i in range(n):
        if py_lines[i] != cpp_lines[i]:
            first = i
            break
    if first is None:
        # Common prefix identical; the shorter is a strict prefix.
        first = n

    lo = max(0, first - context)
    hi = first + context + 1
    out = [
        f"{label} mismatch: python {len(py_lines)} lines / "
        f"{len(py_text)} bytes vs cpp {len(cpp_lines)} lines / "
        f"{len(cpp_text)} bytes; first divergence at line {first + 1}",
    ]
    out.append(f"  --- python[{lo + 1}:{hi}] ---")
    for i in range(lo, min(hi, len(py_lines))):
        out.append(f"  {'>' if i == first else ' '} {py_lines[i]}")
    out.append(f"  --- cpp[{lo + 1}:{hi}] ---")
    for i in range(lo, min(hi, len(cpp_lines))):
        out.append(f"  {'>' if i == first else ' '} {cpp_lines[i]}")
    return "\n".join(out)


# ----------------------------------------------------------- result bundle


@dataclass
class GemmLowerResult:
    """Outcome of a backend-dispatched universal-GEMM lowering.

    ``llvm_text`` / ``ir_text`` carry the chosen-backend artifacts.
    ``backend`` records which engine produced the returned artifacts
    (in ``"both"`` mode this is the Python result, after the equality
    check passed).
    """

    backend: str
    llvm_text: str
    ir_text: str


# -------------------------------------------------------- dispatch entry pt


def lower_universal_gemm(
    spec: Any,
    *,
    arch: str = "gfx950",
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> GemmLowerResult:
    """Lower a universal-GEMM ``spec`` through the selected backend.

    ``backend`` follows :func:`resolve_backend` precedence. The Python
    path builds the kernel and lowers it natively; the cpp path routes the
    spec dict through ``ckc_engine``; ``"both"`` runs both and raises
    :class:`BackendMismatch` on any artifact divergence.

    Set ``want_ir=True`` to also produce the serialized ``ck.dsl.ir/v1``
    text (always cross-checked under ``"both"``).
    """
    chosen = resolve_backend(backend)

    if chosen == BACKEND_PYTHON:
        ll, ir = _lower_python(spec, arch, want_ir)
        return GemmLowerResult(BACKEND_PYTHON, ll, ir)

    if chosen == BACKEND_CPP:
        ll, ir = _lower_cpp(spec, arch, want_ir)
        return GemmLowerResult(BACKEND_CPP, ll, ir)

    # ---- both: run both, compare, return python on agreement ----
    py_ll, py_ir = _lower_python(spec, arch, want_ir=True)
    cpp_ll, cpp_ir = _lower_cpp(spec, arch, want_ir=True)

    if py_ir != cpp_ir:
        raise BackendMismatch(
            "python vs cpp engine disagree for universal GEMM "
            f"'{getattr(spec, 'name', '?')}' on {arch}:\n"
            + _text_diff("serialized IR (ck.dsl.ir/v1)", py_ir, cpp_ir)
        )
    if py_ll != cpp_ll:
        raise BackendMismatch(
            "python vs cpp engine disagree for universal GEMM "
            f"'{getattr(spec, 'name', '?')}' on {arch}:\n"
            + _text_diff("lowered AMDGPU .ll", py_ll, cpp_ll)
        )
    return GemmLowerResult(BACKEND_BOTH, py_ll, py_ir if want_ir else "")


def _lower_python(spec: Any, arch: str, want_ir: bool) -> Tuple[str, str]:
    """Native Python lowering of a universal-GEMM spec."""
    from ..instances.common.gemm_universal import build_universal_gemm
    from .lower_llvm import lower_kernel_to_llvm

    kernel = build_universal_gemm(spec, arch=arch)
    ll = lower_kernel_to_llvm(kernel, arch=arch)
    ir = ""
    if want_ir:
        from .ir_serialize import serialize

        ir = serialize(kernel)
    return ll, ir


def _lower_cpp(spec: Any, arch: str, want_ir: bool) -> Tuple[str, str]:
    """C++-engine lowering of a universal-GEMM spec via ``ckc_engine``."""
    engine = _import_engine()
    sd = universal_gemm_spec_to_dict(spec)
    ll = engine.gemm_lower_llvm(sd, arch=arch)
    ir = ""
    if want_ir:
        ir = engine.gemm_serialize_ir(sd, arch=arch)
    return ll, ir

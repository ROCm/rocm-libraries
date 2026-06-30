# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dual-backend dispatch for the rocke authoring layer.

The same public authoring API can lower a kernel through either of two
interchangeable engines:

  - ``"python"``  the native Python lowerer (:func:`lower_kernel_to_llvm`
                  / :func:`serialize`). Byte-for-byte the historical
                  behaviour; selectable via ``ROCKE_BACKEND=python`` and
                  used as the differential oracle.

  - ``"cpp"``     the C++ engine, reached through the ``rocke_engine``
                  Python extension. This is now the DEFAULT lowerer for
                  Python-authored kernels: the family-agnostic
                  ``lower_serialized_ir`` endpoint lowers any kernel from
                  its serialized IR, byte-identically to the native
                  lowerer across every family. Per-family spec endpoints
                  (``gemm_lower_llvm`` / ...) remain available.

  - ``"both"``    run both engines and assert they agree, returning the
                  Python result on success and raising a precise diff on
                  mismatch. Use this as a differential gate while the two
                  engines are being kept in lock-step.

Backend selection
-----------------

The active backend is chosen, in order of precedence:

  1. an explicit ``backend=`` argument to the dispatch entry point,
  2. the ``ROCKE_BACKEND`` environment variable, else
  3. ``_DEFAULT_BACKEND`` — currently the **C++** engine (flipped from Python).
     The Python engine stays available via ``backend="python"`` /
     ``ROCKE_BACKEND=python``; the cpp path auto-falls back to Python when the
     binding is unavailable, unless ``ROCKE_CPP_STRICT=1`` is set.

Engine coverage
---------------

The default cpp path lowers Python-authored IR through
``lower_serialized_ir`` (the serialized-IR seam), which is family-agnostic:
any kernel the Python frontend builds can be lowered by the C++ engine. The
standalone C++ instance emitters (used by the hipDNN provider / C-JIT, no
Python) cover the broad instance set exercised by the byte-identity gate —
GEMM variants, convolution, attention/FMHA, MoE, norm/reduce/elementwise —
with some host-orchestration islands still stubbed (notably the fused-MoE
end-to-end launch chain; grep ``TODO(port)`` / ``ROCKE_ERR_NOTIMPL``). For
anything not covered, dispatch falls back to the Python engine, or raises a
clear, structured error under ``ROCKE_CPP_STRICT=1`` so an opted-in caller
never sees a silent behaviour change.

Spec equivalence caveat
-----------------------

The earlier reconciliation drifts where the C++ engine tracked the
merge-target codegen ahead of this branch's Python — the convolution
filter-dim naming, the GEMM ``sched.group.barrier`` scheduler hints, the
bf16 32x32 MFMA-catalog breadth, the llvm22 datalayout — have since merged
in, so the two engines now agree across families on the default gate. The
general principle still holds: ``"both"`` compares the two engines and is
meant for configurations where they are expected to agree, and it will
correctly report a mismatch if a kernel is mid-reconciliation (e.g. while a
correctness fix is being applied to both engines and one side has not yet
caught up). The byte-identity gate is the authority on what currently
agrees.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# Public backend identifiers.
BACKEND_PYTHON = "python"
BACKEND_CPP = "cpp"
BACKEND_BOTH = "both"
_VALID_BACKENDS = (BACKEND_PYTHON, BACKEND_CPP, BACKEND_BOTH)

_ENV_VAR = "ROCKE_BACKEND"

# Package default backend for Python-authored kernel lowering. Flipped from
# ``python`` to ``cpp`` after the full guarantee sweep proved the C++ engine
# byte-identical across every family and example. Overridable per call
# (``backend=`` argument) or per process (``ROCKE_BACKEND`` env). The chokepoint
# falls back to the native lowerer if the C++ engine is unavailable.
_DEFAULT_BACKEND = BACKEND_CPP


class BackendError(RuntimeError):
    """Raised for backend-selection / engine-availability failures."""


class BackendMismatch(AssertionError):
    """Raised by ``"both"`` mode when the two engines disagree.

    Carries a precise, human-readable diff describing which artifact
    (serialized IR or lowered ``.ll``) diverged and where.
    """


def resolve_backend(backend: Optional[str] = None) -> str:
    """Resolve the active backend name.

    Precedence: explicit ``backend`` argument, then the ``ROCKE_BACKEND``
    environment variable, then the package default :data:`_DEFAULT_BACKEND`
    (``"cpp"`` -- the C++ engine). The result is validated against the known
    backend set.

    The default lowerer for Python-authored kernels is the C++ engine: it is
    byte-identical to the native Python lowerer across the full family surface
    (proven by the differential gate, the random-spec fuzzer, and the
    per-example lower sweep), so the flip is behaviour-preserving. Select the
    native engine explicitly with ``ROCKE_BACKEND=python`` (it remains the
    differential oracle). If the C++ engine extension is unavailable, the
    lowering chokepoint falls back to the native lowerer automatically and
    records the reason (see :func:`lower_kernel_via_backend` /
    :func:`cpp_fallbacks`), so the result is always well-defined.
    """
    chosen = backend
    if chosen is None:
        chosen = os.environ.get(_ENV_VAR)
    if chosen is None or chosen == "":
        chosen = _DEFAULT_BACKEND
    chosen = chosen.strip().lower()
    if chosen not in _VALID_BACKENDS:
        raise BackendError(
            f"unknown backend {chosen!r}; expected one of {_VALID_BACKENDS} "
            f"(via backend= argument or {_ENV_VAR} environment variable)"
        )
    return chosen


# =====================================================================
# Family-agnostic single-kernel lowering dispatch.
#
# This is the chokepoint every Python-authored kernel reaches through
# ``lower_kernel_to_llvm``. It routes the lowering of a built ``KernelDef`` to
# the resolved backend, going through the serialized ``ck.dsl.ir/v1`` artifact
# (NOT a per-family spec dict), so it is completely family-agnostic: any kernel
# the Python front end can build is lowerable by the C++ engine.
# =====================================================================


# Every cpp-path fallback to the native lowerer is recorded here (kernel name +
# reason), so the guarantee sweep can enumerate exactly which kernels did not
# lower under the C++ engine and why. Empty on a clean run.
_CPP_FALLBACKS: List[Tuple[str, str]] = []

# A cpp->python fallback is LOUD by default (see _record_fallback). De-dup the
# stderr warning per (kernel, reason) so a repeatedly-lowered kernel doesn't
# spam, while still surfacing every distinct gap. Set ROCKE_CPP_QUIET_FALLBACK=1
# to silence (for callers that legitimately expect fallback, e.g. an arch the
# C++ engine does not implement), or ROCKE_CPP_STRICT=1 to make it an error.
_WARNED_FALLBACKS: set = set()
_ENV_QUIET_FALLBACK = "ROCKE_CPP_QUIET_FALLBACK"


def cpp_fallbacks() -> List[Tuple[str, str]]:
    """Return the recorded ``(kernel_name, reason)`` cpp-engine fallbacks.

    A non-empty list means one or more kernels could not be lowered through
    the C++ engine and were served by the native Python lowerer instead. The
    result is always well-defined; this is the audit trail for the
    user-facing guarantee.
    """
    return list(_CPP_FALLBACKS)


def reset_cpp_fallbacks() -> None:
    """Clear the recorded cpp-engine fallback log (test/sweep helper)."""
    _CPP_FALLBACKS.clear()
    _WARNED_FALLBACKS.clear()


def _record_fallback(kernel_name: str, reason: str) -> None:
    """Record a cpp->python fallback AND warn about it LOUDLY by default.

    A fallback means the C++ engine could not lower this kernel and the result
    came from the Python lowerer instead. That is a real gap (e.g. an opcode the
    family-agnostic ``lower_serialized_ir`` does not implement) and must NOT pass
    silently — a silent fallback let a broken cpp path masquerade as working.
    So emit a stderr warning (once per distinct ``(kernel, reason)``). Escalate
    to a hard error with ``ROCKE_CPP_STRICT=1``; silence with
    ``ROCKE_CPP_QUIET_FALLBACK=1`` only when fallback is genuinely expected.
    """
    _CPP_FALLBACKS.append((kernel_name, reason))
    if os.environ.get(_ENV_QUIET_FALLBACK, "").strip() in ("", "0"):
        key = (kernel_name, reason)
        if key not in _WARNED_FALLBACKS:
            _WARNED_FALLBACKS.add(key)
            print(
                f"[rocke] WARNING: backend='cpp' fell back to the Python "
                f"lowerer for kernel '{kernel_name}': {reason}. The C++ engine "
                f"could not lower this kernel; the emitted IR is the Python "
                f"engine's. Set ROCKE_CPP_STRICT=1 to treat this as an error, "
                f"or ROCKE_CPP_QUIET_FALLBACK=1 to silence.",
                file=sys.stderr,
                flush=True,
            )


# When set, the cpp path does NOT silently fall back to Python: an engine
# error is raised. The guarantee sweep sets this so a regression surfaces
# loudly instead of being masked by a fallback.
_ENV_STRICT = "ROCKE_CPP_STRICT"


def _cpp_strict() -> bool:
    return os.environ.get(_ENV_STRICT, "").strip() not in ("", "0")


def _lower_via_cpp_engine(
    ir_text: str, arch: str, llvm_flavor: Optional[str] = None
) -> str:
    """Lower serialized ``ck.dsl.ir/v1`` text through the C++ engine.

    Wraps ``rocke_engine.lower_serialized_ir`` (the family-agnostic
    parse+lower endpoint). ``llvm_flavor`` pins the intrinsic-declaration
    shape, matching the Python lowerer's ``llvm_flavor`` parameter.

    Flavor auto-resolution parity: when ``llvm_flavor is None`` we resolve
    the flavor in *Python* (via :func:`lower_llvm._resolve_llvm_flavor`)
    before handing it to the engine, rather than passing ``""`` (engine
    AUTO). The C99 engine's own AUTO resolver only consults
    ``$ROCKE_LLVM_FLAVOR`` -> ``/opt/rocm/.info/version`` -> default
    llvm22; it cannot portably introspect ``torch.version.hip``. The Python
    autodetector adds the torch step in between, which is what the bundled
    comgr actually keys off. Resolving here makes ``backend="cpp"`` pick the
    SAME flavor as ``backend="python"`` on a torch-rocm box where
    ``/opt/rocm`` is absent or a different vintage; without it the two
    backends emit non-byte-identical IR unless ``ROCKE_LLVM_FLAVOR`` is
    forced. An explicit ``llvm_flavor`` argument still overrides. On a
    torch-less box the torch step returns ``None`` and the Python resolver
    falls through to ``/opt/rocm`` / default exactly as the engine would,
    so behaviour is unchanged.

    Raises :class:`BackendError` if the engine extension is unavailable.
    """
    engine = _import_engine()
    flavor = llvm_flavor
    if flavor is None:
        from .lower_llvm import _resolve_llvm_flavor

        flavor = _resolve_llvm_flavor()
    # Validate an explicit flavor in Python, mirroring the native lowerer
    # (lower_llvm raises ``ValueError("unknown LLVM flavor ...")``), so the error
    # contract is backend-INDEPENDENT: backend="cpp" must reject a bad flavor with
    # the same ValueError as backend="python"/"both", not the engine's RuntimeError
    # (which only surfaces once the rocke_engine .so is on the path).
    from .lower_llvm import LLVM_FLAVOR_LLVM20, LLVM_FLAVOR_LLVM22

    if flavor not in (LLVM_FLAVOR_LLVM20, LLVM_FLAVOR_LLVM22):
        raise ValueError(f"unknown LLVM flavor {flavor!r}")
    try:
        return engine.lower_serialized_ir(ir_text, arch=arch, flavor=flavor or "")
    except (
        Exception
    ) as e:  # noqa: BLE001 -- augment a stale-binary footgun, then re-raise
        msg = str(e)
        if "unknown opcode" in msg:
            so = getattr(engine, "__file__", "<unknown>")
            raise type(e)(
                f"{msg}\n[rocke] hint: if this opcode exists in the current "
                f"rocke source, the loaded rocke_engine binary is STALE — rebuild "
                f"it (the .so predates the opcode). Loaded from: {so}"
            ) from e
        raise


def lower_kernel_via_backend(
    kernel: Any,
    *,
    llvm_flavor: Optional[str] = None,
    arch: Optional[str] = None,
    python_lower: Callable[..., str],
) -> str:
    """Lower a built ``KernelDef`` through the resolved backend.

    ``python_lower(kernel, llvm_flavor=, arch=)`` is the native lowerer
    (passed in to avoid an import cycle with ``lower_llvm``).

    Dispatch:

      - ``"python"`` -> native lowerer, byte-identical to historical behaviour.
      - ``"cpp"``    -> serialize the kernel and lower through the C++ engine.
        On engine unavailability or an engine-side rejection, fall back to the
        native lowerer. The fallback keeps the user-facing guarantee ("all
        examples and tests work") intact for any case the C++ engine cannot
        serve, but it is **LOUD by default**: every fallback emits a stderr
        warning (de-duped per kernel+reason) so a broken cpp path cannot
        masquerade as working. ``ROCKE_CPP_STRICT=1`` re-raises instead;
        ``ROCKE_CPP_QUIET_FALLBACK=1`` silences the warning when fallback is
        genuinely expected.
      - ``"both"``   -> lower with both and assert byte-equality, returning the
        Python result (the differential oracle). A mismatch raises
        :class:`BackendMismatch`.
    """
    chosen = resolve_backend()

    if chosen == BACKEND_PYTHON:
        return python_lower(kernel, llvm_flavor=llvm_flavor, arch=arch)

    _arch = arch or "gfx950"
    name = getattr(kernel, "name", "?")

    # The serialized artifact is the family-agnostic hand-off to the C++ engine.
    from .ir_serialize import serialize

    if chosen == BACKEND_CPP:
        try:
            ir_text = serialize(kernel)
            return _lower_via_cpp_engine(ir_text, _arch, llvm_flavor)
        except BaseException as e:  # noqa: BLE001 -- includes BackendError
            if _cpp_strict():
                raise
            _record_fallback(name, f"{type(e).__name__}: {e}")
            return python_lower(kernel, llvm_flavor=llvm_flavor, arch=arch)

    # both: differential gate.
    py_ll = python_lower(kernel, llvm_flavor=llvm_flavor, arch=arch)
    ir_text = serialize(kernel)
    cpp_ll = _lower_via_cpp_engine(ir_text, _arch, llvm_flavor)
    if py_ll != cpp_ll:
        raise BackendMismatch(
            f"python vs cpp engine disagree for kernel '{name}' on {_arch}:\n"
            + _text_diff("lowered AMDGPU .ll", py_ll, cpp_ll)
        )
    return py_ll


# ------------------------------------------------------------------ binding


def _import_engine():
    """Import and return the ``rocke_engine`` extension module.

    Raises :class:`BackendError` with actionable build guidance when the
    extension is not importable (it is built out-of-tree into a temporary
    directory; the caller must put that directory on ``sys.path`` or the
    package must be installed alongside the built extension).
    """
    try:
        import rocke_engine  # type: ignore
    except ImportError as e:
        raise BackendError(
            "the C++ engine extension 'rocke_engine' is not importable: "
            f"{e}. Build it from Cpp/bindings/ (see its README) and put "
            "the resulting build directory on sys.path, then retry with "
            "backend='cpp' or backend='both'."
        ) from e
    return rocke_engine


# ------------------------------------------------- spec dict serialization


def universal_gemm_spec_to_dict(spec: Any) -> Dict[str, Any]:
    """Translate a :class:`UniversalGemmSpec` into the flat dict the
    ``rocke_engine`` GEMM binding consumes.

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
    spec dict through ``rocke_engine``; ``"both"`` runs both and raises
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
    """C++-engine lowering of a universal-GEMM spec via ``rocke_engine``."""
    engine = _import_engine()
    sd = universal_gemm_spec_to_dict(spec)
    ll = engine.gemm_lower_llvm(sd, arch=arch)
    ir = ""
    if want_ir:
        ir = engine.gemm_serialize_ir(sd, arch=arch)
    return ll, ir


# =====================================================================
# Generalised multi-family dispatch.
#
# Every additional op family follows the same shape as the universal GEMM:
#   - a Python build + lower path (the native engine), and
#   - a ``rocke_engine`` ``<fam>_lower_llvm`` / ``<fam>_serialize_ir`` pair fed
#     a flat spec dict.
# The python/cpp/both selection, the artifact comparison, and the precise diff
# are entirely family-agnostic, so they live in one driver below. Each family
# contributes (1) a spec-dataclass -> dict converter and (2) a tiny pair of
# callables that produce the Python ``.ll`` / IR and the cpp ``.ll`` / IR.
# =====================================================================


def _gemm_subspec_to_dict(spec: Any) -> Dict[str, Any]:
    """Flatten the tile/trait pair shared by the GEMM-wrapper families
    (batched / grouped / flatmm) into the dict the binding consumes.

    The binding reads the same flat tile/trait key view as the universal GEMM,
    so this reuses those field names. The caller layers the family-specific
    scalars (``dtype``, ``batch_size`` …) on top.
    """
    t = spec.tile
    tr = spec.trait
    return dict(
        tile_m=t.tile_m,
        tile_n=t.tile_n,
        tile_k=t.tile_k,
        warp_m=t.warp_m,
        warp_n=t.warp_n,
        warp_k=t.warp_k,
        warp_tile_m=t.warp_tile_m,
        warp_tile_n=t.warp_tile_n,
        warp_tile_k=t.warp_tile_k,
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
        wave_size=spec.wave_size,
        block_size=spec.block_size,
    )


# ----------------------------- spec_to_dict ---------------------------


def batched_gemm_spec_to_dict(spec: Any) -> Dict[str, Any]:
    """:class:`BatchedGemmSpec` -> flat dict for the ``rocke_engine`` binding."""
    d = _gemm_subspec_to_dict(spec)
    d.update(name=spec.name, dtype=spec.dtype, batch_size=spec.batch_size)
    return d


def grouped_gemm_spec_to_dict(spec: Any) -> Dict[str, Any]:
    """:class:`GroupedGemmSpec` -> flat dict."""
    d = _gemm_subspec_to_dict(spec)
    d.update(name=spec.name, dtype=spec.dtype)
    return d


def flatmm_spec_to_dict(spec: Any) -> Dict[str, Any]:
    """:class:`FlatMMSpec` -> flat dict."""
    d = _gemm_subspec_to_dict(spec)
    d.update(name=spec.name, batch_size=spec.batch_size, preshuffle_b=spec.preshuffle_b)
    return d


def streamk_gemm_spec_to_dict(spec: Any) -> Dict[str, Any]:
    """:class:`StreamKGemmSpec` -> flat dict."""
    return dict(
        name=spec.name,
        M=spec.M,
        N=spec.N,
        K=spec.K,
        tile_m=spec.tile_m,
        tile_n=spec.tile_n,
        tile_k=spec.tile_k,
        dtype=spec.dtype,
        num_cus=spec.num_cus,
        blocks_per_cu=spec.blocks_per_cu,
        persistent=spec.persistent,
    )


def block_scale_gemm_spec_to_dict(spec: Any) -> Dict[str, Any]:
    """:class:`BlockScaleGemmSpec` -> flat dict. The Python ``group_size_mnk``
    tuple is split into the three scalar group fields the binding expects."""
    gm, gn, gk = spec.group_size_mnk
    return dict(
        name=spec.name,
        M=spec.M,
        N=spec.N,
        K=spec.K,
        quant_mode=spec.quant_mode,
        mantissa_dtype=spec.mantissa_dtype,
        preshuffle_b=spec.preshuffle_b,
        group_m=gm,
        group_n=gn,
        group_k=gk,
        block_tile_m=spec.block_tile_m,
        block_tile_n=spec.block_tile_n,
        per_input_row=spec.per_input_row,
    )


def mx_gemm_spec_to_dict(spec: Any) -> Dict[str, Any]:
    """:class:`MxGemmSpec` -> flat dict."""
    return dict(
        name=spec.name,
        M=spec.M,
        N=spec.N,
        K=spec.K,
        mantissa_dtype=spec.mantissa_dtype,
        group_k=spec.group_k,
        block_tile_m=spec.block_tile_m,
        block_tile_n=spec.block_tile_n,
        per_input_row=spec.per_input_row,
    )


def mfma_gemm_spec_to_dict(spec: Any) -> Dict[str, Any]:
    """:class:`MfmaGemmSpec` -> flat dict."""
    return dict(
        name=spec.name,
        M=spec.M,
        N=spec.N,
        K=spec.K,
        dtype=spec.dtype,
        tile_m=spec.tile_m,
        tile_n=spec.tile_n,
        kpack=spec.kpack,
    )


def matmul_nbits_spec_to_dict(spec: Any) -> Dict[str, Any]:
    """:class:`MatMulNBitsSpec` -> flat dict (nested tile flattened)."""
    t = spec.tile
    return dict(
        name=spec.name,
        N=spec.N,
        K=spec.K,
        tile_m=t.tile_m,
        tile_n=t.tile_n,
        tile_k=t.tile_k,
        warp_m=t.warp_m,
        warp_n=t.warp_n,
        warp_k=t.warp_k,
        warp_tile_m=t.warp_tile_m,
        warp_tile_n=t.warp_tile_n,
        warp_tile_k=t.warp_tile_k,
        group_size=spec.group_size,
        seq_len_tile=spec.seq_len_tile,
        wave_size=spec.wave_size,
        block_size=spec.block_size,
        scale_dtype=spec.scale_dtype,
        zero_points=spec.zero_points,
        packing=spec.packing,
        family=spec.family,
        optimized=spec.optimized,
    )


def gemm_multi_d_spec_to_dict(spec: Any) -> Dict[str, Any]:
    """:class:`GemmMultiDSpec` -> nested dict (base + operand list)."""
    base = universal_gemm_spec_to_dict(spec.base)
    return dict(
        base=base,
        d_operands=[(nm, op) for (nm, op) in spec.d_operands],
        d_dtype=spec.d_dtype,
        d_load_kind=spec.d_load_kind,
    )


def gemm_multi_abd_spec_to_dict(spec: Any) -> Dict[str, Any]:
    """:class:`GemmMultiAbdSpec` -> nested dict (base + operand lists)."""
    base = universal_gemm_spec_to_dict(spec.base)
    return dict(
        base=base,
        a_operands=[(nm, dt) for (nm, dt) in spec.a_operands],
        b_operands=[(nm, dt) for (nm, dt) in spec.b_operands],
        d_operands=[(nm, op) for (nm, op) in spec.d_operands],
        d_dtype=spec.d_dtype,
        d_load_kind=spec.d_load_kind,
    )


def _conv_problem_to_dict(p: Any) -> Dict[str, Any]:
    """:class:`ConvProblem` -> dict (shared by conv_implicit_gemm / img2col).

    The filter-window extent is named ``Y``/``X`` in the C engine; on this
    branch the Python ``ConvProblem`` still exposes it as ``R``/``S`` (the
    merge-target rename to ``Y``/``X`` is a separate change). Read whichever the
    Python object exposes and always emit the engine's ``Y``/``X`` keys.
    3-D fields are passed through when present (default 2-D leaves them 0)."""
    y = getattr(p, "Y", None)
    if y is None:
        y = p.R
    x = getattr(p, "X", None)
    if x is None:
        x = p.S
    d = dict(
        N=p.N,
        Hi=p.Hi,
        Wi=p.Wi,
        C=p.C,
        K=p.K,
        Y=y,
        X=x,
        sH=p.sH,
        sW=p.sW,
        pH=p.pH,
        pW=p.pW,
        dH=p.dH,
        dW=p.dW,
    )
    for f in ("Di", "Z", "sD", "pD", "dD"):
        v = getattr(p, f, None)
        if v is not None:
            d[f] = v
            d["is_3d"] = True
    return d


def conv_implicit_gemm_spec_to_dict(spec: Any) -> Dict[str, Any]:
    """:class:`ImplicitGemmConvSpec` -> flat dict (problem nested). Optional
    fields (lds_k_pad/waves_per_eu/vector_size_*) are forwarded when set; the
    binding leaves them at the engine default (Python ``None``) otherwise."""
    d = dict(
        problem=_conv_problem_to_dict(spec.problem),
        name=spec.name,
        tile_m=spec.tile_m,
        tile_n=spec.tile_n,
        tile_k=spec.tile_k,
        warp_m=spec.warp_m,
        warp_n=spec.warp_n,
        warp_tile_m=spec.warp_tile_m,
        warp_tile_n=spec.warp_tile_n,
        warp_tile_k=spec.warp_tile_k,
        wave_size=spec.wave_size,
        pipeline=spec.pipeline,
        epilogue=spec.epilogue,
        async_dma=spec.async_dma,
        unroll_k=spec.unroll_k,
        chiplet_swizzle=spec.chiplet_swizzle,
        chiplet_wgm=spec.chiplet_wgm,
        chiplet_num_xcds=spec.chiplet_num_xcds,
        chiplet_chunk_size=spec.chiplet_chunk_size,
        k0_k1_split=spec.k0_k1_split,
        groups=spec.groups,
    )
    # dtype_* and the vector-size/optional knobs are merge-target additions
    # (#8624); forward them only when this branch's spec exposes them so the
    # binding falls back to the engine default otherwise.
    for f in (
        "dtype_a",
        "dtype_b",
        "dtype_d",
        "dtype_acc",
        "lds_k_pad",
        "waves_per_eu",
        "vector_size_a",
        "vector_size_b",
        "vector_size_c",
    ):
        v = getattr(spec, f, None)
        if v is not None:
            d[f] = v
    return d


def conv_direct_grouped_spec_to_dict(spec: Any, kind: str) -> Dict[str, Any]:
    """:class:`DirectConv16cSpec` / :class:`DirectConv4cSpec` -> flat dict.
    ``kind`` ("16c"|"4c") selects the binding's spec path. The 16c-only
    ``double_buffer``/``fold_k32`` fields are forwarded when present."""
    p = spec.problem
    d = dict(
        kind=kind,
        problem=dict(
            N=p.N,
            H=p.H,
            W=p.W,
            groups=p.groups,
            cpg=p.cpg,
            kpg=p.kpg,
            KH=p.KH,
            KW=p.KW,
            PAD=p.PAD,
            stride=p.stride,
        ),
        name=spec.name,
        block_q=spec.block_q,
        block_groups=spec.block_groups,
        wave_size=spec.wave_size,
    )
    for f in ("double_buffer", "fold_k32"):
        v = getattr(spec, f, None)
        if v is not None:
            d[f] = v
    return d


def img2col_spec_to_dict(spec: Any) -> Dict[str, Any]:
    """:class:`Img2ColSpec` -> flat dict (problem nested)."""
    return dict(
        problem=_conv_problem_to_dict(spec.problem),
        dtype=spec.dtype,
        block_tile_m=spec.block_tile_m,
        block_tile_k=spec.block_tile_k,
        vec_k=spec.vec_k,
        name=spec.name,
    )


def deep_fused_conv_pool_spec_to_dict(spec: Any) -> Dict[str, Any]:
    """:class:`DeepFusedConvPoolSpec` -> the factory-argument dict the binding
    feeds to ``rocke_make_deep_fused_conv_pool_spec``. The factory derives the
    same fields both engines do, so the flat conv/pool shape + tiling knobs are
    sufficient."""
    prob = spec.problem
    conv = prob.conv
    r = getattr(conv, "Y", None)
    if r is None:
        r = conv.R
    s = getattr(conv, "X", None)
    if s is None:
        s = conv.S
    return dict(
        n=conv.N,
        h=conv.Hi,
        w=conv.Wi,
        c=conv.C,
        k0=conv.K,
        k1=prob.conv1_k,
        r=r,
        s=s,
        pool_tile_h=spec.pool_tile_h,
        pool_tile_w=spec.pool_tile_w,
        tile_n=spec.tile_n,
        tile_k=spec.tile_k,
        conv1_tile_k=spec.conv1_tile_k,
        warp_m=spec.warp_m,
        warp_n=spec.warp_n,
        warp_tile_m=spec.warp_tile_m,
        warp_tile_n=spec.warp_tile_n,
        warp_tile_k=spec.warp_tile_k,
        wave_size=spec.wave_size,
        name=spec.name,
        pipeline=spec.pipeline,
        unroll_k=spec.unroll_k,
        async_dma=spec.async_dma,
        cache_input_footprint=spec.cache_input_footprint,
        direct_conv0_from_input_cache=spec.direct_conv0_from_input_cache,
    )


# --------------------------- generic dispatch -------------------------


def _diff_and_pick(
    family: str,
    name: str,
    arch: str,
    py_ll: str,
    py_ir: str,
    cpp_ll: str,
    cpp_ir: str,
    want_ir: bool,
) -> "GemmLowerResult":
    """Compare the two engines' artifacts and return the agreed result, or
    raise :class:`BackendMismatch` with a precise diff. Family-agnostic."""
    if py_ir != cpp_ir:
        raise BackendMismatch(
            f"python vs cpp engine disagree for {family} '{name}' on {arch}:\n"
            + _text_diff("serialized IR (ck.dsl.ir/v1)", py_ir, cpp_ir)
        )
    if py_ll != cpp_ll:
        raise BackendMismatch(
            f"python vs cpp engine disagree for {family} '{name}' on {arch}:\n"
            + _text_diff("lowered AMDGPU .ll", py_ll, cpp_ll)
        )
    return GemmLowerResult(BACKEND_BOTH, py_ll, py_ir if want_ir else "")


def _lower_family(
    family: str,
    spec: Any,
    arch: str,
    backend: Optional[str],
    want_ir: bool,
    py_fn,
    cpp_ll_fn,
    cpp_ir_fn,
    spec_name: str,
) -> "GemmLowerResult":
    """The family-agnostic python/cpp/both driver.

    ``py_fn(want_ir) -> (ll, ir)`` runs the native Python engine.
    ``cpp_ll_fn() -> ll`` / ``cpp_ir_fn() -> ir`` reach the ``rocke_engine``
    binding for this family. The selection precedence and the differential
    comparison are identical to :func:`lower_universal_gemm`.
    """
    chosen = resolve_backend(backend)

    if chosen == BACKEND_PYTHON:
        ll, ir = py_fn(want_ir)
        return GemmLowerResult(BACKEND_PYTHON, ll, ir)

    if chosen == BACKEND_CPP:
        ll = cpp_ll_fn()
        ir = cpp_ir_fn() if want_ir else ""
        return GemmLowerResult(BACKEND_CPP, ll, ir)

    # both
    py_ll, py_ir = py_fn(True)
    cpp_ll = cpp_ll_fn()
    cpp_ir = cpp_ir_fn()
    return _diff_and_pick(
        family, spec_name, arch, py_ll, py_ir, cpp_ll, cpp_ir, want_ir
    )


def _name_of(spec: Any) -> str:
    return getattr(spec, "name", "?")


def lower_batched_gemm(
    spec: Any,
    *,
    arch: str = "gfx950",
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> "GemmLowerResult":
    """Lower a :class:`BatchedGemmSpec` through the selected backend."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from ..instances.common.batched_gemm import build_batched_gemm
        from .lower_llvm import lower_kernel_to_llvm

        k = build_batched_gemm(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from .ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = _import_engine()
    sd = batched_gemm_spec_to_dict(spec)
    return _lower_family(
        "batched GEMM",
        spec,
        arch,
        backend,
        want_ir,
        py_fn,
        lambda: eng.batched_gemm_lower_llvm(sd, arch=arch),
        lambda: eng.batched_gemm_serialize_ir(sd, arch=arch),
        _name_of(spec),
    )


def lower_grouped_gemm(
    spec: Any,
    *,
    arch: str = "gfx950",
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> "GemmLowerResult":
    """Lower a :class:`GroupedGemmSpec` (per-group base kernel)."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from ..instances.common.grouped_gemm import build_grouped_gemm
        from .lower_llvm import lower_kernel_to_llvm

        k = build_grouped_gemm(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from .ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = _import_engine()
    sd = grouped_gemm_spec_to_dict(spec)
    return _lower_family(
        "grouped GEMM",
        spec,
        arch,
        backend,
        want_ir,
        py_fn,
        lambda: eng.grouped_gemm_lower_llvm(sd, arch=arch),
        lambda: eng.grouped_gemm_serialize_ir(sd, arch=arch),
        _name_of(spec),
    )


def lower_flatmm(
    spec: Any,
    *,
    arch: str = "gfx950",
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> "GemmLowerResult":
    """Lower a :class:`FlatMMSpec`."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from ..instances.common.flatmm import build_flatmm
        from .lower_llvm import lower_kernel_to_llvm

        k = build_flatmm(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from .ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = _import_engine()
    sd = flatmm_spec_to_dict(spec)
    return _lower_family(
        "flatmm",
        spec,
        arch,
        backend,
        want_ir,
        py_fn,
        lambda: eng.flatmm_lower_llvm(sd, arch=arch),
        lambda: eng.flatmm_serialize_ir(sd, arch=arch),
        _name_of(spec),
    )


def lower_streamk_gemm(
    spec: Any,
    *,
    arch: str = "gfx950",
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> "GemmLowerResult":
    """Lower a :class:`StreamKGemmSpec`."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from ..instances.common.streamk_gemm import build_streamk_gemm
        from .lower_llvm import lower_kernel_to_llvm

        k = build_streamk_gemm(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from .ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = _import_engine()
    sd = streamk_gemm_spec_to_dict(spec)
    return _lower_family(
        "stream-K GEMM",
        spec,
        arch,
        backend,
        want_ir,
        py_fn,
        lambda: eng.streamk_gemm_lower_llvm(sd, arch=arch),
        lambda: eng.streamk_gemm_serialize_ir(sd, arch=arch),
        _name_of(spec),
    )


def lower_block_scale_gemm(
    spec: Any,
    *,
    arch: str = "gfx950",
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> "GemmLowerResult":
    """Lower a :class:`BlockScaleGemmSpec`."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from ..instances.common.block_scale_gemm import build_block_scale_gemm
        from .lower_llvm import lower_kernel_to_llvm

        k = build_block_scale_gemm(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from .ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = _import_engine()
    sd = block_scale_gemm_spec_to_dict(spec)
    return _lower_family(
        "block-scale GEMM",
        spec,
        arch,
        backend,
        want_ir,
        py_fn,
        lambda: eng.block_scale_gemm_lower_llvm(sd, arch=arch),
        lambda: eng.block_scale_gemm_serialize_ir(sd, arch=arch),
        _name_of(spec),
    )


def lower_mx_gemm(
    spec: Any,
    *,
    arch: str = "gfx950",
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> "GemmLowerResult":
    """Lower a :class:`MxGemmSpec`."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from ..instances.common.mx_gemm import build_mx_gemm
        from .lower_llvm import lower_kernel_to_llvm

        k = build_mx_gemm(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from .ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = _import_engine()
    sd = mx_gemm_spec_to_dict(spec)
    return _lower_family(
        "mx GEMM",
        spec,
        arch,
        backend,
        want_ir,
        py_fn,
        lambda: eng.mx_gemm_lower_llvm(sd, arch=arch),
        lambda: eng.mx_gemm_serialize_ir(sd, arch=arch),
        _name_of(spec),
    )


def lower_mfma_gemm(
    spec: Any,
    *,
    arch: str = "gfx950",
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> "GemmLowerResult":
    """Lower a :class:`MfmaGemmSpec`."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from ..instances.common.mfma_gemm import build_mfma_gemm
        from .lower_llvm import lower_kernel_to_llvm

        k = build_mfma_gemm(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from .ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = _import_engine()
    sd = mfma_gemm_spec_to_dict(spec)
    return _lower_family(
        "mfma GEMM",
        spec,
        arch,
        backend,
        want_ir,
        py_fn,
        lambda: eng.mfma_gemm_lower_llvm(sd, arch=arch),
        lambda: eng.mfma_gemm_serialize_ir(sd, arch=arch),
        _name_of(spec),
    )


def lower_matmul_nbits(
    spec: Any,
    *,
    arch: str = "gfx1201",
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> "GemmLowerResult":
    """Lower a :class:`MatMulNBitsSpec`. Default arch is an RDNA WMMA target
    (gfx1201); the family validator rejects CDNA archs on both engines."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from ..instances.common.matmul_nbits import build_matmul_nbits
        from .lower_llvm import lower_kernel_to_llvm

        k = build_matmul_nbits(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from .ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = _import_engine()
    sd = matmul_nbits_spec_to_dict(spec)
    return _lower_family(
        "matmul_nbits",
        spec,
        arch,
        backend,
        want_ir,
        py_fn,
        lambda: eng.matmul_nbits_lower_llvm(sd, arch=arch),
        lambda: eng.matmul_nbits_serialize_ir(sd, arch=arch),
        _name_of(spec),
    )


def lower_gemm_multi_d(
    spec: Any,
    *,
    arch: str = "gfx950",
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> "GemmLowerResult":
    """Lower a :class:`GemmMultiDSpec`."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from ..instances.common.gemm_multi_d import build_gemm_multi_d
        from .lower_llvm import lower_kernel_to_llvm

        k = build_gemm_multi_d(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from .ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = _import_engine()
    sd = gemm_multi_d_spec_to_dict(spec)
    return _lower_family(
        "gemm_multi_d",
        spec,
        arch,
        backend,
        want_ir,
        py_fn,
        lambda: eng.gemm_multi_d_lower_llvm(sd, arch=arch),
        lambda: eng.gemm_multi_d_serialize_ir(sd, arch=arch),
        _name_of(spec.base),
    )


def lower_gemm_multi_abd(
    spec: Any,
    *,
    arch: str = "gfx950",
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> "GemmLowerResult":
    """Lower a :class:`GemmMultiAbdSpec`."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from ..instances.common.gemm_multi_abd import build_gemm_multi_abd
        from .lower_llvm import lower_kernel_to_llvm

        k = build_gemm_multi_abd(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from .ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = _import_engine()
    sd = gemm_multi_abd_spec_to_dict(spec)
    return _lower_family(
        "gemm_multi_abd",
        spec,
        arch,
        backend,
        want_ir,
        py_fn,
        lambda: eng.gemm_multi_abd_lower_llvm(sd, arch=arch),
        lambda: eng.gemm_multi_abd_serialize_ir(sd, arch=arch),
        _name_of(spec.base),
    )


def lower_conv_implicit_gemm(
    spec: Any,
    *,
    arch: str = "gfx950",
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> "GemmLowerResult":
    """Lower an :class:`ImplicitGemmConvSpec`."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from ..instances.common.conv_implicit_gemm import build_implicit_gemm_conv
        from .lower_llvm import lower_kernel_to_llvm

        k = build_implicit_gemm_conv(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from .ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = _import_engine()
    sd = conv_implicit_gemm_spec_to_dict(spec)
    return _lower_family(
        "conv_implicit_gemm",
        spec,
        arch,
        backend,
        want_ir,
        py_fn,
        lambda: eng.conv_implicit_gemm_lower_llvm(sd, arch=arch),
        lambda: eng.conv_implicit_gemm_serialize_ir(sd, arch=arch),
        _name_of(spec),
    )


def lower_conv_direct_grouped(
    spec: Any,
    *,
    kind: str = "16c",
    arch: str = "gfx950",
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> "GemmLowerResult":
    """Lower a :class:`DirectConv16cSpec` / :class:`DirectConv4cSpec`.
    ``kind`` ("16c"|"4c") selects the channel-blocking variant."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from ..instances.common.conv_direct_grouped import (
            build_direct_conv_16c,
            build_direct_conv_4c,
        )
        from .lower_llvm import lower_kernel_to_llvm

        build = build_direct_conv_4c if kind == "4c" else build_direct_conv_16c
        k = build(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from .ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = _import_engine()
    sd = conv_direct_grouped_spec_to_dict(spec, kind)
    return _lower_family(
        "conv_direct_grouped",
        spec,
        arch,
        backend,
        want_ir,
        py_fn,
        lambda: eng.conv_direct_grouped_lower_llvm(sd, arch=arch),
        lambda: eng.conv_direct_grouped_serialize_ir(sd, arch=arch),
        _name_of(spec),
    )


def lower_img2col(
    spec: Any,
    *,
    arch: str = "gfx950",
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> "GemmLowerResult":
    """Lower an :class:`Img2ColSpec`."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from ..instances.common.img2col import build_img2col
        from .lower_llvm import lower_kernel_to_llvm

        k = build_img2col(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from .ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = _import_engine()
    sd = img2col_spec_to_dict(spec)
    return _lower_family(
        "img2col",
        spec,
        arch,
        backend,
        want_ir,
        py_fn,
        lambda: eng.img2col_lower_llvm(sd, arch=arch),
        lambda: eng.img2col_serialize_ir(sd, arch=arch),
        _name_of(spec),
    )


def lower_deep_fused_conv_pool(
    spec: Any,
    *,
    arch: str = "gfx950",
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> "GemmLowerResult":
    """Lower a :class:`DeepFusedConvPoolSpec`."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from ..instances.common.deep_fused_conv_pool import (
            build_deep_fused_conv_pool,
        )
        from .lower_llvm import lower_kernel_to_llvm

        k = build_deep_fused_conv_pool(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from .ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = _import_engine()
    sd = deep_fused_conv_pool_spec_to_dict(spec)
    return _lower_family(
        "deep_fused_conv_pool",
        spec,
        arch,
        backend,
        want_ir,
        py_fn,
        lambda: eng.deep_fused_conv_pool_lower_llvm(sd, arch=arch),
        lambda: eng.deep_fused_conv_pool_serialize_ir(sd, arch=arch),
        _name_of(spec),
    )

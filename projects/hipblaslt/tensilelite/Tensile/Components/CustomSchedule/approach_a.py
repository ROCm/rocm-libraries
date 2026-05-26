################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Approach A — true non-CMS reference build (rocm-libraries-nyb5).

The CMS validator's reference side has historically been a SHADOW capture:
a synthetic re-assembly of what the default scheduler *would have produced*
with the CMS-mutated ``kernel`` dict (see
``Tensile/Components/2LZD_INVESTIGATION.md §1 (A)`` for the upstream
flag-flip mechanism). The shadow is not assembled into runnable code and
never executes on hardware. Comparing against it is comparing against a
fiction.

Per the user decision recorded in ``2LZD_INVESTIGATION.md §6``
(2026-05-12), the validator's reference must instead be a real emittable
kernel built with ``UseCustomMainLoopSchedule=0``. This module implements
that second build.

Public API:

    build_non_cms_reference(kernel_config, asm, isaInfoMap) -> FourPartCapture

The helper is the foundation of the meta-bead ``rocm-libraries-71hw``
work decomposition. Its capture is consumed by ``compare_graphs`` as the
``ref`` argument; the existing CMS-side capture is the ``subj``.

Isolation strategy (Q5 — second writer instance):

The function spins up a *separate* ``KernelWriterAssembly`` instance for
Build #2, NOT a flag-swap on the same writer. This mirrors the prior-art
pattern at ``Tensile/Tests/unit/_dump_carveout_assembly.py:229`` and
guarantees zero state contamination between the CMS build and the non-CMS
reference build.

Cross-references:
    - ``2LZD_INVESTIGATION.md §6 + §6.2`` — Approach A pick + Q2/Q3
      framing (two builds, accept whatever Tensilelite mutates).
    - ``PRELOOP_CAPTURE_PHASE1.md §7`` — body-label tolerance is
      critical-path for Approach A.
    - ``D3ZJ_SCMPEQI32_INVESTIGATION.md §3.4`` — the SHADOW capture's
      ML/ML-1 LCC absence; Build #2's post-closeLoop finalize closes
      this defect as a side effect.
    - ``NYB5_IMPLEMENTATION.md`` — design memo for this implementation.
"""

import os
import sys
import tempfile
from copy import deepcopy


# rocm-libraries-m7o5: framework-derived (NOT YAML-tunable) CMS-only flags.
# When Solution.assignDerivedParameters runs on a kernel with
# UseCustomMainLoopSchedule=1, these flags are flipped by the CMS post-match
# block (Solution.py:2007-2013) and the F32X-emulation block (Solution.py:
# 2027-2030). For a true non-CMS reference build we must reset them to the
# Solution defaults so the framework's normal non-CMS derivation runs from a
# clean slate and the resulting kernel fits in the same vgpr budget as the
# standalone CMS=0 cross-product solution. See rocm-libraries-m7o5 bead
# diagnosis 2026-05-26 for the symptom (523 vgpr blowup on MT 192x256x32
# TF32 emulation kernels when only UseCustomMainLoopSchedule is flipped).
_CMS_FRAMEWORK_DERIVED_DEFAULTS = {
    # Solution.py:648 unconditionally sets this False; CMS path then sets
    # True at 2013, MFMA F32X-emulation non-CMS path also at 2030.
    "MfmaInitCVgprs": False,
    # Solution.py:649 unconditionally sets this False; no CMS schedule
    # writes True per rocm-libraries-2bww audit.
    "UseDot2F32XEmulation": False,
}

# Markers Solution.__init__ inspects to short-circuit re-derivation
# (Solution.py:393-396, 457-459, 1223-1225). Their presence on the input
# config tells the constructor "already derived; skip derivation" — which
# is exactly the wrong behaviour when we want to re-run derivation under
# UseCustomMainLoopSchedule=0.
_DERIVATION_GATE_KEYS = (
    "AssignedProblemIndependentDerivedParameters",
    "AssignedDerivedParameters",
)


def _scrub_for_non_cms_rederivation(kernel_config):
    """Return a deep-copied ``kernel_config`` with the CMS-induced
    framework-derived flags reset to their Solution defaults, the
    derivation gate markers removed, and ``UseCustomMainLoopSchedule``
    forced to 0.

    This is the in-memory equivalent of writing a temp YAML and re-loading
    it: the resulting dict has no derived-state contamination from the
    prior CMS pass, so passing it through ``_make_solution`` runs
    ``Solution.assignDerivedParameters`` end-to-end on the non-CMS branch
    and produces a clean kernel config that the non-CMS scheduler can
    honour within its normal vgpr budget.
    """
    config = deepcopy(kernel_config)

    # Force non-CMS path
    config["UseCustomMainLoopSchedule"] = 0

    # Reset framework-derived (non-YAML-tunable) flags so Solution re-derives
    # them from scratch on the non-CMS branch.
    for flag, default in _CMS_FRAMEWORK_DERIVED_DEFAULTS.items():
        config[flag] = default

    # `UsePLRPack` is YAML-tunable, but its non-CMS-path gating (Solution.py
    # 2035-2053) requires the YAML preconditions (F32X emulation, SIA3,
    # ForceUnrollSubIter, DTL=1, PGR>0, PLR>0) to actually be met on the
    # non-CMS branch. The CMS-side dict guaranteed those held *for the CMS
    # path* because the schedule's required_flags required them; the
    # non-CMS reference build doesn't need pack semantics. Reset to False
    # to match the standalone CMS=0 cross-product solution's footprint.
    # See test_non_cms_reference_compare_graphs_surfaces_only_known_residuals
    # in Tests/unit/test_approach_a_non_cms_reference.py for the historical
    # manual fix-up this internalizes.
    config["UsePLRPack"] = False

    # Strip derivation gates so Solution.__init__ re-runs full derivation
    # rather than short-circuiting on the already-derived state from the
    # CMS pass.
    for k in _DERIVATION_GATE_KEYS:
        config.pop(k, None)

    return config


def _serialize_for_yaml(obj):
    """Best-effort conversion of objects in `kernel_config` into something
    PyYAML can dump. Numeric, string, bool, None pass through; lists and
    dicts recurse; tuples become lists; everything else falls back to
    ``str()`` so the temp-YAML stays readable even when the source dict
    contains rocisa / ISA objects without a stable serializer."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _serialize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_yaml(x) for x in obj]
    # Fallback — keep the dump human-readable even if not round-trippable.
    return str(obj)


def _maybe_dump_temp_yaml(scrubbed_config):
    """Write `scrubbed_config` to a temp YAML for debuggability. The path
    is returned so the caller can keep it on exception, delete it on
    success (unless the user opted in via the env var below), and surface
    it in any error messages.

    Returns ``(path_or_None, should_keep_on_success)``. Returns
    ``(None, False)`` if PyYAML is unavailable (we still re-derive
    in-process; the YAML is purely a debugging aid).
    """
    try:
        import yaml  # noqa: F401  (optional — only used for debug dump)
    except ImportError:
        return (None, False)

    keep_on_success = bool(os.environ.get("TENSILE_KEEP_NON_CMS_TEMP_YAML"))

    try:
        fh = tempfile.NamedTemporaryFile(
            suffix=".yaml",
            prefix="build_non_cms_reference_",
            mode="w",
            delete=False,
        )
        with fh:
            yaml.safe_dump(
                _serialize_for_yaml(scrubbed_config),
                fh,
                default_flow_style=False,
                sort_keys=True,
            )
        return (fh.name, keep_on_success)
    except Exception:
        # Debug dump is best-effort. Never let a YAML serialization
        # failure block the actual re-derivation.
        return (None, False)


def build_non_cms_reference(kernel_config, asm, isaInfoMap):
    """Build a non-CMS reference kernel and return its ``FourPartCapture``.

    Args:
        kernel_config: dict-shaped solution config (the same shape
            consumed by ``cms_test_utils._make_solution``). The caller's
            ``UseCustomMainLoopSchedule`` value is overridden to 0 so
            this is always a non-CMS build, regardless of the input.
        asm: The ``Assembler`` instance from ``isa_infrastructure``.
        isaInfoMap: The ISA info map from ``isa_infrastructure``.

    Returns:
        ``Tensile.Components.ScheduleCapture.FourPartCapture`` whose
        ``main_loop_prev``/``main_loop``/``n_gl``/``n_ll`` reflect the
        natural emission of the non-CMS scheduler. ``source`` is set to
        ``'non-cms-reference'`` to distinguish from ``'default-sia3'``
        (the legacy shadow capture).

    The reference build's main-loop body includes the loop-counter code
    (LCC: ``SSubU32`` + ``SCmpEQI32``) that the SHADOW capture missed —
    the non-CMS path emits ``closeLoop`` naturally and the capture
    builder finalizes after that emission (vs. shadow's pre-closeLoop
    finalize at ``KernelWriter.py:4591``).

    Build-time cost note (Q3 — deferred): each call drives a full
    ``_getKernelSource`` invocation. Per the user's Q3 decision, ~2x
    build time on the assert path is acceptable in the near-term;
    caching, test/CI-only gating, and process-pool isolation are
    reserved for after correctness lands.

    rocm-libraries-m7o5 (2026-05-26): the input ``kernel_config`` is the
    post-CMS-derivation dict — flipping only ``UseCustomMainLoopSchedule``
    leaves the CMS-only framework-derived flags asserted, and the non-CMS
    scheduler can't honour them within its vgpr budget (blew 523 vgpr on
    MT 192x256x32 TF32-emulation kernels). The fix re-derives a clean
    Solution by scrubbing those flags + the ``AssignedDerivedParameters``
    short-circuit markers from the input, optionally dumping the scrubbed
    config to a temp YAML for debuggability, then running the full
    ``Solution.assignDerivedParameters`` pipeline via ``_make_solution``.

    Option B1 (in-place scrub + re-derive) was chosen over Option B2
    (snapshot pre-mutation state in ``dispatch.wrapped_func``) because
    under the rocm-libraries-2bww strict model the dispatcher's
    ``wrapped_func`` is pure-validation and does not mutate kernel state
    — the mutations now happen inside ``Solution.assignDerivedParameters``
    (Solution.py:2007-2013, 2027-2030). A pre-mutation snapshot would have
    to live inside Solution itself, which is more invasive than this
    helper's localized scrub. The framework-derived flag set is short and
    centralized at module level (``_CMS_FRAMEWORK_DERIVED_DEFAULTS``) so
    future additions are caught at one site.
    """
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, DebugConfig

    # Defer the cms_test_utils import — that module reaches into rocisa
    # and we want this file importable even in environments where the
    # full kernel-build path isn't available (e.g. doc generation).
    here = os.path.dirname(os.path.abspath(__file__))
    tests_unit = os.path.normpath(os.path.join(
        here, "..", "..", "Tests", "unit"))
    if tests_unit not in sys.path:
        sys.path.insert(0, tests_unit)
    from cms_test_utils import _make_solution

    # rocm-libraries-m7o5: in-process equivalent of "write temp YAML, then
    # re-run Tensile's kernel-config derivation pipeline". The temp YAML is
    # kept as a debug-only artifact (env var opt-in or on exception).
    scrubbed_config = _scrub_for_non_cms_rederivation(kernel_config)
    yaml_path, keep_on_success = _maybe_dump_temp_yaml(scrubbed_config)
    keep_yaml = False  # set True on exception to preserve for inspection

    try:
        # Q5 — second writer instance, fully isolated.
        solution = _make_solution(scrubbed_config, asm, isaInfoMap)

        writer = KernelWriterAssembly(asm, DebugConfig())
        # Switch on the new non-CMS capture path (gated separately from
        # the legacy shadow `_captureDefaultSchedule` flag — the shadow
        # path is owned by `rocm-libraries-czby` and stays untouched in
        # this bead).
        writer.enable_capture_non_cms_build()

        writer._getKernelSource(solution)

        capture = writer._last_default_capture
        if capture is None:
            raise RuntimeError(
                "build_non_cms_reference: writer._last_default_capture "
                "was not populated. The non-CMS capture path in "
                "`_loopBody` / `noLoadLoop` / `kernelBody` did not run — "
                "check that enable_capture_non_cms_build() set the "
                "`_captureNonCmsBuild` flag and the kernelBody assembly "
                "site at KernelWriter.py (post-loop) consumed the "
                "builder outputs."
            )
        return capture
    except Exception:
        # Preserve the temp YAML on failure so the user can inspect the
        # scrubbed config that triggered the build error.
        keep_yaml = True
        raise
    finally:
        if yaml_path is not None and not keep_yaml and not keep_on_success:
            try:
                os.unlink(yaml_path)
            except OSError:
                pass

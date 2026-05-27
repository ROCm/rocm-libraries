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

    build_non_cms_reference(solution, asm, isaInfoMap) -> FourPartCapture

The helper is the foundation of the meta-bead ``rocm-libraries-71hw``
work decomposition. Its capture is consumed by ``compare_graphs`` as the
``ref`` argument; the existing CMS-side capture is the ``subj``.

Isolation strategy (Q5 — second writer instance):

The function spins up a *separate* ``KernelWriterAssembly`` instance for
Build #2, NOT a flag-swap on the same writer. This mirrors the prior-art
pattern at ``Tensile/Tests/unit/_dump_carveout_assembly.py:229`` and
guarantees zero state contamination between the CMS build and the non-CMS
reference build.

Pre-CMS-derivation snapshot (rocm-libraries-y391, refined by
rocm-libraries-l1l6):

For the production call site (``KernelWriter.py`` mid-CMS-build), the
input ``solution`` is one that has already been through
``Solution.assignDerivedParameters`` with its CMS-injection block having
fired (``Solution.py:2007-2013`` and ``2027-2030``). Flipping only
``UseCustomMainLoopSchedule`` on the post-CMS state dict leaves CMS-only
framework-derived flags asserted and blows the non-CMS scheduler's vgpr
budget (m7o5: 523 vgpr on MT 192x256x32 TF32 emulation kernels).

The principled fix is for ``Solution.assignDerivedParameters`` to
snapshot its own pre-CMS-derivation state and stash it on the Solution
itself; ``build_non_cms_reference`` consumes that snapshot via the
``solution.pre_cms_state()`` accessor (rocm-libraries-l1l6 replaced the
y391-era ``state["_pre_cms_derived_state"]`` magic-key contract with a
real ``Solution._pre_cms_state`` attribute + accessor — see
``Solution.py``'s ``pre_cms_state`` for the canonical reference). No
local knowledge of which flags Solution.py injects for CMS lives in this
module, so future schedules adding new ``required_flags`` inject
correctly with zero maintenance here.

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
from copy import deepcopy


# Markers Solution.__init__ inspects to short-circuit re-derivation
# (Solution.py:393-396, 457-459, 1223-1225). Their presence on the input
# config tells the constructor "already derived; skip derivation" — which
# is exactly the wrong behaviour when we want to re-run derivation under
# UseCustomMainLoopSchedule=0.
_DERIVATION_GATE_KEYS = (
    "AssignedProblemIndependentDerivedParameters",
    "AssignedDerivedParameters",
)


def _prepare_non_cms_config(solution):
    """Return a deep-copied config suitable for non-CMS re-derivation.

    Reads the pre-CMS-derivation snapshot from
    ``solution.pre_cms_state()`` (rocm-libraries-l1l6, Solution.py
    ``pre_cms_state``). Raises ``AttributeError`` via the accessor if
    ``Solution.assignDerivedParameters`` has not run or bailed before
    reaching the snapshot site (Solution.py ~:2007). Every Solution
    that completed normal construction carries the attribute.

    The returned config:
      * Comes from the snapshot — no CMS-only mutations carried over.
      * Has ``UseCustomMainLoopSchedule=0`` forced so the non-CMS
        branch of ``assignDerivedParameters`` fires.
      * Has ``Assigned*DerivedParameters`` gate markers stripped so the
        second derivation pass runs end-to-end (Solution.py:457-459,
        1223-1225 short-circuit on these).
    """
    try:
        snapshot = solution.pre_cms_state()
    except AttributeError as exc:
        raise RuntimeError(
            "build_non_cms_reference: Solution has no pre-CMS snapshot. "
            "Solution.assignDerivedParameters sets self._pre_cms_state "
            "before its CMS-dispatch block (Solution.py ~:2007); the "
            "accessor raises if it never ran or bailed early. See "
            "rocm-libraries-l1l6 (supersedes y391's magic-key contract)."
        ) from exc

    config = deepcopy(dict(snapshot))
    config["UseCustomMainLoopSchedule"] = 0
    for k in _DERIVATION_GATE_KEYS:
        config.pop(k, None)
    return config


def build_non_cms_reference(solution, asm, isaInfoMap):
    """Build a non-CMS reference kernel and return its ``FourPartCapture``.

    Args:
        solution: a ``Solution`` instance (post-``assignDerivedParameters``)
            from which to derive the non-CMS reference build. The
            ``solution.pre_cms_state()`` snapshot is the re-derivation
            base; the rest of ``solution`` is ignored.
            ``UseCustomMainLoopSchedule=0`` is forced on the snapshot so
            the non-CMS branch of ``assignDerivedParameters`` fires on
            the second pass. Raises ``RuntimeError`` (wrapping
            ``AttributeError``) if ``solution`` carries no snapshot —
            see ``_prepare_non_cms_config``.
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

    rocm-libraries-l1l6 (2026-05-26): takes a ``Solution`` and consumes
    the snapshot via the ``pre_cms_state()`` accessor instead of probing
    a dict for the ``_pre_cms_derived_state`` magic key (y391's contract,
    now retired). Solution stays idempotent; ``dict(solution)`` no
    longer carries the snapshot.

    rocm-libraries-y391 (2026-05-27): replaces the brittle
    ``_CMS_FRAMEWORK_DERIVED_DEFAULTS`` scrub list from m7o5
    (d21c97fb431). The scrub list required this module to track every
    flag Solution.py mutates inside its CMS-injection block; any new
    schedule adding a ``required_flag`` would silently break Build #2's
    vgpr budget. The snapshot mechanism in Solution.py means this
    module owns no knowledge of CMS-injected flags at all.
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

    non_cms_config = _prepare_non_cms_config(solution)

    # Q5 — second writer instance, fully isolated.
    non_cms_solution = _make_solution(non_cms_config, asm, isaInfoMap)

    writer = KernelWriterAssembly(asm, DebugConfig())
    # Switch on the new non-CMS capture path (gated separately from
    # the legacy shadow `_captureDefaultSchedule` flag — the shadow
    # path is owned by `rocm-libraries-czby` and stays untouched in
    # this bead).
    writer.enable_capture_non_cms_build()

    writer._getKernelSource(non_cms_solution)

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

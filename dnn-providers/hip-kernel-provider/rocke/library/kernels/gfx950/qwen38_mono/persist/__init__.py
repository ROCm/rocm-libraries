# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The true-mono persistent-framework platform (Qwen3.8 true mono, WP1).

``experiments/qwen38-true-mono/D0.md`` sections 1 (D0.2, D0.6, D0.8) and 5
(wave 1a, WP1). Everything here is additive: no existing build imports it, and
every knob it adds to :mod:`..framework` defaults to the shipped behaviour
(the identity tests prove the IR bytes are unchanged with the knobs off).

Modules:

* :mod:`.knobs`: :class:`PersistKnobs` (P6's R1-R5, ``inline``, ``args``,
  D0.8's hand-off / sync defaults), :class:`EdgeKnobs` / :class:`EdgePlan`
  (per-edge protocol, NUMA polarity, poll implementation, placement and queue:
  ``XCD-SCHEDULING.md``), :class:`RustScope` (per-op direct-backend lever
  scoping, D0.6), and the named presets (``D0_DEFAULTS``).
* :mod:`.sync`: the synchronization primitives P4 measured (striped counter
  barrier with abort-on-timeout, residency ticket in its per-XCD-start form,
  deadline polls, per-CTA epoch arrivals of P9, the launch-fatal error word).
* :mod:`.kv`: P5's in-launch coherence protocols (``bulk=sc1`` KV rows, the
  bulk rule, ``s_dcache_inv`` scalar re-reads) with the ISA markers the static
  KV-acquire check reads.
* :mod:`.lint`: static checks: polls that never reload (P2 finding 1), every
  in-launch KV load acquired (P5), fused-stage private staging never aliasing
  a live mailbox (P8), the CI resource gate (spills, scratch, exec-join; P6).
* :mod:`.isa_gate`: WP0a's ISA gate (build-key identity for host edits, the
  pooled distribution gate for kernel knobs), vendored for rocKE builds.
* :mod:`.parity`: the P4 / P5 probe kernels rebuilt on these primitives (the
  hardware parity gate: the platform reproduces the probes' verdicts).

Python engine only where a primitive uses a Python-only opcode (the ``ir`` /
``irf`` mailboxes, ``buffer_*_cp``); everything else is mirrored ops.
"""

from .knobs import (  # noqa: F401
    D0_DEFAULTS,
    EDGE_PLACEMENTS,
    EDGE_POLARITIES,
    EDGE_POLLS,
    EDGE_PROTOCOLS,
    EDGE_QUEUES,
    EdgeKnobs,
    EdgePlan,
    PersistKnobs,
    RustScope,
)

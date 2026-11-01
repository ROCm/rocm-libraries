# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Search-driven tuning of the Qwen3.8 true monokernel.

Applies the rocKE search engine (``rocke.dispatch.tuning.search``) to the
monokernel's stage, layer and round knobs, under the measurement discipline
of the Qwen3.8 work (DESIGN 9.1 / 9.2):

- :mod:`.knobs` -- knob spaces over the owners' dataclasses / lever strings,
  legality pruning, canonical tuning ids;
- :mod:`.catalog` -- every registered space (framework, toy chain, fused MoE,
  MoE port, B3, stage-function ABI, MXFP4 GEMV, CCL, ATOM schedule levers,
  Rust direct-compiler levers) and the accuracy-gated N-lever registry;
- :mod:`.strategies` -- the signed-lookback climber, the mono portfolio
  (with hill climbing, SHADE DE, TPE / GP-EI under Thompson sampling),
  priors from a lever ledger, transfer warm starts;
- :mod:`.gate` -- shuffled interleaved rounds, A-vs-A and recompiled
  controls, the win rules, SCLK / power sampling;
- :mod:`.objective` -- stage / layer / round / e2e objectives ranked by the
  share of the gap to the headline targets;
- :mod:`.tdb` -- tuning DB records, the signed-gain ledger, per-variant
  defaults and a learned selector;
- :mod:`.evaluate` -- batch jobs as an evaluator (``runner``);
- :mod:`.live_toy` -- the live gfx950 backend of the framework's toy chain;
- :mod:`.replay` / :mod:`.experiments` -- replay of recorded A/B data.

CLI: ``python -m kernels.gfx950.qwen38_mono.tune --help``. The guide for the
tracks is ``experiments/qwen38-true-mono/search/GUIDE.md`` at the repo root.
"""

from .knobs import BITWISE, TOLERANCE, CompositeKnobs, DataclassKnobs, Knob, KnobAdapter, LeverStringKnobs

__all__ = [
    "BITWISE",
    "TOLERANCE",
    "CompositeKnobs",
    "DataclassKnobs",
    "Knob",
    "KnobAdapter",
    "LeverStringKnobs",
]

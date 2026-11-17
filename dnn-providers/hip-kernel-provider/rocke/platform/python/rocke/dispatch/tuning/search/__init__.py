# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Search engine for tuning spaces: ask/tell strategies instead of sweeps.

Family-neutral: any :class:`~rocke.dispatch.tuning.space.KnobSpace` (via
:class:`KnobSearchSpace`), any explicit parameter list
(:class:`SearchSpace`), or any flat config list
(:meth:`SearchSpace.from_configs`) can be searched.

- :mod:`.params` -- :class:`Param`, :class:`SearchSpace`: mixed
  ordinal/categorical/bool/gated parameters, validity, repair, keys;
- :mod:`.knobspace` -- a dispatcher ``KnobSpace`` as a search space;
- :mod:`.trial` -- :class:`Trial`, :class:`Observation`, :class:`Fidelity`,
  :class:`Budget`;
- :mod:`.strategy` -- the :class:`Strategy` protocol, random and exhaustive;
- :mod:`.de` -- mixed-variable differential evolution (rand/1, current-to-
  (p)best/1; jDE / SHADE adaptation);
- :mod:`.halving` -- successive halving and Hyperband over fidelities;
- :mod:`.stats` -- paired / Welch t, signed-rank, Holm, :class:`Race`, noise;
- :mod:`.surrogate` -- priors (selector, LightGBM), TPE, GP + EI;
- :mod:`.local` -- hill climbing with lookback and combination re-sweeps;
- :mod:`.portfolio` -- the bandit portfolio and warm starts;
- :mod:`.measure` -- the backend protocol and the measurement rules;
- :mod:`.db` -- the tuning DB dispatch serves and search traces;
- :mod:`.driver` -- the ask/tell loop with checkpoint/resume;
- :mod:`.api` -- :func:`tune`, :func:`tune_candidate`, :func:`search_configs`;
- :mod:`.replay` -- offline scoring against recorded sweeps.

See ``dsl_docs/autotune/overview.md`` ("Search engine") for the design.
"""

from .api import (
    DEFAULT_PORTFOLIO,
    FAMILY_DEFAULTS,
    STRATEGY_NAMES,
    SpecBackend,
    default_ladder,
    make_strategy,
    recommend,
    record_result,
    search_configs,
    tune,
    tune_candidate,
)
from .db import (
    TraceWriter,
    TunedRecord,
    TuningDB,
    problem_key,
    read_trace,
    records_from_rows,
)
from .de import JDE, DERand, DifferentialEvolution
from .driver import SearchResult, run_search
from .halving import Hyperband, SuccessiveHalving
from .knobspace import KnobSearchSpace
from .local import HillClimb
from .measure import (
    Backend,
    CallableBackend,
    CompileOutcome,
    Evaluator,
    SyntheticBackend,
    TableBackend,
    toolchain_environment,
)
from .params import BOOL, CATEGORICAL, ORDINAL, Param, SearchSpace, flatten_spec
from .portfolio import Portfolio, WarmStart
from .stats import NoiseModel, Race
from .strategy import Exhaustive, RandomSearch, Strategy
from .surrogate import (
    GPEI,
    TPE,
    CallablePrior,
    LightGBMPrior,
    Prior,
    PriorScale,
    TablePrior,
)
from .trial import Budget, Fidelity, Observation, Status, Trial

__all__ = [
    "BOOL",
    "CATEGORICAL",
    "DEFAULT_PORTFOLIO",
    "FAMILY_DEFAULTS",
    "GPEI",
    "JDE",
    "ORDINAL",
    "STRATEGY_NAMES",
    "TPE",
    "Backend",
    "Budget",
    "CallableBackend",
    "CallablePrior",
    "CompileOutcome",
    "DERand",
    "DifferentialEvolution",
    "Evaluator",
    "Exhaustive",
    "Fidelity",
    "HillClimb",
    "Hyperband",
    "KnobSearchSpace",
    "LightGBMPrior",
    "NoiseModel",
    "Observation",
    "Param",
    "Portfolio",
    "Prior",
    "PriorScale",
    "Race",
    "RandomSearch",
    "SearchResult",
    "SearchSpace",
    "SpecBackend",
    "Status",
    "Strategy",
    "SuccessiveHalving",
    "SyntheticBackend",
    "TableBackend",
    "TablePrior",
    "TraceWriter",
    "Trial",
    "TunedRecord",
    "TuningDB",
    "WarmStart",
    "default_ladder",
    "flatten_spec",
    "make_strategy",
    "problem_key",
    "read_trace",
    "recommend",
    "record_result",
    "records_from_rows",
    "run_search",
    "search_configs",
    "toolchain_environment",
    "tune",
    "tune_candidate",
]

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Autotune mode of the RCR UniversalGemm dispatcher.

The default dispatch (:func:`rocke.dispatch.gemm.dispatch_gemm_fp16` /
``_bf16``) takes the learned selector's prediction without measuring. This
module measures instead, inside a budget, with the search engine
(:mod:`rocke.dispatch.tuning.search`):

* the space is the selector's gated candidate set for the request
  (:func:`selector.selector_candidates`: registered candidates and their sweep
  variants, one-step trait mutations, library tilings), cut to the
  ``pool`` best predictions;
* the selector model is the search prior (``prior_ranked`` / TPE / GP use it;
  the portfolio shares every observation), and the rule pick plus the
  selector's top predictions are the warm starts;
* the caller's ``check`` is mandatory (the evaluator refuses a backend without
  one), so a config that fails the numeric check can never win.

``mode="lite"`` re-times only the warm starts (the rule pick and the
selector's top ``lite_k``); ``mode="search"`` runs the strategy for
``budget`` evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

from . import selector as S


@dataclass
class GemmTuneResult:
    spec: Any
    candidate: Any
    origin: str
    evaluations: int
    ranked: Tuple[Tuple[float, Any], ...] = field(default=())
    search: Any = None

    def dispatch_result(self, req: Any) -> Any:
        """A launchable :class:`DispatchResult` for the tuned spec."""
        from ..core import make_dispatch_result, pin_to_spec
        from .bf16_rcr import _kernel_id as _kid_bf16
        from .fp16_rcr import _kernel_id as _kid_fp16

        kid = _kid_bf16 if "bf16" in self.candidate.name else _kid_fp16
        probe = pin_to_spec(req, self.candidate, self.spec)
        return make_dispatch_result(
            probe,
            self.candidate,
            self.spec,
            kernel_id=kid(probe, self.candidate, self.spec),
            headline=f"autotuned {self.candidate.name} ({self.origin}) on {req.arch}",
        )


def _registry(req: Any):
    from .bf16_rcr import GEMM_BF16_REGISTRY
    from .common import normalize_dtype
    from .fp16_rcr import GEMM_FP16_REGISTRY

    return (
        GEMM_BF16_REGISTRY
        if normalize_dtype(req.dtype) == "bf16"
        else GEMM_FP16_REGISTRY
    )


def _serving_candidate(registry: Any, c: S.SelectorCandidate, req: Any) -> Any:
    """The registered candidate that serves ``c.spec``: its own candidate when
    the spec is that candidate's select_spec, else the opt-in selected one."""
    by_name = {k.name: k for k in registry.candidates()}
    own = by_name.get(c.candidate_name)
    if own is not None and c.origin == "registered":
        try:
            if S._spec_key(own.select_spec(S._auto(req))) == S._spec_key(c.spec):
                return own
        except Exception:  # noqa: BLE001
            pass
    sid = "cdna_selected_skinny" if S._skinny(c.spec) else "cdna_selected"
    sel = [k for k in registry.candidates() if k.spec_id == sid]
    return sel[0] if sel else own


def gemm_candidates(
    req: Any, *, pool: int = 64
) -> List[Tuple[float, S.SelectorCandidate]]:
    """The selector-ranked candidate set of ``req`` (best first), at most
    ``pool`` entries; the rule pick is always kept."""
    registry = _registry(req)
    areq = S._auto(req)
    try:
        cand = registry.select(areq)
        rule = (cand, cand.select_spec(areq))
    except Exception:  # noqa: BLE001
        rule = None
    model = S.load_model(req.arch)
    if model is not None:
        ranked = S.rank(areq, registry, model=model, rule_pick=rule)
    else:
        ranked = [
            (0.0, c) for c in S.selector_candidates(areq, registry, rule_pick=rule)
        ]
    head = list(ranked[: max(1, int(pool))])
    if not any(c.is_rule_pick for _, c in head):
        head += [(s, c) for s, c in ranked if c.is_rule_pick][:1]
    return head


def tune_gemm(
    req: Any,
    *,
    measure: Callable[[Any, Any], float | Sequence[float]],
    check: Callable[[Any], bool | Tuple[bool, str]],
    mode: str = "search",
    budget: int = 24,
    strategy: str = "portfolio",
    pool: int = 64,
    lite_k: int = 3,
    seed: int = 0,
    **tune_kwargs: Any,
) -> GemmTuneResult:
    """Measure-and-pick for one request (see the module docstring).

    ``measure(spec, fidelity)`` returns a time (or per-rep times) of the
    compiled spec; ``check(spec)`` runs the full-output numeric check.
    """
    from ..tuning.search import Budget, CallablePrior, WarmStart
    from ..tuning.search.api import search_configs
    from ..tuning.search.params import config_items_key, flatten_spec

    registry = _registry(req)
    ranked = gemm_candidates(req, pool=pool)
    if not ranked:
        raise ValueError(f"no candidate for {req}")
    if mode == "lite":
        pool_c = S.diverse_top(ranked, lite_k)
        pool_c += [
            (s, c)
            for s, c in ranked
            if c.is_rule_pick and all(c is not d for _, d in pool_c)
        ]
        ranked = pool_c
    specs = [c.spec for _, c in ranked]
    rows = [dict(flatten_spec(s)) for s in specs]
    table = {config_items_key(r): sc for r, (sc, _) in zip(rows, ranked)}
    prior = CallablePrior(lambda cfg: table[config_items_key(cfg)])
    warm = [
        WarmStart(rows[i], "rule") for i, (_, c) in enumerate(ranked) if c.is_rule_pick
    ]
    warm += [
        WarmStart(rows[i], "selector") for i, _ in enumerate(ranked[: max(1, lite_k)])
    ]
    # the engine's final interleaved confirmation of the top finalists is
    # paid from the same budget: lite mode adds it on top of one pass
    confirm = int(tune_kwargs.pop("confirm", min(3, len(specs))))
    n = len(specs) + confirm if mode == "lite" else max(1, int(budget))
    winner, result = search_configs(
        specs,
        measure=measure,
        check=check,
        strategy="exhaustive" if mode == "lite" else strategy,
        budget=Budget(evaluations=n),
        seed=seed,
        prior=prior,
        warm=warm if mode != "lite" else (),
        confirm=confirm,
        **tune_kwargs,
    )
    if winner is None:
        raise ValueError("autotune found no config that passed the numeric check")
    c = ranked[winner][1]
    return GemmTuneResult(
        spec=c.spec,
        candidate=_serving_candidate(registry, c, req),
        origin=c.origin,
        evaluations=len(getattr(result, "history", []) or []) or n,
        ranked=tuple((s, k.spec) for s, k in ranked),
        search=result,
    )

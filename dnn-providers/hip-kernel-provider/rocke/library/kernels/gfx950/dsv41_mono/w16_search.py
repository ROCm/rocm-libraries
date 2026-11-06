# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""W16 I1: the dsv41 fast-iteration funnel as the search engine's fidelity ladder (PLAN decisions 12 and 15).

The rocKE search engine (``rocke.dispatch.tuning.search``) measures configs through a ``Backend`` and climbs a
fidelity ladder (``halving.SuccessiveHalving`` / ``Hyperband``: cheap rungs on many configs, the last rung on few).
Here a rung is one way to measure a V4.1 arm (a build-time knob set: rocKE stage-function sets linked through
``ATOM_DSV41_RK_EXTERN``, ``rk_lev`` levers, the schedule surface):

* ``proxy="rig"``: the IN-SITU RIG in one running eager engine (ATOM ``rk/rig.py``): every candidate's K1 / K2 is
  launched from the same live state as ATOM's build, checked byte for byte on every launch and timed in shuffled
  rounds against it. Seconds a batch of candidates; many candidates a pass.
* ``proxy="served"``: the SERVED rung: one server, the arm switched at run time (``POST /rk/switch``: rebuild +
  graph re-capture, code objects from the code-object cache), a fixed steady-state decode window timed against the
  base arm in the same server, interleaved (A B A B ...).
* ``proxy=""`` (the last rung, ``Fidelity()``): FRESH-SERVER confirmation, a new server per arm-run (the acceptance
  rung: same-job comparator, A-vs-A twins, rotated passes, both nodes).

A rung is any object with ``run(items, fidelity) -> {key: RungResult}`` (``items``: ``[(key, config)]``); the
transports (an engine handle, an HTTP server) live in the harness. :class:`LadderBackend` turns rungs into the
engine's backend protocol: ``compile`` / ``check`` / ``measure`` take the fidelity and dispatch on its proxy.
:meth:`LadderBackend.prefetch` runs one rung pass for a whole round of configs (one engine pass checks and times
every candidate together); :class:`PrefetchEvaluator` calls it before each round, so ``run_search`` drives the rig
in batches. Samples are ratios to the rung's base arm (lower is faster), one per round of the pass.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rocke.dispatch.tuning.search.measure import CompileOutcome, Evaluator
from rocke.dispatch.tuning.search.trial import Fidelity

__all__ = ["CallableRung", "LadderBackend", "PrefetchEvaluator", "RungResult", "ServedRung", "default_ladder"]


@dataclass
class RungResult:
    """One config's outcome in one rung pass."""

    built: bool
    bitwise: bool | None  # None: the rung did not check (a served rung trusts the rig's / the gate's verdict)
    samples: list[float] = field(default_factory=list)  # ratio to the base arm, one per round
    detail: str = ""
    resources: Mapping[str, float] = field(default_factory=dict)


def default_ladder() -> tuple[Fidelity, ...]:
    """rig (cheap) -> served (one server, switched) -> fresh-server confirmation (the full fidelity)."""
    return (
        Fidelity(name="rig", reps=6, warmup=1, proxy="rig", cost=1.0),
        Fidelity(name="served", reps=4, warmup=1, proxy="served", cost=40.0),
        Fidelity(name="fresh", reps=4, warmup=1, proxy="", cost=200.0),
    )


class LadderBackend:
    """The search engine backend over named rungs (``{proxy: rung}``; ``""`` the fresh-server rung)."""

    def __init__(self, rungs: Mapping[str, Any], *, environment: Mapping[str, object] | None = None,
                 expect_fail: Callable[[Mapping[str, object]], bool] | None = None):  # fmt: skip
        self.rungs = dict(rungs)
        self._environment = dict(environment or {})
        self.expect_fail = expect_fail or (lambda config: False)
        self._results: dict[tuple[str, str], RungResult] = {}
        self._consumed: set[tuple[str, str]] = set()
        self.passes = 0

    def environment(self) -> dict:
        return dict(self._environment)

    def _rung(self, fidelity: Fidelity):
        try:
            return self.rungs[fidelity.proxy]
        except KeyError:
            raise ValueError(f"LadderBackend: no rung for proxy {fidelity.proxy!r} (have {sorted(self.rungs)})") from None

    def prefetch(self, items: Sequence[tuple[str, Mapping[str, object]]], fidelity: Fidelity) -> None:
        """One rung pass for every config of ``items`` not yet measured (or whose samples were used)."""
        todo = [(k, c) for k, c in items if (fidelity.proxy, k) not in self._results
                or (fidelity.proxy, k) in self._consumed]  # fmt: skip
        if not todo:
            return
        got = self._rung(fidelity).run(todo, fidelity)
        self.passes += 1
        for k, _ in todo:
            self._results[(fidelity.proxy, k)] = got.get(k) or RungResult(False, None, detail="no result from the rung")
            self._consumed.discard((fidelity.proxy, k))

    def _get(self, config, key, fidelity) -> RungResult:
        if (fidelity.proxy, key) not in self._results:
            self.prefetch([(key, config)], fidelity)
        return self._results[(fidelity.proxy, key)]

    def compile(self, config, key, fidelity) -> CompileOutcome:
        r = self._get(config, key, fidelity)
        return CompileOutcome(ok=r.built, artifact=key, resources=dict(r.resources), detail=r.detail)

    def check(self, config, key, artifact, fidelity) -> tuple[bool, str]:
        r = self._get(config, key, fidelity)
        if r.bitwise is None:
            return True, "not checked at this rung"
        if self.expect_fail(config):  # a must-fail mutant: passes the search's check only by differing
            return (not r.bitwise), "mutant " + ("caught" if not r.bitwise else "NOT caught")
        return r.bitwise, "bitwise" if r.bitwise else f"differs from ATOM's build: {r.detail}"

    def measure(self, config, key, artifact, fidelity) -> list[float]:
        if (fidelity.proxy, key) in self._consumed:
            self.prefetch([(key, config)], fidelity)
        r = self._get(config, key, fidelity)
        self._consumed.add((fidelity.proxy, key))
        if not r.samples:
            raise RuntimeError(f"no samples for {key} at {fidelity.name}: {r.detail}")
        return list(r.samples)

    def measure_batch(self, items, fidelity) -> list[list[float]]:
        stale = [(k, c) for c, k, _a in items if (fidelity.proxy, k) in self._consumed]
        if stale:
            self.prefetch(stale, fidelity)
        return [self.measure(c, k, a, fidelity) for c, k, a in items]


class PrefetchEvaluator(Evaluator):
    """An ``Evaluator`` that hands each round's fresh configs to ``backend.prefetch`` first, per fidelity, so one
    rung pass checks and times the whole round (the rig's batch economics); the measurement rules are unchanged
    (every artifact checked before its samples are used, spills rejected, results cached)."""

    def evaluate(self, trials, *, max_fresh=None, deadline=None):
        pre = getattr(self.backend, "prefetch", None)
        if callable(pre):
            by_fid: dict[int, list] = {}
            for t in trials:
                key = self.space.key(t.genome)
                fid = self.resolve(t.fidelity)
                if key is None or self.known(key, fid) is not None:
                    continue
                by_fid.setdefault(fid, []).append((key, self.space.decode(t.genome)))
            for fid, items in sorted(by_fid.items()):
                if max_fresh is not None:
                    items = items[: max(0, int(max_fresh))]
                seen, uniq = set(), []
                for k, c in items:
                    if k not in seen:
                        seen.add(k)
                        uniq.append((k, c))
                if uniq:
                    pre(uniq, self.ladder[fid])
        return super().evaluate(trials, max_fresh=max_fresh, deadline=deadline)


class CallableRung:
    """A rung from one callable ``fn(key, config, fidelity) -> RungResult`` (e.g. the fresh-server confirmation: one
    new server per arm-run with its same-job comparator and A-vs-A twin, ``fidelity.reps`` rotated passes)."""

    def __init__(self, fn: Callable[[str, Mapping[str, object], Fidelity], RungResult]):
        self.fn = fn

    def run(self, items, fidelity):
        out = {}
        for key, config in items:
            try:
                out[key] = self.fn(key, config, fidelity)
            except Exception as e:  # noqa: BLE001 - a failed arm-run is a failed build at this rung
                out[key] = RungResult(False, None, [], detail=f"{type(e).__name__}: {e}"[:500])
        return out


class ServedRung:
    """The served rung over two callables: ``switch(env) -> dict`` (the server's ``/rk/switch``) and
    ``window() -> float`` (one fixed steady-state decode window's time per DSpark decode step, a request's median; the
    time per output token also works but carries the acceptance variance, about 6x the noise). Per config:
    ``reps`` interleaved pairs base / arm (the order alternates), sample = arm / base per pair. ``env_of(config)``
    maps a config to its build-time knob set; ``base_env`` is the comparator's."""

    def __init__(self, switch: Callable[[Mapping[str, str]], Mapping], window: Callable[[], float],
                 env_of: Callable[[Mapping[str, object]], Mapping[str, str]], base_env: Mapping[str, str]):  # fmt: skip
        self.switch, self.window, self.env_of, self.base_env = switch, window, env_of, dict(base_env)
        self.log: list[dict] = []

    def run(self, items, fidelity):
        out = {}
        for key, config in items:
            env = dict(self.env_of(config))
            samples = []
            try:
                for rep in range(fidelity.warmup + fidelity.reps):
                    order = [("base", self.base_env), ("arm", env)] if rep % 2 == 0 else [("arm", env), ("base", self.base_env)]
                    got = {}
                    for name, e in order:
                        sw = self.switch(e)
                        got[name] = self.window()
                        self.log.append({"key": key, "rep": rep, "arm": name, "switch_s": sw.get("wall_s"),
                                         "window": got[name]})  # fmt: skip
                    if rep >= fidelity.warmup:
                        samples.append(got["arm"] / got["base"])
                out[key] = RungResult(True, None, samples)
            except Exception as e:  # noqa: BLE001 - an arm that cannot be served is a failed build at this rung
                out[key] = RungResult(False, None, [], detail=f"{type(e).__name__}: {e}"[:500])
        return out

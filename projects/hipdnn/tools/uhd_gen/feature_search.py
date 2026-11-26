# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Search a feature recipe: backward elimination over published columns and derived expressions.

The candidates are the published columns every training row binds (constant columns drop
out; columns that are the same expression collapse) and canonical jsonexpr expressions over
them -- padding, ratios, stride order, grid/wave counts, and rounding up to the tiers a
catalog offers. LightGBM scores a set by grouped cross-validation on the training rows (never
the evaluation slice); the search drops the least important features round by round, keeps
the smallest set whose score is within the noise of the best (the larger of the spread over
seeds and MIN_TOLERANCE), then removes what it still can one feature at a time. The result is
a canonical features-signature array, usable as `uhd_gen generate --feature-signature`.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .coverage import device_field_coverage
from .features import (
    derive_categorical_encoding,
    evaluate_feature_rows,
    feature_reference,
)
from .corpus_io import write_json
from .optional import guarded_read
from .proposal import presence_flag, search_inputs, settle_omissions

logger = logging.getLogger(__name__)

SCHEMA = "uhd_gen.feature_search/1"
RECIPE = "recipe.json"
REPORT = "feature_search_report.json"
SUMMARY = "feature_search_report.md"

PAD_MULTIPLES = (8, 16, 32, 64, 128)
#: Grid tiles when the catalog names no tile-like field.
FALLBACK_TILES = (64, 128)
#: A catalog field that names a tile: block_m, tile_n, ...
_TILE_FIELD = re.compile(r"(^|_)(block|tile)(_|$)")
#: Distinct numeric values a catalog field may hold and still name tiers.
TIER_VALUES = (2, 64)
#: More candidates than this are refused, not sampled.
MAX_CANDIDATES = 2000
DEFAULT_SEEDS = (0, 1, 2)
FOLDS = 5
#: Features dropped per round, as a fraction of those left (at least one).
DROP_FRACTION = 0.15
#: The smallest score spread that counts as noise.
MIN_TOLERANCE = 0.002
#: Rounds stop at this many features.
FLOOR = 8
#: Columns this correlated carry the same information and leave a round together.
CLUSTER_CORRELATION = 0.999
BOOST_ROUNDS = 150
#: The adapter pass tries at most this many candidates.
ADAPTER_PASS_BUDGET = 60
#: The adapters a search is scored for, spelled as `generate --adapter` spells them, each
#: with the model `search_features` scores for it. `linear_data` is absent: it fits a
#: per-problem slowdown, which no pooled regression on the log target stands in for.
SEARCH_ADAPTERS = {"tree_data": "tree", "mlp_data": "mlp_data"}
#: The adapter pass reads one layout of the graphs into folds for every seed (the seed varies
#: the model, not the folds), so these folds are disjoint graph sets: it picks candidates on
#: DISCOVERY, confirms each pick on CONFIRM, checks removals on REMOVAL, and chooses between
#: the pruned and pre-prune sets on CONFIRM + REMOVAL.
ADAPTER_FOLD_SEED = 0
ADAPTER_DISCOVERY = (0, 1)
ADAPTER_CONFIRM = (2, 3)
ADAPTER_REMOVAL = (4,)
#: A paired difference counts when it exceeds this many standard errors.
ADAPTER_Z = 2.0
#: The order templates are offered to the adapter pass.
ADAPTER_TEMPLATES = ("tier", "grid", "pad", "ratio")
#: Sets of at most this many features are fit on every feature at every split.
SUBSAMPLE_ABOVE = 32
LGBM_PARAMS = {
    # Huber on the log target: it fits a few boosting rounds closer than l1, which is what
    # lets a derived feature show a gain over the raw dims it is made of.
    "objective": "huber",
    "alpha": 0.3,
    "learning_rate": 0.2,
    "num_leaves": 63,
    "min_data_in_leaf": 10,
    "feature_fraction": 0.8,
    "max_bin": 127,
    "verbose": -1,
    "deterministic": True,
    "force_row_wise": True,
}

_DIMS = re.compile(r"(.*?)\.?dims\[(\d+)\]")
_STRIDES = re.compile(r"(.*?)\.?strides\[(\d+)\]")


@dataclass(frozen=True)
class Candidate:
    """One feature: its signature entry and its value on every training row."""

    name: str
    entry: object
    template: str
    values: np.ndarray


def _canonical(entry) -> str:
    return (
        entry
        if isinstance(entry, str)
        else json.dumps(entry, sort_keys=True, separators=(",", ":"))
    )


def _literal(value: float):
    """A JSON number the evaluator reads back as the same value."""
    return int(value) if float(value).is_integer() else float(value)


def _margin(error: float, z: float = 1.0) -> float:
    """The smallest difference that counts against noise `error`: `z` times it, and never
    less than MIN_TOLERANCE."""
    return max(z * error, MIN_TOLERANCE)


class _Pool:
    """The candidates, deduplicated by expression and bounded."""

    def __init__(self, bound: int) -> None:
        self.bound = bound
        self.candidates: list[Candidate] = []
        self.stats: dict[str, collections.Counter] = {}
        self._names: set[str] = set()
        #: values -> the candidates that have exactly them
        self._equal: dict[bytes, list[str]] = {}

    def add(self, template: str, entry, values: np.ndarray) -> str | None:
        """Admit `entry`; its name, or None if it is constant or not finite on some row. An
        expression already admitted is admitted once. Candidates that are different
        expressions are all kept, though their values are equal on the rows: nothing in the
        rows says the two differ where they would be read."""
        stats = self.stats.setdefault(template, collections.Counter())
        stats["generated"] += 1
        if not np.isfinite(values).all():
            stats["non_finite"] += 1
            return None
        if values.min() == values.max():
            stats["constant"] += 1
            return None
        name = _canonical(entry)
        if name in self._names:
            stats["duplicate"] += 1
            return name
        if len(self.candidates) >= self.bound:
            raise ValueError(
                f"the search has more than {self.bound} candidate features "
                f"({self.counts()}); refusing rather than sampling. Raise --max-candidates, "
                "or narrow the published columns (`feature-search --columns`; `generate` "
                "skips the search when it is given --features or --feature-signature)"
            )
        digest = hashlib.blake2b(values.tobytes(), digest_size=16).digest()
        self._equal.setdefault(digest, []).append(name)
        self._names.add(name)
        self.candidates.append(Candidate(name, entry, template, values))
        stats["kept"] += 1
        return name

    def counts(self) -> dict[str, int]:
        return {name: stat["kept"] for name, stat in self.stats.items()}

    def equal_candidates(self) -> list[list[str]]:
        """The groups of candidates whose values are equal on every row."""
        return sorted(sorted(names) for names in self._equal.values() if len(names) > 1)


def _column_values(frame: pd.DataFrame, name: str, encoding: dict) -> np.ndarray:
    """A column as the evaluator reads it: numbers as they are, a string column as the codes
    `encoding` (`derive_categorical_encoding`) assigns, an absent binding as NaN."""
    codes = encoding.get(feature_reference(name))
    series = frame[name]
    if codes is None:
        return series.to_numpy(dtype=np.float64)
    return series.map(codes).to_numpy(dtype=np.float64)


class _Term:
    """An expression and its values on the rows, built together so they cannot disagree."""

    def __init__(self, expr, values: np.ndarray) -> None:
        self.expr, self.values = expr, values

    def ceil_div(self, divisor: float) -> "_Term":
        return _Term(
            {"ceil_div": [self.expr, _literal(divisor)]},
            np.ceil(self.values / divisor),
        )

    def mod(self, divisor: float) -> "_Term":
        return _Term(
            {"%": [self.expr, _literal(divisor)]}, np.fmod(self.values, divisor)
        )

    def times(self, literal: float) -> "_Term":
        return _Term(
            {"*": [self.expr, _literal(literal)]}, self.values * float(literal)
        )

    def over(self, other: "_Term") -> "_Term":
        with np.errstate(divide="ignore", invalid="ignore"):
            values = self.values / other.values
        return _Term({"/": [self.expr, other.expr]}, values)

    def over_positive(self, other: "_Term", otherwise: float) -> "_Term":
        """`self / other` where `other` is positive, else `otherwise`. The evaluator's `/`
        declines on a zero divisor, and `if` evaluates only the branch it takes."""
        positive = other.values > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(positive, self.values / other.values, float(otherwise))
        return _Term(
            {
                "if": [
                    {">": [other.expr, 0]},
                    {"/": [self.expr, other.expr]},
                    _literal(otherwise),
                ]
            },
            values,
        )

    @property
    def is_guarded(self) -> bool:
        """Whether this reads an optional feature through its default."""
        return isinstance(self.expr, dict) and "value_or_default" in self.expr


def _product(terms: list[_Term]) -> _Term:
    values = terms[0].values
    for term in terms[1:]:
        values = values * term.values
    return _Term({"*": [term.expr for term in terms]}, values)


def _is_integral(values: np.ndarray) -> bool:
    return bool(np.all(values == np.floor(values)))


def _derived_candidates(
    pool: _Pool,
    frame: pd.DataFrame,
    columns: list[str],
    representative: dict[str, str],
    catalog_values: dict[str, list[float]],
    tiles: list[int],
    stats: dict,
    column_terms: dict[str, _Term],
) -> None:
    """Add the pad, ratio, stride-order, grid and tier templates over the base columns that
    the pool kept (`representative`: column name -> its candidate's name).

    A column's term is its signature entry: a bare `$name`, or `value_or_default` of it where
    the column is an optional feature, so every expression built on it reads it guarded.
    """
    by_name = {candidate.name: candidate for candidate in pool.candidates}

    def term(name: str) -> _Term:
        candidate = by_name[name]
        return _Term(candidate.entry, candidate.values)

    # The dim columns: `...dims[<k>]` with integer values that vary, by tensor role. Each
    # role reads its own dims, whether or not another tensor's equal them on the rows.
    roles: dict[str, list[tuple[int, str]]] = {}
    for name in sorted(representative):
        match = _DIMS.fullmatch(name)
        if match:
            values = by_name[representative[name]].values
            if _is_integral(values):
                roles.setdefault(match.group(1), []).append(
                    (int(match.group(2)), representative[name])
                )
    all_dims = sorted({name for dims in roles.values() for _, name in dims})
    role_of = {}
    for role, dims in sorted(roles.items()):
        for _, name in dims:
            role_of.setdefault(name, role)

    for name in all_dims:
        dim = term(name)
        top = dim.values.max()
        for multiple in PAD_MULTIPLES:
            if multiple >= top:
                continue
            padded = dim.ceil_div(multiple).times(multiple)
            pool.add("pad", padded.expr, padded.values)
            # An optional dim reads its default (-1) where unbound, which pads to 0: a full
            # tensor is the fill there, as if nothing were padded.
            fill = dim.over_positive(padded, 1) if dim.is_guarded else dim.over(padded)
            pool.add("pad", fill.expr, fill.values)
            remainder = dim.mod(multiple)
            pool.add("pad", remainder.expr, remainder.values)

    for first, second in itertools.combinations(all_dims, 2):
        if role_of[first] != role_of[second]:
            ratio = term(first).over(term(second))
            pool.add("ratio", ratio.expr, ratio.values)

    # Stride order: which of two adjacent axes of a tensor is the outer one (head-major or
    # token-major), as strides[i] > strides[i+1] for each pair before the innermost stride.
    strides: dict[str, list[tuple[int, str]]] = {}
    for name in sorted(columns):
        match = _STRIDES.fullmatch(name)
        if match:
            strides.setdefault(match.group(1), []).append((int(match.group(2)), name))

    def stride(name: str) -> _Term | None:
        # The column itself, whatever else its values equal.
        if frame[name].map(lambda value: isinstance(value, str)).any():
            return None
        return column_terms[name]

    for role, members in sorted(strides.items()):
        if len(members) < 3:
            continue
        members.sort()
        for (_, outer), (_, inner) in zip(members[:-2], members[1:-1]):
            first, second = stride(outer), stride(inner)
            if first is not None and second is not None:
                order = _Term(
                    {">": [first.expr, second.expr]},
                    (first.values > second.values).astype(np.float64),
                )
                pool.add("stride_order", order.expr, order.values)

    cu = None
    if "device.cu_count" in frame.columns:
        coverage = device_field_coverage(
            frame[[c for c in ("device", "device.cu_count") if c in frame.columns]]
        )
        if coverage["fields"]["device.cu_count"]["varies"]:
            cu = _Term("$device.cu_count", frame["device.cu_count"].to_numpy(float))
    # A grid: the tiles of one non-innermost axis times the extents of the tensor's other
    # non-innermost axes (the innermost axis is the contiguous one: it is read along, not
    # tiled across the launch).
    every_dim: dict[str, set[int]] = {}
    for name in columns:
        match = _DIMS.fullmatch(name)
        if match:
            every_dim.setdefault(match.group(1), set()).add(int(match.group(2)))
    for role, dims in sorted(roles.items()):
        if len(every_dim[role]) < 3:
            continue
        innermost = max(every_dim[role])
        for index, stride_column in strides.get(role, ()):
            if np.all(column_terms[stride_column].values == 1):
                innermost = index
        members = sorted({name for index, name in dims if index != innermost})
        for name in members:
            dim = term(name)
            others = [term(other) for other in members if other != name]
            for tile in tiles:
                if tile >= dim.values.max():
                    continue
                grid = _product(others + [dim.ceil_div(tile)])
                pool.add("grid", grid.expr, grid.values)
                if cu is not None:
                    waves = grid.over(cu)
                    pool.add("grid", waves.expr, waves.values)
    stats["grid_per_cu"] = (
        "added" if cu is not None else "skipped: device.cu_count absent or constant"
    )

    for field, values in sorted(catalog_values.items()):
        # A 0/1 flag, or a ladder with one value above 1, rounds to nothing but itself.
        ladder = sorted(values)
        if sum(1 for value in ladder if value > 1) < 2:
            continue
        for name in all_dims:
            dim = term(name)
            if ladder[-1] < dim.values.min() or ladder[0] >= dim.values.max():
                continue
            chain: list = []
            for value in ladder:
                chain += [{"<=": [dim.expr, _literal(value)]}, _literal(value)]
            conditions = [dim.values <= value for value in ladder]
            rounded = _Term(
                {"if": [*chain, dim.expr]},
                np.select(conditions, [float(value) for value in ladder], dim.values),
            )
            pool.add("tier", rounded.expr, rounded.values)
            fill = dim.over(rounded)
            pool.add("tier", fill.expr, fill.values)


def _column_terms(
    frame: pd.DataFrame,
    names: list[str],
    guards: dict[str, object],
    executable: str | None,
) -> dict[str, _Term]:
    """Each published column as the signature reads it. A column in `guards` (an optional
    feature, with the default a model reads for it when unbound) is
    `value_or_default($name, default)`, and its values are the shared evaluator's on every
    row, the ones that leave it unbound included; any other column is read bare."""
    guarded = sorted(name for name in names if name in guards)
    bare = [name for name in names if name not in guards]
    encoding = derive_categorical_encoding(frame, bare)
    terms = {
        name: _Term("$" + name, _column_values(frame, name, encoding)) for name in bare
    }
    if guarded:
        entries = [guarded_read("$" + name, guards[name]) for name in guarded]
        _, values = evaluate_feature_rows(frame, entries, {}, executable)
        matrix = np.asarray(values, dtype=np.float64).reshape(len(frame), len(entries))
        terms.update(
            (name, _Term(entry, matrix[:, place]))
            for place, (name, entry) in enumerate(zip(guarded, entries))
        )
    return terms


def build_candidates(
    frame: pd.DataFrame,
    columns: list[str],
    presence_columns: list[str],
    catalog_values: dict[str, list[float]] | None,
    max_candidates: int,
    tiles: list[int] = FALLBACK_TILES,
    guards: dict[str, object] | None = None,
    feature_evaluator: str | None = None,
) -> tuple[_Pool, dict]:
    """(pool, details): the base columns and presence flags, then the derived templates.

    `guards` maps an optional column to the default it is read with when unbound; every
    candidate over it reads it through `value_or_default`."""
    pool = _Pool(max_candidates)
    details: dict = {}
    representative: dict[str, str] = {}
    terms = _column_terms(
        frame, [*columns, *presence_columns], guards or {}, feature_evaluator
    )
    for name in sorted(columns):
        kept = pool.add("base", terms[name].expr, terms[name].values)
        if kept is not None:
            representative[name] = kept
    for name in sorted(presence_columns):
        # Whether the tensor is bound, never its identifier's value; a guarded identifier
        # reads -1 (the sentinel an absent tensor has) where it is unbound.
        flag = presence_flag(name, (guards or {}).get(name))
        pool.add("presence", flag, (terms[name].values != -1).astype(float))
    details["constant_columns"] = sorted(
        name for name in columns if name not in representative
    )
    details["grid_tiles"] = list(tiles)
    _derived_candidates(
        pool,
        frame,
        columns,
        representative,
        catalog_values or {},
        list(tiles),
        details,
        terms,
    )
    return pool, details


def tile_sizes(catalog_values: dict) -> tuple[list[int], str]:
    """(tiles, source): the integer values of the catalog's tile-like fields (block_m, tile_n,
    ...), else FALLBACK_TILES."""
    found = sorted(
        {
            int(value)
            for field, values in catalog_values.items()
            if _TILE_FIELD.search(field)
            for value in values
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value).is_integer()
            and value > 1
        }
    )
    return (found, "catalog") if found else (list(FALLBACK_TILES), "fallback")


def _tier_fields(catalog_values: dict) -> tuple[dict, dict]:
    """(fields that name tiers, why the others do not): 2 to 64 distinct numeric values."""
    usable, skipped = {}, {}
    low, high = TIER_VALUES
    for field, values in sorted(catalog_values.items()):
        numbers = {
            float(value)
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if len(numbers) == len(set(map(str, values))) and low <= len(numbers) <= high:
            usable[field] = sorted(numbers)
        else:
            skipped[field] = f"not {low}..{high} distinct numeric values"
    return usable, skipped


def catalog_values_from_tree(tree: Path, engine: str | None) -> dict[str, list]:
    """The numeric metadata values of an engine's KDP kernels, by field.

    `engine` is a UED name or id; None reads every kernel in the tree."""
    from .provenance import load_descriptor_tree

    index = load_descriptor_tree(tree)
    identities = {engine}
    for identity, (_, ued) in index["ued"].items():
        if engine in (identity, ued.get("name")):
            identities.add(identity)
    values: dict[str, set] = {}
    for _, kdp in index["kdp"].values():
        if engine is not None and kdp.get("engine") not in identities:
            continue
        for kernel in kdp.get("kernelDescriptors", []):
            for field, value in (kernel.get("metadata") or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.setdefault(field, set()).add(value)
    return {field: sorted(found) for field, found in sorted(values.items())}


def _folds(group_ids: np.ndarray, seed: int, folds: int) -> np.ndarray:
    """A fold per row: whole problems, shuffled by `seed`, dealt round-robin."""
    problems = np.unique(group_ids)
    order = np.random.default_rng(seed).permutation(len(problems))
    fold_of_problem = np.empty(len(problems), dtype=np.int64)
    fold_of_problem[order] = np.arange(len(problems)) % folds
    return fold_of_problem[np.searchsorted(problems, group_ids)]


class _Scorer:
    """Grouped K-fold held-out MAE of LightGBM on the log target, over the seed list."""

    def __init__(
        self, matrix: np.ndarray, target: np.ndarray, groups: np.ndarray, seeds, folds
    ) -> None:
        import lightgbm

        self.lightgbm = lightgbm
        self.groups = groups
        self.matrix, self.target = matrix, target
        self.seeds, self.folds = list(seeds), folds
        self.fold_of = [_folds(groups, seed, folds) for seed in self.seeds]
        #: the adapter pass's folds: one layout, whatever the seeds
        self.adapter_fold_of = _folds(groups, ADAPTER_FOLD_SEED, folds)
        self.fits = 0

    def score(
        self, columns: list[int], clusters: list[list[int]] | None = None
    ) -> tuple[float, float, np.ndarray | None]:
        """(mean MAE over seeds, its std over seeds, permutation importance per cluster).

        Importance is the rise in held-out error when a cluster's columns are shuffled
        together within the held-out fold."""
        position = {column: place for place, column in enumerate(columns)}
        places = [[position[c] for c in cluster] for cluster in clusters or ()]
        errors, rise = [], np.zeros(len(places))
        for seed, fold_of in zip(self.seeds, self.fold_of):
            total = 0.0
            shuffled = np.zeros(len(places))
            params = {**LGBM_PARAMS, "seed": seed}
            if len(columns) <= SUBSAMPLE_ABOVE:
                # A tree that sees a random 80% of two features often sees only the noise.
                params["feature_fraction"] = 1.0
            for fold in range(self.folds):
                train = np.flatnonzero(fold_of != fold)
                test = np.flatnonzero(fold_of == fold)
                booster = self.lightgbm.train(
                    params,
                    self.lightgbm.Dataset(
                        self.matrix[np.ix_(train, columns)],
                        self.target[train],
                        params={"max_bin": LGBM_PARAMS["max_bin"], "verbose": -1},
                    ),
                    num_boost_round=BOOST_ROUNDS,
                )
                self.fits += 1
                held = self.matrix[np.ix_(test, columns)]
                actual = self.target[test]
                base = np.abs(booster.predict(held) - actual).sum()
                total += base
                order = np.random.default_rng(1000 * seed + fold).permutation(len(test))
                for index, place in enumerate(places):
                    saved = held[:, place].copy()
                    held[:, place] = saved[order]
                    shuffled[index] += (
                        np.abs(booster.predict(held) - actual).sum() - base
                    )
                    held[:, place] = saved
            errors.append(total / len(self.target))
            rise += shuffled / len(self.target)
        importance = rise / len(self.seeds) if clusters else None
        return float(np.mean(errors)), float(np.std(errors)), importance


def _clusters(matrix: np.ndarray) -> list[list[int]]:
    """Connected components of the columns correlated past CLUSTER_CORRELATION."""
    centered = matrix - matrix.mean(axis=0)
    centered /= np.linalg.norm(centered, axis=0)
    correlated = np.abs(centered.T @ centered) > CLUSTER_CORRELATION
    parent = list(range(matrix.shape[1]))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for first, second in zip(*np.nonzero(np.triu(correlated, 1))):
        parent[find(int(second))] = find(int(first))
    members: dict[int, list[int]] = {}
    for column in range(matrix.shape[1]):
        members.setdefault(find(column), []).append(column)
    return sorted(members.values())


def _prune(
    scorer: "_Scorer",
    current: list[int],
    names: list[str],
    estimate: dict,
    dropped: list,
) -> tuple[list[int], dict, float, float, list]:
    """Remove features until each one left costs more than the noise to remove: refit
    without it and the held-out score must rise by more than max(spread, MIN_TOLERANCE).

    The weakest feature by its last known cost (to begin with, `estimate`: its permutation
    importance) is refit without; it goes if the cost is within the noise, else its cost is
    updated. Costs are exact only on a sweep of the whole set, so a sweep follows any removal
    and ends the pass when it finds nothing weak. Returns (features, their exact costs,
    score, spread, steps)."""
    current = sorted(current)
    score, spread, _ = scorer.score(current)
    costs = dict(estimate)
    steps = []
    fresh = False
    while True:
        while len(current) > 1:
            threshold = _margin(spread)
            weak = [c for c in current if costs[c] <= threshold]
            if not weak:
                break
            worst = min(weak, key=lambda c: (costs[c], names[c]))
            trial = [c for c in current if c != worst]
            trial_score, trial_spread, _ = scorer.score(trial)
            cost = trial_score - score
            removed = cost <= threshold
            steps.append({"feature": names[worst], "cost": cost, "removed": removed})
            logger.info(
                "feature search prune: %s %s (cost %+.5f, tolerance %.5f, %d left)",
                "removed" if removed else "kept",
                names[worst],
                cost,
                threshold,
                len(current) - removed,
            )
            if removed:
                current, score, spread = trial, trial_score, trial_spread
                del costs[worst]
                fresh = False
                dropped.append(
                    {
                        "feature": names[worst],
                        "reason": f"prune: removal cost {cost:+.5f} within the noise "
                        f"{threshold:.5f}",
                    }
                )
            else:
                costs[worst] = cost
        if fresh or len(current) == 1:
            break
        costs = {
            c: scorer.score([x for x in current if x != c])[0] - score for c in current
        }
        fresh = True
    if len(current) == 1:
        costs = {current[0]: None}
    return current, costs, score, spread, steps


class _MlpScorer:
    """Held-out error of the real MLP trainer (`mlp.fit_mlp`, one ensemble member, early
    stopping on whole graphs as `train` does, `options` its hidden/epochs/max_rows) per
    (seed, fold) cell, on the adapter pass's one fold layout: the seed varies the model, so
    the cells of a fold are repeats of one held-out set. The cells of a set are paired with
    another set's."""

    def __init__(self, scorer: "_Scorer", options: dict | None = None) -> None:
        self.scorer = scorer
        self.options = dict(options or {})
        self.fits = 0
        self._cells: dict = {}

    def cells(self, columns: list[int], folds: tuple[int, ...]) -> np.ndarray:
        """The held-out error of each (seed, fold) cell, shape (seeds, folds)."""
        key = (tuple(columns), folds)
        if key in self._cells:
            return self._cells[key]
        from .mlp import fit_mlp

        scorer = self.scorer
        fold_of = scorer.adapter_fold_of
        errors = []
        for seed in scorer.seeds:
            for fold in folds:
                train = np.flatnonzero(fold_of != fold)
                test = np.flatnonzero(fold_of == fold)
                ensemble, _ = fit_mlp(
                    scorer.matrix[np.ix_(train, columns)],
                    scorer.target[train],
                    **{**self.options, "members": 1, "seed": seed},
                    groups=scorer.groups[train],
                )
                self.fits += 1
                predicted = ensemble.predict(scorer.matrix[np.ix_(test, columns)])
                errors.append(float(np.abs(predicted - scorer.target[test]).mean()))
        self._cells[key] = np.array(errors).reshape(len(scorer.seeds), len(folds))
        return self._cells[key]


def _paired(worse: np.ndarray, better: np.ndarray) -> tuple[float, float]:
    """(mean, standard error) of the fall in error from `worse` to `better`, cells shaped
    (seeds, folds).

    The seeds of one fold score the same held-out graphs, so they are averaged first and the
    standard error is taken over the folds: cells that share graphs are not independent
    draws. One fold has no spread to estimate: its standard error is 0, so the margin is the
    MIN_TOLERANCE floor."""
    difference = (worse - better).mean(axis=0)
    count = len(difference)
    error = float(difference.std(ddof=1) / math.sqrt(count)) if count > 1 else 0.0
    return float(difference.mean()), error


def _adapter_candidates(
    scorer: "_Scorer", candidates: list[Candidate], current: list[int], budget: int
) -> list[tuple[int, float]]:
    """The derived features the adapter pass tries, as (index, LightGBM gain): per template
    (tier, grid, pad, ratio) those not already kept, best gain first, dealt out across the
    templates in that order until `budget` are chosen, so no template takes the whole pass.
    The gain comes from a fit, with the first seed, on the rows outside the discovery folds.
    """
    pool = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.template in ADAPTER_TEMPLATES and index not in current
    ]
    if not pool:
        return []
    columns = sorted(set(current) | set(pool))
    rows = np.flatnonzero(scorer.adapter_fold_of > max(ADAPTER_DISCOVERY))
    booster = scorer.lightgbm.train(
        {**LGBM_PARAMS, "seed": scorer.seeds[0]},
        scorer.lightgbm.Dataset(
            scorer.matrix[np.ix_(rows, columns)],
            scorer.target[rows],
            params={"max_bin": LGBM_PARAMS["max_bin"], "verbose": -1},
        ),
        num_boost_round=BOOST_ROUNDS,
    )
    gain = dict(zip(columns, booster.feature_importance("gain")))
    queues = [
        sorted(
            (index for index in pool if candidates[index].template == template),
            key=lambda index: (-gain[index], candidates[index].name),
        )
        for template in ADAPTER_TEMPLATES
    ]
    chosen = []
    while len(chosen) < budget and any(queues):
        for queue in queues:
            if queue and len(chosen) < budget:
                index = queue.pop(0)
                chosen.append((index, float(gain[index])))
    return chosen


def _adapter_pass(
    scorer: "_Scorer",
    candidates: list[Candidate],
    current: list[int],
    budget: int,
    alternative: list[int] | None = None,
    options: dict | None = None,
) -> tuple[list[int], dict, dict]:
    """Forward pass scored by the MLP the model will be trained as, on paired per-fold
    differences (a fold's seeds averaged, the standard error over the folds): a derived
    feature is picked if the held-out error on the DISCOVERY folds falls by more than
    max(ADAPTER_Z standard errors, MIN_TOLERANCE), and kept only if the CONFIRM folds agree
    (by more than one standard error, and MIN_TOLERANCE); an added feature is dropped again
    if the REMOVAL fold, one fold whose standard error is 0, finds it costs no more than
    MIN_TOLERANCE. The LightGBM prune can remove what the MLP uses, so `alternative` (the set
    before the prune) plus the added features is scored too, and kept only if it beats the
    pruned set by the pick's margin on the CONFIRM and REMOVAL folds. Returns (features,
    report, the kept added features' removal costs).
    """
    started = time.monotonic()
    mlp = _MlpScorer(scorer, options)
    current = list(current)
    discovery, confirm = ADAPTER_DISCOVERY, ADAPTER_CONFIRM
    base = float(mlp.cells(sorted(current), discovery).mean())
    tried, added = [], []
    for index, gain in _adapter_candidates(scorer, candidates, current, budget):
        before = mlp.cells(sorted(current), discovery)
        after = mlp.cells(sorted(current + [index]), discovery)
        delta, error = _paired(before, after)
        picked = delta > _margin(error, ADAPTER_Z)
        entry = {
            "feature": candidates[index].name,
            "template": candidates[index].template,
            "lightgbm_gain": gain,
            "mlp_score": float(after.mean()),
            "mlp_delta": delta,
            "mlp_delta_se": error,
            "picked": picked,
            "confirm_delta": None,
            "added": False,
        }
        if picked:
            entry["confirm_delta"], confirm_error = _paired(
                mlp.cells(sorted(current), confirm),
                mlp.cells(sorted(current + [index]), confirm),
            )
            entry["added"] = entry["confirm_delta"] > _margin(confirm_error)
        tried.append(entry)
        logger.info(
            "feature search adapter pass: %s %s (delta %+.5f +- %.5f, confirm %s)",
            "added" if entry["added"] else "tried",
            candidates[index].name,
            delta,
            error,
            entry["confirm_delta"],
        )
        if entry["added"]:
            current.append(index)
            added.append(index)
    checks, costs = [], {}
    for index in reversed(added):
        without = sorted(c for c in current if c != index)
        cost, error = _paired(
            mlp.cells(without, ADAPTER_REMOVAL),
            mlp.cells(sorted(current), ADAPTER_REMOVAL),
        )
        # One fold has no spread to estimate (its error is 0): the floor alone decides.
        kept = cost > _margin(error)
        checks.append(
            {"feature": candidates[index].name, "mlp_removal_cost": cost, "kept": kept}
        )
        if kept:
            costs[index] = cost
        else:
            current.remove(index)
    choice = None
    if alternative is not None:
        wider = sorted(set(alternative) | set(costs))
        if wider != sorted(current):
            held = ADAPTER_CONFIRM + ADAPTER_REMOVAL
            gain, error = _paired(
                mlp.cells(sorted(current), held), mlp.cells(wider, held)
            )
            keep_wider = gain > _margin(error, ADAPTER_Z)
            choice = {
                "pruned_features": len(current),
                "pruned_mlp_score": float(mlp.cells(sorted(current), held).mean()),
                "pre_prune_features": len(wider),
                "pre_prune_mlp_score": float(mlp.cells(wider, held).mean()),
                "pre_prune_gain": gain,
                "pre_prune_gain_se": error,
                "chosen": "pre_prune" if keep_wider else "pruned",
            }
            if keep_wider:
                current = wider
        else:
            choice = {"chosen": "pruned", "pre_prune_equals_pruned": True}
    final = sorted(current)
    report = {
        "adapter": "mlp_data",
        "set_choice": choice,
        "budget": budget,
        "members": 1,
        "folds": {
            "discovery": list(ADAPTER_DISCOVERY),
            "confirm": list(ADAPTER_CONFIRM),
            "removal": list(ADAPTER_REMOVAL),
        },
        "options": {
            k: list(v) if isinstance(v, tuple) else v for k, v in mlp.options.items()
        },
        "base_score": base,
        "final_score": float(mlp.cells(final, discovery).mean()),
        "candidates": tried,
        "removal_check": checks,
        "added": [candidates[index].name for index in sorted(costs)],
        "mlp_fits": mlp.fits,
        "wall_seconds": time.monotonic() - started,
    }
    return final, report, costs


def search_features(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    presence_columns: list[str] = (),
    target: str,
    objective: str,
    groups: pd.Series,
    catalog_values: dict | None = None,
    catalog_source: str | None = None,
    seeds=DEFAULT_SEEDS,
    max_candidates: int = MAX_CANDIDATES,
    floor: int = FLOOR,
    omitted: list = (),
    feature_evaluator: str | None = None,
    adapter: str = "tree",
    adapter_pass_budget: int = ADAPTER_PASS_BUDGET,
    mlp_options: dict | None = None,
    guards: dict[str, object] | None = None,
) -> dict:
    """Search a recipe over the training rows `frame`; the report, whose `recipe` is the
    features signature.

    `columns` are the published columns the search may read (already past the exclusion
    rules), `presence_columns` the tensor identifiers it may read only as presence flags,
    `guards` the optional ones among them with the default a model reads when unbound (rows
    that leave them unbound stay in the frame and take it, as the runtime does),
    `groups` the graph each row belongs to: every fold holds whole graphs, so a graph's
    rows on several devices never fall on both sides. The target's log is the regression label for
    either objective: the search wants the features that explain it, not a ranking.

    `seeds` deal the LightGBM validation folds anew; the adapter pass deals its folds once
    (ADAPTER_FOLD_SEED), whatever the seeds, and its seeds vary the MLP.

    `adapter` is the model the recipe will train: with `mlp_data` a forward pass scored by
    the MLP follows the LightGBM elimination."""
    started = time.monotonic()
    measured = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=np.float64)
    rows = np.isfinite(measured) & (measured > 0)
    if not rows.any():
        raise ValueError(f"no training row has a positive {target}")
    unusable = int((~rows).sum())
    frame, groups = frame[rows], groups[rows]
    group_ids = pd.factorize(groups)[0]
    if group_ids.max() + 1 < FOLDS:
        raise ValueError(
            f"the search needs at least {FOLDS} training problems for its {FOLDS}-fold "
            f"validation; got {group_ids.max() + 1}"
        )
    tiers, skipped_fields = _tier_fields(catalog_values or {})
    tiles, tile_source = tile_sizes(catalog_values or {})
    pool, details = build_candidates(
        frame,
        list(columns),
        list(presence_columns),
        tiers,
        max_candidates,
        tiles,
        guards,
        feature_evaluator,
    )
    candidates = pool.candidates
    if not candidates:
        raise ValueError("every candidate feature is constant on the training rows")
    names = [candidate.name for candidate in candidates]
    matrix = np.column_stack([candidate.values for candidate in candidates])
    label = np.log(frame[target].to_numpy(dtype=np.float64))
    logger.info(
        "feature search: %d row(s), %d problem(s), %d candidate(s) %s",
        len(frame),
        group_ids.max() + 1,
        len(candidates),
        pool.counts(),
    )
    scorer = _Scorer(matrix, label, group_ids, seeds, FOLDS)
    clusters = _clusters(matrix)
    cluster_of = {column: index for index, c in enumerate(clusters) for column in c}

    # Backward elimination from every candidate.
    active = list(range(len(clusters)))
    rounds, dropped = [], []
    best = None
    worse = 0
    while True:
        columns_now = sorted(c for index in active for c in clusters[index])
        score, spread, importance = scorer.score(
            columns_now, [clusters[index] for index in active]
        )
        rounds.append(
            {
                "round": len(rounds),
                "score": score,
                "spread": spread,
                "features": len(columns_now),
                "kept": [names[c] for c in columns_now],
                "importance": dict(zip(active, importance)),
                "active": list(active),
            }
        )
        logger.info(
            "feature search round %d: %d feature(s), score %.5f +- %.5f",
            len(rounds) - 1,
            len(columns_now),
            score,
            spread,
        )
        if best is None or score < best["score"]:
            best = rounds[-1]
        worse = worse + 1 if score > best["score"] + _margin(best["spread"]) else 0
        if worse >= 2 or len(columns_now) <= floor:
            break
        # Whole clusters leave together, least important first.
        order = sorted(
            range(len(active)),
            key=lambda place: (importance[place], clusters[active[place]][0]),
        )
        wanted = max(1, math.ceil(DROP_FRACTION * len(columns_now)))
        leaving, count = [], 0
        for place in order:
            size = len(clusters[active[place]])
            if count >= wanted:
                break
            if len(columns_now) - count - size >= floor:
                leaving.append(active[place])
                count += size
        if not leaving:
            break
        for index in leaving:
            dropped.extend(
                {
                    "feature": names[c],
                    "reason": f"round {len(rounds) - 1}: least important cluster",
                }
                for c in clusters[index]
            )
        active = [index for index in active if index not in leaving]

    # The smallest set whose score is within the noise of the best round: the spread over
    # seeds, and not under MIN_TOLERANCE.
    chosen = min(
        (r for r in rounds if r["score"] <= best["score"] + _margin(best["spread"])),
        key=lambda r: (r["features"], r["score"]),
    )
    survivors = sorted(c for index in chosen["active"] for c in clusters[index])
    rank = {c: chosen["importance"][cluster_of[c]] for c in survivors}
    current, costs, final_score, final_spread, pruned = _prune(
        scorer,
        survivors,
        names,
        {c: float(rank[c]) for c in survivors},
        dropped,
    )
    adapter_report, mlp_costs = None, {}
    if adapter == "mlp_data":
        current, adapter_report, mlp_costs = _adapter_pass(
            scorer, candidates, current, adapter_pass_budget, survivors, mlp_options
        )
    final = sorted(current)
    recipe = [candidates[c].entry for c in final]
    verify_recipe(recipe, [candidates[c] for c in final], frame, feature_evaluator)
    return {
        "schema": SCHEMA,
        "target": target,
        "objective": objective,
        "rows": len(frame),
        "rows_without_positive_target": unusable,
        "problems": int(group_ids.max() + 1),
        "seeds": list(seeds),
        "folds": FOLDS,
        "model": {"lightgbm": LGBM_PARAMS, "boost_rounds": BOOST_ROUNDS},
        "exclusions": settle_omissions(omitted, recipe),
        "catalog": {
            "source": catalog_source,
            "tier_fields": tiers,
            "skipped_fields": skipped_fields,
            "note": (
                None
                if tiers
                else "no catalog field with 2..64 numeric values: the tier template was skipped"
            ),
        },
        "candidates": {
            "total": len(candidates),
            "by_template": {
                template: dict(stat) for template, stat in sorted(pool.stats.items())
            },
            "constant_columns": details["constant_columns"],
            "equal_candidates": pool.equal_candidates(),
            "grid_per_cu": details["grid_per_cu"],
            "grid_tiles": {"tiles": details["grid_tiles"], "source": tile_source},
            "correlated_clusters": [
                [names[column] for column in cluster]
                for cluster in clusters
                if len(cluster) > 1
            ],
        },
        "rounds": [
            {
                key: round_[key]
                for key in ("round", "features", "score", "spread", "kept")
            }
            for round_ in rounds
        ],
        "selected": {
            "round": chosen["round"],
            "features": chosen["features"],
            "score": chosen["score"],
            "spread": chosen["spread"],
            "best_round": best["round"],
            "best_score": best["score"],
        },
        "prune": pruned,
        "adapter_pass": adapter_report,
        # The score and removal costs are LightGBM's, on the pruned set; a feature the
        # adapter pass added carries the MLP's removal cost instead. When the adapter
        # pass kept the pre-prune set, its extra features have neither.
        "final": {
            "score": final_score,
            "spread": final_spread,
            "mlp_score": (
                None if adapter_report is None else adapter_report["final_score"]
            ),
            "features": [
                {
                    "feature": names[c],
                    "template": candidates[c].template,
                    "removal_cost": costs.get(c),
                    "mlp_removal_cost": mlp_costs.get(c),
                }
                for c in final
            ],
        },
        "dropped": dropped,
        "model_fits": scorer.fits,
        "wall_seconds": time.monotonic() - started,
        "recipe": recipe,
    }


def verify_recipe(
    recipe: list, kept: list[Candidate], frame: pd.DataFrame, executable: str | None
) -> None:
    """Every row of the recipe through the shared evaluator, against the values the search
    scored: the search's arithmetic must be the runtime's."""
    encoding = derive_categorical_encoding(
        frame, [entry[1:] for entry in recipe if isinstance(entry, str)]
    )
    _, values = evaluate_feature_rows(frame, recipe, encoding, executable)
    expected = np.column_stack([candidate.values for candidate in kept])
    actual = np.asarray(values, dtype=np.float64)
    wrong = ~np.all(np.isclose(actual, expected, rtol=1e-9, atol=0), axis=0)
    if wrong.any():
        names = [kept[c].name for c in np.flatnonzero(wrong)]
        raise ValueError(
            f"the shared evaluator disagrees with the search on {names}; refusing the recipe"
        )


def write_search(report: dict, directory: Path) -> dict[str, Path]:
    """`recipe.json`, the full report and its Markdown summary under `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "recipe": directory / RECIPE,
        "report": directory / REPORT,
        "summary": directory / SUMMARY,
    }
    for path, value in (
        (paths["recipe"], report["recipe"]),
        (paths["report"], report),
    ):
        write_json(path, value)
    paths["summary"].write_text(_summary(report), encoding="utf-8", newline="\n")
    return paths


def search_summary(report: dict, paths: dict[str, Path]) -> dict:
    """What `generation_manifest.json` records of a search."""
    return {
        "report": str(paths["report"]),
        "recipe": str(paths["recipe"]),
        "candidates": report["candidates"]["total"],
        "by_template": {
            template: stat.get("kept", 0)
            for template, stat in report["candidates"]["by_template"].items()
        },
        "rounds": len(report["rounds"]),
        "features": len(report["recipe"]),
        "adapter": report["adapter_pass"] and report["adapter_pass"]["adapter"],
        "set_choice": report["adapter_pass"] and report["adapter_pass"]["set_choice"],
        "mlp_score": report["adapter_pass"] and report["adapter_pass"]["final_score"],
        "adapter_added": report["adapter_pass"] and report["adapter_pass"]["added"],
        "score": report["final"]["score"],
        "spread": report["final"]["spread"],
        "wall_seconds": report["wall_seconds"],
    }


def _summary(report: dict) -> str:
    lines = [
        "# Feature search",
        "",
        f"- target: `{report['target']}` ({report['objective']}), log MAE over "
        f"{report['folds']}-fold validation by problem, seeds {report['seeds']}",
        f"- rows: {report['rows']} in {report['problems']} problems",
        f"- candidates: {report['candidates']['total']} "
        + ", ".join(
            f"{template} {stat.get('kept', 0)}"
            for template, stat in report["candidates"]["by_template"].items()
        ),
        f"- exclusions: {len(report['exclusions'])} column(s) "
        + ", ".join(
            sorted(
                {
                    f"{reason}"
                    for reason in (e.get("reason") for e in report["exclusions"])
                    if reason
                }
            )
        ),
        f"- rounds: {len(report['rounds'])}, best round {report['selected']['best_round']} "
        f"({report['selected']['best_score']:.5f}); selected round "
        f"{report['selected']['round']} ({report['selected']['features']} features, "
        f"{report['selected']['score']:.5f})",
        f"- final: {len(report['final']['features'])} features, score "
        f"{report['final']['score']:.5f} +- {report['final']['spread']:.5f}",
        f"- wall time: {report['wall_seconds']:.0f} s, {report['model_fits']} model fits",
    ]
    adapter = report["adapter_pass"]
    if adapter:
        lines.append(
            f"- adapter pass ({adapter['adapter']}): tried {len(adapter['candidates'])}, "
            f"kept {len(adapter['added'])}; MLP score {adapter['base_score']:.5f} -> "
            f"{adapter['final_score']:.5f}; set kept: "
            f"{(adapter['set_choice'] or {}).get('chosen', 'pruned')}"
        )
    if report["catalog"]["note"]:
        lines.append(f"- catalog: {report['catalog']['note']}")
    lines += [
        "",
        "| feature | template | removal cost | MLP removal cost |",
        "|---|---|---|---|",
    ]
    for entry in report["final"]["features"]:
        costs = [entry["removal_cost"], entry["mlp_removal_cost"]]
        lines.append(
            f"| `{entry['feature']}` | {entry['template']} | "
            + " | ".join("" if cost is None else f"{cost:+.5f}" for cost in costs)
            + " |"
        )
    return "\n".join(lines) + "\n"


def search_seeds(seed: int) -> list[int]:
    """The validation seeds of a search whose `--seed` is `seed`: that and the next two."""
    return [seed + offset for offset in range(len(DEFAULT_SEEDS))]


def _non_negative_int(text: str) -> int:
    """An `--adapter-pass-budget`: a number of candidates, none included."""
    try:
        count = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if count < 0:
        raise argparse.ArgumentTypeError("the budget must not be negative")
    return count


def add_search_budget_arguments(parser: argparse.ArgumentParser) -> None:
    """The bounds of a search, which `feature-search` and `generate --feature-search` take
    alike."""
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=MAX_CANDIDATES,
        help=f"Refuse a search over more candidates (default: {MAX_CANDIDATES})",
    )
    parser.add_argument(
        "--adapter-pass-budget",
        type=_non_negative_int,
        default=ADAPTER_PASS_BUDGET,
        help="With --adapter mlp_data: the derived features the MLP-scored pass tries, "
        "dealt out across the tier, grid, pad and ratio templates "
        f"(default: {ADAPTER_PASS_BUDGET})",
    )


def add_feature_search_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        required=True,
        help="Training rows: a collected .csv/.json corpus or the published .parquet "
        "(never the evaluation slice)",
    )
    parser.add_argument(
        "--target",
        default="tflops",
        help="The measurement column whose log the search explains (default: tflops)",
    )
    parser.add_argument(
        "--objective",
        choices=("min", "max"),
        default="max",
        help="Whether the target is minimised or maximised; recorded in the report "
        "(default: max)",
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        help="Columns that together identify a graph: rows that agree on all of them stay on "
        "one side of every fold, whatever order they are listed in (default: benchmark, "
        "so a graph measured on several devices stays whole)",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        help="Published columns the search may read (default: the keys of the rows' "
        "`features` column)",
    )
    parser.add_argument(
        "--catalog-values",
        metavar="JSON_FILE",
        help='A JSON file {"field": [values...]} of the values a catalog field takes, '
        "for the tier template",
    )
    parser.add_argument(
        "--descriptor-tree",
        help="Read the catalog values from the metadata of this tree's KDP kernels",
    )
    parser.add_argument(
        "--engine",
        help="With --descriptor-tree: the UED name or id whose kernels count (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The first validation seed: the search runs over this seed and the next two, "
        "and the score is the mean over them and its spread their std; `generate "
        "--feature-search` seeds its search the same way from its own --seed (default: 0)",
    )
    parser.add_argument(
        "--adapter",
        choices=tuple(SEARCH_ADAPTERS),
        default="tree_data",
        help="The model the recipe will train, spelled as `generate --adapter` spells it: "
        "mlp_data adds a forward pass scored by the MLP trainer after the LightGBM "
        "elimination (default: tree_data)",
    )
    add_search_budget_arguments(parser)
    parser.add_argument(
        "--feature-evaluator", help="Shared hipdnn_uhd_features executable"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New directory for recipe.json, the report and its summary",
    )


def _graph_keys(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """One id per distinct combination of `columns`, numbered by first appearance: rows that
    agree on all of them share a graph, whatever order the columns are listed in."""
    columns = list(dict.fromkeys(columns))
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"--group-by columns absent from the rows: {missing}")
    return frame.groupby(columns, sort=False, dropna=False).ngroup()


def run_feature_search(args: argparse.Namespace) -> int:
    from .correctness import measured_rows
    from .corpus_io import read_corpus_frame
    from .evaluate import resolve_grouping

    try:
        output = Path(args.output_dir).resolve()
        if output.exists():
            raise ValueError("--output-dir must not exist")
        if args.catalog_values and args.descriptor_tree:
            raise ValueError(
                "--catalog-values and --descriptor-tree both name the catalog"
            )
        frame, _ = measured_rows(read_corpus_frame(Path(args.input)), args.target)
        frame = frame.reset_index(drop=True)
        published = set(args.columns or ())
        if not published and "features" in frame.columns:
            # A row publishes only the features it binds: the union over the rows.
            published = set().union(*map(json.loads, frame["features"]))
        if not published:
            raise ValueError(
                "the rows name no published columns: pass --columns, or rows with a "
                "`features` column"
            )
        missing = sorted(published - set(frame.columns))
        if missing:
            raise ValueError(f"published columns absent from the rows: {missing}")
        # Folds hold whole graphs. By default a graph is its benchmark, so one measured on
        # several devices is never on both sides of a split; --group-by names the columns
        # that together identify one.
        keys = _graph_keys(frame, args.group_by or [resolve_grouping(frame).columns[0]])
        # No metric: every optional feature is guarded, whichever model reads the recipe.
        columns, identifiers, guards, omitted = search_inputs(
            frame, published, None, [], None, args.feature_evaluator
        )
        if args.catalog_values:
            catalog = json.loads(Path(args.catalog_values).read_text(encoding="utf-8"))
            source = str(Path(args.catalog_values).resolve())
        elif args.descriptor_tree:
            catalog = catalog_values_from_tree(Path(args.descriptor_tree), args.engine)
            source = str(Path(args.descriptor_tree).resolve())
        else:
            catalog, source = {}, None
        report = search_features(
            frame,
            columns=columns,
            presence_columns=identifiers,
            guards=guards,
            target=args.target,
            objective=args.objective,
            groups=keys,
            catalog_values=catalog,
            catalog_source=source,
            seeds=search_seeds(args.seed),
            max_candidates=args.max_candidates,
            omitted=omitted,
            feature_evaluator=args.feature_evaluator,
            adapter=SEARCH_ADAPTERS[args.adapter],
            adapter_pass_budget=args.adapter_pass_budget,
        )
        paths = write_search(report, output)
    except (OSError, TypeError, ValueError, KeyError) as error:
        logger.error("%s", error)
        return 1
    print(f"Searched {len(report['recipe'])} feature(s): {paths['recipe']}")
    return 0

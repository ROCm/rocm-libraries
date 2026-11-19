# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Search a feature recipe: backward elimination over published columns and derived expressions.

The candidates are the published columns every training row binds (after duplicate and
constant columns collapse) and canonical jsonexpr expressions over them -- padding, ratios,
stride order, grid/wave counts, and rounding up to the tiers a catalog offers. LightGBM scores a set by
grouped cross-validation on the training rows (never the evaluation slice); the search drops
the least important features round by round, keeps the smallest set within one standard error
of the best, then removes what it still can one feature at a time. The result is a canonical
features-signature array, usable as `uhd_gen generate --feature-signature`.
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
from .features import derive_categorical_encoding, evaluate_feature_rows

logger = logging.getLogger(__name__)

SCHEMA = "uhd_gen.feature_search/1"
RECIPE = "recipe.json"
REPORT = "feature_search_report.json"
SUMMARY = "feature_search_report.md"

PAD_MULTIPLES = (8, 16, 32, 64, 128)
GRID_TILES = (16, 32, 64, 128, 256)
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


class _Pool:
    """The candidates, deduplicated by value and bounded."""

    def __init__(self, bound: int) -> None:
        self.bound = bound
        self.candidates: list[Candidate] = []
        self.stats: dict[str, collections.Counter] = {}
        #: kept name -> the names with exactly its values, collapsed into it
        self.duplicates: dict[str, list[str]] = {}
        self._index: dict[bytes, int] = {}

    def add(self, template: str, entry, values: np.ndarray) -> str | None:
        """Admit `entry`; its kept name (itself or the equal one before it), or None if it
        is constant or not finite on some row."""
        stats = self.stats.setdefault(template, collections.Counter())
        stats["generated"] += 1
        if not np.isfinite(values).all():
            stats["non_finite"] += 1
            return None
        if values.min() == values.max():
            stats["constant"] += 1
            return None
        name = _canonical(entry)
        digest = hashlib.blake2b(values.tobytes(), digest_size=16).digest()
        index = self._index.get(digest)
        if index is not None:
            kept = self.candidates[index].name
            self.duplicates.setdefault(kept, []).append(name)
            stats["duplicate"] += 1
            return kept
        if len(self.candidates) >= self.bound:
            raise ValueError(
                f"the search has more than {self.bound} candidate features "
                f"({self.counts()}); refusing rather than sampling. Narrow the published "
                "columns with --features/--feature-signature, or raise --max-candidates"
            )
        self._index[digest] = len(self.candidates)
        self.candidates.append(Candidate(name, entry, template, values))
        stats["kept"] += 1
        return name

    def counts(self) -> dict[str, int]:
        return {name: stat["kept"] for name, stat in self.stats.items()}


def _column_values(frame: pd.DataFrame, name: str) -> np.ndarray:
    """A column as the evaluator reads it: numbers as they are, strings as the sorted codes
    `derive_categorical_encoding` assigns."""
    series = frame[name]
    strings = series.map(lambda value: isinstance(value, str))
    if strings.all():
        codes = {value: code for code, value in enumerate(sorted(set(series)))}
        return series.map(codes).to_numpy(dtype=np.float64)
    if strings.any():
        raise ValueError(f"feature column {name!r} mixes strings with numbers")
    return series.to_numpy(dtype=np.float64)


class _Term:
    """An expression and its values on the rows, built together so they cannot disagree."""

    def __init__(self, expr, values: np.ndarray) -> None:
        self.expr, self.values = expr, values

    @classmethod
    def column(cls, name: str, values: np.ndarray) -> "_Term":
        return cls("$" + name, values)

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
    stats: dict,
) -> None:
    """Add the pad, ratio, stride-order, grid and tier templates over the base columns' representatives."""
    by_name = {candidate.name: candidate for candidate in pool.candidates}

    def term(name: str) -> _Term:
        return _Term.column(name, by_name["$" + name].values)

    # The dim columns: `...dims[<k>]` with integer values that vary, by tensor role. A dim
    # that duplicates another tensor's (q and o share B, H, Sq) is read through the
    # lexicographically first column of its group, so a role still reaches every dim.
    roles: dict[str, list[tuple[int, str]]] = {}
    for name in sorted(representative):
        match = _DIMS.fullmatch(name)
        if match:
            values = by_name["$" + representative[name]].values
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
            fill = dim.over(padded)
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
        if name in representative:
            return term(representative[name])
        if frame[name].map(lambda value: isinstance(value, str)).any():
            return None
        return _Term.column(name, _column_values(frame, name))

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
            cu = _Term.column(
                "device.cu_count", frame["device.cu_count"].to_numpy(float)
            )
    for role, dims in sorted(roles.items()):
        members = sorted({name for _, name in dims})
        if len(dims) < 3:
            continue
        for name in members:
            dim = term(name)
            others = [other for other in members if other != name]
            for size in (1, 2):
                for subset in itertools.combinations(others, size):
                    for tile in GRID_TILES:
                        if tile >= dim.values.max():
                            continue
                        grid = _product(
                            [term(other) for other in subset] + [dim.ceil_div(tile)]
                        )
                        pool.add("grid", grid.expr, grid.values)
                        if cu is not None:
                            waves = grid.over(cu)
                            pool.add("grid", waves.expr, waves.values)
    stats["grid_per_cu"] = (
        "added" if cu is not None else "skipped: device.cu_count absent or constant"
    )

    for field, values in sorted(catalog_values.items()):
        ladder = sorted(values)
        for name in all_dims:
            dim = term(name)
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


def build_candidates(
    frame: pd.DataFrame,
    columns: list[str],
    presence_columns: list[str],
    catalog_values: dict[str, list[float]] | None,
    max_candidates: int,
) -> tuple[_Pool, dict]:
    """(pool, details): the base columns and presence flags, then the derived templates."""
    pool = _Pool(max_candidates)
    details: dict = {}
    representative: dict[str, str] = {}
    for name in sorted(columns):
        kept = pool.add("base", "$" + name, _column_values(frame, name))
        if kept is not None:
            representative[name] = kept[1:]
    for name in sorted(presence_columns):
        flag = {"!=": ["$" + name, -1]}
        pool.add("presence", flag, (frame[name].to_numpy(float) != -1).astype(float))
    details["constant_columns"] = sorted(
        name for name in columns if name not in representative
    )
    _derived_candidates(
        pool, frame, columns, representative, catalog_values or {}, details
    )
    return pool, details


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
        self.matrix, self.target = matrix, target
        self.seeds, self.folds = list(seeds), folds
        self.fold_of = [_folds(groups, seed, folds) for seed in self.seeds]
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
) -> dict:
    """Search a recipe over the training rows `frame`; the report, whose `recipe` is the
    features signature.

    `columns` are the published columns the search may read (already past the exclusion
    rules), `presence_columns` the tensor identifiers it may read only as presence flags,
    `groups` the problem each row belongs to. The target's log is the regression label for
    either objective: the search wants the features that explain it, not a ranking."""
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
    pool, details = build_candidates(
        frame, list(columns), list(presence_columns), tiers, max_candidates
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

    def tolerance(spread: float) -> float:
        return max(spread, MIN_TOLERANCE)

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
        worse = worse + 1 if score > best["score"] + tolerance(best["spread"]) else 0
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

    # The smallest set within one standard error of the best round; a spread under the
    # noise floor (three seeds of one fold layout) does not narrow it.
    chosen = min(
        (r for r in rounds if r["score"] <= best["score"] + tolerance(best["spread"])),
        key=lambda r: (r["features"], r["score"]),
    )
    survivors = sorted(c for index in chosen["active"] for c in clusters[index])
    rank = {c: chosen["importance"][cluster_of[c]] for c in survivors}
    current = sorted(survivors, key=lambda c: (rank[c], names[c]))
    anchor, allowed = chosen["score"], tolerance(chosen["spread"])
    greedy = []
    for column in list(current):
        if len(current) == 1:
            break
        trial = [c for c in current if c != column]
        score, _, _ = scorer.score(sorted(trial))
        accepted = score <= anchor + allowed
        greedy.append({"feature": names[column], "score": score, "removed": accepted})
        logger.info(
            "feature search final pass: %s %s (score %.5f, %d left)",
            "removed" if accepted else "kept",
            names[column],
            score,
            len(current) - accepted,
        )
        if accepted:
            current = trial
            dropped.append(
                {
                    "feature": names[column],
                    "reason": f"final pass: score {score:.5f} within {allowed:.5f} of "
                    f"{anchor:.5f}",
                }
            )
    final = sorted(current)
    final_score, final_spread, _ = scorer.score(final)
    costs = {}
    for column in final:
        without = [c for c in final if c != column]
        costs[column] = scorer.score(without)[0] - final_score if without else None
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
        "exclusions": list(omitted),
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
            "duplicate_groups": [
                {"kept": kept, "collapsed": sorted(collapsed)}
                for kept, collapsed in sorted(pool.duplicates.items())
            ],
            "grid_per_cu": details["grid_per_cu"],
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
        "final_pass": greedy,
        "final": {
            "score": final_score,
            "spread": final_spread,
            "features": [
                {
                    "feature": names[c],
                    "template": candidates[c].template,
                    "removal_cost": costs[c],
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
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
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
    if report["catalog"]["note"]:
        lines.append(f"- catalog: {report['catalog']['note']}")
    lines += ["", "| feature | template | removal cost |", "|---|---|---|"]
    for entry in report["final"]["features"]:
        cost = entry["removal_cost"]
        lines.append(
            f"| `{entry['feature']}` | {entry['template']} | "
            f"{'' if cost is None else f'{cost:+.5f}'} |"
        )
    return "\n".join(lines) + "\n"


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
        help="Columns identifying a problem (default: benchmark and device)",
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
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Validation seeds; the score is the mean over them and its spread their "
        f"std (default: {' '.join(map(str, DEFAULT_SEEDS))})",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=MAX_CANDIDATES,
        help=f"Refuse a search over more candidates (default: {MAX_CANDIDATES})",
    )
    parser.add_argument(
        "--feature-evaluator", help="Shared hipdnn_uhd_features executable"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New directory for recipe.json, the report and its summary",
    )


def run_feature_search(args: argparse.Namespace) -> int:
    from .correctness import measured_rows
    from .corpus_io import read_corpus_frame
    from .evaluate import Grouping, problem_keys, resolve_grouping
    from .generate import proposal_columns

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
            published = set(json.loads(frame["features"].iloc[0]))
        if not published:
            raise ValueError(
                "the rows name no published columns: pass --columns, or rows with a "
                "`features` column"
            )
        missing = sorted(published - set(frame.columns))
        if missing:
            raise ValueError(f"published columns absent from the rows: {missing}")
        grouping = (
            Grouping(tuple(args.group_by), False, "--group-by")
            if args.group_by
            else resolve_grouping(frame)
        )
        keys = problem_keys(frame, grouping)
        scalars, identifiers, _, omitted = proposal_columns(frame, published, None, [])
        columns = [name for name in scalars if not name.startswith("kernel.")]
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
            target=args.target,
            objective=args.objective,
            groups=keys,
            catalog_values=catalog,
            catalog_source=source,
            seeds=args.seed,
            max_candidates=args.max_candidates,
            omitted=omitted,
            feature_evaluator=args.feature_evaluator,
        )
        paths = write_search(report, output)
    except (OSError, TypeError, ValueError, KeyError) as error:
        logger.error("%s", error)
        return 1
    print(f"Searched {len(report['recipe'])} feature(s): {paths['recipe']}")
    return 0

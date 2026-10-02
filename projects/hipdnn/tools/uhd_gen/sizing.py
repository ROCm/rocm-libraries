# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`uhd_gen size`: how many more unique shapes a model needs, and from which regimes.

Trains on nested subsets of the pool, scores each on a test set pinned across rounds,
fits miss(n) = floor + a * n^-b with floor = 1 - the ceiling measured from repeats, and
splits new shapes over regimes as `--regime-quotas` for `hipdnn_corpus_gen`.
Engine-immediate (L1) models only: a catalog ranker's accuracy is regret, not row error.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import shutil
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .generate import (
    COLLECTION_MANIFEST,
    _corpus_name,
    _measurement_key,
    _write_json,
    load_collections,
    read_regime_manifest,
    write_collection,
)
from .immediate import ROLE
from .ranking_metrics import DEFAULT_RANKING_METRIC, ranking_metric

logger = logging.getLogger(__name__)

SCHEMA = "uhd_gen.sizing/1"
TEST_SET_SCHEMA = "uhd_gen.sizing_test_set/1"
#: A regime this thin cannot be stratified or scored; RFC 0019.13's problem floor.
REGIME_FLOOR = 30
#: Pseudo-count of test shapes shrinking a regime's miss rate toward the global rate.
SHRINKAGE = 10
BOOTSTRAP = 200


def add_size_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--collection",
        nargs="+",
        required=True,
        metavar="DIR",
        help="Collections to size from (every measurement in them is used)",
    )
    parser.add_argument(
        "--corpus-manifest",
        nargs="*",
        default=[],
        metavar="CSV",
        help="hipdnn_corpus_gen manifest.csv files: regime labels for rows "
        "collected before rows carried them, and regimes nothing measured yet",
    )
    parser.add_argument(
        "--test-set",
        required=True,
        help="The pinned test set (JSON). Reused every round; see --create-test-set",
    )
    parser.add_argument(
        "--create-test-set",
        action="store_true",
        help="Create --test-set if it does not exist (stratified by regime); an "
        "existing one is always reused, never replaced",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--target",
        type=float,
        help="Wanted fraction of test predictions within --tolerance (e.g. 0.70)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.10,
        help="Relative error that counts as a hit (default 0.10: within 10%%)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        help="Training-pool sizes to fit at (default: 1/16 .. all of the pool)",
    )
    parser.add_argument("--draws", type=int, default=3, help="Random subsets per size")
    parser.add_argument(
        "--floor",
        type=int,
        default=REGIME_FLOOR,
        help="Training shapes every regime is brought to before the rest is split",
    )
    parser.add_argument(
        "--previous", help="The last round's sizing_report.json, to check"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    # Passed through to the `generate --collection` run at every curve point.
    parser.add_argument("--descriptor-tree", required=True)
    parser.add_argument("--engine")
    parser.add_argument("--engine-id", type=int)
    parser.add_argument("--arch")
    parser.add_argument("--uhd-id")
    parser.add_argument("--metric", default=DEFAULT_RANKING_METRIC)
    parser.add_argument("--features", nargs="+")
    parser.add_argument("--feature-evaluator")
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.1,
        help="generate's internal holdout per curve point; the curve's x is the "
        "shapes actually fitted, so this is accounted for",
    )
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=50)


# ---------------------------------------------------------------------------------------------
# Pure pieces: no training, no files.
# ---------------------------------------------------------------------------------------------


def ceiling_from_repeats(
    labels: dict, measurements: list, label_column: str, tolerance: float, shapes: set
) -> dict:
    """How often a perfect model could land within `tolerance` of a label, from repeats.

    A repeat is the same key from another session (collection x device). Disagreements
    are divided by sqrt(2): a repeat carries two independent measurement errors.
    `within` is None when nothing was measured twice; unknown is not assumed perfect.
    """
    errors = []
    for session, row in measurements:
        key = _measurement_key(row)
        if key not in labels or key[0] not in shapes:
            continue
        label_session, label = labels[key]
        if session == label_session:
            continue
        measured, repeated = label.get(label_column), row.get(label_column)
        if _positive(measured) and _positive(repeated):
            errors.append(abs(repeated / measured - 1.0))
    if not errors:
        return {
            "repeats": 0,
            "within": None,
            "median_abs_pct": None,
            "repeat_within": None,
        }
    single = [e / math.sqrt(2.0) for e in errors]
    return {
        "repeats": len(errors),
        "within": sum(e <= tolerance for e in single) / len(single),
        "median_abs_pct": 100.0 * statistics.median(single),
        # What a second measurement achieves against the first, for comparison.
        "repeat_within": sum(e <= tolerance for e in errors) / len(errors),
    }


def _positive(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def fit_curve(points: list, floor: float | None) -> dict | None:
    """miss(n) = floor + a * n^-b through (n, miss) points, by least squares in log-log.

    A None `floor` is searched over [0, min miss). Returns None with fewer than three
    points clear of the floor.
    """

    def solve(fixed: float):
        usable = [(n, m) for n, m in points if n > 0 and m - fixed > 1e-9]
        if len(usable) < 3:
            return None
        xs = [math.log(n) for n, _ in usable]
        ys = [math.log(m - fixed) for _, m in usable]
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx <= 0:
            return None
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        intercept = my - slope * mx
        residual = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
        return {
            "a": math.exp(intercept),
            "b": -slope,
            "floor": fixed,
            "sse": residual,
            "points": len(usable),
        }

    if floor is not None:
        fit = solve(floor)
        if fit:
            fit["floor_source"] = "measured"
        return fit
    lowest = min(m for _, m in points)
    best = None
    for step in range(0, 96):
        candidate = solve(lowest * step / 100.0)
        if candidate and (best is None or candidate["sse"] < best["sse"]):
            best = candidate
    if best:
        best["floor_source"] = "estimated"
    return best


def miss_at(fit: dict, n: float) -> float:
    return fit["floor"] + fit["a"] * n ** (-fit["b"])


def size_for(fit: dict, target_within: float) -> float | None:
    """Fitted shapes for `target_within`, or None when the target is at or past the ceiling."""
    wanted_miss = 1.0 - target_within
    if wanted_miss <= fit["floor"] or fit["b"] <= 0:
        return None
    return (fit["a"] / (wanted_miss - fit["floor"])) ** (1.0 / fit["b"])


def allocate(new_shapes: int, regimes: dict, floor: int) -> dict:
    """Split `new_shapes` over regimes: `floor` each first, the rest by test misses.

    The rest follows test count x miss rate (shrunk toward the global rate), so untested
    regimes get only their floor. Largest-remainder rounding makes quotas sum exactly.
    """
    deficits = {r: max(0, floor - info["train"]) for r, info in regimes.items()}
    rest = max(0, new_shapes - sum(deficits.values()))
    tested = sum(info["test"] for info in regimes.values())
    global_miss = (
        (sum(info["misses"] for info in regimes.values()) / tested) if tested else 0.0
    )
    weights = {}
    for regime, info in regimes.items():
        shrunk = (info["misses"] + SHRINKAGE * global_miss) / (info["test"] + SHRINKAGE)
        weights[regime] = info["test"] * shrunk
    total = sum(weights.values())
    quotas = dict(deficits)
    if rest and total > 0:
        exact = {r: rest * w / total for r, w in weights.items()}
        whole = {r: math.floor(v) for r, v in exact.items()}
        left = rest - sum(whole.values())
        for regime in sorted(exact, key=lambda r: (whole[r] - exact[r], r))[:left]:
            whole[regime] += 1
        for regime, extra in whole.items():
            quotas[regime] += extra
    return {r: q for r, q in quotas.items() if q > 0}


def stratified_test_set(shapes_by_regime: dict, fraction: float, seed: int) -> list:
    """`fraction` of every regime's shapes, at least one where a regime has two or more."""
    rng = random.Random(seed)
    chosen = []
    for regime in sorted(shapes_by_regime):
        shapes = sorted(shapes_by_regime[regime])
        rng.shuffle(shapes)
        take = round(len(shapes) * fraction)
        if take == 0 and len(shapes) >= 2:
            take = 1
        chosen.extend(shapes[:take])
    return sorted(chosen)


def default_sizes(pool: int) -> list:
    sizes = {max(1, round(pool * f)) for f in (1 / 16, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 1)}
    return sorted(s for s in sizes if s >= min(20, pool))


# ---------------------------------------------------------------------------------------------
# The run.
# ---------------------------------------------------------------------------------------------


def _load_everything(paths: list, metric: str) -> tuple[dict, list, dict]:
    """(labels, every measurement, the merge) for `metric`'s corpora in `paths`.

    Labels follow training's newest-session-wins rule but keep the session, so the
    ceiling can tell a label from its repeats; `load_collections` validates the merge.
    """
    merged = load_collections(paths, role=ROLE, sources=[metric])
    loaded = []
    for supplied in paths:
        directory = Path(supplied).resolve()
        manifest = json.loads(
            (directory / COLLECTION_MANIFEST).read_text(encoding="utf-8")
        )
        loaded.append((manifest["collected_at"], str(directory), manifest))
    loaded.sort(key=lambda item: (item[0], item[1]))
    measurements = []
    for order, (_, directory, manifest) in enumerate(loaded):
        corpus = Path(directory) / f"{_corpus_name(manifest['sources'], metric)}.json"
        measurements.extend(
            ((order, str(row["device"])), row)
            for row in json.loads(corpus.read_text(encoding="utf-8"))
        )
    labels = {}
    for session, row in measurements:
        labels[_measurement_key(row)] = (session, row)
    if len(labels) != len(merged["rows"][metric]):
        raise ValueError(
            "sizing and training disagree on which measurement is each shape's label"
        )
    return labels, measurements, merged


def _write_subset(
    template: Path, merged: dict, metric: str, rows: list, destination: Path
) -> Path:
    """A collection holding only `rows`, written as a measuring run would have written it."""
    manifest = json.loads((template / COLLECTION_MANIFEST).read_text(encoding="utf-8"))
    destination.mkdir(parents=True)
    write_collection(
        destination,
        collected_at=manifest["collected_at"],
        role=ROLE,
        engine=manifest.get("engine"),
        engine_id=merged["engine_id"],
        sources=[metric],
        rows={metric: rows},
        published=merged["published"],
        commands=[],
        graph_inputs=[],
        provenance=merged["provenance"],
        knob_encodings=merged["knob_encodings"],
        shipping_knobs=merged["shipping_knobs"],
        collection_knobs=merged["collection_knobs"],
        kernel_fields=merged["kernel_fields"],
    )
    return destination


def _score(report_path: Path, tolerance: float) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    errors = {}
    for row in report.get("per_row") or []:
        value = row.get("signed_relative_error")
        if isinstance(value, (int, float)) and math.isfinite(value):
            errors[row["key"][0]] = abs(value)
    if not errors:
        raise ValueError(f"{report_path} scored no test row")
    values = list(errors.values())
    return {
        "rows": len(values),
        "within": sum(e <= tolerance for e in values) / len(values),
        "median_abs_pct": 100.0 * statistics.median(values),
        "errors": errors,
    }


def run_size(args: argparse.Namespace) -> int:
    from .__main__ import main

    try:
        output = Path(args.output_dir).resolve()
        if output.exists():
            raise ValueError("--output-dir must not exist")
        if not 0 < args.tolerance < 1 or not 0 < args.test_fraction < 1:
            raise ValueError("--tolerance and --test-fraction must be in (0, 1)")
        if args.target is not None and not 0 < args.target < 1:
            raise ValueError("--target is a fraction in (0, 1)")
        label_column = ranking_metric(args.metric).label
        labels, measurements, merged = _load_everything(args.collection, args.metric)

        # Row regimes win; manifests fill older rows and add regimes nothing measured.
        regime_of, operation_of = {}, {}
        for manifest in args.corpus_manifest:
            for benchmark, columns in read_regime_manifest(Path(manifest)).items():
                regime_of.setdefault(benchmark, columns["regime"])
                if columns.get("regime.operation"):
                    operation_of.setdefault(
                        columns["regime"], columns["regime.operation"]
                    )
        for _, row in labels.values():
            if row.get("regime"):
                regime_of[str(row["benchmark"])] = row["regime"]
                if row.get("regime.operation"):
                    operation_of.setdefault(row["regime"], row["regime.operation"])
        usable = {
            key: row
            for key, (_, row) in labels.items()
            if _positive(row.get(label_column))
        }
        shapes = sorted({key[0] for key in usable})
        shapes_by_regime: dict = {}
        for shape in shapes:
            shapes_by_regime.setdefault(regime_of.get(shape, "unlabelled"), []).append(
                shape
            )

        test_path = Path(args.test_set)
        if test_path.is_file():
            test_set = json.loads(test_path.read_text(encoding="utf-8"))
            if test_set.get("schema") != TEST_SET_SCHEMA:
                raise ValueError(f"{test_path} is not a {TEST_SET_SCHEMA} test set")
        elif args.create_test_set:
            test_set = {
                "schema": TEST_SET_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "fraction": args.test_fraction,
                "seed": args.seed,
                "shapes": stratified_test_set(
                    shapes_by_regime, args.test_fraction, args.seed
                ),
            }
            test_path.parent.mkdir(parents=True, exist_ok=True)
            _write_json(test_path, test_set)
        else:
            raise ValueError(
                f"{test_path} does not exist; pass --create-test-set to pin one"
            )
        pinned = set(test_set["shapes"])
        test_keys = [key for key in usable if key[0] in pinned]
        pool = sorted(key for key in usable if key[0] not in pinned)
        if len(test_keys) < 10 or len(pool) < 20:
            raise ValueError(
                f"too little to size from: {len(test_keys)} test and {len(pool)} "
                f"training shapes"
            )

        output.mkdir(parents=True)
        work = Path(tempfile.mkdtemp(prefix=".uhd-size-", dir=output.parent))
        try:
            template = max(
                (Path(p).resolve() for p in args.collection),
                key=lambda p: json.loads((p / COLLECTION_MANIFEST).read_text())[
                    "collected_at"
                ],
            )
            test_corpus = work / "test.json"
            _write_json(test_corpus, [usable[key] for key in test_keys])
            sizes = sorted(set(args.sizes)) if args.sizes else default_sizes(len(pool))
            sizes = [s for s in sizes if 0 < s <= len(pool)]
            curve, full_errors = [], None
            for draw in range(args.draws):
                order = pool[:]
                random.Random(args.seed * 1000 + draw).shuffle(order)
                for size in sizes:
                    if size == len(pool) and draw > 0:
                        continue  # the whole pool is one subset, whatever the draw
                    tag = f"n{size}_d{draw}"
                    subset = _write_subset(
                        template,
                        merged,
                        args.metric,
                        [usable[key] for key in order[:size]],
                        work / "collections" / tag,
                    )
                    model = work / "models" / tag
                    argv = [
                        "generate",
                        "--collection",
                        str(subset),
                        "--descriptor-tree",
                        str(args.descriptor_tree),
                        "--role",
                        ROLE,
                        "--metric",
                        args.metric,
                        "--eval-fraction",
                        str(args.eval_fraction),
                        "--num-boost-round",
                        str(args.num_boost_round),
                        "--early-stopping",
                        str(args.early_stopping),
                        "--seed",
                        str(args.seed),
                        "--no-promote",
                        "--output-dir",
                        str(model),
                    ]
                    for flag, value in (
                        ("--engine", args.engine),
                        ("--engine-id", args.engine_id),
                        ("--arch", args.arch),
                        ("--uhd-id", args.uhd_id),
                        ("--feature-evaluator", args.feature_evaluator),
                    ):
                        if value is not None:
                            argv += [flag, str(value)]
                    if args.features:
                        argv += ["--features", *args.features]
                    if main(argv) != 0:
                        raise ValueError(f"training the {tag} curve point failed")
                    report = model / "test_report.json"
                    evaluate = [
                        "evaluate",
                        "--input",
                        str(test_corpus),
                        "--model-dir",
                        str(model / "model"),
                        "--eval-fraction",
                        "1.0",
                        "--include-per-problem",
                        "--output",
                        str(report),
                    ]
                    if args.feature_evaluator:
                        evaluate += ["--feature-evaluator", args.feature_evaluator]
                    if main(evaluate) != 0:
                        raise ValueError(f"scoring the {tag} curve point failed")
                    scored = _score(report, args.tolerance)
                    fitted = size * (1.0 - args.eval_fraction)
                    curve.append(
                        {
                            "shapes": size,
                            "fitted_shapes": fitted,
                            "draw": draw,
                            "test_rows": scored["rows"],
                            "within": scored["within"],
                            "median_abs_pct": scored["median_abs_pct"],
                        }
                    )
                    logger.info(
                        "curve %s: within %.1f%% of %.0f%% on %d test shapes",
                        tag,
                        100 * scored["within"],
                        100 * args.tolerance,
                        scored["rows"],
                    )
                    if size == len(pool):
                        full_errors = scored["errors"]
        finally:
            shutil.rmtree(work, ignore_errors=True)

        # The fit, on each size's mean, with an interval from resampling the draws.
        by_size: dict = {}
        for point in curve:
            by_size.setdefault(point["fitted_shapes"], []).append(1.0 - point["within"])
        means = [(n, statistics.fmean(m)) for n, m in sorted(by_size.items())]
        ceiling = ceiling_from_repeats(
            labels,
            measurements,
            label_column,
            args.tolerance,
            {key[0] for key in test_keys},
        )
        floor = None if ceiling["within"] is None else 1.0 - ceiling["within"]
        if floor is not None and floor >= min(m for _, m in means):
            # Curve already beats the measured ceiling: the fit estimates its own floor.
            ceiling["binding"] = False
            ceiling["note"] = (
                "the curve already exceeds the measured ceiling; the fit estimates "
                "its own floor"
            )
            floor = None
        elif floor is not None:
            ceiling["binding"] = True
        fit = fit_curve(means, floor)
        rng = random.Random(args.seed)
        resampled = []
        for _ in range(BOOTSTRAP):
            sample = [
                (n, statistics.fmean(rng.choice(m) for _ in m))
                for n, m in sorted(by_size.items())
            ]
            refit = fit_curve(sample, floor)
            if refit:
                resampled.append(refit)

        now_fitted = len(pool) * (1.0 - args.eval_fraction)
        observed = next(1.0 - m for n, m in reversed(means))
        recommendation: dict = {
            "tolerance": args.tolerance,
            "current_training_shapes": len(pool),
            "observed_within": observed,
            "target": args.target,
        }
        prediction = None
        if fit:
            doubled = 1.0 - miss_at(fit, 2 * now_fitted)
            fitted_now = 1.0 - miss_at(fit, now_fitted)
            recommendation["fitted_within"] = fitted_now
            recommendation["next_doubling"] = {
                "training_shapes": 2 * len(pool),
                "predicted_within": doubled,
                "gain": doubled - fitted_now,
            }
            if args.target is not None:
                needed = size_for(fit, args.target)
                if needed is None:
                    recommendation["reachable"] = False
                    recommendation["reason"] = (
                        f"the target {args.target:.2f} is at or above the ceiling "
                        f"{1 - fit['floor']:.2f} ({fit['floor_source']}): more shapes cannot reach "
                        f"it; better measurement (repeatability) can"
                    )
                else:
                    interval = sorted(
                        n for n in (size_for(r, args.target) for r in resampled) if n
                    )
                    to_pool = lambda n: math.ceil(
                        n / (1.0 - args.eval_fraction)
                    )  # noqa: E731
                    recommendation.update(
                        {
                            "reachable": True,
                            "training_shapes": to_pool(needed),
                            "new_shapes": max(0, to_pool(needed) - len(pool)),
                            "new_shapes_interval": (
                                [
                                    max(
                                        0,
                                        to_pool(
                                            interval[int(0.1 * (len(interval) - 1))]
                                        )
                                        - len(pool),
                                    ),
                                    max(
                                        0,
                                        to_pool(
                                            interval[int(0.9 * (len(interval) - 1))]
                                        )
                                        - len(pool),
                                    ),
                                ]
                                if interval
                                else None
                            ),
                        }
                    )
                    prediction = {
                        "training_shapes": to_pool(needed),
                        "within": args.target,
                    }
            if prediction is None:
                # No reachable target: recommend the next doubling (the basis).
                recommendation["basis"] = "next_doubling"
                recommendation["new_shapes"] = len(pool)
                prediction = {"training_shapes": 2 * len(pool), "within": doubled}
            else:
                recommendation["basis"] = "target"

        # Per regime, from the whole-pool model.
        regimes: dict = {}
        for regime in set(regime_of.values()) | set(shapes_by_regime):
            regimes[regime] = {"train": 0, "test": 0, "misses": 0, "errors": []}
        for key in pool:
            regimes[regime_of.get(key[0], "unlabelled")]["train"] += 1
        for benchmark, error in (full_errors or {}).items():
            info = regimes[regime_of.get(benchmark, "unlabelled")]
            info["test"] += 1
            info["misses"] += error > args.tolerance
            info["errors"].append(error)
        table = {
            regime: {
                "train": info["train"],
                "test": info["test"],
                "within": (1 - info["misses"] / info["test"]) if info["test"] else None,
                "median_abs_pct": (
                    100 * statistics.median(info["errors"]) if info["errors"] else None
                ),
            }
            for regime, info in sorted(regimes.items())
        }

        new_shapes = recommendation.get("new_shapes") or 0
        labelled = {r: v for r, v in regimes.items() if r != "unlabelled"}
        quotas = allocate(new_shapes, labelled, args.floor) if labelled else {}
        # Inflate quotas by the usable-row rate, since untimed shapes are lost.
        yielded = sum(
            _positive(row.get(label_column)) for _, row in labels.values()
        ) / max(1, len(labels))
        corpus_quotas: dict = {}
        for regime, quota in quotas.items():
            operation = operation_of.get(regime, "unknown")
            corpus_quotas.setdefault(operation, {})[regime] = math.ceil(
                quota / max(yielded, 1e-6)
            )
        recommendation["regime_quotas"] = quotas
        recommendation["usable_row_rate"] = yielded

        checked = None
        if args.previous:
            previous = json.loads(Path(args.previous).read_text(encoding="utf-8"))
            if previous.get("test_set", {}).get("shapes_digest") != _digest(
                test_set["shapes"]
            ):
                checked = {
                    "comparable": False,
                    "reason": "the previous round used a different test set",
                }
            else:
                old = previous.get("fit")
                predicted = (1.0 - miss_at(old, now_fitted)) if old else None
                checked = {
                    "comparable": True,
                    "previous_prediction": previous.get("prediction"),
                    "predicted_within_at_this_size": predicted,
                    "observed_within": observed,
                    "error": None if predicted is None else observed - predicted,
                }

        report = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metric": args.metric,
            "label": label_column,
            "tolerance": args.tolerance,
            "collections": [str(Path(p).resolve()) for p in args.collection],
            "test_set": {
                "path": str(test_path.resolve()),
                "shapes": len(test_set["shapes"]),
                "scored": len(test_keys),
                "shapes_digest": _digest(test_set["shapes"]),
            },
            "training_pool": len(pool),
            "sizes": sizes,
            "draws": args.draws,
            "eval_fraction": args.eval_fraction,
            "curve": curve,
            "ceiling": ceiling,
            "fit": fit,
            "fit_resamples": len(resampled),
            "recommendation": recommendation,
            "prediction": prediction,
            "per_regime": table,
            "previous": checked,
        }
        _write_json(output / "sizing_report.json", report)
        _write_json(output / "regime_quotas.json", corpus_quotas)
        with (output / "curve.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)
        _print(report, output)
        return 0
    except (OSError, ValueError, KeyError) as error:
        logger.error("%s", error)
        return 1


def _digest(shapes: list) -> str:
    import hashlib

    return hashlib.sha256("\n".join(sorted(shapes)).encode("utf-8")).hexdigest()[:16]


def _print(report: dict, output: Path) -> None:
    rec, ceiling, fit = report["recommendation"], report["ceiling"], report["fit"]
    tol = round(100 * report["tolerance"])
    print(f"Sizing report: {output / 'sizing_report.json'}")
    print(
        f"  now: {rec['current_training_shapes']} training shapes, "
        f"{100 * rec['observed_within']:.1f}% within {tol}%"
    )
    print(
        "  ceiling: "
        + (
            "unknown (no shape measured twice)"
            if ceiling["within"] is None
            else f"{100 * ceiling['within']:.1f}% ({ceiling['repeats']} repeats"
            + ("" if ceiling.get("binding", True) else "; not binding")
            + ")"
        )
    )
    if fit is None:
        print("  fit: unavailable (fewer than three curve points clear of the floor)")
        return
    doubling = rec["next_doubling"]
    print(
        f"  fit: {100 * rec['fitted_within']:.1f}% at this size; doubling to "
        f"{doubling['training_shapes']}: {100 * doubling['predicted_within']:.1f}% "
        f"({100 * doubling['gain']:+.1f} points)"
    )
    if rec.get("target") is not None:
        if rec.get("reachable"):
            print(
                f"  for {100 * rec['target']:.0f}%: {rec['new_shapes']} new shapes "
                f"(interval {rec['new_shapes_interval']})"
            )
        else:
            print(f"  for {100 * rec['target']:.0f}%: {rec['reason']}")
    if rec["regime_quotas"]:
        print(f"  quotas: {output / 'regime_quotas.json'}")

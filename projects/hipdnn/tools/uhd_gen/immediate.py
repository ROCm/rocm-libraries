# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Engine-immediate measurements shared by collection, training and evaluation."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

from .correctness import (
    REASON,
    VERDICT,
    known_wrong,
    numerical_reason,
    numerical_verdict,
)
from .features import evaluate_feature_maps, signature_references
from .corpus_io import read_corpus_frame
from .provenance import compare_provenance, validate_provenance
from .ranking_metrics import (
    RANKING_METRICS,
    RankingMetric,
    is_valid_metric_value,
    ranking_metric,
)
from .score_transform import INVERTIBLE as INVERTIBLE_TRANSFORMS

ROLE = "predict_engine"
_ARCH = re.compile(r"^gfx[a-z0-9_-]+$")
_LEAKED_FIELDS = frozenset(
    {
        "kernel",
        "kernel_features",
        "candidate",
        "candidate_id",
        "candidates",
        "results",
        "knob",
        "knobs",
        "knob_settings",
        "configuration",
        "engine_config",
        "prediction",
        *(f"predicted_{name}" for name in RANKING_METRICS),
        "tflops",
        "timing",
        "latency",
        "robustMeanMs",
        "robust_time_ms",
        "minTimeMs",
        "avgTimeMs",
        "succeeded",
        "is_valid",
    }
)

#: RFC 0019.13 §11.2: a `calibrated: true` score MUST train on `avgTimeMs`, and L1 always
#: declares calibrated (`validate_model`), so the label is the arithmetic mean.
LABEL_STATISTIC = "avgTimeMs"

#: Envelope-only exemption from `_LEAKED_FIELDS`; `validate_signature` still rejects these
#: names in features, so no L1 feature can read its own label.
_LABEL_FIELDS = frozenset({"tflops", "is_valid", "robustMeanMs", LABEL_STATISTIC})


def _object(value, where: str) -> dict:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a JSON object")
    return value


def _text(value, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{where} must be a nonempty canonical string")
    return value


def _positive(value, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{where} must be a positive finite number")
    return float(value)


def _missing(value) -> bool:
    """Never written, JSON null, or the NaN pandas fills in for either."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _optional_spread(value, where: str) -> float | None:
    """§8.3's `stddevMs`: nonnegative when the row carries a measurement, else absent."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{where} must be a nonnegative finite number when present")
    return float(value)


def _optional_count(value, where: str) -> int | None:
    """§8.3's `iters`: a whole iteration count when the row carries a measurement."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or float(value) != int(value)
    ):
        raise ValueError(f"{where} must be a nonnegative whole number when present")
    return int(value)


def validate_binding(value) -> dict:
    binding = dict(_object(value, "binding"))
    for name in ("engine", "selector_revision", "arch"):
        _text(binding.get(name), f"binding.{name}")
    if binding.get("role") != ROLE:
        raise ValueError(f"binding.role must be {ROLE}")
    if binding["arch"] != "default" and not _ARCH.fullmatch(binding["arch"]):
        raise ValueError("binding.arch must be a bare gfx architecture or default")
    # Which label a row may train or be scored against depends on the metric it was
    # collected under, whatever columns it carries.
    if binding.get("metric") not in RANKING_METRICS:
        raise ValueError(
            f"binding.metric must name a registered ranking metric "
            f"({', '.join(RANKING_METRICS)}); got {binding.get('metric')!r}"
        )
    binding["trained_against"] = validate_provenance(binding.get("trained_against"))
    return binding


def binding_identity(binding: dict, *, include_uhd_id: bool = True) -> dict:
    """`binding` without `provider_build`: the loaded plugin's name and version, which the
    runtime reports as a diagnostic and never as a model compatibility key.

    Two measurements whose identities match were taken under one contract, whichever build
    answered. `include_uhd_id=False` also drops the model id an opaque engine declares.
    """
    ignored = ("provider_build",) if include_uhd_id else ("provider_build", "uhd_id")
    return {key: value for key, value in binding.items() if key not in ignored}


def require_binding_metric(
    binding: dict, metric: str, where: str, selector_revision: str | None = None
) -> None:
    """Refuse a binding whose metric (or selector revision) differs from the model's.

    A binding that names no metric cannot be checked and is refused.
    """
    recorded = binding.get("metric")
    if recorded is None:
        raise ValueError(
            f"{where}: collection binding records no metric; the model predicts {metric!r}"
        )
    if recorded != metric:
        raise ValueError(
            f"{where}: rows were collected for metric {recorded!r}, "
            f"but the model predicts {metric!r}"
        )
    if (
        selector_revision is not None
        and binding.get("selector_revision") != selector_revision
    ):
        raise ValueError(
            f"{where}: rows were collected under selector revision "
            f"{binding.get('selector_revision')!r}, but the model was trained "
            f"against {selector_revision!r}"
        )


def validate_signature(signature: list) -> None:
    """Reject references to measured outputs; only graph/device/constraint inputs.

    Whether entries evaluate on published features is `check_signature_evaluates`'s job.
    """
    for reference in signature_references(signature):
        name = reference.removeprefix("$")
        parts = set(re.split(r"[.\[\]]+", name))
        if "." not in name or parts & _LEAKED_FIELDS:
            raise ValueError(
                f"L1 features cannot depend on kernel/candidate/timing inputs: {reference}"
            )


def check_signature_evaluates(
    signature: list,
    published,
    categorical_encoding: dict | None = None,
    executable: str | Path | None = None,
) -> None:
    """Every published feature map yields a full row through the runtime's FeatureExtractor.

    The evaluator decides, not a name-membership test: `value_or_default`/`present` may
    reference names a graph does not publish.
    """
    distinct = {
        json.dumps(_object(value, "features"), sort_keys=True) for value in published
    }
    try:
        evaluate_feature_maps(
            [json.loads(value) for value in sorted(distinct)],
            signature,
            categorical_encoding,
            executable,
        )
    except ValueError as error:
        raise ValueError(
            f"the L1 features_signature does not evaluate on every published "
            f"feature map: {error}"
        ) from error


def normalize_row(value: dict) -> dict:
    """Import exactly one no-search execution; derive, rather than trust, TFLOPS."""
    value = _object(value, "immediate measurement")
    forbidden = {
        name
        for name in value
        if name in _LEAKED_FIELDS - _LABEL_FIELDS
        or name.startswith(("kernel.", "candidate.", "knob."))
    }
    if forbidden:
        raise ValueError(
            f"immediate measurements contain candidate/search data: {sorted(forbidden)}"
        )
    if value.get("selection_mode") != "immediate":
        raise ValueError("L1 labels require selection_mode=immediate")
    # `hipdnn_bench` declares `robustMeanMs`; an already-normalized row declares the label
    # statistic. Both read back, and the label is `avgTimeMs` either way.
    if value.get("timing_statistic") not in ("robustMeanMs", LABEL_STATISTIC):
        raise ValueError(
            f"L1 labels require timing_statistic robustMeanMs or {LABEL_STATISTIC}"
        )
    if value.get("is_valid") is not True:
        raise ValueError("L1 labels require is_valid=true")
    binding = validate_binding(value.get("binding"))
    # A row metric that disagrees with its binding means the row is not what it claims.
    if not _missing(value.get("metric")):
        require_binding_metric(binding, value["metric"], "immediate measurement")
    engine = value.get("engine_id", value.get("engine"))
    if isinstance(engine, bool) or not isinstance(engine, int):
        raise ValueError("engine_id must be an integer")
    name = _text(value.get("engine_name"), "engine_name")
    if name != binding["engine"]:
        raise ValueError("measurement engine_name differs from binding.engine")
    graph = _text(value.get("graph_id", value.get("benchmark")), "graph_id")
    device = _text(value.get("device_id", value.get("device")), "device_id")
    arch = _text(value.get("arch"), "arch")
    if not _ARCH.fullmatch(arch) or binding["arch"] not in (arch, "default"):
        raise ValueError("measurement architecture is invalid or differs from binding")
    features = _object(value.get("features"), "features")
    if any(not isinstance(key, str) or key.startswith("$") for key in features):
        raise ValueError("features require canonical published names without '$'")
    validate_signature(["$" + key for key in features])
    constraints = _object(value.get("constraints", {}), "constraints")
    if any(not name.startswith("global.") for name in constraints):
        raise ValueError("L1 measurements cannot pin kernel configuration knobs")
    if constraints.get("global.benchmarking", 0) != 0:
        raise ValueError("L1 measurements cannot enable benchmark searches")
    if "global.workspace_size_limit" in constraints:
        limit = constraints["global.workspace_size_limit"]
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 0
            or features.get("constraint.workspace_limit") != limit
        ):
            raise ValueError(
                "workspace constraint must match the published constraint.workspace_limit feature"
            )
    for key, item in features.items():
        if not isinstance(item, (str, int, float, bool)) or (
            isinstance(item, (int, float)) and not math.isfinite(item)
        ):
            raise ValueError(f"published feature {key} must be a finite scalar")
        if key in value and value[key] != item:
            raise ValueError(
                f"flattened feature {key} differs from the published feature map"
            )
    # RFC 0019 §13.2: the tri-state verdict rides on the row. A row checked wrong keeps its
    # place but loses every timing label (excluded later via `known_wrong`); null means
    # unknown, not correct.
    verdict = numerical_verdict(value.get(VERDICT))
    reason = numerical_reason(value.get(REASON))
    average = elapsed = spread = iterations = tflops = None
    if verdict is not False:
        # §11.2: the calibrated score trains on `avgTimeMs`; `robustMeanMs` is informational.
        average = _positive(
            value.get(LABEL_STATISTIC, value.get("avg_time_ms")), LABEL_STATISTIC
        )
        elapsed = _positive(value.get("robustMeanMs"), "robustMeanMs")
        spread = _optional_spread(
            value.get("stddevMs", value.get("stddev_ms")), "stddevMs"
        )
        iterations = _optional_count(
            value.get("iters", value.get("iterations")), "iters"
        )
        # Only a throughput label needs graph.flops (conv backward publishes none); a
        # published count must still be valid.
        if binding["metric"] == "tflops" or "graph.flops" in features:
            flops = _positive(features.get("graph.flops"), "full-graph graph.flops")
            tflops = flops / average / 1e9
            _positive(tflops, "derived tflops")
        if not _missing(value.get("tflops")):
            supplied = _positive(value["tflops"], "tflops")
            if tflops is None or not math.isclose(supplied, tflops, rel_tol=1e-10):
                raise ValueError(
                    f"supplied tflops differs from graph.flops/({LABEL_STATISTIC}*1e9)"
                )
    return {
        "benchmark": graph,
        "device": device,
        "arch": arch,
        "engine": engine,
        "engine_name": name,
        "binding": json.dumps(binding, sort_keys=True),
        "features": json.dumps(features, sort_keys=True),
        "is_valid": True,
        VERDICT: verdict,
        REASON: reason,
        "selection_mode": "immediate",
        "timing_statistic": LABEL_STATISTIC,
        LABEL_STATISTIC: average,
        "robustMeanMs": elapsed,
        "stddevMs": spread,
        "iters": iterations,
        "tflops": tflops,
        **features,
    }


def normalize_corpus(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("immediate corpus is empty")
    rows = [normalize_row(row) for row in frame.to_dict(orient="records")]
    result = pd.DataFrame(rows)
    if result.duplicated(["benchmark", "device", "engine"]).any():
        raise ValueError(
            "immediate corpus must contain one row per engine/graph/device, not a candidate sweep"
        )
    if (result.groupby("engine_name")["engine"].nunique() > 1).any():
        raise ValueError(
            "a canonical engine name cannot refer to multiple public engine identities"
        )
    # Rows that publish no count are not a second value: `nunique` skips them.
    flops = (
        result["graph.flops"]
        if "graph.flops" in result.columns
        else pd.Series(index=result.index, dtype=float)
    )
    if (flops.groupby(result["benchmark"]).nunique() > 1).any():
        raise ValueError(
            "full-graph logical work count cannot change across engines or devices"
        )
    for _, group in result.groupby("engine", sort=False):
        bindings = [validate_binding(value) for value in group["binding"]]
        recorded = (
            bindings[0]["selector_revision"],
            bindings[0]["metric"],
            bindings[0]["trained_against"],
        )
        if any(
            (
                binding["selector_revision"],
                binding["metric"],
                binding["trained_against"],
            )
            != recorded
            for binding in bindings[1:]
        ):
            raise ValueError(
                "engine selector, metric or descriptor provenance changed within the immediate corpus"
            )
    for _, group in result.groupby(["benchmark", "device"], sort=False):
        if flops[group.index].nunique() > 1 or group["arch"].nunique() != 1:
            raise ValueError(
                "engines disagree on full-graph work count or device architecture"
            )
    return result


def read_corpus(path: Path) -> pd.DataFrame:
    # `read_corpus_frame` picks the reader by suffix and pins `graph_id`/`device_id` to
    # text, which `_text` requires (a graph named `0123` must not become an int).
    return normalize_corpus(read_corpus_frame(path))


def training_binding(
    frame: pd.DataFrame, engine: str | None = None
) -> tuple[pd.DataFrame, dict]:
    if engine is not None:
        frame = frame[
            frame["engine_name"].eq(engine) | frame["engine"].astype(str).eq(engine)
        ]
    if frame.empty or frame["engine"].nunique() != 1:
        raise ValueError(
            "train one immediate engine at a time; select --engine by canonical name or public ID"
        )
    binding = validate_binding(frame.iloc[0]["binding"])
    return frame.copy(), binding


def validate_model(descriptor: dict) -> RankingMetric:
    """The admission rule for an L1 estimate (RFC 0019 §11.1); returns its metric.

    Engine selection compares these numbers across engines, so the score must name a
    registered metric, be calibrated, and carry that metric's direction.
    """
    score = descriptor.get("score", {})
    metric = ranking_metric(score.get("metric"))
    if descriptor.get("objective") != metric.objective:
        raise ValueError(
            f"L1 prediction of {metric.name!r} requires objective={metric.objective}"
        )
    # The runtime owns the transform vocabulary (`score_transform::isSupported`); this tool
    # can only score the INVERTIBLE inverses plus the omitted/empty identity transform.
    if score.get("calibrated") is not True or score.get("transform", "") not in (
        "",
        *INVERTIBLE_TRANSFORMS,
    ):
        raise ValueError(
            "L1 prediction requires a calibrated score, and uhd_gen can only "
            f"score identity (or omitted), {', '.join(INVERTIBLE_TRANSFORMS)} transforms"
        )
    validate_provenance(descriptor.get("trained_against"))
    validate_signature(descriptor.get("features_signature", []))
    return metric


def check_model_binding(
    descriptor: dict, frame: pd.DataFrame, feature_evaluator: str | Path | None = None
) -> None:
    """Whether `frame` is data this L1 model may be trained on or scored against."""
    metric = validate_model(descriptor)
    selector_revision = descriptor["trained_against"].get("selector_revision")
    for value in frame["binding"].unique():
        binding = validate_binding(value)
        require_binding_metric(
            binding,
            metric.name,
            f"L1 corpus for {binding['engine']}",
            selector_revision,
        )
        compare_provenance(descriptor["trained_against"], binding["trained_against"])
    signature = descriptor.get("features_signature", [])
    if signature:
        check_signature_evaluates(
            signature,
            frame["features"].unique(),
            descriptor.get("categorical_encoding"),
            feature_evaluator,
        )


def prediction_scorer(descriptor: dict, responses: list[dict]):
    """Use real runtime predictions for provider-owned native/custom implementations."""
    import numpy as np

    identity = descriptor["id"]
    metric = validate_model(descriptor).name
    selected = {}
    for response in responses:
        if response.get("model") != identity:
            continue
        status = response.get("status")
        if status not in (
            "AVAILABLE",
            "available",
            1,
            "UNAVAILABLE",
            "unavailable",
            0,
            "INVALID",
            "invalid",
            2,
        ):
            raise ValueError(f"runtime prediction has an unknown status {status!r}")
        binding = validate_binding(response.get("binding"))
        compare_provenance(descriptor["trained_against"], binding["trained_against"])
        engine = response.get("engine_id")
        if isinstance(engine, bool) or not isinstance(engine, int):
            raise ValueError("runtime prediction engine_id must be an integer")
        key = (
            _text(response.get("graph_id"), "graph_id"),
            _text(response.get("device_id"), "device_id"),
            engine,
        )
        if key in selected:
            raise ValueError(
                "duplicate runtime prediction for the same engine/graph/device"
            )
        # RFC 0019 §11.4: the answer names its metric and the host checks it. A value in
        # another metric is not converted, it is the wrong answer.
        if response.get("metric") != metric:
            raise ValueError(
                f"runtime prediction answers metric {response.get('metric')!r}, "
                f"not the model's {metric!r}"
            )
        # A declined answer (UNAVAILABLE, INVALID, or a value the metric cannot take) leaves
        # the engine unscored for that graph, as the runtime would.
        value = response.get("value")
        available = status in ("AVAILABLE", "available", 1) and is_valid_metric_value(
            metric, value
        )
        selected[key] = (response, binding, float(value) if available else float("nan"))
    if not selected:
        raise ValueError(f"no runtime predictions for UHD {identity}")

    def score(frame):
        values = []
        for row in frame.to_dict(orient="records"):
            key = (row["benchmark"], row["device"], row["engine"])
            if key not in selected:
                raise ValueError(f"runtime prediction missing for {key}")
            response, binding, prediction = selected[key]
            measured_binding = validate_binding(row["binding"])
            if (
                binding_identity(binding, include_uhd_id=False)
                != binding_identity(measured_binding, include_uhd_id=False)
                or response.get("arch") != row["arch"]
                or _object(response.get("features"), "prediction features")
                != _object(row["features"], "features")
            ):
                raise ValueError(
                    "runtime prediction request differs from the measured graph/device/constraints"
                )
            values.append(prediction)
        return np.asarray(values, dtype=float)

    return score


def _engine_name_to_id(name: str) -> int:
    """`engineNameToId`: FNV-1a over the registered name, read back as a signed int64."""
    digest = 0xCBF29CE484222325
    for byte in name.encode("utf-8"):
        digest = ((digest ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return digest - (1 << 64) if digest >= 1 << 63 else digest


_STATIC_PRECEDENCE = {
    _engine_name_to_id(name): rank
    for name, rank in (
        ("MIOPEN_ENGINE", 0),
        ("ASM_SDPA_ENGINE", 1),
        ("ROCKE_ENGINE", 2),
        ("MIOPEN_ENGINE_DETERMINISTIC", 4),
    )
}


def _static_engine_rank(engine_id: int) -> tuple[int, int]:
    """Where the runtime's static rules (`sortEngineIds`) put an engine it could not score.

    Unlisted engines keep provider enumeration order, which a corpus does not record, so
    the public ID stands in; HIPDNN_HEUR_FALLBACK_ENGINE_ORDER is not applied.
    """
    return _STATIC_PRECEDENCE.get(engine_id, 3), engine_id


def evaluate_immediate(
    frame: pd.DataFrame,
    bundles: list,
    *,
    eval_fraction: float,
    seed: int,
    include_per_problem: bool = False,
    feature_evaluator: str | Path | None = None,
) -> dict:
    """Physical score error and selection regret against other immediate engines only."""
    import numpy as np
    from .evaluate import (
        REPORT_SCHEMA,
        _holdout_integrity,
        _summarise,
        problem_keys,
        regret_of,
        resolve_grouping,
        split_problems,
    )

    frame = normalize_corpus(frame)
    grouping = resolve_grouping(frame)
    keys = problem_keys(frame, grouping)
    split = split_problems(keys, eval_fraction, seed)
    # Split over every row, like training, so both agree on held-out problems; then drop
    # rows checked wrong from scoring and the oracle (RFC 0019 §13.2).
    held_out = frame[keys.isin(split.eval_problems) & ~known_wrong(frame)].copy()
    by_engine = {}
    integrity = []
    metrics = set()
    for bundle in bundles:
        metrics.add(validate_model(bundle.descriptor))
        engine = validate_binding(bundle.manifest.get("binding"))["engine"]
        if engine in by_engine:
            raise ValueError(f"multiple models supplied for immediate engine {engine}")
        by_engine[engine] = bundle
        integrity.append(_holdout_integrity(bundle, split.eval_problems)["status"])
    # Engines are compared in one metric at a time (RFC 0019 §4.4): a `time` model and a
    # `tflops` model rank in opposite directions and different units.
    if len(metrics) != 1:
        raise ValueError(
            "cross-engine L1 evaluation needs every model to predict the same metric; got "
            + ", ".join(sorted(metric.name for metric in metrics))
        )
    metric = metrics.pop()
    name, label = metric.name, metric.label
    predicted_column = f"predicted_{name}"
    missing = set(frame["engine_name"]) - set(by_engine)
    if missing:
        raise ValueError(
            f"missing per-engine models; pass --additional-model-dir for {sorted(missing)}"
        )
    predicted = pd.Series(index=held_out.index, dtype=float)
    declined: dict[str, int] = {}
    for engine, group in frame.groupby("engine_name", sort=False):
        bundle = by_engine[engine]
        check_model_binding(bundle.descriptor, group, feature_evaluator)
        recorded_arch = bundle.manifest.get("arch", "default")
        if recorded_arch != "default" and not group["arch"].eq(recorded_arch).all():
            raise ValueError("model training architecture differs from corpus")
        measured_arches = set(group["arch"])
        training_arches = set(bundle.manifest.get("training_arches", []))
        if training_arches and not measured_arches <= training_arches:
            raise ValueError(f"evaluation contains unseen architectures for {engine}")
        selected = held_out[held_out["engine_name"].eq(engine)]
        if selected.empty:
            continue
        values = np.asarray(bundle.scorer(selected), dtype=float)
        if values.shape != (len(selected),):
            raise ValueError("L1 model returned the wrong number of predictions")
        # The runtime refuses a value the metric cannot take as INVALID and orders the
        # engine statically, so score it as a decline (log1p models can predict in (-1, 0)).
        impossible = np.array(
            [not is_valid_metric_value(name, float(value)) for value in values],
            dtype=bool,
        )
        values[impossible] = np.nan
        declined[engine] = int(impossible.sum())
        predicted.loc[selected.index] = values
    held_out = held_out.assign(**{predicted_column: predicted})
    scored = held_out[held_out[predicted_column].notna()]
    if scored.empty:
        raise ValueError(f"L1 model scored no evaluation row with a valid {name} value")

    # Keys are named by metric so a `time` report has no throughput-named fields.
    def calibration(group):
        measured = group[label].to_numpy(dtype=float)
        error = group[predicted_column].to_numpy(dtype=float) - measured
        relative = error / measured
        return {
            "rows": len(group),
            f"signed_bias_{name}": float(np.mean(error)),
            f"mean_absolute_error_{name}": float(np.mean(np.abs(error))),
            f"rmse_{name}": float(np.sqrt(np.mean(error * error))),
            "signed_relative_bias": float(np.mean(relative)),
            "mean_absolute_relative_error": float(np.mean(np.abs(relative))),
        }

    # The oracle is every valid measurement, scored or not, so a decline's loss shows. The
    # pick mirrors PredictionPolicy: scored engines best-first, then unscored, ties static.
    per_problem, regrets = [], []
    better = -1.0 if metric.objective == "max" else 1.0

    def runtime_rank(row):
        value = row[predicted_column]
        unscored = bool(np.isnan(value))
        return (
            unscored,
            0.0 if unscored else better * value,
            _static_engine_rank(int(row["engine"])),
        )

    for key, group in held_out.groupby(list(grouping.columns), sort=True):
        picked = min((row for _, row in group.iterrows()), key=runtime_rank)
        best = float(
            group[label].max() if metric.objective == "max" else group[label].min()
        )
        regret = (
            regret_of(float(picked[label]), best, metric.objective)
            if len(group) > 1
            else None
        )
        if regret is not None:
            regrets.append(regret)
        per_problem.append(
            {
                "key": list(key),
                "engines": len(group),
                "scored_engines": int(group[predicted_column].notna().sum()),
                "picked_engine": int(picked["engine"]),
                "picked_by": (
                    "static_order"
                    if np.isnan(picked[predicted_column])
                    else "prediction"
                ),
                f"picked_{name}": float(picked[label]),
                f"best_immediate_{name}": best,
                "immediate_selection_regret": regret,
            }
        )
    rows_per_engine = held_out.groupby("engine_name")[predicted_column]
    status = (
        "COMPROMISED"
        if "COMPROMISED" in integrity
        else "unknown" if "unknown" in integrity else "held_out"
    )
    warnings = []
    if status != "held_out":
        warnings.append(
            f"Holdout integrity is {status}; only recorded disjoint graph/device keys prove independence."
        )
    if eval_fraction == 1:
        warnings.append(
            "Full supplied corpus evaluated; training overlap is reported separately."
        )
    report = {
        "schema": REPORT_SCHEMA,
        "role": ROLE,
        "metric": name,
        "target": label,
        "objective": metric.objective,
        "corpus": {"rows": len(frame), "problems": len(set(keys))},
        "grouping": {
            "columns": list(grouping.columns),
            "degraded": False,
            "detail": grouping.detail,
        },
        "split": {
            "method": split.method,
            "unit": "graph/device",
            "seed": seed,
            "eval_fraction": eval_fraction,
            "train_problems": len(split.train_problems),
            "eval_problems": len(split.eval_problems),
            "eval_problem_keys": [list(key) for key in split.eval_problems],
        },
        "metrics": {
            "problems_scored": len(per_problem),
            "calibration": calibration(scored),
            "prediction_coverage": {
                "rows": len(held_out),
                "scored_rows": len(scored),
                "fraction": len(scored) / len(held_out),
                "per_engine": {
                    engine: {
                        "rows": int(values.size),
                        "scored_rows": int(values.notna().sum()),
                    }
                    for engine, values in rows_per_engine
                },
                "problems_fully_scored": sum(
                    item["scored_engines"] == item["engines"] for item in per_problem
                ),
                "problems_picked_by_static_order": sum(
                    item["picked_by"] == "static_order" for item in per_problem
                ),
                "detail": "Coverage is separate from accuracy: calibration is over scored rows "
                "only, while selection regret counts every measured engine and places "
                "unscored ones in static order after the scored ones, as the runtime does",
            },
            "unscored_rows": {
                "total": len(held_out) - len(scored),
                "per_engine": declined,
                "detail": f"predictions the runtime would refuse as "
                f"invalid {name} values; the engine is "
                f"unscored for these",
            },
            "per_engine": {
                engine: calibration(group)
                for engine, group in scored.groupby("engine_name")
            },
            "immediate_selection": {
                "problems_compared": len(regrets),
                "regret": _summarise(regrets),
                "baseline": "best measured immediate engine; never tuned configurations",
            },
        },
        "holdout_integrity": {
            "status": status,
            "detail": "Compared recorded training graph/device keys with every evaluated problem",
        },
        "warnings": warnings,
    }
    if include_per_problem:
        report["per_problem"] = per_problem
        report["per_row"] = [
            {
                "key": [row["benchmark"], row["device"]],
                "engine": row["engine"],
                f"measured_{name}": row[label],
                predicted_column: row[predicted_column],
                f"signed_error_{name}": row[predicted_column] - row[label],
                "signed_relative_error": row[predicted_column] / row[label] - 1,
            }
            for row in scored.to_dict(orient="records")
        ]
    return report

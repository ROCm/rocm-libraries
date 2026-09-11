# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Engine-immediate measurements shared by collection, training and evaluation."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

from .features import signature_references
from .provenance import compare_provenance, validate_provenance

ROLE = "predict_engine_tflops"
_ARCH = re.compile(r"^gfx[a-z0-9_-]+$")
_LEAKED_FIELDS = frozenset({
    "kernel", "kernel_features", "candidate", "candidate_id", "candidates", "results",
    "knob", "knobs", "knob_settings", "configuration", "engine_config",
    "prediction", "predicted_tflops", "tflops", "timing", "latency", "robustMeanMs",
    "robust_time_ms", "minTimeMs", "avgTimeMs", "succeeded", "is_valid",
})

#: RFC 0019.13 §11.2 (:2003): "A UHD declaring `calibrated: true` MUST train its score
#: on `avgTimeMs`", and §10.6.2 (:1914-1916) repeats it for the engine-level estimate --
#: minimum- and robust-mean-over-iterations are optimistically biased, and two engines
#: trained on different statistics are not comparable at all. L1 declares
#: `score.calibrated: true` unconditionally (`validate_model`), so its label is the
#: arithmetic mean and nothing else.
LABEL_STATISTIC = "avgTimeMs"

#: What a normalized L1 row is allowed to carry back in: the label, the §8.5 statistic
#: kept beside it for information, the derived rate and the validity flag. Every other
#: name in `_LEAKED_FIELDS` is candidate/search data an immediate measurement must not
#: have. This exemption is for the ENVELOPE only -- `validate_signature` still rejects
#: every `_LEAKED_FIELDS` name in a feature, so no L1 feature can read its own label.
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
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{where} must be a positive finite number")
    return float(value)


def _optional_spread(value, where: str) -> float | None:
    """§8.3's `stddevMs`: nonnegative when the row carries a measurement, else absent."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{where} must be a nonnegative finite number when present")
    return float(value)


def _optional_count(value, where: str) -> int | None:
    """§8.3's `iters`: a whole iteration count when the row carries a measurement."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or float(value) != int(value)):
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
    binding["trained_against"] = validate_provenance(binding.get("trained_against"))
    return binding


def validate_signature(signature: list, published: set[str] | None = None) -> None:
    """Only provider-published graph/device/constraint inputs, never measured outputs."""
    for reference in signature_references(signature):
        name = reference.removeprefix("$")
        parts = set(re.split(r"[.\[\]]+", name))
        if "." not in name or parts & _LEAKED_FIELDS:
            raise ValueError(f"L1 features cannot depend on kernel/candidate/timing inputs: {reference}")
        if published is not None and name not in published:
            raise ValueError(f"L1 feature is not published by the engine: {reference}")


def normalize_row(value: dict) -> dict:
    """Import exactly one no-search execution; derive, rather than trust, TFLOPS."""
    value = _object(value, "immediate measurement")
    forbidden = {name for name in value if name in _LEAKED_FIELDS - _LABEL_FIELDS
                 or name.startswith(("kernel.", "candidate.", "knob."))}
    if forbidden:
        raise ValueError(f"immediate measurements contain candidate/search data: {sorted(forbidden)}")
    if value.get("selection_mode") != "immediate":
        raise ValueError("L1 labels require selection_mode=immediate")
    # `hipdnn_bench --collect-immediate` declares `robustMeanMs` -- the statistic it
    # ranks and reports on. A row this function has already produced declares the
    # LABEL statistic instead, so both spellings read back and the label below is
    # `avgTimeMs` either way.
    if value.get("timing_statistic") not in ("robustMeanMs", LABEL_STATISTIC):
        raise ValueError(
            f"L1 labels require timing_statistic robustMeanMs or {LABEL_STATISTIC}")
    if value.get("is_valid") is not True:
        raise ValueError("L1 labels require is_valid=true")
    binding = validate_binding(value.get("binding"))
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
        if (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
                or features.get("constraint.workspace_limit") != limit):
            raise ValueError("workspace constraint must match the published constraint.workspace_limit feature")
    for key, item in features.items():
        if not isinstance(item, (str, int, float, bool)) or (isinstance(item, (int, float)) and not math.isfinite(item)):
            raise ValueError(f"published feature {key} must be a finite scalar")
        if key in value and value[key] != item:
            raise ValueError(f"flattened feature {key} differs from the published feature map")
    flops = _positive(features.get("graph.flops"), "full-graph graph.flops")
    # RFC 0019.13 §11.2 (:2003) and §10.6.2 (:1914-1916): the score this model declares
    # `calibrated: true` MUST be trained on `avgTimeMs`. `robustMeanMs` stays on the row
    # as the informational §8.5 statistic, never as the label.
    average = _positive(value.get(LABEL_STATISTIC, value.get("avg_time_ms")), LABEL_STATISTIC)
    elapsed = _positive(value.get("robustMeanMs"), "robustMeanMs")
    spread = _optional_spread(value.get("stddevMs", value.get("stddev_ms")), "stddevMs")
    iterations = _optional_count(value.get("iters", value.get("iterations")), "iters")
    tflops = flops / average / 1e9
    _positive(tflops, "derived tflops")
    if "tflops" in value and not math.isclose(_positive(value["tflops"], "tflops"), tflops, rel_tol=1e-10):
        raise ValueError(f"supplied tflops differs from graph.flops/({LABEL_STATISTIC}*1e9)")
    return {"benchmark": graph, "device": device, "arch": arch, "engine": engine,
            "engine_name": name, "binding": json.dumps(binding, sort_keys=True),
            "features": json.dumps(features, sort_keys=True), "is_valid": True,
            "selection_mode": "immediate", "timing_statistic": LABEL_STATISTIC,
            LABEL_STATISTIC: average, "robustMeanMs": elapsed, "stddevMs": spread,
            "iters": iterations, "tflops": tflops, **features}


def normalize_corpus(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("immediate corpus is empty")
    rows = [normalize_row(row) for row in frame.to_dict(orient="records")]
    result = pd.DataFrame(rows)
    if result.duplicated(["benchmark", "device", "engine"]).any():
        raise ValueError("immediate corpus must contain one row per engine/graph/device, not a candidate sweep")
    if (result.groupby("engine_name")["engine"].nunique() > 1).any():
        raise ValueError("a canonical engine name cannot refer to multiple public engine identities")
    if (result.groupby("benchmark")["graph.flops"].nunique() > 1).any():
        raise ValueError("full-graph logical work count cannot change across engines or devices")
    for _, group in result.groupby("engine", sort=False):
        bindings = [validate_binding(value) for value in group["binding"]]
        recorded = (bindings[0]["selector_revision"], bindings[0]["trained_against"])
        if any((binding["selector_revision"], binding["trained_against"]) != recorded
               for binding in bindings[1:]):
            raise ValueError("engine selector or descriptor provenance changed within the immediate corpus")
    for _, group in result.groupby(["benchmark", "device"], sort=False):
        if group["graph.flops"].nunique() != 1 or group["arch"].nunique() != 1:
            raise ValueError("engines disagree on full-graph work count or device architecture")
    return result


def read_corpus(path: Path) -> pd.DataFrame:
    # The trainer's suffix rule, applied here too: --input decides the reader by its
    # suffix whichever role is being trained, so a published .parquet dataset is never
    # handed to the CSV reader.
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".json":
        content = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame([content] if isinstance(content, dict) else content)
    else:
        frame = pd.read_csv(path, dtype={"benchmark": str, "device": str, "graph_id": str, "device_id": str})
    return normalize_corpus(frame)


def training_binding(frame: pd.DataFrame, engine: str | None = None) -> tuple[pd.DataFrame, dict]:
    if engine is not None:
        frame = frame[frame["engine_name"].eq(engine) | frame["engine"].astype(str).eq(engine)]
    if frame.empty or frame["engine"].nunique() != 1:
        raise ValueError("train one immediate engine at a time; select --engine by canonical name or public ID")
    binding = validate_binding(frame.iloc[0]["binding"])
    return frame.copy(), binding


def validate_model(descriptor: dict) -> None:
    if descriptor.get("objective") != "max":
        raise ValueError("L1 prediction requires objective=max")
    score = descriptor.get("score", {})
    # The transform vocabulary belongs to `score_transform::isSupported` on the runtime
    # side; this narrower pair is not a second opinion about it. `evaluate`'s scorers
    # implement the identity and log1p inverses only, so a descriptor declaring any
    # other supported transform is loadable by the engine and not scoreable here --
    # a capability limit of this tool, reported where the scoring happens.
    if (score.get("units") != "tflops" or score.get("calibrated") is not True
            or score.get("transform") not in ("identity", "log1p")):
        raise ValueError("L1 prediction requires calibrated tflops, and uhd_gen can only "
                         "score identity or log1p transforms")
    validate_provenance(descriptor.get("trained_against"))
    validate_signature(descriptor.get("features_signature", []))


def check_model_binding(descriptor: dict, frame: pd.DataFrame) -> None:
    validate_model(descriptor)
    for value in frame["binding"].unique():
        binding = validate_binding(value)
        compare_provenance(descriptor["trained_against"], binding["trained_against"])
    for value in frame["features"].unique():
        validate_signature(descriptor.get("features_signature", []), set(_object(value, "features")))


def prediction_scorer(descriptor: dict, responses: list[dict]):
    """Use real runtime predictions for provider-owned native/custom implementations."""
    import numpy as np

    identity = descriptor["id"]
    selected = {}
    for response in responses:
        if response.get("model") != identity:
            continue
        if response.get("status") not in ("AVAILABLE", "available", 1):
            raise ValueError("runtime prediction is unavailable or invalid")
        binding = validate_binding(response.get("binding"))
        compare_provenance(descriptor["trained_against"], binding["trained_against"])
        engine = response.get("engine_id")
        if isinstance(engine, bool) or not isinstance(engine, int):
            raise ValueError("runtime prediction engine_id must be an integer")
        key = (_text(response.get("graph_id"), "graph_id"),
               _text(response.get("device_id"), "device_id"), engine)
        if key in selected:
            raise ValueError("duplicate runtime prediction for the same engine/graph/device")
        value = response.get("tflops")
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError("runtime prediction must be finite nonnegative physical tflops")
        selected[key] = (response, binding, float(value))
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
            if ({key: value for key, value in binding.items() if key != "uhd_id"}
                    != {key: value for key, value in measured_binding.items() if key != "uhd_id"}
                    or response.get("arch") != row["arch"]
                    or _object(response.get("features"), "prediction features") != _object(row["features"], "features")):
                raise ValueError("runtime prediction request differs from the measured graph/device/constraints")
            values.append(prediction)
        return np.asarray(values, dtype=float)

    return score


def evaluate_immediate(frame: pd.DataFrame, bundles: list, *, eval_fraction: float, seed: int,
                       include_per_problem: bool = False) -> dict:
    """Physical score error and selection regret against other immediate engines only."""
    import numpy as np
    from .evaluate import REPORT_SCHEMA, problem_keys, resolve_grouping, split_problems, regret_of, _summarise

    frame = normalize_corpus(frame)
    grouping = resolve_grouping(frame)
    keys = problem_keys(frame, grouping)
    split = split_problems(keys, eval_fraction, seed)
    held_out = frame[keys.isin(split.eval_problems)].copy()
    by_engine = {}
    integrity = []
    for bundle in bundles:
        validate_model(bundle.descriptor)
        engine = validate_binding(bundle.manifest.get("binding"))["engine"]
        if engine in by_engine:
            raise ValueError(f"multiple models supplied for immediate engine {engine}")
        by_engine[engine] = bundle
        recorded = bundle.manifest.get("training_problem_keys")
        if recorded is None:
            integrity.append("unknown")
        elif set(map(tuple, recorded)) & set(split.eval_problems):
            integrity.append("COMPROMISED")
        else:
            integrity.append("held_out")
    missing = set(frame["engine_name"]) - set(by_engine)
    if missing:
        raise ValueError(f"missing per-engine models; pass --additional-model-dir for {sorted(missing)}")
    predicted = pd.Series(index=held_out.index, dtype=float)
    for engine, group in frame.groupby("engine_name", sort=False):
        bundle = by_engine[engine]
        check_model_binding(bundle.descriptor, group)
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
        if values.shape != (len(selected),) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("L1 model returned invalid physical tflops predictions")
        predicted.loc[selected.index] = values
    held_out["predicted_tflops"] = predicted

    def calibration(group):
        measured = group["tflops"].to_numpy(dtype=float)
        error = group["predicted_tflops"].to_numpy(dtype=float) - measured
        relative = error / measured
        return {"rows": len(group), "signed_bias_tflops": float(np.mean(error)),
                "mean_absolute_error_tflops": float(np.mean(np.abs(error))),
                "rmse_tflops": float(np.sqrt(np.mean(error * error))),
                "signed_relative_bias": float(np.mean(relative)),
                "mean_absolute_relative_error": float(np.mean(np.abs(relative)))}

    per_problem, regrets = [], []
    for key, group in held_out.groupby(list(grouping.columns), sort=True):
        # Stable engine identity breaks prediction ties, independent of import order.
        ordered = group.sort_values("engine", kind="stable")
        picked = ordered.loc[ordered["predicted_tflops"].idxmax()]
        best = float(group["tflops"].max())
        regret = regret_of(float(picked["tflops"]), best, "max") if len(group) > 1 else None
        if regret is not None:
            regrets.append(regret)
        per_problem.append({"key": list(key), "engines": len(group), "picked_engine": int(picked["engine"]),
                            "picked_tflops": float(picked["tflops"]), "best_immediate_tflops": best,
                            "immediate_selection_regret": regret})
    status = "COMPROMISED" if "COMPROMISED" in integrity else "unknown" if "unknown" in integrity else "held_out"
    warnings = []
    if status != "held_out":
        warnings.append(f"Holdout integrity is {status}; only recorded disjoint graph/device keys prove independence.")
    if eval_fraction == 1:
        warnings.append("Full supplied corpus evaluated; training overlap is reported separately.")
    report = {
        "schema": REPORT_SCHEMA, "role": ROLE, "target": "tflops", "objective": "max",
        "corpus": {"rows": len(frame), "problems": len(set(keys))},
        "grouping": {"columns": list(grouping.columns), "degraded": False, "detail": grouping.detail},
        "split": {"method": split.method, "unit": "graph/device", "seed": seed,
                  "eval_fraction": eval_fraction, "train_problems": len(split.train_problems),
                  "eval_problems": len(split.eval_problems),
                  "eval_problem_keys": [list(key) for key in split.eval_problems]},
        "metrics": {"problems_scored": len(per_problem), "calibration": calibration(held_out),
                    "per_engine": {name: calibration(group) for name, group in held_out.groupby("engine_name")},
                    "immediate_selection": {"problems_compared": len(regrets),
                                            "regret": _summarise(regrets),
                                            "baseline": "best measured immediate engine; never tuned configurations"}},
        "holdout_integrity": {"status": status, "detail": "Compared recorded training graph/device keys with every evaluated problem"},
        "warnings": warnings,
    }
    if include_per_problem:
        report["per_problem"] = per_problem
        report["per_row"] = [{"key": [row["benchmark"], row["device"]], "engine": row["engine"],
                              "measured_tflops": row["tflops"], "predicted_tflops": row["predicted_tflops"],
                              "signed_error_tflops": row["predicted_tflops"] - row["tflops"],
                              "signed_relative_error": row["predicted_tflops"] / row["tflops"] - 1}
                             for row in held_out.to_dict(orient="records")]
    return report

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Audit queue/reference evidence and compare coverage before reporting timings."""

import argparse
import ast
from collections import Counter
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import yaml

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

from rocjitsu_evidence import digest, source_name
from rocjitsu_pytest import write_json


def key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def percentiles(values):
    if not values:
        return dict(count=0)
    values = sorted(values)

    def percentile(q):
        index = (len(values) - 1) * q
        lo = int(index)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    return dict(
        count=len(values),
        sum=sum(values),
        min=min(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
        p95=percentile(0.95),
        max=max(values),
    )


def workload_coverage(plan):
    admitted, excluded = plan["jobs"], plan["excluded"]
    admitted_sources = {source_name(Path(j["source"])) for j in admitted}
    excluded_sources = {source_name(Path(j["source"])) for j in excluded}
    selected_sources = (
        {source_name(Path(s["path"])) for s in plan.get("sources", [])}
        | admitted_sources
        | excluded_sources
    )
    admitted_work = sum(j["work"] for j in admitted)
    excluded_work = sum(j["work"] for j in excluded)
    total_work = admitted_work + excluded_work
    return dict(
        ceiling=plan.get("ceiling"),
        selected_yaml_files=len(selected_sources),
        admitted_queue_jobs=len(admitted),
        admitted_problem_descriptors=sum(len(j["problems"]) for j in admitted),
        threshold_excluded_queue_jobs=len(excluded),
        threshold_excluded_problem_descriptors=sum(
            len(j["problems"]) for j in excluded
        ),
        threshold_affected_yaml_files=len(excluded_sources),
        fully_excluded_yaml_files=len(excluded_sources - admitted_sources),
        admitted_declared_work=admitted_work,
        excluded_declared_work=excluded_work,
        excluded_declared_work_fraction=(
            excluded_work / total_work if total_work else None
        ),
        exclusions=sorted(
            [
                dict(
                    nodeid=j["nodeid"],
                    source=source_name(Path(j["source"])),
                    source_sha256=j["source_sha256"],
                    work=j["work"],
                    problems=j["problems"],
                    reason=j["reason"],
                )
                for j in excluded
            ],
            key=lambda j: j["nodeid"],
        ),
        work_definition="Batch/count times logical GEMM dimensions; excludes solution, variant and repetition multipliers. This is not a runtime estimate.",
    )


def resource_statistics(root, runtime):
    path = root / "resource-time.json"
    if not path.exists():
        return None
    usage = json.loads(path.read_text())
    wall = usage["wall_seconds"]
    user, system = usage["user_seconds"], usage["system_seconds"]
    if (
        not all(math.isfinite(v) for v in (wall, user, system))
        or wall <= 0
        or min(user, system) < 0
    ):
        raise ValueError("invalid elapsed/CPU time in resource-time.json")
    mean_cores = (user + system) / wall
    cores = runtime.get("available_cpus")
    if cores is not None and (type(cores) is not int or cores <= 0):
        raise ValueError("available_cpus must be a positive integer")
    return dict(
        **usage,
        mean_cpu_cores=mean_cores,
        cpu_capacity_fraction=mean_cores / cores if cores else None,
        timing_scope="Workload phase including job startup, build, run and cleanup; excludes shared artifact/environment setup and initial workload planning.",
    )


def semantic_solution(solution):
    result = copy.deepcopy(solution)
    # These fields label entries in a generated library. CustomKernel metadata
    # and all other fields stay in the key, including future unknown fields.
    for field in ("BaseName", "KernelNameMin", "SolutionNameMin", "SolutionIndex"):
        result.pop(field, None)
    return result


def solution_digest(solution):
    return hashlib.sha256(key(semantic_solution(solution)).encode()).hexdigest()


def solutions(path):
    data = yaml.load(path.read_text(), Loader=SafeLoader)
    headers = {
        "MinimumRequiredVersion",
        "ProblemSizes",
        "BiasTypeArgs",
        "ActivationArgs",
        "GateTypeArgs",
    }
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a solution list")
    result = []
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: malformed solution entry")
        if set(entry) <= headers:
            continue
        if not {"ProblemType", "SolutionIndex", "SolutionNameMin"} <= entry.keys():
            raise ValueError(f"{path}: unknown solution/header schema")
        result.append(entry)
    names = [s["SolutionNameMin"] for s in result]
    if not result or len(set(names)) != len(names):
        raise ValueError(f"{path}: empty or ambiguous solution inventory")
    return result


def read_ini(path):
    values = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        field, separator, value = line.partition("=")
        if not separator:
            raise ValueError(f"{path}: malformed INI line")
        values.setdefault(field, []).append(value)
    for field in (
        "problem-size",
        "init-seed",
        "num-elements-to-validate",
        "check-synchronizer",
        "PrintWinnersOnly",
        "num-benchmarks",
    ):
        if field not in values:
            raise ValueError(f"{path}: missing {field}")
    if values["PrintWinnersOnly"] != ["False"]:
        raise ValueError(f"{path}: winner-only output hides coverage")
    if values["init-seed"] == ["0"]:
        raise ValueError(f"{path}: time-dependent initialization seed")
    if values["log-level"] != ["Debug"]:
        raise ValueError(f"{path}: client log level can hide verdicts")
    return values


PROBLEM_FIELDS = {
    "problem-size",
    "a-strides",
    "b-strides",
    "c-strides",
    "d-strides",
    "e-strides",
    "gate-strides",
    "bias-strides",
}
PATH_FIELDS = {
    "library-file",
    "code-object",
    "results-file",
    "library-update-file",
    "library-update-comment",
}
VALIDATIONS = {
    "PASSED",
    "FAILED",
    "NO_CHECK",
    "WRONG_HARDWARE",
    "DID_NOT_SATISFY_ASSERTS",
    "UNSUPPORTED_XCD_TOPOLOGY",
    "INVALID",
}


def ini_shapes(values):
    shapes = []
    count = len(values["problem-size"])
    for index in range(count):
        shape = {}
        for field in PROBLEM_FIELDS:
            if field not in values:
                continue
            if len(values[field]) not in (1, count):
                raise ValueError(f"{field}: unexpected stride/size list length")
            shape[field] = values[field][index if len(values[field]) == count else 0]
        shapes.append(shape)
    return shapes


def group_indices(path):
    match = re.fullmatch(r".+_(\d+)_(\d+)", path)
    if not match:
        raise ValueError(f"unknown benchmark group directory: {path}")
    return int(match[1]), int(match[2])


def client_events(path):
    header, ini, block = None, None, -1
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.startswith("loading config file "):
            ini = line.removeprefix("loading config file ").strip()
            header = None
        elif line.startswith("run,problem-progress,"):
            header = next(csv.reader([line]))
            required = {
                "run",
                "problem-progress",
                "solution-progress",
                "operation",
                "problem-sizes",
                "solution",
                "validation",
            }
            if not required <= set(header) or len(set(header)) != len(header):
                raise ValueError(f"{path}:{number}: unknown client header")
            block += 1
        elif re.match(r"^\d+,\d+/\d+,\d+/\d+,", line):
            if not header or not ini:
                raise ValueError(f"{path}:{number}: client row lacks header/config")
            skipped = re.fullmatch(
                r"(\d+),(\d+/\d+),(\d+/\d+),Solution Skipped \(slow or doing restoration\): (.+)",
                line,
            )
            if skipped:
                row = dict(
                    zip(
                        ("run", "problem-progress", "solution-progress", "solution"),
                        skipped.groups(),
                    )
                )
                row["validation"] = "SLOW_OR_RESTORATION_SKIP"
            else:
                values = next(csv.reader([line]))
                if len(values) != len(header):
                    raise ValueError(f"{path}:{number}: malformed client row")
                row = dict(zip(header, values))
                if row["validation"] not in VALIDATIONS:
                    raise ValueError(
                        f"{path}:{number}: unknown validation {row['validation']!r}"
                    )
            yield dict(row=row, ini=ini, line=number, block=block)


def audit(root):
    plan = json.loads((root / "shape-plan.json").read_text())
    progress = (
        json.loads((root / "progress.json").read_text())
        if (root / "progress.json").exists()
        else dict(selected=0, tests={})
    )
    execution = (
        json.loads((root / "execution.json").read_text())
        if (root / "execution.json").exists()
        else dict(
            complete=False, exit_code=1, error="missing supervisor execution record"
        )
    )
    runtime = json.loads((root / "runtime.json").read_text())
    shape_mode = runtime["strategy"] == "shape"
    if shape_mode and (root / "queue-events.jsonl").exists():
        from rocjitsu_shape_queue import read_progress

        progress = read_progress(root)
    admitted = plan["jobs"]
    excluded = plan["excluded"]
    expected = (
        {j["nodeid"] for j in admitted + excluded}
        if shape_mode
        else {s["nodeid"] for s in plan["sources"]}
    )
    errors, ledger, inventory, timings = [], [], [], []
    try:
        resource_time = resource_statistics(root, runtime)
    except (ValueError, KeyError, TypeError, OSError) as error:
        resource_time = None
        errors.append(f"resource accounting: {error}")
    if progress.get("queue_journal_error"):
        errors.append(progress["queue_journal_error"])
    if set(progress["tests"]) != expected or progress["selected"] != len(expected):
        errors.append("planned and reported tests differ")
    if not execution.get("complete") or execution["exit_code"]:
        errors.append("suite failed, timed out or was incomplete")
    if not shape_mode and excluded:
        errors.append("whole-YAML reference requires a below-ceiling cohort")
    categories, verdicts = Counter(), Counter()
    sessions = []
    for job in excluded:
        categories["threshold_exclusion"] += 1
        timings.append(
            dict(
                nodeid=job["nodeid"],
                category="threshold_exclusion",
                reason=job["reason"],
                wall_seconds=None,
                build_seconds=None,
                run_seconds=None,
                evidence_seconds=None,
                worker=None,
                queue_position=None,
                source=source_name(Path(job["source"])),
                work=job["work"],
                censored=False,
                client_verdicts="{}",
            )
        )
        if (
            shape_mode
            and progress["tests"].get(job["nodeid"], {}).get("outcome") != "skipped"
        ):
            errors.append(f"{job['nodeid']}: gate exclusion was not reported")
    item_paths = (
        list(root.glob("jobs/*/evidence/*/item.json"))
        if shape_mode
        else list(root.glob("evidence/*/item.json"))
    )
    items = {}
    for path in item_paths:
        item = json.loads(path.read_text())
        if item["nodeid"] in items:
            errors.append(f"duplicate evidence for {item['nodeid']}")
        items[item["nodeid"]] = path, item
    execution_expected = {j["nodeid"] for j in admitted} if shape_mode else expected
    if set(items) - execution_expected:
        errors.append("unexpected executed items in evidence")
    if len(expected) != len(admitted + excluded if shape_mode else plan["sources"]):
        errors.append("duplicate planned identities")
    collections = (
        list(root.glob("jobs/*/evidence/collection-*.json"))
        if shape_mode
        else list(root.glob("evidence/collection-*.json"))
    )
    if not collections:
        errors.append("missing collection manifest")
    for collection in collections:
        value = json.loads(collection.read_text())
        selected = value["selected"]
        if not shape_mode and selected and {i["nodeid"] for i in selected} != expected:
            errors.append("reference collection differs from plan")
    if shape_mode:
        for job in admitted:
            generated = root / "shape-yamls" / Path(job["yaml"]).name
            if not generated.is_file() or digest(generated) != job["yaml_sha256"]:
                errors.append(f"{job['nodeid']}: generated input YAML changed")
    for nodeid in sorted(execution_expected):
        record = progress["tests"].get(nodeid, {})
        reports = record.get("phases", {})
        reason = next(
            (
                p.get("detail") or p.get("wasxfail")
                for p in reports.values()
                if p.get("outcome") in ("skipped", "failed")
            ),
            None,
        )
        if shape_mode and record.get("timed_out"):
            category = "job_timeout"
        elif record.get("outcome") == "incomplete":
            category = (
                "claimed_without_result" if "queue_position" in record else "unstarted"
            )
        elif record.get("outcome") == "skipped":
            category = (
                "xfail"
                if any(p.get("wasxfail") for p in reports.values())
                else "pytest_skip"
            )
        else:
            category = record.get("outcome", "missing_result")
        categories[category] += 1
        timing = dict(
            nodeid=nodeid,
            category=category,
            reason=reason,
            wall_seconds=record.get(
                "wall_seconds", sum(p["seconds"] for p in reports.values())
            ),
            build_seconds=None,
            run_seconds=None,
            evidence_seconds=None,
            worker=record.get("worker"),
            queue_position=record.get("queue_position"),
            source=None,
            work=next((j["work"] for j in admitted if j["nodeid"] == nodeid), None),
            censored=category
            in (
                "job_timeout",
                "claimed_without_result",
                "unstarted",
                "incomplete",
                "missing_result",
            ),
            client_verdicts="{}",
        )
        if category in ("claimed_without_result", "unstarted", "missing_result"):
            timing["wall_seconds"] = None
        timings.append(timing)
        if nodeid not in items:
            errors.append(f"{nodeid}: no retained item evidence")
            continue
        path, item = items[nodeid]
        if shape_mode:
            job = next(j for j in admitted if j["nodeid"] == nodeid)
            log = root / "jobs" / job["id"] / "test.log"
            found = (
                re.findall(r"^mirage: session (\S+)", log.read_text(), re.M)
                if log.exists()
                else []
            )
            if len(found) != 1:
                errors.append(f"{nodeid}: expected one independent Mirage session")
            sessions.extend(found)
        timing["source"] = item["source"]
        timing["worker"] = item.get("worker", timing["worker"])
        if shape_mode and item.get("job") != next(
            j for j in admitted if j["nodeid"] == nodeid
        ):
            errors.append(f"{nodeid}: executed job differs from plan")
        if item["source_sha256"] != next(
            (
                s["sha256"]
                for s in plan["sources"]
                if source_name(Path(s["path"])) == item["source"]
            ),
            None,
        ):
            errors.append(f"{nodeid}: source hash differs from plan")
        if category in ("pytest_skip", "xfail"):
            continue
        phases = item["phases"]
        if [p["name"] for p in phases] != ["build", "run"]:
            errors.append(f"{nodeid}: build/run did not both complete")
            continue
        if not all(p["completed"] for p in phases):
            errors.append(f"{nodeid}: build/run helper failed or was interrupted")
        timing.update(
            build_seconds=phases[0].get("wall_seconds"),
            run_seconds=phases[1].get("wall_seconds"),
            evidence_seconds=sum(p.get("evidence_seconds", 0) for p in phases),
        )
        try:
            inv, rows = audit_item(root, path.parent, item)
            inventory.extend(inv)
            ledger.extend(rows)
            statuses = Counter(row["validation"] for row in rows)
            verdicts.update(statuses)
            timing["client_verdicts"] = key(statuses)
            if not rows:
                raise ValueError("no client verdicts")
            if record.get("outcome") == "passed" and (
                statuses["FAILED"] or statuses["INVALID"]
            ):
                raise ValueError("client failure contradicts pytest pass")
        except (ValueError, KeyError, OSError) as error:
            errors.append(f"{nodeid}: {error}")
    requested = [
        dict(
            source=source_name(Path(j["source"])),
            source_sha256=j["source_sha256"],
            group=[j.get("problem_type_index"), j.get("group_index")],
            problems=j["problems"],
        )
        for j in admitted
    ]
    test_code = sorted(
        {
            (i.get("test_sha256"), i.get("helper_sha256"))
            for _, i in items.values()
            if i["phases"]
        }
    )
    if len(sessions) != len(set(sessions)):
        errors.append("Mirage sessions are reused across shape jobs")
    summary = dict(
        schema=1,
        runtime=runtime,
        coverage=workload_coverage(plan),
        resource_time=resource_time,
        test_code=test_code,
        sources=plan["sources"],
        execution=execution,
        counts=categories,
        client_verdicts=verdicts,
        unique_mirage_sessions=len(set(sessions)) if shape_mode else None,
        timing_seconds={
            field: percentiles([t[field] for t in timings if t[field] is not None])
            for field in (
                "wall_seconds",
                "build_seconds",
                "run_seconds",
                "evidence_seconds",
            )
        },
        timing_by_category={
            category: percentiles(
                [
                    t["wall_seconds"]
                    for t in timings
                    if t["category"] == category and t["wall_seconds"] is not None
                ]
            )
            for category in categories
        },
        requested=requested,
        inventory=inventory,
        ledger=ledger,
        audit_errors=errors,
        evidence_limits=[
            "Client records show processed solution/problem verdicts, not exact device launch counts.",
            "Equal seeds do not prove identical tensors across different allocation extents.",
            "One paired trial is calibration, not a statistical speedup estimate.",
        ],
    )
    destination = root / "audit"
    destination.mkdir(exist_ok=True)
    write_json(destination / "summary.json", summary)
    with (destination / "jobs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(timings[0]) if timings else ["nodeid"]
        )
        writer.writeheader()
        writer.writerows(timings)
    for name, entries in (
        ("client-verdicts", ledger),
        ("accepted-solutions", inventory),
    ):
        (destination / f"{name}.jsonl").write_text(
            "".join(key(e) + "\n" for e in entries)
        )
    return summary


def audit_item(root, directory, item):
    phases = {p["name"]: p for p in item["phases"]}
    for phase in phases.values():
        for entry in phase["files"]:
            if digest(directory / phase["name"] / entry["path"]) != entry["sha256"]:
                raise ValueError(f"retained metadata hash mismatch: {entry['path']}")
    output = Path(phases["run"]["output_dir"])
    groups, inis, inventory = {}, {}, []
    for entry in phases["run"]["files"]:
        relative = Path(entry["path"])
        if (
            len(relative.parts) == 4
            and relative.parts[0] == "1_BenchmarkProblems"
            and relative.parts[2] == "Data"
            and relative.suffix == ".yaml"
        ):
            groups[relative.parts[1]] = solutions(directory / "run" / relative)
        if relative.name == "ClientParameters.ini":
            inis[str(output / relative)] = read_ini(directory / "run" / relative)
    if not groups or not inis:
        raise ValueError("missing accepted solutions or effective client INIs")
    for ini, values in inis.items():
        relative = Path(ini).relative_to(output)
        group = relative.parts[1]
        if group not in groups:
            raise ValueError("client config has no matching solution inventory")
        identity = item.get("job", {})
        indices = (
            [identity["problem_type_index"], identity["group_index"]]
            if "problem_type_index" in identity
            else list(group_indices(group))
        )
        shapes = ini_shapes(values)
        controls = {
            k: v for k, v in values.items() if k not in PROBLEM_FIELDS | PATH_FIELDS
        }
        for shape in shapes:
            for solution in groups[group]:
                inventory.append(
                    dict(
                        source=item["source"],
                        group=indices,
                        shape=shape,
                        solution_sha256=solution_digest(solution),
                        controls=controls,
                    )
                )
    log = directory / "call-stdout.log"
    rows = []
    seen = Counter()
    events_all = list(client_events(log))
    log_hash = digest(log)
    for event in events_all:
        if event["ini"] not in inis:
            raise ValueError("client log references an unretained INI")
        values = inis[event["ini"]]
        group = Path(event["ini"]).relative_to(output).parts[1]
        identity = item.get("job", {})
        indices = (
            [identity["problem_type_index"], identity["group_index"]]
            if "problem_type_index" in identity
            else list(group_indices(group))
        )
        row = event["row"]
        matches = [s for s in groups[group] if s["SolutionNameMin"] == row["solution"]]
        if len(matches) != 1:
            raise ValueError(
                "client solution name does not resolve uniquely to metadata"
            )
        shapes = ini_shapes(values)
        problem_index = int(row["problem-progress"].split("/")[0])
        grouped = values.get("grouped-gemm") == ["True"]
        variant = problem_index // len(shapes) if not grouped else 0
        if row["validation"] == "SLOW_OR_RESTORATION_SKIP":
            shape = (
                dict(grouped=shapes) if grouped else shapes[problem_index % len(shapes)]
            )
        else:
            logical = ast.literal_eval(row["problem-sizes"])
            if grouped:
                expected_sizes = [
                    tuple(int(n) for n in s["problem-size"].split(",")) for s in shapes
                ]
                if list(logical) != expected_sizes:
                    raise ValueError("grouped client shapes differ from effective INI")
                shape = dict(grouped=shapes)
            else:
                shape = shapes[problem_index % len(shapes)]
                if tuple(int(n) for n in shape["problem-size"].split(",")) != logical:
                    raise ValueError(
                        "client progress/shape does not match INI ordering"
                    )
        processed = (
            event["ini"],
            row["run"],
            row["problem-progress"],
            row["solution-progress"],
        )
        seen[processed] += 1
        if seen[processed] > 1:
            raise ValueError("duplicate processed client entry")
        rows.append(
            dict(
                source=item["source"],
                group=indices,
                shape=shape,
                solution_sha256=solution_digest(matches[0]),
                axes={
                    k: row[k]
                    for k in ("operation", "bias-type", "factor-dim", "activation-type")
                    if k in row
                },
                run=row["run"],
                variant_index=variant,
                validation=row["validation"],
                log_path=str(log.relative_to(root)),
                log_line=event["line"],
                log_sha256=log_hash,
                table_block=event["block"],
            )
        )
    # Progress indices include runtime applicability rejections and explicit
    # restoration skips. Missing entries invalidate a completed sweep claim.
    for ini in inis:
        events = [e for e in events_all if e["ini"] == ini]
        if not events:
            raise ValueError("INI has no processed client entries")
        problem_last = {int(e["row"]["problem-progress"].split("/")[1]) for e in events}
        solution_last = {
            int(e["row"]["solution-progress"].split("/")[1]) for e in events
        }
        if len(problem_last) != 1 or len(solution_last) != 1:
            raise ValueError("inconsistent client progress bounds")
        solutions_idx = {
            s["SolutionIndex"] for s in groups[Path(ini).relative_to(output).parts[1]]
        }
        if max(solutions_idx) != next(iter(solution_last)):
            raise ValueError("client progress and solution inventory differ")
        expected = {
            (
                ini,
                str(run),
                f"{p}/{next(iter(problem_last))}",
                f"{s}/{next(iter(solution_last))}",
            )
            for run in range(int(inis[ini]["num-benchmarks"][0]))
            for p in range(next(iter(problem_last)) + 1)
            for s in solutions_idx
        }
        if {e for e in seen if e[0] == ini} != expected:
            raise ValueError("processed client sweep is incomplete")
    return inventory, rows


def compare(left, right):
    errors = [
        f"{name}: {e}"
        for name, report in (("reference", left), ("candidate", right))
        for e in report["audit_errors"]
    ]
    for field in (
        "artifact_run_id",
        "client_sha256",
        "rocjitsu_sha256",
        "mirage_sha256",
        "compiler_sha256",
        "seed",
        "target",
        "profile",
    ):
        if left["runtime"][field] != right["runtime"][field]:
            errors.append(f"runtime identity differs: {field}")
    for field in ("available_cpus", "runner_name"):
        if left["runtime"].get(field) != right["runtime"].get(field):
            errors.append(f"runtime identity differs: {field}")
    model_configs = []
    for report in (left, right):
        config = copy.deepcopy(report["runtime"].get("config", {}))
        for field in (
            "cpu_thread_budget",
            "num_threads",
            "cpu_dispatch_threads",
            "async_helper_threads",
        ):
            config.pop(field, None)
        model_configs.append(config)
    if model_configs[0] != model_configs[1]:
        errors.append("emulator model/configuration differs beyond thread allocation")
    if left.get("coverage") != right.get("coverage"):
        errors.append("threshold policy or workload accounting differs")
    differences = {}
    if left["test_code"] != right["test_code"]:
        errors.append("executed test/helper code differs")

    def source_inventory(report):
        return Counter(
            key(
                dict(
                    path=source_name(Path(s["path"])),
                    sha256=s["sha256"],
                    markers=s["markers"],
                )
            )
            for s in report["sources"]
        )

    if source_inventory(left) != source_inventory(right):
        errors.append("source/marker collection differs")
    for field in ("requested", "inventory", "ledger"):

        def canonical(entry):
            return key(
                {
                    k: v
                    for k, v in entry.items()
                    if k not in ("log_path", "log_line", "log_sha256", "table_block")
                }
            )

        a, b = (
            Counter(canonical(e) for e in report[field]) for report in (left, right)
        )
        differences[field] = dict(
            missing=[dict(entry=json.loads(k), count=n) for k, n in (a - b).items()],
            extra=[dict(entry=json.loads(k), count=n) for k, n in (b - a).items()],
        )
        if a != b:
            errors.append(f"{field} multisets differ")
    for field in (
        "pytest_skip",
        "xfail",
        "job_timeout",
        "claimed_without_result",
        "unstarted",
        "failed",
        "missing_result",
    ):
        if left["counts"].get(field) or right["counts"].get(field):
            errors.append(f"equivalence unresolved with {field} work")
    if bool(left.get("resource_time")) != bool(right.get("resource_time")):
        errors.append("elapsed timing scope differs")
    a, b = (
        (report.get("resource_time") or report["execution"]).get("wall_seconds")
        for report in (left, right)
    )
    return dict(
        schema=1,
        coverage_equal=not errors,
        audit_errors=errors,
        differences=differences,
        reference_seconds=a,
        candidate_seconds=b,
        elapsed_time_source=(
            "resource-time.json" if left.get("resource_time") else "execution"
        ),
        reference_over_candidate_ratio=a / b if a and b and not errors else None,
        statistical_claim=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.report_dir:
        report = audit(args.report_dir)
    elif args.reference and args.candidate and args.output:
        report = compare(
            *(json.loads(p.read_text()) for p in (args.reference, args.candidate))
        )
        write_json(args.output, report)
    else:
        parser.error("specify --report-dir or --reference/--candidate/--output")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("requested", "inventory", "ledger", "differences")
            },
            indent=2,
        )
    )
    return bool(report["audit_errors"])


if __name__ == "__main__":
    raise SystemExit(main())

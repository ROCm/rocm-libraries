# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Plan gated, independent shape jobs after pytest selects the original YAMLs."""

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def problem_signature(problem):
    return {
        field: list(value) if isinstance(value, tuple) else value
        for field in (
            "sizes",
            "stridesA",
            "stridesB",
            "stridesC",
            "stridesD",
            "stridesGate",
            "count",
        )
        if (value := getattr(problem, field, None)) is not None
    }


def split_yaml(source, nodeid, directory, ceiling):
    from tensilelite.SolutionStructs import ProblemSizes, ProblemType

    document = yaml.safe_load(source.read_text())
    sections = {
        "TestParameters",
        "GlobalParameters",
        "BenchmarkProblems",
        "LibraryLogic",
        "Backend",
    }
    if any(key not in sections and not key.startswith(".") for key in document):
        raise ValueError(f"{nodeid}: unsupported top-level config sections")
    backend = document.get("Backend", {}).get("Name")
    if backend not in (None, "Ductile"):
        raise ValueError(f"{nodeid}: unsupported backend {backend}")
    jobs, excluded = [], []
    for pi, problem in enumerate(document["BenchmarkProblems"]):
        problem_type = ProblemType(
            copy.deepcopy(problem[0]), printIndexAssignmentInfo=False
        )
        if problem_type["OperationType"] != "GEMM":
            raise ValueError(f"{nodeid}: only GEMM is supported")
        for gi, group in enumerate(problem[1:], 1):
            if any(
                group.get(key)
                for key in (
                    "BenchmarkForkParameters",
                    "JoinParameters",
                    "BenchmarkJoinParameters",
                )
            ):
                raise ValueError(
                    f"{nodeid}: intermediate benchmark stages cannot be split"
                )
            finals = group.get("BenchmarkFinalParameters") or []
            indices = [i for i, final in enumerate(finals) if "ProblemSizes" in final]
            if len(indices) != 1:
                raise ValueError(
                    f"{nodeid}: expected exactly one final ProblemSizes list"
                )
            fi = indices[0]
            sizes = finals[fi]["ProblemSizes"]
            parsed = ProblemSizes(problem_type, copy.deepcopy(sizes))
            if not parsed.problems:
                raise ValueError(f"{nodeid}: empty problem domain")
            grouped = bool(problem_type.get("GroupedGemm", False))
            batches = [parsed.problems] if grouped else [[p] for p in parsed.problems]
            for si, batch in enumerate(batches):
                label = f"p{pi}_g{gi}_s{si}"
                jobid = hashlib.sha256(f"{nodeid}/{label}".encode()).hexdigest()[:20]
                work = 0
                signatures = []
                for p in batch:
                    logical = p.sizes[: problem_type["TotalIndices"]]
                    if any(type(n) is not int or n < 0 for n in logical):
                        raise ValueError(f"{nodeid}: invalid logical sizes {logical}")
                    count = p.count if p.count is not None else 1
                    if type(count) is not int or count < 1:
                        raise ValueError(f"{nodeid}: invalid problem count {count}")
                    work += math.prod(logical) * count
                    signatures.append(problem_signature(p))
                record = dict(
                    id=jobid,
                    nodeid=f"{nodeid}/{label}",
                    source=str(source),
                    source_sha256=digest(source),
                    work=work,
                    grouped=grouped,
                    problem_type_index=pi,
                    group_index=gi - 1,
                    problems=signatures,
                )
                # Ductile's search and weights span its shape list. Splitting it
                # changes the search, so keep the original config as one task.
                if backend == "Ductile":
                    jobs.append(record)
                    continue
                if work > ceiling:
                    excluded.append(
                        dict(record, reason=f"logical work exceeds {ceiling}")
                    )
                    continue
                selected = copy.deepcopy(group)
                # ExactDict handles explicit strides differently from ExactList's
                # appended leading dimensions. Keep the appropriate representation.
                exacts = []
                for p in batch:
                    descriptor = problem_signature(p)
                    if set(descriptor) == {"sizes"}:
                        exact = descriptor["sizes"]
                    else:
                        descriptor["sizes"] = descriptor["sizes"][
                            : problem_type["TotalIndices"]
                        ]
                        exact = descriptor
                    exacts.append({"Exact": exact})
                for entry in sizes:
                    if "MinStride" in entry:
                        exacts.append({"MinStride": copy.deepcopy(entry["MinStride"])})
                selected["BenchmarkFinalParameters"][fi]["ProblemSizes"] = exacts
                shard = copy.deepcopy(document)
                shard["BenchmarkProblems"] = [[copy.deepcopy(problem[0]), selected]]
                destination = directory / f"{source.stem}__{jobid}.yaml"
                destination.write_text(yaml.safe_dump(shard, sort_keys=False))
                loaded = yaml.safe_load(destination.read_text())
                if loaded != shard:
                    raise ValueError(f"{nodeid}: YAML round trip changed the config")
                reparsed = ProblemSizes(problem_type, copy.deepcopy(exacts))
                if [problem_signature(p) for p in reparsed.problems] != signatures:
                    raise ValueError(
                        f"{nodeid}: normalized problem descriptors changed"
                    )
                if tuple(reparsed.minStrides) != tuple(parsed.minStrides):
                    raise ValueError(f"{nodeid}: minimum strides changed")
                jobs.append(
                    dict(record, yaml=str(destination), yaml_sha256=digest(destination))
                )
    if backend == "Ductile":
        jobid = hashlib.sha256(f"{nodeid}/whole-config".encode()).hexdigest()[:20]
        record = dict(
            id=jobid,
            nodeid=f"{nodeid}/whole-config",
            source=str(source),
            source_sha256=digest(source),
            work=sum(job["work"] for job in jobs),
            grouped=False,
            atomic_reason="Ductile search spans multiple shapes",
            problems=[p for job in jobs for p in job["problems"]],
        )
        if record["work"] > ceiling:
            return [], [
                dict(record, reason=f"indivisible config work exceeds {ceiling}")
            ]
        destination = directory / f"{source.stem}__{jobid}.yaml"
        destination.write_bytes(source.read_bytes())
        return [
            dict(record, yaml=str(destination), yaml_sha256=digest(destination))
        ], []
    return jobs, excluded


def pytest_addoption(parser):
    parser.addoption("--rocjitsu-shape-plan", type=Path)
    parser.addoption("--rocjitsu-max-work", type=int, default=4096**3)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_collection_modifyitems(config, items):
    result = yield
    destination = config.getoption("--rocjitsu-shape-plan")
    if destination is None:
        return result
    if not config.getoption("collectonly") or hasattr(config, "workerinput"):
        raise pytest.UsageError("shape planning requires serial --collect-only")
    ceiling = config.getoption("--rocjitsu-max-work")
    if ceiling < 1:
        raise pytest.UsageError("--rocjitsu-max-work must be positive")
    directory = destination.parent / "shape-yamls"
    directory.mkdir(parents=True, exist_ok=True)
    jobs, excluded = [], []
    sources = []
    for item in items:
        source = Path(item.callspec.params["config"])
        try:
            admitted, rejected = split_yaml(source, item.nodeid, directory, ceiling)
        except (ValueError, KeyError, TypeError) as error:
            raise pytest.UsageError(
                f"Cannot safely split {item.nodeid}: {error}"
            ) from error
        markers = []
        for mark in item.iter_markers():
            values = [*mark.args, *mark.kwargs.values()]
            if any(
                type(value) not in (str, int, float, bool, type(None))
                for value in values
            ):
                raise pytest.UsageError(
                    f"{item.nodeid}: cannot preserve marker {mark.name}"
                )
            markers.append(dict(name=mark.name, args=mark.args, kwargs=mark.kwargs))
        for job in admitted + rejected:
            job["markers"] = markers
        sources.append(
            dict(
                nodeid=item.nodeid,
                path=str(source),
                sha256=digest(source),
                markers=markers,
            )
        )
        jobs.extend(admitted)
        excluded.extend(rejected)
    jobs.sort(key=lambda job: (-job["work"], job["nodeid"]))
    if not jobs:
        raise pytest.UsageError("the selection and ceiling admitted no shape jobs")
    plan = dict(
        version=1, ceiling=ceiling, sources=sources, jobs=jobs, excluded=excluded
    )
    destination.write_text(json.dumps(plan, indent=2) + "\n")
    print(
        f"Shape queue: {len(sources)} YAMLs, {len(jobs)} jobs, {len(excluded)} excluded"
    )
    for job in excluded:
        print(f"SIZE GATE: {job['nodeid']} work={job['work']}")
    return result

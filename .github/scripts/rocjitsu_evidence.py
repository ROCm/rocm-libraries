# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Retain execution evidence from the original TensileLite build/run test.

The helper still executes in its own interpreter with its original arguments.
Metadata is copied before the test or queue deletes successful output. FD
capture keeps passing helper output available even when pytest uses xdist.
"""

import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import shutil
import time

import pytest

from rocjitsu_pytest import write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_name(path):
    return "tensilelite/Tests/" + path.as_posix().split("tensilelite/Tests/", 1)[-1]


def markers(item):
    return [
        dict(name=m.name, args=list(m.args), kwargs=m.kwargs)
        for m in item.iter_markers()
    ]


def describe(item):
    result = dict(nodeid=item.nodeid, markers=markers(item))
    source = getattr(item, "callspec", None)
    if source and "config" in source.params:
        path = Path(source.params["config"])
        result.update(source=source_name(path), source_sha256=digest(path))
    return result


def pytest_addoption(parser):
    parser.addoption("--rocjitsu-evidence-dir", type=Path)


def pytest_configure(config):
    directory = config.getoption("--rocjitsu-evidence-dir")
    if directory:
        if config.getoption("capture") != "fd":
            raise pytest.UsageError("RocJITsu evidence requires --capture=fd")
        directory.mkdir(parents=True, exist_ok=True)
        config.pluginmanager.register(
            Evidence(directory), "rocjitsu-execution-evidence"
        )


class Evidence:
    def __init__(self, directory):
        self.directory = directory
        self.items = {}

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, config, items):
        collected = [describe(item) for item in items]
        result = yield
        worker = getattr(config, "workerinput", {}).get("workerid", "serial")
        selected = [describe(item) for item in items]
        selected_ids = {item["nodeid"] for item in selected}
        write_json(
            self.directory / f"collection-{worker}.json",
            dict(
                schema=1,
                keyword=config.getoption("keyword"),
                markexpr=config.getoption("markexpr"),
                collected=collected,
                selected=selected,
                deselected=[i for i in collected if i["nodeid"] not in selected_ids],
            ),
        )
        return result

    def record(self, item):
        if item.nodeid not in self.items:
            name = hashlib.sha256(item.nodeid.encode()).hexdigest()[:24]
            directory = self.directory / name
            directory.mkdir()
            metadata = describe(item)
            jobpath = item.config.getoption("--rocjitsu-shape-job", default=None)
            if jobpath:
                job = json.loads(jobpath.read_text())
                metadata.update(
                    nodeid=job["nodeid"],
                    source=source_name(Path(job["source"])),
                    source_sha256=job["source_sha256"],
                    job=job,
                )
            metadata.update(schema=1, phases=[], reports={}, worker=os.getpid())
            self.items[item.nodeid] = directory, metadata
            write_json(directory / "item.json", metadata)
        return self.items[item.nodeid]

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_call(self, item):
        directory, metadata = self.record(item)
        namespace = item.obj.__globals__
        original = namespace.get("_call_helper_in_subprocess")
        parameters = [
            "module",
            "func",
            "config",
            "output_dir",
            "artifact_dir",
            "tensile_args",
        ]
        if (
            original is None
            or list(inspect.signature(original).parameters) != parameters
        ):
            raise RuntimeError(
                "TensileLite build/run helper changed; update evidence adapter"
            )
        metadata["test_sha256"] = digest(Path(inspect.getsourcefile(item.obj)))
        metadata["helper_sha256"] = hashlib.sha256(
            inspect.getsource(original).encode()
        ).hexdigest()

        @functools.wraps(original)
        def observed(*args, **kwargs):
            bound = inspect.signature(original).bind(*args, **kwargs).arguments
            pair = bound["module"], bound["func"]
            if pair not in (
                ("test_config_build", "_build"),
                ("test_config_run", "_run"),
            ):
                raise RuntimeError(f"unknown TensileLite helper phase: {pair}")
            phase = dict(
                name="build" if pair[1] == "_build" else "run",
                args=[str(v) for v in args[:-1]] + [bound["tensile_args"]],
                output_dir=bound["output_dir"],
                completed=False,
            )
            metadata["phases"].append(phase)
            write_json(directory / "item.json", metadata)
            start = time.monotonic()
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            try:
                result = original(*args, **kwargs)
                phase["completed"] = True
                return result
            finally:
                after = resource.getrusage(resource.RUSAGE_CHILDREN)
                phase.update(
                    wall_seconds=time.monotonic() - start,
                    user_seconds=after.ru_utime - before.ru_utime,
                    system_seconds=after.ru_stime - before.ru_stime,
                )
                copy_start = time.monotonic()
                phase["files"] = retain(
                    Path(bound["output_dir"]), directory / phase["name"]
                )
                phase["evidence_seconds"] = time.monotonic() - copy_start
                write_json(directory / "item.json", metadata)

        namespace["_call_helper_in_subprocess"] = observed
        try:
            return (yield)
        finally:
            namespace["_call_helper_in_subprocess"] = original

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(self, item, call):
        report = yield
        directory, metadata = self.record(item)
        metadata["reports"][report.when] = dict(
            outcome=report.outcome,
            seconds=report.duration,
            wasxfail=getattr(report, "wasxfail", None),
            detail=str(report.longrepr) if report.failed or report.skipped else None,
        )
        # Sections in later reports include earlier phases; save each only once.
        for name, content in report.sections:
            if name in (
                f"Captured stdout {report.when}",
                f"Captured stderr {report.when}",
            ):
                stream = "stdout" if "stdout" in name else "stderr"
                (directory / f"{report.when}-{stream}.log").write_text(content)
        write_json(directory / "item.json", metadata)
        return report


def retain(output, destination):
    destination.mkdir()
    files = []
    if output.is_dir():
        for path in sorted(output.rglob("*")):
            relative = path.relative_to(output)
            keep = (
                path.name == "ClientParameters.ini"
                or path.suffix == ".csv"
                or (
                    path.suffix == ".yaml"
                    and (
                        "Data" in relative.parts
                        or relative.parts[0] == "2_BenchmarkData"
                    )
                )
            )
            if not path.is_file() or not keep:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            files.append(
                dict(
                    path=str(relative),
                    sha256=digest(target),
                    bytes=target.stat().st_size,
                )
            )
    return files

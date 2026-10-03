# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check HIPBLASLT_JIT through hipblaslt-jit-heuristic-test on a real GPU.

Each route runs the test binary in fresh processes with its own JIT library,
temporary and cache directories under the output directory, and an empty
HIPBLASLT_TENSILE_LIBPATH. The build's JIT backend is the test backend, which
replays the --replay bundles.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import functools
import json
import os
from pathlib import Path
import re
import subprocess
import time

import msgpack

JIT_INDEX = 1 << 30
INVALID_VALUE = 3
IGNORED = "is ignored: hipBLASLt was built without HIPBLASLT_ENABLE_JIT"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class Runner:
    def __init__(self, executable, output, replay=None):
        self.executable = executable
        self.output = output
        empty = output / "empty-device-library"
        empty.mkdir()
        (output / "tmp").mkdir()
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("HIPBLASLT_JIT", "AMD_COMGR_"))
        }
        self.env.update(
            HIPBLASLT_TENSILE_LIBPATH=str(empty),
            HIPBLASLT_JIT_LIBRARY_PATH=str(output / "lib"),
            TMPDIR=str(output / "tmp"),
            XDG_CACHE_HOME=str(output / "xdg"),
        )
        if replay:
            self.env["HIPBLASLT_JIT_TEST_REPLAY"] = replay

    def __call__(self, name, args, drop=(), **overrides):
        env = {key: value for key, value in self.env.items() if key not in drop}
        env.update(overrides)
        command = [str(self.executable), *args]
        result = subprocess.run(
            command, env=env, text=True, capture_output=True, timeout=900
        )
        (self.output / f"{name}.command.json").write_text(
            json.dumps(dict(command=command, environment=overrides), indent=2)
        )
        (self.output / f"{name}.stdout").write_text(result.stdout)
        (self.output / f"{name}.stderr").write_text(result.stderr)
        require(
            result.returncode == 0,
            f"{name} exited with {result.returncode}; see {self.output}",
        )
        records = [
            json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
        ]
        return result.stderr, records


def reports(stderr, severity="(?:error|warning)"):
    return re.findall(r"^hipblaslt " + severity + r": JIT .*$", stderr, re.MULTILINE)


def entries(library):
    return list(library.rglob("TensileLibrary_JIT_*.dat"))


def allocated(library):
    """The next index the library's allocator hands out."""
    data = (library / "v1" / "allocator.dat").read_bytes()
    return msgpack.unpackb(data)["next"]


def queries(records, api):
    return [record for record in records if record["api"] == api]


def recording(path):
    """The environment in which generation records its request in path and fails."""
    return dict(HIPBLASLT_JIT_TEST_FAULT="record", HIPBLASLT_JIT_TEST_RECORD=str(path))


def check_jit_results(stderr, records, apis, requested):
    for api in apis:
        found = queries(records, api)
        require(found, f"No {api} query ran")
        for record in found:
            require(record["status"] == 0, f"{api} query failed: {record}")
            require(
                1 <= record["count"] <= requested,
                f"{api} query returned {record['count']} of {requested}",
            )
            require(
                all(index >= JIT_INDEX for index in record["indices"]),
                f"{api} query returned a pre-tuned solution: {record['indices']}",
            )
            if record["count"] < requested:
                require(
                    reports(stderr, "warning"), f"{api} shortfall was not reported"
                )
        label = {"c": "C", "cpp": "C++"}[api]
        for index in range(found[0]["count"]):
            require(
                f"{label} result {index} PASS" in stderr,
                f"{label} result {index} was not checked",
            )
    require(not reports(stderr, "error"), "A JIT error was reported")


def fallback(run, output, api):
    for requested in (1, 3):
        stderr, records = run(
            f"requested-{requested}",
            ["--api", api, "--requested", str(requested)],
            HIPBLASLT_JIT="1",
        )
        check_jit_results(stderr, records, (api,), requested)
    require(entries(output / "lib"), "JIT results were not published")
    print(f"PASS heuristic-fallback-{api}: JIT fills an empty pre-tuned library")


def forced(run, output):
    for name, drop in (
        ("empty-device-library", ()),
        ("default-device-library", ("HIPBLASLT_TENSILE_LIBPATH",)),
    ):
        stderr, records = run(
            name, ["--api", "both", "--requested", "2"], drop, HIPBLASLT_JIT="2"
        )
        check_jit_results(stderr, records, ("c", "cpp"), 2)
    print("PASS heuristic-forced: only JIT solutions")


def cache_hit(run, output):
    trap = output / "trap.txt"
    stderr, first = run("publish", ["--api", "c", "--requested", "1"], HIPBLASLT_JIT="2")
    check_jit_results(stderr, first, ("c",), 1)
    published = queries(first, "c")[0]["indices"]
    stderr, second = run(
        "reuse", ["--api", "both", "--requested", "1"], HIPBLASLT_JIT="2", **recording(trap)
    )
    check_jit_results(stderr, second, ("c", "cpp"), 1)
    for api in ("c", "cpp"):
        require(
            queries(second, api)[0]["indices"] == published,
            f"{api} query did not return the published solution",
        )
    require(not trap.exists(), f"The second process generated: {trap}")
    require(not reports(stderr), "The second process reported a JIT problem")
    stderr, third = run(
        "resolve",
        ["--api", "none", "--from-index", ",".join(map(str, published))],
        HIPBLASLT_JIT="0",
        **recording(trap),
    )
    (resolved,) = queries(third, "from-index")
    require(
        resolved["status"] == 0 and resolved["indices"] == published,
        f"HIPBLASLT_JIT=0 did not resolve {published}: {resolved}",
    )
    require("Index result 0 PASS" in stderr, "The resolved solution was not checked")
    require(not trap.exists(), f"The third process generated: {trap}")
    require(not reports(stderr), "The third process reported a JIT problem")
    print(
        "PASS heuristic-cache-hit: a second process reuses the library without generating;"
        " HIPBLASLT_JIT=0 resolves the published index"
    )


def distinct(run, output):
    stderr, first = run(
        "publish", ["--api", "c", "--requested", "1", "--no-run"], HIPBLASLT_JIT="2"
    )
    (published,) = queries(first, "c")
    require(published["count"] == 1, f"The first query returned {published}")
    requested = 3
    stderr, records = run(
        "fill", ["--api", "both", "--requested", str(requested)], HIPBLASLT_JIT="2"
    )
    check_jit_results(stderr, records, ("c", "cpp"), requested)
    for api in ("c", "cpp"):
        (record,) = queries(records, api)
        require(
            record["count"] == requested,
            f"{api} returned {record['count']} of {requested}",
        )
        require(
            record["indices"][0] == published["indices"][0],
            f"{api} did not return the cached solution first: {record['indices']}",
        )
        require(
            len(set(record["indices"])) == len(set(record["kernels"])) == requested,
            f"{api} returned a solution twice: {record['kernels']}",
        )
    require(not reports(stderr), "A JIT problem was reported")
    require(
        len(entries(output / "lib")) == requested,
        f"The library does not hold {requested} solutions",
    )
    print(
        "PASS heuristic-distinct: a request for 3 with 1 cached returns 3 distinct kernels"
    )


def unsupported(run, output):
    # Origami has no ranking for K = 0, so the backend rejects the problem.
    args = ["--api", "both", "--k", "0", "--handles", "2", "--queries", "2", "--no-run"]
    for mode, status in (("1", INVALID_VALUE), ("2", 0)):
        name = f"mode-{mode}"
        library = output / f"lib-{name}"
        stderr, records = run(
            name, args, HIPBLASLT_JIT=mode, HIPBLASLT_JIT_LIBRARY_PATH=str(library)
        )
        lines = reports(stderr)
        require(
            len(lines) == 1
            and lines[0].startswith("hipblaslt error: JIT predict failed")
            and "No Origami ranking" in lines[0],
            f"{name}: expected one 'predict failed' error naming the reason, got {lines}",
        )
        require(
            len(records) == 8
            and all(
                record["status"] == status and record["count"] == 0
                for record in records
            ),
            f"{name}: expected 8 empty queries with status {status}: {records}",
        )
        require(not entries(library), f"{name} published a solution")
    print("PASS heuristic-unsupported: one visible error naming the reason, no results")


def concurrent(run, output):
    processes, threads, requested = 4, 4, 2
    for mode in ("1", "2"):
        library = output / f"lib-mode-{mode}"
        barrier = output / f"barrier-mode-{mode}"
        barrier.mkdir()
        args = ["--api", "both", "--requested", str(requested)]
        args += ["--threads", str(threads), "--barrier", str(barrier)]
        with ThreadPoolExecutor(processes) as pool:
            started = [
                pool.submit(
                    run,
                    f"mode-{mode}-process-{process}",
                    args,
                    HIPBLASLT_JIT=mode,
                    HIPBLASLT_JIT_LIBRARY_PATH=str(library),
                )
                for process in range(processes)
            ]
            deadline = time.monotonic() + 300
            while (
                len(list(barrier.glob("ready-*"))) < processes * threads
                and not any(future.done() for future in started)
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            ready = len(list(barrier.glob("ready-*")))
            (barrier / "go").touch()
            outputs = [future.result() for future in started]
        require(
            ready == processes * threads,
            f"Mode {mode}: only {ready} threads reached the barrier",
        )
        results = set()
        for stderr, records in outputs:
            check_jit_results(stderr, records, ("c", "cpp"), requested)
            require(not reports(stderr), f"Mode {mode}: a JIT problem was reported")
            require(
                len(records) == 2 * threads,
                f"Mode {mode}: expected {2 * threads} queries per process",
            )
            for record in records:
                results.add((tuple(record["indices"]), tuple(record["kernels"])))
        require(
            len(results) == 1,
            f"Mode {mode}: queries returned different results: {sorted(results)}",
        )
        ((indices, kernels),) = results
        require(
            len(set(indices)) == len(set(kernels)) == requested,
            f"Mode {mode}: a solution was returned twice: {kernels}",
        )
        require(
            len(entries(library)) == requested
            and allocated(library) == JIT_INDEX + requested
            and set(indices) == set(range(JIT_INDEX, JIT_INDEX + requested)),
            f"Mode {mode}: the library holds duplicate or stray entries",
        )
    print(
        f"PASS heuristic-concurrent: {processes} processes x {threads} threads agree"
        " and publish each solution once"
    )


def report(run, output):
    # The configure failure names the variable to set.
    causes = (
        ("configure", dict(HIPBLASLT_JIT_TEST_FAULT="unknown"), "HIPBLASLT_JIT_TEST_FAULT"),
        ("generate", dict(HIPBLASLT_JIT_TEST_FAULT="generate"), None),
    )
    for mode, status in (("1", INVALID_VALUE), ("2", 0)):
        for cause, overrides, variable in causes:
            phrase = cause + " failed"
            name = f"mode-{mode}-{cause}"
            scratch = output / name
            scratch.mkdir()
            stderr, records = run(
                name,
                ["--api", "both", "--handles", "2", "--queries", "2", "--no-run"],
                HIPBLASLT_JIT=mode,
                TMPDIR=str(scratch),
                **overrides,
            )
            lines = reports(stderr)
            require(
                len(lines) == 1 and lines[0].startswith("hipblaslt error: JIT " + phrase),
                f"{name}: expected one '{phrase}' error, got {lines}",
            )
            require(
                all(
                    record["status"] == status and record["count"] == 0
                    for record in records
                )
                and len(records) == 8,
                f"{name}: expected 8 empty queries with status {status}: {records}",
            )
            if variable:
                require(variable in lines[0], "The variable to set is not named")
            else:
                (log,) = re.findall(r"see (\S+\.log)", lines[0])
                require(
                    Path(log).is_file() and Path(log).is_relative_to(scratch),
                    f"The generator log was not kept: {log}",
                )
    print("PASS heuristic-report: one visible error per cause, empty results")


def jit_off(run, output):
    args = ["--api", "both", "--handles", "2", "--queries", "3", "--no-run"]
    stderr, ignored = run("jit-1", args, HIPBLASLT_JIT="1")
    require(
        stderr.count("hipblaslt warning: HIPBLASLT_JIT=1 " + IGNORED) == 1,
        "HIPBLASLT_JIT=1 was not reported once",
    )
    unset_stderr, unset = run("unset", args)
    require(IGNORED not in unset_stderr, "A warning was printed without HIPBLASLT_JIT")
    require(ignored == unset, "HIPBLASLT_JIT changed the results of a build without JIT")
    require(not (output / "lib").exists(), "A build without JIT created a JIT library")
    print("PASS jit-off: HIPBLASLT_JIT ignored with one warning")


ROUTES = {
    "fallback-c": functools.partial(fallback, api="c"),
    "fallback-cpp": functools.partial(fallback, api="cpp"),
    "forced": forced,
    "cache-hit": cache_hit,
    "distinct": distinct,
    "unsupported": unsupported,
    "concurrent": concurrent,
    "report": report,
    "jit-off": jit_off,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("route", choices=ROUTES)
    parser.add_argument("fresh_output", type=Path)
    parser.add_argument(
        "--replay", action="append", type=Path, help="a bundle for the test backend to replay"
    )
    args = parser.parse_args()
    if (args.route != "jit-off") != bool(args.replay):
        parser.error("--replay is required for every route but jit-off")
    args.fresh_output.mkdir(parents=True, exist_ok=False)
    output = args.fresh_output.resolve()
    replay = args.replay and os.pathsep.join(str(path.resolve(strict=True)) for path in args.replay)
    runner = Runner(args.executable.resolve(strict=True), output, replay)
    ROUTES[args.route](runner, output)


if __name__ == "__main__":
    main()

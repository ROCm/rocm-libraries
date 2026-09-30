# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check HIPBLASLT_JIT through hipblaslt-jit-heuristic-test on a real GPU.

Each route runs the test binary in fresh processes with its own JIT library,
temporary and cache directories under the output directory, and an empty
HIPBLASLT_TENSILE_LIBPATH unless the route needs the build's device library.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import functools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import msgpack

JIT_INDEX = 1 << 30
INVALID_VALUE = 3
IGNORED = "is ignored: hipBLASLt was built without HIPBLASLT_ENABLE_JIT"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class Runner:
    def __init__(self, executable, output):
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
    require(
        not (output / "xdg/comgr").exists(), "JIT processes used comgr's on-disk cache"
    )
    print("PASS heuristic-forced: only JIT solutions, comgr cache off")


def cache_hit(run, output):
    # The tool paths are part of the cache key, so both processes use this
    # wrapper; the second one fails if it generates.
    python = output / "python"
    python.write_text(
        "#!/bin/sh\n"
        'if [ -n "$HIPBLASLT_JIT_TEST_TRAP" ]; then\n'
        '    echo "$@" > "$HIPBLASLT_JIT_TEST_TRAP"\n'
        "    exit 1\n"
        "fi\n"
        f'exec "{sys.executable}" "$@"\n'
    )
    python.chmod(0o700)
    trap = output / "trap.txt"
    stderr, first = run(
        "publish",
        ["--api", "c", "--requested", "1"],
        HIPBLASLT_JIT="2",
        HIPBLASLT_JIT_PYTHON=str(python),
    )
    check_jit_results(stderr, first, ("c",), 1)
    published = queries(first, "c")[0]["indices"]
    stderr, second = run(
        "reuse",
        ["--api", "both", "--requested", "1", "--null-algo"],
        HIPBLASLT_JIT="2",
        HIPBLASLT_JIT_PYTHON=str(python),
        HIPBLASLT_JIT_TEST_TRAP=str(trap),
    )
    check_jit_results(stderr, second, ("c", "cpp"), 1)
    for api in ("c", "cpp"):
        require(
            queries(second, api)[0]["indices"] == published,
            f"{api} query did not return the published solution",
        )
    require(queries(second, "null-algo")[0]["status"] == 0, "Null algorithm failed")
    require(not trap.exists(), f"The second process generated: {trap}")
    require(not reports(stderr), "The second process reported a JIT problem")
    stderr, third = run(
        "resolve",
        ["--api", "none", "--from-index", ",".join(map(str, published))],
        HIPBLASLT_JIT="0",
        HIPBLASLT_JIT_PYTHON=str(python),
        HIPBLASLT_JIT_TEST_TRAP=str(trap),
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


def null_algo(run, output):
    for mode in ("1", "2"):
        library = output / f"lib-mode-{mode}"
        stderr, records = run(
            f"mode-{mode}",
            ["--api", "none", "--null-algo"],
            HIPBLASLT_JIT=mode,
            HIPBLASLT_JIT_LIBRARY_PATH=str(library),
        )
        require(queries(records, "null-algo")[0]["status"] == 0, f"Mode {mode} failed")
        require(
            "hipblasLtMatmul without an algorithm PASS" in stderr,
            f"Mode {mode} result was not checked",
        )
        require(entries(library), f"Mode {mode} did not publish its solution")
    library = output / "lib-mode-0"
    stderr, records = run(
        "mode-0",
        ["--api", "none", "--null-algo", "--no-run"],
        HIPBLASLT_JIT_LIBRARY_PATH=str(library),
    )
    require(
        queries(records, "null-algo")[0]["status"] != 0,
        "Mode 0 found a solution without a device library",
    )
    require(not library.exists() and not reports(stderr), "Mode 0 used JIT")
    print("PASS heuristic-null-algo: modes 1 and 2 select a JIT solution, mode 0 does not")


def report(run, output):
    for mode, status in (("1", INVALID_VALUE), ("2", 0)):
        for cause, python, phrase in (
            ("configure", "/nonexistent", "configure failed"),
            ("generate", "/bin/false", "generate failed"),
        ):
            name = f"mode-{mode}-{cause}"
            scratch = output / name
            scratch.mkdir()
            stderr, records = run(
                name,
                ["--api", "both", "--handles", "2", "--queries", "2", "--no-run"],
                HIPBLASLT_JIT=mode,
                HIPBLASLT_JIT_PYTHON=python,
                TMPDIR=str(scratch),
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
            if cause == "configure":
                require("HIPBLASLT_JIT_PYTHON" in lines[0], "The variable to set is not named")
            else:
                (log,) = re.findall(r"see (\S+\.log)", lines[0])
                require(
                    Path(log).is_file() and Path(log).is_relative_to(scratch),
                    f"The generator log was not kept: {log}",
                )
    print("PASS heuristic-report: one visible error per cause, empty results")


def partial_fill(run, output):
    drop = ("HIPBLASLT_TENSILE_LIBPATH",)
    probe = 4096
    stderr, records = run(
        "pre-tuned", ["--api", "both", "--requested", str(probe), "--no-run"], drop
    )
    pretuned = {api: queries(records, api)[0] for api in ("c", "cpp")}
    if any(record["status"] != 0 or not record["count"] for record in pretuned.values()):
        print("SKIP heuristic-partial-fill: the build has no device library for the problem")
        return
    if any(record["count"] >= probe for record in pretuned.values()):
        print(f"SKIP heuristic-partial-fill: the device library fills {probe} requests")
        return
    for api, record in pretuned.items():
        count = record["count"]
        stderr, records = run(
            f"fill-{api}",
            ["--api", api, "--requested", str(count + 1), "--no-run"],
            drop,
            HIPBLASLT_JIT="1",
        )
        (filled,) = queries(records, api)
        require(filled["status"] == 0, f"{api} fill failed: {filled}")
        # The pre-tuned library can order tied solutions differently in each process.
        require(
            sorted(filled["indices"][:count]) == sorted(record["indices"]),
            f"{api} fill changed the pre-tuned results",
        )
        require(
            all(index >= JIT_INDEX for index in filled["indices"][count:]),
            f"{api} fill added a pre-tuned solution",
        )
        require(
            filled["count"] == count + 1,
            f"{api} fill returned {filled['count']} of {count + 1}",
        )
        require(
            filled["kernels"][count] not in filled["kernels"][:count],
            f"{api} fill repeated a pre-tuned kernel: {filled['kernels'][count]}",
        )
    print("PASS heuristic-partial-fill: a new JIT kernel follows the pre-tuned results")


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
    "null-algo": null_algo,
    "report": report,
    "partial-fill": partial_fill,
    "jit-off": jit_off,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("route", choices=ROUTES)
    parser.add_argument("fresh_output", type=Path)
    args = parser.parse_args()
    args.fresh_output.mkdir(parents=True, exist_ok=False)
    output = args.fresh_output.resolve()
    ROUTES[args.route](Runner(args.executable.resolve(strict=True), output), output)


if __name__ == "__main__":
    main()

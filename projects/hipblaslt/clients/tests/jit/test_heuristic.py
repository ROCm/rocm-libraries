# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check HIPBLASLT_JIT through hipblaslt-jit-heuristic-test on a real GPU.

Each route runs the test binary in fresh processes with its own JIT library,
temporary and cache directories under the output directory, and an empty
HIPBLASLT_TENSILE_LIBPATH unless the route needs the build's device library.
With --backend test the build's JIT backend is the test backend, which replays
the --replay bundles; otherwise it is the TensileLite generator.
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
DEFAULT_SIZE = (256, 128, 512)
# Sizes the FP16 NN Equality logic tunes; the default size has no Equality hit.
EQUALITY_SIZES = ((1024, 4096, 20), (2048, 128, 16), (864, 512, 432), (128, 5120, 1024))
# "tensilelite" or "test"; main sets it.
BACKEND = "tensilelite"
# Routes that need the generator child process.
TENSILELITE_ROUTES = ("debug-killed-child",)


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


def size_args(size):
    m, n, k = size
    return ["--m", str(m), "--n", str(n), "--k", str(k)]


def generation_trap(output):
    """Environments for a route whose later processes must not generate.

    Returns (generating, trapped): generating is the environment that generates,
    and trapped(path) the one in which generation records its request in path
    and fails. Both keep the backend version, which keys the JIT library, so
    every process of the route that shares a library uses one of them. The
    TensileLite version covers the tool paths, so both run a Python wrapper.
    """
    if BACKEND == "test":
        return {}, lambda path: dict(
            HIPBLASLT_JIT_TEST_FAULT="record", HIPBLASLT_JIT_TEST_RECORD=str(path)
        )
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
    generating = dict(HIPBLASLT_JIT_PYTHON=str(python))
    return generating, lambda path: dict(generating, HIPBLASLT_JIT_TEST_TRAP=str(path))


def picked(stderr):
    """The solution index of the last hipblasLtMatmul, from its bench log line."""
    found = re.findall(r"--solution_index (\d+)", stderr)
    require(found, "No hipblasLtMatmul was logged")
    return int(found[-1])


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
    generating, trapped = generation_trap(output)
    trap = output / "trap.txt"
    stderr, first = run(
        "publish",
        ["--api", "c", "--requested", "1"],
        HIPBLASLT_JIT="2",
        **generating,
    )
    check_jit_results(stderr, first, ("c",), 1)
    published = queries(first, "c")[0]["indices"]
    stderr, second = run(
        "reuse",
        ["--api", "both", "--requested", "1", "--null-algo"],
        HIPBLASLT_JIT="2",
        **trapped(trap),
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
        **trapped(trap),
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


def capture(run, output):
    unpublished = (256, 128, 576)
    generating, trapping = generation_trap(output)
    trap = output / "trap.txt"
    trapped = trapping(trap)
    _, records = run(
        "publish",
        ["--api", "c", "--requested", "1", "--no-run"],
        HIPBLASLT_JIT="2",
        **generating,
    )
    (published,) = queries(records, "c")
    require(published["count"] == 1, f"The JIT library was not seeded: {published}")
    library = output / "lib"
    held = (len(entries(library)), allocated(library))
    failed = output / "failing-generation.txt"
    _, records = run(
        "failing-generation",
        ["--api", "none", "--null-algo"] + size_args(unpublished),
        HIPBLASLT_JIT="2",
        **trapping(failed),
    )
    missing = queries(records, "null-algo")[0]["status"]
    require(
        missing != 0 and failed.exists(), "Generation outside a capture did not fail"
    )

    def captured(name, capture_mode, size, drop=(), **overrides):
        args = ["--api", "none", "--null-algo", "--capture", capture_mode]
        stderr, records = run(
            name, args + size_args(size), drop, **trapped, **overrides
        )
        (record,) = queries(records, "null-algo")
        require(
            record["capture"] == "active" and record["ended"] == 0,
            f"{name}: the capture did not stay valid: {record}",
        )
        require(not trap.exists(), f"{name}: JIT generated during the capture")
        return stderr, record

    def replayed(name, stderr, record):
        require(
            record["status"] == 0 and record["nodes"] > 0,
            f"{name}: no solution was captured: {record}",
        )
        require(
            all(f"replay {replay} PASS" in stderr for replay in (0, 1)),
            f"{name}: the graph replays were not checked",
        )
        require(not reports(stderr), f"{name}: a JIT problem was reported")

    for mode in ("1", "2"):
        for capture_mode in ("global", "thread-local", "relaxed"):
            name = f"mode-{mode}-{capture_mode}"
            stderr, record = captured(
                name, capture_mode, DEFAULT_SIZE, HIPBLASLT_JIT=mode
            )
            replayed(name, stderr, record)
            name += "-unpublished"
            stderr, record = captured(
                name, capture_mode, unpublished, HIPBLASLT_JIT=mode
            )
            require(
                record["status"] == missing and record["nodes"] == 0,
                f"{name}: expected status {missing} and an empty graph: {record}",
            )
            lines = reports(stderr)
            require(
                len(lines) == 1
                and lines[0].startswith(
                    "hipblaslt error: JIT generation skipped during stream capture for"
                    " GEMM M=256 N=128 K=576 "
                ),
                f"{name}: expected one 'generation skipped' error, got {lines}",
            )
    require(
        (len(entries(library)), allocated(library)) == held,
        "A solution was published during a capture",
    )

    drop = ("HIPBLASLT_TENSILE_LIBPATH",)
    _, records = run(
        "mode-0-device-library",
        ["--api", "none", "--null-algo"] + size_args(unpublished),
        drop,
    )
    pretuned = queries(records, "null-algo")[0]["status"] == 0
    for capture_mode in ("global", "thread-local", "relaxed") if pretuned else ():
        name = f"mode-1-{capture_mode}-device-library"
        stderr, record = captured(
            name, capture_mode, unpublished, drop, HIPBLASLT_JIT="1"
        )
        replayed(name, stderr, record)
    print(
        "PASS heuristic-capture: during stream capture, hipblasLtMatmul without an"
        " algorithm runs published JIT solutions and reports instead of generating"
        + ("; a pre-tuned solution fills in mode 1" if pretuned else "")
    )


def capture_query(run, output):
    for mode in ("1", "2"):
        for capture_mode in ("global", "thread-local", "relaxed"):
            for api, label in (("c", "C"), ("cpp", "C++")):
                name = f"mode-{mode}-{capture_mode}-{api}"
                library = output / f"lib-{name}"
                stderr, records = run(
                    name,
                    ["--api", api, "--requested", "1", "--capture", capture_mode],
                    HIPBLASLT_JIT=mode,
                    HIPBLASLT_JIT_LIBRARY_PATH=str(library),
                )
                (record,) = queries(records, api)
                require(
                    record["status"] == 0
                    and record["count"] == 1
                    and record["indices"][0] >= JIT_INDEX,
                    f"{name}: the query did not return a JIT solution: {record}",
                )
                require(
                    len(entries(library)) == 1,
                    f"{name}: the query did not generate and publish its solution",
                )
                require(
                    record["capture"] == "active"
                    and record["ended"] == 0
                    and record["launch"] == 0
                    and record["nodes"] > 0,
                    f"{name}: the capture did not stay valid: {record}",
                )
                require(
                    all(
                        f"Captured {label} result 0, replay {replay} PASS" in stderr
                        for replay in (0, 1)
                    ),
                    f"{name}: the graph replays were not checked",
                )
                require(not reports(stderr), f"{name}: a JIT problem was reported")
    print(
        "PASS heuristic-capture-query: during stream capture, both heuristic queries"
        " generate and publish, the capture stays valid, and the captured launch"
        " replays"
    )


def report(run, output):
    # The configure failure names the variable to set.
    if BACKEND == "test":
        causes = (
            ("configure", dict(HIPBLASLT_JIT_TEST_FAULT="unknown"), "HIPBLASLT_JIT_TEST_FAULT"),
            ("generate", dict(HIPBLASLT_JIT_TEST_FAULT="generate"), None),
        )
    else:
        causes = (
            ("configure", dict(HIPBLASLT_JIT_PYTHON="/nonexistent"), "HIPBLASLT_JIT_PYTHON"),
            ("generate", dict(HIPBLASLT_JIT_PYTHON="/bin/false"), None),
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
    generating, trapped = generation_trap(output)
    trap = output / "trap.txt"
    stderr, records = run(
        "publish",
        ["--api", "c", "--requested", "1", "--no-run"],
        HIPBLASLT_JIT="2",
        **generating,
    )
    (published,) = queries(records, "c")
    require(published["count"] == 1, f"The JIT library was not seeded: {published}")
    (jit_index,), (jit_kernel,) = published["indices"], published["kernels"]
    for api, record in pretuned.items():
        stderr, records = run(
            f"fill-{api}",
            ["--api", api, "--requested", str(record["count"] + 1), "--no-run"],
            drop,
            HIPBLASLT_JIT="1",
            **trapped(trap),
        )
        (filled,) = queries(records, api)
        require(filled["status"] == 0, f"{api} fill failed: {filled}")
        require(trap.exists(), f"{api} fill did not try to generate what was missing")
        trap.unlink()
        require(
            reports(stderr, "warning") and not reports(stderr, "error"),
            f"{api} shortfall was not reported as a warning",
        )
        require(
            filled["indices"].count(jit_index) == 1
            and filled["kernels"].count(jit_kernel) == 1,
            f"{api} fill did not return the JIT solution once",
        )
        rest = [index for index in filled["indices"] if index != jit_index]
        others = [
            index
            for index, kernel in zip(record["indices"], record["kernels"])
            if kernel != jit_kernel
        ]
        # The pre-tuned library can order tied solutions differently in each process.
        require(
            sorted(rest) == sorted(others),
            f"{api} fill did not complete with the other pre-tuned solutions",
        )
    print(
        "PASS heuristic-partial-fill: the pre-tuned results complete what JIT leaves,"
        " without repeating its kernel"
    )


def tuning_override(run, output):
    stderr, records = run(
        "publish", ["--api", "cpp", "--requested", "2", "--no-run"], HIPBLASLT_JIT="2"
    )
    (published,) = queries(records, "cpp")
    require(published["count"] == 2, f"The JIT library was not seeded: {published}")
    stderr, records = run("revision", ["--api", "none", "--git-revision"])
    (revision,) = queries(records, "revision")
    m, n, k = DEFAULT_SIZE
    for first, second in (published["indices"], published["indices"][::-1]):
        # The C query ignores the file unless its first line names this
        # library's revision.
        tuning = output / f"override-{first}.csv"
        tuning.write_text(
            f"Git Version: {revision['revision']}\n"
            "transA,transB,batch_count,m,n,k,a_type,b_type,c_type,compute_type,"
            "solution_index\n"
            f"N,N,1,{m},{n},{k},f16_r,f16_r,f16_r,f32_r,{first}\n"
        )
        stderr, records = run(
            f"override-{first}",
            ["--api", "both", "--requested", "3"],
            HIPBLASLT_JIT="1",
            HIPBLASLT_TUNING_OVERRIDE_FILE=str(tuning),
        )
        for api in ("c", "cpp"):
            (record,) = queries(records, api)
            require(record["status"] == 0, f"{api} query failed: {record}")
            require(
                record["indices"][:2] == [first, second],
                f"{api} query did not return the override, then the other JIT solution:"
                f" {record['indices']}",
            )
            require(
                len(set(record["kernels"])) == len(record["kernels"]),
                f"{api} query repeated a kernel: {record['indices']}",
            )
        require(not reports(stderr, "error"), "A JIT error was reported")
    print(
        "PASS heuristic-override: a tuning override that names a JIT solution comes"
        " first, and JIT does not return its kernel again"
    )


def provider_order(run, output):
    drop = ("HIPBLASLT_TENSILE_LIBPATH",)

    def baseline(size):
        name = "mode-0-" + "x".join(map(str, size))
        args = ["--api", "both", "--requested", "8", "--no-run", "--tuned"]
        _, records = run(name, args + size_args(size), drop)
        (tuned,) = queries(records, "tuned")
        found = {api: queries(records, api)[0] for api in ("c", "cpp")}
        complete = all(
            record["status"] == 0 and record["count"] == 8 for record in found.values()
        )
        return tuned["tuned"] == 1, complete, found

    equality = None
    for size in EQUALITY_SIZES:
        tuned, complete, base_equality = baseline(size)
        if tuned and complete:
            equality = size
            break
    tuned, complete, base_other = baseline(DEFAULT_SIZE)
    if equality is None or tuned or not complete:
        print(
            "SKIP heuristic-provider-order: the device library lacks an Equality size"
            " or a size without one"
        )
        return
    generating, trapping = generation_trap(output)
    trap = output / "trap.txt"
    jit = dict(HIPBLASLT_JIT="1", **generating)
    trapped = dict(HIPBLASLT_JIT="1", **trapping(trap))
    library = output / "lib"
    cases = (
        ("equality", size_args(equality), 3, base_equality),
        ("other", size_args(DEFAULT_SIZE), 2, base_other),
    )

    stderr, records = run(
        "equality-fills",
        ["--api", "both", "--requested", "1", "--no-run", "--null-algo"]
        + size_args(equality),
        drop,
        HIPBLASLT_LOG_MASK="32",
        **trapped,
    )
    for api in ("c", "cpp"):
        (record,) = queries(records, api)
        require(
            record["indices"] == base_equality[api]["indices"][:1],
            f"{api} did not return the mode 0 result: {record['indices']}",
        )
    require(queries(records, "null-algo")[0]["status"] == 0, "Null algorithm failed")
    require(
        picked(stderr) == base_equality["c"]["indices"][0],
        "The null algorithm did not pick the Equality solution",
    )
    require(
        not trap.exists() and not entries(library) and not reports(stderr),
        "JIT was consulted although the Equality results fill the request",
    )

    first = {}
    for name, args, requested, base in cases:
        stderr, records = run(
            f"publish-{name}",
            ["--api", "both", "--requested", str(requested)] + args,
            drop,
            **jit,
        )
        require(not reports(stderr), f"{name}: a JIT problem was reported")
        first[name] = {api: queries(records, api)[0] for api in ("c", "cpp")}
        for api, record in first[name].items():
            label = {"c": "C", "cpp": "C++"}[api]
            require(
                record["status"] == 0 and record["count"] == requested,
                f"{name}: {api} returned {record}",
            )
            require(
                all(f"{label} result {i} PASS" in stderr for i in range(requested)),
                f"{name}: {label} results were not checked",
            )
            indices = record["indices"]
            pretuned = sum(index < JIT_INDEX for index in indices)
            require(
                all(index >= JIT_INDEX for index in indices[pretuned:])
                and indices[:pretuned] == base[api]["indices"][:pretuned]
                and (pretuned > 0) == (name == "equality")
                and pretuned < requested,
                f"{name}: {api} did not return the Equality results, then JIT: {indices}",
            )
        require(
            first[name]["c"]["indices"] == first[name]["cpp"]["indices"],
            f"{name}: the C and C++ queries disagree",
        )

    for name, args, requested, base in cases:
        stderr, records = run(
            f"reuse-{name}",
            ["--api", "both", "--requested", str(requested), "--no-run", "--null-algo"]
            + args,
            drop,
            HIPBLASLT_LOG_MASK="32",
            **trapped,
        )
        for api in ("c", "cpp"):
            (record,) = queries(records, api)
            require(
                record["indices"] == first[name][api]["indices"]
                and record["kernels"] == first[name][api]["kernels"],
                f"{name}: {api} differs in a second process: {record['indices']}",
            )
        require(queries(records, "null-algo")[0]["status"] == 0, "Null algorithm failed")
        choice = picked(stderr)
        require(
            choice == first[name]["c"]["indices"][0]
            and (choice >= JIT_INDEX) == (name == "other"),
            f"{name}: the null algorithm picked {choice}",
        )
        require(
            not trap.exists() and not reports(stderr),
            f"{name}: the second process generated or reported a JIT problem",
        )

    for name, args, requested, base in cases:
        stderr, records = run(
            f"short-{name}",
            ["--api", "both", "--requested", str(requested + 2), "--no-run"] + args,
            drop,
            **trapped,
        )
        require(trap.exists(), f"{name}: JIT did not try to generate what was missing")
        trap.unlink()
        require(
            reports(stderr, "warning") and not reports(stderr, "error"),
            f"{name}: the shortfall was not reported as a warning",
        )
        for api in ("c", "cpp"):
            (record,) = queries(records, api)
            known = first[name][api]
            kernels = set(known["kernels"])
            others = [
                index
                for index, kernel in zip(base[api]["indices"], base[api]["kernels"])
                if index not in known["indices"] and kernel not in kernels
            ]
            expected = known["indices"] + others[:2]
            require(
                record["indices"] == expected,
                f"{name}: {api} returned {record['indices']}, expected {expected}",
            )
    print(
        "PASS heuristic-provider-order: Equality results, then JIT, then the other"
        " providers; the null algorithm picks the same way"
    )


DEBUG_PREFIX = "hipblaslt jit-debug "
DEBUG_KEYS = ("v", "cat", "ev", "pid", "tid", "t_ms", "q")


def debug_lines(text):
    """The HIPBLASLT_JIT_DEBUG lines of text, parsed and checked for the common keys."""
    lines = []
    for line in text.splitlines():
        if not line.startswith(DEBUG_PREFIX):
            continue
        event = json.loads(line[len(DEBUG_PREFIX) :])
        require(
            list(event)[: len(DEBUG_KEYS)] == list(DEBUG_KEYS) and event["v"] == 1,
            f"Malformed debug line: {line}",
        )
        lines.append(event)
    return lines


def events(lines, name):
    return [line for line in lines if line["ev"] == name]


def debug_warnings(stderr):
    return re.findall(r"^hipblaslt warning: HIPBLASLT_JIT_DEBUG.*$", stderr, re.MULTILINE)


def results(records):
    return [(r["api"], r["indices"], r["kernels"]) for r in records if "indices" in r]


def check_sources(query):
    """The query's from counts add up to what it returned."""
    found = query.get("from", {})
    if "equality" in found:
        require(
            found["equality"] + found["jit"] + found["others"] == found["best"],
            f"from does not split best: {query}",
        )
        total = found["best"]
    else:
        total = found.get("best", 0) + found.get("jit", 0)
    total += found.get("all", 0) + found.get("override", 0)
    require(total == query["returned"], f"from does not add up to returned: {query}")


def argv_python(output):
    """A Python wrapper that appends its arguments to $HIPBLASLT_JIT_TEST_ARGV."""
    python = output / "argv-python"
    python.write_text(
        "#!/bin/sh\n"
        'echo "$@" >> "$HIPBLASLT_JIT_TEST_ARGV"\n'
        f'exec "{sys.executable}" "$@"\n'
    )
    python.chmod(0o700)
    return python


def debug_timing(run, output):
    args = ["--api", "both", "--requested", "3"]
    timing = dict(HIPBLASLT_JIT="1", HIPBLASLT_JIT_DEBUG="timing")
    stderr, records = run("first", args, **timing)
    check_jit_results(stderr, records, ("c", "cpp"), 3)
    lines = debug_lines(stderr)
    require(all(line["cat"] == "timing" for line in lines), "A progress line was written")
    (process,) = events(lines, "process")
    require(
        process["mode"] == 1 and process["categories"] == "timing"
        and process["destination"] == "stderr",
        f"Wrong process line: {process}",
    )
    (setup,) = events(lines, "setup")
    require(setup["status"] == "ok" and setup["ns"]["total"] > 0, f"Wrong setup: {setup}")
    (generation,) = events(lines, "generation")
    ns = generation["ns"]
    if BACKEND == "test":
        generator, child_ok = ns["backend"], "child" not in generation
    else:
        child = generation["child"]
        generator = ns["child"]
        child_ok = child["status"] == "ok" and ns["child"] >= child["total"]
    require(
        child_ok
        and ns["total"] >= generator + ns["build"] + ns["publish"]
        and "publish_lock_wait" in ns
        and generation["published"] == generation["generated"] == 3,
        f"Wrong generation line: {generation}",
    )
    solutions = events(lines, "solution")
    require(
        [s["rank"] for s in solutions] == list(range(3))
        and all(
            s["outcome"] == "published"
            and s["index"] >= JIT_INDEX
            and s["ns"]["compile_hip"] == sum(u["ns"] for u in s["hip_units"])
            and s["ns"]["build"] > 0
            for s in solutions
        ),
        f"Wrong solution lines: {solutions}",
    )
    query_lines = events(lines, "query")
    require(
        [q["api"] for q in query_lines] == ["c", "cpp"]
        and query_lines[0]["gen"] == generation["gen"]
        and query_lines[1]["jit"]["hits"] == 3,
        f"Wrong query lines: {query_lines}",
    )
    for query, record in zip(query_lines, records):
        require(query["returned"] == record["count"], f"{query} returned {record}")
        check_sources(query)

    stderr, plain = run(
        "plain", args, HIPBLASLT_JIT="1", HIPBLASLT_JIT_LIBRARY_PATH=str(output / "plain")
    )
    require(not debug_lines(stderr), "Debug lines without HIPBLASLT_JIT_DEBUG")
    require(results(plain) == results(records), "HIPBLASLT_JIT_DEBUG changed the results")

    stderr, records = run("second", args, **timing)
    lines = debug_lines(stderr)
    require(
        not events(lines, "generation") and len(events(lines, "query")) == 2,
        "The second process generated",
    )
    for query in events(lines, "query"):
        require(query["jit"]["hits"] == 3, f"The second process missed: {query}")
        check_sources(query)

    stderr, records = run("forced", args, HIPBLASLT_JIT="2", HIPBLASLT_JIT_DEBUG="timing")
    for query in events(debug_lines(stderr), "query"):
        require(
            query["mode"] == 2 and query["from"] == {"jit": 3} and "forced_jit" in query["ns"],
            f"Wrong mode 2 query: {query}",
        )

    barrier = output / "barrier"
    barrier.mkdir()
    library = output / "lib-threads"
    with ThreadPoolExecutor(1) as pool:
        started = pool.submit(
            run,
            "threads",
            ["--api", "c", "--requested", "1", "--threads", "2", "--barrier", str(barrier)],
            HIPBLASLT_JIT_LIBRARY_PATH=str(library),
            **timing,
        )
        deadline = time.monotonic() + 300
        while len(list(barrier.glob("ready-*"))) < 2 and not started.done():
            require(time.monotonic() < deadline, "The threads did not reach the barrier")
            time.sleep(0.01)
        (barrier / "go").touch()
        stderr, records = started.result()
    lines = debug_lines(stderr)
    (generation,) = events(lines, "generation")
    waiters = [q for q in events(lines, "query") if "waited_on" in q.get("jit", {})]
    require(
        len(events(lines, "query")) == 2
        and [q["jit"]["waited_on"] for q in waiters] == [generation["gen"]]
        and waiters[0]["jit"]["hits_after_wait"] == 1,
        f"The waiting thread did not name the generation it waited on: {waiters}",
    )
    print(
        "PASS heuristic-debug-timing: process, setup, query, generation and solution"
        " lines add up, a cache hit does not generate, a waiter names its generation"
    )


def debug_progress(run, output):
    stderr, records = run(
        "first",
        ["--api", "c", "--requested", "2"],
        HIPBLASLT_JIT="1",
        HIPBLASLT_JIT_DEBUG="progress",
    )
    check_jit_results(stderr, records, ("c",), 2)
    lines = debug_lines(stderr)
    require(
        all(line["cat"] == "progress" and "ns" not in line for line in lines),
        "A timing line or duration was written",
    )
    names = [line["ev"] for line in lines]
    child = ["child.start", "child.stage", "child.candidate", "child.done", "child.exit"]
    order = [
        "process",
        "query.start",
        "lookup",
        "generation.start",
        *(child if BACKEND == "tensilelite" else ()),
        "build.start",
        "build.end",
        "publish.start",
        "publish.done",
        "generation.end",
        "query.end",
    ]
    positions = [names.index(name) for name in order]
    require(positions == sorted(positions), f"Events out of order: {names}")
    require(events(lines, "lookup")[0]["result"] == "miss", "The first lookup hit")
    if BACKEND == "tensilelite":
        exit_ = names.index("child.exit")
        relayed = [
            i for i, name in enumerate(names) if name.startswith("child.") and i != exit_
        ]
        require(
            all(i < exit_ for i in relayed)
            and all("child_t_ms" in lines[i] for i in relayed if names[i] != "child.start"),
            "Child events were relayed after child.exit or without child_t_ms",
        )
    else:
        require(not any(name.startswith("child.") for name in names), "A child event without a child")
    (end,) = events(lines, "generation.end")
    require(end["outcome"] == "ok" and end["published"] == 2, f"Wrong end: {end}")

    stderr, records = run(
        "capture",
        ["--api", "none", "--null-algo", "--capture", "global"] + size_args((256, 128, 576)),
        HIPBLASLT_JIT="2",
        HIPBLASLT_JIT_DEBUG="progress",
    )
    lines = debug_lines(stderr)
    (skip,) = events(lines, "capture.skip")
    require(skip["found"] == 0 and not events(lines, "generation.start"), f"{skip}")
    require(
        len(reports(stderr)) == 1
        and reports(stderr)[0].startswith(
            "hipblaslt error: JIT generation skipped during stream capture for"
        ),
        "The capture report changed",
    )
    child = "child, " if BACKEND == "tensilelite" else ""
    print(
        f"PASS heuristic-debug-progress: query, lookup, generation, {child}build and"
        " publish events in order; a capture reports capture.skip"
    )


def debug_off(run, output):
    # The TensileLite generator's arguments must not change either.
    python = argv_python(output) if BACKEND == "tensilelite" else None
    args = ["--api", "c", "--requested", "1"]
    seen = None
    warning = (
        "hipblaslt warning: HIPBLASLT_JIT_DEBUG=0: ignoring 0;"
        " the value is timing, progress or all, comma-separated"
    )
    cases = (("unset", None, False), ("empty", "", False), ("zero", "0", True))
    for name, value, warned in cases:
        argv = output / f"{name}.argv"
        overrides = dict(
            HIPBLASLT_JIT="1",
            HIPBLASLT_JIT_LIBRARY_PATH=str(output / f"lib-{name}"),
        )
        if python:
            overrides.update(HIPBLASLT_JIT_PYTHON=str(python), HIPBLASLT_JIT_TEST_ARGV=str(argv))
        if value is not None:
            overrides["HIPBLASLT_JIT_DEBUG"] = value
        stderr, records = run(name, args, **overrides)
        check_jit_results(stderr, records, ("c",), 1)
        require(not debug_lines(stderr), f"{name}: debug lines were written")
        require(
            debug_warnings(stderr) == ([warning] if warned else []),
            f"{name}: wrong warnings {debug_warnings(stderr)}",
        )
        require(
            not python or (argv.exists() and "--debug" not in argv.read_text()),
            f"{name}: the generator got --debug",
        )
        require(seen is None or results(records) == seen, f"{name}: results changed")
        seen = results(records)

    stderr, records = run(
        "bogus", args, HIPBLASLT_JIT="1", HIPBLASLT_JIT_DEBUG="timing,bogus"
    )
    warnings = debug_warnings(stderr)
    require(
        len(warnings) == 1 and "ignoring bogus" in warnings[0],
        f"timing,bogus: expected one warning, got {warnings}",
    )
    (process,) = events(debug_lines(stderr), "process")
    require(process["categories"] == "timing", f"timing,bogus: {process}")

    library = output / "lib-mode-0"
    plain_stderr, plain = run(
        "mode-0", ["--api", "both", "--no-run"], HIPBLASLT_JIT_LIBRARY_PATH=str(library)
    )
    for value in ("all", "timing", "1"):
        stderr, records = run(
            f"mode-0-{value}",
            ["--api", "both", "--no-run"],
            HIPBLASLT_JIT="0",
            HIPBLASLT_JIT_DEBUG=value,
            HIPBLASLT_JIT_LIBRARY_PATH=str(library),
        )
        require(
            stderr == plain_stderr and records == plain and not library.exists(),
            f"HIPBLASLT_JIT=0 HIPBLASLT_JIT_DEBUG={value} changed the output",
        )
    print(
        "PASS heuristic-debug-off: unset, empty and 0 write nothing and pass no --debug;"
        " an unknown name warns once; mode 0 ignores the variable"
    )


def debug_file(run, output):
    args = ["--api", "both", "--requested", "1"]
    pattern = output / "debug-%i.jsonl"
    stderr, records = run(
        "per-pid",
        args,
        HIPBLASLT_JIT="1",
        HIPBLASLT_JIT_DEBUG="all",
        HIPBLASLT_JIT_DEBUG_FILE=str(pattern),
    )
    check_jit_results(stderr, records, ("c", "cpp"), 1)
    require(not debug_lines(stderr), "Debug lines went to stderr")
    (written,) = output.glob("debug-*.jsonl")
    lines = debug_lines(written.read_text())
    (process,) = events(lines, "process")
    require(
        written.name == f"debug-{process['pid']}.jsonl"
        and process["destination"] == str(written)
        and events(lines, "generation")
        and (written.stat().st_mode & 0o777) == 0o600,
        f"Wrong per-process file {written}: {process}",
    )

    shared = output / "shared.jsonl"
    queries_args = ["--api", "both", "--handles", "2", "--queries", "10", "--no-run"]
    queries_args += ["--requested", "1"]
    with ThreadPoolExecutor(2) as pool:
        outputs = list(
            pool.map(
                lambda name: run(
                    name,
                    queries_args,
                    HIPBLASLT_JIT="1",
                    HIPBLASLT_JIT_DEBUG="all",
                    HIPBLASLT_JIT_DEBUG_FILE=str(shared),
                ),
                ("shared-a", "shared-b"),
            )
        )
    text = shared.read_text()
    lines = debug_lines(text)
    require(
        len(lines) == len(text.splitlines())
        and len({line["pid"] for line in lines}) == 2
        and len(events(lines, "process")) == 2,
        "Lines from two processes sharing a file were not intact",
    )
    require(not any(debug_lines(stderr) for stderr, _ in outputs), "Lines went to stderr")

    unwritable = output / "missing" / "debug.jsonl"
    stderr, _ = run(
        "unwritable",
        args,
        HIPBLASLT_JIT="1",
        HIPBLASLT_JIT_DEBUG="timing",
        HIPBLASLT_JIT_DEBUG_FILE=str(unwritable),
    )
    warnings = debug_warnings(stderr)
    require(
        len(warnings) == 1
        and warnings[0].startswith(f"hipblaslt warning: HIPBLASLT_JIT_DEBUG_FILE={unwritable}")
        and events(debug_lines(stderr), "process"),
        f"An unwritable file did not fall back to stderr once: {warnings}",
    )
    print(
        "PASS heuristic-debug-file: %i names a file per process, processes share a file"
        " line by line, an unwritable file falls back to stderr"
    )


def killing_python(output):
    """A Python wrapper that SIGKILLs the generator, then itself, once it starts kernel_source."""
    python = output / "killing-python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, signal, subprocess, sys, time\n"
        "args = sys.argv[1:]\n"
        "if '--debug-dir' not in args:\n"
        f"    os.execv({sys.executable!r}, [{sys.executable!r}, *args])\n"
        "events = os.path.join(args[args.index('--debug-dir') + 1], 'events.jsonl')\n"
        "target = ('kernel_source', 'start')\n"
        f"child = subprocess.Popen([{sys.executable!r}, *args])\n"
        "while child.poll() is None:\n"
        "    try:\n"
        "        with open(events) as file:\n"
        "            for line in file:\n"
        "                event = json.loads(line)\n"
        "                if (event.get('stage'), event.get('phase')) == target:\n"
        "                    child.kill()\n"
        "                    child.wait()\n"
        "                    os.kill(os.getpid(), signal.SIGKILL)\n"
        "    except (OSError, ValueError):\n"
        "        pass\n"
        "    time.sleep(0.02)\n"
        "sys.exit(child.returncode)\n"
    )
    python.chmod(0o700)
    return python


def debug_killed_child(run, output):
    python = killing_python(output)
    for mode, drop in (("2", ()), ("1", ("HIPBLASLT_TENSILE_LIBPATH",))):
        name = f"mode-{mode}"
        scratch = output / f"tmp-{name}"
        scratch.mkdir()
        library = output / f"lib-{name}"
        stderr, records = run(
            name,
            ["--api", "c", "--requested", "2", "--no-run"],
            drop,
            HIPBLASLT_JIT=mode,
            HIPBLASLT_JIT_DEBUG="all",
            HIPBLASLT_JIT_PYTHON=str(python),
            HIPBLASLT_JIT_LIBRARY_PATH=str(library),
            TMPDIR=str(scratch),
        )
        lines = debug_lines(stderr)
        stages = [
            (line["stage"], line["phase"]) for line in events(lines, "child.stage")
        ]
        (exited,) = events(lines, "child.exit")
        (end,) = events(lines, "generation.end")
        (generation,) = events(lines, "generation")
        require(
            ("kernel_source", "start") in stages
            and exited["signal"] == 9
            and exited["started"]
            and end["outcome"] == "failed"
            and generation["child_timing"] == "missing"
            and generation["exit"]["signal"] == 9,
            f"{name}: wrong child lines {stages} {exited} {end}",
        )
        lines = reports(stderr)
        require(
            len(lines) == 1 and lines[0].startswith("hipblaslt error: JIT generate failed"),
            f"{name}: expected one 'generate failed' error, got {lines}",
        )
        require(not entries(library), f"{name}: a solution was published")
        (record,) = queries(records, "c")
        require(
            all(index < JIT_INDEX for index in record["indices"])
            and (mode == "1" or record["count"] == 0),
            f"{name}: wrong results {record}",
        )
        kept = list(scratch.glob("hipblaslt-jit-*/**/jit-debug/events.jsonl"))
        require(len(kept) == 1, f"{name}: the scratch with events.jsonl was not kept")
    print(
        "PASS heuristic-debug-killed-child: a killed generator gives child.exit with"
        " signal 9, a failed generation, one report and a kept scratch"
    )


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
    "capture": capture,
    "capture-query": capture_query,
    "report": report,
    "partial-fill": partial_fill,
    "override": tuning_override,
    "provider-order": provider_order,
    "debug-timing": debug_timing,
    "debug-progress": debug_progress,
    "debug-off": debug_off,
    "debug-file": debug_file,
    "debug-killed-child": debug_killed_child,
    "jit-off": jit_off,
}


def main():
    global BACKEND
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("route", choices=ROUTES)
    parser.add_argument("fresh_output", type=Path)
    parser.add_argument("--backend", choices=("tensilelite", "test"), default="tensilelite")
    parser.add_argument(
        "--replay", action="append", type=Path, help="a bundle for the test backend to replay"
    )
    args = parser.parse_args()
    if (args.backend == "test") != bool(args.replay):
        parser.error("--replay is required with --backend test, and only then")
    if args.backend == "test" and args.route in TENSILELITE_ROUTES:
        parser.error(f"{args.route} needs the TensileLite generator")
    BACKEND = args.backend
    args.fresh_output.mkdir(parents=True, exist_ok=False)
    output = args.fresh_output.resolve()
    replay = args.replay and os.pathsep.join(str(path.resolve(strict=True)) for path in args.replay)
    runner = Runner(args.executable.resolve(strict=True), output, replay)
    ROUTES[args.route](runner, output)


if __name__ == "__main__":
    main()

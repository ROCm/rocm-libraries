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
    total = found.get("best", 0) + found.get("jit", 0)
    total += found.get("all", 0) + found.get("override", 0)
    require(total == query["returned"], f"from does not add up to returned: {query}")


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
    require(
        ns["total"] >= ns["backend"] + ns["build"] + ns["publish"]
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
    order = [
        "process",
        "query.start",
        "lookup",
        "generation.start",
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
    (end,) = events(lines, "generation.end")
    require(end["outcome"] == "ok" and end["published"] == 2, f"Wrong end: {end}")
    print(
        "PASS heuristic-debug-progress: query, lookup, generation, build and publish"
        " events in order"
    )


def debug_off(run, output):
    args = ["--api", "c", "--requested", "1"]
    seen = None
    warning = (
        "hipblaslt warning: HIPBLASLT_JIT_DEBUG=0: ignoring 0;"
        " the value is timing, progress, knowledge, prediction or all, comma-separated"
    )
    cases = (("unset", None, False), ("empty", "", False), ("zero", "0", True))
    for name, value, warned in cases:
        overrides = dict(
            HIPBLASLT_JIT="1",
            HIPBLASLT_JIT_LIBRARY_PATH=str(output / f"lib-{name}"),
        )
        if value is not None:
            overrides["HIPBLASLT_JIT_DEBUG"] = value
        stderr, records = run(name, args, **overrides)
        check_jit_results(stderr, records, ("c",), 1)
        require(not debug_lines(stderr), f"{name}: debug lines were written")
        require(
            debug_warnings(stderr) == ([warning] if warned else []),
            f"{name}: wrong warnings {debug_warnings(stderr)}",
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
        "PASS heuristic-debug-off: unset, empty and 0 write nothing; an unknown name"
        " warns once; mode 0 ignores the variable"
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
    "debug-timing": debug_timing,
    "debug-progress": debug_progress,
    "debug-off": debug_off,
    "debug-file": debug_file,
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

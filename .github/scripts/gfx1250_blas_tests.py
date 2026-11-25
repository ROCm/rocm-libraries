#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx1250-strict hipBLASLt + TensileLite CI job's checks and test runs.

The workflow (gfx1250_hipblaslt_tensilelite_test.yml) installs the ROCm SDK, builds
hipBLASLt and runs these subcommands; on a gfx1250 machine the same commands reproduce
the job. From the repo root, with the SDK's environment (ROCM_PATH, PATH) set and
hipBLASLt built by ``install.sh -c -a gfx1250-strict --skip_rocroller``:

  python3 .github/scripts/gfx1250_blas_tests.py check-compiler --therock-dir PATH
  python3 .github/scripts/gfx1250_blas_tests.py check-runtime
  python3 .github/scripts/gfx1250_blas_tests.py hipblaslt [--gtest-filter=FILTER]
  python3 .github/scripts/gfx1250_blas_tests.py tensilelite [--pytest-k=EXPR]

Every subcommand runs in gfx1250-strict mode (HSA_DISABLE_GFX12_STRICT=0).

Results come from the XML reports gtest and pytest write, not from the console. The
console and the uploaded logs go through scrub_test_log.py because the job log is public
(see that file for why). The uploaded XML copies have timings and host names removed for
the same reason. A failed hipBLASLt test is listed with the command that reruns it.
Unit tests: .github/scripts/tests/test_gfx1250_blas_tests.py.
"""

import argparse
import fnmatch
import glob
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrub_test_log import Scrubber, text_stream  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HIPBLASLT = REPO / "projects" / "hipblaslt"
ARCH = "gfx1250-strict"

# Large-memory tests (2^31/2^32 sizes). The shards skip them, and they run after the
# shards, one group per process. The client checks each case's memory before allocating,
# but the shards check one moment at a time and reserve nothing, so these tests stay out
# of them. The two stress groups run apart: together they keep ~1.4x the larger one's
# memory.
LARGE = [
    "*fast_check_size_t_index_overflow*",
    "*stress_matmul_size_t_index_overflow*",
    "*size_t_output_overflow*",
    "*offset_above_2_32*",
    "*batch_offset_large*",
]
LARGE_GROUPS = [
    "*fast_check_size_t_index_overflow*",
    "*stress_matmul_size_t_index_overflow_hpa*",
    "*stress_matmul_size_t_index_overflow_*-*stress_matmul_size_t_index_overflow_hpa*",
    "*size_t_output_overflow*",
    "*offset_above_2_32*",
    "*batch_offset_large*",
]
# Skips at this GPU's default Stream-K grid (grid == tiles, no remainder path).
# TENSILE_PERSISTENT_FIXED_GRID applies to every Stream-K GEMM in a process, so the test
# runs in its own process. 360 = 2 x the test's 180 tiles: uniform summation order (on by
# default) keeps it (a grid below the tile count snaps up to 180), and every tile is split
# across two workgroups that hand off through a flag.
SK_DEADLOCK = "StreamKMultiStream.ConcurrentStreamsDoNotDeadlock"
SK_ENV = {"TENSILE_PERSISTENT_FIXED_GRID": "360"}
# The large tests are why the job runs on this host: one that skips for lack of memory
# fails rather than passing quietly.
LARGE_ENV = {"HIPBLASLT_TEST_FAIL_ON_MEMORY_SKIP": "1"}
# How often a running hipBLASLt process reports its progress to the console.
PROGRESS_SECONDS = 120
# gtest's line for one finished test, e.g. "[  FAILED  ] Suite.Test/3, where GetParam() = 7"
# (no "(N ms)": the job sets GTEST_PRINT_TIME=0).
TEST_RESULT = re.compile(r"^\[ +(OK|FAILED|SKIPPED) +\] [A-Za-z_][^ ,]*[.][^ ,]+")
# XML attributes that describe the machine or its speed; removed from the uploaded copies.
PRIVATE_XML_ATTRIBUTES = ("time", "timestamp", "hostname")


class CheckFailed(Exception):
    """A check or a test run failed; main() reports the message and exits 1."""


def error(message):
    print(f"::error::{message}", flush=True)


def warning(message):
    print(f"::warning::{message}", flush=True)


def summary(text):
    """Print text and add it to the job's step summary."""
    print(text, flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text + "\n\n")


def gpus_per_host():
    """GPUs (one job each) sharing this machine's CPUs: GPUS_PER_HOST, default 1."""
    return max(1, int(os.environ.get("GPUS_PER_HOST", "1")))


def job_cpus(allowed=None, total=None):
    """CPUs this job may use. A runner that pins its container (--cpuset-cpus) already gives
    the job its GPU's share; otherwise the affinity mask is the whole host, split by GPU.
    """
    allowed = allowed if allowed is not None else len(os.sched_getaffinity(0))
    total = total if total is not None else (os.cpu_count() or allowed)
    if allowed * gpus_per_host() <= total:
        return allowed
    return max(1, allowed // gpus_per_host())


def in_filter(name, gtest_filter):
    """True if gtest would run test ``name`` under ``gtest_filter`` (POSITIVE[-NEGATIVE])."""
    positive, _, negative = gtest_filter.partition("-")

    def matches(patterns):
        return any(fnmatch.fnmatchcase(name, p) for p in patterns.split(":") if p)

    return matches(positive or "*") and not matches(negative)


def scrub_text(scrubber, text):
    """Scrub multi-line text as the logs are; a dropped line becomes "<redacted>"."""
    lines = [scrubber.scrub(line) for line in text.split("\n")]
    return "\n".join("<redacted>" if line is None else line for line in lines)


def write_public_xml(source, destination):
    """Copy an XML report for the public artifact: no timings or host names, and its
    messages (failure, skip, captured output) scrubbed like the logs."""
    tree = ET.parse(source)
    scrubber = Scrubber(timing="pytest")
    for element in tree.iter():
        for attribute in PRIVATE_XML_ATTRIBUTES:
            element.attrib.pop(attribute, None)
        for key, value in element.attrib.items():
            element.set(key, scrub_text(scrubber, value))
        if element.text and element.text.strip():
            element.text = scrub_text(scrubber, element.text)
    tree.write(destination, encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------- checks


def compiler_of_sdk(rocm_path):
    """(llvm-project commit, TheRock commit) the installed ROCm SDK was built from.

    The SDK's therock_manifest.json pins every submodule; it sits under the SDK root or,
    for the pip SDK, in a sibling _rocm_sdk_* package.
    """
    candidates = [f"{rocm_path}/share/therock/therock_manifest.json"] + glob.glob(
        f"{os.path.dirname(rocm_path)}/_rocm_sdk_*/share/therock/therock_manifest.json"
    )
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        raise FileNotFoundError(f"no therock_manifest.json under {rocm_path}")
    print(f"SDK manifest: {path}")
    with open(path) as f:
        manifest = json.load(f)
    llvm = next(
        submodule.get("pin_sha", "")
        for submodule in manifest.get("submodules", [])
        if submodule.get("submodule_name") == "llvm-project"
    )
    return llvm, manifest.get("the_rock_commit", "")


def compiler_of_therock(therock_dir):
    """(amd-llvm commit, TheRock commit) of a TheRock checkout, from its submodule pin."""

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(therock_dir), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    # "160000 commit <sha>\tcompiler/amd-llvm": the gitlink is the pinned compiler.
    entry = git("ls-tree", "HEAD", "compiler/amd-llvm").split()
    if len(entry) < 3 or entry[1] != "commit":
        raise ValueError(f"compiler/amd-llvm is not a submodule in {therock_dir}")
    return entry[2], git("rev-parse", "HEAD")


def check_compiler(args):
    """Warn when the pinned SDK was built with a different compiler than per-PR CI uses
    (the amd-llvm at ci-env's therock-ref, checked out by the workflow). Never fails: the
    pin stays fixed."""
    sdk = os.environ.get("ROCM_SDK_VERSION", "")
    therock_dir = args.therock_dir or os.environ.get("THEROCK_DIR", "")
    try:
        if not therock_dir:
            raise ValueError(
                "no TheRock checkout: pass --therock-dir or set THEROCK_DIR"
            )
        print(f"TheRock checkout: {therock_dir}")
        want, ref = compiler_of_therock(therock_dir)
        have, built = compiler_of_sdk(os.environ["ROCM_PATH"])
    except (
        OSError,
        ValueError,
        KeyError,
        StopIteration,
        subprocess.CalledProcessError,
    ) as e:
        warning(
            "Could not compare the ROCm SDK's compiler with per-PR CI's "
            f"({type(e).__name__}: {e})."
        )
        return 0
    line = (
        f"ROCm SDK {sdk} (TheRock {built[:10]}) has compiler {have[:10]}; "
        f"per-PR CI (TheRock {ref[:10]}) builds with {want[:10]}."
    )
    if have == want:
        summary(line)
    else:
        warning(
            f"{line} Testing with an older or different compiler than per-PR CI: "
            f"update rocm_sdk_version's default to a nightly built with {want[:10]}."
        )
    return 0


def check_runtime(_args):
    """Fail unless this job sees one GPU and the runtime reports it as gfx1250-strict."""
    # One runner per GPU, set by ROCR_VISIBLE_DEVICES in the runner's .env. The pool
    # registers runner N as ...-gpuN with ROCR_VISIBLE_DEVICES=N, so the name is the
    # fallback.
    gpu = os.environ.get("ROCR_VISIBLE_DEVICES", "")
    named = re.search(r"-gpu([0-9]+)$", os.environ.get("RUNNER_NAME", ""))
    if not gpu and named:
        gpu = os.environ["ROCR_VISIBLE_DEVICES"] = named[1]
        warning(
            f"ROCR_VISIBLE_DEVICES is not set on this runner; using GPU {gpu} from the runner name"
        )
        if os.environ.get("GITHUB_ENV"):
            with open(os.environ["GITHUB_ENV"], "a") as f:
                f.write(f"ROCR_VISIBLE_DEVICES={gpu}\n")
    if not gpu.isdigit():
        raise CheckFailed(
            f"ROCR_VISIBLE_DEVICES='{gpu}' must be a single GPU index: set it in the runner's .env"
        )
    # The host pins each runner's container to its GPU's CPUs; a job that sees them all
    # means that setup is missing on this runner (the job still runs, on a split share).
    allowed, total = len(os.sched_getaffinity(0)), os.cpu_count() or 0
    print(f"CPUs: {allowed} of {total}, {job_cpus()} for this job")
    if gpus_per_host() > 1 and allowed == total:
        warning(
            f"this job's container sees all {total} CPUs: the runner does not pin it to its GPU's CPUs"
        )
    # The same detection tox uses for the client target; prints arch names only.
    spec = importlib.util.spec_from_file_location(
        "GpuArch", HIPBLASLT / "tensilelite/tensilelite/GpuArch.py"
    )
    gpu_arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpu_arch)
    detected = gpu_arch.detect_gpu_arch()
    arch = gpu_arch.cmake_gpu_target(detected) if detected else ""
    if arch != ARCH:
        raise CheckFailed(f"runtime reports '{arch or 'no GPU'}', not {ARCH}")
    print(f"GPU {gpu}: {arch}")
    return 0


def find_linked_libhipblaslt(binary, install):
    """The libhipblaslt hipblaslt-test loads (from ldd); it must be the one built here,
    not the SDK's copy."""
    ldd = subprocess.run(["ldd", str(binary)], capture_output=True, text=True).stdout
    library = next(
        (
            line.split()[2]
            for line in ldd.splitlines()
            if "libhipblaslt.so" in line and len(line.split()) > 2
        ),
        "",
    )
    built_here = (install, HIPBLASLT / "build")
    if not library or not any(
        Path(library).resolve().is_relative_to(d) for d in built_here
    ):
        raise CheckFailed(
            f"hipblaslt-test loads libhipblaslt from '{library or 'nowhere'}', "
            "not the one built here"
        )
    print(f"hipblaslt-test loads {library}")
    return Path(library)


def check_code_objects_target(library):
    """The hipBLASLt library next to ``library`` must hold code objects labelled
    gfx1250-strict: with a stale bundler the label is empty and no GEMM kernel loads."""
    code_objects = sorted(
        library.parent.glob(f"hipblaslt/library/{ARCH}/**/TensileLibrary*.co")
    )
    if not code_objects:
        raise CheckFailed(f"the build produced no {ARCH} hipBLASLt library")
    bundler = Path(os.environ["ROCM_PATH"]) / "lib/llvm/bin/clang-offload-bundler"
    listed = subprocess.run(
        [str(bundler), "--list", "--type=o", f"--input={code_objects[0]}"],
        capture_output=True,
        text=True,
    )
    if listed.returncode:
        raise CheckFailed(
            "cannot list a hipBLASLt code object with the SDK's clang-offload-bundler"
        )
    targets = listed.stdout.split()
    print(
        f"{len(code_objects)} {ARCH} code objects; {code_objects[0].name} "
        f"targets: {', '.join(targets)}"
    )
    if not any(target.endswith(f"-{ARCH}") for target in targets):
        raise CheckFailed(f"hipBLASLt code objects are not labelled {ARCH}")


# ------------------------------------------------------------------------- hipBLASLt


def parse_gtest_xml(path):
    """{"run": count, "failed": [names], "skipped": {name: message}} from a gtest report."""
    root = ET.parse(path).getroot()
    results = {"run": 0, "failed": [], "skipped": {}}
    for case in root.iter("testcase"):
        if case.get("status") != "run":
            continue
        results["run"] += 1
        name = f"{case.get('classname')}.{case.get('name')}"
        if case.find("failure") is not None:
            results["failed"].append(name)
        elif case.get("result") == "skipped":
            skip = case.find("skipped")
            results["skipped"][name] = "" if skip is None else skip.get("message", "")
    return results


class GtestRun:
    """One hipblaslt-test process.

    Its results come from the gtest XML report it writes. Its console output is scrubbed
    and written to ``<out>/<name>.log`` for people reading a failure. While it runs, the
    console shows only its failures and a progress line every PROGRESS_SECONDS (shards run
    at once, so their full output would interleave). The progress line names the test in
    progress, so a hang or a job killed mid-run still shows where each shard was.
    report_hipblaslt() then prints each log in its own collapsible group. Two lines hipblaslt-test prints at the end, peak
    memory and the tests it stopped for lack of memory, are not in the XML, so they are
    read from the output.
    """

    _print_lock = threading.Lock()

    def __init__(self, name, binary, args, env, out, xml_dir):
        self.name = name
        self.log = out / f"{name}.log"
        self.xml = Path(xml_dir) / f"{name}.xml"
        self.public_xml = out / f"{name}.xml"
        self.peak_rss, self.memory_stops = "", []
        self.tests_done = self.tests_failed = 0
        self.running = ""  # the test in progress: where a hang or a killed job stopped
        self.proc = subprocess.Popen(
            [str(binary), *args, f"--gtest_output=xml:{self.xml}"],
            env={**os.environ, **env},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._done = threading.Event()
        self._reader = threading.Thread(target=self._stream, daemon=True)
        self._reader.start()
        # A timer, not the output, drives progress: a hung test prints nothing.
        threading.Thread(target=self._report_progress, daemon=True).start()

    def _say(self, text):
        with self._print_lock:
            print(f"[{self.name}] {text}", flush=True)

    def _report_progress(self):
        while not self._done.wait(PROGRESS_SECONDS):
            self._say(
                f"{self.tests_done} tests done, {self.tests_failed} failed; "
                f"running {self.running or '-'}"
            )

    def _stream(self):
        scrubber = Scrubber()
        # gtest repeats the failed and skipped tests in a list after "[==========] ... ran.";
        # only the per-test result lines before it count.
        in_results = True
        with open(self.log, "w", encoding="utf-8", errors="surrogateescape") as log:
            for raw in text_stream(self.proc.stdout.fileno()):
                line = scrubber.scrub(raw.rstrip("\n"))
                if line is None:
                    continue
                log.write(line + "\n")
                log.flush()
                if line.startswith("[==========]") and line.rstrip().endswith("ran."):
                    in_results = False
                result = TEST_RESULT.match(line) if in_results else None
                if line.startswith("[ RUN      ] "):
                    self.running = line[len("[ RUN      ] ") :]
                elif result:
                    self.tests_done += 1
                    if result[1] == "FAILED":
                        self.tests_failed += 1
                        self._say(line)
                elif line.startswith("[ PEAK RSS ] "):
                    self.peak_rss = line.split()[4]
                elif re.match(r"\[ MEMORY   \] [^0-9]", line):
                    self.memory_stops.append(line[len("[ MEMORY   ] ") :])

    def wait(self):
        """Wait for the process; return (exit code, results or None if it wrote none)."""
        exit_code = self.proc.wait()
        self._reader.join()
        self._done.set()
        try:
            results = parse_gtest_xml(self.xml)
            write_public_xml(self.xml, self.public_xml)
        except (OSError, ET.ParseError):
            return exit_code, None
        self._say(
            f"finished: exit {exit_code}, {results['run']} run, "
            f"{len(results['failed'])} failed, {len(results['skipped'])} skipped"
        )
        return exit_code, results


def parse_test_list(text):
    """Full test names from ``--gtest_list_tests`` output."""
    names, suite = [], ""
    for line in text.splitlines():
        if not line.startswith(" ") and line.endswith("."):
            suite = line.split()[0]
        elif line.startswith("  ") and suite:
            names.append(suite + line.split()[0])
    return names


def list_tests(binary, gtest_filter):
    """Full names of the tests gtest selects for gtest_filter."""
    output = subprocess.run(
        [str(binary), "--gtest_list_tests", f"--gtest_filter={gtest_filter}"],
        capture_output=True,
        text=True,
        errors="replace",
        check=True,  # an empty listing would silently skip the large-memory groups
    ).stdout
    return parse_test_list(output)


def run_hipblaslt(args):
    binary = Path(args.binary).resolve()
    install = (HIPBLASLT / "hipblaslt-install").resolve()
    user_filter = args.gtest_filter or ""

    # The CPU reference's OpenMP and OpenBLAS pools start one thread per CPU by default;
    # with every GPU's job busy that oversubscribes the host ~16x. Split this job's share
    # of the CPUs between the shards. BLIS caps itself at 8 threads in-process
    # (blis_interface.cpp), but only while OMP_NUM_THREADS is unset, so keep that cap.
    cpus = job_cpus()
    threads = max(1, cpus // args.shards)
    os.environ.update(
        OMP_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS=str(threads),
        BLIS_NUM_THREADS=str(min(threads, 8)),
        # StreamKMultiStream.ConcurrentStreamsDoNotDeadlock fails without it.
        HSA_HOTSWAP_DISABLE="1",
        HIPBLASLT_TEST_TIMEOUT="3600",
        GTEST_PRINT_TIME="0",
        LD_LIBRARY_PATH=":".join(
            p for p in (f"{install}/lib", os.environ.get("LD_LIBRARY_PATH")) if p
        ),
    )
    # The runtime loads library/gfx1250-strict/ next to libhipblaslt.
    os.environ.pop("HIPBLASLT_TENSILE_LIBPATH", None)
    library = find_linked_libhipblaslt(binary, install)
    check_code_objects_target(library)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.shards} shards, {threads} CPU threads each; logs and XML in {out}")

    # gtest's own XML keeps timings: it stays out of the uploaded directory.
    xml_dir = tempfile.mkdtemp(
        prefix="hipblaslt-gtest-", dir=os.environ.get("RUNNER_TEMP")
    )
    started, finished = [], []  # GtestRun; (GtestRun, exit code, results)
    problems = []  # filters that select no test: a renamed test would drop out silently

    def start(name, gtest_filter, env):
        started.append(
            GtestRun(
                name, binary, [f"--gtest_filter={gtest_filter}"], env, out, xml_dir
            )
        )
        return started[-1]

    def run(name, gtest_filter, env):
        process = start(name, gtest_filter, env)
        finished.append((process, *process.wait()))
        return finished[-1][2]

    try:
        # Shards in parallel: one shard's CPU reference checks overlap another's GPU work.
        positive, _, negative = user_filter.partition("-")
        negatives = ":".join(([negative] if negative else []) + LARGE + [SK_DEADLOCK])
        shard_filter = f"{positive or '*'}-{negatives}"
        shards = [
            start(
                f"hipblaslt_test_shard{i}",
                shard_filter,
                {"GTEST_TOTAL_SHARDS": str(args.shards), "GTEST_SHARD_INDEX": str(i)},
            )
            for i in range(args.shards)
        ]
        finished += [(s, *s.wait()) for s in shards]

        if not user_filter or in_filter(SK_DEADLOCK, user_filter):
            results = run("hipblaslt_test_shard_streamk", SK_DEADLOCK, SK_ENV)
            if results and not results["run"]:
                problems.append(f"{SK_DEADLOCK} matches no test; update SK_DEADLOCK")
            elif results and SK_DEADLOCK in results["skipped"]:
                warning(
                    f"{SK_DEADLOCK} skipped at TENSILE_PERSISTENT_FIXED_GRID="
                    f"{SK_ENV['TENSILE_PERSISTENT_FIXED_GRID']}: {results['skipped'][SK_DEADLOCK]}"
                )

        # The shards are done, so the large tests get the job's whole CPU share.
        os.environ.update(OMP_NUM_THREADS=str(cpus), OPENBLAS_NUM_THREADS=str(cpus))
        if user_filter:
            # gtest picks the tests the run's filter selects; each goes to its group.
            names = [
                n
                for n in list_tests(binary, user_filter)
                if in_filter(n, ":".join(LARGE))
            ]
            groups = [
                ":".join(n for n in names if in_filter(n, g)) for g in LARGE_GROUPS
            ]
        else:
            groups = LARGE_GROUPS
        for index, group in enumerate(groups):
            if not group:
                continue
            results = run(f"hipblaslt_test_shard_large{index}", group, LARGE_ENV)
            if results and not results["run"]:
                problems.append(
                    f"large-memory group '{group}' matches no test; update LARGE_GROUPS"
                )
    finally:
        for process in started:
            if process.proc.poll() is None:
                process.proc.terminate()
        shutil.rmtree(xml_dir, ignore_errors=True)

    report_hipblaslt(finished, problems)
    return 0


def report_hipblaslt(finished, problems):
    """Print each process's log in a collapsible group, the per-process table and every
    failure with its rerun command; raise CheckFailed if anything failed."""
    for process, exit_code, results in finished:
        failed = len(results["failed"]) if results else "?"
        print(f"::group::{process.name}: exit {exit_code}, {failed} failed", flush=True)
        print(process.log.read_text(errors="replace"), end="", flush=True)
        print("::endgroup::", flush=True)

    rows = [
        "### hipBLASLt test processes",
        "| Process | Exit | Ran | Failed | Skipped | Peak resident memory (GiB) | Tests stopped for memory |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    failed, memory_stops = [], []
    for process, exit_code, results in finished:
        if results is None:
            problems.append(
                f"{process.name} exited with {exit_code} and wrote no results: "
                f"see {process.log.name}"
            )
            rows.append(
                f"| {process.name} | {exit_code} | no results | | | {process.peak_rss} | |"
            )
            continue
        if exit_code and not results["failed"]:
            problems.append(
                f"{process.name} exited with {exit_code}: see {process.log.name}"
            )
        failed += results["failed"]
        memory_stops += process.memory_stops
        rows.append(
            f"| {process.name} | {exit_code} | {results['run']} | {len(results['failed'])} | "
            f"{len(results['skipped'])} | {process.peak_rss} | {len(process.memory_stops)} |"
        )
    summary("\n".join(rows))
    if memory_stops:
        summary("```\n" + "\n".join(memory_stops) + "\n```")
    if failed:
        script = shlex.quote(os.path.relpath(__file__, REPO))
        rerun = [
            f"python3 {script} hipblaslt --gtest-filter={shlex.quote(t)}"
            for t in failed
        ]
        summary(
            f"{len(failed)} hipBLASLt tests failed. Rerun one as this job did, on a gfx1250 "
            "machine from the repo root:\n```\n"
            + "\n".join(sorted(set(rerun)))
            + "\n```"
        )
        problems.append(f"{len(failed)} hipBLASLt tests failed")
    if problems:
        raise CheckFailed("; ".join(problems))


# ------------------------------------------------------------------------ TensileLite


def parse_junit_xml(path):
    """{"tests", "failures", "errors", "skipped"} totals from a pytest JUnit report."""
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for suite in suites:
        for key in totals:
            totals[key] += int(suite.get(key, 0))
    return totals


def run_tensilelite(args):
    if re.search(r"\s", args.pytest_k):
        raise CheckFailed(
            "--pytest-k must not contain spaces (tox splits them into separate arguments)"
        )
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / "common_test.log"
    tensilelite = HIPBLASLT / "tensilelite"
    # tox.ini's py3 command writes pytest's JUnit report here.
    junit = tensilelite / "python_tests.xml"
    junit.unlink(missing_ok=True)
    root = os.environ["ROCM_PATH"]
    env = {
        **os.environ,
        "CC": f"{root}/lib/llvm/bin/amdclang",
        "CXX": f"{root}/lib/llvm/bin/amdclang++",
    }
    # As the nightly: tox builds the client for the reported target (gfx1250-strict
    # here) and runs the common tests against it. The pytest-xdist workers share this
    # GPU's client lock (conftest.py): the GPU runs take turns, but one test's kernel
    # compile overlaps another's GPU run. skip-gfx1250-strict configs are skipped on this
    # target (config_helpers.py). Each argument after -- stays a single word: tox inserts
    # them into tox.ini's sh -c command unquoted.
    command = ["tox", "-v", "-e", "py3", "--", "tensilelite/Tests", "-m", "common"]
    command += ["-k", args.pytest_k] if args.pytest_k else []
    command += [f"--timeout={args.timeout}", "-n", str(args.workers), "-rfxX"]
    print(f"Running: {shlex.join(command)} (in {tensilelite}); log: {log}", flush=True)
    proc = subprocess.Popen(
        command,
        cwd=tensilelite,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # One process: its output goes to the console as it runs, scrubbed.
    scrubber = Scrubber(timing="pytest")
    with open(log, "w", encoding="utf-8", errors="surrogateescape") as f:
        for raw in text_stream(proc.stdout.fileno()):
            line = scrubber.scrub(raw.rstrip("\n"))
            if line is None:
                continue
            f.write(line + "\n")
            f.flush()
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
    exit_code = proc.wait()

    if not junit.is_file():
        raise CheckFailed(
            f"tox exited with {exit_code} and wrote no {junit.name}: see {log.name}"
        )
    totals = parse_junit_xml(junit)
    write_public_xml(junit, out / "common_tests.xml")
    summary(
        f"### TensileLite common tests\n{totals['tests']} tests: {totals['failures']} failed, "
        f"{totals['errors']} errors, {totals['skipped']} skipped (tox exit {exit_code})"
    )
    # tox.ini maps pytest's "no tests collected" (exit 5) to success.
    if totals["tests"] == 0:
        raise CheckFailed("TensileLite ran no tests")
    if exit_code or totals["failures"] or totals["errors"]:
        raise CheckFailed(
            f"TensileLite: {totals['failures']} failed, {totals['errors']} errors "
            f"(tox exit {exit_code}); see {log.name}"
        )
    return 0


# ------------------------------------------------------------------------------- main


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subcommands = parser.add_subparsers(dest="command", required=True)

    compiler_parser = subcommands.add_parser(
        "check-compiler", help=check_compiler.__doc__.splitlines()[0]
    )
    compiler_parser.add_argument(
        "--therock-dir",
        default="",
        help="TheRock checkout at ci-env's therock-ref (default: $THEROCK_DIR)",
    )
    subcommands.add_parser("check-runtime", help=check_runtime.__doc__.splitlines()[0])

    hipblaslt_parser = subcommands.add_parser(
        "hipblaslt", help="run the hipBLASLt gtest suite"
    )
    hipblaslt_parser.add_argument(
        "--gtest-filter", default="", help="gtest filter; empty runs everything"
    )
    hipblaslt_parser.add_argument(
        "--binary",
        default=str(HIPBLASLT / "build/release/clients/hipblaslt-test"),
        help="hipblaslt-test to run",
    )
    hipblaslt_parser.add_argument(
        "--out",
        default=str(REPO / "test_results"),
        help="directory for the scrubbed logs and XML reports",
    )
    hipblaslt_parser.add_argument(
        "--shards", type=int, default=4, help="gtest shards run at once on the GPU"
    )

    tensilelite_parser = subcommands.add_parser(
        "tensilelite", help="run the TensileLite common tests via tox"
    )
    tensilelite_parser.add_argument(
        "--pytest-k", default="", help="pytest -k expression, no spaces"
    )
    tensilelite_parser.add_argument(
        "--out",
        default=str(REPO / "tensilelite_test_results"),
        help="directory for the scrubbed log and XML report",
    )
    tensilelite_parser.add_argument(
        "--workers", type=int, default=4, help="pytest-xdist workers"
    )
    # decouple_pgr_tdm_fuse_gfx1250.yaml's sweep runs past 600 s.
    tensilelite_parser.add_argument(
        "--timeout", type=int, default=1200, help="per-test timeout in seconds"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ["HSA_DISABLE_GFX12_STRICT"] = "0"
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(encoding="utf-8", errors="surrogateescape")
    command = {
        "check-compiler": check_compiler,
        "check-runtime": check_runtime,
        "hipblaslt": run_hipblaslt,
        "tensilelite": run_tensilelite,
    }[args.command]
    try:
        return command(args)
    except CheckFailed as failure:
        error(str(failure))
        return 1


if __name__ == "__main__":
    sys.exit(main())

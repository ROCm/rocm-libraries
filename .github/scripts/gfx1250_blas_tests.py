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
from typing import NamedTuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrub_test_log import Scrubber, text_stream  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
HIPBLASLT_DIR = REPO_ROOT / "projects" / "hipblaslt"
# The target the runtime reports for this GPU in strict mode, and the hipBLASLt library
# directory it loads kernels from.
GPU_TARGET = "gfx1250-strict"

# Large-memory tests (2^31/2^32 sizes). The shards skip them, and they run after the
# shards, one group per process. The client checks each case's memory before allocating,
# but the shards check one moment at a time and reserve nothing, so these tests stay out
# of them.
LARGE_MEMORY_TESTS = [
    "*fast_check_size_t_index_overflow*",
    "*stress_matmul_size_t_index_overflow*",
    "*size_t_output_overflow*",
    "*offset_above_2_32*",
    "*batch_offset_large*",
]
# One gtest filter per process for LARGE_MEMORY_TESTS. The two stress groups run apart
# because together they keep ~1.4x the larger one's memory: the _hpa tests, then the other
# stress tests ("everything matching ..._overflow_ except ..._overflow_hpa").
LARGE_MEMORY_GROUPS = [
    "*fast_check_size_t_index_overflow*",
    "*stress_matmul_size_t_index_overflow_hpa*",
    "*stress_matmul_size_t_index_overflow_*-*stress_matmul_size_t_index_overflow_hpa*",
    "*size_t_output_overflow*",
    "*offset_above_2_32*",
    "*batch_offset_large*",
]
# The large tests are why the job runs on this host: one that skips for lack of memory
# fails rather than passing quietly.
LARGE_MEMORY_ENV = {"HIPBLASLT_TEST_FAIL_ON_MEMORY_SKIP": "1"}
# This Stream-K test skips itself at the GPU's default grid (one workgroup per tile, so no
# tile is split and there is nothing to deadlock). TENSILE_PERSISTENT_FIXED_GRID=360 forces
# 2 workgroups per tile for its 180 tiles, so every tile is split and the two halves hand
# off through a flag. (A grid below the tile count snaps up to 180 under uniform summation
# order, which is on by default.) The variable changes every Stream-K GEMM in a process, so
# this test runs in its own process.
STREAMK_DEADLOCK_TEST = "StreamKMultiStream.ConcurrentStreamsDoNotDeadlock"
STREAMK_DEADLOCK_ENV = {"TENSILE_PERSISTENT_FIXED_GRID": "360"}
# How often a running hipBLASLt process reports its progress to the console.
PROGRESS_SECONDS = 120
# gtest's line for one finished test, e.g. "[  FAILED  ] Suite.Test/3, where GetParam() = 7"
# (no "(N ms)": the job sets GTEST_PRINT_TIME=0).
TEST_RESULT = re.compile(r"^\[ +(OK|FAILED|SKIPPED) +\] [A-Za-z_][^ ,]*[.][^ ,]+")
# XML attributes that describe the machine or its speed; removed from the uploaded copies.
PRIVATE_XML_ATTRIBUTES = ("time", "timestamp", "hostname")
# Commit hashes are printed this short: enough to tell commits apart.
SHORT_SHA_LENGTH = 10


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


def short_sha(commit):
    return commit[:SHORT_SHA_LENGTH]


def gpus_per_host():
    """GPUs (one job each) sharing this machine's CPUs: GPUS_PER_HOST, default 1."""
    return max(1, int(os.environ.get("GPUS_PER_HOST", "1")))


def job_cpus(cpus_visible=None, cpus_on_host=None):
    """CPUs this job may use. A runner that pins its container (--cpuset-cpus) already gives
    the job its GPU's share; otherwise the job sees the whole host and takes 1/GPUS_PER_HOST.

    E.g. with 4 GPUs on a 256-CPU host: a pinned container sees 64 CPUs (64 x 4 <= 256) and
    keeps them; an unpinned one sees all 256 and takes 256 / 4 = 64.
    """
    if cpus_visible is None:
        cpus_visible = len(os.sched_getaffinity(0))
    if cpus_on_host is None:
        cpus_on_host = os.cpu_count() or cpus_visible
    if cpus_visible * gpus_per_host() <= cpus_on_host:
        return cpus_visible
    return max(1, cpus_visible // gpus_per_host())


def gtest_filter_selects(test_name, gtest_filter):
    """True if gtest runs ``test_name`` under ``gtest_filter``: POSITIVE[-NEGATIVE], each a
    ':'-separated list of globs, e.g. "*smoke*:*quick*-*bad_arg*"."""
    positive, _, negative = gtest_filter.partition("-")

    def matches_any(patterns):
        return any(
            fnmatch.fnmatchcase(test_name, pattern)
            for pattern in patterns.split(":")
            if pattern
        )

    return matches_any(positive or "*") and not matches_any(negative)


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


def sdk_compiler_and_therock_commits(rocm_path):
    """(llvm-project commit, TheRock commit) the installed ROCm SDK was built from.

    The SDK's therock_manifest.json pins every submodule; it sits under the SDK root or,
    for the pip SDK, in a sibling _rocm_sdk_* package.
    """
    candidates = [f"{rocm_path}/share/therock/therock_manifest.json"] + glob.glob(
        f"{os.path.dirname(rocm_path)}/_rocm_sdk_*/share/therock/therock_manifest.json"
    )
    manifest_path = next((p for p in candidates if os.path.isfile(p)), None)
    if manifest_path is None:
        raise FileNotFoundError(f"no therock_manifest.json under {rocm_path}")
    print(f"SDK manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    llvm_commit = next(
        (
            submodule.get("pin_sha", "")
            for submodule in manifest.get("submodules", [])
            if submodule.get("submodule_name") == "llvm-project"
        ),
        None,
    )
    if llvm_commit is None:
        raise ValueError(f"{manifest_path} pins no llvm-project submodule")
    return llvm_commit, manifest.get("the_rock_commit", "")


def therock_compiler_and_commit(therock_dir):
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
    sdk_version = os.environ.get("ROCM_SDK_VERSION", "")
    therock_dir = args.therock_dir or os.environ.get("THEROCK_DIR", "")
    try:
        if not therock_dir:
            raise ValueError(
                "no TheRock checkout: pass --therock-dir or set THEROCK_DIR"
            )
        print(f"TheRock checkout: {therock_dir}")
        ci_compiler, ci_therock = therock_compiler_and_commit(therock_dir)
        sdk_compiler, sdk_therock = sdk_compiler_and_therock_commits(
            os.environ["ROCM_PATH"]
        )
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as e:
        warning(
            "Could not compare the ROCm SDK's compiler with per-PR CI's "
            f"({type(e).__name__}: {e})."
        )
        return 0
    comparison = (
        f"ROCm SDK {sdk_version} (TheRock {short_sha(sdk_therock)}) has compiler "
        f"{short_sha(sdk_compiler)}; per-PR CI (TheRock {short_sha(ci_therock)}) builds "
        f"with {short_sha(ci_compiler)}."
    )
    if sdk_compiler == ci_compiler:
        summary(comparison)
    else:
        warning(
            f"{comparison} Testing with an older or different compiler than per-PR CI: "
            f"update rocm_sdk_version's default to a nightly built with "
            f"{short_sha(ci_compiler)}."
        )
    return 0


def use_gpu_from_runner_name():
    """The GPU index this job may use. One runner per GPU, set by ROCR_VISIBLE_DEVICES in
    the runner's .env; the pool names runner N ...-gpuN, so the name is the fallback."""
    gpu = os.environ.get("ROCR_VISIBLE_DEVICES", "")
    runner_gpu = re.search(r"-gpu([0-9]+)$", os.environ.get("RUNNER_NAME", ""))
    if not gpu and runner_gpu:
        gpu = os.environ["ROCR_VISIBLE_DEVICES"] = runner_gpu.group(1)
        warning(
            f"ROCR_VISIBLE_DEVICES is not set on this runner; using GPU {gpu} from the runner name"
        )
        # Later steps of the job read it from GITHUB_ENV.
        if os.environ.get("GITHUB_ENV"):
            with open(os.environ["GITHUB_ENV"], "a") as f:
                f.write(f"ROCR_VISIBLE_DEVICES={gpu}\n")
    return gpu


def check_runtime(_args):
    """Fail unless this job sees one GPU and the runtime reports it as gfx1250-strict."""
    gpu = use_gpu_from_runner_name()
    if not gpu.isdigit():
        raise CheckFailed(
            f"ROCR_VISIBLE_DEVICES='{gpu}' must be a single GPU index: set it in the runner's .env"
        )
    # The host pins each runner's container to its GPU's CPUs; a job that sees them all
    # means that setup is missing on this runner (the job still runs, on a split share).
    cpus_visible, cpus_on_host = len(os.sched_getaffinity(0)), os.cpu_count() or 0
    print(f"CPUs: {cpus_visible} of {cpus_on_host}, {job_cpus()} for this job")
    if gpus_per_host() > 1 and cpus_visible == cpus_on_host:
        warning(
            f"this job's container sees all {cpus_on_host} CPUs: the runner does not pin it to its GPU's CPUs"
        )
    # The same detection tox uses for the client target; prints arch names only.
    spec = importlib.util.spec_from_file_location(
        "GpuArch", HIPBLASLT_DIR / "tensilelite/tensilelite/GpuArch.py"
    )
    gpu_arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpu_arch)
    detected = gpu_arch.detect_gpu_arch()
    reported_target = gpu_arch.cmake_gpu_target(detected) if detected else ""
    if reported_target != GPU_TARGET:
        raise CheckFailed(
            f"runtime reports '{reported_target or 'no GPU'}', not {GPU_TARGET}"
        )
    print(f"GPU {gpu}: {reported_target}")
    return 0


def find_linked_libhipblaslt(test_binary, install_dir):
    """The libhipblaslt hipblaslt-test loads (from ldd); it must be the one built here,
    not the SDK's copy. A failed ldd lists nothing and fails the check below."""
    ldd_output = subprocess.run(
        ["ldd", str(test_binary)], capture_output=True, text=True
    ).stdout
    # ldd line: "libhipblaslt.so.1 => /path/to/libhipblaslt.so.1 (0x...)"; field 3 is the path.
    library = next(
        (
            line.split()[2]
            for line in ldd_output.splitlines()
            if "libhipblaslt.so" in line and len(line.split()) > 2
        ),
        "",
    )
    built_here = (install_dir, HIPBLASLT_DIR / "build")
    if not library or not any(
        Path(library).resolve().is_relative_to(directory) for directory in built_here
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
        library.parent.glob(f"hipblaslt/library/{GPU_TARGET}/**/TensileLibrary*.co")
    )
    if not code_objects:
        raise CheckFailed(f"the build produced no {GPU_TARGET} hipBLASLt library")
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
    # One target per line, e.g. "hipv4-amdgcn-amd-amdhsa--gfx1250-strict".
    targets = listed.stdout.split()
    print(
        f"{len(code_objects)} {GPU_TARGET} code objects; {code_objects[0].name} "
        f"targets: {', '.join(targets)}"
    )
    if not any(target.endswith(f"-{GPU_TARGET}") for target in targets):
        raise CheckFailed(f"hipBLASLt code objects are not labelled {GPU_TARGET}")


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

    Results come from the gtest XML report it writes. Its console output is scrubbed and
    written to ``<output_dir>/<name>.log`` for people reading a failure.

    While it runs, the console shows a start line, each failure as it happens and a
    progress line every PROGRESS_SECONDS naming the test in progress, so a hang or a job
    killed mid-run still shows where it was. Not the full output: shards run at once, so
    their output would interleave. report_hipblaslt() prints each full log afterwards.

    Two lines hipblaslt-test prints at the end are not in the XML, so they are read from
    the output: its peak memory and the tests it stopped for lack of memory.
    """

    _print_lock = threading.Lock()

    def __init__(self, name, test_binary, gtest_filter, env, output_dir, xml_dir):
        self.name = name
        self.log = output_dir / f"{name}.log"
        self.xml = Path(xml_dir) / f"{name}.xml"
        self.public_xml = output_dir / f"{name}.xml"
        self.peak_memory_gib, self.memory_stops = "", []
        self.tests_done = self.tests_failed = 0
        self.running = ""  # the test in progress: where a hang or a killed job stopped
        command = [
            str(test_binary),
            f"--gtest_filter={gtest_filter}",
            f"--gtest_output=xml:{self.xml}",
        ]
        self._print_prefixed(
            f"started: {shlex.join(command[:2])}"
            + "".join(f" {key}={value}" for key, value in env.items())
        )
        self.process = subprocess.Popen(
            command,
            env={**os.environ, **env},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._done = threading.Event()
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()
        # A timer, not the output, drives progress: a hung test prints nothing.
        threading.Thread(target=self._report_progress, daemon=True).start()

    def _print_prefixed(self, text):
        """Print one line tagged with this process's name; shards print at the same time."""
        with self._print_lock:
            print(f"[{self.name}] {text}", flush=True)

    def _report_progress(self):
        while not self._done.wait(PROGRESS_SECONDS):
            self._print_prefixed(
                f"{self.tests_done} tests done, {self.tests_failed} failed; "
                f"running {self.running or '-'}"
            )

    def _read_output(self):
        """Scrub each output line into the log, and track progress, failures and the two
        end-of-run lines."""
        scrubber = Scrubber()
        # gtest repeats the failed and skipped tests in a list after "[==========] ... ran.";
        # only the per-test result lines before it count.
        in_results = True
        with open(self.log, "w", encoding="utf-8", errors="surrogateescape") as log:
            for raw in text_stream(self.process.stdout.fileno()):
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
                        self._print_prefixed(line)
                elif line.startswith("[ PEAK RSS ] "):
                    # "[ PEAK RSS ] 12.3 GiB of host memory": the 5th word is the value.
                    self.peak_memory_gib = line.split()[4]
                elif re.match(r"\[ MEMORY   \] [^0-9]", line):
                    # One line per stopped test, "[ MEMORY   ] Suite.big: this case needs
                    # ..."; the count line before them, "[ MEMORY   ] 2 tests stopped ...",
                    # starts with a digit and is skipped.
                    self.memory_stops.append(line[len("[ MEMORY   ] ") :])

    def wait(self):
        """Wait for the process; return its FinishedRun (results None if it wrote none)."""
        exit_code = self.process.wait()
        self._reader.join()
        self._done.set()
        try:
            results = parse_gtest_xml(self.xml)
            write_public_xml(self.xml, self.public_xml)
        except (OSError, ET.ParseError):
            return FinishedRun(self, exit_code, None)
        self._print_prefixed(
            f"finished: exit {exit_code}, {results['run']} run, "
            f"{len(results['failed'])} failed, {len(results['skipped'])} skipped"
        )
        return FinishedRun(self, exit_code, results)


class FinishedRun(NamedTuple):
    process: GtestRun
    exit_code: int
    results: Optional[dict]  # parse_gtest_xml()'s result; None if the process wrote no XML


def parse_test_list(text):
    """Full test names from ``--gtest_list_tests`` output, which prints each suite then its
    tests indented: "Suite.\\n  Test/0  # GetParam() = 1" gives "Suite.Test/0"."""
    names, suite = [], ""
    for line in text.splitlines():
        if not line.startswith(" ") and line.endswith("."):
            suite = line.split()[0]
        elif line.startswith("  ") and suite:
            names.append(suite + line.split()[0])
    return names


def list_tests(test_binary, gtest_filter):
    """Full names of the tests gtest selects for gtest_filter."""
    output = subprocess.run(
        [str(test_binary), "--gtest_list_tests", f"--gtest_filter={gtest_filter}"],
        capture_output=True,
        text=True,
        errors="replace",
        check=True,  # an empty listing would silently skip the large-memory groups
    ).stdout
    return parse_test_list(output)


def run_hipblaslt(args):
    """Run the hipBLASLt gtests in three phases, then report:
    1. args.shards shards at once, everything except the tests below;
    2. the Stream-K deadlock test, in its own process;
    3. the large-memory groups, one process each, one after another.
    """
    test_binary = Path(args.binary).resolve()
    install_dir = (HIPBLASLT_DIR / "hipblaslt-install").resolve()
    user_filter = args.gtest_filter or ""

    # The CPU reference's OpenMP and OpenBLAS pools start one thread per CPU by default.
    # With 4 shards in each of 4 GPU jobs on one host, that is 16 processes each starting
    # a thread per CPU. Split this job's share of the CPUs between its shards instead. BLIS
    # caps itself at 8 threads in-process (blis_interface.cpp), but only while
    # OMP_NUM_THREADS is unset, so keep that cap.
    cpus = job_cpus()
    threads = max(1, cpus // args.shards)
    os.environ.update(
        OMP_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS=str(threads),
        BLIS_NUM_THREADS=str(min(threads, 8)),
        # StreamKMultiStream.ConcurrentStreamsDoNotDeadlock fails without it.
        HSA_HOTSWAP_DISABLE="1",
        # Seconds one test may take before hipblaslt-test stops all testing (default 600).
        HIPBLASLT_TEST_TIMEOUT="3600",
        # No "(N ms)" per test: timings stay out of the public log.
        GTEST_PRINT_TIME="0",
        LD_LIBRARY_PATH=":".join(
            path
            for path in (f"{install_dir}/lib", os.environ.get("LD_LIBRARY_PATH"))
            if path
        ),
    )
    # The runtime loads library/gfx1250-strict/ next to libhipblaslt.
    os.environ.pop("HIPBLASLT_TENSILE_LIBPATH", None)
    library = find_linked_libhipblaslt(test_binary, install_dir)
    check_code_objects_target(library)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"{args.shards} shards, {threads} CPU threads each; logs and XML in {output_dir}"
    )

    # gtest's own XML keeps timings: it stays out of the uploaded directory.
    xml_dir = tempfile.mkdtemp(
        prefix="hipblaslt-gtest-", dir=os.environ.get("RUNNER_TEMP")
    )
    started = []  # every GtestRun, so the finally block can stop any still running
    finished = []  # FinishedRun of each, in the order they finished
    # Why the step fails besides failed tests, e.g. a filter that now selects no test
    # (a renamed test would otherwise drop out silently).
    failure_reasons = []

    def start_process(name, gtest_filter, env):
        process = GtestRun(name, test_binary, gtest_filter, env, output_dir, xml_dir)
        started.append(process)
        return process

    def run_process_to_end(name, gtest_filter, env):
        finished.append(start_process(name, gtest_filter, env).wait())
        return finished[-1].results

    try:
        # Phase 1. The shards run in parallel: one shard's CPU reference checks overlap
        # another's GPU work. They exclude what phases 2 and 3 run, e.g. the user filter
        # "*smoke*-*bad_arg*" becomes "*smoke*-*bad_arg*:<large-memory tests>:<Stream-K test>".
        included, _, excluded = user_filter.partition("-")
        excluded_patterns = ":".join(
            ([excluded] if excluded else [])
            + LARGE_MEMORY_TESTS
            + [STREAMK_DEADLOCK_TEST]
        )
        shard_filter = f"{included or '*'}-{excluded_patterns}"
        shards = [
            start_process(
                f"hipblaslt_test_shard{index}",
                shard_filter,
                {
                    "GTEST_TOTAL_SHARDS": str(args.shards),
                    "GTEST_SHARD_INDEX": str(index),
                },
            )
            for index in range(args.shards)
        ]
        finished += [shard.wait() for shard in shards]

        # Phase 2.
        if not user_filter or gtest_filter_selects(STREAMK_DEADLOCK_TEST, user_filter):
            results = run_process_to_end(
                "hipblaslt_test_shard_streamk",
                STREAMK_DEADLOCK_TEST,
                STREAMK_DEADLOCK_ENV,
            )
            if results and not results["run"]:
                failure_reasons.append(
                    f"{STREAMK_DEADLOCK_TEST} matches no test; update STREAMK_DEADLOCK_TEST"
                )
            elif results and STREAMK_DEADLOCK_TEST in results["skipped"]:
                warning(
                    f"{STREAMK_DEADLOCK_TEST} skipped at TENSILE_PERSISTENT_FIXED_GRID="
                    f"{STREAMK_DEADLOCK_ENV['TENSILE_PERSISTENT_FIXED_GRID']}: "
                    f"{results['skipped'][STREAMK_DEADLOCK_TEST]}"
                )

        # Phase 3. The shards are done, so the large tests get the job's whole CPU share.
        os.environ.update(OMP_NUM_THREADS=str(cpus), OPENBLAS_NUM_THREADS=str(cpus))
        if user_filter:
            # Run only the large-memory tests the user filter selects: list them, then
            # give each group the explicit list of its own, e.g.
            # "A.size_t_output_overflow/0:A.size_t_output_overflow/1" (empty: skipped).
            selected_large_tests = [
                test_name
                for test_name in list_tests(test_binary, user_filter)
                if gtest_filter_selects(test_name, ":".join(LARGE_MEMORY_TESTS))
            ]
            group_filters = [
                ":".join(
                    test_name
                    for test_name in selected_large_tests
                    if gtest_filter_selects(test_name, group_filter)
                )
                for group_filter in LARGE_MEMORY_GROUPS
            ]
        else:
            group_filters = LARGE_MEMORY_GROUPS
        for index, group_filter in enumerate(group_filters):
            if not group_filter:
                continue
            results = run_process_to_end(
                f"hipblaslt_test_shard_large{index}", group_filter, LARGE_MEMORY_ENV
            )
            if results and not results["run"]:
                failure_reasons.append(
                    f"large-memory group '{group_filter}' matches no test; "
                    "update LARGE_MEMORY_GROUPS"
                )
    finally:
        for process in started:
            if process.process.poll() is None:
                process.process.terminate()
        shutil.rmtree(xml_dir, ignore_errors=True)

    report_hipblaslt(finished, failure_reasons)
    return 0


def report_hipblaslt(finished, failure_reasons):
    """Print each process's log in a collapsible group, the per-process table and every
    failed test with its rerun command; raise CheckFailed with failure_reasons plus any
    crash or failed test."""
    failure_reasons = list(failure_reasons)
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
    failed_tests, memory_stops = [], []
    for process, exit_code, results in finished:
        if results is None:
            failure_reasons.append(
                f"{process.name} exited with {exit_code} and wrote no results: "
                f"see {process.log.name}"
            )
            rows.append(
                f"| {process.name} | {exit_code} | no results | | | {process.peak_memory_gib} | |"
            )
            continue
        # A crash after the last test: the XML has results but none failed.
        if exit_code and not results["failed"]:
            failure_reasons.append(
                f"{process.name} exited with {exit_code}: see {process.log.name}"
            )
        failed_tests += results["failed"]
        memory_stops += process.memory_stops
        rows.append(
            f"| {process.name} | {exit_code} | {results['run']} | {len(results['failed'])} | "
            f"{len(results['skipped'])} | {process.peak_memory_gib} | {len(process.memory_stops)} |"
        )
    summary("\n".join(rows))
    if memory_stops:
        summary("```\n" + "\n".join(memory_stops) + "\n```")
    if failed_tests:
        script = shlex.quote(os.path.relpath(__file__, REPO_ROOT))
        rerun_commands = [
            f"python3 {script} hipblaslt --gtest-filter={shlex.quote(test_name)}"
            for test_name in failed_tests
        ]
        summary(
            f"{len(failed_tests)} hipBLASLt tests failed. Rerun one as this job did, on a "
            "gfx1250 machine from the repo root:\n```\n"
            + "\n".join(sorted(set(rerun_commands)))
            + "\n```"
        )
        failure_reasons.append(f"{len(failed_tests)} hipBLASLt tests failed")
    if failure_reasons:
        raise CheckFailed("; ".join(failure_reasons))


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
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log = output_dir / "common_test.log"
    tensilelite_dir = HIPBLASLT_DIR / "tensilelite"
    # tox.ini's py3 command writes pytest's JUnit report here.
    junit = tensilelite_dir / "python_tests.xml"
    junit.unlink(missing_ok=True)
    rocm_path = os.environ["ROCM_PATH"]
    env = {
        **os.environ,
        "CC": f"{rocm_path}/lib/llvm/bin/amdclang",
        "CXX": f"{rocm_path}/lib/llvm/bin/amdclang++",
    }
    # As the nightly: tox builds the client for the reported target (gfx1250-strict
    # here) and runs the common tests against it. The pytest-xdist workers share this
    # GPU's client lock (conftest.py): the GPU runs take turns, but one test's kernel
    # compile overlaps another's GPU run. skip-gfx1250-strict configs are skipped on this
    # target (config_helpers.py). Each argument after -- stays a single word: tox inserts
    # them into tox.ini's sh -c command unquoted. -rfxX: list failed, xfailed and xpassed
    # tests at the end.
    command = ["tox", "-v", "-e", "py3", "--", "tensilelite/Tests", "-m", "common"]
    command += ["-k", args.pytest_k] if args.pytest_k else []
    command += [f"--timeout={args.timeout}", "-n", str(args.workers), "-rfxX"]
    print(
        f"Running: {shlex.join(command)} (in {tensilelite_dir}); log: {log}", flush=True
    )
    process = subprocess.Popen(
        command,
        cwd=tensilelite_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # One process: its output goes to the console as it runs, scrubbed.
    scrubber = Scrubber(timing="pytest")
    with open(log, "w", encoding="utf-8", errors="surrogateescape") as f:
        for raw in text_stream(process.stdout.fileno()):
            line = scrubber.scrub(raw.rstrip("\n"))
            if line is None:
                continue
            f.write(line + "\n")
            f.flush()
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
    exit_code = process.wait()

    if not junit.is_file():
        raise CheckFailed(
            f"tox exited with {exit_code} and wrote no {junit.name}: see {log.name}"
        )
    totals = parse_junit_xml(junit)
    write_public_xml(junit, output_dir / "common_tests.xml")
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
        default=str(HIPBLASLT_DIR / "build/release/clients/hipblaslt-test"),
        help="hipblaslt-test to run",
    )
    hipblaslt_parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "test_results"),
        help="directory for the scrubbed logs and XML reports",
    )
    # 4 keeps the GPU busy while other shards run their CPU reference checks.
    hipblaslt_parser.add_argument(
        "--shards", type=int, default=4, help="gtest shards run at once on the GPU"
    )

    tensilelite_parser = subcommands.add_parser(
        "tensilelite", help="run the TensileLite common tests via tox"
    )
    tensilelite_parser.add_argument(
        "--pytest-k", default="", help="pytest -k test-name expression, no spaces"
    )
    tensilelite_parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "tensilelite_test_results"),
        help="directory for the scrubbed log and XML report",
    )
    # tox.ini's default (TENSILE_NUM_PYTEST_WORKERS) is also 4.
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

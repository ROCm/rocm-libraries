#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx1250-strict hipBLASLt + TensileLite CI job's checks and test runs.

The workflow (gfx1250_hipblaslt_tensilelite_test.yml) installs the ROCm SDK, builds
hipBLASLt and runs these subcommands; on a gfx1250 machine the same commands reproduce
the job. From the repo root, with the SDK's environment (ROCM_PATH, PATH) set and
hipBLASLt built by ``install.sh -c -a gfx1250-strict --skip_rocroller``:

  python3 .github/scripts/gfx1250_blas_tests.py check-compiler
  python3 .github/scripts/gfx1250_blas_tests.py check-runtime
  python3 .github/scripts/gfx1250_blas_tests.py hipblaslt [--gtest-filter=FILTER]
  python3 .github/scripts/gfx1250_blas_tests.py tensilelite [--pytest-k=EXPR]

Every subcommand runs in gfx1250-strict mode (HSA_DISABLE_GFX12_STRICT=0). Test output
goes through scrub_test_log.py before it reaches the console or a log file, because the
job log is public. A failed hipBLASLt test is listed with the command that reruns it.
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
import urllib.request
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


def error(msg):
    print(f"::error::{msg}", flush=True)


def warning(msg):
    print(f"::warning::{msg}", flush=True)


def summary(text):
    """Print text and add it to the job's step summary."""
    print(text, flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text + "\n\n")


def gpus_per_host():
    """GPUs (one job each) sharing this machine's CPUs: GPUS_PER_HOST, default 1."""
    return max(1, int(os.environ.get("GPUS_PER_HOST", "1")))


def job_cpus():
    """CPUs this job may use. A runner that pins its container (--cpuset-cpus) already gives
    the job its GPU's share; otherwise the affinity mask is the whole host, split by GPU.
    """
    allowed = len(os.sched_getaffinity(0))
    if allowed * gpus_per_host() <= (os.cpu_count() or allowed):
        return allowed
    return max(1, allowed // gpus_per_host())


def in_filter(name, gtest_filter):
    """True if gtest would run test ``name`` under ``gtest_filter`` (POSITIVE[-NEGATIVE])."""
    pos, _, neg = gtest_filter.partition("-")

    def matches(patterns):
        return any(fnmatch.fnmatchcase(name, p) for p in patterns.split(":") if p)

    return matches(pos or "*") and not matches(neg)


# ---------------------------------------------------------------------------- checks


def check_compiler(_args):
    """Warn when the pinned SDK was built with a different compiler than per-PR CI uses
    (the amd-llvm at ci-env's therock-ref). Never fails: the pin stays fixed."""
    sdk = os.environ.get("ROCM_SDK_VERSION", "")
    try:
        action = (REPO / ".github/actions/ci-env/action.yml").read_text()
        ref = re.search(r'therock-ref:.*?value: *"([0-9a-f]{40})"', action, re.S)[1]
        token = os.environ.get("GH_TOKEN")
        req = urllib.request.Request(
            f"https://api.github.com/repos/ROCm/TheRock/contents/compiler/amd-llvm?ref={ref}",
            headers={"Authorization": f"Bearer {token}"} if token else {},
        )
        want = json.load(urllib.request.urlopen(req, timeout=60))["sha"]
        root = os.environ["ROCM_PATH"]
        manifests = [f"{root}/share/therock/therock_manifest.json"] + glob.glob(
            f"{os.path.dirname(root)}/_rocm_sdk_*/share/therock/therock_manifest.json"
        )
        with open(next(p for p in manifests if os.path.isfile(p))) as f:
            manifest = json.load(f)
        have = next(
            s.get("pin_sha", "")
            for s in manifest.get("submodules", [])
            if s.get("submodule_name") == "llvm-project"
        )
        built = manifest.get("the_rock_commit", "")
    except Exception as e:
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
        error(
            f"ROCR_VISIBLE_DEVICES='{gpu}' must be a single GPU index: set it in the runner's .env"
        )
        return 1
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
        error(f"runtime reports '{arch or 'no GPU'}', not {ARCH}")
        return 1
    print(f"GPU {gpu}: {arch}")
    return 0


def check_build(binary, install):
    """hipblaslt-test must link the libhipblaslt built here, not the SDK's copy, and the
    library next to it must hold code objects labelled gfx1250-strict: with a stale
    bundler the label is empty and no GEMM kernel loads."""
    ldd = subprocess.run(["ldd", str(binary)], capture_output=True, text=True).stdout
    lib = next(
        (
            l.split()[2]
            for l in ldd.splitlines()
            if "libhipblaslt.so" in l and len(l.split()) > 2
        ),
        "",
    )
    if not lib or not any(
        Path(lib).resolve().is_relative_to(d) for d in (install, HIPBLASLT / "build")
    ):
        error(
            f"hipblaslt-test links '{lib or 'nothing'}', not the libhipblaslt built here"
        )
        return False
    cos = sorted(
        Path(lib).parent.glob(f"hipblaslt/library/{ARCH}/**/TensileLibrary*.co")
    )
    if not cos:
        error(f"the build produced no {ARCH} hipBLASLt library")
        return False
    bundler = Path(os.environ["ROCM_PATH"]) / "lib/llvm/bin/clang-offload-bundler"
    ids = subprocess.run(
        [str(bundler), "--list", "--type=o", f"--input={cos[0]}"],
        capture_output=True,
        text=True,
    )
    if ids.returncode:
        error(
            "cannot list a hipBLASLt code object with the SDK's clang-offload-bundler"
        )
        return False
    if not any(i.endswith(f"-{ARCH}") for i in ids.stdout.split()):
        error(f"hipBLASLt code objects are not labelled {ARCH}")
        return False
    return True


# ------------------------------------------------------------------------- hipBLASLt


class GtestRun:
    """One hipblaslt-test process. Its output is scrubbed line by line, written to
    ``<out>/<name>.log`` and echoed as ``[name] line``; results come from gtest's XML
    report, which stays in a private directory (it holds timings)."""

    _print_lock = threading.Lock()

    def __init__(self, name, binary, args, env, out, xml_dir):
        self.name = name
        self.log = out / f"{name}.log"
        self.xml = Path(xml_dir) / f"{name}.xml"
        self.peak_rss, self.memory_stops = "", []
        self.proc = subprocess.Popen(
            [str(binary), *args, f"--gtest_output=xml:{self.xml}"],
            env={**os.environ, **env},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._reader = threading.Thread(target=self._stream, daemon=True)
        self._reader.start()

    def _stream(self):
        scrubber = Scrubber()
        with open(self.log, "w", encoding="utf-8", errors="surrogateescape") as log:
            for raw in text_stream(self.proc.stdout.fileno()):
                line = scrubber.scrub(raw.rstrip("\n"))
                if line is None:
                    continue
                log.write(line + "\n")
                log.flush()
                # hipblaslt-test's own end-of-run lines: peak memory, and one line per
                # test it stopped for lack of memory (after a count line).
                if line.startswith("[ PEAK RSS ] "):
                    self.peak_rss = line.split()[4]
                elif re.match(r"\[ MEMORY   \] [^0-9]", line):
                    self.memory_stops.append(line[len("[ MEMORY   ] ") :])
                with self._print_lock:
                    sys.stdout.write(f"[{self.name}] {line}\n")
                    sys.stdout.flush()

    def wait(self):
        """Wait for the process; return (exit code, results or None if it wrote none)."""
        rc = self.proc.wait()
        self._reader.join()
        try:
            root = ET.parse(self.xml).getroot()
        except (OSError, ET.ParseError):
            return rc, None
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
                results["skipped"][name] = (
                    "" if skip is None else skip.get("message", "")
                )
        return rc, results


def list_tests(binary, gtest_filter):
    """Full names of the tests gtest selects for gtest_filter."""
    out = subprocess.run(
        [str(binary), "--gtest_list_tests", f"--gtest_filter={gtest_filter}"],
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    names, suite = [], ""
    for line in out.splitlines():
        if not line.startswith(" ") and line.endswith("."):
            suite = line.split()[0]
        elif line.startswith("  ") and suite:
            names.append(suite + line.split()[0])
    return names


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
    if not check_build(binary, install):
        return 1
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    xml_dir = tempfile.mkdtemp(
        prefix="hipblaslt-gtest-", dir=os.environ.get("RUNNER_TEMP")
    )
    started, finished = [], []  # GtestRun; (GtestRun, exit code, results)
    ok = True

    def start(name, gtest_filter, env):
        started.append(
            GtestRun(
                name, binary, [f"--gtest_filter={gtest_filter}"], env, out, xml_dir
            )
        )
        return started[-1]

    def run(name, gtest_filter, env):
        r = start(name, gtest_filter, env)
        finished.append((r, *r.wait()))
        return finished[-1][2]

    try:
        # Shards in parallel: one shard's CPU reference checks overlap another's GPU work.
        pos, _, neg = user_filter.partition("-")
        negatives = ":".join(([neg] if neg else []) + LARGE + [SK_DEADLOCK])
        shard_filter = f"{pos or '*'}-{negatives}"
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
                error(f"{SK_DEADLOCK} matches no test; update SK_DEADLOCK")
                ok = False
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
        for k, group in enumerate(groups):
            if not group:
                continue
            results = run(f"hipblaslt_test_shard_large{k}", group, LARGE_ENV)
            # A filter that matches nothing runs nothing and passes, so a renamed test
            # would silently drop out of the run.
            if results and not results["run"]:
                error(
                    f"large-memory group '{group}' matches no test; update LARGE_GROUPS"
                )
                ok = False
    finally:
        for r in started:
            if r.proc.poll() is None:
                r.proc.terminate()
        shutil.rmtree(xml_dir, ignore_errors=True)

    return 0 if report_hipblaslt(finished) and ok else 1


def report_hipblaslt(finished):
    """Print the per-process table and every failure with its rerun command; True if all
    processes passed."""
    rows = [
        "### hipBLASLt test processes",
        "| Process | Ran | Failed | Skipped | Peak resident memory (GiB) | Tests stopped for memory |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    failed, memory_stops, ok = [], [], True
    for r, rc, results in finished:
        if results is None:
            error(f"{r.name} exited with {rc} and wrote no results: see {r.log.name}")
            rows.append(f"| {r.name} | exit {rc}, no results | | | {r.peak_rss} | |")
            ok = False
            continue
        if rc and not results["failed"]:
            error(f"{r.name} exited with {rc}: see {r.log.name}")
        ok = ok and rc == 0
        failed += results["failed"]
        memory_stops += r.memory_stops
        rows.append(
            f"| {r.name} | {results['run']} | {len(results['failed'])} | "
            f"{len(results['skipped'])} | {r.peak_rss} | {len(r.memory_stops)} |"
        )
    summary("\n".join(rows))
    if memory_stops:
        summary("```\n" + "\n".join(memory_stops) + "\n```")
    if failed:
        error(f"{len(failed)} hipBLASLt tests failed")
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
    return ok


# ------------------------------------------------------------------------ TensileLite


def run_tensilelite(args):
    if re.search(r"\s", args.pytest_k):
        error(
            "--pytest-k must not contain spaces (tox splits them into separate arguments)"
        )
        return 1
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / "common_test.log"
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
    cmd = ["tox", "-v", "-e", "py3", "--", "tensilelite/Tests", "-m", "common"]
    cmd += ["-k", args.pytest_k] if args.pytest_k else []
    cmd += [f"--timeout={args.timeout}", "-n", str(args.workers), "-rfxX"]
    proc = subprocess.Popen(
        cmd,
        cwd=HIPBLASLT / "tensilelite",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    scrubber = Scrubber(timing="pytest")
    no_tests = False
    with open(log, "w", encoding="utf-8", errors="surrogateescape") as f:
        for raw in text_stream(proc.stdout.fileno()):
            line = scrubber.scrub(raw.rstrip("\n"))
            if line is None:
                continue
            no_tests = no_tests or bool(
                re.search(r"workers \[0 items\]|no tests ran", line)
            )
            f.write(line + "\n")
            f.flush()
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
    rc = proc.wait()
    # tox.ini maps pytest's "no tests collected" (exit 5) to success.
    if no_tests:
        error("TensileLite ran no tests")
        return 1
    return rc


# ------------------------------------------------------------------------------- main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-compiler", help=check_compiler.__doc__.splitlines()[0])
    sub.add_parser("check-runtime", help=check_runtime.__doc__.splitlines()[0])
    hip = sub.add_parser("hipblaslt", help="run the hipBLASLt gtest suite")
    hip.add_argument(
        "--gtest-filter", default="", help="gtest filter; empty runs everything"
    )
    hip.add_argument(
        "--binary", default=str(HIPBLASLT / "build/release/clients/hipblaslt-test")
    )
    hip.add_argument("--out", default=str(REPO / "test_results"))
    hip.add_argument(
        "--shards", type=int, default=4, help="gtest shards run at once on the GPU"
    )
    tl = sub.add_parser("tensilelite", help="run the TensileLite common tests via tox")
    tl.add_argument("--pytest-k", default="", help="pytest -k expression, no spaces")
    tl.add_argument("--out", default=str(REPO / "tensilelite_test_results"))
    tl.add_argument("--workers", type=int, default=4, help="pytest-xdist workers")
    # decouple_pgr_tdm_fuse_gfx1250.yaml's sweep runs past 600 s.
    tl.add_argument("--timeout", type=int, default=1200, help="per-test timeout (s)")
    args = parser.parse_args(argv)

    os.environ["HSA_DISABLE_GFX12_STRICT"] = "0"
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(encoding="utf-8", errors="surrogateescape")
    command = {
        "check-compiler": check_compiler,
        "check-runtime": check_runtime,
        "hipblaslt": run_hipblaslt,
        "tensilelite": run_tensilelite,
    }[args.command]
    return command(args)


if __name__ == "__main__":
    sys.exit(main())

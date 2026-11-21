#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run a TensileLite test command under the rocjitsu emulator (CPU only).

Usage: run_rocjitsu.py -- <command...>
TheRock's tensilelite-rocjitsu job runs ``-- python pytest_runner.py`` with
TEST_CATEGORY=rocjitsu-quick. This script:
- picks the device config for AMDGPU_TARGETS (or the AMDGPU_FAMILIES family)
  from the rocjitsu installed with ROCm;
- splits the CPUs it may use (affinity, capped by a cgroup CPU quota) between
  pytest-xdist workers and each worker's emulator threads, with no more workers
  than the memory (cgroup limit, else physical) holds, and passes the worker
  count, per-test timeout and stall-dump time to pytest_runner.py / the
  rocjitsu_pytest plugin through the environment;
- runs ``rocjitsu --config ... -- <command>`` (in-process simulation) as a
  child subreaper with a suite timeout, so timed-out tests' clients never
  outlive the job step;
- fails a run that finished no passing test, and prints the result summary and
  any stall dumps (the CI job uploads no files); with GITHUB_STEP_SUMMARY set,
  also lists the failed, timed-out and unfinished tests in the job summary.

Environment: THEROCK_BIN_DIR (its parent is the ROCm install; else ROCM_PATH,
else this install tree),
AMDGPU_TARGETS / AMDGPU_FAMILIES, optional ROCJITSU_REPORT_DIR,
ROCJITSU_SUITE_TIMEOUT / ROCJITSU_SUITE_GRACE (seconds), PYTEST_TEST_TIMEOUT,
PYTEST_NUM_WORKERS, SHARD_INDEX / TOTAL_SHARDS, GITHUB_STEP_SUMMARY.
"""

import contextlib
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent

# Emulated target -> device config under share/rocjitsu/configs.
TARGETS = {
    "gfx942": "gfx942_cdna3_kmd.json",
    "gfx950": "gfx950_mi355x_kmd.json",
    "gfx1250": "gfx1250_mi455x.json",
}
FAMILY_TARGETS = {"gfx94": "gfx942", "gfx950": "gfx950", "gfx125": "gfx1250"}

MAX_WORKERS = 16
THREADS_PER_WORKER = 6
# 16 workers peaked at 151 GB on a 96-core host, about 9.4 GiB each.
MEMORY_PER_WORKER = 10 * 2**30
# Per test: rocjitsu-quick leaves out configs over 45 min (rocjitsu_slow).
DEFAULT_TEST_TIMEOUT = 45 * 60
# A ~40-min shard plus one test hung to its timeout, inside TheRock's 100-min
# step with time left to reap, summarize and exit.
DEFAULT_SUITE_TIMEOUT = 90 * 60
STALL_LEAD = 300


def rocm_root(env):
    # THEROCK_BIN_DIR first, like pytest_runner.py: it names the install the
    # CI job fetched, while ROCM_PATH may point at an unrelated ROCm.
    if env.get("THEROCK_BIN_DIR"):
        return Path(env["THEROCK_BIN_DIR"]).resolve().parent
    if env.get("ROCM_PATH"):
        return Path(env["ROCM_PATH"])
    # <rocm>/share/hipblaslt/tensilelite/tensilelite/Tests/emulation
    return HERE.parents[5]


def emulated_target(env):
    targets = [t for t in env.get("AMDGPU_TARGETS", "").split(",") if t]
    if targets:
        target = targets[0]
    else:
        family = env.get("AMDGPU_FAMILIES", "")
        target = next(
            (t for prefix, t in FAMILY_TARGETS.items() if family.startswith(prefix)),
            "",
        )
    if target not in TARGETS:
        raise SystemExit(f"rocjitsu has no device config for target {target!r}")
    return target


def cgroup_limits(root=Path("/sys/fs/cgroup")):
    """(CPUs, memory bytes) a cgroup v2 or v1 quota allows; None where unlimited.

    A container's CPU quota does not show in its CPU affinity, so sizing by
    affinity alone oversubscribes a quota-limited runner.
    """

    def read(name):
        try:
            return (root / name).read_text().split()
        except OSError:
            return None

    cpus = memory = None
    if cpu_max := read("cpu.max"):
        if cpu_max[0] != "max":
            cpus = max(1, int(cpu_max[0]) // int(cpu_max[1]))
    elif (quota := read("cpu/cpu.cfs_quota_us")) and (
        period := read("cpu/cpu.cfs_period_us")
    ):
        if int(quota[0]) > 0:
            cpus = max(1, int(quota[0]) // int(period[0]))
    limit = read("memory.max") or read("memory/memory.limit_in_bytes")
    # cgroup v1 reports "unlimited" as a value near 2**63.
    if limit and limit[0] != "max" and int(limit[0]) < 2**60:
        memory = int(limit[0])
    return cpus, memory


def split_cpus(cores, memory=None):
    """(xdist workers, emulator threads per worker) for these CPUs and memory."""
    workers = min(MAX_WORKERS, cores // THREADS_PER_WORKER)
    if memory:
        workers = min(workers, memory // MEMORY_PER_WORKER)
    workers = max(1, workers)
    return workers, max(1, cores // workers)


def write_config(source, destination, budget):
    """Copy the device config with this host's thread budget; return its CU count."""
    config = json.loads(Path(source).read_text())
    config["cpu_thread_budget"] = budget
    # One engine thread per emulator: the only setting validated on the full
    # gfx1250 set.
    config["num_threads"] = 1
    Path(destination).write_text(json.dumps(config, indent=2))
    device = config["vm"]["gpu"]["device"]
    return device["simd_count"] // device["simd_per_cu"]


def llvm_lib_dirs(rocm):
    """The compiler runtime (libomp) is flat or under a host-triple directory."""
    base = rocm / "lib" / "llvm" / "lib"
    triple = base / f"{os.uname().machine}-unknown-linux-gnu"
    return [str(base)] + ([str(triple)] if (triple / "libomp.so").exists() else [])


def prepend(env, name, *paths):
    env[name] = os.pathsep.join([*paths, *filter(None, [env.get(name)])])


def emulation_setup(env, cores, report_dir, memory=None):
    """Return (rocjitsu command prefix, environment) for running a test command."""
    rocm = rocm_root(env)
    target = emulated_target(env)
    rocjitsu = rocm / "bin" / "rocjitsu"
    config = rocm / "share" / "rocjitsu" / "configs" / TARGETS[target]
    for required in (rocjitsu, config):
        if not required.exists():
            raise SystemExit(f"missing from the ROCm install: {required}")

    workers, budget = split_cpus(cores, memory)
    report_dir.mkdir(parents=True, exist_ok=True)
    run_config = report_dir / "rocjitsu-config.json"
    compute_units = write_config(config, run_config, budget)
    test_timeout = int(env.get("PYTEST_TEST_TIMEOUT") or DEFAULT_TEST_TIMEOUT)

    env = dict(env)
    env.update(
        ROCM_PATH=str(rocm),
        # Tensile asks rocminfo for the CU count unless CU is set; rocminfo
        # needs the amdgpu driver, which rocjitsu does not emulate.
        CU=str(compute_units),
        # The client's CPU reference would otherwise ask OpenMP for 64 threads.
        OMP_NUM_THREADS=str(budget),
        OMP_THREAD_LIMIT=str(budget),
        PYTEST_NUM_WORKERS=env.get("PYTEST_NUM_WORKERS") or str(workers),
        PYTEST_TEST_TIMEOUT=str(test_timeout),
        ROCJITSU_REPORT_DIR=str(report_dir),
        ROCJITSU_STALL_DUMP_AFTER=str(max(1, test_timeout - STALL_LEAD)),
        # Private dir for rocjitsu's per-process files (it reaps stale ones).
        ROCJITSU_RUNTIME_DIR=tempfile.mkdtemp(prefix="rocjitsu-"),
    )
    prepend(env, "PYTHONPATH", str(HERE))
    prepend(env, "PATH", str(rocm / "bin"), str(rocm / "lib" / "llvm" / "bin"))
    prepend(
        env,
        "LD_LIBRARY_PATH",
        str(rocm / "lib"),
        str(rocm / "lib" / "rocm_sysdeps" / "lib"),
        *llvm_lib_dirs(rocm),
    )
    print(
        f"rocjitsu: target={target} config={TARGETS[target]} cores={cores} "
        f"memory_gib={memory // 2**30 if memory else None} "
        f"workers={env['PYTEST_NUM_WORKERS']} threads_per_worker={budget} "
        f"cu={compute_units} shard={env.get('SHARD_INDEX', 1)}/{env.get('TOTAL_SHARDS', 1)}",
        flush=True,
    )
    # Without --daemon, rocjitsu preloads the simulator into the command itself.
    return [str(rocjitsu), "--config", str(run_config), "--"], env


def reap_descendants():
    """Kill and reap only children owned/adopted by this supervisor."""
    children_file = Path(f"/proc/self/task/{os.getpid()}/children")
    deadline = time.monotonic() + 10
    while True:
        children = [int(pid) for pid in children_file.read_text().split()]
        if not children:
            return
        for pid in children:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        # Killing an intermediate parent reparents its descendants to us.
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == 0:
                break
        if time.monotonic() >= deadline:
            raise RuntimeError("emulation descendants did not exit after SIGKILL")
        time.sleep(0.01)


def supervise(command, env, timeout, grace):
    if sys.platform != "linux":
        raise RuntimeError("rocjitsu emulation runs require Linux")
    # PR_SET_CHILD_SUBREAPER applies only to this process and needs no privilege.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")

    def interrupt(signum, frame):
        raise InterruptedError(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupt)
    start = time.monotonic()
    timed_out = False
    proc = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        try:
            status = proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, InterruptedError) as error:
            timed_out = isinstance(error, subprocess.TimeoutExpired)
            status = 124 if timed_out else 128 + error.args[0]
            # Give pytest time to stop workers and write its normal JUnit file.
            for signum in (signal.SIGTERM, signal.SIGINT):
                signal.signal(signum, signal.SIG_IGN)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
    finally:
        # This also handles a successful parent that left children behind.
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, signal.SIG_IGN)
        reap_descendants()
    return dict(
        exit_code=status if status >= 0 else 128 - status,
        timed_out=timed_out,
        wall_seconds=time.monotonic() - start,
    )


def not_passed(progress):
    """{"failed" | "timed out" | "incomplete": [node ids]} from progress.json."""
    groups = {"failed": [], "timed out": [], "incomplete": []}
    for nodeid, test in progress.get("tests", {}).items():
        if test["outcome"] == "failed":
            detail = str(test["phases"].get("call", {}).get("detail"))
            # pytest-timeout ends its failure message with this.
            groups["timed out" if "from pytest-timeout" in detail else "failed"].append(nodeid)
        elif test["outcome"] == "incomplete":
            groups["incomplete"].append(nodeid)
    return groups


def step_summary(title, progress, result, groups):
    """Markdown for the CI job summary: counts, then every test that did not pass."""
    status = "passed" if result["exit_code"] == 0 else "failed"
    lines = [
        f"### {title}: {status}",
        "",
        "| selected | passed | failed | timed out | skipped | incomplete | minutes |",
        "|---|---|---|---|---|---|---|",
        "| {} | {} | {} | {} | {} | {} | {:.0f} |".format(
            progress.get("selected", 0),
            progress.get("passed", 0),
            len(groups["failed"]),
            len(groups["timed out"]),
            progress.get("skipped", 0),
            len(groups["incomplete"]),
            result.get("wall_seconds", 0) / 60,
        ),
    ]
    if result.get("error"):
        lines += ["", f"Error: {result['error']}"]
    for name, nodeids in groups.items():
        if nodeids:
            lines += ["", f"{name.capitalize()}:", *(f"- `{nodeid}`" for nodeid in nodeids)]
    return "\n".join(lines) + "\n"


def summarize(report_dir, result, title="rocjitsu", summary_path=None):
    """Fail runs that passed nothing; print the summary and any stall dumps.

    With summary_path (the job's GITHUB_STEP_SUMMARY), also append the counts
    and the tests that failed, timed out or never finished.
    """
    progress_path = report_dir / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    result["complete"] = bool(progress.get("selected")) and (
        progress.get("selected") == progress.get("finished") and not result["timed_out"]
    )
    if result["exit_code"] == 0 and (
        not result["complete"] or progress.get("passed", 0) == 0
    ):
        result["exit_code"] = 1
        result["error"] = "pytest did not complete any passing emulation tests"
    (report_dir / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
    counts = {key: value for key, value in progress.items() if key != "tests"}
    print(f"rocjitsu pytest results: {counts}", flush=True)
    groups = not_passed(progress)
    for name, nodeids in groups.items():
        for nodeid in nodeids:
            print(f"  {name}: {nodeid}", flush=True)
    for stall in sorted((report_dir / "stalls").glob("*.txt")):
        print(f"--- stall dump {stall.name} ---\n{stall.read_text()}", flush=True)
    print(f"rocjitsu pytest execution: {result}", flush=True)
    if summary_path:
        try:
            with open(summary_path, "a") as summary:
                summary.write(step_summary(title, progress, result, groups))
        except OSError as error:
            print(f"rocjitsu: could not write the job summary: {error}", flush=True)
    return result["exit_code"]


def main(argv=None, env=None):
    argv = sys.argv[1:] if argv is None else argv
    env = dict(os.environ if env is None else env)
    command = argv[1:] if argv[:1] == ["--"] else argv
    if not command:
        raise SystemExit("usage: run_rocjitsu.py -- <command...>")
    report_dir = Path(env.get("ROCJITSU_REPORT_DIR") or "rocjitsu-report").resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    # A reused report directory must never turn an interrupted run into a pass.
    for name in ("progress.json", "execution.json"):
        (report_dir / name).unlink(missing_ok=True)
    cores = len(os.sched_getaffinity(0))
    quota_cpus, memory = cgroup_limits()
    if quota_cpus:
        cores = min(cores, quota_cpus)
    if not memory:
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    prefix, run_env = emulation_setup(env, cores, report_dir, memory)
    timeout = float(env.get("ROCJITSU_SUITE_TIMEOUT") or DEFAULT_SUITE_TIMEOUT)
    try:
        grace = float(env.get("ROCJITSU_SUITE_GRACE") or 30)
        result = supervise([*prefix, *command], run_env, timeout, grace)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        result = dict(exit_code=1, timed_out=False, error=str(error))
    title = "rocjitsu {} shard {}/{}".format(
        emulated_target(env), env.get("SHARD_INDEX", 1), env.get("TOTAL_SHARDS", 1)
    )
    return summarize(report_dir, result, title, env.get("GITHUB_STEP_SUMMARY"))


if __name__ == "__main__":
    sys.exit(main())

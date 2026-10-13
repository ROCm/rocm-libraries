#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run a Tensile config with tensilelite-client on one or more GPUs at once while agt_internal logs PM telemetry, and
report each GPU's throughput and GFXCLK over the client's timed window.

usage: run_tensile_with_agt.py <config.yaml> <gpus> <hipblaslt> <agt_internal> [options]

  config.yaml   Tensile config, e.g. mxf4_tn_maf.yaml in this directory
  gpus          HIP device indices to run on, all at the same time: 0 | 0,1 | 0-3 | all
  hipblaslt     hipBLASLt source dir (projects/hipblaslt). Its tensilelite/Tensile/bin/Tensile builds the config, and
                the tensilelite-client that `invoke build-client` puts in tensilelite/build_tmp runs it. Run this
                script with the Python environment Tensile needs (rocisa etc.).
  agt_internal  AGT binary; it runs as root through `sudo -n`

What it does:
  1. Tensile --build-only builds the config into <out>/build/ (reused while the config, Tensile and Python are the
     same; --rebuild forces a new build).
  2. agt_internal logs the packages of the selected GPUs every 50 ms into <run dir>/unilog.csv.
  3. One tensilelite-client per GPU loads and initializes on its own, then waits at a release gate (agt_gate.c,
     preloaded) before its first kernel launch. Once every client waits they are released together, so their
     warmups and timed windows line up and no client's setup runs during another's window. A batch is not released
     while another process has GPU queues on those packages.
  4. A client's timed window is [enqueue-time - launches x time-us, enqueue-time]: the client stamps enqueue-time
     right after the sync that closes its timed launches. If the shortest window is under --min-window, the batch
     is run again with num-enqueues-per-sync scaled up; the short batch is kept in the outputs but not in the table.
  5. Per GPU: time-us, PFLOPS and TB/s from the client and, from the AGT samples wholly inside the window after its
     first 200 ms (while the power limit settles): the mean GFXCLK of the GPU's XCDs (Pre Deep Sleep Freq), their
     target clock, the PPT_0 GFXCLK limit, package power, busy % of the GPU's XCDs and of the package's other XCDs,
     and the share of the window that every other GPU's window covered.

The client runs the build's ini, except: device-idx 0 (each client sees only its GPU), results-file, no validation,
one benchmark of one sync (the syncs' launches merged into it), num-enqueues-per-sync set exactly (no
min-flops-per-sync or max-enqueues-per-sync adjustment), and --enqueues or --min-window scaling.
GPU indices are HIP's (KFD topology order). Each GPU's AGT instance is found by PCI address in `agt_internal -i`; when
a package is partitioned (e.g. DPX), its k-th GPU owns XCDs k x num_xcc to (k + 1) x num_xcc - 1.
Outputs in <out>/<config>.gpu<list>.<time>/: unilog.csv, run<i>.gpu<d>.{ini,log,csv}, runs.tsv, summary.csv and
summary.md. Exit status 0 when every run completed.
"""

import argparse
import bisect
import csv
import ctypes
import datetime
import fcntl
import glob
import hashlib
import math
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GATE_SRC = os.path.join(HERE, "agt_gate.c")
KFD = "/sys/class/kfd/kfd"
PERIOD = 0.05     # AGT sampling period (its minimum), s
ONSET = 0.2       # start of a window left out while PPT_0 settles, s
NO_GPU = "GPU-ffffffffffffffff"   # matches no GPU: what a client sees if the gate is not loaded
VISIBILITY = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
GATED = ("hipExtModuleLaunchKernel", "hipModuleLaunchKernel", "hipDrvLaunchKernelEx")
LOCK = "/tmp/run_tensile_with_agt.lock"
RESULT_ROW = re.compile(r"^\d+,\d+/\d+,\d+/\d+,")
STOP = []


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def say(msg):
    print(f"{now()[11:19]} {msg}", flush=True)


def fail(msg):
    raise SystemExit(f"error: {msg}")


def tail(path, n=15):
    try:
        return "".join(open(path, errors="replace").readlines()[-n:])
    except OSError:
        return ""


# ---------------------------------------------------------------------------------------------------- topology

def kfd_gpus():
    """HIP devices in HIP order (KFD topology node order, CPU nodes skipped)."""
    out = []
    root = f"{KFD}/topology/nodes"
    for n in sorted(os.listdir(root), key=int):
        try:
            gpu_id = int(open(f"{root}/{n}/gpu_id").read())
        except (OSError, ValueError):
            continue
        if not gpu_id:
            continue
        props = {}
        for line in open(f"{root}/{n}/properties"):
            f = line.split()
            if len(f) == 2 and f[1].isdigit():
                props[f[0]] = int(f[1])
        loc, uid = props.get("location_id", 0), props.get("unique_id", 0)
        out.append({"index": len(out), "gpu_id": gpu_id,
                    "uuid": f"GPU-{uid:016x}" if uid else str(len(out)),
                    "bdf": f"{props.get('domain', 0):04x}:{loc >> 8:02x}:{(loc >> 3) & 0x1f:02x}.{loc & 7:x}",
                    "xcc": max(1, props.get("num_xcc", 1)),
                    "cu": props.get("simd_count", 0) // max(1, props.get("simd_per_cu", 1))})
    return out


def parse_gpus(text, count):
    if text.strip().lower() == "all":
        return list(range(count))
    out = []
    for part in text.split(","):
        m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", part)
        if not m or int(m.group(2) or m.group(1)) < int(m.group(1)):
            fail(f"bad GPU list {text!r}: use e.g. 0 | 0,1 | 0-3 | all")
        out += range(int(m.group(1)), int(m.group(2) or m.group(1)) + 1)
    if len(set(out)) != len(out):
        fail(f"GPU list {text!r} names a GPU twice")
    if max(out) >= count:
        fail(f"GPU {max(out)} does not exist: this node has {count} GPUs (0-{count - 1})")
    return sorted(out)


def agt_gpu_instances(agt):
    """{PCI address: AGT instance} of the GPUs `agt_internal -i` lists, and its output."""
    r = subprocess.run(["sudo", "-n", agt, "-i"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, errors="replace", timeout=300)
    found = {}
    for line in r.stdout.splitlines():
        m = re.match(r"\s*(\d+)\s.*\(([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])\)", line, re.I)
        if m:
            found[m.group(2).lower()] = int(m.group(1))
    return found, r.stdout


def layout(gpus, instances, selected):
    """Per selected GPU: its AGT instance, its XCDs and the other XCDs of its package. A package is what AGT logs as
    one instance; the k-th GPU on it owns the k-th block of num_xcc XCDs."""
    inst_of = {}
    for g in gpus:
        inst = instances.get(g["bdf"])
        if inst is None:   # a partition that does not carry the package's PCI address: match the PCI domain
            inst = next((i for b, i in sorted(instances.items()) if b[:4] == g["bdf"][:4]), None)
        inst_of[g["index"]] = inst
    out = {}
    for d in selected:
        inst = inst_of[d]
        if inst is None:
            fail(f"GPU {d} ({gpus[d]['bdf']}) is not in agt_internal's device list")
        members = [g for g in gpus if inst_of[g["index"]] == inst]
        k = [g["index"] for g in members].index(d)
        first = sum(g["xcc"] for g in members[:k])
        own = list(range(first, first + gpus[d]["xcc"]))
        total = sum(g["xcc"] for g in members)
        out[d] = {"inst": inst, "xcds": own, "other": [x for x in range(total) if x not in own],
                  "package": [g["index"] for g in members]}
    return out


def kfd_error():
    """Why no GPU program can start here, or None. Opening /dev/kfd fails, e.g. with EINVAL while KFD is locked by a
    GPU reset; the check runs in a child so this process never holds a KFD context."""
    r = subprocess.run([sys.executable, "-c", "import os; os.close(os.open('/dev/kfd', os.O_RDWR))"],
                       stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return None if r.returncode == 0 else (r.stdout.strip().splitlines() or [f"exit {r.returncode}"])[-1]


def queue_gpus(pid):
    """KFD gpu_ids of the GPU queues process pid has open (from /sys/class/kfd only)."""
    out = set()
    try:
        for q in os.listdir(f"{KFD}/proc/{pid}/queues"):
            try:
                out.add(int(open(f"{KFD}/proc/{pid}/queues/{q}/gpuid").read()))
            except (OSError, ValueError):
                pass
    except OSError:
        pass
    return out


def other_gpu_users(ours, gpu_ids):
    """Pids of other processes with GPU queues on any of gpu_ids."""
    try:
        pids = [p for p in os.listdir(f"{KFD}/proc") if p.isdigit()]
    except OSError:
        return ["kfd-unreadable"]
    return sorted(p for p in pids if int(p) not in ours and queue_gpus(p) & gpu_ids)


# ---------------------------------------------------------------------------------------------------- build

def tree_fingerprint(tl):
    """Hash of the paths, sizes and mtimes of the Tensile and rocisa sources."""
    h = hashlib.sha256()
    for sub in ("Tensile", "rocisa"):
        for dirpath, dirnames, filenames in os.walk(os.path.join(tl, sub)):
            dirnames[:] = sorted(d for d in dirnames if d not in ("__pycache__", "Tests", "build", ".git")
                                 and not d.endswith(".egg-info"))
            for name in sorted(filenames):
                path = os.path.join(dirpath, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                h.update(f"{os.path.relpath(path, tl)}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def find_inis(build_dir):
    return glob.glob(f"{build_dir}/1_BenchmarkProblems/*/00_Final/caches/*/source/ClientParameters.ini")


def tensile_build(config, tl, client, root, rebuild):
    """ClientParameters.ini of the config's Tensile --build-only output, building it when needed."""
    key = hashlib.sha256(b"\0".join([open(config, "rb").read(), os.path.realpath(tl).encode(),
                                     sys.executable.encode(), tree_fingerprint(tl).encode()])).hexdigest()[:12]
    build_dir = os.path.join(root, "build", f"{os.path.splitext(os.path.basename(config))[0]}-{key}")
    log = f"{build_dir}.log"
    if os.path.exists(f"{build_dir}/build.done") and not rebuild:
        say(f"reusing the build in {build_dir}")
    else:
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        os.makedirs(build_dir)
        say(f"building with Tensile --build-only into {build_dir} (log {log})")
        cmd = ["nice", "-n", "10", sys.executable, f"{tl}/Tensile/bin/Tensile", config, build_dir, "--build-only",
               "--prebuilt-client", client]
        with open(log, "w") as f:
            rc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc != 0 or len(find_inis(build_dir)) != 1:
            fail(f"Tensile --build-only failed (rc {rc}); the end of {log}:\n{tail(log, 25)}")
        open(f"{build_dir}/build.done", "w").write(f"{now()}\n")
    inis = find_inis(build_dir)
    if len(inis) != 1:
        fail(f"{build_dir} has {len(inis)} client inis; this script runs configs with one benchmark problem")
    return inis[0]


def build_gate(root):
    """agt_gate.c compiled into <out>/gate/, named after its source hash."""
    src = open(GATE_SRC, "rb").read()
    so = os.path.join(root, "gate", f"agt_gate-{hashlib.sha256(src).hexdigest()[:12]}.so")
    if not os.path.exists(so):
        os.makedirs(os.path.dirname(so), exist_ok=True)
        tmp = f"{so}.{os.getpid()}.tmp"
        r = subprocess.run([os.environ.get("CC", "cc"), "-O2", "-Wall", "-fPIC", "-shared", "-o", tmp, GATE_SRC,
                            "-ldl", "-lpthread"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode:
            fail(f"cannot compile {GATE_SRC}:\n{r.stdout}")
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    missing = [s for s in GATED if not hasattr(lib, s)]
    if missing:
        fail(f"{so} does not export {missing}")
    return so


# ---------------------------------------------------------------------------------------------------- client ini

def ini_value(text, key):
    m = re.search(rf"^{re.escape(key)}=(.*)$", text, re.M)
    return m.group(1).strip() if m else None


def edit_ini(text, changes):
    for key, value in changes.items():
        text, n = re.subn(rf"^{re.escape(key)}=.*$", f"{key}={value}", text, flags=re.M)
        if n == 0:
            text = text.rstrip("\n") + f"\n{key}={value}\n"
        elif n > 1:
            fail(f"{key} appears {n} times in the client ini")
    return text


def timing_ini(text, enqueues):
    """The build's ini made into one benchmark of one sync of exactly `enqueues` launches, without validation."""
    return edit_ini(text, {"device-idx": 0, "num-elements-to-validate": 0, "num-benchmarks": 1,
                           "num-syncs-per-benchmark": 1, "num-enqueues-per-sync": enqueues,
                           "min-flops-per-sync": 0, "max-enqueues-per-sync": -1})


# ---------------------------------------------------------------------------------------------------- processes

class Agt:
    """agt_internal PM logging through sudo, without a terminal; stopped through -unilogstopcheck."""

    def __init__(self, exe, instances, rundir, max_seconds):
        self.dir = os.path.join(rundir, "agt")   # its working directory, where -unilogstopcheck looks for terminate.txt
        os.makedirs(self.dir, exist_ok=True)
        self.csv, self.log = os.path.join(rundir, "unilog.csv"), os.path.join(rundir, "agt.log")
        self.stopfile = os.path.join(self.dir, "terminate.txt")
        # -unilogcount bounds the session should this script die without stopping it.
        self.cmd = ["sudo", "-n", exe, f"-i={','.join(map(str, instances))}", "-unilog=PM", "-unilogallgroups",
                    f"-unilogperiod={round(PERIOD * 1000)}", f"-unilogoutput={self.csv}", "-unilognoesckey",
                    "-unilogstopcheck", f"-unilogcount={math.ceil(max_seconds / PERIOD)}", "-unilogexitonerror"]
        self.proc = self.watchdog = None

    def start(self, columns):
        with open(self.log, "w") as log:
            self.proc = subprocess.Popen(self.cmd, cwd=self.dir, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
        # If this script dies without stopping AGT, the watchdog stops it.
        self.watchdog = subprocess.Popen(
            ["sh", "-c", 'while kill -0 "$1" 2>/dev/null; do sleep 1; done; : > "$2"', "sh", str(os.getpid()),
             self.stopfile], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        t0 = time.monotonic()
        while True:
            if self.proc.poll() is not None:
                fail(f"agt_internal exited with {self.proc.returncode}:\n{tail(self.log)}")
            header, rows = self.peek()
            if header and rows >= 2:
                break
            if time.monotonic() - t0 > 60:
                fail(f"agt_internal wrote no samples in 60 s:\n{tail(self.log)}")
            time.sleep(0.2)
        missing = [c for c in columns if c not in header]
        if missing:
            fail(f"{self.csv} lacks {len(missing)} expected columns, e.g. {missing[:3]}")
        say(f"agt_internal is logging AGT instance(s) {self.cmd[3][3:]} into {self.csv}")

    def peek(self):
        try:
            with open(self.csv, newline="", errors="replace") as f:
                header = [h.strip() for h in next(csv.reader(f), [])]
                return header, sum(1 for _ in f)
        except OSError:
            return [], 0

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.watchdog:
            try:
                os.killpg(self.watchdog.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.watchdog.wait()
            self.watchdog = None
        if not self.alive():
            return
        open(self.stopfile, "w").close()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.terminate()   # sudo relays it
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                say(f"agt_internal did not stop: it ends after {self.cmd[-2]} samples, or stop it with "
                    f"`sudo pkill -f {self.cmd[2]}`")
                return
        say("agt_internal stopped")


class Client:
    """One tensilelite-client, held at the release gate until released."""

    def __init__(self, ctx, tag, ini_text, gpu):
        self.tag, self.gpu = tag, gpu
        gate = os.path.join(ctx.rundir, "gate")
        self.fifo, self.ready = f"{gate}/{tag}.fifo", f"{gate}/{tag}.ready"
        os.mkfifo(self.fifo)
        self.ini, self.logpath, self.csv = (os.path.join(ctx.rundir, f"{tag}.{ext}") for ext in ("ini", "log", "csv"))
        open(self.ini, "w").write(edit_ini(ini_text, {"results-file": self.csv}))
        env = dict(ctx.env, AGTGATE_FIFO=self.fifo, AGTGATE_READY=self.ready, AGTGATE_PARENT=str(os.getpid()),
                   AGTGATE_TIMEOUT=str(int(ctx.gate_timeout)), AGTGATE_ROCR_VISIBLE_DEVICES=ctx.gpus[gpu]["uuid"])
        with open(self.logpath, "w") as log:
            self.proc = subprocess.Popen([ctx.client, "--config-file", self.ini], env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.t_prep, self.t_release, self.t_end, self.rc = now(), None, None, None
        self.queues = set()

    def armed(self):
        return os.path.exists(self.ready + ".armed")

    def waiting(self):
        return os.path.exists(self.ready)

    def send(self, byte):
        fd = os.open(self.fifo, os.O_WRONLY | os.O_NONBLOCK)
        try:
            os.write(fd, byte)
        finally:
            os.close(fd)

    def stop(self):
        """'q' to a client at the gate, SIGKILL to one past it."""
        if self.proc.poll() is not None:
            return
        if self.t_release is None:
            try:
                self.send(b"q")
                return
            except OSError:
                pass
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def reap(self):
        for wait in (20, 20):
            try:
                self.rc = self.proc.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.t_end = self.t_end or now()
        for p in (self.fifo, self.ready, self.ready + ".armed"):
            if os.path.lexists(p):
                os.unlink(p)


def run_batch(ctx, run, ini_text):
    """Start a client per GPU, release them together once all wait at the gate, wait for all of them. Returns the
    batch status (None when every client ran) and the clients."""
    say(f"run {run}: starting {len(ctx.selected)} client(s), {ini_value(ini_text, 'num-enqueues-per-sync')} "
        f"launches per timed window")
    clients, status = {}, None
    try:
        for d in ctx.selected:
            clients[d] = Client(ctx, f"run{run}.gpu{d}", ini_text, d)
        ours = {c.proc.pid for c in clients.values()}
        t0 = time.monotonic()
        while status is None:
            if STOP:
                status = "stopped"
            elif any(c.proc.poll() is not None for c in clients.values()):
                status = "prep-failed"
            elif time.monotonic() - t0 > ctx.args.arm_timeout and not all(c.armed() for c in clients.values()):
                status = "gate-not-loaded"
            elif all(c.waiting() for c in clients.values()):
                break
            elif time.monotonic() - t0 > ctx.args.ready_timeout:
                status = "ready-timeout"
            else:
                time.sleep(0.1)
        t1, last = time.monotonic(), -1
        while status is None and (busy := other_gpu_users(ours, ctx.package_ids)):
            if STOP:
                status = "stopped"
            elif time.monotonic() - t1 > ctx.args.busy_wait:
                status = "busy"
            else:
                if int(time.monotonic() - t1) // 30 != last:
                    last = int(time.monotonic() - t1) // 30
                    say(f"  waiting: other processes use GPUs on these packages: pids {busy}")
                time.sleep(1)
        if status is None and not ctx.agt.alive():
            status = "agt-exited"
        seen = set()
        if status is None:
            for c in clients.values():
                try:
                    c.send(b"g")
                    c.t_release = now()
                except OSError as e:
                    say(f"  cannot release {c.tag}: {e}")
                    status = "release-failed"
                    break
        if status is None:
            say(f"run {run}: released")
            t1 = time.monotonic()
            while any(c.proc.poll() is None for c in clients.values()):
                seen.update(other_gpu_users(ours, ctx.package_ids))
                for c in clients.values():
                    c.queues |= queue_gpus(c.proc.pid)
                if STOP or time.monotonic() - t1 > ctx.args.run_timeout:
                    status = "stopped" if STOP else "run-timeout"
                    break
                time.sleep(0.1)
            for c in clients.values():
                if c.proc.poll() is not None and c.t_end is None:
                    c.t_end = now()
            if status is None and not ctx.agt.alive():
                status = "agt-exited"
    finally:
        for c in clients.values():
            c.stop()
        for c in clients.values():
            c.reap()
    for c in clients.values():
        c.foreign = ",".join(sorted(seen)) if status is None else ""
        c.status = status or ("done" if c.rc == 0 else "failed")
        c.rows = result_rows(c.logpath) if c.status == "done" else []
        if c.status == "done" and not c.rows:
            c.status = "no-result"
        ctx.tsv.write("\t".join(str(v) for v in (run, c.gpu, c.status, ini_value(ini_text, "num-enqueues-per-sync"),
                                                  c.t_prep, c.t_release or "", c.t_end or "", c.rc, c.foreign,
                                                  ",".join(map(str, sorted(c.queues))), c.logpath)) + "\n")
    bad = [c for c in clients.values() if c.status != "done"]
    for c in bad:
        end = tail(c.logpath, 8) if c.status != "stopped" else ""
        say(f"  GPU {c.gpu}: {c.status} (rc {c.rc})" + (f"; the end of {c.logpath}:\n{end}" if end.strip() else ""))
    if not bad:
        say(f"run {run}: done" + (f"; OTHER GPU USERS ON THE PACKAGES DURING THE RUN: pids {sorted(seen)}"
                                  if seen else ""))
    return (status or ("failed" if bad else None)), clients


# ---------------------------------------------------------------------------------------------------- results

def result_rows(log):
    """Every result row the client printed, as dicts."""
    header, rows = None, []
    for line in open(log, errors="replace"):
        if line.startswith("run,problem-progress"):
            header = next(csv.reader([line]))
        elif header and RESULT_ROW.match(line):
            rows.append(dict(zip(header, next(csv.reader([line])))))
    return rows


def number(text):
    try:
        x = float(text)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def window(row, launches):
    """(start, end) of a result row's timed window, in epoch seconds, or None."""
    time_us = number(row.get("time-us"))
    try:
        end = datetime.datetime.strptime(row.get("enqueue-time", "").strip(), "%Y-%m-%d %H:%M:%S.%f").timestamp()
    except ValueError:
        return None
    if not time_us or time_us <= 0:
        return None
    return end - launches * time_us * 1e-6, end


def load_unilog(path, columns):
    """Sample times and samples ({key: value}) of the wanted columns {key: column name}, sorted by time."""
    samples = []
    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        col = {h: i for i, h in enumerate(header)}
        want = {key: col.get(name) for key, name in columns.items()}
        ts = col.get("Time Stamp", 0)
        for row in reader:
            m = re.search(r"\[\s*(\S+ \S+)\s*\]", row[ts]) if len(row) > ts else None
            if not m:
                continue
            values = {}
            for key, i in want.items():
                values[key] = number(row[i]) if i is not None and i < len(row) else None
            samples.append((datetime.datetime.strptime(m.group(1), "%d.%m.%Y %H:%M:%S.%f").timestamp(), values))
    samples.sort(key=lambda s: s[0])
    return [t for t, _ in samples], [v for _, v in samples]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def window_stats(times, samples, start, end, lay):
    """Telemetry of one GPU over [start, end]: a sample stamped t covers (t - PERIOD, t]."""
    inst = lay["inst"]
    span = range(bisect.bisect_left(times, start + PERIOD), bisect.bisect_right(times, end))
    inside = [samples[i] for i in span]
    steady = [samples[i] for i in span if times[i] - PERIOD >= start + ONSET]
    use = steady or inside
    xcd = lambda key, xcds: mean([mean([s.get(f"{inst}.{key}{x}") for x in xcds]) for s in use]) if xcds else None
    pkg = lambda key: mean([s.get(f"{inst}.{key}") for s in use])
    over = [s for s in use if None not in (s.get(f"{inst}.power"), s.get(f"{inst}.limit"))
            and s[f"{inst}.power"] > 1.1 * s[f"{inst}.limit"]]
    return {"samples": len(inside), "steady": len(steady), "gfxclk": xcd("clk", lay["xcds"]),
            "target": xcd("tgt", lay["xcds"]), "ppt0_gfxclk": pkg("ppt0"), "power": pkg("power"),
            "power_limit": pkg("limit"), "over_limit": round(len(over) / len(use), 2) if use else None,
            "busy": xcd("busy", lay["xcds"]), "other_busy": xcd("busy", lay["other"])}


def telemetry_columns(lay):
    cols = {}
    for d in lay.values():
        inst = d["inst"]
        for x in d["xcds"] + d["other"]:
            for key, name in (("clk", "Pre Deep Sleep Freq"), ("tgt", "Target Freq"), ("busy", "Busy")):
                cols[f"{inst}.{key}{x}"] = f"GPU{inst} XCD XCD{x} {name}"
        cols[f"{inst}.ppt0"] = f"GPU{inst} Frequency Limiters PPT_0 GFXCLK"
        cols[f"{inst}.power"] = f"GPU{inst} Input Telemetry Telemetry Power"
        cols[f"{inst}.limit"] = f"GPU{inst} INFRASTRUCTURE Limit PPT_0"
    return cols


def summarize(ctx, batches):
    """summary.csv and summary.md of every client result; prints the table."""
    times, samples = load_unilog(ctx.agt.csv, telemetry_columns(ctx.lay)) if os.path.exists(ctx.agt.csv) else ([], [])
    out = []
    for run, short, clients in batches:
        wins = {}
        for d, c in clients.items():
            launches = int(ini_value(open(c.ini).read(), "num-enqueues-per-sync"))
            for k, row in enumerate(c.rows):
                w = window(row, launches)
                gflops, gbps = number(row.get("gflops")), number(row.get("gbps-bandwidth"))
                r = {"run": run, "gpu": d, "row": k, "problem": row.get("problem-sizes", ""),
                     "validation": row.get("validation", ""), "time_us": number(row.get("time-us")),
                     "pflops": gflops / 1e6 if gflops is not None else None,
                     "tbps": gbps / 1e3 if gbps is not None else None, "launches": launches,
                     "window_s": round(w[1] - w[0], 4) if w else None, "start": w[0] if w else None,
                     "end": w[1] if w else None, "foreign": c.foreign,
                     "queue_gpus": ",".join(map(str, sorted(c.queues))), "excluded": "short window" if short else "",
                     "solution": row.get("solution", "")}
                if w and times:
                    r.update(window_stats(times, samples, w[0], w[1], ctx.lay[d]))
                wins.setdefault(k, {})[d] = w
                out.append(r)
        for r in (r for r in out if r["run"] == run):
            others = [w for d, w in wins[r["row"]].items() if d != r["gpu"]]
            if r["start"] is not None and others and None not in others:
                both = min([r["end"]] + [w[1] for w in others]) - max([r["start"]] + [w[0] for w in others])
                r["overlap"] = round(max(0.0, both) / (r["end"] - r["start"]), 3)
            elif r["start"] is not None and not others:
                r["overlap"] = 1.0
            reasons = [r["excluded"]] if r["excluded"] else []
            if r["foreign"]:
                reasons.append("other GPU users on the package")
            if r["queue_gpus"] and r["queue_gpus"] != str(ctx.gpus[r["gpu"]]["gpu_id"]):
                reasons.append("queues on another GPU")
            if r["validation"].strip() not in ("NO_CHECK", "PASSED") or r["time_us"] is None:
                reasons.append(f"result {r['validation'].strip()} time-us {r['time_us']}")
            if r.get("overlap") is not None and r["overlap"] < 0.9:
                reasons.append("other GPUs' windows cover < 90 %")
            r["excluded"] = "; ".join(reasons)
    if not out:
        say("no client results")
        return
    keys = list(dict.fromkeys(k for r in out for k in r))
    with open(os.path.join(ctx.rundir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)

    med = lambda rs, k: statistics.median(v) if (v := [r[k] for r in rs if r.get(k) is not None]) else None
    fmt = lambda x, spec: format(x, spec) if x is not None else "-"
    multi = len({r["row"] for r in out}) > 1
    lines = [f"{os.path.basename(ctx.config)} on GPU(s) {','.join(map(str, ctx.selected))}: medians over the "
             f"runs, {ctx.host}, {now()[:16]}", "",
             "| GPU |" + (" problem |" if multi else "") + " runs | time-us | PFLOPS | TB/s | window s | samples "
             "(steady) | GFXCLK MHz | target MHz | PPT_0 GFXCLK MHz | power W | over PPT_0 | busy % | "
             "package's other XCDs busy % | overlap |",
             "|---|" + ("---|" if multi else "") + "---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in ctx.selected:
        for k in sorted({r["row"] for r in out if r["gpu"] == d}):
            rs = [r for r in out if r["gpu"] == d and r["row"] == k and not r["excluded"]]
            lines.append(
                f"| {d} |" + (f" {rs[0]['problem'] if rs else ''} |" if multi else "")
                + f" {len(rs)} | {fmt(med(rs, 'time_us'), '.2f')} | {fmt(med(rs, 'pflops'), '.3f')} | "
                f"{fmt(med(rs, 'tbps'), '.3f')} | {fmt(med(rs, 'window_s'), '.3f')} | "
                f"{fmt(med(rs, 'samples'), '.0f')} ({fmt(med(rs, 'steady'), '.0f')}) | {fmt(med(rs, 'gfxclk'), '.0f')} | "
                f"{fmt(med(rs, 'target'), '.0f')} | {fmt(med(rs, 'ppt0_gfxclk'), '.0f')} | "
                f"{fmt(med(rs, 'power'), '.0f')} | {fmt(med(rs, 'over_limit'), '.2f')} | {fmt(med(rs, 'busy'), '.1f')} | "
                f"{fmt(med(rs, 'other_busy'), '.1f')} | {fmt(med(rs, 'overlap'), '.3f')} |")
    excluded = [r for r in out if r["excluded"]]
    if excluded:
        lines += ["", "left out of the table:"] + [f"  run {r['run']} GPU {r['gpu']} row {r['row']}: {r['excluded']}"
                                                   for r in excluded]
    weak = [r for r in out if not r["excluded"] and r.get("busy") is not None and r["busy"] < 50]
    if weak:
        lines += ["", f"warning: {len(weak)} window(s) show the GPU's XCDs under 50 % busy; check the AGT instance "
                      "and XCD mapping in run_info.txt"]
    lines += ["", f"per-run rows: {ctx.rundir}/summary.csv"]
    text = "\n".join(lines)
    open(os.path.join(ctx.rundir, "summary.md"), "w").write(text + "\n")
    print("\n" + text, flush=True)


# ---------------------------------------------------------------------------------------------------- main

class Context:
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("config", help="Tensile config yaml")
    ap.add_argument("gpus", help="HIP device indices run at the same time: 0 | 0,1 | 0-3 | all")
    ap.add_argument("hipblaslt", help="hipBLASLt source dir (projects/hipblaslt)")
    ap.add_argument("agt", help="path to agt_internal")
    ap.add_argument("--runs", type=int, default=1, help="timed runs (batches) to keep (default 1)")
    ap.add_argument("--min-window", type=float, default=1.0,
                    help="shortest timed window to keep, s; a shorter first batch is rerun with more launches "
                         "(default 1.0; 0 keeps the config's launch count)")
    ap.add_argument("--enqueues", type=int, help="launches per timed window, instead of the config's and --min-window")
    ap.add_argument("--out", default="agt_runs", help="output root (default ./agt_runs)")
    ap.add_argument("--client", help="tensilelite-client to run (default <hipblaslt>/tensilelite/build_tmp/"
                                     "tensilelite/client/tensilelite-client)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the config even if a matching build exists")
    ap.add_argument("--omp-threads", type=int, default=16, help="OpenMP threads per client for data init (default 16)")
    ap.add_argument("--arm-timeout", type=float, default=15.0, help="time a client has to load the gate, s")
    ap.add_argument("--ready-timeout", type=float, default=1800.0, help="time the clients have to reach the gate, s")
    ap.add_argument("--busy-wait", type=float, default=600.0,
                    help="how long a batch waits for other GPU users to leave its packages, s")
    ap.add_argument("--run-timeout", type=float, default=1800.0, help="time a batch may run after its release, s")
    args = ap.parse_args()
    if args.runs < 1 or (args.enqueues is not None and args.enqueues < 1) or args.min_window < 0:
        fail("--runs and --enqueues must be at least 1, --min-window at least 0")

    ctx = Context()
    ctx.args, ctx.host = args, socket.gethostname()
    ctx.config = os.path.abspath(args.config)
    if not os.path.isfile(ctx.config):
        fail(f"no config {ctx.config}")
    hipblaslt = os.path.abspath(args.hipblaslt)
    tl = os.path.join(hipblaslt, "tensilelite")
    if not os.path.exists(f"{tl}/Tensile/bin/Tensile") and os.path.exists(f"{hipblaslt}/Tensile/bin/Tensile"):
        tl = hipblaslt
    if not os.path.exists(f"{tl}/Tensile/bin/Tensile"):
        fail(f"no Tensile/bin/Tensile under {hipblaslt}")
    ctx.client = os.path.abspath(args.client or f"{tl}/build_tmp/tensilelite/client/tensilelite-client")
    if not os.access(ctx.client, os.X_OK):
        fail(f"no tensilelite-client at {ctx.client}: build it with `invoke build-client` in {tl}, or pass --client")
    agt = os.path.abspath(args.agt)
    if not os.access(agt, os.X_OK):
        fail(f"{agt} is not an executable")
    if subprocess.run(["sudo", "-n", "true"], stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        fail("agt_internal runs through `sudo -n`, which asks for a password here")
    err = kfd_error()
    if err:
        fail(f"cannot open /dev/kfd, so no GPU program can start on this node: {err}")

    # One run at a time per node, whoever runs it: runs on the same packages would disturb each other.
    try:
        lockfd = os.open(LOCK, os.O_RDWR | os.O_CREAT, 0o666)
        os.fchmod(lockfd, 0o666)
    except PermissionError:
        lockfd = os.open(LOCK, os.O_RDONLY)   # another user's lock file; flock works on a read-only descriptor
    try:
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fail(f"another run_tensile_with_agt.py is running on this node ({LOCK})")

    ctx.gpus = kfd_gpus()
    if not ctx.gpus:
        fail("no GPUs in the KFD topology")
    ctx.selected = parse_gpus(args.gpus, len(ctx.gpus))
    instances, listing = agt_gpu_instances(agt)
    if not instances:
        fail(f"`sudo -n {agt} -i` listed no GPUs:\n{listing[-2000:]}")
    ctx.lay = layout(ctx.gpus, instances, ctx.selected)
    ctx.package_ids = {ctx.gpus[i]["gpu_id"] for d in ctx.lay.values() for i in d["package"]}
    m = re.search(r"(\d+)cu", os.path.basename(os.path.dirname(ctx.config)))
    for d in ctx.selected:
        g, lay = ctx.gpus[d], ctx.lay[d]
        say(f"GPU {d}: {g['bdf']}, {g['cu']} CUs, {g['xcc']} XCDs; AGT instance {lay['inst']}, XCDs "
            f"{lay['xcds'][0]}-{lay['xcds'][-1]}" + (f" of a package shared with GPU(s) "
                                                      f"{[i for i in lay['package'] if i != d]}" if lay["other"] else ""))
        if m and int(m.group(1)) != g["cu"]:
            say(f"  note: the config's directory says {m.group(1)} CUs; GPU {d} has {g['cu']}")

    root = os.path.abspath(args.out)
    os.makedirs(root, exist_ok=True)
    ini = tensile_build(ctx.config, tl, ctx.client, root, args.rebuild)
    gate = build_gate(root)
    base = open(ini).read()
    enqueues = args.enqueues or int(ini_value(base, "num-enqueues-per-sync") or 1) * \
        int(ini_value(base, "num-syncs-per-benchmark") or 1)

    stem = os.path.splitext(os.path.basename(ctx.config))[0]
    ctx.rundir = os.path.join(root, f"{stem}.gpu{'_'.join(map(str, ctx.selected))}."
                                    f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(os.path.join(ctx.rundir, "gate"))
    env = {k: v for k, v in os.environ.items() if k not in VISIBILITY}
    host_lib = os.path.dirname(os.path.dirname(ctx.client))   # build_tmp/tensilelite: the client's libtensilelite-host
    env.update(LD_PRELOAD=":".join(filter(None, [gate, os.environ.get("LD_PRELOAD")])),
               LD_LIBRARY_PATH=":".join(filter(None, [host_lib, os.environ.get("LD_LIBRARY_PATH")])),
               OMP_NUM_THREADS=str(args.omp_threads), ROCR_VISIBLE_DEVICES=NO_GPU)
    ctx.env = env
    ctx.gate_timeout = args.ready_timeout + args.busy_wait + 120
    with open(os.path.join(ctx.rundir, "run_info.txt"), "w") as f:
        f.write(f"command: {' '.join(sys.argv)}\nhost: {ctx.host}\nconfig: {ctx.config} (sha256 "
                f"{hashlib.sha256(open(ctx.config, 'rb').read()).hexdigest()})\nTensile: {tl}\nclient: {ctx.client}\n"
                f"client ini: {ini}\nagt_internal: {agt}\n")
        for d in ctx.selected:
            f.write(f"GPU {d}: {ctx.gpus[d]} -> {ctx.lay[d]}\n")
    say(f"outputs in {ctx.rundir}")

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, lambda *_: STOP.append(1))
    per_batch = args.ready_timeout + args.busy_wait + args.run_timeout + 300
    ctx.agt = Agt(agt, sorted({d["inst"] for d in ctx.lay.values()}), ctx.rundir, per_batch * (args.runs + 1))
    ctx.tsv = open(os.path.join(ctx.rundir, "runs.tsv"), "w", buffering=1)
    ctx.tsv.write("run\tgpu\tstatus\tlaunches\tprep\trelease\tend\trc\tother_gpu_users\tqueue_gpu_ids\tlog\n")
    batches, kept, run, failed = [], 0, 0, None
    try:
        ctx.agt.start(list(telemetry_columns(ctx.lay).values()))
        while kept < args.runs and not STOP:
            run += 1
            status, clients = run_batch(ctx, run, timing_ini(base, enqueues))
            if status:
                failed = status
                break
            spans = [w[1] - w[0] for c in clients.values() for row in c.rows
                     if (w := window(row, enqueues)) is not None]
            shortest = min(spans) if spans else None
            short = (kept == 0 and run == 1 and args.enqueues is None and shortest is not None
                     and shortest < args.min_window)
            batches.append((run, short, clients))
            if short:
                enqueues = math.ceil(enqueues * min(1e4, args.min_window * 1.2 / max(shortest, 1e-6)))
                say(f"shortest timed window {shortest:.3f} s < --min-window {args.min_window} s: running again with "
                    f"{enqueues} launches (run {run} stays in summary.csv, out of the table)")
                continue
            kept += 1
            if shortest is not None and shortest < args.min_window and args.enqueues is None:
                say(f"  note: the shortest timed window is still {shortest:.3f} s")
        time.sleep(2 * PERIOD + 0.2)   # let AGT log past the last window
    finally:
        ctx.agt.stop()
        ctx.tsv.close()
    summarize(ctx, batches)
    if failed or STOP:
        say(f"stopped early: {failed or 'interrupted'}")
    return 0 if kept == args.runs else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:   # before the run starts (e.g. during the build), Ctrl-C needs no cleanup
        sys.exit(130)

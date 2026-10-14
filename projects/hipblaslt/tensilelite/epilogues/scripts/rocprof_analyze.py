#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Hardware-counter profiling and ATT collection for TensileLite GEMM epilogue variants.

Provenance: the pass-driver idea (one rocprofv3 pass per counter group, results read back from
the rocpd SQLite database) comes from /home/fmoracor/aster/tools/rocprof_analyze.py. This file is
a rewrite that fixes its known problems:

  B1  exact kernel-name match (anchored --kernel-include-regex, no substring matching)
  B2  no silent defaults: a missing counter is NaN and fails validation
  B3  TA_DATA_STALLED_BY_TC_CYCLES_sum is collected
  B4  gfx950 occupancy model (512 unified VGPRs per SIMD lane, 160 KiB LDS, 8 waves per SIMD)
  B5  fabric bytes follow the rocprof-compute gfx950 formulas (32/64/128 B read, 32/64 B write)
  B6  timing comes from dedicated kernel-trace passes; PMC-pass durations are reported as perturbed
  B7  a non-zero rocprofv3 exit status is fatal and every pass runs under `timeout`
  B8  exactly one results database with target dispatches per pass
  B9  per-XCC/SE rows are summed (GRBM is divided by the XCC count to get a per-XCD value)
  B10 median is the primary statistic, with mean/std/cv/n alongside
  B11 stall fractions follow rocprof-compute (SQ wave-cycle counters are in quad-cycles)
  B12 /opt/rocm/bin/rocprofv3 (overridable through ROCPROFV3) with `-f rocpd csv`

Subcommands: preflight, list-avail, discover, run, merge.  Standard library only.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time

NAN = float("nan")
PASS_TIMEOUT_S = 900
ROCPROFV3 = os.environ.get("ROCPROFV3", "/opt/rocm/bin/rocprofv3")
PROBE_ENQUEUES = 6000
PROBE_BUSY_PCT = 95.0
TENSILE_CLIENT_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "build_tmp", "tensilelite", "client", "tensilelite-client")
HIP_OCCUPANCY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hip_occupancy.py")

# Problem shape: M, N, batch, K (Tensile size order I, J, L, K).
PROBLEM_M, PROBLEM_N, PROBLEM_BATCH, PROBLEM_K = 2048, 16384, 1, 2048
PROBLEM_FLOPS = 2 * PROBLEM_M * PROBLEM_N * PROBLEM_K * PROBLEM_BATCH
BF16_BYTES = 2
# MOPS counters are flops / 512.
MFMA_MOPS_FLOPS = 512
FLOPS_PER_CLK_PER_CU_BF16 = 4096

# gfx950 compute-unit limits used by the theoretical occupancy model.
SIMDS_PER_CU = 4
MAX_WAVES_PER_SIMD = 8
WAVE_SIZE = 64
UNIFIED_VGPRS_PER_SIMD_LANE = 512
VGPR_ALLOC_GRANULE = 8
LDS_BYTES_PER_CU = 160 * 1024
LDS_ALLOC_GRANULE = 1280

VARIANT_ORDER = ["rms", "beta", "none"]

# Counter groups. Per-block limits per pass on gfx950 (rocprof-compute perfmon):
# GRBM 2, SPI 6, SQ 8, SQC 6, TA 2, TCC 4, TCP 4, TD 2. There is no limit on the total number of
# counters per pass; a block over its limit aborts rocprofv3. Every hardware group also collects
# the GRBM pair so that each rate is normalised by the same pass's own GRBM_GUI_ACTIVE (F2).
GRBM_PAIR = ["GRBM_COUNT", "GRBM_GUI_ACTIVE"]
COUNTER_GROUPS = {
    "sq_occ": ["SQ_WAVES", "SQ_WAVE_CYCLES", "SQ_BUSY_CYCLES", "SQ_BUSY_CU_CYCLES", "SQ_INSTS",
               "SQ_INSTS_VALU", "SQ_INSTS_MFMA", "SQ_VALU_MFMA_BUSY_CYCLES"],
    "sq_valu": ["SQ_WAVES", "SQ_WAVE_CYCLES", "SQ_ACTIVE_INST_VALU", "SQ_THREAD_CYCLES_VALU",
                "SQ_ACTIVE_INST_ANY", "SQ_INSTS_MFMA", "SQ_INSTS_VALU_MFMA_MOPS_BF16",
                "SQ_INSTS_VALU_MFMA_BF16"],
    "sq_mix": ["SQ_WAVES", "SQ_INSTS_SALU", "SQ_INSTS_SMEM", "SQ_INSTS_LDS", "SQ_INSTS_VMEM_RD",
               "SQ_INSTS_VMEM_WR", "SQ_INSTS_BRANCH", "SQ_INSTS_FLAT"],
    "sq_mix2": ["SQ_WAVES", "SQ_INSTS_VMEM", "SQ_INSTS_SENDMSG", "SQ_IFETCH", "SQ_ACTIVE_INST_SCA",
                "SQ_ACTIVE_INST_FLAT", "SQ_ACTIVE_INST_MISC"],
    "sq_wait": ["SQ_WAVE_CYCLES", "SQ_WAIT_ANY", "SQ_WAIT_INST_ANY", "SQ_WAIT_INST_LDS",
                "SQ_ACTIVE_INST_ANY", "SQ_ACTIVE_INST_LDS", "SQ_ACTIVE_INST_VMEM",
                "SQ_ACTIVE_INST_VALU"],
    "sq_lds": ["SQ_WAVES", "SQ_INSTS_LDS", "SQ_LDS_IDX_ACTIVE", "SQ_LDS_BANK_CONFLICT",
               "SQ_LDS_ADDR_CONFLICT", "SQ_LDS_UNALIGNED_STALL", "SQ_LDS_DATA_FIFO_FULL",
               "SQ_LDS_CMD_FIFO_FULL"],
    "sq_lds_inst": ["SQ_WAVES", "SQ_INSTS_LDS_LOAD", "SQ_INSTS_LDS_STORE",
                    "SQ_INSTS_LDS_LOAD_BANDWIDTH", "SQ_INSTS_LDS_STORE_BANDWIDTH", "SQ_INSTS_LDS"],
    "sqc_icache": ["SQC_ICACHE_REQ", "SQC_ICACHE_HITS", "SQC_ICACHE_MISSES",
                   "SQC_ICACHE_MISSES_DUPLICATE", "SQC_ICACHE_BUSY_CYCLES", "SQC_TC_INST_REQ"],
    "sqc_dcache": ["SQC_DCACHE_REQ", "SQC_DCACHE_HITS", "SQC_DCACHE_MISSES",
                   "SQC_DCACHE_MISSES_DUPLICATE", "SQC_TC_STALL"],
    # vL1D coalescing and hit rate: TA_TOTAL_WAVEFRONTS with TCP_TOTAL_ACCESSES, and the request
    # counters of the 1600 hit-rate formula. The five-counter TCP form exceeds the TCP limit of 4,
    # so the two atomic request counters live in tcp_atomic (see the vl1d_hit_rate_pct row). The SQ
    # vmem instruction counts ride along so that requests per instruction are same-pass ratios.
    "tcp_coal": ["TA_TOTAL_WAVEFRONTS_sum", "TA_TA_BUSY_sum", "TCP_TOTAL_ACCESSES_sum",
                 "TCP_TOTAL_CACHE_ACCESSES_sum", "TCP_TCC_READ_REQ_sum", "TCP_TCC_WRITE_REQ_sum",
                 "SQ_INSTS_VMEM", "SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR"],
    "tcp_atomic": ["TCP_TOTAL_CACHE_ACCESSES_sum", "TCP_TCC_ATOMIC_WITH_RET_REQ_sum",
                   "TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum"],
    "tcp_gate": ["TCP_GATE_EN1_sum", "TCP_GATE_EN2_sum", "TCP_PENDING_STALL_CYCLES_sum",
                 "TCP_TCR_TCP_STALL_CYCLES_sum"],
    "tcp_ta": ["TCP_GATE_EN1_sum", "TCP_TCP_TA_ADDR_STALL_CYCLES_sum",
               "TCP_TCP_TA_DATA_STALL_CYCLES_sum", "TCP_READ_TAGCONFLICT_STALL_CYCLES_sum"],
    "tcp_ret": ["TCP_GATE_EN1_sum", "TCP_TCR_RDRET_STALL_sum", "TCP_LFIFO_STALL_CYCLES_sum",
                "TCP_TOTAL_READ_sum"],
    "ta_rd": ["TA_BUFFER_READ_WAVEFRONTS_sum", "TA_BUFFER_COALESCED_READ_CYCLES_sum"],
    "ta_wr": ["TA_BUFFER_WRITE_WAVEFRONTS_sum", "TA_BUFFER_COALESCED_WRITE_CYCLES_sum"],
    "ta_buf": ["TA_BUFFER_WAVEFRONTS_sum", "TA_BUFFER_TOTAL_CYCLES_sum"],
    "ta_stall_tc": ["TA_ADDR_STALLED_BY_TC_CYCLES_sum", "TA_DATA_STALLED_BY_TC_CYCLES_sum"],
    "ta_stall_td": ["TA_ADDR_STALLED_BY_TD_CYCLES_sum", "TA_FLAT_WAVEFRONTS_sum"],
    "td_busy": ["TD_TD_BUSY_sum", "TD_LOAD_WAVEFRONT_sum"],
    "td_stall": ["TD_TC_STALL_sum", "TD_SPI_STALL_sum"],
    "td_store": ["TD_STORE_WAVEFRONT_sum", "TD_COALESCABLE_WAVEFRONT_sum"],
    "tcc_hit": ["TCC_HIT_sum", "TCC_MISS_sum", "TCC_REQ_sum", "TCC_TAG_STALL_sum"],
    "tcc_ea": ["TCC_WRITE_sum", "TCC_EA0_WRREQ_sum", "TCC_EA0_WRREQ_64B_sum",
               "TCC_EA0_WRREQ_STALL_sum"],
    "tcc_rd": ["TCC_EA0_RDREQ_sum", "TCC_EA0_RDREQ_32B_sum", "TCC_EA0_RDREQ_64B_sum",
               "TCC_EA0_RDREQ_128B_sum"],
    "tcc_dram": ["TCC_EA0_RDREQ_DRAM_sum", "TCC_EA0_WRREQ_DRAM_sum", "TCC_WRITEBACK_sum",
                 "TCC_READ_sum"],
}
# Derived rocprofv3 counters, collected without the GRBM pair. Each gets its own pass: collecting
# MeanOccupancyPerCU together with MeanOccupancyPerActiveCU changes its value about 4x (F5), and
# FETCH_SIZE with WRITE_SIZE exceeds the TCC limit (rocprofv3 error 38).
DERIVED_GROUPS = {
    "drv_fetch": ["FETCH_SIZE"],
    "drv_write": ["WRITE_SIZE"],
    "drv_occ_cu": ["MeanOccupancyPerCU"],
    "drv_occ_pct": ["OccupancyPercent"],
    "drv_occ_active": ["MeanOccupancyPerActiveCU"],
    "drv_valu": ["VALUUtilization", "VALUBusy"],
    "drv_lds": ["LDSBankConflict"],
}
GROUPS_WITHOUT_GRBM = set(DERIVED_GROUPS)
PER_XCD_COUNTERS = set(GRBM_PAIR)
BLOCK_LIMITS = {"TA": 2, "TCC": 4, "TCP": 4, "TD": 2, "GRBM": 2, "SPI": 6, "SQ": 8, "SQC": 6}


class FatalError(Exception):
    """Unrecoverable data-collection failure."""


def fatal(message):
    raise FatalError(message)


def log(message):
    print(f"[rocprof_analyze] {message}", flush=True)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def read_csv_dicts(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fmt(x):
    if isinstance(x, float):
        return "" if not math.isfinite(x) else repr(x)
    return x


def safe_div(a, b):
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b) or b == 0:
        return NAN
    return a / b


def counter_block(name):
    return name.split("_", 1)[0]


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------
class Stats:
    """Median-primary summary of a list of samples (B10)."""

    def __init__(self, values):
        vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        self.n = len(vals)
        if not vals:
            self.median = self.mean = self.std = self.cv = self.min = self.max = NAN
            return
        self.median = statistics.median(vals)
        self.mean = statistics.fmean(vals)
        self.std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        self.cv = safe_div(self.std, abs(self.mean)) if self.mean != 0 else 0.0
        self.min = min(vals)
        self.max = max(vals)


# ---------------------------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------------------------
def pinned_env(selector):
    env = dict(os.environ)
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        env.pop(var, None)
    if selector:
        env["ROCR_VISIBLE_DEVICES"] = selector
    return env


def run_logged(cmd, log_path, env, timeout_s=PASS_TIMEOUT_S, cwd=None):
    """Run a command under `timeout`, tee its output to log_path, and raise on failure (B7)."""
    full = ["timeout", "-k", "10", str(timeout_s)] + cmd
    with open(log_path, "w") as f:
        f.write("# " + " ".join(full) + "\n")
        f.flush()
        proc = subprocess.run(full, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd)
    if proc.returncode != 0:
        fatal(f"command failed with rc={proc.returncode} (124 means timeout): "
              f"{' '.join(cmd)[:200]} (log {log_path})")


def capture(cmd, env=None, timeout_s=60):
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_s)
    return proc.returncode, proc.stdout, proc.stderr


def rocm_smi_json(args):
    rc, out, _ = capture(["rocm-smi"] + args + ["--json"])
    if rc != 0:
        fatal(f"rocm-smi {' '.join(args)} failed")
    start = out.find("{")
    return json.loads(out[start:])


# ---------------------------------------------------------------------------------------------
# preflight (S1)
# ---------------------------------------------------------------------------------------------
def gpu_identity(gpu):
    card = f"card{gpu}"
    uid = rocm_smi_json(["-d", str(gpu), "--showuniqueid"])[card]["Unique ID"]
    rc, out, _ = capture(["rocm-smi", "-d", str(gpu), "--showbus"])
    bus = re.search(r"PCI Bus:\s*([0-9A-Fa-f:.]+)", out)
    if rc != 0 or not bus:
        fatal(f"cannot read PCI bus of gpu {gpu}")
    pci = bus.group(1)
    bus_number = int(pci.split(":")[1], 16)
    return uid.lower().replace("0x", ""), pci, bus_number


def gpu_use_percent(gpu):
    data = rocm_smi_json(["-d", str(gpu), "--showuse"])[f"card{gpu}"]
    return float(data["GPU use (%)"])


def check_idle(gpu, force):
    samples = []
    for _ in range(4):
        samples.append(gpu_use_percent(gpu))
        time.sleep(0.5)
    if max(samples) > 3.0 and not force:
        fatal(f"gpu {gpu} is not idle (use% samples {samples}); pass --force to override")
    return samples


def tool_version(cmd):
    try:
        rc, out, err = capture(cmd)
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    text = (out + err).strip()
    return text.splitlines()[0] if rc == 0 and text else "unavailable"


def rocm_core_version():
    header = "/opt/rocm/include/rocm-core/rocm_version.h"
    try:
        with open(header) as f:
            text = f.read()
    except OSError:
        return "unavailable"
    parts = [re.search(rf"#define ROCM_VERSION_{k}\s+(\d+)", text) for k in ("MAJOR", "MINOR", "PATCH")]
    return ".".join(m.group(1) for m in parts) if all(parts) else "unavailable"


def git_info(repo_dir):
    def git(*args):
        rc, out, _ = capture(["git", "-C", repo_dir] + list(args))
        return out.strip() if rc == 0 else "unavailable"
    return {"branch": git("rev-parse", "--abbrev-ref", "HEAD"), "commit": git("rev-parse", "HEAD"),
            "dirty_files": len(git("status", "--porcelain", "--untracked-files=no").splitlines())}


def cmd_preflight(args):
    uid, pci, bus_number = gpu_identity(args.gpu)
    selector = f"GPU-{uid}"
    samples = check_idle(args.gpu, args.force)
    os.makedirs(args.out_dir, exist_ok=True)
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": os.uname().nodename,
        "kernel": os.uname().release,
        "gpu_index": args.gpu,
        "gpu_unique_id": uid,
        "gpu_pci_bus": pci,
        "gpu_pci_bus_number": bus_number,
        "rocr_visible_devices": selector,
        "idle_use_percent_samples": samples,
        "forced": bool(args.force),
        "rocprofv3": ROCPROFV3,
        "rocprofv3_version": tool_version([ROCPROFV3, "--version"]),
        "rocm_version": rocm_core_version(),
        "clang_version": tool_version([os.path.join(LLVM_BIN, "clang"), "--version"]),
        "python": sys.version.split()[0],
        "git": git_info(args.tensile_dir),
        "problem": {"M": PROBLEM_M, "N": PROBLEM_N, "batch": PROBLEM_BATCH, "K": PROBLEM_K},
        "cleared_env": ["HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"],
    }
    write_json(os.path.join(args.out_dir, "run_metadata.json"), meta)
    print(selector)


# ---------------------------------------------------------------------------------------------
# list-avail (P0)
# ---------------------------------------------------------------------------------------------
def parse_available(path):
    names = set()
    with open(path) as f:
        for line in f:
            m = re.match(r"\s*Counter_Name\s*:\s*(\S+)", line)
            if m:
                names.add(m.group(1))
    return names


def requested_counters():
    names = set()
    for group in list(COUNTER_GROUPS.values()) + list(DERIVED_GROUPS.values()):
        names.update(group)
    return names


def check_group_limits():
    for name, counters in list(COUNTER_GROUPS.items()) + list(DERIVED_GROUPS.items()):
        full = counters if name in GROUPS_WITHOUT_GRBM else counters + GRBM_PAIR
        per_block = {}
        for c in full:
            per_block[counter_block(c)] = per_block.get(counter_block(c), 0) + 1
        for block, count in per_block.items():
            if block in BLOCK_LIMITS and count > BLOCK_LIMITS[block]:
                fatal(f"group {name} has {count} {block} counters, limit {BLOCK_LIMITS[block]}")


def check_counter_availability(avail_path):
    available = parse_available(avail_path)
    if not available:
        fatal(f"no counters parsed from {avail_path}")
    missing = sorted(requested_counters() - available)
    if missing:
        fatal(f"requested counters not available on this system: {missing}")
    check_group_limits()


def cmd_list_avail(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    run_logged([ROCPROFV3, "-L"], args.out, pinned_env(None), timeout_s=300,
               cwd=os.path.dirname(os.path.abspath(args.out)))
    check_counter_availability(args.out)
    log(f"counter availability verified ({len(parse_available(args.out))} counters listed)")


# ---------------------------------------------------------------------------------------------
# discover (S3)
# ---------------------------------------------------------------------------------------------
def yaml_bool_field(text, key):
    values = set(re.findall(rf"^\s*{key}:\s*(true|false)\s*$", text, re.M | re.I))
    if len(values) != 1:
        fatal(f"library yaml field {key} is missing or inconsistent: {sorted(values)}")
    return values.pop().lower() == "true"


def yaml_kernel_name(text):
    # Kernel entries are indented by four spaces; the solution name (two spaces) is longer.
    names = set(re.findall(r"^ {4}name:\s*(Cijk_\S+)\s*$", text, re.M))
    if len(names) != 1:
        fatal(f"expected exactly one kernel name in library yaml, found {len(names)}")
    return names.pop()


def winner_time_us(tensile_dir, problem_dir_name):
    path = os.path.join(tensile_dir, "2_BenchmarkData", f"{problem_dir_name}_CSVWinner.csv")
    if not os.path.exists(path):
        fatal(f"missing winner csv {path}")
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    return float(rows[1][header.index("WinnerTimeUS")])


def classify(use_beta, use_rms):
    if use_rms and not use_beta:
        return "rms"
    if use_beta and not use_rms:
        return "beta"
    if not use_beta and not use_rms:
        return "none"
    fatal("variant with both UseBeta and UseRMSEpilogue is not expected")


def discover_one(tensile_dir, problem_dir):
    caches = glob.glob(os.path.join(problem_dir, "00_Final", "caches", "*", "source"))
    if len(caches) != 1:
        fatal(f"expected exactly one cache source dir under {problem_dir}, found {len(caches)}")
    source = caches[0]
    lib_dir = os.path.join(source, "library", "gfx950")
    yaml_path = os.path.join(lib_dir, "TensileLibrary.yaml")
    with open(yaml_path) as f:
        text = f.read()
    use_beta = yaml_bool_field(text, "useBeta")
    use_rms = yaml_bool_field(text, "useRMSEpilogue")
    name = os.path.basename(problem_dir)
    return {
        "variant": classify(use_beta, use_rms),
        "problem_name": name,
        "use_beta": use_beta,
        "use_rms_epilogue": use_rms,
        "kernel_name": yaml_kernel_name(text),
        "library_yaml": yaml_path,
        "code_object": os.path.join(lib_dir, "TensileLibrary_gfx950.co"),
        "client_ini": os.path.join(source, "ClientParameters.ini"),
        "tensile_winner_time_us": winner_time_us(tensile_dir, name),
    }


def cmd_discover(args):
    problem_dirs = sorted(glob.glob(os.path.join(args.tensile_out, "1_BenchmarkProblems", "*")))
    found = [discover_one(args.tensile_out, d) for d in problem_dirs]
    by_variant = {}
    for entry in found:
        by_variant.setdefault(entry["variant"], []).append(entry)
    for variant in VARIANT_ORDER:
        if len(by_variant.get(variant, [])) != 1:
            fatal(f"expected exactly one '{variant}' variant, found {len(by_variant.get(variant, []))}")
    if len(found) != len(VARIANT_ORDER):
        fatal(f"expected {len(VARIANT_ORDER)} problems, found {len(found)}")
    for entry in found:
        for key in ("library_yaml", "code_object", "client_ini"):
            if not os.path.exists(entry[key]):
                fatal(f"{entry['variant']}: missing {key} {entry[key]}")
    write_json(args.out, {v: by_variant[v][0] for v in VARIANT_ORDER})
    log("variants: " + ", ".join(f"{v}={by_variant[v][0]['problem_name']}" for v in VARIANT_ORDER))


# ---------------------------------------------------------------------------------------------
# Client ini derivation
# ---------------------------------------------------------------------------------------------
def read_ini(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if "=" in line:
                key, value = line.split("=", 1)
                entries.append((key.strip(), value.strip()))
    return entries


def ini_value(entries, key):
    for k, v in entries:
        if k == key:
            return v
    fatal(f"ini key {key} not found")


def expected_target_dispatches(entries):
    warmups = int(ini_value(entries, "num-warmups"))
    enqueues = int(ini_value(entries, "num-enqueues-per-sync"))
    syncs = int(ini_value(entries, "num-syncs-per-benchmark"))
    benchmarks = int(ini_value(entries, "num-benchmarks"))
    return warmups, warmups + enqueues * syncs * benchmarks


def write_derived_ini(variant, out_path, results_file, overrides=None):
    """Derive a client ini with validation off (the client ini beats the CLI) and local results."""
    entries = read_ini(variant["client_ini"])
    forced = {
        "library-file": variant["library_yaml"],
        "code-object": variant["code_object"],
        "results-file": results_file,
        "num-elements-to-validate": "0",
        "device-idx": "0",
    }
    forced.update(overrides or {})
    seen = set()
    with open(out_path, "w") as f:
        for key, value in entries:
            f.write(f"{key}={forced.get(key, value)}\n")
            seen.add(key)
        for key, value in forced.items():
            if key not in seen:
                f.write(f"{key}={value}\n")
    return read_ini(out_path)


# ---------------------------------------------------------------------------------------------
# rocpd database access
# ---------------------------------------------------------------------------------------------
def find_single_db(directory):
    """B8: a pass directory must hold exactly one results database."""
    dbs = glob.glob(os.path.join(directory, "*", "*_results.db"))
    if len(dbs) != 1:
        fatal(f"expected exactly one results db under {directory}, found {len(dbs)}")
    return dbs[0]


def open_db(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def kernel_short_name(name):
    return name[:-3] if name.endswith(".kd") else name


DISPATCH_QUERY = """
SELECT d.dispatch_id, d.event_id, d.agent_id, d.start, d.end,
       d.grid_size_x, d.grid_size_y, d.grid_size_z,
       d.workgroup_size_x, d.workgroup_size_y, d.workgroup_size_z,
       d.group_segment_size, d.private_segment_size, d.queue_id,
       s.kernel_name, s.sgpr_count, s.arch_vgpr_count, s.accum_vgpr_count
FROM rocpd_kernel_dispatch d JOIN rocpd_info_kernel_symbol s ON s.id = d.kernel_id
ORDER BY d.start
"""
DISPATCH_FIELDS = ["dispatch_id", "event_id", "agent_id", "start", "end", "grid_x", "grid_y",
                   "grid_z", "wg_x", "wg_y", "wg_z", "lds_bytes", "scratch_bytes", "queue_id",
                   "kernel_name", "sgpr", "arch_vgpr", "accum_vgpr"]


def load_dispatches(db_path):
    con = open_db(db_path)
    try:
        rows = con.execute(DISPATCH_QUERY).fetchall()
    finally:
        con.close()
    out = []
    for row in rows:
        d = dict(zip(DISPATCH_FIELDS, row))
        d["kernel_name"] = kernel_short_name(d["kernel_name"])
        d["duration_us"] = (d["end"] - d["start"]) / 1000.0
        out.append(d)
    return out


def load_agent(db_path, agent_id):
    con = open_db(db_path)
    try:
        row = con.execute("SELECT extdata, name FROM rocpd_info_agent WHERE id = ?",
                          (agent_id,)).fetchone()
    finally:
        con.close()
    if row is None:
        fatal(f"agent {agent_id} not found in {db_path}")
    info = json.loads(row[0])
    info["agent_row_id"] = agent_id
    return info


def kernel_symbols(db_path):
    con = open_db(db_path)
    try:
        return {kernel_short_name(r[0]) for r in
                con.execute("SELECT kernel_name FROM rocpd_info_kernel_symbol")}
    finally:
        con.close()


def load_pmc_rows(db_path):
    """Return {(counter, event_id): (n_rows, sum)}; each row is one XCC/SE/instance sample."""
    con = open_db(db_path)
    try:
        rows = con.execute(
            "SELECT p.name, e.event_id, COUNT(*), SUM(e.value) FROM rocpd_pmc_event e "
            "JOIN rocpd_info_pmc p ON p.id = e.pmc_id GROUP BY 1, 2").fetchall()
    finally:
        con.close()
    return {(name, event): (n, total) for name, event, n, total in rows}


def target_dispatches(dispatches, target_name):
    """B1: exact name match only."""
    return [d for d in dispatches if d["kernel_name"] == target_name]


def post_warmup(dispatches, warmups):
    return dispatches[warmups:]


# ---------------------------------------------------------------------------------------------
# Occupancy model (B4)
# ---------------------------------------------------------------------------------------------
def round_up(value, granule):
    return -(-value // granule) * granule


def theoretical_occupancy(arch_vgpr, accum_vgpr, lds_bytes, wg_size):
    waves_per_wg = -(-wg_size // WAVE_SIZE)
    total_vgpr = round_up(round_up(arch_vgpr, 4) + accum_vgpr, VGPR_ALLOC_GRANULE)
    waves_per_simd = min(MAX_WAVES_PER_SIMD, UNIFIED_VGPRS_PER_SIMD_LANE // max(total_vgpr, 1))
    limits = {
        "vgpr": (waves_per_simd * SIMDS_PER_CU) // waves_per_wg,
        "waves": (MAX_WAVES_PER_SIMD * SIMDS_PER_CU) // waves_per_wg,
    }
    if lds_bytes > 0:
        limits["lds"] = LDS_BYTES_PER_CU // round_up(lds_bytes, LDS_ALLOC_GRANULE)
    limiter = min(limits, key=lambda k: limits[k])
    return {"limits": limits, "wg_per_cu": limits[limiter], "limiter": limiter,
            "waves_per_wg": waves_per_wg, "total_vgpr": total_vgpr}


def hip_api_occupancy(variant, wg_size, env):
    cmd = [sys.executable, HIP_OCCUPANCY, "--co", variant["code_object"], "--kernel",
           variant["kernel_name"], "--block-size", str(wg_size)]
    rc, out, err = capture(cmd, env=env, timeout_s=120)
    if rc != 0:
        fatal(f"hip_occupancy.py failed: {err.strip()[:300]}")
    return int(json.loads(out.strip().splitlines()[-1])["max_active_blocks_per_cu"])


# ---------------------------------------------------------------------------------------------
# Clock probe (V5): sample rocm-smi sclk while the kernel runs long enough to reach steady state
# ---------------------------------------------------------------------------------------------
def parse_mhz(text):
    m = re.search(r"([0-9.]+)\s*Mhz", text, re.I)
    return float(m.group(1)) if m else NAN


def sample_sclk(gpu):
    data = rocm_smi_json(["-d", str(gpu), "--showclocks", "--showuse"])[f"card{gpu}"]
    return parse_mhz(data["sclk clock speed:"]), float(data["GPU use (%)"])


def first_burst_samples(proc, gpu):
    """Samples taken while the GPU is saturated, stopping when the first busy burst ends."""
    samples = []
    seen_busy = False
    while proc.poll() is None:
        mhz, use = sample_sclk(gpu)
        busy = use >= PROBE_BUSY_PCT and not math.isnan(mhz)
        if busy:
            samples.append(mhz)
        seen_busy = seen_busy or busy
        if seen_busy and not busy and use < 50.0:
            break
    return samples


def probe_sclk_mhz(gpu, tool_cmd, env, log_path, cwd):
    """Median sclk (MHz) sampled under the same profiled, serialized dispatch pattern as the PMC
    passes. Back-to-back unprofiled dispatches run power-limited at a lower clock, so they are not
    comparable with the counter-derived clock."""
    with open(log_path, "w") as f:
        proc = subprocess.Popen(["timeout", "-k", "10", str(PASS_TIMEOUT_S)] + tool_cmd, stdout=f,
                                stderr=subprocess.STDOUT, env=env, cwd=cwd)
        samples = first_burst_samples(proc, gpu)
        proc.wait()
    if proc.returncode != 0:
        fatal(f"clock probe failed rc={proc.returncode} (log {log_path})")
    return statistics.median(samples) if samples else NAN, len(samples)


# ---------------------------------------------------------------------------------------------
# Per-variant run context and validation bookkeeping
# ---------------------------------------------------------------------------------------------
class Ctx:
    """State shared by the passes of one variant."""

    def __init__(self, args, variant):
        self.args = args
        self.variant = variant
        self.name = variant["variant"]
        self.target = variant["kernel_name"]
        self.vdir = os.path.abspath(args.out)
        self.logdir = os.path.join(self.vdir, "logs")
        self.passdir = os.path.join(self.vdir, "passes")
        self.env = pinned_env(args.gpu_selector)
        self.client = os.path.abspath(args.client)
        self.metadata = read_json(args.metadata)
        self.ini_path = os.path.join(self.vdir, "client.ini")
        self.ini = None
        self.warmups = 0
        self.expected_total = 0
        self.agent = None
        self.num_xcc = 0
        self.cu_count = 0
        self.timing_us = []
        self.timing_dispatches = []
        self.inventory = []
        self.raw = []
        self.pass_durations_us = {}
        self.checks = []
        self.metrics = []
        self.notes = {}
        self.passes = {}
        self.mfma = None
        self.mfma_expect = None
        self.occ = None
        self.mrows = {}
        self.mv = {}
        self.att_manifest = []

    def check(self, check_id, name, status, observed, expected, tolerance, detail=""):
        self.checks.append({"variant": self.name, "check_id": check_id, "check": name,
                            "status": status, "observed": fmt(observed), "expected": fmt(expected),
                            "tolerance": tolerance, "detail": detail})

    def check_within(self, check_id, name, observed, expected, rel_tol, fail_status="FAIL",
                     detail=""):
        ok = (not math.isnan(observed) and not math.isnan(expected) and expected != 0
              and abs(observed - expected) <= rel_tol * abs(expected))
        self.check(check_id, name, "PASS" if ok else fail_status, fmt(observed), fmt(expected),
                   f"rel {rel_tol}", detail)


def run_pass(ctx, name, tool_args, base=None):
    directory = os.path.join(base or ctx.passdir, name)
    if os.path.exists(directory):
        shutil.rmtree(directory)
    cmd = [ROCPROFV3] + tool_args + ["-f", "rocpd", "csv", "-d", directory, "--", ctx.client,
                                      "--config-file", ctx.ini_path]
    run_logged(cmd, os.path.join(ctx.logdir, f"{name}.log"), ctx.env, cwd=ctx.vdir)
    return directory


def target_regex(ctx):
    return f"^{re.escape(ctx.target)}$"


# ---------------------------------------------------------------------------------------------
# P1: kernel-trace timing passes
# ---------------------------------------------------------------------------------------------
def timing_pass(ctx, index):
    directory = run_pass(ctx, f"timing_r{index}", ["--kernel-trace", "--stats"])
    db = find_single_db(directory)
    dispatches = load_dispatches(db)
    targets = target_dispatches(dispatches, ctx.target)
    if len(targets) != ctx.expected_total:
        fatal(f"timing_r{index}: {len(targets)} target dispatches, expected {ctx.expected_total}")
    if index == 0:
        ctx.agent = load_agent(db, targets[0]["agent_id"])
        ctx.inventory = dispatches
    ctx.timing_dispatches.append(targets)
    ctx.timing_us.extend(d["duration_us"] for d in post_warmup(targets, ctx.warmups))
    return targets


def run_timing_passes(ctx):
    for i in range(ctx.args.timing_repeats):
        timing_pass(ctx, i)
    first = ctx.timing_dispatches[0]
    ids = [[d["dispatch_id"] for d in t] for t in ctx.timing_dispatches]
    agents = {d["agent_id"] for t in ctx.timing_dispatches for d in t}
    queues = {d["queue_id"] for t in ctx.timing_dispatches for d in t}
    ok = all(i == ids[0] for i in ids) and len(agents) == 1 and len(queues) == 1
    ctx.check("V1", "timing passes isolate target kernel with expected dispatch count",
              "PASS" if ok else "FAIL", f"{[len(t) for t in ctx.timing_dispatches]} dispatches",
              ctx.expected_total, "exact",
              f"agents={sorted(agents)} queues={sorted(queues)} first_dispatch_id={first[0]['dispatch_id']}")
    ctx.num_xcc = int(ctx.agent["num_xcc"])
    ctx.cu_count = int(ctx.agent["cu_count"])


# ---------------------------------------------------------------------------------------------
# P2: counter passes
# ---------------------------------------------------------------------------------------------
def counter_pass(ctx, group, counters):
    directory = run_pass(ctx, f"pmc_{group}", ["--pmc"] + counters +
                         ["--kernel-include-regex", target_regex(ctx)])
    db = find_single_db(directory)
    dispatches = load_dispatches(db)
    targets = target_dispatches(dispatches, ctx.target)
    foreign = len(dispatches) - len(targets)
    complete = len(targets) == ctx.expected_total and foreign == 0
    ctx.check("V1", f"pmc pass {group} isolates target kernel", "PASS" if complete else "FAIL",
              f"{len(targets)} target + {foreign} other dispatches", ctx.expected_total, "exact")
    ctx.pass_durations_us[group] = [d["duration_us"] for d in post_warmup(targets, ctx.warmups)]
    pmc = load_pmc_rows(db)
    ctx.passes[group] = {"counters": {}, "dur_us": ctx.pass_durations_us[group]}
    for counter in counters:
        record_counter(ctx, group, counter, targets, pmc)


def record_counter(ctx, group, counter, targets, pmc):
    post_values = []
    for index, d in enumerate(targets):
        n_rows, total = pmc.get((counter, d["event_id"]), (0, NAN))
        value = total / ctx.num_xcc if counter in PER_XCD_COUNTERS and n_rows else total
        ctx.raw.append({"group": group, "counter": counter, "dispatch_index": index,
                        "dispatch_id": d["dispatch_id"], "post_warmup": index >= ctx.warmups,
                        "n_instance_rows": n_rows, "sum_value": total, "agg_value": value,
                        "duration_ns": d["end"] - d["start"]})
        if index >= ctx.warmups:
            post_values.append(value)
    ctx.passes[group]["counters"][counter] = post_values


def run_counter_passes(ctx, avail_path):
    check_counter_availability(avail_path)
    for group, counters in COUNTER_GROUPS.items():
        log(f"{ctx.name}: pmc group {group}")
        counter_pass(ctx, group, counters + GRBM_PAIR)
    for group, counters in DERIVED_GROUPS.items():
        log(f"{ctx.name}: derived group {group}")
        counter_pass(ctx, group, counters)


def pass_stats(ctx, group, counter):
    """Per-dispatch summary of one counter of one pass (never pooled across passes)."""
    return Stats(ctx.passes[group]["counters"][counter])


# ---------------------------------------------------------------------------------------------
# P3: ATT collection (collect only; decoding is done inline by rocprofv3 and not post-processed)
# ---------------------------------------------------------------------------------------------
ATT_ITERATION = 25  # 1-based among regex-matching dispatches; the 5th after 20 warmups.
ATT_KINDS = [(r"\.att$", "att_raw"), (r"code_object_id_\d+\.out$", "code_object"),
             (r"stats_ui_output.*\.csv$", "att_stats_csv"), (r"/ui_output_[^/]*/", "att_ui_json"),
             (r"_results\.db$", "rocpd_db")]


def att_kind(relpath):
    for pattern, kind in ATT_KINDS:
        if re.search(pattern, "/" + relpath):
            return kind
    return "other"


def att_files(att_dir):
    out = []
    for root, _, names in os.walk(att_dir):
        for n in sorted(names):
            path = os.path.join(root, n)
            out.append((os.path.relpath(path, att_dir), os.path.getsize(path)))
    return sorted(out)


def run_att_pass(ctx):
    if len(ctx.timing_dispatches[0]) < ATT_ITERATION or ATT_ITERATION <= ctx.warmups:
        fatal("ATT iteration must be a post-warmup dispatch that exists")
    expected_id = ctx.timing_dispatches[0][ATT_ITERATION - 1]["dispatch_id"]
    att_dir = run_pass(ctx, "att", ["--att", "--kernel-include-regex", target_regex(ctx),
                                    "--kernel-iteration-range", f"[{ATT_ITERATION}]"],
                       base=ctx.vdir)
    files = att_files(att_dir)
    ctx.att_dir = att_dir
    ctx.att_manifest = [{"variant": ctx.name, "kernel_name": ctx.target,
                         "iteration_index": ATT_ITERATION, "expected_dispatch_id": expected_id,
                         "kind": att_kind(rel), "bytes": size,
                         "path": os.path.join("att", rel)} for rel, size in files]
    check_att(ctx, att_dir, expected_id, files)


def check_att(ctx, att_dir, expected_id, files):
    kinds = {}
    for entry in ctx.att_manifest:
        kinds[entry["kind"]] = kinds.get(entry["kind"], 0) + 1
    ids = {int(m) for rel, _ in files for m in re.findall(r"dispatch_(\d+)", rel)}
    ids |= {int(m) for rel, _ in files for m in re.findall(r"_(\d+)\.att$", rel)}
    raw_sizes = [e["bytes"] for e in ctx.att_manifest if e["kind"] == "att_raw"]
    traced = sorted({rel for rel, _ in files if re.search(r"(^|/)ui_output_[^/]*/code\.json$", rel)})
    symbols = kernel_symbols(find_single_db(att_dir))
    problems = []
    if ids != {expected_id}:
        problems.append(f"dispatch ids in artifacts {sorted(ids)} != {expected_id}")
    if len(traced) != 1 or len(raw_sizes) != 1:
        problems.append(f"{len(traced)} decoded dispatches and {len(raw_sizes)} raw files, expected 1 each")
    if not raw_sizes or min(raw_sizes) == 0:
        problems.append("missing or empty raw .att file")
    for needed in ("att_ui_json", "att_stats_csv", "code_object"):
        if not kinds.get(needed):
            problems.append(f"no {needed} artifacts")
    if ctx.target not in symbols:
        problems.append("target kernel symbol not present in ATT rocpd database")
    ctx.check("V12", "ATT captured exactly one post-warmup target dispatch",
              "FAIL" if problems else "PASS", f"ids={sorted(ids)} kinds={kinds}", expected_id,
              "exact", "; ".join(problems))


# ---------------------------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------------------------
REF_SOL = "rocprof-compute gfx950 0200_system_speed_of_light"
REF_SPI = "rocprof-compute gfx950 0600_workgroup_manager_spi"
REF_WAVE = "rocprof-compute gfx950 0700_wavefront (SQ wave counters are quad-cycles)"
REF_PIPE = "rocprof-compute gfx950 1100_compute_units_compute_pipeline"
REF_LDS = "rocprof-compute gfx950 1200_local_data_share_lds"
REF_ICACHE = "rocprof-compute gfx950 1300_instruction_cache"
REF_SCACHE = "rocprof-compute gfx950 1400_scalar_l1_data_cache"
REF_TATD = "rocprof-compute gfx950 1500_address_processing_unit_and_data_return_path_ta_td"
REF_VL1D = "rocprof-compute gfx950 1600_vector_l1_data_cache"
REF_L2 = "rocprof-compute gfx950 1700_l2_cache"
REF_ANALYTIC = "analytic (problem shape and epilogue operands)"
REF_DRV = "rocprofv3 derived_counters.xml"
PEAK_HBM_GBS = 8000.0
HW_MAX_WAVES_PER_CU = SIMDS_PER_CU * MAX_WAVES_PER_SIMD


class Sample:
    """Per-dispatch counter view. Each counter is read from the first listed pass that collected
    it, at the same post-warmup dispatch index, so a rate whose counters share one pass is
    normalised by that pass's own GRBM_GUI_ACTIVE and duration (F2)."""

    def __init__(self, passes, groups, index, consts):
        self.passes, self.groups, self.index, self.consts = passes, groups, index, consts

    def __call__(self, name):
        first = self.passes[self.groups[0]]
        if name == "@duration_us":
            return first["dur_us"][self.index]
        if name == "@cu_cycles":
            return self("GRBM_GUI_ACTIVE") * self.consts["cu_count"]
        for group in self.groups:
            values = self.passes[group]["counters"].get(name)
            if values is not None:
                return values[self.index]
        fatal(f"counter {name} is not collected by passes {self.groups}")


def pct(a, b):
    return 100.0 * safe_div(a, b)


def ceil_div(a, b):
    return -(-a // b)


def compulsory_bytes(use_beta, use_rms, mt0, mt1):
    """Analytic minimum DRAM traffic for one dispatch (batch 1, bf16 operands)."""
    a = PROBLEM_M * PROBLEM_K * BF16_BYTES
    b = PROBLEM_N * PROBLEM_K * BF16_BYTES
    d = PROBLEM_M * PROBLEM_N * BF16_BYTES
    read, write = a + b, d
    if use_beta:
        read += d
    if use_rms:
        # Mirrors ContractionSolution::partialRMSPartialBufBytes: tokens (N) padded by MT1, one fp32
        # per feature (M) macro tile.
        partial = ceil_div(PROBLEM_N, mt1) * mt1 * ceil_div(PROBLEM_M, mt0) * 4
        read += d + PROBLEM_M * BF16_BYTES
        write += d + partial
    return read, write


# Entries of DERIVED_TABLE. Each is evaluated once per post-warmup dispatch of its pass(es); the
# row value is the median of those per-dispatch values and cv their coefficient of variation.
# groups == () marks a timing-regime row that uses only constants (cv comes from the timing passes).
DERIVED_TABLE = []


def row(category, metric, unit, groups, fn, formula, reference, regime="pmc", kind="derived"):
    groups = (groups,) if isinstance(groups, str) else tuple(groups)
    DERIVED_TABLE.append({"category": category, "metric": metric, "unit": unit, "groups": groups,
                          "fn": fn, "formula": formula, "reference": reference, "regime": regime,
                          "kind": kind})


def row_dual(category, metric, unit, groups, fn, formula, reference):
    """A duration-dependent rate reported in both regimes (F3); fn(v, c, duration_us)."""
    row(category, f"{metric}_timing_regime", unit, groups,
        lambda v, c: fn(v, c, c["timing_us"]),
        f"{formula}; duration = timing-pass median", reference, "timing")
    row(category, f"{metric}_pmc_regime", unit, groups,
        lambda v, c: fn(v, c, v("@duration_us")),
        f"{formula}; duration = same-pass per-dispatch duration", reference, "pmc")


def evaluate_entry(ctx, entry, consts):
    def safe(v):
        try:
            return entry["fn"](v, consts)
        except (ZeroDivisionError, OverflowError, ValueError):
            return NAN
    if not entry["groups"]:
        return [safe(None)]
    n_post = ctx.expected_total - ctx.warmups
    return [safe(Sample(ctx.passes, entry["groups"], i, consts)) for i in range(n_post)]


def apply_table(ctx, consts):
    rows = []
    for e in DERIVED_TABLE:
        s = Stats(evaluate_entry(ctx, e, consts))
        formula = e["formula"]
        n, cv = s.n, s.cv
        if not e["groups"]:
            n, cv = consts["timing_n"], consts["timing_cv"]
        if len(e["groups"]) > 1:
            formula += (f"; CROSS-PASS: counters taken from passes {list(e['groups'])} (first pass "
                        "holding each counter), paired by dispatch index; see cv")
        rows.append(metric_row(e["category"], e["metric"], s.median, e["unit"], e["kind"], "median",
                               n, cv, formula, e["reference"], e["regime"]))
    return rows


def metric_row(category, metric, value, unit, kind, aggregation, n, cv, formula, reference,
               regime):
    return {"category": category, "metric": metric, "value": value, "unit": unit, "kind": kind,
            "regime": regime, "aggregation": aggregation, "n_dispatches": n, "cv": cv,
            "formula": formula, "reference": reference}


GU = "GRBM_GUI_ACTIVE"
GPU_CYCLES_CAVEAT = ("gpu_cycles (per-XCD GRBM_GUI_ACTIVE) compares variants within the PMC regime only; "
                     "cost ratios differ between gpu_cycles, PMC-regime duration and timing-regime "
                     "duration, and the data cannot say which is the deployable cost; timing-regime "
                     "microseconds are the measured wall time")
BUSY_NOTE = ("SQ_BUSY_CU_CYCLES == SQ_WAVE_CYCLES is observed here (one resident wave per SIMD; check "
             "V18 reports the ratio), so cu_busy_pct and achieved occupancy are not independent evidence")
NOT_COMPARABLE = ("measured/compulsory ratios are not comparable across variants (each divides by "
                  "its own compulsory model); compare excess_*_bytes_vs_none instead")


# --- timing and compute -----------------------------------------------------------------------
row("timing", "tflops_bf16", "TFLOP/s", (), lambda v, c: PROBLEM_FLOPS / (c["timing_us"] * 1e6),
    "2*M*N*K / timing-pass median duration", REF_ANALYTIC, "timing")
row("timing", "tflops_bf16_pmc_regime", "TFLOP/s", "sq_occ",
    lambda v, c: PROBLEM_FLOPS / (v("@duration_us") * 1e6),
    "2*M*N*K / same-pass per-dispatch duration", REF_ANALYTIC)
row("timing", "pct_peak_bf16", "percent", (),
    lambda v, c: 100.0 * PROBLEM_FLOPS / (c["timing_us"] * 1e6) / c["peak_tflops"],
    "tflops_bf16 / (cu * 4096 flop/clk * max_sclk), timing regime", REF_ANALYTIC, "timing")
row("timing", "gpu_cycles", "cycles/XCD", "sq_occ", lambda v, c: v(GU),
    "GRBM_GUI_ACTIVE / num_xcc, pass sq_occ; " + GPU_CYCLES_CAVEAT, REF_SOL)
row("timing", "gpu_cycles_grbm_count", "cycles/XCD", "sq_occ", lambda v, c: v("GRBM_COUNT"),
    "GRBM_COUNT / num_xcc, pass sq_occ", REF_SOL)
row("timing", "implied_sclk_mhz", "MHz", "sq_occ", lambda v, c: v(GU) / v("@duration_us"),
    "GRBM_GUI_ACTIVE per XCD / same-pass duration (cycles per us)", "analytic")
row("timing", "duration_vs_tensile_pct", "percent", (),
    lambda v, c: 100.0 * (c["timing_us"] - c["tensile_us"]) / c["tensile_us"],
    "(timing-pass median - Tensile WinnerTimeUS) / WinnerTimeUS", "Tensile CSVWinner", "timing")

row("compute", "mfma_flops_bf16", "FLOP", "sq_valu",
    lambda v, c: MFMA_MOPS_FLOPS * v("SQ_INSTS_VALU_MFMA_MOPS_BF16"),
    "512 * SQ_INSTS_VALU_MFMA_MOPS_BF16", REF_SOL)
row_dual("compute", "mfma_tflops_counter", "TFLOP/s", "sq_valu",
         lambda v, c, d: MFMA_MOPS_FLOPS * v("SQ_INSTS_VALU_MFMA_MOPS_BF16") / (d * 1e6),
         "512 * SQ_INSTS_VALU_MFMA_MOPS_BF16 / duration", REF_SOL)
row("compute", "mfma_inst_count", "inst", "sq_valu", lambda v, c: v("SQ_INSTS_MFMA"),
    "SQ_INSTS_MFMA (all MFMA types, including the non-bf16 accumulator-clear prologue)", REF_SOL)
row("compute", "mfma_bf16_inst_count", "inst", "sq_valu",
    lambda v, c: v("SQ_INSTS_VALU_MFMA_BF16"), "SQ_INSTS_VALU_MFMA_BF16", REF_SOL)
row("compute", "mfma_flops_per_bf16_inst", "FLOP/inst", "sq_valu",
    lambda v, c: MFMA_MOPS_FLOPS * v("SQ_INSTS_VALU_MFMA_MOPS_BF16") / v("SQ_INSTS_VALU_MFMA_BF16"),
    "512 * MOPS_BF16 / MFMA_BF16", REF_SOL)
row("compute", "mfma_util_pct", "percent", "sq_occ",
    lambda v, c: pct(v("SQ_VALU_MFMA_BUSY_CYCLES"), 4 * v("@cu_cycles")),
    "100 * SQ_VALU_MFMA_BUSY_CYCLES / (4 * GRBM_GUI_ACTIVE * cu), pass sq_occ", REF_SOL + " 'MFMA Utilization'")
row("compute", "valu_util_pct", "percent", "sq_valu",
    lambda v, c: pct(v("SQ_ACTIVE_INST_VALU"), v("@cu_cycles")),
    "100 * SQ_ACTIVE_INST_VALU / (GRBM_GUI_ACTIVE * cu), pass sq_valu", REF_SOL + " 'VALU Utilization'")
row("compute", "valu_active_threads", "work-items", "sq_valu",
    lambda v, c: safe_div(v("SQ_THREAD_CYCLES_VALU"), v("SQ_ACTIVE_INST_VALU")),
    "SQ_THREAD_CYCLES_VALU / SQ_ACTIVE_INST_VALU", REF_SOL + " 'VALU Active Threads'")
row("compute", "ipc", "inst/cycle", "sq_occ",
    lambda v, c: safe_div(v("SQ_INSTS"), v("SQ_BUSY_CU_CYCLES")),
    "SQ_INSTS / SQ_BUSY_CU_CYCLES", REF_SOL + " 'IPC'")
row("compute", "cu_busy_pct", "percent", "sq_occ",
    lambda v, c: pct(v("SQ_BUSY_CU_CYCLES"), v("@cu_cycles")),
    "100 * SQ_BUSY_CU_CYCLES / (GRBM_GUI_ACTIVE * cu); " + BUSY_NOTE, REF_SPI)
row("compute", "valu_utilization_drv", "percent", "drv_valu", lambda v, c: v("VALUUtilization"),
    "rocprofv3 derived VALUUtilization = 100*SQ_THREAD_CYCLES_VALU/(SQ_ACTIVE_INST_VALU*64)", REF_DRV)
row("compute", "valu_busy_drv", "percent", "drv_valu", lambda v, c: v("VALUBusy"),
    "rocprofv3 derived VALUBusy = 100*SQ_ACTIVE_INST_VALU*4/SIMD_NUM/GRBM_GUI_ACTIVE", REF_DRV)

PER_WAVE_INSTS = [("insts", "SQ_INSTS", "sq_occ"), ("valu_insts", "SQ_INSTS_VALU", "sq_occ"),
                  ("mfma_insts", "SQ_INSTS_MFMA", "sq_occ"), ("salu_insts", "SQ_INSTS_SALU", "sq_mix"),
                  ("smem_insts", "SQ_INSTS_SMEM", "sq_mix"), ("vmem_rd_insts", "SQ_INSTS_VMEM_RD", "sq_mix"),
                  ("vmem_wr_insts", "SQ_INSTS_VMEM_WR", "sq_mix"), ("flat_insts", "SQ_INSTS_FLAT", "sq_mix"),
                  ("branch_insts", "SQ_INSTS_BRANCH", "sq_mix"), ("vmem_insts", "SQ_INSTS_VMEM", "sq_mix2"),
                  ("sendmsg_insts", "SQ_INSTS_SENDMSG", "sq_mix2")]
for _name, _counter, _group in PER_WAVE_INSTS:
    row("compute", f"{_name}_per_wave", "inst/wave", _group,
        (lambda counter: lambda v, c: safe_div(v(counter), v("SQ_WAVES")))(_counter),
        f"{_counter} / SQ_WAVES, pass {_group}", REF_WAVE)

# --- wavefront stalls (0700): SQ_WAIT_ANY is Dependency Wait, SQ_WAIT_INST_ANY is Issue Wait ---
row("stalls", "wave_cycles_per_wave", "cycles", "sq_occ",
    lambda v, c: 4 * safe_div(v("SQ_WAVE_CYCLES"), v("SQ_WAVES")),
    "4 * SQ_WAVE_CYCLES / SQ_WAVES", REF_WAVE + " 'Wave Cycles'")
row("stalls", "dependency_wait_pct", "percent", "sq_wait",
    lambda v, c: pct(v("SQ_WAIT_ANY"), v("SQ_WAVE_CYCLES")),
    "100 * SQ_WAIT_ANY / SQ_WAVE_CYCLES (0700 'Dependency Wait Cycles' = 4*SQ_WAIT_ANY/denom; "
    "the quad-cycle factor cancels against SQ_WAVE_CYCLES)", REF_WAVE + " 'Dependency Wait Cycles'")
row("stalls", "issue_wait_pct", "percent", "sq_wait",
    lambda v, c: pct(v("SQ_WAIT_INST_ANY"), v("SQ_WAVE_CYCLES")),
    "100 * SQ_WAIT_INST_ANY / SQ_WAVE_CYCLES (0700 'Issue Wait Cycles' = 4*SQ_WAIT_INST_ANY/denom)",
    REF_WAVE + " 'Issue Wait Cycles'")
row("stalls", "active_inst_any_pct", "percent", "sq_wait",
    lambda v, c: pct(v("SQ_ACTIVE_INST_ANY"), v("SQ_WAVE_CYCLES")),
    "100 * SQ_ACTIVE_INST_ANY / SQ_WAVE_CYCLES (0700 'Active Cycles' = 4*SQ_ACTIVE_INST_ANY/denom)",
    REF_WAVE + " 'Active Cycles'")
for _metric, _counter, _label in (("active_inst_valu_pct", "SQ_ACTIVE_INST_VALU", "VALU"),
                                  ("active_inst_lds_pct", "SQ_ACTIVE_INST_LDS", "LDS"),
                                  ("active_inst_vmem_pct", "SQ_ACTIVE_INST_VMEM", "VMEM")):
    row("stalls", _metric, "percent", "sq_wait",
        (lambda counter: lambda v, c: pct(v(counter), v("SQ_WAVE_CYCLES")))(_counter),
        f"100 * {_counter} / SQ_WAVE_CYCLES (wave cycles spent working on {_label} instructions)",
        "rocprofv3 -L counter description")

# --- LDS (1200) ---------------------------------------------------------------------------------
row("lds", "utilization_pct", "percent", "sq_lds",
    lambda v, c: pct(v("SQ_LDS_IDX_ACTIVE"), v("@cu_cycles")),
    "100 * SQ_LDS_IDX_ACTIVE / (GRBM_GUI_ACTIVE * cu)", REF_LDS + " 'Utilization'")
row("lds", "access_rate_pct", "percent", "sq_wait",
    lambda v, c: 200.0 * safe_div(v("SQ_ACTIVE_INST_LDS"), v("@cu_cycles")),
    "200 * SQ_ACTIVE_INST_LDS / (GRBM_GUI_ACTIVE * cu)", REF_LDS + " 'Access Rate'")
row("lds", "bank_conflict_rate_pct", "percent", "sq_lds",
    lambda v, c: 3.125 * safe_div(v("SQ_LDS_BANK_CONFLICT"),
                                  v("SQ_LDS_IDX_ACTIVE") - v("SQ_LDS_BANK_CONFLICT")),
    "3.125 * SQ_LDS_BANK_CONFLICT / (SQ_LDS_IDX_ACTIVE - SQ_LDS_BANK_CONFLICT)",
    REF_LDS + " 'Bank Conflict Rate' (unit Percent)")
row("lds", "bank_conflicts_per_access", "cycles/access", "sq_lds",
    lambda v, c: safe_div(v("SQ_LDS_BANK_CONFLICT"),
                          v("SQ_LDS_IDX_ACTIVE") - v("SQ_LDS_BANK_CONFLICT")),
    "SQ_LDS_BANK_CONFLICT / (SQ_LDS_IDX_ACTIVE - SQ_LDS_BANK_CONFLICT)", REF_LDS + " 'Bank Conflicts/Access'")
row("lds", "bank_conflict_pct_of_idx_active", "percent", "sq_lds",
    lambda v, c: pct(v("SQ_LDS_BANK_CONFLICT"), v("SQ_LDS_IDX_ACTIVE")),
    "100 * SQ_LDS_BANK_CONFLICT / SQ_LDS_IDX_ACTIVE; relation to the official bank_conflict_rate_pct = "
    "3.125 * C / (IDX - C): with p = this row / 100, rate = 3.125 * p / (1 - p), about this row / 32, "
    "so the two differ by the 3.125 factor of the official formula and the percent scale (about 32x)", "analytic from rocprof-compute 1200 counters")
row("lds", "bank_conflict_pct_of_gpu_time", "percent", "sq_lds",
    lambda v, c: pct(v("SQ_LDS_BANK_CONFLICT"), v("@cu_cycles")),
    "100 * SQ_LDS_BANK_CONFLICT / (GRBM_GUI_ACTIVE * cu), the rocprofv3 LDSBankConflict form", REF_DRV)
row("lds", "bank_conflict_pct_of_gpu_time_drv", "percent", "drv_lds", lambda v, c: v("LDSBankConflict"),
    "rocprofv3 derived LDSBankConflict = 100*SQ_LDS_BANK_CONFLICT/GRBM_GUI_ACTIVE/CU_NUM", REF_DRV)
row("lds", "addr_conflict_pct_of_idx_active", "percent", "sq_lds",
    lambda v, c: pct(v("SQ_LDS_ADDR_CONFLICT"), v("SQ_LDS_IDX_ACTIVE")),
    "100 * SQ_LDS_ADDR_CONFLICT / SQ_LDS_IDX_ACTIVE", "analytic from rocprof-compute 1200 counters")
row("lds", "unaligned_stall_pct_of_idx_active", "percent", "sq_lds",
    lambda v, c: pct(v("SQ_LDS_UNALIGNED_STALL"), v("SQ_LDS_IDX_ACTIVE")),
    "100 * SQ_LDS_UNALIGNED_STALL / SQ_LDS_IDX_ACTIVE", "analytic from rocprof-compute 1200 counters")
row("lds", "insts_per_wave", "inst/wave", "sq_lds",
    lambda v, c: safe_div(v("SQ_INSTS_LDS"), v("SQ_WAVES")), "SQ_INSTS_LDS / SQ_WAVES",
    REF_LDS + " 'LDS Instructions' (per wave)")
row("lds", "load_insts_per_wave", "inst/wave", "sq_lds_inst",
    lambda v, c: safe_div(v("SQ_INSTS_LDS_LOAD"), v("SQ_WAVES")), "SQ_INSTS_LDS_LOAD / SQ_WAVES", REF_LDS)
row("lds", "store_insts_per_wave", "inst/wave", "sq_lds_inst",
    lambda v, c: safe_div(v("SQ_INSTS_LDS_STORE"), v("SQ_WAVES")), "SQ_INSTS_LDS_STORE / SQ_WAVES", REF_LDS)
row("lds", "issue_wait_share_pct", "percent", "sq_wait",
    lambda v, c: pct(v("SQ_WAIT_INST_LDS"), v("SQ_WAVE_CYCLES")),
    "100 * SQ_WAIT_INST_LDS / SQ_WAVE_CYCLES (wave cycles waiting for LDS instruction issue)",
    "rocprofv3 -L counter description")

# --- instruction and scalar caches (1300, 1400) -------------------------------------------------
row("icache", "l1i_hit_rate_pct", "percent", "sqc_icache",
    lambda v, c: pct(v("SQC_ICACHE_HITS"), v("SQC_ICACHE_HITS") + v("SQC_ICACHE_MISSES")
                     + v("SQC_ICACHE_MISSES_DUPLICATE")),
    "100 * HITS / (HITS + MISSES + MISSES_DUPLICATE)", REF_ICACHE + " 'Cache Hit Rate'")
for _metric, _counter in (("l1i_requests", "SQC_ICACHE_REQ"), ("l1i_hits", "SQC_ICACHE_HITS"),
                          ("l1i_misses", "SQC_ICACHE_MISSES"),
                          ("l1i_misses_duplicate", "SQC_ICACHE_MISSES_DUPLICATE")):
    row("icache", _metric, "requests", "sqc_icache",
        (lambda counter: lambda v, c: v(counter))(_counter), _counter, REF_ICACHE)
row("icache", "ifetch_per_wave", "fetches/wave", "sq_mix2",
    lambda v, c: safe_div(v("SQ_IFETCH"), v("SQ_WAVES")), "SQ_IFETCH / SQ_WAVES", REF_ICACHE)
row("scache", "sl1d_hit_rate_pct", "percent", "sqc_dcache",
    lambda v, c: pct(v("SQC_DCACHE_HITS"), v("SQC_DCACHE_HITS") + v("SQC_DCACHE_MISSES")
                     + v("SQC_DCACHE_MISSES_DUPLICATE")),
    "100 * HITS / (HITS + MISSES + MISSES_DUPLICATE)", REF_SCACHE + " 'Cache Hit Rate'")
for _metric, _counter in (("sl1d_requests", "SQC_DCACHE_REQ"), ("sl1d_hits", "SQC_DCACHE_HITS"),
                          ("sl1d_misses", "SQC_DCACHE_MISSES"),
                          ("sl1d_misses_duplicate", "SQC_DCACHE_MISSES_DUPLICATE")):
    row("scache", _metric, "requests", "sqc_dcache",
        (lambda counter: lambda v, c: v(counter))(_counter), _counter, REF_SCACHE)

# --- address processing, data return and vL1D (1500, 1600) -------------------------------------
TA_BUFFER_NOTE = ("-L: 'Number of buffer coalesced {d} cycles issued to TC'; rocprof-compute 1500 "
                  "labels it 'Spill/Stack Coalesced {D}', but Tensile loads its operands with "
                  "buffer instructions so it is not spill-specific here")
COALESCING_LABEL = "rocprof-compute Coalescing (lane-count based)"
COALESCING_WARNING = ("rocprof-compute Coalescing = 1600 / avg active lanes per VMEM wavefront; equals "
                      "25% when all 64 lanes are active; higher values only mean some wavefronts had "
                      "inactive lanes; not a coalescing-efficiency metric on gfx950; see the coalescing "
                      "category rows for request-level measures")
HIT_RATE_WARNING = ("mixes 128-B L2 requests (READ_REQ, WRITE_REQ) with 64-B tag accesses "
                    "(TCP_TOTAL_CACHE_ACCESSES), so a hit rate of at least 50% appears by construction "
                    "for streamed loads and cross-variant differences reflect request size, not locality")
row("l1", "coalescing_pct_lane_count_based", "percent", "tcp_coal",
    lambda v, c: 100.0 * 64 * safe_div(v("TA_TOTAL_WAVEFRONTS_sum"), 4 * v("TCP_TOTAL_ACCESSES_sum")),
    f"{COALESCING_LABEL}: 100 * 64 * TA_TOTAL_WAVEFRONTS / (4 * TCP_TOTAL_ACCESSES); "
    + COALESCING_WARNING, REF_VL1D + " 'Coalescing'")
row("l1", "ta_total_wavefronts", "wavefronts", "tcp_coal", lambda v, c: v("TA_TOTAL_WAVEFRONTS_sum"),
    "TA_TOTAL_WAVEFRONTS", REF_VL1D)
row("l1", "tcp_total_accesses", "lane-requests", "tcp_coal", lambda v, c: v("TCP_TOTAL_ACCESSES_sum"),
    "TCP_TOTAL_ACCESSES (= TCP_TOTAL_READ + TCP_TOTAL_NONREAD); counts active lanes (pixels/buffers "
    "from TA), not cache-line requests", REF_VL1D)
row("l1", "tcp_lane_requests_per_vmem_wavefront", "lane-requests/wavefront", "tcp_coal",
    lambda v, c: safe_div(v("TCP_TOTAL_ACCESSES_sum"), v("TA_TOTAL_WAVEFRONTS_sum")),
    "TCP_TOTAL_ACCESSES / TA_TOTAL_WAVEFRONTS (same pass tcp_coal); 64 means every wave64 VMEM "
    "instruction had all lanes active, which is what pins coalescing_pct_lane_count_based at 25%", REF_VL1D)
row("l1", "vl1d_hit_rate_pct", "percent", ("tcp_coal", "tcp_atomic"),
    lambda v, c: 100.0 - pct(v("TCP_TCC_READ_REQ_sum") + v("TCP_TCC_WRITE_REQ_sum")
                             + v("TCP_TCC_ATOMIC_WITH_RET_REQ_sum")
                             + v("TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum"), v("TCP_TOTAL_CACHE_ACCESSES_sum")),
    "100 - 100 * (READ_REQ + WRITE_REQ + ATOMIC_WITH_RET_REQ + ATOMIC_WITHOUT_RET_REQ) / "
    f"TCP_TOTAL_CACHE_ACCESSES; {HIT_RATE_WARNING}; the TCP limit of 4 forces the atomic counters into "
    "pass tcp_atomic", REF_VL1D + " 'Hit rate'")
row("l1", "vl1d_hit_rate_excl_atomics_pct", "percent", "tcp_coal",
    lambda v, c: 100.0 - pct(v("TCP_TCC_READ_REQ_sum") + v("TCP_TCC_WRITE_REQ_sum"),
                             v("TCP_TOTAL_CACHE_ACCESSES_sum")),
    f"same-pass form without the atomic request terms (single pass tcp_coal); {HIT_RATE_WARNING}", REF_VL1D)
row("l1", "vl1d_atomic_reqs", "requests", "tcp_atomic",
    lambda v, c: v("TCP_TCC_ATOMIC_WITH_RET_REQ_sum") + v("TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum"),
    "TCP_TCC_ATOMIC_WITH_RET_REQ + TCP_TCC_ATOMIC_WITHOUT_RET_REQ (0 means the cross-pass hit-rate "
    "mix is exact)", REF_VL1D)
row_dual("l1", "vl1d_bw_gbs", "GB/s", "tcp_coal",
         lambda v, c, d: 128 * v("TCP_TOTAL_CACHE_ACCESSES_sum") / (d * 1e3),
         "128 * TCP_TOTAL_CACHE_ACCESSES / duration", REF_VL1D + " 'Cache BW'")
row("l1", "ta_busy_pct", "percent", "tcp_coal", lambda v, c: pct(v("TA_TA_BUSY_sum"), v("@cu_cycles")),
    "100 * TA_TA_BUSY / (GRBM_GUI_ACTIVE * cu)", REF_TATD)
row("l1", "td_busy_pct", "percent", "td_busy", lambda v, c: pct(v("TD_TD_BUSY_sum"), v("@cu_cycles")),
    "100 * TD_TD_BUSY / (GRBM_GUI_ACTIVE * cu)", REF_TATD)
for _metric, _counter, _group in (
        ("ta_addr_stall_tc_pct", "TA_ADDR_STALLED_BY_TC_CYCLES_sum", "ta_stall_tc"),
        ("ta_data_stall_tc_pct", "TA_DATA_STALLED_BY_TC_CYCLES_sum", "ta_stall_tc"),
        ("ta_addr_stall_td_pct", "TA_ADDR_STALLED_BY_TD_CYCLES_sum", "ta_stall_td"),
        ("td_tc_stall_pct", "TD_TC_STALL_sum", "td_stall"),
        ("td_spi_stall_pct", "TD_SPI_STALL_sum", "td_stall"),
        ("ta_buffer_total_cycles_pct", "TA_BUFFER_TOTAL_CYCLES_sum", "ta_buf")):
    row("l1", _metric, "percent", _group,
        (lambda counter: lambda v, c: pct(v(counter), v("@cu_cycles")))(_counter),
        f"100 * {_counter} / (GRBM_GUI_ACTIVE * cu)", REF_TATD)
row("l1", "ta_buffer_coalesced_read_cycles_pct", "percent", "ta_rd",
    lambda v, c: pct(v("TA_BUFFER_COALESCED_READ_CYCLES_sum"), v("@cu_cycles")),
    "100 * TA_BUFFER_COALESCED_READ_CYCLES / (GRBM_GUI_ACTIVE * cu); "
    + TA_BUFFER_NOTE.format(d="read", D="Read"), REF_TATD)
row("l1", "ta_buffer_coalesced_read_cycles_per_wavefront", "cycles/wavefront", "ta_rd",
    lambda v, c: safe_div(v("TA_BUFFER_COALESCED_READ_CYCLES_sum"), v("TA_BUFFER_READ_WAVEFRONTS_sum")),
    "TA_BUFFER_COALESCED_READ_CYCLES / TA_BUFFER_READ_WAVEFRONTS; " + TA_BUFFER_NOTE.format(d="read", D="Read"),
    REF_TATD)
row("l1", "ta_buffer_coalesced_write_cycles_pct", "percent", "ta_wr",
    lambda v, c: pct(v("TA_BUFFER_COALESCED_WRITE_CYCLES_sum"), v("@cu_cycles")),
    "100 * TA_BUFFER_COALESCED_WRITE_CYCLES / (GRBM_GUI_ACTIVE * cu); "
    + TA_BUFFER_NOTE.format(d="write", D="Write"), REF_TATD)
row("l1", "ta_buffer_coalesced_write_cycles_per_wavefront", "cycles/wavefront", "ta_wr",
    lambda v, c: safe_div(v("TA_BUFFER_COALESCED_WRITE_CYCLES_sum"), v("TA_BUFFER_WRITE_WAVEFRONTS_sum")),
    "TA_BUFFER_COALESCED_WRITE_CYCLES / TA_BUFFER_WRITE_WAVEFRONTS; " + TA_BUFFER_NOTE.format(d="write", D="Write"),
    REF_TATD)
row("l1", "tcp_utilization_pct", "percent", "tcp_gate",
    lambda v, c: pct(v("TCP_GATE_EN2_sum"), v("TCP_GATE_EN1_sum")),
    "100 * TCP_GATE_EN2 / TCP_GATE_EN1", REF_VL1D + " 'Utilization'")
for _metric, _counter, _group in (
        ("tcp_stall_l2_data_pct", "TCP_PENDING_STALL_CYCLES_sum", "tcp_gate"),
        ("tcp_stall_l2_req_pct", "TCP_TCR_TCP_STALL_CYCLES_sum", "tcp_gate"),
        ("tcp_stall_ta_addr_pct", "TCP_TCP_TA_ADDR_STALL_CYCLES_sum", "tcp_ta"),
        ("tcp_stall_ta_data_pct", "TCP_TCP_TA_DATA_STALL_CYCLES_sum", "tcp_ta"),
        ("tcp_tag_conflict_read_pct", "TCP_READ_TAGCONFLICT_STALL_CYCLES_sum", "tcp_ta"),
        ("tcp_stall_lfifo_pct", "TCP_LFIFO_STALL_CYCLES_sum", "tcp_ret"),
        ("tcp_stall_rdret_pct", "TCP_TCR_RDRET_STALL_sum", "tcp_ret")):
    row("l1", _metric, "percent", _group,
        (lambda counter: lambda v, c: pct(v(counter), v("TCP_GATE_EN1_sum")))(_counter),
        f"100 * {_counter} / TCP_GATE_EN1", REF_VL1D)

# --- request-level coalescing (all from pass tcp_coal, same-pass ratios, except where noted) --------
REF_COAL = "request-level coalescing from rocprofv3 TCP/TA/SQ counters; analytic ideal"
row("coalescing", "l1_tag_accesses_per_vmem_wavefront", "tag accesses/wavefront", "tcp_coal",
    lambda v, c: safe_div(v("TCP_TOTAL_CACHE_ACCESSES_sum"), v("TA_TOTAL_WAVEFRONTS_sum")),
    "TCP_TOTAL_CACHE_ACCESSES / TA_TOTAL_WAVEFRONTS (same pass tcp_coal); a fully contiguous dwordx4 "
    "wave64 access (1024 B) needs 16 64-B tag accesses", REF_COAL)
row("coalescing", "l2_read_req_per_vmem_rd_inst", "requests/instruction", "tcp_coal",
    lambda v, c: safe_div(v("TCP_TCC_READ_REQ_sum"), v("SQ_INSTS_VMEM_RD")),
    "TCP_TCC_READ_REQ / SQ_INSTS_VMEM_RD (same pass tcp_coal); 8 = minimum for buffer_load_dwordx4 "
    "(1024 B/instr in 128-B lines); instruction-mix average: values below 8 mean narrower loads "
    "(buffer_load_dwordx2 -> 4 req/instr), above 8 mean sub-line requests (half-line 64-B requests); "
    "not a coalescing comparison across variants; epilogue reads of beta/rms are included in both terms", REF_COAL)
row("coalescing", "l2_write_req_per_vmem_wr_inst", "requests/instruction", "tcp_coal",
    lambda v, c: safe_div(v("TCP_TCC_WRITE_REQ_sum"), v("SQ_INSTS_VMEM_WR")),
    "TCP_TCC_WRITE_REQ / SQ_INSTS_VMEM_WR (same pass tcp_coal)", REF_COAL)
row("coalescing", "ideal_operand_over_total_l2_read_req", "ratio", "tcp_coal",
    lambda v, c: safe_div(c["ideal_l2_read_req"], v("TCP_TCC_READ_REQ_sum")),
    "ideal operand-load 128-B requests / total TCP_TCC_READ_REQ; equals operand-load efficiency only "
    "when the kernel has no epilogue reads (none variant); for beta/rms it is the operand share of all "
    "read requests, not an inefficiency of operand loads", REF_COAL)
row("coalescing", "epilogue_l2_read_req", "requests", "tcp_coal",
    lambda v, c: v("TCP_TCC_READ_REQ_sum") - c["ideal_l2_read_req"],
    "TCP_TCC_READ_REQ - operand_load_ideal_l2_read_req; assumes operand loads are ideal (supported by "
    "none = 1.000 and beta's exact 128 B/request)", REF_COAL)
row("coalescing", "epilogue_bytes_per_l2_read_req", "B/request", "tcp_coal",
    lambda v, c: NAN if c["epilogue_read_bytes"] == 0
    else safe_div(c["epilogue_read_bytes"], v("TCP_TCC_READ_REQ_sum") - c["ideal_l2_read_req"]),
    "analytic epilogue read bytes (C, or residual plus actual gamma L2 traffic) / epilogue_l2_read_req; "
    "gamma L2 traffic = tiles * MT0 * gamma bytes because every output tile re-reads its MT0-sized "
    "gamma slice (gamma is indexed by free0 = M), not the compulsory gamma size; 128 is full-line "
    "requests, 64 is half-line requests; undefined (empty) for the none variant, which reads no "
    "epilogue operand", REF_COAL)
row("coalescing", "l2_write_req_efficiency", "ratio", "tcp_coal",
    lambda v, c: safe_div(c["comp_write"] / 64.0, v("TCP_TCC_WRITE_REQ_sum")),
    "(analytic written bytes / 64) / TCP_TCC_WRITE_REQ; written bytes = D (+ residual-out and the "
    "partial-sum buffer for rms); 1.0 means every write request carries 64 B; for rms the shortfall is "
    "the partial-sum stores (small requests, see the excess_*_write rows)", REF_COAL)

# --- L2 (1700) and memory traffic ---------------------------------------------------------------
row("l2", "hit_rate_pct", "percent", "tcc_hit",
    lambda v, c: pct(v("TCC_HIT_sum"), v("TCC_HIT_sum") + v("TCC_MISS_sum")),
    "100 * TCC_HIT / (TCC_HIT + TCC_MISS)", REF_SOL)
row("l2", "requests", "requests", "tcc_hit", lambda v, c: v("TCC_REQ_sum"), "TCC_REQ", REF_SOL)
row_dual("l2", "bw_gbs", "GB/s", "tcc_hit", lambda v, c, d: 128 * v("TCC_REQ_sum") / (d * 1e3),
         "128 * TCC_REQ / duration", REF_SOL)
row("l2", "tag_stall_per_req", "cycles/request", "tcc_hit",
    lambda v, c: safe_div(v("TCC_TAG_STALL_sum"), v("TCC_REQ_sum")), "TCC_TAG_STALL / TCC_REQ", REF_L2)
row("l2", "ea_wr_stall_per_wrreq", "cycles/request", "tcc_ea",
    lambda v, c: safe_div(v("TCC_EA0_WRREQ_STALL_sum"), v("TCC_EA0_WRREQ_sum")),
    "TCC_EA0_WRREQ_STALL / TCC_EA0_WRREQ", REF_L2)
row("l2", "writebacks", "requests", "tcc_dram", lambda v, c: v("TCC_WRITEBACK_sum"), "TCC_WRITEBACK", REF_L2)


def fabric_read(v):
    return (32 * v("TCC_EA0_RDREQ_32B_sum") + 64 * v("TCC_EA0_RDREQ_64B_sum")
            + 128 * v("TCC_EA0_RDREQ_128B_sum"))


def fabric_write(v):
    return 64 * v("TCC_EA0_WRREQ_64B_sum") + 32 * (v("TCC_EA0_WRREQ_sum") - v("TCC_EA0_WRREQ_64B_sum"))


FABRIC_READ_FORMULA = "32*RDREQ_32B + 64*RDREQ_64B + 128*RDREQ_128B"
FABRIC_WRITE_FORMULA = "64*WRREQ_64B + 32*(WRREQ - WRREQ_64B)"
row("memory", "fabric_read_bytes", "B", "tcc_rd", lambda v, c: fabric_read(v), FABRIC_READ_FORMULA, REF_SOL)
row("memory", "fabric_write_bytes", "B", "tcc_ea", lambda v, c: fabric_write(v), FABRIC_WRITE_FORMULA, REF_SOL)
row("memory", "fetch_size_drv", "KiB", "drv_fetch", lambda v, c: v("FETCH_SIZE"),
    "rocprofv3 derived FETCH_SIZE", REF_DRV)
row("memory", "write_size_drv", "KiB", "drv_write", lambda v, c: v("WRITE_SIZE"),
    "rocprofv3 derived WRITE_SIZE", REF_DRV)
row("memory", "read_over_compulsory", "ratio", "tcc_rd", lambda v, c: safe_div(fabric_read(v), c["comp_read"]),
    f"({FABRIC_READ_FORMULA}) / compulsory read bytes; {NOT_COMPARABLE}", REF_ANALYTIC)
row("memory", "write_over_compulsory", "ratio", "tcc_ea",
    lambda v, c: safe_div(fabric_write(v), c["comp_write"]),
    f"({FABRIC_WRITE_FORMULA}) / compulsory write bytes; {NOT_COMPARABLE}", REF_ANALYTIC)
row("memory", "total_over_compulsory", "ratio", ("tcc_rd", "tcc_ea"),
    lambda v, c: safe_div(fabric_read(v) + fabric_write(v), c["comp_read"] + c["comp_write"]),
    f"fabric total / compulsory total; {NOT_COMPARABLE}", REF_ANALYTIC)
row("memory", "read_bytes_minus_compulsory", "B", "tcc_rd", lambda v, c: fabric_read(v) - c["comp_read"],
    "measured fabric read bytes - compulsory read bytes", REF_ANALYTIC)
row("memory", "write_bytes_minus_compulsory", "B", "tcc_ea",
    lambda v, c: fabric_write(v) - c["comp_write"], "measured fabric write bytes - compulsory write bytes",
    REF_ANALYTIC)
row_dual("memory", "bw_read_gbs", "GB/s", "tcc_rd", lambda v, c, d: fabric_read(v) / (d * 1e3),
         "fabric read bytes / duration", REF_ANALYTIC)
row_dual("memory", "bw_write_gbs", "GB/s", "tcc_ea", lambda v, c, d: fabric_write(v) / (d * 1e3),
         "fabric write bytes / duration", REF_ANALYTIC)
row_dual("memory", "bw_total_gbs", "GB/s", ("tcc_rd", "tcc_ea"),
         lambda v, c, d: (fabric_read(v) + fabric_write(v)) / (d * 1e3),
         "fabric read + write bytes / duration", REF_ANALYTIC)
row_dual("memory", "pct_of_peak_hbm", "percent", ("tcc_rd", "tcc_ea"),
         lambda v, c, d: 100.0 * (fabric_read(v) + fabric_write(v)) / (d * 1e3) / PEAK_HBM_GBS,
         f"bw_total_gbs / {PEAK_HBM_GBS:.0f} GB/s peak HBM (fabric bytes include MALL hits, so this is "
         "not DRAM traffic)", REF_ANALYTIC)

# --- occupancy (F5): theoretical, api and measured values are never mixed in one row ------------
row("occupancy", "sq_waves", "waves", "sq_occ", lambda v, c: v("SQ_WAVES"), "SQ_WAVES", REF_WAVE)
ACHIEVED_FORMULA = ("4 * SQ_WAVE_CYCLES / (GRBM_GUI_ACTIVE * cu), same pass sq_occ; SQ_WAVE_CYCLES is "
                    "in quad-cycles (x4), GRBM_GUI_ACTIVE is per XCD; " + BUSY_NOTE)
row("occupancy", "achieved_waves_per_cu", "waves/CU", "sq_occ",
    lambda v, c: 4 * v("SQ_WAVE_CYCLES") / v("@cu_cycles"), ACHIEVED_FORMULA, REF_WAVE, kind="measured")
row("occupancy", "achieved_waves_per_simd", "waves/SIMD", "sq_occ",
    lambda v, c: 4 * v("SQ_WAVE_CYCLES") / v("@cu_cycles") / SIMDS_PER_CU,
    ACHIEVED_FORMULA + ", divided by 4 SIMDs per CU", REF_WAVE, kind="measured")
row("occupancy", "achieved_pct_of_theoretical", "percent", "sq_occ",
    lambda v, c: pct(4 * v("SQ_WAVE_CYCLES") / v("@cu_cycles"), c["theoretical_waves_per_cu"]),
    "achieved_waves_per_cu / theoretical_waves_per_cu", "analytic", kind="measured")
row("occupancy", "achieved_pct_of_hw_max", "percent", "sq_occ",
    lambda v, c: pct(4 * v("SQ_WAVE_CYCLES") / v("@cu_cycles"), HW_MAX_WAVES_PER_CU),
    f"achieved_waves_per_cu / {HW_MAX_WAVES_PER_CU} (8 waves per SIMD x 4 SIMDs)", "analytic", kind="measured")
MEAN_OCC = "rocprofv3 derived MeanOccupancyPerCU = SQ_ACCUM_PREV_HIRES / GRBM_GUI_ACTIVE / CU_NUM, own pass"
row("occupancy", "mean_occupancy_per_cu", "waves/CU", "drv_occ_cu", lambda v, c: v("MeanOccupancyPerCU"),
    MEAN_OCC + "; unit established empirically by mean_occupancy_per_cu_over_measured", REF_DRV, kind="api")
row("occupancy", "mean_occupancy_per_cu_pct_of_theoretical", "percent", "drv_occ_cu",
    lambda v, c: pct(v("MeanOccupancyPerCU"), c["theoretical_waves_per_cu"]),
    "mean_occupancy_per_cu / theoretical_waves_per_cu", "analytic", kind="api")
row("occupancy", "mean_occupancy_per_cu_over_measured", "ratio", ("sq_occ", "drv_occ_cu"),
    lambda v, c: v("MeanOccupancyPerCU") / (4 * v("SQ_WAVE_CYCLES") / v("@cu_cycles")),
    "MeanOccupancyPerCU / achieved_waves_per_cu; a ratio near 1 means MeanOccupancyPerCU is in waves "
    "per CU (about 0.25 would mean waves per SIMD)", REF_DRV)
row("occupancy", "occupancy_percent", "percent of hw max", "drv_occ_pct", lambda v, c: v("OccupancyPercent"),
    f"rocprofv3 derived OccupancyPercent, own pass; percent of the {HW_MAX_WAVES_PER_CU} waves/CU hardware max",
    REF_DRV, kind="api")
row("occupancy", "occupancy_percent_waves_per_cu", "waves/CU", "drv_occ_pct",
    lambda v, c: v("OccupancyPercent") / 100.0 * HW_MAX_WAVES_PER_CU,
    f"OccupancyPercent / 100 * {HW_MAX_WAVES_PER_CU}", REF_DRV, kind="api")
row("occupancy", "occupancy_percent_pct_of_theoretical", "percent", "drv_occ_pct",
    lambda v, c: pct(v("OccupancyPercent") / 100.0 * HW_MAX_WAVES_PER_CU, c["theoretical_waves_per_cu"]),
    "occupancy_percent_waves_per_cu / theoretical_waves_per_cu", "analytic", kind="api")
row("occupancy", "occupancy_percent_over_mean_per_cu", "ratio", ("drv_occ_pct", "drv_occ_cu"),
    lambda v, c: (v("OccupancyPercent") / 100.0 * HW_MAX_WAVES_PER_CU) / v("MeanOccupancyPerCU"),
    "OccupancyPercent-derived waves/CU / MeanOccupancyPerCU (two independent api passes)", REF_DRV)
row("occupancy", "mean_occupancy_per_active_cu_raw", "unitless (raw, not interpreted)", "drv_occ_active",
    lambda v, c: v("MeanOccupancyPerActiveCU"),
    "rocprofv3 derived MeanOccupancyPerActiveCU = SQ_ACCUM_PREV_HIRES*4 / SQ_BUSY_CYCLES / CU_NUM, own pass; "
    "reported raw: it is not consistent with measured waves/CU so it is not used for validation", REF_DRV,
    kind="api")
def dispatch_geometry(d):
    wg_size = d["wg_x"] * d["wg_y"] * d["wg_z"]
    work_items = d["grid_x"] * d["grid_y"] * d["grid_z"]
    return wg_size, ceil_div(work_items, wg_size)


LLVM_BIN = os.environ.get("LLVM_BIN", "/opt/rocm/llvm/bin")
REF_DISASM = "llvm-objdump --mcpu=gfx950 of the code object"


def disassemble_mfma(variant):
    """Static MFMA census (F7). The code object must define exactly the target kernel."""
    code_object = variant["code_object"]
    rc, out, err = capture([os.path.join(LLVM_BIN, "llvm-nm"), "--defined-only", code_object])
    if rc != 0:
        fatal(f"llvm-nm failed: {err.strip()[:200]}")
    functions = [f[2] for f in (line.split() for line in out.splitlines()) if len(f) == 3 and f[1] == "T"]
    if functions != [variant["kernel_name"]]:
        fatal(f"code object defines kernels {functions}, expected only {variant['kernel_name']}")
    rc, out, err = capture([os.path.join(LLVM_BIN, "llvm-objdump"), "-d", "--no-show-raw-insn",
                            "--mcpu=gfx950", code_object], timeout_s=300)
    if rc != 0:
        fatal(f"llvm-objdump failed: {err.strip()[:200]}")
    counts = {}
    for name in re.findall(r"^\s+(v_mfma\w*)", out, re.M):
        counts[name] = counts.get(name, 0) + 1
    bf16 = sum(n for name, n in counts.items() if "bf16" in name)
    return {"counts": counts, "bf16": bf16, "prologue": sum(counts.values()) - bf16,
            "flops_per_bf16_inst": bf16_mfma_flops(counts)}


def bf16_mfma_flops(counts):
    """Flops of one bf16 MFMA, read from its mnemonic (v_mfma_f32_16x16x32_bf16 is 2 * 16 * 16 * 32)."""
    matches = {name: re.search(r"_(\d+)x(\d+)x(\d+)_?bf16", name) for name in counts if "bf16" in name}
    unparsed = sorted(name for name, match in matches.items() if not match)
    if unparsed:
        fatal(f"cannot read the mfma shape from bf16 mnemonics {unparsed}")
    shapes = {match.groups() for match in matches.values()}
    if len(shapes) != 1:
        fatal(f"expected exactly one bf16 MFMA shape in the code object, found {sorted(shapes)}")
    m, n, k = (int(x) for x in shapes.pop())
    return 2 * m * n * k


def kernel_macro_tile(kernel_name):
    m = re.search(r"_MT(\d+)x(\d+)x\d+_", kernel_name)
    if not m:
        fatal(f"cannot read the macro tile from kernel name {kernel_name}")
    return int(m.group(1)), int(m.group(2))


def mfma_expectation(ctx):
    """Dynamic MFMA count: bf16 work from the problem, plus the non-bf16 accumulator-clear prologue
    that every wave executes once per output tile (persistent data-parallel kernel)."""
    mt0, mt1 = kernel_macro_tile(ctx.target)
    tiles = ceil_div(PROBLEM_M, mt0) * ceil_div(PROBLEM_N, mt1) * PROBLEM_BATCH
    waves_per_tile = ctx.occ["waves_per_wg"]
    bf16 = PROBLEM_FLOPS // ctx.mfma["flops_per_bf16_inst"]
    prologue = ctx.mfma["prologue"] * tiles * waves_per_tile
    return {"tiles": tiles, "waves_per_tile": waves_per_tile, "bf16": bf16, "prologue": prologue,
            "total": bf16 + prologue}


def static_row(ctx, category, metric, value, unit, kind, regime, formula, reference):
    return metric_row(category, metric, value, unit, kind, "exact", len(ctx.timing_dispatches[0]),
                      0.0, formula, reference, regime)


def static_rows(ctx):
    """Resource usage, geometry, theoretical and API occupancy, and the MFMA census."""
    d = ctx.timing_dispatches[0][0]
    wg_size, n_wg = dispatch_geometry(d)
    occ = theoretical_occupancy(d["arch_vgpr"], d["accum_vgpr"], d["lds_bytes"], wg_size)
    hip = hip_api_occupancy(ctx.variant, wg_size, ctx.env)
    waves_per_wg = occ["waves_per_wg"]
    theoretical_waves = occ["wg_per_cu"] * waves_per_wg
    ctx.occ = {"wg_size": wg_size, "n_wg": n_wg, "grid_waves": n_wg * waves_per_wg, "hip": hip,
               "theoretical_wg": occ["wg_per_cu"], "waves_per_wg": waves_per_wg,
               "theoretical_waves": theoretical_waves, "limiter": occ["limiter"]}
    ctx.mfma_expect = mfma_expectation(ctx)
    resources = [("arch_vgpr", d["arch_vgpr"], "regs"), ("accum_vgpr", d["accum_vgpr"], "regs"),
                 ("vgpr_alloc_total", occ["total_vgpr"], "regs"), ("sgpr", d["sgpr"], "regs"),
                 ("lds_bytes", d["lds_bytes"], "B"), ("scratch_bytes", d["scratch_bytes"], "B"),
                 ("workgroup_size", wg_size, "threads"), ("grid_workgroups", n_wg, "WG"),
                 ("grid_waves", n_wg * waves_per_wg, "waves")]
    rows = [static_row(ctx, "resources", m, v, u, "static", "static",
                       "rocpd dispatch/kernel-symbol field", "rocpd_info_kernel_symbol")
            for m, v, u in resources]
    rows += theoretical_rows(ctx, occ, theoretical_waves)
    rows += api_occupancy_rows(ctx, hip, waves_per_wg, theoretical_waves)
    rows += mfma_static_rows(ctx)
    return rows


def theoretical_rows(ctx, occ, theoretical_waves):
    model = "gfx950 occupancy model: min(VGPR, wave-slot, LDS limits)"
    items = [("theoretical_wg_per_cu_vgpr", occ["limits"]["vgpr"], "WG/CU"),
             ("theoretical_wg_per_cu_waves", occ["limits"]["waves"], "WG/CU"),
             ("theoretical_wg_per_cu_lds", occ["limits"].get("lds", NAN), "WG/CU"),
             ("theoretical_wg_per_cu", occ["wg_per_cu"], "WG/CU"),
             ("theoretical_limiter", occ["limiter"], "name"),
             ("theoretical_waves_per_cu", theoretical_waves, "waves/CU"),
             ("theoretical_waves_per_simd", theoretical_waves / SIMDS_PER_CU, "waves/SIMD"),
             ("theoretical_pct_of_hw_max", 100.0 * theoretical_waves / HW_MAX_WAVES_PER_CU,
              "percent"),
             ("grid_limited_wg_per_cu", min(occ["wg_per_cu"], ceil_div(ctx.occ["n_wg"], ctx.cu_count)),
              "WG/CU")]
    return [static_row(ctx, "occupancy", m, v, u, "theoretical", "static", model,
                       "rocpd_info_kernel_symbol; gfx950 occupancy model") for m, v, u in items]


def api_occupancy_rows(ctx, hip, waves_per_wg, theoretical_waves):
    formula = "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor on the code object"
    items = [("hip_api_wg_per_cu", hip, "WG/CU"), ("hip_api_waves_per_cu", hip * waves_per_wg, "waves/CU"),
             ("hip_api_pct_of_theoretical", 100.0 * hip * waves_per_wg / theoretical_waves, "percent")]
    return [static_row(ctx, "occupancy", m, v, u, "api", "static", formula, "hip_occupancy.py")
            for m, v, u in items]


def mfma_static_rows(ctx):
    e = ctx.mfma_expect
    items = [("static_mfma_bf16_insts", ctx.mfma["bf16"], "inst", "static", "static count in the code object"),
             ("static_mfma_prologue_insts", ctx.mfma["prologue"], "inst", "static",
              "static non-bf16 MFMA count (accumulator-clear prologue)"),
             ("expected_mfma_tiles", e["tiles"], "tiles", "analytic",
              "ceil(M/MT0) * ceil(N/MT1) * batch, macro tile read from the kernel name"),
             ("expected_mfma_bf16_inst_count", e["bf16"], "inst", "analytic", "2*M*N*K / flops per bf16 MFMA (from the code object mnemonic)"),
             ("expected_mfma_inst_count", e["total"], "inst", "analytic",
              f"bf16 + prologue_mfma * tiles * waves_per_tile ({e['waves_per_tile']})")]
    return [static_row(ctx, "compute", m, v, u, "static" if r == "static" else "analytic", r, f, REF_DISASM)
            for m, v, u, r, f in items]


def timing_rows(ctx, timing, pmc):
    rows = []
    for label, s, regime, ref in (("duration_us", timing, "timing", "kernel-trace pass (no counters)"),
                                  ("duration_pmc_us", pmc, "pmc",
                                   "PMC passes pooled (counter collection perturbs execution)")):
        for agg in ("median", "mean", "std", "min", "max"):
            rows.append(metric_row("timing", label, getattr(s, agg), "us", "measured", agg, s.n, s.cv,
                                   f"end - start of post-warmup target dispatches ({agg})", ref, regime))
    ratio = safe_div(pmc.median, timing.median)
    cv = math.hypot(pmc.cv, timing.cv)
    formula = "median PMC-pass duration / median timing-pass duration; cv is the quadrature sum"
    rows.append(metric_row("timing", "pmc_over_timing_ratio", ratio, "ratio", "derived", "median",
                           min(pmc.n, timing.n), cv, formula, "analytic", "pmc"))
    rows.append(metric_row("timing", "pmc_perturbation_pct", 100.0 * (ratio - 1.0), "percent", "derived",
                           "median", min(pmc.n, timing.n), cv, "100 * (pmc_over_timing_ratio - 1)",
                           "analytic", "pmc"))
    return rows


def raw_counter_rows(ctx):
    rows = []
    for group, data in ctx.passes.items():
        for counter, values in data["counters"].items():
            s = Stats(values)
            formula = "per-dispatch value summed over instance rows"
            if counter in PER_XCD_COUNTERS:
                formula += ", divided by num_xcc"
            for agg in ("median", "mean"):
                rows.append(metric_row(f"raw_{group}", counter, getattr(s, agg), "count", "counter",
                                       agg, s.n, s.cv, formula, "rocpd_pmc_event", "pmc"))
    return rows


def analytic_rows(ctx, comp_read, comp_write, ideal_l2_read_req):
    items = [("compulsory_read_bytes", comp_read, "operand sizes (A, B, plus C or residual and gamma)"),
             ("compulsory_write_bytes", comp_write, "output sizes (D, plus ResidualOut and partial sums)")]
    rows = [metric_row("memory", m, v, "B", "analytic", "exact", 0, 0.0, f, REF_ANALYTIC, "analytic")
            for m, v, f in items]
    rows.append(metric_row(
        "coalescing", "operand_load_ideal_l2_read_req", ideal_l2_read_req, "requests", "analytic", "exact",
        0, 0.0, "tiles * (MT0 + MT1) * K * bytes_per_elem / 128: every output tile streams its A and B "
        f"panels once in 128-B requests (tiles={ctx.mfma_expect['tiles']}, macro tile from the kernel "
        "name)", REF_COAL, "analytic"))
    return rows


def ideal_operand_l2_read_req(ctx):
    mt0, mt1 = kernel_macro_tile(ctx.target)
    return ctx.mfma_expect["tiles"] * (mt0 + mt1) * PROBLEM_K * BF16_BYTES / 128.0


def epilogue_l2_read_bytes(ctx, comp_read, operand_bytes):
    """Epilogue read bytes as seen by L2: compulsory bytes, except that every output tile re-reads its
    MT0-sized gamma slice (gamma is indexed by free0 = M, see ClientProblemFactory setRMSGamma)."""
    epilogue = comp_read - operand_bytes
    if not ctx.variant["use_rms_epilogue"]:
        return epilogue
    mt0, _ = kernel_macro_tile(ctx.target)
    gamma_l2 = ctx.mfma_expect["tiles"] * mt0 * BF16_BYTES
    return epilogue - PROBLEM_M * BF16_BYTES + gamma_l2


def metric_index(rows):
    return {f"{r['category']}.{r['metric']}": r for r in rows if r["aggregation"] in ("median", "exact")}


def build_metrics(ctx):
    timing = Stats(ctx.timing_us)
    pmc = Stats([x for g in ctx.pass_durations_us.values() for x in g])
    rows = static_rows(ctx)
    mt0, mt1 = kernel_macro_tile(ctx.target)
    comp_read, comp_write = compulsory_bytes(ctx.variant["use_beta"], ctx.variant["use_rms_epilogue"], mt0, mt1)
    ideal_l2_read_req = ideal_operand_l2_read_req(ctx)
    operand_bytes = (PROBLEM_M + PROBLEM_N) * PROBLEM_K * BF16_BYTES
    consts = {"cu_count": ctx.cu_count, "num_xcc": ctx.num_xcc,
              "peak_tflops": ctx.cu_count * FLOPS_PER_CLK_PER_CU_BF16
              * float(ctx.agent["max_engine_clk_fcompute"]) / 1e6,
              "tensile_us": ctx.variant["tensile_winner_time_us"],
              "theoretical_waves_per_cu": ctx.occ["theoretical_waves"],
              "timing_us": timing.median, "timing_cv": timing.cv, "timing_n": timing.n,
              "comp_read": comp_read, "comp_write": comp_write,
              "ideal_l2_read_req": ideal_l2_read_req,
              "epilogue_read_bytes": epilogue_l2_read_bytes(ctx, comp_read, operand_bytes)}
    rows += timing_rows(ctx, timing, pmc) + apply_table(ctx, consts)
    rows += analytic_rows(ctx, comp_read, comp_write, ideal_l2_read_req) + raw_counter_rows(ctx)
    ctx.consts, ctx.metrics = consts, rows
    ctx.mrows = metric_index(rows)
    ctx.mv = {k: r["value"] for k, r in ctx.mrows.items()}


# ---------------------------------------------------------------------------------------------
# Validation (V1 and V12 are recorded by the passes that own them)
# ---------------------------------------------------------------------------------------------
def check_pinning(ctx):
    bus = int(ctx.agent["location_id"]) >> 8
    expected = ctx.metadata["gpu_pci_bus_number"]
    ok = bus == expected and ctx.env.get("ROCR_VISIBLE_DEVICES") == ctx.metadata["rocr_visible_devices"]
    ctx.check("V2", "dispatches ran on the pinned GPU", "PASS" if ok else "FAIL", f"pci bus {bus:#x}",
              f"pci bus {expected:#x}", "exact",
              f"ROCR_VISIBLE_DEVICES={ctx.env.get('ROCR_VISIBLE_DEVICES')} agent={ctx.agent['agent_row_id']}")


def check_waves(ctx):
    row = ctx.mrows["occupancy.sq_waves"]
    ctx.check_within("V3", "SQ_WAVES equals grid waves", row["value"], float(ctx.occ["grid_waves"]),
                     1e-6, detail=f"cv={fmt(row['cv'])} n={row['n_dispatches']}")


def check_mfma_work(ctx):
    mv, e = ctx.mv, ctx.mfma_expect
    detail = (f"static bf16={ctx.mfma['bf16']} prologue={ctx.mfma['prologue']} mnemonics={ctx.mfma['counts']}; "
              f"tiles={e['tiles']} waves_per_tile={e['waves_per_tile']}")
    ctx.check_within("V4", "512 * SQ_INSTS_VALU_MFMA_MOPS_BF16 equals 2*M*N*K (exact)",
                     mv["compute.mfma_flops_bf16"], float(PROBLEM_FLOPS), 1e-9)
    ctx.check_within("V4", "SQ_INSTS_VALU_MFMA_BF16 equals 2*M*N*K / flops per bf16 MFMA",
                     mv["compute.mfma_bf16_inst_count"], float(e["bf16"]), 1e-4, detail=detail)
    ctx.check_within("V4", "SQ_INSTS_MFMA equals bf16 MFMA + prologue MFMA * tiles * waves_per_tile",
                     mv["compute.mfma_inst_count"], float(e["total"]), 1e-4,
                     detail=f"expected total {e['total']} = {e['bf16']} + {e['prologue']}; {detail}")


def check_clock(ctx):
    mhz = ctx.mv["timing.implied_sclk_mhz"]
    ok = not math.isnan(mhz) and 800.0 <= mhz <= 2600.0
    ctx.check("V5", "implied sclk within [0.8, 2.6] GHz", "PASS" if ok else "FAIL", mhz,
              "800..2600 MHz", "range")
    probe = ctx.mv["timing.sclk_probe_mhz"]
    close = (not math.isnan(mhz) and not math.isnan(probe) and probe != 0
             and abs(mhz - probe) <= 0.10 * abs(probe))
    detail = f"probe samples={ctx.notes.get('probe_samples')}"
    if not close:
        detail += ("; the implied clock is an average over each dispatch while rocm-smi samples "
                   "instantaneous clocks; possible causes are clock variation within or across "
                   "dispatches AND GRBM_GUI_ACTIVE not spanning the full kernel-trace interval "
                   "(which lowers the implied sclk below the probe)")
    ctx.check("V5", "implied sclk within 10% of rocm-smi sclk under load", "PASS" if close else "WARN",
              fmt(mhz), fmt(probe), "rel 0.1", detail)


def check_grbm_coverage(ctx):
    row = ctx.mrows["timing.grbm_coverage_of_pmc_duration"]
    ctx.check("V19", "GRBM_GUI_ACTIVE coverage of the PMC-pass duration at the probe clock", "INFO",
              row["value"], "informational", "none",
              f"cv={fmt(row['cv'])} n={row['n_dispatches']}; CROSS-PASS probe clock; a value below 1 "
              "means GRBM_GUI_ACTIVE does not span the full kernel-trace interval at the probe clock "
              "or the clock was below the probe; algebraically implied_sclk_mhz / sclk_probe_mhz, derived "
              "from the same inputs as the implied-clock check (V5), so not independent evidence; "
              + GPU_CYCLES_CAVEAT)


def check_vmem_wavefronts(ctx):
    ta = pass_stats(ctx, "tcp_coal", "TA_TOTAL_WAVEFRONTS_sum").median
    vmem = pass_stats(ctx, "tcp_coal", "SQ_INSTS_VMEM").median
    other = pass_stats(ctx, "sq_mix2", "SQ_INSTS_VMEM").median
    ctx.check_within("V17", "TA_TOTAL_WAVEFRONTS equals SQ_INSTS_VMEM (same pass tcp_coal)", ta, vmem,
                     1e-4, detail=f"SQ_INSTS_VMEM from pass sq_mix2 (cross-pass) = {fmt(other)}")


def check_busy_cu_cycles(ctx):
    busy = pass_stats(ctx, "sq_occ", "SQ_BUSY_CU_CYCLES").median
    wave = pass_stats(ctx, "sq_occ", "SQ_WAVE_CYCLES").median
    ctx.check("V18", "SQ_BUSY_CU_CYCLES over SQ_WAVE_CYCLES (same pass sq_occ)", "INFO",
              fmt(safe_div(busy, wave)), "informational", "none",
              "a ratio of 1 means cu_busy_pct and achieved occupancy carry the same information here "
              "(one resident wave per SIMD) and are not independent evidence")


def csv_counter_values(directory):
    paths = glob.glob(os.path.join(directory, "*", "*_counter_collection.csv"))
    if len(paths) != 1:
        fatal(f"expected one counter_collection csv under {directory}")
    return {(r["Counter_Name"], int(r["Dispatch_Id"])): float(r["Counter_Value"])
            for r in read_csv_dicts(paths[0])}


def check_db_vs_csv(ctx):
    worst, count = 0.0, 0
    for group in list(COUNTER_GROUPS) + list(DERIVED_GROUPS):
        csv_vals = csv_counter_values(os.path.join(ctx.passdir, f"pmc_{group}"))
        for r in (r for r in ctx.raw if r["group"] == group):
            ref = csv_vals.get((r["counter"], r["dispatch_id"]), NAN)
            diff = abs(r["sum_value"] - ref) / max(abs(ref), 1.0) if not math.isnan(ref) else 1.0
            worst, count = max(worst, diff), count + 1
    ctx.check("V6", "rocpd database counters match the csv output", "PASS" if worst <= 1e-6 else "FAIL",
              f"max rel diff {worst:.3g} over {count} values", 0, "1e-6")
    check_trace_csv(ctx)


def check_trace_csv(ctx):
    paths = glob.glob(os.path.join(ctx.passdir, "timing_r0", "*", "*_kernel_trace.csv"))
    rows = [r for r in read_csv_dicts(paths[0]) if r["Kernel_Name"] == ctx.target]
    rows.sort(key=lambda r: int(r["Start_Timestamp"]))
    csv_us = [(int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1000.0 for r in rows]
    db_us = [d["duration_us"] for d in ctx.timing_dispatches[0]]
    worst = max(abs(a - b) for a, b in zip(csv_us, db_us)) if len(csv_us) == len(db_us) else NAN
    ok = not math.isnan(worst) and worst <= 1e-6
    ctx.check("V6", "rocpd database durations match the kernel-trace csv", "PASS" if ok else "FAIL",
              f"max abs diff {fmt(worst)} us over {len(csv_us)} dispatches", 0, "1e-6 us")


def check_traffic(ctx):
    mv = ctx.mv
    ctx.check_within("V7", "fabric read bytes vs FETCH_SIZE", mv["memory.fabric_read_bytes"],
                     mv["memory.fetch_size_drv"] * 1024.0, 0.05)
    ctx.check_within("V7", "fabric write bytes vs WRITE_SIZE", mv["memory.fabric_write_bytes"],
                     mv["memory.write_size_drv"] * 1024.0, 0.05)
    for direction in ("read", "write"):
        ratio = mv[f"memory.{direction}_over_compulsory"]
        ctx.check("V7", f"measured {direction} bytes at least 0.9x compulsory",
                  "PASS" if ratio >= 0.9 else "FAIL", ratio, ">= 0.9", "ratio",
                  f"compulsory={mv[f'memory.compulsory_{direction}_bytes']:.0f} B")


def check_l2_consistency(ctx):
    hit, miss, req = (pass_stats(ctx, "tcc_hit", n).median for n in ("TCC_HIT_sum", "TCC_MISS_sum", "TCC_REQ_sum"))
    ctx.check_within("V8", "TCC_HIT + TCC_MISS equals TCC_REQ (same pass)", hit + miss, req, 0.05)


def check_occupancy(ctx):
    occ, mv = ctx.occ, ctx.mv
    ctx.check("V9", "HIP API occupancy equals the theoretical model",
              "PASS" if occ["hip"] == occ["theoretical_wg"] else "FAIL", occ["hip"],
              occ["theoretical_wg"], "exact", f"limiter={occ['limiter']}")
    achieved = mv["occupancy.achieved_waves_per_cu"]
    ok = not math.isnan(achieved) and 0.0 < achieved <= occ["theoretical_waves"] * 1.02
    ctx.check("V9", "measured achieved waves/CU within (0, theoretical]", "PASS" if ok else "FAIL",
              achieved, f"<= {occ['theoretical_waves']}", "2%",
              "achieved = 4 * SQ_WAVE_CYCLES / (GRBM_GUI_ACTIVE * cu), waves/CU")
    api = mv["occupancy.mean_occupancy_per_cu"]
    ctx.check_within("V9", "api MeanOccupancyPerCU (waves/CU) vs measured achieved waves/CU", api,
                     achieved, 0.10, detail="both in waves/CU; measured from SQ_WAVE_CYCLES in a "
                     "separate pass, api value from SQ_ACCUM_PREV_HIRES / GRBM_GUI_ACTIVE / CU_NUM")
    ctx.check_within("V9", "api OccupancyPercent (as waves/CU) vs api MeanOccupancyPerCU",
                     mv["occupancy.occupancy_percent_waves_per_cu"], api, 0.05,
                     detail=f"OccupancyPercent / 100 * {HW_MAX_WAVES_PER_CU} waves/CU hardware max")
    ratio = mv["occupancy.mean_occupancy_per_cu_over_measured"]
    ok = not math.isnan(ratio) and 0.8 <= ratio <= 1.2
    ctx.check("V9", "MeanOccupancyPerCU unit is waves/CU (ratio to measured near 1)",
              "PASS" if ok else "FAIL", ratio, "0.8..1.2", "range",
              "a ratio near 0.25 would mean waves/SIMD")


def check_timing(ctx):
    ctx.check_within("V10", "timing-pass median within 10% of Tensile WinnerTimeUS",
                     ctx.mv["timing.duration_us"], ctx.variant["tensile_winner_time_us"], 0.10,
                     fail_status="WARN")


def check_cv(ctx):
    for label, cv in (("timing-pass duration", Stats(ctx.timing_us).cv),
                      ("GRBM_GUI_ACTIVE", pass_stats(ctx, "sq_occ", "GRBM_GUI_ACTIVE").cv),
                      ("SQ_WAVE_CYCLES", pass_stats(ctx, "sq_occ", "SQ_WAVE_CYCLES").cv)):
        ok = not math.isnan(cv) and cv < 0.05
        ctx.check("V11", f"cv of {label} below 5%", "PASS" if ok else "WARN", cv, "< 0.05", "abs")


def check_pmc_regime(ctx):
    mv = ctx.mv
    ctx.check("V14", "PMC-regime duration over timing-regime duration", "INFO",
              mv["timing.pmc_over_timing_ratio"], "informational", "none",
              f"timing {fmt(mv['timing.duration_us'])} us, pmc {fmt(mv['timing.duration_pmc_us'])} us; "
              "bandwidth and rate rows are reported in both regimes, the timing regime is the headline "
              "(measured wall time); " + GPU_CYCLES_CAVEAT)


def check_coverage(ctx):
    post = ctx.expected_total - ctx.warmups
    total = sum(len(data["counters"]) for data in ctx.passes.values())
    short = [f"{g}/{c}:n={Stats(v).n}" for g, data in ctx.passes.items()
             for c, v in data["counters"].items() if Stats(v).n != post]
    ctx.check("V13", "every counter of every pass has a value for each post-warmup dispatch",
              "FAIL" if short else "PASS", f"{total - len(short)} complete", total, "exact", " ".join(short))
    nonfinite = sorted(k for k, r in ctx.mrows.items()
                       if isinstance(r["value"], float) and not math.isfinite(r["value"]))
    undefined = {"occupancy.theoretical_wg_per_cu_lds"}
    if ctx.consts["epilogue_read_bytes"] == 0:
        undefined.add("coalescing.epilogue_bytes_per_l2_read_req")
    nonfinite = [m for m in nonfinite if m not in undefined]
    ctx.check("V13", "no derived metric is NaN or infinite", "FAIL" if nonfinite else "PASS",
              len(nonfinite), 0, "exact", " ".join(nonfinite))
    bad_rows = sorted({r["counter"] for r in ctx.raw if r["counter"] in PER_XCD_COUNTERS
                       and r["n_instance_rows"] != ctx.num_xcc})
    ctx.check("V13", "GRBM counters report one row per XCC", "FAIL" if bad_rows else "PASS",
              " ".join(bad_rows) or ctx.num_xcc, ctx.num_xcc, "exact")


def run_validation(ctx):
    for fn in (check_pinning, check_waves, check_mfma_work, check_clock, check_db_vs_csv,
               check_traffic, check_l2_consistency, check_occupancy, check_timing, check_cv,
               check_pmc_regime, check_vmem_wavefronts, check_busy_cu_cycles, check_grbm_coverage,
               check_coverage):
        fn(ctx)


# ---------------------------------------------------------------------------------------------
# Per-variant outputs
# ---------------------------------------------------------------------------------------------
METRIC_HEADER = ["variant", "kernel_name", "category", "metric", "value", "unit", "kind", "regime",
                 "aggregation", "n_dispatches", "cv", "formula", "reference"]
RAW_HEADER = ["variant", "pass_group", "counter", "dispatch_index", "dispatch_id", "post_warmup",
              "n_instance_rows", "sum_value", "agg_value", "duration_ns"]
VALIDATION_HEADER = ["variant", "check_id", "check", "status", "observed", "expected", "tolerance",
                     "detail"]
INVENTORY_HEADER = ["variant", "kernel_name", "is_target", "n_dispatches", "median_duration_us",
                    "grid_x", "grid_y", "grid_z", "wg_x", "wg_y", "wg_z", "lds_bytes",
                    "scratch_bytes", "arch_vgpr", "accum_vgpr", "sgpr", "agent_id"]
ATT_HEADER = ["variant", "kernel_name", "iteration_index", "expected_dispatch_id", "kind", "bytes",
              "path"]


def inventory_rows(ctx):
    by_name = {}
    for d in ctx.inventory:
        by_name.setdefault(d["kernel_name"], []).append(d)
    rows = []
    for name, ds in by_name.items():
        d = ds[0]
        rows.append([ctx.name, name, name == ctx.target, len(ds),
                     statistics.median(x["duration_us"] for x in ds), d["grid_x"], d["grid_y"],
                     d["grid_z"], d["wg_x"], d["wg_y"], d["wg_z"], d["lds_bytes"], d["scratch_bytes"],
                     d["arch_vgpr"], d["accum_vgpr"], d["sgpr"], d["agent_id"]])
    return rows


def write_variant_csvs(ctx):
    write_csv(os.path.join(ctx.vdir, "metrics_long.csv"), METRIC_HEADER,
              [[ctx.name, ctx.target] + [fmt(r[k]) for k in METRIC_HEADER[2:]] for r in ctx.metrics])
    write_csv(os.path.join(ctx.vdir, "raw_counters.csv"), RAW_HEADER,
              [[ctx.name, r["group"], r["counter"], r["dispatch_index"], r["dispatch_id"],
                r["post_warmup"], r["n_instance_rows"], fmt(r["sum_value"]), fmt(r["agg_value"]),
                r["duration_ns"]] for r in ctx.raw])
    write_csv(os.path.join(ctx.vdir, "validation.csv"), VALIDATION_HEADER,
              [[c[k] for k in VALIDATION_HEADER] for c in ctx.checks])
    write_csv(os.path.join(ctx.vdir, "kernel_inventory.csv"), INVENTORY_HEADER, inventory_rows(ctx))
    write_csv(os.path.join(ctx.vdir, "att_manifest.csv"), ATT_HEADER,
              [[e[k] for k in ATT_HEADER] for e in ctx.att_manifest])


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


REPORT_NOTES = [
    "coalescing_pct_lane_count_based is the rocprof-compute Coalescing metric (1600 / avg active lanes "
    "per VMEM wavefront): it equals 25% when all 64 lanes are active, higher values only mean some "
    "wavefronts had inactive lanes, and it is not a coalescing-efficiency metric on gfx950; use the "
    "coalescing category (requests per instruction, epilogue_bytes_per_l2_read_req) instead.",
    "ideal_operand_over_total_l2_read_req is the operand share of all L2 read requests; it equals "
    "operand-load efficiency only for the none variant, which has no epilogue reads.",
    "vl1d_hit_rate_pct mixes 128-B L2 requests with 64-B tag accesses, so at least 50% appears for "
    "streamed loads and differences between variants reflect request size, not locality.",
    "The " + GPU_CYCLES_CAVEAT + " (see V19 for GRBM_GUI_ACTIVE coverage).",
    "cu_busy_pct and achieved occupancy are not independent evidence while SQ_BUSY_CU_CYCLES == "
    "SQ_WAVE_CYCLES (check V18).",
    "pct_of_peak_hbm uses a peak HBM bandwidth; fabric bytes include MALL hits.",
]


def write_report(ctx):
    meta = ctx.metadata
    out = [f"# Epilogue profile: {ctx.name}", "",
           f"- kernel: `{ctx.target}`",
           f"- gpu: index {meta['gpu_index']} ({meta['gpu_unique_id']}, pci {meta['gpu_pci_bus']})",
           f"- dispatches: {ctx.expected_total} per pass, first {ctx.warmups} are warmups",
           f"- timing repeats: {ctx.args.timing_repeats}; counter groups: "
           f"{len(COUNTER_GROUPS)} hardware + {len(DERIVED_GROUPS)} derived", "",
           "## Validation", "",
           md_table(["id", "check", "status", "observed", "expected", "tolerance"],
                    [[c["check_id"], c["check"], c["status"], c["observed"], c["expected"],
                      c["tolerance"]] for c in ctx.checks]), ""]
    out += ["## Reading notes", ""] + [f"- {n}" for n in REPORT_NOTES] + [""]
    by_cat = {}
    for r in ctx.metrics:
        if r["aggregation"] in ("median", "exact") and r["kind"] != "counter":
            by_cat.setdefault(r["category"], []).append(r)
    for cat, rows in by_cat.items():
        out += [f"## {cat}", "", md_table(["metric", "value", "unit", "kind", "regime", "n", "cv"],
                [[r["metric"], fmt(r["value"]), r["unit"], r["kind"], r["regime"], r["n_dispatches"],
                  fmt(r["cv"])]
                 for r in rows]), ""]
    with open(os.path.join(ctx.vdir, "report.md"), "w") as f:
        f.write("\n".join(out) + "\n")


# ---------------------------------------------------------------------------------------------
# run (P0-P3 for one variant)
# ---------------------------------------------------------------------------------------------
def run_clock_probe(ctx):
    probe_ini = os.path.join(ctx.vdir, "probe.ini")
    write_derived_ini(ctx.variant, probe_ini, os.path.join(ctx.vdir, "probe_results.csv"),
                      {"num-enqueues-per-sync": str(PROBE_ENQUEUES)})
    probe_dir = os.path.join(ctx.vdir, "probe")
    if os.path.exists(probe_dir):
        shutil.rmtree(probe_dir)
    cmd = [ROCPROFV3, "--pmc", "GRBM_COUNT", "GRBM_GUI_ACTIVE", "--kernel-include-regex",
           f"^{ctx.target}$", "-f", "csv", "-d", probe_dir, "--", ctx.client, "--config-file",
           probe_ini]
    mhz, n = probe_sclk_mhz(ctx.metadata["gpu_index"], cmd, ctx.env,
                            os.path.join(ctx.logdir, "clock_probe.log"), ctx.vdir)
    ctx.notes["probe_samples"] = n
    row = metric_row("timing", "sclk_probe_mhz", mhz, "MHz", "probe", "median", n, NAN,
                     "median rocm-smi sclk while the profiled enqueue loop of the target runs",
                     "rocm-smi --showclocks", "pmc")
    ctx.metrics.append(row)
    ctx.mrows["timing.sclk_probe_mhz"] = row
    ctx.mv["timing.sclk_probe_mhz"] = mhz
    add_grbm_coverage_row(ctx, mhz)


def add_grbm_coverage_row(ctx, probe_mhz):
    """Fraction of the PMC-pass duration spanned by GRBM_GUI_ACTIVE at the probe clock."""
    n_post = ctx.expected_total - ctx.warmups
    samples = [Sample(ctx.passes, ("sq_occ",), i, ctx.consts) for i in range(n_post)]
    stats = Stats([safe_div(s(GU) / probe_mhz, s("@duration_us")) for s in samples])
    formula = ("(per-XCD GRBM_GUI_ACTIVE / sclk_probe_mhz) / PMC-pass duration, per-dispatch median "
               "(pass sq_occ); CROSS-PASS: the probe clock comes from the separate clock-probe run; "
               "values below 1 mean GRBM_GUI_ACTIVE covers less than the kernel-trace interval at the "
               "probe clock, or the clock was lower than the probe; algebraically implied_sclk_mhz / "
               "sclk_probe_mhz, derived from the same inputs as the implied-clock check and not "
               "independent evidence")
    row = metric_row("timing", "grbm_coverage_of_pmc_duration", stats.median, "ratio", "derived",
                     "median", stats.n, stats.cv, formula, "analytic", "pmc")
    ctx.metrics.append(row)
    ctx.mrows["timing.grbm_coverage_of_pmc_duration"] = row
    ctx.mv["timing.grbm_coverage_of_pmc_duration"] = row["value"]


def cmd_run(args):
    variant = read_json(args.variants_json)[args.variant]
    ctx = Ctx(args, variant)
    os.makedirs(ctx.logdir, exist_ok=True)
    os.makedirs(ctx.passdir, exist_ok=True)
    ctx.ini = write_derived_ini(variant, ctx.ini_path, os.path.join(ctx.vdir, "client_results.csv"))
    ctx.warmups, ctx.expected_total = expected_target_dispatches(ctx.ini)
    check_counter_availability(args.avail)
    log(f"{ctx.name}: kernel {ctx.target}")
    ctx.mfma = disassemble_mfma(variant)
    run_timing_passes(ctx)
    run_counter_passes(ctx, args.avail)
    build_metrics(ctx)
    run_clock_probe(ctx)
    run_att_pass(ctx)
    run_validation(ctx)
    write_variant_csvs(ctx)
    write_report(ctx)
    counts = {s: sum(1 for c in ctx.checks if c["status"] == s) for s in ("PASS", "INFO", "WARN", "FAIL")}
    log(f"{ctx.name}: validation {counts}")


# ---------------------------------------------------------------------------------------------
# merge (S5)
# ---------------------------------------------------------------------------------------------
def read_variant_csvs(root, filename):
    header, rows = None, []
    for variant in VARIANT_ORDER:
        path = os.path.join(root, "variants", variant, filename)
        with open(path, newline="") as f:
            reader = csv.reader(f)
            h = next(reader)
            if header is None:
                header = h
            elif h != header:
                fatal(f"header mismatch in {path}")
            rows += list(reader)
    return header, rows


def concat_variant_csv(root, filename, out_path, extra_rows=()):
    header, rows = read_variant_csvs(root, filename)
    rows += list(extra_rows)
    write_csv(out_path, header, rows)
    return header, rows


def wide_column(row):
    return f"{row['category']}.{row['metric']}[{row['unit']}]"


def build_wide(long_rows):
    wide = {v: {} for v in VARIANT_ORDER}
    columns = []
    for r in long_rows:
        if r["aggregation"] not in ("median", "exact"):
            continue
        col = wide_column(r)
        if col not in columns:
            columns.append(col)
        wide[r["variant"]][col] = r["value"]
    return columns, wide


def to_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return NAN


def difference_row(category, metric, a, b, unit, kind, regime, formula, reference):
    """a - b as a metric dict; cv from the cvs of a and b treated as independent."""
    va, vb = to_float(a["value"]), to_float(b["value"])
    diff = va - vb
    sigma = math.hypot(to_float(a["cv"]) * abs(va), to_float(b["cv"]) * abs(vb))
    ns = [int(x["n_dispatches"]) for x in (a, b) if int(x["n_dispatches"]) > 0]
    return {"category": category, "metric": metric, "value": diff, "unit": unit, "kind": kind,
            "regime": regime, "aggregation": "median", "n_dispatches": min(ns) if ns else 0,
            "cv": safe_div(sigma, abs(diff)), "formula": formula, "reference": reference}


def metric_lookup(dicts):
    return {(r["variant"], f"{r['category']}.{r['metric']}"): r for r in dicts
            if r["aggregation"] in ("median", "exact")}


def variant_excess_rows(idx, variant, direction):
    measured = f"memory.fabric_{direction}_bytes"
    compulsory = f"memory.compulsory_{direction}_bytes"
    excess = difference_row(
        "memory", f"excess_{direction}_bytes_vs_none", idx[(variant, measured)], idx[("none", measured)],
        "B", "derived", "pmc",
        f"measured fabric {direction} bytes of this variant - those of the none variant "
        "(each in its own tcc pass)", REF_ANALYTIC)
    expected = difference_row(
        "memory", f"expected_extra_{direction}_bytes_vs_none", idx[(variant, compulsory)],
        idx[("none", compulsory)], "B", "analytic", "analytic",
        f"compulsory {direction} bytes of this variant - those of the none variant", REF_ANALYTIC)
    residual = difference_row(
        "memory", f"excess_minus_expected_{direction}_bytes_vs_none", excess, expected, "B", "derived",
        "pmc", f"excess_{direction}_bytes_vs_none - expected_extra_{direction}_bytes_vs_none; "
        "nonzero means traffic beyond the compulsory model", REF_ANALYTIC)
    return [excess, expected, residual]


READ_REQ_HYPOTHESIS = ("hypothesis, unproven: one 128-B EA read per extra 64-B TCP read request; "
                       "CROSS-PASS: TCC_EA0_RDREQ_128B from pass tcc_rd, TCP_TCC_READ_REQ from pass tcp_coal")


def read_request_rows(idx, variant):
    ea_key, tcp_key = "raw_tcc_rd.TCC_EA0_RDREQ_128B_sum", "raw_tcp_coal.TCP_TCC_READ_REQ_sum"
    ea = difference_row(
        "coalescing", "excess_ea_rdreq_128b_vs_none", idx[(variant, ea_key)], idx[("none", ea_key)],
        "requests", "derived", "pmc", f"TCC_EA0_RDREQ_128B of this variant - none's; {READ_REQ_HYPOTHESIS}",
        REF_COAL)
    tcp = difference_row(
        "coalescing", "excess_tcp_tcc_read_req_vs_none", idx[(variant, tcp_key)], idx[("none", tcp_key)],
        "requests", "derived", "pmc", f"TCP_TCC_READ_REQ of this variant - none's; {READ_REQ_HYPOTHESIS}",
        REF_COAL)
    rows = [ea, tcp]
    if variant != "none":
        ratio = safe_div(ea["value"], tcp["value"])
        cv = math.hypot(to_float(ea["cv"]), to_float(tcp["cv"]))
        rows.append({"category": "coalescing", "metric": "ea_128b_reads_per_extra_tcp_read_req",
                     "value": ratio, "unit": "ratio", "kind": "derived", "regime": "pmc",
                     "aggregation": "median", "n_dispatches": min(ea["n_dispatches"], tcp["n_dispatches"]),
                     "cv": cv, "formula": "excess_ea_rdreq_128b_vs_none / excess_tcp_tcc_read_req_vs_none; "
                     + READ_REQ_HYPOTHESIS, "reference": REF_COAL})
    return rows


def cross_variant_rows(dicts):
    idx = metric_lookup(dicts)
    kernels = {r["variant"]: r["kernel_name"] for r in dicts}
    rows = []
    for variant in VARIANT_ORDER:
        derived = read_request_rows(idx, variant)
        for direction in ("read", "write"):
            derived += variant_excess_rows(idx, variant, direction)
        for d in derived:
            rows.append([variant, kernels[variant]] + [fmt(d[k]) for k in METRIC_HEADER[2:]])
    return rows


def regime_consistency_check(dicts):
    idx = metric_lookup(dicts)
    ratios = {v: to_float(idx[(v, "timing.pmc_over_timing_ratio")]["value"]) for v in VARIANT_ORDER}
    mean = statistics.fmean(ratios.values())
    spread = 100.0 * (max(ratios.values()) - min(ratios.values())) / mean
    verdict = "consistent" if spread < 2.0 else "NOT consistent"
    observed = " ".join(f"{v}={ratios[v]:.4f}" for v in VARIANT_ORDER)
    return ["all", "V14", "PMC/timing duration ratio across variants", "INFO", observed,
            "informational", "spread < 2% would be consistent",
            f"{verdict}: spread {spread:.1f}% of the mean; rates compared across variants in the PMC "
            "regime inherit this perturbation, the timing regime is the headline"]


def clock_check(dicts):
    idx = metric_lookup(dicts)
    mhz = {v: to_float(idx[(v, "timing.implied_sclk_mhz")]["value"]) for v in VARIANT_ORDER}
    cycles = {v: to_float(idx[(v, "timing.gpu_cycles")]["value"]) for v in VARIANT_ORDER}
    mean = statistics.fmean(mhz.values())
    spread = 100.0 * (max(mhz.values()) - min(mhz.values())) / mean
    probe = {v: to_float(idx[(v, "timing.sclk_probe_mhz")]["value"]) for v in VARIANT_ORDER}
    observed = " ".join(f"{v}={mhz[v]:.0f}MHz" for v in VARIANT_ORDER)
    cycle_text = " ".join(f"{v}={cycles[v]:.0f}" for v in VARIANT_ORDER)
    probe_text = " ".join(f"{v} {mhz[v]:.0f} vs {probe[v]:.0f}" for v in VARIANT_ORDER)
    return ["all", "V16", "implied sclk per variant (PMC regime)", "INFO", observed, "informational",
            "none", f"spread {spread:.1f}% of the mean; gpu_cycles per XCD (timing.gpu_cycles) {cycle_text}; "
            f"implied vs rocm-smi probe sclk in MHz: {probe_text}; possible causes of the spread are "
            "clock variation AND GRBM_GUI_ACTIVE not spanning the full kernel-trace interval (implied "
            "sclk below the probe for every variant), so the spread may not be a real clock spread; "
            + GPU_CYCLES_CAVEAT]


def write_vs_baseline(path, columns, wide):
    rows, zero_base, unusable = [], 0, 0
    for variant in ("rms", "beta"):
        for col in columns:
            value, base = wide[variant].get(col, ""), wide["none"].get(col, "")
            a, b = to_float(value), to_float(base)
            delta, delta_pct, note = "", "", ""
            if not (math.isfinite(a) and math.isfinite(b)):
                note, unusable = "value or baseline is empty, non-finite or non-numeric", unusable + 1
            else:
                delta = fmt(a - b)
                if b == 0:
                    note, zero_base = "baseline is zero, delta_pct left empty", zero_base + 1
                else:
                    delta_pct = fmt(100.0 * (a - b) / b)
            rows.append([variant, "none", col, value, base, delta, delta_pct, note])
    write_csv(path, ["variant", "baseline", "column", "value", "baseline_value", "delta",
                     "delta_pct", "note"], rows)
    return zero_base, unusable


def baseline_check(zero_base, unusable):
    return ["all", "V15", "metrics_vs_baseline empty cells", "INFO",
            f"{zero_base} zero-baseline, {unusable} empty or non-numeric", "informational", "none",
            "empty delta_pct cells replace nan/inf; the note column says why"]


def cmd_merge(args):
    root = args.root
    long_header, long_rows = read_variant_csvs(root, "metrics_long.csv")
    dicts = [dict(zip(long_header, r)) for r in long_rows]
    cross_rows = cross_variant_rows(dicts)
    concat_variant_csv(root, "metrics_long.csv", os.path.join(root, "metrics_long.csv"), cross_rows)
    long_dicts = [dict(zip(long_header, r)) for r in long_rows + cross_rows]
    columns, wide = build_wide(long_dicts)
    write_csv(os.path.join(root, "metrics_wide.csv"), ["variant"] + columns,
              [[v] + [wide[v].get(c, "") for c in columns] for v in VARIANT_ORDER])
    zero_base, unusable = write_vs_baseline(os.path.join(root, "metrics_vs_baseline.csv"), columns, wide)
    extra = [regime_consistency_check(dicts), clock_check(dicts), baseline_check(zero_base, unusable)]
    _, val_rows = concat_variant_csv(root, "validation.csv", os.path.join(root, "validation.csv"), extra)
    concat_variant_csv(root, "kernel_inventory.csv", os.path.join(root, "kernel_inventory.csv"))
    concat_variant_csv(root, "att_manifest.csv", os.path.join(root, "att_manifest.csv"))
    statuses = [r[3] for r in val_rows]
    counts = {s: statuses.count(s) for s in ("PASS", "INFO", "WARN", "FAIL")}
    for r in val_rows:
        if r[3] != "PASS":
            print(f"{r[3]}: [{r[0]}] {r[1]} {r[2]} observed={r[4]} expected={r[5]} {r[7][:200]}")
    print("validation: " + " ".join(f"{s}={n}" for s, n in counts.items()))
    return 1 if counts["FAIL"] else 0


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("preflight", help="pin the GPU, check idleness, write run_metadata.json")
    s.add_argument("--gpu", type=int, required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--tensile-dir", required=True)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_preflight)

    s = sub.add_parser("list-avail", help="save rocprofv3 -L and verify requested counters")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_list_avail)

    s = sub.add_parser("discover", help="classify built variants from the library yaml")
    s.add_argument("--tensile-out", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_discover)

    s = sub.add_parser("run", help="timing, counter and ATT passes for one variant")
    s.add_argument("--variant", required=True, choices=VARIANT_ORDER)
    s.add_argument("--variants-json", required=True)
    s.add_argument("--metadata", required=True)
    s.add_argument("--avail", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--gpu-selector", required=True)
    s.add_argument("--client", default=TENSILE_CLIENT_DEFAULT)
    s.add_argument("--timing-repeats", type=int, default=3)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("merge", help="merge per-variant outputs and report validation")
    s.add_argument("--root", required=True)
    s.set_defaults(func=cmd_merge)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except FatalError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

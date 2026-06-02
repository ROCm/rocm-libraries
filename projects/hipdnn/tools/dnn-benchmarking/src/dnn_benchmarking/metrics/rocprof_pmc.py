# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""rocprofv3 PMC counter collection.

Re-runs the workload under ``rocprofv3 --pmc <counters>`` and parses
the resulting rocpd SQLite database into per-engine counter aggregates.

Counter sets are hardcoded per-arch and are intentionally small enough
to fit a single-pass replay on every supported arch. The ``"all"`` set
unions every group and is gated by ``MetricsConfig.pmc_allow_multipass``
because rocprofv3's multi-pass replay has been observed to hang for
minutes on sub-second workloads.

We do not pre-validate counter availability because the
``rocprofv3-avail counters`` subcommand is missing in rocprofv3 1.2.2.
Instead we let rocprofv3 fail at run time and surface its stderr tail
under ``extra_metrics["pmc"]["error_tail"]``. This is also more robust
to per-build counter renames than a static validity check.

All three gfx90a sets are verified against
``/opt/rocm/share/rocprofiler-sdk/basic_counters.xml`` and
``derived_counters.xml`` plus a single-pass ``rocprofv3 --pmc`` run on
a gfx90a host. The fallback set is conservative and known-good across
every supported arch.
"""

import sqlite3
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from ._artifact_paths import (
    DEFAULT_PROFILING_TIMEOUT_S,
    find_first,
    flatten_hostname_dir,
)
from ._diagnostic import warn_once
from ._tool_resolver import resolve_rocm_tool
from .arch import detect_arch

PMC_SETS: Dict[str, Dict[str, List[str]]] = {
    "gfx942": {
        "basic": [
            "GRBM_GUI_ACTIVE",
            "SQ_WAVES",
            "SQ_INSTS_VALU",
            "SQ_BUSY_CYCLES",
        ],
        "memory": [
            "TCC_HIT_sum",
            "TCC_MISS_sum",
            "TCP_TCC_READ_REQ_sum",
            "TCC_EA_RDREQ_sum",
        ],
        "flops": [
            "SQ_INSTS_VALU_MFMA_F16",
            "SQ_INSTS_VALU_MFMA_BF16",
            "SQ_INSTS_VALU_MFMA_F32",
        ],
    },
    "gfx90a": {
        "basic": [
            "GRBM_GUI_ACTIVE",
            "SQ_WAVES",
            "SQ_INSTS_VALU",
            "SQ_BUSY_CYCLES",
        ],
        "memory": [
            "TCC_HIT_sum",
            "TCC_MISS_sum",
            "TCP_TCC_READ_REQ_sum",
            "TCC_EA_RDREQ_sum",
        ],
        "flops": [
            "SQ_INSTS_VALU_MFMA_F16",
            "SQ_INSTS_VALU_MFMA_BF16",
            "SQ_INSTS_VALU_MFMA_F32",
        ],
    },
    "fallback": {
        "basic": ["GRBM_GUI_ACTIVE", "SQ_WAVES"],
    },
}


def _resolve_counter_groups(arch: str, pmc_set: str) -> List[List[str]]:
    """Return one counter group per intended rocprofv3 pass.

    rocprofv3 expresses multipass via repeated ``--pmc`` flags — one
    flag per group, each group collected in a separate pass. The
    historical flat-list emit ``--pmc <every counter from every group>``
    was a single-pass request regardless of group count; on arches where
    the union exceeded the hardware single-pass counter budget,
    rocprofv3 would either fail or silently drop counters.

    Named sets (``basic``/``memory``/``flops``) always return exactly
    one group. ``all`` returns one group per source group from the arch
    table, preserving pass boundaries. The fallback arch table only
    defines ``basic``, so ``all`` on an unknown arch returns that single
    group.

    Empty outer list signals "nothing to collect" — the caller should
    skip the run.
    """
    arch_table = PMC_SETS.get(arch) or PMC_SETS["fallback"]
    if pmc_set == "all":
        return [list(group) for group in arch_table.values() if group]
    group = arch_table.get(pmc_set)
    if not group:
        return []
    return [list(group)]


def _resolve_counter_list(arch: str, pmc_set: str) -> List[str]:
    """Flat, dedup-preserved view of every counter across every group.

    Kept for diagnostic logging and ``counters_requested`` in the JSON
    payload — the argv builder consumes groups directly via
    ``_resolve_counter_groups``. Order matches first-occurrence across
    groups so the JSON view is stable across rocprofv3 invocations.
    """
    seen: Dict[str, None] = {}
    for group in _resolve_counter_groups(arch, pmc_set):
        for name in group:
            seen.setdefault(name, None)
    return list(seen)


def _build_argv(
    counter_groups: List[List[str]],
    out_dir: Path,
    inner_argv: List[str],
    rocprofv3_binary: str,
) -> List[str]:
    """Emit one ``--pmc`` per group.

    Single-group invocations (named sets, fallback arches) collapse to
    one ``--pmc <counters...>`` — identical wire format to the historical
    single-flag emit. Multi-group invocations (``--pmc all`` on a known
    arch) emit one flag per group, which rocprofv3 reads as a multipass
    request — replay budget is per-pass rather than per-aggregate-union.

    `-o results` drops the `<pid>_` prefix from rocprofv3's default
    `<hostname>/<pid>_results.<ext>` filename. The hostname segment is
    still added by rocprofv3 itself and is hoisted out post-run by
    ``profiling_orchestrator.flatten_hostname_dir``.
    """
    argv: List[str] = [rocprofv3_binary]
    for group in counter_groups:
        argv.append("--pmc")
        argv.extend(group)
    argv += [
        "-d",
        str(out_dir),
        "-o",
        "results",
        "--",
        *inner_argv,
    ]
    return argv


def _parse_rocpd_db(db_path: Path) -> Dict[str, Any]:
    """Walk the rocpd schema and aggregate per-kernel PMC values.

    The rocpd schema names tables with a uuid suffix that varies per
    run (e.g. ``rocpd_pmc_event_<uuid>``). We discover them via
    ``sqlite_master`` rather than hardcoding the suffix.

    Join keys (rocprofv3 1.2.2 schema):
      * ``pmc_event.event_id`` references ``kernel_dispatch.dispatch_id``
        (not ``kernel_dispatch.id`` — those happen to coincide for
        single-stream single-process runs but diverge in general).
      * ``kernel_dispatch.kernel_id`` references
        ``info_kernel_symbol.id``, which carries the demangled
        ``kernel_name`` column. ``kernel_dispatch`` itself has no
        ``kernel_name``.
    """
    # Open by str path rather than a `file:<path>?mode=ro` URI so a
    # `?`, `#`, or `%` anywhere in the user-controlled profiling-output
    # directory doesn't get parsed as a URI query string. PRAGMA
    # query_only enforces the read-only intent at the connection level.
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA query_only = ON")
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        pmc_event_table = next(
            (t for t in tables if t.startswith("rocpd_pmc_event")), None
        )
        kernel_table = next(
            (t for t in tables if t.startswith("rocpd_kernel_dispatch")), None
        )
        symbol_table = next(
            (t for t in tables if t.startswith("rocpd_info_kernel_symbol")), None
        )
        info_pmc_table = next(
            (t for t in tables if t.startswith("rocpd_info_pmc")), None
        )
        if pmc_event_table is None or kernel_table is None:
            return {
                "warnings": [
                    "rocpd db missing pmc_event or kernel_dispatch table; "
                    f"present tables: {sorted(tables)}"
                ]
            }
        if symbol_table is None:
            return {
                "warnings": [
                    "rocpd db missing info_kernel_symbol table; "
                    f"present tables: {sorted(tables)}"
                ]
            }

        # Map pmc_id -> counter name when the info table is present;
        # otherwise fall back to numeric ids in the output.
        id_to_name: Dict[int, str] = {}
        if info_pmc_table is not None:
            for row in conn.execute(f"SELECT id, name FROM {info_pmc_table}"):
                id_to_name[int(row[0])] = str(row[1])

        per_kernel_totals: Dict[str, Dict[str, float]] = defaultdict(dict)
        per_counter_running: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"sum": 0.0, "n": 0.0}
        )
        rows = conn.execute(
            f"""
            SELECT sym.kernel_name, p.pmc_id,
                   AVG(p.value), SUM(p.value), COUNT(p.value)
            FROM {pmc_event_table} p
            JOIN {kernel_table} k  ON p.event_id  = k.dispatch_id
            JOIN {symbol_table}  sym ON k.kernel_id = sym.id
            GROUP BY sym.kernel_name, p.pmc_id
            """
        )
        for kname, pmc_id, mean_v, sum_v, count_v in rows:
            counter_name = id_to_name.get(int(pmc_id), f"pmc_id_{pmc_id}")
            per_kernel_totals[str(kname)][counter_name] = float(mean_v)
            running = per_counter_running[counter_name]
            running["sum"] += float(sum_v)
            running["n"] += float(count_v)

        per_counter: Dict[str, Dict[str, float]] = {}
        for counter, agg in per_counter_running.items():
            per_counter[counter] = {
                "sum": agg["sum"],
                "mean_per_kernel": agg["sum"] / agg["n"] if agg["n"] else 0.0,
            }
        return {"counters": per_counter, "per_kernel": dict(per_kernel_totals)}
    finally:
        conn.close()


def run(
    inner_argv: List[str],
    out_dir: Path,
    pmc_set: str,
    timeout_s: int = DEFAULT_PROFILING_TIMEOUT_S,
) -> Dict[str, Any]:
    """Run rocprofv3 PMC collection and return the extra_metrics slice.

    Args:
        inner_argv: The argv to invoke under rocprofv3 (the
            ``--internal-profiling-run`` sub-mode of dnn-benchmarking).
        out_dir: Per-source output directory; the rocpd db lands inside
            ``<out_dir>/<hostname>/``.
        pmc_set: One of ``basic``, ``memory``, ``flops``, ``all``.
        timeout_s: Wall-clock budget for the rocprofv3 subprocess.
            ``0`` disables. Sourced from ``MetricsConfig.profiling_timeout_s``
            (CLI ``--profiling-timeout``).

    Never raises — failures are reported via the returned dict's
    ``error_tail`` / ``warnings`` keys plus a ``warn_once`` to stderr.
    """
    arch = detect_arch()
    counter_groups = _resolve_counter_groups(arch, pmc_set)
    counters = _resolve_counter_list(arch, pmc_set)
    if not counters:
        warn_once("rocprof_pmc", f"no counters defined for arch={arch} set={pmc_set}")
        return {
            "pmc": {
                "set": pmc_set,
                "arch": arch,
                "skipped": "no counters defined",
            }
        }

    # `--pmc all` on an arch not in PMC_SETS silently narrows to the
    # fallback's 2-counter `basic` group — the user paid the
    # --pmc-allow-multipass opt-in cost expecting a unioned counter set
    # and would otherwise see no diagnostic explaining the small result.
    arch_narrowed = pmc_set == "all" and arch not in PMC_SETS
    if arch_narrowed:
        warn_once(
            "rocprof_pmc",
            f"arch '{arch}' has no PMC table; --pmc all narrowed to "
            f"fallback basic ({len(counters)} counters)",
        )

    rocprofv3_binary = resolve_rocm_tool("rocprofv3")
    if rocprofv3_binary is None:
        warn_once("rocprof_pmc", "rocprofv3 binary not found; skipping PMC pass")
        return {
            "pmc": {
                "set": pmc_set,
                "arch": arch,
                "skipped": "rocprofv3 binary not found",
            }
        }

    # rocprofv3 nests its output under <out_dir>/<hostname>/. We hoist
    # those files up to <out_dir>/ post-run so the artifact path the
    # user sees in the JSON / console doesn't carry the hostname
    # segment for single-host runs.
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = _build_argv(counter_groups, out_dir, inner_argv, rocprofv3_binary)

    subprocess_timeout = timeout_s or None
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=subprocess_timeout,
        )
    except subprocess.TimeoutExpired:
        warn_once(
            "rocprof_pmc",
            f"rocprofv3 PMC pass timed out after {subprocess_timeout}s — "
            "likely a wedged kernel or multi-pass replay; raise "
            "--profiling-timeout to extend",
        )
        return {
            "pmc": {
                "set": pmc_set,
                "arch": arch,
                "skipped": f"rocprofv3 PMC pass timed out after {subprocess_timeout}s",
            }
        }
    except (OSError, subprocess.SubprocessError) as e:
        warn_once("rocprof_pmc", f"rocprofv3 invocation failed: {e}")
        return {
            "pmc": {
                "set": pmc_set,
                "arch": arch,
                "skipped": f"rocprofv3 invocation failed: {e}",
            }
        }

    # Hoist results.db out of <out_dir>/<hostname>/ to <out_dir>/ so
    # the artifact path the user sees in the JSON doesn't carry the
    # hostname segment. Runs on the error path too so any partial
    # files rocprofv3 wrote before exiting are still reachable on
    # disk for manual inspection, even though we don't parse them.
    flatten_hostname_dir(out_dir)

    result: Dict[str, Any] = {
        "set": pmc_set,
        "arch": arch,
        "counters_requested": counters,
    }
    if arch_narrowed:
        result["arch_narrowed_to_fallback"] = True
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-40:])
        warn_once(
            "rocprof_pmc",
            f"rocprofv3 exited {proc.returncode}; see extra_metrics['pmc']['error_tail']",
        )
        result["error_tail"] = tail
        result["returncode"] = proc.returncode
        return {"pmc": result}

    db_path = find_first(out_dir, "*.db")
    if db_path is None:
        warn_once("rocprof_pmc", "rocprofv3 produced no .db file")
        result["warnings"] = ["no .db file found in output directory"]
        return {"pmc": result}

    result["db_path"] = str(db_path)
    try:
        parsed = _parse_rocpd_db(db_path)
    except sqlite3.Error as e:
        warn_once("rocprof_pmc", f"rocpd db parse failed: {e}")
        result["warnings"] = [f"rocpd db parse failed: {e}"]
        return {"pmc": result}

    result.update(parsed)
    return {"pmc": result}

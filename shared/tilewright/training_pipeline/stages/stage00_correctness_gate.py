#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage00_correctness_gate -- which kernels of the pool compute wrong
results (lib/correctness_gate.py).

The orchestrator runs it before the rounds of an invocation when
`correctness.enabled`. It checks every kernel of the config's pool (the
library's table and the bank; a blocklist never shrinks the pool it checks)
that no list of `correctness.lists` names, the hazard canary did not
quarantine (--quarantine) and the verdict cache (--cache-dir) holds no
verdict of, with the exact harness on the GPUs of --devices under
their leases (lib/gpu_lease.py). With nothing to check it builds nothing and
takes no GPU. A signal stops the harness processes (lib/procs.py) after the
verdicts no pass left to run could change are cached.

Outputs in --out-dir:
  blocklist.json     the kernels the run leaves out: those the lists name
                     and those the gate saw fail, now or before (the cache),
                     by name, kernel id and signature (lib/correctness.py)
  verdicts.json      every kernel of the pool: its verdict (pass, fail,
                     crash, untested, listed, unchecked), where it comes from
                     (gate, cache, list) and what the harness saw
  gate_summary.json  counts, protocol, device, harness, tasks, GPU seconds
  runs/gpu<N>.jsonl  the harness's records, one per kernel and repetition
  harness_build.log  the build of the harness, when it was built

The exit status is 0 when the gate ran (a failing kernel is a finding), 1
for a config, library or harness problem, and 2 when a GPU lease stayed
busy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import yaml  # noqa: E402

from lib import correctness as cx  # noqa: E402
from lib import correctness_gate as cg  # noqa: E402
from lib import gpu_lease  # noqa: E402
from lib import kernel_identity as ki  # noqa: E402
from lib import procs  # noqa: E402
from lib import ui  # noqa: E402


def _devices(text: Optional[str], cfg: dict) -> List[int]:
    if text:
        return [int(d) for d in text.split(",") if d.strip()]
    sec = cfg.get("correctness") or {}
    return [int(d) for d in sec.get("devices") or cfg["runtime"]["devices"]]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="stage00: check the kernels of the pool with the exact harness"
    )
    ap.add_argument("--config-yaml", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--dat-dir",
        type=Path,
        required=True,
        help="Tensile library dir with the logic files (<build>/Tensile/library/<arch>)",
    )
    ap.add_argument("--library-stem", default=None, help="default: the config's")
    ap.add_argument(
        "--hardware-device",
        type=int,
        default=0,
        help="device whose bank and kind the verdicts are of",
    )
    ap.add_argument(
        "--devices",
        default=None,
        help="comma list of devices to run on (default: correctness.devices, "
        "else runtime.devices)",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="verdict cache (default: correctness.cache_dir, else <out-dir>/cache)",
    )
    ap.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="the hipBLASLt build the harness links (the one --dat-dir belongs to)",
    )
    ap.add_argument("--lease-wait-s", type=float, default=0.0)
    ap.add_argument(
        "--quarantine",
        type=Path,
        default=None,
        help="the hazard canary's quarantine (stage00a): kernels the gate blocks "
        "as listed and never runs",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be checked; check nothing",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config_yaml.read_text()) or {}
    stem = args.library_stem or str((cfg.get("hipblaslt") or {}).get("library_stem"))
    sec = cfg.get("correctness") or {}
    cache_dir = args.cache_dir or (
        Path(sec["cache_dir"]) if sec.get("cache_dir") else args.out_dir / "cache"
    )
    say = (lambda _m: None) if args.quiet else (lambda m: ui.info("gate", m))
    if not args.quiet:
        ui.banner("stage00  correctness gate", ui.C.HEAD)
        ui.info("cfg", f"library stem : {stem}")
        ui.info("cfg", f"library dir  : {args.dat_dir}")
        ui.info("cfg", f"cache        : {cache_dir}")
    children = procs.ChildGroups()
    try:
        with procs.stop_children_on_exit(children):
            summary = cg.run_gate(
                cg.Spec(
                    cfg=cfg,
                    dat_dir=args.dat_dir,
                    library_stem=stem,
                    out_dir=args.out_dir,
                    cache_dir=cache_dir,
                    build_dir=args.build_dir,
                    hardware_device=args.hardware_device,
                    devices=_devices(args.devices, cfg),
                    lease_wait_s=args.lease_wait_s,
                    dry_run=args.dry_run,
                    quarantine=args.quarantine,
                ),
                say=say,
                warn=lambda m: ui.warn("gate", m),
                children=children,
            )
    except gpu_lease.LeaseBusy as e:
        ui.err("lease", str(e))
        return 2
    except (
        cg.GateError,
        gpu_lease.LeaseError,
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
    ) as e:
        ui.err("gate", str(e))
        return 1
    if not args.quiet:
        v = summary["verdicts"]
        ui.ok(
            "gate",
            f"{summary['pool']:,} kernels: "
            + ", ".join(f"{n:,} {k}" for k, n in v.items())
            + f"; {summary['blocked']} blocked"
            + (
                f"; {summary['checked']:,} checked now in {summary['tasks']} "
                f"harness processes on GPUs {summary['gpus']}"
                if summary["checked"]
                else "; nothing to check"
            ),
        )
        doc = json.loads((args.out_dir / cx.BLOCKLIST_FILE).read_text())
        for e in doc["entries"]:
            label = e.get("kernel_id") or ki.label_of_name(e["name"])
            ui.warn("blocked", f"{label}: {e.get('reason') or e.get('source')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

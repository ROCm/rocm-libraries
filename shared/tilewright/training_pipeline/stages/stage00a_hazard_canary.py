#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage00a_hazard_canary -- the kernels of the hazard classes, each alone
(lib/hazards.py).

The orchestrator runs it before anything else of an invocation when
`hazards.enabled` (the default): before the correctness gate and the audit
set, which run many kernels in one process. Every kernel of the config's pool
(the library's table and the bank) that a hazard class names, no list of
`correctness.lists` names and the canary store (--cache-dir) holds no verdict
of runs alone on the GPUs of --devices, one process at a time on each, under
their leases (lib/gpu_lease.py) and the stall watchdog. With nothing to run
it takes no GPU.

Outputs in --out-dir:
  quarantine.json      the pool's quarantined kernels, and the hazard-class
                       kernels whose run was cut short (kept out of the
                       invocation), as a blocklist (lib/correctness.py) that
                       the orchestrator adds to the invocation's
  verdicts.json        every kernel of the pool's hazard classes: its classes
                       and verdict (pass, quarantined, unchecked), where it
                       comes from (canary, cache) and what its run saw
  canary_summary.json  counts, device, protocol, settings, why it stopped
  logs/                each run's bench yaml and output, and the control runs'

The exit status is 0 when the canary ran (a quarantine is a finding), 1 for
a config, library or bench problem, such as control kernels that run none of
the canary's problems (no kernel is blamed), 2 when a GPU lease stayed busy,
and 3 when a GPU became unusable (a control kernel that fails, or after a
hang or fault one that runs no canary problem, a process stuck in the GPU
driver or a pool without a kernel outside the hazard classes to check the
device with) or a process stuck in the GPU driver kept the canary from
starting: the verdicts are recorded, and the same command goes on once the
device is reset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import yaml  # noqa: E402

from lib import gpu_lease  # noqa: E402
from lib import hazards as hz  # noqa: E402
from lib import procs  # noqa: E402
from lib import ui  # noqa: E402


def _devices(text: Optional[str], cfg: dict) -> List[int]:
    if text:
        return [int(d) for d in text.split(",") if d.strip()]
    return list(hz.Settings.of(cfg).devices or cfg["runtime"]["devices"][:1])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="stage00a: run the pool's hazard-class kernels alone"
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
    ap.add_argument("--bench-binary", type=Path, required=True)
    ap.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="the hipBLASLt build the bench belongs to (its host libraries enter "
        "the protocol)",
    )
    ap.add_argument(
        "--hardware-device",
        type=int,
        default=0,
        help="device whose bank and kind the verdicts are of",
    )
    ap.add_argument(
        "--devices",
        default=None,
        help="comma list of devices to run on (default: hazards.devices, else "
        "the first of runtime.devices)",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="canary store (default: hazards.cache_dir, else <out-dir>/cache)",
    )
    ap.add_argument("--lease-wait-s", type=float, default=0.0)
    ap.add_argument(
        "--startup-grace-s",
        type=float,
        default=900.0,
        help="seconds a bench may print nothing before its first problem ends",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would run; run nothing",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config_yaml.read_text()) or {}
    stem = args.library_stem or str((cfg.get("hipblaslt") or {}).get("library_stem"))
    st = hz.Settings.of(cfg)
    cache_dir = args.cache_dir or (
        Path(st.cache_dir) if st.cache_dir else args.out_dir / "cache"
    )
    say = (lambda _m: None) if args.quiet else (lambda m: ui.info("canary", m))
    if not args.quiet:
        ui.banner("stage00a  hazard canary", ui.C.HEAD)
        ui.info("cfg", f"library stem : {stem}")
        ui.info("cfg", f"library dir  : {args.dat_dir}")
        ui.info("cfg", f"store        : {cache_dir}")
    children = procs.ChildGroups()
    try:
        with procs.stop_children_on_exit(children):
            summary = hz.run_canary(
                hz.Spec(
                    cfg=cfg,
                    dat_dir=args.dat_dir,
                    library_stem=stem,
                    out_dir=args.out_dir,
                    cache_dir=cache_dir,
                    bench_binary=args.bench_binary,
                    build_dir=args.build_dir,
                    hardware_device=args.hardware_device,
                    devices=_devices(args.devices, cfg),
                    lease_wait_s=args.lease_wait_s,
                    startup_grace_s=args.startup_grace_s,
                    dry_run=args.dry_run,
                ),
                say=say,
                warn=lambda m: ui.warn("canary", m),
                children=children,
            )
    except gpu_lease.LeaseBusy as e:
        ui.err("lease", str(e))
        return 2
    except gpu_lease.LeaseError as e:
        ui.err("lease", str(e))
        return 1
    except hz.CanaryError as e:
        ui.err("canary", str(e))
        return 1
    if summary["stopped"]:
        ui.err("canary", f"stopped: {summary['stopped']}")
    if not args.quiet:
        ui.info(
            "canary",
            f"{summary['hazard']:,} hazard-class kernels, "
            f"{summary['quarantined']:,} quarantined, "
            f"{summary['kept_out']:,} kept out unchecked, {summary['checked']:,} ran now "
            f"({summary['seconds']:g} s)",
        )
    return int(summary["rc"])


if __name__ == "__main__":
    sys.exit(main())

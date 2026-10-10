#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Evaluate models on an audit set (lib/audit.py), e.g. the rounds of a run
that finished before the run had one.

    audit_eval.py --audit-dir <audit> --config <run config> \
        --round-dir <run>/round_0 --round-dir <run>/round_1 ... --out audit.json

--round-dir evaluates the round's candidate: the bundle stage06 staged
(deploy_record.json `staged_model`) when there is one, else its stage05
models.pt (`model_source` says which), and leaves out audit shapes found in
that round or an earlier one (for a final retrain's directory, in the rounds
its final_retrain.json names); --model-file evaluates a models.pt or MLREC
file, leaving out the shapes of the --trained-round-dir rounds. A models.pt
is written at --weight-dtype (default deploy.weight_dtype), as stage06
deploys it; the pool is the config's library and bank, the pick rule
validate.pick_measurement.

While the audit set is benched a record is `partial`: the groups whose
parts are all benched and enriched are evaluated, the others are null with
their status. --enrich-dir enriches, there (never in the audit directory),
the groups whose parts are all benched but whose rows the audit builder has
not enriched yet, and evaluates them too; it is reused while the group's
logs are unchanged.

Every record holds protocol v1, the audit's single exhaustive reading per
kernel (`protocol` v1), and, unless --protocol v1, protocol v2 in its `v2`
(lib/audit_v2.py: medians of repeated joint benches of each GEMM's
near-winners and picks, from `<audit>/v2/` or --v2-dir); v2 is null or
partial with the GEMMs it lacks while they are benched or need an
increment. --v2-bench GPUS first registers the models with the v2
measurement and benches the GEMMs whose sets lack their picks on those
physical GPUs (scripts/audit_v2.py run), then scores.

--out receives the record of the single model, or {"models": [...]} for
several, with `changes` (v1) and `changes_v2`: each model against the one
before it, paired GEMM by GEMM on every group both evaluated (lib/audit.py
`compare`). A model's record is null (status missing / incomplete / error)
when the audit set cannot measure it; the exit status is still 0, 1 when a
--v2-bench worker failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from lib import audit as auditlib  # noqa: E402
from lib import hardware as hwlib  # noqa: E402
from lib import ui  # noqa: E402


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--audit-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True, help="the run's config")
    ap.add_argument("--round-dir", type=Path, action="append", default=[])
    ap.add_argument("--model-file", type=Path, action="append", default=[])
    ap.add_argument(
        "--trained-round-dir",
        type=Path,
        action="append",
        default=[],
        help="rounds the --model-file models trained on",
    )
    ap.add_argument("--weight-dtype", default=None)
    ap.add_argument(
        "--library-dir",
        type=Path,
        default=None,
        help="Tensile library dir (default: the config's build)",
    )
    ap.add_argument("--hardware-device", type=int, default=None)
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--bootstrap", type=int, default=None)
    ap.add_argument(
        "--enrich-dir",
        type=Path,
        default=None,
        help="enrich benched groups the audit builder has not enriched here",
    )
    ap.add_argument(
        "--protocol",
        choices=("both", "v1"),
        default="both",
        help="v1 alone, or v1 with the v2 record (default)",
    )
    ap.add_argument("--v2-dir", type=Path, default=None, help="default <audit-dir>/v2")
    ap.add_argument(
        "--v2-bench",
        default=None,
        metavar="GPUS",
        help="register the models with the v2 measurement and bench what "
        "their picks need on these physical GPUs (e.g. 1,2) before scoring",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.v2_bench and (args.protocol == "v1" or args.v2_dir is not None):
        ap.error("--v2-bench benches into <audit-dir>/v2 and needs --protocol both")
    if not args.round_dir and not args.model_file:
        ap.error("give --round-dir or --model-file")
    if args.enrich_dir is not None and (
        auditlib.audit_root(args.enrich_dir) is not None
        or _inside(args.enrich_dir, args.audit_dir)
    ):
        ap.error("--enrich-dir must lie outside the audit set")
    return args


def _inside(path: Path, root: Path) -> bool:
    p, r = Path(path).resolve(), Path(root).resolve()
    return p == r or r in p.parents


def round_model(rd: Path) -> Tuple[Path, str]:
    """(model file, source) of a round's candidate (`lib.audit.bundle_model`):
    stage06's staged bundle, else stage05's models.pt."""
    return auditlib.bundle_model(rd)


def trained_rounds(rd: Path) -> List[Path]:
    """The rounds whose shapes the bundle in `rd` trained on
    (`lib.audit.bundle_trained_rounds`)."""
    return auditlib.bundle_trained_rounds(rd)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import run_orchestrator as ro

    args = _parse_args(argv)
    try:
        cfg, _ = ro.load_config(args.config)
    except ro.ConfigError as e:
        for err in e.errors:
            ui.err("cfg", err)
        return 2
    st = auditlib.settings(cfg)
    library_dir = args.library_dir or ro.derive_paths(cfg).library_dir
    device = (
        args.hardware_device
        if args.hardware_device is not None
        else int(cfg["runtime"]["devices"][0])
    )
    dtype = args.weight_dtype or str(ro.cfg_get(cfg, "deploy.weight_dtype"))
    try:
        hardware = hwlib.device_hardware(cfg, device)
    except (ValueError, RuntimeError, OSError) as e:
        ui.err("hardware", str(e))
        return 2
    try:
        cfg, _ = ro.resolve_contexts(cfg, device)
    except ro.ConfigError as e:
        for err in e.errors:
            ui.err("cfg", err)
        return 2

    def spec(trained: List[Path]) -> auditlib.Spec:
        return auditlib.Spec(
            audit_dir=args.audit_dir,
            library_dir=library_dir,
            library_stem=str(cfg["hipblaslt"]["library_stem"]),
            hardware_device=device,
            pick_measurement=str(ro.cfg_get(cfg, "validate.pick_measurement")),
            target=float(args.target if args.target is not None else st["target"]),
            bootstrap=int(
                args.bootstrap if args.bootstrap is not None else st["bootstrap"]
            ),
            trained_rounds=trained,
            enrich_root=args.enrich_dir,
            protocol=args.protocol,
            v2_dir=args.v2_dir,
        )

    if args.enrich_dir is not None:
        for label, problem in ro.enrich_audit_groups(
            args.audit_dir, args.enrich_dir
        ).items():
            ui.warn("enrich", f"{label}: {problem}")
    bench_rc = 0
    if args.v2_bench:
        bench_rc = _v2_bench(args)
        if bench_rc == 2:
            return 2
    jobs = []
    for rd in args.round_dir:
        path, source = round_model(rd)
        jobs.append((rd.name, path, source, trained_rounds(rd)))
    jobs += [
        (f.name, f, "model file", list(args.trained_round_dir)) for f in args.model_file
    ]
    records: List[Dict[str, Any]] = []
    for name, path, source, trained in jobs:
        s = spec(trained)
        if not path.is_file():
            rec = auditlib.null_record(
                "error", f"{path} does not exist", audit_dir=str(args.audit_dir)
            )
        else:
            try:
                data = auditlib.model_bytes(path, str(cfg["arch"]), dtype)
                rec = auditlib.evaluate(data, s, cfg, hardware=hardware)
            except Exception as e:  # noqa: BLE001
                rec = auditlib.null_record(
                    "error", f"{type(e).__name__}: {e}", audit_dir=str(args.audit_dir)
                )
        rec.update(what=name, model=str(path), model_source=source)
        records.append(auditlib.json_ready(rec))
        if not args.quiet:
            say = ui.ok if rec.get("status") == "ok" else ui.warn
            for line in auditlib.summary_lines(rec):
                say(name, line)
    out: Any = records[0]
    if len(records) > 1:
        out = {
            "models": records,
            "changes": auditlib.changes(records, spec([]).bootstrap),
        }
        if args.protocol == "both":
            out["changes_v2"] = auditlib.changes(records, spec([]).bootstrap, "v2")
        if not args.quiet:
            for c in out["changes"] + out.get("changes_v2", []):
                for line in auditlib.change_lines(c):
                    ui.info("change", line)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    if not args.quiet:
        ui.ok("write", str(args.out))
    return bench_rc


def model_file_name(path: Path, data: bytes) -> str:
    """The v2 name of a --model-file model: its stem and sha256 prefix."""
    import hashlib

    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(path).stem)
    return f"{stem}.{hashlib.sha256(data).hexdigest()[:12]}"


def _v2_bench(args: argparse.Namespace) -> int:
    """scripts/audit_v2.py run of the --round-dir and --model-file models."""
    import run_orchestrator as ro

    import audit_v2

    cfg, _ = ro.load_config(args.config)
    dtype = args.weight_dtype or str(ro.cfg_get(cfg, "deploy.weight_dtype"))
    argv = ["run", "--audit-dir", str(args.audit_dir), "--config", str(args.config)]
    argv += ["--gpus", str(args.v2_bench)]
    for rd in args.round_dir:
        argv += ["--round-dir", str(rd)]
    for f in args.model_file:
        data = auditlib.model_bytes(f, str(cfg["arch"]), dtype)
        argv += ["--model", f"{model_file_name(f, data)}={f}"]
    for key in ("weight_dtype", "library_dir", "hardware_device"):
        if getattr(args, key) is not None:
            argv += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    return audit_v2.main(argv)


if __name__ == "__main__":
    sys.exit(main())

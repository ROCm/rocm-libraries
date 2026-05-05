"""Validator coverage probe — proves whether the graph-native validator
(`compare_graphs` + `validate_edge_wait_coverage`) actually runs end-to-end
when TensileCreateLibrary builds a given logic yaml.

Background
----------
KernelWriter.kernelBody auto-activates `_captureDefaultSchedule = True` for
any kernel with `UseCustomMainLoopSchedule` truthy (KernelWriter.py:4692).
The validator hook at KernelWriter.py:5192 then builds the dataflow graph
(`build_dataflow_graph`) and calls `compare_graphs` + `validate_edge_wait_coverage`.

Build success alone does NOT prove the validator ran — a quiet skip or an
exception caught upstream would also produce a successful build. This probe
wraps the validator entry points with counters so we have direct evidence.

Usage
-----
    env -C <tensilelite_root> \
      PYTHONPATH=<tensilelite_root> \
      python3 Tensile/Components/GFX1151_AUDIT/validator_coverage_probe.py \
      [--yaml <path>] [--out <dir>] [--jobs N]

Default yaml: gfx1151 HHS_BH_Bias_HAS_SAV_UserArgs from the rocBLASLt logic tree.
Default out: /tmp/validator_coverage_probe_out
Default jobs: 1 (parallel workers may not inherit the parent-process monkey-
patches; --jobs 1 is the only mode that gives reliable counter readings).

Output
------
Prints `[PROBE]` lines for each wrapped function call. At end, prints the
final tally and writes a JSON report next to the build output.

Known limitations
-----------------
- Parent-process monkey-patching of `Tensile.Components.ScheduleCapture` may
  not propagate to multiprocessing worker subprocesses depending on start
  method. Use `--jobs 1` when counting calls.
- The probe does NOT modify ScheduleCapture.py; it intercepts at the module
  attribute level. If the validator-hook code in KernelWriter.py changes its
  import pattern, this probe needs to be updated.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

DEFAULT_YAML_REL_TO_TENSILELITE = (
    "../library/src/amd_detail/rocblaslt/src/Tensile/"
    "Logic/asm_full/gfx1151/Equality/"
    "gfx1151_Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml"
)
DEFAULT_OUT = "/tmp/validator_coverage_probe_out"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--yaml", default=None,
                    help="Logic yaml to feed TensileCreateLibrary (default: "
                         "gfx1151 HHS_BH_Bias from rocBLASLt logic tree).")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="Output dir for TensileCreateLibrary.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="--jobs to pass through. Counters are only reliable "
                         "with --jobs 1 (workers may not inherit patches).")
    ap.add_argument("--arch", default="gfx1151",
                    help="--architecture to pass through.")
    args = ap.parse_args()

    # Resolve the yaml path. Caller is expected to be in the tensilelite root
    # (relative paths above resolve from there).
    cwd = Path.cwd()
    yaml_path = Path(args.yaml) if args.yaml else (cwd / DEFAULT_YAML_REL_TO_TENSILELITE).resolve()
    if not yaml_path.exists():
        print(f"[PROBE] FATAL: yaml not found at {yaml_path}", file=sys.stderr)
        sys.exit(2)

    # Stage a logic dir containing only the chosen yaml so the build is small
    # and targeted.
    out_dir = Path(args.out)
    logic_dir = Path(args.out + "_logic")
    if logic_dir.exists():
        shutil.rmtree(logic_dir)
    logic_dir.mkdir(parents=True)
    shutil.copy(yaml_path, logic_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    print(f"[PROBE] yaml: {yaml_path}")
    print(f"[PROBE] logic dir: {logic_dir}")
    print(f"[PROBE] out dir: {out_dir}")
    print(f"[PROBE] jobs: {args.jobs}")

    # Wrap the three validator entry points used by KernelWriter's hook.
    import Tensile.Components.ScheduleCapture as scap

    counts = {
        "build_dataflow_graph": 0,
        "compare_graphs": 0,
        "validate_edge_wait_coverage": 0,
        "errors": [],
    }

    def make_wrapper(name, orig):
        def wrapped(*a, **kw):
            counts[name] += 1
            try:
                out = orig(*a, **kw)
            except Exception as e:
                counts["errors"].append(f"{name}: {type(e).__name__}: {e}")
                print(f"[PROBE] {name}() #{counts[name]} raised "
                      f"{type(e).__name__}: {e}", flush=True)
                raise
            n = len(out) if hasattr(out, "__len__") else "?"
            print(f"[PROBE] {name}() #{counts[name]} returned (len={n})",
                  flush=True)
            return out
        return wrapped

    scap.build_dataflow_graph = make_wrapper(
        "build_dataflow_graph", scap.build_dataflow_graph)
    scap.compare_graphs = make_wrapper(
        "compare_graphs", scap.compare_graphs)
    scap.validate_edge_wait_coverage = make_wrapper(
        "validate_edge_wait_coverage", scap.validate_edge_wait_coverage)

    sys.argv = [
        "TensileCreateLibrary",
        "--architecture", args.arch,
        "--no-enumerate",
        "--jobs", str(args.jobs),
        str(logic_dir),
        str(out_dir),
        "HIP",
    ]

    from Tensile.TensileCreateLibrary.Run import run as tcl_run
    tcl_status = "ok"
    tcl_error = None
    try:
        tcl_run()
    except SystemExit as e:
        tcl_status = f"SystemExit({e.code})"
    except Exception as e:
        tcl_status = f"raised: {type(e).__name__}"
        tcl_error = f"{type(e).__name__}: {e}"
        print(f"[PROBE] TensileCreateLibrary {tcl_status}: {tcl_error}",
              flush=True)

    obj_files = sorted(str(p) for p in
                       Path(args.out).rglob("*.o")) if Path(args.out).exists() else []
    asm_files = sorted(str(p) for p in
                       Path(args.out).rglob("*.s")) if Path(args.out).exists() else []

    report = {
        "yaml": str(yaml_path),
        "out_dir": str(out_dir),
        "jobs": args.jobs,
        "tcl_status": tcl_status,
        "tcl_error": tcl_error,
        "build_dataflow_graph_calls": counts["build_dataflow_graph"],
        "compare_graphs_calls": counts["compare_graphs"],
        "validate_edge_wait_coverage_calls": counts["validate_edge_wait_coverage"],
        "errors_during_calls": counts["errors"],
        "obj_files_emitted": len(obj_files),
        "asm_files_emitted": len(asm_files),
    }

    print()
    print("[PROBE] === FINAL REPORT ===")
    for k, v in report.items():
        if isinstance(v, list):
            print(f"[PROBE]   {k}: {len(v)} entr{'y' if len(v) == 1 else 'ies'}")
            for entry in v[:5]:
                print(f"[PROBE]     - {entry}")
            if len(v) > 5:
                print(f"[PROBE]     ... ({len(v) - 5} more)")
        else:
            print(f"[PROBE]   {k}: {v}")

    report_path = Path(args.out + "_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[PROBE] report written to {report_path}")


if __name__ == "__main__":
    main()

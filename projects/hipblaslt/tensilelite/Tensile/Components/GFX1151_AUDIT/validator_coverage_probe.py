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
    ap.add_argument("--disable-validator", action="store_true",
                    help="No-op the validator entry points so the build can "
                         "complete even if it would otherwise crash. Useful "
                         "for getting raw .s/.o output to inspect the kernel "
                         "shape independent of the validator's expectations.")
    ap.add_argument("--keep-asm", action="store_true",
                    help="Pass --keep-build-tmp to TensileCreateLibrary so "
                         ".s files survive in build_tmp/.")
    ap.add_argument("--per-kernel-log", action="store_true",
                    help="Hook KernelWriter.kernelBody to log per-kernel: "
                         "kernel name, PGR/PLR/DTL settings, "
                         "_captureDefaultSchedule activation, ctx.cms presence, "
                         "default body lengths (main/n_gl/n_ll). Useful for "
                         "diagnosing which kernels reach the validator and "
                         "which silently skip.")
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
    per_kernel_records = []

    if args.per_kernel_log:
        import Tensile.KernelWriter as kw_mod
        _orig_kernelBody = kw_mod.KernelWriter.kernelBody

        def _name_param_value(name, key):
            """Extract value of an underscore-delimited param like 'PGR1' from
            a Tensile-encoded kernel name."""
            for part in name.split("_"):
                if part.startswith(key) and len(part) > len(key):
                    rest = part[len(key):]
                    return rest
            return None

        def wrapped_kernelBody(self, kernel, tpA, tpB):
            kname = kernel.get("Name", kernel.get("KernelName", "<unknown>"))
            mt0 = kernel.get("MacroTile0")
            mt1 = kernel.get("MacroTile1")
            depthU = kernel.get("DepthU")
            pgr = kernel.get("PrefetchGlobalRead")
            plr = kernel.get("PrefetchLocalRead")
            dtla = kernel.get("DirectToLdsA")
            dtlb = kernel.get("DirectToLdsB")
            cms = kernel.get("UseCustomMainLoopSchedule")
            isa = kernel.get("ISA")
            sia = kernel.get("ScheduleIterAlg")

            try:
                result = _orig_kernelBody(self, kernel, tpA, tpB)
                exc = None
            except Exception as e:
                result = None
                exc = f"{type(e).__name__}: {e}"
                raise
            finally:
                # Sample CaptureContext state if present (may have been reset
                # by the validator-hook finally; we sample BEFORE that runs by
                # also hooking reset, but easier path: read consumer-facing
                # ctx.default which is preserved across reset).
                ctx = getattr(self.states, "_capture_context", None) or \
                    getattr(self, "_capture_context", None)
                ctx_default = getattr(ctx, "default", None) if ctx else None
                ctx_cms = getattr(ctx, "cms", None) if ctx else None

                def body_lens(fpc, attr):
                    if fpc is None:
                        return None
                    by_cp = getattr(fpc, attr, None)
                    if not isinstance(by_cp, dict):
                        return None
                    return {cp: len(b.instructions) for cp, b in by_cp.items()
                            if b is not None}

                cap_default_active = bool(getattr(self.states, "_captureDefaultSchedule", False))

                rec = {
                    "kernel": kname,
                    "isa": list(isa) if isa else None,
                    "MT": (mt0, mt1, depthU),
                    "PGR": pgr, "PLR": plr, "DTLA": dtla, "DTLB": dtlb,
                    "CMS": cms, "SIA": sia,
                    "captureDefault_active": cap_default_active,
                    "ctx_default_present": ctx_default is not None,
                    "ctx_cms_present": ctx_cms is not None,
                    "default_main_lens": body_lens(ctx_default, "main_loop"),
                    "default_n_gl_lens": body_lens(ctx_default, "n_gl"),
                    "default_n_ll_lens": body_lens(ctx_default, "n_ll"),
                    "cms_main_lens": body_lens(ctx_cms, "main_loop"),
                    "cms_n_gl_lens": body_lens(ctx_cms, "n_gl"),
                    "cms_n_ll_lens": body_lens(ctx_cms, "n_ll"),
                    "exception": exc,
                }
                per_kernel_records.append(rec)
                # Compact one-line summary
                ngl_default = rec["default_n_gl_lens"]
                print(f"[PROBE-KERNEL] MT={mt0}x{mt1}x{depthU} PGR={pgr} "
                      f"PLR={plr} DTL={dtla}/{dtlb} CMS={cms} SIA={sia} "
                      f"capDef={cap_default_active} ctx.cms={ctx_cms is not None} "
                      f"default.n_gl={ngl_default} exc={exc}", flush=True)
            return result

        kw_mod.KernelWriter.kernelBody = wrapped_kernelBody

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

    if args.disable_validator:
        # Replace the three validator entry points with no-ops that count
        # invocations but never raise and return inert values. KernelWriter's
        # hook chains them: build_dataflow_graph -> compare_graphs ->
        # validate_edge_wait_coverage. Both compare_graphs and
        # validate_edge_wait_coverage return failure lists (empty list = ok).
        # build_dataflow_graph returns a DataflowGraph; we return a sentinel
        # the next two ignore.
        from Tensile.Components.ScheduleCapture import DataflowGraph
        empty_graph = DataflowGraph(nodes={}, edges=[], captures={})

        def disabled_build(*a, **kw):
            counts["build_dataflow_graph"] += 1
            print(f"[PROBE] build_dataflow_graph() #{counts['build_dataflow_graph']} DISABLED",
                  flush=True)
            return empty_graph

        def disabled_compare(*a, **kw):
            counts["compare_graphs"] += 1
            print(f"[PROBE] compare_graphs() #{counts['compare_graphs']} DISABLED",
                  flush=True)
            return []

        def disabled_validate(*a, **kw):
            counts["validate_edge_wait_coverage"] += 1
            print(f"[PROBE] validate_edge_wait_coverage() #{counts['validate_edge_wait_coverage']} DISABLED",
                  flush=True)
            return []

        scap.build_dataflow_graph = disabled_build
        scap.compare_graphs = disabled_compare
        scap.validate_edge_wait_coverage = disabled_validate
    else:
        scap.build_dataflow_graph = make_wrapper(
            "build_dataflow_graph", scap.build_dataflow_graph)
        scap.compare_graphs = make_wrapper(
            "compare_graphs", scap.compare_graphs)
        scap.validate_edge_wait_coverage = make_wrapper(
            "validate_edge_wait_coverage", scap.validate_edge_wait_coverage)

    tcl_argv = [
        "TensileCreateLibrary",
        "--architecture", args.arch,
        "--no-enumerate",
        "--jobs", str(args.jobs),
    ]
    if args.keep_asm:
        tcl_argv.append("--keep-build-tmp")
    tcl_argv += [str(logic_dir), str(out_dir), "HIP"]
    sys.argv = tcl_argv

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
        "per_kernel": per_kernel_records,
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

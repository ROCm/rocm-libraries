"""LR divergence capture probe — dumps per-(body, category) instruction-level
captures from BOTH default-side and CMS-side FourPartCapture for every kernel
that reaches the validator hook.

This is an extension of validator_coverage_probe.py that intercepts
`compare_graphs` so we can sample the FourPartCapture object pair just BEFORE
the consistency check fires, and write a JSON report containing each
TaggedInstruction's (category, slot, rocisa class name, asm-rendered string)
on both sides.

The probe disables the validator (build_dataflow_graph / compare_graphs /
validate_edge_wait_coverage no-op'd) so the build completes past the
CaptureConsistencyError and all kernels report.

Usage
-----
    env -C <tensilelite_root> \
      PYTHONPATH=<tensilelite_root> \
      python3 Tensile/Components/GFX1151_AUDIT/lr_divergence_capture_probe.py \
      [--yaml <path>] [--out <dir>] [--report <path>]

Output
------
A JSON report with shape:

    {
      "yaml": "...",
      "tcl_status": "ok",
      "kernels": [
        {
          "kernel": "<kernel name>",
          "MT": [m, n, k], "PGR": ..., "PLR": ..., "CMS": ..., "SIA": ...,
          "default": {
            "main_loop":      {<codepath>: [{cat, slot, cls, asm}, ...]},
            "main_loop_prev": {...},
            "n_gl":           {0: [...]},
            "n_ll":           {0: [...]},
          },
          "cms": { ...same shape... }
        },
        ...
      ]
    }

The kernel-config gate matches the bead's targets: only kernels with CMS=1.
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
DEFAULT_OUT = "/tmp/lr_divergence_probe_out"
DEFAULT_REPORT = "/tmp/lr_divergence_probe_report.json"


def _slot_to_obj(slot):
    return {
        "subiter": getattr(slot, "subiter", None),
        "slot_kind": getattr(slot, "slot_kind", None),
        "mfma_index": getattr(slot, "mfma_index", None),
        "sequence": getattr(slot, "sequence", None),
    }


def _ti_to_obj(ti):
    """Convert a TaggedInstruction to a JSON-friendly dict."""
    wrapped = getattr(ti, "wrapped", None)
    inst = getattr(wrapped, "rocisa_inst", None) if wrapped is not None else None
    cls = type(inst).__name__ if inst is not None else None
    try:
        asm = str(inst).strip() if inst is not None else None
    except Exception as e:
        asm = f"<render-error: {type(e).__name__}: {e}>"
    return {
        "category": ti.category,
        "slot": _slot_to_obj(ti.slot),
        "cls": cls,
        "asm": asm,
    }


def _body_dict_to_obj(body_dict):
    if not isinstance(body_dict, dict):
        return None
    out = {}
    for cp, body in body_dict.items():
        if body is None:
            out[str(cp)] = None
            continue
        instrs = getattr(body, "instructions", None)
        if instrs is None:
            out[str(cp)] = None
            continue
        out[str(cp)] = [_ti_to_obj(ti) for ti in instrs]
    return out


def _fpc_to_obj(fpc):
    if fpc is None:
        return None
    return {
        "source": getattr(fpc, "source", None),
        "num_mfma": getattr(fpc, "num_mfma", None),
        "num_codepaths": getattr(fpc, "num_codepaths", None),
        "main_loop":      _body_dict_to_obj(getattr(fpc, "main_loop",      None)),
        "main_loop_prev": _body_dict_to_obj(getattr(fpc, "main_loop_prev", None)),
        "n_gl":           _body_dict_to_obj(getattr(fpc, "n_gl",           None)),
        "n_ll":           _body_dict_to_obj(getattr(fpc, "n_ll",           None)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--yaml", default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--arch", default="gfx1151")
    args = ap.parse_args()

    cwd = Path.cwd()
    yaml_path = Path(args.yaml) if args.yaml else (cwd / DEFAULT_YAML_REL_TO_TENSILELITE).resolve()
    if not yaml_path.exists():
        print(f"[PROBE] FATAL: yaml not found at {yaml_path}", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out)
    logic_dir = Path(args.out + "_logic")
    if logic_dir.exists():
        shutil.rmtree(logic_dir)
    logic_dir.mkdir(parents=True)
    shutil.copy(yaml_path, logic_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    print(f"[PROBE] yaml: {yaml_path}")
    print(f"[PROBE] out: {out_dir}")
    print(f"[PROBE] report: {args.report}")

    import Tensile.Components.ScheduleCapture as scap
    from Tensile.Components.ScheduleCapture import DataflowGraph
    empty_graph = DataflowGraph(nodes={}, edges=[], captures={})

    # Hook KernelWriter.kernelBody so we can sample ctx.default + ctx.cms
    # AFTER the kernel body builds but BEFORE the validator-hook reset.
    import Tensile.KernelWriter as kw_mod
    _orig_kernelBody = kw_mod.KernelWriter.kernelBody
    kernel_records = []

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
        sia = kernel.get("ScheduleIterAlg")
        isa = kernel.get("ISA")

        try:
            result = _orig_kernelBody(self, kernel, tpA, tpB)
            exc = None
        except Exception as e:
            result = None
            exc = f"{type(e).__name__}: {e}"
            raise
        finally:
            ctx = getattr(self.states, "_capture_context", None) or \
                getattr(self, "_capture_context", None)
            ctx_default = getattr(ctx, "default", None) if ctx else None
            ctx_cms = getattr(ctx, "cms", None) if ctx else None

            rec = {
                "kernel": kname,
                "isa": list(isa) if isa else None,
                "MT": [mt0, mt1, depthU],
                "PGR": pgr, "PLR": plr,
                "DTLA": dtla, "DTLB": dtlb,
                "CMS": cms, "SIA": sia,
                "exception": exc,
                "default": _fpc_to_obj(ctx_default),
                "cms":     _fpc_to_obj(ctx_cms),
            }
            kernel_records.append(rec)
            ml_def = (rec["default"]["main_loop"] if rec["default"] else None)
            ml_cms = (rec["cms"]["main_loop"] if rec["cms"] else None)
            n_def = sum(len(v) for v in (ml_def or {}).values()) if ml_def else None
            n_cms = sum(len(v) for v in (ml_cms or {}).values()) if ml_cms else None
            print(f"[PROBE-KERNEL] MT={mt0}x{mt1}x{depthU} PGR={pgr} PLR={plr} "
                  f"CMS={cms} SIA={sia} ml_def={n_def} ml_cms={n_cms} exc={exc}",
                  flush=True)
        return result

    kw_mod.KernelWriter.kernelBody = wrapped_kernelBody

    # Disable the validator entry points so the build runs past
    # CaptureConsistencyError and all kernels report.
    def disabled_build(*a, **kw):
        return empty_graph

    def disabled_compare(*a, **kw):
        return []

    def disabled_validate(*a, **kw):
        return []

    scap.build_dataflow_graph = disabled_build
    scap.compare_graphs = disabled_compare
    scap.validate_edge_wait_coverage = disabled_validate

    sys.argv = [
        "TensileCreateLibrary",
        "--architecture", args.arch,
        "--no-enumerate",
        "--jobs", "1",
        str(logic_dir), str(out_dir), "HIP",
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
        print(f"[PROBE] TCL {tcl_status}: {tcl_error}", flush=True)

    report = {
        "yaml": str(yaml_path),
        "tcl_status": tcl_status,
        "tcl_error": tcl_error,
        "kernels": kernel_records,
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"[PROBE] wrote {args.report} ({len(kernel_records)} kernels)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""gfx942 BF16 D128 causal prefill: production dispatch vs the QK-lookahead knob
pin vs AITER's asm FMHA v3 forward, in one process on one HIP stream.

Arms (all on the same shared inputs, all on one dedicated HIP stream):

   production      rocKE production dispatch (``spec_id`` gfx942_dense, no knobs)
   qk_lookahead    rocKE ``spec_id`` gfx942_dense pinned to ``QKLA_KNOBS``
   aiter_shipped   AITER ``flash_attn_func(..., causal=True)``: its shipped selection,
                   the default ``how_v3_bf16_cvt`` (RTNA bf16 output rounding)
   aiter_rtne      the same with ``how_v3_bf16_cvt=0``: RTNE output rounding, the
                   rounding the rocKE kernels use (equivalent work)

Method, per shape:

1. Resolve both rocKE specs through dispatch, record each arm's replay record
   (spec_id, tuning_id, knobs, spec_hash) and replay it before timing. Compile one
   HSACO per rocKE arm and record its static resources.
2. One shared input set: N(0, 1) rounded to bf16, scaled by ``--input-scale``
   (default 0.2, as ``bench_rocke_7way_specs.py``) and rounded again. One chunked
   NumPy FP32 causal GQA reference over those exact inputs.
3. Before timing, every arm runs once (rocKE outputs poisoned with 0xFF first) and
   every output element is checked against the reference (``--atol`` / ``--rtol``).
4. ``--warmup`` round-robin launches per arm, then ``--rounds`` rounds with the arm
   order shuffled per round. Each arm's block is a HIP start event, ``--repeat``
   back-to-back launches with no host synchronization between them (rocKE
   launchers under ``no_fence``, as ``rocke.runtime.time_launches`` times), and a
   HIP stop event; per-launch time = elapsed / repeat. The last output of every
   block is checked against the reference outside the timed span.
5. Per arm, the median of its per-round times; a ratio is the ratio of two
   medians, and its paired range the min/max of the same-round ratios.

Run it at least three times (independent processes) and combine the runs with
``--combine``: the reported ratio is the median of the runs' ratios with their
min/max. Absolute times stay in the local ``--out-dir`` CSVs; publish ratios only.
A dirty checkout is refused unless ``--allow-dirty``. torch and AITER are needed
only for the two AITER arms (``--no-aiter`` runs the rocKE arms torch-free).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_rocke_7way_specs as b7  # noqa: E402  (shared harness helpers)

# The QK-lookahead pipeline as a gfx942_dense knob pin: the flag, its prerequisites
# and the xcd_local decode it runs on.
QKLA_KNOBS = {
    "qk_lookahead": True,
    "block_m": 256,
    "lds_num_buffers": 2,
    "v_reg_prefetch": True,
    "causal_diag_split": True,
    "lds_row_pad": 0,
    "persist_decode": "xcd_local",
}
# rocKE arm -> knobs pinned on spec_id gfx942_dense ({}: production's default).
ROCKE_ARMS = {"production": {}, "qk_lookahead": QKLA_KNOBS}
AITER_ARMS = {"aiter_shipped": {}, "aiter_rtne": {"how_v3_bf16_cvt": 0}}
# (baseline, candidate): speedup = median(baseline) / median(candidate).
PAIRS = (
    ("production", "qk_lookahead"),
    ("aiter_shipped", "qk_lookahead"),
    ("aiter_rtne", "qk_lookahead"),
    ("aiter_shipped", "production"),
)
HASHED_SOURCES = (
    *b7.HASHED_SOURCES,
    "library/kernels/common/attention_dense_decode.py",
    "library/benchmarks/gfx942/attention/prefill/bench_dense_qkla_vs_aiter.py",
)


def _aiter_provenance() -> dict:
    import aiter
    import torch

    root = Path(aiter.__file__).resolve().parent.parent
    return {
        "aiter_version": b7._package_version("amd-aiter"),
        "aiter_commit": b7._command_output(["git", "rev-parse", "HEAD"], cwd=root),
        "aiter_dirty": bool(
            b7._command_output(["git", "status", "--porcelain"], cwd=root)
        ),
        "torch_version": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
    }


def _source_hashes(rocke: Path) -> dict[str, str]:
    return {rel: b7._file_sha256(rocke / rel) for rel in HASHED_SOURCES}


def _bf16_inputs(shape, scale: float, seed: int):
    from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16

    rng = np.random.default_rng(seed)

    def one(dims):
        x = bf16_to_f32(f32_to_bf16(rng.standard_normal(dims, dtype=np.float32)))
        return f32_to_bf16(x * np.float32(scale))

    q = one((shape.batch, shape.seqlen, shape.hq, shape.d))
    k = one((shape.batch, shape.seqlen, shape.hkv, shape.d))
    v = one((shape.batch, shape.seqlen, shape.hkv, shape.d))
    return q, k, v


def _run_shape(args, shape, stream, out_dir, rng, aiter_on, rows):
    import kernels.gfx942.attention_dense as dense
    from dispatch.attention import attention_tuning_spec
    from rocke.helpers.compile import compile_kernel
    from rocke.numeric.bf16 import bf16_to_f32
    from rocke.runtime import KernelLauncher, LaunchConfig, no_fence
    from rocke.runtime.hip_module import Runtime

    rt = Runtime()
    arms = (*ROCKE_ARMS, *(AITER_ARMS if aiter_on else ()))
    print(f"\n=== {shape.label}: arms {list(arms)} ===", flush=True)

    # --- rocKE arms through dispatch, replayed before anything is timed.
    specs, replay = {}, {}
    for name, knobs in ROCKE_ARMS.items():
        req = replace(b7.make_request(shape), spec_id=b7.SPEC_ID)
        tuned = attention_tuning_spec(req, b7.SPEC_ID, knobs=knobs)
        record = b7.replay_record(req, tuned)
        if b7.replay_spec_hash(req, record) != record["spec_hash"]:
            raise RuntimeError(f"{shape.label}/{name}: replay changed spec_hash")
        ok, why = dense.supports_attention_dense(tuned.kernel_spec, arch="gfx942")
        if not ok:
            raise RuntimeError(f"{shape.label}/{name} rejected: {why}")
        specs[name], replay[name] = tuned.kernel_spec, record
    if not specs["qk_lookahead"].qk_lookahead or specs["production"].qk_lookahead:
        raise RuntimeError(f"{shape.label}: arms do not differ in qk_lookahead")

    q_bits, k_bits, v_bits = _bf16_inputs(shape, args.input_scale, args.seed)
    q, k, v = (b7._device(x) for x in (q_bits, k_bits, v_bits))
    poison = np.full(q_bits.shape, 0xFFFF, dtype=np.uint16)
    outputs = {name: b7._device(poison) for name in ROCKE_ARMS}
    scale = 1.0 / math.sqrt(shape.d)

    launch, static, compiled = {}, [], {}
    for name, spec in specs.items():
        art = compile_kernel(
            dense.build_attention_dense(spec, arch="gfx942"),
            arch="gfx942",
            backend="python",
            capture_ir_text=False,
        )
        path = out_dir / "hsaco" / shape.label / f"{name}.hsaco"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(art.hsaco)
        digest = hashlib.sha256(art.hsaco).hexdigest()
        compiled[name] = {"kernel_name": art.kernel_name, "hsaco_sha256": digest}
        static.append(
            {
                "shape": shape.label,
                "arm": name,
                "kernel_name": art.kernel_name,
                "hsaco_sha256": digest,
                **b7.static_resources(art.hsaco, path),
            }
        )
        launcher = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=dense.attention_dense_signature(spec),
        )
        config = LaunchConfig(
            grid=dense.attention_dense_grid(spec),
            block=dense.attention_dense_block(spec),
            stream=stream,
        )
        values = {
            "q_ptr": q,
            "k_ptr": k,
            "v_ptr": v,
            "o_ptr": outputs[name],
            "scale": scale,
            **dense.attention_dense_runtime_args(spec),
        }
        launch[name] = lambda launcher=launcher, values=values, config=config: launcher(
            values, config=config
        )

    last = {}  # aiter arm -> its last output tensor
    if aiter_on:
        import torch
        from aiter.ops.mha import flash_attn_func

        torch_stream = torch.cuda.ExternalStream(stream)

        def as_torch(bits):
            return torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16).cuda()

        tq, tk, tv = (as_torch(x) for x in (q_bits, k_bits, v_bits))
        torch.cuda.synchronize()
        for name, extra in AITER_ARMS.items():

            def call(name=name, extra=extra):
                with torch.cuda.stream(torch_stream):
                    last[name] = flash_attn_func(
                        tq, tk, tv, softmax_scale=scale, causal=True, **extra
                    )

            launch[name] = call

    print("  FP32 reference...", flush=True)
    reference = b7.fp32_reference(
        bf16_to_f32(q_bits),
        bf16_to_f32(k_bits),
        bf16_to_f32(v_bits),
        scale=scale,
        chunk=args.reference_chunk,
    )
    del q_bits, k_bits, v_bits, poison

    def check(name, stage, rnd=""):
        if name in ROCKE_ARMS:
            bits = b7._download_bf16(outputs[name], reference.shape)
        else:
            bits = last[name].view(torch.int16).cpu().numpy().view(np.uint16)
        max_abs, max_rel = b7.validate(bits, reference, atol=args.atol, rtol=args.rtol)
        rows["validation"].append(
            dict(
                shape=shape.label,
                arm=name,
                stage=stage,
                round=rnd,
                max_abs=f"{max_abs:.6e}",
                max_rel=f"{max_rel:.6e}",
                atol=args.atol,
                rtol=args.rtol,
                passed=True,
            )
        )

    for name in arms:
        launch[name]()
        rt.wait_stream(stream)
        check(name, "pre_timing")

    with no_fence():
        for _ in range(args.warmup):
            for name in arms:
                launch[name]()
    rt.wait_stream(stream)

    for rnd in range(args.rounds):
        order = list(arms)
        rng.shuffle(order)
        for position, name in enumerate(order):
            start, stop = rt.event(), rt.event()
            try:
                with no_fence():
                    start.record(stream=stream)
                    for _ in range(args.repeat):
                        launch[name]()
                    stop.record(stream=stream)
                stop.synchronize()
                ms = float(start.elapsed_to(stop)) / args.repeat
            finally:
                start.destroy()
                stop.destroy()
            rt.wait_stream(stream)
            check(name, "post_block", rnd)
            rows["runs"].append(
                dict(
                    shape=shape.label,
                    arm=name,
                    round=rnd,
                    position=position,
                    repeat=args.repeat,
                    ms_per_launch=f"{ms:.9f}",
                )
            )

    rows["static"].extend(static)
    rows["resolved"].append(
        {
            "shape": shape.label,
            "shape_fields": asdict(shape),
            "specs": {n: asdict(s) for n, s in specs.items()},
            "kernel_names": {n: s.kernel_name() for n, s in specs.items()},
            "replay": replay,
            "compiled": compiled,
            "aiter_calls": (
                {
                    n: {"causal": True, "softmax_scale": scale, **extra}
                    for n, extra in AITER_ARMS.items()
                }
                if aiter_on
                else {}
            ),
        }
    )


def _summarize(runs: list[dict]) -> list[dict]:
    """Per shape and pair: ratio of medians and the same-round min/max."""
    by = {}
    for r in runs:
        by.setdefault(r["shape"], {}).setdefault(r["arm"], {})[int(r["round"])] = float(
            r["ms_per_launch"]
        )
    out = []
    for shape, arms in by.items():
        for base, cand in PAIRS:
            if base not in arms or cand not in arms:
                continue
            per_round = [arms[base][k] / arms[cand][k] for k in arms[cand]]
            out.append(
                dict(
                    shape=shape,
                    baseline=base,
                    candidate=cand,
                    speedup=statistics.median(arms[base].values())
                    / statistics.median(arms[cand].values()),
                    paired_min=min(per_round),
                    paired_max=max(per_round),
                )
            )
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        b7.write_csv(path, tuple(rows[0]), rows)


def combine(dirs: list[Path]) -> int:
    """Median over independent runs of each pair's speedup, with min/max."""
    table = {}
    for d in dirs:
        prov = json.loads((d / "provenance.json").read_text())
        print(
            f"{d}: HEAD {prov['git']['head_sha']} dirty={prov['git']['working_tree_dirty']}"
        )
        with (d / "ratios.csv").open() as f:
            for r in csv.DictReader(f):
                key = (r["shape"], r["baseline"], r["candidate"])
                table.setdefault(key, []).append(float(r["speedup"]))
    pairs = sorted({(b, c) for _, b, c in table})
    shapes = sorted({s for s, _, _ in table})
    print("\n| shape | " + " | ".join(f"{c} vs {b}" for b, c in pairs) + " |")
    print("|---" * (len(pairs) + 1) + "|")
    for s in shapes:
        cells = []
        for b, c in pairs:
            v = table.get((s, b, c))
            cells.append(
                f"{statistics.median(v):.3f}x ({min(v):.3f}-{max(v):.3f}, n={len(v)})"
                if v
                else "-"
            )
        print(f"| {s} | " + " | ".join(cells) + " |")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--rocke-root")
    p.add_argument("--shape", type=b7.parse_shape, action="append")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument("--seed", type=int, default=942)
    p.add_argument("--input-scale", type=float, default=0.2)
    p.add_argument("--reference-chunk", type=int, default=128)
    p.add_argument("--atol", type=float, default=1e-2)
    p.add_argument("--rtol", type=float, default=1e-2)
    p.add_argument("--out-dir", default="~/rocke_qkla_vs_aiter_results")
    p.add_argument(
        "--no-aiter", action="store_true", help="rocKE arms only, torch-free"
    )
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--combine", nargs="+", metavar="RUN_DIR")
    args = p.parse_args()
    if args.combine:
        return combine([Path(d).expanduser() for d in args.combine])
    if args.rounds < 3 or args.repeat <= 0 or args.warmup < 0:
        p.error("require rounds >= 3, repeat > 0 and warmup >= 0")

    rocke = b7.locate_rocke(args.rocke_root)
    git_state = b7._capture_git_checkout(rocke, allow_dirty=args.allow_dirty)
    b7.add_module_paths(rocke)
    from rocke.runtime.hip_module import current_device, get_device_arch

    device = current_device()
    if get_device_arch(device) != "gfx942":
        raise SystemExit("This benchmark requires a gfx942 GPU")
    aiter_on = not args.no_aiter
    aiter_info = _aiter_provenance() if aiter_on else {}
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    shapes = args.shape or list(b7.DEFAULT_SHAPES)
    rng = random.Random(args.seed)
    rows = {"runs": [], "validation": [], "static": [], "resolved": []}
    started = datetime.now(timezone.utc).isoformat()
    with b7._hip_stream() as stream:
        for shape in shapes:
            _run_shape(args, shape, stream, out_dir, rng, aiter_on, rows)

    ratios = _summarize(rows["runs"])
    _write_csv(out_dir / "runs.csv", rows["runs"])
    _write_csv(out_dir / "validation.csv", rows["validation"])
    _write_csv(out_dir / "static_resources.csv", rows["static"])
    _write_csv(
        out_dir / "ratios.csv",
        [
            {**r, **{k: f"{r[k]:.6f}" for k in ("speedup", "paired_min", "paired_max")}}
            for r in ratios
        ],
    )
    (out_dir / "resolved_specs.json").write_text(json.dumps(rows["resolved"], indent=1))
    provenance = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state,
        "dirty_checkout_allowed": bool(args.allow_dirty),
        "source_sha256": _source_hashes(rocke),
        "environment": b7._environment(
            device, rocke, Path(__file__).resolve(), git_state
        ),
        "aiter": aiter_info,
        "method": {
            "arms": [*ROCKE_ARMS, *(AITER_ARMS if aiter_on else ())],
            "warmup": args.warmup,
            "rounds": args.rounds,
            "repeat": args.repeat,
            "seed": args.seed,
            "inputs": f"N(0,1) -> bf16, x{args.input_scale} -> bf16",
            "timing": "HIP events around repeat back-to-back launches, no host sync",
            "tolerance": {"atol": args.atol, "rtol": args.rtol},
        },
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=1))
    print(f"\nChecks passed: {len(rows['validation'])}; results in {out_dir}")
    for r in ratios:
        print(
            f"  {r['shape']:32s} {r['candidate']} vs {r['baseline']}: "
            f"{r['speedup']:.3f}x (paired {r['paired_min']:.3f}-{r['paired_max']:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

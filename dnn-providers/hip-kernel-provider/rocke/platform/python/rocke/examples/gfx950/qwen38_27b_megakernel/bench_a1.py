# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Time the A1 rocKE megakernel against ATOM's FlyDSL megakernel, in one process, on identical inputs.

Runs inside ATOM's Qwen3.8-27B megakernel environment (the ``blog_benchmark`` image of
https://github.com/proaditya/ATOM/tree/qwen38-megakernels), on one MI355X:

    MODEL_DIR=/path/to/Qwen3.8-27B  PROMPTS_DIR=/path/to/prompts  A1_OBJECT_DIR=/path/to/a1_build \\
        python bench_a1.py --process 1 [--contexts 1024,8192,32768] [--n 200]

Environment:
  MODEL_DIR       Qwen3.8-27B checkpoint directory (what ATOM's ``qwen38_27b_mono_smoke.Harness`` loads)
  PROMPTS_DIR     directory of ``*.jsonl`` files whose rows have a ``"prompt"`` key; their tokens are repeated
                  to fill the longest context
  A1_OBJECT_DIR   output directory of ``build_a1.py`` (kernel.hsaco + object.json)
  ATOM_TOOLS      directory holding ATOM's ``qwen38_27b_mono_smoke.py`` (default: /opt/atom/tools, its place in the
                  ATOM image)
  RESULTS_DIR     where ``timing-p<process>.json`` is written (default: ./results)

What one process does:
  1. load the weights and ATOM's cached FlyDSL ``m8`` build (arm ``flydsl``), and load A1 (arm ``a1``) on the same
     pointer table (A1 keeps FlyDSL's ABI: one pointer-table argument);
  2. prefill real prompt tokens up to each context length with the FlyDSL engine;
  3. per context and launch mode (``cooperative`` = standalone launch, ``graph`` = replay of a captured graph node,
     ATOM's serving path), alternate flydsl, a1, flydsl, a1 ... on identical inputs. Nothing is committed, so every
     launch sees the same state. Per launch, on one stream:

         e0  prepare (launch_begin + conv_gather + kv_import, as MonoEngine.step)  e1
             kernel.launch(table)                                                  e2
             copy the failure word, then compare outputs with the first launch

     ``kernel_ms`` is e1 -> e2 (CUDA/HIP events). One host sync per batch.

Two kinds of batch: ``timing`` (n per arm, only cheap checks between launches: failure word, finite logits, same
top token) and ``validation`` (n_check per arm, untimed: every buffer a later step consumes must be finite and within
STATE_RTOL of the first launch; the pools a launch only reads must stay bit-identical).

Read ``a1/flydsl`` median ratios from ``kernel_stats``. The kernel is not bit-reproducible launch to launch (atomic
split-K and atomic mixer sums), so compare medians over many launches and repeat in >= 3 fresh processes. The
reference here is ATOM's shipped FlyDSL build, not the FlyDSL reference arm used for the README's numbers.
Exit 0 = gate passed, 2 = incomplete (a mode could not run), 1 = failed.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.environ.get("ATOM_TOOLS", "/opt/atom/tools"))
from qwen38_27b_mono_smoke import Harness  # noqa: E402  (ATOM's own test harness)

from atom.models.qwen38_27b.mono import dims as d  # noqa: E402
from atom.models.qwen38_27b.mono import launch_kernels as lk  # noqa: E402
from atom.models.qwen38_27b.mono import state_kernels as sk  # noqa: E402
from atom.models.qwen38_27b.mono.engine import StepInputs  # noqa: E402
from atom.models.qwen38_27b.mono.kernels.hip import Megakernel  # noqa: E402

ROWS = 8
# The kernel is not bit-reproducible (atomic split-K + MXFP8 + atomic mixer sums), so identical inputs need not give
# identical bits; the relative max difference per buffer must stay within this.
STATE_RTOL = 1e-2


def env_path(name: str) -> Path:
    """A required local directory from the environment (never a hub id: nothing is downloaded)."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"set {name} (see the module docstring)")
    path = Path(value)
    if not path.is_dir():
        raise SystemExit(f"{name}={value} is not a directory")
    return path


def prompt_tokens(model: str, prompts: Path, need: int) -> tuple[list[int], int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    pieces = []
    for path in sorted(prompts.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            pieces.extend(
                tokenizer.encode(
                    json.loads(line)["prompt"] + "\n\n", add_special_tokens=False
                )
            )
    if not pieces:
        raise SystemExit(f"no prompt tokens found in {prompts}/*.jsonl")
    passes = 1
    ids = list(pieces)
    while len(ids) < need:  # real text, repeated to reach the longest context
        ids.extend(pieces)
        passes += 1
    return ids[:need], passes


def inputs(h: Harness, ids: list[int], base: int) -> StepInputs:
    """The inputs Harness.launch would build for the next launch at ``base``."""
    return StepInputs(
        token_ids=torch.tensor(ids, dtype=torch.int64, device=h.dev),
        base=h.i32(base),
        real_rows=h.i32(ROWS),
        state_slot=h.i32(h.accepted_prev - 1),
        conv_in=h.i32(0),
        window=h.i32(h.accepted_prev),
        conv_out=h.i32(0),
        block_table=h.table,
        kv_slots=h.slots(base, ROWS),
        request=h.i32(0),
    )


def prepare(h: Harness, binding, inp: StepInputs) -> None:
    """MonoEngine.step up to the kernel launch; never timed as kernel."""
    e, s = h.eng, h.eng.shared
    lk.launch_begin(
        binding,
        e._delta_rows,
        e._state_col,
        h.pools,
        inp,
        e.mirror_slot,
        e.mirror_len,
        e.import_start,
        e.import_end,
    )
    sk.conv_gather(h.pools.conv, inp.conv_in, inp.window, s.conv_history)
    sk.kv_import(
        h.pools.k_ptrs,
        h.pools.v_ptrs,
        inp.block_table,
        e.import_start,
        e.import_end,
        s.kv_ptrs,
        h.pools.block_size,
    )


def outputs(h: Harness, binding, full: bool) -> dict:
    """What one launch writes that a later step consumes."""
    if not full:
        return {"logits": binding.logits}
    s = h.eng.shared
    named = {
        "logits": binding.logits,
        "final_state": s.final_state,
        "replay_records": s.records,
        "conv_final": s.conv_final,
        "projected_rows": s.raw,
    }
    named.update({f"kv_mirror_{i}": t for i, t in enumerate(s.kv)})
    return named


def read_only_pools(h: Harness) -> list[torch.Tensor]:
    """Pools a launch reads but must never write (only a commit writes them)."""
    return [
        h.pools.state.clone(),
        h.pools.conv.clone(),
        *[t.clone() for t in h.kc + h.vc],
    ]


def stats(values: list[float]) -> dict:
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p10": pick(0.10),
        "p90": pick(0.90),
        "min": ordered[0],
        "max": ordered[-1],
    }


def batch(
    h,
    binding,
    launchers: dict,
    ids: list[int],
    base: int,
    n: int,
    warmup: int,
    full: bool,
) -> dict:
    """Interleave the launchers (A, B, A, B ...) with identical inputs; one host sync."""
    inp = inputs(h, ids, base)
    names = list(launchers)
    order = [names[i % len(names)] for i in range(warmup + n * len(names))]
    total = len(order)
    ev = [
        [torch.cuda.Event(enable_timing=True) for _ in range(3)] for _ in range(total)
    ]
    failed = torch.zeros(total, dtype=torch.int32, device=h.dev)
    agree = torch.zeros(total, dtype=torch.int32, device=h.dev)
    before = read_only_pools(h) if full else None
    prepare(h, binding, inp)
    launchers[names[0]]()
    torch.cuda.synchronize()
    if int(binding.failed):
        raise RuntimeError("reference launch set the failure word")
    current = outputs(h, binding, full)
    reference = {name: t.float().clone() for name, t in current.items()}
    scale = {name: max(float(t.abs().max()), 1e-30) for name, t in reference.items()}
    reference_argmax = reference["logits"].argmax(1)
    rel = torch.zeros(total, len(current), dtype=torch.float32, device=h.dev)
    finite = torch.ones(total, len(current), dtype=torch.int32, device=h.dev)
    word = slice(d.FAILURE_WORD, d.FAILURE_WORD + 1)
    for i, name in enumerate(order):
        ev[i][0].record()
        prepare(h, binding, inp)
        ev[i][1].record()
        launchers[name]()
        ev[i][2].record()
        failed[i : i + 1].copy_(
            binding.native[word]
        )  # before the next launch_begin erases it
        for j, (buffer, t) in enumerate(current.items()):
            value = t.float()
            rel[i, j] = (value - reference[buffer]).abs().max() / scale[buffer]
            finite[i, j] = torch.isfinite(value).all().to(torch.int32)
        agree[i] = (
            (current["logits"].float().argmax(1) == reference_argmax)
            .all()
            .to(torch.int32)
        )
    torch.cuda.synchronize()
    out = {
        "full_state_checked": full,
        "reference_finite": {
            buffer: bool(torch.isfinite(t).all()) for buffer, t in reference.items()
        },
        "failed_any_including_warmup": int((failed != 0).sum()),
    }
    if full:
        out["read_only_pools_unchanged"] = all(
            torch.equal(x, y) for x, y in zip(before, read_only_pools(h))
        )
    for name in names:
        rows = [i for i, arm in enumerate(order) if arm == name and i >= warmup]
        kernel_ms = [ev[i][1].elapsed_time(ev[i][2]) for i in rows]
        prepare_ms = [ev[i][0].elapsed_time(ev[i][1]) for i in rows]
        out[name] = {
            "kernel_ms": kernel_ms,
            "prepare_ms": prepare_ms,
            "kernel_stats": stats(kernel_ms),
            "prepare_stats": stats(prepare_ms),
            "failed_launches": int((failed[rows] != 0).sum()),
            "argmax_disagreements": int((agree[rows] == 0).sum()),
            "max_rel_diff_vs_first": {
                buffer: float(rel[rows, j].max()) for j, buffer in enumerate(current)
            },
            "nonfinite_launches": {
                buffer: int((finite[rows, j] == 0).sum())
                for j, buffer in enumerate(current)
            },
        }
    return out


def gate(results: dict) -> list[str]:
    """Reasons the run's checks do not pass; empty means they pass. Top-token flips between launches can happen
    from atomic summation order alone (both arms), so read them together with max_rel_diff_vs_first.
    """
    reasons = []
    for context, entry in results.items():
        for mode, kinds in entry["modes"].items():
            if "error" in kinds:
                reasons.append(f"{context}/{mode}: missing ({kinds['error']})")
                continue
            for kind, block in kinds.items():
                where = f"{context}/{mode}/{kind}"
                if block["failed_any_including_warmup"]:
                    reasons.append(f"{where}: failure word set")
                if not all(block["reference_finite"].values()):
                    reasons.append(f"{where}: non-finite reference output")
                if block.get("read_only_pools_unchanged") is False:
                    reasons.append(f"{where}: read-only pools changed")
                for arm, values in block.items():
                    if not isinstance(values, dict) or "kernel_ms" not in values:
                        continue
                    if values["argmax_disagreements"]:
                        reasons.append(
                            f"{where}/{arm}: top token changed in {values['argmax_disagreements']} launches"
                        )
                    for buffer, count in values["nonfinite_launches"].items():
                        if count:
                            reasons.append(
                                f"{where}/{arm}: {buffer} non-finite in {count} launches"
                            )
                    for buffer, value in values["max_rel_diff_vs_first"].items():
                        if value > STATE_RTOL:
                            reasons.append(
                                f"{where}/{arm}: {buffer} rel diff {value:.3g} > {STATE_RTOL}"
                            )
    return reasons


def verdict(reasons: list[str]) -> str:
    missing = [r for r in reasons if ": missing (" in r]
    return (
        "passed"
        if not reasons
        else "incomplete" if len(missing) == len(reasons) else "failed"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--process",
        type=int,
        required=True,
        help="process index, used in the output file name",
    )
    parser.add_argument("--contexts", default="1024,8192,32768")
    parser.add_argument(
        "--n", type=int, default=200, help="timed launches per arm per batch"
    )
    parser.add_argument(
        "--n-check", type=int, default=20, help="untimed validation launches per arm"
    )
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    model = env_path("MODEL_DIR")
    prompts = env_path("PROMPTS_DIR")
    a1_dir = env_path("A1_OBJECT_DIR")
    results_dir = Path(os.environ.get("RESULTS_DIR", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    contexts = [int(c) for c in args.contexts.split(",")]
    report = {
        "schema_version": 1,
        "process": args.process,
        "rows": ROWS,
        "contexts": contexts,
        "n_per_arm": args.n,
        "n_check_per_arm": args.n_check,
        "warmup": args.warmup,
        "state_rtol": STATE_RTOL,
        "device": torch.cuda.get_device_name(0),
    }
    started = time.time()
    h = Harness(str(model), "m8", max(40000, max(contexts) + 2 * ROWS))
    report["load_seconds"] = round(time.time() - started, 1)
    flydsl, binding = h.eng.builds["m8"]
    a1 = Megakernel(a1_dir)
    report["kernels"] = {"flydsl": dict(flydsl.meta), "a1": dict(a1.meta)}
    ids, passes = prompt_tokens(str(model), prompts, max(contexts) + ROWS)
    report["prompt_passes"] = passes
    cooperative = {
        "flydsl": lambda: flydsl.launch(binding.table),
        "a1": lambda: a1.launch(binding.table),
    }

    report["results"] = {}
    position = 0
    h.fresh()
    for context in contexts:
        t0 = time.time()
        for base in range(
            position, context, ROWS
        ):  # commit real tokens up to `context` (FlyDSL engine)
            h.launch(ids[base : base + ROWS], base)
        position = context
        prefill_s = round(time.time() - t0, 1)
        next_ids = ids[context : context + ROWS]
        modes = {
            "cooperative": {
                "timing": batch(
                    h,
                    binding,
                    cooperative,
                    next_ids,
                    context,
                    args.n,
                    args.warmup,
                    full=False,
                ),
                "validation": batch(
                    h,
                    binding,
                    cooperative,
                    next_ids,
                    context,
                    args.n_check,
                    2,
                    full=True,
                ),
            }
        }
        try:
            prepare(h, binding, inputs(h, next_ids, context))
            torch.cuda.synchronize()
            graphs = {}
            for name, kernel in (("flydsl", flydsl), ("a1", a1)):
                graphs[name] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graphs[name]):
                    kernel.launch(binding.table)
            replay = {name: g.replay for name, g in graphs.items()}
            modes["graph"] = {
                "timing": batch(
                    h,
                    binding,
                    replay,
                    next_ids,
                    context,
                    args.n,
                    args.warmup,
                    full=False,
                ),
                "validation": batch(
                    h, binding, replay, next_ids, context, args.n_check, 2, full=True
                ),
            }
        except Exception as error:  # reported, and the gate marks the run incomplete
            modes["graph"] = {"error": f"{type(error).__name__}: {error}"}
        entry = {"prefill_seconds": prefill_s, "modes": modes}
        summary = {"context": context, "prefill_seconds": prefill_s}
        for mode, kinds in modes.items():
            timing = kinds.get("timing")
            if timing:
                f_ms, a_ms = (
                    timing["flydsl"]["kernel_stats"]["median"],
                    timing["a1"]["kernel_stats"]["median"],
                )
                entry[f"{mode}_median_ratio_a1_over_flydsl"] = a_ms / f_ms
                summary.update(
                    {
                        f"{mode}_flydsl_ms": f_ms,
                        f"{mode}_a1_ms": a_ms,
                        f"{mode}_a1_over_flydsl": a_ms / f_ms,
                    }
                )
        report["results"][str(context)] = entry
        print(json.dumps(summary), flush=True)
    reasons = gate(report["results"])
    report["gate"] = {"verdict": verdict(reasons), "reasons": reasons}
    report["elapsed_seconds"] = round(time.time() - started, 1)
    (results_dir / f"timing-p{args.process}.json").write_text(
        json.dumps(report, indent=1) + "\n"
    )
    print(json.dumps({"gate": report["gate"]}), flush=True)
    return {"passed": 0, "incomplete": 2}.get(report["gate"]["verdict"], 1)


if __name__ == "__main__":
    sys.exit(main())

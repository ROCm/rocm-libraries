# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""End-to-end check of the ``hipdnn`` rms_norm / fused_add_rms_norm IR providers.

usage:
  python norm_ir_e2e.py cli --model M
      Parses the --ir-op-priority forms with vLLM's own CLI parser, prints the
      resulting IrOpPriorityConfig and the priority in effect after vLLM adds
      the platform defaults (needs the plugin installed: vLLM loads it then).
  python norm_ir_e2e.py run --model M --mode default|vllm_c|hipdnn --out X.json
                         [--ref default.json] [--enforce-eager]
      One LLM in this process (the V1 engine runs in-process so the plugin's
      counters are readable here). Records engine start-up time, greedy tokens,
      throughput, the IR priority in effect and the plugin counters. With
      --ref, also feeds each reference completion back as a prompt and counts
      the positions where the reference token is this run's top-1.
  python norm_ir_e2e.py compare REF.json RUN.json [--min-top1 0.99]
      Exits non-zero unless greedy outputs are identical, or the top-1
      agreement is at least --min-top1; for hipdnn runs rms_norm must also have
      been routed to hipDNN.
"""

import argparse
import json
import os
import sys
import time

PROMPTS = [
    "The capital of France is",
    "Write one sentence about GPUs:",
    "1, 2, 3, 4,",
    "In a distant galaxy, a small robot",
    "def fibonacci(n):",
    "The three primary colors are",
    "Translate to German: Good morning, how are you?",
    "Summarize the plot of Hamlet in two sentences.",
]
MAX_TOKENS = 64
THROUGHPUT_PROMPTS = 64
THROUGHPUT_TOKENS = 128

PRIORITY_JSON = {
    "vllm_c": {"rms_norm": ["vllm_c", "native"], "fused_add_rms_norm": ["vllm_c", "native"]},
    "hipdnn": {
        "rms_norm": ["hipdnn", "vllm_c", "native"],
        "fused_add_rms_norm": ["hipdnn", "vllm_c", "native"],
    },
}


def cmd_cli(args) -> int:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    forms = {
        "json": ["--ir-op-priority", json.dumps(PRIORITY_JSON["hipdnn"])],
        "json_comma_strings": [
            "--ir-op-priority",
            '{"rms_norm":"hipdnn,vllm_c,native","fused_add_rms_norm":"hipdnn,vllm_c,native"}',
        ],
        "dotted": [
            "--ir-op-priority.rms_norm",
            "hipdnn,vllm_c,native",
            "--ir-op-priority.fused_add_rms_norm",
            "hipdnn,vllm_c,native",
        ],
    }
    ok = True
    for name, argv in forms.items():
        parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
        ns = parser.parse_args(["--model", args.model, *argv])
        engine_args = EngineArgs.from_cli_args(ns)
        prio = engine_args.ir_op_priority
        got = {"rms_norm": list(prio.rms_norm), "fused_add_rms_norm": list(prio.fused_add_rms_norm)}
        good = got == PRIORITY_JSON["hipdnn"]
        ok &= good
        print(f"CLI {name}: {' '.join(argv)!s} -> {got} {'OK' if good else 'MISMATCH'}")
    # The priority in effect after vLLM appends the platform defaults.
    final = engine_args.create_engine_config().kernel_config.ir_op_priority
    print(f"CLI final rms_norm={final.rms_norm} fused_add_rms_norm={final.fused_add_rms_norm}")
    ok &= final.rms_norm[0] == "hipdnn" and final.fused_add_rms_norm[0] == "hipdnn"
    return 0 if ok else 1


def _counters():
    from hipdnn_vllm import stats

    snap = stats.snapshot()
    mod = sys.modules.get("hipdnn_vllm._C")
    snap["norm_prebuild"] = mod.norm_prebuild_totals() if mod is not None else None
    return snap


def _argmax(llm, outputs) -> list[list[int]]:
    """Feed each prompt + completion back as one prompt (teacher forcing) and
    return this run's top-1 token at every completion position."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    forced = [TokensPrompt(prompt_token_ids=r["prompt_ids"] + r["ids"]) for r in outputs]
    tf = llm.generate(forced, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1))
    out = []
    for r, o in zip(outputs, tf):
        start = len(r["prompt_ids"])
        row = []
        for pos in range(len(r["ids"])):
            entry = o.prompt_logprobs[start + pos]
            row.append(next(tok for tok, lp in entry.items() if lp.rank == 1))
        out.append(row)
    return out


def _agreement(a: list[list[int]], b: list[list[int]]) -> dict:
    total = sum(len(r) for r in a)
    agree = sum(int(x == y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))
    return {"agree": agree, "total": total, "ratio": agree / max(total, 1)}


def cmd_run(args) -> int:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import logging

    import torch
    from vllm import LLM, SamplingParams

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("hipdnn_vllm").setLevel(logging.INFO)

    kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
        seed=0,
    )
    if args.mode != "default":
        kwargs["ir_op_priority"] = PRIORITY_JSON[args.mode]

    t0 = time.perf_counter()
    llm = LLM(**kwargs)
    init_s = time.perf_counter() - t0
    vllm_config = llm.llm_engine.vllm_config
    prio = vllm_config.kernel_config.ir_op_priority
    cc = vllm_config.compilation_config

    greedy = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    outs = llm.generate(PROMPTS, greedy)
    result = {
        "mode": args.mode,
        "enforce_eager": args.enforce_eager,
        "dtype": str(vllm_config.model_config.dtype),
        "device": torch.cuda.get_device_name(0),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "init_s": init_s,
        "ir_op_priority": {
            "rms_norm": list(prio.rms_norm),
            "fused_add_rms_norm": list(prio.fused_add_rms_norm),
        },
        "compilation": {
            "mode": str(cc.mode),
            "cudagraph_mode": str(cc.cudagraph_mode),
            "cudagraph_capture_sizes": list(cc.cudagraph_capture_sizes or []),
        },
        "outputs": [
            {
                "prompt": p,
                "prompt_ids": list(o.prompt_token_ids),
                "ids": list(o.outputs[0].token_ids),
                "text": o.outputs[0].text,
            }
            for p, o in zip(PROMPTS, outs)
        ],
    }

    # Top-1 on identical inputs: every run scores the reference completions by
    # teacher forcing; agreement compares this run's top-1 with the reference
    # run's top-1 at each position. top1_self (prefill top-1 vs this run's own
    # decoded tokens) is the kernel-path noise floor, for scale.
    result["argmax_self"] = _argmax(llm, result["outputs"])
    result["top1_self"] = _agreement(result["argmax_self"], [r["ids"] for r in result["outputs"]])
    if args.ref:
        with open(args.ref, encoding="utf-8") as f:
            ref = json.load(f)
        result["argmax_ref"] = _argmax(llm, ref["outputs"])
        result["top1_vs_ref"] = _agreement(result["argmax_ref"], ref["argmax_self"])

    # Throughput: fixed batch, fixed output length.
    tp_prompts = [PROMPTS[i % len(PROMPTS)] + f" ({i})" for i in range(THROUGHPUT_PROMPTS)]
    tp = SamplingParams(temperature=0.0, max_tokens=THROUGHPUT_TOKENS, ignore_eos=True)
    llm.generate(tp_prompts[:8], tp)  # warm-up
    best = None
    for _ in range(3):
        torch.cuda.synchronize()
        t = time.perf_counter()
        res = llm.generate(tp_prompts, tp)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        n = sum(len(o.outputs[0].token_ids) for o in res)
        rate = n / dt
        best = rate if best is None else max(best, rate)
    result["throughput_tok_s"] = best

    result["hipdnn"] = _counters() if "hipdnn_vllm" in sys.modules else None
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(
        f"RUN mode={args.mode} eager={args.enforce_eager} init_s={init_s:.1f} "
        f"tok/s={best:.1f} prio={result['ir_op_priority']}"
    )
    print(f"TOP1 self: {result['top1_self']}")
    if result.get("top1_vs_ref"):
        print(f"TOP1 vs ref: {result['top1_vs_ref']}")
    del llm
    return 0


def _routed(run) -> dict:
    out = {}
    for r in (run.get("hipdnn") or {}).get("counters", []):
        key = f"{r['op']}:{r['route']}:{r['reason'] or '-'}:{r['phase']}"
        out[key] = out.get(key, 0) + r["count"]
    return out


def cmd_compare(args) -> int:
    with open(args.ref, encoding="utf-8") as f:
        ref = json.load(f)
    with open(args.run, encoding="utf-8") as f:
        run = json.load(f)
    same = sum(int(a["ids"] == b["ids"]) for a, b in zip(ref["outputs"], run["outputs"]))
    n = len(ref["outputs"])
    print(f"greedy identical: {same}/{n}")
    for a, b in zip(ref["outputs"], run["outputs"]):
        if a["ids"] != b["ids"]:
            k = next((i for i, (u, v) in enumerate(zip(a["ids"], b["ids"])) if u != v), None)
            print(f"  DIFF {a['prompt']!r} first divergence at token {k}")
            print(f"    ref: {a['text']!r}")
            print(f"    run: {b['text']!r}")
    # Both runs' top-1 on the same teacher-forced corpus: the corpus owner's
    # own scores (argmax_self) or, when both scored a third run's corpus
    # (--ref), those (argmax_ref).
    top1 = None
    if "argmax_ref" in run:
        other = ref.get("argmax_ref") or ref["argmax_self"]
        top1 = _agreement(run["argmax_ref"], other)
        print(f"top-1 agreement on identical inputs: {top1['agree']}/{top1['total']} = {top1['ratio']:.4f}")
        top1 = top1["ratio"]
    print(
        "decode-vs-prefill self-consistency (noise floor): "
        f"ref={ref['top1_self']['ratio']:.4f} run={run['top1_self']['ratio']:.4f}"
    )
    ratio = run["throughput_tok_s"] / ref["throughput_tok_s"]
    print(
        f"throughput ratio run/ref: {ratio:.3f}; start-up ratio run/ref: "
        f"{run['init_s'] / ref['init_s']:.2f} (+{run['init_s'] - ref['init_s']:.1f} s)"
    )
    routed = _routed(run)
    for k, v in sorted(routed.items()):
        print(f"  counter {k} = {v}")
    pre = (run.get("hipdnn") or {}).get("norm_prebuild")
    rt = (run.get("hipdnn") or {}).get("runtime")
    if pre:
        print(
            f"norm exact-row prebuild: {pre['signatures']} signatures, {pre['plans']} plans, "
            f"{pre['declines']} declines, {pre['ns'] / 1e9:.1f} s"
        )
    if rt:
        print(
            f"plan builds: {rt['plan_builds']} ({rt['plan_build_declines']} declined), "
            f"{rt['plan_build_ns'] / 1e9:.1f} s total, cache size {rt['plan_cache_size']}"
        )
    ok = same == n or (top1 is not None and top1 >= args.min_top1)
    if run["mode"] == "hipdnn":
        hip = sum(v for k, v in routed.items() if k.startswith("rms_norm:hipdnn"))
        if hip == 0:
            print("FAIL: rms_norm was never routed to hipDNN")
            ok = False
    print("COMPARE_OK" if ok else "COMPARE_FAIL")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    cli = sub.add_parser("cli")
    cli.add_argument("--model", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--mode", choices=["default", "vllm_c", "hipdnn"], required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--ref")
    r.add_argument("--enforce-eager", action="store_true")
    r.add_argument("--max-model-len", type=int, default=2048)
    r.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    r.add_argument("--dtype", default="auto")
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("run")
    c.add_argument("--min-top1", type=float, default=0.99)
    args = p.parse_args()
    return {"cli": cmd_cli, "run": cmd_run, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

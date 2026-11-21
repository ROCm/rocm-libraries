# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""End-to-end check for the MMEncoderAttention override: a vision
language model (Qwen2.5-VL) answers image prompts with the plugin's
encoder_attention enabled. Its greedy output tokens are compared with a run of
the same vLLM build with the plugin disabled, and the coverage report must show
where encoder_attention ran.

usage: python vlm_encoder_attention.py --model <path> --report-dir <dir>
       [--enforce-eager] [--max-tokens N] [--json-out f] [--require-hipdnn]
       [--ref-runs N] [--tie-margin NATS]
       [--teacher-forced [--min-agreement F] [--alt-vit-backend NAME]]

The plugin must be installed (vLLM loads it through the vllm.general_plugins
entry point in every process). The reference run is a child process with
HIPDNN_VLLM_DISABLE=1, repeated --ref-runs times to show whether the reference
is itself deterministic.

Greedy tokens must match exactly. With --tie-margin > 0, a prompt may instead
diverge at a near tie: everything before the divergence matches, and at that
step the reference ranks the plugin's token second with a log-probability
within the margin of its own choice. This is a logprob comparison,
for when a different (but correct) attention kernel shifts the vision features
by rounding noise. The margin at every divergence is reported either way.

Exits non-zero on a disallowed mismatch, on no encoder_attention calls, or
(with --require-hipdnn) when no call ran on hipDNN.

Two prompts are used. A 448x448 image gives uniform images and windows (one
batched graph per call). A 504x364 image gives ragged edge windows (one launch
per window, plans keyed by length).

--teacher-forced applies a teacher-forced top-1 agreement criterion instead of
exact greedy tokens.
A plugin-disabled base run generates greedy completions for twelve image
prompts (uniform and ragged sizes). Then a plugin-disabled control run and the
plugin run each re-score base's prompts plus completions with prompt
logprobs, all in separate processes. Each run's top-1 token at every
completion position is compared with base's own re-scored top-1. The plugin
must agree on at least --min-agreement of the positions, and no logprob may be
NaN. The control's agreement is the run-to-run noise floor, which is reported
next to the plugin's. --alt-vit-backend adds a plugin-disabled run whose ViT
attention uses another vLLM backend: the agreement expected from swapping one
correct attention kernel for another. With it, a plugin below --min-agreement
still passes if it agrees at least as well as that kernel swap.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

QUESTION = "Describe this image in one sentence."


def prompt_for(question: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n<|im_start|>assistant\n"
    )


PROMPT = prompt_for(QUESTION)
IMAGE_SIZES = [(448, 448), (504, 364)]
TOP_LOGPROBS = 2
# Teacher-forced cases: (image size, question). Sizes mix uniform windows
# (multiples of 112) with ragged edge windows.
TF_QUESTIONS = [
    QUESTION,
    "List every object you can see in this image and say where it is.",
    "Which colors dominate this picture? Answer in two sentences.",
    "Describe this image in detail: the objects, their colors, positions and sizes.",
]
TF_CASES = [
    ((448, 448), 0), ((504, 364), 1), ((336, 336), 2), ((392, 560), 3),
    ((616, 448), 1), ((280, 420), 2), ((448, 308), 3), ((532, 532), 1),
    ((448, 448), 3), ((504, 364), 3), ((560, 392), 1), ((364, 504), 3),
]  # fmt: skip


def make_image(width: int, height: int):
    """A deterministic synthetic scene (no network or dataset needed)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (135, 206, 235))  # sky
    d = ImageDraw.Draw(img)
    d.rectangle([0, height * 2 // 3, width, height], fill=(34, 139, 34))  # grass
    d.ellipse([width - 120, 20, width - 40, 100], fill=(255, 215, 0))  # sun
    d.rectangle([60, height // 2 - 20, 200, height * 2 // 3], fill=(178, 34, 34))  # house
    d.polygon([(50, height // 2 - 20), (130, height // 2 - 90), (210, height // 2 - 20)], fill=(120, 60, 20))
    d.rectangle([115, height * 2 // 3 - 50, 145, height * 2 // 3], fill=(80, 40, 10))  # door
    return img


def generate(model: str, enforce_eager: bool, max_tokens: int, flush_wait: float) -> list[dict]:
    """Per prompt: generated token ids and, per step, the top logprobs as
    [[token, logprob], ...] in rank order."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=enforce_eager,
        gpu_memory_utilization=0.6,
        seed=0,
    )
    requests = [
        {"prompt": PROMPT, "multi_modal_data": {"image": make_image(w, h)}} for (w, h) in IMAGE_SIZES
    ]
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=TOP_LOGPROBS)
    outs = llm.generate(requests, params)
    result = []
    for o in outs:
        steps = []
        for step in o.outputs[0].logprobs or []:
            ranked = sorted(step.items(), key=lambda kv: kv[1].rank)
            steps.append([[tok, lp.logprob] for tok, lp in ranked])
        result.append({"tokens": list(o.outputs[0].token_ids), "top": steps, "text": o.outputs[0].text})
    if flush_wait > 0:
        # The engine process writes its coverage report on the report
        # interval; give it one interval after the last call before shutdown.
        time.sleep(flush_wait)
    del llm
    for size, r in zip(IMAGE_SIZES, result):
        print(f"GEN {size}: {r['text']!r}", file=sys.stderr)
    return result


def compare(got: dict, ref: dict, tie_margin: float) -> dict:
    a, b = got["tokens"], ref["tokens"]
    first = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if first is None and len(a) == len(b):
        return {"verdict": "exact"}
    if first is None:
        first = min(len(a), len(b))
    out = {"verdict": "mismatch", "first_divergence": first, "prefix_tokens": first}
    if first < len(a) and first < len(b) and first < len(ref["top"]):
        ref_top = {tok: lp for tok, lp in ref["top"][first]}
        out["ref_top"] = ref["top"][first]
        out["got_top"] = got["top"][first] if first < len(got["top"]) else None
        if a[first] in ref_top:
            margin = ref_top[b[first]] - ref_top[a[first]]
            out["ref_margin_nats"] = margin
            if tie_margin > 0 and margin <= tie_margin:
                out["verdict"] = "near_tie"
    return out


def teacher_forced(model: str, enforce_eager: bool, max_tokens: int, reference: str | None,
                   flush_wait: float, vit_backend: str | None = None) -> dict:
    """Without ``reference`` (the base run): greedy-generate every TF case.
    Then score base's prompt + completion tokens with prompt logprobs and keep
    the top-2 [[token, logprob], ...] at every completion position."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    extra = {"mm_encoder_attn_backend": vit_backend} if vit_backend else {}
    llm = LLM(
        model=model,
        max_model_len=4096,
        max_num_seqs=4,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=enforce_eager,
        gpu_memory_utilization=0.6,
        seed=0,
        **extra,
    )
    cases = [(make_image(w, h), prompt_for(TF_QUESTIONS[q])) for (w, h), q in TF_CASES]
    result: dict = {}
    if reference is None:
        outs = llm.generate(
            [{"prompt": p, "multi_modal_data": {"image": img}} for img, p in cases],
            SamplingParams(temperature=0.0, max_tokens=max_tokens),
            use_tqdm=False,
        )
        result["gen_ids"] = [list(o.outputs[0].token_ids) for o in outs]
        result["texts"] = [o.outputs[0].text for o in outs]
    else:
        result["gen_ids"] = json.load(open(reference))["gen_ids"]
    tok = llm.get_tokenizer()
    # The unexpanded prompt (one <|image_pad|>) plus the completion; vLLM
    # expands the image placeholder, so completion positions are the tail.
    scored = llm.generate(
        [
            TokensPrompt(prompt_token_ids=tok.encode(p, add_special_tokens=False) + g,
                         multi_modal_data={"image": img})
            for (img, p), g in zip(cases, result["gen_ids"])
        ],
        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=TOP_LOGPROBS),
        use_tqdm=False,
    )
    top = []
    for o, g in zip(scored, result["gen_ids"]):
        ids = list(o.prompt_token_ids)
        assert ids[len(ids) - len(g):] == g, "scored prompt does not end with the completion"
        rows = []
        for pos in range(len(ids) - len(g), len(ids)):
            ranked = sorted((v.rank, t, v.logprob) for t, v in o.prompt_logprobs[pos].items()
                            if v.rank is not None and v.rank <= TOP_LOGPROBS)
            rows.append([[t, lp] for _, t, lp in ranked])
        top.append(rows)
    result["top"] = top
    if flush_wait > 0:
        time.sleep(flush_wait)
    del llm
    return result


def tf_agreement(base: dict, cand: dict) -> dict:
    """Top-1 agreement of ``cand`` with ``base`` over every completion
    position, plus base's top-2 margin at each disagreement."""
    match = total = 0
    nan = False
    disagreements = []
    for i, (b_rows, c_rows) in enumerate(zip(base["top"], cand["top"])):
        assert len(b_rows) == len(c_rows)
        for pos, (b, c) in enumerate(zip(b_rows, c_rows)):
            nan |= any(lp != lp for _, lp in b + c)
            total += 1
            if b[0][0] == c[0][0]:
                match += 1
                continue
            b_lp = dict((t, lp) for t, lp in b)
            disagreements.append({"case": i, "pos": pos, "base_top": b, "got_top": c,
                                  "base_margin_nats": b[0][1] - b_lp[c[0][0]] if c[0][0] in b_lp else None})
    return {"agreement": match / total if total else 0.0, "positions": total, "disagreements": disagreements,
            "nan": nan}


def routes_from(report_dir: str) -> tuple[list[str], dict, dict]:
    reports = sorted(glob.glob(os.path.join(report_dir, "report-*.json")))
    routes: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for path in reports:
        data = json.load(open(path))
        for r in data["counters"]:
            if r["op"] == "encoder_attention":
                routes[r["route"]] = routes.get(r["route"], 0) + r["count"]
                if r["route"] == "fallback":
                    reasons[r["reason"]] = reasons.get(r["reason"], 0) + r["count"]
    return reports, routes, reasons


def tf_main(args) -> int:
    os.makedirs(args.report_dir, exist_ok=True)
    for old in glob.glob(os.path.join(args.report_dir, "report-*.json")):
        os.remove(old)
    out_dir = os.path.dirname(os.path.abspath(args.json_out or os.path.join(args.report_dir, "x")))
    interval = os.environ.get("HIPDNN_VLLM_REPORT_INTERVAL", "5")

    def run(role: str, plugin: bool, reference: str | None, vit_backend: str | None = None) -> dict:
        path = os.path.join(out_dir, f"tf-{role}.json")
        cmd = [sys.executable, __file__, "--tf-child", role, "--model", args.model, "--report-dir", args.report_dir,
               "--max-tokens", str(args.max_tokens), "--json-out", path]
        if args.enforce_eager:
            cmd.append("--enforce-eager")
        if reference:
            cmd += ["--tf-reference", reference]
        if vit_backend:
            cmd += ["--alt-vit-backend", vit_backend]
        env = dict(os.environ)
        if plugin:
            env.update(HIPDNN_VLLM_REPORT_DIR=args.report_dir, HIPDNN_VLLM_REPORT_INTERVAL=interval)
            env.setdefault("HIPDNN_VLLM_OPS", "encoder_attention")
            env.pop("HIPDNN_VLLM_DISABLE", None)
        else:
            env.update(HIPDNN_VLLM_DISABLE="1", HIPDNN_VLLM_REPORT_DIR="")
        print(f"== {role} (plugin {'on' if plugin else 'off'}, ViT backend {vit_backend or 'default'})", flush=True)
        subprocess.run(cmd, env=env, check=True)
        return json.load(open(path))

    base = run("base", False, None)
    base_path = os.path.join(out_dir, "tf-base.json")
    control = run("control", False, base_path)
    plugin = run("plugin", True, base_path)
    summary = {
        "criterion": f"teacher-forced top-1 agreement >= {args.min_agreement}, no NaN",
        "base_texts": base["texts"],
        "control": tf_agreement(base, control),
        "plugin": tf_agreement(base, plugin),
    }
    if args.alt_vit_backend:
        # Kernel-swap noise floor: vLLM's own ViT attention through another backend.
        summary["alt_vit_backend"] = args.alt_vit_backend
        summary["alt"] = tf_agreement(base, run("alt", False, base_path, args.alt_vit_backend))
    summary["reports"], summary["encoder_attention_routes"], summary["encoder_attention_fallback_reasons"] = (
        routes_from(args.report_dir))
    print(json.dumps(summary, indent=2))
    if args.json_out:
        json.dump(summary, open(args.json_out, "w"), indent=2)
    failures = []
    for role in ("control", "plugin", "alt"):
        if role in summary and summary[role]["nan"]:
            failures.append(f"{role}: NaN logprob")
    plugin_ok = summary["plugin"]["agreement"] >= args.min_agreement
    if not plugin_ok and "alt" in summary:
        # Below the absolute bar but no worse than vLLM's own kernel swap: the
        # prompt set's near ties, not the plugin's kernel, set the floor.
        plugin_ok = summary["plugin"]["agreement"] >= summary["alt"]["agreement"]
    if not plugin_ok:
        alt = f", {args.alt_vit_backend} {summary['alt']['agreement']:.4f}" if "alt" in summary else ""
        failures.append(f"plugin top-1 agreement {summary['plugin']['agreement']:.4f} < {args.min_agreement} "
                        f"(control {summary['control']['agreement']:.4f}{alt})")
    if not summary["reports"]:
        failures.append("no coverage report written")
    if not sum(summary["encoder_attention_routes"].values()):
        failures.append("no encoder_attention calls recorded")
    if args.require_hipdnn and not summary["encoder_attention_routes"].get("hipdnn"):
        failures.append("encoder_attention never ran on hipDNN")
    for f in failures:
        print("FAIL:", f, file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--json-out")
    p.add_argument("--require-hipdnn", action="store_true")
    p.add_argument("--ref-runs", type=int, default=1,
                   help="run the plugin-disabled reference N times to show whether it is itself deterministic")
    p.add_argument("--tie-margin", type=float, default=0.0,
                   help="allow a divergence at a reference near tie within this many nats (0: exact only)")
    p.add_argument("--teacher-forced", action="store_true",
                   help="teacher-forced top-1 agreement against a plugin-disabled base and control")
    p.add_argument("--min-agreement", type=float, default=0.99)
    p.add_argument("--alt-vit-backend",
                   help="teacher-forced: also score with the plugin off and this mm_encoder_attn_backend")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--tf-child", help=argparse.SUPPRESS)
    p.add_argument("--tf-reference", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.child:
        print(json.dumps(generate(args.model, args.enforce_eager, args.max_tokens, 0.0)))
        return 0
    if args.tf_child:
        flush = float(os.environ.get("HIPDNN_VLLM_REPORT_INTERVAL", "5")) + 5.0 if args.tf_child == "plugin" else 0.0
        json.dump(teacher_forced(args.model, args.enforce_eager, args.max_tokens, args.tf_reference, flush,
                                 args.alt_vit_backend if args.tf_child == "alt" else None),
                  open(args.json_out, "w"))
        return 0
    if args.teacher_forced:
        return tf_main(args)

    os.environ["HIPDNN_VLLM_REPORT_DIR"] = args.report_dir
    os.environ.setdefault("HIPDNN_VLLM_REPORT_INTERVAL", "5")
    os.environ.setdefault("HIPDNN_VLLM_OPS", "encoder_attention")
    os.makedirs(args.report_dir, exist_ok=True)
    for old in glob.glob(os.path.join(args.report_dir, "report-*.json")):
        os.remove(old)

    flush_wait = float(os.environ["HIPDNN_VLLM_REPORT_INTERVAL"]) + 5.0
    got = generate(args.model, args.enforce_eager, args.max_tokens, flush_wait)

    cmd = [sys.executable, __file__, "--child", "--model", args.model, "--report-dir", args.report_dir,
           "--max-tokens", str(args.max_tokens)]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    env = dict(os.environ, HIPDNN_VLLM_DISABLE="1", HIPDNN_VLLM_REPORT_DIR="")
    refs = []
    for _ in range(max(1, args.ref_runs)):
        out = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, text=True, check=True)
        refs.append(json.loads(out.stdout.strip().splitlines()[-1]))
    ref = refs[0]
    ref_deterministic = all([r["tokens"] for r in rr] == [r["tokens"] for r in ref] for rr in refs)
    if not ref_deterministic:
        print(f"NOTE: the plugin-disabled reference itself differs across {len(refs)} runs", file=sys.stderr)

    failures = []
    comparisons = [compare(g, r, args.tie_margin) for g, r in zip(got, ref)]
    for i, c in enumerate(comparisons):
        print(f"prompt {i} {IMAGE_SIZES[i]}: {json.dumps(c)}")
        if c["verdict"] == "mismatch":
            failures.append(f"prompt {i} {IMAGE_SIZES[i]}: tokens differ from the default run at index "
                            f"{c['first_divergence']} (reference margin {c.get('ref_margin_nats')})")
    if all(c["verdict"] == "exact" for c in comparisons):
        print("greedy output tokens identical to the default (plugin disabled) run")

    reports, routes, reasons = routes_from(args.report_dir)
    runtime = [d["runtime"] for d in (json.load(open(p)) for p in reports) if d.get("runtime")]
    summary = {
        "reports": reports,
        "encoder_attention_routes": routes,
        "encoder_attention_fallback_reasons": reasons,
        "runtime": runtime,
        "comparisons": comparisons,
        "tokens_match": all(c["verdict"] == "exact" for c in comparisons),
        "ref_runs": len(refs),
        "ref_deterministic": ref_deterministic,
        "texts": [g["text"] for g in got],
        "ref_texts": [[r["text"] for r in rr] for rr in refs],
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        json.dump(summary, open(args.json_out, "w"), indent=2)

    if not reports:
        failures.append("no coverage report written")
    if not sum(routes.values()):
        failures.append("no encoder_attention calls recorded")
    if args.require_hipdnn and not routes.get("hipdnn"):
        failures.append("encoder_attention never ran on hipDNN")
    for f in failures:
        print("FAIL:", f, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

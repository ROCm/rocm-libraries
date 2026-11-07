# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""End-to-end check for the LayerNorm and Conv OOT overrides.

Runs vLLM offline generation twice in separate processes, once with the plugin
routing ``layer_norm`` / ``conv`` to hipDNN and once with the plugin disabled
(``HIPDNN_VLLM_DISABLE=1``, stock vLLM). Greedy tokens must match, and the
plugin run's coverage report must show hipDNN routes for the expected ops.

usage: python offline_lnconv.py --model <path> --report-dir <dir>
       [--image] [--expect layer_norm,conv_fwd] [--enforce-eager]
       [--max-model-len N] [--json-out file]

``--image`` sends one synthetic image per prompt (vision-language models).
Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

PROMPTS = [
    "Describe the picture in one sentence.",
    "What colors are in this image?",
]
TEXT_PROMPTS = [
    "The capital of France is",
    "1, 2, 3, 4,",
    "In a distant galaxy, a small robot",
]


def _image():
    import numpy as np
    from PIL import Image

    # Deterministic, non-trivial content: colour gradients plus a checkerboard.
    h, w = 384, 512
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack(
        [(xx * 255 // w), (yy * 255 // h), (((xx // 32 + yy // 32) % 2) * 255)], axis=-1
    ).astype(np.uint8)
    return Image.fromarray(img)


def _requests(model: str, image: bool):
    if not image:
        return TEXT_PROMPTS
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model)
    reqs = []
    for q in PROMPTS:
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
        reqs.append({"prompt": prompt, "multi_modal_data": {"image": _image()}})
    return reqs


def generate(model: str, image: bool, enforce_eager: bool, max_model_len: int) -> list[list[int]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.5,
        limit_mm_per_prompt={"image": 1} if image else None,
        seed=0,
    )
    outs = llm.generate(_requests(model, image), SamplingParams(temperature=0.0, max_tokens=32))
    for o in outs:
        print("GEN:", repr(o.outputs[0].text), flush=True)
    return [list(o.outputs[0].token_ids) for o in outs]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--image", action="store_true")
    p.add_argument("--expect", default="layer_norm,conv_fwd")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--json-out")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.child:
        print("TOKENS:" + json.dumps(generate(args.model, args.image, args.enforce_eager, args.max_model_len)))
        return 0

    os.makedirs(args.report_dir, exist_ok=True)
    for old in glob.glob(os.path.join(args.report_dir, "report-*.json")):
        os.remove(old)

    def run(env_extra: dict[str, str]) -> list[list[int]]:
        cmd = [sys.executable, __file__, "--child", "--model", args.model, "--report-dir", args.report_dir,
               "--max-model-len", str(args.max_model_len)]
        if args.image:
            cmd.append("--image")
        if args.enforce_eager:
            cmd.append("--enforce-eager")
        env = dict(os.environ, **env_extra)
        out = subprocess.run(cmd, env=env, capture_output=True, text=True)
        sys.stdout.write("".join(line + "\n" for line in out.stdout.splitlines() if line.startswith("GEN:")))
        if out.returncode != 0:
            sys.stderr.write(out.stdout[-4000:] + out.stderr[-8000:])
            raise SystemExit(f"generation failed ({env_extra})")
        line = [ln for ln in out.stdout.splitlines() if ln.startswith("TOKENS:")][-1]
        return json.loads(line[len("TOKENS:"):])

    print("== hipdnn_vllm (layer_norm, conv)")
    ours = run({
        "HIPDNN_VLLM_OPS": "layer_norm,conv",
        "HIPDNN_VLLM_REPORT_DIR": args.report_dir,
        "HIPDNN_VLLM_REPORT_INTERVAL": "5",
    })
    print("== stock vLLM (HIPDNN_VLLM_DISABLE=1)")
    ref = run({"HIPDNN_VLLM_DISABLE": "1", "HIPDNN_VLLM_REPORT_DIR": ""})

    failures = []
    if ours != ref:
        failures.append(f"greedy tokens differ:\n  hipdnn {ours}\n  stock  {ref}")
    else:
        print("greedy tokens identical to stock vLLM")

    routes: dict[str, dict[str, int]] = {}
    reports = sorted(glob.glob(os.path.join(args.report_dir, "report-*.json")))
    for path in reports:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for r in data["counters"]:
            key = f"{r['route']}:{r['reason']}" if r["reason"] else r["route"]
            key += f"@{r['phase']}"
            routes.setdefault(r["op"], {})
            routes[r["op"]][key] = routes[r["op"]].get(key, 0) + r["count"]
    summary = {"reports": reports, "routes": routes, "tokens_match": ours == ref}
    print(json.dumps(summary, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    for op in [o for o in args.expect.split(",") if o]:
        hits = sum(n for k, n in routes.get(op, {}).items() if k.startswith("hipdnn"))
        if not hits:
            failures.append(f"no hipDNN route recorded for {op}")
    for f in failures:
        print("FAIL:", f, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

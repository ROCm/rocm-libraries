#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Realistic SDPA shapes as the q.<parameter> CSV `hipdnn_corpus_gen --model-shapes` reads.

Attention geometry (query heads, KV heads, head dim) comes from the published configurations of
the models below; batch and sequence lengths are the serving and training settings those models
are typically run at. Each row is one problem, named <model>/<phase>/..., so a result can be
split by model and phase.

Phases:
  prefill   inference forward over the prompt (causal, square)
  decode    one query token against the KV cache (causal, bottom-right)
  append    chunked prefill / speculative decoding: a block of queries against a longer cache
  encoder   bidirectional forward (no mask)
  train     training forward (generate_stats, the backward's softmax statistics) and the
            matching backward, square and causal for decoders, unmasked for encoders

usage: sdpa_realistic_shapes.py <out prefix>   (writes <prefix>sdpa_fwd.csv and <prefix>sdpa_bwd.csv;
each file carries only the parameters its operation declares)
"""
import csv
import sys

# name: (heads, kv_heads, head_dim, kind, dtype); kind decoder|encoder|diffusion
MODELS = {
    "llama3.1-8b": (32, 8, 128, "decoder", "bf16"),
    "llama3.1-70b": (64, 8, 128, "decoder", "bf16"),
    "llama3.1-405b": (128, 8, 128, "decoder", "bf16"),
    "llama2-7b": (32, 32, 128, "decoder", "fp16"),
    "llama2-13b": (40, 40, 128, "decoder", "fp16"),
    "mistral-7b": (32, 8, 128, "decoder", "bf16"),
    "qwen2.5-7b": (28, 4, 128, "decoder", "bf16"),
    "qwen2.5-32b": (40, 8, 128, "decoder", "bf16"),
    "qwen2.5-72b": (64, 8, 128, "decoder", "bf16"),
    "gemma2-9b": (16, 8, 256, "decoder", "bf16"),
    "gemma2-27b": (32, 16, 128, "decoder", "bf16"),
    "phi3-mini": (32, 32, 96, "decoder", "bf16"),
    "gpt-oss-120b": (64, 8, 64, "decoder", "bf16"),
    "falcon-7b": (71, 1, 64, "decoder", "bf16"),
    "bert-base": (12, 12, 64, "encoder", "fp16"),
    "bert-large": (16, 16, 64, "encoder", "fp16"),
    "vit-l14": (16, 16, 64, "encoder", "fp16"),
    "whisper-large-enc": (20, 20, 64, "encoder", "fp16"),
    "sdxl-unet": (20, 20, 64, "diffusion", "fp16"),
    "sd3-mmdit": (24, 24, 64, "diffusion", "bf16"),
    "flux-dit": (24, 24, 128, "diffusion", "bf16"),
}
ENCODER_SEQ = {
    "bert-base": [128, 512],
    "bert-large": [128, 512],
    "vit-l14": [257],
    "whisper-large-enc": [1500],
}
DIFFUSION_SEQ = {"sdxl-unet": [1024, 4096], "sd3-mmdit": [4250], "flux-dit": [4608]}
# Training: every decoder at the usual pre-training/fine-tuning lengths (the largest models at
# smaller micro-batches), the encoders at theirs, and the diffusion backbones at their latents.
LARGE = {"llama3.1-70b", "llama3.1-405b", "qwen2.5-72b", "gpt-oss-120b", "qwen2.5-32b", "gemma2-27b"}
TRAIN = {
    m: ([2048, 4096, 8192], [1, 2] if m in LARGE else [1, 2, 4])
    for m, spec in MODELS.items()
    if spec[3] == "decoder"
}
TRAIN_ENCODER = {
    "bert-base": ([128, 512], [32, 64]),
    "bert-large": ([512], [16, 32]),
    "vit-l14": ([257], [64, 128]),
    "sdxl-unet": ([1024, 4096], [2, 4]),
    "flux-dit": ([4608], [1, 2]),
}

COMMON = ["q.batch", "q.heads", "q.heads_kv", "q.seqlen_q", "q.seqlen_k", "q.head_dim",
          "q.is_causal", "q.alignment"]
COLUMNS = {
    "sdpa_fwd": ["name", "op", *COMMON, "q.generate_stats", "q.dtype", "q.form"],
    "sdpa_bwd": ["name", "op", *COMMON, "q.dtype", "q.form"],
}


def rows():
    def row(name, op, b, h, hk, sq, sk, d, causal, align, stats, dtype):
        out = [name, op, b, h, hk, sq, sk, d, str(causal).lower(), align]
        if op == "sdpa_fwd":
            out.append(str(stats).lower())
        return op, out + [dtype, "plain"]

    for model, (h, hk, d, kind, dtype) in MODELS.items():
        if kind == "decoder":
            for b in (1, 4, 8):
                for s in (512, 2048, 8192) + ((32768,) if b == 1 else ()):
                    yield row(f"{model}/prefill/b{b}s{s}", "sdpa_fwd", b, h, hk, s, s, d, True,
                              "top_left", False, dtype)
            for b in (1, 8, 32, 64, 128):
                for k in (1024, 4096, 16384):
                    if b * k > 128 * 16384:
                        continue
                    yield row(f"{model}/decode/b{b}k{k}", "sdpa_fwd", b, h, hk, 1, k, d, True,
                              "bottom_right", False, dtype)
            for b in (1, 4):
                for q, k in ((128, 4096), (512, 8192), (2048, 16384), (4, 2048)):
                    yield row(f"{model}/append/b{b}q{q}k{k}", "sdpa_fwd", b, h, hk, q, k, d, True,
                              "bottom_right", False, dtype)
        elif kind == "encoder":
            for b in (8, 32, 64):
                for s in ENCODER_SEQ[model]:
                    yield row(f"{model}/encoder/b{b}s{s}", "sdpa_fwd", b, h, hk, s, s, d, False,
                              "top_left", False, dtype)
        else:
            for b in (1, 2, 4):
                for s in DIFFUSION_SEQ[model]:
                    yield row(f"{model}/diffusion/b{b}s{s}", "sdpa_fwd", b, h, hk, s, s, d, False,
                              "top_left", False, dtype)
    for model, (seqs, batches) in TRAIN.items():
        h, hk, d, _, dtype = MODELS[model]
        for b in batches:
            for s in seqs:
                for op, stats in (("sdpa_fwd", True), ("sdpa_bwd", None)):
                    yield row(f"{model}/train/b{b}s{s}", op, b, h, hk, s, s, d, True, "top_left",
                              stats, dtype)
    for model, (seqs, batches) in TRAIN_ENCODER.items():
        h, hk, d, _, dtype = MODELS[model]
        for b in batches:
            for s in seqs:
                for op, stats in (("sdpa_fwd", True), ("sdpa_bwd", None)):
                    yield row(f"{model}/train/b{b}s{s}", op, b, h, hk, s, s, d, False,
                              "top_left", stats, dtype)


def main(prefix):
    handles, writers, counts = {}, {}, {}
    try:
        for op, columns in COLUMNS.items():
            handles[op] = open(f"{prefix}{op}.csv", "w", newline="")
            writers[op] = csv.writer(handles[op])
            writers[op].writerow(columns)
            counts[op] = 0
        for op, r in rows():
            writers[op].writerow(r)
            counts[op] += 1
    finally:
        for handle in handles.values():
            handle.close()
    for op, n in counts.items():
        print(f"{n} rows -> {prefix}{op}.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])

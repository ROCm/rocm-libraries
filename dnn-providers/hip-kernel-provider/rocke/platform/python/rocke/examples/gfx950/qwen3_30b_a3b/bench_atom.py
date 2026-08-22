#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Measure ATOM decode-step latency and throughput — one configuration per run.

Run three times with different --config values and compare the results:

  python bench_atom.py --model <path> --config baseline
  python bench_atom.py --model <path> --config dsl_gemm
  python bench_atom.py --model <path> --config dsl_all

Why one-per-process
-------------------
CUDA contexts cannot be torn down and re-initialised in the same Python
process without multiprocessing.spawn, and the ATOM engine manages its own
worker sub-processes.  Running one config per invocation avoids any
cross-configuration kernel-cache or GPU-state contamination.

Timing methodology
------------------
For each repeat:
  1. Submit BATCH_SIZE prompts of INPUT_LEN tokens with max_tokens=OUTPUT_LEN.
  2. Record wall-clock start immediately before llm.generate().
  3. Record wall-clock end immediately after llm.generate() returns.
  4. Decompose total time into:
       TTFT  = prefill_ms   (measured once via max_tokens=1 call)
       Decode time = total_ms - prefill_ms
       Decode steps = OUTPUT_LEN - 1  (first output token is prefill)
       Decode step latency = decode_time / decode_steps
       Throughput = BATCH_SIZE * OUTPUT_LEN / total_seconds  (tokens/s)
       TPOT = decode_step_latency / BATCH_SIZE  (ms per output token per seq)

The first two repeats are discarded as warmup.  Mean (± stdev) over the
remaining repeats is reported.  Each repeat is an independent generate() call
starting from the same prompt so the KV cache length is fixed across repeats.

Reported metrics
----------------
  TTFT         Time-to-first-token (ms) — includes prefill + first decode step
  Decode step  Median wall-clock time for one decode step (µs)
  TPOT         Time-per-output-token per sequence (ms) — decode step / bs
  Throughput   Total output tokens / total wall-clock time (tok/s)

Measured results (Qwen3-30B-A3B config, random weights, bs=2, input=512,
output=200 tok, MI355X, level=3 CUDAGraph, kv_cache_dtype=bf16):

Methodology: bench_atom_sweep.py runs configs in interleaved round-robin order
(baseline -> dsl_gemm -> dsl_all, repeat) so each round experiences the same
GPU power/thermal state.  30 rounds, one subprocess per (round, config).

  Config              Step (us)
  baseline (rocBLAS)  4654 +/- 117
  dsl_gemm (DSL GEMM) 4605 +/- 131   (-1.1% vs baseline, within 1-sigma)
  dsl_all  (GEMM+ATT) 4631 +/- 141   (-0.5% vs baseline, within 1-sigma)

All three configs are statistically indistinguishable.  The ~120-140 us
stdev is itself larger than any kernel savings (~50-100 us GPU time saved).

NOTE — why serving speedup is smaller than kernel speedup:
The ATOM engine loop (IPC between main process and ModelRunner subprocess,
scheduler, sampler) contributes ~4200 us per decode step on top of ~200-300 us
of GPU kernel time.  DSL saves ~50-100 us of GPU time (< 1% of the total step),
which is below the serving-level noise floor (~120 us stdev).
The per-kernel benchmarks (scripts 01-07) show the true 1.28x GPU-level speedup.
To expose it end-to-end, the ATOM engine overhead must drop below ~500 us/step.

The DSL GEMM is registered as a torch.library custom op (atom_dsl::gemm) so
CUDAGraph capture records the HIP kernel launch on the current stream.  A plain
@torch._dynamo.disable function creates a graph break not captured in the
CUDAGraph, so the kernel would never actually run during decode.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time


WARMUP_REPEATS = 2  # full generate() calls to discard (full KV length, proper warmup)
MEASURE_REPEATS = 30  # generate() calls to time; mean is reported

_CONFIG_ENV = {
    "baseline": {},
    "dsl_gemm": {"ATOM_USE_DSL_GEMM": "1"},
    "dsl_all": {"ATOM_USE_DSL_GEMM": "1", "ATOM_USE_DSL_ATTENTION": "1"},
}
_CONFIG_LABEL = {
    "baseline": "Baseline     (AITER/rocBLAS defaults)",
    "dsl_gemm": "DSL GEMM     (ATOM_USE_DSL_GEMM=1)",
    "dsl_all": "DSL GEMM+ATT (ATOM_USE_DSL_GEMM=1 ATOM_USE_DSL_ATTENTION=1)",
}


def _make_prompt(tok, input_len: int) -> str:
    """Build a prompt that tokenises to exactly input_len tokens."""
    ids = tok.encode(" the" * (input_len + 10))[:input_len]
    return tok.decode(ids)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model path or name")
    parser.add_argument(
        "--config",
        choices=list(_CONFIG_ENV),
        default="baseline",
        help="Which backend to benchmark (default: baseline)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Concurrent sequences (default: 2 = A3B decode shape)",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=512,
        help="Prompt length in tokens (default: 512)",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=200,
        help="Output tokens to generate per sequence (default: 200)",
    )
    parser.add_argument("--kv-cache-dtype", dest="kv_cache_dtype", default="bf16")
    parser.add_argument("--max-model-len", dest="max_model_len", type=int, default=None)
    parser.add_argument("-tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument(
        "--level",
        type=int,
        default=3,
        help="ATOM compilation level: 0=eager, 3=full CUDAGraph (default: 3)",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path (defaults to --model; override when model "
        "directory lacks tokenizer files)",
    )
    parser.add_argument(
        "--single-shot",
        action="store_true",
        help="Warmup then emit exactly one timed rep as a machine-readable "
        "line: 'SINGLE_SHOT config total_ms step_us throughput'. "
        "Used by bench_atom_sweep.py to interleave configs fairly.",
    )
    args = parser.parse_args()

    # Apply env vars before any ATOM import so lazy env-var readers pick them up
    for k, v in _CONFIG_ENV[args.config].items():
        os.environ[k] = v

    from atom.model_engine.arg_utils import EngineArgs
    from atom import SamplingParams
    from transformers import AutoTokenizer

    print("=" * 68)
    print("ATOM decode latency / throughput benchmark")
    print("=" * 68)
    print(f"  Model:       {args.model}")
    print(f"  Config:      {_CONFIG_LABEL[args.config]}")
    print(f"  Batch size:  {args.batch_size} sequences")
    print(f"  Input len:   {args.input_len} tokens")
    print(f"  Output len:  {args.output_len} tokens")
    print(f"  Warmup:      {WARMUP_REPEATS} full generate() call(s) discarded")
    print(f"  Measure:     {MEASURE_REPEATS} generate() calls → median")
    print()

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

    engine_args = EngineArgs(
        model=args.model,
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        level=args.level,
    )
    llm = engine_args.create_engine(tokenizer=tok)

    bs = args.batch_size
    prompts = [_make_prompt(tok, args.input_len)] * bs

    # -----------------------------------------------------------------------
    # TTFT: time to first token
    # One generate() with max_tokens=1 — this is prefill + one decode step.
    # We call it once for warmup, once timed.
    # -----------------------------------------------------------------------
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    llm.generate(prompts, sp1)  # compile / graph-capture warmup
    t0 = time.perf_counter()
    llm.generate(prompts, sp1)
    ttft_ms = (time.perf_counter() - t0) * 1000
    print(
        f"  TTFT (prefill {args.input_len} tok × {bs} seqs, 1 output tok): "
        f"{ttft_ms:.1f} ms"
    )

    # -----------------------------------------------------------------------
    # Decode step latency and throughput
    # Each repeat: fresh generate() from the same fixed prompts → same KV length
    # across all repeats, so results are comparable.
    # -----------------------------------------------------------------------
    sp = SamplingParams(temperature=0.0, max_tokens=args.output_len, ignore_eos=True)
    decode_steps = args.output_len - 1  # first output token included in TTFT

    # Warmup repeat(s) — full generate with output_len tokens
    for _ in range(WARMUP_REPEATS):
        llm.generate(prompts, sp)

    if args.single_shot:
        t0 = time.perf_counter()
        llm.generate(prompts, sp)
        total_ms = (time.perf_counter() - t0) * 1000
        decode_ms = max(total_ms - ttft_ms, 0.0)
        step_us = (decode_ms / decode_steps * 1000) if decode_steps > 0 else 0.0
        tpot_ms = (decode_ms / decode_steps / bs) if decode_steps > 0 else 0.0
        throughput = bs * args.output_len / (total_ms / 1000)
        print(
            f"SINGLE_SHOT {args.config} {total_ms:.3f} {step_us:.3f} "
            f"{tpot_ms:.4f} {ttft_ms:.3f} {throughput:.3f}"
        )
        return

    print(f"  Warmup: {WARMUP_REPEATS} × generate(max_tokens={args.output_len}) done")
    print()
    print(
        f"  {'Rep':>4}  {'Total(ms)':>10}  {'Step(µs)':>10}  "
        f"{'TPOT(ms)':>9}  {'Throughput(tok/s)':>18}"
    )
    print(f"  {'-' * 4}  {'-' * 10}  {'-' * 10}  {'-' * 9}  {'-' * 18}")

    total_ms_list: list[float] = []
    step_us_list: list[float] = []
    tpot_ms_list: list[float] = []
    throughput_list: list[float] = []

    for rep in range(MEASURE_REPEATS):
        t0 = time.perf_counter()
        llm.generate(prompts, sp)
        total_ms = (time.perf_counter() - t0) * 1000

        decode_ms = max(total_ms - ttft_ms, 0.0)
        step_us = (decode_ms / decode_steps * 1000) if decode_steps > 0 else 0.0
        tpot_ms = (decode_ms / decode_steps / bs) if decode_steps > 0 else 0.0
        throughput = bs * args.output_len / (total_ms / 1000)

        total_ms_list.append(total_ms)
        step_us_list.append(step_us)
        tpot_ms_list.append(tpot_ms)
        throughput_list.append(throughput)

        print(
            f"  {rep + 1:>4}  {total_ms:>10.1f}  {step_us:>10.1f}  "
            f"{tpot_ms:>9.2f}  {throughput:>18.1f}"
        )

    print()
    mean_total = statistics.mean(total_ms_list)
    mean_step = statistics.mean(step_us_list)
    mean_tpot = statistics.mean(tpot_ms_list)
    mean_thru = statistics.mean(throughput_list)
    stdev_step = statistics.stdev(step_us_list)
    stdev_thru = statistics.stdev(throughput_list)

    print("=" * 68)
    print(f"RESULTS (mean ± stdev, n={MEASURE_REPEATS})")
    print("=" * 68)
    print(f"  Config:              {_CONFIG_LABEL[args.config]}")
    print(f"  TTFT:                {ttft_ms:.1f} ms")
    print(
        f"  Total generate():    {mean_total:.1f} ms  "
        f"({bs} seqs × {args.output_len} tok)"
    )
    print(
        f"  Decode step:         {mean_step:.1f} ± {stdev_step:.1f} µs/step  "
        f"({decode_steps} steps after first token)"
    )
    print(
        f"  TPOT:                {mean_tpot:.2f} ms/tok/seq  (decode step ÷ batch_size)"
    )
    print(
        f"  Throughput:          {mean_thru:.1f} ± {stdev_thru:.1f} tok/s  "
        f"(batch_size × output_len ÷ total_s)"
    )
    print()
    print("  Definitions:")
    print("    TTFT     = wall-clock from generate() call to first token returned")
    print("    Step     = (total − TTFT) / (output_len − 1)  [decode only]")
    print("    TPOT     = step / batch_size  [per-sequence decode latency]")
    print("    Throughput = batch_size × output_len / total_s  [output tokens/s]")
    print()


if __name__ == "__main__":
    main()

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build, verify, and benchmark the gfx942 FP8 MQA-logits instance.

The module is also the shared runner used by the live AITER comparison under
``library/benchmarks/gfx942/fp8_mqa_logits``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics

import torch

from rocke.helpers.compile import compile_kernel
from rocke.instances.gfx942.fp8_mqa_logits import (
    Fp8MqaLogitsSpec,
    build_fp8_mqa_logits,
    fp8_mqa_logits_grid,
    fp8_mqa_logits_num_splits,
    fp8_mqa_logits_signature,
)
from rocke.runtime import (
    KernelLauncher,
    LaunchConfig,
    synchronize_and_release,
    time_launches,
)


ARCH = "gfx942"
DEFAULT_SHAPE = (4, 128)


def parse_shape(value: str) -> tuple[int, int]:
    """Parse ``SQxSKV`` and require a non-empty KV suffix."""

    normalized = value.lower().replace("×", "x")
    try:
        seq_q, seq_kv = (int(part) for part in normalized.split("x", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"shape must be SQxSKV, got {value!r}"
        ) from exc
    if seq_q <= 0 or seq_kv <= 0 or seq_kv < seq_q:
        raise argparse.ArgumentTypeError("shape requires 0 < SQ <= SKV")
    return seq_q, seq_kv


def gfx_name() -> str:
    """Return the current HIP device's architecture without feature suffixes."""

    value = torch.cuda.get_device_properties(0).gcnArchName
    return str(value).split(":", 1)[0]


def make_inputs(
    seq_q: int,
    seq_kv: int,
    num_heads: int,
    head_dim: int,
    *,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Create deterministic native-gfx942 FP8 inputs and causal-like windows."""

    torch.manual_seed(seed)
    dtype = torch.float8_e4m3fnuz
    q = torch.randn(
        seq_q,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(dtype)
    kv = torch.randn(
        seq_kv,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(dtype)
    kv_scales = torch.rand(seq_kv, dtype=torch.float32, device="cuda") + 0.5
    weights = torch.randn(
        seq_q,
        num_heads,
        dtype=torch.float32,
        device="cuda",
    )
    starts = torch.zeros(seq_q, dtype=torch.int32, device="cuda")
    ends = torch.arange(seq_q, dtype=torch.int32, device="cuda") + (seq_kv - seq_q)
    return {
        "q": q,
        "kv": kv,
        "kv_scales": kv_scales,
        "weights": weights,
        "starts": starts,
        "ends": ends,
    }


def pad_rows(
    inputs: dict[str, torch.Tensor],
    rows_per_block: int,
) -> tuple[dict[str, torch.Tensor], int]:
    """Pad query-side tensors so every block owns complete rows."""

    seq_q = inputs["q"].shape[0]
    padded = math.ceil(seq_q / rows_per_block) * rows_per_block
    if padded == seq_q:
        return inputs, padded
    pad = padded - seq_q
    result = dict(inputs)
    result["q"] = torch.cat(
        [
            inputs["q"],
            inputs["q"].new_zeros((pad, inputs["q"].shape[1], inputs["q"].shape[2])),
        ]
    )
    result["weights"] = torch.cat(
        [
            inputs["weights"],
            inputs["weights"].new_zeros((pad, inputs["weights"].shape[1])),
        ]
    )
    result["starts"] = torch.cat([inputs["starts"], inputs["starts"].new_zeros(pad)])
    result["ends"] = torch.cat([inputs["ends"], inputs["ends"].new_zeros(pad)])
    return result, padded


def select_spec(
    seq_q: int,
    seq_kv: int,
    num_heads: int,
    head_dim: int,
    *,
    block_kv: int | None = None,
    rows_per_block: int | None = None,
    waves_per_block: int | None = None,
    waves_per_eu: int | None = 2,
) -> Fp8MqaLogitsSpec:
    """Select the measured gfx942 geometry unless explicitly overridden."""

    if seq_kv >= 65536:
        default_block_kv, default_rows, default_waves = 64, 4, 4
    elif seq_q >= 4096:
        default_block_kv, default_rows, default_waves = 64, 4, 2
    else:
        default_block_kv, default_rows, default_waves = 128, 2, 2
    return Fp8MqaLogitsSpec(
        num_heads=num_heads,
        head_dim=head_dim,
        block_kv=block_kv if block_kv is not None else default_block_kv,
        rows_per_block=(rows_per_block if rows_per_block is not None else default_rows),
        waves_per_block=(
            waves_per_block if waves_per_block is not None else default_waves
        ),
        waves_per_eu=waves_per_eu,
    )


def select_num_splits(
    seq_q: int,
    seq_q_padded: int,
    seq_kv: int,
    spec: Fp8MqaLogitsSpec,
    *,
    num_cus: int,
    target_blocks_per_cu: int = 4,
    override: int | None = None,
) -> int:
    """Select grid-y parallelism, including the measured long-context winner."""

    if override is not None:
        if override <= 0:
            raise ValueError("num_splits override must be positive")
        return override
    if seq_q == 671 and seq_kv == 131072 and spec.block_kv == 64:
        return 18
    return fp8_mqa_logits_num_splits(
        seq_q_padded,
        seq_kv,
        rows_per_block=spec.rows_per_block,
        block_kv=spec.block_kv,
        num_cus=num_cus,
        target_blocks_per_cu=target_blocks_per_cu,
    )


def build_runner(
    inputs: dict[str, torch.Tensor],
    seq_q: int,
    seq_kv: int,
    spec: Fp8MqaLogitsSpec,
    *,
    target_blocks_per_cu: int = 4,
    num_splits_override: int | None = None,
):
    """Compile the instance and return a callable launch plus its output metadata."""

    padded_inputs, seq_q_padded = pad_rows(inputs, spec.rows_per_block)
    num_cus = torch.cuda.get_device_properties(0).multi_processor_count
    num_splits = select_num_splits(
        seq_q,
        seq_q_padded,
        seq_kv,
        spec,
        num_cus=num_cus,
        target_blocks_per_cu=target_blocks_per_cu,
        override=num_splits_override,
    )
    output = torch.full(
        (seq_q_padded, seq_kv),
        -float("inf"),
        dtype=torch.float32,
        device="cuda",
    )
    artifact = compile_kernel(
        build_fp8_mqa_logits(spec, arch=ARCH),
        arch=ARCH,
        backend="python",
        capture_ir_text=False,
    )
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=fp8_mqa_logits_signature(spec),
        cache_key=("fp8_mqa_logits_example", spec),
    )
    stream = int(torch.cuda.current_stream().cuda_stream)
    config = LaunchConfig(
        grid=fp8_mqa_logits_grid(seq_q_padded, num_splits, spec),
        block=(spec.block_size, 1, 1),
        stream=stream,
    )
    values = {
        "Q": padded_inputs["q"],
        "KV": padded_inputs["kv"],
        "kv_scales": padded_inputs["kv_scales"],
        "weights": padded_inputs["weights"],
        "cu_starts": padded_inputs["starts"],
        "cu_ends": padded_inputs["ends"],
        "logits": output,
        "seq_len": seq_q_padded,
        "seq_len_kv": seq_kv,
        "stride_logits_s": output.stride(0),
        "num_splits": num_splits,
    }

    def call_once():
        output.fill_(-float("inf"))
        launcher(values, config=config)

    return call_once, output, stream, num_splits, artifact.kernel_name


def calc_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return the scale-insensitive similarity error used by AITER tests."""

    left = left.double()
    right = right.double()
    denominator = (left * left + right * right).sum()
    if not bool(denominator):
        return 0.0
    similarity = 2 * (left * right).sum() / denominator
    return float((1 - similarity).item())


def compare_outputs(
    left: torch.Tensor,
    right: torch.Tensor,
    seq_q: int,
    *,
    threshold: float = 1e-3,
) -> tuple[float, float]:
    """Check masks and finite values, returning similarity and max-absolute errors."""

    left = left[:seq_q]
    right = right[:seq_q]
    left_mask = torch.isneginf(left)
    right_mask = torch.isneginf(right)
    if not torch.equal(left_mask, right_mask):
        raise AssertionError("output masks differ")
    left_finite = left.masked_fill(left_mask, 0)
    right_finite = right.masked_fill(right_mask, 0)
    diff = calc_diff(left_finite, right_finite)
    max_abs = float((left_finite - right_finite).abs().max().item())
    if diff >= threshold:
        raise AssertionError(f"calc_diff={diff} exceeds {threshold}")
    return diff, max_abs


def torch_reference(
    inputs: dict[str, torch.Tensor],
    seq_q: int,
    seq_kv: int,
) -> torch.Tensor:
    """Compute a row-at-a-time FP32 reference without materializing M×H×N."""

    q = inputs["q"].float()
    kv = inputs["kv"].float()
    scales = inputs["kv_scales"].float()
    weights = inputs["weights"].float()
    starts = inputs["starts"].cpu()
    ends = inputs["ends"].cpu()
    output = torch.full(
        (seq_q, seq_kv),
        -float("inf"),
        dtype=torch.float32,
        device="cuda",
    )
    for row in range(seq_q):
        start = max(0, int(starts[row]))
        end = min(seq_kv, int(ends[row]))
        if start >= end:
            continue
        scores = torch.matmul(q[row], kv[start:end].transpose(0, 1))
        weighted = torch.relu(scores) * weights[row, :, None]
        output[row, start:end] = weighted.sum(dim=0) * scales[start:end]
    return output


def time_runner(
    call_once,
    *,
    stream: int,
    warmup: int,
    iters: int,
    repeats: int,
) -> float:
    """Return the median latency across repeated HIP-event measurements."""

    samples = []
    for _ in range(repeats):
        samples.append(
            time_launches(
                call_once,
                warmup=warmup,
                iters=iters,
                stream=stream,
            )
        )
        synchronize_and_release(stream)
    return statistics.median(samples)


def variant_name(spec: Fp8MqaLogitsSpec, num_splits: int) -> str:
    """Return a compact, stable geometry label."""

    return (
        f"b{spec.block_kv}_r{spec.rows_per_block}"
        f"_w{spec.waves_per_block}_wpe{spec.waves_per_eu or 0}"
        f"_s{num_splits}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        type=parse_shape,
        default=DEFAULT_SHAPE,
        metavar="SQxSKV",
    )
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-kv", type=int)
    parser.add_argument("--rows-per-block", type=int)
    parser.add_argument("--waves-per-block", type=int)
    parser.add_argument("--waves-per-eu", type=int, default=2)
    parser.add_argument("--target-blocks-per-cu", type=int, default=4)
    parser.add_argument("--num-splits", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    if gfx_name() != ARCH:
        raise RuntimeError(
            f"this example requires {ARCH}; current device is {gfx_name()}"
        )

    seq_q, seq_kv = args.shape
    inputs = make_inputs(
        seq_q,
        seq_kv,
        args.num_heads,
        args.head_dim,
        seed=args.seed,
    )
    spec = select_spec(
        seq_q,
        seq_kv,
        args.num_heads,
        args.head_dim,
        block_kv=args.block_kv,
        rows_per_block=args.rows_per_block,
        waves_per_block=args.waves_per_block,
        waves_per_eu=None if args.waves_per_eu == 0 else args.waves_per_eu,
    )
    call_once, output, stream, num_splits, kernel_name = build_runner(
        inputs,
        seq_q,
        seq_kv,
        spec,
        target_blocks_per_cu=args.target_blocks_per_cu,
        num_splits_override=args.num_splits,
    )

    call_once()
    synchronize_and_release(stream)
    result = {
        "arch": ARCH,
        "kernel": kernel_name,
        "shape": f"{seq_q}x{seq_kv}",
        "variant": variant_name(spec, num_splits),
    }

    if args.verify:
        reference = torch_reference(inputs, seq_q, seq_kv)
        diff, max_abs = compare_outputs(reference, output, seq_q)
        result.update(
            {
                "calc_diff": diff,
                "max_abs_diff": max_abs,
                "bad_count": 0,
                "total": seq_q * seq_kv,
            }
        )
        print(f"verify: calc_diff={diff:.6g} max_abs_diff={max_abs:.6g} " "bad=0 PASS")

    if args.bench:
        result["ms"] = time_runner(
            call_once,
            stream=stream,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        print(f"latency: {result['ms']:.6f} ms")

    print(f"kernel: {kernel_name}")
    print(f"variant: {variant_name(spec, num_splits)}")
    print("PerfJSON:", json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

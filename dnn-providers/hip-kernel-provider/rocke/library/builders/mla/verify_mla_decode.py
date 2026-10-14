# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-device parity for MLA decode: latent kernel, split-KV combine, whole op.

Three modes over one ragged, paged decode problem generator:

``--mode latent`` (default)
    Runs the dispatched decode kernel(s) on ``Qa`` and compares the normalized
    latent and the LSE against :func:`builders.mla.ref_mla_decode.ref_mla_decode_latent`
    (fp64, fed the same bf16 ``Qa`` the kernel reads).

``--mode full``
    The whole decode op through :class:`builders.mla.mla_decode_runner.MlaDecodeRunner`:
    ``torch.bmm`` absorb, the kernel(s), ``torch.bmm`` ``W_UV``. Compared
    against :func:`builders.mla.ref_mla_decode.ref_mla_decode_absorb`.

``--mode oracle``
    An independent GPU oracle: the prefill ``absorbed_query`` kernel run on the
    *same* device buffers at one query per sequence. Same math, different code
    (one query per wave row, a different grid and block numbering), so it
    catches a misconception mirrored into both the reference and this kernel.
    It runs on a clean cache by default: the prefill kernel does not survive
    NaN in the unused rows of a partial last page (``--poison-unused`` shows
    it), which the decode kernel is gated on.

``--num-seg`` takes ``auto`` (the dispatched split count) and/or forced
counts; a forced count larger than a sequence's tile count leaves trailing
segments empty, which the combine must weigh to zero.

Tolerance: elementwise ``|got - ref| <= ATOL + RTOL * |ref|`` on the bf16
output, with ``ATOL = 4e-2`` (``builders/mla/DESIGN.md`` section 8.1) and
``RTOL = 2e-2`` (the MLA prefill harness's bound). The relative term is what
lets the gate scale with the output: a bf16 output rounds by up to half an ulp
of its own magnitude, which a fixed absolute bound cannot absorb once
``|out|`` passes about 4. The LSE is fp32 and gated at ``ATOL`` alone, in
natural-log units. The run also reports ``max_abs`` and ``max_abs / max|ref|``.
Fixed values are a stopgap: thresholds derived from the accumulation depth
(CK-Tile's ``check_err``) are the follow-up.

Rows of sequences with ``S_k = 0`` are checked exactly instead (zero output,
``lse = -inf``). The output buffers and the split-KV workspace are NaN-seeded,
and so are the cache rows no sequence owns -- including each sequence's
partial last page -- so an unwritten slot or a key read past ``S_k`` fails
loudly rather than reading as a plausible value.

Run, from ``rocke/library`` on a GPU node::

    python -m builders.mla.verify_mla_decode                       # every scenario
    python -m builders.mla.verify_mla_decode --mode full --num-seg auto 1 7 64
    python -m builders.mla.verify_mla_decode --mode oracle --scenario ragged_h128

The benchmark shape files are verified through the benchmark driver, which may
import this module (``builders`` may not import ``benchmarks``)::

    python -m benchmarks.common.benchmark_mla_decode_live --mode verify \\
        --shapes benchmarks/gfx942/attention/decode/mla_shapes.json

Inputs are generated pre-rounded to bf16, so the device and the reference read
identical numbers. ``q_nope`` is drawn at unit scale so the softmax is peaked
(score std ~2) rather than near-uniform, where a broken softmax would hide.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from builders.mla.ref_mla_decode import (
    absorb_query,
    ref_mla_decode_absorb,
    ref_mla_decode_latent,
)

ATOL = 4e-2
RTOL = 2e-2


class Scenario(NamedTuple):
    heads: int
    q_lens: tuple[int, ...]
    k_lens: tuple[int, ...]
    why: str


_SHAPES = (
    (
        "ragged",
        (1,) * 8,
        (1, 15, 16, 17, 100, 333, 1000, 4097),
        "one key, sub-page, exact page, page + 1, and long tails in one batch",
    ),
    (
        "no_query",
        (1, 0, 1, 1),
        (37, 50, 1000, 2048),
        "a sequence with no query this step must not shift its neighbours",
    ),
    (
        "zero_keys",
        (1, 1, 1),
        (0, 20, 600),
        "S_k = 0: zero latent and lse = -inf, through either epilogue",
    ),
    (
        "long",
        (1, 1),
        (16384, 32768 - 5),
        "the deepest fp32 accumulation, with a page tail",
    ),
)

SCENARIOS = {
    f"{name}_h{heads}": Scenario(heads, q, k, why)
    for (name, q, k, why) in _SHAPES
    for heads in (64, 128)
}


# --------------------------------------------------------------------------
# Problem construction
# --------------------------------------------------------------------------


def _bf16(a: np.ndarray) -> np.ndarray:
    """Round-trip fp32 through bf16 (round-to-nearest-even)."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return u.astype(np.uint32).view(np.float32)


class Problem(NamedTuple):
    q_nope: np.ndarray  # [T, H, d_nope]
    q_rope: np.ndarray  # [T, H, d_rope]
    c_kv: np.ndarray  # [pages, page, r_kv]
    k_rope: np.ndarray  # [pages, page, d_rope]
    w_uk: np.ndarray  # [H, r_kv, d_nope + d_v]
    cu_seqlens_q: np.ndarray
    cu_seqlens_k: np.ndarray
    block_table: np.ndarray
    scale: float

    @property
    def num_seqs(self) -> int:
        return len(self.cu_seqlens_q) - 1


def make_problem(
    *,
    heads: int,
    q_lens: Sequence[int],
    k_lens: Sequence[int],
    d_nope: int = 128,
    d_rope: int = 64,
    d_v: int = 128,
    r_kv: int = 512,
    page: int = 16,
    seed: int = 0,
    poison_unused: bool = True,
) -> Problem:
    """A ragged paged decode batch at the real MLA geometry.

    Pages are shuffled across the whole cache so a kernel that assumes
    sequence-contiguous pages fails. With ``poison_unused`` every cache row no
    sequence owns is NaN: a kernel that lets a key past ``S_k`` reach the
    result, even multiplied by a zero probability, produces NaN.
    """
    rng = np.random.default_rng(seed)
    cu_q = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    cu_k = np.concatenate([[0], np.cumsum(k_lens)]).astype(np.int32)
    total_q = int(cu_q[-1])
    max_blocks = max(1, max(-(-k // page) for k in k_lens))
    num_blocks = len(k_lens) * max_blocks
    block_table = (
        rng.permutation(num_blocks).astype(np.int32).reshape(len(k_lens), max_blocks)
    )

    def rand(shape, sigma):
        return _bf16(rng.standard_normal(shape, dtype=np.float32) * sigma)

    c_kv = rand((num_blocks, page, r_kv), 1.0)
    k_rope = rand((num_blocks, page, d_rope), 1.0)
    if poison_unused:
        owned = np.zeros((num_blocks, page), dtype=bool)
        for b, k in enumerate(k_lens):
            rows = np.arange(k)
            owned[block_table[b, rows // page], rows % page] = True
        c_kv[~owned] = np.nan
        k_rope[~owned] = np.nan
    return Problem(
        q_nope=rand((total_q, heads, d_nope), 1.0),
        q_rope=rand((total_q, heads, d_rope), 1.0),
        c_kv=c_kv,
        k_rope=k_rope,
        w_uk=rand((heads, r_kv, d_nope + d_v), 0.1),
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        block_table=block_table,
        scale=1.0 / math.sqrt(d_nope + d_rope),
    )


def uniform_problem(*, heads: int, batch: int, seqlen_k: int, seed: int) -> Problem:
    """A shape-file point, with lengths jittered below ``seqlen_k`` so most
    sequences end in a page tail and no two share a tile count."""
    k_lens = [max(1, seqlen_k - (7 * b) % 23) for b in range(batch)]
    k_lens[0] = seqlen_k
    return make_problem(heads=heads, q_lens=(1,) * batch, k_lens=k_lens, seed=seed)


# --------------------------------------------------------------------------
# Device run
# --------------------------------------------------------------------------


def to_device(a: np.ndarray, dtype):
    import torch

    return torch.from_numpy(np.ascontiguousarray(a)).to(device="cuda", dtype=dtype)


def _request(p: Problem, *, arch: str, algorithm: str = "auto"):
    from dispatch.mla import MLARequest

    k_lens = np.diff(p.cu_seqlens_k)
    q_lens = np.diff(p.cu_seqlens_q)
    return MLARequest(
        num_heads=p.q_nope.shape[1],
        total_q=int(p.cu_seqlens_q[-1]),
        num_seqs=p.num_seqs,
        max_seqlen_q=max(1, int(q_lens.max())),
        max_seqlen_k=int(k_lens.max()),
        arch=arch,
        algorithm=algorithm,
    )


def device_inputs(p: Problem) -> dict:
    import torch

    return {
        "c_kv": to_device(p.c_kv, torch.bfloat16),
        "k_rope": to_device(p.k_rope, torch.bfloat16),
        "cu_seqlens_q": to_device(p.cu_seqlens_q, torch.int32),
        "cu_seqlens_k": to_device(p.cu_seqlens_k, torch.int32),
        "block_table": to_device(p.block_table, torch.int32),
    }


def _nan_outputs(total_q: int, heads: int, width: int):
    import torch

    out = torch.full((total_q, heads, width), float("nan"), dtype=torch.bfloat16)
    lse = torch.full((total_q, heads), float("nan"), dtype=torch.float32)
    return out.cuda(), lse.cuda()


def _poison_workspace(runner, total_q: int) -> None:
    """NaN-fill the split-KV workspace, so a partial the latent kernel never
    wrote cannot merge as a stale but plausible one."""
    import torch

    if runner.spec.split_kv:
        device = torch.device("cuda", torch.cuda.current_device())
        for t in runner.workspace(total_q, device):
            t.fill_(float("nan"))


def run_latent(runner, p: Problem, qa_bf16: np.ndarray):
    import torch

    _poison_workspace(runner, qa_bf16.shape[0])
    dev = device_inputs(p)
    out, lse = _nan_outputs(qa_bf16.shape[0], qa_bf16.shape[1], runner.spec.r_kv)
    runner.latent(
        qa=to_device(qa_bf16, torch.bfloat16), scale=p.scale, out=out, lse=lse, **dev
    )
    torch.cuda.synchronize()
    return out.float().cpu().numpy(), lse.cpu().numpy()


def run_full(runner, p: Problem):
    import torch

    _poison_workspace(runner, p.q_nope.shape[0])
    out, lse = runner.forward(
        q_nope=to_device(p.q_nope, torch.bfloat16),
        q_rope=to_device(p.q_rope, torch.bfloat16),
        w_uk=to_device(p.w_uk, torch.bfloat16),
        scale=p.scale,
        **device_inputs(p),
    )
    torch.cuda.synchronize()
    return out.float().cpu().numpy(), lse.cpu().numpy()


def run_prefill_oracle(p: Problem, qa_bf16: np.ndarray, *, arch: str):
    """The prefill ``absorbed_query`` kernel on the same buffers: the 12-argument
    pack is the decode kernel's, so only the grid differs."""
    import dataclasses

    import torch
    from dispatch.mla import dispatch_mla, num_q_blocks_for
    from kernels.mla.mla_prefill_gfx942 import (
        build_mla_prefill_fwd,
        mla_prefill_block,
        mla_prefill_fwd_grid,
        mla_prefill_fwd_signature,
    )
    from rocke import compile_kernel
    from rocke.runtime import KernelLauncher, LaunchConfig

    req = _request(p, arch=arch, algorithm="mla_prefill_chunked")
    spec = dataclasses.replace(dispatch_mla(req).spec, absorbed_query=True)
    art = compile_kernel(build_mla_prefill_fwd(spec, arch=arch), arch=arch)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=tuple(mla_prefill_fwd_signature(spec)),
    )
    dev = device_inputs(p)
    out, lse = _nan_outputs(qa_bf16.shape[0], qa_bf16.shape[1], spec.r_kv)
    launcher(
        {
            "out_ptr": out,
            "lse_ptr": lse,
            "qa_ptr": to_device(qa_bf16, torch.bfloat16),
            "c_kv_ptr": dev["c_kv"],
            "k_rope_ptr": dev["k_rope"],
            "cu_seqlens_q_ptr": dev["cu_seqlens_q"],
            "cu_seqlens_k_ptr": dev["cu_seqlens_k"],
            "block_table_ptr": dev["block_table"],
            "scale": float(p.scale),
            "num_seqs": p.num_seqs,
            "block_table_stride": int(p.block_table.shape[1]),
            "max_k": int(np.diff(p.cu_seqlens_k).max()),
        },
        config=LaunchConfig(
            grid=mla_prefill_fwd_grid(
                spec, num_q_blocks=num_q_blocks_for(req, spec.block_q)
            ),
            block=mla_prefill_block(spec),
        ),
    )
    torch.cuda.synchronize()
    return out.float().cpu().numpy(), lse.cpu().numpy()


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


class Residual(NamedTuple):
    max_abs: float
    rel: float
    peak: float


def _residual(got: np.ndarray, ref: np.ndarray) -> Residual:
    if got.size == 0:
        return Residual(0.0, 0.0, 0.0)
    peak = float(np.max(np.abs(ref)))
    err = float(np.max(np.abs(got.astype(np.float64) - ref)))
    return Residual(err, err / (peak if peak > 0 else 1.0), peak)


def compare(
    p: Problem,
    got_out,
    got_lse,
    ref_out,
    ref_lse,
    *,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> tuple[bool, str]:
    """Gate one result; ``S_k = 0`` rows are checked exactly, not by residual."""
    k_lens = np.diff(p.cu_seqlens_k)
    q_lens = np.diff(p.cu_seqlens_q)
    tok_of_seq = p.cu_seqlens_q[:-1]
    dead = np.zeros(got_out.shape[0], dtype=bool)
    for b in range(p.num_seqs):
        if q_lens[b] and k_lens[b] == 0:
            dead[tok_of_seq[b]] = True
    problems = []
    if np.isnan(got_out).any() or np.isnan(got_lse).any():
        problems.append(
            f"NaN in output ({int(np.isnan(got_out).sum())} out, "
            f"{int(np.isnan(got_lse).sum())} lse): unwritten or poisoned slots"
        )
    if dead.any():
        if not (np.all(got_out[dead] == 0.0) and np.all(got_lse[dead] == -np.inf)):
            problems.append("S_k = 0 rows must be exactly 0 with lse = -inf")
    live = ~dead
    r_o = _residual(got_out[live], ref_out[live])
    r_l = _residual(got_lse[live], ref_lse[live])
    live_got = got_out[live].astype(np.float64)
    live_ref = np.asarray(ref_out[live], dtype=np.float64)
    excess = np.abs(live_got - live_ref) - (atol + rtol * np.abs(live_ref))
    if excess.size and np.nanmax(excess) > 0:
        problems.append(
            f"out exceeds {atol:g} + {rtol:g}*|ref| by up to {np.nanmax(excess):.3e}"
        )
    if r_l.max_abs > atol:
        problems.append(f"lse max_abs {r_l.max_abs:.3e} > {atol:g}")
    line = (
        f"out[peak={r_o.peak:.3f} max_abs={r_o.max_abs:.3e} rel={r_o.rel:.3e}] "
        f"lse[max_abs={r_l.max_abs:.3e} rel={r_l.rel:.3e}]"
    )
    if problems:
        line += "  FAIL: " + "; ".join(problems)
    else:
        line += "  ok"
    return not problems, line


def run_case(
    label: str,
    p: Problem,
    *,
    mode: str,
    num_segs: Sequence[str],
    arch: str,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> bool:
    from builders.mla.mla_decode_runner import MlaDecodeRunner
    from dispatch.mla import dispatch_mla

    k_lens = np.diff(p.cu_seqlens_k)
    print(
        f"== {label}: H={p.q_nope.shape[1]} seqs={p.num_seqs} "
        f"S_k=[{int(k_lens.min())}..{int(k_lens.max())}] mode={mode}",
        flush=True,
    )
    result = dispatch_mla(_request(p, arch=arch))
    qa_bf16 = _bf16(absorb_query(p.q_nope, p.q_rope, p.w_uk).astype(np.float32))
    if mode == "full":
        ref = ref_mla_decode_absorb(
            p.q_nope,
            p.q_rope,
            p.c_kv,
            p.k_rope,
            p.w_uk,
            p.block_table,
            p.cu_seqlens_q,
            p.cu_seqlens_k,
            p.scale,
        )
    else:
        ref = ref_mla_decode_latent(
            qa_bf16,
            p.c_kv,
            p.k_rope,
            p.block_table,
            p.cu_seqlens_q,
            p.cu_seqlens_k,
            p.scale,
        )
    ok = True
    if mode == "oracle":
        got = run_prefill_oracle(p, qa_bf16, arch=arch)
        passed, line = compare(p, *got, *ref, atol=atol, rtol=rtol)
        print(f"   prefill-oracle vs reference  {line}", flush=True)
        ok &= passed
        oracle = got
    for ns in num_segs:
        forced = None if ns == "auto" else int(ns)
        runner = MlaDecodeRunner(result, num_seg=forced)
        tag = f"num_seg={runner.spec.num_seg}{'(auto)' if forced is None else ''}"
        if mode == "full":
            got = run_full(runner, p)
        else:
            got = run_latent(runner, p, qa_bf16)
        passed, line = compare(p, *got, *ref, atol=atol, rtol=rtol)
        print(f"   {result.candidate.name} {tag:<16} {line}", flush=True)
        ok &= passed
        if mode == "oracle":
            passed, line = compare(p, *got, *oracle, atol=atol, rtol=rtol)
            print(f"   {'':<{len(result.candidate.name)}} vs prefill-oracle  {line}")
            ok &= passed
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("latent", "full", "oracle"), default="latent")
    ap.add_argument(
        "--scenario",
        action="append",
        choices=sorted(SCENARIOS),
        help="named scenario(s); default: all of them",
    )
    ap.add_argument(
        "--num-seg",
        nargs="+",
        default=["auto"],
        help="'auto' and/or forced split counts, each run against the reference",
    )
    ap.add_argument("--atol", type=float, default=ATOL)
    ap.add_argument("--rtol", type=float, default=RTOL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default=None, help="default: the local device")
    ap.add_argument(
        "--poison-unused",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="NaN-fill cache rows no sequence owns (default: on, off in --mode oracle)",
    )
    args = ap.parse_args(argv)
    # The prefill kernel lets a NaN past S_k on a sequence's last page reach
    # its PV (the score masks it; 0 * NaN does not), so the oracle runs on a
    # clean cache unless asked otherwise. Decode is gated on poisoned caches.
    poison = args.mode != "oracle" if args.poison_unused is None else args.poison_unused

    import torch  # noqa: F401  -- binds the torch-bundled comgr before rocke
    from rocke.runtime.hip_module import get_device_arch

    arch = args.arch or (get_device_arch() or "gfx942").split(":")[0]
    cases = []
    for name in args.scenario or sorted(SCENARIOS):
        sc = SCENARIOS[name]
        cases.append(
            (
                name,
                make_problem(
                    heads=sc.heads,
                    q_lens=sc.q_lens,
                    k_lens=sc.k_lens,
                    seed=args.seed,
                    poison_unused=poison,
                ),
            )
        )
    print(
        f"arch={arch} atol={args.atol:g} rtol={args.rtol:g} num_seg={args.num_seg} "
        f"poison_unused={poison}",
        flush=True,
    )
    failed = [
        label
        for label, p in cases
        if not run_case(
            label,
            p,
            mode=args.mode,
            num_segs=args.num_seg,
            arch=arch,
            atol=args.atol,
            rtol=args.rtol,
        )
    ]
    print(f"\nSUMMARY {len(cases) - len(failed)}/{len(cases)} cases passed")
    for label in failed:
        print(f"  FAILED {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

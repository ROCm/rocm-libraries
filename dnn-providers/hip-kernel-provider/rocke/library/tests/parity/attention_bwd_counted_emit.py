#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_bwd_counted_emit.py -- Python reference emitter for
# the counted in-loop waits of the attention backward q step
# (kernels/common/_attention_bwd_counted.py). The module has no kernel of its
# own, so this builds a probe kernel that calls every entry point (tile_chunks,
# issue_tile_loads, store_tile, zero_like, StagedStep, CountedWaits) the way the
# backward body does; the C twin attention_bwd_counted_emit.c builds the same
# kernel through helper_rocke.instances.common._attention_bwd_counted.h and
# the two .ll must match.
#
# Configs are decoded from argv[1] as an index into _CONFIGS, built by the same
# loops in the C twin (keep both in sync): every (shape, wait mode, arch) with
# the dtype, dQ atomic count and stats count cycling over the combinations,
# then the special configs (rejects and edge counts) listed in _SPECIAL.
# Shapes are (k_m0, head_size, threads) of the backward tuned tables (block_m
# 16 / 32 / 128, head sizes 64 / 128, 2 / 4 waves) plus shapes that exercise a
# partial round of the chunk deal (a tail guard), one chunk per row and
# 512 threads.
#
# argv[2] selects the output: ll (default), ir, verify (see _emit_common.py).
from _emit_common import run_emit
from kernels.common._attention_bwd_counted import (
    CountedWaits,
    StagedStep,
    issue_tile_loads,
    store_tile,
    tile_chunks,
    zero_like,
)
from rocke.core.ir import BF16, F16, F32, I32, IRBuilder, PtrType, VectorType

ARCHES = ("gfx942", "gfx950")
DTYPES = ("f16", "bf16")
# (k_m0, head_size, threads)
SHAPES = (
    (16, 64, 128),
    (16, 128, 128),
    (32, 64, 128),
    (32, 128, 256),
    (16, 128, 256),
    (16, 32, 128),
    (32, 96, 256),
    (128, 64, 256),
    (16, 8, 64),
    (16, 64, 512),
)
MODES = ("early", "late", "deferred", "two_steps")
N_ATOMICS = (0, 4, 8, 16, 63)
N_STATS = (8, 16)
# (arch, dtype, shape, n_atomics, n_stats, mode, vec); the main grid uses vec 8
_SPECIAL = (
    ("gfx950", "f16", (16, 64, 128), 8, 8, "bad_end"),
    ("gfx942", "bf16", (32, 128, 256), 4, 16, "bad_end"),
    ("gfx950", "f16", (16, 64, 128), 64, 8, "early"),
    ("gfx942", "f16", (16, 64, 128), -1, 8, "early"),
    ("gfx950", "bf16", (16, 64, 128), 8, 8, "zero_bad"),
    ("gfx942", "f16", (16, 128, 256), 8, 8, "store_mismatch"),
    ("gfx950", "bf16", (32, 64, 128), 63, 16, "late"),
    ("gfx942", "f16", (16, 128, 128), 0, 8, "deferred"),
    # non-default chunk widths: the cols // vec floor division and chunk_coords
    ("gfx950", "f16", (16, 64, 128), 8, 8, "early", 4),
    ("gfx942", "bf16", (32, 128, 256), 4, 16, "two_steps", 16),
    ("gfx950", "bf16", (16, 64, 128), 16, 8, "deferred", 2),
    ("gfx942", "f16", (16, 96, 128), 8, 8, "late", 16),
    ("gfx950", "f16", (16, 8, 64), 0, 8, "early", 4),
)


def _configs():
    out = []
    for s, shape in enumerate(SHAPES):
        for m, mode in enumerate(MODES):
            for arch in ARCHES:
                out.append(
                    (
                        arch,
                        DTYPES[(s + m) % 2],
                        shape,
                        N_ATOMICS[(s + m) % len(N_ATOMICS)],
                        N_STATS[s % 2],
                        mode,
                        8,
                    )
                )
    out.extend(c if len(c) == 7 else (*c, 8) for c in _SPECIAL)
    return tuple(out)


_CONFIGS = _configs()


def _spec(idx: int):
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    cfg = _CONFIGS[idx]
    return cfg, cfg[0]


class _LdsView:
    """A tile view of the probe: ``store`` writes ``<vec x elem>`` to an LDS tile."""

    def __init__(self, lds) -> None:
        self.lds = lds

    def store(self, b, row, col, val, vec):
        b.smem_store_vN(self.lds, [row, col], val, vec)


def _sequence(b, w, n, mode):
    """The counted-wait event sequence of one q step (or two) per issue mode."""
    if mode == "early":
        w.step_entry()
        w.barrier(b)
        w.loads()
        w.barrier(b)
        w.atomics(n)
        w.barrier(b)
        w.end_step()
    elif mode == "late":
        w.step_entry()
        w.barrier(b)
        w.barrier(b)
        w.loads()
        w.atomics(n)
        w.barrier(b)
        w.end_step()
    elif mode == "deferred":
        w.step_entry()
        w.barrier(b)
        w.loads()
        w.atomics(n)
        w.barrier(b)
        w.barrier(b)
        w.end_step()
    elif mode == "two_steps":
        _sequence(b, w, n, "early")
        _sequence(b, w, n, "late")
    elif mode == "bad_end":
        w.step_entry()
        w.barrier(b)
        w.atomics(n + 1)
        w.end_step()
    else:
        raise SystemExit(f"unknown mode {mode}")


def build_attention_bwd_counted_kernel(spec, *, arch: str):
    """The probe kernel of one config."""
    _arch, dtype, (rows, cols, threads), n_atomics, n_stats, mode, vec = spec
    elem = F16 if dtype == "f16" else BF16
    b = IRBuilder("attn_bwd_counted_probe")
    src_q = b.param("src_q", PtrType(elem, "global"))
    src_do = b.param("src_do", PtrType(elem, "global"))
    stride = b.param("stride", I32)
    out = b.param("out", PtrType(F32, "global"))
    tid = b.thread_id_x()
    lds_q = b.smem_alloc(elem, (rows, cols), "lds_q")
    lds_do = b.smem_alloc(elem, (rows, cols), "lds_do")

    def loader(ptr):
        def load(row, col):
            idx = b.add(b.mul(row, stride), col)
            return b.global_load_vN(ptr, idx, elem, vec, align=16)

        return load

    chunks = tile_chunks(b, rows=rows, cols=cols, tid=tid, threads=threads, vec=vec)
    values = issue_tile_loads(b, [loader(src_q), loader(src_do)], chunks)
    views = [_LdsView(lds_q), _LdsView(lds_do)]
    if mode == "store_mismatch":
        store_tile(b, views, chunks, values[:-1], vec)
    store_tile(b, views, chunks, values, vec)
    types = [F32] * n_stats + [VectorType(elem, vec)] * len(values)
    if mode == "zero_bad":
        types.append(I32)
    zeros = zero_like(b, types)
    step = StagedStep(stats=tuple(zeros[:n_stats]), qdo=tuple(zeros[n_stats:]))
    flat = step.flat()
    back = StagedStep.from_flat(flat, n_stats)
    res = list(back.stats) + [b.cast_to_f32(b.vec_extract(v, 0)) for v in back.qdo]
    res += [b.cast_to_f32(b.vec_extract(v, 0)) for v in values]
    waits = CountedWaits(n_atomics)
    _sequence(b, waits, n_atomics, mode)
    res += [b.const_f32(float(n)) for n in waits.waits]
    for i, v in enumerate(res):
        b.global_store(out, b.const_i32(i), v)
    b.ret()
    return b.kernel


def main() -> int:
    return run_emit(
        _spec,
        build_attention_bwd_counted_kernel,
        usage="usage: attention_bwd_counted_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())

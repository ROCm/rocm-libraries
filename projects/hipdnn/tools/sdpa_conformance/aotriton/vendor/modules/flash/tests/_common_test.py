#!/usr/bin/env python
# Copyright ©2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import io
import os
from typing import List, Tuple, Optional
from collections import namedtuple
import numpy as np
import math
import torch
import hashlib

HAS_REDUCED_SDPA = hasattr(torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp")
# Set this env var when GPU system is not reliable
# In addition to compute reference in CPU. The any other operators like input generations
# are done on CPU as well
AOTRITON_TORCH_ONLY_USE_CPU = bool(int(os.getenv('AOTRITON_TORCH_ONLY_USE_CPU', default='0')))
# Usually we compare with GPU because it is much faster
# Overrides by AOTRITON_TORCH_ONLY_USE_CPU=1
AOTRITON_REF_DEVICE_OPTION_SET = (os.getenv('AOTRITON_REF_DEVICE_OPTION', default=None) is not None)
AOTRITON_REF_DEVICE_OPTION = os.getenv('AOTRITON_REF_DEVICE_OPTION', default='default')

def _get_torch_version():
    strver = str(torch .__version__).split('.')[:2]
    intver = [int(e) for e in strver]
    return tuple(intver)

TORCH_VERSION_TUPLE = _get_torch_version()

TORCH_GE_2_7 = (TORCH_VERSION_TUPLE >= (2, 7))

def _get_hip_version():
    # None on a CUDA build. torch 2.12.0+rocm7.14.0 reports '7.14.60850'.
    if torch.version.hip is None:
        return None
    strver = str(torch.version.hip).split('.')[:2]
    return tuple([int(e) for e in strver])

HIP_VERSION_TUPLE = _get_hip_version()

ROCM_IS_7_14 = (HIP_VERSION_TUPLE == (7, 14))

def fmt_hdim(val):
    if isinstance(val, tuple):
        return 'hdim(' + ','.join([str(e) for e in val]) + ')'
    return f'hdim{val}'

def fmt_nheads(val):
    if isinstance(val, tuple):
        return '(' + ','.join([str(e) for e in val]) + ')'
    return str(val)


def cdiv(x, div):
    return (x + div - 1) // div

def narrow_to_prime(t, prime_d, poison=float('nan')):
    """A `[..., :prime_d]` view whose slack is poisoned, or `t` unchanged.

    **The 8xD input contract, handed to the kernel the way the contract says.**
    Loads and stores are 8 columns wide, so the kernel touches `ceil8(hdim)`
    columns of every row; `flash_attn_func_gfx950.py` states that and `_args`
    enforces it, but the C++ launcher never calls `_args`, so nothing checks it
    on the shipped path. A tightly-packed odd width -- which is what
    `torch.rand(3, 5, 1033, 57)` gives you, and the reason `PRIME_HEADDIMS` was
    disabled in `_core_test_backward.py` -- has no slack and is outside the
    contract, so testing one proves nothing about the kernel.

    The tensor is therefore ALLOCATED at the 8-multiple, by the ordinary
    pipeline, and only narrowed here: extent `prime_d`, pitch `ceil8(prime_d)`
    visible through the strides alone. That is exactly what an AOTriton caller
    with a padded buffer passes, and `T4` reads both numbers off the tensor.

    The slack is filled with NaN rather than a large finite value on purpose:
    `0 * x` kills a finite leak and cannot kill a NaN, so a masking bug that
    multiplies by zero instead of discarding still shows up.
    """
    if t is None or prime_d is None or t.shape[-1] == prime_d:
        return t
    assert t.shape[-1] >= prime_d, f'{tuple(t.shape)} is narrower than prime_hdim {prime_d}'
    t[..., prime_d:] = poison
    return t[..., :prime_d]

# The three OUTER axes of a rank-4 attention tensor can be stored in any of the
# 3! = 6 orders. The innermost axis never moves, and is not the same axis for
# every tensor: on q/k/v/o/dout and every gradient it is the head dim, which the
# kernel loads and stores 8 columns at a time; on a bias it is the KV sequence,
# which is that tensor's own vectorised axis. Either way it must stay stride-1,
# so only the outer three permute.
#
# A layout is written in MEMORY ORDER -- outermost stride first -- naming
# LOGICAL axes. `(0, 2, 1)` therefore means "batch outermost, then the sequence,
# then the head", i.e. BSHD, which is what `storage_flip=(1, 2)` has always
# produced. `BHSD` is the identity, and is what everything else allocates.
BHSD = (0, 1, 2)
BSHD = (0, 2, 1)
HBSD = (1, 0, 2)
HSBD = (1, 2, 0)
SBHD = (2, 0, 1)
SHBD = (2, 1, 0)
ALL_LAYOUTS = (BHSD, BSHD, HBSD, HSBD, SBHD, SHBD)
LAYOUT_NAMES = {BHSD: 'BHSD', BSHD: 'BSHD', HBSD: 'HBSD',
                HSBD: 'HSBD', SBHD: 'SBHD', SHBD: 'SHBD'}

def layout_inverse(perm):
    """The `permute()` argument taking an allocation in order `perm` to logical order."""
    inv = [0, 0, 0]
    for pos, axis in enumerate(perm):
        inv[axis] = pos
    return (inv[0], inv[1], inv[2], 3)

def layout_alloc_dims(dims, perm):
    """Logical rank-4 `dims`, reordered into the allocation order `perm`."""
    return tuple(dims[axis] for axis in perm) + (dims[3],)

def layout_of(t):
    """The layout `t` is actually stored in, read back off its strides.

    Descending stride *is* memory order, by definition. The tie-break on the
    axis index only decides anything for an axis of extent 1, whose stride says
    nothing about where it sits; the layout tests keep all three extents above 1
    for exactly that reason, so that no two layouts can come back equal here.
    """
    return tuple(sorted(range(3), key=lambda axis: (-t.stride(axis), axis)))

def alloc_with_layout(dims, perm, *, dtype, device, generator=None, rand=False):
    """A tensor of logical shape `dims`, stored in memory order `perm`.

    Allocated in `perm` order -- so it is genuinely contiguous, with no gaps --
    and handed back as the permuted view, which is the only thing the caller and
    the kernel ever see. `rand`, or a `generator`, fills it; the values land in
    ALLOCATION order, so two layouts of the same logical shape do not hold the
    same numbers. That is intentional: a test that needs identical inputs under
    two layouts must copy, not re-seed.
    """
    adims = layout_alloc_dims(dims, perm)
    if rand or generator is not None:
        raw = torch.rand(*adims, generator=generator, dtype=dtype, device=device)
    else:
        raw = torch.empty(adims, dtype=dtype, device=device)
    return raw.permute(*layout_inverse(perm))

# Here rather than beside its one caller (core_test_sm_scale_magnitude) because
# everything it knows is knowledge about THIS file: that `alloc_with_layout`
# above is the only thing that ever fills an SdpaContext input, that it uses
# `torch.rand`, and that the tensor it hands back is a permuted view whose
# strides an in-place fill has to preserve. A second test that needs a
# different input distribution should find this next to the allocation it
# overrides, not in whichever suite happened to need it first.
def refill_inputs_normal(ctx, seed=0x9be9_98d4_cf17_5339):
    """Redraw an SdpaContext's q/k/v/b from N(0, 1), in place, keeping every stride.

    `alloc_with_layout` fills from `torch.rand` -- uniform [0, 1), all positive
    and all the same order of magnitude. That is a deliberately benign
    distribution and it hides an entire class of softmax error.

    Consider a kernel that perturbs Q by a relative `eps` before the QK GEMM
    (rounding `Q * qk_scale` back to the input dtype does exactly this). Row
    `i` of the score matrix picks up `S_ij += <q_i * eps_i, k_j>`, a quantity
    that depends on `j` only through `k_j`. With every `k_j` drawn from [0, 1)
    the `k_j` are near-parallel, so that inner product is very nearly the SAME
    number for every column of the row -- a common-mode shift, and softmax
    subtracts the row max, so it cancels. Draw `k_j` from N(0, 1) and the signs
    vary from column to column, the perturbation stops being common mode, and
    it survives into the output.

    Measured on gfx950, flyc forward, bf16, 289x289, hdim 512, `sm_scale=1.0`,
    max |err| vs an f64 reference: 0.0029 with uniform inputs, 0.0094 with
    normal ones, against a 0.0021 / 0.0020 baseline at `sm_scale=rsqrt(hdim)`.
    Uniform inputs move by 1.4x where normal inputs move by 4.7x.

    In place rather than as an `SdpaContext` argument because the layout is the
    allocation's: `alloc_with_layout` allocates in storage order and hands back
    a permuted view, so filling the view preserves exactly the strides the
    layout asked for. Must run BEFORE `create_ref_inputs`, which snapshots
    these tensors into the high-precision and low-precision references.
    """
    # ONE generator for the whole tuple, advanced from tensor to tensor. Seeding
    # per tensor instead would hand q, k and v the same numbers, and identical
    # q/k/v is a degenerate input: every score in a row is equal, softmax comes
    # out uniform, and the kernel matches the reference exactly no matter what
    # it does to the scale.
    g = None
    for t in ctx.dev_tensors:
        if t is None:
            continue
        if g is None:
            g = torch.Generator(device=t.device)
            g.manual_seed(seed)
        t.normal_(mean=0.0, std=1.0, generator=g)

# Every tensor whose layout a caller may choose, in the order `StorageLayout`'s
# round robin walks them. The first five are allocated by `SdpaContext`; the
# rest are outputs, allocated inside `attn_torch_function.py` and reached
# through `AttentionExtraArgs.output_layouts`.
LAYOUT_INPUT_TENSORS = ('q', 'k', 'v', 'b', 'dout')
LAYOUT_OUTPUT_TENSORS = ('o', 'dq', 'dk', 'dv', 'db')
LAYOUT_TENSOR_ORDER = LAYOUT_INPUT_TENSORS + LAYOUT_OUTPUT_TENSORS

class StorageLayout:
    """One memory layout per tensor, for a whole SDPA call.

    **`storage_flip` cannot express this, and is not being replaced by it.**
    That parameter is a single TRANSPOSITION applied IDENTICALLY to q/k/v/b: it
    reaches two of the six orders, never a 3-cycle, it cannot give two tensors
    different orders, and it says nothing at all about dout or about any tensor
    the kernel WRITES. `O`, `dK` and `dV` each carry their own buffer descriptor
    in the gfx950 kernels, so a layout test that leaves them contiguous leaves
    most of the descriptor arithmetic unexercised. Both spellings are accepted
    and every existing caller keeps passing `True`/`False`/`(x, y)`.

    Look up with `layout['q']`; an unnamed tensor defaults to `BHSD`.
    """

    def __init__(self, perms, case=None):
        self._perms = dict(perms)
        self._case = case

    @classmethod
    def round_robin(cls, case):
        """Case `case` of the round robin: tensor `t` gets layout `(case + t) % 6`.

        **A Latin square, not a cross product.** Ten tensors over six layouts is
        6**10 combinations, which is not a test suite. Rows are cases, columns
        are tensors in `LAYOUT_TENSOR_ORDER`, and the entry is
        `ALL_LAYOUTS[(case + column) % 6]`. Reading it down a column, each tensor
        visits all six layouts across the six cases, independently of every
        other tensor -- so no tensor is ever stuck on the contiguous layout that
        would hide a stride bug. Reading it across a row, all six layouts are
        live in the same kernel launch at once, which is the case that catches a
        descriptor reusing one tensor's strides for another.

        What it deliberately does NOT cover is a particular PAIR of layouts on a
        particular pair of tensors. The defect this exists for is per-descriptor
        arithmetic, one tensor at a time -- each descriptor reads its own
        tensor's three strides and nothing else -- so pairs would buy nothing
        for the 6**10 / 6 times the cost.
        """
        n = len(ALL_LAYOUTS)
        return cls({tname: ALL_LAYOUTS[(case + column) % n]
                    for column, tname in enumerate(LAYOUT_TENSOR_ORDER)},
                   case=case)

    @classmethod
    def from_storage_flip(cls, storage_flip):
        """The `storage_flip` spelling, as a `StorageLayout`.

        `dout` keeps `BHSD`: `storage_flip` predates it and has never reached
        it, and this is a translation, not an improvement.
        """
        x, y = storage_flip
        assert x != 3 and y != 3, 'Cannot storage_flip last dimension. Last dimension must be continuous'
        flipped = list(BHSD)
        flipped[x], flipped[y] = flipped[y], flipped[x]
        flipped = tuple(flipped)
        perms = {tname: flipped for tname in ('q', 'k', 'v', 'b')}
        perms['dout'] = BHSD
        return cls(perms)

    def __getitem__(self, tname):
        return self._perms.get(tname, BHSD)

    def outputs(self):
        """The `{tname: perm}` dict `AttentionExtraArgs.output_layouts` wants."""
        return {tname: self[tname] for tname in LAYOUT_OUTPUT_TENSORS}

    @property
    def case(self):
        return self._case

    def __repr__(self):
        body = ' '.join(f'{t}={LAYOUT_NAMES[self[t]]}' for t in LAYOUT_TENSOR_ORDER)
        return f'StorageLayout(case={self._case}, {body})'

def assert_layout(t, perm, min_pitch, tname):
    """`t` is stored in `perm`, with at least `min_pitch` between its rows.

    Both halves guard the same thing from different sides, and neither is
    redundant. The first catches a refactor that silently NORMALISES a layout --
    a `.contiguous()` slipped in somewhere, an `empty_like` on a view that is
    not dense -- which would leave the test green while testing nothing but
    BHSD. The second catches a refactor that silently COMPACTS the D axis: the
    8xD contract is what makes `ceil8(hdim)` columns the caller's, and a
    descriptor capped at the true row pitch is only meaningfully capped if that
    pitch is on the grid. Allocate at `hdim` instead of `ceil8(hdim)` and the
    cap starts doing the clipping the fix exists to stop, quietly.
    """
    if t is None:
        return
    got = layout_of(t)
    assert got == perm, (f'{tname} is stored as {LAYOUT_NAMES.get(got, got)}, '
                         f'expected {LAYOUT_NAMES.get(perm, perm)}: '
                         f'{tuple(t.shape)=} {t.stride()=}')
    pitch = min(t.stride(axis) for axis in range(3))
    assert pitch >= min_pitch, (f'{tname} has row pitch {pitch}, below the {min_pitch} '
                                f'its innermost axis needs: {tuple(t.shape)=} {t.stride()=}')

def calc_checksums(tensors):
    def checksum(t):
        if t is None:
            return None
        tensor_bytes = io.BytesIO()
        torch.save(t, tensor_bytes)
        return hashlib.blake2s(tensor_bytes.getvalue()).hexdigest()
    ret = [ checksum(t) for t in tensors ]
    # print(f'{ret=}')
    return ret

def allow_fp16_bf16_reduction_math_sdp(v : bool):
    if HAS_REDUCED_SDPA:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(v)

def sdpa_math(query, key, value, attn_mask=None, dropout_p=0.0, dropout_mask=None, is_causal=False, scale=None, enable_gqa=False):
    if TORCH_VERSION_TUPLE >= (2, 5):
        allow_fp16_bf16_reduction_math_sdp(True)
        retv = torch.ops.aten._scaled_dot_product_attention_math(query, key, value,
                                                                 dropout_p=dropout_p,
                                                                 is_causal=is_causal,
                                                                 attn_mask=attn_mask,
                                                                 scale=scale,
                                                                 dropout_mask=dropout_mask,
                                                                 enable_gqa=enable_gqa)
        allow_fp16_bf16_reduction_math_sdp(False)
        return retv
    else:
        return torch.ops.aten._scaled_dot_product_attention_math(query, key, value,
                                                                 dropout_p=dropout_p,
                                                                 is_causal=is_causal,
                                                                 attn_mask=attn_mask,
                                                                 scale=scale,
                                                                 dropout_mask=dropout_mask)


def windowed_attn_mask(seqlen_q, seqlen_k, window_left, window_right, *, dtype, device):
    """Additive mask for top-left aligned windowed attention.

    Query i attends key j iff i - window_left <= j <= i + window_right, which is
    the left_mask/right_mask pair the kernels build from parse_window.
    """
    i = torch.arange(seqlen_q, device=device)[:, None]
    j = torch.arange(seqlen_k, device=device)[None, :]
    keep = (j >= i - window_left) & (j <= i + window_right)
    mask = torch.zeros((seqlen_q, seqlen_k), dtype=dtype, device=device)
    return mask.masked_fill_(keep.logical_not(), float('-inf'))


def _reference_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    SPARSE_HEAD_SINCE = 5
    SPARSE_SEQ_SINCE = 5
    # attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if dropout_p > 0.0:
        if dropout_mask is not None:
            attn_weight.masked_fill_(dropout_mask.logical_not(), float("0.0"))
            value = value / (1 - dropout_p)
        else:
            # assert False, "TESTING dropout_mask code path"
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    else:
        # assert False, "TESTING dropout_mask code path"
        pass
    av = attn_weight @ value
    return av, attn_weight

default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-5}
default_rtol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}

# def get_rtol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
#     deviation = true_value - computed_value
#     deviation = torch.abs(deviation / true_value)
#     # Fill in the nans with the default rtol
#     torch.nan_to_num_(deviation, nan=default_rtol[computed_value.dtype])
#     return deviation.max().item()
#
# def get_atol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
#     # Low precision may yield NAN due to numerical instability
#     # See https://github.com/pytorch/pytorch/issues/116176 for a real-world example.
#     # Section 3 in https://arxiv.org/abs/2112.05682v3 explains how accelerated
#     # SDPA does not suffer from it.
#     deviation = torch.nan_to_num(true_value - computed_value)
#     atol = torch.abs(deviation).max().item()
#     return atol
#
# def get_tolerances(
#     true_value: torch.Tensor,
#     computed_value: torch.Tensor,
#     fudge_factor: Optional[float] = None,
# ) -> Tuple[float, float]:
#     """Returns the absolute and relative tolerances for comparing two tensors."""
#     fudge_factor = fudge_factor if fudge_factor is not None else 1.0
#     raw_atol = get_atol(true_value, computed_value)
#     raw_rtol = get_rtol(true_value, computed_value)
#
#     atol = fudge_factor * max(raw_atol, default_atol[computed_value.dtype])
#     rtol = fudge_factor * max(raw_rtol, default_rtol[computed_value.dtype])
#     # torch.isclose() has weird behavior around see:
#     # https://github.com/pytorch/pytorch/issues/102400
#     if rtol > 1e30:
#         rtol = default_rtol[computed_value.dtype]
#     return atol, rtol, raw_atol, raw_rtol

SdpaParams = namedtuple('SdpaParams', ['causal', 'sm_scale', 'dropout_p', 'dropout_mask'])

class SdpaContext(object):
    TENSOR_NAMES = ('q', 'k', 'v', 'b')

    def __init__(self, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, dtype,
                 bias_type=None, storage_flip=None, device='cuda', fillnan=False,
                 prng_seed=0x9be9_98d4_cf17_5339,
                 with_backward=True,
                 prime_hdim=None,
                 storage_layout=None,
                 ):
        self._set_real_device(device)
        self._prng_seed = prng_seed
        self._target_device = device
        # `int` or `(qk, vo)`; see `narrow_to_prime`. `D_HEAD` stays the
        # ALLOCATED width and must be the 8-multiple that covers it.
        self._prime_hdim = prime_hdim
        # A `StorageLayout`, or None to take the layout from `storage_flip`.
        # Mutually exclusive with it: the two say the same kind of thing, and
        # honouring both at once would only raise the question of which wins.
        assert storage_layout is None or storage_flip is None, \
            'storage_layout and storage_flip both given; they are two spellings of one thing'
        self._storage_layout = storage_layout
        self._input_shapes = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, dtype, bias_type, storage_flip, device, fillnan, prng_seed, with_backward)
        self._create_inputs()
        # Maximal value from tune_flash.py and table_tool.py --fudge_factor_tolerance 5.0
        # Note: Navi 3x is experimental and YMMV
        self.OUT_FUDGE_FACTOR = 3.0
        if dtype == torch.float32:
            self.OUT_FUDGE_FACTOR = 14.0
        if torch.version.hip:
            if 'gfx90a' in torch.cuda.get_device_properties(0).gcnArchName:
                self.OUT_FUDGE_FACTOR = 12.0
            if 'gfx1201' in torch.cuda.get_device_properties(0).gcnArchName:
                self.OUT_FUDGE_FACTOR = max(self.OUT_FUDGE_FACTOR, 10.0)
        if AOTRITON_TORCH_ONLY_USE_CPU:
            self.OUT_FUDGE_FACTOR = 12.0


    def _create_inputs(self):
        BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, dtype, bias_type, storage_flip, device, fillnan, prng_seed, with_backward = self._input_shapes
        if isinstance(N_HEADS, int):
            Q_HEADS = K_HEADS = N_HEADS
        else:
            Q_HEADS, K_HEADS = N_HEADS
        if isinstance(D_HEAD, int):
            HDIM_QK = HDIM_VO = D_HEAD
        else:
            HDIM_QK, HDIM_VO = D_HEAD
        qdims = (BATCH, Q_HEADS, seqlen_q, HDIM_QK)
        kdims = (BATCH, K_HEADS, seqlen_k, HDIM_QK)
        vdims = (BATCH, K_HEADS, seqlen_k, HDIM_VO)
        odims = (BATCH, Q_HEADS, seqlen_q, HDIM_VO)
        def round_to_8x(n):
            return 8 * cdiv(n, 8)
        bdims = (BATCH, Q_HEADS, seqlen_q, round_to_8x(seqlen_k))
        # One code path for both spellings: `storage_flip` becomes the
        # `StorageLayout` it always meant, and `alloc_with_layout` does what the
        # dims-permute-then-`torch.transpose` pair below used to do. The
        # translation is exact -- a transposition is its own inverse, so the
        # `permute()` it emits IS the old `transpose()` -- down to the order the
        # generator is drawn in, so no existing case changes value.
        if self._storage_layout is not None:
            layout = self._storage_layout
        elif storage_flip is not None:
            layout = StorageLayout.from_storage_flip(storage_flip)
        else:
            layout = StorageLayout({})
        # print(f'{qdims=}')
        # print(f'{kdims=}')
        # print(f'{vdims=}')
        # print(f'{bdims=}')
        # q = torch.empty(qdims, dtype=dtype, device=device).normal_(mean=0., std=0.5)
        # k = torch.empty(kdims, dtype=dtype, device=device).normal_(mean=0., std=0.5)
        # v = torch.empty(vdims, dtype=dtype, device=device).normal_(mean=0., std=0.5)
        g = torch.Generator(device=self._real_device)
        g.manual_seed(self._prng_seed)
        def rng(dims, tname):
            return alloc_with_layout(dims, layout[tname], dtype=dtype,
                                     device=self._real_device, generator=g)
        q = rng(qdims, 'q')
        k = rng(kdims, 'k')
        v = rng(vdims, 'v')
        if bias_type is None or bias_type == 0:
            b = None
        elif bias_type == 'matrix' or bias_type == 1:
            # b = torch.empty(bdims, dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
            # Allocated at `round_to_8x(seqlen_k)` and narrowed, so the bias
            # carries the slack the kernel's 8-wide KV loads read past the end
            # of a ragged row. That is a property of the LAST axis and survives
            # any permutation of the other three, because the last axis is the
            # one that never permutes.
            b = rng(bdims, 'b')
            b = b[:, :, :, :seqlen_k]
            # b = b.expand(BATCH, Q_HEADS, b.shape[0], b.shape[1])
        else:
            assert False, f'Unsupported bias_type {bias_type}'
        dout = rng(odims, 'dout') if with_backward else None

        # **The narrowing step, after every allocation and before anything
        # reads these.** The reference is built from `dev_tensors`, so it has to
        # see the same columns the kernel does -- narrowing here rather than at
        # the launch keeps the two from disagreeing about what the head dim is.
        # Outputs are narrowed on the other side, at their own allocation; see
        # `AttentionExtraArgs.prime_hdim`.
        if self._prime_hdim is not None:
            pq, pv = ((self._prime_hdim, self._prime_hdim)
                      if isinstance(self._prime_hdim, int) else self._prime_hdim)
            q = narrow_to_prime(q, pq)
            k = narrow_to_prime(k, pq)
            v = narrow_to_prime(v, pv)
            dout = narrow_to_prime(dout, pv)

        self.dev_tensors = ( q, k, v, b )
        self.ddev_tensors = tuple([dout])

    '''
    Create Tensors that will be kept b/w forward and backward pass
    '''
    def create_ctx_tensors(self):
        q, k, v, b = self.dev_tensors
        o = torch.empty((q.shape[0], q.shape[1], q.shape[2], v.shape[3]), device=q.device, dtype=q.dtype)
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        self.ctx_tensors = (o, M)

    def create_bwd_tensors(self):
        q, k, v, b = self.dev_tensors
        o, L = self.ctx_tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        db = torch.empty_like(b) if b is not None else None
        delta = torch.empty_like(L)
        self.bwd_tensors = (dq, dk, dv, db, delta)

    @staticmethod
    def fillnan(tensors):
        for t in tensors:
            if t is None:
                continue
            t.fill_(float('nan'))

    @property
    def dtype(self):
        return self.dev_tensors[0].dtype

    @property
    def hdim(self):
        return self.dev_tensors[0].shape[-1]

    @property
    def is_hdim_NPOT_optimized(self):
        def is_power_of_two(n: int) -> bool:
            return (n & (n - 1) == 0) and n != 0
        hdim = self.hdim
        return not is_power_of_two(hdim)

    @property
    def seqlen_q(self):
        q, k, v, b = self.dev_tensors
        seqlen_q = q.shape[2]
        return seqlen_q

    @property
    def seqlen_k(self):
        q, k, v, b = self.dev_tensors
        seqlen_k = k.shape[2]
        return seqlen_k

    @property
    def ref_device(self):
        return self.ref_tensors[0].device

    def _set_real_device(self, device):
        '''
        The device the dev tensors actually live on, which is where a 'cuda'
        reference goes. EVERY __init__ must call this: the subclasses do not
        chain to SdpaContext.__init__, so anything set only there is missing on
        a VarlenSdpaContext, and create_ref_inputs' policy reads it.
        '''
        self._real_device = 'cpu' if AOTRITON_TORCH_ONLY_USE_CPU else device

    @staticmethod
    def clone_tensor(t, dtype, device=None):
        if t is None:
            return None
        return t.clone().detach().to(dtype=dtype, device=device).requires_grad_(t.requires_grad)

    @staticmethod
    def clone_tensor_tuple(in_tensors, dtype, device=None):
        return tuple([SdpaContext.clone_tensor(t, dtype=dtype, device=device) for t in in_tensors])

    def create_ref_inputs(self, target_device_policy=None, target_device=None):
        '''
        Device policy is not actual device, it's which class of device we should use
            a) cpu -> device = cpu
            b) cuda -> device = self._real_device, friendly to gpu-lease
            c) default -> use faulty reference detection logic baked below

        Device policy precedence:
            1. AOTRITON_TORCH_ONLY_USE_CPU      # torch is cpu only
            2. AOTRITON_REF_DEVICE_OPTION       # env var
            3. caller's target_device_policy    # usually configured by adiff's CPUREF
            4. default
        '''
        if AOTRITON_TORCH_ONLY_USE_CPU:
            ref_device_policy = 'cpu'
        elif AOTRITON_REF_DEVICE_OPTION_SET:
            ref_device_policy = AOTRITON_REF_DEVICE_OPTION
        elif target_device_policy is not None:
            ref_device_policy = target_device_policy
        else:
            ref_device_policy = 'default'

        def apply_policy():
            if target_device is not None:
                return target_device
            if (ref_device_policy == 'default' and ROCM_IS_7_14
                and self.dtype == torch.bfloat16
                and (self.seqlen_q, self.seqlen_k) == (257, 2081)):
                # torch's bf16 bmm leaves 274733 output elements UNWRITTEN on gfx950
                # under ROCm 7.14 -- uninitialized memory, not a wrong computation.
                # Repro needs neither AOTriton nor FlyDSL:
                #   torch.bmm(torch.rand(15, 257, 16, device='cuda', dtype=torch.bfloat16),
                #             torch.rand(15, 16, 2081, device='cuda', dtype=torch.bfloat16))
                # The resulting nan in ref_error is the LUCKY case, because a nan
                # threshold fails every tensor at once and gets noticed; when the
                # stale bytes decode as plausible floats the oracle is quietly wrong
                # and the test passes. So this shape is WHERE WE SAW IT, not the
                # trigger surface -- the failing set moves between runs. Widening to
                # every bf16 case is sound and far too slow; catching the quiet ones
                # needs _validate to reject a nan REFERENCE, which it does not yet.
                # ROCm-gated so it retires itself.
                return 'cpu'
            if ref_device_policy == 'default' and TORCH_GE_2_7:
                return self._real_device  # Known softmax issues have been fixed in 2.7
            if ref_device_policy == 'default':
                seqlen_k = self.seqlen_k
                hdim = self.hdim
                '''
                test_gqa[False-1.2-dtype0-0.5-False-579-2048-203-N_HEADS1-4]
                triggers GPU segfault when testing with -k 'test_gqa[False-1.2-'
                (Cannot be reproduced independently)
                '''
                if seqlen_k == 579:
                    return 'cpu'
                '''
                Shader _ZN2at6native12_GLOBAL__N_119cunn_SoftMaxForwardILi2EdddNS1_22SoftMaxForwardEpilogueEEEvPT2_PKT0_i causes Segfault
                for Case test_op_bwd[False-0.0-dtype2-0.0-False-587-64-8-4-4], but cannot be reproduced by running this individual UT.
                Avoiding running it on GPU for now
                '''
                if self.dtype == torch.float32:
                    if seqlen_k == 587 or hdim % 16 != 0:
                        return 'cpu'
                return self._real_device
            if ref_device_policy == 'cuda':
                return self._real_device
            if ref_device_policy == 'cpu':
                return 'cpu'
            assert False, f'Unknown ref_device_policy value {ref_device_policy}. Allowed choices "default" "cpu" "cuda"'

        ref_device = apply_policy()
        self.create_ref_inputs_with_device(ref_device)

    def create_ref_inputs_with_device(self, ref_device):
        dtype = self.dtype
        hp_dtype = torch.float64 if dtype == torch.float32 else torch.float32
        self.ref_tensors = self.clone_tensor_tuple(self.dev_tensors, dtype=hp_dtype, device=ref_device)
        self.lp_ref_tensors = self.clone_tensor_tuple(self.dev_tensors, dtype=dtype, device=ref_device)

    @staticmethod
    def _require_grads(tensors, skip_dq=False, skip_dk_dv=False, skip_db=False):
        q, k, v, b = tensors
        if not skip_dq:
            q.requires_grad_()
        if not skip_dk_dv:
            k.requires_grad_()
            v.requires_grad_()
        if not skip_db:
            assert b is not None
            b.requires_grad_()

    def set_require_grads(self, skip_dq=False, skip_dk_dv=False, skip_db=False):
        self._require_grads(self.dev_tensors, skip_dq=skip_dq, skip_dk_dv=skip_dk_dv, skip_db=skip_db)
        self._require_grads(self.ref_tensors, skip_dq=skip_dq, skip_dk_dv=skip_dk_dv, skip_db=skip_db)
        self._require_grads(self.lp_ref_tensors, skip_dq=skip_dq, skip_dk_dv=skip_dk_dv, skip_db=skip_db)

    def _compute_fudge_factors(self, p : SdpaParams):
        ref_q, ref_k, ref_v, ref_b = self.ref_tensors
        dtype = self.dtype
        seqlen_k = self.seqlen_k
        seqlen_q = self.seqlen_q

        # seqlen_k_fudge_factor = 1.0 if seqlen_k < 1024 else 2.0
        # seqlen_k_fudge_factor = seqlen_k_fudge_factor if seqlen_k < 8192 else 4.0
        # dropout_fudge_factor = 1.0 if p.dropout_p == 0.0 else 2.0
        # query_fudge_factor = 8 * dropout_fudge_factor * seqlen_k_fudge_factor # TODO: Investigate why grad_q needs larger tolerances
        # key_fudge_factor = 8 * dropout_fudge_factor
        # value_fudge_factor = 7
        # bias_fudge_factor = 12

        # Maximal value from tune_flash.py and table_tool.py --fudge_factor_tolerance 5.0
        # Note: Navi 3x is experimental and YMMV
        query_fudge_factor = 148.0  # NPOT
        key_fudge_factor = 48.0
        # value_fudge_factor = 16.0
        value_fudge_factor = 36.0
        bias_fudge_factor = 17.0
        # print(f'{torch.cuda.get_device_properties(0).gcnArchName=}')
        if torch.version.hip:
            if 'gfx90a' in torch.cuda.get_device_properties(0).gcnArchName:
                query_fudge_factor = max(query_fudge_factor, 130.0 if isinstance(self, VarlenSdpaContext) else 80.0)
                key_fudge_factor = 500.0 if self.is_hdim_NPOT_optimized else 340.0
                bias_fudge_factor = 45.0 if self.is_hdim_NPOT_optimized else 36.0
            # Navi31 needs larger factors
            if 'gfx1100' in torch.cuda.get_device_properties(0).gcnArchName:
                query_fudge_factor = max(query_fudge_factor, 768.0 if p.dropout_p > 0.0 else 320.0)
        if AOTRITON_TORCH_ONLY_USE_CPU:
            query_fudge_factor = 128.0
            key_fudge_factor = 330.0
            bias_fudge_factor = 36.0
            # value_fudge_factor = 36.0
        if dtype == torch.float32:
            key_fudge_factor = 180.0
            value_fudge_factor = 50.0
            bias_fudge_factor = 24.0
        return (query_fudge_factor, key_fudge_factor, value_fudge_factor, bias_fudge_factor)

    @staticmethod
    def _compute_ref_forward(ref_tensors, p : SdpaParams):
        ref_q, ref_k, ref_v, ref_b = ref_tensors
        num_head_q = ref_q.shape[1]
        num_head_k = ref_k.shape[1]
        num_head_v = ref_v.shape[1]
        assert num_head_k == num_head_v
        assert num_head_q % num_head_k == 0
        enable_gqa = num_head_q != num_head_k
        dropout_mask = p.dropout_mask if p.dropout_mask is None else p.dropout_mask.to(device=ref_q.device)
        assert not (p.causal and ref_b is not None), \
            'causal/window with an explicit bias is undefined'
        if isinstance(p.causal, tuple):
            ref_b = windowed_attn_mask(ref_q.shape[2], ref_k.shape[2], *p.causal,
                                       dtype=ref_q.dtype, device=ref_q.device)
        # _scaled_dot_product_attention_math seems also working for nested tensor
        ref_out, ref_mask = sdpa_math(ref_q, ref_k, ref_v,
                                      dropout_p=p.dropout_p,
                                      is_causal=p.causal if not isinstance(p.causal, tuple) else False,
                                      attn_mask=ref_b,
                                      scale=p.sm_scale,
                                      dropout_mask=dropout_mask,
                                      enable_gqa=enable_gqa)
        return (ref_out, ref_mask)

    def compute_ref_forward(self, p : SdpaParams):
        self.fudge_factors = self._compute_fudge_factors(p)
        self.refout_tensors = self._compute_ref_forward(self.ref_tensors, p)
        self.lp_refout_tensors = self._compute_ref_forward(self.lp_ref_tensors, p)
        return self.lp_refout_tensors

    @staticmethod
    def _compute_backward(in_tensors, out, dout):
        q, k, v, b = in_tensors
        out.backward(dout.to(device=out.device, dtype=out.dtype))
        dq, q.grad = None if not q.requires_grad else q.grad.clone(), None
        dk, k.grad = None if not k.requires_grad else k.grad.clone(), None
        dv, v.grad = None if not v.requires_grad else v.grad.clone(), None
        if b is None or not b.requires_grad:
            db = None
        else:
            db, b.grad = b.grad.clone(), None
        return (dq, dk, dv, db)

    # Note: this follows pytorch's testing approach and expects low precision dout
    def compute_backward(self, out, _in_dout, *, ref_only=False):
        dout = _in_dout if _in_dout is not None else self.ddev_tensors[0]
        self.dref_tensors = self._compute_backward(self.ref_tensors, self.refout_tensors[0], dout)
        self.lp_dref_tensors = self._compute_backward(self.lp_ref_tensors, self.lp_refout_tensors[0], dout)
        if not ref_only:
            self.dout_tensors = self._compute_backward(self.dev_tensors, out, dout)

    def _mask_tensor(self, out, ref, lp_ref, tname: str):
        pass

    @staticmethod
    def _validate(out, ref, lp_ref, fudge_factor, tname,
                  *,
                  return_target_fudge_factors=False,
                  adiff=None):
        if out is None and ref is None:
            return True, 0.0, 1.0
        # atol, rtol, raw_atol, raw_rtol = get_tolerances(ref, lp_ref, fudge_factor)
        assert out is not None, f'd{tname} is none'
        assert ref is not None, f'd{tname}_ref is none'
        # print(f'{out=}')
        # print(f'{ref=}')
        def lmax(x) -> float:
            return x.abs().max().item()
        max_adiff = test_error = lmax(ref - out.to(device=ref.device))
        ref_error = lmax(ref - lp_ref)
        if math.isnan(test_error) and not math.isnan(ref_error):
            # TODO: More detailed feedback
            reason = f"Tensor {tname} has NaN output but not NaN reference"
            # print(f'{max_adiff=} {test_error=} {tname=}')
            return False, max_adiff, None
        # print(f"{adiff=} {test_error=}")
        if adiff is not None:
            valid = test_error < (adiff * 2.0)
        else:
            atol = default_atol[torch.float32]
            threshold = max(atol, ref_error * fudge_factor)
            valid = test_error <= threshold
        # tft = test_error / ref_error if ref_error * fudge_factor > atol else 1.0
        tft = test_error / ref_error if not valid else 1.0
        if not valid:
            pass
            # print(f'For {tname}, Consider bump fudge_factor to {tft} = {test_error=} / {ref_error=}. So that {test_error=} < {threshold=} = max({atol=}, {ref_error=} * {tft=})')
        if return_target_fudge_factors:
            return valid, max_adiff, tft
        else:
            return valid, max_adiff, None

    def validate_with_reference(self, out, grads,
                                *,
                                no_forward=False,
                                no_backward=False,
                                return_target_fudge_factors=False,
                                use_adiff_entry=None):
        if no_forward:
            out_allclose, out_adiff, tft = True, None, None
        else:
            use_adiff = None if use_adiff_entry is None else use_adiff_entry["adiff"]
            self._mask_tensor(out, self.refout_tensors[0], self.lp_refout_tensors[0], 'out')
            out_allclose, out_adiff, tft = self._validate(out,
                                                          self.refout_tensors[0],
                                                          self.lp_refout_tensors[0],
                                                          self.OUT_FUDGE_FACTOR,
                                                          'out',
                                                          return_target_fudge_factors=return_target_fudge_factors,
                                                          adiff=use_adiff)
        target_fudge_factors = {'out' : tft}
        if no_backward:
            if return_target_fudge_factors:
                return out_allclose, out_adiff, [], [], target_fudge_factors
            else:
                return out_allclose, out_adiff, [], []
        grads_allclose = []
        grads_adiff = []
        use_adiffs = [None] * len(grads) if use_adiff_entry is None else use_adiff_entry["grads_adiff"]
        # print(f'using {self.fudge_factors=}')
        for grad, ref, lp_ref, fudge_factor, tname, adiff in zip(grads, self.dref_tensors, self.lp_dref_tensors, self.fudge_factors, self.TENSOR_NAMES, use_adiffs):
            self._mask_tensor(grad, ref, lp_ref, tname)
            allclose, adiff, tft = self._validate(grad,
                                                  ref,
                                                  lp_ref,
                                                  fudge_factor,
                                                  tname,
                                                  return_target_fudge_factors=return_target_fudge_factors,
                                                  adiff=adiff)
            grads_allclose.append(allclose)
            grads_adiff.append(adiff)
            # if math.isnan(adiff):
            #     print(f'{adiff=} {grads_adiff=} {tname=}')
            target_fudge_factors[tname] = tft
        if return_target_fudge_factors:
            return out_allclose, out_adiff, grads_allclose, grads_adiff, target_fudge_factors
        else:
            return out_allclose, out_adiff, grads_allclose, grads_adiff

    def display_validation_results(self, tri_out, is_allclose, adiff, grads_allclose, grads_adiff):
        q, k, v, b = self.dev_tensors
        def TO(ref_tensor):
            return ref_tensor.to(device=q.device, dtype=dtype)
        dtype = q.dtype
        SPARSE_HEAD_SINCE = 1
        SPARSE_SEQ_SINCE = 1
        ref_out = self.lp_refout_tensors[0]
        if not is_allclose:
            err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_out) - tri_out)).cpu().numpy(), ref_out.shape)
            print(f'{err_idx=}')
            print(f'{tri_out[err_idx]=}')
            print(f'{ref_out[err_idx]=}')
            print(f'{tri_out[0, 0, :4, :16]=}')
            print(f'{ref_out[0, 0, :4, :16]=}')
        if not grads_allclose:
            # Forward-only validation (validate_with_reference(no_backward=True),
            # as SKIP_BWD uses): there are no gradients to diagnose, and the
            # unpack below would raise before printing anything useful.
            return
        dq_allclose, dk_allclose, dv_allclose, db_allclose = grads_allclose
        tri_dq, tri_dk, tri_dv, tri_db = self.dout_tensors
        ref_dq, ref_dk, ref_dv, ref_db = self.dref_tensors
        if not dv_allclose:
            err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dv) - tri_dv)).cpu().numpy(), ref_dv.shape)
            print(f'{q.shape=} {q.stride()=} {q.dtype=}')
            print(f'{k.shape=} {k.stride()=} {k.dtype=}')
            print(f'{v.shape=} {v.stride()=} {v.dtype=}')
            # print(f'{q[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
            # print(f'{k[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
            # print(f'{v[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
            # print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
            # print(f'{dropout_mask.shape=}')
            print(f'{err_idx=}')
            print(f'{tri_dv[err_idx]=}')
            print(f'{ref_dv[err_idx]=}')
            print(f'{torch.isnan(ref_dv).any()=}')
            '''
            any_nan = torch.isnan(ref_dv).any()
            if any_nan:
                torch.set_printoptions(linewidth=200)
                print(f'{q=}')
                print(f'{k=}')
                print(f'{v=}')
                print(f'{dropout_p=}')
                print(f'{causal=}')
                print(f'{sm_scale=}')
            if seqlen_q <= 16:
                # torch.set_printoptions(linewidth=200, threshold=4096)
                print(f'{tri_dk[0,0]=}')
                print(f'{ref_dk[0,0]=}')
                print(f'{tri_dv[0,0]=}')
                print(f'{ref_dv[0,0]=}')
                # print(f'{tri_dq[0,0]=}')
                # print(f'{ref_dq[0,0]=}')
            '''

        if dv_allclose and not dk_allclose:
            print(f'{tri_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
            print(f'{ref_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
            print(f'{tri_dk[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
            print(f'{ref_dk[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
            err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dk) - tri_dk)).cpu().numpy(), ref_dk.shape)
            print(f'{err_idx=}')
            print(f'{tri_dk[err_idx]=} {ref_dk[err_idx]=} error = {torch.abs(tri_dk[err_idx] - ref_dk[err_idx])}')
            # print(f'{tri_dk[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]/ref_dk[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
            # print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')

        if dk_allclose and dv_allclose and not dq_allclose:
            err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dq) - tri_dq)).cpu().numpy(), ref_dq.shape)
            print(f'{err_idx=}')
            print(f'{tri_dq[err_idx]=} {ref_dq[err_idx]=} error = {torch.abs(tri_dq[err_idx] - ref_dq[err_idx])}')

        if dk_allclose and dv_allclose and dq_allclose and not db_allclose:
            err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_db) - tri_db)).cpu().numpy(), ref_db.shape)
            print(f'{err_idx=}')
            print(f'{tri_db[err_idx]=} {ref_db[err_idx]=} error = {torch.abs(tri_db[err_idx] - ref_db[err_idx])}')


    def save_integrity_checksum(self):
        self._tensor_checksums = {}
        self._tensor_checksums['dev'] = calc_checksums(self.dev_tensors)
        self._tensor_checksums['ref'] = calc_checksums(self.ref_tensors)
        self._tensor_checksums['lp_ref'] = calc_checksums(self.lp_ref_tensors)
        if hasattr(self, 'dref_tensors'):
            self._tensor_checksums['dref'] = calc_checksums(self.dref_tensors)
            self._tensor_checksums['lp_dref'] = calc_checksums(self.lp_dref_tensors)

    def check_integrity(self):
        for key in ['dev', 'ref', 'lp_ref', 'dref', 'lp_dref']:
            prop_key = f'{key}_tensors'
            if not hasattr(self, prop_key):
                continue
            tensors = getattr(self, prop_key)
            saved_checksums = self._tensor_checksums[key]
            current_checksums = calc_checksums(tensors)
            if saved_checksums != current_checksums:
                # print(f'who={key} {current_checksums=} != {saved_checksums=}')
                return False, key
        return True, None

    def restore_integrity(self, who, sdpa_params):
        if who == 'dev':
            todo = [ 'input', 'fwd_ref', 'bwd_ref' ]
        if who in [ 'ref', 'lp_ref' ]:
            todo = [ 'fwd_ref', 'bwd_ref' ]
        if who in [ 'dref', 'lp_dref' ]:
            todo = [ 'fwd_ref', 'bwd_ref' ]
        if 'input' in todo:
            del self.dev_tensors
            del self.ddev_tensors
            self._create_inputs()
            q, k, v, b = self.dev_tensors
            self.set_require_grads(skip_db=True if b is None else False)
        if 'fwd_ref' in todo:
            del self.ref_tensors
            del self.lp_ref_tensors
            self.create_ref_inputs(target_device=self._real_device)
            self.compute_ref_forward(sdpa_params)
        if hasattr(self, 'dref') and 'bwd_ref' in todo:
            del self.dref_tensors
            del self.lp_dref_tensors
            self.compute_backward(None, None, sdpa_params)
            # print('self.compute_backward')

class VarlenSdpaContext(SdpaContext):
    TENSOR_NAMES = ('q', 'k', 'v', 'b')

    @staticmethod
    def _rng_varlen_tensor(num_heads, seqlens, head_dim, dtype, device, packed=False):
        # Note: do NOT use nested tensor here
        #       PyTorch's UT can use nested tensor because PyTorch preprocessed
        #       the input nested tensors before sending them to its SDPA
        #       backends (See sdpa_nested_preprocessing in
        #       aten/src/ATen/native/nested/cuda/NestedTensorTransformerUtils.cpp)
        #
        #       AOTriton works in the same level as PyTorch's SDPA backends and
        #       its UT should generate the tensor in the preprocessed format directly.
        dims = (1, np.sum(seqlens), num_heads, head_dim)
        return torch.rand(*dims, dtype=dtype, device=device).transpose(1, 2)
        '''
        def _size(seqlen):
            return (seqlen, num_heads, head_dim) if not packed else (seq_len[i], 3 * num_heads * head_dim)

        return torch.nested.nested_tensor([
            torch.rand(_size(seqlen), device=device, dtype=dtype, requires_grad=True)
            for seqlen in seqlens])
        '''

    def __init__(self, N_HEADS, D_HEAD, seqlens_q, seqlens_k, dtype, device='cuda'):
        self._set_real_device(device)
        if isinstance(D_HEAD, int):
            HDIM_QK = HDIM_VO = D_HEAD
        else:
            HDIM_QK, HDIM_VO = D_HEAD
        q  = self._rng_varlen_tensor(N_HEADS, seqlens_q, HDIM_QK, dtype, device)
        k  = self._rng_varlen_tensor(N_HEADS, seqlens_k, HDIM_QK, dtype, device)
        v  = self._rng_varlen_tensor(N_HEADS, seqlens_k, HDIM_VO, dtype, device)
        b = None
        self.dev_tensors = (q, k, v, b)
        self.OUT_FUDGE_FACTOR = 3
        if dtype == torch.float32:
            self.OUT_FUDGE_FACTOR = 14.0
        if torch.version.hip:
            if 'gfx90a' in torch.cuda.get_device_properties(0).gcnArchName:
                self.OUT_FUDGE_FACTOR = 12.0
        if AOTRITON_TORCH_ONLY_USE_CPU:
            self.OUT_FUDGE_FACTOR = 12.0
        self._seqlens_q = np.array(seqlens_q)
        self._seqlens_k = np.array(seqlens_k)

    # Not perfect but fits our needs.
    @property
    def seqlen_k(self):
        return np.max(self._seqlens_k)

    def create_ctx_tensors(self):
        q, k, v, b = self.dev_tensors
        o = torch.empty((q.shape[0], q.shape[1], q.shape[2], v.shape[3]), device=q.device, dtype=q.dtype)
        M = torch.empty((q.shape[1], int(np.sum(self._seqlens_q))), device=q.device, dtype=torch.float32)
        self.ctx_tensors = (o, M)

    @staticmethod
    def _gen_seqaccess(seqlens_q, seqlens_k):
        seqlen_q_start = 0
        seqlen_k_start = 0
        for i, (seqlen_q, seqlen_k) in enumerate(zip(seqlens_q, seqlens_k)):
            yield i, (int(seqlen_q), int(seqlen_k)), (seqlen_q_start, seqlen_k_start)
            seqlen_q_start += int(seqlen_q)
            seqlen_k_start += int(seqlen_k)

    def _compute_ref_forward_varlen(self, ref_tensors, seqlens_q, seqlens_k, p : SdpaParams):
        packed_ref_q, packed_ref_k, packed_ref_v, _ = ref_tensors
        q, k, v, _ = ref_tensors
        num_head_q = packed_ref_q.shape[1]
        num_head_k = packed_ref_k.shape[1]
        num_head_v = packed_ref_v.shape[1]
        packed_dropout_mask = p.dropout_mask if p.dropout_mask is None else p.dropout_mask.to(device=packed_ref_q.device)
        ref_out_array = torch.zeros((q.shape[0], q.shape[1], q.shape[2], v.shape[3]), device=q.device, dtype=q.dtype)
        ref_mask_array = []
        print(f'REF {seqlens_q=} {seqlens_k=}')
        for i, (seqlen_q, seqlen_k), (seqlen_q_start, seqlen_k_start) in self._gen_seqaccess(seqlens_q, seqlens_k):
            ref_q = packed_ref_q[0, :, seqlen_q_start:seqlen_q_start+seqlen_q, :]
            ref_k = packed_ref_k[0, :, seqlen_k_start:seqlen_k_start+seqlen_k, :]
            ref_v = packed_ref_v[0, :, seqlen_k_start:seqlen_k_start+seqlen_k, :]
            dropout_mask = packed_dropout_mask[i, :, :, :] if packed_dropout_mask is not None else None
            print(f'REF {seqlen_q_start=} {seqlen_q_start+seqlen_q=} {ref_q.shape=} {ref_k.shape=} {ref_v.shape=}')
            print(f'REF {ref_q.stride()=}')
            if dropout_mask is not None:
                print(f'REF {packed_dropout_mask.shape=}')
                print(f'REF {dropout_mask.shape=}')
                dropout_mask = dropout_mask[:, :seqlen_q, :seqlen_k]  # Trim to actual seqlen
                print(f'REF CLAMPED {dropout_mask.shape=}')
                # print(f'REF {dropout_mask=}')
            ref_out, ref_mask = torch.ops.aten._scaled_dot_product_attention_math(ref_q, ref_k, ref_v,
                                                                        dropout_p=p.dropout_p,
                                                                        is_causal=p.causal,
                                                                        scale=p.sm_scale,
                                                                        dropout_mask=dropout_mask)
            ref_out_array[0, :, seqlen_q_start:seqlen_q_start+seqlen_q, :] = ref_out
            ref_mask_array.append(ref_mask)
        return ref_out_array, None

    def compute_ref_forward(self, p : SdpaParams):
        self.fudge_factors = self._compute_fudge_factors(p)
        self.refout_tensors = self._compute_ref_forward_varlen(self.ref_tensors, self._seqlens_q, self._seqlens_k, p)
        self.lp_refout_tensors = self._compute_ref_forward_varlen(self.lp_ref_tensors, self._seqlens_q, self._seqlens_k, p)
        return self.lp_refout_tensors

class PaddedVarlenSdpaContext(VarlenSdpaContext):
    '''
    PaddedVarlenSdpaContext uses regular BHSD shape, where S is max(seqlens) and data are padded.
    '''
    @staticmethod
    def _rng_varlen_tensor(num_heads, seqlens, head_dim, dtype, device, packed=False):
        B = len(seqlens)
        S = int(np.max(seqlens))
        dims = (B, num_heads, S, head_dim)
        # TODO: fill nan to padded sequences
        return torch.rand(*dims, dtype=dtype, device=device)

    def create_ctx_tensors(self):
        q, k, v, b = self.dev_tensors
        o = torch.empty((q.shape[0], q.shape[1], q.shape[2], v.shape[3]), device=q.device, dtype=q.dtype)
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        self.ctx_tensors = (o, M)

    def _compute_ref_forward_varlen(self, ref_tensors, seqlens_q, seqlens_k, p : SdpaParams):
        q, k, v, b = self.dev_tensors
        packed_ref_q, packed_ref_k, packed_ref_v, _ = ref_tensors
        packed_dropout_mask = p.dropout_mask if p.dropout_mask is None else p.dropout_mask.to(device=packed_ref_q.device)
        num_head_q = packed_ref_q.shape[1]
        num_head_k = packed_ref_k.shape[1]
        num_head_v = packed_ref_v.shape[1]
        enable_gqa = num_head_q != num_head_k
        ref_out_array = torch.zeros((q.shape[0], q.shape[1], q.shape[2], v.shape[3]), device=q.device, dtype=q.dtype)
        ref_mask_array = []
        print(f'REF {seqlens_q=} {seqlens_k=}')
        for i, (seqlen_q, seqlen_k) in enumerate(zip(seqlens_q, seqlens_k)):
            ref_q = packed_ref_q[i, :, :seqlen_q, :]
            ref_k = packed_ref_k[i, :, :seqlen_k, :]
            ref_v = packed_ref_v[i, :, :seqlen_k, :]
            dropout_mask = packed_dropout_mask[i, :, :, :] if packed_dropout_mask is not None else None
            if dropout_mask is not None:
                dropout_mask = dropout_mask[:, :seqlen_q, :seqlen_k]  # Trim to actual seqlen
                # print(f'REF {dropout_mask=}')
            ref_out, ref_mask = sdpa_math(ref_q, ref_k, ref_v,
                                          dropout_p=p.dropout_p,
                                          is_causal=p.causal,
                                          scale=p.sm_scale,
                                          dropout_mask=dropout_mask,
                                          enable_gqa=enable_gqa)
            print(f'REF {seqlen_q=} {ref_out.shape=}')
            ref_out_array[i, :, :seqlen_q, :] = ref_out
        return ref_out_array, None

    def _get_valid_seqlens(self, tname: str):
        if tname in ['out', 'q']:
            return self._seqlens_q
        if tname in ['k', 'v']:
            return self._seqlens_k
        if tname == 'b':
            return None
        assert False, f'Unknown tensor name {tname}'

    def _mask_tensor(self, out, ref, lp_ref, tname: str):
        valid_seqlens = self._get_valid_seqlens(tname)
        if valid_seqlens is None:
            return
        for b, seqlen in enumerate(valid_seqlens):
            out[b, :, seqlen:, :].fill_(0)
            ref[b, :, seqlen:, :].fill_(0)
            lp_ref[b, :, seqlen:, :].fill_(0)

class StridedVarlenSdpaContext(VarlenSdpaContext):
    '''
    seqlens_q/k passed to StridedVarlenSdpaContext.__init__() are tuples (seqlens_q/k, padlens_q/k)
    Hence _rng_varlen_tensor and _compute_ref_forward_varlen should be able to handle them
    However, fortunately np.sum works well for _rng_varlen_tensor and create_ctx_tensors
    '''
    @property
    def seqlen_k(self):
        return int(np.max(self._seqlens_k[0]))

    @staticmethod
    def _gen_seqaccess(seqlens_q, seqlens_k):
        seqlens_q, padlens_q = seqlens_q
        seqlens_k, padlens_k = seqlens_k
        seqlen_q_start = 0
        seqlen_k_start = 0
        for i, (seqlen_q, seqlen_k, padlen_q, padlen_k) in enumerate(zip(seqlens_q, seqlens_k, padlens_q, padlens_k)):
            yield i, (int(seqlen_q), int(seqlen_k)), (seqlen_q_start, seqlen_k_start)
            seqlen_q_start += int(seqlen_q + padlen_q)
            seqlen_k_start += int(seqlen_k + padlen_k)

    def _mask_tensor(self, out, ref, lp_ref, tname: str):
        def gen():
            seqlens_q, padlens_q = self._seqlens_q
            seqlens_k, padlens_k = self._seqlens_k
            if tname in ['out', 'q']:
                yield from zip(seqlens_q, padlens_q)
            elif tname in ['k', 'v']:
                yield from zip(seqlens_k, padlens_k)
            elif tname == 'b':
                return None
            else:
                assert False, f'Unknown tensor name {tname}'
        def gen_access():
            start = 0
            for i, (seqlen, padlen) in enumerate(gen()):
                yield start+seqlen, start+seqlen+padlen
                start += seqlen + padlen
        for start, end in gen_access():
            for t in [out, ref, lp_ref]:
                # print(f'MASKING {tname=} {start}:{end}')
                t[0,:,start:end,:].fill_(0)

class SdpaContextFromNPZ(SdpaContext):
    def __init__(self, fn, dtype, device='cuda'):
        self._set_real_device(device)
        d = np.load(fn)
        def real_dtype():
            if d['is_fp16']:
                return torch.float16
            if d['is_bf16']:
                return torch.bfloat16
            if d['is_fp32']:
                return torch.float32
        if dtype is not None:
            assert dtype == real_dtype()
        else:
            dtype = real_dtype()
        def load(n, *, cast_to=dtype):
            return torch.tensor(d[n], dtype=cast_to, device=device)
        def load_qkv(*, prefix='', suffix='', keep_dtype=False):
            cast_to = None if keep_dtype else dtype
            # return tuple([torch.tensor(d[f'{prefix}{n}{suffix}'], dtype=cast_to, device=device) for n in 'qkvo'])
            return tuple([load(f'{prefix}{n}{suffix}', cast_to=cast_to) for n in 'qkv'])
        q, k, v = load_qkv()
        b = None
        self.dev_tensors = (q, k, v, b)
        self.OUT_FUDGE_FACTOR = 3

        # TODO: load dropout_mask

        sm_scale = float(d['scale'])
        assert not np.isnan(sm_scale), 'FIXME: suppport scale=None when capturing torch.nn.functional.scaled_dot_product_attention'
        self.sdpa_params = SdpaParams(causal=bool(d['is_causal']),
                                      sm_scale=sm_scale,
                                      dropout_p=float(d['dropout_p']),
                                      dropout_mask=None)

        self.dout = load('upstream_grad')
        self.refout_tensors = (load('o_ref'), None)
        self.lp_refout_tensors = (load('o_lp_ref'), None)

        dq, dk, dv = load_qkv(prefix='grads_', suffix='_ref', keep_dtype=True)
        self.dref_tensors = (dq, dk, dv, None)
        dq, dk, dv = load_qkv(prefix='grads_', suffix='_ref_lp', keep_dtype=True)
        self.lp_dref_tensors = (dq, dk, dv, None)

    def compute_ref_forward(self, p : SdpaParams):
        self.fudge_factors = self._compute_fudge_factors(p)
        pass

    def compute_backward(self, out, dout, *, ref_only=False):
        assert ref_only == False, 'SdpaContextFromNPZ.compute_backward is incompatible with ref_out=True'
        self.dout_tensors = self._compute_backward(self.dev_tensors, out, dout)

    def democode_save_tensors(self):
        import numpy as np
        np.savez('dump.npz',
                 is_fp16=int(query.dtype == torch.float16),
                 is_bf16=int(query.dtype == torch.bfloat16),
                 is_fp32=int(query.dtype == torch.float32),
                 q=query.float().numpy(force=True),
                 k=key.float().numpy(force=True),
                 v=value.float().numpy(force=True),
                 o=out.float().numpy(force=True),
                 o_ref=out_ref.float().numpy(force=True),
                 o_lp_ref=out_lp_ref.float().numpy(force=True),
                 upstream_grad=upstream_grad.float().numpy(force=True),
                 dropout_p=dropout_p,
                 is_causal=int(is_causal),
                 scale=float('nan') if scale is None else float(scale),
                 enable_gqa=bool(enable_gqa),
                 grads_q=grads[0].float().numpy(force=True),
                 grads_k=grads[1].float().numpy(force=True),
                 grads_v=grads[2].float().numpy(force=True),
                 grads_q_ref_lp=grads_ref_lp[0].float().numpy(force=True),
                 grads_k_ref_lp=grads_ref_lp[1].float().numpy(force=True),
                 grads_v_ref_lp=grads_ref_lp[2].float().numpy(force=True),
                 grads_q_ref=grads_ref[0].float().numpy(force=True),
                 grads_k_ref=grads_ref[1].float().numpy(force=True),
                 grads_v_ref=grads_ref[2].float().numpy(force=True),
                 )

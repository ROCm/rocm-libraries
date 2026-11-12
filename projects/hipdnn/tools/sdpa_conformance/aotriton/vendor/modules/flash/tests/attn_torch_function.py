#!/usr/bin/env python
# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import torch
import queue
from torch.multiprocessing import Process
from aotriton_flash import IGNORE_BACKWARD_IMPORT
from aotriton_flash import (
    attn_fwd,
    debug_simulate_encoded_softmax,
    hipError_t,
    hipGetLastError,
    AOTRITON_TORCH_ONLY_USE_CPU,
    HipMemory,
    attn_options,
)
if not IGNORE_BACKWARD_IMPORT:
    from aotriton_flash import (
        attn_bwd,
    )
from collections import namedtuple
from dataclasses import dataclass
from _common_test import (
    BHSD,
    alloc_with_layout,
    assert_layout,
    cdiv,
    layout_of,
    narrow_to_prime,
)
from typing import Callable

# FWD_IMPL / BWD_IMPL name the backend to pin, in the vocabulary @ati.backend
# declares and the library publishes through OpAttn{Fwd,Bwd}Backend.by_index:
#
#   op_attn_fwd : triton  aiter  flyc
#   op_attn_bwd : triton_split  triton_fuse  aiter  flyc
#
# Unset means the operator selects. An explicit name is NOT the same as leaving
# it unset even when it names the default: forcing sets disable_fallback in the
# dispatcher (modules/flash/csrc/attn_fwd.cc), which suppresses the tuning-DB
# lookup.
#
# Names, not indices, because the indices are internal and have moved once
# already -- flyc took 2 on op_attn_fwd. The resolved index lives in
# FWD_IMPL_IDX / BWD_IMPL_IDX and goes to attn_options.force_backend_index,
# which is the one place an integer is the interface.
#
# Test parameter restrictions per BWD_IMPL (see _core_test_backward.py):
#   unset, triton_split : full head-dim coverage; all dtypes
#   triton_fuse         : POT up to 256, M8 up to 216; all dtypes
#   aiter               : POT up to 128, NPOT up to 192, M8 up to 184;
#                         fp16/bf16 only; no GQA; no dropout
#   flyc                : full head-dim coverage; fp16/bf16 only
def _resolve_impl(envvar, backend_struct, opname):
    """(name, index) for `$envvar`, or (None, None) when it is unset."""
    name = os.getenv(envvar, default=None)
    if name is None:
        return None, None
    by_name = {v: k for k, v in backend_struct.by_index.items()}
    if name in by_name:
        return name, by_name[name]
    # This runs at MODULE IMPORT, so raising anything vaguer here is a
    # collection error on every test in every file that imports this module.
    # The value comes from a human or from .ci/run-test.sh, and the valid set is
    # a property of the library that was built, so say both.
    published = sorted(by_name)
    if name.lstrip('-').isdigit():
        raise ValueError(
            f'{envvar}={name} is an index; it now takes a backend NAME. '
            f'{opname} publishes {published} in this build.')
    raise ValueError(
        f'{envvar}={name!r} is not a backend of {opname} in this build. '
        f'Published: {published}.')

from pyaotriton.v3.flash import OpAttnFwdBackend, OpAttnBwdBackend

FWD_IMPL, FWD_IMPL_IDX = _resolve_impl('FWD_IMPL', OpAttnFwdBackend, 'op_attn_fwd')
BWD_IMPL, BWD_IMPL_IDX = _resolve_impl('BWD_IMPL', OpAttnBwdBackend, 'op_attn_bwd')

# PROBE_UNSUPPORTED is independent of BWD_IMPL: enables NotImplementedError on
# hipErrorPeerAccessUnsupported so callers can skip unsupported configurations.
PROBE_UNSUPPORTED = bool(int(os.getenv('PROBE_UNSUPPORTED', default='0')))

from aotriton_flash import lazy_dq_acc, lazy_delta


def empty_handler():
    pass

@dataclass
class AttentionExtraArgs:
    return_encoded_softmax : bool = False
    autotune : bool = False
    return_autotune : bool = False
    is_testing : bool = True
    fillnan : bool = False
    return_logsumexp : bool = False
    illaddr_handler : Callable = empty_handler
    # PER-CALL backend override, taking precedence over the FWD_IMPL/BWD_IMPL
    # environment when set. None keeps the env behaviour, so every existing
    # caller is unaffected.
    #
    # The env variables pin a backend for a whole PROCESS, which is right for a
    # test pass but useless for comparing backends against each other: two
    # processes differ in clock state, allocator state and thermal history, and
    # that difference lands entirely in the measurement. performance_forward.py
    # and performance_backward.py use these to put every backend in one process,
    # interleaved by triton.testing.do_bench.
    force_fwd_backend_index : int | None = None
    force_bwd_backend_index : int | None = None
    # `int` or `(qk, vo)`: the real head dim, when the inputs carry the 8xD
    # slack the kernel's contract asks for. The OUTPUTS have to carry it too --
    # the kernel writes `ceil8(hdim)` columns of O and of every gradient -- and
    # `torch.empty_like` on a narrowed input silently compacts back to the odd
    # width, which is the trap this field exists to avoid. Allocate at the
    # 8-multiple, then narrow with `narrow_to_prime`. See that function.
    prime_hdim : int | tuple[int, int] | None = None
    # `{tname: perm}` over `'o'`/`'dq'`/`'dk'`/`'dv'`/`'db'`, from a
    # `StorageLayout`; see `_common_test.StorageLayout`. None -- every existing
    # caller -- keeps the layout each output has always had, which is the input
    # tensor's for a gradient and BHSD for `O`.
    #
    # The outputs need their own entry because they are not derivable from the
    # inputs: `O`, `dK` and `dV` each carry a separate buffer descriptor in the
    # gfx950 kernels, so pinning them to whatever q/k/v happen to use would
    # leave most of that arithmetic reading one layout forever.
    output_layouts : dict[str, tuple[int, int, int]] | None = None


def _alloc_output(tname, dims, *, dtype, device, attn_extra_args, narrow_to, like):
    """An output tensor with the right LAYOUT and the right 8xD SLACK.

    Two independent things have to be right at once, and `torch.empty_like`
    gets each of them wrong under the conditions the other one cares about.

    *Slack.* `torch.empty_like` on a `[..., :73]` view returns a COMPACT
    `(..., 73)` tensor -- pitch 73, no slack -- and the kernel then writes
    `ceil8(73) = 80` columns of it, over the next row. `narrow_to` is the real
    extent; the allocation rounds it up to the 8-multiple and hands back the
    narrowed view, so the output has the same shape-73/pitch-80 view the input
    does. It is spelled as an extent rather than a flag because the bias
    gradient wants the same treatment on an axis that is not the head dim: `db`
    mirrors `b`, which `_create_inputs` allocates at `round_to_8x(seqlen_k)`.

    *Layout.* `torch.empty_like` preserves strides only for a NON-OVERLAPPING
    DENSE tensor, and a narrowed view is neither -- so the moment the slack
    above exists, `empty_like` also silently flattens the layout back to BHSD.
    `like` reproduces the old behaviour where it was right (an input's own
    layout, read off its strides) and `output_layouts` overrides it where a
    caller wants to choose.
    """
    if dims is None:
        return None
    layouts = attn_extra_args.output_layouts
    if layouts is not None and tname in layouts:
        perm = layouts[tname]
    elif like is not None:
        perm = layout_of(like)
    else:
        perm = BHSD
    width = dims[3] if narrow_to is None else 8 * cdiv(narrow_to, 8)
    full = alloc_with_layout(tuple(dims[:3]) + (width,), perm, dtype=dtype, device=device)
    # Checked HERE rather than in the test, for the gradients' sake. A test can
    # read `o` back off the forward's return value, but `dq`/`dk`/`dv`/`db` only
    # reach it through `Tensor.grad`, and `SdpaContext._compute_backward` clones
    # that -- `clone()` keeps strides only for a non-overlapping dense tensor, so
    # a narrowed gradient arrives flattened to BHSD no matter what was
    # allocated. The one place that still knows is this one.
    assert_layout(full, perm, width, tname)
    return narrow_to_prime(full, narrow_to)


def _alloc_like(t, tname, attn_extra_args, narrow_to):
    """`_alloc_output` for a gradient, whose shape and default layout are its input's."""
    if t is None:
        return None
    return _alloc_output(tname, tuple(t.shape), dtype=t.dtype, device=t.device,
                         attn_extra_args=attn_extra_args, narrow_to=narrow_to, like=t)


def _prime_pair(attn_extra_args):
    p = attn_extra_args.prime_hdim
    return (None, None) if p is None else ((p, p) if isinstance(p, int) else p)


VERBOSE=False
DEFAULT_PHILOX_SEED = 0x1BF52
DEFAULT_PHILOX_OFFSET_1 = 0x1D4000
DEFAULT_PHILOX_OFFSET_2 = 0x000B42
DEFAULT_PHILOX_OFFSET = DEFAULT_PHILOX_OFFSET_1 + DEFAULT_PHILOX_OFFSET_2

def is_power_of_two(n: int) -> bool:
    return (n & (n - 1) == 0) and n != 0

def is_supported_by_tl_dot(n: int) -> bool:
    return is_power_of_two(n) and n >= 16

class _attention(torch.autograd.Function):

    # DEBUG_MASK_DTYPE = torch.int32
    # DEBUG_MASK_DTYPE = torch.float32

    @staticmethod
    def forward(ctx, q, k, v, b, causal, sm_scale, dropout_p,
                attn_extra_args=AttentionExtraArgs()):
        return_encoded_softmax = attn_extra_args.return_encoded_softmax
        autotune = attn_extra_args.autotune
        return_autotune = attn_extra_args.return_autotune
        if return_autotune and attn_extra_args.return_logsumexp:
            assert False, 'Cannot set return_autotune and return_logsumexp at the same time. Both are returned as 3rd value'
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk
        # assert Lk in {16, 32, 64, 128}
        seqlen_q = q.shape[2]
        seqlen_k = k.shape[2]
        _pqk, _pvo = _prime_pair(attn_extra_args)
        # `like=None`: O has always been allocated BHSD regardless of what q/k/v
        # are stored as, and only an explicit `output_layouts['o']` moves it.
        # See `_alloc_output` for the slack.
        o = _alloc_output('o', (q.shape[0], q.shape[1], q.shape[2], v.shape[3]),
                          dtype=q.dtype, device=q.device,
                          attn_extra_args=attn_extra_args, narrow_to=_pvo, like=None)

        # def round_to_16x(x):
        #     return ((x + 15) // 16) * 16
        # M_padded = torch.empty((q.shape[0] * q.shape[1], round_to_16x(q.shape[2])), device=q.device, dtype=torch.float32)
        # M = M_padded[:,:q.shape[2]]
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        if attn_extra_args.fillnan:
            for t in (o, M):
                t.fill_(float('nan'))
        if return_encoded_softmax:
            encoded_softmax = torch.zeros((q.shape[0], q.shape[1], q.shape[2], k.shape[2]), device=q.device, dtype=q.dtype)
        else:
            encoded_softmax = None
        if False or VERBOSE:
            print(f'{q.shape=}')
            print(f'{k.shape=}')
            print(f'{v.shape=}')
            print(f'{o.shape=}')
            print(f'{q.data_ptr()=:x}')
            print(f'{k.data_ptr()=:x}')
            print(f'{v.data_ptr()=:x}')
            print(f'{M.data_ptr()=:x}')
            print(f'{o.data_ptr()=:x}')
            print(f'{stage=}')
            print(f'seqlen_q={q.shape[2]}')
            print(f'seqlen_k={k.shape[2]}')
            print(f'{v.data_ptr()=:x}')
            print(f'{v.stride(1)=:x}')
            print(f'{v.data_ptr() + q.shape[0] * q.shape[1] * v.stride(1)=:x}')
            if encoded_softmax is not None:
                print(f'{encoded_softmax.shape=} {encoded_softmax.dtype=}')

        philox_null = torch.empty([0], device=q.device, dtype=torch.uint64)
        if dropout_p > 0.0:
            assert philox_null.data_ptr() == 0
            philox_seed = torch.tensor([DEFAULT_PHILOX_SEED], device=q.device, dtype=torch.uint64)
            philox_offset1 = torch.tensor([DEFAULT_PHILOX_OFFSET_1], device=q.device, dtype=torch.uint64)
            philox_offset2 = DEFAULT_PHILOX_OFFSET_2
            philox_seed_output = torch.tensor([0], device=q.device, dtype=torch.uint64)
            philox_offset_output = torch.tensor([0], device=q.device, dtype=torch.uint64)
            assert philox_seed_output.data_ptr() != 0
            assert philox_offset_output.data_ptr() != 0
        else:
            philox_seed = philox_null
            philox_offset1 = philox_null
            philox_offset2 = 0
            philox_seed_output = philox_null
            philox_offset_output = philox_null

        if causal:
            atomic = torch.zeros([1], device=q.device, dtype=torch.int32)
        else:
            atomic = torch.empty([0], device=q.device, dtype=torch.int32)

        if attn_extra_args.force_fwd_backend_index is not None:
            extargs = attn_options()
            extargs.force_backend_index = attn_extra_args.force_fwd_backend_index
        elif FWD_IMPL is not None:
            extargs = attn_options()
            extargs.force_backend_index = FWD_IMPL_IDX
        else:
            extargs = None

        # print(f'{attn_extra_args=}')
        # Check GPU kernel accepts nullptr for philox_*_output
        if attn_extra_args.is_testing:
            ret = attn_fwd(q, k, v, b, sm_scale, M, o,
                           dropout_p, philox_seed, philox_offset1, philox_offset2,
                           philox_null, philox_null,
                           encoded_softmax, causal, atomic, extargs=extargs)
            if PROBE_UNSUPPORTED and ret == hipError_t.hipErrorPeerAccessUnsupported:
                raise NotImplementedError()
            assert ret == hipError_t.hipSuccess, ret
            # Reset the persistent atomic tile counter between the pre-flight
            # probe call and the real call. Without this, the second attn_fwd
            # sees the counter already at num_tiles_total + Num_WG and the
            # persistent loop body never executes -- the real call becomes a
            # no-op, and any NaN-prefilled cells in o/M survive into the
            # is_testing assert below.
            if causal:
                atomic.zero_()

        ret = attn_fwd(q, k, v, b, sm_scale, M, o,
                       dropout_p, philox_seed, philox_offset1, philox_offset2,
                       philox_seed_output, philox_offset_output,
                       encoded_softmax, causal, atomic, extargs=extargs)
        if PROBE_UNSUPPORTED and ret == hipError_t.hipErrorPeerAccessUnsupported:
            raise NotImplementedError()
        if attn_extra_args.is_testing:
            try:
                torch.cuda.synchronize()
            except:
                pass
            last_err = hipGetLastError()
            if last_err == hipError_t.hipErrorIllegalAddress:
                attn_extra_args.illaddr_handler()
            assert last_err == hipError_t.hipSuccess, last_err
        else:
            assert ret == hipError_t.hipSuccess, ret
        tuning_result = None

        ctx.save_for_backward(q, k, v, b, o, M)
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.philox_seed = philox_seed_output
        ctx.philox_offset = philox_offset_output
        ctx.encoded_softmax = encoded_softmax # FIXME: for debugging only
        ctx.tuning_result = [('attn_fwd', tuning_result)] if tuning_result is not None else None
        ctx.fwd_tuning_result = tuning_result
        ctx.attn_extra_args = attn_extra_args
        ctx.autotune = autotune
        ctx.return_autotune = return_autotune
        if attn_extra_args.is_testing:
            assert not torch.isnan(M).any(), f'L tensor has NaN'
        ret3 = M if attn_extra_args.return_logsumexp else ctx.tuning_result
        return o, encoded_softmax, ret3

    @staticmethod
    def backward_v3(ctx, do, _, __):
        q, k, v, b, o, L = ctx.saved_tensors
        # print(f'{b=}')
        sm_scale = ctx.sm_scale
        dropout_p = ctx.dropout_p
        philox_seed = ctx.philox_seed
        philox_offset = ctx.philox_offset
        causal = ctx.causal
        attn_extra_args = ctx.attn_extra_args
        autotune = ctx.autotune
        return_autotune = ctx.return_autotune
        # if q.shape[-1] <= 32:
        # do = do.contiguous()
        _pqk, _pvo = _prime_pair(attn_extra_args)
        dq = _alloc_like(q, 'dq', attn_extra_args, _pqk)
        dq_acc = lazy_dq_acc(q)  # dq_acc only supports BHSD; lazy_dq_acc always produces BHSD to satisfy this.
        dk = _alloc_like(k, 'dk', attn_extra_args, _pqk)
        dv = _alloc_like(v, 'dv', attn_extra_args, _pvo)
        # `db` mirrors `b`, on the KV axis rather than the head dim: the bias is
        # allocated at `round_to_8x(seqlen_k)` and narrowed, so its gradient is
        # too, and the two agree on strides instead of only on shape.
        #
        # **Unless the bias does not require a gradient**, which AOTriton spells
        # as a null DB over an all-zero stride triple -- `mk_aotensor(None,
        # if_empty_then_like=q)` produces exactly the `empty_t4` PyTorch passes
        # from attention_backward.cu for a bool attn_mask. Allocating a dB anyway
        # would keep that path unreachable from every test that goes through
        # autograd, which is how a null-DB store survived in the flyc gfx1201 dQ
        # kernel. `b.requires_grad` rather than `ctx.needs_input_grad`, so this
        # agrees with SdpaContext._compute_backward's own test.
        db = (_alloc_like(b, 'db', attn_extra_args, b.shape[-1])
              if b is not None and b.requires_grad else None)
        delta = lazy_delta(L)
        seqlen_q = q.shape[2]
        seqlen_k = k.shape[2]
        if attn_extra_args.force_bwd_backend_index is not None:
            extargs = attn_options()
            extargs.force_backend_index = attn_extra_args.force_bwd_backend_index
        elif BWD_IMPL is not None:
            extargs = attn_options()
            extargs.force_backend_index = BWD_IMPL_IDX
        else:
            extargs = None

        ret = attn_bwd(q, k, v, b, sm_scale, o, do, dq, dk, dv, db, dq_acc, L, delta,
                       dropout_p, philox_seed, philox_offset, 0, causal,
                       extargs=extargs)
        if PROBE_UNSUPPORTED and ret == hipError_t.hipErrorPeerAccessUnsupported:
            raise NotImplementedError()
        assert ret == hipError_t.hipSuccess, ret
        tuning_result = None

        if tuning_result is not None:
            ctx.tuning_result += tuning_result

        return dq, dk, dv, db, None, None, None, None, None

    backward = backward_v3

attention = _attention.apply

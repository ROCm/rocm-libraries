#!/usr/bin/env python
# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import torch
import os
import bisect
import math
import pathlib
import json
import gc

from attn_torch_function import (
    DEFAULT_PHILOX_SEED,
    DEFAULT_PHILOX_OFFSET,
    attention,
    AttentionExtraArgs,
    BWD_IMPL,
    FWD_IMPL,
    PROBE_UNSUPPORTED,
    hipError_t,
    hipGetLastError,
)
# Bottom-right causal cannot be spelled with a bool: translate_causal() maps
# True onto TOP_LEFT_ALIGNED. Used by core_test_bottom_right_fully_masked_rows.
from aotriton_flash import CausalType
from _common_test import (
    BHSD,
    SdpaContext,
    SdpaParams,
    SdpaContextFromNPZ,
    StorageLayout,
    AOTRITON_TORCH_ONLY_USE_CPU,
    alloc_with_layout,
    assert_layout,
    fmt_hdim,
    fmt_nheads,
    narrow_to_prime,
    refill_inputs_normal,
    cdiv,
)

RECORD_ADIFFS_TO = os.getenv('RECORD_ADIFFS_TO', default=None)
USE_ADIFFS_TXT = os.getenv('USE_ADIFFS_TXT', default=None)

# <utname> TAB <value>, where <value> is OOM, NAN, CPUREF, or a JSON adiff.
#
# CPUREF validates the test against the CPU reference instead of the GPU one,
# which is the same thing as saying the GPU reference is not trustworthy for it.
# No writer emits CPUREF -- the recorder below prints `utname TAB json` and
# .tune/bin/append_oom_to_adiffs.sh prints `utname (call) TAB OOM` -- so every
# such line is hand-added.
#
# Comments and blank lines are allowed because the file is hand-edited.
adiffs = {}
if USE_ADIFFS_TXT is not None:
    with open(USE_ADIFFS_TXT) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip('\n')
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            fields = line.rstrip().split('\t')
            if len(fields) != 2:
                raise ValueError(f'{USE_ADIFFS_TXT}:{lineno}: expected '
                                 f'<utname> TAB <value>, got {line!r}')
            utname, adiff_str = fields
            if adiff_str in ("OOM", "NAN", "CPUREF"):
                adiffs[utname] = adiff_str
            else:
                adiffs[utname] = json.loads(adiff_str)

# SIGSEGV_ERROR_CODE = signal.SIGSEGV

def exit_pytest():
    # os.kill(os.getpid(), SIGSEGV_ERROR_CODE)
    os._exit(139)

FOR_RELEASE = int(os.getenv('FOR_RELEASE', default='0'))
SMALL_VRAM = bool(int(os.getenv('SMALL_VRAM', default='0')))
# SKIP_BWD=1 runs the forward half of these tests only: the forward launch, the
# reference comparison, and the dropout-mask path -- everything up to and
# excluding .backward().
#
# This exists so a forward-only backend can be exercised by the same tests as
# everything else. The flyc backend (FWD_IMPL=flyc) has no backward at all, so
# every backward case would fail for a reason that says nothing about the
# forward kernel under test.
#
# It also makes test_forward.py retirable: with SKIP_BWD=1 these tests cover
# the same ground, over a wider parameter set, and there is then one file
# describing a forward rather than two that must agree.
SKIP_BWD = bool(int(os.getenv('SKIP_BWD', default='0')))

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

if FWD_IMPL == 'flyc':
    # flyc is f16/bf16 WMMA only -- modules/flash/aot/flyc_attn_fwd.py's
    # _flyc_fwd_disabled rejects fp32 outright, so no hsaco exists for those
    # functionals. Forcing the backend bypasses that predicate (it is the
    # operator's selection it overrides, not the kernel's own support), so
    # without this every fp32 case asks for a kernel that was never built.
    DTYPES = [torch.float16, torch.bfloat16]

if BWD_IMPL in (None, 'triton_split'):
    POT_HEADDIMS = [16, 32, 64, 128, 256, 512]
    NPOT_HEADDIMS = [48, 80, 96, 160, 192, 224]
    M8_HEADDIMS = [8, 24, 40, 56, 72, 88, 96, 120, 152, 184, 216, 248, 408]
elif BWD_IMPL == 'triton_fuse':
    POT_HEADDIMS = [16, 32, 64, 128, 256]
    NPOT_HEADDIMS = [48, 80, 96, 160, 192, 224]
    M8_HEADDIMS = [8, 24, 40, 56, 72, 88, 96, 120, 152, 184, 216]
elif BWD_IMPL == 'aiter':
    POT_HEADDIMS = [16, 32, 64, 128]
    NPOT_HEADDIMS = [48, 80, 96, 160, 192]
    M8_HEADDIMS = [8, 24, 40, 56, 72, 88, 96, 120, 152, 184]
    DTYPES = [torch.float16, torch.bfloat16]
elif BWD_IMPL == 'flyc':
    # flyc. Full head-dim coverage, same as the split path: both FlyDSL backward
    # tile ladders (fmha_tuning_bwd_{dkdv,dq}_gfx1201._BLOCK_DMODEL_LADDER) cover
    # every value of the operator's BLOCK_DMODEL axis, so an off-ladder test head
    # dim rounds up to a compiled tile and rides the PADDED_HEAD axis exactly as
    # it does for Triton.
    #
    # f16/bf16 only, and this is the BACKWARD's own exclusion, not an echo of
    # the forward's above: _flyc_common.py's predicate rejects fp32 too, so a
    # mixed Triton-forward run still has no fp32 backward kernel to call.
    DTYPES = [torch.float16, torch.bfloat16]
    POT_HEADDIMS = [16, 32, 64, 128, 256, 512]
    NPOT_HEADDIMS = [48, 80, 96, 160, 192, 224]
    M8_HEADDIMS = [8, 24, 40, 56, 72, 88, 96, 120, 152, 184, 216, 248, 408]
else:
    assert False, f'Unsupported BWD_IMPL {BWD_IMPL}'
# **Prime head dimensions, re-enabled with the allocation the contract wants.**
# They were disabled because PyTorch allocates compactly by default --
#   torch.rand((3, 5, 1033, 57)).stride() == (294405, 58881, 57, 1)
# -- and the kernel's input contract is 8xD: loads and stores are 8 columns
# wide, so it touches `ceil8(hdim)` columns of every row and needs the caller to
# own them. A compact 57 has no slack, so such a tensor is outside the contract
# and testing one measures the harness rather than the kernel.
#
# `_do_test_op_bwd` now allocates an off-grid D_HEAD at `ceil8(D_HEAD)` and
# passes a `[..., :D_HEAD]` view: extent odd, pitch on the grid, exactly what an
# AOTriton caller with a padded buffer hands over. The slack is filled with NaN,
# so a mask that multiplies by zero instead of discarding is caught rather than
# passing quietly. Multiples of 8 are unaffected.
# 401 allocates at 408, which only the full-coverage backends have a kernel for.
_FULL_HEADDIM_COVERAGE = (None, 'triton_split', 'flyc')
PRIME_HEADDIMS = ([7, 23, 37, 53, 67, 73, 83, 113, 149, 179, 211, 241]
                  + ([401] if BWD_IMPL in _FULL_HEADDIM_COVERAGE else []))
REGULAR_SEQLEN = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
REGULAR_SEQLEN_2K = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]  # OOM when test with bias
PRIME_SEQLEN_Q = [11, 17, 37, 67, 157, 257, 523, 1033, 2063, 4919]
PRIME_SEQLEN_K = [13, 31, 41, 71, 223, 337, 571, 1063, 2081, 5237]
PRIME_SEQLEN_Q_1K = [11, 17, 37, 67, 157, 257, 523]
PRIME_SEQLEN_K_1K = [13, 31, 41, 71, 223, 337, 571]

SMALL_HEADDIM_ONLY = bool(int(os.getenv('SMALL_HEADDIM_ONLY', default='0')))
LARGE_HEADDIM_ONLY = bool(int(os.getenv('LARGE_HEADDIM_ONLY', default='0')))

def remove_larger_than(data_list, threshold):
    return [x for x in data_list if x <= threshold]

def remove_not_larger_than(data_list, threshold):
    return [x for x in data_list if x > threshold]

def cdiv(x, div):
    return (x + div - 1) // div

def round_list_to_8x(data_list):
    return [cdiv(x, 8) * 8 for x in data_list]

if SMALL_HEADDIM_ONLY:
    POT_HEADDIMS = remove_larger_than(POT_HEADDIMS, 192)
    NPOT_HEADDIMS = remove_larger_than(NPOT_HEADDIMS, 192)
    # Re-enabled with PRIME_HEADDIMS itself: while that list was dead the filter
    # was pointless, but test_prime_hdim parametrises over it again. Unfiltered,
    # the two shards BOTH run the whole prime matrix, and 401 (allocated at 408)
    # lands on the shard that exists to avoid exactly that memory pressure.
    PRIME_HEADDIMS = remove_larger_than(PRIME_HEADDIMS, 192)
    M8_HEADDIMS = remove_larger_than(M8_HEADDIMS, 192)

if LARGE_HEADDIM_ONLY:
    POT_HEADDIMS = remove_not_larger_than(POT_HEADDIMS, 192)
    NPOT_HEADDIMS = remove_not_larger_than(NPOT_HEADDIMS, 192)
    PRIME_HEADDIMS = remove_not_larger_than(PRIME_HEADDIMS, 192)
    M8_HEADDIMS = remove_not_larger_than(M8_HEADDIMS, 192)

ALL_INT_HEADDIMS = POT_HEADDIMS + NPOT_HEADDIMS + M8_HEADDIMS
ALL_INT_HEADDIMS = sorted(list(set(ALL_INT_HEADDIMS)))

# Goal: for each hdim_1, find its POT decomposed hdims, and for each POT hdim
#       tensor block, test the loading the half of the block.
# For example, when testing hdim_1 = 216, the input will be padded as hdim = 224 = 32 + 64 + 128
# Then we should test
#   - 216, 64 = 128 / 2                 (padding load block_0)
#   - 216, 160 = 128 + 64 / 2           (full load block_0, padding load block_1)
#   - 216, 208 = 128 + 64 + 32 / 2      (full load block_0, padding load block_1)
#   - and the flipped combination
#
# The whole process will force us to test ~3x more tests:
#   len(ALL_INT_HEADDIMS)=24
#   len(ALL_TUP_HEADDIMS)=73
def _generate_inequal_hdims():
    ALL_COMPILED_HDIMS = sorted(POT_HEADDIMS + NPOT_HEADDIMS)
    def decompose(hdim_1):
        compiled_hdim = ALL_COMPILED_HDIMS[bisect.bisect_left(ALL_COMPILED_HDIMS, hdim_1)]
        tmp = compiled_hdim
        block_0 = 2 ** (tmp.bit_length() - 1)
        tmp -= block_0
        block_1 = 2 ** (tmp.bit_length() - 1) if tmp > 0 else 0
        tmp -= block_1
        block_2 = 2 ** (tmp.bit_length() - 1) if tmp > 0 else 0
        tmp -= block_2
        assert tmp == 0
        blocks = [item for item in [block_0, block_1, block_2] if item != 0]
        solid = 0
        for b in blocks:
            yield hdim_1, solid + b // 2
            yield solid + b // 2, hdim_1
            solid += b

    for hdim_1 in ALL_INT_HEADDIMS:
        yield from decompose(hdim_1)

ALL_TUP_HEADDIMS = sorted(list(set(_generate_inequal_hdims())))

ALL_HEADDIMS = ALL_INT_HEADDIMS + ALL_TUP_HEADDIMS

# Deduplication

'''
Note: for now we cannot really test both fused and split kernel at the same
      time. Env var BWD_IMPL is used to make the switch.

      However we still add BWDOP to the tests arguments so we can easily tell
      the actual bwd op being tested.
'''
#TODO: Let BWDOP determine the real backward op at runtime

# The pytest id for the pinned backward backend. Its own vocabulary, kept
# because it is printed in every test name that has ever been reported.
_BWDOP_ID = {
    None            : 'V3',
    'triton_split'  : 'Split',
    'triton_fuse'   : 'Fused',
    'aiter'         : 'AITERASM',
    'flyc'          : 'Flyc',
}
BWDOP_ids = [_BWDOP_ID[BWD_IMPL]]

def _make_block_eyes(q, base=1.0, inc=0.0):
    dhead = q.shape[-1]
    seqlen = q.shape[2]
    assert seqlen % dhead == 0
    scale = base
    for i in range(0, seqlen, dhead):
        q[:, :, i:i+dhead, :] = torch.eye(dhead, device=q.device, dtype=q.dtype) * scale
        scale += inc

def RP(x):
    rounded = 2 ** (x - 1).bit_length()
    return max(16, rounded)

'''
Flash Attention is batch operator that evaluates sm(QK')V
Q = batch_size x ... x seqlen_q x head_size
K = batch_size x ... x seqlen_k x head_size
    => K' = batch_size x ... x head_size x seqlen_k
V = batch_size x ... x seqlen_k x head_size
sm(.) = softmax(.)
The output size is
batch_size x ... x seqlen_q x head_size

Note: In Flash V2 API the ... is denoted as "num_heads", serving as uniformly sized sequences
but in PyTorch API it does not present at all
'''

def _do_test_op_bwd(request, args, device_str='cuda'):
    BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type = args
    if isinstance(D_HEAD, int):
        HDIM_QK = HDIM_VO = D_HEAD
    else:
        HDIM_QK, HDIM_VO = D_HEAD
    HDIM_MAX = max(HDIM_QK, HDIM_VO)
    if sm_scale == 'l1':
        sm_scale = 1.0 / HDIM_QK
    elif sm_scale == 'l2':
        sm_scale = 1.0 / math.sqrt(HDIM_QK)
    if BWD_IMPL == 'aiter':  # AITER ASM
        if dropout_p > 0.0:
            pytest.skip("Dropout unsupported in AITER ASM backend for now. Need adjust FWD PRNG function")
        if HDIM_MAX < 64:
            pytest.skip("hdim < 64 AITER ASM kernel does not exist.")
        if HDIM_MAX > 192:
            pytest.skip("hdim > 192 AITER ASM kernel does not exist.")
        if HDIM_QK != HDIM_VO:
            pytest.skip("hdim_qk != hdim_vo is not exposed by AITER ASM backend.")
        if not isinstance(N_HEADS, int):
            pytest.skip("GQA is not exposed in AITER ASM backend.")
    if causal and bias_type is not None:
        pytest.skip("_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True")
    if SMALL_VRAM and seqlen_q * seqlen_k * HDIM_MAX * dtype.itemsize > 4096 * 8192 * 256 * 2:
        pytest.skip("Skip large tests (qkd > 4096 * 8192 * 256) due to low VRAM.")
    if seqlen_q * seqlen_k * HDIM_MAX * dtype.itemsize > 2048 * 2048 * 64 * 2:
        gc.collect()
        torch.cuda.empty_cache()
    if 'gfx11' in torch.cuda.get_device_properties(0).gcnArchName:
        if HDIM_MAX > 256:
            pytest.skip("Skip hdim > 256 on gfx11 arch due to register pressure.")
    utname = os.environ.get('PYTEST_CURRENT_TEST')
    use_adiff_entry = adiffs.get(utname, None)
    if use_adiff_entry == "OOM":
        pytest.skip("[Adiffs] Skip due to known OOM.")
    if use_adiff_entry == "NAN":
        mark = pytest.mark.xfail(reason="[Adiffs] XPASS due to known NAN.")
        request.node.add_marker(mark)
        return 0
    # CPUREF does NOT return -- the test still runs, just against the CPU
    # reference -- so unlike OOM/NAN the sentinel has to be cleared here.
    # validate_with_reference() subscripts this entry as a dict
    # (_common_test.py's use_adiff_entry["adiff"] / ["grads_adiff"]) and it is
    # passed straight through below, so leaving the string in place would raise
    # TypeError: string indices must be integers. None is also the honest value:
    # a CPUREF line carries no recorded adiff.
    adiff_ref_device_policy = None
    if use_adiff_entry == "CPUREF":
        adiff_ref_device_policy = 'cpu'
        use_adiff_entry = None
        print("[Adiffs] CPUREF: validating against the CPU reference")
    print(f"{use_adiff_entry=}")
    torch.cuda.empty_cache()
    SKIP_DK_DV = False
    SKIP_DQ = False
    # `bias_type` is a comma-separated token set: the bias kind, plus any
    # modifiers, in any order. Split rather than matched, so nothing here has to
    # know the kind names -- 'vector,nograd' works the day a vector bias does,
    # and so does 'nograd,vector'.
    #
    # `nograd` is the only modifier so far: the caller supplies a bias but does
    # not want its gradient. AOTriton spells that as an all-zero dB stride triple
    # over a null DB, and PyTorch takes it on every bool-masked SDPA -- the mask
    # becomes an additive bias that is not a leaf and carries no grad, so
    # attention_backward.cu passes `empty_t4` for DB. Without a case here that
    # path has no coverage at all: a bias otherwise always requires grad (the
    # SKIP_DB line below), and attn_torch_function's backward used to allocate dB
    # from `b` unconditionally, so even a non-grad bias got a real, writable one.
    if isinstance(bias_type, str):
        _bias_tokens = bias_type.split(',')
        BIAS_NOGRAD = 'nograd' in _bias_tokens
        _bias_kinds = [t for t in _bias_tokens if t != 'nograd']
        assert len(_bias_kinds) == 1, \
            f'bias_type {bias_type!r} must name exactly one bias kind, got {_bias_kinds}'
        bias_type = _bias_kinds[0]
    else:
        BIAS_NOGRAD = False
    SKIP_DB = True if bias_type is None else BIAS_NOGRAD
    USE_AUTOTUNE = True
    torch.manual_seed(20)
    # The `storage_flip` slot carries either spelling. A `StorageLayout` names a
    # layout per tensor, including the ones the kernel writes; a bool or an
    # `(x, y)` pair is the old single transposition of q/k/v/b. See
    # `_common_test.StorageLayout` for why both exist.
    if isinstance(storage_flip, StorageLayout):
        transpose = None
        storage_layout = storage_flip
    elif isinstance(storage_flip, tuple):
        transpose = storage_flip
        storage_layout = None
    else:
        transpose = (1, 2) if storage_flip else None
        storage_layout = None
    # **A head dim off the 8-multiple grid is allocated on it and narrowed.**
    # The kernel's input contract is 8xD -- loads and stores are 8 columns wide,
    # so it touches `ceil8(hdim)` columns of every row and the caller must own
    # them. `torch.rand(3, 5, 1033, 57)` does not: PyTorch packs it compactly,
    # there is no slack, and the kernel walks into the next row. That is why
    # `PRIME_HEADDIMS` above was disabled, and it was the right call for a
    # compact tensor.
    #
    # It is the wrong call for the kernel. Allocating at `ceil8(D_HEAD)` and
    # handing over a `[..., :D_HEAD]` view gives the extent and the pitch the
    # contract actually describes -- and `narrow_to_prime` poisons the slack
    # with NaN, so a masking bug shows up rather than passing quietly. Nothing
    # changes for a D_HEAD already on the grid.
    _prime_hdim = D_HEAD if (HDIM_QK % 8 or HDIM_VO % 8) else None
    _alloc_hdim = D_HEAD if _prime_hdim is None else (
        8 * cdiv(HDIM_QK, 8) if isinstance(D_HEAD, int)
        else (8 * cdiv(HDIM_QK, 8), 8 * cdiv(HDIM_VO, 8)))
    ctx = SdpaContext(BATCH, N_HEADS, _alloc_hdim, seqlen_q, seqlen_k, dtype,
                      bias_type=bias_type, storage_flip=transpose, device=device_str, fillnan=True,
                      prime_hdim=_prime_hdim, storage_layout=storage_layout)
    ctx.create_ref_inputs(target_device_policy=adiff_ref_device_policy)
    ctx.set_require_grads(skip_dq=SKIP_DQ, skip_dk_dv=SKIP_DK_DV, skip_db=SKIP_DB)
    q, k, v, b = ctx.dev_tensors
    # The row pitch each tensor's INNERMOST axis requires. For everything but a
    # bias that is the 8xD contract, restated: the kernel accesses `ceil8(hdim)`
    # columns of every row, and the gfx950 descriptors cap their D round-up at
    # the true row pitch -- so a pitch below the grid turns that cap into a clip
    # of the last real column. A bias's innermost axis is the KV sequence, which
    # carries no such contract (`_bias_slab_num_records_bytes`), so `seqlen_k`
    # is all it owes.
    _row_pitch = {'q': 8 * cdiv(HDIM_QK, 8), 'k': 8 * cdiv(HDIM_QK, 8),
                  'v': 8 * cdiv(HDIM_VO, 8), 'b': seqlen_k,
                  'dout': 8 * cdiv(HDIM_VO, 8), 'o': 8 * cdiv(HDIM_VO, 8)}
    def _check_layout(t, tname):
        if storage_layout is not None:
            assert_layout(t, storage_layout[tname], _row_pitch[tname], tname)
    for _tname, _t in zip(('q', 'k', 'v', 'b'), ctx.dev_tensors):
        _check_layout(_t, _tname)
    # autotune = True
    # # triton implementation
    ext = AttentionExtraArgs(return_encoded_softmax=False if dropout_p == 0 else True,
                             autotune=False,
                             return_autotune=False,
                             fillnan=True,
                             illaddr_handler=exit_pytest,
                             # O and every gradient carry the same slack the
                             # inputs do; `torch.empty_like` on a narrowed view
                             # would compact it away.
                             prime_hdim=_prime_hdim,
                             # O and the gradients get their own layouts; the
                             # kernel gives each of O, dK and dV a buffer
                             # descriptor of its own, so leaving them BHSD would
                             # leave that arithmetic untested.
                             output_layouts=None if storage_layout is None else storage_layout.outputs(),
                             )
    tri_out, encoded_softmax, _ = attention(q, k, v, b, causal, sm_scale, dropout_p, ext)
    _check_layout(tri_out, 'o')
    dropout_mask = encoded_softmax >= 0 if encoded_softmax is not None else None
    sdpa_params = SdpaParams(causal=causal, sm_scale=sm_scale, dropout_p=dropout_p, dropout_mask=dropout_mask)
    ref_out, _ = ctx.compute_ref_forward(sdpa_params)

    if SKIP_BWD:
        # Forward-only: validate the forward output against the reference and
        # stop. Uses validate_with_reference's existing no_backward= rather than
        # a separate path, so the forward assertion below is exactly the one the
        # full path makes.
        is_allclose, adiff, _grads_allclose, _grads_adiff, tfts = ctx.validate_with_reference(
            tri_out, [], no_backward=True,
            return_target_fudge_factors=True, use_adiff_entry=use_adiff_entry)
        ctx.display_validation_results(tri_out, is_allclose, adiff, [], [])
        # RECORD_ADIFFS_TO before the assert, exactly as the full path below has
        # it. A tolerance-calibration pass exists to COLLECT the mismatches, so
        # asserting first would hard-fail the very run whose job is to record
        # them -- and the only builds that set SKIP_BWD=1 are the flyc ones
        # whose tolerances are least well known.
        if RECORD_ADIFFS_TO is not None and not is_allclose:
            with open(RECORD_ADIFFS_TO, 'a') as f:
                # grads_adiff is [] rather than absent: the consumer reads a
                # fixed pair of keys, and "no backward was run" is honestly an
                # empty list of gradient diffs.
                dj = { "adiff" : adiff, "grads_adiff" : [] }
                print(utname, "\t", json.dumps(dj), file=f, flush=True, sep='')
            pytest.xfail(f"RECORD ADIFFS {adiff=} (SKIP_BWD=1)")
        assert is_allclose, f'Forward pass {is_allclose=} {tfts=}'
        print(f'{tri_out=}')
        print(f'{adiff=} (SKIP_BWD=1, backward not run)')
        return seqlen_q * seqlen_k * HDIM_MAX

    # dO is read column-wise by the backward, so it needs the slack too --
    # `rand_like` on the narrowed `tri_out` would hand over a compact row. It
    # carries a layout for the same reason O does: dO reaches the kernel through
    # its own descriptor, riding the forward's V slot.
    _dout_width = tri_out.shape[-1] if _prime_hdim is None else 8 * cdiv(HDIM_VO, 8)
    dout = narrow_to_prime(alloc_with_layout(tuple(tri_out.shape[:-1]) + (_dout_width,),
                                             BHSD if storage_layout is None else storage_layout['dout'],
                                             dtype=tri_out.dtype, device=tri_out.device, rand=True),
                           None if _prime_hdim is None else HDIM_VO)
    _check_layout(dout, 'dout')
    if PROBE_UNSUPPORTED:
        try:
            ctx.compute_backward(tri_out, dout)
        except NotImplementedError as e:
            pytest.xfail("Unsupported Config in AITER")
    else:
        ctx.compute_backward(tri_out, dout)
    is_allclose, adiff, grads_allclose, grads_adiff, tfts = ctx.validate_with_reference(tri_out, ctx.dout_tensors, return_target_fudge_factors=True, use_adiff_entry=use_adiff_entry)
    ctx.display_validation_results(tri_out, is_allclose, adiff, grads_allclose, grads_adiff)

    if RECORD_ADIFFS_TO is not None and (not is_allclose or not all(grads_allclose)):
        with open(RECORD_ADIFFS_TO, 'a') as f:
            dj = { "adiff" : adiff, "grads_adiff" : grads_adiff }
            print(utname, "\t", json.dumps(dj), file=f, flush=True, sep='')
        pytest.xfail(f"RECORD ADIFFS {adiff=} {grads_adiff=}")
    assert is_allclose, f'Forward pass {is_allclose=} {tfts=}'
    dq_allclose, dk_allclose, dv_allclose, db_allclose = grads_allclose
    tri_dq, tri_dk, tri_dv, tri_db = ctx.dout_tensors
    ref_dq, ref_dk, ref_dv, ref_db = ctx.dref_tensors
    if not SKIP_DQ:
        assert tri_dq is not None
        assert ref_dq is not None
    if not SKIP_DK_DV:
        assert tri_dk is not None
        assert tri_dv is not None
        assert ref_dk is not None
        assert ref_dv is not None
    if not SKIP_DB:
        assert tri_db is not None
        assert ref_db is not None
    assert dk_allclose and dv_allclose and dq_allclose and db_allclose, f'{dk_allclose=} {dv_allclose=} {dq_allclose=} {db_allclose=} {tfts=}'
    print(f'{tri_out=}')
    print(f'{adiff=} {grads_adiff=}')
    return seqlen_q * seqlen_k * HDIM_MAX

def core_test_op_bwd(request, args, device : int | None = None):
    # The reclaim is in a `finally` because it used to sit inside the `try`,
    # right after the call, and so ran ONLY when the test returned normally.
    # Every other exit skipped it -- and the one that matters most is
    # torch.OutOfMemoryError, a RuntimeError, which the handler below re-raises:
    # a test that ran out of memory left its partial allocations in the caching
    # allocator and made the next large test likelier to OOM too. That cascades,
    # is order-dependent, and evaporates when the test is re-run alone. An
    # assertion failure and an xfail leaked the same way, just less visibly.
    qkh = 0
    completed = False
    skipped = False
    try:
        if device is None:
            qkh = _do_test_op_bwd(request, args, device_str='cuda')
        else:
            with torch.cuda.device(device):
                qkh = _do_test_op_bwd(request, args, device_str=f'cuda:{device}')
        completed = True
    except pytest.skip.Exception:
        # Skipped before anything was allocated: every skip guard in
        # _do_test_op_bwd runs before the SdpaContext is built, so there is
        # nothing to reclaim and empty_cache() would only buy a device sync.
        # Listed first because Skipped derives from BaseException, not from the
        # exceptions below.
        skipped = True
        raise
    except torch.AcceleratorError as e:
        print(f'AcceleratorError: {e}')
        exit_pytest()
    except RuntimeError as e:
        if hipGetLastError() == hipError_t.hipErrorIllegalAddress:
            exit_pytest()
        raise e
    finally:
        # On a clean run keep the size guard -- empty_cache() synchronises, and
        # a small test has nothing worth reclaiming. After a FAILURE reclaim
        # unconditionally: the test may have died part-way through allocating,
        # so its footprint has no relation to the shape it was asked for.
        if not skipped and (not completed or qkh > 2048 * 2048 * 64):
            gc.collect()
            torch.cuda.empty_cache()

# Deliberately unparametrized: only need one dtype+hdim to detect the defect
def core_test_bottom_right_fully_masked_rows(device_str='cuda'):
    '''ROCm/aotriton#235: the persistent loop must not skip tiles after a
    fully-masked early exit.

    attn_fwd initialised `continue_condition` once per program rather than once
    per tile. The fully-masked causal early exit cleared it and nothing restored
    it, so every later tile the same workgroup claimed from
    persistent_atomic_counter was silently skipped -- neither Out nor LSE
    written, leaving those rows with stale buffer contents.

    Two conditions arm it, and both are load-bearing here:

      * bottom-right causal with seqlen_q - seqlen_k >= BLOCK_M, so that an
        ENTIRE tile falls inside the masked prefix and the early exit is taken
        at all. A partially-masked tile does not arm it -- that row block still
        attends, so the kernel runs it normally. The largest BLOCK_M attn_fwd
        builds is 256 (gfx950, modules/flash/aot/attn_fwd.py), hence a masked
        prefix of 256 here; 128 would arm only the tile sizes the tuning
        database happens to pick today.
      * more tiles than workgroups (Num_CU * GRID_CU_MULTIP), so a workgroup is
        handed a further tile after the one that cleared the flag. Sized from
        the device's CU count rather than hardcoded -- a fixed size would
        quietly stop covering the bug on a larger GPU.

    Whether a workgroup wins the next atomic_add before a fresh one starts is a
    scheduling question, so the failure is not perfectly deterministic; hence
    the repeats. At this tile count it reproduced on every attempt.
    '''
    seqlen_q, seqlen_k, d_head = 320, 64, 64
    n_masked = seqlen_q - seqlen_k        # leading rows that attend nothing
    n_heads = 16
    dtype = torch.float16
    REPEATS = 4

    num_cu = torch.cuda.get_device_properties(device_str).multi_processor_count
    # seqlen_q=320 is >= 2 tiles for any BLOCK_M <= 256, and GRID_CU_MULTIP is
    # 2, so this lands the tile count at 4x the workgroup count or better.
    batch = max(1, (num_cu * 4) // n_heads)

    sm_scale = 1.0 / math.sqrt(d_head)
    # is_testing=False: the wrapper's own NaN check on L would fire first and
    # report 'L tensor has NaN' instead of the diagnostics below. fillnan makes
    # an unwritten element unambiguous rather than whatever the allocator left.
    ext = AttentionExtraArgs(return_encoded_softmax=False,
                             autotune=False,
                             return_autotune=False,
                             is_testing=False,
                             fillnan=True,
                             return_logsumexp=True)
    for i in range(REPEATS):
        torch.manual_seed(i)
        q = torch.randn((batch, n_heads, seqlen_q, d_head), device=device_str, dtype=dtype)
        k = torch.randn((batch, n_heads, seqlen_k, d_head), device=device_str, dtype=dtype)
        v = torch.randn_like(k)
        tri_out, _, L = attention(q, k, v, None, CausalType.BOTTOM_RIGHT,
                                  sm_scale, 0.0, ext)
        torch.cuda.synchronize()

        ctx = (f'iter {i}: batch={batch} n_heads={n_heads} num_cu={num_cu} '
               f'seqlen_q={seqlen_q} seqlen_k={seqlen_k}')
        # LSE is (B * H_Q, S_Q). Fully-masked rows must be +inf: the backward
        # subtracts it from qk so exp(qk - inf) == 0 for those blocks.
        masked_lse = L.view(batch, n_heads, seqlen_q)[:, :, :n_masked]
        n_inf = int(torch.isinf(masked_lse).sum())
        expect_inf = batch * n_heads * n_masked
        assert n_inf == expect_inf, \
            f'{ctx}: only {n_inf}/{expect_inf} fully-masked LSE rows were written'
        assert not torch.isnan(L).any(), f'{ctx}: LSE contains unwritten (NaN) rows'
        assert not torch.isnan(tri_out).any(), f'{ctx}: Out contains unwritten (NaN) rows'
        n_nonzero = int((tri_out[:, :, :n_masked] != 0).sum())
        assert n_nonzero == 0, \
            f'{ctx}: {n_nonzero} nonzero elements in fully-masked Out rows'


def core_test_sm_scale_magnitude(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal,
                                 sm_scale, dtype, storage_flip, max_adiff,
                                 device_str='cuda'):
    '''**The softmax scale magnitude, as a precision axis of the forward.**

    `sm_scale` is a caller-supplied number, not necessarily `1/sqrt(hdim)`. A
    kernel may fold `qk_scale = sm_scale * log2e` into Q before the QK GEMM
    instead of applying it to the f32 QK accumulator afterwards; when Q is
    f16/bf16 that fold rounds the product back to the input dtype, and the
    ~2**-8 of relative error it costs lands in the *exponent* of
    `exp2(S - m)`. The resulting output error is proportional to
    `sm_scale * sqrt(hdim)`, so it is invisible at the usual
    `sm_scale = 1/sqrt(hdim)` (where that product is 1) and grows without bound
    as the caller raises the scale. Max |err| against an f32 reference,
    measured on gfx950 over the exact cases test_backward.py parametrises
    (`sm_scale=1.2`, normal inputs), with the fold and without:

        dtype  hdim  shape        Q-folded   scale on f32 scores
        bf16     64  289x289       0.0647          0.0158
        bf16     64  289x400 C     0.0656          0.0128
        bf16    256  289x289       0.1165          0.0158
        bf16    256  289x400 C     0.1246          0.0156
        bf16    512  289x289       0.1682          0.0139
        bf16    512  289x400 C     0.1682          0.0120
        f16     512  289x289       0.0299          0.0014
        f16     512  289x400 C     0.0255          0.0014

    At `sm_scale = 1/sqrt(hdim)` the same cases run 0.0016 - 0.0089 (bf16) and
    0.0002 - 0.0010 (f16) either way: the fold is free where `|S|` is O(1).

    Guarded implementation: modules/flash/flyc/fmha_dualwave_gfx950.py, where
    `ParityQLoader.scale_all` refuses the fold and
    `ParityGemmHelper.scale_scores` applies the scale to the f32 scores,
    matching modules/flash/kernel/fwd_kernel_inner.py's
    `qk += (Qk_scale * tl.dot(q0, k0))`.

    Reported by
    test/test_transformers.py::test_mem_eff_attention_single_query_tail, which
    fails on gfx950 because it passes `scale=1.0` at `head_dim=512`. Its shapes
    are reproduced here, but NOT because of its name: the CUDA "single-query
    tail block" the UT is written about has no analogue on ROCm, and a q_len
    sweep across every ROCm block boundary (63/64/65, 127/128/129, 255/256/257,
    288/289, 320/321) is clean at `sm_scale = 1/sqrt(hdim)`. The tail shape is
    here only to keep this test recognisable as that UT; the axis that actually
    moves the error is `sm_scale * sqrt(hdim)`.

    **The normal inputs and the `max_adiff` bound are both load-bearing.**
    Neither is a preference; without either one this test passes on a kernel
    that has the bug, and the two failures are independent:

      * Inputs. Every other test in this suite takes `SdpaContext`'s default
        uniform [0, 1). Re-measured on the same Q-folded kernel as the table
        above but with uniform inputs, `sm_scale=1.2` gives 0.0031 - 0.0078
        (bf16) and 0.0004 - 0.0009 (f16) -- a factor of 1.4 to 3.6 over the
        control, i.e. nothing, and the f16 NaN below does not appear at all.
        `refill_inputs_normal` explains the mechanism: a uniform, all-positive
        K turns a Q-side perturbation into a common-mode row shift, which is
        exactly what softmax subtracts away. The reporting UT uses `randn`.
      * Bound. `validate_with_reference`'s threshold is
        `OUT_FUDGE_FACTOR * ref_error`, and `ref_error` is how far torch's OWN
        low-precision `sdpa_math` lands from the f32/f64 one on the same
        inputs. That reference is a materialized P in the input dtype, so it
        degrades far faster than a flash kernel with f32 accumulators does as
        the softmax sharpens -- bf16, 289x289, hdim 512, normal inputs:

            sm_scale   kernel adiff   ref_error   threshold (3x)
            rsqrt(512)    0.0019       0.0050        0.0150
            1.2           0.0139       0.5340        1.6021

        The tolerance grows 100x while the kernel's error grows 7x, so at a
        large `sm_scale` the fudge factor stops discriminating: it admits the
        0.168 above with a factor of 9 to spare. A test of precision AT a large
        `sm_scale` therefore has to bring its own absolute bound, which is what
        `max_adiff` is. The fudge factor is still checked first -- it is the
        one that catches a NaN.

    Forward only, and self-contained rather than routed through
    `core_test_op_bwd`: the property is a forward one, the backward backend is
    irrelevant to it, and `_do_test_op_fwd` lives in test_forward.py, which
    conftest.py excludes from directory collection.
    '''
    # Same bound core_test_op_bwd applies, for the same reason: gfx11 has no
    # kernel above hdim 256. Coverage survives it -- the bf16 hdim 256 cases
    # are 0.1165 and 0.1246 Q-folded against a 0.03 bound, so the defect is
    # still caught on a chip that never reaches 512.
    if 'gfx11' in torch.cuda.get_device_properties(0).gcnArchName:
        if D_HEAD > 256:
            pytest.skip("Skip hdim > 256 on gfx11 arch due to register pressure.")
    if sm_scale == 'l1':
        sm_scale = 1.0 / D_HEAD
    elif sm_scale == 'l2':
        sm_scale = 1.0 / math.sqrt(D_HEAD)
    torch.manual_seed(20)
    transpose = (1, 2) if storage_flip else None
    ctx = SdpaContext(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, dtype,
                      bias_type=None, storage_flip=transpose, device=device_str)
    refill_inputs_normal(ctx)
    ctx.create_ref_inputs()
    ctx.set_require_grads(skip_dq=True, skip_dk_dv=True, skip_db=True)
    q, k, v, b = ctx.dev_tensors
    ext = AttentionExtraArgs(return_encoded_softmax=False,
                             autotune=False,
                             return_autotune=False)
    tri_out, _, _ = attention(q, k, v, b, causal, sm_scale, 0.0, ext)
    sdpa_params = SdpaParams(causal=causal, sm_scale=sm_scale, dropout_p=0.0, dropout_mask=None)
    ref_out, _ = ctx.compute_ref_forward(sdpa_params)

    is_allclose, adiff, _, _, tfts = ctx.validate_with_reference(tri_out, None, no_backward=True,
                                                                return_target_fudge_factors=True)
    if not is_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_out.to(device=tri_out.device) - tri_out)).cpu().numpy(),
                                   ref_out.shape)
        print(f'{err_idx=}')
        print(f'{tri_out[err_idx]=}')
        print(f'{ref_out[err_idx]=}')
    assert is_allclose, f'Forward pass {is_allclose=} {tfts=}'
    assert adiff <= max_adiff, \
        f'Forward pass {adiff=} exceeds the absolute bound {max_adiff=} at {sm_scale=} ' \
        f'(the fudge factor {tfts=} did not catch it)'
    print(f'{adiff=}')

def core_test_logsumexp_scaling(dtype):
    REF_VALUE = 2.79018449783325195
    device = 'cuda'
    q = torch.eye(16, device=device, dtype=dtype).reshape((1,1,16,16))
    k = torch.eye(16, device=device, dtype=dtype).reshape((1,1,16,16))
    v = torch.eye(16, device=device, dtype=dtype).reshape((1,1,16,16))
    b = None
    causal = False
    sm_scale = 1.0 / math.sqrt(16)
    dropout_p = 0.0

    ext = AttentionExtraArgs(return_encoded_softmax=False,
                             autotune=False,
                             return_autotune=False,
                             return_logsumexp=True)
    tri_out, _, L = attention(q, k, v, b, causal, sm_scale, dropout_p, ext)
    ref_tensor = torch.full_like(L, REF_VALUE)
    # allclose's default rtol of 1e-5 is fp32's tolerance, but L is the log of a
    # sum of exponentials of bf16/fp16 inputs, and the exp2 and the accumulation
    # order differ between backends. Measured relative error against REF_VALUE:
    # 1.5e-06 for fp16 and 1.3e-05 for bf16, identically on the Triton reference
    # and on both flyc backends, so bf16 fails the default by a hair on all of
    # them. 1e-4 covers that with room to spare and is still three orders of
    # magnitude tighter than what this test is here to catch, which is a base-2
    # versus base-e mixup in the LSE scaling -- an error of a factor of 1.44.
    assert torch.allclose(L, ref_tensor, rtol=1e-4)

# 0.0 takes the qk_scale == 0 path; -1.2 is the harder negative case.
NONPOS_SCALES = [0.0, -1.2]


def core_test_nonpositive_scale_symmetry(dtype, sm_scale, seqlen_q, seqlen_k):
    if SKIP_BWD:
        pytest.skip('SKIP_BWD=1 excludes backward checks')
    if BWD_IMPL == 'aiter':
        pytest.skip('AITER ASM does not support matrix bias')
    # Q = K = 0, so every score is `bias_val` regardless of sm_scale (including
    # 0 and negative). Softmax is uniform over seqlen_k keys. This is the
    # non-positive-scale analogue of core_test_matrix_bias_fwd_bwd_symmetry, with
    # partial tiles and an exact dB / dV check.
    device = 'cuda'
    bias_val = 16.0
    D_HEAD = 16
    torch.manual_seed(20)
    q = torch.zeros((1, 1, seqlen_q, D_HEAD), device=device, dtype=dtype, requires_grad=True)
    k = torch.zeros((1, 1, seqlen_k, D_HEAD), device=device, dtype=dtype, requires_grad=True)
    v = torch.randn((1, 1, seqlen_k, D_HEAD), device=device, dtype=dtype, requires_grad=True)
    b = torch.full((1, 1, seqlen_q, seqlen_k), bias_val, device=device, dtype=dtype, requires_grad=True)

    ext = AttentionExtraArgs(return_encoded_softmax=False,
                             autotune=False,
                             return_autotune=False,
                             return_logsumexp=True)
    tri_out, _, L = attention(q, k, v, b, False, sm_scale, 0.0, ext)
    dout = torch.randn_like(tri_out)
    dq, dk, dv, db = torch.autograd.grad(tri_out, [q, k, v, b], dout)

    inv_n = 1.0 / seqlen_k
    v_f = v.float()
    dout_f = dout.float()
    out_ref = v_f.mean(dim=2, keepdim=True).expand_as(tri_out)
    l_ref = bias_val + math.log(seqlen_k)
    dv_ref = (inv_n * dout_f.sum(dim=2, keepdim=True)).expand_as(dv)
    # dB[i,j] = P_ij * (dO_i·V_j - dO_i·Out_i); P_ij = 1/seqlen_k
    do_dot_v = torch.einsum('bhqd,bhkd->bhqk', dout_f, v_f)
    do_dot_out = torch.einsum('bhqd,bhqd->bhq', dout_f, out_ref).unsqueeze(-1)
    db_ref = inv_n * (do_dot_v - do_dot_out)

    atol_lse = 1e-5 if dtype == torch.float32 else 1e-4
    atol = {torch.float32: 1e-4, torch.float16: 2e-3}.get(dtype, 5e-3)
    assert torch.allclose(L.float(), torch.full_like(L.float(), l_ref), atol=atol_lse, rtol=atol_lse), \
        f'L {L.flatten()[:4].tolist()} should be bias+ln(N)={l_ref}'
    assert torch.allclose(tri_out.float(), out_ref, atol=atol, rtol=atol), \
        f'Out should be mean(V), maxerr={(tri_out.float() - out_ref).abs().max().item()}'
    assert torch.allclose(dq.float(), torch.zeros_like(dq.float()), atol=atol, rtol=atol), \
        f'dQ should be 0, maxerr={dq.abs().max().item()}'
    assert torch.allclose(dk.float(), torch.zeros_like(dk.float()), atol=atol, rtol=atol), \
        f'dK should be 0, maxerr={dk.abs().max().item()}'
    assert torch.allclose(dv.float(), dv_ref, atol=atol, rtol=atol), \
        f'dV off by {(dv.float() - dv_ref).abs().max().item()}'
    assert torch.allclose(db.float(), db_ref, atol=atol, rtol=atol), \
        f'dB off by {(db.float() - db_ref).abs().max().item()}'


def core_test_matrix_bias_fwd_bwd_symmetry(dtype, bias_val):
    # Softmax over a single key is exactly 1 whatever the bias is, so the fwd must
    # pass V through untouched, the saved LSE must equal the bias, and the bwd must
    # pass dO through to dV. The bwd recomputes p from the LSE the fwd saved, so all
    # three only hold while both kernels apply the log2(e) factor to the bias at the
    # same precision. Spelling the fwd scale as a bare `bias * 1.44269504089`
    # evaluates it at the bias dtype, rounding log2(e) to 1.4453125 in bf16 while the
    # bwd applies it in fp32; dV then comes back as
    # 2**(bias * (log2(e)_fp32 - log2(e)_bf16)), i.e. 0.973 at bias=16 and 0.891 at
    # bias=64. Biases this large are the point of the test: the random biases in
    # test_op_bwd_with_matrix_bias are small enough to hide the error under its
    # tolerance.
    device = 'cuda'
    D_HEAD = 16
    q = torch.zeros((1, 1, 1, D_HEAD), device=device, dtype=dtype, requires_grad=True)
    k = torch.zeros((1, 1, 1, D_HEAD), device=device, dtype=dtype, requires_grad=True)
    v = torch.ones((1, 1, 1, D_HEAD), device=device, dtype=dtype, requires_grad=True)
    b = torch.full((1, 1, 1, 1), bias_val, device=device, dtype=dtype)
    sm_scale = 1.0 / math.sqrt(D_HEAD)

    ext = AttentionExtraArgs(return_encoded_softmax=False,
                             autotune=False,
                             return_autotune=False,
                             return_logsumexp=True)
    tri_out, _, L = attention(q, k, v, b, False, sm_scale, 0.0, ext)

    # Not asserted exactly: converting the LSE to base 2 and back costs fp32 a few
    # 1e-7. ATOL still sits ~800x below the smallest error the asymmetry produces
    # (7.8e-3, at bias=4 in bf16).
    ATOL = 1e-5
    assert torch.allclose(L, torch.full_like(L, bias_val), atol=ATOL, rtol=ATOL), \
        f'lse {L.flatten().tolist()} should equal bias {bias_val}'
    assert torch.allclose(tri_out, v, atol=ATOL, rtol=ATOL), \
        f'single-key fwd should return V, off by {(tri_out - v).abs().max().item()}'

    dout = torch.ones_like(tri_out)
    dq, dk, dv = torch.autograd.grad(tri_out, [q, k, v], dout)
    assert torch.allclose(dv, dout, atol=ATOL, rtol=ATOL), \
        f'single-key bwd should return dO as dV, off by {(dv - dout).abs().max().item()}'

def core_test_large_bf16_nan_values(hdim):
    real_device = "cuda" if not AOTRITON_TORCH_ONLY_USE_CPU else "cpu"
    q = torch.full((1, 1, 1, hdim), 133120.0, dtype=torch.bfloat16, device=real_device)
    k = torch.full((1, 1, 1, hdim), 133120.0, dtype=torch.bfloat16, device=real_device)
    v = torch.full((1, 1, 1, hdim), 133120.0, dtype=torch.bfloat16, device=real_device)
    b = None
    from torch.nn.functional import scaled_dot_product_attention
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        out = scaled_dot_product_attention(q, k, v)
    print(out)

    causal = False
    sm_scale = 0.125
    dropout_p = 0
    ext = AttentionExtraArgs(return_encoded_softmax=causal,
                             autotune=False,
                             return_autotune=False)
    tri_out, encoded_softmax, _ = attention(q, k, v, b, causal, sm_scale, dropout_p, ext)

    print(tri_out)
    assert not torch.isnan(tri_out).any(), "Output should not contain NaNs!"

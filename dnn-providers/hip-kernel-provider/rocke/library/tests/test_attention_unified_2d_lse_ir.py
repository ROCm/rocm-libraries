# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-only IR tests for opt-in LSE output on the tiled 2D unified-attention
kernels (gfx942 + gfx950). No GPU: specs are built and lowered to LLVM text."""

import importlib
import re

import pytest

from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.core.passes import eliminate_dead_pure_ops
from rocke.helpers.tiling.lds_conflict import NumBuilder

_BASE = dict(
    head_size=128,
    block_size=64,
    num_query_heads=4,
    num_kv_heads=2,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    use_alibi=False,
)

# Spec kwargs that select each epilogue branch of build_unified_attention_2d_tiled.
BRANCH_SPEC_KW = {
    "gfx950": {
        "transposed32": dict(
            use_mfma_32x32=True,
            block_m_per_warp=32,
            use_transposed_qk_32x32=True,
            use_transposed_scalar_state=True,
        ),
        "mfma32": dict(use_mfma_32x32=True, block_m_per_warp=32),
        "striped16": {},
    },
    "gfx942": {
        "transposed32": dict(
            use_mfma_32x32x8=True, use_transposed_qk_32x32=True, block_m_per_warp=32
        ),
        "mfma32": dict(use_mfma_32x32x8=True, block_m_per_warp=32),
        "striped16": {},
    },
}

# One FP32 LSE store per (writer lane, softmax-state slot), Python-unrolled.
_EXPECTED_LSE_STORES = {
    "gfx950": {"transposed32": 1, "mfma32": 16, "striped16": 4},
    "gfx942": {"transposed32": 1, "mfma32": 16, "striped16": 4},
}

_CASES = [
    (arch, branch)
    for arch in ("gfx950", "gfx942")
    for branch in ("transposed32", "mfma32", "striped16")
]
_SPEC_ARCHES = ["gfx950", "gfx942"]


def _module(arch):
    return importlib.import_module(f"kernels.{arch}.attention_tiled_2d")


def _spec(arch, branch, **kw):
    return _module(arch).UnifiedAttention2DTiledSpec(
        **_BASE, **BRANCH_SPEC_KW[arch][branch], **kw
    )


def _build(spec, arch):
    return _module(arch).build_unified_attention_2d_tiled(spec, arch=arch)


def _lower(spec, arch):
    return lower_kernel_to_llvm(_build(spec, arch), arch=arch)


def _walk(region):
    for op in region.ops:
        yield op
        for nested in op.regions:
            yield from _walk(nested)


def _drop_ops(region, drop):
    """Remove the ops ``drop`` selects, at any depth; return how many went."""
    kept, gone = [], 0
    for op in region.ops:
        for nested in op.regions:
            gone += _drop_ops(nested, drop)
        if drop(op):
            gone += 1
        else:
            kept.append(op)
    region.ops = kept
    return gone


def _drop_dead(kernel, guards=frozenset()):
    """Remove dead pure ops, and the scf.if ops in ``guards`` once empty."""
    while eliminate_dead_pure_ops(kernel.body) + _drop_ops(
        kernel.body,
        lambda op: id(op) in guards and not any(r.ops for r in op.regions),
    ):
        pass


def _strip_lse(kernel):
    """Remove the LSE output from an LSE-on kernel, in place, on the IR before
    lowering: the ``lse_ptr`` param, every global store through it, the scf.if
    guards those stores leave empty, and the pure ops that only fed them.
    Return how many stores went."""
    lse = next(p for p in kernel.params if p.name == "lse_ptr")
    kernel.params.remove(lse)

    def is_lse_store(op):
        return (
            op.name.startswith("memref.global_store")
            and op.operands[0].name == f"%{lse.name}"
        )

    guards = set()

    def find_guards(region, ifs):
        for op in region.ops:
            if is_lse_store(op):
                guards.update(ifs)
            inner = ifs + (id(op),) if op.name == "scf.if" else ifs
            for nested in op.regions:
                find_guards(nested, inner)

    find_guards(kernel.body, ())
    stores = _drop_ops(kernel.body, is_lse_store)
    _drop_dead(kernel, guards)
    suffix = re.search(r"_lse(hm)?$", kernel.name)
    assert suffix, kernel.name
    kernel.name = kernel.name[: suffix.start()]
    return stores


def _canonical_names(kernel):
    """Rename every non-param value by first use, in place. Value names come
    from a builder-wide counter, so LSE ops shift every later name.

    An op's result object is also the later ops' operand object, so map the
    original names first and rename each object once; renaming while walking
    would rekey a value on every reuse and can merge two values."""
    params = {f"%{p.name}" for p in kernel.params}
    names, values = {}, {}
    for op in _walk(kernel.body):
        for value in (*op.operands, *op.results):
            if value.name in params or id(value) in values:
                continue
            values[id(value)] = value
            names.setdefault(value.name, f"%v{len(names)}")
    for value in values.values():
        value.name = names[value.name]


def test_canonical_names_keep_reused_values_distinct():
    from rocke.core.ir import I32, IRBuilder

    b = IRBuilder("k")
    x = b.param("x", I32)
    first = b.add(x, x)
    second = b.add(first, x)
    third = b.mul(first, second)
    # Original names already in the canonical %vN space, in the reverse order.
    first.name, second.name = "%v1", "%v0"
    _canonical_names(b.kernel)
    assert (first.name, second.name, third.name) == ("%v0", "%v1", "%v2")


def _lse_store_count(ir):
    """FP32 global stores. The LSE-off 2D kernels have none (bf16/fp16 O)."""
    return sum(
        1
        for line in ir.splitlines()
        if "store float" in line and "ptr addrspace(1)" in line
    )


def _define_line(ir):
    return next(line for line in ir.splitlines() if line.startswith("define"))


@pytest.mark.parametrize("arch", _SPEC_ARCHES)
@pytest.mark.parametrize("bad", ["false", 0, 1, None])
def test_2d_spec_rejects_non_boolean_emit_lse(arch, bad):
    with pytest.raises(ValueError, match="emit_lse must be a bool"):
        _spec(arch, "striped16", emit_lse=bad)


@pytest.mark.parametrize("arch,branch", _CASES)
def test_2d_kernel_name_gains_lse_suffix_only_when_on(arch, branch):
    default = _spec(arch, branch)
    off = _spec(arch, branch, emit_lse=False)
    on = _spec(arch, branch, emit_lse=True)
    assert default.emit_lse is False
    assert off.kernel_name() == default.kernel_name()
    assert not default.kernel_name().endswith("_lse")
    assert on.kernel_name() == default.kernel_name() + "_lse"


# gfx942 also builds branch A with one softmax-state slot (production flash
# specs set it) and the q-major grid; the name rule must hold for those too.
@pytest.mark.parametrize(
    "branch,extra",
    [
        ("transposed32", {}),
        ("transposed32", {"use_transposed_scalar_state": True}),
        ("mfma32", {}),
        ("striped16", {}),
        ("striped16", {"use_q_major_grid": True}),
    ],
)
def test_gfx942_lse_name_is_off_name_plus_lse_suffix(branch, extra):
    off = _spec("gfx942", branch, **extra)
    on = _spec("gfx942", branch, emit_lse=True, **extra)
    assert off.emit_lse is False
    assert not off.kernel_name().endswith("_lse")
    assert on.kernel_name() == off.kernel_name() + "_lse"


@pytest.mark.parametrize("arch,branch", _CASES)
def test_2d_lse_off_ir_has_no_lse_param_or_store(arch, branch):
    default_ir = _lower(_spec(arch, branch), arch)
    off_ir = _lower(_spec(arch, branch, emit_lse=False), arch)
    assert off_ir == default_ir
    assert "lse_ptr" not in off_ir
    assert _lse_store_count(off_ir) == 0


@pytest.mark.parametrize("arch,branch", _CASES)
def test_2d_lse_on_ir_minus_lse_output_is_lse_off_ir(arch, branch):
    """LSE on adds only the LSE output: with its param, stores, their guards
    and the pure ops that fed only them removed, it lowers to the LSE-off
    text. Dead pure ops go from both sides (LLVM drops them anyway). The
    LSE-off text itself is pinned by rocke_representative_ir_sha256.json."""
    off = _build(_spec(arch, branch), arch)
    on = _build(_spec(arch, branch, emit_lse=True), arch)
    assert _strip_lse(on) == _EXPECTED_LSE_STORES[arch][branch]
    _drop_dead(off)
    for kernel in (off, on):
        _canonical_names(kernel)
    off_ir = lower_kernel_to_llvm(off, arch=arch)
    on_ir = lower_kernel_to_llvm(on, arch=arch)
    assert on_ir == off_ir, f"LSE on changes more than its output: {arch}/{branch}"


@pytest.mark.parametrize("arch,branch", _CASES)
def test_2d_lse_on_appends_lse_ptr_last_and_stores_once_per_slot(arch, branch):
    on_ir = _lower(_spec(arch, branch, emit_lse=True), arch)
    define = _define_line(on_ir)
    assert re.search(
        r"ptr addrspace\(1\) noalias writeonly nocapture align 4 %lse_ptr\) #\d+ \{$",
        define,
    ), define
    assert _lse_store_count(on_ir) == _EXPECTED_LSE_STORES[arch][branch]
    # natural_lse_from_log2_stats: (m + log2 l) * ln2
    assert "@llvm.log2.f32" in on_ir


# Additive-bias params are appended after qq_bias_stride_0; branch B rejects bias.
@pytest.mark.parametrize(
    "arch,branch", [("gfx950", "transposed32"), ("gfx950", "striped16")]
)
def test_2d_lse_ptr_stays_last_with_additive_bias(arch, branch):
    define = _define_line(
        _lower(_spec(arch, branch, emit_lse=True, use_additive_bias=True), arch)
    )
    assert "%additive_bias_sq_stride, ptr addrspace(1)" in define, define
    assert re.search(r"%lse_ptr\) #\d+ \{$", define), define


@pytest.mark.parametrize(
    "branch,extra,stores",
    [
        # Production flash specs set use_transposed_scalar_state (1 state slot).
        ("transposed32", {"use_transposed_scalar_state": True}, 1),
        # Swapped launch grid: same kv_head_idx / q_block names, same index.
        ("striped16", {"use_q_major_grid": True}, 4),
        ("transposed32", {"use_q_major_grid": True}, 1),
        # Two 16-row atoms per warp: 8 state slots per lane.
        ("striped16", {"block_m_per_warp": 32}, 8),
    ],
)
def test_gfx942_lse_variants_store_once_per_row_slot(branch, extra, stores):
    off_ir = _lower(_spec("gfx942", branch, **extra), "gfx942")
    on_ir = _lower(_spec("gfx942", branch, emit_lse=True, **extra), "gfx942")
    assert "lse_ptr" not in off_ir
    assert _lse_store_count(off_ir) == 0
    assert _lse_store_count(on_ir) == stores
    assert "@llvm.log2.f32" in on_ir


def _gfx942_branch_of(spec):
    if spec.use_mfma_32x32x8:
        return "transposed32" if spec.use_transposed_qk_32x32 else "mfma32"
    return "striped16"


@pytest.mark.parametrize(
    "fields,branch",
    [
        # Mandate shape, bf16 flash (_spec_gfx942_bf16_flash).
        (
            dict(dtype="bf16", total_q=8192, num_seqs=1, max_seqlen_q=8192),
            "transposed32",
        ),
        # fp16 flash (_spec_gfx942_fp16_flash).
        (
            dict(dtype="fp16", total_q=8192, num_seqs=1, max_seqlen_q=8192),
            "transposed32",
        ),
        # Decode-like 2D shape: generic narrow (_spec_generic_2d_non_gfx950).
        (dict(dtype="bf16", total_q=4, num_seqs=4, max_seqlen_q=1), "striped16"),
        # D256 bf16 prefill: generic narrow when the 4-warp cohort is not used.
        (
            dict(
                dtype="bf16", head_size=256, total_q=8192, num_seqs=1, max_seqlen_q=8192
            ),
            "striped16",
        ),
    ],
)
def test_gfx942_production_specs_use_branches_the_numeric_gate_covers(fields, branch):
    """Production gfx942 2D specs land on branch A or C, the branches the GPU
    numeric gate runs by default. Branch B (non-transposed x8) is reachable
    only through explicit or tuning specs, which the gate pins separately."""
    from kernels.common import attention_unified as au

    base = dict(
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_k=8192,
    )
    problem = au.UnifiedAttentionProblem(**{**base, **fields})
    assert _gfx942_branch_of(au._tiled_spec_from_problem(problem, "gfx942")) == branch


# ---------------------------------------------------------------------------
# gfx950 head-major LSE layout (lse_layout="head_major")
# ---------------------------------------------------------------------------

_GFX950_BRANCHES = ("transposed32", "mfma32", "striped16")


def _offsets(layout, q_lens, heads):
    """Evaluates the index arithmetic ``unified_lse_offset`` emits on ints, via
    ``NumBuilder`` (its ``add``/``mul``/``const_i32`` cover what this needs)."""
    from kernels.common._lse_store import unified_lse_offset

    b, out, start = NumBuilder(0), {}, 0
    for seq, q_len in enumerate(q_lens):
        for pos in range(q_len):
            for head in range(heads):
                out[(seq, pos, head)] = unified_lse_offset(
                    b,
                    layout,
                    cu_q_start=start,
                    q_len=q_len,
                    pos=pos,
                    head=head,
                    num_heads=heads,
                )
        start += q_len
    return out


@pytest.mark.parametrize("q_lens", [[4, 4, 4], [3, 1, 7], [5]])
def test_lse_offsets_are_a_bijection_onto_the_packed_buffer(q_lens):
    """Both layouts write every element of the total_q * Hq buffer exactly once,
    and never write past another sequence's block."""
    heads = 3
    for layout in ("token_major", "head_major"):
        offsets = _offsets(layout, q_lens, heads)
        assert sorted(offsets.values()) == list(range(sum(q_lens) * heads)), layout
        start = 0
        for seq, q_len in enumerate(q_lens):
            mine = {o for (s, _, _), o in offsets.items() if s == seq}
            assert mine == set(range(start * heads, (start + q_len) * heads)), layout
            start += q_len


def test_head_major_offsets_are_the_bhs1_stats_layout():
    """Equal-length sequences: head_major is [B, Hq, S, 1] with strides
    [Hq*S, S, 1] (the SDPA stats tensor); token_major is [B*S, Hq, 1]."""
    batch, seq_len, heads = 2, 5, 3
    head_major = _offsets("head_major", [seq_len] * batch, heads)
    token_major = _offsets("token_major", [seq_len] * batch, heads)
    for (b, s, h), off in head_major.items():
        assert off == (b * heads + h) * seq_len + s
        assert token_major[(b, s, h)] == (b * seq_len + s) * heads + h


def test_lse_offset_rejects_an_unknown_layout():
    from kernels.common._lse_store import unified_lse_offset

    with pytest.raises(ValueError, match="lse layout must be one of"):
        unified_lse_offset(
            NumBuilder(0), "bshd", cu_q_start=0, q_len=1, pos=0, head=0, num_heads=1
        )


@pytest.mark.parametrize("bad", ["HEAD_MAJOR", "bhsd", None])
def test_gfx950_spec_rejects_an_unknown_lse_layout(bad):
    with pytest.raises(ValueError, match="lse_layout must be one of"):
        _spec("gfx950", "striped16", emit_lse=True, lse_layout=bad)


def test_gfx950_head_major_requires_emit_lse():
    """An LSE-off spec with a layout would be a second name for one kernel."""
    with pytest.raises(ValueError, match="requires emit_lse"):
        _spec("gfx950", "striped16", lse_layout="head_major")


@pytest.mark.parametrize("branch", _GFX950_BRANCHES)
def test_gfx950_head_major_name_is_off_name_plus_lsehm(branch):
    off = _spec("gfx950", branch)
    token = _spec("gfx950", branch, emit_lse=True, lse_layout="token_major")
    head = _spec("gfx950", branch, emit_lse=True, lse_layout="head_major")
    assert token.kernel_name() == off.kernel_name() + "_lse"
    assert head.kernel_name() == off.kernel_name() + "_lsehm"


@pytest.mark.parametrize("branch", _GFX950_BRANCHES)
def test_gfx950_explicit_token_major_lowers_to_the_default_lse_ir(branch):
    """lse_layout's default is the layout every LSE-on kernel had before it."""
    default = _lower(_spec("gfx950", branch, emit_lse=True), "gfx950")
    token = _lower(
        _spec("gfx950", branch, emit_lse=True, lse_layout="token_major"), "gfx950"
    )
    assert token == default


@pytest.mark.parametrize("branch", _GFX950_BRANCHES)
def test_gfx950_head_major_minus_lse_output_is_lse_off_ir(branch):
    """Head-major changes only the LSE store index, never O."""
    off = _build(_spec("gfx950", branch), "gfx950")
    on = _build(
        _spec("gfx950", branch, emit_lse=True, lse_layout="head_major"), "gfx950"
    )
    assert _strip_lse(on) == _EXPECTED_LSE_STORES["gfx950"][branch]
    _drop_dead(off)
    for kernel in (off, on):
        _canonical_names(kernel)
    assert lower_kernel_to_llvm(on, arch="gfx950") == lower_kernel_to_llvm(
        off, arch="gfx950"
    )


@pytest.mark.parametrize("branch", _GFX950_BRANCHES)
def test_gfx950_head_major_ir_differs_from_token_major(branch):
    token = _lower(_spec("gfx950", branch, emit_lse=True), "gfx950")
    head = _lower(
        _spec("gfx950", branch, emit_lse=True, lse_layout="head_major"), "gfx950"
    )
    assert head != token
    assert _lse_store_count(head) == _lse_store_count(token)


def test_tuning_cache_key_leaves_out_the_default_lse_layout():
    """A token-major spec keys as before lse_layout existed; head-major rekeys."""
    from dataclasses import replace

    from dispatch.attention.gfx950_unified import GFX950_TUNING_VARIANTS
    from dispatch.attention.unified_rules import canonicalize_tuning_spec
    from kernels.common import attention_unified as au

    problem = au.UnifiedAttentionProblem(
        total_q=256,
        num_seqs=1,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=256,
        max_seqlen_k=256,
        dtype="bf16",
        emit_lse=True,
    )
    variant = next(
        v
        for v in GFX950_TUNING_VARIANTS
        if v.spec_id == "gfx950_u2d_narrow_nw1_mw16_t1xb_llvm"
    )
    spec, why = canonicalize_tuning_spec(problem, variant, {})
    assert spec is not None, why
    names = [name for name, _ in spec.cache_key()[4]]
    assert "emit_lse" in names and "lse_layout" not in names
    head = replace(spec, kernel_spec=replace(spec.kernel_spec, lse_layout="head_major"))
    assert ("lse_layout", repr("head_major")) in head.cache_key()[4]

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host fixtures and scalar addressing support for paged decode tests."""

from dataclasses import asdict, replace

import pytest


class Tensor:
    next_pointer = 4096

    def __init__(self, shape, dtype="torch.float16"):
        self.pointer = Tensor.next_pointer
        Tensor.next_pointer += 1 << 48
        self.shape, self.dtype = shape, dtype
        self.device, self.is_cuda = "cuda:0", True

    def stride(self, dim=None):
        from math import prod

        strides = tuple(prod(self.shape[i + 1 :]) for i in range(len(self.shape)))
        return strides if dim is None else strides[dim]

    def numel(self):
        import math

        return math.prod(self.shape)

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return self.pointer


def inputs(page=1):
    from kernels.common.attention_unified import UnifiedAttentionProblem

    p = UnifiedAttentionProblem(
        total_q=3,
        num_seqs=3,
        num_query_heads=8,
        num_kv_heads=2,
        head_size=64,
        block_size=page,
        max_seqlen_q=1,
        max_seqlen_k=33,
        dtype="fp16",
        clamp_arch="gfx950",
    )
    ts = [
        Tensor((3, 8, 64)),
        Tensor((99, page, 2, 64)),
        Tensor((99, page, 2, 64)),
        Tensor((3, 8, 64)),
        Tensor((4,), "torch.int32"),
        Tensor((3,), "torch.int32"),
        Tensor((3, (33 + page - 1) // page), "torch.int32"),
    ]
    return p, ts


class AddressInterpreter:
    """Evaluate the emitted scalar address operations against a finite table."""

    const_i32 = staticmethod(int)
    add = staticmethod(lambda x, y: x + y)
    mul = staticmethod(lambda x, y: x * y)
    div = staticmethod(lambda x, y: x // y)
    mod = staticmethod(lambda x, y: x % y)
    cmp_lt = staticmethod(lambda x, y: x < y)
    select = staticmethod(lambda pred, x, y: x if pred else y)

    @staticmethod
    def global_load(table, index, dtype, align):
        assert 0 <= index < len(table), "speculative page-table read escaped allocation"
        return table[index]


def _paged_runtime_kwargs(arch, page):
    from dispatch.attention import AttentionRequest, dispatch_attention

    p, ts = inputs(page)
    p = replace(p, clamp_arch=arch)
    req = AttentionRequest(
        batch=3,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=33,
        hdim_q=64,
        hdim_v=64,
        arch=arch,
        dtype="fp16",
        kv_block_size=page,
        target_ctas=8,
        algorithm="paged_decode_t32",
    )
    spec = dispatch_attention(req).spec
    # Gather launches require the bounded ABI. Page-contained controls retain
    # the legacy direct/structural loader contract.
    if not spec.kernel_spec.uses_paged_gather:
        spec = replace(
            spec,
            kernel_spec=replace(spec.kernel_spec, metadata_containment=False),
            reduce_spec=replace(spec.reduce_spec, metadata_containment=False),
        )
    return {
        "problem": p,
        "q": ts[0],
        "k": ts[1],
        "v": ts[2],
        "out": ts[3],
        "cu_seqlens_q": ts[4],
        "seqused_k": ts[5],
        "block_table": ts[6],
        "softmax_scale": 1 / 8,
        "softcap": 0,
        "backend": "3d",
        "tuning_spec": spec,
    }


def _structural_paged_spec(concrete, *, kernel_spec=None, retarget=None):
    """Expose only the runtime protocol, with no candidate identity metadata."""
    from types import SimpleNamespace

    segment = concrete.kernel_spec
    snapshot = asdict(segment)
    snapshot.update(
        tile_size=segment.tile_size, uses_paged_gather=segment.uses_paged_gather
    )
    spec = SimpleNamespace(
        arch=concrete.arch,
        path=concrete.path,
        kernel_spec=(
            kernel_spec if kernel_spec is not None else SimpleNamespace(**snapshot)
        ),
        reduce_spec=concrete.reduce_spec,
        compile_backend=concrete.compile_backend,
        allow_unsupported=concrete.allow_unsupported,
        cache_key=concrete.cache_key,
        build=concrete.build,
        launch_grid=concrete.launch_grid,
        launch_block=concrete.launch_block,
    )
    spec.with_num_kv_blocks = retarget if retarget is not None else lambda count: spec
    assert not hasattr(spec, "candidate_name") and not hasattr(spec, "tuning_id")
    return spec


def _forbid_paged_launch(arch, monkeypatch):
    from kernels.common import attention_unified as au

    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: arch)

    def forbidden(**kwargs):
        pytest.fail("invalid paged features reached launch")

    monkeypatch.setattr(au, "_run_3d_tiled", forbidden)
    return au


class MetadataTensor(Tensor):
    def __init__(self, shape, values):
        super().__init__(shape, "torch.int32")
        self.values, self.reads, self.allow_read = values, [], False

    def stride(self, dim):
        return self.shape[1] if dim == 0 else 1

    def detach(self):
        return self

    def cpu(self):
        self.reads.append("cpu")
        assert self.allow_read, "unexpected device metadata read"
        return self

    def tolist(self):
        self.reads.append("tolist")
        return self.values


@pytest.fixture
def paged_binding_case():
    from dispatch.attention import AttentionRequest, dispatch_attention

    kwargs = _paged_runtime_kwargs("gfx950", 1)
    req = AttentionRequest(
        batch=3,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=33,
        hdim_q=64,
        hdim_v=64,
        arch="gfx950",
        dtype="fp16",
        kv_block_size=1,
        target_ctas=8,
    )
    result = dispatch_attention(req)
    tensors = {name: kwargs[name] for name in ("problem", "q", "k", "v", "out")}
    tensors.update(
        cu_seqlens_q=MetadataTensor((4,), [0, 1, 2, 3]),
        seqused_k=MetadataTensor((3,), [33, 17, 0]),
        block_table=MetadataTensor((3, 33), [list(range(33)) for _ in range(3)]),
    )
    return req, result, tensors


def _retarget_binding_case(case, arch, page):
    from dispatch.attention import dispatch_attention

    req, _, tensors = case
    req = replace(req, arch=arch, kv_block_size=page, algorithm="paged_decode_t32")
    tensors = dict(tensors)
    tensors["problem"] = replace(tensors["problem"], clamp_arch=arch, block_size=page)
    tensors["k"] = Tensor((99, page, 2, 64))
    tensors["v"] = Tensor((99, page, 2, 64))
    return req, dispatch_attention(req), tensors

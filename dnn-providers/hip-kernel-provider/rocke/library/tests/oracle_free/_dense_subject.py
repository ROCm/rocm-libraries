# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Narrow subject adapter of the oracle-free lanes for the shipped dense attention
kernels.

Torch-free end to end: inputs and the output are raw ``uint16`` host arrays,
device buffers are :class:`rocke.runtime.DeviceMem`, and the launch goes through
the architecture's public ``run_attention_dense_torch``. That entry accepts
device buffers whenever the spec has no sinks, no LSE and no KV split, which is
exactly the scope the adapter admits; the pinned SDPA lane launches it the same
way. Dispatch, the runner and the HIP runtime are imported only inside the
functions that use them, so CPU collection and the analytical contracts need
neither a GPU nor ROCm.

A lane describes its case through :class:`DenseCase`. Adding an architecture is
one :data:`DENSE_SUBJECTS` entry plus its cases in :data:`QUICK_SUBJECT_CASES` or
:data:`PERMUTATION_SUBJECT_CASES`.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import numpy as np

from ._analytical import UNIFORM_QUICK_CASES, UniformCase
from ._permutation import PERMUTATION_QUICK_CASES, PermutationCase

_POINTER_ALIGNMENT = 16
# Quiet-NaN output prefill: a coordinate the kernel never writes reads back as
# invalid instead of passing as a stale zero.
_NAN_BITS = {"fp16": 0x7E00, "bf16": 0x7FC0}
# Dispatch mask ordinal of each case mask: NO_MASK, or TOP_LEFT_CAUSAL with the
# window carried by ``sliding_window``.
_MASK_TYPES = {"none": 0, "causal": 1, "window": 1}


class DenseCase(Protocol):
    """What the adapter reads from one lane's case.

    ``mask`` is ``"none"``, ``"causal"`` or ``"window"`` (with ``window > 0``).
    ``require_scope`` raises ``ValueError`` for a case the lane's derivation does
    not cover."""

    batch: int
    seqlen_q: int
    seqlen_k: int
    nhead_q: int
    nhead_k: int
    head_dim: int
    dtype: str

    @property
    def mask(self) -> str: ...

    @property
    def window(self) -> int: ...

    @property
    def output_shape(self) -> tuple[int, int, int, int]: ...

    @property
    def key_shape(self) -> tuple[int, int, int, int]: ...

    @property
    def case_id(self) -> str: ...

    def require_scope(self) -> None: ...


@dataclass(frozen=True)
class DenseSubject:
    """How one architecture's dense kernel is selected and launched."""

    spec_id: str
    candidate_name: str
    runner_module: str


DENSE_SUBJECTS: Mapping[str, DenseSubject] = MappingProxyType(
    {
        "gfx942": DenseSubject(
            "gfx942_dense", "attention_gfx942_dense", "kernels.gfx942.attention_dense"
        ),
        "gfx950": DenseSubject(
            "gfx950_dense_grid",
            "attention_gfx950_dense_grid",
            "kernels.gfx950.attention_dense",
        ),
    }
)

# The GPU lane's (arch, case) matrix. gfx950 has a subject but no cases yet: PR
# CI has no gfx950 provider runner, so its cases are tracked separately.
QUICK_SUBJECT_CASES: tuple[tuple[str, UniformCase], ...] = tuple(
    ("gfx942", case) for case in UNIFORM_QUICK_CASES
)
# The key-permutation lane's (arch, case) matrix: unmasked cases on gfx942.
PERMUTATION_SUBJECT_CASES: tuple[tuple[str, PermutationCase], ...] = tuple(
    ("gfx942", case) for case in PERMUTATION_QUICK_CASES
)


@dataclass(frozen=True)
class ResolvedDenseSubject:
    """Dispatch-resolved dense subject for one architecture and case."""

    arch: str
    case_id: str
    tuning_spec: object
    candidate_name: str

    def key_tiles(self, seqlen_k: int) -> int:
        """KV tiles one query row's single pass visits over ``seqlen_k`` keys."""
        return -(-seqlen_k // int(self.tuning_spec.kernel_spec.block_n))


def softmax_scale(case: DenseCase) -> float:
    """The softmax scale every launch passes: ``1 / sqrt(head_dim)``."""
    return 1.0 / math.sqrt(case.head_dim)


def _validate_host_bits(case: DenseCase, q, k, v) -> None:
    """Validate the dense ABI on the raw-bit host inputs."""
    expected_shapes = {
        "q": case.output_shape,
        "k": case.key_shape,
        "v": case.key_shape,
    }
    for name, array in (("q", q), ("k", k), ("v", v)):
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} must be a numpy array of raw bits")
        if array.shape != expected_shapes[name]:
            raise ValueError(f"{name} shape {array.shape} != {expected_shapes[name]}")
        if array.dtype != np.uint16:
            raise ValueError(f"{name} dtype {array.dtype} != uint16 raw bits")
        if not array.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")


def _validate_device_buffers(
    buffers: Mapping[str, object], nbytes: Mapping[str, int]
) -> None:
    """Check each device buffer's size, alignment and separation.

    Duck-typed on ``ptr()`` / ``nbytes()`` so the contract is CPU-testable with
    small fakes; ``run_dense_case`` passes real ``DeviceMem`` objects."""
    spans: dict[str, tuple[int, int]] = {}
    for name, buffer in buffers.items():
        size = int(buffer.nbytes())
        if size != nbytes[name]:
            raise ValueError(f"{name} device size {size} != host size {nbytes[name]}")
        pointer = int(buffer.ptr())
        if pointer == 0:
            raise ValueError(f"{name} device pointer is null")
        if pointer % _POINTER_ALIGNMENT:
            raise ValueError(
                f"{name} pointer must be {_POINTER_ALIGNMENT}-byte aligned"
            )
        spans[name] = (pointer, pointer + size)

    names = tuple(spans)
    for index, left_name in enumerate(names):
        left_start, left_end = spans[left_name]
        for right_name in names[index + 1 :]:
            right_start, right_end = spans[right_name]
            if left_start < right_end and right_start < left_end:
                raise ValueError(f"{left_name} overlaps {right_name}")


def _validate_supported_case(arch: str, case: DenseCase) -> None:
    """Reject an architecture without a subject or a case outside its lane's
    derivation."""
    if arch not in DENSE_SUBJECTS:
        raise ValueError(
            f"no dense oracle-free subject for {arch}; known: {sorted(DENSE_SUBJECTS)}"
        )
    if case.mask not in _MASK_TYPES:
        raise ValueError(f"case mask must be one of {sorted(_MASK_TYPES)}")
    case.require_scope()


def _verify_resolved_spec(arch: str, case: DenseCase, tuning_spec) -> None:
    """Check the dispatch-resolved spec against the scope the lanes derive for.

    The uniform-probability proof and the key-permutation bound both cover every
    single-pass dense variant, so tile, grid, persistence, fast exp2 and
    scheduling knobs are deliberately free: a retune keeps running, and a defect
    in any of them still fails the lane's policy. What they do not cover is
    refused here, before any launch, as a failure: another candidate, a
    KV-segment split (its partial-softmax combine), sinks and LSE (extra buffers
    and epilogues the adapter does not supply), ragged tail tiles, the
    bottom-right band, and paged or varlen layouts."""
    subject = DENSE_SUBJECTS[arch]
    if tuning_spec.path != "dense":
        raise AssertionError(f"dispatch returned path={tuning_spec.path!r}")
    if tuning_spec.arch != arch:
        raise AssertionError(f"dispatch returned arch={tuning_spec.arch!r}")
    if tuning_spec.candidate_name != subject.candidate_name:
        raise AssertionError(
            f"dispatch returned candidate={tuning_spec.candidate_name!r}"
        )
    spec = tuning_spec.kernel_spec
    expected = {
        "batch": case.batch,
        "seqlen_q": case.seqlen_q,
        "seqlen_kv": case.seqlen_k,
        "num_query_heads": case.nhead_q,
        "num_kv_heads": case.nhead_k,
        "head_size": case.head_dim,
        "dtype": case.dtype,
        "causal": case.mask != "none",
        "sliding_window": case.window if case.mask == "window" else 0,
    }
    for field, value in expected.items():
        actual = getattr(spec, field)
        if actual != value:
            raise AssertionError(
                f"resolved spec {field}={actual!r}, expected {value!r}"
            )
    if spec.use_sinks:
        raise AssertionError(
            "the oracle-free dense checks require attention sinks to be disabled"
        )
    if spec.emit_lse:
        raise AssertionError(
            "the oracle-free dense checks require the direct non-LSE epilogue"
        )
    if spec.kv_segments != 1:
        raise AssertionError(
            "the oracle-free dense checks require single-pass KV, "
            f"kv_segments={spec.kv_segments}"
        )
    if spec.ragged:
        raise AssertionError("the oracle-free dense checks require a non-ragged shape")
    if spec.causal_bottom_right:
        raise AssertionError(
            "the oracle-free dense checks require the top-left causal band"
        )
    if spec.paged or spec.varlen:
        raise AssertionError(
            "the oracle-free dense checks require a dense, non-paged, "
            "non-varlen layout"
        )


def _resolve_tuning_spec(arch: str, case: DenseCase):
    """Resolve the subject's dense candidate at its default configuration."""
    from dispatch.attention import AttentionRequest, attention_tuning_spec

    request = AttentionRequest(
        batch=case.batch,
        nhead_q=case.nhead_q,
        nhead_k=case.nhead_k,
        seqlen_q=case.seqlen_q,
        seqlen_k=case.seqlen_k,
        hdim_q=case.head_dim,
        hdim_v=case.head_dim,
        arch=arch,
        mask_type=_MASK_TYPES[case.mask],
        sliding_window=case.window if case.mask == "window" else 0,
        dtype=case.dtype,
    )
    tuning_spec = attention_tuning_spec(request, DENSE_SUBJECTS[arch].spec_id)
    _verify_resolved_spec(arch, case, tuning_spec)
    return tuning_spec


def resolve_dense_subject(arch: str, case: DenseCase) -> ResolvedDenseSubject:
    """Resolve one supported case on one architecture."""
    _validate_supported_case(arch, case)
    tuning_spec = _resolve_tuning_spec(arch, case)
    return ResolvedDenseSubject(
        arch=arch,
        case_id=case.case_id,
        tuning_spec=tuning_spec,
        candidate_name=str(tuning_spec.candidate_name),
    )


def run_dense_case(
    arch: str,
    case: DenseCase,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    subject: ResolvedDenseSubject | None = None,
) -> np.ndarray:
    """Launch one validated subject on raw input bits and return raw output bits.

    ``q``/``k``/``v`` are ``uint16`` FP16/BF16 bit arrays in the runner layout
    (``[B, Sq, Hq, D]`` and ``[B, Sk, Hkv, D]``). The output is NaN-prefilled and
    read back without any float conversion, so the analyzer judges exactly the
    stored bits. Numerical judgment is left to the analyzer."""
    from rocke.runtime import DeviceMem, Runtime
    from rocke.runtime.host_buffers import as_u8_buffer

    _validate_supported_case(arch, case)
    _validate_host_bits(case, q, k, v)
    resolved = resolve_dense_subject(arch, case) if subject is None else subject
    if (resolved.arch, resolved.case_id) != (arch, case.case_id):
        raise ValueError(
            f"resolved subject {resolved.arch}/{resolved.case_id} != "
            f"{arch}/{case.case_id}"
        )
    _verify_resolved_spec(arch, case, resolved.tuning_spec)
    runner = importlib.import_module(DENSE_SUBJECTS[arch].runner_module)

    # Private writable copies: ``as_u8_buffer`` views storage via ``from_buffer``,
    # which refuses read-only arrays.
    host = {
        "q": q.copy(),
        "k": k.copy(),
        "v": v.copy(),
        "out": np.full(case.output_shape, _NAN_BITS[case.dtype], dtype=np.uint16),
    }
    runtime = Runtime()
    device = {name: DeviceMem(array.nbytes) for name, array in host.items()}
    _validate_device_buffers(
        device, {name: int(array.nbytes) for name, array in host.items()}
    )
    for name, array in host.items():
        runtime.memcpy_h2d(device[name].ptr(), as_u8_buffer(array), array.nbytes)
    runner.run_attention_dense_torch(
        spec=resolved.tuning_spec.kernel_spec,
        q=device["q"],
        k=device["k"],
        v=device["v"],
        out=device["out"],
        scale=softmax_scale(case),
    )
    runtime.sync()
    out = host["out"]
    runtime.memcpy_d2h(as_u8_buffer(out), device["out"].ptr(), out.nbytes)
    return out

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host launch glue of the layout-general attention backward.

Runs an :class:`~kernels.common.attention_bwd_plan.AttnBwdPlan` on one
stream: ``prep -> main -> convert(dQ) [-> convert(dK), convert(dV)]``, or
under ``dq_mode = "split"`` ``prep -> main -> dq [-> convert(dK),
convert(dV)]``, in the plan's launch order, with every kernel argument packed
in the ``rocke.attn_bwd.v3`` order (a dense-route plan's main launch, stage
``"dense"``, in the ``rocke.attn_dense_bwd.v1`` order, and its split dQ
launch, stage ``"dense_dq"``, in the ``rocke.attn_dense_bwd_dq.v1`` order).
With ``dkv_sink = "slices"`` the dK / dV converts are slice reduces (stage
``reduce``).

The caller owns every buffer. The workspace is supplied by the caller and
sized by ``plan.workspace_bytes`` (256-byte aligned base); the run function
allocates no device memory and issues no memset: the prep kernel initialises
every workspace byte a later kernel reads.

* :func:`attn_bwd_kernels` compiles (and loads) the plan's kernels, cached by
  kernel name, arch, LLVM flavor and comgr version; with
  ``ROCKE_BWD_HSACO_CACHE=<dir>`` the code objects are also read from and
  written to the on-disk cache of :mod:`kernels.common.attention_bwd_cache`
  (off by default).
* :func:`pack_attn_bwd_args` packs one launch's kernel arguments (host only).
* :func:`run_attn_bwd` checks the workspace and issues the launches.

Host validation. The run functions take device pointers only (the hipDNN
contract: the caller validates its tensors against the request it planned
with), so they cannot see a tensor's shape, strides or size. A caller that
holds the tensors (or their metadata) checks them first:

* :func:`check_attn_bwd_tensors`: every tensor of the request (shape, element
  strides, dtype, and the bytes addressable from its base pointer, which must
  cover the largest offset the kernels form) against the request;
* :func:`check_attn_bwd_lengths`: host-visible length data: ``SEQ_LEN``
  values in ``[0, S_max]`` (a larger value is rejected, never clamped; on the
  device an out-of-range value is clamped into ``[0, S_max]``, the defined
  behaviour for out-of-contract data), THD offsets non-decreasing,
  non-negative, whole tokens, each span at most ``S_max`` and inside the
  token bound / tensors;
* :func:`check_attn_dense_bwd_tensors`: a dense-family spec (compile-time
  shape) against its tensors and runtime batch.

Each raises ``ValueError("<CODE>: detail")`` with a code of
:data:`HOST_CHECK_CODES`; :class:`TensorMeta` describes one tensor (built
directly, or from a torch-like tensor with :meth:`TensorMeta.of`).
"""

from __future__ import annotations

import os
import struct
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

from kernels.common.attention_bwd import attn_bwd_params, build_attn_bwd_main
from kernels.common.attention_bwd_aux import (
    build_attn_bwd_convert,
    build_attn_bwd_prep,
    build_attn_bwd_reduce,
)
from kernels.common.attention_bwd_dq import build_attn_bwd_dq

__all__ = [
    "HOST_CHECK_CODES",
    "WORKSPACE_ALIGN",
    "TensorMeta",
    "attn_bwd_kernels",
    "check_attn_bwd_lengths",
    "check_attn_bwd_tensors",
    "check_attn_dense_bwd_tensors",
    "pack_attn_bwd_args",
    "run_attn_bwd",
]

WORKSPACE_ALIGN = 256

# Codes of the host checks (the ValueError message starts with one).
HOST_CHECK_CODES = (
    "MISSING_TENSOR",
    "TENSOR_DTYPE",
    "TENSOR_SHAPE",
    "TENSOR_STRIDES",
    "TENSOR_EXTENT",
    "SEQ_LEN_RANGE",
    "RAGGED_RANGE",
    "BLOCK_TABLE_RANGE",
    "BLOCK_TABLE_NOT_INJECTIVE",
)
ELEM_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "int32": 4, "int64": 8}
_DTYPE_NAMES = {
    "torch.float16": "fp16",
    "torch.bfloat16": "bf16",
    "torch.float32": "fp32",
    "torch.int32": "int32",
    "torch.int64": "int64",
    "float16": "fp16",
    "bfloat16": "bf16",
    "float32": "fp32",
    "int32": "int32",
    "int64": "int64",
}


@dataclass(frozen=True)
class TensorMeta:
    """Host view of one caller tensor.

    ``shape`` and ``strides`` (elements) in the same dim order, ``dtype`` a
    name of :data:`ELEM_BYTES` and ``nbytes`` the bytes addressable from the
    tensor's base pointer (its allocation from the base on).
    """

    shape: tuple
    strides: tuple
    dtype: str
    nbytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(x) for x in self.shape))
        object.__setattr__(self, "strides", tuple(int(x) for x in self.strides))
        if len(self.shape) != len(self.strides):
            raise ValueError(
                f"shape {self.shape} and strides {self.strides} differ in rank"
            )
        if self.dtype not in ELEM_BYTES:
            raise ValueError(f"dtype {self.dtype!r} not in {tuple(ELEM_BYTES)}")

    @classmethod
    def of(cls, t) -> "TensorMeta":
        """From a torch-like tensor (``shape``, ``stride()``, ``dtype``,
        ``element_size()``). ``nbytes`` is the storage from the tensor's
        offset on when the storage is reachable, else the tensor's own
        extent."""
        dtype = _DTYPE_NAMES.get(str(t.dtype))
        if dtype is None:
            raise ValueError(f"TENSOR_DTYPE: unsupported tensor dtype {t.dtype}")
        shape, strides = tuple(t.shape), tuple(t.stride())
        esize = int(t.element_size())
        try:
            nbytes = int(t.untyped_storage().nbytes()) - int(t.storage_offset()) * esize
        except AttributeError:
            nbytes = extent_elems(shape, strides) * esize
        return cls(shape, strides, dtype, nbytes)


def extent_elems(shape: Sequence[int], strides: Sequence[int]) -> int:
    """Elements from the base to the last addressed element, inclusive (0 for
    an empty tensor); strides are non-negative."""
    if any(int(n) == 0 for n in shape):
        return 0
    return 1 + sum((int(n) - 1) * int(s) for n, s in zip(shape, strides))


def _fail(code: str, detail: str):
    raise ValueError(f"{code}: {detail}")


def _meta(tensors: Mapping[str, TensorMeta], role: str) -> TensorMeta:
    m = tensors.get(role)
    if m is None:
        _fail("MISSING_TENSOR", f"no metadata for tensor role {role!r}")
    return m


def check_tensor(
    role: str,
    m: TensorMeta,
    *,
    dtype: str,
    shape: Sequence,
    strides: Sequence[int] | None,
) -> None:
    """One tensor against its expected dtype, shape (``None`` entries: any
    extent) and strides (compared on dims of extent > 1; ``None``: any), then
    its extent against ``nbytes``."""
    if m.dtype != dtype:
        _fail("TENSOR_DTYPE", f"{role}: dtype {m.dtype}, want {dtype}")
    if len(m.shape) != len(shape) or any(
        w is not None and int(w) != g for w, g in zip(shape, m.shape)
    ):
        _fail("TENSOR_SHAPE", f"{role}: shape {m.shape}, want {tuple(shape)}")
    if any(s < 0 for s in m.strides):
        _fail("TENSOR_STRIDES", f"{role}: negative stride in {m.strides}")
    if strides is not None:
        for n, got, want in zip(m.shape, m.strides, strides):
            if n > 1 and got != int(want):
                _fail(
                    "TENSOR_STRIDES",
                    f"{role}: strides {m.strides}, want {tuple(strides)} (dims of "
                    "extent > 1)",
                )
    need = extent_elems(m.shape, m.strides) * ELEM_BYTES[m.dtype]
    if need > m.nbytes:
        _fail(
            "TENSOR_EXTENT",
            f"{role}: the kernels address {need} bytes from the base, the tensor "
            f"holds {m.nbytes}",
        )


def int_list(name: str, values, code: str) -> list[int]:
    """``values`` as Python ints (``code`` for a non-integer entry)."""
    out = []
    for v in values:
        if isinstance(v, bool) or int(v) != v:
            _fail(code, f"{name}: non-integer entry {v!r}")
        out.append(int(v))
    return out


def check_attn_bwd_tensors(req, tensors: Mapping[str, TensorMeta]) -> None:
    """Check the tensors of a general-family request (``ValueError``).

    ``tensors`` maps the plan's tensor roles (``q k v o do dq dk dv lse``, plus
    ``seq_len_q`` / ``seq_len_kv`` with padding and ``offsets_q`` /
    ``offsets_kv`` for THD) to their :class:`TensorMeta`. Dense layout: every
    tensor is ``[B, H, S, D]`` with the request's ``(B, H, S, D)`` strides.
    THD: ``[T, H, D]`` with the request's ``(token, head, D)`` strides (``T``
    any; :func:`check_attn_bwd_lengths` checks the offsets against it). The
    extent each kernel addresses must fit the tensor's bytes.
    """
    from kernels.common.attention_bwd_plan import _lse_kernarg_strides

    dts = {
        "q": req.dtype, "k": req.k_dtype, "v": req.v_dtype, "o": req.o_dtype,
        "do": req.do_dtype, "dq": req.dq_dtype, "dk": req.dk_dtype,
        "dv": req.dv_dtype,
    }  # fmt: skip
    for n, dt in dts.items():
        sb, sh, ss, sd = req.tensor_strides(n)
        h = req.tensor_heads(n)
        m = _meta(tensors, n)
        dt = req.dtype if dt is None else dt
        if req.is_thd:
            check_tensor(
                n, m, dtype=dt, shape=(None, h, req.d_qk), strides=(ss, sh, sd)
            )
        else:
            shape = (req.b, h, req.tensor_seq(n), req.d_qk)
            check_tensor(n, m, dtype=dt, shape=shape, strides=(sb, sh, ss, sd))
    lb, lh, lt = _lse_kernarg_strides(req)
    m = _meta(tensors, "lse")
    if req.is_thd:
        # [T, H_q(, 1)] or the batched [B, H_q, S, 1] form whose S dim holds
        # the tokens (its B stride is unused); the prep reads token t, head h
        # at t * lt + h * lh
        if m.dtype != "fp32":
            _fail("TENSOR_DTYPE", f"lse: dtype {m.dtype}, want fp32")
        rank = len(m.shape)
        t_dim, h_dim = (2, 1) if rank == 4 else (0, 1)
        if (
            rank not in (2, 3, 4)
            or m.shape[h_dim] != req.h_q
            or (rank > 2 and m.shape[-1] != 1)
        ):
            _fail("TENSOR_SHAPE", f"lse: shape {m.shape}, want [T, {req.h_q}(, 1)] "
                  f"or [B, {req.h_q}, T, 1]")  # fmt: skip
        tokens = m.shape[t_dim]
        if (tokens > 1 and m.strides[t_dim] != lt) or (
            req.h_q > 1 and m.strides[h_dim] != lh
        ):
            _fail("TENSOR_STRIDES", f"lse: strides {m.strides}, want token {lt}, "
                  f"head {lh}")  # fmt: skip
        need = 0 if tokens == 0 else ((tokens - 1) * lt + (req.h_q - 1) * lh + 1) * 4
        if need > m.nbytes:
            _fail("TENSOR_EXTENT", f"lse: needs {need} bytes, holds {m.nbytes}")
    else:
        rank = len(m.shape)
        shape = (req.b, req.h_q, req.s_q) + ((1,) if rank == 4 else ())
        check_tensor("lse", m, dtype="fp32", shape=shape,
                     strides=(lb, lh, lt) + ((0,) if rank == 4 else ()))  # fmt: skip
    if req.padding:
        for n in ("seq_len_q", "seq_len_kv"):
            m = _meta(tensors, n)
            if m.dtype != req.seq_len_dtype:
                _fail("TENSOR_DTYPE", f"{n}: dtype {m.dtype}, want {req.seq_len_dtype}")
            need = ((req.b - 1) * req.seq_len_stride + 1) * ELEM_BYTES[m.dtype]
            if need > m.nbytes:
                _fail("TENSOR_EXTENT", f"{n}: needs {need} bytes, holds {m.nbytes}")
    if req.is_thd:
        for n in ("offsets_q", "offsets_kv"):
            m = _meta(tensors, n)
            if m.dtype != req.ragged_offset_dtype:
                _fail(
                    "TENSOR_DTYPE",
                    f"{n}: dtype {m.dtype}, want {req.ragged_offset_dtype}",
                )
            need = (req.b + 1) * ELEM_BYTES[m.dtype]
            if need > m.nbytes:
                _fail("TENSOR_EXTENT", f"{n}: needs {need} bytes, holds {m.nbytes}")


def check_attn_bwd_lengths(
    req,
    *,
    seq_len_q=None,
    seq_len_kv=None,
    offsets_q=None,
    offsets_kv=None,
    tokens_q: int | None = None,
    tokens_kv: int | None = None,
) -> None:
    """Check host-visible length data of a general-family request.

    Padding: ``seq_len_q`` / ``seq_len_kv`` hold the ``B`` per-batch lengths
    (in batch order, whatever the device stride): each in ``[0, s_q]`` /
    ``[0, s_kv]``, else ``SEQ_LEN_RANGE`` (rejected; the device would clamp).
    THD: ``offsets_q`` / ``offsets_kv`` hold the ``B + 1`` offsets: non-negative,
    non-decreasing, whole tokens after ``* ragged_offset_multiplier / token
    stride``, each span at most ``s_q`` / ``s_kv``; the end token of every
    sequence within ``max_total_q`` / ``max_total_kv`` (when both are given)
    and within ``tokens_q`` / ``tokens_kv`` (the tensors' token extents, when
    given), else ``RAGGED_RANGE``. Arguments left ``None`` are not checked.
    """
    for name, vals, cap in (("seq_len_q", seq_len_q, req.s_q),
                            ("seq_len_kv", seq_len_kv, req.s_kv)):  # fmt: skip
        if vals is None:
            continue
        vals = int_list(name, vals, "SEQ_LEN_RANGE")
        if len(vals) != req.b:
            _fail("SEQ_LEN_RANGE", f"{name}: {len(vals)} values for B = {req.b}")
        for i, v in enumerate(vals):
            if not 0 <= v <= cap:
                _fail("SEQ_LEN_RANGE", f"{name}[{i}] = {v} outside [0, {cap}]")
    by_token = (
        req.is_thd and req.max_total_q is not None and req.max_total_kv is not None
    )
    sides = (
        ("offsets_q", offsets_q, "q", req.s_q, req.max_total_q, tokens_q),
        ("offsets_kv", offsets_kv, "k", req.s_kv, req.max_total_kv, tokens_kv),
    )
    for name, vals, tensor, cap, total, toks in sides:
        if vals is None:
            continue
        vals = int_list(name, vals, "RAGGED_RANGE")
        if len(vals) != req.b + 1:
            _fail(
                "RAGGED_RANGE", f"{name}: {len(vals)} offsets for B + 1 = {req.b + 1}"
            )
        mult, div = req.ragged_offset_multiplier, req.tensor_strides(tensor)[2]
        for i, v in enumerate(vals):
            if v < 0:
                _fail("RAGGED_RANGE", f"{name}[{i}] = {v} is negative")
            if i and v < vals[i - 1]:
                _fail("RAGGED_RANGE", f"{name} decreases at {i}: {vals[i - 1]} -> {v}")
            if (v * mult) % div:
                _fail("RAGGED_RANGE", f"{name}[{i}] = {v} is not a whole token "
                      f"(* {mult} / {div})")  # fmt: skip
        tok = [v * mult // div for v in vals]
        for i in range(req.b):
            span = tok[i + 1] - tok[i]
            if span > cap:
                _fail(
                    "RAGGED_RANGE", f"{name}: sequence {i} spans {span} > {cap} tokens"
                )
            if by_token and tok[i + 1] > total:
                _fail("RAGGED_RANGE", f"{name}: sequence {i} ends at token "
                      f"{tok[i + 1]} > the bound {total}")  # fmt: skip
            if toks is not None and tok[i + 1] > toks:
                _fail("RAGGED_RANGE", f"{name}: sequence {i} ends at token "
                      f"{tok[i + 1]} past the tensor's {toks} tokens")  # fmt: skip


def check_attn_dense_bwd_tensors(
    spec, tensors: Mapping[str, TensorMeta], *, batch: int
) -> None:
    """Check the tensors of a dense-family kernel (compile-time shape).

    Every tensor of ``q k v o do dq dk dv`` is ``[batch, H, S, D]`` with the
    spec's ``(batch, head, token, 1)`` strides (contiguous BSHD or BHSD), and
    ``lse`` is ``[batch, H_q, S_q]`` packed fp32; ``batch`` is the runtime
    batch (it must equal a batch the spec pins).
    """
    if spec.batch is not None and batch != spec.batch:
        _fail("TENSOR_SHAPE", f"batch {batch}, the spec pins {spec.batch}")
    for n in ("q", "k", "v", "o", "do", "dq", "dk", "dv"):
        q_side = n in ("q", "o", "do", "dq")
        h = spec.num_query_heads if q_side else spec.num_kv_heads
        s = spec.seqlen_q if q_side else spec.seqlen_kv
        sb, sh, st = spec.strides(n)
        check_tensor(n, _meta(tensors, n), dtype=spec.dtype,
                     shape=(batch, h, s, spec.head_size), strides=(sb, sh, st, 1))  # fmt: skip
    hq, sq = spec.num_query_heads, spec.seqlen_q
    check_tensor("lse", _meta(tensors, "lse"), dtype="fp32", shape=(batch, hq, sq),
                 strides=(hq * sq, sq, 1))  # fmt: skip


def _build_dense(spec, *, arch):
    """The dense family's main kernel (per-arch builder, imported lazily)."""
    import importlib

    mod = importlib.import_module(f"kernels.{arch}.attention_dense_bwd")
    return mod.build_attention_dense_bwd(spec, arch=arch)


def _build_dense_dq(spec, *, arch):
    """The dense family's split dQ kernel (imported lazily)."""
    from kernels.common.attention_dense_bwd_dq import build_attention_dense_bwd_dq

    return build_attention_dense_bwd_dq(spec, arch=arch)


def _stage_params(stage: str):
    """Ordered kernargs of a launch stage (the dense kernels have their own ABIs)."""
    if stage == "dense_dq":
        from kernels.common.attention_dense_bwd_dq import attn_dense_bwd_dq_params

        return attn_dense_bwd_dq_params()
    if stage == "dense":
        from kernels.common.attention_dense_bwd_spec import attn_dense_bwd_params

        return attn_dense_bwd_params()
    return attn_bwd_params(stage)


_BUILDERS = {
    "main": build_attn_bwd_main,
    "prep": build_attn_bwd_prep,
    "convert": build_attn_bwd_convert,
    "reduce": build_attn_bwd_reduce,
    "dq": build_attn_bwd_dq,
    "dense": _build_dense,
    "dense_dq": _build_dense_dq,
}
# The shipped builders. A caller that swaps a builder in ``_BUILDERS`` builds
# another kernel from the same spec: its code objects go through the disk
# cache only under the ``cache_variant`` string the swapped builder declares
# (which must name everything that makes it differ), else never.
_STOCK_BUILDERS = dict(_BUILDERS)

_PROCESS_CACHE: dict = {}


def _flavor() -> str:
    flavor = os.environ.get("ROCKE_LLVM_FLAVOR")
    if flavor:
        return flavor
    from rocke.core.lower_llvm import _detect_llvm_flavor

    return _detect_llvm_flavor()


def _comgr_version() -> str:
    try:
        from rocke.runtime.comgr import resolved_lib_rocm_version

        v = resolved_lib_rocm_version()
    except Exception:  # noqa: BLE001 - recorded as unknown in the cache key
        v = None
    return "unknown" if v is None else f"{v[0]}.{v[1]}"


def _code_object(step, plan, disk):
    """The artifact of one launch: from ``disk`` (a ``CompileCache`` or
    ``None``) when it holds it, else built and compiled (and written back)."""
    spec = plan.specs[step.stage]
    builder = _BUILDERS[step.stage]
    variant = ""
    if builder is not _STOCK_BUILDERS[step.stage]:
        # a swapped builder is cached only under the variant it declares
        variant = getattr(builder, "cache_variant", "") or ""
        if not variant:
            disk = None
    if disk is not None:
        entry = disk.read(step.stage, spec, plan.arch, variant=variant)
        if entry is not None:
            if entry.kernel_name != step.kernel:
                raise AssertionError(
                    f"cached kernel {entry.kernel_name} differs from the plan's "
                    f"{step.kernel}"
                )
            from kernels.common.attention_bwd_cache import CachedArtifact

            return CachedArtifact(entry)
    from rocke.helpers.compile import compile_kernel

    art = compile_kernel(
        builder(spec, arch=plan.arch),
        arch=plan.arch,
        capture_ir_text=False,
        backend="python",
    )
    if art.kernel_name != step.kernel:
        raise AssertionError(
            f"built kernel {art.kernel_name} differs from the plan's {step.kernel}"
        )
    if disk is not None:
        try:
            disk.write(
                step.stage,
                spec,
                plan.arch,
                kernel_name=art.kernel_name,
                hsaco=art.hsaco,
                variant=variant,
            )
        except OSError:  # a read-only or full cache directory costs a rebuild
            pass
    return art


def attn_bwd_kernels(plan, *, cache: MutableMapping | None = None) -> dict:
    """``{kernel name: (artifact, module, function)}`` for every plan launch.

    Each kernel is built from the plan's spec for its stage, compiled through
    comgr on the Python lowering backend and loaded once; ``cache`` (default:
    a process-wide dict) is keyed by ``(name, arch, flavor, comgr)``.

    With ``ROCKE_BWD_HSACO_CACHE=<dir>`` a kernel missing from ``cache`` is
    first looked up in that on-disk cache (keyed by stage, resolved spec,
    arch, toolchain and backward source hash; see
    :mod:`kernels.common.attention_bwd_cache`) and a fresh compile is written
    there; a stage whose builder was swapped in ``_BUILDERS`` uses it only
    under the builder's ``cache_variant`` (none: bypassed), and while an
    emitter is patched (:func:`~kernels.common.attention_bwd_cache.patched_emission`:
    a negative control building a mutant from a shipped spec) the disk is
    neither read nor written. Unset (the default), nothing is read from or
    written to disk.
    """
    from rocke.runtime.hip_module import Runtime

    cache = _PROCESS_CACHE if cache is None else cache
    flavor, comgr = _flavor(), _comgr_version()
    disk: Any = False  # resolved on the first in-memory miss
    bypassed = None  # the disk cache a patched emitter keeps out
    out = {}
    for step in plan.launches:
        key = (step.kernel, plan.arch, flavor, comgr)
        if key not in cache:
            if disk is False:
                from kernels.common.attention_bwd_cache import (
                    env_cache,
                    patched_emission,
                )

                disk = env_cache()
                if disk is not None and patched_emission():
                    bypassed, disk = disk, None
            art = _code_object(step, plan, disk)
            if bypassed is not None:
                bypassed.bypassed += 1
            mod = Runtime().load_module(art.hsaco)
            cache[key] = (art, mod, mod.get_function(art.kernel_name))
        out[step.kernel] = cache[key]
    return out


def pack_attn_bwd_args(
    step,
    *,
    tensors: Mapping[str, int],
    workspace: int,
    workspace_layout: Mapping[str, tuple[int, int]],
) -> bytes:
    """Kernel-argument bytes of one :class:`LaunchStep` (``rocke.attn_bwd.v3``).

    Pointer roles resolve to ``workspace + offset`` for workspace sub-buffers,
    to ``tensors[role]`` for tensor roles, and to 0 for unused (``None``)
    pointers. A tensor role missing from ``tensors`` raises ``ValueError``.
    """
    out = []
    for name, fmt in _stage_params(step.stage):
        if fmt == "Q":
            role = step.pointers[name]
            if role is None:
                ptr = 0
            elif role in workspace_layout:
                ptr = int(workspace) + int(workspace_layout[role][0])
            elif role in tensors and tensors[role] is not None:
                ptr = int(tensors[role])
            else:
                raise ValueError(f"{step.stage}: no device pointer for role {role!r}")
            out.append(struct.pack("<Q", ptr))
        elif fmt == "f":
            out.append(struct.pack("<f", float(step.scalars[name])))
        else:
            # "q" (i64) or "i" (i32); a value outside the kernarg type is a
            # predicate bug, reported by name rather than as a struct error.
            v = int(step.scalars[name])
            bits = 64 if fmt == "q" else 32
            if not -(1 << (bits - 1)) <= v < (1 << (bits - 1)):
                raise ValueError(
                    f"{step.stage}: kernarg {name}={v} does not fit i{bits}"
                )
            out.append(struct.pack("<" + fmt, v))
    return b"".join(out)


def run_attn_bwd(
    plan,
    *,
    tensors: Mapping[str, int],
    workspace: int,
    workspace_size: int,
    stream: int = 0,
    cache: MutableMapping | None = None,
) -> list[Any]:
    """Launch ``prep -> main -> convert(s)`` of ``plan`` on ``stream``.

    ``tensors`` maps the plan's pointer roles (``q k v o do lse dq dk dv``,
    plus ``seq_len_q``/``seq_len_kv``/``offsets_q``/``offsets_kv`` when the
    plan uses them) to device addresses. ``workspace`` is the caller's
    workspace base (``WORKSPACE_ALIGN``-byte aligned) of ``workspace_size``
    bytes, at least ``plan.workspace_bytes``. Nothing is allocated and no
    memset is issued. Returns the launched kernel names in order.

    The pointers are trusted: a caller holding the tensors validates them
    against the planned request first (:func:`check_attn_bwd_tensors`,
    :func:`check_attn_bwd_lengths`).
    """
    from rocke.runtime.hip_module import Runtime

    if workspace_size < plan.workspace_bytes:
        raise ValueError(
            f"workspace of {workspace_size} bytes is smaller than the plan's "
            f"{plan.workspace_bytes}"
        )
    if plan.workspace_bytes and int(workspace) % WORKSPACE_ALIGN:
        raise ValueError(f"workspace base is not {WORKSPACE_ALIGN}-byte aligned")
    kernels = attn_bwd_kernels(plan, cache=cache)
    packed = [
        pack_attn_bwd_args(
            step,
            tensors=tensors,
            workspace=workspace,
            workspace_layout=plan.workspace_layout,
        )
        for step in plan.launches
    ]
    rt = Runtime()
    names = []
    for step, args in zip(plan.launches, packed):
        _art, _mod, fn = kernels[step.kernel]
        rt.launch(fn, tuple(step.grid), (int(step.block), 1, 1), args, stream=stream)
        names.append(step.kernel)
    return names

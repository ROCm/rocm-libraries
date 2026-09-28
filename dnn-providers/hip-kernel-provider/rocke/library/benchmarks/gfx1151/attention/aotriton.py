# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ctypes wrapper around aotriton_bridge.{so,dll}: AOTriton 0.14.2b's v3
flash-attention forward API (``aotriton::v3::flash::attn_fwd``), prepared once
and launched with zero per-launch allocation/rebuild so it can be replayed
inside a captured HIP graph.

This module owns no benchmarking, no case selection, and no fallback: every
configuration AOTriton 0.14.2b genuinely cannot do raises ``AotritonError``
(or is documented below as never reaching this module at all, per the
parent runner's own pre-filter in ``benchmark_sdpa.py``). Nothing here
silently approximates an unsupported feature.

Verified upstream facts (AOTriton tag ``0.14.2b``; every claim below was read
directly from these sources, not inferred from documentation prose):

* Native forward entry point is ``AOTRITON_NS::v3::flash::attn_fwd`` taking an
  ``attn_fwd_params`` (``kVersion == 4``) plus an ``attn_options`` with
  ``force_backend_index``.
  https://github.com/ROCm/aotriton/blob/0.14.2b/include/aotriton/flash.h

* ``TensorView<Rank>`` is pointer + fixed-size shape/stride arrays + ``DType``,
  value semantics, no owned memory; ``get_null_tensor()`` yields a
  ``base_ptr == nullptr`` view the kernel treats as absent (``operator bool()``
  is ``false``). ``DType`` values: kFloat32=1, kFloat16=2, kBFloat16=3, kInt8=10,
  kInt16=11, kInt32=12, kInt64=13, kUInt8=20, kUInt16=21, kUInt32=22, kUInt64=23.
  https://github.com/ROCm/aotriton/blob/0.14.2b/include/aotriton/util.h
  https://github.com/ROCm/aotriton/blob/0.14.2b/include/aotriton/dtypes.h

* Causal/window encoding. ``CausalType`` only ever compiles ``None`` (0) and
  ``WindowedAttention`` (3): ``modules/flash/aot/attn_fwd.py`` declares
  ``@ati.scalar('CAUSAL_TYPE', options=[0, 3])``, and flash.h's own comment
  above ``CausalType`` says ``TopLeftAligned``/``BottomRightAligned`` (1/2) are
  "supported in Triton kernel, but not compiled into the binary GPU kernels".
  Top-left/bottom-right causal masks are instead expressed as
  ``WindowedAttention`` with both ``window_left``/``window_right`` set to the
  matching ``WindowValue`` sentinel (``TopLeftAligned``/``BottomRightAligned``
  = ``-2147483647``/``-2147483646``) -- this is exactly PyTorch's own encoding:
  https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/cuda/attention.cu
  (search ``CustomMaskType::CausalFromTopLeft``/``CausalFromBottomRight``).
  Finite windows use the GPU kernel's actual inequalities, not the caller's
  window naming: ``m - Window_left <= n <= m + Window_right``. A causal width
  W therefore needs ``Window_left = W - 1 - context``, where context is zero
  for top-left and Sk-Sq for bottom-right. Packed bottom-right finite windows
  require per-sequence left diagonals and are rejected by this bridge.
  https://github.com/ROCm/aotriton/blob/0.14.2b/modules/flash/kernel/fwd_kernel_inner.py

* Additive bias (``params.B``) is compiled, but shares the ``T_io`` dtype
  variable with Q/K/V/Out. rocKE's FP32 QQ-bias with FP16/BF16 operands is not
  equivalent to that interface. The harness reports the mismatch rather
  than silently rounding the bias or treating the failed launch as a win.
  https://github.com/ROCm/aotriton/blob/0.14.2b/modules/flash/aot/attn_fwd.py

* ALiBi (``params.A``, rank-2) is present in the ABI but is **not functional**
  in shipped 0.14.2b binaries: ``attn_fwd.py`` declares
  ``@ati.scalar('USE_ALIBI', options=[False])`` -- ``False`` is the *only*
  functional ever compiled -- and ``attn_fwd.cc``'s host dispatcher hardcodes
  ``.USE_ALIBI = false`` unconditionally. Setting ``params.A`` would be a
  silent no-op, not a working feature, so this bridge never exposes it, and
  the parent runner rejects ALiBi cases before calling ``prepare()``.

* Attention sinks have **no field at all** in ``attn_fwd_params`` and no
  corresponding scalar/tensor anywhere in ``modules/flash/aot/attn_fwd.py``'s
  compiled functional description -- there is no "sink" identifier anywhere in
  the 0.14.2b source tree. ``prepare(sinks=...)`` therefore raises
  ``AotritonError`` immediately, before any ctypes call, whenever ``sinks`` is
  not ``None``.

* FP8/INT8 KV has no ``k_scale``/``v_scale`` field in ``attn_fwd_params``, and
  the internal ``INT8``/``INT8_KV``/``USE_P_SCALE`` flags are likewise
  compile-time fixed to ``False`` (``options=[False]`` only). Not reachable
  through this ABI at all; the parent runner rejects these before calling
  ``prepare()``.

* Softcap has no field anywhere in ``attn_fwd_params`` or in
  ``attn_fwd.py``'s functional description. Not reachable through this ABI;
  the parent runner rejects these before calling ``prepare()``.

* Paged KV (block tables) has no representation in ``attn_fwd_params``.
  PyTorch's own varlen entry point rejects it outright
  (``TORCH_CHECK(!paged_KV, ...)`` in ``mha_all_aot.hip``). Not reachable
  through this ABI; the parent runner rejects paged cases before calling
  ``prepare()``.

* Ragged/varlen addressing is ``VarlenBits``: independent per-side
  (stacked/length/position) mode, replacing the older ``VarlenType`` enum.
  Dense uses the in-class default (``BHSD``/``MAX``/``IMPLIED`` both sides).
  Packed variable length uses ``THD``/``CUMULATIVE``/``REUSE`` on both sides
  driven by ``cu_seqlens_q``/``cu_seqlens_k`` (``(batch+1,)`` int32,
  monotonic prefix sums). The ``seqused_k`` pairing (K takes its length from a
  ``(batch,)`` int32 array and its start offset from ``cu_seqlens_k``) is, in
  flash.h's own words, "the only [varlen_bits combination] exercised through
  this API; the rest of the combination space is expected to work but
  untested." Packed tensors use AOTriton's required
  ``(1, num_heads, total_tokens, head_dim)`` shape -- confirmed by
  ``mha_all_aot.hip``'s own comment on that exact shape requirement, and by
  the parent runner's ``DeviceBuffers.tensor(..., attention=True)`` producing
  precisely that shape for 3-D packed arrays.
  https://github.com/ROCm/aotriton/blob/0.14.2b/include/aotriton/flash.h

* Null ``L`` (logsumexp) is a verified, upstream-used pattern for inference:
  PyTorch's own mem_eff path passes a ``TensorView<2>`` with
  ``base_ptr == nullptr`` when ``compute_logsumexp`` is false. This bridge
  always does the same (backward/training is out of this harness's scope).

* Backend selection: ``op_attn_fwd`` has three backends -- ``triton``
  (default/backend-index 0, the always-present kernel every functional is
  described against), ``aiter`` (``@ati.affine.arch(['gfx942', 'gfx950'])``,
  ``modules/flash/aot/aiter_fwd.py``), and ``flyc``
  (``_FLYC_FWD_LADDERS = {'gfx1201': ..., 'gfx950': ...}``,
  ``modules/flash/aot/flyc_attn_fwd.py``). Neither ``aiter`` nor ``flyc``
  targets gfx1151, so on this GPU the auto-selecting ``backend=-1`` and an
  explicit force to the Triton backend resolve to the same compiled kernel --
  there is no stronger alternative to choose. This bridge forwards whatever
  integer ``backend`` the caller supplies straight into
  ``attn_options.force_backend_index`` with no name-to-index table: that
  table is a ``pyaotriton`` (pybind11) construct from a package this bridge
  does not link, since it calls the plain C++ shared library directly.
  https://github.com/ROCm/aotriton/blob/0.14.2b/modules/flash/aot/aiter_fwd.py
  https://github.com/ROCm/aotriton/blob/0.14.2b/modules/flash/aot/flyc_attn_fwd.py

* Library name: the installed shared object is ``libaotriton_v2`` even for the
  v3 API (confirmed by the project's own compatibility notes, e.g.
  "symlinking ``libaotriton_v2.so.0.9.2`` to ``libaotriton_v2.so.0.10.0``").
  https://github.com/ROCm/aotriton/blob/0.14.2b/README.md

Host compile/link command (parent compiles once outside timing; no device
code in this translation unit, so no ``--offload-arch`` is needed -- hipcc is
used only for its bundled HIP host headers/libs)::

    hipcc -std=c++17 -O2 -fPIC -shared \\
        -I "$AOTRITON_INSTALL/include" \\
        aotriton_bridge.cpp -o libaotriton_bridge.so \\
        -L "$AOTRITON_INSTALL/lib" -laotriton_v2 -Wl,-rpath,"$AOTRITON_INSTALL/lib"

[INFERENCE]: ``aotriton_get_git_sha1()``/``aotriton_get_name_suffix()`` rely on
``AOTRITON_GIT_SHA1``/``AOTRITON_NAME_SUFFIX`` from the installed, CMake
-generated ``config.h`` (``include/aotriton/config.h.in`` is the template;
the finished header is produced by the parent's install and was not directly
inspected here). ``aotriton_get_version()`` (major/minor/patch as plain
integers) is the reliable one and should be treated as authoritative;
``git_sha1``/``name_suffix`` are best-effort strings for logging only.
"""

from __future__ import annotations

import ctypes
import dataclasses
from pathlib import Path
from typing import Optional, Tuple, Union

__all__ = ["TensorView", "Aotriton", "PreparedCall", "AotritonError"]


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TensorView:
    """A non-owning view of a parent-allocated device buffer.

    ``shape``/``strides`` are in element units. For rank-4 views this is
    AOTriton's logical BHSD regardless of the underlying storage order (the
    parent runner passes BSHD storage transposed to a BHSD *view*, i.e. only
    shape/strides change, not memory).
    """

    ptr: int
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    dtype: str


class AotritonError(Exception):
    """Raised for both preparation-time rejections and launch failures.

    ``code`` is the native ``hipError_t`` value for a launch failure, ``None``
    for a rejection this Python layer made before ever calling into the
    bridge (unsupported feature, malformed input).
    """

    def __init__(self, code: Optional[int], message: str):
        self.code = code
        self.message = message
        super().__init__(f"AOTriton error (hip code={code}): {message}")


# ---------------------------------------------------------------------------
# dtype / mask code tables (single source of truth: include/aotriton/dtypes.h
# and the mask encoding documented in aotriton_bridge.cpp::translate_mask)
# ---------------------------------------------------------------------------

_DTYPE_CODES = {
    "fp32": 1,
    "fp16": 2,
    "bf16": 3,
    "i8": 10,
    "i16": 11,
    "i32": 12,
    "i64": 13,
    "u8": 20,
    "u16": 21,
    "u32": 22,
    "u64": 23,
}

_MASK_CODES = {"none": 0, "causal_topleft": 1, "causal_bottomright": 2}


def _dtype_code(name: str) -> int:
    try:
        return _DTYPE_CODES[name]
    except KeyError as exc:
        raise AotritonError(None, f"unknown TensorView dtype {name!r}; expected one of {sorted(_DTYPE_CODES)}") from exc


# ---------------------------------------------------------------------------
# ctypes ABI mirror of aotriton_bridge.cpp's AotritonTensorDesc/AotritonFwdDesc.
# Field order, widths and natural alignment (8-byte, driven by the leading
# uint64) must match the C++ structs byte-for-byte; _validate_abi() below
# checks this against the library's own reported sizes at load time so a
# layout drift fails loudly instead of corrupting memory.
# ---------------------------------------------------------------------------


class _CTensorDesc(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.c_uint64),
        ("shape", ctypes.c_int64 * 4),
        ("strides", ctypes.c_int64 * 4),
        ("rank", ctypes.c_int32),
        ("dtype", ctypes.c_int32),
    ]


class _CFwdDesc(ctypes.Structure):
    _fields_ = [
        ("q", _CTensorDesc),
        ("k", _CTensorDesc),
        ("v", _CTensorDesc),
        ("out", _CTensorDesc),
        ("bias", _CTensorDesc),
        ("cu_seqlens_q", _CTensorDesc),
        ("cu_seqlens_k", _CTensorDesc),
        ("seqused_k", _CTensorDesc),
        ("scale", ctypes.c_float),
        ("mask_kind", ctypes.c_int32),
        ("window", ctypes.c_int32),
        ("max_seqlen_q", ctypes.c_int32),
        ("max_seqlen_k", ctypes.c_int32),
        ("batch", ctypes.c_int32),
    ]


def _pack_tensor(view: Optional[TensorView], *, expect_rank: Optional[int] = None) -> _CTensorDesc:
    desc = _CTensorDesc()
    if view is None:
        return desc  # all-zero: ptr==0 means "absent" to the C++ side
    rank = len(view.shape)
    if rank not in (1, 2, 4):
        raise AotritonError(None, f"TensorView rank {rank} is not one attn_fwd_params uses (1, 2 or 4)")
    if expect_rank is not None and rank != expect_rank:
        raise AotritonError(None, f"expected rank {expect_rank} TensorView, got rank {rank}")
    if len(view.strides) != rank:
        raise AotritonError(None, "TensorView shape/strides length mismatch")
    desc.ptr = view.ptr
    desc.rank = rank
    desc.dtype = _dtype_code(view.dtype)
    for i in range(rank):
        desc.shape[i] = view.shape[i]
        desc.strides[i] = view.strides[i]
    return desc


# ---------------------------------------------------------------------------
# Bridge wrapper
# ---------------------------------------------------------------------------


class Aotriton:
    """Loads ``aotriton_bridge`` and records native version/build facts.

    ``library_path`` names the compiled bridge shared object (see the compile
    command in this module's docstring), not AOTriton's own
    ``libaotriton_v2``; the bridge links that in at build time.
    """

    def __init__(self, library_path: Union[str, "Path"]):
        self._lib = ctypes.CDLL(str(library_path))
        self._bind()
        self._validate_abi()

        major, minor, patch = ctypes.c_int32(), ctypes.c_int32(), ctypes.c_int32()
        self._lib.aotriton_get_version(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch))
        self.version: Tuple[int, int, int] = (major.value, minor.value, patch.value)

        git_sha1 = self._lib.aotriton_get_git_sha1()
        self.git_sha1: str = git_sha1.decode("utf-8", "replace") if git_sha1 else ""
        name_suffix = self._lib.aotriton_get_name_suffix()
        self.name_suffix: str = name_suffix.decode("utf-8", "replace") if name_suffix else ""
        self.params_version: int = self._lib.aotriton_fwd_params_version()

    def _bind(self) -> None:
        lib = self._lib

        lib.aotriton_get_version.argtypes = [ctypes.POINTER(ctypes.c_int32)] * 3
        lib.aotriton_get_version.restype = None
        lib.aotriton_get_git_sha1.argtypes = []
        lib.aotriton_get_git_sha1.restype = ctypes.c_char_p
        lib.aotriton_get_name_suffix.argtypes = []
        lib.aotriton_get_name_suffix.restype = ctypes.c_char_p
        lib.aotriton_fwd_params_version.argtypes = []
        lib.aotriton_fwd_params_version.restype = ctypes.c_int32
        lib.aotriton_abi_sizeof_tensor_desc.argtypes = []
        lib.aotriton_abi_sizeof_tensor_desc.restype = ctypes.c_int32
        lib.aotriton_abi_sizeof_fwd_desc.argtypes = []
        lib.aotriton_abi_sizeof_fwd_desc.restype = ctypes.c_int32
        lib.aotriton_last_error.argtypes = []
        lib.aotriton_last_error.restype = ctypes.c_char_p

        lib.aotriton_prepare.argtypes = [ctypes.POINTER(_CFwdDesc), ctypes.POINTER(ctypes.c_void_p)]
        lib.aotriton_prepare.restype = ctypes.c_int32
        lib.aotriton_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
        lib.aotriton_launch.restype = ctypes.c_int32
        lib.aotriton_destroy.argtypes = [ctypes.c_void_p]
        lib.aotriton_destroy.restype = None

    def _validate_abi(self) -> None:
        native_tensor_desc = self._lib.aotriton_abi_sizeof_tensor_desc()
        native_fwd_desc = self._lib.aotriton_abi_sizeof_fwd_desc()
        if native_tensor_desc != ctypes.sizeof(_CTensorDesc) or native_fwd_desc != ctypes.sizeof(_CFwdDesc):
            raise AotritonError(
                None,
                "aotriton_bridge ABI mismatch: native sizeof(AotritonTensorDesc)="
                f"{native_tensor_desc} sizeof(AotritonFwdDesc)={native_fwd_desc} vs this module's "
                f"ctypes sizeof(_CTensorDesc)={ctypes.sizeof(_CTensorDesc)} "
                f"sizeof(_CFwdDesc)={ctypes.sizeof(_CFwdDesc)}; aotriton.py and aotriton_bridge.cpp "
                "have drifted out of lockstep and must be rebuilt/updated together.",
            )

    def _last_error(self) -> str:
        msg = self._lib.aotriton_last_error()
        return msg.decode("utf-8", "replace") if msg else ""

    def prepare(
        self,
        *,
        q: TensorView,
        k: TensorView,
        v: TensorView,
        out: TensorView,
        scale: float,
        mask: str = "none",
        window: int = 0,
        bias: Optional[TensorView] = None,
        sinks: Optional[TensorView] = None,
        cu_seqlens_q: Optional[TensorView] = None,
        cu_seqlens_k: Optional[TensorView] = None,
        seqused_k: Optional[TensorView] = None,
        max_seqlen_q: int = 0,
        max_seqlen_k: int = 0,
        batch: int = 0,
    ) -> "PreparedCall":
        # Attention sinks: no field in attn_fwd_params, no scalar/tensor in
        # attn_fwd.py's compiled functional description anywhere in 0.14.2b.
        # Reject up front rather than silently dropping the tensor.
        if sinks is not None:
            raise AotritonError(
                None,
                "AOTriton 0.14.2b attn_fwd has no attention-sinks input: attn_fwd_params "
                "(include/aotriton/flash.h) carries no sinks tensor, and "
                "modules/flash/aot/attn_fwd.py declares no corresponding scalar/tensor for one. "
                "Rejecting rather than silently ignoring the tensor.",
            )
        if mask not in _MASK_CODES:
            raise AotritonError(None, f"unknown mask {mask!r}; expected one of {sorted(_MASK_CODES)}")
        if window < 0:
            raise AotritonError(None, "window must be >= 0 (0 disables windowing)")
        if window and mask == "none":
            raise AotritonError(None, "finite window requires a causal mask")
        if window and mask == "causal_bottomright" and cu_seqlens_q is not None:
            raise AotritonError(None, "packed bottom-right finite windows need per-sequence left diagonals")
        if bias is not None and bias.dtype != q.dtype:
            raise AotritonError(None, "AOTriton 0.14.2b requires bias and Q to share their dtype")
        if (cu_seqlens_q is None) != (cu_seqlens_k is None):
            raise AotritonError(None, "cu_seqlens_q and cu_seqlens_k must both be given or both omitted")
        if seqused_k is not None and cu_seqlens_q is None:
            raise AotritonError(
                None, "seqused_k requires cu_seqlens_q/cu_seqlens_k (it pairs with cu_seqlens_k for KV-cache addressing)"
            )

        desc = _CFwdDesc()
        desc.q = _pack_tensor(q, expect_rank=4)
        desc.k = _pack_tensor(k, expect_rank=4)
        desc.v = _pack_tensor(v, expect_rank=4)
        desc.out = _pack_tensor(out, expect_rank=4)
        desc.bias = _pack_tensor(bias, expect_rank=4)
        desc.cu_seqlens_q = _pack_tensor(cu_seqlens_q, expect_rank=1)
        desc.cu_seqlens_k = _pack_tensor(cu_seqlens_k, expect_rank=1)
        desc.seqused_k = _pack_tensor(seqused_k, expect_rank=1)
        desc.scale = scale
        desc.mask_kind = _MASK_CODES[mask]
        desc.window = window
        desc.max_seqlen_q = max_seqlen_q
        desc.max_seqlen_k = max_seqlen_k
        desc.batch = batch

        handle = ctypes.c_void_p()
        status = self._lib.aotriton_prepare(ctypes.byref(desc), ctypes.byref(handle))
        if status != 0 or not handle.value:
            raise AotritonError(None, self._last_error())
        return PreparedCall(self._lib, handle)


class PreparedCall:
    """A prepared native call. Not thread-safe; not reentrant across launches
    on different streams without external synchronization -- exactly one
    ``attn_fwd_params`` (and, for causal/windowed cases, one small owned
    device scratch counter) backs every ``launch()``."""

    def __init__(self, lib: ctypes.CDLL, handle: ctypes.c_void_p):
        self._lib = lib
        self._handle = handle
        self._closed = False

    def launch(self, stream: int = 0, backend: int = -1) -> None:
        """Enqueue the prepared forward call on ``stream`` (a raw
        ``hipStream_t`` value, ``0`` for the default stream).

        ``backend`` forwards to ``attn_options.force_backend_index``; ``-1``
        lets AOTriton pick (``context.lookup_optimal``). No allocation, no
        parameter rebuilding -- safe to call from inside a HIP graph capture.
        """
        if self._closed:
            raise AotritonError(None, "launch() called on a closed PreparedCall")
        hip_status = ctypes.c_int32()
        status = self._lib.aotriton_launch(self._handle, ctypes.c_void_p(stream), backend, ctypes.byref(hip_status))
        if status != 0:
            msg = self._lib.aotriton_last_error()
            raise AotritonError(hip_status.value, msg.decode("utf-8", "replace") if msg else "")

    def close(self) -> None:
        if not self._closed:
            self._lib.aotriton_destroy(self._handle)
            self._closed = True

    def __enter__(self) -> "PreparedCall":
        return self

    def __exit__(self, *exc_info) -> bool:
        self.close()
        return False

    def __del__(self) -> None:  # best-effort; explicit close()/context-manager is the real contract
        try:
            self.close()
        except Exception:
            pass

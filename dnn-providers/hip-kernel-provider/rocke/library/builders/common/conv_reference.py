# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Reference convolutions for verify paths.

The float64 references of a ``ConvProblem`` -- :func:`conv_fwd_reference`,
:func:`conv_dgrad_reference` and :func:`conv_wgrad_reference` -- run
:mod:`rocke.numeric.conv` on numpy arrays in the kernel layouts (NHWC / NDHWC
activations, KYXC / KZYXC weights), over image slices that bound their host
memory. They need neither torch nor a GPU.

The TF32 helpers (:func:`tf32_reference_inputs`, :func:`tf32_reference_tol`)
say which operand values and which bound a TF32 kernel is judged with.

The torch references below remain for the benchmarks and tests that still
call them:

* ``conv_reference``         — torch-based (``F.conv2d`` / ``F.conv3d``), works
  on any GPU target.
* ``conv_reference_gfx1250`` — hand-written numpy reference for gfx1250.
  Accumulates in float32 with no GPU driver dependency.  Only supports 2-D
  NHWC forward convolution.
* ``dgrad_reference``        — torch-based backward-data reference via
  ``torch.nn.grad.conv2d_input``.
* ``wgrad_reference``        — torch-based backward-weight reference via
  ``torch.nn.grad.conv{2,3}d_weight``.
* ``dgrad_reference_gfx1250`` — hand-written numpy reference for gfx1250.
  Accumulates in float32 with no GPU driver dependency.  Only supports 2-D
  NHWK/KYXC → NHWC dgrad convolution.
* ``wgrad_reference_gfx1250`` — hand-written numpy reference for gfx1250.
  Accumulates in float32 with no GPU driver dependency.  Only supports 2-D
  NHWC/NHWK → KYXC wgrad convolution.
"""

from __future__ import annotations

import contextlib
import functools
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from rocke.numeric.conv import conv_dgrad, conv_fwd, conv_wgrad
from rocke.numeric.storage import xf32_truncate

# torch is optional for the library: each reference imports it when called, so
# the torch-free helpers here (batch slicing, tolerances) and the modules that
# import this one load without it.
if TYPE_CHECKING:
    import torch
    from kernels.common.conv_implicit_gemm import ConvProblem


# torch's GPU convolutions mis-compute an fp32 operand larger than 2^32 bytes:
# the batch images past that offset come back wrong (measured with MIOpen on a
# 4.8 GB fp32 activation). The torch forward and backward-data references
# therefore run batch slices whose fp32 operands stay below a 2^31-byte cap --
# well under the 2^32 break point, so a slice can never straddle it -- and
# concatenate them; every image is independent in those two directions.
_MAX_SLICE_BYTES = 2**31 - 1

#: Default cap on the float64 bytes of one image slice's input or output
#: activation in the numpy references. They hold a few float64 copies of the
#: activations they work on, so a larger batch runs as image slices: this
#: bounds their host memory, not their result.
REFERENCE_SLICE_BYTES = 2**28


def _out_extent(i: int, f: int, s: int, pad: int, d: int) -> int:
    return (i + 2 * pad - d * (f - 1) - 1) // s + 1


def _image_elems(p) -> int:
    """Elements of one image's input or output activation, whichever is larger."""
    out = (
        _out_extent(p.Hi, p.Y, p.sH, p.pH, p.dH)
        * _out_extent(p.Wi, p.X, p.sW, p.pW, p.dW)
        * p.K
    )
    inp = p.Hi * p.Wi * p.C
    if p.is_3d:
        out *= _out_extent(p.Di, p.Z, p.sD, p.pD, p.dD)
        inp *= p.Di
    return max(out, inp)


def _batch_slices(
    p, n: int, *, element_bytes: int = 4, max_bytes: int = _MAX_SLICE_BYTES
):
    """``[(n0, count), ...]`` covering a batch of ``n`` in slices whose larger
    activation, at ``element_bytes`` per element, stays within ``max_bytes``
    (one image per slice at least). The defaults are the torch references'
    fp32 cap."""
    step = max(1, max_bytes // (element_bytes * _image_elems(p)))
    return [(n0, min(step, n - n0)) for n0 in range(0, n, step)]


class ConvExtents(NamedTuple):
    """Spatial extents of a ``ConvProblem``, one per axis, depth first for 3-D."""

    in_spatial: tuple[int, ...]
    out_spatial: tuple[int, ...]
    filter_spatial: tuple[int, ...]


def conv_extents(problem: ConvProblem) -> ConvExtents:
    """The input, output and filter extents of ``problem``."""
    p = problem
    if p.is_3d:
        return ConvExtents((p.Di, p.Hi, p.Wi), (p.Do, p.Ho, p.Wo), (p.Z, p.Y, p.X))
    return ConvExtents((p.Hi, p.Wi), (p.Ho, p.Wo), (p.Y, p.X))


def conv_geometry(problem: ConvProblem) -> dict[str, tuple[int, ...] | int]:
    """``stride`` / ``padding`` / ``dilation`` (one per spatial axis, depth
    first for 3-D) and ``groups`` of ``problem``: the keyword arguments of the
    :mod:`rocke.numeric.conv` references."""
    p = problem
    if p.is_3d:
        return {
            "stride": (p.sD, p.sH, p.sW),
            "padding": (p.pD, p.pH, p.pW),
            "dilation": (p.dD, p.dH, p.dW),
            "groups": p.groups,
        }
    return {
        "stride": (p.sH, p.sW),
        "padding": (p.pH, p.pW),
        "dilation": (p.dH, p.dW),
        "groups": p.groups,
    }


def _image_batches(problem: ConvProblem, n: int, max_slice_bytes: int):
    """The batch of ``n`` images as slices of float64 activations of at most
    ``max_slice_bytes`` each."""
    return [
        slice(n0, n0 + count)
        for n0, count in _batch_slices(
            problem, n, element_bytes=8, max_bytes=max_slice_bytes
        )
    ]


def conv_fwd_reference(
    x: np.ndarray,
    w: np.ndarray,
    problem: ConvProblem,
    *,
    max_slice_bytes: int = REFERENCE_SLICE_BYTES,
) -> np.ndarray:
    """float64 forward output ``[N, (Do,) Ho, Wo, K]`` of ``problem`` for the
    activations ``x [N, (Di,) Hi, Wi, C]`` and the weights
    ``w [K, (Z,) Y, X, C/groups]``, computed over image slices of at most
    ``max_slice_bytes`` float64 activation bytes."""
    geometry = conv_geometry(problem)
    batches = _image_batches(problem, x.shape[0], max_slice_bytes)
    return np.concatenate([conv_fwd(x[batch], w, **geometry) for batch in batches])


def conv_dgrad_reference(
    dy: np.ndarray,
    w: np.ndarray,
    problem: ConvProblem,
    *,
    max_slice_bytes: int = REFERENCE_SLICE_BYTES,
) -> np.ndarray:
    """float64 data gradient ``[N, (Di,) Hi, Wi, C]`` of ``problem`` for the
    output gradient ``dy [N, (Do,) Ho, Wo, K]`` and the weights
    ``w [K, (Z,) Y, X, C/groups]``, computed over image slices of at most
    ``max_slice_bytes`` float64 activation bytes."""
    geometry = conv_geometry(problem)
    in_spatial = conv_extents(problem).in_spatial
    batches = _image_batches(problem, dy.shape[0], max_slice_bytes)
    return np.concatenate(
        [conv_dgrad(dy[b], w, in_spatial=in_spatial, **geometry) for b in batches]
    )


def conv_wgrad_reference(
    x: np.ndarray,
    dy: np.ndarray,
    problem: ConvProblem,
    *,
    max_slice_bytes: int = REFERENCE_SLICE_BYTES,
) -> np.ndarray:
    """float64 weight gradient ``[K, (Z,) Y, X, C/groups]`` of ``problem`` for
    the activations ``x [N, (Di,) Hi, Wi, C]`` and the output gradient
    ``dy [N, (Do,) Ho, Wo, K]``: the sum of the image slices' gradients, each
    slice at most ``max_slice_bytes`` float64 activation bytes."""
    geometry = conv_geometry(problem)
    filter_spatial = conv_extents(problem).filter_spatial
    batches = _image_batches(problem, x.shape[0], max_slice_bytes)
    return sum(
        conv_wgrad(x[b], dy[b], filter_spatial=filter_spatial, **geometry)
        for b in batches
    )


@contextlib.contextmanager
def ieee_fp32():
    """Run torch's fp32 convolutions in IEEE fp32, never TF32.

    The references below are the fp32 truth a TF32 kernel is judged against,
    and torch >= 2.9 defaults ``torch.backends.cudnn.conv.fp32_precision`` to
    ``"tf32"`` (``allow_tf32=True`` before that): once the backend honours it,
    the reference itself would silently become TF32. The new per-operator API
    is used where it exists, for the GPU (cuDNN / MIOpen) and the CPU (oneDNN)
    convolutions alike -- mixing it with the legacy ``allow_tf32`` flag makes
    torch raise -- and the previous settings are restored on exit.
    """
    import torch

    cudnn = torch.backends.cudnn
    convs = [
        conv
        for conv in (
            getattr(cudnn, "conv", None),
            getattr(getattr(torch.backends, "mkldnn", None), "conv", None),
        )
        if conv is not None and hasattr(conv, "fp32_precision")
    ]
    if convs:
        old = [conv.fp32_precision for conv in convs]
        for conv in convs:
            conv.fp32_precision = "ieee"
        try:
            yield
        finally:
            for conv, prec in zip(convs, old):
                conv.fp32_precision = prec
    else:
        old = cudnn.allow_tf32
        cudnn.allow_tf32 = False
        try:
            yield
        finally:
            cudnn.allow_tf32 = old


# A TF32 kernel's error, peak-relative, against :func:`tf32_reference_inputs`:
# gfx950's BF16x3 emulation is about fp32 accurate, and 1e-4 fails a single
# bf16 pass or a dropped cross term. gfx942's native XF32 is judged on masked
# inputs, which leaves only accumulation error, held to 1e-3.
TF32_REFERENCE_TOL = {"emulated": 1e-4, "native": 1e-3}


def _tf32_native(arch: str) -> bool:
    from kernels.conv.direct_aot_jobs import tf32_is_emulated

    return not tf32_is_emulated(arch)


def tf32_reference_inputs(values: np.ndarray, arch: str, dtype: str = "tf32"):
    """fp32 ``values`` as the MMA on ``arch`` consumes them for a ``dtype`` problem.

    A native XF32 MFMA (gfx942) reads the top 19 bits of each fp32 operand
    (:func:`rocke.numeric.storage.xf32_truncate`); the BF16x3 emulation (gfx950)
    and the fp32 atoms use the full payload, so every other case returns
    ``values`` unchanged. The arch question is the catalog's (``tf32_is_emulated``).
    """
    if dtype != "tf32" or not _tf32_native(arch):
        return values
    return xf32_truncate(values)


def tf32_reference_tol(arch: str) -> float:
    """The TF32 bound for ``arch`` against :func:`tf32_reference_inputs`."""
    return TF32_REFERENCE_TOL["native" if _tf32_native(arch) else "emulated"]


def _ieee(fn):
    """Decorate a torch reference to run under :func:`ieee_fp32`."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with ieee_fp32():
            return fn(*args, **kwargs)

    return wrapper


@_ieee
def wgrad_reference(X: torch.Tensor, dY: torch.Tensor, p) -> torch.Tensor:
    """Compute a float32 reference weight gradient for a convolution problem.

    Uses ``torch.nn.grad.conv2d_weight`` / ``conv3d_weight`` so the result is
    numerically identical to what autograd would produce.  The output layout
    matches the wgrad kernel convention: KYXC for 2-D, KZYXC for 3-D.

    Args:
        X:  Input activations, shape (N, H, W, C) or (N, D, H, W, C), any dtype.
        dY: Output gradient, shape (N, Ho, Wo, K) or (N, Do, Ho, Wo, K), any dtype.
        p:  ConvProblem carrying stride/padding/dilation/groups.

    Returns:
        Weight gradient as a float32 torch.Tensor in KYXC / KZYXC layout.
    """
    import torch

    if not p.is_3d:
        X_t = X.float().cuda().permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
        dY_t = dY.float().cuda().permute(0, 3, 1, 2).contiguous()  # NHWK -> NKHW
        dW_nchw = torch.nn.grad.conv2d_weight(
            X_t,
            weight_size=(p.K, p.C // p.groups, p.Y, p.X),
            grad_output=dY_t,
            stride=(p.sH, p.sW),
            padding=(p.pH, p.pW),
            dilation=(p.dH, p.dW),
            groups=p.groups,
        )
        # KCHW -> KHWC (KYXC)
        return dW_nchw.permute(0, 2, 3, 1).contiguous()
    else:
        X_t = X.float().cuda().permute(0, 4, 1, 2, 3).contiguous()  # NDHWC -> NCDHW
        dY_t = dY.float().cuda().permute(0, 4, 1, 2, 3).contiguous()  # NDHWK -> NKDHW
        dW_ncdhw = torch.nn.grad.conv3d_weight(
            X_t,
            weight_size=(p.K, p.C // p.groups, p.Z, p.Y, p.X),
            grad_output=dY_t,
            stride=(p.sD, p.sH, p.sW),
            padding=(p.pD, p.pH, p.pW),
            dilation=(p.dD, p.dH, p.dW),
            groups=p.groups,
        )
        # KCDHW -> KDHWC (KZYXC)
        return dW_ncdhw.permute(0, 2, 3, 4, 1).contiguous()


@_ieee
def dgrad_reference(dY: torch.Tensor, W: torch.Tensor, p) -> torch.Tensor:
    """Compute a float32 reference input gradient for a convolution problem.

    Uses ``torch.nn.grad.conv2d_input`` so the result is numerically identical
    to what autograd would produce.  The output layout matches the dgrad kernel
    convention: NHWC for 2-D.

    Args:
        dY: Output gradient, shape (N, Ho, Wo, K), any dtype.
        W:  Weight tensor, shape (K, Y, X, C), any dtype.
        p:  ConvProblem carrying stride/padding/dilation/groups.

    Returns:
        Input gradient as a float32 torch.Tensor in NHWC layout.
    """
    import torch

    slices = _batch_slices(p, dY.shape[0])
    if len(slices) > 1:
        return torch.cat([dgrad_reference(dY[n0 : n0 + n], W, p) for n0, n in slices])
    dY_t = dY.float().cuda().permute(0, 3, 1, 2).contiguous()  # NHWK -> NKHW
    W_t = W.float().cuda().permute(0, 3, 1, 2).contiguous()  # KYXC -> KCYX
    dX_nchw = torch.nn.grad.conv2d_input(
        input_size=(dY.shape[0], p.C, p.Hi, p.Wi),
        weight=W_t,
        grad_output=dY_t,
        stride=(p.sH, p.sW),
        padding=(p.pH, p.pW),
        dilation=(p.dH, p.dW),
        groups=p.groups,
    )
    return dX_nchw.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC


@_ieee
def conv_reference(
    A: torch.Tensor, B: torch.Tensor, p, out_dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Compute a reference output for a forward convolution problem.

    Both 2-D (NHWC input, KHWC weight) and 3-D (NDHWC input, KDHWC weight)
    problems are supported.  The computation is always done in float32.
    The output layout matches the kernel convention: NHWC for 2-D, NDHWC for 3-D.

    Args:
        A: Input tensor, shape (N, H, W, C) or (N, D, H, W, C), any dtype.
        B: Weight tensor, shape (K, Y, X, C) or (K, Z, Y, X, C), any dtype.
        p: ConvProblem instance carrying stride/padding/dilation/groups.
        out_dtype: If given, the fp32 result is cast to this dtype then back to
            float32.  This simulates the kernel's output rounding so that the
            reference matches the precision the kernel can actually achieve.

    Returns:
        Reference output as a float32 torch.Tensor.
    """
    import torch
    import torch.nn.functional as F

    # The backend's conv produces garbage once an fp32 operand passes 2**31
    # bytes (seen on an 8x768x768x256 batch: the last image is wrong). The
    # batch slices keep each call under that; an image past it on its own
    # cannot be split, so it is refused instead of verified against garbage.
    img_bytes = 4 * _image_elems(p)
    if img_bytes > _MAX_SLICE_BYTES:
        raise ValueError(
            f"conv_reference: one {p.Hi}x{p.Wi}x{p.C} image needs "
            f"{img_bytes} fp32 bytes, past the backend's 2**31 limit; "
            "the batch split cannot make it fit"
        )
    slices = _batch_slices(p, A.shape[0])
    if len(slices) > 1:
        return torch.cat(
            [conv_reference(A[n0 : n0 + n], B, p, out_dtype) for n0, n in slices]
        )
    if not p.is_3d:
        # B layout is KYXC with C = C_total. PyTorch F.conv2d expects weights
        # shaped [K, C/groups, Y, X]. Each filter stores cpg channels contiguously
        # at the start of its C dimension; slice to cpg then permute.
        Cg = p.C // p.groups
        B_t = (
            B.float().cuda()[:, :, :, :Cg].permute(0, 3, 1, 2)
        )  # [K,Y,X,Cg]->[K,Cg,Y,X]
        A_t = A.float().cuda().permute(0, 3, 1, 2)  # NHWC -> NCHW
        result = (
            F.conv2d(
                A_t,
                B_t,
                stride=(p.sH, p.sW),
                padding=(p.pH, p.pW),
                dilation=(p.dH, p.dW),
                groups=p.groups,
            )
            .permute(0, 2, 3, 1)  # NCHW -> NHWC
            .contiguous()
        )
    else:
        A_t = A.float().cuda().permute(0, 4, 1, 2, 3)  # NDHWC -> NCDHW
        Cg = p.C // p.groups
        B_t = (
            B.float().cuda()[:, :, :, :, :Cg].permute(0, 4, 1, 2, 3)
        )  # [K,Z,Y,X,Cg]->[K,Cg,Z,Y,X]
        result = (
            F.conv3d(
                A_t,
                B_t,
                stride=(p.sD, p.sH, p.sW),
                padding=(p.pD, p.pH, p.pW),
                dilation=(p.dD, p.dH, p.dW),
                groups=p.groups,
            )
            .permute(0, 2, 3, 4, 1)  # NCDHW -> NDHWC
            .contiguous()
        )
    if out_dtype is not None:
        result = result.to(out_dtype).float()
    return result


def conv_reference_gfx1250(
    A: torch.Tensor,
    B: torch.Tensor,
    p,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Hand-written float32 reference for 2-D NHWC forward convolution.

    Implements the convolution directly in numpy so there is no dependency on
    torch.nn or any GPU driver.  Accumulation is always in float32 to keep
    the reference numerically clean even for large filter footprints.

    Supports grouped convolution (``p.groups > 1``) and arbitrary
    stride / padding / dilation.  3-D problems are not supported; call
    ``conv_reference`` instead.

    Args:
        A: Input tensor, shape ``(N, Hi, Wi, C)``, any dtype.
        B: Weight tensor, shape ``(K, Y, X, C)``, any dtype.
        p: ``ConvProblem`` instance (carries stride / padding / dilation /
           groups).
        out_dtype: If given, the fp32 result is cast to this dtype then back to
            float32.  This simulates the kernel's output rounding so that the
            reference matches the precision the kernel can actually achieve.

    Returns:
        Float32 torch.Tensor of shape ``(N, Ho, Wo, K)`` on CPU.
    """
    import numpy as np
    import torch

    A_np = A.float().cpu().numpy()  # (N, Hi, Wi, C)
    B_np = B.float().cpu().numpy()  # (K, Y, X, C)

    N, Hi, Wi, C = A_np.shape
    K = B_np.shape[0]
    Ho, Wo = p.Ho, p.Wo
    g = p.groups
    Cg = C // g  # input channels per group
    Kg = K // g  # output channels per group

    out = np.zeros((N, Ho, Wo, K), dtype=np.float32)

    for grp in range(g):
        a_grp = A_np[:, :, :, grp * Cg : (grp + 1) * Cg]  # (N, Hi, Wi, Cg)
        b_grp = B_np[grp * Kg : (grp + 1) * Kg]  # (Kg, Y, X, Cg)
        for n in range(N):
            for ho in range(Ho):
                for wo in range(Wo):
                    for y in range(p.Y):
                        hi = ho * p.sH - p.pH + y * p.dH
                        if hi < 0 or hi >= Hi:
                            continue
                        for x in range(p.X):
                            wi = wo * p.sW - p.pW + x * p.dW
                            if wi < 0 or wi >= Wi:
                                continue
                            # a_vec: (Cg,)  b_slice: (Kg, Cg)
                            a_vec = a_grp[n, hi, wi, :].astype(np.float32)
                            b_slice = b_grp[:, y, x, :].astype(np.float32)
                            out[n, ho, wo, grp * Kg : (grp + 1) * Kg] += b_slice @ a_vec

    result = torch.from_numpy(out.astype(np.float32))
    if out_dtype is not None:
        result = result.to(out_dtype).float()
    return result


def dgrad_reference_gfx1250(
    dY: torch.Tensor,
    W: torch.Tensor,
    p,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Hand-written float32 reference for 2-D NHWK dgrad convolution.

    Implements the input-gradient computation directly in numpy so there is no
    dependency on torch.nn or any GPU driver.  Accumulation is always in float32.

    Supports grouped convolution (``p.groups > 1``) and arbitrary
    stride / padding / dilation.  3-D problems are not supported; call
    ``dgrad_reference`` instead.

    Args:
        dY: Output gradient, shape ``(N, Ho, Wo, K)``, any dtype.
        W:  Weight tensor, shape ``(K, Y, X, C)``, any dtype.
        p:  ``ConvProblem`` instance (carries stride / padding / dilation /
            groups, Hi, Wi).
        out_dtype: If given, the fp32 result is cast to this dtype then back to
            float32.

    Returns:
        Float32 torch.Tensor of shape ``(N, Hi, Wi, C)`` on CPU.
    """
    import numpy as np
    import torch

    dY_np = dY.float().cpu().numpy()  # (N, Ho, Wo, K)
    W_np = W.float().cpu().numpy()  # (K, Y, X, C)

    N, Ho, Wo, K = dY_np.shape
    Hi, Wi = p.Hi, p.Wi
    g = p.groups
    C = p.C
    Cg = C // g  # input channels per group
    Kg = K // g  # output channels per group
    if W_np.shape[3] != Cg:
        raise ValueError(
            f"expected W last dim Cg={Cg} (=C/groups), got {W_np.shape[3]}"
        )

    dX = np.zeros((N, Hi, Wi, C), dtype=np.float32)
    for grp in range(g):
        w_grp = W_np[grp * Kg : (grp + 1) * Kg]  # (Kg, Y, X, Cg)
        for n in range(N):
            for ho in range(Ho):
                for wo in range(Wo):
                    dy_vec = dY_np[n, ho, wo, grp * Kg : (grp + 1) * Kg].astype(
                        np.float32
                    )  # (Kg,)
                    for y in range(p.Y):
                        hi = ho * p.sH - p.pH + y * p.dH
                        if hi < 0 or hi >= Hi:
                            continue
                        for x in range(p.X):
                            wi = wo * p.sW - p.pW + x * p.dW
                            if wi < 0 or wi >= Wi:
                                continue
                            # w_grp[:, y, x, :] shape: (Kg, Cg)
                            # dy_vec @ w_grp[:, y, x, :] -> (Cg,)
                            dX[n, hi, wi, grp * Cg : (grp + 1) * Cg] += dy_vec @ w_grp[
                                :, y, x, :
                            ].astype(np.float32)

    result = torch.from_numpy(dX)
    if out_dtype is not None:
        result = result.to(out_dtype).float()
    return result


def wgrad_reference_gfx1250(
    X: torch.Tensor,
    dY: torch.Tensor,
    p,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Hand-written float32 reference for 2-D KYXC wgrad convolution.

    Implements the weight-gradient computation directly in numpy so there is no
    dependency on torch.nn or any GPU driver.  Accumulation is always in float32.

    Supports grouped convolution (``p.groups > 1``) and arbitrary
    stride / padding / dilation.  3-D problems are not supported; call
    ``wgrad_reference`` instead.

    Args:
        X:  Input activations, shape ``(N, Hi, Wi, C)``, any dtype.
        dY: Output gradient, shape ``(N, Ho, Wo, K)``, any dtype.
        p:  ``ConvProblem`` instance (carries stride / padding / dilation /
            groups).
        out_dtype: If given, the fp32 result is cast to this dtype then back to
            float32.

    Returns:
        Float32 torch.Tensor of shape ``(K, Y, X, Cg)`` on CPU, where
        ``Cg = C // groups`` (per-group input channels).
    """
    import numpy as np
    import torch

    X_np = X.float().cpu().numpy()  # (N, Hi, Wi, C)
    dY_np = dY.float().cpu().numpy()  # (N, Ho, Wo, K)

    N, Hi, Wi, C = X_np.shape
    _, Ho, Wo, K = dY_np.shape
    g = p.groups
    Cg = C // g  # input channels per group
    Kg = K // g  # output channels per group

    dW = np.zeros((K, p.Y, p.X, Cg), dtype=np.float32)
    for grp in range(g):
        x_grp = X_np[:, :, :, grp * Cg : (grp + 1) * Cg]  # (N, Hi, Wi, Cg)
        for n in range(N):
            for ho in range(Ho):
                for wo in range(Wo):
                    dy_vec = dY_np[n, ho, wo, grp * Kg : (grp + 1) * Kg].astype(
                        np.float32
                    )  # (Kg,)
                    for y in range(p.Y):
                        hi = ho * p.sH - p.pH + y * p.dH
                        if hi < 0 or hi >= Hi:
                            continue
                        for x in range(p.X):
                            wi = wo * p.sW - p.pW + x * p.dW
                            if wi < 0 or wi >= Wi:
                                continue
                            x_vec = x_grp[n, hi, wi, :].astype(np.float32)  # (Cg,)
                            # outer product: (Kg,1) * (1,Cg) -> (Kg, Cg)
                            dW[
                                grp * Kg : (grp + 1) * Kg,
                                y,
                                x,
                                :,
                            ] += (
                                dy_vec[:, None] * x_vec[None, :]
                            )

    result = torch.from_numpy(dW)
    if out_dtype is not None:
        result = result.to(out_dtype).float()
    return result

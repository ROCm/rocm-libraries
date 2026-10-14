# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host launcher for the MLA decode pipeline, over caller-owned torch tensors.

A dispatched decode selection names one kernel (the latent kernel); with
``spec.split_kv`` the launch is two, the latent kernel writing fp32 partials
and :mod:`kernels.mla.mla_decode_combine` merging them. This module owns that
composition, so the verify harness and the benchmark launch exactly the same
thing, from the same dispatched spec.

The whole op around it, for a decode step::

    [A] qa     = [q_nope @ W_UK[:, :, :d_nope]ᵀ | q_rope]   torch.bmm, per head
    [B] latent = decode(qa, c_kv, k_rope)                   this runner
    [C]          (combine, when split)                      this runner
    [D] out    = latent @ W_UK[:, :, d_nope:]               torch.bmm, per head

:meth:`MlaDecodeRunner.forward` runs all four; :meth:`MlaDecodeRunner.latent`
runs [B] + [C] only. torch is imported lazily: this is a GPU harness, and the
library itself stays importable without it.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from kernels.mla.mla_decode import mla_decode_combine_spec, mla_decode_workspace_shapes
from kernels.mla.mla_decode_combine import (
    build_mla_decode_combine,
    mla_decode_combine_block,
    mla_decode_combine_grid,
    mla_decode_combine_signature,
)

if TYPE_CHECKING:
    from rocke.dispatch.core import DispatchResult

__all__ = ["MlaDecodeRunner"]


class MlaDecodeRunner:
    """Compile-once launcher for one dispatched decode selection.

    ``result`` is a ``DispatchResult`` from ``dispatch_mla``; its request is the
    launch contract, and the tensors handed to :meth:`latent` are checked
    against it. ``num_seg`` overrides the dispatched split count (for parity
    sweeps); the kernel is rebuilt through the same candidate, so the override
    can change the split count but not the kernel family.
    """

    def __init__(self, result: DispatchResult, *, num_seg: int | None = None) -> None:
        from rocke import compile_kernel
        from rocke.runtime import KernelLauncher

        spec = result.spec
        if num_seg is not None:
            spec = dataclasses.replace(spec, num_seg=int(num_seg))
        self.request = result.request
        self.candidate = result.candidate
        self.spec = spec
        self.arch = self.request.arch
        self.grid = tuple(self.candidate.grid(spec, self.request))
        self.block = tuple(self.candidate.block(spec))
        art = compile_kernel(
            self.candidate.built(spec, self.arch), arch=self.arch, capture_ir_text=False
        )
        # Keyed by kernel name: every split count shares one split binary.
        self._latent = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=tuple(self.candidate.signature(spec)),
            cache_key=("mla_decode", self.arch, spec.kernel_name()),
        )
        self._combine = None
        if spec.split_kv:
            cspec = mla_decode_combine_spec(spec)
            cart = compile_kernel(
                build_mla_decode_combine(cspec, arch=self.arch),
                arch=self.arch,
                capture_ir_text=False,
            )
            self._combine_spec = cspec
            self._combine = KernelLauncher(
                hsaco=cart.hsaco,
                kernel_name=cart.kernel_name,
                signature=tuple(mla_decode_combine_signature(cspec)),
                cache_key=("mla_decode_combine", self.arch, cspec.kernel_name()),
            )
        self._workspace: dict[tuple[int, str], tuple[Any, Any]] = {}

    @property
    def kernel_names(self) -> tuple[str, ...]:
        names = (self._latent.kernel_name,)
        if self._combine is not None:
            names += (self._combine.kernel_name,)
        return names

    def workspace(self, total_q: int, device) -> tuple[Any, Any]:
        """The split kernel's fp32 ``(part_acc, part_ml)``, kept for reuse
        across calls of the same ``total_q``."""
        import torch

        key = (total_q, str(device))
        if key not in self._workspace:
            shapes = mla_decode_workspace_shapes(self.spec, total_q=total_q)
            self._workspace = {
                key: tuple(
                    torch.empty(shapes[name], dtype=torch.float32, device=device)
                    for name in ("part_acc_ptr", "part_ml_ptr")
                )
            }
        return self._workspace[key]

    def _check_inputs(self, qa, c_kv, k_rope, cu_seqlens_q, cu_seqlens_k, block_table):
        """Tie the caller's tensors to the dispatched request before launch.

        The grid, ``num_seqs`` and the kernel's indexing all come from the
        request; a tensor that disagrees would be read out of bounds on the
        device rather than fail here.
        """
        req, spec = self.request, self.spec
        expect = {
            "qa": ((req.total_q, spec.num_heads, spec.qa_cols), qa),
            "cu_seqlens_q": ((req.num_seqs + 1,), cu_seqlens_q),
            "cu_seqlens_k": ((req.num_seqs + 1,), cu_seqlens_k),
        }
        for name, (shape, t) in expect.items():
            if tuple(t.shape) != shape:
                raise ValueError(
                    f"{name} has shape {tuple(t.shape)}, the dispatched request "
                    f"needs {shape}; dispatch again for a new batch"
                )
        for name, t in (("qa", qa), ("c_kv", c_kv), ("k_rope", k_rope)):
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if block_table.dim() != 2 or block_table.stride(1) != 1:
            raise ValueError("block_table must be 2-D with unit column stride")
        if block_table.shape[0] < req.num_seqs:
            raise ValueError(
                f"block_table has {block_table.shape[0]} rows < num_seqs={req.num_seqs}"
            )

    def latent(
        self,
        *,
        qa,
        c_kv,
        k_rope,
        cu_seqlens_q,
        cu_seqlens_k,
        block_table,
        scale: float,
        out=None,
        lse=None,
        stream: int = 0,
    ):
        """[B] (+ [C]): ``(latent [total_q, H, r_kv] bf16, lse [total_q, H] f32)``.

        ``out`` / ``lse`` are allocated when not given. ``scale`` is the raw
        host scale, the same value the reference receives.
        """
        import torch
        from rocke.runtime import LaunchConfig

        self._check_inputs(qa, c_kv, k_rope, cu_seqlens_q, cu_seqlens_k, block_table)
        spec = self.spec
        total_q = int(qa.shape[0])
        if out is None:
            out = torch.empty(
                total_q, spec.num_heads, spec.r_kv, dtype=qa.dtype, device=qa.device
            )
        if lse is None:
            lse = torch.empty(
                total_q, spec.num_heads, dtype=torch.float32, device=qa.device
            )
        vals = {
            "qa_ptr": qa,
            "c_kv_ptr": c_kv,
            "k_rope_ptr": k_rope,
            "cu_seqlens_q_ptr": cu_seqlens_q,
            "cu_seqlens_k_ptr": cu_seqlens_k,
            "block_table_ptr": block_table,
            "scale": float(scale),
            "num_seqs": int(self.request.num_seqs),
            # The kernel's row pitch: the stride, which a column-sliced view
            # does not share with its shape.
            "block_table_stride": int(block_table.stride(0)),
            "max_k": int(self.request.max_seqlen_k),
        }
        config = LaunchConfig(grid=self.grid, block=self.block, stream=stream)
        if self._combine is None:
            vals.update(out_ptr=out, lse_ptr=lse)
            self._latent(vals, config=config)
            return out, lse
        part_acc, part_ml = self.workspace(total_q, qa.device)
        vals.update(part_acc_ptr=part_acc, part_ml_ptr=part_ml, num_seg=spec.num_seg)
        self._latent(vals, config=config)
        self._combine(
            {
                "out_ptr": out,
                "lse_ptr": lse,
                "part_acc_ptr": part_acc,
                "part_ml_ptr": part_ml,
                "num_seg": spec.num_seg,
            },
            config=LaunchConfig(
                grid=mla_decode_combine_grid(self._combine_spec, total_q=total_q),
                block=mla_decode_combine_block(self._combine_spec),
                stream=stream,
            ),
        )
        return out, lse

    def absorb(self, *, q_nope, q_rope, w_uk, out) -> None:
        """[A] into ``out`` ``[total_q, H, r_kv + d_rope]``: per head,
        ``Qa_nope[h] = Q_nope[h] @ W_UK[h, :, :d_nope]ᵀ``, then ``Q_rope``."""
        import torch

        d_nope = q_nope.shape[-1]
        r_kv = self.spec.r_kv
        torch.bmm(
            q_nope.transpose(0, 1),
            w_uk[:, :, :d_nope].transpose(1, 2),
            out=out[:, :, :r_kv].transpose(0, 1),
        )
        out[:, :, r_kv:].copy_(q_rope)

    def project(self, *, latent, w_uk, d_nope: int, out) -> None:
        """[D] into ``out`` ``[total_q, H, d_v]``: per head,
        ``out[h] = latent[h] @ W_UK[h, :, d_nope:]``."""
        import torch

        torch.bmm(latent.transpose(0, 1), w_uk[:, :, d_nope:], out=out.transpose(0, 1))

    def forward(
        self,
        *,
        q_nope,
        q_rope,
        w_uk,
        c_kv,
        k_rope,
        cu_seqlens_q,
        cu_seqlens_k,
        block_table,
        scale: float,
        stream: int = 0,
    ):
        """The whole decode op, [A] to [D]: ``(out [total_q, H, d_v], lse)``.

        ``w_uk`` is the per-head ``[H, r_kv, d_nope + d_v]`` weight, both
        halves read as strided views; nothing is materialized but ``qa`` and
        the latent.
        """
        import torch

        total_q, num_heads, d_nope = q_nope.shape
        qa = torch.empty(
            total_q,
            num_heads,
            self.spec.qa_cols,
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
        self.absorb(q_nope=q_nope, q_rope=q_rope, w_uk=w_uk, out=qa)
        latent, lse = self.latent(
            qa=qa,
            c_kv=c_kv,
            k_rope=k_rope,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_table=block_table,
            scale=scale,
            stream=stream,
        )
        out = torch.empty(
            total_q,
            num_heads,
            w_uk.shape[2] - d_nope,
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
        self.project(latent=latent, w_uk=w_uk, d_nope=d_nope, out=out)
        return out, lse

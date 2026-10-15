# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host launcher for the MLA decode pipeline, over caller-owned torch tensors.

A dispatched decode selection is one launch (the latent kernel) or, with
``spec.split_kv``, two: the latent kernel writing fp32 partials and
:mod:`kernels.mla.mla_decode_combine` merging them.
:func:`dispatch.mla.mla_decode_launch` describes both; this module launches
them, so the verify harness and the benchmark run exactly what dispatch
selected.

The whole op around it, for a decode step::

    [A] qa     = [q_nope @ W_UK_nopeᵀ | q_rope]   torch.bmm, per head
    [B] latent = decode(qa, c_kv, k_rope)         this runner
    [C]          (combine, when split)            this runner
    [D] out    = latent @ W_UV                    torch.bmm, per head

``w_uk`` is the fused per-head up-projection ``[H, r_kv, d_nope + d_v]``:
``W_UK_nope = w_uk[:, :, :d_nope]`` and ``W_UV = w_uk[:, :, d_nope:]``.
:meth:`MlaDecodeRunner.forward` runs all four; :meth:`MlaDecodeRunner.latent`
runs [B] + [C] only. torch is imported lazily: this is a GPU harness, and the
library itself stays importable without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dispatch.mla import mla_decode_launch

if TYPE_CHECKING:
    import torch
    from rocke.dispatch.core import DispatchResult

__all__ = ["MlaDecodeRunner"]


class MlaDecodeRunner:
    """Compile-once launcher for one dispatched decode selection.

    ``result`` is a decode ``DispatchResult`` from ``dispatch_mla``; its
    request is the launch contract, and the tensors handed to :meth:`latent`
    are checked against it. ``num_seg`` overrides the dispatched split count
    (for parity sweeps) through :meth:`dispatch.mla.MlaDecodeLaunch.with_num_seg`,
    which admits it by dispatch's own gate.

    Single stream: the split workspace is one buffer per runner, reused by
    every launch, so concurrent launches on different streams would share it.
    Use one runner per stream.
    """

    def __init__(self, result: DispatchResult, *, num_seg: int | None = None) -> None:
        from rocke import compile_kernel
        from rocke.runtime import KernelLauncher

        launch = mla_decode_launch(result)
        if num_seg is not None:
            launch = launch.with_num_seg(num_seg)
        result = launch.result
        self.launch = launch
        self.request = result.request
        self.candidate = result.candidate
        self.spec = launch.spec
        self.arch = self.request.arch
        self.grid = tuple(result.grid)
        self.block = tuple(result.block)
        art = compile_kernel(
            self.candidate.built(self.spec, self.arch),
            arch=self.arch,
            capture_ir_text=False,
        )
        # Diagnostic only: every split count shares the one split binary.
        self._latent = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=tuple(result.signature),
            cache_key=("mla_decode", self.arch, self.spec.kernel_name()),
        )
        self._combine = None
        if launch.combine_spec is not None:
            cart = compile_kernel(
                launch.build_combine(), arch=self.arch, capture_ir_text=False
            )
            self._combine = KernelLauncher(
                hsaco=cart.hsaco,
                kernel_name=cart.kernel_name,
                signature=launch.combine_signature(),
                cache_key=(
                    "mla_decode_combine",
                    self.arch,
                    launch.combine_spec.kernel_name(),
                ),
            )
        self._workspace: tuple[str, tuple[torch.Tensor, torch.Tensor]] | None = None

    @property
    def kernel_names(self) -> tuple[str, ...]:
        return self.launch.kernel_names()

    def workspace(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """The split kernel's fp32 ``(part_acc, part_ml)`` for the dispatched
        ``total_q``, allocated on first use and reused by every launch."""
        import torch

        if self._workspace is None or self._workspace[0] != str(device):
            shapes = self.launch.workspace_shapes()
            self._workspace = (
                str(device),
                tuple(
                    torch.empty(shapes[name], dtype=torch.float32, device=device)
                    for name in ("part_acc_ptr", "part_ml_ptr")
                ),
            )
        return self._workspace[1]

    def _check_inputs(
        self,
        qa: torch.Tensor,
        c_kv: torch.Tensor,
        k_rope: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        block_table: torch.Tensor,
        out: torch.Tensor | None,
        lse: torch.Tensor | None,
    ) -> None:
        """Tie the caller's tensors to the dispatched request before launch.

        The grid, ``num_seqs`` and the kernel's indexing all come from the
        request, and the kernel indexes every tensor by flat offset; a tensor
        that disagrees would be read or written out of bounds on the device
        rather than fail here.
        """
        import torch

        req, spec = self.request, self.spec
        pages = max(1, -(-int(req.max_seqlen_k) // spec.page_block_size))
        expect = {
            "qa": ((req.total_q, spec.num_heads, spec.qa_cols), torch.bfloat16, qa),
            "cu_seqlens_q": ((req.num_seqs + 1,), torch.int32, cu_seqlens_q),
            "cu_seqlens_k": ((req.num_seqs + 1,), torch.int32, cu_seqlens_k),
        }
        if out is not None:
            expect["out"] = (
                (req.total_q, spec.num_heads, spec.r_kv),
                torch.bfloat16,
                out,
            )
        if lse is not None:
            expect["lse"] = ((req.total_q, spec.num_heads), torch.float32, lse)
        for name, (shape, dtype, t) in expect.items():
            if tuple(t.shape) != shape:
                raise ValueError(
                    f"{name} has shape {tuple(t.shape)}, the dispatched request "
                    f"needs {shape}; dispatch again for a new batch"
                )
            if t.dtype != dtype:
                raise ValueError(f"{name} must be {dtype}, got {t.dtype}")
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        for name, t, width in (
            ("c_kv", c_kv, spec.r_kv),
            ("k_rope", k_rope, spec.d_rope),
        ):
            if t.dim() != 3 or tuple(t.shape[1:]) != (spec.page_block_size, width):
                raise ValueError(
                    f"{name} must be paged [pages, {spec.page_block_size}, {width}], "
                    f"got {tuple(t.shape)}"
                )
            if t.dtype != torch.bfloat16:
                raise ValueError(f"{name} must be torch.bfloat16, got {t.dtype}")
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        for name, t in (
            ("c_kv", c_kv),
            ("k_rope", k_rope),
            ("cu_seqlens_q", cu_seqlens_q),
            ("cu_seqlens_k", cu_seqlens_k),
            ("block_table", block_table),
            ("out", out),
            ("lse", lse),
        ):
            if t is not None and t.device != qa.device:
                raise ValueError(f"{name} is on {t.device}; qa is on {qa.device}")
        if block_table.dtype != torch.int32:
            raise ValueError(
                f"block_table must be torch.int32, got {block_table.dtype}"
            )
        if block_table.dim() != 2 or block_table.stride(1) != 1:
            raise ValueError("block_table must be 2-D with unit column stride")
        if block_table.shape[0] < req.num_seqs:
            raise ValueError(
                f"block_table has {block_table.shape[0]} rows < num_seqs={req.num_seqs}"
            )
        # The kernel reads slot 0 of every row, even for an empty sequence, and
        # up to the last page of max_seqlen_k.
        if block_table.shape[1] < pages:
            raise ValueError(
                f"block_table has {block_table.shape[1]} columns; max_seqlen_k="
                f"{req.max_seqlen_k} needs {pages}"
            )

    def check_contents(
        self,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        block_table: torch.Tensor | None = None,
        *,
        num_pages: int | None = None,
    ) -> None:
        """Check the offsets and page ids against the request (host sync).

        The request's ``total_q``, ``max_seqlen_q`` / ``max_seqlen_k`` and the
        cache's page count are preconditions the kernel cannot check: a query
        offset past ``total_q`` writes out of bounds, a second query in a
        sequence is left unwritten, a key range past ``max_seqlen_k`` reads
        past the block-table width :meth:`latent` checks, and a page id
        outside the cache reads out of bounds. Slots past a sequence's last
        page are not page ids and are not checked. This reads the arrays back,
        so it is opt-in.
        """
        req = self.request
        cu_q, cu_k = cu_seqlens_q.cpu(), cu_seqlens_k.cpu()
        q_lens, k_lens = cu_q[1:] - cu_q[:-1], cu_k[1:] - cu_k[:-1]
        if int(q_lens.min()) < 0 or int(k_lens.min()) < 0:
            raise ValueError("cu_seqlens arrays must be non-decreasing")
        if int(cu_q[0]) < 0 or int(cu_q[-1]) > req.total_q:
            raise ValueError(
                f"cu_seqlens_q spans [{int(cu_q[0])}, {int(cu_q[-1])}); the "
                f"dispatched request has total_q={req.total_q} output rows"
            )
        if int(q_lens.max()) > req.max_seqlen_q:
            raise ValueError(
                f"a sequence has {int(q_lens.max())} queries; the dispatched "
                f"request states max_seqlen_q={req.max_seqlen_q}"
            )
        if int(k_lens.max()) > req.max_seqlen_k:
            raise ValueError(
                f"a sequence has {int(k_lens.max())} keys; the dispatched "
                f"request states max_seqlen_k={req.max_seqlen_k}"
            )
        if block_table is not None and num_pages is not None:
            bt = block_table[: req.num_seqs].cpu()
            pbs = self.spec.page_block_size
            for s, k in enumerate(k_lens.tolist()):
                row = bt[s, : -(-k // pbs)]
                if row.numel() and (int(row.min()) < 0 or int(row.max()) >= num_pages):
                    raise ValueError(
                        f"block_table row {s} names a page outside [0, {num_pages})"
                    )

    def latent(
        self,
        *,
        qa: torch.Tensor,
        c_kv: torch.Tensor,
        k_rope: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        block_table: torch.Tensor,
        scale: float,
        out: torch.Tensor | None = None,
        lse: torch.Tensor | None = None,
        stream: int = 0,
        check_contents: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """[B] (+ [C]): ``(latent [total_q, H, r_kv] bf16, lse [total_q, H] f32)``.

        ``out`` / ``lse`` are allocated when not given. ``scale`` is the raw
        host scale, the same value the reference receives. ``check_contents``
        adds :meth:`check_contents` (a host sync) before the launch.
        """
        import torch
        from rocke.runtime import LaunchConfig

        self._check_inputs(
            qa, c_kv, k_rope, cu_seqlens_q, cu_seqlens_k, block_table, out, lse
        )
        if check_contents:
            self.check_contents(
                cu_seqlens_q, cu_seqlens_k, block_table, num_pages=int(c_kv.shape[0])
            )
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
        part_acc, part_ml = self.workspace(qa.device)
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
                grid=self.launch.combine_grid(),
                block=self.launch.combine_block(),
                stream=stream,
            ),
        )
        return out, lse

    def absorb(
        self,
        *,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        w_uk: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """[A] into ``out`` ``[total_q, H, r_kv + d_rope]``: per head,
        ``qa_nope[h] = q_nope[h] @ W_UK_nope[h]ᵀ``, then ``q_rope``."""
        import torch

        d_nope = q_nope.shape[-1]
        r_kv = self.spec.r_kv
        torch.bmm(
            q_nope.transpose(0, 1),
            w_uk[:, :, :d_nope].transpose(1, 2),
            out=out[:, :, :r_kv].transpose(0, 1),
        )
        out[:, :, r_kv:].copy_(q_rope)

    def project(
        self,
        *,
        latent: torch.Tensor,
        w_uk: torch.Tensor,
        d_nope: int,
        out: torch.Tensor,
    ) -> None:
        """[D] into ``out`` ``[total_q, H, d_v]``: per head,
        ``out[h] = latent[h] @ W_UV[h]``, ``W_UV = w_uk[:, :, d_nope:]``."""
        import torch

        torch.bmm(latent.transpose(0, 1), w_uk[:, :, d_nope:], out=out.transpose(0, 1))

    def forward(
        self,
        *,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        w_uk: torch.Tensor,
        c_kv: torch.Tensor,
        k_rope: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        block_table: torch.Tensor,
        scale: float,
        stream: int = 0,
        check_contents: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The whole decode op, [A] to [D]: ``(out [total_q, H, d_v], lse)``.

        ``w_uk`` is the fused per-head ``[H, r_kv, d_nope + d_v]`` weight, both
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
            check_contents=check_contents,
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

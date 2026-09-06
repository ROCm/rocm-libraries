# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch.varlen -- route **paged** variable-length attention onto hipDNN.

``F.scaled_dot_product_attention`` has no KV-cache argument, so the dense
:mod:`~hipdnn_torch.sdpa` override can never see a paged workload however it is
written. PyTorch exposes paged attention one module over, as
``torch.nn.attention.varlen.varlen_attn(..., block_table=, seqused_k=)``, which is
what this override patches (via :attr:`OpOverride.target_module`).

**The layout mapping is a pure re-description -- no data moves.** Both operands
reach hipDNN as strided views of the caller's own buffers:

  * ``Q`` arrives packed ``[total_q, H, D]``; hipDNN wants rank-4 BSHD, so
    ``q.unsqueeze(0).permute(0, 2, 1, 3)`` gives ``[1, H, total_q, D]`` with
    strides ``[T*H*D, D, H*D, 1]``.
  * ``K``/``V`` arrive as the paged container ``[pages, page_size, H, D]``; the
    kernel's baked layout is ``[pages, H, page_size, D]``, i.e. the same bytes
    under ``permute(0, 2, 1, 3)``.

Both were checked against the committed bundle
``integration-test-bundles/quick/SdpaFwd/paged/bf16/hd128_page64_gqa4/Small`` and
reproduce its dims *and* strides exactly, so this override submits the geometry the
tiled engine already accepts rather than a new one.

**Offsets vs lengths** is the only real arithmetic. ``cu_seq_q`` is cumulative
*offsets* (``num_seqs + 1`` entries); hipDNN's ``set_seq_len_q`` wants per-sequence
*lengths*, so we ``diff()``. ``seqused_k`` is already lengths and passes through to
``set_seq_len_kv``. Paged K/V carry no ragged offsets of their own -- the page table
*is* the per-sequence indirection -- which is the same distinction the cuDNN backend
draws in ``MHA.cpp``.

**The gate is ours, deliberately not cuDNN's.** ``_cudnn_rejection_reasons``
(torch's ``varlen.py``) refuses paged+causal outright and refuses GQA; the tiled
engine serves paged+GQA, so deferring to it would hand us far less than we can
serve. Anything genuinely unmapped raises :class:`NotApplicable` and falls back to
native, counted with a reason, exactly like every other override.
"""

import logging
import math

from .base import NotApplicable, OpOverride

_Q_UID, _K_UID, _V_UID, _O_UID = 1, 2, 3, 4
_PT_K_UID, _PT_V_UID, _SQ_UID, _SKV_UID = 5, 6, 7, 8

#: Head sizes and page sizes the gfx950 tiled pack bakes kernels for. Used only to
#: skip a doomed graph build cheaply -- the engine's matcher remains the authority,
#: and anything that slips past here is still declined by ``check_support``.
_HEAD_SIZES = (64, 128, 256)
_BLOCK_SIZES = (16, 32, 64)


class VarlenSdpaOverride(OpOverride):
    op_name = "varlen_attn"
    target_module = "torch.nn.attention.varlen"

    # -- gate ---------------------------------------------------------------
    def _gate(self, query, block_table, seqused_k, num_splits):
        """Our own applicability gate. Only refuses what is genuinely unmapped;
        every size question (GQA ratio, sequence lengths, causality) is described
        into the graph and left to hipDNN."""
        st = self.state
        if not query.is_cuda:
            return False, "query not on cuda"  # execute() needs a device pointer
        if query.dtype not in st.dtype_map:
            return False, f"dtype {self._tok(query.dtype)} not graph-mappable"
        if block_table is None:
            # Not a decline of the workload -- it is simply the dense varlen case,
            # which this override does not claim.
            return False, "no block_table (dense varlen, not paged)"
        if seqused_k is None:
            # The page table indexes pages; without per-sequence KV lengths there
            # is nothing to say how much of the last page is live.
            return False, "block_table without seqused_k"
        if num_splits is not None:
            return False, "num_splits not expressible in the graph"
        if query.dim() != 3:
            return False, f"query rank {query.dim()} != 3 (expected [total_q, H, D])"
        d = int(query.shape[-1])
        if d not in _HEAD_SIZES:
            return False, f"head_size {d} not in {_HEAD_SIZES}"
        return True, ""

    # -- graph --------------------------------------------------------------
    def _graph(
        self,
        q_spec,
        k_spec,
        v_spec,
        o_spec,
        pt_spec,
        sq_spec,
        skv_spec,
        scale,
        causal,
        window,
        max_seq_len_kv,
    ):
        st = self.state
        hipdnn = st.hipdnn
        g = hipdnn.Graph()
        g.set_io_data_type(st.dtype_map[q_spec[2]])
        g.set_compute_data_type(hipdnn.DataType.FLOAT)

        def _t(name, spec, uid):
            dims, strides, dtype = spec
            return g.tensor(
                hipdnn.Tensor()
                .set_name(name)
                .set_dim(list(dims))
                .set_stride(list(strides))
                .set_data_type(st.dtype_map[dtype])
                .set_uid(uid)
            )

        q_t = _t("Q", q_spec, _Q_UID)
        k_t = _t("K", k_spec, _K_UID)
        v_t = _t("V", v_spec, _V_UID)

        attrs = hipdnn.SdpaAttributes()
        # Newer frontends spell this set_attn_scale, older ones
        # set_attn_scale_value; same float signature. The tiled matcher requires
        # the value to be PRESENT -- it infers no 1/sqrt(D) default -- so this is
        # always set, never skipped.
        _set_scale = (
            getattr(attrs, "set_attn_scale", None) or attrs.set_attn_scale_value
        )
        _set_scale(float(scale))
        attrs.set_causal_mask(bool(causal))

        # Paged indirection. One torch block_table drives both K and V: they share
        # a page geometry, and the matcher requires both tensors to be present and
        # shape-identical.
        attrs.set_paged_attention_k_table(_t("PAGE_TABLE_K", pt_spec, _PT_K_UID))
        attrs.set_paged_attention_v_table(_t("PAGE_TABLE_V", pt_spec, _PT_V_UID))
        attrs.set_paged_attention_max_seq_len_kv(int(max_seq_len_kv))

        # Lengths, not offsets (see module docstring).
        attrs.set_seq_len_q(_t("SEQ_LEN_Q", sq_spec, _SQ_UID))
        attrs.set_seq_len_kv(_t("SEQ_LEN_KV", skv_spec, _SKV_UID))

        # Sliding window. torch's (left, right) maps onto the diagonal band; the
        # committed paged bundle carries (-1, 0) for the unbounded case, so an
        # absent window is expressed exactly as that bundle does.
        left, right = window
        if hasattr(attrs, "set_diagonal_band_left_bound"):
            attrs.set_diagonal_band_left_bound(int(left))
            attrs.set_diagonal_band_right_bound(int(right))

        o_t = g.sdpa(q_t, k_t, v_t, attrs)[0]  # [o, stats]; stats is None
        o_dims, o_str, o_dtype = o_spec
        o_t.set_dim(list(o_dims))
        o_t.set_stride(list(o_str))
        o_t.set_data_type(st.dtype_map[o_dtype])
        o_t.set_uid(_O_UID)
        o_t.set_output(True)
        return g

    # -- shared paged entry point -------------------------------------------
    def run_paged(
        self,
        query,
        key,
        value,
        cu_seq_q,
        max_k,
        *,
        scale=None,
        window=(-1, -1),
        seqused_k=None,
        block_table=None,
        num_splits=None,
        census_key=None,
    ):
        """Map a paged varlen call onto hipDNN and execute it.

        Returns the packed ``[total_q, H, Dv]`` output, or **None** when the call
        is not served -- in which case the reason has already been counted and the
        caller should fall back to its own native path. Returning None rather than
        raising keeps the two routes' fallbacks identical: the ATen route has a
        different native call than the wrapper route, so only the caller can make
        it.

        Shared by :meth:`_call` (the ``varlen_attn`` monkeypatch) and
        :class:`~hipdnn_torch.varlen_aten.AtenVarlenRoute` (the dispatcher
        registration) so the mapping cannot drift between them."""
        torch = self.state.torch
        try:
            total_q, hq, d = (int(x) for x in query.shape)
            hkv = int(key.shape[-2])
        except Exception:  # noqa: BLE001 -- exotic call site, let native have it
            return None
        if census_key is None:
            census_key = (
                f"Tq={total_q},Hq={hq},Hkv={hkv},D={d},dtype={self._tok(query.dtype)}"
            )

        ok, reason = self._gate(query, block_table, seqused_k, num_splits)
        if not ok:
            self.note_native(census_key, reason)
            return None

        try:
            window_size = tuple(window)
            eff_scale = float(scale) if scale is not None else 1.0 / math.sqrt(d)

            # --- lengths, not offsets ---------------------------------------
            # cu_seq_q is cumulative offsets with num_seqs+1 entries; hipDNN wants
            # per-sequence lengths. seqused_k is already lengths.
            seq_len_q = torch.diff(cu_seq_q).to(torch.int32)
            seq_len_kv = seqused_k.to(torch.int32)
            if seq_len_q.numel() != seq_len_kv.numel():
                raise NotApplicable(
                    f"num_seqs mismatch: q={seq_len_q.numel()} kv={seq_len_kv.numel()}"
                )
            seq_len_q = seq_len_q.contiguous()
            seq_len_kv = seq_len_kv.contiguous()
            page_table = block_table.to(torch.int32).contiguous()

            # --- zero-copy re-description ------------------------------------
            # Q [T,H,D] -> [1,H,T,D]; paged K/V [pages,page,H,D] -> [pages,H,page,D].
            qv = query.unsqueeze(0).permute(0, 2, 1, 3)
            kv_ = key.permute(0, 2, 1, 3)
            vv = value.permute(0, 2, 1, 3)

            page_size = int(key.shape[1])
            if page_size not in _BLOCK_SIZES:
                raise NotApplicable(f"page_size {page_size} not in {_BLOCK_SIZES}")

            # max_seq_len_kv must satisfy 0 < value <= blocks_per_seq * page_size.
            blocks_per_seq = int(page_table.shape[1])
            cap = blocks_per_seq * page_size
            max_seq_len_kv = min(int(max_k), cap)
            if max_seq_len_kv <= 0:
                raise NotApplicable(f"max_seq_len_kv {max_seq_len_kv} not positive")

            # Output mirrors native's contract: packed [total_q, H, Dv], contiguous.
            dv = int(value.shape[-1])
            o = torch.empty((total_q, hq, dv), dtype=query.dtype, device=query.device)
            ov = o.unsqueeze(0).permute(0, 2, 1, 3)

            left, right = (int(window_size[0]), int(window_size[1]))
            causal = window_size == (-1, 0) or window_size == [-1, 0]

            def spec(t):
                return (tuple(t.shape), tuple(t.stride()), t.dtype)

            q_spec, k_spec = spec(qv), spec(kv_)
            v_spec, o_spec = spec(vv), spec(ov)
            pt_spec, sq_spec, skv_spec = (
                spec(page_table),
                spec(seq_len_q),
                spec(seq_len_kv),
            )

            entry = self._cached_graph(
                (
                    q_spec,
                    k_spec,
                    v_spec,
                    o_spec,
                    pt_spec,
                    sq_spec,
                    skv_spec,
                    eff_scale,
                    bool(causal),
                    (left, right),
                    max_seq_len_kv,
                ),
                lambda: self._graph(
                    q_spec,
                    k_spec,
                    v_spec,
                    o_spec,
                    pt_spec,
                    sq_spec,
                    skv_spec,
                    eff_scale,
                    bool(causal),
                    (left, right),
                    max_seq_len_kv,
                ),
                f"paged {list(qv.shape)}x{list(kv_.shape)} page={page_size} "
                f"gqa={hq}/{hkv} {query.dtype}",
            )
            self._execute(
                entry,
                {
                    _Q_UID: query.data_ptr(),
                    _K_UID: key.data_ptr(),
                    _V_UID: value.data_ptr(),
                    _O_UID: o.data_ptr(),
                    _PT_K_UID: page_table.data_ptr(),
                    _PT_V_UID: page_table.data_ptr(),
                    _SQ_UID: seq_len_q.data_ptr(),
                    _SKV_UID: seq_len_kv.data_ptr(),
                },
                query.device,
            )
            self.note_aot(census_key, paged=1)
            return o
        except NotApplicable as na:
            self.note_native(census_key, str(na))
            return None
        except Exception as ex:  # noqa: BLE001 -- never break the model
            self.note_native(
                census_key,
                f"exception: {type(ex).__name__}: {ex}",
                level=logging.WARNING,
            )
            return None

    # -- call ---------------------------------------------------------------
    def _call(
        self,
        real,
        query,
        key,
        value,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        *,
        return_aux=None,
        scale=None,
        window_size=(-1, -1),
        seqused_k=None,
        block_table=None,
        num_splits=None,
    ):
        def _native():
            return real(
                query,
                key,
                value,
                cu_seq_q,
                cu_seq_k,
                max_q,
                max_k,
                return_aux=return_aux,
                scale=scale,
                window_size=window_size,
                seqused_k=seqused_k,
                block_table=block_table,
                num_splits=num_splits,
            )

        # return_aux asks for the log-sumexp alongside the output. The graph is
        # built without a stats output (the matcher declines one), so rather than
        # return a silently wrong aux value, decline the call.
        if return_aux is not None and getattr(return_aux, "lse", False):
            self.note_native(
                self._census_key(query, key),
                "return_aux.lse not served by the graph",
            )
            return _native()

        out = self.run_paged(
            query,
            key,
            value,
            cu_seq_q,
            max_k,
            scale=scale,
            window=window_size,
            seqused_k=seqused_k,
            block_table=block_table,
            num_splits=num_splits,
        )
        return _native() if out is None else out

    def _census_key(self, query, key):
        try:
            total_q, hq, d = (int(x) for x in query.shape)
            return (
                f"Tq={total_q},Hq={hq},Hkv={int(key.shape[-2])},D={d},"
                f"dtype={self._tok(query.dtype)}"
            )
        except Exception:  # noqa: BLE001 -- census must never break dispatch
            return "varlen"

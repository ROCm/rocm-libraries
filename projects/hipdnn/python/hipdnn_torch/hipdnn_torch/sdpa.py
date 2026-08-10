# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch.sdpa -- route ``F.scaled_dot_product_attention`` (dense, non-causal,
unmasked attention) onto the hipDNN engine's fused fwd attention.

The tensors are BHSD (``[B, H, S, D]``). The default gfx1151 kernel *bakes* the
head count and head dim (:data:`BAKED_HEADS` / :data:`BAKED_HEAD_DIM`) and takes
seqlen at runtime, so one binary per dtype serves both self- and cross-attention.
The graph carries the attention scale (``scale`` or ``1/sqrt(D)``); the adapter
converts it to the kernel's base-2 ``scale_log2`` in C++ (the #1 correctness
gotcha, handled below the frontend).

Gate: cuda f16/bf16, rank-4 BHSD, ``attn_mask=None``, ``dropout_p==0``, not causal,
no GQA, ``H==BAKED_HEADS``, ``D==BAKED_HEAD_DIM``, ``B==1`` (batch folds into grid
z), and ``S_q``/``S_kv`` multiples of :data:`TILE`. Anything else falls back to
native and is logged.

.. note::
   :data:`BAKED_HEADS` / :data:`BAKED_HEAD_DIM` / :data:`TILE` MUST match the
   shipped kernel family. They are constants here; making them catalog-metadata
   driven (so a differently-baked family is picked up automatically) is a
   documented follow-up.
"""

import logging
import math

from .base import NotApplicable, OpOverride

#: The kernel bakes these; only seqlen is runtime. Must match the shipped family.
BAKED_HEADS = 32
BAKED_HEAD_DIM = 64
TILE = 16  # S_q / S_kv must be multiples of the kernel's tile granularity

_Q_UID, _K_UID, _V_UID, _O_UID = 1, 2, 3, 4


class SdpaOverride(OpOverride):
    op_name = "scaled_dot_product_attention"

    def _gate(self, query, key, value, attn_mask, dropout_p, is_causal, enable_gqa):
        torch = self.state.torch
        if not query.is_cuda:
            return False, "query not on cuda"
        if query.dtype not in (torch.float16, torch.bfloat16):
            return False, f"dtype {self._tok(query.dtype)} (need f16/bf16)"
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            return False, "not rank-4 BHSD"
        if attn_mask is not None:
            return False, "attn_mask unsupported"
        if float(dropout_p) != 0.0:
            return False, "dropout_p != 0"
        if is_causal:
            return False, "is_causal unsupported"
        if enable_gqa:
            return False, "GQA unsupported"
        b, h, _, d = query.shape
        if b != 1:
            return False, f"B={b} (only B==1; batch folds into grid z)"
        if h != BAKED_HEADS or int(key.shape[1]) != BAKED_HEADS or int(value.shape[1]) != BAKED_HEADS:
            return False, f"H!={BAKED_HEADS} (kernel bakes H={BAKED_HEADS})"
        if d != BAKED_HEAD_DIM or int(key.shape[-1]) != BAKED_HEAD_DIM or int(value.shape[-1]) != BAKED_HEAD_DIM:
            return False, f"D!={BAKED_HEAD_DIM} (kernel bakes D={BAKED_HEAD_DIM})"
        sq = int(query.shape[-2])
        skv = int(key.shape[-2])
        if int(value.shape[-2]) != skv:
            return False, "K/V disagree on S_kv"
        if sq % TILE or skv % TILE:
            return False, f"S not %{TILE} (Sq={sq},Skv={skv})"
        return True, ""

    def _graph(self, b, h, sq, skv, d, scale, dtype):
        st = self.state
        hipdnn = st.hipdnn
        hf = st.dtype_map[dtype]
        g = hipdnn.Graph()
        g.set_io_data_type(hf)
        g.set_compute_data_type(hipdnn.DataType.FLOAT)

        def _bhsd(name, s, uid):
            return g.tensor(
                hipdnn.Tensor().set_name(name).set_dim([b, h, s, d])
                .set_stride([h * s * d, s * d, d, 1]).set_data_type(hf).set_uid(uid)
            )

        q_t = _bhsd("Q", sq, _Q_UID)
        k_t = _bhsd("K", skv, _K_UID)
        v_t = _bhsd("V", skv, _V_UID)

        attrs = hipdnn.SdpaAttributes()
        attrs.set_attn_scale_value(float(scale))
        # Non-causal, no mask -- leave every other feature unset (mask=none).

        o_t = g.sdpa(q_t, k_t, v_t, attrs)[0]  # [o, stats]; stats is None
        o_t.set_dim([b, h, sq, d])
        o_t.set_stride([h * sq * d, sq * d, d, 1])
        o_t.set_data_type(hf)
        o_t.set_uid(_O_UID)
        return g

    def _call(self, real, query, key, value, attn_mask=None, dropout_p=0.0,
              is_causal=False, scale=None, enable_gqa=False):
        def _native():
            return real(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                        is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)

        torch = self.state.torch
        # Census dims are best-effort; guard the shape reads for exotic call sites.
        try:
            sq = int(query.shape[-2])
            skv = int(key.shape[-2])
        except Exception:  # noqa: BLE001
            return _native()
        census_key = f"Sq={sq},Skv={skv},dtype={self._tok(query.dtype)}"

        ok, reason = self._gate(query, key, value, attn_mask, dropout_p, is_causal,
                                enable_gqa)
        if not ok:
            self.note_native(census_key, reason)
            return _native()

        eff_scale = float(scale) if scale is not None else 1.0 / math.sqrt(BAKED_HEAD_DIM)
        try:
            # Contiguous BHSD so the baked graph strides (token=D, head=S*D) hold.
            q = query.contiguous()
            k = key.contiguous()
            v = value.contiguous()
            b, h, _, d = q.shape
            entry = self._cached_graph(
                (b, h, sq, skv, d, eff_scale, q.dtype),
                lambda: self._graph(b, h, sq, skv, d, eff_scale, q.dtype),
                f"[B={b},H={h},Sq={sq},Skv={skv},D={d}] {q.dtype}",
            )
            o = torch.empty(b, h, sq, d, dtype=q.dtype, device=q.device)
            self._execute(entry, {
                _Q_UID: q.data_ptr(), _K_UID: k.data_ptr(),
                _V_UID: v.data_ptr(), _O_UID: o.data_ptr(),
            }, q.device)
            self.note_aot(census_key)
            return o
        except NotApplicable as na:  # engine can't serve this shape -> native
            self.note_native(census_key, str(na))
            return _native()
        except Exception as ex:  # noqa: BLE001 -- any failure -> native, never break the model
            self.note_native(census_key, f"exception: {type(ex).__name__}: {ex}",
                             level=logging.WARNING)
            return _native()

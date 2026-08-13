# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch.sdpa -- route ``F.scaled_dot_product_attention`` onto hipDNN's fused
forward attention.

**Passthrough, not pre-filter.** The override does not pre-judge which shapes an
engine can serve: it builds the SDPA graph from the *actual* Q/K/V tensor shapes
(``[B, H, S, D]`` -- any B, any head count, any head dim, any seqlen, and GQA when
``H_kv < H``) and lets hipDNN's ``check_support`` decide. A shape no loaded engine
claims (e.g. rocKE bakes D=64 and wants S%16==0) simply returns ``NotApplicable``
and falls back to native -- the *engine* declines it, not a hard-coded Python gate.

The only things the override still catches are the ones it **cannot represent** in
the hipDNN graph (per the frontend ``SdpaAttributes`` surface: ``attn_scale_value``,
``causal_mask``, ``dropout_probability``):

  * ``attn_mask`` -- no mask-tensor binding, so an explicit additive/boolean mask
    cannot be expressed; fall back.
  * ``dropout_p != 0`` -- hipDNN's dropout has its own RNG, which cannot reproduce
    torch's exact inference-time mask; fall back (inference dropout is normally 0).

``is_causal`` **is** representable (``set_causal_mask``) so it is built into the
graph and passed through; GQA is representable as ``H_kv < H`` operands and is built
directly. The graph carries the attention scale (``scale`` or ``1/sqrt(D)`` from the
*actual* D); the adapter converts it to the kernel's base-2 ``scale_log2`` in C++.
"""

import logging
import math

from .base import NotApplicable, OpOverride

_Q_UID, _K_UID, _V_UID, _O_UID = 1, 2, 3, 4


class SdpaOverride(OpOverride):
    op_name = "scaled_dot_product_attention"

    def _gate(self, query, key, value, attn_mask, dropout_p):
        # Only structural facts (needed to build/execute the graph at all) and
        # features hipDNN's SDPA graph cannot represent. Everything about size --
        # B, H, D, S, GQA, causal -- is built into the graph and left to
        # check_support; nothing here pre-judges engine capability.
        torch = self.state.torch
        if not query.is_cuda:
            return False, "query not on cuda"  # execute() needs a device pointer
        if query.dtype not in self.state.dtype_map:
            return False, f"dtype {self._tok(query.dtype)} not graph-mappable"
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            return False, "not rank-4 BHSD"  # builder unpacks [B,H,S,D]
        if int(key.shape[-2]) != int(value.shape[-2]):
            return False, "K/V disagree on S_kv"  # graph carries one S_kv
        if int(key.shape[1]) != int(value.shape[1]):
            return False, "K/V disagree on head count"
        if int(query.shape[-1]) != int(key.shape[-1]) or int(query.shape[-1]) != int(
            value.shape[-1]
        ):
            return False, "Q/K/V disagree on D"
        # --- not representable in the hipDNN SDPA graph -> catch here ---
        if attn_mask is not None:
            return False, "attn_mask not representable in graph"
        if float(dropout_p) != 0.0:
            return False, "dropout_p != 0 not representable (RNG can't match torch)"
        return True, ""

    def _graph(self, b, hq, hkv, sq, skv, d, scale, causal, dtype):
        st = self.state
        hipdnn = st.hipdnn
        hf = st.dtype_map[dtype]
        g = hipdnn.Graph()
        g.set_io_data_type(hf)
        g.set_compute_data_type(hipdnn.DataType.FLOAT)

        def _bhsd(name, h, s, uid):
            return g.tensor(
                hipdnn.Tensor()
                .set_name(name)
                .set_dim([b, h, s, d])
                .set_stride([h * s * d, s * d, d, 1])
                .set_data_type(hf)
                .set_uid(uid)
            )

        q_t = _bhsd("Q", hq, sq, _Q_UID)
        k_t = _bhsd("K", hkv, skv, _K_UID)  # hkv<hq -> GQA, built directly
        v_t = _bhsd("V", hkv, skv, _V_UID)

        attrs = hipdnn.SdpaAttributes()
        attrs.set_attn_scale_value(float(scale))
        attrs.set_causal_mask(bool(causal))  # is_causal expressed, not gated

        o_t = g.sdpa(q_t, k_t, v_t, attrs)[0]  # [o, stats]; stats is None
        o_t.set_dim([b, hq, sq, d])
        o_t.set_stride([hq * sq * d, sq * d, d, 1])
        o_t.set_data_type(hf)
        o_t.set_uid(_O_UID)
        o_t.set_output(True)  # terminal output must be non-virtual (MIOpen requires it)
        return g

    def _call(
        self,
        real,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        def _native():
            return real(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

        torch = self.state.torch
        # Census dims are best-effort; guard the shape reads for exotic call sites.
        try:
            sq = int(query.shape[-2])
            skv = int(key.shape[-2])
            d = int(query.shape[-1])
        except Exception:  # noqa: BLE001
            return _native()
        census_key = f"Sq={sq},Skv={skv},D={d},dtype={self._tok(query.dtype)}"

        ok, reason = self._gate(query, key, value, attn_mask, dropout_p)
        if not ok:
            self.note_native(census_key, reason)
            return _native()

        eff_scale = float(scale) if scale is not None else 1.0 / math.sqrt(d)
        try:
            # Contiguous BHSD so the graph strides (token=D, head=S*D) hold.
            q = query.contiguous()
            k = key.contiguous()
            v = value.contiguous()
            b, hq, _, _ = q.shape
            hkv = int(k.shape[1])
            entry = self._cached_graph(
                (b, hq, hkv, sq, skv, d, eff_scale, bool(is_causal), q.dtype),
                lambda: self._graph(
                    b, hq, hkv, sq, skv, d, eff_scale, bool(is_causal), q.dtype
                ),
                f"[B={b},Hq={hq},Hkv={hkv},Sq={sq},Skv={skv},D={d}]"
                f"{' causal' if is_causal else ''} {q.dtype}",
            )
            o = torch.empty(b, hq, sq, d, dtype=q.dtype, device=q.device)
            self._execute(
                entry,
                {
                    _Q_UID: q.data_ptr(),
                    _K_UID: k.data_ptr(),
                    _V_UID: v.data_ptr(),
                    _O_UID: o.data_ptr(),
                },
                q.device,
            )
            self.note_aot(census_key)
            return o
        except NotApplicable as na:  # no engine claims this shape -> native
            self.note_native(census_key, str(na))
            return _native()
        except (
            Exception
        ) as ex:  # noqa: BLE001 -- any failure -> native, never break the model
            self.note_native(
                census_key,
                f"exception: {type(ex).__name__}: {ex}",
                level=logging.WARNING,
            )
            return _native()

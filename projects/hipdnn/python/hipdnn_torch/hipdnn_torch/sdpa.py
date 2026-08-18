# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch.sdpa -- route ``F.scaled_dot_product_attention`` onto hipDNN's fused
forward attention.

**Pure passthrough.** The override builds the SDPA graph from Q/K/V's *actual* dims
+ strides -- whatever layout they arrive in (Q/K/V are almost always permuted views
of a packed QKV buffer, never contiguous BHSD) -- and submits it. It imposes no
layout (no ``.contiguous()``), pre-judges nothing about which shapes an engine can
serve, and makes no structural correctness check (rank, GQA head counts, D
agreement): hipDNN's ``check_support`` / ``execute`` decides, and a shape no loaded
engine claims falls back to native and is counted. The output is allocated
contiguous with Q's ``[B, Hq, Sq, Dv]`` shape (native's output contract) and its
actual strides are fed to the graph.

Every torch SDPA parameter is translated into the graph:

  * **scale** (or ``1/sqrt(Dqk)`` from the *actual* Dqk) → ``set_attn_scale``.
  * **is_causal** → ``set_causal_mask``.
  * **attn_mask** → ``set_bias`` (an additive score bias). A float mask passes
    through as its own additive tensor at its actual dims/strides; a bool mask is
    materialized to a 0/−inf additive tensor (``True`` keeps, ``False`` masks) --
    an intrinsic, documented materialization, the only way an additive-bias node
    can carry a boolean mask.
  * **dropout_p != 0** → ``set_dropout_probability``. hipDNN's dropout RNG differs
    from torch's, so numerical parity is only defined at ``p == 0``; a non-zero p is
    still submitted and whatever hipDNN does (serve or decline) is what the census
    records. (``p == 0`` never touches the setter, keeping the common path's graph
    identical.)

GQA (``H_kv < H_q``) needs no special handling -- it is simply K/V operands with a
smaller head count, described directly.
"""

import logging
import math
import os

from .base import NotApplicable, OpOverride

_Q_UID, _K_UID, _V_UID, _O_UID, _BIAS_UID = 1, 2, 3, 4, 5


class SdpaOverride(OpOverride):
    op_name = "scaled_dot_product_attention"

    def _gate(self, query):
        # The only two legitimate stops: we need a device pointer to execute
        # against, and a dtype the graph builder can express. Everything about
        # size -- B, H, D, S, GQA, causal, mask, dropout -- is built into the graph
        # and left to hipDNN.
        if not query.is_cuda:
            return False, "query not on cuda"  # execute() needs a device pointer
        if query.dtype not in self.state.dtype_map:
            return False, f"dtype {self._tok(query.dtype)} not graph-mappable"
        return True, ""

    def _graph(
        self,
        q_dims,
        q_str,
        q_dtype,
        k_dims,
        k_str,
        k_dtype,
        v_dims,
        v_str,
        v_dtype,
        o_dims,
        o_str,
        o_dtype,
        scale,
        causal,
        dropout_p,
        bias_spec,
    ):
        st = self.state
        hipdnn = st.hipdnn
        hq = st.dtype_map[q_dtype]
        g = hipdnn.Graph()
        g.set_io_data_type(hq)
        g.set_compute_data_type(hipdnn.DataType.FLOAT)

        def _t(name, dims, strides, dtype, uid):
            return g.tensor(
                hipdnn.Tensor()
                .set_name(name)
                .set_dim(list(dims))
                .set_stride(list(strides))
                .set_data_type(st.dtype_map[dtype])
                .set_uid(uid)
            )

        q_t = _t("Q", q_dims, q_str, q_dtype, _Q_UID)
        k_t = _t("K", k_dims, k_str, k_dtype, _K_UID)
        v_t = _t("V", v_dims, v_str, v_dtype, _V_UID)

        attrs = hipdnn.SdpaAttributes()
        # Scalar softmax scale. The binding is named ``set_attn_scale`` in newer
        # frontends and ``set_attn_scale_value`` in older ones (same float signature);
        # accept whichever this build exposes -- version skew, not a capability gap.
        _set_scale = (
            getattr(attrs, "set_attn_scale", None) or attrs.set_attn_scale_value
        )
        _set_scale(float(scale))
        attrs.set_causal_mask(bool(causal))
        if dropout_p:  # 0 -> leave the setter untouched (identical to no dropout)
            attrs.set_dropout_probability(float(dropout_p))
        if bias_spec is not None:
            b_dims, b_str, b_dtype = bias_spec
            attrs.set_bias(_t("bias", b_dims, b_str, b_dtype, _BIAS_UID))

        o_t = g.sdpa(q_t, k_t, v_t, attrs)[0]  # [o, stats]; stats is None
        o_t.set_dim(list(o_dims))
        o_t.set_stride(list(o_str))
        o_t.set_data_type(st.dtype_map[o_dtype])
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

        ok, reason = self._gate(query)
        if not ok:
            self.note_native(census_key, reason)
            return _native()

        eff_scale = float(scale) if scale is not None else 1.0 / math.sqrt(d)
        try:
            # attn_mask -> additive bias. A bool mask (True=keep) is materialized to
            # a 0/-inf additive tensor; a float mask is used as-is. We own the
            # materialized tensor, so its strides are known.
            bias = None
            bias_spec = None
            extras = {}
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    bias = torch.zeros(
                        attn_mask.shape, dtype=query.dtype, device=query.device
                    )
                    bias.masked_fill_(~attn_mask, float("-inf"))
                    extras["mask_bool"] = 1
                else:
                    bias = attn_mask
                    extras["mask_additive"] = 1
                bias_spec = (
                    tuple(bias.shape),
                    tuple(bias.stride()),
                    bias.dtype,
                )

            # Output mirrors native's contract: contiguous, Q's batch/heads/seq with
            # V's head dim (Dv). We own it, so its strides are known.
            dv = int(value.shape[-1])
            # Default: native's contract -- contiguous [B, Hq, Sq, Dv] (BHSD).
            #
            # HIPDNN_TORCH_SDPA_OUT_LAYOUT=match_q instead allocates O in Q's own
            # packed layout. Q/K/V are normally [B, S, H, D] buffers viewed through
            # .transpose(1, 2), i.e. packed BSHD, and a kernel with no stride
            # arguments (rocKE's gfx950 attention_dense) reads AND writes that
            # layout -- it can serve such a Q/K/V but not a contiguous BHSD O, so a
            # mixed graph is declined. Same shape, dtype and values either way; only
            # the strides differ, and the usual `.transpose(1,2).reshape(...)`
            # consumer is indifferent.
            if os.environ.get("HIPDNN_TORCH_SDPA_OUT_LAYOUT") == "match_q":
                b, hq, sq = int(query.shape[0]), int(query.shape[1]), int(query.shape[2])
                o = torch.empty(
                    (b, sq, hq, dv), dtype=query.dtype, device=query.device
                ).transpose(1, 2)
            else:
                o = torch.empty(
                    (*query.shape[:-1], dv), dtype=query.dtype, device=query.device
                )

            entry = self._cached_graph(
                (
                    tuple(query.shape),
                    tuple(query.stride()),
                    query.dtype,
                    tuple(key.shape),
                    tuple(key.stride()),
                    key.dtype,
                    tuple(value.shape),
                    tuple(value.stride()),
                    value.dtype,
                    tuple(o.stride()),
                    eff_scale,
                    bool(is_causal),
                    float(dropout_p),
                    bias_spec,
                ),
                lambda: self._graph(
                    list(query.shape),
                    list(query.stride()),
                    query.dtype,
                    list(key.shape),
                    list(key.stride()),
                    key.dtype,
                    list(value.shape),
                    list(value.stride()),
                    value.dtype,
                    list(o.shape),
                    list(o.stride()),
                    query.dtype,
                    eff_scale,
                    bool(is_causal),
                    float(dropout_p),
                    bias_spec,
                ),
                f"{list(query.shape)}x{list(key.shape)}x{list(value.shape)}"
                f"{' causal' if is_causal else ''} {query.dtype}",
            )
            variant = {
                _Q_UID: query.data_ptr(),
                _K_UID: key.data_ptr(),
                _V_UID: value.data_ptr(),
                _O_UID: o.data_ptr(),
            }
            if bias is not None:
                variant[_BIAS_UID] = bias.data_ptr()
            self._execute(entry, variant, query.device)
            self.note_aot(census_key, **extras)
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

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""``hipdnn-vllm probe --model <hf-id>``.

Builds every hipDNN graph signature a model would need on this machine, at a
few representative shapes, and prints the route each one takes. Nothing is
hard-coded about engines: a signature is "hipdnn" exactly when the installed
providers build a plan for it, and the engine that would run it is printed.

Two columns per attention signature:

* ``route``: what the HIPDNN backend does with the call today (the same
  :func:`hipdnn_vllm.attention.routing.select` the backend uses);
* ``graph``: whether hipDNN accepts the graph the path builds. For decoder
  attention these are the backend's own P2a (paged uniform decode) and P2b
  (ragged + paged) graphs, built for the scheduler bounds given on the command
  line (``--max-num-seqs``, ``--max-num-batched-tokens``, ``--max-kv``), exactly
  as the backend builds them at a layer's first call.

Models whose vLLM attention layers pass ``chunk_lookback`` (``layer_types``
containing ``chunked_attention``, e.g. Rnj1) need the local vLLM patch
``patches/vllm/0001`` (unpatched vLLM asserts the backend name is
``TRITON_ATTN``); their chunked layers are then served by the TRITON_ATTN
fallback with ``UNSUPPORTED_FEATURE:chunk_lookback``. The probe prints this
requirement.
"""

from __future__ import annotations

import argparse
import json
import os
import math
import sys
from dataclasses import asdict, dataclass, replace
from typing import Any

__all__ = ["main", "model_layers", "probe_model"]


@dataclass(frozen=True)
class ProbeRow:
    op: str
    layer: str
    path: str
    shape: str
    route: str  # "hipdnn" or "fallback:<reason>"
    graph: str  # "hipdnn" or "<decline reason>", "-" when no graph applies
    detail: str = ""  # engine name, or the backend's decline message


@dataclass(frozen=True)
class ModelInfo:
    model: str
    attn_type: str
    dtype: Any
    hidden_size: int
    max_model_len: int
    rms_norm_eps: float | None
    requires: str  # environment requirement for HIPDNN (e.g. a local vLLM patch)
    layers: dict[str, Any]  # layer kind -> LayerStatic


def _text_config(model_config) -> Any:
    return getattr(model_config, "hf_text_config", None) or model_config.hf_config


def model_layers(model_config) -> ModelInfo:
    """The distinct attention-layer signatures of a vLLM ``ModelConfig``."""
    from vllm.config import ParallelConfig

    from .attention.routing import LayerStatic

    cfg = _text_config(model_config)
    parallel = ParallelConfig()
    attn_type = str(model_config.attn_type)
    head_size = int(model_config.get_head_size())
    pre_scalar = getattr(cfg, "query_pre_attn_scalar", None)
    scale = float(pre_scalar) ** -0.5 if pre_scalar else head_size**-0.5
    softcap = float(getattr(cfg, "attn_logit_softcapping", None) or 0.0)
    archs = [a.lower() for a in (getattr(model_config.hf_config, "architectures", None) or [])]
    has_sinks = any("gptoss" in a for a in archs)
    has_alibi = bool(getattr(cfg, "alibi", False)) or getattr(cfg, "position_embedding_type", "") == "alibi"
    window = model_config.get_sliding_window()
    layer_types = list(getattr(cfg, "layer_types", None) or [])
    encoder = attn_type in ("encoder_only", "encoder")

    def window_pair(w: int | None) -> tuple[int, int]:
        if not w:
            return (-1, -1)
        return (w - 1, w - 1) if encoder else (w - 1, 0)

    base = LayerStatic(
        num_heads=int(model_config.get_num_attention_heads(parallel)),
        num_kv_heads=int(model_config.get_num_kv_heads(parallel)),
        head_size=head_size,
        scale=scale,
        sliding_window=window_pair(None),
        logits_soft_cap=softcap,
        has_alibi=has_alibi,
        has_sinks=has_sinks,
        kv_cache_dtype="auto",
        attn_type=attn_type if attn_type in ("decoder", "encoder_only", "encoder", "encoder_decoder") else "decoder",
    )
    layers: dict[str, Any] = {}
    requires = ""
    if layer_types:
        for kind in dict.fromkeys(layer_types):
            chunked = kind == "chunked_attention"
            if chunked:
                requires = (
                    "local vLLM patch 0001 (chunk_lookback capability check); "
                    "chunked layers fall back to TRITON_ATTN"
                )
            sliding = kind in ("sliding_attention", "chunked_attention")
            layers[kind] = replace(
                base,
                sliding_window=window_pair(window if sliding else None),
                chunk_lookback=1 if chunked else -1,  # vLLM Rnj1 passes 1
            )
    else:
        layers["attention"] = replace(base, sliding_window=window_pair(window))
    return ModelInfo(
        model=model_config.model,
        attn_type=attn_type,
        dtype=model_config.dtype,
        hidden_size=int(model_config.get_hidden_size()),
        max_model_len=int(model_config.max_model_len),
        rms_norm_eps=getattr(cfg, "rms_norm_eps", None),
        requires=requires,
        layers=layers,
    )


def _route_text(route) -> str:
    return "hipdnn" if route.path is not None else f"fallback:{route.reason}"


def _graph_text(reason: str) -> str:
    return "hipdnn" if not reason else reason


def _qkv(torch, tokens: int, layer, dtype, device):
    """Strided q/k/v views of one fused QKV projection, as vLLM passes them."""
    hq, hkv, d = layer.num_heads, layer.num_kv_heads, layer.head_size
    qkv = torch.zeros(tokens, (hq + 2 * hkv) * d, dtype=dtype, device=device)
    q, k, v = qkv.split([hq * d, hkv * d, hkv * d], dim=-1)
    return q.view(tokens, hq, d), k.view(tokens, hkv, d), v.view(tokens, hkv, d)


def _sink(torch, layer, dtype, device):
    """The layer's sink_token as the backend binds it (vLLM's ``sinks`` [Hq] in
    the model dtype viewed as [1, Hq, 1, 1]), or None for a layer without sinks.
    Its values do not matter for a build."""
    if not layer.has_sinks:
        return None
    return torch.zeros(layer.num_heads, dtype=dtype, device=device).view(1, layer.num_heads, 1, 1)


def _probe_encoder(torch, C, kind, layer, info, dtype, device, lengths, counts):
    from .attention.metadata import BatchClass
    from .attention.routing import select

    route = _route_text(select(layer, BatchClass.ENCODER_ONLY, True))
    rows = []
    for length in lengths:
        for count in counts:
            tokens = count * length
            q, k, v = _qkv(torch, tokens, layer, dtype, device)
            o = torch.empty(tokens, layer.num_heads, layer.head_size, dtype=dtype, device=device)
            views = [
                t.view(count, length, t.shape[1], t.shape[2]).transpose(1, 2) for t in (q, k, v, o)
            ]
            reason, detail = C.sdpa_fwd_probe(
                *views,
                layer.scale,
                causal=False,
                left_bound=layer.sliding_window[0],
                right_bound=layer.sliding_window[1],
                sink=_sink(torch, layer, dtype, device),
            )
            rows.append(
                ProbeRow(
                    "attention", kind, "p1e", f"{count}x[L={length}]",
                    route, _graph_text(reason), detail,
                )
            )
    return rows


def _probe_decoder(torch, C, kind, layer, info, dtype, device, max_num_seqs, max_batched, block_size, max_kv):
    from .attention.metadata import BatchClass, PagedBounds, PagedBuffers
    from .attention.paged import graph_kwargs, split_kv_cache
    from .attention.routing import P2A, P2B, select

    hq, d = layer.num_heads, layer.head_size
    blocks = math.ceil(max_kv / block_size)
    b = max_num_seqs
    bounds = PagedBounds(max_num_seqs=b, max_decode_query_len=1, max_query_len=min(max_batched, max_kv),
                         max_model_len=max_kv)  # fmt: skip
    # Runner-shaped buffers; contents do not matter for a build.
    views = PagedBuffers(bounds, device).refresh(
        torch.zeros(b + 1, dtype=torch.int32, device=device),
        torch.zeros(b, dtype=torch.int32, device=device),
        torch.zeros(b, blocks, dtype=torch.int32, device=device),
    )
    # TRITON_ATTN cache (num_blocks, Hkv, block_size, 2D) split into K/V.
    cache = torch.zeros(b * blocks + 1, layer.num_kv_heads, block_size, 2 * d, dtype=dtype, device=device)
    k_cache, v_cache = split_kv_cache(cache, d)
    q, _, _ = _qkv(torch, 1, layer, dtype, device)
    o = torch.empty(1, hq, d, dtype=dtype, device=device)
    rows = []
    for path, batch_class, q_bound in ((P2A, BatchClass.UNIFORM_DECODE, 1),
                                       (P2B, BatchClass.MIXED, bounds.max_query_len)):  # fmt: skip
        kw = graph_kwargs(path, views, block_size, layer.sliding_window[0], sink=_sink(torch, layer, dtype, device))
        reason, detail = C.sdpa_fwd_probe(q, k_cache, v_cache, o, layer.scale, **kw)
        route = _route_text(select(layer, batch_class, True))
        rows.append(
            ProbeRow("attention", kind, path, f"B<={b} q<={q_bound} kv<={kw['max_seq_len_kv']}", route,
                     _graph_text(reason), detail)  # fmt: skip
        )
    return rows


def _probe_rms_norm(torch, C, info, dtype, device, row_counts):
    rows = []
    weight = torch.ones(info.hidden_size, dtype=dtype, device=device)
    for n in row_counts:
        x = torch.zeros(n, info.hidden_size, dtype=dtype, device=device)
        reason = C.rms_norm_probe(x, weight, float(info.rms_norm_eps))
        rows.append(
            ProbeRow("rms_norm", "-", "-", f"[{n},{info.hidden_size}]", "-", _graph_text(reason))
        )
    return rows


def probe_model(
    model_config,
    device: int = 0,
    block_size: int = 16,
    max_kv: int = 4096,
    lengths: tuple[int, ...] = (1, 16, 128, 512),
    max_num_seqs: int = 256,
    max_num_batched_tokens: int = 8192,
) -> tuple[ModelInfo, list[ProbeRow]]:
    import torch

    from . import _bootstrap

    _bootstrap.load()
    from . import _C

    info = model_layers(model_config)
    dtype = info.dtype if isinstance(info.dtype, torch.dtype) else torch.bfloat16
    dev = torch.device("cuda", device)
    from .attention.routing import prefilter

    rows: list[ProbeRow] = []
    for kind, layer in info.layers.items():
        pre = prefilter(layer)
        if pre is not None:
            # Pre-filtered layers never build a graph.
            rows.append(ProbeRow("attention", kind, "-", "*", f"fallback:{pre.reason}", "-"))
            continue
        if layer.attn_type == "encoder_only":
            lens = tuple(n for n in lengths if n <= info.max_model_len)
            rows += _probe_encoder(torch, _C, kind, layer, info, dtype, dev, lens, (1, 4))
        else:
            kv = min(max_kv, info.max_model_len)
            rows += _probe_decoder(torch, _C, kind, layer, info, dtype, dev, max_num_seqs,
                                   max_num_batched_tokens, block_size, kv)  # fmt: skip
    if info.rms_norm_eps is not None:
        rows += _probe_rms_norm(torch, _C, info, dtype, dev, (1, 128))
    names = engine_names(_C, device)
    rows = [replace(r, detail=names.get(r.detail, r.detail)) if r.graph == "hipdnn" else r for r in rows]
    return info, rows


def engine_names(C, device: int) -> dict[str, str]:
    """Plan name (the engine id as ``0x%016x``) -> ``name (plugin)``."""
    out = {}
    for e in C.engines(device):
        label = f"{e['name']} ({os.path.basename(e['plugin'])})" if e.get("plugin") else e["name"]
        out[f"0x{e['id'] & 0xFFFFFFFFFFFFFFFF:016x}"] = label
        out[str(e["id"])] = label
    return out


def _print_table(info: ModelInfo, rows: list[ProbeRow], out=sys.stdout) -> None:
    print(f"model: {info.model}  attn_type: {info.attn_type}  dtype: {info.dtype}", file=out)
    for kind, layer in info.layers.items():
        print(
            f"  layer {kind}: Hq={layer.num_heads} Hkv={layer.num_kv_heads} D={layer.head_size} "
            f"window={layer.sliding_window} softcap={layer.logits_soft_cap} "
            f"sinks={layer.has_sinks} alibi={layer.has_alibi}",
            file=out,
        )
    if info.requires:
        print(f"  requires for --attention-backend HIPDNN: {info.requires}", file=out)
    header = ("op", "layer", "path", "shape", "route", "graph", "engine / message")
    table = [header] + [
        (r.op, r.layer, r.path, r.shape, r.route, r.graph, r.detail.splitlines()[0][:120] if r.detail else "")
        for r in rows
    ]
    widths = [max(len(row[i]) for row in table) for i in range(len(header) - 1)]
    for row in table:
        cells = [row[i].ljust(widths[i]) for i in range(len(widths))] + [row[-1]]
        print("  " + "  ".join(cells).rstrip(), file=out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hipdnn-vllm probe", description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--max-kv", type=int, default=4096, help="max_seq_len_kv bound for the paged probes")
    p.add_argument("--max-num-seqs", type=int, default=256, help="batch bound B of the paged graphs")
    p.add_argument("--max-num-batched-tokens", type=int, default=8192, help="P2b query-length bound")
    p.add_argument("--runner", default="auto", help="vLLM runner: auto, generate or pooling")
    p.add_argument("--hf-overrides", default="", help='JSON, e.g. \'{"is_causal": false}\'')
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    from vllm.config import ModelConfig

    model_config = ModelConfig(
        model=args.model,
        runner=args.runner,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        **({"hf_overrides": json.loads(args.hf_overrides)} if args.hf_overrides else {}),
    )
    info, rows = probe_model(
        model_config, device=args.device, block_size=args.block_size, max_kv=args.max_kv,
        max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
    )  # fmt: skip
    if args.json:
        json.dump(
            {
                "model": info.model,
                "attn_type": info.attn_type,
                "dtype": str(info.dtype),
                "requires": info.requires,
                "layers": {k: asdict(v) for k, v in info.layers.items()},
                "rows": [asdict(r) for r in rows],
            },
            sys.stdout,
            indent=2,
        )
        print()
    else:
        _print_table(info, rows)
    return 0

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Day-0 AITER/ATOM integration scaffold for Qwen3-30B-A3B on gfx1250.

This module is a routing and test harness, not a production dispatcher. It keeps
accelerated paths behind explicit environment gates and an explicit operator
availability object so incomplete gfx1250 kernels never enter dispatch by
accident.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from .qwen3_30b_a3b_shapes import (
    ARCH,
    QWEN3_30B_A3B_CONFIG,
    Qwen3A3BConfig,
    QwenOpShape,
    layer_shapes,
)

MASTER_ENV = "ROCKE_GFX1250_QWEN_AITER_ATOM"
REGISTER_OPS_ENV = "ROCKE_GFX1250_QWEN_REGISTER_OPS"
GATE_ENVS = {
    "gemm": "ROCKE_GFX1250_QWEN_GEMM",
    "attention": "ROCKE_GFX1250_QWEN_ATTENTION",
    "norm": "ROCKE_GFX1250_QWEN_NORM",
    "routing": "ROCKE_GFX1250_QWEN_ROUTING",
    "moe": "ROCKE_GFX1250_QWEN_MOE",
}

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"", "0", "false", "no", "off"}
_DEFAULT_AVAILABILITY = {
    "gemm": False,
    "attention": False,
    "norm": False,
    "routing": False,
    "moe": False,
}


def env_flag(env: Mapping[str, str], name: str, *, default: bool = False) -> bool:
    """Parse a boolean env flag with conservative defaults."""
    raw = env.get(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    return bool(default)


@dataclass(frozen=True)
class GateConfig:
    """Resolved integration gates for one process."""

    master: bool = False
    gemm: bool = False
    attention: bool = False
    norm: bool = False
    routing: bool = False
    moe: bool = False
    register_torch_ops: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "GateConfig":
        env = os.environ if env is None else env
        return cls(
            master=env_flag(env, MASTER_ENV),
            gemm=env_flag(env, GATE_ENVS["gemm"]),
            attention=env_flag(env, GATE_ENVS["attention"]),
            norm=env_flag(env, GATE_ENVS["norm"]),
            routing=env_flag(env, GATE_ENVS["routing"]),
            moe=env_flag(env, GATE_ENVS["moe"]),
            register_torch_ops=env_flag(env, REGISTER_OPS_ENV),
        )

    def enabled(self, gate: str) -> bool:
        return bool(self.master and getattr(self, gate))


@dataclass(frozen=True)
class OperatorAvailability:
    """Operator-branch readiness supplied by integration tests or callers."""

    gemm: bool = False
    attention: bool = False
    norm: bool = False
    routing: bool = False
    moe: bool = False

    @classmethod
    def none(cls) -> "OperatorAvailability":
        return cls()

    @classmethod
    def from_mapping(cls, values: Mapping[str, bool]) -> "OperatorAvailability":
        merged = {**_DEFAULT_AVAILABILITY, **dict(values)}
        return cls(**{k: bool(v) for k, v in merged.items()})

    def ready(self, gate: str) -> bool:
        return bool(getattr(self, gate))


@dataclass(frozen=True)
class RouteDecision:
    """One dispatch decision for an AITER/ATOM integration point."""

    op_name: str
    kind: str
    gate: str
    backend: str
    fallback_backend: str
    enabled: bool
    accelerated: bool
    graph_capture_safe: bool
    reason: str
    dims: Mapping[str, int | str]
    custom_op: str | None = None


@dataclass(frozen=True)
class CustomOpSpec:
    """Torch custom-op ABI needed by graph-captured integration paths."""

    name: str
    schema: str
    gate: str
    graph_capture_safe: bool
    reason: str


CUSTOM_OP_SPECS: Tuple[CustomOpSpec, ...] = (
    CustomOpSpec(
        name="rmsnorm_add_out",
        schema=(
            "rmsnorm_add_out(Tensor x, Tensor residual, Tensor weight, "
            "Tensor(a!) norm_out, Tensor(b!) residual_out, float eps) -> ()"
        ),
        gate="norm",
        graph_capture_safe=True,
        reason="out-parameter ABI keeps output storage stable across graph replay",
    ),
    CustomOpSpec(
        name="router_topk_out",
        schema=(
            "router_topk_out(Tensor logits, Tensor(a!) weights_out, "
            "Tensor(b!) ids_out, int k) -> ()"
        ),
        gate="routing",
        graph_capture_safe=True,
        reason="caller-owned output buffers avoid capture-time allocation",
    ),
)

_TORCH_LIBS: list[Any] = []
_REGISTERED_NAMESPACES: set[str] = set()


def _selected_backend(shape: QwenOpShape) -> str:
    return f"rocke.gfx1250.qwen3_30b_a3b.{shape.kind}"


def _expected_shapes(config: Qwen3A3BConfig, *, kv_len: int) -> dict[str, QwenOpShape]:
    return {shape.name: shape for shape in layer_shapes(config, kv_len=kv_len)}


def _shape_supported(
    shape: QwenOpShape, config: Qwen3A3BConfig, *, kv_len: int
) -> bool:
    expected = _expected_shapes(config, kv_len=kv_len).get(shape.name)
    return expected is not None and dict(expected.dims) == dict(shape.dims)


def select_route(
    shape: QwenOpShape,
    *,
    gates: GateConfig | None = None,
    availability: OperatorAvailability | None = None,
    config: Qwen3A3BConfig = QWEN3_30B_A3B_CONFIG,
    kv_len: int = 1024,
) -> RouteDecision:
    """Select an accelerated route or an explicit fallback for one shape."""
    gates = GateConfig.from_env() if gates is None else gates
    availability = OperatorAvailability.none() if availability is None else availability
    custom_op = None
    if shape.kind == "rmsnorm_add":
        custom_op = "rocke_gfx1250_qwen::rmsnorm_add_out"
    elif shape.kind == "router_topk":
        custom_op = "rocke_gfx1250_qwen::router_topk_out"

    if config.arch != ARCH:
        return RouteDecision(
            shape.name,
            shape.kind,
            shape.gate,
            shape.fallback_backend,
            shape.fallback_backend,
            False,
            False,
            False,
            f"unsupported arch {config.arch!r}; expected {ARCH!r}",
            shape.dims,
            custom_op,
        )
    if not _shape_supported(shape, config, kv_len=kv_len):
        return RouteDecision(
            shape.name,
            shape.kind,
            shape.gate,
            shape.fallback_backend,
            shape.fallback_backend,
            False,
            False,
            False,
            "shape is not in the Qwen3-30B-A3B gfx1250 routing table",
            shape.dims,
            custom_op,
        )
    if not gates.master:
        return RouteDecision(
            shape.name,
            shape.kind,
            shape.gate,
            shape.fallback_backend,
            shape.fallback_backend,
            False,
            False,
            False,
            f"master gate {MASTER_ENV}=1 is not set",
            shape.dims,
            custom_op,
        )
    if not gates.enabled(shape.gate):
        return RouteDecision(
            shape.name,
            shape.kind,
            shape.gate,
            shape.fallback_backend,
            shape.fallback_backend,
            False,
            False,
            False,
            f"{GATE_ENVS[shape.gate]}=1 is not set",
            shape.dims,
            custom_op,
        )
    if not availability.ready(shape.gate):
        return RouteDecision(
            shape.name,
            shape.kind,
            shape.gate,
            shape.fallback_backend,
            shape.fallback_backend,
            True,
            False,
            False,
            f"operator branch unavailable: {shape.operator_dependency}",
            shape.dims,
            custom_op,
        )
    return RouteDecision(
        shape.name,
        shape.kind,
        shape.gate,
        _selected_backend(shape),
        shape.fallback_backend,
        True,
        True,
        shape.graph_capture_safe,
        "accelerated route selected by explicit gate and availability",
        shape.dims,
        custom_op,
    )


def plan_qwen_layer(
    *,
    gates: GateConfig | None = None,
    availability: OperatorAvailability | None = None,
    config: Qwen3A3BConfig = QWEN3_30B_A3B_CONFIG,
    kv_len: int = 1024,
) -> Tuple[RouteDecision, ...]:
    """Return ordered route decisions for one Qwen decode layer."""
    return tuple(
        select_route(
            shape,
            gates=gates,
            availability=availability,
            config=config,
            kv_len=kv_len,
        )
        for shape in layer_shapes(config, kv_len=kv_len)
    )


def custom_op_specs() -> Tuple[CustomOpSpec, ...]:
    return CUSTOM_OP_SPECS


def register_torch_custom_ops(namespace: str = "rocke_gfx1250_qwen") -> bool:
    """Register graph-compatible torch custom-op schemas and CPU references."""
    if namespace in _REGISTERED_NAMESPACES:
        return True
    try:
        import torch
    except Exception:
        return False

    def rmsnorm_add_out(
        x, residual, weight, norm_out, residual_out, eps: float
    ) -> None:
        y = x + residual
        residual_out.copy_(y)
        inv = torch.rsqrt(torch.mean(y.float() * y.float(), dim=-1, keepdim=True) + eps)
        norm_out.copy_((y.float() * inv).to(dtype=x.dtype) * weight)

    def router_topk_out(logits, weights_out, ids_out, k: int) -> None:
        values, ids = torch.topk(logits.float(), int(k), dim=-1)
        weights_out.copy_(torch.softmax(values, dim=-1).to(dtype=weights_out.dtype))
        ids_out.copy_(ids.to(dtype=ids_out.dtype))

    try:
        def_lib = torch.library.Library(namespace, "DEF")
        for spec in CUSTOM_OP_SPECS:
            def_lib.define(spec.schema)
        impl_lib = torch.library.Library(namespace, "IMPL")
        impl_lib.impl("rmsnorm_add_out", rmsnorm_add_out, "CompositeExplicitAutograd")
        impl_lib.impl("router_topk_out", router_topk_out, "CompositeExplicitAutograd")
        meta_lib = torch.library.Library(namespace, "IMPL")
        meta_lib.impl("rmsnorm_add_out", lambda *args, **kwargs: None, "Meta")
        meta_lib.impl("router_topk_out", lambda *args, **kwargs: None, "Meta")
    except RuntimeError as exc:
        if "Only a single TORCH_LIBRARY" not in str(exc) and "already" not in str(exc):
            raise
    _TORCH_LIBS.extend([def_lib, impl_lib, meta_lib])
    _REGISTERED_NAMESPACES.add(namespace)
    return True


def maybe_register_torch_custom_ops(
    gates: GateConfig | None = None, namespace: str = "rocke_gfx1250_qwen"
) -> bool:
    gates = GateConfig.from_env() if gates is None else gates
    if not gates.master or not gates.register_torch_ops:
        return False
    return register_torch_custom_ops(namespace)


@dataclass(frozen=True)
class QwenOneLayerHarness:
    """Tiny one-layer functional harness with independently gated route planning."""

    config: Qwen3A3BConfig = QWEN3_30B_A3B_CONFIG
    kv_len: int = 1024
    gates: GateConfig = GateConfig()
    availability: OperatorAvailability = OperatorAvailability()

    def plan(self) -> Tuple[RouteDecision, ...]:
        return plan_qwen_layer(
            gates=self.gates,
            availability=self.availability,
            config=self.config,
            kv_len=self.kv_len,
        )

    def run_reference_torch(self, torch_module: Any | None = None) -> Any:
        """Run a deterministic small torch reference for functional scaffolding."""
        if torch_module is None:
            import torch as torch_module

        torch = torch_module
        cfg = self.config
        dtype = torch.float32
        torch.manual_seed(123)
        x = torch.randn(cfg.batch, cfg.hidden, dtype=dtype)
        residual = torch.randn_like(x) * 0.01
        norm_weight = torch.ones(cfg.hidden, dtype=dtype)
        qkv_weight = torch.randn(cfg.qkv_width, cfg.hidden, dtype=dtype) / math.sqrt(
            cfg.hidden
        )
        o_weight = torch.randn(
            cfg.hidden, cfg.attention_width, dtype=dtype
        ) / math.sqrt(max(1, cfg.attention_width))
        router_weight = torch.randn(
            cfg.num_experts, cfg.hidden, dtype=dtype
        ) / math.sqrt(cfg.hidden)
        w_gate = torch.randn(
            cfg.num_experts, cfg.moe_intermediate, cfg.hidden, dtype=dtype
        ) / math.sqrt(cfg.hidden)
        w_up = torch.randn(
            cfg.num_experts, cfg.moe_intermediate, cfg.hidden, dtype=dtype
        ) / math.sqrt(cfg.hidden)
        w_down = torch.randn(
            cfg.num_experts, cfg.hidden, cfg.moe_intermediate, dtype=dtype
        ) / math.sqrt(max(1, cfg.moe_intermediate))

        normed, _ = _rmsnorm_add_ref(torch, x, residual, norm_weight, eps=1e-6)
        qkv = normed.matmul(qkv_weight.t())
        q = qkv[:, : cfg.attention_width]
        dense = q.reshape(cfg.batch, cfg.attention_width).matmul(o_weight.t())
        logits = dense.matmul(router_weight.t())
        top_weights, top_ids = _router_topk_ref(torch, logits, cfg.topk)
        moe = _moe_ref(torch, dense, top_weights, top_ids, w_gate, w_up, w_down)
        return {
            "plan": self.plan(),
            "hidden_states": dense,
            "router_weights": top_weights,
            "router_ids": top_ids,
            "moe_out": moe,
        }


def _rmsnorm_add_ref(
    torch: Any, x: Any, residual: Any, weight: Any, *, eps: float
) -> Any:
    y = x + residual
    inv = torch.rsqrt(torch.mean(y.float() * y.float(), dim=-1, keepdim=True) + eps)
    return y.float() * inv * weight, y


def _router_topk_ref(torch: Any, logits: Any, topk: int) -> Any:
    values, ids = torch.topk(logits.float(), int(topk), dim=-1)
    return torch.softmax(values, dim=-1), ids.int()


def _moe_ref(
    torch: Any, x: Any, weights: Any, ids: Any, w_gate: Any, w_up: Any, w_down: Any
) -> Any:
    out = torch.zeros_like(x)
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            gate = torch.nn.functional.silu(x[token].matmul(w_gate[expert].t()))
            up = x[token].matmul(w_up[expert].t())
            hidden = gate * up
            out[token] += weights[token, slot] * hidden.matmul(w_down[expert].t())
    return out


if env_flag(os.environ, REGISTER_OPS_ENV):
    maybe_register_torch_custom_ops()

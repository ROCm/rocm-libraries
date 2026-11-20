# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Features a valid graph may leave unbound, and what a model reads instead (RFC 0019 §6.7, §6.8).

hipDNN lists the optional features: `hipdnn_uhd_features --optional-features`, generated from
the schemas with the operand visitor, plus the work features. This module holds what a model
reads when one is absent, and makes and checks signatures that read each of them through
`value_or_default`. Which references are unguarded is asked of the evaluator (`--lint`), so the
validator, promotion and training share one definition. The runtime never fills in a missing
value: the default is the model's choice, recorded in its signature. A signature that reads
an optional feature bare is refused for the reasons of RFC 0019 §6.7 (absent is not zero) and
§6.8 item 9.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from .features import (
    parse_signature_entry,
    resolve_feature_evaluator,
    run_evaluator,
)

#: Read for an absent feature whose unset has no number of its own: an omitted operand's field,
#: unknown work, an unset uid or sequence length. It differs from 0 ("absent, zero and unknown
#: are different values", RFC 0019 §6.8 item 9), and no such feature is negative when set.
#: These are markers, not API meanings.
SENTINEL = -1

#: Read for an unset ReLU upper clip, whose API meaning is no upper bound: cuDNN's default is
#: the float maximum (its double maximum clamped to float, cudnn-frontend `pointwise.h`). A
#: literal stays below 1e15, so the expression writes it as an exact product:
#: (2**24 - 1) * 2**40 * 2**40 * 2**24 = 3.4028234663852886e38.
FLOAT_MAX = {"*": [16777215, 1099511627776, 1099511627776, 16777216]}

#: What the API means when an optional attribute or a vector element is unset, keyed by the
#: optional feature's pattern: the number, or an expression of the reading node's own operands
#: (`{node}` is its index). Where unset has no number, SENTINEL stands in.
ATTRIBUTE_DEFAULTS: dict[str, object] = {
    # SDPA forward and backward (SdpaAttributes.hpp, SdpaBackwardAttributes.hpp: the scale is
    # 1.0 when neither a value nor a tensor is set; the fbs files state no default). A scale
    # given as a tensor also leaves attn_scale_value unset; its value is the tensor's.
    "graph.nodes[*].attn_scale_value": 1.0,
    "graph.nodes[*].dropout_probability": 0.0,
    "graph.nodes[*].generate_stats": False,
    "graph.nodes[*].left_bound": -1,  # the API's own value for unbounded
    "graph.nodes[*].right_bound": -1,  # likewise
    "graph.nodes[*].max_seq_len_kv": SENTINEL,  # not declared
    # Pointwise (hipdnn_frontend/attributes/PointwiseAttributes.hpp; cuDNN's defaults)
    "graph.nodes[*].relu_lower_clip": 0.0,
    "graph.nodes[*].relu_upper_clip": FLOAT_MAX,  # no upper bound
    "graph.nodes[*].relu_lower_clip_slope": 0.0,
    "graph.nodes[*].swish_beta": 1.0,
    "graph.nodes[*].elu_alpha": 1.0,
    "graph.nodes[*].softplus_beta": 1.0,
    "graph.nodes[*].axis_tensor_uid": SENTINEL,  # GEN_INDEX's axis, a plain index
    # Block-scale quantize (BlockScaleQuantizeNode.hpp: an unset axis is the last axis of x)
    # and resample (the axis of a pointwise GEN_INDEX is a tensor's uid, above)
    "graph.nodes[*].axis": {"-": ["$graph.nodes[{node}].x.rank", 1]},
    "graph.nodes[*].generate_index": False,
    # GDN and KDA (gdn_attributes.fbs, kda_attributes.fbs): 1/sqrt(K) of q [T, H, K], and
    # KDA's safe-gate lower bound
    "graph.nodes[*].scale": {"rsqrt": ["$graph.nodes[{node}].q.dims[2]"]},
    "graph.nodes[*].gate_lower_bound": -5.0,
    # Vector attributes: an element past the end is a spatial dimension the node does not
    # have, which neither pads, strides, dilates, windows nor blocks.
    "graph.nodes[*].pre_padding[*]": 0,
    "graph.nodes[*].post_padding[*]": 0,
    "graph.nodes[*].stride[*]": 1,
    "graph.nodes[*].dilation[*]": 1,
    "graph.nodes[*].window[*]": 1,
    "graph.nodes[*].block_size[*]": 1,
}

#: Kinds whose every feature reads SENTINEL when absent: an omitted operand's fields, work that
#: is unknown, and an engine-config constraint the config does not pin (no limit, knob unset).
SENTINEL_KINDS = ("tensor", "tensors", "work", "constraint")

#: Where an optional feature without node types is absent.
_ABSENT_WITHOUT_NODE_TYPES = {
    "work": "any graph of unknown work",
    "constraint": "any engine config that does not pin it",
}

_NODE_INDEX = re.compile(r"^\$?graph\.nodes\[(\d+)\]")


def _ask(mode: str, request: dict | None, executable, key: str):
    """`key` of the evaluator's answer to `mode`; an evaluator that cannot answer predates the
    optional-feature checks."""
    try:
        return run_evaluator(request, executable, mode)[key]
    except (ValueError, KeyError) as error:
        raise ValueError(
            f"{error}; an evaluator without {mode} predates optional-feature checks, so "
            "rebuild it"
        ) from error


@lru_cache(maxsize=None)
def _optional_features(resolved: str) -> tuple[dict, ...]:
    return tuple(_ask("--optional-features", None, resolved, "optional_features"))


def optional_features(executable: str | Path | None = None) -> tuple[dict, ...]:
    """hipDNN's optional features, `{pattern, kind, node_types}`, from the evaluator."""
    return _optional_features(resolve_feature_evaluator(executable))


def _pattern_regex(pattern: str) -> re.Pattern:
    body = re.escape(pattern).replace(r"\[\*\]", r"\[\d+\]")
    if body.endswith(r"\.\*"):
        body = body[: -len(r"\.\*")] + r"\..+"
    return re.compile(body)


def optional_feature_of(name: str, features) -> dict | None:
    """The optional feature published `name` (with or without `$`) belongs to, or None."""
    name = name.removeprefix("$")
    for feature in features:
        if _pattern_regex(feature["pattern"]).fullmatch(name):
            return feature
    return None


def default_for(reference: str, feature: dict):
    """The value a model reads for `reference`, of optional `feature`, when it is absent."""
    if feature["kind"] in SENTINEL_KINDS:
        return SENTINEL
    try:
        default = ATTRIBUTE_DEFAULTS[feature["pattern"]]
    except KeyError:
        raise ValueError(
            f"no default for optional feature {feature['pattern']} (optional in "
            f"{', '.join(feature['node_types'])}): add what the API means when it is unset "
            "to uhd_gen.optional.ATTRIBUTE_DEFAULTS"
        ) from None
    if isinstance(default, dict):
        match = _NODE_INDEX.match(reference)
        if match is None:
            raise ValueError(f"{reference} names no node for its default {default}")
        return json.loads(json.dumps(default).replace("{node}", match.group(1)))
    return default


def missing_defaults(features) -> list[str]:
    """Patterns of optional attributes and elements the defaults table does not cover."""
    return sorted(
        feature["pattern"]
        for feature in features
        if feature["kind"] not in SENTINEL_KINDS
        and feature["pattern"] not in ATTRIBUTE_DEFAULTS
    )


def feature_name(entry, index: int) -> str:
    """A model input's name: the reference it reads, bare or through value_or_default, else
    `expression_<index>`."""
    if isinstance(entry, str):
        return entry[1:]
    if isinstance(entry, dict) and list(entry) == ["value_or_default"]:
        first = entry["value_or_default"][0]
        if isinstance(first, str) and first.startswith("$"):
            return first[1:]
    return f"expression_{index}"


def lint(signature: list, metric: str | None, executable=None) -> list[dict]:
    """The references of `signature` a valid graph may leave unbound and nothing guards, for
    a model of `metric`: each `{reference, pattern, kind, node_types}`."""
    request = {
        "signature": [parse_signature_entry(entry) for entry in signature],
        "metric": metric,
    }
    return _ask("--lint", request, executable, "unguarded")


def _wrap(node, targets: dict):
    """`node` with every read of a reference in `targets` taking its default when unbound.

    A read that is already guarded is wrapped too, which changes no value: value_or_default
    returns its first argument whenever that resolves.
    """
    if isinstance(node, str):
        if node in targets:
            return {"value_or_default": [node, targets[node]]}
        return node
    if isinstance(node, list):
        return [_wrap(item, targets) for item in node]
    if isinstance(node, dict) and len(node) == 1:
        ((op, arguments),) = node.items()
        if op in ("present", "not_present"):
            return node
        if op == "value_or_default" and isinstance(arguments, list) and arguments:
            return {
                op: [arguments[0], *(_wrap(item, targets) for item in arguments[1:])]
            }
        return {op: _wrap(arguments, targets)}
    return node


def safe_signature(
    signature: list, metric: str | None, executable=None
) -> tuple[list, list[dict]]:
    """`signature` reading every optional feature through `value_or_default`, and what was
    wrapped. On a graph that binds a feature the value is unchanged, so a model's predictions
    on every graph it could already score are too; a graph without it now gets its default.
    """
    unguarded = lint(signature, metric, executable)
    if not unguarded:
        return list(signature), []
    targets = {
        found["reference"]: default_for(found["reference"], found)
        for found in unguarded
    }
    wrapped = [_wrap(parse_signature_entry(entry), targets) for entry in signature]
    remaining = lint(wrapped, metric, executable)
    if remaining:
        raise ValueError(
            "wrapping left optional features unguarded: "
            + ", ".join(found["reference"] for found in remaining)
        )
    return wrapped, [
        {"reference": reference, "default": default}
        for reference, default in targets.items()
    ]


def _optional_in(found: dict) -> str:
    return ", ".join(found["node_types"]) or _ABSENT_WITHOUT_NODE_TYPES.get(
        found["kind"], "some graphs"
    )


def require_safe_signature(
    signature: list, metric: str | None, where: str, executable=None
):
    """Refuse a signature that leaves a valid graph unscorable."""
    unguarded = lint(signature, metric, executable)
    if unguarded:
        raise ValueError(
            f"{where} reads features a valid graph may leave unbound, so the engine cannot "
            "score such a graph: "
            + ", ".join(
                f"{found['reference']} ({found['kind']}, optional in {_optional_in(found)})"
                for found in unguarded
            )
            + ". Read each through value_or_default (uhd_gen rewrite-signature does, without "
            "retraining)"
        )

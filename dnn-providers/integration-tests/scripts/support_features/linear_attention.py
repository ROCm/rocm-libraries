# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Features shared by the two linear-attention ops.

``GdnAttributes`` (Gated DeltaNet) and ``KdaAttributes`` (Kimi Delta
Attention) are one recurrence with two decays: GDN's is a scalar per token and
head, KDA's one value per key channel. ``gdn_attributes.fbs`` and
``kda_attributes.fbs`` carry the same fields otherwise, plus KDA's
``gate_lower_bound``, so the groups below are read the same way for both. The
serializers (``utilities/json/GdnAttributes.hpp``, ``KdaAttributes.hpp``)
write the tensor ports under ``inputs`` and ``outputs`` and the scalars under
``attributes``; an unset optional is ``null``. Groups:

- ``dtype`` (exclusive): the dtype of ``q``.
- ``gate`` (exclusive): how ``g`` encodes the decay. ``log`` (``ln(alpha)``,
  the default) and ``linear`` (``alpha`` itself) are the ``gate_domain``;
  ``safe`` is ``safe_gate``, where ``g`` holds logits and the op derives the
  decay with ``a_log`` and ``dt_bias``. ``safe`` has no domain of its own:
  ``LINEAR`` with ``safe_gate`` is invalid.
- ``beta`` (exclusive): ``direct`` (``beta`` is the write strength),
  ``sigmoid`` (it holds logits) or ``sigmoid_neg_eigval`` (logits, and the
  strength is ``2 * sigmoid``).
- ``norm:qk_l2``: ``q`` and ``k`` rows are L2-normalized.
- ``scale`` (exclusive): ``default`` (``scale`` absent, ``1/sqrt(K)``) or
  ``fixed`` (a baked value).
- ``state:initial``, ``state:final``, ``state:pool``, ``state:in_place``,
  ``state:checkpoints``: the recurrent-state ports and modes; a graph may
  carry any combination the rules below allow.
- ``batch_invariant:on``: the result does not depend on the batch.

The schema comments list cross-field rules that the backend does not check and
the frontend is to enforce. A graph that breaks one raises rather than
counting, as one the API would reject does for every other op: a ``gate_domain``
the serializer does not write, ``output_final_state`` without a final state
(or the reverse), checkpoints without ``checkpoint_every_n_tokens`` (or the
reverse), ``state_indices`` without an initial state or with checkpoints,
``a_log`` / ``dt_bias`` / ``gate_lower_bound`` without ``safe_gate``,
``allow_neg_eigval`` without ``use_beta_sigmoid``, and ``overwrite_initial_state``
without both an initial and a final state.
"""

from __future__ import annotations

from support_features import Feature, FeatureError, batchnorm, convolution
from support_matrix_units import dtype_label

DTYPES = convolution.DTYPES
GATE_DOMAINS = {"log": "gate:log", "linear": "gate:linear"}

REQUIRED = (
    "q_tensor_uid",
    "k_tensor_uid",
    "v_tensor_uid",
    "g_tensor_uid",
    "beta_tensor_uid",
    "cu_seqlens_tensor_uid",
    "o_tensor_uid",
)

NOT_FEATURES: dict[str, str] = {
    "k_tensor_uid": "required, token-packed like q: it varies nothing q does not",
    "v_tensor_uid": "required, token-packed like q: it varies nothing q does not",
    "g_tensor_uid": "required; the gate feature is how its values are read, not "
    "whether it is bound",
    "beta_tensor_uid": "required; the beta feature is how its values are read, not "
    "whether it is bound",
    "cu_seqlens_tensor_uid": "required [N + 1] sequence offsets: the packing is "
    "fixed, where each sequence starts is data the graph does not carry",
    "o_tensor_uid": "required; its shape follows q and v, so it varies nothing "
    "they do not",
}


def catalog(*, lower_bound: bool) -> tuple[Feature, ...]:
    """The shared groups; ``lower_bound`` adds KDA's ``gate_lower_bound``."""
    safe_fields = ("safe_gate", "a_log_tensor_uid", "dt_bias_tensor_uid")
    return (
        *(Feature(f"dtype:{dtype}", ("q_tensor_uid",)) for dtype in DTYPES),
        Feature("gate:log", ("gate_domain",), "g holds ln(alpha)"),
        Feature("gate:linear", ("gate_domain",), "g holds alpha in (0, 1]"),
        Feature("gate:safe", safe_fields, "g holds logits; a_log and dt_bias shape it"),
        *(
            (Feature("gate_lower_bound:set", ("gate_lower_bound",)),)
            if lower_bound
            else ()
        ),
        Feature("beta:direct", ("use_beta_sigmoid", "allow_neg_eigval")),
        Feature("beta:sigmoid", ("use_beta_sigmoid", "allow_neg_eigval")),
        Feature("beta:sigmoid_neg_eigval", ("use_beta_sigmoid", "allow_neg_eigval")),
        Feature("norm:qk_l2", ("use_qk_l2norm",)),
        Feature("scale:default", ("scale",), "absent: 1/sqrt(K)"),
        Feature("scale:fixed", ("scale",)),
        Feature("state:initial", ("initial_state_tensor_uid",)),
        Feature("state:final", ("final_state_tensor_uid", "output_final_state")),
        Feature(
            "state:pool",
            ("state_indices_tensor_uid",),
            "state rows addressed per sequence",
        ),
        Feature(
            "state:in_place",
            ("overwrite_initial_state",),
            "final state shares the initial buffer",
        ),
        Feature(
            "state:checkpoints",
            ("state_checkpoints_tensor_uid", "checkpoint_every_n_tokens"),
        ),
        Feature("batch_invariant:on", ("batch_invariant",)),
    )


def extract(node: dict, tensors_by_uid: dict, *, lower_bound: bool) -> set[str]:
    by_port, has = batchnorm.bound(node), batchnorm.binds(node)
    attrs = node.get("attributes") or {}
    for port in REQUIRED:
        if not has(port):
            raise FeatureError(f"required port {port} is not bound")

    q = tensors_by_uid[by_port["q_tensor_uid"]]
    dtype = dtype_label(q.get("data_type"))
    if dtype not in DTYPES:
        raise FeatureError(f"q_tensor_uid dtype {dtype!r} is not catalogued")
    features = {f"dtype:{dtype}"}

    safe = bool(attrs.get("safe_gate"))
    domain = attrs.get("gate_domain", "log")
    if domain not in GATE_DOMAINS:
        raise FeatureError(f"gate_domain {domain!r} is neither 'log' nor 'linear'")
    if safe and domain != "log":
        raise FeatureError("gate_domain 'linear' is invalid with safe_gate")
    for port in ("a_log_tensor_uid", "dt_bias_tensor_uid"):
        if has(port) and not safe:
            raise FeatureError(f"{port} requires safe_gate")
    features.add("gate:safe" if safe else GATE_DOMAINS[domain])

    bound = attrs.get("gate_lower_bound")
    if lower_bound and bound is not None:
        if not safe:
            raise FeatureError("gate_lower_bound requires safe_gate")
        features.add("gate_lower_bound:set")

    sigmoid = bool(attrs.get("use_beta_sigmoid"))
    neg_eigval = bool(attrs.get("allow_neg_eigval"))
    if neg_eigval and not sigmoid:
        raise FeatureError("allow_neg_eigval requires use_beta_sigmoid")
    features.add(
        "beta:sigmoid_neg_eigval"
        if neg_eigval
        else "beta:sigmoid" if sigmoid else "beta:direct"
    )

    if attrs.get("use_qk_l2norm"):
        features.add("norm:qk_l2")
    features.add("scale:default" if attrs.get("scale") is None else "scale:fixed")

    final = has("final_state_tensor_uid")
    if bool(attrs.get("output_final_state")) != final:
        raise FeatureError("output_final_state and final_state_tensor_uid disagree")
    every = attrs.get("checkpoint_every_n_tokens") or 0
    if every < 0:
        raise FeatureError(f"checkpoint_every_n_tokens {every} is negative")
    checkpoints = has("state_checkpoints_tensor_uid")
    if (every > 0) != checkpoints:
        raise FeatureError(
            "checkpoint_every_n_tokens and state_checkpoints_tensor_uid disagree"
        )

    initial, pool = has("initial_state_tensor_uid"), has("state_indices_tensor_uid")
    if pool and not initial:
        raise FeatureError("state_indices_tensor_uid requires initial_state_tensor_uid")
    if pool and checkpoints:
        raise FeatureError("state_indices_tensor_uid cannot combine with checkpoints")
    in_place = bool(attrs.get("overwrite_initial_state"))
    if in_place and not (initial and final):
        raise FeatureError(
            "overwrite_initial_state requires an initial and a final state"
        )

    for condition, feature in (
        (initial, "state:initial"),
        (final, "state:final"),
        (pool, "state:pool"),
        (in_place, "state:in_place"),
        (checkpoints, "state:checkpoints"),
        (attrs.get("batch_invariant"), "batch_invariant:on"),
    ):
        if condition:
            features.add(feature)
    return features

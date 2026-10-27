# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``KdaAttributes`` (Kimi Delta Attention forward).

The features come from ``kda_attributes.fbs``; the groups, and what makes a
graph raise, are described in ``linear_attention.py``, which KDA shares with
GDN. KDA's decay is one value per key channel, and it adds
``gate_lower_bound``, the lower bound of the safe gate in log space.
"""

from __future__ import annotations

from support_features import Feature, linear_attention

NODE_TYPE = "KdaAttributes"
SCHEMA_FILE = "kda_attributes.fbs"
SCHEMA_TABLE = "KdaAttributes"

FEATURES: tuple[Feature, ...] = linear_attention.catalog(lower_bound=True)
NOT_FEATURES = linear_attention.NOT_FEATURES


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    return linear_attention.extract(node, tensors_by_uid, lower_bound=True)

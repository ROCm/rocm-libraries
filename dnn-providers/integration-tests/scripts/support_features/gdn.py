# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature catalog for ``GdnAttributes`` (Gated DeltaNet forward).

The features come from ``gdn_attributes.fbs``; the groups, and what makes a
graph raise, are described in ``linear_attention.py``, which GDN shares with
KDA. GDN's decay is a scalar per token and head.
"""

from __future__ import annotations

from support_features import Feature, linear_attention

NODE_TYPE = "GdnAttributes"
SCHEMA_FILE = "gdn_attributes.fbs"
SCHEMA_TABLE = "GdnAttributes"

FEATURES: tuple[Feature, ...] = linear_attention.catalog(lower_bound=False)
NOT_FEATURES = linear_attention.NOT_FEATURES


def extract(node: dict, tensors_by_uid: dict) -> set[str]:
    return linear_attention.extract(node, tensors_by_uid, lower_bound=False)

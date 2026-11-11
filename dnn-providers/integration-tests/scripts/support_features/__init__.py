# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature coverage per op: how much of an op's API the bundles exercise.

The support matrix counts graphs. A graph count cannot show that an op has a
feature no bundle touches -- forty causal-mask graphs and no sliding-window
graph still read as forty. This package counts *features* instead: for each
op, the share of its API surface that at least one bundle graph exercises, and
the share that at least one claimed graph exercises on each arch.

The denominator is the op's API, never the bundles. Each op module catalogs
the features of one node type in the flatbuffers schema
(``projects/hipdnn/flatbuffers_sdk/schemas/``), and ``test_support_features.py``
holds every schema field to account: a field drives some feature, or is listed
in ``NOT_FEATURES`` with the reason it does not. A catalog built from the
bundles would cover itself and read 100%.

Layout:

- ``registry.py`` -- API node type -> op module; ops not yet catalogued.
- ``coverage.py`` -- joins claim units with extracted features; the CLI.
- ``convolution.py``, ``batchnorm.py`` -- the features a family of ops shares.
- one module per op (``sdpa.py``, ...), each exposing ``NODE_TYPE``,
  ``SCHEMA_FILE``, ``SCHEMA_TABLE``, ``FEATURES``, ``NOT_FEATURES`` and
  ``extract(node, tensors_by_uid) -> set[str]``.

``extract`` raises ``FeatureError`` on a value its catalog does not name (a
head dim of 96, a new dtype), so a bundle that reaches past the catalog fails
loudly instead of disappearing from both numerator and denominator.
"""

from __future__ import annotations

from dataclasses import dataclass


class FeatureError(ValueError):
    """A graph uses something the op's feature catalog does not name."""


@dataclass(frozen=True)
class Feature:
    """One countable API feature of an op.

    ``id`` reads ``<group>:<value>`` -- ``mask:sliding_window``. ``fields``
    names the schema fields the feature is reached through, which is what the
    schema exhaustiveness test checks against.
    """

    id: str
    fields: tuple[str, ...]
    description: str = ""

    @property
    def group(self) -> str:
        return self.id.split(":", 1)[0]


def scalar_source(tensor: dict, port: str) -> str:
    """How a scalar operand (an epsilon, a momentum) gets its value.

    ``"runtime"``: the tensor is runtime pass-by-value, so the caller passes
    the value with each execution (with or without a default baked into the
    graph). ``"fixed"``: the value is a constant baked into the graph. These
    are the states ``isPassByValueTensor`` (FlatbufferUtils.hpp) accepts; a
    scalar in device memory is in neither and raises, as does one with more
    than one element.
    """
    elements = 1
    for dim in tensor.get("dims") or ():
        elements *= dim
    if not tensor.get("dims") or elements != 1:
        raise FeatureError(f"{port} dims {tensor.get('dims')} are not one scalar")
    if tensor.get("is_runtime_pass_by_value"):
        return "runtime"
    if "value" in tensor:
        return "fixed"
    raise FeatureError(f"{port} is neither a baked constant nor runtime pass-by-value")

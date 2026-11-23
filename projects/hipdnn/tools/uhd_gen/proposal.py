# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What a feature proposal and a feature search may read of the published columns.

`generate`'s default proposal and `feature-search` apply one policy: the exclusion rules
(columns a model must not learn the corpus from) and the presence flag a tensor identifier is
read as. An authored recipe is never filtered by either. Imports nothing from `generate` or
`feature_search`, which both read it.
"""
from __future__ import annotations

import re

import pandas as pd

from .optional import (
    guarded_read,
    optional_feature_of,
    optional_features,
    optional_guards,
)


def is_tensor_identifier(name: str) -> bool:
    """A published `<namespace>.<token>.uid`: the uid of the tensor a matcher bound, kept so
    dispatch finds its buffer. It labels the tensor and says nothing about the problem, and a
    corpus fixes it per form, so a model reading its value learns the corpus's label."""
    return name.rsplit(".", 1)[-1] == "uid"


def is_positional_tensor(name: str) -> bool:
    """`graph.tensors[<i>].*`: the graph's tensors by position, each a copy of a node's role
    (`graph.nodes[0].q`, ...). The position is not a stable role (2.1% of the audited SDPA
    rows disagree with the node's k and o), so a model reading it learns the ordering of the
    graph's tensor table; the node-role columns stay."""
    return re.match(r"graph\.tensors\[\d+\]\.", name) is not None


def is_derived_numel(name: str) -> bool:
    """A `<tensor>.numel`: the product of the tensor's published dims."""
    return name.endswith(".numel")


def is_caller_value(name: str) -> bool:
    """A multiplier the caller chose, with no path to kernel time: a node's
    `attn_scale_value`, or a matcher's `<namespace>.<token>` (the namespace is not `graph`,
    `device` or `kernel`) whose token is `scale`, `sm_scale` or `softmax_scale`, each with an
    optional `_bits` or `_value` suffix; `scale_source` is a mode, and stays. A corpus that
    sets it to 1/sqrt(head_dim) makes it a second head_dim to a model."""
    if re.fullmatch(r"graph\.nodes\[\d+\]\.attn_scale_value", name):
        return True
    namespace, _, token = name.partition(".")
    return namespace not in ("graph", "device", "kernel") and (
        re.fullmatch(r"(sm_|softmax_)?scale(_bits|_value)?", token) is not None
    )


def is_board_identity(name: str) -> bool:
    """`device.total_global_mem`: it varies only between the boards a training set pools,
    and capacity has no path to kernel time, so a model reading it learns the board."""
    return name == "device.total_global_mem"


#: The columns a proposal and a feature search never read, by reason: the first predicate a
#: column satisfies names it. An authored recipe is never filtered.
EXCLUSIONS = (
    ("positional_tensor", is_positional_tensor),
    ("derived_numel", is_derived_numel),
    ("caller_value", is_caller_value),
    ("board_identity", is_board_identity),
)


def exclusion_reason(name: str) -> str | None:
    """Why `name` is never proposed, or None."""
    return next((reason for reason, rule in EXCLUSIONS if rule(name)), None)


def presence_flag(name: str, default=None) -> dict:
    """What a tensor identifier carries about the problem: whether the tensor is bound. The
    flag is `uid != -1`: the attention packs of the hip-kernel-provider (Flydsl SDPA, the
    gfx942/gfx950 dense and unified attention) bind an absent optional tensor's uid as -1
    (NO_UID, NO_STATS, NO_BIAS, ...), and any other value, 0 included, is a bound tensor.
    Not every matcher follows it: `gfx1151_wmma_attention` binds `lse.uid` as the output's
    uid when stats are off (read its `write_lse` instead) and leaves `scale.uid`, `bias.uid`,
    `sink.uid`, `q_offset.uid`, `page_table.uid` and `seq_len_*.uid` unbound when the tensor
    is absent, and `flydsl_rmsnorm` leaves `inv_rms.uid` unbound outside a training graph.
    A column with such gaps is not an identifier a proposal can read: it enters a proposal
    only as one of hipDNN's optional features, which an engine's namespace never is.

    An identifier a valid graph may leave unbound (an optional tensor's uid) is read through
    value_or_default, with `default` (the -1 an absent tensor has), so the flag is still
    never a read of the uid's value and an unbound row reads as absent."""
    read = "$" + name
    if default is not None:
        read = guarded_read(read, default)
    return {"!=": [read, -1]}


def identifier_omissions(identifiers: list[str], guards: dict | None = None) -> list:
    """The `omitted_proposals` entries of the tensor identifiers read as presence flags;
    `guards` holds the default of each one that is optional."""
    guards = guards or {}
    return [
        {
            "expression": "$" + name,
            "reason": "tensor_identifier",
            "read_instead": presence_flag(name, guards.get(name)),
        }
        for name in identifiers
    ]


def settle_omissions(omitted: list, recipe: list) -> list:
    """`omitted` for the `recipe` a search chose: an identifier names the flag read instead
    only if the recipe reads it, since the search may leave a flag out."""
    return [
        (
            {key: value for key, value in entry.items() if key != "read_instead"}
            if "read_instead" in entry and entry["read_instead"] not in recipe
            else entry
        )
        for entry in omitted
    ]


def proposal_columns(
    train_frame: pd.DataFrame,
    published: set[str],
    kmd_fields: list[str] | None,
    knobs: list[str],
    optional: tuple = (),
) -> tuple[list[str], list[str], set[str], list]:
    """What a proposal starts from: (scalar columns, tensor identifiers, offered kernel
    fields, omitted). A column every training row binds as a scalar is a candidate, and so
    is one some rows leave unbound if it is one of hipDNN's `optional` features (read through
    value_or_default); an exclusion rule names the others, and a tensor identifier is read
    only as a presence flag. Each exclusion is recorded in `omitted` with its reason."""
    offered = {"kernel." + field for field in kmd_fields or () if field in knobs}
    scalar_columns = [
        name
        for name in sorted(published)
        if (
            train_frame[name].notna().all()
            or (
                train_frame[name].notna().any()
                and optional_feature_of(name, optional) is not None
            )
        )
        and train_frame[name]
        .dropna()
        .map(lambda value: isinstance(value, (str, int, float, bool)))
        .all()
    ]
    omitted = [
        {"expression": "$" + name, "reason": reason}
        for name in scalar_columns
        if (reason := exclusion_reason(name))
    ]
    scalar_columns = [name for name in scalar_columns if not exclusion_reason(name)]
    identifiers = [name for name in scalar_columns if is_tensor_identifier(name)]
    scalar_columns = [name for name in scalar_columns if not is_tensor_identifier(name)]
    legal_kernel_fields = {
        name for name in scalar_columns if name.split("[", 1)[0] in offered
    }
    return scalar_columns, identifiers, legal_kernel_fields, omitted


def search_inputs(
    frame: pd.DataFrame,
    published: set[str],
    kmd_fields: list[str] | None,
    knobs: list[str],
    metric: str | None,
    executable: str | None,
) -> tuple[list[str], list[str], dict, list]:
    """What a feature search reads, from the training rows: (columns, tensor identifiers it
    may read only as presence flags, guards, omitted).

    `guards` maps each of those that is an optional feature (for a model of `metric`: the
    evaluator's own lint decides which a valid graph may leave unbound and nothing guards)
    to the default a model reads for it, so the search reads it through value_or_default and
    scores the rows that leave it unbound with that default. An optional column that holds
    strings cannot take a default and is left out."""
    scalars, identifiers, legal, omitted = proposal_columns(
        frame, published, kmd_fields, knobs, optional_features(executable)
    )
    columns = [
        name for name in scalars if not name.startswith("kernel.") or name in legal
    ]
    guards = optional_guards([*columns, *identifiers], metric, executable)
    categorical = {
        name
        for name in guards
        if frame[name].map(lambda value: isinstance(value, str)).any()
    }
    for name in sorted(categorical):
        del guards[name]
        omitted.append({"expression": "$" + name, "reason": "optional_categorical"})
    columns = [name for name in columns if name not in categorical]
    omitted += identifier_omissions(identifiers, guards)
    return columns, identifiers, guards, omitted

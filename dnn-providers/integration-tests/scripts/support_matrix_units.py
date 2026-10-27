#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Collection pass of the support matrix renderer: bundle tree -> ``ClaimUnit``.

``render_support_matrix.py`` is the entry point; this module discovers every
support sidecar under the bundle tree, joins each to the graph it claims, and
returns one ``ClaimUnit`` per graph. It knows nothing about output formats --
``support_matrix_markdown.py`` projects the units.

Two on-disk bundle shapes are handled, via the shared discovery mirror in
``bundle_discovery.py``:

- **Template-sweep bundle** — ``graph.template.json`` + ``sweep.json`` and a
  bare ``support.json`` whose claims are keyed per ``cases[].id``.
- **Single-graph bundle** — ``{Name}.json`` and a ``{Name}.support.json`` whose
  claims are keyed directly by ``arch -> [platform]``.

A bundle without a sidecar carries no claims, but it still exists: it counts in
the denominator of every cell on its row and in the numerator of none. Dropping
such bundles from both is what once reported ``Sdpa 12/12`` for a family with 35
graphs, and hid two families that had no sidecars at all.

Bundles move between tiers, so the
tier is an attribute of a graph, never part of its identity. Every bundle path
reads ``<tier>/<op>/<rest>``; a graph is identified by the path below the tier
plus, for a sweep, its case id. The same graph present in two tiers is one
unit listing both tiers, with the claims of both sidecars merged -- counted
once, so moving a case between tiers changes nothing but its ``tiers``. Two
copies that share an identity but resolve to different graphs are an error.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bundle_discovery import (  # noqa: E402
    SWEEP_MANIFEST_NAME,
    SWEEP_SUPPORT_NAME,
    SWEEP_TEMPLATE_NAME,
    discover_bundle_tree,
    graph_for_sidecar,
    is_single_graph_sidecar,
    load_json,
    read_sweep_cases,
    sidecar_for_graph,
)
from support_claim_schema import (  # noqa: E402
    validate_single_graph_schema,
    validate_sweep_schema,
)

# --------------------------------------------------------------------------
# Row-metadata vocabulary
# --------------------------------------------------------------------------

# Layout label for a graph whose rank carries no NCHW-family layout (a rank-2
# matmul operand, say). Kept distinct from the renderer's "none claimed" cell
# so it cannot be misread as "unsupported" inside a layout list.
NO_LAYOUT = "n/a"

# Graph dtype token, as the serializer writes DataType (Common.hpp) -> matrix
# label. One entry per enumerator but UNSET; test_support_features locks it to
# data_types.fbs. FNUZ fp8 keeps its own label: gfx942 runs it, gfx950 runs
# OCP fp8, and the two are not bit-compatible.
DTYPE_LABELS = {
    "bfloat16": "bf16",
    "half": "fp16",
    "float": "fp32",
    "double": "fp64",
    "int4": "int4",
    "int8": "int8",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "boolean": "bool",
    "fp4_e2m1": "fp4e2m1",
    "fp6_e2m3": "fp6e2m3",
    "fp6_e3m2": "fp6e3m2",
    "fp8_e4m3": "fp8e4m3",
    "fp8_e5m2": "fp8e5m2",
    "fp8_e8m0": "fp8e8m0",
    "fp8_e4m3_fnuz": "fp8e4m3_fnuz",
    "fp8_e5m2_fnuz": "fp8e5m2_fnuz",
}

# Layout tokens recognised in case ids and bundle path segments. Keyed by the
# lowercase token, valued by the display label.
LAYOUT_TOKENS = {
    "nc": "NC",
    "cn": "CN",
    "ncl": "NCL",
    "nlc": "NLC",
    "nchw": "NCHW",
    "nhwc": "NHWC",
    "ncdhw": "NCDHW",
    "ndhwc": "NDHWC",
    "bhsd": "BHSD",
    "bshd": "BSHD",
    "sbhd": "SBHD",
}

# Rank -> (channel-first label, channel-last label), for the stride fallback.
RANK_LAYOUTS = {
    3: ("NCL", "NLC"),
    4: ("NCHW", "NHWC"),
    5: ("NCDHW", "NDHWC"),
}

TOKEN_SPLIT_RE = re.compile(r"[_\-./]+")
SCENARIO_TAG_RE = re.compile(r"\[[^\[\]]+\]")


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def report_error(errors: list[str] | None, message: str) -> None:
    """Print a defect that makes the rendered matrix untrustworthy.

    The C++ harness throws on every one of these, so a matrix rendered past
    them would publish claims the tests never load. ``main`` fails the run when
    any were reported; library callers that pass no list still see them.
    """
    print(f"error: {message}", file=sys.stderr)
    if errors is not None:
        errors.append(message)


def write_stdout(document: str) -> None:
    """Write a rendered document to stdout as UTF-8, whatever the locale.

    The documents carry emoji, which a cp1252 Windows console cannot encode,
    so the bytes go to the buffer directly rather than through ``sys.stdout``.
    """
    try:
        sys.stdout.buffer.write(document.encode("utf-8"))
        sys.stdout.buffer.flush()
    except BrokenPipeError:
        # `| head` closing the pipe early is the caller's choice, not a
        # failure. Point the remaining buffered writes at devnull so the
        # interpreter does not report the same broken pipe again at exit.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())


# --------------------------------------------------------------------------
# JSON loading
# --------------------------------------------------------------------------


def read_json_object(path: pathlib.Path, errors: list[str] | None) -> dict | None:
    """The JSON object in ``path``, or None after reporting why there is none.

    A graph or template that parses to anything but an object -- ``null``
    included -- is as unusable as one that does not parse.
    """
    data, error = load_json(path)
    if error is None and not isinstance(data, dict):
        error = "not a JSON object"
    if error is not None:
        report_error(errors, f"{path}: {error}")
        return None
    return data


def load_sidecar(path: pathlib.Path, validate, errors: list[str] | None) -> dict:
    """The ``claims`` object of a sidecar, or ``{}`` if absent or invalid.

    Loading and validation are the verifier's own, so the two tools cannot
    disagree about what a well-formed sidecar is. A sidecar that fails them
    contributes no claims, but its bundle still counts in every denominator.
    """
    if not path.is_file():
        return {}
    data, error = load_json(path)
    if error is not None:
        report_error(errors, f"{path}: {error}")
        return {}
    problems: list[str] = []
    validate(data, path, problems)
    for problem in problems:
        report_error(errors, problem)
    if problems:
        return {}
    return data.get("claims") or {}


def load_sweep_cases(sweep_path: pathlib.Path, errors: list[str] | None):
    """The ``cases`` of a sweep manifest, or None after reporting why not.

    The rules are ``read_sweep_cases``'s, the mirror of readSweepCaseIds():
    whatever it rejects, the harness registers none of the sweep's cases.
    """
    cases, error = read_sweep_cases(sweep_path)
    if error is not None:
        report_error(errors, f"{sweep_path}: {error}")
    return cases


# --------------------------------------------------------------------------
# Template resolution
# --------------------------------------------------------------------------


# The functions below transcribe the sweep-case expansion in
# src/harness/bundle/IntegrationTestBundle.hpp, one function per C++ function,
# so a case reads here as the graph the harness builds from it. Each raises
# ``SweepCaseError`` where its C++ counterpart throws: loadIntegrationTestBundle()
# turns every such throw into INVALID_SWEEP_CASE, and the case never runs.
#
# Not transcribed: warnUnusedSweepValues(). It only logs, so it cannot change
# the graph or whether the case loads.

CASE_PLACEHOLDER_PREFIX = "${case."

# Fields a tensor must take from its own values.tensors[] spec when a
# placeholder for them sits inside it; a top-level value never stands in.
PER_TENSOR_FIELDS = frozenset({"dims", "strides", "data_type"})

# lookup_json_path()'s "not found", distinct from a found JSON null.
_MISSING = object()


class SweepCaseError(ValueError):
    """A sweep case the harness refuses to load (INVALID_SWEEP_CASE)."""


def _is_uid(value) -> bool:
    # nlohmann::json's is_number_integer(): JSON true / false are not integers.
    return isinstance(value, int) and not isinstance(value, bool)


def first_path_token(path: str) -> str:
    """Mirror firstPathToken(): ``a.b.c`` -> ``a``."""
    return path.split(".", 1)[0]


def lookup_json_path(node, path: str):
    """Mirror lookupJsonPath(): the value at dotted ``path``, or ``_MISSING``."""
    current = node
    start = 0
    while start < len(path):
        end = path.find(".", start)
        key = path[start:] if end == -1 else path[start:end]
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
        if end == -1:
            return current
        start = end + 1
    return current


def placeholder_field(node) -> str | None:
    """Mirror placeholderField(): ``path`` if ``node`` is exactly ``${case.path}``.

    Only a whole string is a placeholder; ``"x${case.y}"`` stays a literal.
    """
    if (
        not isinstance(node, str)
        or len(node) <= len(CASE_PLACEHOLDER_PREFIX) + 1
        or not node.startswith(CASE_PLACEHOLDER_PREFIX)
        or not node.endswith("}")
    ):
        return None
    return node[len(CASE_PLACEHOLDER_PREFIX) : -1]


def template_tensor_uids(template: dict) -> set[int]:
    """Mirror collectTemplateTensorUids()."""
    tensors = template.get("tensors")
    if not isinstance(tensors, list):
        return set()
    uids = set()
    for tensor in tensors:
        if not isinstance(tensor, dict) or not _is_uid(tensor.get("uid")):
            raise SweepCaseError("template tensor missing integer uid")
        uids.add(tensor["uid"])
    return uids


def case_tensor_map(values: dict, template_uids: set[int]) -> dict[int, dict]:
    """Mirror buildCaseTensorMap(): a case's values.tensors[] specs by uid."""
    if "tensors" not in values:
        return {}
    specs = values["tensors"]
    if not isinstance(specs, list):
        raise SweepCaseError("values.tensors must be an array")
    by_uid: dict[int, dict] = {}
    for spec in specs:
        if not isinstance(spec, dict) or not _is_uid(spec.get("uid")):
            raise SweepCaseError("case tensor missing integer uid")
        uid = spec["uid"]
        if uid not in template_uids:
            raise SweepCaseError(f"case tensor uid {uid} not present in template graph")
        if uid in by_uid:
            raise SweepCaseError(f"duplicate case tensor uid {uid}")
        by_uid[uid] = spec
    return by_uid


def resolve_placeholder(
    path: str, tensor_uid: int | None, values: dict, case_tensors: dict[int, dict]
):
    """Mirror resolvePlaceholder().

    Inside an object with integer uid N, the case's values.tensors[] spec for
    N is searched first. A ``dims``, ``strides`` or ``data_type`` placeholder
    must be found there; anything else falls back to the top-level values.
    """
    if tensor_uid is not None:
        spec = case_tensors.get(tensor_uid)
        if spec is not None:
            value = lookup_json_path(spec, path)
            if value is not _MISSING:
                return copy.deepcopy(value)
        if first_path_token(path) in PER_TENSOR_FIELDS:
            raise SweepCaseError(
                f"missing per-tensor placeholder value "
                f"'{CASE_PLACEHOLDER_PREFIX}{path}}}' for tensor uid {tensor_uid}"
            )
    value = lookup_json_path(values, path)
    if value is _MISSING:
        raise SweepCaseError(
            f"missing placeholder value '{CASE_PLACEHOLDER_PREFIX}{path}}}'"
        )
    return copy.deepcopy(value)


def expand_template_node(
    node, tensor_uid: int | None, values: dict, case_tensors: dict[int, dict]
):
    """Mirror expandTemplateNode().

    Any object with an integer ``uid`` opens that uid's scope for everything
    beneath it. A resolved value is used as is, never expanded again.
    """
    path = placeholder_field(node)
    if path is not None:
        return resolve_placeholder(path, tensor_uid, values, case_tensors)
    if isinstance(node, list):
        return [
            expand_template_node(item, tensor_uid, values, case_tensors)
            for item in node
        ]
    if isinstance(node, dict):
        if _is_uid(node.get("uid")):
            tensor_uid = node["uid"]
        return {
            key: expand_template_node(value, tensor_uid, values, case_tensors)
            for key, value in node.items()
        }
    return node


def expand_template_graph(template: dict, case: dict) -> dict:
    """Mirror expandTemplateGraph(): the graph a sweep case runs, before patches."""
    values = case.get("values") if isinstance(case.get("values"), dict) else {}
    case_tensors = case_tensor_map(values, template_tensor_uids(template))
    return expand_template_node(template, None, values, case_tensors)


def check_sweep_golden(case: dict) -> None:
    """Mirror resolveSweepGoldenDirectory()'s one throw.

    The directory itself is not needed here, but a malformed ``golden`` costs
    the case as surely as a malformed graph does.
    """
    golden = case.get("golden")
    if golden is None:
        return
    if not isinstance(golden, dict) or not isinstance(golden.get("path"), str):
        raise SweepCaseError("golden.path is required when golden is present")


def apply_tensor_patches(graph: dict, case: dict) -> None:
    """Apply a sweep case's ``tensor_patches``, as applyTensorPatches() does.

    Each patch upserts its ``set`` keys into the tensor with its uid and
    erases its ``remove`` keys. This is how a case makes a scalar runtime
    pass-by-value, so without it the graph read here is not the graph the
    harness runs. Raises ``SweepCaseError`` where the harness throws: a patch
    without an integer uid, or one whose uid no tensor holds.
    """
    patches = case.get("tensor_patches")
    tensors = graph.get("tensors")
    if not isinstance(patches, list) or not isinstance(tensors, list):
        return
    for patch in patches:
        if not isinstance(patch, dict) or not _is_uid(patch.get("uid")):
            raise SweepCaseError("tensor_patch entry is missing an integer uid")
        target = next(
            (
                tensor
                for tensor in tensors
                if isinstance(tensor, dict) and tensor.get("uid") == patch["uid"]
            ),
            None,
        )
        if target is None:
            raise SweepCaseError(
                f"tensor_patch uid {patch['uid']} not found in expanded graph"
            )
        if isinstance(patch.get("set"), dict):
            target.update(patch["set"])
        if isinstance(patch.get("remove"), list):
            for key in patch["remove"]:
                if isinstance(key, str):
                    target.pop(key, None)


# --------------------------------------------------------------------------
# Row-metadata extraction
# --------------------------------------------------------------------------


def dtype_label(raw) -> str | None:
    """Map a graph dtype token to its matrix label."""
    if not isinstance(raw, str) or not raw or raw == "unset":
        return None
    return DTYPE_LABELS.get(raw, raw)


def representative_tensor(graph: dict) -> dict | None:
    """The tensor whose shape best characterises the graph's layout.

    Highest rank wins, then largest element count, then lowest uid — a total
    order, so the choice is deterministic.
    """
    best = None
    best_key = None
    for tensor in graph.get("tensors", []) or []:
        if not isinstance(tensor, dict):
            continue
        dims = tensor.get("dims")
        if not isinstance(dims, list) or not dims:
            continue
        volume = 1
        for dim in dims:
            if isinstance(dim, int):
                volume *= dim
        # Negate uid so that "largest key" still means "lowest uid".
        key = (len(dims), volume, -(tensor.get("uid") or 0))
        if best_key is None or key > best_key:
            best, best_key = tensor, key
    return best


def layout_from_name(name: str) -> str | None:
    """The last layout token in a case id or path fragment, if any."""
    found = None
    for token in TOKEN_SPLIT_RE.split(name.lower()):
        if token in LAYOUT_TOKENS:
            found = LAYOUT_TOKENS[token]
    return found


def layout_from_strides(graph: dict) -> str | None:
    """Fall back to the representative tensor's rank and stride ordering."""
    tensor = representative_tensor(graph)
    if not tensor:
        return None
    dims = tensor.get("dims") or []
    labels = RANK_LAYOUTS.get(len(dims))
    if not labels:
        return None

    strides = tensor.get("strides")
    if not isinstance(strides, list) or len(strides) != len(dims):
        return labels[0]
    # Channels-last packs the channel dim innermost, so its stride drops below
    # the trailing dim's. Ties (C == 1) are reported as channel-first.
    channels_last = strides[1] < strides[-1]
    return labels[1] if channels_last else labels[0]


def pointwise_label(node: dict) -> str:
    """``Pointwise:RELU_FWD[lower_clip,upper_clip]`` for one pointwise node."""
    inputs = node.get("inputs") or {}
    operation = inputs.get("operation")
    if isinstance(operation, str):
        label = f"Pointwise:{operation.upper()}"
    else:
        label = "Pointwise"

    clips = []
    if inputs.get("relu_lower_clip") is not None:
        clips.append("lower_clip")
    if inputs.get("relu_lower_clip_slope") is not None:
        clips.append("lower_clip_slope")
    if inputs.get("relu_upper_clip") is not None:
        clips.append("upper_clip")
    if clips:
        label += "[" + ",".join(clips) + "]"
    return label


def node_label(node: dict) -> str:
    node_type = node.get("type") or "Unknown"
    if node_type == "PointwiseAttributes":
        return pointwise_label(node)
    return node_type_label(node_type)


def _off_neutral(parameters: dict, key: str, neutral: int) -> bool:
    """True when a spatial parameter vector departs from its no-op value."""
    values = parameters.get(key)
    if not isinstance(values, list):
        return False
    return any(isinstance(v, int) and v != neutral for v in values)


# Activation and filter roles across the three convolution directions: forward
# reads x and w, data-grad reads w and writes dx, weight-grad reads x and
# writes dw. Grouping is the same relation between them in every direction.
_CONV_ACTIVATION_KEYS = ("x_tensor_uid", "dx_tensor_uid")
_CONV_FILTER_KEYS = ("w_tensor_uid", "dw_tensor_uid")


def _conv_dims(ports: dict, keys: tuple[str, ...], by_uid: dict) -> list:
    for key in keys:
        if key in ports:
            return (by_uid.get(ports[key]) or {}).get("dims") or []
    return []


def shape_tags_of(graph: dict) -> set[str]:
    """Feature tags read off the resolved graph's shapes and parameters.

    The fusion chain alone does not separate a strided grouped convolution from
    a plain one, yet an engine may well support one and refuse the other. These
    tags split such graphs onto their own rows so a refusal has somewhere to
    show up. Each tag marks a departure from the simplest case, so an untagged
    row means batch 1, unit stride, unit dilation, no padding, no groups.
    """
    tags: set[str] = set()

    tensor = representative_tensor(graph) or {}
    dims = tensor.get("dims") or []
    # Rank 2 has no batch axis to speak of -- dim 0 there is a matmul row count.
    if len(dims) >= 3 and isinstance(dims[0], int) and dims[0] > 1:
        tags.add("multi_batch")

    by_uid = {
        t.get("uid"): t for t in (graph.get("tensors") or []) if isinstance(t, dict)
    }
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        # Convolution nests its spatial parameters under "parameters"; pooling
        # (ResampleFwd) carries the same keys on the node itself.
        for parameters in (node.get("parameters") or {}, node):
            if _off_neutral(parameters, "stride", 1):
                tags.add("stride")
            if _off_neutral(parameters, "dilation", 1):
                tags.add("dilation")
            if _off_neutral(parameters, "pre_padding", 0) or _off_neutral(
                parameters, "post_padding", 0
            ):
                tags.add("padding")

        if not str(node.get("type") or "").startswith("Convolution"):
            continue
        ports = {**(node.get("inputs") or {}), **(node.get("outputs") or {})}
        x_dims = _conv_dims(ports, _CONV_ACTIVATION_KEYS, by_uid)
        w_dims = _conv_dims(ports, _CONV_FILTER_KEYS, by_uid)
        # Grouping shows up as a filter that spans fewer channels than the
        # activation carries; ungrouped convolution has them equal by
        # definition.
        if len(x_dims) >= 2 and len(w_dims) >= 2 and x_dims[1] != w_dims[1]:
            tags.add("grouped")

    return tags


def variant_of(graph: dict, scenario_tags: list[str]) -> str:
    """The fusion pattern and feature tags, relative to the graph's first node."""
    nodes = [n for n in (graph.get("nodes") or []) if isinstance(n, dict)]
    if len(nodes) <= 1:
        variant = "(bare)"
    else:
        variant = "".join(f" + {node_label(node)}" for node in nodes[1:])

    # Path tags arrive bracketed from the regex; shape tags are bare names.
    # Normalise both to bare, then emit one bracket so a graph carrying several
    # tags reads "[grouped,stride]" rather than "[grouped][stride]".
    tags = {t.strip("[]") for t in scenario_tags} | shape_tags_of(graph)
    if tags:
        prefix = "[" + ",".join(sorted(tags)) + "]"
        variant = prefix if variant == "(bare)" else prefix + variant
    return variant


def dtypes_of(graph: dict) -> str:
    """``[io=bf16, compute=fp32, intermediate=fp32]`` for one resolved graph."""
    io = dtype_label(graph.get("io_data_type"))
    if io is None:
        # A graph that leaves io_data_type unset takes it from its data tensors.
        tensor = representative_tensor(graph)
        if tensor:
            io = dtype_label(tensor.get("data_type"))

    parts = []
    if io:
        parts.append(f"io={io}")
    compute = dtype_label(graph.get("compute_data_type"))
    if compute:
        parts.append(f"compute={compute}")
    intermediate = dtype_label(graph.get("intermediate_data_type"))
    if intermediate:
        parts.append(f"intermediate={intermediate}")
    return "[" + ", ".join(parts) + "]" if parts else "[unspecified]"


def scenario_tags_in(relative_path: pathlib.PurePath) -> list[str]:
    """Bracketed ``[tag]`` markers anywhere in the bundle's path."""
    tags = []
    for part in relative_path.parts:
        for tag in SCENARIO_TAG_RE.findall(part):
            if tag not in tags:
                tags.append(tag)
    return sorted(tags)


@dataclass(frozen=True)
class BundlePlace:
    """Where a bundle sits: ``<tier>/<op>/<rest>`` under the bundle root.

    ``location`` is the path below the tier (``<op>/<rest>``). It is what
    identifies a bundle, because moving a bundle to another tier changes the
    tier directory and nothing else about the path.
    """

    tier: str
    op: str
    location: str


def place_of(
    relative_path: pathlib.PurePath, where: pathlib.Path, errors: list[str] | None
) -> BundlePlace | None:
    """Split a bundle directory's path into tier, op and location.

    Neither the tier nor the op is looked up in a list: any first directory
    is a tier, any second an op, so a new tier or op needs no change here. A
    bundle too shallow to have both cannot be placed in the matrix at all.
    """
    parts = relative_path.parts
    if len(parts) < 2:
        report_error(
            errors,
            f"{where}: bundle must sit under <tier>/<op>/ in the bundle tree "
            f"(found '{relative_path.as_posix() or '.'}')",
        )
        return None
    return BundlePlace(
        tier=parts[0],
        op=parts[1],
        location=pathlib.PurePosixPath(*parts[1:]).as_posix(),
    )


def node_type_label(node_type: str) -> str:
    """``BatchnormInferenceAttributesVarianceExt`` -> ``BatchnormInferenceVarianceExt``.

    ``Attributes`` is a schema wart, not part of the op's name, and it is not
    always the trailing token -- so strip it wherever it appears rather than
    only off the end. Every view names an op through this, so one op reads
    the same in the matrix, the feature matrix and the coverage report.
    """
    return node_type.replace("Attributes", "", 1) or node_type


def primary_of(graph: dict) -> str:
    """The graph's first node, which the Variant column's fusion chain hangs off.

    Rows are grouped by the op directory, but that directory does not always
    name the first node -- ``ConvolutionFwdPointwise`` holds graphs that start
    with ``ConvolutionFwd`` -- so the chain needs its own head.
    """
    nodes = [n for n in (graph.get("nodes") or []) if isinstance(n, dict)]
    if not nodes:
        return "(no nodes)"
    return node_type_label(nodes[0].get("type") or "") or "Unknown"


def graph_digest(graph: dict) -> str:
    """A fingerprint of the resolved graph, to tell real duplicates from clashes.

    Taken after template expansion: retiering may rewrite a sweep's template
    and ``values`` while leaving every expanded graph identical.
    """
    canonical = json.dumps(graph, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Claim units
# --------------------------------------------------------------------------


@dataclass
class ClaimUnit:
    """One claimed graph: a sweep case, or a whole single-graph bundle.

    ``bundle`` is the bundle directory *below the tier*, and ``tiers`` lists
    every tier holding a copy. Nothing that identifies, orders or groups a
    unit reads the tier, so moving a bundle between tiers changes ``tiers``
    and nothing else.
    """

    op: str
    primary: str
    variant: str
    dtypes: str
    layout: str
    bundle: str
    case_id: str | None
    tiers: list[str]
    # engine -> {(arch, platform)}
    claims: dict[str, set[tuple[str, str]]] = field(default_factory=dict)
    # Identity and graph fingerprint, for merging cross-tier copies. Neither
    # is rendered.
    key: tuple = ()
    digest: str = ""
    # The resolved graph, for readers that need more than the labels above
    # (support_features/ extracts API features from it). Never rendered, and
    # left out of equality and repr: ``digest`` already stands for it.
    graph: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def label(self) -> str:
        return self.case_id or pathlib.PurePath(self.bundle).name

    @property
    def variant_label(self) -> str:
        """The Variant column: the fusion chain, headed by the first node
        when that node is not the op the row is filed under."""
        if self.primary == self.op:
            return self.variant
        return self.primary + ("" if self.variant == "(bare)" else self.variant)

    def supported_by(self, engine: str, arch: str, platform: str) -> bool:
        return (arch, platform) in self.claims.get(engine, ())

    def sort_key(self) -> tuple:
        return (
            self.op,
            self.variant_label,
            self.dtypes,
            self.layout,
            self.bundle,
            self.case_id or "",
        )


def support_pairs(support: dict) -> set[tuple[str, str]]:
    """Flatten a schema-checked ``{arch: [platform, ...]}`` map into pairs."""
    return {
        (arch, platform)
        for arch, platforms in support.items()
        for platform in platforms
    }


def collect_sweep_units(
    sweep_dir: pathlib.Path, root: pathlib.Path, errors: list[str] | None = None
) -> list[ClaimUnit]:
    """Build one ClaimUnit per case of a template-sweep bundle.

    A bundle with no sidecar still yields units -- claimless ones. Absent a
    sidecar there is nothing to put in a numerator, but the cases are still
    graphs the tree holds, so they belong in the denominator. Dropping them
    would erase a wholly unclaimed family from the document rather than
    showing it as a row of dashes, which is the one thing the matrix exists
    to make visible.
    """
    support_path = sweep_dir / SWEEP_SUPPORT_NAME
    claims = load_sidecar(support_path, validate_sweep_schema, errors)

    template = read_json_object(sweep_dir / SWEEP_TEMPLATE_NAME, errors)
    cases = load_sweep_cases(sweep_dir / SWEEP_MANIFEST_NAME, errors)
    if template is None or cases is None:
        return []

    place = place_of(sweep_dir.relative_to(root), sweep_dir, errors)
    if place is None:
        return []
    tags = scenario_tags_in(pathlib.PurePosixPath(place.location))

    # engine -> case id -> {(arch, platform)}. The schema check above has
    # already guaranteed the shape, and that no case appears twice per engine.
    claims_by_case: dict[str, dict[str, set[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    known_ids = {case["id"] for case in cases}
    for engine, groups in claims.items():
        for group in groups:
            pairs = support_pairs(group["support"])
            for case_id in group["cases"]:
                if case_id not in known_ids:
                    report_error(
                        errors,
                        f"{support_path}: engine '{engine}' claims unknown case "
                        f"'{case_id}' (not in {SWEEP_MANIFEST_NAME})",
                    )
                    continue
                claims_by_case[engine][case_id] |= pairs

    units = []
    for case in cases:
        case_id = case["id"]
        # The order of loadIntegrationTestBundle(), so the first defect
        # reported is the one the harness trips on.
        try:
            graph = expand_template_graph(template, case)
            check_sweep_golden(case)
            apply_tensor_patches(graph, case)
        except SweepCaseError as error:
            report_error(
                errors,
                f"{sweep_dir / SWEEP_MANIFEST_NAME}: case '{case_id}': {error}; "
                "the harness skips it, so it never runs",
            )
            continue

        layout = layout_from_name(case_id) or layout_from_strides(graph) or NO_LAYOUT
        unit = ClaimUnit(
            op=place.op,
            primary=primary_of(graph),
            variant=variant_of(graph, tags),
            dtypes=dtypes_of(graph),
            layout=layout,
            bundle=place.location,
            case_id=case_id,
            tiers=[place.tier],
            key=("sweep", place.location, case_id),
            digest=graph_digest(graph),
            graph=graph,
        )
        for engine, per_case in claims_by_case.items():
            pairs = per_case.get(case_id)
            if pairs:
                unit.claims[engine] = set(pairs)
        units.append(unit)
    return units


def collect_single_graph_unit(
    graph_path: pathlib.Path, root: pathlib.Path, errors: list[str] | None = None
) -> ClaimUnit | None:
    """Build the single ClaimUnit of a single-graph bundle.

    Claimless when the bundle has no sidecar; see ``collect_sweep_units`` for
    why that is a unit rather than nothing.
    """
    support_path = sidecar_for_graph(graph_path)
    claims = load_sidecar(support_path, validate_single_graph_schema, errors)

    graph = read_json_object(graph_path, errors)
    if graph is None:
        return None

    place = place_of(graph_path.parent.relative_to(root), graph_path, errors)
    if place is None:
        return None
    location = pathlib.PurePosixPath(place.location)
    tags = scenario_tags_in(location)

    # The layout hint lives in the bundle path for a single-graph bundle; the
    # graph's own strides are the fallback. Read from below the tier, so a
    # tier name can never be mistaken for a layout token.
    layout = layout_from_name(place.location) or layout_from_strides(graph) or NO_LAYOUT
    unit = ClaimUnit(
        op=place.op,
        primary=primary_of(graph),
        variant=variant_of(graph, tags),
        dtypes=dtypes_of(graph),
        layout=layout,
        bundle=place.location,
        case_id=None,
        tiers=[place.tier],
        key=("single", place.location, graph_path.name),
        digest=graph_digest(graph),
        graph=graph,
    )
    for engine, arch_map in claims.items():
        pairs = support_pairs(arch_map)
        if pairs:
            unit.claims[engine] = pairs
    return unit


# A graph the schema check cannot see into -- a uid that is a string, an
# ``inputs`` that is a list -- surfaces as one of these while its metadata is
# read. It is a defect in that one bundle, not a reason to abandon the rest.
BUNDLE_READ_ERRORS = (TypeError, AttributeError, KeyError, ValueError, RecursionError)


def check_orphaned_sidecars(
    files: list[pathlib.Path],
    sweep_roots: list[pathlib.Path],
    graph_files: list[pathlib.Path],
    errors: list[str] | None,
) -> None:
    """Report every sidecar no discovered bundle reads.

    Units are built from bundles, and a bundle finds its own sidecar; a
    sidecar left behind by a half-finished move therefore has nothing to pull
    it in. Its claims would vanish from the matrix with no trace, so it is
    reported instead.
    """
    sweep_set = set(sweep_roots)
    graph_set = set(graph_files)
    for path in files:
        if path.name == SWEEP_SUPPORT_NAME:
            if path.parent not in sweep_set:
                report_error(
                    errors,
                    f"{path}: orphaned {SWEEP_SUPPORT_NAME} -- its directory is "
                    "not a sweep root, so no bundle reads these claims",
                )
        elif is_single_graph_sidecar(path):
            if any(parent in sweep_set for parent in path.parents):
                continue
            graph = graph_for_sidecar(path)
            if graph not in graph_set:
                report_error(
                    errors,
                    f"{path}: orphaned sidecar -- no sibling graph {graph.name}, "
                    "so no bundle reads these claims",
                )


def merge_units(
    units: list[ClaimUnit], errors: list[str] | None = None
) -> list[ClaimUnit]:
    """Fold copies of one graph held in several tiers into a single unit.

    A copy is a unit with the same identity -- the same path below the tier,
    plus the same case id for a sweep. Copies must hold the same resolved
    graph; if they do not, the tree has two different graphs under one name
    and which one a reader is shown would depend on walk order, so that is
    reported rather than guessed. Claims are unioned, and every tier is kept.
    """
    merged: dict[tuple, ClaimUnit] = {}
    for unit in units:
        first = merged.get(unit.key)
        if first is None:
            merged[unit.key] = unit
            continue
        if unit.digest != first.digest:
            where = unit.bundle + (f" case '{unit.case_id}'" if unit.case_id else "")
            report_error(
                errors,
                f"{where}: the copies in tiers {', '.join(first.tiers)} and "
                f"{', '.join(unit.tiers)} hold different graphs under the same "
                "identity",
            )
            continue
        for engine, pairs in unit.claims.items():
            first.claims.setdefault(engine, set()).update(pairs)
        first.tiers = sorted(set(first.tiers) | set(unit.tiers))
    return list(merged.values())


def collect_units(
    root: pathlib.Path, errors: list[str] | None = None
) -> list[ClaimUnit]:
    sweep_roots, graph_files, files = discover_bundle_tree(root)
    check_orphaned_sidecars(files, sweep_roots, graph_files, errors)
    units: list[ClaimUnit] = []
    for sweep_dir in sweep_roots:
        try:
            units.extend(collect_sweep_units(sweep_dir, root, errors))
        except BUNDLE_READ_ERRORS as exc:
            report_error(errors, f"{sweep_dir}: cannot read bundle ({exc!r})")
    for graph_path in graph_files:
        try:
            unit = collect_single_graph_unit(graph_path, root, errors)
        except BUNDLE_READ_ERRORS as exc:
            report_error(errors, f"{graph_path}: cannot read bundle ({exc!r})")
            continue
        if unit is not None:
            units.append(unit)
    return merge_units(units, errors)

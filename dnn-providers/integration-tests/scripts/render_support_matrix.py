#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Render the combined engine support matrix from committed claim sidecars.

Implements the renderer described by RFC 0015 §11
(``projects/hipdnn/docs/rfcs/0015_EngineSupportClaims.md``): it discovers every
support sidecar under the bundle tree, joins each to the graph it claims, and
projects the result.

The matrix is a *view* of the committed claims, never a second state to
maintain. Generation is hardware-free: only ``.json`` / ``.support.json`` files
are read, so a plain checkout with no GPU, no plugin, and no DVC pull can
regenerate it.

Two projections come off one collection pass, so they cannot disagree:

- ``--format markdown`` (default) — the human map. Four zoom levels: per-target
  overview, per-op variant table, per-(variant, dtype) detail, and HTML
  traceability comments naming the bundle each row came from.
- ``--format json`` — the machine index. One flat record per claimed graph with
  every field the markdown aggregates away. This is the layer to build a
  browsable viewer on; markdown is not a queryable format.

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

Bundles move between tiers (``migration-scripts/retier_bundles.py``), so the
tier is an attribute of a graph, never part of its identity. Every bundle path
reads ``<tier>/<op>/<rest>``; a graph is identified by the path below the tier
plus, for a sweep, its case id. The same graph present in two tiers is one
unit listing both tiers, with the claims of both sidecars merged -- counted
once, so moving a case between tiers changes nothing but its ``tiers``. Two
copies that share an identity but resolve to different graphs are an error.

The rendered document is **generated, not committed** -- see
``dnn-providers/integration-tests/.gitignore`` for why, and RFC 0015 §11.

Usage:

    python3 dnn-providers/integration-tests/scripts/render_support_matrix.py
"""

from __future__ import annotations

import argparse
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
)
from verify_support_claims import (  # noqa: E402
    validate_single_graph_schema,
    validate_sweep_schema,
)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

SCRIPT_PATH = pathlib.Path(__file__).resolve()
INTEGRATION_TESTS_DIR = SCRIPT_PATH.parent.parent
REPO_ROOT = INTEGRATION_TESTS_DIR.parent.parent
DEFAULT_BUNDLES_DIR = INTEGRATION_TESTS_DIR / "integration-test-bundles"
DEFAULT_MATRIX_PATH = INTEGRATION_TESTS_DIR / "SUPPORT_MATRIX.md"
DEFAULT_OVERVIEW_PATH = INTEGRATION_TESTS_DIR / "SUPPORT_MATRIX_OVERVIEW.md"
DEFAULT_GROUPS_PATH = SCRIPT_PATH.parent / "summary_groups.json"

# No --output: it defaults to DEFAULT_MATRIX_PATH, so the shortest command that
# reproduces the document is the one worth quoting back to the reader.
REGEN_COMMAND = f"python3 {SCRIPT_PATH.relative_to(REPO_ROOT).as_posix()}"
REGEN_OVERVIEW_COMMAND = f"{REGEN_COMMAND} --overview-only"
REGEN_JSON_COMMAND = f"{REGEN_COMMAND} --format json"

# --------------------------------------------------------------------------
# Presentation constants
# --------------------------------------------------------------------------

FULL = "✅"
PARTIAL = "🟡"
NONE = "—"

# An empty cell whose engine never reported on that target: the sidecars hold
# no claim at all from it there, for any graph, so its tests most likely did
# not run. Kept distinct from NONE, which is an engine that ran and claimed
# none of the row's graphs.
NOT_RUN = "not run"

# Layout label for a graph whose rank carries no NCHW-family layout (a rank-2
# matmul operand, say). Kept distinct from NONE so it cannot be misread as
# "unsupported" inside a layout list.
NO_LAYOUT = "n/a"

# Header of the column counting a graph once if any engine claims it.
ANY_ENGINE = "Any engine"

# Joins a summary cell's mark to its count, so the pair never wraps.
NBSP = "&nbsp;"

# Short key at the top — just enough to read the tables. The full reference
# (what a claim means, how variants are tagged, how tiers merge) goes at the
# end so the reader hits the data first.
LEGEND_DETAIL = (
    "Expand a row for variant and per-dtype detail. "
    "Full legend at the [end of the document](#reading-guide)."
)

_LEGEND_CELLS = (
    f"`{FULL} 108/108` all graphs claimed · "
    f"`{PARTIAL} 72/108` 72 of 108 claimed · "
    f"`{NONE}` none claimed (not the same as *known unsupported*) · "
    f"`{NOT_RUN}` the engine has no claim at all on this target, so its tests "
    "most likely did not run there"
)

LEGEND_OVERVIEW = (
    f"{_LEGEND_CELLS}\n"
    "\n"
    "A box is one op (row) × engine (column) on this target: claimed graphs "
    "out of all the op's graphs, with each template-sweep case counted as one "
    "graph. A claim is a CI pass recorded in a committed sidecar, so new "
    "passes show up only after a harvest.\n"
    "\n"
    f"**{ANY_ENGINE}** counts a graph once if at least one engine claims it, "
    "so it is the union of the engine columns, not their sum. The summary tables "
    f"roll the {ANY_ENGINE} counts up by op group, one table per arch group; "
    "there a cell shows only the claimed count, out of the row's **Graphs**."
)

_LEGEND_PREFIX = f"{_LEGEND_CELLS}. "


def _legend(overview_only: bool) -> list[str]:
    if overview_only:
        return [LEGEND_OVERVIEW, ""]
    return [_LEGEND_PREFIX + LEGEND_DETAIL, ""]


# The full reference, rendered after the last target section.
READING_GUIDE = [
    "## Reading guide",
    "",
    "_Each overview row is one op directory of the bundle tree. Open a "
    "disclosure triangle to expand it into per-variant rows, and the one "
    "inside that for full per-(variant, dtype) rows. Every level counts the "
    "same way, so an op's "
    "variant counts sum to its overview count and a variant's dtype counts sum "
    "to the variant._",
    "",
    "### Cell vocabulary",
    "",
    "| Cell | Meaning |",
    "|------|---------|",
    f"| `{FULL} 108/108` | every graph in the row is claimed by that engine "
    "on that target |",
    f"| `{PARTIAL} 72/108` | 72 of the row's 108 graphs are claimed |",
    f"| `{PARTIAL} 8/25 NCHW, NHWC` | the same count, plus the layouts the "
    "claimed graphs cover |",
    f"| `{NONE}` | no claim recorded, though the engine did report on that " "target |",
    f"| `{NOT_RUN}` | no claim recorded, and the engine has none at all on that "
    f"target — for {ANY_ENGINE} and the summary, none of the engines that "
    "claim the row anywhere has. The sidecars record claims, not runs, so this "
    "is inferred: an engine whose tests ran claims something |",
    f"| `{NO_LAYOUT}` | the graph's rank carries no NCHW-family layout "
    "(a rank-2 matmul operand, say) |",
    "",
    "**Denominator.** Every graph the bundle tree holds for that row: a "
    "single-graph bundle counts once, a template sweep counts once per case. "
    "It does not depend on the target, so the same row shows the same "
    "denominator in every `arch / platform` section — only the numerator "
    "moves.",
    "",
    f"**{ANY_ENGINE}.** The first column counts a graph once if at least one "
    "engine claims it on that target: the union of the engine columns, never "
    "their sum. The summary table at the top adds those counts up by op group "
    "(`scripts/summary_groups.json`), in one table per arch group; every graph "
    "sits under exactly one op, so a "
    "group's cell is the plain sum of its ops' cells.",
    "",
    "**A claim is an assertion, not a measurement taken here.** Each one is an "
    "entry in a `.support.json` sidecar (RFC 0015 §4) saying the engine "
    "supports that exact graph, and the integration tests enforce it. So "
    f"`{NONE}` — and the unclaimed share of a `{PARTIAL}` — means *unclaimed*, "
    "which is not the same as *known unsupported*: the engine may genuinely "
    "reject those graphs, or no one has claimed them yet.",
    "",
    "**Targets.** Sidecars key claims by gfx target, so that is what the "
    "section headings use; the MI series name after the dash is a reading aid "
    "and matches nothing. One gfx target covers a whole series, so a claim on "
    "`gfx90a` is a claim for every MI200 part.",
    "",
    "**Variants** read `[feature tags] + fusion chain`: whatever is fused onto "
    "the graph's first node. Where that node is not the op the row is filed "
    "under (`ConvolutionFwdPointwise` graphs start with `ConvolutionFwd`), the "
    "variant names it first. The tags each mark a "
    "departure from the simplest case, so an *untagged* row means batch 1, "
    "unit stride, unit dilation, no padding and no groups:",
    "",
    "| Tag | Read off |",
    "|-----|----------|",
    "| `multi_batch` | leading dimension above 1 (rank 3 and up — a rank-2 "
    "operand has no batch axis) |",
    "| `grouped` | filter spanning fewer channels than the input carries |",
    "| `stride` | any stride other than 1 |",
    "| `dilation` | any dilation other than 1 |",
    "| `padding` | any non-zero pre- or post-padding |",
    "",
    "They exist because the fusion chain alone cannot separate a strided "
    "grouped convolution from a plain one, so an engine that takes the second "
    "and refuses the first would have nowhere to say so.",
    "",
    "**Tiers are merged.** A row can draw on `quick`, `standard`, "
    "`comprehensive`, and `full` bundles at once. A graph is identified by "
    "its path below the tier directory (plus its case id, for a sweep), so "
    "the same graph committed to two tiers counts once, with both sidecars' "
    "claims merged. The bundle paths and tiers behind each row are in HTML "
    "comments in the appendix below, "
    "recorded once because provenance does not vary by target; "
    "`--format json` emits the same data per case with nothing aggregated "
    "away.",
    "",
]

# Graph dtype token -> matrix label.
DTYPE_LABELS = {
    "bfloat16": "bf16",
    "half": "fp16",
    "float": "fp32",
    "double": "fp64",
    "int8": "int8",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "boolean": "bool",
    "e4m3_fnuz": "fp8e4m3",
    "e5m2_fnuz": "fp8e5m2",
    "fp8_e4m3": "fp8e4m3",
    "fp8_e5m2": "fp8e5m2",
    "fp8_e8m0": "fp8e8m0",
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
TEMPLATE_VAR_RE = re.compile(r"^\$\{case\.([A-Za-z0-9_.]+)\}$")
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


# --------------------------------------------------------------------------
# JSON loading
# --------------------------------------------------------------------------


def load_json(path: pathlib.Path, errors: list[str] | None = None):
    """Parse ``path`` as JSON, reporting an error and returning None on failure.

    ``ValueError`` covers both a JSON syntax error and a file that is not UTF-8;
    ``RecursionError`` is what a pathologically nested document raises.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, RecursionError) as exc:
        report_error(errors, f"{path}: failed to parse JSON ({exc})")
        return None


def load_sidecar(path: pathlib.Path, validate, errors: list[str] | None) -> dict:
    """The ``claims`` object of a sidecar, or ``{}`` if absent or invalid.

    Validation is the verifier's own, so the two tools cannot disagree about
    what a well-formed sidecar is. A sidecar that fails it contributes no
    claims, but its bundle still counts in every denominator.
    """
    if not path.is_file():
        return {}
    data = load_json(path, errors)
    if data is None:
        return {}
    problems: list[str] = []
    validate(data, path, problems)
    for problem in problems:
        report_error(errors, problem)
    if problems:
        return {}
    return data.get("claims") or {}


def read_sweep_cases(sweep_path: pathlib.Path, errors: list[str] | None):
    """The ``cases`` of a sweep manifest, or None if the manifest is unusable.

    Mirrors readSweepCaseIds() in BundleDiscovery.hpp, which throws on every
    condition reported here: the harness registers none of the sweep's cases.
    """
    sweep = load_json(sweep_path, errors)
    if sweep is None:
        return None
    if not isinstance(sweep, dict) or not isinstance(sweep.get("cases"), list):
        report_error(errors, f"{sweep_path}: missing cases[] array")
        return None
    seen: set[str] = set()
    for index, case in enumerate(sweep["cases"]):
        case_id = case.get("id") if isinstance(case, dict) else None
        if not isinstance(case_id, str):
            report_error(errors, f"{sweep_path}: case {index} has no string id")
            return None
        if not case_id:
            report_error(errors, f"{sweep_path}: case {index} has an empty id")
            return None
        if case_id in seen:
            report_error(errors, f"{sweep_path}: duplicate case id '{case_id}'")
            return None
        seen.add(case_id)
    return sweep["cases"]


# --------------------------------------------------------------------------
# Template resolution
# --------------------------------------------------------------------------


def lookup_case_value(values: dict, dotted: str):
    """Resolve a dotted ``${case.a.b}`` path against a sweep case's values."""
    node = values
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def resolve_templates(node, values: dict):
    """Recursively substitute ``${case.x}`` placeholders from case values.

    An unresolved placeholder collapses to None, matching the "attribute not
    set" reading used everywhere else in the bundle format.
    """
    if isinstance(node, str):
        match = TEMPLATE_VAR_RE.match(node)
        if match:
            return lookup_case_value(values, match.group(1))
        return node
    if isinstance(node, dict):
        return {k: resolve_templates(v, values) for k, v in node.items()}
    if isinstance(node, list):
        return [resolve_templates(v, values) for v in node]
    return node


def apply_case_tensors(graph: dict, values: dict) -> None:
    """Overlay a sweep case's per-uid tensor specs onto the resolved graph."""
    by_uid = {}
    for spec in values.get("tensors", []) or []:
        if isinstance(spec, dict) and "uid" in spec:
            by_uid[spec["uid"]] = spec

    for tensor in graph.get("tensors", []) or []:
        if not isinstance(tensor, dict):
            continue
        spec = by_uid.get(tensor.get("uid"))
        if not spec:
            continue
        for key in ("dims", "strides", "data_type"):
            if key in spec:
                tensor[key] = spec[key]


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


def short_node_type(node_type: str) -> str:
    if node_type.endswith("Attributes"):
        return node_type[: -len("Attributes")]
    return node_type


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
    return short_node_type(node_type)


def _off_neutral(parameters: dict, key: str, neutral: int) -> bool:
    """True when a convolution parameter vector departs from its no-op value."""
    values = parameters.get(key)
    if not isinstance(values, list):
        return False
    return any(isinstance(v, int) and v != neutral for v in values)


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
        parameters = node.get("parameters") or {}
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
        inputs = node.get("inputs") or {}
        x = by_uid.get(inputs.get("x_tensor_uid")) or {}
        w = by_uid.get(inputs.get("w_tensor_uid")) or {}
        x_dims, w_dims = x.get("dims") or [], w.get("dims") or []
        # Grouping shows up as a filter that spans fewer channels than the
        # input carries; ungrouped convolution has them equal by definition.
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
    identifies a bundle, because ``retier_bundles.py`` moves bundles between
    tier directories and nothing else about the path changes.
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
    only off the end.
    """
    return node_type.replace("Attributes", "") or node_type


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


def target_id(pair: tuple[str, str]) -> str:
    """``("gfx942", "linux") -> "gfx942/linux"``, matching the section heading."""
    return f"{pair[0]}/{pair[1]}"


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

    def as_record(self) -> dict:
        """The flat form used by ``--format json``.

        Targets are ``"arch/platform"`` tokens rather than nested objects:
        every unit repeats them, so the split form costs several megabytes
        across the tree to say nothing extra. ``targets`` at the document root
        expands each token once.
        """
        return {
            "op": self.op,
            "primary": self.primary,
            "variant": self.variant,
            "dtypes": self.dtypes,
            "layout": self.layout,
            "bundle": self.bundle,
            "case_id": self.case_id,
            "tiers": list(self.tiers),
            "claims": {
                engine: [target_id(pair) for pair in sorted(pairs)]
                for engine, pairs in sorted(self.claims.items())
            },
        }


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

    template = load_json(sweep_dir / SWEEP_TEMPLATE_NAME, errors)
    cases = read_sweep_cases(sweep_dir / SWEEP_MANIFEST_NAME, errors)
    if template is not None and not isinstance(template, dict):
        report_error(errors, f"{sweep_dir / SWEEP_TEMPLATE_NAME}: not a JSON object")
    if not isinstance(template, dict) or cases is None:
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
        values = case.get("values") if isinstance(case.get("values"), dict) else {}

        graph = resolve_templates(template, values)
        apply_case_tensors(graph, values)

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
    support_path = graph_path.with_name(f"{graph_path.stem}.support.json")
    claims = load_sidecar(support_path, validate_single_graph_schema, errors)

    graph = load_json(graph_path, errors)
    if graph is None:
        return None
    if not isinstance(graph, dict):
        report_error(errors, f"{graph_path}: not a JSON object")
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
        elif path.name.endswith(".support.json"):
            if any(parent in sweep_set for parent in path.parents):
                continue
            graph = path.with_name(path.name[: -len(".support.json")] + ".json")
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


# --------------------------------------------------------------------------
# Cell rendering
# --------------------------------------------------------------------------


def count_cell(supported: int, total: int) -> str:
    if total == 0 or supported == 0:
        return NONE
    if supported == total:
        return f"{FULL} {supported}/{total}"
    return f"{PARTIAL} {supported}/{total}"


def variant_sort_key(variant: str) -> tuple[int, str]:
    """Alphabetical, with the unfused graph pulled to the front.

    Fused variants all begin with ``" + "``, so a plain sort buries ``(bare)``
    -- the one row a reader scanning a family wants first -- under them.
    """
    return (1, variant) if variant != "(bare)" else (0, variant)


def layout_sort_key(layout: str) -> tuple[int, str]:
    """Alphabetical, with the "no layout" label sorted last."""
    return (1, layout) if layout == NO_LAYOUT else (0, layout)


def layout_cell(units: list[ClaimUnit], engine: str, arch: str, platform: str) -> str:
    """Supported layouts for a row, behind the same count the overview uses.

    The layouts cannot carry the row on their own. A family with one layout
    throughout -- Sdpa is BHSD everywhere -- renders every row identically, so
    a variant claimed 4 of 5 and one claimed 8 of 15 both read ``PARTIAL BHSD``
    and the reader cannot tell a nearly-complete row from a mostly-empty one.

    Leading with the count also makes the zoom levels checkable against each
    other: a family's variant counts must sum to its overview count, and a
    variant's dtype counts must sum to the variant. A cell that disagrees with
    its own children is then visible rather than merely wrong.
    """
    supported = [u for u in units if u.supported_by(engine, arch, platform)]
    if not supported:
        return NONE
    layouts = sorted({u.layout for u in supported}, key=layout_sort_key)
    return f"{count_cell(len(supported), len(units))} {', '.join(layouts)}"


def summary_cell(units: list[ClaimUnit], engine: str, arch: str, platform: str) -> str:
    supported = sum(1 for u in units if u.supported_by(engine, arch, platform))
    return count_cell(supported, len(units))


def claimed_by_any(unit: ClaimUnit, arch: str, platform: str) -> bool:
    return any((arch, platform) in pairs for pairs in unit.claims.values())


def any_summary_cell(units: list[ClaimUnit], arch: str, platform: str) -> str:
    """The row's graphs that at least one engine claims: a union, so two
    engines claiming the same graph count it once."""
    supported = sum(1 for u in units if claimed_by_any(u, arch, platform))
    return count_cell(supported, len(units))


def any_layout_cell(units: list[ClaimUnit], arch: str, platform: str) -> str:
    """``layout_cell`` for the union of the engines."""
    supported = [u for u in units if claimed_by_any(u, arch, platform)]
    if not supported:
        return NONE
    layouts = sorted({u.layout for u in supported}, key=layout_sort_key)
    return f"{count_cell(len(supported), len(units))} {', '.join(layouts)}"


# --------------------------------------------------------------------------
# Not run versus unclaimed
# --------------------------------------------------------------------------

Lane = tuple[str, tuple[str, str]]


def lanes_of(units: list[ClaimUnit]) -> set[Lane]:
    """Every (engine, target) holding at least one claim, on any graph.

    The sidecars record claims, not runs, so this is the evidence a lane ran:
    a run that claims anything at all claims something. An engine whose tests
    did not run on a target -- its CI job timed out, printed no summary, or is
    not scheduled there -- leaves no claim on it anywhere.
    """
    return {
        (engine, pair)
        for u in units
        for engine, pairs in u.claims.items()
        for pair in pairs
    }


def engines_claiming(units: list[ClaimUnit]) -> set[str]:
    """Engines claiming any of ``units``, on any target."""
    return {engine for u in units for engine, pairs in u.claims.items() if pairs}


def mark_not_run(
    cell: str, engines: set[str], target: tuple[str, str], lanes: set[Lane] | None
) -> str:
    """``cell``, or NOT_RUN if it is empty because none of ``engines`` ran on
    ``target``.

    An empty cell whose engines did run stays NONE: they ran and claimed none
    of the row. So does a row no engine claims anywhere -- with nothing that
    could have claimed it, not running loses nothing.
    """
    if cell != NONE or lanes is None or not engines:
        return cell
    if any((engine, target) in lanes for engine in engines):
        return cell
    return NOT_RUN


# --------------------------------------------------------------------------
# Summary groups
# --------------------------------------------------------------------------

# Title of the summary table for targets whose arch no arch group lists.
OTHER_ARCH_GROUP = "Other"


@dataclass
class SummaryGroups:
    """The summary's layout, from ``summary_groups.json``: one table per arch
    group, one row per op group, each as ``[(name, [member, ...]), ...]`` in
    display order. Display data only -- it arranges cells and changes no
    count."""

    ops: list[tuple[str, list[str]]] = field(default_factory=list)
    archs: list[tuple[str, list[str]]] = field(default_factory=list)


def _named_lists(
    document: dict, key: str, members: str, path: pathlib.Path, errors
) -> list[tuple[str, list[str]]]:
    """``document[key]`` as ``[(name, [member, ...]), ...]``.

    A member listed under two names is an error: an op would have its graphs
    counted in two rows, an arch its targets shown in two tables.
    """
    grouped: list[tuple[str, list[str]]] = []
    owner: dict[str, str] = {}
    for entry in document.get(key, []):
        name, listed = entry["name"], list(entry[members])
        for member in listed:
            if member in owner:
                report_error(
                    errors,
                    f"{path}: {member!r} is listed under both "
                    f"{owner[member]!r} and {name!r}",
                )
            owner[member] = name
        grouped.append((name, listed))
    return grouped


def load_summary_groups(
    path: pathlib.Path, errors: list[str] | None = None
) -> SummaryGroups:
    document = load_json(path, errors)
    if document is None:
        return SummaryGroups()
    return SummaryGroups(
        ops=_named_lists(document, "op_groups", "ops", path, errors),
        archs=_named_lists(document, "arch_groups", "archs", path, errors),
    )


def group_targets(
    targets: list[tuple[str, str]], arch_groups: list[tuple[str, list[str]]]
) -> list[tuple[str | None, list[tuple[str, str]]]]:
    """The targets split into summary tables, one per arch group.

    Groups keep their listed order and their targets the order given; a group
    with no target present is skipped. Targets of an arch no group lists go in
    a trailing ``Other`` table rather than vanishing. With no arch groups at
    all, every target shares one untitled table.
    """
    if not arch_groups:
        return [(None, list(targets))]
    tables: list[tuple[str | None, list[tuple[str, str]]]] = []
    listed: set[str] = set()
    for name, archs in arch_groups:
        listed.update(archs)
        held = [t for t in targets if t[0] in archs]
        if held:
            tables.append((name, held))
    rest = [t for t in targets if t[0] not in listed]
    if rest:
        tables.append((OTHER_ARCH_GROUP, rest))
    return tables


def group_ops(
    ops: list[str], groups: list[tuple[str, list[str]]] | None
) -> list[tuple[str, list[str]]]:
    """The present ops, grouped for the summary table.

    Groups keep their listed order and drop ops the tree does not hold; a
    group left with none is skipped. Every op no group lists becomes a row of
    its own after them, so a newly added op directory shows up rather than
    vanishing from the totals.
    """
    present = set(ops)
    grouped = []
    listed: set[str] = set()
    for name, members in groups or []:
        held = [op for op in members if op in present]
        listed.update(members)
        if held:
            grouped.append((name, held))
    grouped.extend((op, [op]) for op in sorted(present - listed))
    return grouped


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------


def traceability_comment(
    row: str, units: list[ClaimUnit], max_case_ids: int
) -> list[str]:
    """One HTML comment per row, listing every bundle that fed it.

    Invisible when rendered, but it is what makes every aggregated row lead
    back to the exact bundle directory that produced it -- and from there to
    the case ids, via that bundle's ``sweep.json``. The ids are deliberately
    *not* inlined by default: enumerating all of them costs several times the
    size of the visible document and pushes it past the point where GitHub
    renders markdown at all. Pass ``--max-case-ids`` to inline them anyway, or
    use ``--format json`` for the complete per-case index.

    One comment per row rather than per (row, bundle): a row draws on 2.7
    bundles on average, and repeating its label -- now long enough to carry a
    tag list and a clipped pointwise chain -- once per bundle spent most of the
    comment bytes saying nothing new, on a document with a hard size ceiling.

    Emitted as a block *after* the table it annotates: a comment line between
    table rows would terminate the table in GitHub-flavored markdown.
    """
    # Grouped by bundle alone, with the union of its tiers: a sweep whose cases
    # are split across tiers is still one entry, so retiering a case changes
    # the tier list and nothing else. ``--format json`` has per-case tiers.
    by_bundle: dict[str, list[ClaimUnit]] = defaultdict(list)
    for unit in units:
        by_bundle[unit.bundle].append(unit)

    entries = []
    for bundle in sorted(by_bundle):
        members = by_bundle[bundle]
        tiers = sorted({tier for u in members for tier in u.tiers})
        entry = f"{bundle} ({'+'.join(tiers)}, {len(members)} case(s))"
        if max_case_ids != 0:
            labels = sorted(u.label for u in members)
            if 0 < max_case_ids < len(labels):
                shown = ", ".join(labels[:max_case_ids])
                entry += f": {shown}, … +{len(labels) - max_case_ids} more"
            else:
                entry += ": " + ", ".join(labels)
        entries.append(entry)

    if not entries:
        return []
    return [f"<!-- row: {row} | bundles: " + "; ".join(entries) + " -->"]


def detail_row_label(primary: str, variant: str, dtypes: str) -> str:
    """``Batchnorm[multi_batch] + Pointwise:RELU_FWD [io=fp16, ...]``.

    Shared with the traceability appendix so a row and its provenance entry are
    keyed by the same string.
    """
    suffix = "" if variant == "(bare)" else variant
    return f"{primary}{suffix} {dtypes}"


def group_detail_rows(
    units: list[ClaimUnit],
) -> list[tuple[str, list[ClaimUnit]]]:
    """(row label, units) per (variant, dtype) row, in display order."""
    grouped: dict[tuple[str, str], list[ClaimUnit]] = defaultdict(list)
    for unit in units:
        grouped[(unit.variant_label, unit.dtypes)].append(unit)
    rows = []
    for key in sorted(grouped, key=lambda key: (variant_sort_key(key[0]), key[1])):
        members = grouped[key]
        # Every member shares the variant label, and with it the head node
        # and fusion chain the row label spells out.
        label = detail_row_label(members[0].primary, members[0].variant, key[1])
        rows.append((label, members))
    return rows


def row_cells(
    rows: list[ClaimUnit],
    engines: list[str],
    target: tuple[str, str],
    lanes: set[Lane] | None,
    *,
    with_layouts: bool,
) -> list[str]:
    """A table row's cells: the Any engine union first, then each engine,
    with an empty cell marked NOT_RUN where its engines never reported."""
    if with_layouts:
        first = any_layout_cell(rows, *target)
        rest = [layout_cell(rows, e, *target) for e in engines]
    else:
        first = any_summary_cell(rows, *target)
        rest = [summary_cell(rows, e, *target) for e in engines]
    cells = [mark_not_run(first, engines_claiming(rows), target, lanes)]
    cells += [mark_not_run(c, {e}, target, lanes) for c, e in zip(rest, engines)]
    return cells


def render_detail(
    lines: list[str],
    units: list[ClaimUnit],
    engines: list[str],
    arch: str,
    platform: str,
    lanes: set[Lane] | None = None,
) -> None:
    """The innermost drill-down: one row per (variant, dtype)."""
    lines.append("<details>")
    lines.append("<summary>🔎 per-(variant, dtype) detail</summary>")
    lines.append("")
    lines.append(f"| Operations | {ANY_ENGINE} | " + " | ".join(engines) + " |")
    lines.append("|" + "---|" * (len(engines) + 2))

    for label, rows in group_detail_rows(units):
        cells = row_cells(rows, engines, (arch, platform), lanes, with_layouts=True)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("</details>")


def render_family(
    lines: list[str],
    op: str,
    units: list[ClaimUnit],
    engines: list[str],
    arch: str,
    platform: str,
    lanes: set[Lane] | None = None,
) -> None:
    lines.append("<details>")
    lines.append(f"<summary>📂 <b>{op}</b></summary>")
    lines.append("")
    lines.append(f"| Variant | {ANY_ENGINE} | " + " | ".join(engines) + " |")
    lines.append("|" + "---|" * (len(engines) + 2))

    by_variant: dict[str, list[ClaimUnit]] = defaultdict(list)
    for unit in units:
        by_variant[unit.variant_label].append(unit)
    for variant in sorted(by_variant, key=variant_sort_key):
        rows = by_variant[variant]
        cells = row_cells(rows, engines, (arch, platform), lanes, with_layouts=True)
        lines.append(f"| `{variant}` | " + " | ".join(cells) + " |")

    lines.append("")
    dtypes = sorted({u.dtypes for u in units})
    lines.append(f"_Dtypes observed: {', '.join(dtypes)}_")
    lines.append("")
    render_detail(lines, units, engines, arch, platform, lanes)
    lines.append("</details>")
    lines.append("")


def render_traceability(
    lines: list[str], units: list[ClaimUnit], max_case_ids: int
) -> None:
    """The bundle provenance index, emitted once for the whole document.

    Which bundles feed a row does not depend on the target, so emitting this
    inside each ``arch / platform`` section duplicated all of it verbatim --
    growing linearly with every target added,
    on a document that stops rendering on GitHub past a few hundred KB.
    """
    by_op: dict[str, list[ClaimUnit]] = defaultdict(list)
    for unit in units:
        by_op[unit.op].append(unit)

    lines.append("<!-- Row provenance. Target-independent, so recorded once. -->")
    for op in sorted(by_op):
        for label, rows in group_detail_rows(by_op[op]):
            lines.extend(traceability_comment(f"{op}: {label}", rows, max_case_ids))


def render_rollup(
    lines: list[str],
    by_op: dict[str, list[ClaimUnit]],
    targets: list[tuple[str, str]],
    groups: SummaryGroups,
    lanes: set[Lane] | None = None,
) -> None:
    """One table per arch group; in each, one row per op group and one column
    per target: the Any engine counts of the group's ops, added up.

    An op's graphs belong to it alone, so adding its ops' union counts gives
    the group's union count, with nothing counted twice.

    The denominator does not depend on the target, so it is printed once, in
    a ``Graphs`` column, and each target cell carries only the claimed count.
    That halves the cell width: a dozen ``mark claimed/total`` columns are
    wide enough that viewers squeeze them and split the count mid-number.
    """
    lines.append(f"## Summary — {ANY_ENGINE.lower()}, by op group")
    lines.append("")
    rows = [
        (name, [u for op in ops for u in by_op[op]])
        for name, ops in group_ops(sorted(by_op), groups.ops)
    ]
    for title, columns in group_targets(targets, groups.archs):
        if title is not None:
            lines.append(f"### {title}")
            lines.append("")
        headers = [f"{arch}<br>{platform}" for arch, platform in columns]
        lines.append("| Op group | Graphs | " + " | ".join(headers) + " |")
        lines.append("|----|---:|" + "|".join(["----------"] * len(columns)) + "|")
        for name, members in rows:
            engines = engines_claiming(members)
            cells = [str(len(members))]
            cells += [
                mark_not_run(
                    summary_count_cell(members, *target), engines, target, lanes
                ).replace(" ", NBSP)
                for target in columns
            ]
            lines.append(f"| **{name}** | " + " | ".join(cells) + " |")
        lines.append("")


def summary_count_cell(units: list[ClaimUnit], arch: str, platform: str) -> str:
    """``any_summary_cell`` without the denominator, which the summary prints
    once per row. A non-breaking space keeps the mark on the count's line."""
    supported = sum(1 for u in units if claimed_by_any(u, arch, platform))
    if supported == 0:
        return NONE
    mark = FULL if supported == len(units) else PARTIAL
    return f"{mark}{NBSP}{supported}"


def render_markdown(
    units: list[ClaimUnit],
    max_case_ids: int,
    *,
    overview_only: bool = False,
    groups: SummaryGroups | None = None,
) -> str:
    engines = sorted({e for u in units for e in u.claims})
    targets = sorted({pair for u in units for s in u.claims.values() for pair in s})

    lines = [
        "# Combined Engine Support Matrix",
        "",
        "Generated by `render_support_matrix.py` from committed `.support.json` "
        "sidecars.",
        "Do not hand-edit — regenerate with: "
        f"`{REGEN_OVERVIEW_COMMAND if overview_only else REGEN_COMMAND}`",
        *(
            [
                "",
                "For the full per-variant and per-dtype detail matrix, run: "
                f"`{REGEN_COMMAND}`",
            ]
            if overview_only
            else []
        ),
        "",
        *_legend(overview_only),
    ]

    if not units:
        lines.append("_No claim-bearing bundles found._")
        return "\n".join(lines) + "\n"

    if not targets:
        lines.append(
            f"_{len(units)} bundle(s) found, none carrying any support claim._"
        )
        return "\n".join(lines) + "\n"

    by_op: dict[str, list[ClaimUnit]] = defaultdict(list)
    for unit in units:
        by_op[unit.op].append(unit)
    ops = sorted(by_op)
    lanes = lanes_of(units)

    render_rollup(lines, by_op, targets, groups or SummaryGroups(), lanes)

    for arch, platform in targets:
        if overview_only:
            lines.append(f"## {arch} / {platform}")
        else:
            lines.append("<details>")
            lines.append(f"<summary><big><b>{arch} / {platform}</b></big></summary>")
        lines.append("")
        if not overview_only:
            lines.append("### Overview")
            lines.append("")
        lines.append(f"| Op | {ANY_ENGINE} | " + " | ".join(engines) + " |")
        lines.append("|----|" + "|".join(["----------"] * (len(engines) + 1)) + "|")
        for op in ops:
            cells = row_cells(
                by_op[op], engines, (arch, platform), lanes, with_layouts=False
            )
            lines.append(f"| **{op}** | " + " | ".join(cells) + " |")
        lines.append("")

        if not overview_only:
            for op in ops:
                render_family(lines, op, by_op[op], engines, arch, platform, lanes)

            if lines and lines[-1] == "":
                lines.pop()
            lines.append("</details>")
            lines.append("<br>")

    if not overview_only:
        lines.append("<details>")
        lines.append("<summary><big><b>Reading guide</b></big></summary>")
        lines.append('<a id="reading-guide"></a>')
        lines.append("")
        lines.extend(READING_GUIDE[2:])
        render_traceability(lines, units, max_case_ids)
        lines.append("")
        lines.append("</details>")

    return "\n".join(lines).rstrip("\n") + "\n"


# --------------------------------------------------------------------------
# JSON rendering
# --------------------------------------------------------------------------


def render_json(units: list[ClaimUnit]) -> str:
    """The flat index: every claimed graph, with nothing aggregated away.

    Markdown is a map; this is the data layer under it. A browsable viewer
    should read this, not scrape the rendered tables.
    """
    engines = sorted({e for u in units for e in u.claims})
    targets = sorted({pair for u in units for s in u.claims.values() for pair in s})

    document = {
        "version": 1,
        "generated_by": "render_support_matrix.py",
        "regenerate_with": REGEN_JSON_COMMAND,
        "engines": engines,
        "targets": [
            {"id": target_id(pair), "arch": pair[0], "platform": pair[1]}
            for pair in targets
        ],
        "units": [
            unit.as_record() for unit in sorted(units, key=lambda u: u.sort_key())
        ],
    }
    return json.dumps(document, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render the combined engine support matrix from committed "
            "support-claim sidecars (RFC 0015 §11)."
        )
    )
    parser.add_argument(
        "--bundles-dir",
        type=pathlib.Path,
        default=DEFAULT_BUNDLES_DIR,
        help=f"Bundle tree to scan (default: {DEFAULT_BUNDLES_DIR}).",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help=(
            "markdown: the human matrix (default). json: the flat per-graph "
            "index, for tooling that needs the data rather than the view."
        ),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_MATRIX_PATH,
        help=(
            f"Write here (default: {DEFAULT_MATRIX_PATH}). Use '-' for stdout. "
            "The default is a file rather than stdout because the document is "
            "large, and producing it is the point of running this."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Do not write. Re-render and compare against an existing file "
            f"(--output, or {DEFAULT_MATRIX_PATH} by default); exit 1 if they "
            "differ or it is missing. The matrix is not committed, so this is "
            "not wired into CI -- it is for anyone keeping a local copy in "
            "sync."
        ),
    )
    parser.add_argument(
        "--max-case-ids",
        type=int,
        default=0,
        help=(
            "Inline this many sweep case ids per bundle in the markdown "
            "traceability comments (0 = none, the default; negative = all). "
            "The bundle path is always emitted; use --format json for the "
            "complete per-case index."
        ),
    )
    parser.add_argument(
        "--overview-only",
        action="store_true",
        help=(
            "Emit only the per-target overview tables (no per-family detail "
            "sections, no collapsible wrappers). The output is small enough "
            "to commit and render directly on GitHub."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.overview_only and args.output == DEFAULT_MATRIX_PATH:
        args.output = DEFAULT_OVERVIEW_PATH

    root = args.bundles_dir
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    errors: list[str] = []
    units = collect_units(root, errors)
    groups = load_summary_groups(DEFAULT_GROUPS_PATH, errors)
    if errors:
        # Each was printed as found. Writing (or blessing) a matrix past them
        # would publish claims the C++ harness refuses to load.
        print(
            f"error: {len(errors)} bundle defect(s); nothing rendered",
            file=sys.stderr,
        )
        return 1
    if not units:
        warn(f"{root}: no claim-bearing bundles found")

    if args.format == "json":
        document = render_json(units)
    else:
        document = render_markdown(
            units,
            args.max_case_ids,
            overview_only=args.overview_only,
            groups=groups,
        )

    if args.check:
        target = args.output
        if args.format == "json":
            regen_cmd = REGEN_JSON_COMMAND
        elif args.overview_only:
            regen_cmd = REGEN_OVERVIEW_COMMAND
        else:
            regen_cmd = REGEN_COMMAND
        if str(target) == "-":
            print("error: --check needs a file to compare against", file=sys.stderr)
            return 2
        try:
            committed = target.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot read {target} ({exc})", file=sys.stderr)
            print(f"       regenerate it with: {regen_cmd}", file=sys.stderr)
            return 1
        if committed != document:
            print(
                f"error: {target} is stale -- it does not match a re-render of "
                f"the current sidecars",
                file=sys.stderr,
            )
            print(f"       regenerate it with: {regen_cmd}", file=sys.stderr)
            return 1
        return 0

    if str(args.output) == "-":
        try:
            sys.stdout.buffer.write(document.encode("utf-8"))
            sys.stdout.buffer.flush()
        except BrokenPipeError:
            # `| head` closing the pipe early is the caller's choice, not a
            # failure. Point the remaining buffered writes at devnull so the
            # interpreter does not report the same broken pipe again at exit.
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    # To stderr, so `--output -` piping stays clean and this stays out of the
    # document. The file is git-ignored, so saying where it went is the only
    # way the reader learns it exists.
    print(
        f"wrote {args.output} ({len(document.encode('utf-8')):,} bytes)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

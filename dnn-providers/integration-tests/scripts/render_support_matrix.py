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
  overview, per-family variant table, per-(variant, dtype) detail, and HTML
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

The rendered document is **generated, not committed** -- see
``dnn-providers/integration-tests/.gitignore`` for why, and RFC 0015 §11.

Usage:

    python3 dnn-providers/integration-tests/scripts/render_support_matrix.py
"""

from __future__ import annotations

import argparse
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
    find_graph_files,
    find_sweep_roots,
)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

INTEGRATION_TESTS_DIR = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_BUNDLES_DIR = INTEGRATION_TESTS_DIR / "integration-test-bundles"
DEFAULT_MATRIX_PATH = INTEGRATION_TESTS_DIR / "SUPPORT_MATRIX.md"
DEFAULT_OVERVIEW_PATH = INTEGRATION_TESTS_DIR / "SUPPORT_MATRIX_OVERVIEW.md"

# No --output: it defaults to DEFAULT_MATRIX_PATH, so the shortest command that
# reproduces the document is the one worth quoting back to the reader.
REGEN_COMMAND = (
    "python3 dnn-providers/integration-tests/scripts/render_support_matrix.py"
)
REGEN_OVERVIEW_COMMAND = f"{REGEN_COMMAND} --overview-only"
REGEN_JSON_COMMAND = f"{REGEN_COMMAND} --format json"

# --------------------------------------------------------------------------
# Presentation constants
# --------------------------------------------------------------------------

FULL = "✅"
PARTIAL = "🟡"
NONE = "—"

# Informational only: the sidecars key support by gfx target, which is what the
# harness matches on, but a reader looking for "does this run on MI300" should
# not have to know the mapping. An unlisted arch simply renders bare.
ARCH_MARKETING_NAMES = {
    "gfx908": "MI100",
    "gfx90a": "MI200 series (MI210/MI250/MI250X)",
    "gfx942": "MI300 series (MI300A/MI300X/MI325X)",
    "gfx950": "MI350 series (MI350X/MI355X)",
}

# Layout label for a graph whose rank carries no NCHW-family layout (a rank-2
# matmul operand, say). Kept distinct from NONE so it cannot be misread as
# "unsupported" inside a layout list.
NO_LAYOUT = "n/a"

# Short key at the top — just enough to read the tables. The full reference
# (what a claim means, how variants are tagged, how tiers merge) goes at the
# end so the reader hits the data first.
LEGEND_DETAIL = (
    "Expand a row for variant and per-dtype detail. "
    "Full legend at the [end of the document](#reading-guide)."
)

LEGEND_OVERVIEW = (
    f"`{FULL}` = all bundles supported · "
    f"`{PARTIAL}` = some supported · "
    f"`{NONE}` = none supported.\n"
    "\n"
    "> **How to read a cell:** "
    "`🟡 624/840` → **624** bundles supported out of **840** total bundles "
    "in the test suite. Each bundle is a graph."
)

_LEGEND_PREFIX = (
    f"`{FULL}` = all graphs claimed · "
    f"`{PARTIAL}` = some claimed · "
    f"`{NONE}` = unclaimed (not the same as *known unsupported*). "
)


def _legend(overview_only: bool) -> list[str]:
    if overview_only:
        return [LEGEND_OVERVIEW, ""]
    return [_LEGEND_PREFIX + LEGEND_DETAIL, ""]


# The full reference, rendered after the last target section.
READING_GUIDE = [
    "## Reading guide",
    "",
    "_Each overview row is one op family. Open a disclosure triangle "
    "to expand it into per-variant rows, and the one inside that for full "
    "per-(variant, dtype) rows. Every level counts the same way, so a family's "
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
    f"| `{NONE}` | no claim recorded |",
    f"| `{NO_LAYOUT}` | the graph's rank carries no NCHW-family layout "
    "(a rank-2 matmul operand, say) |",
    "",
    "**Denominator.** Every graph the bundle tree holds for that row: a "
    "single-graph bundle counts once, a template sweep counts once per case. "
    "It does not depend on the target, so the same row shows the same "
    "denominator in every `arch / platform` section — only the numerator "
    "moves.",
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
    "**Variants** read `[feature tags] + fusion chain`. The chain is the "
    "family's node followed by whatever is fused onto it. The tags each mark a "
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
    "**Tiers are merged.** A row can draw on `quick`, `standard`, and `full` "
    "bundles at once. The tier and bundle paths behind each row are in HTML "
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


# --------------------------------------------------------------------------
# JSON loading
# --------------------------------------------------------------------------


def load_json(path: pathlib.Path):
    """Parse ``path`` as JSON, warning and returning None on any failure."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"{path}: failed to parse JSON ({exc}); skipping")
        return None


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
    """The fusion pattern and feature tags, relative to the section's family."""
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


def tier_of(relative_path: pathlib.PurePath) -> str:
    """The enforcement tier a bundle sits under: ``quick``/``standard``/``full``."""
    parts = relative_path.parts
    return parts[0] if parts else ""


def family_label(node_type: str) -> str:
    """``BatchnormInferenceAttributesVarianceExt`` -> ``BatchnormInferenceVarianceExt``.

    ``Attributes`` is a schema wart, not part of the op's name, and it is not
    always the trailing token -- so strip it wherever it appears rather than
    only off the end.
    """
    return node_type.replace("Attributes", "") or node_type


def family_of(graph: dict, relative_path: pathlib.PurePath, bundle: str) -> str:
    """The op family: the graph's primary node type, not its directory.

    Bundle directories are split by fusion -- ``Batchnorm`` next to
    ``BatchnormPointwise``, ``Matmul`` next to ``MatmulPointwise``. Keying the
    family off the path therefore scatters one op's variants across sibling
    sections, each showing a lone ``(bare)`` row, and the reader has to already
    know the sibling exists to go find the fused rows. Keying off ``nodes[0]``
    folds them back into one disclosure: the fusion is precisely what the
    Variant column already spells out.

    Falls back to the directory for a graph with no nodes, which is malformed
    but should not lose its claims.
    """
    nodes = [n for n in (graph.get("nodes") or []) if isinstance(n, dict)]
    primary = family_label(nodes[0].get("type") or "") if nodes else ""
    parts = relative_path.parts
    directory = parts[1] if len(parts) > 1 else (parts[0] if parts else "")
    if not primary:
        warn(f"{bundle}: no primary node type; falling back to directory name")
        return directory
    # The directory should still name the op it holds -- a bundle filed under
    # the wrong family is a real mistake, even though we no longer read the
    # family from it.
    if primary.lower()[:6] not in directory.lower():
        warn(
            f"{bundle}: directory does not echo primary node type "
            f"'{primary}'; is the bundle misfiled?"
        )
    return primary


# --------------------------------------------------------------------------
# Claim units
# --------------------------------------------------------------------------


def target_id(pair: tuple[str, str]) -> str:
    """``("gfx942", "linux") -> "gfx942/linux"``, matching the section heading."""
    return f"{pair[0]}/{pair[1]}"


@dataclass
class ClaimUnit:
    """One claimed graph: a sweep case, or a whole single-graph bundle."""

    family: str
    variant: str
    dtypes: str
    layout: str
    tier: str
    bundle: str
    case_id: str | None
    # engine -> {(arch, platform)}
    claims: dict[str, set[tuple[str, str]]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.case_id or pathlib.PurePath(self.bundle).name

    def supported_by(self, engine: str, arch: str, platform: str) -> bool:
        return (arch, platform) in self.claims.get(engine, ())

    def sort_key(self) -> tuple:
        return (
            self.family,
            self.variant,
            self.dtypes,
            self.layout,
            self.tier,
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
            "family": self.family,
            "variant": self.variant,
            "dtypes": self.dtypes,
            "layout": self.layout,
            "tier": self.tier,
            "bundle": self.bundle,
            "case_id": self.case_id,
            "claims": {
                engine: [target_id(pair) for pair in sorted(pairs)]
                for engine, pairs in sorted(self.claims.items())
            },
        }


def support_pairs(support: object, context: str) -> set[tuple[str, str]]:
    """Flatten a ``{arch: [platform, ...]}`` map into (arch, platform) pairs."""
    pairs = set()
    if not isinstance(support, dict):
        warn(f"{context}: 'support' must be an arch -> platform-array object; ignoring")
        return pairs
    for arch, platforms in support.items():
        if not isinstance(platforms, list):
            warn(f"{context}: arch '{arch}' platform value must be an array; ignoring")
            continue
        for platform in platforms:
            if isinstance(platform, str):
                pairs.add((arch, platform))
    return pairs


def collect_sweep_units(sweep_dir: pathlib.Path, root: pathlib.Path) -> list[ClaimUnit]:
    """Build one ClaimUnit per case of a template-sweep bundle.

    A bundle with no sidecar still yields units -- claimless ones. Absent a
    sidecar there is nothing to put in a numerator, but the cases are still
    graphs the tree holds, so they belong in the denominator. Dropping them
    would erase a wholly unclaimed family from the document rather than
    showing it as a row of dashes, which is the one thing the matrix exists
    to make visible.
    """
    support_path = sweep_dir / SWEEP_SUPPORT_NAME
    sidecar = load_json(support_path) if support_path.is_file() else {}
    if not isinstance(sidecar, dict):
        sidecar = {}  # Malformed; load_json warned. Count the cases anyway.

    template = load_json(sweep_dir / SWEEP_TEMPLATE_NAME)
    sweep = load_json(sweep_dir / SWEEP_MANIFEST_NAME)
    if not isinstance(template, dict):
        return []
    if not isinstance(sweep, dict) or not isinstance(sweep.get("cases"), list):
        warn(f"{sweep_dir / SWEEP_MANIFEST_NAME}: malformed sweep.json; skipping")
        return []

    relative = sweep_dir.relative_to(root)
    bundle = relative.as_posix()
    tier = tier_of(relative)
    family = family_of(template, relative, bundle)
    tags = scenario_tags_in(relative)

    # engine -> case id -> {(arch, platform)}
    claims_by_case: dict[str, dict[str, set[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    claims = sidecar.get("claims") or {}
    if not isinstance(claims, dict):
        warn(f"{support_path}: 'claims' must be an object; skipping")
        claims = {}

    known_ids = {
        case["id"]
        for case in sweep["cases"]
        if isinstance(case, dict) and isinstance(case.get("id"), str)
    }
    for engine, groups in claims.items():
        if not isinstance(groups, list):
            warn(f"{support_path}: engine '{engine}' claims must be an array; skipping")
            continue
        for index, group in enumerate(groups):
            if not isinstance(group, dict):
                warn(
                    f"{support_path}: engine '{engine}' group {index} is not an object"
                )
                continue
            pairs = support_pairs(group.get("support"), f"{support_path} [{engine}]")
            case_ids = group.get("cases")
            if not isinstance(case_ids, list):
                warn(
                    f"{support_path}: engine '{engine}' group {index} has no case array"
                )
                continue
            for case_id in case_ids:
                if case_id not in known_ids:
                    warn(
                        f"{support_path}: engine '{engine}' claims unknown case "
                        f"'{case_id}' (not in {SWEEP_MANIFEST_NAME}); ignoring"
                    )
                    continue
                claims_by_case[engine][case_id] |= pairs

    units = []
    for case in sweep["cases"]:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            continue
        case_id = case["id"]
        values = case.get("values") if isinstance(case.get("values"), dict) else {}

        graph = resolve_templates(template, values)
        apply_case_tensors(graph, values)

        layout = layout_from_name(case_id) or layout_from_strides(graph) or NO_LAYOUT
        unit = ClaimUnit(
            family=family,
            variant=variant_of(graph, tags),
            dtypes=dtypes_of(graph),
            layout=layout,
            tier=tier,
            bundle=bundle,
            case_id=case_id,
        )
        for engine, per_case in claims_by_case.items():
            pairs = per_case.get(case_id)
            if pairs:
                unit.claims[engine] = set(pairs)
        units.append(unit)
    return units


def collect_single_graph_unit(
    graph_path: pathlib.Path, root: pathlib.Path
) -> ClaimUnit | None:
    """Build the single ClaimUnit of a single-graph bundle.

    Claimless when the bundle has no sidecar; see ``collect_sweep_units`` for
    why that is a unit rather than nothing.
    """
    support_path = graph_path.with_name(f"{graph_path.stem}.support.json")
    sidecar = load_json(support_path) if support_path.is_file() else {}
    if not isinstance(sidecar, dict):
        sidecar = {}  # Malformed; load_json warned. Count the graph anyway.

    graph = load_json(graph_path)
    if not isinstance(graph, dict):
        return None

    relative = graph_path.parent.relative_to(root)
    bundle = relative.as_posix()
    tier = tier_of(relative)
    family = family_of(graph, relative, bundle)
    tags = scenario_tags_in(relative)

    # The layout hint lives in the bundle path for a single-graph bundle; the
    # graph's own strides are the fallback.
    layout = (
        layout_from_name(relative.as_posix()) or layout_from_strides(graph) or NO_LAYOUT
    )
    unit = ClaimUnit(
        family=family,
        variant=variant_of(graph, tags),
        dtypes=dtypes_of(graph),
        layout=layout,
        tier=tier,
        bundle=bundle,
        case_id=None,
    )

    claims = sidecar.get("claims") or {}
    if not isinstance(claims, dict):
        warn(f"{support_path}: 'claims' must be an object; skipping")
        return unit
    for engine, arch_map in claims.items():
        pairs = support_pairs(arch_map, f"{support_path} [{engine}]")
        if pairs:
            unit.claims[engine] = pairs
    return unit


def collect_units(root: pathlib.Path) -> list[ClaimUnit]:
    sweep_roots = find_sweep_roots(root)
    units: list[ClaimUnit] = []
    for sweep_dir in sweep_roots:
        units.extend(collect_sweep_units(sweep_dir, root))
    for graph_path in find_graph_files(root, sweep_roots):
        unit = collect_single_graph_unit(graph_path, root)
        if unit is not None:
            units.append(unit)
    return units


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
    tag list and a clipped pointwise chain -- once per bundle spent 131 KB
    saying nothing new, on a document with a hard size ceiling.

    Emitted as a block *after* the table it annotates: a comment line between
    table rows would terminate the table in GitHub-flavored markdown.
    """
    by_bundle: dict[str, list[ClaimUnit]] = defaultdict(list)
    for unit in units:
        by_bundle[unit.bundle].append(unit)

    entries = []
    for bundle in sorted(by_bundle):
        members = by_bundle[bundle]
        entry = f"{bundle} ({members[0].tier}, {len(members)} case(s))"
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


def detail_row_label(family: str, variant: str, dtypes: str) -> str:
    """``Batchnorm[multi_batch] + Pointwise:RELU_FWD [io=fp16, ...]``.

    Shared with the traceability appendix so a row and its provenance entry are
    keyed by the same string.
    """
    suffix = "" if variant == "(bare)" else variant
    return f"{family}{suffix} {dtypes}"


def render_detail(
    lines: list[str],
    units: list[ClaimUnit],
    family: str,
    engines: list[str],
    arch: str,
    platform: str,
) -> None:
    """The innermost drill-down: one row per (variant, dtype)."""
    lines.append("<details>")
    lines.append("<summary>🔎 per-(variant, dtype) detail</summary>")
    lines.append("")
    lines.append("| Operations | " + " | ".join(engines) + " |")
    lines.append("|" + "---|" * (len(engines) + 1))

    grouped: dict[tuple[str, str], list[ClaimUnit]] = defaultdict(list)
    for unit in units:
        grouped[(unit.variant, unit.dtypes)].append(unit)

    for variant, dtypes in sorted(
        grouped, key=lambda key: (variant_sort_key(key[0]), key[1])
    ):
        rows = grouped[(variant, dtypes)]
        label = detail_row_label(family, variant, dtypes)
        cells = [layout_cell(rows, e, arch, platform) for e in engines]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("</details>")


def render_family(
    lines: list[str],
    family: str,
    units: list[ClaimUnit],
    engines: list[str],
    arch: str,
    platform: str,
) -> None:
    lines.append("<details>")
    lines.append(f"<summary>📂 <b>{family}</b></summary>")
    lines.append("")
    lines.append("| Variant | " + " | ".join(engines) + " |")
    lines.append("|" + "---|" * (len(engines) + 1))

    by_variant: dict[str, list[ClaimUnit]] = defaultdict(list)
    for unit in units:
        by_variant[unit.variant].append(unit)
    for variant in sorted(by_variant, key=variant_sort_key):
        rows = by_variant[variant]
        cells = [layout_cell(rows, e, arch, platform) for e in engines]
        lines.append(f"| `{variant}` | " + " | ".join(cells) + " |")

    lines.append("")
    dtypes = sorted({u.dtypes for u in units})
    lines.append(f"_Dtypes observed: {', '.join(dtypes)}_")
    lines.append("")
    render_detail(lines, units, family, engines, arch, platform)
    lines.append("</details>")
    lines.append("")


def render_traceability(
    lines: list[str], units: list[ClaimUnit], max_case_ids: int
) -> None:
    """The bundle provenance index, emitted once for the whole document.

    Which bundles feed a row does not depend on the target, so emitting this
    inside each ``arch / platform`` section duplicated all of it verbatim --
    94 KB today at two targets, and growing linearly with every target added,
    on a document that stops rendering on GitHub past a few hundred KB.
    """
    by_family: dict[str, list[ClaimUnit]] = defaultdict(list)
    for unit in units:
        by_family[unit.family].append(unit)

    lines.append("<!-- Row provenance. Target-independent, so recorded once. -->")
    for family in sorted(by_family):
        grouped: dict[tuple[str, str], list[ClaimUnit]] = defaultdict(list)
        for unit in by_family[family]:
            grouped[(unit.variant, unit.dtypes)].append(unit)
        for variant, dtypes in sorted(
            grouped, key=lambda key: (variant_sort_key(key[0]), key[1])
        ):
            label = detail_row_label(family, variant, dtypes)
            lines.extend(
                traceability_comment(label, grouped[(variant, dtypes)], max_case_ids)
            )


def render_markdown(
    units: list[ClaimUnit],
    max_case_ids: int,
    *,
    overview_only: bool = False,
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

    by_family: dict[str, list[ClaimUnit]] = defaultdict(list)
    for unit in units:
        by_family[unit.family].append(unit)
    families = sorted(by_family)

    for arch, platform in targets:
        marketing = ARCH_MARKETING_NAMES.get(arch)
        suffix = f" — {marketing}" if marketing else ""
        if overview_only:
            lines.append(f"## {arch} / {platform}{suffix}")
        else:
            lines.append("<details>")
            lines.append(
                f"<summary><big><b>{arch} / {platform}{suffix}</b></big></summary>"
            )
        lines.append("")
        if not overview_only:
            lines.append("### Overview")
            lines.append("")
        lines.append("| Op family | " + " | ".join(engines) + " |")
        lines.append("|-----------|" + "|".join(["----------"] * len(engines)) + "|")
        for family in families:
            cells = [
                summary_cell(by_family[family], e, arch, platform) for e in engines
            ]
            lines.append(f"| **{family}** | " + " | ".join(cells) + " |")
        lines.append("")

        if not overview_only:
            for family in families:
                render_family(lines, family, by_family[family], engines, arch, platform)

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
            "~300 KB, and producing it is the point of running this."
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

    units = collect_units(root)
    if not units:
        warn(f"{root}: no claim-bearing bundles found")

    if args.format == "json":
        document = render_json(units)
    else:
        document = render_markdown(
            units, args.max_case_ids, overview_only=args.overview_only
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

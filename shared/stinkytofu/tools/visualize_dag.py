#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Create a standalone web visualization of a RegionDAG text dump."""

from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path


NODE_RE = re.compile(r"^(\d+):\s?(.*)$")
EDGE_RE = re.compile(r"^(\d+)\s*->\s*(\d+)\s*$")
ATTRIBUTE_BLOCK_RE = re.compile(r"\s+\{\s*issueCycles\s*=.*$")


@dataclass
class DAG:
    nodes: dict[int, str]
    edges: list[tuple[int, int]]


def parse_dags(text: str) -> list[DAG]:
    """Parse every `DAG nodes`/`DAG edges` section from a debug log."""
    dags: list[DAG] = []
    nodes: dict[int, str] | None = None
    edges: list[tuple[int, int]] = []
    section: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "DAG nodes:":
            if nodes is not None:
                dags.append(DAG(nodes, edges))
            nodes = {}
            edges = []
            section = "nodes"
            continue
        if stripped == "DAG edges:" and nodes is not None:
            section = "edges"
            continue

        if section == "nodes" and nodes is not None:
            match = NODE_RE.match(stripped)
            if match:
                nodes[int(match.group(1))] = match.group(2)
        elif section == "edges":
            match = EDGE_RE.match(stripped)
            if match:
                edges.append((int(match.group(1)), int(match.group(2))))

    if nodes is not None:
        dags.append(DAG(nodes, edges))
    return dags


def assign_edge_lanes(
    edges: list[tuple[int, int]], positions: dict[int, int]
) -> list[tuple[int, int, int]]:
    """Assign overlapping edge intervals to separate lanes in the left gutter."""
    intervals: list[tuple[int, int, int, int]] = []
    for from_id, to_id in sorted(edges):
        if from_id not in positions or to_id not in positions:
            continue
        start, end = sorted((positions[from_id], positions[to_id]))
        intervals.append((start, end, from_id, to_id))

    lane_ends: list[int] = []
    routed: list[tuple[int, int, int]] = []
    for start, end, from_id, to_id in intervals:
        lane = next(
            (i for i, lane_end in enumerate(lane_ends) if lane_end < start), None
        )
        if lane is None:
            lane = len(lane_ends)
            lane_ends.append(end)
        else:
            lane_ends[lane] = end
        routed.append((from_id, to_id, lane))
    return routed


def simplify_instruction(instruction: str) -> str:
    """Remove the trailing instruction attribute block from dumped IR."""
    return ATTRIBUTE_BLOCK_RE.sub("", instruction)


def make_html(dag: DAG, title: str) -> str:
    """Generate a self-contained HTML/SVG DAG viewer."""
    node_ids = sorted(dag.nodes)
    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    routed_edges = assign_edge_lanes(dag.edges, positions)
    lane_count = max((lane for _, _, lane in routed_edges), default=-1) + 1

    row_height = 54
    node_height = 36
    lane_spacing = 12
    node_x = max(80, 42 + lane_count * lane_spacing)
    simplified_nodes = {
        node_id: simplify_instruction(instruction)
        for node_id, instruction in dag.nodes.items()
    }
    longest_instruction = max(
        (len(text) for text in simplified_nodes.values()), default=40
    )
    node_width = max(720, min(1800, longest_instruction * 7 + 80))
    svg_width = node_x + node_width + 32
    svg_height = max(100, len(node_ids) * row_height + 24)

    edge_elements: list[str] = []
    for from_id, to_id, lane in routed_edges:
        from_y = positions[from_id] * row_height + 12 + node_height / 2
        to_y = positions[to_id] * row_height + 12 + node_height / 2
        lane_x = node_x - 24 - lane * lane_spacing
        edge_elements.append(
            f'<path class="edge" data-from="{from_id}" data-to="{to_id}" '
            f'd="M {node_x} {from_y:g} H {lane_x} V {to_y:g} H {node_x - 3}" '
            'marker-end="url(#arrow)"/>'
        )

    node_elements: list[str] = []
    for index, node_id in enumerate(node_ids):
        y = index * row_height + 12
        instruction = html.escape(simplified_nodes[node_id])
        node_elements.append(
            f'<g class="node" data-id="{node_id}">'
            f'<rect x="{node_x}" y="{y}" width="{node_width}" height="{node_height}" rx="4"/>'
            f'<text x="{node_x + 10}" y="{y + 23}">'
            f'<tspan class="node-id">{node_id}:</tspan> {instruction}</text>'
            "</g>"
        )

    escaped_title = html.escape(title)
    edges_svg = "\n".join(edge_elements)
    nodes_svg = "\n".join(node_elements)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escaped_title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: system-ui, sans-serif; background: #111827; color: #e5e7eb; }}
  header {{
    position: sticky; top: 0; z-index: 2; display: flex; align-items: center; gap: 12px;
    padding: 10px 16px; background: #1f2937; border-bottom: 1px solid #374151;
  }}
  header strong {{ margin-right: auto; }}
  button {{
    border: 1px solid #4b5563; border-radius: 4px; padding: 4px 10px;
    background: #111827; color: #e5e7eb; cursor: pointer;
  }}
  #viewport {{ overflow: auto; height: calc(100vh - 51px); padding: 16px; }}
  svg {{ display: block; background: #0b1020; transform-origin: top left; }}
  .node rect {{ fill: #1f2937; stroke: #6b7280; stroke-width: 1; }}
  .node text {{
    fill: #e5e7eb; font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    text-anchor: start;
  }}
  .node-id {{ fill: #93c5fd; font-weight: 700; }}
  .edge {{ fill: none; stroke: #64748b; stroke-width: 1.2; }}
  .edge.active {{ stroke: #f59e0b; stroke-width: 2.5; }}
  .node.active rect {{ stroke: #f59e0b; stroke-width: 2; }}
  #edge-details {{
    position: fixed; z-index: 10; min-width: 220px; max-width: 420px;
    margin: 0; padding: 10px 12px; border: 1px solid #6b7280; border-radius: 6px;
    background: #111827; color: #e5e7eb; box-shadow: 0 6px 20px #0008;
    font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    pointer-events: none; white-space: pre-wrap;
  }}
  #edge-details[hidden] {{ display: none; }}
</style>
</head>
<body>
<header>
  <strong>{escaped_title}</strong>
  <span>{len(node_ids)} nodes · {len(routed_edges)} edges</span>
  <button type="button" onclick="zoomBy(0.8)">−</button>
  <button type="button" onclick="resetZoom()">100%</button>
  <button type="button" onclick="zoomBy(1.25)">+</button>
</header>
<main id="viewport">
<svg id="dag" width="{svg_width}" height="{svg_height}"
     viewBox="0 0 {svg_width} {svg_height}" role="img" aria-label="{escaped_title}">
  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4"
            orient="auto" markerUnits="strokeWidth">
      <path d="M 0 0 L 8 4 L 0 8 z" fill="context-stroke"/>
    </marker>
  </defs>
  <g id="edges">{edges_svg}</g>
  <g id="nodes">{nodes_svg}</g>
</svg>
</main>
<pre id="edge-details" hidden></pre>
<script>
  const svg = document.getElementById("dag");
  const edgeDetails = document.getElementById("edge-details");
  let zoom = 1;
  function applyZoom() {{
    svg.style.width = `${{{svg_width} * zoom}}px`;
    svg.style.height = `${{{svg_height} * zoom}}px`;
  }}
  function zoomBy(factor) {{ zoom = Math.max(0.25, Math.min(4, zoom * factor)); applyZoom(); }}
  function resetZoom() {{ zoom = 1; applyZoom(); }}

  function positionEdgeDetails(event) {{
    const gap = 14;
    edgeDetails.style.left = `${{event.clientX + gap}}px`;
    edgeDetails.style.top = `${{event.clientY + gap}}px`;
    const bounds = edgeDetails.getBoundingClientRect();
    if (bounds.right > window.innerWidth - gap)
      edgeDetails.style.left = `${{event.clientX - bounds.width - gap}}px`;
    if (bounds.bottom > window.innerHeight - gap)
      edgeDetails.style.top = `${{event.clientY - bounds.height - gap}}px`;
  }}

  function showEdgeDetails(node, event) {{
    const id = node.dataset.id;
    const incoming = Array.from(document.querySelectorAll(`.edge[data-to="${{id}}"]`))
      .map(edge => edge.dataset.from + " -> " + edge.dataset.to);
    const outgoing = Array.from(document.querySelectorAll(`.edge[data-from="${{id}}"]`))
      .map(edge => edge.dataset.from + " -> " + edge.dataset.to);
    edgeDetails.textContent =
      "In-edges:\\n" + (incoming.length ? incoming.join("\\n") : "(none)") +
      "\\n\\nOut-edges:\\n" + (outgoing.length ? outgoing.join("\\n") : "(none)");
    edgeDetails.hidden = false;
    positionEdgeDetails(event);
  }}

  document.querySelectorAll(".node").forEach(node => {{
    node.addEventListener("mouseenter", event => {{
      const id = node.dataset.id;
      node.classList.add("active");
      document.querySelectorAll(`.edge[data-from="${{id}}"], .edge[data-to="${{id}}"]`)
        .forEach(edge => edge.classList.add("active"));
      showEdgeDetails(node, event);
    }});
    node.addEventListener("mousemove", positionEdgeDetails);
    node.addEventListener("mouseleave", () => {{
      node.classList.remove("active");
      document.querySelectorAll(".edge.active").forEach(edge => edge.classList.remove("active"));
      edgeDetails.hidden = true;
    }});
  }});
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a web visualization of a `dumpDAGGraph` log."
    )
    parser.add_argument("input", type=Path, help="DAG dump or debug-log path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("dag.html"),
        help="standalone HTML output path (default: dag.html)",
    )
    parser.add_argument(
        "--graph-index",
        type=int,
        default=0,
        help="zero-based DAG section to render when the log contains several",
    )
    parser.add_argument("--title", default="Region DAG", help="page title")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dags = parse_dags(args.input.read_text(encoding="utf-8"))
    if not dags:
        print(f"error: no DAG dump found in {args.input}", file=sys.stderr)
        return 1
    if args.graph_index < 0 or args.graph_index >= len(dags):
        print(
            f"error: graph index {args.graph_index} is out of range "
            f"(found {len(dags)} DAG dumps)",
            file=sys.stderr,
        )
        return 1

    if args.output.suffix.lower() not in {".html", ".htm"}:
        print("error: output extension must be .html or .htm", file=sys.stderr)
        return 1

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            make_html(dags[args.graph_index], args.title),
            encoding="utf-8",
        )
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

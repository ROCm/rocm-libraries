################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Debug-only visualization helpers for the CMS validator's dataflow graphs.

This module is NOT part of the validator's runtime path. It depends on
`networkx` and `matplotlib`, both of which are intentionally kept as soft
dependencies — imports live INSIDE function bodies so importing this
module does not pull them in, and the validator itself never imports
this module. Use only for debugging / inspection (test fixtures, ad-hoc
scripts under `visualizations/`, etc.).
"""

from typing import Optional, Set, Tuple


def visualize_dataflow_graph(graph, output_path: str,
                             title: Optional[str] = None,
                             highlight_edges: Optional[Set[Tuple]] = None,
                             highlight_label: str = "ARTIFACT") -> str:
    """Render a `DataflowGraph` to PNG using networkx + matplotlib.

    Imports are local so neither package becomes a runtime dependency of
    the validator. Use only for debugging/inspection.

    Each node is labeled with its category, body, and SchedulePosition.
    Each edge is labeled with edge_kind, intra-operand byte offsets, and
    the producer/consumer operand-slot indices (rocm-libraries-wx9.3
    phase 3 — operand-slot is part of the cross-graph edge identity, so
    surfacing it in the visualization makes the slot-flip pattern of a
    within-graph reorder visible to the eye).

    Arrowhead sizing: `arrowsize=40` paired with `min_target_margin=20`
    keeps the arrowhead from being swallowed by the node circle
    (`node_size=3500` is large enough that the default margin lands the
    arrow tip inside the node and hides it). With the larger margin the
    arrow tip lands clearly outside the node boundary and the direction
    of every edge is unambiguous.

    `highlight_edges`: optional set of `(producer.identity, consumer.identity)`
    tuples; matching edges are rendered in red with a dashed style and a
    leading `[<highlight_label>]` annotation. Used by the cross-subiter
    Pack-artifact companion to bwfr (CROSS_SUBITER_ALU_FP_MINIMAL_REPRO.md)
    so the artifactual edge in the default-side graph is visually distinct
    from the semantically valid edges. No effect if None or empty.
    """
    import networkx as nx
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Local import keeps CMSValidatorVisualization importable in environments
    # without networkx/matplotlib (the function-level imports above already
    # establish that pattern); CMSValidator is always available wherever the
    # validator runs.
    from Tensile.Components.CMSValidator import render_node_label

    def _node_label(node):
        pos = node.position
        pos_str = f"L{pos.loop_index}/S{pos.stream_index}"
        # rocm-libraries-aprv: render_node_label distinguishes PRE_LOOP /
        # POST_LOOP @-1 entries (which collide on the raw node.name string)
        # so the visualization labels are unambiguous in the same way the
        # dump tool's per-node lines are.
        rendered = render_node_label(node)
        name = rendered or node.category
        return f"{name}\n[{node.body_label}]\n{pos_str}"

    def _edge_label(e):
        offsets = e.intra_operand_byte_offset
        offsets_str = "(" + ",".join(str(x) for x in offsets) + ")" if offsets else "()"
        slot_str = f"slot {e.src_operand_slot}->{e.sink_operand_slot}"
        return f"{e.edge_kind}\nbytes={offsets_str}\n{slot_str}"

    g = nx.DiGraph()
    seen = set()

    def _put(node):
        nid = id(node)
        if nid not in seen:
            seen.add(nid)
            g.add_node(nid, label=_node_label(node))
        return nid

    for n in graph.nodes.values():
        _put(n)
    for e in graph.edges:
        _put(e.producer)
        _put(e.consumer)

    edge_labels = {}
    highlighted_uv = set()
    hl_set = highlight_edges or set()
    for e in graph.edges:
        u, v = id(e.producer), id(e.consumer)
        is_hl = (e.producer.identity, e.consumer.identity) in hl_set
        lab = _edge_label(e)
        if is_hl:
            lab = f"[{highlight_label}]\n" + lab
            highlighted_uv.add((u, v))
        if (u, v) in edge_labels:
            edge_labels[(u, v)] += "\n" + lab
        else:
            edge_labels[(u, v)] = lab
            g.add_edge(u, v)

    try:
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    except Exception:
        try:
            pos = nx.nx_pydot.graphviz_layout(g, prog="dot")
        except Exception:
            pos = nx.spring_layout(g, seed=42, k=2.0)

    fig, ax = plt.subplots(figsize=(14, 10))
    if title:
        ax.set_title(title, fontsize=12)

    nx.draw_networkx_nodes(g, pos, ax=ax, node_color="lightblue",
                           node_size=3500, edgecolors="black")
    nx.draw_networkx_labels(g, pos,
                            labels={n: g.nodes[n]["label"] for n in g.nodes},
                            ax=ax, font_size=7)
    # arrowsize=40 + min_target_margin=20 lands the arrowhead clearly
    # outside the node circle (node_size=3500 otherwise swallows it).
    normal_edges = [uv for uv in g.edges() if uv not in highlighted_uv]
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=normal_edges,
                           arrows=True, arrowsize=40,
                           edge_color="gray", width=1.5,
                           connectionstyle="arc3,rad=0.1",
                           min_target_margin=20)
    if highlighted_uv:
        nx.draw_networkx_edges(g, pos, ax=ax, edgelist=list(highlighted_uv),
                               arrows=True, arrowsize=40,
                               edge_color="red", width=2.5,
                               style="dashed",
                               connectionstyle="arc3,rad=0.1",
                               min_target_margin=20)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                 ax=ax, font_size=7,
                                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path

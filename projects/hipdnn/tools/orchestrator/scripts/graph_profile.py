#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Read a hipDNN graph and work out which integration cases exercise it.

A flow that is handed an arbitrary graph cannot hard-code the test filter that
validates the kernels written for it: a batchnorm graph and a matmul graph need
different suites, and picking the wrong one produces a run that passes without
testing anything the graph touches.

The mapping is mechanical, not clever. A node's `type` is the attribute class name
(`BatchnormInferenceAttributes`, `MatmulAttributes`); the bundle directories under
`integration-test-bundles/<tier>/` are the same names with `Attributes` dropped
(`BatchnormInference/`, `Matmul/`), and fused bundles concatenate them
(`ConvolutionFwdPointwise/`). So each node type becomes one `<tier>_*<Token>*`
pattern, and a graph's filter is those patterns joined -- which selects the plain
suite for each op the graph uses plus every fused suite that op appears in.

    graph_profile.py --graph g.json --tier quick --out profile.json

`--filter-override` wins outright when non-empty, and the override is recorded in the
output next to the filter it replaced, so a run that used a hand-written filter says
so in its own artifacts rather than in someone's memory.

Exit codes: 0 profile written, 1 the graph is unreadable or declares no nodes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Bundle suites are named `<tier>_<BundleDir>_<Variant>`, and fused bundles concatenate op
#: names (`BatchnormInferencePointwise`, `ConvolutionFwdPointwise`). Anchoring the token
#: between `_` separators is what keeps a single-op graph from dragging in every fusion that
#: merely starts with the same name: `quick_BatchnormInference_*` selects
#: `quick_BatchnormInference_Default` and not `quick_BatchnormInferencePointwise_Default`.
#: A graph that genuinely contains both ops contributes both tokens and selects both.
PATTERN = "{tier}_{token}_*"

ATTRIBUTE_SUFFIX = "Attributes"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, help="hipDNN graph JSON")
    parser.add_argument(
        "--tier",
        default="quick",
        help="bundle tier to select: quick, standard, comprehensive or full (default quick)",
    )
    parser.add_argument(
        "--filter-override",
        default="",
        help="use this GTest filter verbatim instead of the derived one; empty means derive",
    )
    parser.add_argument("--out", required=True, help="where to write the profile JSON")
    args = parser.parse_args()

    try:
        graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"error: could not read graph {args.graph}: {error}", file=sys.stderr)
        return 1

    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        print(f"error: {args.graph} declares no nodes", file=sys.stderr)
        return 1

    node_types: list[str] = []
    for node in nodes:
        node_type = node.get("type") if isinstance(node, dict) else None
        if not isinstance(node_type, str) or not node_type:
            print(f"error: {args.graph} has a node with no type", file=sys.stderr)
            return 1
        if node_type not in node_types:
            node_types.append(node_type)

    tokens = []
    for node_type in node_types:
        token = node_type
        if token.endswith(ATTRIBUTE_SUFFIX):
            token = token[: -len(ATTRIBUTE_SUFFIX)]
        if token and token not in tokens:
            tokens.append(token)

    derived = ":".join(PATTERN.format(tier=args.tier, token=token) for token in tokens)
    override = args.filter_override.strip()

    gtest_filter = override or derived

    profile = {
        "graph": Path(args.graph).resolve().as_posix(),
        "graph_name": graph.get("name", ""),
        "tier": args.tier,
        "node_types": node_types,
        "op_tokens": tokens,
        # Joined here rather than in the flow: a list rendered into a prompt via
        # ${...} comes out as a Python repr, which is not what you want an agent to
        # read back as "the operations in your graph".
        "op_tokens_text": ", ".join(tokens),
        "derived_filter": derived,
        "filter_override": override,
        "gtest_filter": gtest_filter,
        "op_count": len(tokens),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    print(
        f"graph {profile['graph_name'] or Path(args.graph).name}: {len(tokens)} op(s) "
        f"{', '.join(tokens)}"
    )
    print(f"gtest filter: {gtest_filter}" + ("  (override)" if override else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

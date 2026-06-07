#!/usr/bin/env python3
"""Harvest Tensile's own constraint machinery instead of inventing one (v2 p.10).

Two harvests, both static (no instantiation needed):
  1. op-surface: the AST node types ExpressionEvaluator.evaluate dispatches on
     (Configuration.py L606+), mapped to op categories -> the first Z3 encoder boundary.
  2. constraints: every `.addConstraint("<expr>")` call site in the tree, with the
     expression string parsed the same way Tensile parses it (ast.parse(..., mode='exec')).

Writes <outdir>/constraints_harvested.jsonl. Prints a small JSON summary to stdout.
Stdlib only.
"""
import argparse
import ast
import json
import os

OP_CATEGORY = {
    "BoolOp": "boolean", "Compare": "compare", "BinOp": "arith/bit", "UnaryOp": "unary",
    "IfExp": "conditional", "Call": "call", "Name": "leaf", "Attribute": "leaf",
    "Num": "leaf", "Constant": "leaf", "Str": "leaf", "Assign": "assign",
    "Module": "root", "Expr": "root",
}


def harvest_op_surface(src):
    """Find ExpressionEvaluator and the node-type strings it dispatches on."""
    tree = ast.parse(src)
    supported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ExpressionEvaluator":
            for cmp in ast.walk(node):
                # match: nodeType == "X"
                if (isinstance(cmp, ast.Compare) and len(cmp.ops) == 1
                        and isinstance(cmp.ops[0], ast.Eq)
                        and isinstance(cmp.left, ast.Name) and cmp.left.id == "nodeType"
                        and isinstance(cmp.comparators[0], ast.Constant)):
                    supported.append(cmp.comparators[0].value)
    seen, ordered = set(), []
    for s in supported:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def harvest_constraints(root, files):
    out = []
    for rel in files:
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            src = f.read()
        try:
            tree = ast.parse(src, filename=rel)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "addConstraint" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                expr = node.args[0].value
                try:
                    norm = ast.dump(ast.parse(expr, mode="exec"))
                except SyntaxError:
                    norm = "<parse-error>"
                out.append({
                    "kind": "constraint",
                    "file": rel,
                    "line": node.lineno,
                    "expression": expr,
                    "normalized_ast": norm,
                })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--config", default="Tensile/Configuration.py",
                    help="path (rel to root) of the file defining ExpressionEvaluator")
    ap.add_argument("--scan", nargs="*", default=None,
                    help="files (rel to root) to scan for addConstraint call sites; "
                         "default = a small known set")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    with open(os.path.join(a.root, a.config)) as f:
        cfg_src = f.read()
    op_surface = harvest_op_surface(cfg_src)

    scan = a.scan or [
        "Tensile/Configuration.py",
        "Tensile/TensileBenchmarkCluster.py",
    ]
    constraints = harvest_constraints(a.root, scan)

    records = []
    records.append({
        "kind": "op-surface",
        "source": a.config,
        "evaluator": "ExpressionEvaluator",
        "supported_node_types": op_surface,
        "op_categories": sorted({OP_CATEGORY.get(n, "other") for n in op_surface}),
    })
    records.extend(constraints)

    out_path = os.path.join(a.outdir, "constraints_harvested.jsonl")
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(json.dumps({
        "constraints_harvested": len(constraints),
        "op_surface_size": len(op_surface),
        "supported_node_types": op_surface,
        "op_categories": records[0]["op_categories"],
        "constraint_expressions": [c["expression"] for c in constraints],
    }, indent=0))


if __name__ == "__main__":
    main()

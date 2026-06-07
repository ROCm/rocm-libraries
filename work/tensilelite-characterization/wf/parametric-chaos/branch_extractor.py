#!/usr/bin/env python3
"""Stdlib-ast branch census + local def-use seed for the parametric-chaos pipeline (v2).

Walks each target file's AST, emits one record per branch site
(if / elif / while / ifexp / assert / comprehension-if / guard-return / guard-raise),
ranks by the v2 prioritization heuristic, and writes:
  - <outdir>/file_inventory.csv
  - <outdir>/branch_census.jsonl
Prints JSON {"units":[...top N...], "total_branches":N, "files":[...]} to stdout.

Stdlib only. Paths are resolved relative to --root (default: cwd).
"""
import argparse
import ast
import csv
import hashlib
import json
import os
import sys

# Symbols/text that mark a predicate as touching a public input or external state.
EXTERNAL_HINTS = ("os.environ", "os.getenv", "getenv", "os.name", "environ", "platform",
                  "subprocess", "Popen", "which", "os.path.exists", "isfile", "isdir",
                  "Path(", "open(")
EXIT_CALL_HINTS = ("printExit", "sys.exit", "exit(", "SystemExit", "assert")
PUBLIC_SYMBOL_HINTS = ("args", "config", "yaml", "globalParameters", "GlobalParameters",
                       "environ", "argv", "params", "userArgs")


def _names(node):
    return sorted({n.id for n in ast.walk(node) if isinstance(n, ast.Name)})


# Byte-identical, O(n)-amortized replacement for ast.get_source_segment(src, node, padded=False).
# The stdlib helper re-splits the WHOLE source on every call (O(file) each); called per-branch and
# per-statement that is O(file^2) and takes minutes on the 18k-line KernelWriterAssembly.py. Here we
# split each source exactly ONCE (holding a reference so the object identity can't be id-reused) and
# slice the relevant line(s) only. Output is identical to the stdlib (verified by diff on every Run).
_SEG_LAST = [None, None]  # [source_obj, lines]


def _seg(source, node):
    try:
        if node.end_lineno is None or node.end_col_offset is None:
            return None
        lineno = node.lineno - 1
        end_lineno = node.end_lineno - 1
        col_offset = node.col_offset
        end_col_offset = node.end_col_offset
    except AttributeError:
        return None
    if _SEG_LAST[0] is not source:
        _SEG_LAST[0] = source
        _SEG_LAST[1] = ast._splitlines_no_ff(source)
    lines = _SEG_LAST[1]
    if end_lineno == lineno:
        return lines[lineno].encode()[col_offset:end_col_offset].decode()
    first = lines[lineno].encode()[col_offset:].decode()
    last = lines[end_lineno].encode()[:end_col_offset].decode()
    return first + "".join(lines[lineno + 1:end_lineno]) + last


def _first_line(file_src, node):
    try:
        seg = _seg(file_src, node)
        if seg:
            return seg.strip().splitlines()[0][:240]
    except Exception:
        pass
    try:
        return ast.unparse(node)[:240]
    except Exception:
        return "<unparse-failed>"


def _normalize(node):
    """A small, stable normalized form of a predicate AST (operator surface only)."""
    if isinstance(node, ast.BoolOp):
        return {"op": type(node.op).__name__.lower(),
                "args": [_normalize(v) for v in node.values]}
    if isinstance(node, ast.UnaryOp):
        return {"op": type(node.op).__name__.lower(), "arg": _normalize(node.operand)}
    if isinstance(node, ast.BinOp):
        return {"op": type(node.op).__name__.lower(),
                "left": _normalize(node.left), "right": _normalize(node.right)}
    if isinstance(node, ast.Compare):
        return {"op": "compare", "ops": [type(o).__name__ for o in node.ops],
                "left": _normalize(node.left),
                "comparators": [_normalize(c) for c in node.comparators]}
    if isinstance(node, ast.Call):
        fn = getattr(node.func, "id", None) or getattr(node.func, "attr", "call")
        return {"call": fn, "nargs": len(node.args)}
    if isinstance(node, ast.Name):
        return {"var": node.id}
    if isinstance(node, ast.Constant):
        return {"const": node.value if isinstance(node.value, (int, float, bool, str, type(None))) else str(node.value)}
    if isinstance(node, ast.Attribute):
        return {"attr": node.attr}
    return {"node": type(node).__name__}


def _shape(norm):
    """Structural fingerprint ignoring variable names (for recurrence detection)."""
    if not isinstance(norm, dict):
        return str(norm)
    if "var" in norm:
        return "V"
    if "const" in norm:
        return "C"
    if "attr" in norm:
        return "A"
    if "call" in norm:
        return "call(%d)" % norm.get("nargs", 0)
    if "op" in norm:
        if norm["op"] == "compare":
            return "cmp[%s](%s,%s)" % (",".join(norm["ops"]), _shape(norm["left"]),
                                       ",".join(_shape(c) for c in norm["comparators"]))
        if "args" in norm:
            return "%s(%s)" % (norm["op"], ",".join(_shape(a) for a in norm["args"]))
        if "arg" in norm:
            return "%s(%s)" % (norm["op"], _shape(norm["arg"]))
        return "%s(%s,%s)" % (norm["op"], _shape(norm["left"]), _shape(norm["right"]))
    return "?"


def _source_category(text):
    """Tag a derivation source by where the value ultimately comes from (v2 source-category)."""
    t = text
    if "args." in t or "argParser" in t or "parse_args" in t or "sys.argv" in t or "argv" in t:
        return "cli"
    if "os.environ" in t or "getenv" in t or "environ" in t:
        return "env"
    if "os.name" in t or "platform" in t:
        return "os"
    if "globalParameters" in t or "GlobalParameters" in t:
        return "global-parameter"
    if "yaml" in t or "yamlLoad" in t or "safe_load" in t or "load(" in t:
        return "yaml"
    if "os.path" in t or "Path(" in t or "exists(" in t or "isfile" in t or "isdir" in t or "open(" in t:
        return "filesystem"
    if ".get(" in t or "config[" in t or "config." in t or "[" in t:
        return "derived-local"
    return "derived-local"


PUBLIC_SOURCE_HINTS = ("args.", "argParser", "parse_args", "sys.argv", "os.environ", "getenv",
                       "environ", "os.name", "platform", "globalParameters", "GlobalParameters",
                       "yaml", "config", ".get(", "os.path", "Path(")


class DefUseCollector(ast.NodeVisitor):
    """Local (intra-function) def-use: map simple `name = <expr touching a public source>`.

    Records, per (function, name): the assignment source text and its source-category.
    One level of resolution — enough to tie predicate locals (altFormat, configPaths) back
    to their public inputs (args.AlternateFormat, args.ConfigFile)."""

    def __init__(self, file_src):
        self.src = file_src
        self.func_stack = ["<module>"]
        self.maps = {}  # func name -> {symbol: {"derived_from": str, "category": str}}

    def _cur(self):
        return self.maps.setdefault(self.func_stack[-1], {})

    def visit_FunctionDef(self, node):
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                vtext = _seg(self.src, node.value) or ast.unparse(node.value)
            except Exception:
                vtext = ""
            if any(h in vtext for h in PUBLIC_SOURCE_HINTS):
                self._cur()[node.targets[0].id] = {
                    "derived_from": vtext.strip()[:160],
                    "category": _source_category(vtext),
                }
        self.generic_visit(node)


def _body_text(file_src, body):
    out = []
    for st in body:
        try:
            out.append(_seg(file_src, st) or ast.unparse(st))
        except Exception:
            pass
    return "\n".join(out)


def _bid(file, func, line, col, kind):
    return hashlib.sha1(("%s:%s:%d:%d:%s" % (file, func, line, col, kind)).encode()).hexdigest()


class BranchExtractor(ast.NodeVisitor):
    def __init__(self, file_rel, file_src):
        self.file = file_rel
        self.src = file_src
        self.func_stack = ["<module>"]
        self.elif_nodes = set()  # id() of If nodes reached via a parent's orelse
        self.records = []

    def _func(self):
        return self.func_stack[-1]

    def visit_FunctionDef(self, node):
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def _emit(self, node, test, kind, body):
        loc = (getattr(test, "lineno", node.lineno), getattr(test, "col_offset", node.col_offset))
        bid = _bid(self.file, self._func(), node.lineno, node.col_offset, kind)
        body_text = _body_text(self.src, body) if body else ""
        rec = {
            "branch_id": bid,
            "file": self.file,
            "function": self._func(),
            "branch_kind": kind,
            "location": {"line": node.lineno, "col": node.col_offset},
            "predicate_source": _first_line(self.src, test) if test is not None else "",
            "predicate_normalized": _normalize(test) if test is not None else {},
            "referenced_symbols": _names(test) if test is not None else [],
            "_body_text": body_text,
        }
        self.records.append(rec)

    def visit_If(self, node):
        is_elif = id(node) in self.elif_nodes
        body = node.body
        kind = "elif" if is_elif else "if"
        # guard-return / guard-raise: body is exactly one Return / Raise
        if len(body) == 1 and isinstance(body[0], ast.Return):
            kind = "guard-return"
        elif len(body) == 1 and isinstance(body[0], ast.Raise):
            kind = "guard-raise"
        self._emit(node, node.test, kind, body)
        # mark a single-If orelse as elif
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            self.elif_nodes.add(id(node.orelse[0]))
        self.generic_visit(node)

    def visit_While(self, node):
        self._emit(node, node.test, "while", node.body)
        self.generic_visit(node)

    def visit_IfExp(self, node):
        self._emit(node, node.test, "ifexp", [node.body])
        self.generic_visit(node)

    def visit_Assert(self, node):
        self._emit(node, node.test, "assert", [])
        self.generic_visit(node)

    def _comp(self, node):
        for gen in node.generators:
            for cond in gen.ifs:
                self._emit(cond, cond, "comprehension-if", [])
        self.generic_visit(node)

    visit_ListComp = _comp
    visit_SetComp = _comp
    visit_DictComp = _comp
    visit_GeneratorExp = _comp


def rank(records):
    """v2 prioritization heuristic. Higher rank => earlier in the work-list."""
    shape_counts = {}
    for r in records:
        shape_counts[_shape(r["predicate_normalized"])] = shape_counts.get(_shape(r["predicate_normalized"]), 0) + 1
    for r in records:
        pred = r["predicate_source"]
        syms = r["referenced_symbols"]
        body = r.get("_body_text", "")
        derived_names = {d["name"] for d in r.get("derived_symbols", [])}
        public_inputs = sum(1 for s in syms
                            if s in derived_names or any(h.lower() in s.lower() for h in PUBLIC_SYMBOL_HINTS))
        external = any(h in pred for h in EXTERNAL_HINTS) or any(
            d["category"] in ("env", "os", "filesystem") for d in r.get("derived_symbols", []))
        guards_exit = (r["branch_kind"] in ("guard-return", "guard-raise")
                       or any(h in body for h in EXIT_CALL_HINTS)
                       or r["branch_kind"] == "assert")
        dominators = min(len(body.splitlines()), 12)
        recurs = shape_counts.get(_shape(r["predicate_normalized"]), 1)
        score = 0.0
        score += max(0, public_inputs - 1) * 2.0      # (1) >1 public input
        score += 3.0 if external else 0.0             # (2) external state / platform gate
        score += 3.0 if guards_exit else 0.0          # (3) guards error exit / cache / codegen
        score += dominators * 0.25                    # (4) dominates downstream blocks (proxy)
        score += 2.0 if recurs > 1 else 0.0           # (5) recurs across sites
        score += len(syms) * 0.1                       # tie-break: richer predicates first
        r["_score"] = round(score, 3)
        r["_public_input_count"] = public_inputs
        r["_external_state"] = external
        r["_guards_exit"] = guards_exit
    records.sort(key=lambda r: (-r["_score"], r["file"], r["location"]["line"]))
    for i, r in enumerate(records):
        r["rank"] = i
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-units", type=int, default=20)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    all_records = []
    inventory = []
    for rel in a.files:
        path = os.path.join(a.root, rel)
        with open(path, "r") as f:
            src = f.read()
        tree = ast.parse(src, filename=rel)
        du = DefUseCollector(src)
        du.visit(tree)
        ex = BranchExtractor(rel, src)
        ex.visit(tree)
        # attach local def-use: which referenced symbols derive from a public source
        for r in ex.records:
            fmap = du.maps.get(r["function"], {})
            r["derived_symbols"] = [
                {"name": s, "derived_from": fmap[s]["derived_from"], "category": fmap[s]["category"]}
                for s in r["referenced_symbols"] if s in fmap
            ]
        by_kind = {}
        for r in ex.records:
            by_kind[r["branch_kind"]] = by_kind.get(r["branch_kind"], 0) + 1
        inventory.append({
            "file": rel,
            "loc": len(src.splitlines()),
            "branches": len(ex.records),
            "kinds": by_kind,
        })
        all_records.extend(ex.records)

    rank(all_records)

    # file_inventory.csv
    inv_path = os.path.join(a.outdir, "file_inventory.csv")
    with open(inv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "loc", "branch_sites", "if", "elif", "while", "ifexp",
                    "assert", "comprehension-if", "guard-return", "guard-raise"])
        for it in inventory:
            k = it["kinds"]
            w.writerow([it["file"], it["loc"], it["branches"],
                        k.get("if", 0), k.get("elif", 0), k.get("while", 0),
                        k.get("ifexp", 0), k.get("assert", 0), k.get("comprehension-if", 0),
                        k.get("guard-return", 0), k.get("guard-raise", 0)])

    # branch_census.jsonl (full census, ranked)
    census_path = os.path.join(a.outdir, "branch_census.jsonl")
    with open(census_path, "w") as f:
        for r in all_records:
            rec = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(rec) + "\n")

    units = []
    for r in all_records[: a.max_units]:
        units.append({
            "id": r["branch_id"],
            "file": r["file"],
            "function": r["function"],
            "branch_kind": r["branch_kind"],
            "location": r["location"],
            "predicate_source": r["predicate_source"],
            "referenced_symbols": r["referenced_symbols"],
            "derived_symbols": r.get("derived_symbols", []),
            "rank": r["rank"],
            "score": r["_score"],
            "public_input_count": r["_public_input_count"],
            "external_state": r["_external_state"],
            "guards_exit": r["_guards_exit"],
        })
    print(json.dumps({"units": units, "total_branches": len(all_records),
                      "files": [it["file"] for it in inventory]}, indent=0))


if __name__ == "__main__":
    main()

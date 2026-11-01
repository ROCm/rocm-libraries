# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""BURS generator for the gfx11 instruction description (used by isa_tables.py).

The ``[burs]`` section of ``isa/gfx11.isa`` is an iburg-style grammar for
32-bit integer arithmetic: nonterminals (the forms a value takes), terminals
(rocKE integer operations and tree leaves), predicates and rules. Each rule's
cost is the issue cycles the description gives the instruction it emits.
``generate_burs`` turns it into ``src/rir/gen_burs.rs``: the labeler, which
finds the cheapest rule per nonterminal at a node bottom-up (chain rules
closed after), and the binders that list a chosen rule's pattern variables
and interior nodes. ``src/rir/burs.rs`` plans, covers and emits with them.
``isel_hunks`` are the switch-over edits that hook the selector into
``isel.rs`` behind ``ROCKE_DIRECT_DEBUG=burs``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import isa_tables as it

COMMUTE_EMITS = {"v3c"}


@dataclass
class Pat:
    """A pattern node: a terminal with children, a variable or a bare name."""

    term: str | None = None
    kids: list[Pat] | None = None
    var: str | None = None
    nt: str | None = None
    name: str | None = None


@dataclass
class Rule:
    index: int
    lhs: str
    text: str
    pat: Pat
    kind: str  # leaf, chain, pattern
    preds: list[tuple[bool, str, list[str]]]
    emit: str
    emit_vars: list[str]
    inst: str | None
    cost: int


def parse_pattern(text: str) -> Pat:
    pos = 0

    def name() -> str:
        nonlocal pos
        m = re.compile(r"[a-z][a-z0-9_]*").match(text, pos)
        if not m:
            raise it.DescError(f"burs: bad pattern {text!r} at {text[pos:]!r}")
        pos = m.end()
        return m.group(0)

    def node() -> Pat:
        nonlocal pos
        n = name()
        if pos < len(text) and text[pos] == "(":
            pos += 1
            kids = [node()]
            while pos < len(text) and text[pos] == ",":
                pos += 1
                kids.append(node())
            if pos >= len(text) or text[pos] != ")":
                raise it.DescError(f"burs: bad pattern {text!r}")
            pos += 1
            return Pat(term=n, kids=kids)
        if pos < len(text) and text[pos] == ":":
            pos += 1
            return Pat(var=n, nt=name())
        return Pat(name=n)

    p = node()
    if pos != len(text):
        raise it.DescError(f"burs: trailing text in pattern {text!r}")
    return p


def pattern_vars(p: Pat, out: list[tuple[str, str]]) -> None:
    if p.var:
        out.append((p.var, p.nt))
    for k in p.kids or []:
        pattern_vars(k, out)


@dataclass
class Grammar:
    nts: list[str]
    nt_doc: dict[str, str]
    terms: dict[str, it.Record]
    preds: dict[str, str]
    rules: list[Rule]


def parse_grammar(desc: it.Desc) -> Grammar:
    recs = desc.section("burs")
    nts = [r.args[0] for r in recs if r.kind == "nt"]
    nt_doc = {
        r.args[0]: (r.args[1] if len(r.args) > 1 else r.args[0])
        for r in recs
        if r.kind == "nt"
    }
    terms = {r.args[0]: r for r in recs if r.kind == "term"}
    preds = {
        r.args[0]: (r.args[1] if len(r.args) > 1 else r.args[0])
        for r in recs
        if r.kind == "pred"
    }
    unit_issue = {r.args[0]: int(r.get("issue") or 1) for r in desc.section("unit")}
    weight = {r.args[0]: int(r.args[1]) for r in recs if r.kind == "weight"}
    issue = {
        r.args[0]: int(r.get("issue") or unit_issue.get(r.get("unit"), 1))
        * weight.get(r.get("unit"), 1)
        for r in desc.insts
    }
    rules = []
    for r in recs:
        if r.kind != "rule":
            continue
        lhs, ptext = r.args[0], r.args[1]
        if lhs not in nts:
            raise it.DescError(f"{r.where}: unknown nonterminal {lhs}")
        pat = parse_pattern(ptext)
        if pat.name in terms:
            kind = "leaf"
        elif pat.name in nts:
            kind = "chain"
        elif pat.term in terms:
            kind = "pattern"
        else:
            raise it.DescError(
                f"{r.where}: pattern {ptext} names no terminal or nonterminal"
            )
        preds_l = []
        for p in filter(None, (r.get("if") or "").split("&")):
            m = re.fullmatch(r"(!?)([a-z0-9_]+)\(([a-z,]*)\)", p)
            if not m or m.group(2) not in preds:
                raise it.DescError(f"{r.where}: bad predicate {p}")
            preds_l.append((m.group(1) == "!", m.group(2), m.group(3).split(",")))
        emit, evars = "", []
        if r.get("emit"):
            m = re.fullmatch(r"([a-z0-9]+)(?:\(([a-z,]*)\))?", r.get("emit"))
            if not m:
                raise it.DescError(f"{r.where}: bad emit {r.get('emit')}")
            emit, evars = m.group(1), (m.group(2).split(",") if m.group(2) else [])
        vars_ = []
        pattern_vars(pat, vars_)
        names = [v for v, _ in vars_]
        for v in evars + [a for _, _, args in preds_l for a in args]:
            if v not in names:
                raise it.DescError(f"{r.where}: variable {v} is not in the pattern")
        inst = r.get("inst")
        if kind == "pattern" and inst not in issue:
            raise it.DescError(
                f"{r.where}: inst {inst} is not an opcode of the description"
            )
        if kind == "pattern" and emit not in ("hand", "v3", "v3c", "lshladd"):
            raise it.DescError(f"{r.where}: unknown emit {emit}")
        if emit == "hand" and not terms[pat.term].get("kind"):
            raise it.DescError(f"{r.where}: emit=hand needs a terminal with an IR kind")
        cost = issue[inst] if kind == "pattern" else 0
        rules.append(
            Rule(
                len(rules),
                lhs,
                f"{lhs} {ptext}",
                pat,
                kind,
                preds_l,
                emit,
                evars,
                inst,
                cost,
            )
        )
    return Grammar(nts, nt_doc, terms, preds, rules)


def camel(n: str) -> str:
    return it.camel(n)


def _walk(
    p: Pat,
    expr: str,
    items: list,
    binds: dict[str, str],
    interior: list[str],
    counter: list[int],
) -> None:
    """Lets and term guards that reach pattern `p` at node `expr`."""
    for i, k in enumerate(p.kids or []):
        if k.var:
            name = f"v_{k.var}"
            items.append(("let", name, f"t.kid({expr}, {i})"))
            binds[k.var] = name
        else:
            counter[0] += 1
            name = f"x{counter[0]}"
            items.append(("let", name, f"t.kid({expr}, {i})"))
            items.append(("if", f"t.term({name}) == Term::{camel(k.term)}"))
            interior.append(name)
            _walk(k, name, items, binds, interior, counter)


def _pred_expr(neg: bool, name: str, args: list[str], binds: dict[str, str]) -> str:
    call = f"t.{name}({', '.join(binds[a] for a in args)})"
    return f"!{call}" if neg else call


def generate_burs(desc: it.Desc) -> str:
    g = parse_grammar(desc)
    nt_idx = {n: i for i, n in enumerate(g.nts)}
    o = [
        it.HEADER_RS,
        f"//! @generated by `{it.TOOL} generate` (isa_burs.py) from the `[burs]` section of",
        f"//! `{it.DESCRIPTION}`. Do not edit: change the grammar in the overlay, then rerun.",
        "//!",
        "//! The BURS grammar for 32-bit integer arithmetic: nonterminals, terminals,",
        "//! rules with their costs (the issue cycles of the instruction each emits),",
        "//! the labeler, which finds the cheapest rule per nonterminal at a node whose",
        "//! kids are labelled, and the binders that list a chosen rule's pattern",
        "//! variables and interior nodes. `burs.rs` plans, covers and emits with them.",
        "",
        "#![allow(dead_code)]",
        "",
        "use super::burs::{Cell, Tree};",
        "use crate::rir::ast::OpKind;",
        "use crate::rir::mir::Opc;",
        "",
        f"pub const NTS: usize = {len(g.nts)};",
        "",
        "/// Nonterminals: the forms a value takes.",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]",
        "pub enum Nt {",
    ]
    for n in g.nts:
        o.append(f"    /// {g.nt_doc[n]}")
        o.append(f"    {camel(n)} = {nt_idx[n]},")
    o += [
        "}",
        "",
        "/// Terminals: integer operations and tree leaves.",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]",
        "pub enum Term {",
    ]
    for t in g.terms:
        o.append(f"    {camel(t)},")
    o += [
        "}",
        "",
        "/// The terminal of an IR operation, if the grammar has one.",
        "pub fn term_of(kind: OpKind) -> Option<Term> {",
        "    match kind {",
    ]
    for t, r in g.terms.items():
        if r.get("kind"):
            o.append(f"        OpKind::{r.get('kind')} => Some(Term::{camel(t)}),")
    o += ["        _ => None,", "    }", "}", ""]
    leaves = {r.get("leaf"): t for t, r in g.terms.items() if r.get("leaf")}
    o.append("/// The leaf terminals: constants, uniform and divergent registers.")
    o.append(f"pub const KONST: Term = Term::{camel(leaves['konst'])};")
    o.append(f"pub const SREG: Term = Term::{camel(leaves['sreg'])};")
    o.append(f"pub const VREG: Term = Term::{camel(leaves['vreg'])};")
    o += [
        "",
        "/// What reducing a rule emits.",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq)]",
        "pub enum Action {",
        "    /// A leaf: the value computed elsewhere.",
        "    Leaf,",
        "    /// A chain rule: the node as another nonterminal.",
        "    Chain(Nt),",
        "    /// One operation, through isel's own selection of it (`int_bin`).",
        "    Hand(OpKind),",
        "    /// A three-source VOP3 of pattern variables `vars`; `commute`: put an",
        "    /// inner commutative operation's VGPR source second, as isel orders it.",
        "    V3 { op: Opc, vars: [u8; 3], commute: bool },",
        "    /// `s_lshl<k>_add_u32 a, b` for variables (a, k, b).",
        "    LshlAdd { vars: [u8; 3] },",
        "}",
        "",
        "/// A rule: left-hand side, text, cost and action.",
        "pub struct Rule {",
        "    pub lhs: Nt,",
        "    pub text: &'static str,",
        "    pub cost: u32,",
        "    pub action: Action,",
        "}",
        "",
        f"pub const RULES: [Rule; {len(g.rules)}] = [",
    ]
    for r in g.rules:
        if r.kind == "leaf":
            action = "Action::Leaf"
        elif r.kind == "chain":
            action = f"Action::Chain(Nt::{camel(r.pat.name)})"
        else:
            names = []
            pattern_vars(r.pat, names)
            order = [v for v, _ in names]
            if r.emit == "hand":
                action = f"Action::Hand(OpKind::{g.terms[r.pat.term].get('kind')})"
            elif r.emit in ("v3", "v3c"):
                idx = ", ".join(str(order.index(v)) for v in r.emit_vars)
                action = f"Action::V3 {{ op: Opc::{r.inst}, vars: [{idx}], commute: {str(r.emit in COMMUTE_EMITS).lower()} }}"
            else:
                idx = ", ".join(str(order.index(v)) for v in r.emit_vars)
                action = f"Action::LshlAdd {{ vars: [{idx}] }}"
        o.append(
            f'    Rule {{ lhs: Nt::{camel(r.lhs)}, text: "{r.text}", cost: {r.cost}, action: {action} }},'
        )
    o.append("];")
    o.append("")

    # The labeler.
    by_term: dict[str, list[Rule]] = {}
    for r in g.rules:
        if r.kind == "leaf":
            by_term.setdefault(r.pat.name, []).append(r)
        elif r.kind == "pattern":
            by_term.setdefault(r.pat.term, []).append(r)
    o += [
        "/// The cheapest rule per nonterminal at node `n`, whose kids are labelled.",
        "/// Rules for `s` apply at uniform nodes, rules for `v` at divergent ones.",
        "pub fn label(t: &Tree, n: usize, uniform: bool) -> [Cell; NTS] {",
        "    let mut c = [Cell::NONE; NTS];",
        "    match t.term(n) {",
    ]
    for term in g.terms:
        rules = by_term.get(term, [])
        if not rules:
            continue
        o.append(f"        Term::{camel(term)} => {{")
        for r in rules:
            gate = {"s": "uniform", "v": "!uniform"}.get(r.lhs)
            body: list[str] = []
            if r.kind == "leaf":
                body.append(f"c[{nt_idx[r.lhs]}].offer(0, {r.index});")
            else:
                items: list = []
                binds: dict[str, str] = {}
                _walk(r.pat, "n", items, binds, [], [0])
                for neg, name, args in r.preds:
                    items.append(("if", _pred_expr(neg, name, args, binds)))
                vars_: list[tuple[str, str]] = []
                pattern_vars(r.pat, vars_)
                cost = " + ".join(
                    [f"t.cost({binds[v]}, Nt::{camel(nt)})" for v, nt in vars_]
                    + [str(r.cost)]
                )
                depth = 0
                for kind, *rest in items:
                    pad = "    " * depth
                    if kind == "let":
                        body.append(f"{pad}let {rest[0]} = {rest[1]};")
                    else:
                        body.append(f"{pad}if {rest[0]} {{")
                        depth += 1
                body.append(
                    "    " * depth + f"c[{nt_idx[r.lhs]}].offer({cost}, {r.index});"
                )
                for d in range(depth, 0, -1):
                    body.append("    " * (d - 1) + "}")
            o.append(f"            // {r.index}: {r.text}")
            if gate:
                o.append(f"            if {gate} {{")
                o += [f"                {line}" for line in body]
                o.append("            }")
            else:
                o.append("            {")
                o += [f"                {line}" for line in body]
                o.append("            }")
        o.append("        }")
    if any(not by_term.get(t) for t in g.terms):
        o.append("        _ => {}")
    o.append("    }")
    chains = [r for r in g.rules if r.kind == "chain"]
    o.append("    // Chain rules, to a fixed point.")
    o.append(f"    for _ in 0..{max(1, len(g.nts) - 1)} {{")
    for r in chains:
        o.append(f"        let k = c[{nt_idx[r.pat.name]}].cost;")
        o.append(f"        c[{nt_idx[r.lhs]}].offer(k, {r.index});")
    o += ["    }", "    c", "}", ""]

    # The binders.
    o += [
        "/// The variables of `rule`'s pattern at node `n`, in pattern order: (node,",
        "/// nonterminal). Returns how many.",
        "pub fn vars(t: &Tree, n: usize, rule: usize, out: &mut [(usize, Nt); 3]) -> usize {",
        "    match rule {",
    ]
    for r in g.rules:
        if r.kind != "pattern":
            continue
        items: list = []
        binds = {}
        _walk(r.pat, "n", items, binds, [], [0])
        vars_ = []
        pattern_vars(r.pat, vars_)
        o.append(f"        {r.index} => {{")
        for kind, *rest in items:
            if kind == "let":
                o.append(f"            let {rest[0]} = {rest[1]};")
        for i, (v, nt) in enumerate(vars_):
            o.append(f"            out[{i}] = ({binds[v]}, Nt::{camel(nt)});")
        o.append(f"            {len(vars_)}")
        o.append("        }")
    o += ["        _ => 0,", "    }", "}", ""]
    o += [
        "/// The interior nodes of `rule`'s pattern at node `n`: operations the",
        "/// pattern folds into the instruction it emits. Returns how many.",
        "pub fn interior(t: &Tree, n: usize, rule: usize, out: &mut [usize; 2]) -> usize {",
        "    match rule {",
    ]
    for r in g.rules:
        if r.kind != "pattern":
            continue
        items = []
        interior: list[str] = []
        _walk(r.pat, "n", items, {}, interior, [0])
        if not interior:
            continue
        o.append(f"        {r.index} => {{")
        for kind, *rest in items:
            if kind == "let" and rest[0] in interior:
                o.append(f"            let {rest[0]} = {rest[1]};")
        for i, name in enumerate(interior):
            o.append(f"            out[{i}] = {name};")
        o.append(f"            {len(interior)}")
        o.append("        }")
    o += ["        _ => 0,", "    }", "}"]
    return "\n".join(o) + "\n"


# --------------------------------------------------------------------------
# Switch-over edits in isel.rs
# --------------------------------------------------------------------------

MOD_OLD = "fn unsupported(message: impl Into<String>) -> CompileError {"
MOD_NEW = (
    '#[path = "burs.rs"]\nmod burs;\n#[path = "gen_burs.rs"]\n#[rustfmt::skip]\nmod gen_burs;\n\n'
    "fn unsupported(message: impl Into<String>) -> CompileError {"
)
FIELD_OLD = "struct Isel<'a> {\n"
FIELD_NEW = (
    "struct Isel<'a> {\n"
    "    /// The BURS prototype's plan (`ROCKE_DIRECT_DEBUG=burs`), else `None`.\n"
    "    burs: Option<burs::Plan>,\n"
)
INIT_OLD = "    let mut isel = Isel {\n"
INIT_NEW = "    let mut isel = Isel {\n        burs: None,\n"
PLAN_OLD = "    isel.f.blocks.push(MBlock::default());\n"
PLAN_NEW = (
    '    if super::debug_flag("burs") {\n        isel.burs = Some(burs::plan(&isel));\n    }\n'
    "    isel.f.blocks.push(MBlock::default());\n"
)
HOOK_OLD = "        if res.is_some_and(|v| self.chains.skip.contains(&v)) {\n            return Ok(());\n        }\n"
HOOK_NEW = (
    HOOK_OLD
    + "        if self.burs.is_some() && burs::select(self, op)? {\n            return Ok(());\n        }\n"
)


def _once(old: str, new: str):
    return lambda text: it._replace_once(text, old, new)


def isel_hunks() -> list[tuple[str, str, object]]:
    return [
        ("isel: declare the BURS selector", "mod burs;", _once(MOD_OLD, MOD_NEW)),
        (
            "isel: BURS plan field",
            "burs: Option<burs::Plan>,",
            _once(FIELD_OLD, FIELD_NEW),
        ),
        (
            "isel: BURS plan initialised",
            "        burs: None,",
            _once(INIT_OLD, INIT_NEW),
        ),
        (
            "isel: BURS plan when ROCKE_DIRECT_DEBUG=burs",
            "burs::plan(&isel)",
            _once(PLAN_OLD, PLAN_NEW),
        ),
        (
            "isel: BURS selection hook",
            "burs::select(self, op)?",
            _once(HOOK_OLD, HOOK_NEW),
        ),
    ]

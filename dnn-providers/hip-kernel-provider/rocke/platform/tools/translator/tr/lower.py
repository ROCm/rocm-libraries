# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""S5 Lower: Python function bodies to C++ in A-normal form.

Every call that can emit IR is bound to its own local, inner calls first, left
to right, so the C++ creates IR ops in exactly Python's order (the plan's core
invariant: no two emitting calls share one C++ expression). Anything the rules
do not cover raises Refuse, which turns the whole function into a typed hole.

Two passes per function: a dry pass records the C++ type of every local and of
every comprehension element (an in-translator recorder), then the real pass
emits code with declarations hoisted to the block that dominates every use.
"""

from __future__ import annotations

import ast
import copy
import functools
import itertools
import math
import re
import sys
from dataclasses import dataclass, field

from . import ctypes as T
from .env import Env
from .factkeys import code_keys, first_line, src_hash
from .family import family_of, kind_const, kind_type, union_fields, union_kind
from .package import PKG
from .host import HOST_VALUE, host_builders


class Refuse(Exception):
    def __init__(self, node, reason: str):
        super().__init__(reason)
        self.node = node
        self.reason = reason
        self.line = getattr(node, "lineno", 0)


@dataclass
class CE:
    code: str
    t: str
    pure: bool = True  # no IR emitted, no host side effect: may be inlined
    lit: object = (
        None  # Python value when the expression is a translation-time constant
    )
    meta: dict = None  # encoding details (owned-list count and capacity fields)


def lit_code(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        if -(2**31) <= v < 2**31:
            return str(v)
        return f"INT64_C({v})"
    if isinstance(v, float):
        if math.isinf(v):
            return "INFINITY" if v > 0 else "-INFINITY"
        if math.isnan(v):
            return "NAN"
        return repr(v)
    if isinstance(v, str):
        return c_string(v)
    raise ValueError(v)


def c_string(s: str) -> str:
    out = '"'
    for ch in s:
        if ch in '"\\':
            out += "\\" + ch
        elif ch == "\n":
            out += "\\n"
        elif ord(ch) < 32 or ord(ch) > 126:
            out += f"\\x{ord(ch):02x}"
        else:
            out += ch
    return out + '"'


def lit_type(v) -> str:
    if isinstance(v, bool):
        return T.BOOL
    if isinstance(v, int):
        return T.I64
    if isinstance(v, float):
        return T.F64
    if isinstance(v, str):
        return T.STR
    if v is None:
        return T.NONE
    raise ValueError(v)


# ---------------------------------------------------------------- module ctx


@dataclass
class Sig:
    """Internal C++ signature of a translated Python function."""

    name: str  # C++ name
    params: list  # [(pyname, ctype)]
    ret: str
    py: ast.FunctionDef = None
    defaults: dict = field(default_factory=dict)  # pyname -> ast default
    host: str | None = None  # host clones: the host builder class of a host clone


CPP_KEYWORDS = frozenset(
    "alignas alignof asm auto bool case catch char char8_t char16_t char32_t const consteval "
    "constexpr constinit const_cast decltype default delete do double dynamic_cast enum explicit "
    "export extern float friend goto inline int long mutable namespace new noexcept operator "
    "private protected public register reinterpret_cast short signed sizeof static static_assert "
    "static_cast struct switch template this thread_local throw typedef typeid typename union "
    "unsigned using virtual void volatile wchar_t".split()
)


def rename_cpp_keywords(tree: ast.Module) -> ast.Module:
    """Locals and parameters named like a C++ keyword (``do``, ``signed``)
    get a trailing underscore. Keyword arguments keep the Python name; the
    binding maps them (bind_call_args), and recorded facts are looked up by
    the Python name (Module.fact_of)."""
    params = {a.arg for a in ast.walk(tree) if isinstance(a, ast.arg)}
    stored = {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    names = (stored | params) & CPP_KEYWORDS  # never builtins like float()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id in names:
            n.id += "_"
        if isinstance(n, ast.arg) and n.arg in names:
            n.arg += "_"  # callers' keyword arguments bind to it by the old name
    return tree


def record_return(fn: ast.FunctionDef) -> bool:
    """R-REC: a function annotated to return a dict of mixed values
    (dict[str, Any]) returns a record: a struct of its literal keys."""
    if fn.returns is None:
        return False
    t = ast.unparse(fn.returns)
    return bool(re.fullmatch(r"(?:dict|Dict)\[str, ?Any\]|dict", t))


def record_locals(fn: ast.FunctionDef) -> set:
    """Locals bound to a dict display with literal str keys whose every use
    is a literal-key read / write, .get(literal), `literal in d`, a truth test
    or a return: records (R-REC)."""
    cand = set()
    for n in ast.walk(fn):
        tgt = (
            n.targets[0]
            if isinstance(n, ast.Assign) and len(n.targets) == 1
            else (n.target if isinstance(n, ast.AnnAssign) else None)
        )
        if isinstance(tgt, ast.Name) and isinstance(
            getattr(n, "value", None), ast.Dict
        ):
            if all(
                isinstance(k, ast.Constant) and isinstance(k.value, str)
                for k in n.value.keys
            ):
                cand.add(tgt.id)
    if not cand:
        return set()
    parents = {}
    for p in ast.walk(fn):
        for c in ast.iter_child_nodes(p):
            parents[c] = p
    bad = set()
    for n in ast.walk(fn):
        if not (isinstance(n, ast.Name) and n.id in cand):
            continue
        p = parents.get(n)
        lit = lambda x: isinstance(x, ast.Constant) and isinstance(x.value, str)
        ok = (
            (isinstance(p, ast.Subscript) and p.value is n and lit(p.slice))
            or (isinstance(p, ast.Assign) and n in p.targets)
            or (isinstance(p, ast.AnnAssign) and p.target is n)
            or (isinstance(p, ast.Assign) and p.value is n and False)
            or (
                isinstance(p, ast.Attribute)
                and p.attr == "get"
                and isinstance(parents.get(p), ast.Call)
                and parents[p].args
                and lit(parents[p].args[0])
            )
            or (
                isinstance(p, ast.Compare)
                and isinstance(p.ops[0], (ast.In, ast.NotIn))
                and p.comparators[0] is n
                and lit(p.left)
            )
            or (isinstance(p, (ast.If, ast.While, ast.BoolOp, ast.IfExp)))
            or (isinstance(p, ast.UnaryOp) and isinstance(p.op, ast.Not))
            or isinstance(p, ast.Return)
        )
        if not ok:
            bad.add(n.id)
    return cand - bad


def none_aliases(fn: ast.FunctionDef) -> dict:
    """flag = x is None / x is not None, each name assigned once: testing flag
    narrows x like the test itself. {flag: Compare node}."""
    stores = {}
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            stores[n.id] = stores.get(n.id, 0) + 1
    params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    out = {}
    for n in ast.walk(fn):
        if (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
        ):
            v = n.value
            if (
                isinstance(v, ast.Compare)
                and len(v.ops) == 1
                and isinstance(v.ops[0], (ast.Is, ast.IsNot))
                and isinstance(v.left, ast.Name)
                and isinstance(v.comparators[0], ast.Constant)
                and v.comparators[0].value is None
            ):
                flag, x = n.targets[0].id, v.left.id
                if stores.get(flag) == 1 and stores.get(x, 0) == (
                    0 if x in params else 1
                ):
                    out[flag] = v
    return out


def record_field(k: str) -> str:
    f = re.sub(r"\W", "_", k)
    if f[:1].isdigit():
        f = "k_" + f
    return f + "_" if f in CPP_KEYWORDS else f


class Module:
    def __init__(self, env: Env, modname: str, path: str):
        self.env = env
        self.modname = modname
        self.path = path
        self.src = open(path).read()
        self.tree = rename_cpp_keywords(ast.parse(self.src))
        self.pymod = env.py.module(modname)
        self.imports: dict[str, tuple[str, str]] = {}
        self.functions: dict[str, ast.FunctionDef] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        self.sigs: dict[str, Sig] = {}
        self.private_consts: dict[str, tuple[str, str]] = {}  # pyname -> (cname, decl)
        self.records: dict[str, dict] = {}  # record dict id -> {key: C++ type} (R-REC)
        # module-level type aliases (Frags = Sequence[Sequence[Value]])
        self.type_aliases: dict[str, str] = {}
        for n in self.tree.body:
            if (
                isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and isinstance(n.value, (ast.Subscript, ast.BinOp))
            ):
                txt = ast.unparse(n.value)
                if re.match(
                    r"(Sequence|list|tuple|dict|Optional|Callable)\[|\w+ \|", txt
                ):
                    self.type_aliases[n.targets[0].id] = txt
        for n in self.tree.body:
            if isinstance(n, ast.ImportFrom):
                # a sibling stays bare; a parent-package import becomes absolute
                mod = PKG.import_source(n.module, n.level)
                for a in n.names:
                    self.imports[a.asname or a.name] = (mod, a.name)
            elif isinstance(n, ast.FunctionDef):
                self.functions[n.name] = n
            elif isinstance(n, ast.ClassDef):
                self.classes[n.name] = n
        self.class_struct: dict[str, str] = {}
        self.struct_class: dict[str, str] = {}
        self.methods: dict[tuple, ast.FunctionDef] = {}
        for cname, cnode in self.classes.items():
            for m in cnode.body:
                if isinstance(m, ast.FunctionDef):
                    self.methods[(cname, m.name)] = m
        self.kw_base: dict[int, ast.FunctionDef] = (
            {}
        )  # id(method) -> the base method its **kw feeds
        self._kw_busy: set = set()
        for key in self.methods:
            self.expand_kw_forward(*key)
        from .kernel import load_facts, qualnames

        self.facts = load_facts(modname)
        self.qualname = qualnames(self.tree)
        # facts keys and source stamps come from the unrenamed source
        self.fact_keys = {
            (k.split("#")[0], first_line(n)): (k, n)
            for k, n in code_keys(ast.parse(self.src))
            if isinstance(n, ast.FunctionDef)
        }
        self.stale_warned: set[str] = set()
        self.tmap = None  # set by translate (needs the class maps)
        self.dispatchers: dict[str, list[str]] = (
            {}
        )  # tagged values: internal kind-tag dispatchers
        self.host_demand: set = (
            set()
        )  # host clones: (unit, class or None, host builder) clones callers asked for
        self.host_images: dict = (
            {}
        )  # host clones: (IR struct, host builder) -> host image struct (designs/host_clones.md)
        self.host_image_defs: dict = (
            {}
        )  # host clones: host image struct -> its C++ definition lines
        self.fence_methods = {
            "s_barrier_bare": "rocke_b_s_barrier_bare_fence",
            "sync": "rocke_b_sync_fence",
            "sync_lds_only": "rocke_b_sync_lds_only_fence",
        }

    def bind_classes(self, twin_symbols: dict):
        """Python class -> C struct, from twin_map."""
        for cname in self.classes:
            ent = twin_symbols.get(self.qual(cname)) or {}
            for tw in ent.get("twins", []):
                sym = tw.get("symbol") or tw.get("c") or tw.get("name")
                if sym in self.env.headers.structs:
                    self.class_struct[cname] = sym
                    self.struct_class[sym] = cname
                    break

    def method_owner(self, cls: str, meth: str):
        """The class (cls or a base in this module) that defines cls.meth."""
        seen = set()
        while cls in self.classes and cls not in seen:
            seen.add(cls)
            if (cls, meth) in self.methods:
                return cls
            cls = next(
                (
                    ast.unparse(b)
                    for b in self.classes[cls].bases
                    if ast.unparse(b) in self.classes
                ),
                None,
            )
        return None

    def expand_kw_forward(self, cls: str, meth: str) -> None:
        """KW-F1: a method that only forwards **kw to super().meth(...) takes
        the base method's parameters it does not pass itself, as keyword-only
        parameters, and the super() call passes them by name.
        KW-F2: the same when the one forward is to a method of this class,
        self.other(..., **kw) (resolved through the module's bases)."""
        fn = self.methods[(cls, meth)]
        kw = fn.args.kwarg.arg if fn.args.kwarg else None
        if kw is None or id(fn) in self.kw_base:
            return
        uses = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == kw]
        fwd_calls = [
            c
            for c in ast.walk(fn)
            if isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and any(
                k.arg is None and isinstance(k.value, ast.Name) and k.value.id == kw
                for k in c.keywords
            )
        ]
        if len(fwd_calls) != 1 or len(uses) != 1:
            return  # **kw read elsewhere: not a plain forward
        call = fwd_calls[0]
        tgt, recv = call.func.attr, call.func.value
        self_name = (
            (fn.args.posonlyargs + fn.args.args)[0].arg
            if fn.args.posonlyargs + fn.args.args
            else None
        )
        if (
            isinstance(recv, ast.Call)
            and ast.unparse(recv) == "super()"
            and tgt == meth
        ):
            base = next(
                (
                    ast.unparse(b)
                    for b in self.classes[cls].bases
                    if (ast.unparse(b), meth) in self.methods
                ),
                None,
            )
        elif (
            isinstance(recv, ast.Name)
            and self_name is not None
            and recv.id == self_name
            and not any(
                isinstance(d, ast.Name) and d.id in ("staticmethod", "classmethod")
                for d in fn.decorator_list
            )
        ):
            base = self.method_owner(cls, tgt)
            if (base, tgt) == (cls, meth):
                return  # recursion through **kw
        else:
            return
        if base is None or (base, tgt) in self._kw_busy:
            return
        self._kw_busy.add((cls, meth))
        self.expand_kw_forward(base, tgt)
        self._kw_busy.discard((cls, meth))
        bfn = self.methods[(base, tgt)]
        if bfn.args.kwarg or bfn.args.vararg:
            return
        b = bfn.args
        plain = b.posonlyargs + b.args
        pdefs = [None] * (len(plain) - len(b.defaults)) + list(b.defaults)
        passed = {p.arg for p in plain[1 : 1 + len(call.args)]} | {
            k.arg for k in call.keywords if k.arg
        }
        own = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
        fwd = [
            (a, d)
            for a, d in list(zip(plain, pdefs))[1 + len(b.posonlyargs) :]
            + list(zip(b.kwonlyargs, b.kw_defaults))
            if a.arg not in passed | own
        ]
        fn.args.kwonlyargs += [copy.deepcopy(a) for a, _ in fwd]
        fn.args.kw_defaults += [copy.deepcopy(d) for _, d in fwd]
        fn.args.kwarg = None
        call.keywords = [k for k in call.keywords if k.arg is not None] + [
            ast.copy_location(
                ast.keyword(
                    arg=a.arg,
                    value=ast.copy_location(ast.Name(a.arg, ast.Load()), call),
                ),
                call,
            )
            for a, _ in fwd
        ]
        self.kw_base[id(fn)] = bfn

    def fact_of(self, fn: ast.FunctionDef):
        if id(fn) in self.kw_base:
            # the forwarded parameters are typed as the base method records them
            own, base = self._fact_of(fn), self.fact_of(self.kw_base[id(fn)])
            if base is None:
                return own
            names = {a.arg for a in fn.args.kwonlyargs}
            params = {n: t for n, t in base.get("params", {}).items() if n in names}
            return dict(
                own or {"return": []},
                params={**params, **((own or {}).get("params", {}))},
            )
        return self._fact_of(fn)

    def _fact_of(self, fn: ast.FunctionDef):
        q = self.qualname.get(id(fn))
        funcs = self.facts.get("funcs", {})
        f = None
        if q:
            key, node = self.fact_keys.get((q, first_line(fn)), (q, None))
            f = funcs.get(key)
            if f is None:
                # facts recorded before the qualname keys: qualname@co_firstlineno
                f = funcs.get(f"{q}@{first_line(fn)}") or funcs.get(f"{q}@{fn.lineno}")
            elif (
                node is not None
                and f.get("src")
                and key not in self.stale_warned
                and f["src"] != src_hash(node)
            ):
                self.stale_warned.add(key)
                print(
                    f"warning: facts stale for {self.modname}.{key}; re-run tr.record",
                    file=sys.stderr,
                )
        if f and any(n in CPP_KEYWORDS for n in f.get("params", {})):
            # parameters renamed like C++ keywords (signed -> signed_)
            f = dict(
                f,
                params={
                    (n + "_" if n in CPP_KEYWORDS else n): t
                    for n, t in f["params"].items()
                },
            )
        return f

    def family(self, cls: str | None):
        """cls's family when this module owns it and the family struct is the
        generated tagged one (tagged values); None otherwise (hand ABI, no family)."""
        fam = family_of(self, cls) if cls else None
        if fam is None or fam.module != self.modname:
            return None
        sname = self.class_struct.get(fam.base)
        s = self.env.headers.structs.get(sname) if sname else None
        if s is None or s.fields.get("kind") != kind_type(fam):
            return None
        return fam

    def kind_test(self, fam, cls: str, code: str, is_p: bool) -> str:
        """`isinstance(x, cls)` of a family value: a test of its kind tag."""
        ks = [kind_const(fam, m) for m in fam.derived(cls) if not fam.is_peer(m)]
        k = f"({code}){'->' if is_p else '.'}kind"
        if cls == fam.base and not fam.peers:
            return "true"
        return " || ".join(f"{k} == {c}" for c in ks)

    def dispatcher(self, fam, scls: str, attr: str) -> "Sig":
        """tagged values: the internal dispatcher of attr on a scls value: a switch on the
        kind tag to each member's implementation (designs/tagged_values.md)."""
        impls = fam.impls(scls, attr)
        if any(fam.is_peer(m) for m in impls):
            raise Refuse(
                None,
                f"{scls}.{attr} may be a duck-typed peer's (dispatch to a hand C entry from inside)",
            )
        owners = list(dict.fromkeys(impls.values()))
        dflt = impls.get(scls, owners[0])
        sigs = {}
        for o in owners:
            sg = self.sigs.get(f"{o}.{attr}")
            if sg is None:
                raise Refuse(None, f"method {o}.{attr} is not translated")
            sigs[o] = sg
        s0 = sigs[dflt]
        for o, sg in sigs.items():
            if [t for _, t in sg.params] != [
                t for _, t in s0.params
            ] or sg.ret != s0.ret:
                raise Refuse(
                    None, f"{attr}: family members' signatures differ ({o} and {dflt})"
                )
        name = f"py_{scls}__{attr}"
        if name not in self.dispatchers:
            ps = []
            for n, t in s0.params:
                dt = decl_type(t)
                ps.append(f"{'const ' + dt + '&' if T.is_seq(t) else dt} {n}")
            args = ", ".join(n for n, _ in s0.params)
            sv = s0.params[0][0]
            sep = "->" if s0.params[0][1].endswith("*") else "."
            void = s0.ret == T.VOID
            lines = [
                f"/* tagged-value dispatch: {scls}.{attr} on the kind tag */",
                f"{decl_type(s0.ret)} {name}({', '.join(ps)})",
                "{",
                f"    switch({sv}{sep}kind)",
                "    {",
            ]
            for m, o in impls.items():
                if o != dflt:
                    call = f"{sigs[o].name}({args})"
                    lines.append(f"    case {kind_const(fam, m)}:")
                    lines.append(
                        f"        {call};" if void else f"        return {call};"
                    )
                    if void:
                        lines.append("        return;")
            call = f"{sigs[dflt].name}({args})"
            lines += [
                "    default:",
                f"        {call};" if void else f"        return {call};",
            ]
            if void:
                lines.append("        return;")
            lines += ["    }", "}"]
            self.dispatchers[name] = lines
        return Sig(name, list(s0.params), s0.ret, s0.py, dict(s0.defaults))

    def class_of(self, ctype: str) -> str | None:
        st = T.struct_of(ctype)
        return self.struct_class.get(st[0]) if st else None

    def qual(self, name: str) -> str:
        return f"{self.modname}:{name}"


# ------------------------------------------------------------ declaration plan


class DeclPlan:
    """Where each local is declared: the block that dominates every occurrence.

    A block is a statement list; an occurrence is recorded with its stack of
    (block id, statement index) pairs. Names read inside a nested def count as
    read at the def statement (a lambda captures by reference, so the name must
    exist before it).
    """

    def __init__(self, fn: ast.FunctionDef, params: set[str]):
        self.params = params
        self.occ: dict[str, list] = {}  # name -> [(pos, stack, is_store, stmt)]
        self.walk_block(fn.body, [])
        self.decl: dict[str, tuple] = {}
        # only names the function binds are locals; the rest are globals
        self.occ = {n: o for n, o in self.occ.items() if any(x[2] for x in o)}
        for name, occs in self.occ.items():
            if name in params:
                continue
            stacks = [o[1] for o in occs]
            d = 0
            while (
                all(len(s) > d for s in stacks) and len({s[d][0] for s in stacks}) == 1
            ):
                d += 1
            first = min(occs, key=lambda o: o[0])
            idx = min(s[d - 1][1] for s in stacks)
            inline = (
                first[2]
                and len(first[1]) == d
                and first[1][d - 1][1] == idx
                and isinstance(first[3], (ast.Assign, ast.AnnAssign))
                and self._simple_target(first[3], name)
            )
            self.decl[name] = (stacks[0][d - 1][0], idx, inline, d)

    @staticmethod
    def _simple_target(st, name):
        if isinstance(st, ast.AnnAssign):
            return isinstance(st.target, ast.Name) and st.target.id == name
        if len(st.targets) != 1:
            return False
        t = st.targets[0]
        if isinstance(t, ast.Name):
            return t.id == name
        if isinstance(t, (ast.Tuple, ast.List)):
            return any(isinstance(e, ast.Name) and e.id == name for e in t.elts)
        return False

    def add(self, node: ast.Name, stack, stmt):
        self.occ.setdefault(node.id, []).append(
            (
                (node.lineno, node.col_offset),
                list(stack),
                isinstance(node.ctx, ast.Store),
                stmt,
            )
        )

    def names_in(self, node, stack, stmt, bound=frozenset()):
        """Names in an expression; comprehension targets are private to it."""
        if isinstance(
            node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
        ):
            inner = set(bound)
            for g in node.generators:
                self.names_in(g.iter, stack, stmt, frozenset(inner))
                inner |= {n.id for n in ast.walk(g.target) if isinstance(n, ast.Name)}
                for c in g.ifs:
                    self.names_in(c, stack, stmt, frozenset(inner))
            elts = (
                [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
            )
            for e in elts:
                self.names_in(e, stack, stmt, frozenset(inner))
            return
        if isinstance(node, ast.Name):
            if node.id not in bound:
                self.add(node, stack, stmt)
            return
        for child in ast.iter_child_nodes(node):
            self.names_in(child, stack, stmt, bound)

    def own_exprs(self, st):
        if isinstance(st, (ast.Assign,)):
            return [st.value] + st.targets
        if isinstance(st, ast.AnnAssign):
            return [st.target] + ([st.value] if st.value else [])
        if isinstance(st, ast.AugAssign):
            return [st.target, st.value]
        if isinstance(st, (ast.Expr, ast.Return)):
            return [st.value] if st.value else []
        if isinstance(st, ast.If):
            return [st.test]
        if isinstance(st, ast.While):
            return [st.test]
        if isinstance(st, ast.For):
            return [st.iter]
        if isinstance(st, ast.With):
            return [i.context_expr for i in st.items]
        if isinstance(st, ast.Raise):
            return [x for x in (st.exc, st.cause) if x]
        if isinstance(st, ast.Assert):
            return [x for x in (st.test, st.msg) if x]
        return []

    def walk_block(self, stmts, stack):
        for i, st in enumerate(stmts):
            here = stack + [(id(stmts), i)]
            if isinstance(st, ast.FunctionDef):
                self.add(
                    ast.Name(
                        id=st.name,
                        ctx=ast.Store(),
                        lineno=st.lineno,
                        col_offset=st.col_offset,
                    ),
                    here,
                    st,
                )
                inner_params = {a.arg for a in st.args.args + st.args.kwonlyargs}
                for sub in ast.walk(st):
                    if (
                        isinstance(sub, ast.Name)
                        and isinstance(sub.ctx, ast.Load)
                        and sub.id not in inner_params
                    ):
                        self.add(
                            ast.Name(
                                id=sub.id,
                                ctx=ast.Load(),
                                lineno=st.lineno,
                                col_offset=st.col_offset,
                            ),
                            here,
                            st,
                        )
                for d in st.args.defaults + st.args.kw_defaults:
                    if d is not None:
                        self.names_in(d, here, st)
                continue
            for e in self.own_exprs(st):
                self.names_in(e, here, st)
            if isinstance(st, ast.For):
                # the loop target lives in the body block
                body_stack = here + [(id(st.body), -1)]
                for sub in ast.walk(st.target):
                    if isinstance(sub, ast.Name):
                        self.add(sub, body_stack, st)
                self.walk_block(st.body, here)
                self.walk_block(st.orelse, here)
            elif isinstance(st, (ast.If, ast.While)):
                self.walk_block(st.body, here)
                self.walk_block(st.orelse, here)
            elif isinstance(st, ast.With):
                for it in st.items:
                    if it.optional_vars is not None:
                        for sub in ast.walk(it.optional_vars):
                            if isinstance(sub, ast.Name):
                                self.add(sub, here + [(id(st.body), -1)], st)
                self.walk_block(st.body, here)
            elif isinstance(st, ast.Try):
                self.walk_block(st.body, here)
                for h in st.handlers:
                    self.walk_block(h.body, here)
                self.walk_block(st.orelse, here)
                self.walk_block(st.finalbody, here)


# -------------------------------------------------------------- the lowerer


class Fn:
    def __init__(
        self,
        mod: Module,
        node: ast.FunctionDef,
        sig: Sig,
        hints=None,
        outer: "Fn" = None,
        dry=False,
    ):
        self.mod = mod
        self.env = mod.env
        self.node = node
        self.sig = sig
        self.outer = outer
        self.dry = dry
        self.lines: list[str] = []
        self.ind = 1
        self.vars: dict[str, str] = {}
        self.declared: set[str] = set()
        self.returning = False  # lowering a return value (st_Return)
        self.narrow: list[dict] = []
        self.counter = itertools.count(1) if outer is None else outer.counter
        self.hints = (
            hints if hints is not None else {"vars": {}, "expr": {}, "elem": {}}
        )
        for name, t in sig.params:
            self.vars[name] = t
            self.declared.add(name)
        self.fence = (
            "fence"
            if any(n == "fence" for n, _ in sig.params)
            else (outer.fence if outer is not None else None)
        )
        self.seq_guards = (
            "seq_guards"
            if any(n == "seq_guards" for n, _ in sig.params)
            else (outer.seq_guards if outer is not None else None)
        )
        self.builder = next((n for n, t in sig.params if t == T.B), None)
        if self.builder is None and outer is not None:
            self.builder = outer.builder
        self.host = sig.host or (
            outer.host if outer is not None else None
        )  # host clones: a host clone
        self.host_scratch = (
            None  # host clones: the scratch builder of a host fold in progress
        )
        self.plan = DeclPlan(node, {n for n, _ in sig.params})
        self.block_ids: list[int] = []
        # names Python never reads (an unpacked `_pid`): declared [[maybe_unused]]
        self.loaded = {
            n.id
            for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        self.unit = sig.name[3:] if sig.name.startswith("py_") else sig.name
        self.rec_ret = record_return(node)
        self.rec_vars = record_locals(node)
        self.none_alias = none_aliases(node)

    def _never_none_param(self, node, x: CE) -> bool:
        """N-R4: `node` names a parameter of a plain scalar C++ type (int,
        float, bool) that no statement of the function rebinds. Its value is
        the caller's, converted to that type: it is never None. A local is
        left out (a has_ field read into a local loses its flag)."""
        if not isinstance(node, ast.Name) or x.code != node.id or x.meta:
            return False
        if (
            node.id not in {n for n, _ in self.sig.params}
            or self.vars.get(node.id) != x.t
        ):
            return False
        if not (T.is_int(x.t) or x.t in (T.BOOL, T.F64)):
            return False
        a = self.node.args
        pos = a.posonlyargs + a.args
        defaults = dict(
            zip([p.arg for p in pos][len(pos) - len(a.defaults) :], a.defaults)
        )
        defaults.update(
            {p.arg: d for p, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None}
        )
        d = defaults.get(node.id)
        if isinstance(d, ast.Constant) and d.value is None:
            return False
        return node.id not in self.stored_names

    @functools.cached_property
    def stored_names(self) -> set[str]:
        return {
            n.id
            for n in ast.walk(self.node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
        }

    def unused(self, name: str) -> str:
        return "" if name in self.loaded else "[[maybe_unused]] "

    # ------------------------------------------------------------- emission

    def emit(self, line: str):
        self.lines.append("    " * self.ind + line)

    def tmp(self, base: str) -> str:
        base = (
            "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in base).strip("_")
            or "t"
        )
        return f"{base}_{next(self.counter)}"

    def bind(self, ce: CE, base: str = "t") -> CE:
        if ce.pure:
            return ce
        if ce.t == T.VOID:
            self.emit(ce.code + ";")
            return CE("", T.VOID)
        name = self.tmp(base)
        self.emit(f"{decl_type(ce.t)} {name} = {ce.code};")
        self.last_temp = (len(self.lines) - 1, name, ce.code, ce.t)
        return CE(name, ce.t, True)

    def iter_src(self, code: str) -> str:
        """Code safe to take .begin() and .end() of separately: a braced or
        called temporary (`std::vector<..>{x}`) would make two different
        objects, so bind it to a reference first. A span is a view of one C
        array, so two copies share their pointers."""
        if code.startswith("py::span<") or not any(c in code for c in "({"):
            return code
        name = self.tmp("seq")
        self.emit(f"const auto& {name} = {code};")
        return name

    def take_temp(self, ce: CE):
        """If ce is the temp the last line declared, drop that line and return its expression."""
        lt = getattr(self, "last_temp", None)
        if lt and lt[0] == len(self.lines) - 1 and lt[1] == ce.code and lt[3] == ce.t:
            del self.lines[-1]
            self.last_temp = None
            return CE(lt[2], lt[3], False)
        return None

    def record(self, node, t: str):
        key = (
            getattr(node, "lineno", 0),
            getattr(node, "col_offset", 0),
            type(node).__name__,
        )
        if self.dry and t not in (T.AUTO,):
            self.hints["expr"][key] = t
        return self.hints["expr"].get(key)

    # ------------------------------------------------------------- names

    def lookup(self, name: str):
        for d in reversed(self.narrow):
            if name in d:
                return d[name]
        if name in self.vars:
            return CE(name, self.vars[name])
        if self.outer is not None:
            got = self.outer.lookup(name)
            if got is not None:
                return got
        return None

    def var_type(self, name: str) -> str | None:
        if name in self.vars:
            return self.vars[name]
        h = self.hints["vars"].get(name)
        return h

    def common(self, a: str, b: str):
        """T.common; in a host clone a compare result joins the ints (a
        host builder computes compares and logic on ints, and bool is an int)."""
        c = T.common(a, b)
        if c is None and self.host:
            c = T.common(
                re.sub(r"\bbool\b", HOST_VALUE, a), re.sub(r"\bbool\b", HOST_VALUE, b)
            )
        return c

    def set_var(self, name: str, t: str, node):
        old = self.vars.get(name)
        if old is not None and old != t:
            c = self.common(old, t)
            if c is None:
                raise Refuse(
                    node, f"local {name} holds both {old} and {t} (a union: hole)"
                )
            t = c
        self.vars[name] = t
        if self.dry:
            prev = self.hints["vars"].get(name)
            if prev and prev != t:
                c = self.common(prev, t)
                if c is None:
                    raise Refuse(node, f"local {name} holds both {prev} and {t}")
                t = c
            self.hints["vars"][name] = t

    # ------------------------------------------------------------ functions

    def run(self) -> list[str]:
        self.block(self.node.body)
        return self.lines

    def block(self, stmts, scoped=True):
        saved = set(self.declared) if scoped else None
        pushed = 0
        for i, st in enumerate(stmts):
            self.predeclare(stmts, i)
            self.dead_after = None
            self.stmt(st)
            if self.dead_after is st:
                # a folded `if` whose taken branch returns or raises: the rest
                # of this block is unreachable for every value of the static types
                self.dead_after = None
                break
            # `if isinstance(x, int): ... return` narrows x for the rest of the block
            if (
                isinstance(st, ast.If)
                and not st.orelse
                and st.body
                and isinstance(st.body[-1], (ast.Return, ast.Raise))
            ):
                neg = self._negated_narrowing(st.test)
                if neg:
                    self.narrow.append(neg)
                    pushed += 1
        for _ in range(pushed):
            self.narrow.pop()
        if scoped and stmts is not self.node.body:
            self.declared = saved

    def predeclare(self, stmts, i):
        """Hoisted declarations whose dominating block is this one, before stmt i."""
        for name, (blk, idx, inline, _d) in self.plan.decl.items():
            if blk != id(stmts) or idx != i or inline or name in self.declared:
                continue
            st = stmts[i]
            if isinstance(st, ast.FunctionDef) and st.name == name:
                continue
            if isinstance(st, ast.For) and self._for_target_scoped(st, name):
                continue
            t = self.hints["vars"].get(name) or self.vars.get(name)
            if t is None:
                if self.dry:
                    continue
                raise Refuse(st, f"no type for local {name}")
            self.emit(f"{self.unused(name)}{decl_type(t)} {name}{{}};")
            self.declared.add(name)
            self.vars[name] = t

    def _for_target_scoped(self, st: ast.For, name: str) -> bool:
        tnames = {n.id for n in ast.walk(st.target) if isinstance(n, ast.Name)}
        if name not in tnames:
            return False
        _blk, _idx, _inline, d = self.plan.decl[name]
        # declared at the loop body (or deeper): the header declares it
        occs = self.plan.occ[name]
        return all(any(e[0] == id(st.body) for e in o[1]) for o in occs)

    # ------------------------------------------------------------ statements

    def stmt(self, st):
        m = getattr(self, "st_" + type(st).__name__, None)
        if m is None:
            raise Refuse(st, f"statement {type(st).__name__}")
        m(st)

    def st_Pass(self, st):
        pass

    def st_Expr(self, st):
        if isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
            return  # docstring
        set_ = self._setattr_by_name(st.value)
        if set_ is not None:
            # object.__setattr__(self, "f", v) (a frozen dataclass's
            # __post_init__): the field store self.f = v
            self.st_Assign(
                ast.copy_location(ast.Assign(targets=[set_[0]], value=set_[1]), st)
            )
            return
        ce = self.expr(st.value)
        direct = self.take_temp(ce)
        if direct is not None:
            # a result Python drops: the call as a statement, no unused temp
            self.emit(f"(void){paren(direct.code)};")
            return
        if ce.code and not ce.pure:
            self.emit(ce.code + ";")

    def _setattr_by_name(self, call):
        """``object.__setattr__(self, name, v)`` with ``name`` a str constant or
        a name bound to a literal str: (Attribute target, value node)."""
        if not (
            isinstance(call, ast.Call) and not call.keywords and len(call.args) == 3
        ):
            return None
        f = call.func
        if not (
            isinstance(f, ast.Attribute)
            and f.attr == "__setattr__"
            and isinstance(f.value, ast.Name)
            and f.value.id == "object"
            and self.lookup("object") is None
        ):
            return None
        obj, key, value = call.args
        if not (isinstance(obj, ast.Name) and obj.id == "self"):
            return None
        if isinstance(key, ast.Name):
            lit_ = self.lookup(key.id)
            key = (
                ast.Constant(lit_.lit)
                if lit_ is not None and isinstance(lit_.lit, str)
                else key
            )
        if not (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and key.value.isidentifier()
        ):
            return None
        tgt = ast.copy_location(
            ast.Attribute(
                value=ast.copy_location(ast.Name(id="self", ctx=ast.Load()), obj),
                attr=key.value,
                ctx=ast.Store(),
            ),
            call,
        )
        return tgt, value

    def st_Return(self, st):
        if st.value is None or (
            isinstance(st.value, ast.Constant)
            and st.value.value is None
            and self.sig.ret == T.VOID
        ):
            self.emit("return;")
            return
        ret = self.sig.ret
        if (
            self.dry
            and st.value is not None
            and ret not in (T.AUTO, T.VOID)
            and not self.rec_ret
        ):
            # a return wider than the recorded type (a None never recorded): widen
            try:
                mark = len(self.lines)
                probe = self.expr(st.value)
                del self.lines[mark:]
                w = T.common(self.hints.get("ret_widen") or ret, probe.t)
                if (
                    w is None
                    and self.node.returns is not None
                    and re.search(r"\bAny\b", ast.unparse(self.node.returns))
                ):
                    w = T.union_of(
                        self.hints.get("ret_widen") or ret, probe.t
                    )  # tagged values: an Any of several classes
                if (
                    w is not None
                    and w != ret
                    and (T.optional_inner(w) or w != probe.t or probe.t == T.NONE)
                ):
                    self.hints["ret_widen"] = w
                    if T.variant_items(w) or any(
                        T.variant_items(x) for x in T.tuple_items(w) or []
                    ):
                        ret = w  # tagged values: the dry pass goes on with the union
            except Refuse:
                pass
        if self.rec_ret and (
            isinstance(st.value, ast.Dict)
            or (isinstance(st.value, ast.Name) and st.value.id in self.rec_vars)
        ):
            rid = f"ret_{self.unit}"
            ce = (
                self.record_display(st.value, rid)
                if isinstance(st.value, ast.Dict)
                else self.expr(st.value)
            )
            self.sig.ret = self.record_type(rid)
            self.emit(f"return {ce.code};")
            return
        # a std::string returned as const char* outlives the function (coerce)
        self.returning = True
        try:
            ce = self.expr(st.value, want=ret)
            if self.sig.ret == T.AUTO:
                self.sig.ret = ce.t
                ret = ce.t
            direct = self.take_temp(ce) if ce.t == ret else None
            if direct is not None:
                ce = direct
            self.emit(f"return {self.coerce(ce, ret, st)};")
        finally:
            self.returning = False

    def st_ImportFrom(self, st):
        """A function-local `from .m import name`: the name resolves like a module import."""
        for a in st.names:
            local = a.asname or a.name
            src = PKG.import_source(st.module, st.level)
            if local in self.mod.imports and self.mod.imports[local] != (src, a.name):
                raise Refuse(st, f"local import of {local} shadows another import")
            self.mod.imports[local] = (src, a.name)

    def st_Raise(self, st):
        exc = st.exc
        if isinstance(exc, ast.Name):  # raise NotImplementedError (no call)
            exc = ast.copy_location(ast.Call(func=exc, args=[], keywords=[]), exc)
        if not (isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name)):
            raise Refuse(st, "raise of a non-call")
        cls = {
            "ValueError": "ckc::ValueError",
            "TypeError": "ckc::TypeError",
            "KeyError": "ckc::KeyError",
            "NotImplementedError": "ckc::NotImplError",
            "AssertionError": "ckc::ValueError",
            "RuntimeError": "ckc::ValueError",
            "ZeroDivisionError": "ckc::ValueError",
        }.get(exc.func.id)
        if cls is None:
            raise Refuse(st, f"raise {exc.func.id}")
        if exc.args:
            msg = self.str_expr(exc.args[0])
        else:
            msg = 'std::string("")'
        self.emit(f"throw {cls}({msg});")

    def st_Assert(self, st):
        c = self.test(st.test)
        msg = self.str_expr(st.msg) if st.msg is not None else 'std::string("")'
        self.emit(f"if(!({c})) throw ckc::ValueError({msg});")

    def static_test(self, test):
        """True / False when the test folds statically (a static isinstance,
        literal operands); None otherwise. Nothing is emitted."""
        mark = len(self.lines)
        saved_counter = self.counter
        self.counter = itertools.count(10**6)  # a probe: temp numbering must not move
        try:
            ce = self.expr(test)
        except Refuse:
            ce = None
        finally:
            self.counter = saved_counter
        emitted = len(self.lines) != mark
        del self.lines[mark:]
        if ce is None or emitted:
            return None
        if isinstance(ce.lit, bool):
            return ce.lit
        return None

    def st_If(self, st):
        folded = self.static_test(st.test)
        if folded is not None:
            # a branch the static types rule out is never taken (Python agrees)
            taken = st.body if folded else st.orelse
            self.block(taken, scoped=True) if taken else None
            self.dead_after = (
                st if taken and isinstance(taken[-1], (ast.Return, ast.Raise)) else None
            )
            return
        c = self.test(st.test)
        narrowed = self._narrowing_from(st.test)
        self.emit(f"if({c})")
        self.emit("{")
        self.ind += 1
        self.narrow.append(narrowed)
        self.block(st.body)
        self.narrow.pop()
        self.ind -= 1
        self.emit("}")
        if st.orelse:
            self.narrow.append(self._negated_narrowing(st.test))
            try:
                self._st_else(st)
            finally:
                self.narrow.pop()

    def _st_else(self, st):
        if st.orelse:
            if (
                len(st.orelse) == 1
                and isinstance(st.orelse[0], ast.If)
                and self._if_is_flat(st.orelse[0])
            ):
                mark = len(self.lines)
                self.st_If(st.orelse[0])
                first = self.lines[mark].strip()
                self.lines[mark] = "    " * self.ind + "else " + first
                return
            self.emit("else")
            self.emit("{")
            self.ind += 1
            self.block(st.orelse)
            self.ind -= 1
            self.emit("}")

    def _if_is_flat(self, st) -> bool:
        """An elif whose test emits no IR can be a C++ `else if`."""
        mark = len(self.lines)
        try:
            self.test(st.test)
            flat = len(self.lines) == mark
        except Refuse:
            flat = False
        del self.lines[mark:]
        return flat

    def st_While(self, st):
        if st.orelse:
            raise Refuse(st, "while ... else")
        self.emit("while(true)")
        self.emit("{")
        self.ind += 1
        c = self.test(st.test)
        self.emit(f"if(!({c})) break;")
        self.block(st.body)
        self.ind -= 1
        self.emit("}")

    def st_Break(self, st):
        self.emit("break;")

    def st_Continue(self, st):
        self.emit("continue;")

    def st_AnnAssign(self, st):
        if isinstance(st.target, ast.Attribute) and st.value is not None:
            self.assign_attr(st.target, st.value, st)
            return
        if (
            isinstance(st.target, ast.Name)
            and st.target.id in self.rec_vars
            and isinstance(st.value, ast.Dict)
        ):
            rid = (
                f"ret_{self.unit}"
                if self.rec_ret
                else f"var_{self.unit}_{st.target.id}"
            )
            self.assign_name(st.target.id, self.record_display(st.value, rid), st)
            return
        if not isinstance(st.target, ast.Name):
            raise Refuse(st, "annotated non-name target")
        ann_ = ast.unparse(st.annotation)
        t = T.from_annotation(ann_) if not re.search(r"\bAny\b", ann_) else None
        if (
            t is None
            and st.value is not None
            and re.search(r"\bAny\b|^(list|tuple|dict)$", ann_)
        ):
            # the annotation does not say enough: the value's type decides
            self.assign_name(st.target.id, self.expr(st.value), st)
            return
        if t is None:
            raise Refuse(st, f"annotation {ann_}")
        if st.value is None:
            return
        name = st.target.id
        self.set_var(name, t, st)
        ce = self.expr(st.value, want=t)
        self.assign_name(name, ce, st)

    def st_Assign(self, st):
        if len(st.targets) != 1:
            # a = b = x: x once, then each target left to right
            if not all(isinstance(t_, ast.Name) for t_ in st.targets):
                raise Refuse(st, "chained assignment to a non-name")
            v = self.bind(self.expr(st.value), "v")
            for t_ in st.targets:
                self.assign_name(t_.id, CE(v.code, v.t, True), st)
            return
        tgt = st.targets[0]
        if isinstance(tgt, ast.Name):
            if (
                isinstance(st.value, ast.Call)
                and isinstance(st.value.func, ast.Name)
                and st.value.func.id == "IRBuilder"
                and self.lookup(tgt.id) is not None
                and self.lookup(tgt.id).t == T.B
            ):
                self.init_builder(tgt.id, st.value, st)
                return
            if tgt.id in self.rec_vars and isinstance(st.value, ast.Dict):
                rid = (
                    f"ret_{self.unit}" if self.rec_ret else f"var_{self.unit}_{tgt.id}"
                )
                self.assign_name(tgt.id, self.record_display(st.value, rid), st)
                return
            want = self.var_type(tgt.id)
            ce = self.expr(st.value, want=want)
            self.assign_name(tgt.id, ce, st)
            return
        if isinstance(tgt, (ast.Tuple, ast.List)):
            # T1: evaluate the right-hand side completely, then assign in order
            names = {e.id for e in tgt.elts if isinstance(e, ast.Name)}
            if isinstance(st.value, (ast.Tuple, ast.List)) and len(
                st.value.elts
            ) == len(tgt.elts):
                vals = [self.bind(self.expr(v), "v") for v in st.value.elts]
                vals = [self.snapshot(v) if v.code in names else v for v in vals]
            else:
                src = self.expr(st.value)
                if (
                    src.meta
                    and "items" in src.meta
                    and len(src.meta["items"]) == len(tgt.elts)
                ):
                    el = T.seq_elem(src.t)
                    for t_, item in zip(tgt.elts, src.meta["items"]):
                        self.assign_target(t_, CE(item, el), st)
                    return
                if T.optional_inner(src.t):
                    src = CE(
                        f"py::deref({src.code})", T.optional_inner(src.t), src.pure
                    )
                if not (src.pure and src.code.isidentifier() and src.code not in names):
                    tmp = self.tmp("unpack")
                    self.emit(f"const auto {tmp} = {src.code};")
                    src = CE(tmp, src.t)
                items = T.tuple_items(src.t)
                ts = self.tuple_struct_items(src)
                if ts is not None:
                    if len(ts) != len(tgt.elts):
                        raise Refuse(st, "tuple unpack arity")
                    for t_, (code, it_) in zip(tgt.elts, ts):
                        self.assign_target(t_, CE(code, it_), st)
                    return
                if items is not None:
                    if len(items) != len(tgt.elts):
                        raise Refuse(st, "tuple unpack arity")
                    for i, (t_, it_) in enumerate(zip(tgt.elts, items)):
                        self.assign_target(
                            t_, CE(f"std::get<{i}>({src.code})", it_), st
                        )
                    return
                el = T.seq_elem(src.t)
                if el is None:
                    raise Refuse(st, f"unpack of {src.t}")
                n = T.arr_len(src.t)
                if n is not None and n != len(tgt.elts):
                    raise Refuse(st, "unpack length mismatch")
                if n is None:
                    self.emit(
                        f'if(py::len({src.code}) != {len(tgt.elts)}) throw ckc::ValueError("wrong number of values to unpack");'
                    )
                vals = [CE(f"{src.code}[{i}]", el) for i in range(len(tgt.elts))]
            for t_, v in zip(tgt.elts, vals):
                self.assign_target(t_, v, st)
            return
        if isinstance(tgt, ast.Subscript):
            # Python evaluates the value first, then the target's parts
            ce = self.bind(self.expr(st.value), "v")
            if (
                isinstance(tgt.value, ast.Attribute)
                and tgt.value.attr == "attrs"
                and isinstance(tgt.value.value, ast.Attribute)
                and tgt.value.value.attr == "kernel"
            ):
                key = self.expr(tgt.slice)
                # KA-R1: the setter by the value's type (Python keeps the type in
                # the attrs dict: True stays a bool, 1.0 a float, "x" a str)
                if T.is_int(ce.t):
                    setter, val = "int", ce.code
                elif ce.t == T.BOOL:
                    setter, val = "bool", ce.code
                elif ce.t in (T.F64, "float"):
                    setter, val = "float", ce.code
                elif ce.t == T.STR:
                    setter, val = "str", ce.code
                elif ce.t == T.SSTR:
                    setter, val = "str", f"{paren(ce.code)}.c_str()"
                else:
                    raise Refuse(st, f"kernel attribute of type {ce.t}")
                self.emit(
                    f"rocke_attr_set_{setter}({self.builder}, &rocke_ir_builder_kernel({self.builder})->attrs, {self.coerce(key, T.STR, st)}, {val});"
                )
                return
            obj = self.expr(tgt.value)
            if obj.t == "dict?" and isinstance(tgt.value, ast.Name):
                # the first item assignment types the empty dict
                k_ = self.bind(self.expr(tgt.slice), "k")
                kt = T.SSTR if T.is_str(k_.t) else k_.t
                vt = T.SSTR if T.is_str(ce.t) else ce.t
                dt = f"py::dict<{kt}, {vt}>"
                if self.dry:
                    self.hints["vars"][tgt.value.id] = dt
                    self.vars[tgt.value.id] = dt
                    return
                raise Refuse(st, "empty dict typed too late")
            if self.record_of(obj) is not None:
                if not (
                    isinstance(tgt.slice, ast.Constant)
                    and isinstance(tgt.slice.value, str)
                ):
                    raise Refuse(st, "record key not a literal (R-REC)")
                self.record_set(obj, self.record_of(obj), tgt.slice.value, ce, st)
                return
            idx = self.expr(tgt.slice)
            if obj.t == "py::kargs":
                self.emit(
                    f"{obj.code}.set({self.coerce(idx, T.SSTR, st)}, {self.coerce(ce, T.V, st)});"
                )
                return
            dm_ = re.fullmatch(r"py::dict<(.+)>", obj.t)
            if dm_:
                kt_, vt_ = [
                    x.strip().replace("[", "<").replace("]", ">")
                    for x in T._split_top(
                        dm_.group(1).replace("<", "[").replace(">", "]")
                    )
                ]
                self.emit(
                    f"{obj.code}.set({self.coerce(idx, kt_, st)}, {self.coerce(ce, vt_, st)});"
                )
                return
            el = T.seq_elem(obj.t)
            if el is None:
                raise Refuse(st, f"item assignment on {obj.t}")
            if el == T.NONE and (T.is_ptr(ce.t) or ce.t == T.STR):
                # a list of None filled later: the base name's type widens (dry pass)
                base = tgt.value
                depth = 1
                while isinstance(base, ast.Subscript):
                    base = base.value
                    depth += 1
                if isinstance(base, ast.Name) and self.dry:
                    cur = self.vars.get(base.id) or ""
                    widened = cur.replace(
                        "std::vector<None>", f"std::vector<{ce.t}>"
                    ).replace("py::tuple<None>", f"py::tuple<{ce.t}>")
                    self.hints["vars"][base.id] = widened
                    self.vars[base.id] = widened
                    return
                raise Refuse(st, "item assignment into a list of None")
            self.emit(f"py::at({obj.code}, {idx.code}) = {self.coerce(ce, el, st)};")
            return
        if isinstance(tgt, ast.Attribute):
            self.assign_attr(tgt, st.value, st)
            return
        raise Refuse(st, f"assignment to {type(tgt).__name__}")

    def init_builder(self, name, call, st):
        """IRBuilder(name): the C caller owns the builder; initialise it here."""
        if len(call.args) != 1 or call.keywords:
            raise Refuse(st, "IRBuilder() shape")
        kname = self.bind(self.expr(call.args[0]), "kname")
        self.emit(f"py_adapt::init_builder({name}, {self.coerce(kname, T.STR, st)});")
        # builder typestate Python keeps on the IRBuilder
        if self.fence is None:
            self.emit("[[maybe_unused]] const char* fence = nullptr;")
            self.fence = "fence"
            self.vars["fence"] = T.STR
            self.declared.add("fence")
        if self.seq_guards is None:
            self.emit("int seq_guards_count = 0;")
            self.emit("[[maybe_unused]] int* seq_guards = &seq_guards_count;")
            self.seq_guards = "seq_guards"
            self.vars["seq_guards"] = "int*"
            self.declared.add("seq_guards")

    def assign_attr(self, tgt, value, st):
        if isinstance(tgt.value, ast.Name) and tgt.attr == "fence_barriers":
            recv = self.lookup(tgt.value.id)
            if recv is not None and recv.t == T.B:
                ce = self.expr(value)
                if self.fence is None:
                    raise Refuse(
                        st,
                        "fence_barriers set on a builder this function did not create",
                    )
                self.emit(f"{self.fence} = {self.coerce(ce, T.STR, st)};")
                return
        ce = (
            self.bind(self.expr(value, want=None), "v")
            if not (isinstance(value, ast.List) and not value.elts)
            else None
        )
        obj = self.expr(tgt.value)
        fld = self.field(obj, tgt.attr, tgt)
        if fld.t.startswith(("py::olist<", "py::span<")):
            if ce is not None:
                self.store_field(fld, ce, st)
                return
            self.emit(f"{fld.meta['n']} = 0;")
            return
        if ce is None:
            raise Refuse(st, "empty list into a field")
        if fld.meta and "has" in fld.meta:
            self.store_optional(fld, ce, st)
            return
        self.emit(f"{fld.code} = {self.coerce(ce, fld.t, st)};")

    def assign_target(self, t_, ce: CE, st):
        """One target of an unpacking assignment: a name, an attribute, or a
        nested tuple of targets."""
        if isinstance(t_, ast.Name):
            self.assign_name(t_.id, ce, st)
            return
        if isinstance(t_, ast.Attribute):
            obj = self.expr(t_.value)
            fld = self.field(obj, t_.attr, t_)
            self.store_field(fld, ce, st)
            return
        if isinstance(t_, (ast.Tuple, ast.List)):
            src = (
                ce
                if (ce.pure and ce.code.isidentifier())
                else self.bind(CE(ce.code, ce.t, False), "unpack")
            )
            parts = self.tuple_struct_items(src)
            if parts is None and T.tuple_items(src.t):
                parts = [
                    (f"std::get<{i}>({src.code})", t)
                    for i, t in enumerate(T.tuple_items(src.t))
                ]
            if parts is None and T.arr_len(src.t) is not None:
                parts = [
                    (f"{src.code}[{i}]", T.seq_elem(src.t))
                    for i in range(T.arr_len(src.t))
                ]
            if parts is None or len(parts) != len(t_.elts):
                raise Refuse(st, f"nested unpack of {src.t}")
            for sub, (code, t) in zip(t_.elts, parts):
                self.assign_target(sub, CE(code, t), st)
            return
        raise Refuse(st, f"unpack into {type(t_).__name__}")

    def store_field(self, fld: CE, ce: CE, st):
        """fld = ce for a struct field; an owned list field gets a fresh copy."""
        if (
            fld.t.startswith("py::span<")
            and fld.meta
            and "p" in fld.meta
            and "cap" not in fld.meta
        ):
            # a fixed C array field: copy the items (Python rebinds a tuple of that length)
            src = self.iter_src(self.readable(ce).code)
            self.emit(
                f"if((int64_t)py::len({src}) != {fld.meta['n']}) throw ckc::ValueError(\"wrong length for a fixed field\");"
            )
            self.emit(f"std::copy({src}.begin(), {src}.end(), {fld.meta['p']});")
            return
        if fld.t.startswith("py::olist<"):
            m = fld.meta
            el = T.seq_elem(fld.t)
            if not T.is_seq(ce.t):
                raise Refuse(st, f"{ce.t} into an owned list field")
            src = self.iter_src(ce.code)
            es = self.env.headers.structs.get(el)
            # no free of the old list: a struct copy may share it (Python keeps
            # an old list alive while anything refers to it)
            if es is not None and "_tuple_" in el:
                conv = f"[](const auto& x) {{ return {el}{{{', '.join(f'({es.fields[f]})std::get<{i}>(x)' for i, f in enumerate(es.order))}}}; }}"
                self.emit(f"{m['p']} = py::own_copy_map<{el}>({src}, {conv});")
            else:
                self.emit(f"{m['p']} = py::own_copy<{el}>({src});")
            self.emit(f"{m['n']} = (int)py::len({src});")
            self.emit(f"{m['cap']} = (int)py::len({src});")
            return
        if fld.meta and "has" in fld.meta:
            self.store_optional(fld, ce, st)
            return
        self.emit(f"{fld.code} = {self.coerce(ce, fld.t, st)};")

    def optional_parts(self, ce: CE, ft: str, node) -> tuple[str, str | None]:
        """N-R2: a value for an optional field (a has_ flag next to it) as
        (flag, value): None clears the flag and leaves the value, a plain value
        sets it, an optional value (std::optional, or another flagged field)
        passes its presence on."""
        if ce.t == T.NONE:
            return "false", None
        if ce.meta and "has" in ce.meta:
            return ce.meta["has"], self.coerce(CE(ce.code, ce.t, ce.pure), ft, node)
        if T.optional_inner(ce.t):
            src = (
                ce
                if (ce.pure and ce.code.isidentifier())
                else self.bind(CE(ce.code, ce.t, False), "opt")
            )
            inner = self.coerce(CE(f"(*{src.code})", T.optional_inner(ce.t)), ft, node)
            return (
                f"{src.code}.has_value()",
                f"({src.code}.has_value() ? {inner} : {decl_type(ft)}{{}})",
            )
        return "true", self.coerce(ce, ft, node)

    def store_optional(self, fld: CE, ce: CE, st):
        """fld = ce for a field with a has_ flag (N-R2)."""
        has, val = self.optional_parts(ce, fld.t, st)
        if val is not None:
            self.emit(f"{fld.code} = {val};")
        self.emit(f"{fld.meta['has']} = {has};")

    def snapshot(self, ce: CE) -> CE:
        """A value that later assignments in the same statement cannot change."""
        if (
            ce.pure
            and ce.lit is None
            and ce.code.isidentifier()
            and ce.t not in (T.VOID,)
        ):
            name = self.tmp(ce.code)
            self.emit(f"{decl_type(ce.t)} {name} = {ce.code};")
            return CE(name, ce.t)
        return ce

    def st_AugAssign(self, st):
        if isinstance(st.target, ast.Attribute):
            # Python evaluates the target object first, then the value
            obj = self.expr(st.target.value)
            fld = self.field(obj, st.target.attr, st.target)
            rhs = self.expr(st.value)
            val = self.binop(st.op, fld, rhs, st)
            self.emit(f"{fld.code} = {self.coerce(val, fld.t, st)};")
            return
        if isinstance(st.target, ast.Subscript):
            # x[k] op= v: the item once, then the store
            item = self.bind(self.expr(st.target), "item")
            rhs = self.expr(st.value)
            val = self.binop(st.op, item, rhs, st)
            assign = ast.copy_location(
                ast.Assign(
                    targets=[st.target], value=ast.Name(id="__aug__", ctx=ast.Load())
                ),
                st,
            )
            self.narrow.append({"__aug__": self.bind(val, "aug")})
            try:
                self.st_Assign(assign)
            finally:
                self.narrow.pop()
            return
        if not isinstance(st.target, ast.Name):
            raise Refuse(st, "augmented assignment to a non-name")
        cur = self.lookup(st.target.id)
        if cur is None:
            raise Refuse(st, f"unknown {st.target.id}")
        rhs = self.expr(st.value)
        if (
            isinstance(st.op, ast.Add)
            and (cur.t == "list?" or T.seq_elem(cur.t) is not None)
            and T.seq_elem(rhs.t) is not None
        ):
            rhs = self.bind(rhs, "ext")
            if cur.t == "list?" or T.seq_elem(cur.t) == T.AUTO:
                el = T.seq_elem(rhs.t)
                if self.dry:
                    self.hints["elem"][st.target.id] = el
                    self.vars[st.target.id] = T.vec(el)
                    self.hints["vars"][st.target.id] = T.vec(el)
            src_rhs = self.iter_src(rhs.code)
            self.emit(
                f"{st.target.id}.insert({st.target.id}.end(), {src_rhs}.begin(), {src_rhs}.end());"
            )
            return
        val = self.binop(st.op, cur, rhs, st)
        self.emit(f"{st.target.id} = {val.code};")

    def assign_name(self, name: str, ce: CE, st):
        t = ce.t
        if t == T.NONE:
            t = self.var_type(name) or T.NONE
            if t == T.NONE and not self.dry:
                raise Refuse(st, f"{name} = None with no other type")
        if t == "list?":
            t = self.var_type(name) or self._elem_hint_list(name, st)
        h_ = self.hints["vars"].get(name)
        if not self.dry and h_ and h_ != t and "None" in t and T.common(h_, t) == h_:
            t = h_  # the dry pass widened a list of None by its later stores
        if t == "set?":
            t = self.var_type(name) or self.hints["vars"].get(name)
            if t is None or t == "set?":
                if not self.dry:
                    raise Refuse(st, f"empty set {name} with no element type")
                t = "set?"
        if t == "dict?":
            # {} whose key / value types the item assignments decide (dry pass)
            t = self.var_type(name) or self.hints["vars"].get(name)
            if t is None or t == "dict?":
                if not self.dry:
                    raise Refuse(st, f"empty dict {name} with no key / value type")
                t = "dict?"
        if T.is_seq(t) and T.seq_elem(t) == T.NONE:
            raise Refuse(st, f"list of None only: {name}")
        self.set_var(name, t, st)
        direct = self.take_temp(ce) if self.vars[name] == ce.t else None
        if direct is not None:
            ce = direct
        if (
            ce.code.startswith(decl_type(self.vars[name]) + "{}")
            and ce.code == decl_type(self.vars[name]) + "{}"
            and name not in self.declared
            and self.plan.decl.get(name, (0, 0, False))[2]
        ):
            self.emit(f"{self.unused(name)}{decl_type(self.vars[name])} {name};")
            self.declared.add(name)
            return
        code = self.coerce(ce, self.vars[name], st)
        if name in self.declared:
            self.emit(f"{name} = {code};")
            return
        plan = self.plan.decl.get(name)
        if plan is not None and plan[2]:
            self.emit(
                f"{self.unused(name)}{decl_type(self.vars[name])} {name} = {code};"
            )
            self.declared.add(name)
            return
        if self.dry:
            self.declared.add(name)
            return
        raise Refuse(st, f"assignment to {name} before its declaration")

    def _elem_hint_list(self, name, st):
        el = self.hints["elem"].get(name)
        if el is None:
            if self.dry:
                return "list?"
            raise Refuse(st, f"empty list {name} with no element type")
        return T.vec(el)

    # ------------------------------------------------------------- loops

    def st_For(self, st):
        if st.orelse:
            raise Refuse(st, "for ... else")
        it = st.iter
        # literal tuple of items: unroll, each item in its own block
        if isinstance(it, (ast.Tuple, ast.List)) and not any(
            isinstance(e, ast.Starred) for e in it.elts
        ):
            for e in it.elts:
                saved = set(self.declared)
                self.emit("{")
                self.ind += 1
                # a target bound to a str constant that names an attribute
                # (getattr / object.__setattr__ by that name) stays a literal in
                # this copy of the body: the field itself (attribute by name)
                lits = self._literal_str_targets(st.target, e, st.body)
                self.bind_target(st.target, e, st, lits)
                self.narrow.append(lits)
                try:
                    self.loop_body(st.body)
                finally:
                    self.narrow.pop()
                self.ind -= 1
                self.emit("}")
                self.declared = saved
            return
        if (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == "range"
            and self.lookup("range") is None
        ):
            args = [
                self.coerce(self.bind(self.expr(a), "r"), T.I64, st) for a in it.args
            ]
            if not isinstance(st.target, ast.Name):
                raise Refuse(st, "range loop with a tuple target")
            self.for_header(st, st.target.id, T.I64, f"py::range({', '.join(args)})")
            return
        if (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id in ("enumerate", "zip")
            and self.lookup(it.func.id) is None
        ):
            self.for_indexed(
                st.target, it, lambda: self.loop_body(st.body), st, scoped_body=True
            )
            return
        if (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id in self.mod.imports
        ):
            src_m, attr = self.mod.imports[it.func.id]
            from .adapters import generator_split

            gen = generator_split(f"{src_m.split('.')[-1]}:{attr}")
            if gen is not None:
                self.for_generator(st, it, gen)
                return
        it_ce = self.expr(it)
        if T.optional_inner(it_ce.t):
            it_ce = CE(
                f"py::deref({it_ce.code})", T.optional_inner(it_ce.t), it_ce.pure
            )
        if (
            isinstance(st.iter, (ast.Tuple, ast.List))
            and st.iter.elts
            and not any(isinstance(e, ast.Starred) for e in st.iter.elts)
        ):
            # a loop over a literal display: the items first (Python builds the
            # tuple before iterating), then one body per item with its own types
            items = [self.bind(self.expr(e), "item") for e in st.iter.elts]
            items = [
                self.snapshot(v) if v.code.isidentifier() and not v.lit else v
                for v in items
            ]
            for v in items:
                self.emit("{")
                self.ind += 1
                saved = set(self.declared)
                self.bind_loop_target(st.target, v, st)
                self.loop_body(st.body)
                self.declared = saved
                self.ind -= 1
                self.emit("}")
            return
        vals = self._unroll_names(st, st)
        if vals is not None:
            for v in vals:
                self.emit("{")
                self.ind += 1
                self.narrow.append({st.target.id: CE(lit_code(v), T.STR, lit=v)})
                try:
                    self.loop_body(st.body)
                finally:
                    self.narrow.pop()
                self.ind -= 1
                self.emit("}")
            return
        src = self.bind(self.readable(it_ce), "it")
        el = T.seq_elem(src.t)
        if el is None:
            raise Refuse(st, f"iteration over {src.t}")
        if isinstance(st.target, ast.Name):
            stored = any(
                isinstance(n, ast.Name)
                and n.id == st.target.id
                and isinstance(n.ctx, ast.Store)
                for b_ in st.body
                for n in ast.walk(b_)
            )
            self.for_header(st, st.target.id, el, src.code, ref=not stored)
            return
        if isinstance(st.target, ast.Tuple):
            item = self.tmp("item")
            self.emit(f"for(const auto& {item} : {src.code})")
            self.emit("{")
            self.ind += 1
            n = T.arr_len(el)
            sub = T.seq_elem(el)
            ts = self.tuple_struct_items(CE(item, el)) or (
                [(f"std::get<{i}>({item})", x) for i, x in enumerate(T.tuple_items(el))]
                if T.tuple_items(el)
                else None
            )
            if (
                ts is not None
                and len(ts) == len(st.target.elts)
                and all(isinstance(x, ast.Name) for x in st.target.elts)
            ):
                saved = set(self.declared)
                for t_, (code, it_) in zip(st.target.elts, ts):
                    if (
                        self._for_target_scoped(st, t_.id)
                        and t_.id not in self.declared
                    ):
                        self.set_var(t_.id, it_, st)
                        self.emit(
                            f"{self.unused(t_.id)}{decl_type(it_)} {t_.id} = {code};"
                        )
                        self.declared.add(t_.id)
                    else:
                        self.assign_name(t_.id, CE(code, it_), st)
                self.loop_body(st.body)
                self.declared = saved
                self.ind -= 1
                self.emit("}")
                return
            if sub is None or n != len(st.target.elts):
                raise Refuse(st, f"tuple target over {el}")
            saved = set(self.declared)
            if not all(isinstance(t_, ast.Name) for t_ in st.target.elts):
                # nested targets: bind each level as for a comprehension
                saved = set(self.declared)
                self.bind_loop_target(st.target, CE(item, el), st)
                self.loop_body(st.body)
                self.declared = saved
                self.ind -= 1
                self.emit("}")
                return
            for i, t_ in enumerate(st.target.elts):
                if not isinstance(t_, ast.Name):
                    raise Refuse(st, "nested tuple target")
                if self._for_target_scoped(st, t_.id) and t_.id not in self.declared:
                    self.set_var(t_.id, sub, st)
                    self.emit(
                        f"{self.unused(t_.id)}{decl_type(sub)} {t_.id} = {item}[{i}];"
                    )
                    self.declared.add(t_.id)
                else:
                    self.assign_name(t_.id, CE(f"{item}[{i}]", sub), st)
            self.loop_body(st.body)
            self.declared = saved
            self.ind -= 1
            self.emit("}")
            return
        raise Refuse(st, "for target")

    def for_generator(self, st, call, gen):
        """C4: an emitting generator split into a round count and one function per round."""
        params = [{"name": n, "kind": "pos", "default": None} for n in gen["params"]]
        bound = self.bind_call_args(call, params)
        fmt = {k: (v.code if k != "b" else self.builder) for k, v in bound.items()}
        fmt["b"] = self.builder
        cnt = self.tmp("rounds")
        self.emit(f"const int {cnt} = {gen['count'].format(**fmt)};")
        i = self.tmp("round")
        tnames = (
            [e.id for e in st.target.elts]
            if isinstance(st.target, ast.Tuple)
            else [st.target.id]
        )
        if len(tnames) != len(gen["yields"]):
            raise Refuse(st, "generator target arity")
        self.emit(f"for(int {i} = 0; {i} < {cnt}; ++{i})")
        self.emit("{")
        self.ind += 1
        saved = set(self.declared)
        outs = {}
        for (t, role), n in zip(gen["yields"], tnames):
            self.set_var(n, t, st)
            if role != "ret":
                pre = "" if n in self.declared else f"{decl_type(t)} "
                if pre:
                    self.emit(f"{decl_type(t)} {n}{{}};")
                    self.declared.add(n)
                outs[role] = "&" + n
        ret_name = next(
            n for (t, role), n in zip(gen["yields"], tnames) if role == "ret"
        )
        fmt.update(outs)
        fmt["i"] = i
        call_code = gen["round"].format(**fmt)
        ret_t = next(t for t, role in gen["yields"] if role == "ret")
        if ret_name in self.declared:
            self.emit(f"{ret_name} = {call_code};")
        else:
            self.emit(f"{decl_type(ret_t)} {ret_name} = {call_code};")
            self.declared.add(ret_name)
        self.loop_body(st.body)
        self.declared = saved
        self.ind -= 1
        self.emit("}")

    def for_header(self, st, name, t, rng, ref=False):
        scoped = self._for_target_scoped(st, name)
        self.set_var(name, t, st)
        if scoped:
            dt = (
                f"const {decl_type(t)}&"
                if ref and not (T.is_int(t) or T.is_ptr(t) or t == T.BOOL)
                else decl_type(t)
            )
            self.emit(f"for({dt} {name} : {rng})")
            self.emit("{")
            self.ind += 1
            was = name in self.declared
            self.declared.add(name)
            self.loop_body(st.body)
            if not was:
                self.declared.discard(name)
        else:
            it = self.tmp(name)
            self.emit(f"for({decl_type(t)} {it} : {rng})")
            self.emit("{")
            self.ind += 1
            self.emit(f"{name} = {it};")
            self.loop_body(st.body)
        self.ind -= 1
        self.emit("}")

    def loop_body(self, body):
        self.block(body)

    @staticmethod
    def _literal_str_targets(target, value_node, body) -> dict:
        """Names of a loop target bound to a str constant of one literal item
        (``for name, v in (("a", 1), ...)``) and used as an attribute name in
        the body (``getattr(x, name)``, ``object.__setattr__(x, name, v)``):
        name -> the literal."""
        named = {
            c.args[1].id
            for b_ in body
            for c in ast.walk(b_)
            if isinstance(c, ast.Call)
            and len(c.args) >= 2
            and isinstance(c.args[1], ast.Name)
            and (
                (isinstance(c.func, ast.Name) and c.func.id == "getattr")
                or (isinstance(c.func, ast.Attribute) and c.func.attr == "__setattr__")
            )
        }
        if not named:
            return {}
        pairs = []
        if isinstance(target, ast.Name):
            pairs = [(target, value_node)]
        elif (
            isinstance(target, ast.Tuple)
            and isinstance(value_node, (ast.Tuple, ast.List))
            and len(target.elts) == len(value_node.elts)
        ):
            pairs = list(zip(target.elts, value_node.elts))
        return {
            t.id: CE(lit_code(v.value), T.STR, lit=v.value)
            for t, v in pairs
            if isinstance(t, ast.Name)
            and t.id in named
            and isinstance(v, ast.Constant)
            and isinstance(v.value, str)
        }

    def bind_target(self, target, value_node, st, lits=()):
        # names in lits are read as their literal (attribute by name): the
        # C++ local may go unused
        unused = lambda n: "[[maybe_unused]] " if n in lits else self.unused(n)
        if isinstance(target, ast.Name):
            ce = self.bind(self.expr(value_node), "v")
            self.set_var(target.id, ce.t, st)
            if target.id in self.declared:
                self.emit(f"{target.id} = {ce.code};")
            else:
                self.emit(
                    f"{unused(target.id)}{decl_type(ce.t)} {target.id} = {ce.code};"
                )
            return
        if (
            isinstance(target, ast.Tuple)
            and isinstance(value_node, (ast.Tuple, ast.List))
            and len(target.elts) == len(value_node.elts)
        ):
            vals = [self.bind(self.expr(v), "v") for v in value_node.elts]
            for t_, v in zip(target.elts, vals):
                if not isinstance(t_, ast.Name):
                    raise Refuse(st, "nested target")
                self.set_var(t_.id, v.t, st)
                if t_.id in self.declared:
                    self.emit(f"{t_.id} = {v.code};")
                else:
                    self.emit(f"{unused(t_.id)}{decl_type(v.t)} {t_.id} = {v.code};")
            return
        raise Refuse(st, "loop target over a literal tuple")

    # -------------------------------------------------------------- with

    def st_With(self, st):
        """CE-R4: builder regions; `with A, B:` nests, the second header inside the first region."""
        self.with_items(st.items, st)

    def with_items(self, items, st):
        if not items:
            saved = set(self.declared)
            self.emit("{")
            self.ind += 1
            self.block(st.body)
            self.ind -= 1
            self.emit("}")
            self.declared = saved
            return
        item, rest = items[0], items[1:]
        cm, var = item.context_expr, item.optional_vars
        if (
            isinstance(cm, ast.Call)
            and isinstance(cm.func, ast.Attribute)
            and isinstance(cm.func.value, ast.Name)
        ):
            recv = self.lookup(cm.func.value.id)
            if (
                recv is not None
                and recv.t == T.B
                and recv.code
                and recv.code != self.builder
            ):
                # a builder alias (b = self.b): it is the builder of this region
                saved_b, self.builder = self.builder, recv.code
                try:
                    return self.with_items(items, st)
                finally:
                    self.builder = saved_b
        region = None  # C++ expression of the region to enter, or None
        if (
            isinstance(cm, ast.Call)
            and isinstance(cm.func, ast.Attribute)
            and isinstance(cm.func.value, ast.Name)
            and (self.lookup(cm.func.value.id) or CE("", "")).t == T.B
        ):
            m = cm.func.attr
            if m == "scf_if" and len(cm.args) == 1 and not cm.keywords and var is None:
                cond = self.bind(self.expr(cm.args[0]), "cond")
                h = self.tmp("if")
                self.emit(
                    f"rocke_if_t {h} = rocke_b_scf_if({self.builder}, {cond.code});"
                )
                region = f"{h}.then_region"
            elif (
                m == "scf_if_else"
                and len(cm.args) == 1
                and isinstance(var, ast.Tuple)
                and len(var.elts) == 2
            ):
                cond = self.bind(self.expr(cm.args[0]), "cond")
                h = self.tmp("ifelse")
                self.emit(
                    f"rocke_if_else_t {h} = rocke_b_scf_if_else({self.builder}, {cond.code});"
                )
                for n_, part in zip(var.elts, ("then_region", "else_region")):
                    self.vars[n_.id] = "py::region"
                    self.region_of = getattr(self, "region_of", {})
                    self.region_of[n_.id] = f"{h}.{part}"
                    self.declared.add(n_.id)
                self.with_items(rest, st)
                return
            elif m in ("scf_for", "scf_for_iter"):
                f = self.bind(self.call_ir(m, None, cm), "loop")
                self.enter_for(f, var, rest, st)
                return
            else:
                raise Refuse(st, f"with {ast.unparse(cm)} (manager not in the table)")
        elif (
            isinstance(cm, ast.Name)
            and self.vars.get(cm.id) == "py::region"
            and var is None
        ):
            region = self.region_of[cm.id]
        elif (
            isinstance(cm, ast.Name)
            and (self.lookup(cm.id) or CE("", "")).t == "rocke_for_t"
        ):
            self.enter_for(self.lookup(cm.id), var, rest, st)
            return
        else:
            raise Refuse(st, f"with {ast.unparse(cm)} (manager not in the table)")
        self.emit(f"rocke_b_region_enter({self.builder}, {region});")
        self.with_items(rest, st)
        self.emit(f"rocke_b_region_leave({self.builder});")

    def enter_for(self, f: CE, var, rest, st):
        """`with loop as iv` / `as (iv, iter_vars)`: enter the body region, bind the targets."""
        self.emit(f"rocke_b_region_enter({self.builder}, {f.code}.body);")
        saved = set(self.declared)
        self.emit("{")
        self.ind += 1
        if isinstance(var, ast.Name):
            self.emit(f"{self.unused(var.id)}rocke_value_t* {var.id} = {f.code}.iv;")
            self.vars[var.id] = T.V
            self.declared.add(var.id)
        elif (
            isinstance(var, ast.Tuple)
            and len(var.elts) == 2
            and isinstance(var.elts[0], ast.Name)
        ):
            self.emit(
                f"{self.unused(var.elts[0].id)}rocke_value_t* {var.elts[0].id} = {f.code}.iv;"
            )
            self.vars[var.elts[0].id] = T.V
            self.declared.add(var.elts[0].id)
            cur = var.elts[1]
            if isinstance(cur, ast.Name):
                self.emit(
                    f"{self.unused(cur.id)}std::vector<rocke_value_t*> {cur.id}({f.code}.iter_vars, {f.code}.iter_vars + {f.code}.num_iter_vars);"
                )
                self.vars[cur.id] = T.vec(T.V)
                self.declared.add(cur.id)
            elif isinstance(cur, ast.Tuple):
                for k, n_ in enumerate(cur.elts):
                    self.emit(
                        f"{self.unused(n_.id)}rocke_value_t* {n_.id} = {f.code}.iter_vars[{k}];"
                    )
                    self.vars[n_.id] = T.V
                    self.declared.add(n_.id)
            else:
                raise Refuse(st, "loop target shape")
        elif var is not None:
            raise Refuse(st, "loop target shape")
        self.with_items(rest, st)
        self.ind -= 1
        self.emit("}")
        self.declared = saved
        self.emit(f"rocke_b_region_leave({self.builder});")

    def call_loop(self, method, node) -> CE:
        """scf_for / scf_for_iter: the iter_args list becomes a rocke_iter_arg_t array."""
        e = self.env.sig.ir[method]
        pyparams = [p for p in e["python"]["params"] if p["name"] != "self"]
        bound = self.bind_call_args(node, pyparams)

        def arg(n, t, default=None):
            if n in bound:
                return self.coerce(bound[n], t, node)
            p = next(p for p in pyparams if p["name"] == n)
            d = self.default_ce(p.get("default"), node)
            return self.coerce(d, t, node)

        lo, hi, step = (arg(n, T.V) for n in ("lower", "upper", "step"))
        iv = arg("iv_name", T.STR)
        if method == "scf_for":
            return CE(
                f"rocke_b_scf_for({self.builder}, {lo}, {hi}, {step}, {iv})",
                "rocke_for_t",
                False,
            )
        ia = bound.get("iter_args")
        if ia is None or T.seq_elem(ia.t) is None:
            raise Refuse(node, "scf_for_iter iter_args")
        arr_ = self.tmp("iter_args")
        self.emit(
            f"std::vector<rocke_iter_arg_t> {arr_} = py_adapt::iter_args({ia.code});"
        )
        unroll = arg("unroll", T.BOOL)
        elide = arg("elide_trailing_barrier", T.BOOL)
        return CE(
            f"rocke_b_scf_for_iter({self.builder}, {lo}, {hi}, {step}, {arr_}.data(), (int){arr_}.size(), {iv}, {unroll}, {elide})",
            "rocke_for_t",
            False,
        )

    def st_Try(self, st):
        """RT-R1: try/except over ckc exception classes; no else or finally.

        `except KeyError as e:` binds e to the caught ckc exception (type
        T.EXC) for its handler only, as Python unbinds it at the handler's end.
        The one use of e with a rule is str(e): a ckc exception's message is
        the Python str() of the exception it stands for (a KeyError's is the
        repr of its argument, as every translated raise and adapter writes it),
        so str(e) is its message."""
        if st.orelse or st.finalbody:
            raise Refuse(st, "try with else or finally")
        saved = set(self.declared)
        self.emit("try")
        self.emit("{")
        self.ind += 1
        self.block(st.body)
        self.ind -= 1
        self.emit("}")
        self.declared = saved
        cls_of = {
            "KeyError": "ckc::KeyError",
            "ValueError": "ckc::ValueError",
            "TypeError": "ckc::TypeError",
            "NotImplementedError": "ckc::NotImplError",
        }
        for h in st.handlers:
            if not isinstance(h.type, ast.Name) or h.type.id not in cls_of:
                raise Refuse(
                    st, f"except {ast.unparse(h.type) if h.type else ''} (RT-R1 shape)"
                )
            if h.name and (h.name in self.vars or h.name in self.declared):
                raise Refuse(st, f"except ... as {h.name}: the name is already a local")
            self.emit(
                f"catch(const {cls_of[h.type.id]}&{' ' + h.name if h.name else ''})"
            )
            self.emit("{")
            self.ind += 1
            if h.name:
                self.vars[h.name] = T.EXC
                self.declared.add(h.name)
            self.block(h.body)
            if h.name:
                del self.vars[h.name]
            self.ind -= 1
            self.emit("}")
            self.declared = saved

    def st_FunctionDef(self, st):
        """C1: a local closure becomes a [&] lambda."""
        from .host import closure_fields

        for (cls_, fld_), sites in closure_fields(self.mod).items():
            for i_, site in enumerate(sites):
                if site.node is st:
                    # host clones: stored into a Callable field: lifted to its own
                    # functions (one per value kind); the struct holds its kind
                    # and captured values (designs/host_clones.md)
                    self.vars[st.name] = f"closure:{cls_}.{fld_}:{i_}"
                    self.declared.add(st.name)
                    return
        params = []
        fact = self.mod.fact_of(st)
        for a in st.args.args:
            t = (
                T.from_annotation(ast.unparse(a.annotation), self.mod.class_struct)
                if a.annotation is not None
                else None
            )
            if t is None and fact and a.arg in fact["params"]:
                t = self.mod.tmap.tags(fact["params"][a.arg])
            if t is None:
                raise Refuse(
                    st, f"closure parameter {a.arg}: no annotation and no recorded type"
                )
            params.append((a.arg, t))
        if st.args.kwonlyargs or st.args.vararg or st.args.kwarg or st.args.defaults:
            raise Refuse(st, "closure with defaults or star arguments")
        ret = (
            T.from_annotation(ast.unparse(st.returns), self.mod.class_struct)
            if st.returns is not None
            else None
        )
        if ret is None and fact:
            rs = [r for r in fact["return"] if r != "None"] or fact["return"]
            ret = T.VOID if rs == ["None"] else self.mod.tmap.tags(rs, param=False)
        if ret is None:
            ret = T.AUTO
        sig = Sig(st.name, params, ret, st)
        inner = Fn(
            self.mod,
            st,
            sig,
            hints=self.hints.setdefault(
                "inner:" + st.name, {"vars": {}, "expr": {}, "elem": {}}
            ),
            outer=self,
            dry=self.dry,
        )
        inner.ind = self.ind + 1
        inner.run()
        ps = ", ".join(f"{decl_type(t)} {n}" for n, t in params)
        if sig.ret == T.AUTO:
            raise Refuse(st, f"closure {st.name}: no return type")
        self.emit(f"auto {st.name} = [&]({ps}) -> {decl_type(sig.ret)}")
        self.emit("{")
        self.lines.extend(inner.lines)
        self.emit("};")
        self.vars[st.name] = "lambda:" + st.name
        self.declared.add(st.name)
        self.mod.__dict__.setdefault("lambdas", {})[id(st)] = sig
        self.vars["sig:" + st.name] = sig

    # ------------------------------------------------------------- tests

    def test(self, node) -> str:
        ce = self.expr(node)
        return self.truth(ce, node)

    def truth(self, ce: CE, node) -> str:
        t = ce.t
        if t == T.BOOL:
            return ce.code
        if T.is_int(t):
            return f"{paren(ce.code)} != 0"
        if t == T.V:
            raise Refuse(node, "truthiness of a Value (CE-R3)")
        if T.is_ptr(t) or t == T.STR:
            if t == T.STR:
                return f"(({ce.code}) != nullptr && ({ce.code})[0] != '\\0')"
            return f"({ce.code}) != nullptr"
        oi = T.optional_inner(t)
        if oi:
            return f"({paren(ce.code)}.has_value() && ({self.truth(CE(f'(*{paren(ce.code)})', oi), node)}))"
        if t in ("dict?", "set?", "list?"):
            return "false"  # an empty display
        st_ = T.struct_of(t)
        if (
            st_
            and st_[0] in self.env.headers.structs
            and not st_[0].startswith("py_rec_")
        ):
            cls = self.mod.struct_class.get(st_[0]) or (
                self.mod.tmap.struct_class.get(st_[0]) if self.mod.tmap else None
            )
            pycls = self.py_class(cls) if cls else None
            if (
                pycls is not None
                and not hasattr(pycls, "__bool__")
                and not hasattr(pycls, "__len__")
            ):
                # an object with no __bool__ / __len__ is true (a pointer: when not NULL)
                return f"({ce.code}) != nullptr" if st_[1] else "true"
        rid = self.record_of(ce)
        if rid is not None:  # a record: some key is set
            sep = "->" if ce.t.endswith("*") else "."
            fs = [
                f"{ce.code}{sep}has_{record_field(k)}"
                for k in self.mod.records.get(rid, {})
            ]
            return "(" + " || ".join(fs) + ")" if fs else "false"
        if re.fullmatch(r"(?:const )?py::dict<.+>&?", t):
            return f"{paren(ce.code)}.size() != 0"
        if ce.meta and "n" in ce.meta:  # an owned or viewed C list: its count
            return f"({ce.meta['n']}) != 0"
        if T.is_seq(t):
            return f"!({ce.code}).empty()"
        if t == T.SSTR:
            return f"!({ce.code}).empty()"
        raise Refuse(node, f"truthiness of {t}")

    def _narrowing_from(self, test) -> dict:
        """isinstance(x, int) or `x is not None` (an optional) in an if test or
        an `and` narrows x in the body."""
        out = {}
        parts = (
            test.values
            if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)
            else [test]
        )
        for p in parts:
            if isinstance(p, ast.Name) and p.id in self.none_alias:
                p = self.none_alias[p.id]  # if flag: where flag = x is (not) None
            if (
                isinstance(p, ast.UnaryOp)
                and isinstance(p.op, ast.Not)
                and isinstance(p.operand, ast.Name)
                and p.operand.id in self.none_alias
            ):
                n = self._negated_narrowing(self.none_alias[p.operand.id])
                if n:
                    out.update(n)
                continue
            n = self._isinstance_narrow(p) or self._optional_narrow(p, ast.IsNot)
            if n:
                out.update(n)
        return out

    def _optional_narrow(self, test, op_kind) -> dict | None:
        """`x is not None` (op_kind IsNot) / `x is None` (Is) on a std::optional local: x is its value."""
        if not (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], op_kind)
        ):
            return None
        l, r = test.left, test.comparators[0]
        if not (
            isinstance(l, ast.Name) and isinstance(r, ast.Constant) and r.value is None
        ):
            return None
        cur = self.lookup(l.id)
        inner = T.optional_inner(cur.t) if cur is not None else None
        return {l.id: CE(f"(*{paren(cur.code)})", inner)} if inner else None

    def _negated_narrowing(self, test) -> dict:
        """not isinstance(x, int) narrows x to its Value (and the other way round);
        after `x is None` fails, an optional x is its value."""
        if isinstance(test, ast.Name) and test.id in self.none_alias:
            test = self.none_alias[test.id]
        o = self._optional_narrow(test, ast.Is)
        if o:
            return o
        n = self._isinstance_narrow(test)
        if not n:
            return {}
        x = test.args[0].id
        cur = self.lookup(x)
        alt_u = self.union_alt(cur.t, test.args[1]) if cur is not None else None
        if alt_u:
            others = [
                f for f in union_fields(self.env, T.struct_of(cur.t)[0]) if f != alt_u
            ]
            return {x: self.union_member(cur, others[0])} if len(others) == 1 else {}
        if cur is not None and T.variant_items(cur.t):
            alt = self.variant_alt(cur.t, test.args[1])
            others = [a for a in T.variant_items(cur.t) if a != alt]
            return (
                {x: CE(f"std::get<{others[0]}>({cur.code})", others[0])}
                if len(others) == 1
                else {}
            )
        cls = test.args[1].id
        return (
            {x: CE(f"({cur.code}).v", T.V)}
            if cls == "int"
            else {x: CE(f"({cur.code}).i", T.I64)}
        )

    def variant_alt(self, vt: str, cls_node) -> str | None:
        """The alternative of std::variant vt that isinstance(x, cls) selects."""
        alts = T.variant_items(vt)
        if not alts or not isinstance(cls_node, ast.Name):
            return None
        name = cls_node.id
        want = self.mod.class_struct.get(name) or (
            self.mod.tmap.class_struct.get(name) if self.mod.tmap else None
        )
        for a in alts:
            st_ = T.struct_of(a)
            if want and st_ and st_[0] == want:
                return a
            if name in ("list", "tuple", "Sequence") and T.is_seq(a):
                return a
            if name == "int" and T.is_int(a):
                return a
        return None

    def union_alt(self, ut: str, cls_node) -> str | None:
        """The member of a tagged values union struct that isinstance(x, cls) selects."""
        st_ = T.struct_of(ut)
        uf = union_fields(self.env, st_[0]) if st_ else None
        if not uf or not isinstance(cls_node, ast.Name):
            return None
        name = cls_node.id
        if name == "Value":
            return next((f for f, t in uf.items() if t == T.V), None)
        fam = self.mod.family(name) or (
            family_of(self.mod, name) if self.mod.tmap is not None else None
        )
        if fam is not None and name != fam.base:
            return None  # a subclass inside the alternative: not one member
        want = self.mod.class_struct.get(name) or (
            self.mod.tmap.class_struct.get(name) if self.mod.tmap else None
        )
        return next((f for f, t in uf.items() if want and t == want), None)

    def union_member(self, cur: CE, f: str) -> CE:
        st_ = T.struct_of(cur.t)
        uf = union_fields(self.env, st_[0])
        return CE(f"{paren(cur.code)}{'->' if st_[1] else '.'}{f}", uf[f], cur.pure)

    def _isinstance_narrow(self, node) -> dict | None:
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
        ):
            return None
        x, cls = node.args
        if not isinstance(x, ast.Name):
            return None
        cur = self.lookup(x.id)
        if cur is not None and self.union_alt(cur.t, cls):
            return {x.id: self.union_member(cur, self.union_alt(cur.t, cls))}
        if cur is not None and T.variant_items(cur.t):
            alt = self.variant_alt(cur.t, cls)
            return {x.id: CE(f"std::get<{alt}>({cur.code})", alt)} if alt else None
        if cur is None or cur.t != T.IV:
            return None
        if isinstance(cls, ast.Name) and cls.id == "int":
            return {x.id: CE(f"({cur.code}).i", T.I64)}
        if isinstance(cls, ast.Name) and cls.id == "Value":
            return {x.id: CE(f"({cur.code}).v", T.V)}
        return None

    # ------------------------------------------------------------- expressions

    def expr(self, node, want: str | None = None) -> CE:
        m = getattr(self, "ex_" + type(node).__name__, None)
        if m is None:
            raise Refuse(node, f"expression {type(node).__name__}")
        ce = m(node, want) if "want" in m.__code__.co_varnames else m(node)
        self.record(node, ce.t)
        return ce

    def ex_Constant(self, node):
        v = node.value
        if v is None:
            return CE("nullptr", T.NONE, lit=None)
        if isinstance(v, (bool, int, float, str)):
            return CE(lit_code(v), lit_type(v), lit=v)
        raise Refuse(node, f"constant {v!r}")

    def ex_Name(self, node):
        got = self.lookup(node.id)
        if got is not None:
            if got.t.startswith("lambda:"):
                raise Refuse(node, f"closure {node.id} used as a value (C2)")
            return got
        return self.global_name(node.id, node)

    def global_name(self, name, node) -> CE:
        mod = self.mod
        if name in mod.imports:
            src, attr = mod.imports[name]
            if src == "rocke.core.ir" and attr in T.IR_SCALARS:
                return CE(T.IR_SCALARS[attr], T.TYPE)
            val = getattr(mod.pymod, name, None)
            if val is None and src:
                try:  # a function-local import: the name lives in its module
                    val = getattr(
                        self.env.py.module(src if "." in src else src), attr, None
                    )
                except Exception:
                    val = None
            if isinstance(val, (bool, int, float, str)) and not callable(val):
                return CE(f"{lit_code(val)} /* {attr} */", lit_type(val), lit=val)
            from collections.abc import Mapping

            if isinstance(val, Mapping) and all(
                isinstance(k, (str, int)) and isinstance(v, (bool, int, float, str))
                for k, v in val.items()
            ):
                # an imported constant dict (MappingProxyType too): D1 lookups
                return CE(
                    attr.lstrip("_"), "pydict", lit=dict(val), meta={"name": attr}
                )
            raise Refuse(node, f"imported name {name} from {src}")
        if name in ("True", "False", "None"):
            return self.ex_Constant(
                ast.Constant(value={"True": True, "False": False, "None": None}[name])
            )
        val = getattr(mod.pymod, name, None)
        if name in mod.functions:
            raise Refuse(node, f"function {name} used as a value")
        if isinstance(val, (bool, int, float, str)):
            return CE(self.module_const(name, val), lit_type(val), lit=val)
        if isinstance(val, (tuple, frozenset)):
            return self.const_table(name, val, node)
        if isinstance(val, dict):
            return CE(name.lstrip("_"), "pydict", lit=val, meta={"name": name})
        hb = host_builders(mod)
        if type(val).__name__ in hb and getattr(
            mod.pymod, type(val).__name__, None
        ) is type(val):
            return self.host_instance(
                type(val).__name__
            )  # host clones: `_INT = IntLaneBuilder()`
        raise Refuse(node, f"global {name}")

    def const_table(self, name, val, node) -> CE:
        from .kernel import const_ctype, const_init

        t = const_ctype(val)
        if t is None:
            raise Refuse(node, f"constant {name} of mixed types")
        cname = name.lstrip("_")
        if name not in self.mod.private_consts:
            self.mod.private_consts[name] = (
                cname,
                f"static const {t} {cname} = {const_init(val, t, lit_code)};",
            )
        return CE(cname, t, True, lit=val)

    def module_const(self, name: str, val) -> str:
        """A module constant: the header's extern twin if any, else a file static."""
        mod = self.mod
        cname = name.lstrip("_")
        if name in mod.private_consts:
            return mod.private_consts[name][0]
        ext = PKG.macro + cname.upper()
        if not name.startswith("_") and (
            ext in self.env.headers.vars or ext in self.env.headers.macros
        ):
            return ext
        t = lit_type(val)
        ct = {
            "int64_t": "int64_t",
            "bool": "bool",
            "double": "double",
            "const char*": "const char*",
        }[t]
        src_lit = lit_code(val)
        for n in mod.tree.body:
            if (
                isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == name
                and isinstance(n.value, ast.Constant)
                and isinstance(val, int)
                and not isinstance(val, bool)
            ):
                seg = ast.get_source_segment(mod.src, n.value)
                if seg and seg.lower().startswith("0x"):
                    src_lit = seg
        decl = f"static const {ct} {cname} = {src_lit};"
        if ct == "const char*":
            decl = f"static const char* const {cname} = {lit_code(val)};"
        mod.private_consts[name] = (cname, decl)
        return cname

    def ex_Attribute(self, node):
        if (
            isinstance(node.value, ast.Name)
            and self.lookup(node.value.id) is None
            and (node.value.id in self.mod.classes or node.value.id in self.mod.imports)
        ):
            # a class constant (DmaSlabLayout.CHUNK): its literal value
            pycls = self.py_class(node.value.id) or getattr(
                self.mod.pymod, node.value.id, None
            )
            v = getattr(pycls, node.attr, None) if isinstance(pycls, type) else None
            if isinstance(v, (bool, int, float, str)):
                return CE(lit_code(v), lit_type(v), lit=v)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in self.mod.imports
            and self.lookup(node.value.id) is None
        ):
            raise Refuse(node, f"module attribute {ast.unparse(node)}")
        obj = self.expr(node.value)
        return self.field(obj, node.attr, node)

    def class_constant(self, sname: str, attr: str, obj: CE | None = None):
        """self.X for a class-level literal X no subclass in this module
        overrides (a subclass that does is virtual dispatch: a hole). In a
        tagged-value family the kind tag selects each member's value."""
        cls = self.mod.struct_class.get(sname) or (
            self.mod.tmap.struct_class.get(sname) if self.mod.tmap else None
        )
        fam = self.mod.family(cls)
        if fam is not None and obj is not None:
            scls = self.static_class(obj, cls)
            vals = {}
            for m in fam.derived(scls):
                pc = self.py_class(m) if not fam.is_peer(m) else None
                v = getattr(pc, attr, None) if pc is not None else None
                if not isinstance(v, (bool, int, float, str)) or callable(v):
                    return None
                vals[m] = v
            if not vals:
                return None
            if len({repr(v) for v in vals.values()}) == 1:
                v = next(iter(vals.values()))
                return CE(lit_code(v), lit_type(v), lit=v)
            if len({lit_type(v) for v in vals.values()}) != 1:
                return None
            k = f"{paren(obj.code)}{'->' if obj.t.endswith('*') else '.'}kind"
            dflt = vals.get(scls, next(iter(vals.values())))
            code = lit_code(dflt)
            for m, v in vals.items():
                if v != dflt:
                    code = f"({k} == {kind_const(fam, m)} ? {lit_code(v)} : {code})"
            return CE(code, lit_type(dflt), obj.pure)
        pycls = self.py_class(cls) if cls else None
        if pycls is None:
            return None
        v = getattr(pycls, attr, None)
        if not isinstance(v, (bool, int, float, str)) or callable(v):
            return None
        for other in vars(self.mod.pymod).values():
            if (
                isinstance(other, type)
                and other is not pycls
                and issubclass(other, pycls)
                and getattr(other, attr, v) != v
            ):
                return None
        # every subclass sharing this struct sees the same value
        return CE(lit_code(v), lit_type(v), lit=v)

    def property_call(self, obj: CE, sname: str, attr: str, node):
        """obj.attr where attr is a @property / @cached_property: a call of the
        method (a cached_property is pure here: computed from frozen fields)."""
        import functools

        cls = self.mod.struct_class.get(sname) or (
            self.mod.tmap.struct_class.get(sname) if self.mod.tmap else None
        )
        fam = self.mod.family(cls)
        if fam is not None:
            impl = fam.impl(self.static_class(obj, cls), attr) or next(
                iter(fam.impls(cls, attr).values()), None
            )
            cls = impl or cls
        pycls = self.py_class(cls) if cls else None
        d = getattr(pycls, "__dict__", {}).get(attr) if pycls is not None else None
        if d is None and pycls is not None:
            d = next(
                (k.__dict__[attr] for k in pycls.__mro__ if attr in k.__dict__), None
            )
        if not isinstance(d, (property, functools.cached_property)):
            return None
        call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="_", ctx=ast.Load()), attr=attr, ctx=ast.Load()
            ),
            args=[],
            keywords=[],
        )
        ast.copy_location(call, node)
        return self.call_struct(obj, attr, call)

    def field(self, obj: CE, attr: str, node) -> CE:
        t = obj.t
        from .adapters import attr_adapter

        ad = attr_adapter(t, attr)
        if ad is not None:
            return CE(ad[0].format(obj=obj.code), ad[1], obj.pure)
        if t == T.B and attr == "kernel":
            return CE(f"rocke_ir_builder_kernel({obj.code})", "rocke_kernel_def_t*")
        if t == "rocke_for_t" and attr == "results":
            return CE(
                f"py::span<rocke_value_t*>{{{obj.code}.op->results, {obj.code}.op->num_results}}",
                "py::span<rocke_value_t*>",
                meta={
                    "p": f"{obj.code}.op->results",
                    "n": f"{obj.code}.op->num_results",
                },
            )
        if t == T.V and attr == "type":
            return CE(f"{obj.code}->type", T.TYPE)
        if t == T.TYPE:
            if attr == "count":
                return CE(f"{obj.code}->count", T.INT)
            if attr == "elem":
                return CE(f"{obj.code}->elem", T.TYPE)
            if attr == "name":
                return CE(
                    f"{obj.code}->name", T.STR, obj.pure
                )  # the canonical text, as Python's
            raise Refuse(node, f"type attribute {attr}")
        st = T.struct_of(t)
        if st is not None:
            name, is_p = st
            s = self.env.headers.structs.get(name)
            if s is None:
                raise Refuse(node, f"no struct {name}")
            sep = "->" if is_p else "."
            cf = attr if attr in s.fields else attr.lstrip("_")
            if cf not in s.fields:
                prop = self.property_call(obj, name, attr, node)
                if prop is not None:
                    return prop
                const = self.class_constant(name, attr, obj)
                if const is not None:
                    return const
                raise Refuse(node, f"{name} has no field {attr}")
            ft = s.fields[cf]
            fst = T.struct_of(ft)
            hs = self.env.headers.structs.get(fst[0]) if fst and not fst[1] else None
            if hs is not None and len(hs.order) == 2 and hs.order[0] == "has":
                # {has, value}: the hand port's Optional[X] (rocke_band_limit_t)
                inner_f = hs.order[1]
                it = hs.fields[inner_f]
                base_ = f"{obj.code}{sep}{cf}"
                if it == "rocke_int_or_value_t":
                    return CE(
                        f"({base_}.has ? std::optional<{T.IV}>(py_adapt::iv_of<{T.IV}>({base_}.{inner_f})) : std::optional<{T.IV}>())",
                        T.optional(T.IV),
                        obj.pure,
                    )
                return CE(
                    f"({base_}.has ? std::optional<{it}>({base_}.{inner_f}) : std::optional<{it}>())",
                    T.optional(it),
                    obj.pure,
                )
            if ft == "rocke_int_or_value_t":
                return CE(
                    f"py_adapt::iv_of<{T.IV}>({obj.code}{sep}{cf})", T.IV, obj.pure
                )
            n_f, cap_f = "n_" + cf, "cap_" + cf
            if n_f in s.fields and ft.endswith("*"):
                el = T.pointee_elem(
                    ft if ("const" in ft or ft.endswith("**")) else "const " + ft
                )
                if el is None:
                    raise Refuse(node, f"sequence field {cf} of {ft}")
                base = f"{obj.code}{sep}"
                if cap_f in s.fields:
                    return CE(
                        f"{base}{cf}",
                        f"py::olist<{el}>",
                        True,
                        meta={
                            "n": f"{base}{n_f}",
                            "cap": f"{base}{cap_f}",
                            "p": f"{base}{cf}",
                        },
                    )
                return CE(
                    f"py::span<{el}>{{{base}{cf}, {base}{n_f}}}",
                    f"py::span<{el}>",
                    True,
                    meta={"n": f"{base}{n_f}", "p": f"{base}{cf}"},
                )
            attr = cf
            if "has_" + cf in s.fields:
                return CE(
                    f"{obj.code}{sep}{cf}",
                    s.fields[cf],
                    meta={"has": f"{obj.code}{sep}has_{cf}"},
                )
            if ft.endswith("[]"):
                dim = getattr(s, "dims", {}).get(cf)
                if not dim or "[" in ft[:-2]:
                    raise Refuse(node, f"array field {attr}")
                el = ft[:-2]
                base_ = f"{obj.code}{'->' if is_p else '.'}{cf}"
                # a fixed C array field, read as a view of its declared length
                return CE(
                    f"py::span<{el}>{{{base_}, (int64_t)({dim})}}",
                    f"py::span<{el}>",
                    obj.pure,
                    meta={"p": base_, "n": f"(int64_t)({dim})"},
                )
            return CE(f"{obj.code}{'->' if is_p else '.'}{attr}", ft)
        raise Refuse(node, f"attribute {attr} of {t}")

    def readable(self, ce: CE) -> CE:
        """An owned C list field (py::olist: pointer, n_, cap_) read as a view."""
        if (
            ce.t.startswith("py::olist<")
            and ce.meta
            and "p" in ce.meta
            and "n" in ce.meta
        ):
            el = T.seq_elem(ce.t)
            return CE(
                f"py::span<{el}>{{{ce.meta['p']}, {ce.meta['n']}}}",
                f"py::span<{el}>",
                ce.pure,
                meta=ce.meta,
            )
        return ce

    def coerce_seq(self, ce: CE, el: str, node) -> CE:
        """A sequence as std::vector<el> (a copy when the element type differs)."""
        if T.seq_elem(ce.t) == el and ce.t.startswith("std::vector"):
            return ce
        if T.seq_elem(ce.t) is None:
            raise Refuse(node, f"{ce.t} is not a sequence of {el}")
        src = self.iter_src(self.readable(ce).code)
        return CE(f"std::vector<{el}>({src}.begin(), {src}.end())", T.vec(el), False)

    def tuple_struct_items(self, ce: CE):
        """A generated fixed-tuple struct (rocke_*_tuple_*_t): its items f0..fn
        as (code, type), else None."""
        st = T.struct_of(ce.t)
        if st is None or "_tuple_" not in st[0]:
            return None
        s = self.env.headers.structs.get(st[0])
        if s is None or not s.order or any(f != f"f{i}" for i, f in enumerate(s.order)):
            return None
        sep = "->" if st[1] else "."
        return [(f"{ce.code}{sep}{f}", s.fields[f]) for f in s.order]

    def ex_Subscript(self, node):
        if isinstance(node.value, ast.Dict):
            return self.dict_display_at(node)
        obj = self.expr(node.value)
        if self.record_of(obj) is not None:
            if not (
                isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                raise Refuse(node, "record key not a literal (R-REC)")
            return self.record_get(obj, node.slice.value, node)
        if isinstance(node.slice, ast.Slice):
            sl = node.slice
            if sl.step is None and T.is_str(obj.t):
                lo = (
                    self.expr(sl.lower)
                    if sl.lower is not None
                    else CE("0", T.I64, lit=0)
                )
                hi = self.expr(sl.upper) if sl.upper is not None else None
                return CE(
                    f"py::str_slice({obj.code}, {lo.code}{', ' + hi.code if hi else ''})",
                    T.SSTR,
                    obj.pure and lo.pure,
                )
            if sl.step is not None or T.seq_elem(obj.t) is None:
                raise Refuse(node, "slice with a step or of a non-sequence (T5)")
            lo = self.expr(sl.lower) if sl.lower is not None else CE("0", T.I64, lit=0)
            hi = (
                self.expr(sl.upper)
                if sl.upper is not None
                else CE(f"py::len({obj.code})", T.I64)
            )
            t = T.vec(T.seq_elem(obj.t))
            return CE(
                f"py::slice({obj.code}, {lo.code}, {hi.code})",
                t,
                obj.pure and lo.pure and hi.pure,
            )
        idx = self.expr(node.slice)
        obj = self.readable(obj)
        if T.optional_inner(obj.t):
            # None[...] raises TypeError in Python: py::deref raises the same
            obj = CE(f"py::deref({obj.code})", T.optional_inner(obj.t), obj.pure)
        if re.fullmatch(r"(?:const )?py::dict<.+>&?", obj.t):
            kv_ = re.fullmatch(r"(?:const )?py::dict<(.+)>&?", obj.t).group(1)
            kt_, vt = [
                x.strip().replace("[", "<").replace("]", ">")
                for x in T._split_top(kv_.replace("<", "[").replace(">", "]"))
            ]
            return CE(
                f"{paren(obj.code)}.at({self.coerce(idx, kt_, node)})",
                vt,
                obj.pure and idx.pure,
            )
        if obj.t == "py::kargs":
            return CE(
                f"{obj.code}.at({self.coerce(idx, T.SSTR, node)})",
                T.V,
                obj.pure and idx.pure,
            )
        if obj.t == "pydict":
            return self.const_dict_at(obj, idx, node)
        ti = T.tuple_items(obj.t)
        if ti is not None:
            if not (isinstance(idx.lit, int) and -len(ti) <= idx.lit < len(ti)):
                raise Refuse(node, f"index {idx.code} of a tuple {obj.t}")
            i_ = idx.lit % len(ti)
            return CE(f"std::get<{i_}>({obj.code})", ti[i_], obj.pure)
        ts = self.tuple_struct_items(obj)
        if ts is not None:
            if not (isinstance(idx.lit, int) and -len(ts) <= idx.lit < len(ts)):
                raise Refuse(node, f"index {idx.code} of a fixed tuple {obj.t}")
            code, t = ts[idx.lit]
            return CE(code, t, obj.pure)
        if T.is_str(obj.t) and T.is_int(idx.t):
            # S-IDX: s[i] of a str is the one-character str at i (Python's
            # negative-index rule; out of range raises, as py::at does)
            return CE(
                f"py::str_at({obj.code}, {idx.code})", T.SSTR, obj.pure and idx.pure
            )
        el = T.seq_elem(obj.t)
        if el is None:
            raise Refuse(node, f"subscript of {obj.t}")
        if not T.is_int(idx.t):
            raise Refuse(node, f"index of type {idx.t}")
        if (
            idx.lit is not None
            and isinstance(idx.lit, int)
            and idx.lit >= 0
            and T.arr_len(obj.t) is not None
            and idx.lit < T.arr_len(obj.t)
        ):
            return CE(f"{obj.code}[{idx.lit}]", el, obj.pure and idx.pure)
        return CE(f"py::at({obj.code}, {idx.code})", el, obj.pure and idx.pure)

    def const_dict_get(self, obj: CE, node) -> CE:
        """D1: d.get(key[, default]) on a constant dict: the value, else the default
        (None: a null pointer)."""
        d = obj.lit
        name = obj.meta["name"].lstrip("_")
        key = self.bind(self.expr(node.args[0]), "key")
        if not all(isinstance(k, str) for k in d):
            raise Refuse(node, f"constant dict {name} with non-str keys")
        vals = [self._const_value(v, node) for v in d.values()]
        vt = vals[0].t
        if any(v.t != vt for v in vals):
            raise Refuse(node, f"constant dict {name} with mixed values")
        if len(node.args) == 2:
            dflt = self.bind(self.expr(node.args[1]), "dflt")
        elif T.is_ptr(vt) or vt == T.STR:
            dflt = CE("nullptr", T.NONE, lit=None)
        else:
            raise Refuse(node, f"{name}.get() without a default on {vt} values")
        fn = f"{name}_get"
        body = [
            f"static {decl_type(vt)} {fn}(const char* key, {decl_type(vt)} dflt)",
            "{",
        ]
        for k, v in zip(d, vals):
            body.append(f"    if(py::eq(key, {lit_code(k)})) return {v.code};")
        body += ["    return dflt;", "}"]
        self.mod.private_consts.setdefault("fn:" + fn, (fn, "\n".join(body)))
        return CE(
            f"{fn}({self.coerce(key, T.STR, node)}, {self.coerce(dflt, vt, node)})",
            vt,
            key.pure and dflt.pure,
        )

    def const_dict_at(self, obj: CE, idx: CE, node) -> CE:
        """D1: a constant dict indexed by a run-time key: a lookup function, KeyError on a miss."""
        from .kernel import const_ctype, const_init

        d = obj.lit
        name = obj.meta["name"].lstrip("_")
        int_keys = all(isinstance(k, int) and not isinstance(k, bool) for k in d)
        if not int_keys and not all(isinstance(k, str) for k in d):
            raise Refuse(node, f"constant dict {name} with mixed keys")
        if idx.lit is not None and idx.lit in d:
            v = d[idx.lit]
            return self._const_value(v, node)
        vals = list(d.values())
        fn = f"{name}_at"
        if all(self._ir_type_name(v) for v in vals):
            vt = T.TYPE
            body = [f"static const rocke_type_t* {fn}(const char* key)", "{"]
            for k, v in d.items():
                body.append(
                    f"    if(py::eq(key, {lit_code(k)})) return {self._ir_type_name(v)};"
                )
            body.append("    throw ckc::KeyError(py::repr(key));")
            body.append("}")
        else:
            vt = const_ctype(vals[0])
            if vt is None or any(const_ctype(v) != vt for v in vals):
                raise Refuse(node, f"constant dict {name} with mixed values")
            body = [
                f"static {vt} {fn}({'int64_t' if int_keys else 'const char*'} key)",
                "{",
            ]
            for k, v in d.items():
                init = const_init(v, vt, lit_code)
                body.append(
                    f"    if({'key == ' + lit_code(k) if int_keys else 'py::eq(key, ' + lit_code(k) + ')'}) return {vt + init if init.startswith('{') else init};"
                )
            body.append("    throw ckc::KeyError(py::repr(key));")
            body.append("}")
        self.mod.private_consts.setdefault("fn:" + fn, (fn, "\n".join(body)))
        return CE(
            f"{fn}({self.coerce(idx, T.I64 if int_keys else T.STR, node)})",
            vt,
            idx.pure,
        )

    def _ir_type_name(self, v):
        import rocke.core.ir as I

        for k, c in T.IR_SCALARS.items():
            if getattr(I, k, None) is v:
                return c
        return None

    def _const_value(self, v, node) -> CE:
        c = self._ir_type_name(v)
        if c:
            return CE(c, T.TYPE)
        if isinstance(v, (bool, int, float, str)):
            return CE(lit_code(v), lit_type(v), lit=v)
        if isinstance(v, (tuple, frozenset)):
            from .kernel import const_ctype, const_init

            t = const_ctype(v)
            if t is None:
                raise Refuse(node, "mixed constant tuple")
            return CE(f"{t}{const_init(v, t, lit_code)}", t, lit=v)
        raise Refuse(node, f"constant value {v!r}")

    def dict_display_call(self, node) -> CE:
        """{k: obj.m, ...}[key](args): only the chosen callable is called, so
        an if-chain on the key with one call per branch (S4, lazy)."""
        sub = node.func
        d = sub.value
        if not all(
            isinstance(k, ast.Constant) and isinstance(k.value, str) for k in d.keys
        ):
            raise Refuse(node, "called dict display with non-literal keys (S4)")
        key = self.bind(self.expr(sub.slice), "key")
        res = None
        t = None
        first = True
        for k, v in zip(d.keys, d.values):
            call = ast.copy_location(
                ast.Call(func=v, args=node.args, keywords=node.keywords), node
            )
            self.emit(
                f"{'if' if first else 'else if'}(py::eq({self.coerce(key, T.STR, node)}, {lit_code(k.value)}))"
            )
            self.emit("{")
            self.ind += 1
            ce = self.expr(call)
            if res is None:
                t = ce.t
                res = self.tmp("pick")
                # declared before the chain: move the declaration up
                self.lines.insert(
                    self.lines.index(
                        "    " * (self.ind - 1)
                        + f"if(py::eq({self.coerce(key, T.STR, node)}, {lit_code(d.keys[0].value)}))"
                    ),
                    "    " * (self.ind - 1) + f"{decl_type(t)} {res}{{}};",
                )
            self.emit(f"{res} = {self.coerce(ce, t, node)};")
            self.ind -= 1
            self.emit("}")
            first = False
        self.emit(
            f"else throw ckc::KeyError(py::repr({self.coerce(key, T.STR, node)}));"
        )
        return CE(res, t)

    def dict_display_at(self, node) -> CE:
        """S4: {k: v, ...}[key] with literal keys: an if-chain on the key."""
        d = node.value
        if not all(
            isinstance(k, ast.Constant) and isinstance(k.value, str) for k in d.keys
        ):
            raise Refuse(node, "indexed dict display with non-literal keys (S4)")
        key = self.bind(self.expr(node.slice), "key")
        vals = [self.expr(v) for v in d.values]
        if not all(v.pure for v in vals):
            raise Refuse(node, "indexed dict display with emitting values")
        t = vals[0].t
        for v in vals[1:]:
            t = T.common(t, v.t) or t
        res = self.tmp("pick")
        self.emit(f"{decl_type(t)} {res}{{}};")
        first = True
        for k, v in zip(d.keys, vals):
            kw = "if" if first else "else if"
            self.emit(
                f"{kw}(py::eq({self.coerce(key, T.STR, node)}, {lit_code(k.value)})) {res} = {self.coerce(v, t, node)};"
            )
            first = False
        self.emit(
            f"else throw ckc::KeyError(py::repr({self.coerce(key, T.STR, node)}));"
        )
        return CE(res, t)

    def ex_Tuple(self, node, want=None):
        return self.display(node, want)

    def ex_List(self, node, want=None):
        if not node.elts:
            if want and T.is_seq(want):
                return CE(f"{decl_type(want)}{{}}", want)
            return CE("{}", "list?")
        return self.display(node, want, as_list=True)

    def display(self, node, want, as_list=False):
        vals = [self.bind(self.expr(e), "e") for e in node.elts]
        ts = [v.t for v in vals]
        if not vals:
            if want is not None and T.is_seq(want):
                return CE(f"{decl_type(want)}{{}}", want)
            return CE("{}", "list?")
        el = ts[0]
        for t in ts[1:]:
            c = T.common(el, t)
            if c is None:
                if as_list:
                    raise Refuse(node, f"list of mixed {ts} (T3)")
                wi = T.tuple_items(want) if want else None
                if wi and len(wi) == len(vals):
                    # the expected tuple types each item ({} and None take the item's type)
                    items = ", ".join(self.coerce(v, w, node) for v, w in zip(vals, wi))
                    return CE(
                        f"{decl_type(want)}{{{items}}}", want, all(v.pure for v in vals)
                    )
                if any("?" in x for x in ts):
                    raise Refuse(node, f"tuple item of unknown type in {ts}")
                tt = f"std::tuple<{', '.join(decl_type(x) for x in ts)}>"
                return CE(
                    f"{tt}{{{', '.join(v.code for v in vals)}}}",
                    tt,
                    all(v.pure for v in vals),
                )
            el = c
        if want is not None and T.seq_elem(want) is not None:
            el = T.seq_elem(want)
        t = T.vec(el) if as_list else T.arr(el, len(vals))
        if want is not None and T.is_seq(want):
            t = want
        items = ", ".join(self.coerce(v, el, node) for v in vals)
        return CE(f"{decl_type(t)}{{{items}}}", t, all(v.pure for v in vals))

    def ex_BinOp(self, node):
        if (
            isinstance(node.op, ast.Mult)
            and isinstance(node.left, ast.List)
            and len(node.left.elts) == 1
        ):
            v = self.bind(self.expr(node.left.elts[0]), "e")
            n = self.expr(node.right)
            if not T.is_int(n.t):
                raise Refuse(node, "list repetition by a non-int")
            return CE(
                f"std::vector<{decl_type(v.t)}>((size_t)std::max<int64_t>(0, {n.code}), {v.code})",
                T.vec(v.t),
                v.pure and n.pure,
            )
        if isinstance(node.op, ast.Add) and (
            self._is_list_node(node.left) or self._is_list_node(node.right)
        ):
            return self.list_concat(node)
        a = self.expr(node.left)
        b = self.expr(node.right)
        if isinstance(node.op, ast.Add) and T.is_seq(a.t) and T.is_seq(b.t):
            el = T.common(T.seq_elem(a.t), T.seq_elem(b.t))
            if el is None:
                raise Refuse(node, f"concatenating {a.t} and {b.t}")
            name = self.tmp("cat")
            src_a = self.iter_src(a.code)
            self.emit(f"{decl_type(T.vec(el))} {name}({src_a}.begin(), {src_a}.end());")
            src_b = self.iter_src(b.code)
            self.emit(f"{name}.insert({name}.end(), {src_b}.begin(), {src_b}.end());")
            return CE(name, T.vec(el))
        return self.binop(node.op, a, b, node)

    def _is_list_node(self, n):
        return isinstance(n, (ast.List, ast.ListComp)) or (
            isinstance(n, ast.BinOp)
            and isinstance(n.op, ast.Add)
            and (self._is_list_node(n.left) or self._is_list_node(n.right))
        )

    def list_concat(self, node):
        parts = []

        def flat(n):
            if (
                isinstance(n, ast.BinOp)
                and isinstance(n.op, ast.Add)
                and self._is_list_node(n)
            ):
                flat(n.left)
                flat(n.right)
            else:
                parts.append(n)

        flat(node)
        key = (node.lineno, node.col_offset, "concat")
        el = self.hints["expr"].get(key)
        name = self.tmp("lst")
        if el is None and not self.dry:
            raise Refuse(node, "list concatenation with unknown element type")
        self.emit(f"{decl_type(T.vec(el or 'auto'))} {name};")
        types = []
        for p in parts:
            if isinstance(p, ast.List):
                for e in p.elts:
                    v = self.bind(self.expr(e), "e")
                    types.append(v.t)
                    self.emit(
                        f"{name}.push_back({self.coerce(v, el, p) if el else v.code});"
                    )
            elif isinstance(p, ast.ListComp):
                types.append(self.comp_into(p, name, el))
            else:
                v = self.bind(self.expr(p), "e")
                ve = T.seq_elem(v.t)
                if ve is None:
                    raise Refuse(p, f"concatenating {v.t}")
                types.append(ve)
                src_v = self.iter_src(v.code)
                self.emit(
                    f"{name}.insert({name}.end(), {src_v}.begin(), {src_v}.end());"
                )
        if self.dry:
            c = None
            for t in types:
                if t is None or t == "list?":
                    continue
                c = t if c is None else T.common(c, t)
                if c is None:
                    raise Refuse(node, f"list of mixed {types}")
            if c is not None:
                self.hints["expr"][key] = c
            el = c
        return CE(name, T.vec(el or T.AUTO))

    def binop(self, op, a: CE, b: CE, node) -> CE:
        if a.lit is not None and b.lit is not None and not isinstance(a.lit, tuple):
            try:
                v = eval(
                    compile(
                        ast.Expression(
                            ast.BinOp(ast.Constant(a.lit), op, ast.Constant(b.lit))
                        ),
                        "<fold>",
                        "eval",
                    )
                )
                return CE(lit_code(v), lit_type(v), lit=v)
            except Exception:
                pass
        if (a.t == T.IV and T.is_int(b.t)) or (b.t == T.IV and T.is_int(a.t)):
            # int | Value in host arithmetic: an IR value has no Python operators
            a = (
                CE(f"py_adapt::iv_int_of({a.code})", T.I64, a.pure)
                if a.t == T.IV
                else a
            )
            b = (
                CE(f"py_adapt::iv_int_of({b.code})", T.I64, b.pure)
                if b.t == T.IV
                else b
            )
        if T.is_int(a.t) and T.is_int(b.t):
            pure = a.pure and b.pure
            sym = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.BitAnd: "&",
                ast.BitOr: "|",
                ast.BitXor: "^",
                ast.LShift: "<<",
                ast.RShift: ">>",
            }.get(type(op))
            if sym in ("<<", ">>"):
                # Python ints do not overflow at 32 bits: shift a 64-bit left operand
                return CE(
                    f"((int64_t){paren(a.code)} {sym} {paren(b.code)})", T.I64, pure
                )
            if sym:
                if a.t != T.I64 and b.t != T.I64:
                    return CE(
                        f"((int64_t){paren(a.code)} {sym} {paren(b.code)})", T.I64, pure
                    )
                return CE(f"({a.code} {sym} {b.code})", T.I64, pure)
            if isinstance(op, ast.FloorDiv):
                return CE(f"py::floordiv({a.code}, {b.code})", T.I64, pure)
            if isinstance(op, ast.Mod):
                return CE(f"py::mod({a.code}, {b.code})", T.I64, pure)
        if isinstance(op, ast.Mod) and T.is_str(a.t) and a.lit is not None:
            # "...%d..." % (x, y): %d / %s / %x / %% only
            if re.search(r"%[^dsx%]", a.lit.replace("%%", "")):
                raise Refuse(node, "printf-style format beyond %d %s %x")
            vals = []
            if isinstance(node.right, ast.Tuple):
                vals = [self.bind(self.expr(e), "f") for e in node.right.elts]
            else:
                vals = [b]
            convs = re.findall(r"%([dsx%])", a.lit)
            convs = [c for c in convs if c != "%"]
            if len(convs) != len(vals):
                raise Refuse(node, "printf-style format arity")
            k = 0
            out = []
            for piece in re.split(r"(%[dsx%])", a.lit):
                if piece == "%%":
                    out.append(c_string("%"))
                elif piece in ("%d", "%s", "%x"):
                    v = vals[k]
                    k += 1
                    out.append(
                        f"py::hex({v.code}, false)"
                        if piece == "%x"
                        else f"py::str({v.code})"
                    )
                elif piece:
                    out.append(f"std::string({c_string(piece)})")
            return CE(
                "(" + " + ".join(out) + ")" if out else 'std::string("")', T.SSTR, False
            )
        if isinstance(op, ast.Add) and T.is_str(a.t) and T.is_str(b.t):
            return CE(f"(std::string({a.code}) + {b.code})", T.SSTR, a.pure and b.pure)
        raise Refuse(node, f"binary {type(op).__name__} on {a.t}, {b.t}")

    def ex_UnaryOp(self, node):
        v = self.expr(node.operand)
        if isinstance(node.op, ast.Not):
            if isinstance(v.lit, bool):
                return CE("false" if v.lit else "true", T.BOOL, lit=not v.lit)
            return CE(negate(self.truth(v, node)), T.BOOL, v.pure)
        if isinstance(node.op, ast.USub):
            if v.lit is not None:
                return CE(lit_code(-v.lit), v.t, lit=-v.lit)
            if T.is_int(v.t) or v.t == T.F64:
                return CE(f"-({v.code})", v.t, v.pure)
        raise Refuse(node, f"unary {type(node.op).__name__} on {v.t}")

    def ex_BoolOp(self, node):
        value = self.bool_op_value(node)
        if value is not None:
            return value
        sym = "&&" if isinstance(node.op, ast.And) else "||"
        codes = []
        pure = True
        pushed = 0
        # emitting operands stay inside their arm: lower each in a test block
        start = len(self.lines)
        parts = []
        for v in node.values:
            mark = len(self.lines)
            ce = self.expr(v)
            emitted = self.lines[mark:]
            if emitted:
                pure = False
            parts.append((ce, emitted, v))
            # later operands run only when this one decided to go on:
            # `x and y` sees x, `x or y` sees not x
            n = (
                self._narrowing_from(v)
                if isinstance(node.op, ast.And)
                else self._negated_narrowing(v)
            )
            if n:
                self.narrow.append(n)
                pushed += 1
        for _ in range(pushed):
            self.narrow.pop()
        if pure:
            lits = [ce.lit for ce, _e, _v in parts]
            if isinstance(node.op, ast.And) and any(l is False for l in lits):
                return CE("false", T.BOOL, lit=False)
            if isinstance(node.op, ast.Or) and any(l is True for l in lits):
                return CE("true", T.BOOL, lit=True)
            if all(isinstance(l, bool) for l in lits):
                v_ = all(lits) if isinstance(node.op, ast.And) else any(lits)
                return CE("true" if v_ else "false", T.BOOL, lit=v_)
            for ce, _e, v in parts:
                t_ = self.truth(ce, v)
                codes.append(t_ if is_atom(t_) else f"({t_})")
            return CE(
                (
                    " " + sym + " ".join(codes)[len(sym) :]
                    if False
                    else f" {sym} ".join(codes)
                ),
                T.BOOL,
                all(c.pure for c, _, _ in parts),
            )
        # impure: rebuild as nested ifs so later operands emit only when reached
        del self.lines[start:]
        res = self.tmp("cond")
        self.emit(f"bool {res} = false;")
        self.narrow_stack_len = len(self.narrow)
        depth = 0
        for i, v in enumerate(node.values):
            ce = self.expr(v)
            t = self.truth(ce, v)
            self.emit(f"{res} = {t};")
            if i < len(node.values) - 1:
                self.emit(f"if({'' if sym == '&&' else '!'}{res})")
                self.emit("{")
                self.ind += 1
                depth += 1
                n = (
                    self._narrowing_from(v)
                    if isinstance(node.op, ast.And)
                    else self._negated_narrowing(v)
                )
                if n:
                    self.narrow.append(n)
        for _ in range(depth):
            self.ind -= 1
            self.emit("}")
        while len(self.narrow) > self.narrow_stack_len:
            self.narrow.pop()
        return CE(res, T.BOOL)

    def bool_op_value(self, node):
        """`a or b` / `a and b` on non-bool operands: Python's value (the
        deciding operand), when the operands are pure and share a type."""
        mark = len(self.lines)
        try:
            vals = [self.expr(v) for v in node.values]
        except Refuse:
            del self.lines[mark:]
            return None
        if (len(self.lines) != mark or not all(v.pure for v in vals)) and not any(
            v.t == T.BOOL for v in vals
        ):
            # emitting operands: Python evaluates the next one only when needed
            del self.lines[mark:]
            return self.bool_op_lazy(node)
        if (
            len(self.lines) != mark
            or not all(v.pure for v in vals)
            or any(v.t == T.BOOL for v in vals)
        ):
            del self.lines[mark:]
            return None
        t = vals[0].t
        for v in vals[1:]:
            t = T.common(t, v.t)
            if t is None:
                del self.lines[mark:]
                return None
        if t == T.STR and any(v.t == T.SSTR for v in vals):
            t = T.SSTR
        out = self.coerce(vals[-1], t, node)
        for v in reversed(vals[:-1]):
            c = self.truth(v, node)
            pick = self.coerce(v, t, node)
            out = (
                f"(({c}) ? {pick} : {out})"
                if isinstance(node.op, ast.Or)
                else f"(!({c}) ? {pick} : {out})"
            )
        return CE(out, t, True)

    def bool_op_lazy(self, node):
        """a or b / a and b with emitting operands: the value of the deciding
        operand, the later ones evaluated only when reached."""
        first = self.expr(node.values[0])
        t = first.t
        if t == T.BOOL or t == T.VOID:
            return None
        res = self.tmp("val")
        self.emit(f"{decl_type(t)} {res} = {first.code};")
        depth = 0
        for v in node.values[1:]:
            c = self.truth(CE(res, t), node)
            self.emit(f"if({'!' if isinstance(node.op, ast.Or) else ''}({c}))")
            self.emit("{")
            self.ind += 1
            depth += 1
            nxt = self.expr(v)
            if T.common(t, nxt.t) != t:
                raise Refuse(
                    node,
                    f"{'or' if isinstance(node.op, ast.Or) else 'and'} of {t} and {nxt.t}",
                )
            self.emit(f"{res} = {self.coerce(nxt, t, node)};")
        for _ in range(depth):
            self.ind -= 1
            self.emit("}")
        return CE(res, t)

    def ex_Compare(self, node, pre=None):
        if len(node.ops) != 1:
            # CE-R6: a <= x <= b evaluates x once and short-circuits; pure operands only
            vals = [self.expr(node.left)] + [self.expr(c) for c in node.comparators]
            if not all(v.pure for v in vals):
                raise Refuse(
                    node, "chained comparison with an emitting operand (CE-R6)"
                )
            terms = []
            for (a, b_), op in zip(zip(vals, vals[1:]), node.ops):
                terms.append(
                    self.ex_Compare(
                        ast.Compare(
                            left=ast.Name(id="__a"),
                            ops=[op],
                            comparators=[ast.Name(id="__b")],
                        ),
                        pre=(a, b_),
                    ).code
                )
            return CE(" && ".join(f"({t})" for t in terms), T.BOOL, True)
        op = node.ops[0]
        left = pre[0] if pre else self.expr(node.left)
        if isinstance(op, (ast.In, ast.NotIn)):
            rn = node.comparators[0]
            if isinstance(rn, (ast.Tuple, ast.List, ast.Set)) and all(
                isinstance(e, ast.Constant) for e in rn.elts
            ):
                terms = [
                    self.eq_code(
                        left,
                        CE(lit_code(e.value), lit_type(e.value), lit=e.value),
                        node,
                    )
                    for e in rn.elts
                ]
                code = (
                    " || ".join(terms)
                    if len(terms) > 1
                    else (terms[0] if terms else "false")
                )
                if isinstance(op, ast.NotIn):
                    code = f"!({code})" if len(terms) > 1 else negate(code)
                return CE(code, T.BOOL, left.pure)
            right = self.readable(self.expr(node.comparators[0]))
            if right.t == "pydict":
                keys = tuple(right.lit.keys())
                right = CE("", "pytuple", lit=keys)
            if isinstance(right.lit, (tuple, frozenset)):
                items = (
                    sorted(right.lit) if isinstance(right.lit, frozenset) else right.lit
                )
                if not items:
                    code = "false"
                else:
                    terms = []
                    for it in items:
                        terms.append(
                            self.eq_code(
                                left, CE(lit_code(it), lit_type(it), lit=it), node
                            )
                        )
                    code = " || ".join(terms) if len(terms) > 1 else terms[0]
                if isinstance(op, ast.NotIn):
                    code = f"!({code})" if len(items) > 1 else negate(code)
                return CE(code, T.BOOL, left.pure)
            if (
                self.record_of(right) is not None
                and left.lit is not None
                and isinstance(left.lit, str)
            ):
                f = record_field(left.lit)
                sep = "->" if right.t.endswith("*") else "."
                has = (
                    f"{right.code}{sep}has_{f}"
                    if left.lit in self.mod.records.get(self.record_of(right), {})
                    else "false"
                )
                return CE(
                    negate(has) if isinstance(op, ast.NotIn) else has,
                    T.BOOL,
                    right.pure,
                )
            dm = re.fullmatch(r"(?:const )?py::dict<(.+)>&?", right.t)
            if dm:
                kt_ = (
                    T._split_top(dm.group(1).replace("<", "[").replace(">", "]"))[0]
                    .strip()
                    .replace("[", "<")
                    .replace("]", ">")
                )
                code = f"{paren(right.code)}.has({self.coerce(left, kt_, node)})"
                return CE(
                    negate(code) if isinstance(op, ast.NotIn) else code,
                    T.BOOL,
                    left.pure and right.pure,
                )
            if T.seq_elem(right.t) is not None:
                ts = self.tuple_struct_items(left)
                rel = T.seq_elem(right.t)
                if ts is not None and T.seq_elem(rel) is not None:
                    # a tuple struct against a list of arrays / tuples: compare as the element type
                    left = CE(
                        f"{decl_type(rel)}{{{', '.join(self.coerce(CE(c, t), T.seq_elem(rel) if T.arr_len(rel) else t, node) for c, t in ts)}}}",
                        rel,
                        left.pure,
                    )
                code = f"py::contains({right.code}, {left.code})"
                return CE(
                    negate(code) if isinstance(op, ast.NotIn) else code,
                    T.BOOL,
                    left.pure and right.pure,
                )
            raise Refuse(node, f"`in` over {right.t}")
        right = pre[1] if pre else self.expr(node.comparators[0])
        if isinstance(op, (ast.Is, ast.IsNot)):
            if right.t == T.NONE or left.t == T.NONE:
                x = left if right.t == T.NONE else right
                if x.t in ("std::nullptr_t", T.NONE):
                    code = "true"  # only ever None
                elif (
                    (T.is_seq(x.t) or x.t.startswith("py::dict<"))
                    and not T.optional_inner(x.t)
                    and not (x.meta and "has" in x.meta)
                ):
                    # a list / dict type never holds None (a None-able one is an optional
                    # or a pointer); scalars are left out: a has_ field read into a
                    # local loses its flag, so `is None` there must stay a hole
                    code = "false"
                elif x.meta and "has" in x.meta:
                    code = f"!{paren(x.meta['has'])}"
                elif T.optional_inner(x.t):
                    code = f"!{paren(x.code)}.has_value()"
                elif T.is_ptr(x.t) or x.t == T.STR:
                    code = f"{paren(x.code)} == nullptr"
                elif self._never_none_param(
                    node.comparators[0] if x is right else node.left, x
                ):
                    # N-R4: a scalar parameter the function never rebinds holds
                    # exactly its C++ type (recorded or annotated without None),
                    # so `is None` folds like a static isinstance
                    lit = isinstance(op, ast.IsNot)
                    return CE("true" if lit else "false", T.BOOL, lit=lit)
                else:
                    raise Refuse(node, f"`is None` on {x.t} (N-R1)")
            elif T.is_ptr(left.t) and T.is_ptr(right.t):
                code = f"{paren(left.code)} == {paren(right.code)}"
            else:
                raise Refuse(node, f"`is` on {left.t}, {right.t} (R-HS3)")
            if isinstance(op, ast.IsNot):
                code = negate(code)
            return CE(code, T.BOOL, left.pure and right.pure)
        if isinstance(op, (ast.Eq, ast.NotEq)):
            code = self.eq_code(left, right, node)
            if isinstance(op, ast.NotEq):
                code = negate(code)
            return CE(code, T.BOOL, left.pure and right.pure)
        sym = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}[type(op)]
        if (T.is_int(left.t) or left.t == T.F64) and (
            T.is_int(right.t) or right.t == T.F64
        ):
            return CE(
                f"{paren(left.code)} {sym} {paren(right.code)}",
                T.BOOL,
                left.pure and right.pure,
            )
        raise Refuse(node, f"ordering compare on {left.t}, {right.t}")

    def eq_code(self, a: CE, b: CE, node) -> str:
        for x, y in ((a, b), (b, a)):
            ox = T.optional_inner(x.t)
            if ox and not T.optional_inner(y.t) and y.t != T.NONE:
                # an optional against a value: equal when present and equal
                return f"({paren(x.code)}.has_value() && {self.eq_code(CE(f'(*{paren(x.code)})', ox), y, node)})"
        ax, bx = self.readable(a), self.readable(b)
        ea, eb = T.seq_elem(ax.t), T.seq_elem(bx.t)
        if (
            ea
            and eb
            and ea != eb
            and (
                self.tuple_struct_items(CE("x", ea))
                or self.tuple_struct_items(CE("y", eb))
            )
        ):
            # lists whose items differ in spelling (tuple struct vs std::array): item by item
            xi = CE("x_", ea)
            yi = CE("y_", eb)
            item = self.eq_code(xi, yi, node)
            return f"py::seq_eq({ax.code}, {bx.code}, [&]({decl_type(ea)} const& x_, {decl_type(eb)} const& y_) {{ return {item}; }})"
        if (T.is_int(a.t) and T.is_int(b.t)) or (a.t == T.BOOL and b.t == T.BOOL):
            return f"{paren(a.code)} == {paren(b.code)}"
        for x, y in ((a, b), (b, a)):
            if y.t == T.STR and y.lit is not None and x.t in self.env.headers.enums:
                e = self.env.enum_for_string(x.t, y.lit)
                if e is None:
                    raise Refuse(node, f"no enumerator of {x.t} for {y.lit!r} (S2)")
                return f"{paren(x.code)} == {e}"
        if T.is_str(a.t) and T.is_str(b.t):
            if a.t == T.SSTR:
                return f"py::eq({a.code}, {b.code})"
            return (
                f"py::eq({a.code}, {b.code if b.t == T.STR else b.code + '.c_str()'})"
            )
        if a.t == T.TYPE and b.t == T.TYPE:
            return f"rocke_type_eq({a.code}, {b.code})"
        if a.t.startswith(("py::olist<", "py::span<")) or b.t.startswith(
            ("py::olist<", "py::span<")
        ):
            ra, rb = self.readable(a), self.readable(b)
            return f"py::seq_eq({ra.code}, {rb.code}, [](const auto& x_, const auto& y_) {{ return x_ == y_; }})"
        if T.is_seq(a.t) or T.is_seq(b.t):
            t = (
                a.t
                if T.is_seq(a.t) and not a.code.startswith("std::array<auto")
                else b.t
            )
            return (
                f"{paren(self.coerce(a, t, node))} == {paren(self.coerce(b, t, node))}"
            )
        sa, sb = T.struct_of(a.t), T.struct_of(b.t)
        if (
            sa
            and sb
            and sa[0] == sb[0]
            and sa[0] in self.env.headers.structs
            and sa[0] != T.IV
        ):
            # dataclass ==: field by field (Python compares the field tuples)
            fn = self.struct_eq_fn(sa[0], node)
            pa = a.code if sa[1] else f"&{paren(a.code)}"
            pb = b.code if sb[1] else f"&{paren(b.code)}"
            return f"{fn}({pa}, {pb})"
        if a.t == T.IV and b.t == T.IV:
            return f"py_adapt::iv_eq({a.code}, {b.code})"
        for x, y in ((a, b), (b, a)):
            xs = self.tuple_struct_items(x)
            yn = (
                T.arr_len(y.t)
                if T.arr_len(y.t) is not None
                else (len(T.tuple_items(y.t)) if T.tuple_items(y.t) else None)
            )
            if xs is not None and yn == len(xs):
                yt = (
                    [T.seq_elem(y.t)] * yn
                    if T.arr_len(y.t) is not None
                    else T.tuple_items(y.t)
                )
                return (
                    "("
                    + " && ".join(
                        self.eq_code(
                            CE(c, t), CE(f"std::get<{i}>({y.code})", yt[i]), node
                        )
                        for i, (c, t) in enumerate(xs)
                    )
                    + ")"
                )
        ia, ib = T.tuple_items(a.t), T.tuple_items(b.t)
        if ia and ib and len(ia) == len(ib):
            # tuples: item by item (Python compares them that way; widths may differ)
            return (
                "("
                + " && ".join(
                    self.eq_code(
                        CE(f"std::get<{i}>({a.code})", x),
                        CE(f"std::get<{i}>({b.code})", y),
                        node,
                    )
                    for i, (x, y) in enumerate(zip(ia, ib))
                )
                + ")"
            )
        if T.is_ptr(a.t) and T.is_ptr(b.t):
            raise Refuse(node, f"== on pointers {a.t} (R-HS3)")
        raise Refuse(node, f"== on {a.t}, {b.t}")

    def struct_eq_fn(self, sname: str, node, seen=None) -> str:
        """py_eq_<struct>(a, b): the dataclass == of a C struct, field by field."""
        fn = f"py_eq_{sname}"
        key = "fn:" + fn
        if key in self.mod.private_consts or (seen and sname in seen):
            return fn
        seen = (seen or set()) | {sname}
        s_ = self.env.headers.structs[sname]
        terms = []
        owned = {f for f in s_.fields if "n_" + f in s_.fields}
        skip = {"n_" + f for f in owned} | {"cap_" + f for f in owned}
        for f in s_.order:
            if f in skip:
                continue
            t = s_.fields[f]
            if f in owned:
                terms.append(
                    f"a->n_{f} == b->n_{f} && std::equal(a->{f}, a->{f} + a->n_{f}, b->{f})"
                )
            elif t == T.STR:
                terms.append(f"py::eq(a->{f}, b->{f})")
            elif t == T.TYPE:
                terms.append(f"rocke_type_eq(a->{f}, b->{f})")
            elif t == T.IV:
                terms.append(f"py_adapt::iv_eq(a->{f}, b->{f})")
            elif (
                T.struct_of(t)
                and T.struct_of(t)[0] in self.env.headers.structs
                and not t.endswith("[]")
            ):
                inner = T.struct_of(t)
                sub = self.struct_eq_fn(inner[0], node, seen)
                terms.append(
                    f"{sub}({'' if inner[1] else '&'}a->{f}, {'' if inner[1] else '&'}b->{f})"
                    if not inner[1]
                    else f"(a->{f} == b->{f} || (a->{f} && b->{f} && {sub}(a->{f}, b->{f})))"
                )
            elif t.endswith("[]"):
                raise Refuse(node, f"== of {sname}: array field {f}")
            else:
                terms.append(f"a->{f} == b->{f}")
        body = [
            f"static bool {fn}(const {sname}* a, const {sname}* b)",
            "{",
            f"    return {' && '.join(terms) or 'true'};",
            "}",
        ]
        self.mod.private_consts[key] = (fn, "\n".join(body))
        return fn

    def arm(self, narrowing: dict, node):
        self.narrow.append(narrowing)
        try:
            return self.expr(node)
        finally:
            self.narrow.pop()

    def ex_IfExp(self, node):
        c = self.expr(node.test)
        pos, neg = self._narrowing_from(node.test), self._negated_narrowing(node.test)
        mark = len(self.lines)
        a = self.arm(pos, node.body)
        b = self.arm(neg, node.orelse)
        if len(self.lines) == mark and a.pure and b.pure and c.pure:
            t = T.common(a.t, b.t)
            if t is None:
                raise Refuse(node, f"ternary of {a.t} and {b.t}")
            return CE(
                f"(({self.truth(c, node)}) ? {self.coerce(a, t, node)} : {self.coerce(b, t, node)})",
                t,
            )
        del self.lines[mark:]
        key = (node.lineno, node.col_offset, "ifexp")
        t = self.hints["expr"].get(key) or self.common(a.t, b.t)
        if t is None:
            raise Refuse(node, "ternary type")
        res = self.tmp("sel")
        self.emit(f"{decl_type(t)} {res}{{}};")
        self.emit(f"if({self.truth(c, node)})")
        self.emit("{")
        self.ind += 1
        a = self.arm(pos, node.body)
        self.emit(f"{res} = {self.coerce(a, t, node)};")
        self.ind -= 1
        self.emit("}")
        self.emit("else")
        self.emit("{")
        self.ind += 1
        b = self.arm(neg, node.orelse)
        self.emit(f"{res} = {self.coerce(b, t, node)};")
        self.ind -= 1
        self.emit("}")
        if self.dry:
            self.hints["expr"][key] = t
        return CE(res, t)

    def ex_JoinedStr(self, node):
        return CE(self.fstring(node), T.SSTR, False)

    def str_expr(self, node) -> str:
        if isinstance(node, ast.JoinedStr):
            return self.fstring(node)
        ce = self.expr(node)
        if ce.lit is not None and isinstance(ce.lit, str):
            return f"std::string({ce.code})"
        if T.is_str(ce.t):
            return f"py::str({ce.code})"
        raise Refuse(node, f"message of type {ce.t}")

    def fstring(self, node: ast.JoinedStr) -> str:
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(c_string(v.value))
                continue
            if v.format_spec is not None:
                spec = v.format_spec
                if (
                    isinstance(spec, ast.JoinedStr)
                    and len(spec.values) == 1
                    and isinstance(spec.values[0], ast.Constant)
                    and spec.values[0].value in ("#x", "x", "d")
                ):
                    ce_ = self.expr(v.value)
                    if T.is_int(ce_.t) or ce_.t == T.BOOL:
                        fmt = spec.values[0].value
                        parts.append(
                            {
                                "#x": f"py::hex({ce_.code}, true)",
                                "x": f"py::hex({ce_.code}, false)",
                                "d": f"py::str((int64_t){paren(ce_.code)})",
                            }[fmt]
                        )
                        continue
                raise Refuse(v, "format spec in an f-string (RT-R2)")
            ce = self.expr(v.value)
            if ce.lit is not None:
                s = repr(ce.lit) if v.conversion == ord("r") else str(ce.lit)
                parts.append(c_string(s))
                continue
            fn = "py::repr" if v.conversion == ord("r") else "py::str"
            ts = self.tuple_struct_items(ce)
            if ts is not None:
                # str of a tuple: "(" repr items ", " ")", like Python
                inner = ' + ", " + '.join(f"py::repr({c})" for c, _t in ts)
                tail = "," if len(ts) == 1 else ""
                parts.append(f'(std::string("(") + {inner} + std::string("{tail})"))')
                continue
            parts.append(f"{fn}({ce.code})")
        merged = []
        for p in parts:
            if (
                merged
                and p.startswith('"')
                and merged[-1].startswith('"')
                and not p.startswith("py::")
            ):
                merged[-1] = merged[-1][:-1] + p[1:]
            else:
                merged.append(p)
        if not merged:
            return 'std::string("")'
        out = f"std::string({merged[0]})" if merged[0].startswith('"') else merged[0]
        if (
            not merged[0].startswith('"')
            and len(merged) > 1
            and merged[1].startswith('"')
        ):
            out = f"{merged[0]} + std::string({merged[1]})"
            merged = merged[1:]
        for p in merged[1:]:
            out += f" + {p}"
        return out

    # ------------------------------------------------------- comprehensions

    def ex_ListComp(self, node):
        key = (node.lineno, node.col_offset, "comp")
        el = self.hints["expr"].get(key)
        if el is None and not self.dry:
            raise Refuse(node, "comprehension with unknown element type")
        name = self.tmp("lst")
        self.emit(f"{decl_type(T.vec(el or T.AUTO))} {name};")
        got = self.comp_into(node, name, el)
        if self.dry and got:
            self.hints["expr"][key] = got
        return CE(name, T.vec(el or got or T.AUTO))

    def ex_SetComp(self, node):
        """{x for ...} of ints: an ascending set (CPython's order for small ints)."""
        key = (node.lineno, node.col_offset, "comp")
        el = self.hints["expr"].get(key)
        if el is None and not self.dry:
            raise Refuse(node, "set comprehension with unknown element type")
        if el is not None and not T.is_int(el):
            raise Refuse(
                node, f"set of {el}: Python's iteration order is not reproducible"
            )
        name = self.tmp("set")
        self.emit(f"py::set<int64_t> {name};")
        lc = ast.copy_location(
            ast.ListComp(elt=node.elt, generators=node.generators), node
        )
        got = self.comp_into(lc, name + "_add", el) if False else None
        lst = self.tmp("lst")
        self.emit(f"std::vector<int64_t> {lst};")
        got = self.comp_into(lc, lst, T.I64 if el else None)
        if self.dry and got:
            self.hints["expr"][key] = got
        self.emit(f"for(int64_t x_ : {lst}) {name}.add(x_);")
        return CE(name, "py::set<int64_t>")

    def ex_DictComp(self, node):
        """CE-R1 for {k: v for ...}: an ordered py::dict filled in source order
        (a repeated key keeps its first position, as in Python)."""
        key = (node.lineno, node.col_offset, "comp")
        kv = self.hints["expr"].get(key)
        if kv is None and not self.dry:
            raise Refuse(node, "dict comprehension with unknown key or value type")
        name = self.tmp("dct")
        t = f"py::dict<{kv[0]}, {kv[1]}>" if kv else "py::dict<auto, auto>"
        self.emit(f"{t} {name};")
        got = self.comp_into(node, name, kv)
        if self.dry and got:
            self.hints["expr"][key] = got
            t = f"py::dict<{got[0]}, {got[1]}>"
        return CE(name, t)

    def comp_into(self, node, name, el) -> str | None:
        """CE-R1: every comprehension is an eager loop in source order."""
        if any(g.is_async for g in node.generators):
            raise Refuse(node, "async comprehension")
        saved = dict(self.vars)
        saved_decl = set(self.declared)
        out_t = [None]

        def level(i):
            if i == len(node.generators):
                if isinstance(node, ast.DictComp):
                    k = self.bind(self.expr(node.key), "k")
                    v = self.bind(self.expr(node.value), "e")
                    out_t[0] = (k.t, v.t)
                    kc = self.coerce(k, el[0], node) if el else k.code
                    vc = self.coerce(v, el[1], node) if el else v.code
                    self.emit(f"{name}.set({kc}, {vc});")
                    return
                v = self.bind(self.expr(node.elt), "e")
                out_t[0] = v.t
                self.emit(
                    f"{name}.push_back({self.coerce(v, el, node) if el else v.code});"
                )
                return
            g = node.generators[i]

            def body():
                for cond in g.ifs:
                    self.emit(f"if(!({self.test(cond)})) continue;")
                level(i + 1)

            self.comp_loop(g, body, node)

        level(0)
        self.vars = saved
        self.declared = saved_decl
        return out_t[0]

    def for_indexed(self, target, it, body, node, scoped_body=False):
        """enumerate(xs) / zip(a, b, ...) as an index loop (zip stops at the shortest)."""
        if it.keywords or not isinstance(target, ast.Tuple):
            raise Refuse(node, f"{it.func.id}() loop shape")
        seqs = [self.bind(self.expr(a), "seq") for a in it.args]
        for v in seqs:
            if T.seq_elem(v.t) is None:
                raise Refuse(node, f"{it.func.id}() over {v.t}")
        i = self.tmp("i")
        if it.func.id == "enumerate":
            if len(seqs) != 1 or len(target.elts) != 2:
                raise Refuse(node, "enumerate() shape")
            n = f"py::len({seqs[0].code})"
            items = [
                (target.elts[0], CE(i, T.I64)),
                (
                    target.elts[1],
                    CE(f"{seqs[0].code}[(size_t){i}]", T.seq_elem(seqs[0].t)),
                ),
            ]
        else:
            if len(seqs) != len(target.elts):
                raise Refuse(node, "zip() arity")
            n = " , ".join(f"py::len({v.code})" for v in seqs)
            n = f"std::min<int64_t>({{{n}}})" if len(seqs) > 1 else n
            items = [
                (t_, CE(f"{v.code}[(size_t){i}]", T.seq_elem(v.t)))
                for t_, v in zip(target.elts, seqs)
            ]
        self.emit(f"for(int64_t {i} = 0; {i} < {n}; ++{i})")
        self.emit("{")
        self.ind += 1
        saved = set(self.declared)
        for t_, v in items:
            if not isinstance(t_, ast.Name):
                self.bind_loop_target(
                    t_, v, node
                )  # enumerate(...) over pairs: i, (a, b)
                continue
            self.vars[t_.id] = v.t
            if t_.id in self.declared and not scoped_body:
                self.emit(f"{t_.id} = {v.code};")
            else:
                self.emit(f"{self.unused(t_.id)}{decl_type(v.t)} {t_.id} = {v.code};")
                self.declared.add(t_.id)
        body()
        self.declared = saved
        self.ind -= 1
        self.emit("}")

    def _unroll_names(self, g, node):
        """A loop over a constant tuple of str whose target names an attribute
        (getattr(x, target)): unrolled, the target a literal each time."""
        if not isinstance(g.target, ast.Name):
            return None
        uses = [
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "getattr"
            and len(n.args) >= 2
            and isinstance(n.args[1], ast.Name)
            and n.args[1].id == g.target.id
        ]
        if not uses:
            return None
        try:
            src = self.expr(g.iter)
        except Refuse:
            return None
        vals = src.lit if isinstance(src.lit, tuple) else None
        if vals is None or not all(isinstance(v, str) for v in vals):
            return None
        return vals

    def comp_loop(self, g, body, node):
        it = g.iter
        vals = self._unroll_names(g, node)
        if vals is not None:
            for v in vals:
                self.emit("{")
                self.ind += 1
                self.narrow.append({g.target.id: CE(lit_code(v), T.STR, lit=v)})
                try:
                    body()
                finally:
                    self.narrow.pop()
                self.ind -= 1
                self.emit("}")
            return
        if (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id in ("enumerate", "zip")
            and self.lookup(it.func.id) is None
        ):
            self.for_indexed(g.target, it, body, node, scoped_body=True)
            return
        if (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == "range"
        ):
            args = [
                self.coerce(self.bind(self.expr(a), "r"), T.I64, node) for a in it.args
            ]
            if not isinstance(g.target, ast.Name):
                raise Refuse(node, "comprehension target")
            self.vars[g.target.id] = T.I64
            self.declared.add(g.target.id)
            self.emit(f"for(int64_t {g.target.id} : py::range({', '.join(args)}))")
            self.emit("{")
            self.ind += 1
            body()
            self.ind -= 1
            self.emit("}")
            return
        if (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Attribute)
            and it.func.attr == "items"
            and not it.args
        ):
            d = self.bind(self.expr(it.func.value), "d")
            m = re.fullmatch(r"(?:const )?py::dict<(.+)>&?", d.t)
            if not m:
                raise Refuse(node, f"items() of {d.t} (D4)")
            kt, vt = [
                x.strip()
                for x in T._split_top(m.group(1).replace("<", "[").replace(">", "]"))
            ]
            kt, vt = kt.replace("[", "<").replace("]", ">"), vt.replace(
                "[", "<"
            ).replace("]", ">")
            kv = self.tmp("kv")
            self.emit(f"for(const auto& {kv} : {d.code}.items())")
            self.emit("{")
            self.ind += 1
            self.bind_loop_target(
                g.target,
                CE(
                    f"std::make_pair({kv}.first, {kv}.second)", f"std::pair<{kt}, {vt}>"
                ),
                node,
                parts=[(f"{kv}.first", kt), (f"{kv}.second", vt)],
            )
            body()
            self.ind -= 1
            self.emit("}")
            return
        src = self.bind(self.readable(self.expr(it)), "it")
        el = T.seq_elem(src.t)
        if el is None:
            raise Refuse(node, f"comprehension over {src.t}")
        if not isinstance(g.target, ast.Name):
            # (a, b) over a sequence of pairs / tuple structs
            item = self.tmp("item")
            self.emit(f"for({decl_type(el)} const& {item} : {src.code})")
            self.emit("{")
            self.ind += 1
            self.bind_loop_target(g.target, CE(item, el), node)
            body()
            self.ind -= 1
            self.emit("}")
            return
        self.vars[g.target.id] = el
        self.declared.add(g.target.id)
        self.emit(f"for({decl_type(el)} const& {g.target.id} : {src.code})")
        self.emit("{")
        self.ind += 1
        body()
        self.ind -= 1
        self.emit("}")

    def bind_loop_target(self, target, whole: CE, node, parts=None):
        """A loop or comprehension target: a name, or a (nested) tuple of names
        over a pair, std::array or std::tuple."""
        if isinstance(target, ast.Name):
            self.vars[target.id] = whole.t
            self.declared.add(target.id)
            self.emit(
                f"{self.unused(target.id)}{decl_type(whole.t)} const {target.id} = {whole.code};"
            )
            return
        if not isinstance(target, ast.Tuple):
            raise Refuse(node, "loop target")
        if parts is None:
            n = T.arr_len(whole.t)
            items = T.tuple_items(whole.t)
            if n is not None:
                parts = [(f"{whole.code}[{i}]", T.seq_elem(whole.t)) for i in range(n)]
            elif self.tuple_struct_items(whole) is not None:
                parts = self.tuple_struct_items(whole)
            elif items is not None:
                parts = [
                    (f"std::get<{i}>({whole.code})", t) for i, t in enumerate(items)
                ]
            else:
                raise Refuse(node, f"unpack of {whole.t}")
        if len(parts) != len(target.elts):
            raise Refuse(node, "unpack arity")
        for t_, (code, t) in zip(target.elts, parts):
            self.bind_loop_target(t_, CE(code, t), node)

    def any_all(self, node, is_any: bool) -> CE:
        """CE-R5: any/all over a generator is a loop with an early exit. Several
        ``for`` clauses (CE-R5b) are nested loops, outermost first, each with its
        own ``if`` filters; the exit leaves all of them (a goto past the loops)."""
        gen = node.args[0]
        if not isinstance(gen, ast.GeneratorExp):
            raise Refuse(node, "any/all over a non-generator")
        res = self.tmp("any" if is_any else "all")
        self.emit(f"bool {res} = {'false' if is_any else 'true'};")
        nested = len(gen.generators) > 1
        done = self.tmp("done") if nested else None
        saved = dict(self.vars)

        def level(i):
            g = gen.generators[i]

            def body():
                for cond in g.ifs:
                    self.emit(f"if(!({self.test(cond)})) continue;")
                if i + 1 < len(gen.generators):
                    self.emit("{")
                    self.ind += 1
                    level(i + 1)
                    self.ind -= 1
                    self.emit("}")
                    return
                t = self.test(gen.elt)
                if is_any:
                    self.emit(f"if({t})")
                else:
                    self.emit(f"if(!({t}))")
                self.emit("{")
                self.ind += 1
                self.emit(f"{res} = {'true' if is_any else 'false'};")
                self.emit(f"goto {done};" if nested else "break;")
                self.ind -= 1
                self.emit("}")

            self.comp_loop(g, body, node)

        if nested:
            # the loops sit in their own block so the goto skips no declaration
            self.emit("{")
            self.ind += 1
            level(0)
            self.ind -= 1
            self.emit("}")
            self.emit(f"{done}:;")
        else:
            level(0)
        self.vars = saved
        return CE(res, T.BOOL)

    # ---------------------------------------------------------------- calls

    def ex_Call(self, node, want=None):
        f = node.func
        if isinstance(f, ast.Subscript) and isinstance(f.value, ast.Dict):
            return self.dict_display_call(node)
        if (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Call)
            and isinstance(f.value.func, ast.Name)
            and f.value.func.id == "super"
            and not f.value.args
        ):
            return self.call_super(f.attr, node)
        if (
            isinstance(f, ast.Attribute)
            and f.attr == "fget"
            and isinstance(f.value, ast.Attribute)
            and isinstance(f.value.value, ast.Name)
            and f.value.value.id in self.mod.classes
            and self.lookup(f.value.value.id) is None
            and len(node.args) == 1
            and not node.keywords
        ):
            # Base.prop.fget(obj): the base's implementation, no dispatch (tagged values)
            return self.call_class_method(
                f.value.value.id,
                f.value.attr,
                node.args[0],
                ast.Call(func=f.value, args=[], keywords=[]),
                node,
            )
        if (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id in self.mod.classes
            and self.lookup(f.value.id) is None
            and (f.value.id, f.attr) in self.mod.methods
            and node.args
            and self.mod.family(f.value.id) is not None
        ):
            # Base.meth(obj, ...): the named class's implementation, no dispatch (tagged values)
            rest = ast.Call(func=f, args=node.args[1:], keywords=node.keywords)
            return self.call_class_method(f.value.id, f.attr, node.args[0], rest, node)
        if isinstance(f, ast.Call):
            # f(...)(...): call the callable the inner call returns
            inner = self.bind(self.expr(f), "fn")
            if inner.t in ("std::nullptr_t", T.NONE):
                # only ever None: calling it is Python's TypeError
                return CE(
                    f"((void){inner.code}, throw ckc::TypeError(\"'NoneType' object is not callable\"))",
                    T.VOID,
                    False,
                )
            cs = self.call_callable_struct(inner, node)
            if cs is None:
                raise Refuse(node, f"call of a {inner.t}")
            return cs
        if any(k.arg is None for k in node.keywords) and not (
            isinstance(f, ast.Attribute) and f.attr == "param"
        ):
            raise Refuse(node, "** arguments (KW3)")
        # builder methods
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            recv = self.lookup(f.value.id)
            if recv is not None and recv.t == T.B:
                return self.call_ir(f.attr, recv, node)
        if (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id == "ArchTarget"
            and f.attr == "from_gfx"
            and self.lookup("ArchTarget") is None
        ):
            arg = self.bind(self.expr(node.args[0]), "gfx")
            return CE(
                f"py_adapt::arch_target({self.coerce(arg, T.STR, node)})",
                "const rocke_arch_target_t*",
                False,
            )
        if isinstance(f, ast.Attribute):
            if f.attr == "append" and isinstance(f.value, ast.Name):
                return self.call_append(f.value.id, node)
            if (
                f.attr == "append"
                and isinstance(f.value, ast.Subscript)
                and len(node.args) == 1
            ):
                # xs[i].append(v): the item is an lvalue (py::at returns a reference)
                lst = self.expr(f.value)
                el = T.seq_elem(lst.t)
                if el is None or not lst.t.startswith("std::vector<"):
                    raise Refuse(node, f"append to {lst.t}")
                v = self.bind(self.expr(node.args[0]), "e")
                return CE(
                    f"{lst.code}.push_back({self.coerce(v, el, node)})", T.VOID, False
                )
            if f.attr == "append" and isinstance(f.value, ast.Attribute):
                lst = self.expr(f.value)
                if lst.t.startswith("py::olist<"):
                    v = self.bind(self.expr(node.args[0]), "e")
                    m = lst.meta
                    return CE(
                        f"py::opush({m['p']}, {m['n']}, {m['cap']}, {v.code})",
                        T.VOID,
                        False,
                    )
                raise Refuse(node, f"append to a {lst.t} field")
            if (
                isinstance(f.value, ast.Name)
                and f.value.id == "dataclasses"
                and f.attr == "replace"
                and self.lookup("dataclasses") is None
            ):
                return self.call_replace(node)
            if (
                isinstance(f.value, ast.Name)
                and self.lookup(f.value.id) is None
                and f.value.id in self.mod.classes
            ):
                raise Refuse(node, f"class-level call {ast.unparse(f)}")
            recv = self.expr(f.value)
            if T.optional_inner(recv.t):
                # a method of an optional: Python fails on None the same way
                recv = CE(
                    f"py::deref({recv.code})", T.optional_inner(recv.t), recv.pure
                )
            if recv.t == T.B:
                # a builder held in a field (self.b.cmp_lt)
                return self.call_ir(f.attr, self.bind(recv, "b"), node)
            cm = self.call_struct(recv, f.attr, node)
            if cm is not None:
                return cm
            from .adapters import method_adapter

            obj = recv
            if f.attr == "join" and T.is_str(obj.t) and len(node.args) == 1:
                seq = self.bind(self.expr(node.args[0]), "parts")
                if T.seq_elem(seq.t) is None or not T.is_str(T.seq_elem(seq.t)):
                    raise Refuse(node, f"join over {seq.t}")
                return CE(f"py::join({obj.code}, {seq.code})", T.SSTR, False)
            if (
                obj.t == "pydict"
                and f.attr == "get"
                and 1 <= len(node.args) <= 2
                and not node.keywords
            ):
                return self.const_dict_get(obj, node)
            if (
                (obj.t == "set?" or obj.t.startswith("py::set<"))
                and f.attr in ("add", "discard")
                and len(node.args) == 1
                and isinstance(f.value, ast.Name)
            ):
                v_ = self.bind(self.expr(node.args[0]), "e")
                if obj.t == "set?":
                    st_t = f"py::set<{T.I64 if T.is_int(v_.t) else v_.t}>"
                    if not self.dry:
                        raise Refuse(node, "set typed too late")
                    self.hints["vars"][f.value.id] = st_t
                    self.vars[f.value.id] = st_t
                    return CE("", T.VOID)
                el_ = obj.t[len("py::set<") : -1]
                return CE(
                    f"{obj.code}.{f.attr}({self.coerce(v_, el_, node)})", T.VOID, False
                )
            if (
                f.attr == "index"
                and len(node.args) == 1
                and not node.keywords
                and T.seq_elem(self.readable(obj).t) is not None
            ):
                seq_ = self.bind(self.readable(obj), "seq")
                x_ = self.bind(self.expr(node.args[0]), "x")
                return CE(
                    f"py::index_of({seq_.code}, {self.coerce(x_, T.seq_elem(seq_.t), node)})",
                    T.I64,
                    False,
                )
            if (
                obj.t.startswith("std::vector<")
                and f.attr == "pop"
                and len(node.args) <= 1
                and not node.keywords
            ):
                el_ = T.seq_elem(obj.t)
                i_ = (
                    self.coerce(self.bind(self.expr(node.args[0]), "i"), T.I64, node)
                    if node.args
                    else "-1"
                )
                return CE(f"py::pop({obj.code}, {i_})", el_, False)
            if obj.t.startswith("py::set<") and f.attr == "pop" and not node.args:
                return CE(f"{obj.code}.pop()", obj.t[len("py::set<") : -1], False)
            if (
                self.record_of(obj) is not None
                and f.attr == "get"
                and 1 <= len(node.args) <= 2
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                dflt = (
                    self.bind(self.expr(node.args[1]), "dflt")
                    if len(node.args) == 2
                    else None
                )
                return self.record_get(
                    obj, node.args[0].value, node, default=dflt, raising=False
                )
            if (
                T.is_int(obj.t)
                and f.attr == "bit_length"
                and not node.args
                and not node.keywords
            ):
                return CE(f"py::bit_length({obj.code})", T.I64, obj.pure)
            if (
                T.is_str(obj.t)
                and f.attr in ("strip", "lower", "upper")
                and not node.args
                and not node.keywords
            ):
                return CE(f"py::{f.attr}({obj.code})", T.SSTR, obj.pure)
            if (
                T.is_str(obj.t)
                and f.attr in ("split", "partition")
                and len(node.args) == 1
                and not node.keywords
            ):
                sep = self.bind(self.expr(node.args[0]), "sep")
                if not T.is_str(sep.t):
                    raise Refuse(node, f"{f.attr} by {sep.t}")
                if f.attr == "split":
                    return CE(
                        f"py::split({obj.code}, {sep.code})", T.vec(T.SSTR), False
                    )
                return CE(
                    f"py::partition({obj.code}, {sep.code})", T.arr(T.SSTR, 3), False
                )
            if (
                re.fullmatch(r"(?:const )?py::dict<.+>&?", obj.t)
                and f.attr == "items"
                and not node.args
            ):
                m_ = re.fullmatch(r"(?:const )?py::dict<(.+)>&?", obj.t).group(1)
                kt_, vt_ = [
                    x.strip().replace("[", "<").replace("]", ">")
                    for x in T._split_top(m_.replace("<", "[").replace(">", "]"))
                ]
                return CE(
                    f"py::items_list({obj.code})",
                    T.vec(f"std::tuple<{kt_}, {vt_}>"),
                    False,
                )
            if (
                re.fullmatch(r"(?:const )?py::dict<.+>&?", obj.t)
                and f.attr in ("keys", "values")
                and not node.args
            ):
                m_ = re.fullmatch(r"(?:const )?py::dict<(.+)>&?", obj.t).group(1)
                kt_, vt_ = [
                    x.strip().replace("[", "<").replace("]", ">")
                    for x in T._split_top(m_.replace("<", "[").replace(">", "]"))
                ]
                return CE(
                    f"{paren(obj.code)}.{f.attr}()",
                    T.vec(kt_ if f.attr == "keys" else vt_),
                    False,
                )
            if (
                re.fullmatch(r"py::dict<.+>", obj.t)
                and f.attr == "setdefault"
                and len(node.args) == 2
                and not node.keywords
            ):
                m_ = re.fullmatch(r"py::dict<(.+)>", obj.t).group(1)
                kt_, vt_ = [
                    x.strip().replace("[", "<").replace("]", ">")
                    for x in T._split_top(m_.replace("<", "[").replace(">", "]"))
                ]
                k_ = self.bind(self.expr(node.args[0]), "k")
                d_ = self.bind(self.expr(node.args[1]), "dflt")
                return CE(
                    f"{obj.code}.setdefault({self.coerce(k_, kt_, node)}, {self.coerce(d_, vt_, node)})",
                    vt_,
                    False,
                )
            if (
                re.fullmatch(r"(?:const )?py::dict<.+>&?", obj.t)
                and f.attr == "get"
                and len(node.args) == 2
                and not node.keywords
            ):
                m_ = re.fullmatch(r"(?:const )?py::dict<(.+)>&?", obj.t).group(1)
                vt = (
                    [
                        x.strip()
                        for x in T._split_top(m_.replace("<", "[").replace(">", "]"))
                    ][1]
                    .replace("[", "<")
                    .replace("]", ">")
                )
                k_ = self.bind(self.expr(node.args[0]), "k")
                d_ = self.bind(self.expr(node.args[1]), "dflt")
                return CE(
                    f"{paren(obj.code)}.get({k_.code}, {self.coerce(d_, vt, node)})",
                    vt,
                    obj.pure and k_.pure and d_.pure,
                )
            if (
                T.is_str(obj.t)
                and f.attr in ("startswith", "endswith")
                and len(node.args) == 1
                and not node.keywords
            ):
                arg = self.bind(self.expr(node.args[0]), "s")
                if not T.is_str(arg.t):
                    raise Refuse(node, f"{f.attr} of {arg.t}")
                return CE(
                    f"py::{f.attr}({obj.code}, {arg.code})",
                    T.BOOL,
                    obj.pure and arg.pure,
                )
            from .adapters import kw_method_adapter

            kad = kw_method_adapter(obj.t, f.attr)
            if kad is not None:
                tmpl, rt, order_, dflt = kad
                got = {}
                for p_, a_ in zip(order_, node.args):
                    got[p_] = self.bind(self.expr(a_), "a").code
                for k in node.keywords:
                    if k.arg not in order_:
                        raise Refuse(node, f"{f.attr}(): unknown keyword {k.arg}")
                    v = self.bind(self.expr(k.value), k.arg)
                    got[k.arg] = (
                        "-1" if v.t == T.NONE and dflt.get(k.arg) == "-1" else v.code
                    )
                missing = [p_ for p_ in order_ if p_ not in got and p_ not in dflt]
                if missing:
                    raise Refuse(node, f"{f.attr}(): missing {missing}")
                args_ = ", ".join(got.get(p_, dflt.get(p_)) for p_ in order_)
                o_ = obj.code if not T.is_ptr(obj.t) else f"(*{obj.code})"
                return self.bind(CE(tmpl.format(obj=o_, args=args_), rt, False), f.attr)
            ad = method_adapter(obj.t, f.attr)
            if ad is not None:
                args = [self.bind(self.expr(a), "a") for a in node.args]
                return CE(
                    ad[0].format(
                        obj=obj.code,
                        args=", ".join(a.code for a in args),
                        b=self.builder,
                    ),
                    ad[1],
                    False,
                )
            raise Refuse(node, f"method {f.attr} on {obj.t}")
        if isinstance(f, ast.Name):
            local = self.lookup(f.id)
            if local is not None and local.t.startswith("lambda:"):
                return self.call_lambda(f.id, node)
            if local is not None:
                cs = self.call_callable_struct(local, node)
                if cs is not None:
                    return cs
            owner = self.classmethod_owner()
            if local is None and owner is not None and f.id == owner[1]:
                return self.construct(owner[0], node)  # cls(...) in a classmethod
            if (
                local is None
                and not node.args
                and not node.keywords
                and f.id in host_builders(self.mod)
            ):
                return self.host_instance(
                    f.id
                )  # host clones: a host builder has no state
            if local is None and f.id in self.mod.classes:
                if f.id in self.mod.private_classes:
                    return self.construct_private(f.id, node)
                return self.construct(f.id, node)
            if (
                local is None
                and f.id in self.mod.imports
                and not (self.mod.tmap and f.id in self.mod.tmap.class_struct)
            ):
                pc = self.platform_class(f.id)
                if pc is not None:
                    return self.construct_platform(f.id, pc[0], pc[1], node)
            if (
                local is None
                and f.id in self.mod.imports
                and self.mod.tmap
                and f.id in self.mod.tmap.class_struct
            ):
                return self.construct_imported(f.id, node)
            if (
                local is None
                and f.id == "replace"
                and self.mod.imports.get("replace", ("", ""))[0] == "dataclasses"
            ):
                return self.call_replace(node)
            b = self.builtin(f.id, node, want)
            if b is not None:
                return b
            if f.id in self.mod.functions:
                return self.call_internal(f.id, node)
            if f.id in self.mod.imports:
                return self.call_helper(f.id, node)
            if f.id == "VectorType":
                pass
        raise Refuse(node, f"call {ast.unparse(f)}")

    def call_callable_struct(self, obj: CE, node, field: str = "fn"):
        """A Python callable or a duck object passed as a C struct of function
        pointers plus ctx: {fn, ctx}, or {method..., ctx} (a protocol)."""
        st = T.struct_of(obj.t)
        if st is None:
            return None
        s = self.env.headers.structs.get(st[0])
        if (
            s is None
            or "ctx" not in s.fields
            or field not in s.fields
            or (field == "fn" and set(s.fields) != {"fn", "ctx"})
        ):
            return None
        fp = self.env.headers.fnptrs.get(s.fields[field])
        if fp is None:
            raise Refuse(node, f"callable struct {st[0]} with an unknown fn type")
        ret, params = fp
        sep = "->" if st[1] else "."
        args = [self.bind(self.expr(a), "a") for a in node.args]
        cps = params[1:]
        if len(args) != len(cps) or node.keywords:
            raise Refuse(node, f"call shape of {st[0]}")
        cargs = [f"{obj.code}{sep}ctx"] + [
            self.coerce(a, t, node) for a, (t, _n) in zip(args, cps)
        ]
        return self.bind(
            CE(f"{obj.code}{sep}{field}({', '.join(cargs)})", ret, False), "call"
        )

    def call_class_method(self, cls: str, attr: str, obj_node, call, node) -> CE:
        """cls.attr(obj, ...) called through the class: that implementation."""
        sig = self.mod.sigs.get(f"{cls}.{attr}")
        fn = self.mod.methods.get((cls, attr))
        if sig is None or fn is None:
            raise Refuse(node, f"method {cls}.{attr} is not translated")
        recv = self.expr(obj_node)
        ast.copy_location(call, node)
        bound = self.bind_call_args(call, py_params(fn), skip=1)
        args = []
        for pname, t in sig.params:
            if pname == sig.params[0][0] and pname not in bound:
                args.append(self.coerce(recv, t, node))
            elif pname in bound:
                args.append(self.coerce(bound[pname], t, node))
            elif pname in ("fence", "seq_guards"):
                args.append(getattr(self, pname) or "nullptr")
            elif pname == "b" and t == T.B:
                args.append(self.builder)
            elif pname in sig.defaults:
                args.append(self.coerce(self.expr(sig.defaults[pname]), t, node))
            else:
                raise Refuse(node, f"missing argument {pname} to {cls}.{attr}")
        return self.bind(CE(f"{sig.name}({', '.join(args)})", sig.ret, False), attr)

    def static_class(self, recv: CE, cls: str) -> str:
        """The class a family value is known to be (tagged values): `self` inside a
        member's method is that member (or below); anything else the class its
        struct maps to (the family base)."""
        f = self
        while f.outer is not None:
            f = f.outer
        owner = next(
            (c for (c, _m), fn in self.mod.methods.items() if fn is f.node), None
        )
        args = f.node.args.posonlyargs + f.node.args.args
        if (
            owner is not None
            and args
            and recv.code == args[0].arg
            and self.lookup(args[0].arg) is not None
            and self.lookup(args[0].arg).code == recv.code
        ):
            fam = self.mod.family(owner)
            if fam is not None and cls in (fam.base, owner):
                return owner
        return cls

    def family_resolve(self, recv: CE, cls: str, attr: str, node):
        """tagged values: (class whose method to call, dispatcher Sig or None) for attr
        on a family value."""
        fam = self.mod.family(cls)
        if fam is None or attr in ("__init__", "__post_init__"):
            return cls, None
        scls = self.static_class(recv, cls)
        impls = fam.impls(scls, attr)
        owners = set(impls.values())
        if len(owners) > 1:
            try:
                return scls, self.mod.dispatcher(fam, scls, attr)
            except Refuse as e:
                raise Refuse(node, e.reason)
        if len(owners) == 1:
            return owners.pop(), None
        return scls, None

    def call_super(self, meth: str, node) -> CE:
        """super().meth(...) in a method: the base class's translated method,
        self passed as the base when the base's fields are a prefix of ours."""
        owner = next(
            ((c, m) for (c, m), f in self.mod.methods.items() if f is self.node), None
        )
        if owner is None:
            raise Refuse(node, "super() outside a method")
        cls = owner[0]
        cnode = self.mod.classes[cls]
        base = next(
            (
                ast.unparse(b)
                for b in cnode.bases
                if ast.unparse(b) in self.mod.classes
                and (ast.unparse(b), meth) in self.mod.methods
            ),
            None,
        )
        if base is None:
            raise Refuse(node, f"super().{meth}: no base in this module defines it")
        sig = self.mod.sigs.get(f"{base}.{meth}")
        if sig is None:
            raise Refuse(node, f"method {base}.{meth} is not translated")
        me = self.lookup(self.node.args.args[0].arg)
        ms, bs = self.mod.class_struct.get(cls), self.mod.class_struct.get(base)
        if ms != bs:
            a_, b_ = self.env.headers.structs.get(ms), self.env.headers.structs.get(bs)
            if (
                a_ is None
                or b_ is None
                or a_.order[: len(b_.order)] != b_.order
                or any(a_.fields[k] != b_.fields[k] for k in b_.order)
            ):
                raise Refuse(node, f"super(): {cls} does not extend {base}'s fields")
            self_code = f"reinterpret_cast<{'const ' if sig.params[0][1].startswith('const') else ''}{bs}*>({me.code})"
        else:
            self_code = me.code
        fn = self.mod.methods[(base, meth)]
        bound = self.bind_call_args(node, py_params(fn), skip=1)
        args = []
        for pname, t in sig.params:
            if pname == sig.params[0][0]:
                args.append(self_code)
            elif pname == "b" and t == T.B and "b" not in bound:
                args.append(self.builder)
            elif pname in ("fence", "seq_guards") and pname not in bound:
                args.append(self.fence if pname == "fence" else self.seq_guards)
            elif pname in bound:
                args.append(self.coerce(bound[pname], t, node))
            else:
                d = sig.defaults.get(pname)
                if d is None:
                    raise Refuse(node, f"super().{meth}: missing {pname}")
                args.append(self.coerce(self.expr(d), t, node))
        return self.bind(CE(f"{sig.name}({', '.join(args)})", sig.ret, False), meth)

    def call_struct(self, recv: CE, attr: str, node):
        """obj.method(...) on a struct: a one-method duck object or a translated class method."""
        st = T.struct_of(recv.t)
        if st is None:
            return None
        s = self.env.headers.structs.get(st[0])
        if s is not None and set(s.fields) == {"fn", "ctx"}:
            return self.call_callable_struct(recv, node)
        if (
            s is not None
            and "ctx" in s.fields
            and attr in s.fields
            and s.fields[attr] in self.env.headers.fnptrs
        ):
            return self.call_callable_struct(recv, node, field=attr)
        if s is not None and attr in s.fields:
            # a field holding a callable ({fn, ctx}): call it
            fs = T.struct_of(s.fields[attr])
            inner = self.env.headers.structs.get(fs[0]) if fs else None
            if inner is not None and set(inner.fields) == {"fn", "ctx"}:
                sep = "->" if st[1] else "."
                return self.call_callable_struct(
                    CE(f"{recv.code}{sep}{attr}", s.fields[attr], recv.pure), node
                )
        cls = self.mod.struct_class.get(st[0])
        from .host import closure_fields

        if cls is not None and (cls, attr) in closure_fields(self.mod):
            return self.call_closure_field(recv, st, cls, attr, node)
        if (
            cls is None
            and self.mod.tmap is not None
            and st[0] in self.mod.tmap.struct_class
        ):
            return self.call_imported_method(recv, st, attr, node)
        if cls is None:
            ha = self.host_arg(node)
            if ha is not None:
                return self.host_fold_call(recv, st, attr, node, *ha)
            return self.call_platform_method(recv, st, attr, node)
        cls, dsig = self.family_resolve(recv, cls, attr, node)
        sig = dsig or self.mod.sigs.get(f"{cls}.{attr}")
        fn = self.mod.methods.get((cls, attr)) or (dsig.py if dsig else None)
        if sig is None and fn is not None and self.mod.family(cls) is None:
            filled = self.call_filled(f"{cls}.{attr}", fn, node, recv=recv)
            if filled is not None:
                return filled
        if sig is None or fn is None:
            raise Refuse(node, f"method {cls}.{attr} is not translated")
        bound = self.bind_call_args(node, py_params(fn), skip=1)
        args = []
        for pname, t in sig.params:
            if pname == "self":
                args.append(self.coerce(recv, t, node))
            elif pname in bound:
                args.append(self.coerce(bound[pname], t, node))
            elif pname == "fence":
                args.append(self.fence or "nullptr")
            elif pname == "seq_guards":
                args.append(self.seq_guards or "nullptr")
            elif pname == "b" and t == T.B and pname not in bound:
                args.append(self.builder)
            elif pname in sig.defaults:
                args.append(self.coerce(self.expr(sig.defaults[pname]), t, node))
            else:
                raise Refuse(node, f"missing argument {pname} to {cls}.{attr}")
        return self.bind(CE(f"{sig.name}({', '.join(args)})", sig.ret, False), attr)

    def call_platform_method(self, recv: CE, st, attr, node):
        """PM-R1: obj.method(...) on a platform struct with no twin_map class.

        The C twin is found by the header's naming convention and confirmed by
        its doc comment: rocke_<stem>_t -> rocke_<stem>_<method>(self, ...), or
        <prefix>_<method> when the header declares "Class -> rocke_<stem>_t and
        <prefix>_*"; the comment above it says "Mirrors <Class>.<method>" or
        starts with "<method>(". Python's positional arguments fill the C
        parameters after self in order; trailing out pointers (out, out<k>,
        out_<name>) of one type are the result (one value, or a fixed tuple).
        A bool return with a builder parameter is the builder error, a
        rocke_status_t return a ValueError (the Python raise); any other
        return type is the result itself."""
        if (
            not st[0].endswith("_t")
            or node.keywords
            or any(isinstance(a, ast.Starred) for a in node.args)
        ):
            return None
        cf = platform_c_method(self.env, st[0], attr)
        if cf is None:
            return None
        rest = cf.params[1:]
        nout = 0
        while (
            nout < len(rest)
            and re.fullmatch(r"out(\d*|_\w+)", rest[len(rest) - 1 - nout][1])
            and rest[len(rest) - 1 - nout][0].endswith("*")
            and not rest[len(rest) - 1 - nout][0].startswith("const ")
        ):
            nout += 1
        ins, outs = rest[: len(rest) - nout], rest[len(rest) - nout :]
        if len(node.args) != len(ins) or len({t for t, _ in outs}) > 1:
            return None
        takes_b = any(t == T.B for t, _ in ins)
        if cf.ret == "bool" and not takes_b:
            return None
        if outs and cf.ret not in ("void", "bool", "rocke_status_t"):
            return None
        if not st[1] and not cf.params[0][0].startswith("const "):
            # by value and the C method may update it: only through a path
            # that is not const (a local, or fields of a non-const pointer)
            m_ = re.fullmatch(r"(\w+)((?:(?:->|\.)\w+)*)", recv.code)
            root = self.lookup(m_.group(1)) if m_ else None
            if root is None or root.t.startswith("const "):
                return None
        if st[1] and not recv.code.isidentifier():
            recv = self.bind(CE(recv.code, recv.t, False), attr + "_self")
        # a struct held by value is passed by address, never copied: the C
        # method may update it (a cache), as the Python method updates self
        args = [recv.code if st[1] else f"&{paren(recv.code)}"]
        for (cty, _cn), a in zip(ins, node.args):
            args.append(self.coerce(self.bind(self.expr(a), "a"), cty, node))
        names = []
        el = outs[0][0][:-1] if outs else None
        for _t, o in outs:
            nm = self.tmp(o)
            self.emit(f"{decl_type(el)} {nm}{{}};")
            names.append(nm)
            args.append(f"&{nm}")
        call = f"{cf.name}({', '.join(args)})"
        if cf.ret == "bool":
            self.emit(
                f"if(!{call}) py_adapt::raise_builder({self.builder or 'nullptr'});"
            )
        elif cf.ret == "rocke_status_t":
            self.emit(f'if({call} != ROCKE_OK) throw ckc::ValueError("{attr} failed");')
        elif cf.ret != "void":
            return self.bind(CE(call, cf.ret, False), attr)
        else:
            self.emit(f"{call};")
        if not outs:
            return CE("", T.VOID)
        if len(outs) == 1:
            return CE(names[0], el, True)
        return CE(
            f"{T.arr(el, len(outs))}{{{', '.join(names)}}}",
            T.arr(el, len(outs)),
            True,
            meta={"items": names},
        )

    def host_arg(self, node):
        """host clones: (argument index, host builder class) of the argument of a call
        that is a host builder value, if any."""
        from .host import host_class_of

        for i, a in enumerate(node.args):
            if not isinstance(a, ast.Name):
                continue
            lv = self.lookup(a.id)
            if lv is not None:
                h = host_class_of(self.mod, lv.t)
                if h is not None:
                    return i, h
                continue
            val = getattr(self.mod.pymod, a.id, None)
            if a.id not in self.mod.classes and type(val).__name__ in host_builders(
                self.mod
            ):
                return i, type(val).__name__
        return None

    def host_ops(self, hcls: str, inst: CE) -> str:
        """host clones: a lambda applying host builder hcls's own methods by IRBuilder
        method name (the op table py_adapt::host_fold folds with)."""
        if hcls not in self.mod.classes:
            raise Refuse(
                self.node, f"host fold on {hcls}, a host builder of another module"
            )
        arms = []
        for (c, m), fn in self.mod.methods.items():
            sig = self.mod.sigs.get(f"{c}.{m}")
            if c != hcls or sig is None:
                continue
            n = len(sig.params) - 1
            if n not in (1, 2):
                continue
            args = ", ".join(["a_", "c_"][:n])
            ptr = inst.code if T.is_ptr(inst.t) else f"&{inst.code}"
            arms.append(
                f'if(strcmp(op_, "{m}") == 0) return (int64_t){sig.name}({ptr}, {args});'
            )
        if not arms:
            raise Refuse(self.node, f"host builder {hcls} has no translated methods")
        name = self.tmp("ops")
        self.emit(
            f"auto {name} = [&](const char* op_, [[maybe_unused]] int64_t a_, [[maybe_unused]] int64_t c_) -> int64_t"
        )
        self.emit("{")
        for a in arms:
            self.emit("    " + a)
        self.emit(
            f'    throw ckc::ValueError("host fold: {hcls} has no method for this op");'
        )
        self.emit("};")
        return name

    def host_fold_call(
        self, recv: CE, st, attr, node, i: int, hcls: str, call_ir=None
    ) -> CE:
        """host clones: a platform method (or helper, call_ir) called with a host
        builder: its IR twin on a scratch builder, the values it returns folded
        with the host builder's methods (designs/host_clones.md)."""
        inst = self.bind(self.expr(node.args[i]), "hb")
        ops = self.host_ops(hcls, inst)
        sb = self.tmp("scratch")
        self.emit(f"py_adapt::host_scratch {sb};")
        bname = self.tmp("sb")
        self.emit(f"rocke_ir_builder_t* {bname} = &{sb}.b;")
        self.vars[bname] = T.B
        self.declared.add(bname)
        call = copy.copy(node)
        call.args = list(node.args)
        call.args[i] = ast.copy_location(
            ast.Name(id=bname, ctx=ast.Load()), node.args[i]
        )
        saved, saved_fold = self.builder, getattr(self, "host_scratch", None)
        self.builder, self.host_scratch = bname, bname
        try:
            got = (
                call_ir(call)
                if call_ir is not None
                else self.call_platform_method(recv, st, attr, call)
            )
        finally:
            self.builder, self.host_scratch = saved, saved_fold
        if got is None:
            raise Refuse(
                node, f"method {attr} on {recv.t if recv else None} with a host builder"
            )
        if got.t == T.V:
            return self.bind(
                CE(f"py_adapt::host_fold({got.code}, {ops})", HOST_VALUE, False), attr
            )
        items = (got.meta or {}).get("items")
        if T.seq_elem(got.t) == T.V and items:
            folded = ", ".join(f"py_adapt::host_fold({x}, {ops})" for x in items)
            return self.bind(
                CE(
                    f"{T.arr(HOST_VALUE, len(items))}{{{folded}}}",
                    T.arr(HOST_VALUE, len(items)),
                    False,
                ),
                attr,
            )
        raise Refuse(node, f"host fold of a {got.t}")

    def closure_value(self, sname: str, fld: str, ce: CE, node) -> list[str]:
        """host clones: designated initializers of a closure stored into Callable field
        fld: its definition site's kind and the values it captured."""
        from .host import _bindings, closure_fields, closure_kind_const, site_field

        key, i = ce.t[len("closure:") :].rsplit(":", 1)
        cls_, fld_ = key.split(".", 1)
        site = closure_fields(self.mod)[(cls_, fld_)][int(i)]
        if fld_ != fld:
            raise Refuse(node, f"closure {ce.code} of field {fld_} stored into {fld}")
        if site.enclosing is not self.node:
            raise Refuse(node, f"closure {ce.code} stored outside {site.name}")
        once = _bindings(site.enclosing)
        s = self.env.headers.structs[sname]
        items = [f".{fld.lstrip('_')}_kind = {closure_kind_const(sname, fld, site)}"]
        plain = site.node.args.posonlyargs + site.node.args.args
        nd = len(site.node.args.defaults)
        defaults = {
            x.arg: d for x, d in zip(plain[len(plain) - nd :], site.node.args.defaults)
        }
        for cap, kind in site.captured:
            src = (
                defaults[cap] if kind == "default" else ast.Name(id=cap, ctx=ast.Load())
            )
            for n in ast.walk(src):
                if isinstance(n, ast.Name) and once.get(n.id, 0) > 1:
                    # Python reads the variable when the closure runs: one binding keeps the value
                    raise Refuse(
                        node,
                        f"closure {site.node.name} captures {n.id}, which {site.name} rebinds",
                    )
            ast.copy_location(src, node)
            v = self.bind(self.expr(src), cap.lstrip("_"))
            cf = site_field(fld, site, cap)
            if cf not in s.fields:
                raise Refuse(node, f"{sname} has no field {cf} (regenerate the ABI)")
            items.append(f".{cf} = {self.coerce(v, s.fields[cf], node)}")
        return items

    def call_closure_field(self, recv: CE, st, cls: str, fld: str, node) -> CE:
        """host clones: obj.fld(...) of a Callable field holding closures of this module:
        a switch on the stored kind, each arm the site's lifted function of
        the call's value kind (the host clone when the builder argument is a
        host builder)."""
        from .host import closure_fields, closure_kind_const, host_class_of, site_field

        sites = closure_fields(self.mod)[(cls, fld)]
        own = sites[0].own
        if (
            node.keywords
            or any(isinstance(a, ast.Starred) for a in node.args)
            or len(node.args) != len(own)
            or any(s_.own != own for s_ in sites)
        ):
            raise Refuse(node, f"call shape of the closure field {cls}.{fld}")
        args = [self.bind(self.expr(a), p) for a, p in zip(node.args, own)]
        bp = builder_param(sites[0].node)
        hcls = host_class_of(self.mod, args[own.index(bp)].t) if bp in own else None
        sigs = []
        for site in sites:
            key = closure_key(site) + (f"@{hcls}" if hcls else "")
            sig = self.mod.sigs.get(key)
            if sig is None:
                self.mod.host_demand.add((closure_key(site), "<closure>", hcls))
                raise Refuse(
                    node, f"closure {key} of {cls}.{fld}, which is not translated"
                )
            sigs.append(sig)
        rets = {sg.ret for sg in sigs}
        if len(rets) != 1:
            raise Refuse(node, f"closures of {cls}.{fld} return {sorted(rets)}")
        ret = rets.pop()
        if not recv.code.isidentifier():
            recv = self.bind(CE(recv.code, recv.t, False), "self")
        sep = "->" if st[1] else "."
        r = self.tmp(fld.lstrip("_"))
        self.emit(f"{decl_type(ret)} {r}{{}};")
        self.emit(f"switch({recv.code}{sep}{fld.lstrip('_')}_kind)")
        self.emit("{")
        for site, sig in zip(sites, sigs):
            cargs = []
            for (pn, t), i in zip(sig.params, range(len(sig.params))):
                if i < len(own):
                    cargs.append(self.coerce(args[i], t, node))
                else:
                    cargs.append(f"{recv.code}{sep}{site_field(fld, site, pn)}")
            self.emit(f"case {closure_kind_const(st[0], fld, site)}:")
            self.emit(f"    {r} = {sig.name}({', '.join(cargs)});")
            self.emit("    break;")
        self.emit("default:")
        self.emit(f'    throw ckc::ValueError("{cls}.{fld}: no closure of that kind");')
        self.emit("}")
        return CE(r, ret, False)

    def platform_class(self, cls: str):
        """An imported platform class with a C struct by the header's naming
        convention (AttnRuntimeBounds -> rocke_attn_runtime_bounds_t): (struct, class).
        """
        import inspect

        pycls = getattr(self.mod.pymod, cls, None)
        if not inspect.isclass(pycls):
            return None
        snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", cls).lower()
        sname = f"rocke_{snake}_t"
        if sname not in self.env.headers.structs:
            return None
        return sname, pycls

    def construct_platform(self, cls: str, sname: str, pycls, node) -> CE:
        """X(...) of an imported platform class found by naming: a dataclass is
        a struct with its field defaults, then its C validate (the Python
        __post_init__); any other class a struct local plus its C init
        (rocke_<stem>_init(self, ...), "X.__init__" in its comment). Built on
        a host builder (host clones): its host image."""
        import dataclasses

        stem = sname[:-2]
        if dataclasses.is_dataclass(pycls):
            fields = [f_ for f_ in dataclasses.fields(pycls) if f_.init]
            bound = self.bind_call_args(
                node,
                [{"name": f_.name, "kind": "pos", "default": None} for f_ in fields],
            )
            st = self.env.headers.structs[sname]
            items = []
            for f_ in fields:
                cf = f_.name if f_.name in st.fields else f_.name.lstrip("_")
                if cf not in st.fields:
                    raise Refuse(node, f"{sname} has no field {f_.name}")
                if f_.name in bound:
                    items.append((cf, self.coerce(bound[f_.name], st.fields[cf], node)))
                elif f_.default is not dataclasses.MISSING and isinstance(
                    f_.default, (bool, int, float, str)
                ):
                    items.append(
                        (
                            cf,
                            self.coerce(
                                CE(lit_code(f_.default), lit_type(f_.default)),
                                st.fields[cf],
                                node,
                            ),
                        )
                    )
                else:
                    raise Refuse(
                        node, f"{cls}() without {f_.name} (no literal default)"
                    )
            order_ = {f: i for i, f in enumerate(st.order)}
            items.sort(key=lambda x: order_[x[0]])
            obj = self.tmp(cls.lower())
            self.emit(
                f"const {sname} {obj}{{{', '.join(f'.{c} = {v}' for c, v in items)}}};"
            )
            val = self.env.c_func(f"{stem}_validate")
            if hasattr(pycls, "__post_init__"):
                if val is None or len(val.params) != 3:
                    raise Refuse(node, f"{cls}.__post_init__ has no C validate")
                why = self.tmp("why")
                self.emit(f"char {why}[256] = {{0}};")
                self.emit(
                    f"if(!{val.name}(&{obj}, {why}, sizeof {why})) throw ckc::ValueError({why});"
                )
            return CE(obj, sname)
        cf = self.env.c_func(f"{stem}_init")
        if cf is None or not cf.params or cf.params[0][0] != f"{sname}*":
            raise Refuse(node, f"{cls}() has no C init {stem}_init")
        path = self.env.headers.paths.get(cf.file)
        doc = (
            " ".join(open(path).read().splitlines()[max(0, cf.line - 8) : cf.line - 1])
            if path
            else ""
        )
        if not re.search(rf"\b{cls}\.__init__\b", doc):
            raise Refuse(node, f"{cf.name} does not say it mirrors {cls}.__init__")
        if node.keywords or len(node.args) != len(cf.params) - 1:
            raise Refuse(node, f"{cls}(): call shape of {cf.name}")
        ha = self.host_arg(node)
        args = [self.bind(self.expr(a), "a") for a in node.args]
        sb = None
        if ha is not None:
            # host clones: built on a host builder: the IR init on a scratch builder, folded (host image)
            sc = self.tmp("scratch")
            self.emit(f"py_adapt::host_scratch {sc};")
            sb = f"&{sc}.b"
        cargs = []
        for (ct, _cn), a, i in zip(cf.params[1:], args, range(len(args))):
            if ha is not None and i == ha[0]:
                cargs.append(sb)
            elif ct == T.B:
                cargs.append(self.coerce(a, ct, node))
            elif (
                ct.startswith("const ")
                and ct.endswith("*")
                and a.t == ct[len("const ") : -1].strip()
            ):
                cargs.append(f"&{a.code}")
            else:
                cargs.append(self.coerce(a, ct, node))
        obj = self.tmp(cls.lower())
        self.emit(f"{sname} {obj}{{}};")
        call = f"{cf.name}(&{obj}, {', '.join(cargs)})"
        if cf.ret == "rocke_status_t":
            self.emit(
                f"if({call} != ROCKE_OK) py_adapt::raise_builder({sb or self.builder or 'nullptr'});"
            )
        else:
            self.emit(f"{call};")
        if ha is None:
            return CE(obj, sname)
        return self.fill_host_image(cls, sname, obj, ha[1], args[ha[0]], pycls)

    def image_host(self, t: str):
        """The host builder class of a host image type, if t is one."""
        st = T.struct_of(t) if t else None
        if st is None:
            return None
        for (_ir, h), img in self.mod.host_images.items():
            if img == st[0]:
                return h
        return None

    def fill_host_image(
        self, cls: str, sname: str, obj: str, hcls: str, hb: CE, pycls
    ) -> CE:
        """host clones: the host image of platform struct sname built on host builder
        hcls: its fields with the values folded to ints (designs/host_clones.md)."""
        from .env import CStruct
        import inspect
        import textwrap

        s = self.env.headers.structs[sname]
        hst = host_builders(self.mod)[hcls]
        img = f"py_{cls}__{hcls}_t"
        # Python attributes that may hold None (the IR field is then a NULL Value)
        none_ok = set()
        try:
            init = ast.parse(textwrap.dedent(inspect.getsource(pycls.__init__)))
            for n in ast.walk(init):
                if (
                    isinstance(n, ast.Assign)
                    and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Attribute)
                ):
                    if any(
                        isinstance(x, ast.Constant) and x.value is None
                        for x in ast.walk(n.value)
                    ):
                        none_ok.add(n.targets[0].attr.lstrip("_"))
            assigned = {
                n.targets[0].attr.lstrip("_")
                for n in ast.walk(init)
                if isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Attribute)
            }
        except (OSError, TypeError, SyntaxError):
            assigned = set()
        fields, conv = {}, {}
        for f in s.order:
            ft = s.fields[f]
            sub = self.env.headers.structs.get(ft)
            if ft == T.B:
                fields[f] = f"const {hst}*"
                conv[f] = hb.code if T.is_ptr(hb.t) else f"&{hb.code}"
            elif ft == T.V:
                if f in assigned and f not in none_ok:
                    fields[f] = HOST_VALUE
                    conv[f] = f"py_adapt::host_fold({obj}.{f}, OPS)"
                else:
                    fields[f] = T.optional(HOST_VALUE)
                    conv[f] = (
                        f"({obj}.{f} ? {T.optional(HOST_VALUE)}(py_adapt::host_fold({obj}.{f}, OPS)) : {T.optional(HOST_VALUE)}())"
                    )
            elif ft == "rocke_int_or_value_t":
                fields[f] = HOST_VALUE
                conv[f] = (
                    f"({obj}.{f}.is_value ? py_adapt::host_fold({obj}.{f}.v, OPS) : {obj}.{f}.i)"
                )
            elif (
                sub is not None
                and len(sub.order) == 2
                and sub.order[0] == "has"
                and sub.fields[sub.order[1]] in ("rocke_int_or_value_t", T.V)
            ):
                inner = f"{obj}.{f}.{sub.order[1]}"
                iv = (
                    f"({inner}.is_value ? py_adapt::host_fold({inner}.v, OPS) : {inner}.i)"
                    if sub.fields[sub.order[1]] != T.V
                    else f"py_adapt::host_fold({inner}, OPS)"
                )
                fields[f] = T.optional(HOST_VALUE)
                conv[f] = (
                    f"({obj}.{f}.has ? {T.optional(HOST_VALUE)}({iv}) : {T.optional(HOST_VALUE)}())"
                )
            elif sub is not None and any(x in (T.V, T.B) for x in sub.fields.values()):
                raise Refuse(
                    self.node, f"host image of {sname}: field {f} holds IR objects"
                )
            else:
                fields[f] = ft
                conv[f] = f"{obj}.{f}"
        key = (sname, hcls)
        if key not in self.mod.host_images:
            self.mod.host_images[key] = img
            self.mod.host_image_defs[img] = (
                [
                    f"/* host clones: {cls} built on {hcls} (host image of {sname}) */",
                    f"struct {img}",
                    "{",
                ]
                + [f"    {decl_type(t)} {f}{{}};" for f, t in fields.items()]
                + ["};", ""]
            )
            self.env.headers.structs[img] = CStruct(img, dict(fields), list(fields))
        ops = self.host_ops(hcls, hb)
        out = self.tmp(cls.lower())
        self.emit(
            f"const {img} {out}{{{', '.join(f'.{f} = ' + c.replace('OPS', ops) for f, c in conv.items())}}};"
        )
        return CE(out, img)

    def construct_private(self, cls: str, node) -> CE:
        """A module class with no C twin: its generated struct plus the translated __init__."""
        sname = self.mod.class_struct.get(cls)
        sig = self.mod.sigs.get(f"{cls}.__init__")
        fn = self.mod.methods.get((cls, "__init__"))
        if sname is None or sig is None or fn is None:
            raise Refuse(
                node, f"class {cls} has no generated struct or translated __init__"
            )
        bound = self.bind_call_args(node, py_params(fn), skip=1)
        obj = self.tmp(cls.lstrip("_").lower())
        self.emit(f"{sname} {obj}{{}};")
        fam = self.mod.family(cls)
        if fam is not None and cls != fam.base:
            self.emit(
                f"{obj}.kind = {kind_const(fam, cls)};"
            )  # tagged values: the member's tag
        args = []
        for pname, t in sig.params:
            if pname == "self":
                args.append(f"&{obj}")
            elif pname in bound:
                args.append(self.coerce(bound[pname], t, node))
            elif pname in ("fence", "seq_guards"):
                args.append(getattr(self, pname) or "nullptr")
            elif pname in sig.defaults:
                args.append(self.coerce(self.expr(sig.defaults[pname]), t, node))
            else:
                raise Refuse(node, f"missing argument {pname} to {cls}()")
        self.emit(f"{sig.name}({', '.join(args)});")
        return CE(obj, sname)

    def host_instance(self, cls: str) -> CE:
        """host clones: a host builder value (no state: a value-initialized struct)."""
        st = host_builders(self.mod)[cls]
        obj = self.tmp(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", cls).lower())
        self.emit(f"const {st} {obj}{{}};")
        return CE(obj, st)

    def helper_entry(self, cls: str, meth: str):
        m = self.mod.tmap.class_module.get(cls)
        return self.env.sig.helpers.get(f"{m}:{cls}.{meth}") if m else None

    def construct_imported(self, cls: str, node) -> CE:
        """An imported class with a C twin: a struct local plus its _init function."""
        sname = self.mod.tmap.class_struct[cls]
        e = self.helper_entry(cls, "__init__")
        if e is None:
            pycls = self.py_class(cls)
            if pycls is not None and dataclasses_is(pycls):
                return self.construct_dataclass(cls, sname, pycls, node)
            raise Refuse(node, f"{cls}() has no C init in sigmap")
        cf = self.env.c_func(e["c"]["name"])
        obj = self.tmp(cls.lower())
        bound = self.bind_call_args(node, e["python"]["params"], skip=1)
        bound.pop("b", None)
        self.emit(f"{sname} {obj}{{}};")
        slots = {s_["c"]: s_ for s_ in e["c"]["slots"]}
        extra = {x["c"]: x["role"] for x in e["c"].get("extra_c", [])}
        args = self.c_args(cf, bound, slots, extra, node, self_arg=f"&{obj}")
        self.emit(self.err_wrap(f"{cf.name}({', '.join(args)})", "void") + ";")
        return CE(obj, sname)

    def construct_dataclass(self, cls, sname, pycls, node) -> CE:
        import dataclasses

        fields = [f_.name for f_ in dataclasses.fields(pycls) if f_.init]
        bound = self.bind_call_args(
            node, [{"name": n, "kind": "pos", "default": None} for n in fields]
        )
        s_ = self.env.headers.structs[sname]
        items = [
            f".{c} = {self.coerce(bound[c], s_.fields[c], node)}"
            for c in s_.order
            if c in bound
        ]
        return CE(
            f"{sname}{{{', '.join(items)}}}", sname, all(v.pure for v in bound.values())
        )

    def call_imported_method(self, recv: CE, st, attr, node) -> CE:
        cls = self.mod.tmap.struct_class[st[0]]
        e = self.helper_entry(cls, attr)
        if e is None:
            raise Refuse(node, f"{cls}.{attr} not in sigmap")
        cf = self.env.c_func(e["c"]["name"])
        if cf is None:
            raise Refuse(node, f"no exported C function for {cls}.{attr}")
        bound = self.bind_call_args(node, e["python"]["params"], skip=1)
        bound.pop("b", None)
        slots = {s_["c"]: s_ for s_ in e["c"]["slots"]}
        extra = {x["c"]: x["role"] for x in e["c"].get("extra_c", [])}
        if not recv.code.isidentifier():
            recv = self.bind(CE(recv.code, recv.t, False), cls.lower())
        self_arg = recv.code if st[1] else f"&{recv.code}"
        args = self.c_args(cf, bound, slots, extra, node, self_arg=self_arg)
        return self.gen_call(cf, args, attr, node)

    def call_replace(self, node) -> CE:
        """DC-R2: replace(obj, f=v): copy, then set (these dataclasses have no __post_init__)."""
        src = self.bind(self.expr(node.args[0]), "src")
        st_ = T.struct_of(src.t)
        if st_ is None:
            raise Refuse(node, f"replace() of {src.t}")
        sname = st_[0]
        cls = (
            self.mod.tmap.struct_class.get(sname) if self.mod.tmap else None
        ) or self.mod.struct_class.get(sname)
        pycls = self.py_class(cls) if cls else None
        fam = self.mod.family(cls) or self.sibling_family(cls, sname)
        if pycls is None or (hasattr(pycls, "__post_init__") and fam is None):
            raise Refuse(
                node, f"replace() of {sname}: unknown class or __post_init__ (DC-R2)"
            )
        vals = {k.arg: self.bind(self.expr(k.value), k.arg) for k in node.keywords}
        out = self.tmp("repl")
        self.emit(
            f"{sname} {out} = {src.code if not st_[1] else '*' + paren(src.code)};"
        )
        s_ = self.env.headers.structs[sname]
        for k, v in vals.items():
            if k not in s_.fields:
                raise Refuse(node, f"replace(): {sname} has no field {k}")
            fld = self.field(CE(out, sname), k, node)
            self.store_field(fld, v, node)
        if fam is not None and hasattr(pycls, "__post_init__"):
            self.revalidate(
                out, sname, fam, node
            )  # tagged values: the copy's own class checks it
        return CE(out, sname)

    def sibling_family(self, cls: str | None, sname: str):
        """A sibling module's tagged-value family whose tagged struct sname is."""
        f = (
            self.mod.tmap.families.get(cls)
            if cls and self.mod.tmap is not None
            else None
        )
        s_ = self.env.headers.structs.get(sname)
        if (
            f is None
            or f.module == self.mod.modname
            or s_ is None
            or s_.fields.get("kind") != kind_type(f)
        ):
            return None
        return f

    def revalidate(self, obj: str, sname: str, fam, node):
        """dataclasses.replace re-runs __post_init__ of the value's own class:
        a switch on the kind tag (tagged values), to each member's implementation."""
        if fam.module == self.mod.modname:
            try:
                dsig = self.mod.dispatcher(fam, fam.base, "__post_init__")
            except Refuse as e:
                raise Refuse(node, e.reason)
            args = []
            for i, (pname, t) in enumerate(dsig.params):
                if i == 0:
                    args.append(self.coerce(CE(obj, sname), t, node))
                elif pname == "b" and t == T.B:
                    args.append(self.builder)
                elif pname in ("fence", "seq_guards"):
                    args.append(getattr(self, pname) or "nullptr")
                else:
                    raise Refuse(node, f"replace(): {dsig.name} needs {pname}")
            self.emit(f"{dsig.name}({', '.join(args)});")
            return
        impls = {
            m: i
            for m, i in fam.impls(fam.base, "__post_init__").items()
            if not fam.is_peer(m)
        }
        self.emit(f"switch({obj}.kind)")
        self.emit("{")
        for m in fam.members:
            if m not in impls:
                continue
            e = self.helper_entry(impls[m], "__post_init__")
            cf = self.env.c_func(e["c"]["name"]) if e else None
            if cf is None:
                raise Refuse(
                    node,
                    f"replace(): no exported C function for {impls[m]}.__post_init__",
                )
            slots = {s_["c"]: s_ for s_ in e["c"]["slots"]}
            extra = {x["c"]: x["role"] for x in e["c"].get("extra_c", [])}
            self.emit(f"case {kind_const(fam, m)}:" if m != fam.base else "default:")
            self.emit("{")
            args = self.c_args(cf, {}, slots, extra, node, self_arg=f"&{obj}")
            self.gen_call(cf, args, "valid", node)
            self.emit("break;")
            self.emit("}")
        self.emit("}")

    def py_class(self, cls: str):
        """The Python class: this module's own or imported name, else its defining module (twin_map)."""
        pycls = getattr(self.mod.pymod, cls, None)
        if (
            pycls is None
            and self.mod.tmap is not None
            and cls in self.mod.tmap.class_module
        ):
            pycls = getattr(
                self.env.py.module(self.mod.tmap.class_module[cls]), cls, None
            )
        return pycls

    def classmethod_owner(self):
        """(class, name of its cls parameter) when lowering a classmethod."""
        fn = self.node if self.outer is None else None
        if fn is None or not any(
            isinstance(d, ast.Name) and d.id == "classmethod" for d in fn.decorator_list
        ):
            return None
        cls = next((c for (c, _m), f in self.mod.methods.items() if f is fn), None)
        return (cls, fn.args.args[0].arg) if cls and fn.args.args else None

    def construct(self, cls: str, node) -> CE:
        """A dataclass constructor: a struct with designated initializers in C field order."""
        import dataclasses

        sname = self.mod.class_struct.get(cls)
        pycls = getattr(self.mod.pymod, cls, None)
        if (
            sname is not None
            and pycls is not None
            and not dataclasses.is_dataclass(pycls)
            and (cls, "__init__") in self.mod.methods
        ):
            return self.construct_private(
                cls, node
            )  # its struct plus the translated __init__
        if sname is None or pycls is None or not dataclasses.is_dataclass(pycls):
            raise Refuse(
                node, f"constructor of {cls} (not a dataclass with a C struct)"
            )
        fields = [f.name for f in dataclasses.fields(pycls) if f.init]
        params = [{"name": n, "kind": "pos", "default": None} for n in fields]
        bound = self.bind_call_args(node, params)
        s = self.env.headers.structs[sname]
        # a list field the struct owns (pointer, n_, cap_): a malloc'd copy
        owned = {f for f in s.fields if "n_" + f in s.fields and "cap_" + f in s.fields}
        lens = {}
        items = []
        for cfld in s.order:
            if cfld.startswith(("n_", "cap_")) and cfld.split("_", 1)[1] in lens:
                items.append(f".{cfld} = {lens[cfld.split('_', 1)[1]]}")
                continue
            py = (
                cfld if cfld in bound else ("_" + cfld if "_" + cfld in bound else None)
            )
            if py is None:
                continue
            if cfld in owned and T.is_seq(bound[py].t):
                src = self.iter_src(bound[py].code)
                el = s.fields[cfld][:-1]
                lens[cfld] = f"(int)py::len({src})"
                es = self.env.headers.structs.get(el)
                if es is not None and "_tuple_" in el:
                    conv = f"[](const auto& x) {{ return {el}{{{', '.join(f'({es.fields[f]})std::get<{i}>(x)' for i, f in enumerate(es.order))}}}; }}"
                    items.append(f".{cfld} = py::own_copy_map<{el}>({src}, {conv})")
                else:
                    items.append(f".{cfld} = py::own_copy<{el}>({src})")
                continue
            if "has_" + cfld in s.fields:
                has, val = self.optional_parts(bound[py], s.fields[cfld], node)  # N-R2
                items.append(f".has_{cfld} = {has}")
                if val is not None:
                    items.append(f".{cfld} = {val}")
                continue
            items.append(f".{cfld} = {self.coerce(bound[py], s.fields[cfld], node)}")
        missing = set(fields) - set(bound)
        for f_ in dataclasses.fields(pycls):
            # a field the call leaves out: its default (a literal, or an empty
            # default_factory list / tuple / dict)
            if f_.name not in missing:
                continue
            cf_ = f_.name if f_.name in s.fields else f_.name.lstrip("_")
            if cf_ not in s.fields:
                continue
            if f_.default is not dataclasses.MISSING:
                d_ = f_.default
                if d_ is None:
                    if "has_" + cf_ in s.fields:
                        items.append(f".has_{cf_} = false")
                    elif not T.is_ptr(s.fields[cf_]):
                        continue
                    missing.discard(f_.name)
                    continue
                if isinstance(d_, (tuple, list)) and not d_:
                    missing.discard(f_.name)
                    continue
                if isinstance(d_, (bool, int, float, str)):
                    items.append(f".{cf_} = {lit_code(d_)}")
                    missing.discard(f_.name)
            elif f_.default_factory in (list, tuple, dict):
                missing.discard(f_.name)  # empty: the struct's zero
        if missing:
            raise Refuse(
                node, f"{cls}() without {sorted(missing)} (defaults not handled)"
            )
        fam = self.mod.family(cls)
        if fam is not None and cls != fam.base and "kind" in s.fields:
            items.append(
                f".kind = {kind_const(fam, cls)}"
            )  # tagged values: the member's tag
        for py_, v_ in bound.items():
            if v_.t.startswith("closure:"):
                items += self.closure_value(sname, py_, v_, node)
        # designated initializers in field order
        order_ = {f: i for i, f in enumerate(s.order)}
        items.sort(key=lambda x: order_.get(x[1:].split(" ", 1)[0], 0))
        return CE(
            f"{sname}{{{', '.join(items)}}}", sname, all(v.pure for v in bound.values())
        )

    def call_append(self, name, node):
        cur = self.lookup(name)
        if cur is None:
            raise Refuse(node, f"append to unknown {name}")
        el = T.seq_elem(cur.t) if cur.t != "list?" else None
        # a tuple display appended to a list of known tuples: built as that tuple
        want = (
            el
            if el and isinstance(node.args[0], ast.Tuple) and T.tuple_items(el)
            else None
        )
        v = self.bind(self.expr(node.args[0], want=want), "e")
        if el is None or el == T.AUTO:
            if self.dry:
                prev = self.hints["elem"].get(name)
                el = v.t if prev is None else (T.common(prev, v.t) or v.t)
                self.hints["elem"][name] = el
                self.vars[name] = T.vec(el)
                self.hints["vars"][name] = T.vec(el)
                return CE(f"{name}.push_back({v.code})", T.VOID, False)
            raise Refuse(node, f"append to {name} of unknown element type")
        if self.dry:
            self.hints["elem"].setdefault(name, el)
        return CE(f"{name}.push_back({self.coerce(v, el, node)})", T.VOID, False)

    def _is_seq_guard_counter(self, n) -> bool:
        return (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "setdefault"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "_SEQ_GUARDS"
        )

    def builtin(self, name, node, want):
        if self.lookup(name) is not None:
            return None
        args = node.args
        if name == "len":
            v = self.expr(args[0])
            if v.meta and "n" in v.meta:
                return CE(f"(int64_t){v.meta['n']}", T.I64, True)
            if T.is_seq(v.t) or v.t == T.SSTR:
                return CE(f"py::len({v.code})", T.I64, v.pure)
            raise Refuse(node, f"len of {v.t}")
        if name == "getattr" and len(args) in (2, 3) and isinstance(args[1], ast.Name):
            lit_ = self.lookup(args[1].id)
            if lit_ is not None and isinstance(lit_.lit, str):
                return self.expr(
                    ast.copy_location(
                        ast.Call(
                            func=node.func,
                            args=[args[0], ast.Constant(lit_.lit)] + list(args[2:]),
                            keywords=[],
                        ),
                        node,
                    )
                )
        if (
            name == "getattr"
            and len(args) == 3
            and isinstance(args[1], ast.Constant)
            and isinstance(args[1].value, str)
            and not node.keywords
        ):
            # getattr(x, "k", d): the field when the type has it, else d (a static probe)
            obj = self.expr(args[0])
            st_ = T.struct_of(obj.t)
            s_ = self.env.headers.structs.get(st_[0]) if st_ else None
            has = s_ is not None and (
                args[1].value in s_.fields or args[1].value.lstrip("_") in s_.fields
            )
            if not has and st_:
                cls = self.mod.struct_class.get(st_[0]) or (
                    self.mod.tmap.struct_class.get(st_[0]) if self.mod.tmap else None
                )
                pycls = self.py_class(cls) if cls else None
                has = pycls is not None and hasattr(pycls, args[1].value)
            if has:
                return self.field(obj, args[1].value, node)
            return self.expr(args[2])
        if (
            name == "getattr"
            and len(args) == 2
            and isinstance(args[1], ast.Constant)
            and isinstance(args[1].value, str)
            and not node.keywords
        ):
            return self.expr(
                ast.copy_location(
                    ast.Attribute(value=args[0], attr=args[1].value, ctx=ast.Load()),
                    node,
                )
            )
        if name == "divmod" and len(args) == 2 and not node.keywords:
            x = self.bind(self.expr(args[0]), "n")
            y = self.bind(self.expr(args[1]), "d")
            if not (T.is_int(x.t) and T.is_int(y.t)):
                raise Refuse(node, f"divmod of {x.t}, {y.t}")
            return CE(
                f"std::array<int64_t, 2>{{py::floordiv({x.code}, {y.code}), py::mod({x.code}, {y.code})}}",
                T.arr(T.I64, 2),
                x.pure and y.pure,
            )
        if name == "range" and 1 <= len(args) <= 3 and not node.keywords:
            # range(...) as a value (tuple(range(n)), list(range(...))): its ints
            vs = [
                self.coerce(self.bind(self.expr(a_), "r"), T.I64, node) for a_ in args
            ]
            return CE(f"py::range_list({', '.join(vs)})", T.vec(T.I64), False)
        if name == "dict" and len(args) == 1 and not node.keywords:
            # dict(d): a copy (of a dict, or of a field holding (key, value) pairs)
            v = self.expr(args[0], want=want)
            if re.fullmatch(r"(?:const )?py::dict<.+>&?", v.t):
                t_ = re.fullmatch(r"(?:const )?(py::dict<.+>)&?", v.t).group(1)
                return CE(f"{t_}({v.code})", t_, v.pure)
            if v.t == "dict?":
                return v
            el = T.seq_elem(self.readable(v).t)
            pair = self.tuple_struct_items(CE("x", el)) if el else None
            if el and (
                T.arr_len(el) == 2
                or (pair and len(pair) == 2)
                or (T.tuple_items(el) and len(T.tuple_items(el)) == 2)
            ):
                kt = (
                    pair[0][1]
                    if pair
                    else (T.seq_elem(el) if T.arr_len(el) else T.tuple_items(el)[0])
                )
                vt = (
                    pair[1][1]
                    if pair
                    else (T.seq_elem(el) if T.arr_len(el) else T.tuple_items(el)[1])
                )
                kt, vt = (T.I64 if T.is_int(kt) else kt), (
                    T.I64 if T.is_int(vt) else vt
                )
                d_ = self.tmp("dct")
                src = self.bind(self.readable(v), "pairs")
                self.emit(f"py::dict<{kt}, {vt}> {d_};")
                a0, a1 = (
                    ("x_.f0", "x_.f1")
                    if pair
                    else ("std::get<0>(x_)", "std::get<1>(x_)")
                )
                self.emit(f"for(const auto& x_ : {src.code}) {d_}.set({a0}, {a1});")
                return CE(d_, f"py::dict<{kt}, {vt}>")
            raise Refuse(node, f"dict() of {v.t}")
        if name == "set" and not args and not node.keywords:
            return CE("{}", "set?")  # typed by its first add()
        if name in ("str", "repr") and len(args) == 1 and not node.keywords:
            v = self.expr(args[0])
            if v.t == T.EXC:
                if name != "str":
                    raise Refuse(node, f"{name}() of a caught exception")
                # str(e): the ckc message is the Python str() (st_Try)
                return CE(f"{v.code}.message()", T.SSTR, True)
            ts = self.tuple_struct_items(v)
            if (
                ts is not None
                or T.optional_inner(v.t)
                or v.t in (T.V, T.TYPE)
                or T.struct_of(v.t)
            ):
                raise Refuse(node, f"{name}() of {v.t}")
            return CE(f"py::{name}({v.code})", T.SSTR, v.pure)
        if name == "bool" and len(args) == 1 and not node.keywords:
            v = self.expr(args[0])
            return CE(f"({self.truth(v, node)})", T.BOOL, v.pure)
        if name == "int" and len(args) == 1 and not node.keywords:
            v = self.expr(args[0])
            if T.is_int(v.t) or v.t == T.BOOL:
                return CE(
                    f"(int64_t){paren(v.code)}",
                    T.I64,
                    v.pure,
                    lit=int(v.lit) if isinstance(v.lit, (int, bool)) else None,
                )
            if v.t == T.F64:
                return CE(
                    f"(int64_t){paren(v.code)}", T.I64, v.pure
                )  # truncates toward zero, like int()
            if T.is_str(v.t):
                return CE(
                    f"py::int_of({v.code})", T.I64, False
                )  # ValueError like int("x")
            raise Refuse(node, f"int() of {v.t}")
        if (
            name == "float"
            and len(args) == 1
            and isinstance(args[0], ast.Constant)
            and isinstance(args[0].value, str)
        ):
            v = float(args[0].value)
            return CE(lit_code(v), T.F64, lit=v)
        if (
            name == "sorted"
            and len(args) == 1
            and [k.arg for k in node.keywords] == ["key"]
            and isinstance(node.keywords[0].value, ast.Lambda)
        ):
            # sorted(xs, key=lambda x: k(x)): a stable sort on the key (Python's sort is stable)
            lam = node.keywords[0].value
            if len(lam.args.args) != 1:
                raise Refuse(node, "sort key lambda shape")
            src = self.bind(self.readable(self.expr(args[0])), "seq")
            el = T.seq_elem(src.t)
            if el is None:
                raise Refuse(node, f"sorted() of {src.t}")
            x = lam.args.args[0].arg
            keys = []
            for side in ("a_", "b_"):
                mark = len(self.lines)
                self.narrow.append({x: CE(side, el)})
                try:
                    k = self.expr(lam.body)
                finally:
                    self.narrow.pop()
                if len(self.lines) != mark or not (
                    T.is_int(k.t)
                    or k.t in (T.BOOL, T.F64)
                    or T.arr_len(k.t) is not None
                ):
                    del self.lines[mark:]
                    raise Refuse(node, f"sort key of type {k.t} or with emitting code")
                keys.append(k.code)
            out = self.tmp("sorted")
            src_ = self.iter_src(src.code)
            self.emit(
                f"std::vector<{decl_type(el)}> {out}({src_}.begin(), {src_}.end());"
            )
            self.emit(
                f"std::stable_sort({out}.begin(), {out}.end(), [&]({decl_type(el)} const& a_, {decl_type(el)} const& b_) {{ return ({keys[0]}) < ({keys[1]}); }});"
            )
            return CE(out, T.vec(el))
        if (
            name in ("list", "tuple", "sorted", "len")
            and len(args) == 1
            and not node.keywords
        ):
            try:
                v0 = (
                    self.expr(args[0])
                    if not isinstance(args[0], ast.GeneratorExp)
                    else None
                )
            except Refuse:
                v0 = None
            if (
                v0 is not None
                and v0.lit is not None
                and not isinstance(v0.lit, (str,))
                or (v0 is not None and v0.t == "pydict")
            ):
                val = v0.lit
                folded = {"list": list, "tuple": tuple, "sorted": sorted, "len": len}[
                    name
                ](val)
                if name == "len":
                    return CE(lit_code(folded), T.I64, lit=folded)
                return CE("", "pytuple", lit=tuple(folded))
        if (
            name == "sorted"
            and len(args) == 1
            and not node.keywords
            and not isinstance(args[0], ast.GeneratorExp)
        ):
            v = self.readable(self.expr(args[0]))
            dm = re.fullmatch(r"(?:const )?py::dict<(.+)>&?", v.t)
            if dm:  # sorted(d): its keys
                kt_ = (
                    T._split_top(dm.group(1).replace("<", "[").replace(">", "]"))[0]
                    .strip()
                    .replace("[", "<")
                    .replace("]", ">")
                )
                v = CE(f"{paren(v.code)}.keys()", T.vec(kt_), False)
            el = T.seq_elem(v.t)
            if el is None or not (
                T.is_int(el) or el in (T.F64, T.SSTR) or T.arr_len(el) is not None
            ):
                raise Refuse(node, f"sorted() of {v.t}")
            out = self.tmp("sorted")
            src_ = self.iter_src(self.bind(v, "seq").code)
            self.emit(
                f"std::vector<{decl_type(el)}> {out}({src_}.begin(), {src_}.end());"
            )
            self.emit(f"std::stable_sort({out}.begin(), {out}.end());")
            return CE(out, T.vec(el))
        if (
            name in ("list", "tuple")
            and len(args) == 1
            and isinstance(args[0], ast.GeneratorExp)
        ):
            g = args[0]
            return self.ex_ListComp(
                ast.ListComp(
                    elt=g.elt,
                    generators=g.generators,
                    lineno=g.lineno,
                    col_offset=g.col_offset,
                )
            )
        if name in ("list", "tuple") and len(args) == 1:
            v = self.expr(args[0])
            if name == "tuple" and self.tuple_struct_items(v) is not None:
                # tuple() of a fixed tuple is the same tuple (a value copy)
                st_ = T.struct_of(v.t)
                return CE(f"(*{paren(v.code)})" if st_[1] else v.code, st_[0], v.pure)
            el = T.seq_elem(v.t)
            if el is None:
                raise Refuse(node, f"{name}() of {v.t}")
            t = T.vec(el) if name == "list" else T.ptuple(el)
            if v.meta and "p" in v.meta:
                return self.bind(
                    CE(f"{t}({v.meta['p']}, {v.meta['p']} + {v.meta['n']})", t, False),
                    name,
                )
            src_v = self.iter_src(v.code)
            return self.bind(CE(f"{t}({src_v}.begin(), {src_v}.end())", t, False), name)
        if name == "enumerate" or name == "zip":
            raise Refuse(node, f"{name}() outside a loop header")
        if name == "iter" and len(args) == 1:
            v = self.expr(args[0])
            el = T.seq_elem(v.t)
            if el is None:
                raise Refuse(node, f"iter() of {v.t}")
            return CE(f"py::iter<{el}>({v.code})", f"py::iter<{el}>", v.pure)
        if name == "next" and len(args) == 1 and self._is_seq_guard_counter(args[0]):
            # R-HS1: next(_SEQ_GUARDS.setdefault(b, itertools.count())), the
            # per-builder counter the C side threads as int* seq_guards
            if self.seq_guards is None:
                raise Refuse(node, "the per-builder counter needs seq_guards (R-HS1)")
            return CE(f"(int64_t)((*{self.seq_guards})++)", T.I64, False)
        if name == "next" and len(args) == 1:
            v = self.expr(args[0])
            if not v.t.startswith("py::iter<"):
                raise Refuse(node, f"next() of {v.t}")
            return CE(f"{v.code}.next()", v.t[len("py::iter<") : -1], False)
        if (
            name == "isinstance"
            and isinstance(args[1], ast.Tuple)
            and all(isinstance(e, ast.Name) for e in args[1].elts)
            and {e.id for e in args[1].elts}
            <= {"list", "tuple", "dict", "set", "frozenset"}
        ):
            # a container check, decided by the static type
            v = self.expr(args[0])
            names_ = {e.id for e in args[1].elts}
            seq = T.is_seq(v.t) or (v.meta or {}).get("n") is not None
            hit = (seq and names_ & {"list", "tuple"}) or (
                re.fullmatch(r"(?:const )?py::dict<.+>&?", v.t) and "dict" in names_
            )
            return CE("true" if hit else "false", T.BOOL, lit=bool(hit))
        if (
            name == "isinstance"
            and isinstance(args[1], ast.Name)
            and args[1].id in ("bool", "int", "str", "float")
        ):
            # decided by the static type (Python's bool is an int)
            v = self.expr(args[0])
            cls = args[1].id
            inner = T.optional_inner(v.t)
            base = inner or v.t
            if v.t != T.IV:
                if cls == "bool":
                    hit = base == T.BOOL
                elif cls == "int":
                    hit = T.is_int(base) or base == T.BOOL
                elif cls == "float":
                    hit = base == T.F64
                else:
                    hit = T.is_str(base)
                if not hit:
                    return CE("false", T.BOOL, lit=False)
                if inner:
                    return CE(f"{paren(v.code)}.has_value()", T.BOOL, v.pure)
                return CE("true", T.BOOL, lit=True)
        if (
            name == "isinstance"
            and isinstance(args[0], ast.Name)
            and self.lookup(args[0].id) is not None
            and self.union_alt(self.lookup(args[0].id).t, args[1])
        ):
            cur = self.lookup(args[0].id)
            st_ = T.struct_of(cur.t)
            f_ = self.union_alt(cur.t, args[1])
            return CE(
                f"({paren(cur.code)}{'->' if st_[1] else '.'}kind == {union_kind(st_[0], f_)})",
                T.BOOL,
                cur.pure,
            )
        if (
            name == "isinstance"
            and isinstance(args[1], ast.Name)
            and self.mod.family(args[1].id) is not None
        ):
            v = self.expr(args[0])
            st_ = T.struct_of(v.t)
            fam = self.mod.family(args[1].id)
            if st_ and st_[0] == self.mod.class_struct.get(fam.base):
                code = self.mod.kind_test(fam, args[1].id, v.code, st_[1])
                if code == "true":
                    return (
                        CE(f"({v.code}) != nullptr", T.BOOL, v.pure)
                        if st_[1]
                        else CE("true", T.BOOL, lit=True)
                    )
                return CE(f"({code})", T.BOOL, v.pure)
        if (
            name == "isinstance"
            and isinstance(args[1], ast.Name)
            and (args[1].id in self.mod.classes or args[1].id in self.mod.imports)
        ):
            v = self.expr(args[0])
            st_ = T.struct_of(v.t)
            want_s = self.mod.class_struct.get(args[1].id) or (
                self.mod.tmap.class_struct.get(args[1].id) if self.mod.tmap else None
            )
            if st_ and want_s:
                if st_[0] == want_s:
                    return (
                        CE(
                            f"({v.code}) != nullptr" if st_[1] else "true",
                            T.BOOL,
                            v.pure,
                        )
                        if st_[1]
                        else CE("true", T.BOOL, lit=True)
                    )
                return CE("false", T.BOOL, lit=False)
        if (
            name == "isinstance"
            and isinstance(args[1], ast.Name)
            and args[1].id == "VectorType"
        ):
            v = self.expr(args[0])
            if v.t != T.TYPE:
                raise Refuse(node, f"isinstance(.., VectorType) of {v.t}")
            return CE(f"{paren(v.code)}->kind == ROCKE_TYPE_VECTOR", T.BOOL, v.pure)
        if (
            name == "isinstance"
            and isinstance(args[0], ast.Name)
            and self.lookup(args[0].id) is not None
            and T.variant_items(self.lookup(args[0].id).t)
        ):
            cur = self.lookup(args[0].id)
            alt = self.variant_alt(cur.t, args[1])
            if alt is None:
                raise Refuse(node, f"isinstance on {cur.t}: no matching alternative")
            return CE(f"std::holds_alternative<{alt}>({cur.code})", T.BOOL, cur.pure)
        if name == "isinstance":
            n = self._isinstance_narrow(node)
            if n is None:
                raise Refuse(node, "isinstance outside the IntOrValue table (R1)")
            x = args[0].id
            cur = self.lookup(x)
            cls = args[1].id
            if cls == "int":
                return CE(f"({cur.code}).v == nullptr", T.BOOL)
            return CE(f"({cur.code}).v != nullptr", T.BOOL)
        if name in ("any", "all"):
            return self.any_all(node, name == "any")
        if name in ("min", "max") and len(args) == 2 and not node.keywords:
            a = self.expr(args[0])
            b = self.expr(args[1])
            if T.is_int(a.t) and T.is_int(b.t):
                return CE(
                    f"std::{name}<int64_t>({a.code}, {b.code})",
                    T.I64,
                    a.pure and b.pure,
                )
            raise Refuse(node, f"{name} of {a.t}, {b.t}")
        if (
            name in ("min", "max")
            and len(args) == 1
            and not node.keywords
            and not isinstance(args[0], ast.Starred)
        ):
            v = self.bind(self.readable(self.expr(args[0])), "seq")
            el = T.seq_elem(v.t)
            if el is None or not (T.is_int(el) or el in (T.BOOL, T.F64)):
                raise Refuse(node, f"{name}() of {v.t}")
            return CE(f"py::{name}_of({v.code})", el, v.pure)
        if (
            name in ("min", "max")
            and len(args) > 2
            and not node.keywords
            and not any(isinstance(x, ast.Starred) for x in args)
        ):
            vs = [self.bind(self.expr(x), "m") for x in args]
            if all(T.is_int(v.t) for v in vs):
                # a braced list is evaluated left to right, like Python's arguments
                return CE(
                    f"std::{name}<int64_t>({{{', '.join(v.code for v in vs)}}})",
                    T.I64,
                    all(v.pure for v in vs),
                )
            raise Refuse(node, f"{name} of {[v.t for v in vs]}")
        if name == "PtrType" and name in self.mod.imports:
            e = self.expr(args[0])
            sp = (
                self.expr(args[1])
                if len(args) > 1
                else CE('"global"', T.STR, lit="global")
            )
            if e.t != T.TYPE:
                raise Refuse(node, "PtrType arguments")
            if self.builder is None:
                raise Refuse(node, "PtrType() needs the builder's arena")
            return CE(
                f"rocke_ptr_type({self.builder}, {e.code}, {self.coerce(sp, T.STR, node)})",
                T.TYPE,
                e.pure and sp.pure,
            )
        if name == "VectorType" and name in self.mod.imports:
            e = self.expr(args[0])
            n = self.expr(args[1])
            if e.t != T.TYPE or not T.is_int(n.t):
                raise Refuse(node, "VectorType arguments")
            return CE(
                f"rocke_vector_type({self.builder}, {e.code}, (int)({n.code}))",
                T.TYPE,
                e.pure and n.pure,
            )
        if name == "abs" and len(args) == 1:
            a = self.expr(args[0])
            if T.is_int(a.t):
                return CE(f"std::llabs({a.code})", T.I64, a.pure)
        return None

    def bind_call_args(self, node, pyparams, skip=0) -> dict:
        """Signature.bind in source order; returns {pyname: CE} for passed args."""
        params = pyparams[skip:]
        pos = [p for p in params if p["kind"] in ("pos", "posonly", "pos_or_kw")]
        var = next(
            (p for p in params if p["kind"] in ("var", "var_pos", "varargs", "*")), None
        )
        out = {}
        if any(isinstance(a, ast.Starred) for a in node.args) or (
            var is not None and len(node.args) > len(pos)
        ):
            if var is None:
                raise Refuse(node, "star argument into a fixed signature (KW4)")
            fixed = node.args[: len(pos)]
            for p, a in zip(pos, fixed):
                out[p["name"]] = self.bind(self.expr(a), p["name"])
            name = self.tmp(var["name"])
            parts = []
            for a in node.args[len(pos) :]:
                if isinstance(a, ast.Starred):
                    v = self.bind(self.expr(a.value), "star")
                    if T.seq_elem(v.t) is None:
                        raise Refuse(node, f"*{v.t}")
                    parts.append(("seq", v))
                else:
                    parts.append(("one", self.bind(self.expr(a), "e")))
            els = {T.seq_elem(v.t) if k == "seq" else v.t for k, v in parts}
            if len(els) != 1:
                raise Refuse(node, f"variadic arguments of types {els}")
            el = els.pop()
            self.emit(f"std::vector<{decl_type(el)}> {name};")
            for k, v in parts:
                if k == "seq":
                    src_v = self.iter_src(v.code)
                    self.emit(
                        f"{name}.insert({name}.end(), {src_v}.begin(), {src_v}.end());"
                    )
                else:
                    self.emit(f"{name}.push_back({v.code});")
            out[var["name"]] = CE(name, T.vec(el))
            for k in node.keywords:
                out[k.arg] = self.bind(self.expr(k.value), k.arg)
            return out
        if len(node.args) > len(pos):
            raise Refuse(
                node, f"too many positional arguments for {[p['name'] for p in params]}"
            )
        for p, a in zip(pos, node.args):
            out[p["name"]] = self.bind(self.expr(a), p["name"])
        names = {p["name"] for p in params}
        for k in node.keywords:
            kn = (
                k.arg
                if k.arg in names
                else (k.arg + "_" if k.arg + "_" in names else None)
            )
            if kn is None:
                raise Refuse(node, f"unknown keyword {k.arg}")
            out[kn] = self.bind(self.expr(k.value), kn)
        for p in params:
            if (
                p.get("kind") in ("var", "var_pos", "varargs", "*")
                and p["name"] not in out
                and p.get("annotation")
            ):
                el = T.from_annotation(p["annotation"], {})
                if el in (T.V, T.TYPE) or (el and T.is_int(el)):
                    out[p["name"]] = CE(
                        f"std::vector<{el}>{{}}", T.vec(el)
                    )  # *args left empty
        return out

    def default_ce(self, text, node) -> CE:
        if text is None:
            raise Refuse(node, "missing required argument")
        v = ast.literal_eval(text) if text not in ("None",) else None
        if v is None:
            return CE("nullptr", T.NONE, lit=None)
        return CE(lit_code(v), lit_type(v), lit=v)

    def err_wrap(self, call: str, ret: str) -> str:
        """A C helper that reports a Python raise through rocke_attn_bwd_err_t:
        the call, then the recorded error thrown as its exception."""
        e, self._err = getattr(self, "_err", None), None
        if e is None:
            return call
        if ret == "void":
            return f"({call}, py_adapt::check_err_void({e}))"
        return f"py_adapt::check_err({call}, {e})"

    def c_args(
        self,
        cf,
        bound: dict,
        slots: dict,
        extra: dict,
        node,
        py_b=True,
        outs=None,
        self_arg=None,
    ) -> list[str]:
        self._err = None
        self._gen = {}  # generated-ABI outputs of this call (gen_call reads them)
        has_of = {sl.get("has"): sl for sl in slots.values() if sl.get("has")}
        args = []
        last_seq = None
        for i, (cty, cname) in enumerate(cf.params):
            if i == 0 and self_arg is not None and T.struct_of(cty):
                args.append(self_arg)
                continue
            if cty == T.B and (cname not in slots or slots[cname]["py"] not in bound):
                args.append(self.builder)
                continue
            if (
                cname == "err"
                and cty.replace(" ", "") == PKG.err_t + "*"
                and cname not in slots
            ):
                e = self.tmp("err")
                self.emit(f"{PKG.err_t} {e}{{}};")
                self._err = e
                args.append(f"&{e}")
                continue
            if cname == "seq_guards" and cty == "int*":
                if self.seq_guards is None:
                    raise Refuse(
                        node,
                        f"{cf.name} needs the builder's seq_guards counter (R-HS1)",
                    )
                args.append(self.seq_guards)
                continue
            if (
                cty.startswith("const ")
                and cty.endswith("_t*")
                and cname not in slots
                and T.struct_of(cty)
                and (
                    T.struct_of(cty)[0].endswith("_args_t")
                    or T.struct_of(cty)[0].endswith("_opts_t")
                )
            ):
                taken = {
                    sl["py"]
                    for c_, sl in slots.items()
                    if c_ in {n for _, n in cf.params}
                }
                args.append(
                    self.opts_struct(
                        T.struct_of(cty)[0],
                        {k: v for k, v in bound.items() if k not in taken},
                        node,
                    )
                )
                continue
            if outs and cname in outs:
                args.append(outs[cname])
                continue
            slot = slots.get(cname)
            if slot is not None:
                ce = bound.get(slot["py"])
                if ce is None:
                    ce = self.default_ce(slot.get("default"), node)
                if slot.get("has") and ce.t == T.NONE:
                    args.append(
                        f"{decl_type(cty)}{{}}"
                    )  # absent: its has_ flag is false
                    continue
                if T.is_seq(ce.t):
                    last_seq = ce
                args.append(self.coerce(ce, cty, node))
                continue
            role = extra.get(cname)
            if role is not None and self.gen_arg(
                role, cty, cname, args, has_of, bound, node
            ):
                continue
            if role == "count":
                if last_seq is None:
                    raise Refuse(
                        node, f"count parameter {cname} with no sequence before it"
                    )
                args.append(f"(int)py::len({last_seq.code})")
                continue
            raise Refuse(
                node, f"C parameter {cname} of {cf.name} has no Python source ({role})"
            )
        return args

    def gen_arg(self, role, cty, cname, args, has_of, bound, node) -> bool:
        """The generated-ABI parameters a Python call has no argument for."""
        g = self._gen
        if role == "owned_out":
            o = self.tmp("own")
            self.emit(f"{cty[:-1]} {o} = nullptr;")
            g.setdefault("owned", []).append((o, cty[:-2]))
            args.append(f"&{o}")
            return True
        if role == "owned_count":
            n = self.tmp("n")
            self.emit(f"int {n} = 0;")
            g["n"] = n
            args.append(f"&{n}")
            return True
        if role.startswith("fixed_out:"):
            k = int(role.split(":")[1])
            f = self.tmp("fix")
            self.emit(f"int {f}[{k}] = {{}};")
            g["fixed"] = (f, k)
            args.append(f)
            return True
        if role == "opt_out":
            o = self.tmp("opt")
            self.emit(f"{cty[:-1]} {o}{{}};")
            g["opt"] = (o, cty[:-1])
            args.append(f"&{o}")
            return True
        if role == "opt_has":
            h = self.tmp("has")
            self.emit(f"bool {h} = false;")
            g["opt_has"] = h
            args.append(f"&{h}")
            return True
        if role in ("reason", "reason_str"):
            # reason: a raise's text; reason_str: the str of an (ok, reason) result
            r = self.tmp("reason")
            self.emit(f'char {r}[512] = "";')
            g[role] = r
            g["reason_buf"] = r
            args.append(r)
            return True
        if role == "reason_cap":
            args.append(f"sizeof({g['reason_buf']})")
            return True
        if role == "str_out":
            b_ = self.tmp("buf")
            self.emit(f'char {b_}[1024] = "";')
            g["str"] = b_
            args.append(b_)
            return True
        if role == "str_cap":
            args.append(f"sizeof({g['str']})")
            return True
        if role == "has":
            sl = has_of.get(cname)
            ce = bound.get(sl["py"]) if sl else None
            if (
                ce is None
                and sl is not None
                and sl.get("default") not in (None, "None")
            ):
                ce = CE("x", "int")  # a non-None default
            args.append("false" if ce is None or ce.t == T.NONE else "true")
            return True
        return False

    def gen_call(self, cf, args, label, node) -> CE:
        """The call of a C function, then its generated-ABI outputs as the Python value."""
        call = self.err_wrap(f"{cf.name}({', '.join(args)})", cf.ret)
        g = getattr(self, "_gen", {}) or {}
        if not g:
            return self.bind(CE(call, cf.ret, False), label)
        ok = None
        if cf.ret != "void":
            ok = self.tmp("ok")
            self.emit(f"[[maybe_unused]] const {decl_type(cf.ret)} {ok} = {call};")
        else:
            self.emit(f"{call};")
        takes_b = any(t == T.B for t, _ in cf.params)
        if "reason_str" in g and ok:
            # (ok, reason) through the validity-gate ABI: the Python tuple, no raise
            return CE(
                f"std::tuple<bool, std::string>{{{ok}, std::string({g['reason_str']})}}",
                "std::tuple<bool, std::string>",
                True,
            )
        if "reason" in g and ok:
            self.emit(
                f"if({'!' + ok if cf.ret == 'bool' else ok + ' < 0'}) throw ckc::ValueError(std::string({g['reason']}));"
            )
        elif "str" in g and ok:
            self.emit(
                f'if({ok} != ROCKE_OK) throw ckc::ValueError("string result does not fit");'
            )
        elif takes_b and cf.ret == "bool" and ok:
            self.emit(f"if(!{ok}) py_adapt::raise_builder({self.builder});")
        if "owned" in g:
            outs, n = g["owned"], g["n"]
            res = self.tmp("list")
            if len(outs) == 1:
                o, el = outs[0]
                self.emit(f"std::vector<{decl_type(el)}> {res}({o}, {o} + {n});")
                self.emit(f"free({o});")
                return CE(res, T.vec(el))
            els = [el for _o, el in outs]
            item = (
                T.arr(els[0], len(els))
                if len(set(els)) == 1
                else f"std::tuple<{', '.join(els)}>"
            )
            self.emit(f"std::vector<{item}> {res};")
            self.emit(
                f"for(int i_ = 0; i_ < {n}; ++i_) {res}.push_back({decl_type(item)}{{{', '.join(o + '[i_]' for o, _e in outs)}}});"
            )
            for o, _e in outs:
                self.emit(f"free({o});")
            return CE(res, T.vec(item))
        if "fixed" in g:
            f, k = g["fixed"]
            return CE(
                f"std::array<int64_t, {k}>{{{', '.join(f'(int64_t){f}[{i}]' for i in range(k))}}}",
                T.arr(T.I64, k),
                True,
            )
        if "opt" in g:
            o, t = g["opt"]
            ot = T.optional(t)
            return CE(f"({g['opt_has']} ? {ot}({o}) : {ot}())", ot, True)
        if "str" in g:
            return CE(f"std::string({g['str']})", T.SSTR, True)
        return CE(ok, cf.ret, True) if ok else CE("", T.VOID)

    def enum_from_str(self, enum_t: str) -> str:
        """S2: a C enum from its Python string; the strings are the enumerators minus their common prefix."""
        names = self.env.headers.enums[enum_t]
        import os as _os

        pre = _os.path.commonprefix(names)
        pre = pre[: pre.rfind("_") + 1]
        fn = (
            enum_t[:-2] + "_from_str" if enum_t.endswith("_t") else enum_t + "_from_str"
        )
        body = [f"static {enum_t} {fn}(const char* s)", "{"]
        for n in names:
            body.append(
                f"    if(py::eq(s, {lit_code(n[len(pre):].lower())})) return {n};"
            )
        body.append(
            f'    throw ckc::ValueError(std::string("unknown {enum_t}: ") + py::str(s));'
        )
        body.append("}")
        self.mod.private_consts.setdefault("fn:" + fn, (fn, "\n".join(body)))
        return fn

    def call_param(self, node) -> CE:
        """b.param(name, t, **attrs): attrs (a py::attrs value) becomes rocke_param_opts_t or NULL."""
        star = [k for k in node.keywords if k.arg is None]
        if len(star) != 1 or len(node.args) != 2 or len(node.keywords) != 1:
            raise Refuse(node, "param(**attrs) shape")
        name = self.bind(self.expr(node.args[0]), "pname")
        ty = self.bind(self.expr(node.args[1]), "ptype")
        attrs = self.bind(self.expr(star[0].value), "attrs")
        if attrs.t != "py::attrs":
            raise Refuse(node, f"**{attrs.t} into param")
        o = self.tmp("opts")
        self.emit(f"rocke_param_opts_t {o}{{}};")
        return self.bind(
            CE(
                f"rocke_b_param({self.builder}, {self.coerce(name, T.STR, node)}, {ty.code}, py_adapt::param_opts({attrs.code}, &{o}))",
                T.V,
                False,
            ),
            "param",
        )

    def record_type(self, rid: str) -> str:
        return f"py_rec_{rid}_t"

    def record_set(self, obj: CE, rid: str, key: str, v: CE, node):
        """d["key"] = v on a record: the field and its has_ flag."""
        fields = self.mod.records.setdefault(rid, {})
        f = record_field(key)
        if self.dry or key not in fields:
            prev = fields.get(key)
            fields[key] = v.t if prev is None else (T.common(prev, v.t) or prev)
        ft = fields[key]
        sep = "->" if obj.t.endswith("*") else "."
        code = v.code if self.dry else self.coerce(v, ft, node)
        self.emit(f"{obj.code}{sep}has_{f} = true;")
        self.emit(f"{obj.code}{sep}{f} = {code};")

    def record_display(self, node, rid: str) -> CE:
        tmp = self.tmp("rec")
        self.mod.records.setdefault(rid, {})
        self.emit(f"{self.record_type(rid)} {tmp}{{}};")
        obj = CE(tmp, self.record_type(rid))
        for k, v in zip(node.keys, node.values):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                raise Refuse(node, "record dict with a non-literal key (R-REC)")
            self.record_set(obj, rid, k.value, self.bind(self.expr(v), "v"), node)
        return obj

    def record_of(self, ce: CE):
        m = re.fullmatch(r"(?:const )?py_rec_(\w+)_t\*?&?", ce.t)
        return m.group(1) if m else None

    def record_get(
        self, ce: CE, key: str, node, default: CE | None = None, raising=True
    ) -> CE:
        rid = self.record_of(ce)
        fields = self.mod.records.get(rid, {})
        if key not in fields:
            if self.dry:
                return CE("0", T.AUTO)
            if default is not None:
                return default
            if raising:
                raise Refuse(node, f"record key {key!r} is never set (R-REC)")
        f = record_field(key)
        ft = fields[key]
        sep = "->" if ce.t.endswith("*") else "."
        if default is not None:
            return CE(
                f"({ce.code}{sep}has_{f} ? {ce.code}{sep}{f} : {self.coerce(default, ft, node)})",
                ft,
                ce.pure and default.pure,
            )
        if not raising:
            if T.is_ptr(ft) or ft == T.STR:
                return CE(
                    f"({ce.code}{sep}has_{f} ? {ce.code}{sep}{f} : nullptr)",
                    ft,
                    ce.pure,
                )
            ot = T.optional(ft)
            return CE(
                f"({ce.code}{sep}has_{f} ? {ot}({ce.code}{sep}{f}) : {ot}())",
                ot,
                ce.pure,
            )
        return CE(
            f"({ce.code}{sep}has_{f} ? {ce.code}{sep}{f} : (throw ckc::KeyError({lit_code(repr(key))}), {ce.code}{sep}{f}))",
            ft,
            ce.pure,
        )

    def ex_Dict(self, node, want=None):
        """A dict display of literal str keys and bool/int values: keyword attributes (py::attrs)."""
        if want and want.startswith("record:"):
            return self.record_display(node, want[len("record:") :])
        if not node.keys and not want:
            return CE("{}", "dict?")  # typed by its item assignments
        if (
            want
            and T.optional_inner(want)
            and T.optional_inner(want).startswith("py::dict<")
        ):
            inner = self.ex_Dict(node, want=T.optional_inner(want))
            return CE(f"{decl_type(want)}({inner.code})", want, inner.pure)
        if want and re.fullmatch(r"py::dict<.+>", want):
            # a display of a dict whose type is known: built in source order
            if not node.keys:
                return CE(f"{want}{{}}", want)
            m = re.fullmatch(r"py::dict<(.+)>", want).group(1)
            kt, vt = [
                x.strip() for x in T._split_top(m.replace("<", "[").replace(">", "]"))
            ]
            kt, vt = kt.replace("[", "<").replace("]", ">"), vt.replace(
                "[", "<"
            ).replace("]", ">")
            d = self.tmp("dct")
            self.emit(f"{want} {d};")
            for k, v in zip(node.keys, node.values):
                kc = self.bind(self.expr(k), "k")
                vc = self.bind(self.expr(v), "v")
                self.emit(
                    f"{d}.set({self.coerce(kc, kt, node)}, {self.coerce(vc, vt, node)});"
                )
            return CE(d, want)
        if not node.keys and want == "py::kargs":
            return CE("py::kargs{}", "py::kargs")
        if node.keys and not all(
            isinstance(k, ast.Constant) and isinstance(k.value, str) for k in node.keys
        ):
            return self.dict_display_typed(node)
        items = []
        for i, (k, v) in enumerate(zip(node.keys, node.values)):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                raise Refuse(node, "dict display with non-literal keys (D4)")
            ce = self.expr(v)
            if i == 0 and T.struct_of(ce.t) and not T.struct_of(ce.t)[1]:
                return self.struct_dict_display(node, self.bind(ce, "v"))
            if not (T.is_int(ce.t) or ce.t == T.BOOL):
                raise Refuse(node, f"dict display value of {ce.t}")
            items.append(f"{{{lit_code(k.value)}, (int64_t)({ce.code})}}")
        return CE(f"py::attrs{{{{{', '.join(items)}}}}}", "py::attrs")

    def struct_dict_display(self, node, first: CE) -> CE:
        """{"q": v_q, "do": v_do}: literal str keys, values of one struct type
        (a tagged-value family value too): a py::dict of that struct, in source order.
        """
        vs = [first] + [self.bind(self.expr(v), "v") for v in node.values[1:]]
        if len({v.t for v in vs}) != 1:
            raise Refuse(node, f"dict display values of {sorted({v.t for v in vs})}")
        want = f"py::dict<{T.SSTR}, {first.t}>"
        d = self.tmp("dct")
        self.emit(f"{want} {d};")
        for k, v in zip(node.keys, vs):
            self.emit(f"{d}.set({lit_code(k.value)}, {v.code});")
        return CE(d, want)

    def dict_display_typed(self, node) -> CE:
        """D4b: a dict display of computed keys with no expected type: the items
        decide it. Keys and values are evaluated in source order (k1, v1, k2,
        ...); ints are one Python type (int64_t), strings are owned."""
        if any(k is None for k in node.keys):
            raise Refuse(node, "dict display with ** unpacking (D4)")
        ks, vs = [], []
        for k, v in zip(node.keys, node.values):
            ks.append(self.bind(self.expr(k), "k"))
            vs.append(self.bind(self.expr(v), "v"))

        def one(ts, what):
            if all(T.is_int(t) for t in ts):
                return T.I64
            if all(T.is_str(t) for t in ts):
                return T.SSTR
            if all(t in (T.BOOL, T.F64) for t in ts) and len(set(ts)) == 1:
                return ts[0]
            raise Refuse(
                node,
                f"dict display with non-literal keys (D4): {what} of {sorted(set(ts))}",
            )

        kt, vt = one([k.t for k in ks], "keys"), one([v.t for v in vs], "values")
        want = f"py::dict<{kt}, {vt}>"
        d = self.tmp("dct")
        self.emit(f"{want} {d};")
        for kc, vc in zip(ks, vs):
            self.emit(
                f"{d}.set({self.coerce(kc, kt, node)}, {self.coerce(vc, vt, node)});"
            )
        return CE(d, want)

    def opts_struct(self, sname: str, bound: dict, node) -> str:
        """D2: keyword arguments into an options struct filled from its _default()."""
        s_ = self.env.headers.structs.get(sname)
        if s_ is None:
            raise Refuse(node, f"no struct {sname}")
        stem = sname[:-2] if sname.endswith("_t") else sname
        name = self.tmp(stem.split("_")[-1])
        dflt = self.env.headers.funcs.get(stem + "_default")
        self.emit(f"{sname} {name} = {dflt.name + '()' if dflt else '{}'};")
        for k, v in bound.items():
            if k not in s_.fields:
                raise Refuse(node, f"{sname} has no field {k}")
            self.emit(f"{name}.{k} = {self.coerce(v, s_.fields[k], node)};")
            if "has_" + k in s_.fields:
                self.emit(f"{name}.has_{k} = true;")
        return f"&{name}"

    def call_ir(self, method, recv, node) -> CE:
        """b.method(...): the receiver is the builder of this call (a parameter,
        a local alias of self.b, or a field)."""
        if recv.code == self.builder or not recv.code:
            return self._call_ir(method, recv, node)
        saved = self.builder
        self.builder = recv.code
        try:
            return self._call_ir(method, recv, node)
        finally:
            self.builder = saved

    def _call_ir(self, method, recv, node) -> CE:
        if method == "inline_asm_multi" and any(
            k.arg == "result_name_hint" for k in node.keywords
        ):
            # the C entry has no name hint: the routine library builds the op
            ps = [
                {"name": n, "kind": "pos", "default": d}
                for n, d in (
                    ("template", None),
                    ("constraints", None),
                    ("operands", None),
                    ("result_types", None),
                )
            ]
            ps += [
                {"name": n, "kind": "kwonly", "default": d}
                for n, d in (
                    ("sideeffect", "True"),
                    ("convergent", "False"),
                    ("result_name_hint", "'asm'"),
                )
            ]
            bd = self.bind_call_args(node, ps)
            get = lambda n, d: bd.get(n) or self.default_ce(d, node)
            ops = self.bind(self.coerce_seq(get("operands", None), T.V, node), "ops")
            rts = self.bind(
                self.coerce_seq(get("result_types", None), T.TYPE, node), "rts"
            )
            args = [
                self.coerce(get("template", None), T.STR, node),
                self.coerce(get("constraints", None), T.STR, node),
                ops.code,
                rts.code,
                self.coerce(get("sideeffect", "True"), T.BOOL, node),
                self.coerce(get("convergent", "False"), T.BOOL, node),
                self.coerce(get("result_name_hint", "'asm'"), T.STR, node),
            ]
            return self.bind(
                CE(
                    f"py_adapt::inline_asm_multi({self.builder}, {', '.join(args)})",
                    T.vec(T.V),
                    False,
                ),
                "asm",
            )
        if method in ("scf_for", "scf_for_iter"):
            return self.call_loop(method, node)
        if method == "param" and any(k.arg is None for k in node.keywords):
            return self.call_param(node)
        if method == "scf_yield":
            vals = self.bind_call_args(
                node, [{"name": "values", "kind": "var_pos", "default": None}]
            )
            v = vals.get("values") or CE("std::vector<rocke_value_t*>{}", T.vec(T.V))
            if T.seq_elem(v.t) is None:
                v = CE(f"std::vector<rocke_value_t*>{{{v.code}}}", T.vec(T.V))
            return CE(
                f"rocke_b_scf_yield({self.builder}, {v.code}.data(), (int){v.code}.size())",
                T.VOID,
                False,
            )
        if (
            method in self.mod.fence_methods
            and not any(k.arg == "fence" for k in node.keywords)
            and not node.args
        ):
            # R-HS2: Python reads b.fence_barriers; C passes the fence explicitly
            if self.fence is not None:
                return self.bind(
                    CE(
                        f"{self.mod.fence_methods[method]}({self.builder}, {self.fence})",
                        T.VOID,
                        False,
                    ),
                    method,
                )
        e = self.env.sig.ir.get(method)
        if e is None or not e.get("c"):
            raise Refuse(node, f"builder method {method} has no C twin")
        c = e["c"]
        if c.get("opts_variants") and any(
            k.arg not in {s["py"] for s in c["slots"]} for k in node.keywords
        ):
            raise Refuse(node, f"{method} keyword needs the opts variant (D2)")
        cf = self.env.c_func(c["name"])
        if cf is None:
            raise Refuse(node, f"no C function {c['name']}")
        pyparams = [p for p in e["python"]["params"] if p["name"] != "self"]
        bound = self.bind_call_args(node, pyparams)
        slots = {s["c"]: s for s in c["slots"]}
        extra = {x["c"]: x["role"] for x in c.get("extra_c", [])}
        args = self.c_args(cf, bound, slots, extra, node)
        return self.bind(
            CE(self.err_wrap(f"{cf.name}({', '.join(args)})", cf.ret), cf.ret, False),
            method,
        )

    def call_helper(self, name, node) -> CE:
        ha = self.host_arg(node)
        if ha is not None:
            # host clones: an imported emitter called with a host builder: folded
            return self.host_fold_call(
                None,
                None,
                name,
                node,
                ha[0],
                ha[1],
                call_ir=lambda c: self.call_helper(name, c),
            )
        src, attr = self.mod.imports[name]
        modstem = src.split(".")[-1] if src else ""
        from .adapters import function_adapter

        ad = function_adapter(f"{modstem}:{attr}")
        if ad is not None:
            args = [self.bind(self.expr(a), "a") for a in node.args]
            rt_ = ad[1]
            if rt_ is None:  # a typed-by-argument adapter (itertools.pairwise)
                el_ = T.seq_elem(self.readable(args[0]).t) if args else None
                if el_ is None:
                    raise Refuse(node, f"{attr}() of {args[0].t if args else None}")
                rt_ = T.vec(T.arr(el_, 2))
                args = [self.readable(a_) for a_ in args]
            return self.bind(
                CE(
                    ad[0].format(args=", ".join(a.code for a in args), b=self.builder),
                    rt_,
                    False,
                ),
                attr,
            )
        e = self.env.sig.helpers.get(f"{modstem}:{attr}")
        if e is None:
            raise Refuse(node, f"helper {modstem}:{attr} not in sigmap")
        c = e["c"]
        cf = self.env.c_func(c["name"])
        if cf is None:
            raise Refuse(node, f"no exported C function for {c['name']}")
        pyparams = e["python"]["params"]
        bound = self.bind_call_args(node, pyparams)
        bound.pop("b", None)
        slots = {s["c"]: s for s in c["slots"]}
        extra = {x["c"]: x["role"] for x in c.get("extra_c", [])}
        outs = [n for n, r in extra.items() if r == "out"]
        ctys0 = dict((n, t) for t, n in cf.params)
        if (
            len(outs) == 1
            and T.struct_of(ctys0[outs[0]])
            and T.struct_of(ctys0[outs[0]])[1]
            and cf.ret in ("bool", "rocke_status_t", "void")
        ):
            # one struct result through an out parameter (arch_caps): a local struct
            st_ = T.struct_of(ctys0[outs[0]])[0]
            tmp = self.tmp(outs[0])
            self.emit(f"{st_} {tmp}{{}};")
            args = self.c_args(
                cf,
                bound,
                slots,
                {k: v for k, v in extra.items() if v != "out"},
                node,
                outs={outs[0]: f"&{tmp}"},
            )
            call = self.err_wrap(f"{cf.name}({', '.join(args)})", cf.ret)
            if cf.ret == "rocke_status_t":
                self.emit(
                    f'if({call} != ROCKE_OK) throw ckc::ValueError("{attr} failed");'
                )
            elif cf.ret == "bool":
                self.emit(
                    f"if(!{call}) py_adapt::raise_builder({self.builder or 'nullptr'});"
                )
            else:
                self.emit(f"{call};")
            return CE(tmp, st_, True)
        if outs:
            ret_ann = e["python"].get("returns") or ""
            if cf.ret != "void" or not (
                ret_ann.startswith("tuple") or (not ret_ann and len(outs) >= 2)
            ):
                raise Refuse(
                    node, f"{cf.name} returns through out parameters: needs an adapter"
                )
            ctys = dict((n, t) for t, n in cf.params)
            names = {}
            for o in outs:
                el = ctys[o][:-1]
                nm = self.tmp(o.replace("out_", ""))
                self.emit(f"{decl_type(el)} {nm}{{}};")
                names[o] = (nm, el)
            args = self.c_args(
                cf,
                bound,
                slots,
                {k: v for k, v in extra.items() if v != "out"},
                node,
                outs={o: "&" + names[o][0] for o in outs},
            )
            self.emit(self.err_wrap(f"{cf.name}({', '.join(args)})", "void") + ";")
            els = {names[o][1] for o in outs}
            if len(els) != 1:
                raise Refuse(node, "out parameters of different types")
            el = els.pop()
            return CE(
                f"{T.arr(el, len(outs))}{{{', '.join(names[o][0] for o in outs)}}}",
                T.arr(el, len(outs)),
                True,
                meta={"items": [names[o][0] for o in outs]},
            )
        args = self.c_args(cf, bound, slots, extra, node)
        return self.gen_call(cf, args, attr, node)

    def call_internal(self, name, node) -> CE:
        from .adapters import function_adapter

        ad = function_adapter(self.mod.qual(name))
        if ad is not None and self.mod.sigs.get(name) is None:
            # the routine library already has this function (eval_layout)
            args = [self.bind(self.expr(a), "a") for a in node.args]
            if self.builder is None and "{b}" in ad[0]:
                if len(ad) < 3:
                    raise Refuse(node, f"adapter of {name} needs the builder")
                ad = (ad[2], ad[1])  # the form that makes its own scratch builder
            return self.bind(
                CE(
                    ad[0].format(args=", ".join(a.code for a in args), b=self.builder),
                    ad[1],
                    False,
                ),
                name,
            )
        fn = self.mod.functions[name]
        if self.mod.sigs.get(name) is None:
            filled = self.call_filled(name, fn, node)
            if filled is not None:
                return filled
        if self.mod.sigs.get(name) is None and builder_param(fn) is None:
            raise Refuse(node, f"call to {name}, which is not translated")
        pyparams = py_params(fn)
        bound = self.bind_call_args(node, pyparams)
        sig = self.callee_sig(name, None, fn, bound, node)
        sig = self.ir_version(name, fn, sig, bound, node)
        args = []
        for pname, t in sig.params:
            if pname in bound:
                args.append(self.coerce(bound[pname], t, node))
            elif pname == "b" and t == T.B:
                args.append(self.builder)
            elif pname == "fence":
                args.append(self.fence or "nullptr")
            elif pname == "seq_guards":
                if self.seq_guards is None:
                    raise Refuse(node, f"{name} needs the builder's seq_guards counter")
                args.append(self.seq_guards)
            elif pname in sig.defaults:
                d = self.expr(sig.defaults[pname])
                args.append(self.coerce(d, t, node))
            else:
                raise Refuse(node, f"missing argument {pname} to {name}")
        return self.bind(CE(f"{sig.name}({', '.join(args)})", sig.ret, False), name)

    def call_filled(self, qual: str, fn, node, recv: CE | None = None) -> CE | None:
        """A call of a function (or method, `Class.method` on recv) that did not
        translate but has a hand fill (holes/<module>/<f>.cpp): the fill defines
        its C entry, so the call goes through that entry (its header signature),
        as a call into another module's C twin does. None when there is no fill
        or no C entry in this module's header."""
        import os

        from .env import holes_dir
        from .translate import abi_of

        fill = os.path.join(
            holes_dir(), self.mod.modname, qual.replace(".", "_") + ".cpp"
        )
        if not os.path.isfile(fill):
            return None
        entry, cf = abi_of(self.env, self.mod, qual)
        if cf is None:
            return None
        bound = self.bind_call_args(node, py_params(fn), skip=0 if recv is None else 1)
        slots = {s["c"]: s for s in entry["c"]["slots"]}
        extra = {x["c"]: x["role"] for x in entry["c"].get("extra_c", [])}
        self_arg = None
        if recv is not None:
            if not cf.params or not T.struct_of(cf.params[0][0]):
                raise Refuse(node, f"filled method {qual}: its C entry takes no self")
            self_arg = self.coerce(recv, cf.params[0][0], node)
        args = self.c_args(cf, bound, slots, extra, node, self_arg=self_arg)
        return self.gen_call(cf, args, qual.rsplit(".", 1)[-1], node)

    def callee_sig(self, name: str, cls: str | None, fn, bound: dict, node):
        """The internal signature a call reaches: the IR version, or (host clones) the
        host clone on H when the builder argument is a host builder of class H."""
        from .host import host_class_of

        key = f"{cls}.{name}" if cls else name
        bp = builder_param(fn)
        hcls = (
            host_class_of(self.mod, bound[bp].t)
            if bp is not None and bp in bound
            else None
        )
        if hcls is None and bp is None:
            # no builder: a host image argument (a platform object built on a host builder)
            hcls = next(
                (self.image_host(v.t) for v in bound.values() if self.image_host(v.t)),
                None,
            )
        if hcls is None:
            sig = self.mod.sigs.get(key)
            if sig is None:
                raise Refuse(
                    node,
                    f"{'method' if cls else 'call to'} {key}{' is' if cls else ', which is'} not translated",
                )
            return sig
        sig = self.mod.sigs.get(f"{key}@{hcls}")
        if sig is None:
            self.mod.host_demand.add((name, cls, hcls))
            raise Refuse(node, f"call to {key} on {hcls}, which is not translated")
        return sig

    def ir_version(self, name: str, fn, sig, bound: dict, node):
        """host clones: a function with no builder whose own version takes ints, called
        with IR values where it takes them (`_mn_k` on a coordinate the lane
        map emitted): its IR version (`py_<f>__IRBuilder`)."""
        from .host import IR_KIND, host_type

        if (
            sig.host is not None
            or builder_param(fn) is not None
            or sig.name.endswith("__" + IR_KIND)
        ):
            return sig
        for pname, t in sig.params:
            v = bound.get(pname)
            if v is None or T.V not in v.t or T.V in t or T.IV in t:
                continue
            if host_type(v.t) == host_type(t):
                got = self.mod.sigs.get(f"{name}@{IR_KIND}")
                if got is None:
                    self.mod.host_demand.add((name, None, IR_KIND))
                    raise Refuse(
                        node, f"call to {name} on {IR_KIND}, which is not translated"
                    )
                return got
        return sig

    def call_lambda(self, name, node) -> CE:
        sig = self.vars.get("sig:" + name) or (
            self.outer.vars.get("sig:" + name) if self.outer else None
        )
        if sig is None:
            raise Refuse(node, f"closure {name}")
        if node.keywords or len(node.args) != len(sig.params):
            raise Refuse(node, "closure call shape")
        args = [
            self.coerce(self.bind(self.expr(a), p), t, node)
            for a, (p, t) in zip(node.args, sig.params)
        ]
        return self.bind(CE(f"{name}({', '.join(args)})", sig.ret, False), name)

    # -------------------------------------------------------------- coercion

    def coerce(self, ce: CE, to: str, node) -> str:
        t = ce.t
        if to is None or to == T.AUTO or t == to:
            return ce.code
        if self.host_scratch and to == T.V and T.is_int(t):
            return f"rocke_b_const_i32({self.host_scratch}, {ce.code})"  # host clones: a host int into the scratch program
        if self.host and t == T.BOOL and T.is_int(to):
            return f"({to})({ce.code})"  # host clones: a compare result into a host builder method (bool is an int)
        if to == "rocke_int_or_value_t" and (T.is_int(t) or t in (T.V, T.IV)):
            # the platform's IntOrValue: an int, a Value, or the backward int | Value
            if T.is_int(t):
                return f"rocke_int_or_value_t{{false, (int64_t)({ce.code}), nullptr}}"
            if t == T.V:
                return f"rocke_int_or_value_t{{true, 0, {ce.code}}}"
            src = (
                self.bind(CE(ce.code, t, ce.pure), "iv")
                if not ce.code.isidentifier()
                else ce
            )
            return f"rocke_int_or_value_t{{{src.code}.v != nullptr, {src.code}.i, {src.code}.v}}"
        fs = self.tuple_struct_items(ce)
        if fs is not None and T.arr_len(to) == len(fs):
            el_to = T.seq_elem(to)
            return f"{decl_type(to)}{{{', '.join(self.coerce(CE(c, t), el_to, node) for c, t in fs)}}}"
        ti_to = T.tuple_items(to)
        ti_from = T.tuple_items(t) or (
            [T.seq_elem(t)] * T.arr_len(t) if T.arr_len(t) is not None else None
        )
        if ti_to and ti_from and len(ti_to) == len(ti_from) and t != to:
            # tuple into tuple: item by item
            src = (
                ce
                if (ce.pure and ce.code.isidentifier())
                else self.bind(CE(ce.code, t, False), "tup")
            )
            items = [
                self.coerce(CE(f"std::get<{i}>({src.code})", a_), b_, node)
                for i, (a_, b_) in enumerate(zip(ti_from, ti_to))
            ]
            return f"{decl_type(to)}{{{', '.join(items)}}}"
        tst = T.struct_of(to)
        if (
            tst
            and not tst[1]
            and "_tuple_" in tst[0]
            and (T.arr_len(t) is not None or T.tuple_items(t) is not None)
        ):
            # std::array / std::tuple into a generated tuple struct {f0, f1, ...}
            s_ = self.env.headers.structs.get(tst[0])
            n = T.arr_len(t) if T.arr_len(t) is not None else len(T.tuple_items(t))
            if s_ is not None and len(s_.order) == n:
                src = (
                    ce
                    if (ce.pure and ce.code.isidentifier())
                    else self.bind(CE(ce.code, t, False), "tup")
                )
                items = []
                for i, f in enumerate(s_.order):
                    it_ = (
                        T.seq_elem(t)
                        if T.arr_len(t) is not None
                        else T.tuple_items(t)[i]
                    )
                    items.append(
                        self.coerce(
                            CE(f"std::get<{i}>({src.code})", it_), s_.fields[f], node
                        )
                    )
                return f"{tst[0]}{{{', '.join(items)}}}"
        if (
            T.is_seq(t)
            and T.is_seq(to)
            and T.seq_elem(T.seq_elem(t) or "")
            and T.seq_elem(T.seq_elem(to) or "")
            and T.arr_len(to) is None
            and T.seq_elem(t) != T.seq_elem(to)
            and T.arr_len(T.seq_elem(to)) is None
        ):
            # nested containers of other kinds (tuple of vectors -> tuple of tuples)
            return f"py::retype<{decl_type(to)}>({self.readable(ce).code})"
        el_t, el_to = (
            (T.seq_elem(t), T.seq_elem(to))
            if T.is_seq(t) and T.is_seq(to)
            else (None, None)
        )
        if (
            el_t
            and el_to
            and "_tuple_" in el_t
            and T.arr_len(to) is None
            and (T.arr_len(el_to) is not None or T.tuple_items(el_to))
        ):
            # generated tuple structs (an owned list field) into a sequence of std::array / std::tuple
            s_ = self.env.headers.structs.get(el_t)
            its = T.tuple_items(el_to) or [T.seq_elem(el_to)] * T.arr_len(el_to)
            if s_ is not None and len(s_.order) == len(its):
                conv = ", ".join(
                    self.coerce(CE(f"x_.{f}", s_.fields[f]), it, node)
                    for f, it in zip(s_.order, its)
                )
                return f"py::map_to<{decl_type(to)}>({self.readable(ce).code}, [&](const {el_t}& x_) {{ return {decl_type(el_to)}{{{conv}}}; }})"
        if "None" in t and t.replace("None", "X") != t and T.is_seq(t) and T.is_seq(to):
            # a list (of lists) of None into the widened element type
            return f"py::retype<{decl_type(to)}>({ce.code})"
        if t == "set?" and to.startswith("py::set<"):
            return f"{to}{{}}"
        if t == T.IV and T.is_int(to):
            # an int | Value where Python does int arithmetic: it must be an int
            return f"({to})py_adapt::iv_int_of({ce.code})"
        if t == T.IV and to == T.V:
            # an int | Value where an IR operand is needed: Python fails on an int
            # there ('int' has no .type), so the value is checked to be a Value
            return f"py_adapt::iv_value({ce.code})"
        mt, mo = re.fullmatch(r"(?:const )?py::dict<(.+)>&?", t), re.fullmatch(
            r"py::dict<(.+)>", to
        )
        if mt and mo and t.replace("const ", "").rstrip("&") != to:
            kt_t, vt_t = [
                x.strip().replace("[", "<").replace("]", ">")
                for x in T._split_top(mt.group(1).replace("<", "[").replace(">", "]"))
            ]
            kt_o, vt_o = [
                x.strip().replace("[", "<").replace("]", ">")
                for x in T._split_top(mo.group(1).replace("<", "[").replace(">", "]"))
            ]
            src = self.bind(CE(ce.code, t, ce.pure), "dsrc")
            d_ = self.tmp("dconv")
            self.emit(f"{to} {d_};")
            self.emit(
                f"for(const auto& kv_ : {src.code}.items()) {d_}.set({self.coerce(CE('kv_.first', kt_t), kt_o, node)}, {self.coerce(CE('kv_.second', vt_t), vt_o, node)});"
            )
            return d_
        if T.variant_items(to) and t in T.variant_items(to):
            return f"{decl_type(to)}({ce.code})"
        if T.variant_items(to):
            for a in T.variant_items(to):
                try:
                    return f"{decl_type(to)}({self.coerce(ce, a, node)})"
                except Refuse:
                    continue
        if t == "dict?" and to == "py::attrs":
            return "py::attrs{{}}"  # the empty attribute list, as the display writes it
        if t == "dict?" and (to.startswith("py::dict<") or to == "py::kargs"):
            return f"{to}{{}}"
        if t == "list?" and T.is_seq(to):
            return f"{decl_type(to)}{{}}"
        if t in (T.NONE, "std::nullptr_t"):
            # N-R3: None, also as an item of a tuple display (typed std::nullptr_t there)
            if T.is_ptr(to) or to == T.STR:
                return "nullptr"
            if "std::monostate" in (T.variant_items(to) or []):
                return f"{decl_type(to)}(std::monostate{{}})"  # tagged values: None in a union
            if T.optional_inner(to):
                return f"{decl_type(to)}()"  # None: an empty optional of that type
            raise Refuse(node, f"None into {to}")
        if (
            T.optional_inner(t)
            and not T.optional_inner(to)
            and T.common(T.optional_inner(t), to) == to
        ):
            return f"(*{paren(ce.code)})"  # reached only when present (a or b, narrowed tests)
        if T.optional_inner(to) and not T.optional_inner(t):
            inner_to = T.optional_inner(to)
            return f"{decl_type(to)}({self.coerce(ce, inner_to, node)})"  # explicit: ternary arms must agree
        if to in (f"const {T.IV}*", f"const {T.IV}[]") and T.seq_elem(t) is not None:
            # a tuple of int | Value into a C array of iv_t: convert element by element
            el = T.seq_elem(t)
            src = (
                self.bind(CE(ce.code, t, ce.pure), "seq")
                if not ce.code.isidentifier()
                else ce
            )
            n = T.arr_len(t)
            if n is None:
                raise Refuse(node, f"{t} into an iv_t array needs a fixed length")
            tmp = self.tmp("ivs")
            items = ", ".join(
                self.coerce(CE(f"{src.code}[{i}]", el), T.IV, node) for i in range(n)
            )
            self.emit(f"const {T.IV} {tmp}[{n}] = {{{items}}};")
            return tmp
        if to == f"const {T.IV}*" and (t in (T.V, T.NONE) or T.is_int(t) or t == T.IV):
            # an Optional[int | Value] argument: a pointer to a temporary, NULL for None
            if t == T.NONE:
                return "nullptr"
            tmp = self.tmp("iv")
            inner = self.coerce(ce, T.IV, node)
            self.emit(f"const {T.IV} {tmp} = {inner};")
            if t == T.V:
                return f"({ce.code} != nullptr ? &{tmp} : nullptr)"
            return f"&{tmp}"
        if to == T.IV:
            if t == T.V:
                return f"{PKG.c}iv_val({ce.code})"
            if T.is_int(t):
                return f"{PKG.c}iv_int({ce.code})"
        if T.is_int(to) and T.is_int(t):
            if to == T.I64 or ce.lit is not None or t == to:
                return ce.code
            return f"({to}){paren(ce.code)}"
        if to == T.BOOL and t == T.BOOL:
            return ce.code
        if to in ("double", "float") and (T.is_int(t) or t == T.F64):
            return ce.code
        if to == T.STR and t == T.SSTR:
            if self.returning:
                # the std::string dies with this function: intern its text
                return f"py::keep({ce.code})"
            return f"({ce.code}).c_str()"
        if to == T.STR and t == "const rocke_mma_op_t*":
            # b.mma(op, ...) takes an MmaOp or its op_id; the C builder the op_id
            return f"{paren(ce.code)}->op_id"
        if to == T.SSTR and t == T.STR:
            return f"std::string({ce.code})"
        if to in self.env.headers.enums and t == T.STR and ce.lit is not None:
            e = self.env.enum_for_string(to, ce.lit)
            if e:
                return e
        if to in self.env.headers.enums and T.is_str(t):
            return f"{self.enum_from_str(to)}({ce.code if t == T.STR else ce.code + '.c_str()'})"
        el = T.seq_elem(t)
        if (
            el is not None
            and T.is_int(el)
            and to in ("const int*", "int*")
            and el != "int"
        ):
            tmp = self.tmp("ints")
            src_ce = self.iter_src(ce.code)
            self.emit(f"std::vector<int> {tmp}({src_ce}.begin(), {src_ce}.end());")
            return f"{tmp}.data()"
        if (
            el is not None
            and to.endswith("*")
            and to.replace("const ", "").rstrip("*").replace("* ", "*").strip()
            in (el, el.replace("const ", ""))
        ):
            return f"{ce.code}.data()"
        if el is not None and (to.endswith("* const*") or to.endswith("**")):
            return f"{ce.code}.data()"
        if (
            T.is_seq(t)
            and T.is_seq(to)
            and T.arr_len(to) is None
            and T.is_str(T.seq_elem(t) or "")
            and T.is_str(T.seq_elem(to) or "")
            and T.seq_elem(t) != T.seq_elem(to)
        ):
            # strings of the other spelling (const char* <-> std::string): element by element
            src_w = self.iter_src(ce.code)
            if T.seq_elem(to) == T.STR:
                raise Refuse(
                    node,
                    "std::string list into a const char* list (the pointers would dangle)",
                )
            return f"{decl_type(to)}({src_w}.begin(), {src_w}.end())"
        if (
            T.arr_len(to) is not None
            and T.arr_len(t) == T.arr_len(to)
            and T.is_int(T.seq_elem(t) or "")
            and T.is_int(T.seq_elem(to) or "")
            and T.seq_elem(t) != T.seq_elem(to)
        ):
            src = (
                ce
                if (ce.pure and ce.code.isidentifier())
                else self.bind(CE(ce.code, t, False), "arr")
            )
            return f"{decl_type(to)}{{{', '.join(f'({T.seq_elem(to)}){src.code}[{i}]' for i in range(T.arr_len(to)))}}}"
        if (
            T.is_seq(t)
            and T.is_seq(to)
            and T.is_int(T.seq_elem(t) or "")
            and T.is_int(T.seq_elem(to) or "")
            and T.seq_elem(t) != T.seq_elem(to)
        ):
            # ints of another width: element by element (Python ints are one type)
            src_w = self.iter_src(ce.code)
            return f"{decl_type(to)}({src_w}.begin(), {src_w}.end())"
        if (
            T.is_seq(t)
            and T.arr_len(to) is not None
            and T.arr_len(t) is None
            and (T.seq_elem(t) == T.seq_elem(to) or T.seq_elem(t) in (T.AUTO, None))
        ):
            # a list into a fixed tuple: its length must match (Python unpacks it so)
            return (
                f"py::to_array<{decl_type(T.seq_elem(to))}, {T.arr_len(to)}>({ce.code})"
            )
        if T.is_seq(t) and T.is_seq(to):
            if T.seq_elem(t) == T.seq_elem(to) or T.seq_elem(t) in (T.AUTO, None):
                src_ce = (
                    self.iter_src(ce.code)
                    if not ce.code.startswith(decl_type(t) + "{")
                    else ce.code
                )
                return (
                    f"{decl_type(to)}({src_ce}.begin(), {src_ce}.end())"
                    if not ce.code.startswith(decl_type(t) + "{")
                    else ce.code.replace(decl_type(t), decl_type(to), 1)
                )
            if T.arr_len(to) is not None and ce.code.startswith("std::array<"):
                inner = ce.code[ce.code.index("{") :]
                return f"{decl_type(to)}{inner}"
        uf = union_fields(self.env, to)
        if uf is not None and t not in (to, f"const {to}*"):
            # tagged values: a value of one alternative into the tagged union struct
            for f_, ft in uf.items():
                try:
                    v = self.coerce(ce, ft, node)
                except Refuse:
                    continue
                return f"{to}{{.kind = {union_kind(to, f_)}, .{f_} = {v}}}"
        sa = T.struct_of(t)
        sb = T.struct_of(to)
        if sa and sb and sa[0] == sb[0]:
            if sa[1] == sb[1]:
                return ce.code
            if sb[1] and not sa[1]:
                return f"&({ce.code})"
            return f"*({ce.code})"
        if t == T.V and to == "rocke_value_t* const":
            return ce.code
        raise Refuse(node, f"cannot pass {t} as {to}")


def closure_key(site) -> str:
    """The unit name of a lifted closure (its qualname)."""
    return f"{site.name}.<locals>.{site.node.name}"


def builder_param(fn: ast.FunctionDef) -> str | None:
    """The IR builder parameter of fn (annotated IRBuilder, or `b` by naming)."""
    a = fn.args
    for x in a.posonlyargs + a.args + a.kwonlyargs:
        if x.arg in ("self", "cls"):
            continue
        ann = ast.unparse(x.annotation) if x.annotation is not None else ""
        if ann == "IRBuilder" or (
            x.arg == "b" and (not ann or re.fullmatch(r"\s*Any\s*", ann))
        ):
            return x.arg
    return None


def paren(code: str) -> str:
    return code if is_atom(code) else f"({code})"


def is_atom(code: str) -> bool:
    """A name, literal, call or member access: needs no parentheses."""
    c = code.strip()
    if c.replace("_", "").replace(".", "").replace("->", "").isalnum():
        return True
    if c.startswith('"') and c.endswith('"') and c.count('"') == 2:
        return True
    depth = 0
    for i, ch in enumerate(c):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif (
            depth == 0
            and ch in " +-*/%<>=!&|?:^"
            and not (ch == "-" and c[i : i + 2] == "->")
            and not (ch == ">" and c[i - 1 : i + 1] == "->")
            and not (ch == ":" and (c[i : i + 2] == "::" or c[i - 1 : i + 1] == "::"))
        ):
            return False
    return depth == 0


def negate(code: str) -> str:
    for a, b in ((" == ", " != "), (" != ", " == ")):
        if (
            code.count(a) == 1
            and code.count("&&") == 0
            and code.count("||") == 0
            and code.count(" < ") == 0
            and code.count(" > ") == 0
        ):
            l, r = code.split(a)
            if is_atom(l) and is_atom(r):
                return f"{l}{b}{r}"
    return f"!{code}" if is_atom(code) else f"!({code})"


def dataclasses_is(c) -> bool:
    import dataclasses

    return dataclasses.is_dataclass(c)


def decl_type(t: str) -> str:
    if t == T.STR:
        return "const char*"
    if "None" in t:
        return re.sub(r"\bNone\b", "std::nullptr_t", t)  # None inside a container
    return t


def platform_c_method(env: Env, sname: str, attr: str):
    """PM-R1: the C twin of method attr on platform struct sname by the
    header's naming convention (rocke_<stem>_<attr>, or <prefix>_<attr> when
    the header says "Class -> sname and <prefix>_*"), its first parameter a
    pointer to sname, confirmed by the doc comment above it ("Mirrors
    <Class>.<attr>" or starting "<attr>("). None if there is none."""
    hdr = env.headers
    sfile = hdr.typedef_file.get(sname)
    prefixes = [sname[:-2]]
    if sfile and hdr.paths.get(sfile):
        prefixes += re.findall(
            r"->\s*" + re.escape(sname) + r"\s+and\s+(\w+?)_\*",
            open(hdr.paths[sfile]).read(),
        )
    for pre in prefixes:
        cf = env.c_func(f"{pre}_{attr}")
        if (
            cf is not None
            and cf.params
            and T.struct_of(cf.params[0][0]) == (sname, True)
        ):
            break
    else:
        return None
    path = hdr.paths.get(cf.file)
    doc = (
        " ".join(open(path).read().splitlines()[max(0, cf.line - 12) : cf.line - 1])
        if path
        else ""
    )
    doc = doc[doc.rfind("/*") :] if "/*" in doc else ""
    if not (
        re.search(r"Mirrors\s+\w+\." + re.escape(attr) + r"\b", doc)
        or re.match(r"/\*+\s*" + re.escape(attr) + r"\(", doc)
    ):
        return None
    return cf


def py_params(fn: ast.FunctionDef) -> list[dict]:
    a = fn.args
    out = []
    n_def = len(a.defaults)
    plain = a.posonlyargs + a.args
    for i, x in enumerate(plain):
        d = a.defaults[i - (len(plain) - n_def)] if i >= len(plain) - n_def else None
        out.append(
            {
                "name": x.arg,
                "kind": "pos",
                "default": ast.unparse(d) if d is not None else None,
                "annotation": ast.unparse(x.annotation) if x.annotation else "",
            }
        )
    for x, d in zip(a.kwonlyargs, a.kw_defaults):
        out.append(
            {
                "name": x.arg,
                "kind": "kwonly",
                "default": ast.unparse(d) if d is not None else None,
                "annotation": ast.unparse(x.annotation) if x.annotation else "",
            }
        )
    return out


def split_block_locals(fn: ast.FunctionDef, name: str) -> ast.FunctionDef | None:
    """tagged values (locals): a local bound to values of different classes in blocks
    that each end in a return or raise is one C++ variable per block: the
    block's binding never flows out of it. Returns a renamed copy of fn, or
    None when no block qualifies. Each qualifying block's first use of the
    name is a plain store whose value does not read it."""
    fn = copy.deepcopy(fn)
    taken = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {
        a.arg for a in ast.walk(fn) if isinstance(a, ast.arg)
    }
    k = 0
    for st in ast.walk(fn):
        if not isinstance(st, ast.If):
            continue
        for body in (st.body, st.orelse):
            if not body or not isinstance(body[-1], (ast.Return, ast.Raise)):
                continue
            first = next(
                (
                    s
                    for s in body
                    if any(
                        isinstance(n, ast.Name) and n.id == name for n in ast.walk(s)
                    )
                ),
                None,
            )
            if not (
                isinstance(first, ast.Assign)
                and len(first.targets) == 1
                and isinstance(first.targets[0], ast.Name)
                and first.targets[0].id == name
            ):
                continue
            if any(
                isinstance(n, ast.Name) and n.id == name for n in ast.walk(first.value)
            ):
                continue
            if any(
                isinstance(s, (ast.FunctionDef, ast.Lambda))
                for b in body
                for s in ast.walk(b)
            ):
                continue
            k += 1
            new = f"{name}_{k}"
            while new in taken:
                k += 1
                new = f"{name}_{k}"
            taken.add(new)
            for s in body:
                for n in ast.walk(s):
                    if isinstance(n, ast.Name) and n.id == name:
                        n.id = new
    return fn if k else None

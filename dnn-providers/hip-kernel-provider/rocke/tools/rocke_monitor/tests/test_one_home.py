# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""One home per default: no code of the package repeats a default the registry holds.

Each package, deploy and state key has its default in one place (rkmon_lib/registry.py takes them from
config.defaults.json and the schemas), and every reader takes it from there: registry.value, registry.number,
registry.default, or a function of the module that owns the key. A scan of the source's syntax trees finds a literal
that repeats a key's default beside a read of that key: the fallback of a `get`, the right side of an `or`, the other
branch of a conditional, a literal argument beside the key's dotted name, a keyword argument or a parameter named as the
key, and a table that holds two or more of one section's keys at their defaults. A module constant counts as its
literal. Neutral values (None, False, 0, "" and the empty collections) are no default of their own and never count. A
fix's edit (checks.Edit, or a helper named edit that makes one) is no read: the value it names is the one a person sets.

The compose file keeps no default of a value docker/.env always gives: compose takes such a variable as required
(${NAME:?...}), so a .env without it stops compose with the way on, and the defaults webserve knows compose by are
compose's own.
"""

import ast
import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import (  # noqa: E402
    deploy_plan,
    file_types,
    registry,
    settings_file,
    webserve,
)

SOURCES = (
    "rkmon.py",
    "rkmon_lib/*.py",
    "worker/*.py",
    "tools/*.py",
    "docker/*.py",
    "docker/rkmon",
)
# The files the scan leaves out, with why.
EXEMPT = {
    "rkmon_lib/from_layout1.py": "it translates a layout 1 file by that layout's own meanings, its defaults included",
}
NOTHING = object()
# The calls and methods that convert the value they are given: a read inside one is still a read.
CONVERSIONS = {"str", "int", "float", "bool", "list", "tuple", "dict", "sorted"}
# The calls that make a fix's edit (checks.Edit): no read of the key they name.
EDITS = {"Edit", "edit"}
METHODS = {"strip", "lower", "upper", "rstrip", "lstrip", "split"}
REPEAT = "# repeat"  # the planted source's mark on each line the scan must report
COMPOSE = ROOT / "docker" / "compose.yaml"
# A variable compose reads: ${NAME}, ${NAME:-DEFAULT} or ${NAME:?MESSAGE}: its name, its form ("-", "?" or none) and
# the rest.
VARIABLE = re.compile(r"\$\{(\w+)(?::([-?])([^}]*))?\}")


def neutral(value) -> bool:
    """A value that is no default of its own: None, False, 0 and the empty ones."""
    if value is None or value is False:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 0
    if isinstance(value, (str, list, tuple, dict, set, frozenset)):
        return len(value) == 0
    return False


def _plain(value):
    if isinstance(value, (list, tuple)):
        return [_plain(x) for x in value]
    if isinstance(value, dict):
        return {k: _plain(x) for k, x in value.items()}
    return value


def same(a, b) -> bool:
    """A and B are one value: a bool matches only a bool, a number any equal number, a tuple its list."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    a, b = _plain(a), _plain(b)
    return type(a) is type(b) and a == b


def literal(node):
    """The value of a literal NODE, else NOTHING."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return NOTHING


def constants(tree: ast.Module) -> dict:
    """The module's constants with a literal value: {name: value}."""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            pairs = [(target, value)]
            if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):
                pairs = list(zip(target.elts, value.elts))
            for name, v in pairs:
                v = literal(v)
                if isinstance(name, ast.Name) and v is not NOTHING:
                    out[name.id] = v
    return out


class Keys:
    """The registry's keys that have a default of their own, by dotted name and by their last part."""

    def __init__(self):
        self.by_name, self.by_leaf = {}, {}
        for k in registry.keys():
            if k.default is registry.MISSING or neutral(k.default):
                continue
            self.by_name[k.name] = k
            self.by_leaf.setdefault(k.name.rsplit(".", 1)[-1], []).append(k)
        self.names = {k.name for k in registry.keys()}

    def candidates(self, read: tuple) -> list:
        """The keys a read (section, leaf) may be: the one key when its section is known, else each key of that
        leaf."""
        section, leaf = read
        if section is None:
            return list(self.by_leaf.get(leaf, ()))
        name = f"{section}.{leaf}" if section else leaf
        return [self.by_name[name]] if name in self.by_name else []


class Scan:
    """The repeats in one module: "<file>:<line>: <key> default <value> [<form>]"."""

    def __init__(self, keys: Keys):
        self.keys = keys

    def module(self, rel: str, tree: ast.Module) -> list[str]:
        self.rel, self.out, self.consts = rel, [], constants(tree)
        functions = [
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        self.function_of = {}
        for fn in functions:
            for node in ast.walk(fn):
                self.function_of[id(node)] = fn
        self.local_reads = {}
        for fn in functions:
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                ):
                    r = self.read(node.value)
                    if r is not None:
                        self.local_reads[
                            (self.function_of.get(id(node)), node.targets[0].id)
                        ] = r
        for node in ast.walk(tree):
            self.visit(node)
        return sorted(set(self.out))

    def value(self, node):
        """The literal NODE stands for (a module constant's included), else NOTHING."""
        if isinstance(node, ast.Name):
            return self.consts.get(node.id, NOTHING)
        if isinstance(node, (ast.Attribute, ast.Call)):
            return NOTHING
        return literal(node)

    @staticmethod
    def unwrap(node):
        while True:
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in CONVERSIONS
                and len(node.args) == 1
            ):
                node = node.args[0]
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in METHODS
                and not node.args
            ):
                node = node.func.value
            else:
                return node

    def section(self, node):
        """The registry section a mapping NODE is, when the scan can tell (cfg.get("worker") or {}), else None."""
        node = self.unwrap(node)
        if (
            isinstance(node, ast.BoolOp)
            and isinstance(node.op, ast.Or)
            and len(node.values) == 2
            and isinstance(node.values[1], ast.Dict)
            and not node.values[1].keys
        ):
            node = node.values[0]
        r = self.read(node)
        if r is None or r[0] is None:
            return None
        name = f"{r[0]}.{r[1]}" if r[0] else r[1]
        return name if registry.kind(name) == registry.SECTION else None

    def read(self, node):
        """(section, leaf) of the key NODE reads, section None when the mapping is not known; None for no read."""
        node = self.unwrap(node)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
        ):
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                return (self.section(node.func.value), node.args[0].value)
            return None
        if isinstance(node, ast.Subscript):
            s = node.slice
            if isinstance(s, ast.Constant) and isinstance(s.value, str):
                return (self.section(node.value), s.value)
            return None
        if isinstance(node, ast.Call):
            for a in node.args:
                r = self.named(a)
                if r is not None:
                    return r
            return None
        if isinstance(node, ast.Name):
            return self.local_reads.get((self.function_of.get(id(node)), node.id))
        return None

    def named(self, node):
        """(section, leaf) of a key's dotted name given as a literal, else None."""
        if (
            isinstance(node, ast.Constant)
            and node.value in self.keys.names
            and "." in node.value
        ):
            section, _, leaf = node.value.rpartition(".")
            return (section, leaf)
        return None

    def check(self, line: int, read, value_node, form: str) -> None:
        if read is None:
            return
        v = self.value(value_node)
        if v is NOTHING or neutral(v):
            return
        for k in self.keys.candidates(read):
            if same(v, k.default):
                self.out.append(
                    f"{self.rel}:{line}: {k.name} default {k.default!r} [{form}]"
                )
                return

    def visit(self, node) -> None:
        if isinstance(node, ast.Call):
            self.call(node)
        elif isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            for v in node.values[:-1]:
                self.check(node.lineno, self.read(v), node.values[-1], "or")
        elif isinstance(node, ast.IfExp):
            test = node.test.left if isinstance(node.test, ast.Compare) else node.test
            for r in (self.read(node.body), self.read(test)):
                self.check(node.lineno, r, node.orelse, "conditional")
            self.check(node.lineno, self.read(node.orelse), node.body, "conditional")
        elif isinstance(node, ast.Dict):
            self.table(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            positional = a.posonlyargs + a.args
            pairs = list(
                zip(positional[len(positional) - len(a.defaults) :], a.defaults)
            )
            pairs += [
                (x, d) for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None
            ]
            for arg, d in pairs:
                self.check(d.lineno, (None, arg.arg), d, "parameter")

    def call(self, node: ast.Call) -> None:
        f = node.func
        name = getattr(f, "id", None) or getattr(f, "attr", None)
        if name in EDITS:
            return
        if isinstance(f, ast.Attribute) and f.attr == "get":
            if len(node.args) == 2:
                plain = ast.Call(func=f, args=node.args[:1], keywords=[])
                self.check(node.lineno, self.read(plain), node.args[1], "get")
            return
        for kw in node.keywords:
            if kw.arg:
                self.check(node.lineno, (None, kw.arg), kw.value, "keyword")
        reads = {i: self.read(a) or self.named(a) for i, a in enumerate(node.args)}
        reads = {i: r for i, r in reads.items() if r is not None}
        others = [a for i, a in enumerate(node.args) if i not in reads]
        others += [kw.value for kw in node.keywords]
        for r in reads.values():
            for o in others:
                if not isinstance(self.value(o), bool):
                    self.check(node.lineno, r, o, "argument")

    def table(self, node: ast.Dict) -> None:
        """A dict of keys at their defaults: a dotted name's own, or two or more of one section's, at least half the
        table's values that are no neutral ones."""
        groups, counted = {}, 0
        for kn, vn in zip(node.keys, node.values):
            if not (isinstance(kn, ast.Constant) and isinstance(kn.value, str)):
                continue
            v = self.value(vn)
            if v is NOTHING or neutral(v):
                continue
            counted += 1
            k = self.keys.by_name.get(kn.value)
            if "." in kn.value and k is not None and same(v, k.default):
                self.out.append(
                    f"{self.rel}:{kn.lineno}: {k.name} default {k.default!r} [table]"
                )
                continue
            for k in self.keys.by_leaf.get(kn.value, ()):
                if same(v, k.default):
                    groups.setdefault(k.name.rpartition(".")[0], []).append((kn, k))
        for hits in groups.values():
            if len(hits) >= 2 and 2 * len(hits) >= counted:
                for kn, k in hits:
                    self.out.append(
                        f"{self.rel}:{kn.lineno}: {k.name} default {k.default!r} [table]"
                    )


def sources() -> list[pathlib.Path]:
    out = []
    for pattern in SOURCES:
        out += sorted(ROOT.glob(pattern))
    return out


class TestOneHome(unittest.TestCase):
    def test_no_code_repeats_a_default(self):
        scan, found = Scan(Keys()), []
        for path in sources():
            rel = path.relative_to(ROOT).as_posix()
            if rel not in EXEMPT:
                found += scan.module(rel, ast.parse(path.read_text(encoding="utf-8")))
        self.assertEqual(
            found,
            [],
            "a default repeated outside its home: read the key through the registry (registry.value, registry.number,"
            " registry.default) or its owner's function\n" + "\n".join(found),
        )

    def test_the_sources_and_the_exempt_files_are_there(self):
        files = {path.relative_to(ROOT).as_posix() for path in sources()}
        self.assertGreater(len(files), 100)
        for rel in (
            "rkmon.py",
            "docker/rkmon",
            "worker/drafter.py",
            "tools/make_package.py",
            *EXEMPT,
        ):
            self.assertIn(rel, files)

    def test_each_form_of_a_repeat_is_found(self):
        top, section, leaf = "draft_parallel_max", "worker", "max_hangs"
        d, dl = registry.default(top), registry.default(f"{section}.{leaf}")
        other = registry.default(f"{section}.max_resends")
        planted = f"""
LIMIT = {d!r}


def by_get(cfg):
    return cfg.get({top!r}, {d!r})  {REPEAT}


def by_or(cfg):
    return cfg.get({top!r}) or LIMIT  {REPEAT}


def by_section(cfg):
    w = cfg.get({section!r}) or {{}}
    return int(w.get({leaf!r}) or {dl!r})  {REPEAT}


def by_conditional(cfg):
    return cfg[{top!r}] if {top!r} in cfg else {d!r}  {REPEAT}


def by_name(cfg):
    return number(cfg, "{section}.{leaf}", int, {dl!r})  {REPEAT}


def by_keyword(cfg):
    return run(cfg, {leaf}={dl!r})  {REPEAT}


def by_parameter({leaf}={dl!r}):  {REPEAT}
    return {leaf}


TABLE = {{"{leaf}": {dl!r}, "max_resends": {other!r}}}  {REPEAT}


def no_repeat(cfg):
    a = cfg.get({top!r}, {d + 1!r})
    b = cfg.get({top!r}) or 0
    c = value(cfg, {top!r})
    return a, b, c


def a_fix_s_edits(file):
    return edit("{section}.{leaf}", {dl!r}), Edit("{section}.{leaf}", file, {dl!r})
"""
        found = Scan(Keys()).module("planted.py", ast.parse(planted))
        lines = {int(f.split(":")[1]) for f in found}
        marked = {
            i for i, text in enumerate(planted.splitlines(), 1) if text.endswith(REPEAT)
        }
        self.assertEqual(lines, marked, "\n".join(found))
        self.assertTrue(all(f.startswith("planted.py:") for f in found))
        for form in (
            "get",
            "or",
            "conditional",
            "argument",
            "keyword",
            "parameter",
            "table",
        ):
            self.assertTrue(any(f.endswith(f"[{form}]") for f in found), form)


class TestCompose(unittest.TestCase):
    def variables(self) -> dict:
        """Each variable compose.yaml reads outside its comments: {name: {(form, rest)}}."""
        out = {}
        for line in COMPOSE.read_text(encoding="utf-8").splitlines():
            if not line.lstrip().startswith("#"):
                for name, form, rest in VARIABLE.findall(line):
                    out.setdefault(name, set()).add((form, rest))
        return out

    def test_compose_requires_what_docker_env_always_gives(self):
        """docker/.env has each variable the deploy plan makes, the team dashboard's only while it is served:
        compose.yaml requires each of the others, or reads it with no default of its own.
        """
        doc = settings_file.loads(
            settings_file.template(file_types.DEPLOY, "review-only"), file_types.DEPLOY
        )
        values = deploy_plan.settings(doc)
        self.assertFalse(deploy_plan.served(values))
        made = deploy_plan.env_values(values, deploy_plan.Host(1000, 1000, ""))
        found = self.variables()
        for name in made:
            for form, rest in found.get(name, ()):
                with self.subTest(name=name):
                    self.assertTrue(
                        form == "?" or not rest,
                        f"{name}: compose's {form or 'plain'} form {rest!r}",
                    )
        for name in (
            "COMPOSE_PROJECT_NAME",  # the compose project's name: none before the first confirm
            "RKMON_IMAGE",
            "RKMON_TAG",
            "RKMON_CPUS",
            "RKMON_MEMORY",
            "RKMON_PORT",
            "RKMON_ADMIN_HOST",
        ):
            self.assertEqual({form for form, _rest in found[name]}, {"?"}, name)

    def test_webserve_knows_compose_s_defaults(self):
        """webserve compares the container's variables with the settings, a blank one at compose's default."""
        found = self.variables()
        known = {var: default for var, _want, default in webserve.publish_env({})}
        known.update(webserve.PUBLISH_DEFAULTS)
        self.assertEqual(
            set(known),
            {"RKMON_ADMIN_HOST", "RKMON_PORT", "RKMON_TEAM_HOST", "RKMON_TEAM_PORT"},
        )
        for var, default in known.items():
            with self.subTest(var=var):
                self.assertEqual(
                    {rest if form == "-" else "" for form, rest in found[var]},
                    {default},
                )


if __name__ == "__main__":
    unittest.main()

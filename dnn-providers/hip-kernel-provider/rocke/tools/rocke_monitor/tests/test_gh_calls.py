# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Every GitHub call of the monitor and of the writer is in gh.CALLS once, with the permissions it needs, and each
GitHub login has in the secrets registry the permissions its calls need.

The code is read, never run (ast):
- Each call is found with what it calls: `GET path` for gh.rest and gh.raw; for each argv of gh (gh.run_read,
  gh.exec_read, gh.write by way of post._gh_write, the probe's own and gh.login_with_token's (OWN_EXEC), and the gh
  the files of tests/test_gh_split.py's OWN_GH start themselves), `METHOD path` for gh api and else `gh` and the
  command's two words. A path or an argv held in a name is followed to its values: its assignments, a loop's items,
  the arguments each call gives a parameter.
  The table of the commands that send is data (rkmon_lib/outward.py, which loads with the standard library alone): a
  call of its `names` with known columns, and a name of its module or of rkmon_lib/writer.py, which builds the
  writer's lists from it, are read from the module itself (TABLES).
  Each GraphQL query is a module's constant, and is the call `graphql MODULE.NAME`.
- A call that is not in gh.CALLS fails, and so does an entry that no code makes.
- The writer's reads: the broker's rk reads with the write login (gh.login_dirs), so these reads are made by both
  logins: those of rkmon.py's code that runs for a command the broker runs (writer.COMMANDS, never with
  --premises), followed from main (an if on args.cmd, with ==, !=, in, not in or
  startswith, runs a branch for the commands its test lets in; after a return, the commands that returned run
  nothing more); those of a function of post.py or rkmon_lib/gpu.py that writes (it calls post._gh_write, or
  another such function); and those of every function these call or name, all the way down. A method called on an
  object that is not `self` is not followed, nor a function's own variable (a parameter, a name it stores) that has
  a function's name.
- The registry's GitHub logins have the permissions gh.permissions gives their side.
"""

import ast
import collections
import pathlib
import re
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from rkmon_lib import gh, outward, writer  # noqa: E402
from rkmon_lib import secrets_registry as sr  # noqa: E402
from test_gh_split import OWN_GH, python_files  # noqa: E402

GH_PY, POST_PY, GPU_PY = "rkmon_lib/gh.py", "rkmon_lib/post.py", "rkmon_lib/gpu.py"
WRITER_PY, OUTWARD_PY = "rkmon_lib/writer.py", "rkmon_lib/outward.py"
# The modules whose names are read from the module itself: the table of the commands that send, and the writer's lists
# built from it
TABLES = {OUTWARD_PY: outward, WRITER_PY: writer}
RK_PY = "rkmon.py"  # rk: its main runs each command, picked by args.cmd
# The commands the broker runs as the writer (writer.COMMANDS)
BROKERED = frozenset(writer.COMMANDS)
# The options of rk's command line that the broker's commands never have (writer.refused: rk post --premises runs as
# the monitor)
OFF = ("premises",)
REST = ("rest", "raw")  # gh.py's REST reads: each one a GET of its path
ARGV = {
    "run_read": gh.READ,
    "_run": gh.READ,
    "exec_read": gh.READ,
    "write": gh.WRITE,
}  # runs of an argv, and the login
STARTS = {*REST, "graphql", *ARGV}  # gh.py's functions that start gh
# gh.py's functions whose argv or path comes from their callers: their own calls of the others are no calls of their own
PLUMBING = {"_exec", "_run", "run_read", "exec_read", "rest", "raw", "graphql", "write"}
# gh.py's functions that start gh themselves (_exec), each with an argv of its own, under the login of the folder their
# caller gives them, and the kind of their calls: the probe's, and a login made from a token
OWN_EXEC = {"probe": "probe", "login_with_token": "login"}
# gh's commands that reach GitHub (`gh --version` does not)
GH_COMMANDS = {
    "api",
    "auth",
    "cache",
    "codespace",
    "gist",
    "issue",
    "label",
    "pr",
    "project",
    "release",
    "repo",
    "ruleset",
    "run",
    "search",
    "secret",
    "variable",
    "workflow",
}
VALUE_FLAGS = gh._API_VALUE_FLAGS | gh._WRITE_FLAGS  # gh api's flags that take a value
GRAPHQL = re.compile(r"\s*query\s*(\(\s*\$|\{)")  # the start of a GraphQL query
VALUE = re.compile(r"\{[^{}]*\}")  # a value in a call's path
GH_BINARY = re.compile(
    r"\bgh\b|\bgh_|\bbin"
)  # an expression that names gh's binary: gh's module, a "gh", a binary
FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)
LOGIN = {
    side: name for name, side in sr.GH_SIDES.items()
}  # the registry's GitHub login of each side
# One GitHub call in the code: its file and line, the function it is in ((file, name); name None at a module's level),
# its kind (read, write, probe, login: a login made from a token, own: a file's own gh, query: a GraphQL query's
# constant), the logins that make it (None: a file's own gh, whose folder it names itself) and the calls it may make.
Site = collections.namedtuple("Site", "rel line where kind sides calls")


def call_key(call: str) -> str:
    """A call as the scan and gh.CALLS compare it: a REST path without its query string, and `{}` for each value."""
    head, _, rest = call.partition(" ")
    if head.isupper():  # a REST method and its path
        rest = VALUE.sub("{}", rest.split("?", 1)[0])
    return f"{head} {rest}"


def argv_call(words: list) -> str | None:
    """The call an argv of gh makes: `METHOD path` for gh api (`graphql` for a GraphQL query), else `gh` and the
    command's two words; None when its path or its command is a value."""
    if words[:1] != ["api"]:
        return None if "{}" in words[:2] else "gh " + " ".join(words[:2])
    method, path, skip = "GET", "", False
    for i, w in enumerate(words[1:], 1):
        if skip:
            skip = False
        elif w in VALUE_FLAGS:
            skip = True
            if w in ("-X", "--method") and i + 1 < len(words):
                method = words[i + 1].upper()
        elif not w.startswith("-") and not path:
            path = w
    if path == "graphql":
        return "graphql"
    return None if not path or path.startswith("{}") else call_key(f"{method} {path}")


def own_nodes(scope):
    """The nodes of a function or a module, but those of the functions and classes in it."""
    todo = list(ast.iter_child_nodes(scope))
    while todo:
        n = todo.pop()
        yield n
        if not isinstance(n, (*FUNCTION, ast.Lambda, ast.ClassDef)):
            todo.extend(ast.iter_child_nodes(n))


def unpack(target, value, name: str) -> list:
    """What an assignment of VALUE to TARGET gives NAME: [the expression], [] when it gives NAME nothing."""
    if isinstance(target, ast.Name):
        return [value] if target.id == name else []
    if isinstance(target, (ast.Tuple, ast.List)):
        if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(
            target.elts
        ):
            return [
                v for t, e in zip(target.elts, value.elts) for v in unpack(t, e, name)
            ]
        if any(isinstance(t, ast.Name) and t.id == name for t in target.elts):
            return [value]  # unpacked from what cannot be told
    return []


def names_gh(expr) -> bool:
    """Whether the expression before a command's words is gh's binary: the word gh or gh-write, or an expression that
    names gh or a binary (not the atlassian CLI's)."""
    if isinstance(expr, ast.Constant):
        return expr.value in ("gh", "gh-write")
    text = ast.unparse(expr)
    return (
        not isinstance(expr, ast.Starred)
        and "atlassian" not in text
        and bool(GH_BINARY.search(text))
    )


def option(node) -> str | None:
    """The option of rk's command line an expression is, args.NAME (`cmd`: the command), else None."""
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    ):
        return node.attr
    return None


def told(v):
    """V when it is a string or a tuple of strings (a value the scan can use), else None."""
    if isinstance(v, str) or (
        isinstance(v, tuple) and all(isinstance(x, str) for x in v)
    ):
        return v
    return None


def parts(statement) -> tuple:
    """A statement's own expressions, and its lists of statements: a loop's, a with's, a try's (its handlers' too), a
    match's cases'."""
    heads, bodies, todo = [], [], [statement]
    while todo:
        for field, value in ast.iter_fields(todo.pop()):
            if field in ("body", "orelse", "finalbody"):
                bodies.append(value)
            elif field in ("handlers", "cases"):
                todo += value
            else:
                items = value if isinstance(value, list) else [value]
                heads += [v for v in items if isinstance(v, ast.AST)]
    return heads, bodies


class Code:
    """The package's Python files (the files tests/test_gh_split.py reads; `sources`, {file: text}, for a test's own),
    parsed, with each file's imports and functions, and the calls of each function. OWN_GH: the files that start gh
    themselves."""

    def __init__(self, sources: dict | None = None, own_gh=OWN_GH):
        if sources is None:
            sources = {
                str(p.relative_to(ROOT)): p.read_text(encoding="utf-8")
                for p in python_files()
            }
        self.own_gh, self.trees, self.parent, self.problems = own_gh, {}, {}, []
        self.bound = {}  # each function's variables (local)
        for rel, text in sources.items():
            self.trees[rel] = tree = ast.parse(text)
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    self.parent[child] = node
        self.imports = {rel: self._imports(rel, t) for rel, t in self.trees.items()}
        self.functions = {}  # (file, name): [each function of that name in the file]
        for rel, tree in self.trees.items():
            for node in ast.walk(tree):
                if isinstance(node, FUNCTION):
                    self.functions.setdefault((rel, node.name), []).append(node)
        self.callers = {}  # (file, name): [(file, each call of it)]
        for rel, tree in self.trees.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for target in self.targets(rel, node.func):
                        self.callers.setdefault(target, []).append((rel, node))
        self.queries = (
            {}
        )  # each GraphQL query's constant: {its value's node: (file, "MODULE.NAME")}
        for rel, tree in self.trees.items():
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and GRAPHQL.match(node.value)
                ):
                    up = self.parent.get(node)
                    if (
                        isinstance(up, ast.Assign)
                        and self.parent.get(up) is tree
                        and [type(t) for t in up.targets] == [ast.Name]
                    ):
                        self.queries[node] = (
                            rel,
                            f"{pathlib.PurePosixPath(rel).stem}.{up.targets[0].id}",
                        )
                    else:
                        self.problems.append(
                            f"{rel}:{node.lineno}: a GraphQL query that is no module's constant: name it"
                        )

    def _imports(self, rel: str, tree) -> dict:
        """{name: (file, None)} for each module of the package the file imports, {name: (file, NAME)} for each name it
        imports from one."""
        here, out = pathlib.PurePosixPath(rel).parent, {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = here if node.level else pathlib.PurePosixPath(".")
                for _ in range(max(node.level - 1, 0)):
                    base = base.parent
                if node.module:
                    base = base.joinpath(*node.module.split("."))
                for a in node.names:
                    if str(base / f"{a.name}.py") in self.trees:
                        out[a.asname or a.name] = (str(base / f"{a.name}.py"), None)
                    elif f"{base}.py" in self.trees:
                        out[a.asname or a.name] = (f"{base}.py", a.name)
                        if f"{base}.py" == GH_PY and a.name in STARTS:
                            self.problems.append(
                                f"{rel}:{node.lineno}: imports gh.{a.name}: call it as gh.{a.name}"
                            )
            elif isinstance(node, ast.Import):
                for a in node.names:
                    module = a.name.replace(".", "/") + ".py"
                    if a.asname and module in self.trees:
                        out[a.asname] = (module, None)
        return out

    def targets(self, rel: str, node) -> list:
        """The functions (file, name) an expression may be: a function of the file, one it imports, a module's, a
        method of `self`; a variable of a function it is in (local) is none."""
        if isinstance(node, ast.Name):
            module, name = self.imports[rel].get(node.id, (rel, node.id))
            if (module, name) in self.functions and not self.local(node):
                return [(module, name)]
            return []
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "self":
                return [(rel, node.attr)] if (rel, node.attr) in self.functions else []
            module = self.module_of(rel, node.value)
            if module and (module, node.attr) in self.functions:
                return [(module, node.attr)]
        return []

    def local(self, name) -> bool:
        """Whether a name is a variable of a function whose body it is in: the function's parameter, or a name it
        stores and does not declare global."""
        child = name
        while child in self.parent:
            fn = self.parent[child]
            body = getattr(fn, "body", None)
            if isinstance(fn, (*FUNCTION, ast.Lambda)) and any(
                child is b for b in (body if isinstance(body, list) else [body])
            ):
                if fn not in self.bound:
                    a, nodes = fn.args, list(own_nodes(fn))
                    given = (*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg)
                    self.bound[fn] = {x.arg for x in given if x} | {
                        n.id
                        for n in nodes
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                    }
                    self.bound[fn] -= {
                        g for n in nodes if isinstance(n, ast.Global) for g in n.names
                    }
                if name.id in self.bound[fn]:
                    return True
            child = fn
        return False

    def module_of(self, rel: str, node) -> str | None:
        """The file of the module an expression names (an import of the file), None when it names none."""
        if isinstance(node, ast.Name):
            module, name = self.imports[rel].get(node.id, (None, None))
            return module if name is None else None
        return None

    def enclosing(self, node) -> list:
        """The functions a node is in, the innermost first."""
        out = []
        while node in self.parent:
            node = self.parent[node]
            if isinstance(node, (*FUNCTION, ast.Lambda)):
                out.append(node)
        return out

    def owner(self, rel: str, node) -> tuple:
        """The function a node is in, (file, name): its outermost one, a method or a module's function; (file, None)
        at a module's level."""
        functions = [f for f in self.enclosing(node) if isinstance(f, FUNCTION)]
        return (rel, functions[-1].name if functions else None)

    def bindings(self, rel: str, name) -> list:
        """The expressions a name may hold, (file, node): in the innermost function that gives it any, its assignments
        and loops and, for a parameter, what each call of the function gives it (else its default); else the module's,
        or those of the module it is imported from."""
        for fn in self.enclosing(name):
            found = self._assigned(rel, fn, name.id)
            if not isinstance(fn, ast.Lambda) and name.id in [
                a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
            ]:
                found += self._arguments(rel, fn, name.id)
            if found:
                return found
        found = self._assigned(rel, self.trees[rel], name.id)
        module, imported = self.imports[rel].get(name.id, (None, None))
        if not found and imported:
            found = self._assigned(module, self.trees[module], imported)
        return found

    def _assigned(self, rel: str, scope, name: str) -> list:
        out = []
        for n in own_nodes(scope):
            if isinstance(n, ast.Assign):
                out += [v for t in n.targets for v in unpack(t, n.value, name)]
            elif isinstance(n, ast.AnnAssign) and n.value is not None:
                out += unpack(n.target, n.value, name)
            elif isinstance(n, (ast.For, ast.comprehension)) and isinstance(
                n.iter, (ast.Tuple, ast.List)
            ):
                out += [v for item in n.iter.elts for v in unpack(n.target, item, name)]
        return [(rel, v) for v in out]

    def _arguments(self, rel: str, fn, param: str) -> list:
        """What each call of the function FN gives its parameter PARAM, (file, node); its default when a call gives
        none, or when no call is found."""
        every = [a.arg for a in fn.args.posonlyargs + fn.args.args]
        default = dict(
            zip(every[len(every) - len(fn.args.defaults) :], fn.args.defaults)
        )
        default.update(
            (a.arg, d) for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults) if d
        )
        given = every[1:] if every[:1] in (["self"], ["cls"]) else every
        out = []
        for crel, call in self.callers.get((rel, fn.name), []):
            value = next((k.value for k in call.keywords if k.arg == param), None)
            if param in given and given.index(param) < len(call.args):
                value = call.args[given.index(param)]
            if value is not None and not isinstance(value, ast.Starred):
                out.append((crel, value))
            elif param in default:
                out.append((rel, default[param]))
        if not out and param in default:
            out.append((rel, default[param]))
        return out

    def texts(self, rel: str, node, depth: int = 0) -> list:
        """The texts an expression may have: a string's; an f-string's, `{}` for each value in it but a part of a
        path a name holds (a text with a / in it); `{}` for what cannot be told."""
        if depth > 12:
            return ["{}"]
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, ast.JoinedStr):
            out = [""]
            for v in node.values:
                if isinstance(v, ast.Constant):
                    parts = [str(v.value)]
                else:
                    parts = [
                        t for t in self.texts(rel, v.value, depth + 1) if "/" in t
                    ] or ["{}"]
                out = sorted({o + p for o in out for p in parts})
            return out
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self.texts(rel, node.left, depth + 1), self.texts(
                rel, node.right, depth + 1
            )
            return sorted({a + b for a in left for b in right})
        if isinstance(node, ast.Name):
            return sorted(
                {
                    t
                    for r, n in self.bindings(rel, node)
                    for t in self.texts(r, n, depth + 1)
                }
            ) or ["{}"]
        return ["{}"]

    def argvs(self, rel: str, node, depth: int = 0) -> list:
        """The argvs of gh an expression may hold: a list's or a tuple's words, up to a starred one; a name's values;
        [] when they cannot be told."""
        if depth > 12:
            return []
        if isinstance(node, (ast.List, ast.Tuple)):
            out = [[]]
            for e in node.elts:
                if isinstance(e, ast.Starred):
                    break
                out = [w + [t] for w in out for t in self.texts(rel, e)]
            return out
        if isinstance(node, ast.Name):
            return [
                a
                for r, n in self.bindings(rel, node)
                for a in self.argvs(r, n, depth + 1)
            ]
        return []

    def documents(self, rel: str, node, depth: int = 0) -> list:
        """The GraphQL queries an expression may be, as MODULE.NAME: a query's constant, a module's, one formatted
        with % or changed by a call (re.sub, str.replace)."""
        if depth > 12:
            return []
        if node in self.queries:
            return [self.queries[node][1]]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            return self.documents(rel, node.left, depth + 1)
        if isinstance(node, ast.Call):
            parts = [
                *node.args,
                *([node.func.value] if isinstance(node.func, ast.Attribute) else []),
            ]
            return sorted({d for p in parts for d in self.documents(rel, p, depth + 1)})
        if isinstance(node, ast.Attribute) and self.module_of(rel, node.value):
            name = f"{pathlib.PurePosixPath(self.module_of(rel, node.value)).stem}.{node.attr}"
            return [name] if any(name == q for _r, q in self.queries.values()) else []
        if isinstance(node, ast.Name):
            return sorted(
                {
                    d
                    for r, n in self.bindings(rel, node)
                    for d in self.documents(r, n, depth + 1)
                }
            )
        return []

    def argv_calls(self, rel: str, node) -> list:
        """The calls an argv expression of gh may make; a GraphQL query's: the query its `query=` field names."""
        out = set()
        for words in self.argvs(rel, node):
            call = argv_call(words)
            if call == "graphql":
                out.update(f"graphql {d}" for d in self.queries_in(rel, node))
            elif call:
                out.add(call)
        return sorted(out)

    def queries_in(self, rel: str, node, depth: int = 0) -> list:
        """The GraphQL queries an argv names in its `query=` field (a name's values followed)."""
        if isinstance(node, ast.Name):
            return [
                d
                for r, n in self.bindings(rel, node)
                if depth < 12
                for d in self.queries_in(r, n, depth + 1)
            ]
        out = []
        for n in ast.walk(node):
            if (
                isinstance(n, ast.JoinedStr)
                and n.values
                and isinstance(n.values[0], ast.Constant)
            ):
                if str(n.values[0].value).startswith("query="):
                    values = [
                        v.value
                        for v in n.values[1:]
                        if isinstance(v, ast.FormattedValue)
                    ]
                    out += [d for v in values for d in self.documents(rel, v)]
        return out

    def sites(self) -> list:
        """Each GitHub call in the code (Site); a read the writer runs (writer) is made by both logins."""
        writer, handled = self.writer()
        out = []
        for rel, tree in self.trees.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    out += [
                        (
                            s._replace(sides=(gh.READ, gh.WRITE))
                            if s.kind == "read"
                            and (s.where in writer or node in handled)
                            else s
                        )
                        for s in self._call_sites(rel, node)
                    ]
                elif (
                    isinstance(node, ast.Attribute)
                    and node.attr in STARTS
                    and self.module_of(rel, node.value) == GH_PY
                    and not (
                        isinstance(self.parent.get(node), ast.Call)
                        and self.parent[node].func is node
                    )
                ):
                    self.problems.append(
                        f"{rel}:{node.lineno}: gh.{node.attr} used as a value: call it"
                    )
            if rel in self.own_gh:
                out += self._own_gh(rel, tree)
        for node, (rel, name) in self.queries.items():
            out.append(
                Site(
                    rel,
                    node.lineno,
                    (rel, None),
                    "query",
                    (gh.READ,),
                    [f"graphql {name}"],
                )
            )
        return out

    def _call_sites(self, rel: str, call) -> list:
        f, where = call.func, self.owner(rel, call)
        if isinstance(f, ast.Attribute) and self.module_of(rel, f.value) == GH_PY:
            name = f.attr
        elif rel == GH_PY and isinstance(f, ast.Name) and where[1] not in PLUMBING:
            name = f.id
        else:
            return []
        if name == "_exec":
            if (
                where[1] not in OWN_EXEC
                or not call.args
                or not isinstance(call.args[0], ast.List)
            ):
                self.problems.append(
                    f"{rel}:{call.lineno}: gh starts outside run_read, write, the probe and login_with_token: list "
                    "its calls"
                )
                return []
            # their own calls, under the login of the folder they are given: the one probed, or the one logged in to
            return [
                Site(
                    rel,
                    call.lineno,
                    where,
                    OWN_EXEC[where[1]],
                    (gh.READ, gh.WRITE),
                    self.argv_calls(rel, call.args[0]),
                )
            ]
        if name not in STARTS:
            return []
        first = (
            call.args[0]
            if call.args
            else next(
                (k.value for k in call.keywords if k.arg in ("path", "args")), None
            )
        )
        if first is None:
            calls = []
        elif name in REST:
            calls = sorted(
                {
                    call_key(f"GET {t}")
                    for t in self.texts(rel, first)
                    if not t.startswith("{}")
                }
            )
        elif name == "graphql":
            calls = [f"graphql {d}" for d in self.documents(rel, first)]
        else:
            calls = self.argv_calls(rel, first)
            if not calls and name == "exec_read":
                calls = [gh.ANY_READ]  # tools/ghr.py: the reads its command line gives
        kind = "write" if ARGV.get(name) == gh.WRITE else "read"
        return [
            Site(
                rel,
                call.lineno,
                where,
                kind,
                (ARGV.get(name, gh.READ),),
                sorted(set(calls)),
            )
        ]

    def _own_gh(self, rel: str, tree) -> list:
        """The argvs a file that starts gh itself gives it: after gh's binary (names_gh) in a list, a tuple or a call's
        arguments; a call of a function named gh."""
        out = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "gh"
            ):
                argv = (
                    node.args[0].value
                    if node.args and isinstance(node.args[0], ast.Starred)
                    else None
                )
                argv = argv or ast.List(elts=node.args, ctx=ast.Load())
                out.append(
                    Site(
                        rel,
                        node.lineno,
                        self.owner(rel, node),
                        "own",
                        None,
                        self.argv_calls(rel, argv),
                    )
                )
                continue
            seqs = [node.elts] if isinstance(node, (ast.List, ast.Tuple)) else []
            seqs += [node.args] if isinstance(node, ast.Call) else []
            for elts in seqs:
                for i in range(1, len(elts)):
                    if not (
                        isinstance(elts[i], ast.Constant)
                        and elts[i].value in GH_COMMANDS
                    ):
                        continue
                    if names_gh(elts[i - 1]):
                        argv = ast.List(elts=elts[i:], ctx=ast.Load())
                        site = Site(
                            rel,
                            node.lineno,
                            self.owner(rel, node),
                            "own",
                            None,
                            self.argv_calls(rel, argv),
                        )
                        out.append(site)
                    break
        return out

    def value(self, rel: str, node, depth: int = 0):
        """The value an expression has when the code tells it: a string, or a tuple of strings (a tuple's, a list's or
        a set's, a starred one's spread in it); a name or a module's constant is followed to its one assignment, but a
        name of TABLES' modules, and a call of the table's `names`, are read from the module. None when it cannot be
        told."""
        if depth > 12:
            return None
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, str) else None
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            out = []
            for e in node.elts:
                spread = isinstance(e, ast.Starred)
                v = self.value(rel, e.value if spread else e, depth + 1)
                if not isinstance(v, tuple if spread else str):
                    return None
                out += v if spread else [v]
            return tuple(out)
        if isinstance(node, ast.Call):
            return self.table_names(rel, node, depth)
        found = []
        if isinstance(node, ast.Name):
            found = self.bindings(rel, node)
        elif isinstance(node, ast.Attribute) and self.module_of(rel, node.value):
            module = self.module_of(rel, node.value)
            if module in TABLES:
                return told(getattr(TABLES[module], node.attr, None))
            found = self._assigned(module, self.trees[module], node.attr)
        return self.value(*found[0], depth + 1) if len(found) == 1 else None

    def table_names(self, rel: str, call, depth: int):
        """The names a call of the table's `names` gives (outward.names), when the code tells each column's value; None
        for any other call."""
        func = call.func
        if (
            not isinstance(func, ast.Attribute)
            or func.attr != "names"
            or self.module_of(rel, func.value) != OUTWARD_PY
            or call.args
        ):
            return None
        columns = {}
        for kw in call.keywords:
            v = kw.value
            v = (
                v.value
                if isinstance(v, ast.Constant)
                else self.value(rel, v, depth + 1)
            )
            if kw.arg is None or v is None:
                return None
            columns[kw.arg] = v
        return outward.names(**columns)

    def split(self, rel: str, test, cmds: frozenset) -> tuple:
        """(the commands of CMDS that may pass an if's TEST, those that may fail it): a test of args.cmd (==, !=, in,
        not in, startswith) tells them apart when the code tells what it compares with, and an option they never have
        (OFF) passes none; and, or and not join tests; any other test may go either way.
        """
        if option(test) in OFF:
            return frozenset(), cmds
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            yes, no = self.split(rel, test.operand, cmds)
            return no, yes
        if isinstance(test, ast.BoolOp):  # and: every test passes; or: one does
            every, left, out = isinstance(test.op, ast.And), cmds, frozenset()
            for v in test.values:
                yes, no = self.split(rel, v, left)
                out, left = (out | no, yes) if every else (out | yes, no)
            return (left, out) if every else (out, left)
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and option(test.left) == "cmd"
        ):
            op, v = test.ops[0], self.value(rel, test.comparators[0])
            if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(v, str):
                yes = cmds & {v}
            elif isinstance(op, (ast.In, ast.NotIn)) and isinstance(v, tuple):
                yes = cmds & set(v)
            else:
                return cmds, cmds
            if isinstance(op, (ast.Eq, ast.In)):
                return yes, cmds - yes
            return cmds - yes, yes
        if (
            isinstance(test, ast.Call)
            and isinstance(test.func, ast.Attribute)
            and test.func.attr == "startswith"
            and option(test.func.value) == "cmd"
            and len(test.args) == 1
            and isinstance(self.value(rel, test.args[0]), (str, tuple))
        ):
            start = self.value(rel, test.args[0])
            yes = frozenset(c for c in cmds if c.startswith(start))
            return yes, cmds - yes
        return cmds, cmds

    def _flow(self, body: list, cmds: frozenset, ran: list) -> frozenset:
        """Walk rk's statements BODY, which the commands CMDS may run, adding to RAN each part they run, (node, those
        commands): an if's branches run for the commands its test lets in (split); a return, a raise, a break or a
        continue ends BODY. Returns the commands that may run past BODY's end; a loop, a with, a try or a match
        narrows none."""
        for st in body:
            if not cmds:
                break
            if isinstance(st, ast.If):
                ran.append((st.test, cmds))
                yes, no = self.split(RK_PY, st.test, cmds)
                cmds = self._flow(st.body, yes, ran) | self._flow(st.orelse, no, ran)
                continue
            definition = isinstance(st, (*FUNCTION, ast.ClassDef))
            heads, bodies = ([st], []) if definition else parts(st)
            ran += [(h, cmds) for h in heads]
            for b in bodies:
                self._flow(b, cmds, ran)
            if isinstance(st, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                return frozenset()
        return cmds

    def handlers(self) -> tuple:
        """rk's handlers of the commands the broker runs (BROKERED): its main, and each function of RK_PY the code that
        runs calls or names, each walked (_flow) for the commands that reach it. Returns (the nodes of RK_PY that run,
        the functions (file, name) of the other files that they call or name)."""
        given, todo = {}, [((RK_PY, "main"), BROKERED)]
        nodes, called = set(), set()
        while todo:
            key, cmds = todo.pop()
            if cmds <= given.get(key, frozenset()):
                continue
            given[key] = cmds = cmds | given.get(key, frozenset())
            ran = []
            for fn in self.functions.get(key, []):
                self._flow(fn.body, cmds, ran)
            for part, those in ran:
                for n in ast.walk(part):
                    nodes.add(n)
                    if isinstance(n, (ast.Name, ast.Attribute)) and isinstance(
                        n.ctx, ast.Load
                    ):
                        for t in self.targets(RK_PY, n):
                            if t[0] == RK_PY:
                                todo.append((t, those))
                            else:
                                called.add(t)
        return nodes, called

    def writer(self) -> tuple:
        """What the writer runs: rk's handlers of the commands the broker runs (handlers), each function of post.py and
        gpu.py that writes (it calls post._gh_write, or another such function), and every function these call or name,
        all the way down. Returns (those functions (file, name), the nodes of RK_PY that run).
        """
        writes, todo = set(), [(POST_PY, "_gh_write")]
        while todo:
            for rel, call in self.callers.get(todo.pop(), []):
                where = self.owner(rel, call)
                if rel in (POST_PY, GPU_PY) and where[1] and where not in writes:
                    writes.add(where)
                    todo.append(where)
        handled, called = self.handlers()
        seen = writes | called
        todo = list(seen)
        while todo:
            rel, name = todo.pop()
            for fn in self.functions.get((rel, name), []):
                for n in ast.walk(fn):
                    if isinstance(n, (ast.Name, ast.Attribute)) and isinstance(
                        n.ctx, ast.Load
                    ):
                        for t in self.targets(rel, n):
                            if t not in seen:
                                seen.add(t)
                                todo.append(t)
        return seen, handled


def unlisted(sites: list, calls) -> list:
    """What the scan finds that CALLS (gh.CALLS) does not say: a call that is not listed, one whose path or command
    cannot be told, one a login makes that its entry does not give."""
    listed = {call_key(c): sides for sides, c, _what, _perms in calls}
    bad = []
    for s in sites:
        if not s.calls:
            bad.append(
                f"{s.rel}:{s.line}: a GitHub call whose path or command cannot be told: name it"
            )
        for c in s.calls:
            if c not in listed:
                bad.append(
                    f"{s.rel}:{s.line}: {c}: not in gh.CALLS: list it with the permissions it needs"
                )
            elif s.sides and not set(s.sides) <= set(listed[c]):
                bad.append(
                    f"{s.rel}:{s.line}: {c}: made by the {'/'.join(s.sides)} login: list it for that login"
                )
    return bad


class GitHubCallsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code = Code()
        cls.sites = cls.code.sites()

    def test_each_call_is_listed_once_with_its_logins_and_permissions(self):
        keys = [call_key(c) for _sides, c, _what, _perms in gh.CALLS]
        self.assertEqual(
            sorted({k for k in keys if keys.count(k) > 1}), [], "a call listed twice"
        )
        for sides, c, what, perms in gh.CALLS:
            with self.subTest(call=c):
                self.assertTrue(sides and set(sides) <= {gh.READ, gh.WRITE}, sides)
                self.assertTrue(what.strip())
                self.assertTrue(
                    all(len(p) == 2 and p[1] in sr.ACCESS for p in perms), perms
                )

    def test_every_github_call_is_listed(self):
        self.assertEqual(self.code.problems + unlisted(self.sites, gh.CALLS), [])

    def test_every_listed_call_is_made_by_its_logins(self):
        made = (
            {}
        )  # call: the logins its sites make it with (None: a file's own gh, which names its folder itself)
        for s in self.sites:
            for c in s.calls:
                made.setdefault(c, set()).update(s.sides or (None,))
        bad = []
        for sides, c, _what, _perms in gh.CALLS:
            have = made.get(call_key(c), set())
            if not have:
                bad.append(f"{c}: no code makes it: remove it from gh.CALLS")
            elif None not in have and not set(sides) <= have:
                bad.append(
                    f"{c}: only the {'/'.join(sorted(have))} login makes it, not the {'/'.join(sides)} one"
                )
        self.assertEqual(bad, [])

    def test_each_login_has_in_the_registry_the_permissions_its_calls_need(self):
        e = {x.name: x for x in sr.entries({})}
        self.assertEqual(e[LOGIN[gh.READ]].permissions, gh.permissions(gh.READ))
        self.assertEqual(e[LOGIN[gh.WRITE]].permissions, gh.permissions(gh.WRITE))

    def test_the_organization_s_permissions_are_needed_and_are_all_its_calls_need(self):
        """gh.ORGANIZATION_PERMISSIONS, the permissions a token gives on the organization: each is one a call needs,
        and a call on the organization (`GET orgs/...`) needs nothing but them."""
        needed = {p for _sides, _c, _what, perms in gh.CALLS for p, _access in perms}
        self.assertTrue(gh.ORGANIZATION_PERMISSIONS)
        self.assertLessEqual(set(gh.ORGANIZATION_PERMISSIONS), needed)
        on_org = [
            (c, perms)
            for _sides, c, _what, perms in gh.CALLS
            if c.startswith("GET orgs/")
        ]
        self.assertTrue(on_org)
        for c, perms in on_org:
            with self.subTest(call=c):
                self.assertTrue(perms, c)
                self.assertLessEqual(
                    {p for p, _access in perms}, set(gh.ORGANIZATION_PERMISSIONS)
                )

    def test_the_deploy_guide_s_step_5_names_each_permission_of_each_token(self):
        """Step 5 of the deploy guide (its generated logins section) has a row for each GitHub login,
        whose last two cells give each permission its calls need with its access, `NAME, NAME: ACCESS; NAME:
        ACCESS` or `none`, each in its own column: the organization's (gh.ORGANIZATION_PERMISSIONS) in the last, every
        other, the repository's, in the one before it.
        """

        def given(cell) -> dict:
            out = {}
            for group in [] if cell == "none" else cell.split("; "):
                names, access = group.rsplit(": ", 1)
                out.update((name, access) for name in names.split(", "))
            return out

        text = (ROOT / "QUICK_DEPLOY_GUIDE.md").read_text(encoding="utf-8")
        step = text.split("\n## Step 5: login\n", 1)[1].split("\n## ", 1)[0]
        for side in (gh.READ, gh.WRITE):
            row = next(
                ln for ln in step.splitlines() if ln.startswith(f"| `{LOGIN[side]}` | ")
            )
            repository, organization = (
                c.strip() for c in row.rstrip().rstrip("|").rsplit(" | ", 2)[1:]
            )
            needs = gh.permissions(side)
            for column, cell, wanted in (
                (
                    "repository",
                    repository,
                    [p for p in needs if p[0] not in gh.ORGANIZATION_PERMISSIONS],
                ),
                (
                    "organization",
                    organization,
                    [p for p in needs if p[0] in gh.ORGANIZATION_PERMISSIONS],
                ),
            ):
                with self.subTest(login=LOGIN[side], column=column):
                    self.assertEqual(given(cell), dict(wanted))

    def test_a_login_needs_metadata_and_what_its_calls_need_write_over_read(self):
        calls = (
            ((gh.READ,), "GET a", "", (("Pull requests", "read"),)),
            (
                (gh.WRITE,),
                "gh b c",
                "",
                (("Issues", "write"), ("Pull requests", "write")),
            ),
            ((gh.READ, gh.WRITE), "GET d", "", (("Issues", "read"),)),
            ((gh.READ,), "GET e", "", ()),
        )
        with mock.patch.object(gh, "CALLS", calls):
            self.assertEqual(
                gh.permissions(gh.READ),
                (("Issues", "read"), ("Metadata", "read"), ("Pull requests", "read")),
            )
            self.assertEqual(
                gh.permissions(gh.WRITE),
                (("Issues", "write"), ("Metadata", "read"), ("Pull requests", "write")),
            )


# Invented code: each kind of call the scan finds.
INVENTED = '''
from rkmon_lib import gh

_LIST = """
query($n: Int!) { repository(owner: "o", name: "r") { pullRequest(number: $n) { title } } }
"""


def files(repo, n):
    return gh.rest(f"repos/{repo}/pulls/{n}/files?per_page=100", paginate=True)


def checks(n, flag="--watch"):
    argv = ["pr", "checks", str(n), flag]
    return gh.run_read(argv)


def page(n, query=_LIST):
    return gh.graphql(query, n=n)


def guess(path):
    return gh.rest(path)


def inline(n):
    return gh.graphql("""query { viewer { login } }""")


def later():
    return [gh.raw]
'''
INVENTED_POST = """
from . import gh, helper


def _gh_write(args):
    return gh.write(args)


def label(repo, n):
    helper.check(repo, n)
    return _gh_write(["issue", "edit", str(n), "--add-label", "x"])


def events(repo, n):
    return gh.rest(f"repos/{repo}/issues/{n}/events")
"""
INVENTED_HELPER = """
from . import gh


def check(repo, n):
    return gh.rest(f"repos/{repo}/milestones/{n}")
"""
# Invented rk: the code the broker's commands run.
INVENTED_RK = """
from rkmon_lib import gh, helper, writer

LOOKS = ("status", "gate")


def main(args):
    if args.cmd in LOOKS:  # commands the broker does not run
        return look(args)
    if args.cmd == "post" and args.premises:  # rk post --premises runs as the session user
        return gh.rest(f"repos/{args.repo}/commits/{args.head}")
    if args.cmd in ("notice", "queue-advance"):
        if args.cmd != "notice":
            return gh.rest(f"repos/{args.repo}/pulls/{args.n}/commits")
        helper.check(args.repo, args.n)
    if args.cmd == "label-create" or args.cmd.startswith(("issue-", "gh-")):
        return create(args)
    if args.cmd in writer.COMMANDS:
        return 0
    return gh.rest(f"repos/{args.repo}/issues/{args.n}/comments")  # every command the broker runs has returned


def look(args):
    return gh.rest(f"repos/{args.repo}/pulls/{args.n}/reviews")


def create(args):
    look = gh.rest(f"repos/{args.repo}/labels")  # a variable, not the function look
    return look
"""
INVENTED_TOOL = """
import os
import subprocess

subprocess.run([os.environ.get("RKMON_GH") or "gh", "release", "list"])
subprocess.run(["docker", "run", "--rm", "x"])
"""


class ScanTest(unittest.TestCase):
    """The scan, on invented code."""

    def scan(self, **files) -> tuple:
        sources = {
            rel: (ROOT / rel).read_text(encoding="utf-8") for rel in (GH_PY, WRITER_PY)
        }
        sources.update(files)
        code = Code(sources, own_gh={f for f in files if f.startswith("tools/")})
        return code, [s for s in code.sites() if s.rel != GH_PY]

    def test_each_kind_of_call_is_found_and_an_unlisted_one_fails(self):
        code, sites = self.scan(
            **{
                "rkmon_lib/invented.py": INVENTED,
                "tools/invented_tool.py": INVENTED_TOOL,
            }
        )
        self.assertEqual(
            sorted(
                (s.where[1] or "", s.kind, s.sides or (), tuple(s.calls)) for s in sites
            ),
            [
                ("", "own", (), ("gh release list",)),
                ("", "query", (gh.READ,), ("graphql invented._LIST",)),
                ("checks", "read", (gh.READ,), ("gh pr checks",)),
                ("files", "read", (gh.READ,), ("GET repos/{}/pulls/{}/files",)),
                ("guess", "read", (gh.READ,), ()),
                ("inline", "read", (gh.READ,), ()),
                ("page", "read", (gh.READ,), ("graphql invented._LIST",)),
            ],
        )
        self.assertEqual(
            sorted(p.split(": ", 1)[1] for p in code.problems),
            [
                "a GraphQL query that is no module's constant: name it",
                "gh.raw used as a value: call it",
            ],
        )
        bad = "\n".join(unlisted(sites, ()))
        self.assertIn(
            "rkmon_lib/invented.py:10: GET repos/{}/pulls/{}/files: not in gh.CALLS",
            bad,
        )
        self.assertIn(
            "rkmon_lib/invented.py:23: a GitHub call whose path or command cannot be told",
            bad,
        )
        self.assertIn("tools/invented_tool.py:5: gh release list: not in gh.CALLS", bad)

    def test_a_read_the_writer_makes_is_made_by_both_logins(self):
        _code, sites = self.scan(
            **{POST_PY: INVENTED_POST, "rkmon_lib/helper.py": INVENTED_HELPER}
        )
        self.assertEqual(
            sorted((s.where[1], s.kind, s.sides, tuple(s.calls)) for s in sites),
            [
                ("_gh_write", "write", (gh.WRITE,), ("gh issue edit",)),
                ("check", "read", (gh.READ, gh.WRITE), ("GET repos/{}/milestones/{}",)),
                ("events", "read", (gh.READ,), ("GET repos/{}/issues/{}/events",)),
            ],
        )
        listed = (((gh.READ,), "GET repos/{repo}/milestones/{n}", "", ()),)
        self.assertIn(
            "GET repos/{}/milestones/{}: made by the read/write login",
            "\n".join(unlisted(sites, listed)),
        )

    def test_a_read_a_command_the_broker_runs_makes_is_made_by_both_logins(self):
        _code, sites = self.scan(
            **{RK_PY: INVENTED_RK, "rkmon_lib/helper.py": INVENTED_HELPER}
        )
        self.assertEqual(
            sorted((s.where[1], s.kind, s.sides, tuple(s.calls)) for s in sites),
            [
                ("check", "read", (gh.READ, gh.WRITE), ("GET repos/{}/milestones/{}",)),
                ("create", "read", (gh.READ, gh.WRITE), ("GET repos/{}/labels",)),
                ("look", "read", (gh.READ,), ("GET repos/{}/pulls/{}/reviews",)),
                ("main", "read", (gh.READ,), ("GET repos/{}/commits/{}",)),
                ("main", "read", (gh.READ,), ("GET repos/{}/issues/{}/comments",)),
                ("main", "read", (gh.READ,), ("GET repos/{}/pulls/{}/commits",)),
            ],
        )
        listed = (((gh.READ,), "GET repos/{repo}/labels", "", ()),)
        self.assertIn(
            "GET repos/{}/labels: made by the read/write login",
            "\n".join(unlisted(sites, listed)),
        )


if __name__ == "__main__":
    unittest.main()

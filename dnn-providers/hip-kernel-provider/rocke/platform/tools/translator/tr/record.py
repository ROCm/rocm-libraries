# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""S2 Record: run every parity config in-process and record, per function, the
types its arguments and return value took (sys.setprofile at function
boundaries; no source rewriting).

usage: python -m tr.record <emit_module> <module> [<module> ...] [--rocke DIR] [--out FILE]

Writes data/facts/<first module>.json:
  funcs: "<qualname>" -> {params: {name: [tags]}, return: [tags], calls: n, src: hash}
         (a closure also has cells: {free variable: [tags]}, the values it captured;
         keys and the src stamp are described in tr/factkeys.py; functions of
         the other modules are keyed "<module>:<qualname>")
  attrs: "<class>" -> {attr: [tags]} (instance attributes when __init__ returns)
  configs / built / rejected
A tag is a type name; containers carry their element tags: "list[Value]",
"tuple[int|str,...]", "dict[str:Value]"; a tuple of at most 4 items is a
record, one tag per position: "tuple[Value,Value]". Complete sets, never
samples. "driver" holds methods of the emit module's own objects that the
module calls back into (duck-typed parameters).
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import importlib
import json
import ast
import os
import sys
import time

from .env import ROCKE_HELP, data_dir, library_dir, rocke_root
from .factkeys import code_keys, first_line, src_hash


def tag(v, depth=0) -> str:
    t = type(v)
    if v is None:
        return "None"
    if t in (bool, int, float, str):
        return t.__name__
    if t is tuple and 0 < len(v) <= 4 and depth <= 2:
        # short tuples are records: one tag per position
        return f"tuple[{','.join(tag(x, depth + 1) for x in v)}]"
    if t in (tuple, list, set, frozenset):
        if depth > 2:
            return t.__name__
        if t is tuple:
            inner = sorted({tag(x, depth + 1) for x in list(v)[:32]})
            return f"tuple[{'|'.join(inner)},...]"
        inner = sorted({tag(x, depth + 1) for x in list(v)[:32]})
        return f"{t.__name__}[{'|'.join(inner)}]"
    if t is dict:
        if depth > 2:
            return "dict"
        ks = sorted({tag(k, depth + 1) for k in list(v)[:64]})
        vs = sorted({tag(x, depth + 1) for x in list(v.values())[:64]})
        return f"dict[{'|'.join(ks)}:{'|'.join(vs)}]"
    if (
        callable(v)
        and not isinstance(v, type)
        and t.__name__ in ("function", "method", "builtin_function_or_method")
    ):
        return "callable"
    return t.__name__


def merge_facts(old: dict, new: dict) -> dict:
    """Union of two recordings of the same module from different drivers."""

    def tags(a, b):
        return sorted(set(a) | set(b))

    out = dict(new)
    out["emit"] = ",".join(
        sorted((set(str(old.get("emit", "")).split(",")) | {new["emit"]}) - {""})
    )
    for k in ("configs", "built", "rejected", "seconds"):
        out[k] = (old.get(k) or 0) + (new.get(k) or 0)
    for sec in ("funcs", "driver"):
        merged = {k: dict(v) for k, v in old.get(sec, {}).items()}
        for k, v in new.get(sec, {}).items():
            if k not in merged:
                merged[k] = v
                continue
            m = merged[k]
            m["calls"] = m.get("calls", 0) + v.get("calls", 0)
            m["return"] = tags(m.get("return", []), v.get("return", []))
            m["args"] = m.get("args") or v.get("args")
            ps = dict(m.get("params", {}))
            for n, t in v.get("params", {}).items():
                ps[n] = tags(ps.get(n, []), t)
            m["params"] = ps
            cs = dict(m.get("cells", {}))
            for n, t in v.get("cells", {}).items():
                cs[n] = tags(cs.get(n, []), t)
            if cs:
                m["cells"] = cs
            if v.get("src"):
                m["src"] = v["src"]  # the newer recording's source
        out[sec] = merged
    attrs = {c: dict(d) for c, d in old.get("attrs", {}).items()}
    for c, d in new.get("attrs", {}).items():
        for k, t in d.items():
            attrs.setdefault(c, {})[k] = tags(attrs.get(c, {}).get(k, []), t)
    out["attrs"] = attrs
    return out


class Keys:
    """Facts key and source stamp of each code object the profiler sees.
    Functions of a module other than the first carry "<module>:" in front,
    so equal qualnames in two recorded modules stay apart."""

    def __init__(self, prefix: dict[str, str]):
        self.prefix = prefix  # source file -> key prefix
        self.by_code: dict = {}
        self.files: dict[str, dict] = {}
        self.src: dict[str, str] = {}

    def __call__(self, co) -> str:
        key = self.by_code.get(co)
        if key is None:
            index = self.files.get(co.co_filename)
            if index is None:
                index = self.files[co.co_filename] = self._index(co.co_filename)
            key, src = index.get(
                (co.co_qualname, co.co_firstlineno),
                (f"{co.co_qualname}@{co.co_firstlineno}", None),
            )
            key = self.prefix.get(co.co_filename, "") + key
            if src is not None:
                self.src[key] = src
            self.by_code[co] = key
        return key

    @staticmethod
    def _index(path: str) -> dict:
        """(co_qualname, co_firstlineno) -> (key, src) over one source file."""
        try:
            tree = ast.parse(open(path).read())
        except (OSError, SyntaxError, ValueError):
            return {}
        out = {}
        for key, node in code_keys(tree):
            q = key.split("#")[0]
            # two code objects on one line with one qualname: the first one wins
            out.setdefault((q, first_line(node)), (key, src_hash(node)))
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("emit")
    ap.add_argument("modules", nargs="+")
    ap.add_argument("--rocke", help=ROCKE_HELP)
    ap.add_argument("--out")
    ap.add_argument(
        "--merge",
        action="store_true",
        help="union with the facts already in the output file (several drivers)",
    )
    a = ap.parse_args(argv)
    a.rocke = rocke_root(a.rocke)
    for p in ("platform/python", library_dir(a.rocke), "library/tests/parity"):
        full = os.path.join(a.rocke, p)
        if full not in sys.path:
            sys.path.insert(0, full)
    targets = {}
    for m in a.modules:
        mod = importlib.import_module(m if "." in m else f"kernels.common.{m}")
        targets[mod.__file__] = m

    params = collections.defaultdict(lambda: collections.defaultdict(set))
    rets = collections.defaultdict(set)
    cells = collections.defaultdict(lambda: collections.defaultdict(set))
    calls = collections.Counter()
    attrs = collections.defaultdict(lambda: collections.defaultdict(set))
    names = {}
    keyof = Keys({f: f"{m}:" for f, m in targets.items() if m != a.modules[0]})

    driver_file = []  # the emit module's file, once imported
    driver_keys = set()

    def prof(frame, event, arg):
        co = frame.f_code
        if co.co_filename not in targets:
            # a driver object the module calls back into (a duck-typed view):
            # its signature is the protocol the C ABI needs
            back = frame.f_back
            if not (
                driver_file
                and co.co_filename == driver_file[0]
                and back is not None
                and back.f_code.co_filename in targets
            ):
                if not (
                    event == "return"
                    and driver_file
                    and co.co_filename == driver_file[0]
                ):
                    return
            key = keyof(co)
            if event == "call":
                driver_keys.add(key)
            elif key not in driver_keys:
                return
        key = keyof(co)
        if event == "call":
            names[key] = co.co_varnames[: co.co_argcount + co.co_kwonlyargcount]
            calls[key] += 1
            loc = frame.f_locals
            for n in names[key]:
                if n in loc:
                    params[key][n].add(tag(loc[n]))
            for n in co.co_freevars:
                if n in loc:
                    cells[key][n].add(
                        tag(loc[n])
                    )  # a closure's captured values (host clones)
        elif event == "return":
            rets[key].add(tag(arg))
            if (
                co.co_argcount
                and co.co_varnames[0] == "self"
                and "self" in frame.f_locals
            ):
                # instance state after any method (fields set outside __init__ too)
                obj = frame.f_locals["self"]
                cls = type(obj).__name__
                for k, v in getattr(obj, "__dict__", {}).items():
                    attrs[cls][k].add(tag(v))

    import _emit_common  # noqa: F401

    grabbed = {}

    def fake_run_emit(spec_fn, build_fn, **kw):
        grabbed.update(
            spec_fn=spec_fn, build_fn=build_fn, arch=kw.get("arch", "gfx950")
        )
        return 0

    emit = importlib.import_module(a.emit)
    driver_file.append(emit.__file__)
    emit.run_emit = fake_run_emit
    saved = sys.argv
    sys.argv = sys.argv[:1]
    try:
        with open(os.devnull, "w") as null, contextlib.redirect_stderr(null):
            emit.main()  # run_emit drivers hand over their spec / build functions
    except (SystemExit, Exception):
        pass  # a driver with its own main: usage error without an index
    sys.argv = saved
    ok = rej = 0
    t0 = time.time()
    n_configs = len(emit._CONFIGS) if hasattr(emit, "_CONFIGS") else None
    rejects = (
        ValueError,
        TypeError,
        NotImplementedError,
        KeyError,
        AssertionError,
        RuntimeError,
        IndexError,
    )
    sys.setprofile(prof)
    try:
        idx = 0
        while n_configs is None or idx < n_configs:
            if idx > 100000:
                break
            try:
                if "spec_fn" in grabbed:
                    # run_emit drivers: the spec and build functions, no lowering
                    sel = grabbed["spec_fn"](idx)
                    sel, arch = (
                        sel if isinstance(sel, tuple) else (sel, grabbed["arch"])
                    )
                    grabbed["build_fn"](sel, arch=arch)
                else:
                    # a driver with its own main: run it on this index, output dropped
                    sys.argv = [a.emit, str(idx)]
                    with open(os.devnull, "w") as null, contextlib.redirect_stdout(
                        null
                    ), contextlib.redirect_stderr(null):
                        rc = emit.main()
                    if rc == 2:
                        break  # unknown config index
                ok += 1
            except SystemExit:
                if n_configs is None:
                    break  # the driver's "unknown config index"
                rej += 1
            except rejects:
                rej += 1
            idx += 1
        n_configs = idx if n_configs is None else n_configs
    finally:
        sys.setprofile(None)
        sys.argv = saved
    out = {
        "emit": a.emit,
        "configs": n_configs,
        "built": ok,
        "rejected": rej,
        "seconds": round(time.time() - t0, 2),
        "funcs": {
            k: {
                "args": list(names[k]),
                "params": {n: sorted(params[k][n]) for n in names[k] if n in params[k]},
                "return": sorted(rets[k]),
                "calls": calls[k],
            }
            | (
                {"cells": {n: sorted(v) for n, v in sorted(cells[k].items())}}
                if cells.get(k)
                else {}
            )
            | ({"src": keyof.src[k]} if k in keyof.src else {})
            for k in sorted(calls)
            if k not in driver_keys
        },
        "driver": {
            k: {
                "args": list(names[k]),
                "params": {n: sorted(params[k][n]) for n in names[k] if n in params[k]},
                "return": sorted(rets[k]),
                "calls": calls[k],
            }
            for k in sorted(driver_keys)
        },
        "attrs": {
            c: {k: sorted(v) for k, v in sorted(d.items())}
            for c, d in sorted(attrs.items())
        },
    }
    path = a.out or os.path.join(data_dir(), "facts", f"{a.modules[0]}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if a.merge and os.path.exists(path):
        out = merge_facts(json.load(open(path)), out)
    with open(path, "w") as f:
        f.write(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(
        f"configs={out['configs']} built={ok} rejected={rej} functions={len(out['funcs'])} time={out['seconds']}s -> {path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

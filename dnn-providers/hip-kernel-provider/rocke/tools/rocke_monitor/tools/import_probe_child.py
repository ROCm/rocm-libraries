# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The child of tools/import_probe.py: imports rocKE modules and calls their backend="python" entry points in a Python
where `rocke_engine` imports as a stand-in (the platform engine, built) and every other rocKE engine module
(`rocke_<x>_engine`) is not built, whatever the environment holds. Imports nothing of rkmon.

Run as `python -B -P -s import_probe_child.py SPEC RESULTS` with PYTHONPATH set to the tree's import roots. SPEC (JSON):
{"root": the rocKE tree, "built": ["rocke_engine"], "per_call_s": 5, "per_module": 1, "also": [names],
 "modules": [names]}. Each module is imported; then its public functions with a `backend` parameter are called, the
`per_module` cheapest (fewest required arguments, then the first defined) and any named in `also`, with
backend="python", arch="gfx950" and a StandIn for every other required argument. RESULTS gets one JSON line per module
as it goes, so a crash keeps what was done.
"""
import importlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
import re
import signal
import sys
import traceback

ENGINE = re.compile(r"^rocke(?:_\w+)?_engine$")
ARCH_NAMES = {"arch", "gpu_arch", "target_arch"}
ARCH = "gfx950"
# The stand-in's own errors: an attribute it lacks, or a `from rocke_engine import x` (origin "rkmon stand-in").
STAND_IN = re.compile(r"is a stand-in here|rkmon stand-in")


class Timeout(BaseException):
    """A call over its time: a BaseException, so the code under test cannot catch it as an error of its own."""


class StandIn:
    """An argument that answers everything: any attribute, call or item is another StandIn; it is unequal to
    everything and less than nothing; as a number it is 1, as a sequence empty, as text "stand-in". Most code stops on
    it soon (a type check, a dataclass replace); an engine import that comes before the arguments are read does not.
    """

    __slots__ = ()

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return StandIn()

    def __call__(self, *a, **k):
        return StandIn()

    def __getitem__(self, key):
        return StandIn()

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __bool__(self):
        return True

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True

    def __lt__(self, other):
        return False

    __le__ = __gt__ = __ge__ = __lt__

    def __hash__(self):
        return 0

    def __int__(self):
        return 1

    __index__ = __int__

    def __float__(self):
        return 1.0

    def __str__(self):
        return "stand-in"

    __repr__ = __str__

    def __contains__(self, item):
        return False

    def _same(self, *a):
        return StandIn()

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = (
        __rtruediv__
    ) = _same
    __floordiv__ = __rfloordiv__ = __mod__ = __rmod__ = __pow__ = __neg__ = __and__ = (
        __or__
    ) = _same


class _StandInLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        def __getattr__(name):
            raise AttributeError(
                f"{module.__name__} is a stand-in here (rkmon import probe): it has no {name!r}"
            )

        module.__getattr__ = __getattr__
        module.__rkmon_stand_in__ = True


class Engines(importlib.abc.MetaPathFinder):
    """Ahead of every other finder: a rocKE engine module imports as a stand-in when built, else raises
    ModuleNotFoundError; each attempt is logged with where in the tree it came from."""

    def __init__(self, built, root):
        self.built, self.root, self.log = set(built), os.path.realpath(root), []

    def site(self, frames) -> list[list]:
        out = []
        for f in frames:
            path = os.path.realpath(f.filename)
            if path.startswith(self.root + os.sep):
                out.append([os.path.relpath(path, self.root), f.lineno, f.name])
        return out

    def find_spec(self, name, path=None, target=None):
        if "." in name or not ENGINE.match(name):
            return None
        built = name in self.built
        self.log.append(
            {
                "module": name,
                "built": built,
                "site": self.site(traceback.extract_stack()[:-1]),
            }
        )
        if built:
            return importlib.machinery.ModuleSpec(
                name, _StandInLoader(), origin="rkmon stand-in"
            )
        raise ModuleNotFoundError(
            f"No module named {name!r} (not built in this probe)", name=name
        )


def _alarm(signum, frame):
    raise Timeout()


def chain(exc) -> list[BaseException]:
    out = []
    while exc is not None and exc not in out:
        out.append(exc)
        exc = exc.__cause__ or exc.__context__
    return out


def describe(exc: BaseException, eng: Engines) -> dict:
    """An exception as the probe reports it: type, message, its frames in the tree (outermost first), the unbuilt
    engine module an ImportError in its chain names, if any, and whether it came from the stand-in engine (an attribute
    the stand-in lacks: the code calls into the built engine)."""
    missing = next(
        (
            x.name
            for x in chain(exc)
            if isinstance(x, ImportError)
            and getattr(x, "name", None)
            and ENGINE.match(x.name)
            and x.name not in eng.built
        ),
        None,
    )
    stand_in = any(
        isinstance(x, (AttributeError, ImportError)) and STAND_IN.search(str(x))
        for x in chain(exc)
    )
    text = str(exc).strip()
    return {
        "type": type(exc).__name__,
        "msg": text.splitlines()[0][:200] if text else "",
        "missing": missing,
        "frames": eng.site(traceback.extract_tb(exc.__traceback__)),
        "stand_in": stand_in,
    }


def entry_points(mod) -> list[tuple[int, int, str]]:
    """(required arguments, line, name) of each public function the module defines with a `backend` parameter,
    cheapest first."""
    out = []
    for name, fn in vars(mod).items():
        if (
            name.startswith("_")
            or not inspect.isfunction(fn)
            or fn.__module__ != mod.__name__
        ):
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        if "backend" not in params:
            continue
        req = sum(
            1
            for p in params.values()
            if p.default is p.empty
            and p.name != "backend"
            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        )
        out.append((req, fn.__code__.co_firstlineno, name))
    return sorted(out)


def arguments(fn) -> tuple[list, dict]:
    args, kwargs = [], {}
    for p in inspect.signature(fn).parameters.values():
        if p.name == "backend":
            if p.kind == p.POSITIONAL_ONLY:
                args.append("python")
            else:
                kwargs["backend"] = "python"
            continue
        if p.default is not p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        value = ARCH if p.name in ARCH_NAMES else StandIn()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            args.append(value)
        else:
            kwargs[p.name] = value
    return args, kwargs


def timed(fn, seconds: float):
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def probe(name: str, spec: dict, eng: Engines) -> dict:
    per_call = float(spec.get("per_call_s", 5))
    rec = {"module": name, "import": None, "engines": [], "entries": [], "names": []}
    eng.log.clear()
    try:
        mod = timed(lambda: importlib.import_module(name), 2 * per_call)
        rec["import"] = {"ok": True}
    except Timeout:
        rec["import"] = {
            "ok": False,
            "type": "Timeout",
            "msg": f"the import took over {2 * per_call:.0f} s",
            "missing": None,
            "frames": [],
            "stand_in": False,
        }
        return rec
    except BaseException as exc:  # SystemExit from a script's top level too
        rec["import"] = {"ok": False, **describe(exc, eng)}
        return rec
    finally:
        rec["engines"] = list(eng.log)
    eps = entry_points(mod)
    rec["names"] = [[n, line, req] for req, line, n in eps]
    chosen = [n for _, _, n in eps[: int(spec.get("per_module", 1))]]
    chosen += [
        n for _, _, n in eps if n in set(spec.get("also") or ()) and n not in chosen
    ]
    for n in chosen:
        fn = getattr(mod, n)
        eng.log.clear()
        e = {"name": n, "line": fn.__code__.co_firstlineno}
        try:
            args, kwargs = arguments(fn)
            timed(lambda: fn(*args, **kwargs), per_call)
            e.update(ok=True)
        except Timeout:
            e.update(
                ok=False,
                type="Timeout",
                msg=f"still running after {per_call:.0f} s",
                missing=None,
                frames=[],
                stand_in=False,
            )
        except BaseException as exc:
            e.update(ok=False, **describe(exc, eng))
        e["engines"] = list(eng.log)
        rec["entries"].append(e)
    return rec


def main(argv: list[str]) -> int:
    spec = json.loads(open(argv[1]).read())
    eng = Engines(spec.get("built") or ["rocke_engine"], spec["root"])
    sys.meta_path.insert(0, eng)
    for m in [m for m in sys.modules if ENGINE.match(m)]:
        del sys.modules[m]
    sys.argv = ["rkmon-import-probe"]
    signal.signal(signal.SIGALRM, _alarm)
    with open(argv[2], "a") as out:
        for name in spec["modules"]:
            try:
                rec = probe(name, spec, eng)
            except (
                BaseException
            ) as exc:  # the probe's own failure for this module; the next one still runs
                rec = {
                    "module": name,
                    "import": {
                        "ok": False,
                        "type": type(exc).__name__,
                        "msg": str(exc)[:200],
                        "missing": None,
                        "frames": [],
                        "stand_in": False,
                    },
                    "engines": [],
                    "entries": [],
                    "names": [],
                    "probe_error": True,
                }
            out.write(json.dumps(rec) + "\n")
            out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

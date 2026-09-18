# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Record, roll and ship ONE named kernel — the generic form of the gates.

Every other roll driver here carries a hard-coded family list pinned to
``kernels/gfx950``, because each is a gate defending a fixed claim. This one is
the developer-facing tool: you name a kernel module and its axes on the command
line, and it runs the same pipeline the gates run.

    record -> roll (N axes) -> verify .ll/HSACO against the Python oracle
           -> derive guard -> stamp ABI -> write a CBOR bundle

Start with ``--probe``, which is the step most people skip and then misread.
For each axis it answers two separate questions:

  * **does it roll?** — a refusal is a normal, safe outcome, and the reason
    tells you whether it is a modelling gap or a real structural change.
  * **does it matter?** — an axis the emitted program does not depend on
    "rolls" trivially. That is a vacuous pass: the recipe covers the axis
    because nothing varies with it, so the coverage it appears to buy is not
    real. Only a probe that compares recorded programs can tell you this, which
    is why it runs before the roll and not after.

There are two ways in, and they run the same code: :func:`roll` from Python, and
this file from a shell. The command line is the quicker one for a flat spec; the
function is the sane one for a nested spec, where the equivalent invocation is a
wall of ``--fixed`` flags::

    from rocke.portable_ir.drivers.roll_kernel import roll

    r = roll(kernel="rocke.instances.common.gemm_universal", arch="gfx950",
             fixed={"name": "gemm", "tile": {"tile_m": 16, "tile_k": 16},
                    "data.dtype_a": "bf16"},
             axes={"tile": {"tile_n": [32, 64]}}, structural="tile_n",
             holdout={"tile_n": [128]}, verify=True)
    if r:                      # truthy only when everything asked for passed
        ship_it(r.cbor)

``fixed`` and ``axes`` take nested dicts, dotted keys, or both mixed, and
``fixed`` also takes a spec *instance* — usually what a kernel author already
has — whose fields the axes then override.

Spec fields are addressed by name, or by **dotted path** when the spec nests
other dataclasses — ``head_size=64`` for a flat spec like ``WmmaFmhaFwdSpec``,
``tile.tile_m=16`` for a nested one like ``UniversalGemmSpec``. Values are
converted using the field's declared type, and a nested spec that has no
required fields of its own is filled in from its defaults, so you only spell out
the path you actually want to set. An axis is named by its leaf
(``tile.tile_n`` rolls as ``tile_n``), which is both the existing convention and
a requirement of ``kernel_name_fmt``.

**Choosing the target architecture is yours; deciding whether the kernel serves
it is the kernel's.** This driver asks the module's own ``is_valid_spec`` /
``supports_*`` about every point *before* recording, and relays a rejection as a
refusal in the kernel's own words rather than letting it surface as a traceback
from inside the builder. It then re-asks on the other ``known_arches()`` to say
which kind of refusal it was: a spec accepted elsewhere means you aimed at the
wrong target (exit 3, skippable), while one refused everywhere means the flags
are wrong (exit 2).

Examples::

    # triage: which axes are worth rolling?
    python3 -m rocke.portable_ir.drivers.roll_kernel \\
        --kernel kernels.gfx1151.wmma_fmha_fwd --arch gfx1151 \\
        --fixed head_size=64 --fixed mask_mode=causal \\
        --axis num_query_heads=8,16 --axis sliding_window=64,128 --probe

    # a nested spec: only the paths you care about
    python3 -m rocke.portable_ir.drivers.roll_kernel \\
        --kernel rocke.instances.common.gemm_universal --arch gfx950 \\
        --spec UniversalGemmSpec --fixed name=gemm --fixed tile.tile_m=16 \\
        --fixed tile.tile_k=16 --fixed tile.warp_m=1 --fixed tile.warp_n=1 \\
        --axis tile.tile_n=32,64 --holdout tile.tile_n=128 --verify

    # roll, verify byte-identity, and write a shippable bundle
    python3 -m rocke.portable_ir.drivers.roll_kernel \\
        --kernel kernels.gfx1151.wmma_fmha_fwd --arch gfx1151 \\
        --fixed head_size=64 --fixed mask_mode=causal \\
        --axis num_query_heads=8,16 --holdout num_query_heads=32 \\
        --verify --hsaco --guard --out /tmp/wmma_fmha.cbor

Exit status, so a CI matrix can tell the cases apart. :func:`roll` returns the
same number as ``RollResult.code``, and raises :class:`UsageError` for the 2s
rather than exiting the interpreter under a caller:

===  ==========================================================================
0    every requested stage passed
1    a stage failed: parity mismatch, or ``probe`` found a vacuous axis
2    usage error: an unknown arch, a missing or misspelled spec field, or a
     spec the kernel refuses on every known arch
3    refused: the kernel does not serve *this* arch, though it serves others.
     Not a failure. A per-arch matrix should skip on 3, not go red.
===  ==========================================================================

``--verify`` needs a shared ``librocke`` (``ROCKE_ONLINE_LIB``, or
``online.build_lib()``); ``--hsaco`` additionally needs comgr.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib
import inspect
import itertools
import os
import sys
import textwrap
import time
import typing
from typing import Any, Callable, Dict, List, Optional, Tuple

Point = Dict[str, Any]


class UsageError(ValueError):
    """A mistake in how the roll was *described*: a misspelled field, a
    non-integer axis, an arch that does not exist.

    Distinct from a refusal, which is the kernel's considered answer about a
    perfectly well-formed request. ``main`` turns this into exit 2; a caller
    using :func:`roll` from Python catches it like any other ValueError instead
    of having the interpreter exit underneath them."""


@dataclasses.dataclass
class RollResult:
    """What a roll did, for callers who need more than an exit status."""

    code: int  # 0 rolled, 1 failed, 2 usage, 3 refused
    recipe: Optional[Dict[str, Any]] = None
    points: Tuple[Point, ...] = ()
    cbor: Optional[bytes] = None
    parity: Optional[bool] = None
    reason: str = ""
    refusals: List[Tuple[Point, str]] = dataclasses.field(default_factory=list)
    elsewhere: List[str] = dataclasses.field(default_factory=list)
    n_recorded: int = 0  # traces used for INFERENCE, not for verification
    trace_bytes: Tuple[int, ...] = ()  # CBOR size of each concrete trace

    @property
    def rolled(self) -> bool:
        return self.recipe is not None

    @property
    def refused(self) -> bool:
        return self.code == 3

    def __bool__(self) -> bool:
        return self.code == 0


# --------------------------------------------------------------------------
# resolving the kernel module
# --------------------------------------------------------------------------
def _public(mod: Any) -> List[str]:
    """Names the module owns, preferring __all__ over whatever it imported."""
    names = getattr(mod, "__all__", None)
    if names:
        return list(names)
    return [
        n
        for n in dir(mod)
        if not n.startswith("_")
        and getattr(getattr(mod, n), "__module__", None) == mod.__name__
    ]


def _pick(mod: Any, kind: str, want: Optional[str], match: Callable[[Any, str], bool]):
    """The one name of ``kind`` in ``mod``, or a hard error naming the choices."""
    if want:
        if not hasattr(mod, want):
            raise UsageError(f"{mod.__name__} has no {kind} named {want!r}")
        return getattr(mod, want)
    found = [n for n in _public(mod) if match(getattr(mod, n), n)]
    if len(found) == 1:
        return getattr(mod, found[0])
    if not found:
        raise UsageError(f"no {kind} found in {mod.__name__}; pass it explicitly")
    raise UsageError(
        f"{mod.__name__} exposes several {kind}s ({', '.join(sorted(found))}); "
        f"pick one explicitly"
    )


@dataclasses.dataclass
class Kernel:
    """How to make, gate and build one kernel's spec.

    Naming a module covers the common case, and :func:`resolve` fills this in
    from that module's conventions. The fields exist because the surveys in this
    directory legitimately do not fit those conventions: ``attention_dense``
    gates through ``supports_*`` with a dozen keyword arguments rather than a
    spec, ``fastkv_regp`` builds its spec from another kernel's spec, and the
    examples ``qk_block`` and ``export_mha`` have no spec dataclass at all.
    Rather than let each survey re-grow the record-and-roll plumbing around
    those differences, each one describes itself here once."""

    label: str = ""
    build_fn: Optional[Callable[..., Any]] = None  # (spec, arch=) -> IRBuilder
    spec_cls: Optional[type] = None
    make_spec: Optional[Callable[..., Any]] = None  # (**fields) -> spec
    gate: Optional[Callable[..., Any]] = None  # is_valid_spec / supports_*
    coherent: Optional[Callable[[Point], bool]] = None
    build_at: Optional[Callable[..., Any]] = None  # (**point) -> IRBuilder
    gate_note: str = ""  # why there is no gate, when one was found but unusable

    def spec_from(self, values: Dict[str, Any]) -> Any:
        if self.make_spec is not None:
            return self.make_spec(**values)
        if self.spec_cls is None:
            raise UsageError(f"{self.label or 'kernel'} has no spec to construct")
        return _construct(self.spec_cls, values)

    @property
    def spec_name(self) -> str:
        if self.spec_cls is not None:
            return self.spec_cls.__name__
        return getattr(self.make_spec, "__name__", "(callable)")


def resolve(
    kernel: Any, build: Optional[str] = None, spec: Optional[str] = None
) -> Kernel:
    """A :class:`Kernel` for a module path, a module, a callable, or one already.

    A bare callable is taken as the builder itself and called with the point,
    which is how the example kernels and any ad-hoc closure get in."""
    if isinstance(kernel, Kernel):
        return kernel
    if callable(kernel) and not isinstance(kernel, type):
        return Kernel(
            label=getattr(kernel, "__name__", "kernel"),
            build_at=kernel,
        )
    mod = importlib.import_module(kernel) if isinstance(kernel, str) else kernel
    gate = getattr(mod, "is_valid_spec", None)
    if gate is None:  # the other convention in this tree
        cands = [n for n in _public(mod) if n.startswith("supports_")]
        gate = getattr(mod, cands[0]) if len(cands) == 1 else None
    note = ""
    if gate is not None and gate_call(gate) is None:
        note = (
            f"{gate.__name__} takes keyword arguments describing a shape rather "
            f"than a spec,\n         so it cannot be asked about the one being "
            f"built. Pass Kernel(gate=...) to\n         adapt it and have the "
            f"target checked before anything is recorded."
        )
        gate = None
    return Kernel(
        label=mod.__name__,
        gate_note=note,
        spec_cls=_pick(
            mod, "spec dataclass", spec, lambda o, n: dataclasses.is_dataclass(o)
        ),
        build_fn=_pick(
            mod,
            "build function",
            build,
            lambda o, n: callable(o) and n.startswith("build"),
        ),
        gate=gate,
    )


# --------------------------------------------------------------------------
# spec fields: flat or nested
# --------------------------------------------------------------------------
# Specs in this tree come in both shapes. WmmaFmhaFwdSpec is flat, so
# ``head_size=64`` is the whole story. UniversalGemmSpec nests TileSpec,
# TraitSpec and DataSpec, so a field is addressed by a dotted path,
# ``tile.tile_m=16``, and the intermediate dataclasses are built on the way in.
def _hints(cls: Any) -> Dict[str, Any]:
    """Resolved annotations. Needed because these modules use PEP 563, so
    ``field.type`` is the *string* "TileSpec" rather than the class."""
    try:
        return typing.get_type_hints(cls)
    except Exception:  # noqa: BLE001 - fall back to unresolved, still usable
        return {f.name: f.type for f in dataclasses.fields(cls)}


def _is_dataclass_type(ann: Any) -> bool:
    return isinstance(ann, type) and dataclasses.is_dataclass(ann)


def _defaulted(f: dataclasses.Field) -> bool:
    return (
        f.default is not dataclasses.MISSING
        or f.default_factory is not dataclasses.MISSING  # type: ignore[misc]
    )


def _coerce(text: str, ann: Any = None) -> Any:
    """A CLI string as the field's declared type, falling back to a guess.

    Guessing alone is wrong for a ``str`` field holding digits — a kernel named
    ``123`` would silently become the integer 123 — so the annotation wins
    whenever we could resolve one."""
    if typing.get_origin(ann) is typing.Union:  # unwrap Optional[X]
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        ann = args[0] if len(args) == 1 else None
    if ann is str:
        return text
    if ann is bool:
        return text.strip().lower() in ("1", "true", "yes", "on")
    if ann is int:
        try:
            return int(text)
        except ValueError:
            raise UsageError(f"expected an int, got {text!r}")
    if ann is float:
        return float(text)
    try:
        return int(text)
    except ValueError:
        pass
    low = text.strip().lower()
    return low == "true" if low in ("true", "false") else text


def _put(tree: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    for part in path[:-1]:
        tree = tree.setdefault(part, {})
    tree[path[-1]] = value


def _ann_at(cls: Any, path: Tuple[str, ...]) -> Any:
    """The declared type of a dotted path, or None if it cannot be followed."""
    ann: Any = cls
    for part in path:
        if not _is_dataclass_type(ann):
            return None
        ann = _hints(ann).get(part)
    return ann


def _construct(cls: Any, values: Dict[str, Any], where: str = "") -> Any:
    """Build ``cls`` from a nested dict, recursing into dataclass fields.

    A required field that is itself a dataclass with no required fields of its
    own is filled in with its defaults, so ``UniversalGemmSpec`` does not make
    you spell out ``trait`` and ``data`` just to reach ``tile``."""
    types = _hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    kwargs: Dict[str, Any] = {}
    for name, value in values.items():
        if name not in known:
            raise UsageError(
                f"no such spec field {where + name!r}; "
                f"{cls.__name__} has: {', '.join(sorted(known))}"
            )
        ann = types.get(name)
        if isinstance(value, dict):
            if not _is_dataclass_type(ann):
                raise UsageError(
                    f"{where}{name} is not a nested spec, so it takes a value, "
                    f"not {where}{name}.<field>"
                )
            kwargs[name] = _construct(ann, value, f"{where}{name}.")
        else:
            kwargs[name] = _coerce(value, ann) if isinstance(value, str) else value

    for f in dataclasses.fields(cls):
        if f.name in kwargs or _defaulted(f):
            continue
        ann = types.get(f.name)
        if _is_dataclass_type(ann) and all(
            _defaulted(x) for x in dataclasses.fields(ann)
        ):
            kwargs[f.name] = ann()
    return cls(**kwargs)


def _missing(cls: Any, given: Dict[str, Any], where: str = "") -> List[str]:
    """Dotted paths still needed to construct ``cls``, for the error message."""
    out: List[str] = []
    types = _hints(cls)
    for f in dataclasses.fields(cls):
        ann = types.get(f.name)
        sub = given.get(f.name)
        if _is_dataclass_type(ann):
            if isinstance(sub, dict) or not _defaulted(f):
                out += _missing(
                    ann, sub if isinstance(sub, dict) else {}, f"{where}{f.name}."
                )
        elif f.name not in given and not _defaulted(f):
            out.append(f"{where}{f.name}")
    return out


def _kv(items: List[str], *, many: bool) -> Dict[Tuple[str, ...], Any]:
    """``a.b=v`` (or ``a.b=v1,v2`` when ``many``) into {path tuple: value}."""
    out: Dict[Tuple[str, ...], Any] = {}
    for item in items:
        if "=" not in item:
            raise UsageError(f"expected name=value, got {item!r}")
        name, _, rhs = item.partition("=")
        path = tuple(p.strip() for p in name.strip().split(".") if p.strip())
        if not path:
            raise UsageError(f"empty field name in {item!r}")
        vals = [v for v in rhs.split(",") if v != ""]
        if not vals:
            raise UsageError(f"no value given for {name!r}")
        out[path] = vals if many else vals[0]
    return out


def _flatten(obj: Any, where: Tuple[str, ...] = ()) -> Dict[Tuple[str, ...], Any]:
    """Any of the three ways to say the same thing, into {path tuple: value}.

    ``{"tile": {"tile_m": 16}}``, ``{"tile.tile_m": 16}`` and a spec instance
    all flatten to ``{("tile", "tile_m"): 16}``. Recursion stops at anything
    that is not a dict, so an axis's list of values arrives intact."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    if not isinstance(obj, dict):
        return {where: obj}
    out: Dict[Tuple[str, ...], Any] = {}
    for key, value in obj.items():
        path = where + tuple(p for p in str(key).split(".") if p)
        if not path:
            raise UsageError(f"empty field name in {obj!r}")
        if isinstance(value, dict):
            out.update(_flatten(value, path))
        else:
            out[path] = value
    return out


def _summarize(given: Any, paths: Dict[Tuple[str, ...], Any]) -> str:
    """The fixed fields, echoed back. A spec instance flattens to every field it
    has, defaults included, which is 40-odd lines of noise for a GEMM, so that
    case reports the instance instead of reciting it."""
    if not paths:
        return "(none)"
    if dataclasses.is_dataclass(given) and not isinstance(given, type):
        return f"the given {type(given).__name__} ({len(paths)} fields)"
    body = ", ".join(f"{'.'.join(p)}={v}" for p, v in paths.items())
    return textwrap.fill(body, width=94, subsequent_indent=" " * 12)


def _listed(value: Any, name: str, what: str) -> List[Any]:
    """A list of values, accepting a lone value as a list of one."""
    if isinstance(value, (list, tuple, set)):
        out = list(value)
    else:
        out = [value]
    if not out:
        raise UsageError(f"{what} {name!r} has no values")
    return out


def _axis_names(paths: List[Tuple[str, ...]]) -> Dict[str, Tuple[str, ...]]:
    """Leaf name -> path. The recipe sees the leaf, which is both the existing
    convention (the gates roll ``tile_n``, not ``tile.tile_n``) and a hard
    requirement: an axis name reaches ``kernel_name_fmt`` as a ``{placeholder}``,
    and a dot there means attribute access to Python's formatter."""
    out: Dict[str, Tuple[str, ...]] = {}
    for path in paths:
        leaf = path[-1]
        if leaf in out:
            raise UsageError(
                f"two axes share the leaf name {leaf!r} ({'.'.join(out[leaf])} "
                f"and {'.'.join(path)}); rename one in the spec to roll both"
            )
        out[leaf] = path
    return out


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]


def recipe_ops(program: Any) -> int:
    """Instructions in a recipe program, counting into nested bodies.

    ``param`` entries are declarations rather than instructions, so they do not
    count; a recipe's size in ops is meant to be comparable to the concrete
    trace it replaces."""
    if isinstance(program, dict):
        program = program.get("program", [])
    n = 0
    for inst in program or []:
        if inst.get("op") != "param":
            n += 1
        for key in ("body", "then", "else"):
            if key in inst:
                n += recipe_ops(inst[key])
    return n


def gate_call(gate: Any) -> Optional[str]:
    """How to hand a spec to a gate: "arch", "spec", "probe", or None for one
    that cannot be asked about a spec at all.

    Both conventions here put the spec first, but ``is_valid_spec(spec, arch)``
    leaves the arch positional while ``supports_attention_dense(spec, *, arch=)``
    makes it keyword-only. Passing it positionally and falling back on TypeError
    looks like it handles both and does not: the fallback drops the arch, so the
    gate answers about its *default* target while the recipe is built for the
    requested one. Sending it by keyword is what both actually accept.

    A ``supports_*`` taking only keyword arguments (``head_size=``,
    ``block_size=``, ...) describes a shape rather than a spec, and there is no
    honest way to ask it about one."""
    try:
        params = inspect.signature(gate).parameters
    except (TypeError, ValueError):  # a builtin or C callable
        return "probe"
    kinds = [p.kind for p in params.values()]
    if (
        not any(
            k
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for k in kinds
        )
        and inspect.Parameter.VAR_POSITIONAL not in kinds
    ):
        return None
    return "arch" if "arch" in params else "spec"


def _verdict(gate: Any, spec: Any, arch: str) -> Tuple[bool, str]:
    """Normalize the two gate conventions to ``(ok, reason)``.

    ``is_valid_spec`` returns the pair; some ``supports_*`` return a bare bool."""
    how = gate_call(gate)
    if how == "arch":
        out = gate(spec, arch=arch)
    elif how == "spec":
        out = gate(spec)
    elif how is None:
        raise UsageError(
            f"{getattr(gate, '__name__', 'the gate')} cannot be asked about a spec; "
            f"describe the kernel with Kernel(gate=...) instead"
        )
    else:
        try:
            out = gate(spec, arch)
        except TypeError:
            out = gate(spec)
    if isinstance(out, tuple):
        return bool(out[0]), str(out[1]) if len(out) > 1 else ""
    return bool(out), ""


def arch_refusals(bound: Kernel, make_values, points: List[Point], arch: str):
    """Ask the kernel's own gate about every point, before recording anything.

    Choosing the target is the caller's job; saying whether the kernel accepts
    it is the kernel's. This driver only has to relay that answer clearly and
    early, so a target a kernel does not serve reads as a refusal with the
    kernel's own words rather than a stack trace from somewhere inside the
    builder — and so a CI matrix can tell "not applicable here" apart from
    "broken here" by exit status alone."""
    if bound.gate is None and bound.coherent is None:
        return []
    out = []
    for point in points:
        if bound.coherent is not None and not bound.coherent(point):
            out.append((point, "incoherent combination (see the kernel's own note)"))
            continue
        if bound.gate is None:
            continue
        try:
            spec = bound.spec_from(make_values(point))
        except UsageError:
            raise
        except Exception as e:  # noqa: BLE001 - __post_init__ rejects per-field
            out.append((point, f"{type(e).__name__}: {e}"))
            continue
        ok, why = _verdict(bound.gate, spec, arch)
        if not ok:
            out.append((point, why or "gate returned False"))
    return out


def accepted_elsewhere(bound: Kernel, make_values, points, arch: str) -> List[str]:
    """Known arches that accept every one of ``points``.

    This is what separates "you aimed at the wrong target" from "this spec is
    refused everywhere". Both arrive as the same gate refusal, but only the
    first is a target choice: a tile config the kernel rejects on every arch is
    a flag to fix, not a matrix entry to skip."""
    if bound.gate is None:
        return []
    try:
        from rocke.core.arch import known_arches
    except Exception:  # noqa: BLE001 - the disambiguation is a nicety
        return []
    out = []
    for other in known_arches():
        if other == arch:
            continue
        ok = True
        for point in points:
            try:
                spec = bound.spec_from(make_values(point))
                ok, _ = _verdict(bound.gate, spec, other)
            except Exception:  # noqa: BLE001
                ok = False
            if not ok:
                break
        if ok:
            out.append(other)
    return out


def probe_axes(
    build_at, axes: Dict[str, List[Any]], structural: Optional[str], say=print
) -> bool:
    """Per-axis triage. Returns False only if an axis is VACUOUS.

    A refusal is a normal outcome everywhere else in this tree and it is one
    here: an axis that declines costs coverage, not correctness, and some will
    decline until the roller grows. Failing on that would leave a CI job
    permanently red for a known gap. A vacuous axis is different — it is an
    authoring mistake that quietly claims coverage it does not have."""
    from rocke.portable_ir.src import recipe_bundle
    from rocke.portable_ir.src.recording_builder import record_kernel
    from rocke.portable_ir.src.roll_nd import roll_nd

    base = {a: v[0] for a, v in axes.items()}
    say("-- per-axis probe: does it roll, and does it change the program? --\n")
    say("%-22s %-10s %s" % ("axis", "verdict", "detail"))
    clean = True

    for axis, values in axes.items():
        if len(values) < 2:
            say("%-22s %-10s %s" % (axis, "skipped", "needs >= 2 sample values"))
            continue

        # Does the recorded program actually depend on this axis?
        traces = []
        for v in values[:2]:
            _, rec = record_kernel(lambda p={**base, axis: v}: build_at(**p))
            traces.append(recipe_bundle.cbor_encode(rec))
        if traces[0] == traces[1]:
            clean = False
            say(
                "%-22s %-10s %s"
                % (
                    axis,
                    "VACUOUS",
                    f"identical program at {values[0]} and {values[1]} — "
                    f"rolling it proves nothing",
                )
            )
            continue

        r = roll_nd(
            lambda **p: build_at(**{**base, **p}),
            axes={axis: list(values)},
            structural_axis=axis if structural == axis else None,
            extra_spec={},
        )
        if r.ok:
            say("%-22s %-10s %s" % (axis, "rolls", f"{len(r.points)} points verified"))
        else:
            say("%-22s %-10s %s" % (axis, "declines", r.reason[:96]))
    return clean


def verify_parity(
    build_at, recipe, points: List[Point], arch: str, hsaco: bool, say=print
) -> bool:
    """Replay each point through the C engine and diff against the oracle."""
    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.portable_ir.src import online, recipe_bundle

    cbor = recipe_bundle.cbor_encode(recipe)
    flavor = os.environ.get("ROCKE_LLVM_FLAVOR", "")
    say("\n-- verify: Python oracle vs C replay of the rolled recipe --\n")
    say("%-38s %-8s %-14s %s" % ("point", ".ll", "ll sha", "HSACO"))
    ok = True
    for p in points:
        py_ll = lower_kernel_to_llvm(
            build_at(**p), arch=arch, **({"llvm_flavor": flavor} if flavor else {})
        )
        vm_ll, _ = online.recipe_cbor_to_llvm(cbor, arch=arch, ints=dict(p))
        same = py_ll == vm_ll
        ok &= same
        cell = "-"
        if hsaco:
            from rocke.core.arch import ArchTarget
            from rocke.runtime.comgr import build_hsaco_from_llvm_ir

            isa = ArchTarget.from_gfx(arch).isa_triple
            py_h, _ = build_hsaco_from_llvm_ir(py_ll, isa=isa, options=["-O3"])
            vm_h, _ = build_hsaco_from_llvm_ir(vm_ll, isa=isa, options=["-O3"])
            ok &= py_h == vm_h
            cell = f"{_sha(py_h)} ({len(py_h)} B)" if py_h == vm_h else "DIFFER"
        label = " ".join(f"{k}={v}" for k, v in sorted(p.items()))
        say(
            "%-38s %-8s %-14s %s"
            % (label[:38], "EXACT" if same else "DIFFER", _sha(py_ll.encode()), cell)
        )
    return ok


def ship(
    recipe, make_spec, gate_fn, domain, arch, out: str, want_guard: bool, say=print
):
    """Attach a guard, stamp the wire ABI, and write a one-entry bundle.

    ``domain`` is the set of values the bundle is meant to *serve*, which is not
    the set it was fitted from. Deriving a guard from the two sample points
    produces a rule admitting exactly those two, so the recipe refuses shapes it
    replays byte-identically — over-strict, and silently so."""
    from rocke.portable_ir.src import abi as _abi
    from rocke.portable_ir.src import recipe_bundle

    say("\n-- ship --\n")
    if want_guard:
        from rocke.portable_ir.src.guard import derive_guard, gate_from_spec

        if gate_fn is None:
            say("   guard   : skipped (module exposes no is_valid_spec/supports_*)")
        else:
            gate = gate_from_spec(
                make_spec, admits=lambda s: _verdict(gate_fn, s, arch)
            )
            t0 = time.perf_counter()
            guard = derive_guard(
                gate, {a: list(v) for a, v in domain.items()}, arch=arch
            )
            recipe = {**recipe, "guard": guard}
            say(
                "   guard   : %d rule(s) over %s, derived in %.0f ms"
                % (
                    len(guard.get("rules", [])),
                    ", ".join(f"{a}[{len(v)}]" for a, v in domain.items()),
                    (time.perf_counter() - t0) * 1e3,
                )
            )
            for rule in guard.get("rules", []):
                say(f"             - {rule.get('reason', '')}")

    key = recipe.get("kernel_name_fmt") or recipe.get("kernel_name") or "kernel"
    bundle = recipe_bundle.build_bundle([{"key": key, "arch": arch, "recipe": recipe}])
    blob = recipe_bundle.cbor_encode(bundle)
    with open(out, "wb") as fh:
        fh.write(blob)
    say("   abi     :", _abi.describe(bundle))
    say("   key     :", key)
    say("   wrote   : %s (%.1f KiB)" % (out, len(blob) / 1024.0))


# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--kernel", required=True, help="module path, e.g. kernels.gfx.foo")
    ap.add_argument("--arch", required=True)
    ap.add_argument(
        "--build", default=None, help="build fn name (default: the only one)"
    )
    ap.add_argument(
        "--spec", default=None, help="spec class name (default: the only one)"
    )
    ap.add_argument(
        "--axis",
        action="append",
        default=[],
        metavar="PATH=V1,V2",
        help="a free integer axis and its >=2 sample values; PATH may be dotted "
        "for a nested spec, and the axis is named by its leaf; repeatable",
    )
    ap.add_argument(
        "--holdout",
        action="append",
        default=[],
        metavar="PATH=V",
        help="values never used for fitting, verified after; repeatable",
    )
    ap.add_argument(
        "--fixed",
        action="append",
        default=[],
        metavar="PATH=V",
        help="spec fields held constant and baked into the recipe; PATH may be "
        "dotted, e.g. tile.tile_m=16; repeatable",
    )
    ap.add_argument(
        "--domain",
        action="append",
        default=[],
        metavar="NAME=V1,V2,..",
        help="every value the guard should ADMIT for an axis — the shapes you "
        "intend to serve, not the ones you fitted from. Defaults to the sample "
        "values plus the holdouts, which is almost certainly too narrow",
    )
    ap.add_argument("--structural", default=None, help="the one axis that may reshape")
    ap.add_argument("--probe", action="store_true", help="per-axis triage, then stop")
    ap.add_argument("--verify", action="store_true", help=".ll parity vs the oracle")
    ap.add_argument("--hsaco", action="store_true", help="and compare HSACO (comgr)")
    ap.add_argument("--guard", action="store_true", help="derive an admission guard")
    ap.add_argument("--out", default="", help="write a CBOR bundle here")
    args = ap.parse_args(argv)

    os.environ.setdefault("ROCKE_CPP_QUIET_FALLBACK", "1")
    return roll(
        kernel=args.kernel,
        arch=args.arch,
        fixed={".".join(p): v for p, v in _kv(args.fixed, many=False).items()},
        axes={".".join(p): v for p, v in _kv(args.axis, many=True).items()},
        holdout={".".join(p): v for p, v in _kv(args.holdout, many=True).items()},
        domain={".".join(p): v for p, v in _kv(args.domain, many=True).items()},
        structural=args.structural,
        build=args.build,
        spec=args.spec,
        probe=args.probe,
        verify=args.verify,
        hsaco=args.hsaco,
        guard=args.guard,
        out=args.out,
    ).code


def roll(
    kernel: Any,
    arch: str,
    axes: Any,
    fixed: Any = None,
    holdout: Any = None,
    domain: Any = None,
    structural: Optional[str] = None,
    build: Optional[str] = None,
    spec: Optional[str] = None,
    probe: bool = False,
    verify: bool = False,
    hsaco: bool = False,
    guard: bool = False,
    out: str = "",
    extra_spec: Optional[Dict[str, Any]] = None,
    quiet: bool = False,
) -> RollResult:
    """Record, roll, verify and ship one kernel — the CLI's whole body.

    ``fixed`` and ``axes`` take nested dicts, dotted keys, or both mixed; the
    values may already be typed, so this is the comfortable way in for a nested
    spec, where the command line turns into a wall of ``--fixed`` flags::

        roll(kernel="rocke.instances.common.gemm_universal", arch="gfx950",
             fixed={"name": "g", "tile": {"tile_m": 16, "tile_k": 16},
                    "data.dtype_a": "bf16"},
             axes={"tile": {"tile_n": [32, 64]}}, structural="tile_n")

    ``fixed`` also accepts a spec *instance*, which is usually what a kernel
    author already has: build it the normal way, then name the fields to vary
    and they override that base.

    ``kernel`` is a module path, a module, a :class:`Kernel` describing a kernel
    that does not follow this tree's conventions, or a plain callable taking the
    point — which is how the examples with no spec dataclass get in.

    ``holdout`` and ``domain`` are keyed by axis (so by leaf name), and take a
    list of values each. ``extra_spec`` adds non-axis parameters to the recipe's
    spec declaration, and ``quiet`` suppresses the narration for a caller
    printing its own table. Raises :class:`UsageError` if the request itself is
    malformed; a refusal by the kernel comes back as a ``RollResult``."""
    say = (lambda *a, **k: None) if quiet else print
    axis_paths = _flatten(axes)
    fixed_paths = _flatten(fixed) if fixed is not None else {}
    if not axis_paths:
        raise UsageError("need at least one axis")

    try:  # a typo'd target is a usage error, not a refusal
        from rocke.core.arch import known_arches, validate_arch

        validate_arch(arch)
    except ImportError:
        pass
    except UsageError:
        raise
    except Exception as e:  # noqa: BLE001
        raise UsageError(f"{e}\nknown arches: {', '.join(known_arches())}")

    bound = resolve(kernel, build, spec)
    spec_cls, gate_fn = bound.spec_cls, bound.gate
    kernel_name = bound.label or str(kernel)

    # Axes are addressed by path but named by their leaf everywhere downstream:
    # in `axes`, in holdouts, in the recipe's spec declaration and in
    # kernel_name_fmt.
    where = _axis_names(list(axis_paths))
    axes = {
        leaf: [
            _coerce(v, _ann_at(spec_cls, path))
            for v in _listed(axis_paths[path], leaf, "axis")
        ]
        for leaf, path in where.items()
    }
    for leaf, values in axes.items():
        bad = [v for v in values if not isinstance(v, int) or isinstance(v, bool)]
        if bad:
            raise UsageError(
                f"axis {leaf!r} has non-integer values {bad}; rolling is over "
                f"integer axes, so pass a fixed value with --fixed instead"
            )
    holds, declared_domain = (
        {
            p[-1]: [
                _coerce(v, _ann_at(spec_cls, where.get(p[-1], p)))
                for v in _listed(vals, p[-1], what)
            ]
            for p, vals in _flatten(given or {}).items()
        }
        for given, what in ((holdout, "holdout"), (domain, "domain"))
    )
    for what, named in (("holdout", holds), ("domain", declared_domain)):
        stray = [n for n in named if n not in axes]
        if stray:
            raise UsageError(
                f"{what} names non-axes: {', '.join(stray)} "
                f"(axes here are {', '.join(axes)})"
            )
    if structural and structural not in axes:
        raise UsageError(
            f"structural {structural!r} is not one of the axes ({', '.join(axes)})"
        )

    fixed_tree: Dict[str, Any] = {}
    for path, value in fixed_paths.items():
        _put(fixed_tree, path, value)

    def values_for(point: Point) -> Dict[str, Any]:
        tree = copy.deepcopy(fixed_tree)
        for leaf, value in point.items():
            _put(tree, where[leaf], value)
        return tree

    if bound.build_at is not None:
        build_at = lambda **point: bound.build_at(**{**values_for(point)})  # noqa: E731
    else:

        def build_at(**point: Any):
            return bound.build_fn(bound.spec_from(values_for(point)), arch=arch)

    makes_spec = bound.spec_cls is not None or bound.make_spec is not None
    say(f"== {kernel_name} on {arch} ==")
    if makes_spec:
        builder = bound.build_fn or bound.build_at
        say(f"   spec   : {bound.spec_name}   build: {builder.__name__}")
    say(f"   fixed  : {_summarize(fixed, fixed_paths)}")
    say("   axes   : " + ", ".join(f"{a}={v}" for a, v in axes.items()))
    say(
        "   holdout: "
        + (", ".join(f"{a}={v}" for a, v in holds.items()) if holds else "(none)")
        + "\n"
    )

    # Every point the run will touch, so an unbuildable spec or an arch the
    # kernel does not serve is reported now rather than from inside the roller.
    base = {a: v[0] for a, v in axes.items()}
    grid = [
        dict(zip(axes, combo))
        for combo in itertools.islice(itertools.product(*axes.values()), 64)
    ]
    try:
        if makes_spec:
            bound.spec_from(values_for(base))
    except UsageError:
        raise
    except TypeError as e:
        say(f"cannot build the spec at the base point {base}: {e}")
        need = _missing(spec_cls, values_for(base)) if spec_cls else []
        if need:
            say(f"   {bound.spec_name} still needs: {', '.join(need)}")
            say("   name each one in fixed= or axes=, dotted for a nested spec")
            say("   (--fixed tile.tile_m=16 / --axis tile.tile_n=32,64 on the CLI)")
        return RollResult(2, reason=str(e))
    except Exception as e:  # noqa: BLE001 - __post_init__ rejecting is an answer
        say(f"REFUSED: {bound.spec_name} rejects the base point {base}")
        say(f"   {type(e).__name__}: {e}")
        return RollResult(3, reason=str(e), refusals=[(base, str(e))])

    refused = arch_refusals(bound, values_for, grid, arch)
    if refused:
        total = len(grid)
        head = (
            "every point" if len(refused) == total else f"{len(refused)}/{total} points"
        )
        say(f"REFUSED on {arch}: the kernel's own gate rejects {head}.\n")
        seen = set()
        for point, why in refused:
            if why not in seen:
                seen.add(why)
                say(f"   {point}\n      {why}")

        elsewhere = accepted_elsewhere(bound, values_for, [p for p, _ in refused], arch)
        answer = RollResult(
            3, reason=refused[0][1], refusals=refused, elsewhere=elsewhere
        )
        if not elsewhere:
            say(
                "\nThis is the spec, not the target: the same points are refused on"
                "\nevery other known arch too. Fix the request rather than the matrix."
            )
            answer.code = 2
            return answer
        say(f"\n   the same spec is accepted on: {', '.join(elsewhere)}")
        if len(refused) == total:
            say(
                f"\nChoosing the target is yours; whether the kernel serves it is"
                f"\nthe kernel's answer, and {kernel_name.rsplit('.', 1)[-1]} says no"
                f"\nfor {arch}. Nothing was recorded. Exit 3 marks this not-applicable"
                f"\nrather than broken, so a per-arch matrix can skip it, not go red."
            )
        else:
            say(
                "\nThe accepted points could still roll — drop the refused values"
                "\nfrom the axis, and keep them out of the domain so the guard agrees."
            )
        return answer
    if gate_fn is None and makes_spec:
        say(
            f"   note: {bound.gate_note}\n"
            if bound.gate_note
            else f"   note: {kernel_name} exposes no is_valid_spec/supports_* gate,\n"
            f"         so nothing verified that it serves {arch} before build.\n"
        )

    if probe:
        return RollResult(0 if probe_axes(build_at, axes, structural, say) else 1)

    if verify or hsaco:
        from rocke.portable_ir.src import online

        online.load()

    from rocke.portable_ir.src.roll_nd import roll_nd

    # A holdout must name every axis, so a per-axis list becomes a point list by
    # position, with unnamed axes held at their base value.
    n_hold = max((len(v) for v in holds.values()), default=0)
    hold_points = [
        {
            a: (holds[a][i] if a in holds and i < len(holds[a]) else axes[a][0])
            for a in axes
        }
        for i in range(n_hold)
    ]

    t0 = time.perf_counter()
    try:
        r = roll_nd(
            build_at,
            axes=axes,
            structural_axis=structural,
            holdout_points=hold_points,
            extra_spec=dict(extra_spec or {}),
        )
    except Exception as e:  # noqa: BLE001 - a builder assert inside a combination
        say(f"DECLINED: the builder raised {type(e).__name__}: {e}")
        return RollResult(1, reason=f"{type(e).__name__}: {e}")
    if not r.ok:
        say(f"DECLINED after {(time.perf_counter() - t0) * 1e3:.0f} ms")
        say(f"   {r.reason}")
        say("\nThe concrete path still works: ship r.traces as per-point recipes.")
        say("Run with probe=True to see which single axis is responsible.")
        return RollResult(1, reason=r.reason)

    from rocke.portable_ir.src import recipe_bundle

    cbor = recipe_bundle.cbor_encode(r.recipe)
    trace_bytes = [len(recipe_bundle.cbor_encode(t)) for t in r.traces.values()]
    say("ROLLED in %.0f ms" % ((time.perf_counter() - t0) * 1e3))
    say(f"   recorded {r.n_recorded} trace(s), verified {len(r.points)} point(s)")
    say(f"   name_fmt : {r.recipe.get('kernel_name_fmt')}")
    say(
        "   CBOR     : %.1f KiB parametric vs %.1f KiB for the same points concrete"
        % (len(cbor) / 1024.0, sum(trace_bytes) / 1024.0)
    )

    ok: Optional[bool] = None
    if verify or hsaco:
        ok = verify_parity(build_at, r.recipe, r.points, arch, hsaco, say)
        say("\n  " + ("all points byte-identical" if ok else "PARITY FAILED"))

    if out or guard:
        admits = {
            a: declared_domain.get(a, sorted(set(axes[a]) | set(holds.get(a, []))))
            for a in axes
        }
        if guard and not declared_domain:
            say(
                "\n   note: no domain given, so the guard can only admit the "
                "values seen here.\n         Pass the real serving domain or the "
                "bundle will refuse shapes it can build."
            )
        ship(
            r.recipe,
            lambda **p: bound.spec_from(values_for(p)),
            gate_fn,
            admits,
            arch,
            out or "/dev/null",
            guard,
            say=say,
        )
    return RollResult(
        0 if ok is not False else 1,
        recipe=r.recipe,
        points=tuple(r.points),
        cbor=cbor,
        parity=ok,
        n_recorded=r.n_recorded,
        trace_bytes=tuple(trace_bytes),
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except UsageError as bad:
        # A malformed request exits 2 like argparse's own errors, so that 1
        # keeps meaning "a stage failed".
        print(bad, file=sys.stderr)
        sys.exit(2)

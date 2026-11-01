# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Knob spaces of the true monokernel as search spaces of the rocKE search
engine (:mod:`rocke.dispatch.tuning.search`).

Every lever of the monokernel is a default-off knob on some frozen dataclass
(``FrameworkKnobs``, ``FusedKnobs``, ``B3Knobs``, ``MxDecodeGemvSpec``,
``CclKnobs``, ``ToyChainSpec`` ...), on ATOM's schedule string
(``ATOM_Q38_SCHED``) or on the Rust direct compiler's lever list
(``ROCKE_DIRECT_LEVERS``). A :class:`KnobAdapter` turns one of them into a
:class:`~rocke.dispatch.tuning.search.params.SearchSpace`:

- **domains** (:class:`Knob`): a dotted path into the (nested) dataclass, its
  levels, kind, optional gate, numerics class (``bitwise`` or an accuracy-gated
  ``tolerance`` lever, DESIGN rev3 12.4) and whether it is *structural* (a
  base change that re-opens the lookback, DESIGN 9.2 step 3). The shipped
  default is level 0 of every categorical / bool knob, so the all-zero genome
  is the shipped build and every gate is "off at its default".
- **legality pruning**: a config is legal when the owner builds it (its
  ``__post_init__`` / ``is_valid_spec`` accepts it) and the adapter's extra
  rules hold (LDS arena, known miscompiles, ...). Illegal configs never reach a
  compile: the engine repairs them to the nearest legal config.
- **canonical identity**: knobs equal to the base value are dropped, gated-off
  knobs are dropped, an optional ``normalize`` folds meaningless combinations;
  the key is :func:`rocke.dispatch.tuning.identity.config_key` over the family
  ABI, arch, adapter, variant, the canonical knobs and a fingerprint of the
  defaults they are relative to (a changed in-tree default re-keys every
  config, so a stored record can never silently mean another build). The
  tuning id is ``<adapter>.<variant>@<config_key>``.

:class:`CompositeKnobs` joins adapters (``fw.`` + ``moe.`` + ``sched.`` ...)
into one round- or layer-level space with one key.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from rocke.dispatch.tuning.identity import (
    config_key,
    defaults_fingerprint,
    knob_items,
    tuning_id,
)
from rocke.dispatch.tuning.search.params import BOOL, CATEGORICAL, ORDINAL, Param, SearchSpace

__all__ = [
    "SEQ_SEP",
    "seq_decode",
    "seq_encode",
    "ABI",
    "BITWISE",
    "TOLERANCE",
    "CompositeKnobs",
    "DataclassKnobs",
    "Knob",
    "KnobAdapter",
    "LeverStringKnobs",
    "flatten_instance",
]

ABI = "qwen38_mono.tune/v1"
SEQ_SEP = ">"


def seq_decode(value) -> tuple:
    """``"a>b>c"`` -> ``("a", "b", "c")`` (a sequence knob's value)."""
    return tuple(str(value).split(SEQ_SEP)) if value != "" else ()


def seq_encode(items) -> str:
    return SEQ_SEP.join(str(x) for x in items)
BITWISE = "bitwise"
TOLERANCE = "tolerance"


@dataclass(frozen=True)
class Knob:
    """One searchable knob.

    ``path``: dotted field path (``mailbox.pause``) or lever name (``hold.b1``).
    ``values``: its levels. For ``categorical`` / ``bool`` knobs the first
    value must be the shipped default (the gate convention of the engine:
    level 0 is "off"); ``ordinal`` values are numeric and sorted by the engine.
    ``gate``: the path of an earlier knob; this knob is read only while that
    knob is off its default (``numa_flip_arg`` under ``mailbox.numa_copies``).
    ``numerics``: ``bitwise`` (the knob keeps the mono's bits) or
    ``tolerance`` (an accuracy-gated lever: its builds pass the T0..T5 ladder
    of DESIGN rev3 12.4 instead of the bitwise gate).
    ``structural``: applying it changes the base enough that every retained
    lever is re-swept (lookback after a structural change).
    """

    path: str
    values: tuple
    kind: str = CATEGORICAL
    gate: str | None = None
    numerics: str = BITWISE
    structural: bool = False
    note: str = ""
    seq: bool = False

    def __post_init__(self):
        object.__setattr__(self, "values", tuple(self.values))
        if not self.values:
            raise ValueError(f"Knob {self.path!r}: no values")
        if len(set(map(repr, self.values))) != len(self.values):
            raise ValueError(f"Knob {self.path!r}: repeated values {self.values!r}")
        if self.kind not in (ORDINAL, CATEGORICAL, BOOL):
            raise ValueError(f"Knob {self.path!r}: kind {self.kind!r}")
        if self.kind == BOOL and len(self.values) != 2:
            raise ValueError(f"Knob {self.path!r}: a bool knob has two values")
        if self.numerics not in (BITWISE, TOLERANCE):
            raise ValueError(f"Knob {self.path!r}: numerics {self.numerics!r}")

    @classmethod
    def sequence(cls, path: str, items: Sequence[str], *, max_levels: int = 720, seed: int = 0, **kw) -> "Knob":
        """An ORDER as a knob (transforms, pass order, restructure order):
        the levels are orderings of ``items``, encoded as ``"a>b>c"`` strings
        (JSON scalars, so keys and the DB work unchanged; the owner receives a
        tuple, :func:`seq_decode`). Level 0 is the given order. Up to
        ``max_levels`` every permutation is a level; past it, the given order,
        every single-item move (insertion) and every adjacent swap of it, the
        reversal, then seeded random permutations up to the cap."""
        import itertools
        import random

        items = [str(x) for x in items]
        if len(set(items)) != len(items) or any(SEQ_SEP in x for x in items):
            raise ValueError(f"Knob.sequence {path!r}: items must be distinct and not contain {SEQ_SEP!r}")
        base = tuple(items)
        n = len(items)
        if math.factorial(n) <= max_levels:
            perms = [base] + [p for p in itertools.permutations(items) if p != base]
        else:
            perms = [base]
            seen = {base}

            def add(p):
                p = tuple(p)
                if p not in seen and len(perms) < max_levels:
                    seen.add(p)
                    perms.append(p)

            for i in range(n):
                for j in range(n):
                    if i != j:
                        q = list(base)
                        x = q.pop(i)
                        q.insert(j, x)
                        add(q)
            add(reversed(base))
            rng = random.Random(seed)
            while len(perms) < max_levels:
                q = list(items)
                rng.shuffle(q)
                add(q)
        return cls(path, tuple(SEQ_SEP.join(p) for p in perms), CATEGORICAL, seq=True, **kw)

    def param(self) -> Param:
        if self.kind == ORDINAL:
            return Param.ordinal(self.path, self.values, gate=self.gate)
        if self.kind == BOOL:
            lv = tuple(((self.path, v),) for v in self.values)
            return Param(self.path, lv, BOOL, self.gate)
        return Param.categorical(self.path, self.values, gate=self.gate)


def _json_scalar(v):
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    raise TypeError(f"knob value {v!r} is not a JSON scalar")


def flatten_instance(obj, paths: Iterable[str]) -> dict:
    """``{path: value}`` of ``obj`` at dotted ``paths`` (missing: skipped)."""
    out = {}
    for p in paths:
        cur = obj
        try:
            for part in p.split("."):
                cur = getattr(cur, part)
        except AttributeError:
            continue
        out[p] = cur
    return out


def _build_nested(base, overrides: Mapping[str, object]):
    """``base`` with dotted ``overrides`` applied bottom-up, each dataclass
    constructed once (its validation sees the final values)."""
    leaf, nested = {}, {}
    for path, value in overrides.items():
        head, _, rest = path.partition(".")
        if rest:
            nested.setdefault(head, {})[rest] = value
        else:
            leaf[head] = value
    for head, sub in nested.items():
        leaf[head] = _build_nested(getattr(base, head), sub)
    if not leaf:
        return base
    return dataclasses.replace(base, **leaf)


class KnobAdapter:
    """Base: a named, variant-specific set of :class:`Knob` over some owner.

    Subclasses implement :meth:`base_values` (the shipped value of every knob
    path) and :meth:`build` (an owner object, or raise ``ValueError`` /
    ``TypeError`` when illegal)."""

    #: legality rules beyond the owner's own validation: ``fn(built, config)
    #: -> (ok, why)``
    def __init__(
        self,
        name: str,
        knobs: Sequence[Knob],
        *,
        variant: str = "",
        arch: str = "gfx950",
        rules: Sequence[Callable[[object, Mapping[str, object]], tuple[bool, str]]] = (),
        normalize: Callable[[dict], dict] | None = None,
        excluded: Mapping[str, str] | None = None,
    ):
        names = [k.path for k in knobs]
        if len(set(names)) != len(names):
            raise ValueError(f"{name}: knob paths repeat: {names}")
        index = {k.path: i for i, k in enumerate(knobs)}
        for i, k in enumerate(knobs):
            if k.gate is not None and index.get(k.gate, len(knobs)) >= i:
                raise ValueError(f"{name}: knob {k.path!r} gate {k.gate!r} must be an earlier knob")
        self.name = name
        self.knobs = tuple(knobs)
        self.variant = variant
        self.arch = arch
        self.rules = tuple(rules)
        self._normalize = normalize
        #: knobs deliberately left out of the space, with the reason
        #: (measurement-only, test-only mutants, layout constants)
        self.excluded = dict(excluded or {})
        self._space: SearchSpace | None = None
        self._why: dict[str, str] = {}

    # ----------------------------------------------------------- owner hooks
    def base_values(self) -> dict:
        raise NotImplementedError

    def build(self, config: Mapping[str, object]):
        raise NotImplementedError

    def label(self, config: Mapping[str, object]) -> str:
        """Human-readable name of a config (the owner's suffix when it has
        one)."""
        canon = self.canonical_knobs(config)
        return ",".join(f"{k}={v}" for k, v in sorted(canon.items())) or "default"

    # -------------------------------------------------------------- identity
    @property
    def stem(self) -> str:
        return f"{self.name}.{self.variant}" if self.variant else self.name

    def knob(self, path: str) -> Knob:
        for k in self.knobs:
            if k.path == path:
                return k
        raise KeyError(path)

    def defaults_fp(self) -> str:
        base = self.base_values()
        return defaults_fingerprint(
            {"base": {p: _json_scalar(base.get(p)) for p in sorted(base)}, "abi": ABI}
        )

    def _active(self, config: Mapping[str, object]) -> dict:
        base = self.base_values()
        act: dict[str, bool] = {}
        out = {}
        for k in self.knobs:
            on = True
            if k.gate is not None:
                on = act.get(k.gate, False) and config.get(k.gate, base.get(k.gate)) != base.get(k.gate)
            act[k.path] = on
            if on and k.path in config:
                out[k.path] = config[k.path]
        return out

    def canonical_knobs(self, config: Mapping[str, object]) -> dict:
        """The knobs of ``config`` that change the build: active, off the
        base value, after :attr:`normalize`."""
        base = self.base_values()
        cfg = self._active(config)
        if self._normalize is not None:
            cfg = self._normalize(dict(cfg))
        return {k: _json_scalar(v) for k, v in cfg.items() if base.get(k, object()) != v}

    def legal(self, config: Mapping[str, object]) -> tuple[bool, str]:
        knobs = self.canonical_knobs(config)
        try:
            built = self.build(knobs)
        except (ValueError, TypeError, KeyError, AssertionError) as e:
            return False, f"{type(e).__name__}: {e}"
        for rule in self.rules:
            ok, why = rule(built, knobs)
            if not ok:
                return False, why
        return True, ""

    def key(self, config: Mapping[str, object]) -> str | None:
        """The canonical config key, or ``None`` when illegal."""
        ok, why = self.legal(config)
        knobs = self.canonical_knobs(config)
        if not ok:
            self._why[repr(sorted(knobs.items()))] = why
            return None
        return config_key(
            abi=ABI,
            arch=self.arch,
            path=self.name,
            variant_id=self.variant,
            knobs=knob_items(knobs),
            defaults=self.defaults_fp(),
        )

    def why_illegal(self, config: Mapping[str, object]) -> str:
        return self.legal(config)[1]

    def tuning_id(self, config: Mapping[str, object]) -> str | None:
        k = self.key(config)
        return None if k is None else tuning_id(self.stem, k)

    def numerics(self, config: Mapping[str, object]) -> str:
        """``tolerance`` when any changed knob is an accuracy-gated lever."""
        changed = self.canonical_knobs(config)
        for k in self.knobs:
            if k.path in changed and k.numerics == TOLERANCE:
                return TOLERANCE
        return BITWISE

    def structural(self, config: Mapping[str, object]) -> list[str]:
        changed = self.canonical_knobs(config)
        return [k.path for k in self.knobs if k.structural and k.path in changed]

    # ----------------------------------------------------------------- space
    def space(self) -> SearchSpace:
        """The engine search space (cached): keys are :meth:`key`."""
        if self._space is None:
            self._space = SearchSpace(
                [k.param() for k in self.knobs],
                canonical=self.key,
                name=self.stem,
                fingerprint_extra={"abi": ABI, "defaults": self.defaults_fp(), "arch": self.arch},
            )
        return self._space

    def config_of(self, owner_or_knobs) -> dict:
        """A config (``{path: value}``) from an owner object or a knob
        mapping; values outside a knob's levels are kept (``encode`` then
        picks the nearest level)."""
        if isinstance(owner_or_knobs, Mapping):
            return {k: v for k, v in owner_or_knobs.items() if any(x.path == k for x in self.knobs)}
        return flatten_instance(owner_or_knobs, [k.path for k in self.knobs])

    def genome_of(self, owner_or_knobs) -> tuple[int, ...]:
        cfg = dict(self.base_values())
        cfg.update(self.config_of(owner_or_knobs))
        return self.space().encode(cfg)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "variant": self.variant,
            "arch": self.arch,
            "knobs": [
                {
                    "path": k.path,
                    "values": [repr(v) for v in k.values],
                    "kind": k.kind,
                    "gate": k.gate,
                    "numerics": k.numerics,
                    "structural": k.structural,
                }
                for k in self.knobs
            ],
            "excluded": dict(self.excluded),
            "size_bound": math.prod(len(k.values) for k in self.knobs),
        }


class DataclassKnobs(KnobAdapter):
    """Knobs over a (nested) frozen dataclass ``base`` (the shipped build of
    one variant; problem fields already set)."""

    def __init__(self, name: str, base, knobs: Sequence[Knob], **kw):
        super().__init__(name, knobs, **kw)
        if not dataclasses.is_dataclass(base):
            raise TypeError(f"{name}: base must be a dataclass instance")
        self.base = base
        self._base_values = flatten_instance(base, [k.path for k in self.knobs])
        for k in self.knobs:  # a sequence field holds a tuple; the knob compares its "a>b>c" form
            if k.seq and k.path in self._base_values and not isinstance(self._base_values[k.path], str):
                self._base_values[k.path] = seq_encode(self._base_values[k.path])
        missing = [k.path for k in self.knobs if k.path not in self._base_values]
        if missing:
            raise ValueError(f"{name}: no field at {missing} of {type(base).__name__}")
        for k in self.knobs:
            if k.kind != ORDINAL and k.values[0] != self._base_values[k.path]:
                raise ValueError(
                    f"{name}: knob {k.path!r} level 0 is {k.values[0]!r}, expected the "
                    f"shipped default {self._base_values[k.path]!r}"
                )

    def base_values(self) -> dict:
        return dict(self._base_values)

    def build(self, config: Mapping[str, object]):
        cfg = dict(config)
        for k in self.knobs:
            if k.seq and k.path in cfg and isinstance(cfg[k.path], str):
                cfg[k.path] = seq_decode(cfg[k.path])  # the owner's field holds a tuple
        return _build_nested(self.base, cfg)

    def label(self, config: Mapping[str, object]) -> str:
        built = self.build(self.canonical_knobs(config))
        for attr in ("suffix", "name", "kernel_name"):
            fn = getattr(built, attr, None)
            if callable(fn):
                try:
                    s = fn()
                except Exception:  # noqa: BLE001 - a label is cosmetic
                    continue
                if isinstance(s, str):
                    return s or "default"
        return super().label(config)


class LeverStringKnobs(KnobAdapter):
    """Knobs rendered as a lever string: ATOM's ``ATOM_Q38_SCHED`` groups
    (``key=value`` joined by ``,``) or the Rust direct compiler's
    ``ROCKE_DIRECT_LEVERS`` flags (``+``-joined names of bool knobs that are
    on). ``build`` returns the string; ``validate(string)`` (optional, e.g.
    ATOM's ``sched.parse_levers``) is the owner's legality check."""

    def __init__(
        self,
        name: str,
        knobs: Sequence[Knob],
        *,
        style: str = "kv",
        base: Mapping[str, object] | None = None,
        validate: Callable[[str], object] | None = None,
        prefix: str = "",
        **kw,
    ):
        super().__init__(name, knobs, **kw)
        if style not in ("kv", "flags"):
            raise ValueError(f"{name}: style {style!r}")
        self.style = style
        self.prefix = prefix
        self._validate = validate
        self._base = {k.path: (base or {}).get(k.path, k.values[0]) for k in self.knobs}
        if style == "flags":
            for k in self.knobs:
                if k.kind != BOOL or self._base[k.path] is not False:
                    raise ValueError(f"{name}: flag knob {k.path!r} must be a bool off by default")

    def base_values(self) -> dict:
        return dict(self._base)

    def render(self, knobs: Mapping[str, object]) -> str:
        if self.style == "flags":
            return "+".join(k for k, v in sorted(knobs.items()) if v)
        body = ",".join(f"{k}={v}" for k, v in sorted(knobs.items()))
        return f"{self.prefix}{body}" if self.prefix and body else body

    def build(self, config: Mapping[str, object]) -> str:
        s = self.render(config)
        if self._validate is not None and s:
            self._validate(s)
        return s

    def label(self, config: Mapping[str, object]) -> str:
        return self.render(self.canonical_knobs(config)) or "default"


class CompositeKnobs(KnobAdapter):
    """Several adapters as one space: knob ``<part>.<path>``; one key over
    all parts. ``build`` returns ``{part: built}``."""

    def __init__(self, name: str, parts: Sequence[tuple[str, KnobAdapter]], *, rules=(), **kw):
        knobs = []
        for part, ad in parts:
            for k in ad.knobs:
                knobs.append(
                    dataclasses.replace(
                        k,
                        path=f"{part}.{k.path}",
                        gate=None if k.gate is None else f"{part}.{k.gate}",
                    )
                )
        super().__init__(name, knobs, rules=rules, **kw)
        self.parts = tuple(parts)

    def _split(self, config: Mapping[str, object]) -> dict[str, dict]:
        out: dict[str, dict] = {p: {} for p, _ in self.parts}
        for k, v in config.items():
            part, _, rest = k.partition(".")
            if part in out:
                out[part][rest] = v
        return out

    def base_values(self) -> dict:
        out = {}
        for part, ad in self.parts:
            for k, v in ad.base_values().items():
                out[f"{part}.{k}"] = v
        return out

    def canonical_knobs(self, config: Mapping[str, object]) -> dict:
        out = {}
        split = self._split(config)
        for part, ad in self.parts:
            for k, v in ad.canonical_knobs(split[part]).items():
                out[f"{part}.{k}"] = v
        return out

    def build(self, config: Mapping[str, object]) -> dict:
        split = self._split(config)
        built = {}
        for part, ad in self.parts:
            ok, why = ad.legal(split[part])
            if not ok:
                raise ValueError(f"{part}: {why}")
            built[part] = ad.build(ad.canonical_knobs(split[part]))
        return built

    def label(self, config: Mapping[str, object]) -> str:
        split = self._split(config)
        parts = [f"{p}[{ad.label(split[p])}]" for p, ad in self.parts if ad.canonical_knobs(split[p])]
        return " ".join(parts) or "default"

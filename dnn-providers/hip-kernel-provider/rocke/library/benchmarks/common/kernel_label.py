# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Lossless, human-readable names for AOT convolution kernels.

A benchmark table has to name each kernel it ran, and that name is only
useful if it pins the kernel down: two binaries that differ in an LDS layout
or an operand dtype must not print alike, or a result cannot be traced back
to the kernel that produced it. :func:`encode` therefore turns a
:class:`~benchmarks.common.kernel_cache.KernelIdentity` into a string that
carries *every* identity field, and :func:`decode` turns the string back into
an equal identity -- ``decode(encode(i)) == i`` for every identity. That
round trip is what ``reproduce_kernel.py`` builds on: a name copied out of a
log is enough to rebuild the kernel.

Format
------
Tokens joined by ``-``::

    <direction>-<algorithm>-<dtype>-<2d|3d>-<head...>-<tags...>-wave<N>-<arch>-<llvm flavor>

``head`` is positional and depends on the family:

* implicit GEMM: ``t<M>x<N>x<K>`` block tile, ``w<M>x<N>`` waves,
  ``a<M>x<N>x<K>`` warp tile, ``v<A>x<B>x<C>`` vector widths, pipeline,
  epilogue;
* direct conv: ``f<KH>x<KW>`` filter, ``p<H>x<W>`` padding, ``s<H>x<W>``
  stride, ``d<H>x<W>`` dilation, ``c<cpg>k<kpg>`` channels per group (0 =
  runtime). The direction drops its ``direct_`` prefix, which the algorithm
  already says.

``tags`` are order-free and present only where a field leaves its baseline
(:data:`_BASELINE`): ``grp``, ``unroll``, ``async``, ``sk2``, ``2stage``,
``gm4`` and so on (see :data:`_TAGS`), direct-conv tuning knobs as
``key=value``, and -- for any field no tag covers, including fields added to
the identity later -- ``@field=value``. Free-form strings are
percent-escaped, so no value can introduce a separator.

The baseline is frozen on purpose: it gives omitted tags their meaning, so
changing one of its values would silently change what existing names decode
to. A field added to the identity needs no change here to stay lossless (it
is spelled ``@field=value`` until it gets a tag); give it a baseline entry
only to keep it out of names while it sits at that value.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote, unquote

__all__ = ["decode", "encode", "strip_launch_suffix"]

_SEP = "-"

# What every field reads as when its tag is absent. Never change a value here
# (see the module docstring).
_BASELINE: Dict[str, Any] = dict(
    async_dma=False,
    unroll_k=False,
    chiplet_swizzle=False,
    lds_layout="default",
    lds_k_pad=0,
    lds_k_outer=False,
    waves_per_eu=None,
    acc_epilogue="none",
    split_k=1,
    two_stage=False,
    ws_replicas=0,
    group_merge=1,
    num_load_waves=0,
    cshuffle_no_alias=False,
    is_3d=False,
    is_pointwise=False,
    filter_h=0,
    filter_w=0,
    filter_d=0,
    stride_h=0,
    stride_w=0,
    dilation_h=0,
    dilation_w=0,
    pad_h=0,
    pad_w=0,
    cpg=0,
    kpg=0,
    grouped=False,
    block_h=0,
    max_sub_gemms=0,
    async_chunk_a=0,
    async_chunk_b=0,
    knobs="",
)

# Direct conv is not a GEMM: its identities zero the GEMM fields, so those are
# its baseline rather than positional head tokens.
_DIRECT_BASELINE: Dict[str, Any] = dict(
    _BASELINE,
    tile_m=0,
    tile_n=0,
    tile_k=0,
    warp_m=0,
    warp_n=0,
    warp_tile_m=0,
    warp_tile_n=0,
    warp_tile_k=0,
    pipeline="direct",
    epilogue="direct",
    vector_size_a=0,
    vector_size_b=0,
    vector_size_c=0,
    knobs="{}",
)

_DIRECT_DIRECTION_PREFIX = "direct_"
# Marks a direct-conv direction spelled verbatim (one without the prefix).
_VERBATIM = "~"


# ---------------------------------------------------------------------------
# Escaping and scalar values
# ---------------------------------------------------------------------------

# Characters kept as-is in free-form strings; everything else, notably the
# ``-`` separator, ``=``, ``@``, ``.`` and ``~``, is percent-escaped.
_SAFE = "_+"


def _esc(s: str) -> str:
    # quote() never escapes "-", "." or "~" (they are always "safe" to it).
    return (
        quote(s, safe=_SAFE).replace("-", "%2D").replace(".", "%2E").replace("~", "%7E")
    )


def _unesc(s: str) -> str:
    return unquote(s)


_INT_RE = re.compile(r"-?\d+")
_KEYWORDS = {"true": True, "false": False, "null": None}
_STR_MARK = "'"


def _enc_value(v: Any) -> str:
    """One scalar: bool, None, int or str, typed so it decodes to itself."""
    if v is True or v is False or v is None:
        return {True: "true", False: "false", None: "null"}[v]
    if type(v) is int:
        return _esc(str(v))
    if type(v) is str:
        # A string that would read back as another type gets a marker.
        if v in _KEYWORDS or _INT_RE.fullmatch(v) or v.startswith(_STR_MARK):
            return _STR_MARK + _esc(v)
        return _esc(v)
    raise TypeError(f"cannot name a {type(v).__name__} value ({v!r})")


def _dec_value(s: str) -> Any:
    if s.startswith(_STR_MARK):
        return _unesc(s[len(_STR_MARK) :])
    if s in _KEYWORDS:
        return _KEYWORDS[s]
    raw = _unesc(s)
    if _INT_RE.fullmatch(raw):
        return int(raw)
    return raw


def _is_count(v: Any) -> bool:
    return type(v) is int and v >= 0


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Tag:
    """A token for one or more fields.

    A flag tag (no capture groups) stands for its fields being ``True``; a
    numeric tag spells non-negative ints into its pattern's groups. A value a
    tag cannot spell falls back to ``@field=value``.
    """

    fields: Tuple[str, ...]
    pattern: str
    fmt: Optional[str] = None  # numeric tags: str.format template

    @property
    def regex(self) -> "re.Pattern[str]":
        return re.compile(self.pattern)

    @property
    def is_flag(self) -> bool:
        return self.fmt is None

    def fits(self, values: Sequence[Any]) -> bool:
        if self.is_flag:
            return all(v is True for v in values)
        return all(_is_count(v) for v in values)

    def encode(self, values: Sequence[Any]) -> str:
        return self.pattern if self.is_flag else self.fmt.format(*values)

    def decode(self, m: "re.Match[str]") -> Dict[str, Any]:
        if self.is_flag:
            return {f: True for f in self.fields}
        return {f: int(g) for f, g in zip(self.fields, m.groups())}


def _num(fields: Sequence[str], prefix: str, sep: Sequence[str] = ()) -> _Tag:
    """``prefix<a>[sep0<b>...]`` over ints, e.g. ``s1x1`` or ``c4k4``."""
    seps = list(sep) or ["x"] * (len(fields) - 1)
    pattern = re.escape(prefix) + r"(\d+)"
    fmt = prefix + "{}"
    for s in seps:
        pattern += re.escape(s) + r"(\d+)"
        fmt += s + "{}"
    return _Tag(tuple(fields), pattern, fmt)


_FILTER = _num(("filter_h", "filter_w"), "f")
_PAD = _num(("pad_h", "pad_w"), "p")
_STRIDE = _num(("stride_h", "stride_w"), "s")
_DILATION = _num(("dilation_h", "dilation_w"), "d")
_CHANNELS = _num(("cpg", "kpg"), "c", sep=("k",))

# The direct-conv head: always spelled, even at 0 -- a direct binary bakes
# every one of these.
_DIRECT_HEAD = (_FILTER, _PAD, _STRIDE, _DILATION, _CHANNELS)

# Optional tags, in the order they are written. The patterns must not
# overlap; test_kernel_label checks that every tag only matches itself.
_TAGS: Tuple[_Tag, ...] = (
    _Tag(("grouped",), "grp"),
    _Tag(("is_pointwise",), "pw"),
    _FILTER,
    _num(("filter_d",), "fd"),
    _PAD,
    _STRIDE,
    _DILATION,
    _CHANNELS,
    _Tag(("unroll_k",), "unroll"),
    _Tag(("async_dma",), "async"),
    _num(("async_chunk_a", "async_chunk_b"), "ac"),
    _Tag(("chiplet_swizzle",), "chip"),
    _num(("lds_k_pad",), "kpad"),
    _Tag(("lds_k_outer",), "kouter"),
    _num(("waves_per_eu",), "wpe"),
    _num(("num_load_waves",), "nlw"),
    _Tag(("cshuffle_no_alias",), "noalias"),
    _num(("split_k",), "sk"),
    _Tag(("two_stage",), "2stage"),
    _num(("ws_replicas",), "wsr"),
    _num(("group_merge",), "gm"),
    _num(("max_sub_gemms",), "msg"),
    _num(("block_h",), "bh"),
)

_WAVE = _num(("wave_size",), "wave")
_GENERIC = "@"
_KNOB_RE = re.compile(r"([a-z_][a-z0-9_]*)=(.*)")
_GENERIC_RE = re.compile(re.escape(_GENERIC) + r"([a-z_][a-z0-9_]*)=(.*)")

_IMPLICIT_HEAD: Tuple[_Tag, ...] = (
    _num(("tile_m", "tile_n", "tile_k"), "t"),
    _num(("warp_m", "warp_n"), "w"),
    _num(("warp_tile_m", "warp_tile_n", "warp_tile_k"), "a"),
    _num(("vector_size_a", "vector_size_b", "vector_size_c"), "v"),
)


def _is_direct(algorithm: Any) -> bool:
    # Mirrors KernelIdentity.is_direct.
    return isinstance(algorithm, str) and algorithm.startswith("direct")


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


def _knob_tokens(knobs: str, baseline: str) -> Optional[List[str]]:
    """``key=value`` tokens for a knob JSON string, or ``None`` if they could
    not reproduce it byte for byte (it is then spelled ``@knobs=...``)."""
    if knobs == baseline:
        return []
    try:
        parsed = json.loads(knobs)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict) or json.dumps(parsed, sort_keys=True) != knobs:
        return None
    # A non-baseline empty dict would encode to no tokens, i.e. to the
    # baseline's name.
    if not parsed:
        return None
    out = []
    for key in sorted(parsed):
        if not _KNOB_RE.fullmatch(f"{key}=") or key.startswith(_GENERIC):
            return None
        try:
            out.append(f"{key}={_enc_value(parsed[key])}")
        except TypeError:
            return None
    return out


def encode(identity) -> str:
    """The name of ``identity``; :func:`decode` inverts it exactly."""
    values = dataclasses.asdict(identity)
    direct = _is_direct(identity.algorithm)
    baseline = _DIRECT_BASELINE if direct else _BASELINE
    covered = set()

    def _text(f: str, value: Optional[str] = None) -> str:
        """A free-form string field (``value``: an already-transformed
        spelling of it); anything but a string is an empty placeholder that
        the ``@field`` token written for it overrides."""
        if type(values[f]) is not str:
            return ""
        covered.add(f)
        return _esc(values[f] if value is None else value)

    def _spell(tag: _Tag) -> str:
        """``tag``'s token; a value it cannot spell becomes a zero placeholder
        that the ``@field`` token written for it overrides."""
        vals = [values[f] for f in tag.fields]
        if not tag.fits(vals):
            return tag.encode([0] * len(tag.fields))
        covered.update(tag.fields)
        return tag.encode(vals)

    def _at_baseline(fields: Sequence[str]) -> bool:
        return all(
            f in baseline
            and values[f] == baseline[f]
            and type(values[f]) is type(baseline[f])
            for f in fields
        )

    direction = values["direction"]
    if not direct:
        direction = _text("direction")
    elif isinstance(direction, str) and direction.startswith(_DIRECT_DIRECTION_PREFIX):
        direction = _text("direction", direction[len(_DIRECT_DIRECTION_PREFIX) :])
    else:
        direction = _VERBATIM + _text("direction")
    dtypes = ("dtype_a", "dtype_b", "dtype_d")
    if len({repr(values[f]) for f in dtypes}) == 1 and type(values["dtype_a"]) is str:
        dtype = _text("dtype_a")
        covered.update(dtypes)
    else:
        dtype = ".".join(_text(f) for f in dtypes)
    tokens = [
        direction,
        _text("algorithm"),
        dtype,
        "3d" if identity.is_3d is True else "2d",
    ]
    if type(identity.is_3d) is bool:
        covered.add("is_3d")

    # Positional head.
    if direct:
        tokens += [_spell(tag) for tag in _DIRECT_HEAD]
    else:
        tokens += [_spell(tag) for tag in _IMPLICIT_HEAD]
        tokens += [_text("pipeline"), _text("epilogue")]

    # Optional tags, only where a field leaves its baseline.
    for tag in _TAGS:
        if direct and tag in _DIRECT_HEAD:
            continue
        if _at_baseline(tag.fields):
            covered.update(tag.fields)
        elif tag.fits([values[f] for f in tag.fields]):
            tokens.append(_spell(tag))

    knob_tokens = _knob_tokens(identity.knobs, baseline["knobs"])
    if knob_tokens is not None:
        tokens += knob_tokens
        covered.add("knobs")

    tail = [_spell(_WAVE), _text("arch"), _text("llvm_flavor")]
    # Everything no token above carried, including fields this module has
    # never heard of.
    for f in sorted(values):
        if f not in covered and not _at_baseline((f,)):
            tokens.append(f"{_GENERIC}{f}={_enc_value(values[f])}")

    return _SEP.join(tokens + tail)


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


def strip_launch_suffix(name: str) -> str:
    """The kernel name in a benchmark table cell.

    A cell may append launch-time parameters after whitespace (``... @split_k=4``
    for wgrad: the degree is a kernarg, not part of the binary).
    """
    return name.strip().split()[0] if name.strip() else ""


def _match_head(tag: _Tag, token: str, name: str) -> Dict[str, Any]:
    m = tag.regex.fullmatch(token)
    if m is None:
        raise ValueError(f"{name!r}: expected {tag.fmt!r}-shaped token, got {token!r}")
    return tag.decode(m)


def decode(name: str, cls: Optional[Callable[..., Any]] = None):
    """The identity named ``name`` (as produced by :func:`encode`).

    ``cls`` defaults to :class:`~benchmarks.common.kernel_cache.KernelIdentity`.
    Raises ``ValueError`` on a malformed name.
    """
    if cls is None:
        from benchmarks.common.kernel_cache import KernelIdentity as cls

    name = strip_launch_suffix(name)
    tokens = name.split(_SEP)
    if len(tokens) < 7:
        raise ValueError(f"{name!r} is not a kernel name: too few tokens")
    direction_tok, algorithm_tok, dtype_tok, dim_tok = tokens[:4]
    algorithm = _unesc(algorithm_tok)
    direct = _is_direct(algorithm)
    out: Dict[str, Any] = dict(_DIRECT_BASELINE if direct else _BASELINE)
    out["algorithm"] = algorithm

    if direct and not direction_tok.startswith(_VERBATIM):
        out["direction"] = _DIRECT_DIRECTION_PREFIX + _unesc(direction_tok)
    elif direct:
        out["direction"] = _unesc(direction_tok[len(_VERBATIM) :])
    else:
        out["direction"] = _unesc(direction_tok)

    parts = dtype_tok.split(".")
    if len(parts) == 1:
        out["dtype_a"] = out["dtype_b"] = out["dtype_d"] = _unesc(parts[0])
    elif len(parts) == 3:
        out["dtype_a"], out["dtype_b"], out["dtype_d"] = (_unesc(p) for p in parts)
    else:
        raise ValueError(f"{name!r}: bad dtype token {dtype_tok!r}")
    if dim_tok not in ("2d", "3d"):
        raise ValueError(f"{name!r}: expected 2d or 3d, got {dim_tok!r}")
    out["is_3d"] = dim_tok == "3d"

    rest = tokens[4:]
    head_len = len(_DIRECT_HEAD) if direct else len(_IMPLICIT_HEAD) + 2
    if len(rest) < head_len + 3:
        raise ValueError(f"{name!r} is not a kernel name: too few tokens")
    head, free, tail = rest[:head_len], rest[head_len:-3], rest[-3:]
    if direct:
        for tag, tok in zip(_DIRECT_HEAD, head):
            out.update(_match_head(tag, tok, name))
    else:
        for tag, tok in zip(_IMPLICIT_HEAD, head):
            out.update(_match_head(tag, tok, name))
        out["pipeline"], out["epilogue"] = (_unesc(t) for t in head[-2:])

    wave_tok, arch_tok, flavor_tok = tail
    out.update(_match_head(_WAVE, wave_tok, name))
    out["arch"] = _unesc(arch_tok)
    out["llvm_flavor"] = _unesc(flavor_tok)

    seen = set()
    knobs: Dict[str, Any] = {}
    generic: Dict[str, Any] = {}
    for tok in free:
        if tok in seen:
            raise ValueError(f"{name!r}: token {tok!r} repeated")
        seen.add(tok)
        m = _GENERIC_RE.fullmatch(tok)
        if m:
            generic[m.group(1)] = _dec_value(m.group(2))
            continue
        m = _KNOB_RE.fullmatch(tok)
        if m:
            knobs[m.group(1)] = _dec_value(m.group(2))
            continue
        for tag in _TAGS:
            m = tag.regex.fullmatch(tok)
            if m:
                out.update(tag.decode(m))
                break
        else:
            raise ValueError(f"{name!r}: unknown token {tok!r}")
    if knobs:
        out["knobs"] = json.dumps(knobs, sort_keys=True)
    # @field overrides win: they carry what no other token could.
    out.update(generic)

    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(out) - known)
    if unknown:
        raise ValueError(f"{name!r}: names fields this build does not know: {unknown}")
    return cls(**out)

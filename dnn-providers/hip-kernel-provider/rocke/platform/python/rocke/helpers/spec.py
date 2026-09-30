# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Spec validation, signature building, and kernel-name helpers.

Every kernel instance file defines:

* a frozen :class:`dataclass` ``Spec`` (with one ``kernel_name(self)``
  method that concatenates fields into a unique kernel identifier),
* an ``is_valid_spec(spec)`` predicate that rejects unsupported dtype /
  ``block_size`` / ``vec`` / divisibility combinations,
* a ``<op>_signature(spec)`` function that builds the manifest-style
  list of ``{"name": ..., "type": ...}`` dicts used by
  :class:`rocke.runtime.launcher.KernelLauncher`, and
* a ``<op>_grid(...)`` helper that computes the launch grid (usually
  ``(ceil_div(N, tile_n), ceil_div(M, tile_m), batch)``).

These four chunks are formulaic and duplicated 7+ times. This module
provides the shared primitives so a new instance is ~10 lines of
glue instead of ~40.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


__all__ = [
    "IOSpecRule",
    "SignatureBuilder",
    "WarpTileBlockSizeMixin",
    "ceil_div_grid",
    "choose_load_vec",
    "derive_block_size",
    "kernel_name_join",
    "ptr_type_str",
    "sig_param",
    "sig_scalar",
    "validate_io",
]


# ---------------------------------------------------------------------
# IOSpecRule / validate_io
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class IOSpecRule:
    """The predicate inputs every f16/bf16 small-op spec validates.

    Pass this to :func:`validate_io`; the helper returns the same
    ``(ok, reason)`` tuple every ``is_valid_spec`` in the codebase
    already returns, so adopting it is a one-line replacement.
    """

    dtype: str
    block_size: int
    vec: int
    # When set, enforce ``n_per_block % (block_size * vec) == 0``.
    n_per_block: Optional[int] = None
    # When set, cap ``n_per_block / block_size`` (= elems-per-thread).
    max_elems_per_thread: Optional[int] = None

    allowed_dtypes: Tuple[str, ...] = ("f16", "fp16", "bf16")
    allowed_block_sizes: Tuple[int, ...] = (64, 128, 256, 512, 1024)
    allowed_vecs: Tuple[int, ...] = (2, 4, 8)


def validate_io(rule: IOSpecRule) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for one :class:`IOSpecRule`.

    Tries every predicate in order; returns at the first failure so
    callers get the most actionable single-line reason.
    """
    if rule.dtype not in rule.allowed_dtypes:
        return False, f"unsupported dtype {rule.dtype!r}"
    if rule.block_size not in rule.allowed_block_sizes:
        return False, (
            f"block_size {rule.block_size} not in {set(rule.allowed_block_sizes)}"
        )
    if rule.vec not in rule.allowed_vecs:
        return False, f"vec {rule.vec} not in {set(rule.allowed_vecs)}"
    if rule.n_per_block is not None:
        chunk = rule.block_size * rule.vec
        if rule.n_per_block % chunk:
            return False, (
                f"n_per_block ({rule.n_per_block}) must be divisible by "
                f"block_size*vec ({chunk})"
            )
        if rule.max_elems_per_thread is not None:
            elems = rule.n_per_block // rule.block_size
            if elems > rule.max_elems_per_thread:
                return False, (
                    f"elems_per_thread {elems} > {rule.max_elems_per_thread}; "
                    "pick a larger block_size or a multi-pass kernel"
                )
    return True, "ok"


# ---------------------------------------------------------------------
# choose_load_vec
# ---------------------------------------------------------------------


def choose_load_vec(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    block_size: int,
    elem_bytes: int = 2,
) -> int:
    """Pick the widest global-load vector width for a GEMM block tile.

    Returns the largest ``v`` such that ``v`` divides the K tile, the
    per-thread A/B load distribution is coalesced over ``block_size``
    threads, and ``v * elem_bytes <= 16`` (hardware buffer_load limit
    of 4 dwords = 16 bytes).

    ``elem_bytes`` defaults to 2 (fp16/bf16); pass 4 for fp32/i32 to
    cap the vector at 4 elements (= 4 dwords).

    This is the single source of truth for the compile-time picker that
    ``gemm_universal`` / ``conv_implicit_gemm`` / ``moe_gemm_fused`` each
    re-implemented; the returned ``int`` is baked into a ``const_i32`` so an
    identical result means byte-identical IR.
    """
    # 16 bytes == 4 dwords, the hardware maximum for buffer_load_vN.
    max_vec = 16 // elem_bytes
    threads = block_size
    v = max_vec
    while v >= 1:
        if tile_k % v:
            v //= 2
            continue
        a_vecs = (tile_m * tile_k) // v
        b_vecs = (tile_n * tile_k) // v
        if a_vecs < threads or b_vecs < threads or a_vecs % threads or b_vecs % threads:
            v //= 2
            continue
        return v
    raise ValueError(
        f"no usable load_vec for tile_m={tile_m} tile_n={tile_n} "
        f"tile_k={tile_k} block_size={block_size} elem_bytes={elem_bytes}"
    )


# ---------------------------------------------------------------------
# kernel_name_join
# ---------------------------------------------------------------------


def kernel_name_join(
    prefix: str, *parts: str, flags: Optional[dict[str, bool]] = None
) -> str:
    """Assemble a deterministic kernel name from a prefix, ordered
    parts, and a flag map.

    ``parts`` are joined by ``_``; empty strings are dropped.
    ``flags`` maps an attribute name (e.g. ``"smv"`` for save-mean-var)
    to a bool; entries that are True become an ``_<name>`` suffix in
    iteration order.

    Example::

        kernel_name_join("rocke_layernorm2d_fwd",
                         "f16", "N4096", "b256", "v8",
                         flags={"smv": True})
        # -> "rocke_layernorm2d_fwd_f16_N4096_b256_v8_smv"
    """
    body = "_".join(p for p in (prefix, *parts) if p)
    if flags:
        for name, on in flags.items():
            if on:
                body += f"_{name}"
    return body.replace("/", "_")


# ---------------------------------------------------------------------
# Signature builder
# ---------------------------------------------------------------------


def ptr_type_str(dtype: str, addr_space: str = "global") -> str:
    """The manifest-side dtype-string for a pointer arg.

    Accepts the same dtype aliases :mod:`rocke.helpers.io` does
    (``f16`` / ``fp16`` / ``bf16``); the manifest layer canonicalises
    to ``f16`` (no ``fp16`` alias on the wire) so the runtime arg
    packer stays single-source.
    """
    canon = "f16" if dtype in ("f16", "fp16") else dtype
    return f"ptr<{canon}, {addr_space}>"


def sig_param(name: str, dtype: str, addr_space: str = "global") -> dict:
    """One pointer-kind signature entry."""
    return {"name": name, "type": ptr_type_str(dtype, addr_space)}


def sig_scalar(name: str, ty: str) -> dict:
    """One scalar signature entry (``i32`` / ``i64`` / ``f32``)."""
    if ty not in ("i32", "i64", "f32"):
        raise ValueError(f"unsupported scalar arg type {ty!r}")
    return {"name": name, "type": ty}


@dataclass
class SignatureBuilder:
    """Fluent constructor for the manifest signature list.

    Replaces the 8-12 line ``f"ptr<{dtype}, global>"`` boilerplate that
    every ``<op>_signature(spec)`` was hand-rolling. Calls chain::

        SignatureBuilder()
            .ptr("X", spec.dtype)
            .ptr("Gamma", spec.dtype)
            .ptr("Beta", spec.dtype)
            .ptr("Y", spec.dtype)
            .scalar("M", "i32")
            .scalar("N", "i32")
            .scalar("eps", "f32")
            .build()
    """

    _items: List[dict] = field(default_factory=list)

    def ptr(
        self, name: str, dtype: str, addr_space: str = "global"
    ) -> "SignatureBuilder":
        self._items.append(sig_param(name, dtype, addr_space))
        return self

    def scalar(self, name: str, ty: str) -> "SignatureBuilder":
        self._items.append(sig_scalar(name, ty))
        return self

    def extend(self, items: Sequence[dict]) -> "SignatureBuilder":
        """Append a pre-built list (e.g. from another helper)."""
        self._items.extend(items)
        return self

    def build(self) -> List[dict]:
        return list(self._items)


# ---------------------------------------------------------------------
# ceil_div_grid
# ---------------------------------------------------------------------


def ceil_div_grid(*dims: Tuple[int, int]) -> Tuple[int, int, int]:
    """Compute a 3D grid from ``(total, tile)`` pairs.

    ``ceil_div_grid((N, tile_n), (M, tile_m))`` returns
    ``(ceil_div(N, tile_n), ceil_div(M, tile_m), 1)``. Pass three
    pairs to populate the ``z`` axis (typically a batch count, where
    ``tile=1`` so the second element is just the count itself).

    Examples::

        ceil_div_grid((N, tile_n), (M, tile_m))            # GEMM
        ceil_div_grid((N, tile_n), (M, tile_m), (B, 1))    # batched GEMM
        ceil_div_grid((M, 1))                               # one CTA per row
    """
    if not (1 <= len(dims) <= 3):
        raise ValueError(f"ceil_div_grid takes 1-3 pairs (got {len(dims)})")
    out: List[int] = []
    for total, tile in dims:
        if int(tile) <= 0:
            raise ValueError(f"tile must be positive, got {tile}")
        out.append((int(total) + int(tile) - 1) // int(tile))
    while len(out) < 3:
        out.append(1)
    return (out[0], out[1], out[2])


# ---------------------------------------------------------------------
# derive_block_size / WarpTileBlockSizeMixin
# ---------------------------------------------------------------------


def derive_block_size(tile: object, wave_size: int) -> int:
    """Threads-per-block for a warp-tiled GEMM-family spec.

    The canonical rule (CK's ``gemm_validation_utils.py``: ``BlockSize =
    NumWarps * warp_size``) is::

        block_size = warp_m * warp_n * warp_k * wave_size

    where ``warp_m`` / ``warp_n`` / ``warp_k`` are the warp-grid counts on the
    spec's :class:`TileSpec`. This is the single source of truth shared by
    :class:`~rocke.instances.common.gemm_universal.UniversalGemmSpec` and the
    five wrapper specs (batched / grouped / flatmm / multi-d / multi-abd) that
    previously each re-implemented it in ``__post_init__``.
    """
    return int(tile.warp_m) * int(tile.warp_n) * int(tile.warp_k) * int(wave_size)


class WarpTileBlockSizeMixin:
    """Mixin for frozen GEMM-family specs that lazily derive ``block_size``.

    A spec opting in must expose a ``tile`` (with ``warp_m`` / ``warp_n`` /
    ``warp_k`` ints), a ``wave_size`` int, and a ``block_size`` int field that
    defaults to ``0`` ("derive me"). On a frozen dataclass the assignment goes
    through ``object.__setattr__`` (the only legal write on a frozen instance),
    exactly matching the hand-rolled ``__post_init__`` bodies it replaces — so
    adopting it keeps the produced ``block_size`` (and therefore the kernel
    name and the emitted IR) byte-identical.

    Usage::

        @dataclass(frozen=True)
        class BatchedGemmSpec(WarpTileBlockSizeMixin):
            tile: TileSpec
            wave_size: int = 64
            block_size: int = 0

            def __post_init__(self) -> None:
                self._init_block_size()
    """

    def _init_block_size(self) -> None:
        if getattr(self, "block_size", 0) == 0:
            object.__setattr__(
                self,
                "block_size",
                derive_block_size(self.tile, self.wave_size),
            )

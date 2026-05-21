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
  :class:`ck_dsl.runtime.launcher.KernelLauncher`, and
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
    "ceil_div_grid",
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

        kernel_name_join("ck_dsl_layernorm2d_fwd",
                         "f16", "N4096", "b256", "v8",
                         flags={"smv": True})
        # -> "ck_dsl_layernorm2d_fwd_f16_N4096_b256_v8_smv"
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

    Accepts the same dtype aliases :mod:`ck_dsl.helpers.io` does
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

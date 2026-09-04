# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# abi.py -- the WIRE compatibility contract for CBOR recipes and bundles: which
# engines can read an artifact this generator just produced.
#
# The C mirror of everything here is cpp/include/rocke/abi.h, which also
# documents the second, separate contract (the BINARY ABI of the shared library,
# `ROCKE_ABI_VERSION`) and why the two are not one number.
#
# The shape of the check
# ----------------------
# A bundle is a persisted artifact. It can be read by an engine older than the
# generator that wrote it (a deployed hipDNN that has not been rebuilt) or newer
# (a bundle from the last release). So compatibility is a property of the
# artifact, decided per artifact, not a property of the process.
#
# Each artifact declares the OLDEST reader that can read it CORRECTLY:
#
#     "abi": {"min_reader": 1, "writer": 1, "engine": "...", "build_id": "..."}
#
# and a reader refuses exactly when `min_reader` exceeds its own level. The
# tempting alternative -- one monotonic format version, compared for equality or
# for "artifact <= reader" -- is worse in a way that only shows up in the field:
# it rejects newer artifacts wholesale, whether or not they use anything new, so
# a generator upgrade becomes a flag day for every deployed engine even though
# almost every recipe it emits is byte-for-byte the kind of thing the old engine
# has always read. Declaring the requirement instead of the origin means an old
# engine refuses exactly the artifacts it would get wrong.
#
# `writer`, `engine` and `build_id` are provenance, for working out where a bad
# artifact came from. Nothing compares them. Only `min_reader` decides.
#
# Deriving rather than declaring
# ------------------------------
# `min_reader` is COMPUTED from what the recipe uses (`recipe_min_reader`), not
# passed in by a caller. A hand-set requirement is a second copy of the truth,
# and it drifts the first time someone adds a construct and forgets -- which is
# the same failure this package spent a lot of effort designing out of guards
# and of the intexpr evaluators. Deriving it means the stamp cannot disagree
# with the content.
#
# What the derivation can and cannot see is worth being precise about, because
# it decides how much the version is actually worth:
#
#   covered      Constructs the VMs DISPATCH on -- instruction ops and intexpr
#                node kinds. Both engines already fail loudly on an unknown one,
#                so these are self-policing even without a version; the stamp
#                just turns a confusing "unknown instr op 'foo'" into "you need a
#                newer engine".
#   NOT covered  Attribute VALUES. The VM passes attrs through to the IR builder
#                uninterpreted, so their meaning belongs to the lowerer, not to
#                the recipe format. A lowerer that silently ignores an attribute
#                it does not understand is a real hazard and this version number
#                cannot catch it.
#
# So the genuinely dangerous change -- one an old engine ACCEPTS and gets wrong
# -- is the one a human has to notice and bump for. `stamp` refuses to stamp a
# recipe containing an instruction op it does not recognize, which is the one
# piece of that a machine can enforce.

from __future__ import annotations

from typing import Any, Dict, List, Optional

#: Wire ABI level of this generator. MUST equal ROCKE_RECIPE_ABI in
#: cpp/include/rocke/abi.h; a test pins the two together against the built
#: engine, because a Python generator that believed it was newer than the C
#: engine would stamp artifacts that engine then refuses.
RECIPE_ABI = 1

#: Binary ABI the ctypes bindings in online.py are written against. MUST equal
#: ROCKE_ABI_VERSION in cpp/include/rocke/abi.h. Unlike the wire level, this one
#: is checked at library load: ctypes argtypes are a hand-written mirror of the C
#: signatures, and calling a mismatched library with them is memory-unsafe rather
#: than merely wrong.
BINARY_ABI = 1


class AbiError(RuntimeError):
    """An artifact this reader cannot read, or a generator that cannot describe
    what it just produced."""


# Instruction ops, and the ABI level that introduced each. Everything the two
# engines ship today is level 1. Add an entry in the same change that teaches
# the VMs a new op; `stamp` fails on an unregistered one, so the list cannot
# quietly fall behind recipe_expand.py and recipe_vm.cpp.
INSTR_OPS: Dict[str, int] = {
    "param": 1,
    "const_i32": 1,
    "const_f32": 1,
    "thread_id_x": 1,
    "emit": 1,
    "alias": 1,
    "static_for": 1,
    "static_if": 1,
    "scf_for": 1,
    "scf_if": 1,
    "ret": 1,
}

# intexpr node kinds. Unknown ones already raise in both evaluators.
INTEXPR_NODES: Dict[str, int] = {
    "spec": 1,
    "var": 1,
    "spec_str_eq": 1,
    "magic_multiplier": 1,
    "magic_shift": 1,
    "add": 1,
    "sub": 1,
    "mul": 1,
    "div": 1,
    "mod": 1,
    "eq": 1,
    "ne": 1,
    "lt": 1,
    "le": 1,
    "gt": 1,
    "ge": 1,
}

# Nested instruction lists, by the key that holds them.
_BODY_KEYS = ("body", "then", "else")


def _walk_instrs(program: Any):
    """Every instruction in a program, including nested compile-time bodies."""
    if not isinstance(program, list):
        return
    for instr in program:
        if not isinstance(instr, dict):
            continue
        yield instr
        for key in _BODY_KEYS:
            yield from _walk_instrs(instr.get(key))


def _walk_dicts(node: Any):
    for value in node.values() if isinstance(node, dict) else []:
        yield from _walk_dicts(value)
    for item in node if isinstance(node, list) else []:
        yield from _walk_dicts(item)
    if isinstance(node, dict):
        yield node


def recipe_min_reader(recipe: Dict[str, Any], *, strict: bool = True) -> int:
    """The oldest reader ABI level that can read `recipe` correctly.

    Level 1 for anything using only the constructs both engines have always had,
    which is the overwhelmingly common answer and the reason this is worth
    computing rather than assuming the current level.

    `strict` (the default) raises on an instruction op that is not registered,
    on the theory that a generator which cannot describe what it just emitted
    should not be stamping compatibility claims about it. Pass False to inspect
    a foreign or future recipe without tripping over its unknowns."""
    level = 1
    for instr in _walk_instrs(recipe.get("program")):
        op = instr.get("op")
        if op in INSTR_OPS:
            level = max(level, INSTR_OPS[op])
        elif strict:
            raise AbiError(
                f"instruction op {op!r} is not in abi.INSTR_OPS, so this "
                f"generator cannot say which engines can read a recipe using "
                f"it. Register it (and bump RECIPE_ABI if an older engine would "
                f"MISREAD rather than reject it)."
            )
    # Every subtree that can hold an intexpr, not just the program. `guard`
    # rules and `launch` geometry are expressions over the same spec axes and
    # are evaluated by the same evaluator, so a construct appearing in one of
    # them constrains a reader exactly as much as one in the program.
    for section in ("program", "launch", "guard"):
        for node in _walk_dicts(recipe.get(section)):
            for key in node:
                if key in INTEXPR_NODES:
                    level = max(level, INTEXPR_NODES[key])
    return level


def stamp(
    artifact: Dict[str, Any],
    *,
    min_reader: Optional[int] = None,
    engine: str = "",
    build_id: str = "",
) -> Dict[str, Any]:
    """Return `artifact` with an `abi` block describing what it needs.

    `min_reader` is derived from the content unless given; passing it explicitly
    is for tests that need to describe an artifact from the future."""
    need = recipe_min_reader(artifact) if min_reader is None else min_reader
    if need > RECIPE_ABI:
        raise AbiError(
            f"artifact needs reader >= {need} but this generator is {RECIPE_ABI}"
        )
    out = dict(artifact)
    out["abi"] = {
        "min_reader": need,
        "writer": RECIPE_ABI,
        "engine": engine,
        "build_id": build_id,
    }
    return out


def check(artifact: Dict[str, Any], *, level: int = RECIPE_ABI) -> None:
    """Raise AbiError if this reader cannot read `artifact`.

    A missing `abi` block means level 1 -- every recipe recorded before the
    block existed stays readable. Absence has to mean the floor rather than
    "unknown": refusing those would strand existing bundles for no safety gain,
    since a level-1 artifact is exactly what a level-1 reader was written for.

    Mirrors rv_abi_ok in recipe_vm.cpp."""
    abi = artifact.get("abi")
    if not isinstance(abi, dict):
        return
    need = abi.get("min_reader")
    if not isinstance(need, int) or need <= level:
        return
    raise AbiError(
        f"artifact needs a recipe reader >= {need}, this reader is {level} "
        f"(written by engine {abi.get('engine') or '?'})"
    )


def describe(artifact: Dict[str, Any]) -> str:
    """One line of provenance, for a generation report or a bug report."""
    abi = artifact.get("abi") or {}
    return (
        f"min_reader={abi.get('min_reader', 1)} writer={abi.get('writer', '?')} "
        f"engine={abi.get('engine') or '?'} build_id={(abi.get('build_id') or '?')[:12]}"
    )


def registered_ops() -> List[str]:
    return sorted(INSTR_OPS)

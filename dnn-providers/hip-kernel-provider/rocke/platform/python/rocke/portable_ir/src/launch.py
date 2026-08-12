# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# launch.py -- attach launch geometry to a recipe, and read back everything
# needed to launch the kernel it builds.
#
# The C mirror is cpp/include/rocke/recipe_launch.h (implemented in
# recipe_vm.cpp), which is what a no-CPython client such as hipDNN actually
# calls. This module is the authoring side and the oracle: it produces the
# `launch` block that ships in the recipe, and it computes the same plan the C
# engine will, so a test can pin the two together.
#
# Why geometry belongs in the recipe
# ----------------------------------
# It was the one part of the chain that could not survive the trip to a shipped
# bundle. A caller could take CBOR to a correct .ll to a HSACO with no Python in
# the process, and then be stuck holding a compiled kernel with no idea what
# grid to launch it with, because the grid lived in host Python -- expressions
# like (n + tile_n - 1) // tile_n inside a dispatch function.
#
# A grid is a function of the shape, and the recipe language already exists to
# say exactly that. Carrying it as an intexpr over the spec axes means it is
# evaluated by the same evaluator as every loop bound the recipe emits, ships in
# the same artifact as the kernel it launches, and is covered by the same guard
# and ABI checks. Nothing has to be kept in sync by hand, because there is only
# one copy.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from rocke.portable_ir.utils.recipe_expand import (
    ExpandError,
    eval_intexpr,
    expand_recipe,
)

# Kernarg representations, mirroring rv_arg_classify in recipe_vm.cpp:
# (kind, size in bytes). Alignment equals size -- the AMDGPU natural-alignment
# rule -- so it is not stored separately.
_SCALARS: Dict[str, Tuple[str, int]] = {
    "i32": ("i32", 4),
    "i64": ("i64", 8),
    "f32": ("f32", 4),
}
_PTR = ("pointer", 8)


def attach_launch(
    recipe: Dict[str, Any],
    *,
    grid: Sequence[Any],
    block: Sequence[Any],
    lds_bytes: Any = 0,
) -> Dict[str, Any]:
    """Return `recipe` carrying launch geometry.

    `grid` and `block` are three intexprs each -- a plain int is a valid
    intexpr, so constant geometry needs no ceremony. `lds_bytes` is DYNAMIC
    shared memory, the argument hipModuleLaunchKernel takes; static LDS is
    already inside the HSACO and must not be counted here.

    Shape is validated now rather than at replay: a malformed block would
    otherwise surface inside a JIT on a machine with no way to trace it back to
    the generator that wrote it."""
    for name, dims in (("grid", grid), ("block", block)):
        if len(tuple(dims)) != 3:
            raise ExpandError(
                f"launch {name} needs exactly 3 intexprs, got {len(tuple(dims))}"
            )
    out = dict(recipe)
    out["launch"] = {
        "grid": list(grid),
        "block": list(block),
        "lds_bytes": lds_bytes,
    }
    return out


def eval_launch(
    recipe: Dict[str, Any],
    spec_int: Dict[str, int],
    spec_str: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Geometry for this shape, or None if the recipe carries none.

    Absence is None rather than a 1x1x1 default, matching
    rocke_launch_plan_geometry: a recipe recorded before geometry existed is not
    the same as a kernel that wants one workgroup, and defaulting would turn
    missing metadata into a silently wrong launch.

    Mirrors rv_plan_geometry in recipe_vm.cpp, including the >= 1 check."""
    block = recipe.get("launch")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ExpandError("recipe 'launch' is not an object")
    spec_str = spec_str or {}
    out: Dict[str, Any] = {}
    for key in ("grid", "block"):
        dims = block.get(key)
        if not isinstance(dims, list) or len(dims) != 3:
            raise ExpandError(f"recipe launch.{key} must be 3 intexprs")
        vals = []
        for i, node in enumerate(dims):
            v = eval_intexpr(node, {}, spec_int, spec_str)
            if v < 1:
                raise ExpandError(
                    f"recipe launch.{key}[{i}] evaluates to {v}, must be >= 1"
                )
            vals.append(v)
        out[key] = tuple(vals)
    lds = block.get("lds_bytes", 0)
    v = eval_intexpr(lds, {}, spec_int, spec_str)
    if v < 0:
        raise ExpandError(f"recipe launch.lds_bytes evaluates to {v}, must be >= 0")
    out["lds_bytes"] = v
    return out


def format_kernel_name(
    fmt: str, spec_int: Dict[str, int], spec_str: Optional[Dict[str, str]] = None
) -> str:
    """Mirror of rv_format_name: substitute {key} from the spec.

    Not str.format. An unknown key is DROPPED, token and all, which is what the
    C formatter does; str.format would raise, and leaving the token verbatim
    would produce a name that no HSACO contains. The point of mirroring the odd
    behaviour rather than improving on it is that both engines have to agree on
    the string passed to hipModuleGetFunction."""
    spec_str = spec_str or {}
    out: List[str] = []
    i = 0
    while i < len(fmt):
        if fmt[i] != "{":
            out.append(fmt[i])
            i += 1
            continue
        close = fmt.find("}", i)
        if close < 0:
            out.append(fmt[i])
            i += 1
            continue
        key = fmt[i + 1 : close]
        if key in spec_int:
            out.append(str(spec_int[key]))
        elif key in spec_str:
            out.append(str(spec_str[key]))
        i = close + 1
    return "".join(out)


def _type_name(node: Any) -> str:
    """Canonical name for a recipe type node.

    Spelled exactly as the C engine spells it (rocke_type_t::name) -- no space
    after the comma. The manifest vocabulary in runtime/packing.py writes
    'ptr<f32, global>' with a space, so it is easy to assume either is fine;
    they are different vocabularies, and this one is the engine's."""
    if isinstance(node, str):
        return node
    if isinstance(node, dict) and node.get("kind") == "ptr":
        return f"ptr<{node.get('pointee')},{node.get('space', 'global')}>"
    return str(node)


def _classify(node: Any) -> Tuple[str, int]:
    """(kind, size) for a kernarg, mirroring rv_arg_classify.

    Refuses anything it cannot size exactly. A guessed width does not fail, it
    shifts every following argument, and the kernel then reads garbage from
    offsets that look plausible -- so an unsupported type has to stop the plan
    rather than degrade it."""
    if isinstance(node, dict) and node.get("kind") == "ptr":
        return _PTR
    if isinstance(node, str):
        if node.startswith("ptr<"):
            return _PTR
        if node in _SCALARS:
            return _SCALARS[node]
    raise ExpandError(
        f"kernel arg type {_type_name(node)!r} has no kernarg representation here"
    )


def signature(
    recipe: Dict[str, Any],
    spec_int: Dict[str, int],
    spec_str: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """The kernel's arguments, in order, with kernarg offsets.

    Read off the EXPANDED recipe rather than the rolled one, for the same reason
    the C side reads the built kernel: a param could sit inside a static_if, and
    a signature that assumed otherwise would be wrong in exactly the cases that
    are hardest to notice.

    Offsets follow the AMDGPU natural-alignment rule, matching
    runtime/packing.py and rv_plan_on. Packing fields back to back is correct
    only until a signature mixes widths -- (ptr, i32, ptr) puts its last pointer
    at 16, not 12 -- and then it is wrong for everything after the mix."""
    flat = expand_recipe(recipe, {**spec_int, **(spec_str or {})})
    args: List[Dict[str, Any]] = []
    off = 0
    for instr in flat["program"]:
        if instr.get("op") != "param":
            continue
        kind, size = _classify(instr["type"])
        off = -(-off // size) * size
        args.append(
            {
                "name": instr["name"],
                "type": _type_name(instr["type"]),
                "kind": kind,
                "size": size,
                "offset": off,
            }
        )
        off += size
    return args


def kernarg_size(args: Sequence[Dict[str, Any]]) -> int:
    """Bytes to allocate: the end of the last argument.

    Deliberately not rounded up to the widest alignment, so this equals
    len(pack_args(...)) exactly -- runtime/packing.py packs a GEMM's
    (ptr,ptr,ptr,i32,i32,i32) as 36 bytes, not 40, and that is the size the
    working launch path uses. Mirrors rocke_launch_plan_kernarg_size; if the
    convention ever changes it has to change in both engines together."""
    if not args:
        return 0
    return int(args[-1]["offset"]) + int(args[-1]["size"])


def plan(
    recipe: Dict[str, Any],
    spec_int: Dict[str, int],
    spec_str: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Everything needed to launch: name, args, geometry. The Python mirror of
    rocke_launch_plan_t, and the oracle the C plan is tested against."""
    spec_str = spec_str or {}
    args = signature(recipe, spec_int, spec_str)
    return {
        "kernel_name": format_kernel_name(
            recipe.get("kernel_name_fmt", ""), spec_int, spec_str
        ),
        "args": args,
        "kernarg_size": kernarg_size(args),
        "geometry": eval_launch(recipe, spec_int, spec_str),
    }

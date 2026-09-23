# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""RecordingIRBuilder -- a build-time interception recorder.

Subclass of :class:`rocke.core.ir.IRBuilder` that records every emitted op into
a recipe (schema ``rocke.recipe/v1``) *as the kernel is built*. Any production
``build_*`` records with zero changes -- helpers, closures, dataclasses, and
descriptor math execute normally; only their emitted ops are captured.

This is Step 1 of ``dsl_docs/architecture/portable_ir_scaling_plan.md``: the
universal "record" path (the recorded recipe is concrete / per-shape, == portable
IR). Rolling builds on top of it.

Interception points (the IRBuilder internals this depends on):
  - ``param``                  -> records a param instruction
  - ``_emit(op)``              -> the single choke point every op flows through
  - ``push_region``/``pop_region`` -> tracks region nesting for scf.for/scf.if

Because recording rides ``_emit``/region management rather than the public
op-builder methods, **new ops are captured automatically** (no per-op code). If
a future IRBuilder change routes ops around ``_emit`` or changes region/Op/Param
structure, the recorded recipe diverges from the built KernelDef and the
``tests/test_recording_builder.py`` suite fails -- alerting developers.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, List, Tuple

from rocke.core.ir import IRBuilder, KernelDef, Op
from rocke.core.ir_export import _attrs_to_json, _type_to_json

# record_kernel() rebinds the name ``IRBuilder`` on every module that holds the
# real class -- including THIS one. Keep an unpatchable handle to the real class
# so the scan and the restore always compare/assign the genuine builder rather
# than whatever factory happens to be installed.
_REAL_IRBUILDER = IRBuilder


def _reg(v) -> str:
    """SSA register name without the leading '%'."""
    return v.name[1:] if v.name.startswith("%") else v.name


def _bare(name: str) -> str:
    return name[1:] if isinstance(name, str) and name.startswith("%") else name


def result_pfx(op: Op) -> str:
    """The ``result_name_hint`` Python used to name this op's results, or "".

    Python names every value ``%<hint><counter>`` (``IRBuilder._fresh``). A
    *concrete* recipe does not need the hint -- its binds are the finished names
    and the VM replays them verbatim. A *rolled* recipe does: one instruction
    expands N times, so each expansion must draw a fresh counter, and without the
    hint the VM falls back to the engine default "v" and emits ``%v14`` where
    Python wrote ``%mul14``. Recording the hint here (rather than mirroring
    Python's per-opcode table in C++) keeps the two engines from drifting: the
    hint travels with the data, so a new op needs no C++ change.

    Set by :meth:`RecordingIRBuilder._op` on the Op itself, so the post-hoc
    ``kerneldef_to_recipe`` walk over the same objects agrees with the live
    recording."""
    pfx = getattr(op, "_rec_pfx", None)
    return pfx if isinstance(pfx, str) and pfx and pfx != "v" else ""


def _result_fields(op: Op) -> Dict[str, Any]:
    """0 results -> {}; 1 -> {"out": {...}}; N>1 -> {"outs": [...]}."""
    if not op.results:
        return {}
    pfx = result_pfx(op)
    extra = {"pfx": pfx} if pfx else {}
    if len(op.results) == 1:
        r = op.results[0]
        return {"out": {"bind": _reg(r), "type": _type_to_json(r.type), **extra}}
    return {
        "outs": [
            {"bind": _reg(r), "type": _type_to_json(r.type), **extra}
            for r in op.results
        ]
    }


def shallow_instr(op: Op) -> Tuple[Dict[str, Any], List[List]]:
    """Build the recipe instruction for `op` with EMPTY region bodies, returning
    (instr, [body_list_per_region]). The body lists are filled by the recorder
    as the region's ops are emitted. Shared shape so the live recorder and any
    post-hoc walk agree."""
    if op.name == "scf.for":
        body: List = []
        inits = op.operands[3:]
        iter_meta = op.attrs.get("iter_args", [])
        instr = {
            "op": "scf_for",
            "iv": _bare(op.attrs["iv"]),
            "lo": _reg(op.operands[0]),
            "hi": _reg(op.operands[1]),
            "step": _reg(op.operands[2]),
            "iter": [
                {"name": _bare(m["name"]), "init": _reg(inits[i])}
                for i, m in enumerate(iter_meta)
            ],
            "results": [_reg(r) for r in op.results],
            "unroll": bool(op.attrs.get("unroll", False)),
            "elide_trailing_barrier": bool(
                op.attrs.get("elide_trailing_barrier", True)
            ),
            "body": body,
        }
        return instr, [body]
    if op.name == "scf.if":
        then_body: List = []
        instr = {"op": "scf_if", "cond": _reg(op.operands[0]), "then": then_body}
        bodies = [then_body]
        if len(op.regions) > 1:
            else_body: List = []
            instr["else"] = else_body
            bodies.append(else_body)
        return instr, bodies
    if op.name == "cf.return":
        return {"op": "ret"}, []
    # Generic op. Region-bearing ops other than scf.* are not representable yet;
    # raising here makes a new region-bearing op a loud test failure (the intended
    # "alert developers" behavior) rather than a silent capture gap.
    if op.regions:
        raise NotImplementedError(
            f"RecordingIRBuilder: region-bearing op {op.name!r} is not supported "
            f"(only scf.for / scf.if). Extend shallow_instr() to record it."
        )
    instr: Dict[str, Any] = {
        "op": "emit",
        "opcode": op.name,
        "in": [_reg(o) for o in op.operands],
    }
    instr.update(_result_fields(op))
    if op.attrs:
        instr["attrs"] = _attrs_to_json(op.attrs)
    return instr, []


def _conv_ops(ops: List[Op]) -> List[Dict[str, Any]]:
    out = []
    for op in ops:
        instr, _bodies = shallow_instr(op)
        if op.name == "scf.for":
            instr["body"] = _conv_ops(op.regions[0].ops)
        elif op.name == "scf.if":
            instr["then"] = _conv_ops(op.regions[0].ops)
            if len(op.regions) > 1:
                instr["else"] = _conv_ops(op.regions[1].ops)
        out.append(instr)
    return out


def kernel_to_recipe(kernel: KernelDef) -> Dict[str, Any]:
    """Independent post-hoc derivation of the recipe from a built KernelDef.

    Walks the final graph (rather than intercepting the build). Shares
    ``shallow_instr`` with :class:`RecordingIRBuilder`, so comparing this against
    a live recording verifies the interception (region routing, op ordering)
    stayed faithful. Unlike the older ``kerneldef_to_recipe`` walk this is
    multi-result aware (emits ``outs`` for N>1 result ops)."""
    params = [
        {
            "op": "param",
            "name": p.name,
            "type": _type_to_json(p.type),
            "bind": p.name,
            "attrs": dict(p.attrs),
        }
        for p in kernel.params
    ]
    return {
        "schema": "rocke.recipe/v1",
        "kernel_name_fmt": kernel.name,
        "spec": [],
        "attrs": _attrs_to_json(kernel.attrs),
        "program": params + _conv_ops(kernel.body.ops),
    }


class RecordingIRBuilder(IRBuilder):
    def __init__(self, kernel_name: str) -> None:
        super().__init__(kernel_name)
        self._rec_params: List[Dict[str, Any]] = []
        self._rec_body: List[Dict[str, Any]] = []
        self._rec_stack: List[List] = [self._rec_body]
        self._region_bodies: Dict[int, List] = {}
        self._pending_pfx: str = ""

    # ----- interception -----

    def _op(self, name: str, *a: Any, **k: Any) -> Op:
        """Capture the result-name hint so the recipe can carry it.

        ``_op`` emits from inside itself, so the hint has to be parked before
        delegating -- ``_emit`` reads it off ``_pending_pfx``. Ops built without
        going through ``_op`` simply record no hint, which is the same "v"
        default Python would have used."""
        hint = k.get("result_name_hint")
        if hint is None and len(a) >= 6:
            hint = a[5]
        prev = self._pending_pfx
        self._pending_pfx = hint if isinstance(hint, str) else "v"
        try:
            return super()._op(name, *a, **k)
        finally:
            self._pending_pfx = prev

    def param(self, name: str, t, **attrs: Any):
        v = super().param(name, t, **attrs)
        self._rec_params.append(
            {
                "op": "param",
                "name": name,
                "type": _type_to_json(t),
                "bind": name,
                "attrs": dict(attrs),
            }
        )
        return v

    def _emit(self, op: Op) -> None:
        if self._pending_pfx and getattr(op, "_rec_pfx", None) is None:
            op._rec_pfx = self._pending_pfx
        instr, bodies = shallow_instr(op)
        self._rec_stack[-1].append(instr)
        # Map each of this op's regions to the recipe body it should fill, so the
        # matching push_region routes subsequent ops into the right place.
        for region, body in zip(op.regions, bodies):
            self._region_bodies[id(region)] = body
        super()._emit(op)

    def push_region(self, region) -> None:
        super().push_region(region)
        body = self._region_bodies.get(id(region))
        if body is None:
            # A region pushed without a preceding _emit that registered it -> the
            # recorder's region tracking is out of sync with IRBuilder.
            raise RuntimeError(
                "RecordingIRBuilder: pushed an unregistered region; IRBuilder "
                "region handling may have changed."
            )
        self._rec_stack.append(body)

    def pop_region(self) -> None:
        super().pop_region()
        self._rec_stack.pop()

    # ----- output -----

    def recipe(self) -> Dict[str, Any]:
        """The recorded concrete recipe (schema rocke.recipe/v1)."""
        return {
            "schema": "rocke.recipe/v1",
            "kernel_name_fmt": self.kernel.name,
            # Concrete recipe (empty spec): the binds are the production builder's
            # unique SSA names, so the C VM names values verbatim and reproduces
            # the Python .ll byte-for-byte (not just an equivalent HSACO).
            "spec": [],
            "attrs": _attrs_to_json(self.kernel.attrs),
            "program": self._rec_params + self._rec_body,
        }


def _rocke_irbuilder_modules() -> List[Any]:
    """Every imported module that bound the real IRBuilder class via
    ``from ...core.ir import IRBuilder`` (or otherwise). Patching all of them
    makes recording work regardless of *which* module a builder constructs
    IRBuilder in -- production modules, helper modules (e.g. ``_fmha_common``),
    and builders defined in test/driver code alike. Only modules whose attribute
    *is* the real class are touched, and they are restored afterward."""
    import sys

    mods = []
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, "IRBuilder", None) is _REAL_IRBUILDER:
                mods.append(mod)
        except Exception:  # noqa: BLE001 - some modules raise on attribute access
            continue
    return mods


def _restore_late_bound(factory: Any) -> None:
    """Undo the factory binding in modules imported *during* a recording window.

    Such a module runs ``from ...core.ir import IRBuilder`` while ``core.ir``
    itself is patched, so it binds the factory instead of the real class. It was
    not in the pre-build scan, so the per-module restore never touches it and it
    would keep the (now dead) factory forever -- every later record_kernel()
    would then append to a stale ``created`` list and fail to match the kernel.
    """
    import sys

    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, "IRBuilder", None) is factory:
                mod.IRBuilder = _REAL_IRBUILDER
        except Exception:  # noqa: BLE001 - some modules raise on attribute access
            continue


def record_kernel(
    build_callable: Callable[[], KernelDef], *modules: Any
) -> Tuple[KernelDef, Dict[str, Any]]:
    """Run an *unmodified* production builder and capture its recorded recipe.

    Production ``build_*`` functions construct ``IRBuilder(spec.kernel_name())``
    internally and return ``b.kernel``. We temporarily rebind the ``IRBuilder``
    name (the real class) to a RecordingIRBuilder factory, run the build
    untouched, then return ``(kernel, recipe)`` for the builder that produced the
    returned kernel.

    With no ``modules`` given, every imported ``rocke`` module that bound the
    real IRBuilder is patched (auto-discovery) -- so it records any builder no
    matter where IRBuilder is constructed. Pass explicit modules to narrow it.
    """
    targets = list(modules) if modules else _rocke_irbuilder_modules()
    if not targets:
        raise RuntimeError("record_kernel: no module exposes the IRBuilder name")
    created: List[RecordingIRBuilder] = []

    def _factory(name: str, *a: Any, **k: Any) -> RecordingIRBuilder:
        b = RecordingIRBuilder(name, *a, **k)
        created.append(b)
        return b

    with contextlib.ExitStack() as stack:
        for mod in targets:
            orig = mod.IRBuilder
            mod.IRBuilder = _factory
            stack.callback(setattr, mod, "IRBuilder", orig)
        # Registered last => runs first on unwind, catching modules that were
        # imported while the patch was live (see _restore_late_bound).
        stack.callback(_restore_late_bound, _factory)
        kernel = build_callable()

    for b in created:
        if b.kernel is kernel:
            return kernel, b.recipe()
    raise RuntimeError(
        "record_kernel: no RecordingIRBuilder produced the returned kernel"
    )

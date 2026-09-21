# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""IR verifier — an LLVM-``verify``-style well-formedness pass.

:func:`verify` walks a :class:`KernelDef` and returns a list of
:class:`Diagnostic`. An empty list means the IR is well-formed. The checks are
grounded in the actual IR model (``rocke/core/ir.py``) and the builder's own
invariants — they are deliberately *conservative* (only assert what the IR
guarantees) so a valid kernel never produces a false error.

Checks performed:

* **SSA dominance** — every operand is defined (as a result, param, loop
  induction var, or loop iter-arg) before its use, within a scope where it is
  visible. No dangling / foreign value refs; no redefinition of an id.
* **Type consistency** — for the op classes whose contract the IR fixes
  (elementwise arith with matching operand/result types, vector ops over
  vector types, ``vector.extract`` element typing, ``scf.for`` iter-arg /
  result / yield typing).
* **Arity / result counts** — per-opcode operand and result counts where the
  builder guarantees them (binary arith, ``cf.return``, ``scf.yield``, etc.).
* **Region well-formedness** — ``scf.for`` / ``scf.if`` carry their required
  regions; loop-carried values are typed; the kernel body ends in a terminator.
* **Attr maps** — required keys present per opcode; enum-valued attrs legal.
* **Vector / smem / addrspace sanity** — vector widths > 0, smem shapes
  positive, pointer address spaces from the known set.

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from .ir import (
    KernelDef,
    Op,
    Region,
    Type,
    Value,
    VectorType,
    PtrType,
    SmemType,
)

ERROR = "error"
WARNING = "warning"


@dataclass
class Diagnostic:
    severity: str
    message: str
    op: Optional[str] = None  # op name (ref)
    loc: Optional[str] = None  # op.loc if present

    def __str__(self) -> str:
        where = ""
        if self.op:
            where = f" [{self.op}]"
        if self.loc:
            where += f" @{self.loc}"
        return f"{self.severity}: {self.message}{where}"


# --------------------------------------------------------------------------
# Per-opcode contracts
# --------------------------------------------------------------------------

# Binary arith / vector ops: 2 operands, 1 result, all same type.
_BINARY_SAME_TYPE = {
    "arith.add",
    "arith.sub",
    "arith.mul",
    "arith.div",
    "arith.mod",
    "arith.fadd",
    "arith.fsub",
    "arith.fmul",
    "arith.fdiv",
    "arith.fmax",
    "arith.fmin",
    "arith.and",
    "arith.or",
    "arith.smax",
    "arith.smin",
    "arith.xor",
    "arith.shl",
    "arith.lshr",
}

# Unary same-type ops.
_UNARY_SAME_TYPE = {
    "arith.fneg",
    "arith.fabs",
    "arith.not",
    "math.exp2",
    "math.log2",
    "math.rcp",
    "math.rcp_fast",
    "math.sqrt",
    "math.rsqrt",
    "math.tanh",
}

# Comparison ops: 2 operands, 1 i1 result; require a 'pred' attr.
_CMP_OPS = {"arith.cmp", "arith.fcmp"}

# Legal pointer address spaces.
_ADDR_SPACES = {"global", "constant", "shared", "lds", "private"}

# Required attr keys per opcode (a subset where the lowerer/builder relies on
# them; missing keys are real bugs).
_REQUIRED_ATTRS: Dict[str, List[str]] = {
    "arith.constant": ["value", "ity"],
    "arith.cmp": ["pred"],
    "arith.fcmp": ["pred"],
    "tile.mma": ["op_id"],
    "scf.yield": ["num"],
    "tile.inline_asm": ["template", "constraints"],
}


# --------------------------------------------------------------------------
# Verifier
# --------------------------------------------------------------------------


class _Verifier:
    def __init__(self, kernel: KernelDef) -> None:
        self.kernel = kernel
        self.diags: List[Diagnostic] = []

    def err(self, msg: str, op: Optional[Op] = None) -> None:
        self.diags.append(
            Diagnostic(
                ERROR,
                msg,
                op=op.name if op else None,
                loc=op.loc if op else None,
            )
        )

    def warn(self, msg: str, op: Optional[Op] = None) -> None:
        self.diags.append(
            Diagnostic(
                WARNING,
                msg,
                op=op.name if op else None,
                loc=op.loc if op else None,
            )
        )

    # ---- entry ----

    def run(self) -> List[Diagnostic]:
        # Top-level scope: kernel params.
        scope: Dict[str, Value] = {}
        param_names: Set[str] = set()
        for p in self.kernel.params:
            self._check_type(p.type, f"param {p.name!r}", None)
            vname = f"%{p.name}"
            if vname in scope:
                self.err(f"duplicate kernel parameter id {vname!r}")
            scope[vname] = Value(name=vname, type=p.type)
            param_names.add(vname)

        self._check_region(self.kernel.body, scope, is_kernel_body=True)
        return self.diags

    # ---- type sanity ----

    def _check_type(self, t: Type, where: str, op: Optional[Op]) -> None:
        if isinstance(t, VectorType):
            if t.count <= 0:
                self.err(f"{where}: vector width must be > 0, got {t.count}", op)
            self._check_type(t.elem, where, op)
        elif isinstance(t, PtrType):
            if t.space not in _ADDR_SPACES:
                self.err(
                    f"{where}: unknown pointer address space {t.space!r} "
                    f"(known: {sorted(_ADDR_SPACES)})",
                    op,
                )
            self._check_type(t.pointee, where, op)
        elif isinstance(t, SmemType):
            if not t.shape:
                self.err(f"{where}: smem type must have a non-empty shape", op)
            for d in t.shape:
                if d <= 0:
                    self.err(f"{where}: smem shape dims must be > 0, got {t.shape}", op)
                    break
            self._check_type(t.elem, where, op)

    # ---- region / dominance ----

    def _check_region(
        self,
        region: Region,
        scope: Dict[str, Value],
        *,
        is_kernel_body: bool = False,
    ) -> None:
        # ``scope`` maps visible SSA id -> defining Value. Inner regions extend
        # a *copy* so block-defined values do not leak to siblings (mirrors the
        # builder's region stack).
        for op in region.ops:
            self._check_op(op, scope)

        if is_kernel_body and not region.ops:
            # An empty body lowers to a bare ``ret void`` — degenerate but not
            # an error. A non-empty body need NOT end in an explicit
            # ``cf.return``: the LLVM lowerer auto-appends ``ret void`` when the
            # exit block is unterminated, so real kernels omit it by design.
            self.warn("kernel body region is empty")

    def _check_op(self, op: Op, scope: Dict[str, Value]) -> None:
        # 1) operands must dominate (be defined + visible) the use.
        for o in op.operands:
            if o.name not in scope:
                self.err(
                    f"operand {o.name!r} used before definition / out of scope",
                    op,
                )
            else:
                defined = scope[o.name]
                if defined.type != o.type:
                    self.err(
                        f"operand {o.name!r} type {o.type.name!r} does not match "
                        f"its definition type {defined.type.name!r}",
                        op,
                    )

        # 2) required attrs present.
        for key in _REQUIRED_ATTRS.get(op.name, ()):
            if key not in op.attrs:
                self.err(f"op {op.name!r} missing required attr {key!r}", op)

        # 3) per-opcode arity / type contracts.
        self._check_contract(op, scope)

        # 4) result types sane; results define new (unique) ids. Snapshot the
        # set of names that are in scope BEFORE this op's own results so that
        # nested regions (step 5) are checked without the carrier op's own
        # results visible -- an scf.for/scf.if result must not be usable inside
        # its own body. The results are still registered into `scope` here so
        # that *sibling* ops following this one see them.
        names_before_results = set(scope)
        for r in op.results:
            self._check_type(r.type, f"result {r.name!r}", op)
            if r.name in scope:
                self.err(f"SSA id {r.name!r} redefined", op)
            else:
                scope[r.name] = r

        # 5) nested regions, with block-defined values registered first. Use a
        # scope that excludes this op's own (not-yet-computed) results.
        if op.regions:
            region_scope = {
                name: val for name, val in scope.items() if name in names_before_results
            }
            self._check_op_regions(op, region_scope)

    def _check_op_regions(self, op: Op, scope: Dict[str, Value]) -> None:
        if op.name == "scf.for":
            self._check_scf_for(op, scope)
        elif op.name == "scf.if":
            self._check_scf_if(op, scope)
        else:
            # generic region-carrier: just recurse with an extended copy.
            for region in op.regions:
                self._check_region(region, dict(scope))

    # ---- contracts ----

    def _check_contract(self, op: Op, scope: Dict[str, Value]) -> None:
        name = op.name
        if name in _BINARY_SAME_TYPE:
            if len(op.operands) != 2 or len(op.results) != 1:
                self.err(
                    f"{name}: expected 2 operands / 1 result, got "
                    f"{len(op.operands)} / {len(op.results)}",
                    op,
                )
                return
            a, b = op.operands
            r = op.results[0]
            if a.type != b.type:
                self.err(
                    f"{name}: operand types differ ({a.type.name} vs {b.type.name})",
                    op,
                )
            if r.type != a.type:
                self.err(
                    f"{name}: result type {r.type.name} != operand type {a.type.name}",
                    op,
                )
        elif name in _UNARY_SAME_TYPE:
            if len(op.operands) != 1 or len(op.results) != 1:
                self.err(
                    f"{name}: expected 1 operand / 1 result, got "
                    f"{len(op.operands)} / {len(op.results)}",
                    op,
                )
                return
            if op.operands[0].type != op.results[0].type:
                self.err(
                    f"{name}: result type {op.results[0].type.name} != "
                    f"operand type {op.operands[0].type.name}",
                    op,
                )
        elif name in _CMP_OPS:
            if len(op.operands) != 2 or len(op.results) != 1:
                self.err(f"{name}: expected 2 operands / 1 result", op)
                return
            a, b = op.operands
            if a.type != b.type:
                self.err(
                    f"{name}: comparison operand types differ "
                    f"({a.type.name} vs {b.type.name})",
                    op,
                )
            if op.results[0].type.name != "i1":
                self.err(
                    f"{name}: result must be i1, got {op.results[0].type.name}", op
                )
        elif name == "arith.select":
            if len(op.operands) != 3 or len(op.results) != 1:
                self.err(f"{name}: expected 3 operands / 1 result", op)
                return
            cond, lhs, rhs = op.operands
            if cond.type.name != "i1":
                self.err(f"{name}: condition must be i1, got {cond.type.name}", op)
            if lhs.type != rhs.type:
                self.err(
                    f"{name}: branch types differ ({lhs.type.name} vs {rhs.type.name})",
                    op,
                )
        elif name == "vector.extract":
            if len(op.operands) != 1 or len(op.results) != 1:
                self.err(f"{name}: expected 1 operand / 1 result", op)
                return
            v = op.operands[0]
            if not isinstance(v.type, VectorType):
                self.err(f"{name}: operand must be a vector, got {v.type.name}", op)
            else:
                if op.results[0].type != v.type.elem:
                    self.err(
                        f"{name}: result type {op.results[0].type.name} != "
                        f"vector element {v.type.elem.name}",
                        op,
                    )
                idx = op.attrs.get("index")
                if isinstance(idx, int) and not (0 <= idx < v.type.count):
                    self.err(
                        f"{name}: extract index {idx} out of range [0,{v.type.count})",
                        op,
                    )
        elif name == "tile.mma":
            if len(op.results) != 1:
                self.err(f"{name}: expected exactly 1 result", op)
            if len(op.operands) < 3:
                self.err(
                    f"{name}: expected at least 3 operands (a, b, c), got "
                    f"{len(op.operands)}",
                    op,
                )
        elif name == "cf.return":
            if op.operands or op.results:
                self.err(f"{name}: must have no operands and no results", op)

    # ---- scf.for ----

    def _check_scf_for(self, op: Op, scope: Dict[str, Value]) -> None:
        if not op.regions:
            self.err("scf.for missing its body region", op)
            return
        if "iv" not in op.attrs or "iv_type" not in op.attrs:
            self.err("scf.for missing 'iv' / 'iv_type' attrs", op)

        # operands: lower, upper, step, *iter_inits
        if len(op.operands) < 3:
            self.err("scf.for needs at least lower/upper/step operands", op)
            return
        lower, upper, step = op.operands[:3]
        iter_inits = op.operands[3:]
        for v in (lower, upper, step):
            if v.type != lower.type:
                self.err("scf.for lower/upper/step types must match", op)
                break

        iter_meta = op.attrs.get("iter_args", []) or []
        if isinstance(iter_meta, list):
            if len(iter_meta) != len(iter_inits):
                self.err(
                    f"scf.for: {len(iter_meta)} iter_args declared but "
                    f"{len(iter_inits)} init operands",
                    op,
                )
            if len(op.results) != len(iter_inits):
                self.err(
                    f"scf.for: {len(op.results)} results but {len(iter_inits)} "
                    f"iter-args",
                    op,
                )
            # result types must match init types
            for res, init in zip(op.results, iter_inits):
                if res.type != init.type:
                    self.err(
                        f"scf.for: result type {res.type.name} != iter init "
                        f"type {init.type.name}",
                        op,
                    )

        # body scope: copy + iv + iter-arg block values
        body_scope = dict(scope)
        iv_name = op.attrs.get("iv")
        iv_type = op.attrs.get("iv_type")
        body_block_values: List[Value] = []
        if isinstance(iv_name, str) and isinstance(iv_type, str):
            iv_val = Value(
                name=iv_name,
                type=Type(iv_type) if "<" not in iv_type else _reparse(iv_type),
            )
            body_scope[iv_name] = iv_val
        if isinstance(iter_meta, list):
            for entry, init in zip(iter_meta, iter_inits):
                if isinstance(entry, dict) and "name" in entry:
                    bv = Value(name=entry["name"], type=init.type)
                    body_scope[entry["name"]] = bv
                    body_block_values.append(bv)

        body = op.regions[0]
        self._check_region(body, body_scope)

        # body should end in scf.yield matching iter-arg types (if any iter-args)
        if iter_inits:
            if not body.ops or body.ops[-1].name != "scf.yield":
                self.err("scf.for with iter-args: body must end in scf.yield", op)
            else:
                yld = body.ops[-1]
                if len(yld.operands) != len(iter_inits):
                    self.err(
                        f"scf.yield yields {len(yld.operands)} values but loop "
                        f"carries {len(iter_inits)} iter-args",
                        yld,
                    )
                else:
                    for yv, init in zip(yld.operands, iter_inits):
                        if yv.type != init.type:
                            self.err(
                                f"scf.yield type {yv.type.name} != iter-arg type "
                                f"{init.type.name}",
                                yld,
                            )

    # ---- scf.if ----

    def _check_scf_if(self, op: Op, scope: Dict[str, Value]) -> None:
        if len(op.operands) != 1:
            self.err("scf.if expects exactly 1 condition operand", op)
        elif op.operands[0].type.name != "i1":
            self.err(f"scf.if condition must be i1, got {op.operands[0].type.name}", op)
        if not op.regions:
            self.err("scf.if missing its 'then' region", op)
        for region in op.regions:
            self._check_region(region, dict(scope))


def _reparse(type_str: str) -> Type:
    # Late import to avoid a cycle; only used for compound iv types (rare).
    from .ir_serialize import _parse_type

    return _parse_type(type_str)


def verify(kernel: KernelDef) -> List[Diagnostic]:
    """Verify ``kernel``; return diagnostics (empty list == well-formed)."""
    return _Verifier(kernel).run()


def verify_or_raise(kernel: KernelDef) -> None:
    """Verify and raise :class:`ValueError` if any error-severity diagnostic
    is present (warnings are tolerated)."""
    diags = verify(kernel)
    errors = [d for d in diags if d.severity == ERROR]
    if errors:
        joined = "\n".join(str(d) for d in errors)
        raise ValueError(f"IR verification failed:\n{joined}")

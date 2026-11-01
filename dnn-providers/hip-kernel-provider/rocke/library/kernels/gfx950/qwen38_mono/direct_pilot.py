# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Rust direct code-object compiler pilot (Qwen3.8 true monokernel, R1).

The representative Qwen3.8 builds the direct compiler gates run on, and the
check that every one of them is either compiled by the direct backend or
refused cleanly (``ROCKE_DIRECT_UNSUPPORTED`` -> ``DirectUnsupported`` ->
COMGR), never a hard parse / malformed error.

``representative_builds()`` uses only builders and knobs that exist in the
mono tree as well (absolute imports, no new API), so the knob-off byte-identity
script can load this file by path against either tree. ``knob`` builds (the
framework's ``FrameworkKnobs.code_object_backend``) need this tree.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable

__all__ = [
    "FRAMEWORK_BUILDS",
    "HYBRID_HEAD",
    "R2_PILOTS",
    "neutralize_census",
    "pilot_builds",
    "representative_builds",
    "with_code_object_backend",
]

HEAD = dict(n=248320, k=2560)

#: The hybrid stack's served greedy-argmax head (ATOM ``rocke_head``: the
#: ``fused`` fold and ``DEFAULT_SPEC["argmax"]``), at the rows <= 4 build.
HYBRID_HEAD = dict(HEAD, rows_max=4, epilogue="fused", waves_per_eu=1)

#: R2 pilots (labels of :func:`pilot_builds`): the hybrid head and the toy
#: chain at ``best_spec``, each as shipped and with ``irops`` (device ops as
#: IR ops, the IR-only form the direct code-object compiler lowers).
R2_PILOTS = (
    "head_hybrid", "head_hybrid_irops",
    "toy_best_d4w8_l2", "toy_best_d4w8_l2_irops",
    "toy_best_d8w0", "toy_best_d8w0_irops",
)  # fmt: skip

# Labels of the builds that go through ``framework.build_layer_kernel`` (the
# ones the framework knob reaches).
FRAMEWORK_BUILDS = ("toy_default", "toy_best_d4w8_l2", "toy_best_d8w0")


def _toy(spec_fn: Callable, code_object_backend: str = ""):
    from kernels.gfx950.qwen38_mono import toy_chain as T

    spec = spec_fn(T)
    if code_object_backend:
        knobs = dataclasses.replace(spec.knobs, code_object_backend=code_object_backend)
        spec = dataclasses.replace(spec, knobs=knobs)
    return T.build_toy_chain(spec)


def _toy_builds(code_object_backend: str = "") -> list[tuple[str, Callable]]:
    return [
        ("toy_default", lambda: _toy(lambda T: T.ToyChainSpec(), code_object_backend)),
        ("toy_best_d4w8_l2", lambda: _toy(lambda T: T.best_spec(4, 8, layers=2), code_object_backend)),
        ("toy_best_d8w0", lambda: _toy(lambda T: T.best_spec(8, 0), code_object_backend)),
    ]


def _b3(variant: str | None):
    from kernels.gfx950.qwen38_mono import stage_b3 as SB

    if variant is None:
        return SB.build_b3_kernel(SB.B3Shape(1))
    shape = {"r1": SB.B3Shape(1), "r4s": SB.B3Shape(4, True)}[variant]
    return SB.build_b3_kernel(shape, SB.knobs_from_spec(SB.TUNED_SPECS[variant]))


def _fused(rows: int):
    from kernels.gfx950.qwen38_mono import moe_fused as MF
    from kernels.gfx950.qwen38_mono import stage_moe_fused as SMF

    return MF.build_fused_suffix(rows, SMF.best_knobs(rows, "best_plain"))


def _suffix(rows: int):
    from kernels.gfx950.qwen38_mono import moe as MO
    from kernels.gfx950.qwen38_mono import stage_moe as SM

    return MO.build_moe_suffix(MO.MoeSuffixSpec(rows=rows, knobs=SM.best_knobs(rows)))


def _head(**kw):
    from kernels.gfx950 import mx_decode_gemv as M

    return M.build_mx_decode_gemv(M.MxDecodeGemvSpec(**HEAD, **kw))


def _probe():
    from kernels.gfx950 import mx_mfma as K

    return K.build_mx_mfma_probe(K.MxMfmaProbeSpec(), arch="gfx950")


def representative_builds() -> list[tuple[str, Callable]]:
    """``(label, build)`` of the representative Qwen3.8 kernels (research
    ``rust-direct.md`` 2.1), knobs as shipped: the toy chain (default asm
    mailbox; ``best_spec`` with the ``ir`` mailbox, NUMA copies, kernarg
    preload, prefetch), B3 r1 default and r1 / r4s tuned, the fused MoE r1 /
    r4, the MoE suffix r1 / r4, the MXFP4 head GEMV (logits, fused argmax,
    ``nt=2``) and the scaled-MFMA probe."""
    return _toy_builds() + [
        ("b3_r1_default", lambda: _b3(None)),
        ("b3_r1_tuned", lambda: _b3("r1")),
        ("b3_r4s_tuned", lambda: _b3("r4s")),
        ("moe_fused_r1", lambda: _fused(1)),
        ("moe_fused_r4", lambda: _fused(4)),
        ("moe_suffix_r1", lambda: _suffix(1)),
        ("moe_suffix_r4", lambda: _suffix(4)),
        ("head_logits", lambda: _head()),
        ("head_fused_argmax", lambda: _head(epilogue="fused")),
        ("head_nt2", lambda: _head(nt=2)),
        ("mx_mfma_probe", _probe),
    ]


def _toy_irops(spec_fn: Callable, irops: bool):
    from kernels.gfx950.qwen38_mono import toy_chain as T

    spec = spec_fn(T)
    if irops:
        spec = dataclasses.replace(spec, knobs=dataclasses.replace(spec.knobs, irops=True))
    return T.build_toy_chain(spec)


def pilot_builds() -> list[tuple[str, Callable]]:
    """``(label, build)`` of :data:`R2_PILOTS`."""
    return [
        ("head_hybrid", lambda: _head(**{k: v for k, v in HYBRID_HEAD.items() if k not in HEAD})),
        ("head_hybrid_irops", lambda: _head(irops=True, **{k: v for k, v in HYBRID_HEAD.items() if k not in HEAD})),
        ("toy_best_d4w8_l2", lambda: _toy_irops(lambda T: T.best_spec(4, 8, layers=2), False)),
        ("toy_best_d4w8_l2_irops", lambda: _toy_irops(lambda T: T.best_spec(4, 8, layers=2), True)),
        ("toy_best_d8w0", lambda: _toy_irops(lambda T: T.best_spec(8, 0), False)),
        ("toy_best_d8w0_irops", lambda: _toy_irops(lambda T: T.best_spec(8, 0), True)),
    ]


def with_code_object_backend(code_object_backend: str) -> list[tuple[str, Callable]]:
    """The framework builds of :func:`representative_builds` with the
    framework knob ``FrameworkKnobs.code_object_backend`` set."""
    return _toy_builds(code_object_backend)


# ---------------------------------------------------------------- census
#
# A build stops at its FIRST refusal. ``neutralize_census`` looks past it: it
# replaces every construct the direct compiler refused by a placeholder of the
# same result types (or drops it when it has none), recompiles, and repeats,
# until the IR compiles or the compiler fails with an error other than
# ``ROCKE_DIRECT_UNSUPPORTED``. The placeholders change what the kernel
# computes: the objects are diagnostics, never launched. The result lists
# every refusal in order (what the direct backend still has to lower for the
# build) and the final state (``compiled`` or the error a later stage of the
# port must clear).

_OP_NAME = re.compile(r" = ([a-z_]+\.[A-Za-z0-9_]+) \(|^\s*([a-z_]+\.[A-Za-z0-9_]+) \(")
_REFUSED_OP = re.compile(r'op "([^"]+)" is not lowered by the direct backend')
_REFUSED_VALUE = re.compile(r"^\w+ (%[\w.$]+): ")
_TYPE_VEC = re.compile(r"^vec<([a-z0-9]+)x(\d+)>$")
_TYPE_PTR = re.compile(r"^ptr<([a-z0-9]+),global>$")
_FLOATS = ("f32", "f16", "bf16", "f64")
_UNSUPPORTED = "ROCKE_DIRECT_UNSUPPORTED: "


def _defs(line: str) -> tuple[list[tuple[str, str]], str]:
    """``([(name, type)], indent)`` of an op line's results."""
    indent = line[: len(line) - len(line.lstrip())]
    head = line.strip()
    eq = head.find(" = ")
    if not head.startswith("%") or eq < 0:
        return [], indent
    out = []
    for entry in re.split(r", (?=%)", head[:eq]):
        name, _, ty = entry.partition(" : ")
        out.append((name.strip(), ty.strip()))
    return out, indent


class _Placeholders:
    """Lines defining a wave-uniform value of a type from the block id alone."""

    def __init__(self) -> None:
        self.ptr_params: dict[str, str] = {}
        self.n = 0

    def _tmp(self, name: str) -> str:
        self.n += 1
        return f"{name}__nz{self.n}"

    def lines(self, name: str, ty: str, indent: str) -> list[str] | None:
        t = self._tmp(name)
        tid = f'{indent}{t} : i32 = gpu.block_id (  ) {{ axis = s:"x" }}'
        if ty == "i32":
            return [f'{indent}{name} : i32 = gpu.block_id (  ) {{ axis = s:"x" }}']
        if ty == "i64":
            return [tid, f"{indent}{name} : i64 = arith.zext ( {t} )"]
        if ty == "f32":
            return [tid, f"{indent}{name} : f32 = arith.bitcast ( {t} )"]
        if ty == "i1":
            return [tid, f'{indent}{name} : i1 = arith.cmp ( {t}, {t} ) {{ pred = s:"eq" }}']
        if ty in ("f16", "bf16"):
            f = self._tmp(name)
            return [
                tid,
                f"{indent}{f} : f32 = arith.bitcast ( {t} )",
                f'{indent}{name} : {ty} = arith.cast_f32_to ( {f} ) {{ target = s:"{ty}" }}',
            ]
        m = _TYPE_VEC.match(ty)
        if m:
            elem, n = m.group(1), int(m.group(2))
            fill = "f:0.0" if elem in _FLOATS else "i:0"
            return [
                f'{indent}{name} : {ty} = arith.constant_vec (  ) '
                f'{{ elem = s:"{elem}", fill = {fill}, vec = i:{n} }}'
            ]
        m = _TYPE_PTR.match(ty)
        if m:
            base = self.ptr_params.setdefault(m.group(1), f"%__nz_ptr_{m.group(1)}")
            return [tid, f"{indent}{name} : {ty} = tile.global_ptr_add ( {base}, {t} )"]
        return None

    def params(self) -> list[str]:
        return [f"    {p} : ptr<{e},global>" for e, p in sorted(self.ptr_params.items())]


def _is_pin(line: str) -> bool:
    return 'constraints = s:"=v,0"' in line and 'template = s:""' in line


def neutralize_census(ir_text: str, compile_fn: Callable[[str], object], *, max_steps: int = 64) -> dict:
    """Refusals of ``ir_text`` past the first (see the section comment).
    ``compile_fn(text)`` compiles serialized IR, raising the
    ``rocke.runtime.code_object`` errors. Returns ``{"refusals": [...],
    "final": "compiled" | "<error>", "steps": n}``."""
    from rocke.runtime.code_object import CodeObjectError, DirectUnsupported

    lines = ir_text.split("\n")
    ph = _Placeholders()
    refusals: list[str] = []

    def result(final: str, step: int) -> dict:
        return {"refusals": refusals, "final": final, "steps": step}

    def replace(pick: Callable[[str], bool]) -> tuple[list[str], str | None, bool]:
        new, changed = [], False
        for ln in lines:
            if not pick(ln):
                new.append(ln)
                continue
            out: list[str] = []
            for name, ty in _defs(ln)[0]:
                repl = ph.lines(name, ty, _defs(ln)[1])
                if repl is None:
                    return lines, f"no placeholder for {ln.strip()[:160]}", False
                out += repl
            new += out
            changed = True
        return new, None, changed

    for step in range(max_steps):
        text = "\n".join(lines)
        if ph.ptr_params:
            text = text.replace("  params {\n", "  params {\n" + "\n".join(ph.params()) + "\n", 1)
        try:
            compile_fn(text)
            return result("compiled", step)
        except DirectUnsupported as e:
            msg = str(e).split("rejected the kernel: ", 1)[-1]
        except CodeObjectError as e:
            return result(str(e).split("rejected the kernel: ", 1)[-1], step)
        refusals.append(msg)
        detail = msg[len(_UNSUPPORTED):] if msg.startswith(_UNSUPPORTED) else msg
        problem, changed = None, False
        m = _REFUSED_OP.search(detail)
        if m:  # every line of the op

            def pick(ln: str, op: str = m.group(1)) -> bool:
                mm = _OP_NAME.search(ln)
                return bool(mm) and (mm.group(1) or mm.group(2)) == op

            lines, problem, changed = replace(pick)
        elif detail.startswith("kernarg preload (inreg)"):
            lines = [ln.replace(" { inreg = b:true }", "") for ln in lines]
            changed = True
        elif "tile.inline_asm other than" in detail:

            def pick(ln: str) -> bool:
                s = ln.lstrip()
                return (" = tile.inline_asm (" in ln or s.startswith("tile.inline_asm (")) and not _is_pin(ln)

            lines, problem, changed = replace(pick)
        else:
            m = _REFUSED_VALUE.match(detail)
            if m:  # one op, named by its result

                def pick(ln: str, name: str = m.group(1)) -> bool:
                    return any(n == name for n, _ in _defs(ln)[0])

                lines, problem, changed = replace(pick)
        if problem:
            return result(problem, step)
        if not changed:
            return result("stuck: no rule neutralizes the last refusal", step)
    return result(f"stopped after {max_steps} steps", max_steps)

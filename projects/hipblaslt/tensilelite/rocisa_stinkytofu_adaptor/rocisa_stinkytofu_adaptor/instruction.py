# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Instruction shims mirroring rocisa/src/instruction/.

Real: VMovB32, SMovB32/B64, SLoadB32-B512, SNop, and base Instruction/CommonInstruction.
Each exposes to_stinky_logical() for logical IR lowering.
"""

from __future__ import annotations

from copy import deepcopy as _deepcopy
from typing import Any, Dict, List, Optional

from ._dummy import make_dummy_class, make_dummy_func
from .enum import InstType

_P = "rocisa.instruction"


# ==========================================================================
# Base instruction types
# source: rocisa/rocisa/src/instruction/instruction.{hpp,cpp}
# ==========================================================================
#
# The full rocisa hierarchy is:
#
#   Item -> Instruction -> CommonInstruction (VMovB32, VAddF32, ...)
#                       -> CompositeInstruction (VMovB64, SLongBranch*, ...)
#                       -> MacroInstruction (V_MAGIC_DIV, MAINLOOP, ...)
#
# ``Instruction`` / ``CommonInstruction`` / ``MacroInstruction`` are
# real; ``CompositeInstruction`` is still a dummy until a concrete
# subclass needs it.
#
# KernelWriter exercises:
#   - ``isinstance(x, Instruction)`` (KernelWriter.py:1105, 1790, Activation.py)
#   - ``isinstance(x, CommonInstruction) and ... and x.dst.regType == 'm'``
#     (Components/SubtileBasedInstructionScheduler.py:151)
#   - ``inst.getParams()`` (Components/SIA.py, Activation.py)
#   - ``inst.dst = vgpr(...)`` post-construction rebinding
#     (Components/GSU.py:1144 etc.)
#   - ``inst.comment = ...`` post-construction mutation (KernelWriter:3422 etc.)
#   - ``MacroInstruction(name="V_MAGIC_DIV", args=[...])`` macro-call emit
#     (KernelWriterAssembly:2784, 3291, 9452 + Components/StreamK).


def _format_str(output_inline_asm: bool, inst_str: str, comment: str,
                no_comment: bool) -> str:
    """Byte-parity port of ``rocisa::formatStr`` (format.hpp:54-75).

    Reproduces the exact padding-to-50 + ``" // "`` + comment + ``"\\n"``
    layout so emitted asm diffs byte-for-byte against rocisa native.
    """
    formatted = inst_str
    if output_inline_asm:
        # Inline-asm path wraps the instruction in quotes + ``\n\t``.
        formatted = '"' + formatted + '\\n\\t"'
    if comment and not no_comment:
        pad = max(0, 50 - len(formatted))
        return formatted + (" " * pad) + " // " + comment + "\n"
    return formatted + "\n"


# ``outputNoComment`` lives in ``base.py`` so ``code.TextBlock.toString``
# and the instruction formatter share one reader. See ``base.outputNoComment``
# for the full decoupling rationale (Python-side mirror, lazy-import, etc.).
from .base import outputNoComment as _output_no_comment  # noqa: E402


class Instruction:
    """Minimal port of ``rocisa::Instruction`` (instruction.hpp:119-286).

    Carries the per-instance state KernelWriter sets / reads but leaves
    ``toString`` / ``getParams`` family to subclasses. The MSB workaround
    (``setMsb``) is intentionally omitted -- the stinkytofu left-path
    runs ``InsertVgprMsbPass`` to inject ``s_set_vgpr_msb`` instructions
    closer to the truth source. If a future test diffs rocisa-emitted
    text against the right-path this gap will need to be revisited.
    """

    __slots__ = (
        "name", "parent",
        "instType", "comment", "instStr", "outputInlineAsm",
        "m_memToken",
    )

    def __init__(self, instType: Any, comment: str = ""):
        self.name: str = ""  # rocisa Item ctor uses "" (not the class name)
        self.parent: Any = None
        self.instType: Any = instType
        # Coerce None -> "" so downstream ``inst.comment += ...`` (e.g.
        # Activation.FuseInstruction) never hits ``NoneType += str``. A None
        # comment only ever arises from a positional-arg mismatch against a
        # native rocisa signature; treat it as "no comment".
        self.comment: str = comment if comment is not None else ""
        self.instStr: str = ""
        self.outputInlineAsm: bool = False
        self.m_memToken: Any = None

    # ---------------------------------------------------- memToken / inline
    def setMemToken(self, token: Any) -> None:
        self.m_memToken = token

    def getMemToken(self) -> Any:
        return self.m_memToken

    def setInlineAsm(self, is_true: bool) -> None:
        self.outputInlineAsm = bool(is_true)

    # ---------------------------------------------------------- inst label
    def setInst(self, inst_str: str) -> None:
        self.instStr = inst_str

    def preStr(self) -> str:
        # rocisa default; CompositeInstruction overrides.
        return self.instStr

    # ----------------------------------------------------- comment formats
    def formatOnly(self, inst_str: str, comment: str) -> str:
        return _format_str(self.outputInlineAsm, inst_str, comment,
                           _output_no_comment())

    def formatWithComment(self, inst_str: str) -> str:
        return _format_str(self.outputInlineAsm, inst_str, self.comment,
                           _output_no_comment())

    def formatWithExtraComment(self, inst_str: str, extra: str) -> str:
        return _format_str(self.outputInlineAsm, inst_str, self.comment + extra,
                           _output_no_comment())

    # ----------------------------------------------------- to be overridden
    def toString(self) -> str:
        raise NotImplementedError(
            "Subclass must override toString() (rocisa::Instruction is abstract)."
        )

    def __str__(self) -> str:
        return self.toString()

    def getParams(self):
        raise NotImplementedError("Subclass must override getParams()")

    def getDstParams(self):
        raise NotImplementedError("Subclass must override getDstParams()")

    def getSrcParams(self):
        raise NotImplementedError("Subclass must override getSrcParams()")

    def getIssueLatency(self) -> int:
        return 1  # rocisa default; override in subclasses with real values.

    def getIssueCycles(self) -> int:
        return 1

    # ----------------------------------------------------- copy / pickle
    def __deepcopy__(self, memo):
        # rocisa raises std::runtime_error("Deepcopy not supported for
        # Instruction"); KernelWriter doesn't deepcopy bare Instructions
        # (it deepcopies concrete subclasses which override this). We
        # mirror the rocisa intent.
        raise RuntimeError("Deepcopy not supported for Instruction")

    def __reduce__(self):
        raise RuntimeError("Pickling not supported for Instruction")


def _input_to_str(arg: Any) -> str:
    """Port of ``rocisa::InstructionInputToString`` (instruction.hpp:41-71).

    Handles the std::variant<Container, int, double, string> by Python
    duck-typing: anything with ``toString`` (Container, RegisterContainer,
    Holders) gets ``.toString()``; everything else uses ``str()`` with
    the C++ ``double`` quirk (append ``.0`` when neither ``.`` nor ``e``
    is present so a literal ``1.0`` round-trips byte-for-byte).
    """
    if hasattr(arg, "toString"):
        return arg.toString()
    if isinstance(arg, bool):
        # bool is an int subclass; rocisa stores it as int -> "0"/"1".
        return str(int(arg))
    if isinstance(arg, int):
        return str(arg)
    if isinstance(arg, float):
        # ``%.17g`` matches C++ snprintf; trailing ``.0`` mirrors the
        # std::visit branch in instruction.hpp:59-62.
        s = format(arg, ".17g")
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s
    return str(arg)


class CommonInstruction(Instruction):
    """Port of ``rocisa::CommonInstruction`` (instruction.hpp:382-526).

    Holds ``dst`` (+ optional ``dst1``), a list of ``srcs``, and modifier
    bundles. ``toString`` produces ``"<inst> <dst>, <src0>, <src1> ... //
    comment\\n"`` to match rocisa's ``getArgStr`` + ``formatWithComment``.
    """

    __slots__ = ("dst", "dst1", "srcs", "dpp", "sdwa", "vop3")

    def __init__(self, instType: Any, dst: Any, srcs: List[Any],
                 dpp: Any = None, sdwa: Any = None, vop3: Any = None,
                 comment: str = ""):
        super().__init__(instType, comment)
        self.dst: Any = dst
        self.dst1: Any = None
        self.srcs: List[Any] = list(srcs)
        self.dpp: Any = dpp
        self.sdwa: Any = sdwa
        self.vop3: Any = vop3

    # ----------------------------------------------------- accessors
    def setSrc(self, idx: int, src: Any) -> None:
        self.srcs[idx] = src

    def getParams(self):
        l = []
        if self.dst is not None:
            l.append(self.dst)
        if self.dst1 is not None:
            l.append(self.dst1)
        l.extend(self.srcs)
        return l

    def getDstParams(self):
        dsts = []
        if self.dst is not None:
            dsts.append(self.dst)
        if self.dst1 is not None:
            dsts.append(self.dst1)
        return dsts

    def getSrcParams(self):
        return list(self.srcs)

    # ----------------------------------------------------- rendering
    def getArgStr(self) -> str:
        parts = []
        if self.dst is not None:
            ds = self.dst.toString() if hasattr(self.dst, "toString") else str(self.dst)
            if ds:
                parts.append(ds)
        if self.dst1 is not None:
            ds1 = self.dst1.toString() if hasattr(self.dst1, "toString") else str(self.dst1)
            if ds1:
                parts.append(ds1)
        for s in self.srcs:
            parts.append(_input_to_str(s))
        return ", ".join(parts)

    def toString(self) -> str:
        # NB: gfx1250 ``s_set_vgpr_msb`` auto-prepend is deliberately
        # omitted -- the stinkytofu left-path handles MSB via
        # ``InsertVgprMsbPass``. See module docstring for the gap.
        kstr = self.preStr() + " " + self.getArgStr()
        if self.dpp is not None and hasattr(self.dpp, "toString"):
            kstr += self.dpp.toString()
        if self.sdwa is not None and hasattr(self.sdwa, "toString"):
            kstr += self.sdwa.toString()
        if self.vop3 is not None and hasattr(self.vop3, "toString"):
            kstr += self.vop3.toString()
        return self.formatWithComment(kstr)

    # ----------------------------------------------------- copy
    def __deepcopy__(self, memo):
        # rocisa CommonInstruction copy ctor clones dst / dst1 and
        # deep-copies each src (Container ones via ``->clone()``, scalars
        # by value). deepcopy gives us that for free.
        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        Instruction.__init__(clone, self.instType, self.comment)
        clone.outputInlineAsm = self.outputInlineAsm
        clone.instStr = self.instStr
        clone.m_memToken = _deepcopy(self.m_memToken, memo) if self.m_memToken else None
        clone.dst = _deepcopy(self.dst, memo) if self.dst is not None else None
        clone.dst1 = _deepcopy(self.dst1, memo) if self.dst1 is not None else None
        clone.srcs = [_deepcopy(s, memo) for s in self.srcs]
        clone.dpp = _deepcopy(self.dpp, memo) if self.dpp is not None else None
        clone.sdwa = _deepcopy(self.sdwa, memo) if self.sdwa is not None else None
        clone.vop3 = _deepcopy(self.vop3, memo) if self.vop3 is not None else None
        # Dynamic factory subclasses (e.g. DS store/load, reg-jump) have no
        # __slots__, so extra state such as ``ds`` / ``calleeFuncs`` lives in
        # the instance __dict__. __new__ does not carry it over, so replicate
        # any such attributes here to keep clones faithful (rocisa's copy ctor
        # copies the whole concrete object).
        extra = getattr(self, "__dict__", None)
        if extra:
            for key, value in extra.items():
                try:
                    setattr(clone, key, _deepcopy(value, memo))
                except Exception:  # noqa: BLE001 (match nanobind tolerance)
                    setattr(clone, key, value)
        return clone


# ``CompositeInstruction`` is still on the deferred list (no concrete
# subclass needs it yet). ``MacroInstruction`` is real -- see below.
CompositeInstruction = make_dummy_class(f"{_P}.CompositeInstruction")


class MacroInstruction(Instruction):
    """Macro-call leaf -- mirror of ``rocisa::MacroInstruction``.

    Emits ``<name> <arg0>, <arg1>, ...`` followed by the standard
    ``Instruction`` comment formatting. ``args`` is read-write (matches
    rocisa's ``def_rw`` binding); ``setSrc`` is the indexed-write helper.

    ``getDstParams`` / ``getSrcParams`` raise ``RuntimeError`` by design --
    macro args have no dst/src split, and rocisa actively throws here so
    callers crash loudly instead of treating raw args as instruction
    operands.
    """

    __slots__ = ("args",)

    def __init__(self, name: str, args: List[Any], comment: str = ""):
        super().__init__(InstType.INST_MACRO, comment)
        # C++ shadows Item::name with its own ``name`` field (Item::name
        # stays ""); Python __slots__ can't shadow, so we overwrite the
        # inherited slot. KernelWriter never findNamedItems by macro name
        # so the divergence is unobservable.
        self.name = name
        self.args: List[Any] = list(args)

    def setSrc(self, idx: int, arg: Any) -> None:
        self.args[idx] = arg

    def getParams(self):
        return self.args

    def getDstParams(self):
        raise RuntimeError(
            "MacroInstruction does not have destination parameters"
        )

    def getSrcParams(self):
        raise RuntimeError(
            "MacroInstruction does not have source parameters"
        )

    def getArgStr(self) -> str:
        # rocisa: " arg0, arg1, ..." (leading space when non-empty);
        # empty args -> "".
        if not self.args:
            return ""
        return " " + ", ".join(_input_to_str(a) for a in self.args)

    def toString(self) -> str:
        return self.formatWithComment(self.name + self.getArgStr())

    def __deepcopy__(self, memo):
        clone = MacroInstruction.__new__(MacroInstruction)
        memo[id(self)] = clone
        Instruction.__init__(clone, self.instType, self.comment)
        clone.outputInlineAsm = self.outputInlineAsm
        clone.instStr = self.instStr
        clone.m_memToken = _deepcopy(self.m_memToken, memo) if self.m_memToken else None
        clone.name = self.name
        # Containers deep-clone via their own __deepcopy__; primitives
        # (int/float/str) round-trip identity-equal, matching C++ visit.
        clone.args = [_deepcopy(a, memo) for a in self.args]
        return clone

    def clone(self):
        # rocisa Instruction::clone() returns a new shared_ptr; we mirror
        # by deepcopy with a fresh memo so it behaves like a standalone
        # copy (no memo sharing with outer deepcopy contexts).
        return _deepcopy(self, {})


# ==========================================================================
# logicalIR coercion -- rocisa operand -> stinkytofu Register.
# ==========================================================================
#
# rocisa stores instruction operands as a std::variant<Container, int,
# double, string>. Stinkytofu's ``_stinkytofu.Register`` has matching
# constructors for the literal cases. RegisterContainer already knows
# how to build a stinky.Register via ``to_stinky()`` (see
# ``rocisa_stinkytofu_adaptor.container.RegisterContainer.to_stinky``).
#
# String operands deserve a note: KernelWriter sometimes passes raw asm
# text like ``"0x0"`` or ``"s[\\sgprOffset0]"`` as a src. We round-trip
# both through ``Register(literal_string)`` -- the stinkytofu emit path
# prints literal strings verbatim, so ``Register("0x0")`` produces the
# same byte sequence on output as the rocisa right-path would. The
# ``"s[\\..." form is a TODO: stinkytofu has no analogous shorthand and
# the right behaviour is to lower the macro register reference into a
# concrete sgpr index ahead of the stinky call; for Step 3 we punt on
# this case and let the literal-string fallthrough emit it untouched,
# which is byte-correct for inspection but not semantically meaningful
# until the macro-expansion pass lands.
def _to_stinky_register(arg: Any) -> Any:
    """Convert a rocisa-side instruction operand to a stinkytofu Register.

    Accepted inputs (matches ``InstructionInput`` variants):
      * RegisterContainer (or subclass) -- delegate to ``.to_stinky()``.
      * int / float -- wraps as a numeric literal Register.
      * str -- wraps as a literal-string Register (verbatim emit).

    Raises:
        ImportError: when the ``_stinkytofu.so`` binding is missing.
        TypeError: on an unsupported operand type.
    """
    import stinkytofu as _st  # noqa: WPS433  (runtime: optional dep)
    from .container import Container as _Container  # noqa: WPS433

    if hasattr(arg, "to_stinky"):
        return arg.to_stinky()
    if isinstance(arg, bool):
        return _st.Register(int(arg))
    if isinstance(arg, int):
        return _st.Register(arg)
    if isinstance(arg, float):
        return _st.Register(arg)
    if isinstance(arg, str):
        return _st.Register(arg)
    if isinstance(arg, _Container):
        return _st.Register(arg.toString())
    raise TypeError(
        f"rocisa_stinkytofu_adaptor.instruction: cannot coerce operand of "
        f"type {type(arg).__name__!r} into a stinkytofu Register. Supported "
        f"types are RegisterContainer (.to_stinky()), int, float, str, Container."
    )


# gfx12+ style suffix on ``s_load_*`` (matches rocisa ``ReadWriteInstruction``
# ``typeConvert`` for ``RW_TYPE0`` when ISA major >= 11).
_SMEM_LOAD_TYPE_SUFFIX = {
    InstType.INST_B32: "b32",
    InstType.INST_B64: "b64",
    InstType.INST_B128: "b128",
    InstType.INST_B256: "b256",
    InstType.INST_B512: "b512",
}


def _smem_load_type_suffix(inst_type: Any) -> str:
    try:
        return _SMEM_LOAD_TYPE_SUFFIX[inst_type]
    except KeyError as e:
        raise ValueError(f"unsupported InstType for s_load: {inst_type!r}") from e


_ST_SLOAD_LOGICAL_BY_INST_TYPE = {
    InstType.INST_B32: "SLoadB32",
    InstType.INST_B64: "SLoadB64",
    InstType.INST_B128: "SLoadB128",
    InstType.INST_B256: "SLoadB256",
    InstType.INST_B512: "SLoadB512",
}


# ==========================================================================
# VMovB32 -- first real instruction shim (Step-3 vertical slice).
# ==========================================================================
#
# source: rocisa/rocisa/include/instruction/common.hpp:5054-5076 (struct)
#         rocisa/rocisa/src/instruction/common.cpp:1930-1942 (binding)
# logicalIR counterpart:
#         shared/stinkytofu/src/ir/logical/LogicalInstructionDefs.inc
#         (entry: VMovB32; auto-generated _stinkytofu.VMovB32 takes
#          (dest, src0, comment="") -> LogicalInstruction shared_ptr).
class VMovB32(CommonInstruction):
    """``v_mov_b32 dst, src`` shim with stinkytofu left-path bridge.

    Matches ``rocisa::VMovB32``'s ctor signature byte-for-byte:
    ``VMovB32(dst, src, sdwa=None, comment="", dpp=None)``. Constructed
    instances are slotted into a ``rocisa.code.Module``; when the module
    is lowered via ``Module.to_stinky_asm(arch)`` the
    ``to_stinky_logical()`` method here fabricates a fresh
    ``_stinkytofu.VMovB32`` shared_ptr that the C++ pipeline consumes.

    Notes:
        - dst can be any object exposing ``toString()`` (typically a
          ``RegisterContainer`` from ``rocisa_stinkytofu_adaptor.container``).
        - src is an ``InstructionInput``-shaped value: Container | int |
          float | str. See ``_to_stinky_register`` for the routing table.
        - ``sdwa`` / ``dpp`` are accepted for rocisa-API parity but are
          NOT forwarded to the stinkytofu binding (the auto-generated
          binding doesn't expose them). A non-None modifier today will
          render correctly in rocisa-side ``__str__`` but disappear from
          the lowered asm; document & raise once a KernelWriter caller
          actually needs them.
    """

    def __init__(self, dst: Any, src: Any, sdwa: Any = None,
                 comment: str = "", dpp: Any = None):
        super().__init__(
            instType=InstType.INST_B32,
            dst=dst,
            srcs=[src],
            dpp=dpp,
            sdwa=sdwa,
            vop3=None,
            comment=comment,
        )
        self.setInst("v_mov_b32")

    def to_stinky_logical(self) -> Any:
        """Build a fresh ``_stinkytofu.VMovB32(dst_reg, src_reg, comment)``.

        Called by ``rocisa.code.Module._collect_logical_insts`` during
        ``Module.to_stinky_asm(arch)``. The returned shared_ptr is added
        to a logical IR module and lowered to asm by the C++ pipeline.

        Raises:
            ImportError: when the stinkytofu binding is missing.
            TypeError: when ``dst`` or ``src`` is not a supported type
                (see ``_to_stinky_register``).
        """
        import stinkytofu as _st  # noqa: WPS433  (runtime: optional dep)

        dst_reg = _to_stinky_register(self.dst)
        src_reg = _to_stinky_register(self.srcs[0])
        return _st.VMovB32(dst_reg, src_reg, self.comment)

    def __deepcopy__(self, memo):
        # Inherit CommonInstruction's clone behaviour but ensure
        # ``__init__`` isn't re-invoked (which would re-run setInst etc.).
        clone = CommonInstruction.__deepcopy__(self, memo)
        # ``setInst`` was already called in __init__; the clone's
        # instStr was copied from self in the base __deepcopy__, so we
        # don't need to redo it. Returning ``clone`` directly preserves
        # subclass identity because we used ``self.__class__.__new__``.
        return clone


# ==========================================================================
# VMovRelsD2B32 -- indirect VGPR write via M0 offset (CompactLoopStore).
# ==========================================================================
#
# source: rocisa/rocisa/include/instruction/common.hpp:5584-5604
# ISA mnemonic: v_movrelsd_2_b32
# Note: stinkytofu LogicalIR does not define this instruction, so
# to_stinky_logical() returns None and _populate_logical_module will
# fall back to textblock emission via toString().
class VMovRelsD2B32(CommonInstruction):
    """``v_movrelsd_2_b32 dst, src`` -- indirect VGPR write offset by M0."""

    def __init__(self, dst: Any, src: Any, sdwa: Any = None,
                 comment: str = "", dpp: Any = None):
        super().__init__(
            instType=InstType.INST_B32,
            dst=dst,
            srcs=[src],
            dpp=dpp,
            sdwa=sdwa,
            vop3=None,
            comment=comment,
        )
        self.setInst("v_movrelsd_2_b32")

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433
        dst_reg = _to_stinky_register(self.dst)
        src_reg = _to_stinky_register(self.srcs[0])
        return _st.VMovRelsD2B32(dst_reg, src_reg, self.comment)

    def __deepcopy__(self, memo):
        clone = CommonInstruction.__deepcopy__(self, memo)
        return clone


# ==========================================================================
# VPrngB32 -- pseudo-random number generator (VOP1, 1 src).
# ==========================================================================
class VPrngB32(CommonInstruction):
    """``v_prng_b32 dst, src`` -- PRNG instruction."""

    def __init__(self, dst: Any, src: Any, sdwa: Any = None,
                 comment: str = "", dpp: Any = None):
        super().__init__(
            instType=InstType.INST_B32,
            dst=dst,
            srcs=[src],
            dpp=dpp,
            sdwa=sdwa,
            vop3=None,
            comment=comment,
        )
        self.setInst("v_prng_b32")

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433
        dst_reg = _to_stinky_register(self.dst)
        src_reg = _to_stinky_register(self.srcs[0])
        return _st.VPrngB32(dst_reg, src_reg, comment=self.comment)

    def __deepcopy__(self, memo):
        clone = CommonInstruction.__deepcopy__(self, memo)
        return clone


# ==========================================================================
# SMovB32 / SMovB64 -- scalar moves (Phase A; same pattern as VMovB32).
# ==========================================================================
#
# source: rocisa/rocisa/include/instruction/common.hpp:1076-1117
#         rocisa/rocisa/src/instruction/common.cpp:506-524
# logicalIR: ``SMovB32`` / ``SMovB64`` in LogicalInstructionDefs.inc
#         (generated ``_stinkytofu.SMovB32(dest, src0, comment)``).
class SMovB32(CommonInstruction):
    """``s_mov_b32 dst, src`` shim with stinkytofu left-path bridge."""

    def __init__(self, dst: Any, src: Any, sdwa: Any = None,
                 comment: str = "", dpp: Any = None):
        super().__init__(
            instType=InstType.INST_B32,
            dst=dst,
            srcs=[src],
            dpp=dpp,
            sdwa=sdwa,
            vop3=None,
            comment=comment,
        )
        self.setInst("s_mov_b32")

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        dst_reg = _to_stinky_register(self.dst)
        src_reg = _to_stinky_register(self.srcs[0])
        return _st.SMovB32(dst_reg, src_reg, self.comment)

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)


class SMovB64(CommonInstruction):
    """``s_mov_b64 dst, src`` shim with stinkytofu left-path bridge."""

    def __init__(self, dst: Any, src: Any, sdwa: Any = None,
                 comment: str = "", dpp: Any = None):
        super().__init__(
            instType=InstType.INST_B64,
            dst=dst,
            srcs=[src],
            dpp=dpp,
            sdwa=sdwa,
            vop3=None,
            comment=comment,
        )
        self.setInst("s_mov_b64")

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        dst_reg = _to_stinky_register(self.dst)
        src_reg = _to_stinky_register(self.srcs[0])
        return _st.SMovB64(dst_reg, src_reg, self.comment)

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)


# ==========================================================================
# SNop -- scalar NOP (SOPP wait field)
# ==========================================================================
#
# source: rocisa/rocisa/include/instruction/common.hpp:1750-1794
# logicalIR: ``SNop`` in LogicalInstructionDefs.inc
#         (generated ``_stinkytofu.SNop(src0, comment)`` with ``src0`` = wait
#         count as a literal ``Register``).


class SNop(Instruction):
    """``s_nop <wait>`` shim with stinkytofu left-path bridge."""

    __slots__ = ("wait_state",)

    def __init__(self, waitState: int = 0, comment: str = ""):
        super().__init__(InstType.INST_NOTYPE, comment)
        self.wait_state = int(waitState)
        self.setInst("s_nop")

    def getParams(self):
        return [self.wait_state]

    def getDstParams(self):
        return []

    def getSrcParams(self):
        return [self.wait_state]

    def toString(self) -> str:
        kstr = self.preStr() + " " + _input_to_str(self.wait_state)
        return self.formatWithComment(kstr)

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        return _st.SNop(_to_stinky_register(self.wait_state), self.comment)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = SNop(waitState=self.wait_state, comment=self.comment)
        memo[id(self)] = dup
        return dup


# ==========================================================================
# Scalar ALU -- arithmetic, shift, bitwise (Phase 6 Step 1).
# ==========================================================================
#
# source: rocisa/rocisa/include/instruction/common.hpp
# logicalIR: All have entries in LogicalInstructionDefs.inc (2 srcs, hasDest=true).
#
# Pattern: CommonInstruction(dst, src0, src1, comment) with to_stinky_logical()
# forwarding to the matching _stinkytofu.<ClassName>(dst, src0, src1, comment).


def _make_scalar_alu_class(class_name: str, mnemonic: str, inst_type: "InstType",
                           base: type = None):
    """Factory for scalar/vector ALU instruction shim classes (dst, src0, src1).

    ``base`` selects the concrete parent class so ``isinstance`` checks in
    KernelWriter (e.g. ``VCvtInstruction``) discriminate correctly; it defaults
    to ``CommonInstruction`` to preserve behaviour for plain ALU ops.
    """
    if base is None:
        base = CommonInstruction

    def __init__(self, dst: Any, src0: Any = None, src1: Any = None,
                 *extra: Any, comment: str = "", sdwa: Any = None,
                 dpp: Any = None, vop3: Any = None, sels: Any = None, **kw):
        _ = kw
        # Native rocisa binary-VALU ctors are heterogeneous in their trailing
        # positional slots: some are (dst, src0, src1, sdwa, comment), some
        # (dst, src0, src1, comment), and e.g. VAddF32 is
        # (dst, src0, src1, dpp, sdwa, comment). Positional callers such as
        # Activation.FuseInstruction -- ``type(oldInst)(dst, *srcs, oldInst.sdwa)``
        # -- are written against the *native* signature, so disambiguate any
        # positional extras by type rather than a fixed slot: a str is the
        # comment, any other (modifier) object is the sdwa.
        for a in extra:
            if isinstance(a, str):
                comment = a
            elif a is not None:
                sdwa = a
        # ``sels`` is the byte-select used by SR/pk CVT ctors (e.g.
        # VCvtSRF32toFP8(..., sels=[vi%4])). Native maps it to VOP3P byte_sel
        # (gfx11+/gfx1250 branch); dropping it makes every partial conversion
        # write byte 0, so packed FP8 outputs overwrite each other -> wrong
        # results. Mirror rocisa cvt.hpp: size-1 -> byte_sel=[s0], size-2 ->
        # byte_sel=[s1 + (s0<<1)].
        if sels and vop3 is None:
            from .container import VOP3PModifiers  # noqa: WPS433
            if len(sels) == 1:
                byte_sel = [sels[0]]
            else:
                byte_sel = [sels[1] + (sels[0] << 1)]
            vop3 = VOP3PModifiers(byte_sel=byte_sel)
        CommonInstruction.__init__(
            self,
            instType=inst_type,
            dst=dst,
            srcs=[src0, src1],
            dpp=dpp,
            sdwa=sdwa,
            vop3=vop3,
            comment=comment,
        )
        self.setInst(mnemonic)

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        dst_reg = _to_stinky_register(self.dst)
        src0_reg = _to_stinky_register(self.srcs[0])
        src1_reg = _to_stinky_register(self.srcs[1])
        factory = getattr(_st, class_name)
        inst = factory(dst_reg, src0_reg, src1_reg, comment=self.comment)
        # Propagate VOP3P packed modifiers (op_sel / op_sel_hi / byte_sel).
        # e.g. GlobalWriteBatch emits ``v_pk_mul_f32 ... op_sel_hi:[0,1,1]``
        # for the alpha-scale packed multiply; dropping op_sel_hi yields an
        # "invalid op_sel operand" from the assembler.
        v = getattr(self, "vop3", None)
        if v is not None:
            inst.set_vop3(
                op_sel=list(getattr(v, "op_sel", None) or []),
                op_sel_hi=list(getattr(v, "op_sel_hi", None) or []),
                byte_sel=list(getattr(v, "byte_sel", None) or []),
            )
        return inst

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (base,), {
        "__doc__": f"``{mnemonic} dst, src0, src1`` shim with stinkytofu left-path bridge.",
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
    })
    return cls


def _make_scalar_unary_class(class_name: str, mnemonic: str, inst_type: "InstType",
                             base: type = None):
    """Factory for scalar/vector unary instruction shim classes (dst, src) — 1 source.

    ``base`` selects the concrete parent class (see ``_make_scalar_alu_class``);
    defaults to ``CommonInstruction``.
    """
    if base is None:
        base = CommonInstruction

    def __init__(self, dst: Any, src: Any = None,
                 comment: str = "", sdwa: Any = None, dpp: Any = None, **kw):
        _ = kw
        CommonInstruction.__init__(
            self,
            instType=inst_type,
            dst=dst,
            srcs=[src],
            dpp=dpp,
            sdwa=sdwa,
            vop3=None,
            comment=comment,
        )
        self.setInst(mnemonic)

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        dst_reg = _to_stinky_register(self.dst)
        src_reg = _to_stinky_register(self.srcs[0])
        factory = getattr(_st, class_name)
        inst = factory(dst_reg, src_reg, comment=self.comment)
        if getattr(self, 'vop3', None) is not None:
            inst.set_vop3(op_sel=self.vop3.op_sel)
        return inst

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (base,), {
        "__doc__": f"``{mnemonic} dst, src`` shim with stinkytofu left-path bridge.",
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
    })
    return cls


def _make_no_operand_class(class_name: str, mnemonic: str):
    """Factory for zero-operand instructions (no dst, no srcs): VNop, GlobalInv, etc."""

    def __init__(self, comment: str = "", **kw):
        _ = kw
        Instruction.__init__(self, InstType.INST_NOTYPE, comment)
        self.setInst(mnemonic)

    def getParams(self):
        return []

    def getDstParams(self):
        return []

    def getSrcParams(self):
        return []

    def toString(self) -> str:
        return self.formatWithComment(self.instStr)

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        factory = getattr(_st, class_name)
        return factory(self.comment)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = object.__new__(type(self))
        memo[id(self)] = dup
        Instruction.__init__(dup, InstType.INST_NOTYPE, self.comment)
        dup.setInst(mnemonic)
        return dup

    cls = type(class_name, (Instruction,), {
        "__doc__": f"``{mnemonic}`` shim with stinkytofu left-path bridge.",
        "__init__": __init__,
        "__slots__": (),
        "getParams": getParams,
        "getDstParams": getDstParams,
        "getSrcParams": getSrcParams,
        "toString": toString,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
    })
    cls.__qualname__ = class_name
    cls.__module__ = __name__
    return cls


def _make_imm_no_dest_class(class_name: str, mnemonic: str, param_name: str = "simm16"):
    """Factory for 1-imm, no-dest instructions: SSleep, SSetPrior, SDelayAlu, etc."""

    def __init__(self, value: int = 0, comment: str = "", **kw):
        _ = kw
        Instruction.__init__(self, InstType.INST_NOTYPE, comment)
        self._imm_value = int(value)
        self.setInst(mnemonic)

    def getParams(self):
        return [self._imm_value]

    def getDstParams(self):
        return []

    def getSrcParams(self):
        return [self._imm_value]

    def toString(self) -> str:
        kstr = self.instStr + " " + _input_to_str(self._imm_value)
        return self.formatWithComment(kstr)

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        factory = getattr(_st, class_name)
        return factory(_to_stinky_register(self._imm_value), self.comment)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = object.__new__(type(self))
        memo[id(self)] = dup
        Instruction.__init__(dup, InstType.INST_NOTYPE, self.comment)
        dup._imm_value = self._imm_value
        dup.setInst(mnemonic)
        return dup

    cls = type(class_name, (Instruction,), {
        "__doc__": f"``{mnemonic} {{imm}}`` shim with stinkytofu left-path bridge.",
        "__init__": __init__,
        "__slots__": ("_imm_value",),
        "getParams": getParams,
        "getDstParams": getDstParams,
        "getSrcParams": getSrcParams,
        "toString": toString,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
    })
    cls.__qualname__ = class_name
    cls.__module__ = __name__
    return cls


def _make_reg_jump_class(class_name: str, mnemonic: str, has_dest: bool = False):
    """Factory for register-indirect jump/call: SSetPCB64, SSwapPCB64."""

    if has_dest:
        def __init__(self, dst: Any = None, src: Any = None, comment: str = "", **kw):
            _ = kw
            Instruction.__init__(self, InstType.INST_NOTYPE, comment)
            self.dst = dst
            self.src = src
            # Mirror native rocisa SSwapPCB64::calleeFuncs (std::vector<std::string>):
            # candidate callee names for consumers modelling s_swappc_b64 as a
            # function call. Metadata only; does not affect emitted assembly.
            self.calleeFuncs = []
            self.setInst(mnemonic)

        def getParams(self):
            return [self.dst, self.src]

        def getDstParams(self):
            return [self.dst]

        def getSrcParams(self):
            return [self.src]

        def toString(self) -> str:
            kstr = self.instStr + " " + _input_to_str(self.dst) + ", " + _input_to_str(self.src)
            return self.formatWithComment(kstr)

        def to_stinky_logical(self) -> Any:
            import stinkytofu as _st  # noqa: WPS433

            factory = getattr(_st, class_name)
            return factory(
                _to_stinky_register(self.dst),
                _to_stinky_register(self.src),
                self.comment,
            )

        def __deepcopy__(self, memo):
            if id(self) in memo:
                return memo[id(self)]
            dup = object.__new__(type(self))
            memo[id(self)] = dup
            Instruction.__init__(dup, InstType.INST_NOTYPE, self.comment)
            dup.dst = copy.deepcopy(self.dst, memo)
            dup.src = copy.deepcopy(self.src, memo)
            dup.calleeFuncs = list(getattr(self, "calleeFuncs", []))
            dup.setInst(mnemonic)
            return dup

        slots = ("dst", "src", "calleeFuncs")
    else:
        def __init__(self, src: Any = None, comment: str = "", **kw):
            _ = kw
            Instruction.__init__(self, InstType.INST_NOTYPE, comment)
            self.src = src
            self.setInst(mnemonic)

        def getParams(self):
            return [self.src]

        def getDstParams(self):
            return []

        def getSrcParams(self):
            return [self.src]

        def toString(self) -> str:
            kstr = self.instStr + " " + _input_to_str(self.src)
            return self.formatWithComment(kstr)

        def to_stinky_logical(self) -> Any:
            import stinkytofu as _st  # noqa: WPS433

            factory = getattr(_st, class_name)
            return factory(_to_stinky_register(self.src), self.comment)

        def __deepcopy__(self, memo):
            if id(self) in memo:
                return memo[id(self)]
            dup = object.__new__(type(self))
            memo[id(self)] = dup
            Instruction.__init__(dup, InstType.INST_NOTYPE, self.comment)
            dup.src = copy.deepcopy(self.src, memo)
            dup.setInst(mnemonic)
            return dup

        slots = ("src",)

    cls = type(class_name, (Instruction,), {
        "__doc__": f"``{mnemonic}`` shim with stinkytofu left-path bridge.",
        "__init__": __init__,
        "__slots__": slots,
        "getParams": getParams,
        "getDstParams": getDstParams,
        "getSrcParams": getSrcParams,
        "toString": toString,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
    })
    cls.__qualname__ = class_name
    cls.__module__ = __name__
    return cls


def _make_zero_src_class(class_name: str, mnemonic: str, inst_type: "InstType"):
    """Factory for zero-source instruction shim classes (dst only, no srcs)."""

    def __init__(self, dst: Any, comment: str = "", **kw):
        _ = kw
        CommonInstruction.__init__(
            self,
            instType=inst_type,
            dst=dst,
            srcs=[],
            dpp=None,
            sdwa=None,
            vop3=None,
            comment=comment,
        )
        self.setInst(mnemonic)

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        dst_reg = _to_stinky_register(self.dst)
        factory = getattr(_st, class_name)
        return factory(dst_reg, comment=self.comment)

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (CommonInstruction,), {
        "__doc__": f"``{mnemonic} dst`` shim with stinkytofu left-path bridge.",
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
    })
    return cls


# -- Scalar Arithmetic --
# logicalIR: SAddI32
SAddI32 = _make_scalar_alu_class("SAddI32", "s_add_i32", InstType.INST_I32)
# logicalIR: SAddU32
SAddU32 = _make_scalar_alu_class("SAddU32", "s_add_u32", InstType.INST_U32)
# logicalIR: SAddCU32
SAddCU32 = _make_scalar_alu_class("SAddCU32", "s_addc_u32", InstType.INST_U32)
# logicalIR: SMulI32
SMulI32 = _make_scalar_alu_class("SMulI32", "s_mul_i32", InstType.INST_I32)
# logicalIR: SMulHII32
SMulHII32 = _make_scalar_alu_class("SMulHII32", "s_mul_hi_i32", InstType.INST_I32)
# logicalIR: SMulHIU32
SMulHIU32 = _make_scalar_alu_class("SMulHIU32", "s_mul_hi_u32", InstType.INST_U32)
# logicalIR: SMulLOU32
SMulLOU32 = _make_scalar_alu_class("SMulLOU32", "s_mul_lo_u32", InstType.INST_U32)
# logicalIR: SSubI32
SSubI32 = _make_scalar_alu_class("SSubI32", "s_sub_i32", InstType.INST_I32)
# logicalIR: SSubU32
SSubU32 = _make_scalar_alu_class("SSubU32", "s_sub_u32", InstType.INST_U32)
# logicalIR: SSubBU32
SSubBU32 = _make_scalar_alu_class("SSubBU32", "s_subb_u32", InstType.INST_U32)

# -- Scalar Shift --
# NOTE: Native rocisa uses (dst, shiftHex, src) API where shiftHex is the shift
# amount and src is the value.  Internally srcs=[src, shiftHex] (value first).
# Logical IR factory expects (dst, src0=value, src1=shiftAmount).


def _make_scalar_shift_class(class_name: str, mnemonic: str, inst_type: "InstType"):
    """Factory for scalar shift instruction shims matching rocisa (dst, shiftHex, src) API."""

    def __init__(self, dst: Any, shiftHex: Any = None, src: Any = None,
                 comment: str = "",
                 src0: Any = None, src1: Any = None,
                 sdwa: Any = None, dpp: Any = None, **kw):
        _ = kw
        # Accept either (shiftHex=, src=) or (src0=, src1=) calling conventions.
        if src is not None or shiftHex is not None:
            value = src
            shift_amount = shiftHex
        else:
            value = src0
            shift_amount = src1
        CommonInstruction.__init__(
            self,
            instType=inst_type,
            dst=dst,
            srcs=[value, shift_amount],
            dpp=dpp,
            sdwa=sdwa,
            vop3=None,
            comment=comment,
        )
        self.setInst(mnemonic)

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        dst_reg = _to_stinky_register(self.dst)
        src0_reg = _to_stinky_register(self.srcs[0])  # value
        src1_reg = _to_stinky_register(self.srcs[1])  # shift amount
        factory = getattr(_st, class_name)
        return factory(dst_reg, src0_reg, src1_reg, comment=self.comment)

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (CommonInstruction,), {
        "__doc__": f"``{mnemonic} dst, src, shiftHex`` shim with stinkytofu left-path bridge.",
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
    })
    return cls


# logicalIR: SLShiftLeftB32
SLShiftLeftB32 = _make_scalar_shift_class("SLShiftLeftB32", "s_lshl_b32", InstType.INST_B32)
# logicalIR: SLShiftRightB32
SLShiftRightB32 = _make_scalar_shift_class("SLShiftRightB32", "s_lshr_b32", InstType.INST_B32)
# logicalIR: SLShiftLeftB64
SLShiftLeftB64 = _make_scalar_shift_class("SLShiftLeftB64", "s_lshl_b64", InstType.INST_B64)
# logicalIR: SLShiftRightB64
SLShiftRightB64 = _make_scalar_shift_class("SLShiftRightB64", "s_lshr_b64", InstType.INST_B64)
# logicalIR: SAShiftRightI32
SAShiftRightI32 = _make_scalar_shift_class("SAShiftRightI32", "s_ashr_i32", InstType.INST_I32)
# logicalIR: SLShiftLeft1AddU32
SLShiftLeft1AddU32 = _make_scalar_alu_class("SLShiftLeft1AddU32", "s_lshl1_add_u32", InstType.INST_U32)
# logicalIR: SLShiftLeft2AddU32
SLShiftLeft2AddU32 = _make_scalar_alu_class("SLShiftLeft2AddU32", "s_lshl2_add_u32", InstType.INST_U32)
# logicalIR: SLShiftLeft3AddU32
SLShiftLeft3AddU32 = _make_scalar_alu_class("SLShiftLeft3AddU32", "s_lshl3_add_u32", InstType.INST_U32)
# logicalIR: SLShiftLeft4AddU32
SLShiftLeft4AddU32 = _make_scalar_alu_class("SLShiftLeft4AddU32", "s_lshl4_add_u32", InstType.INST_U32)

# -- Scalar Bitwise --
# logicalIR: SAndB32
SAndB32 = _make_scalar_alu_class("SAndB32", "s_and_b32", InstType.INST_B32)
# logicalIR: SAndB64
SAndB64 = _make_scalar_alu_class("SAndB64", "s_and_b64", InstType.INST_B64)
# logicalIR: SAndN2B32
SAndN2B32 = _make_scalar_alu_class("SAndN2B32", "s_andn2_b32", InstType.INST_B32)
# logicalIR: SOrB32
SOrB32 = _make_scalar_alu_class("SOrB32", "s_or_b32", InstType.INST_B32)
# logicalIR: SOrB64
SOrB64 = _make_scalar_alu_class("SOrB64", "s_or_b64", InstType.INST_B64)
# logicalIR: SXorB32
SXorB32 = _make_scalar_alu_class("SXorB32", "s_xor_b32", InstType.INST_B32)
# logicalIR: SAndSaveExecB32 (1 src)
SAndSaveExecB32 = _make_scalar_unary_class("SAndSaveExecB32", "s_and_saveexec_b32", InstType.INST_B32)
# logicalIR: SAndSaveExecB64 (1 src)
SAndSaveExecB64 = _make_scalar_unary_class("SAndSaveExecB64", "s_and_saveexec_b64", InstType.INST_B64)
# logicalIR: SOrSaveExecB32 (1 src)
SOrSaveExecB32 = _make_scalar_unary_class("SOrSaveExecB32", "s_or_saveexec_b32", InstType.INST_B32)
# logicalIR: SOrSaveExecB64 (1 src)
SOrSaveExecB64 = _make_scalar_unary_class("SOrSaveExecB64", "s_or_saveexec_b64", InstType.INST_B64)


# ==========================================================================
# Vector ALU -- arithmetic, shift, bitwise, other (Phase 6 Step 4).
# ==========================================================================
#
# source: rocisa/rocisa/include/instruction/common.hpp
# logicalIR: All have entries in LogicalInstructionDefs.inc.
#
# Pattern: reuse the scalar factories where the API shape is identical
# (dst, src0, src1) or create specialized ones for ternary / vector-shift.


def _make_ternary_class(class_name: str, mnemonic: str, inst_type: "InstType",
                        shift_position: int = 0):
    """Factory for ternary instruction shim classes (dst, src0, src1, src2).

    Also accepts native rocisa shift-ternary API ``(dst, shiftHex, src0, src1)``
    where ``shiftHex`` is placed at ISA operand position ``shift_position``:
      - 0: srcs = [shiftHex, src0, src1]  (default, legacy)
      - 1: srcs = [src0, shiftHex, src1]  (v_lshl_add, v_lshl_or)
      - 2: srcs = [src0, src1, shiftHex]  (v_add_lshl)
    """

    def __init__(self, dst: Any, src0: Any = None, src1: Any = None,
                 src2: Any = None, shiftHex: Any = None,
                 comment: str = "", sdwa: Any = None, dpp: Any = None,
                 vop3: Any = None, **kw):
        _ = kw
        if shiftHex is not None:
            if shift_position == 1:
                s0, s1, s2 = src0, shiftHex, src1
            elif shift_position == 2:
                s0, s1, s2 = src0, src1, shiftHex
            else:
                s0, s1, s2 = shiftHex, src0, src1
        else:
            s0, s1, s2 = src0, src1, src2
        CommonInstruction.__init__(
            self,
            instType=inst_type,
            dst=dst,
            srcs=[s0, s1, s2],
            dpp=dpp,
            sdwa=sdwa,
            vop3=vop3,
            comment=comment,
        )
        self.setInst(mnemonic)

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        dst_reg = _to_stinky_register(self.dst)
        src0_reg = _to_stinky_register(self.srcs[0])
        src1_reg = _to_stinky_register(self.srcs[1])
        src2_reg = _to_stinky_register(self.srcs[2])
        factory = getattr(_st, class_name)
        inst = factory(dst_reg, src0_reg, src1_reg, src2_reg, comment=self.comment)
        # Propagate VOP3P packed modifiers (op_sel / op_sel_hi / byte_sel).
        # e.g. GlobalWriteBatch emits ``v_fma_mix_f32 ... op_sel:[..] op_sel_hi:[0,1,0]``
        # for the beta-scale mix-precision FMA; dropping op_sel_hi makes the
        # kernel read the wrong fp16 half and produces incorrect results.
        v = getattr(self, "vop3", None)
        if v is not None:
            inst.set_vop3(
                op_sel=list(getattr(v, "op_sel", None) or []),
                op_sel_hi=list(getattr(v, "op_sel_hi", None) or []),
                byte_sel=list(getattr(v, "byte_sel", None) or []),
            )
        return inst

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (CommonInstruction,), {
        "__doc__": f"``{mnemonic} dst, src0, src1, src2`` shim with stinkytofu left-path bridge.",
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
    })
    return cls


def _make_vector_shift_class(class_name: str, mnemonic: str, inst_type: "InstType"):
    """Factory for vector shift instruction shims matching rocisa (dst, shiftHex, src) API.

    Unlike scalar shifts (which internally reorder srcs to [value, shift]), vector
    shifts store srcs in the native order [shiftHex, src] and emit as-is.  The
    stinkytofu factory also expects (dst, src0=shift, src1=value, comment).
    """

    def __init__(self, dst: Any, shiftHex: Any = None, src: Any = None,
                 comment: str = "",
                 src0: Any = None, src1: Any = None,
                 sdwa: Any = None, dpp: Any = None, **kw):
        _ = kw
        if shiftHex is not None or src is not None:
            s0 = shiftHex
            s1 = src
        else:
            s0 = src0
            s1 = src1
        CommonInstruction.__init__(
            self,
            instType=inst_type,
            dst=dst,
            srcs=[s0, s1],
            dpp=dpp,
            sdwa=sdwa,
            vop3=None,
            comment=comment,
        )
        self.setInst(mnemonic)

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        dst_reg = _to_stinky_register(self.dst)
        src0_reg = _to_stinky_register(self.srcs[0])
        src1_reg = _to_stinky_register(self.srcs[1])
        factory = getattr(_st, class_name)
        return factory(dst_reg, src0_reg, src1_reg, comment=self.comment)

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (CommonInstruction,), {
        "__doc__": f"``{mnemonic} dst, shiftHex, src`` shim with stinkytofu left-path bridge.",
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
    })
    return cls


# -- Vector Arithmetic (binary: dst, src0, src1) --
# logicalIR: VAddU32
VAddU32 = _make_scalar_alu_class("VAddU32", "v_add_nc_u32", InstType.INST_U32)
# logicalIR: VAddF32
VAddF32 = _make_scalar_alu_class("VAddF32", "v_add_f32", InstType.INST_F32)
# logicalIR: VSubF32
VSubF32 = _make_scalar_alu_class("VSubF32", "v_sub_f32", InstType.INST_F32)
# logicalIR: VSubI32
VSubI32 = _make_scalar_alu_class("VSubI32", "v_sub_nc_i32", InstType.INST_I32)
# logicalIR: VSubU32
VSubU32 = _make_scalar_alu_class("VSubU32", "v_sub_nc_u32", InstType.INST_U32)
# logicalIR: VMulF32
VMulF32 = _make_scalar_alu_class("VMulF32", "v_mul_f32", InstType.INST_F32)
# logicalIR: VMulLOU32
VMulLOU32 = _make_scalar_alu_class("VMulLOU32", "v_mul_lo_u32", InstType.INST_LO_U32)
# logicalIR: VMulHIU32
VMulHIU32 = _make_scalar_alu_class("VMulHIU32", "v_mul_hi_u32", InstType.INST_HI_U32)
# logicalIR: VMulHII32
VMulHII32 = _make_scalar_alu_class("VMulHII32", "v_mul_hi_i32", InstType.INST_HI_I32)
# logicalIR: VMulI32I24
VMulI32I24 = _make_scalar_alu_class("VMulI32I24", "v_mul_i32_i24", InstType.INST_I32)
# logicalIR: VMulU32U24
VMulU32U24 = _make_scalar_alu_class("VMulU32U24", "v_mul_u32_u24", InstType.INST_U32)
# logicalIR: VAndB32
VAndB32 = _make_scalar_alu_class("VAndB32", "v_and_b32", InstType.INST_B32)
# logicalIR: VOrB32
VOrB32 = _make_scalar_alu_class("VOrB32", "v_or_b32", InstType.INST_B32)
# logicalIR: VXorB32
VXorB32 = _make_scalar_alu_class("VXorB32", "v_xor_b32", InstType.INST_B32)

# -- Vector Shift (dst, shiftHex, src) → rev mnemonic --
# logicalIR: VLShiftLeftB32
VLShiftLeftB32 = _make_vector_shift_class("VLShiftLeftB32", "v_lshlrev_b32", InstType.INST_B32)
# logicalIR: VLShiftRightB32
VLShiftRightB32 = _make_vector_shift_class("VLShiftRightB32", "v_lshrrev_b32", InstType.INST_B32)
# logicalIR: VLShiftLeftB64
VLShiftLeftB64 = _make_vector_shift_class("VLShiftLeftB64", "v_lshlrev_b64", InstType.INST_B64)
# logicalIR: VLShiftRightB64
VLShiftRightB64 = _make_vector_shift_class("VLShiftRightB64", "v_lshrrev_b64", InstType.INST_B64)

# -- Vector Ternary (dst, src0, src1, src2) --
# logicalIR: VFmaF32
VFmaF32 = _make_ternary_class("VFmaF32", "v_fma_f32", InstType.INST_F32)
# logicalIR: VFmaMixF32
VFmaMixF32 = _make_ternary_class("VFmaMixF32", "v_fma_mix_f32", InstType.INST_F32)
# logicalIR: VAndOrB32
VAndOrB32 = _make_ternary_class("VAndOrB32", "v_and_or_b32", InstType.INST_B32)

# -- VCndMaskB32 (native: dst, src0, src1, src2=VCC; logical IR: 2 srcs) --
# logicalIR: VCndMaskB32


class VCndMaskB32(CommonInstruction):
    """``v_cndmask_b32 dst, src0, src1, vcc`` shim with stinkytofu left-path bridge.

    Native rocisa takes (dst, src0, src1, src2=VCC). The logical IR version has
    only 2 sources (the VCC mask is implicit); ``to_stinky_logical`` forwards
    only (dst, src0, src1).
    """

    def __init__(self, dst: Any, src0: Any = None, src1: Any = None,
                 src2: Any = None,
                 sdwa: Any = None, comment: str = "", dpp: Any = None, **kw):
        _ = kw
        srcs = [src0, src1]
        if src2 is not None:
            srcs.append(src2)
        super().__init__(
            instType=InstType.INST_B32,
            dst=dst,
            srcs=srcs,
            dpp=dpp,
            sdwa=sdwa,
            vop3=None,
            comment=comment,
        )
        self.setInst("v_cndmask_b32")

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        dst_reg = _to_stinky_register(self.dst)
        src0_reg = _to_stinky_register(self.srcs[0])
        src1_reg = _to_stinky_register(self.srcs[1])
        src2_reg = _to_stinky_register(self.srcs[2]) if len(self.srcs) > 2 else _st.Register("vcc_lo")
        return _st.VCndMaskB32(dst_reg, src0_reg, src1_reg, src2_reg, comment=self.comment)

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)


# -- VReadfirstlaneB32 (unary: dst, src) --
# logicalIR: VReadfirstlaneB32
VReadfirstlaneB32 = _make_scalar_unary_class("VReadfirstlaneB32", "v_readfirstlane_b32", InstType.INST_B32)


# ==========================================================================
# SBarrier -- workgroup barrier (Phase 6 Step 2).
# ==========================================================================
#
# source: rocisa/rocisa/include/instruction/common.hpp:1470-1548
# logicalIR: ``SBarrier`` in LogicalInstructionDefs.inc (0 srcs, no dest).
#
# Native rocisa SBarrier has complex logic (separate signal/wait, cluster
# barrier) gated on HasNewBarrier/HasClusterBarrier caps.  For gfx1250 the
# logical IR lowering pass handles the arch-specific emit; we only bridge
# the constructor parameters through.


class SBarrier(Instruction):
    """``s_barrier`` shim with stinkytofu left-path bridge."""

    __slots__ = ("separate", "wait_flag", "cluster_barrier")

    def __init__(self, separate: bool = False, wait: bool = False,
                 clusterBarrier: bool = False, comment: str = ""):
        super().__init__(InstType.INST_NOTYPE, comment)
        self.separate = bool(separate)
        self.wait_flag = bool(wait)
        self.cluster_barrier = bool(clusterBarrier)
        self.setInst("s_barrier")

    def getParams(self):
        return []

    def getDstParams(self):
        return []

    def getSrcParams(self):
        return []

    def toString(self) -> str:
        return self.formatWithComment(self.instStr)

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        return _st.SBarrier(self.comment)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = SBarrier(
            separate=self.separate, wait=self.wait_flag,
            clusterBarrier=self.cluster_barrier, comment=self.comment,
        )
        memo[id(self)] = dup
        return dup


# ==========================================================================
# SGetRegB32 / SSetRegB32 / SSetRegIMM32B32 -- HW register access (Step 2).
# ==========================================================================
#
# source: rocisa/rocisa/include/instruction/common.hpp:1990-2054
# logicalIR: 1 src, hasDest=true, "Scalar Control"

# logicalIR: SGetRegB32
SGetRegB32 = _make_scalar_unary_class("SGetRegB32", "s_getreg_b32", InstType.INST_B32)
# logicalIR: SSetRegB32
SSetRegB32 = _make_scalar_unary_class("SSetRegB32", "s_setreg_b32", InstType.INST_B32)
# logicalIR: SSetRegIMM32B32
SSetRegIMM32B32 = _make_scalar_unary_class("SSetRegIMM32B32", "s_setreg_IMM32_b32", InstType.INST_B32)


# ==========================================================================
# SMemLoadInstruction / SLoadB* -- scalar memory loads (Phase A).
# ==========================================================================
#
# source: rocisa/rocisa/include/instruction/mem.hpp:514-573
# logicalIR: ``SLoadB32`` … ``SLoadB512`` in LogicalInstructionDefs.inc
#         (auto-generated ``_stinkytofu.SLoadB32(dest, src0, src1, comment)``).
#
# ``SMEMModifiers`` on the logical-IR Python binding is not wired yet;
# ``to_stinky_logical`` raises if ``smem`` is not ``None``.


class SMemLoadInstruction(Instruction):
    """``s_load_<b*> dst, base, soffset`` base (rocisa ``SMemLoadInstruction``)."""

    __slots__ = ("dst", "base", "soffset", "smem")

    def __init__(
        self,
        inst_type: Any,
        dst: Any = None,
        base: Any = None,
        soffset: Any = None,
        smem: Any = None,
        comment: str = "",
        **kwargs: Any,
    ):
        _ = kwargs  # rocisa binding may forward ignored kwargs
        super().__init__(inst_type, comment)
        self.dst = dst
        self.base = base
        self.soffset = soffset
        self.smem = smem
        self.setInst("s_load_")

    def preStr(self) -> str:
        return self.instStr + _smem_load_type_suffix(self.instType)

    def getArgStr(self) -> str:
        parts: List[str] = []
        if self.dst is not None:
            parts.append(
                self.dst.toString() if hasattr(self.dst, "toString") else str(self.dst))
        if self.base is not None:
            parts.append(
                self.base.toString() if hasattr(self.base, "toString") else str(self.base))
        parts.append(_input_to_str(self.soffset))
        return ", ".join(parts)

    def toString(self) -> str:
        kstr = self.preStr() + " " + self.getArgStr()
        if self.smem is not None and hasattr(self.smem, "toString"):
            kstr += self.smem.toString()
        return self.formatWithComment(kstr)

    def getParams(self):
        return [self.dst, self.base, self.soffset]

    def getDstParams(self):
        return [self.dst] if self.dst is not None else []

    def getSrcParams(self):
        return [self.base, self.soffset]

    def to_stinky_logical(self) -> Any:
        if self.smem is not None:
            raise NotImplementedError(
                "rocisa_stinkytofu_adaptor: SMemLoad with non-None SMEMModifiers "
                "is not yet supported on the stinkytofu logical-IR path.",
            )
        import stinkytofu as _st  # noqa: WPS433

        try:
            fac_name = _ST_SLOAD_LOGICAL_BY_INST_TYPE[self.instType]
        except KeyError as e:
            raise ValueError(f"SMemLoad: unsupported instType {self.instType!r}") from e
        factory = getattr(_st, fac_name, None)
        if factory is None:
            raise ImportError(
                f"stinkytofu binding has no factory {fac_name!r}; rebuild stinkytofu "
                f"(tablegen) after adding SLoad* to LogicalInstructionDefs.inc.",
            )
        return factory(
            _to_stinky_register(self.dst),
            _to_stinky_register(self.base),
            _to_stinky_register(self.soffset),
            self.comment,
        )

    def __deepcopy__(self, memo):
        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        Instruction.__init__(clone, self.instType, self.comment)
        clone.outputInlineAsm = self.outputInlineAsm
        clone.instStr = self.instStr
        clone.m_memToken = (
            _deepcopy(self.m_memToken, memo) if self.m_memToken is not None else None
        )
        clone.dst = _deepcopy(self.dst, memo) if self.dst is not None else None
        clone.base = _deepcopy(self.base, memo) if self.base is not None else None
        if isinstance(self.soffset, (int, float, str, bool)):
            clone.soffset = self.soffset
        else:
            clone.soffset = _deepcopy(self.soffset, memo)
        clone.smem = _deepcopy(self.smem, memo) if self.smem is not None else None
        return clone


class SLoadB32(SMemLoadInstruction):
    """``s_load_b32`` shim."""

    def __init__(
        self,
        dst: Any = None,
        base: Any = None,
        soffset: Any = None,
        smem: Any = None,
        comment: str = "",
        **kwargs: Any,
    ):
        super().__init__(InstType.INST_B32, dst, base, soffset, smem, comment, **kwargs)


class SLoadB64(SMemLoadInstruction):
    """``s_load_b64`` shim."""

    def __init__(
        self,
        dst: Any = None,
        base: Any = None,
        soffset: Any = None,
        smem: Any = None,
        comment: str = "",
        **kwargs: Any,
    ):
        super().__init__(InstType.INST_B64, dst, base, soffset, smem, comment, **kwargs)


class SLoadB128(SMemLoadInstruction):
    """``s_load_b128`` shim."""

    def __init__(
        self,
        dst: Any = None,
        base: Any = None,
        soffset: Any = None,
        smem: Any = None,
        comment: str = "",
        **kwargs: Any,
    ):
        super().__init__(InstType.INST_B128, dst, base, soffset, smem, comment, **kwargs)


class SLoadB256(SMemLoadInstruction):
    """``s_load_b256`` shim."""

    def __init__(
        self,
        dst: Any = None,
        base: Any = None,
        soffset: Any = None,
        smem: Any = None,
        comment: str = "",
        **kwargs: Any,
    ):
        super().__init__(InstType.INST_B256, dst, base, soffset, smem, comment, **kwargs)


class SLoadB512(SMemLoadInstruction):
    """``s_load_b512`` shim."""

    def __init__(
        self,
        dst: Any = None,
        base: Any = None,
        soffset: Any = None,
        smem: Any = None,
        comment: str = "",
        **kwargs: Any,
    ):
        super().__init__(InstType.INST_B512, dst, base, soffset, smem, comment, **kwargs)


_ST_SSTORE_LOGICAL_BY_INST_TYPE: Dict[Any, str] = {
    InstType.INST_B32: "SStoreB32",
    InstType.INST_B64: "SStoreB64",
    InstType.INST_B128: "SStoreB128",
    InstType.INST_B256: "SStoreB256",
    InstType.INST_B512: "SStoreB512",
}


class SMemStoreInstruction(Instruction):
    """``s_store_<b*> src, base, soffset`` base class."""

    __slots__ = ("src", "base", "soffset", "smem")

    def __init__(
        self,
        inst_type: Any,
        src: Any = None,
        base: Any = None,
        soffset: Any = None,
        smem: Any = None,
        comment: str = "",
        **kwargs: Any,
    ):
        _ = kwargs
        super().__init__(inst_type, comment)
        self.src = src
        self.base = base
        self.soffset = soffset
        self.smem = smem
        self.setInst("s_store_")

    def toString(self) -> str:
        parts: List[str] = []
        if self.src is not None:
            parts.append(
                self.src.toString() if hasattr(self.src, "toString") else str(self.src))
        if self.base is not None:
            parts.append(
                self.base.toString() if hasattr(self.base, "toString") else str(self.base))
        parts.append(_input_to_str(self.soffset))
        kstr = self.instStr + _smem_load_type_suffix(self.instType) + " " + ", ".join(parts)
        if self.smem is not None and hasattr(self.smem, "toString"):
            kstr += self.smem.toString()
        return self.formatWithComment(kstr)

    def getParams(self):
        return [self.src, self.base, self.soffset]

    def getDstParams(self):
        return []

    def getSrcParams(self):
        return [self.src, self.base, self.soffset]

    def to_stinky_logical(self) -> Any:
        if self.smem is not None:
            raise NotImplementedError(
                "rocisa_stinkytofu_adaptor: SMemStore with non-None SMEMModifiers "
                "is not yet supported on the stinkytofu logical-IR path.",
            )
        import stinkytofu as _st  # noqa: WPS433

        fac_name = _ST_SSTORE_LOGICAL_BY_INST_TYPE.get(self.instType)
        if fac_name is None:
            raise ValueError(f"SMemStore: unsupported instType {self.instType!r}")
        factory = getattr(_st, fac_name)
        return factory(
            _to_stinky_register(self.src),
            _to_stinky_register(self.base),
            _to_stinky_register(self.soffset),
            self.comment,
        )

    def __deepcopy__(self, memo):
        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        Instruction.__init__(clone, self.instType, self.comment)
        clone.outputInlineAsm = self.outputInlineAsm
        clone.instStr = self.instStr
        clone.m_memToken = (
            _deepcopy(self.m_memToken, memo) if self.m_memToken is not None else None
        )
        clone.src = _deepcopy(self.src, memo) if self.src is not None else None
        clone.base = _deepcopy(self.base, memo) if self.base is not None else None
        if isinstance(self.soffset, (int, float, str, bool)):
            clone.soffset = self.soffset
        else:
            clone.soffset = _deepcopy(self.soffset, memo)
        clone.smem = _deepcopy(self.smem, memo) if self.smem is not None else None
        return clone


class SStoreB32(SMemStoreInstruction):
    def __init__(self, src=None, base=None, soffset=None, smem=None, comment="", **kw):
        super().__init__(InstType.INST_B32, src, base, soffset, smem, comment, **kw)


class SStoreB64(SMemStoreInstruction):
    def __init__(self, src=None, base=None, soffset=None, smem=None, comment="", **kw):
        super().__init__(InstType.INST_B64, src, base, soffset, smem, comment, **kw)


class SStoreB128(SMemStoreInstruction):
    def __init__(self, src=None, base=None, soffset=None, smem=None, comment="", **kw):
        super().__init__(InstType.INST_B128, src, base, soffset, smem, comment, **kw)


class SStoreB256(SMemStoreInstruction):
    def __init__(self, src=None, base=None, soffset=None, smem=None, comment="", **kw):
        super().__init__(InstType.INST_B256, src, base, soffset, smem, comment, **kw)


class SStoreB512(SMemStoreInstruction):
    def __init__(self, src=None, base=None, soffset=None, smem=None, comment="", **kw):
        super().__init__(InstType.INST_B512, src, base, soffset, smem, comment, **kw)


# ==========================================================================
# Branch instructions
# source: rocisa/rocisa/src/instruction/branch.cpp
# logicalIR: SBranch, SCBranchSCC0, SCBranchSCC1, SCBranchVCCNZ,
#            SCBranchVCCZ, SCBranchExecZ, SCBranchExecNZ
# ==========================================================================


class BranchInstruction(Instruction):
    """Base class for branch instructions (labelName target)."""

    __slots__ = ("labelName",)

    def __init__(self, labelName: str = "", comment: str = ""):
        super().__init__(InstType.INST_NOTYPE, comment)
        self.labelName = str(labelName)

    def getParams(self):
        return [self.labelName]

    def getDstParams(self):
        return []

    def getSrcParams(self):
        return [self.labelName]

    def toString(self) -> str:
        return self.formatWithComment(self.instStr + " " + self.labelName)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = self.__class__(labelName=self.labelName, comment=self.comment)
        memo[id(self)] = dup
        return dup


def _make_branch_class(class_name: str, mnemonic: str):
    """Factory for branch instruction adaptor classes."""

    def _init(self, labelName: str = "", comment: str = ""):
        BranchInstruction.__init__(self, labelName, comment)
        self.setInst(mnemonic)

    def _to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        factory = getattr(_st, class_name)
        return factory(_to_stinky_register(self.labelName), self.comment)

    cls = type(class_name, (BranchInstruction,), {
        "__init__": _init,
        "to_stinky_logical": _to_stinky_logical,
        "__slots__": (),
    })
    cls.__qualname__ = class_name
    cls.__module__ = __name__
    return cls


# logicalIR: SBranch
SBranch = _make_branch_class("SBranch", "s_branch")
# logicalIR: SCBranchSCC0
SCBranchSCC0 = _make_branch_class("SCBranchSCC0", "s_cbranch_scc0")
# logicalIR: SCBranchSCC1
SCBranchSCC1 = _make_branch_class("SCBranchSCC1", "s_cbranch_scc1")
SAddPCI64_SIMM = _make_branch_class("SAddPCI64_SIMM", "s_add_pc_i64")


SAddPCI64_SIMM.to_stinky_logical = lambda self: None  # no logicalIR mapping
# logicalIR: SCBranchVCCNZ
SCBranchVCCNZ = _make_branch_class("SCBranchVCCNZ", "s_cbranch_vccnz")
# logicalIR: SCBranchVCCZ
SCBranchVCCZ = _make_branch_class("SCBranchVCCZ", "s_cbranch_vccz")
# logicalIR: SSetPCB64
SSetPCB64 = _make_reg_jump_class("SSetPCB64", "s_setpc_b64", has_dest=False)
# logicalIR: SSwapPCB64
SSwapPCB64 = _make_reg_jump_class("SSwapPCB64", "s_swappc_b64", has_dest=True)
# logicalIR: SCBranchExecZ
SCBranchExecZ = _make_branch_class("SCBranchExecZ", "s_cbranch_execz")
# logicalIR: SCBranchExecNZ
SCBranchExecNZ = _make_branch_class("SCBranchExecNZ", "s_cbranch_execnz")


# ==========================================================================
# Compare instructions (Phase 6 Step 5)
# source: rocisa/rocisa/src/instruction/cmp.cpp
# ==========================================================================
#
# Three shapes:
#   - Scalar Compare (SCmp*, SBitcmp1B32): no dst, 2 srcs.
#     Native API: (src0, src1, comment="")
#     Stinkytofu binding: (src0, src1, comment="")
#   - Vector Compare (VCmp*): dst, src0, src1.
#     Native API: (dst, src0, src1, sdwa=None, comment="")
#     Stinkytofu binding: (dest, src0, src1, dpp=None, sdwa=None, comment="")
#   - Vector CompareX (VCmpX*): same shape as VCmp.
#
# VCmpInstruction / VCmpXInstruction base classes remain dummies (not in
# logical IR); concrete subclasses are what matters.
VCmpInstruction = make_dummy_class(f"{_P}.VCmpInstruction")
VCmpXInstruction = make_dummy_class(f"{_P}.VCmpXInstruction")


def _make_scalar_cmp_class(class_name: str, mnemonic: str, inst_type: "InstType"):
    """Factory for scalar compare instruction shims (no dst, 2 srcs)."""

    def __init__(self, src0: Any = None, src1: Any = None,
                 comment: str = "", **kw):
        _ = kw
        CommonInstruction.__init__(
            self,
            instType=inst_type,
            dst=None,
            srcs=[src0, src1],
            dpp=None,
            sdwa=None,
            vop3=None,
            comment=comment,
        )
        self.setInst(mnemonic)

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        src0_reg = _to_stinky_register(self.srcs[0])
        src1_reg = _to_stinky_register(self.srcs[1])
        factory = getattr(_st, class_name)
        return factory(src0_reg, src1_reg, comment=self.comment)

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (CommonInstruction,), {
        "__doc__": f"``{mnemonic} src0, src1`` shim with stinkytofu left-path bridge.",
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
    })
    return cls


def _make_vcmp_class(class_name: str, mnemonic: str, inst_type: "InstType"):
    """Factory for vector compare instruction shims (dst, src0, src1)."""

    def __init__(self, dst: Any = None, src0: Any = None, src1: Any = None,
                 sdwa: Any = None, comment: str = "", dpp: Any = None, **kw):
        _ = kw
        CommonInstruction.__init__(
            self,
            instType=inst_type,
            dst=dst,
            srcs=[src0, src1],
            dpp=dpp,
            sdwa=sdwa,
            vop3=None,
            comment=comment,
        )
        self.setInst(mnemonic)

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        dst_reg = _to_stinky_register(self.dst)
        src0_reg = _to_stinky_register(self.srcs[0])
        src1_reg = _to_stinky_register(self.srcs[1])
        factory = getattr(_st, class_name)
        return factory(dst_reg, src0_reg, src1_reg, comment=self.comment)

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (CommonInstruction,), {
        "__doc__": f"``{mnemonic} dst, src0, src1`` shim with stinkytofu left-path bridge.",
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
    })
    return cls


# -- Scalar Compare (no dst, 2 srcs) --
# logicalIR: SCmpEQI32
SCmpEQI32 = _make_scalar_cmp_class("SCmpEQI32", "s_cmp_eq_i32", InstType.INST_I32)
# logicalIR: SCmpEQU32
SCmpEQU32 = _make_scalar_cmp_class("SCmpEQU32", "s_cmp_eq_u32", InstType.INST_U32)
# logicalIR: SCmpEQU64
SCmpEQU64 = _make_scalar_cmp_class("SCmpEQU64", "s_cmp_eq_u64", InstType.INST_U64)
# logicalIR: SCmpGeI32
SCmpGeI32 = _make_scalar_cmp_class("SCmpGeI32", "s_cmp_ge_i32", InstType.INST_I32)
# logicalIR: SCmpGeU32
SCmpGeU32 = _make_scalar_cmp_class("SCmpGeU32", "s_cmp_ge_u32", InstType.INST_U32)
# logicalIR: SCmpGtI32
SCmpGtI32 = _make_scalar_cmp_class("SCmpGtI32", "s_cmp_gt_i32", InstType.INST_I32)
# logicalIR: SCmpGtU32
SCmpGtU32 = _make_scalar_cmp_class("SCmpGtU32", "s_cmp_gt_u32", InstType.INST_U32)
# logicalIR: SCmpLeI32
SCmpLeI32 = _make_scalar_cmp_class("SCmpLeI32", "s_cmp_le_i32", InstType.INST_I32)
# logicalIR: SCmpLeU32
SCmpLeU32 = _make_scalar_cmp_class("SCmpLeU32", "s_cmp_le_u32", InstType.INST_U32)
# logicalIR: SCmpLgU32
SCmpLgU32 = _make_scalar_cmp_class("SCmpLgU32", "s_cmp_lg_u32", InstType.INST_U32)
# logicalIR: SCmpLgI32
SCmpLgI32 = _make_scalar_cmp_class("SCmpLgI32", "s_cmp_lg_i32", InstType.INST_I32)
# logicalIR: SCmpLgU64
SCmpLgU64 = _make_scalar_cmp_class("SCmpLgU64", "s_cmp_lg_u64", InstType.INST_U64)
# logicalIR: SCmpLtI32
SCmpLtI32 = _make_scalar_cmp_class("SCmpLtI32", "s_cmp_lt_i32", InstType.INST_I32)
# logicalIR: SCmpLtU32
SCmpLtU32 = _make_scalar_cmp_class("SCmpLtU32", "s_cmp_lt_u32", InstType.INST_U32)
# logicalIR: SBitcmp1B32
SBitcmp1B32 = _make_scalar_cmp_class("SBitcmp1B32", "s_bitcmp1_b32", InstType.INST_B32)
# logicalIR: SCmpKEQU32
SCmpKEQU32 = _make_scalar_cmp_class("SCmpKEQU32", "s_cmpk_eq_u32", InstType.INST_U32)
# logicalIR: SCmpKGeU32
SCmpKGeU32 = _make_scalar_cmp_class("SCmpKGeU32", "s_cmpk_ge_u32", InstType.INST_U32)
# logicalIR: SCmpKGtU32
SCmpKGtU32 = _make_scalar_cmp_class("SCmpKGtU32", "s_cmpk_gt_u32", InstType.INST_U32)
# logicalIR: SCmpKLGU32
SCmpKLGU32 = _make_scalar_cmp_class("SCmpKLGU32", "s_cmpk_lg_u32", InstType.INST_U32)

# -- Vector Compare (dst, src0, src1) --
# logicalIR: VCmpEQF32
VCmpEQF32 = _make_vcmp_class("VCmpEQF32", "v_cmp_eq_f32", InstType.INST_F32)
# logicalIR: VCmpEQF64
VCmpEQF64 = _make_vcmp_class("VCmpEQF64", "v_cmp_eq_f64", InstType.INST_F64)
# logicalIR: VCmpEQU32
VCmpEQU32 = _make_vcmp_class("VCmpEQU32", "v_cmp_eq_u32", InstType.INST_U32)
# logicalIR: VCmpEQI32
VCmpEQI32 = _make_vcmp_class("VCmpEQI32", "v_cmp_eq_i32", InstType.INST_I32)
# logicalIR: VCmpGEF16
VCmpGEF16 = _make_vcmp_class("VCmpGEF16", "v_cmp_ge_f16", InstType.INST_F16)
# logicalIR: VCmpGTF16
VCmpGTF16 = _make_vcmp_class("VCmpGTF16", "v_cmp_gt_f16", InstType.INST_F16)
# logicalIR: VCmpGEF32
VCmpGEF32 = _make_vcmp_class("VCmpGEF32", "v_cmp_ge_f32", InstType.INST_F32)
# logicalIR: VCmpGTF32
VCmpGTF32 = _make_vcmp_class("VCmpGTF32", "v_cmp_gt_f32", InstType.INST_F32)
# logicalIR: VCmpGEF64
VCmpGEF64 = _make_vcmp_class("VCmpGEF64", "v_cmp_ge_f64", InstType.INST_F64)
# logicalIR: VCmpGTF64
VCmpGTF64 = _make_vcmp_class("VCmpGTF64", "v_cmp_gt_f64", InstType.INST_F64)
# logicalIR: VCmpGEI32
VCmpGEI32 = _make_vcmp_class("VCmpGEI32", "v_cmp_ge_i32", InstType.INST_I32)
# logicalIR: VCmpGTI32
VCmpGTI32 = _make_vcmp_class("VCmpGTI32", "v_cmp_gt_i32", InstType.INST_I32)
# logicalIR: VCmpGEU32
VCmpGEU32 = _make_vcmp_class("VCmpGEU32", "v_cmp_ge_u32", InstType.INST_U32)
# logicalIR: VCmpGtU32
VCmpGtU32 = _make_vcmp_class("VCmpGtU32", "v_cmp_gt_u32", InstType.INST_U32)
# logicalIR: VCmpLeU32
VCmpLeU32 = _make_vcmp_class("VCmpLeU32", "v_cmp_le_u32", InstType.INST_U32)
# logicalIR: VCmpLeI32
VCmpLeI32 = _make_vcmp_class("VCmpLeI32", "v_cmp_le_i32", InstType.INST_I32)
# logicalIR: VCmpLtI32
VCmpLtI32 = _make_vcmp_class("VCmpLtI32", "v_cmp_lt_i32", InstType.INST_I32)
# logicalIR: VCmpLtU32
VCmpLtU32 = _make_vcmp_class("VCmpLtU32", "v_cmp_lt_u32", InstType.INST_U32)
# logicalIR: VCmpUF32
VCmpUF32 = _make_vcmp_class("VCmpUF32", "v_cmp_u_f32", InstType.INST_F32)
# logicalIR: VCmpNeI32
VCmpNeI32 = _make_vcmp_class("VCmpNeI32", "v_cmp_ne_i32", InstType.INST_I32)
# logicalIR: VCmpNeU32
VCmpNeU32 = _make_vcmp_class("VCmpNeU32", "v_cmp_ne_u32", InstType.INST_U32)
# logicalIR: VCmpNeU64
VCmpNeU64 = _make_vcmp_class("VCmpNeU64", "v_cmp_ne_u64", InstType.INST_U64)
# logicalIR: VCmpClassF32
VCmpClassF32 = _make_vcmp_class("VCmpClassF32", "v_cmp_class_f32", InstType.INST_F32)

# -- Vector CompareX (dst, src0, src1) --
# logicalIR: VCmpXClassF32
VCmpXClassF32 = _make_vcmp_class("VCmpXClassF32", "v_cmpx_class_f32", InstType.INST_F32)
# logicalIR: VCmpXEqU32
VCmpXEqU32 = _make_vcmp_class("VCmpXEqU32", "v_cmpx_eq_u32", InstType.INST_U32)
# logicalIR: VCmpXGeU32
VCmpXGeU32 = _make_vcmp_class("VCmpXGeU32", "v_cmpx_ge_u32", InstType.INST_U32)
# logicalIR: VCmpXGtU32
VCmpXGtU32 = _make_vcmp_class("VCmpXGtU32", "v_cmpx_gt_u32", InstType.INST_U32)
# logicalIR: VCmpXLeU32
VCmpXLeU32 = _make_vcmp_class("VCmpXLeU32", "v_cmpx_le_u32", InstType.INST_U32)
# logicalIR: VCmpXLeI32
VCmpXLeI32 = _make_vcmp_class("VCmpXLeI32", "v_cmpx_le_i32", InstType.INST_I32)
# logicalIR: VCmpXLtF32
VCmpXLtF32 = _make_vcmp_class("VCmpXLtF32", "v_cmpx_lt_f32", InstType.INST_F32)
# logicalIR: VCmpXLtI32
VCmpXLtI32 = _make_vcmp_class("VCmpXLtI32", "v_cmpx_lt_i32", InstType.INST_I32)
# logicalIR: VCmpXLtU32
VCmpXLtU32 = _make_vcmp_class("VCmpXLtU32", "v_cmpx_lt_u32", InstType.INST_U32)
# logicalIR: VCmpXLtU64
VCmpXLtU64 = _make_vcmp_class("VCmpXLtU64", "v_cmpx_lt_u64", InstType.INST_U64)
# logicalIR: VCmpXNeU16
VCmpXNeU16 = _make_vcmp_class("VCmpXNeU16", "v_cmpx_ne_u16", InstType.INST_U16)
# logicalIR: VCmpXNeU32
VCmpXNeU32 = _make_vcmp_class("VCmpXNeU32", "v_cmpx_ne_u32", InstType.INST_U32)


# ==========================================================================
# Common ALU / control instructions
# source: rocisa/rocisa/src/instruction/common.cpp
# ==========================================================================
# logicalIR: SAbsI32
SAbsI32 = _make_scalar_unary_class("SAbsI32", "s_abs_i32", InstType.INST_I32)
# logicalIR: SMaxI32
SMaxI32 = _make_scalar_alu_class("SMaxI32", "s_max_i32", InstType.INST_I32)
# logicalIR: SMaxU32
SMaxU32 = _make_scalar_alu_class("SMaxU32", "s_max_u32", InstType.INST_U32)
# logicalIR: SMinI32
SMinI32 = _make_scalar_alu_class("SMinI32", "s_min_i32", InstType.INST_I32)
# logicalIR: SMinU32
SMinU32 = _make_scalar_alu_class("SMinU32", "s_min_u32", InstType.INST_U32)
# SAddI32 — real class (see Scalar ALU section above)
# SAddU32 — real class (see Scalar ALU section above)
# SAddCU32 — real class (see Scalar ALU section above)
_SAddU64 = _make_scalar_alu_class("SAddU64", "s_add_u64", InstType.INST_U64)
# logicalIR: SAddU64 (composite)
SAddU64 = _make_scalar_alu_class("SAddU64", "s_add_u64", InstType.INST_U64)
# SMulI32 — real class (see Scalar ALU section above)
# SMulHII32 — real class (see Scalar ALU section above)
# SMulHIU32 — real class (see Scalar ALU section above)
# SMulLOU32 — real class (see Scalar ALU section above)
# SSubI32 — real class (see Scalar ALU section above)
# SSubU32 — real class (see Scalar ALU section above)
# SSubBU32 — real class (see Scalar ALU section above)
# logicalIR: SCSelectB32
SCSelectB32 = _make_scalar_alu_class("SCSelectB32", "s_cselect_b32", InstType.INST_B32)
# logicalIR: SCSelectB64
SCSelectB64 = _make_scalar_alu_class("SCSelectB64", "s_cselect_b64", InstType.INST_B64)
# SAndB32 — real class (see Scalar ALU section above)
# SAndB64 — real class (see Scalar ALU section above)
# SAndN2B32 — real class (see Scalar ALU section above)
# SOrB32 — real class (see Scalar ALU section above)
# SXorB32 — real class (see Scalar ALU section above)
# SOrB64 — real class (see Scalar ALU section above)
# logicalIR: SSubU64
SSubU64 = _make_scalar_alu_class("SSubU64", "s_sub_u64", InstType.INST_U64)
# logicalIR: SGetPCB64
SGetPCB64 = _make_zero_src_class("SGetPCB64", "s_getpc_b64", InstType.INST_B64)
# SLShiftLeftB32 — real class (see Scalar ALU section above)
# SLShiftRightB32 — real class (see Scalar ALU section above)
# SLShiftLeftB64 — real class (see Scalar ALU section above)
# SLShiftRightB64 — real class (see Scalar ALU section above)
# SAShiftRightI32 — real class (see Scalar ALU section above)
# SLShiftLeft1AddU32 — real class (see Scalar ALU section above)
# SLShiftLeft2AddU32 — real class (see Scalar ALU section above)
# SLShiftLeft3AddU32 — real class (see Scalar ALU section above)
# SLShiftLeft4AddU32 — real class (see Scalar ALU section above)
# logicalIR: SSetMask
SSetMask = _make_scalar_unary_class("SSetMask", "s_mov_b32", InstType.INST_B32)
# logicalIR: SCMovB32
SCMovB32 = _make_scalar_unary_class("SCMovB32", "s_cmov_b32", InstType.INST_B32)
# logicalIR: SCMovB64
SCMovB64 = _make_scalar_unary_class("SCMovB64", "s_cmov_b64", InstType.INST_B64)
# logicalIR: SFf1B32
SFf1B32 = _make_scalar_unary_class("SFf1B32", "s_ff1_i32_b32", InstType.INST_B32)
# logicalIR: SBfmB32
SBfmB32 = _make_scalar_alu_class("SBfmB32", "s_bfm_b32", InstType.INST_B32)
# logicalIR: SBfeU32
SBfeU32 = _make_scalar_alu_class("SBfeU32", "s_bfe_u32", InstType.INST_U32)
# logicalIR: SFlbitI32B32
SFlbitI32B32 = _make_scalar_unary_class("SFlbitI32B32", "s_flbit_i32_b32", InstType.INST_B32)
# logicalIR: SMovkI32
SMovkI32 = _make_scalar_unary_class("SMovkI32", "s_movk_i32", InstType.INST_I32)
# logicalIR: SSExtI16toI32
SSExtI16toI32 = _make_scalar_unary_class("SSExtI16toI32", "s_sext_i32_i16", InstType.INST_I32)
# SAndSaveExecB32 — real class (see Scalar ALU section above)
# SAndSaveExecB64 — real class (see Scalar ALU section above)
# SOrSaveExecB32 — real class (see Scalar ALU section above)
# SOrSaveExecB64 — real class (see Scalar ALU section above)
# logicalIR: SSetPrior
SSetPrior = _make_imm_no_dest_class("SSetPrior", "s_setprio")
# SBarrier — real class (see SBarrier section above)
# logicalIR: SDcacheWb
SDcacheWb = _make_no_operand_class("SDcacheWb", "s_dcache_wb")
# logicalIR: GlobalWb
GlobalWb = _make_no_operand_class("GlobalWb", "global_wb")
# logicalIR: GlobalInv
GlobalInv = _make_no_operand_class("GlobalInv", "global_inv")
# logicalIR: STtraceData
STtraceData = _make_no_operand_class("STtraceData", "s_ttracedata")
# SNop — real class (see class SNop above, after SMovB64).
# logicalIR: VNop
class VNop(Instruction):
    """``v_nop`` shim — emits *count* copies of ``v_nop``."""

    __slots__ = ("count",)

    def __init__(self, count: int = 1, comment: str = ""):
        super().__init__(InstType.INST_NOTYPE, comment)
        self.count = int(count)
        self.setInst("v_nop")

    def getParams(self):
        return [self.count]

    def getDstParams(self):
        return []

    def getSrcParams(self):
        return [self.count]

    def to_stinky_logical(self, _module=None):
        import stinkytofu as _st
        if self.count <= 1:
            return _st.VNop(self.comment)
        return [_st.VNop(self.comment) for _ in range(self.count)]

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = VNop(count=self.count, comment=self.comment)
        memo[id(self)] = dup
        return dup


# logicalIR: SEndpgm
class SEndpgm(Instruction):
    """``s_endpgm`` shim with stinkytofu left-path bridge."""

    __slots__ = ()

    def __init__(self, comment: str = ""):
        super().__init__(InstType.INST_NOTYPE, comment)
        self.setInst("s_endpgm")

    def getParams(self):
        return []

    def getDstParams(self):
        return []

    def getSrcParams(self):
        return []

    def toString(self) -> str:
        return self.formatWithComment(self.instStr)

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        return _st.SEndpgm(self.comment)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = SEndpgm(comment=self.comment)
        memo[id(self)] = dup
        return dup


# logicalIR: SSleep
SSleep = _make_imm_no_dest_class("SSleep", "s_sleep")
# logicalIR: SSetVgprMsb
SSetVgprMsb = _make_imm_no_dest_class("SSetVgprMsb", "s_set_vgpr_msb")
# SGetRegB32 — real class (see Scalar Control section above)
# SSetRegB32 — real class (see Scalar Control section above)
# SSetRegIMM32B32 — real class (see Scalar Control section above)


# ==========================================================================
# Wait-count instructions
# source: rocisa/rocisa/include/instruction/common.hpp
# logicalIR: SWaitCnt, SWaitTensorcnt, SWaitXCnt
# ==========================================================================

# Markers embedded in the comment field of SWaitCnt logical instructions to
# encode which gfx12+ wait opcode the instruction should lower to.  The
# post-processing step in Module.to_stinky_asm() (code.py) uses these to
# replace generic ``s_waitcnt N`` assembly text with the correct instruction.
_WAIT_MARKER_LOADCNT = "\x00@WL@"
_WAIT_MARKER_STORECNT = "\x00@WS@"
_WAIT_MARKER_DSCNT = "\x00@WD@"
_WAIT_MARKER_KMCNT = "\x00@WK@"


class _SWaitCnt(Instruction):
    """``s_waitcnt`` primitive (lgkmcnt/vmcnt combined)."""

    __slots__ = ("lgkmcnt", "vmcnt")

    def __init__(self, lgkmcnt: int = -1, vmcnt: int = -1, comment: str = ""):
        super().__init__(InstType.INST_NOTYPE, comment)
        self.lgkmcnt = int(lgkmcnt)
        self.vmcnt = int(vmcnt)
        self.setInst("s_waitcnt")

    def getParams(self):
        return [self.lgkmcnt, self.vmcnt]

    def getDstParams(self):
        return []

    def getSrcParams(self):
        return [self.lgkmcnt, self.vmcnt]

    def toString(self) -> str:
        if self.lgkmcnt == 0 and self.vmcnt == 0:
            wait_str = "0"
        else:
            parts: List[str] = []
            if self.lgkmcnt != -1:
                parts.append(f"lgkmcnt({self.lgkmcnt})")
            if self.vmcnt != -1:
                parts.append(f"vmcnt({self.vmcnt})")
            wait_str = ", ".join(parts)
        return self.formatWithComment("s_waitcnt " + wait_str)

    def to_stinky_logical(self) -> Any:
        """Map legacy _SWaitCnt to gfx12+ typed waits.

        On gfx12+, lgkmcnt maps to dscnt (DS/LDS counter) and vmcnt maps
        to loadcnt (VMEM load counter), mirroring the native C++
        AllHwMappings.cpp logic that sets dlcnt=lgkmcnt, vlcnt=vmcnt.
        """
        import stinkytofu as _st  # noqa: WPS433

        insts: List[Any] = []
        if self.lgkmcnt != -1:
            insts.append(_st.SWaitCnt(
                _to_stinky_register(self.lgkmcnt),
                _WAIT_MARKER_DSCNT + self.comment,
            ))
        if self.vmcnt != -1:
            insts.append(_st.SWaitCnt(
                _to_stinky_register(self.vmcnt),
                _WAIT_MARKER_LOADCNT + self.comment,
            ))
        if not insts:
            return _st.SWaitCnt(_to_stinky_register(0), self.comment)
        return insts

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = _SWaitCnt(lgkmcnt=self.lgkmcnt, vmcnt=self.vmcnt, comment=self.comment)
        memo[id(self)] = dup
        return dup


class _SWaitCntVscnt(Instruction):
    """``s_waitcnt_vscnt`` primitive (vscnt counter only, gfx10+)."""

    __slots__ = ("cnt",)

    def __init__(self, cnt: int = 0, comment: str = ""):
        super().__init__(InstType.INST_NOTYPE, comment)
        self.cnt = int(cnt)
        self.setInst("s_waitcnt_vscnt")

    def getParams(self):
        return [self.cnt]

    def getDstParams(self):
        return []

    def getSrcParams(self):
        return [self.cnt]

    def toString(self) -> str:
        return self.formatWithComment(f"s_waitcnt_vscnt null, {self.cnt}")

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        return _st.SWaitCnt(
            _to_stinky_register(self.cnt),
            _WAIT_MARKER_STORECNT + self.comment,
        )

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = _SWaitCntVscnt(cnt=self.cnt, comment=self.comment)
        memo[id(self)] = dup
        return dup


class _SWaitStorecnt(Instruction):
    """``s_wait_storecnt`` primitive (store counter, gfx11+)."""

    __slots__ = ("cnt",)

    def __init__(self, cnt: int = 0, comment: str = ""):
        super().__init__(InstType.INST_NOTYPE, comment)
        self.cnt = int(cnt)
        self.setInst("s_wait_storecnt")

    def getParams(self):
        return [self.cnt]

    def getDstParams(self):
        return []

    def getSrcParams(self):
        return [self.cnt]

    def toString(self) -> str:
        return self.formatWithComment(f"s_wait_storecnt {self.cnt}")

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        return _st.SWaitCnt(
            _to_stinky_register(self.cnt),
            _WAIT_MARKER_STORECNT + self.comment,
        )

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = _SWaitStorecnt(cnt=self.cnt, comment=self.comment)
        memo[id(self)] = dup
        return dup


class _SWaitLoadcnt(Instruction):
    """``s_wait_loadcnt`` primitive (load counter, gfx11+)."""

    __slots__ = ("cnt",)

    def __init__(self, cnt: int = 0, comment: str = ""):
        super().__init__(InstType.INST_NOTYPE, comment)
        self.cnt = int(cnt)
        self.setInst("s_wait_loadcnt")

    def getParams(self):
        return [self.cnt]

    def getDstParams(self):
        return []

    def getSrcParams(self):
        return [self.cnt]

    def toString(self) -> str:
        return self.formatWithComment(f"s_wait_loadcnt {self.cnt}")

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        return _st.SWaitCnt(
            _to_stinky_register(self.cnt),
            _WAIT_MARKER_LOADCNT + self.comment,
        )

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = _SWaitLoadcnt(cnt=self.cnt, comment=self.comment)
        memo[id(self)] = dup
        return dup


class _SWaitKMcnt(Instruction):
    """``s_wait_kmcnt`` primitive (kernel memory counter, gfx11+)."""

    __slots__ = ("cnt",)

    def __init__(self, cnt: int = 0, comment: str = ""):
        super().__init__(InstType.INST_NOTYPE, comment)
        self.cnt = int(cnt)
        self.setInst("s_wait_kmcnt")

    def getParams(self):
        return [self.cnt]

    def getDstParams(self):
        return []

    def getSrcParams(self):
        return [self.cnt]

    def toString(self) -> str:
        return self.formatWithComment(f"s_wait_kmcnt {self.cnt}")

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        return _st.SWaitCnt(
            _to_stinky_register(self.cnt),
            _WAIT_MARKER_KMCNT + self.comment,
        )

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = _SWaitKMcnt(cnt=self.cnt, comment=self.comment)
        memo[id(self)] = dup
        return dup


class _SWaitDscnt(Instruction):
    """``s_wait_dscnt`` primitive (DS/LDS counter, gfx11+)."""

    __slots__ = ("cnt",)

    def __init__(self, cnt: int = 0, comment: str = ""):
        super().__init__(InstType.INST_NOTYPE, comment)
        self.cnt = int(cnt)
        self.setInst("s_wait_dscnt")

    def getParams(self):
        return [self.cnt]

    def getDstParams(self):
        return []

    def getSrcParams(self):
        return [self.cnt]

    def toString(self) -> str:
        return self.formatWithComment(f"s_wait_dscnt {self.cnt}")

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        return _st.SWaitCnt(
            _to_stinky_register(self.cnt),
            _WAIT_MARKER_DSCNT + self.comment,
        )

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = _SWaitDscnt(cnt=self.cnt, comment=self.comment)
        memo[id(self)] = dup
        return dup


class SWaitCnt(Instruction):
    """High-level ``s_waitcnt`` composite (vlcnt/vscnt/dscnt/kmcnt).

    Mirrors rocisa::SWaitCnt which is a CompositeInstruction that
    decomposes into _SWaitCnt/_SWaitCntVscnt. For the logical IR path,
    we decompose into individual gfx12+ typed wait instructions.
    """

    __slots__ = ("vlcnt", "vscnt", "dscnt", "kmcnt", "waitAll")

    def __init__(self, vlcnt: int = -1, vscnt: int = -1,
                 dscnt: int = -1, kmcnt: int = -1,
                 comment: str = "", waitAll: bool = False):
        super().__init__(InstType.INST_NOTYPE, comment)
        self.vlcnt = int(vlcnt)
        self.vscnt = int(vscnt)
        self.dscnt = int(dscnt)
        self.kmcnt = int(kmcnt)
        self.waitAll = bool(waitAll)
        self.setInst("s_waitcnt")

    def getParams(self):
        return []

    def getDstParams(self):
        return []

    def getSrcParams(self):
        return []

    def toString(self) -> str:
        return self.formatWithComment(self.instStr)

    def to_stinky_logical(self) -> Any:
        """Decompose into individual gfx12+ wait logical instructions.

        Returns a list of SWaitCnt logical instructions with type markers
        so the post-processing step can emit the correct opcodes.
        """
        import stinkytofu as _st  # noqa: WPS433

        # rocisa SWaitCnt::setupInstructions treats waitAll as "wait for
        # everything": vlcnt = vscnt = dscnt = kmcnt = 0. Mirror that here so a
        # bare SWaitCnt(waitAll=True) lowers to the four typed gfx12 waits rather
        # than the unsupported ``s_waitcnt 0``.
        dscnt = 0 if self.waitAll else self.dscnt
        kmcnt = 0 if self.waitAll else self.kmcnt
        vlcnt = 0 if self.waitAll else self.vlcnt
        vscnt = 0 if self.waitAll else self.vscnt

        insts: List[Any] = []
        if dscnt != -1:
            insts.append(_st.SWaitCnt(
                _to_stinky_register(dscnt),
                _WAIT_MARKER_DSCNT + self.comment,
            ))
        if kmcnt != -1:
            insts.append(_st.SWaitCnt(
                _to_stinky_register(kmcnt),
                _WAIT_MARKER_KMCNT + self.comment,
            ))
        if vlcnt != -1:
            insts.append(_st.SWaitCnt(
                _to_stinky_register(vlcnt),
                _WAIT_MARKER_LOADCNT + self.comment,
            ))
        if vscnt != -1:
            insts.append(_st.SWaitCnt(
                _to_stinky_register(vscnt),
                _WAIT_MARKER_STORECNT + self.comment,
            ))
        if not insts:
            return _st.SWaitCnt(_to_stinky_register(0), self.comment)
        return insts

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = SWaitCnt(
            vlcnt=self.vlcnt, vscnt=self.vscnt, dscnt=self.dscnt,
            kmcnt=self.kmcnt, comment=self.comment, waitAll=self.waitAll,
        )
        memo[id(self)] = dup
        return dup


class SWaitXCnt(Instruction):
    """``s_wait_xcnt`` shim."""

    __slots__ = ("cnt",)

    def __init__(self, xcnt: int = 0, comment: str = "", *, cnt: int = None):
        super().__init__(InstType.INST_NOTYPE, comment)
        # Native rocisa ctor keyword is ``xcnt`` and clamps to min(xcnt, 63);
        # accept legacy ``cnt`` too for backward compatibility.
        value = xcnt if cnt is None else cnt
        self.cnt = min(int(value), 63)
        self.setInst("s_wait_xcnt")

    def getParams(self):
        return [self.cnt]

    def getDstParams(self):
        return []

    def getSrcParams(self):
        return [self.cnt]

    def toString(self) -> str:
        return self.formatWithComment(f"s_wait_xcnt {self.cnt}")

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        return _st.SWaitXCnt(_to_stinky_register(self.cnt), self.comment)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = SWaitXCnt(cnt=self.cnt, comment=self.comment)
        memo[id(self)] = dup
        return dup


class SWaitTensorcnt(Instruction):
    """``s_wait_tensorcnt`` shim."""

    __slots__ = ("cnt",)

    def __init__(self, cnt: int = 0, tensorcnt: int = None, comment: str = ""):
        if tensorcnt is not None:
            cnt = tensorcnt
        super().__init__(InstType.INST_NOTYPE, comment)
        self.cnt = int(cnt)
        self.setInst("s_wait_tensorcnt")

    def getParams(self):
        return [self.cnt]

    def getDstParams(self):
        return []

    def getSrcParams(self):
        return [self.cnt]

    def toString(self) -> str:
        return self.formatWithComment(f"s_wait_tensorcnt {self.cnt}")

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        return _st.SWaitTensorcnt(_to_stinky_register(self.cnt), self.comment)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = SWaitTensorcnt(cnt=self.cnt, comment=self.comment)
        memo[id(self)] = dup
        return dup


class SWaitAlu(Instruction):
    """SWaitAlu — dependency counter wait instruction.

    Carries 7 optional counter fields; -1 means "not specified".
    Emits a LogicalIR SWaitAlu instruction with specialData.
    """

    __slots__ = ("va_vdst", "va_sdst", "va_ssrc", "hold_cnt", "vm_vsrc", "va_vcc", "sa_sdst")

    def __init__(self, va_vdst=-1, va_sdst=-1, va_ssrc=-1,
                 hold_cnt=-1, vm_vsrc=-1, va_vcc=-1, sa_sdst=-1,
                 comment=""):
        super().__init__(InstType.INST_NOTYPE, comment)
        self.va_vdst = va_vdst
        self.va_sdst = va_sdst
        self.va_ssrc = va_ssrc
        self.hold_cnt = hold_cnt
        self.vm_vsrc = vm_vsrc
        self.va_vcc = va_vcc
        self.sa_sdst = sa_sdst
        self.setInst("s_wait_alu")

    def getParams(self):
        return [self.va_vdst, self.va_sdst, self.va_ssrc,
                self.hold_cnt, self.vm_vsrc, self.va_vcc, self.sa_sdst]

    def getDstParams(self):
        return []

    def getSrcParams(self):
        return []

    def to_stinky_logical(self, _module=None):
        import stinkytofu as st
        return st.SWaitAlu(
            va_vdst=self.va_vdst,
            va_sdst=self.va_sdst,
            va_ssrc=self.va_ssrc,
            hold_cnt=self.hold_cnt,
            vm_vsrc=self.vm_vsrc,
            va_vcc=self.va_vcc,
            sa_sdst=self.sa_sdst,
            comment=self.comment,
        )

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = SWaitAlu(
            va_vdst=self.va_vdst,
            va_sdst=self.va_sdst,
            va_ssrc=self.va_ssrc,
            hold_cnt=self.hold_cnt,
            vm_vsrc=self.vm_vsrc,
            va_vcc=self.va_vcc,
            sa_sdst=self.sa_sdst,
            comment=self.comment,
        )
        memo[id(self)] = dup
        return dup
# logicalIR: SDelayAlu
SDelayAlu = _make_imm_no_dest_class("SDelayAlu", "s_delay_alu")


def _sdelayalu_to_stinky_logical(self) -> Any:
    """SNop placeholder workaround for stinkytofu SDelayAluData assertion bug."""
    import stinkytofu as _st  # noqa: WPS433

    # The raw immediate encodes the full s_delay_alu operand; emit it verbatim
    # so post-processing can restore the original instruction text.
    alu_text = _input_to_str(self._imm_value)
    return _st.SNop(_st.Register(0), "DELAY_ALU:" + alu_text)


SDelayAlu.to_stinky_logical = _sdelayalu_to_stinky_logical
# logicalIR: VAddF16
VAddF16 = _make_scalar_alu_class("VAddF16", "v_add_f16", InstType.INST_F16)
# VAddF32 — real class (see Vector ALU section above)
# logicalIR: VAddF64
VAddF64 = _make_scalar_alu_class("VAddF64", "v_add_f64", InstType.INST_F64)
# logicalIR: VAddI32
VAddI32 = _make_scalar_alu_class("VAddI32", "v_add_nc_i32", InstType.INST_I32)
# VAddU32 — real class (see Vector ALU section above)
# logicalIR: VAddCOU32
class VAddCOU32(CommonInstruction):
    """``v_add_co_u32`` -- integer add with explicit carry-out.

    Native rocisa signature is ``(dst, dst1, src0, src1)`` where ``dst1``
    (e.g. ``VCC()``) is the carry-out *destination*, NOT a source. The
    generic ALU factory has no positional ``dst1`` slot, so a positional
    call like ``VAddCOU32(dst, VCC(), src0, src1)`` would misread VCC as
    src0 and drop src1. stinkytofu re-derives the implicit ``vcc_lo``
    carry-out itself, so ``dst1`` is dropped when lowering.
    """

    def __init__(self, dst: Any, dst1: Any = None, src0: Any = None,
                 src1: Any = None, *extra: Any, comment: str = "", **kw):
        _ = kw
        for a in extra:
            if isinstance(a, str):
                comment = a
        super().__init__(InstType.INST_U32, dst, [src0, src1], comment=comment)
        self.dst1 = dst1
        self.setInst("v_add_co_u32")

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433
        return _st.VAddCOU32(
            _to_stinky_register(self.dst),
            _to_stinky_register(self.srcs[0]),
            _to_stinky_register(self.srcs[1]),
            comment=self.comment)


# logicalIR: VAddCCOU32
class VAddCCOU32(CommonInstruction):
    """``v_add_co_ci_u32`` -- integer add with carry-in and carry-out.

    Native rocisa signature is ``(dst, dst1, src0, src1, src2)``: ``dst1``
    is the carry-out destination and ``src2`` the carry-in (both ``VCC()``
    in practice). stinkytofu handles both implicit VCC operands, so they
    are dropped when lowering (only ``dst, src0, src1`` are forwarded).
    """

    def __init__(self, dst: Any, dst1: Any = None, src0: Any = None,
                 src1: Any = None, src2: Any = None, *extra: Any,
                 comment: str = "", **kw):
        _ = kw
        for a in extra:
            if isinstance(a, str):
                comment = a
        super().__init__(InstType.INST_U32, dst, [src0, src1, src2], comment=comment)
        self.dst1 = dst1
        self.setInst("v_add_co_ci_u32")

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433
        return _st.VAddCCOU32(
            _to_stinky_register(self.dst),
            _to_stinky_register(self.srcs[0]),
            _to_stinky_register(self.srcs[1]),
            comment=self.comment)
_VAddNCU64 = _make_scalar_alu_class("VAddNCU64", "v_add_nc_u64", InstType.INST_U64)
# logicalIR: VAddNCU64 (composite)
VAddNCU64 = _make_scalar_alu_class("VAddNCU64", "v_add_nc_u64", InstType.INST_U64)
# logicalIR: VAddPKF16
VAddPKF16 = _make_scalar_alu_class("VAddPKF16", "v_pk_add_f16", InstType.INST_F16)
_VAddPKF32 = _make_scalar_alu_class("VAddPKF32", "v_pk_add_f32", InstType.INST_F32)
# logicalIR: VAddPKF32
VAddPKF32 = _make_scalar_alu_class("VAddPKF32", "v_pk_add_f32", InstType.INST_F32)
# logicalIR: VAdd3U32
VAdd3U32 = _make_scalar_alu_class("VAdd3U32", "v_add3_u32", InstType.INST_U32)
# logicalIR: VMulF16
VMulF16 = _make_scalar_alu_class("VMulF16", "v_mul_f16", InstType.INST_F16)
# VMulF32 — real class (see Vector ALU section above)
# logicalIR: VMulF64
VMulF64 = _make_scalar_alu_class("VMulF64", "v_mul_f64", InstType.INST_F64)
# logicalIR: VMulPKF16
VMulPKF16 = _make_scalar_alu_class("VMulPKF16", "v_pk_mul_f16", InstType.INST_F16)
# logicalIR: VMulPKF32S
VMulPKF32S = _make_scalar_alu_class("VMulPKF32S", "v_pk_mul_f32", InstType.INST_F32)
_VMulPKF32 = _make_scalar_alu_class("VMulPKF32", "v_pk_mul_f32", InstType.INST_F32)
# logicalIR: VMulPKF32
VMulPKF32 = _make_scalar_alu_class("VMulPKF32", "v_pk_mul_f32", InstType.INST_F32)
# VMulLOU32 — real class (see Vector ALU section above)
# VMulHII32 — real class (see Vector ALU section above)
# VMulHIU32 — real class (see Vector ALU section above)
# VMulI32I24 — real class (see Vector ALU section above)
# VMulU32U24 — real class (see Vector ALU section above)
# VSubF32 — real class (see Vector ALU section above)
# VSubI32 — real class (see Vector ALU section above)
# VSubU32 — real class (see Vector ALU section above)
# logicalIR: VSubCoU32
VSubCoU32 = _make_scalar_alu_class("VSubCoU32", "v_sub_co_u32", InstType.INST_U32)
# logicalIR: VMacF32
VMacF32 = _make_scalar_alu_class("VMacF32", "v_fmac_f32", InstType.INST_F32)


class VDualFMACF32(CommonInstruction):
    """VOPD dual-issue FMA: two independent v_fmac_f32 in one slot."""

    def __init__(self, dstX, src0X, src1X, dstY, src0Y, src1Y, comment=""):
        super().__init__(InstType.INST_F32, dstX, [src0X, src1X, src0Y, src1Y], comment=comment)
        self.dst1 = dstY
        self.setInst("v_dual_fmac_f32")

    def getArgStr(self) -> str:
        dstX_s = self.dst.toString() if hasattr(self.dst, "toString") else str(self.dst)
        s0X = self.srcs[0].toString() if hasattr(self.srcs[0], "toString") else str(self.srcs[0])
        s1X = self.srcs[1].toString() if hasattr(self.srcs[1], "toString") else str(self.srcs[1])
        dstY_s = self.dst1.toString() if hasattr(self.dst1, "toString") else str(self.dst1)
        s0Y = self.srcs[2].toString() if hasattr(self.srcs[2], "toString") else str(self.srcs[2])
        s1Y = self.srcs[3].toString() if hasattr(self.srcs[3], "toString") else str(self.srcs[3])
        return f"{dstX_s}, {s0X}, {s1X} :: v_dual_fmac_f32 {dstY_s}, {s0Y}, {s1Y}"

    def getSrcParams(self):
        params = list(self.srcs)
        if self.dst is not None:
            params.append(self.dst)
        if self.dst1 is not None:
            params.append(self.dst1)
        return params


# logicalIR: VDot2CF32F16
VDot2CF32F16 = _make_ternary_class("VDot2CF32F16", "v_dot2_c_f32_f16", InstType.INST_F32)
# logicalIR: VDot2CF32BF16
VDot2CF32BF16 = _make_ternary_class("VDot2CF32BF16", "v_dot2_c_f32_b_f16", InstType.INST_F32)
# logicalIR: VDot2F32F16
VDot2F32F16 = _make_ternary_class("VDot2F32F16", "v_dot2_f32_f16", InstType.INST_F32)
# logicalIR: VDot2F32BF16
VDot2F32BF16 = _make_ternary_class("VDot2F32BF16", "v_dot2_f32_b_f16", InstType.INST_F32)
# logicalIR: VFmaF16
VFmaF16 = _make_ternary_class("VFmaF16", "v_fma_f16", InstType.INST_F16)
# VFmaF32 — real class (see Vector ALU section above)
# logicalIR: VFmaF64
VFmaF64 = _make_ternary_class("VFmaF64", "v_fma_f64", InstType.INST_F64)
# logicalIR: VFmaPKF16
VFmaPKF16 = _make_ternary_class("VFmaPKF16", "v_pk_fma_f16", InstType.INST_F16)
# logicalIR: VFmaPKF32
VFmaPKF32 = _make_ternary_class("VFmaPKF32", "v_pk_fma_f32", InstType.INST_F32)
# VFmaMixF32 — real class (see Vector ALU section above)
# logicalIR: VMadI32I24
VMadI32I24 = _make_ternary_class("VMadI32I24", "v_mad_i32_i24", InstType.INST_I32)
# logicalIR: VMadU32U24
VMadU32U24 = _make_ternary_class("VMadU32U24", "v_mad_u32_u24", InstType.INST_U32)
# logicalIR: VMadMixF32
VMadMixF32 = _make_ternary_class("VMadMixF32", "v_mad_mix_f32", InstType.INST_F32)
# logicalIR: VExpF16
VExpF16 = _make_scalar_unary_class("VExpF16", "v_exp_f16", InstType.INST_F16)
# logicalIR: VExpF32
VExpF32 = _make_scalar_unary_class("VExpF32", "v_exp_f32", InstType.INST_F32)
# logicalIR: VRcpF16
VRcpF16 = _make_scalar_unary_class("VRcpF16", "v_rcp_f16", InstType.INST_F16)
# logicalIR: VRcpF32
VRcpF32 = _make_scalar_unary_class("VRcpF32", "v_rcp_f32", InstType.INST_F32)
# logicalIR: VRcpIFlagF32
VRcpIFlagF32 = _make_scalar_unary_class("VRcpIFlagF32", "v_rcp_iflag_f32", InstType.INST_F32)
# logicalIR: VRcpF64
VRcpF64 = _make_scalar_unary_class("VRcpF64", "v_rcp_f64", InstType.INST_F64)
# logicalIR: VRsqF16
VRsqF16 = _make_scalar_unary_class("VRsqF16", "v_rsq_f16", InstType.INST_F16)
# logicalIR: VRsqF32
VRsqF32 = _make_scalar_unary_class("VRsqF32", "v_rsq_f32", InstType.INST_F32)
# logicalIR: VRsqIFlagF32
VRsqIFlagF32 = _make_scalar_unary_class("VRsqIFlagF32", "v_rsq_iflag_f32", InstType.INST_F32)
# logicalIR: VMaxF16
VMaxF16 = _make_scalar_alu_class("VMaxF16", "v_max_f16", InstType.INST_F16)
# logicalIR: VMaxF32
VMaxF32 = _make_scalar_alu_class("VMaxF32", "v_max_f32", InstType.INST_F32)
# logicalIR: VMaxF64
VMaxF64 = _make_scalar_alu_class("VMaxF64", "v_max_f64", InstType.INST_F64)
# logicalIR: VMaxI32
VMaxI32 = _make_scalar_alu_class("VMaxI32", "v_max_i32", InstType.INST_I32)
# logicalIR: VMaxPKF16
VMaxPKF16 = _make_scalar_alu_class("VMaxPKF16", "v_pk_max_f16", InstType.INST_F16)
# logicalIR: VMed3I32
VMed3I32 = _make_ternary_class("VMed3I32", "v_med3_i32", InstType.INST_I32)
# logicalIR: VMed3F32
VMed3F32 = _make_ternary_class("VMed3F32", "v_med3_f32", InstType.INST_F32)
# logicalIR: VMinF16
VMinF16 = _make_scalar_alu_class("VMinF16", "v_min_f16", InstType.INST_F16)
# logicalIR: VMinF32
VMinF32 = _make_scalar_alu_class("VMinF32", "v_min_f32", InstType.INST_F32)
# logicalIR: VMinF64
VMinF64 = _make_scalar_alu_class("VMinF64", "v_min_f64", InstType.INST_F64)
# logicalIR: VMinI32
VMinI32 = _make_scalar_alu_class("VMinI32", "v_min_i32", InstType.INST_I32)
# VAndB32 — real class (see Vector ALU section above)
# VAndOrB32 — real class (see Vector ALU section above)
# logicalIR: VNotB32
VNotB32 = _make_scalar_unary_class("VNotB32", "v_not_b32", InstType.INST_B32)
# VOrB32 — real class (see Vector ALU section above)
# VXorB32 — real class (see Vector ALU section above)
# logicalIR: VPrngB32
VPrngB32 = _make_scalar_unary_class("VPrngB32", "v_prng_b32", InstType.INST_B32)
# VCndMaskB32 — real class (see Vector ALU section above)
# logicalIR: VLShiftLeftB16
VLShiftLeftB16 = _make_vector_shift_class("VLShiftLeftB16", "v_lshlrev_b16", InstType.INST_B16)
# VLShiftLeftB32 — real class (see Vector ALU section above)
# VLShiftRightB32 — real class (see Vector ALU section above)
# VLShiftLeftB64 — real class (see Vector ALU section above)
# VLShiftRightB64 — real class (see Vector ALU section above)
_VLShiftLeftOrB32 = _make_ternary_class("VLShiftLeftOrB32", "v_lshl_or_b32", InstType.INST_B32, shift_position=1)
# logicalIR: VAShiftRightI32
VAShiftRightI32 = _make_vector_shift_class("VAShiftRightI32", "v_ashrrev_i32", InstType.INST_I32)
# logicalIR: VLShiftLeftOrB32
VLShiftLeftOrB32 = _make_ternary_class("VLShiftLeftOrB32", "v_lshl_or_b32", InstType.INST_B32, shift_position=1)
_VAddLShiftLeftU32 = _make_ternary_class("VAddLShiftLeftU32", "v_add_lshl_u32", InstType.INST_U32, shift_position=2)
# logicalIR: VAddLShiftLeftU32 (composite)
VAddLShiftLeftU32 = _make_ternary_class("VAddLShiftLeftU32", "v_add_lshl_u32", InstType.INST_U32, shift_position=2)
_VLShiftLeftAddU32 = _make_ternary_class("VLShiftLeftAddU32", "v_lshl_add_u32", InstType.INST_U32, shift_position=1)
# logicalIR: VLShiftLeftAddU32 (composite)
VLShiftLeftAddU32 = _make_ternary_class("VLShiftLeftAddU32", "v_lshl_add_u32", InstType.INST_U32, shift_position=1)
# logicalIR: VMovB32  -- real class defined at the bottom of this file
# (after ``CommonInstruction`` / ``_to_stinky_register`` are in scope).
# Intentionally NOT declared here so ``from rocisa.instruction import
# VMovB32`` resolves to the real class via the module-scope binding.
_VMovB64 = _make_scalar_unary_class("VMovB64", "v_mov_b64", InstType.INST_B64)
# logicalIR: VMovB64
VMovB64 = _make_scalar_unary_class("VMovB64", "v_mov_b64", InstType.INST_B64)
# logicalIR: VSwapB32
VSwapB32 = _make_scalar_unary_class("VSwapB32", "v_swap_b32", InstType.INST_B32)
# logicalIR: VBfeI32
VBfeI32 = _make_ternary_class("VBfeI32", "v_bfe_i32", InstType.INST_I32)
# logicalIR: VBfeU32
VBfeU32 = _make_ternary_class("VBfeU32", "v_bfe_u32", InstType.INST_U32)
# logicalIR: VBfiB32
VBfiB32 = _make_ternary_class("VBfiB32", "v_bfi_b32", InstType.INST_B32)
# logicalIR: VPackF16toB32
VPackF16toB32 = _make_scalar_alu_class("VPackF16toB32", "v_pack_b32_f16", InstType.INST_B32)
# logicalIR: VAccvgprReadB32
VAccvgprReadB32 = _make_scalar_unary_class("VAccvgprReadB32", "v_accvgpr_read_b32", InstType.INST_B32)
# logicalIR: VAccvgprWrite
VAccvgprWrite = _make_scalar_unary_class("VAccvgprWrite", "v_accvgpr_write", InstType.INST_B32)
# logicalIR: VAccvgprWriteB32
VAccvgprWriteB32 = _make_scalar_unary_class("VAccvgprWriteB32", "v_accvgpr_write_b32", InstType.INST_B32)
# VReadfirstlaneB32 — real class (see Vector ALU section above)
# logicalIR: VReadlaneB32
VReadlaneB32 = _make_scalar_alu_class("VReadlaneB32", "v_readlane_b32", InstType.INST_B32)
# logicalIR: VWritelaneB32
VWritelaneB32 = _make_scalar_alu_class("VWritelaneB32", "v_writelane_b32", InstType.INST_B32)
# logicalIR: VRndneF32
VRndneF32 = _make_scalar_unary_class("VRndneF32", "v_rndne_f32", InstType.INST_F32)
# logicalIR: VPermB32
VPermB32 = _make_ternary_class("VPermB32", "v_perm_b32", InstType.INST_B32)
# logicalIR: VPermlane16SwapB32
VPermlane16SwapB32 = _make_scalar_unary_class("VPermlane16SwapB32", "v_permlane16_swap_b32", InstType.INST_B32)
# logicalIR: VPermlane32SwapB32
VPermlane32SwapB32 = _make_scalar_unary_class("VPermlane32SwapB32", "v_permlane32_swap_b32", InstType.INST_B32)
class SSchedulingFence(Instruction):
    """SSchedulingFence — scheduling barrier pseudo-instruction.

    Emits a SchedulingFence LogicalIR instruction that lowers to a FENCE
    in the hardware scheduler (no assembly output, just a DAG barrier).
    """

    __slots__ = ()

    def __init__(self, comment=""):
        super().__init__(InstType.INST_NOTYPE, comment)
        self.setInst("scheduling_fence")

    def getParams(self):
        return []

    def getDstParams(self):
        return []

    def getSrcParams(self):
        return []

    def to_stinky_logical(self, _module=None):
        import stinkytofu as st
        return st.SchedulingFence(comment=self.comment)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = SSchedulingFence(comment=self.comment)
        memo[id(self)] = dup
        return dup


# ==========================================================================
# Conversion instructions
# source: rocisa/rocisa/src/instruction/cvt.cpp
# ==========================================================================


def _make_cvt_scale_class(class_name: str, mnemonic: str, inst_type: "InstType"):
    """Factory for scale CVT shim classes with (dst, src, scale) rocisa API."""

    def __init__(self, dst: Any, src: Any = None, scale: Any = None,
                 sdwa: Any = None, vop3: Any = None, comment: str = "", **kw):
        _ = kw
        CommonInstruction.__init__(
            self,
            instType=inst_type,
            dst=dst,
            srcs=[src, scale],
            dpp=None,
            sdwa=sdwa,
            vop3=vop3,
            comment=comment,
        )
        self.setInst(mnemonic)

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433

        dst_reg = _to_stinky_register(self.dst)
        src_reg = _to_stinky_register(self.srcs[0])
        scale_reg = _to_stinky_register(self.srcs[1])
        factory = getattr(_st, class_name)
        return factory(dst_reg, src_reg, scale_reg, comment=self.comment)

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (VCvtInstruction,), {
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
    })
    cls.__qualname__ = class_name
    return cls


# VCvtInstruction: distinct subclass of CommonInstruction so KernelWriter's
# ``isinstance(item, VCvtInstruction)`` discrimination (e.g. the epilogue store
# rearrange loop in globalWriteBatch) matches only genuine CVT ops -- mirrors
# rocisa's cvt.hpp ``struct VCvtInstruction : public CommonInstruction``.
class VCvtInstruction(CommonInstruction):
    __slots__ = ()

# --- Unary CVTs: (dst, src) → logicalIR(dst, src) ---
VCvtF16toF32 = _make_scalar_unary_class("VCvtF16toF32", "v_cvt_f32_f16", InstType.INST_NOTYPE, base=VCvtInstruction)
VCvtF32toF16 = _make_scalar_unary_class("VCvtF32toF16", "v_cvt_f16_f32", InstType.INST_NOTYPE, base=VCvtInstruction)
VCvtF32toU32 = _make_scalar_unary_class("VCvtF32toU32", "v_cvt_u32_f32", InstType.INST_NOTYPE, base=VCvtInstruction)
VCvtU32toF32 = _make_scalar_unary_class("VCvtU32toF32", "v_cvt_f32_u32", InstType.INST_NOTYPE, base=VCvtInstruction)
VCvtI32toF32 = _make_scalar_unary_class("VCvtI32toF32", "v_cvt_f32_i32", InstType.INST_NOTYPE, base=VCvtInstruction)
VCvtF32toI32 = _make_scalar_unary_class("VCvtF32toI32", "v_cvt_i32_f32", InstType.INST_NOTYPE, base=VCvtInstruction)
VCvtFP8toF32 = _make_scalar_unary_class("VCvtFP8toF32", "v_cvt_f32_fp8", InstType.INST_NOTYPE, base=VCvtInstruction)
VCvtBF8toF32 = _make_scalar_unary_class("VCvtBF8toF32", "v_cvt_f32_bf8", InstType.INST_NOTYPE, base=VCvtInstruction)
VCvtPkFP8toF32 = _make_scalar_unary_class("VCvtPkFP8toF32", "v_cvt_pk_f32_fp8", InstType.INST_NOTYPE, base=VCvtInstruction)
VCvtPkBF8toF32 = _make_scalar_unary_class("VCvtPkBF8toF32", "v_cvt_pk_f32_bf8", InstType.INST_NOTYPE, base=VCvtInstruction)

# --- Binary CVTs: (dst, src0, src1) → logicalIR(dst, src0, src1) ---
VCvtPkF32toBF8 = _make_scalar_alu_class("VCvtPkF32toBF8", "v_cvt_pk_bf8_f32", InstType.INST_NOTYPE, base=VCvtInstruction)
VCvtSRF32toFP8 = _make_scalar_alu_class("VCvtSRF32toFP8", "v_cvt_sr_fp8_f32", InstType.INST_NOTYPE, base=VCvtInstruction)
VCvtSRF32toBF8 = _make_scalar_alu_class("VCvtSRF32toBF8", "v_cvt_sr_bf8_f32", InstType.INST_NOTYPE, base=VCvtInstruction)
VCvtPkF32toFP8 = _make_scalar_alu_class("VCvtPkF32toFP8", "v_cvt_pk_fp8_f32", InstType.INST_NOTYPE, base=VCvtInstruction)
VCvtPkF32toBF16 = _make_scalar_alu_class("VCvtPkF32toBF16", "v_cvt_pk_bf16_f32", InstType.INST_NOTYPE, base=VCvtInstruction)

# --- Scale CVTs: (dst, src, scale) → logicalIR(dst, src, scale) ---
VCvtScalePkFP8toF16 = _make_cvt_scale_class("VCvtScalePkFP8toF16", "v_cvt_scalef32_pk_f16_fp8", InstType.INST_NOTYPE)
VCvtScalePkBF8toF16 = _make_cvt_scale_class("VCvtScalePkBF8toF16", "v_cvt_scalef32_pk_f16_bf8", InstType.INST_NOTYPE)
VCvtScaleFP8toF16 = _make_cvt_scale_class("VCvtScaleFP8toF16", "v_cvt_scalef32_f16_fp8", InstType.INST_NOTYPE)
VCvtScalePkF16toFP8 = _make_cvt_scale_class("VCvtScalePkF16toFP8", "v_cvt_scalef32_pk_fp8_f16", InstType.INST_NOTYPE)
VCvtScalePkF16toBF8 = _make_cvt_scale_class("VCvtScalePkF16toBF8", "v_cvt_scalef32_pk_bf8_f16", InstType.INST_NOTYPE)
VCvtScaleSRF16toFP8 = _make_cvt_scale_class("VCvtScaleSRF16toFP8", "v_cvt_scalef32_sr_fp8_f16", InstType.INST_NOTYPE)
VCvtScaleSRF16toBF8 = _make_cvt_scale_class("VCvtScaleSRF16toBF8", "v_cvt_scalef32_sr_bf8_f16", InstType.INST_NOTYPE)
VCvtScalePk8F32toFP8 = _make_cvt_scale_class("VCvtScalePk8F32toFP8", "v_cvt_scalef32_pk8_fp8_f32", InstType.INST_NOTYPE)
VCvtScalePk8F32toBF8 = _make_cvt_scale_class("VCvtScalePk8F32toBF8", "v_cvt_scalef32_pk8_bf8_f32", InstType.INST_NOTYPE)


def _make_cvt_scale_sr_class(class_name: str, mnemonic: str, inst_type: "InstType"):
    """Factory for stochastic-rounding scale CVT shims with (dst, src0, src1, scale) API."""

    def __init__(self, dst: Any, src0: Any = None, src1: Any = None,
                 scale: Any = None, comment: str = "", **kw):
        _ = kw
        CommonInstruction.__init__(
            self,
            instType=inst_type,
            dst=dst,
            srcs=[src0, src1, scale],
            dpp=None,
            sdwa=None,
            vop3=None,
            comment=comment,
        )
        self.setInst(mnemonic)

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433
        dst_reg = _to_stinky_register(self.dst)
        src0_reg = _to_stinky_register(self.srcs[0])
        src1_reg = _to_stinky_register(self.srcs[1])
        scale_reg = _to_stinky_register(self.srcs[2])
        factory = getattr(_st, class_name)
        return factory(dst_reg, src0_reg, src1_reg, scale_reg, comment=self.comment)

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (VCvtInstruction,), {
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
    })
    cls.__qualname__ = class_name
    return cls


VCvtScaleSRPkF32toFP8 = _make_cvt_scale_sr_class("VCvtScaleSRPkF32toFP8", "v_cvt_scalef32_sr_pk8_fp8_f32", InstType.INST_NOTYPE)


# ==========================================================================
# FlatAtomicDecU32 -- flat atomic decrement (MBSK GSU kernel).
# ==========================================================================
class FlatAtomicDecU32(CommonInstruction):
    """``flat_atomic_dec_u32 dst, addr, data`` -- flat atomic decrement."""

    def __init__(self, dst: Any, addr: Any, data: Any,
                 modifier: Any = None, comment: str = ""):
        super().__init__(
            instType=InstType.INST_B32,
            dst=dst,
            srcs=[addr, data],
            dpp=None,
            sdwa=None,
            vop3=None,
            comment=comment,
        )
        self.setInst("flat_atomic_dec_u32")
        self.flat = modifier

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433
        dst_reg = _to_stinky_register(self.dst)
        addr_reg = _to_stinky_register(self.srcs[0])
        data_reg = _to_stinky_register(self.srcs[1])
        return _st.FlatAtomicDecU32(dst_reg, addr_reg, data_reg, comment=self.comment)

    def __deepcopy__(self, memo):
        clone = CommonInstruction.__deepcopy__(self, memo)
        return clone


# ==========================================================================
# GlobalAtomicIncU32Saddr -- SADDR-returning global atomic increment (StreamK).
# ==========================================================================
class GlobalAtomicIncU32Saddr(CommonInstruction):
    """``global_atomic_inc_u32 dst, vaddr, data, saddr`` -- SADDR atomic inc."""

    def __init__(self, dst: Any, vaddr: Any, data: Any, saddr: Any,
                 modifier: Any = None, comment: str = ""):
        super().__init__(
            instType=InstType.INST_B32,
            dst=dst,
            srcs=[vaddr, data, saddr],
            dpp=None,
            sdwa=None,
            vop3=None,
            comment=comment,
        )
        self.setInst("global_atomic_inc_u32")
        self.glob = modifier

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st  # noqa: WPS433
        dst_reg = _to_stinky_register(self.dst)
        vaddr_reg = _to_stinky_register(self.srcs[0])
        data_reg = _to_stinky_register(self.srcs[1])
        saddr_reg = _to_stinky_register(self.srcs[2])
        return _st.GlobalAtomicIncU32Saddr(dst_reg, vaddr_reg, data_reg, saddr_reg,
                                           comment=self.comment)

    def __deepcopy__(self, memo):
        clone = CommonInstruction.__deepcopy__(self, memo)
        return clone


# --- Gfx1250 vector conversions ---
# logicalIR: VCvtPkF32toF16
VCvtPkF32toF16 = _make_scalar_alu_class("VCvtPkF32toF16", "v_cvt_pk_f16_f32", InstType.INST_NOTYPE)
# logicalIR: VCvtF64toU32
VCvtF64toU32 = _make_scalar_unary_class("VCvtF64toU32", "v_cvt_u32_f64", InstType.INST_U32)
# logicalIR: VCvtU32toF64
VCvtU32toF64 = _make_scalar_unary_class("VCvtU32toF64", "v_cvt_f64_u32", InstType.INST_F64)
# logicalIR: PVCvtBF16toFP32
PVCvtBF16toFP32 = _make_scalar_unary_class("PVCvtBF16toFP32", "v_cvt_f32_bf16", InstType.INST_F32)
# logicalIR: VCvtPkF32toFP16
VCvtPkF32toFP16 = _make_scalar_alu_class("VCvtPkF32toFP16", "v_cvt_pk_f16_f32", InstType.INST_NOTYPE)
# logicalIR: VCvtFP8toF16
VCvtFP8toF16 = _make_scalar_unary_class("VCvtFP8toF16", "v_cvt_f16_fp8", InstType.INST_F16)


# ==========================================================================
# Memory (Buffer/Flat/Global/DS/SMEM) instructions
# source: rocisa/rocisa/src/instruction/mem.cpp
# ==========================================================================


def _make_buffer_load_class(class_name: str, mnemonic: str, latency: int = 1, base: type = None):
    """Factory for MUBUF load shims: rocisa(dst, vaddr, saddr, soffset, mubuf, comment)."""
    if base is None:
        base = MUBUFReadInstruction

    def __init__(self, dst: Any = None, vaddr: Any = None, saddr: Any = None,
                 soffset: Any = None, mubuf: Any = None, comment: str = "", **kw):
        _ = kw
        CommonInstruction.__init__(
            self, instType=InstType.INST_NOTYPE, dst=dst,
            srcs=[vaddr, saddr, soffset], dpp=None, sdwa=None, vop3=None,
            comment=comment)
        self.setInst(mnemonic)
        self.mubuf = mubuf

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st
        factory = getattr(_st, class_name)
        inst = factory(
            _to_stinky_register(self.dst),
            _to_stinky_register(self.srcs[0]),
            comment=self.comment)
        if self.srcs[1] is not None:
            inst.add_src(_to_stinky_register(self.srcs[1]))
        # rocisa (mem.hpp MUBUFReadInstruction::getArgStr) renders a soffset
        # whose operand string is "0" as ``null`` when HasMUBUFConst is false
        # (true for gfx12+). Tensile passes literal 0 for "no scalar offset",
        # which otherwise assembles to the invalid operand ``0``.
        _soffset = self.srcs[2]
        if _soffset is not None and _input_to_str(_soffset) != "0":
            inst.add_src(_to_stinky_register(_soffset))
        elif self.srcs[1] is not None:
            inst.add_src(_st.Register("null"))
        if self.mubuf is not None:
            inst.set_mubuf(
                offen=getattr(self.mubuf, "offen", False),
                offset=getattr(self.mubuf, "offset12", 0),
                glc=getattr(self.mubuf, "glc", False),
                slc=getattr(self.mubuf, "slc", False),
                nt=getattr(self.mubuf, "nt", False),
                scope=getattr(self.mubuf, "scope", 0) if isinstance(getattr(self.mubuf, "scope", 0), int) else getattr(self.mubuf, "scope", 0).value,
                th=int(getattr(self.mubuf, "th", -1)),
                is_store=getattr(self.mubuf, "isStore", False),
            )
        return inst

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (base,), {
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
        "issueLatency": staticmethod(lambda: latency),
    })
    cls.__qualname__ = class_name
    return cls


def _make_buffer_store_class(class_name: str, mnemonic: str, latency: int = 1, base: type = None):
    """Factory for MUBUF store/atomic shims: rocisa(src, vaddr, saddr, soffset, mubuf, comment)."""
    if base is None:
        base = MUBUFStoreInstruction

    def __init__(self, src: Any = None, vaddr: Any = None, saddr: Any = None,
                 soffset: Any = None, mubuf: Any = None, comment: str = "", **kw):
        _ = kw
        CommonInstruction.__init__(
            self, instType=InstType.INST_NOTYPE, dst=src,
            srcs=[vaddr, saddr, soffset], dpp=None, sdwa=None, vop3=None,
            comment=comment)
        self.setInst(mnemonic)
        self.mubuf = mubuf

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st
        factory = getattr(_st, class_name)
        inst = factory(
            _to_stinky_register(self.dst),
            _to_stinky_register(self.srcs[0]),
            _to_stinky_register(self.srcs[1]),
            comment=self.comment)
        # rocisa (mem.hpp MUBUFReadInstruction::getArgStr) renders a soffset
        # whose operand string is "0" as ``null`` when HasMUBUFConst is false
        # (true for gfx12+). Tensile passes literal 0 for "no scalar offset",
        # which otherwise assembles to the invalid operand ``0``.
        _soffset = self.srcs[2]
        if _soffset is not None and _input_to_str(_soffset) != "0":
            inst.add_src(_to_stinky_register(_soffset))
        elif self.srcs[1] is not None:
            inst.add_src(_st.Register("null"))
        if self.mubuf is not None:
            inst.set_mubuf(
                offen=getattr(self.mubuf, "offen", False),
                offset=getattr(self.mubuf, "offset12", 0),
                glc=getattr(self.mubuf, "glc", False),
                slc=getattr(self.mubuf, "slc", False),
                nt=getattr(self.mubuf, "nt", False),
                scope=getattr(self.mubuf, "scope", 0) if isinstance(getattr(self.mubuf, "scope", 0), int) else getattr(self.mubuf, "scope", 0).value,
                th=int(getattr(self.mubuf, "th", -1)),
                is_store=getattr(self.mubuf, "isStore", False),
            )
        return inst

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (base,), {
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
        "issueLatency": staticmethod(lambda: latency),
    })
    cls.__qualname__ = class_name
    return cls


def _make_flat_load_class(class_name: str, mnemonic: str, latency: int = 1, base: type = None):
    """Factory for Flat load shims: rocisa(dst, vaddr, flat, comment)."""
    if base is None:
        base = FLATReadInstruction

    def __init__(self, dst: Any = None, vaddr: Any = None,
                 flat: Any = None, comment: str = "", **kw):
        _ = kw
        CommonInstruction.__init__(
            self, instType=InstType.INST_NOTYPE, dst=dst,
            srcs=[vaddr], dpp=None, sdwa=None, vop3=None, comment=comment)
        self.setInst(mnemonic)
        self.flat = flat

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st
        factory = getattr(_st, class_name)
        return factory(
            _to_stinky_register(self.dst),
            _to_stinky_register(self.srcs[0]),
            comment=self.comment)

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (base,), {
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
        "issueLatency": staticmethod(lambda: latency),
    })
    cls.__qualname__ = class_name
    return cls


def _make_flat_store_class(class_name: str, mnemonic: str, latency: int = 1, base: type = None):
    """Factory for Flat store shims: rocisa(src, vaddr, flat, comment)."""
    if base is None:
        base = FLATStoreInstruction

    def __init__(self, src: Any = None, vaddr: Any = None,
                 flat: Any = None, comment: str = "", **kw):
        _ = kw
        CommonInstruction.__init__(
            self, instType=InstType.INST_NOTYPE, dst=src,
            srcs=[vaddr], dpp=None, sdwa=None, vop3=None, comment=comment)
        self.setInst(mnemonic)
        self.flat = flat

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st
        factory = getattr(_st, class_name)
        return factory(
            _to_stinky_register(self.dst),
            _to_stinky_register(self.srcs[0]),
            _to_stinky_register(self.dst),
            comment=self.comment)

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (base,), {
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
        "issueLatency": staticmethod(lambda: latency),
    })
    cls.__qualname__ = class_name
    return cls


def _make_flat_atomic_class(class_name: str, mnemonic: str, latency: int = 1, base: type = None):
    """Factory for Flat atomic shims: rocisa(vaddr, tmp, src, flat, comment)."""
    if base is None:
        base = AtomicReadWriteInstruction

    def __init__(self, vaddr: Any = None, tmp: Any = None, src: Any = None,
                 flat: Any = None, comment: str = "", **kw):
        _ = kw
        CommonInstruction.__init__(
            self, instType=InstType.INST_NOTYPE, dst=vaddr,
            srcs=[tmp, src], dpp=None, sdwa=None, vop3=None, comment=comment)
        self.setInst(mnemonic)
        self.flat = flat

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st
        factory = getattr(_st, class_name)
        return factory(
            _to_stinky_register(self.dst),
            _to_stinky_register(self.srcs[0]),
            _to_stinky_register(self.srcs[1]),
            comment=self.comment)

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (base,), {
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
        "issueLatency": staticmethod(lambda: latency),
    })
    cls.__qualname__ = class_name
    return cls


def _make_ds_load_class(class_name: str, mnemonic: str, latency: int = 1, base: type = None):
    """Factory for DS load shims: rocisa(dst, src, ds, comment)."""
    if base is None:
        base = DSLoadInstruction

    def __init__(self, dst: Any = None, src: Any = None,
                 ds: Any = None, comment: str = "", **kw):
        _ = kw
        CommonInstruction.__init__(
            self, instType=InstType.INST_NOTYPE, dst=dst,
            srcs=[src], dpp=None, sdwa=None, vop3=None, comment=comment)
        self.setInst(mnemonic)
        self.ds = ds

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st
        factory = getattr(_st, class_name)
        inst = factory(
            _to_stinky_register(self.dst),
            _to_stinky_register(self.srcs[0]),
            comment=self.comment)
        if self.ds is not None:
            inst.set_ds(offset=getattr(self.ds, "offset", 0))
        return inst

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (base,), {
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
        "issueLatency": staticmethod(lambda: latency),
    })
    cls.__qualname__ = class_name
    return cls


def _make_ds_store_class(class_name: str, mnemonic: str, latency: int = 1, base: type = None):
    """Factory for DS store (binary) shims: rocisa(dstAddr, src, ds, comment)."""
    if base is None:
        base = DSStoreInstruction

    def __init__(self, dstAddr: Any = None, src: Any = None,
                 ds: Any = None, comment: str = "", **kw):
        _ = kw
        CommonInstruction.__init__(
            self, instType=InstType.INST_NOTYPE, dst=dstAddr,
            srcs=[src], dpp=None, sdwa=None, vop3=None, comment=comment)
        self.setInst(mnemonic)
        self.ds = ds

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st
        factory = getattr(_st, class_name)
        inst = factory(
            _to_stinky_register(self.dst),
            _to_stinky_register(self.srcs[0]),
            comment=self.comment)
        if self.ds is not None:
            inst.set_ds(offset=getattr(self.ds, "offset", 0))
        return inst

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (base,), {
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
        "issueLatency": staticmethod(lambda: latency),
    })
    cls.__qualname__ = class_name
    return cls


def _make_ds_store2_class(class_name: str, mnemonic: str, latency: int = 1, base: type = None):
    """Factory for DS store2/permute (ternary) shims: rocisa(dstAddr, src0, src1, ds, comment)."""
    if base is None:
        base = DSStoreInstruction

    def __init__(self, dstAddr: Any = None, src0: Any = None, src1: Any = None,
                 ds: Any = None, comment: str = "", **kw):
        _ = kw
        CommonInstruction.__init__(
            self, instType=InstType.INST_NOTYPE, dst=dstAddr,
            srcs=[src0, src1], dpp=None, sdwa=None, vop3=None, comment=comment)
        self.setInst(mnemonic)
        self.ds = ds

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st
        factory = getattr(_st, class_name)
        inst = factory(
            _to_stinky_register(self.dst),
            _to_stinky_register(self.srcs[0]),
            _to_stinky_register(self.srcs[1]),
            comment=self.comment)
        if self.ds is not None:
            inst.set_ds(offset=getattr(self.ds, "offset", 0))
        return inst

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (base,), {
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
        "issueLatency": staticmethod(lambda: latency),
    })
    cls.__qualname__ = class_name
    return cls


# Memory instruction category hierarchy. These MUST be distinct classes (not
# aliases of CommonInstruction) because KernelWriter relies on ``isinstance``
# to discriminate categories -- e.g. globalWriteBatch counts
# ``GlobalReadInstruction`` items to seed the load waitcnt and decrements it for
# each ``DSStoreInstruction`` / ``VCvtInstruction`` in the store rearrange loop.
# Collapsing them to CommonInstruction made every instruction match, corrupting
# the waitcnt counts (negative ``s_waitcnt``). Mirrors rocisa mem.hpp:
#   ReadWriteInstruction
#     GlobalReadInstruction (FLATReadInstruction, GLOBALLoadInstruction, MUBUFReadInstruction)
#     AtomicReadWriteInstruction
#     GlobalWriteInstruction (FLATStoreInstruction, MUBUFStoreInstruction)
#     LocalReadInstruction (DSLoadInstruction)
#     LocalWriteInstruction (DSStoreInstruction)
class ReadWriteInstruction(CommonInstruction):
    __slots__ = ()


class GlobalReadInstruction(ReadWriteInstruction):
    __slots__ = ()


class FLATReadInstruction(GlobalReadInstruction):
    __slots__ = ()


class GLOBALLoadInstruction(GlobalReadInstruction):
    __slots__ = ()


class MUBUFReadInstruction(GlobalReadInstruction):
    __slots__ = ()


class AtomicReadWriteInstruction(ReadWriteInstruction):
    __slots__ = ()


SMemAtomicIncInstruction = make_dummy_class(f"{_P}.SMemAtomicIncInstruction")
SMemAtomicDecInstruction = make_dummy_class(f"{_P}.SMemAtomicDecInstruction")


# SMemLoadInstruction / SLoadB* — real classes (see above after SMovB64).
class GlobalWriteInstruction(ReadWriteInstruction):
    __slots__ = ()


SMemStoreInstruction = make_dummy_class(f"{_P}.SMemStoreInstruction")


class FLATStoreInstruction(GlobalWriteInstruction):
    __slots__ = ()


class MUBUFStoreInstruction(GlobalWriteInstruction):
    __slots__ = ()


class LocalReadInstruction(ReadWriteInstruction):
    __slots__ = ()


class DSLoadInstruction(LocalReadInstruction):
    __slots__ = ()


class LocalWriteInstruction(ReadWriteInstruction):
    __slots__ = ()


class DSStoreInstruction(LocalWriteInstruction):
    __slots__ = ()

# --- Buffer Load (MUBUF): rocisa(dst, vaddr, saddr, soffset, mubuf, comment) ---
BufferLoadU8 = _make_buffer_load_class("BufferLoadU8", "buffer_load_u8")
BufferLoadI8 = _make_buffer_load_class("BufferLoadI8", "buffer_load_i8")
BufferLoadD16HIU8 = _make_buffer_load_class("BufferLoadD16HIU8", "buffer_load_d16_hi_u8")
BufferLoadD16U8 = _make_buffer_load_class("BufferLoadD16U8", "buffer_load_d16_u8")
BufferLoadD16I8 = _make_buffer_load_class("BufferLoadD16I8", "buffer_load_d16_i8")
BufferLoadD16HII8 = _make_buffer_load_class("BufferLoadD16HII8", "buffer_load_d16_hi_i8")
BufferLoadD16HIB16 = _make_buffer_load_class("BufferLoadD16HIB16", "buffer_load_d16_hi_b16")
BufferLoadD16B16 = _make_buffer_load_class("BufferLoadD16B16", "buffer_load_d16_b16")
# logicalIR: BufferLoadB16
BufferLoadB16 = _make_buffer_load_class("BufferLoadB16", "buffer_load_b16")
BufferLoadI16 = _make_buffer_load_class("BufferLoadI16", "buffer_load_i16")
BufferLoadU16 = _make_buffer_load_class("BufferLoadU16", "buffer_load_u16")
BufferLoadB32 = _make_buffer_load_class("BufferLoadB32", "buffer_load_b32")
BufferLoadB64 = _make_buffer_load_class("BufferLoadB64", "buffer_load_b64")
BufferLoadB96 = _make_buffer_load_class("BufferLoadB96", "buffer_load_b96")
BufferLoadB128 = _make_buffer_load_class("BufferLoadB128", "buffer_load_b128")
# logicalIR: BufferLoadB192
BufferLoadB192 = _make_buffer_load_class("BufferLoadB192", "buffer_load_b192")

# --- Flat Load: rocisa(dst, vaddr, flat, comment) ---
FlatLoadU8 = _make_flat_load_class("FlatLoadU8", "flat_load_u8")
FlatLoadI8 = _make_flat_load_class("FlatLoadI8", "flat_load_i8")
FlatLoadD16HIU8 = _make_flat_load_class("FlatLoadD16HIU8", "flat_load_d16_hi_u8")
FlatLoadD16U8 = _make_flat_load_class("FlatLoadD16U8", "flat_load_d16_u8")
FlatLoadD16I8 = _make_flat_load_class("FlatLoadD16I8", "flat_load_d16_i8")
FlatLoadD16HII8 = _make_flat_load_class("FlatLoadD16HII8", "flat_load_d16_hi_i8")
FlatLoadD16HIB16 = _make_flat_load_class("FlatLoadD16HIB16", "flat_load_d16_hi_b16")
FlatLoadD16B16 = _make_flat_load_class("FlatLoadD16B16", "flat_load_d16_b16")
FlatLoadU16 = _make_flat_load_class("FlatLoadU16", "flat_load_u16")
FlatLoadI16 = _make_flat_load_class("FlatLoadI16", "flat_load_i16")
FlatLoadB32 = _make_flat_load_class("FlatLoadB32", "flat_load_b32")
FlatLoadB64 = _make_flat_load_class("FlatLoadB64", "flat_load_b64")
FlatLoadB96 = _make_flat_load_class("FlatLoadB96", "flat_load_b96")
FlatLoadB128 = _make_flat_load_class("FlatLoadB128", "flat_load_b128")
# logicalIR: FlatLoadB192
FlatLoadB192 = _make_flat_load_class("FlatLoadB192", "flat_load_b192")
# --- Global Load: rocisa(dst, vaddr, saddr, modifier, comment) ---
def _make_global_load_class(class_name: str, mnemonic: str, latency: int = 1, base: type = None):
    """Factory for Global load shims: rocisa(dst, vaddr, saddr, modifier, comment)."""
    if base is None:
        base = GLOBALLoadInstruction

    def __init__(self, dst: Any = None, vaddr: Any = None,
                 saddr: Any = None, modifier: Any = None, comment: str = "", **kw):
        _ = kw
        CommonInstruction.__init__(
            self, instType=InstType.INST_NOTYPE, dst=dst,
            srcs=[vaddr, saddr], dpp=None, sdwa=None, vop3=None, comment=comment)
        self.setInst(mnemonic)
        self._modifier = modifier

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st
        factory = getattr(_st, class_name)
        return factory(
            _to_stinky_register(self.dst),
            _to_stinky_register(self.srcs[0]),
            _to_stinky_register(self.srcs[1]),
            comment=self.comment)

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (base,), {
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
        "issueLatency": staticmethod(lambda: latency),
    })
    cls.__qualname__ = class_name
    return cls


GlobalLoadB32 = _make_global_load_class("GlobalLoadB32", "global_load_b32")
GlobalLoadB64 = _make_global_load_class("GlobalLoadB64", "global_load_b64")
GlobalLoadB96 = _make_global_load_class("GlobalLoadB96", "global_load_b96")
GlobalLoadB128 = _make_global_load_class("GlobalLoadB128", "global_load_b128")
GlobalLoadB192 = _make_global_load_class("GlobalLoadB192", "global_load_b192")
GlobalLoadD16B16 = _make_global_load_class("GlobalLoadD16B16", "global_load_d16_b16")
GlobalLoadD16HIB16 = _make_global_load_class("GlobalLoadD16HIB16", "global_load_d16_hi_b16")
GlobalLoadD16U8 = _make_global_load_class("GlobalLoadD16U8", "global_load_d16_u8")
GlobalLoadD16HIU8 = _make_global_load_class("GlobalLoadD16HIU8", "global_load_d16_hi_u8")


# --- Global Store: rocisa(vaddr, src, saddr, modifier, comment) ---
def _make_global_store_class(class_name: str, mnemonic: str, latency: int = 1, base: type = None):
    """Factory for Global store shims: rocisa(vaddr, src, saddr, modifier, comment)."""
    if base is None:
        base = GlobalWriteInstruction

    def __init__(self, vaddr: Any = None, src: Any = None,
                 saddr: Any = None, modifier: Any = None, comment: str = "", **kw):
        _ = kw
        CommonInstruction.__init__(
            self, instType=InstType.INST_NOTYPE, dst=src,
            srcs=[vaddr, saddr], dpp=None, sdwa=None, vop3=None, comment=comment)
        self.setInst(mnemonic)
        self._modifier = modifier

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st
        factory = getattr(_st, class_name)
        return factory(
            _to_stinky_register(self.dst),
            _to_stinky_register(self.srcs[0]),
            _to_stinky_register(self.srcs[1]),
            comment=self.comment)

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (base,), {
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
        "issueLatency": staticmethod(lambda: latency),
    })
    cls.__qualname__ = class_name
    return cls


GlobalStoreB8 = _make_global_store_class("GlobalStoreB8", "global_store_b8")
GlobalStoreB16 = _make_global_store_class("GlobalStoreB16", "global_store_b16")
GlobalStoreD16HIB16 = _make_global_store_class("GlobalStoreD16HIB16", "global_store_d16_hi_b16")
GlobalStoreB32 = _make_global_store_class("GlobalStoreB32", "global_store_b32")
GlobalStoreB64 = _make_global_store_class("GlobalStoreB64", "global_store_b64")
GlobalStoreB128 = _make_global_store_class("GlobalStoreB128", "global_store_b128")


# logicalIR: GlobalLoadTR8B64
def _make_global_load_tr_class(class_name: str, mnemonic: str):
    """Factory for global_load_tr* shims: rocisa(dst, vaddr, saddr, modifier, comment)."""

    def __init__(self, dst: Any = None, vaddr: Any = None,
                 saddr: Any = None, modifier: Any = None, comment: str = "", **kw):
        _ = kw
        CommonInstruction.__init__(
            self, instType=InstType.INST_NOTYPE, dst=dst,
            srcs=[vaddr, saddr], dpp=None, sdwa=None, vop3=None, comment=comment)
        self.setInst(mnemonic)
        self._modifier = modifier

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st
        factory = getattr(_st, class_name)
        return factory(
            _to_stinky_register(self.dst),
            _to_stinky_register(self.srcs[0]),
            _to_stinky_register(self.srcs[1]),
            self.comment)

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (CommonInstruction,), {
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
    })
    return cls


GlobalLoadTR8B64 = _make_global_load_tr_class("GlobalLoadTR8B64", "global_load_tr_b64_b8")
# logicalIR: GlobalLoadTR16B128
GlobalLoadTR16B128 = _make_global_load_tr_class("GlobalLoadTR16B128", "global_load_tr_b128_b16")

# --- Buffer Store / Atomic: rocisa(src, vaddr, saddr, soffset, mubuf, comment) ---
BufferStoreB8 = _make_buffer_store_class("BufferStoreB8", "buffer_store_b8")
BufferStoreD16HIU8 = _make_buffer_store_class("BufferStoreD16HIU8", "buffer_store_d16_hi_b8")
# logicalIR: BufferStoreD16U8
BufferStoreD16U8 = _make_buffer_store_class("BufferStoreD16U8", "buffer_store_d16_u8")
BufferStoreD16HIB16 = _make_buffer_store_class("BufferStoreD16HIB16", "buffer_store_d16_hi_b16")
# logicalIR: BufferStoreD16B16
BufferStoreD16B16 = _make_buffer_store_class("BufferStoreD16B16", "buffer_store_d16_b16")
BufferStoreB16 = _make_buffer_store_class("BufferStoreB16", "buffer_store_b16")
BufferStoreB32 = _make_buffer_store_class("BufferStoreB32", "buffer_store_b32")
BufferStoreB64 = _make_buffer_store_class("BufferStoreB64", "buffer_store_b64")
BufferStoreB96 = _make_buffer_store_class("BufferStoreB96", "buffer_store_b96")
BufferStoreB128 = _make_buffer_store_class("BufferStoreB128", "buffer_store_b128")
BufferAtomicAddF32 = _make_buffer_load_class("BufferAtomicAddF32", "buffer_atomic_add_f32")
BufferAtomicCmpswapB32 = _make_buffer_store_class("BufferAtomicCmpswapB32", "buffer_atomic_cmpswap_b32")
BufferAtomicCmpswapB64 = _make_buffer_store_class("BufferAtomicCmpswapB64", "buffer_atomic_cmpswap_b64")

# --- Flat Store: rocisa(src, vaddr, flat, comment) ---
FlatStoreB8 = _make_flat_store_class("FlatStoreB8", "flat_store_b8")
FlatStoreD16HIB8 = _make_flat_store_class("FlatStoreD16HIB8", "flat_store_d16_hi_b8")
FlatStoreB16 = _make_flat_store_class("FlatStoreB16", "flat_store_b16")
FlatStoreD16HIB16 = _make_flat_store_class("FlatStoreD16HIB16", "flat_store_d16_hi_b16")
# logicalIR: FlatStoreD16B16
FlatStoreD16B16 = _make_flat_store_class("FlatStoreD16B16", "flat_store_d16_b16")
FlatStoreB32 = _make_flat_store_class("FlatStoreB32", "flat_store_b32")
FlatStoreB64 = _make_flat_store_class("FlatStoreB64", "flat_store_b64")
FlatStoreB96 = _make_flat_store_class("FlatStoreB96", "flat_store_b96")
FlatStoreB128 = _make_flat_store_class("FlatStoreB128", "flat_store_b128")

# --- Flat Atomic: rocisa(vaddr, tmp, src, flat, comment) ---
FlatAtomicCmpswapB32 = _make_flat_atomic_class("FlatAtomicCmpswapB32", "flat_atomic_cmpswap_b32")

# --- DS Load: rocisa(dst, src, ds, comment) ---
DSLoadU8 = _make_ds_load_class("DSLoadU8", "ds_load_u8")
DSLoadI8 = _make_ds_load_class("DSLoadI8", "ds_load_i8")
# logicalIR: DSLoadD16HIU8
DSLoadD16HIU8 = _make_ds_load_class("DSLoadD16HIU8", "ds_load_d16_hi_u8")
DSLoadU16 = _make_ds_load_class("DSLoadU16", "ds_load_u16")
DSLoadI16 = _make_ds_load_class("DSLoadI16", "ds_load_i16")
# logicalIR: DSLoadD16HIU16
DSLoadD16HIU16 = _make_ds_load_class("DSLoadD16HIU16", "ds_load_d16_hi_u16")
# logicalIR: DSLoadB16
DSLoadB16 = _make_ds_load_class("DSLoadB16", "ds_load_b16")
DSLoadB32 = _make_ds_load_class("DSLoadB32", "ds_load_b32")
DSLoadB64 = _make_ds_load_class("DSLoadB64", "ds_load_b64")
DSLoadB96 = _make_ds_load_class("DSLoadB96", "ds_load_b96")
# logicalIR: DSLoadB96TrB6
DSLoadB96TrB6 = _make_ds_load_class("DSLoadB96TrB6", "ds_load_tr6_b96")
# logicalIR: DSLoadB64TrB4
DSLoadB64TrB4 = _make_ds_load_class("DSLoadB64TrB4", "ds_load_tr4_b64")
# logicalIR: DSLoadB64TrB16
DSLoadB64TrB16 = _make_ds_load_class("DSLoadB64TrB16", "ds_load_tr16_b64")
# logicalIR: DSLoadB128TrB16
DSLoadB128TrB16 = _make_ds_load_class("DSLoadB128TrB16", "ds_load_tr16_b128")
# logicalIR: DSLoadB64TrB8
DSLoadB64TrB8 = _make_ds_load_class("DSLoadB64TrB8", "ds_load_tr8_b64")
DSLoadB128 = _make_ds_load_class("DSLoadB128", "ds_load_b128", latency=2)
# logicalIR: DSLoadB192
DSLoadB192 = _make_ds_load_class("DSLoadB192", "ds_load_b192", latency=2)
def _make_ds_load2_class(class_name: str, mnemonic: str, latency: int = 1, base: type = None):
    """Factory for DS load2 (dual-address): rocisa(dst, src, ds, comment) → logicalIR 3 args."""
    if base is None:
        base = DSLoadInstruction

    def __init__(self, dst: Any = None, src: Any = None,
                 ds: Any = None, comment: str = "", **kw):
        _ = kw
        CommonInstruction.__init__(
            self, instType=InstType.INST_NOTYPE, dst=dst,
            srcs=[src], dpp=None, sdwa=None, vop3=None, comment=comment)
        self.setInst(mnemonic)
        self.ds = ds

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st
        factory = getattr(_st, class_name)
        src_reg = _to_stinky_register(self.srcs[0])
        inst = factory(
            _to_stinky_register(self.dst),
            src_reg,
            src_reg,
            comment=self.comment)
        if self.ds is not None:
            inst.set_ds(
                na=getattr(self.ds, "na", 2),
                offset=getattr(self.ds, "offset", 0),
                offset0=getattr(self.ds, "offset0", 0),
                offset1=getattr(self.ds, "offset1", 0),
                gds=getattr(self.ds, "gds", False),
            )
        return inst

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (base,), {
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
        "issueLatency": staticmethod(lambda: latency),
    })
    cls.__qualname__ = class_name
    return cls


DSLoad2B32 = _make_ds_load2_class("DSLoad2B32", "ds_load2_b32")
DSLoad2B64 = _make_ds_load2_class("DSLoad2B64", "ds_load2_b64")

# --- DS Store (binary): rocisa(dstAddr, src, ds, comment) ---
# logicalIR: DSStoreU16
DSStoreU16 = _make_ds_store_class("DSStoreU16", "ds_store_u16")
DSStoreB8 = _make_ds_store_class("DSStoreB8", "ds_store_b8")
DSStoreB16 = _make_ds_store_class("DSStoreB16", "ds_store_b16")
# logicalIR: DSStoreB8HID16
DSStoreB8HID16 = _make_ds_store_class("DSStoreB8HID16", "ds_store_b8_d16_hi")
# logicalIR: DSStoreD16HIB16
DSStoreD16HIB16 = _make_ds_store_class("DSStoreD16HIB16", "ds_store_b16_d16_hi")
DSStoreB32 = _make_ds_store_class("DSStoreB32", "ds_store_b32", latency=2)
DSStoreB64 = _make_ds_store_class("DSStoreB64", "ds_store_b64", latency=3)
DSStoreB96 = _make_ds_store_class("DSStoreB96", "ds_store_b96", latency=4)
DSStoreB128 = _make_ds_store_class("DSStoreB128", "ds_store_b128", latency=5)
# logicalIR: DSStoreB192
DSStoreB192 = _make_ds_store_class("DSStoreB192", "ds_store_b192", latency=6)
# logicalIR: DSStoreB256
DSStoreB256 = _make_ds_store_class("DSStoreB256", "ds_store_b256", latency=7)

# --- DS Store2 / Permute (ternary): rocisa(dstAddr, src0, src1, ds, comment) ---
DSStore2B32 = _make_ds_store2_class("DSStore2B32", "ds_store2_b32", latency=3)
DSStore2B64 = _make_ds_store2_class("DSStore2B64", "ds_store2_b64", latency=3)


def _make_ds_permute_class(class_name: str, mnemonic: str, latency: int = 1, base: type = None):
    """Factory for DS permute: rocisa(dst, src0, src1, ds, comment) → logicalIR 2 args."""
    if base is None:
        # rocisa mem.hpp: DSBPermuteB32 : public DSStoreInstruction.
        base = DSStoreInstruction

    def __init__(self, dst: Any = None, src0: Any = None, src1: Any = None,
                 ds: Any = None, comment: str = "", **kw):
        _ = kw
        CommonInstruction.__init__(
            self, instType=InstType.INST_NOTYPE, dst=dst,
            srcs=[src0, src1], dpp=None, sdwa=None, vop3=None, comment=comment)
        self.setInst(mnemonic)
        self.ds = ds

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st
        factory = getattr(_st, class_name)
        # gfx1250 ds_bpermute_b32 needs 3 operands: vdst, addr(src0), data0(src1).
        # data0 defaults to dst when the caller omits it (matches native rocisa).
        data0 = self.srcs[1] if len(self.srcs) > 1 and self.srcs[1] is not None else self.dst
        inst = factory(
            _to_stinky_register(self.dst),
            _to_stinky_register(self.srcs[0]),
            _to_stinky_register(data0),
            comment=self.comment)
        if self.ds is not None:
            inst.set_ds(offset=getattr(self.ds, "offset", 0))
        return inst

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type(class_name, (base,), {
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
        "issueLatency": staticmethod(lambda: latency),
    })
    cls.__qualname__ = class_name
    return cls


DSBPermuteB32 = _make_ds_permute_class("DSBPermuteB32", "ds_bpermute_b32")

# --- SMEM Store / Atomic ---
# SStoreB32 … SStoreB512 — real classes (``SMemStoreInstruction`` subclasses, defined above).
# logicalIR: SAtomicInc
class SAtomicInc(Instruction):
    """``s_atomic_inc dst, base, soffset`` shim."""

    __slots__ = ("dst", "base", "soffset", "smem")

    def __init__(self, dst=None, base=None, soffset=None, smem=None, comment="", **kw):
        _ = kw
        super().__init__(InstType.INST_B32, comment)
        self.dst = dst
        self.base = base
        self.soffset = soffset
        self.smem = smem
        self.setInst("s_atomic_inc")

    def getParams(self):
        return [self.dst, self.base, self.soffset]

    def getDstParams(self):
        return [self.dst] if self.dst else []

    def getSrcParams(self):
        return [self.base, self.soffset]

    def toString(self) -> str:
        parts = [_input_to_str(self.dst), _input_to_str(self.base), _input_to_str(self.soffset)]
        kstr = self.instStr + " " + ", ".join(parts)
        if self.smem is not None and hasattr(self.smem, "toString"):
            kstr += self.smem.toString()
        return self.formatWithComment(kstr)

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st
        return _st.SAtomicInc(
            _to_stinky_register(self.dst),
            _to_stinky_register(self.base),
            _to_stinky_register(self.soffset),
            self.comment)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = self.__class__.__new__(self.__class__)
        memo[id(self)] = dup
        Instruction.__init__(dup, self.instType, self.comment)
        dup.instStr = self.instStr
        dup.dst = _deepcopy(self.dst, memo) if self.dst is not None else None
        dup.base = _deepcopy(self.base, memo) if self.base is not None else None
        dup.soffset = self.soffset if isinstance(self.soffset, (int, float, str, bool)) else _deepcopy(self.soffset, memo)
        dup.smem = _deepcopy(self.smem, memo) if self.smem is not None else None
        return dup


# logicalIR: SAtomicDec
class SAtomicDec(Instruction):
    """``s_atomic_dec dst, base`` shim (no soffset)."""

    __slots__ = ("dst", "base", "smem")

    def __init__(self, dst=None, base=None, smem=None, comment="", **kw):
        _ = kw
        super().__init__(InstType.INST_B32, comment)
        self.dst = dst
        self.base = base
        self.smem = smem
        self.setInst("s_atomic_dec")

    def getParams(self):
        return [self.dst, self.base]

    def getDstParams(self):
        return [self.dst] if self.dst else []

    def getSrcParams(self):
        return [self.base]

    def toString(self) -> str:
        parts = [_input_to_str(self.dst), _input_to_str(self.base)]
        kstr = self.instStr + " " + ", ".join(parts)
        if self.smem is not None and hasattr(self.smem, "toString"):
            kstr += self.smem.toString()
        return self.formatWithComment(kstr)

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st
        return _st.SAtomicDec(
            _to_stinky_register(self.dst),
            _to_stinky_register(self.base),
            self.comment)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = self.__class__.__new__(self.__class__)
        memo[id(self)] = dup
        Instruction.__init__(dup, self.instType, self.comment)
        dup.instStr = self.instStr
        dup.dst = _deepcopy(self.dst, memo) if self.dst is not None else None
        dup.base = _deepcopy(self.base, memo) if self.base is not None else None
        dup.smem = _deepcopy(self.smem, memo) if self.smem is not None else None
        return dup

# --- TensorLoadToLds: rocisa(group0, group1, group2, group3, comment) ---
def _make_tensor_load_class():
    def __init__(self, group0: Any = None, group1: Any = None,
                 group2: Any = None, group3: Any = None,
                 comment: str = "", **kw):
        _ = kw
        CommonInstruction.__init__(
            self, instType=InstType.INST_NOTYPE, dst=group0,
            srcs=[group1, group2, group3], dpp=None, sdwa=None,
            vop3=None, comment=comment)
        self.setInst("tensor_load_to_lds")

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st
        # group2/group3 are the TDM iterate-mode (TDMIterateMode) descriptors.
        # They are None for non-iterate loads but MUST be forwarded when set,
        # otherwise native emits a 4-operand ``tensor_load_to_lds`` while the
        # adaptor drops to 2 operands -> wrong LDS addressing -> wrong results.
        group2 = self.srcs[1] if len(self.srcs) > 1 else None
        group3 = self.srcs[2] if len(self.srcs) > 2 else None
        kwargs = {"comment": self.comment}
        if group2 is not None:
            kwargs["group2"] = _to_stinky_register(group2)
        if group3 is not None:
            kwargs["group3"] = _to_stinky_register(group3)
        return _st.TensorLoadToLds(
            _to_stinky_register(self.dst),
            _to_stinky_register(self.srcs[0]),
            **kwargs)

    def __deepcopy__(self, memo):
        return CommonInstruction.__deepcopy__(self, memo)

    cls = type("TensorLoadToLds", (CommonInstruction,), {
        "__init__": __init__,
        "to_stinky_logical": to_stinky_logical,
        "__deepcopy__": __deepcopy__,
        "issueLatency": staticmethod(lambda: 1),
    })
    cls.__qualname__ = "TensorLoadToLds"
    return cls


TensorLoadToLds = _make_tensor_load_class()
# logicalIR: GlobalPrefetchB8 — 2 srcs (vaddr, saddr), no dest
def _make_global_prefetch_class():
    """GlobalPrefetchB8: (vaddr, saddr, globalModifiers) → logicalIR 2 srcs."""

    class GlobalPrefetchB8(CommonInstruction):
        __slots__ = ("_modifiers",)

        def __init__(self, vaddr: Any = None, saddr: Any = None,
                     modifiers: Any = None, comment: str = "", **kw):
            _ = kw
            CommonInstruction.__init__(
                self, instType=InstType.INST_NOTYPE, dst=None,
                srcs=[vaddr, saddr], dpp=None, sdwa=None, vop3=None, comment=comment)
            self.setInst("global_prefetch_b8")
            self._modifiers = modifiers

        def to_stinky_logical(self) -> Any:
            import stinkytofu as _st
            factory = getattr(_st, "GlobalPrefetchB8")
            return factory(
                _to_stinky_register(self.srcs[0]),
                _to_stinky_register(self.srcs[1]),
                self.comment)

        def __deepcopy__(self, memo):
            return CommonInstruction.__deepcopy__(self, memo)

    return GlobalPrefetchB8

GlobalPrefetchB8 = _make_global_prefetch_class()


# ==========================================================================
# MFMA / SMFMA / MXMFMA instructions
# source: rocisa/rocisa/src/instruction/mfma.cpp
# ==========================================================================

_INST_TYPE_TO_STR: Dict[Any, str] = {}


def _inst_type_to_str(it: Any) -> str:
    """Convert an InstType enum value to the string used by logicalIR MFMA factories."""
    if not _INST_TYPE_TO_STR:
        _INST_TYPE_TO_STR.update({
            InstType.INST_F8: "fp8", InstType.INST_F16: "f16",
            InstType.INST_F32: "f32", InstType.INST_F64: "f64",
            InstType.INST_BF16: "bf16", InstType.INST_XF32: "xf32",
            InstType.INST_BF8: "bf8", InstType.INST_I8: "i8",
            InstType.INST_I32: "i32", InstType.INST_U8: "u8",
            InstType.INST_F4: "f4", InstType.INST_F6: "f6",
            InstType.INST_BF6: "bf6", InstType.INST_B8: "b8",
            InstType.INST_E5M3: "e5m3", InstType.INST_E8: "e8",
        })
    if it in _INST_TYPE_TO_STR:
        return _INST_TYPE_TO_STR[it]
    s = str(it)
    if "INST_" in s:
        return s.split("INST_")[-1].lower()
    return s


# InstType values that lower to the unified f8f6f4 WMMA opcode regardless of K
# (rocisa mfma.hpp:typeConvert F6/BF6/F6_B6/B6_F6 and every mixed low-precision
# combination). The actual input format is carried by matrix_a_fmt/matrix_b_fmt.
_WMMA_F8F6F4_ALWAYS = frozenset({
    InstType.INST_F6, InstType.INST_BF6, InstType.INST_F6_B6, InstType.INST_B6_F6,
    InstType.INST_F8_F4, InstType.INST_F4_F8, InstType.INST_F6_F4, InstType.INST_F4_F6,
    InstType.INST_F8_F6, InstType.INST_F6_F8, InstType.INST_F8_B6, InstType.INST_B6_F8,
    InstType.INST_B8_F4, InstType.INST_F4_B8, InstType.INST_B6_F4, InstType.INST_F4_B6,
    InstType.INST_B8_F6, InstType.INST_F6_B8, InstType.INST_B8_B6, InstType.INST_B6_B8,
})

# Per-matrix input format tokens for the f8f6f4-family WMMA (rocisa mfma.hpp
# getArgStr, HasWMMA_f8f6f4 branch). Emitted only when the K/tile condition holds.
_WMMA_MATRIX_FMT: Dict[Any, Any] = {}


def _init_wmma_matrix_fmt() -> None:
    if _WMMA_MATRIX_FMT:
        return
    _WMMA_MATRIX_FMT.update({
        InstType.INST_F8: ("MATRIX_FMT_FP8", "MATRIX_FMT_FP8"),
        InstType.INST_BF8: ("MATRIX_FMT_BF8", "MATRIX_FMT_BF8"),
        InstType.INST_F8_BF8: ("MATRIX_FMT_FP8", "MATRIX_FMT_BF8"),
        InstType.INST_BF8_F8: ("MATRIX_FMT_BF8", "MATRIX_FMT_FP8"),
        InstType.INST_F6: ("MATRIX_FMT_FP6", "MATRIX_FMT_FP6"),
        InstType.INST_BF6: ("MATRIX_FMT_BF6", "MATRIX_FMT_BF6"),
        InstType.INST_F6_B6: ("MATRIX_FMT_FP6", "MATRIX_FMT_BF6"),
        InstType.INST_B6_F6: ("MATRIX_FMT_BF6", "MATRIX_FMT_FP6"),
        InstType.INST_F8_F4: ("MATRIX_FMT_FP8", "MATRIX_FMT_FP4"),
        InstType.INST_F4_F8: ("MATRIX_FMT_FP4", "MATRIX_FMT_FP8"),
        InstType.INST_F6_F4: ("MATRIX_FMT_FP6", "MATRIX_FMT_FP4"),
        InstType.INST_F4_F6: ("MATRIX_FMT_FP4", "MATRIX_FMT_FP6"),
        InstType.INST_F8_F6: ("MATRIX_FMT_FP8", "MATRIX_FMT_FP6"),
        InstType.INST_F6_F8: ("MATRIX_FMT_FP6", "MATRIX_FMT_FP8"),
        InstType.INST_F8_B6: ("MATRIX_FMT_FP8", "MATRIX_FMT_BF6"),
        InstType.INST_B6_F8: ("MATRIX_FMT_BF6", "MATRIX_FMT_FP8"),
        InstType.INST_B8_F4: ("MATRIX_FMT_BF8", "MATRIX_FMT_FP4"),
        InstType.INST_F4_B8: ("MATRIX_FMT_FP4", "MATRIX_FMT_BF8"),
        InstType.INST_B6_F4: ("MATRIX_FMT_BF6", "MATRIX_FMT_FP4"),
        InstType.INST_F4_B6: ("MATRIX_FMT_FP4", "MATRIX_FMT_BF6"),
        InstType.INST_B8_F6: ("MATRIX_FMT_BF8", "MATRIX_FMT_FP6"),
        InstType.INST_F6_B8: ("MATRIX_FMT_FP6", "MATRIX_FMT_BF8"),
        InstType.INST_B8_B6: ("MATRIX_FMT_BF8", "MATRIX_FMT_BF6"),
        InstType.INST_B6_B8: ("MATRIX_FMT_BF6", "MATRIX_FMT_BF8"),
    })


def _wmma_type_convert(it: Any, m: int, n: int, k: int, has_wmma_v3: bool) -> str:
    """Port of rocisa ``MFMAInstruction::typeConvert`` for the low-precision
    (f8f6f4-family) inputs. Standard types defer to :func:`_inst_type_to_str`."""
    f8f6f4_k = 128 if has_wmma_v3 else 64
    f4_t = 32 if has_wmma_v3 else 0
    if it in _WMMA_F8F6F4_ALWAYS:
        return "f8f6f4"
    if it == InstType.INST_F8:
        return "f8f6f4" if k >= f8f6f4_k else "fp8_fp8"
    if it == InstType.INST_BF8:
        return "f8f6f4" if k >= f8f6f4_k else "bf8_bf8"
    if it == InstType.INST_F8_BF8:
        return "f8f6f4" if k >= f8f6f4_k else "fp8_bf8"
    if it == InstType.INST_BF8_F8:
        return "f8f6f4" if k >= f8f6f4_k else "bf8_fp8"
    if it == InstType.INST_F4:
        return "f8f6f4" if (m < f4_t and n < f4_t) else "f4"
    # rocisa MFMAInstruction::typeConvert maps the 8-bit integer inputs to the
    # unified "iu8" WMMA suffix (I8 stays "i8"); the generic table would emit
    # "u8", which has no gfx1250 ISA opcode (e.g. v_wmma_i32_16x16x64_iu8).
    if it == InstType.INST_U8:
        return "iu8"
    if it == InstType.INST_I8:
        return "i8"
    return _inst_type_to_str(it)


def _wmma_matrix_fmts(it: Any, m: int, n: int, k: int, has_wmma_v3: bool):
    """Port of rocisa ``getArgStr`` (HasWMMA_f8f6f4 branch): the matrix_*_fmt
    tokens, gated by the same K/tile condition rocisa uses."""
    _init_wmma_matrix_fmt()
    f4_t = 32 if has_wmma_v3 else 0
    if it == InstType.INST_F4:
        if m < f4_t and n < f4_t:
            return ("MATRIX_FMT_FP4", "MATRIX_FMT_FP4")
        return ("", "")
    if it in _WMMA_MATRIX_FMT and k > 64:
        return _WMMA_MATRIX_FMT[it]
    return ("", "")


class MFMAInstruction(Instruction):
    """``v_mfma_*`` shim (rocisa ``MFMAInstruction``)."""

    __slots__ = ("accType", "variant", "mfma1k", "acc", "a", "b", "acc2", "acc2_imm", "neg")

    def __init__(self, instType: Any = None, accType: Any = None,
                 variant: Any = None, mfma1k: bool = False,
                 acc: Any = None, a: Any = None, b: Any = None,
                 acc2: Any = None, acc2_imm: Any = None, neg: bool = False,
                 comment: str = "", **kw):
        _ = kw
        super().__init__(instType, comment)
        self.accType = accType
        self.variant = variant if variant is not None else []
        self.mfma1k = mfma1k
        self.acc = acc
        self.a = a
        self.b = b
        self.acc2 = acc2
        self.acc2_imm = acc2_imm
        self.neg = neg

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st
        from .base import getAsmCaps
        m = self.variant[0] if len(self.variant) > 0 else 0
        n = self.variant[1] if len(self.variant) > 1 else 0
        k = self.variant[2] if len(self.variant) > 2 else 0
        blocks = self.variant[3] if len(self.variant) > 3 else 1

        caps = getAsmCaps()
        has_wmma_v3 = bool(caps.get("HasWMMA_V3", 0))
        has_wmma_f8f6f4 = bool(caps.get("HasWMMA_f8f6f4", 0))
        is_wmma = not bool(caps.get("HasMFMA", 0))
        try:
            from . import rocIsa  # noqa: WPS433
            isa = tuple(rocIsa.getInstance().getKernel().isa)
        except Exception:  # noqa: BLE001 — best effort; fall back to no scaling
            isa = ()

        # rocisa MFMAInstruction::typeConvert + forceScaledWMMA (mfma.hpp).
        # gfx1250 low-precision (f8f6f4/f4) WMMA uses the v_wmma_scale_* encoding
        # with zero scales, plus per-matrix input formats.
        type_str = _wmma_type_convert(self.instType, m, n, k, has_wmma_v3)
        # forceScaledWMMA() gates only the mnemonic (isaVersion, not caps).
        scaled = is_wmma and isa == (12, 5, 0) and type_str in ("f8f6f4", "f4")
        # rocisa emits the ", 0, 0" scale operands and matrix_*_fmt only inside
        # the HasWMMA_f8f6f4 getArgStr branch, so both are caps-gated.
        scale_operands = scaled and has_wmma_f8f6f4
        matrix_a_fmt = ""
        matrix_b_fmt = ""
        if has_wmma_f8f6f4:
            matrix_a_fmt, matrix_b_fmt = _wmma_matrix_fmts(
                self.instType, m, n, k, has_wmma_v3)

        acc2_reg = None
        if self.acc2_imm is not None:
            acc2_reg = _to_stinky_register(self.acc2_imm)
        elif self.acc2 is not None:
            acc2_reg = _to_stinky_register(self.acc2)
        return _st.MFMA(
            type_str,
            _inst_type_to_str(self.accType),
            m, n, k, blocks, self.mfma1k,
            _to_stinky_register(self.acc),
            _to_stinky_register(self.a),
            _to_stinky_register(self.b),
            acc2=acc2_reg,
            neg=self.neg,
            matrixAFmt=matrix_a_fmt,
            matrixBFmt=matrix_b_fmt,
            scaled=scaled,
            scaleOperands=scale_operands,
            comment=self.comment)

    def getParams(self):
        return [self.acc, self.a, self.b]

    def getDstParams(self):
        return [self.acc]

    def getSrcParams(self):
        return [self.a, self.b]

    def getIssueLatency(self) -> int:
        m = self.variant[0] if len(self.variant) > 0 else 16
        blocks = self.variant[3] if len(self.variant) > 3 else 1
        return getMFMAIssueLatency(None, m, blocks)[0]

    def __deepcopy__(self, memo):
        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        Instruction.__init__(clone, self.instType, self.comment)
        clone.outputInlineAsm = self.outputInlineAsm
        clone.instStr = self.instStr
        clone.m_memToken = None
        clone.accType = self.accType
        clone.variant = list(self.variant)
        clone.mfma1k = self.mfma1k
        clone.acc = _deepcopy(self.acc, memo) if self.acc is not None else None
        clone.a = _deepcopy(self.a, memo) if self.a is not None else None
        clone.b = _deepcopy(self.b, memo) if self.b is not None else None
        clone.acc2 = _deepcopy(self.acc2, memo) if self.acc2 is not None else None
        clone.acc2_imm = self.acc2_imm
        clone.neg = self.neg
        return clone


class MXMFMAInstruction(Instruction):
    """``v_wmma_scale_*`` / ``v_mfma_scale_*`` shim (rocisa ``MXMFMAInstruction``)."""

    __slots__ = ("accType", "mxScaleAType", "mxScaleBType", "variant",
                 "acc", "a", "b", "acc2", "mxsa", "mxsb", "vop3", "mxCBSZ", "block")

    def __init__(self, *, instType: Any = None, accType: Any = None,
                 variant: Any = None, acc: Any = None,
                 a: Any = None, b: Any = None,
                 acc2: Any = None, mxsa: Any = None, mxsb: Any = None,
                 vop3: Any = None,
                 mxScaleAType: Any = None, mxScaleBType: Any = None,
                 mxCBSZ: int = 0, block: int = 32,
                 comment: str = "", **kw):
        _ = kw
        super().__init__(instType, comment)
        self.accType = accType
        self.mxScaleAType = mxScaleAType
        self.mxScaleBType = mxScaleBType
        self.variant = variant if variant is not None else []
        self.acc = acc
        self.a = a
        self.b = b
        self.acc2 = acc2
        self.mxsa = mxsa
        self.mxsb = mxsb
        self.vop3 = vop3
        self.mxCBSZ = mxCBSZ
        # MX scale block size (16 or 32); selects v_wmma_scale vs v_wmma_scale16.
        # rocisa passes this via `block=max(MXBlockA, MXBlockB)`. It is distinct
        # from variant[3] (the MI blocks count, typically 1).
        self.block = block

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st
        from .base import getAsmCaps
        m = self.variant[0] if len(self.variant) > 0 else 0
        n = self.variant[1] if len(self.variant) > 1 else 0
        k = self.variant[2] if len(self.variant) > 2 else 0
        # rocisa MXMFMAInstruction::typeConvert derives the v_wmma_scale suffix
        # purely from the tile dims (fixed 32 threshold), NOT from the input
        # datatype: 16x16 -> "f8f6f4", 32x16 -> "f4". Passing the raw input
        # abbreviation (e.g. "f4") emits v_wmma_scale_f32_16x16x128_f4, which has
        # no gfx1250 ISA opcode (only the _f8f6f4 form exists for 16x16).
        mx_type_str = "f8f6f4" if (m < 32 and n < 32) else "f4"
        # rocisa MXMFMAInstruction::wmmaInputPermuteStr: the f8f6f4-family scaled
        # WMMA needs per-matrix input formats (matrix_a_fmt:MATRIX_FMT_FP4 ...);
        # without them the assembler assumes FP8 and rejects the FP4 tuple size.
        has_wmma_v3 = bool(getAsmCaps().get("HasWMMA_V3", 0))
        matrix_a_fmt, matrix_b_fmt = _wmma_matrix_fmts(
            self.instType, m, n, k, has_wmma_v3)
        return _st.MXMFMA(
            mx_type_str,
            _inst_type_to_str(self.accType),
            _inst_type_to_str(self.mxScaleAType) if self.mxScaleAType else "f32",
            _inst_type_to_str(self.mxScaleBType) if self.mxScaleBType else "f32",
            m, n, k, self.block,
            _to_stinky_register(self.acc),
            _to_stinky_register(self.a),
            _to_stinky_register(self.b),
            _to_stinky_register(self.acc2) if self.acc2 else None,
            _to_stinky_register(self.mxsa) if self.mxsa else None,
            _to_stinky_register(self.mxsb) if self.mxsb else None,
            matrixAFmt=matrix_a_fmt,
            matrixBFmt=matrix_b_fmt,
            comment=self.comment)

    def getParams(self):
        return [self.acc, self.a, self.b]

    def getDstParams(self):
        return [self.acc]

    def getSrcParams(self):
        return [self.a, self.b]

    def getIssueLatency(self) -> int:
        m = self.variant[0] if len(self.variant) > 0 else 16
        blocks = self.variant[3] if len(self.variant) > 3 else 1
        return getMFMAIssueLatency(None, m, blocks)[0]

    def __deepcopy__(self, memo):
        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        Instruction.__init__(clone, self.instType, self.comment)
        clone.outputInlineAsm = self.outputInlineAsm
        clone.instStr = self.instStr
        clone.m_memToken = None
        clone.accType = self.accType
        clone.mxScaleAType = self.mxScaleAType
        clone.mxScaleBType = self.mxScaleBType
        clone.variant = list(self.variant)
        clone.acc = _deepcopy(self.acc, memo) if self.acc is not None else None
        clone.a = _deepcopy(self.a, memo) if self.a is not None else None
        clone.b = _deepcopy(self.b, memo) if self.b is not None else None
        clone.acc2 = _deepcopy(self.acc2, memo) if self.acc2 is not None else None
        clone.mxsa = _deepcopy(self.mxsa, memo) if self.mxsa is not None else None
        clone.mxsb = _deepcopy(self.mxsb, memo) if self.mxsb is not None else None
        clone.vop3 = self.vop3
        clone.mxCBSZ = self.mxCBSZ
        clone.block = self.block
        return clone


def _smfma_type_convert(it: Any) -> str:
    """Port of rocisa ``SMFMAInstruction::typeConvert`` (mfma.hpp:1007-1034).

    The sparse (v_smfmac_/v_swmmac_) 8-bit float families take *two* input
    format suffixes (A_B), e.g. ``bf8_bf8`` / ``fp8_bf8`` -- unlike the generic
    ``_inst_type_to_str`` which returns a single token (``bf8``) and would emit
    a mnemonic with no ISA opcode (e.g. v_swmmac_f32_16x16x128_bf8). Integer
    inputs map to ``iu8`` (U8) / ``i8`` (I8) like rocisa."""
    mapping = {
        InstType.INST_F16: "f16",
        InstType.INST_F32: "f32",
        InstType.INST_BF16: "bf16",
        InstType.INST_I8: "i8",
        InstType.INST_U8: "iu8",
        InstType.INST_I32: "i32",
        InstType.INST_F8: "fp8_fp8",
        InstType.INST_BF8: "bf8_bf8",
        InstType.INST_F8_BF8: "fp8_bf8",
        InstType.INST_BF8_F8: "bf8_fp8",
    }
    if it in mapping:
        return mapping[it]
    return _inst_type_to_str(it)


class SMFMAInstruction(Instruction):
    """``v_smfma_*`` shim (rocisa ``SMFMAInstruction``)."""

    __slots__ = ("accType", "variant", "mfma1k", "acc", "a", "b", "metadata", "neg")

    def __init__(self, instType: Any = None, accType: Any = None,
                 variant: Any = None, mfma1k: bool = False,
                 acc: Any = None, a: Any = None, b: Any = None,
                 metadata: Any = None, neg: bool = False,
                 comment: str = "", **kw):
        _ = kw
        super().__init__(instType, comment)
        self.accType = accType
        self.variant = variant if variant is not None else []
        self.mfma1k = mfma1k
        self.acc = acc
        self.a = a
        self.b = b
        self.metadata = metadata
        self.neg = neg

    def to_stinky_logical(self) -> Any:
        import stinkytofu as _st
        m = self.variant[0] if len(self.variant) > 0 else 0
        n = self.variant[1] if len(self.variant) > 1 else 0
        k = self.variant[2] if len(self.variant) > 2 else 0
        blocks = self.variant[3] if len(self.variant) > 3 else 1
        return _st.SMFMA(
            _smfma_type_convert(self.instType),
            _inst_type_to_str(self.accType),
            m, n, k, blocks, self.mfma1k,
            _to_stinky_register(self.acc),
            _to_stinky_register(self.a),
            _to_stinky_register(self.b),
            _to_stinky_register(self.metadata),
            neg=self.neg,
            comment=self.comment)

    def getParams(self):
        return [self.acc, self.a, self.b, self.metadata]

    def getDstParams(self):
        return [self.acc]

    def getSrcParams(self):
        return [self.a, self.b, self.metadata]

    def getIssueLatency(self) -> int:
        m = self.variant[0] if len(self.variant) > 0 else 16
        blocks = self.variant[3] if len(self.variant) > 3 else 1
        return getSMFMAIssueLatency(None, m, blocks)[0]

    def __deepcopy__(self, memo):
        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        Instruction.__init__(clone, self.instType, self.comment)
        clone.outputInlineAsm = self.outputInlineAsm
        clone.instStr = self.instStr
        clone.m_memToken = None
        clone.accType = self.accType
        clone.variant = list(self.variant)
        clone.mfma1k = self.mfma1k
        clone.acc = _deepcopy(self.acc, memo) if self.acc is not None else None
        clone.a = _deepcopy(self.a, memo) if self.a is not None else None
        clone.b = _deepcopy(self.b, memo) if self.b is not None else None
        clone.metadata = _deepcopy(self.metadata, memo) if self.metadata is not None else None
        clone.neg = self.neg
        return clone
def getMFMAIssueLatency(dataType, matrixInstM, matrixInstB):
    """Workaround port of ``rocisa::getMFMAIssueLatency<false>``.

    Returns ``(matrixInstM // mi_divisor, miIssueLatency)`` matching the
    C++ default branch (``mi_divisor=2``, ``miIssueLatency=2``). The C++
    template has ISA-specific overrides (gfx940/941/942/950 + matrixInstB==1
    + halve-precision → divisor=4 / latency=1; XFloat32 → divisor=4;
    gfx950 8-bit float → divisor=2 again). None of those branches fire for
    gfx1250, so the default values are byte-for-byte equivalent for the
    only ISA the logicalIR backend supports today.

    TODO: re-derive ``mi_divisor`` from
    ``rocIsa.getInstance().getKernel().isa`` when extending beyond gfx1250.
    """
    return (matrixInstM // 2, 2)


def getSMFMAIssueLatency(dataType, matrixInstM, matrixInstB):
    """Workaround port of the sparse branch of
    ``rocisa::getMFMAIssueLatency<true>`` (mfma.hpp:55-58 hard-codes
    ``mi_divisor = 4`` when ``isSparse`` is true). Same ISA caveats as
    ``getMFMAIssueLatency``.
    """
    return (matrixInstM // 4, 2)


# ==========================================================================
# Extension helpers (exposed as functions)
# source: rocisa/rocisa/src/instruction/extension.cpp
# ==========================================================================

def _ext_lazy():
    """Lazy imports to avoid circular dependency (code.py imports instruction.py)."""
    from .code import Module, Label  # noqa: F811
    from .container import ContinuousRegister, sgpr  # noqa: F811
    return Module, Label, ContinuousRegister, sgpr


def SGetPositivePCOffset(sgprIdx, label, tmpSgprRes):
    """Port of ``rocisa::SGetPositivePCOffset`` (extension.hpp)."""
    Module, Label, ContinuousRegister, sgpr = _ext_lazy()
    if isinstance(tmpSgprRes, int):
        tmpSgprRes = ContinuousRegister(tmpSgprRes, 1)
    labelName = label.getLabelName()
    module = Module("SGetPositivePCOffset " + labelName)
    if tmpSgprRes.size < 1:
        raise RuntimeError("ContinuousRegister size must be at least 1.")
    tmpSgpr = tmpSgprRes.idx
    module.add(SGetPCB64(dst=sgpr(sgprIdx, 2), comment="addr of next instr"))
    module.add(SAddI32(dst=sgpr(tmpSgpr), src0=labelName, src1=4,
                       comment="target branch offset"))
    module.add(SAddU32(dst=sgpr(sgprIdx), src0=sgpr(sgprIdx),
                       src1=sgpr(tmpSgpr), comment="add target branch offset"))
    module.add(SAddCU32(dst=sgpr(sgprIdx + 1), src0=sgpr(sgprIdx + 1),
                        src1=0, comment="add high and carry"))
    return module


def _split_tmp_regs(tmpSgprRes):
    """Split a >= 3-register ContinuousRegister into (aligned_pair_idx, off_idx)."""
    if tmpSgprRes.idx % 2 == 0:
        return tmpSgprRes.idx, tmpSgprRes.idx + 2
    else:
        return tmpSgprRes.idx + 1, tmpSgprRes.idx


def SLongBranch(label, tmpSgprRes_or_pcPair, offSgpr_or_posLabel=None,
                positiveLabelStr=None, comment=""):
    """Port of ``rocisa::SLongBranch`` (extension.hpp).

    Two calling conventions:
      SLongBranch(label, tmpSgprRes, positiveLabelStr, comment="")
      SLongBranch(label, pcPair, offSgpr, positiveLabelStr, comment="")
    """
    if isinstance(offSgpr_or_posLabel, str) or offSgpr_or_posLabel is None:
        tmpSgprRes = tmpSgprRes_or_pcPair
        posLabel = offSgpr_or_posLabel or ""
        cmt = positiveLabelStr if positiveLabelStr is not None else comment
        if tmpSgprRes.size < 3:
            raise RuntimeError("ContinuousRegister size must be at least 3.")
        tmpSgprX2, tmpSgprX1 = _split_tmp_regs(tmpSgprRes)
        return _SLongBranchImpl(label, tmpSgprX2, tmpSgprX1, posLabel, cmt)
    else:
        pcPair = tmpSgprRes_or_pcPair
        offSgpr = offSgpr_or_posLabel
        posLabel = positiveLabelStr or ""
        cmt = comment
        if pcPair.size < 2 or pcPair.idx % 2 != 0:
            raise RuntimeError("pcPair must be a 2-aligned pair.")
        if offSgpr.size < 1:
            raise RuntimeError("offSgpr must have at least 1 register.")
        return _SLongBranchImpl(label, pcPair.idx, offSgpr.idx, posLabel, cmt)


def _SLongBranchImpl(label, tmpSgprX2, tmpSgprX1, positiveLabelStr, comment):
    Module, Label, ContinuousRegister, sgpr = _ext_lazy()
    labelName = label.getLabelName()
    module = Module("SLongBranch " + labelName)
    if comment:
        module.addComment(comment)
    positiveLabel = Label(positiveLabelStr, "")
    module.add(SGetPCB64(dst=sgpr(tmpSgprX2, 2), comment="addr of next instr"))
    module.add(SAddI32(dst=sgpr(tmpSgprX1), src0=labelName, src1=4,
                       comment="target branch offset"))
    module.add(SCmpGeI32(src0=sgpr(tmpSgprX1), src1=0,
                         comment="check positive or negative"))
    module.add(SCBranchSCC1(labelName=positiveLabel.getLabelName(),
                            comment="jump when positive"))
    module.add(SAbsI32(dst=sgpr(tmpSgprX1), src=sgpr(tmpSgprX1),
                       comment="abs offset"))
    module.add(SSubU32(dst=sgpr(tmpSgprX2), src0=sgpr(tmpSgprX2),
                       src1=sgpr(tmpSgprX1), comment="sub target branch offset"))
    module.add(SSubBU32(dst=sgpr(tmpSgprX2 + 1), src0=sgpr(tmpSgprX2 + 1),
                        src1=0, comment="sub high and carry"))
    module.add(SSetPCB64(src=sgpr(tmpSgprX2, 2),
                         comment="branch to " + labelName))
    module.add(positiveLabel)
    module.add(SAddU32(dst=sgpr(tmpSgprX2), src0=sgpr(tmpSgprX2),
                       src1=sgpr(tmpSgprX1), comment="add target branch offset"))
    module.add(SAddCU32(dst=sgpr(tmpSgprX2 + 1), src0=sgpr(tmpSgprX2 + 1),
                        src1=0, comment="add high and carry"))
    module.add(SSetPCB64(src=sgpr(tmpSgprX2, 2),
                         comment="branch to " + labelName))
    return module


def SLongBranchPositive(label, tmpSgprRes_or_pcPair, offSgpr_or_comment=None,
                        comment=""):
    """Port of ``rocisa::SLongBranchPositive`` (extension.hpp).

    Two calling conventions:
      SLongBranchPositive(label, tmpSgprRes, comment="")
      SLongBranchPositive(label, pcPair, offSgpr, comment="")
    """
    from .base import getAsmCaps
    Module, Label, ContinuousRegister, sgpr = _ext_lazy()

    if isinstance(offSgpr_or_comment, str) or offSgpr_or_comment is None:
        cmt = offSgpr_or_comment or ""
        labelName = label.getLabelName()
        module = Module("SLongBranchPositive " + labelName)
        if cmt:
            module.addComment(cmt)
        if getAsmCaps().get("HasAdd_PC_i64", 0):
            module.add(SAddPCI64_SIMM(
                labelName=labelName + "-.-12",
                comment="Add PC to " + labelName
                + ", the constant correction is dependent on the"
                " current assembler behavior."))
        else:
            tmpSgprRes = tmpSgprRes_or_pcPair
            if tmpSgprRes.size < 3:
                raise RuntimeError(
                    "ContinuousRegister size must be at least 3.")
            tmpSgprX2, tmpSgprX1 = _split_tmp_regs(tmpSgprRes)
            cr = ContinuousRegister(tmpSgprX1, 1)
            module.add(SGetPositivePCOffset(tmpSgprX2, label, cr))
            module.add(SSetPCB64(src=sgpr(tmpSgprX2, 2),
                                 comment="branch to " + labelName))
        return module
    else:
        pcPair = tmpSgprRes_or_pcPair
        offSgpr = offSgpr_or_comment
        cmt = comment
        labelName = label.getLabelName()
        module = Module("SLongBranchPositive " + labelName)
        if cmt:
            module.addComment(cmt)
        if pcPair.size < 2 or pcPair.idx % 2 != 0:
            raise RuntimeError("pcPair must be a 2-aligned pair.")
        if offSgpr.size < 1:
            raise RuntimeError("offSgpr must have at least 1 register.")
        module.add(SGetPositivePCOffset(pcPair.idx, label, offSgpr.idx))
        module.add(SSetPCB64(src=sgpr(pcPair.idx, 2),
                             comment="branch to " + labelName))
        return module


def SLongBranchNegative(label, tmpSgprRes_or_pcPair, offSgpr_or_comment=None,
                        comment=""):
    """Port of ``rocisa::SLongBranchNegative`` (extension.hpp).

    Two calling conventions:
      SLongBranchNegative(label, tmpSgprRes, comment="")
      SLongBranchNegative(label, pcPair, offSgpr, comment="")
    """
    if isinstance(offSgpr_or_comment, str) or offSgpr_or_comment is None:
        tmpSgprRes = tmpSgprRes_or_pcPair
        cmt = offSgpr_or_comment or ""
        if tmpSgprRes.size < 3:
            raise RuntimeError(
                "ContinuousRegister size must be at least 3.")
        tmpSgprX2, tmpSgprX1 = _split_tmp_regs(tmpSgprRes)
        return _SLongBranchNegativeImpl(label, tmpSgprX2, tmpSgprX1, cmt)
    else:
        pcPair = tmpSgprRes_or_pcPair
        offSgpr = offSgpr_or_comment
        cmt = comment
        if pcPair.size < 2 or pcPair.idx % 2 != 0:
            raise RuntimeError("pcPair must be a 2-aligned pair.")
        if offSgpr.size < 1:
            raise RuntimeError("offSgpr must have at least 1 register.")
        return _SLongBranchNegativeImpl(
            label, pcPair.idx, offSgpr.idx, cmt)


def _SLongBranchNegativeImpl(label, tmpSgprX2, tmpSgprX1, comment):
    Module, Label, ContinuousRegister, sgpr = _ext_lazy()
    labelName = label.getLabelName()
    module = Module("SLongBranchNegative " + labelName)
    if comment:
        module.addComment(comment)
    module.add(SGetPCB64(dst=sgpr(tmpSgprX2, 2), comment="addr of next instr"))
    module.add(SAddI32(dst=sgpr(tmpSgprX1), src0=labelName, src1=4,
                       comment="target branch offset"))
    module.add(SAbsI32(dst=sgpr(tmpSgprX1), src=sgpr(tmpSgprX1),
                       comment="abs offset"))
    module.add(SSubU32(dst=sgpr(tmpSgprX2), src0=sgpr(tmpSgprX2),
                       src1=sgpr(tmpSgprX1), comment="sub target branch offset"))
    module.add(SSubBU32(dst=sgpr(tmpSgprX2 + 1), src0=sgpr(tmpSgprX2 + 1),
                        src1=0, comment="sub high and carry"))
    module.add(SSetPCB64(src=sgpr(tmpSgprX2, 2),
                         comment="branch to " + labelName))
    return module


def SCLongBranchScc0(label, tmpSgprRes, noBranchLabelStr,
                     positiveLabelStr, posNeg=0, comment=""):
    """Port of ``rocisa::SCLongBranchScc0`` (extension.hpp)."""
    Module, Label, ContinuousRegister, sgpr = _ext_lazy()
    module = Module("SCLongBranchScc0 " + label.getLabelName())
    noBranchLabel = Label(noBranchLabelStr, "")
    module.add(SCBranchSCC1(labelName=noBranchLabel.getLabelName(),
                            comment="Only branch on scc0"))
    if posNeg > 0:
        module.add(SLongBranchPositive(label, tmpSgprRes, comment))
    elif posNeg < 0:
        module.add(SLongBranchNegative(label, tmpSgprRes, comment))
    else:
        module.add(SLongBranch(label, tmpSgprRes, positiveLabelStr, comment))
    module.add(noBranchLabel)
    return module


def SCLongBranchScc1(label, tmpSgprRes, noBranchLabelStr,
                     positiveLabelStr, posNeg=0, comment=""):
    """Port of ``rocisa::SCLongBranchScc1`` (extension.hpp)."""
    Module, Label, ContinuousRegister, sgpr = _ext_lazy()
    module = Module("SCLongBranchScc1 " + label.getLabelName())
    noBranchLabel = Label(noBranchLabelStr, "")
    module.add(SCBranchSCC0(labelName=noBranchLabel.getLabelName(),
                            comment="Only branch on scc1"))
    if posNeg > 0:
        module.add(SLongBranchPositive(label, tmpSgprRes, comment))
    elif posNeg < 0:
        module.add(SLongBranchNegative(label, tmpSgprRes, comment))
    else:
        module.add(SLongBranch(label, tmpSgprRes, positiveLabelStr, comment))
    module.add(noBranchLabel)
    return module


def SCLongBranchVccnz(label, tmpSgprRes, noBranchLabelStr,
                      positiveLabelStr, posNeg=0, comment=""):
    """Port of ``rocisa::SCLongBranchVccnz`` (extension.hpp)."""
    Module, Label, ContinuousRegister, sgpr = _ext_lazy()
    module = Module("SCLongBranchVccnz " + label.getLabelName())
    noBranchLabel = Label(noBranchLabelStr, "")
    module.add(SCBranchVCCZ(labelName=noBranchLabel.getLabelName(),
                            comment="Only branch on vccz"))
    if posNeg > 0:
        module.add(SLongBranchPositive(label, tmpSgprRes, comment))
    elif posNeg < 0:
        module.add(SLongBranchNegative(label, tmpSgprRes, comment))
    else:
        module.add(SLongBranch(label, tmpSgprRes, positiveLabelStr, comment))
    module.add(noBranchLabel)
    return module
def SMulInt64to32(dst0: Any, dst1: Any, src0: Any, src1: Any,
                  tmpVgprRes: Any, hasSMulHi: bool, sign: bool = False,
                  comment: str = "") -> Any:
    """64-bit = 32×32 multiply, result in dst0:dst1 (extension.hpp:391)."""
    Module, _, ContinuousRegister, _ = _ext_lazy()
    from .container import vgpr  # noqa: WPS433

    module = Module("SMulInt64to32")
    if hasattr(tmpVgprRes, "size") and tmpVgprRes.size < 2:
        raise RuntimeError("ContinuousRegister size must be at least 2.")

    if hasSMulHi:
        if sign:
            module.add(SMulHII32(dst=dst1, src0=src0, src1=src1, comment=comment))
        else:
            module.add(SMulHIU32(dst=dst1, src0=src0, src1=src1, comment=comment))
        module.add(SMulI32(dst=dst0, src0=src0, src1=src1, comment=comment))
    else:
        idx = tmpVgprRes.idx if hasattr(tmpVgprRes, "idx") else int(tmpVgprRes)
        vgprTmp = vgpr(idx)
        vgprTmp1 = vgpr(idx + 1)
        swapSrc0, swapSrc1 = (src0, src1) if _is_container(src1) else (src1, src0)
        module.add(VMovB32(dst=vgprTmp, src=swapSrc0, comment=comment))
        if sign:
            module.add(VMulHII32(dst=vgprTmp1, src0=vgprTmp, src1=swapSrc1,
                                 comment=comment))
        else:
            module.add(VMulHIU32(dst=vgprTmp1, src0=vgprTmp, src1=swapSrc1,
                                 comment=comment))
        module.add(VReadfirstlaneB32(dst=dst1, src=vgprTmp1, comment=comment))
        module.add(VMulLOU32(dst=vgprTmp1, src0=vgprTmp, src1=swapSrc1,
                             comment=comment))
        module.add(VReadfirstlaneB32(dst=dst0, src=vgprTmp1, comment=comment))
    return module


def _is_container(val: Any) -> bool:
    """Check if val is a Container-like object (not a scalar int/hex)."""
    return hasattr(val, "toString")
# ==========================================================================
# True16 / conversion helpers (extension.hpp:473-627)
#
# Each factory checks hardware capabilities and returns a single instruction
# with the appropriate SDWA, VOP3P, or True16 modifiers.
# ==========================================================================


class _True16Wrap:
    """Wraps a register/input and appends a True16 ``.h``/``.l`` suffix."""

    __slots__ = ("_inner", "_suffix")

    def __init__(self, inner: Any, sel: Any) -> None:
        from .enum import HighBitSel  # noqa: WPS433
        self._inner = inner
        self._suffix = ".h" if sel == HighBitSel.HIGH else ".l"

    def toString(self) -> str:
        return _input_to_str(self._inner) + self._suffix

    def to_stinky(self) -> Any:
        """Return the inner register as a proper StinkyRegister.

        The True16 .h/.l selection is encoded via the VOP3 op_sel modifier
        on the instruction, NOT as a string suffix on the register. Returning
        the structured register preserves the physical index needed by
        InsertVgprMsbPass for correct MSB computation.
        """
        if hasattr(self._inner, "to_stinky"):
            return self._inner.to_stinky()
        import stinkytofu as _st  # noqa: WPS433
        return _st.Register(self.toString())

    def __str__(self) -> str:
        return self.toString()


def ECvtF16toF32(dst: Any, src: Any, sel: Any, comment: str = "") -> Any:
    """Convert F16 → F32, selecting src half-word by *sel* (extension.hpp:474)."""
    from .base import getArchCaps  # noqa: WPS433
    from .container import SDWAModifiers  # noqa: WPS433
    from .enum import HighBitSel, SelectBit  # noqa: WPS433

    if getArchCaps().get("NoSDWA", 0):
        return VCvtF16toF32(dst=dst, src=_True16Wrap(src, sel), comment=comment)

    src0_sel = SelectBit.WORD_1 if sel == HighBitSel.HIGH else SelectBit.WORD_0
    return VCvtF16toF32(
        dst=dst, src=src, sdwa=SDWAModifiers(src0_sel=src0_sel), comment=comment,
    )


def ECvtF32toF16(dst: Any, src: Any, sel: Any = None, comment: str = "") -> Any:
    """Convert F32 → F16 with optional half-word packing (extension.hpp:506)."""
    from .base import getArchCaps  # noqa: WPS433
    from .container import SDWAModifiers  # noqa: WPS433
    from .enum import HighBitSel, SelectBit  # noqa: WPS433

    if sel is None:
        return VCvtF32toF16(dst=dst, src=src, comment=comment)

    if getArchCaps().get("NoSDWA", 0):
        return VCvtF32toF16(dst=_True16Wrap(dst, sel), src=src, comment=comment)

    dst_sel = SelectBit.WORD_1 if sel == HighBitSel.HIGH else SelectBit.WORD_0
    return VCvtF32toF16(
        dst=dst, src=src, sdwa=SDWAModifiers(dst_sel=dst_sel), comment=comment,
    )


def ECvtPkFP8toF32(dst: Any, src: Any, sel: Any, comment: str = "") -> Any:
    """Unpack packed-FP8 → 2×F32, selecting src half-word (extension.hpp:537)."""
    from .base import getArchCaps  # noqa: WPS433
    from .container import SDWAModifiers, VOP3PModifiers  # noqa: WPS433
    from .enum import HighBitSel, SelectBit  # noqa: WPS433

    sel_int = 1 if sel == HighBitSel.HIGH else 0
    if getArchCaps().get("NoSDWA", 0):
        inst = VCvtPkFP8toF32(
            dst=dst, src=_True16Wrap(src, sel), comment=comment,
        )
        inst.vop3 = VOP3PModifiers(op_sel=[sel_int])
        return inst

    src0_sel = SelectBit.WORD_1 if sel == HighBitSel.HIGH else SelectBit.WORD_0
    return VCvtPkFP8toF32(
        dst=dst, src=src, sdwa=SDWAModifiers(src0_sel=src0_sel), comment=comment,
    )


def ECvtPkBF8toF32(dst: Any, src: Any, sel: Any, comment: str = "") -> Any:
    """Unpack packed-BF8 → 2×F32, selecting src half-word (extension.hpp:566)."""
    from .base import getArchCaps  # noqa: WPS433
    from .container import SDWAModifiers, VOP3PModifiers  # noqa: WPS433
    from .enum import HighBitSel, SelectBit  # noqa: WPS433

    sel_int = 1 if sel == HighBitSel.HIGH else 0
    if getArchCaps().get("NoSDWA", 0):
        inst = VCvtPkBF8toF32(
            dst=dst, src=_True16Wrap(src, sel), comment=comment,
        )
        inst.vop3 = VOP3PModifiers(op_sel=[sel_int])
        return inst

    src0_sel = SelectBit.WORD_1 if sel == HighBitSel.HIGH else SelectBit.WORD_0
    return VCvtPkBF8toF32(
        dst=dst, src=src, sdwa=SDWAModifiers(src0_sel=src0_sel), comment=comment,
    )


def VCvtBF16toFP32(dst: Any, src: Any, vgprMask: Any, vi: int,
                   comment: str = "") -> Any:
    """BF16 → FP32 conversion with architecture dispatch (extension.hpp:594)."""
    from .base import getAsmCaps, getArchCaps  # noqa: WPS433
    from .container import SDWAModifiers, VOP3PModifiers  # noqa: WPS433
    from .enum import HighBitSel, SelectBit  # noqa: WPS433

    if not getAsmCaps().get("HasBF16CVT", 0):
        if (vi % 2) == 1:
            if vgprMask is None:
                raise RuntimeError("vgprMask is null")
            return VAndB32(
                dst=dst, src0=src, src1=vgprMask,
                comment="cvt bf16 to fp32. " + comment,
            )
        return VLShiftLeftB32(
            dst=dst, src0=16, src1=src,
            comment="cvt bf16 to fp32. " + comment,
        )

    if getArchCaps().get("NoSDWA", 0):
        sel = HighBitSel.HIGH if (vi % 2) == 1 else HighBitSel.LOW
        inst = PVCvtBF16toFP32(
            dst=dst, src=_True16Wrap(src, sel), comment="cvt bf16 to f32",
        )
        inst.vop3 = VOP3PModifiers(op_sel=[vi % 2])
        return inst

    src0_sel = SelectBit.WORD_1 if (vi % 2) == 1 else SelectBit.WORD_0
    return PVCvtBF16toFP32(
        dst=dst, src=src, sdwa=SDWAModifiers(src0_sel=src0_sel),
        comment="cvt bf16 to f32",
    )


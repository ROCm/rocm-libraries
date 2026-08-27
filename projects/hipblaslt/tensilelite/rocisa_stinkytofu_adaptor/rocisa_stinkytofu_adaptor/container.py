# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Register references, factory helpers (sgpr/vgpr/accvgpr/mgpr), and asm modifiers.

All register containers, holders, HW aliases (VCC/EXEC), and modifier
descriptors (DS/FLAT/GLOBAL/SMEM/SDWA/DPP/VOP3P/True16) are real.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional, Tuple, Union

from .caps import glc_bit_name_from_caps, slc_bit_name_from_caps
from .enum import CacheScope, HighBitSel, NonVolatile, SelectBit, TemporalHint, UnusedBit

_P = "rocisa.container"


# ---------------------------------------------------------------------------
# Container ABC -- polymorphic base (rocisa::Container).
# ---------------------------------------------------------------------------
#
# C++ ``rocisa::Container`` exposes pure ``clone()`` / ``toString()``; nanobind
# subclasses inherit through it (``RegisterContainer``, modifiers, EXEC/VCC, …).
# Python mirrors that hierarchy via this ABC.
#
# Note: native rocisa can ``Container()`` via a nanobind trampoline
# (``PyContainer``); this ABC is not constructible, which matches the C++
# abstract type more closely than the binding quirk.
class Container(ABC):
    """Abstract base matching ``rocisa::Container`` (clone + toString)."""

    @abstractmethod
    def toString(self) -> str:
        """Return the asm operand / modifier token string."""

    def clone(self) -> "Container":
        """Deep copy (C++ ``clone()`` returns ``shared_ptr<Container>``)."""
        return deepcopy(self)

    def __str__(self) -> str:
        return self.toString()


# ---------------------------------------------------------------------------
# RegName -- symbolic register name + per-offset chain.
# ---------------------------------------------------------------------------
#
# Single source of truth for the (name, offsets) struct view that
# KernelWriter touches. Stinkytofu encodes the same idea as a
# "name+off1+off2+..." string on StinkyRegister.literalValue; that
# string is rehydrated only at to_stinky() time.
class RegName:
    """Symbolic register name with optional offset chain.

    Encodings:
        * ``str(rn)`` -> ``"name+off1+off2+..."``.
        * ``rn.getTotalIdx()`` -> ``rocIsa.getVgprIdx()[name] +
          sum(offsets)``.
    """

    __slots__ = ("name", "offsets", "nameIdx")

    def __init__(self, name: str = "", offsets: Optional[List[int]] = None) -> None:
        self.name: str = name
        # Copy on assign to avoid sharing the caller's list.
        self.offsets: List[int] = list(offsets) if offsets else []
        self.nameIdx: int = 0

    # --- Offset accessors. ------------------------------------------------

    def getOffsets(self) -> List[int]:
        return self.offsets

    def setOffset(self, i: int, offset: int) -> None:
        if i >= len(self.offsets):
            raise IndexError("Index out of range")
        self.offsets[i] = offset

    def addOffset(self, offset: int) -> None:
        self.offsets.append(offset)

    def getTotalOffsets(self) -> int:
        return sum(self.offsets)

    # --- Symbol -> base-index resolution. ---------------------------------

    def setNameIdx(self) -> None:
        # Lazy import: avoids pulling the rocIsa singleton at module
        # import time (otherwise hits an adapter -> caps -> adapter loop).
        from . import rocIsa  # noqa: WPS433  (runtime import is intentional)

        self.nameIdx = rocIsa.getInstance().getVgprIdx().get(self.name, 0)

    def getTotalIdx(self) -> int:
        self.setNameIdx()
        return self.getTotalOffsets() + self.nameIdx

    # --- Dunder surface. --------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegName):
            return NotImplemented
        return self.name == other.name and self.offsets == other.offsets

    def __ne__(self, other: object) -> bool:
        eq = self.__eq__(other)
        return NotImplemented if eq is NotImplemented else not eq

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.offsets)))

    def __str__(self) -> str:
        if not self.offsets:
            return self.name
        return self.name + "".join("+" + str(o) for o in self.offsets)

    def __repr__(self) -> str:
        return f"RegName(name={self.name!r}, offsets={self.offsets!r})"

    def __deepcopy__(self, memo: dict) -> "RegName":
        return RegName(self.name, list(self.offsets))

    def __copy__(self) -> "RegName":
        return RegName(self.name, list(self.offsets))

    # Pickle support. Used by KernelWriter when shipping kernels across
    # worker processes.
    def __getstate__(self) -> Tuple[str, List[int]]:
        return (self.name, list(self.offsets))

    def __setstate__(self, state: Tuple[str, List[int]]) -> None:
        name, offsets = state
        self.name = name
        self.offsets = list(offsets)
        self.nameIdx = 0


# ---------------------------------------------------------------------------
# RegisterContainer -- register reference with rocisa-specific decorations.
# ---------------------------------------------------------------------------
#
# Stores the full attribute set (regType, regName, regIdx, regNum plus
# the isInlineAsm/isMacro/isOff/msb decoration flags) as plain Python
# fields and rebuilds a stinky.Register on demand via to_stinky().
#
# Why not subclass / hold an eager stinky.Register?
#   * Nanobind-exposed classes are not natural Python bases.
#   * Most attributes (isInlineAsm/isMacro/isOff/msb, regName as a
#     struct) have no stinkytofu equivalent; this wrapper has to be
#     the single source of truth.
#   * Lazy to_stinky avoids the build cost (and the unknown-regtype
#     validation) for wrappers that never reach emit.
class RegisterContainer(Container):
    """Reference to one or more architectural registers.

    Optional kwargs ``isAbs`` / ``isMacro`` / ``isOff`` mirror the
    extended ctor variant.

    Attributes (all mutable):
        regType (str)    -- "v" / "s" / "a" / "m".
        regName (RegName | None)
        regIdx  (int)
        regNum  (int)    -- stored as ``int(ceil(regNum))``.
        msb     (int)    -- transient, populated by ``setMsb``.
        isInlineAsm, isMinus, isAbs, isMacro, isOff (bool)
    """

    __slots__ = (
        "regType",
        "regName",
        "regIdx",
        "regNum",
        "msb",
        "isInlineAsm",
        "isMinus",
        "isAbs",
        "isMacro",
        "isOff",
    )

    def __init__(
        self,
        regType: str,
        regName: Optional[RegName],
        regIdx: int = 0,
        regNum: float = 1,
        *,
        isAbs: bool = False,
        isMacro: bool = False,
        isOff: bool = False,
    ) -> None:
        self.regType: str = regType
        self.regName: Optional[RegName] = regName
        self.regIdx: int = regIdx
        # Round up so regNum=1.5 -> 2.
        self.regNum: int = int(math.ceil(regNum))
        self.msb: int = 0
        self.isInlineAsm: bool = False
        self.isMinus: bool = False
        self.isAbs: bool = isAbs
        self.isMacro: bool = isMacro
        self.isOff: bool = isOff

    # --- Setters. ---------------------------------------------------------

    def setInlineAsm(self, setting: bool) -> None:
        self.isInlineAsm = setting

    def setMinus(self, isMinus: bool) -> None:
        self.isMinus = isMinus

    def setAbs(self, isAbs: bool) -> None:
        self.isAbs = isAbs

    def getMinus(self) -> "RegisterContainer":
        """Return a copy with ``isMinus=True`` (for negation operands)."""
        c = self._shallow_clone()
        c.setMinus(True)
        return c

    # --- regName mutation. ------------------------------------------------

    def replaceRegName(self, srcName: str, dst: Union[int, str]) -> None:
        """In-place substitute ``srcName`` in ``regName.name``.

        Two overloads:
            * ``dst: int``  -> exact match collapses the symbolic name
              into a literal index (``regIdx``); partial match substring-
              replaces ``str(dst)``.
            * ``dst: str``  -> substring-replace ``srcName`` with ``dst``.
        """
        if self.regName is None:
            return
        if isinstance(dst, bool):
            # bool is a subclass of int; reject so callers don't hit the
            # int overload accidentally.
            raise TypeError("replaceRegName dst must be int or str, not bool")
        if isinstance(dst, int):
            if self.regName.name == srcName:
                self.regIdx = dst + self.regName.getTotalOffsets()
                self.regName = None
                return
            pos = self.regName.name.find(srcName)
            if pos != -1:
                self.regName.name = (
                    self.regName.name[:pos]
                    + str(dst)
                    + self.regName.name[pos + len(srcName) :]
                )
            return
        if isinstance(dst, str):
            pos = self.regName.name.find(srcName)
            if pos != -1:
                self.regName.name = (
                    self.regName.name[:pos] + dst + self.regName.name[pos + len(srcName) :]
                )
            return
        raise TypeError(f"replaceRegName dst must be int or str, not {type(dst).__name__}")

    # --- Composite name accessors. ----------------------------------------

    def getRegNameWithType(self) -> str:
        # Crashes if regName is None (caller's responsibility).
        return self.regType + "gpr" + self.regName.name  # type: ignore[union-attr]

    def getCompleteRegNameWithType(self) -> str:
        return self.regType + "gpr" + str(self.regName)  # type: ignore[arg-type]

    def getCompleteRegName(self) -> str:
        return str(self.regName)  # type: ignore[arg-type]

    # --- splitRegContainer. -----------------------------------------------

    def splitRegContainer(self) -> Tuple["RegisterContainer", "RegisterContainer"]:
        """Split into two halves; second half is offset by 1 reg.

        Returns two independent instances; caller can mutate either side.
        """
        r1 = self._deep_clone()
        r2 = self._deep_clone()
        new_reg_num = math.ceil(self.regNum / 2)
        if self.regName is not None:
            r2.regName.addOffset(1)  # type: ignore[union-attr]
        else:
            r2.regIdx += 1
        r1.regNum = new_reg_num
        r2.regNum = self.regNum - new_reg_num
        return (r1, r2)

    # --- msb (HasVgprMSB byte-pair encoding). -----------------------------

    def setMsb(self) -> None:
        if self.regName is not None:
            self.msb = self.regName.getTotalIdx() // 256
        else:
            self.msb = self.regIdx // 256

    # --- Hash / equality. -------------------------------------------------

    def __hash__(self) -> int:
        return hash((self.regType, self.regIdx, self.regNum, self.regName))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegisterContainer):
            return False
        return (
            self.regType == other.regType
            and self.regIdx == other.regIdx
            and self.regNum == other.regNum
            and self.regName == other.regName
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    # --- Aliasing predicates. ---------------------------------------------

    def sameRegBaseAddr(self, b: "RegisterContainer") -> bool:
        if self.regName is not None and b.regName is not None:
            return self.regName.name == b.regName.name
        if self.regName is None and b.regName is None:
            return self.regIdx == b.regIdx
        return False

    def __and__(self, b: "RegisterContainer") -> bool:
        # Overlap check on the (offset, offset+regNum) interval.
        if not self.sameRegBaseAddr(b):
            return False
        len_a = self.regNum
        off_a = sum(self.regName.offsets) if self.regName is not None else 0
        len_b = b.regNum
        off_b = sum(b.regName.offsets) if b.regName is not None else 0
        range_a = (off_a, off_a + len_a)
        range_b = (off_b, off_b + len_b)
        if range_a[0] > range_b[0]:
            range_a, range_b = range_b, range_a
        return range_a[1] > range_b[0]

    # --- Stringification. -------------------------------------------------

    def toString(self) -> str:
        if self.isOff:
            return "off"

        minus_str = "-" if self.isMinus else ""
        if self.isAbs:
            minus_str = "abs(" + minus_str
        abs_str = ")" if self.isAbs else ""

        # Empty when HasVgprMSB cap isn't set or rocIsa wasn't initialised,
        # so __str__ stays side-effect-free for unit tests.
        msb_str = self._msb_suffix()

        if self.isInlineAsm:
            return minus_str + "%" + str(self.regIdx) + abs_str

        if self.regName is not None:
            macro_slash = "\\" if self.isMacro else ""
            if self.regNum == 1:
                return (
                    minus_str
                    + self.regType
                    + "["
                    + macro_slash
                    + self.regType
                    + "gpr"
                    + str(self.regName)
                    + msb_str
                    + "]"
                    + abs_str
                )
            return (
                minus_str
                + self.regType
                + "["
                + macro_slash
                + self.regType
                + "gpr"
                + str(self.regName)
                + msb_str
                + ":"
                + self.regType
                + "gpr"
                + str(self.regName)
                + msb_str
                + "+"
                + str(self.regNum - 1)
                + "]"
                + abs_str
            )

        if self.regNum == 1:
            if self.msb > 0:
                return (
                    minus_str
                    + self.regType
                    + "["
                    + str(self.regIdx)
                    + msb_str
                    + "]"
                    + abs_str
                )
            return minus_str + self.regType + str(self.regIdx) + abs_str

        return (
            minus_str
            + self.regType
            + "["
            + str(self.regIdx)
            + msb_str
            + ":"
            + str(self.regIdx + self.regNum - 1)
            + msb_str
            + "]"
            + abs_str
        )

    def _msb_suffix(self) -> str:
        """Compute the ``-256*msb`` suffix appended to ``toString``."""
        try:
            from . import rocIsa  # lazy import; see RegName.setNameIdx
        except ImportError:
            return ""
        inst = rocIsa.getInstance()
        if not inst.isInit():
            return ""
        try:
            asm_caps = inst.getAsmCaps()
        except RuntimeError:
            return ""
        if not asm_caps.get("HasVgprMSB", 0) or self.regType != "v":
            return ""
        self.setMsb()
        if self.msb > 0:
            return str(-256 * self.msb)
        return ""

    def __str__(self) -> str:
        return self.toString()

    def __repr__(self) -> str:
        return (
            f"RegisterContainer(regType={self.regType!r}, regName={self.regName!r}, "
            f"regIdx={self.regIdx}, regNum={self.regNum})"
        )

    # --- Copy semantics. --------------------------------------------------

    def _shallow_clone(self) -> "RegisterContainer":
        # regName is rebuilt as a new RegName so downstream mutation
        # (e.g. getMinus -> setMinus) does not bleed back.
        c = RegisterContainer.__new__(RegisterContainer)
        c.regType = self.regType
        c.regName = (
            RegName(self.regName.name, list(self.regName.offsets))
            if self.regName is not None
            else None
        )
        c.regIdx = self.regIdx
        c.regNum = self.regNum
        c.msb = self.msb
        c.isInlineAsm = self.isInlineAsm
        c.isMinus = self.isMinus
        c.isAbs = self.isAbs
        c.isMacro = self.isMacro
        c.isOff = self.isOff
        return c

    def _deep_clone(self) -> "RegisterContainer":
        # Independent regName so splitRegContainer.addOffset(1) on the
        # right half does not contaminate the left.
        return self._shallow_clone()

    def __copy__(self) -> "RegisterContainer":
        return self._shallow_clone()

    def __deepcopy__(self, memo: dict) -> "RegisterContainer":
        return self._shallow_clone()

    def __getstate__(self) -> Tuple[str, Optional[RegName], int, int, bool, bool, bool, bool, bool, int]:
        return (
            self.regType,
            deepcopy(self.regName),
            self.regIdx,
            self.regNum,
            self.isInlineAsm,
            self.isMinus,
            self.isAbs,
            self.isMacro,
            self.isOff,
            self.msb,
        )

    def __setstate__(
        self,
        state: Tuple[str, Optional[RegName], int, int, bool, bool, bool, bool, bool, int],
    ) -> None:
        (
            self.regType,
            self.regName,
            self.regIdx,
            self.regNum,
            self.isInlineAsm,
            self.isMinus,
            self.isAbs,
            self.isMacro,
            self.isOff,
            self.msb,
        ) = state

    # --- logicalIR handoff -------------------------------------------------

    def to_stinky(self):
        """Build a fresh ``stinky.Register`` from this wrapper's state.

        ``RegisterContainer`` is the source of truth; the stinky register
        is built fresh on each call so wrapper mutations
        (``replaceRegName``, ``setMinus``, ``setAbs``) become visible
        without an explicit sync step. rocisa-only emit decorations
        (``isMacro`` / ``isInlineAsm`` / ``msb`` / ``isOff``-as-flag) are
        translated here into stinky's vocabulary.

        Returns:
            ``stinky.Register`` -- register-typed for normal wrappers,
            ``LiteralString("off")`` for ``isOff=True``.

        Raises:
            NotImplementedError: when ``isInlineAsm=True``.

        TODO:
            - isMacro currently injects a backslash into the symbolic
              name as a stop-gap; T6 should route macros through proper
              TEXTBLOCK handling at the Module layer.
            - isInlineAsm has no stinkytofu path; T6 Module layer must
              gate inline-asm modules and route them back to rocisa
              native via ``$ROCISA_BACKEND``.
        """
        import stinkytofu as _stinky  # noqa: WPS433  (runtime: optional dep)

        if self.isOff:
            return _stinky.Register("off")

        if self.isInlineAsm:
            raise NotImplementedError(
                "RegisterContainer.to_stinky: isInlineAsm=True has no "
                "stinkytofu emit path (rocisa-only %-operand format). "
                "Module layer (T6) must route inline-asm modules back "
                "to rocisa native."
            )

        # Resolve physical index.
        if self.regName is not None and self.regType == "v":
            physical_idx = self.regName.getTotalIdx()
        else:
            physical_idx = self.regIdx

        # Route through the rocisa-style helper that python_bindings.cpp
        # exposes (vgpr/sgpr/accvgpr/mgpr). Keeps the emit boundary
        # symmetric with the rocisa factory surface and avoids encoding
        # the regType string twice.
        helper = {
            "v": _stinky.vgpr,
            "s": _stinky.sgpr,
            "acc": _stinky.accvgpr,
            "m": _stinky.mgpr,
        }.get(self.regType)
        if helper is None:
            raise NotImplementedError(
                f"RegisterContainer.to_stinky: regType={self.regType!r} has "
                "no stinkytofu helper. Add one in python_bindings.cpp first."
            )
        reg = helper(physical_idx, self.regNum)

        # VGPR MSB offset: for idx >= 256, the asm emitter needs
        # reg.offset = -(idx//256)*256 to produce "v[516-512]" format.
        # Mirrors native C++ ToStinkyTofuUtils.cpp getMsbOffsetFromStinkyVgpr().
        if self.regType == "v" and physical_idx >= 256:
            reg.set_offset(-(physical_idx // 256) * 256)

        if self.isMinus:
            reg.set_minus(True)
        if self.isAbs:
            reg.set_abs(True)

        if self.regName is not None:
            name = self.getRegNameWithType()
            offsets = list(self.regName.offsets)
            if self.isMacro:
                name = "\\" + name
            reg.set_reg_name(name, offsets)

        return reg


# ---------------------------------------------------------------------------
# generateRegName -- "Foo+1+2" -> RegName(name="Foo", offsets=[1, 2]).
# ---------------------------------------------------------------------------
#
# Parses a string-form symbolic ref back into a structured RegName.
# Inverse of RegName.__str__. Used by Holder(name=...) ctor.
_LEADING_INT_RE = re.compile(r"\s*([+-]?\d+)")


def _stoi(text: str) -> int:
    """Match C++ ``std::stoi``: parse the leading integer prefix, ignoring any
    trailing characters (e.g. ``"3.0"`` -> ``3``).

    KernelWriter sometimes builds symbolic offsets from float arithmetic
    (``idx / 4``), producing names like ``"G2LA+3.0"``. Native rocisa parses
    these via ``std::stoi`` which truncates at the first non-digit, so the
    adaptor must do the same to keep parity.
    """
    m = _LEADING_INT_RE.match(text)
    if not m:
        raise ValueError(f"invalid literal for _stoi(): {text!r}")
    return int(m.group(1))


def _generateRegName(rawText: str) -> RegName:
    parts = rawText.split("+")
    name = parts[0]
    offsets = [_stoi(p) for p in parts[1:]] if len(parts) > 1 else []
    return RegName(name, offsets)


# ---------------------------------------------------------------------------
# Holder -- deferred register reference resolved at replaceHolder time.
# ---------------------------------------------------------------------------
#
# An unresolved register reference: KernelWriter creates a Holder before
# knowing the final register index, hands it to an instruction inside a
# Module template, and later calls replaceHolder(module, dst) to resolve
# all Holders in the tree to concrete RegisterContainer instances.
#
# Two flavours:
#   * Holder(idx)   -- numeric base; resolved idx = holder_idx + dst
#   * Holder(name)  -- symbolic base ("Foo" or "Foo+1+2"); resolved
#                      regName = RegName(name, [dst, *parsed_offsets])
class Holder:
    """Deferred register reference (resolved later by ``replaceHolder``).

    Constructors:
        * ``Holder(idx)``  -- int. Stores ``idx``, leaves ``name=None``.
        * ``Holder(name)`` -- str. Stores ``idx=-1``, ``name=
          generateRegName(name)``.

    Attributes (both mutable):
        idx (int):              numeric base, or ``-1`` when ``name`` set.
        name (RegName | None):  symbolic base, or ``None`` when ``idx`` set.
    """

    __slots__ = ("idx", "name")

    def __init__(self, *args, **kwargs) -> None:
        # Dispatch on keyword first, then positional arg type. bool is
        # rejected so callers don't accidentally hit the int overload.
        if "idx" in kwargs and "name" not in kwargs:
            arg = kwargs["idx"]
            if isinstance(arg, bool):
                raise TypeError("Holder(idx=...): idx must be int, not bool")
            if not isinstance(arg, int):
                raise TypeError(
                    f"Holder(idx=...): idx must be int, got {type(arg).__name__}"
                )
            self.idx = arg
            self.name = None
            return
        if "name" in kwargs and "idx" not in kwargs:
            arg = kwargs["name"]
            if not isinstance(arg, str):
                raise TypeError(
                    f"Holder(name=...): name must be str, got {type(arg).__name__}"
                )
            self.idx = -1
            self.name = _generateRegName(arg)
            return
        if kwargs:
            raise TypeError(
                "Holder() takes exactly one of (idx, name); got: " + ", ".join(kwargs)
            )
        if len(args) != 1:
            raise TypeError("Holder() takes exactly one positional arg (idx or name)")
        arg = args[0]
        if isinstance(arg, bool):
            raise TypeError("Holder(): arg must be int or str, not bool")
        if isinstance(arg, int):
            self.idx = arg
            self.name = None
        elif isinstance(arg, str):
            self.idx = -1
            self.name = _generateRegName(arg)
        else:
            raise TypeError(
                f"Holder(): arg must be int or str, got {type(arg).__name__}"
            )

    def __repr__(self) -> str:
        return f"Holder(idx={self.idx}, name={self.name!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Holder):
            return NotImplemented
        return self.idx == other.idx and self.name == other.name

    def __ne__(self, other: object) -> bool:
        eq = self.__eq__(other)
        return NotImplemented if eq is NotImplemented else not eq

    def __hash__(self) -> int:
        return hash((self.idx, self.name))

    def __copy__(self) -> "Holder":
        c = Holder.__new__(Holder)
        c.idx = self.idx
        c.name = (
            RegName(self.name.name, list(self.name.offsets))
            if self.name is not None
            else None
        )
        return c

    def __deepcopy__(self, memo: dict) -> "Holder":
        return self.__copy__()

    # Pickle support.
    def __getstate__(self) -> Tuple[int, Optional[RegName]]:
        return (self.idx, self.name)

    def __setstate__(self, state: Tuple[int, Optional[RegName]]) -> None:
        self.idx, self.name = state


# ---------------------------------------------------------------------------
# HolderContainer -- RegisterContainer subclass with deferred resolution.
# ---------------------------------------------------------------------------
#
# Three ctors:
#   * (regType, holderName: str, regNum)   -- type 1, named via string
#   * (regType, regName: RegName, regNum)  -- type 1, named via RegName
#                                             (preserves RegName's
#                                             own offsets in holderOffsets)
#   * (regType, holderIdx: int, regNum)    -- type 0, numeric
#
# setRegNum(dst) is the deferred-resolution mutator: it mutates the
# parent RegisterContainer state in place using dst as the resolved
# offset/idx. replaceHolder() calls it before swapping the
# HolderContainer out for a plain RegisterContainer snapshot via
# getCopiedRC().
class HolderContainer(RegisterContainer):
    """Deferred-resolution ``RegisterContainer`` subclass.

    Two ``holderType`` modes:
        * ``holderType == 0``: numeric -- ``setRegNum(num)`` resolves
          ``regIdx = holderIdx + num``.
        * ``holderType == 1``: named -- ``setRegNum(num)`` resolves
          ``regName = RegName(holderName, [num, *holderOffsets])``.

    Attributes (all mutable):
        holderName (str)
        holderIdx (int)
        holderType (int)        -- 0 (numeric) or 1 (named)
        holderOffsets (list[int])
    """

    __slots__ = ("holderName", "holderIdx", "holderType", "holderOffsets")

    def __init__(self, regType: str, holderArg, regNum: float) -> None:
        # bool is a subclass of int; reject so True/False doesn't fall
        # into the numeric branch silently.
        if isinstance(holderArg, bool):
            raise TypeError(
                "HolderContainer: 2nd arg must be str/RegName/int, not bool"
            )
        if isinstance(holderArg, str):
            # Named via string: parent gets bare RegName(holderName, [])
            # so toString/emit still works before setRegNum lands.
            super().__init__(regType, RegName(holderArg), 0, regNum)
            self.holderName = holderArg
            self.holderIdx = 0
            self.holderType = 1
            self.holderOffsets: List[int] = []
        elif isinstance(holderArg, RegName):
            # Named via RegName: RegName.offsets become holderOffsets so
            # setRegNum can later prepend dst and re-append them.
            super().__init__(regType, holderArg, 0, regNum)
            self.holderName = holderArg.name
            self.holderIdx = 0
            self.holderType = 1
            self.holderOffsets = list(holderArg.offsets)
        elif isinstance(holderArg, int):
            # Numeric: parent regName=None, regIdx=holderIdx so toString
            # prints a placeholder numeric register pre-resolution.
            super().__init__(regType, None, holderArg, regNum)
            self.holderName = ""
            self.holderIdx = holderArg
            self.holderType = 0
            self.holderOffsets = []
        else:
            raise TypeError(
                f"HolderContainer: 2nd arg must be str/RegName/int, "
                f"got {type(holderArg).__name__}"
            )

    # --- Deferred resolution. ---------------------------------------------

    def setRegNum(self, num: int) -> None:
        """Resolve the holder using ``num`` as the produced offset/idx.

        Mutates the parent RegisterContainer state in place. After this
        call ``getCopiedRC()`` returns a fully-resolved snapshot.
        """
        if self.holderType == 0:
            self.regIdx = self.holderIdx + num
        elif self.holderType == 1:
            # Re-seed regName from holderName (drop stale offsets), then
            # prepend num and append the saved holderOffsets.
            self.regName = RegName(self.holderName)
            self.regName.offsets.insert(0, num)
            for off in self.holderOffsets:
                self.regName.offsets.append(off)

    # --- Snapshot to plain RegisterContainer. -----------------------------

    def getCopiedRC(self) -> "RegisterContainer":
        """Snapshot current state as a plain ``RegisterContainer``.

        Called by ``replaceHolder`` after ``setRegNum`` to substitute the
        HolderContainer with a fully-resolved RC. The snapshot is
        independent: subsequent mutation of either side does not bleed.
        """
        if self.holderType == 0:
            return RegisterContainer(self.regType, None, self.regIdx, self.regNum)
        return RegisterContainer(
            self.regType,
            RegName(self.regName.name, list(self.regName.offsets))
            if self.regName is not None
            else None,
            self.regIdx,
            self.regNum,
        )

    # --- splitRegContainer override. --------------------------------------

    def splitRegContainer(self) -> Tuple["HolderContainer", "HolderContainer"]:
        """Split into two HolderContainer halves.

        Differs from the parent in how the right half is shifted:

        * type 1 (named): push ``1`` onto ``r2.regName.offsets``.
        * type 0 (numeric): bump ``r2.holderIdx`` by 1 (the holder index
          shifts, not the resolved ``regIdx`` -- so subsequent
          ``setRegNum`` on r2 produces a shifted result).
        """
        r1 = self._shallow_clone()
        r2 = self._shallow_clone()
        new_reg_num = math.ceil(self.regNum / 2)
        if self.holderName:
            # Named: parent regName is already set so r2.regName is safe
            # to mutate.
            r2.regName.addOffset(1)  # type: ignore[union-attr]
        else:
            r2.holderIdx += 1
        r1.regNum = new_reg_num
        r2.regNum = self.regNum - new_reg_num
        return (r1, r2)

    # --- Copy semantics. --------------------------------------------------

    def _shallow_clone(self) -> "HolderContainer":  # type: ignore[override]
        c = HolderContainer.__new__(HolderContainer)
        # Parent fields.
        c.regType = self.regType
        c.regName = (
            RegName(self.regName.name, list(self.regName.offsets))
            if self.regName is not None
            else None
        )
        c.regIdx = self.regIdx
        c.regNum = self.regNum
        c.msb = self.msb
        c.isInlineAsm = self.isInlineAsm
        c.isMinus = self.isMinus
        c.isAbs = self.isAbs
        c.isMacro = self.isMacro
        c.isOff = self.isOff
        # Subclass fields.
        c.holderName = self.holderName
        c.holderIdx = self.holderIdx
        c.holderType = self.holderType
        c.holderOffsets = list(self.holderOffsets)
        return c

    def __copy__(self) -> "HolderContainer":  # type: ignore[override]
        return self._shallow_clone()

    def __deepcopy__(self, memo: dict) -> "HolderContainer":  # type: ignore[override]
        return self._shallow_clone()

    # --- Pickle. ----------------------------------------------------------

    def __getstate__(  # type: ignore[override]
        self,
    ) -> Tuple[str, int, int, str, Optional[RegName], int, int]:
        return (
            self.holderName,
            self.holderIdx,
            self.holderType,
            self.regType,
            deepcopy(self.regName),
            self.regIdx,
            self.regNum,
        )

    def __setstate__(  # type: ignore[override]
        self,
        state: Tuple[str, int, int, str, Optional[RegName], int, int],
    ) -> None:
        (
            holder_name,
            holder_idx,
            holder_type,
            reg_type,
            reg_name,
            reg_idx,
            reg_num,
        ) = state
        # Restore the snapshot verbatim, skipping ctor's holderType-derived
        # state setup. RC flags default to False (not preserved in state).
        self.regType = reg_type
        self.regName = reg_name
        self.regIdx = reg_idx
        self.regNum = reg_num
        self.msb = 0
        self.isInlineAsm = False
        self.isMinus = False
        self.isAbs = False
        self.isMacro = False
        self.isOff = False
        self.holderName = holder_name
        self.holderIdx = holder_idx
        self.holderType = holder_type
        # holderOffsets is not in the state tuple; re-derive from regName.
        self.holderOffsets = (
            list(reg_name.offsets) if reg_name is not None and reg_name.offsets else []
        )


# ---------------------------------------------------------------------------
# replaceHolder -- in-place holder resolution walker.
# ---------------------------------------------------------------------------
#
# Walks an instruction / module tree; when a HolderContainer is found in
# an Instruction's parameter list, calls its setRegNum(dst) and replaces
# the slot with the resolved RegisterContainer snapshot.
#
# Tree shape follows real ``Module.items()`` / ``Instruction.getParams()``
# from this adaptor (historical T6 note removed).
def replaceHolder(inst, dst: int):
    """Resolve all ``HolderContainer``s in ``inst`` using ``dst``.

    Walks Module-like (``.items()``) and Instruction-like
    (``.getParams()``) objects. For each ``HolderContainer`` found in an
    Instruction's parameter list, calls ``setRegNum(dst)`` then replaces
    the slot with the holder's ``getCopiedRC()`` (a plain
    ``RegisterContainer`` snapshot). Returns ``inst`` unchanged
    structurally; the resolution is in-place on the param lists.

    Args:
        inst: A Module-like, Instruction-like, or arbitrary object.
            Unknown types are passed through unchanged.
        dst (int): The base value to feed into each holder's
            ``setRegNum``.

    Returns:
        The same ``inst`` (mutated in place for Module/Instruction; left
        untouched otherwise).

    Raises:
        RuntimeError: if ``inst`` is an ``SWaitCnt`` (intentional gap).
    """
    # Detect by class name to avoid importing the dummy SWaitCnt class.
    if type(inst).__name__ == "SWaitCnt":
        raise RuntimeError("SWaitCnt is not supported yet")

    items_method = getattr(inst, "items", None)
    if callable(items_method):
        # Module-like: recurse into each child.
        for item in items_method():
            replaceHolder(item, dst)
        return inst

    get_params = getattr(inst, "getParams", None)
    if callable(get_params):
        # Instruction-like: walk parameter list, mutate any
        # HolderContainer slot in place.
        params = get_params()
        try:
            indices = range(len(params))
        except TypeError:
            # Non-indexable iterable (e.g. generator): can't mutate in place.
            return inst
        for i in indices:
            param = params[i]
            if isinstance(param, HolderContainer):
                param.setRegNum(dst)
                params[i] = param.getCopiedRC()
        return inst

    # Unknown / leaf object: return as-is.
    return inst


# ---------------------------------------------------------------------------
# GPR factory functions -- vgpr / sgpr / accvgpr / mgpr.
# ---------------------------------------------------------------------------
#
# Mirror of rocisa createGPR + the four type-specific factory wrappers.
# Three dispatch forms per factory: Holder / int / str. Each form maps
# to either a RegisterContainer (int / str) or a HolderContainer (Holder).
def _create_gpr(
    gpr_type: str,
    arg,
    reg_num: float,
    *,
    isMacro: bool = False,
    isAbs: bool = False,
    isOff: bool = False,
) -> RegisterContainer:
    """Internal dispatcher (rocisa::createGPR analogue).

    Dispatches on ``arg`` type: ``Holder`` → ``HolderContainer``,
    ``int`` → numeric ``RegisterContainer``, ``str`` → symbolic
    ``RegisterContainer`` whose ``regIdx`` defaults to ``-1``.
    """
    if isinstance(arg, Holder):
        if arg.idx == -1:
            return HolderContainer(gpr_type, arg.name, reg_num)
        return HolderContainer(gpr_type, arg.idx, reg_num)
    if isinstance(arg, bool):
        # bool is a subclass of int; reject so callers don't drift into
        # the int overload by accident.
        raise TypeError(
            f"{gpr_type}gpr factory: positional arg must be Holder/int/str, not bool"
        )
    if isinstance(arg, int):
        return RegisterContainer(gpr_type, None, arg, reg_num)
    if isinstance(arg, str):
        return RegisterContainer(
            gpr_type,
            _generateRegName(arg),
            -1,
            reg_num,
            isAbs=isAbs,
            isMacro=isMacro,
            isOff=isOff,
        )
    raise TypeError(
        f"{gpr_type}gpr factory: arg must be Holder/int/str, got {type(arg).__name__}"
    )


def vgpr(
    arg,
    regNum: float = 1.0,
    isMacro: bool = False,
    isAbs: bool = False,
    isOff: bool = False,
) -> RegisterContainer:
    """Build a VGPR ``RegisterContainer`` (or ``HolderContainer``).

    Three forms: ``vgpr(holder, regNum)``, ``vgpr(idx, regNum)``,
    ``vgpr(name, regNum, isMacro, isAbs, isOff)``.
    """
    return _create_gpr("v", arg, regNum, isMacro=isMacro, isAbs=isAbs, isOff=isOff)


def sgpr(arg, regNum: float = 1.0, isMacro: bool = False,
         isOff: bool = False) -> RegisterContainer:
    """Build an SGPR ``RegisterContainer`` (or ``HolderContainer``).

    String form matches rocisa ``sgpr(name, regNum, isMacro, isOff)``
    (container.hpp: ``isMacro`` + ``isOff``, no ``isAbs``).
    """
    return _create_gpr("s", arg, regNum, isMacro=isMacro, isOff=isOff)


def accvgpr(arg, regNum: float = 1.0) -> RegisterContainer:
    """Build an accumulator VGPR ``RegisterContainer`` (rocisa type ``"acc"``).

    No modifier kwargs in rocisa signature.
    """
    return _create_gpr("acc", arg, regNum)


def mgpr(arg, regNum: float = 1.0) -> RegisterContainer:
    """Build an MGPR (memory descriptor) ``RegisterContainer``.

    No modifier kwargs in rocisa signature.
    """
    return _create_gpr("m", arg, regNum)


# ---------------------------------------------------------------------------
# ContinuousRegister -- POD {idx, size} value object.
# ---------------------------------------------------------------------------
#
# RegisterPool yields one of these from each allocTmpGpr / allocTmpVgpr
# context; KernelWriter then reads .idx / .size off it. rocisa exposes
# the fields as ``def_ro`` so the instance is effectively immutable;
# we mirror that semantics in Python.
class ContinuousRegister:
    """Immutable ``{idx, size}`` register-range descriptor.

    Constructor: ``ContinuousRegister(idx, size)`` (positional or
    keyword). Both fields are read-only after construction; mutate by
    building a fresh instance.
    """

    __slots__ = ("_idx", "_size", "_frozen")

    def __init__(self, idx: int, size: int) -> None:
        # bool is an int subclass; reject so True/False don't slip in.
        if isinstance(idx, bool) or isinstance(size, bool):
            raise TypeError("ContinuousRegister: idx/size must be int, not bool")
        if not isinstance(idx, int) or not isinstance(size, int):
            raise TypeError(
                f"ContinuousRegister: idx/size must be int, got "
                f"{type(idx).__name__}/{type(size).__name__}"
            )
        object.__setattr__(self, "_idx", idx)
        object.__setattr__(self, "_size", size)
        object.__setattr__(self, "_frozen", True)

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def size(self) -> int:
        return self._size

    def __setattr__(self, name: str, value) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"ContinuousRegister is read-only; cannot set {name!r}"
            )
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return f"ContinuousRegister(idx={self._idx}, size={self._size})"

    def __copy__(self) -> "ContinuousRegister":
        return ContinuousRegister(self._idx, self._size)

    def __deepcopy__(self, memo: dict) -> "ContinuousRegister":
        return ContinuousRegister(self._idx, self._size)

    def __getstate__(self) -> Tuple[int, int]:
        return (self._idx, self._size)

    def __setstate__(self, state: Tuple[int, int]) -> None:
        idx, size = state
        object.__setattr__(self, "_idx", idx)
        object.__setattr__(self, "_size", size)
        object.__setattr__(self, "_frozen", True)


def _kernel_wavefront_size() -> int:
    """Current kernel wavefront from ``rocIsa.getInstance().getKernel()``."""
    from . import rocIsa  # lazy: avoid import cycle at module load

    return int(rocIsa.getInstance().getKernel().wavefrontSize)


# ---------------------------------------------------------------------------
# Hardware register tokens (T4) -- EXEC / VCC / HWRegContainer family.
# ---------------------------------------------------------------------------


class EXEC(Container):
    """Execution mask token; ``toString()`` depends on wavefront size."""

    __slots__ = ("setHi",)

    def __init__(self, setHi: bool = False) -> None:
        self.setHi = bool(setHi)

    def toString(self) -> str:
        if _kernel_wavefront_size() == 64:
            return "exec"
        return "exec_lo"

    def __repr__(self) -> str:
        return f"EXEC(setHi={self.setHi!r})"

    def __copy__(self) -> "EXEC":
        return EXEC(self.setHi)

    def __deepcopy__(self, memo: dict) -> "EXEC":
        return EXEC(self.setHi)

    def __getstate__(self) -> Tuple[bool]:
        return (self.setHi,)

    def __setstate__(self, state: Tuple[bool]) -> None:
        self.setHi = state[0]


class EXECLO(Container):
    """Low half of EXEC (32-lane wavefronts)."""

    __slots__ = ()

    def toString(self) -> str:
        return "exec_lo"

    def __repr__(self) -> str:
        return "EXECLO()"

    def __copy__(self) -> "EXECLO":
        return EXECLO()

    def __deepcopy__(self, memo: dict) -> "EXECLO":
        return EXECLO()

    def __getstate__(self) -> Tuple[()]:
        return ()

    def __setstate__(self, state: Tuple[()]) -> None:
        pass


class EXECHI(Container):
    """High half of EXEC (32-lane wavefronts)."""

    __slots__ = ()

    def toString(self) -> str:
        return "exec_hi"

    def __repr__(self) -> str:
        return "EXECHI()"

    def __copy__(self) -> "EXECHI":
        return EXECHI()

    def __deepcopy__(self, memo: dict) -> "EXECHI":
        return EXECHI()

    def __getstate__(self) -> Tuple[()]:
        return ()

    def __setstate__(self, state: Tuple[()]) -> None:
        pass


class VCC(Container):
    """Vector condition code token; ``toString()`` depends on wavefront / setHi."""

    __slots__ = ("setHi",)

    def __init__(self, setHi: bool = False) -> None:
        self.setHi = bool(setHi)

    def toString(self) -> str:
        if _kernel_wavefront_size() == 64:
            return "vcc"
        return "vcc_hi" if self.setHi else "vcc_lo"

    def __repr__(self) -> str:
        return f"VCC(setHi={self.setHi!r})"

    def __copy__(self) -> "VCC":
        return VCC(self.setHi)

    def __deepcopy__(self, memo: dict) -> "VCC":
        return VCC(self.setHi)

    def __getstate__(self) -> Tuple[bool]:
        return (self.setHi,)

    def __setstate__(self, state: Tuple[bool]) -> None:
        self.setHi = state[0]


class HWRegContainer(Container):
    """Hardware register immediate bundle for ``SSetRegIMM32B32`` operands."""

    __slots__ = ("reg", "value")

    def __init__(self, reg: str, value: List[int]) -> None:
        self.reg = reg
        self.value = list(value)

    def toString(self) -> str:
        parts = ",".join(str(v) for v in self.value)
        return f"hwreg({self.reg},{parts})"

    def __repr__(self) -> str:
        return f"HWRegContainer(reg={self.reg!r}, value={self.value!r})"

    def __copy__(self) -> "HWRegContainer":
        return HWRegContainer(self.reg, self.value)

    def __deepcopy__(self, memo: dict) -> "HWRegContainer":
        return HWRegContainer(self.reg, list(self.value))

    def __getstate__(self) -> Tuple[str, List[int]]:
        return (self.reg, list(self.value))

    def __setstate__(self, state: Tuple[str, List[int]]) -> None:
        self.reg, self.value = state[0], list(state[1])


# ---------------------------------------------------------------------------
# MemTokenData -- virtual scheduling fence metadata (T5).
# ---------------------------------------------------------------------------
#
# Attached to instructions via setMemToken(); emits no asm by itself but
# carries token ids for the dependency graph. C++ binding exposes a
# mutable ``tokens`` vector (def_rw).
class MemTokenData(Container):
    """Memory-token list carried on fence/barrier instructions.

    Constructor: ``MemTokenData(tokens=[])``. ``tokens`` is a mutable
    list of ints (mirrors C++ ``std::vector<int>``).
    """

    __slots__ = ("tokens",)

    def __init__(self, tokens: Optional[List[int]] = None) -> None:
        self.tokens: List[int] = list(tokens) if tokens is not None else []

    def toString(self) -> str:
        result = "mem_token:"
        for i, tok in enumerate(self.tokens):
            if i > 0:
                result += ","
            result += f" {tok}"
        return result

    def __repr__(self) -> str:
        return f"MemTokenData(tokens={self.tokens!r})"

    def __copy__(self) -> "MemTokenData":
        return MemTokenData(self.tokens)

    def __deepcopy__(self, memo: dict) -> "MemTokenData":
        return MemTokenData(list(self.tokens))

    def __getstate__(self) -> List[int]:
        return list(self.tokens)

    def __setstate__(self, state: List[int]) -> None:
        self.tokens = list(state)


# ---------------------------------------------------------------------------
# Instruction modifier descriptors (T5.2) — rocisa::Container subclasses.
# ---------------------------------------------------------------------------


def _asm_caps() -> dict:
    """Active ISA asm caps (requires ``rocIsa.init`` or ``setKernel``)."""
    from . import rocIsa  # noqa: WPS433

    return rocIsa.getInstance().getAsmCaps()


def _cache_scope_to_string(scope: CacheScope) -> str:
    """Mirror ``rocisa::toString(CacheScope)`` (``enum.hpp``)."""
    if scope == CacheScope.SCOPE_NONE:
        return ""
    return scope.name


def _select_bit_to_string(bit: SelectBit) -> str:
    if bit == SelectBit.SEL_NONE:
        return ""
    return bit.name


def _unused_bit_to_string(bit: UnusedBit) -> str:
    if bit == UnusedBit.UNUSED_NONE:
        return ""
    return bit.name


def _int_vector_to_string(vec: List[int]) -> str:
    inner = ",".join(str(v) for v in vec)
    return f"[{inner}]"


def _has_temporal_hint(th: TemporalHint) -> bool:
    """Mirror ``rocisa::hasTemporalHint`` (``enum.hpp``)."""
    return th != TemporalHint.TH_NONE


def _temporal_hint_to_string(th: TemporalHint, is_store: bool) -> str:
    """Mirror ``rocisa::toString(TemporalHint, bool isStore)`` (``enum.hpp``)."""
    prefix = "TH_STORE_" if is_store else "TH_LOAD_"
    if th == TemporalHint.TH_RT:
        return prefix + "RT"
    if th == TemporalHint.TH_NT:
        return prefix + "NT"
    if th == TemporalHint.TH_HT:
        return prefix + "HT"
    if th in (TemporalHint.TH_LU, TemporalHint.TH_WB):
        return prefix + ("WB" if is_store else "LU")
    if th == TemporalHint.TH_NT_RT:
        return prefix + "NT_RT"
    if th == TemporalHint.TH_RT_NT:
        return prefix + "RT_NT"
    if th == TemporalHint.TH_NT_HT:
        return prefix + "NT_HT"
    if th in (TemporalHint.TH_RESERVED, TemporalHint.TH_NT_WB):
        return prefix + ("NT_WB" if is_store else "RESERVED")
    return ""


def _non_volatile_to_string(nv: NonVolatile) -> str:
    """Mirror ``rocisa::toString(NonVolatile)`` (``enum.hpp``)."""
    return "nv" if nv == NonVolatile.NV else ""


class DSModifiers(Container):
    """LDS/GDS modifier bundle (``rocisa::DSModifiers``)."""

    __slots__ = ("na", "offset", "offset0", "offset1", "gds")

    def __init__(
        self,
        na: int = 1,
        offset: int = 0,
        offset0: int = 0,
        offset1: int = 0,
        gds: bool = False,
    ) -> None:
        self.na = na
        self.offset = offset
        self.offset0 = offset0
        self.offset1 = offset1
        self.gds = gds

    def toString(self) -> str:
        k_str = ""
        if self.na == 1:
            k_str += f" offset:{self.offset}"
        elif self.na == 2:
            k_str += f" offset0:{self.offset0} offset1:{self.offset1}"
        if self.gds:
            k_str += " gds"
        return k_str

    def __repr__(self) -> str:
        return (
            f"DSModifiers(na={self.na!r}, offset={self.offset!r}, "
            f"offset0={self.offset0!r}, offset1={self.offset1!r}, gds={self.gds!r})"
        )

    def __copy__(self) -> "DSModifiers":
        return DSModifiers(self.na, self.offset, self.offset0, self.offset1, self.gds)

    def __deepcopy__(self, memo: dict) -> "DSModifiers":
        return DSModifiers(self.na, self.offset, self.offset0, self.offset1, self.gds)

    def __getstate__(self) -> Tuple[int, int, int, int, bool]:
        return (self.na, self.offset, self.offset0, self.offset1, self.gds)

    def __setstate__(self, state: Tuple[int, int, int, int, bool]) -> None:
        self.na, self.offset, self.offset0, self.offset1, self.gds = state


class FLATModifiers(Container):
    """FLAT memory modifier bundle (``rocisa::FLATModifiers``)."""

    __slots__ = (
        "offset12", "glc", "slc", "dlc", "lds", "isStore", "scope", "th", "nv",
    )

    def __init__(
        self,
        offset12: int = 0,
        glc: bool = False,
        slc: bool = False,
        dlc: bool = False,
        lds: bool = False,
        isStore: bool = False,
        scope: CacheScope = CacheScope.SCOPE_NONE,
        th: TemporalHint = TemporalHint.TH_NONE,
        nv: NonVolatile = NonVolatile.NV_NONE,
    ) -> None:
        self.offset12 = offset12
        self.glc = glc
        self.slc = slc
        self.dlc = dlc
        self.lds = lds
        self.isStore = isStore
        self.scope = scope
        self.th = th
        self.nv = nv

    def toString(self) -> str:
        caps = _asm_caps()
        has_dlc = bool(caps.get("HasDLCModifier"))
        has_scope = bool(caps.get("HasSCOPEModifier"))
        has_th = bool(caps.get("HasTHModifier"))
        has_nv = bool(caps.get("HasNVModifier"))
        k_str = ""
        if self.offset12 != 0:
            k_str += f" offset:{self.offset12}"
        if self.glc:
            k_str += f" {glc_bit_name_from_caps(caps)}"
        if self.slc:
            k_str += f" {slc_bit_name_from_caps(caps)}"
        if has_dlc and self.dlc:
            k_str += " dlc"
        if has_scope and self.scope != CacheScope.SCOPE_NONE:
            k_str += f" scope:{_cache_scope_to_string(self.scope)}"
        if has_th and _has_temporal_hint(self.th):
            k_str += " th:" + _temporal_hint_to_string(self.th, self.isStore)
        if has_nv and self.nv != NonVolatile.NV_NONE:
            k_str += " " + _non_volatile_to_string(self.nv)
        if self.lds:
            k_str += " lds"
        return k_str

    def __repr__(self) -> str:
        return (
            f"FLATModifiers(offset12={self.offset12!r}, glc={self.glc!r}, "
            f"slc={self.slc!r}, dlc={self.dlc!r}, lds={self.lds!r}, "
            f"isStore={self.isStore!r}, scope={self.scope!r}, "
            f"th={self.th!r}, nv={self.nv!r})"
        )

    def __copy__(self) -> "FLATModifiers":
        return FLATModifiers(
            self.offset12, self.glc, self.slc, self.dlc,
            self.lds, self.isStore, self.scope, self.th, self.nv,
        )

    def __deepcopy__(self, memo: dict) -> "FLATModifiers":
        return self.__copy__()

    def __getstate__(
        self,
    ) -> Tuple[int, bool, bool, bool, bool, bool, int, int, int]:
        return (
            self.offset12, self.glc, self.slc, self.dlc,
            self.lds, self.isStore, int(self.scope), int(self.th), int(self.nv),
        )

    def __setstate__(
        self, state: Tuple[int, bool, bool, bool, bool, bool, int, int, int],
    ) -> None:
        (
            self.offset12, self.glc, self.slc, self.dlc,
            self.lds, self.isStore, scope_val, th_val, nv_val,
        ) = state
        self.scope = CacheScope(scope_val)
        self.th = TemporalHint(th_val)
        self.nv = NonVolatile(nv_val)


class GLOBALModifiers(Container):
    """GLOBAL memory modifier bundle (``rocisa::GLOBALModifiers``)."""

    __slots__ = ("offset", "th", "scope")

    def __init__(
        self,
        offset: int = 0,
        th: TemporalHint = TemporalHint.TH_NONE,
        scope: CacheScope = CacheScope.SCOPE_NONE,
    ) -> None:
        self.offset = offset
        self.th = th
        self.scope = scope

    def toString(self) -> str:
        k_str = ""
        if self.offset != 0:
            k_str += f" offset:{self.offset}"
        if _has_temporal_hint(self.th):
            k_str += " th:" + _temporal_hint_to_string(self.th, False)
        if self.scope != CacheScope.SCOPE_NONE:
            k_str += f" scope:{_cache_scope_to_string(self.scope)}"
        return k_str

    def __repr__(self) -> str:
        return (
            f"GLOBALModifiers(offset={self.offset!r}, th={self.th!r}, "
            f"scope={self.scope!r})"
        )

    def __copy__(self) -> "GLOBALModifiers":
        return GLOBALModifiers(self.offset, self.th, self.scope)

    def __deepcopy__(self, memo: dict) -> "GLOBALModifiers":
        return GLOBALModifiers(self.offset, self.th, self.scope)

    def __getstate__(self) -> Tuple[int, int, int]:
        return (self.offset, int(self.th), int(self.scope))

    def __setstate__(self, state: Tuple[int, int, int]) -> None:
        self.offset, th_val, scope_val = state
        self.th = TemporalHint(th_val)
        self.scope = CacheScope(scope_val)


class MUBUFModifiers(Container):
    """MUBUF memory modifier bundle (``rocisa::MUBUFModifiers``)."""

    __slots__ = (
        "offen", "offset12", "glc", "slc", "dlc", "nt", "lds", "isStore",
        "scope", "th", "nv",
    )

    def __init__(
        self,
        offen: bool = False,
        offset12: int = 0,
        glc: bool = False,
        slc: bool = False,
        dlc: bool = False,
        nt: bool = False,
        lds: bool = False,
        isStore: bool = False,
        scope: CacheScope = CacheScope.SCOPE_NONE,
        th: TemporalHint = TemporalHint.TH_NONE,
        nv: NonVolatile = NonVolatile.NV_NONE,
    ) -> None:
        self.offen = offen
        self.offset12 = offset12
        self.glc = glc
        self.slc = slc
        self.dlc = dlc
        self.nt = nt
        self.lds = lds
        self.isStore = isStore
        self.scope = scope
        self.th = th
        self.nv = nv

    def toString(self) -> str:
        caps = _asm_caps()
        has_dlc = bool(caps.get("HasDLCModifier"))
        has_scope = bool(caps.get("HasSCOPEModifier"))
        has_nt = bool(caps.get("HasNTModifier"))
        has_th = bool(caps.get("HasTHModifier"))
        has_nv = bool(caps.get("HasNVModifier"))
        k_str = ""
        if self.offen:
            k_str += f" offen offset:{self.offset12}"
        if self.glc:
            k_str += f" {glc_bit_name_from_caps(caps)}"
        if self.slc:
            k_str += f" {slc_bit_name_from_caps(caps)}"
        if has_dlc and self.dlc:
            k_str += " dlc"
        if has_scope and self.scope != CacheScope.SCOPE_NONE:
            k_str += f" scope:{_cache_scope_to_string(self.scope)}"
        if has_th and _has_temporal_hint(self.th):
            k_str += " th:" + _temporal_hint_to_string(self.th, self.isStore)
        elif has_nt and self.nt:
            k_str += " nt"
        if has_nv and self.nv != NonVolatile.NV_NONE:
            k_str += " " + _non_volatile_to_string(self.nv)
        if self.lds:
            k_str += " lds"
        return k_str

    def __repr__(self) -> str:
        return (
            f"MUBUFModifiers(offen={self.offen!r}, offset12={self.offset12!r}, "
            f"glc={self.glc!r}, slc={self.slc!r}, dlc={self.dlc!r}, "
            f"nt={self.nt!r}, lds={self.lds!r}, isStore={self.isStore!r}, "
            f"scope={self.scope!r}, th={self.th!r}, nv={self.nv!r})"
        )

    def __copy__(self) -> "MUBUFModifiers":
        return MUBUFModifiers(
            self.offen, self.offset12, self.glc, self.slc, self.dlc,
            self.nt, self.lds, self.isStore, self.scope, self.th, self.nv,
        )

    def __deepcopy__(self, memo: dict) -> "MUBUFModifiers":
        return self.__copy__()

    def __getstate__(
        self,
    ) -> Tuple[bool, int, bool, bool, bool, bool, bool, bool, int, int, int]:
        return (
            self.offen, self.offset12, self.glc, self.slc, self.dlc,
            self.nt, self.lds, self.isStore, int(self.scope),
            int(self.th), int(self.nv),
        )

    def __setstate__(
        self,
        state: Tuple[bool, int, bool, bool, bool, bool, bool, bool, int, int, int],
    ) -> None:
        (
            self.offen, self.offset12, self.glc, self.slc, self.dlc,
            self.nt, self.lds, self.isStore, scope_val, th_val, nv_val,
        ) = state
        self.scope = CacheScope(scope_val)
        self.th = TemporalHint(th_val)
        self.nv = NonVolatile(nv_val)


class SMEMModifiers(Container):
    """SMEM modifier bundle (``rocisa::SMEMModifiers``)."""

    __slots__ = ("glc", "dlc", "offset", "isStore", "scope", "th", "nv")

    def __init__(
        self,
        glc: bool = False,
        dlc: bool = False,
        offset: int = 0,
        isStore: bool = False,
        scope: CacheScope = CacheScope.SCOPE_NONE,
        th: TemporalHint = TemporalHint.TH_NONE,
        nv: NonVolatile = NonVolatile.NV_NONE,
    ) -> None:
        self.glc = glc
        self.dlc = dlc
        self.offset = offset
        self.isStore = isStore
        self.scope = scope
        self.th = th
        self.nv = nv

    def toString(self) -> str:
        caps = _asm_caps()
        has_dlc = bool(caps.get("HasDLCModifier"))
        has_scope = bool(caps.get("HasSCOPEModifier"))
        has_th = bool(caps.get("HasTHModifier"))
        has_nv = bool(caps.get("HasNVModifier"))
        k_str = ""
        if self.offset != 0:
            k_str += f" offset:{self.offset}"
        if not has_scope and self.glc:
            k_str += " glc"
        if has_dlc and self.dlc:
            k_str += " dlc"
        if has_scope and self.scope != CacheScope.SCOPE_NONE:
            k_str += f" scope:{_cache_scope_to_string(self.scope)}"
        if has_th and _has_temporal_hint(self.th):
            k_str += " th:" + _temporal_hint_to_string(self.th, self.isStore)
        if has_nv and self.nv != NonVolatile.NV_NONE:
            k_str += " " + _non_volatile_to_string(self.nv)
        return k_str

    def __repr__(self) -> str:
        return (
            f"SMEMModifiers(glc={self.glc!r}, dlc={self.dlc!r}, "
            f"offset={self.offset!r}, isStore={self.isStore!r}, "
            f"scope={self.scope!r}, th={self.th!r}, nv={self.nv!r})"
        )

    def __copy__(self) -> "SMEMModifiers":
        return SMEMModifiers(
            self.glc, self.dlc, self.offset, self.isStore,
            self.scope, self.th, self.nv,
        )

    def __deepcopy__(self, memo: dict) -> "SMEMModifiers":
        return SMEMModifiers(
            self.glc, self.dlc, self.offset, self.isStore,
            self.scope, self.th, self.nv,
        )

    def __getstate__(self) -> Tuple[bool, bool, int, bool, int, int, int]:
        return (
            self.glc, self.dlc, self.offset, self.isStore,
            int(self.scope), int(self.th), int(self.nv),
        )

    def __setstate__(self, state: Tuple[bool, bool, int, bool, int, int, int]) -> None:
        (
            self.glc, self.dlc, self.offset, self.isStore,
            scope_val, th_val, nv_val,
        ) = state
        self.scope = CacheScope(scope_val)
        self.th = TemporalHint(th_val)
        self.nv = NonVolatile(nv_val)


class SDWAModifiers(Container):
    """SDWA modifier bundle (``rocisa::SDWAModifiers``)."""

    __slots__ = ("dst_sel", "dst_unused", "src0_sel", "src1_sel")

    def __init__(
        self,
        dst_sel: SelectBit = SelectBit.SEL_NONE,
        dst_unused: UnusedBit = UnusedBit.UNUSED_NONE,
        src0_sel: SelectBit = SelectBit.SEL_NONE,
        src1_sel: SelectBit = SelectBit.SEL_NONE,
    ) -> None:
        self.dst_sel = dst_sel
        self.dst_unused = dst_unused
        self.src0_sel = src0_sel
        self.src1_sel = src1_sel

    def toString(self) -> str:
        k_str = ""
        if self.dst_sel != SelectBit.SEL_NONE:
            k_str += f" dst_sel:{_select_bit_to_string(self.dst_sel)}"
        if self.dst_unused != UnusedBit.UNUSED_NONE:
            k_str += f" dst_unused:{_unused_bit_to_string(self.dst_unused)}"
        if self.src0_sel != SelectBit.SEL_NONE:
            k_str += f" src0_sel:{_select_bit_to_string(self.src0_sel)}"
        if self.src1_sel != SelectBit.SEL_NONE:
            k_str += f" src1_sel:{_select_bit_to_string(self.src1_sel)}"
        return k_str

    def __repr__(self) -> str:
        return (
            f"SDWAModifiers(dst_sel={self.dst_sel!r}, "
            f"dst_unused={self.dst_unused!r}, src0_sel={self.src0_sel!r}, "
            f"src1_sel={self.src1_sel!r})"
        )

    def __copy__(self) -> "SDWAModifiers":
        return SDWAModifiers(
            self.dst_sel, self.dst_unused, self.src0_sel, self.src1_sel,
        )

    def __deepcopy__(self, memo: dict) -> "SDWAModifiers":
        return SDWAModifiers(
            self.dst_sel, self.dst_unused, self.src0_sel, self.src1_sel,
        )

    def __getstate__(self) -> Tuple[int, int, int, int]:
        return (
            int(self.dst_sel), int(self.dst_unused),
            int(self.src0_sel), int(self.src1_sel),
        )

    def __setstate__(self, state: Tuple[int, int, int, int]) -> None:
        self.dst_sel = SelectBit(state[0])
        self.dst_unused = UnusedBit(state[1])
        self.src0_sel = SelectBit(state[2])
        self.src1_sel = SelectBit(state[3])


class DPPModifiers(Container):
    """DPP modifier bundle (``rocisa::DPPModifiers``)."""

    __slots__ = ("row_shr", "row_bcast", "bound_ctrl", "quad_perm")

    def __init__(
        self,
        row_shr: int = -1,
        row_bcast: int = -1,
        bound_ctrl: int = -1,
        quad_perm: Optional[List[int]] = None,
    ) -> None:
        self.row_shr = row_shr
        self.row_bcast = row_bcast
        self.bound_ctrl = bound_ctrl
        self.quad_perm: List[int] = list(quad_perm) if quad_perm else []

    def toString(self) -> str:
        k_str = ""
        if self.row_shr != -1:
            k_str += f" row_shr:{self.row_shr}"
        if self.row_bcast != -1:
            k_str += f" row_bcast:{self.row_bcast}"
        if self.bound_ctrl != -1:
            k_str += f" bound_ctrl:{self.bound_ctrl}"
        if self.quad_perm:
            k_str += f" quad_perm:{_int_vector_to_string(self.quad_perm)}"
        return k_str

    def __repr__(self) -> str:
        return (
            f"DPPModifiers(row_shr={self.row_shr!r}, row_bcast={self.row_bcast!r}, "
            f"bound_ctrl={self.bound_ctrl!r}, quad_perm={self.quad_perm!r})"
        )

    def __copy__(self) -> "DPPModifiers":
        return DPPModifiers(
            self.row_shr, self.row_bcast, self.bound_ctrl, self.quad_perm,
        )

    def __deepcopy__(self, memo: dict) -> "DPPModifiers":
        return DPPModifiers(
            self.row_shr, self.row_bcast, self.bound_ctrl, list(self.quad_perm),
        )

    def __getstate__(self) -> Tuple[int, int, int, List[int]]:
        return (self.row_shr, self.row_bcast, self.bound_ctrl, list(self.quad_perm))

    def __setstate__(self, state: Tuple[int, int, int, List[int]]) -> None:
        self.row_shr, self.row_bcast, self.bound_ctrl, self.quad_perm = state
        self.quad_perm = list(self.quad_perm)


class VOP3PModifiers(Container):
    """VOP3P modifier bundle (``rocisa::VOP3PModifiers``)."""

    __slots__ = ("op_sel", "op_sel_hi", "byte_sel")

    def __init__(
        self,
        op_sel: Optional[List[int]] = None,
        op_sel_hi: Optional[List[int]] = None,
        byte_sel: Optional[List[int]] = None,
    ) -> None:
        self.op_sel: List[int] = list(op_sel) if op_sel else []
        self.op_sel_hi: List[int] = list(op_sel_hi) if op_sel_hi else []
        self.byte_sel: List[int] = list(byte_sel) if byte_sel else []

    def toString(self) -> str:
        k_str = ""
        if self.op_sel:
            k_str += f" op_sel:{_int_vector_to_string(self.op_sel)}"
        if self.op_sel_hi:
            k_str += f" op_sel_hi:{_int_vector_to_string(self.op_sel_hi)}"
        if self.byte_sel:
            k_str += f" byte_sel:{_int_vector_to_string(self.byte_sel)}"
        return k_str

    def __repr__(self) -> str:
        return (
            f"VOP3PModifiers(op_sel={self.op_sel!r}, "
            f"op_sel_hi={self.op_sel_hi!r}, byte_sel={self.byte_sel!r})"
        )

    def __copy__(self) -> "VOP3PModifiers":
        return VOP3PModifiers(self.op_sel, self.op_sel_hi, self.byte_sel)

    def __deepcopy__(self, memo: dict) -> "VOP3PModifiers":
        return VOP3PModifiers(
            list(self.op_sel), list(self.op_sel_hi), list(self.byte_sel),
        )

    def __getstate__(self) -> Tuple[List[int], List[int], List[int]]:
        return (list(self.op_sel), list(self.op_sel_hi), list(self.byte_sel))

    def __setstate__(self, state: Tuple[List[int], List[int], List[int]]) -> None:
        self.op_sel, self.op_sel_hi, self.byte_sel = state
        self.op_sel = list(self.op_sel)
        self.op_sel_hi = list(self.op_sel_hi)
        self.byte_sel = list(self.byte_sel)


class True16Modifiers(Container):
    """True16 high/low selector (``rocisa::True16Modifiers``)."""

    __slots__ = ("high_bit",)

    def __init__(self, high_bit: Union[HighBitSel, int] = HighBitSel.NONE) -> None:
        if isinstance(high_bit, int):
            self.high_bit = HighBitSel(high_bit)
        else:
            self.high_bit = high_bit

    def toString(self) -> str:
        if self.high_bit == HighBitSel.NONE:
            return ""
        return ".h" if self.high_bit == HighBitSel.HIGH else ".l"

    def __repr__(self) -> str:
        return f"True16Modifiers(high_bit={self.high_bit!r})"

    def __copy__(self) -> "True16Modifiers":
        return True16Modifiers(self.high_bit)

    def __deepcopy__(self, memo: dict) -> "True16Modifiers":
        return True16Modifiers(self.high_bit)

    def __getstate__(self) -> Tuple[int]:
        return (int(self.high_bit),)

    def __setstate__(self, state: Tuple[int]) -> None:
        self.high_bit = HighBitSel(state[0])

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Process-wide rocIsa singleton state and code-composition root types.

Provides KernelInfo, IsaInfo, OutputOptions, Item, and all rocIsa accessors.
Not yet done: IsaVersion is a marker only.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from . import caps as _caps
from ._dummy import make_dummy_class

_P = "rocisa.base"


IsaVersion = make_dummy_class(f"{_P}.IsaVersion")

# ``Item`` and ``DummyItem`` are real classes; they live at the bottom of
# this file because their default ``toString`` / ``prettyPrint`` / cap-
# proxy methods reach into the module-level accessors defined further
# down. See the "Item base class" section near the end.


# Re-export the public ISA key alias from caps for nicer typing downstream.
IsaKey = Tuple[int, int, int]


class OutputOptions:
    """Mirror of ``rocisa::OutputOptions`` (base.hpp:59-62) — mutable, picklable.

    The C++ struct only carries one bool today. We keep the Python shape
    identical so ``rocIsa.getInstance().getOutputOptions().outputNoComment =
    True`` and the subsequent ``setOutputOptions(opts)`` round-trip across
    multiprocessing pickles unchanged.
    """

    __slots__ = ("outputNoComment",)

    def __init__(self, outputNoComment: bool = False) -> None:
        self.outputNoComment = bool(outputNoComment)

    def __repr__(self) -> str:
        return f"OutputOptions(outputNoComment={self.outputNoComment})"

    # Pickle support — needed because Tensile passes this object across the
    # ParallelMap2 fork/spawn boundary.
    def __getstate__(self) -> tuple:
        return (self.outputNoComment,)

    def __setstate__(self, state: tuple) -> None:
        (self.outputNoComment,) = state


class IsaInfo:
    """Mirror of ``rocisa::IsaInfo`` (base.hpp:64-70) — asm/arch/reg/bug dicts.

    Moved here from ``__init__.py`` to honour the rocisa file layout
    (the C++ struct is declared in ``base.hpp`` alongside ``KernelInfo``
    / ``OutputOptions``). ``from rocisa import IsaInfo`` continues to
    work because ``__init__.py`` re-exports the symbol.

    Pickle support so workers spawned by ``ParallelMap2`` can rehydrate
    the data dict via ``setData(getData())``.
    """

    __slots__ = ("asmCaps", "archCaps", "regCaps", "asmBugs")

    def __init__(self, asmCaps, archCaps, regCaps, asmBugs):
        self.asmCaps = asmCaps
        self.archCaps = archCaps
        self.regCaps = regCaps
        self.asmBugs = asmBugs

    def __repr__(self) -> str:
        return (
            f"IsaInfo(asmCaps={self.asmCaps}, archCaps={self.archCaps}, "
            f"regCaps={self.regCaps}, asmBugs={self.asmBugs})"
        )

    def __getstate__(self) -> tuple:
        return (self.asmCaps, self.archCaps, self.regCaps, self.asmBugs)

    def __setstate__(self, state: tuple) -> None:
        self.asmCaps, self.archCaps, self.regCaps, self.asmBugs = state


class KernelInfo:
    """Mirror of ``rocisa::KernelInfo`` (base.hpp:47-57) — per-thread current
    kernel state.

    Only the attributes Tensile actually reads back are typed: ``isa``
    (a 3-tuple) and ``wavefrontSize``.
    """

    __slots__ = ("isa", "wavefrontSize")

    def __init__(self, isa=None, wavefrontSize: int = 0) -> None:
        self.isa = isa
        self.wavefrontSize = int(wavefrontSize)

    def __repr__(self) -> str:
        return f"KernelInfo(isa={self.isa}, wavefrontSize={self.wavefrontSize})"

    def __getstate__(self) -> tuple:
        return (self.isa, self.wavefrontSize)

    def __setstate__(self, state: tuple) -> None:
        self.isa, self.wavefrontSize = state


# ---------------------------------------------------------------------------
# Process-wide state mirror of ``rocisa::rocIsa`` (base.hpp:72-218).
# ---------------------------------------------------------------------------
#
# Module globals ARE the per-process singleton in Python — no Singleton
# class needed. The ``rocIsa`` class in ``__init__.py`` is a forwarding
# shell that preserves the rocisa API surface (KernelWriter / Tensile
# callers can keep doing ``rocIsa.getInstance().getXxx()``); the actual
# data lives down here.
#
# Each global below has a corresponding pair of accessor functions
# whose names track the C++ ``rocisa::rocIsa::*`` methods 1:1. Mutating
# the globals directly from outside this module is supported but
# discouraged — prefer the accessor functions so future refactors
# (e.g. switching to ``threading.local()`` if KernelWriter ever calls
# from worker threads) need only touch this file.
#
# Layout mirrors the order of declarations in rocisa C++ ``base.hpp``
# (lines 210-217) for easy side-by-side comparison.

_current_output_options: "OutputOptions" = OutputOptions()
"""Live ``OutputOptions`` for the codegen comment-suppression flag.
Mirror of ``rocisa::rocIsa::m_outputOptions`` (base.hpp:215)."""

_current_isa: Optional[IsaKey] = None
"""Currently-selected ISA, or ``None`` before ``init()`` / ``setKernel()``.
No direct C++ analogue (C++ derives the current ISA from the per-thread
``m_threads[id].isaVersion``); we keep a separate flag because Tensile
calls ``init()`` before any ``setKernel()`` to populate caps."""

_is_init: bool = False
"""``True`` once ``init()`` has populated at least one ISA's caps.
Mirror of ``rocisa::rocIsa::isInit()`` (base.hpp:103-106), which the
C++ derives from ``m_isainfo.size() > 0``."""

_assembler_path: str = ""
"""Path to the external assembler (``hipcc`` / ``llvm-mc``). Recorded by
``init()`` for parity with C++; the logical adaptor never shells out."""

_kernel_info: "KernelInfo" = KernelInfo()
"""Live ``KernelInfo`` for the currently-selected kernel. Mirror of
``rocisa::rocIsa::m_threads[id]`` (base.hpp:213) collapsed to a single
process-wide slot — see thread-semantics note in the module docstring."""

_data: Dict[IsaKey, "IsaInfo"] = {}
"""ISA-keyed snapshot of capability dicts. Mirror of
``rocisa::rocIsa::m_isainfo`` (base.hpp:214). ``ParallelMap2`` workers
pickle / unpickle this via ``getData()`` / ``setData()``."""

_vgpr_idx: Dict[str, int] = {}
"""Symbol -> base-VGPR-index map shared by ``RegName`` instances.
Mirror of ``rocisa::rocIsa::m_vgpridx[id]`` (base.hpp:216)."""

_vgpr_msb: int = 0
"""Current value of ``s_set_vgpr_msb`` (gfx1250 register-bank prefix).
Mirror of ``rocisa::rocIsa::m_vgprmsb[id]`` (base.hpp:217). Default 0
matches the C++ ``int()`` default-construct of a missing map entry.
``setVgprMsb`` is wired in for Commit Y (``Label.toString`` side
effect, code.hpp:122-125) — no consumer reads it today."""


# ---------------------------------------------------------------------------
# OutputOptions accessors (unchanged from the prior commit).
# ---------------------------------------------------------------------------

def getOutputOptions() -> OutputOptions:
    """Return the process-wide ``OutputOptions`` instance.

    The returned object IS the source of truth: mutating
    ``getOutputOptions().outputNoComment = True`` immediately affects
    every subsequent ``outputNoComment()`` call and every TextBlock
    rendered after this point. That mirrors rocisa C++ behaviour where
    ``rocIsa::getOutputOptions()`` returns a mutable reference.
    """
    return _current_output_options


def setOutputOptions(options: OutputOptions) -> None:
    """Replace the process-wide ``OutputOptions`` instance.

    Used by ``rocIsa.getInstance().setOutputOptions(...)`` -- which is
    what Tensile / ParallelMap2 workers call to ship a pickled
    ``OutputOptions`` from the parent into the worker process.
    """
    global _current_output_options
    _current_output_options = options


def outputNoComment() -> bool:
    """Return whether the codegen should suppress all comments / TextBlocks.

    Used by ``code.TextBlock.toString`` (gates the entire text payload,
    mirroring rocisa ``code.hpp:154-159``) and ``instruction._fmt_str``
    (drops the per-instruction ``// comment`` tail). One-line direct
    read of the module-level singleton: no lazy import, no try/except,
    no defensive ``getattr`` -- the state object is guaranteed to exist
    because we eagerly construct it at module load.
    """
    return _current_output_options.outputNoComment


# ---------------------------------------------------------------------------
# ISA init & active-ISA accessors.
# ---------------------------------------------------------------------------
#
# These four collectively mirror ``rocisa::rocIsa::init`` /
# ``isInit`` / ``getIsaInfo`` and the four ``get{Asm,Arch,Reg}Caps`` /
# ``getAsmBugs`` proxies (base.hpp:88-155).

def init(arch: Any, assemblerPath: str = "", debug: bool = False) -> None:
    """Mirror of ``rocisa::rocIsa::init`` (base.hpp:88-101).

    Selects an ISA, lazily populates its capability snapshot from
    ``caps.getCaps``, records the assembler path, and marks the
    singleton as initialised. Idempotent for the same ISA — the C++
    version short-circuits on ``m_isainfo.find(isaVersion) !=
    m_isainfo.end()``, we mirror that with ``if key not in _data``.

    ``debug`` is accepted for signature parity but ignored. Capability
    probing is delegated to ``stinkytofu.getHardwareCaps`` via
    ``caps.getCaps`` (comgr inside stinkytofu, not ``llvm-mc`` here).

    For unsupported ISAs, the call falls back to the native ``_rocisa``
    extension transparently.
    """
    global _current_isa, _assembler_path, _is_init
    key = _caps.normalize_isa_key(arch)
    _current_isa = key
    _assembler_path = assemblerPath
    if key not in _data:
        try:
            asm, arch_c, reg, bugs = _caps.getCaps(key)
        except KeyError:
            _init_native_fallback(arch, assemblerPath, debug, key)
            _is_init = True
            return
        _data[key] = IsaInfo(asm, arch_c, reg, bugs)
    _is_init = True
    del debug


def _init_native_fallback(arch: Any, assemblerPath: str, debug: bool,
                          key: "tuple") -> None:
    """Query ISA caps from native rocisa via a subprocess.

    Loading ``_rocisa.so`` in-process would conflict with the already-loaded
    ``_stinkytofu.so`` (nanobind duplicate type registration).  A subprocess
    gets its own address space, so both extensions can coexist safely.
    """
    import json
    import os
    import subprocess
    import sys
    import textwrap

    isa_tuple = tuple(int(x) for x in key)
    script = textwrap.dedent(f"""\
        import os, json
        os.environ["ROCISA_BACKEND"] = "rocisa"
        import rocisa
        ti = rocisa.rocIsa.getInstance()
        arch = {isa_tuple!r}
        ti.init(arch, {assemblerPath!r}, {debug!r})
        info = ti.getIsaInfo(arch)
        print(json.dumps({{
            "asmCaps":  {{str(k): int(v) for k, v in dict(info.asmCaps).items()}},
            "archCaps": {{str(k): int(v) for k, v in dict(info.archCaps).items()}},
            "regCaps":  {{str(k): int(v) for k, v in dict(info.regCaps).items()}},
            "asmBugs":  {{str(k): int(v) for k, v in dict(info.asmBugs).items()}},
        }}))
    """)

    env = {**os.environ, "ROCISA_BACKEND": "rocisa"}
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Native rocisa fallback subprocess failed for ISA {key}:\n"
            f"{result.stderr}"
        )
    data = json.loads(result.stdout)
    _data[key] = IsaInfo(
        data["asmCaps"], data["archCaps"], data["regCaps"], data["asmBugs"],
    )


def isInit() -> bool:
    """Mirror of ``rocisa::rocIsa::isInit`` (base.hpp:103-106).

    C++ derives this from ``m_isainfo.size() > 0``; we use the explicit
    ``_is_init`` flag so ``init()`` and ``setData()`` can both flip it
    deterministically.
    """
    return _is_init


def getIsaInfo(arch: Any) -> "IsaInfo":
    """Mirror of ``rocisa::rocIsa::getIsaInfo`` (base.hpp:125-135).

    Lazy-populates the cache so callers can ask for any supported ISA
    without a prior ``init()`` call (Tensile's
    ``Tensile.Common.Capabilities.makeIsaInfoMap`` relies on this).

    Falls back to native ``_rocisa`` for ISAs not supported by stinkytofu.
    """
    key = _caps.normalize_isa_key(arch)
    info = _data.get(key)
    if info is None:
        try:
            asm, arch_c, reg, bugs = _caps.getCaps(key)
        except KeyError:
            _init_native_fallback(arch, "", False, key)
            return _data[key]
        info = IsaInfo(asm, arch_c, reg, bugs)
        _data[key] = info
    return info


def _activeCaps() -> Tuple[Dict, Dict, Dict, Dict]:
    """Internal: return ``(asmCaps, archCaps, regCaps, asmBugs)`` for the
    currently-selected ISA. Used by the four public ``get*Caps`` helpers.

    Raises ``RuntimeError`` if no ISA has been selected yet so a missing
    ``init()`` / ``setKernel()`` produces a loud error rather than a
    confusing ``KeyError`` deep inside instruction emission.
    """
    if _current_isa is None:
        raise RuntimeError(
            "rocisa.base: init(arch, ...) or setKernel(arch, ...) must "
            "be called before getAsmCaps()/getArchCaps()/getRegCaps()/"
            "getAsmBugs()."
        )
    info = _data.get(_current_isa)
    if info is None:
        asm, arch_c, reg, bugs = _caps.getCaps(_current_isa)
        info = IsaInfo(asm, arch_c, reg, bugs)
        _data[_current_isa] = info
    return (info.asmCaps, info.archCaps, info.regCaps, info.asmBugs)


def getAsmCaps() -> Dict[str, int]:
    """Mirror of ``rocisa::rocIsa::getAsmCaps`` (base.hpp:137-140)."""
    return _activeCaps()[0]


def getArchCaps() -> Dict[str, int]:
    """Mirror of ``rocisa::rocIsa::getArchCaps`` (base.hpp:147-150)."""
    return _activeCaps()[1]


def getRegCaps() -> Dict[str, int]:
    """Mirror of ``rocisa::rocIsa::getRegCaps`` (base.hpp:142-145)."""
    return _activeCaps()[2]


def getAsmBugs() -> Dict[str, bool]:
    """Mirror of ``rocisa::rocIsa::getAsmBugs`` (base.hpp:152-155)."""
    return _activeCaps()[3]


# ---------------------------------------------------------------------------
# KernelInfo accessors.
# ---------------------------------------------------------------------------

def setKernel(arch: Any, wavefrontSize: int) -> None:
    """Mirror of ``rocisa::rocIsa::setKernel`` (base.hpp:108-118).

    Pins the active ISA AND records the wavefront size in the live
    ``KernelInfo``. NOTE: the C++ version ALSO resets ``m_vgpridx[id]``
    and ``m_vgprmsb[id]`` on every call (base.hpp:115-116). The Python
    adaptor historically did NOT mirror that reset; preserving that
    omission here to keep this commit a pure state-sink (no behaviour
    change). Re-evaluate when wiring Label.toString in Commit Y.
    """
    global _current_isa, _kernel_info
    key = _caps.normalize_isa_key(arch)
    _current_isa = key
    _kernel_info = KernelInfo(isa=key, wavefrontSize=wavefrontSize)


def setKernelInfo(info: KernelInfo) -> None:
    """Replace the live ``KernelInfo`` directly. No C++ equivalent.

    Used by callers that need to restore a previously captured ``KernelInfo`` --
    in particular the initial ``KernelInfo()`` state where ``info.isa is None``,
    which ``setKernel`` cannot represent (it requires a concrete ISA tuple).

    Also resets ``_current_isa`` to match, so capability lookups
    (``getAsmCaps`` / ``getAsmBugs`` / ...) go back to raising rather than
    silently continuing to answer for whatever ISA was pinned before this call
    -- otherwise ``getKernel()`` would report the restored (unpinned) state
    while capability queries stayed pinned to the old ISA.
    """
    global _kernel_info, _current_isa
    _kernel_info = info
    _current_isa = info.isa


def getKernel() -> KernelInfo:
    """Mirror of ``rocisa::rocIsa::getKernel`` (base.hpp:120-123)."""
    return _kernel_info


def isaToGfx(arch: Any) -> str:
    """Mirror of ``rocisa::isaToGfx`` / ``getGfxNameTuple`` (helper.hpp).

    Accepts a 3-tuple/list ``(major, minor, stepping)`` or any object
    with ``major / minor / stepping`` attributes (rocisa ``IsaVersion``).
    """
    if hasattr(arch, "major"):
        major = int(arch.major)
        minor = int(arch.minor)
        stepping = int(arch.stepping)
    else:
        major = int(arch[0])
        minor = int(arch[1])
        stepping = int(arch[2])
    hex_digit = "0123456789abcdef"[stepping & 0xF]
    return f"gfx{major}{minor}{hex_digit}"


# ---------------------------------------------------------------------------
# Data-dict accessors (pickled across ParallelMap2 worker boundary).
# ---------------------------------------------------------------------------

def getData() -> Dict[IsaKey, "IsaInfo"]:
    """Mirror of ``rocisa::rocIsa::getData`` (base.hpp:157-160).

    Returns the live dict (NOT a copy). Tensile pickles the result and
    ships it to ``ParallelMap2`` workers, which call ``setData`` to
    rehydrate. Mutating the returned dict mutates the process-wide
    state -- usually what you want.
    """
    return _data


def setData(data: Dict[IsaKey, "IsaInfo"]) -> None:
    """Mirror of ``rocisa::rocIsa::setData`` (base.hpp:172-175).

    Replaces the cap snapshot table wholesale. Also flips ``_is_init``
    True/False based on emptiness so ``isInit()`` matches the C++
    behaviour (``m_isainfo.size() > 0``).
    """
    global _data, _is_init
    _data = dict(data)
    _is_init = bool(_data)


# ---------------------------------------------------------------------------
# VGPR-index map accessors.
# ---------------------------------------------------------------------------

def getVgprIdx() -> Dict[str, int]:
    """Mirror of ``rocisa::rocIsa::getVgprIdx`` (base.hpp:162-165).

    Returns the live dict so callers can ``getVgprIdx()[name]`` for
    a single lookup AND ``getVgprIdx().clear()`` for a process-wide
    reset (used by test harnesses). The C++ version returns by value
    so ``.clear()`` would not propagate there; Python dict semantics
    make the mutation visible, which matches what callers actually
    want.
    """
    return _vgpr_idx


def setVgprIdx(name: str, idx: int) -> None:
    """Mirror of ``rocisa::rocIsa::setVgprIdx`` (base.hpp:191-198)."""
    _vgpr_idx[name] = idx


# ---------------------------------------------------------------------------
# VGPR-MSB accessors.
# ---------------------------------------------------------------------------

def getVgprMsb() -> int:
    """Mirror of ``rocisa::rocIsa::getVgprMsb`` (base.hpp:167-170).

    Default ``0`` matches the C++ ``int()`` default-construct of an
    absent ``m_vgprmsb[id]`` entry.
    """
    return _vgpr_msb


def setVgprMsb(msb: int) -> None:
    """Mirror of ``rocisa::rocIsa::setVgprMsb`` (base.hpp:200-207).

    Used by ``Label::toString`` (code.hpp:122-125) as a side effect
    when the active ISA has ``HasVgprMSB``. No Python consumer reads
    it today; the accessor is wired in here so Commit Y (Label real
    implementation) can pick it up without further base.py changes.
    """
    global _vgpr_msb
    _vgpr_msb = int(msb)


# ---------------------------------------------------------------------------
# Item base class -- polymorphic root of the code composition tree.
# ---------------------------------------------------------------------------
#
# Mirrors ``rocisa::Item`` (base.hpp:220-297). Lives at the bottom of
# this file so its capability-proxy methods can refer to the module-
# level accessors above by simple name (Python's LEGB lookup resolves
# ``getAsmCaps`` etc. to the module globals, not to ``self.getAsmCaps``
# which would recurse -- method names are not in the method's local
# scope, only in the class dict).
#
# Design choices vs. C++:
#   * ``Item`` is a regular (concrete) class, NOT an ``abc.ABC``.
#     ``Item("foo")`` is valid (matches C++ where ``rocisa::Item it("foo")``
#     compiles and ``.toString()`` returns "foo"); only ``clone()`` raises
#     by default.
#   * ``__slots__ = ("name", "parent")`` -- subclasses (Module / TextBlock /
#     ...) must NOT redeclare these. They declare only their own new
#     slots; ``name`` / ``parent`` are inherited.
#   * Capability proxies (``getAsmCaps`` / ``getArchCaps`` / ``getRegCaps`` /
#     ``getAsmBugs`` / ``getVgprIdx`` / ``getVgprMsb`` / ``kernel``) all
#     forward to the module-level accessors -- they reach module-level
#     ``base.py`` state, NOT the ``rocIsa`` singleton class. That keeps
#     the dependency direction one-way and means an Item subclass can
#     read caps even if the package facade hasn't been imported yet.
#   * ``countType`` accepts a Python ``type`` (not a nanobind ``nb::object``)
#     and uses ``isinstance``; ``countExactType`` checks ``type(self) is
#     target`` to match the C++ ``typeid(*this) == targetType`` semantics.

class Item:
    """Polymorphic root of the code composition tree; mirror of
    ``rocisa::Item`` (base.hpp:220-297).

    Concrete (not abstract): ``Item("foo").toString()`` returns ``"foo"``
    and ``Item().prettyPrint()`` returns ``"<class-name> "``; only
    ``clone()`` raises by default, matching the C++ ``throw
    std::runtime_error("clone() not implemented")``.

    Subclassing:
        Subclasses declare only their own additional ``__slots__`` --
        ``name`` and ``parent`` come from this class. ``__init__``
        must call ``super().__init__(name)`` (or
        ``super().__init__()`` for the default empty name) so the
        inherited slots get populated.

    Capability proxies (``getAsmCaps`` etc.) forward to the module-
    level accessors in this file; an Item subclass need never know
    that the actual state lives in ``_data`` / ``_kernel_info`` /
    ``_vgpr_idx`` globals.

    Item-inherited proxies NOT exposed on ``Module`` instances
    (matches Phase-4 audit -- KernelWriter only reads these off
    Instruction subclasses) are still present on the base class
    itself, so explicit ``isinstance(x, Item)`` callers get the
    full surface.
    """

    __slots__ = ("name", "parent")

    def __init__(self, name: str = "") -> None:
        self.name: str = name
        self.parent: "Item | None" = None

    # ---- Polymorphic operations with default impls -----------------------

    def toString(self) -> str:
        """Default: return ``name`` (mirror of ``Item::toString``,
        base.hpp:282-285)."""
        return self.name

    def __str__(self) -> str:
        # Bound to toString() so subclass overrides automatically
        # propagate to ``str(item)``, matching nanobind's
        # ``.def("__str__", &Item::toString)`` pattern.
        return self.toString()

    def prettyPrint(self, indent: str = "") -> str:
        """Default: ``indent + className + " " + toString()``
        (mirror of ``Item::prettyPrint``, base.hpp:287-293).

        Note the trailing space + ``toString()`` with NO newline --
        callers (e.g. ``Module::prettyPrint``) concatenate child
        ``prettyPrint`` results verbatim and the children that need
        a newline (``Module``, ``Instruction``) emit it themselves
        in their override.
        """
        return f"{indent}{type(self).__name__} {self.toString()}"

    def countType(self, type_obj: type) -> int:
        """Default: ``1`` if ``self`` is an instance of ``type_obj``,
        else ``0`` (mirror of ``Item::countType``, base.hpp:271-275).

        C++ accepts an ``nb::object`` (i.e. a Python class object) and
        uses ``nb::isinstance``; the Python adaptor accepts a regular
        ``type`` directly and uses the builtin ``isinstance``.
        Subclasses with children (Module, Macro, StructuredModule)
        override this to recurse.
        """
        return int(isinstance(self, type_obj))

    def countExactType(self, type_obj: type) -> int:
        """Default: ``1`` if ``type(self) is type_obj``, else ``0``
        (mirror of ``Item::countExactType``, base.hpp:277-280).

        The exact-type check uses ``is`` rather than ``isinstance``
        to match C++ ``typeid(*this) == targetType`` semantics -- a
        ``StructuredModule`` does NOT count as a ``Module`` here even
        though it inherits from it. Containers (Module, Macro,
        StructuredModule) override this to recurse.
        """
        return int(type(self) is type_obj)

    def clone(self) -> "Item":
        """Mirror of ``Item::clone`` (base.hpp:230-234) -- raises by
        default, subclasses override (the C++ original throws
        ``std::runtime_error("clone() not implemented")``). Python
        subclasses typically provide ``__deepcopy__`` instead; this
        method is here for rocisa API parity."""
        raise NotImplementedError("clone() not implemented")

    # ---- Capability proxies (forward to module-level accessors) ----------
    #
    # Mirror of the seven ``def_prop_ro`` / member-function proxies on
    # ``rocisa::Item`` (base.hpp:236-269 + base.cpp:202-212). All of
    # them defer to the active-ISA state in ``base.py`` module globals.
    #
    # NB: writing ``return getAsmCaps()`` resolves to the module-level
    # ``getAsmCaps`` defined above (Python LEGB: local -> enclosing ->
    # module-globals -> builtins; class-method names are NOT in scope
    # inside the method body). No self-recursion.

    def getAsmCaps(self) -> Dict[str, int]:
        """Mirror of ``Item::getAsmCaps`` (base.hpp:236-239)."""
        return getAsmCaps()

    def getArchCaps(self) -> Dict[str, int]:
        """Mirror of ``Item::getArchCaps`` (base.hpp:246-249)."""
        return getArchCaps()

    def getRegCaps(self) -> Dict[str, int]:
        """Mirror of ``Item::getRegCaps`` (base.hpp:241-244)."""
        return getRegCaps()

    def getAsmBugs(self) -> Dict[str, bool]:
        """Mirror of ``Item::getAsmBugs`` (base.hpp:251-254)."""
        return getAsmBugs()

    def getVgprIdx(self) -> Dict[str, int]:
        """Mirror of ``Item::getVgprIdx`` (base.hpp:256-259)."""
        return getVgprIdx()

    def getVgprMsb(self) -> int:
        """Mirror of ``Item::getVgprMsb`` (base.hpp:261-264)."""
        return getVgprMsb()

    def kernel(self) -> "KernelInfo":
        """Mirror of ``Item::kernel`` (base.hpp:266-269)."""
        return getKernel()


class DummyItem(Item):
    """Mirror of ``rocisa::DummyItem`` (base.hpp:299-310).

    A no-op Item subclass that exists purely so KernelWriter can stick
    a marker into a Module tree without triggering ``countType`` /
    instance accounting. The C++ override returns 0 regardless of the
    queried type; we mirror that.
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(name="")

    def countType(self, type_obj: type) -> int:
        # rocisa C++ override (base.hpp:306-309) hardcodes 0.
        return 0

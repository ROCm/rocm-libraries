# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The toolchain pieces the tests share: one ``Assembler`` and one capability map per ISA.

Tests need an ``Assembler`` (what ``Solution`` reads ``rocm_version`` and the
code-object version from) and a capability map (``isaInfoMap``) for the ISA they
target. Both are built once per process and shared, so they are read-only: the
maps refuse changes with a ``TypeError``. A test that needs to change one, for
example with ``applyArchCapOverrides``, takes a copy first with
``copy_isa_info_map``.

Building a map shells out to ``amdclang++`` once per ISA per process (rocisa
caches the probe), about 1.5 s for the first use of an ISA.

Import from tests by bare module name (``from toolchain_test_helpers import ...``):
the tests are not part of the installed package, so ``Tests/`` is put on
``sys.path`` by conftest.py, as for rocisa_test_state.
"""

import copy
import functools
import multiprocessing
import os

import pytest

# The compiler every helper here probes with.
CXX_COMPILER = "amdclang++"

# Architectures every toolchain the tests run against has to be able to target.
# A map that reports one of these as unsupported is a toolchain regression, so
# ``shared_isa_info_map_or_skip`` fails on it. Newer architectures, which an
# older compiler may not know yet, are skipped.
_ALWAYS_SUPPORTED_ARCHS = ("gfx90a", "gfx942", "gfx950")

_READ_ONLY = (
    "the shared capability map is read-only; take a copy with "
    "copy_isa_info_map(...) to change one"
)

_ISA_INFO_FIELDS = ("asmCaps", "archCaps", "regCaps", "asmBugs")


def cxx_compiler():
    """Path of the C++ compiler the capability probe uses."""
    from tensilelite.Toolchain.Validators import validateToolchain

    return validateToolchain(CXX_COMPILER)


@functools.lru_cache(maxsize=None)
def shared_assembler(codeObjectVersion="default"):
    """The process-wide ``Assembler`` for ``codeObjectVersion``.

    ``"default"`` is harmless while nothing invokes the assembler but is
    rejected by clang as ``-mcode-object-version=default``; callers that do
    assemble pass a real version.
    """
    from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
    from tensilelite.Toolchain.Validators import ToolchainDefaults, validateToolchain

    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx_compiler(), bundler, codeObjectVersion).assembler


class _ReadOnlyDict(dict):
    """A dict that refuses changes. Copying it, with ``copy.deepcopy`` or
    ``copy_isa_info_map``, gives an ordinary mutable dict."""

    def _refuse(self, *args, **kwargs):
        raise TypeError(_READ_ONLY)

    __setitem__ = __delitem__ = __ior__ = _refuse
    clear = pop = popitem = setdefault = update = _refuse

    def __copy__(self):
        return dict(self)

    def __deepcopy__(self, memo):
        return {copy.deepcopy(k, memo): copy.deepcopy(v, memo) for k, v in self.items()}

    def __reduce__(self):
        return (dict, (dict(self),))


@functools.lru_cache(maxsize=None)
def _read_only_isa_info_class():
    from tensilelite.Common.Types import IsaInfo

    class ReadOnlyIsaInfo(IsaInfo):
        """An ``IsaInfo`` whose attributes cannot be rebound."""

        def __setattr__(self, name, value):
            raise TypeError(_READ_ONLY)

        def __delattr__(self, name):
            raise TypeError(_READ_ONLY)

        def __eq__(self, other):
            return isinstance(other, IsaInfo) and all(
                getattr(self, f) == getattr(other, f) for f in _ISA_INFO_FIELDS
            )

        __hash__ = None

        def __reduce__(self):
            # Pickles (for example to a joblib worker) as an ordinary IsaInfo.
            return (IsaInfo, tuple(dict(getattr(self, f)) for f in _ISA_INFO_FIELDS))

        def __deepcopy__(self, memo):
            return IsaInfo(**{f: copy.deepcopy(getattr(self, f), memo) for f in _ISA_INFO_FIELDS})

    return ReadOnlyIsaInfo


def _read_only_isa_info(info):
    """``info`` as an ``IsaInfo`` whose attributes and capability dicts cannot be changed."""
    frozen = object.__new__(_read_only_isa_info_class())
    for f in _ISA_INFO_FIELDS:
        object.__setattr__(frozen, f, _ReadOnlyDict(getattr(info, f)))
    return frozen


def _isa_of(arch):
    from tensilelite.Common.Architectures import gfxToIsa

    if not isinstance(arch, str):
        return arch  # already an IsaVersion
    isa = gfxToIsa(arch)
    if isa is None:
        raise ValueError(f"Unrecognized gfx architecture: {arch!r}")
    return isa


@functools.lru_cache(maxsize=None)
def _shared_isa_info_map(isa):
    from tensilelite.Common.Capabilities import makeIsaInfoMap

    iim = makeIsaInfoMap([isa], cxx_compiler())
    return _ReadOnlyDict({k: _read_only_isa_info(v) for k, v in iim.items()})


def shared_isa_info_map(arch):
    """The process-wide, read-only capability map holding only ``arch`` (a gfx
    name or an IsaVersion). Every caller gets the same object."""
    return _shared_isa_info_map(_isa_of(arch))


def use_isa(arch, wavefront):
    """Initialize cached capabilities and select an ISA and wavefront for this test.

    Selection is not cached: another test may have changed the active kernel.
    The unit fixtures restore the previous kernel state after each test.
    """
    import rocisa

    isa = _isa_of(arch)
    shared_isa_info_map(isa)
    ri = rocisa.rocIsa.getInstance()
    ri.setKernel(tuple(isa), wavefront)
    return ri


def _probe_isa_chunk(arguments):
    """Probe in a fresh process and return rocisa's pickleable capability data."""
    import rocisa

    isas, compiler = arguments
    ri = rocisa.rocIsa.getInstance()
    for isa in isas:
        ri.init(tuple(isa), compiler, False)
    requested = {tuple(isa) for isa in isas}
    return {isa: info for isa, info in ri.getData().items() if tuple(isa) in requested}


def _prime_isa_cache(isas, compiler):
    import rocisa

    ri = rocisa.rocIsa.getInstance()
    # Native rocisa uses a C++ IsaVersion as its key, while the Python callers
    # use IsaVersion namedtuples. Compare plain tuples rather than their hashes.
    cached = {tuple(isa) for isa in ri.getData()}
    missing = [isa for isa in isas if tuple(isa) not in cached]
    if not missing:
        return

    workers = min(8, len(missing), os.cpu_count() or 1)
    if workers == 1 or multiprocessing.current_process().daemon:
        # Daemonic processes cannot create a child pool.
        for isa in missing:
            ri.init(tuple(isa), compiler, False)
        return

    chunks = [missing[i::workers] for i in range(workers)]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_probe_isa_chunk, [(chunk, compiler) for chunk in chunks])

    # setData replaces the cache: retain entries already initialized by callers,
    # including ISAs outside SUPPORTED_ISA. It does not change the active kernel.
    merged = ri.getData()
    for part in parts:
        merged.update(part)
    ri.setData(merged)


@functools.lru_cache(maxsize=1)
def shared_full_isa_info_map():
    """One read-only map for SUPPORTED_ISA per process, primed by up to eight workers.

    Only missing ISAs are probed. xdist workers keep separate process caches;
    the probe pool uses spawn so it does not inherit pytest or rocisa state.
    No test compares a spawned probe with a serial ``rocIsa.init``; the
    characterization goldens that consume this map are what guard that they agree.
    """
    from tensilelite.Common.Architectures import SUPPORTED_ISA

    _prime_isa_cache(SUPPORTED_ISA, cxx_compiler())
    return _ReadOnlyDict({isa: shared_isa_info_map(isa)[isa] for isa in SUPPORTED_ISA})


def shared_isa_info_map_or_skip(arch, toolchainRequired=True):
    """``shared_isa_info_map(arch)``, or a clean skip where the compiler cannot
    target a newer ``arch``.

    gfx90a, gfx942 and gfx950 (``_ALWAYS_SUPPORTED_ARCHS``) fail instead of
    skipping when the compiler cannot target them, so a toolchain that drops a
    stable architecture shows up as failures, not as a quieter run.

    ``toolchainRequired=False`` also skips when ``amdclang++`` is missing, for
    callers that probe for it rather than expect it.
    """
    if not toolchainRequired:
        try:
            cxx_compiler()
        except (ValueError, FileNotFoundError) as e:
            pytest.skip(f"amdclang++ is unavailable: {e}")
    isa = _isa_of(arch)
    iim = shared_isa_info_map(arch)
    if not iim[isa].asmCaps["SupportedISA"]:
        message = f"amdclang++ in this environment does not support {arch}"
        if tuple(isa) in {tuple(_isa_of(name)) for name in _ALWAYS_SUPPORTED_ARCHS}:
            pytest.fail(f"{message}, but every supported toolchain has to", pytrace=False)
        pytest.skip(message)
    return iim


def copy_isa_info_map(isaInfoMap):
    """A mutable deep copy of a capability map, for a test that has to change one."""
    return copy.deepcopy(isaInfoMap)


def build_toolchain_for(arch):
    """``(assembler, isaInfoMap)`` for one ``arch``, the map a mutable copy that
    belongs to the caller."""
    return shared_assembler(), copy_isa_info_map(shared_isa_info_map(arch))


@functools.lru_cache(maxsize=None)
def toolchain_for(arch):
    """``(assembler, isaInfoMap)`` for one ``arch``, both shared and the map read-only.

    The map holds only ``arch`` so that ``_generate_single_solution``'s
    ``next(iter(isaInfoMap.keys()))`` deterministically selects it.
    """
    return shared_assembler(), shared_isa_info_map(arch)

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Gate the arch-domain *lane* -- what the lowerers do with the artifact.

``tests/tools/test_arch_domain_artifact.py`` gates the committed data. This file gates what
the lowerers do with it: every intrinsic demand goes through one chokepoint,
that chokepoint consults the table, it speaks only for a measured, definite
negative, and in ``error`` mode it refuses one unless a waiver covers it -- the
same way in both engines.

The things that can quietly break the lane:

* **A demand that skips the chokepoint.** Writing ``self._needs_intrin[k] =
  True`` directly still emits the declare, so the check never runs for that key.
  Caught by AST walk, in the style of ``library/tests/test_library_layering.py``.
* **A no-data status read as a negative.** ``target_unsupported``,
  ``toolchain_crash``, ``toolchain_timeout`` and a missing column all mean the
  question was never answered; treating them as "no" gives a developer on an
  older ROCm a wall of warnings -- 14% of the llvm20 column is no-data.
* **The implicit target default coming back.** A silent ``gfx950`` makes the
  lookup unanswerable, because the target it is consulted for would be a guess.
* **The engines disagreeing.** A kernel one engine refuses and the other lowers
  is a byte-identity failure whose cause is a data file. The C++ engine reads a
  copy generated from the same columns; the tests that need it built skip
  without it.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from rocke.core.arch import domain as arch_domain
from rocke.core.lower_llvm import (
    ArchDomainError,
    ArchDomainWarning,
    lower_kernel_to_llvm,
)

PLATFORM = Path(__file__).resolve().parents[2]
LOWER_LLVM = PLATFORM / "python" / "rocke" / "core" / "lower_llvm.py"
CPP_CORE = PLATFORM / "cpp" / "core" / "lower_llvm" / "core.cpp"
CHECK_GATE = PLATFORM / "tools" / "check_arch_domain.py"

# Three tests below read the *sources* -- the Python emitter, the C++ core, the
# gate script -- which `install(DIRECTORY tests/)` does not ship. Skip those
# rather than excluding the module: the rest of the file is portable and is
# exactly what an installed-package consumer wants run.
_needs_sources = pytest.mark.skipif(
    not (LOWER_LLVM.exists() and CPP_CORE.exists() and CHECK_GATE.exists()),
    reason="reads the platform sources, which an installed test layout omits",
)


# --------------------------------------------------------------------------
# the loader's contract
# --------------------------------------------------------------------------


def _committed_flavors() -> list[str]:
    from rocke.core.lower_llvm import LLVM_FLAVORS

    return [f for f in LLVM_FLAVORS if arch_domain.load(f) is not None]


def test_at_least_one_column_is_committed() -> None:
    """Without this, every test below would pass vacuously."""
    assert _committed_flavors(), "no committed arch-domain column to test against"


def test_only_arch_absent_is_negative() -> None:
    for status in (
        arch_domain.STATUS_OK,
        arch_domain.STATUS_NAME_ABSENT,
        *sorted(arch_domain.NO_DATA_STATUSES),
    ):
        assert not arch_domain.is_negative(status), status
    assert arch_domain.is_negative(arch_domain.STATUS_ARCH_ABSENT)


def test_no_data_statuses_partition_the_vocabulary() -> None:
    """Answers and non-answers partition the vocabulary with nothing left over,
    so a status added later cannot fall silently into neither bucket."""
    every = {
        v
        for name, v in vars(arch_domain).items()
        if name.startswith("STATUS_") and isinstance(v, str)
    }
    answers = {
        arch_domain.STATUS_OK,
        arch_domain.STATUS_NAME_ABSENT,
        arch_domain.STATUS_ARCH_ABSENT,
    }
    assert answers & arch_domain.NO_DATA_STATUSES == set()
    assert answers | arch_domain.NO_DATA_STATUSES == every


def test_missing_column_is_none_not_an_error() -> None:
    assert arch_domain.load("llvm-that-nobody-ships") is None


def test_an_unreadable_column_raises_instead_of_reading_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only absence is no data.

    ``None`` from ``load`` means the lane has nothing to say, so a column that
    exists but cannot be read (permissions, I/O) must not come back as one: that
    would switch the lane off for the host with nothing said, which is the one
    failure the lane exists to prevent.
    """
    flavor = "llvm-with-an-unreadable-column"
    name = f"intrinsic_arch_domain.{flavor}.json"
    real_read_text = Path.read_text

    def read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == name:
            raise PermissionError(13, "Permission denied", str(self))
        return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", read_text)
    with pytest.raises(PermissionError):
        arch_domain.load(flavor)


def test_unknown_key_and_unknown_arch_are_no_data() -> None:
    table = arch_domain.load(_committed_flavors()[0])
    assert table is not None
    assert table.lookup("not.a.real.intrinsic", "gfx950") is None
    known_key = table.keys()[0]
    assert table.lookup(known_key, "gfx-not-a-target") is None


def test_a_malformed_column_is_rejected_not_emptied() -> None:
    """A wrong document must not become an empty table.

    ``None`` from :func:`lookup` means "not measured", so a column that parses
    as JSON but is not *this* schema would hand back an ``ArchDomain`` whose
    every lookup says no-data -- the lane silently switched off for everyone.
    """
    for doc in (
        {},
        [],
        {"keys": {}},  # no schema at all
        {"schema": "rocke.intrinsic_arch_domain/v2", "keys": {}},
        {"schema": arch_domain.SCHEMA},  # schema but nothing measured
        {"schema": arch_domain.SCHEMA, "keys": []},
    ):
        with pytest.raises(ValueError):
            arch_domain.ArchDomain("llvm22", doc)


def test_every_committed_column_declares_the_schema_it_is() -> None:
    """The check above is only worth having if the real columns pass it."""
    for flavor in _committed_flavors():
        table = arch_domain.load(flavor)
        assert table is not None
        assert table.schema == arch_domain.SCHEMA


def test_load_is_cached() -> None:
    """The lookup sits on the hot path of every intrinsic demand; re-reading and
    re-parsing a ~1000-cell column per demand would be felt."""
    flavor = _committed_flavors()[0]
    assert arch_domain.load(flavor) is arch_domain.load(flavor)


# --------------------------------------------------------------------------
# the chokepoint
# --------------------------------------------------------------------------


def _needs_intrin_writes() -> list[tuple[str, int]]:
    """Every function that assigns into ``self._needs_intrin``, with its line."""
    tree = ast.parse(LOWER_LLVM.read_text(encoding="utf-8"))
    enclosing: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                enclosing.setdefault(id(child), node.name)

    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == "_needs_intrin"
            ):
                out.append((enclosing.get(id(node), "<module>"), node.lineno))
    return out


@_needs_sources
def test_needs_intrin_is_only_written_by_need() -> None:
    """A path that writes the map directly gets its declare emitted and skips
    the arch-domain lookup, leaving no trace in the emitted bytes."""
    offenders = [(fn, line) for fn, line in _needs_intrin_writes() if fn != "_need"]
    assert not offenders, (
        "these write _Lowerer._needs_intrin outside _need() and so bypass the "
        f"arch-domain check: {offenders}"
    )


# --------------------------------------------------------------------------
# no implicit target default, either engine
# --------------------------------------------------------------------------


def _a_kernel():
    """An ordinary gfx950 GEMM the table has nothing negative to say about, so
    these tests isolate the arch argument rather than the warning."""
    from rocke.instances.common.gemm_universal import (
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
        build_universal_gemm,
    )

    return build_universal_gemm(
        UniversalGemmSpec(
            name="rocke_arch_domain_lane",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(
                pipeline="compv4",
                scheduler="intrawave",
                epilogue="cshuffle",
            ),
        )
    )


def test_python_lowering_requires_an_explicit_arch() -> None:
    with pytest.raises(ValueError, match="explicit gfx target"):
        lower_kernel_to_llvm(_a_kernel())


@_needs_sources
def test_cpp_backend_resolution_has_no_default_arch() -> None:
    """The C++ side has no diagnostic channel to assert against from Python
    without a built engine, so assert on the source: a NULL arch must not fall
    through to a gfx target."""
    src = CPP_CORE.read_text(encoding="utf-8")
    assert 'arch == NULL || strcmp(arch, "gfx950")' not in src
    assert "if(arch == NULL)" in src


def test_every_binding_requires_a_target() -> None:
    """Every ``rocke_engine`` entry point that takes a ``(spec, arch)`` raises
    for an empty target, before it looks at the spec.

    Each used to stand for its family's usual arch when ``arch`` was empty, so a
    caller that forgot it built for one target and compiled for another. Two
    bindings kept a hard-coded gfx950 after the rest were fixed, which is what
    this sweep is for: it reads the names off the module, so a binding added
    later is held to the same rule.
    """
    import re

    engine = pytest.importorskip("rocke_engine")
    takes_target = re.compile(r"^\w+\(\w+: dict, arch: str\)")
    offenders = []
    checked = 0
    for name in sorted(dir(engine)):
        fn = getattr(engine, name)
        doc = (getattr(fn, "__doc__", None) or "").splitlines()
        if not doc or not takes_target.match(doc[0]):
            continue
        checked += 1
        try:
            fn({}, arch="")
        except ValueError as exc:
            if "explicit gfx target" not in str(exc):
                offenders.append((name, f"ValueError: {exc}"[:90]))
        except Exception as exc:  # noqa: BLE001
            offenders.append((name, f"{type(exc).__name__}: {exc}"[:90]))
        else:
            offenders.append((name, "returned instead of raising"))
    assert checked > 50, f"only {checked} (spec, arch) bindings found"
    assert not offenders, offenders


@_needs_sources
def test_no_binding_source_defaults_the_target() -> None:
    """The sweep above needs a built ``rocke_engine``, and CI builds none, so a
    binding that guesses its target gets past every run without one. Each of the
    two ways it happens shows in the source: the helper that stood for "usual
    arch" when ``arch`` was empty, and a gfx target as the default of an
    ``arch`` argument. Code written against develop keeps reaching for the first,
    so this is checked from a source checkout, where a run without the extension
    can still see it.
    """
    import re

    sources = sorted((PLATFORM / "cpp" / "bindings").glob("*.[ch]pp"))
    assert sources, "no binding sources found"
    offenders = [
        f"{path.name}:{number}: {line.strip()[:90]}"
        for path in sources
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "arch_or_default" in line or re.search(r'py::arg\("arch"\)\s*=', line)
    ]
    assert not offenders, "\n".join(offenders)


def test_explicit_arch_still_lowers() -> None:
    ir = lower_kernel_to_llvm(_a_kernel(), arch="gfx950")
    assert "target triple" in ir


# --------------------------------------------------------------------------
# the warning itself
# --------------------------------------------------------------------------


def _a_negative_cell() -> tuple[str, str, str]:
    """A measured (key, arch, flavor) the table calls arch_absent."""
    for flavor in _committed_flavors():
        table = arch_domain.load(flavor)
        assert table is not None
        # ArchDomain is not a mapping; .keys() is its accessor.
        for key in table.keys():  # noqa: SIM118
            for arch in ("gfx942", "gfx90a", "gfx1151", "gfx11-generic", "gfx950"):
                cell = table.lookup(key, arch)
                if cell is not None and arch_domain.is_negative(cell.status):
                    return key, arch, flavor
    pytest.skip("no arch_absent cell in any committed column")


def test_warning_carries_structured_fields() -> None:
    """The gate keys on (key, arch, flavor) rather than parsing prose, so these
    attributes are load-bearing, not decoration."""
    key, arch, flavor = _a_negative_cell()
    w = ArchDomainWarning(key, arch, flavor, "arch_absent", "some diagnostic")
    assert (w.key, w.arch, w.flavor, w.status) == (key, arch, flavor, "arch_absent")
    assert key in str(w) and arch in str(w) and "some diagnostic" in str(w)


def test_lane_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(arch_domain.MODE_ENV, "off")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lower_kernel_to_llvm(_a_kernel(), arch="gfx950")
    assert not [w for w in caught if isinstance(w.message, ArchDomainWarning)]


def test_every_mode_is_case_and_space_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for mode in arch_domain.MODES:
        for raw in (mode, mode.upper(), f" {mode.capitalize()}\t"):
            monkeypatch.setenv(arch_domain.MODE_ENV, raw)
            assert arch_domain.mode() == mode, raw


def test_an_unset_or_empty_lane_setting_is_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(arch_domain.MODE_ENV, raising=False)
    assert arch_domain.mode() == arch_domain.DEFAULT_MODE
    for raw in ("", "  "):
        monkeypatch.setenv(arch_domain.MODE_ENV, raw)
        assert arch_domain.mode() == arch_domain.DEFAULT_MODE


def test_a_misspelled_lane_setting_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falling back to the default here does the opposite of what was asked.

    Every plausible misspelling -- ``0``, ``false``, ``no``, a typo -- is
    somebody trying to turn the lane *off*. Warning anyway leaves the lane on
    while the developer believes this escape hatch opened it.
    """
    for raw in ("0", "1", "false", "no", "offf", "on", "warning", "errors", "fail"):
        monkeypatch.setenv(arch_domain.MODE_ENV, raw)
        with pytest.raises(ValueError, match=arch_domain.MODE_ENV):
            arch_domain.mode()


def test_a_misspelled_lane_setting_raises_for_a_kernel_that_demands_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mode was read where a demand is vetted, so a kernel that demands no
    intrinsic never read it.

    The same bad value then raised for one kernel and not another, and on the
    backend that vets demands and not on the one that checks emitted text. It
    is validated where every lowering enters, before dispatch.
    """
    from rocke.core.ir import IRBuilder

    builder = IRBuilder("demands_no_intrinsic")
    builder.kernel.attrs["max_workgroup_size"] = 64
    builder.const_i32(1)
    monkeypatch.setenv(arch_domain.MODE_ENV, "offf")
    for backend in ("python", "cpp"):
        monkeypatch.setenv("ROCKE_BACKEND", backend)
        with pytest.raises(ValueError, match=arch_domain.MODE_ENV):
            lower_kernel_to_llvm(builder.kernel, arch="gfx950", llvm_flavor="llvm22")


@pytest.mark.parametrize("backend", ["python", "cpp", "both"])
def test_a_misspelled_lane_setting_raises_through_the_dispatcher(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """``compile_kernel`` and the per-family wrappers reach the dispatcher
    without passing through ``lower_kernel_to_llvm``, so the check lives in the
    dispatcher itself, before any lowering or backend choice."""
    from rocke.core import backend as B

    monkeypatch.setenv(arch_domain.MODE_ENV, "offf")
    monkeypatch.setenv("ROCKE_BACKEND", backend)
    with pytest.raises(ValueError, match=arch_domain.MODE_ENV):
        B.lower_kernel_via_backend(
            _a_kernel(),
            llvm_flavor="llvm22",
            arch="gfx950",
            python_lower=lambda *a, **k: pytest.fail(
                "the dispatcher must reject the mode before lowering"
            ),
        )


# --------------------------------------------------------------------------
# the lane reaches the default backend
# --------------------------------------------------------------------------


def _negative_declare() -> tuple[str, str, str, str]:
    """A (declare line, key, arch, flavor) the table calls arch_absent.

    Built from the flavor's own declaration table so the line is byte-for-byte
    what ``finalize`` emits: the post-pass maps declares back to keys by exact
    match, so a hand-written approximation would match nothing in the real run.
    """
    from rocke.core.lower_llvm import _decls_for_flavor

    key, arch, flavor = _a_negative_cell()
    return _decls_for_flavor(flavor)[key], key, arch, flavor


def test_the_post_pass_warns_on_a_negative_declare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The positive control for the whole post-pass."""
    from rocke.core.lower_llvm import check_arch_domain_in_ir

    decl, key, arch, flavor = _negative_declare()
    monkeypatch.setenv(arch_domain.MODE_ENV, "warn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_arch_domain_in_ir(f"{decl}\n", arch, flavor)
    got = [w.message for w in caught if isinstance(w.message, ArchDomainWarning)]
    assert [(w.key, w.arch) for w in got] == [(key, arch)]


def test_the_post_pass_is_silent_on_an_unmeasured_declare() -> None:
    """A declare the table never measured is not evidence of anything.

    ``llvm.smax.v<N>i<W>`` is registered dynamically at lowering time and so
    appears in no column; silence has to hold here as everywhere else in the
    lane, or the default backend becomes the noisy one.
    """
    from rocke.core.lower_llvm import check_arch_domain_in_ir

    _decl, _key, arch, flavor = _negative_declare()
    ir = "declare <4 x i32> @llvm.smax.v4i32(<4 x i32>, <4 x i32>)\n"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_arch_domain_in_ir(ir, arch, flavor)
    assert not [w for w in caught if isinstance(w.message, ArchDomainWarning)]


def _aliased_negative_declare() -> tuple[str, str, tuple[str, ...], str]:
    """A (declare line, arch, keys, flavor) where several keys share one declare
    and every one of them is a measured negative on that arch.

    Found from the committed tables rather than named, so the test keeps
    meaning something when the data moves. At llvm23 the pair is
    ``mfma.scale.f32.16x16x128.f8f6f4`` / ``mfma.f32.16x16x128.fp8.hero``.
    """
    from rocke.core.lower_llvm import _decls_for_flavor

    for flavor in _committed_flavors():
        table = arch_domain.load(flavor)
        if table is None:
            continue
        by_decl: dict[str, list[str]] = {}
        for key, decl in _decls_for_flavor(flavor).items():
            by_decl.setdefault(decl, []).append(key)
        for decl, keys in sorted(by_decl.items()):
            if len(keys) < 2:
                continue
            for arch in sorted(set(table.arches(keys[0]))):
                cells = [table.lookup(k, arch) for k in keys]
                if all(
                    c is not None and arch_domain.is_negative(c.status) for c in cells
                ):
                    return decl, arch, tuple(sorted(keys)), flavor
    pytest.skip("no committed column has an all-negative aliased declare")


def test_the_post_pass_reports_every_key_an_aliased_declare_could_be(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identical declare text is the same LLVM intrinsic under two table keys.

    The post-pass reads keys back out of emitted IR, so for such a pair it
    cannot know which one the kernel demanded -- and naming just one (whichever
    the reverse map happened to keep) would point at the wrong table row half
    the time, disagreeing with the key the Python lowerer warns under. Report
    every alias that is itself a measured negative.
    """
    from rocke.core.lower_llvm import check_arch_domain_in_ir

    decl, arch, keys, flavor = _aliased_negative_declare()
    monkeypatch.setenv(arch_domain.MODE_ENV, "warn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_arch_domain_in_ir(f"{decl}\n", arch, flavor)
    got = [w.message for w in caught if isinstance(w.message, ArchDomainWarning)]
    assert tuple(sorted(w.key for w in got)) == keys
    assert {w.arch for w in got} == {arch}


def test_a_waived_alias_does_not_turn_its_sibling_into_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine lowered the kernel, so whichever alias it demanded was
    acceptable to it.

    With one alias waived, the unwaived sibling may be a key the kernel never
    asked for, and refusing over it would overrule a waiver on a guess. Both are
    still reported, each against its own cell.
    """
    from rocke.core.lower_llvm import check_arch_domain_in_ir

    decl, arch, keys, flavor = _aliased_negative_declare()
    _waive(monkeypatch, (keys[0], arch, flavor))
    monkeypatch.setenv(arch_domain.MODE_ENV, "error")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_arch_domain_in_ir(f"{decl}\n", arch, flavor)
    got = {
        w.message.key: w.message.waived
        for w in caught
        if isinstance(w.message, ArchDomainWarning)
    }
    assert got == {keys[0]: True, **{k: False for k in keys[1:]}}


def test_an_aliased_declare_no_alias_accounts_for_is_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every alias sharing the declare is an unwaived negative, nothing
    could explain the engine having lowered it: the extension was built from a
    different table than this checkout's, and the post-pass says so."""
    from rocke.core.lower_llvm import check_arch_domain_in_ir

    decl, arch, keys, flavor = _aliased_negative_declare()
    _waive(monkeypatch)  # nothing waived, whatever is on file
    monkeypatch.setenv(arch_domain.MODE_ENV, "error")
    with pytest.raises(ArchDomainError) as excinfo:
        check_arch_domain_in_ir(f"{decl}\n", arch, flavor)
    assert excinfo.value.key in keys


BINDINGS = PLATFORM / "cpp" / "bindings"


@_needs_sources
@pytest.mark.skipif(not BINDINGS.is_dir(), reason="installed layout omits cpp/")
def test_no_binding_lowers_without_a_diagnostic_buffer() -> None:
    """``rocke_lower_kernel_to_llvm`` has no ``err`` out-parameter.

    A refusal's status alone says only *that* the target cannot lower the
    kernel; the intrinsic, target and flavor live in the message, which is
    also all ``backend.py`` has to rebuild the Python error from. Lowering
    through the err-less entry reached Python as an ArchDomainError with an
    empty message. The ``take_*`` helpers now take ``err`` by reference so a
    null cannot be passed, and this keeps the err-less entry itself out of the
    bindings, where the type system has nothing to say.
    """
    offenders = [
        f"{path.name}:{n}"
        for path in sorted(BINDINGS.glob("*.cpp"))
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if "rocke_lower_kernel_to_llvm(" in line
    ]
    assert not offenders, "lower through rocke_lower_kernel_to_llvm_ex instead"


@_needs_sources
@pytest.mark.skipif(not BINDINGS.is_dir(), reason="installed layout omits cpp/")
def test_every_binding_lowering_converts_a_refusal() -> None:
    """A refusal reaches Python as ``ArchDomainError`` only if the binding that
    lowered the kernel hands its status and message to ``throw_if_arch_domain``,
    directly or through ``take_lowered`` / ``take_ll``. A binding that formats
    its own failure raises a plain ``RuntimeError`` instead, and the two engines
    then disagree about what a refusal is.

    Two bindings that lowered through a family convenience wrapper, rather than
    calling the lowerer themselves, did exactly that, and the err-less-entry scan
    above could not see them. This one looks at every call whose name ends in
    ``to_llvm`` and requires a conversion in the same function.
    """
    call = re.compile(r"rocke_\w*to_llvm\w*\(")
    handled = re.compile(r"take_lowered\(|take_ll\(|throw_if_arch_domain\(")
    offenders = []
    for path in sorted(BINDINGS.glob("*.cpp")):
        text = path.read_text()
        for m in call.finditer(text):
            end = text.find("\n}\n", m.end())
            body = text[m.end() : end if end != -1 else len(text)]
            if not handled.search(body):
                line = text.count("\n", 0, m.start()) + 1
                offenders.append(f"{path.name}:{line}")
    assert not offenders, f"no refusal conversion after the lowering at {offenders}"


def test_the_post_pass_honours_the_off_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unlike ``test_lane_can_be_switched_off``, this input provably warns when
    the lane is on, so the assertion is about the switch and not about a kernel
    that had nothing to say."""
    from rocke.core.lower_llvm import check_arch_domain_in_ir

    decl, _key, arch, flavor = _negative_declare()
    monkeypatch.setenv(arch_domain.MODE_ENV, "off")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_arch_domain_in_ir(f"{decl}\n", arch, flavor)
    assert not [w for w in caught if isinstance(w.message, ArchDomainWarning)]


def test_the_cpp_backend_runs_the_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default backend warns, which is the whole point of the post-pass.

    The C++ engine is stubbed rather than required: without the extension
    built, the real call falls back to the Python lowerer, whose ``_need``
    runs the lane and would make the test pass while proving nothing.
    """
    from rocke.core import backend as B

    decl, key, arch, flavor = _negative_declare()
    monkeypatch.setenv(arch_domain.MODE_ENV, "warn")
    monkeypatch.setenv("ROCKE_BACKEND", "cpp")
    monkeypatch.setattr(B, "_lower_via_cpp_engine", lambda *a, **k: f"{decl}\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = B.lower_kernel_via_backend(
            _a_kernel(),
            llvm_flavor=flavor,
            arch=arch,
            python_lower=lambda *a, **k: pytest.fail("python lowerer must not run"),
        )
    assert out == f"{decl}\n"
    got = [w.message for w in caught if isinstance(w.message, ArchDomainWarning)]
    assert [w.key for w in got] == [key]


@pytest.mark.parametrize("backend", ["cpp", "both"])
def test_one_lowering_resolves_the_flavor_once(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """Compiler discovery is retryable, so asking twice can answer differently.

    The engine lowered at one flavor and the arch-domain check, asking again,
    read another flavor's column: warnings vanished or named the wrong row.
    Under ``both`` the two engines could be compared at different flavors and the
    gap reported as a parity failure. A lowering resolves the flavor once and
    every stage gets that value.
    """
    from rocke.core import backend as B
    from rocke.core import lower_llvm

    answers = iter(["llvm22", "llvm23"])  # a second ask would not agree with the first
    seen: dict[str, str | None] = {}

    def engine(ir_text: str, arch: str, flavor: str | None = None) -> str:
        seen["engine"] = flavor
        return "; ir\n"

    def post_pass(ir_text: str, arch: str, flavor: str | None) -> None:
        seen["post_pass"] = flavor

    def python_lower(
        kernel: object, *, llvm_flavor: str | None = None, arch: str | None = None
    ) -> str:
        seen["python"] = llvm_flavor
        return "; ir\n"

    monkeypatch.setenv("ROCKE_BACKEND", backend)
    monkeypatch.setattr(lower_llvm, "_resolve_llvm_flavor", lambda: next(answers))
    monkeypatch.setattr(B, "_lower_via_cpp_engine", engine)
    monkeypatch.setattr(lower_llvm, "check_arch_domain_in_ir", post_pass)
    B.lower_kernel_via_backend(
        _a_kernel(), llvm_flavor=None, arch="gfx950", python_lower=python_lower
    )
    assert set(seen.values()) == {"llvm22"}, seen
    assert len(seen) == 2, seen


def test_the_cpp_fallback_does_not_warn_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    """When cpp fails, the Python lowerer runs and its ``_need`` is the lane.

    Adding the post-pass to that path too would double every warning on the
    one path that already had them.
    """
    from rocke.core import backend as B

    decl, key, arch, flavor = _negative_declare()
    monkeypatch.setenv("ROCKE_BACKEND", "cpp")
    monkeypatch.setenv("ROCKE_CPP_QUIET_FALLBACK", "1")

    def _boom(*a, **k):
        raise B.BackendError("stubbed engine failure")

    monkeypatch.setattr(B, "_lower_via_cpp_engine", _boom)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        B.lower_kernel_via_backend(
            _a_kernel(),
            llvm_flavor=flavor,
            arch=arch,
            # Stands in for the Python lowerer, which would have warned once
            # from `_need`, without depending on a kernel that demands this key.
            python_lower=lambda *a, **k: (
                warnings.warn(
                    ArchDomainWarning(key, arch, flavor, "arch_absent", ""),
                    stacklevel=2,
                ),
                f"{decl}\n",
            )[1],
        )
    got = [w.message for w in caught if isinstance(w.message, ArchDomainWarning)]
    assert len(got) == 1, "the fallback path warned twice"


# --------------------------------------------------------------------------
# the corpus is clean
# --------------------------------------------------------------------------


@_needs_sources
def test_the_corpus_gate_is_green() -> None:
    """Delegates to the gate rather than re-deriving its rules, so the tool and
    the test cannot disagree about what the lane is allowed to say."""
    tool = PLATFORM / "tools" / "check_arch_domain.py"
    proc = subprocess.run([sys.executable, str(tool)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# --------------------------------------------------------------------------
# error mode, the message, and the waivers
# --------------------------------------------------------------------------

_DPP8_KEY = "mov.dpp8.i32"
_DPP8_ARCH = "gfx950"


def _dpp8_kernel():
    """One ``mov.dpp8.i32`` demand: an RDNA intrinsic the committed columns
    measure ``arch_absent`` on CDNA, so gfx950 is a real negative to refuse."""
    from rocke.core.ir import IRBuilder

    b = IRBuilder("rocke_arch_domain_dpp8")
    b.kernel.attrs["max_workgroup_size"] = 64
    b.mov_dpp8(b.const_i32(1), 0x765432)
    return b.kernel


def _refusing_flavor() -> str:
    """A committed flavor whose column calls the dpp8 demand negative."""
    for flavor in _committed_flavors():
        cell = arch_domain.load(flavor).lookup(_DPP8_KEY, _DPP8_ARCH)
        if cell is not None and arch_domain.is_negative(cell.status):
            return flavor
    pytest.skip(f"no committed column calls {_DPP8_KEY} negative on {_DPP8_ARCH}")


def _lower_dpp8(flavor: str) -> str:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    return _lower_kernel_to_llvm_python(
        _dpp8_kernel(), arch=_DPP8_ARCH, llvm_flavor=flavor
    )


def _waive(monkeypatch: pytest.MonkeyPatch, *cells: tuple[str, str, str]) -> None:
    table = {c: arch_domain.Waiver(*c, "test: the measurement is wrong") for c in cells}
    monkeypatch.setattr(arch_domain, "waivers", lambda: table)


def test_error_mode_refuses_an_unwaived_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flavor = _refusing_flavor()
    cell = arch_domain.load(flavor).lookup(_DPP8_KEY, _DPP8_ARCH)
    monkeypatch.setenv(arch_domain.MODE_ENV, "error")
    with pytest.raises(ArchDomainError) as info:
        _lower_dpp8(flavor)
    e = info.value
    assert (e.key, e.arch, e.flavor, e.status, e.evidence) == (
        _DPP8_KEY,
        _DPP8_ARCH,
        flavor,
        cell.status,
        cell.evidence,
    )
    assert str(e) == arch_domain.message(
        _DPP8_KEY, _DPP8_ARCH, flavor, cell.status, cell.evidence
    )
    # The convention for a kernel the lowerer declines for its target, which
    # is how hsaco_parity counts a refusal.
    assert isinstance(e, NotImplementedError)


def test_warn_mode_reports_what_error_mode_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flavor = _refusing_flavor()
    monkeypatch.setenv(arch_domain.MODE_ENV, "warn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _lower_dpp8(flavor)
    got = [w.message for w in caught if isinstance(w.message, ArchDomainWarning)]
    assert [(w.key, w.arch, w.flavor, w.waived) for w in got] == [
        (_DPP8_KEY, _DPP8_ARCH, flavor, False)
    ]


def test_off_mode_lowers_without_looking(monkeypatch: pytest.MonkeyPatch) -> None:
    flavor = _refusing_flavor()
    monkeypatch.setenv(arch_domain.MODE_ENV, "off")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert "@llvm.amdgcn.mov.dpp8.i32" in _lower_dpp8(flavor)
    assert not [w for w in caught if isinstance(w.message, ArchDomainWarning)]


def test_a_waiver_turns_the_refusal_into_a_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The escape hatch for a wrong row: the kernel lowers, and still says so."""
    flavor = _refusing_flavor()
    _waive(monkeypatch, (_DPP8_KEY, _DPP8_ARCH, flavor))
    monkeypatch.setenv(arch_domain.MODE_ENV, "error")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert "@llvm.amdgcn.mov.dpp8.i32" in _lower_dpp8(flavor)
    got = [w.message for w in caught if isinstance(w.message, ArchDomainWarning)]
    assert [(w.key, w.waived) for w in got] == [(_DPP8_KEY, True)]


def test_a_waiver_covers_one_cell_not_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Narrow by construction: the same key on another target or at another
    flavor is a different cell, and leaves this one refused."""
    flavor = _refusing_flavor()
    _waive(
        monkeypatch,
        (_DPP8_KEY, "gfx942", flavor),
        (_DPP8_KEY, _DPP8_ARCH, "llvm-another"),
    )
    monkeypatch.setenv(arch_domain.MODE_ENV, "error")
    with pytest.raises(ArchDomainError):
        _lower_dpp8(flavor)


def test_a_refusal_survives_pickle_and_copy() -> None:
    """A refusal raised in a worker process must reach its parent as itself.

    The constructor takes the measured cell but the base class keeps only the
    message, so the default pickle rebuilt it as ``ArchDomainError(message)`` and
    failed with TypeError: a pool worker's refusal surfaced as a broken pool
    instead of a verdict.
    """
    import copy
    import pickle

    from rocke.core.lower_llvm import ArchDomainError

    err = ArchDomainError(
        "some.key", "gfx942", "llvm22", "arch_absent", "Cannot select"
    )
    for clone in (
        pickle.loads(pickle.dumps(err)),
        copy.copy(err),
        copy.deepcopy(err),
    ):
        assert isinstance(clone, ArchDomainError)
        assert (clone.key, clone.arch, clone.flavor, clone.status, clone.evidence) == (
            err.key,
            err.arch,
            err.flavor,
            err.status,
            err.evidence,
        )
        assert str(clone) == str(err)


def test_every_negative_cell_is_ascii() -> None:
    """The C++ engine cuts a message at a byte count and Python at a character
    count, and the generator orders keys by ``strcmp``. Both assume ASCII, so a
    non-ASCII diagnostic entering a column should fail here, not surface as two
    engines printing the same refusal differently."""
    for flavor in _committed_flavors():
        table = arch_domain.load(flavor)
        assert table is not None
        for key in table.keys():  # noqa: SIM118
            for arch in table.arches(key):
                cell = table.lookup(key, arch)
                if cell is None or not arch_domain.is_negative(cell.status):
                    continue
                for field in (key, arch, cell.evidence):
                    assert field.isascii(), (flavor, key, arch, field)


def test_an_unreadable_waiver_file_raises_instead_of_reading_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only absence is "no waivers", the same rule the columns follow.

    A waiver file that exists but cannot be read is a broken checkout. Reading it
    as empty would quietly start refusing cells the repository waived, and bake
    no waivers into the C++ table.
    """
    real_read_text = Path.read_text

    def read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == "arch_domain_waivers.json":
            raise PermissionError(13, "Permission denied", str(self))
        return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    arch_domain.waivers.cache_clear()
    monkeypatch.setattr(Path, "read_text", read_text)
    try:
        with pytest.raises(PermissionError):
            arch_domain.waivers()
    finally:
        monkeypatch.undo()
        arch_domain.waivers.cache_clear()


def test_the_message_fits_the_cpp_error_buffer_and_reads_back() -> None:
    msg = arch_domain.message(
        "k" * 67, "gfx11-generic", "llvm22", "arch_absent", "x" * 400
    )
    assert len(msg) == arch_domain.MESSAGE_CAP
    assert arch_domain.parse_message(msg) == (
        "k" * 67,
        "gfx11-generic",
        "llvm22",
        "arch_absent",
    )
    assert arch_domain.parse_message("intrinsic 'x' is not available") is None


_WAIVER = {"key": "k", "arch": "gfx942", "flavor": "llvm22", "reason": "r"}


@pytest.mark.parametrize(
    "doc",
    [
        {},
        [],
        {"schema": "rocke.arch_domain_waivers/v0", "waivers": []},
        {"schema": arch_domain.WAIVERS_SCHEMA},
        {"schema": arch_domain.WAIVERS_SCHEMA, "waivers": {}},
        {"schema": arch_domain.WAIVERS_SCHEMA, "waivers": [["k", "gfx942"]]},
        # A misspelled field would otherwise waive nothing, silently.
        {
            "schema": arch_domain.WAIVERS_SCHEMA,
            "waivers": [{**_WAIVER, "flavour": "llvm22"}],
        },
        {
            "schema": arch_domain.WAIVERS_SCHEMA,
            "waivers": [{k: v for k, v in _WAIVER.items() if k != "reason"}],
        },
        {"schema": arch_domain.WAIVERS_SCHEMA, "waivers": [{**_WAIVER, "reason": " "}]},
        {"schema": arch_domain.WAIVERS_SCHEMA, "waivers": [{**_WAIVER, "arch": 942}]},
        {"schema": arch_domain.WAIVERS_SCHEMA, "waivers": [_WAIVER, dict(_WAIVER)]},
    ],
)
def test_a_malformed_waiver_file_is_refused(doc: object) -> None:
    with pytest.raises(ValueError):
        arch_domain.parse_waivers(doc)


def test_a_well_formed_waiver_parses() -> None:
    doc = {"schema": arch_domain.WAIVERS_SCHEMA, "waivers": [_WAIVER]}
    assert arch_domain.parse_waivers(doc) == {
        ("k", "gfx942", "llvm22"): arch_domain.Waiver(**_WAIVER)
    }


def test_every_committed_waiver_names_a_negative_cell() -> None:
    """A waiver for a cell that is not negative waives nothing. The gate fails
    on one too; this keeps the failure next to the file."""
    for (key, arch, flavor), waiver in arch_domain.waivers().items():
        table = arch_domain.load(flavor)
        assert table is not None, waiver
        cell = table.lookup(key, arch)
        assert cell is not None and arch_domain.is_negative(cell.status), waiver


def test_the_post_pass_refuses_in_error_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """A negative declare in C++ output means the engine did not refuse what
    this checkout's table refuses; the kernel is still not buildable."""
    from rocke.core.lower_llvm import check_arch_domain_in_ir

    decl, key, arch, flavor = _negative_declare()
    monkeypatch.setenv(arch_domain.MODE_ENV, "error")
    with pytest.raises(ArchDomainError) as info:
        check_arch_domain_in_ir(f"{decl}\n", arch, flavor)
    assert (info.value.key, info.value.arch) == (key, arch)


def test_the_post_pass_reports_a_waived_cell_in_error_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The C++ engine has no warning channel, so for a waived cell this pass is
    the only thing that says anything on the default backend."""
    from rocke.core.lower_llvm import check_arch_domain_in_ir

    decl, key, arch, flavor = _negative_declare()
    _waive(monkeypatch, (key, arch, flavor))
    monkeypatch.setenv(arch_domain.MODE_ENV, "error")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_arch_domain_in_ir(f"{decl}\n", arch, flavor)
    got = [w.message for w in caught if isinstance(w.message, ArchDomainWarning)]
    assert [(w.key, w.waived) for w in got] == [(key, True)]


# --------------------------------------------------------------------------
# the backend: a refusal is a verdict, not an engine failure
# --------------------------------------------------------------------------


def _refusal(key: str | None = None) -> ArchDomainError:
    _decl, neg_key, arch, flavor = _negative_declare()
    return ArchDomainError(key or neg_key, arch, flavor, "arch_absent", "")


def _raiser(exc: BaseException):
    def _raise(*a, **k):
        raise exc

    return _raise


def test_a_cpp_refusal_is_raised_not_fallen_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falling back would run the Python lowerer, which reads the same table and
    refuses too -- after printing that the C++ engine 'failed'."""
    from rocke.core import backend as B

    _decl, _key, arch, flavor = _negative_declare()
    refusal = _refusal()
    monkeypatch.setenv("ROCKE_BACKEND", "cpp")
    monkeypatch.delenv("ROCKE_CPP_STRICT", raising=False)
    monkeypatch.setattr(B, "_lower_via_cpp_engine", _raiser(refusal))
    B.reset_cpp_fallbacks()
    with pytest.raises(ArchDomainError) as info:
        B.lower_kernel_via_backend(
            _a_kernel(),
            llvm_flavor=flavor,
            arch=arch,
            python_lower=lambda *a, **k: pytest.fail("python lowerer must not run"),
        )
    assert info.value is refusal
    assert B.cpp_fallbacks() == []


def test_an_engine_refusal_becomes_the_python_error() -> None:
    from rocke.core import backend as B

    _decl, key, arch, flavor = _negative_declare()
    cell = arch_domain.load(flavor).lookup(key, arch)
    msg = arch_domain.message(key, arch, flavor, cell.status, cell.evidence)
    err = B._arch_domain_refusal(msg)
    assert isinstance(err, ArchDomainError)
    assert (err.key, err.arch, err.flavor, err.evidence) == (
        key,
        arch,
        flavor,
        cell.evidence,
    )
    assert str(err) == msg


def test_a_refusal_this_checkout_would_not_make_is_a_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An extension built from other columns or waivers: say the engines
    disagree, rather than raise a verdict this checkout does not hold."""
    from rocke.core import backend as B

    _decl, key, arch, flavor = _negative_declare()
    cell = arch_domain.load(flavor).lookup(key, arch)
    msg = arch_domain.message(key, arch, flavor, cell.status, cell.evidence)
    _waive(monkeypatch, (key, arch, flavor))
    assert isinstance(B._arch_domain_refusal(msg), B.BackendMismatch)
    assert isinstance(B._arch_domain_refusal("not a refusal"), B.BackendMismatch)


def _both(monkeypatch: pytest.MonkeyPatch, python_lower, cpp) -> str:
    from rocke.core import backend as B

    _decl, _key, arch, flavor = _negative_declare()
    monkeypatch.setenv("ROCKE_BACKEND", "both")
    monkeypatch.setattr(B, "_lower_via_cpp_engine", cpp)
    return B.lower_kernel_via_backend(
        _a_kernel(), llvm_flavor=flavor, arch=arch, python_lower=python_lower
    )


@_needs_sources
def test_the_gate_counts_a_refusal_this_checkout_does_not_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An engine built from other columns than this checkout refuses what Python
    would not, and the backend reports that as a ``BackendMismatch`` rather than
    an ``ArchDomainError``. The gate filed it with the cases it could not judge,
    a printed count, and went green over the one skew it exists to catch."""
    monkeypatch.syspath_prepend(str(PLATFORM / "tools"))
    import check_arch_domain as gate
    from rocke.core import backend as B

    def stale_engine(*args: object, **kwargs: object) -> str:
        raise B.BackendMismatch(
            "the C++ engine refused a lowering that this checkout's "
            "arch-domain table does not refuse"
        )

    monkeypatch.setattr(B, "_lower_via_cpp_engine", stale_engine)
    case = {"case_id": "a_case", "arch": "gfx950", "build": _a_kernel}
    disagreements, unjudged = gate._cpp_verdicts([case], "llvm22", {})
    assert len(disagreements) == 1, disagreements
    assert not unjudged, unjudged


def test_both_mode_raises_the_refusal_both_engines_make(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    py = _refusal()
    with pytest.raises(ArchDomainError) as info:
        _both(monkeypatch, _raiser(py), _raiser(_refusal()))
    assert info.value is py


@pytest.mark.parametrize("refuses", ["python", "cpp"])
def test_both_mode_flags_a_refusal_one_engine_makes(
    monkeypatch: pytest.MonkeyPatch, refuses: str
) -> None:
    from rocke.core import backend as B

    def lowered(*a, **k) -> str:
        return "ir"

    py = _raiser(_refusal()) if refuses == "python" else lowered
    cpp = _raiser(_refusal()) if refuses == "cpp" else lowered
    with pytest.raises(B.BackendMismatch, match="refused"):
        _both(monkeypatch, py, cpp)


def test_both_mode_flags_refusals_over_different_intrinsics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rocke.core import backend as B

    with pytest.raises(B.BackendMismatch, match="different intrinsics"):
        _both(monkeypatch, _raiser(_refusal()), _raiser(_refusal("other.key")))


# --------------------------------------------------------------------------
# the C++ engine itself (needs the rocke_engine extension)
# --------------------------------------------------------------------------


def test_the_message_text_is_the_one_the_cpp_engine_pins() -> None:
    """``test_arch_domain_refusal.cpp`` pins these two sentences for the C++
    engine. This test needs no extension, so CI compares the engines' bytes
    through the two literals."""
    assert arch_domain.message(
        "mov.dpp8.i32", "gfx950", "llvm22", "arch_absent", "no such intrinsic on gfx950"
    ) == (
        "arch-domain: intrinsic 'mov.dpp8.i32' is not available on gfx950 at llvm22 "
        "(arch_absent): no such intrinsic on gfx950"
    )
    assert arch_domain.message(
        "mov.dpp8.i32", "gfx950", "llvm22", "arch_absent", ""
    ) == (
        "arch-domain: intrinsic 'mov.dpp8.i32' is not available on gfx950 at llvm22 "
        "(arch_absent)"
    )


@pytest.mark.parametrize("flavor", _committed_flavors())
def test_the_cpp_engine_refuses_with_the_python_message(
    monkeypatch: pytest.MonkeyPatch, flavor: str
) -> None:
    engine = pytest.importorskip("rocke_engine")
    from rocke.core.ir_serialize import serialize

    cell = arch_domain.load(flavor).lookup(_DPP8_KEY, _DPP8_ARCH)
    if cell is None or not arch_domain.is_negative(cell.status):
        pytest.skip(f"{_DPP8_KEY} is not negative on {_DPP8_ARCH} at {flavor}")
    monkeypatch.setenv(arch_domain.MODE_ENV, "error")
    with pytest.raises(engine.ArchDomainError) as cpp:
        engine.lower_serialized_ir(
            serialize(_dpp8_kernel()), arch=_DPP8_ARCH, flavor=flavor
        )
    with pytest.raises(ArchDomainError) as py:
        _lower_dpp8(flavor)
    assert str(cpp.value) == str(py.value)
    assert isinstance(cpp.value, NotImplementedError)


@pytest.mark.parametrize("mode", ["warn", "off"])
def test_the_cpp_engine_lowers_outside_error_mode(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    engine = pytest.importorskip("rocke_engine")
    from rocke.core.ir_serialize import serialize

    flavor = _refusing_flavor()
    monkeypatch.setenv(arch_domain.MODE_ENV, mode)
    ll = engine.lower_serialized_ir(
        serialize(_dpp8_kernel()), arch=_DPP8_ARCH, flavor=flavor
    )
    assert "@llvm.amdgcn.mov.dpp8.i32" in ll


def test_the_cpp_engine_rejects_a_misspelled_mode_as_python_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = pytest.importorskip("rocke_engine")
    from rocke.core.ir_serialize import serialize

    flavor = _refusing_flavor()
    monkeypatch.setenv(arch_domain.MODE_ENV, " Warning ")
    with pytest.raises(ValueError) as py:
        arch_domain.mode()
    with pytest.raises(RuntimeError) as cpp:
        engine.lower_serialized_ir(
            serialize(_dpp8_kernel()), arch=_DPP8_ARCH, flavor=flavor
        )
    assert str(py.value) in str(cpp.value)


def test_the_cpp_engine_rejects_a_misspelled_mode_for_a_kernel_that_demands_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The C++ engine reads the mode once, at entry, as the Python entry does.

    It used to read it only when a kernel demanded an intrinsic, so a kernel that
    demanded none lowered under a misspelled value because nothing asked.
    """
    engine = pytest.importorskip("rocke_engine")
    from rocke.core.ir import IRBuilder
    from rocke.core.ir_serialize import serialize

    builder = IRBuilder("demands_no_intrinsic")
    builder.kernel.attrs["max_workgroup_size"] = 64
    builder.const_i32(1)
    monkeypatch.setenv(arch_domain.MODE_ENV, " Warning ")
    with pytest.raises(RuntimeError, match=arch_domain.MODE_ENV):
        engine.lower_serialized_ir(
            serialize(builder.kernel), arch="gfx950", flavor="llvm22"
        )


def test_the_cpp_backend_raises_the_python_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end on the default backend: the engine refuses, the binding
    raises its ArchDomainError, and the caller gets the Python one."""
    pytest.importorskip("rocke_engine")
    flavor = _refusing_flavor()
    monkeypatch.setenv("ROCKE_BACKEND", "cpp")
    monkeypatch.setenv(arch_domain.MODE_ENV, "error")
    with pytest.raises(ArchDomainError) as info:
        lower_kernel_to_llvm(_dpp8_kernel(), arch=_DPP8_ARCH, llvm_flavor=flavor)
    assert (info.value.key, info.value.arch, info.value.flavor) == (
        _DPP8_KEY,
        _DPP8_ARCH,
        flavor,
    )

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Gate the arch-domain *warning lane* — the first consumer of the artifact.

``test_arch_domain_artifact.py`` gates the committed data. This file gates what
the lowerer does with it: every intrinsic demand goes through one chokepoint,
that chokepoint consults the table, and it speaks only for a measured, definite
negative.

The three things that can quietly break the lane:

* **A demand that skips the chokepoint.** Writing ``self._needs_intrin[k] =
  True`` directly still emits the declare, so the check never runs for that key.
  Caught by AST walk, in the style of ``library/tests/test_library_layering.py``.
* **A no-data status read as a negative.** ``target_unsupported``,
  ``toolchain_crash``, ``toolchain_timeout`` and a missing column all mean the
  question was never answered; treating them as "no" gives a developer on an
  older ROCm a wall of warnings -- 14% of the llvm20 column is no-data.
* **The implicit target default coming back.** A silent ``gfx950`` makes the
  lookup unanswerable, because the target it is consulted for would be a guess.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from rocke.core.arch import domain as arch_domain
from rocke.core.lower_llvm import ArchDomainWarning, lower_kernel_to_llvm

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
    from rocke.core.lower_llvm import _ARCH_DOMAIN_ENV

    monkeypatch.setenv(_ARCH_DOMAIN_ENV, "off")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lower_kernel_to_llvm(_a_kernel(), arch="gfx950")
    assert not [w for w in caught if isinstance(w.message, ArchDomainWarning)]


def test_the_off_switch_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    from rocke.core.lower_llvm import _ARCH_DOMAIN_ENV, _arch_domain_mode

    for raw in ("off", "OFF", " Off "):
        monkeypatch.setenv(_ARCH_DOMAIN_ENV, raw)
        assert _arch_domain_mode() == "off", raw


def test_an_unset_or_empty_lane_setting_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    from rocke.core.lower_llvm import _ARCH_DOMAIN_ENV, _arch_domain_mode

    monkeypatch.delenv(_ARCH_DOMAIN_ENV, raising=False)
    assert _arch_domain_mode() == "warn"
    monkeypatch.setenv(_ARCH_DOMAIN_ENV, "")
    assert _arch_domain_mode() == "warn"


def test_a_misspelled_lane_setting_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falling back to the default here does the opposite of what was asked.

    Every plausible misspelling -- ``0``, ``false``, ``no``, a typo -- is
    somebody trying to turn the lane *off*. Warning anyway leaves the lane on
    while the developer believes this escape hatch opened it.
    """
    from rocke.core.lower_llvm import _ARCH_DOMAIN_ENV, _arch_domain_mode

    # `error` is a planned third mode, not a current one: the day it is added,
    # this line is the reminder that `_ARCH_DOMAIN_MODES` must learn about it.
    for raw in ("0", "1", "false", "no", "offf", "on", "warning", "error"):
        monkeypatch.setenv(_ARCH_DOMAIN_ENV, raw)
        with pytest.raises(ValueError, match=_ARCH_DOMAIN_ENV):
            _arch_domain_mode()


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
    from rocke.core.lower_llvm import _ARCH_DOMAIN_ENV

    builder = IRBuilder("demands_no_intrinsic")
    builder.kernel.attrs["max_workgroup_size"] = 64
    builder.const_i32(1)
    monkeypatch.setenv(_ARCH_DOMAIN_ENV, "offf")
    for backend in ("python", "cpp"):
        monkeypatch.setenv("ROCKE_BACKEND", backend)
        with pytest.raises(ValueError, match=_ARCH_DOMAIN_ENV):
            lower_kernel_to_llvm(builder.kernel, arch="gfx950", llvm_flavor="llvm22")


@pytest.mark.parametrize("backend", ["python", "cpp", "both"])
def test_a_misspelled_lane_setting_raises_through_the_dispatcher(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """``compile_kernel`` and the per-family wrappers reach the dispatcher
    without passing through ``lower_kernel_to_llvm``, so the check lives in the
    dispatcher itself, before any lowering or backend choice."""
    from rocke.core import backend as B
    from rocke.core.lower_llvm import _ARCH_DOMAIN_ENV

    monkeypatch.setenv(_ARCH_DOMAIN_ENV, "offf")
    monkeypatch.setenv("ROCKE_BACKEND", backend)
    with pytest.raises(ValueError, match=_ARCH_DOMAIN_ENV):
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


def test_the_post_pass_warns_on_a_negative_declare() -> None:
    """The positive control for the whole post-pass."""
    from rocke.core.lower_llvm import check_arch_domain_in_ir

    decl, key, arch, flavor = _negative_declare()
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


def test_the_post_pass_honours_the_off_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unlike ``test_lane_can_be_switched_off``, this input provably warns when
    the lane is on, so the assertion is about the switch and not about a kernel
    that had nothing to say."""
    from rocke.core.lower_llvm import _ARCH_DOMAIN_ENV, check_arch_domain_in_ir

    decl, _key, arch, flavor = _negative_declare()
    monkeypatch.setenv(_ARCH_DOMAIN_ENV, "off")
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
# the lane matches its allowlist
# --------------------------------------------------------------------------


@_needs_sources
def test_corpus_matches_the_expected_warning_set() -> None:
    """Delegates to the gate rather than re-deriving the set, so the tool and
    the test cannot disagree about what the lane is allowed to say."""
    tool = PLATFORM / "tools" / "check_arch_domain.py"
    proc = subprocess.run([sys.executable, str(tool)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch):
    """The gate script as a module: it is a tool, not part of the package."""
    import importlib.util

    monkeypatch.syspath_prepend(str(CHECK_GATE.parent))
    spec = importlib.util.spec_from_file_location(
        "check_arch_domain_under_test", CHECK_GATE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TRIPLE = ("ds.read.tr16.b64", "gfx942", "llvm22")


def _expect(gate, *cases: str) -> dict:
    return {_TRIPLE: gate.Expected("a reason", frozenset(cases))}


@_needs_sources
def test_every_allowlisted_entry_names_its_cases(gate) -> None:
    """An entry with no cases would allow any case to demand its intrinsic."""
    for triple, entry in gate.EXPECTED_WARNINGS.items():
        assert entry.reason and entry.cases, triple


@_needs_sources
def test_the_expected_demand_is_clean(gate) -> None:
    unexpected, stale = gate.judge(
        {_TRIPLE: {"a", "b"}}, _expect(gate, "a", "b"), {"llvm22": {"a", "b"}},
        {"a", "b"}, ["llvm22"],
    )  # fmt: skip
    assert (unexpected, stale) == ({}, [])


@_needs_sources
def test_another_case_demanding_an_allowlisted_intrinsic_is_unexpected(gate) -> None:
    """The entry is keyed by triple *and* cases, so a new case cannot ride on
    an old entry and leave the gate green."""
    unexpected, stale = gate.judge(
        {_TRIPLE: {"a", "b", "c"}}, _expect(gate, "a", "b"),
        {"llvm22": {"a", "b", "c"}}, {"a", "b", "c"}, ["llvm22"],
    )  # fmt: skip
    assert unexpected == {_TRIPLE: {"c"}}
    assert stale == []


@_needs_sources
def test_fixing_one_of_two_cases_makes_the_entry_stale(gate) -> None:
    unexpected, stale = gate.judge(
        {_TRIPLE: {"a"}}, _expect(gate, "a", "b"), {"llvm22": {"a", "b"}},
        {"a", "b"}, ["llvm22"],
    )  # fmt: skip
    assert unexpected == {}
    assert [(triple, case) for triple, case, _ in stale] == [(_TRIPLE, "b")]


@_needs_sources
def test_a_case_that_did_not_run_is_not_judged(gate) -> None:
    """A filtered selection, or an instance that declines the flavor, says
    nothing about an allowlisted case: it is neither confirmed nor stale."""
    unexpected, stale = gate.judge(
        {_TRIPLE: {"a"}}, _expect(gate, "a", "b"), {"llvm22": {"a"}},
        {"a", "b"}, ["llvm22"],
    )  # fmt: skip
    assert (unexpected, stale) == ({}, [])


@_needs_sources
def test_an_entry_naming_a_removed_case_is_stale(gate) -> None:
    unexpected, stale = gate.judge(
        {_TRIPLE: {"a"}}, _expect(gate, "a", "gone"), {"llvm22": {"a"}},
        {"a"}, ["llvm22"],
    )  # fmt: skip
    assert unexpected == {}
    assert [(case, why) for _, case, why in stale] == [
        ("gone", "the corpus has no such case")
    ]


@_needs_sources
def test_a_triple_the_allowlist_lacks_is_unexpected_whole(gate) -> None:
    other = ("mfma.f32.16x16x32.f16", "gfx942", "llvm22")
    unexpected, stale = gate.judge(
        {other: {"a"}}, {}, {"llvm22": {"a"}}, {"a"}, ["llvm22"]
    )
    assert unexpected == {other: {"a"}}
    assert stale == []


@_needs_sources
def test_an_unmeasured_flavor_is_not_judged(gate) -> None:
    unexpected, stale = gate.judge(
        {}, _expect(gate, "a"), {"llvm20": {"a"}}, {"a"}, ["llvm20"]
    )
    assert (unexpected, stale) == ({}, [])

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The C++ engine's arch-domain table must say exactly what the Python reader says.

The build generates the C++ copy from the committed columns
(``tools/gen_arch_domain_cpp.py``). These tests hold that copy to
``rocke.core.arch.domain`` cell for cell, at three points: the generator's
output text, whether that text compiles into the lookup, and -- when the
``rocke_engine`` extension is built -- the table the engine actually carries.
A copy that drifted would make the two engines disagree about which kernels a
target can build, which is the failure one table exists to rule out.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from rocke.core.arch import domain as arch_domain

PLATFORM = Path(__file__).resolve().parents[2]
GENERATOR = PLATFORM / "tools" / "gen_arch_domain_cpp.py"
DOMAIN_CPP = PLATFORM / "cpp" / "core" / "arch" / "domain.cpp"
CPP_INCLUDE = PLATFORM / "cpp" / "include"

# The installed test layout ships neither tools/ nor cpp/.
pytestmark = pytest.mark.skipif(
    not (GENERATOR.exists() and DOMAIN_CPP.exists()),
    reason="reads the platform sources, which an installed test layout omits",
)


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location(
        "rocke_gen_arch_domain_cpp", GENERATOR
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _python_view() -> dict[str, list[tuple[str, str, str, str, bool]]]:
    """Every negative cell per committed flavor, as the Python lane sees them,
    with whether a committed waiver covers it."""
    view = {}
    for flavor in arch_domain.committed_flavors():
        table = arch_domain.load(flavor)
        assert table is not None
        cells = []
        for key in table.keys():  # noqa: SIM118 -- ArchDomain is not a mapping
            for arch in table.arches(key):
                cell = table.lookup(key, arch)
                if cell is not None and arch_domain.is_negative(cell.status):
                    waived = arch_domain.is_waived(key, arch, flavor)
                    cells.append((key, arch, cell.status, cell.evidence, waived))
        view[flavor] = sorted(cells)
    return view


def _c_unescape(literal: str) -> str:
    """Decode the body of a C string literal written by ``c_string``: the
    generator emits only ``\\"``, ``\\\\`` and three-digit octal bytes."""
    out = bytearray()
    i = 0
    while i < len(literal):
        if literal[i] != "\\":
            out += literal[i].encode("ascii")
            i += 1
        elif literal[i + 1] in '"\\':
            out += literal[i + 1].encode("ascii")
            i += 2
        else:
            out.append(int(literal[i + 1 : i + 4], 8))
            i += 4
    return out.decode("utf-8")


_STR = r'"((?:[^"\\]|\\.)*)"'
_CELL = re.compile(
    rf"^    \{{{_STR}, {_STR}, {_STR}, {_STR}, ([01])\}},$", re.MULTILINE
)
_ARRAY = re.compile(
    r"^const rocke_arch_domain_cell_t (\w+)\[\] = \{$(.*?)^\};$",
    re.MULTILINE | re.DOTALL,
)
_COLUMN = re.compile(rf"^    \{{{_STR}, (\w+), (\d+)\}},$", re.MULTILINE)


def _parse(text: str) -> dict[str, list[tuple[str, str, str, str, bool]]]:
    """The generated text read back into cells, keyed by flavor."""
    arrays = {
        name: [
            (*(_c_unescape(f) for f in m.groups()[:4]), m.group(5) == "1")
            for m in _CELL.finditer(body)
        ]
        for name, body in _ARRAY.findall(text)
    }
    out = {}
    for flavor, name, count in _COLUMN.findall(text):
        cells = arrays.get(name, []) if name != "nullptr" else []
        assert len(cells) == int(count), (flavor, name, count)
        out[_c_unescape(flavor)] = cells
    return out


def test_at_least_one_column_is_committed() -> None:
    """Without this, every comparison below would pass vacuously."""
    assert arch_domain.committed_flavors()


def test_the_generated_table_is_the_python_readers_negative_cells(gen) -> None:
    assert _parse(gen.render(gen.load_domain())) == _python_view()


def test_generation_is_deterministic(gen) -> None:
    """CMake reruns the generator on every input change; output that varied run
    to run would rebuild the engine for nothing and defeat any later diff."""
    domain = gen.load_domain()
    assert gen.render(domain) == gen.render(domain)


def test_the_generator_reads_through_the_python_reader(gen) -> None:
    """The policy is decided once, in Python: a cell is emitted exactly when
    ``is_negative`` says so, whatever its status string is."""
    table = SimpleNamespace(
        keys=lambda: ("k.one", "k.two"),
        arches=lambda key: ("gfx942",),
        lookup=lambda key, arch: SimpleNamespace(status=key, evidence="e"),
    )
    fake = SimpleNamespace(
        committed_flavors=lambda: ("llvm99",),
        load=lambda flavor: table,
        is_negative=lambda status: status == "k.two",
        is_waived=lambda key, arch, flavor: False,
        DEFAULT_MODE="warn",
    )
    assert _parse(gen.render(fake)) == {
        "llvm99": [("k.two", "gfx942", "k.two", "e", False)]
    }


def test_an_empty_column_is_a_null_entry_not_an_empty_array(gen) -> None:
    """A zero-length array does not compile, and a flavor whose column has no
    negative cell is normal -- every probe could come back ``ok``."""
    fake = SimpleNamespace(
        committed_flavors=lambda: ("llvm99",),
        load=lambda flavor: SimpleNamespace(keys=lambda: (), arches=lambda key: ()),
        is_negative=lambda status: True,
        is_waived=lambda key, arch, flavor: False,
        DEFAULT_MODE="warn",
    )
    text = gen.render(fake)
    assert '{"llvm99", nullptr, 0},' in text
    assert "const rocke_arch_domain_cell_t" not in text


def test_a_non_ascii_key_is_refused(gen) -> None:
    """The C++ lookup binary-searches with strcmp, which only agrees with
    Python's sort order on ASCII; a table sorted otherwise would miss cells."""
    table = SimpleNamespace(
        keys=lambda: ("kéy",),
        arches=lambda key: ("gfx942",),
        lookup=lambda key, arch: SimpleNamespace(status="arch_absent", evidence=""),
    )
    fake = SimpleNamespace(
        committed_flavors=lambda: ("llvm99",),
        load=lambda flavor: table,
        is_negative=lambda status: True,
        is_waived=lambda key, arch, flavor: False,
        DEFAULT_MODE="warn",
    )
    with pytest.raises(ValueError, match="non-ASCII"):
        gen.render(fake)


def test_a_waiver_reaches_the_cpp_table(gen) -> None:
    """The C++ engine decides waived-or-refused from this flag alone, so it has
    to be ``domain.is_waived`` for that exact cell, not for the key."""
    table = SimpleNamespace(
        keys=lambda: ("k",),
        arches=lambda key: ("gfx942", "gfx950"),
        lookup=lambda key, arch: SimpleNamespace(status="arch_absent", evidence=""),
    )
    fake = SimpleNamespace(
        committed_flavors=lambda: ("llvm99",),
        load=lambda flavor: table,
        is_negative=lambda status: True,
        is_waived=lambda key, arch, flavor: (key, arch, flavor)
        == ("k", "gfx950", "llvm99"),
        DEFAULT_MODE="warn",
    )
    assert _parse(gen.render(fake)) == {
        "llvm99": [
            ("k", "gfx942", "arch_absent", "", False),
            ("k", "gfx950", "arch_absent", "", True),
        ]
    }


def test_the_default_mode_is_generated_from_python(gen) -> None:
    """Changing ``domain.DEFAULT_MODE`` must move both engines, with no second
    place to edit."""
    for mode in ("warn", "error"):
        fake = SimpleNamespace(
            committed_flavors=lambda: ("llvm99",),
            load=lambda flavor: SimpleNamespace(keys=lambda: (), arches=lambda key: ()),
            is_negative=lambda status: True,
            is_waived=lambda key, arch, flavor: False,
            DEFAULT_MODE=mode,
        )
        assert f'ROCKE_ARCH_DOMAIN_DEFAULT_MODE = "{mode}";' in gen.render(fake)


@pytest.mark.parametrize(
    ("text", "literal"),
    [
        ("plain", '"plain"'),
        ('say "x"', r'"say \"x\""'),
        ("back\\slash", r'"back\\slash"'),
        # A hex escape would swallow the trailing digit; octal stops at three.
        ("é1", r'"\303\2511"'),
        ("tab\there", r'"tab\011here"'),
    ],
)
def test_c_string_escaping(gen, text: str, literal: str) -> None:
    assert gen.c_string(text) == literal
    assert _c_unescape(literal[1:-1]) == text


def test_the_generated_table_compiles_into_the_lookup(gen, tmp_path: Path) -> None:
    """Syntax-checks the real output against the real ``domain.cpp`` with the
    engine's warning flags, so a generator change that breaks the build fails
    here rather than in a CMake run."""
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler on PATH")
    inc = tmp_path / "rocke" / "arch_domain_table.inc"
    gen.main(["--out", str(inc)])
    proc = subprocess.run(
        [
            cxx,
            "-std=c++20",
            "-fsyntax-only",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{CPP_INCLUDE}",
            f"-I{tmp_path}",
            str(DOMAIN_CPP),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_the_engine_carries_the_python_readers_table() -> None:
    """The copy compiled into ``rocke_engine`` -- the one that will refuse a
    kernel -- not just the generator's output."""
    engine = pytest.importorskip("rocke_engine")
    view = _python_view()
    for flavor, cells in view.items():
        assert [tuple(c) for c in engine.arch_domain_column(flavor)] == cells, flavor
    assert engine.arch_domain_column("llvm-that-nobody-ships") == []
    assert engine.arch_domain_default_mode() == arch_domain.DEFAULT_MODE

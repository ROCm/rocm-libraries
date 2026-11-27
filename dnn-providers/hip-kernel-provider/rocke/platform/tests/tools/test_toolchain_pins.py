# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No new toolchain pin in rocKE's test and library code.

Tests use the compiler rocKE's discovery loads
(``dsl_docs/reference/env_flags.md``, "Compiler detection and provenance"). The
pins that remain are listed, one line per file with the kinds it has and the
issue that removes them, in ``toolchain_pins_allowlist.txt`` beside this file.
The gate reads every test and library Python file under ``rocke/`` with
:mod:`rocke.tests.toolchain_pins` and fails when a file not on the list has a
pin, when a listed file gains a kind, and when a listed file or kind no longer
has one (delete or shorten its line: the list only shrinks).

The finder's own cases below plant each kind of pin and the code that is not
one. The gate reads the source checkout through git, so it runs in the build
tree only (``rocke_toolchain_pins_check``), never in the installed test tree.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from rocke.assets import platform_root
from rocke.tests.source_allowlist import (
    AllowlistEntry,
    file_kinds,
    format_allowlist_entry,
)
from rocke.tests.toolchain_pins import (
    ToolchainPin,
    ToolchainPinKind,
    allowlist_breaches,
    checked_kinds,
    read_allowlist,
    toolchain_pins_in_python,
    tree_toolchain_pins,
)

ROCKE = platform_root().parent
ALLOWLIST = Path(__file__).with_name("toolchain_pins_allowlist.txt")
GUIDANCE = (
    "Tests use the compiler rocKE's discovery loads: no LLVM-flavor or ROCm-release "
    "pins, no decision on the comgr's ROCm release, no private runtime or lowering "
    'names in library code (dsl_docs/reference/env_flags.md, "Compiler detection and '
    'provenance"). A per-compiler exception is a record in rocke.tests.pending_lanes '
    "with a ticket and an expiry. The allowlist only shrinks: remove the lines of "
    "pins you removed, never add one."
)

FLAVOR_PIN = ToolchainPinKind.FLAVOR_PIN
ROCM_PIN = ToolchainPinKind.ROCM_PIN
ROCM_VERSION = ToolchainPinKind.ROCM_VERSION
PRIVATE_IMPORT = ToolchainPinKind.PRIVATE_IMPORT


def test_the_tree_adds_no_toolchain_pin():
    breaches = allowlist_breaches(tree_toolchain_pins(ROCKE), read_allowlist(ALLOWLIST))
    assert not breaches, GUIDANCE + "\n" + "\n".join(breaches)


def test_the_allowlist_is_sorted_one_canonical_line_per_file():
    listed = [
        line
        for line in ALLOWLIST.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    canonical = [
        format_allowlist_entry(path, entry.counts, entry.issue)
        for path, entry in sorted(read_allowlist(ALLOWLIST).items())
    ]
    assert listed == canonical


@pytest.mark.parametrize(
    "source, kinds",
    [
        ('GATE_FLAVOR = "llvm22"\n', {FLAVOR_PIN}),
        ("class Gate:\n    FLAVORS: tuple = ('llvm23',)\n", {FLAVOR_PIN}),
        ('if current_flavor() == "llvm22":\n    pass\n', {FLAVOR_PIN}),
        (
            (
                "from rocke.core import lower_llvm\n"
                "skip = lower_llvm.resolve_flavor() in ('llvm20', 'llvm22')\n"
            ),
            {FLAVOR_PIN},
        ),
        (
            "import os\nok = os.environ.get('ROCKE_LLVM_FLAVOR') != 'llvm22'\n",
            {FLAVOR_PIN},
        ),
        ("import os\nok = os.environ['ROCKE_LLVM_FLAVOR'] == 'llvm22'\n", {FLAVOR_PIN}),
        ("FIRST, SECOND = 'llvm22', 1\n", {FLAVOR_PIN}),
        (
            (
                "def test_it(monkeypatch):\n"
                "    monkeypatch.setenv('ROCKE_LLVM_FLAVOR', 'llvm22')\n"
            ),
            {FLAVOR_PIN},
        ),
        (
            "import os\ndef test_it():\n    os.environ['ROCKE_LLVM_FLAVOR'] = 'llvm23'\n",
            {FLAVOR_PIN},
        ),
        (
            (
                "import os\nfrom unittest import mock\n"
                "patched = mock.patch.dict(os.environ, {'ROCKE_LLVM_FLAVOR': 'llvm23'})\n"
            ),
            {FLAVOR_PIN},
        ),
        (
            "import os\nenv = dict(os.environ, ROCKE_LLVM_FLAVOR='llvm22')\n",
            {FLAVOR_PIN},
        ),
        (
            "import os\nflavor = os.environ.get('ROCKE_LLVM_FLAVOR', 'llvm20')\n",
            {FLAVOR_PIN},
        ),
        (
            (
                "def test_it(monkeypatch):\n"
                "    monkeypatch.setenv('ROCKE_COMGR_LIB', '/opt/rocm-7.1.0/lib/x.so')\n"
            ),
            {ROCM_PIN},
        ),
        ('GATE_COMGR_ROCM = "7.2"\n', {ROCM_PIN}),
        ("REFERENCE_ROCM = (7, 2)\n", {ROCM_PIN}),
        ('COMGR = "/opt/rocm-7.1.0/lib/libamd_comgr.so"\n', {ROCM_PIN}),
        ("def check(rocm_release):\n    return rocm_release < (7, 2)\n", {ROCM_PIN}),
        ('ok = toolchain["torch"] == "2.9.0+rocm7.1"\n', {ROCM_PIN}),
        (
            (
                "from rocke.runtime.comgr import resolved_lib_rocm_version\n"
                "release = resolved_lib_rocm_version()\n"
            ),
            {ROCM_VERSION},
        ),
        (
            (
                "from rocke.runtime.comgr import resolved_lib_rocm_version as rv\n"
                "release = rv()\n"
            ),
            {ROCM_VERSION},
        ),
        (
            (
                "def gap():\n"
                "    from rocke.core.lower_llvm import _detect_llvm_flavor\n"
                "    return _detect_llvm_flavor()\n"
            ),
            {PRIVATE_IMPORT},
        ),
        (
            "from rocke.runtime import comgr as _comgr\nlib = _comgr._resolve_lib()\n",
            {PRIVATE_IMPORT},
        ),
        (
            "import rocke.runtime.comgr\nlib = rocke.runtime.comgr._lib\n",
            {PRIVATE_IMPORT},
        ),
        ("from rocke import runtime\nlib = runtime.comgr._lib\n", {PRIVATE_IMPORT}),
        ("from rocke.runtime import _coexistence\n", {PRIVATE_IMPORT}),
    ],
    ids=[
        "flavor-constant",
        "flavor-class-attribute",
        "flavor-comparison",
        "flavor-membership",
        "flavor-env-get-compared",
        "flavor-env-index-compared",
        "flavor-tuple-assigned",
        "flavor-monkeypatch-setenv",
        "flavor-env-assigned",
        "flavor-mock-patch-dict",
        "flavor-env-dict-copy",
        "flavor-env-get-default",
        "rocm-comgr-lib-setenv",
        "rocm-release-constant",
        "rocm-release-tuple",
        "rocm-install-path",
        "rocm-release-compared",
        "rocm-torch-build-tag",
        "rocm-version-call",
        "rocm-version-call-alias",
        "private-import-in-function",
        "private-attribute-of-alias",
        "private-attribute-dotted",
        "private-attribute-of-package",
        "private-module-import",
    ],
)
def test_each_kind_of_pin_is_found(source, kinds):
    assert file_kinds(toolchain_pins_in_python(source, "<synthetic>")) == kinds


@pytest.mark.parametrize(
    "source",
    [
        # One of every flavor in turn, and a copy of the list: no single pin.
        "for flavor in LLVM_FLAVORS:\n    if flavor == 'llvm20':\n        pass\n",
        "FLAVORS = ('llvm20', 'llvm22', 'llvm23')\n",
        # Emission at a flavor the test chooses is not what the compiler reports.
        "ir = lower(kernel, llvm_flavor='llvm22')\n",
        "assert resolve_flavor('llvm20') == 'llvm20'\n",
        "def build():\n    flavor = 'llvm22'\n    return flavor\n",
        '"""Calibrated on llvm22 with the ROCm 7.2 comgr."""\n',
        "# GATE_FLAVOR = 'llvm22'\nx = 1\n",
        "VERSION = (7, 2)\n",
        # A flavor chosen for offline emission, and a mapping that is no
        # environment (lower-case keys).
        "spec = {'llvm_flavor': 'llvm22'}\n",
        "def test_it(monkeypatch):\n    monkeypatch.delenv('ROCKE_LLVM_FLAVOR')\n",
        "from rocke.runtime.comgr import loaded_compiler_info\n",
        "from rocke.core.lower_llvm import LLVM_FLAVORS, lower_kernel_to_llvm\n",
        "from rocke.core.llvm_profile import measured_flavor_for_llvm\n",
        "from rocke.runtime import comgr\nname = comgr.__name__\n",
        "from kernels.common import cache as _cache\nx = _cache._key\n",
    ],
)
def test_code_that_does_not_pin_a_toolchain_is_not_a_pin(source):
    assert toolchain_pins_in_python(source, "<synthetic>") == []


def test_a_pin_names_its_line_and_code():
    source = 'import os\n\nGATE_FLAVOR = "llvm22"\nGATE_COMGR_ROCM = "7.2"\n'
    assert toolchain_pins_in_python(source, "<synthetic>") == [
        ToolchainPin(3, FLAVOR_PIN, "GATE_FLAVOR = 'llvm22'"),
        ToolchainPin(4, ROCM_PIN, "GATE_COMGR_ROCM = '7.2'"),
    ]


def test_source_that_does_not_parse_is_an_error_not_a_clean_file():
    with pytest.raises(SyntaxError):
        toolchain_pins_in_python('GATE = "llvm22"\ndef broken(:\n', "<synthetic>")


@pytest.mark.parametrize(
    "path, kinds",
    [
        ("library/kernels/common/cache.py", set(ToolchainPinKind)),
        ("library/tests/test_example.py", set(ToolchainPinKind)),
        ("platform/tests/core/test_example.py", {FLAVOR_PIN, ROCM_PIN, ROCM_VERSION}),
        ("platform/python/rocke/tests/comgr.py", {FLAVOR_PIN, ROCM_PIN, ROCM_VERSION}),
        ("platform/tests/conftest.py", {FLAVOR_PIN, ROCM_PIN, ROCM_VERSION}),
        ("platform/tests/test_compiler_detection.py", set()),
        ("platform/python/rocke/runtime/comgr.py", set()),
        ("platform/tools/check_byte_identity.py", set()),
    ],
)
def test_library_code_is_held_to_every_kind_and_platform_tests_to_the_pins(path, kinds):
    """The platform's tests may test its private helpers, and its detector unit
    tests check what the detector reports, so they are not read."""
    assert checked_kinds(path) == kinds


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        [shutil.which("git") or "git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """A git work tree with a test file holding a pin, listed in the allowlist
    below, a clean library file, and a pin outside test and library code."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "library" / "tests").mkdir(parents=True)
    (tmp_path / "library" / "tests" / "listed.py").write_text(
        'GATE_FLAVOR = "llvm22"\n', encoding="utf-8"
    )
    (tmp_path / "library" / "clean.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "platform" / "tools").mkdir(parents=True)
    (tmp_path / "platform" / "tools" / "not_checked.py").write_text(
        'FLAVOR = "llvm22"\n', encoding="utf-8"
    )
    _git(tmp_path, "add", ".")
    return tmp_path


LISTED = {"library/tests/listed.py": AllowlistEntry({FLAVOR_PIN: 1}, "AICK-1")}


@pytest.mark.parametrize(
    "path, source, breach",
    [
        (
            "library/tests/test_new.py",
            'GATE_FLAVOR = "llvm22"\n',
            (
                "library/tests/test_new.py: new toolchain pin: line 1 flavor_pin: "
                "GATE_FLAVOR = 'llvm22'"
            ),
        ),
        (
            "platform/tests/test_new.py",
            'if current_flavor() != "llvm22":\n    pass\n',
            (
                "platform/tests/test_new.py: new toolchain pin: line 1 flavor_pin: "
                "current_flavor() != 'llvm22'"
            ),
        ),
        (
            "library/tests/test_new.py",
            'GATE_COMGR_ROCM = "7.2"\n',
            (
                "library/tests/test_new.py: new toolchain pin: line 1 rocm_pin: "
                "GATE_COMGR_ROCM = '7.2'"
            ),
        ),
        (
            "library/kernels/new.py",
            (
                "from rocke.runtime import comgr\n"
                "release = comgr.resolved_lib_rocm_version()\n"
            ),
            (
                "library/kernels/new.py: new toolchain pin: line 2 rocm_version: "
                "comgr.resolved_lib_rocm_version()"
            ),
        ),
        (
            "platform/tests/test_new.py",
            (
                "from rocke.runtime.comgr import resolved_lib_rocm_version\n"
                "def test_it():\n"
                "    release = resolved_lib_rocm_version()\n"
            ),
            (
                "platform/tests/test_new.py: new toolchain pin: line 3 rocm_version: "
                "resolved_lib_rocm_version()"
            ),
        ),
        (
            "library/kernels/new.py",
            "from rocke.core.lower_llvm import _detect_llvm_flavor\n",
            (
                "library/kernels/new.py: new toolchain pin: line 1 private_import: "
                "from rocke.core.lower_llvm import _detect_llvm_flavor"
            ),
        ),
    ],
    ids=[
        "flavor-pin-in-library-test",
        "flavor-pin-in-platform-test",
        "rocm-pin-in-library-test",
        "rocm-version-in-library-code",
        "rocm-version-in-platform-test",
        "private-import-in-library-code",
    ],
)
def test_a_planted_pin_of_each_kind_fails_the_gate(repository, path, source, breach):
    planted = repository / path
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_text(source, encoding="utf-8")
    assert allowlist_breaches(tree_toolchain_pins(repository), LISTED) == [breach]


def test_a_planted_new_kind_in_a_listed_file_fails_the_gate(repository):
    (repository / "library" / "tests" / "listed.py").write_text(
        'GATE_FLAVOR = "llvm22"\nGATE_COMGR_ROCM = "7.2"\n', encoding="utf-8"
    )
    assert allowlist_breaches(tree_toolchain_pins(repository), LISTED) == [
        (
            "library/tests/listed.py: new kind of toolchain pin: line 2 rocm_pin: "
            "GATE_COMGR_ROCM = '7.2'"
        )
    ]


def test_a_planted_second_pin_of_a_listed_kind_fails_the_gate(repository):
    (repository / "library" / "tests" / "listed.py").write_text(
        'GATE_FLAVOR = "llvm22"\nOTHER_FLAVOR = "llvm23"\n', encoding="utf-8"
    )
    breaches = allowlist_breaches(tree_toolchain_pins(repository), LISTED)
    assert breaches == [
        (
            "library/tests/listed.py: more toolchain pin than listed: "
            "flavor_pin 1 -> 2: line 1 flavor_pin: GATE_FLAVOR = 'llvm22'; "
            "line 2 flavor_pin: OTHER_FLAVOR = 'llvm23'"
        )
    ]


def test_a_listed_file_whose_pin_was_removed_fails_the_gate(repository):
    (repository / "library" / "tests" / "listed.py").write_text(
        "x = 1\n", encoding="utf-8"
    )
    assert allowlist_breaches(tree_toolchain_pins(repository), LISTED) == [
        "library/tests/listed.py: stale entry, no toolchain pin left: delete its line"
    ]


def test_the_tree_passes_with_its_listed_pin_and_reads_test_and_library_code_only(
    repository,
):
    assert allowlist_breaches(tree_toolchain_pins(repository), LISTED) == []

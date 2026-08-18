# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import importlib
import sys
from pathlib import Path
from zipfile import Path as ZipPath, ZipFile

import pytest
import yaml

from Tensile import resources
from Tensile.TensileCreateLibrary import copyStaticFiles


pytestmark = pytest.mark.unit

EXPECTED_STATIC_HEADERS = (
    "TensileTypes.h",
    "tensile_bfloat16.h",
    "tensile_float8_bfloat8.h",
    "KernelHeader.h",
    "ReductionTemplate.h",
    "memory_gfx.h",
)

KNOWN_CUSTOM_KERNEL = (
    "Custom_Cijk_Ailk_Bljk_F8NH_HHS_BH_Bias_AS_SAB_SAV_shortname0_gfx942"
)


def _fake_resource_tree(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "pkg"
    (root / "Source").mkdir(parents=True)
    (root / "CustomKernels").mkdir()
    (root / "TensileLogic").mkdir()
    monkeypatch.setattr(resources, "_root", lambda: root)
    return root


def _write_static_headers(root: Path) -> None:
    for name in EXPECTED_STATIC_HEADERS:
        (root / "Source" / name).write_text(f"contents for {name}\n", encoding="utf-8")


def test_static_header_paths_have_expected_order():
    assert tuple(
        path.name for path in resources.static_header_paths()
    ) == EXPECTED_STATIC_HEADERS


def test_copy_static_files_copies_expected_files_from_resource_root(tmp_path, monkeypatch):
    root = _fake_resource_tree(tmp_path, monkeypatch)
    _write_static_headers(root)
    output = tmp_path / "out"
    output.mkdir()

    copied = copyStaticFiles(output)

    assert copied == list(EXPECTED_STATIC_HEADERS)
    assert {path.name for path in output.iterdir()} == set(copied)
    for name in EXPECTED_STATIC_HEADERS:
        assert (output / name).read_text(encoding="utf-8") == f"contents for {name}\n"


def test_copy_static_files_creates_nonexistent_directory(tmp_path, monkeypatch):
    root = _fake_resource_tree(tmp_path, monkeypatch)
    _write_static_headers(root)
    output = tmp_path / "parent" / "out"

    copied = copyStaticFiles(output)

    assert output.is_dir()
    assert {path.name for path in output.iterdir()} == set(copied)
    assert copied == list(EXPECTED_STATIC_HEADERS)


def test_copy_static_files_rejects_regular_file_output(tmp_path, monkeypatch):
    root = _fake_resource_tree(tmp_path, monkeypatch)
    _write_static_headers(root)
    output = tmp_path / "out"
    output.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="not a directory"):
        copyStaticFiles(output)


def test_copy_static_files_preflights_all_resources(tmp_path, monkeypatch):
    root = _fake_resource_tree(tmp_path, monkeypatch)
    _write_static_headers(root)
    missing_name = EXPECTED_STATIC_HEADERS[2]
    (root / "Source" / missing_name).unlink()
    output = tmp_path / "out"

    with pytest.raises(FileNotFoundError, match=missing_name):
        copyStaticFiles(output)

    assert not output.exists()


def test_custom_kernel_names_filters_suffixes_and_sorts_deterministically(monkeypatch):
    class Resource:
        def __init__(self, name, is_file=True):
            self.name = name
            self._is_file = is_file

        def is_file(self):
            return self._is_file

    first_order = [
        Resource("b.s"),
        Resource("a.s"),
        Resource("foo.bar.s"),
        Resource(".hidden.s"),
        Resource("note.txt"),
        Resource("directory.s", is_file=False),
    ]
    orders = [first_order, list(reversed(first_order))]

    class ResourceDir:
        def iterdir(self):
            return iter(orders.pop(0))

    monkeypatch.setattr(resources, "_custom_kernels", lambda: ResourceDir())

    expected = [".hidden", "a", "b", "foo.bar"]
    assert resources.custom_kernel_names() == expected
    assert resources.custom_kernel_names() == expected


def test_custom_kernel_text_uses_resource_root(tmp_path, monkeypatch):
    root = _fake_resource_tree(tmp_path, monkeypatch)
    (root / "CustomKernels" / "kernel.s").write_text("s_nop 0\n", encoding="utf-8")

    assert resources.custom_kernel_text("kernel") == "s_nop 0\n"


def test_custom_kernel_text_raises_for_missing_resource(tmp_path, monkeypatch):
    _fake_resource_tree(tmp_path, monkeypatch)

    with pytest.raises(FileNotFoundError, match="missing.s"):
        resources.custom_kernel_text("missing")


@pytest.mark.parametrize(
    "name",
    [
        "../kernel",
        "/tmp/kernel",
        "dir/kernel",
        r"dir\kernel",
        "C:kernel",
    ],
)
def test_custom_kernel_text_rejects_paths(name):
    with pytest.raises(ValueError):
        resources.custom_kernel_text(name)


def test_known_bugs_text_uses_resource_root(tmp_path, monkeypatch):
    root = _fake_resource_tree(tmp_path, monkeypatch)
    text = "version: 1\nskips: []\n"
    (root / "TensileLogic" / "known_bugs.yaml").write_text(text, encoding="utf-8")

    assert resources.known_bugs_text() == text


def test_ductile_defaults_text_uses_resource_root(tmp_path, monkeypatch):
    root = _fake_resource_tree(tmp_path, monkeypatch)
    defaults = root / "ductile" / "config"
    defaults.mkdir(parents=True)
    text = "runner:\n  name: pytest\n"
    (defaults / "defaults.yaml").write_text(text, encoding="utf-8")

    assert resources.ductile_defaults_text() == text


def test_known_bugs_text_raises_for_missing_resource(tmp_path, monkeypatch):
    _fake_resource_tree(tmp_path, monkeypatch)

    with pytest.raises(FileNotFoundError, match="known_bugs.yaml"):
        resources.known_bugs_text()


def test_resource_helpers_work_from_zip_package(tmp_path, monkeypatch):
    archive = tmp_path / "resources.zip"
    package = "zip_tensile_resources"
    known_bugs = "version: 1\nskips: []\n"
    defaults = "runner:\n  name: pytest\n"

    # Import the real module from the archive instead of monkeypatching _root.
    # This fails if resource lookup regresses to a __file__-derived Path.
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr(f"{package}/__init__.py", "")
        zip_file.write(resources.__file__, f"{package}/resources.py")
        for name in EXPECTED_STATIC_HEADERS:
            zip_file.writestr(
                f"{package}/Source/{name}", f"contents for {name}\n"
            )
        zip_file.writestr(f"{package}/CustomKernels/b.s", "kernel b\n")
        zip_file.writestr(f"{package}/CustomKernels/a.s", "kernel a\n")
        zip_file.writestr(f"{package}/CustomKernels/readme.txt", "not a kernel\n")
        zip_file.writestr(
            f"{package}/TensileLogic/known_bugs.yaml", known_bugs
        )
        zip_file.writestr(
            f"{package}/ductile/config/defaults.yaml", defaults
        )

    monkeypatch.syspath_prepend(str(archive))
    zip_resources = importlib.import_module(f"{package}.resources")
    try:
        assert zip_resources.__file__.startswith(f"{archive}/")
        assert isinstance(zip_resources._root(), ZipPath)
        assert zip_resources.custom_kernel_names() == ["a", "b"]
        assert zip_resources.custom_kernel_text("a") == "kernel a\n"
        assert zip_resources.known_bugs_text() == known_bugs
        assert zip_resources.ductile_defaults_text() == defaults

        output = tmp_path / "headers"
        copied = zip_resources.copy_static_headers(output)

        assert copied == list(EXPECTED_STATIC_HEADERS)
        for name in EXPECTED_STATIC_HEADERS:
            assert (output / name).read_text(encoding="utf-8") == (
                f"contents for {name}\n"
            )
    finally:
        sys.modules.pop(f"{package}.resources", None)
        sys.modules.pop(package, None)


def test_real_static_header_resources_are_available(tmp_path):
    copied = copyStaticFiles(tmp_path)

    assert copied == list(EXPECTED_STATIC_HEADERS)
    for name in EXPECTED_STATIC_HEADERS:
        assert (tmp_path / name).is_file()


def test_real_custom_kernel_resources_are_available():
    names = resources.custom_kernel_names()

    assert names
    assert KNOWN_CUSTOM_KERNEL in names


def test_real_known_bugs_resource_is_parseable_yaml():
    data = yaml.safe_load(resources.known_bugs_text())

    assert data["version"] == 1
    assert isinstance(data["skips"], list)

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
import yaml

from Tensile import Resources
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
    monkeypatch.setattr(Resources, "_root", lambda: root)
    return root


def _write_static_headers(root: Path) -> None:
    for name in EXPECTED_STATIC_HEADERS:
        (root / "Source" / name).write_text(f"contents for {name}\n", encoding="utf-8")


def test_static_header_paths_are_exact_and_immutable():
    paths = Resources.static_header_paths()

    assert tuple(path.name for path in paths) == EXPECTED_STATIC_HEADERS
    assert isinstance(paths, tuple)

    with pytest.raises(AttributeError):
        paths.append("unexpected.h")
    assert tuple(path.name for path in Resources.static_header_paths()) == EXPECTED_STATIC_HEADERS


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

    monkeypatch.setattr(Resources, "_custom_kernels", lambda: ResourceDir())

    expected = [".hidden", "a", "b", "foo.bar"]
    assert Resources.custom_kernel_names() == expected
    assert Resources.custom_kernel_names() == expected


def test_custom_kernel_text_uses_resource_root(tmp_path, monkeypatch):
    root = _fake_resource_tree(tmp_path, monkeypatch)
    (root / "CustomKernels" / "kernel.s").write_text("s_nop 0\n", encoding="utf-8")

    assert Resources.custom_kernel_text("kernel") == "s_nop 0\n"


def test_custom_kernel_text_raises_for_missing_resource(tmp_path, monkeypatch):
    _fake_resource_tree(tmp_path, monkeypatch)

    with pytest.raises(FileNotFoundError, match="missing.s"):
        Resources.custom_kernel_text("missing")


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
        Resources.custom_kernel_text(name)


def test_known_bugs_text_uses_resource_root(tmp_path, monkeypatch):
    root = _fake_resource_tree(tmp_path, monkeypatch)
    text = "version: 1\nskips: []\n"
    (root / "TensileLogic" / "known_bugs.yaml").write_text(text, encoding="utf-8")

    assert Resources.known_bugs_text() == text


def test_ductile_defaults_text_uses_resource_root(tmp_path, monkeypatch):
    root = _fake_resource_tree(tmp_path, monkeypatch)
    defaults = root / "ductile" / "config"
    defaults.mkdir(parents=True)
    text = "runner:\n  name: pytest\n"
    (defaults / "defaults.yaml").write_text(text, encoding="utf-8")

    assert Resources.ductile_defaults_text() == text


def test_known_bugs_text_raises_for_missing_resource(tmp_path, monkeypatch):
    _fake_resource_tree(tmp_path, monkeypatch)

    with pytest.raises(FileNotFoundError, match="known_bugs.yaml"):
        Resources.known_bugs_text()


def test_real_static_header_resources_are_available(tmp_path):
    copied = copyStaticFiles(tmp_path)

    assert copied == list(EXPECTED_STATIC_HEADERS)
    for name in EXPECTED_STATIC_HEADERS:
        assert (tmp_path / name).is_file()


def test_real_custom_kernel_resources_are_available():
    names = Resources.custom_kernel_names()

    assert names
    assert KNOWN_CUSTOM_KERNEL in names


def test_real_known_bugs_resource_is_parseable_yaml():
    data = yaml.safe_load(Resources.known_bugs_text())

    assert data["version"] == 1
    assert isinstance(data["skips"], list)

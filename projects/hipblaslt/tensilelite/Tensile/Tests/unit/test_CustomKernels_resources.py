# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest

import Tensile.CustomKernels as CustomKernels


pytestmark = pytest.mark.unit

KNOWN_CUSTOM_KERNEL = (
    "Custom_Cijk_Ailk_Bljk_F8NH_HHS_BH_Bias_AS_SAB_SAV_shortname0_gfx942"
)


def test_default_names_use_deterministic_bundled_order(monkeypatch):
    monkeypatch.setattr(CustomKernels, "custom_kernel_names", lambda: ["a", "b"])

    assert CustomKernels.getAllCustomKernelNames() == ["a", "b"]


def test_default_contents_use_bundled_resources(monkeypatch):
    monkeypatch.setattr(CustomKernels, "custom_kernel_text", lambda name: f"contents: {name}")

    assert CustomKernels.getCustomKernelContents("kernel") == "contents: kernel"


def test_custom_kernel_source_filters_preload_for_old_rocm(monkeypatch):
    text = "a\n.amdhsa_user_sgpr_kernarg_preload 1\nb\n"
    monkeypatch.setattr(CustomKernels, "custom_kernel_text", lambda name: text)

    assert (
        CustomKernels.getCustomKernelSource("kernel", SimpleNamespace(major=6, patch=32649))
        == "a\nb\n"
    )


def test_custom_kernel_source_keeps_preload_for_new_rocm(monkeypatch):
    text = "a\n.amdhsa_user_sgpr_kernarg_preload 1\nb\n"
    monkeypatch.setattr(CustomKernels, "custom_kernel_text", lambda name: text)

    assert CustomKernels.getCustomKernelSource(
        "kernel", SimpleNamespace(major=6, patch=32650)
    ) == text


def test_custom_kernel_source_keeps_preload_for_later_rocm_major(monkeypatch):
    text = "a\n.amdhsa_user_sgpr_kernarg_preload 1\nb\n"
    monkeypatch.setattr(CustomKernels, "custom_kernel_text", lambda name: text)

    assert CustomKernels.getCustomKernelSource(
        "kernel", SimpleNamespace(major=7, patch=0)
    ) == text


def test_default_missing_custom_kernel_preserves_runtime_error(monkeypatch):
    def raise_missing(name):
        raise FileNotFoundError(f"{name}.s")

    monkeypatch.setattr(CustomKernels, "custom_kernel_text", raise_missing)

    with pytest.raises(RuntimeError, match="missing"):
        CustomKernels.getCustomKernelContents("missing")


def test_default_resource_read_error_preserves_runtime_error(monkeypatch):
    def raise_permission_error(name):
        raise PermissionError(name)

    monkeypatch.setattr(CustomKernels, "custom_kernel_text", raise_permission_error)

    with pytest.raises(RuntimeError, match="kernel"):
        CustomKernels.getCustomKernelContents("kernel")


def test_explicit_directory_bypasses_bundled_resources(tmp_path, monkeypatch):
    def fail_if_called():
        raise AssertionError("bundled resource path should not be used")

    monkeypatch.setattr(CustomKernels, "custom_kernel_names", fail_if_called)
    monkeypatch.setattr(CustomKernels, "custom_kernel_text", fail_if_called)
    (tmp_path / "local.s").write_text("local custom kernel\n", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("not a kernel\n", encoding="utf-8")

    assert CustomKernels.getAllCustomKernelNames(directory=str(tmp_path)) == ["local"]
    assert (
        CustomKernels.getCustomKernelContents("local", directory=str(tmp_path))
        == "local custom kernel\n"
    )


def test_explicit_directory_missing_uses_legacy_runtime_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        CustomKernels,
        "custom_kernel_text",
        lambda name: (_ for _ in ()).throw(AssertionError("should not be used")),
    )

    with pytest.raises(RuntimeError, match=str(tmp_path / "missing")):
        CustomKernels.getCustomKernelContents("missing", directory=str(tmp_path))


def test_real_default_custom_kernel_resource_is_available():
    names = CustomKernels.getAllCustomKernelNames()

    assert KNOWN_CUSTOM_KERNEL in names
    assert "custom.config" in CustomKernels.getCustomKernelContents(KNOWN_CUSTOM_KERNEL)

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx1250 A0 as the gfx1250-strict target: ARCH names, logic fields, build and run env."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from geko.cli import build_parser
from geko.config_generator.constants import HARDWARE_MAP, library_logic_architecture
from geko.config_generator.load_input_config import apply_input_config_defaults
from geko.config_generator.output_writer import EntityOutputWriter, write_run_script
from geko.constants import LEGACY_ARCH_ALIASES, SUPPORTED_ARCH, canonical_arch, runtime_env
from geko.optim import optim
from geko.schemas import GemmConfig, GemmType

STRICT_KEYS = ("gfx1250-strict", "gfx1250-strict_96cu", "gfx1250-strict_192cu")


@pytest.mark.parametrize("old,new", sorted(LEGACY_ARCH_ALIASES.items()))
def test_retired_arch_is_rewritten_with_a_warning(old: str, new: str, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="GEKO"):
        assert canonical_arch(old) == new
    assert "deprecated" in caplog.text and new in caplog.text


def test_supported_arch_passes_through_silently(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="GEKO"):
        assert [canonical_arch(a) for a in SUPPORTED_ARCH] == list(SUPPORTED_ARCH)
    assert caplog.text == ""


def test_no_retired_name_is_still_supported() -> None:
    assert not set(LEGACY_ARCH_ALIASES) & set(SUPPORTED_ARCH)
    assert set(LEGACY_ARCH_ALIASES.values()) <= set(SUPPORTED_ARCH)


def test_cli_accepts_the_retired_arch_name() -> None:
    args = build_parser().parse_args(
        ["--tune", "--hipblaslt", "/unused", "--devices", "0", "--list", "/unused.yaml",
         "--arch", "gfx1250v0_96cu"]
    )
    assert args.arch == "gfx1250-strict_96cu"


def test_input_config_defaults_accept_the_retired_arch_name() -> None:
    cfg = {"ARCH": "gfx1250v0"}
    apply_input_config_defaults(cfg)
    assert cfg["ARCH"] == "gfx1250-strict"
    assert cfg["CUs"] == HARDWARE_MAP["gfx1250-strict"]["CUs"]


def test_configure_rewrites_the_retired_arch_name(tmp_path: Path, monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(optim.cg, "run", lambda config, *a, **k: seen.update(config))
    gc = GemmConfig(GemmType.from_tensile("T", "N", "B", "B", "S"), [[512, 512, 1, 512]])
    optim.configure("/unused", gc, tmp_path, arch="gfx1250v0_96cu")
    assert seen["ARCH"] == "gfx1250-strict_96cu"


@pytest.mark.parametrize("arch", STRICT_KEYS)
def test_strict_logic_names_the_strict_target_in_both_fields(arch: str) -> None:
    logic = HARDWARE_MAP[arch]["LibraryLogic"]
    assert logic["ScheduleName"] == logic["ArchitectureName"] == '"gfx1250-strict"'


@pytest.mark.parametrize(
    "arch,target",
    [
        ("gfx1250_96cu", "gfx1250"),
        ("gfx1250-strict_96cu", "gfx1250-strict"),
        ("gfx942_80cu", "gfx942"),
        ("gfx950", "gfx950"),
    ],
)
def test_client_builds_for_the_compiler_target_not_the_arch_key(arch: str, target: str) -> None:
    assert library_logic_architecture(arch) == target


def test_only_the_strict_target_needs_strict_runtime_mode() -> None:
    assert runtime_env("gfx1250-strict") == {"HSA_DISABLE_GFX12_STRICT": "0"}
    assert runtime_env("gfx1250") == {}
    assert runtime_env(None) == {}


def test_run_script_sets_extra_env_on_the_command(tmp_path: Path) -> None:
    script = tmp_path / "e.sh"
    write_run_script(script, "e", tmp_path, extra_env={"HSA_DISABLE_GFX12_STRICT": "0"})
    assert "HSA_DISABLE_GFX12_STRICT=0 PYTHONPATH=" in script.read_text()


@pytest.mark.parametrize("architecture,strict", [('"gfx1250-strict"', True), ('"gfx1250"', False)])
def test_entity_run_script_follows_the_logic_architecture(
    tmp_path: Path, monkeypatch, architecture: str, strict: bool
) -> None:
    writer = EntityOutputWriter(tmp_path, "BBS_TN", tmp_path)
    monkeypatch.setattr(writer._tuning_writer, "write", lambda *a, **k: None)
    writer.write_entity_files_only(None, {"LibraryLogic": {"ArchitectureName": architecture}}, "", "e")
    assert ("HSA_DISABLE_GFX12_STRICT=0" in (tmp_path / "e.sh").read_text()) is strict

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``run_all.py``'s auto-forwarded ``--rocke-require-*`` flags.

``test_installed_hipdnn_assets_imply_require_flag`` and
``test_installed_pack_headers_imply_require_flag`` fail an installed tree that
does not pass the matching flag, on the premise that the installed CTest
entries always do. ``run_all.py`` is itself part of an installed tree (its
own top-of-file usage runs it there), so ``_forwarded_asset_require_flags``
must add each flag under the same condition those tests check: the matching
copy actually present, and the flag not already forwarded.
"""

from __future__ import annotations

from . import run_all


def test_adds_neither_flag_without_either_copy(monkeypatch):
    monkeypatch.setattr(run_all.hipdnn_assets, "installed_copy_present", lambda: False)
    monkeypatch.setattr(
        run_all.hkp_pack_headers, "installed_copy_present", lambda: False
    )
    assert run_all._forwarded_asset_require_flags([]) == []


def test_adds_hipdnn_assets_flag_when_that_copy_is_installed(monkeypatch):
    monkeypatch.setattr(run_all.hipdnn_assets, "installed_copy_present", lambda: True)
    monkeypatch.setattr(
        run_all.hkp_pack_headers, "installed_copy_present", lambda: False
    )
    assert run_all._forwarded_asset_require_flags([]) == [
        "--rocke-require-hipdnn-assets"
    ]


def test_adds_hkp_pack_headers_flag_when_that_copy_is_installed(monkeypatch):
    monkeypatch.setattr(run_all.hipdnn_assets, "installed_copy_present", lambda: False)
    monkeypatch.setattr(
        run_all.hkp_pack_headers, "installed_copy_present", lambda: True
    )
    assert run_all._forwarded_asset_require_flags([]) == [
        "--rocke-require-hkp-pack-headers"
    ]


def test_adds_both_flags_when_both_copies_are_installed(monkeypatch):
    monkeypatch.setattr(run_all.hipdnn_assets, "installed_copy_present", lambda: True)
    monkeypatch.setattr(
        run_all.hkp_pack_headers, "installed_copy_present", lambda: True
    )
    assert run_all._forwarded_asset_require_flags([]) == [
        "--rocke-require-hipdnn-assets",
        "--rocke-require-hkp-pack-headers",
    ]


def test_does_not_repeat_a_flag_the_caller_already_forwarded(monkeypatch):
    monkeypatch.setattr(run_all.hipdnn_assets, "installed_copy_present", lambda: True)
    monkeypatch.setattr(
        run_all.hkp_pack_headers, "installed_copy_present", lambda: True
    )
    assert run_all._forwarded_asset_require_flags(
        ["--rocke-require-hipdnn-assets"]
    ) == ["--rocke-require-hkp-pack-headers"]


def _installed(monkeypatch, *, hipdnn: bool, headers: bool) -> None:
    monkeypatch.setattr(run_all.hipdnn_assets, "installed_copy_present", lambda: hipdnn)
    monkeypatch.setattr(
        run_all.hkp_pack_headers, "installed_copy_present", lambda: headers
    )


def test_main_forwards_both_flags_to_file_workers(monkeypatch):
    _installed(monkeypatch, hipdnn=True, headers=True)
    calls = []
    monkeypatch.setattr(
        run_all,
        "_run_affordable_file_stage",
        lambda **kwargs: calls.append(kwargs) or run_all._StageResult(0),
    )
    monkeypatch.setattr(
        run_all.sys,
        "argv",
        ["run_all.py", "--pytest-only", "--no-ir-gates", "--jobs", "2", "--", "-q"],
    )
    assert run_all.main() == 0
    assert [call["extra"] for call in calls] == [
        ["-q", "--rocke-require-hipdnn-assets", "--rocke-require-hkp-pack-headers"]
    ]


def test_main_forwards_the_present_flag_to_a_single_pytest_process(monkeypatch):
    _installed(monkeypatch, hipdnn=False, headers=True)
    commands = []
    monkeypatch.setattr(
        run_all,
        "_run",
        lambda name, command, **kwargs: commands.append(command)
        or run_all._StageResult(0),
    )
    monkeypatch.setattr(
        run_all.sys,
        "argv",
        ["run_all.py", "--pytest-only", "--no-ir-gates", "--jobs", "0"],
    )
    assert run_all.main() == 0
    assert len(commands) == 1
    assert commands[0].count("--rocke-require-hkp-pack-headers") == 1
    assert "--rocke-require-hipdnn-assets" not in commands[0]


def test_main_does_not_forward_flags_to_explicit_gates(monkeypatch):
    _installed(monkeypatch, hipdnn=True, headers=True)
    commands = []
    monkeypatch.setattr(
        run_all,
        "_run",
        lambda name, command, **kwargs: commands.append(command)
        or run_all._StageResult(0, skipped=True),
    )
    monkeypatch.setattr(
        run_all.sys, "argv", ["run_all.py", "--gate", "counted-structural"]
    )
    run_all.main()
    assert commands
    assert not any(
        flag.startswith("--rocke-require-") for command in commands for flag in command
    )

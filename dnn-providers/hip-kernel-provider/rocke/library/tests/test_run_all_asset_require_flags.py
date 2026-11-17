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

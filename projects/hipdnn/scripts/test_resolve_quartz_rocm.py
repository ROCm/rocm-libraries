# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for projects/hipdnn/scripts/resolve_quartz_rocm.py."""

from __future__ import annotations

import http.client
import importlib.util
import json
from pathlib import Path
import sys
import urllib.error
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent / "resolve_quartz_rocm.py"
_SPEC = importlib.util.spec_from_file_location("resolve_quartz_rocm", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
resolve_quartz_rocm = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = resolve_quartz_rocm
_SPEC.loader.exec_module(resolve_quartz_rocm)


def _status_doc(
    day: str,
    *,
    platform: str = "linux",
    arch: str = "gfx94X-dcgpu",
    architectures: list[str] | None = None,
    build_status: str = "success",
    test_status: str | None = "success",
    wheels_url: str | None = "https://nightly.repo.amd.com/rocm/whl-next/",
    tarballs_url: str | None = "https://nightly.repo.amd.com/rocm/core/tarball/",
    schema_version: str = "2.3",
    rocm_version: str | None = None,
    therock_commit: str = "abcdef1234567890abcdef1234567890abcdef12",
) -> dict:
    urls: dict[str, str] = {}
    if wheels_url is not None:
        urls["wheels"] = wheels_url
    if tarballs_url is not None:
        urls["tarballs"] = tarballs_url
    doc = {
        "schema_version": schema_version,
        "build_date": day,
        "rocm_version": rocm_version or f"10.3.0a{day}",
        "therock_commit": therock_commit,
        "summary": {
            platform: {
                "architectures": (
                    architectures if architectures is not None else [arch]
                ),
                "rocm": {"build": {"status": build_status}},
                "urls": urls,
            }
        },
        "pipelines": {"rocm": {"test": {platform: {}}}},
    }
    if test_status is not None:
        doc["pipelines"]["rocm"]["test"][platform][arch] = {"status": test_status}
    return doc


def _fake_fetcher(pages: dict[str, str | None | Exception]):
    def fetch(url: str) -> str | None:
        for suffix, body in pages.items():
            if url.endswith(suffix):
                if isinstance(body, Exception):
                    raise body
                return body
        return None

    return fetch


def test_gpu_target_to_quartz_arch_mappings() -> None:
    assert resolve_quartz_rocm.gpu_target_to_quartz_arch("gfx942") == "gfx94X-dcgpu"
    assert resolve_quartz_rocm.gpu_target_to_quartz_arch("gfx950") == "gfx950-dcgpu"
    assert resolve_quartz_rocm.gpu_target_to_quartz_arch("gfx1151") == "gfx1151"
    assert resolve_quartz_rocm.gpu_target_to_quartz_arch("gfx1151-all") == "gfx1151"
    assert resolve_quartz_rocm.gpu_target_to_quartz_arch("gfx90a") == "gfx90a"
    assert resolve_quartz_rocm.gpu_target_to_quartz_arch("gfx906-all") == "gfx906"
    assert resolve_quartz_rocm.gpu_target_to_quartz_arch("gfx908-all") == "gfx908"
    assert resolve_quartz_rocm.gpu_target_to_quartz_arch("gfx1100-all") == "gfx110X-all"
    assert resolve_quartz_rocm.gpu_target_to_quartz_arch("gfx110X-all") == "gfx110X-all"
    assert resolve_quartz_rocm.gpu_target_to_quartz_arch("gfx1201") == "gfx120X-all"
    assert resolve_quartz_rocm.gpu_target_to_quartz_arch("gfx1030") == "gfx103X-all"
    assert resolve_quartz_rocm.gpu_target_to_quartz_arch("gfx1033") == "gfx103X-all"
    assert resolve_quartz_rocm.gpu_target_to_quartz_arch("gfx1170") == "gfx117X-all"
    assert resolve_quartz_rocm.gpu_target_to_quartz_arch("gfx1171") == "gfx117X-all"
    assert (
        resolve_quartz_rocm.gpu_target_to_quartz_arch("gfx94X-dcgpu") == "gfx94X-dcgpu"
    )


def test_skips_failed_tests_and_selects_newest_passing_nightly() -> None:
    pages = {
        "latest.json": "20261127/status.json\n",
        "20261127/status.json": json.dumps(
            _status_doc("20261127", test_status="failure")
        ),
        "20261126/status.json": json.dumps(
            _status_doc("20261126", build_status="failure", test_status=None)
        ),
        "20261125/status.json": json.dumps(
            _status_doc("20261125", test_status="success")
        ),
    }
    sel = resolve_quartz_rocm.resolve_quartz_rocm(
        platform_name="linux",
        gpu_target="gfx942",
        max_age_days=7,
        fetcher=_fake_fetcher(pages),
    )
    assert sel.rocm_version == "10.3.0a20261125"
    assert sel.wheels_url == "https://nightly.repo.amd.com/rocm/whl-next/"
    assert sel.quartz_arch == "gfx94X-dcgpu"
    assert sel.test_status == "success"
    assert sel.fallback_used is False


def test_passing_tests_beat_newer_untested_nightly() -> None:
    pages = {
        "latest.json": "20261127/status.json",
        "20261127/status.json": json.dumps(_status_doc("20261127", test_status=None)),
        "20261126/status.json": json.dumps(
            _status_doc("20261126", test_status="success")
        ),
    }
    sel = resolve_quartz_rocm.resolve_quartz_rocm(
        platform_name="linux",
        gpu_target="gfx942",
        max_age_days=3,
        fetcher=_fake_fetcher(pages),
    )
    assert sel.rocm_version == "10.3.0a20261126"
    assert sel.test_status == "success"
    assert sel.fallback_used is False


def test_accepts_untested_arch_when_no_day_has_passing_tests() -> None:
    pages = {
        "latest.json": "20261127/status.json",
        "20261127/status.json": json.dumps(
            _status_doc("20261127", arch="gfx950-dcgpu", test_status=None)
        ),
        "20261126/status.json": json.dumps(
            _status_doc("20261126", arch="gfx950-dcgpu", test_status=None)
        ),
    }
    sel = resolve_quartz_rocm.resolve_quartz_rocm(
        platform_name="linux",
        gpu_target="gfx950",
        max_age_days=2,
        fetcher=_fake_fetcher(pages),
    )
    assert sel.rocm_version == "10.3.0a20261127"
    assert sel.test_status == "untested"
    assert sel.fallback_used is False


def test_tarballs_and_multiarch_artifact_selection() -> None:
    pages = {
        "latest.json": "20261127/status.json",
        "20261127/status.json": json.dumps(
            _status_doc("20261127", wheels_url=None, tarballs_url=None)
        ),
        "20261126/status.json": json.dumps(
            _status_doc("20261126", wheels_url=None, test_status="failure")
        ),
    }
    sel = resolve_quartz_rocm.resolve_quartz_rocm(
        platform_name="linux",
        gpu_target="multiarch",
        artifact="tarballs",
        max_age_days=3,
        fetcher=_fake_fetcher(pages),
    )
    assert sel.rocm_version == "10.3.0a20261126"
    assert sel.tarballs_url == "https://nightly.repo.amd.com/rocm/core/tarball/"
    assert sel.quartz_arch == "multiarch"


def test_rejects_nightly_when_arch_absent_from_architectures() -> None:
    pages = {
        "latest.json": "20261127/status.json",
        "20261127/status.json": json.dumps(
            _status_doc("20261127", architectures=["gfx90a"], test_status=None)
        ),
    }
    with pytest.raises(resolve_quartz_rocm.QuartzResolutionError):
        resolve_quartz_rocm.resolve_quartz_rocm(
            platform_name="linux",
            gpu_target="gfx942",
            max_age_days=1,
            allow_build_only_fallback=True,
            fetcher=_fake_fetcher(pages),
        )


def test_respects_exact_lookback_window_boundary() -> None:
    pages = {
        "latest.json": "20261127/status.json",
        "20261127/status.json": None,
        "20261126/status.json": None,
        "20261125/status.json": json.dumps(_status_doc("20261125")),
    }
    with pytest.raises(resolve_quartz_rocm.QuartzResolutionError):
        resolve_quartz_rocm.resolve_quartz_rocm(
            platform_name="linux",
            gpu_target="gfx942",
            max_age_days=2,
            fetcher=_fake_fetcher(pages),
        )
    sel = resolve_quartz_rocm.resolve_quartz_rocm(
        platform_name="linux",
        gpu_target="gfx942",
        max_age_days=3,
        fetcher=_fake_fetcher(pages),
    )
    assert sel.rocm_version == "10.3.0a20261125"


def test_build_only_fallback_used_only_when_enabled() -> None:
    pages = {
        "latest.json": "20261127/status.json",
        "20261127/status.json": json.dumps(
            _status_doc(
                "20261127", platform="windows", arch="gfx1151", test_status="cancelled"
            )
        ),
        "20261126/status.json": json.dumps(
            _status_doc(
                "20261126", platform="windows", arch="gfx1151", test_status="failure"
            )
        ),
    }
    with pytest.raises(resolve_quartz_rocm.QuartzResolutionError):
        resolve_quartz_rocm.resolve_quartz_rocm(
            platform_name="windows",
            gpu_target="gfx1151",
            max_age_days=2,
            allow_build_only_fallback=False,
            fetcher=_fake_fetcher(pages),
        )

    sel = resolve_quartz_rocm.resolve_quartz_rocm(
        platform_name="windows",
        gpu_target="gfx1151",
        max_age_days=2,
        allow_build_only_fallback=True,
        fetcher=_fake_fetcher(pages),
    )
    assert sel.rocm_version == "10.3.0a20261127"
    assert sel.test_status == "cancelled"
    assert sel.fallback_used is True


def test_rejects_disallowed_host_non_https_and_unsupported_schema() -> None:
    pages = {
        "latest.json": "20261127/status.json",
        "20261127/status.json": json.dumps(
            _status_doc(
                "20261127",
                wheels_url="https://therock-nightly-artifacts.s3.amazonaws.com/123/python/",
            )
        ),
        "20261126/status.json": json.dumps(
            _status_doc("20261126", wheels_url="http://nightly.repo.amd.com/rocm/whl/")
        ),
        "20261125/status.json": json.dumps(
            _status_doc("20261125", schema_version="3.0")
        ),
        "20261124/status.json": json.dumps(_status_doc("20261124")),
    }
    sel = resolve_quartz_rocm.resolve_quartz_rocm(
        platform_name="linux",
        gpu_target="gfx942",
        max_age_days=4,
        fetcher=_fake_fetcher(pages),
    )
    assert sel.rocm_version == "10.3.0a20261124"


def test_skips_null_nested_fields_invalid_json_and_per_day_fetch_error() -> None:
    null_doc = _status_doc("20261127")
    null_doc["pipelines"] = {"rocm": {"test": None}}
    null_doc["summary"]["linux"]["rocm"] = None
    pages = {
        "latest.json": json.dumps({"build_date": "20261127"}),
        "20261127/status.json": json.dumps(null_doc),
        "20261126/status.json": "{not json",
        "20261125/status.json": resolve_quartz_rocm.QuartzResolutionError("HTTP 503"),
        "20261124/status.json": json.dumps(_status_doc("20261124")),
    }
    sel = resolve_quartz_rocm.resolve_quartz_rocm(
        platform_name="linux",
        gpu_target="gfx942",
        max_age_days=4,
        fetcher=_fake_fetcher(pages),
    )
    assert sel.rocm_version == "10.3.0a20261124"


def test_rejects_unsafe_version_short_commit_invalid_args_and_empty_test_status() -> (
    None
):
    bad_status_doc = _status_doc("20261127")
    bad_status_doc["pipelines"]["rocm"]["test"]["linux"]["gfx94X-dcgpu"] = {}
    pages = {
        "latest.json": "20261127/status.json",
        "20261127/status.json": json.dumps(bad_status_doc),
        "20261126/status.json": json.dumps(
            _status_doc("20261126", rocm_version="10.3.0; rm -rf /")
        ),
        "20261125/status.json": json.dumps(
            _status_doc("20261125", therock_commit="abcdef1")
        ),
    }
    with pytest.raises(resolve_quartz_rocm.QuartzResolutionError):
        resolve_quartz_rocm.resolve_quartz_rocm(
            platform_name="linux",
            gpu_target="gfx942",
            max_age_days=3,
            allow_build_only_fallback=False,
            fetcher=_fake_fetcher(pages),
        )
    with pytest.raises(resolve_quartz_rocm.QuartzResolutionError):
        resolve_quartz_rocm.resolve_quartz_rocm(
            platform_name="darwin", gpu_target="gfx942", fetcher=_fake_fetcher(pages)
        )
    with pytest.raises(resolve_quartz_rocm.QuartzResolutionError):
        resolve_quartz_rocm.resolve_quartz_rocm(
            platform_name="linux",
            gpu_target="gfx942",
            artifact="rpm",
            fetcher=_fake_fetcher(pages),
        )
    with pytest.raises(resolve_quartz_rocm.QuartzResolutionError):
        resolve_quartz_rocm.resolve_quartz_rocm(
            platform_name="linux",
            gpu_target="gfx942",
            max_age_days=0,
            fetcher=_fake_fetcher(pages),
        )


def test_fetch_text_retries_transient_http_and_incomplete_read(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(resolve_quartz_rocm.time, "sleep", sleeps.append)

    class _FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> None:
            return None

        def read(self) -> bytes:
            return self._payload

    # 404 returns None immediately without sleeping.
    def raise_404(_req, timeout=20):
        raise urllib.error.HTTPError("https://x", 404, "Not Found", {}, None)

    monkeypatch.setattr(resolve_quartz_rocm.urllib.request, "urlopen", raise_404)
    assert resolve_quartz_rocm._fetch_text("https://x", retries=2) is None
    assert sleeps == []

    # Transient 503 retries once and succeeds.
    attempts = {"n": 0}

    def flaky_503(_req, timeout=20):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise urllib.error.HTTPError("https://x", 503, "Unavailable", {}, None)
        return _FakeResponse(b"ok-after-503")

    monkeypatch.setattr(resolve_quartz_rocm.urllib.request, "urlopen", flaky_503)
    assert resolve_quartz_rocm._fetch_text("https://x", retries=1) == "ok-after-503"
    assert attempts["n"] == 2
    assert len(sleeps) == 1

    # Truncated body (http.client.IncompleteRead) retries and succeeds.
    sleeps.clear()
    attempts["n"] = 0

    class _FlakyReadResponse(_FakeResponse):
        def read(self) -> bytes:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise http.client.IncompleteRead(b"partial")
            return b"ok-after-incomplete"

    monkeypatch.setattr(
        resolve_quartz_rocm.urllib.request,
        "urlopen",
        lambda _req, timeout=20: _FlakyReadResponse(b""),
    )
    assert (
        resolve_quartz_rocm._fetch_text("https://x", retries=1) == "ok-after-incomplete"
    )
    assert attempts["n"] == 2
    assert len(sleeps) == 1

    # Invalid UTF-8 wraps into QuartzResolutionError instead of escaping raw.
    monkeypatch.setattr(
        resolve_quartz_rocm.urllib.request,
        "urlopen",
        lambda _req, timeout=20: _FakeResponse(b"\xff\xfe"),
    )
    with pytest.raises(resolve_quartz_rocm.QuartzResolutionError):
        resolve_quartz_rocm._fetch_text("https://x", retries=0)

    # Exhausted 500 raises QuartzResolutionError.
    def raise_500(_req, timeout=20):
        raise urllib.error.HTTPError("https://x", 500, "Server Error", {}, None)

    monkeypatch.setattr(resolve_quartz_rocm.urllib.request, "urlopen", raise_500)
    with pytest.raises(resolve_quartz_rocm.QuartzResolutionError):
        resolve_quartz_rocm._fetch_text("https://x", retries=0)


def test_write_github_env_rejects_multiline(tmp_path: Path) -> None:
    sel = resolve_quartz_rocm.QuartzSelection(
        rocm_version="10.3.0a20261127\nINJECT=1",
        wheels_url="https://nightly.repo.amd.com/rocm/whl-next/",
        tarballs_url="https://nightly.repo.amd.com/rocm/core/tarball/",
        build_date="20261127",
        therock_commit="a" * 40,
        quartz_arch="gfx94X-dcgpu",
        test_status="success",
        fallback_used=False,
    )
    with pytest.raises(resolve_quartz_rocm.QuartzResolutionError):
        resolve_quartz_rocm._write_github_env(tmp_path / "env", sel)


def test_cli_output_modes_github_env_fallback_and_errors(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    sha = "abcdef1234567890abcdef1234567890abcdef12"
    pages = {
        "latest.json": "20261127/status.json",
        "20261127/status.json": json.dumps(
            _status_doc(
                "20261127",
                platform="windows",
                arch="gfx1151",
                test_status="failure",
                therock_commit=sha,
            )
        ),
    }
    monkeypatch.setattr(resolve_quartz_rocm, "_fetch_text", _fake_fetcher(pages))
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    env_file = tmp_path / "github_env"

    assert (
        resolve_quartz_rocm.main(
            ["--platform", "windows", "--gpu-target", "gfx1151", "--max-age-days", "1"]
        )
        == 1
    )
    assert "ERROR:" in capsys.readouterr().err

    assert (
        resolve_quartz_rocm.main(
            [
                "--platform",
                "windows",
                "--gpu-target",
                "gfx1151",
                "--max-age-days",
                "1",
                "--allow-build-only-fallback",
                "--github-env",
                str(env_file),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "::warning::" in captured.err
    out_lines = captured.out.splitlines()
    assert out_lines == [
        "ROCM_VERSION=10.3.0a20261127",
        "ROCM_WHEELS_URL=https://nightly.repo.amd.com/rocm/whl-next/",
    ]
    env_lines = env_file.read_text(encoding="utf-8").splitlines()
    assert env_lines == [
        "ROCM_VERSION=10.3.0a20261127",
        "ROCM_WHEELS_URL=https://nightly.repo.amd.com/rocm/whl-next/",
    ]

    assert (
        resolve_quartz_rocm.main(
            [
                "--platform",
                "windows",
                "--gpu-target",
                "gfx1151",
                "--max-age-days",
                "1",
                "--allow-build-only-fallback",
                "--version-only",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "10.3.0a20261127"

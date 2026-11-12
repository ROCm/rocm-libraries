# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


@pytest.mark.parametrize(
    "worker_args", [(), ("--numprocesses", "2")], ids=["serial", "xdist"]
)
@pytest.mark.parametrize(
    "test_args,exit_code",
    [
        pytest.param(("-k", "test_skipped"), 6, id="all-skipped"),
        pytest.param(("-k", "test_skipped", "--no-fail-if-no-tests"), 0, id="opt-out"),
        pytest.param(("-k", "test_passed or test_skipped"), 0, id="passing"),
        pytest.param(("--collect-only",), 0, id="collect-only"),
        pytest.param(("-k", "test_failed"), 1, id="failure"),
        pytest.param(("-k", "no_such_test"), 5, id="deselected"),
    ],
)
def test_pytest_session_exit_codes(pytester, monkeypatch, worker_args, test_args, exit_code):
    # Let the temporary pytest session load the full project conftest.
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[3]), prepend=os.pathsep)
    pytester.makepyfile(
        test_session="""
        import pytest

        def test_passed():
            assert True

        def test_skipped():
            pytest.skip("synthetic skip")

        def test_failed():
            assert False
        """
    )
    result = pytester.runpytest_subprocess(
        "--quiet", "-p", "tensilelite.Tests.conftest", *worker_args, *test_args, timeout=60
    )
    assert result.ret == exit_code

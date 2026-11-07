# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check when the artifact gate releases, keeps waiting, or gives up."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))
import rocjitsu_wait_for_artifacts as gate  # noqa: E402

FAMILY, TARGET = "gfx125X-dcgpu", "gfx1250"
ALL = [
    f"{name}.tar.zst.sha256sum"
    for name in gate.math_libs_artifacts(TARGET) + gate.EMULATION_ARTIFACTS
]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def run(present, logs=(), active=lambda: True, timeout=10_000, appear_after=None):
    clock = Clock()
    store = set(present)

    def exists(key):
        if appear_after and clock.now >= appear_after[0]:
            store.update(appear_after[1])
        return key in store

    return gate.wait(
        FAMILY,
        TARGET,
        exists,
        lambda prefix: prefix in logs,
        active,
        timeout,
        poll=60,
        clock=clock,
        sleep=clock.sleep,
    )


class TestWaitForArtifacts(unittest.TestCase):
    def test_releases_when_everything_is_uploaded(self):
        self.assertIsNone(run(ALL))

    def test_keeps_waiting_until_late_uploads(self):
        self.assertIsNone(run(ALL[:2], appear_after=(600, ALL)))

    def test_archive_without_checksum_is_not_ready(self):
        partial = [key for key in ALL if "blas_lib_gfx1250" not in key]
        reason = run(partial, timeout=120)
        self.assertIn("timed out", reason)
        self.assertIn("blas_lib_gfx1250", reason)

    def test_finished_stage_without_artifacts_fails_fast(self):
        emulation_only = [key for key in ALL if "blas" not in key]
        reason = run(emulation_only, logs={f"logs/math-libs/{FAMILY}/"})
        self.assertEqual(
            reason,
            "math-libs stage finished without blas_lib_gfx1250, blas_test_gfx1250, "
            "blas_lib_generic, blas_test_generic",
        )

    def test_gives_up_once_build_jobs_are_gone(self):
        states = iter([True, True, False])
        reason = run([], active=lambda: next(states))
        self.assertIn("no Linux build job is running", reason)

    def test_waits_for_build_jobs_to_appear(self):
        reason = run([], active=lambda: False, appear_after=(900, ALL))
        self.assertIsNone(reason)

    def test_transient_errors_keep_waiting(self):
        calls = iter([OSError("503 Slow Down"), OSError("timed out")])

        def flaky():
            error = next(calls, None)
            if error:
                raise error
            return True

        self.assertIsNone(run(ALL[:2], active=flaky, appear_after=(300, ALL)))

    def test_gives_up_if_build_jobs_never_appear(self):
        reason = run([], active=lambda: False)
        self.assertIn("no Linux build job is running", reason)


if __name__ == "__main__":
    unittest.main()

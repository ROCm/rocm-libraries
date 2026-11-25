# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The older package's code in the work volume, /work/code (rkmon_lib/one_time_move.py, OLDER_CODE):
the one-time move keeps it, as the way back to the older package runs it; `rkmon standby off`'s end of the move removes
it with the other older copies, and its lines say so; one it cannot remove is kept, and the next start removes it. A
link in its place goes alone: what it links to stays. A home that is not the work volume's monitor home never removes
the folder beside it. The container's layout is the test's (the move's tests' Base)."""

import errno
import os
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_one_time_move import MOVE, RC, Base  # noqa: E402
from rkmon_lib import container_layout  # noqa: E402

GONE = "the older package's code, which nothing runs once the move has ended"


class OlderCodeTest(Base):
    def setUp(self):
        super().setUp()
        self.layer()
        self.older_config()
        # The older package's code: a release, and the link to it that ran.
        self.code = self.dir / "work" / "code"
        self.file(self.code / "releases" / "r1" / "rk")
        (self.code / "current").symlink_to("releases/r1")
        self.start()

    def end(self) -> list:
        return MOVE.end(self.home, RC.load(self.home).values)

    def test_the_move_keeps_it_and_the_end_removes_it(self):
        self.assertTrue((self.code / "current" / "rk").is_file())
        lines = self.end()
        self.assertFalse(os.path.lexists(self.code))
        self.assertIn(f"removed {self.code}: {GONE}", lines)
        self.assertEqual(MOVE.record(self.home)["kept"], [])

    def test_one_the_end_cannot_remove_is_kept_and_the_next_start_removes_it(self):
        why = os.strerror(errno.EACCES)
        with mock.patch.object(
            MOVE.shutil, "rmtree", side_effect=OSError(errno.EACCES, why)
        ):
            lines = self.end()
        self.assertIn(
            f"kept {self.code}: it cannot be removed: {why}; the next start removes it",
            lines,
        )
        self.assertEqual(
            MOVE.record(self.home)["kept"], [{"entry": "", "place": str(self.code)}]
        )
        self.assertEqual(self.start(), [f"removed {self.code}: {GONE}"])
        self.assertFalse(os.path.lexists(self.code))
        self.assertEqual(MOVE.record(self.home)["kept"], [])

    def test_a_link_in_its_place_goes_and_what_it_links_to_stays(self):
        target = self.dir / "code-elsewhere"
        self.code.rename(target)
        self.code.symlink_to(target)
        lines = self.end()
        self.assertFalse(os.path.lexists(self.code))
        self.assertTrue((target / "current" / "rk").is_file())
        self.assertIn(f"removed {self.code}: {GONE}", lines)
        self.assertEqual(MOVE.record(self.home)["kept"], [])

    def test_a_home_elsewhere_never_removes_the_folder_beside_it(self):
        elsewhere = self.dir / "elsewhere"
        elsewhere.mkdir()
        with mock.patch.object(container_layout, "MONITOR_HOME", str(elsewhere)):
            lines = self.end()
        self.assertTrue((self.code / "current" / "rk").is_file())
        self.assertEqual([line for line in lines if str(self.code) in line], [])


if __name__ == "__main__":
    unittest.main()

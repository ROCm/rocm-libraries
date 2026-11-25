# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/automerge_msg.py: an armed auto-merge keeps the squash title and message from when it was armed (lessons
L-136: a rename or a new body after arming never reaches the squash commit unnoticed).
"""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "automerge_msg.py"
sys.path.insert(0, str(TOOL.parent))
import automerge_msg  # noqa: E402


def pr(
    title="perf(rocke): add opt-in KDB/VPF",
    body="Line one.\n\nLine two.",
    armed_title=None,
    armed_msg=None,
    on=True,
):
    return {
        "title": title,
        "body": body,
        "auto_merge": (
            {
                "merge_method": "squash",
                "enabled_by": {"login": "someone"},
                "commit_title": armed_title,
                "commit_message": armed_msg,
            }
            if on
            else None
        ),
    }


class TestAutomergeMsg(unittest.TestCase):
    def test_off_says_off(self):
        self.assertEqual(automerge_msg.compare(pr(on=False), 1111), ["auto-merge: off"])

    def test_an_armed_title_and_body_that_match_are_current(self):
        out = automerge_msg.compare(
            pr(
                armed_title="perf(rocke): add opt-in KDB/VPF (#1111)",
                armed_msg="Line one.\r\n\r\nLine two.  \r\n",
            ),
            1111,
        )
        self.assertEqual(
            out,
            [
                "auto-merge: on (squash), armed by someone",
                "armed title = live title",
                "armed message = live body",
            ],
        )

    def test_a_rewrapped_message_with_the_same_words_is_current(self):
        out = automerge_msg.compare(
            pr(
                body="# This PR adds K LDS double buffering and V register prefetch.",
                armed_title="perf(rocke): add opt-in KDB/VPF (#1111)",
                armed_msg="# This PR adds K LDS double buffering and V\nregister prefetch.",
            ),
            1111,
        )
        self.assertEqual(out[2], "armed message = live body")

    def test_githubs_co_author_trailer_is_not_a_difference(self):
        # The armed message is the live body plus GitHub's squash trailer
        trailer = "\r\n\r\n---------\r\n\r\nCo-authored-by: Claude <noreply@example.com>\r\nCo-authored-by: A B <a@b.c>"
        out = automerge_msg.compare(
            pr(
                armed_title="perf(rocke): add opt-in KDB/VPF (#1111)",
                armed_msg="Line one.\n\nLine two." + trailer,
            ),
            1111,
        )
        self.assertEqual(out[2], "armed message = live body")
        out = automerge_msg.compare(
            pr(
                armed_title="perf(rocke): add opt-in KDB/VPF (#1111)",
                armed_msg="Line one.\n\nLine three." + trailer,
            ),
            1111,
        )
        self.assertEqual(
            out[2],
            "STALE message: the armed message differs from the live body from word 4 (armed "
            "'three.', live 'two.'); re-arm to commit the live body",
        )

    def test_a_single_commits_trailer_has_no_rule_and_is_not_a_difference(self):
        # One commit: GitHub appends the co-author line with no dashed rule
        trailer = "\n\nCo-authored-by: Claude <noreply@example.com>"
        out = automerge_msg.compare(
            pr(
                armed_title="perf(rocke): add opt-in KDB/VPF (#1111)",
                armed_msg="Line one.\n\nLine two." + trailer,
            ),
            1111,
        )
        self.assertEqual(out[2], "armed message = live body")
        out = automerge_msg.compare(
            pr(
                armed_title="perf(rocke): add opt-in KDB/VPF (#1111)",
                armed_msg="Line one." + trailer,
            ),
            1111,
        )
        self.assertEqual(
            out[2],
            "STALE message: the armed message differs from the live body from word 3 (armed "
            "'(end)', live 'Line two.'); re-arm to commit the live body",
        )

    def test_a_rename_after_arming_reads_stale(self):
        out = automerge_msg.compare(
            pr(
                armed_title="perf(rocke): add opt-in CFVST and KDB/VPF (#1111)",
                armed_msg="Line one.\n\nLine two.",
            ),
            1111,
        )
        self.assertTrue(
            out[1].startswith(
                "STALE title: armed 'perf(rocke): add opt-in CFVST and KDB/VPF (#1111)'"
            ),
            out[1],
        )
        self.assertEqual(out[2], "armed message = live body")

    def test_a_body_edit_after_arming_names_the_first_differing_words(self):
        out = automerge_msg.compare(
            pr(
                armed_title="perf(rocke): add opt-in KDB/VPF (#1111)",
                armed_msg="Line one.\n\nB1 CFVST off (production default).",
            ),
            1111,
        )
        self.assertEqual(
            out[2],
            "STALE message: the armed message differs from the live body from word 3 (armed "
            "'B1 CFVST off (production default).', live 'Line two.'); re-arm to commit the live "
            "body",
        )

    def test_nothing_stored_means_the_merge_takes_the_live_text(self):
        out = automerge_msg.compare(pr(), 1111)
        self.assertEqual(
            out[1:],
            [
                "armed title: none stored (the merge takes the live title)",
                "armed message: none stored (the merge takes the live body)",
            ],
        )

    def test_the_command_line_reads_a_saved_pull(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(pr(armed_title="old (#7)", armed_msg="x"), f)
        try:
            r = subprocess.run(
                [sys.executable, "-I", str(TOOL), "ROCm/r", "7", "--json", f.name],
                capture_output=True,
                text=True,
                timeout=30,
            )
        finally:
            pathlib.Path(f.name).unlink()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("STALE title: armed 'old (#7)'", r.stdout)
        self.assertIn("STALE message", r.stdout)

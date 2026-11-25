# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Whose container a name belongs to, and whose a volume is (rkmon_lib/identity.py; check 15): NONE,
OURS and OTHER in their cases, from docker's answers as docker ps and docker inspect give them, among docker's
warnings; docker text it cannot read, which is never NONE; the question that asks docker for the answers; and the
name filter that lists one name alone.
"""

import json
import os
import pathlib
import re
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import identity  # noqa: E402
from rkmon_lib.identity import NONE, OTHER, OURS  # noqa: E402


def ps_row(project, folder=None, **more):
    """A row as docker ps --all --format json prints it: the labels as one key=value,... text."""
    labels = {"com.docker.compose.service": "rkmon", identity.PROJECT: project}
    if folder is not None:
        labels[identity.WORKING_FOLDER] = folder
    labels.update(more)
    return {
        "Names": f"{project}-rkmon-1",
        "Labels": ",".join(f"{k}={v}" for k, v in labels.items()),
    }


class TestIdentity(unittest.TestCase):
    def setUp(self):
        top = tempfile.TemporaryDirectory()
        self.addCleanup(top.cleanup)
        self.top = pathlib.Path(top.name)
        self.here = self.top / "pkg" / "docker"
        self.there = self.top / "other-pkg" / "docker"
        self.here.mkdir(parents=True)
        self.there.mkdir(parents=True)

    def lines(self, *rows):
        """docker ps's answer: one JSON object per line, among docker's warnings."""
        return "WARNING: a line that is not JSON\n" + "".join(
            json.dumps(r) + "\n" for r in rows
        )

    def test_none_when_no_container_is_the_projects(self):
        self.assertEqual(identity.identity("rocke-monitor", str(self.here), ""), NONE)
        self.assertEqual(identity.identity("rocke-monitor", str(self.here), []), NONE)
        answers = self.lines(ps_row("rocke-monitor-2", str(self.there)))
        self.assertEqual(
            identity.identity("rocke-monitor", str(self.here), answers), NONE
        )

    def test_ours_when_its_working_folder_is_here(self):
        answers = self.lines(
            ps_row("rocke-monitor", str(self.here)),
            ps_row("rocke-monitor-2", str(self.there)),
        )
        self.assertEqual(
            identity.identity("rocke-monitor", str(self.here), answers), OURS
        )

    def test_ours_through_a_link_to_the_same_folder(self):
        link = self.top / "link"
        os.symlink(self.top / "pkg", link)
        answers = self.lines(ps_row("rocke-monitor", str(link / "docker")))
        self.assertEqual(
            identity.identity("rocke-monitor", str(self.here), answers), OURS
        )

    def test_other_when_another_folder_made_it(self):
        answers = self.lines(ps_row("rocke-monitor", str(self.there)))
        self.assertEqual(
            identity.identity("rocke-monitor", str(self.here), answers), OTHER
        )

    def test_other_when_its_folder_is_not_known(self):
        answers = self.lines(ps_row("rocke-monitor"))
        self.assertEqual(
            identity.identity("rocke-monitor", str(self.here), answers), OTHER
        )

    def test_other_when_one_of_the_projects_containers_is_from_another_folder(self):
        answers = [
            ps_row("rocke-monitor", str(self.here)),
            ps_row("rocke-monitor", str(self.there)),
        ]
        self.assertEqual(
            identity.identity("rocke-monitor", str(self.here), answers), OTHER
        )

    def test_docker_inspect_and_a_json_array(self):
        inspect = [
            {
                "Name": "/rocke-monitor-rkmon-1",
                "Config": {
                    "Labels": {
                        identity.PROJECT: "rocke-monitor",
                        identity.WORKING_FOLDER: str(self.here),
                    }
                },
            }
        ]
        self.assertEqual(
            identity.identity(
                "rocke-monitor", str(self.here), json.dumps(inspect, indent=4)
            ),
            OURS,
        )
        self.assertEqual(
            identity.identity("rocke-monitor", str(self.here), inspect), OURS
        )
        self.assertEqual(
            identity.identity(
                "rocke-monitor", str(self.there), json.dumps(inspect).encode()
            ),
            OTHER,
        )

    def test_a_warning_before_an_indented_array(self):
        inspect = [
            {
                "Name": "/rocke-monitor-rkmon-1",
                "Config": {
                    "Labels": {
                        identity.PROJECT: "rocke-monitor",
                        identity.WORKING_FOLDER: str(self.there),
                    }
                },
            }
        ]
        answers = "WARNING: a line that is not JSON\n" + json.dumps(inspect, indent=4)
        self.assertEqual(
            identity.identity("rocke-monitor", str(self.here), answers), OTHER
        )

    def test_text_it_cannot_read_is_never_none(self):
        line = json.dumps(ps_row("rocke-monitor", str(self.there)))
        for answers in (
            "WARNING: a line that is not JSON\n",
            "[WARN] a line that starts as JSON does\n",
            # A JSON line cut short, after one that reads.
            line + "\n" + line[:-5] + "\n",
        ):
            with self.subTest(answers=answers):
                with self.assertRaises(ValueError) as cm:
                    identity.identity("rocke-monitor", str(self.here), answers)
                # It quotes none of docker's text.
                self.assertEqual(str(cm.exception), identity.UNREADABLE)

    def test_empty_text_is_no_container(self):
        for answers in ("", " \n\n", b""):
            with self.subTest(answers=answers):
                self.assertEqual(
                    identity.identity("rocke-monitor", str(self.here), answers), NONE
                )

    def test_the_question_asks_for_the_projects_containers_running_or_not(self):
        self.assertEqual(
            identity.question("rocke-monitor"),
            (
                "ps",
                "--all",
                "--filter",
                "label=com.docker.compose.project=rocke-monitor",
                "--format",
                "{{json .}}",
            ),
        )
        # One JSON object on each line, as that format prints them: the answer identity() reads.
        answers = self.lines(ps_row("rocke-monitor", str(self.here)))
        self.assertEqual(
            identity.identity("rocke-monitor", str(self.here), answers), OURS
        )

    def test_a_name_filter_that_lists_one_name_alone(self):
        """docker reads a name filter as a pattern that may match any part of a name: the filter is anchored at both
        ends, and each character a pattern reads as more than itself is escaped, so it matches its name alone.
        """
        for name, value in (
            ("rkmon-test_rkmon-work", "^rkmon-test_rkmon-work$"),
            ("a.b", "^a\\.b$"),
            ("x+(y)", "^x\\+\\(y\\)$"),
        ):
            with self.subTest(name=name):
                self.assertEqual(identity.exact_name(name), value)
                self.assertRegex(name, value)
        for other in ("rkmon-test_rkmon-work-copy", "old_rkmon-test_rkmon-work"):
            self.assertNotRegex(other, identity.exact_name("rkmon-test_rkmon-work"))
        self.assertNotRegex("aXb", identity.exact_name("a.b"))
        self.assertIsNone(re.search(identity.exact_name("a.b"), "a.b.c"))

    def test_a_comma_inside_a_label_value(self):
        folder = self.top / "a,b"
        folder.mkdir()
        row = ps_row(
            "rocke-monitor", str(folder), **{"com.docker.compose.config-hash": "abc"}
        )
        self.assertEqual(identity.labels(row)[identity.WORKING_FOLDER], str(folder))
        self.assertEqual(identity.identity("rocke-monitor", str(folder), [row]), OURS)

    def volume(self, project=None) -> dict:
        """A volume as docker volume ls --format json prints it: made by the compose project PROJECT, or by none."""
        labels = (
            f"{identity.PROJECT}={project},com.docker.compose.volume=rkmon-work"
            if project
            else ""
        )
        return {
            "Driver": "local",
            "Labels": labels,
            "Name": "rocke-monitor_rkmon-work",
        }

    def test_a_volume_no_other_deployments_container_mounts_is_ours(self):
        """A volume has no folder's label: one the project made, or none did, is the deployment's while only the
        project's containers from this folder mount it, or containers of no compose project; no volume is NONE.
        """
        p, here = "rocke-monitor", str(self.here)
        self.assertEqual(identity.volume_owner(p, here, None, ""), NONE)
        users = self.lines(ps_row(p, here), {"Names": "backup", "Labels": ""})
        for made_by in (p, None):
            with self.subTest(made_by=made_by):
                volume = self.volume(made_by)
                self.assertEqual(identity.volume_owner(p, here, volume, users), OURS)
                self.assertEqual(identity.volume_owner(p, here, volume, ""), OURS)

    def test_a_volume_another_deployment_made_or_mounts_is_others(self):
        """OTHER: the volume's compose project label names another project, or a container of another compose project
        mounts it, or one of this project's from another folder, or of no known folder.
        """
        p, here, there = "rocke-monitor", str(self.here), str(self.there)
        volume = self.volume("rocke-monitor-2")
        self.assertEqual(identity.volume_owner(p, here, volume, ""), OTHER)
        for user in (ps_row("rocke-monitor-2", there), ps_row(p, there), ps_row(p)):
            with self.subTest(user=user["Labels"]):
                users = self.lines(ps_row(p, here), user)
                self.assertEqual(
                    identity.volume_owner(p, here, self.volume(p), users), OTHER
                )


if __name__ == "__main__":
    unittest.main()

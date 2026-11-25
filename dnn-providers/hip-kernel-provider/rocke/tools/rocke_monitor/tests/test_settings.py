# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Settings files (rkmon_lib/settings.py), offline, every value invented: a settings file is known by its header, in
either kind and any layout (settings.header_line, the privacy check's and the deploy path's).
"""

import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rkmon_lib import settings  # noqa: E402

OLD = "rkmon" + "_settings"  # the earliest layout's header


class TestTheHeader(unittest.TestCase):
    def test_the_header_marks_a_settings_file(self):
        for text, line in (
            ("# a comment\n\nkind: deploy_settings\nschema_version: 1\n", 3),
            ('{"kind": "deploy_settings"}', 1),
            ('{"advanced": {}, "kind": "deploy_settings"}', 1),
            # YAML with the header after another top-level key, in either kind or layout
            ("advanced:\n  x: 1\nkind: deploy_settings\n", 3),
            ("# notes\nschema_version: 1\n\nkind: 'build_settings'\nrocm: {}\n", 4),
            ("config: {}\nrkmon_settings: 1\n", 2),
            ("  rocm:\n    kind: x\n  rkmon_build_settings: 1\n", 3),
            # the header nested deeper, or text the YAML subset cannot read (rkmon_lib/answers.py): no settings
            # file
            ("advanced:\n  kind: deploy_settings\n", 0),
            ("A file that shows the header:\n    kind: deploy_settings\n", 0),
            (
                "# Settings\n\nAn example:\n\n```yaml\nschema_version: 1\nkind: deploy_settings\n```\n",
                0,
            ),
            ('KINDS = ("deploy_settings", "build_settings")\nkind = KINDS[0]\n', 0),
            ("x: [unclosed\nkind: deploy_settings\n", 0),
            (f"# the older layout\n{OLD}: 1\nconfig: {{}}\n", 2),
            # a build settings file is a settings file too, in either layout and either form
            ("kind: build_settings\n", 1),
            ("# the older layout\nrkmon_build_settings: 1\n", 2),
            ('{"rocm": {}, "kind": "build_settings"}', 1),
            ('{"rocm": {}, "rkmon_build_settings": 1}', 1),
            ("kind: answers\n", 0),
            ('{"rocm": {}, "kind": "answers"}', 0),
            ("", 0),
            ('{"a": 1}', 0),
        ):
            with self.subTest(text=text):
                self.assertEqual(settings.header_line(text), line)


if __name__ == "__main__":
    unittest.main()

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The package's three file types (rkmon_lib/file_types.py): their kinds, layouts, schema files and commands; the
headers of a settings file of either kind; no schema read on import; each schema loads (rkmon_lib/settings_file.py
checks it) but those NOT_LOADING names; and each key of a schema that loads ends its description with its closing
sentence.
"""

import dataclasses
import importlib
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import file_types, settings_file, steps  # noqa: E402

# Each file type: its kind, its layout, its schema's file, and its template and section commands.
EXPECTED = {
    "DEPLOY": (
        "deploy_settings",
        2,
        "deploy_settings.schema.json",
        "template deploy [--profile PROFILE] FILE",
        "template section KEY",
    ),
    "BUILD": (
        "build_settings",
        2,
        "build_settings.schema.json",
        "template build FILE",
        "",
    ),
    "INTERNAL_STATE": ("internal_state", 1, "internal_state.schema.json", "", ""),
}
# The file types whose schema file does not load: it is in another layout than the file type's, or not there. A
# schema that loads fails test_every_schema_loads until its file type leaves this set.
NOT_LOADING = set()


class TestFileTypes(unittest.TestCase):
    def test_each_kind_layout_and_schema_file(self):
        for name, (kind, version, schema, template, section) in EXPECTED.items():
            with self.subTest(name=name):
                file_type = getattr(file_types, name)
                self.assertEqual((file_type.kind, file_type.version), (kind, version))
                self.assertEqual(
                    pathlib.Path(file_type.schema), ROOT / "schema" / schema
                )
                self.assertEqual(
                    (file_type.template_command, file_type.section_command),
                    (template, section),
                )

    def test_the_headers_of_a_settings_file(self):
        self.assertEqual(file_types.KINDS, ("deploy_settings", "build_settings"))
        self.assertEqual(
            file_types.KINDS, (file_types.DEPLOY.kind, file_types.BUILD.kind)
        )
        self.assertEqual(
            file_types.OLD_HEADERS, ("rkmon_settings", "rkmon_build_settings")
        )

    def test_the_template_commands_are_the_paths_first_steps(self):
        with mock.patch.object(
            steps,
            "DEPLOY_STEPS",
            (
                dataclasses.replace(
                    steps.DEPLOY_STEPS[0], command="new deploy [-x] FILE"
                ),
            ),
        ), mock.patch.object(
            steps,
            "BUILD_STEPS",
            (dataclasses.replace(steps.BUILD_STEPS[0], command="new build FILE"),),
        ):
            importlib.reload(file_types)
            commands = (
                file_types.DEPLOY.template_command,
                file_types.BUILD.template_command,
            )
        importlib.reload(file_types)
        self.assertEqual(commands, ("new deploy [-x] FILE", "new build FILE"))

    def test_every_schema_loads(self):
        for name in EXPECTED:
            file_type = getattr(file_types, name)
            with self.subTest(name=name):
                if name in NOT_LOADING:
                    with self.assertRaises((ValueError, OSError)):
                        file_type.load()
                else:
                    self.assertEqual(
                        file_type.load()["properties"]["kind"]["const"], file_type.kind
                    )

    def test_every_key_ends_with_its_closing_sentence(self):
        for name in sorted(EXPECTED.keys() - NOT_LOADING):
            with self.subTest(name=name):
                schema = getattr(file_types, name).load()
                self.assertEqual(settings_file.closing_problems(schema), [])

    def test_importing_it_reads_no_schema(self):
        read = mock.patch.object(
            pathlib.Path, "read_text", side_effect=AssertionError("a schema was read")
        )
        with read as reader:
            importlib.reload(file_types)
        reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()

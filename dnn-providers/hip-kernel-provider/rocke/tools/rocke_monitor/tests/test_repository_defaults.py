# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rkmon_lib/repository_defaults.py: the repository defaults chosen by the repository and its folder; the
keys they set, which any other repository or folder takes generic; a configuration's value of its own; the words of
the WARN; and what is wrong with a repository's defaults. The package's own read clean, config.defaults.json holds the
generic values of the keys they set, and the doctor and rk config show name them."""

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

from rkmon_lib import (  # noqa: E402
    config,
    config_show,
    doctor,
    paths,
    registry,
    runtime_config,
)
from rkmon_lib import repository_defaults as RD  # noqa: E402
from rkmon_lib.checks import Status  # noqa: E402
import home_repository  # noqa: E402

# An invented repository's defaults: my-org/my-repo's, for its folder projects/mine.
INVENTED = ROOT / "tests" / "data" / "repository"
INVENTED_KEYS = ["ci_absent_modules", "develop_rocke_job_pattern"]
# The keys that belong to one repository rather than to the monitor, which the home repository's defaults set.
HOME_KEYS = {
    "test_sparse",
    "hot_spots",
    "ci_test_tiers",
    "ci_absent_modules",
    "ci_infra_signatures",
    "drift_extract",
    "develop_watched_workflows",
    "develop_rocke_job_pattern",
    "incident_issue.time_limit_labels",
    "incident_issue.title_pattern",
    "incident_issue.test_name_pattern",
    "incident_issue.umbrella_test_names",
    "gpu.skip_pattern",
    "gpu.base_env",
}


class Folder(unittest.TestCase):
    """The repository defaults under FOLDER: the invented ones, or those a test writes."""

    def use(self, folder) -> None:
        patch = mock.patch.object(RD, "FOLDER", pathlib.Path(folder))
        patch.start()
        self.addCleanup(patch.stop)

    def write(self, name: str, data) -> pathlib.Path:
        """The folder NAME under a temporary FOLDER, holding DATA as its defaults (text as it is); None: no file."""
        if not hasattr(self, "tmp"):
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            self.tmp = pathlib.Path(tmp.name)
            self.use(self.tmp)
        folder = self.tmp / name
        folder.mkdir()
        if data is not None:
            text = data if isinstance(data, str) else json.dumps(data)
            (folder / RD.FILE).write_text(text, encoding="utf-8")
        return folder / RD.FILE


class TestChoosing(Folder):
    def test_by_the_repository_whatever_its_case_and_by_one_of_their_folders(self):
        self.use(INVENTED)
        found = RD.find("my-org/my-repo", "projects/mine")
        self.assertEqual(
            found,
            RD.Found(
                INVENTED / "my-org__my-repo" / RD.FILE,
                "my-org/my-repo",
                ("projects/mine",),
                {"ci_absent_modules": ["torch"], "develop_rocke_job_pattern": "mine"},
            ),
        )
        self.assertEqual(
            RD.find("My-Org/MY-REPO", "/projects/mine/"), found
        )  # a slash at either end is no part
        for repository, folder in (
            ("my-org/my-repo", "projects/other"),
            ("my-org/my-repo", "Projects/mine"),  # a folder's case counts, as in git
            ("my-org/my-repo", "projects"),
            ("my-org/other", "projects/mine"),
            ("", "projects/mine"),
            ("my-org/my-repo", ""),
            (None, None),
        ):
            with self.subTest(repository=repository, folder=folder):
                self.assertIsNone(RD.find(repository, folder))

    def test_the_keys_they_set(self):
        self.use(INVENTED)
        self.assertEqual(RD.keys(), INVENTED_KEYS)
        values = {
            "gpu": {"base_env": {"A": "1"}, "skip_pattern": "x"},
            "repo": "o/r",
            "no_such_key": 1,
        }
        self.assertEqual(
            RD.keys_of(values), ["gpu.base_env", "gpu.skip_pattern"]
        )  # package keys alone
        self.assertEqual(RD.keys_of({}), [])

    def test_a_value_of_its_own_is_neither_blank_nor_the_generic_one(self):
        for value, generic, own in (
            (registry.MISSING, "x", False),
            ("", "x", False),
            (None, "x", False),
            ([], ["x"], False),
            ({}, {"x": 1}, False),
            ("x", "x", False),
            (["a"], ["a"], False),
            ("y", "x", True),
            (["a"], [], True),
            (False, True, True),
            (0, 5, True),
        ):
            with self.subTest(value=value, generic=generic):
                self.assertIs(RD.own(value, generic), own)

    def test_another_repository_or_folder_takes_their_keys_generic(self):
        self.use(INVENTED)
        self.assertEqual(RD.generic("my-org/my-repo", "projects/mine"), [])
        self.assertEqual(RD.generic("my-org/my-repo", "projects/other"), INVENTED_KEYS)
        self.assertEqual(RD.generic("acme/widgets", "a/b"), INVENTED_KEYS)
        # A running configuration that gives a key a value of its own takes it generic no longer.
        own = {"ci_absent_modules": ["numpy"], "develop_rocke_job_pattern": ""}
        self.assertEqual(
            RD.generic("acme/widgets", "a/b", own), ["develop_rocke_job_pattern"]
        )
        generic = {"ci_absent_modules": [], "develop_rocke_job_pattern": ""}
        self.assertEqual(RD.generic("acme/widgets", "a/b", generic), INVENTED_KEYS)
        self.assertEqual(
            RD.generic_found(
                "acme/widgets", "/a/b/", RD.generic("acme/widgets", "a/b")
            ),
            "none for acme/widgets and its folder a/b: the package's generic values apply to ci_absent_modules and "
            "develop_rocke_job_pattern",
        )


class TestProblems(Folder):
    def test_what_is_wrong_with_a_repositorys_defaults_and_none_of_it_applies(self):
        good = {"folders": ["f"], "values": {"ci_absent_modules": ["torch"]}}
        unreadable = self.write("o__a", "{")
        no_form = self.write("o__b", {"folders": ["f"]})
        no_folders = self.write("o__c", {"folders": [], "values": {}})
        blank_folder = self.write("o__d", {"folders": ["/"], "values": {}})
        no_values = self.write("o__e", {"folders": ["f"], "values": []})
        stray = self.write(
            "o__f", {"folders": ["f"], "values": {"repo": "x/y", "no_such_key": 1}}
        )
        self.write("o__g", None)
        self.write("o__ok", good)
        self.write(
            "no-separator", {"folders": ["f"], "values": {"no_such_key": 1}}
        )  # no repository's folder
        problems = RD.problems()
        self.assertEqual(len(problems), 8, problems)
        self.assertIn(f"{unreadable} cannot be read: ", problems[0])
        self.assertEqual(
            problems[1:],
            [
                f'{no_form} is not of the form {{"folders": [...], "values": {{...}}}}',
                f"{no_folders}: folders must list the repository's folders they were written for",
                f"{blank_folder}: folders must list the repository's folders they were written for",
                f"{no_values}: values must hold package keys, nested",
                f"{stray}: repo is no package key: repository defaults set package keys only",
                f"{stray}: no_such_key is no package key: repository defaults set package keys only",
                f"{self.tmp / 'o__g'} has no {RD.FILE}",
            ],
        )
        for name in "abcdefg":
            self.assertIsNone(RD.find(f"o/{name}", "f"))
        self.assertEqual(RD.find("o/ok", "f").values, good["values"])
        self.assertEqual(RD.keys(), ["ci_absent_modules"])


class TestThePackagesOwn(unittest.TestCase):
    def test_they_read_clean_and_set_the_keys_25_moves(self):
        self.assertEqual(RD.problems(), [])
        found = RD.find(home_repository.REPOSITORY, home_repository.FOLDER)
        self.assertEqual(
            found.path, paths.PACKAGE / "repository" / "ROCm__rocm-libraries" / RD.FILE
        )
        self.assertEqual(set(RD.keys_of(found.values)), HOME_KEYS)
        self.assertEqual(set(RD.keys()), HOME_KEYS)

    def test_config_defaults_holds_the_generic_values_of_their_keys(self):
        """Blank, but the GPU tests' skip pattern, whose generic value names a GPU or a device."""
        generic = config.load_defaults()
        for name in sorted(HOME_KEYS - {"gpu.skip_pattern"}):
            with self.subTest(key=name):
                self.assertIn(registry.get(generic, name), ("", [], {}))
        pattern = registry.get(generic, "gpu.skip_pattern")
        self.assertRegex("needs a GPU", pattern)
        self.assertNotRegex("needs gfx942", pattern)


class TestShown(Folder):
    def setUp(self):
        self.use(INVENTED)
        patch = mock.patch.object(doctor, "deployment_image", return_value=False)
        patch.start()
        self.addCleanup(patch.stop)

    def test_the_doctor_names_those_that_apply_or_what_is_generic(self):
        name = "repository defaults"
        mine = {"repo": "my-org/my-repo", "rocke_path": "projects/mine"}
        self.assertEqual(
            [
                (c.name, c.status, c.found)
                for c in doctor.repository_defaults_checks(mine)
            ],
            [
                (
                    name,
                    Status.PASS,
                    "tests/data/repository/my-org__my-repo/defaults.json, for my-org/my-repo and its folder "
                    "projects/mine",
                )
            ],
        )
        other = dict(mine, rocke_path="projects/other")
        (row,) = doctor.repository_defaults_checks(other)
        self.assertEqual(
            (row.name, row.status, row.found, str(row.fix)),
            (
                name,
                Status.WARN,
                RD.generic_found("my-org/my-repo", "projects/other", INVENTED_KEYS),
                "rkmon rk config show --source shows each value the monitor runs with and where it comes from",
            ),
        )
        own = dict(
            other, ci_absent_modules=["numpy"], develop_rocke_job_pattern="other"
        )
        self.assertEqual(
            [(c.status, c.found) for c in doctor.repository_defaults_checks(own)],
            [
                (
                    Status.PASS,
                    "none for my-org/my-repo and its folder projects/other, and the configuration gives each of their "
                    "keys a value",
                )
            ],
        )
        self.assertEqual(doctor.repository_defaults_checks(dict(mine, repo="")), [])

    def test_config_show_names_the_repository_layer(self):
        settings = {"repository": {"name": "my-org/my-repo", "folder": "projects/mine"}}
        rows = {
            n: (v, s)
            for n, v, s in config_show.rows(runtime_config.from_settings(settings))
        }
        self.assertEqual(
            rows["ci_absent_modules"], (["torch"], runtime_config.REPOSITORY)
        )
        self.assertEqual(
            rows["develop_rocke_job_pattern"], ("mine", runtime_config.REPOSITORY)
        )
        settings["repository"]["folder"] = "projects/other"
        rows = {
            n: (v, s)
            for n, v, s in config_show.rows(runtime_config.from_settings(settings))
        }
        self.assertEqual(rows["ci_absent_modules"], ([], runtime_config.PACKAGE_SOURCE))


if __name__ == "__main__":
    unittest.main()

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rkmon migrate --settings FILE [--dry-run]` (rkmon_lib/migrate.py), on invented files of the earlier
layout in a home folder of the test's own: the whole output for a small file; the file it writes (the one writer's
text, read back in layout 2, with no date and no history line, an off feature as its switch alone unless a key under
it is set) and the earlier file kept as FILE.v1, yours only, as it was; --dry-run; a file in this layout; a whole
deploy file, every key fitting the schema, and a team dashboard that is on, which check takes with either delivery;
the two settings that stop it and every other refusal, each with nothing written and exit 4, and the row every reader
refuses a file in the layout before layout 1 with; a value its key does not take, carried and reported, a value
written as text, the update source written as the updates section, a value the writer cannot write, a ROCm pin the
package's build does not give, a file that changes while migrate runs, and a read or a write that fails; the profile
the title line names; no network; the step from layout 1, each file type's own, opening no package file
but layout 2's schemas and layout 1's frozen defaults; the command's row; and docker/rkmon migrate, which runs on the
host with no docker command.
"""

import contextlib
import copy
import dataclasses
import errno
import io
import os
import pathlib
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
WRAPPER = ROOT / "docker" / "rkmon"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import sandbox  # noqa: E402
import test_from_layout1 as layout1  # noqa: E402  (its invented files of layout 1; its own tests run in their module)
from rkmon_lib import (  # noqa: E402
    build_plan,
    checks,
    commands,
    file_types,
    from_layout1,
    migrate,
    private_files,
    schema,
    settings_file,
    wrapping,
)
from rkmon_lib.paths import PACKAGE  # noqa: E402
from rkmon_lib.settings_file import Document  # noqa: E402

DEPLOY, BUILD = file_types.DEPLOY, file_types.BUILD
RA, MQ = layout1.RA, layout1.MQ
GIVEN = "~/rkmon-deploy.yaml"  # the file, as you typed it
BANNER = "== migrate: rewrite a settings file in this package's layout"
# The fix and the next line for a file migrate cannot read, as it has no header.
NEW_FILE = [
    "fix: start from a new file, rkmon template deploy NEW_FILE (a build file: rkmon template build NEW_FILE), and "
    "copy",
    "     your values into it",
    "next: rkmon check --settings NEW_FILE",
]
AGAIN = "next: rkmon migrate --settings ~/rkmon-deploy.yaml"
HEAD = "kind: deploy_settings\nschema_version: 1\n"


def with_profiles(profiles) -> settings_file.FileType:
    """The deploy settings, with PROFILES in its schema in place of the package's profiles."""
    root = copy.deepcopy(DEPLOY.load())
    root["x-profiles"] = copy.deepcopy(profiles)
    return dataclasses.replace(DEPLOY, schema=root)


SMALL = """\
# My own notes: the monitor of my team's repository.
kind: deploy_settings
schema_version: 1
container:
  name: my-monitor
repository:
  name: my-org/my-repo
  path: projects/my-project
monitor:
  tick_min: 5
features:
  merge_queue:
    enabled: false
"""
SMALL_REPORT = [
    "~/rkmon-deploy.yaml: deploy settings, from layout 1 to layout 2",
    '  image.tag: "my-monitor", the container\'s name, what a blank tag meant;',
    "    rkmon confirm says whether that image suits this package",
    "  repository.path is now repository.folder",
    "  monitor.tick_min is now monitor.tick_interval_minutes",
    '  the title line names the profile "review-only" (reviews and the admin dashboard), which starts with',
    "    features.review_automation.enabled true and features.merge_queue.enabled false, as your file has them",
]
MIGRATED = [
    "MIGRATED: ~/rkmon-deploy.yaml is in layout 2 (the earlier file is kept as ~/rkmon-deploy.yaml.v1)",
    "next: rkmon check --settings ~/rkmon-deploy.yaml",
]
# A dry run's outcome on a file migrate can write, wrapped as every outcome line is.
CAN_WRITE = [
    "PASSED: migrate can write ~/rkmon-deploy.yaml in layout 2 and keep the earlier file as ~/rkmon-deploy.yaml.v1;",
    "nothing changed",
]
SECRETS = (
    HEAD
    + "secret_values:\n  github_token: one\n  teams_webhook: two\n  jira_token: three\n"
)
ADMIN = HEAD + "features:\n  dashboards:\n    admin:\n      bind: 0.0.0.0\n"
# The develop guard, off, as the file migrate writes has it: its switch alone, or, with a value of its own,
# in full, so that no value is lost.
GUARD_TITLE = """\
    # Develop guard: after every merge, the monitor checks develop and traces any break to the pull request that
    # caused it. In live, each break becomes a GitHub issue, and an open pull request that fixes it is linked to the
    # issue. Without review automation, breaks wait for you on the admin dashboard instead.
    develop_guard:
"""
GUARD_ALONE = (
    GUARD_TITLE
    + """\
      # true: check develop. false: don't. To turn it on, see QUICK_USER_GUIDE.md, "Turn on a feature". Required when
      # the merge queue is on.
      enabled: false
"""
)
GUARD_IN_FULL = (
    GUARD_TITLE
    + """\
      # true: check develop. false: don't. Required when the merge queue is on.
      enabled: false
      # The repository to file develop issues in, as <org>/<repo>. Leave blank to use repository.name.
      issue_repository: "my-org/issues"
      # The label for an issue about a break that turns develop's CI red. The label must already exist in
      # issue_repository (rkmon rk label-create NAME --description TEXT --color HEX creates one). Example: "ci-red".
      # Required when the develop guard is on and the merge queue is live.
      ci_red_label: ""
      # More labels for those issues. Each must already exist. Default: none.
      ci_red_extra_labels: []
      # The label for an issue about a break that CI doesn't catch; the issue is assigned to the author of the pull
      # request that caused it. The label must already exist in issue_repository. Example: "stability". Required when
      # the develop guard is on and the merge queue is live.
      stability_label: ""
"""
)


class Migrate(unittest.TestCase):
    """A home folder of the test's own, the package's file types and their profiles."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = pathlib.Path(os.path.realpath(tmp.name)) / "home"
        self.home.mkdir()
        patch = mock.patch.dict(os.environ, {"HOME": str(self.home)})
        patch.start()
        self.addCleanup(patch.stop)
        self.path = self.home / "rkmon-deploy.yaml"
        self.kept = self.home / "rkmon-deploy.yaml.v1"

    def migrate(
        self,
        text=None,
        dry_run=False,
        mode=0o600,
        path=None,
        given=GIVEN,
        argv=("rkmon", "migrate", "--settings"),
    ) -> tuple:
        """(exit code, output lines) of migrate on the file at PATH (the deploy file), TEXT written there first at
        MODE; GIVEN is the file as you typed it, after the words ARGV."""
        path = path or self.path
        if text is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            path.chmod(mode)
        out = io.StringIO()
        rc = migrate.run((*argv, given), given, str(path), dry_run=dry_run, out=out)
        return rc, out.getvalue().splitlines()

    def files(self) -> dict:
        """Each file under the home folder, by its path there: (its bytes, its permissions)."""
        return {
            str(p.relative_to(self.home)): (
                p.read_bytes(),
                stat.S_IMODE(p.stat().st_mode),
            )
            for p in sorted(self.home.rglob("*"))
            if p.is_file()
        }

    def assertNothingWritten(self, text, mode=0o600, name="rkmon-deploy.yaml"):
        self.assertEqual(self.files(), {name: (text.encode(), mode)})

    def assertNoted(self, lines: list, note: str) -> None:
        """NOTE is a line of the report in LINES, wrapped as the report wraps it."""
        wrapped = wrapping.wrap(note, checks.WIDTH, "  ", "    ")
        self.assertIn(wrapped[0], lines)
        start = lines.index(wrapped[0])
        self.assertEqual(lines[start : start + len(wrapped)], wrapped)


class TestMigrate(Migrate):
    def test_the_whole_output_for_a_small_file(self):
        self.assertEqual(self.migrate(SMALL), (0, [BANNER, *SMALL_REPORT, *MIGRATED]))

    def test_the_file_written_is_the_one_writers_text_with_its_profile(self):
        self.migrate(SMALL)
        written = self.path.read_text(encoding="utf-8")
        doc = settings_file.upgrade(settings_file.read(self.kept, DEPLOY, older=True))[
            0
        ]
        self.assertEqual(
            written, settings_file.dump(dataclasses.replace(doc, profile="review-only"))
        )
        self.assertEqual(
            written.splitlines()[0],
            '# rocKE monitor: deploy settings (profile "review-only": reviews and the admin dashboard).',
        )
        back = settings_file.read(self.path, DEPLOY)
        self.assertEqual((back.version, back.profile), (2, "review-only"))
        self.assertEqual(settings_file.dump(back), written)
        # No date and no history line: nothing says when it moved, or from what.
        self.assertIsNone(
            re.search(r"\d{4}-\d\d-\d\d|layout 1|\.v1|My own notes", written)
        )

    def test_the_earlier_file_is_kept_as_it_was_yours_only(self):
        self.path.write_text(SMALL, encoding="utf-8")
        self.path.chmod(0o400)  # readable by you only: a private file
        self.migrate()
        self.assertEqual(
            self.kept.read_bytes(), SMALL.encode()
        )  # your own comments stay there
        self.assertEqual(
            {k: mode for k, (_, mode) in self.files().items()},
            {"rkmon-deploy.yaml": 0o600, "rkmon-deploy.yaml.v1": 0o600},
        )

    def test_an_off_feature_is_its_switch_alone_unless_a_key_under_it_is_set(self):
        self.migrate(
            SMALL + "  review_automation:\n    enabled: false\n    mode: live\n"
        )
        lines = [
            line
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")
        ]

        def section(key: str) -> list:
            """The lines under the feature KEY, without their indent."""
            start = lines.index(f"  {key}:")
            under = []
            for line in lines[start + 1 :]:
                if len(line) - len(line.lstrip()) <= 2:
                    break
                under.append(line.strip())
            return under

        self.assertEqual(section("merge_queue"), ["enabled: false"])
        automation = section("review_automation")
        self.assertEqual(automation[:2], ["enabled: false", 'mode: "live"'])
        self.assertIn("orchestrator:", automation)

    def test_dry_run_prints_the_report_and_the_new_file_and_writes_nothing(self):
        rc, lines = self.migrate(SMALL, dry_run=True)
        self.assertNothingWritten(SMALL)
        self.assertEqual(self.migrate(SMALL)[0], 0)
        written = self.path.read_text(encoding="utf-8")
        self.assertEqual(
            (rc, lines),
            (
                0,
                [
                    BANNER + " (dry run: nothing changes)",
                    *SMALL_REPORT,
                    "",
                    *written.splitlines(),
                    "",
                    *CAN_WRITE,
                    AGAIN,
                ],
            ),
        )

    def test_a_file_in_this_layout_is_nothing_to_do(self):
        self.migrate(SMALL)
        files = self.files()
        self.assertEqual(
            self.migrate(),
            (
                0,
                [
                    BANNER,
                    "NOTHING TO DO: ~/rkmon-deploy.yaml is in layout 2 already",
                    "next: rkmon check --settings ~/rkmon-deploy.yaml",
                ],
            ),
        )
        self.assertEqual(self.files(), files)

    def test_a_build_file(self):
        build = self.home / "rkmon-build.yaml"
        rc, lines = self.migrate(
            layout1.FULL_BUILD, path=build, given="~/rkmon-build.yaml"
        )
        notes = settings_file.upgrade(
            settings_file.read(self.home / "rkmon-build.yaml.v1", BUILD, older=True)
        )[1]
        report = ["~/rkmon-build.yaml: build settings, from layout 1 to layout 2"]
        for note in [
            *notes,
            "the title line names no profile, as build settings have no profiles",
        ]:
            report += wrapping.wrap(note, checks.WIDTH, "  ", "    ")
        self.assertEqual(
            (rc, lines),
            (
                0,
                [
                    BANNER,
                    *report,
                    "MIGRATED: ~/rkmon-build.yaml is in layout 2 (the earlier file is kept as ~/rkmon-build.yaml.v1)",
                    "next: rkmon check --settings ~/rkmon-build.yaml",
                ],
            ),
        )
        back = settings_file.read(build, BUILD)
        self.assertEqual((back.version, back.profile), (2, None))
        self.assertEqual(
            build.read_text(encoding="utf-8").splitlines()[0],
            "# rocKE monitor: build settings.",
        )
        self.assertEqual(
            self.files()["rkmon-build.yaml.v1"], (layout1.FULL_BUILD.encode(), 0o600)
        )

    def test_a_whole_deploy_file_of_the_earlier_layout(self):
        rc, lines = self.migrate(layout1.FULL_DEPLOY)
        self.assertEqual((rc, lines[-2:]), (0, MIGRATED))
        back = settings_file.read(self.path, DEPLOY)
        self.assertEqual((back.version, back.profile), (2, "full"))
        # Every key fits the schema, none is missing and no <...> is left; the checks across keys are the next test's.
        rows = settings_file.validate(back)
        self.assertEqual(
            (rows[0].name, rows[0].status), ("settings", checks.Status.PASS)
        )

    def test_a_team_dashboard_on_is_written_with_its_deliverys_block_alone(self):
        """The block that delivery does not take is written commented out, so check 7 takes the file."""
        for delivery in ("serve", "publish"):
            with self.subTest(delivery=delivery):
                for p in (self.path, self.kept):
                    p.unlink(missing_ok=True)
                self.migrate(
                    layout1.FULL_DEPLOY.replace(
                        "delivery: serve", f"delivery: {delivery}"
                    )
                )
                back = settings_file.read(self.path, DEPLOY)
                self.assertEqual(
                    [
                        r.name
                        for r in settings_file.validate(back)
                        if r.status == checks.Status.FAIL
                    ],
                    [],
                )

    def test_it_uses_no_network(self):
        def refused(*args, **kwargs):
            raise AssertionError("migrate reached for the network")

        with mock.patch.object(socket, "socket", refused), mock.patch.object(
            socket, "create_connection", refused
        ), mock.patch.object(socket, "getaddrinfo", refused):
            self.assertEqual(self.migrate(layout1.FULL_DEPLOY)[0], 0)


class TestRefusals(Migrate):
    """Each refusal writes nothing and exits 4; its next line runs migrate again."""

    def test_the_two_settings_that_stop_it(self):
        # Each refusal, the outcome line wrapped as the fix is.
        for text, refusal in (
            (
                SECRETS,
                [
                    "REFUSED: ~/rkmon-deploy.yaml has a secret_values section (3 entries), and no secret goes in a "
                    "settings file; nothing",
                    "changed",
                    "fix: move each value into a file only you can read, delete secret_values from "
                    "~/rkmon-deploy.yaml, and run migrate",
                    "     again; after rkmon confirm, rkmon login lists each secret and installs it",
                    AGAIN,
                ],
            ),
            (
                ADMIN,
                [
                    'REFUSED: features.dashboards.admin.bind is "0.0.0.0", and the admin dashboard listens on this '
                    "host's 127.0.0.1 only;",
                    "nothing changed",
                    'fix: delete features.dashboards.admin.bind from ~/rkmon-deploy.yaml; README.md, "The admin '
                    'dashboard", says how to',
                    "     reach it from another machine",
                    AGAIN,
                ],
            ),
        ):
            with self.subTest(refusal=refusal[0]):
                self.assertEqual(self.migrate(text), (4, [BANNER, *refusal]))
                self.assertNothingWritten(text)

    def test_the_files_own_checks_come_first(self):
        # Its row shows the file as migrate's own lines do; the refusal shows itself, as every private file's
        # does: its row, then the line that names the rules.
        self.assertEqual(
            self.migrate(SECRETS, mode=0o644),
            (
                4,
                [
                    BANNER,
                    "  FAIL  settings file: ~/rkmon-deploy.yaml has permissions 644: others on this machine can read "
                    "it",
                    "        fix: chmod 600 ~/rkmon-deploy.yaml",
                    private_files.RULES,
                    "REFUSED: ~/rkmon-deploy.yaml is not migrated; nothing changed (fix each FAIL above)",
                    AGAIN,
                ],
            ),
        )
        self.assertNothingWritten(SECRETS, 0o644)

    def test_several_rows_are_printed_before_the_outcome(self):
        # A file in a cloud-synced folder (Box's), that others can read.
        cloud = self.home / "Box" / "rkmon-deploy.yaml"
        given = "~/Box/rkmon-deploy.yaml"
        rc, lines = self.migrate(SMALL, mode=0o644, path=cloud, given=given)
        retry = private_files.again(("rkmon", "migrate", "--settings", given), given)
        rows = [
            r
            for r in private_files.problems(
                cloud, PACKAGE, migrate.NAME, retry, shown=given
            )
            if r.status == checks.Status.FAIL
        ]
        self.assertEqual(len(rows), 2)  # its permissions and its place
        self.assertEqual(
            (rc, lines),
            (
                4,
                [
                    BANNER,
                    *checks.render(rows, "problems").splitlines(),
                    private_files.RULES,
                    "REFUSED: ~/Box/rkmon-deploy.yaml is not migrated; nothing changed (fix each FAIL above)",
                    "next: rkmon migrate --settings ~/Box/rkmon-deploy.yaml",
                ],
            ),
        )
        # The fix that names another file is the command you ran, naming that file.
        self.assertIn(
            "then rkmon migrate --settings ~/rkmon-deploy.yaml",
            " ".join(" ".join(lines).split()),
        )
        self.assertEqual(
            self.files(), {"Box/rkmon-deploy.yaml": (SMALL.encode(), 0o644)}
        )

    def test_a_file_of_neither_kind(self):
        text = "kind: internal_state\nschema_version: 1\n"
        self.assertEqual(
            self.migrate(text),
            (
                4,
                [
                    BANNER,
                    "REFUSED: ~/rkmon-deploy.yaml is not a deploy settings or build settings file: its kind line names "
                    "another kind;",
                    "nothing changed",
                    "fix: name your settings file instead: rkmon migrate --settings FILE",
                    AGAIN,
                ],
            ),
        )
        self.assertNothingWritten(text)

    def test_a_newer_layout(self):
        # The fix and the next line are state 4's, the package that wrote the file: migrate again would refuse
        # it again.
        package = "rkmon package update --from DIR, with DIR the package that wrote your file, or a newer one"
        for kind in ("deploy_settings", "build_settings"):
            text = f"kind: {kind}\nschema_version: 3\n"
            with self.subTest(kind=kind):
                self.assertEqual(
                    self.migrate(text),
                    (
                        4,
                        [
                            BANNER,
                            "REFUSED: ~/rkmon-deploy.yaml is in layout 3, newer than this package's layout 2; nothing "
                            "changed",
                            f"fix: {package}",
                            f"next: {package}",
                        ],
                    ),
                )
                self.assertNothingWritten(text)

    def test_a_file_in_the_layout_before_layout_1(self):
        # The fix starts a new file of either kind, and the next line checks it.
        before = "REFUSED: ~/rkmon-deploy.yaml is in the layout before layout 1 (it starts with"
        settings = [
            f"{before} rkmon_settings), which this package",
            "does not read; nothing changed",
        ]
        build = [
            f"{before} rkmon_build_settings), which this",
            "package does not read; nothing changed",
        ]
        for text, refusal in (
            ("rkmon_settings: 1\nconfig:\n  repo: <org>/<repo>\n", settings),
            ('rkmon_build_settings: 1\nimage:\n  tag: "1.0"\n', build),
            ('{"rkmon_settings": 1, "config": {}}\n', settings),
        ):
            with self.subTest(text=text):
                self.assertEqual(self.migrate(text), (4, [BANNER, *refusal, *NEW_FILE]))
                self.assertNothingWritten(text)

    def test_the_row_every_reader_refuses_a_file_before_layout_1_with(self):
        """before_layout_1: the row for a file whose first key is one of file_types.OLD_HEADERS, named NAME unless the
        reader names its rows otherwise; None for any other first key."""
        fix = (
            "start from a new file, rkmon template deploy NEW_FILE (a build file: rkmon template build NEW_FILE), "
            "and copy your values into it"
        )
        for first in file_types.OLD_HEADERS:
            with self.subTest(first=first):
                row = migrate.before_layout_1(first, "~/f.yaml")
                self.assertEqual(
                    (row.name, row.status, row.found, str(row.fix)),
                    (
                        migrate.NAME,
                        checks.Status.FAIL,
                        f"~/f.yaml is in the layout before layout 1 (it starts with {first}), which this package "
                        "does not read",
                        fix,
                    ),
                )
                self.assertEqual(
                    migrate.before_layout_1(first, "~/f.yaml", "deploy settings file"),
                    dataclasses.replace(row, name="deploy settings file"),
                )
        for first in (None, "", "kind", "repository"):
            with self.subTest(first=first):
                self.assertIsNone(migrate.before_layout_1(first, "~/f.yaml"))

    def test_a_file_with_no_header(self):
        # Without a kind line, it names both kinds as a file in the layout before layout 1 does.
        for text in ("repository:\n  name: <org>/<repo>\n", "{}\n"):
            with self.subTest(text=text):
                self.assertEqual(
                    self.migrate(text),
                    (
                        4,
                        [
                            BANNER,
                            "REFUSED: ~/rkmon-deploy.yaml has no kind line, so it is not a deploy settings or build "
                            "settings file; nothing changed",
                            *NEW_FILE,
                        ],
                    ),
                )
                self.assertNothingWritten(text)
        # With a kind line but no layout, a new file of its kind.
        for kind in ("deploy", "build"):
            text = f"kind: {kind}_settings\n"
            with self.subTest(text=text):
                self.assertEqual(
                    self.migrate(text),
                    (
                        4,
                        [
                            BANNER,
                            "REFUSED: ~/rkmon-deploy.yaml has no schema_version line with a whole number, so its "
                            "layout is unknown; nothing",
                            "changed",
                            f"fix: start from a new file, rkmon template {kind} NEW_FILE, and copy your values into it",
                            NEW_FILE[-1],
                        ],
                    ),
                )
                self.assertNothingWritten(text)

    def test_a_kept_place_that_holds_another_file(self):
        self.kept.write_text("another file\n", encoding="utf-8")
        self.kept.chmod(0o600)
        # The command as you typed it, from your home folder.
        rc, lines = self.migrate(
            SMALL, given="rkmon-deploy.yaml", argv=("./rkmon", "migrate", "--settings")
        )
        self.assertEqual(
            (rc, lines[-6:]),
            (
                4,
                [
                    "  FAIL  deploy settings file: ~/rkmon-deploy.yaml.v1 is another file, and a kept file is never "
                    "written over",
                    "        fix: mv ~/rkmon-deploy.yaml.v1 ~/rkmon-deploy.yaml.v1.old, then",
                    "             ./rkmon migrate --settings ~/rkmon-deploy.yaml",
                    private_files.RULES,
                    "REFUSED: ~/rkmon-deploy.yaml is not migrated; nothing changed (fix each FAIL above)",
                    AGAIN,
                ],
            ),
        )
        self.assertEqual(
            self.files(),
            {
                "rkmon-deploy.yaml": (SMALL.encode(), 0o600),
                "rkmon-deploy.yaml.v1": (b"another file\n", 0o600),
            },
        )

    def test_a_kept_file_a_stopped_migrate_left_stays(self):
        self.kept.write_text(SMALL, encoding="utf-8")
        self.kept.chmod(0o644)
        self.assertEqual(self.migrate(SMALL), (0, [BANNER, *SMALL_REPORT, *MIGRATED]))
        self.assertEqual(self.files()["rkmon-deploy.yaml.v1"], (SMALL.encode(), 0o600))


class TestValues(Migrate):
    def test_a_value_its_key_does_not_take_is_carried_and_reported(self):
        # Text where a number goes, a time limit out of its key's form (minutes and seconds), and an id past its bound.
        rc, lines = self.migrate(
            HEAD
            + "host:\n  users:\n    monitor_uid: 3000000000\nmonitor:\n  tick_min: abc\n"
            + 'advanced:\n  gpu:\n    time_limit: "30:00"\n'
        )
        self.assertEqual((rc, lines[-2:]), (0, MIGRATED))
        carried = "; carried as it is, and check fails it with its fix"
        for note in (
            f"host.users.monitor_uid: 3000000000 is more than 2147483647{carried}",
            f'monitor.tick_interval_minutes: "abc" is not a whole number{carried}',
            f'{RA}.remote_gpu_tests.cluster.time_limit: "30:00" is not in the form this key takes{carried}',
        ):
            with self.subTest(note=note):
                self.assertNoted(lines, note)
        self.assertNotIn("written as the text", "\n".join(lines))
        back = settings_file.read(self.path, DEPLOY)
        self.assertEqual(
            (
                back.values["host"]["users"]["monitor_uid"],
                back.values["monitor"]["tick_interval_minutes"],
                back.values["features"]["review_automation"]["remote_gpu_tests"][
                    "cluster"
                ]["time_limit"],
            ),
            (3000000000, "abc", "30:00"),
        )
        self.assertEqual(
            sorted(r.name for r in settings_file.misfits(back)),
            [
                f"{RA}.remote_gpu_tests.cluster.time_limit",
                "host.users.monitor_uid",
                "monitor.tick_interval_minutes",
            ],
        )

    def test_a_value_its_key_takes_as_text_is_written_as_the_text_it_is(self):
        rc, lines = self.migrate(HEAD + "container:\n  image:\n    tag: 1.10\n")
        self.assertEqual((rc, lines[-2:]), (0, MIGRATED))
        self.assertNoted(lines, 'image.tag: 1.10 is written as the text "1.10"')
        back = settings_file.read(self.path, DEPLOY)
        self.assertEqual(
            (back.values["image"]["tag"], settings_file.misfits(back)), ("1.10", [])
        )

    def test_the_update_source_is_written_as_the_updates_section(self):
        # Where rkmon package update fetches newer code, as the file migrate writes has it.
        rc, lines = self.migrate(
            HEAD
            + "advanced:\n  update:\n    repo: https://github.com/ROCm/my-repo.git\n    branch: develop\n"
            + "    path: tools/monitor\n"
        )
        self.assertEqual((rc, lines[-2:]), (0, MIGRATED))
        for note in (
            'advanced.update.repo is now updates.repository "ROCm/my-repo"',
            "advanced.update.branch is now updates.branch",
            'advanced.update.path "tools/monitor": dropped, as the package knows its own folder in the repository it '
            "comes from",
        ):
            with self.subTest(note=note):
                self.assertNoted(lines, note)
        back = settings_file.read(self.path, DEPLOY)
        self.assertEqual(
            (back.values["updates"], settings_file.misfits(back)),
            ({"repository": "ROCm/my-repo", "branch": "develop"}, []),
        )

    def test_a_false_or_blank_requests_file_is_no_priority_requests(self):
        # Layout 1 read a false or blank advanced.publish.html.requests as no requests file, so the box stays off.
        team = (
            HEAD
            + "features:\n  dashboards:\n    team:\n      enabled: true\n      delivery: publish\n"
            + "advanced:\n  publish:\n    html:\n      requests: {}\n"
        )
        enabled = f"{from_layout1.TEAM}.priority_requests.enabled"
        for value, on in (("false", False), ('""', False), ("requests.jsonl", True)):
            with self.subTest(requests=value):
                for p in self.home.iterdir():
                    p.unlink()
                rc, lines = self.migrate(team.format(value))
                self.assertEqual((rc, lines[-2:]), (0, MIGRATED))
                back = settings_file.read(self.path, DEPLOY)
                self.assertIs(
                    back.values["features"]["dashboards"]["team"]["priority_requests"][
                        "enabled"
                    ],
                    on,
                )
                self.assertEqual(
                    f"is now {enabled} true" in " ".join(" ".join(lines).split()), on
                )

    def test_an_off_feature_with_a_value_of_its_own_is_written_in_full(self):
        # A guard that is off and keeps layout 1's labels keeps its defaults (from_layout1.guard_labels_written).
        guard_off = HEAD + "features:\n  merge_queue:\n    develop_guard: false\n"
        for text, expected in (
            (guard_off, GUARD_ALONE),
            (
                guard_off
                + "advanced:\n  incident_issue:\n    labels: [post-merge]\n    stability_label: rocke-stability\n",
                GUARD_ALONE,
            ),
            (
                guard_off + "advanced:\n  incident_issue:\n    repo: my-org/issues\n",
                GUARD_IN_FULL,
            ),
        ):
            with self.subTest(expected=expected.splitlines()[-1]):
                for p in self.home.iterdir():
                    p.unlink()
                self.assertEqual(self.migrate(text)[0], 0)
                written = self.path.read_text(encoding="utf-8").splitlines()
                start = written.index("    develop_guard:") - 3
                end = start + 4
                while written[end].startswith(" " * 6):  # the lines under the key
                    end += 1
                self.assertEqual(
                    "".join(f"{line}\n" for line in written[start:end]), expected
                )

    def test_a_value_the_writer_cannot_write_is_an_error(self):
        # The error names the value's key as the file has it, and its new home when a step moved it.
        for body, yours in (
            (
                "monitor:\n  tick_min:\n    every: 5\n",
                "your monitor.tick_min, now monitor.tick_interval_minutes.every",
            ),
            ("repository:\n  name:\n    first: 5\n", "your repository.name.first"),
        ):
            with self.subTest(yours=yours):
                text = HEAD + body
                rc, lines = self.migrate(text)
                upgraded = settings_file.upgrade(
                    Document(DEPLOY, 1, layout1.loaded(text.split("\n", 2)[2]))
                )[1]
                report = [
                    "~/rkmon-deploy.yaml: deploy settings, from layout 1 to layout 2"
                ]
                for note in upgraded:  # the steps' notes alone: no value is written
                    report += wrapping.wrap(note, checks.WIDTH, "  ", "    ")
                self.assertEqual(
                    (rc, lines),
                    (
                        1,
                        [
                            BANNER,
                            *report,
                            *wrapping.wrap(
                                "ERROR: ~/rkmon-deploy.yaml is not migrated, as its new file cannot be written: "
                                f"{yours}: no key of deploy settings, so the writer has no place for it",
                                checks.WIDTH,
                            ),
                            "fix: correct that value in ~/rkmon-deploy.yaml, then run rkmon migrate --settings "
                            "~/rkmon-deploy.yaml",
                            AGAIN,
                        ],
                    ),
                )
                self.assertNothingWritten(text)

    def test_a_step_that_fails_is_an_error(self):
        # The build file's ROCm pins are compared with this package's build defaults: a step that fails
        # there (a ValueError) is an error naming it, never a traceback, and nothing is written.
        text = (
            "kind: build_settings\nschema_version: 1\nrocm:\n  clang_commit: abc123\n"
        )
        path, given = self.home / "rkmon-build.yaml", "~/rkmon-build.yaml"
        failed = ValueError("the package's build defaults cannot be read")
        with mock.patch.object(build_plan, "clang_commit", side_effect=failed):
            rc, lines = self.migrate(text, path=path, given=given)
        self.assertEqual(
            (rc, lines[-2:]),
            (
                1,
                [
                    "ERROR: ~/rkmon-build.yaml is not migrated: the package's build defaults cannot be read",
                    "next: rkmon migrate --settings ~/rkmon-build.yaml",
                ],
            ),
        )
        self.assertNothingWritten(text, name="rkmon-build.yaml")

    def test_a_read_or_a_write_that_fails_is_an_error(self):
        def fails(*args, **kwargs):
            raise OSError(errno.ENOSPC, "No space left on device", str(self.path))

        for name in ("read", "rewrite"):
            with self.subTest(name=name), mock.patch.object(settings_file, name, fails):
                rc, lines = self.migrate(SMALL)
                self.assertEqual(
                    (rc, lines[-2:]),
                    (
                        1,
                        [
                            "ERROR: ~/rkmon-deploy.yaml is not migrated: ~/rkmon-deploy.yaml: No space left on device",
                            AGAIN,
                        ],
                    ),
                )
                self.assertNothingWritten(SMALL)

    def test_a_file_that_changes_while_migrate_runs_is_an_error(self):
        # Edited once migrate has read it: your edit stays in the file, and nothing is written or kept.
        edited = SMALL.replace("tick_min: 5", "tick_min: 7")
        real = settings_file.upgrade

        def then_edited(doc):
            upgraded = real(doc)
            self.path.write_text(edited, encoding="utf-8")
            return upgraded

        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run), mock.patch.object(
                settings_file, "upgrade", then_edited
            ):
                rc, lines = self.migrate(SMALL, dry_run=dry_run)
                self.assertEqual(
                    (rc, lines[-2:]),
                    (
                        1,
                        [
                            "ERROR: ~/rkmon-deploy.yaml is not migrated: ~/rkmon-deploy.yaml changed after migrate "
                            "read it",
                            AGAIN,
                        ],
                    ),
                )
                self.assertNothingWritten(edited)


class TestProfiles(Migrate):
    """The profile the title line names, and why, as the report says it."""

    def test_the_profile_the_title_line_names(self):
        for text, name, why in (
            (
                "",
                "full",
                'the title line names the profile "full" (reviews, the merge queue and the dashboards), which starts '
                "with features.review_automation.enabled true and features.merge_queue.enabled true, as your file "
                "has them",
            ),
            (
                "features:\n  merge_queue:\n    enabled: false\n",
                "review-only",
                'the title line names the profile "review-only" (reviews and the admin dashboard), which starts with '
                "features.review_automation.enabled true and features.merge_queue.enabled false, as your file has "
                "them",
            ),
            (
                "features:\n  review_automation:\n    enabled: false\n",
                "merge-queue",
                'the title line names the profile "merge-queue" (the merge queue and the admin dashboard, without '
                "Claude), which starts with features.review_automation.enabled false and features.merge_queue.enabled "
                "true, as your file has them",
            ),
            (
                "features:\n  review_automation:\n    enabled: false\n  merge_queue:\n    enabled: false\n",
                None,
                "the title line names no profile, as none starts with features.review_automation.enabled false and "
                "features.merge_queue.enabled false, as your file has them",
            ),
        ):
            with self.subTest(name=name):
                for p in self.home.iterdir():
                    p.unlink()
                rc, lines = self.migrate(HEAD + text)
                wrapped = wrapping.wrap(why, checks.WIDTH, "  ", "    ")
                self.assertEqual(
                    (rc, lines[-2 - len(wrapped) :]), (0, [*wrapped, *MIGRATED])
                )
                self.assertEqual(settings_file.read(self.path, DEPLOY).profile, name)
                title = self.path.read_text(encoding="utf-8").splitlines()[0]
                whats = {p.name: p.what for p in settings_file.profiles(DEPLOY)}
                self.assertEqual(
                    title,
                    (
                        f'# rocKE monitor: deploy settings (profile "{name}": {whats[name]}).'
                        if name
                        else "# rocKE monitor: deploy settings."
                    ),
                )

    def test_the_profiles_are_the_ones_settings_file_profiles_gives(self):
        # The one reader of the schema's profiles: the title line names a profile it gives, and no other.
        solo = settings_file.Profile("solo", "one profile", (), {})
        self.path.write_text(
            '# rocKE monitor: deploy settings (profile "solo": one profile).\n'
            "kind: deploy_settings\nschema_version: 2\n",
            encoding="utf-8",
        )
        with mock.patch.object(settings_file, "profiles", return_value=(solo,)):
            self.assertEqual(
                settings_file.profile_of(Document(DEPLOY, 2, {})),
                ("solo", 'the title line names the profile "solo" (one profile)'),
            )
            self.assertEqual(settings_file.read(self.path, DEPLOY).profile, "solo")
        self.assertIsNone(settings_file.read(self.path, DEPLOY).profile)

    def test_a_title_line_that_names_the_product_by_its_earlier_name(self):
        # A title line an earlier package wrote (settings_file.EARLIER_PRODUCT_NAME): it names its profile all the
        # same, and the file written from it names the product as the schema's title does. Another name does not.
        what = "reviews and the admin dashboard"
        self.path.write_text(
            f'# rocKE PR monitor: deploy settings (profile "review-only": {what}).\n'
            "kind: deploy_settings\nschema_version: 2\n",
            encoding="utf-8",
        )
        doc = settings_file.read(self.path, DEPLOY)
        self.assertEqual(doc.profile, "review-only")
        text = settings_file.dump(doc)
        self.assertIn(
            f'# rocKE monitor: deploy settings (profile "review-only": {what}).',
            text.splitlines(),
        )
        self.assertNotIn("rocKE PR monitor", text)
        self.path.write_text(
            f'# rocKE PR review: deploy settings (profile "review-only": {what}).\n'
            "kind: deploy_settings\nschema_version: 2\n",
            encoding="utf-8",
        )
        self.assertIsNone(settings_file.read(self.path, DEPLOY).profile)

    def test_what_tells_the_profiles_apart(self):
        def named(values) -> str:
            return settings_file.profile_of(Document(DEPLOY, 2, values))[0]

        # A switch the file leaves out, or blank, is the template's: on.
        self.assertEqual(named({}), "full")
        self.assertEqual(
            named({"features": {"merge_queue": {"enabled": None}}}), "full"
        )
        # A feature's own parts go with it: the merge queue's drift notices tell no profile apart.
        self.assertEqual(
            named({"features": {"merge_queue": {"drift_notices": {"enabled": False}}}}),
            "full",
        )
        self.assertEqual(
            named(
                {
                    "features": {
                        "review_automation": {"enabled": False},
                        "merge_queue": {"drift_notices": {"enabled": True}},
                    }
                }
            ),
            "merge-queue",
        )
        # Nor does a feature every profile starts alike.
        self.assertEqual(
            named({"features": {"weekly_report": {"enabled": True}}}), "full"
        )

    def test_the_first_profile_that_matches_and_a_profile_that_sets_a_section(self):
        file_type = with_profiles(
            {
                "a": {"what": "A", "values": {MQ: {"enabled": False}}},
                "b": {"what": "B", "values": {f"{MQ}.enabled": False}},
                "c": {"what": "C"},
            }
        )
        self.assertEqual(
            settings_file.profile_of(
                Document(
                    file_type, 2, {"features": {"merge_queue": {"enabled": False}}}
                )
            ),
            (
                "a",
                'the title line names the profile "a" (A), which starts with features.merge_queue.enabled false, as '
                "your file has them",
            ),
        )
        self.assertEqual(settings_file.profile_of(Document(file_type, 2, {}))[0], "c")

    def test_a_file_type_without_profiles(self):
        for file_type in (BUILD, with_profiles({})):
            with self.subTest(kind=file_type.noun):
                self.assertEqual(
                    settings_file.profile_of(Document(file_type, 2, {})),
                    (
                        None,
                        f"the title line names no profile, as {file_type.noun} have no profiles",
                    ),
                )


class TestTheCommand(unittest.TestCase):
    """The command's row (commands.TABLE), the file types it reads, and their step from layout 1."""

    def test_its_row(self):
        row = commands.find(("migrate",), commands.TABLE)
        text = commands.help_text(row)
        self.assertEqual(
            text.splitlines()[:3],
            [
                "rkmon migrate: rewrite a settings file in this package's layout.",
                "",
                "Usage:  rkmon migrate --settings FILE [--dry-run]",
            ],
        )
        self.assertLessEqual(
            max(len(line) for line in text.splitlines()), commands.WIDTH
        )
        parser = commands.parser(commands.TABLE)
        for argv, dry_run in (
            (["migrate", "--settings", "f.yaml"], False),
            (["--dry-run", "migrate", "--settings", "f.yaml"], True),
            (["migrate", "--settings=f.yaml", "--dry-run"], True),
        ):
            with self.subTest(argv=argv):
                ns = parser.parse_args(argv)
                self.assertEqual(
                    (ns.row, ns.settings, ns.dry_run), (row, "f.yaml", dry_run)
                )
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            parser.parse_args(["migrate"])
        self.assertEqual(
            (cm.exception.code, err.getvalue()),
            (2, "rkmon migrate: the following arguments are required: --settings\n"),
        )

    def test_the_file_types_and_their_step_from_layout_1(self):
        self.assertEqual(migrate.FILE_TYPES, (DEPLOY, BUILD))
        self.assertEqual(DEPLOY.upgrades, {1: from_layout1.deploy})
        self.assertEqual(BUILD.upgrades, {1: from_layout1.build})

    def test_upgrading_opens_no_package_file_but_layout_2s_schemas_and_layout_1s_frozen_defaults(
        self,
    ):
        """The step means the same in every later package: it reads the earlier file by the moves and rules of
        rkmon_lib/from_layout1.py, and of the package's files it opens layout 2's schemas and layout 1's frozen
        defaults alone, never a schema of layout 1 nor the package's own defaults."""
        package = os.path.realpath(ROOT) + os.sep
        allowed = {
            os.path.realpath(p)
            for p in (DEPLOY.schema, BUILD.schema, from_layout1.DEFAULTS_FILE)
        }
        documents = [
            Document(file_type, 1, layout1.loaded(text.split("\n", 3)[3]))
            for file_type, text in (
                (DEPLOY, layout1.FULL_DEPLOY),
                (BUILD, layout1.FULL_BUILD),
            )
        ]
        opened, real = [], io.open

        def recorded(file, *args, **kwargs):
            if isinstance(file, (str, os.PathLike)):
                opened.append(os.path.realpath(file))
            return real(file, *args, **kwargs)

        def refused(*args, **kwargs):
            raise AssertionError("the step read a schema of layout 1")

        with mock.patch("builtins.open", recorded), mock.patch(
            "io.open", recorded
        ), mock.patch.object(schema, "load", refused):
            for doc in documents:
                settings_file.upgrade(doc)
        self.assertIn(os.path.realpath(from_layout1.DEFAULTS_FILE), opened)
        self.assertEqual(
            [p for p in opened if p.startswith(package) and p not in allowed], []
        )


class TestThroughRkmon(unittest.TestCase):
    """docker/rkmon migrate, against tests/test_wrapper.py's stand-in for docker on a PATH of ordinary tools only: it
    runs on this host, and runs no docker command."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = pathlib.Path(os.path.realpath(tmp.name))
        bin_dir = pathlib.Path(sandbox.safe_path(base))
        shutil.copy(HERE / "fakes" / "fake_docker", bin_dir / "docker")
        (bin_dir / "docker").chmod(0o755)
        self.log = base / "docker.log"
        self.home = base / "home"
        self.home.mkdir()
        self.path = self.home / "rkmon-deploy.yaml"
        self.path.write_text(SMALL, encoding="utf-8")
        self.path.chmod(0o600)
        self.w = str(WRAPPER)  # not on the PATH: the messages name it as it was run
        self.env = {
            "PATH": str(bin_dir),
            "HOME": str(self.home),
            "LANG": "C.UTF-8",
            "FAKE_DOCKER_LOG": str(self.log),
        }

    def again(self, words: str) -> list:
        """The next line that runs docker/rkmon WORDS, as it was run, on the lines a long path wraps it on."""
        return wrapping.wrap(f"next: {self.w} {words}", checks.WIDTH)

    def rkmon(self, *args) -> tuple:
        """(exit code, stdout lines, stderr) of docker/rkmon ARGS, run in the home folder."""
        p = subprocess.run(
            [self.w, *args],
            cwd=self.home,
            env=self.env,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=60,
        )
        self.assertFalse(self.log.exists(), "rkmon migrate ran docker")
        return p.returncode, p.stdout.splitlines(), p.stderr

    def test_it_migrates_the_file_you_name(self):
        rc, out, err = self.rkmon("migrate", "--settings", "rkmon-deploy.yaml")
        self.assertEqual((rc, err, out[0]), (0, "", BANNER))
        then = self.again("check --settings ~/rkmon-deploy.yaml")
        self.assertEqual(out[-1 - len(then) :], [MIGRATED[0], *then])
        self.assertEqual(
            (self.home / "rkmon-deploy.yaml.v1").read_bytes(), SMALL.encode()
        )
        self.assertEqual(settings_file.read(self.path, DEPLOY).version, 2)

    def test_dry_run_before_or_after_the_command(self):
        for args in (
            ("--dry-run", "migrate", "--settings", "rkmon-deploy.yaml"),
            ("migrate", "--settings=rkmon-deploy.yaml", "--dry-run"),
        ):
            with self.subTest(args=args):
                rc, out, err = self.rkmon(*args)
                end = [
                    *CAN_WRITE,
                    *self.again("migrate --settings ~/rkmon-deploy.yaml"),
                ]
                self.assertEqual(
                    (rc, err, out[0], out[-len(end) :]),
                    (0, "", BANNER + " (dry run: nothing changes)", end),
                )
                self.assertEqual(os.listdir(self.home), ["rkmon-deploy.yaml"])
                self.assertEqual(self.path.read_bytes(), SMALL.encode())

    def test_a_refusals_fix_runs_the_command_you_typed(self):
        kept = self.home / "rkmon-deploy.yaml.v1"
        kept.write_text("another file\n", encoding="utf-8")
        rc, out, _err = self.rkmon("migrate", "--settings", "rkmon-deploy.yaml")
        again = self.again("migrate --settings ~/rkmon-deploy.yaml")
        self.assertEqual((rc, out[-len(again) :]), (4, again))
        self.assertIn(
            f"then {self.w} migrate --settings ~/rkmon-deploy.yaml",
            " ".join(" ".join(out).split()),
        )
        self.assertEqual(self.path.read_bytes(), SMALL.encode())

    def test_its_help_and_its_usage_error(self):
        rc, out, err = self.rkmon("migrate", "--help")
        first = " ".join(out[: out.index("")])  # its first paragraph, wrapped
        self.assertEqual(
            (rc, err, first),
            (
                0,
                "",
                f"{self.w} migrate: rewrite a settings file in this package's layout.",
            ),
        )
        rc, out, err = self.rkmon("migrate")
        self.assertEqual(
            (rc, out, err),
            (
                2,
                [],
                f"{self.w} migrate: the following arguments are required: --settings\n",
            ),
        )


if __name__ == "__main__":
    unittest.main()

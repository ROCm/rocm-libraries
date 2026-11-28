# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rkmon_lib/deployment.py, the container's two jobs on the host: the build path's and the deploy path's guards, as
plain functions over made-up settings and host facts (no docker, no network); and each path's steps (rkmon_lib/steps.py)
in its guide and in rkmon help, in order.
"""

import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.append(str(ROOT / "tools"))  # doc_lints, for sizing_home
sys.dont_write_bytecode = True

import sandbox  # noqa: E402
import test_build_plan as plans  # noqa: E402  (its build settings documents; its own tests run in their module)
import test_deploy_plan as deploy_plans  # noqa: E402  (its deploy settings documents, likewise)
from rkmon_lib import (  # noqa: E402
    build_plan,
    checks,
    coherence,
    commands,
    config,
    container_layout,
    deployment as d,
    file_types,
    identity,
    image,
    program,
    ramdisk,
    registry,
    settings_file,
    steps,
    webserve,
)  # noqa: E402

# A Python floor as a text states it ("Python 3.10 or newer", "3.10+", "3.6 or later", across a line's end too), and
# the floors of machines other than the host and the monitor's, with whose each is.
FLOOR = re.compile(r"(?<![\d.])(3\.\d+)(?:\+|,?\s+or\s+(?:newer|later))")
OTHER_FLOORS = {
    "3.12": "the image's base (docker/pkgs.sh)",
    "3.6": "the team web host (tools/webhost_server.py)",
}


def sizing_home() -> tuple:
    """(the guide that says what a machine needs, the heading line of its section that says it), as
    tests/doc_fact_homes.yaml names them (the fact machine_sizing), the one place that names that section.
    """
    import doc_lints

    [fact] = [f for f in doc_lints.load_homes(ROOT) if f.name == "machine_sizing"]
    return fact.home, f"\n### {fact.section}\n"


def machine_sizing(guide: str) -> slice:
    """Where the section that says what a machine needs (sizing_home) is in GUIDE, the deploy guide's text: from the
    end of its heading line to the next heading of its level, or the end."""
    heading = sizing_home()[1]
    start = guide.index(heading) + len(heading)
    end = guide.find("\n### ", start)
    return slice(start, len(guide) if end == -1 else end)


def floors_outside_their_home(texts: dict, host_floor: str, pymin: str) -> list:
    """Each Python floor (FLOOR finds them) that TEXTS (a file's path -> its text) states outside its home, as
    "path:line: words": the home of the host's floor, HOST_FLOOR, is the deploy guide's Machine sizing; the monitor's
    own, PYMIN, and other machines' (OTHER_FLOORS) stand anywhere."""
    out = []
    for rel, text in texts.items():
        home = machine_sizing(text) if rel == sizing_home()[0] else slice(0, 0)
        for m in FLOOR.finditer(text):
            v = m.group(1)
            at_home = v == host_floor and home.start <= m.start() < home.stop
            if v == pymin or v in OTHER_FLOORS or at_home:
                continue
            out.append(f"{rel}:{text.count(chr(10), 0, m.start()) + 1}: {m.group(0)}")
    return out


def cfg(**over) -> dict:
    out = {
        "repo": "acme/widgets",
        "rocke_path": "widgets/rocke",
        "operator": {"github_login": "op-login"},
        "scope": {"labels": ["widgets"]},
        "features": {},
        # the container name, which a deploy settings file's configuration always holds (its default when the file
        # leaves it out)
        "deployment": {"project": "rkmon"},
    }
    for k, v in over.items():
        node = out
        parts = k.split("__")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return out


def help_line(usage: str, s, start: int = 0) -> int:
    """Where `rkmon help` lists a step, from START on: its number, its name, then its command ("  3 confirm
    rkmon confirm ..."); -1 when it does not."""
    prog, command = program.name(), steps.command_name(s.command)
    m = re.compile(
        rf"(?m)^  {s.n} {re.escape(s.name)}\s{{2,}}{re.escape(prog)} {re.escape(command)}\b"
    ).search(usage, start)
    return m.start() if m else -1


class TestThePath(unittest.TestCase):
    def test_the_guides_and_the_wrapper_name_the_same_steps_in_the_same_order(self):
        """Each path's steps (rkmon_lib/steps.py) have their headings in their guide, in order, each naming its
        command, and their lines in rkmon help (the command table's overview), in order, within that path's part; the
        build path's part comes first.
        """
        usage = commands.table_help()
        build = usage.find("\nThe build path, ")
        deploy = usage.find("\nThe deploy path, ")
        self.assertLess(-1, build)
        self.assertLess(build, deploy)
        for path, name, start, end in (
            (steps.BUILD_STEPS, "QUICK_BUILD_GUIDE.md", build, deploy),
            (steps.DEPLOY_STEPS, "QUICK_DEPLOY_GUIDE.md", deploy, len(usage)),
        ):
            guide = (ROOT / name).read_text(encoding="utf-8")
            at = []
            for s in path:
                m = re.search(rf"(?m)^## Step {s.n}: {re.escape(s.name)}$", guide)
                at.append(m.start() if m else -1)
            self.assertNotIn(-1, at, f"each step's heading in {name}")
            self.assertEqual(at, sorted(at))
            for s in path:
                self.assertIn("rkmon " + steps.command_name(s.command), guide)
            at = [help_line(usage, s, start) for s in path]
            self.assertNotIn(-1, at, f"each step in rkmon help's {name} part")
            self.assertEqual(at, sorted(at))
            self.assertLess(at[-1], end)

    def test_the_sizing_constants_are_the_package_s_and_the_ramdisk_s(
        self,
    ):
        """The memory the work needs beside the RAM disk and the RAM disk's first estimates are the package's defaults
        (config.defaults.json, through the registry), and the RAM disk's estimates are these.
        """
        self.assertEqual(
            (d.WORK_GB, d.PIPE_GB, d.HEAVY_GB),
            (
                registry.default("resources.work_memory_gib"),
                registry.default("ramdisk.review_estimate_gib"),
                registry.default("ramdisk.build_estimate_gib"),
            ),
        )
        self.assertEqual(
            (d.PIPE_GB, d.HEAVY_GB), (ramdisk.SEED_GB["pipe"], ramdisk.SEED_GB["heavy"])
        )
        # A blank admin port is its default, the registry's, in docker/.env and in the container's own check
        port = registry.default("admin_dashboard.public_port")
        self.assertEqual(
            d.admin_public_port({"admin_dashboard": {"public_port": 0}}), port
        )
        self.assertIn(("RKMON_PORT", str(port), ""), webserve.publish_env({}))


class TestYourFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.pkg = self.d / "pkg"
        (self.pkg / "docker").mkdir(parents=True)

    def put(self, *rels: str, text: str = "x\n") -> None:
        """Files RELS in the package folder, each holding TEXT."""
        for rel in rels:
            (self.pkg / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.pkg / rel).write_text(text, encoding="utf-8")

    def test_a_settings_file_in_the_package_folder_is_found(self):
        """Of either kind, by its header (settings.header_line), whatever its name and wherever it is, the tests' and
        the agentic files' folders too; a file that only shows a header in its text is none.
        """
        # A settings file's own rules (yours only, outside the package, git and cloud sync) are private_files.problems,
        # tested in tests/test_private_files.py.
        self.put("docker/my-settings.yaml", text="rkmon_settings: 1\nconfig: {}\n")
        self.put(
            "tests/fixture.json", text='{"schema_version": 2, "kind": "build_settings"}'
        )
        self.put(
            "agentic/prod.conf",
            text="# ours\nschema_version: 2\nkind: deploy_settings\n",
        )
        self.put("docs/how.md", text="Start the file with `kind: deploy_settings`.\n")
        self.put("rkmon_lib/x.py", text='HEADER = "kind: deploy_settings"\n')
        self.assertEqual(
            d.stray_private_files(self.pkg),
            ["agentic/prod.conf", "docker/my-settings.yaml", "tests/fixture.json"],
        )

    def test_a_settings_file_s_kind_is_its_header(self):
        """Its first line after the comments: either layout's header, in YAML or JSON; any other first key, another
        kind, or a file that cannot be read is no settings file."""
        f = self.d / "s.yaml"
        for text, kind in (
            ("kind: deploy_settings\nschema_version: 2\n", "deploy"),
            ("# notes\n---\nrkmon_settings: 1\nconfig: {}\n", "deploy"),
            ('{"kind": "build_settings", "schema_version": 2}', "build"),
            ("\ufeffrkmon_build_settings: 1\n", "build"),
            ("kind: 'build_settings'\n", "build"),
            ("a: 1\nkind: deploy_settings\n", ""),
            ("kind: deploy_settings_draft\n", ""),
            ("kind: internal_state\n", ""),
        ):
            with self.subTest(text=text):
                f.write_text(text, encoding="utf-8")
                self.assertEqual(d.settings_kind(f), kind)
        f.write_bytes(b"\xff\xfe\x00k")
        self.assertEqual(d.settings_kind(f), "")
        self.assertEqual(d.settings_kind(self.d / "none.yaml"), "")

    def test_secrets_in_the_package_folder_are_found(self):
        (self.pkg / "docker" / "teams_webhook").write_text("x")
        (self.pkg / "docker" / "web_host_ssh_key").write_text("x")
        (self.pkg / "docker" / "webhost_ed25519").write_text("x")  # its older name
        (self.pkg / "docker" / "compose.yaml").write_text("x")
        self.assertEqual(
            d.stray_private_files(self.pkg),
            [
                "docker/teams_webhook",
                "docker/web_host_ssh_key",
                "docker/webhost_ed25519",
            ],
        )

    def test_each_name_the_secrets_registry_gives_is_found(self):
        """The secrets registry's names (secrets_registry.secret_patterns): each secret's file and its copies, a login's
        files, a model binding's key, a key file, wherever it is: a docker/ca-certificates folder is no folder of the
        package's (the build takes your root certificates from the files your build settings name), so a key or a
        .pem there is found too."""
        names = (
            "claude_env",
            "teams_webhook.old",
            "glm_api_key",
            "hosts.yml",
            "credentials",
            "site.pem",
            "id_rsa",
        )
        for name in names:
            (self.pkg / "docker" / name).write_text("x")
        staged = self.pkg / "docker" / "ca-certificates"
        staged.mkdir()
        for name in ("corp-root.pem", "corp-root.key"):
            (staged / name).write_text("x")
        self.assertEqual(
            d.stray_private_files(self.pkg),
            sorted(
                [f"docker/{n}" for n in names]
                + [
                    "docker/ca-certificates/corp-root.key",
                    "docker/ca-certificates/corp-root.pem",
                ]
            ),
        )

    def test_a_folder_that_may_hold_a_secret_is_found_and_not_looked_into(self):
        """A login's folder and the secrets folders, wherever they are: named with a trailing /, nothing in it listed;
        a file of such a name is no folder of secrets."""
        self.put(
            "gh/hosts.yml",
            "tools/.config/gh/hosts.yml",
            "docker/secrets-write/github_write",
            "rkmon_lib/deep/atlassian/credentials",
            "worker/gh",
        )
        self.assertEqual(
            d.stray_private_files(self.pkg),
            [
                "docker/secrets-write/",
                "gh/",
                "rkmon_lib/deep/atlassian/",
                "tools/.config/",
            ],
        )

    def test_the_package_s_own_files_are_strays_only_out_of_their_places(self):
        """In a package folder in use, .env and the override, their copies, a write's new file and the host's records
        lie in their places (private_files.in_own_places); the same names anywhere else are strays. A package's code
        that comes in (an update's new tree) holds none of them: there, each is a stray.
        """
        own = (
            "docker/.env",
            "docker/.env.prev",
            "docker/compose.override.yaml",
            "docker/compose.override.yaml.mine",
            "docker/..env.4242.0a1b2c3d.new",
        )
        self.put(*own, "docker/.rkmon/records.json")
        self.put(".env", "tests/compose.override.yaml", "rkmon_lib/.rkmon/records.json")
        out_of_place = [".env", "rkmon_lib/.rkmon/", "tests/compose.override.yaml"]
        self.assertEqual(d.stray_private_files(self.pkg), out_of_place)
        self.assertEqual(
            d.stray_private_files(self.pkg, in_use=False),
            sorted(out_of_place + list(own) + ["docker/.rkmon/"]),
        )

    def test_what_the_scan_does_not_look_into(self):
        """Git's own files and the caches, wherever they are; in a package folder in use, the folders a developer's
        copy makes at its top (paths.COPY_OWN_FOLDERS), which the scan of an update's new tree looks into.
        """
        self.put(
            ".git/claude_env",
            "rkmon_lib/__pycache__/credentials",
            "tests/.pytest_cache/hosts.yml",
            "state/claude_env",
            "dist/rkmon/credentials",
            "sandbox/my-settings.yaml",
            "rkmon_lib/state/claude_env",
        )
        self.assertEqual(
            d.stray_private_files(self.pkg), ["rkmon_lib/state/claude_env"]
        )
        self.assertEqual(
            d.stray_private_files(self.pkg, in_use=False),
            [
                "dist/rkmon/credentials",
                "rkmon_lib/state/claude_env",
                "sandbox/my-settings.yaml",
                "state/claude_env",
            ],
        )

    def test_a_link_is_judged_by_its_name_alone(self):
        """Never followed: a link named as a secret's file is found; a link to a settings file elsewhere is not read."""
        theirs = self.d / "s.yaml"
        theirs.write_text("kind: deploy_settings\n", encoding="utf-8")
        (self.pkg / "docker" / "notes.yaml").symlink_to(theirs)
        (self.pkg / "docker" / "claude_env").symlink_to(self.d / "nowhere")
        self.assertEqual(d.stray_private_files(self.pkg), ["docker/claude_env"])

    def test_the_package_s_code_holds_no_private_file(self):
        """The package's files (git ls-files), copied as an update copies them, pass the scan of a new tree: an update
        to this code is never refused for it."""
        try:
            files = subprocess.run(
                ["git", "-C", str(ROOT), "ls-files"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.splitlines()
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("not a git checkout")
        new = self.d / "new"
        for rel in files:
            src = ROOT / rel
            if src.is_file() and not src.is_symlink():
                (new / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, new / rel)
        self.assertTrue((new / "rkmon_lib").is_dir())
        self.assertEqual(d.stray_private_files(new, in_use=False), [])

    def test_the_row_names_each_stray_and_where_it_goes(self):
        """None while the package folder holds none; else one FAIL row naming each, its fix to move each out of the
        package folder and run the command again; for a new package's code (SOURCE), to take each out of SOURCE.
        """
        again = checks.Command(("rkmon", "build", "--settings", "~/b.yaml"))
        self.assertEqual(d.stray_rows(self.pkg, again), [])
        self.put("docker/claude_env", "docker/.env")
        [row] = d.stray_rows(self.pkg, again)
        self.assertEqual(
            (row.name, row.status, row.found),
            (
                "the package folder",
                "FAIL",
                "private files inside it: docker/claude_env",
            ),
        )
        self.assertIn(
            "move each out of the package folder (a build, a commit or an update would carry it), then run rkmon "
            "build --settings ~/b.yaml",
            " ".join(checks.render([row], "all").split()),
        )
        [row] = d.stray_rows(self.pkg, again, source="/src/new")
        self.assertEqual(
            (row.name, row.status, row.found),
            (
                "the new package",
                "FAIL",
                "private files inside it, which no package carries: docker/.env, docker/claude_env",
            ),
        )
        self.assertIn(
            "take each out of /src/new, then run rkmon build --settings ~/b.yaml",
            " ".join(checks.render([row], "all").split()),
        )

    def test_the_ca_certificates_are_staged_from_the_settings(self):
        """Into the build's own folder, as ca-1.crt, ca-2.crt in the settings' order (mode 644), whatever their names:
        no name of yours reaches the build."""
        pem = self.d / "corp-root.pem"
        pem.write_text("-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n")
        crt = self.d / "proxy.crt"
        crt.write_text("-----BEGIN CERTIFICATE-----\nBBBB\n-----END CERTIFICATE-----\n")

        def named(*files):
            return plans.make_plan(
                operating_system={"ca_certificates": [str(f) for f in files]}
            ).ca_files

        self.assertEqual(
            [x.status for x in d.ca_checks(named(pem, crt))], ["PASS", "PASS"]
        )
        folder = self.d / "build-context"
        folder.mkdir()
        self.assertEqual(d.stage_ca(named(pem, crt), folder), 2)
        self.assertEqual(
            sorted(p.name for p in folder.iterdir()), ["ca-1.crt", "ca-2.crt"]
        )
        self.assertEqual((folder / "ca-1.crt").read_text(), pem.read_text())
        self.assertEqual((folder / "ca-2.crt").read_text(), crt.read_text())
        self.assertEqual((folder / "ca-1.crt").stat().st_mode & 0o777, 0o644)
        self.assertEqual(
            d.stage_ca(named(), self.d / "none"), 0
        )  # none named: nothing copied, no folder needed
        gone = d.ca_checks(named(self.d / "gone.crt"), "~/b.yaml")[0]
        self.assertEqual(gone.status, "FAIL")
        self.assertIn(
            "set operating_system.ca_certificates in ~/b.yaml to", str(gone.fix)
        )
        (self.d / "bad.crt").write_text("not a certificate")
        self.assertIn("no PEM", d.ca_checks(named(self.d / "bad.crt"))[0].found)
        self.assertEqual(
            d.ca_checks(named())[0].status, "PASS"
        )  # none: a network that does not inspect TLS

    def test_a_root_certificate_file_is_yours(self):
        """The image trusts whatever the file holds, so no one but you or root may change it: another user's file, or
        one that others, or a group not your own, may write, is a FAIL with its fix; your own group may write it.
        """
        pem = self.d / "corp-root.pem"
        pem.write_text("-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n")
        files = plans.make_plan(
            operating_system={"ca_certificates": [str(pem)]}
        ).ca_files
        uid, gid = pem.stat().st_uid, pem.stat().st_gid
        real_stat = pathlib.Path.stat

        def row(owner=None, you=uid, your_group=gid):
            """The file's one row, as if OWNER owned it, YOU were you and YOUR_GROUP your primary group."""

            def stat(p, *a, **k):
                st = real_stat(p, *a, **k)
                if p != pem or owner is None:
                    return st
                return os.stat_result((*st[:4], owner, *st[5:10]))

            with mock.patch.object(pathlib.Path, "stat", stat), mock.patch.object(
                d.os, "getuid", return_value=you
            ), mock.patch.object(d.os, "getgid", return_value=your_group):
                [r] = d.ca_checks(files, "~/b.yaml")
            return r.status, r.found, str(r.fix) if r.status == "FAIL" else ""

        trust = "could change the certificates the image trusts"
        for mode in (0o644, 0o664, 0o646, 0o666):
            pem.chmod(mode)
            with self.subTest(mode=f"{mode:03o}"):
                self.assertEqual(row()[0], "FAIL" if mode & 0o002 else "PASS")
                self.assertEqual(
                    row(your_group=gid + 1)[0], "FAIL" if mode & 0o022 else "PASS"
                )
        pem.chmod(0o644)
        self.assertEqual(row(), ("PASS", str(pem), ""))
        pem.chmod(0o666)
        self.assertEqual(
            row(),
            (
                "FAIL",
                f"{pem} has permissions 666: others on this machine {trust}",
                f"chmod go-w {pem}",
            ),
        )
        pem.chmod(0o644)
        self.assertEqual(
            row(owner=uid + 1),
            (
                "FAIL",
                f"{pem} belongs to another user, who {trust}",
                "copy it to a file of yours, then set operating_system.ca_certificates in ~/b.yaml to your "
                "network's root certificate files, as PEM (Base-64, .crt), outside the package folder",
            ),
        )
        self.assertEqual(row(owner=0, you=uid + 1)[0], "PASS")  # root's
        pem.chmod(0o646)
        self.assertEqual(
            row(owner=0, you=uid + 1)[2], f"sudo chmod go-w {pem}"
        )  # root's, which others may write


class TestTheSettings(unittest.TestCase):
    def fails(self, c) -> list:
        return [x for x in d.settings_checks(c) if x.status == "FAIL"]

    def test_good_settings_pass(self):
        self.assertEqual(self.fails(cfg()), [])

    def test_the_features_row_word_for_word(self):
        """Each feature's state, the row wrapped before a "·"."""
        (row,) = [
            x
            for x in d.settings_checks(
                cfg(features={"merge_queue": {"enabled": False}})
            )
            if x.name == "features"
        ]
        self.assertEqual(
            checks.render([row]),
            "  PASS  features: review automation prototype · merge queue off · admin dashboard on"
            " · team dashboard off\n        · alerts off · ticket creation off",
        )

    def test_the_template_s_placeholders_stop(self):
        bad = self.fails(cfg(operator={"github_login": "<user_login>"}))
        found = " ".join(x.found for x in bad)
        self.assertIn(
            "integrations.github_cli.username is still the template's <user_login>",
            found,
        )

    def test_a_placeholder_is_known_by_the_settings_files_pattern(self):
        """A template's placeholder has one pattern, settings_file.PLACEHOLDER: the deploy checks stop on what it takes
        (here a stand-in pattern in its place)."""
        stand_in = re.compile(r"\{\{[^{}]+\}\}")
        with mock.patch.object(settings_file, "PLACEHOLDER", stand_in):
            bad = self.fails(cfg(operator={"github_login": "{{login}}"}))
        self.assertIn(
            "integrations.github_cli.username is still the template's {{login}}",
            " ".join(x.found for x in bad),
        )

    def test_the_image_to_run_and_the_users_ids(self):
        self.assertEqual(
            d.image_ref(cfg(deployment__project="rk-x")), "rocke-monitor:rk-x"
        )
        self.assertEqual(
            d.image_tag(cfg(deployment__image="rocke-monitor:split-1")), "split-1"
        )
        # A name of one's own, with the container's name as its tag when the tag is blank.
        mine = cfg(deployment__image="team/rkmon:", deployment__project="rk-x")
        self.assertEqual(d.image_ref(mine), "team/rkmon:rk-x")
        self.assertEqual(d.image_name(mine), "team/rkmon")
        self.assertEqual(self.fails(mine), [])
        # The form of the image's name and tag is the file's to check, by their keys' patterns: the deploy checks
        # leave it, so a name or tag docker does not take gets one FAIL.
        for name, tag in (
            ("registry.example.com/rocke-monitor", "1"),
            ("Rocke PR", "1"),
            ("rocke-monitor", "bad tag"),
        ):
            doc = settings_file.Document(
                file_types.DEPLOY,
                file_types.DEPLOY.version,
                {"image": {"name": name, "tag": tag}},
            )
            [row] = settings_file.misfits(doc)
            self.assertEqual(
                (row.name, row.status),
                ("image.tag" if tag == "bad tag" else "image.name", "FAIL"),
            )
            self.assertEqual(self.fails(cfg(deployment__image=f"{name}:{tag}")), [])
        # The GitHub writer's and the PR runners' ids are the package's: the monitor's user and group keep off
        # them, by their own settings (docker/rk_ids.py's clashes, worded).
        writer, runner = container_layout.WRITER.uid, container_layout.PR_RUNNER.uid
        for uid, why in (
            (writer, "the GitHub writer's id"),
            (runner, "a PR runner's id"),
        ):
            bad = self.fails(cfg(container__uid=uid, container__gid=4343))
            self.assertEqual(len(bad), 1, bad)
            self.assertIn(f"host.users.monitor_uid is {uid}: {why}", bad[0].found)
            self.assertEqual(
                str(bad[0].fix),
                "set host.users.monitor_uid in your deploy settings file to another id",
            )
        bad = self.fails(cfg(container__uid=4242, container__gid=runner))
        self.assertIn(
            f"host.users.monitor_gid is {runner}: the PR runners' group's id",
            bad[0].found,
        )
        self.assertEqual(
            str(bad[0].fix),
            "set host.users.monitor_gid in your deploy settings file to another group's id",
        )
        # A configuration's own ids for the two change nothing.
        mine = cfg(
            container={"uid": 4242, "gid": 4343, "write_uid": 4242, "pr_uid": 4343}
        )
        self.assertEqual(self.fails(mine), [])
        self.assertIn("not a user id", self.fails(cfg(container__uid="me"))[0].found)
        self.assertEqual(self.fails(cfg(container__uid=4242, container__gid=4343)), [])

    def test_the_settings_refuse_the_ids_the_entry_point_refuses(self):
        """The monitor's ids: the settings step and docker/rk_ids.py (each start) take and refuse the same ones, beside
        the GitHub writer's and the PR runners', the package's, which docker/.env does not give, and no row names a
        setting for those: a row about the monitor's ids names the monitor's.
        """
        ids = d.rk_ids()
        base = {"uid": 4000, "gid": 4001}
        have = {
            "uid": container_layout.MONITOR.uid,
            "gid": container_layout.MONITOR.gid,
            "write_uid": container_layout.WRITER.uid,
        }
        # The image's own users' ids (the PR runners', their groups', the writer's) and the ids on either side.
        own = (1966, 1967, 1975, 1982, 1983, 1990, 1998, 1999, 2000)
        for key in base:
            for value in (1, 33, 999, 1000, 4242, 65534, 2**31 - 1, 2**31, *own):
                with self.subTest(key=key, value=value):
                    given = dict(base, **{key: value})
                    c = cfg(container=given)
                    settings = [x for x in self.fails(c) if x.name == "container ids"]
                    # docker/.env's variables for the same ids, as deliver writes them
                    made = deploy_plans.plan(
                        {d.N(f"container.{k}"): v for k, v in given.items()}
                    )
                    try:
                        ids.wanted(made.variables, have)
                        entry = False
                    except ids.Refused:
                        entry = True
                    self.assertEqual(bool(settings), entry, settings)
                    for row in settings:
                        self.assertIn(d.N(f"container.{key}"), str(row.fix))
                        self.assertNotRegex(
                            str(row.fix), r"write_uid|pr_uid|runner_uid"
                        )
        bad = self.fails(cfg(container=dict(base, uid=33)))
        self.assertIn(
            "host.users.monitor_uid is 33: a system account's id", bad[0].found
        )
        self.assertEqual(
            str(bad[0].fix),
            "set host.users.monitor_uid in your deploy settings file to an id of 1000 or more",
        )
        bad = self.fails(cfg(container=dict(base, uid=1990)))
        self.assertEqual(
            bad[0].found,
            "host.users.monitor_uid is 1990: a PR runner's id (1983 to 1998, the package's), which the container's "
            "entry point refuses at each start",
        )
        self.assertEqual(
            str(bad[0].fix),
            "set host.users.monitor_uid in your deploy settings file to another id",
        )
        bad = self.fails(cfg(container=dict(base, gid=1998)))
        self.assertEqual(
            (bad[0].found, str(bad[0].fix)),
            (
                "host.users.monitor_gid is 1998: the PR runners' group's id, which the container's entry point "
                "refuses at each start",
                "set host.users.monitor_gid in your deploy settings file to another group's id",
            ),
        )
        bad = self.fails(cfg(container=dict(base, gid=1982)))
        self.assertEqual(
            bad[0].found,
            "host.users.monitor_gid is 1982: a PR runner's own group's id (1967 to 1982, the package's), which the "
            "container's entry point refuses at each start",
        )
        with mock.patch.object(d.os, "getuid", lambda: 1990):
            bad = [c.found for c in self.fails(cfg()) if c.name == "container ids"]
        self.assertIn(
            "your user id (1990): a PR runner's id (1983 to 1998, the package's), which the container's entry point "
            "refuses at each start",
            bad,
        )
        with mock.patch.object(d.os, "getgid", lambda: 1975):
            bad = [c.found for c in self.fails(cfg()) if c.name == "container ids"]
        self.assertEqual(
            bad,
            [
                "your group id (1975): a PR runner's own group's id (1967 to 1982, the package's), which the "
                "container's entry point refuses at each start"
            ],
        )
        # Left out, the ids are whoever runs rkmon: root's are no ids the container's users take.
        with mock.patch.object(d.os, "getuid", lambda: 0), mock.patch.object(
            d.os, "getgid", lambda: 0
        ):
            bad = [c.found for c in self.fails(cfg()) if c.name == "container ids"]
        self.assertIn(
            "host.users.monitor_uid is left out, and yours (0) is not an id the container's users take",
            bad,
        )
        self.assertEqual(len(bad), 2, bad)
        with mock.patch.object(d.os, "getuid", lambda: 500):
            bad = [c.found for c in self.fails(cfg()) if c.name == "container ids"]
        self.assertEqual(
            bad,
            [
                "your user id (500): a system account's id (below 1000, or nobody's 65534), which the container's "
                "entry point refuses at each start"
            ],
        )

    def test_the_rules_on_the_container_s_share_of_this_host_are_the_file_s_own(self):
        """A RAM disk smaller than the starting reviews and builds need, a memory limit with no room for the RAM disk
        beside the work, and fewer CPUs than the reviews and builds that may run at once are the deploy file's own
        rules on the container's share of this host (coherence.resources, checks 8 and 9), which rkmon check and
        rkmon confirm run with this host's memory and CPUs: the settings' own checks say none of them.
        """
        for c in (
            cfg(
                ramdisk__enabled=True,
                ramdisk__size_gb=4,
                review__pipes=3,
                checks__limit=1,
                resources__memory_gb=40,
            ),
            cfg(ramdisk__enabled=True, ramdisk__size_gb=24, resources__memory_gb=16),
            cfg(
                ramdisk__enabled=False,
                resources__cpus=1,
                review__pipes=3,
                checks__limit=1,
            ),
        ):
            with self.subTest(c=c):
                self.assertEqual(
                    [
                        (x.name, x.status)
                        for x in d.settings_checks(c)
                        if x.name in ("RAM disk", "memory limit", "memory", "CPUs")
                        and x.status != "PASS"
                    ],
                    [],
                )

    def test_a_ramdisk_the_starting_reviews_and_builds_fit_on_but_not_all_of_them(self):
        c = cfg(
            ramdisk__enabled=True,
            ramdisk__size_gb=8,
            review__pipes=3,
            checks__limit=1,
            resources__memory_gb=40,
        )
        self.assertEqual(self.fails(c), [])
        (row,) = [x for x in d.settings_checks(c) if x.name == "RAM disk"]
        self.assertEqual(
            (row.status, row.found, str(row.fix)),
            (
                "WARN",
                "host.resources.ramdisk_gib 8 fits the starting reviews and builds, and no more run at once than fit "
                "on it (all 5 parallel reviews and 8 parallel builds need 23.5 GiB)",
                "fine while fewer run at once; for all of them, set host.resources.ramdisk_gib in your deploy settings "
                "file to 24",
            ),
        )
        # without review automation, the builds alone: the starting one fits on 4 GiB, and all of them need 16
        off = cfg(
            features={"review_automation": {"enabled": False}},
            ramdisk__enabled=True,
            ramdisk__size_gb=4,
            review__pipes=3,
            checks__limit=1,
            resources__memory_gb=40,
        )
        (row,) = [x for x in d.settings_checks(off) if x.name == "RAM disk"]
        self.assertEqual(
            (row.status, row.found, str(row.fix)),
            (
                "WARN",
                "host.resources.ramdisk_gib 4 fits the starting reviews and builds, and no more run at once than fit "
                "on it (all 8 parallel builds need 16 GiB)",
                "fine while fewer run at once; for all of them, set host.resources.ramdisk_gib in your deploy settings "
                "file to 16",
            ),
        )

    def test_a_container_name_compose_takes(self):
        for name in ("My Monitor", ""):
            with self.subTest(name=name):
                bad = [
                    x
                    for x in self.fails(cfg(deployment__project=name))
                    if x.name == "container name"
                ]
                self.assertIn("lowercase", bad[0].found)
        self.assertEqual(d.project(cfg(deployment__project=" rk-x ")), "rk-x")
        # no default: a configuration that names no container has none
        self.assertEqual(d.project({}), "")


class TestTheLocalFiles(unittest.TestCase):
    def test_the_ports_docker_env_publishes(self):
        """The dashboards' ports by docker/.env's variables, as deliver writes them (deploy_plan): the admin
        dashboard's on this host only, the team dashboard's while it is served; a blank address is this host only.
        """
        served = deploy_plans.plan(deploy_plans.SERVED).variables
        self.assertEqual(d.ports(served), [("127.0.0.1", 18614), ("0.0.0.0", 18700)])
        self.assertEqual(d.ports(deploy_plans.plan().variables), [("127.0.0.1", 18614)])
        blank = dict(served, RKMON_ADMIN_HOST="", RKMON_TEAM_HOST="")
        self.assertEqual(d.ports(blank), [("127.0.0.1", 18614), ("127.0.0.1", 18700)])

    def test_the_image_step_s_checks(self):
        c = d.no_image_row(cfg())
        self.assertEqual(
            (c.status, c.found),
            (
                "FAIL",
                "no image rocke-monitor:rkmon here (image.name and image.tag name it)",
            ),
        )
        self.assertIn(
            "rkmon image load IMAGE_FILE, IMAGE_FILE being the image's file", str(c.fix)
        )
        self.assertIn("rkmon image pull, from image.registry", str(c.fix))
        self.assertIn(
            "set image.name in your deploy settings file to its name and set image.tag in your deploy settings file "
            "to its tag, then run rkmon confirm",
            str(c.fix),
        )
        self.assertEqual(d.image_tls_checks("TLS ok\n")[0].status, "PASS")
        bad = d.image_tls_checks(
            "TLS untrusted unable to get local issuer certificate\n"
        )[0]
        self.assertEqual(bad.status, "FAIL")
        self.assertIn(
            "set operating_system.ca_certificates in your build settings file to",
            str(bad.fix),
        )
        self.assertEqual(
            d.image_tls_checks("TLS unreachable [Errno -3]\n")[0].status, "WARN"
        )
        # The probe is the one the check reads.
        self.assertIn("print('TLS ok')", d.TLS_PROBE)
        self.assertIn("print('TLS untrusted'", d.TLS_PROBE)


class TestPreflight(unittest.TestCase):
    def facts(self, **kw):
        f = d.Facts(
            docker="29.0",
            compose="2.40",
            buildx="v0.30",
            disk_free_gb=200,
            disk_where="/var/lib/docker",
            host_gb=64,
            host_cpus=16,
        )
        for k, v in kw.items():
            setattr(f, k, v)
        return f

    def statuses(self, checks, name) -> list:
        return [c.status for c in checks if c.name == name]

    def test_a_ready_host_passes(self):
        rows = d.preflight(cfg(), self.facts(), [])
        self.assertFalse(checks.failed(rows), checks.render(rows))

    def test_the_container_s_share_of_this_host_stands_after_the_disk(self):
        """Step 2's memory and CPUs rows are the deploy file's own rules on the container's share of this host, which
        the caller gives (SHARE: coherence.resources with the host's figures): after the disk's row, before the
        ports'."""
        share = [
            checks.Check(
                "memory", "PASS", "the container may use 8 of this host's 64 GiB"
            )
        ]
        names = [c.name for c in d.preflight(cfg(), self.facts(), [], share=share)]
        self.assertEqual(
            names[names.index("disk") + 1 : names.index("ports")], ["memory"]
        )

    def test_the_build_machine_s_disk(self):
        """Build step 2's disk row: less free than a build with ROCm takes (BUILD_DISK_NEED_GIB) is a FAIL, less than
        BUILD_DISK_WARN_GIB a WARN, and the row says what a build takes."""
        p = plans.make_plan()
        need, warn = d.BUILD_DISK_NEED_GIB, d.BUILD_DISK_WARN_GIB
        for free, status in (
            (need - 1, "FAIL"),
            (need, "WARN"),
            (warn - 1, "WARN"),
            (warn, "PASS"),
        ):
            with self.subTest(free=free):
                rows = d.build_checks(p, self.facts(disk_free_gb=free), [])
                [row] = [r for r in rows if r.name == "disk"]
                self.assertEqual(row.status, status)
                self.assertIn(f"(a build with ROCm takes about {need} GiB:", row.found)

    def test_the_host_s_python_has_one_home(self):
        """The oldest Python rkmon runs on has one home, HOST_PYTHON: step 2's check refuses an older one
        and names it, and the deploy guide's "Machine sizing", which says what a machine needs, states it from there;
        the build guide names that section for it. No other text of the docs, the wrapper or its library, nor the
        deploy guide's other sections, states a floor but the monitor's own (doctor.PY_MIN) and another machine's
        (OTHER_FLOORS): the rest name the section or the constant."""
        from rkmon_lib import doctor

        def row(python):
            found = d.preflight(cfg(), self.facts(python=python), [])
            return [c for c in found if c.name == "host Python"]

        floor = ".".join(map(str, d.HOST_PYTHON))
        [old] = row((d.HOST_PYTHON[0], d.HOST_PYTHON[1] - 1, 10))
        self.assertEqual(old.status, "FAIL")
        self.assertIn(f"needs Python {floor} or newer", old.found)
        self.assertIn(f"python{floor} docker/rkmon check --settings", str(old.fix))
        self.assertEqual(row(d.HOST_PYTHON + (0,)), [])
        with mock.patch.object(d, "HOST_PYTHON", (3, 42)):
            self.assertIn("needs Python 3.42 or newer", row((3, 41, 0))[0].found)
        guide = (ROOT / sizing_home()[0]).read_text(encoding="utf-8")
        self.assertIn(f"Python {floor} or newer", guide[machine_sizing(guide)])
        texts = {
            str(f.relative_to(ROOT)): f.read_text(encoding="utf-8")
            for f in (
                *ROOT.glob("*.md"),
                *(ROOT / "docs").glob("*.md"),
                ROOT / "docker" / "rkmon",
                *(ROOT / "rkmon_lib").glob("*.py"),
            )
        }
        pymin = ".".join(map(str, doctor.PY_MIN))
        self.assertEqual(
            floors_outside_their_home(texts, floor, pymin),
            [],
            "a Python floor stated outside its home",
        )
        # In the deploy guide, the host's floor stands in Machine sizing alone; another machine's, anywhere.
        deploy_guide, heading = sizing_home()
        invented = (
            f"## Before you start\n{heading}\nPython 3.42 or newer.\n\n### Have these ready\n\n"
            "Python 3.42 or newer, and Python 3.6 or newer on the web host.\n"
        )
        self.assertEqual(
            floors_outside_their_home({deploy_guide: invented}, "3.42", pymin),
            [f"{deploy_guide}:9: 3.42 or newer"],
        )
        for sample in (
            "Python 3.8 or newer",
            "Python 3.10+",
            "Python 3.8\nor newer",
            "3.6, or later",
        ):
            self.assertTrue(FLOOR.search(sample), sample)
        for fine in (
            "git 2.44 or newer",
            "Python 3.14 with the checks venv",
            "13.10 or newer",
        ):
            self.assertFalse(FLOOR.search(fine), fine)

    def test_the_deploy_guide_s_machine_sizing_states_the_checks_numbers(self):
        """Machine sizing (QUICK_DEPLOY_GUIDE.md) is the one home of what a machine needs: it states the
        numbers the checks judge by, the deploy host's free disk as step 2's disk row judges it, the RAM disk's first
        estimates and its room for every review and build, the memory beside it, the monitor's ranges, and the build
        machine's disk."""
        guide = (ROOT / sizing_home()[0]).read_text(encoding="utf-8")
        flat = " ".join(guide[machine_sizing(guide)].split())

        def disk(gb):
            found = d.preflight(cfg(), self.facts(disk_free_gb=gb), [])
            [row] = [c for c in found if c.name == "disk"]
            return row.status

        fails_below = next(gb for gb in range(1, 1000) if disk(gb) != "FAIL")
        warns_below = next(gb for gb in range(fails_below, 1000) if disk(gb) == "PASS")
        most_pipes, most_checks = config.pipes_max(), config.checks_max()
        full = coherence.least(reviews=most_pipes, builds=most_checks)
        for words in (
            f"Python {'.'.join(map(str, d.HOST_PYTHON))} or newer",
            f"Less than {fails_below} GiB free is a FAIL, and less than {warns_below} GiB a WARN",
            f"about {d.PIPE_GB:g} GiB for each review and {d.HEAVY_GB:g} GiB for each build",
            f"all {most_pipes} parallel reviews and {most_checks} parallel builds need, about "
            f"{full.about} GiB",
            f"`memory_gib` must be at least `ramdisk_gib` plus {d.WORK_GB:g}",
            f"the parallel reviews, from 1 to {most_pipes}, and the parallel builds, from 1 to {most_checks}",
            f"{d.BUILD_DISK_NEED_GIB} GiB free where Docker keeps its images: less is a FAIL at its step 2, and less "
            f"than {d.BUILD_DISK_WARN_GIB} GiB a WARN",
        ):
            with self.subTest(words=words):
                self.assertIn(words, flat)

    def test_the_machine_sizing_states_the_python_of_each_machine(self):
        """Machine sizing states the Python rkmon needs (deployment.HOST_PYTHON) twice: for this host, in its first
        paragraph, and for a machine that builds the image, in that machine's paragraph; so neither line stands in for
        the other."""
        guide = (ROOT / sizing_home()[0]).read_text(encoding="utf-8")
        section = guide[machine_sizing(guide)]
        paragraphs = [" ".join(p.split()) for p in section.strip().split("\n\n")]
        floor = f"Python {'.'.join(map(str, d.HOST_PYTHON))} or newer"
        [host] = [p for p in paragraphs if p.startswith("This host needs ")]
        [build] = [p for p in paragraphs if p.startswith("A machine that builds ")]
        self.assertIn(f"{floor} for `rkmon`", host)
        self.assertIn(f"needs Docker with buildx, {floor}", build)

    def test_a_busy_port_names_the_key_that_sets_it(self):
        """busy_port_checks, the check's rows and the start's: a FAIL for each busy dashboard port, whose fix sets
        that dashboard's port in the deploy settings file. While the ports the file makes are not known (its own rows
        FAIL), the check has no row for them."""
        c = cfg(
            admin_dashboard__public_port=18714,
            features={"dashboards": {"team": {"enabled": True}}},
            team_dashboard__public_port=18715,
        )
        busy = [("127.0.0.1", 18714), ("0.0.0.0", 18715)]
        rows = d.busy_port_checks(c, busy, "~/d.yaml")
        self.assertEqual(
            [(r.status, r.name, r.found, str(r.fix)) for r in rows],
            [
                (
                    "FAIL",
                    "port",
                    f"{host}:{port} is in use by another program or container",
                    f"set {key} in ~/d.yaml to a port that is free on this host",
                )
                for (host, port), key in zip(
                    busy,
                    (
                        "features.dashboards.admin.port",
                        "features.dashboards.team.serve.port",
                    ),
                )
            ],
        )
        self.assertEqual(d.busy_port_checks(c, [], "~/d.yaml"), [])
        # the row with a fix of its own: rollback's
        own = checks.Steps(("stop it, then", checks.Command(("rkmon", "rollback"))))
        row = d.busy_port_row("127.0.0.1", 18714, own)
        self.assertEqual((row.status, row.name, row.fix), ("FAIL", "port", own))
        self.assertEqual(row.found, rows[0].found)
        # The check's port rows are these; with no port busy, one PASS.
        check = d.preflight(c, self.facts(busy_ports=busy), [], "~/d.yaml")
        self.assertEqual([r for r in check if r.name == "port"], rows)
        free = d.preflight(c, self.facts(), [], "~/d.yaml")
        self.assertEqual(
            [(r.name, r.status) for r in free if r.name in ("port", "ports")],
            [("ports", "PASS")],
        )
        unknown = d.preflight(c, self.facts(busy_ports=None), [], "~/d.yaml")
        self.assertEqual([r for r in unknown if r.name in ("port", "ports")], [])

    def test_each_guard(self):
        self.assertEqual(
            self.statuses(
                d.preflight(cfg(), self.facts(docker="", docker_error="no daemon"), []),
                "docker",
            ),
            ["FAIL"],
        )
        self.assertEqual(
            self.statuses(
                d.preflight(cfg(), self.facts(buildx=""), []), "docker buildx"
            ),
            [],
        )
        self.assertEqual(
            self.statuses(
                d.build_checks(None, self.facts(buildx=""), []), "docker buildx"
            ),
            ["FAIL"],
        )
        self.assertEqual(
            self.statuses(d.preflight(cfg(), self.facts(disk_free_gb=12), []), "disk"),
            ["FAIL"],
        )
        busy = d.preflight(cfg(), self.facts(busy_ports=[("127.0.0.1", 18614)]), [])
        self.assertEqual(self.statuses(busy, "port"), ["FAIL"])
        # The container name: another deployment's, whose it is unknown, free or this folder's own.
        taken = d.preflight(
            cfg(), self.facts(name_owner=identity.OTHER), [], "~/d.yaml"
        )
        row = [c for c in taken if c.name == "container name"][0]
        self.assertEqual(
            (row.status, row.found, str(row.fix)),
            (
                "FAIL",
                "rkmon is used by another deployment on this host",
                "set container.name in ~/d.yaml to a name of its own, such as rkmon-2 (docker compose ls --all "
                "lists the names in use)",
            ),
        )
        unread = d.preflight(
            cfg(), self.facts(name_unknown="docker does not answer (x)"), []
        )
        row = [c for c in unread if c.name == "container name"][0]
        self.assertEqual(
            (row.status, row.found),
            (
                "FAIL",
                "whose rkmon is on this host is unknown: docker does not answer (x)",
            ),
        )
        self.assertIn("docker ps --all", str(row.fix))
        for owner in (identity.NONE, identity.OURS):
            rows = d.preflight(cfg(), self.facts(name_owner=owner), [])
            self.assertEqual(
                [(c.status, c.found) for c in rows if c.name == "container name"],
                [("PASS", "rkmon: no other deployment on this host uses it")],
            )
        # docker not asked (it does not answer): the docker row says so, and no row for the name.
        self.assertEqual(
            self.statuses(d.preflight(cfg(), self.facts(), []), "container name"), []
        )
        old = d.preflight(
            cfg(),
            self.facts(name_owner=identity.NONE, project_volumes=["rkmon_rkmon-work"]),
            [],
        )
        self.assertEqual(self.statuses(old, "container name"), ["WARN"])
        self.assertEqual(
            self.statuses(
                d.preflight(cfg(), self.facts(python=(3, 8, 10)), []), "host Python"
            ),
            ["FAIL"],
        )

    def test_tls_inspection_needs_the_ca_certificate(self):
        self.assertFalse(d.inspected({"ok": True, "issuer": "Sectigo Limited"}))
        self.assertTrue(d.inspected({"ok": True, "issuer": "Example Corp Proxy CA"}))
        self.assertTrue(
            d.inspected(
                {
                    "ok": False,
                    "error": "the certificate is not trusted (unable to get local issuer)",
                }
            )
        )
        probe = {
            "host": "github.com",
            "ok": False,
            "error": "the certificate is not trusted (x)",
            "with_ca": None,
        }
        none = plans.make_plan()
        [row] = d.tls_checks(none, probe)
        self.assertEqual(
            (row.name, row.status, row.found),
            (
                "TLS",
                "FAIL",
                "this network inspects TLS, and no root certificate is named",
            ),
        )
        self.assertIn(
            "set operating_system.ca_certificates in your build settings file to",
            str(row.fix),
        )
        with tempfile.TemporaryDirectory() as t:
            pem = pathlib.Path(t) / "root.crt"
            pem.write_text("-----BEGIN CERTIFICATE-----\n")
            c = plans.make_plan(operating_system={"ca_certificates": [str(pem)]})
            self.assertEqual(
                d.tls_checks(c, dict(probe, with_ca="root"))[0].status, "PASS"
            )
            self.assertEqual(
                d.tls_checks(c, dict(probe, with_ca=False))[0].status, "FAIL"
            )
            # No row names the issuer of the certificate this network presents.
            inspecting = dict(probe, ok=True, error="", issuer="Example Proxy")
            for plan in (none, c):
                for with_ca in (None, False, "root", "partial"):
                    for r in d.tls_checks(plan, dict(inspecting, with_ca=with_ca)):
                        self.assertNotIn("Example Proxy", r.found)
                        self.assertNotIn("Example Proxy", str(r.fix))
        self.assertEqual(
            d.tls_checks(
                none, {"host": "github.com", "ok": True, "issuer": "DigiCert Inc"}
            )[0].status,
            "PASS",
        )
        self.assertEqual(d.tls_checks(none, {})[:1], [])
        # The build path's step 2 runs them, with its settings' certificates.
        checks = d.build_checks(none, self.facts(tls=probe), [])
        self.assertEqual(self.statuses(checks, "TLS"), ["FAIL"])
        self.assertEqual(self.statuses(checks, "CA certificate"), ["PASS"])

    def test_issuing_certificates_without_their_root_fail_on_every_base(self):
        """Certificates that verify this network only from an issuing certificate fail step 2's TLS row on every base,
        the default among them: pip, which installs the image's Python packages, trusts a chain only up to a root. The
        fix says what the root is and where to get it, and names no other base; the network's root passes on every
        base."""
        probe = {
            "host": "github.com",
            "ok": False,
            "error": "the certificate is not trusted (x)",
            "issuer": "Example Proxy",
            "with_ca": "partial",
        }
        with tempfile.TemporaryDirectory() as t:
            pem = pathlib.Path(t) / "issuing.crt"
            pem.write_text("-----BEGIN CERTIFICATE-----\n")
            default = plans.make_plan(operating_system={"ca_certificates": [str(pem)]})
            self.assertEqual(default.args["BASE_IMAGE"], image.DEFAULT_BASE)
            others = [
                plans.make_plan(
                    operating_system={"ca_certificates": [str(pem)], "base_image": base}
                )
                for base, *_ in image.BASES[1:]
            ]
            for plan in [default, *others]:
                with self.subTest(base=plan.args["BASE_IMAGE"]):
                    [row] = d.tls_checks(plan, probe)
                    self.assertEqual(
                        (row.name, row.status, row.found),
                        (
                            "TLS",
                            "FAIL",
                            "this network inspects TLS, and operating_system.ca_certificates names only issuing "
                            "certificates of its chain, not its root: pip, which installs the image's Python "
                            "packages, needs the root",
                        ),
                    )
                    self.assertEqual(
                        str(row.fix),
                        "get your network's root certificate, the top certificate of its chain (the one that signs "
                        "itself): ask your IT team for it, or export the top certificate from your browser's "
                        "certificate viewer; then set operating_system.ca_certificates in your build settings file "
                        f"to {d.CA_FILES}",
                    )
                    self.assertNotIn("Example Proxy", row.found + str(row.fix))
                    self.assertEqual(
                        d.tls_checks(plan, dict(probe, with_ca="root"))[0].status,
                        "PASS",
                    )

    def test_the_build_plan_s_own_rows(self):
        """A FAIL for each value the build cannot have, by its key, with the keys that pick another (but a key whose
        own row fails already, and the root certificates, which their own rows read); a WARN for each blank version
        taken as the newest release, with the pin as its fix; a WARN when the image is on this machine already.
        """
        why = "AMD's public folder of ROCm installers lists no build of ROCm 10.1.0"
        p = plans.make_plan()._replace(
            unresolved=(
                build_plan.Unresolved("rocm.version", why),
                build_plan.Unresolved("claude_agent_sdk.version", "no wheel"),
                build_plan.Unresolved(d.CA_KEY, "its file number 1 cannot be read"),
            ),
            newest=(("github_cli.version", "2.81.0"),),
        )
        rows = d.plan_checks(p, [], "~/b.yaml")
        self.assertEqual(
            [(r.name, r.status, r.found) for r in rows],
            [
                ("rocm.version", "FAIL", why),
                ("claude_agent_sdk.version", "FAIL", "no wheel"),
                (
                    "github_cli.version",
                    "WARN",
                    "not pinned; this build would take the newest release, 2.81.0",
                ),
            ],
        )
        self.assertEqual(
            str(rows[0].fix),
            "set rocm.version in ~/b.yaml to a release that folder lists, or set rocm.package_url in ~/b.yaml to the "
            "installer's full URL",
        )
        self.assertIn("set claude_agent_sdk.version in ~/b.yaml to", str(rows[1].fix))
        self.assertIn("claude_agent_sdk.package_url", str(rows[1].fix))
        self.assertEqual(
            str(rows[2].fix),
            'set github_cli.version in ~/b.yaml to "2.81.0", so a later build makes the same image',
        )
        # A key whose own row fails already gets no second row.
        own = [
            checks.Check(
                "rocm.version",
                "FAIL",
                "is not in the form this key takes",
                checks.Edit("rocm.version", "~/b.yaml", "10.1.0"),
            )
        ]
        self.assertEqual(
            [r.name for r in d.plan_checks(p, own, "~/b.yaml")],
            ["claude_agent_sdk.version", "github_cli.version"],
        )
        # The image is on this machine already: the build would replace it under the same tag.
        [here] = d.plan_checks(
            plans.make_plan(image={"tag": "t1"})._replace(unresolved=()),
            [],
            "~/b.yaml",
            True,
        )
        self.assertEqual(
            (here.name, here.status, here.found),
            (
                "image.tag",
                "WARN",
                f"{image.DEFAULT_NAME}:t1 is on this machine already; this build would replace it under the same tag",
            ),
        )
        self.assertIn('such as "t1-2"', str(here.fix))
        self.assertEqual(
            d.plan_checks(plans.make_plan()._replace(unresolved=()), [], "~/b.yaml"), []
        )

    def test_the_tls_probe_asks_the_site_the_environment_names(self):
        """The probe asks github.com at 443; RKMON_TLS_SITE (host:port) moves it, so the tests reach no public site: a
        port held closed answers no connection, the network row's FAIL."""
        self.assertEqual(d.tls_site({}), ("github.com", 443))
        self.assertEqual(d.tls_site({"RKMON_TLS_SITE": "example.com"}), d.TLS_SITE)
        closed = sandbox.closed_port(self)
        env = {"RKMON_TLS_SITE": f"127.0.0.1:{closed}"}
        self.assertEqual(d.tls_site(env), ("127.0.0.1", closed))
        probe = d.tls_probe([], timeout=5, environ=env)
        self.assertEqual((probe["host"], probe["ok"]), ("127.0.0.1", False))
        self.assertTrue(probe["error"].startswith("no connection"), probe)
        self.assertEqual(
            [(c.name, c.status) for c in d.tls_checks(plans.make_plan(), probe)],
            [("network", "FAIL")],
        )

    def test_the_host_size_is_the_environment_s_while_it_names_one(self):
        """This host's memory and CPUs, which the container's share is judged against: MemTotal in GiB and
        os.cpu_count(); RKMON_HOST_SIZE (GIB,CPUS) sets both, so the tests judge their files the same on every
        machine, and a value of another form is not taken."""
        with tempfile.TemporaryDirectory() as t:
            meminfo = pathlib.Path(t) / "meminfo"
            meminfo.write_text("MemTotal:       16777216 kB\nMemFree: 1024 kB\n")
            own = (16.0, os.cpu_count() or 0)
            self.assertEqual(d.host_size({}, str(meminfo)), own)
            for value, size in (
                ("11,4", (11.0, 4)),
                ("11.5,4", (11.5, 4)),
                ("11", own),
                ("11,", own),
                (",4", own),
                ("11,4.5", own),
                ("11, 4", own),
                ("eleven,4", own),
            ):
                with self.subTest(value=value):
                    self.assertEqual(
                        d.host_size({"RKMON_HOST_SIZE": value}, str(meminfo)), size
                    )

    def test_a_new_failure_after_a_swap(self):
        self.assertEqual(d.new_failures(["a"], ["a", "b"]), ["b"])
        # An older image's report cut a name at its first colon: it still covers the full names.
        self.assertEqual(
            d.new_failures(
                ["session login"], ["session login: gh", "session login: x"]
            ),
            [],
        )
        self.assertEqual(d.new_failures(["session login: gh"], ["session login"]), [])

    def test_the_doctor_s_failures_are_read_by_name(self):
        """rk doctor's FAIL rows by name (doctor_rows), as checks.render prints them (a found text over two lines too,
        whose second line starts with FAIL), and an older image's rows, without the indent.
        """
        from rkmon_lib import checks

        def fails(text):
            return [name for status, name, _ in d.doctor_rows(text) if status == "FAIL"]

        fix = checks.Command(("rk", "doctor", "--fresh"))
        rows = [
            checks.Check("a", "PASS", "b"),
            checks.Check("GitHub write login", "FAIL", "x", fix),
            checks.Check("c", "WARN", "d", fix),
            checks.Check(
                "ramdisk", "FAIL", "NOT MOUNTED;" + " FAIL  it: blocks" * 9, fix
            ),
        ]
        # the last row's found text goes on over a second line, which starts with FAIL: no row of its own
        self.assertIn("\n        FAIL  it: blocks", checks.render(rows))
        self.assertEqual(fails(checks.render(rows)), ["GitHub write login", "ramdisk"])
        text = "PASS  a: b\nFAIL  GitHub write login: x\n      fix: y\nWARN  c: d\nFAIL  ramdisk: NOT MOUNTED\n"
        self.assertEqual(fails(text), ["GitHub write login", "ramdisk"])

    def test_the_doctor_s_rows_with_the_lines_they_go_on_over(self):
        """Each row of rk doctor's report with its status, its name and its lines: as checks.render prints them (the
        found text's further lines and the fix), and as an older image prints them; a heading between rows is none.
        """
        from rkmon_lib import checks

        fix = checks.Command(("rk", "doctor", "--fresh"))
        rows = [
            checks.Check("a", "PASS", "b"),
            checks.Check("GitHub write login", "FAIL", "x " * 70, fix),
            checks.Check("c", "OFF", "d"),
        ]
        text = (
            "== the logins\n"
            + checks.render(rows[:2])
            + "\n== more\n"
            + checks.render(rows[2:])
        )
        self.assertEqual(
            d.doctor_rows(text),
            [(c.status, c.name, checks.render([c]).splitlines()) for c in rows],
        )
        # the second row's lines: the row, its found text's second line, its fix
        self.assertEqual(len(d.doctor_rows(text)[1][2]), 3)
        older = "PASS  a: b\nFAIL  GitHub write login: x\n      fix: y\n\n      not the row's\nWARN  c: d\n"
        self.assertEqual(
            d.doctor_rows(older),
            [
                ("PASS", "a", ["PASS  a: b"]),
                (
                    "FAIL",
                    "GitHub write login",
                    ["FAIL  GitHub write login: x", "      fix: y"],
                ),
                ("WARN", "c", ["WARN  c: d"]),
            ],
        )

    def test_the_doctor_s_rows_count_as_checks(self):
        """Each row reads as a check row does (its status and name by name), so outcome.tally counts a report."""
        from rkmon_lib import checks, outcome

        fix = checks.Command(("rk", "doctor", "--fresh"))
        rows = [
            checks.Check("a", "PASS", "b"),
            checks.Check("c", "WARN", "d", fix),
            checks.Check("e", "OFF", "f"),
            checks.Check("GitHub write login", "FAIL", "x", fix),
        ]
        read = d.doctor_rows(checks.render(rows))
        self.assertEqual(
            [(r.status, r.name) for r in read], [(c.status, c.name) for c in rows]
        )
        self.assertEqual(outcome.tally(read), outcome.tally(rows))
        self.assertEqual(outcome.tally(read)[0], "FAILED")
        self.assertEqual(outcome.tally(read[:3]), ("PASSED", "2 checks (1 warning)"))


class TestRequiredSettings(unittest.TestCase):
    def test_a_blank_required_key_fails_at_the_settings_step_with_the_line_to_add(self):
        self.assertFalse(d.required_checks(cfg()))
        blank = d.settings_checks(cfg(scope__labels=[]))
        fails = [c for c in blank if c.status == "FAIL"]
        self.assertEqual(len(fails), 1, fails)
        self.assertIn("repository.scope_labels is blank", fails[0].found)
        self.assertEqual(
            str(fails[0].fix),
            "set repository.scope_labels in your deploy settings file to a list of those labels",
        )
        self.assertTrue(d.required_checks(cfg(scope__labels=["", " "])))
        self.assertTrue(d.required_checks(cfg(rocke_path="")))
        # The template's placeholder is placeholder_checks' finding, not a second one here.
        self.assertFalse(d.required_checks(cfg(operator__github_login="<user_login>")))

    def test_the_template_leaves_only_what_has_no_default_to_fill_in(self):
        """The deploy settings template (settings_file.template) reads as a deploy settings file whose only refusals
        are the values it leaves to you, each a <...> placeholder: the repository, its folder and its scope's labels,
        and the GitHub username.
        """
        with tempfile.TemporaryDirectory() as t:
            f = pathlib.Path(t) / "deploy.yaml"
            f.write_text(settings_file.template(file_types.DEPLOY), encoding="utf-8")
            self.assertEqual(d.settings_kind(f), "deploy")
            c, rows = d.read(f)
        self.assertEqual(d.get(c, "scope.labels"), ["<label>"])
        self.assertEqual(
            sorted((r.name, r.status, r.found) for r in rows),
            [
                (
                    "integrations.github_cli.username",
                    "FAIL",
                    "still the template's <user_login>",
                ),
                ("repository.folder", "FAIL", "still the template's <folder>"),
                ("repository.name", "FAIL", "still the template's <org>/<repo>"),
                ("repository.scope_labels", "FAIL", "still the template's <label>"),
            ],
        )


if __name__ == "__main__":
    unittest.main()

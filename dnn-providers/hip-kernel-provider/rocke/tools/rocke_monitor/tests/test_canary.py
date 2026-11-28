# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The canary test's parts that run without a container (tests/container/canary_search.py, canary.py and
stage_canary.py): a canary's shape and its mask; the search's finds in a buffer, a stream, a folder, the processes and
an image's archive, each named by its place and never by its text; the copy of the build settings with a canary in
each download location, a build settings file `rkmon build` takes, the hosts of yours the search looks for, and the
canary test's refusal of a file this package does not build; and the canary stage's values, each one `rk secret-set`
takes. Every canary is made as the test runs: no file of the package holds one.
"""

import contextlib
import dataclasses
import hashlib
import io
import json
import os
import pathlib
import posixpath
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
import urllib.parse
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "container"))
sys.dont_write_bytecode = True

import canary  # noqa: E402
import canary_search as cs  # noqa: E402
import hostrun  # noqa: E402
import stage_canary  # noqa: E402
import test_build_plan as plans  # noqa: E402  (its build settings documents; its own tests run in their module)
from rkmon_lib import container_layout as L  # noqa: E402
from rkmon_lib import (  # noqa: E402
    build_plan,
    checks,
    deploy,
    deployment,
    file_types,
    gh,
    settings_file,
)
from rkmon_lib import secrets_registry as sr  # noqa: E402
from rkmon_lib.checks import Status  # noqa: E402

HOME = "/home/op"
SHA = "a" * 64


def lines() -> tuple[list, callable]:
    """A list, and an `out` for sweep that adds each line to it."""
    got = []
    return got, got.append


def tar(members: list, mode: str = "w", fmt=tarfile.PAX_FORMAT) -> bytes:
    """A tar archive of MEMBERS: (TarInfo, its bytes or None)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode=mode, format=fmt) as t:
        for info, data in members:
            t.addfile(info, io.BytesIO(data) if data is not None else None)
    return buf.getvalue()


def member(name: str, data: bytes = b"", **attrs) -> tuple:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    for k, v in attrs.items():
        setattr(info, k, v)
    return info, (data if info.isreg() else None)


def link(name: str, target: str) -> tuple:
    return member(name, type=tarfile.SYMTYPE, linkname=target)


def blob(digest: str, data: bytes) -> tuple:
    return member(f"blobs/sha256/{digest}", data)


def engine(folder: str, body: str) -> str:
    """A stand-in for the container engine in FOLDER: a Python script whose body is BODY (sys and os imported, its
    arguments in sys.argv): its path."""
    path = os.path.join(folder, "engine")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"#!{sys.executable}\nimport os, sys\n{body}\n")
    os.chmod(path, 0o755)
    return path


class TestCanaries(unittest.TestCase):
    def test_a_canary_is_made_fresh_of_its_kind(self):
        a, b = cs.make("monitor"), cs.make("monitor")
        self.assertNotEqual(a, b)
        for c in (a, b, cs.make("x"), cs.make("p" * 16)):
            self.assertRegex(c.encode(), cs.PATTERN)
            self.assertLessEqual(len(c), cs.LONGEST_BYTES)
        for bad in ("", "Monitor", "mon-1", "p" * 17, "écrit"):
            with self.subTest(kind=bad), self.assertRaises(ValueError):
                cs.make(bad)

    def test_shown_masks_each_canary_and_nothing_else(self):
        a, b = cs.make("monitor"), cs.make("writer")
        text = f"one {a} two {b}\nthree"
        self.assertEqual(
            cs.shown(text), "one <canary monitor> two <canary writer>\nthree"
        )
        self.assertEqual(cs.shown(text.encode()), cs.shown(text))
        self.assertTrue(cs.shown(a).startswith(cs.MASK))
        self.assertEqual(
            cs.shown("rkmon-canary-monitor-short"), "rkmon-canary-monitor-short"
        )


class TestFinder(unittest.TestCase):
    def test_canaries_by_kind_and_needles_by_their_words(self):
        mine, theirs = cs.make("monitor"), cs.make("writer")
        data = f"a {mine} b {theirs} c {HOME}/x".encode()
        finder = cs.Finder({"the home folder": HOME.encode()})
        self.assertEqual(
            finder.in_bytes(data),
            {"canary monitor", "canary writer", "the home folder"},
        )
        self.assertEqual(cs.Finder(kinds=["writer"]).in_bytes(data), {"canary writer"})
        self.assertEqual(finder.in_bytes(b"nothing here"), set())
        self.assertTrue(finder.hides(data))
        self.assertFalse(finder.hides(mine.encode()))

    def test_a_find_across_two_chunks(self):
        """A canary or a needle cut by the end of a chunk is found whole, wherever the cut is."""
        c = cs.make("monitor")
        data = b"x" * 40 + c.encode() + b"y" * 9 + HOME.encode() + b"z" * 40
        finder = cs.Finder({"the home folder": HOME.encode()})
        for size in (1, 2, 7, 13, 64, len(data)):
            with self.subTest(chunk=size):
                chunks = iter([data[i : i + size] for i in range(0, len(data), size)])
                found, total = finder.in_stream(lambda: next(chunks, b""))
                self.assertEqual(found, {"canary monitor", "the home folder"})
                self.assertEqual(total, len(data))


class TestSweep(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.root = pathlib.Path(d.name)

    def test_names_links_contents_and_counts(self):
        mine, theirs = cs.make("monitor"), cs.make("writer")
        (self.root / "a").mkdir()
        (self.root / "a" / "holds").write_text(f"x {mine} y")
        (self.root / "a" / f"named-{theirs}").write_text("clean")
        (self.root / "clean").write_text("nothing")
        os.symlink(f"/elsewhere/{mine}", self.root / "a" / "points")
        got, out = lines()
        places = cs.sweep([str(self.root)], cs.Finder(), out=out)
        self.assertEqual(places, 3)
        text = "\n".join(got)
        self.assertIn(f"FOUND canary monitor in {self.root}/a/holds", got)
        self.assertIn(
            f"FOUND canary writer in {self.root}/a/named-<canary writer> (its name)",
            got,
        )
        self.assertIn(
            f"FOUND canary monitor in {self.root}/a/points (where the link points)", got
        )
        self.assertNotIn(mine, text)
        self.assertNotIn(theirs, text)
        self.assertRegex(text, r"SWEPT files=3 bytes=\d+ older=0 ")
        self.assertEqual(got[-1], "CANARIES: found in 3 place(s)")
        # only the kinds asked for
        got, out = lines()
        self.assertEqual(
            cs.sweep([str(self.root)], cs.Finder(kinds=["writer"]), out=out), 1
        )

    def test_a_root_that_is_a_file_and_a_clean_tree(self):
        c = cs.make("writer")
        (self.root / "f").write_text(c)
        (self.root / "g").write_text("clean")
        got, out = lines()
        self.assertEqual(cs.sweep([str(self.root / "f")], cs.Finder(), out=out), 1)
        self.assertIn(f"FOUND canary writer in {self.root}/f", got)
        got, out = lines()
        self.assertEqual(cs.sweep([str(self.root / "g")], cs.Finder(), out=out), 0)
        self.assertEqual(got[-1], "CANARIES: none")

    def test_changed_since_reads_only_newer_files_but_every_name(self):
        """A file changed before the time is not read, by its content's time or its inode's (which no one sets back),
        but its name is still searched."""
        (self.root / "older").write_text(cs.make("monitor"))
        (self.root / f"named-{cs.make('monitor')}").write_text("clean")
        later = time.time() + 3600
        got, out = lines()
        places = cs.sweep([str(self.root)], cs.Finder(), changed_since=later, out=out)
        self.assertEqual(places, 1, got)  # the name
        self.assertRegex("\n".join(got), r"SWEPT files=0 bytes=0 older=2 ")
        past = time.time() - 3600
        # its content's time set back: its inode's is still new
        os.utime(self.root / "older", (past, past))
        got, out = lines()
        self.assertEqual(
            cs.sweep([str(self.root)], cs.Finder(), changed_since=past + 60, out=out), 2
        )

    @unittest.skipIf(os.geteuid() == 0, "root reads every file")
    def test_what_cannot_be_read_is_counted(self):
        (self.root / "shut").write_text(cs.make("writer"))
        os.chmod(self.root / "shut", 0)
        (self.root / "closed").mkdir()
        (self.root / "closed" / "x").write_text(cs.make("writer"))
        os.chmod(self.root / "closed", 0)
        self.addCleanup(os.chmod, self.root / "closed", stat.S_IRWXU)
        got, out = lines()
        self.assertEqual(cs.sweep([str(self.root)], cs.Finder(), out=out), 0)
        self.assertRegex("\n".join(got), r"unreadable=1 unlisted=1 ")

    @unittest.skipUnless(os.path.isdir("/proc/self"), "processes are read from /proc")
    def test_a_process_s_environment_and_command_line(self):
        env_canary, arg_canary = cs.make("session"), cs.make("argument")
        p = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)", arg_canary],
            env=dict(os.environ, RKMON_TEST_CANARY=env_canary),
        )
        self.addCleanup(p.wait)
        self.addCleanup(p.kill)
        # The child runs its command once it has started it (exec), which may come after Popen returns: until then
        # /proc shows this process's command line and environment. The canary argument in its command line says it
        # has.
        cmdline = pathlib.Path(f"/proc/{p.pid}/cmdline")
        deadline_s = time.monotonic() + 60
        while arg_canary.encode() not in cmdline.read_bytes():
            self.assertLess(time.monotonic(), deadline_s, "the child never started")
            time.sleep(0.01)
        got, out = lines()
        # the child's own files only: the search reads no other process of yours
        with mock.patch.object(cs.os, "listdir", return_value=[str(p.pid)]):
            cs.sweep(["/proc"], cs.Finder(kinds=["session", "argument"]), out=out)
        for words, part in (("session", "environ"), ("argument", "cmdline")):
            with self.subTest(part=part):
                said = f"FOUND canary {words} in /proc/{p.pid}/{part} ("
                self.assertTrue(any(g.startswith(said) for g in got), got)
        self.assertNotIn(env_canary, "\n".join(got))
        self.assertRegex("\n".join(got), r"processes=[1-9]\d* environments=[1-9]")

    def test_the_command(self):
        c = cs.make("writer")
        (self.root / "f").write_text(c)
        run = [sys.executable, "-I", cs.__file__, "sweep", "--root", str(self.root)]
        p = subprocess.run(run, capture_output=True, text=True)
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertNotIn(c, p.stdout)
        self.assertIn("CANARIES: found in 1 place(s)", p.stdout)
        p = subprocess.run(run + ["--kinds", "monitor"], capture_output=True, text=True)
        self.assertEqual(
            (p.returncode, p.stdout.splitlines()[-1]), (0, "CANARIES: none")
        )
        p = subprocess.run(run + ["--kinds", "Bad"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)


class TestTheBuildSettingsCopy(unittest.TestCase):
    """The canary test's copy of your build settings (canary.decorate, write_settings), the hosts of yours its search
    looks for (host_needles), and its refusal of a file this package does not build."""

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.folder = d.name
        self.sections = {
            "image": {"tag": "t1"},
            "rocm": {
                "version": "10.1.0-5",
                "package_url": "https://files.example.com/rocm/rocm-installer-10.1.0-5.run",
                "package_sha256": SHA,
            },
            "atlassian_cli": {
                "package_url": "https://files.example.com/cli/atlassian#v2",
                "package_sha256": SHA,
            },
            "github_cli": {
                "package_url": "https://files.example.com/gh/v2.60.0/gh_2.60.0_linux_amd64.tar.gz",
                "package_sha256": SHA,
            },
            "claude_agent_sdk": {
                "package_url": "https://files.example.com/wheels/claude_agent_sdk-0.3.0-py3-none-any.whl",
                "package_sha256": SHA,
            },
            "python_packages": {"package_index": "https://pypi.example.com/simple/"},
        }

    def plan(self, **over) -> tuple:
        """(the build settings document of the test's sections, each section of OVER in place of its own; its plan)."""
        doc = plans.build_doc(**dict(self.sections, **over))
        return doc, build_plan.plan(doc, plans.PACKAGE, plans.DATE, "x86_64")

    def test_every_download_has_a_form(self):
        """A download location the build gains gets a canary's form here, or the test says it has none."""
        self.assertEqual(
            set(canary.FRAGMENT_KEYS + canary.SEGMENT_KEYS),
            {d.key for d in build_plan.DOWNLOADS},
        )

    def test_each_download_of_yours_carries_a_canary_and_names_the_same_place(self):
        doc, p = self.plan()
        out, canaries, notes = canary.decorate(doc.values, self.folder, p, "amd64")
        self.assertEqual(notes, [])
        self.assertEqual(
            set(canaries),
            {
                "rocm",
                "atlassiancli",
                "githubcli",
                "claudeagentsdk",
                "pythonpackages",
                "operatingsystem",
            },
        )
        for key in canary.FRAGMENT_KEYS:
            with self.subTest(key=key):
                url, fragment = urllib.parse.urldefrag(canary._get(out, key))
                self.assertEqual(url, canary._get(doc.values, key).partition("#")[0])
                self.assertTrue(fragment.endswith(canaries[canary.kind_of(key)]))
        self.assertTrue(
            canary._get(out, "atlassian_cli.package_url").endswith(
                "#v2-" + canaries["atlassiancli"]
            )
        )
        for key, plain in (
            ("python_packages.package_index", "https://pypi.example.com/simple/"),
            (canary.MIRROR, canary.UBUNTU_ARCHIVE["amd64"]),
        ):
            with self.subTest(key=key):
                got = urllib.parse.urlsplit(canary._get(out, key))
                want = urllib.parse.urlsplit(plain)
                self.assertTrue(
                    got.path.startswith(f"/{canaries[canary.kind_of(key)]}/../")
                )
                self.assertEqual(got.netloc, want.netloc)
                self.assertEqual(
                    posixpath.normpath(got.path).rstrip("/"), want.path.rstrip("/")
                )
        self.assertNotIn(canaries["rocm"], json.dumps(doc.values))  # the original stays

    def test_the_copy_is_a_build_settings_file_only_you_read_that_builds_as_yours(self):
        """The copy passes the file's checks as the original does and makes the same build: the same build arguments,
        and each download of the original, the mirror added."""
        doc, p = self.plan()
        self.assertFalse(checks.failed(settings_file.validate(doc)))
        out, _c, _n = canary.decorate(doc.values, self.folder, p, "amd64")
        path = hostrun.write_settings(self.folder, out)
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        copied = settings_file.read(path, file_types.BUILD)
        self.assertEqual(copied.values, out)
        rows = settings_file.validate(copied)
        self.assertFalse(checks.failed(rows), checks.render(rows, "problems"))
        again = build_plan.plan(copied, plans.PACKAGE, plans.DATE, "x86_64")
        self.assertEqual(again.args, p.args)
        self.assertEqual(
            {s.id for s in again.secrets},
            {s.id for s in p.secrets} | {"package_mirror"},
        )

    def test_what_no_canary_goes_through(self):
        """A blank file URL, for which the build takes the package's public download, and a mirror on a base other
        than Ubuntu, which keeps its own mirror list: a NOTE names each, in build_plan.DOWNLOADS' order. A blank index
        takes the public one the build takes."""
        doc, p = self.plan(
            operating_system={"base_image": "rockylinux:9"},
            github_cli={"version": "2.60.0", "package_sha256": SHA},
            claude_agent_sdk={},
            python_packages={},
        )
        out, canaries, notes = canary.decorate(doc.values, self.folder, p, "amd64")
        self.assertEqual(set(canaries), {"rocm", "atlassiancli", "pythonpackages"})
        blank = {d.key: d.blank for d in build_plan.DOWNLOADS}
        self.assertEqual(
            notes,
            [
                f"{canary.MIRROR}: the base rockylinux:9 keeps its own mirror list, so no canary goes through it",
                *(
                    f"{key} is blank ({blank[key]}): no canary goes through it"
                    for key in (
                        "github_cli.package_url",
                        "claude_agent_sdk.package_url",
                    )
                ),
            ],
        )
        for key in (
            canary.MIRROR,
            "github_cli.package_url",
            "claude_agent_sdk.package_url",
        ):
            with self.subTest(key=key):
                self.assertIsNone(canary._get(out, key))
        index = urllib.parse.urlsplit(canary._get(out, "python_packages.package_index"))
        self.assertEqual(
            index.netloc, urllib.parse.urlsplit(build_plan.PUBLIC_PYPI).netloc
        )
        self.assertTrue(index.path.startswith(f"/{canaries['pythonpackages']}/../"))

    def test_the_root_certificates_go_under_a_canary_s_name(self):
        certs = []
        for n in (1, 2):
            path = pathlib.Path(self.folder) / f"site-{n}.pem"
            path.write_text(f"certificate {n}\n")
            certs.append(str(path))
        doc, p = self.plan(operating_system={"ca_certificates": certs})
        out, canaries, _notes = canary.decorate(doc.values, self.folder, p, "amd64")
        copies = canary._get(out, deployment.CA_KEY)
        c = canaries[canary.CERTIFICATE_KIND]
        self.assertEqual(
            [os.path.basename(x) for x in copies], [f"{c}-1.crt", f"{c}-2.crt"]
        )
        for src, copied in zip(certs, copies):
            self.assertEqual(
                pathlib.Path(copied).read_text(), pathlib.Path(src).read_text()
            )
        self.assertEqual(canary._get(doc.values, deployment.CA_KEY), certs)

    def test_the_hosts_searched_for(self):
        """The host of each download location of yours, a find named by its key and never its text; a public host, a
        host without a dot and a public default are not searched for, and a NOTE says so for each of yours, without
        its host."""
        doc, p = self.plan(
            operating_system={
                "package_mirror": "http://mirror.private.example.com/ubuntu"
            },
            github_cli={
                "package_url": deploy.gh_download(
                    "2.60.0", deploy.gh_file("2.60.0", "amd64")
                ),
                "package_sha256": SHA,
            },
            claude_agent_sdk={
                # a host's short name, written in two parts as canary.UBUNTU_ARCHIVE's addresses are
                "package_url": "http:/"
                + "/wheels/claude_agent_sdk-0.3.0-py3-none-any.whl",
                "package_sha256": SHA,
            },
            python_packages={},
        )
        needles, notes = canary.host_needles(p)
        words = canary.HOST_WORDS.format
        self.assertEqual(
            needles,
            {
                words(key=canary.MIRROR): b"mirror.private.example.com",
                words(key="rocm.package_url"): b"files.example.com",
                words(key="atlassian_cli.package_url"): b"files.example.com",
            },
        )
        self.assertEqual(
            [n.split()[0] for n in notes],
            ["github_cli.package_url", "claude_agent_sdk.package_url"],
        )
        self.assertIn("names a public host", notes[0])
        self.assertIn("names a host without a dot", notes[1])
        for host in ("github.com", "wheels", "example.com"):
            with self.subTest(host=host):
                self.assertNotIn(host, "\n".join(notes))
        self.assertEqual(
            cs.Finder(needles).in_bytes(
                b"deb http://mirror.private.example.com/ubuntu noble main"
            ),
            {words(key=canary.MIRROR)},
        )

    def test_a_wheel_on_pypi_names_a_public_host(self):
        """The SDK's wheel on PyPI (the address its index page gives) names PyPI's file host, which the pip in each
        venv of the image names too: a NOTE, and no search for it."""
        doc, p = self.plan(
            claude_agent_sdk={
                "package_url": f"https://{build_plan.PYPI_FILES_HOST}/packages/ab/cd/"
                "claude_agent_sdk-0.3.0-py3-none-manylinux_2_17_x86_64.whl",
                "package_sha256": SHA,
            }
        )
        needles, notes = canary.host_needles(p)
        self.assertNotIn(
            canary.HOST_WORDS.format(key="claude_agent_sdk.package_url"), needles
        )
        self.assertIn(
            "claude_agent_sdk.package_url names a public host, which the image may name anyway: it is not searched "
            "for",
            notes,
        )

    def test_each_file_download_by_its_sha256(self):
        """own_files: each file download given with its SHA-256, by that digest (in lower case): its location's key
        and its digest's key. One without a digest, a mirror and an index are not there.
        """
        digests = {
            "atlassian_cli": "2" * 64,
            "github_cli": "3" * 64,
            "claude_agent_sdk": "C" * 64,
        }
        over = {k: dict(self.sections[k], package_sha256=v) for k, v in digests.items()}
        over["rocm"] = {
            k: v for k, v in self.sections["rocm"].items() if k != "package_sha256"
        }
        _doc, p = self.plan(**over)
        self.assertEqual(
            canary.own_files(p),
            {
                v.lower(): (f"{k}.package_url", f"{k}.package_sha256")
                for k, v in digests.items()
            },
        )

    def test_a_file_this_package_does_not_build_is_refused_before_anything_runs(self):
        """A build settings file in layout 1: the reader's row, rkmon migrate its fix, and nothing built."""
        path = os.path.join(self.folder, "build.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"kind": file_types.BUILD.kind, "schema_version": 1}, f)
        out = io.StringIO()
        with mock.patch.object(canary.hostrun, "setup_output"):
            with mock.patch.object(canary, "build") as build:
                with contextlib.redirect_stdout(out):
                    rc = canary.main(
                        [
                            "--settings",
                            path,
                            "--engine",
                            "no-engine",
                            "--package",
                            str(ROOT),
                        ]
                    )
        self.assertEqual(rc, 2)
        self.assertIn("rkmon migrate", out.getvalue())
        self.assertIn(
            f"FAIL  {path}: no build settings file this package builds", out.getvalue()
        )
        build.assert_not_called()

    def test_the_build_is_of_the_canary_test_s_own_tag(self):
        """rkmon build takes the tag from the file alone: the private copy it builds names TAG (--tag) as image.tag,
        in place of your file's, and the build's command names no tag; the copy's folder goes after the build.
        """
        doc, _p = self.plan()
        self.assertEqual(doc.values["image"]["tag"], "t1")
        mine = tempfile.mkdtemp(dir=self.folder)
        path = hostrun.write_settings(mine, doc.values)
        built = []

        def build(argv, env, finder, folder):
            copied = argv[argv.index("--settings") + 1]
            built.append((argv, settings_file.read(copied, file_types.BUILD).values))
            return 1, []  # nothing built: nothing to search or remove

        out = io.StringIO()
        with mock.patch.object(canary.hostrun, "setup_output"):
            with mock.patch.object(
                canary.hostrun, "pick_engine", return_value="docker"
            ):
                with mock.patch.object(canary, "build", side_effect=build):
                    with contextlib.redirect_stdout(out):
                        rc = canary.main(
                            ["--settings", path, "--package", str(ROOT), "--tag", "t9"]
                        )
        self.assertEqual(rc, 1, out.getvalue())
        [(argv, values)] = built
        self.assertNotIn("--tag", argv)
        self.assertEqual(argv[-1], "--no-cache")
        self.assertEqual(values["image"]["tag"], "t9")
        self.assertFalse(os.path.exists(argv[argv.index("--settings") + 1]))

    def test_the_builder_s_home_folder(self):
        needles, note = canary.home_needles(HOME, ("/home/rkmon",), windows=False)
        self.assertEqual((needles, note), ({canary.HOME_WORDS: HOME.encode()}, ""))
        for home in ("", "/", "/root", "/home/rkmon/"):
            with self.subTest(home=home):
                needles, note = canary.home_needles(
                    home, ("/home/rkmon",), windows=False
                )
                self.assertEqual(needles, {})
                self.assertIn("is not searched for", note)
        needles, _note = canary.home_needles("C:\\Users\\op", windows=True)
        self.assertEqual(set(needles.values()), {b"C:\\Users\\op", b"C:/Users/op"})


class TestTheImageSearch(unittest.TestCase):
    def setUp(self):
        self.c = cs.make("rocm")
        self.finder = cs.Finder({canary.HOME_WORDS: HOME.encode()})

    def archive(self, marks: bytes = b"", raw: bytes = b"") -> bytes:
        """`docker save`'s layout: a gzip layer with finds in a file's content, two names (one with a canary beside a
        private value's host, one under the builder's home folder), a link and a header, a plain layer, a configuration,
        a manifest and an index, and RAW as a blob of its own.
        """
        found = member("etc/found", f"x {self.c} y".encode())
        headed = member("etc/headed", b"clean")
        headed[0].pax_headers = {"comment": self.c}
        one = tar(
            [
                member("etc", type=tarfile.DIRTYPE),
                found,
                headed,
                member(f"var/lists/mirror.example.com_{self.c}_dists", b"clean"),
                member(f"{HOME.lstrip('/')}/.cache", b"clean"),
                link("etc/points", f"/opt/{self.c}"),
            ],
            mode="w:gz",
        )
        two = tar([member("usr/clean", b"clean")], fmt=tarfile.USTAR_FORMAT)
        config = json.dumps({"history": [{"created_by": "RUN x"}], "x": marks.decode()})
        parts = [
            member("blobs", type=tarfile.DIRTYPE),
            blob("1" * 64, one),
            blob("2" * 64, two),
            blob("3" * 64, config.encode()),
            member("index.json", b'{"manifests": []}'),
            member("manifest.json", b"[]"),
            member("oci-layout", b'{"imageLayoutVersion": "1.0.0"}'),
        ]
        if raw:
            parts.append(blob("4" * 64, raw))
        return tar(parts)

    def test_every_layer_and_part(self):
        scan = canary.scan_archive(
            io.BytesIO(self.archive()), self.finder, canary.Scan()
        )
        self.assertEqual((scan.layers, scan.files, scan.unreadable), (2, 5, []))
        self.assertEqual(scan.marks, set())
        where = "layer " + "1" * 12
        self.assertEqual(
            sorted(scan.found),
            sorted(
                [
                    ("canary rocm", f"{where}: /etc/found"),
                    ("canary rocm", f"{where}: /etc/headed (its extended headers)"),
                    (
                        "canary rocm",
                        f"{where}: /var/lists/(a part that holds a find) (its name)",
                    ),
                    (
                        canary.HOME_WORDS,
                        f"{where}: (a name that holds a find) (its name)",
                    ),
                    ("canary rocm", f"{where}: /etc/points (where the link points)"),
                ]
            ),
        )
        self.assertNotIn(self.c, repr(scan))
        self.assertNotIn("mirror.example.com", repr(scan))

    def test_the_provenance_marks_and_a_raw_blob(self):
        raw = b"\x00\x01" * 300 + self.c.encode() + b"\x02" * 300
        scan = canary.scan_archive(
            io.BytesIO(self.archive(marks=b"vcs:source", raw=raw)),
            self.finder,
            canary.Scan(),
        )
        self.assertEqual(scan.marks, {"vcs:source"})
        self.assertIn(("canary rocm", "blobs/sha256/" + "4" * 64), scan.found)

    def test_a_download_as_it_came_names_its_own_host(self):
        """A file of the image whose SHA-256 is a download's (own_files: the atlassian CLI's binary, installed as it
        came) may name that download's host: a note, no find. Its host in any other file, and another download's host
        in that file, are finds. The digest covers the whole file, past the search's first chunk.
        """
        mine = canary.HOST_WORDS.format(key="atlassian_cli.package_url")
        theirs = canary.HOST_WORDS.format(key="rocm.package_url")
        finder = cs.Finder({mine: b"cli.example.com", theirs: b"rocm.example.org"})
        tool = (
            b"\x7fELF" + b"\x00" * canary.READ_BYTES + b" update site cli.example.com"
        )
        own = {
            hashlib.sha256(tool).hexdigest(): (
                "atlassian_cli.package_url",
                "atlassian_cli.package_sha256",
            )
        }
        layer = tar(
            [
                member("usr/local/bin/atlassian", tool),
                member("etc/copy", b"x cli.example.com"),
                member("usr/local/bin/other", tool + b" rocm.example.org"),
            ],
            mode="w:gz",
        )
        scan = canary.scan_archive(
            io.BytesIO(tar([blob("1" * 64, layer)])), finder, canary.Scan(own=own)
        )
        where = "layer " + "1" * 12
        self.assertEqual(
            sorted(scan.found),
            sorted(
                [
                    (mine, f"{where}: /etc/copy"),
                    (mine, f"{where}: /usr/local/bin/other"),
                    (theirs, f"{where}: /usr/local/bin/other"),
                ]
            ),
        )
        self.assertEqual(
            scan.notes,
            [
                f"the host of atlassian_cli.package_url is in {where}: /usr/local/bin/atlassian, which is that "
                "download as it came (its SHA-256 is atlassian_cli.package_sha256): it names its own host, and that "
                "is no find"
            ],
        )
        self.assertNotIn("example", " ".join(scan.notes))
        plain = canary.scan_archive(
            io.BytesIO(tar([blob("1" * 64, layer)])), finder, canary.Scan()
        )
        self.assertIn((mine, f"{where}: /usr/local/bin/atlassian"), plain.found)
        self.assertEqual(plain.notes, [])

    def test_docker_save_streamed_and_its_exit_code(self):
        with tempfile.TemporaryDirectory() as d:
            archive = os.path.join(d, "image.tar")
            with open(archive, "wb") as f:
                f.write(self.archive())
            run = engine(
                d,
                "sys.stdout.buffer.write(open(os.environ['ARCHIVE'], 'rb').read()); "
                "sys.exit(int(os.environ.get('SAVE_EXIT', '0')))",
            )
            with mock.patch.dict(os.environ, {"ARCHIVE": archive}):
                rc, scan, _err = canary.save_scan(run, "x:t", self.finder)
                self.assertEqual((rc, scan.layers, len(scan.found)), (0, 2, 5))
                with mock.patch.dict(os.environ, {"SAVE_EXIT": "3"}):
                    self.assertEqual(canary.save_scan(run, "x:t", self.finder)[0], 3)
            # an archive cut short in its second layer keeps what it found in the first, and says what it could not
            # read
            with tarfile.open(archive) as t:
                cut = t.getmember("blobs/sha256/" + "2" * 64).offset_data + 600
            os.truncate(archive, cut)
            with mock.patch.dict(os.environ, {"ARCHIVE": archive}):
                rc, scan, _err = canary.save_scan(run, "x:t", self.finder)
            self.assertEqual((rc, len(scan.found)), (0, 5))
            self.assertTrue(scan.unreadable)
            self.assertNotIn(self.c, repr(scan))

    def test_the_build_s_output_without_the_lines_that_hold_a_find(self):
        """A line with a canary is not shown at all: the canary sits in one of your private values, which the rest of
        the line may hold."""
        with tempfile.TemporaryDirectory() as folder:
            code = (
                "import sys; print('one', sys.argv[1]); print(sys.argv[2] + '/build_settings.json'); "
                "print(sys.argv[3] + '/x')"
            )
            argv = [sys.executable, "-c", code, self.c, folder, HOME]
            shown = io.StringIO()
            with contextlib.redirect_stdout(shown):
                rc, found = canary.build(argv, dict(os.environ), self.finder, folder)
        self.assertEqual(rc, 0)
        self.assertEqual(
            found,
            [
                ("canary rocm", "the build's output, line 1"),
                (canary.HOME_WORDS, "the build's output, line 3"),
            ],
        )
        self.assertEqual(
            shown.getvalue().splitlines(),
            [
                "   | (line 1, which holds canary rocm: not shown)",
                "   | <the canary test's folder>/build_settings.json",
                f"   | (line 3, which holds {canary.HOME_WORDS}: not shown)",
            ],
        )

    def test_the_history_and_configuration_the_engine_shows(self):
        with tempfile.TemporaryDirectory() as d:
            run = engine(
                d,
                "print(os.environ['CANARY'] if sys.argv[1] == 'history' else '[]'); "
                "sys.exit(int(os.environ.get('INSPECT_EXIT', '0')) if sys.argv[1] == 'image' else 0)",
            )
            with mock.patch.dict(os.environ, {"CANARY": self.c}):
                self.assertEqual(
                    canary.engine_finds(run, "x:t", self.finder),
                    [("canary rocm", "its history")],
                )
                with mock.patch.dict(os.environ, {"INSPECT_EXIT": "1"}):
                    found = canary.engine_finds(run, "x:t", self.finder)
        self.assertEqual(
            found[-1],
            (
                "nothing it could read",
                "its configuration (image inspect): image exited 1",
            ),
        )


class TestTheCanaryStage(unittest.TestCase):
    def test_each_keeper_has_its_user(self):
        self.assertEqual(set(stage_canary.USERS), set(sr.KEEPERS))
        self.assertEqual(stage_canary.USERS[sr.WRITER], L.WRITER.name)

    def test_each_value_has_its_secret_s_shape(self):
        """rk secret-set takes each planted value: it has the shape of its secret (a login's kept in a file: the
        claude login's key), and holds the canary."""
        c = cs.make("monitor")
        some = next(e for e in sr.entries({}) if e.shape != sr.LOGIN)
        for shape in sr.SHAPES:
            with self.subTest(shape=shape):
                e = (
                    sr.entry({}, sr.CLAUDE_LOGIN)
                    if shape == sr.LOGIN
                    else dataclasses.replace(some, shape=shape)
                )
                v = stage_canary.value(e, c)
                self.assertIn(c, v)
                self.assertEqual(sr.check_value(e, v).status, Status.PASS)
        self.assertIn(c, stage_canary.login_text(gh.HOSTS_FILE, c))
        self.assertEqual(
            json.loads(stage_canary.login_text("token.json", c)), {"token": c}
        )

    def test_each_entry_gets_its_keeper_s_canary(self):
        """Each entry of the registry gets a canary of its keeper's, written as that keeper: a login kept in a folder in
        its files, and every other one through rk secret-set, on stdin, the claude login's key among them (as `rkmon
        login claude` sets it). Each place is one the search must find its canary in."""
        canaries = {k: cs.make(k) for k in stage_canary.USERS}
        cfg = {
            "paths": {"working_dir": L.WORK_DIR, "secrets_dir": L.SECRETS_DIR},
            "model_bindings": {
                stage_canary.BINDING: {
                    "api_key_secret": stage_canary.BINDING + sr.BINDING_KEY_SUFFIX
                }
            },
        }
        calls = []

        class Fake:
            def check(self, need, what, user, cmd, env=None, timeout=600, stdin=None):
                calls.append((user, cmd, stdin))
                return ""

        with mock.patch.object(
            stage_canary, "configure", return_value=cfg
        ), mock.patch.object(
            stage_canary.pwd, "getpwnam", return_value=mock.Mock(pw_dir="/nonexistent")
        ):
            _, places = stage_canary.plant(Fake(), canaries)
        set_by = {
            cmd.rsplit(" ", 1)[1]: (user, stdin)
            for user, cmd, stdin in calls
            if f"{stage_canary.ENTRY} rk secret-set " in cmd
        }
        for e in sr.entries(cfg):
            with self.subTest(entry=e.name):
                # its own place, which the load derives in the container
                where = sr.home(cfg, e, stage_canary.iso.SESSION_ENV)
                user = stage_canary.USERS[e.keeper]
                if sr.held_files(e) is None:
                    self.assertEqual(set_by[e.name][0], user)
                    self.assertIn(canaries[e.keeper], set_by[e.name][1])
                    self.assertEqual(places[where], e.keeper)
                else:
                    for file in e.files:
                        self.assertEqual(places[f"{where}/{file}"], e.keeper)
        self.assertIn(sr.CLAUDE_LOGIN, set_by)

    def test_the_configuration(self):
        """The configuration names the container's secrets folder and a model binding whose key is a secret of the
        monitor's, and holds the canary in its private values."""
        c = cs.make("monitor")
        with tempfile.TemporaryDirectory() as d:
            config = pathlib.Path(d) / "config.json"
            config.write_text(json.dumps({"paths": {"working_dir": L.WORK_DIR}}))
            with mock.patch.object(stage_canary, "CONFIG", str(config)):
                cfg = stage_canary.configure(c)
            self.assertEqual(json.loads(config.read_text()), cfg)
        self.assertEqual(cfg["paths"]["secrets_dir"], L.SECRETS_DIR)
        key = sr.entry(cfg, stage_canary.BINDING + sr.BINDING_KEY_SUFFIX)
        self.assertIsNotNone(key)
        self.assertEqual(key.keeper, sr.MONITOR)
        for dotted in stage_canary.CONFIG_URLS + stage_canary.CONFIG_HOSTS:
            with self.subTest(key=dotted):
                self.assertIn(c, canary._get(cfg, dotted))


if __name__ == "__main__":
    unittest.main()

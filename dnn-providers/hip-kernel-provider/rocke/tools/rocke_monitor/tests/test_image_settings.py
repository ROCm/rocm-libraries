# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The image's build (docker/Dockerfile, docker/param.sh, docker/fetch.sh, docker/masked.sh, docker/install_sdk.sh,
docker/pkgs.sh, and docker/rkmon's build path over rkmon_lib/build_plan.py): the Dockerfile's arguments and build
secrets are the plan's, none with a default of its own; every image full; the SDK's wheel checked against its
SHA-256 before anything is installed; the one `docker build` a build settings file makes, with the flags the plan
lists and no other, each download location a build secret, never printed (a tool's lines show it by its secret's
name), its root certificates and its build record build contexts of their own; a file in another layout or of
another kind refused; the labels a deployment checks an image against (fit); and no ROCm mode on the deploy side.
"""

import contextlib
import functools
import hashlib
import http.server
import importlib.machinery
import importlib.util
import io
import json
import os
import pathlib
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.parse
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import sandbox  # noqa: E402
import test_build_plan as plans  # noqa: E402  (its wheel of the SDK; its own tests run in their module)
from rendered import finished, one_line  # noqa: E402
from rkmon_lib import (
    build_plan,
    deploy,
    deploy_plan,
    deployment,
    doctor,
    file_types,
    image,
    renames,
    settings_file,
)  # noqa: E402

DOCKER = ROOT / "docker"
URL = "https://installers.example.com/rocm/linux-installers"
RUN = URL + "/rocm-installer-10.1.0-5.run"
MIRROR = "https://pypi.mirror.example.com/simple"
CLI = "https://cli.example.invalid/private/atlassian"
SHA = "a" * 64
GH = "2.81.0"
SDK = "0.2.181"
CLAUDE_CODE = "2.1.99"
WHEEL = f"claude_agent_sdk-{SDK}-py3-none-any.whl"
# The guard each build step reads a build argument with (docker/Dockerfile): a build that was not given it stops there.
GIVEN = "not given (rkmon build gives it)"
# The scripts of the build's steps that read a build secret themselves (docker/param.sh).
STEP_SCRIPTS = ("pkgs.sh", "install_rocm.sh", "install_atlassian.sh", "install_sdk.sh")
# A build settings file in layout 1, older than this package's.
BUILD1 = (
    "kind: build_settings\n"
    "schema_version: 1\n"
    "image:\n"
    "  tag: t1\n"
    "  base: ubuntu:24.04\n"
    "rocm:\n"
    '  version: "10.1.0-5"\n'
    f'  package_url: "{RUN}"\n'
    "atlassian_cli:\n"
    f'  package_url: "{CLI}"\n'
    f"  package_sha256: {SHA}\n"
    "python_packages:\n"
    f'  package_index: "{MIRROR}"\n'
)


def build_settings(wheel: str) -> str:
    """A build settings file in this package's layout that builds with nothing from the network but WHEEL, the
    address of the SDK's wheel: every version pinned with its SHA-256, and each download location named but gh's,
    which the build takes from GitHub's releases (gh_public)."""
    return (
        "kind: build_settings\n"
        "schema_version: 2\n"
        "image:\n"
        "  tag: t1\n"
        "rocm:\n"
        '  version: "10.1.0-5"\n'
        f'  package_url: "{RUN}"\n'
        f"  package_sha256: {SHA}\n"
        "atlassian_cli:\n"
        '  version: "1.2.3"\n'
        f'  package_url: "{CLI}"\n'
        f"  package_sha256: {SHA}\n"
        "github_cli:\n"
        f'  version: "{GH}"\n'
        f"  package_sha256: {SHA}\n"
        "claude_agent_sdk:\n"
        f'  version: "{SDK}"\n'
        f'  package_url: "{wheel}"\n'
        f"  package_sha256: {SHA}\n"
        "python_packages:\n"
        f'  package_index: "{MIRROR}"\n'
    )


def gh_public(version: str = GH) -> str:
    """GitHub's release archive of gh VERSION for this machine: the address a build is given for a blank
    github_cli.package_url, public, so shown in full."""
    arch = deploy.GH_ARCH[platform.machine().lower()]
    return deploy.gh_download(version, deploy.gh_file(version, arch))


LABELS = {
    "org.rocke.package.version": "2026.11.13",
    "org.rocke.build.date": "2026-11-13T00:00:00Z",
    "org.rocke.base": "ubuntu:26.04",
    "org.rocke.rocm.version": "10.1.0-5",
}


def dockerfile() -> str:
    return (DOCKER / "Dockerfile").read_text(encoding="utf-8")


def serve_folder(test: unittest.TestCase, folder: pathlib.Path, files: dict) -> str:
    """FILES ({name: text or bytes}) in FOLDER, served over HTTP on 127.0.0.1 until TEST ends: the folder's
    address."""
    folder.mkdir()
    for name, body in files.items():
        if isinstance(body, bytes):
            (folder / name).write_bytes(body)
        else:
            (folder / name).write_text(body)

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Quiet, directory=str(folder))
    )
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    test.addCleanup(srv.server_close)
    test.addCleanup(srv.shutdown)
    return f"http://127.0.0.1:{srv.server_address[1]}"


def serve_answers(test: unittest.TestCase, answers: list) -> tuple:
    """An HTTP server on 127.0.0.1 until TEST ends, which answers its Nth request with ANSWERS[N], and every request
    after them with the last: an HTTP status with no body, a file's bytes (200), or None, the connection closed with no
    answer. Its address, and the list each request's path goes to."""
    asked = []

    class Answers(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            asked.append(self.path)
            answer = answers[min(len(asked), len(answers)) - 1]
            if answer is None:
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            body = answer if isinstance(answer, bytes) else b""
            self.send_response(200 if isinstance(answer, bytes) else answer)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Answers)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    test.addCleanup(srv.server_close)
    test.addCleanup(srv.shutdown)
    return f"http://127.0.0.1:{srv.server_address[1]}", asked


class TestTheParameters(unittest.TestCase):
    """docker/Dockerfile takes rkmon_lib/build_plan.py's build arguments and build secrets and no other, none with a
    default of its own: the plan is their one home."""

    def test_every_argument_is_the_plan_s_without_a_default(self):
        """An ARG for each of build_plan.ARGS and no other, none with a default: each step reads its arguments through
        the guard, and the base stage takes BASE_IMAGE, so a build that is not given one stops.
        """
        text = dockerfile()
        args = re.findall(r"(?m)^ARG ([A-Z0-9_]+)(=.*)?$", text)
        self.assertEqual({n for n, _ in args}, {a.name for a in build_plan.ARGS})
        self.assertEqual([n for n, default in args if default], [])
        self.assertIn("FROM ${BASE_IMAGE} AS base\n", text)
        for a in build_plan.ARGS[1:]:
            with self.subTest(arg=a.name):
                self.assertIn(f'"${{{a.name}?{GIVEN}}}"', text)

    def test_a_step_stops_on_an_argument_it_was_not_given(self):
        """The guard: an argument the build was not given stops the step, naming it; a blank one, which rkmon build
        gives for a value blank by design (another release's clang commit), does not."""
        guard = f': "${{ROCM_CLANG_COMMIT?{GIVEN}}}"'
        for env, said in (
            ({}, f"ROCM_CLANG_COMMIT: {GIVEN}"),
            ({"ROCM_CLANG_COMMIT": ""}, ""),
        ):
            with self.subTest(env=env):
                p = subprocess.run(
                    ["sh", "-c", guard],
                    capture_output=True,
                    text=True,
                    env={"PATH": "/usr/bin:/bin", **env},
                )
                if said:
                    self.assertNotEqual(p.returncode, 0)
                    self.assertIn(said, p.stderr)
                else:
                    self.assertEqual((p.returncode, p.stderr), (0, ""))

    def test_every_location_is_a_secret_and_never_an_argument_or_a_label(self):
        text = dockerfile()
        mounted = set(re.findall(r"--mount=type=secret,id=([a-z_]+)", text))
        self.assertEqual(mounted, {d.id for d in build_plan.DOWNLOADS})
        # Each is read with param.sh: in the Dockerfile's step, or by the script the step runs.
        steps = text + "".join(
            (DOCKER / f).read_text(encoding="utf-8") for f in STEP_SCRIPTS
        )
        for d in build_plan.DOWNLOADS:
            with self.subTest(secret=d.id):
                self.assertRegex(steps, rf"param\.sh\"? (--optional )?{d.id}\)")
                self.assertNotRegex(
                    text, rf"(?m)^ARG {build_plan.secret_variable(d.id)}\b"
                )
        # The image's labels are the build's --label flags, from the plan; the Dockerfile has none.
        self.assertNotRegex(text, r"(?m)^LABEL ")

    def test_every_image_is_full(self):
        """Every part is a stage built every time: ROCm, the venvs, Claude Code, the atlassian CLI and gh; the image
        builds on the ROCm stage and copies each other part in. No parameter chooses a part. Two more stages are build
        contexts the build gives: the root certificates the base trusts, empty unless the build gives them
        (build_plan.CA_CONTEXT), and the build record (build_plan.RECORD_CONTEXT), which the image keeps before its
        code layer.
        """
        text = dockerfile()
        stages = dict((b, a) for a, b in re.findall(r"(?m)^FROM (\S+) AS (\S+)$", text))
        self.assertEqual(
            set(stages),
            {
                "base",
                "rocm",
                "venvs",
                "claude",
                "atlassian",
                "gh",
                "final",
                build_plan.CA_CONTEXT,
                build_plan.RECORD_CONTEXT,
            },
        )
        self.assertEqual(stages[build_plan.CA_CONTEXT], "scratch")
        self.assertEqual(stages[build_plan.RECORD_CONTEXT], "scratch")
        self.assertEqual(stages["final"], "rocm")
        for part in ("venvs", "claude", "atlassian", "gh"):
            self.assertEqual(stages[part], "base")
            self.assertIn(f"COPY --from={part} /out/ /", text)
        self.assertNotRegex(text, r"(?m)^ARG (ROCM|CLAUDE|ATLASSIAN)(=|$)")
        record = (
            f"COPY --from={build_plan.RECORD_CONTEXT} --chmod=644 {build_plan.RECORD_FILE} "
            f"{build_plan.RECORD_PATH}\n"
        )
        self.assertIn(record, text)
        self.assertLess(text.index(record), text.index("COPY . /opt/rocke_monitor\n"))

    def test_the_public_defaults_are_the_plan_s_alone(self):
        """AMD's public folder of ROCm installers and the public package index have one home, build_plan, which gives
        the build each as a build secret: no file of the build names them."""
        for name in ("Dockerfile", "param.sh", "fetch.sh", "masked.sh", *STEP_SCRIPTS):
            text = (DOCKER / name).read_text(encoding="utf-8")
            for public in (
                build_plan.PUBLIC_ROCM_INSTALLERS,
                build_plan.PUBLIC_PYPI,
                urllib.parse.urlsplit(build_plan.PUBLIC_ROCM_INSTALLERS).hostname,
                urllib.parse.urlsplit(build_plan.PUBLIC_PYPI).hostname,
            ):
                with self.subTest(file=name, public=public):
                    self.assertNotIn(public, text)

    def test_every_settings_key_the_build_names_is_one(self):
        """A step's refusal names the build settings to change: each one any file of the build names is a key of the
        build settings file (schema/build_settings.schema.json)."""
        schema = json.loads(
            (ROOT / "schema" / "build_settings.schema.json").read_text(encoding="utf-8")
        )
        keys = {path for path, _node, _required in settings_file.schema_keys(schema)}
        sections = sorted({k.split(".", 1)[0] for k in keys if "." in k})
        # a file's name in a path (/etc/rkmon/rocm.json) is none
        named = re.compile(rf"(?<![\w./-])(?:{'|'.join(sections)})\.[a-z0-9_]+")
        for f in [DOCKER / "Dockerfile", *sorted(DOCKER.glob("*.sh"))]:
            with self.subTest(file=f.name):
                found = set(named.findall(f.read_text(encoding="utf-8")))
                self.assertEqual(sorted(found - keys), [])


class TestParamSh(unittest.TestCase):
    """docker/param.sh, a build step's read of a build secret: the build is given each one it needs (rkmon build), so
    a step has no default of its own."""

    def read(self, d: pathlib.Path, *args) -> tuple:
        p = subprocess.run(
            ["sh", str(DOCKER / "param.sh"), *args],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "SECRETS_DIR": str(d)},
        )
        return p.returncode, p.stdout, p.stderr

    def test_a_secret_the_build_was_not_given_stops_the_step(self):
        """None given, or an empty one: exit 2, naming the secret and rkmon build. A step that can do without one
        reads it with --optional, and decides itself what none means."""
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            refused = (
                2,
                "",
                "param.sh: the build was given no build secret rocm_package_url: build the image with rkmon build, "
                "which gives each one\n",
            )
            self.assertEqual(self.read(d, "rocm_package_url"), refused)
            (d / "rocm_package_url").write_text("")
            self.assertEqual(self.read(d, "rocm_package_url"), refused)
            self.assertEqual(
                self.read(d, "--optional", "package_mirror"), (0, "\n", "")
            )

    def test_the_secret_s_first_line(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "rocm_package_url").write_text(RUN + "  \nsecond line\n")
            for args in (("rocm_package_url",), ("--optional", "rocm_package_url")):
                with self.subTest(args=args):
                    self.assertEqual(self.read(d, *args), (0, RUN + "\n", ""))


# A curl that records its arguments and its standard input (CURL_LOG.argv, CURL_LOG.stdin), names a host on its standard
# error as curl does, and exits with CURL_EXIT (HTTP 404 its status for 22).
FAKE_CURL = """#!/bin/sh
printf '%s\\n' "$@" > "$CURL_LOG.argv"
cat > "$CURL_LOG.stdin"
echo "curl: ($CURL_EXIT) the host private.example.com did not answer" >&2
[ "$CURL_EXIT" = 22 ] && printf 404
exit "$CURL_EXIT"
"""


class TestFetchSh(unittest.TestCase):
    """docker/fetch.sh, the build's download: the address on curl's standard input, never on its command line; a
    passing failure tried again, as the build's downloads in Python are (rkmon_lib/deploy.py); and a failure in fixed
    words for curl's exit code, which name no host."""

    def fetch(self, url: str, file, path: str, **env) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["sh", str(DOCKER / "fetch.sh"), str(file)],
            input=f"{url}\n" if url else "",
            capture_output=True,
            text=True,
            env={"PATH": path, **env},
        )

    def fake(self, d: pathlib.Path) -> str:
        (d / "curl").write_text(FAKE_CURL)
        (d / "curl").chmod(0o755)
        return f"{d}:/usr/bin:/bin"

    def test_the_address_goes_on_curl_s_standard_input(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            path = self.fake(d)
            url = 'https://private.example.com/a "b"\\c?sig=1#part'
            p = self.fetch(url, d / "f", path, CURL_LOG=str(d / "log"), CURL_EXIT="0")
            self.assertEqual((p.returncode, p.stdout, p.stderr), (0, "", ""))
            argv = (d / "log.argv").read_text().splitlines()
            self.assertNotIn("private.example.com", " ".join(argv))
            self.assertIn("--config", argv)
            self.assertEqual(argv[argv.index("--config") + 1], "-")
            self.assertEqual(
                (d / "log.stdin").read_text(),
                'url = "https://private.example.com/a \\"b\\"\\\\c?sig=1#part"\n',
            )

    def test_a_failure_says_why_in_fixed_words(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            path = self.fake(d)
            for code, why in (
                (3, "the address is no URL curl can read"),
                (5, "the proxy's name does not resolve"),
                (6, "the host's name does not resolve"),
                (7, "no connection to the host: it refused, or cannot be reached"),
                (18, "the download ended before the whole file came"),
                (22, "the server answered HTTP 404"),
                (23, "the file could not be written here"),
                (28, "it timed out"),
                (35, "the TLS handshake failed"),
                (47, "too many redirects"),
                (52, "the server answered nothing"),
                (55, "the connection broke during the download"),
                (56, "the connection broke during the download"),
                (
                    60,
                    "the server's certificate is not one the build trusts (a network that inspects TLS: name its "
                    "root certificates in operating_system.ca_certificates)",
                ),
                (99, "curl could not download it"),
            ):
                with self.subTest(code=code):
                    p = self.fetch(
                        "https://private.example.com/x",
                        d / "f",
                        path,
                        CURL_LOG=str(d / "log"),
                        CURL_EXIT=str(code),
                        FETCH_PAUSE_SECONDS="0",
                    )
                    self.assertEqual(p.returncode, code)
                    self.assertEqual(p.stdout, "")
                    self.assertEqual(p.stderr, f"{why}, curl exit {code}\n")

    def test_no_address(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.fetch("", pathlib.Path(d) / "f", "/usr/bin:/bin")
        self.assertEqual(p.returncode, 2)
        self.assertEqual(p.stderr, "no address came on the standard input\n")

    def test_a_download_and_a_refusal_from_a_server(self):
        if not shutil.which("curl"):
            self.skipTest("no curl")
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            base = serve_folder(self, d / "web", {"x.run": "the installer\n"})
            p = self.fetch(f"{base}/x.run?sig=1#part", d / "got", "/usr/bin:/bin")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual((d / "got").read_text(), "the installer\n")
            p = self.fetch(f"{base}/none.run", d / "none", "/usr/bin:/bin")
        self.assertEqual(p.returncode, 22)
        self.assertEqual(p.stderr, "the server answered HTTP 404, curl exit 22\n")

    def test_a_download_that_fails_twice_then_succeeds(self):
        """A connection closed with no answer, then a server's passing failure (HTTP 503), then the file: three
        attempts, the file downloaded, and nothing said."""
        if not shutil.which("curl"):
            self.skipTest("no curl")
        base, asked = serve_answers(self, [None, 503, b"the installer\n"])
        with tempfile.TemporaryDirectory() as d:
            got = pathlib.Path(d) / "got"
            p = self.fetch(
                f"{base}/x.run", got, "/usr/bin:/bin", FETCH_PAUSE_SECONDS="0"
            )
            self.assertEqual((p.returncode, p.stdout, p.stderr), (0, "", ""))
            self.assertEqual(got.read_text(), "the installer\n")
        self.assertEqual(asked, ["/x.run"] * 3)

    def test_a_failure_that_stands_is_tried_once(self):
        """A refusal (HTTP 404) is tried once; a passing failure that lasts, as many times as the build's downloads
        in Python are (deploy.FETCH_ATTEMPTS), then the line says why."""
        if not shutil.which("curl"):
            self.skipTest("no curl")
        for answer, tries in ((404, 1), (503, deploy.FETCH_ATTEMPTS)):
            with self.subTest(answer=answer):
                base, asked = serve_answers(self, [answer])
                with tempfile.TemporaryDirectory() as d:
                    p = self.fetch(
                        f"{base}/x.run",
                        pathlib.Path(d) / "got",
                        "/usr/bin:/bin",
                        FETCH_PAUSE_SECONDS="0",
                    )
                self.assertEqual(
                    (p.returncode, p.stderr),
                    (22, f"the server answered HTTP {answer}, curl exit 22\n"),
                )
                self.assertEqual(len(asked), tries)

    def test_its_attempts_are_the_python_downloads(self):
        """fetch.sh tries a download as rkmon_lib/deploy.py's fetcher does: as many attempts, as far apart, for the
        same HTTP answers."""
        text = (DOCKER / "fetch.sh").read_text(encoding="utf-8")
        self.assertIn(f"\nattempts={deploy.FETCH_ATTEMPTS}\n", text)
        pause = f"{deploy.FETCH_PAUSE_SECONDS:g}"
        self.assertIn(f"\npause=${{FETCH_PAUSE_SECONDS:-{pause}}}\n", text)
        retried = re.search(r"\n +22\) case \$status in ([0-9|]+)\) again=yes", text)
        self.assertEqual(
            tuple(int(s) for s in retried.group(1).split("|")), deploy.RETRIED_STATUS
        )

    def test_every_download_of_the_build_s_scripts_is_fetch_sh_s(self):
        """No build script but fetch.sh runs curl: each download is tried again, and its address is on no command
        line."""
        for f in sorted(DOCKER.glob("*.sh")):
            if f.name != "fetch.sh":
                code = re.sub(r"(?m)^\s*#.*$", "", f.read_text(encoding="utf-8"))
                with self.subTest(script=f.name):
                    self.assertNotRegex(code, r"\bcurl -")


# A tool that fails as apt and pip do (its second line on its standard error, and exit 100), naming the build's package
# mirror and package index (MASKED_SECRETS) in its lines, and an IP address on a line that names neither.
MASKED_SECRETS = {
    "package_mirror": "http://apt.example.com:8080/ubuntu-mirror/",
    "python_package_index": "https://user:secret@index.example.com/simple",
}
PIP_HIDDEN = "****"  # pip shows the password in an address so
MASKED_LINES = (
    "Err:1 http://apt.example.com:8080/ubuntu-mirror noble InRelease",
    "  Could not connect to apt.example.com:8080 (192.0.2.7). - connect (111: Connection refused)",
    "E: Failed to fetch http://APT.example.com:8080/ubuntu-mirror/dists/noble/InRelease",
    f"Could not fetch URL https://user:{PIP_HIDDEN}@index.example.com/simple/pkg/: "
    "HTTPSConnectionPool(host='index.example.com', port=443)",
    "an address on a line that names neither (192.0.2.8)",
)


class TestMasked(unittest.TestCase):
    """docker/masked.sh, a build step's tool whose lines name hosts (apt, the ROCm installer, pip): its lines shown,
    each download location the build was given by its build secret's name, any other host but a public one as <host>,
    every IP address as <address>, and its exit code kept.
    """

    @staticmethod
    def awks() -> list:
        """The awks to run it with: mawk (the Ubuntu bases' awk) and gawk (the RHEL family's), those this machine has,
        else its awk."""
        found = [a for a in ("mawk", "gawk") if shutil.which(a, path="/usr/bin:/bin")]
        return found or ["awk"]

    def run_masked(
        self,
        d: pathlib.Path,
        given: dict,
        *argv: str,
        awk: str = "awk",
        lines: tuple = MASKED_LINES,
    ):
        """masked.sh ARGV (none: both secrets, then the failing tool, which writes LINES, its second on its standard
        error) with the build secrets GIVEN, and AWK as its awk, which records its arguments in d/awk.argv.
        """
        secrets = d / "secrets"
        secrets.mkdir()
        for name, value in given.items():
            (secrets / name).write_text(value + "\n", encoding="utf-8")
        (d / "out1").write_text(lines[0] + "\n", encoding="utf-8")
        (d / "err").write_text(lines[1] + "\n", encoding="utf-8")
        (d / "out2").write_text(
            "".join(f"{line}\n" for line in lines[2:]), encoding="utf-8"
        )
        tool = d / "tool"
        tool.write_text(
            f'#!/bin/sh\ncat "{d}/out1"\ncat "{d}/err" >&2\ncat "{d}/out2"\nexit 100\n',
            encoding="utf-8",
        )
        wrapper = d / "awk" / "awk"
        wrapper.parent.mkdir()
        real = shutil.which(awk, path="/usr/bin:/bin")
        wrapper.write_text(
            f'#!/bin/sh\nprintf "%s\\n" "$@" >> "{d}/awk.argv"\nexec {real} "$@"\n',
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        argv = argv or ("package_mirror", "python_package_index", "--", "sh", str(tool))
        return subprocess.run(
            ["sh", str(DOCKER / "masked.sh"), *argv],
            capture_output=True,
            text=True,
            timeout=60,
            env={
                "PATH": f"{wrapper.parent}:{sandbox.safe_path(d)}",
                "SECRETS_DIR": str(secrets),
            },
        )

    def test_each_location_shows_as_its_secret_s_name(self):
        """Its address and its host, in any case, with a user name and password before it; an IP address in
        parentheses shows as <address> on any line. The locations reach awk in its environment, never on its command
        line; the tool's exit code is masked.sh's."""
        masked = (
            "Err:1 <package_mirror> noble InRelease\n"
            "  Could not connect to <package_mirror>:8080 (<address>). - connect (111: Connection refused)\n"
            "E: Failed to fetch <package_mirror>/dists/noble/InRelease\n"
            "Could not fetch URL https://<python_package_index>/simple/pkg/: HTTPSConnectionPool("
            "host='<python_package_index>', port=443)\n"
            "an address on a line that names neither (<address>)\n"
        )
        for awk in self.awks():
            with self.subTest(awk=awk), tempfile.TemporaryDirectory() as d:
                d = pathlib.Path(d)
                p = self.run_masked(d, MASKED_SECRETS, awk=awk)
                self.assertEqual((p.returncode, p.stdout, p.stderr), (100, masked, ""))
                argv = (d / "awk.argv").read_text(encoding="utf-8")
                self.assertIn("ENVIRON", argv)
                self.assertNotIn("example.com", argv)

    def test_a_location_the_build_was_not_given_shows_as_any_other_host(self):
        """It has no name to show: its host shows as <host>. With no location given, the tool's lines are masked all
        the same, its outputs together."""
        pip = "Could not fetch URL <host>/simple/pkg/: HTTPSConnectionPool(host='<host>', port=443)\n"
        with tempfile.TemporaryDirectory() as d:
            mirror = {"package_mirror": MASKED_SECRETS["package_mirror"]}
            p = self.run_masked(pathlib.Path(d), mirror)
            self.assertEqual(p.returncode, 100)
            self.assertIn("Err:1 <package_mirror> noble InRelease\n", p.stdout)
            self.assertIn(pip, p.stdout)
        masked = (
            "Err:1 <host>/ubuntu-mirror noble InRelease\n"
            "  Could not connect to <host>:8080 (<address>). - connect (111: Connection refused)\n"
            "E: Failed to fetch <host>/ubuntu-mirror/dists/noble/InRelease\n"
            f"{pip}"
            "an address on a line that names neither (<address>)\n"
        )
        for awk in self.awks():
            with self.subTest(awk=awk), tempfile.TemporaryDirectory() as d:
                p = self.run_masked(pathlib.Path(d), {}, awk=awk)
                self.assertEqual((p.returncode, p.stdout, p.stderr), (100, masked, ""))

    def test_another_host_shows_as_host_and_an_ip_address_as_address(self):
        """A host that is neither a location the build was given nor one of build_plan.TOOL_HOSTS, in apt's and pip's
        words, and every IP address, on every line: apt's failure line and its detail line ([IP: address port]), pip's
        line for a file on another host than its index's, a proxy's, and the host and address of a connection. A host
        of TOOL_HOSTS shows as it is, less a user name and password, and so do words that only look like a host.
        """
        files = build_plan.PYPI_FILES_HOST
        archive, security = "archive.ubuntu.com", "Security.Ubuntu.com"
        cases = (
            (
                "E: Failed to fetch http://apt.example.com:8080/ubuntu-mirror/pool/main/c/curl/curl_8.5.0_amd64.deb"
                "  404  Not Found [IP: 192.0.2.7 8080]",
                "E: Failed to fetch <package_mirror>/pool/main/c/curl/curl_8.5.0_amd64.deb  404  Not Found "
                "[IP: <address>]",
            ),
            (
                "  404  Not Found [IP: 192.0.2.7 8080]",
                "  404  Not Found [IP: <address>]",
            ),
            (
                f"Could not fetch URL https://files.example.org/packages/{WHEEL}: There was a problem confirming the "
                "ssl certificate: HTTPSConnectionPool(host='files.example.org', port=443): Max retries exceeded with "
                f"url: /packages/{WHEEL}",
                f"Could not fetch URL <host>/packages/{WHEEL}: There was a problem confirming the ssl certificate: "
                f"HTTPSConnectionPool(host='<host>', port=443): Max retries exceeded with url: /packages/{WHEEL}",
            ),
            (
                "Failed to resolve 'files.example.org' ([Errno -3] Temporary failure in name resolution)",
                "Failed to resolve '<host>' ([Errno -3] Temporary failure in name resolution)",
            ),
            (
                "'Connection to proxy.example.net timed out. (connect timeout=15)'",
                "'Connection to <host> timed out. (connect timeout=15)'",
            ),
            (
                "WARNING: The repository located at 192.0.2.9 is not a trusted or secure host and is being ignored. "
                "Allow it anyway with '--trusted-host 192.0.2.9'.",
                "WARNING: The repository located at <address> is not a trusted or secure host and is being ignored. "
                "Allow it anyway with '--trusted-host <address>'.",
            ),
            (
                "Looking in links: http://192.0.2.9:8000/wheels/",
                "Looking in links: <address>/wheels/",
            ),
            (
                "Temporary failure resolving 'mirror.example.net'",
                "Temporary failure resolving '<host>'",
            ),
            (
                "Could not connect to proxy.example.net:3128 (2001:db8::7), connection timed out",
                "Could not connect to <host>:3128 (<address>), connection timed out",
            ),
            (
                "Cannot initiate the connection to 192.0.2.9:80 (192.0.2.9). - connect (101: Network is unreachable)",
                "Cannot initiate the connection to <address>:80 (<address>). - connect (101: Network is unreachable)",
            ),
            (
                "0% [Connecting to mirror.example.net (192.0.2.9)]",
                "0% [Connecting to <host> (<address>)]",
            ),
            (
                "Fetching https://[2001:db8::9]:8443/simple/",
                "Fetching <address>/simple/",
            ),
            (
                f"Get:1 http://{archive}/ubuntu noble InRelease [256 kB]",
                f"Get:1 http://{archive}/ubuntu noble InRelease [256 kB]",
            ),
            (
                f"Downloading https://{files}/packages/{WHEEL} (40 kB)",
                f"Downloading https://{files}/packages/{WHEEL} (40 kB)",
            ),
            (
                f"Get:2 http://user:{PIP_HIDDEN}@{security}/ubuntu noble-security InRelease [126 kB]",
                f"Get:2 http://{security}/ubuntu noble-security InRelease [126 kB]",
            ),
            (
                "ProxyError('Cannot connect to proxy.', OSError('Tunnel connection failed: 403 Forbidden'))",
                "ProxyError('Cannot connect to proxy.', OSError('Tunnel connection failed: 403 Forbidden'))",
            ),
            (
                "Setting up libfoo:amd64 (1:2.3) ... from setup.py (0.4.1)",
                "Setting up libfoo:amd64 (1:2.3) ... from setup.py (0.4.1)",
            ),
        )
        lines = tuple(line for line, _ in cases)
        masked = "".join(f"{shown}\n" for _, shown in cases)
        for awk in self.awks():
            with self.subTest(awk=awk), tempfile.TemporaryDirectory() as d:
                p = self.run_masked(
                    pathlib.Path(d), MASKED_SECRETS, awk=awk, lines=lines
                )
                self.assertEqual((p.returncode, p.stdout, p.stderr), (100, masked, ""))

    def test_its_copy_of_the_tool_hosts_is_the_build_plan_s(self):
        """masked.sh's copy of the public hosts the tools read with no build secret of their own (TOOL_HOSTS) is
        rkmon_lib/build_plan.py TOOL_HOSTS."""
        text = (DOCKER / "masked.sh").read_text(encoding="utf-8")
        hosts = re.search(r'(?m)^TOOL_HOSTS="([^"]*)"$', text)
        self.assertEqual(tuple(hosts.group(1).split()), build_plan.TOOL_HOSTS)

    def test_its_usage(self):
        for argv in (
            ("--", "true"),
            ("package_mirror", "--"),
            ("package_mirror", "true"),
        ):
            with self.subTest(argv=argv), tempfile.TemporaryDirectory() as d:
                p = self.run_masked(pathlib.Path(d), MASKED_SECRETS, *argv)
                usage = "usage: masked.sh ID... -- COMMAND [ARG...]\n"
                self.assertEqual((p.returncode, p.stdout, p.stderr), (2, "", usage))

    def test_every_step_whose_tool_names_a_location_runs_it_masked(self):
        """apt (docker/pkgs.sh's install), the ROCm installer, which runs apt (docker/install_rocm.sh), and pip in the
        Dockerfile's steps run through masked.sh with the build secret their tool reads, and so does the showing of
        pip's lines, which docker/install_sdk.sh keeps in a file of its own; the base stage, which every other builds
        on, copies masked.sh beside param.sh."""
        for name, tool, secret in (
            ("pkgs.sh", "apt-get ", "package_mirror"),
            ("install_rocm.sh", "./installer.run ", "package_mirror"),
            ("Dockerfile", "pip install ", "python_package_index"),
            ("install_sdk.sh", 'cat "$work/pip.log"', "python_package_index"),
        ):
            text = (DOCKER / name).read_text(encoding="utf-8")
            runs = [
                s for s in re.sub(r"(?m)^\s*#.*$", "", text).splitlines() if tool in s
            ]
            with self.subTest(file=name):
                self.assertTrue(runs)
                for line in runs:
                    self.assertRegex(
                        line,
                        rf'masked\.sh"? (\w+ )*{secret} (\w+ )*-- .*{re.escape(tool.strip())}',
                    )
        base = dockerfile().split("\nFROM base AS ", 1)[0]
        self.assertRegex(
            base,
            r"(?m)^COPY [^\n]*docker/param\.sh [^\n]*docker/masked\.sh [^\n]*/opt/rkmon/build/$",
        )


# Stand-ins for install_sdk.sh's python3, and its venv's pip and python (TestInstallSdk.run_sdk; @FAKES@ is their
# folder). python3 -m venv DIR makes DIR with the other two.
FAKE_PYTHON3 = r"""#!/bin/sh
[ "$1" = -m ] && [ "$2" = venv ] || exit 9
mkdir -p "$3/bin"
cp "@FAKES@/pip" "@FAKES@/python" "$3/bin/"
"""
# pip download|install: when @FAKES@/pip-COMMAND is there, its lines (INDEX: the index pip was given), and it fails;
# else it succeeds, and a download leaves the SDK's wheel in its -d folder.
FAKE_PIP = r"""#!/bin/sh
command=$1
dir=""
while [ $# -gt 0 ]; do
    [ "$1" = -d ] && dir=$2
    shift
done
if [ -f "@FAKES@/pip-$command" ]; then
    sed "s|INDEX|$PIP_INDEX_URL|g" "@FAKES@/pip-$command"
    exit 1
fi
[ -z "$dir" ] || : > "$dir/@WHEEL@"
"""
# -B claude_cli.py --link LINK links a claude that reports @CLAUDE@ (none: no version).
FAKE_PYTHON = r"""#!/bin/sh
if [ -n "@CLAUDE@" ]; then
    printf '#!/bin/sh\necho "@CLAUDE@ (Claude Code)"\n' > "$4"
else
    printf '#!/bin/sh\nexit 1\n' > "$4"
fi
chmod 755 "$4"
"""
# What pip shows when an index names a host of another (a file host) whose certificate the build does not trust.
PIP_OTHER_HOST = (
    "Could not fetch URL https://files.example.org/simple/anyio/: There was a problem confirming the ssl "
    "certificate: HTTPSConnectionPool(host='files.example.org', port=443): Max retries exceeded with url: "
    "/simple/anyio/ (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate "
    "verify failed: unable to get local issuer certificate (_ssl.c:1006)')))"
)
# The tail of each refusal of pip's work that a certificate may cause (install_sdk.sh's CERTS).
CERTS = (
    "if pip says a certificate failed to verify, name your network's root certificates in "
    "operating_system.ca_certificates"
)


class TestInstallSdk(unittest.TestCase):
    """docker/install_sdk.sh, the Claude Agent SDK's install into the orchestrator's venv: a refusal shows pip's own
    lines, masked (docker/masked.sh), and names the build settings to change, and leaves no venv and no claude.
    """

    def run_sdk(
        self,
        d: pathlib.Path,
        pip: dict = None,
        claude: str = CLAUDE_CODE,
        url: str = "",
        sha: str = "",
    ) -> tuple:
        """install_sdk.sh SDK SHA CLAUDE_CODE VENV LINK, with stand-ins for python3 and its venv (FAKE_PYTHON3,
        FAKE_PIP, FAKE_PYTHON): pip fails each command PIP names ({"download" or "install": its lines}), and the
        venv's claude reports CLAUDE ("": no version). The build secrets: the package index, MIRROR, and the wheel's
        address, URL, when given. (the process, the venv, the link)"""
        fakes = d / "fakes"
        fakes.mkdir()
        for name, text in (
            ("python3", FAKE_PYTHON3),
            ("pip", FAKE_PIP),
            ("python", FAKE_PYTHON),
        ):
            for at, value in (("@FAKES@", str(fakes)), ("@WHEEL@", WHEEL)):
                text = text.replace(at, value)
            (fakes / name).write_text(
                text.replace("@CLAUDE@", claude), encoding="utf-8"
            )
            (fakes / name).chmod(0o755)
        for command, lines in (pip or {}).items():
            (fakes / f"pip-{command}").write_text(
                "".join(f"{line}\n" for line in lines), encoding="utf-8"
            )
        bin_ = d / "bin"
        bin_.mkdir()
        (bin_ / "python3").symlink_to(fakes / "python3")
        secrets = d / "secrets"
        secrets.mkdir()
        (secrets / "python_package_index").write_text(f"{MIRROR}\n")
        if url:
            (secrets / "claude_agent_sdk_package_url").write_text(f"{url}\n")
        venv, link = d / "venv-claude", d / "claude"
        p = subprocess.run(
            ["sh", str(DOCKER / "install_sdk.sh"), SDK, sha, CLAUDE_CODE]
            + [str(venv), str(link)],
            capture_output=True,
            text=True,
            timeout=120,
            env={"PATH": f"{bin_}:/usr/bin:/bin", "SECRETS_DIR": str(secrets)},
        )
        return p, venv, link

    def test_a_wheel_from_the_index_installs_and_links_its_claude(self):
        """The stand-ins' own run: the wheel from the package index, its dependencies, and its claude linked."""
        with tempfile.TemporaryDirectory() as d:
            p, venv, link = self.run_sdk(pathlib.Path(d))
            self.assertEqual((p.returncode, p.stderr), (0, ""))
            self.assertTrue(
                p.stdout.endswith(
                    f"install_sdk.sh: claude-agent-sdk {SDK}, with Claude Code {CLAUDE_CODE}\n"
                ),
                p.stdout,
            )
            self.assertTrue(venv.is_dir())
            self.assertTrue(link.is_file())

    def test_no_wheel_from_the_index(self):
        """pip's lines, the index in them by its build secret's name, then the refusal: the settings that choose the
        wheel (claude_agent_sdk.version, python_packages.package_index, claude_agent_sdk.package_url) and the root
        certificates (operating_system.ca_certificates). Exit 4."""
        lines = (
            "Looking in indexes: INDEX",
            f"ERROR: Could not find a version that satisfies the requirement claude-agent-sdk=={SDK} (from versions: "
            "0.1.0, 0.2.0)",
        )
        with tempfile.TemporaryDirectory() as d:
            p, venv, link = self.run_sdk(pathlib.Path(d), pip={"download": lines})
            self.assertEqual(p.returncode, 4)
            self.assertEqual(
                p.stderr,
                "Looking in indexes: <python_package_index>\n"
                f"{lines[1]}\n"
                f"install_sdk.sh: the package index gave no claude-agent-sdk {SDK} wheel for this machine (pip's "
                "lines above say why): nothing installed. Set claude_agent_sdk.version to a release the index has, "
                "python_packages.package_index to an index that has it, or claude_agent_sdk.package_url to the "
                f"wheel's address; {CERTS}\n",
            )
            self.assertFalse(venv.exists())
            self.assertFalse(link.exists())

    def test_dependencies_that_do_not_install(self):
        """pip's lines, a host of another than the index's as <host>, then the refusal: the package index
        (python_packages.package_index) and the root certificates. Exit 4."""
        lines = (
            "Looking in indexes: INDEX",
            PIP_OTHER_HOST,
            "ERROR: Could not find a version that satisfies the requirement anyio>=4.0.0 (from claude-agent-sdk) "
            "(from versions: none)",
        )
        with tempfile.TemporaryDirectory() as d:
            p, venv, link = self.run_sdk(pathlib.Path(d), pip={"install": lines})
            self.assertEqual(p.returncode, 4)
            self.assertEqual(
                p.stderr,
                "Looking in indexes: <python_package_index>\n"
                + PIP_OTHER_HOST.replace("https://files.example.org", "<host>").replace(
                    "host='files.example.org'", "host='<host>'"
                )
                + f"\n{lines[2]}\n"
                "install_sdk.sh: the SDK's dependencies did not install from the package index (pip's lines above "
                "say why): nothing installed. Set python_packages.package_index to an index that has them; "
                f"{CERTS}\n",
            )
            self.assertNotIn("files.example.org", p.stdout + p.stderr)
            self.assertFalse(venv.exists())
            self.assertFalse(link.exists())

    def test_a_wheel_of_another_release(self):
        """The wheel at claude_agent_sdk.package_url matches its SHA-256, but pip finds another release in it: pip's
        lines, then the refusal: claude_agent_sdk.package_url and .package_sha256, or claude_agent_sdk.version.
        Exit 3; the address is never printed."""
        if not shutil.which("curl"):
            self.skipTest("no curl")
        wheel = b"a wheel of 0.2.180\n"
        lines = (
            f"Processing /wheel/{WHEEL}",
            f"ERROR: Requested claude-agent-sdk=={SDK} from /wheel/{WHEEL} has inconsistent version: expected "
            f"'{SDK}', but metadata has '0.2.180'",
        )
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            base = serve_folder(self, d / "web", {WHEEL: wheel})
            p, venv, link = self.run_sdk(
                d,
                pip={"install": lines},
                url=f"{base}/{WHEEL}",
                sha=hashlib.sha256(wheel).hexdigest(),
            )
            self.assertEqual(p.returncode, 3)
            self.assertIn(
                "the SDK's wheel matches claude_agent_sdk.package_sha256", p.stdout
            )
            self.assertEqual(
                p.stderr,
                "".join(f"{line}\n" for line in lines)
                + f"install_sdk.sh: the SDK's wheel is no claude-agent-sdk {SDK} (pip's lines above say why): "
                "nothing installed. Set claude_agent_sdk.package_url to that release's wheel and "
                "claude_agent_sdk.package_sha256 to its SHA-256, or claude_agent_sdk.version to the wheel's "
                "release\n",
            )
            self.assertNotIn(base, p.stdout + p.stderr)
            self.assertFalse(venv.exists())
            self.assertFalse(link.exists())

    def test_a_claude_that_is_not_the_wheel_s(self):
        """The installed claude reports another version than the one its wheel names, or none: the refusal names
        this machine and the settings that choose the wheel. Exit 3, and no venv and no claude are left.
        """
        for claude, shown in (("2.1.98", "2.1.98"), ("", "no version")):
            with self.subTest(claude=claude), tempfile.TemporaryDirectory() as d:
                p, venv, link = self.run_sdk(pathlib.Path(d), claude=claude)
                self.assertEqual(p.returncode, 3)
                self.assertEqual(
                    p.stderr,
                    f"install_sdk.sh: the SDK's claude reports {shown}, not Claude Code {CLAUDE_CODE}, which its "
                    "wheel names: nothing installed. The wheel is not one for this machine "
                    f"({platform.machine()}): set claude_agent_sdk.package_url to claude_agent_sdk.version's wheel "
                    "for it, or claude_agent_sdk.version to another release\n",
                )
                self.assertFalse(venv.exists())
                self.assertFalse(link.exists())

    def test_a_wheel_whose_sha256_differs_installs_nothing(self):
        """The wheel at the address the build secret claude_agent_sdk_package_url holds is downloaded and checked
        first: its SHA-256 is not the one asked for (claude_agent_sdk.package_sha256), so the step stops (exit 3),
        names both, and leaves no venv and no claude; the address is never printed."""
        if not shutil.which("curl"):
            self.skipTest("no curl")
        wheel = b"another wheel\n"
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            base = serve_folder(self, d / "web", {WHEEL: wheel})
            p, venv, link = self.run_sdk(d, url=f"{base}/{WHEEL}", sha=SHA)
            self.assertEqual(p.returncode, 3, p.stderr)
            self.assertEqual(
                p.stderr,
                f"install_sdk.sh: the SDK's wheel's SHA-256 is {hashlib.sha256(wheel).hexdigest()}, not "
                f"claude_agent_sdk.package_sha256 ({SHA}): nothing installed\n",
            )
            self.assertFalse(venv.exists())
            self.assertFalse(link.exists())
            self.assertNotIn(base, p.stdout + p.stderr)


class TestInstallAtlassian(unittest.TestCase):
    """docker/install_atlassian.sh, the atlassian CLI's install (the image test's atlassian stage runs it in the image,
    case by case: tests/container/stage_atlassian.py)."""

    def test_a_binary_that_prints_no_version_names_the_settings_to_change(self):
        """A binary that matches its SHA-256 but prints no version is another program, or one for another
        architecture: the refusal names this machine's, atlassian_cli.package_url and .package_sha256; nothing is
        installed or recorded, and the address is never printed. Exit 3."""
        if not shutil.which("curl"):
            self.skipTest("no curl")
        binary = "#!/bin/sh\necho 'usage: tool COMMAND'\n"
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            base = serve_folder(self, d / "web", {"atlassian": binary})
            secrets = d / "secrets"
            secrets.mkdir()
            (secrets / "atlassian_cli_package_url").write_text(f"{base}/atlassian\n")
            record = d / "atlassian.json"
            p = subprocess.run(
                ["sh", str(DOCKER / "install_atlassian.sh")]
                + [
                    hashlib.sha256(binary.encode("utf-8")).hexdigest(),
                    "",
                    str(d / "bin"),
                ],
                capture_output=True,
                text=True,
                timeout=120,
                env={
                    "PATH": "/usr/bin:/bin",
                    "SECRETS_DIR": str(secrets),
                    "RKMON_ATLASSIAN_RECORD": str(record),
                },
            )
            machine = platform.machine()
            self.assertEqual(
                (p.returncode, p.stdout, p.stderr),
                (
                    3,
                    "",
                    "install_atlassian.sh: the binary prints no version when run as `atlassian version`, so it is "
                    f"not the atlassian tool, or not one for this image's architecture ({machine}): nothing "
                    "installed. Set atlassian_cli.package_url to the atlassian CLI's Linux binary for "
                    f"{machine}, and atlassian_cli.package_sha256 to its SHA-256\n",
                ),
            )
            self.assertFalse((d / "bin" / "atlassian").exists())
            self.assertFalse(record.exists())
            self.assertNotIn(base, p.stdout + p.stderr)


class TestDescribe(unittest.TestCase):
    """An image's line, from its labels (rkmon image check, image ls)."""

    def test_an_image_is_described_by_its_build_labels(self):
        self.assertEqual(
            image.describe(LABELS),
            "package 2026.11.13, built 2026-11-13T00:00:00Z, ubuntu:26.04, ROCm 10.1.0-5",
        )
        self.assertEqual(image.describe({}), "no build labels")


class TestBuildCommand(unittest.TestCase):
    """docker/rkmon build against the stand-in docker, on a copy of the package's docker/ and rkmon_lib/, with the
    schemas and the package's defaults the code reads: the settings become one docker build, run in the package
    folder, with the flags rkmon_lib/build_plan.py lists and no other; the locations reach it only through its
    environment, the root certificates and the build record only as build contexts of their own; what only the network
    gives (here, the SDK's wheel, served on 127.0.0.1) is read first, and a setting that cannot build runs nothing.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.pkg = self.d / "pkg"
        shutil.copytree(
            DOCKER,
            self.pkg / "docker",
            ignore=shutil.ignore_patterns(
                "__pycache__", ".env*", "compose.override.yaml*", "*.crt"
            ),
        )
        shutil.copytree(
            ROOT / "rkmon_lib",
            self.pkg / "rkmon_lib",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copytree(ROOT / "schema", self.pkg / "schema")
        for name in ("VERSION", "config.defaults.json"):
            shutil.copy(ROOT / name, self.pkg / name)
        sandbox.writable(self.pkg)
        self.bin = sandbox.safe_path(self.d)
        shutil.copy(HERE / "fakes" / "fake_docker", pathlib.Path(self.bin) / "docker")
        (pathlib.Path(self.bin) / "docker").chmod(0o755)
        wheels = serve_folder(
            self, self.d / "wheels", {WHEEL: plans.wheel(SDK, CLAUDE_CODE)}
        )
        self.wheel = f"{wheels}/{WHEEL}"
        self.build = build_settings(self.wheel)
        self.settings = self.d / "build.yaml"
        self.settings.write_text(self.build, encoding="utf-8")
        self.settings.chmod(0o600)
        self.log, self.envlog = self.d / "docker.log", self.d / "env.log"
        # The public sites the build path asks (the ROCm installers' folder, the TLS probe's): a closed port instead.
        closed = sandbox.closed_port(self)
        self.env = {
            "PATH": self.bin,
            "HOME": str(self.d),
            "LANG": "C.UTF-8",
            "FAKE_DOCKER_LOG": str(self.log),
            "FAKE_DOCKER_ENV_LOG": str(self.envlog),
            "FAKE_DOCKER_ENV": ",".join(
                [build_plan.secret_variable(d.id) for d in build_plan.DOWNLOADS]
                + list(build_plan.DOCKER_BUILD_ENV)
            ),
            "FAKE_DOCKER_CONTEXTS": str(self.d / "contexts"),
            "ROCM_PUBLIC_INSTALLERS": f"http://127.0.0.1:{closed}/none",
            "RKMON_TLS_SITE": f"127.0.0.1:{closed}",
            "TMPDIR": str(self.d / "tmp"),
        }
        (self.d / "tmp").mkdir()

    @property
    def rkmon(self) -> str:
        """The program's name in its messages as these tests run it: its path (rkmon_lib/program.py)."""
        return str(self.pkg / "docker" / "rkmon")

    def run_(self, *argv):
        p = subprocess.run(
            [sys.executable, "-B", str(self.pkg / "docker" / "rkmon"), *map(str, argv)],
            cwd=self.d,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return p.returncode, p.stdout + p.stderr

    def builds(self) -> list:
        lines = self.log.read_text().splitlines() if self.log.exists() else []
        return [x.split("\t", 1)[1] for x in lines if "\tbuild " in x]

    def test_one_docker_build_the_locations_as_secrets(self):
        """One docker build, in the package folder: the Dockerfile and the build context named from there, so neither
        the command nor the build's output names a folder of yours; each download location is a build secret, in its
        environment alone, and your environment's git record switches give way to the build's own. The build record is
        a build context of its own, in a private folder gone after the build, and names no location.
        """
        self.env.update(BUILDX_GIT_INFO="1", BUILDX_GIT_LABELS="full")
        rc, out = self.run_("build", "--settings", self.settings, "--no-cache")
        self.assertEqual(rc, 0, out)
        [argv] = self.builds()
        self.assertTrue(
            argv.startswith("build -f docker/Dockerfile -t rocke-monitor:t1 "), argv
        )
        self.assertIn(" --provenance=false --no-cache ", argv)
        for d in build_plan.DOWNLOADS:
            if d.id != "package_mirror":  # none named: the base's own package archive
                self.assertIn(
                    f"--secret id={d.id},env={build_plan.secret_variable(d.id)} ", argv
                )
        self.assertNotIn("id=package_mirror", argv)
        self.assertIn(f"--build-arg CLAUDE_CODE_VERSION={CLAUDE_CODE} ", argv)
        self.assertIn(f"--label org.rocke.claude_code.version={CLAUDE_CODE} ", argv)
        [where] = [
            x.split("\t", 1)[0]
            for x in self.log.read_text().splitlines()
            if "\tbuild " in x
        ]
        self.assertEqual(pathlib.Path(where).resolve(), self.pkg.resolve())
        [shown] = [x for x in out.splitlines() if " docker build " in x]
        self.assertNotIn(str(self.d), shown)
        self.assertTrue(
            shown.endswith(" --build-context 'build-record=<the build record>' ."),
            shown,
        )
        self.assertNotIn(build_plan.CA_CONTEXT, argv)  # no root certificate named
        folder = re.search(
            rf" --build-context {build_plan.RECORD_CONTEXT}=(\S+) \.$", argv
        ).group(1)
        self.assertTrue(pathlib.Path(folder).is_relative_to(self.d / "tmp"), folder)
        self.assertNotIn(str(self.d), argv.replace(folder, ""))
        self.assertEqual(list((self.d / "tmp").iterdir()), [])  # gone after the build
        record = (
            self.d / "contexts" / build_plan.RECORD_CONTEXT / build_plan.RECORD_FILE
        ).read_text()
        rec = json.loads(record)
        self.assertEqual(
            (rec["kind"], rec["package_version"], rec["rocm"]["version"]),
            (build_plan.RECORD_KIND, image.package_version(), "10.1.0-5"),
        )
        self.assertEqual(
            rec["claude_agent_sdk"],
            {"version": SDK, "package_sha256": SHA, "claude_code_version": CLAUDE_CODE},
        )
        # Each docker call logs the variables of FAKE_DOCKER_ENV it has, in that order: the build's are its secrets
        # and the build's own git record switches; every other call has only yours.
        lines = self.envlog.read_text().splitlines()
        given = [
            f"ROCM_PACKAGE_URL={RUN}",
            f"ATLASSIAN_CLI_PACKAGE_URL={CLI}",
            f"GITHUB_CLI_PACKAGE_URL={gh_public()}",
            f"CLAUDE_AGENT_SDK_PACKAGE_URL={self.wheel}",
            f"PYTHON_PACKAGE_INDEX={MIRROR}",
            "BUILDX_GIT_INFO=0",
            "BUILDX_GIT_LABELS=0",
        ]
        at = lines.index(given[0])
        self.assertEqual(lines[at : at + len(given)], given)
        self.assertEqual(
            sorted(set(lines) - set(given)),
            ["BUILDX_GIT_INFO=1", "BUILDX_GIT_LABELS=full"],
        )
        for secret in (RUN, MIRROR, CLI, self.wheel, "127.0.0.1", WHEEL):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, out + argv + record)
        self.assertIn("'ROCM_PACKAGE_URL=<rocm.package_url>'", out)
        # The stand-in docker has every image: the build says it would replace the one of its name and tag. Its fix
        # names your file as every line does, ~/… in your home folder.
        self.assertIn(
            "WARN image.tag: rocke-monitor:t1 is on this machine already; this build would replace it under the "
            'same tag fix: set image.tag in ~/build.yaml to a tag of its own, such as "t1-2"',
            one_line(out),
        )

    def test_docker_is_given_only_the_flags_build_lists(self):
        """rkmon build gives docker build_plan's flags and no other: a word after its own options is refused, exit 2,
        and nothing is built."""
        rc, out = self.run_("build", "--settings", self.settings)
        self.assertEqual(rc, 0, out)
        [argv] = self.builds()
        self.assertLessEqual(
            {w for w in argv.split() if w.startswith("-")},
            {
                "-f",
                "-t",
                "--provenance=false",
                "--build-arg",
                "--secret",
                "--label",
                "--build-context",
            },
        )
        for extra in (
            ("--", "--label", "x=y"),
            ("--", "--no-cache"),
            ("--network=host",),
        ):
            with self.subTest(extra=extra):
                rc, out = self.run_("build", "--settings", self.settings, *extra)
                self.assertEqual(rc, 2, out)
                self.assertIn(f"{self.rkmon} build: unrecognized arguments: ", out)
        self.assertEqual(len(self.builds()), 1)

    def test_the_image_s_name_names_it_from_the_build_to_the_export(self):
        """image.name names the image: the build tags it so, and image test and export (--settings) take it; a
        registry's address in it stops the build, its fix naming the key a registry goes in.
        """
        self.settings.write_text(
            self.build.replace("image:\n", "image:\n  name: team/rkmon\n"),
            encoding="utf-8",
        )
        rc, out = self.run_("build", "--settings", self.settings)
        self.assertEqual(rc, 0, out)
        [argv] = self.builds()
        self.assertIn("-t team/rkmon:t1", argv)
        self.assertIn(
            finished(
                f"next: step 4, test: {self.rkmon} image test --settings ~/build.yaml"
            ),
            out,
        )
        for argv in (
            ("image", "test", "--settings", self.settings),
            ("image", "save", self.d / "img.tar.gz", "--settings", self.settings),
        ):
            with self.subTest(argv=argv):
                rc, out = self.run_("--dry-run", *argv)
                self.assertEqual(rc, 0, out)
                self.assertIn("team/rkmon:t1", out)
                self.assertNotIn("rocke-monitor:", out)
        self.settings.write_text(
            self.build.replace(
                "image:\n", "image:\n  name: registry.example.com/rkmon\n"
            ),
            encoding="utf-8",
        )
        built = len(self.builds())
        rc, out = self.run_("build", "--settings", self.settings)
        self.assertEqual(rc, 4, out)
        self.assertIn(
            'FAIL image.name: "registry.example.com/rkmon" is not in the form this key takes fix: set image.name in '
            "~/build.yaml to a value of this form: The image's name, without its registry, which goes in "
            "image.registry.",
            one_line(out),
        )
        self.assertEqual(len(self.builds()), built)

    def test_without_settings_and_with_none_confirmed_no_image_is_named(self):
        """Without --settings, image test and image save act on the image this folder's deployment names, which none
        does before a confirm (docker compose gives the image no default either): REFUSED, an earlier step (exit 3),
        with the command again and --settings FILE, and docker is asked about no image.
        """
        dest = self.d / "img.tar.gz"
        for argv in (("image", "test"), ("image", "save", str(dest))):
            with self.subTest(argv=argv):
                rc, out = self.run_(*argv)
                self.assertEqual(rc, 3, out)
                self.assertIn(
                    finished(
                        "REFUSED: no image is named: no build settings were given (--settings), and no deploy "
                        "settings are confirmed in this package folder; nothing changed",
                        f"next: {self.rkmon} {' '.join(argv)} --settings FILE",
                    ),
                    out,
                )
                self.assertEqual(self.builds(), [])
                log = self.log.read_text() if self.log.exists() else ""
                self.assertNotIn("image inspect", log)
                self.assertFalse(dest.exists())

    def answer(self, *rows):
        """The stand-in docker's answers: [pattern, output, exit code] rows (fake_docker)."""
        answers = self.d / "answers.json"
        answers.write_text(json.dumps([list(r) for r in rows]))
        self.env["FAKE_DOCKER_ANSWERS"] = str(answers)

    def test_each_step_s_banner_outcome_and_next_line(self):
        """The build path's steps 3 to 5 (rkmon_lib/steps.py): each starts with its banner and ends with one outcome
        line and its next line (rkmon_lib/outcome.py): the build, BUILT, then the test; the test, PASSED, then the
        ship; the save, SAVED, then the done line, which names the deploy settings' key for the tag. A next step's line
        names your file as every line does, ~/… in your home folder; your command again names it as you typed it.
        """
        tests = self.pkg / "tests" / "container"
        tests.mkdir(parents=True)
        (tests / "image.py").write_text(
            "import os, sys\nprint('IMAGE TEST: PASS')\nsys.exit(int(os.environ.get('IMAGE_TEST_EXIT', '0')))\n"
        )
        rkmon, settings = self.rkmon, self.settings
        rc, out = self.run_("build", "--settings", settings)
        self.assertEqual(rc, 0, out)
        self.assertIn(
            "== build step 3 of 5, build: build the image from your build settings\n",
            out,
        )
        self.assertIn(
            finished(
                "BUILT rocke-monitor:t1",
                f"next: step 4, test: {rkmon} image test --settings ~/build.yaml",
            ),
            out,
        )
        rc, out = self.run_("image", "test", "--settings", settings)
        self.assertEqual(rc, 0, out)
        self.assertIn(
            "== build step 4 of 5, test: test the image on this machine\n", out
        )
        self.assertIn(
            finished(
                "PASSED: rocke-monitor:t1",
                f"next: step 5, ship: {rkmon} image save IMAGE_FILE --settings ~/build.yaml",
            ),
            out,
        )
        self.env["IMAGE_TEST_EXIT"] = "3"
        rc, out = self.run_("image", "test", "--settings", settings)
        self.assertEqual(rc, 4, out)
        self.assertIn(
            finished(
                "FAILED: rocke-monitor:t1: its tests failed (each FAIL above)",
                f"next: {rkmon} image test --settings {settings}",
            ),
            out,
        )
        dest = self.d / "img.tar.gz"
        rc, out = self.run_("image", "save", dest, "--settings", settings)
        self.assertEqual(rc, 0, out)
        self.assertIn("== build step 5 of 5, ship: ", out)
        self.assertIn(
            finished(
                f"SAVED rocke-monitor:t1 to {dest} (0.0 GiB)",
                f"done: rocke-monitor:t1 is ready (in {dest}). On the deploy machine, follow QUICK_DEPLOY_GUIDE.md "
                'from step 1, with the package of this version; set image.tag to "t1" in your deploy '
                f"settings, and step 4 loads {dest}",
            ),
            out,
        )
        self.assertTrue(dest.is_file())

    def test_the_save_writes_a_file_of_its_own_and_follows_no_link(self):
        """image save writes the image into a new file of its own beside IMAGE_FILE (whole_file.beside), made new and
        never through a link (whole_file.FLAGS), then renamed onto IMAGE_FILE: a link planted where a save may write
        leads it nowhere, and nothing of the save is left beside the file."""
        dest = self.d / "img.tar.gz"
        yours = self.d / "yours.txt"
        yours.write_text("yours\n", encoding="utf-8")
        planted = self.d / "img.tar.gz.partial"
        planted.symlink_to(yours)
        rc, out = self.run_("image", "save", dest, "--settings", self.settings)
        self.assertEqual(rc, 0, out)
        self.assertIn("SAVED rocke-monitor:t1 to ", out)
        self.assertTrue(dest.is_file() and not dest.is_symlink())
        self.assertEqual(yours.read_text(encoding="utf-8"), "yours\n")
        self.assertTrue(planted.is_symlink())
        self.assertEqual(
            [p.name for p in self.d.iterdir() if p.name.startswith(".img")], []
        )

    def test_a_private_file_in_the_package_folder_stops_the_build_and_the_ship(self):
        """A private file in the package folder that is not its own (deployment.stray_private_files): the check shows
        its row, and the build, the save and the push refuse with it, FAILED, exit 4, nothing built or written. The
        package's own .env, in its place, is no such file."""
        (self.pkg / "docker" / "notes.yaml").write_text(
            "kind: build_settings\n", encoding="utf-8"
        )
        (self.pkg / "docker" / ".env").write_text("RKMON_TAG=t1\n", encoding="utf-8")
        row = r"(?m)^  FAIL  the package folder: private files inside it: docker/notes\.yaml$"
        rc, out = self.run_("check", "--settings", self.settings)
        self.assertNotEqual(rc, 0, out)
        self.assertRegex(out, row)
        rc, out = self.run_("build", "--settings", self.settings)
        self.assertEqual(rc, 4, out)
        self.assertRegex(out, row)
        self.assertIn("FAILED: ", out)
        self.assertEqual(self.builds(), [])
        dest = self.d / "img.tar.gz"
        for argv in (
            ("image", "save", dest, "--settings", self.settings),
            ("image", "push", "--settings", self.settings),
        ):
            with self.subTest(argv=argv[:2]):
                rc, out = self.run_(*argv)
                self.assertEqual(rc, 4, out)
                self.assertRegex(out, row)
                self.assertIn("FAILED: ", out)
        self.assertFalse(dest.exists())
        self.assertNotIn("\tsave ", self.log.read_text() if self.log.exists() else "")
        self.assertNotIn("\tpush ", self.log.read_text() if self.log.exists() else "")

    def test_the_tag_is_the_settings_file_s_alone(self):
        """The image's tag is image.tag in the build settings file, with no --tag of its own: build refuses
        --tag, exit 2, and builds nothing; the next step's line names the file alone, which names the image.
        """
        rc, out = self.run_("build", "--settings", self.settings, "--tag", "t9")
        self.assertEqual(rc, 2, out)
        self.assertIn("unrecognized arguments: --tag t9", out)
        self.assertEqual(self.builds(), [])
        rc, out = self.run_("build", "--settings", self.settings)
        self.assertEqual(rc, 0, out)
        self.assertIn("-t rocke-monitor:t1", self.builds()[-1])
        self.assertIn(
            finished(
                "BUILT rocke-monitor:t1",
                f"next: step 4, test: {self.rkmon} image test --settings ~/build.yaml",
            ),
            out,
        )

    def test_a_dry_run_runs_every_guard_and_nothing_else(self):
        """--dry-run: the banner says so, every guard runs, the docker build is shown and not run, no CA certificate
        is staged and no record written, and the outcome is PASSED with the command again without --dry-run, exit 0
        as the real run's.
        """
        pem = self.d / "root.crt"
        pem.write_text("-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n")
        self.settings.write_text(
            self.build + f"operating_system:\n  ca_certificates: [{pem}]\n"
        )
        rc, out = self.run_("--dry-run", "build", "--settings", self.settings)
        self.assertEqual(rc, 0, out)
        self.assertIn(
            "the image trusts the 1 root certificate operating_system.ca_certificates names\n",
            out,
        )
        self.assertIn(
            " --build-context 'ca-certificates=<operating_system.ca_certificates>' --build-context "
            "'build-record=<the build record>' .\n",
            out,
        )
        self.assertIn(
            "== build step 3 of 5, build: build the image from your build settings (dry run: nothing changes)\n",
            out,
        )
        self.assertIn(
            finished(
                "PASSED: rocke-monitor:t1 can be built",
                f"next: {self.rkmon} build --settings {self.settings}",
            ),
            out,
        )
        self.assertEqual(self.builds(), [])
        self.assertEqual(list((self.d / "tmp").iterdir()), [])  # nothing staged
        self.settings.write_text(self.build.split("atlassian_cli:\n", 1)[0])
        rc, out = self.run_("--dry-run", "build", "--settings", self.settings)
        self.assertEqual(rc, 4, out)
        self.assertIn("FAILED: ", out)

    def test_a_dry_run_shows_each_location_of_yours_as_its_key(self):
        """build --dry-run prints the docker build as it would run, and runs nothing: each download location of yours
        as its settings key, and one the build takes by default, public, as its address.
        """
        rc, out = self.run_("build", "--settings", self.settings, "--dry-run")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.builds(), [])
        self.assertIn(
            "BUILDX_GIT_INFO=0 BUILDX_GIT_LABELS=0 'ROCM_PACKAGE_URL=<rocm.package_url>' "
            f"'ATLASSIAN_CLI_PACKAGE_URL=<atlassian_cli.package_url>' GITHUB_CLI_PACKAGE_URL={gh_public()} "
            "'CLAUDE_AGENT_SDK_PACKAGE_URL=<claude_agent_sdk.package_url>' "
            "'PYTHON_PACKAGE_INDEX=<python_packages.package_index>' docker build -f docker/Dockerfile ",
            out,
        )
        for secret in (RUN, CLI, MIRROR, self.wheel):
            self.assertNotIn(secret, out)

    def test_no_image_is_the_build_before_it(self):
        """The test and the ship of an image that is not here: REFUSED, exit 3 (an earlier step), and the build as the
        next line."""
        self.answer([r"^image inspect", "Error: No such image", 1])
        for argv in (
            ("image", "test", "--settings", self.settings),
            ("image", "save", self.d / "img.tar.gz", "--settings", self.settings),
            ("image", "push", "--settings", self.settings),
        ):
            with self.subTest(argv=argv[:2]):
                rc, out = self.run_(*argv)
                self.assertEqual(rc, 3, out)
                self.assertIn(
                    finished(
                        "REFUSED: no image rocke-monitor:t1 on this machine; nothing changed",
                        f"next: step 3, build: {self.rkmon} build --settings ~/build.yaml",
                    ),
                    out,
                )
        self.assertFalse((self.d / "img.tar.gz").exists())

    def test_a_push_names_the_registry_key(self):
        """A push with no registry is REFUSED, exit 4, its fix naming the key and the save, and your file as every
        line does (~/… in your home folder); then your command again, as you typed it. With one, --dry-run's PASSED
        and PUSHED name the registry's address, then the done line for an image in the registry.
        """
        rc, out = self.run_("image", "push", "--settings", self.settings)
        self.assertEqual(rc, 4, out)
        self.assertIn(
            "REFUSED: your build settings name no registry (image.registry); nothing pushed\n",
            out,
        )
        self.assertIn(
            "fix: set image.registry in ~/build.yaml to your registry's address, or save the image to a "
            f"file: {self.rkmon} image save IMAGE_FILE --settings ~/build.yaml next: {self.rkmon} image push "
            f"--settings {self.settings}",
            " ".join(out.split()),
        )
        self.settings.write_text(
            self.build.replace(
                "image:\n", "image:\n  registry: registry.example.com/team\n"
            )
        )
        rc, out = self.run_("--dry-run", "image", "push", "--settings", self.settings)
        self.assertEqual(rc, 0, out)
        self.assertIn(
            "PASSED: rocke-monitor:t1 can be pushed to registry.example.com/team (image.registry)\n",
            out,
        )
        rc, out = self.run_("image", "push", "--settings", self.settings)
        self.assertEqual(rc, 0, out)
        self.assertIn(
            "PUSHED rocke-monitor:t1 to registry.example.com/team (image.registry)\n",
            out,
        )
        self.assertIn(
            finished(
                "done: rocke-monitor:t1 is ready (in the registry). On the deploy machine, follow "
                'QUICK_DEPLOY_GUIDE.md from step 1, with the package of this version; set image.tag to "t1" '
                "in your deploy settings, and step 4 pulls it"
            ),
            out,
        )
        pushed = [x for x in self.log.read_text().splitlines() if "\tpush " in x]
        self.assertEqual(
            [x.split("\t", 1)[1] for x in pushed],
            ["push registry.example.com/team/rocke-monitor:t1"],
        )

    def test_a_failed_docker_build_is_an_error(self):
        """A docker build that fails: ERROR, exit 1, and the command again."""
        self.answer([r"^build ", "", 17])
        rc, out = self.run_("build", "--settings", self.settings)
        self.assertEqual(rc, 1, out)
        self.assertIn(
            finished(
                "ERROR: the docker build exited 17, so rocke-monitor:t1 was not built",
                f"next: {self.rkmon} build --settings {self.settings}",
            ),
            out,
        )

    def test_a_passing_check_lists_the_parameters_then_the_build(self):
        """`rkmon check` with a build settings file that passes: the build path's step 2, PASSED with its count, the
        build's parameters, then the build as the next step, which names your file as every line does (~/… in your
        home folder, which is the test's folder here). Asked in the process, so that docker and this network's TLS
        are the test's answers; the next line names the program as unittest runs it (rkmon_lib/program.py).
        """
        rkmon = self.pkg / "docker" / "rkmon"
        loader = importlib.machinery.SourceFileLoader("rkmon_wrapper", str(rkmon))
        mod = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("rkmon_wrapper", loader)
        )
        loader.exec_module(mod)
        w = mod.Wrapper(str(rkmon))
        w.cwd, w.typed = str(self.d), ["check", "--settings", "build.yaml"]
        facts = deployment.Facts(
            docker="29.0.0",
            buildx="github.com/docker/buildx v0.29.0",
            disk_free_gb=500.0,
            disk_where=str(self.d),
        )
        w.facts = lambda cfg, tls_cafiles=None, ref="": facts
        out = io.StringIO()
        # The wheel is read in this process, from 127.0.0.1: never through a proxy your environment names.
        direct = {"no_proxy": "127.0.0.1", "NO_PROXY": "127.0.0.1", "HOME": str(self.d)}
        with mock.patch.dict(os.environ, direct), mock.patch.object(
            sys, "path", list(sys.path)
        ):
            with contextlib.redirect_stdout(out):
                ns = w.lib("commands").table_parser().parse_args(w.typed)
                rc = w.cmd_check(ns)
        out = out.getvalue()
        self.assertEqual(rc, 0, out)
        self.assertIn("== build step 2 of 5, check: ", out)
        self.assertIn("  PASS  docker: server 29.0.0", out)
        self.assertRegex(
            out,
            r"(?m)^PASSED: \d+ checks( \(\d+ warnings?\))?\nThe build's parameters:\n",
        )
        self.assertRegex(out, r"(?m)^  ROCM_VERSION +10\.1\.0-5$")
        self.assertRegex(out, rf"(?m)^  CLAUDE_CODE_VERSION +{re.escape(CLAUDE_CODE)}$")
        self.assertTrue(
            out.endswith(
                "\nnext: step 3, build: rkmon build --settings ~/build.yaml\n"
            ),
            out,
        )
        for secret in (RUN, CLI, MIRROR, self.wheel):
            self.assertNotIn(secret, out)

    def test_a_missing_atlassian_cli_stops_with_its_key(self):
        """The build's guards are the check's rows: each FAIL row, then FAILED and the command again, exit 4. The
        atlassian CLI's address is required: its fix says what it is (the build schema's own words).
        """
        self.settings.write_text(self.build.replace(f'  package_url: "{CLI}"\n', ""))
        rc, out = self.run_("build", "--settings", self.settings)
        self.assertEqual(rc, 4, out)
        self.assertRegex(out, r"(?m)^  FAIL  atlassian_cli\.package_url: missing$")
        self.assertIn(
            "FAIL atlassian_cli.package_url: missing fix: set atlassian_cli.package_url in ~/build.yaml to the "
            "address of the atlassian tool's Linux binary for x86-64, from wherever your organization keeps it",
            one_line(out),
        )
        self.assertNotIn("  PASS  ", out)
        self.assertRegex(
            out,
            r"(?m)^FAILED: \d+ of \d+ checks failed; nothing changed \(fix each FAIL above\)\n"
            + re.escape(
                finished(f"next: {self.rkmon} build --settings {self.settings}")
            ),
        )
        self.assertEqual(self.builds(), [])
        rc, out = self.run_("build")
        self.assertEqual(rc, 2)
        self.assertIn("build: the following arguments are required: --settings", out)

    def test_the_template_is_the_package_s(self):
        """`rkmon template build -` prints the build template settings_file writes from the build schema."""
        rc, out = self.run_("template", "build", "-")
        self.assertEqual(rc, 0, out)
        self.assertEqual(out, settings_file.template(file_types.BUILD))

    def test_a_refused_build_settings_file(self):
        """The check (step 2) fails on every refusal of the file (outside the package and git, mode 600, readable, its
        syntax, its header, its kind and its layout: migrate's row for a file in the layout before layout 1) with FAILED
        and exit 4, then the command again; so do the build and the push, with the FAIL row of each: the private file's
        rows are the settings file's, the reader's the build settings file's. A file with no build settings header
        gets the deploy path's check: a deploy settings file is checked as one, and a missing file as the deploy's.
        """
        self.answer(
            [r"^version", "29.0.0", 0],
            [r"^buildx version", "github.com/docker/buildx v0.29.0", 0],
            [r"^compose version", "2.40.0", 0],
        )
        deploy_file = self.d / "deploy.yaml"
        deploy_file.write_text(  # the check looks whether the admin dashboard's port is free: a port of its own
            "kind: deploy_settings\nschema_version: 2\nrepository:\n  name: acme/widgets\nfeatures:\n  dashboards:\n"
            f"    admin:\n      enabled: true\n      port: {sandbox.free_port()}\n"
        )
        deploy_file.chmod(0o600)
        older = self.d / "older.yaml"
        older.write_text("rkmon_settings: 1\nconfig:\n  repo: acme/widgets\n")
        older.chmod(0o600)
        older_build = self.d / "older-build.yaml"
        older_build.write_text('rkmon_build_settings: 1\nimage:\n  tag: "1.0"\n')
        older_build.chmod(0o600)
        inside = self.pkg / "docker" / "build.yaml"
        inside.write_text(self.build)
        inside.chmod(0o600)
        broken = self.d / "broken.yaml"
        broken.write_text("rkmon_build_settings: 1\nimage: {tag: [a, {b: c}]}\n")
        broken.chmod(0o600)
        self.settings.chmod(0o644)
        rc, out = self.run_("check", "--settings", deploy_file)
        self.assertIn(
            "== deploy step 2 of 8, check: check this host and your deploy settings; change nothing",
            out,
        )
        self.assertNotIn("not build settings", out)
        for file, words, check_row, row in (
            (
                self.settings,
                "chmod 600 ~/build.yaml",
                "settings file",
                "settings file",
            ),
            (
                deploy_file,
                '~/deploy.yaml is deploy settings (kind "deploy_settings"), not build settings',
                None,
                "build settings file",
            ),
            (
                older,
                "~/older.yaml is in the layout before layout 1 (it starts with rkmon_settings), which this package "
                "does not read",
                "deploy settings file",
                "build settings file",
            ),
            (
                older_build,
                "~/older-build.yaml is in the layout before layout 1 (it starts with rkmon_build_settings), which "
                "this package does not read",
                "build settings file",
                "build settings file",
            ),
            (inside, "is inside the package folder", "settings file", "settings file"),
            (
                broken,
                "correct that line in ~/broken.yaml",
                "build settings file",
                "build settings file",
            ),
            (self.d / "none.yaml", "no file at", "your settings file", "settings file"),
        ):
            with self.subTest(file=file.name):
                if check_row:
                    rc, out = self.run_("check", "--settings", file)
                    self.assertEqual(rc, 4, out)
                    # the found text after the name, or on the next line when its first word, a path, does not fit
                    self.assertRegex(out, rf"(?m)^  FAIL  {check_row}:(?: |$)")
                    self.assertIn(words, " ".join(out.split()))
                    self.assertRegex(
                        out,
                        r"(?m)^FAILED: \d+ of \d+ checks failed; nothing changed \(fix each FAIL above\)\n"
                        + re.escape(
                            finished(
                                f"next: {self.pkg}/docker/rkmon check --settings {file}"
                            )
                        ),
                    )
                for argv in (("build",), ("image", "push")):
                    rc, out = self.run_(*argv, "--settings", file)
                    self.assertEqual(rc, 4, out)
                    self.assertRegex(out, rf"(?m)^  FAIL  {row}:(?: |$)")
                    self.assertIn(words, " ".join(out.split()))
                    self.assertRegex(
                        out,
                        r"(?m)^FAILED: \d+ of \d+ checks failed; nothing changed \(fix each FAIL above\)\n"
                        + re.escape(
                            finished(
                                f"next: {self.rkmon} {' '.join(argv)} --settings {file}"
                            )
                        ),
                    )
        self.assertEqual(self.builds(), [])
        self.assertFalse(
            [x for x in self.log.read_text().splitlines() if "\tpush " in x]
        )
        # A file of another kind: the build settings file is its fix, or a new one.
        rc, out = self.run_("build", "--settings", deploy_file)
        self.assertEqual(rc, 4, out)
        self.assertIn(
            f"fix: name your build settings file instead, or write one: {self.rkmon} template build FILE",
            one_line(out),
        )

    def serve(self, files: dict) -> str:
        return serve_folder(self, self.d / "web", files)

    def test_a_release_takes_its_newest_build_in_amd_s_public_folder(self):
        """A release without rocm.package_url: the build takes its newest build in AMD's public folder, read before
        the build, and is given that installer's public address, shown in full; a release the folder does not list
        builds nothing. With a package URL, the release is what the installed ROCm must report: nothing is listed,
        and another release takes any clang."""
        base = self.serve(
            {
                f"rocm-installer-{v}.run": "x"
                for v in ("10.1.0-3", "10.1.0-7", "10.2.0-1")
            }
        )
        self.env["ROCM_PUBLIC_INSTALLERS"] = base
        no_url = self.build.replace(f'  package_url: "{RUN}"\n', "", 1)
        self.settings.write_text(
            no_url.replace('version: "10.1.0-5"', "version: 10.1.0")
        )
        rc, out = self.run_("build", "--settings", self.settings)
        self.assertEqual(rc, 0, out)
        argv = self.builds()[-1]
        self.assertIn("--build-arg ROCM_VERSION=10.1.0-7 ", argv)
        self.assertIn(
            f"--build-arg ROCM_CLANG_COMMIT={build_plan.ROCM_CLANG_COMMIT} ", argv
        )
        self.assertIn("--secret id=rocm_package_url,env=ROCM_PACKAGE_URL ", argv)
        self.assertIn(f" ROCM_PACKAGE_URL={base}/rocm-installer-10.1.0-7.run ", out)
        self.assertIn(
            f"ROCM_PACKAGE_URL={base}/rocm-installer-10.1.0-7.run\n",
            self.envlog.read_text(),
        )
        self.settings.write_text(
            no_url.replace('version: "10.1.0-5"', "version: 10.3.0")
        )
        rc, out = self.run_("build", "--settings", self.settings)
        self.assertEqual(rc, 4, out)
        self.assertIn(
            "FAIL rocm.version: AMD's public folder of ROCm installers lists no build of ROCm 10.3.0 "
            "fix: set rocm.version in ~/build.yaml to a release that folder lists, or set rocm.package_url in "
            "~/build.yaml to the installer's full URL",
            one_line(out),
        )
        self.assertEqual(len(self.builds()), 1)
        self.settings.write_text(
            self.build.replace('version: "10.1.0-5"', "version: 10.3.0")
        )
        rc, out = self.run_("build", "--settings", self.settings)
        self.assertEqual(rc, 0, out)
        self.assertIn("--build-arg ROCM_VERSION=10.3.0 ", self.builds()[-1])
        self.assertIn("--build-arg ROCM_CLANG_COMMIT= ", self.builds()[-1])

    def test_a_build_settings_file_in_layout_1_names_migrate(self):
        """A build settings file in layout 1, older than this package's: the reader's FAIL row, with rkmon migrate as
        its fix, and nothing built."""
        self.settings.write_text(BUILD1)
        rc, out = self.run_("build", "--settings", self.settings)
        self.assertEqual(rc, 4, out)
        said = one_line(out)
        self.assertIn(
            "FAIL build settings file: ~/build.yaml is in layout 1, older than this package's layout 2 fix: "
            f"{self.rkmon} migrate --settings ~/build.yaml",
            said,
        )
        self.assertIn(
            "FAILED: 1 of 2 checks failed; nothing changed (fix each FAIL above)", said
        )
        self.assertEqual(self.builds(), [])

    def test_a_settings_file_of_the_image_and_the_deployment_in_one_is_refused(self):
        """A settings file from before the image had a file of its own (the image's settings and the deployment's
        under config:) is in the layout before layout 1: refused in migrate's words, as the build settings file's FAIL
        row, nothing built, and nothing written beside it.
        """
        old = self.d / "rkmon-settings.yaml"
        old.write_text(
            "rkmon_settings: 1\nconfig:\n  repo: acme/widgets\n"
            "  image:\n    base: ubuntu:24.04\n"
            f'  rocm:\n    installer_base: "{URL}"\n    version: "10.1.0-5"\n'
        )
        old.chmod(0o600)
        rc, out = self.run_("build", "--settings", old)
        self.assertEqual(rc, 4, out)
        said = one_line(out)
        w = self.pkg / "docker" / "rkmon"  # the name rkmon was run by
        self.assertIn(
            "FAIL build settings file: ~/rkmon-settings.yaml is in the layout before layout 1 (it starts with "
            "rkmon_settings), which this package does not read fix: start from a new file, "
            f"{w} template deploy NEW_FILE (a build file: {w} template build NEW_FILE), and copy your values into "
            "it",
            said,
        )
        self.assertIn(
            "FAILED: 1 of 2 checks failed; nothing changed (fix each FAIL above)",
            said,
        )
        self.assertEqual(self.builds(), [])
        self.assertEqual(
            sorted(p.name for p in self.d.glob("rkmon-settings*")), [old.name]
        )

    def test_the_root_certificates_reach_the_build_as_a_context_of_their_own(self):
        """Copies of the files operating_system.ca_certificates names, in a private folder of the build's own
        (ca-1.crt, ...), as the build context the Dockerfile's base stage trusts them from: never in the package
        folder, and the folder is gone after the build. Neither the command shown nor the output names your file or
        the folder.
        """
        cert = self.d / "corp-root.crt"
        cert.write_text(
            "-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n"
        )
        self.settings.write_text(
            self.build + f"operating_system:\n  ca_certificates: [{cert}]\n"
        )
        rc, out = self.run_("build", "--settings", self.settings)
        self.assertEqual(rc, 0, out)
        argv = self.builds()[-1]
        self.assertIn("-t rocke-monitor:t1", argv)
        folder = re.search(r"--build-context ca-certificates=(\S+) ", argv).group(1)
        self.assertFalse(pathlib.Path(folder).is_relative_to(self.pkg))
        self.assertFalse(pathlib.Path(folder).exists())  # removed after the build
        given = self.d / "contexts" / "ca-certificates"
        self.assertEqual([p.name for p in given.iterdir()], ["ca-1.crt"])
        self.assertEqual((given / "ca-1.crt").read_text(), cert.read_text())
        self.assertEqual(list((self.d / "tmp").iterdir()), [])
        self.assertIn("ca-certificates=<operating_system.ca_certificates>", out)
        for name in (cert.name, folder):
            self.assertNotIn(name, out)
        self.assertFalse(list(self.pkg.rglob("*.crt")))

    def test_a_wheel_that_cannot_be_read_builds_nothing(self):
        """What only the network gives is read before the build: the SDK's wheel that the server at
        claude_agent_sdk.package_url does not have is a FAIL row, in words that name no host, its fix the keys that
        pick another; nothing is built."""
        self.settings.write_text(self.build.replace(WHEEL, "none.whl"))
        rc, out = self.run_("build", "--settings", self.settings)
        self.assertEqual(rc, 4, out)
        said = one_line(out)
        self.assertIn(
            "FAIL claude_agent_sdk.version: the wheel at claude_agent_sdk.package_url could not be read (the server "
            "answered HTTP 404) fix: set claude_agent_sdk.version in ~/build.yaml to a release the package index "
            "has, or set claude_agent_sdk.package_url in",
            said,
        )
        self.assertNotIn("127.0.0.1", out)
        self.assertEqual(self.builds(), [])


class TestPkgs(unittest.TestCase):
    """docker/pkgs.sh against a stand-in apt-get and dnf that record their arguments."""

    def run_pkgs(self, os_release: str, *args, extra_env=None, apt_get=""):
        """pkgs.sh ARGS: its exit code, its output, and the stand-ins' record. APT_GET: the stand-in apt-get's lines
        after its record."""
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            bin_ = sandbox.safe_path(d)
            log = d / "log"
            for tool, then in (("apt-get", apt_get), ("dnf", "")):
                (pathlib.Path(bin_) / tool).write_text(
                    f'#!/bin/sh\necho "{tool} $*" >> "{log}"\n{then}', encoding="utf-8"
                )
                (pathlib.Path(bin_) / tool).chmod(0o755)
            osr = d / "os-release"
            osr.write_text(os_release, encoding="utf-8")
            p = subprocess.run(
                ["sh", str(DOCKER / "pkgs.sh"), *args],
                capture_output=True,
                text=True,
                timeout=60,
                env={"PATH": bin_, "PKGS_OS_RELEASE": str(osr), **(extra_env or {})},
            )
            return (
                p.returncode,
                p.stdout + p.stderr,
                log.read_text() if log.exists() else "",
            )

    def test_the_family_by_os_release(self):
        self.assertEqual(
            self.run_pkgs('ID=ubuntu\nVERSION_ID="24.04"\n', "family")[1], "deb\n"
        )
        self.assertEqual(
            self.run_pkgs('ID="rocky"\nID_LIKE="rhel centos fedora"\n', "family")[1],
            "rpm\n",
        )
        self.assertEqual(
            self.run_pkgs('ID="almalinux"\nID_LIKE="rhel"\n', "family")[1], "rpm\n"
        )
        rc, out, _ = self.run_pkgs("ID=alpine\n", "family")
        self.assertEqual(rc, 2)
        self.assertIn("unsupported base (alpine)", out)

    def test_one_list_of_names_for_apt_and_dnf(self):
        rc, _out, log = self.run_pkgs(
            "ID=ubuntu\n", "install", "openssh-client", "python3-dev", "g++", "xz-utils"
        )
        self.assertEqual(rc, 0)
        self.assertIn(
            "apt-get install -y --no-install-recommends openssh-client python3-dev g++ xz-utils",
            log,
        )
        rc, _out, log = self.run_pkgs(
            'ID="rocky"\nID_LIKE="rhel"\n',
            "install",
            "openssh-client",
            "python3-dev",
            "g++",
            "xz-utils",
            "python-is-python3",
            "procps",
            "iproute2",
        )
        self.assertEqual(rc, 0)
        self.assertIn(
            "dnf install -y --setopt=install_weak_deps=False --allowerasing openssh-clients python3-devel "
            "gcc-c++ xz procps-ng iproute",
            log,
        )

    def test_the_package_mirror_is_on_between_on_and_off_and_never_kept(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            apt, secrets = d / "apt", d / "secrets"
            (apt / "sources.list.d").mkdir(parents=True)
            secrets.mkdir()
            src = (
                "Types: deb\nURIs: http:/"
                + "/archive.ubuntu.com/ubuntu/\nSuites: noble\n"
            )
            (apt / "sources.list.d" / "ubuntu.sources").write_text(src)
            (apt / "sources.list.d" / "other.list").write_text(
                "deb https://ppa.launchpadcontent.net/x/ppa/ubuntu noble main\n"
            )
            env = {"PKGS_APT_DIR": str(apt), "SECRETS_DIR": str(secrets)}
            ubuntu = "ID=ubuntu\n"
            quiet = apt / "apt.conf.d" / "00rkmon-mirror-quiet"
            rc, out, _ = self.run_pkgs(ubuntu, "mirror", "on", extra_env=env)
            self.assertEqual(
                (rc, (apt / "sources.list.d" / "ubuntu.sources").read_text()), (0, src)
            )  # no secret
            self.assertFalse(quiet.exists())
            (secrets / "package_mirror").write_text(
                "http://apt.example.com/ubuntu-mirror/\n"
            )
            rc, out, _ = self.run_pkgs(ubuntu, "mirror", "on", extra_env=env)
            self.assertEqual(rc, 0, out)
            self.assertIn(
                "URIs: http://apt.example.com/ubuntu-mirror/\n",
                (apt / "sources.list.d" / "ubuntu.sources").read_text(),
            )
            self.assertNotIn("apt.example.com", out)
            self.assertNotIn(
                "apt.example.com", (apt / "sources.list.d" / "other.list").read_text()
            )
            # apt names each archive as it reads it: quiet (errors only) while it reads the mirror.
            self.assertEqual(quiet.read_text(), 'quiet "2";\n')
            rc, out, _ = self.run_pkgs(ubuntu, "mirror", "off", extra_env=env)
            self.assertEqual(
                (apt / "sources.list.d" / "ubuntu.sources").read_text(), src
            )
            self.assertFalse(quiet.exists())
            self.assertEqual(
                sorted(p.name for p in (apt / "sources.list.d").iterdir()),
                ["other.list", "ubuntu.sources"],
            )

    def test_a_package_mirror_on_a_base_that_is_not_ubuntu_stops_the_build(self):
        """The mirror is apt's: on a base of the RHEL family, `mirror on` with the build secret stops the step (exit
        2), naming the key and never the address, and changes nothing; without the secret it does nothing.
        """
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            apt, secrets = d / "apt", d / "secrets"
            secrets.mkdir()
            env = {"PKGS_APT_DIR": str(apt), "SECRETS_DIR": str(secrets)}
            rocky = 'ID="rocky"\nID_LIKE="rhel centos fedora"\n'
            self.assertEqual(
                self.run_pkgs(rocky, "mirror", "on", extra_env=env), (0, "", "")
            )
            (secrets / "package_mirror").write_text(
                "http://apt.example.com/ubuntu-mirror/\n"
            )
            rc, out, log = self.run_pkgs(rocky, "mirror", "on", extra_env=env)
            self.assertEqual(rc, 2)
            self.assertEqual(
                out,
                "pkgs.sh: the build was given a package mirror (operating_system.package_mirror), which is for the "
                "Ubuntu base images only: delete it from your build settings, or build on an Ubuntu base; nothing "
                "installed\n",
            )
            self.assertEqual(log, "")
            self.assertFalse(apt.exists())

    def test_apt_s_errors_show_the_mirror_by_its_secret_s_name(self):
        """apt's errors name the archives it reads: with the build's mirror, install shows it by its secret's name
        (docker/masked.sh), runs apt as before and keeps its exit code; without one, a host that is not a public one
        shows as <host>.
        """
        failing = (
            'echo "E: Failed to fetch http://apt.example.com/ubuntu-mirror/dists/noble/InRelease  Could not '
            'connect to apt.example.com:80 (192.0.2.9)" >&2\nexit 100\n'
        )
        ubuntu = "ID=ubuntu\n"
        with tempfile.TemporaryDirectory() as d:
            secrets = pathlib.Path(d)
            env = {"SECRETS_DIR": str(secrets)}
            self.assertEqual(
                self.run_pkgs(
                    ubuntu, "install", "curl", extra_env=env, apt_get=failing
                ),
                (
                    100,
                    "E: Failed to fetch <host>/ubuntu-mirror/dists/noble/InRelease  Could not connect to "
                    "<host>:80 (<address>)\n",
                    "apt-get update -q\n",
                ),
            )
            (secrets / "package_mirror").write_text(
                "http://apt.example.com/ubuntu-mirror/\n"
            )
            self.assertEqual(
                self.run_pkgs(
                    ubuntu, "install", "curl", extra_env=env, apt_get=failing
                ),
                (
                    100,
                    "E: Failed to fetch <package_mirror>/dists/noble/InRelease  Could not connect to "
                    "<package_mirror>:80 (<address>)\n",
                    "apt-get update -q\n",
                ),
            )
            self.assertEqual(
                self.run_pkgs(ubuntu, "install", "curl", extra_env=env),
                (
                    0,
                    "",
                    "apt-get update -q\napt-get install -y --no-install-recommends curl\n",
                ),
            )

    def test_its_hosts_are_the_build_plan_s(self):
        """The hosts pkgs.sh reads (Ubuntu's archives, its key server and the PPA) are rkmon_lib/build_plan.py
        UBUNTU_HOSTS, which docker/masked.sh shows as they are (TOOL_HOSTS)."""
        code = re.sub(
            r"(?m)^\s*#.*$", "", (DOCKER / "pkgs.sh").read_text(encoding="utf-8")
        )
        hosts = set()
        for names, rest in re.findall(
            r"https?\??://\(?([a-z|]+)\)?((?:\\?\.[a-z0-9-]+)+)", code
        ):
            hosts |= {name + rest.replace("\\", "") for name in names.split("|")}
        self.assertEqual(hosts, set(build_plan.UBUNTU_HOSTS))

    def test_certificates_are_trusted_before_the_first_download(self):
        pkgs = (DOCKER / "pkgs.sh").read_text(encoding="utf-8")
        rpm = pkgs.split("base() {", 1)[1].split("rpm)", 1)[1]
        self.assertLess(rpm.index("trust"), rpm.index("dnf install"))
        self.assertEqual(
            subprocess.run(["sh", "-n", str(DOCKER / "pkgs.sh")]).returncode, 0
        )


class TestNoRocmMode(unittest.TestCase):
    """Every image carries its own ROCm: no deployment chooses another, and older settings that did lose the keys with
    a line."""

    def test_the_older_keys_are_dropped_with_a_line(self):
        cfg, lines = renames.migrate(
            {"rocm": {"mode": "host", "host_dir": "/opt/rocm/core-9.9"}, "repo": "a/b"},
            "f",
        )
        self.assertEqual(cfg, {"repo": "a/b"})
        self.assertEqual(
            lines,
            [
                "rocm.mode in f dropped: every image carries its own ROCm",
                "rocm.host_dir in f dropped: every image carries its own ROCm, so no host's ROCm is mounted",
            ],
        )
        # And so no mount of a host's ROCm in the compose override: no key it is made from is ROCm's.
        self.assertFalse([k for k in deploy_plan.OVERRIDE_KEYS if "rocm" in k.lower()])

    def test_doctor_reads_the_image_s_record(self):
        rec = {
            "version": "10.1.0-5",
            "gfx": "gfx942",
            "clang": "AMD clang",
            "llvm_bin": "/opt/rocm/core-10.1/lib/llvm/bin",
        }
        c = doctor.image_rocm_check(rec)
        self.assertEqual(c.status, "PASS")
        self.assertIn("10.1.0-5 (gfx gfx942)", c.found)
        c = doctor.image_rocm_check({})
        self.assertEqual(c.status, "FAIL")
        self.assertIn("not built by this package's Dockerfile", c.found)
        # Its fix: an image of this package, named to the deployment as any new image is (image.tag, then the confirm,
        # whose next line asks for the image's load or pull while it is not here).
        self.assertEqual(
            str(c.fix),
            "run an image built by this package (QUICK_BUILD_GUIDE.md): set image.tag in your deploy settings file to "
            f"its tag, then {doctor.wrapper_name()} confirm",
        )


if __name__ == "__main__":
    unittest.main()

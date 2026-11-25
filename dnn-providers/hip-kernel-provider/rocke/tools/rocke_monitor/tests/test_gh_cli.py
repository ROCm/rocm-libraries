# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The GitHub CLI the package relies on: one resolver for every gh call (paths.gh, else PATH; the read login's
folder), the checks of its version, login and scopes against a fake gh, and its install (deploy.gh_install, which
the image's build runs), against a fake release server with a good and a bad checksum. No real download, no real gh.
"""

import hashlib
import http.server
import io
import os
import pathlib
import sys
import tarfile
import tempfile
import threading
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import sandbox  # noqa: E402
from rkmon_lib import config, deploy, doctor, gh  # noqa: E402

FAKE_GH = """#!/bin/sh
case "$1" in
  --version) echo "gh version 2.46.0 (2025-12-13)";;
  auth) echo "github.com"; echo "  Logged in to github.com account op-login (keyring)";
        echo "  - Token scopes: 'gist', 'read:org', 'repo'"; echo "  config: $GH_CONFIG_DIR";;
esac
"""


def pkg_cfg(d: pathlib.Path, **over) -> dict:
    c = config.load_defaults()
    c.update(
        repo="ROCm/r",
        rocke_path="a/b/rocke",
        operator={"github_login": "op-login"},
        scope={"labels": ["x"], "path_labels": []},
        **over,
    )
    c["paths"] = dict(c["paths"], working_dir=str(d / "work"))
    return config.normalize(c)  # with the places the load derives


class TestResolver(unittest.TestCase):
    def test_binary_and_login_folder(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(
            os.environ, {"RKMON_CONTAINER": ""}
        ):  # a host
            c = pkg_cfg(pathlib.Path(d))
            self.assertEqual(
                gh.settings(c), ("gh", {"GH_CONFIG_DIR": f"{d}/work/.config/gh-read"})
            )
            c["paths"]["gh"] = f"{d}/work/tools/gh/gh"
            self.assertEqual(gh.settings(c)[0], f"{d}/work/tools/gh/gh")
            env = config.env(config.normalize(c))
            self.assertEqual(
                (env["RKMON_GH"], env["GH_CONFIG_DIR"]),
                (f"{d}/work/tools/gh/gh", f"{d}/work/.config/gh-read"),
            )

    def test_in_the_image_every_call_and_every_job_run_gh_on_path(self):
        """gh on PATH there is the GitHub request cap's counter (docker/rk_gh.py): a set paths.gh would run uncounted,
        so it never runs, and the checks refuse it."""
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(
            os.environ, {"RKMON_CONTAINER": "1"}
        ):
            c = pkg_cfg(pathlib.Path(d))
            c["paths"]["gh"] = "/opt/private/gh"
            self.assertEqual(gh.settings(c)[0], "gh")
            self.assertEqual(config.env(config.normalize(c))["RKMON_GH"], "gh")
            self.assertIn(
                "paths.gh is '/opt/private/gh'; in the container every gh call goes through the image's own gh, "
                "behind the GitHub request cap's counter on PATH: blank paths.gh",
                config.problems(config.normalize(c)),
            )
            c["paths"]["gh"] = ""
            self.assertFalse(
                [p for p in config.problems(config.normalize(c)) if "paths.gh" in p]
            )
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(
            os.environ, {"RKMON_CONTAINER": ""}
        ):  # a host: the gh the configuration names
            c = pkg_cfg(pathlib.Path(d))
            c["paths"]["gh"] = "/opt/private/gh"
            self.assertEqual(gh.settings(c)[0], "/opt/private/gh")
            self.assertFalse(
                [p for p in config.problems(config.normalize(c)) if "paths.gh" in p]
            )

    def test_reads_use_the_configured_binary_and_the_read_folder(self):
        seen = {}

        def fake(cmd, **kw):
            seen.update(cmd=cmd, env=kw.get("env"))
            return mock.Mock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as d, mock.patch.object(
            gh.subprocess, "run", fake
        ):
            c = pkg_cfg(pathlib.Path(d))
            c["paths"]["gh"] = "/opt/x/gh"
            gh.configure(c)
            try:
                gh.run_read(["pr", "view", "1"])
            finally:
                gh.configure({})
        self.assertEqual(seen["cmd"][:3], ["/opt/x/gh", "pr", "view"])
        self.assertTrue(seen["env"]["GH_CONFIG_DIR"].endswith("/work/.config/gh-read"))
        self.assertEqual((gh.BIN, gh.READ_DIR, gh.WRITE_DIR), ("gh", "", ""))


class TestChecks(unittest.TestCase):
    def test_version_login_and_scopes_from_a_fake_gh(self):
        with tempfile.TemporaryDirectory() as d:
            fake = pathlib.Path(d) / "gh"
            fake.write_text(FAKE_GH)
            fake.chmod(0o755)
            c = pkg_cfg(
                pathlib.Path(d),
                features=dict(
                    config.load_defaults()["features"],
                    review_automation={"enabled": True, "mode": "live"},
                ),
            )
            c["paths"]["gh"] = str(fake)
            g = doctor.gh_status(c)
            self.assertEqual(
                (g["version"], g["version_ok"], g["login"], g["logged_in"]),
                ("2.46.0", True, "op-login", True),
            )
            self.assertEqual(g["scopes"], ["gist", "read:org", "repo"])
            self.assertTrue(g["config_dir"].endswith("/work/.config/gh-read"))
            self.assertEqual(
                doctor.gh_needed_scopes(c), ["read:org", "repo or public_repo"]
            )
            check = next(
                x for x in doctor.prerequisites(c) if x.name == "gh (GitHub CLI)"
            )
            self.assertEqual(check.status, "PASS", check)
            old = pathlib.Path(d) / "gh-old"
            old.write_text(FAKE_GH.replace("2.46.0", "2.20.1"))
            old.chmod(0o755)
            c["paths"]["gh"] = str(old)
            check = next(
                x for x in doctor.prerequisites(c) if x.name == "gh (GitHub CLI)"
            )
            self.assertEqual(check.status, "FAIL")
            self.assertIn("older than 2.40", check.found)
            c["paths"]["gh"] = str(pathlib.Path(d) / "none")
            self.assertEqual(
                next(
                    x for x in doctor.prerequisites(c) if x.name == "gh (GitHub CLI)"
                ).status,
                "FAIL",
            )


class _Release(http.server.BaseHTTPRequestHandler):
    files: dict = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = self.files.get(self.path.split("?", 1)[0])
        if isinstance(body, tuple):  # (status, Location): a refusal or a redirect
            self.send_response(body[0])
            if body[1]:
                self.send_header("Location", body[1])
            self.end_headers()
            return
        self.send_response(200 if body is not None else 404)
        self.end_headers()
        self.wfile.write(body or b"")


class TestInstall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Release)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def archive(self, version: str, gh: bytes = b"#!/bin/sh\necho fake gh\n") -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            for name, data in (
                (f"gh_{version}_linux_amd64/bin/gh", gh),
                (f"gh_{version}_linux_amd64/LICENSE", b"license\n"),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o644
                t.addfile(info, io.BytesIO(data))
        return buf.getvalue()

    def serve(self, version: str, good: bool = True, gh: bytes = b""):
        data = self.archive(version, gh) if gh else self.archive(version)
        digest = hashlib.sha256(data if good else b"other").hexdigest()
        _Release.files = {
            f"/download/v{version}/gh_{version}_linux_amd64.tar.gz": data,
            f"/download/v{version}/gh_{version}_checksums.txt": f"{digest}  gh_{version}_linux_amd64.tar.gz\n0000  gh_{version}_linux_arm64.tar.gz\n".encode(),
            "/latest": b'{"tag_name": "v' + version.encode() + b'"}',
        }

    def test_a_verified_archive_puts_only_the_binary_in_place(self):
        self.serve("9.9.9")
        with tempfile.TemporaryDirectory() as d, mock.patch.object(
            deploy.platform, "machine", lambda: "x86_64"
        ):
            out = deploy.gh_install(
                f"{d}/tools/gh", base=self.base, latest=f"{self.base}/latest"
            )
            self.assertEqual(out, pathlib.Path(d) / "tools" / "gh" / "gh")
            self.assertTrue(os.access(out, os.X_OK))
            self.assertEqual(
                sorted(p.name for p in out.parent.iterdir()), ["gh"]
            )  # nothing else unpacked

    def test_the_newest_release_when_the_api_refuses(self):
        """The API's rate limit (an image build has no login): the release page's redirect names the version."""
        self.serve("9.9.7")
        _Release.files.update(
            {
                "/api-latest": (403, ""),
                "/latest": (302, "/tag/v9.9.7"),
                "/tag/v9.9.7": b"release page",
            }
        )
        self.assertEqual(
            deploy.latest_gh(self.base, f"{self.base}/api-latest", timeout=10), "9.9.7"
        )
        with tempfile.TemporaryDirectory() as d, mock.patch.object(
            deploy.platform, "machine", lambda: "x86_64"
        ):
            out = deploy.gh_install(
                f"{d}/tools/gh", base=self.base, latest=f"{self.base}/api-latest"
            )
            self.assertTrue(os.access(out, os.X_OK))
        _Release.files["/latest"] = (404, "")
        with self.assertRaises(deploy.DeployError) as cm:
            deploy.latest_gh(self.base, f"{self.base}/api-latest", timeout=10)
        self.assertIn("name a version (github_cli.version)", str(cm.exception))

    def test_an_exact_archive_is_checked_against_its_sha256(self):
        """github_cli.package_url names the exact archive and github_cli.package_sha256 its digest: a match installs it,
        a mismatch installs nothing and names the digest it got, never the address."""
        self.serve("9.9.6")
        url = f"{self.base}/download/v9.9.6/gh_9.9.6_linux_amd64.tar.gz"
        good = hashlib.sha256(self.archive("9.9.6")).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            out = deploy.gh_install(f"{d}/a", url=url, sha256=good.upper())
            self.assertTrue(os.access(out, os.X_OK))
            with self.assertRaises(deploy.DeployError) as cm:
                deploy.gh_install(f"{d}/b", url=url, sha256="0" * 64)
            self.assertIn(
                f"SHA-256 is {good}, not github_cli.package_sha256", str(cm.exception)
            )
            self.assertNotIn("127.0.0.1", str(cm.exception))
            self.assertFalse((pathlib.Path(d) / "b" / "gh").exists())

    def test_an_exact_archive_without_its_sha256_reads_the_published_checksums(self):
        url = f"{self.base}/download/v9.9.5/gh_9.9.5_linux_amd64.tar.gz"
        with tempfile.TemporaryDirectory() as d:
            self.serve("9.9.5")
            self.assertTrue(os.access(deploy.gh_install(f"{d}/a", url=url), os.X_OK))
            self.serve("9.9.5", good=False)
            with self.assertRaises(deploy.DeployError) as cm:
                deploy.gh_install(f"{d}/b", url=url)
            self.assertIn("does not match the release's checksum", str(cm.exception))
            with self.assertRaises(deploy.DeployError) as cm:
                deploy.gh_install(f"{d}/c", url=f"{self.base}/gh.tgz")
            self.assertIn("set github_cli.package_sha256", str(cm.exception))

    def test_an_address_with_a_query_and_a_fragment(self):
        """The archive's file name, and the folder its published checksums are in, come from the address's path."""
        self.serve("9.9.3")
        url = f"{self.base}/download/v9.9.3/gh_9.9.3_linux_amd64.tar.gz?sig=1#part"
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(os.access(deploy.gh_install(f"{d}/a", url=url), os.X_OK))

    def test_a_failed_download_names_no_host(self):
        """A download that fails says why in fixed words: the error's own text can name the host."""
        self.serve("9.9.2")
        closed = sandbox.closed_port(self)
        with tempfile.TemporaryDirectory() as d:
            for url, why in (
                (
                    f"{self.base}/download/v9.9.2/none.tar.gz",
                    "the server answered HTTP 404",
                ),
                (
                    f"http://127.0.0.1:{closed}/gh_9.9.2_linux_amd64.tar.gz",
                    "the host refused the connection",
                ),
                (
                    "nothing://127.0.0.1/gh_9.9.2_linux_amd64.tar.gz",
                    "the address is no URL it can read",
                ),
            ):
                with self.subTest(why=why):
                    with self.assertRaises(deploy.DeployError) as cm:
                        deploy.gh_install(f"{d}/a", url=url, sha256="0" * 64)
                    self.assertEqual(
                        str(cm.exception),
                        f"github_cli.package_url's archive did not download ({why}): nothing installed",
                    )

    def test_each_failure_in_fixed_words(self):
        import http.client
        import socket
        import ssl
        import urllib.error

        host = "private.example.com"
        for exc, why in (
            (
                urllib.error.HTTPError(f"https://{host}/x", 403, "Forbidden", {}, None),
                "the server answered HTTP 403",
            ),
            (
                urllib.error.URLError(
                    socket.gaierror(-2, f"Name or service not known: {host}")
                ),
                "the host's name does not resolve",
            ),
            (
                urllib.error.URLError(
                    ssl.SSLCertVerificationError(1, f"Hostname mismatch: {host}")
                ),
                "the server's certificate is not one this machine trusts (a network that inspects TLS: name its root "
                "certificates in operating_system.ca_certificates)",
            ),
            (urllib.error.URLError(ssl.SSLError(1, host)), "the TLS handshake failed"),
            (
                urllib.error.URLError(ConnectionRefusedError(111, host)),
                "the host refused the connection",
            ),
            (urllib.error.URLError(TimeoutError(host)), "it timed out"),
            (TimeoutError(host), "it timed out"),
            (
                ConnectionResetError(104, host),
                "the connection broke during the download",
            ),
            (
                http.client.IncompleteRead(b"x", 10),
                "the download ended before the whole file came",
            ),
            (
                ValueError(f"unknown url type: {host}"),
                "the address is no URL it can read",
            ),
            (urllib.error.URLError(f"no host {host}"), "it could not be downloaded"),
            # The host's fetch says why already, in the same words.
            (
                deploy.FetchError("the server answered HTTP 502"),
                "the server answered HTTP 502",
            ),
        ):
            with self.subTest(exc=type(exc).__name__, why=why):
                self.assertEqual(deploy.download_failure(exc), why)
                self.assertNotIn(host, why)

    def test_the_version_gh_must_report(self):
        reports = b"#!/bin/sh\necho 'gh version 9.9.4 (2026-01-01)'\n"
        self.serve("9.9.4", gh=reports)
        url = f"{self.base}/download/v9.9.4/gh_9.9.4_linux_amd64.tar.gz"
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(
                os.access(
                    deploy.gh_install(f"{d}/a", version="9.9.4", url=url), os.X_OK
                )
            )
            with self.assertRaises(deploy.DeployError) as cm:
                deploy.gh_install(f"{d}/b", version="9.9.3", url=url)
            self.assertIn(
                "reports version 9.9.4, not github_cli.version (9.9.3)",
                str(cm.exception),
            )
            self.assertFalse((pathlib.Path(d) / "b" / "gh").exists())

    def test_a_checksum_mismatch_installs_nothing(self):
        self.serve("9.9.8", good=False)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(
            deploy.platform, "machine", lambda: "x86_64"
        ):
            with self.assertRaises(deploy.DeployError) as cm:
                deploy.gh_install(f"{d}/tools/gh", version="9.9.8", base=self.base)
            self.assertIn("checksum", str(cm.exception))
            self.assertFalse((pathlib.Path(d) / "tools" / "gh" / "gh").exists())


if __name__ == "__main__":
    unittest.main()

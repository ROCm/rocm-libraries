# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""From build settings to a docker build (rkmon_lib/build_plan.py): the package's build defaults,
one home each; the installers' names, a release's newest build in a folder's listing, and the clang commit each
release must name; the plan a build settings file makes, the one network step that resolves it, against a fake
network, and the record and the labels the image carries; the one docker build of a resolved plan, with the flags
it lists and no other; and the downloads' fetch (rkmon_lib/deploy.py fetcher), which asks again after a passing
failure and asks for a path with its dot segments resolved. No real download.
"""

import hashlib
import io
import json
import os
import pathlib
import random
import shlex
import socket
import ssl
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse
import zipfile
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import build_plan, deploy, file_types, image, settings_file  # noqa: E402

SHA256 = {
    name: hashlib.sha256(name.encode()).hexdigest()
    for name in ("rocm", "atlassian", "gh", "sdk")
}
DATE = "2026-11-19T10:00:00Z"
PACKAGE = "2026.11.19"
FOLDER = (
    "https://rocm.example.com/installers"  # AMD's public folder, as the tests move it
)
FILES = "https://files.example.com/packages"  # where the fake package index's files are
PAGE = f"{build_plan.PUBLIC_PYPI}/{build_plan.SDK_PROJECT}/"
GH = "2.81.0"
LABELS = {
    "org.opencontainers.image.title",
    "org.opencontainers.image.description",
    "org.opencontainers.image.licenses",
    "org.opencontainers.image.version",
    "org.opencontainers.image.created",
    "org.rocke.package.version",
    "org.rocke.build.date",
    "org.rocke.base",
    "org.rocke.rocm.version",
    "org.rocke.rocm.clang_commit",
    "org.rocke.rocm.library_targets",
    "org.rocke.atlassian_cli.version",
    "org.rocke.github_cli.version",
    "org.rocke.claude_agent_sdk.version",
    "org.rocke.claude_code.version",
    "org.rocke.build_settings.fingerprint",
}


def build_doc(**sections) -> settings_file.Document:
    """A build settings document in this layout: rocm.version and the atlassian tool's two required keys, then each
    section of SECTIONS laid over them."""
    values = {
        "rocm": {"version": "10.1.0"},
        "atlassian_cli": {
            "package_url": "https://tools.example.com/atlassian",
            "package_sha256": SHA256["atlassian"],
        },
    }
    for section, keys in sections.items():
        values.setdefault(section, {}).update(keys)
    return settings_file.Document(file_types.BUILD, file_types.BUILD.version, values)


def make_plan(machine: str = "x86_64", **sections):
    return build_plan.plan(build_doc(**sections), PACKAGE, DATE, machine)


def wheel(version: str = "0.3.0", cli: str = "2.1.400") -> bytes:
    """A wheel of the SDK: a large file first, as the real wheel's bundled Claude Code is, then the file that names
    the Claude Code version (none when CLI is blank), and its METADATA."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "claude_agent_sdk/_bundled/claude", random.Random(7).randbytes(400_000)
        )
        z.writestr("claude_agent_sdk/__init__.py", "")
        if cli:
            z.writestr(build_plan.SDK_CLI_VERSION_FILE, f'__cli_version__ = "{cli}"\n')
        z.writestr(
            f"claude_agent_sdk-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: claude-agent-sdk\nVersion: {version}\n",
        )
    return buf.getvalue()


def wheel_name(version: str, platform: str) -> str:
    return f"claude_agent_sdk-{version}-py3-none-{platform}.whl"


def index_page(*files) -> bytes:
    """A PEP 503 page of claude-agent-sdk: a link to FILES/NAME for each (NAME, yanked)."""
    rows = []
    for name, yanked in files:
        mark = ' data-yanked=""' if yanked else ""
        rows.append(f'<a href="{FILES}/{name}#sha256={"0" * 64}"{mark}>{name}</a><br/>')
    return (
        "<!DOCTYPE html><html><body>\n" + "\n".join(rows) + "\n</body></html>"
    ).encode()


class Ranged(bytes):
    """A file the fake network serves in parts, as a server that takes range requests does."""


class Web:
    """A fake network for resolve: each address's answer (bytes: the whole file; Ranged: in parts; None: no such
    file), each request kept as (address, byte range). An address it does not have answers 404.
    """

    def __init__(self, pages: dict):
        self.pages, self.asked = dict(pages), []

    def __call__(self, url: str, byte_range: str = "") -> deploy.Answer:
        self.asked.append((url, byte_range))
        body = self.pages.get(url)
        if body is None:
            raise deploy.FetchError("the server answered HTTP 404")
        if byte_range and isinstance(body, Ranged):
            start, _, end = byte_range.partition("-")
            size = len(body)
            first = max(0, size - int(end)) if not start else int(start)
            last = size - 1 if not start else min(int(end), size - 1)
            return deploy.Answer(
                206,
                url,
                {"content-range": f"bytes {first}-{last}/{size}"},
                bytes(body[first : last + 1]),
            )
        return deploy.Answer(200, url, {}, bytes(body))

    def urls(self) -> list:
        return [u for u, _ in self.asked]


def sdk_pages() -> dict:
    """The fake package index's SDK: releases 0.2.0 and 0.3.0 (for x86-64, aarch64 and macOS), a pre-release, a yanked
    0.5.0 and a source archive. Each wheel names its own Claude Code version, so a test can tell which was read.
    """
    x86, arm = "manylinux_2_17_x86_64", "manylinux_2_17_aarch64"
    return {
        PAGE: index_page(
            (wheel_name("0.2.0", x86), False),
            (wheel_name("0.3.0", "macosx_11_0_arm64"), False),
            (wheel_name("0.3.0", arm), False),
            (wheel_name("0.3.0", x86), False),
            (wheel_name("0.4.0rc1", x86), False),
            (wheel_name("0.5.0", x86), True),
            ("claude_agent_sdk-0.3.0.tar.gz", False),
        ),
        f"{FILES}/{wheel_name('0.2.0', x86)}": Ranged(wheel("0.2.0", "2.1.300")),
        f"{FILES}/{wheel_name('0.3.0', x86)}": Ranged(wheel("0.3.0", "2.1.400")),
        f"{FILES}/{wheel_name('0.3.0', arm)}": Ranged(wheel("0.3.0", "2.1.401")),
    }


def full_web(**more) -> Web:
    """A fake network on which every value of a plan with blank versions resolves: ROCm 10.1.0's builds -3 and -12, gh
    2.81.0 and its checksums, the SDK's releases. MORE: other answers by address (None: no such file).
    """
    listing = " ".join(
        f'<a href="rocm-installer-{v}.run">rocm-installer-{v}.run</a>'
        for v in ("10.1.0-3", "10.1.0-12", "10.2.0-1")
    )
    sums = "".join(
        f"{SHA256['gh'] if arch == 'amd64' else '1' * 64}  {deploy.gh_file(GH, arch)}\n"
        for arch in ("amd64", "arm64")
    )
    pages = {
        FOLDER + "/": listing.encode(),
        deploy.GH_LATEST: json.dumps({"tag_name": f"v{GH}"}).encode(),
        deploy.gh_download(GH, deploy.gh_checksums(GH)): sums.encode(),
        **sdk_pages(),
    }
    pages.update(more)
    return Web(pages)


# Private addresses, as a build settings file may name them, each download with its SHA-256: none may reach the
# record, the labels or a message.
PRIVATE = {
    "operating_system": {"package_mirror": "http://mirror.private.example.com/ubuntu"},
    "rocm": {
        "package_url": "https://files.private.example.com/rocm/rocm-installer-10.1.0-5.run",
        "package_sha256": SHA256["rocm"],
    },
    "atlassian_cli": {
        "package_url": "https://files.private.example.com/tools/atlassian-bin"
    },
    "github_cli": {
        "version": "2.80.0",
        "package_url": "https://files.private.example.com/gh/gh_2.80.0_linux_amd64.tar.gz",
        "package_sha256": SHA256["gh"],
    },
    "claude_agent_sdk": {
        "package_url": f"https://files.private.example.com/wheels/{wheel_name('0.3.0', 'manylinux_2_17_x86_64')}",
        "package_sha256": SHA256["sdk"],
    },
    "python_packages": {"package_index": "https://pypi.private.example.com/simple/"},
    "image": {"registry": "registry.private.example.com/team"},
}


def with_ca(sections: dict, files) -> dict:
    """SECTIONS with operating_system.ca_certificates naming FILES."""
    return dict(
        sections,
        operating_system=dict(
            sections.get("operating_system") or {},
            ca_certificates=[str(f) for f in files],
        ),
    )


def private_web(sections: dict) -> Web:
    """The network of a file that names every download: the SDK's wheel at its own address."""
    return Web(
        {sections["claude_agent_sdk"]["package_url"]: Ranged(wheel("0.3.0", "2.1.400"))}
    )


class PlanCase(unittest.TestCase):
    """AMD's public folder of ROCm installers moved to the fake network's FOLDER."""

    def setUp(self):
        moved = mock.patch.dict(os.environ, {"ROCM_PUBLIC_INSTALLERS": FOLDER})
        moved.start()
        self.addCleanup(moved.stop)


class TestPackageDefaults(unittest.TestCase):
    def test_the_two_target_lists_are_two_values(self):
        """ROCm's libraries are installed for one target; the build's hipcc check and rk doctor's compile for two."""
        self.assertEqual(build_plan.ROCM_LIBRARY_TARGETS, ("gfx942",))
        self.assertEqual(build_plan.HIP_CHECK_TARGETS, ("gfx942", "gfx950"))

    def test_amds_public_folder_moves_only_for_the_tests(self):
        with mock.patch.dict(os.environ, {"ROCM_PUBLIC_INSTALLERS": ""}):
            self.assertEqual(
                build_plan.public_installers(), build_plan.PUBLIC_ROCM_INSTALLERS
            )
        with mock.patch.dict(
            os.environ,
            {"ROCM_PUBLIC_INSTALLERS": "https://installers.example.com/rocm/"},
        ):
            self.assertEqual(
                build_plan.public_installers(), "https://installers.example.com/rocm"
            )


class TestInstallers(unittest.TestCase):
    def test_a_release_names_its_newest_build(self):
        listing = " ".join(
            f'<a href="rocm-installer-{v}.run">'
            for v in ("10.1.0-3", "10.1.0-12", "10.2.0-1", "10.1.0-7", "9.9.0-1")
        )
        self.assertEqual(build_plan.installer_build(listing, "10.1.0"), "10.1.0-12")
        self.assertEqual(build_plan.installer_build(listing, "10.1.0-3"), "10.1.0-3")
        self.assertEqual(build_plan.installer_build(listing, "10.3.0"), "")
        self.assertEqual(build_plan.installer_build("", "10.1.0"), "")

    def test_a_release_and_its_builds(self):
        self.assertEqual(build_plan.release_of("10.1.0-5"), "10.1.0")
        self.assertEqual(build_plan.release_of("10.1.0"), "10.1.0")
        self.assertEqual(
            sorted(["10.1.0-12", "10.1.0-3", "9.10.0-1"], key=build_plan.version_key),
            ["9.10.0-1", "10.1.0-3", "10.1.0-12"],
        )

    def test_the_gate_compilers_commit_only_for_its_release(self):
        """The release the gates are tuned on, or any build of it, names the gate compiler's commit; another release
        names none, so the build takes its clang as it comes."""
        commit = build_plan.ROCM_CLANG_COMMIT
        self.assertRegex(commit, r"^[0-9a-f]{40}$")
        self.assertEqual(build_plan.clang_commit(build_plan.ROCM_RELEASE), commit)
        self.assertEqual(
            build_plan.clang_commit(build_plan.ROCM_RELEASE + "-7"), commit
        )
        self.assertEqual(build_plan.clang_commit("10.2.0"), "")
        self.assertEqual(build_plan.clang_commit("10.2.0-1"), "")


class TestPlan(PlanCase):
    def test_a_file_of_another_type_or_layout_is_refused(self):
        for doc, words in (
            (
                settings_file.Document(
                    file_types.DEPLOY, file_types.DEPLOY.version, {}
                ),
                f"not deploy settings in layout {file_types.DEPLOY.version}",
            ),
            (
                settings_file.Document(file_types.BUILD, 1, {}),
                "not build settings in layout 1",
            ),
        ):
            with self.subTest(words=words):
                with self.assertRaises(ValueError) as cm:
                    build_plan.plan(doc, PACKAGE, DATE, "x86_64")
                self.assertIn(
                    f"the plan reads build settings in layout {file_types.BUILD.version}",
                    str(cm.exception),
                )
                self.assertIn(words, str(cm.exception))

    def test_each_argument_in_the_dockerfiles_order(self):
        """Every argument of ARGS, from its key or the package's defaults: each SHA-256 in lower case, each tool
        version without a leading v, the gate compiler's commit for its release only."""
        p = make_plan(
            operating_system={"base_image": "rockylinux:9"},
            rocm={"version": "10.1.0-5", "package_sha256": SHA256["rocm"].upper()},
            github_cli={"version": "v2.80.0", "package_sha256": SHA256["gh"]},
            claude_agent_sdk={"version": "v0.3.0"},
            atlassian_cli={"version": "1.2.3"},
        )
        self.assertEqual(list(p.args), [a.name for a in build_plan.ARGS])
        self.assertEqual(
            p.args,
            {
                "BASE_IMAGE": "rockylinux:9",
                "ROCM_VERSION": "10.1.0-5",
                "ROCM_PACKAGE_SHA256": SHA256["rocm"],
                "ROCM_CLANG_COMMIT": build_plan.ROCM_CLANG_COMMIT,
                "ROCM_LIBRARY_TARGETS": "gfx942",
                "HIP_CHECK_TARGETS": "gfx942,gfx950",
                "ATLASSIAN_CLI_VERSION": "1.2.3",
                "ATLASSIAN_CLI_PACKAGE_SHA256": SHA256["atlassian"],
                "GITHUB_CLI_VERSION": "2.80.0",
                "GITHUB_CLI_PACKAGE_SHA256": SHA256["gh"],
                "CLAUDE_AGENT_SDK_VERSION": "0.3.0",
                "CLAUDE_AGENT_SDK_PACKAGE_SHA256": "",
                "CLAUDE_CODE_VERSION": "",
            },
        )
        self.assertEqual(
            make_plan(rocm={"version": "10.2.0"}).args["ROCM_CLANG_COMMIT"], ""
        )
        self.assertEqual(
            make_plan().args["BASE_IMAGE"], "ubuntu:26.04"
        )  # the schema's default

    def test_the_downloads_the_build_is_given(self):
        """Each download location the file names, as a build secret that is not public; a blank package index is
        pypi.org's, a public default; a blank mirror and a blank SDK wheel give none; gh's public archive for a
        version the file names, for this machine."""
        p = make_plan(
            github_cli={"version": "2.80.0"},
            operating_system={"package_mirror": "http://mirror.example.com/ubuntu"},
        )
        self.assertEqual(
            [(s.id, s.key, s.public) for s in p.secrets],
            [
                ("package_mirror", "operating_system.package_mirror", False),
                ("atlassian_cli_package_url", "atlassian_cli.package_url", False),
                ("github_cli_package_url", "github_cli.package_url", True),
                ("python_package_index", "python_packages.package_index", True),
            ],
        )
        values = {s.id: s.value for s in p.secrets}
        self.assertEqual(values["python_package_index"], build_plan.PUBLIC_PYPI)
        self.assertEqual(
            values["github_cli_package_url"],
            f"{deploy.GH_RELEASES}/download/v2.80.0/gh_2.80.0_linux_amd64.tar.gz",
        )
        arm = make_plan(machine="aarch64", github_cli={"version": "2.80.0"})
        self.assertTrue(
            {s.id: s.value for s in arm.secrets}["github_cli_package_url"].endswith(
                "gh_2.80.0_linux_arm64.tar.gz"
            )
        )
        # A file that names each download: each is its own, none public, in DOWNLOADS' order.
        own = make_plan(**PRIVATE).secrets
        self.assertEqual([s.id for s in own], [d.id for d in build_plan.DOWNLOADS])
        self.assertEqual({s.public for s in own}, {False})

    def test_what_only_the_network_gives_waits_for_resolve(self):
        p = make_plan()
        self.assertEqual(
            p.unresolved,
            tuple(
                build_plan.Unresolved(k, build_plan.PENDING)
                for k in (
                    "rocm.version",
                    "github_cli.version",
                    "github_cli.package_sha256",
                    "claude_agent_sdk.version",
                )
            ),
        )
        # A file that names each download and pins gh leaves only the SDK's Claude Code version to the network.
        self.assertEqual(
            [u.key for u in make_plan(**PRIVATE).unresolved],
            ["claude_agent_sdk.version"],
        )
        # gh's own archive, unpinned: its version is the archive's.
        own = make_plan(
            github_cli={
                "package_url": "https://files.example.com/gh_2.79.1_linux_amd64.tar.gz",
                "package_sha256": SHA256["gh"],
            }
        )
        self.assertEqual(own.args["GITHUB_CLI_VERSION"], "2.79.1")
        self.assertNotIn("github_cli.version", [u.key for u in own.unresolved])

    def test_a_machine_github_has_no_gh_for(self):
        p = make_plan(machine="riscv64")
        row = next(u for u in p.unresolved if u.key == "github_cli.package_url")
        self.assertEqual(
            row.why,
            "GitHub publishes no gh for this machine's architecture (riscv64): set github_cli.package_url and "
            "github_cli.package_sha256",
        )
        self.assertNotIn("github_cli.version", [u.key for u in p.unresolved])

    def test_a_placeholder_counts_as_blank(self):
        p = make_plan(
            atlassian_cli={"package_url": "<URL>", "package_sha256": "<sha256>"}
        )
        self.assertNotIn("atlassian_cli_package_url", [s.id for s in p.secrets])
        self.assertEqual(p.args["ATLASSIAN_CLI_PACKAGE_SHA256"], "")

    def test_the_image_and_its_tag(self):
        """The tag: image.tag, else the package's version, else "local"; the registry only for a push."""
        self.assertEqual(
            make_plan().image, build_plan.Image("", image.DEFAULT_NAME, PACKAGE)
        )
        p = make_plan(
            image={
                "name": "team/monitor",
                "tag": "t1",
                "registry": "registry.example.com/x",
            }
        )
        self.assertEqual(
            p.image, build_plan.Image("registry.example.com/x", "team/monitor", "t1")
        )
        self.assertEqual(p.image.ref, "team/monitor:t1")
        local = build_plan.plan(build_doc(), "", DATE, "x86_64")
        self.assertEqual(local.image.tag, "local")

    def test_the_root_certificates_the_file_names(self):
        p = make_plan(**with_ca({}, ["~/certs/root.pem", "/etc/x.pem"]))
        self.assertEqual(
            p.ca_files,
            (
                pathlib.Path(os.path.expanduser("~/certs/root.pem")),
                pathlib.Path("/etc/x.pem"),
            ),
        )


class TestResolve(PlanCase):
    def test_every_blank_resolved(self):
        p = build_plan.resolve(make_plan(), full_web())
        self.assertEqual(p.unresolved, ())
        self.assertEqual(
            {
                k: p.args[k]
                for k in (
                    "ROCM_VERSION",
                    "GITHUB_CLI_VERSION",
                    "GITHUB_CLI_PACKAGE_SHA256",
                )
            },
            {
                "ROCM_VERSION": "10.1.0-12",
                "GITHUB_CLI_VERSION": GH,
                "GITHUB_CLI_PACKAGE_SHA256": SHA256["gh"],
            },
        )
        self.assertEqual(
            (p.args["CLAUDE_AGENT_SDK_VERSION"], p.args["CLAUDE_CODE_VERSION"]),
            ("0.3.0", "2.1.400"),
        )
        self.assertEqual(
            p.newest,
            (("github_cli.version", GH), ("claude_agent_sdk.version", "0.3.0")),
        )
        secrets = {s.id: s for s in p.secrets}
        self.assertEqual(
            secrets["rocm_package_url"],
            build_plan.Secret(
                "rocm_package_url",
                "rocm.package_url",
                f"{FOLDER}/rocm-installer-10.1.0-12.run",
                True,
            ),
        )
        self.assertEqual(
            secrets["github_cli_package_url"],
            build_plan.Secret(
                "github_cli_package_url",
                "github_cli.package_url",
                deploy.gh_download(GH, deploy.gh_file(GH, "amd64")),
                True,
            ),
        )
        # The labels follow what resolve found.
        self.assertEqual(p.labels["org.rocke.rocm.version"], "10.1.0-12")
        self.assertEqual(p.labels["org.rocke.claude_code.version"], "2.1.400")

    def test_a_rocm_release_or_build_the_folder_does_not_list(self):
        for version, why in (
            (
                "10.3.0",
                "AMD's public folder of ROCm installers lists no build of ROCm 10.3.0",
            ),
            (
                "10.1.0-99",
                "AMD's public folder of ROCm installers lists no rocm-installer-10.1.0-99.run",
            ),
        ):
            with self.subTest(version=version):
                p = build_plan.resolve(make_plan(rocm={"version": version}), full_web())
                self.assertEqual(
                    p.unresolved, (build_plan.Unresolved("rocm.version", why),)
                )
                self.assertEqual(p.args["ROCM_VERSION"], version)
                self.assertNotIn("rocm_package_url", [s.id for s in p.secrets])
        listed = build_plan.resolve(make_plan(rocm={"version": "10.1.0-3"}), full_web())
        self.assertEqual(
            (listed.unresolved, listed.args["ROCM_VERSION"]), ((), "10.1.0-3")
        )
        gone = build_plan.resolve(make_plan(), full_web(**{FOLDER + "/": None}))
        self.assertIn(
            build_plan.Unresolved(
                "rocm.version",
                "AMD's public folder of ROCm installers could not be read (the server answered HTTP 404)",
            ),
            gone.unresolved,
        )

    def test_an_installer_of_your_own_is_never_listed(self):
        web = full_web()
        p = build_plan.resolve(make_plan(rocm=PRIVATE["rocm"]), web)
        self.assertNotIn(FOLDER + "/", web.urls())
        self.assertEqual(p.args["ROCM_VERSION"], "10.1.0")
        self.assertNotIn("rocm.version", [u.key for u in p.unresolved])

    def test_gh_pinned_takes_nothing_from_the_network(self):
        web = full_web()
        p = build_plan.resolve(
            make_plan(github_cli={"version": "2.80.0", "package_sha256": SHA256["gh"]}),
            web,
        )
        self.assertEqual([u for u in web.urls() if "github" in u], [])
        self.assertEqual(p.newest, (("claude_agent_sdk.version", "0.3.0"),))

    def test_gh_checksums_that_list_no_archive_for_this_machine(self):
        """The published checksums, in sha256sum's binary form, list gh for arm64 alone."""
        sums = deploy.gh_download(GH, deploy.gh_checksums(GH))
        web = full_web(**{sums: f"{'2' * 64} *gh_{GH}_linux_arm64.tar.gz\n".encode()})
        p = build_plan.resolve(make_plan(), web)
        self.assertEqual(
            p.unresolved,
            (
                build_plan.Unresolved(
                    "github_cli.package_sha256",
                    f"GitHub's published checksums of gh {GH} list no gh_{GH}_linux_amd64.tar.gz",
                ),
            ),
        )
        arm = build_plan.resolve(make_plan(machine="aarch64"), web)
        self.assertEqual(arm.args["GITHUB_CLI_PACKAGE_SHA256"], "2" * 64)

    def test_the_sdks_wheel_for_this_machine_read_in_parts(self):
        """The newest plain release that is not yanked, the wheel for this machine's architecture, and of it only its
        table of contents and the file that names the Claude Code version, through range requests.
        """
        web = full_web()
        p = build_plan.resolve(make_plan(), web)
        x86 = f"{FILES}/{wheel_name('0.3.0', 'manylinux_2_17_x86_64')}"
        parts = [r for u, r in web.asked if u == x86]
        self.assertTrue(parts and all(parts), parts)

        def size_of(r: str) -> int:
            start, _, end = r.partition("-")
            return int(end) if not start else int(end) - int(start) + 1

        self.assertLess(sum(size_of(r) for r in parts), len(web.pages[x86]) / 2)
        self.assertEqual(p.args["CLAUDE_CODE_VERSION"], "2.1.400")
        arm = build_plan.resolve(make_plan(machine="aarch64"), full_web())
        self.assertEqual(arm.args["CLAUDE_CODE_VERSION"], "2.1.401")

    def test_a_server_that_sends_the_whole_wheel(self):
        x86 = f"{FILES}/{wheel_name('0.3.0', 'manylinux_2_17_x86_64')}"
        p = build_plan.resolve(
            make_plan(), full_web(**{x86: wheel("0.3.0", "2.1.400")})
        )
        self.assertEqual(p.args["CLAUDE_CODE_VERSION"], "2.1.400")

    def test_an_sdk_version_the_file_names(self):
        p = build_plan.resolve(
            make_plan(claude_agent_sdk={"version": "0.2.0"}), full_web()
        )
        self.assertEqual(
            (p.args["CLAUDE_AGENT_SDK_VERSION"], p.args["CLAUDE_CODE_VERSION"]),
            ("0.2.0", "2.1.300"),
        )
        self.assertNotIn("claude_agent_sdk.version", [k for k, _ in p.newest])
        none = build_plan.resolve(
            make_plan(claude_agent_sdk={"version": "9.9.9"}), full_web()
        )
        self.assertEqual(
            none.unresolved,
            (
                build_plan.Unresolved(
                    "claude_agent_sdk.version",
                    "the package index has no wheel of claude-agent-sdk 9.9.9",
                ),
            ),
        )

    def test_the_sdk_from_a_package_index_of_your_own(self):
        index = "https://pypi.example.com/simple/"
        web = full_web(
            **{f"{index}{build_plan.SDK_PROJECT}/": sdk_pages()[PAGE], PAGE: None}
        )
        p = build_plan.resolve(make_plan(python_packages={"package_index": index}), web)
        self.assertEqual(p.unresolved, ())
        self.assertEqual(p.args["CLAUDE_CODE_VERSION"], "2.1.400")
        self.assertNotIn(PAGE, web.urls())

    def test_the_sdks_own_wheel(self):
        """claude_agent_sdk.package_url: that wheel, its METADATA's version, which a set version must equal."""
        url = PRIVATE["claude_agent_sdk"]["package_url"]
        web = Web({url: Ranged(wheel("0.3.0", "2.1.400"))})
        p = build_plan.resolve(make_plan(**PRIVATE), web)
        self.assertEqual(p.unresolved, ())
        self.assertEqual(
            (p.args["CLAUDE_AGENT_SDK_VERSION"], p.args["CLAUDE_CODE_VERSION"]),
            ("0.3.0", "2.1.400"),
        )
        self.assertEqual(p.newest, ())
        self.assertEqual(set(web.urls()), {url})
        sections = dict(
            PRIVATE, claude_agent_sdk=dict(PRIVATE["claude_agent_sdk"], version="0.2.0")
        )
        other = build_plan.resolve(
            make_plan(**sections), Web({url: Ranged(wheel("0.3.0"))})
        )
        self.assertEqual(
            other.unresolved,
            (
                build_plan.Unresolved(
                    "claude_agent_sdk.version",
                    "the wheel at claude_agent_sdk.package_url is claude-agent-sdk 0.3.0, not "
                    "claude_agent_sdk.version 0.2.0",
                ),
            ),
        )
        for body, why in (
            (Ranged(b"no zip file" * 100), "could not be read as a wheel (a zip file)"),
            (
                Ranged(wheel("0.3.0", cli="")),
                f"names no Claude Code version ({build_plan.SDK_CLI_VERSION_FILE})",
            ),
        ):
            with self.subTest(why=why):
                bad = build_plan.resolve(make_plan(**PRIVATE), Web({url: body}))
                self.assertEqual(
                    bad.unresolved,
                    (
                        build_plan.Unresolved(
                            "claude_agent_sdk.version",
                            f"the wheel at claude_agent_sdk.package_url {why}",
                        ),
                    ),
                )

    def test_a_failure_names_no_host(self):
        """Every download fails with the error a private host gives: each Unresolved says why in words that name no
        host and no address."""
        host = "private.example.com"
        failure = deploy.FetchError(
            deploy.download_failure(
                urllib.error.URLError(
                    socket.gaierror(-2, f"Name or service not known: {host}")
                )
            )
        )

        class Down(Web):
            def __call__(self, url, byte_range=""):
                self.asked.append((url, byte_range))
                raise failure

        web = Down({})
        p = build_plan.resolve(
            make_plan(**dict(PRIVATE, rocm={"version": "10.1.0"}, github_cli={})), web
        )
        self.assertEqual(
            [u.key for u in p.unresolved],
            ["rocm.version", "github_cli.version", "claude_agent_sdk.version"],
        )
        for u in p.unresolved:
            with self.subTest(key=u.key):
                self.assertIn("the host's name does not resolve", u.why)
                self.assertNotIn(host, u.why)
                self.assertNotIn("http", u.why)
        self.assertTrue(
            any(host in url for url in web.urls())
        )  # the private addresses were asked

    def test_the_root_certificates_sha256(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = pathlib.Path(d, "a.pem"), pathlib.Path(d, "b.pem")
            a.write_bytes(b"first")
            b.write_bytes(b"second")
            sections = with_ca(PRIVATE, [a, b])
            p = build_plan.resolve(make_plan(**sections), private_web(sections))
            self.assertEqual(
                p.ca_sha256,
                (
                    hashlib.sha256(b"first").hexdigest(),
                    hashlib.sha256(b"second").hexdigest(),
                ),
            )
            b.unlink()
            gone = build_plan.resolve(make_plan(**sections), private_web(sections))
            self.assertEqual(
                gone.unresolved,
                (
                    build_plan.Unresolved(
                        "operating_system.ca_certificates",
                        "its file number 2 cannot be read",
                    ),
                ),
            )


class TestRecord(PlanCase):
    def resolved(self, sections=PRIVATE, ca=(), **over):
        """The plan of SECTIONS, each of OVER laid over them and CA's root certificate files named, resolved."""
        values = with_ca(dict(sections, **over), ca) if ca else dict(sections, **over)
        return build_plan.resolve(make_plan(**values), private_web(values))

    def test_the_record_and_the_labels(self):
        p = build_plan.resolve(make_plan(), full_web())
        rec = build_plan.record(p)
        self.assertEqual(
            rec,
            {
                "kind": "build_record",
                "schema_version": 1,
                "build_settings_fingerprint": build_plan.fingerprint(p),
                "package_version": PACKAGE,
                "build_date": DATE,
                "image": {"registry": "", "name": image.DEFAULT_NAME, "tag": PACKAGE},
                "base": "ubuntu:26.04",
                "rocm": {
                    "version": "10.1.0-12",
                    "package_sha256": "",
                    "clang_commit": build_plan.ROCM_CLANG_COMMIT,
                    "library_targets": ["gfx942"],
                    "hip_check_targets": ["gfx942", "gfx950"],
                },
                "atlassian_cli": {"version": "", "package_sha256": SHA256["atlassian"]},
                "github_cli": {"version": GH, "package_sha256": SHA256["gh"]},
                "claude_agent_sdk": {
                    "version": "0.3.0",
                    "package_sha256": "",
                    "claude_code_version": "2.1.400",
                },
                "ca_certificates": {"count": 0, "sha256": []},
            },
        )
        self.assertRegex(rec["build_settings_fingerprint"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(set(p.labels), LABELS)
        self.assertEqual(
            {k: v for k, v in p.labels.items() if k.startswith("org.opencontainers.")},
            {
                "org.opencontainers.image.created": DATE,
                "org.opencontainers.image.description": build_plan.DESCRIPTION,
                "org.opencontainers.image.licenses": "MIT",
                "org.opencontainers.image.title": image.TITLE,
                "org.opencontainers.image.version": PACKAGE,
            },
        )
        self.assertEqual(
            {k: v for k, v in p.labels.items() if k.startswith("org.rocke.")},
            {
                "org.rocke.package.version": PACKAGE,
                "org.rocke.build.date": DATE,
                "org.rocke.base": "ubuntu:26.04",
                "org.rocke.rocm.version": "10.1.0-12",
                "org.rocke.rocm.clang_commit": build_plan.ROCM_CLANG_COMMIT,
                "org.rocke.rocm.library_targets": "gfx942",
                "org.rocke.atlassian_cli.version": "",
                "org.rocke.github_cli.version": GH,
                "org.rocke.claude_agent_sdk.version": "0.3.0",
                "org.rocke.claude_code.version": "2.1.400",
                "org.rocke.build_settings.fingerprint": rec[
                    "build_settings_fingerprint"
                ],
            },
        )
        # What rkmon image check and rkmon image ls read of an image: the line that describes it.
        self.assertEqual(
            image.describe(p.labels),
            f"package {PACKAGE}, built {DATE}, ubuntu:26.04, ROCm 10.1.0-12",
        )

    def test_the_labels_names(self):
        """label_names: each label a plan gives, in its order; FINGERPRINT_LABEL is the fingerprint's."""
        p = make_plan()
        self.assertEqual(build_plan.label_names(), tuple(p.labels))
        self.assertEqual(
            p.labels[build_plan.FINGERPRINT_LABEL], build_plan.fingerprint(p)
        )

    def test_no_address_index_mirror_or_file_name_in_the_record_or_the_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d, "my-company-root.pem")
            root.write_bytes(
                b"-----BEGIN CERTIFICATE-----\nx\n-----END CERTIFICATE-----\n"
            )
            p = self.resolved(ca=[root])
            self.assertEqual(p.unresolved, ())
            shown = json.dumps(build_plan.record(p)) + json.dumps(p.labels)
            private = [d, root.name, "private.example.com"]
            for keys in PRIVATE.values():
                for key, value in keys.items():
                    if key in (
                        "package_url",
                        "package_index",
                        "package_mirror",
                        "registry",
                    ):
                        private += [
                            value,
                            urllib.parse.urlsplit(value).netloc or value.split("/")[0],
                        ]
            for value in private:
                with self.subTest(value=value):
                    self.assertNotIn(value, shown)
            self.assertEqual(build_plan.record(p)["image"]["registry"], "")
            self.assertEqual(build_plan.record(p)["ca_certificates"]["count"], 1)

    def test_the_fingerprint_leaves_private_addresses_out(self):
        """Other private addresses for the same files: the same fingerprint, as for another package version and
        build date, which are no build settings. Another file, another root certificate or another tag: another.
        """
        moved = {
            section: {
                k: (
                    v.replace("private.example.com", "elsewhere.example.org")
                    if isinstance(v, str)
                    else v
                )
                for k, v in keys.items()
            }
            for section, keys in PRIVATE.items()
        }
        first = self.resolved()
        self.assertEqual(
            build_plan.fingerprint(first), build_plan.fingerprint(self.resolved(moved))
        )
        later = build_plan.resolve(
            build_plan.plan(
                build_doc(
                    **dict(PRIVATE, image=dict(PRIVATE["image"], tag=first.image.tag))
                ),
                "2027.01.01",
                "2027-01-01T00:00:00Z",
                "x86_64",
            ),
            private_web(PRIVATE),
        )
        self.assertEqual(build_plan.fingerprint(first), build_plan.fingerprint(later))
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d, "root.pem")
            root.write_bytes(b"one")
            one = build_plan.fingerprint(self.resolved(ca=[root]))
            root.write_bytes(b"two")
            two = build_plan.fingerprint(self.resolved(ca=[root]))
        self.assertNotEqual(one, two)
        self.assertNotEqual(one, build_plan.fingerprint(first))
        for other in (
            self.resolved(rocm=dict(PRIVATE["rocm"], package_sha256=SHA256["sdk"])),
            self.resolved(image=dict(PRIVATE["image"], tag="t9")),
        ):
            self.assertNotEqual(
                build_plan.fingerprint(first), build_plan.fingerprint(other)
            )

    def test_the_parameters_the_check_lists(self):
        """The image, the architecture, each build argument by name with its value ("(blank)" for none), then each
        download by its build secret's variable: one of yours by its secret and key, never its value; one the build
        takes by default by what a blank key means. Every value starts in one column, one space past the longest
        name (CLAUDE_AGENT_SDK_PACKAGE_SHA256's, today)."""
        p = self.resolved()
        lines = build_plan.parameter_lines(p)
        names = [
            "image",
            "architecture",
            *(a.name for a in build_plan.ARGS),
            *(build_plan.secret_variable(d.id) for d in build_plan.DOWNLOADS),
        ]
        self.assertEqual([line.split()[0] for line in lines], names)
        w = max(map(len, names))
        self.assertEqual(w, len("CLAUDE_AGENT_SDK_PACKAGE_SHA256"))
        self.assertEqual(lines[0], f"  {'image':<{w}} {p.image.ref}")
        self.assertEqual(lines[1], f"  {'architecture':<{w}} x86_64")
        self.assertIn(f"  {'ROCM_VERSION':<{w}} 10.1.0", lines)
        self.assertIn(
            f"  {'ROCM_PACKAGE_URL':<{w}} build secret rocm_package_url (from rocm.package_url; never shown)",
            lines,
        )
        for line in lines:
            with self.subTest(line=line):
                self.assertEqual(line[2 + w], " ")
                self.assertNotEqual(line[3 + w], " ")
        self.assertNotIn("private.example.com", "\n".join(lines))
        public = build_plan.parameter_lines(build_plan.resolve(make_plan(), full_web()))
        for line in (
            f"  {'ATLASSIAN_CLI_VERSION':<{w}} (blank)",
            f"  {'PACKAGE_MIRROR':<{w}} (blank: the base's own package archive)",
            f"  {'ROCM_PACKAGE_URL':<{w}} (blank: AMD's public installer of rocm.version)",
            f"  {'PYTHON_PACKAGE_INDEX':<{w}} (blank: pypi.org)",
        ):
            with self.subTest(line=line):
                self.assertIn(line, public)


class TestCommand(PlanCase):
    """The one docker build of a resolved plan (build_plan.command)."""

    def resolved(self, sections=PRIVATE, ca=()):
        values = with_ca(dict(sections), ca) if ca else dict(sections)
        return build_plan.resolve(make_plan(**values), private_web(values))

    @staticmethod
    def flags(argv) -> list:
        """(flag, its value or None) for each word of a docker build's ARGV between `-t NAME:TAG` and the context."""
        valued = ("--build-arg", "--secret", "--label", "--build-context")
        words, out = list(argv[6:-1]), []
        while words:
            flag = words.pop(0)
            out.append((flag, words.pop(0) if flag in valued else None))
        return out

    def test_only_the_flags_it_lists(self):
        """A --build-arg for each of ARGS in its order, a --secret for each download, a --label for each label, the
        build contexts, and nothing else: no flag can add a label or a build argument.
        """
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d, "root.pem")
            root.write_bytes(b"one")
            p = self.resolved(ca=[root])
        c = build_plan.command(p, "docker", "/tmp/ca", "/tmp/record")
        self.assertEqual(
            c.argv[:6],
            ("docker", "build", "-f", "docker/Dockerfile", "-t", p.image.ref),
        )
        self.assertEqual((c.argv[-1], c.ref), (".", p.image.ref))
        flags = self.flags(c.argv)
        self.assertEqual(
            {f for f, _v in flags},
            {
                "--provenance=false",
                "--build-arg",
                "--secret",
                "--label",
                "--build-context",
            },
        )
        given = [v.split("=", 1) for f, v in flags if f == "--build-arg"]
        self.assertEqual([n for n, _v in given], [a.name for a in build_plan.ARGS])
        self.assertEqual(dict(given), p.args)
        self.assertEqual(
            dict(v.split("=", 1) for f, v in flags if f == "--label"), p.labels
        )
        self.assertEqual(
            [v for f, v in flags if f == "--secret"],
            [f"id={s.id},env={s.id.upper()}" for s in p.secrets],
        )
        self.assertEqual(
            [v for f, v in flags if f == "--build-context"],
            ["ca-certificates=/tmp/ca", "build-record=/tmp/record"],
        )
        # No root certificate named: the record's context alone.
        alone = build_plan.command(self.resolved(), "docker", "/tmp/ca", "/tmp/record")
        self.assertEqual(
            [v for f, v in self.flags(alone.argv) if f == "--build-context"],
            ["build-record=/tmp/record"],
        )

    def test_no_private_value_or_folder_is_shown(self):
        """Each download of yours is in the environment alone, shown as its settings key; each build context's folder
        is shown by what it holds."""
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d, "my-company-root.pem")
            root.write_bytes(b"one")
            p = self.resolved(ca=[root])
            c = build_plan.command(p, "docker", f"{d}/ca-x1", f"{d}/record-x1")
        self.assertEqual(p.unresolved, ())
        argv = " ".join(c.argv)
        for s in p.secrets:
            with self.subTest(secret=s.id):
                self.assertFalse(s.public)
                self.assertEqual(c.env[s.id.upper()], s.value)
                self.assertNotIn(s.value, argv)
                self.assertNotIn(s.value, c.shown)
                self.assertIn(f"{s.id.upper()}=<{s.key}>", c.shown_env)
        for value in (d, root.name, "private.example.com"):
            with self.subTest(value=value):
                self.assertNotIn(value, c.shown)
        self.assertIn(
            "ca-certificates=<operating_system.ca_certificates>", c.shown_argv
        )
        self.assertIn("build-record=<the build record>", c.shown_argv)
        self.assertEqual(
            c.shown,
            " ".join([*c.shown_env, *(shlex.quote(w) for w in c.shown_argv)]),
        )

    def test_a_public_default_is_shown_as_its_address(self):
        p = build_plan.resolve(make_plan(), full_web())
        c = build_plan.command(p)
        self.assertIn(
            f"ROCM_PACKAGE_URL={FOLDER}/rocm-installer-10.1.0-12.run", c.shown_env
        )
        self.assertIn(f"PYTHON_PACKAGE_INDEX={build_plan.PUBLIC_PYPI}", c.shown_env)
        self.assertIn(
            "ATLASSIAN_CLI_PACKAGE_URL=<atlassian_cli.package_url>", c.shown_env
        )

    def test_a_plan_with_values_it_cannot_have_builds_nothing(self):
        with self.assertRaises(ValueError) as cm:
            build_plan.command(make_plan())
        self.assertIn("rocm.version", str(cm.exception))

    def test_podman_and_no_cache(self):
        """docker's build is told to record nothing of the package folder; podman's makes no such record and takes
        neither its flag nor its variables. --no-cache only when asked."""
        p = self.resolved()
        docker = build_plan.command(p, "docker", no_cache=True)
        self.assertEqual(docker.argv[6:8], ("--provenance=false", "--no-cache"))
        self.assertEqual(
            {k: docker.env[k] for k in build_plan.DOCKER_BUILD_ENV},
            build_plan.DOCKER_BUILD_ENV,
        )
        podman = build_plan.command(p, "/usr/bin/podman")
        self.assertNotIn("--provenance=false", podman.argv)
        self.assertNotIn("--no-cache", podman.argv)
        self.assertFalse(set(build_plan.DOCKER_BUILD_ENV) & set(podman.env))


class Response:
    """What urllib's opener gives for an answer: its status, address, headers and body."""

    def __init__(self, body: bytes = b"", status: int = 200, headers=None):
        self.body, self.status, self.headers = body, status, dict(headers or {})

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def geturl(self) -> str:
        return "https://files.example.com/answered"

    def read(self) -> bytes:
        return self.body


class TestFetch(unittest.TestCase):
    """deploy.fetcher, the downloads' fetch on the host: resolve's, and gh's install."""

    def fetch(self, *answers):
        """deploy.fetcher on a fake connection that gives ANSWERS in turn (a Response, or an exception it raises), each
        request (its address, its Range header and its timeout) and each pause kept."""
        self.requests, self.pauses, left = [], [], list(answers)

        def open_url(request, timeout):
            self.requests.append(
                (request.full_url, request.get_header("Range"), timeout)
            )
            answer = left.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return answer

        return deploy.fetcher(
            timeout_seconds=30,
            pause_seconds=2,
            sleep=self.pauses.append,
            open_url=open_url,
        )

    def test_a_download_that_fails_twice_then_succeeds(self):
        fetch = self.fetch(
            urllib.error.URLError(TimeoutError("timed out")),
            urllib.error.HTTPError(
                "https://private.example.com/f", 503, "Unavailable", {}, None
            ),
            Response(b"the file", headers={"Content-Type": "text/plain"}),
        )
        answer = fetch("https://private.example.com/f")
        self.assertEqual(
            answer,
            deploy.Answer(
                200,
                "https://files.example.com/answered",
                {"content-type": "text/plain"},
                b"the file",
            ),
        )
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(self.pauses, [2, 2])
        self.assertEqual({t for _u, _r, t in self.requests}, {30})

    def test_three_passing_failures_end_it(self):
        fetch = self.fetch(
            *[
                urllib.error.HTTPError(
                    "https://private.example.com/f", 503, "Unavailable", {}, None
                )
            ]
            * 3
        )
        with self.assertRaises(deploy.FetchError) as cm:
            fetch("https://private.example.com/f")
        self.assertEqual(str(cm.exception), "the server answered HTTP 503")
        self.assertEqual((len(self.requests), self.pauses), (3, [2, 2]))

    def test_a_failure_that_stands_is_not_asked_again(self):
        host = "private.example.com"
        for exc, why in (
            (
                urllib.error.HTTPError(f"https://{host}/f", 404, "Not Found", {}, None),
                "the server answered HTTP 404",
            ),
            (
                urllib.error.URLError(socket.gaierror(-2, host)),
                "the host's name does not resolve",
            ),
            (
                urllib.error.URLError(ConnectionRefusedError(111, host)),
                "the host refused the connection",
            ),
            (
                urllib.error.URLError(ssl.SSLCertVerificationError(1, host)),
                "the server's certificate is not one this machine trusts (a network that inspects TLS: name its "
                "root certificates in operating_system.ca_certificates)",
            ),
            (
                ValueError(f"unknown url type: {host}"),
                "the address is no URL it can read",
            ),
        ):
            with self.subTest(why=why):
                fetch = self.fetch(exc)
                with self.assertRaises(deploy.FetchError) as cm:
                    fetch(f"https://{host}/f")
                self.assertEqual(str(cm.exception), why)
                self.assertEqual((len(self.requests), self.pauses), (1, []))

    def test_a_path_s_dot_segments_are_resolved(self):
        """The address asked for has its path's dot segments resolved, as curl and pip ask for it: an index given as
        /x/../simple is asked for as /simple."""
        for given, asked in (
            (
                "https://pypi.example.com/x/../simple/claude-agent-sdk/",
                "https://pypi.example.com/simple/claude-agent-sdk/",
            ),
            (
                "https://files.example.com/a/./b/../c?q=/../",
                "https://files.example.com/a/c?q=/../",
            ),
            ("https://files.example.com/a/b/..", "https://files.example.com/a/"),
            ("https://files.example.com/../a", "https://files.example.com/a"),
            ("https://files.example.com/a/b", "https://files.example.com/a/b"),
        ):
            with self.subTest(given=given):
                fetch = self.fetch(Response(b"page"))
                fetch(given)
                self.assertEqual(self.requests[0][0], asked)

    def test_a_part_of_a_file(self):
        fetch = self.fetch(Response(b"tail", 206, {"Content-Range": "bytes 96-99/100"}))
        answer = fetch("https://files.example.com/w.whl", "-4")
        self.assertEqual(self.requests[0][1], "bytes=-4")
        self.assertEqual(
            (answer.status, answer.headers["content-range"]), (206, "bytes 96-99/100")
        )

    def test_the_root_certificates_the_build_names_are_trusted(self):
        """A fetcher trusts this machine's roots and each file the build names; a file it cannot read is left out (its
        own check says so) and the rest are still trusted."""
        loaded = []

        def load(cafile):
            if cafile.endswith("gone.pem"):
                raise FileNotFoundError(2, "No such file")
            loaded.append(cafile)

        context = mock.Mock(load_verify_locations=load)
        with mock.patch.object(deploy, "tls_context", return_value=context):
            deploy.fetcher(
                ca_files=[
                    pathlib.Path("/certs/a.pem"),
                    "/certs/gone.pem",
                    "/certs/b.pem",
                ]
            )
        self.assertEqual(loaded, ["/certs/a.pem", "/certs/b.pem"])


if __name__ == "__main__":
    unittest.main()

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""From build settings to a docker build: the package's build defaults, which no settings file
sets, one home each; the plan a build settings file makes; the one network step that resolves it; and the record and
the labels the image carries.

The package's build defaults:

- ROCM_RELEASE, ROCM_CLANG_COMMIT: the ROCm release the review gates are tuned on, and the llvm-project commit its
  clang names, the gate compiler's (the live monitor's). `clang_commit(version)`: the commit the clang of ROCm VERSION
  (a release, 10.1.0, or one build of it, 10.1.0-5) must name: ROCM_CLANG_COMMIT for ROCM_RELEASE; "" for another
  release, whose clang no pin names. The build's clang check and rk doctor's clang parity compare with it.
- ROCM_LIBRARY_TARGETS: the GPU targets ROCm's libraries are installed for (the runfile installer's gfx=).
- HIP_CHECK_TARGETS: the GPU targets the review's GPU tests run on, which the build's hipcc check and rk doctor's
  compile <hip/hip_runtime.h> for. Two values, not one: the compiler covers every target either way.
- PUBLIC_ROCM_INSTALLERS: AMD's public folder of ROCm runfile installers, where AMD's public ROCm install documentation
  (https://rocm.docs.amd.com: the runfile installer's page) points; `public_installers()` reads it, moved by
  ROCM_PUBLIC_INSTALLERS in the environment (the tests). PUBLIC_PYPI: the public Python package index. GitHub's gh
  releases are deploy.GH_RELEASES, which the host's own gh install reads too.
- TOOL_HOSTS: the public hosts the build's tools read with no build secret of their own: PyPI's file host, where
  PUBLIC_PYPI's pages send each download (PYPI_FILES_HOST), and the Ubuntu archives, key server and PPA docker/pkgs.sh
  reads (UBUNTU_HOSTS). A build tool's lines show them as they are, each location the build was given by its build
  secret's name, and any other host masked (docker/masked.sh, which keeps a copy of TOOL_HOSTS).
- PUBLIC_HOSTS: the hosts of the build's public downloads, those three's and TOOL_HOSTS, which the canary test leaves
  out of its search.

The installers' names, as that folder lists them (rocm-installer-<release>-<build>.run): INSTALLER, `release_of`
(10.1.0 of 10.1.0-5), `version_key` (`sort -V` for them) and `installer_build(listing, version)`, the build a VERSION
names in a listing: a release, its newest build; a build, itself; "" when the listing has none.

The plan:

- `ARGS`: each build argument the Dockerfile declares, in its order, as `Arg(name, key, what)`: NAME, the argument;
  KEY, the build settings key it comes from, "" for a package build default or a value resolve reads, which WHAT
  describes. The Dockerfile gives none of them a default, so a build that is not given one fails.
- `DOWNLOADS`: each download location, a build secret, never a build argument (it may name a private host), as
  `Download(id, key, blank)`: ID, the secret's id in the Dockerfile (`RUN --mount=type=secret,id=ID`); KEY, its build
  settings key; BLANK, what a blank KEY gives, in words. The build is given the package's public default for a blank
  rocm.package_url, github_cli.package_url and python_packages.package_index, and nothing for a blank package mirror
  or SDK wheel: the step does without it.
- `plan(doc, package, date, machine) -> BuildPlan`: pure. DOC, a build settings document in the layout this package
  reads (a ValueError for any other document); PACKAGE, this package's version (image.package_version); DATE, the
  build's time (UTC, as 2026-11-19T10:00:00Z); MACHINE, this machine's architecture as platform.machine() names it,
  the image's (docker builds for the machine it runs on). The image's tag is image.tag, else PACKAGE, else
  "local". Each value is the file's effective one (settings_file.effective); one that still has a <...> placeholder
  counts as blank, as validate refuses it.
- `BuildPlan(args, secrets, ca_files, labels, unresolved, image, package, date, machine, newest, ca_sha256)`:
  - `args`: NAME -> text for each of ARGS, in its order; each SHA-256 in lower case, each tool version without a
    leading v.
  - `secrets`: a `Secret(id, key, value, public)` for each download the build is given, in DOWNLOADS' order: VALUE,
    the location, never printed; PUBLIC, whether it is the package's public default for a blank KEY. The package
    mirror and the SDK's wheel are given only when their keys are set.
  - `ca_files`: the root certificate files operating_system.ca_certificates names, in its order, `~` expanded.
  - `labels`: the image's labels, each name to its text: org.opencontainers.image.title (image.TITLE),
    description, licenses, version (PACKAGE) and created (DATE); org.rocke.package.version (PACKAGE_LABEL),
    org.rocke.build.date (DATE_LABEL), org.rocke.base (BASE_LABEL) and org.rocke.rocm.version (ROCM_LABEL),
    which describe an image to a deployment (image.describe); org.rocke.rocm.clang_commit,
    org.rocke.rocm.library_targets, org.rocke.atlassian_cli.version, org.rocke.github_cli.version,
    org.rocke.claude_agent_sdk.version, org.rocke.claude_code.version and
    org.rocke.build_settings.fingerprint (FINGERPRINT_LABEL: `fingerprint`). No other: `rkmon build` passes
    docker only the flags it lists. `label_names()`: their names, in this order, each of which the image test
    looks for on the image.
  - `unresolved`: an `Unresolved(key, why)` for each value the build cannot have yet, named by its build settings key,
    WHY in words that name no host and no address. `plan` lists one, its WHY PENDING, for each value only the network
    gives: rocm.version while rocm.package_url is blank (the build of AMD's public installer, which its folder must
    list); github_cli.version left blank while github_cli.package_url is (the newest release), and
    github_cli.package_sha256 likewise (GitHub's published checksum); and always claude_agent_sdk.version (the SDK's
    release, the newest when the key is blank, and the Claude Code version it carries). `resolve` replaces each with
    its value, or with why it cannot be had. A plan with one builds nothing.
  - `image`: `Image(registry, name, tag)`, its `ref` NAME:TAG; the registry, a private value, for a push only.
  - `package`, `date`, `machine`: plan's PACKAGE, DATE and MACHINE.
  - `newest`: (key, version) for each blank version that resolve took as the newest release, which `rkmon check`
    warns on: two builds of one file can then differ.
  - `ca_sha256`: the SHA-256 of each of ca_files, in its order, which resolve reads.
- `resolve(plan, fetch) -> BuildPlan`: the one network step, which runs before a dry run prints, so that the dry run
  shows the build's own command, but for the label of the build's time (DATE); FETCH is deploy.fetcher's
  `fetch(url, byte_range="")`, which raises deploy.FetchError. ROCm: the folder's listing gives the release's newest
  build (a build given exactly must be listed), and the public installer's address. gh: the newest release
  (deploy.latest_gh), the archive for this machine (deploy.gh_file, deploy.gh_download) and its SHA-256 from GitHub's
  published checksums (deploy.published_sha256). The SDK: from the package index's page of claude-agent-sdk (a PEP 503 simple index; a
  yanked file never counts), the newest plain release (digits and dots) when its version is blank, and the wheel of
  that release for this machine, else any of its wheels; with claude_agent_sdk.package_url, that wheel, whose
  METADATA gives the version (which a set claude_agent_sdk.version must equal). Of the wheel, only its table of
  contents and claude_agent_sdk/_cli_version.py are read, through range requests (a server that sends the whole
  file instead is read whole): its __cli_version__ is the Claude Code version the SDK carries. The root
  certificates' SHA-256s are read from their files. Each value that cannot be had is an Unresolved, and a FetchError
  says why in words that name no host.
- `record(plan) -> dict`: the build record, RECORD_PATH in the image: RECORD_KIND, RECORD_SCHEMA_VERSION, the
  fingerprint, PACKAGE, DATE, the image block (its registry always blank: the registry is private), the base, then
  each tool's version and SHA-256 as resolved, ROCm's with its clang commit and both target lists, the SDK's with the
  Claude Code version, and the root certificates' count and SHA-256s. No URL, index, mirror, registry or certificate
  file name.
- `parameter_lines(plan) -> list`: the build's parameters as `rkmon check` lists them after its rows, one line
  each: the image's NAME:TAG and this machine's architecture, then each of ARGS with its value ((blank) for
  none), then each of DOWNLOADS by its variable (secret_variable): the build secret a key of yours makes, never
  its value, else what the build takes instead, in words (a public default is a location too). Every value starts
  in one column, one space past the longest name.
- `fingerprint(plan) -> str`: the build settings' fingerprint, "sha256:…" through settings_file.fingerprint, over
  the build schema's layout and the record's public build values: the image's name and tag, the base, each version
  and SHA-256 as resolved, the package's ROCm defaults and the root certificates' SHA-256s. It leaves out the private
  addresses, which change where a file comes from, not what it is, and PACKAGE and DATE, which are no build settings.

The docker build:

- `command(plan, engine="docker", ca_dir=CA_SHOWN, record_dir=RECORD_SHOWN, no_cache=False) -> Command`: pure. The
  one `docker build` of a resolved PLAN (a ValueError while it has an Unresolved), run in the package folder: the
  Dockerfile (DOCKERFILE) and the image's NAME:TAG; DOCKER_BUILD_FLAGS for docker, and --no-cache when NO_CACHE asks;
  a --build-arg for each of ARGS, in its order; a --secret for each of the plan's secrets, read from its variable
  (secret_variable); a --label for each of the plan's labels; the build contexts, CA_DIR's root certificates (when
  the plan names some) and RECORD_DIR's build record (RECORD_FILE); and the package folder, last. No other flag, so
  nothing can add a label or a build argument. ENGINE is the container engine's command: podman's build is given no
  docker flag and no docker variable.
- `Command(argv, env, shown_argv, shown_env, ref)`: ARGV, the words to run; ENV, the variables to add to the
  environment: DOCKER_BUILD_ENV's for docker, and each secret's value under its variable, never printed; SHOWN_ARGV
  and SHOWN_ENV, the same as `rkmon build --dry-run` prints them: each build context's folder as what it holds
  (CA_SHOWN, RECORD_SHOWN), each download location of yours as its settings key (<rocm.package_url>), a public
  default as its address; `shown`, the two as one line; REF, the image's NAME:TAG.

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import hashlib
import html.parser
import io
import os
import pathlib
import re
import shlex
import urllib.parse
import zipfile
from typing import NamedTuple

from . import deploy, settings_file

ROCM_RELEASE = "10.1.0"
ROCM_CLANG_COMMIT = "d6f6cb691863e9258034ab48c1dfce21689d5606"
ROCM_LIBRARY_TARGETS = ("gfx942",)
HIP_CHECK_TARGETS = ("gfx942", "gfx950")
PUBLIC_ROCM_INSTALLERS = "https://stable.repo.amd.com/rocm/core/linux-installers"
PUBLIC_PYPI = "https://pypi.org/simple"
PYPI_FILES_HOST = "files.pythonhosted.org"
UBUNTU_HOSTS = (
    "archive.ubuntu.com",
    "security.ubuntu.com",
    "ports.ubuntu.com",
    "keyserver.ubuntu.com",
    "ppa.launchpadcontent.net",
)
TOOL_HOSTS = (PYPI_FILES_HOST, *UBUNTU_HOSTS)
PUBLIC_HOSTS = (
    *(
        urllib.parse.urlsplit(u).hostname
        for u in (PUBLIC_ROCM_INSTALLERS, PUBLIC_PYPI, deploy.GH_RELEASES)
    ),
    *TOOL_HOSTS,
)
INSTALLER = re.compile(r"rocm-installer-[0-9][0-9.]*-[0-9]+\.run")

RECORD_PATH = "/etc/rkmon/build.json"
RECORD_KIND = "build_record"
RECORD_SCHEMA_VERSION = 1
DESCRIPTION = (
    "The rocKE monitor: the toolkit, its tools and the Claude session that drives it"
)
LICENSES = "MIT"
PACKAGE_LABEL = "org.rocke.package.version"
DATE_LABEL = "org.rocke.build.date"
BASE_LABEL = "org.rocke.base"
ROCM_LABEL = "org.rocke.rocm.version"
FINGERPRINT_LABEL = "org.rocke.build_settings.fingerprint"
PENDING = "read over the network when the build is checked"

# The Claude Agent SDK on a Python package index, and the file of its wheel that names the Claude Code version it
# carries.
SDK_PROJECT = "claude-agent-sdk"
SDK_CLI_VERSION_FILE = "claude_agent_sdk/_cli_version.py"
CLI_VERSION = re.compile(r"""__cli_version__\s*=\s*["']([^"']+)["']""")
WHEEL = re.compile(
    r"(?P<name>[A-Za-z0-9_.]+)-(?P<version>[^-]+)(?:-[0-9][^-]*)?-[^-]+-[^-]+-(?P<platform>[^-]+)\.whl"
)
PLAIN_RELEASE = re.compile(r"[0-9]+(?:\.[0-9]+)*")
WHEEL_ARCH = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
}
METADATA = re.compile(r"[^/]+\.dist-info/METADATA")
METADATA_VERSION = re.compile(r"^Version:[ \t]*(\S+)", re.M)
CONTENT_RANGE = re.compile(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)")
RANGE_BLOCK_BYTES = 64 * 1024


class Arg(NamedTuple):
    """One build argument: see the module's docstring."""

    name: str
    key: str
    what: str = ""


class Download(NamedTuple):
    """One download location, a build secret: see the module's docstring."""

    id: str
    key: str
    blank: str


ARGS = (
    Arg("BASE_IMAGE", "operating_system.base_image"),
    Arg("ROCM_VERSION", "rocm.version"),
    Arg("ROCM_PACKAGE_SHA256", "rocm.package_sha256"),
    Arg(
        "ROCM_CLANG_COMMIT",
        "",
        "the gate compiler's llvm-project commit, which the installed clang must name (blank for another ROCm "
        "release: any clang)",
    ),
    Arg(
        "ROCM_LIBRARY_TARGETS", "", "the GPU targets ROCm's libraries are installed for"
    ),
    Arg(
        "HIP_CHECK_TARGETS", "", "the GPU targets the build's hipcc check compiles for"
    ),
    Arg("ATLASSIAN_CLI_VERSION", "atlassian_cli.version"),
    Arg("ATLASSIAN_CLI_PACKAGE_SHA256", "atlassian_cli.package_sha256"),
    Arg("GITHUB_CLI_VERSION", "github_cli.version"),
    Arg("GITHUB_CLI_PACKAGE_SHA256", "github_cli.package_sha256"),
    Arg("CLAUDE_AGENT_SDK_VERSION", "claude_agent_sdk.version"),
    Arg("CLAUDE_AGENT_SDK_PACKAGE_SHA256", "claude_agent_sdk.package_sha256"),
    Arg(
        "CLAUDE_CODE_VERSION",
        "",
        "the Claude Code version the SDK's wheel carries, which the installed claude must report",
    ),
)
DOWNLOADS = (
    Download(
        "package_mirror",
        "operating_system.package_mirror",
        "the base's own package archive",
    ),
    Download(
        "rocm_package_url",
        "rocm.package_url",
        "AMD's public installer of rocm.version",
    ),
    Download(
        "atlassian_cli_package_url",
        "atlassian_cli.package_url",
        "nothing: the key is required",
    ),
    Download(
        "github_cli_package_url",
        "github_cli.package_url",
        "GitHub's release archive of gh for this machine",
    ),
    Download(
        "claude_agent_sdk_package_url",
        "claude_agent_sdk.package_url",
        "the SDK from the package index",
    ),
    Download("python_package_index", "python_packages.package_index", "pypi.org"),
)


class Secret(NamedTuple):
    """One build secret the build is given: see the module's docstring."""

    id: str
    key: str
    value: str
    public: bool


class Unresolved(NamedTuple):
    """A value the build cannot have yet: see the module's docstring."""

    key: str
    why: str


class Image(NamedTuple):
    """The image a build makes: see the module's docstring."""

    registry: str
    name: str
    tag: str

    @property
    def ref(self) -> str:
        return f"{self.name}:{self.tag}"


class BuildPlan(NamedTuple):
    """What a build settings file makes: see the module's docstring."""

    args: dict
    secrets: tuple
    ca_files: tuple
    labels: dict
    unresolved: tuple
    image: Image
    package: str
    date: str
    machine: str
    newest: tuple = ()
    ca_sha256: tuple = ()


def public_installers() -> str:
    """AMD's public folder of ROCm runfile installers (ROCM_PUBLIC_INSTALLERS moves it: the tests)."""
    return (os.environ.get("ROCM_PUBLIC_INSTALLERS") or PUBLIC_ROCM_INSTALLERS).rstrip(
        "/"
    )


def release_of(version: str) -> str:
    """10.1.0 for 10.1.0 and 10.1.0-5."""
    return version.rsplit("-", 1)[0]


def clang_commit(version: str) -> str:
    """The llvm-project commit the clang of ROCm VERSION must name: see the module's docstring."""
    return ROCM_CLANG_COMMIT if release_of(version) == ROCM_RELEASE else ""


def version_key(v: str) -> list:
    """`sort -V` for installer versions such as 10.1.0-5: digit runs by value, the rest by text."""
    return [
        (0, int(part), "") if part.isdigit() else (1, 0, part)
        for part in re.split(r"(\d+)", v)
        if part
    ]


def installer_build(listing: str, version: str) -> str:
    """The installer build a VERSION names in a folder's listing: see the module's docstring."""
    if "-" in version:
        return version
    found = {
        m[len("rocm-installer-") : -len(".run")] for m in INSTALLER.findall(listing)
    }
    found = {v for v in found if release_of(v) == version}
    return sorted(found, key=version_key)[-1] if found else ""


# --- the plan --------------------------------------------------------------------------------------------------------


def _at(values: dict, dotted: str):
    node = values
    for part in dotted.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


def _given(value) -> str:
    """A setting's VALUE as text; "" for a blank one, a list or a mapping, or one that still holds a <...>
    placeholder."""
    if value is None or isinstance(value, (dict, list, bool)):
        return ""
    text = str(value).strip()
    return "" if settings_file.PLACEHOLDER.search(text) else text


def _version(v: str) -> str:
    """A tool version as the build compares it: without a leading v (v2.80.0 is 2.80.0)."""
    return v[1:] if v.startswith("v") else v


def _gh_public(version: str, arch: str) -> str:
    """GitHub's release archive of gh VERSION for Linux on ARCH."""
    return deploy.gh_download(version, deploy.gh_file(version, arch))


def plan(doc, package: str, date: str, machine: str) -> BuildPlan:
    """What the build settings document DOC makes: see the module's docstring."""
    from . import file_types

    build = file_types.BUILD
    if doc.file_type.kind != build.kind or doc.version != build.version:
        raise ValueError(
            f"the plan reads {build.noun} in layout {build.version}, not {doc.file_type.noun} in layout "
            f"{doc.version}"
        )
    values = settings_file.effective(doc)

    def given(key: str) -> str:
        return _given(_at(values, key))

    rocm = given("rocm.version")
    found = {
        "BASE_IMAGE": given("operating_system.base_image"),
        "ROCM_VERSION": rocm,
        "ROCM_PACKAGE_SHA256": given("rocm.package_sha256").lower(),
        "ROCM_CLANG_COMMIT": clang_commit(rocm),
        "ROCM_LIBRARY_TARGETS": ",".join(ROCM_LIBRARY_TARGETS),
        "HIP_CHECK_TARGETS": ",".join(HIP_CHECK_TARGETS),
        "ATLASSIAN_CLI_VERSION": given("atlassian_cli.version"),
        "ATLASSIAN_CLI_PACKAGE_SHA256": given("atlassian_cli.package_sha256").lower(),
        "GITHUB_CLI_VERSION": _version(given("github_cli.version")),
        "GITHUB_CLI_PACKAGE_SHA256": given("github_cli.package_sha256").lower(),
        "CLAUDE_AGENT_SDK_VERSION": _version(given("claude_agent_sdk.version")),
        "CLAUDE_AGENT_SDK_PACKAGE_SHA256": given(
            "claude_agent_sdk.package_sha256"
        ).lower(),
        "CLAUDE_CODE_VERSION": "",
    }
    args = {a.name: found[a.name] for a in ARGS}
    secrets = {
        d.id: Secret(d.id, d.key, given(d.key), False)
        for d in DOWNLOADS
        if given(d.key)
    }
    secrets.setdefault(
        "python_package_index",
        Secret(
            "python_package_index", "python_packages.package_index", PUBLIC_PYPI, True
        ),
    )
    unresolved = []
    if "rocm_package_url" not in secrets:
        unresolved.append(Unresolved("rocm.version", PENDING))
    gh = secrets.get("github_cli_package_url")
    arch = deploy.GH_ARCH.get(machine.lower(), "")
    if gh:
        if not args["GITHUB_CLI_VERSION"]:
            m = deploy.gh_archive(gh.value)
            args["GITHUB_CLI_VERSION"] = m.group(2) if m else ""
    elif not arch:
        unresolved.append(
            Unresolved(
                "github_cli.package_url",
                f"GitHub publishes no gh for this machine's architecture ({machine}): set github_cli.package_url "
                "and github_cli.package_sha256",
            )
        )
    else:
        version = args["GITHUB_CLI_VERSION"]
        if version:
            secrets["github_cli_package_url"] = Secret(
                "github_cli_package_url",
                "github_cli.package_url",
                _gh_public(version, arch),
                True,
            )
        else:
            unresolved.append(Unresolved("github_cli.version", PENDING))
        if not args["GITHUB_CLI_PACKAGE_SHA256"]:
            unresolved.append(Unresolved("github_cli.package_sha256", PENDING))
    unresolved.append(Unresolved("claude_agent_sdk.version", PENDING))
    listed = _at(values, "operating_system.ca_certificates")
    ca_files = tuple(
        pathlib.Path(os.path.expanduser(_given(x)))
        for x in (listed if isinstance(listed, list) else ())
        if _given(x)
    )
    image = Image(
        given("image.registry"),
        given("image.name"),
        given("image.tag") or package or "local",
    )
    out = BuildPlan(
        args,
        tuple(secrets[d.id] for d in DOWNLOADS if d.id in secrets),
        ca_files,
        {},
        tuple(unresolved),
        image,
        package,
        date,
        machine,
    )
    return out._replace(labels=_labels(out))


# --- the one network step --------------------------------------------------------------------------------------------


def resolve(p: BuildPlan, fetch) -> BuildPlan:
    """The plan P with what only the network gives: see the module's docstring."""
    args = dict(p.args)
    secrets = {s.id: s for s in p.secrets}
    pending = {u.key for u in p.unresolved if u.why == PENDING}
    left = [u for u in p.unresolved if u.why != PENDING]
    newest = list(p.newest)
    if "rocm.version" in pending:
        left += _rocm(args, secrets, fetch)
    if pending & {"github_cli.version", "github_cli.package_sha256"}:
        left += _gh(args, secrets, fetch, p.machine, newest)
    if "claude_agent_sdk.version" in pending:
        left += _sdk(args, secrets, fetch, p.machine, newest)
    ca_sha256, unread = _ca_sha256(p.ca_files)
    out = p._replace(
        args=args,
        secrets=tuple(secrets[d.id] for d in DOWNLOADS if d.id in secrets),
        unresolved=tuple(left + unread),
        newest=tuple(newest),
        ca_sha256=ca_sha256,
    )
    return out._replace(labels=_labels(out))


def _rocm(args: dict, secrets: dict, fetch) -> list:
    """rocm.version's build in AMD's public folder of ROCm installers, and its installer's address."""
    version, folder = args["ROCM_VERSION"], public_installers()
    where = "AMD's public folder of ROCm installers"
    try:
        listing = fetch(folder + "/").body.decode("utf-8", "replace")
    except deploy.FetchError as exc:
        return [Unresolved("rocm.version", f"{where} could not be read ({exc})")]
    listed = {
        m[len("rocm-installer-") : -len(".run")] for m in INSTALLER.findall(listing)
    }
    build = installer_build(listing, version)
    if not build or build not in listed:
        name = (
            f"rocm-installer-{version}.run"
            if "-" in version
            else f"build of ROCm {version}"
        )
        return [Unresolved("rocm.version", f"{where} lists no {name}")]
    args["ROCM_VERSION"] = build
    secrets["rocm_package_url"] = Secret(
        "rocm_package_url",
        "rocm.package_url",
        f"{folder}/rocm-installer-{build}.run",
        True,
    )
    return []


def _gh(args: dict, secrets: dict, fetch, machine: str, newest: list) -> list:
    """gh's newest release, when github_cli.version is blank, and its archive's published SHA-256, when
    github_cli.package_sha256 is."""
    arch = deploy.GH_ARCH[machine.lower()]
    version = args["GITHUB_CLI_VERSION"]
    if not version:
        try:
            version = deploy.latest_gh(fetch=fetch)
        except deploy.DeployError as exc:
            return [Unresolved("github_cli.version", str(exc))]
        args["GITHUB_CLI_VERSION"] = version
        newest.append(("github_cli.version", version))
        secrets["github_cli_package_url"] = Secret(
            "github_cli_package_url",
            "github_cli.package_url",
            _gh_public(version, arch),
            True,
        )
    if args["GITHUB_CLI_PACKAGE_SHA256"]:
        return []
    file, where = (
        deploy.gh_file(version, arch),
        f"GitHub's published checksums of gh {version}",
    )
    try:
        sums = fetch(deploy.gh_download(version, deploy.gh_checksums(version))).body
    except deploy.FetchError as exc:
        return [
            Unresolved(
                "github_cli.package_sha256", f"{where} could not be read ({exc})"
            )
        ]
    sha256 = deploy.published_sha256(sums.decode("utf-8", "replace"), file)
    if not sha256:
        return [Unresolved("github_cli.package_sha256", f"{where} list no {file}")]
    args["GITHUB_CLI_PACKAGE_SHA256"] = sha256
    return []


class _Links(html.parser.HTMLParser):
    """The links of a package index's page: (href, yanked, text) for each <a>."""

    def __init__(self):
        super().__init__()
        self.links, self._open = [], None

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            named = dict(attrs)
            self._open = [named.get("href") or "", "data-yanked" in named, ""]

    def handle_data(self, data):
        if self._open is not None:
            self._open[2] += data

    def handle_endtag(self, tag):
        if tag == "a" and self._open is not None:
            self.links.append(tuple(self._open))
            self._open = None


def _wheels(page) -> list:
    """(version, platform, address) of each wheel of the SDK that a package index's PAGE (a deploy.Answer) links, a
    yanked one left out."""
    parser = _Links()
    parser.feed(page.body.decode("utf-8", "replace"))
    out = []
    for href, yanked, text in parser.links:
        if yanked:
            continue
        name = text.strip() or urllib.parse.unquote(
            urllib.parse.urlsplit(href).path.rpartition("/")[2]
        )
        m = WHEEL.fullmatch(name)
        if not m or re.sub(r"[-_.]+", "_", m["name"]).lower() != "claude_agent_sdk":
            continue
        address = urllib.parse.urldefrag(urllib.parse.urljoin(page.url, href))[0]
        out.append((m["version"], m["platform"], address))
    return out


def _release_key(version: str) -> tuple:
    return tuple(int(x) for x in version.split("."))


def _this_machine(wheels: list, machine: str) -> str:
    """The address of the wheel of WHEELS (one release's) for this machine: a Linux wheel for its architecture, a
    manylinux one first; else the first."""
    arch = WHEEL_ARCH.get(machine.lower(), machine.lower())

    def fit(w) -> tuple:
        return ("linux" in w[1] and arch in w[1], "manylinux" in w[1])

    return max(wheels, key=fit)[2]


def _sdk(args: dict, secrets: dict, fetch, machine: str, newest: list) -> list:
    """The SDK's release and the Claude Code version its wheel carries."""
    key, wanted = "claude_agent_sdk.version", args["CLAUDE_AGENT_SDK_VERSION"]
    given = secrets.get("claude_agent_sdk_package_url")
    if given:
        address, what = given.value, "the wheel at claude_agent_sdk.package_url"
    else:
        index = secrets["python_package_index"].value.rstrip("/")
        try:
            wheels = _wheels(fetch(f"{index}/{SDK_PROJECT}/"))
        except deploy.FetchError as exc:
            return [
                Unresolved(
                    key,
                    f"the package index's page of {SDK_PROJECT} could not be read ({exc})",
                )
            ]
        if not wanted:
            plain = [w[0] for w in wheels if PLAIN_RELEASE.fullmatch(w[0])]
            if not plain:
                return [
                    Unresolved(
                        key, f"the package index lists no release of {SDK_PROJECT}"
                    )
                ]
            wanted = max(plain, key=_release_key)
            newest.append((key, wanted))
        mine = [w for w in wheels if w[0] == wanted]
        if not mine:
            return [
                Unresolved(
                    key, f"the package index has no wheel of {SDK_PROJECT} {wanted}"
                )
            ]
        address, what = (
            _this_machine(mine, machine),
            f"the wheel of {SDK_PROJECT} {wanted}",
        )
    try:
        with zipfile.ZipFile(_Ranges(fetch, address)) as wheel:
            names = wheel.namelist()
            if given:
                meta = next((n for n in names if METADATA.fullmatch(n)), "")
                m = METADATA_VERSION.search(
                    wheel.read(meta).decode("utf-8", "replace") if meta else ""
                )
                if not m:
                    return [
                        Unresolved(
                            key, f"{what} names no version (its .dist-info/METADATA)"
                        )
                    ]
                if wanted and m.group(1) != wanted:
                    return [
                        Unresolved(
                            key,
                            f"{what} is {SDK_PROJECT} {m.group(1)}, not claude_agent_sdk.version {wanted}",
                        )
                    ]
                wanted = m.group(1)
            m = CLI_VERSION.search(
                wheel.read(SDK_CLI_VERSION_FILE).decode("utf-8", "replace")
                if SDK_CLI_VERSION_FILE in names
                else ""
            )
    except deploy.FetchError as exc:
        return [Unresolved(key, f"{what} could not be read ({exc})")]
    except (zipfile.BadZipFile, EOFError, ValueError, KeyError, OSError):
        return [Unresolved(key, f"{what} could not be read as a wheel (a zip file)")]
    if not m:
        return [
            Unresolved(
                key, f"{what} names no Claude Code version ({SDK_CLI_VERSION_FILE})"
            )
        ]
    args["CLAUDE_AGENT_SDK_VERSION"], args["CLAUDE_CODE_VERSION"] = wanted, m.group(1)
    return []


class _Ranges(io.RawIOBase):
    """The file at URL, read through FETCH a part at a time, for zipfile: a range request of RANGE_BLOCK_BYTES or more
    for each part not read yet, its last part first, so that a wheel's table of contents and one of its files take a
    few small requests, not the whole wheel. A server that answers with the whole file (200) is read from that answer.
    ValueError for a part the server does not say the place of (Content-Range)."""

    def __init__(self, fetch, url: str):
        super().__init__()
        self._fetch, self._url, self._pos, self._parts = fetch, url, 0, []
        self.size = self._keep(fetch(url, f"-{RANGE_BLOCK_BYTES}"))

    def _keep(self, answer) -> int:
        """Keep ANSWER's part of the file; the file's size."""
        if answer.status == 200:
            self._parts = [(0, answer.body)]
            return len(answer.body)
        m = CONTENT_RANGE.fullmatch((answer.headers.get("content-range") or "").strip())
        if answer.status != 206 or not m:
            raise ValueError("the server sent a part of the file without its place")
        self._parts.append((int(m.group(1)), answer.body))
        return int(m.group(3))

    def _span(self, start: int, n: int) -> bytes:
        for at, data in self._parts:
            if at <= start and start + n <= at + len(data):
                return data[start - at : start - at + n]
        return b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def readinto(self, buffer) -> int:
        n = min(len(buffer), max(0, self.size - self._pos))
        if not n:
            return 0
        data = self._span(self._pos, n)
        if len(data) < n:
            end = min(self.size, self._pos + max(n, RANGE_BLOCK_BYTES)) - 1
            self._keep(self._fetch(self._url, f"{self._pos}-{end}"))
            data = self._span(self._pos, n)
            if len(data) < n:
                raise ValueError(
                    "the server sent another part of the file than the one asked for"
                )
        buffer[:n] = data
        self._pos += n
        return n


def _ca_sha256(files: tuple) -> tuple:
    """(the SHA-256 of each root certificate file, the Unresolved for each one that cannot be read)."""
    sums, unread = [], []
    for n, f in enumerate(files, 1):
        try:
            sums.append(hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest())
        except OSError:
            unread.append(
                Unresolved(
                    "operating_system.ca_certificates",
                    f"its file number {n} cannot be read",
                )
            )
    return tuple(sums), unread


# --- the record and the labels ---------------------------------------------------------------------------------------


def _targets(text: str) -> list:
    return [t for t in text.split(",") if t]


def _public(p: BuildPlan) -> dict:
    """The record's public build values: everything but its header, PACKAGE and DATE."""
    a = p.args
    return {
        "image": {"registry": "", "name": p.image.name, "tag": p.image.tag},
        "base": a["BASE_IMAGE"],
        "rocm": {
            "version": a["ROCM_VERSION"],
            "package_sha256": a["ROCM_PACKAGE_SHA256"],
            "clang_commit": a["ROCM_CLANG_COMMIT"],
            "library_targets": _targets(a["ROCM_LIBRARY_TARGETS"]),
            "hip_check_targets": _targets(a["HIP_CHECK_TARGETS"]),
        },
        "atlassian_cli": {
            "version": a["ATLASSIAN_CLI_VERSION"],
            "package_sha256": a["ATLASSIAN_CLI_PACKAGE_SHA256"],
        },
        "github_cli": {
            "version": a["GITHUB_CLI_VERSION"],
            "package_sha256": a["GITHUB_CLI_PACKAGE_SHA256"],
        },
        "claude_agent_sdk": {
            "version": a["CLAUDE_AGENT_SDK_VERSION"],
            "package_sha256": a["CLAUDE_AGENT_SDK_PACKAGE_SHA256"],
            "claude_code_version": a["CLAUDE_CODE_VERSION"],
        },
        "ca_certificates": {"count": len(p.ca_files), "sha256": list(p.ca_sha256)},
    }


def fingerprint(p: BuildPlan) -> str:
    """The build settings' fingerprint: see the module's docstring."""
    from . import file_types

    return settings_file.fingerprint(
        {"build_schema_version": file_types.BUILD.version, "build": _public(p)}
    )


def record(p: BuildPlan) -> dict:
    """The build record: see the module's docstring."""
    return {
        "kind": RECORD_KIND,
        "schema_version": RECORD_SCHEMA_VERSION,
        "build_settings_fingerprint": fingerprint(p),
        "package_version": p.package,
        "build_date": p.date,
        **_public(p),
    }


def parameter_lines(p: BuildPlan) -> list:
    """The build's parameters as `rkmon check` lists them: see the module's docstring."""
    given = {s.id for s in p.secrets if not s.public}
    rows = [("image", p.image.ref), ("architecture", p.machine)]
    rows += [(a.name, p.args.get(a.name) or "(blank)") for a in ARGS]
    rows += [
        (
            secret_variable(d.id),
            (
                f"build secret {d.id} (from {d.key}; never shown)"
                if d.id in given
                else f"(blank: {d.blank})"
            ),
        )
        for d in DOWNLOADS
    ]
    width = max(len(name) for name, _ in rows)
    return [f"  {name:<{width}} {value}" for name, value in rows]


def _labels(p: BuildPlan) -> dict:
    """The image's labels: see the module's docstring."""
    from . import image

    a = p.args
    return {
        "org.opencontainers.image.title": image.TITLE,
        "org.opencontainers.image.description": DESCRIPTION,
        "org.opencontainers.image.licenses": LICENSES,
        "org.opencontainers.image.version": p.package,
        "org.opencontainers.image.created": p.date,
        PACKAGE_LABEL: p.package,
        DATE_LABEL: p.date,
        BASE_LABEL: a["BASE_IMAGE"],
        ROCM_LABEL: a["ROCM_VERSION"],
        "org.rocke.rocm.clang_commit": a["ROCM_CLANG_COMMIT"],
        "org.rocke.rocm.library_targets": a["ROCM_LIBRARY_TARGETS"],
        "org.rocke.atlassian_cli.version": a["ATLASSIAN_CLI_VERSION"],
        "org.rocke.github_cli.version": a["GITHUB_CLI_VERSION"],
        "org.rocke.claude_agent_sdk.version": a["CLAUDE_AGENT_SDK_VERSION"],
        "org.rocke.claude_code.version": a["CLAUDE_CODE_VERSION"],
        FINGERPRINT_LABEL: fingerprint(p),
    }


def label_names() -> tuple:
    """The names of the image's labels, in their order: see the module's docstring."""
    blank = BuildPlan(
        {a.name: "" for a in ARGS}, (), (), {}, (), Image("", "", ""), "", "", ""
    )
    return tuple(_labels(blank))


# --- the docker build ------------------------------------------------------------------------------------------------

DOCKERFILE = "docker/Dockerfile"
# docker's build records where an image was built unless it is told not to: the package folder's git remote and
# commit (in the build's provenance record, and in labels when BUILDX_GIT_LABELS asks for them), and the provenance
# record itself, attached to the image. Every docker build runs with these variables and this flag, whatever your
# environment says, so the image carries none of them; podman's build makes no such record.
DOCKER_BUILD_ENV = {"BUILDX_GIT_INFO": "0", "BUILDX_GIT_LABELS": "0"}
DOCKER_BUILD_FLAGS = ("--provenance=false",)
# The build contexts the Dockerfile takes besides the package's code, each a private folder rkmon build makes for the
# build and removes after it: the root certificates the image trusts (copies of the files
# operating_system.ca_certificates names, deployment.stage_ca; with none named, the Dockerfile's
# `FROM scratch AS ca-certificates` stage, empty), and the build record (record's, as RECORD_FILE), which the image
# keeps at RECORD_PATH. The command as shown names each folder by what it holds: CA_SHOWN, RECORD_SHOWN.
CA_CONTEXT = "ca-certificates"
CA_SHOWN = "<operating_system.ca_certificates>"
RECORD_CONTEXT = "build-record"
RECORD_FILE = "build.json"
RECORD_SHOWN = "<the build record>"


class Command(NamedTuple):
    """One docker build: see the module's docstring."""

    argv: tuple
    env: dict
    shown_argv: tuple
    shown_env: tuple
    ref: str

    @property
    def shown(self) -> str:
        """The command as one line: its variables as shown, then its words, each quoted as a shell needs it."""
        return " ".join([*self.shown_env, *(shlex.quote(a) for a in self.shown_argv)])


def secret_variable(secret_id: str) -> str:
    """The variable docker build reads the build secret SECRET_ID from (`--secret id=ID,env=VARIABLE`): its id in
    capitals."""
    return secret_id.upper()


def command(
    p: BuildPlan,
    engine: str = "docker",
    ca_dir: str = CA_SHOWN,
    record_dir: str = RECORD_SHOWN,
    no_cache: bool = False,
) -> Command:
    """The docker build of plan P: see the module's docstring."""
    if p.unresolved:
        raise ValueError(
            "the plan has values the build cannot have yet: "
            + ", ".join(u.key for u in p.unresolved)
        )
    docker = os.path.basename(engine) == "docker"
    argv = [engine, "build", "-f", DOCKERFILE, "-t", p.image.ref]
    argv += DOCKER_BUILD_FLAGS if docker else ()
    argv += ["--no-cache"] if no_cache else []
    for a in ARGS:
        argv += ["--build-arg", f"{a.name}={p.args[a.name]}"]
    env = dict(DOCKER_BUILD_ENV) if docker else {}
    shown_env = [f"{name}={value}" for name, value in env.items()]
    for s in p.secrets:
        variable = secret_variable(s.id)
        argv += ["--secret", f"id={s.id},env={variable}"]
        env[variable] = s.value
        shown_env.append(f"{variable}={s.value if s.public else f'<{s.key}>'}")
    for name, text in p.labels.items():
        argv += ["--label", f"{name}={text}"]
    contexts = [(RECORD_CONTEXT, record_dir, RECORD_SHOWN)]
    if p.ca_files:
        contexts.insert(0, (CA_CONTEXT, ca_dir, CA_SHOWN))
    shown_argv = list(argv)
    for context, folder, words in contexts:
        argv += ["--build-context", f"{context}={folder}"]
        shown_argv += ["--build-context", f"{context}={words}"]
    argv.append(".")
    shown_argv.append(".")
    return Command(tuple(argv), env, tuple(shown_argv), tuple(shown_env), p.image.ref)

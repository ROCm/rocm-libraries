# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The deployment steps rk provision (rkmon_lib/provision.py) runs, each safe to run again, and the gh install of the
image's build (`gh_install`).

- `layout`: the working folder's subfolders (paths.py names them).
- `install_worker_tools`: worker/ into `paths.worker_tools`, writable (0755 or 0644, never the source's read-only mode);
  a file edited there since the last install is backed up first, once (`<name>.bak-<stamp>`; prcode.MANIFEST
  records each installed file). Beside them, first, the PR runners' group, a slot's home folder and the package's rk
  (prcode.VALUES_FILE): the installed worker/pr_user.py reads those values and runs no file it finds. It copies no
  path a package never includes (package_files.excluded): a cache, a compiled file, a backup, a monitor's state.
  The values and the manifest are each written whole (rkmon_lib/whole_file.py). `rk worker-tools --install` and the
  container's start run it.
- `clone_commands`: the monitor's own clone of `repo` into `paths.repo`, the way the toolkit expects it: blobless
  (`--filter=blob:none`), shallow (`--shallow-since=clone_since`), develop only, no checkout (every PR gets sparse
  worktrees), then its settings (`clone_settings_commands`: per-worktree sparse settings, no automatic gc). Then the
  develop worktree `wtdev`.
- `venv_commands`: the checks venv (and the optional generator venv) from requirements/, inside the working folder;
  none for a venv in place (`venv_in_place`: its install finished, which a mark written after pip says).

Every step returns what it did (or would do) as lines, which rk provision and `rk worker-tools --install` report.
"""

from __future__ import annotations

import datetime as dt
import filecmp
import hashlib
import http.client
import io
import json
import os
import pathlib
import platform
import re
import shlex
import shutil
import socket
import ssl
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import NamedTuple

from . import paths

# The functions that use code_volume, prcode, package_files or whole_file import them themselves: the image's
# gh layer (docker/Dockerfile) loads this module for gh_install with only paths.py beside it.
PACKAGE = pathlib.Path(__file__).resolve().parent.parent
LAYOUT = (
    "working_dir",
    "outputs_dir",
    "workers",
    "tmp",
    "probes",
    "lessons",
    "cache",
    "scratch",
    "build",
    "agentic",
    "worker_tools",
    "daily_dir",
    "weekly_dir",
)
CLONE_DAYS = 120  # the history a new clone keeps, by default


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S")


def default_since(days: int = CLONE_DAYS) -> str:
    return (dt.date.today() - dt.timedelta(days=days)).isoformat()


def layout(cfg: dict, dry: bool = False) -> list[str]:
    """Create the working folder's layout; the lines say what was (or would be) made."""
    p = paths.resolve(cfg)
    out = []
    for key in LAYOUT:
        d = pathlib.Path(p[key])
        extra = (
            [d / "notes", d / "archive"]
            if key == "lessons"
            else [d / "post", d / "sanity"] if key == "scratch" else []
        )
        for x in [d] + extra:
            if not x.is_dir():
                out.append(f"mkdir {x}")
                if not dry:
                    x.mkdir(parents=True, exist_ok=True)
    return out or ["the folders exist"]


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_file(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Copy the content only, then set the mode: 0755 for a source its package runs (code_volume.runs), else 0644.
    The source's own mode is never copied (the image's code is read-only, 0444: a copy with that mode could never be
    updated), and a target that is not writable is removed first."""
    from . import code_volume

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not os.access(dst, os.W_OK):
        dst.unlink()
    shutil.copyfile(src, dst)
    os.chmod(dst, 0o755 if code_volume.runs(src.stat().st_mode) else 0o644)


def _copy_tree(
    src: pathlib.Path,
    dst: pathlib.Path,
    dry: bool,
    backup: bool,
    out: list[str],
    manifest: dict | None = None,
    rel: str = "",
) -> None:
    """Copy `src` into `dst`. With `backup`, a target that differs is backed up first, unless it is the file an
    earlier install wrote (its digest is in `manifest`: not a local edit) or a backup with the same content is there
    already; `manifest` gets each copied file's digest. A path no package includes is never copied.
    """
    from . import package_files

    if src.is_dir():
        for child in sorted(src.iterdir()):
            if package_files.excluded(child.name):
                continue
            _copy_tree(
                child,
                dst / child.name,
                dry,
                backup,
                out,
                manifest,
                f"{rel}/{child.name}" if rel else child.name,
            )
        return
    if dst.exists() and filecmp.cmp(src, dst, shallow=False):
        if manifest is not None:
            manifest[rel or src.name] = _digest(src)
        if not dry and os.access(dst, os.F_OK) and not os.access(dst, os.W_OK):
            _copy_file(
                src, dst
            )  # the same content, but a mode no later install could write over
        return
    if dst.exists() and backup:
        have = _digest(dst)
        ours = manifest is not None and manifest.get(rel or src.name) == have
        kept = any(
            b.is_file() and _digest(b) == have
            for b in dst.parent.glob(f"{dst.name}.bak-*")
        )
        if not ours and not kept:
            out.append(f"back up {dst} -> {dst.name}.bak-{stamp()}")
            if not dry:
                shutil.copyfile(dst, dst.with_name(f"{dst.name}.bak-{stamp()}"))
    out.append(f"{'update' if dst.exists() else 'copy'} {dst}")
    if not dry:
        _copy_file(src, dst)
    if manifest is not None:
        manifest[rel or src.name] = _digest(src)


def _values_current(dest: pathlib.Path, package: pathlib.Path) -> bool:
    """Whether the values beside the tools (prcode.VALUES_FILE) are `package`'s."""
    from . import prcode

    try:
        have = (dest / prcode.VALUES_FILE).read_text(encoding="utf-8")
    except OSError:
        return False
    return have == prcode.installed_values(package)


def install_worker_tools(
    cfg: dict, package: pathlib.Path = PACKAGE, dry: bool = False
) -> list[str]:
    """Copy worker/ of `package` (the code that runs, by default) into paths.worker_tools, writable whatever the
    source's mode. A file there that an earlier install wrote (prcode.MANIFEST) is replaced as it is; one edited
    locally is backed up first, once. The values the installed worker/pr_user.py reads (prcode.VALUES_FILE: the
    layout, and `package`'s rk) go beside them first, so no installed worker/pr_user.py is without them. Safe to run
    again: `rk worker-tools --install` and the container's start run it.
    """
    from . import prcode, whole_file

    dest = pathlib.Path(paths.get(cfg, "worker_tools"))
    if not (package / "worker").is_dir():
        return [f"{package} has no worker/: nothing to install"]
    out: list[str] = []
    if not _values_current(dest, package):
        values = dest / prcode.VALUES_FILE
        out.append(f"write {values}")
        if not dry:
            dest.mkdir(parents=True, exist_ok=True)
            whole_file.write(values, prcode.installed_values(package), 0o644)
    try:
        manifest = json.loads((dest / prcode.MANIFEST).read_text())
        manifest = manifest if isinstance(manifest, dict) else {}
    except (OSError, ValueError):
        manifest = {}
    _copy_tree(package / "worker", dest, dry, backup=True, out=out, manifest=manifest)
    if not dry:
        dest.mkdir(parents=True, exist_ok=True)
        text = json.dumps(manifest, indent=1, sort_keys=True)
        whole_file.write(dest / prcode.MANIFEST, text, 0o644)
    return out or [f"{dest} is up to date"]


def stale_worker_tools(
    cfg: dict, package: pathlib.Path = PACKAGE
) -> list[tuple[str, str]]:
    """The worker tools paths.worker_tools lacks or holds another version of, against `package`'s worker/ (the code
    that runs), and the values beside them (prcode.VALUES_FILE): [(name, "missing" | "differs")].
    """
    from . import package_files, prcode

    dest = pathlib.Path(paths.get(cfg, "worker_tools"))
    out = []
    src = package / "worker"
    if src.is_dir() and not _values_current(dest, package):
        values = dest / prcode.VALUES_FILE
        out.append(
            (prcode.VALUES_FILE, "differs" if os.path.lexists(values) else "missing")
        )
    for f in sorted(src.rglob("*")) if src.is_dir() else []:
        if not f.is_file() or package_files.excluded(f.relative_to(src)):
            continue
        rel = str(f.relative_to(src))
        have = dest / rel
        if not have.is_file():
            out.append((rel, "missing"))
        elif not filecmp.cmp(f, have, shallow=False):
            out.append((rel, "differs"))
    return out


def clone_url(cfg: dict) -> str:
    """Where the clone comes from: the configuration's clone_url, by its rule (derived.clone_address): a mirror's
    address or a local path, else GitHub's address of repo; "" with neither."""
    # a call-time import: the image's gh layer loads this module with only paths.py beside it
    from . import derived

    return derived.clone_address(cfg.get("clone_url"), cfg.get("repo"))


def clone_commands(cfg: dict, url: str | None = None) -> list[list[str]]:
    """The commands that make the monitor's clone (nothing when it exists), its settings after it
    (clone_settings_commands)."""
    repo = paths.get(cfg, "repo")
    since = cfg.get("clone_since") or default_since()
    if (pathlib.Path(repo) / ".git").exists():
        return []
    return [
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--single-branch",
            "--branch",
            "develop",
            f"--shallow-since={since}",
            url or clone_url(cfg),
            repo,
        ],
        *clone_settings_commands(cfg),
    ]


def clone_settings_commands(cfg: dict) -> list[list[str]]:
    """The clone's own settings the toolkit expects: each worktree's own settings (every PR worktree's sparse
    checkout), and no automatic gc. Safe to run again: a setting in place stays as it is.
    """
    repo = paths.get(cfg, "repo")
    return [
        ["git", "-C", repo, "config", "extensions.worktreeConfig", "true"],
        ["git", "-C", repo, "config", "gc.auto", "0"],
    ]


def wtdev_commands(cfg: dict) -> list[list[str]]:
    """The develop worktree the checks and the worker read develop from (mkwt.py refreshes it), in a folder made first
    for a tree PR code reads, as mkwt.py makes it."""
    from . import prcode

    wt = paths.worktree(cfg, "wtdev")
    if pathlib.Path(wt).exists():
        return []
    repo = paths.get(cfg, "repo")
    return [
        prcode.tree_command(wt),
        [
            "git",
            "-C",
            repo,
            "worktree",
            "add",
            "-q",
            "--no-checkout",
            "--detach",
            wt,
            "refs/remotes/origin/develop",
        ],
        [
            "git",
            "-C",
            wt,
            "sparse-checkout",
            "set",
            "--no-cone",
            *paths.sparse_dirs(cfg),
        ],
        ["git", "-C", wt, "checkout", "-q", "--detach", "refs/remotes/origin/develop"],
    ]


def clone_check(cfg: dict) -> str | None:
    """Why the existing clone is not the one the toolkit expects, or None."""
    repo = pathlib.Path(paths.get(cfg, "repo"))
    if not (repo / ".git").exists():
        return f"no clone at {repo}"
    run = lambda *a: subprocess.run(
        ["git", "-C", str(repo), *a], capture_output=True, text=True
    ).stdout.strip()
    url = run("remote", "get-url", "origin")
    want = clone_url(cfg)
    same = url.rstrip("/").removesuffix(".git") == want.rstrip("/").removesuffix(".git")
    if (
        not same
        and not url.rstrip("/").removesuffix(".git").endswith(cfg.get("repo") or "\0")
        and not url.startswith("file:")
    ):
        return f"its origin is {url}, not {want or cfg.get('repo')}"
    if not run("rev-parse", "--verify", "-q", "refs/remotes/origin/develop"):
        return "it has no develop (refs/remotes/origin/develop)"
    return None


# The file a venv's install writes last (venv_commands), once pip has finished: a venv made in the working folder
# counts as in place only with it, so one cut short is made again.
VENV_MARK = ".rkmon-installed"


def venv_in_place(venv: str) -> bool:
    """Whether the venv VENV is in place, its install finished: the image's (container_layout.IMAGE_VENV, which the
    image's build makes whole or not at all) when its python is there; any other when it holds VENV_MARK.
    """
    # A call-time import: the image's gh layer loads this module with only paths.py beside it.
    from . import container_layout

    if pathlib.Path(venv) == pathlib.Path(container_layout.IMAGE_VENV):
        return (pathlib.Path(venv) / "bin" / "python").exists()
    return (pathlib.Path(venv) / VENV_MARK).is_file()


def venv_commands(
    cfg: dict, which: str = "checks", package: pathlib.Path = PACKAGE
) -> list[list[str]]:
    """The commands that make a venv inside the working folder (nothing when it is in place: venv_in_place), from its
    lock (requirements/<which>.lock.txt: the exact versions the image and the live checks run with), else its lower
    bounds (<which>.txt); the last one writes VENV_MARK, once pip has finished."""
    venv = (
        paths.get(cfg, "venv")
        if which == "checks"
        else f"{paths.get(cfg, 'working_dir')}/venv-{which}"
    )
    if venv_in_place(venv):
        return []
    lock = package / "requirements" / f"{which}.lock.txt"
    return [
        ["python3", "-m", "venv", venv],
        [
            f"{venv}/bin/python",
            "-m",
            "pip",
            "install",
            "-q",
            "-r",
            str(lock if lock.is_file() else package / "requirements" / f"{which}.txt"),
        ],
        ["touch", f"{venv}/{VENV_MARK}"],
    ]


GH_RELEASES = "https://github.com/cli/cli/releases"
GH_LATEST = "https://api.github.com/repos/cli/cli/releases/latest"
GH_ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}


class DeployError(RuntimeError):
    pass


class Answer(NamedTuple):
    """An HTTP GET's answer: its STATUS (200, or 206 for a part of the file), the URL it came from once redirects were
    followed, its HEADERS (each name in lower case) and its BODY. The URL may name a private host: it is never
    printed."""

    status: int
    url: str
    headers: dict
    body: bytes


class FetchError(OSError):
    """A download that failed, in words that name no host (download_failure)."""


# The HTTP answers a download is asked again for: a timeout, too many requests, a server's passing failure. Any other
# refusal stands. A connection that timed out or broke is asked again too; a host name that does not resolve, a refused
# connection, a certificate this machine does not trust and an address that is no URL are not.
RETRIED_STATUS = (408, 429, 500, 502, 503, 504)
FETCH_ATTEMPTS = 3
FETCH_PAUSE_SECONDS = 5.0


def _transient(exc: BaseException) -> bool:
    """Whether a download's failure EXC is worth another attempt (RETRIED_STATUS's comment)."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in RETRIED_STATUS
    reason = (
        getattr(exc, "reason", exc) if isinstance(exc, urllib.error.URLError) else exc
    )
    if isinstance(reason, (socket.gaierror, ConnectionRefusedError, ssl.SSLError)):
        return False
    return isinstance(
        reason,
        (TimeoutError, socket.timeout, ConnectionError, http.client.IncompleteRead),
    )


def _resolved(url: str) -> str:
    """URL with its path's dot segments resolved (RFC 3986, 5.2.4): /a/./b and /a/x/../b are /a/b, as curl and pip
    ask for them; a URL without one, as it is."""
    parts = urllib.parse.urlsplit(url)
    segments = parts.path.split("/")
    if "." not in segments and ".." not in segments:
        return url
    out = []
    for n, s in enumerate(segments):
        if s in (".", ".."):
            if s == ".." and len(out) > 1:
                out.pop()
            if n == len(segments) - 1:
                out.append("")  # /a/b/.. is /a/
            continue
        out.append(s)
    return urllib.parse.urlunsplit(parts._replace(path="/".join(out)))


def fetcher(
    ca_files=(),
    timeout_seconds: float = 120,
    attempts: int = FETCH_ATTEMPTS,
    pause_seconds: float = FETCH_PAUSE_SECONDS,
    sleep=time.sleep,
    open_url=None,
):
    """`fetch(url, byte_range="") -> Answer`: an HTTP(S) GET of URL, trusting this machine's root certificates and
    each of CA_FILES (PEM files; one that cannot be read is left out, as its own check says), through the proxy the
    environment names, if any. URL is asked for with its path's dot segments resolved (_resolved), as curl and pip
    ask for it. BYTE_RANGE asks for a part of the file ("START-END", or "-N" for its last N bytes): the server answers
    206 with that part, or 200 with the whole file. A failure worth another attempt (_transient) is tried again,
    ATTEMPTS times in all at most, PAUSE_SECONDS apart (SLEEP waits); a failure that stands raises FetchError, in
    words that name no host (download_failure). OPEN_URL, `open_url(request, timeout=SECONDS)`, opens a request and
    gives its response (the tests' fake server); by default, urllib's opener with that trust.
    """
    if open_url is None:
        context = tls_context()
        for f in ca_files:
            try:
                context.load_verify_locations(cafile=os.fspath(f))
            except (
                OSError,
                ValueError,
            ):  # unreadable, or no PEM certificate: its check's row says so
                pass
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=context)
        )
        open_url = opener.open

    def fetch(url: str, byte_range: str = "") -> Answer:
        headers = {"Range": f"bytes={byte_range}"} if byte_range else {}
        for attempt in range(1, attempts + 1):
            try:
                request = urllib.request.Request(_resolved(url), headers=headers)
                with open_url(request, timeout=timeout_seconds) as r:
                    return Answer(
                        r.status,
                        r.geturl(),
                        {k.lower(): v for k, v in r.headers.items()},
                        r.read(),
                    )
            except (OSError, ValueError, http.client.HTTPException) as exc:
                if attempt >= attempts or not _transient(exc):
                    raise FetchError(download_failure(exc)) from None
            sleep(pause_seconds)
        raise FetchError("it could not be downloaded")  # attempts below 1

    return fetch


def tls_context() -> ssl.SSLContext:
    """The default TLS context, which also takes a trusted issuing CA as the chain's anchor on every Python, as Python
    3.13 and newer do by default: on a network that inspects TLS, this machine may trust the network's issuing CA and
    not its root. The build settings name the root all the same: pip, in the image build, trusts a chain only up to a
    root (deployment.tls_checks). The release is checked against its published checksums either way.
    """
    ctx = ssl.create_default_context()
    ctx.verify_flags |= getattr(ssl, "VERIFY_X509_PARTIAL_CHAIN", 0)
    return ctx


def _get(url: str, timeout: float) -> bytes:
    """URL's body (fetcher's GET, asked again on a passing failure); FetchError when it fails."""
    return fetcher(timeout_seconds=timeout)(url).body


TAG_IN_URL = re.compile(r"/tag/v?(\d+(?:\.\d+)+)/?$")


def latest_gh(
    base: str = GH_RELEASES,
    latest: str = GH_LATEST,
    timeout: float = 120,
    fetch=None,
) -> str:
    """The newest gh release's version: the API's answer (`latest`), else, when the API refuses (its rate limit for
    callers with no login, an image build among them) or cannot be read, the release page's redirect: `<base>/latest`
    ends at `<base>/tag/v<version>`. FETCH reads each (fetcher's `fetch`; by default one that waits TIMEOUT seconds
    for an answer). DeployError when neither names one."""
    fetch = fetch or fetcher(timeout_seconds=timeout)
    try:
        return json.loads(fetch(latest).body)["tag_name"].lstrip("v")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        why = str(exc)[:120]
    try:
        m = TAG_IN_URL.search(fetch(f"{base}/latest").url)
    except OSError as exc:
        raise DeployError(
            f"the newest gh release could not be read ({why}; the release page: {str(exc)[:120]}): "
            "name a version (github_cli.version)"
        ) from None
    if not m:
        raise DeployError(
            f"the newest gh release could not be read ({why}; the release page names no tag): name a "
            "version (github_cli.version)"
        )
    return m.group(1)


GH_FILE = re.compile(r"(gh_([0-9][0-9.]*)_linux_[a-z0-9]+)\.tar\.gz")


def gh_archive(url: str):
    """The gh release archive a download address names (GH_FILE's match: its name, then its version), or None. The
    file's name is the last part of the address's path: a query or a fragment is no part of it.
    """
    return GH_FILE.fullmatch(urllib.parse.urlsplit(url).path.rpartition("/")[2])


def gh_file(version: str, arch: str) -> str:
    """A gh release's archive for Linux on ARCH (one of GH_ARCH's values): gh_<version>_linux_<arch>.tar.gz."""
    return f"gh_{version}_linux_{arch}.tar.gz"


def gh_checksums(version: str) -> str:
    """A gh release's published checksums: gh_<version>_checksums.txt, among its downloads."""
    return f"gh_{version}_checksums.txt"


def gh_download(version: str, file: str, base: str = GH_RELEASES) -> str:
    """The address of FILE among gh release VERSION's downloads, under BASE (GitHub's releases of gh)."""
    return f"{base}/download/v{version}/{file}"


SHA256_LINE = re.compile(r"([0-9a-fA-F]{64}) [ *]?(\S+)")


def published_sha256(sums: str, file: str) -> str:
    """The SHA-256 that a release's published checksums SUMS (sha256sum's lines: `<sha256>  <file>`) give FILE, in
    lower case; "" when they list no such file."""
    for line in sums.splitlines():
        m = SHA256_LINE.fullmatch(line.strip())
        if m and m.group(2) == file:
            return m.group(1).lower()
    return ""


def download_failure(exc: Exception) -> str:
    """Why a download failed, in fixed words that name no host: a package_url may name an internal host, and the
    error's own text can name it (a certificate's host name, an unknown address). A FetchError says it already.
    """
    if isinstance(exc, FetchError):
        return str(exc)
    if isinstance(exc, urllib.error.HTTPError):
        return f"the server answered HTTP {exc.code}"
    reason = getattr(
        exc, "reason", exc
    )  # a URLError carries what went wrong underneath
    if isinstance(reason, socket.gaierror):
        return "the host's name does not resolve"
    if isinstance(reason, ssl.SSLCertVerificationError):
        return (
            "the server's certificate is not one this machine trusts (a network that inspects TLS: name its root "
            "certificates in operating_system.ca_certificates)"
        )
    if isinstance(reason, ssl.SSLError):
        return "the TLS handshake failed"
    if isinstance(reason, ConnectionRefusedError):
        return "the host refused the connection"
    if isinstance(reason, TimeoutError):
        return "it timed out"
    if isinstance(reason, ConnectionError):
        return "the connection broke during the download"
    if isinstance(reason, http.client.IncompleteRead):
        return "the download ended before the whole file came"
    if isinstance(reason, ValueError) or str(reason).startswith("unknown url type"):
        return "the address is no URL it can read"
    return "it could not be downloaded"


def _download(url: str, timeout: float, what: str) -> bytes:
    """A download, its failure in fixed words that name no host (download_failure)."""
    try:
        return _get(url, timeout)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        raise DeployError(
            f"{what} did not download ({download_failure(exc)}): nothing installed"
        ) from None


def gh_install(
    dest_dir: str,
    version: str | None = None,
    base: str = GH_RELEASES,
    latest: str = GH_LATEST,
    timeout: float = 120,
    url: str | None = None,
    sha256: str | None = None,
) -> pathlib.Path:
    """gh in dest_dir (the image's gh stage, docker/Dockerfile): a release archive, checked against its SHA-256 before
    anything is installed (a mismatch installs nothing), and only the binary unpacked. `url` is the exact archive
    (gh_<version>_linux_<arch>.tar.gz; the image's github_cli.package_url); without it, the official release of
    `version` (blank: the latest) for this platform under `base`. `sha256` is the archive's SHA-256
    (github_cli.package_sha256); without it, the release's published checksums (gh_<version>_checksums.txt beside the
    archive). `version`, when given, is what gh must report.
    """
    want_version = (version or "").lstrip("v")
    if url:
        # The archive (gh_archive), and the folder its published checksums are in: the address's path's.
        parts = urllib.parse.urlsplit(url)
        folder = parts.path.rpartition("/")[0]
        m = gh_archive(url)
        name = m.group(1) if m else ""
        file_version = m.group(2) if m else ""
        what = "github_cli.package_url's archive"
        if not m and not sha256:
            raise DeployError(
                f"{what} names no gh_<version>_linux_<arch>.tar.gz, so its published checksums cannot be found: set "
                "github_cli.package_sha256"
            )
        data = _download(url, timeout, what)
        sums_url = (
            urllib.parse.urlunsplit(
                (
                    parts.scheme,
                    parts.netloc,
                    f"{folder}/{gh_checksums(file_version)}",
                    "",
                    "",
                )
            )
            if m
            else ""
        )
    else:
        if not want_version:
            want_version = latest_gh(base, latest, timeout)
        arch = GH_ARCH.get(platform.machine().lower())
        if not arch:
            raise DeployError(
                f"no gh release for this machine ({platform.machine()}): install gh by hand"
            )
        what = gh_file(want_version, arch)
        name = what[: -len(".tar.gz")]
        data = _download(gh_download(want_version, what, base), timeout, what)
        sums_url = gh_download(want_version, gh_checksums(want_version), base)
    got = hashlib.sha256(data).hexdigest()
    if sha256:
        if sha256.strip().lower() != got:
            raise DeployError(
                f"{what}'s SHA-256 is {got}, not github_cli.package_sha256 ({sha256.strip().lower()}): nothing "
                "installed"
            )
    else:
        sums = _download(sums_url, timeout, "the release's published checksums").decode(
            "utf-8", "replace"
        )
        want = published_sha256(sums, f"{name}.tar.gz")
        if not want or want != got:
            raise DeployError(
                f"{what} does not match the release's checksum: nothing installed"
            )
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
        if not name:
            name = next(
                (
                    n.split("/", 1)[0]
                    for n in t.getnames()
                    if n.endswith("/bin/gh") and n.count("/") == 2
                ),
                "",
            )
        try:
            member = t.getmember(f"{name}/bin/gh")
        except KeyError as exc:
            raise DeployError(
                f"{name}.tar.gz has no {name}/bin/gh: nothing installed"
            ) from exc
        if not member.isfile():
            raise DeployError(
                "the archive's gh is not a regular file: nothing installed"
            )
        binary = t.extractfile(member).read()
    dest = pathlib.Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "gh"
    tmp = dest / ".gh.new"
    tmp.write_bytes(binary)
    tmp.chmod(0o755)
    if version:
        try:
            said = subprocess.run(
                [str(tmp), "--version"], capture_output=True, text=True, timeout=60
            ).stdout
        except (OSError, subprocess.SubprocessError):
            said = ""
        m = re.search(r"gh version (\S+)", said)
        reported = m.group(1).lstrip("v") if m else ""
        if reported != version.lstrip("v"):
            tmp.unlink()
            raise DeployError(
                f"the archive's gh reports version {reported or 'none'}, not github_cli.version ({version}): nothing "
                "installed"
            )
    # os.replace, not whole_file.write: the image's gh layer (docker/Dockerfile) loads this module with only paths.py
    # beside it
    os.replace(tmp, out)
    return out


def show(cmds: list[list[str]]) -> list[str]:
    return [" ".join(shlex.quote(c) for c in cmd) for cmd in cmds]


def run(cmds: list[list[str]], env: dict | None = None) -> None:
    for cmd in cmds:
        p = subprocess.run(
            cmd, capture_output=True, text=True, env=dict(os.environ, **(env or {}))
        )
        if p.returncode:
            raise RuntimeError(
                f"{show([cmd])[0]} failed: {(p.stderr or p.stdout).strip()[:400]}"
            )

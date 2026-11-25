# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A sandbox for the end-to-end tests: a PATH with only ordinary tools (no real gh, Claude Code, Ollama or systemctl,
so nothing can reach GitHub or this machine's services), a fake checkout holding a copy of what the package holds
(the files the package build takes) at its place in rocKE's tree (with rocKE's ignore rules), and the environment
the commands run in: HOME and the pointer file inside the sandbox, the fake servers' addresses. For any test: a monitor
home of its own (`own_home`), its copy of the package's code, made its own to change (`writable`), and ports of its own
in place of a fixed one, which a server on this machine or a test running beside it may hold: one free for a server
under test (`free_port`), or one held closed (`closed_port`).
"""

import atexit
import os
import pathlib
import pty
import select
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from unittest import mock

PACKAGE = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PACKAGE / "tools"), str(PACKAGE)]
from rkmon_lib import (
    package_files,
)  # noqa: E402  (what the package holds, and its blank config.json)

# Tools the sandbox's PATH keeps. Everything else (gh, claude, ollama, systemctl, ssh, the Windows interop) is absent.
TOOLS = (
    "python3",
    "git",
    "sh",
    "bash",
    "dash",
    "env",
    "cat",
    "readlink",
    "dirname",
    "basename",
    "sed",
    "grep",
    "awk",
    "mkdir",
    "rm",
    "cp",
    "mv",
    "ls",
    "chmod",
    "date",
    "sleep",
    "head",
    "tail",
    "tr",
    "cut",
    "sort",
    "uniq",
    "wc",
    "find",
    "xargs",
    "tee",
    "nice",
    "timeout",
    "flock",
    "mktemp",
    "ps",
    "kill",
    "true",
    "false",
    "test",
    "printf",
    "echo",
    "id",
    "uname",
    "df",
    "du",
    "stat",
    "touch",
    "ln",
    "realpath",
    "tar",
    "gzip",
    "getconf",
    "setsid",
    "nohup",
    "pgrep",
)
# rocKE's own ignore rules (rocke/.gitignore) and its provider's, as the public repository has them: a package file
# under one of these names would never be committed.
ROCKE_IGNORE = (
    "CLAUDE.md\n.claude/\nSANDBOX_NOTES.md\nPROBLEM.md\n*.egg-info/\n__pycache__/\n*.pyc\n.pytest_cache/\n"
    ".venv*/\nvenv*/\n/build/\n/build-*/\n_deps/\n*.a\n*.o\n*.so\n*.hsaco\nkernel_cache/\n*.log\nlogs/\n"
    "scratch/\nlibrary/PROCESS_AUDIT.md\n"
)
PROVIDER_IGNORE = "install/\nbuild*/\n*.exe\n*.lib\n*.dll\n*.mod\n"
GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def safe_path(base: pathlib.Path, *without: str) -> str:
    """A PATH folder with links to the ordinary tools only (minus `without`)."""
    b = pathlib.Path(base) / "bin"
    b.mkdir(parents=True, exist_ok=True)
    for tool in TOOLS:
        src = shutil.which(tool, path="/usr/bin:/bin")
        if src and tool not in without and not (b / tool).exists():
            (b / tool).symlink_to(src)
    return str(b)


def git(repo, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ, **GIT_ENV),
    ).stdout


def copy_package(dest: pathlib.Path, source: pathlib.Path = PACKAGE) -> pathlib.Path:
    """SOURCE, the package, as it is checked in: the files it holds, as the package build takes them
    (package_files.tracked: no state, caches, git data or file git ignores, such as the CA folder's certificates), and
    the blank config.json."""
    for f in package_files.tracked(source):
        to = dest / f.relative_to(source)
        to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, to, follow_symlinks=False)
    writable(dest)
    (dest / "config.json").write_text(package_files.BLANK_CONFIG, encoding="utf-8")
    return dest


def writable(folder: pathlib.Path) -> pathlib.Path:
    """FOLDER and everything in it writable by its owner, a symbolic link aside: a test's copy of the package's code,
    which may be read-only (the image's, a container's /src/package), keeps its modes, yet the copy is the test's to
    change."""
    for p in [folder, *folder.rglob("*")]:
        if not p.is_symlink():
            p.chmod(p.stat().st_mode | 0o200)
    return folder


def own_home(test) -> pathlib.Path:
    """A monitor home of TEST's own (RKMON_HOME, until the test ends): a new, empty folder, removed after. What the
    code under test reads or writes in its home (config.json, state/) is then never that of a monitor this machine's
    environment or pointer file names."""
    home = tempfile.TemporaryDirectory()
    test.addCleanup(home.cleanup)
    p = mock.patch.dict(os.environ, RKMON_HOME=home.name)
    p.start()
    test.addCleanup(p.stop)
    return pathlib.Path(home.name)


PORT_BLOCK = 32  # the ports of a test process's own block (free_port), its first one kept bound to claim the block
_block: list = (
    []
)  # this process's block: [its pid, the socket bound to its first port, that port, the next to give, its end]


def ephemeral_ports() -> tuple[int, int]:
    """The kernel's range of ports for the local end of a connection (/proc/sys/net/ipv4/ip_local_port_range): a
    client connection, or a bind to port 0, takes its port from it."""
    try:
        text = pathlib.Path("/proc/sys/net/ipv4/ip_local_port_range").read_text()
        low, high = map(int, text.split())
    except (OSError, ValueError):
        low, high = 32768, 60999
    return low, high


def port_block() -> list:
    """This process's block of PORT_BLOCK ports on 127.0.0.1, outside the kernel's ephemeral range (below it, else
    above it): the first block whose first port this process can bind, a socket it keeps bound until it ends (and
    closes then), so no other test process, of this run or another, takes the block. Claimed once per process (a
    forked one claims its own)."""
    if _block and _block[0] == os.getpid():
        return _block
    low, high = ephemeral_ports()
    for start, end in ((max(1024, low - 12288), low), (high + 1, 65536)):
        for first in range(start, end - PORT_BLOCK + 1, PORT_BLOCK):
            s = socket.socket()
            try:
                s.bind(("127.0.0.1", first))
            except OSError:
                s.close()
                continue
            _block[:] = [os.getpid(), s, first, first + 1, first + PORT_BLOCK]
            atexit.register(s.close)
            return _block
    raise OSError(
        "no block of free ports outside the kernel's ephemeral range on 127.0.0.1"
    )


def free_port() -> int:
    """A port on 127.0.0.1 that nothing has bound now: for a server under test, which binds the port its settings
    name. From this process's own block (port_block), so no other test process gives it out, and outside the kernel's
    ephemeral range, so no client connection takes it as its local port meanwhile; each one checked free by a bind, in
    turn, round the block. Free when asked, not kept (the server binds it at once)."""
    block = port_block()
    first, end = block[2], block[4]
    for _ in range(PORT_BLOCK - 1):
        port = block[3]
        block[3] = port + 1 if port + 1 < end else first + 1
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise OSError(
        f"every port of this test process's block {first}-{end - 1} on 127.0.0.1 is in use"
    )


def closed_port(test) -> int:
    """A port on 127.0.0.1 that refuses every connection until TEST ends (a test, or a test class: then until its last
    test ends): bound, never listening, and held, so no server on this machine answers there and no other test takes
    it. For an address a test needs closed."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    (test.addClassCleanup if isinstance(test, type) else test.addCleanup)(s.close)
    return s.getsockname()[1]


def fake_checkout(base: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """A git work tree with rocKE in a folder of no repository's, with its ignore rules and its provider's, and the
    package committed at its place in rocKE's tree: (checkout, package)."""
    co = pathlib.Path(base) / "checkout"
    rocke = co / "team" / "provider" / "rocke"
    rocke.mkdir(parents=True)
    (rocke / ".gitignore").write_text(ROCKE_IGNORE, encoding="utf-8")
    (rocke.parent / ".gitignore").write_text(PROVIDER_IGNORE, encoding="utf-8")
    (rocke / "README.md").write_text("rocKE\n", encoding="utf-8")
    pkg = copy_package(rocke / "tools" / "rocke_monitor")
    git(co, "init", "-q", "-b", "develop")
    git(co, "add", "-A")
    git(co, "commit", "-qm", "the package")
    return co, pkg


def environment(
    base: pathlib.Path, path: str, urls: dict | None = None, **extra
) -> dict:
    """The commands' environment: HOME and XDG_CONFIG_HOME in the sandbox (the pointer file and the units' folder land
    there), the sandbox PATH, and the fake servers' addresses for the fake gh."""
    home = pathlib.Path(base) / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": path,
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(pathlib.Path(base) / "tmp"),
    }
    pathlib.Path(env["TMPDIR"]).mkdir(exist_ok=True)
    if urls:
        env.update(FAKE_GITHUB_URL=urls["github"], FAKE_GH_LOGIN="op-login")
    env.update(extra)
    return env


def on_a_terminal(
    argv: list, cwd, env: dict, typed: bytes = b"", replies=(), timeout: float = 120
) -> tuple[int, str]:
    """ARGV run in CWD on a terminal of its own (a pseudo-terminal for its stdin, its output and its errors), with
    TYPED typed on it (nothing by default; the terminal echoes it), then the TEXT of each (PROMPT, TEXT) of REPLIES once
    what it wrote since the reply before holds PROMPT (a question asked without echo drops what was typed before it):
    its exit code and all it wrote there. A command still running after TIMEOUT seconds is killed, and its exit code
    says so.
    """
    pid, fd = pty.fork()
    if pid == 0:  # the child: the command on the terminal
        try:
            os.chdir(cwd)
            os.execve(argv[0], list(argv), env)
        finally:
            os._exit(127)  # never back into the tests
    if typed:
        os.write(fd, typed)
    replies, chunks, since = list(replies), [], b""
    end, killed = time.monotonic() + timeout, False
    while True:
        left = None if killed else max(0.0, end - time.monotonic())
        if not select.select([fd], [], [], left)[0]:
            # it waits for what is never typed: no test waits for ever
            os.kill(pid, signal.SIGKILL)
            killed = True
            continue
        try:
            data = os.read(fd, 4096)
        except OSError:  # the terminal closed: the command is done
            break
        if not data:
            break
        chunks.append(data)
        since += data
        if replies and replies[0][0] in since:
            prompt, text = replies.pop(0)
            since = since[since.index(prompt) + len(prompt) :]
            os.write(fd, text)
    os.close(fd)
    _pid, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status), b"".join(chunks).decode(
        "utf-8", "replace"
    )

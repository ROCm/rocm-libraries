#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build the rocKE monitor package: the folder exactly as it sits in the repository, and a tarball of it. Or write
its generated parts again from their sources.

  python3 -B tools/make_package.py --out DIR --deny FILE [--black BLACK] [--no-tests] [--new-version]
      [--install-into SANDBOX_CHECKOUT]
  python3 -B tools/make_package.py --refresh-generated

The generated parts (tools/generated.py) are the docs' generated sections and the generated files: .gitignore,
.dockerignore, the settings templates' golden texts, docs/internal_settings.md and VERSION. They are committed with
the code, so a clone of the package's repository builds with --out and --deny alone, and the folder, the image and
`rkmon package update` get them from the checked-in tree. Every build compares each section and file with its source,
and VERSION with its form, and stops, naming each one out of step. --refresh-generated runs alone and builds nothing:
it writes each generated section and file in the tree whose text differs from its source again, for you to review and
commit. VERSION is the package's version, which `rkmon build` labels the image with: only a release changes it. A
build with --new-version makes a release: it writes VERSION, today's date, into the built folder and, once the checks
and the tests pass, into the tree, for you to review and commit. The documents (docs/) and the agentic files
(agentic/) are the package's own: the build copies them as they are.

The build makes the package from a clone of its repository, of the files git lists there: tracked, or untracked and
not ignored, but a Python environment's. It refuses a folder where git tracks no file, since a copy of every file
there would package whatever was left in it, a secret file among them. That rule and the copy are
rkmon_lib/package_files.py's, which `rkmon package update` copies a new package by too.

Its steps, in their order, and --install-into: docs/maintaining.md, "Building the package and its tarball".

What the check looks for comes from the deny list (--deny, which the build requires; rkmon_lib/privacy.py says what it
lists), the build machine's own identity (its git identity, home, user and host names, and GITNAME, GITEMAIL, WINUSER,
WINHOST in the environment), the package's blank defaults (config.defaults.json) and generic patterns. No monitor's
configuration or state is read, nor a Claude workspace: `python3 -B -m rkmon_lib.privacy --seed-deny` writes their
values into the deny list. The build reads the deny list before it makes anything, and stops when it cannot be read or
gives no value; the check reads it again, in its own process. None of these values is written anywhere. The check
reads the built folder with the deny list before the tests; the tests' own scan of the package's files
(tests/test_privacy.py) looks for the build machine's names and the generic patterns.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from rkmon_lib import (  # noqa: E402
    checks,
    image,
    package_files,
    privacy,
)

# What a package holds, and its blank config.json (rkmon_lib/package_files.py), as this module's readers name them.
from rkmon_lib.package_files import (  # noqa: E402,F401
    BLANK_CONFIG,
    SKIP_PARTS,
    tracked,
)

# The package's folder name: the built folder's, and the top of the tarball.
NAME = "rocke_monitor"
# Files that cannot hold a comment, and so carry no header.
NO_HEADER = {".json"}
HEADER_LINES = (
    "Copyright (c) Advanced Micro Devices, Inc., or its affiliates.",
    "SPDX-License-Identifier: MIT",
)


def header_problem(f: pathlib.Path) -> str:
    """Why a file lacks rocKE's header where it belongs, or ""."""
    if f.suffix in NO_HEADER or f.stat().st_size == 0:
        return ""
    try:
        text = f.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""  # binary
    head = text[:1500]
    if not all(h in head for h in HEADER_LINES):
        return "no header"
    if f.suffix == ".md" and text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end < 0 or text.find(HEADER_LINES[0]) < end:
            return "the header must follow the front matter"
    if (
        f.suffix == ".md"
        and text.lstrip().startswith("<!--")
        and re.sub(r"^\s*<!--.*?-->\s*", "", text, count=1, flags=re.S).startswith(
            "---\n"
        )
    ):
        return "the header must follow the front matter"  # front matter is only front matter at the very top
    if text.startswith("#!") and HEADER_LINES[0] not in text.split("\n", 3)[1]:
        return "the header must be the lines right after #!"
    return ""


def python_files(folder: pathlib.Path) -> list[pathlib.Path]:
    """The files pre-commit's black hook reads as Python: every *.py, and every executable file with no suffix whose
    first line is a python shebang (identify's rule: docker/rkmon, the test fakes)."""
    out = []
    for f in sorted(folder.rglob("*")):
        if not f.is_file() or set(f.relative_to(folder).parts) & SKIP_PARTS:
            continue
        if f.suffix == ".py":
            out.append(f)
        elif not f.suffix and os.access(f, os.X_OK):
            with f.open("rb") as fh:
                first = fh.readline()
            if first.startswith(b"#!") and b"python" in first:
                out.append(f)
    return out


def run_black(black: str, folder: pathlib.Path) -> None:
    """black on every Python file, named one by one (as pre-commit does): a folder given to black is skipped when its
    path has a part black excludes by default, such as dist/. The package's own source is formatted with the same
    release and settings, so the built folder's Python files match it."""
    files = [str(f) for f in python_files(folder)]
    p = subprocess.run([black, "-q", *files], capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"black failed: {p.stderr.strip()[:2000]}")
    p = subprocess.run([black, "--check", "-q", *files], capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(
            f"black still wants changes: {(p.stdout + p.stderr).strip()[:2000]}"
        )


def normalized(text: str) -> str:
    """A text file as the package ships it: no trailing blanks on a line, one newline at the end."""
    return (
        "\n".join(line.rstrip() for line in text.splitlines()).rstrip("\n") + "\n"
        if text.strip()
        else text
    )


def version_text(day: dt.date | None = None) -> str:
    """VERSION: the header, then the package's version, the day DAY of its release (today when None)."""
    return (
        f"# {HEADER_LINES[0]}\n# {HEADER_LINES[1]}\n{day or dt.date.today():%Y.%m.%d}\n"
    )


def test_env(base: dict, bindir: pathlib.Path, pycache: pathlib.Path) -> dict:
    """The tests' environment, before the suite's own settings (rkmon_lib/selftest.py `environ`, which `run` applies):
    the caller's, but its PATH is the build's own (BINDIR's `python` and `python3`, this interpreter, first; then the
    system's folders; then the caller's other folders, a venv's aside), so a venv on the caller's PATH, or no `python`
    there, makes no difference; and bytecode a test's own child writes all the same goes to PYCACHE, outside the
    package.
    """
    drop = ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH")
    env = {k: v for k, v in base.items() if k not in drop}
    system = [
        "/usr/local/sbin",
        "/usr/local/bin",
        "/usr/sbin",
        "/usr/bin",
        "/sbin",
        "/bin",
    ]
    # The caller's other folders (a git or ctest of its own) come last; a venv's bin folder never does.
    theirs = [
        d
        for d in base.get("PATH", "").split(os.pathsep)
        if d and not (pathlib.Path(d).parent / "pyvenv.cfg").is_file()
    ]
    env["PATH"] = os.pathsep.join(
        dict.fromkeys([str(bindir), *system, *theirs])
    )  # in order, each once
    env["PYTHONPYCACHEPREFIX"] = str(pycache)
    return env


def python_shim(bindir: pathlib.Path) -> None:
    """BINDIR/python and BINDIR/python3: this interpreter, whatever the caller's PATH names so."""
    bindir.mkdir(parents=True, exist_ok=True)
    for name in ("python", "python3"):
        f = bindir / name
        f.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
        f.chmod(0o755)


def run_tests(folder: pathlib.Path) -> None:
    import tempfile

    from rkmon_lib import selftest

    with tempfile.TemporaryDirectory(prefix="rkmon-build-") as tmp:
        bindir, pycache = pathlib.Path(tmp) / "bin", pathlib.Path(tmp) / "pycache"
        python_shim(bindir)
        env = test_env(dict(os.environ), bindir, pycache)
        # A home folder, a TMPDIR and a monitor home of the suite's own: a test that leaves anything there, or that
        # adds anything to the built folder (bytecode a test's own child writes despite the settings above), fails the
        # build.
        p = selftest.run(folder, env, capture=True, timeout=3600)
    tail = p.output.strip().splitlines()[-3:]
    print("tests: " + " / ".join(tail))
    if p.returncode:
        raise SystemExit(
            "the package's tests failed in the built folder:\n" + p.output[-6000:]
        )


def tar_entry(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """A tarball member as the package ships it: no owner or group name, uid and gid 0, no write bit for the group or
    others. A tar header otherwise carries the build machine's user name in every member, which the privacy check,
    reading file contents only, never sees."""
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode &= ~0o022
    return info


def install_into(folder: pathlib.Path, checkout: pathlib.Path) -> None:
    """Copy the built folder over the package in a folder below the top of a sandbox checkout, found as a package
    update finds the package in a fetched commit (package_files.package_folders); refuse a checkout with a remote (a
    real clone), and one with no such package or more than one."""
    p = subprocess.run(
        ["git", "-C", str(checkout), "remote"], capture_output=True, text=True
    )
    if p.returncode:
        raise SystemExit(f"--install-into: {checkout} is not a git work tree")
    if p.stdout.strip():
        raise SystemExit(
            f"--install-into: {checkout} has a remote ({p.stdout.split()[0]}): it is for sandbox "
            "checkouts only, never a real clone"
        )
    ls = ("ls-files", "-z", "--cached", "--others", "--exclude-standard")
    files = subprocess.run(
        ["git", "-C", str(checkout), *ls], capture_output=True, text=True, check=True
    ).stdout.split("\0")
    # one at its top is the checkout itself, the package's own repository
    found = [f for f in package_files.package_folders(files) if f]
    if len(found) != 1:
        what = (
            f"a package in each of {', '.join(found[:-1])} and {found[-1]}, and the built folder replaces one"
            if found
            else "no package in a folder below its top for the built folder to replace"
        )
        raise SystemExit(f"--install-into: {checkout} has {what}")
    dest = checkout / found[0]
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(folder, dest, symlinks=True)
    st = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "status",
            "--porcelain",
            "--ignored",
            "--",
            found[0],
        ],
        capture_output=True,
        text=True,
    ).stdout
    ignored = [ln for ln in st.splitlines() if ln.startswith("!!")]
    print(
        f"installed into {dest} (not committed); git status --porcelain --ignored there: "
        f"{len(st.splitlines())} line(s), {len(ignored)} ignored"
    )
    if ignored:
        raise SystemExit("git would ignore package files:\n" + "\n".join(ignored))


def main(argv=None) -> int:
    import generated  # tools/generated.py reads this module's functions: imported once this module is whole

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out",
        help="the dist folder (the package goes to OUT/rocke_monitor)",
    )
    ap.add_argument(
        "--deny",
        metavar="FILE",
        help="the deny list, kept outside the package: the values the privacy check looks for",
    )
    ap.add_argument(
        "--black", help="the black executable (25.12.0) to format the Python files with"
    )
    ap.add_argument(
        "--no-tests",
        action="store_true",
        help="do not run the package's tests in the built folder",
    )
    ap.add_argument(
        "--install-into",
        help="a sandbox checkout (no remote) to copy the built folder into",
    )
    ap.add_argument(
        "--new-version",
        action="store_true",
        help="a release: VERSION, today's date, in the built folder and, once the checks and the tests pass, in the "
        "tree: review the change and commit it",
    )
    ap.add_argument(
        "--refresh-generated",
        action="store_true",
        help="alone: write each generated section and file in the tree whose text differs from its source again "
        "(tools/generated.py); it builds nothing and never writes VERSION: review the change and commit it",
    )
    a = ap.parse_args(argv)
    if a.refresh_generated:
        # Alone: another option with it is a usage error. A part its source cannot make stops it before it writes.
        others = [
            "--" + dest.replace("_", "-")
            for dest, value in vars(a).items()
            if dest != "refresh_generated" and value != ap.get_default(dest)
        ]
        if others:
            ap.error(
                "--refresh-generated runs alone, with no other option: "
                + ", ".join(others)
            )
        try:
            wrote = generated.refresh(ROOT)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        print(
            f"wrote {', '.join(wrote)}: review the change and commit it"
            if wrote
            else "nothing to write: every generated part is in step with its source"
        )
        return 0
    missing = ["--" + dest for dest in ("out", "deny") if not getattr(a, dest)]
    if missing:
        ap.error("the following arguments are required: " + ", ".join(missing))
    # A clone of the package's repository only: where git tracks no file, the build refuses (the module's docstring).
    files = tracked(ROOT, every_file=False)
    if not files:
        raise SystemExit(
            f"git tracks no file in {ROOT}: the package is built from a clone of its repository"
        )
    out = pathlib.Path(a.out).resolve()
    folder = out / NAME
    if folder.exists():
        shutil.rmtree(folder)
    # 1. The toolkit, each file ours to write, from a read-only source too (the image's), and the blank config.json.
    package_files.copy(ROOT, folder, files)
    # 2. The deny list, read before anything is built (privacy.sources): one that cannot be read, or that gives no
    # value, stops the build here, and its row's fix runs this command again, naming another. The privacy check (step
    # 5) reads it again itself.
    typed = (
        "python3",
        "-B",
        "tools/make_package.py",
        *(sys.argv[1:] if argv is None else argv),
    )
    _values, rows = privacy.sources(deny=a.deny, typed=typed)
    if any(r.status == checks.Status.FAIL for r in rows):
        print(checks.render(rows, "problems"))
        raise SystemExit(
            "the deny list cannot be read or gives no value: see the problems above"
        )
    # 3. The generated parts, against their sources (tools/generated.py): the checked-in tree carries what a build
    # from it needs (a clone builds, deploys and updates with nothing else). A release replaces VERSION, whatever the
    # tree's is: today's date, in the built folder.
    if a.new_version:
        drift = generated.drift(
            ROOT, files=[f for f in generated.FILES if f.path != "VERSION"]
        )
    else:
        drift = generated.drift(ROOT)
    if drift:
        raise SystemExit(
            "the generated parts are out of step with their sources:\n  "
            + "\n  ".join(drift)
        )
    if a.new_version:
        (folder / "VERSION").write_text(version_text(), encoding="utf-8")
    # 4. Formatting, headers, whitespace.
    if a.black:
        run_black(a.black, folder)
    for f in folder.rglob("*"):
        if f.is_file() and f.suffix not in (".gz", ".png"):
            try:
                text = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            fixed = normalized(text)
            if fixed != text:
                f.write_text(fixed, encoding="utf-8")
    bad = [
        f"{f.relative_to(folder)}: {why}"
        for f in sorted(folder.rglob("*"))
        if f.is_file()
        for why in [header_problem(f)]
        if why
    ]
    if bad:
        raise SystemExit(
            "rocKE's header is missing or misplaced:\n  " + "\n  ".join(bad)
        )
    # 5. The privacy check, as the built folder's own code (privacy.command, run in the folder). It runs in a process
    # of its own, so it reads the deny list again itself: no value is handed to it or written.
    report = out / "privacy_report.txt"
    report.unlink(missing_ok=True)
    p = subprocess.run(
        privacy.command(
            "--deny", pathlib.Path(a.deny).resolve(), "--report", report, "."
        ),
        cwd=folder,
        capture_output=True,
        text=True,
    )
    print(p.stdout, end="")
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit(
            "the privacy check failed: see privacy_report.txt (labels and counts only)"
            if report.exists()
            else "the privacy check could not run in the built folder: see its error above"
        )
    # 6. The tests, in the built folder.
    if not a.no_tests:
        run_tests(folder)
    # A release's VERSION, once the checks and the tests passed: into the package's own tree.
    if a.new_version:
        version = (folder / "VERSION").read_text(encoding="utf-8")
        (ROOT / "VERSION").write_text(version, encoding="utf-8")
        print(
            f"wrote VERSION, {version.splitlines()[-1]}, into {ROOT}: review the change and commit it"
        )
    # 7. The tarball, with nothing of the machine that built it in its headers, named by the package's version
    # (YYYY.MM.DD), which the image built from it is tagged with too (image.package_version)
    tar = out / f"{NAME}-{image.package_version(folder)}.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(folder, arcname=NAME, filter=tar_entry)
    print(f"built {folder} and {tar}")
    if a.install_into:
        install_into(folder, pathlib.Path(a.install_into).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())

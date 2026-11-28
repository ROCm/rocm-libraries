#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
The canary test of the image's build: a build with a canary (tests/container/canary_search.py) in each private value
of your build settings, then a search of everything the build left for the canaries, for the hosts of your download
locations and for your home folder's path.
  1. a private copy of your build settings (in a private folder of its own, removed at the end), its image.tag TAG,
     with a canary in each download location they name and in each root certificate's file name: a file's URL gets it as a fragment
     (#<canary>, which no download sends), an index or a mirror as a path segment the next one undoes (/<canary>/../,
     which the client or the server resolves); a blank index takes the public one the build takes, and a blank mirror
     Ubuntu's own archive. A NOTE names each download no canary goes through: a blank file URL, and a mirror on a base
     other than Ubuntu
  2. `rkmon build` of the copy, with no cache (a build secret is no part of a step's cache key: a step from the cache
     would never see its canary), its output kept in the private folder
  3. a search for the canaries, for the host of each download location of yours (but a public host, which the image
     may name anyway, and a host without a dot, which any file may name: a NOTE names each) and for your home folder,
     in the build's output, in every layer of the image (docker save: each file's name, a link's target, extended
     headers and content), in its configuration, manifests and history; and the image carries no provenance record
     (whose vcs:source would name your checkout). A file the image keeps as it was downloaded (its SHA-256 is its
     download's package_sha256, as the atlassian CLI's binary's is) may name its own host: a NOTE names that place
  4. the image test's labels and canary stages on the image (tests/container/image.py): the build's labels and its
     record are there, every label is org.opencontainers.* or org.rocke.*, and no PR runner finds a canary planted in a
     container
A find is named by where it is and what it is (canary <kind>, the host of <key>, or the builder's home folder), never
by its text.

Usage: tests/container/canary.py --settings FILE [--package DIR] [--engine docker|podman] [--tag TAG]
       [--names PREFIX] [--keep-image]
FILE is your build settings file; DIR the package folder to build (default: the one this script is in); TAG the
image's tag (default canary-test); PREFIX begins the image test's container names (default rkmon-image-test). The
image is removed at the end unless --keep-image.
Network: the whole build (the base image's packages, ROCm, the tools).
Exit 0 when nothing was found anywhere, the image carries no provenance record, and both stages passed.
"""

import copy
import dataclasses
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse

import canary_search as cs
import hostrun

TAG = "canary-test"
FRAGMENT_KEYS = (
    "rocm.package_url",
    "atlassian_cli.package_url",
    "github_cli.package_url",
    "claude_agent_sdk.package_url",
)  # a file's URL: the canary as its fragment
MIRROR = "operating_system.package_mirror"
SEGMENT_KEYS = (
    "python_packages.package_index",
    MIRROR,
)  # an index's or a mirror's: the canary as a path segment the next one undoes
CERTIFICATE_KIND = "certificate"
# The archive pkgs.sh's mirror takes the place of. Each address is written in two parts, as the package's tests
# write one, since the privacy check reads a whole address as a link to check.
UBUNTU_ARCHIVE = {
    "amd64": "http:/" + "/archive.ubuntu.com/ubuntu",
    "arm64": "http:/" + "/ports.ubuntu.com/ubuntu-ports",
}
HOME_WORDS = "the builder's home folder"
HOST_WORDS = "the host of {key}"
# What each search looks for, in its result's words.
SEARCHED = "no canary, no host of yours and not the builder's home folder"
PROVENANCE_MARKS = (
    b"vcs:source",
    b"vcs:revision",
    b"attestation-manifest",
    b"in-toto",
    b"slsa.dev/provenance",
)
COMPRESSED = (
    b"\x1f\x8b",
    b"\x28\xb5\x2f\xfd",
    b"\xfd7zXZ\x00",
    b"BZh",
)  # gzip, zstd, xz, bzip2: a layer
# A configuration, a manifest or an index is read whole up to this size.
JSON_MAX_BYTES = 64 << 20
READ_BYTES = cs.CHUNK_BYTES


def kind_of(key: str) -> str:
    """A key's canary kind: its section's letters (rocm, atlassiancli, ...)."""
    return re.sub(r"[^a-z]", "", key.split(".")[0])[:16]


def with_fragment(url: str, canary: str) -> str:
    """URL with CANARY as its fragment (after the one it has)."""
    head, _, fragment = url.partition("#")
    return f"{head}#{fragment + '-' if fragment else ''}{canary}"


def with_segment(url: str, canary: str) -> str:
    """URL with CANARY as a first path segment that the next one undoes: scheme://host/CANARY/../path."""
    p = urllib.parse.urlsplit(url)
    path = p.path if p.path.startswith("/") else "/" + p.path
    return urllib.parse.urlunsplit(p._replace(path=f"/{canary}/..{path}"))


def ubuntu(base: str) -> bool:
    """Whether BASE is an Ubuntu image: the archives docker/pkgs.sh's mirror takes the place of are Ubuntu's."""
    return base.split("/")[-1].split(":")[0] == "ubuntu"


def _get(d: dict, dotted: str):
    for k in dotted.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _put(d: dict, dotted: str, v) -> None:
    *parents, last = dotted.split(".")
    for k in parents:
        d = d.setdefault(k, {})
    d[last] = v


def decorate(values: dict, folder: str, p, arch: str) -> tuple:
    """VALUES, your build settings (a build settings document's values), with a canary in each download location (the
    module's docstring, step 1) and the root certificates P names copied into FOLDER under a canary's name, P being
    the plan rkmon_lib/build_plan.py makes of them (its downloads in build_plan.DOWNLOADS' order; ARCH, the
    machine's, as Ubuntu's archives name it): (the copy, {kind: canary}, the notes)."""
    from rkmon_lib import build_plan, deployment

    out, canaries, notes = copy.deepcopy(values), {}, []
    given = {s.key: s for s in p.secrets}
    base = p.args["BASE_IMAGE"]
    for d in build_plan.DOWNLOADS:
        s, kind = given.get(d.key), kind_of(d.key)
        blank = f"{d.key} is blank ({d.blank}): no canary goes through it"
        if d.key in FRAGMENT_KEYS:
            if not s or s.public:
                notes.append(blank)
                continue
            canaries[kind] = cs.make(kind)
            _put(out, d.key, with_fragment(s.value, canaries[kind]))
            continue
        if d.key not in SEGMENT_KEYS:
            notes.append(
                f"{d.key}: the canary test has no form for it, so no canary goes through it"
            )
            continue
        if d.key == MIRROR and not ubuntu(base):
            notes.append(
                f"{d.key}: the base {base} keeps its own mirror list, so no canary goes through it"
            )
            continue
        value = s.value if s else ""
        if d.key == MIRROR and not value:
            value = UBUNTU_ARCHIVE.get(arch, UBUNTU_ARCHIVE["amd64"])
        if not value:
            notes.append(blank)
            continue
        canaries[kind] = cs.make(kind)
        _put(out, d.key, with_segment(value, canaries[kind]))
    if p.ca_files:
        canaries[CERTIFICATE_KIND] = cs.make(CERTIFICATE_KIND)
        copies = []
        for n, f in enumerate(p.ca_files, 1):
            d = os.path.join(folder, f"{canaries[CERTIFICATE_KIND]}-{n}.crt")
            shutil.copyfile(f, d)
            copies.append(d)
        _put(out, deployment.CA_KEY, copies)
    return out, canaries, notes


def host_needles(p) -> tuple:
    """The finder's needles for the host of each download location of yours that P, the plan of your build settings,
    gives the build (HOST_WORDS: by its key, never its text), and a note for each one that is not searched for: a
    public host (the hosts of the package's public downloads, rkmon_lib/build_plan.py PUBLIC_HOSTS), which the image
    may name anyway, or a host without a dot, which any file may name. (needles, notes)
    """
    from rkmon_lib import build_plan

    public = set(build_plan.PUBLIC_HOSTS)
    needles, notes = {}, []
    for s in p.secrets:
        host = "" if s.public else urllib.parse.urlsplit(s.value).hostname or ""
        if not host:
            continue
        if host in public:
            notes.append(
                f"{s.key} names a public host, which the image may name anyway: it is not searched for"
            )
        elif "." not in host and ":" not in host:
            notes.append(
                f"{s.key} names a host without a dot, which any file may name: it is not searched for"
            )
        else:
            needles[HOST_WORDS.format(key=s.key)] = host.encode()
    return needles, notes


def own_files(p) -> dict:
    """The SHA-256 of each file download that P, the plan of your build settings, gives the build with its digest ->
    (its location's key, its digest's key). The image keeps such a file as it came when it installs the download itself
    (the atlassian CLI's binary), and a host it names there is the download's own, which it named before any build
    did: the search's note, not a find (`_layer`)."""
    from rkmon_lib import build_plan

    names = {a.key: a.name for a in build_plan.ARGS}
    out = {}
    for s in p.secrets:
        digest_key = f"{s.key.rsplit('.', 1)[0]}.package_sha256"
        digest = p.args.get(names.get(digest_key, ""), "")
        if digest:
            out[digest.lower()] = (s.key, digest_key)
    return out


def home_needles(
    home: str, image_homes=(), windows: bool = os.name == "nt"
) -> tuple[dict, str]:
    """The finder's needles for the builder's HOME (a Windows path in both its forms), and a note instead when there is
    none to search for: no home, /, or a home every image has (/root, IMAGE_HOMES)."""
    if home.rstrip("/\\") in ("", "/root", *image_homes):
        return (
            {},
            f"your home folder ({home or 'none'}) is one every image has: it is not searched for",
        )
    needles = {HOME_WORDS: home.encode()}
    if windows and "\\" in home:
        needles[f"{HOME_WORDS} (with forward slashes)"] = home.replace(
            "\\", "/"
        ).encode()
    return needles, ""


@dataclasses.dataclass
class Scan:
    """What a search of an image's archive found: (what, where) pairs, the provenance marks, the layers read, the
    files read in them, and the parts that could not be read. OWN, given before the search, is own_files' map; NOTES
    gets a line for each file of the image that is a download as it came and names that download's own host, which is
    no find."""

    found: list = dataclasses.field(default_factory=list)
    marks: set = dataclasses.field(default_factory=set)
    layers: int = 0
    files: int = 0
    unreadable: list = dataclasses.field(default_factory=list)
    own: dict = dataclasses.field(default_factory=dict)
    notes: list = dataclasses.field(default_factory=list)


class _Joined:
    """HEAD, then the rest of the file F: the bytes peeked at, read again."""

    def __init__(self, head: bytes, f):
        self.head, self.f = head, f

    def read(self, n=-1):
        if not self.head:
            return self.f.read(n)
        if n is None or n < 0:
            out, self.head = self.head + self.f.read(), b""
            return out
        out, self.head = self.head[:n], self.head[n:]
        return out


def _name(finder: cs.Finder, raw: bytes) -> str:
    """A path in the image as it may be shown: each part of it that holds a canary as `(a part that holds a find)`,
    since the rest of that part may be one of your private values; a path that holds a needle, not at all.
    """
    if finder.hides(raw):
        return "(a name that holds a find)"
    return "/".join(
        "(a part that holds a find)" if cs.PATTERN.search(part) else cs.shown(part)
        for part in raw.split(b"/")
    )


def _in_image(name: str) -> bytes:
    """A layer member's name as the image's absolute path."""
    name = name[2:] if name.startswith("./") else name
    return ("/" + name.lstrip("/")).encode("utf-8", "surrogateescape")


def _hashed(f, digest):
    """A read of F's next chunk (b"" at its end) that adds each chunk to DIGEST."""

    def read() -> bytes:
        chunk = f.read(READ_BYTES)
        digest.update(chunk)
        return chunk

    return read


def _layer(scan: Scan, finder: cs.Finder, member: str, stream) -> None:
    """One layer's files: each name (as the image's absolute path), a link's target, the extended headers, and each
    regular file's content. A regular file whose SHA-256 is a download's in SCAN.own is that download as it came: the
    host of its own location there goes to SCAN.notes, not to its finds."""
    label = member.rsplit("/", 1)[-1][:12]
    scan.layers += 1
    try:
        with tarfile.open(fileobj=stream, mode="r|*") as layer:
            for m in layer:
                name = _in_image(m.name)
                where = f"layer {label}: {_name(finder, name)}"
                headers = " ".join(f"{k}={v}" for k, v in (m.pax_headers or {}).items())
                for what in sorted(finder.in_bytes(name)):
                    scan.found.append((what, f"{where} (its name)"))
                if m.issym() or m.islnk():
                    target = (
                        _in_image(m.linkname)
                        if m.islnk()
                        else m.linkname.encode("utf-8", "surrogateescape")
                    )
                    for what in sorted(finder.in_bytes(target)):
                        scan.found.append((what, f"{where} (where the link points)"))
                for what in sorted(
                    finder.in_bytes(headers.encode("utf-8", "surrogateescape"))
                ):
                    scan.found.append((what, f"{where} (its extended headers)"))
                if m.isreg():
                    scan.files += 1
                    digest = hashlib.sha256()
                    what, _size = finder.in_stream(
                        _hashed(layer.extractfile(m), digest)
                    )
                    own = scan.own.get(digest.hexdigest())
                    if own and HOST_WORDS.format(key=own[0]) in what:
                        what.discard(HOST_WORDS.format(key=own[0]))
                        scan.notes.append(
                            f"the host of {own[0]} is in {where}, which is that download as it came (its SHA-256 is "
                            f"{own[1]}): it names its own host, and that is no find"
                        )
                    scan.found += [(w, where) for w in sorted(what)]
    except (tarfile.TarError, OSError, EOFError) as exc:
        scan.unreadable.append(f"layer {label} ({type(exc).__name__}: {exc})")


def scan_archive(stream, finder: cs.Finder, scan: Scan) -> Scan:
    """Search an image's archive (`docker save`'s, read from STREAM) into SCAN, which keeps what it found if the
    archive breaks off: every layer (_layer), and every other part (configurations, manifests, indexes: JSON, searched
    for the provenance marks too) as it is."""
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            if not member.isreg():
                continue
            f = archive.extractfile(member)
            head = f.read(512)
            if head.startswith(COMPRESSED) or head[257:262] == b"ustar":
                _layer(scan, finder, member.name, _Joined(head, f))
                continue
            if head.lstrip()[:1] in (b"{", b"[") and member.size <= JSON_MAX_BYTES:
                data = head + f.read()
                scan.marks |= {m.decode() for m in PROVENANCE_MARKS if m in data}
                what = finder.in_bytes(data)
            else:
                joined = _Joined(head, f)
                what, _size = finder.in_stream(lambda: joined.read(READ_BYTES))
            scan.found += [(w, member.name) for w in sorted(what)]
    return scan


def save_scan(
    engine: str, ref: str, finder: cs.Finder, own=None
) -> tuple[int, Scan, str]:
    """`docker save REF`, searched as it streams (scan_archive; nothing is written), with OWN as the scan's own
    (own_files' map; none by default): (save's exit code, the scan, the end of its errors).
    """
    scan = Scan(own=dict(own or {}))
    with tempfile.TemporaryFile() as err:
        p = subprocess.Popen([engine, "save", ref], stdout=subprocess.PIPE, stderr=err)
        try:
            scan_archive(p.stdout, finder, scan)
            # To the archive's end, so that save ends by itself.
            while p.stdout.read(READ_BYTES):
                pass
        except (tarfile.TarError, OSError, EOFError) as exc:
            scan.unreadable.append(f"the archive ({type(exc).__name__}: {exc})")
        finally:
            p.stdout.close()
            rc = p.wait()
        err.seek(0)
        return rc, scan, err.read().decode("utf-8", "replace")[-2000:]


def build(argv: list, env: dict, finder: cs.Finder, folder: str) -> tuple[int, list]:
    """Run the build ARGV with ENV, its output shown as it comes and searched, FOLDER's path (the test's own) taken out
    of it first; a line that holds a find is not shown, since a canary sits in one of your private values: (its exit
    code, the finds: (what, "the build's output, line N"))."""
    own = os.fsencode(folder)
    found = []
    sys.stdout.flush()
    with subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
    ) as p:
        for n, line in enumerate(p.stdout, 1):
            line = line.replace(own, b"<the canary test's folder>")
            what = sorted(finder.in_bytes(line))
            found += [(w, f"the build's output, line {n}") for w in what]
            text = line.decode("utf-8", "replace").rstrip("\n")
            if what:
                text = f"(line {n}, which holds {' and '.join(what)}: not shown)"
            print("   | " + text)
    return p.returncode, found


def engine_finds(engine: str, ref: str, finder: cs.Finder) -> list:
    """The finds in the image's history and its configuration as the engine shows them (history, image inspect)."""
    found = []
    for words, argv in (
        (
            "its history",
            [engine, "history", "--no-trunc", "--format", "{{json .}}", ref],
        ),
        ("its configuration (image inspect)", [engine, "image", "inspect", ref]),
    ):
        rc, text = hostrun.capture(argv)
        if rc != 0:
            found.append(("nothing it could read", f"{words}: {argv[1]} exited {rc}"))
            continue
        raw = text.encode("utf-8", "surrogateescape")
        found += [(w, words) for w in sorted(finder.in_bytes(raw))]
    return found


def judged(res, found: list, what: str) -> None:
    """One result: PASS when FOUND is empty; else FAIL, after a line for each find."""
    for words, where in found:
        print(f"   FOUND {words} in {where}")
    res.result("FAIL" if found else "PASS", what)


def main(args: list) -> int:
    hostrun.setup_output()
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = hostrun.folder(os.path.join(here, "..", ".."))
    engine = os.environ.get("CONTAINER_ENGINE", "")
    settings, tag, names, keep = "", TAG, "", False
    while args:
        a = args[0]
        if a in ("--settings", "--package", "--engine", "--tag", "--names"):
            v = hostrun.need_value(args, a)
            if a == "--settings":
                settings = v
            elif a == "--package":
                pkg = hostrun.folder(v)
            elif a == "--engine":
                engine = v
            elif a == "--tag":
                tag = v
            else:
                names = v
            args = args[2:]
        elif a == "--keep-image":
            keep, args = True, args[1:]
        elif a in ("-h", "--help"):
            sys.stdout.write(hostrun.help_lines(__doc__, 30))
            return 0
        else:
            print(f"unknown argument: {a} (see --help)")
            return 2
    if not settings or not pkg:
        print("name --settings FILE, your build settings file (see --help)")
        return 2
    engine = hostrun.pick_engine(engine)
    if not engine:
        return 2
    sys.path.insert(0, pkg)
    from rkmon_lib import build_plan, checks, container_layout, deploy, file_types
    from rkmon_lib import image, settings_file

    try:
        doc = settings_file.read(settings, file_types.BUILD)
        rows = settings_file.validate(doc)
    except settings_file.Refused as exc:
        rows = list(exc.rows)
    except OSError as exc:
        print(f"FAIL  {settings}: {exc.strerror or exc}")
        return 2
    if checks.failed(rows):
        print(checks.render(rows, "problems"))
        print(
            f"FAIL  {settings}: no build settings file this package builds (the rows above)"
        )
        return 2
    # The image this test builds, and removes, is its own: the copy of your settings names TAG as its tag (rkmon
    # build takes the tag from the file alone).
    values = copy.deepcopy(doc.values)
    _put(values, "image.tag", tag)
    doc = dataclasses.replace(doc, values=values)
    p = build_plan.plan(doc, image.package_version(pkg), "", platform.machine())
    homes = tuple(f"/home/{u.name}" for u in container_layout.USERS)
    needles, note = home_needles(os.path.expanduser("~"), homes)
    hosts, host_notes = host_needles(p)
    finder = cs.Finder({**needles, **hosts})
    arch = deploy.GH_ARCH.get(platform.machine().lower(), "amd64")
    res, ref = hostrun.Results(), ""
    folder = tempfile.mkdtemp(prefix="rkmon-canary-")
    try:
        copied, canaries, notes = decorate(doc.values, folder, p, arch)
        for line in [note, *host_notes, *notes]:
            if line:
                print(f"NOTE  {line}")
        print(f"== canaries in: {', '.join(sorted(canaries)) or 'nothing'}")
        print(f"== hosts searched for: {', '.join(sorted(hosts)) or 'none'}")
        file = hostrun.write_settings(folder, copied)
        ref = p.image.ref
        rkmon = [sys.executable, "-B", os.path.join(pkg, "docker", "rkmon")]
        argv = [*rkmon, "build", "--settings", file, "--no-cache"]
        rc, found = build(
            argv, dict(os.environ, CONTAINER_ENGINE=engine), finder, folder
        )
        res.result(
            "PASS" if rc == 0 else "FAIL",
            f"the build of {ref} with the canaries, with no cache",
        )
        if rc != 0:
            ref = ""  # nothing built: nothing to search or remove
            return 1
        judged(res, found, f"the build's output holds {SEARCHED}")
        print(f"== searching {ref} as {engine} save writes it")
        rc, scan, err = save_scan(engine, ref, finder, own_files(p))
        res.result(
            "PASS" if rc == 0 else "FAIL",
            f"{engine} save {ref}" + ("" if rc == 0 else f": {err}"),
        )
        for line in scan.notes:
            print(f"NOTE  {line}")
        judged(
            res,
            scan.found + [("nothing it could read", u) for u in scan.unreadable],
            f"{SEARCHED} in any of its {scan.layers} layers ({scan.files} files: names, links, headers, contents), "
            "nor in its configuration and manifests",
        )
        judged(
            res,
            [(m, "the image's manifests") for m in sorted(scan.marks)],
            "the image carries no provenance record (none of its marks, vcs:source among them)",
        )
        judged(
            res,
            engine_finds(engine, ref, finder),
            f"{SEARCHED} in its history and its configuration",
        )
        stages = [
            sys.executable,
            "-B",
            os.path.join(pkg, "tests", "container", "image.py"),
        ]
        stages += ["--package", pkg, "--engine", engine, "--no-build", "--image", ref]
        stages += ["--stages", "labels,canary"] + (["--names", names] if names else [])
        sys.stdout.flush()
        rc = subprocess.run(stages).returncode
        res.result(
            "PASS" if rc == 0 else "FAIL", "the image test's labels and canary stages"
        )
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        if ref and not keep:
            hostrun.capture([engine, "rmi", ref])
        print("CANARY TEST: PASS" if res.failed == 0 else "CANARY TEST: FAIL")
    return res.failed


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

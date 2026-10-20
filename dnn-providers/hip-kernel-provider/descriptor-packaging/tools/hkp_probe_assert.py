#!/usr/bin/env python3
"""Assert the output of a packaging probe.

A packaging probe packs one descriptor root for one explicit architecture.
This tool inspects the packed output root and fails, with one greppable line per
failure, unless the shipped set is the one `--expect` lists
(hkp_probe_derive_root.py writes it) and every UKD is well formed for its kind: a JSON
list of {"kdp": <KDP path relative to the arch directory>, "name": <UKD name>, "kind":
<authored kernel_source.kind>, "arch": <the pass the UKD packs under>}, plus the
authored "kernel_source" for kinds whose output is a pass-through. The pass is `--arch`,
or a generic target containing it: the packer compiles a generic once and merges its
descriptors and archive into the member folder, so the UKD ships under `<out>/<arch>/`
with `arch` [<generic>]. The checks applied to a UKD follow its EXPECTED kind's output
type (hkp_probe_kinds.KINDS): `kpack` UKDs are checked against the archive of their
pass; `passthrough` UKDs must ship as authored. The archive of a pass
(<out>/<arch>/kpack/hip_kernel_provider_<pass>.kpack) is checked (kpack-missing,
kpack-empty, kpack-toc, sha256, symbol, signature) for each pass that has at least one
kpack-output UKD in the expect list; a pass-through-only root ships no archive.

    hkp_probe_assert: FAIL <assertion-id>: <detail>

Exit status is 0 only when every assertion holds, 1 when any assertion fails,
and 2 for an unusable invocation (rocm_kpack not importable, --expect
unreadable or malformed). Plain python, not pytest: a script has no skip path, so it
cannot pass without having checked.

Assertion ids:
    out-root-missing   --out-root is not a directory
    stamp-missing      <out>/<stamp-name> does not exist
    arch-dir-missing   <out>/<arch>/ does not exist
    extra-arch-dir     another gfx* directory sits beside <out>/<arch>/
    kpack-missing      kpack output expected for a pass but <out>/<arch>/kpack/
                       hip_kernel_provider_<pass>.kpack is absent
    kpack-empty        that archive has size 0
    no-kdp             no *.kdp.json under <out>/<arch>/, or one is not valid JSON
    provenance-sidecar a KDP's provenance sidecar is missing, unreadable, or does not
                       match its UKDs (a packed UKD carries its provenance there, and
                       the checks below read it from it)
    ukd-count          the number of shipped UKDs found in the --expect list differs
                       from its length (or is 0), or a shipped UKD is not in the list
    ukd-kind           a UKD's shipped kernel_source.kind is not what its expected
                       kind's output type requires ("kpack", or the authored kind for
                       a pass-through)
    arch-field         a KDP `arch` present and not [<pass>], or a UKD `arch` absent
                       or not [<pass>], where <pass> is the expected UKD's `arch` (for
                       a KDP, that of its expected UKDs; --arch when it has none)
    passthrough-source a pass-through UKD's kernel_source differs from the authored one
    passthrough-provenance
                       a pass-through UKD's provenance.source_label is empty or its
                       provenance.source_file differs from kernel_source.source_file
    kpack-toc          an archive is unreadable or has no entry, under its pass, for a
                       UKD's toc_key
    sha256             the archive blob's sha256 differs from kernel_source.sha256
    signature          a UKD's signature is not a non-empty list
    symbol             a UKD's symbol does not appear in its archive blob
    no-rules-for-kind  a UKD's expected kind has no entry in hkp_probe_kinds.KINDS
    provenance-origin  provenance.origin_kind is not the UKD's expected kind
    provenance-wheel   kinds with the wheel check (rocke): provenance.rocke_wheel_sha256
                       absent or empty
    provenance-comgr   kinds with the comgr check (rocke): provenance.comgr_path is not
                       the expected comgr library
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Put the package dir ahead of everything, including this script's own dir: tools/
# also holds a module literally named hkp_pack (hkp_pack.py), which would otherwise
# shadow the hkp_pack package when tools/ is sys.path[0].
_PKG_ROOT = str(Path(__file__).resolve().parent.parent / "python")
while _PKG_ROOT in sys.path:
    sys.path.remove(_PKG_ROOT)
sys.path.insert(0, _PKG_ROOT)

from hkp_pack import provenance_sidecar  # noqa: E402
from hkp_pack.errors import HkpPackError  # noqa: E402
from hkp_pack.pipeline import _kpack_filename  # noqa: E402
from hkp_probe_kinds import KINDS  # noqa: E402

_PREFIX = "hkp_probe_assert"


class _Failures:
    def __init__(self):
        self.lines = []

    def add(self, assertion_id, detail):
        self.lines.append(f"{_PREFIX}: FAIL {assertion_id}: {detail}")

    def __bool__(self):
        return bool(self.lines)


def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="Assert the output of a hip-kernel-provider packaging probe."
    )
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--arch", required=True)
    p.add_argument("--kpack-python-dir", required=True)
    p.add_argument("--stamp-name", required=True)
    p.add_argument("--expect", required=True, type=Path)
    p.add_argument("--expect-comgr", default=None)
    return p.parse_args(argv)


class _UsageError(Exception):
    pass


def _load_expect(path):
    """Read --expect into {(kdp, name): entry}; entry keeps its other fields."""
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _UsageError(f"cannot read --expect {path}: {type(exc).__name__}: {exc}")
    if not isinstance(entries, list):
        raise _UsageError(f"--expect {path} is not a JSON list")
    expected = {}
    for entry in entries:
        if not isinstance(entry, dict) or not all(
            isinstance(entry.get(k), str) for k in ("kdp", "name", "kind", "arch")
        ):
            raise _UsageError(
                f"--expect {path} entry {entry!r} lacks string kdp, name, kind and arch"
            )
        if _output_of(entry["kind"]) == "passthrough" and not isinstance(
            entry.get("kernel_source"), dict
        ):
            raise _UsageError(
                f"--expect {path} entry {entry!r}: a {entry['kind']} entry needs the "
                "authored kernel_source object"
            )
        key = (entry["kdp"], entry["name"])
        if key in expected:
            raise _UsageError(f"--expect {path} lists {key} twice")
        expected[key] = entry
    return expected


def _output_of(kind):
    """The registered output type of `kind`, or None when it has no entry."""
    entry = KINDS.get(kind)
    return entry.output if entry is not None else None


def _load_kpack(kpack_python_dir):
    if kpack_python_dir not in sys.path:
        sys.path.insert(0, kpack_python_dir)
    from rocm_kpack import kpack

    return kpack


def _same_path(a, b):
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(
        os.path.realpath(b)
    )


def _check_layout(args, failures):
    """Checks 2-4. Returns the arch dir, or None when later checks are moot."""
    out = args.out_root
    if not (out / args.stamp_name).exists():
        failures.add("stamp-missing", f"{out / args.stamp_name} does not exist")

    arch_dir = out / args.arch
    if not arch_dir.is_dir():
        failures.add("arch-dir-missing", f"{arch_dir} is not a directory")
        arch_dir = None
    others = sorted(
        c.name
        for c in out.iterdir()
        if c.is_dir() and c.name.startswith("gfx") and c.name != args.arch
    )
    if others:
        failures.add(
            "extra-arch-dir",
            f"expected only {args.arch} under {out}, also found {', '.join(others)}",
        )
    return arch_dir


def _check_kpack(arch_dir, pass_arch, failures):
    """Check 4 for one pass. Returns the archive path when it is non-empty."""
    kpack_path = arch_dir / "kpack" / _kpack_filename(pass_arch)
    if not kpack_path.is_file():
        failures.add("kpack-missing", f"{kpack_path} does not exist")
        return None
    if kpack_path.stat().st_size == 0:
        failures.add("kpack-empty", f"{kpack_path} has size 0")
        return None
    return kpack_path


def _check_wheel(args, label, prov, failures):
    wheel = prov.get("rocke_wheel_sha256")
    if not isinstance(wheel, str) or not wheel:
        failures.add("provenance-wheel", f"{label} rocke_wheel_sha256 is {wheel!r}")


def _check_comgr(args, label, prov, failures):
    comgr = prov.get("comgr_path")
    if args.expect_comgr is None:
        if not isinstance(comgr, str) or not comgr:
            failures.add("provenance-comgr", f"{label} comgr_path is {comgr!r}")
    elif not isinstance(comgr, str) or not _same_path(comgr, args.expect_comgr):
        failures.add(
            "provenance-comgr",
            f"{label} comgr_path is {comgr!r}, expected {args.expect_comgr}",
        )


# Implementations of the check names listed in hkp_probe_kinds.Kind.provenance.
_PROVENANCE_CHECKS = {"wheel": _check_wheel, "comgr": _check_comgr}


def _check_provenance(args, label, ukd, kind, failures):
    entry = KINDS[kind]
    prov = ukd.get("provenance") or {}
    origin = prov.get("origin_kind")
    if origin != kind:
        failures.add(
            "provenance-origin", f"{label} origin_kind is {origin!r}, not {kind!r}"
        )
    for name in entry.provenance:
        _PROVENANCE_CHECKS[name](args, label, prov, failures)


def _check_kpack_ukd(args, label, ukd, expected, archives, failures):
    source = ukd.get("kernel_source") or {}
    kind = source.get("kind")
    if kind != "kpack":
        failures.add(
            "ukd-kind",
            f"{label} kernel_source.kind is {kind!r}; expected kind "
            f'{expected["kind"]!r} has kpack output, which ships "kpack"',
        )
        return

    pass_arch = expected["arch"]
    if ukd.get("arch") != [pass_arch]:
        failures.add("arch-field", f"{label} arch is {ukd.get('arch')!r}")

    signature = source.get("signature")
    if not isinstance(signature, list) or not signature:
        failures.add("signature", f"{label} signature is {signature!r}")

    _check_provenance(args, label, ukd, expected["kind"], failures)

    archive = archives.get(pass_arch)
    if archive is None:
        return
    toc_key = source.get("toc_key")
    blob = archive.get_kernel(toc_key, pass_arch) if toc_key else None
    if blob is None:
        failures.add(
            "kpack-toc", f"{label} toc_key {toc_key!r} has no {pass_arch} archive entry"
        )
        return
    digest = hashlib.sha256(blob).hexdigest()
    if digest != source.get("sha256"):
        failures.add(
            "sha256",
            f"{label} archive blob sha256 {digest} != "
            f"descriptor {source.get('sha256')!r}",
        )
    symbol = source.get("symbol")
    if not isinstance(symbol, str) or not symbol or symbol.encode("utf-8") not in blob:
        failures.add("symbol", f"{label} symbol {symbol!r} not found in archive blob")


def _check_passthrough_ukd(args, label, ukd, expected, archives, failures):
    source = ukd.get("kernel_source") or {}
    kind = source.get("kind")
    if kind != expected["kind"]:
        failures.add(
            "ukd-kind",
            f"{label} kernel_source.kind is {kind!r}; expected kind "
            f'{expected["kind"]!r} has pass-through output, which ships as authored',
        )
        return

    if ukd.get("arch") != [expected["arch"]]:
        failures.add("arch-field", f"{label} arch is {ukd.get('arch')!r}")

    if source != expected["kernel_source"]:
        failures.add(
            "passthrough-source",
            f"{label} kernel_source is {source!r}, authored "
            f"{expected['kernel_source']!r}",
        )

    _check_provenance(args, label, ukd, expected["kind"], failures)

    prov = ukd.get("provenance") or {}
    label_value = prov.get("source_label")
    if not isinstance(label_value, str) or not label_value:
        failures.add(
            "passthrough-provenance", f"{label} source_label is {label_value!r}"
        )
    if prov.get("source_file") != source.get("source_file"):
        failures.add(
            "passthrough-provenance",
            f"{label} provenance.source_file {prov.get('source_file')!r} != "
            f"kernel_source.source_file {source.get('source_file')!r}",
        )


# Per-UKD check set for each output type named by hkp_probe_kinds.Kind.output.
_OUTPUT_CHECKS = {"kpack": _check_kpack_ukd, "passthrough": _check_passthrough_ukd}


def _check_ukd(args, kdp_name, ukd, archives, failures):
    """Check one shipped UKD against its expectation. True when it was expected."""
    label = f"{kdp_name}:{ukd.get('name', ukd.get('id', '?'))}"
    expected = args.expected.get((kdp_name, ukd.get("name")))
    if expected is None:
        failures.add(
            "ukd-count", f"{label} is not in the --expect list ({args.expect})"
        )
        return False
    output = _output_of(expected["kind"])
    if output is None:
        failures.add(
            "no-rules-for-kind",
            f"{label} expected kind {expected['kind']!r} has no rules",
        )
        return True
    _OUTPUT_CHECKS[output](args, label, ukd, expected, archives, failures)
    return True


def _kdp_arch(args, kdp_name):
    """The arch a shipped KDP carries: the pass of its expected UKDs, else --arch."""
    passes = sorted({e["arch"] for (k, _), e in args.expected.items() if k == kdp_name})
    return passes if passes else [args.arch]


def _check_descriptors(args, arch_dir, archives, failures):
    """Checks 5-7: every UKD of every shipped KDP."""
    kdps = sorted(arch_dir.rglob("*.kdp.json"))
    if not kdps:
        failures.add("no-kdp", f"no *.kdp.json under {arch_dir}")
    matched = 0
    for kdp_path in kdps:
        name = kdp_path.relative_to(arch_dir).as_posix()
        try:
            kdp = json.loads(kdp_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            failures.add("no-kdp", f"{name} is unreadable: {type(exc).__name__}: {exc}")
            continue
        if not isinstance(kdp, dict):
            failures.add("no-kdp", f"{name} is not a JSON object")
            continue
        try:
            provenance_sidecar.attach(kdp_path, kdp)
        except HkpPackError as exc:
            failures.add("provenance-sidecar", f"{name}: {exc}")
            continue
        if "arch" in kdp and kdp["arch"] != _kdp_arch(args, name):
            failures.add("arch-field", f"{name} arch is {kdp['arch']!r}")
        for ukd in kdp.get("kernelDescriptors", []):
            if _check_ukd(args, name, ukd, archives, failures):
                matched += 1
    expected = len(args.expected)
    if matched != expected or matched < 1:
        failures.add(
            "ukd-count",
            f"{matched} expected UKDs shipped under {arch_dir}, expected {expected} "
            "(and at least 1)",
        )


def _archive_passes(args):
    """The passes with an expected UKD of kpack output, whose archive must exist."""
    return sorted(
        {e["arch"] for e in args.expected.values() if _output_of(e["kind"]) == "kpack"}
    )


def run(args):
    failures = _Failures()
    if not args.out_root.is_dir():
        failures.add("out-root-missing", "build target hkp_packaging_probes first")
        return failures

    arch_dir = _check_layout(args, failures)
    if arch_dir is None:
        return failures
    archives = {}
    for pass_arch in _archive_passes(args):
        kpack_path = _check_kpack(arch_dir, pass_arch, failures)
        if kpack_path is not None:
            kpack = _load_kpack(args.kpack_python_dir)
            try:
                archives[pass_arch] = kpack.PackedKernelArchive.read(kpack_path)
            except Exception as exc:
                failures.add(
                    "kpack-toc",
                    f"{kpack_path} is unreadable: {type(exc).__name__}: {exc}",
                )
    _check_descriptors(args, arch_dir, archives, failures)
    return failures


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        args.expected = _load_expect(args.expect)
    except _UsageError as exc:
        print(f"{_PREFIX}: error: {exc}", file=sys.stderr)
        return 2
    try:
        failures = run(args)
    except ImportError as exc:
        print(f"{_PREFIX}: error: cannot import rocm_kpack: {exc}", file=sys.stderr)
        return 2
    for line in failures.lines:
        print(line, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

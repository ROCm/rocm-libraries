# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Build the split SDPA backward ops (prep, dkdv, dq) of an arch whose rows belong to two builder families, into one manifest.

gfx942 and gfx90a ship 16-bit rows built by the RDNA family (``_sdpa_bwd_rdna_build``: the MFMA16 layer over the gfx11 kernels) and
fp32 rows built by the CDNA family (``_sdpa_bwd_cdna_build``: native MFMA, 4-byte DMA). Each builder keeps its own record shape (the
RDNA family ships its dBias reduce as a companion object, the CDNA family carries the reduce kernel inside the dQ object, a
``combined`` row); this module partitions the arch's rows by ``family(arch, dtype, kernel)``, has each builder build its own (``build_rows``),
and writes one ``<op>/<arch>/manifest.json`` and ``SOURCE.md`` with the records in table order. R-MIR is the build verdict of both
families' rows: the RDNA-family rows of these arches carry the gate record, and a red verdict stops the run as it does for a CDNA row.
Both families read the nondeterministic list (``tests/data/sdpa_bwd_nondeterministic.csv``) as the gfx950 and gfx11 builds do
(``_sdpa_bwd_cdna_build``, "Nondeterministic builds"): a listed row ships its pin, and its rebuild must be the pin or a recorded variant.
The dQ op also lists the tail-zero companions (``build_tailzero``) beside the dBias reduce companions.
"""

from __future__ import annotations

from pathlib import Path

from . import _flydsl_env as env
from . import _sdpa_bwd_cdna_build as cdna
from . import _sdpa_bwd_nondet as nondet
from . import _sdpa_bwd_rdna_build as rdna
from ._codeobject import GeneratorError
from ._instances import SDPA_BWD_CDNA9_TARGETS, instances_for
from ._manifest import refresh_source_md, write_op_manifest
from ._sdpa_bwd_spec import Spec, family

ARCHES = SDPA_BWD_CDNA9_TARGETS


def _partition_only(only: set[str] | None, r_hashes: set[str], c_hashes: set[str]):
    """(RDNA-family subset, CDNA-family subset) of the Spec hashes @p only, which may name rows of either family; ``None`` is all."""
    if only is None:
        return None, None
    unknown = only - r_hashes - c_hashes
    if unknown:
        raise GeneratorError(f"no row with hash {sorted(unknown)}")
    return only & r_hashes, only & c_hashes


def _check_reference(
    specs, reference: dict[str, dict], reference_dir: Path, cdna_family: bool = False
) -> None:
    """Every Spec of @p specs has its reference row, and its pin object in @p reference_dir is the row's sha256.

    A combined row of the CDNA family (@p cdna_family, ``reduce_symbol_of``) ships its stored combined object and is gated by the main-function
    digest (``cdna.load_pin``); it needs its reference row only, and ``reference_dir`` holds no object for it.
    """
    for spec in specs:
        row = reference.get(spec.hash())
        if row is None or row["spec"] != spec.key():
            raise GeneratorError(f"the reference has no row for {spec.key()}")
        if cdna_family and cdna.reduce_symbol_of(spec):
            continue
        path = reference_dir / f"{spec.hash()}.hsaco"
        if not path.is_file() or rdna.sha256(path.read_bytes()) != row["sha256"]:
            raise GeneratorError(f"{path} is not the pin its reference row names")


def generate(
    op: str,
    arch: str,
    out_root: Path,
    only: set[str] | None = None,
    reference_dir: Path | None = None,
    reference_csv: Path | None = None,
    nondeterministic_csv: Path = nondet.NONDETERMINISTIC_CSV,
    build_tailzero: cdna.CompanionBuilder | None = None,
    stored_dir: Path | None = None,
) -> int:
    """Build every row of @p op on @p arch (or those whose Spec hash is in @p only) into ``<out_root>/<op>/<arch>/``.

    Both families' objects must be their S1 pins in @p reference_dir (rows of the arch's one reference CSV, @p reference_csv): byte
    for byte, except the rows of the nondeterministic list @p nondeterministic_csv (a listed combined row ships the object stored under
    @p stored_dir, ``--stored-dir``). As on a shipped gfx950 op, an unstable CDNA-family build (``cdna._unstable``: a ``dual_object`` dump
    or a combined row whose per-kernel gate is red) is rebuilt in a fresh process, up to ``cdna._ATTEMPTS`` builds. @p build_tailzero supplies the op's tail-zero
    companions (``(arch, op_dir) -> manifest entries``); they have no pin and no reference row. Returns the number of ``error`` rows;
    a row that stops either builder's run (not ``ok``, not its pin or a recorded variant, a red R-MIR verdict, a red per-kernel gate)
    ships nothing.
    """
    refusal = cdna._refuse({op}, arch)
    if refusal or arch not in ARCHES:
        raise GeneratorError(
            refusal or f"{op} is not built for {arch!r} by both families"
        )
    if reference_dir is None:
        raise GeneratorError(
            f"{op} checks its objects against the S1 pins and needs --reference-dir"
        )
    if env.rocm_version() == "unknown":
        raise GeneratorError(
            "the ROCm version is unknown: set ROCM_PATH to the ROCm install the build compiles against"
        )
    reference_dir = reference_dir.resolve()
    csv_path = reference_csv or rdna._REFERENCE_CSVS[arch]
    r_only, c_only = _partition_only(
        only,
        {s.hash() for s in rdna.rows_of(op, arch)[1]},
        {s.hash() for s in cdna.rows_of(op, arch)[1]},
    )
    r_instances, r_specs, companions = rdna.rows_of(op, arch, r_only)
    c_instances, c_specs = cdna.rows_of(op, arch, c_only)
    op_dir = out_root / op / arch
    if not cdna._empty_dir(op_dir):
        raise GeneratorError(f"{op_dir} is not empty")
    reference = cdna.read_reference(csv_path)
    _check_reference((*r_specs, *companions), reference, reference_dir)
    _check_reference(c_specs, reference, reference_dir, cdna_family=True)
    env.assert_flydsl_version()
    env.prepare(arch)
    op_dir.mkdir(parents=True, exist_ok=True)

    try:
        r_records, entries, reasons = rdna.build_rows(
            op,
            arch,
            r_instances,
            r_specs,
            companions,
            reference,
            op_dir,
            reference_dir,
            nondeterministic_csv,
        )
        c_records, c_reasons, outcomes = cdna.build_rows(
            op,
            arch,
            c_instances,
            c_specs,
            op_dir,
            reference_dir,
            csv_path,
            rebuild_unstable=True,
            nondeterministic_csv=nondeterministic_csv,
            stored_dir=stored_dir,
        )
    except GeneratorError:
        for written in op_dir.iterdir():
            written.unlink()
        raise
    reasons += c_reasons
    ok_r = [r for r in r_records if r["build_status"] == "ok"]
    reasons += cdna._stop_reasons(
        ok_r,
        [
            cdna.Pin(
                reference[r["spec_hash"]]["sha256"],
                rmir_scan=reference[r["spec_hash"]].get("rmir_scan") or "",
            )
            for r in ok_r
        ],
    )
    tailzero = []
    if not reasons and build_tailzero is not None:
        try:
            tailzero = build_tailzero(arch, op_dir)
        except GeneratorError as exc:
            reasons.append(" ".join(str(exc).split())[:500])
    if reasons:
        for written in op_dir.iterdir():
            written.unlink()
        raise GeneratorError(
            f"{len(reasons)} row(s) stop the run, nothing is shipped:\n  "
            + "\n  ".join(reasons)
        )
    records = r_records + c_records
    order = {instance.name: i for i, instance in enumerate(instances_for(op, arch))}
    records.sort(key=lambda record: order[record["name"]])
    provenance = env.provenance(arch)
    provenance.pop("aiter_kernels_commit", None)
    write_op_manifest(op_dir, arch, op, provenance, records)
    source_md = cdna._complete_source_md(refresh_source_md(op_dir), op, arch, records)
    if entries or tailzero:
        cdna.write_companions(op_dir, [*entries, *tailzero])
    print(
        f"{op}: {len(records)} ok ({len(r_specs)} rdna-family, {len(c_specs)} cdna-family), "
        f"{len(entries) + len(tailzero)} companion(s); wrote {op_dir}"
    )
    print(f"refreshed {source_md}")
    return sum(record["build_status"] == "error" for record in records)


def compile_one(task: tuple[str]) -> cdna.Outcome:
    """The pin build of one Spec by the builder family that owns it: one attempt, no retry (``tools/sdpa_bwd_pin_build.py``).

    Module-level so a spawned worker can import it. A CDNA-family object comes back with its gates (resources, #1087, R-MIR), built
    standalone: a combined row (dB-storing bias dQ) ships one object with the reduce kernel inside, and its pin is the standalone dQ
    object (``cdna.reduce_symbol_of``). An RDNA-family object of a gated arch comes back gated; on a generic arch its R-MIR is scanned
    (``Outcome.gate["rmir"]``) for the pin evidence.
    """
    (spec_s,) = task
    spec = Spec.parse(spec_s)
    if family(spec.arch, spec.dtype, spec.kernel) == "rdna":
        return rdna._compile_one((spec.key(), "scan"))
    return cdna._compile_one((spec.key(), spec.hash(), "", "", "", "", "standalone"))

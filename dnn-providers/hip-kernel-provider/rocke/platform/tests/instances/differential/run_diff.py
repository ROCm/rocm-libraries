#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_diff.py -- differential harness spine for the dual-backend path.
#
# Drives EVERY tests/parity/<family>_emit.{c,py} pair through a differential
# layer and sha256-compares the C engine against the Python reference across all
# sampled configs of every family. Produces a full-surface parity dashboard
# (JSON + table) and a drift inventory (the families/configs that diverge).
#
#   L3  mode=ll      lower_kernel_to_llvm        (the existing .ll byte-identity)
#   L2  mode=ir      ck.dsl.ir/v1 serialization  (catches builder drift upstream
#                                                  of .ll; needs emitter mode arg)
#   L1  mode=verify  verifier diagnostics        (needs emitter mode arg)
#
# An emitter that does not yet accept a mode arg is exercised in its default
# (.ll) mode for L3 and reported as MODE_UNSUPPORTED for L2/L1 -- so this runner
# already works against the current emitters and gains L2/L1 coverage family by
# family as emitters are extended. MODE_UNSUPPORTED is also the one gate-passing
# status that means "not compared" (see GATE_PASS below), which is admissible
# only because the gating L3/ll lane cannot produce it.
#
# Two levels, and the words are not interchangeable: a VERDICT classifies one
# (family, config); a STATUS rolls a family's verdicts up, and the gate rules on
# statuses. Hence DRIFT_VERDICTS below -- a set of verdicts, none of them named
# DRIFT, that force the family status DRIFT.
#
# Verdict per (family, config):
#   IDENTICAL          both non-empty, sha equal
#   MISMATCH           both non-empty, sha differ              (drift)
#   BOTH_REJECTED      both empty + positive rc, not end-of-range (parity-faithful)
#   ASYMMETRIC         one empty / rc disagree                 (drift)
#   CRASH              either side died on a signal (rc < 0)   (drift)
#   END                both report "unknown config" (range end; stops the family)
#
# Per-family statuses, and which of them the gate lets exit zero, are the
# GATE_PASS / GATE_FAIL maps below; the rule over them is gate_verdict(), unit-
# tested host-only in test_gate_verdict.py next to this file.
#
# Build output goes to /tmp (repo tree is slow NFS). No git operations.

import argparse
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
ROCKE = HERE.parents[2]  # rocKE root (differential -> instances -> tests -> rocKE)
CKC = ROCKE  # engine build root (cmake -S <ROCKE> produces librocke_core.a)
PYROOT = ROCKE / "python"  # holds rocke
LIB_ROOT = (
    ROCKE.parent / "library"
)  # kernels.*, builders.*, dispatch.* top-level packages
PARITY = ROCKE / "tests" / "instances" / "parity"
LIB_PARITY = ROCKE.parent / "library" / "tests" / "parity"
INCLUDE = ROCKE / "cpp" / "include"
TMP = Path(tempfile.gettempdir()) / "rocke_diff"
TMP.mkdir(parents=True, exist_ok=True)
TOOLS = ROCKE / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from _hostcaps import available_cpus  # noqa: E402 -- tools path bootstrap above

# L5 golden anchor: committed per-(mode,family,config) reference shas. This is
# the absolute regression anchor -- any future change that alters an emitted
# .ll / canonical-ir for a blessed config fails --check-golden.
GOLDEN_DIR = HERE / "golden"
GOLDEN_FILE = GOLDEN_DIR / "llvm_gfx_all.json"
PYTHON_BATCH_WORKER = HERE / "python_batch_worker.py"


def golden_key(mode, canonical):
    """Stable key under which a run's shas are stored in the golden file."""
    return "ir_canonical" if (mode == "ir" and canonical) else mode


def collect_ref_shas(results):
    """{family: {str(idx): ref_sha}} for every config that produced output."""
    out = {}
    for r in results:
        fam = {}
        for c in r.get("configs", []):
            if c.get("ref_sha"):
                fam[str(c["idx"])] = c["ref_sha"]
        if fam:
            out[r["family"]] = fam
    return out


MAX_CFG = 256  # default cap on config enumeration per family
FAMILY_MAX_CFG = {"gfx1250_lowering": 512}
TIMEOUT = 120  # seconds per emitter invocation


def config_limit(family: str) -> int:
    """Share enumeration caps so every differential path covers the same configs."""
    return FAMILY_MAX_CFG.get(family, MAX_CFG)


_CANON = None


def _canon(text):
    """Canonicalize ck.dsl.ir/v1 text (stable SSA ids, loc stripped) for
    semantic diff. Lazily imports the Python reference canonicalizer; used on
    BOTH the C and Python emitted IR so a byte mismatch that is only incidental
    SSA-id renumbering shows up as CANON_EQUAL, while real structural drift
    (differing attrs / op order) shows up as STRUCT_DRIFT."""
    global _CANON
    if _CANON is None:
        if str(PYROOT) not in sys.path:
            sys.path.insert(0, str(PYROOT))
        from rocke.core.ir_serialize import canonicalize, parse

        _CANON = (canonicalize, parse)
    canonicalize, parse = _CANON
    # canonicalize takes a KernelDef -> parse the emitted text first. A parse
    # failure here means the text is not valid/round-trippable ck.dsl.ir/v1.
    return canonicalize(parse(text))


def sh(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def archive_build_id(archive: Path) -> str:
    """Build-id stamped into the engine archive (link a 2-line probe that calls
    rocke_build_id). Returns '<unavailable: ...>' if the probe cannot be built so
    this stays purely informational and never fails the run."""
    probe_c = TMP / "_build_id_probe.cpp"
    probe_bin = TMP / "_build_id_probe"
    probe_c.write_text(
        'extern "C" const char* rocke_build_id(void);\n'
        'extern "C" const char* rocke_engine_version(void);\n'
        "#include <cstdio>\n"
        'int main(){printf("%s %s\\n", rocke_build_id(), rocke_engine_version());'
        "return 0;}\n"
    )
    try:
        cc = subprocess.run(
            [
                "c++",
                "-std=c++20",
                str(probe_c),
                str(archive),
                "-lm",
                "-o",
                str(probe_bin),
            ],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
        if cc.returncode != 0:
            return "<unavailable: link failed>"
        run = subprocess.run(
            [str(probe_bin)], capture_output=True, text=True, timeout=30
        )
        return run.stdout.strip() or "<unavailable: empty>"
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {e}>"


def find_families():
    fams = []
    for src_dir in (PARITY, LIB_PARITY):
        if not src_dir.exists():
            continue
        for c in sorted(src_dir.glob("*_emit.c")):
            name = c.name[: -len("_emit.c")]
            py = src_dir / f"{name}_emit.py"
            if py.exists():
                fams.append((name, src_dir))
    return fams


_BUILD_INPUT_DIGESTS: dict[Path, str] = {}
_BUILD_INPUT_DIGESTS_LOCK = threading.Lock()


def _shared_build_input_digest(archive: Path) -> str:
    """Hash immutable shared build inputs once per differential process."""
    archive = archive.resolve()
    with _BUILD_INPUT_DIGESTS_LOCK:
        cached = _BUILD_INPUT_DIGESTS.get(archive)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        digest.update(archive.read_bytes())
        for header in sorted(INCLUDE.rglob("*")):
            if header.is_file():
                digest.update(header.relative_to(INCLUDE).as_posix().encode("utf-8"))
                digest.update(header.read_bytes())
        compiler = shutil.which("c++")
        if compiler:
            compiler_path = Path(compiler)
            digest.update(str(compiler_path.resolve()).encode("utf-8"))
            digest.update(compiler_path.read_bytes())
        result = digest.hexdigest()
        _BUILD_INPUT_DIGESTS[archive] = result
        return result


def _native_emitter_fingerprint(source: Path, archive: Path, command: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(_shared_build_input_digest(archive).encode("ascii"))
    digest.update(source.read_bytes())
    digest.update(json.dumps(command, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def compile_c(name, archive, src_dir=None):
    if src_dir is None:
        src_dir = PARITY
    archive = archive.resolve()
    src = src_dir / f"{name}_emit.c"
    # Keep reusable emitters inside the selected build root so deleting that
    # root also removes every content-addressed generation.
    cache_dir = archive.parent / "rocke_diff_emitters"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_stem = cache_dir / f"{name}_emit_c"
    # The engine archive is C++20; emitters are compiled as C++20 against it.
    # A content fingerprint safely reuses the executable across invocations;
    # all public headers are included conservatively because each emitter's
    # transitive include graph is cheap compared with compiling it again.
    cmd = [
        "c++",
        "-std=c++20",
        "-I",
        str(INCLUDE),
        str(src),
        str(archive),
        "-lm",
        *(["-ldl", "-pthread"] if sys.platform == "linux" else []),
        "-o",
        str(output_stem),
    ]
    fingerprint = _native_emitter_fingerprint(src, archive, cmd)
    out = cache_dir / f"{name}_emit_c_{fingerprint[:16]}"
    if out.is_file():
        return out, ""

    # Atomic publication requires the temporary and final paths to be on the
    # same filesystem. The selected build root may not share one with TMP.
    temporary = cache_dir / f"{out.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    compile_cmd = [*cmd[:-1], str(temporary)]
    try:
        p = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        temporary.unlink(missing_ok=True)
        return None, "native emitter compilation TIMEOUT"
    if p.returncode != 0:
        temporary.unlink(missing_ok=True)
        return None, p.stderr
    try:
        temporary.replace(out)
    except OSError:
        # Another gate may have published the same content-addressed executable
        # while this compiler was running. Its output is equivalent by key.
        temporary.unlink(missing_ok=True)
        if not out.is_file():
            raise
    return out, ""


class Emission(NamedTuple):
    """Reduced emitter result; raw IR is unnecessary after hashing."""

    returncode: int
    has_output: bool
    output_sha: str | None
    error: str
    canonical_sha: str | None = None
    canonical_error: str | None = None


class BatchResult(NamedTuple):
    """Actual per-config emissions plus any producer/protocol failure."""

    emissions: list[Emission]
    error: str | None = None


def _emission(
    returncode: int, output: bytes, error: str, *, canonical: bool = False
) -> Emission:
    output_sha = sh(output) if output else None
    canonical_sha = None
    canonical_error = None
    if canonical and output:
        try:
            text = output.decode("utf-8", "replace")
            canonical_sha = sh(_canon(text).encode("utf-8"))
        except Exception as exc:  # noqa: BLE001 -- recorded in the verdict
            canonical_error = f"{type(exc).__name__}: {exc}"
    return Emission(
        returncode,
        bool(output),
        output_sha,
        error,
        canonical_sha,
        canonical_error,
    )


def _timeout_error(stream, default: str = "TIMEOUT") -> str:
    if not stream:
        return default
    if isinstance(stream, bytes):
        stream = stream.decode("utf-8", "replace")
    lines = stream.strip().splitlines()
    return f"{default} ({lines[-1]})" if lines else default


def _run_native_configs(
    binpath: Path,
    indices: Iterable[int],
    mode: str = "ll",
    canonical: bool = False,
) -> BatchResult:
    """Stream native configs through short-lived, pipe-backed emitters."""
    emissions = []
    for index in indices:
        args = [str(binpath), str(index)]
        if mode != "ll":
            args.append(mode)
        try:
            process = subprocess.run(args, capture_output=True, timeout=TIMEOUT)
            emission = _emission(
                process.returncode,
                process.stdout,
                process.stderr.decode("utf-8", "replace"),
                canonical=canonical,
            )
        except subprocess.TimeoutExpired:
            emission = Emission(124, False, None, f"TIMEOUT in native config {index}")
        emissions.append(emission)
        if (
            _is_end_emission(emission)
            or emission.returncode < 0
            or emission.returncode == 124
        ):
            break
    return BatchResult(emissions)


# Reference-side Python root: defaults to this branch's tree, but --pyroot can
# point it at another tree (e.g. the merge-target rocke-prototype) to measure
# C++(this branch) vs Python(target) drift. SHIM_DIR (optional) is
# prepended so modules the target lacks (ir_serialize/verify) import as stubs.
PY_REF_ROOT = PYROOT
SHIM_DIR = None


def _python_emitter_env(src_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    # The C emitters build their IR natively and so have no Python authoring
    # stack to record; letting location capture through here would give the
    # reference side debug metadata the other side cannot have, and report drift
    # for two kernels that lowered identically. The engines' debug emission is
    # compared where the comparison is meaningful -- one kernel, both lowerers --
    # by tests/core/test_debug_info.py under ROCKE_BACKEND=both.
    env.pop("ROCKE_DEBUG_LOC", None)  # ir.LOC_CAPTURE_ENV
    roots = [str(PY_REF_ROOT), str(LIB_ROOT), str(src_dir), str(PARITY)]
    if SHIM_DIR:
        roots.insert(0, str(SHIM_DIR))
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(roots)) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


def _run_python_batch(
    name: str,
    indices: Iterable[int],
    mode: str = "ll",
    src_dir: Path | None = None,
    canonical: bool = False,
) -> BatchResult:
    """Emit one family's configurations with a deadline for each invocation."""
    if src_dir is None:
        src_dir = PARITY
    indices = list(indices)
    with tempfile.TemporaryDirectory(prefix="rocke-python-batch-") as directory:
        output_dir = Path(directory)
        args = [sys.executable, str(PYTHON_BATCH_WORKER)]
        if canonical:
            args += ["--output-dir", str(output_dir)]
        args += [
            str(src_dir / f"{name}_emit.py"),
            mode,
            *(str(index) for index in indices),
        ]
        # stderr is file-backed so a noisy worker cannot block while the parent
        # waits for its next stdout protocol record.
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as errors:
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=errors,
                text=True,
                env=_python_emitter_env(src_dir),
            )
            assert process.stdout is not None
            records: queue.Queue[str | None] = queue.Queue()

            def read_records() -> None:
                for line in process.stdout:
                    records.put(line)
                records.put(None)

            reader = threading.Thread(target=read_records, daemon=True)
            reader.start()
            emissions = []
            try:
                for expected_index in indices:
                    try:
                        line = records.get(timeout=TIMEOUT)
                    except queue.Empty:
                        process.kill()
                        process.wait()
                        errors.seek(0)
                        timeout = Emission(
                            124,
                            False,
                            None,
                            _timeout_error(
                                errors.read(),
                                f"TIMEOUT in Python config {expected_index}",
                            ),
                        )
                        return BatchResult([*emissions, timeout])
                    if line is None:
                        break
                    try:
                        record = json.loads(line)
                        if record["idx"] != expected_index:
                            raise ValueError(
                                "batch worker returned different config index"
                            )
                        if canonical:
                            output = (output_dir / f"{record['idx']}.out").read_bytes()
                            emission = _emission(
                                record["returncode"],
                                output,
                                record["stderr"],
                                canonical=True,
                            )
                            if emission.output_sha != record["sha256"]:
                                raise ValueError(
                                    "batch worker output hash changed in transit"
                                )
                        else:
                            emission = Emission(
                                record["returncode"],
                                record["has_output"],
                                record["sha256"],
                                record["stderr"],
                            )
                    except (
                        KeyError,
                        OSError,
                        TypeError,
                        ValueError,
                        json.JSONDecodeError,
                    ) as exc:
                        process.kill()
                        process.wait()
                        return BatchResult(
                            emissions, f"invalid batch worker response: {exc}"
                        )
                    emissions.append(emission)
                    if _is_end_emission(emission):
                        break

                try:
                    returncode = process.wait(timeout=TIMEOUT)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    errors.seek(0)
                    return BatchResult(
                        emissions,
                        _timeout_error(errors.read(), "TIMEOUT exiting Python batch"),
                    )
                if returncode != 0:
                    errors.seek(0)
                    error = errors.read() or f"batch worker exited {returncode}"
                    return BatchResult(
                        emissions, f"batch worker exited {returncode}: {error}"
                    )
                return BatchResult(emissions)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdout.close()
                reader.join(timeout=1)


def mode_unsupported(err: str) -> bool:
    e = err.lower()
    return "usage" in e or "unknown mode" in e or "too many arg" in e


def _is_end_emission(emission: Emission) -> bool:
    return (
        emission.returncode != 0
        and not emission.has_output
        and "unknown config" in emission.error.lower()
    )


def classify_emissions(
    c_emission: Emission, p_emission: Emission
) -> tuple[str, tuple[str | None, str | None] | None]:
    """Classify two reduced results without retaining their raw IR."""
    # A timed-out emitter is a harness failure, never a clean spec rejection.
    # Fail the family immediately rather than paying the timeout again for every
    # remaining config and making the gate look hung.
    if c_emission.returncode == 124 or p_emission.returncode == 124:
        return "TIMEOUT", None
    if _is_end_emission(c_emission) and _is_end_emission(p_emission):
        return "END", None
    if c_emission.returncode < 0 or p_emission.returncode < 0:
        return "CRASH", None
    if c_emission.has_output and p_emission.has_output:
        shas = (c_emission.output_sha, p_emission.output_sha)
        verdict = (
            "IDENTICAL"
            if c_emission.output_sha == p_emission.output_sha
            else "MISMATCH"
        )
        return verdict, shas
    if not c_emission.has_output and not p_emission.has_output:
        if c_emission.returncode == 0 and p_emission.returncode == 0:
            return "IDENTICAL", None
        if c_emission.returncode > 0 and p_emission.returncode > 0:
            return "BOTH_REJECTED", None
        return "ASYMMETRIC", None
    return "ASYMMETRIC", (c_emission.output_sha, p_emission.output_sha)


# Per-config verdicts that force the family to DRIFT: the engines disagreed, or
# one of them misbehaved outright (CRASH is either side dying, CANON_ERROR is the
# canonicalizer refusing the text). STRUCT_DRIFT/CANON_ERROR/CANON_EQUAL come
# from the canonical re-triage in the hashed family comparison.
DRIFT_VERDICTS = (
    "MISMATCH",
    "ASYMMETRIC",
    "STRUCT_DRIFT",
    "CANON_ERROR",
    "CRASH",
    "TIMEOUT",
)
TERMINAL_VERDICTS = ("CRASH", "TIMEOUT")


class FamilyStatus(NamedTuple):
    """A family's rolled-up status plus the counts that justify it."""

    status: str
    nbad: int
    ncanon: int


def family_status(configs, range_drift):
    """Roll a family's per-config verdicts up into one status.

    Most-specific first. RANGE_DRIFT outranks the two benign statuses because a
    family whose engines disagree about how many configs exist has not been
    fully compared, however clean the configs it did compare were. The last two
    are the vacuous passes: ALL_REJECTED sampled configs but byte-compared none
    of them, and NO_CONFIGS sampled nothing at all. Neither may read as GREEN.
    """
    nbad = sum(1 for c in configs if c["verdict"] in DRIFT_VERDICTS)
    ncanon = sum(1 for c in configs if c["verdict"] == "CANON_EQUAL")
    nrejected = sum(1 for c in configs if c["verdict"] == "BOTH_REJECTED")
    if nbad:
        status = "DRIFT"
    elif range_drift is not None:
        status = "RANGE_DRIFT"
    elif ncanon:
        status = "CANON_ONLY"  # benign: differs only in SSA-id numbering
    elif not configs:
        status = "NO_CONFIGS"
    elif nrejected == len(configs):
        status = "ALL_REJECTED"
    else:
        status = "GREEN"
    return FamilyStatus(status, nbad, ncanon)


def _config_record(
    idx, verdict, shas, ref_sha, c_emission: Emission, p_emission: Emission
):
    record = {
        "idx": idx,
        "verdict": verdict,
        "shas": shas,
        "ref_sha": ref_sha,
        "c_rc": c_emission.returncode,
        "p_rc": p_emission.returncode,
    }
    if verdict in TERMINAL_VERDICTS and c_emission.error:
        record["c_error"] = c_emission.error
    if verdict in TERMINAL_VERDICTS and p_emission.error:
        record["p_error"] = p_emission.error
    return record


def _batch_complete(emissions: list[Emission], mode: str, limit: int = MAX_CFG) -> bool:
    """Require a hard cap or an explicit terminal record from each producer."""
    if not emissions:
        return False
    tail = emissions[-1]
    if len(emissions) == limit:
        return True
    return (
        _is_end_emission(tail)
        or tail.returncode < 0
        or tail.returncode == 124
        or (mode != "ll" and mode_unsupported(tail.error))
    )


def _run_family_hashed(name, binpath, mode, canonical, src_dir):
    """Run both emitters once when comparison needs hashes, not raw text."""
    limit = config_limit(name)
    emitter_workers = min(2, available_cpus())
    with ThreadPoolExecutor(max_workers=emitter_workers) as emitters:
        c_future = emitters.submit(
            _run_native_configs, binpath, range(limit), mode, canonical
        )
        p_future = emitters.submit(
            _run_python_batch,
            name,
            range(limit),
            mode,
            src_dir=src_dir,
            canonical=canonical,
        )
        c_batch = c_future.result()
        p_batch = p_future.result()

    batch_errors = []
    if c_batch.error:
        batch_errors.append(f"native batch failed: {c_batch.error}")
    if p_batch.error:
        batch_errors.append(f"Python batch failed: {p_batch.error}")
    if batch_errors:
        return [], None, False, "; ".join(batch_errors)

    c_emissions = c_batch.emissions
    p_emissions = p_batch.emissions
    incomplete = []
    if not _batch_complete(c_emissions, mode, limit):
        detail = (
            f": {c_emissions[-1].error}"
            if c_emissions and c_emissions[-1].error
            else ""
        )
        incomplete.append(f"native returned {len(c_emissions)} records{detail}")
    if not _batch_complete(p_emissions, mode, limit):
        detail = (
            f": {p_emissions[-1].error}"
            if p_emissions and p_emissions[-1].error
            else ""
        )
        incomplete.append(f"Python returned {len(p_emissions)} records{detail}")
    if incomplete:
        return [], None, False, "; ".join(incomplete)

    configs = []
    range_drift = None
    for idx, (c_emission, p_emission) in enumerate(zip(c_emissions, p_emissions)):
        if mode != "ll" and (
            mode_unsupported(c_emission.error) or mode_unsupported(p_emission.error)
        ):
            return configs, range_drift, True, None
        c_end = _is_end_emission(c_emission)
        p_end = _is_end_emission(p_emission)
        if c_end and p_end:
            break
        if c_end or p_end:
            range_drift = {"idx": idx, "c_end": c_end, "p_end": p_end}
            break
        verdict, shas = classify_emissions(c_emission, p_emission)
        if canonical and verdict == "MISMATCH":
            if c_emission.canonical_error or p_emission.canonical_error:
                verdict = "CANON_ERROR"
            elif c_emission.canonical_sha == p_emission.canonical_sha:
                verdict = "CANON_EQUAL"
            else:
                verdict = "STRUCT_DRIFT"
        ref_sha = (
            p_emission.canonical_sha or p_emission.output_sha
            if canonical
            else p_emission.output_sha
        )
        configs.append(
            _config_record(
                idx,
                verdict,
                shas,
                ref_sha if p_emission.has_output else None,
                c_emission,
                p_emission,
            )
        )
        if verdict in TERMINAL_VERDICTS:
            break
    return configs, range_drift, False, None


def run_family(name, archive, mode, canonical=False, src_dir=None):
    if src_dir is None:
        src_dir = PARITY
    binpath, cerr = compile_c(name, archive, src_dir=src_dir)
    if binpath is None:
        return {
            "family": name,
            "status": "COMPILE_FAIL",
            "error": cerr.strip().splitlines()[-3:],
            "configs": [],
        }
    configs, range_drift, unsupported, batch_error = _run_family_hashed(
        name, binpath, mode, canonical, src_dir
    )
    if batch_error:
        return {
            "family": name,
            "status": "BATCH_INCOMPLETE",
            "error": batch_error,
            "configs": configs,
        }
    if unsupported:
        return {
            "family": name,
            "status": "MODE_UNSUPPORTED",
            "mode": mode,
            "configs": configs,
        }
    status, nbad, ncanon = family_status(configs, range_drift)
    return {
        "family": name,
        "status": status,
        "n": len(configs),
        "bad": nbad,
        "canon": ncanon,
        "range_drift": range_drift,
        "configs": configs,
    }


# ---- gate verdict ---------------------------------------------------------
# Every status run_family can return is named in exactly one of these two maps,
# plus NO_FAMILIES, which gate_verdict raises for the run as a whole.
# A status in NEITHER map fails the gate (see gate_verdict), so a status added
# later fails loudly until someone classifies it. The gate must never widen by
# omission: a one-status "fail if DRIFT" rule passes every other way the run can
# go wrong, including the ways that have not been invented yet.
#
# Only the keys are consulted. Both maps' values are prose for the reader; the
# failing ones are also printed as the gate's reason.
GATE_PASS = {
    "GREEN": "every sampled config agreed -- identical bytes, or both engines "
    "rejected the spec",
    "CANON_ONLY": "differs only in incidental SSA-id numbering",
    # ll mode -- the gating mode -- never yields this; only the non-gating
    # ir/verify modes can, where an emitter that predates the mode arg is a
    # known coverage gap rather than a parity failure.
    "MODE_UNSUPPORTED": "emitter does not accept this mode yet",
}
GATE_FAIL = {
    "DRIFT": "configs diverge between the engines",
    "RANGE_DRIFT": "the engines disagree about how many configs exist",
    "NO_CONFIGS": "the family enumerated zero configs",
    "ALL_REJECTED": "every sampled config was rejected, so no bytes were compared",
    "COMPILE_FAIL": "the C++ emitter did not build, so nothing was compared",
    "BATCH_INCOMPLETE": "an emitter batch ended without a terminal record",
    "NO_FAMILIES": "the run discovered no families to compare",
}


class GateFailure(NamedTuple):
    """One family the gate refuses to pass, and why."""

    family: str
    status: str
    reason: str


def _gate_detail(r):
    """Evidence for a failing family, from whichever fields its record carries.

    A dashboard read back from ``--json`` is complete, but a caller may hand in
    a partial record. Missing evidence yields no clause: a gate that guesses
    which engine truncated prints a confident falsehood.
    """
    status = r.get("status")
    if status == "DRIFT" and r.get("n") is not None and r.get("bad") is not None:
        return f" ({r['bad']} of {r['n']} configs)"
    if status == "RANGE_DRIFT":
        rd = r.get("range_drift") or {}
        if "c_end" in rd:
            return f" (config {rd.get('idx')}: {'C++' if rd['c_end'] else 'Python'} ended first)"
    return ""


def gate_verdict(results):
    """The pass/fail rule over a dashboard.

    Returns ``(exit_code, failures)``, a list of :class:`GateFailure`. Pure over
    ``results`` -- no archive, no subprocess, no filesystem -- so the rule itself
    is testable on synthetic dashboards without a build.

    An empty dashboard fails. A run that compared no families is the whole-run
    form of NO_CONFIGS, and the gate's GREEN line would be claiming byte
    identity over nothing.
    """
    if not results:
        return 1, [GateFailure("<none>", "NO_FAMILIES", GATE_FAIL["NO_FAMILIES"])]
    failures = []
    for r in results:
        family = r.get("family", "<unnamed>")
        status = r.get("status")
        if status in GATE_FAIL:
            failures.append(
                GateFailure(family, status, GATE_FAIL[status] + _gate_detail(r))
            )
        elif status not in GATE_PASS:
            failures.append(
                GateFailure(
                    family,
                    str(status),
                    "unrecognized status -- classify it in GATE_PASS or GATE_FAIL",
                )
            )
    return (1 if failures else 0), failures


def _print_family_result(result: dict) -> None:
    """Print one completed family without affecting dashboard ordering."""
    tag = result["status"]
    extra = ""
    if result.get("n") is not None:
        extra = (
            f"  configs={result['n']} bad={result['bad']} "
            f"canon={result.get('canon', 0)}"
        )
    print(f"  {tag:16s} {result['family']}{extra}", flush=True)
    if tag == "DRIFT":
        for config in result["configs"]:
            if config["verdict"] in DRIFT_VERDICTS:
                print(
                    f"        cfg[{config['idx']}] {config['verdict']} "
                    f"c_rc={config['c_rc']} p_rc={config['p_rc']}",
                    flush=True,
                )
                for side in ("c_error", "p_error"):
                    if config.get(side):
                        print(f"          {side}: {config[side]}", flush=True)


def _default_family_workers(family_count: int, _mode: str) -> int:
    """Size the outer pool from the child processes active per family."""
    children_per_family = 2
    return min(
        family_count,
        max(1, available_cpus() // children_per_family),
        8,
    )


def _run_families(
    families: list[tuple[str, Path]],
    archive: Path,
    mode: str,
    canonical: bool = False,
    family_workers: int | None = None,
) -> list[dict]:
    """Run independent families concurrently and return discovery order.

    Results complete and print in runtime order so a slow family cannot hide
    progress from faster siblings. The returned dashboard remains in stable
    discovery order for reproducible JSON and golden checks.
    """
    if not families:
        return []
    if family_workers is None:
        # Cap concurrency to avoid excessive captured-IR memory.
        family_workers = _default_family_workers(len(families), mode)
    ordered: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=family_workers) as pool:
        futures = {
            pool.submit(
                run_family,
                name,
                archive,
                mode,
                canonical=canonical,
                src_dir=src_dir,
            ): index
            for index, (name, src_dir) in enumerate(families)
        }
        for future in as_completed(futures):
            result = future.result()
            ordered[futures[future]] = result
            _print_family_result(result)
    return [ordered[index] for index in range(len(families))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ll", choices=["ll", "ir", "verify"])
    ap.add_argument(
        "--canonical",
        action="store_true",
        help="for mode=ir, re-triage byte MISMATCH into CANON_EQUAL "
        "(incidental SSA-id drift) vs STRUCT_DRIFT (real divergence)",
    )
    ap.add_argument(
        "--archive",
        default=str(Path(tempfile.gettempdir()) / "rocke_verify" / "librocke_core.a"),
    )
    ap.add_argument(
        "--only", default="", help="comma-separated family substrings to include"
    )
    ap.add_argument(
        "--exclude",
        default="",
        help="comma-separated family names to leave out (applied after --only)",
    )
    ap.add_argument(
        "--pyroot",
        default="",
        help="override the reference-side Python root (e.g. the merge-target "
        "tree) to measure C++(this branch) vs Python(other tree) drift",
    )
    ap.add_argument(
        "--shim",
        default="",
        help="dir prepended to PYTHONPATH for the reference side (stub modules "
        "the target tree lacks, e.g. ir_serialize/verify); ll-mode only",
    )
    ap.add_argument("--json", default=str(TMP / "dashboard.json"))
    ap.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="families to compare concurrently (default: min(8, available CPUs))",
    )
    ap.add_argument(
        "--record-golden",
        action="store_true",
        help="bless the current reference shas into the committed "
        f"golden file ({GOLDEN_FILE}). Run ONLY from a verified-"
        "good state (gates GREEN). Re-blessing is intentional and "
        "should accompany a reviewed, expected output change.",
    )
    ap.add_argument(
        "--check-golden",
        action="store_true",
        help="re-run and fail on any config whose reference sha "
        "differs from (or is missing in) the golden file. This is the "
        "anchor check only -- its exit code answers 'does output match the "
        "blessed shas', not the parity question the default run gates on.",
    )
    args = ap.parse_args()

    global PY_REF_ROOT, SHIM_DIR
    if args.pyroot:
        PY_REF_ROOT = Path(args.pyroot)
        print(f"[diff] reference Python root overridden -> {PY_REF_ROOT}")
    if args.shim:
        SHIM_DIR = Path(args.shim)

    archive = Path(args.archive)
    if not archive.exists():
        sys.exit(
            f"archive not found: {archive}  (build: cmake -S {CKC} -B /tmp/rocke_verify && cmake --build /tmp/rocke_verify -j)"
        )

    fams = find_families()
    if args.only:
        subs = [s for s in args.only.split(",") if s]
        fams = [(n, d) for n, d in fams if any(s in n for s in subs)]
    if args.exclude:
        excluded = {s for s in args.exclude.split(",") if s}
        kept = [(n, d) for n, d in fams if n not in excluded]
        if fams and not kept:
            # The caller compares these families in another lane.
            print(
                f"[diff] every selected family is excluded ({args.exclude}); nothing to do"
            )
            return 0
        fams = kept

    jobs = args.jobs or _default_family_workers(len(fams), args.mode)
    print(f"mode={args.mode}  families={len(fams)}  jobs={jobs}  archive={archive}")
    print(f"engine build-id: {archive_build_id(archive)}")
    results = _run_families(
        fams,
        archive,
        args.mode,
        canonical=args.canonical,
        family_workers=jobs,
    )

    Path(args.json).write_text(json.dumps(results, indent=2))
    # summary
    by = {}
    for r in results:
        by[r["status"]] = by.get(r["status"], 0) + 1
    print("\n=== SUMMARY (mode={}) ===".format(args.mode))
    for k in sorted(by):
        print(f"  {k:16s} {by[k]}")
    exit_code, failures = gate_verdict(results)
    if failures:
        print("\n=== GATE FAILURES ===")
        for family, status, reason in failures:
            print(f"  {status:16s} {family}: {reason}")
    print(f"\ndashboard: {args.json}")

    # ---- L5 golden anchor -------------------------------------------------
    if args.record_golden or args.check_golden:
        key = golden_key(args.mode, args.canonical)
        cur = collect_ref_shas(results)
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        store = {}
        if GOLDEN_FILE.exists():
            store = json.loads(GOLDEN_FILE.read_text())

    if args.record_golden:
        # The golden records the PYTHON reference sha per config, which is the
        # source of truth regardless of whether the C engine currently agrees.
        # A C-vs-Python DRIFT family is reported as a caveat (e.g. the known
        # pre-existing gfx950_attention_tiled_2d_fastkv_regp ir drift) but does
        # not invalidate the reference shas being blessed.
        if failures:
            print(
                "\nNOTE: recording golden from a run the gate would fail: "
                + ", ".join(f"{f.family} ({f.status})" for f in failures)
                + "\n  The golden anchors the Python reference shas, which "
                "remain well-defined. But a family that ended early or compared "
                "nothing blesses FEWER configs than a clean run would, and the "
                "missing ones return later as NEW (unblessed), not as MISSING."
            )
        store[key] = cur
        nblessed = sum(len(v) for v in cur.values())
        GOLDEN_FILE.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n")
        print(
            f"\nrecorded golden[{key}]: {len(cur)} families, {nblessed} configs "
            f"-> {GOLDEN_FILE}"
        )
        return 0

    if args.check_golden:
        ref = store.get(key)
        if not ref:
            print(
                f"\nNO GOLDEN for key '{key}'. Bless first: "
                f"run_diff.py --mode {args.mode}"
                f"{' --canonical' if args.canonical else ''} --record-golden"
            )
            return 1
        mism, missing, extra = [], [], []
        for fam, cfgs in ref.items():
            cur_fam = cur.get(fam, {})
            for idx, sha in cfgs.items():
                got = cur_fam.get(idx)
                if got is None:
                    missing.append((fam, idx))
                elif got != sha:
                    mism.append((fam, idx))
        for fam, cfgs in cur.items():
            for idx in cfgs:
                if idx not in ref.get(fam, {}):
                    extra.append((fam, idx))
        print(f"\n=== GOLDEN CHECK (key={key}) ===")
        print(f"  blessed configs : {sum(len(v) for v in ref.values())}")
        print(f"  UNBLESSED CHANGE: {len(mism)}")
        print(f"  MISSING (gone)  : {len(missing)}")
        print(f"  NEW (unblessed) : {len(extra)}")
        for fam, idx in (mism + missing)[:30]:
            print(f"    {fam}[{idx}]")
        if mism or missing:
            print(
                "\nGOLDEN MISMATCH: emitted output changed vs the committed "
                "anchor. If this change is expected & reviewed, re-bless with "
                "--record-golden; otherwise you introduced a regression."
            )
            return 1
        if extra:
            print("\nNew configs not in golden (informational; re-bless to add).")
        print("GOLDEN OK")
        return 0

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

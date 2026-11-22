# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Time and size one complete TensileLite generation job.

A job is what ``TensileCreateLibrary`` does for a directory of library logic
files and one architecture: derive the solutions, emit the main kernels and
their helpers, write the sources, assemble, compile the helpers, link, and
write the library metadata. This driver calls the production entry points of
``TensileCreateLibrary.Run`` in the order ``run()`` does (the emit entry point
is the one ``codegen_harness.py`` uses), times each stage with
``time.perf_counter`` in one warm process, and measures the artifacts it built.
All times are per job; a logic directory with one solution gives per-solution
times.

Usage::

    python -m tensilelite.Tests.unit.characterization._codegen.measure_generation \\
        --source-root <dir> --rocisa <dir> --logic <dir> --arch gfx950|gfx942 \\
        --out <dir> --repeat N --json <file>

    python -m tensilelite.Tests.unit.characterization._codegen.measure_generation \\
        --parse-timing-log <file> --source-root <dir> --rocisa <dir> \\
        --logic <dir> --out <TensileCreateLibrary output dir> --json <file>

``--source-root`` is a monorepo checkout (or an archive of one); the measured
TensileLite package is ``<source-root>/projects/hipblaslt/tensilelite/tensilelite``.
``--rocisa`` is the directory that holds the ``rocisa`` package, a rocisa
build of that same tree. The launcher starts a separate interpreter whose
``PYTHONPATH`` is exactly ``<rocisa>:<source-root>/projects/hipblaslt/tensilelite``,
so the tree being measured never shares an interpreter with the tree the
launcher came from. That interpreter refuses to measure (writes the JSON with
``import_identity.ok`` false, prints the violations, exits 1) when

* a loaded ``tensilelite``/``Tensile`` module does not resolve under the
  source root, or a ``rocisa`` module not under ``--rocisa``; or
* the rocisa build was not built from the source root: its
  ``rocisa/_build_info.py`` ``SOURCE_ROOT`` is missing or outside
  ``--source-root``.

The check runs after setup and again after the last repetition.

``--out`` must be absent or empty; the driver refuses a used one, so no
leftover of an earlier run is measured. Repetition ``k`` writes
``<out>/rep<k>/`` with the ``TensileCreateLibrary`` output layout
(``library/``, ``build_tmp/``); assembly sources are kept.

The job uses the options ``hipblaslt_codegen.cmake`` passes for a
``TENSILELITE_ENABLE_ASM_COMMENTS=OFF``, ``TENSILELITE_NO_COMPRESS=ON`` build:
``--disable-asm-comments --no-compress``, lazy library loading, msgpack
libraries, default code object version and toolchain, and ``--jobs=1``. Its
``library/`` output is byte-identical to that ``TensileCreateLibrary`` run.

Stages (seconds, or ``"n/a"`` where a stage does not apply):

* ``process_setup``: imports, toolchain validation, ISA capability probe,
  global parameters, from the worker's first statement. Repetition 1 only;
  later repetitions reuse the process.
* ``derive``: find and parse the logic files into solutions, kernels and the
  helper-kernel list (``generateLogicDataAndSolutions``,
  ``generateKernelObjectsFromSolutions``, ``generateKernelHelperObjects``).
* ``codegen``: emit main-kernel assembly (``processKernelSource``) and helper
  HIP source.
* ``write``: write ``.s`` files, static headers and ``Kernels.cpp/.h``.
* ``assemble``: assemble each ``.s`` into an object.
* ``helper_compile``: ``buildSourceCodeObjectFiles`` (compile and unbundle the
  helpers, or restore them from the helper-kernel cache; see ``helper_cache``).
* ``link``: ``buildAssemblyCodeObjectFiles`` (link the main ``.co``).
* ``metadata``: pass kernel info to the library and write the master, lazy and
  mapping library files.

``compile_s`` is the complete compilation time: the sum of the
seven stages from ``derive`` to ``metadata``, without ``process_setup``.
``wall_s`` is measured on its own clock around the whole repetition, from
before ``<out>/rep<k>`` is created until the metadata is written, so it also
holds the gaps between stages. ``job_wall_s`` brackets the whole worker,
from its first statement until the last repetition ends.

``helper_cache`` is ``miss`` when the repetition compiled ``Kernels.cpp``
(its own, fresh ``code_object_tmp/Kernels.o`` exists) and ``hit`` when the
production cache restored the helpers instead. Repetitions 2..N reuse the
cache repetition 1 filled; set ``TENSILE_DISABLE_HELPER_CACHE=1`` to make
every repetition compile.

JSON schema (``schema_version`` 3)::

    {
      "schema_version": 3,
      "driver": {"file": str, "python": str, "executable": str},
      "source_root": str, "tensilelite_root": str, "rocisa": str,
      "logic": str, "arch": str, "out": str, "repeat": int,
      "tensile_create_library_argv": [str],   # the options the job mirrors
      "helper_cache_env": {"TENSILE_HELPER_CACHE_DIR": str|null,
                           "TENSILE_DISABLE_HELPER_CACHE": str|null},
      "host": {"loadavg": [float, float, float], "cpu_affinity": [int]},
      "import_identity": {
        "ok": bool,
        "tensilelite_root": str, "rocisa_root": str,   # resolved paths
        "rocisa_build_source_root": str|null,           # rocisa._build_info
        "modules": {module_name: resolved_file},
        "violations": [{"module": str, "file": str, "expected_root": str}]
      },
      "fingerprint": {                               # provenance of the result
        "driver": {"path": str, "sha256": str},
        "source": {"scope": str,                     # <source-root>/projects/hipblaslt
                   "git_head": str|null,             # when --source-root is a git top level:
                   "git_diff_sha256": str|null,      #   diff vs HEAD + untracked names and bytes
                   "content_sha256": str|null},      # otherwise: every file, build outputs excluded
        "rocisa": {"package": str, "source_root": str|null,
                   "stinkytofu_source_root": str|null,
                   "files": {relpath: sha256},       # every file of the rocisa package
                   "so_path": str|null, "so_sha256": str|null,
                   "loaded_modules": {module: sha256}},
        "toolchain": {"cxx_compiler": str, "cxx_compiler_version": str,
                      "compiler_version": str, "rocm_version": str,
                      "offload_bundler": str, "python": str, "platform": str},
        "inputs": [{"path": str, "sha256": str}],    # logic files
        "tensilelite": {"package": str,
                        "content_sha256": str},      # package files but Tests/
        "artifacts": [{"path": str, "sha256": str}], # repetition 1 library/
        "solution_index_map": {index: {"solution": str, "kernel": str,
                                       "logic_file": str, "logic_index": int,
                                       "code_object": str|null}}
      },
      "logic_files": [{"path": str, "sha256": str}],
      "kernels": [str],          # main kernel symbols
      "helpers": [str],          # helper kernel names, Kernels.cpp order
      "job_wall_s": float,
      "repetitions": [{
        "index": int, "dir": str,
        "stages": {stage: float|"n/a"},              # the eight stages above
        "process_setup_s": float|"n/a",
        "compile_s": float, "wall_s": float,
        "helper_cache": "hit"|"miss"
      }],
      "first": {"stages", "process_setup_s", "compile_s", "wall_s",
                "helper_cache"},                     # repetition 1
      "rest": null | {                               # repetitions 2..N
        "n": int,
        "stages": {stage: "n/a" | {"median", "q1", "q3", "min", "max"}},
        "compile_s": {"median", "q1", "q3", "min", "max"},
        "wall_s": {"median", "q1", "q3", "min", "max"},
        "helper_cache": [str]
      },
      "artifacts": {                                 # from repetition 1
        "code_objects": [{                           # as the build steps returned them
          "path": str, "kind": "main"|"helper",
          "co_bytes": int, "text_bytes": int, "sha256": str,
          "kernels": [{"name": str, "sgpr": int, "vgpr": int, "lds": int,
                       "kernarg": int,                 # from AMDGPU metadata
                       "agpr": int|null,
                       "agpr_source": "metadata"|"assembly"|null}]
        }],
        "assembly": [{"path": str, "s_lines": int}], # comment-free, non-blank
        "library_files": [{"path": str, "bytes": int, "sha256": str}],
        "totals": {"text_bytes": int, "co_bytes": int, "s_lines": int},
        "identical_across_repetitions": bool
      }
    }

TensileLite assembly kernels carry no ``.agpr_count`` in their metadata; for
them ``agpr`` is ``.amdhsa_next_free_vgpr - .amdhsa_accum_offset`` from the
kernel descriptor in their ``.s`` (``agpr_source: "assembly"``). A code object
without a ``.text`` section reports ``text_bytes`` 0.

Quartiles use ``statistics.quantiles(method="inclusive")``. The helper code
object embeds clang's ``__hip_cuid_*`` symbol, a hash of the ``Kernels.cpp``
path, so helpers compiled in different directories differ in a few bytes:
``identical_across_repetitions`` holds when later repetitions restore the
helpers from the cache, and a byte comparison with ``TensileCreateLibrary``
needs its output path equal to ``<out>/rep1``.

``--parse-timing-log`` reads the output of one cold ``TensileCreateLibrary``
run made with ``TENSILE_PRINT_TIMING=ON`` (one run per file; concatenated
logs are not supported). ``--source-root``, ``--rocisa``, ``--logic`` and
``--out`` name the tree, rocisa build, logic directory and output directory
of that run. They are fingerprinted, not imported, and checked as in
measurement mode: a missing tree, a rocisa without ``_build_info.py`` or
built from another tree, a ``--logic`` with no ``*.yaml`` logic file, or an
``--out`` whose ``library/`` is missing or empty is refused (exit 1). The
fingerprint is taken at parse time, not at run time, so parse
right after the run, before the tree, rocisa or output change. It writes::

    {
      "decorated_s": {function: [seconds]},  # "<function> took <s> seconds"
      "phases_s": {"load_yaml", "generate_kernels", "pass_kernel_info",
                   "write_master_libraries", "helper_build", "total"},
      "timing_records_ms": {category: [ms]}, # "TIMING:<category>:<ms>"
      "helper_cache": "hit"|"miss"|null,
      "kernels_processed": int, "kernel_helper_objs": int,
      "fingerprint": {"driver", "source", "rocisa",  # as above, without loaded_modules
                      "inputs": [{"path", "sha256"}], # the logic files, then the log
                      "artifacts": [{"path", "sha256"}],  # <out>/library/
                      "solution_index_map": {...},    # as above, read from <out>/library/
                      "toolchain": {"components": [{"path", "version", "rocm_version"}],
                                    "cxx_compiler": str|null,
                                    "cxx_compiler_version": str|null,
                                    "python": str, "platform": str}}
    }

The toolchain components are the ``Component path`` lines of the logged run.
The library files do not record the logic file, so in the cold index map
``logic_file`` is set only when ``--logic`` holds exactly one logic file.
Library files must be msgpack (``.dat.zlib``) or YAML. Keys whose line is
absent are omitted (``helper_cache`` is null). In both modes, a git source
root whose ``projects/hipblaslt`` cannot be diffed against HEAD is an error.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

STAGES = (
    "process_setup",
    "derive",
    "codegen",
    "write",
    "assemble",
    "helper_compile",
    "link",
    "metadata",
)
NA = "n/a"
SCHEMA_VERSION = 3
_TL_SUBDIR = Path("projects", "hipblaslt", "tensilelite")
_TL_PACKAGES = ("tensilelite", "Tensile", "rocisa_stinkytofu_adaptor")
_ROCISA_PACKAGES = ("rocisa",)


def _argParser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--rocisa", type=Path)
    parser.add_argument("--logic", type=Path)
    parser.add_argument("--arch", choices=("gfx950", "gfx942"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--parse-timing-log", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _writeJson(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- cold path: TENSILE_PRINT_TIMING output ---------------------------------

_NUMBER = r"([0-9]+(?:\.[0-9]*)?(?:[eE][-+]?[0-9]+)?)"
_PHASE_LINES = {
    "load_yaml": re.compile(r"^Time to load yaml files \(s\): " + _NUMBER),
    "generate_kernels": re.compile(r"^Time to generate kernels \(s\): " + _NUMBER),
    "pass_kernel_info": re.compile(r"^Time to pass kernel info to library \(s\): " + _NUMBER),
    "write_master_libraries": re.compile(r"^Time to write master solution libraries \(s\): " + _NUMBER),
    "helper_build": re.compile(r"^buildSourceCodeObjectFile time \(s\): " + _NUMBER),
    "total": re.compile(r"^Total time \(s\): " + _NUMBER),
}
_DECORATED = re.compile(r"^(\w+) took " + _NUMBER + r" seconds$")
_TIMING_RECORD = re.compile(r"^TIMING:([^:]+):" + _NUMBER + r"$")
_KERNELS_PROCESSED = re.compile(r"^Total kernels processed: (\d+)$")
_HELPER_OBJS = re.compile(r"^KernelHelperObjs: (\d+)$")


def parse_print_timing(text):
    """Parse ``TensileCreateLibrary`` output produced with TENSILE_PRINT_TIMING=ON."""
    decorated, phases, records = {}, {}, {}
    helperCache, counts = None, {}
    for raw in text.splitlines():
        line = raw.strip()
        m = _DECORATED.match(line)
        if m:
            decorated.setdefault(m.group(1), []).append(float(m.group(2)))
            continue
        m = _TIMING_RECORD.match(line)
        if m:
            records.setdefault(m.group(1), []).append(float(m.group(2)))
            continue
        for name, pattern in _PHASE_LINES.items():
            m = pattern.match(line)
            if m:
                phases[name] = float(m.group(1))
                if name == "helper_build":
                    helperCache = "hit" if line.endswith("[cache hit]") else "miss"
                break
        m = _KERNELS_PROCESSED.match(line)
        if m:
            counts["kernels_processed"] = int(m.group(1))
        m = _HELPER_OBJS.match(line)
        if m:
            counts["kernel_helper_objs"] = int(m.group(1))
    return {
        "decorated_s": decorated,
        "phases_s": phases,
        "timing_records_ms": records,
        "helper_cache": helperCache,
        **counts,
    }


# --- artifact measurement ---------------------------------------------------

_SHT_NOTE = 7
_NT_AMDGPU_METADATA = 32


def _elfSections(data):
    """Return {name: (type, offset, size)} for a little-endian ELF64 file."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not a little-endian ELF64 file")
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    headers = [struct.unpack_from("<IIQQQQ", data, shoff + i * shentsize) for i in range(shnum)]
    strOffset = headers[shstrndx][4]
    sections = {}
    for nameOffset, shType, _flags, _addr, offset, size in headers:
        end = data.index(b"\0", strOffset + nameOffset)
        sections[data[strOffset + nameOffset:end].decode()] = (shType, offset, size)
    return sections


def _amdgpuKernels(data, sections):
    import msgpack

    kernels = []
    for shType, offset, size in sections.values():
        if shType != _SHT_NOTE:
            continue
        pos, end = offset, offset + size
        while pos + 12 <= end:
            namesz, descsz, noteType = struct.unpack_from("<III", data, pos)
            name = data[pos + 12:pos + 12 + namesz].rstrip(b"\0")
            desc = pos + 12 + (namesz + 3) // 4 * 4
            if name == b"AMDGPU" and noteType == _NT_AMDGPU_METADATA:
                # Custom kernels add a ``custom.config`` map whose assertion maps
                # have integer keys (``AssertSizeEqual: { 1: 1, 2: 1 }``).
                meta = msgpack.unpackb(data[desc:desc + descsz], raw=False, strict_map_key=False)
                for k in meta.get("amdhsa.kernels", []):
                    kernels.append({
                        "name": k[".name"],
                        "sgpr": k[".sgpr_count"],
                        "vgpr": k[".vgpr_count"],
                        "agpr": k.get(".agpr_count"),
                        "agpr_source": "metadata" if ".agpr_count" in k else None,
                        "lds": k[".group_segment_fixed_size"],
                        "kernarg": k[".kernarg_segment_size"],
                    })
            pos = desc + (descsz + 3) // 4 * 4
    return kernels


def measureCodeObject(path, kind):
    data = Path(path).read_bytes()
    sections = _elfSections(data)
    return {
        "path": str(path),
        "kind": kind,
        "co_bytes": len(data),
        "text_bytes": sections[".text"][2] if ".text" in sections else 0,
        "sha256": hashlib.sha256(data).hexdigest(),
        "kernels": _amdgpuKernels(data, sections),
    }


_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def commentFreeLines(text):
    """Non-blank lines of assembly once ``/* */`` and ``//`` comments are removed."""
    text = _BLOCK_COMMENT.sub("", text)
    return sum(1 for line in text.splitlines() if line.split("//", 1)[0].strip())


_DESCRIPTOR = re.compile(
    r"^\s*\.amdhsa_kernel\s+(\S+)(.*?)^\s*\.end_amdhsa_kernel", re.S | re.M)


def descriptorAgprs(text):
    """{kernel: AGPR count} from ``.amdhsa_next_free_vgpr - .amdhsa_accum_offset``.

    Assembly kernels carry no ``.agpr_count`` in their metadata; their kernel
    descriptor places the AGPRs after ``accum_offset`` in the unified register file.
    """
    agprs = {}
    for name, body in _DESCRIPTOR.findall(text):
        nextFree = re.search(r"\.amdhsa_next_free_vgpr\s+(\d+)", body)
        accum = re.search(r"\.amdhsa_accum_offset\s+(\d+)", body)
        if nextFree and accum:
            agprs[name] = int(nextFree.group(1)) - int(accum.group(1))
    return agprs


def _summary(values):
    if len(values) == 1:
        q1 = q3 = values[0]
    else:
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return {"median": statistics.median(values), "q1": q1, "q3": q3,
            "min": min(values), "max": max(values)}


def _rest(repetitions):
    rest = repetitions[1:]
    if not rest:
        return None
    stages = {}
    for stage in STAGES:
        values = [r["stages"][stage] for r in rest if r["stages"][stage] != NA]
        stages[stage] = _summary(values) if values else NA
    return {"n": len(rest), "stages": stages,
            "compile_s": _summary([r["compile_s"] for r in rest]),
            "wall_s": _summary([r["wall_s"] for r in rest]),
            "helper_cache": [r["helper_cache"] for r in rest]}


def _libraryFiles(repDir):
    root = Path(repDir) / "library"
    return [{"path": str(p), "bytes": p.stat().st_size, "sha256": _sha256(p)}
            for p in sorted(root.rglob("*")) if p.is_file()]


def _artifacts(repetitions, facts):
    """Measure the artifacts repetition 1 built; ``facts`` is what its steps returned."""
    first = Path(repetitions[0]["dir"])
    codeObjects = sorted(
        [measureCodeObject(p, "main") for p in facts["main_code_objects"]]
        + [measureCodeObject(p, "helper") for p in facts["helper_code_objects"]],
        key=lambda c: c["path"])
    assembly, agprs = [], {}
    for p in facts["assembly"]:
        text = Path(p).read_text()
        assembly.append({"path": str(p), "s_lines": commentFreeLines(text)})
        agprs.update(descriptorAgprs(text))
    for kernel in (k for c in codeObjects for k in c["kernels"]):
        if kernel["agpr"] is None and kernel["name"] in agprs:
            kernel["agpr"], kernel["agpr_source"] = agprs[kernel["name"]], "assembly"
    libraryFiles = _libraryFiles(first)
    rel = lambda files, rep: [(Path(f["path"]).relative_to(rep), f["sha256"]) for f in files]
    identical = all(
        rel(_libraryFiles(r["dir"]), Path(r["dir"])) == rel(libraryFiles, first)
        for r in repetitions[1:]
    )
    return {
        "code_objects": codeObjects,
        "assembly": assembly,
        "library_files": libraryFiles,
        "totals": {
            "text_bytes": sum(c["text_bytes"] for c in codeObjects),
            "co_bytes": sum(c["co_bytes"] for c in codeObjects),
            "s_lines": sum(a["s_lines"] for a in assembly),
        },
        "identical_across_repetitions": identical,
    }


# --- import identity and fingerprint -----------------------------------------


def checkImportIdentity(modules, tlRoot, rocisaRoot):
    """Map every loaded TensileLite/rocisa module to its file; list strays."""
    tlRoot, rocisaRoot = Path(os.path.realpath(tlRoot)), Path(os.path.realpath(rocisaRoot))
    found, violations = {}, []
    for name, module in sorted(modules.items()):
        top = name.split(".")[0]
        if top in _TL_PACKAGES:
            root = tlRoot
        elif top in _ROCISA_PACKAGES:
            root = rocisaRoot
        else:
            continue
        files = [getattr(module, "__file__", None)] + list(getattr(module, "__path__", None) or [])
        files = [f for f in files if f]
        if not files:
            continue
        resolved = [Path(os.path.realpath(f)) for f in files]
        found[name] = str(resolved[0])
        for f in resolved:
            if not f.is_relative_to(root):
                violations.append({"module": name, "file": str(f), "expected_root": str(root)})
    return found, violations


def _identity(sourceRoot, tlRoot, rocisaRoot):
    modules, violations = checkImportIdentity(sys.modules, tlRoot, rocisaRoot)
    buildInfo = sys.modules.get("rocisa._build_info")
    builtFrom = getattr(buildInfo, "SOURCE_ROOT", None) or None
    # rocisa generates code too: a rocisa built from another tree mixes trees.
    if builtFrom is None or not Path(os.path.realpath(builtFrom)).is_relative_to(
            Path(os.path.realpath(sourceRoot))):
        violations.append({"module": "rocisa._build_info", "file": str(builtFrom),
                           "expected_root": os.path.realpath(sourceRoot)})
    return {
        "ok": not violations,
        "tensilelite_root": os.path.realpath(tlRoot),
        "rocisa_root": os.path.realpath(rocisaRoot),
        "rocisa_build_source_root": builtFrom,
        "modules": modules,
        "violations": violations,
    }


def _reportViolations(identity, rocisaRoot):
    for v in identity["violations"]:
        if v["module"] == "rocisa._build_info":
            sys.stderr.write(
                f"error: the rocisa at {rocisaRoot} was built from {v['file']}, not from "
                f"--source-root {v['expected_root']}; refusing to mix trees\n")
        else:
            sys.stderr.write(f"error: module {v['module']} loaded from {v['file']}, "
                             f"outside {v['expected_root']}\n")


_SKIPPED_DIRS = {"__pycache__", ".git", "build", "build_tmp"}


def _contentSha256(root, skipTop=()):
    """sha256 over the relative path and bytes of every file under ``root``.

    Build outputs and caches (``_SKIPPED_DIRS``, ``*.egg-info``) and the
    top-level names in ``skipTop`` are left out; a symbolic link counts as
    its target string, never followed.
    """
    root = Path(root)
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        keep, links = [], []
        for d in dirnames:
            if d in _SKIPPED_DIRS or d.endswith(".egg-info") or (here == root and d in skipTop):
                continue
            (links if (here / d).is_symlink() else keep).append(d)
        dirnames[:] = sorted(keep)
        for name in sorted(filenames + links):
            path = here / name
            h.update(str(path.relative_to(root)).encode() + b"\0")
            h.update(b"->" + os.readlink(path).encode() if path.is_symlink() else path.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def _git(root, *args):
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    return proc.stdout if proc.returncode == 0 else None


def _sourceFingerprint(sourceRoot):
    """Identity of ``<sourceRoot>/projects/hipblaslt``, the tree that produced a result.

    When ``sourceRoot`` is itself a git top level: its HEAD plus a hash of the
    uncommitted state of ``projects/hipblaslt`` (diff against HEAD, and the
    names and bytes of untracked files). Otherwise (an archive such as an
    exported source tree, even inside another repository): a content hash of the tree.
    """
    scope = Path(sourceRoot) / _TL_SUBDIR.parent
    fp = {"scope": str(scope), "git_head": None, "git_diff_sha256": None, "content_sha256": None}
    top = _git(sourceRoot, "rev-parse", "--show-toplevel")
    if top and os.path.realpath(top.decode().strip()) == os.path.realpath(sourceRoot):
        rel = str(_TL_SUBDIR.parent)
        diff = _git(sourceRoot, "diff", "HEAD", "--binary", "--", rel)
        untracked = _git(sourceRoot, "ls-files", "--others", "--exclude-standard", "-z", "--", rel)
        head = _git(sourceRoot, "rev-parse", "HEAD")
        if diff is None or untracked is None or head is None:
            # Hashing nothing would make an unreadable tree look clean.
            sys.exit(f"error: git cannot diff {scope} against HEAD; refusing to fingerprint it")
        h = hashlib.sha256(diff)
        for name in sorted(n for n in untracked.split(b"\0") if n):
            path = Path(sourceRoot, name.decode())
            h.update(b"\0" + name + b"\0")
            h.update(b"->" + os.readlink(path).encode() if path.is_symlink() else path.read_bytes())
        fp["git_head"] = head.decode().strip()
        fp["git_diff_sha256"] = h.hexdigest()
    else:
        fp["content_sha256"] = _contentSha256(scope)
    return fp


def _rocisaFingerprint(rocisaRoot):
    """Identity of the rocisa package under ``rocisaRoot``, read without importing it."""
    package = Path(os.path.realpath(Path(rocisaRoot) / "rocisa"))
    info = package / "_build_info.py"
    buildInfo = {}
    if info.is_file():
        import runpy
        buildInfo = runpy.run_path(str(info))
    files = {str(p.relative_to(package)): _sha256(p) for p in sorted(package.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts}
    sos = sorted(p for p in package.glob("_rocisa*.so"))
    return {"package": str(package),
            "source_root": buildInfo.get("SOURCE_ROOT"),
            "stinkytofu_source_root": buildInfo.get("STINKYTOFU_SOURCE_ROOT"),
            "files": files,
            "so_path": str(sos[0]) if sos else None,
            "so_sha256": files.get(sos[0].name) if sos else None}


def _compilerVersion(cxx):
    version = subprocess.run([str(cxx), "--version"], capture_output=True, text=True).stdout
    return version.splitlines()[0] if version else ""


def _envelope(sourceRoot, rocisaRoot, inputs, toolchain):
    """The fingerprint every result file carries."""
    import platform

    return {
        "driver": {"path": str(Path(__file__).resolve()), "sha256": _sha256(__file__)},
        "source": _sourceFingerprint(sourceRoot),
        "rocisa": _rocisaFingerprint(rocisaRoot),
        "toolchain": {**toolchain, "python": sys.version, "platform": platform.platform()},
        "inputs": inputs,
    }


def _fingerprint(job, sourceRoot, rocisaRoot, package, logicFiles, libraryFiles, solutionIndexMap):
    compiler = job.src.compiler
    ver = lambda v: f"{v.major}.{v.minor}.{v.patch}"
    cxx = str(job.asm.assembler.path)
    fp = _envelope(sourceRoot, rocisaRoot, logicFiles, {
        "cxx_compiler": cxx,
        "cxx_compiler_version": _compilerVersion(cxx),
        "compiler_version": ver(compiler.version),
        "rocm_version": ver(compiler.rocm_version),
        "offload_bundler": str(job.asm.bundler.path)})
    # The rocisa modules this process actually loaded, by content.
    fp["rocisa"]["loaded_modules"] = {
        name: _sha256(m.__file__) for name, m in sorted(sys.modules.items())
        if name.split(".")[0] in _ROCISA_PACKAGES and getattr(m, "__file__", None)}
    fp["tensilelite"] = {"package": str(package),
                         "content_sha256": _contentSha256(package, skipTop=("Tests",))}
    fp["artifacts"] = [{"path": f["path"], "sha256": f["sha256"]} for f in libraryFiles]
    fp["solution_index_map"] = solutionIndexMap
    return fp


_COMPONENT = re.compile(r"^ROCm (\S+) Component path: (\S+) version: (\S+)$")


def _checkTrees(sourceRoot, rocisaRoot):
    """Refuse a source root, rocisa build or pairing that cannot be what produced a result."""
    if not (Path(sourceRoot) / _TL_SUBDIR / "tensilelite" / "__init__.py").is_file():
        sys.exit(f"error: --source-root: no {_TL_SUBDIR}/tensilelite package under {sourceRoot}")
    info = Path(rocisaRoot) / "rocisa" / "_build_info.py"
    if not (Path(rocisaRoot) / "rocisa" / "__init__.py").is_file() or not info.is_file():
        sys.exit(f"error: --rocisa: no source-built rocisa package (with _build_info.py) under {rocisaRoot}")
    import runpy
    builtFrom = runpy.run_path(str(info)).get("SOURCE_ROOT")
    if not builtFrom or not Path(os.path.realpath(builtFrom)).is_relative_to(
            Path(os.path.realpath(sourceRoot))):
        sys.exit(f"error: the rocisa at {rocisaRoot} was built from {builtFrom}, not from "
                 f"--source-root {os.path.realpath(sourceRoot)}; refusing to mix trees")


def _readLibrary(path):
    """A library file TensileCreateLibrary wrote, decoded; None for other files."""
    name = path.name
    if name.endswith(".dat.zlib"):
        import msgpack, zlib
        return name[:-len(".dat.zlib")], msgpack.unpackb(zlib.decompress(path.read_bytes()),
                                                        raw=False, strict_map_key=False)
    if name.endswith(".yaml"):
        import yaml
        return name[:-len(".yaml")], yaml.safe_load(path.read_text())
    return None


def _libraryIndexMap(library, logicFiles):
    """The build-local solution index map, read back from a TensileCreateLibrary library/.

    Same schema as the measured map. Solutions of a lazy library name it as
    their code object; those of the master have none. The library files do
    not record the logic file, so ``logic_file`` is set only when the job had
    exactly one.
    """
    indexMap = {}
    for path in sorted(Path(library).rglob("*")):
        decoded = _readLibrary(path) if path.is_file() and "_Mapping" not in path.name else None
        if decoded is None or not isinstance(decoded[1], dict):
            continue
        stem, data = decoded
        arch = path.parent.name
        master = stem in (f"TensileLibrary_lazy_{arch}", f"TensileLibrary_{arch}")
        for sol in data.get("solutions", []):
            indexMap[str(sol["index"])] = {
                "solution": sol["name"],
                "kernel": sol["kernelName"],
                "logic_file": logicFiles[0]["path"] if len(logicFiles) == 1 else None,
                "logic_index": sol["libraryLogicIndex"],
                "code_object": None if master else stem,
            }
    return dict(sorted(indexMap.items(), key=lambda kv: int(kv[0])))


def _parseTimingLog(log, sourceRoot, rocisaRoot, logic, out):
    """The cold-path result: parsed records plus the fingerprint envelope.

    The toolchain is the one the logged run reports (its ``Component path``
    lines), with the compiler's current ``--version`` line. Everything else
    is fingerprinted now, at parse time, so parse right after the run.
    """
    _checkTrees(sourceRoot, rocisaRoot)
    library = Path(out) / "library"
    if not library.is_dir():
        sys.exit(f"error: --out {out} holds no library/; give the TensileCreateLibrary output directory")
    libraryFiles = [p for p in sorted(library.rglob("*")) if p.is_file()]
    if not libraryFiles:
        sys.exit(f"error: --out {out}: library/ holds no artifacts; give the output directory of the logged run")
    logicPaths = sorted(Path(logic).rglob("*.yaml"))
    if not logicPaths:
        sys.exit(f"error: --logic: no logic files under {logic}")
    text = log.read_text()
    components = [{"rocm_version": m.group(1), "path": m.group(2), "version": m.group(3)}
                  for m in map(_COMPONENT.match, text.splitlines()) if m]
    cxx = next((c["path"] for c in components if c["path"].endswith("clang++")), None)
    toolchain = {"components": components, "cxx_compiler": cxx,
                 "cxx_compiler_version": _compilerVersion(cxx) if cxx else None}
    logicFiles = [{"path": str(f), "sha256": _sha256(f)} for f in logicPaths]
    fp = _envelope(sourceRoot, rocisaRoot,
                   logicFiles + [{"path": str(log), "sha256": _sha256(log)}], toolchain)
    fp["artifacts"] = [{"path": str(p), "sha256": _sha256(p)} for p in libraryFiles]
    fp["solution_index_map"] = _libraryIndexMap(library, logicFiles)
    return {**parse_print_timing(text), "fingerprint": fp}


# --- the generation job ---------------------------------------------------------


class _EmittedHelper:
    """A helper kernel whose source was emitted during the codegen stage.

    ``writeHelpers`` asks each helper for its name, source and header; this
    hands back what the codegen stage already produced, in the same order.
    """

    def __init__(self, helper):
        self._name = helper.getKernelName()
        self._source = helper.getSourceFileString()
        self._header = helper.getHeaderFileString()

    def getKernelName(self):
        return self._name

    def getSourceFileString(self):
        return self._source

    def getHeaderFileString(self):
        return self._header


def _tensileCreateLibraryArgv(logic, out, arch):
    return [str(logic), str(out), "HIP", f"--architecture={arch}", "--jobs=1",
            "--no-compress", "--disable-asm-comments"]


class _Job:
    """Process-wide state of ``TensileCreateLibrary.run()`` up to logic loading."""

    def __init__(self, logic, arch, argv):
        import rocisa
        from tensilelite.Common import setVerbosity
        from tensilelite.Common.Architectures import (
            baseArchName, expandAllArchitectures, gfxToIsa, splitArchsFromPredicates,
            steppingArchOf, tuningArchOf,
        )
        from tensilelite.Common.Capabilities import applyArchCapOverrides, makeIsaInfoMap
        from tensilelite.Common.GlobalParameters import assignGlobalParameters, globalParameters
        from tensilelite.TensileCreateLibrary import Run
        from tensilelite.TensileCreateLibrary.ParseArguments import parseArguments
        from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
        from tensilelite.Toolchain.Source import makeSourceToolchain
        from tensilelite.Toolchain.Validators import validateToolchain

        sys.argv = ["TensileCreateLibrary"] + argv
        args = parseArguments()
        setVerbosity(args["PrintLevel"])
        cxx, _, bundler, _ = validateToolchain(
            args["CxxCompiler"], args["CCompiler"], args["OffloadBundler"], args["Assembler"])
        archs = expandAllArchitectures(args["Architecture"].split(";"))
        archs, _ = splitArchsFromPredicates(archs)
        isaInfoMap = makeIsaInfoMap([gfxToIsa(a) for a in archs], cxx)
        applyArchCapOverrides(isaInfoMap, archs)
        assignGlobalParameters(args, isaInfoMap)
        steppings = [baseArchName(a) for a in archs if steppingArchOf(a)]
        globalParameters["StinkyTofuArchName"] = tuningArchOf(steppings[0]) if steppings else ""
        self.asm = makeAssemblyToolchain(cxx, bundler, args["CodeObjectVersion"],
                                         args["BuildIdKind"], args["AsmDebug"])
        self.src = makeSourceToolchain(cxx, bundler, args["AsanBuild"], args["BuildIdKind"],
                                       save_temps=False)
        self.rocisa, self.run, self.args = rocisa, Run, args
        self.archs, self.isaInfoMap, self.logic = archs, isaInfoMap, Path(logic)

    def logicFiles(self):
        from tensilelite.Common.Architectures import tuningArchOf
        from tensilelite.CustomYamlLoader import archMatch, load_logic_gfx_arch

        files = sorted(glob.iglob(str(self.logic / "**" / "*.yaml"), recursive=True))
        files = [f for f in files
                 if archMatch(load_logic_gfx_arch(Path(f)), [tuningArchOf(a) for a in self.archs])]
        return [f for f in files if "experimental" not in map(str.lower, Path(f).parts)]

    def measure(self, repDir):
        """Run the job into the new directory ``repDir``; return (stage seconds, facts)."""
        from tensilelite import LibraryIO
        from tensilelite.Common import DebugConfig, ensurePath, state
        from tensilelite.Common.Architectures import (
            archNamesByIsa, baseArchName, compilerTargetOf, gfxToIsa, isaToGfx,
        )
        from tensilelite.KernelWriterAssembly import KernelWriterAssembly
        from tensilelite.KernelWriterBase import KERNEL_HELPER_FILENAME_CPP, KERNEL_HELPER_FILENAME_H
        from tensilelite.SolutionStructs.Naming import getKernelNameMin
        from tensilelite.Toolchain.Assembly import buildAssemblyCodeObjectFiles
        from tensilelite.Toolchain.Source import buildSourceCodeObjectFiles

        Run, args, archs = self.run, self.args, self.archs
        repDir = Path(repDir)
        repDir.mkdir(parents=True)  # never a used directory: its leftovers would be measured
        os.chdir(repDir)  # GenSolTable writes MatchTable into the working directory
        times = {}

        t = time.perf_counter()
        logicFiles = self.logicFiles()
        if not logicFiles:
            sys.exit(f"error: --logic: no {','.join(archs)} logic files under {self.logic}")
        solutions, masterLibraries, libraryMapping = Run.generateLogicDataAndSolutions(
            logicFiles, args, self.asm.assembler, self.isaInfoMap)
        kernels = Run.generateKernelObjectsFromSolutions(solutions)
        # As TensileCreateLibrary's run(): helpers from every library solution,
        # which can need more than the solutions sharing one kernel.
        librarySolutions = [
            sol.originalSolution
            for masterLibrary in masterLibraries.values()
            for lib in (masterLibrary, *masterLibrary.lazyLibraries.values())
            for sol in lib.solutions.values()
        ]
        helperObjs = Run.generateKernelHelperObjects(librarySolutions, str(self.asm.assembler.path), self.isaInfoMap)
        times["derive"] = time.perf_counter() - t

        t = time.perf_counter()
        writer = KernelWriterAssembly(self.asm.assembler, DebugConfig())
        groups = [g for g in Run._normalizeKernelGroups(kernels, False)
                  if g.representative["KernelLanguage"] == "Assembly"]
        unique = [g.representative for g in groups]
        outOptions = self.rocisa.rocIsa.getInstance().getOutputOptions()
        outOptions.outputNoComment = args["DisableAsmComments"]
        data = self.rocisa.rocIsa.getInstance().getData()
        results = [Run.processKernelSource(writer, data, outOptions, False, k) for k in unique]
        helpers = [_EmittedHelper(h) for h in helperObjs]
        times["codegen"] = time.perf_counter() - t

        t = time.perf_counter()
        destRoot = ensurePath(Run.libraryRoot(repDir))
        for base in Run._baseArchs(archs):
            ensurePath(Run.libraryDir(repDir, base))
        buildTmp = ensurePath(Run.buildTmpDir(repDir, archs))
        asmDir = ensurePath(buildTmp / "assembly")
        objDir = ensurePath(buildTmp / "code_object_tmp")
        Run.copyStaticFiles(buildTmp)
        written = [Run.writeAssembly(asmDir, r) for r in results]
        Run.writeHelpers(buildTmp, helpers, KERNEL_HELPER_FILENAME_CPP, KERNEL_HELPER_FILENAME_H)
        times["write"] = time.perf_counter() - t

        t = time.perf_counter()
        archNames = archNamesByIsa(archs)
        for sPath, isa, wavefront, _ in written:
            target = archNames.get(isa) or isaToGfx(isa)
            Run._alignAmdgcnTargetToStepping(sPath, isa, compilerTargetOf(target))
            self.asm.assembler(target, wavefront, str(sPath), str(sPath.with_suffix(".o")))
            if Run._stinky_asm_verify_wanted(isa):
                Run._verify_stinky_asm_comment_vs_elf_text(sPath, sPath.with_suffix(".o"), sPath.stem)
        times["assemble"] = time.perf_counter() - t

        t = time.perf_counter()
        mainCodeObjects = buildAssemblyCodeObjectFiles(
            self.asm.linker, self.asm.bundler, groups, destRoot, asmDir,
            args["UseCompression"], archNames=archNames)
        times["link"] = time.perf_counter() - t

        t = time.perf_counter()
        helperCodeObjects = buildSourceCodeObjectFiles(
            self.src.compiler, self.src.bundler, destRoot, objDir,
            buildTmp, buildTmp / "Kernels.cpp", archs)
        times["helper_compile"] = time.perf_counter() - t
        # objDir is new in this repetition, so Kernels.o exists only if this
        # repetition compiled the helpers; a cache hit restores them instead.
        helperCache = "miss" if (objDir / "Kernels.o").exists() else "hit"

        t = time.perf_counter()
        outArchs = list(dict.fromkeys(
            baseArchName(a) for a in archs if self.isaInfoMap[gfxToIsa(a)].asmCaps["SupportedISA"]))
        Run.passPostKernelInfoToLibrary(results, unique, masterLibraries, False)
        outDirs = {compilerTargetOf(a): a for a in outArchs}
        for archName, outDir in outDirs.items():
            archMapping = {i: n for i, n in libraryMapping.items() if n.endswith("_" + archName)}
            if archMapping:
                LibraryIO.write(
                    os.path.join(Run.libraryDir(repDir, outDir), "TensileLiteLibrary_lazy_" + archName + "_Mapping"),
                    archMapping, "msgpack")
        for archName, master in masterLibraries.items():
            if archName not in outDirs:
                continue
            archDir = Run.libraryDir(repDir, outDirs[archName])
            prefix = "TensileLibrary_lazy_" if args["LazyLibraryLoading"] else "TensileLibrary_"
            master.applyNaming(False)
            LibraryIO.write(os.path.join(archDir, prefix + archName), state(master), args["LibraryFormat"])
            for name, lib in master.lazyLibraries.items():
                lib.applyNaming(False)
                LibraryIO.write(os.path.join(archDir, name), state(lib), args["LibraryFormat"])
        times["metadata"] = time.perf_counter() - t

        solutionIndexMap = {}
        for master in masterLibraries.values():
            for lib in [master] + list(master.lazyLibraries.values()):
                for sol in lib.solutions.values():
                    solutionIndexMap[str(sol.index)] = {
                        "solution": sol.name,
                        "kernel": getKernelNameMin(sol.originalSolution.getKernels()[0], False),
                        "logic_file": sol.srcName,
                        "logic_index": sol.libraryLogicIndex,
                        "code_object": sol.originalSolution._state.get("codeObjectFile"),
                    }
        facts = {
            "logic_files": logicFiles,
            "kernels": [getKernelNameMin(k, False) for k in unique],
            "helpers": [h.getKernelName() for h in helpers],
            "main_code_objects": [str(p) for p in mainCodeObjects],
            "helper_code_objects": [str(p) for p in helperCodeObjects],
            "assembly": [str(w[0]) for w in written],
            "solution_index_map": dict(sorted(solutionIndexMap.items(), key=lambda kv: int(kv[0]))),
        }
        return times, facts, helperCache


def _worker(args):
    start = time.perf_counter()
    # Run as a script: drop this file's directory so every import below comes
    # from the PYTHONPATH the launcher set.
    if sys.path and Path(sys.path[0] or ".").resolve() == Path(__file__).resolve().parent:
        sys.path.pop(0)
    tlRoot = args.source_root / _TL_SUBDIR
    out = args.out.resolve()
    argv = _tensileCreateLibraryArgv(args.logic.resolve(), out / "rep1", args.arch)
    result = {
        "schema_version": SCHEMA_VERSION,
        "driver": {"file": str(Path(__file__).resolve()), "python": sys.version,
                   "executable": sys.executable},
        "source_root": str(args.source_root), "tensilelite_root": str(tlRoot),
        "rocisa": str(args.rocisa), "logic": str(args.logic.resolve()), "arch": args.arch,
        "out": str(out), "repeat": args.repeat,
        "tensile_create_library_argv": argv[2:],
        "helper_cache_env": {k: os.environ.get(k) for k in
                             ("TENSILE_HELPER_CACHE_DIR", "TENSILE_DISABLE_HELPER_CACHE")},
        "host": {"loadavg": list(os.getloadavg()), "cpu_affinity": sorted(os.sched_getaffinity(0))},
        "repetitions": [], "first": None, "rest": None, "artifacts": None,
    }

    job = _Job(args.logic.resolve(), args.arch, argv)
    setup = time.perf_counter() - start
    result["import_identity"] = _identity(args.source_root, tlRoot, args.rocisa)
    if not result["import_identity"]["ok"]:
        _reportViolations(result["import_identity"], args.rocisa)
        _writeJson(args.json, result)
        return 1

    facts = None
    for k in range(1, args.repeat + 1):
        repDir = out / f"rep{k}"
        t = time.perf_counter()
        times, repFacts, helperCache = job.measure(repDir)
        wall = time.perf_counter() - t
        facts = facts or repFacts
        times["process_setup"] = setup if k == 1 else NA
        stages = {s: times[s] for s in STAGES}
        result["repetitions"].append({
            "index": k, "dir": str(repDir), "stages": stages,
            "process_setup_s": stages["process_setup"],
            "compile_s": sum(stages[s] for s in STAGES if s != "process_setup"),
            "wall_s": wall,
            "helper_cache": helperCache,
        })
    result["job_wall_s"] = time.perf_counter() - start

    reps = result["repetitions"]
    result["first"] = {key: reps[0][key] for key in
                       ("stages", "process_setup_s", "compile_s", "wall_s", "helper_cache")}
    result["rest"] = _rest(reps)
    result["logic_files"] = [{"path": f, "sha256": _sha256(f)} for f in facts["logic_files"]]
    result["kernels"] = facts["kernels"]
    result["helpers"] = facts["helpers"]
    result["artifacts"] = _artifacts(reps, facts)
    # Modules loaded while generating count too.
    result["import_identity"] = _identity(args.source_root, tlRoot, args.rocisa)
    result["fingerprint"] = _fingerprint(
        job, args.source_root, args.rocisa, tlRoot / "tensilelite", result["logic_files"],
        result["artifacts"]["library_files"], facts["solution_index_map"])
    _writeJson(args.json, result)
    if not result["import_identity"]["ok"]:
        _reportViolations(result["import_identity"], args.rocisa)
        return 1
    return 0


def _launch(args):
    """Run the measurement in a fresh interpreter that sees only the given trees."""
    tlRoot = args.source_root / _TL_SUBDIR
    if not (tlRoot / "tensilelite" / "__init__.py").is_file():
        sys.exit(f"--source-root: no {_TL_SUBDIR}/tensilelite package under {args.source_root}")
    if not (args.rocisa / "rocisa" / "__init__.py").is_file():
        sys.exit(f"--rocisa: no rocisa package under {args.rocisa}")
    if args.repeat < 1:
        sys.exit("--repeat must be at least 1")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(args.rocisa), str(tlRoot)])
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
           "--source-root", str(args.source_root), "--rocisa", str(args.rocisa),
           "--logic", str(args.logic), "--arch", args.arch, "--out", str(args.out),
           "--repeat", str(args.repeat), "--json", str(args.json)]
    return subprocess.run(cmd, env=env).returncode


def main(argv=None):
    parser = _argParser()
    args = parser.parse_args(argv)
    if args.parse_timing_log is not None:
        missing = [n for n in ("source_root", "rocisa", "logic", "out") if getattr(args, n) is None]
        if missing:
            parser.error("--parse-timing-log needs " + ", ".join("--" + n.replace("_", "-") for n in missing)
                         + " of the logged run")
        _writeJson(args.json, _parseTimingLog(
            args.parse_timing_log.absolute(), args.source_root.absolute(), args.rocisa.absolute(),
            args.logic.absolute(), args.out.absolute()))
        return 0
    missing = [n for n in ("source_root", "rocisa", "logic", "arch", "out") if getattr(args, n) is None]
    if missing:
        parser.error("missing " + ", ".join("--" + n.replace("_", "-") for n in missing))
    for name in ("source_root", "rocisa", "logic", "out", "json"):
        setattr(args, name, getattr(args, name).absolute())
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        sys.exit(f"error: --out {args.out} is not empty; the driver measures only what it "
                 "writes there, so give it a new or empty directory")
    return _worker(args) if args.worker else _launch(args)


if __name__ == "__main__":
    sys.exit(main())

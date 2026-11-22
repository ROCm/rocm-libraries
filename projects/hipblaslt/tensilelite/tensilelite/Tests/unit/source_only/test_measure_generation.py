# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the generation-job measurement driver (``measure_generation``).

Source-only: the driver measures a monorepo checkout (``--source-root``) and a
rocisa build of that checkout, so these tests need the checkout layout and a
source-built rocisa (one with ``rocisa/_build_info.py``). They are not
installed (``source_only`` is excluded from the install and the wheel).

The driver is exercised only through its command line, in subprocesses, the
way baseline measurements run it. Every test needs rocisa and the ROCm
assembler (amdclang++, offload bundler, llvm-readelf) but no GPU.

Fixture: ``data/measure_generation`` holds one dense library logic file with a
single solution, extracted from a stock gfx950 logic file (provenance in the
file header).
"""

import hashlib
import importlib.util
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_HERE = Path(__file__).resolve().parent
_PACKAGE_DIR = _HERE.parents[2]  # .../tensilelite/tensilelite
_TL_ROOT = _PACKAGE_DIR.parent  # .../projects/hipblaslt/tensilelite
_SOURCE_ROOT = _TL_ROOT.parents[2]  # monorepo root
_TL_SUBDIR = Path("projects", "hipblaslt", "tensilelite")
_LOGIC_DIR = _HERE / "data" / "measure_generation"
# One custom-kernel solution (wvSpltK_bf16_tn_m1, the stock pick of the N=1 TN bf16
# decode rows), as a one-solution general logic file for its exact sizes.
_CUSTOM_LOGIC_DIR = _HERE / "data" / "measure_generation_custom_kernel"
_CUSTOM_KERNEL = _PACKAGE_DIR / "CustomKernels" / "rocblas" / "wvSpltK_bf16_tn_m1.s"
_MODULE = "tensilelite.Tests.unit.characterization._codegen.measure_generation"
_DRIVER_FILE = _PACKAGE_DIR / "Tests" / "unit" / "characterization" / "_codegen" / "measure_generation.py"
_ARCH = "gfx950"
_STAGES = ("process_setup", "derive", "codegen", "write", "assemble", "helper_compile", "link", "metadata")
_COMPILE_STAGES = _STAGES[1:]
# TensileCreateLibrary options the driver mirrors: production comment setting
# (TENSILELITE_ENABLE_ASM_COMMENTS=OFF -> --disable-asm-comments), uncompressed
# code objects, one job.
_TCL_OPTIONS = (f"--architecture={_ARCH}", "--jobs=1", "--no-compress", "--disable-asm-comments")
_READELF = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "llvm" / "bin" / "llvm-readelf"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _rocisa_dir() -> Path:
    """The rocisa build directory this interpreter imports rocisa from."""
    spec = importlib.util.find_spec("rocisa")
    assert spec is not None and spec.origin, "rocisa is not importable"
    return Path(spec.origin).resolve().parents[1]


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_driver(*args, cache_dir: Path, env_extra=None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # The launcher itself only needs this tree's package path; the measured
    # tree is whatever --source-root names.
    env["PYTHONPATH"] = str(_TL_ROOT)
    env["TENSILE_HELPER_CACHE_DIR"] = str(cache_dir)
    env.pop("TENSILE_DISABLE_HELPER_CACHE", None)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", _MODULE, *map(str, args)],
        capture_output=True, text=True, env=env, timeout=1800,
    )


def _measure(tmp: Path, source_root: Path, rocisa: Path, repeat: int, out=None, logic=_LOGIC_DIR):
    out, js = out or tmp / "out", tmp / "result.json"
    proc = _run_driver(
        "--source-root", source_root, "--rocisa", rocisa, "--logic", logic,
        "--arch", _ARCH, "--out", out, "--repeat", repeat, "--json", js,
        cache_dir=tmp / "helper_cache",
    )
    data = json.loads(js.read_text()) if js.exists() else None
    return proc, data


def _one(items, what):
    items = list(items)
    assert len(items) == 1, (what, items)
    return items[0]


def _is_under(path, root) -> bool:
    return Path(os.path.realpath(path)).is_relative_to(Path(os.path.realpath(root)))


def _copy_tree(root: Path) -> Path:
    """A second monorepo-shaped tree holding a copy of this TensileLite package."""
    package = root / _TL_SUBDIR / "tensilelite"
    shutil.copytree(_PACKAGE_DIR, package,
                    ignore=shutil.ignore_patterns("Tests", "__pycache__", "CustomKernels"))
    # Data only (no Python modules); linked to keep the copy small.
    os.symlink(_PACKAGE_DIR / "CustomKernels", package / "CustomKernels")
    return package


def _rocisa_build_of(root: Path, build: Path) -> Path:
    """Stand in for a rocisa build of the tree at ``root``.

    Copies this interpreter's rocisa build and its recorded sources into
    ``build`` and ``root``, timestamps kept, and records ``root``'s rocisa
    sources as its SOURCE_ROOT, as building rocisa from ``root`` would.
    """
    real = _rocisa_dir() / "rocisa"
    info = runpy.run_path(str(real / "_build_info.py"))
    sources = root / _TL_SUBDIR / "rocisa" / "rocisa"
    shutil.copytree(info["SOURCE_ROOT"], sources, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(real, build / "rocisa", ignore=shutil.ignore_patterns("__pycache__"))
    (build / "rocisa" / "_build_info.py").write_text(
        f"SOURCE_ROOT = {str(sources)!r}\n"
        f"STINKYTOFU_SOURCE_ROOT = {info['STINKYTOFU_SOURCE_ROOT']!r}\n"
        f"BUILD_DIR = {str(build)!r}\n")
    return build


_DELAY_S = 1.0
# Production function -> the stage that must pay for it. Each is wrapped in the
# copied tree with a fixed sleep, so a stage boundary that moves shows up as a
# delay in the wrong stage.
_DELAYED = {
    ("TensileCreateLibrary/Run.py", "generateLogicDataAndSolutions"): "derive",
    ("TensileCreateLibrary/Run.py", "processKernelSource"): "codegen",
    ("TensileCreateLibrary/Run.py", "writeAssembly"): "write",
    ("Toolchain/Component.py", "Assembler.__call__"): "assemble",
    ("Toolchain/Source.py", "buildSourceCodeObjectFiles"): "helper_compile",
    ("Toolchain/Assembly.py", "buildAssemblyCodeObjectFiles"): "link",
    ("TensileCreateLibrary/Run.py", "passPostKernelInfoToLibrary"): "metadata",
}


def _delay(package: Path):
    for (rel, name), _ in _DELAYED.items():
        with open(package / rel, "a") as f:
            f.write(
                "\n\ndef _mg_delayed(f):\n"
                "    import functools, time\n"
                "    @functools.wraps(f)\n"
                "    def wrapper(*a, **k):\n"
                f"        time.sleep({_DELAY_S})\n"
                "        return f(*a, **k)\n"
                "    return wrapper\n"
                f"{name} = _mg_delayed({name})\n")


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("measure")


@pytest.fixture(scope="module")
def tcl(workdir):
    """TensileCreateLibrary on the same fixture with TENSILE_PRINT_TIMING=ON.

    It writes to the path the driver's first repetition uses afterwards, then
    moves its output aside: the compiled helper code object embeds a hash of
    the helper source path (clang's ``__hip_cuid_*``), so byte equality needs
    equal paths. Each run gets a fresh helper cache, so both compile.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_rocisa_dir()), str(_TL_ROOT)])
    env["TENSILE_PRINT_TIMING"] = "ON"
    env["TENSILE_HELPER_CACHE_DIR"] = str(workdir / "tcl_helper_cache")
    env.pop("TENSILE_DISABLE_HELPER_CACHE", None)
    out = workdir / "out" / "rep1"
    proc = subprocess.run(
        [sys.executable, "-m", "tensilelite.TensileCreateLibrary",
         str(_LOGIC_DIR), str(out), "HIP", *_TCL_OPTIONS, "--keep-build-tmp"],
        capture_output=True, text=True, env=env, cwd=workdir, timeout=1800,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    log = workdir / "tcl.log"
    log.write_text(proc.stdout + proc.stderr)
    moved = workdir / "tcl_out"
    out.rename(moved)
    return moved, log


@pytest.fixture(scope="module")
def measured(workdir, tcl):
    """The driver on this tree, three repetitions in one warm process."""
    proc, data = _measure(workdir, _SOURCE_ROOT, _rocisa_dir(), 3)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    assert data is not None, "driver wrote no JSON"
    return data


@pytest.fixture(scope="module")
def delayed(tmp_path_factory):
    """The driver on a copied tree whose stage functions each sleep _DELAY_S."""
    tmp = tmp_path_factory.mktemp("delayed")
    root = tmp / "tree"
    _delay(_copy_tree(root))
    proc, data = _measure(tmp, root, _rocisa_build_of(root, tmp / "rocisa_build"), 1)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    return data


def _rep_dir(data, index) -> Path:
    return Path(data["repetitions"][index]["dir"])


def _main_object(data):
    mains = [co for co in data["artifacts"]["code_objects"] if co["kind"] == "main"]
    assert len(mains) == 1
    return mains[0]


def _section_size(path, name) -> int:
    out = subprocess.run([str(_READELF), "-S", "--wide", str(path)],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        fields = line.replace("[ ", "[").split()
        if len(fields) > 5 and fields[1] == name:
            return int(fields[5], 16)
    raise AssertionError(f"{name} not found in {path}")


def test_every_stage_is_timed_in_every_repetition(measured):
    reps = measured["repetitions"]
    assert [r["index"] for r in reps] == [1, 2, 3]
    for stage in _STAGES:
        value = reps[0]["stages"][stage]
        assert isinstance(value, float) and value >= 0.0, (stage, value)
    for rep in reps[1:]:
        assert set(rep["stages"]) == set(_STAGES)
        # A warm process does not set itself up again.
        assert rep["stages"]["process_setup"] == "n/a"
        for stage in _COMPILE_STAGES:
            value = rep["stages"][stage]
            assert isinstance(value, float) and value >= 0.0, (rep["index"], stage, value)


def test_compile_time_is_the_six_stages_without_setup(measured):
    """Complete compilation = derive..metadata; setup reported apart."""
    for rep in measured["repetitions"]:
        stages = rep["stages"]
        assert isinstance(rep["compile_s"], float) and isinstance(rep["wall_s"], float), rep
        assert rep["compile_s"] == pytest.approx(sum(stages[s] for s in _COMPILE_STAGES))
        assert rep["process_setup_s"] == stages["process_setup"]
        # The wall clock brackets the whole repetition independently.
        assert rep["wall_s"] >= rep["compile_s"]
    first = measured["repetitions"][0]
    assert first["process_setup_s"] > 0.0
    assert first["wall_s"] < first["compile_s"] + first["process_setup_s"]
    assert measured["job_wall_s"] >= first["process_setup_s"] + sum(
        r["wall_s"] for r in measured["repetitions"])


def test_each_stage_owns_its_production_step(delayed):
    """A sleep injected into one production step lands in exactly that stage."""
    stages = delayed["repetitions"][0]["stages"]
    for stage in set(_DELAYED.values()):
        assert isinstance(stages[stage], float) and stages[stage] >= _DELAY_S, (stage, stages[stage])
        if stage != "helper_compile":  # a helper compile miss alone takes seconds
            assert stages[stage] < 2 * _DELAY_S, (stage, stages[stage])


def test_first_repetition_is_reported_apart_from_the_rest(measured):
    reps = measured["repetitions"]
    for key in ("stages", "compile_s", "wall_s", "process_setup_s", "helper_cache"):
        assert measured["first"][key] == reps[0][key]
    rest = measured["rest"]
    assert rest["n"] == 2
    assert rest["stages"]["process_setup"] == "n/a"
    for stage in _COMPILE_STAGES:
        values = [r["stages"][stage] for r in reps[1:]]
        stats = rest["stages"][stage]
        assert min(values) <= stats["q1"] <= stats["median"] <= stats["q3"] <= max(values)
    for key in ("compile_s", "wall_s"):
        values = [r[key] for r in reps[1:]]
        assert min(values) <= rest[key]["median"] <= max(values)
    assert rest["helper_cache"] == ["hit", "hit"]


def test_helper_cache_state_is_reported_per_repetition(measured):
    # Fresh cache directory: the first job compiles the helpers, the second
    # restores what the first stored.
    states = [r["helper_cache"] for r in measured["repetitions"]]
    assert states == ["miss", "hit", "hit"]


def test_a_used_output_directory_is_refused(measured, tmp_path):
    """Leftovers of an earlier run must not be measured or relabel the cache state."""
    before = sorted(p for p in Path(measured["out"]).rglob("*"))
    proc, data = _measure(tmp_path, _SOURCE_ROOT, _rocisa_dir(), 1, out=Path(measured["out"]))
    assert proc.returncode != 0
    assert "not empty" in proc.stderr
    assert data is None
    assert sorted(p for p in Path(measured["out"]).rglob("*")) == before


def test_sizes_come_from_the_built_code_objects(measured):
    main = _main_object(measured)
    co = Path(main["path"])
    assert _is_under(co, _rep_dir(measured, 0))
    assert main["co_bytes"] == co.stat().st_size
    assert main["text_bytes"] == _section_size(co, ".text") > 0
    helpers = [c for c in measured["artifacts"]["code_objects"] if c["kind"] == "helper"]
    assert helpers
    for helper in helpers:
        assert helper["co_bytes"] == Path(helper["path"]).stat().st_size
        assert helper["text_bytes"] == _section_size(helper["path"], ".text") > 0
    every = measured["artifacts"]["code_objects"]
    on_disk = sorted(str(p) for p in (_rep_dir(measured, 0) / "library").rglob("*")
                     if p.suffix in (".co", ".hsaco"))
    assert sorted(c["path"] for c in every) == on_disk
    totals = measured["artifacts"]["totals"]
    assert totals["text_bytes"] == sum(c["text_bytes"] for c in every)
    assert totals["co_bytes"] == sum(c["co_bytes"] for c in every)


def test_comment_free_assembly_lines_are_counted(measured):
    asm = _one(measured["artifacts"]["assembly"], "assembly")
    lines = Path(asm["path"]).read_text().splitlines()
    nonblank = [line for line in lines if line.strip()]
    comment_only = [line for line in nonblank if re.match(r"^\s*(//|/\*.*\*/\s*$)", line)]
    assert comment_only, "fixture assembly is expected to carry comment-only lines"
    assert asm["s_lines"] == len(nonblank) - len(comment_only)
    assert measured["artifacts"]["totals"]["s_lines"] == asm["s_lines"]


def test_resources_come_from_code_object_metadata(measured):
    main = _main_object(measured)
    kernel = _one(main["kernels"], "main kernels")
    assert kernel["name"] == measured["kernels"][0]
    for key in ("sgpr", "vgpr", "agpr", "lds", "kernarg"):
        assert isinstance(kernel[key], int) and kernel[key] >= 0, key
    src = Path(measured["artifacts"]["assembly"][0]["path"]).read_text()
    field = lambda pattern: int(re.search(pattern, src).group(1))
    assert kernel["lds"] == field(r"\.amdhsa_group_segment_fixed_size\s+(\d+)")
    assert kernel["kernarg"] == field(r"\.kernarg_segment_size:\s+(\d+)")
    assert kernel["sgpr"] == field(r"\.sgpr_count:\s+(\d+)")
    assert kernel["vgpr"] == field(r"\.vgpr_count:\s+(\d+)")
    assert kernel["sgpr"] > 0 and kernel["vgpr"] > 0


def test_agprs_of_assembly_kernels_come_from_their_descriptor(measured):
    """Assembly kernel metadata has no .agpr_count; the descriptor directives fix it."""
    kernel = _one(_main_object(measured)["kernels"], "main kernels")
    src = Path(measured["artifacts"]["assembly"][0]["path"]).read_text()
    assert ".agpr_count" not in src
    nextFree = int(re.search(r"\.amdhsa_next_free_vgpr\s+(\d+)", src).group(1))
    accumOffset = int(re.search(r"\.amdhsa_accum_offset\s+(\d+)", src).group(1))
    assert kernel["agpr"] == nextFree - accumOffset > 0
    assert kernel["agpr_source"] == "assembly"
    helpers = [k for c in measured["artifacts"]["code_objects"] if c["kind"] == "helper"
               for k in c["kernels"]]
    assert helpers and all(k["agpr_source"] == "metadata" for k in helpers)


def test_custom_kernel_metadata_with_integer_map_keys_is_measured(tmp_path):
    """A custom kernel's metadata note carries ``custom.config``, whose assertion
    maps have integer keys (``AssertSizeEqual: { 1: 1, 2: 1 }``); its resources
    are still read from the note."""
    src = _CUSTOM_KERNEL.read_text()
    assert re.search(r"^\s*AssertSizeEqual:\s*\{\s*\d+:", src, re.M)
    proc, data = _measure(tmp_path, _SOURCE_ROOT, _rocisa_dir(), 1, logic=_CUSTOM_LOGIC_DIR)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    kernel = _one(_main_object(data)["kernels"], "main kernels")
    field = lambda pattern: int(re.search(pattern, src).group(1))
    assert kernel["name"] == "wvSpltK_bf16_tn_m1"
    assert kernel["sgpr"] == field(r"\.sgpr_count:\s+(\d+)")
    assert kernel["vgpr"] == field(r"\.vgpr_count:\s+(\d+)")
    assert kernel["agpr"] == field(r"\.agpr_count:\s+(\d+)")
    assert kernel["agpr_source"] == "metadata"
    assert kernel["lds"] == field(r"\.group_segment_fixed_size:\s+(\d+)")
    assert kernel["kernarg"] == field(r"\.kernarg_segment_size:\s+(\d+)")


def test_helper_list_matches_tensile_create_library(measured, tcl):
    """Names and order against what TensileCreateLibrary wrote into Kernels.cpp."""
    tcl_out, log = tcl
    count = re.search(r"^KernelHelperObjs: (\d+)$", log.read_text(), re.M)
    helpers = measured["helpers"]
    assert len(helpers) == int(count.group(1)) > 0
    assert len(set(helpers)) == len(helpers)
    kernelsCpp = _one((tcl_out / "build_tmp").rglob("Kernels.cpp"), "TCL Kernels.cpp")
    globals_ = re.findall(r"^__global__ void (\w+)\(", kernelsCpp.read_text(), re.M)
    kernels = [h for h in helpers if h.startswith("Cijk_")]
    assert kernels, "fixture is expected to need reduction helpers"
    positions = [globals_.index(h) for h in kernels]
    assert positions == sorted(positions)
    # The activation helpers come first, exactly as TensileCreateLibrary's
    # Kernels.h declares them: the enum of its compute type, then the
    # activation function of the logic's activation type.
    header = _one((tcl_out / "build_tmp").rglob("Kernels.h"), "TCL Kernels.h").read_text()
    enums = re.findall(r"^enum class (Gradient)?ActivationType_(\w+)\b", header, re.M)
    funcs = re.findall(r"^__device__ inline \w+ activation\(\s*\w+ value,\s*"
                       r"Tensile::ActivationType_(\w+) const", header, re.M)
    grad, compute = _one(enums, "activation enums in Kernels.h")
    assert _one(funcs, "activation functions in Kernels.h") == compute
    problem = yaml.safe_load(next(_LOGIC_DIR.glob("*.yaml")).read_text())["ProblemType"]
    guard = "NG" if problem["ActivationNoGuard"] else ""
    others = [h for h in helpers if not h.startswith("Cijk_")]
    assert others == [f"Tensile{grad}ActivationEnum_{compute}",
                      f"Tensile{grad}Activation{guard}_{compute}_{problem['ActivationType'].capitalize()}"]
    assert helpers[:len(others)] == others


def test_helper_list_keeps_each_solutions_helpers_on_a_shared_handwritten_kernel(tmp_path):
    """The custom-kernel solution and its bespoke (BespokeNarrowing) UseBeta:false
    twin share one handwritten kernel (named by CustomKernel.name alone) but need
    different conversion helpers; the driver builds the helpers TensileCreateLibrary
    builds, in its order."""
    logic = tmp_path / "logic"
    logic.mkdir()
    general = _one(_CUSTOM_LOGIC_DIR.glob("*.yaml"), "custom-kernel logic")
    shutil.copy(general, logic / "000_general.yaml")
    beta_off = yaml.safe_load(general.read_text())
    beta_off["ProblemType"]["UseBeta"] = False
    for solution in beta_off["Solutions"]:
        solution["BespokeNarrowing"] = True
    (logic / "001_beta_off.yaml").write_text(yaml.safe_dump(beta_off, sort_keys=False))

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_rocisa_dir()), str(_TL_ROOT)])
    env["TENSILE_HELPER_CACHE_DIR"] = str(tmp_path / "tcl_helper_cache")
    env.pop("TENSILE_DISABLE_HELPER_CACHE", None)
    tcl_out = tmp_path / "tcl_out"
    tcl = subprocess.run(
        [sys.executable, "-m", "tensilelite.TensileCreateLibrary",
         str(logic), str(tcl_out), "HIP", *_TCL_OPTIONS, "--keep-build-tmp"],
        capture_output=True, text=True, env=env, cwd=tmp_path, timeout=1800,
    )
    assert tcl.returncode == 0, tcl.stdout[-4000:] + tcl.stderr[-4000:]
    count = re.search(r"^KernelHelperObjs: (\d+)$", tcl.stdout + tcl.stderr, re.M)
    kernelsCpp = _one((tcl_out / "build_tmp").rglob("Kernels.cpp"), "TCL Kernels.cpp")
    globals_ = re.findall(r"^__global__ void (\w+)\(", kernelsCpp.read_text(), re.M)
    assert any("_Beta0_" in g for g in globals_) and any("_Beta0_" not in g for g in globals_), \
        "fixture no longer needs both beta-on and UseBeta:false helpers"

    proc, data = _measure(tmp_path, _SOURCE_ROOT, _rocisa_dir(), 1, logic=logic)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    helpers = data["helpers"]
    assert len(helpers) == int(count.group(1))
    kernels = [h for h in helpers if h.startswith("Cijk_")]
    positions = [globals_.index(h) for h in kernels]
    assert positions == sorted(positions)
    assert any("_Beta0_" in h for h in kernels) and any("_Beta0_" not in h for h in kernels)


def test_library_files_equal_tensile_create_library_output(measured, tcl):
    tcl_out, _ = tcl
    ours = _rep_dir(measured, 0) / "library"
    theirs = tcl_out / "library"
    tcl_files = sorted(p.relative_to(theirs) for p in theirs.rglob("*") if p.is_file())
    our_files = sorted(p.relative_to(ours) for p in ours.rglob("*") if p.is_file())
    assert any(p.suffix == ".co" for p in tcl_files)
    assert our_files == tcl_files
    for rel in tcl_files:
        assert (ours / rel).read_bytes() == (theirs / rel).read_bytes(), str(rel)
    assert measured["artifacts"]["identical_across_repetitions"] is True


def test_assembly_equals_tensile_create_library_output(measured, tcl):
    """Same emitted source, comment setting included (TCL keeps it with --keep-build-tmp)."""
    tcl_out, _ = tcl
    asm = _one(measured["artifacts"]["assembly"], "assembly")
    ours = Path(asm["path"])
    theirs = tcl_out / ours.relative_to(_rep_dir(measured, 0))
    assert ours.read_bytes() == theirs.read_bytes()


def test_result_carries_its_fingerprint(measured):
    assert measured["schema_version"] == 3
    fp = measured["fingerprint"]
    assert fp["driver"]["sha256"] == _sha256(_DRIVER_FILE)
    tl = fp["tensilelite"]
    assert _is_under(tl["package"], _PACKAGE_DIR) and _HEX64.match(tl["content_sha256"])
    _assert_source_identity(fp["source"], _SOURCE_ROOT)
    rocisa = fp["rocisa"]
    _assert_rocisa_identity(rocisa, _rocisa_dir())
    assert rocisa["loaded_modules"]["rocisa"] == _sha256(_rocisa_dir() / "rocisa" / "__init__.py")
    so = _one((_rocisa_dir() / "rocisa").glob("_rocisa*.so"), "rocisa extension")
    assert rocisa["so_sha256"] == _sha256(so) == rocisa["loaded_modules"]["rocisa._rocisa"]
    assert _is_under(rocisa["source_root"], _SOURCE_ROOT)
    tc = fp["toolchain"]
    assert "clang" in tc["cxx_compiler_version"]
    assert tc["rocm_version"] and tc["offload_bundler"] and tc["python"] == sys.version
    assert [f["sha256"] for f in fp["inputs"]] == [_sha256(f) for f in sorted(_LOGIC_DIR.glob("*.yaml"))]
    assert fp["artifacts"] == [{k: f[k] for k in ("path", "sha256")}
                               for f in measured["artifacts"]["library_files"]]
    entry = _one(fp["solution_index_map"].values(), "solution index map")
    assert entry["logic_index"] == 0
    assert Path(entry["logic_file"]).name == next(_LOGIC_DIR.glob("*.yaml")).name
    assert entry["kernel"] == measured["kernels"][0]
    assert entry["code_object"].endswith("_" + _ARCH)


def test_imports_resolve_under_this_tree(measured):
    identity = measured["import_identity"]
    assert identity["ok"] is True
    assert identity["violations"] == []
    modules = identity["modules"]
    assert "tensilelite.TensileCreateLibrary.Run" in modules
    assert "rocisa" in modules
    for name, path in modules.items():
        root = _rocisa_dir() if name.split(".")[0] == "rocisa" else _TL_ROOT
        assert _is_under(path, root), (name, path)


def test_imports_resolve_under_another_source_root(tmp_path):
    """Measuring a different tree loads that tree, not the launcher's."""
    root = tmp_path / "other"
    package = _copy_tree(root)
    proc, data = _measure(tmp_path, root, _rocisa_build_of(root, tmp_path / "rocisa_build"), 1)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    identity = data["import_identity"]
    assert identity["ok"] is True
    run = identity["modules"]["tensilelite.TensileCreateLibrary.Run"]
    assert _is_under(run, package)
    for name, path in identity["modules"].items():
        if name.split(".")[0] != "rocisa":
            assert _is_under(path, root), (name, path)
    assert data["rest"] is None


def test_a_rocisa_module_loaded_from_outside_the_roots_fails_the_run(tmp_path):
    """rocisa reached through a link into another directory is not under --rocisa."""
    fake = tmp_path / "fake_rocisa"
    fake.mkdir()
    os.symlink(_rocisa_dir() / "rocisa", fake / "rocisa")
    proc, data = _measure(tmp_path, _SOURCE_ROOT, fake, 1)
    assert proc.returncode != 0
    assert data is not None, "driver wrote no JSON"
    identity = data["import_identity"]
    assert identity["ok"] is False
    assert any(v["module"].split(".")[0] == "rocisa" for v in identity["violations"])
    assert data["repetitions"] == []


def test_a_tensilelite_module_loaded_from_outside_the_source_root_fails_the_run(tmp_path):
    root = tmp_path / "other"
    package = _copy_tree(root)
    stray = package / "Utilities" / "Decorators" / "Timing.py"
    stray.unlink()
    os.symlink(_PACKAGE_DIR / "Utilities" / "Decorators" / "Timing.py", stray)
    proc, data = _measure(tmp_path, root, _rocisa_build_of(root, tmp_path / "rocisa_build"), 1)
    assert proc.returncode != 0
    assert data is not None, "driver wrote no JSON"
    assert data["import_identity"]["ok"] is False
    assert [v["module"] for v in data["import_identity"]["violations"]] == [
        "tensilelite.Utilities.Decorators.Timing"]
    assert data["repetitions"] == []


def test_a_rocisa_built_from_another_tree_is_refused(tmp_path):
    """The rocisa build must come from --source-root, or trees would mix."""
    root = tmp_path / "other"
    _copy_tree(root)
    proc, data = _measure(tmp_path, root, _rocisa_dir(), 1)
    assert proc.returncode != 0
    assert "built from" in proc.stderr
    assert data["import_identity"]["ok"] is False
    assert [v["module"] for v in data["import_identity"]["violations"]] == ["rocisa._build_info"]
    assert data["repetitions"] == []


def _assert_source_identity(source, root):
    assert source["scope"] == str(root / "projects" / "hipblaslt")
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel", "HEAD"],
                          capture_output=True, text=True)
    lines = head.stdout.split()
    if head.returncode == 0 and os.path.realpath(lines[0]) == os.path.realpath(root):
        assert source["git_head"] == lines[1]
        assert _HEX64.match(source["git_diff_sha256"]) and source["content_sha256"] is None
    else:
        assert source["git_head"] is None and _HEX64.match(source["content_sha256"])


def _assert_rocisa_identity(rocisa, build):
    package = build / "rocisa"
    info = runpy.run_path(str(package / "_build_info.py"))
    assert rocisa["source_root"] == info["SOURCE_ROOT"]
    assert rocisa["stinkytofu_source_root"] == info.get("STINKYTOFU_SOURCE_ROOT")
    files = sorted(p for p in package.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    assert rocisa["files"] == {str(p.relative_to(package)): _sha256(p) for p in files}


def _cold_out(tmp_path) -> Path:
    """A TensileCreateLibrary output directory whose library/ holds one (fake) code object."""
    out = tmp_path / "tcl_out"
    (out / "library" / _ARCH).mkdir(parents=True, exist_ok=True)
    (out / "library" / _ARCH / f"Kernels.so-000-{_ARCH}.hsaco").write_bytes(b"code object")
    return out


def _run_parser(tmp_path, text, source_root=_SOURCE_ROOT, rocisa=None, logic=_LOGIC_DIR, out=None):
    log, js = tmp_path / "timing.log", tmp_path / "timing.json"
    log.write_text(text)
    js.unlink(missing_ok=True)
    proc = _run_driver("--parse-timing-log", log, "--source-root", source_root,
                       "--rocisa", rocisa or _rocisa_dir(), "--logic", logic,
                       "--out", out or _cold_out(tmp_path), "--json", js, cache_dir=tmp_path / "unused")
    return proc, js, log


def _parse_log(tmp_path, text, source_root=_SOURCE_ROOT, rocisa=None, logic=_LOGIC_DIR, out=None):
    proc, js, log = _run_parser(tmp_path, text, source_root, rocisa, logic, out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    parsed = json.loads(js.read_text())
    assert "fingerprint" in parsed, "parser output carries no fingerprint"
    return parsed, log


def test_print_timing_parser_reads_every_record(tmp_path):
    text = "\n".join([
        "generateLogicDataAndSolutions took 1.5 seconds",
        "Time to load yaml files (s): 1.50",
        "generateKernelObjectsFromSolutions took 0.25 seconds",
        "copyStaticFiles took 0.125 seconds",
        "# Helper kernel cache MISS (0123456789ab...)",
        "buildSourceCodeObjectFile time (s): 3.25",
        "Time to generate kernels (s): 7.75",
        "Time to pass kernel info to library (s): 0.01",
        "Time to write master solution libraries (s): 0.02",
        "TIMING:python_kernel_codegen:12.500",
        "Total time (s): 10.00",
        "Total kernels processed: 3",
        "Kernels processed per second: 0.30",
        "KernelHelperObjs: 8",
    ])
    parsed, log = _parse_log(tmp_path, text)
    fingerprint = parsed.pop("fingerprint", None)
    assert parsed == {
        "decorated_s": {
            "generateLogicDataAndSolutions": [1.5],
            "generateKernelObjectsFromSolutions": [0.25],
            "copyStaticFiles": [0.125],
        },
        "phases_s": {
            "load_yaml": 1.5,
            "generate_kernels": 7.75,
            "pass_kernel_info": 0.01,
            "write_master_libraries": 0.02,
            "helper_build": 3.25,
            "total": 10.0,
        },
        "timing_records_ms": {"python_kernel_codegen": [12.5]},
        "helper_cache": "miss",
        "kernels_processed": 3,
        "kernel_helper_objs": 8,
    }
    assert fingerprint is not None, "parser output carries no fingerprint"
    assert fingerprint["inputs"] == [{"path": str(f), "sha256": _sha256(f)}
                                     for f in sorted(_LOGIC_DIR.glob("*.yaml"))] + [
                                    {"path": str(log), "sha256": _sha256(log)}]
    library = tmp_path / "tcl_out" / "library"
    assert fingerprint["artifacts"] == [{"path": str(p), "sha256": _sha256(p)}
                                        for p in sorted(library.rglob("*")) if p.is_file()]
    assert fingerprint["solution_index_map"] == {}
    assert fingerprint["driver"]["sha256"] == _sha256(_DRIVER_FILE)
    _assert_source_identity(fingerprint["source"], _SOURCE_ROOT)
    _assert_rocisa_identity(fingerprint["rocisa"], _rocisa_dir())
    assert fingerprint["toolchain"]["components"] == []
    assert fingerprint["toolchain"]["python"] == sys.version


def test_the_cold_parser_needs_the_tree_identity(tmp_path):
    log = tmp_path / "timing.log"
    log.write_text("Total time (s): 1.00\n")
    proc = _run_driver("--parse-timing-log", log, "--json", tmp_path / "t.json",
                       cache_dir=tmp_path / "unused")
    assert proc.returncode != 0 and "--source-root" in proc.stderr
    assert not (tmp_path / "t.json").exists()


def _fake_tree(root: Path, built_from=None) -> Path:
    """A projects/hipblaslt tree plus a rocisa package (built from it), enough to fingerprint."""
    built_from = built_from or root / "projects/hipblaslt/tensilelite/rocisa/rocisa"
    for rel, text in {"projects/hipblaslt/tensilelite/tensilelite/__init__.py": "",
                      "projects/hipblaslt/library/logic.yaml": "a: 1\n",
                      "rocisa_build/rocisa/__init__.py": "",
                      "rocisa_build/rocisa/_build_info.py":
                          f"SOURCE_ROOT = {str(built_from)!r}\n"
                          "STINKYTOFU_SOURCE_ROOT = 'st'\nBUILD_DIR = 'b'\n"}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root / "rocisa_build"


def test_the_cold_parser_refuses_a_missing_source_root(tmp_path):
    proc, js, _ = _run_parser(tmp_path, "Total time (s): 1.00\n", source_root=tmp_path / "nowhere")
    assert proc.returncode != 0 and "--source-root" in proc.stderr
    assert not js.exists()


def test_the_cold_parser_refuses_a_rocisa_built_from_another_tree(tmp_path):
    root = tmp_path / "tree"
    rocisa = _fake_tree(root, built_from=tmp_path / "elsewhere" / "rocisa")
    proc, js, _ = _run_parser(tmp_path, "Total time (s): 1.00\n", root, rocisa)
    assert proc.returncode != 0 and "built from" in proc.stderr
    assert not js.exists()


def test_the_cold_parser_refuses_an_output_without_a_library(tmp_path):
    proc, js, _ = _run_parser(tmp_path, "Total time (s): 1.00\n", out=tmp_path / "no_tcl_out")
    assert proc.returncode != 0 and "--out" in proc.stderr
    assert not js.exists()


def test_the_cold_parser_refuses_an_output_with_an_empty_library(tmp_path):
    out = tmp_path / "empty_tcl_out"
    (out / "library").mkdir(parents=True)
    proc, js, _ = _run_parser(tmp_path, "Total time (s): 1.00\n", out=out)
    assert proc.returncode != 0 and "--out" in proc.stderr
    assert not js.exists()


def test_the_cold_parser_refuses_a_missing_logic_directory(tmp_path):
    proc, js, _ = _run_parser(tmp_path, "Total time (s): 1.00\n", logic=tmp_path / "no_such_logic")
    assert proc.returncode != 0 and "--logic" in proc.stderr
    assert not js.exists()


def test_the_cold_parser_refuses_a_logic_directory_without_logic_files(tmp_path):
    (tmp_path / "empty_logic").mkdir()
    proc, js, _ = _run_parser(tmp_path, "Total time (s): 1.00\n", logic=tmp_path / "empty_logic")
    assert proc.returncode != 0 and "--logic" in proc.stderr
    assert not js.exists()


def test_a_git_tree_whose_diff_fails_is_refused(tmp_path):
    """A repository without HEAD cannot be diffed; that must not read as clean."""
    root = tmp_path / "tree"
    rocisa = _fake_tree(root)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=test", "-c", "user.email=test@invalid",
                    "init", "-q"], check=True)
    proc, js, _ = _run_parser(tmp_path, "Total time (s): 1.00\n", root, rocisa)
    assert proc.returncode != 0 and "error: git" in proc.stderr
    assert not js.exists()


def test_a_git_tree_with_an_unreadable_index_is_refused(tmp_path):
    """HEAD resolves but the diff fails: that must not read as clean either."""
    root = tmp_path / "tree"
    rocisa = _fake_tree(root)
    git = lambda *a: subprocess.run(["git", "-C", str(root), "-c", "user.name=test",
                                     "-c", "user.email=test@invalid", *a], check=True, capture_output=True)
    git("init", "-q")
    git("add", "projects")
    git("commit", "-q", "-m", "base")
    (root / ".git" / "index").write_bytes(b"not an index")
    proc, js, _ = _run_parser(tmp_path, "Total time (s): 1.00\n", root, rocisa)
    assert proc.returncode != 0 and "error: git" in proc.stderr
    assert not js.exists()


def test_source_fingerprint_covers_all_of_projects_hipblaslt(tmp_path):
    """A change outside tensilelite/tensilelite changes the fingerprint; build outputs do not."""
    root = tmp_path / "tree"
    rocisa = _fake_tree(root)
    fp = lambda: _parse_log(tmp_path, "Total time (s): 1.00\n", root, rocisa)[0]["fingerprint"]
    first = fp()
    assert first["source"]["git_head"] is None and _HEX64.match(first["source"]["content_sha256"])
    (root / "projects/hipblaslt/library/logic.yaml").write_text("a: 2\n")
    edited = fp()
    assert edited["source"]["content_sha256"] != first["source"]["content_sha256"]
    (root / "projects/hipblaslt/build/out").mkdir(parents=True)
    (root / "projects/hipblaslt/build/out/x.o").write_text("object")
    assert fp()["source"] == edited["source"]


def test_source_fingerprint_of_a_git_tree_covers_uncommitted_changes(tmp_path):
    root = tmp_path / "tree"
    rocisa = _fake_tree(root)
    git = lambda *a: subprocess.run(["git", "-C", str(root), "-c", "user.name=test",
                                     "-c", "user.email=test@invalid", *a],
                                    check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    git("add", "projects")
    git("commit", "-q", "-m", "base")
    fp = lambda: _parse_log(tmp_path, "Total time (s): 1.00\n", root, rocisa)[0]["fingerprint"]["source"]
    clean = fp()
    assert clean["git_head"] == git("rev-parse", "HEAD") and clean["content_sha256"] is None
    (root / "projects/hipblaslt/library/logic.yaml").write_text("a: 2\n")
    dirty = fp()
    assert dirty["git_head"] == clean["git_head"] and dirty["git_diff_sha256"] != clean["git_diff_sha256"]
    (root / "projects/hipblaslt/library/new.yaml").write_text("b: 1\n")
    untracked = fp()
    (root / "projects/hipblaslt/library/new.yaml").write_text("b: 2\n")
    assert len({dirty["git_diff_sha256"], untracked["git_diff_sha256"], fp()["git_diff_sha256"]}) == 3


def test_rocisa_fingerprint_covers_its_python_files(tmp_path):
    root = tmp_path / "tree"
    rocisa = _fake_tree(root)
    fp = lambda: _parse_log(tmp_path, "Total time (s): 1.00\n", root, rocisa)[0]["fingerprint"]["rocisa"]
    first = fp()
    assert first["stinkytofu_source_root"] == "st"
    assert first["source_root"] == str(root / "projects/hipblaslt/tensilelite/rocisa/rocisa")
    (rocisa / "rocisa" / "__init__.py").write_text("# changed\n")
    assert fp()["files"]["__init__.py"] != first["files"]["__init__.py"]


def test_print_timing_parser_reports_a_helper_cache_hit(tmp_path):
    parsed, _ = _parse_log(tmp_path, "buildSourceCodeObjectFile time (s): 0.01  [cache hit]\n")
    assert parsed["helper_cache"] == "hit"
    assert parsed["phases_s"] == {"helper_build": 0.01}


def test_print_timing_parser_reads_a_real_cold_run(tmp_path, tcl, measured):
    tcl_out, log = tcl
    parsed, _ = _parse_log(tmp_path, log.read_text(), out=tcl_out)
    for name in ("generateLogicDataAndSolutions", "generateKernelObjectsFromSolutions", "copyStaticFiles"):
        assert len(parsed["decorated_s"].get(name, [])) == 1, name
        assert parsed["decorated_s"][name][0] >= 0.0
    phases = parsed["phases_s"]
    for name in ("load_yaml", "generate_kernels", "pass_kernel_info", "write_master_libraries", "helper_build"):
        assert name in phases and 0.0 <= phases[name] <= phases["total"], name
    assert parsed["kernels_processed"] == 1
    assert parsed["kernel_helper_objs"] > 0
    assert parsed["helper_cache"] == "miss"
    toolchain = parsed["fingerprint"]["toolchain"]
    cxx = re.search(r"Component path: (\S*clang\+\+) version", log.read_text()).group(1)
    assert toolchain["cxx_compiler"] == cxx and "clang" in toolchain["cxx_compiler_version"]
    assert len(toolchain["components"]) >= 2
    fingerprint = parsed["fingerprint"]
    assert {"artifacts", "solution_index_map"} <= set(fingerprint), sorted(fingerprint)
    library = tcl_out / "library"
    assert fingerprint["artifacts"] == [{"path": str(p), "sha256": _sha256(p)}
                                        for p in sorted(library.rglob("*")) if p.is_file()]
    # The same build-local map the measured run of the same logic reports.
    assert fingerprint["solution_index_map"] == measured["fingerprint"]["solution_index_map"]

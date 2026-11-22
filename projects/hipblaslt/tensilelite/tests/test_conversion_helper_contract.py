# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Manual host/device integration checks for conversion helper names.

This file is not part of any CI target or pytest test path; run it by hand. The second
test needs an external single-solution launch harness that this repository does not ship.

The host probe links current production naming code, loads a generated library,
and reports invocation symbols. Dropping a token for only activation-none breaks
this test even when stock helpers are present. No production method is mocked.
The host naming rules themselves are covered without these inputs by
ConversionKernelName.* in tensilelite-tests.

Requirements (each test is skipped when one is missing):
- One visible AMD GPU (``rocminfo`` reports a GPU agent): both tests load the
  generated library through HIP, the host probe included.
- TENSILELITE_CONTRACT_HOST_BUILD: a configured hipBLASLt build tree with
  compile_commands.json and the tensilelite-tests target.
- TENSILELITE_CONTRACT_HARNESS (second test only): an executable that launches
  one solution index of a library (``--solution-index``, ``--call``,
  ``--pointer-modes``, ``--paths``, ``--out``) and writes a JSON report with the
  launch status and numerical verdict.
"""
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import zlib

import msgpack
import pytest
import yaml

TL = Path(__file__).resolve().parents[1]
REPO = TL.parents[2]
CODEGEN = TL / "tensilelite/Tests/unit/characterization/_codegen"
sys.path.insert(0, str(CODEGEN))
sys.path.insert(0, str(TL))
from bespoke_codegen_common import DATA, compile_helpers, disassemble  # noqa: E402
from test_bespoke_codegen_slowdown import FAMILIES  # noqa: E402

ACTIVATIONS = ("none", "relu", "gelu", "sigmoid")
RECIPE = {"StreamK": 0, "GlobalSplitU": 2, "GlobalSplitUAlgorithm": "MultipleBuffer",
          "WorkGroupMappingXCC": 1, "StoreRemapVectorWidth": 0}


def run_checked(argv, log, *, cwd=None, library_dir=None):
    result = subprocess.run([str(arg) for arg in argv], cwd=cwd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=dict(os.environ, HIPBLASLT_TENSILE_LIBPATH=str(library_dir))
                            if library_dir is not None else None)
    log.write_text(result.stdout)
    log.with_suffix(".command.json").write_text(json.dumps(
        {"argv": [str(arg) for arg in argv], "cwd": str(cwd) if cwd else None,
         "exit_code": result.returncode}, indent=2) + "\n")
    assert result.returncode == 0, f"command failed ({result.returncode}); see {log}"
    return result.stdout


def gpu_visible():
    rocminfo = shutil.which("rocminfo") or "/opt/rocm/bin/rocminfo"
    try:
        out = subprocess.run([rocminfo], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
    except OSError:
        return False
    return any(line.split(":", 1)[-1].strip() == "GPU" for line in out.splitlines() if "Device Type" in line)


@pytest.fixture(scope="module")
def host_probe(tmp_path_factory):
    configured = os.environ.get("TENSILELITE_CONTRACT_HOST_BUILD")
    if not configured:
        pytest.skip("TENSILELITE_CONTRACT_HOST_BUILD is not set (configured hipBLASLt build tree)")
    if not gpu_visible():
        pytest.skip("no AMD GPU visible (rocminfo reports no GPU agent); the host probe loads the library through HIP")
    build = Path(configured).resolve()
    commands = json.loads((build / "compile_commands.json").read_text())
    out = tmp_path_factory.mktemp("conversion_host_probe")
    objects = []
    for suffix, source, basename in (
        ("src/ContractionSolution.cpp", TL / "src/ContractionSolution.cpp", "solution"),
        ("src/KernelArguments.cpp", TL / "src/KernelArguments.cpp", "arguments"),
        ("src/ContractionSolution.cpp", TL / "tests/ConversionHelperContract_probe.cpp", "probe"),
    ):
        rule = next(record for record in commands if record["file"].endswith(suffix))
        old_root = rule["file"].split("/projects/hipblaslt/")[0]
        argv = [arg.replace(old_root, str(REPO)) for arg in shlex.split(rule["command"])]
        obj = out / (basename + ".o")
        argv[argv.index("-c") + 1] = str(source)
        argv[argv.index("-o") + 1] = str(obj)
        run_checked(argv, out / (basename + ".log"), cwd=build)
        objects.append(str(obj))
    link_dir = build / "tensilelite/tests"
    link = shlex.split((link_dir / "CMakeFiles/tensilelite-tests.dir/link.txt").read_text())
    link = [arg for arg in link if not arg.endswith(".cpp.o")
            and not arg.endswith("libgtest.a") and arg != "-lgtest"]
    binary = out / "conversion-host-probe"
    link[link.index("-o") + 1] = str(binary)
    link[1:1] = objects
    run_checked(link, out / "link.log", cwd=link_dir)
    return binary


def write_logic(directory, family, activation, flags, bespoke, number):
    recipe, _, _, bias, bias_types, sav = family
    beta, grouped, scale_cd = flags
    logic = yaml.safe_load((DATA / recipe).read_text())
    logic["ProblemType"].update(UseBeta=beta, GroupedGemm=grouped,
                                ActivationType=activation, UseBias=bias,
                                BiasDataTypeList=bias_types, UseScaleCD=scale_cd,
                                UseScaleAlphaVec=sav)
    logic["Solutions"][0].update(RECIPE, BespokeNarrowing=bespoke)
    path = directory / f"{number:03d}_gfx950_{'bespoke' if bespoke else 'stock'}.yaml"
    path.write_text(yaml.safe_dump(logic, sort_keys=False))
    return path


@pytest.fixture(scope="module")
def conversion_library(tmp_path_factory):
    out = tmp_path_factory.mktemp("conversion_contract")
    logic_dir = out / "logic"
    logic_dir.mkdir()
    stock, manifest, seen = [], [], set()

    def add(family, activation, flags, bespoke):
        dtype = family[1].split("_")[0][1:]
        key = (int(bespoke), *map(int, flags[:2]), family[3], int(flags[2]), family[5],
               dtype, activation)
        if key in seen:
            return
        seen.add(key)
        path = write_logic(logic_dir, family, activation, flags, bespoke, len(manifest))
        manifest.append(key)
        if not bespoke:
            stock.append(path)

    for family in FAMILIES:
        for flags in itertools.product((False, True), repeat=3):
            # Grouped GEMM rejects beta-free solutions before helper generation;
            # its emitted but unroutable Beta0 siblings are covered by ISA tests.
            if flags[1] and not flags[0]:
                continue
            add(family, "hipblaslt_all", flags, False)
            add(family, "hipblaslt_all", flags, True)
            # Real library validation disallows single activation with GG or
            # ScaleCD. Those modes exercise their actual general-activation
            # helper with none/relu/gelu/sigmoid runtime call facts instead.
            if family[2] == "none" or not (flags[1] or flags[2]):
                add(family, family[2], flags, True)
    # Genuine legacy helpers distinguish a wrong stock lookup from refusal.
    add(FAMILIES[0], "none", (True, False, False), False)
    built = out / "built"
    run_checked([sys.executable, "-m", "tensilelite.TensileCreateLibrary",
                 "--architecture=gfx950", "--cxx-compiler=/opt/rocm/bin/amdclang++",
                 "--jobs=1", "--no-compress",
                 "--keep-build-tmp", str(logic_dir), str(built), "HIP"], out / "tcl.log", cwd=TL)
    directory = built / "library/gfx950"
    documents = []
    for path in directory.glob("TensileLibrary*.dat*"):
        data = path.read_bytes()
        if path.suffix == ".zlib":
            data = zlib.decompress(data)
        document = msgpack.unpackb(data, raw=False, strict_map_key=False)
        if document.get("solutions"):
            documents.append((path, document))
    assert documents, "lazy fixture must serialize its solution shards"
    (library,) = directory.glob("TensileLibrary_lazy_gfx950.dat*")
    (obj,) = directory.glob("Kernels.so-*.hsaco")
    return {"out": out, "built": built, "directory": directory, "library": library,
            "documents": documents, "object": obj, "symbols": set(disassemble(obj)),
            "stock_logic": stock, "manifest": manifest}


@pytest.fixture(scope="module")
def host_invocations(host_probe, conversion_library):
    case = conversion_library
    output = run_checked([host_probe, case["library"]], case["out"] / "host-names.tsv", cwd=TL)
    rows = [line.split("\t") for line in output.splitlines() if line.startswith(tuple("0123456789"))]
    assert rows and all(len(row) == 14 for row in rows), "host probe produced no complete invocation records"
    return rows


def test_serialized_host_invocations_resolve_emitted_helpers(host_invocations, conversion_library):
    # Preserve the marker/name boundary, not two independently guessed name sets.
    case = conversion_library
    assert sum(len(document["solutions"]) for _, document in case["documents"]) == len(case["manifest"]), \
        "real matrix lost a serialized solution"
    expected = {}
    for *fields, activation in case["manifest"]:
        runtime_activations = ACTIVATIONS if activation == "hipblaslt_all" else (activation,)
        for runtime_activation in runtime_activations:
            key = (*fields, activation, runtime_activation)
            expected[key] = set(itertools.product((2, 4, 8, 16), (1, 2, 4)))
    indices, observed = set(), {}
    for row in host_invocations:
        index, bespoke, beta, gg, bias, cd, sav, accumulation = map(int, row[:8])
        dtype, activation, runtime_activation, gsu, vw, name = row[8:]
        assert accumulation == 2, "fixture must exercise the actual MultipleBuffer helper route"
        key = (bespoke, beta, gg, bias, cd, sav, dtype, activation, runtime_activation)
        indices.add(index)
        observed.setdefault(key, set()).add((int(gsu), int(vw)))
        token = "_Bespoke_" if beta else "_Beta0_"
        if bespoke:
            assert token in name, f"serialized bespoke solution {index} requested a stock helper: {name}"
        else:
            assert "_Bespoke_" not in name and "_Beta0_" not in name, f"stock lookup changed: {name}"
        assert name in case["symbols"], f"serialized solution {index} host invocation has no emitted helper: {name}"
    assert observed == expected, "host invocation epilogue/marker/role matrix is incomplete"
    assert len(indices) == len(case["manifest"]), "host did not check every serialized solution"
    assert len(host_invocations) == sum(map(len, expected.values())), "host duplicated or omitted an invocation"


def test_missing_bespoke_helper_cannot_launch_stock_fallback(host_invocations, conversion_library):
    # Returning the missing-helper error makes this pass. Dropping only the
    # activation-none host token makes the stock-only copy run a legacy helper
    # successfully instead, so the error assertion fails despite correct numbers.
    configured = os.environ.get("TENSILELITE_CONTRACT_HARNESS")
    if not configured:
        pytest.skip("TENSILELITE_CONTRACT_HARNESS is not set (single-solution launch harness)")
    harness = Path(configured)
    case = conversion_library
    candidates = [row for row in host_invocations
                  if row[1:8] == ["1", "1", "0", "0", "0", "0", "2"]
                  and row[8:11] == ["B", "none", "none"]]
    indices = {int(row[0]) for row in candidates}
    assert len(indices) == 1, "fixture must have one beta-kept, plain bf16 bespoke solution"
    index, = indices
    required = {row[-1] for row in candidates}
    assert len(required) == 12 and all("_Bespoke_" in name for name in required)

    stock_logic = next(path for path in case["stock_logic"]
                       if yaml.safe_load(path.read_text())["ProblemType"]["ActivationType"] == "none")
    compiled = case["out"] / "stock-only-helpers"
    stock, _ = compile_helpers(stock_logic, compiled)
    assert all("_Bespoke_" not in name and "_Beta0_" not in name for name in stock)
    assert {name.replace("_Bespoke_", "_") for name in required} <= set(stock), \
        "negative control must contain every corresponding legacy stock helper"
    assert required.isdisjoint(stock), "negative control unexpectedly contains a bespoke helper"
    stock_objects = list(compiled.rglob("Kernels.so-*.hsaco"))
    assert len(stock_objects) == 1
    root = case["directory"]
    missing_root = case["out"] / "stock-only-library"
    shutil.copytree(root, missing_root)
    replacement = missing_root / case["object"].relative_to(root)
    shutil.copyfile(stock_objects[0], replacement)
    for path in root.rglob("*"):
        if path.is_file() and path != case["object"]:
            assert path.read_bytes() == (missing_root / path.relative_to(root)).read_bytes(), \
                "negative control changed metadata or a main-kernel code object"

    call = "transA=T,transB=N,M=128,N=64,K=256,batch=1,a_type=bf16,b_type=bf16,c_type=bf16,d_type=bf16,compute_type=f32,alpha=1,beta=1,bias=none,activation=none,sav=0"

    def launch(label, directory):
        output = case["out"] / f"{label}.json"
        run_checked([harness, "--solution-index", index, "--call", call,
                     "--pointer-modes", "host", "--paths", "matmul_checked", "--out", output],
                    case["out"] / f"{label}.log", library_dir=directory)
        report = json.loads(output.read_text())
        assert report["complete"] and report["solution_index"] == index
        records = [row for row in report["paths"] if row["path"] == "matmul_checked"]
        assert len(records) == 1
        return report, records[0]

    positive, record = launch("positive-helper-control", root)
    assert record["outcome"] == "ran" and record["status"] == "HIPBLAS_STATUS_SUCCESS", \
        "the same solution must launch successfully with its bespoke helper present"
    numerics = record["numerics"]
    assert numerics["verdict"] == "match" and math.isfinite(numerics["norm_error"])
    assert numerics["norm_error"] < numerics["tolerance"]
    assert numerics["d_written"] == numerics["elements"] == 128 * 64
    missing, record = launch("missing-helper-control", missing_root)
    assert missing["library"]["kernel_name"] == positive["library"]["kernel_name"]
    assert missing["library"]["solution_name"] == positive["library"]["solution_name"]
    assert record["outcome"] == "error" and record["status"] != "HIPBLAS_STATUS_SUCCESS", \
        "a missing bespoke helper must return an error, never launch the legacy stock helper"
    assert any(name in line for name in required for line in record["library_log"]), \
        "negative control must fail for the requested bespoke helper, not an unrelated setup error"

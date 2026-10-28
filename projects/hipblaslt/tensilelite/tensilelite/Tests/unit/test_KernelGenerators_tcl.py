# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""End-to-end TensileCreateLibrary runs on a logic file that holds generator kernels."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tensilelite" / "bin" / "TensileCreateLibrary"
FIXTURE_DIR = Path(__file__).resolve().parent / "test_data" / "skinny_gemm_logic"
ORIGAMI_LOGIC = (
    ROOT.parent
    / "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/gfx950/Origami"
    / "gfx950_Cijk_Alik_Bljk_F8BS_MXAE8B32_MXBE8B32_BH_Bias_HA_S_SAB_SAV_UserArgs.yaml"
)
KERNEL_BASE_NAMES = (
    "gemm_BM1_TN1_LANES_K64_WAVES4_UP_ITERS2_PARTIAL_STEPS0_PF2_GROUP1_NUM_XCDS1_GRID_3D1",
    "gemm_BM1_TN1_LANES_K16_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF2_GROUP1_NUM_XCDS1_GRID_3D1",
    "gemm_BM2_TN2_LANES_K16_WAVES4_UP_ITERS0_PARTIAL_STEPS1_PF1_GROUP1_NUM_XCDS1_GRID_3D1",
    "gemm_BM1_TN1_LANES_K64_WAVES4_UP_ITERS2_PARTIAL_STEPS0_PF2_GROUP1_NUM_XCDS1_GRID_3D0",
    "gemm_BM1_TN2_LANES_K16_WAVES2_UP_ITERS3_PARTIAL_STEPS1_PF2_GROUP2_NUM_XCDS8_GRID_3D0",
)
# Every knob set of the fixture is in both useBeta variants.
KERNEL_NAMES = {f"{name}_USE_BETA{useBeta}" for name in KERNEL_BASE_NAMES for useBeta in (0, 1)}
# EI_ABIVERSION is the code object version minus 2.
ABI_V4 = 2
ABI_V6 = 4
EI_OSABI_AMDGPU_HSA = 64


def importedCodegenDirectory():
    try:
        import tensilelite_codegen
    except ImportError:
        return None
    return Path(tensilelite_codegen.__file__).resolve().parent


def llvmReadelf():
    compiler = shutil.which("amdclang++")
    if compiler is None:
        return None
    return shutil.which("llvm-readelf", path=str(Path(compiler).resolve().parent))


BINDINGS = importedCodegenDirectory()
READELF = llvmReadelf()
pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(BINDINGS is None, reason="the tensilelite_codegen module is not importable"),
    pytest.mark.skipif(READELF is None, reason="the ROCm assembler tools are not available"),
]


def runTcl(workDir, logicDir, *options, withBindings=True):
    pythonPath = [str(ROOT)]
    if withBindings:
        pythonPath.append(str(BINDINGS))
    inherited = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    # The bindings may reach this process through PYTHONPATH; the missing-module run must not see them.
    pythonPath += [path for path in inherited if path and (withBindings or Path(path).resolve() != BINDINGS)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(pythonPath), "PYTHONDONTWRITEBYTECODE": "1"}
    command = [
        sys.executable, str(SCRIPT), str(logicDir), "out", "HIP",
        "--architecture", "gfx950", "--no-generate-solution-table", *options,
    ]
    workDir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, cwd=workDir, env=env, capture_output=True, text=True)


def libraryDirectory(workDir):
    return workDir / "out" / "library" / "gfx950"


def abiVersion(codeObject):
    header = codeObject.read_bytes()[:9]
    assert header[:4] == b"\x7fELF"
    assert header[7] == EI_OSABI_AMDGPU_HSA
    return header[8]


def kernelSymbols(codeObject):
    output = subprocess.run(
        [READELF, "-s", str(codeObject)], capture_output=True, text=True, check=True
    ).stdout
    names = {column for line in output.splitlines() for column in line.split() if column.startswith("gemm_BM")}
    return {name for name in names if not name.endswith(".kd")}, {name[:-3] for name in names if name.endswith(".kd")}


def assertGeneratedKernels(codeObject):
    assert abiVersion(codeObject) == ABI_V6
    functions, descriptors = kernelSymbols(codeObject)
    assert functions == KERNEL_NAMES
    assert descriptors == KERNEL_NAMES


@pytest.fixture(scope="module")
def lazyRun(tmp_path_factory):
    workDir = tmp_path_factory.mktemp("lazy")
    result = runTcl(workDir, FIXTURE_DIR, "--library-format", "msgpack", "--no-compress")
    assert result.returncode == 0, result.stdout + result.stderr
    return libraryDirectory(workDir)


def test_lazy_library_has_a_separate_generated_code_object(lazyRun):
    files = sorted(path.name for path in lazyRun.iterdir())
    assert "TensileLibrary_lazy_gfx950.dat" in files or "TensileLibrary_lazy_gfx950.dat.zlib" in files
    generated = [name for name in files if name.endswith("_ID75a0_Generated_gfx950.co")]
    assert len(generated) == 1
    placeholder = generated[0].replace("_Generated_gfx950.co", "_gfx950")
    assert any(name.startswith(placeholder + ".dat") for name in files)
    # Every kernel of the fixture is generated, so no regular code object exists for the placeholder.
    assert placeholder + ".co" not in files
    assertGeneratedKernels(lazyRun / generated[0])


def test_yaml_library_names_the_generated_code_object(tmp_path):
    result = runTcl(tmp_path, FIXTURE_DIR, "--library-format", "yaml", "--no-compress")
    assert result.returncode == 0, result.stdout + result.stderr
    library = libraryDirectory(tmp_path)
    placeholders = [path for path in library.glob("TensileLibrary_*_ID75a0_gfx950.yaml")]
    assert len(placeholders) == 1
    text = placeholders[0].read_text()
    generatedName = placeholders[0].name.replace("_gfx950.yaml", "_Generated_gfx950.co")
    assert text.count(f"codeObjectFile: {generatedName}") == len(KERNEL_NAMES)
    assert "generator" not in text
    assert (library / generatedName).is_file()


def test_non_lazy_library_names_the_generated_code_object(tmp_path):
    result = runTcl(
        tmp_path, FIXTURE_DIR, "--library-format", "yaml", "--no-lazy-library-loading", "--no-compress"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    library = libraryDirectory(tmp_path)
    assertGeneratedKernels(library / "TensileLibrary_Generated_gfx950.co")
    text = (library / "TensileLibrary_gfx950.yaml").read_text()
    assert text.count("codeObjectFile: TensileLibrary_Generated_gfx950.co") == len(KERNEL_NAMES)


@pytest.mark.skipif(not ORIGAMI_LOGIC.is_file(), reason="the Origami MX logic file is not present")
def test_generated_and_regular_kernels_share_a_placeholder(tmp_path):
    logicDir = tmp_path / "logic"
    shutil.copytree(FIXTURE_DIR, logicDir)
    shutil.copy(ORIGAMI_LOGIC, logicDir)
    result = runTcl(tmp_path / "run", logicDir, "--library-format", "msgpack", "--no-compress")
    assert result.returncode == 0, result.stdout + result.stderr
    library = libraryDirectory(tmp_path / "run")
    generated = list(library.glob("*_Generated_gfx950.co"))
    assert len(generated) == 1
    regular = library / generated[0].name.replace("_Generated_gfx950.co", "_gfx950.co")
    assert regular.is_file()
    assert abiVersion(regular) == ABI_V4
    assert not any(name.startswith("gemm_BM") for name in kernelSymbols(regular)[0] | kernelSymbols(regular)[1])
    assertGeneratedKernels(generated[0])


def test_missing_codegen_module_fails_the_build(tmp_path):
    result = runTcl(tmp_path, FIXTURE_DIR, "--no-compress", withBindings=False)
    assert result.returncode != 0
    assert "tensilelite_codegen" in result.stdout + result.stderr

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only production derivation, compilation and ISA inspection for narrowing."""
import hashlib
import re
import subprocess
from pathlib import Path

import yaml

import codegen_harness as H
from measure_generation import _elfSections

DATA = Path(__file__).parent / "data/test_data/_logic/bespoke_codegen"


def variant(recipe, destination, *, problem=None, solution=None):
    logic = yaml.safe_load((DATA / recipe).read_text())
    logic["ProblemType"].update(problem or {})
    assert len(logic["Solutions"]) == 1
    logic["Solutions"][0].update(solution or {})
    destination.write_text(yaml.safe_dump(logic, sort_keys=False))
    return destination


def disassemble(path):
    # Use the same LLVM installation as the production assembler/compiler.
    objdump = Path(H.get_assembler().path).resolve().parent / "llvm-objdump"
    result = subprocess.run([str(objdump), "--disassemble", str(path)],
                            text=True, capture_output=True, check=True)
    path.with_suffix(".dis").write_text(result.stdout)
    symbols = {}
    current = None
    for line in result.stdout.splitlines():
        header = re.fullmatch(r"[0-9a-fA-F]+ <([^>]+)>:", line)
        if header:
            current = header[1]
            symbols[current] = []
        elif current is not None and "//" in line:
            instruction, encoding = line.split("//", 1)
            address = re.match(r"\s*([0-9A-Fa-f]+):", encoding)
            if address and instruction.strip():
                assert not instruction.lstrip().startswith("."), "LLVM could not decode an instruction"
                symbols[current].append((int(address[1], 16), " ".join(instruction.split())))
    assert symbols and all(symbols.values()), "compiled object has no disassembled instructions"
    return symbols


def text_digest(path):
    data = path.read_bytes()
    _, offset, size = _elfSections(data)[".text"]
    assert size > 0
    return hashlib.sha256(data[offset:offset + size]).hexdigest()


def assembly_digest(instructions):
    # Only the objdump decoration (addresses/raw encodings) is removed. Operands,
    # immediates, instructions and their order remain byte-pinned.
    return hashlib.sha256("\n".join(text for _, text in instructions).encode()).hexdigest()


def compile_helpers(path, out):
    from tensilelite.TensileCreateLibrary import Run
    from tensilelite.Toolchain.Source import makeSourceToolchain, buildSourceCodeObjectFiles
    from tensilelite.Toolchain.Validators import ToolchainDefaults, validateToolchain

    out.mkdir(parents=True)
    with H._isolated_globals():
        solutions = H._solutions_from_logic_unguarded(path)
        assert len(solutions) == 1 and solutions[0]["Valid"]
        helpers = Run.generateKernelHelperObjects(solutions, str(H.get_assembler().path),
                                                 H.get_isa_info_map())
        assert helpers, "normal derivation emitted no helpers"
        Run.copyStaticFiles(out)
        Run.writeHelpers(out, helpers, "Kernels.cpp", "Kernels.h")
        toolchain = makeSourceToolchain(validateToolchain("amdclang++"),
                                       validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER))
        objects = buildSourceCodeObjectFiles(toolchain.compiler, toolchain.bundler,
                                            out / "library", out / "objects", out,
                                            out / "Kernels.cpp", ["gfx950"])
    assert len(objects) == 1
    obj = Path(objects[0])
    symbols = disassemble(obj)
    conversions = {name: code for name, code in symbols.items()
                   if name.startswith("Cijk_") and "_PostGSU" in name}
    assert conversions, "compiled object contains no post-GSU helper"
    return conversions, text_digest(obj)


def helper_source(path, out):
    """The production Kernels.cpp text of path's helpers (no compilation)."""
    from tensilelite.TensileCreateLibrary import Run

    out.mkdir(parents=True)
    with H._isolated_globals():
        solutions = H._solutions_from_logic_unguarded(path)
        assert len(solutions) == 1 and solutions[0]["Valid"]
        helpers = Run.generateKernelHelperObjects(solutions, str(H.get_assembler().path),
                                                 H.get_isa_info_map())
        assert helpers, "normal derivation emitted no helpers"
        Run.writeHelpers(out, helpers, "Kernels.cpp", "Kernels.h")
    return (out / "Kernels.cpp").read_text()



def compile_main(path, out):
    out.mkdir(parents=True)
    (solution,) = H.solutions_from_logic(path)
    assert solution["Valid"], "normal derivation refused the recipe"
    (name, source, error), = H.emit_kernels_from_logic(path)
    assert error == 0
    src, obj = out / (name + ".s"), out / (name + ".o")
    src.write_text(source)
    from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
    from tensilelite.Toolchain.Validators import ToolchainDefaults, validateToolchain

    assembler = makeAssemblyToolchain(validateToolchain("amdclang++"),
                                     validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER), "4").assembler
    assembler("gfx950", solution["WavefrontSize"], str(src), str(obj))
    symbols = disassemble(obj)
    # Assembly objects retain local loop labels as symbols; inspect all their
    # instructions in machine-code order, not just the first basic block.
    instructions = sorted(instruction for code in symbols.values() for instruction in code)
    return solution, source, instructions, text_digest(obj)


def stagger_instructions(source):
    # Test executable assembly operands, never comments, .set declarations or
    # label names. compile_main has also assembled this exact source successfully.
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
    return [line.strip() for line in source.splitlines()
            if re.match(r"\s*[sv]_\w+\s", line)
            and re.search(r"\bsgpr(?:StaggerUIter\w*|WrapU\w*)\b", line)]


def workspace_waterfalls(instructions):
    """W buffer loads inside readfirstlane/saveexec backward execnz loops.

    All vector buffer loads in a conversion helper read W; C, bias and scale
    use global loads. Decode the real branch displacement, not comments or
    source labels, so an epilogue's unrelated exec masking is not a failure.
    """
    waterfalls = {}
    for address, text in instructions:
        if not text.startswith("s_cbranch_execnz "):
            continue
        displacement = int(text.split()[1], 0)
        if displacement >= 0x8000:
            displacement -= 0x10000
        target = address + 4 + displacement * 4
        if target >= address:
            continue
        loop = [(pc, code) for pc, code in instructions if target <= pc <= address]
        if (any(code.startswith("v_readfirstlane_") for _, code in loop)
                and any(code.startswith("s_and_saveexec_") for _, code in loop)):
            for pc, code in loop:
                if code.startswith("buffer_load_"):
                    waterfalls[pc] = code
    return [waterfalls[pc] for pc in sorted(waterfalls)]

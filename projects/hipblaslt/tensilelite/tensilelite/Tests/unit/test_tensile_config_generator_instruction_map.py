#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

################################################################################
# Unit tests for tensile_config_generator.py instruction_map function.
#
# Covers tensile_config_generator.py:212-224 (instruction_map function) — the
# DataType/F32XdlMathOp dispatch that returns the correct MFMA instruction list
# for each dtype or None for unsupported combinations.
#
# NOTE: tensile_config_generator.py has module-level argparse/subprocess code
# that prevents direct import. This test extracts the production instruction_map
# and is_fp8 functions via AST parsing, then executes them with mock instruction
# lists. This ensures tests fail if the production dispatch logic changes.
#
# Distinct instruction-list placeholders (different K values) catch branch
# confusion (e.g., swapping FP16 and BF16 returns).
################################################################################

import ast
import os
import pytest

# Path to production script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATOR_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "Utilities", "tensile_generator", "tensile_config_generator.py")
)

# FP8 dtype list from tensile_config_generator.py:203
FP8_DTYPES = ["F8","F8N","B8","B8N","F8B8N","B8F8N","F8B8","B8F8"]

# Distinct instruction lists (different K dimension per dtype to catch swaps)
# Real lists vary by GPU; these are placeholders with unique K values
TF32_INSTRUCTIONS = [[32,32,2,2], [16,16,2,4]]      # K=2
FP32_INSTRUCTIONS = [[32,32,1,2], [16,16,1,4]]      # K=1
FP16_INSTRUCTIONS = [[32,32,4,2], [16,16,4,4]]      # K=4
BF16_INSTRUCTIONS = [[32,32,8,2], [16,16,8,4]]      # K=8 (distinct from FP16!)
FP8_INSTRUCTIONS = [[32,32,16,1], [16,16,32,1]]     # K=16,32


def _extract_function_from_ast(source_path, func_name):
    """Extract a function's source from a Python file via AST parsing.

    Returns the function's AST node. Caller must compile and exec it.
    """
    with open(source_path, 'r') as f:
        source = f.read()

    tree = ast.parse(source, filename=source_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            # Return a Module containing just this function
            return ast.Module(body=[node], type_ignores=[])

    raise ValueError(f"Function '{func_name}' not found in {source_path}")


# Extract production is_fp8 and instruction_map functions
_is_fp8_module = _extract_function_from_ast(GENERATOR_PATH, "is_fp8")
_instruction_map_module = _extract_function_from_ast(GENERATOR_PATH, "instruction_map")

# Compile and execute the extracted functions in a namespace with mock globals
_namespace = {
    'FP8_DTYPES': FP8_DTYPES,
    'tf32_instructions': TF32_INSTRUCTIONS,
    'fp32_instructions': FP32_INSTRUCTIONS,
    'fp16_instructions': FP16_INSTRUCTIONS,
    'bf16_instructions': BF16_INSTRUCTIONS,
    'fp8_instructions': FP8_INSTRUCTIONS,
}

exec(compile(_is_fp8_module, GENERATOR_PATH, 'exec'), _namespace)
exec(compile(_instruction_map_module, GENERATOR_PATH, 'exec'), _namespace)

# Import the production functions from namespace
is_fp8 = _namespace['is_fp8']
instruction_map = _namespace['instruction_map']


# ===========================================================================
# Tests
# ===========================================================================


def test_instruction_map_tf32():
    """instruction_map returns tf32_instructions for DataType='S' with F32XdlMathOp='x'.

    Covers tensile_config_generator.py:213-214:
        if dtype_dict["DataType"] == 'S' and dtype_dict["F32XdlMathOp"] == 'x':
            return tf32_instructions

    Critical: TF32 mode is selected by F32XdlMathOp='x', not 0. Wrong check would
    return fp32_instructions instead, producing wrong tuning configs.
    """
    dtype_dict = {"DataType": "S", "F32XdlMathOp": "x"}
    result = instruction_map(dtype_dict)

    assert result is not None, "TF32 should return instruction list, not None"
    assert isinstance(result, list), "Expected list of MFMA instructions"
    assert len(result) > 0, "TF32 instruction list should not be empty"
    assert result == TF32_INSTRUCTIONS, "Should return TF32 instruction list"


def test_instruction_map_fp32():
    """instruction_map returns fp32_instructions for DataType='S' with F32XdlMathOp=0.

    Covers tensile_config_generator.py:215-216:
        elif dtype_dict["DataType"] == 'S' and dtype_dict["F32XdlMathOp"] == 0:
            return fp32_instructions

    Critical: FP32 mode is selected by F32XdlMathOp=0 (not 'x'). Wrong check would
    return tf32_instructions, causing numeric differences in results.
    """
    dtype_dict = {"DataType": "S", "F32XdlMathOp": 0}
    result = instruction_map(dtype_dict)

    assert result is not None, "FP32 should return instruction list, not None"
    assert isinstance(result, list), "Expected list of MFMA instructions"
    assert len(result) > 0, "FP32 instruction list should not be empty"
    assert result == FP32_INSTRUCTIONS, "Should return FP32 instruction list"


def test_instruction_map_fp16():
    """instruction_map returns fp16_instructions for DataType='H'.

    Covers tensile_config_generator.py:217-218:
        elif dtype_dict["DataType"] == 'H':
            return fp16_instructions

    Critical: FP16 check does not require F32XdlMathOp (only applies to 'S').
    """
    dtype_dict = {"DataType": "H", "F32XdlMathOp": 0}
    result = instruction_map(dtype_dict)

    assert result is not None, "FP16 should return instruction list, not None"
    assert isinstance(result, list), "Expected list of MFMA instructions"
    assert len(result) > 0, "FP16 instruction list should not be empty"
    assert result == FP16_INSTRUCTIONS, "Should return FP16 instruction list"


def test_instruction_map_bf16():
    """instruction_map returns bf16_instructions for DataType='B'.

    Covers tensile_config_generator.py:219-220:
        elif dtype_dict["DataType"] == 'B':
            return bf16_instructions

    Critical: BF16 check does not require F32XdlMathOp (only applies to 'S').
    """
    dtype_dict = {"DataType": "B", "F32XdlMathOp": 0}
    result = instruction_map(dtype_dict)

    assert result is not None, "BF16 should return instruction list, not None"
    assert isinstance(result, list), "Expected list of MFMA instructions"
    assert len(result) > 0, "BF16 instruction list should not be empty"
    assert result == BF16_INSTRUCTIONS, "Should return BF16 instruction list"


@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
def test_instruction_map_fp8_variants(fp8_dtype):
    """instruction_map returns fp8_instructions for all FP8 dtype variants.

    Covers tensile_config_generator.py:221-222:
        elif is_fp8(dtype_dict["DataType"]):
            return fp8_instructions

    Critical: All FP8 dtype variants must return the same fp8_instructions list.
    FP8_DTYPES includes: F8, F8N, B8, B8N, F8B8N, B8F8N, F8B8, B8F8
    """
    dtype_dict = {"DataType": fp8_dtype, "F32XdlMathOp": 0}
    result = instruction_map(dtype_dict)

    assert result is not None, f"FP8 dtype {fp8_dtype} should return instruction list, not None"
    assert isinstance(result, list), "Expected list of MFMA instructions"
    assert len(result) > 0, f"FP8 instruction list should not be empty for {fp8_dtype}"
    assert result == FP8_INSTRUCTIONS, f"Should return FP8 instruction list for {fp8_dtype}"


def test_instruction_map_unsupported_dtype_returns_none():
    """instruction_map returns None for unsupported DataType.

    Covers tensile_config_generator.py:223-224:
        else:
            return None

    Critical: unsupported dtypes (e.g., 'D' for fp64, 'XS' for int8) return None,
    which the caller at line 551 silently skips. No error is raised.
    """
    # Test with fp64 (double precision, not in supported list)
    dtype_dict = {"DataType": "D", "F32XdlMathOp": 0}
    result = instruction_map(dtype_dict)
    assert result is None, "Unsupported dtype 'D' (fp64) should return None"

    # Test with another unsupported dtype
    dtype_dict = {"DataType": "XS", "F32XdlMathOp": 0}
    result = instruction_map(dtype_dict)
    assert result is None, "Unsupported dtype 'XS' should return None"

    # Test with invalid dtype
    dtype_dict = {"DataType": "INVALID", "F32XdlMathOp": 0}
    result = instruction_map(dtype_dict)
    assert result is None, "Invalid dtype should return None"


def test_instruction_map_fp32_wrong_xdlop_returns_none():
    """instruction_map returns None for DataType='S' with unsupported F32XdlMathOp.

    Critical: DataType='S' requires F32XdlMathOp to be either 'x' (TF32) or 0 (FP32).
    Other values should fall through to the else clause and return None.
    """
    # Test with F32XdlMathOp=1 (not 'x' or 0)
    dtype_dict = {"DataType": "S", "F32XdlMathOp": 1}
    result = instruction_map(dtype_dict)
    assert result is None, "DataType='S' with F32XdlMathOp=1 should return None"

    # Test with F32XdlMathOp as string "0" instead of integer 0
    dtype_dict = {"DataType": "S", "F32XdlMathOp": "0"}
    result = instruction_map(dtype_dict)
    assert result is None, "DataType='S' with F32XdlMathOp='0' (string) should return None"


def test_instruction_map_tf32_vs_fp32_different():
    """TF32 and FP32 instruction lists are different.

    Sanity check: the two S-type modes should produce different instruction sets
    (TF32 has different K dimension than FP32 for the same M,N).
    """
    tf32_result = instruction_map({"DataType": "S", "F32XdlMathOp": "x"})
    fp32_result = instruction_map({"DataType": "S", "F32XdlMathOp": 0})

    assert tf32_result != fp32_result, (
        "TF32 and FP32 instruction lists should be different "
        "(different MFMA K dimensions for single precision types)"
    )

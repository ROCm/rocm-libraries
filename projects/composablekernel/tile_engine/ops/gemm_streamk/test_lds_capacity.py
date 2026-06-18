import pytest
from gemm_streamk_validation_utils import validate_lds_capacity

def test_128kb_config_passes_gfx950_fails_gfx942():
    """gfx950 has 160KB LDS, gfx942 has 64KB."""
    valid_950, _ = validate_lds_capacity(256, 256, 128, "fp16", "fp16", "mem", "gfx950")
    valid_942, err = validate_lds_capacity(256, 256, 128, "fp16", "fp16", "mem", "gfx942")
    assert valid_950 and not valid_942
    print("PASSED: 128KB config accepted by gfx950, rejected by gfx942")


def test_double_buffer_halves_capacity():
    """compv4 halves LDS budget. 48KB fits gfx950 (80KB) but not gfx942 (32KB)."""
    valid_950, _ = validate_lds_capacity(256, 128, 64, "fp16", "fp16", "compv4", "gfx950")
    valid_942, _ = validate_lds_capacity(256, 128, 64, "fp16", "fp16", "compv4", "gfx942")
    assert valid_950 and not valid_942
    print("PASSED: Double-buffer capacity correctly halved per GPU")

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for OrigamiMatmulSelector interface."""

import pytest
import torch
import math
import origami
from origami.selector import OrigamiMatmulSelector
from .conftest import create_config_list


class MockConfig:
    """Mock Triton config for testing."""
    
    def __init__(self, block_m, block_n, block_k, waves_per_eu):
        self.kwargs = {
            'BLOCK_M': block_m,
            'BLOCK_N': block_n,
            'BLOCK_K': block_k,
            'waves_per_eu': waves_per_eu
        }


def create_mock_config_gen():
    """Create a simple mock config generator for testing."""
    return [
        MockConfig(128, 128, 32, 1),
        MockConfig(64, 64, 64, 2),
        MockConfig(256, 128, 64, 1),
    ]


@pytest.fixture
def rocm_device():
    """Fixture to check for ROCm device availability."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    try:
        # Try to get hardware info to check if ROCm is available
        origami.get_hardware_for_device(0)
        return torch.device("cuda:0")
    except RuntimeError:
        pytest.skip("No ROCm-capable device detected")


@pytest.mark.integration
def test_selector_initialization_basic(rocm_device):
    """Test basic initialization of OrigamiMatmulSelector."""
    config_gen = create_mock_config_gen()
    
    selector = OrigamiMatmulSelector(
        config_gen=config_gen,
        m=2048,
        n=2048,
        k=2048,
        a_dtype=torch.float16,
        b_dtype=torch.float16,
        out_dtype=torch.float16,
        device=rocm_device
    )
    
    assert selector is not None
    assert selector._m == 2048
    assert selector._n == 2048
    assert selector._k == 2048


@pytest.mark.integration
def test_selector_dtype_conversion(rocm_device):
    """Test dtype to string conversion."""
    config_gen = create_mock_config_gen()
    
    selector = OrigamiMatmulSelector(
        config_gen=config_gen,
        m=1024,
        n=1024,
        k=1024,
        a_dtype=torch.float32,
        b_dtype=torch.float32,
        out_dtype=torch.float32,
        device=rocm_device
    )
    
    assert selector._a_dtype_str == "f32"
    assert selector._b_dtype_str == "f32"
    assert selector._out_dtype_str == "f32"


@pytest.mark.integration
@pytest.mark.parametrize("dtype,expected_str", [
    (torch.float32, "f32"),
    (torch.float16, "f16"),
    (torch.bfloat16, "bf16"),
    (torch.int8, "i8"),
    (torch.int32, "i32"),
])
def test_selector_various_dtypes(rocm_device, dtype, expected_str):
    """Test selector with various data types."""
    config_gen = create_mock_config_gen()
    
    selector = OrigamiMatmulSelector(
        config_gen=config_gen,
        m=512,
        n=512,
        k=512,
        a_dtype=dtype,
        b_dtype=dtype,
        out_dtype=dtype,
        device=rocm_device
    )
    
    assert selector._a_dtype_str == expected_str
    assert selector._b_dtype_str == expected_str
    assert selector._out_dtype_str == expected_str


@pytest.mark.integration
def test_selector_macrotile_properties(rocm_device):
    """Test macrotile size properties."""
    config_gen = create_mock_config_gen()
    
    selector = OrigamiMatmulSelector(
        config_gen=config_gen,
        m=4096,
        n=4096,
        k=4096,
        a_dtype=torch.float16,
        b_dtype=torch.float16,
        out_dtype=torch.float16,
        device=rocm_device
    )
    
    # Check that macrotile properties return valid values
    assert selector.macrotile_m > 0
    assert selector.macrotile_n > 0
    assert selector.macrotile_k > 0
    
    # Macrotile sizes should be reasonable (not too large)
    assert selector.macrotile_m <= 16384
    assert selector.macrotile_n <= 16384
    assert selector.macrotile_k <= 16384


@pytest.mark.integration
def test_selector_wgm_property(rocm_device):
    """Test wgm (workgroup mapping) property."""
    config_gen = create_mock_config_gen()
    
    selector = OrigamiMatmulSelector(
        config_gen=config_gen,
        m=2048,
        n=2048,
        k=2048,
        a_dtype=torch.float16,
        b_dtype=torch.float16,
        out_dtype=torch.float16,
        device=rocm_device
    )
    
    assert selector.wgm > 0


@pytest.mark.integration
def test_selector_number_of_cus_property(rocm_device):
    """Test number_of_cus property."""
    config_gen = create_mock_config_gen()
    
    selector = OrigamiMatmulSelector(
        config_gen=config_gen,
        m=2048,
        n=2048,
        k=2048,
        a_dtype=torch.float16,
        b_dtype=torch.float16,
        out_dtype=torch.float16,
        device=rocm_device
    )
    
    assert selector.number_of_cus > 0


@pytest.mark.integration
def test_selector_occupancy_property(rocm_device):
    """Test occupancy property."""
    config_gen = create_mock_config_gen()
    
    selector = OrigamiMatmulSelector(
        config_gen=config_gen,
        m=2048,
        n=2048,
        k=2048,
        a_dtype=torch.float16,
        b_dtype=torch.float16,
        out_dtype=torch.float16,
        device=rocm_device
    )
    
    # Occupancy should be a reasonable value
    assert selector.occupancy > 0
    assert selector.occupancy <= 64 # max waves per eu is typically 2 or less for efficient schedules.


@pytest.mark.integration
def test_selector_even_k_property(rocm_device):
    """Test even_k property."""
    config_gen = create_mock_config_gen()
    
    # Test with K that should be evenly divisible
    selector = OrigamiMatmulSelector(
        config_gen=config_gen,
        m=2048,
        n=2048,
        k=2048,
        a_dtype=torch.float16,
        b_dtype=torch.float16,
        out_dtype=torch.float16,
        device=rocm_device
    )
    
    # even_k is a boolean
    assert isinstance(selector.even_k, bool)


@pytest.mark.integration
def test_selector_streamk_disabled(rocm_device):
    """Test selector with StreamK disabled (data-parallel grid)."""
    config_gen = create_mock_config_gen()


    m = n = k = 2048
    selector = OrigamiMatmulSelector(
    config_gen=config_gen,
    m=m,
    n=n,
    k=k,
    a_dtype=torch.float16,
    b_dtype=torch.float16,
    out_dtype=torch.float16,
    device=rocm_device,
    streamk=False
    )


    assert selector.grid_size > 0


    # Data-parallel grid: tiles in M and N based on selected macrotiles
    mt_m = selector.macrotile_m
    mt_n = selector.macrotile_n
    assert mt_m > 0 and mt_n > 0


    expected_grid = math.ceil(m / mt_m) * math.ceil(n / mt_n)
    assert selector.grid_size == expected_grid

@pytest.mark.integration
def test_selector_streamk_enabled(rocm_device):
    """Test selector with StreamK enabled."""
    config_gen = create_mock_config_gen()
    
    selector = OrigamiMatmulSelector(
        config_gen=config_gen,
        m=2048,
        n=2048,
        k=2048,
        a_dtype=torch.float16,
        b_dtype=torch.float16,
        out_dtype=torch.float16,
        device=rocm_device,
        streamk=True
    )
    
    # With StreamK enabled, grid is selected analytically
    assert selector.grid_size > 0


@pytest.mark.integration
def test_selector_batch_size(rocm_device):
    """Test selector with custom batch size."""
    config_gen = create_mock_config_gen()
    
    selector = OrigamiMatmulSelector(
        config_gen=config_gen,
        m=1024,
        n=1024,
        k=1024,
        a_dtype=torch.float16,
        b_dtype=torch.float16,
        out_dtype=torch.float16,
        device=rocm_device,
        batch=4
    )
    
    assert selector._batch == 4
    assert selector._problem.batch == 4


@pytest.mark.integration
def test_selector_mx_block_size(rocm_device):
    """Test selector with MX block size parameter."""
    config_gen = create_mock_config_gen()
    
    selector = OrigamiMatmulSelector(
        config_gen=config_gen,
        m=1024,
        n=1024,
        k=1024,
        a_dtype=torch.float16,
        b_dtype=torch.float16,
        out_dtype=torch.float16,
        device=rocm_device,
        mx_block_size=32
    )
    
    assert selector._mx_block_size == 32


@pytest.mark.integration
def test_selector_mixed_precision(rocm_device):
    """Test selector with mixed precision (FP8 input, BF16 output)."""
    # Check if FP8 types are available
    if not hasattr(torch, 'float8_e4m3fn'):
        pytest.skip("FP8 types not available in this PyTorch version")
    
    config_gen = create_mock_config_gen()
    
    selector = OrigamiMatmulSelector(
        config_gen=config_gen,
        m=2048,
        n=2048,
        k=2048,
        a_dtype=torch.float8_e4m3fn,
        b_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        device=rocm_device
    )
    
    assert selector._a_dtype_str == "f8"
    assert selector._b_dtype_str == "f8"
    assert selector._out_dtype_str == "bf16"
    # MI dtype should be based on input types (FP8)
    assert selector.mi_dtype == "f8"


@pytest.mark.integration
def test_selector_small_problem_size(rocm_device):
    """Test selector with small problem sizes."""
    config_gen = create_mock_config_gen()
    
    selector = OrigamiMatmulSelector(
        config_gen=config_gen,
        m=128,
        n=128,
        k=128,
        a_dtype=torch.float16,
        b_dtype=torch.float16,
        out_dtype=torch.float16,
        device=rocm_device
    )
    
    assert selector.macrotile_m > 0
    assert selector.macrotile_n > 0
    assert selector.macrotile_k > 0


@pytest.mark.integration
def test_selector_large_problem_size(rocm_device):
    """Test selector with large problem sizes."""
    config_gen = create_mock_config_gen()
    
    selector = OrigamiMatmulSelector(
        config_gen=config_gen,
        m=16384,
        n=16384,
        k=16384,
        a_dtype=torch.float16,
        b_dtype=torch.float16,
        out_dtype=torch.float16,
        device=rocm_device
    )
    
    assert selector.macrotile_m > 0
    assert selector.macrotile_n > 0
    assert selector.macrotile_k > 0


@pytest.mark.integration
def test_selector_rectangular_matrices(rocm_device):
    """Test selector with non-square matrices."""
    config_gen = create_mock_config_gen()
    
    # Test tall-skinny matrix
    selector = OrigamiMatmulSelector(
        config_gen=config_gen,
        m=8192,
        n=512,
        k=2048,
        a_dtype=torch.float16,
        b_dtype=torch.float16,
        out_dtype=torch.float16,
        device=rocm_device
    )
    
    assert selector.macrotile_m > 0
    assert selector.macrotile_n > 0
    assert selector.macrotile_k > 0


@pytest.mark.integration
def test_selector_hardware_info(rocm_device):
    """Test that selector correctly retrieves hardware info."""
    config_gen = create_mock_config_gen()
    
    selector = OrigamiMatmulSelector(
        config_gen=config_gen,
        m=2048,
        n=2048,
        k=2048,
        a_dtype=torch.float16,
        b_dtype=torch.float16,
        out_dtype=torch.float16,
        device=rocm_device
    )
    
    # Check that hardware info was retrieved
    assert selector._hardware is not None
    assert selector._mx_block_size == 0  # Default value
    assert selector.number_of_cus > 0
    
    # N_CU should be a known AMD GPU value
    # Expand this list as we add more GPUs to the test suite.
    assert selector.number_of_cus in [104, 228, 256, 304]


@pytest.mark.integration
def test_selector_config_generation(rocm_device):
    """Test that configs are properly generated from config_gen."""
    config_gen = create_mock_config_gen()
    
    selector = OrigamiMatmulSelector(
        config_gen=config_gen,
        m=2048,
        n=2048,
        k=2048,
        a_dtype=torch.float16,
        b_dtype=torch.float16,
        out_dtype=torch.float16,
        device=rocm_device
    )
    
    # Should have generated configs from the input generator
    assert len(selector._configs) == len(list(config_gen))
    assert all(isinstance(cfg, origami.config_t) for cfg in selector._configs)


@pytest.mark.integration
def test_selector_problem_creation(rocm_device):
    """Test that problem_t is properly created."""
    config_gen = create_mock_config_gen()
    
    selector = OrigamiMatmulSelector(
        config_gen=config_gen,
        m=2048,
        n=2048,
        k=2048,
        a_dtype=torch.float16,
        b_dtype=torch.float16,
        out_dtype=torch.float16,
        device=rocm_device
    )
    
    # Check that problem was created with correct dimensions
    assert selector._problem is not None
    assert selector._problem.size.m == 2048
    assert selector._problem.size.n == 2048
    assert selector._problem.size.k == 2048
    assert selector._problem.batch == 1

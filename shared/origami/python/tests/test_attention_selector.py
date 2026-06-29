# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for OrigamiAttentionSelector interface."""

import pytest
import math
import origami

# Skip entire module if torch is not available (selector requires torch)
torch = pytest.importorskip("torch", reason="torch is required for OrigamiAttentionSelector tests.")

from origami.selector import OrigamiAttentionSelector


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
        MockConfig(128, 128, 64, 1),
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
def test_attention_selector_initialization_basic(rocm_device):
    """Test basic initialization of OrigamiAttentionSelector."""
    config_gen = create_mock_config_gen()

    selector = OrigamiAttentionSelector(
        config_gen=config_gen,
        q_seq_len=2048,
        kv_seq_len=2048,
        head_dim=128,
        q_heads=32,
        kv_heads=32,
        dtype=torch.float16,
        device=rocm_device
    )

    assert selector is not None
    assert selector._q_seq_len == 2048
    assert selector._kv_seq_len == 2048
    assert selector._head_dim == 128
    assert selector._q_heads == 32
    assert selector._kv_heads == 32


@pytest.mark.integration
def test_attention_selector_dtype_conversion(rocm_device):
    """Test dtype to string conversion."""
    config_gen = create_mock_config_gen()

    selector = OrigamiAttentionSelector(
        config_gen=config_gen,
        q_seq_len=1024,
        kv_seq_len=1024,
        head_dim=64,
        q_heads=16,
        kv_heads=16,
        dtype=torch.bfloat16,
        device=rocm_device
    )

    assert selector._dtype_str == "bf16"


@pytest.mark.integration
@pytest.mark.parametrize("dtype,expected_str", [
    (torch.float32, "f32"),
    (torch.float16, "f16"),
    (torch.bfloat16, "bf16"),
])
def test_attention_selector_various_dtypes(rocm_device, dtype, expected_str):
    """Test selector with various data types."""
    config_gen = create_mock_config_gen()

    selector = OrigamiAttentionSelector(
        config_gen=config_gen,
        q_seq_len=512,
        kv_seq_len=512,
        head_dim=64,
        q_heads=8,
        kv_heads=8,
        dtype=dtype,
        device=rocm_device
    )

    assert selector._dtype_str == expected_str


@pytest.mark.integration
def test_attention_selector_macrotile_properties(rocm_device):
    """Test that macrotile properties are accessible."""
    config_gen = create_mock_config_gen()

    selector = OrigamiAttentionSelector(
        config_gen=config_gen,
        q_seq_len=2048,
        kv_seq_len=2048,
        head_dim=128,
        q_heads=32,
        kv_heads=32,
        dtype=torch.float16,
        device=rocm_device
    )

    # Check that macrotile dimensions are positive
    assert selector.macrotile_m > 0
    assert selector.macrotile_n > 0
    assert selector.macrotile_k > 0

    # Check that selected macrotile is one from the config_gen
    valid_m_values = [64, 128, 256]
    valid_n_values = [64, 128, 256]
    valid_k_values = [32, 64]

    assert selector.macrotile_m in valid_m_values
    assert selector.macrotile_n in valid_n_values
    assert selector.macrotile_k in valid_k_values


@pytest.mark.integration
def test_attention_selector_wgm_property(rocm_device):
    """Test workgroup mapping property."""
    config_gen = create_mock_config_gen()

    selector = OrigamiAttentionSelector(
        config_gen=config_gen,
        q_seq_len=1024,
        kv_seq_len=1024,
        head_dim=64,
        q_heads=16,
        kv_heads=16,
        dtype=torch.float16,
        device=rocm_device
    )

    assert selector.wgm >= 1


@pytest.mark.integration
def test_attention_selector_number_of_cus_property(rocm_device):
    """Test number of CUs property."""
    config_gen = create_mock_config_gen()

    selector = OrigamiAttentionSelector(
        config_gen=config_gen,
        q_seq_len=512,
        kv_seq_len=512,
        head_dim=64,
        q_heads=8,
        kv_heads=8,
        dtype=torch.float16,
        device=rocm_device
    )

    # Number of CUs should be reasonable (common values: 104, 228, 256, 304)
    assert selector.number_of_cus > 0
    assert selector.number_of_cus < 500


@pytest.mark.integration
def test_attention_selector_occupancy_property(rocm_device):
    """Test occupancy property."""
    config_gen = create_mock_config_gen()

    selector = OrigamiAttentionSelector(
        config_gen=config_gen,
        q_seq_len=2048,
        kv_seq_len=2048,
        head_dim=128,
        q_heads=32,
        kv_heads=32,
        dtype=torch.float16,
        device=rocm_device
    )

    # Occupancy should be one of the values from config_gen
    assert selector.occupancy in [1, 2]


@pytest.mark.integration
def test_attention_selector_batch_size(rocm_device):
    """Test selector with different batch sizes."""
    config_gen = create_mock_config_gen()

    selector = OrigamiAttentionSelector(
        config_gen=config_gen,
        q_seq_len=1024,
        kv_seq_len=1024,
        head_dim=64,
        q_heads=16,
        kv_heads=16,
        dtype=torch.float16,
        device=rocm_device,
        batch=8
    )

    assert selector._batch == 8


@pytest.mark.integration
def test_attention_selector_gqa_scenario(rocm_device):
    """Test selector with Grouped Query Attention (GQA) - different q_heads and kv_heads."""
    config_gen = create_mock_config_gen()

    # GQA: e.g., 32 query heads, 8 kv heads
    selector = OrigamiAttentionSelector(
        config_gen=config_gen,
        q_seq_len=2048,
        kv_seq_len=2048,
        head_dim=128,
        q_heads=32,
        kv_heads=8,
        dtype=torch.float16,
        device=rocm_device
    )

    assert selector._q_heads == 32
    assert selector._kv_heads == 8


@pytest.mark.integration
def test_attention_selector_mqa_scenario(rocm_device):
    """Test selector with Multi-Query Attention (MQA) - single kv head."""
    config_gen = create_mock_config_gen()

    # MQA: multiple query heads, single kv head
    selector = OrigamiAttentionSelector(
        config_gen=config_gen,
        q_seq_len=1024,
        kv_seq_len=1024,
        head_dim=64,
        q_heads=16,
        kv_heads=1,
        dtype=torch.float16,
        device=rocm_device
    )

    assert selector._q_heads == 16
    assert selector._kv_heads == 1


@pytest.mark.integration
def test_attention_selector_small_problem_size(rocm_device):
    """Test selector with small sequence lengths."""
    config_gen = create_mock_config_gen()

    selector = OrigamiAttentionSelector(
        config_gen=config_gen,
        q_seq_len=128,
        kv_seq_len=128,
        head_dim=64,
        q_heads=4,
        kv_heads=4,
        dtype=torch.float16,
        device=rocm_device
    )

    assert selector.macrotile_m > 0
    assert selector.macrotile_n > 0


@pytest.mark.integration
def test_attention_selector_large_problem_size(rocm_device):
    """Test selector with large sequence lengths."""
    config_gen = create_mock_config_gen()

    selector = OrigamiAttentionSelector(
        config_gen=config_gen,
        q_seq_len=8192,
        kv_seq_len=8192,
        head_dim=128,
        q_heads=64,
        kv_heads=64,
        dtype=torch.float16,
        device=rocm_device
    )

    assert selector.grid_size > 0


@pytest.mark.integration
def test_attention_selector_rectangular_sequences(rocm_device):
    """Test selector with different query and kv sequence lengths."""
    config_gen = create_mock_config_gen()

    # Different sequence lengths (e.g., cross-attention scenario)
    selector = OrigamiAttentionSelector(
        config_gen=config_gen,
        q_seq_len=512,
        kv_seq_len=2048,
        head_dim=64,
        q_heads=8,
        kv_heads=8,
        dtype=torch.float16,
        device=rocm_device
    )

    assert selector._q_seq_len == 512
    assert selector._kv_seq_len == 2048


@pytest.mark.integration
def test_attention_selector_hardware_info(rocm_device):
    """Test that hardware info is correctly retrieved."""
    config_gen = create_mock_config_gen()

    selector = OrigamiAttentionSelector(
        config_gen=config_gen,
        q_seq_len=1024,
        kv_seq_len=1024,
        head_dim=64,
        q_heads=16,
        kv_heads=16,
        dtype=torch.float16,
        device=rocm_device
    )

    # Check that hardware object exists and has expected attributes
    assert hasattr(selector._hardware, 'N_CU')
    assert hasattr(selector._hardware, 'lds_capacity')
    assert selector._hardware.N_CU == selector.number_of_cus


@pytest.mark.integration
def test_attention_selector_grid_size_property(rocm_device):
    """Test grid size property."""
    config_gen = create_mock_config_gen()

    selector = OrigamiAttentionSelector(
        config_gen=config_gen,
        q_seq_len=2048,
        kv_seq_len=2048,
        head_dim=128,
        q_heads=32,
        kv_heads=32,
        dtype=torch.float16,
        device=rocm_device
    )

    # Grid size should be positive
    assert selector.grid_size > 0

    # Grid size should be reasonable for the problem
    # For Flash Attention: grid ~ (q_seq_len/block_m) * (kv_seq_len/block_n) * batch * q_heads
    max_expected_grid = (2048 // 64 + 1) * (2048 // 64 + 1) * 1 * 32
    assert selector.grid_size <= max_expected_grid * 2  # Allow some margin

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for hardware_t constructor with architecture constants."""

import pytest
import origami
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class MI:
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class Tile:
    MT_M: int
    MT_N: int
    MT_K: int
    bytes_ab: int


def enumerate_tiles_half_lds(
    mi: MI,
    *,
    datatype_size: int,   # bytes/element (fp16/bf16=2)
    lds_bytes: int,       # total LDS bytes available
) -> List[Tile]:
    """
    Generate tiles by incrementing each tile dim in MI-sized steps:

      MT_M in {MI_M, 2*MI_M, 3*MI_M, ...}
      MT_N in {MI_N, 2*MI_N, 3*MI_N, ...}
      MT_K in {MI_K, 2*MI_K, 3*MI_K, ...}

    Keep tiles where:
      datatype_size * (MT_M*MT_K + MT_N*MT_K) <= lds_bytes/2
    """
    if min(mi.m, mi.n, mi.k) <= 0:
        raise ValueError("MI dims must be positive.")
    if datatype_size <= 0:
        raise ValueError("datatype_size must be positive.")
    if lds_bytes <= 0:
        raise ValueError("lds_bytes must be positive.")

    budget = lds_bytes
    out: List[Tile] = []

    # MT_K increments
    MT_K = mi.k
    while True:
        # bytes_ab = dtype * MT_K * (MT_M + MT_N)
        # For a fixed MT_K, minimum MT_M+MT_N is (MI_M + MI_N)
        min_bytes_for_this_K = datatype_size * MT_K * (mi.m + mi.n)
        if min_bytes_for_this_K > budget:
            break

        # MT_M increments
        MT_M = mi.m
        while True:
            # For fixed (MT_K, MT_M), minimum MT_N is MI_N
            min_bytes_for_this_M = datatype_size * MT_K * (MT_M + mi.n)
            if min_bytes_for_this_M > budget:
                break

            # MT_N increments
            MT_N = mi.n
            while True:
                bytes_ab = datatype_size * MT_K * (MT_M + MT_N)
                if bytes_ab > budget:
                    break

                out.append(Tile(MT_M=MT_M, MT_N=MT_N, MT_K=MT_K, bytes_ab=bytes_ab))
                MT_N += mi.n

            MT_M += mi.m

        MT_K += mi.k

    out.sort(key=lambda t: (t.MT_K, t.MT_M, t.MT_N))
    return out


@pytest.mark.integration
def test_hardware_for_arch_gfx950():
    """Test creating hardware object for gfx950 using get_hardware_for_arch."""
    hardware = origami.get_hardware_for_arch(
        arch=origami.architecture_t.gfx950,
        N_CU=304,
        lds_capacity=64 * 1024,
        L2_capacity=32 * 1024 * 1024,
        compute_clock_khz=2100000
    )
    
    # Verify basic properties
    assert hardware.N_CU == 304
    assert hardware.lds_capacity == 64 * 1024
    assert hardware.L2_capacity == 32 * 1024 * 1024
    assert hardware.compute_clock_ghz == pytest.approx(2.1, rel=1e-6)
    
    # Verify architecture-specific constants were applied
    assert hardware.NUM_XCD == 8
    assert hardware.parallel_mi_cu == 4
    # mem1_perf_ratio is calculated based on clock speeds, not a direct constant
    assert hardware.mem1_perf_ratio > 0
    


@pytest.mark.integration
def test_hardware_for_arch_gfx942():
    """Test creating hardware object for gfx942 using get_hardware_for_arch."""
    hardware = origami.get_hardware_for_arch(
        arch=origami.architecture_t.gfx942,
        N_CU=228,
        lds_capacity=64 * 1024,
        L2_capacity=24 * 1024 * 1024,
        compute_clock_khz=1700000
    )
    
    # Verify basic properties
    assert hardware.N_CU == 228
    assert hardware.lds_capacity == 64 * 1024
    assert hardware.L2_capacity == 24 * 1024 * 1024
    assert hardware.compute_clock_ghz == pytest.approx(1.7, rel=1e-6)
    
    # Verify architecture-specific constants
    assert hardware.NUM_XCD == 8
    assert hardware.parallel_mi_cu == 4


@pytest.mark.integration
def test_hardware_for_arch_gfx90a():
    """Test creating hardware object for gfx90a using get_hardware_for_arch."""
    hardware = origami.get_hardware_for_arch(
        arch=origami.architecture_t.gfx90a,
        N_CU=110,
        lds_capacity=64 * 1024,
        L2_capacity=8 * 1024 * 1024,
        compute_clock_khz=1700000
    )
    
    # Verify basic properties
    assert hardware.N_CU == 110
    assert hardware.lds_capacity == 64 * 1024
    
    # Verify architecture-specific constants
    assert hardware.NUM_XCD == 1
    assert hardware.parallel_mi_cu == 4


@pytest.mark.integration
def test_hardware_for_arch_gfx1201():
    """Test creating hardware object for gfx1201 using get_hardware_for_arch."""
    hardware = origami.get_hardware_for_arch(
        arch=origami.architecture_t.gfx1201,
        N_CU=60,
        lds_capacity=128 * 1024,
        L2_capacity=6 * 1024 * 1024,
        compute_clock_khz=2500000
    )
    
    # Verify basic properties
    assert hardware.N_CU == 60
    assert hardware.lds_capacity == 128 * 1024
    
    # Verify architecture-specific constants
    assert hardware.NUM_XCD == 1
    assert hardware.parallel_mi_cu == 2


@pytest.mark.integration
def test_enumerate_tiles_with_hardware():
    """Test tile enumeration using hardware_t LDS capacity."""
    # Create MI350 hardware configuration
    hardware = origami.get_hardware_for_arch(
        arch=origami.architecture_t.gfx950,
        N_CU=304,
        lds_capacity=64 * 1024,
        L2_capacity=32 * 1024 * 1024,
        compute_clock_khz=2100000
    )
    
    # MI350 typical bf16/fp16 inst
    mi = MI(m=16, n=16, k=32)
    tiles = enumerate_tiles_half_lds(
        mi,
        datatype_size=2,
        lds_bytes=hardware.lds_capacity
    )
    
    # Verify we got some tiles
    assert len(tiles) > 0
    
    # All tiles should fit in half LDS
    for tile in tiles:
        assert tile.bytes_ab <= hardware.lds_capacity
        # Verify tile dimensions are multiples of MI dimensions
        assert tile.MT_M % mi.m == 0
        assert tile.MT_N % mi.n == 0
        assert tile.MT_K % mi.k == 0


@pytest.mark.integration
def test_enumerate_tiles_mi350_example():
    """Test the example from the task description."""
    # MI350 typical bf16/fp16 inst
    mi = MI(m=16, n=16, k=32)
    tiles = enumerate_tiles_half_lds(mi, datatype_size=2, lds_bytes=64 * 1024)
    
    # Verify results
    assert len(tiles) > 0
    print(f"{len(tiles)} tiles under half LDS ({(64*1024)//2} bytes)")
    
    # Check some properties of the generated tiles
    for tile in tiles:
        # Verify bytes_ab calculation
        expected_bytes = 2 * tile.MT_K * (tile.MT_M + tile.MT_N)
        assert tile.bytes_ab == expected_bytes
        
        # Verify it fits in budget
        assert tile.bytes_ab <= 64 * 1024
        
        # Verify dimensions are multiples of MI
        assert tile.MT_M % 16 == 0
        assert tile.MT_N % 16 == 0
        assert tile.MT_K % 32 == 0


@pytest.mark.integration
def test_tile_enumeration_empty_cases():
    """Test edge cases for tile enumeration."""
    mi = MI(m=16, n=16, k=32)
    
    # Very small LDS should produce few or no tiles
    tiles = enumerate_tiles_half_lds(mi, datatype_size=2, lds_bytes=1024)
    # With 1024 bytes budget and smallest tile = 2 * 32 * (16 + 16) = 2048 bytes
    # We expect 0 tiles
    assert len(tiles) == 0
    
    # Invalid inputs should raise errors
    with pytest.raises(ValueError):
        enumerate_tiles_half_lds(MI(m=0, n=16, k=32), datatype_size=2, lds_bytes=64*1024)
    
    with pytest.raises(ValueError):
        enumerate_tiles_half_lds(mi, datatype_size=0, lds_bytes=64*1024)
    
    with pytest.raises(ValueError):
        enumerate_tiles_half_lds(mi, datatype_size=2, lds_bytes=0)


@pytest.mark.integration
def test_different_datatypes():
    """Test tile enumeration with different datatype sizes."""
    mi = MI(m=16, n=16, k=32)
    lds_bytes = 64 * 1024
    
    # FP32 (4 bytes)
    tiles_fp32 = enumerate_tiles_half_lds(mi, datatype_size=4, lds_bytes=lds_bytes)
    
    # FP16/BF16 (2 bytes)
    tiles_fp16 = enumerate_tiles_half_lds(mi, datatype_size=2, lds_bytes=lds_bytes)
    
    # FP8 (1 byte)
    tiles_fp8 = enumerate_tiles_half_lds(mi, datatype_size=1, lds_bytes=lds_bytes)
    
    # More tiles should fit with smaller datatypes
    assert len(tiles_fp8) > len(tiles_fp16) > len(tiles_fp32)

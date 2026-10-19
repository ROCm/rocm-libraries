# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Candidate MatrixInstructions for MI Designer.

Two sources, both independent of the problem size:

* ``enumerate_candidate_mis``: every MI geko's tiling rules allow for an ARCH and
  data type (MFMA allowlist x MIBlockM x wave layout x wave tile, within the
  macro-tile bounds). Whether TensileLite can build a kernel for an MI is not
  checked here.
* Buildable-MI lists (``mi_lists/<arch>/<GEMM name>.yaml``): the subset of those
  candidates for which TensileLite code generation builds at least one kernel
  within the generic (GA) search space's fork-parameter ranges. Produced by
  ``scripts/generate_buildable_mis.py``; MI Designer reads them instead of
  enumerating when one exists for the active GEMM type (except for the subtile
  search space, whose kernels the lists do not cover).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from geko.config_generator.constants import (
    HARDWARE_MAP,
    LIST_OF_WAVEs_TO_INCLUDE,
    LSUTHRESHOLD,
    MAX_MT0,
    MAX_MT1,
    MIN_MT0,
    MIN_MT1,
    computeDataTypeSize,
    get_list_of_mt_max_size,
    library_logic_architecture,
    validMFMA,
)

MI = Tuple[int, int, int, int, int, int, int, int, int]

MI_LISTS_DIR = Path(__file__).parent / "mi_lists"


def allowable_mfmas(
    arch: str, data_type: str, mfma_override: Optional[Sequence[int]] = None
) -> List[List[int]]:
    """MFMA instructions ([M, N, K, B]) MI Designer may use: the ARCH allowlist, or the config's override."""
    if mfma_override is not None:
        return [list(mfma_override)]
    return HARDWARE_MAP[arch]["ONLY_INCLUDE_MIs"][data_type]


def smallest_mfma_dims(allowable: Sequence[Sequence[int]]) -> Tuple[int, int]:
    """(smallest M, smallest N) over the allowed MFMA instructions (512 each when none is smaller)."""
    return min([512] + [m[0] for m in allowable]), min([512] + [m[1] for m in allowable])


def enumerate_candidate_mis(
    arch: str,
    data_type: str,
    search_space: Optional[str] = None,
    mfma_override: Optional[Sequence[int]] = None,
) -> List[MI]:
    """Every MatrixInstruction geko's tiling rules allow, as 9-tuples
    (M, N, K, B, MIBlockM, waveTileM, waveTileN, waveM, waveN).

    For each allowed MFMA and wave layout, wave tiles grow until the macro tile
    leaves [MIN_MT, MAX_MT]; a tile is dropped when its local-split-U LDS use
    exceeds LSUTHRESHOLD or its area exceeds the search space's MT cap.
    """
    allowable = allowable_mfmas(arch, data_type, mfma_override)
    mt_max_size = get_list_of_mt_max_size(search_space)

    # MIBlockM > 1 has not produced winners; raise bm_max to log2(MI[3]) to include it.
    bm_max = 0

    mis: List[MI] = []
    for mfma in reversed(validMFMA[data_type]):
        if mfma not in allowable:
            continue
        for bm in range(bm_max + 1):
            mi_block_m = 2**bm
            for wave in LIST_OF_WAVEs_TO_INCLUDE:
                wave_tile_m = 0
                while True:
                    wave_tile_m += 1
                    wave_tile_n = 0
                    mt0 = mfma[0] * mi_block_m * wave_tile_m * wave[0]
                    if mt0 < MIN_MT0:
                        continue
                    if mt0 > MAX_MT0:
                        break
                    while True:
                        wave_tile_n += 1
                        mt1 = int(mfma[1] / mi_block_m * mfma[3] * wave_tile_n * wave[1])
                        if mt1 < MIN_MT1:
                            continue
                        if mt1 > MAX_MT1:
                            break
                        lsu = max(1, 4 // wave[0] // wave[1])
                        if (
                            lsu > 1
                            and mt0 * mt1 * computeDataTypeSize[data_type] * lsu > LSUTHRESHOLD
                        ):
                            continue
                        if mt0 * mt1 > mt_max_size[data_type]:
                            continue
                        mis.append(
                            (
                                mfma[0],
                                mfma[1],
                                mfma[2],
                                mfma[3],
                                mi_block_m,
                                wave_tile_m,
                                wave_tile_n,
                                wave[0],
                                wave[1],
                            )
                        )
    return mis


def buildable_list_path(arch: str, gemm_name: str) -> Path:
    """Where the buildable-MI list for an ARCH and GEMM type (``GemmConfig.name``) lives.

    Lists are per compiler target (``gfx950_128cu`` shares ``gfx950``'s): code generation
    does not depend on the CU count.
    """
    return MI_LISTS_DIR / library_logic_architecture(arch) / f"{gemm_name}.yaml"


def load_buildable_list(path: Path) -> Dict[str, Any]:
    """Read a buildable-MI list. ``MatrixInstructions`` holds the buildable MIs as 9-element lists."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not isinstance(data.get("MatrixInstructions"), list):
        raise ValueError(f"{path}: not a buildable-MI list (no MatrixInstructions list)")
    return data

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

# TODO: Add a size to config, for each filter to test the impact of the filter.
# TODO: add a Filter to MIdesign, when we have golden sizes, no need to tests other tiles.
# TODO: add a filter, if TilesPerCU is greater than 3, go only 5 rounds.


# MI_FILTER 1 trim from the end
# MI_FILTER 2 trim from end and in the middle

import logging
import math
import os
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from geko.config_generator.constants import *
from geko.config_generator.shared_utils import ForkParameter, GroupDimension



logger = logging.getLogger("GEKO")


@dataclass(frozen=True)
class MFMA:
  """Matrix Instruction configuration."""
  M: int  # M dimension of MFMA
  N: int  # N dimension of MFMA
  K: int  # K dimension of MFMA
  B: int  # Number of Blocks / Batch Factor
  MIBlockM: int  # Block size in M dimension
  waveTileM: int  # Wave tile in M dimension
  waveTileN: int  # Wave tile in N dimension
  waveM: int  # Waves in M dimension
  waveN: int  # Waves in N dimension

  @classmethod
  def from_list(cls, vals: Sequence[int]) -> "MFMA":
    """Construct an MFMA from a 9-element MatrixInstruction list."""
    if not isinstance(vals, (list, tuple)) or len(vals) != 9:
        raise ValueError(f"Expected a list/tuple of 9 elements, got {vals}")
    return cls(M=vals[0], N=vals[1], K=vals[2], B=vals[3], MIBlockM=vals[4],
               waveTileM=vals[5], waveTileN=vals[6], waveM=vals[7], waveN=vals[8])


@dataclass(frozen=True)
class MFMAParameters:
  """MFMA-derived tile and workgroup parameters."""
  MT0: int  # Tile size in M dimension
  MT1: int  # Tile size in N dimension
  TT0: int  # Thread tile in M dimension
  TT1: int  # Thread tile in N dimension
  WG0: int  # Workgroup size in first dimension
  WG1: int  # Workgroup size in second dimension
  MIBlockM: int  # Matrix instruction block size in M


@dataclass(frozen=True)
class GranularityMetrics:
  """Granularity and load-balancing metrics for a given MI and size."""
  NumTile0: float  # Number of tiles in M dimension
  NumTile1: float  # Number of tiles in N dimension
  Tile0Granularity: float  # Granularity in M dimension (0-1)
  Tile1Granularity: float  # Granularity in N dimension (0-1)
  TotalTiles: float  # Total number of tiles to process
  TilesPerCU: float  # Average tiles per compute unit
  CUGranularity: float  # Compute unit load granularity
  waveGranularity: float  # Wave-level granularity
  totalGranularity: float  # Combined granularity metric


@dataclass(frozen=True)
class MFMACandidate:
  """Complete MFMA candidate with all filtering and ranking metrics."""
  totalGranularity: float
  TilesPerCU: float
  mfma: MFMA  # MFMA configuration
  NumTile0: float
  NumTile1: float
  Tile0Granularity: float
  Tile1Granularity: float
  TotalTiles: float
  CUGranularity: float
  waveGranularity: float
  GSU: int  # Global split unit
  LSU: int  # Local split unit
  MT0: int  # Tile size in M dimension
  MT1: int  # Tile size in N dimension
  TT0: int  # Thread tile in M dimension
  TT1: int  # Thread tile in N dimension
  WG0: int  # Workgroup size in first dimension
  WG1: int  # Workgroup size in second dimension
  MIBlockM: int  # Matrix instruction block size in M


class MIDesign:
  """Build MatrixInstruction fork-parameter groups from ARCH, dtype, and size."""

  @staticmethod
  def calculate_granularities(
      MT0: int,
      MT1: int,
      M: int,
      N: int,
      batch_count: int,
      CUs: int,
      LSU: int,
      GSU: int,
      wave: Sequence[int],
  ) -> GranularityMetrics:
      """Compute granularity-related metrics for a given size and tile."""

      NumTile0 = M / float(MT0)
      NumTile1 = N / float(MT1)
      Tile0Granularity = NumTile0/math.ceil(NumTile0)
      Tile1Granularity = NumTile1/math.ceil(NumTile1)

      TotalTiles = math.ceil(NumTile0) * math.ceil(NumTile1) * batch_count * GSU * LSU

      TilesPerCU = TotalTiles / CUs
      CUGranularity = TilesPerCU / math.ceil(TilesPerCU)
      SIMDPerCU = 4

      waveGranularity = min(1.0, math.floor(TilesPerCU+1.0) * wave[0]*wave[1]*LSU/SIMDPerCU)
      totalGranularity = Tile0Granularity * Tile1Granularity * CUGranularity * waveGranularity

      return GranularityMetrics(NumTile0, NumTile1, Tile0Granularity, Tile1Granularity, TotalTiles, TilesPerCU, CUGranularity, waveGranularity, totalGranularity)

  @staticmethod
  def calculate_mfma_parameters(MI: MFMA, waveFrontSize: int = 64) -> MFMAParameters:
      """Compute MFMA-derived parameters for a MatrixInstruction."""
      MatrixInstM = MI.M * MI.MIBlockM
      MT0 = MatrixInstM * MI.waveTileM * MI.waveM

      MatrixInstN = MI.N / MI.MIBlockM * MI.B
      MT1 = int(MatrixInstN * MI.waveTileN * MI.waveN)
      TT0 = MI.waveTileM
      TT1 = MI.waveTileN * MI.N
      WG0 = MatrixInstM * MI.waveM
      WG1 = int(MI.waveM * MI.waveN * waveFrontSize / WG0)

      return MFMAParameters(MT0, MT1, TT0, TT1, WG0, WG1, MI.MIBlockM)

  def generate_all_mfmas(self) -> Tuple[List[MFMA], int, int]:
      """Generate all valid MFMA configurations for the current config.

      Returns:
          (valid_mfmas, smallest_M_in_MFMA, smallest_N_in_MFMA). The smallest
          dimensions are taken from the effective MFMA allowlist (including
          a single MFMA override from config).
      """
      valid_mfmas: List[MFMA] = []

      search_space = self.config.get("search_space")
      mt_max_size = get_list_of_mt_max_size(search_space)

      hw = HARDWARE_MAP[self.config["ARCH"]]
      allowable_mfma = hw["ONLY_INCLUDE_MIs"][self._gt.data_type]

      # Override if specified in the input config
      if "MFMA" in self.config:
          mfma = list(self.config["MFMA"])
          allowable_mfma = [mfma]

      bm_max = 0
      # Based on our experiance MIBlockM>1 is not a winner. To test MIBlocM>1, uncomment the following line.
      # bm_max = int(math.log(MI[3], 2))

      smallest_M_in_MFMA = 512
      smallest_N_in_MFMA = 512
      for mfma in allowable_mfma:
          if mfma[0] < smallest_M_in_MFMA:
              smallest_M_in_MFMA = mfma[0]

          if mfma[1] < smallest_N_in_MFMA:
              smallest_N_in_MFMA = mfma[1]
      for MI in reversed(validMFMA[self._gt.data_type]):
          if MI not in allowable_mfma:
              continue
          for bm in range(bm_max + 1):
              MIBlockM = 2 ** bm

              for wave in LIST_OF_WAVEs_TO_INCLUDE:
                  waveTileM = 0
                  waveTileN = 0

                  while True:
                      waveTileM += 1
                      waveTileN = 0
                      MatrixInstM = MI[0] * MIBlockM
                      MT0 = MatrixInstM * waveTileM * wave[0]
                      if MT0 < MIN_MT0:
                          continue
                      if MT0 > MAX_MT0:
                          break

                      while True:
                          waveTileN += 1
                          MatrixInstN = MI[1] / MIBlockM * MI[3]
                          MT1 = int(MatrixInstN * waveTileN * wave[1])

                          if MT1 < MIN_MT1:
                              continue
                          if MT1 > MAX_MT1:
                              break

                          # LDS size check for lsu
                          LSU = max(1, 4 // wave[0] // wave[1])
                          if LSU > 1 and MT0 * MT1 * computeDataTypeSize[self._gt.data_type] * LSU > LSUTHRESHOLD:
                              continue

                          if MT0 * MT1 > mt_max_size[self._gt.data_type]:
                              continue

                          valid_mfmas.append(
                              MFMA(
                                  M=MI[0],
                                  N=MI[1],
                                  K=MI[2],
                                  B=MI[3],
                                  MIBlockM=MIBlockM,
                                  waveTileM=waveTileM,
                                  waveTileN=waveTileN,
                                  waveM=wave[0],
                                  waveN=wave[1],
                              )
                          )
      logger.info(" Total number of valid MatrixInstructions: %s", len(valid_mfmas))
      return valid_mfmas, smallest_M_in_MFMA, smallest_N_in_MFMA

  def _find_mi_for_size(self, valid_mfmas: Sequence[MFMA], smallest_M_in_MFMA: int, smallest_N_in_MFMA: int, size: Tuple[int, int, int, int]) -> Tuple[List[MFMACandidate], float]:
    """Filter MIs for a single size: level 1 + level 2."""

    mfma_list: List[MFMACandidate] = []
    max_TilesPerCU = 0.0
    min_TotalTile = sys.float_info.max
    max_totalGranularity = -1.0

    for mfma in valid_mfmas:
        mfma_params = self.calculate_mfma_parameters(MI=mfma)

        # remove MI4x4 for larger MN
        if self.config["MI_FILTER"] > 0 and (mfma.M == 4 and (size[0] >= 16 and size[1] >= 16)):
            continue

        max_possible_LSU = int(4 / (mfma.waveN * mfma.waveM))

        maxGSU = max(math.floor(size[3]/MinKGSU), 1) 

        # Skip [1, 2], [2, 1], and [1, 1] (and LSU>1) waves for larger sizes 
        if self.config["MI_FILTER"] > 0 and ((size[0] * size[1] > 65536) and (size[0] >= 64 and size[1] >= 64) and (mfma.waveM * mfma.waveN != 4)):
            continue

        if self.config["MI_FILTER"] > 0: # no MI filter (if MI_FILTER is 1 or 2)

            # To remove large edge MFMAs
            if ((size[0] < smallest_M_in_MFMA and mfma_params.MT0 > smallest_M_in_MFMA)  or (size[1] < smallest_N_in_MFMA and mfma_params.MT1 > smallest_N_in_MFMA)):
                continue

            coe = 2 if self.config["StreamK"] else 1

            # to remove all MIs that one dimension is edge
            # for M=16< we still want to test MI16x16, rather than just MI4x4
            if ((mfma_params.MT0 > 16 and mfma_params.MT0 // coe > size[0] and size[0] >= smallest_M_in_MFMA) or (mfma_params.MT1 > 16 and mfma_params.MT1 // coe > size[1] and size[1] >= smallest_N_in_MFMA)):
                continue

        for LSU in range(1, max_possible_LSU+1):
          if LSU == 3: continue

          for GSU in range(1, maxGSU+1):

            granular_metrics = self.calculate_granularities(
                mfma_params.MT0, mfma_params.MT1, size[0], size[1], size[2], self.config['CUs'], LSU,  GSU, [mfma.waveM, mfma.waveN])

            # This condition removes MIs with less than 4 waves. 
            if self.config["MI_FILTER"] > 0 and (mfma.waveM * mfma.waveN * LSU < 4 and (mfma.waveTileM > 1 or mfma.waveTileN > 1)):
                continue

            """
            For large sizes we do not want MIs that do not have 4 weaves. 
            For small MN sizes but batched, this is an exception as the numRounds might be large
            even with < 4 waves
            """
            if self.config["MI_FILTER"] > 0 and (granular_metrics.TilesPerCU >= 2.0 and (mfma.waveN * mfma.waveM) < 4
                and not (size[0] < smallest_M_in_MFMA and size[1] < smallest_N_in_MFMA)): 
                continue

            if not self.config['StreamK']: # DP tuning
                #TODO BBK, check with Alex on the workspace size for streamk
                WorkspaceSizePerElemC = computeDataTypeSize[self._gt.compute_data_type]
                gsuMultiplier = GSU if GSU > 1 else 0

                if size[0] * size[1] * size[2] * WorkspaceSizePerElemC * gsuMultiplier > MAX_GSU_WORKSPACE_SIZE:
                    break

                # TODO: BBK/Koji, please check this condition.
                if (LSU > 1 and ((mfma_params.MT0*mfma_params.MT1*LSU*computeDataTypeSize[self._gt.data_type] > 64*1024) or (mfma_params.MT0*mfma_params.MT1*LSU*computeDataTypeSize[self._gt.data_type] >= 64*1024 and self._gt.transA == "N" and self._gt.transB == "T"))):
                    break
                
            max_TilesPerCU = max(max_TilesPerCU, granular_metrics.TilesPerCU)
            min_TotalTile = min(min_TotalTile, granular_metrics.TotalTiles)
            max_totalGranularity = max(max_totalGranularity, granular_metrics.totalGranularity)

            # TODO for streamK, GSU should be 1, are we taking care of that later? IN the old implementation, GSU will set to 1 for streamK in the fiter function. 
            candidate = MFMACandidate(
                totalGranularity=granular_metrics.totalGranularity,
                TilesPerCU=granular_metrics.TilesPerCU,
                mfma=mfma,
                NumTile0=granular_metrics.NumTile0,
                NumTile1=granular_metrics.NumTile1,
                Tile0Granularity=granular_metrics.Tile0Granularity,
                Tile1Granularity=granular_metrics.Tile1Granularity,
                TotalTiles=granular_metrics.TotalTiles,
                CUGranularity=granular_metrics.CUGranularity,
                waveGranularity=granular_metrics.waveGranularity,
                GSU=GSU,
                LSU=LSU,
                MT0=mfma_params.MT0,
                MT1=mfma_params.MT1,
                TT0=mfma_params.TT0,
                TT1=mfma_params.TT1,
                WG0=mfma_params.WG0,
                WG1=mfma_params.WG1,
                MIBlockM=mfma_params.MIBlockM
            )
            mfma_list.append(candidate)

            # To remove unnecessary large GSUs - if with a smaller GSU, we can reach totalGranularity=1, 
            # checking larger GSUs, just increases TilesPerCU, even though we may get totalGranularity=1 with larger GSU again. So skip it.
            # Example: 1024x1024x8192 can reach to 256CU with MT256x256_GSU16, once we reach to 16, there is 
            if granular_metrics.totalGranularity == 1:
                break

    logger.info(" # Total MIs for %s after level 1 filtering: %s              ", size, len(mfma_list))
    if len(mfma_list) == 0:
        logger.warning("No MI exists for %s after level 1. Try MI_FILTER = 1 or 0. Otherwise, inform GEMM team.", size)        


    if self.config["MI_FILTER"] > 1 : # to filter the most MIs, there is a chance to miss some MT/MIs
        min_rounds = math.ceil(min_TotalTile/self.config["CUs"])

        mfma_indices_to_remove = []

        logger.debug("refine filter (MI_FILTER = 2) for %s:", size)
        for j in range(len(mfma_list)):
            mfma_cand = mfma_list[j]

            num_rounds = math.ceil(mfma_cand.TilesPerCU)

            # triming the tail of each bucket
            # filter based on the MI granularities vs the max_granularity

            # TODO: What about batched sizes?
            if size[0] * size[1] >= 256*256:
                totalGranularity_threshold = 0.85
            else:
                totalGranularity_threshold = 0.5
            
            if max_totalGranularity == 1.0 and mfma_cand.totalGranularity < totalGranularity_threshold:
                mfma_indices_to_remove.append(j)
                logger.debug(" gran_128x128, filtered: %s", mfma_cand)

            if mfma_cand.totalGranularity < GRANTHRESHOLD * max_totalGranularity and max_totalGranularity > 0.2:
                mfma_indices_to_remove.append(j)
                logger.debug(" gran_128x128, filtered: %s", mfma_cand)

            # if (MT0*MT1 > 128*128):
            #     # this is to remove MTs that results in low granularity tiles
            #     if mfma_cand.totalGranularity < GRANTHRESHOLD_128x128 and GRANTHRESHOLD_128x128 < max_totalGranularity:
            #         mfma_indices_to_remove.append(j)
            #         logger.debug(" gran_128x128, filtered: %s", mfma_cand)
            # elif (MT0*MT1 >= 64*32):
            #     if mfma_cand.totalGranularity < GRANTHRESHOLD_64x32 * max_totalGranularity:
            #         mfma_indices_to_remove.append(j)  # comp-bound
            #         logger.debug(" gran_64x32, filtered: %s", mfma_cand)
            # else:
            #     if mfma_cand.totalGranularity < GRANTHRESHOLD_SMALL * max_totalGranularity:
            #         mfma_indices_to_remove.append(j)  # mem-bound
            #         logger.debug(" gran_small, filtered: %s", mfma_cand)
           
            
            if min_rounds > 2:
                num_CU_rounds_comp_bound = min_rounds + 1
            else:
                num_CU_rounds_comp_bound = ROUND1 + min_rounds
            
            # removing buckets
            # filter based on number of CU rounds/min_totalTile
            if (min_TotalTile >= self.config["CUs"]*0.15 and num_rounds > num_CU_rounds_comp_bound and mfma_cand.MT0 * mfma_cand.MT1 < 256*256):  # comp-bound 
                mfma_indices_to_remove.append(j)
                logger.debug(" CU_round_1, filtered: %s", mfma_cand) 
            elif (min_TotalTile < self.config["CUs"]*0.09 and num_rounds > ROUND2 + min_rounds):  # mem-bound 
                mfma_indices_to_remove.append(j)
                logger.debug(" CU_round_2, filtered: %s", mfma_cand)
            elif (min_TotalTile < self.config["CUs"]*0.15 and num_rounds > ROUND3 + min_rounds):  # mem-bound 
                mfma_indices_to_remove.append(j)
                logger.debug(" CU_round_3, filtered: %s", mfma_cand)

        mfma_indices_to_remove = set(mfma_indices_to_remove)
        mfma_list = [mfma_list[j] for j in range(len(mfma_list)) if j not in mfma_indices_to_remove]

        logger.info(" # Total MIs for %s after level 2 filtering: %s", size, len(mfma_list))
        if len(mfma_list) == 0:
            logger.warning("No MI exists for %s after refining level 1 MIs. Try MI_FILTER = 1 or 0. Otherwise, inform GEMM team.", size)        

        # bbk, uncomment after review
        # groups_to_remove = []
        # for mfma_entry in mfma_list:
        #     (totalGranularity, TilesPerCU, mfma, NumTile0, NumTile1, Tile0Granularity, Tile1Granularity, TotalTiles, CUGranularity, waveGranularity, GSU, LSU, MT0, MT1, TT0, TT1, WG0, WG1, MIBlockM) = mfma_entry
        #     # Threshold to remove MI with large MT for small sizes, which causes the TilesPerCU becomes very smaller (<<1).
        #     # Should be less than 1. The smaller this threshold is, the larger the number of MIs are in the outputs.
        #     # self.config['TILETHRESHOLD'] = 0.85  # TODO: BBK see where to add this
        #     # if (max_TilesPerCU > 1.0 and TilesPerCU < self.config['TILETHRESHOLD']):
        #     #     logger.debug("TILETHRESHOLD, filtered: size %s, MI %s", size, mfma_entry)
        #     #     groups_to_remove.append(mfma_entry)
        # for group in groups_to_remove:
        #     mfma_list.remove(group)

    return mfma_list, max_TilesPerCU

  def _sort_mfmas(self, mfma_list: List[MFMACandidate]) -> List[MFMACandidate]:
    """Sort MFMAs by granularity heuristics.

    Prefer MFMAs with lower rounds -> this creates round buckets.
    If rounds same, order by decreasing number of TilesPerCU.
    If TilesPerCU same, order by decreasing order of totalGranularity.
    If totalGranularity is the same, prefer higher GSU.
    """
    mfma_list.sort(key=lambda entry: (math.ceil(entry.TilesPerCU), 1-entry.TilesPerCU, 1-entry.totalGranularity, -entry.GSU))
    return mfma_list
  
  def _remove_GSU_duplicates(self, mfma_list: List[MFMACandidate]) -> List[MFMACandidate]:
    """Remove entries that differ only in GSU by deduplicating on (mfma, LSU)."""
    seen = set()
    unique = []
    for entry in mfma_list:
      key = (entry.mfma, entry.LSU)  # (mfma, LSU)
      if key not in seen:
        seen.add(key)
        unique.append(entry)

    logger.info(" # Total MIs after level 3 filtering (GSU dedup): %s", len(unique))
    if len(unique) == 0:
      logger.warning("No MI exists after GSU dedup. Try MI_FILTER = 1 or 0. Otherwise, inform GEMM team.")

    return unique

  def get_mi_finder_log_name(self, size: Sequence[int]) -> Path:
    """Build the MI finder log path for a given size."""

    GEMM_type = self._gt.gemm_name
    M_dim, N_dim, B_dim, K_dim = size
    catName = f'_M{M_dim}'+f'_N{N_dim}'+f'_B{B_dim}'+f'_K{K_dim}'
    # TODO: match the name with the lib name convention
    return self.outputfile / (GEMM_type + catName + '.log')

  def _create_mi_groups(self, size: Tuple[int, int, int, int], mfma_list: List[MFMACandidate]) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
    """Build MFMA group dicts and comments for a single size.

    Returns:
        (mi_groups, comments, metadata) — parallel lists of group dicts, comment strings, and metadata dicts.
    """
    mi_groups: List[Dict[str, Any]]  = []
    comments: List[str] = []
    metadata: List[Dict[str, Any]] = []

    mi_log_path = self.get_mi_finder_log_name(size)
    log_output = f"Size - {size}\n"

    for idx, mfma_cand in enumerate(mfma_list):
        comment = "MT {:7} - TT {:6} - WG {:6} - MIBlockM {:2} - GSU {:3} - LSU {:2} - totalGranularity {:8.5f} - TilesPerCU: {:8.5f} - TotalTiles: {:8} -- sizes [{}]".format(
            "{}x{}".format(mfma_cand.MT0, mfma_cand.MT1), "{}x{}".format(mfma_cand.TT0, mfma_cand.TT1), "{}x{}".format(mfma_cand.WG0, mfma_cand.WG1), mfma_cand.MIBlockM, mfma_cand.GSU, mfma_cand.LSU, mfma_cand.totalGranularity, mfma_cand.TilesPerCU, mfma_cand.TotalTiles, size)
        mfma_dict = {"MatrixInstruction": [mfma_cand.mfma.M, mfma_cand.mfma.N, mfma_cand.mfma.K, mfma_cand.mfma.B, mfma_cand.mfma.MIBlockM, mfma_cand.mfma.waveTileM, mfma_cand.mfma.waveTileN, mfma_cand.mfma.waveM, mfma_cand.mfma.waveN]}
        log_output += "# - MatrixInstruction: [{:2}, {:2}, {:2}, {:2}, {:2}, {:2}, {:2}, {:2}, {:2}] # {}\n".format(mfma_cand.mfma.M, mfma_cand.mfma.N, mfma_cand.mfma.K, mfma_cand.mfma.B, mfma_cand.mfma.MIBlockM, mfma_cand.mfma.waveTileM, mfma_cand.mfma.waveTileN, mfma_cand.mfma.waveM, mfma_cand.mfma.waveN, comment)

        if mfma_cand.LSU > 1:
            mfma_dict["WorkGroup"] = [mfma_cand.WG0, mfma_cand.WG1, mfma_cand.LSU]
            log_output += f"#   WorkGroup: [{mfma_cand.WG0},{mfma_cand.WG1},{mfma_cand.LSU}]\n"
        if mfma_cand.GSU > 1:
            log_output += f"#   GlobalSplitU: [{mfma_cand.GSU}]\n"
            if (not self.config["StreamK"]):
                mfma_dict["GlobalSplitU"] = [mfma_cand.GSU]
        
        if mfma_dict not in mi_groups: # This is to avoid duplicate MIs in StreamK.
            mi_groups.append(mfma_dict)
            comments.append(comment)
            metadata.append({
                "MT": (mfma_cand.MT0, mfma_cand.MT1),
                "GSU": mfma_cand.GSU,
                "LSU": mfma_cand.LSU,
                "wave": (mfma_cand.mfma.waveM, mfma_cand.mfma.waveN),
            })

    with open(mi_log_path,'w') as out:
        out.write(log_output)

    return mi_groups, comments, metadata

  def __init__(self, outputfile: str | Path, config: Dict[str, Any]) -> None:
      """Initialize MI designer. Enumerates all valid MFMA configurations
      once (one-time setup). Call generate_for_size() per size."""

      self.config = config
      self.outputfile = Path(outputfile)
      self._gt = config["GemmProblem"].gemm_type

      # Config must include GemmProblem (from load_prepared_config_from_yaml, optim.configure,
      # or equivalent) and ARCH/hardware defaults from apply_input_config_defaults.

      self._valid_mfmas, self._smallest_M, self._smallest_N = self.generate_all_mfmas()

  def generate_for_size(self, size: Tuple[int, int, int, int]) -> GroupDimension:
      """Run the per-size MI pipeline: filter, sort, dedup, build groups.

      *size* is ``(M, N, B, K)``.

      Returns:
          GroupDimension — list of dicts mapping param names to
          ForkParameter instances.
      """
      size = tuple(size)

      mfma_list, max_TilesPerCU = self._find_mi_for_size(
          self._valid_mfmas, self._smallest_M, self._smallest_N, size)

      mfma_list = self._sort_mfmas(mfma_list)
      if self.config.get("StreamK", False):
        logger.debug("Removing GSU duplicates for StreamK")
        mfma_list = self._remove_GSU_duplicates(mfma_list)

      raw_groups, comments, metadata = self._create_mi_groups(size, mfma_list)

      return self._to_group_dimension(raw_groups, comments, metadata)

  @staticmethod
  def _to_group_dimension(raw_groups: List[Dict[str, Any]],
                          comments: List[str],
                          metadata: List[Dict[str, Any]]) -> GroupDimension:
      """Wrap raw MI group dicts into ForkParameter instances."""
      result: GroupDimension = []
      for idx, grp in enumerate(raw_groups):
          entry: Dict[str, ForkParameter] = {}
          comment = comments[idx] if idx < len(comments) else ""
          meta = metadata[idx] if idx < len(metadata) else {}
          for name, values in grp.items():
              entry[name] = ForkParameter(
                  name=name,
                  values=values,
                  comment=comment if name == "MatrixInstruction" else "",
                  metadata=meta if name == "MatrixInstruction" else {}
              )
          result.append(entry)
      return result

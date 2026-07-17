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
from pathlib import Path
from geko.config_generator.constants import *

from typing import Any, Dict, List, Sequence, Tuple

from geko.config_generator.shared_utils import ForkParameter, GroupDimension

logger = logging.getLogger("GEKO")


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
  ) -> Tuple[float, float, float, float, float, float, float, float, float]:
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

      return NumTile0, NumTile1, Tile0Granularity, Tile1Granularity, TotalTiles, TilesPerCU, CUGranularity, waveGranularity, totalGranularity

  @staticmethod
  def calculate_mfma_parameters(MI: Sequence[int], waveFrontSize: int = 64) -> Tuple[int, int, int, int, int, int, int]:
      """Compute MFMA-derived parameters for a MatrixInstruction."""
      wave = MI[7], MI[8]
      MIBlockM = MI[4]
      waveTileM, waveTileN = MI[5], MI[6]

      MatrixInstM = MI[0] * MIBlockM
      MT0 = MatrixInstM * waveTileM * wave[0]

      MatrixInstN = MI[1] / MIBlockM * MI[3]
      MT1 = int(MatrixInstN * waveTileN * wave[1])
      TT0 = waveTileM
      TT1 = waveTileN * MI[1]
      WG0 = MatrixInstM * wave[0]
      WG1 = int(wave[0] * wave[1] * waveFrontSize / WG0)

      return MT0, MT1, TT0, TT1, WG0, WG1, MIBlockM

  def generate_all_mfmas(self) -> Tuple[List[List[int]], int, int]:
      """Generate all valid MFMA configurations for the current config.

      Returns:
          (valid_mfmas, smallest_M_in_MFMA, smallest_N_in_MFMA). The smallest
          dimensions are taken from the effective MFMA allowlist (including
          a single MFMA override from config).
      """
      valid_mfmas = []

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
                      waveTileM+=1
                      waveTileN=0
                      MatrixInstM = MI[0] * MIBlockM
                      MT0 = MatrixInstM * waveTileM * wave[0]
                      if MT0 < MIN_MT0:
                          continue
                      if MT0 > MAX_MT0:
                          break

                      while True:
                          waveTileN+=1
                          MatrixInstN = MI[1] / MIBlockM * MI[3]
                          MT1 = int(MatrixInstN * waveTileN * wave[1])

                          if MT1< MIN_MT1:
                              continue
                          if MT1 > MAX_MT1:
                              break

                          # LDS size check for lsu
                          LSU = max(1, 4//wave[0]//wave[1])
                          if LSU > 1 and MT0*MT1*computeDataTypeSize[self._gt.data_type]*LSU > LSUTHRESHOLD:
                            continue

                          if MT0*MT1 > LIST_OF_MT_MAX_SIZE[self._gt.data_type]:
                            continue

                          valid_mfmas.append([MI[0], MI[1], MI[2], MI[3], MIBlockM, waveTileM, waveTileN, wave[0], wave[1]])
      logger.info(" Total number of valid MatrixInstructions: %s", len(valid_mfmas))
      return valid_mfmas, smallest_M_in_MFMA, smallest_N_in_MFMA

  def _find_mi_for_size(self, valid_mfmas: Sequence[Sequence[int]], smallest_M_in_MFMA: int, smallest_N_in_MFMA: int, size: Tuple[int, int, int, int]) -> Tuple[List[Tuple[Any, ...]], float]:
    """Filter MIs for a single size: level 1 + level 2."""

    mfma_list: List[Tuple[Any, ...]] = []
    max_TilesPerCU = 0.0
    min_TotalTile = sys.float_info.max
    max_totalGranularity = -1.0

    for mfma in valid_mfmas:
        MT0, MT1, TT0, TT1, WG0, WG1, MIBlockM = self.calculate_mfma_parameters(MI=mfma)

        # remove MI4x4 for larger MN
        if self.config["MI_FILTER"] > 0 and (mfma[0] == 4 and (size[0] >= 16 and size[1] >= 16)):
            continue

        max_possible_LSU = int(4 / (mfma[-1] * mfma[-2]))

        maxGSU = max(math.floor(size[3]/MinKGSU), 1) 

        # Skip [1, 2], [2, 1], and [1, 1] (and LSU>1) waves for larger sizes 
        if self.config["MI_FILTER"] > 0 and ((size[0] * size[1] > 65536) and (size[0] >= 64 and size[1] >= 64) and (mfma[-2] * mfma[-1] != 4)):
            continue

        if self.config["MI_FILTER"] > 0: # no MI filter (if MI_FILTER is 1 or 2)

            # To remove large edge MFMAs
            if ((size[0] < smallest_M_in_MFMA and MT0 > smallest_M_in_MFMA)  or (size[1] < smallest_N_in_MFMA and MT1 > smallest_N_in_MFMA)):
                continue

            coe = 2 if self.config["StreamK"] else 1

            # to remove all MIs that one dimension is edge
            # for M=16< we still want to test MI16x16, rather than just MI4x4
            if ((MT0 > 16 and MT0 // coe > size[0] and size[0] >= smallest_M_in_MFMA) or (MT1 > 16 and MT1 // coe > size[1] and size[1] >= smallest_N_in_MFMA)):
                continue

        for LSU in range(1, max_possible_LSU+1):
          if LSU == 3: continue

          for GSU in range(1, maxGSU+1):

            NumTile0, NumTile1, Tile0Granularity, Tile1Granularity, TotalTiles, \
            TilesPerCU, CUGranularity, waveGranularity, totalGranularity = self.calculate_granularities(
                MT0, MT1, size[0], size[1], size[2], self.config['CUs'], LSU,  GSU, [mfma[7], mfma[8]])

            # This condition removes MIs with less than 4 waves. 
            if self.config["MI_FILTER"] > 0 and (mfma[7] * mfma[8] * LSU < 4 and (mfma[5] > 1 or mfma[6] > 1)):
                continue

            """
            For large sizes we do not want MIs that do not have 4 weaves. 
            For small MN sizes but batched, this is an exception as the numRounds might be large
            even with < 4 waves
            """
            if self.config["MI_FILTER"] > 0 and (TilesPerCU >= 2.0 and (mfma[-1] * mfma[-2]) < 4
                and not (size[0] < smallest_M_in_MFMA and size[1] < smallest_N_in_MFMA)): 
                continue

            if not self.config['StreamK']: # DP tuning
                #TODO BBK, check with Alex on the workspace size for streamk
                WorkspaceSizePerElemC = computeDataTypeSize[self._gt.compute_data_type]
                gsuMultiplier = GSU if GSU > 1 else 0

                if size[0] * size[1] * size[2] * WorkspaceSizePerElemC * gsuMultiplier > MAX_GSU_WORKSPACE_SIZE:
                    break

                # TODO: BBK/Koji, please check this condition.
                if (LSU > 1 and ((MT0*MT1*LSU*computeDataTypeSize[self._gt.data_type] > 64*1024) or (MT0*MT1*LSU*computeDataTypeSize[self._gt.data_type] >= 64*1024 and self._gt.transA == "N" and self._gt.transB == "T"))):
                    break
                
            max_TilesPerCU = max(max_TilesPerCU, TilesPerCU)
            min_TotalTile = min(min_TotalTile, TotalTiles)
            max_totalGranularity = max(max_totalGranularity, totalGranularity)

            # TODO for streamK, GSU should be 1, are we taking care of that later? IN the old implementation, GSU will set to 1 for streamK in the fiter function. 
            mfma_list.append((totalGranularity, TilesPerCU, tuple(mfma), NumTile0, NumTile1, Tile0Granularity,
                                                    Tile1Granularity, TotalTiles, CUGranularity, waveGranularity, GSU, LSU, MT0, MT1, TT0, TT1, WG0, WG1, MIBlockM))

            # To remove unnecessary large GSUs - if with a smaller GSU, we can reach totalGranularity=1, 
            # checking larger GSUs, just increases TilesPerCU, even though we may get totalGranularity=1 with larger GSU again. So skip it.
            # Example: 1024x1024x8192 can reach to 256CU with MT256x256_GSU16, once we reach to 16, there is 
            if totalGranularity == 1:
                break

    logger.info(" # Total MIs for %s after level 1 filtering: %s              ", size, len(mfma_list))
    if len(mfma_list) == 0:
        logger.warning("No MI exists for %s after level 1. Try MI_FILTER = 1 or 0. Otherwise, inform GEMM team.", size)        


    if self.config["MI_FILTER"] > 1 : # to filter the most MIs, there is a chance to miss some MT/MIs
        min_rounds = math.ceil(min_TotalTile/self.config["CUs"])

        mfma_indices_to_remove = []

        logger.debug("refine filter (MI_FILTER = 2) for %s:", size)
        for j in range(len(mfma_list)):
            mfma = mfma_list[j]

            MT0, MT1 = mfma[-7], mfma[-6]
            cur_totalGranularity = mfma[0]
            num_rounds = math.ceil(mfma[1])  # math.ceil(TilesPerCU)

            # triming the tail of each bucket
            # filter based on the MI granularities vs the max_granularity

            # TODO: What about batched sizes?
            if size[0] * size[1] >= 256*256:
                totalGranularity_threshold = 0.85
            else:
                totalGranularity_threshold = 0.5
            
            if max_totalGranularity == 1.0 and cur_totalGranularity < totalGranularity_threshold:
                mfma_indices_to_remove.append(j)
                logger.debug(" gran_128x128, filtered: %s", mfma)

            if cur_totalGranularity < GRANTHRESHOLD * max_totalGranularity and max_totalGranularity > 0.2:
                mfma_indices_to_remove.append(j)
                logger.debug(" gran_128x128, filtered: %s", mfma)

            # if (MT0*MT1 > 128*128):
            #     # this is to remove MTs that results in low granularity tiles
            #     if cur_totalGranularity < GRANTHRESHOLD_128x128 and GRANTHRESHOLD_128x128 < max_totalGranularity:
            #         mfma_indices_to_remove.append(j)
            #         logger.debug(" gran_128x128, filtered: %s", mfma)
            # elif (MT0*MT1 >= 64*32):
            #     if cur_totalGranularity < GRANTHRESHOLD_64x32 * max_totalGranularity:
            #         mfma_indices_to_remove.append(j)  # comp-bound
            #         logger.debug(" gran_64x32, filtered: %s", mfma)
            # else:
            #     if cur_totalGranularity < GRANTHRESHOLD_SMALL * max_totalGranularity:
            #         mfma_indices_to_remove.append(j)  # mem-bound
            #         logger.debug(" gran_small, filtered: %s", mfma)
           
            
            if min_rounds > 2:
                num_CU_rounds_comp_bound = min_rounds + 1
            else:
                num_CU_rounds_comp_bound = ROUND1 + min_rounds
            
            # removing buckets
            # filter based on number of CU rounds/min_totalTile
            if   (min_TotalTile >= self.config["CUs"]*0.15 and num_rounds > num_CU_rounds_comp_bound and MT0 * MT1 < 256*256):  # comp-bound 
                mfma_indices_to_remove.append(j)
                logger.debug(" CU_round_1, filtered: %s", mfma) 
            elif (min_TotalTile < self.config["CUs"]*0.09 and num_rounds > ROUND2 + min_rounds):  # mem-bound 
                mfma_indices_to_remove.append(j)
                logger.debug(" CU_round_2, filtered: %s", mfma)
            elif (min_TotalTile < self.config["CUs"]*0.15 and num_rounds > ROUND3 + min_rounds):  # mem-bound 
                mfma_indices_to_remove.append(j)
                logger.debug(" CU_round_3, filtered: %s", mfma)

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

  def _sort_mfmas(self, mfma_list: List[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
    """Sort MFMAs by granularity heuristics.

    Prefer MFMAs with lower rounds -> this creates round buckets.
    If rounds same, order by decreasing number of TilesPerCU.
    If TilesPerCU same, order by decreasing order of totalGranularity.
    If totalGranularity is the same, prefer higher GSU.
    """
    mfma_list.sort(key=lambda tup: (math.ceil(tup[1]), 1-tup[1], 1-tup[0],-tup[10]))
    return mfma_list
  
  def _remove_GSU_duplicates(self, mfma_list: List[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
    """Remove entries that differ only in GSU by deduplicating on (mfma, LSU)."""
    seen = set()
    unique = []
    for entry in mfma_list:
      key = (entry[2], entry[11])  # (mfma, LSU)
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

  def _create_mi_groups(self, size: Tuple[int, int, int, int], mfma_list: List[Tuple[Any, ...]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build MFMA group dicts and comments for a single size.

    Returns:
        (mi_groups, comments) — parallel lists of group dicts and comment strings.
    """
    mi_groups: List[Dict[str, Any]]  = []
    comments: List[str] = []
    metadata: List[Dict[str, Any]] = []

    mi_log_path = self.get_mi_finder_log_name(size)
    log_output = f"Size - {size}\n"

    for idx, mfma_info in enumerate(mfma_list):
        totalGranularity, TilesPerCU, mfma, NumTile0, NumTile1, Tile0Granularity, Tile1Granularity, TotalTiles, CUGranularity, waveGranularity, GSU, LSU, MT0, MT1, TT0, TT1, WG0, WG1, MIBlockM = mfma_info
        
        comment = "MT {:7} - TT {:6} - WG {:6} - MIBlockM {:2} - GSU {:3} - LSU {:2} - totalGranularity {:8.5f} - TilesPerCU: {:8.5f} - TotalTiles: {:8} -- sizes [{}]".format(
            "{}x{}".format(MT0, MT1), "{}x{}".format(TT0, TT1), "{}x{}".format(WG0, WG1), MIBlockM, GSU, LSU, totalGranularity, TilesPerCU, TotalTiles, size)
        mfma_dict = {"MatrixInstruction": [mfma[0], mfma[1], mfma[2], mfma[3], mfma[4], mfma[5], mfma[6], mfma[7], mfma[8]]}
        log_output += "# - MatrixInstruction: [{:2}, {:2}, {:2}, {:2}, {:2}, {:2}, {:2}, {:2}, {:2}] # {}\n".format(mfma[0], mfma[1], mfma[2], mfma[3], mfma[4], mfma[5], mfma[6], mfma[7], mfma[8], comment)

        if LSU >1:
            mfma_dict["WorkGroup"]=[WG0, WG1, LSU]
            log_output += f"#   WorkGroup: [{WG0},{WG1},{LSU}]\n"
        if GSU >1:
            log_output += f"#   GlobalSplitU: [{GSU}]\n"
            if (not self.config["StreamK"]):
                mfma_dict["GlobalSplitU"]= [GSU]
        
        if mfma_dict not in mi_groups: # This is to avoid duplicate MIs in StreamK.
            mi_groups.append(mfma_dict)
            comments.append(comment)
            metadata.append({
                "MT": (MT0, MT1),
                "GSU": GSU,
                "LSU": LSU,
                "wave": (mfma[7], mfma[8]),
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

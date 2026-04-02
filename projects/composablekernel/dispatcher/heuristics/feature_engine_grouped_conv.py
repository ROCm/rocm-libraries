#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Feature engineering for grouped convolution kernel performance prediction.

Extends the FeatureEngine interface to support grouped convolution operations.
Follows the same pattern as GEMM: hardware parameters are read from the data
(hw_* columns) with fallback defaults for gfx950.
"""

import math
import numpy as np
import pandas as pd

from feature_engine import FeatureEngine, DTYPE_BYTES, PIPELINE_MAP


class GroupedConvFeatureEngine(FeatureEngine):
    """Feature engine for grouped_conv kernels.

    Hardware parameters are initialized from defaults but can be overridden
    by reading from data columns (hw_num_cus, hw_max_clock_mhz, etc.)
    """

    def __init__(
        self,
        num_cus: int = 256,           # gfx950 MI300 default
        lds_capacity: int = 65536,
        max_clock_mhz: int = 2400,
        simds_per_cu: int = 4,
        shader_engines: int = 32,
        max_waves_per_cu: int = 32,
        wavefront_size: int = 64,
        l1_cache_kb: int = 32,
        l2_cache_kb: int = 4096,
        l3_cache_kb: int = 262144,
        num_xcd: int = 8,
    ):
        self._hw = {
            "num_cus": num_cus,
            "lds_capacity": lds_capacity,
            "max_clock_mhz": max_clock_mhz,
            "simds_per_cu": simds_per_cu,
            "shader_engines": shader_engines,
            "max_waves_per_cu": max_waves_per_cu,
            "wavefront_size": wavefront_size,
            "l1_cache_kb": l1_cache_kb,
            "l2_cache_kb": l2_cache_kb,
            "l3_cache_kb": l3_cache_kb,
            "num_xcd": num_xcd,
            "total_simds": num_cus * simds_per_cu,
        }

    def get_feature_names(self) -> list[str]:
        return [
            # Problem features (30)
            "N", "C", "K", "G",
            "Hi", "Wi", "Y", "X",
            "stride_h", "stride_w",
            "pad_h", "pad_w",
            "Ho", "Wo",  # Computed output dimensions
            "log2_N", "log2_C", "log2_K", "log2_G",
            "log2_Hi", "log2_Wi",
            "log2_spatial",  # log2(Hi * Wi)
            "log2_filter",   # log2(Y * X)
            "log2_output",   # log2(Ho * Wo)
            "arithmetic_intensity",
            "filter_area",   # Y * X
            "is_1x1_conv",
            "is_3x3_conv",
            "channels_per_group",  # C / G
            "aspect_ratio_hw",     # Hi / Wi
            "aspect_ratio_filter", # Y / X
            # Kernel features (15)
            "block_size",
            "gemm_m_per_block",
            "gemm_n_per_block",
            "pipeline",
            "num_warps",  # Estimated from block_size
            "tile_volume",  # gemm_m * gemm_n * block_size
            "tile_mn",      # gemm_m * gemm_n
            "lds_usage_estimate",
            "lds_usage_ratio",
            "block_tile_ratio_m",  # gemm_m / block_size
            "block_tile_ratio_n",  # gemm_n / block_size
            "block_efficiency",    # Degree to which block is square-like
            "is_compv3",
            "is_compv4",
            "is_compv5",
            # Interaction features (18)
            "gemm_m_output",  # Effective GEMM M: N * Ho * Wo
            "gemm_n_output",  # Effective GEMM N: K
            "gemm_k_output",  # Effective GEMM K: (C/G) * Y * X
            "num_tiles_m",
            "num_tiles_n",
            "num_tiles_k",
            "total_output_tiles",
            "tile_eff_m",
            "tile_eff_n",
            "tile_eff_k",
            "overall_tile_efficiency",
            "cu_utilization",
            "ratio_gemm_m_to_tile_m",
            "ratio_gemm_n_to_tile_n",
            "ratio_gemm_k_to_tile_k",
            "problem_smaller_than_tile_m",
            "problem_smaller_than_tile_n",
            "problem_smaller_than_tile_k",
            # Hardware features (12)
            "hw_num_cus",
            "hw_simds_per_cu",
            "hw_total_simds",
            "hw_shader_engines",
            "hw_max_clock_mhz",
            "hw_max_waves_per_cu",
            "hw_wavefront_size",
            "hw_lds_capacity",
            "hw_l1_cache_kb",
            "hw_l2_cache_kb",
            "hw_l3_cache_kb",
            "hw_num_xcd",
        ]

    def get_categorical_features(self) -> list[str]:
        return ["pipeline"]

    def extract(self, problem: dict, kernel: dict) -> np.ndarray:
        # Problem features
        N = int(problem.get("N", 1))
        C = int(problem.get("C", 64))
        K = int(problem.get("K", 64))
        G = int(problem.get("G", 1))
        Hi = int(problem.get("Hi", 32))
        Wi = int(problem.get("Wi", 32))
        Y = int(problem.get("Y", 1))
        X = int(problem.get("X", 1))
        stride_h = int(problem.get("stride_h", 1))
        stride_w = int(problem.get("stride_w", 1))
        pad_h = int(problem.get("pad_h", 0))
        pad_w = int(problem.get("pad_w", 0))

        # Compute output dimensions
        Ho = (Hi + 2 * pad_h - Y) // stride_h + 1
        Wo = (Wi + 2 * pad_w - X) // stride_w + 1

        # Log features
        log2_N = math.log2(max(N, 1))
        log2_C = math.log2(max(C, 1))
        log2_K = math.log2(max(K, 1))
        log2_G = math.log2(max(G, 1))
        log2_Hi = math.log2(max(Hi, 1))
        log2_Wi = math.log2(max(Wi, 1))
        log2_spatial = math.log2(max(Hi * Wi, 1))
        log2_filter = math.log2(max(Y * X, 1))
        log2_output = math.log2(max(Ho * Wo, 1))

        # Arithmetic intensity (FLOPs / bytes)
        dtype = str(problem.get("dtype", "bf16"))
        bpe = DTYPE_BYTES.get(dtype, 2.0)

        # FLOPs: N * K * Ho * Wo * (C/G) * Y * X * 2 (MAC)
        flops = N * K * Ho * Wo * (C / max(G, 1)) * Y * X * 2

        # Bytes: input + filter + output
        bytes_transferred = (N * C * Hi * Wi + K * (C / max(G, 1)) * Y * X + N * K * Ho * Wo) * bpe
        ai = flops / max(bytes_transferred, 1)

        # Derived problem features
        filter_area = Y * X
        is_1x1_conv = float(Y == 1 and X == 1)
        is_3x3_conv = float(Y == 3 and X == 3)
        channels_per_group = C / max(G, 1)
        aspect_ratio_hw = Hi / max(Wi, 1)
        aspect_ratio_filter = Y / max(X, 1)

        # Kernel features
        block_size = int(kernel.get("block_size", 16))
        gemm_m_per_block = int(kernel.get("gemm_m_per_block", 64))
        gemm_n_per_block = int(kernel.get("gemm_n_per_block", 64))
        pipeline_str = str(kernel.get("pipeline", "compv3"))
        pipeline_code = PIPELINE_MAP.get(pipeline_str, 0)

        # Estimate warps (assuming 256 thread block)
        num_warps = block_size / 4.0

        tile_volume = gemm_m_per_block * gemm_n_per_block * block_size
        tile_mn = gemm_m_per_block * gemm_n_per_block

        # LDS usage estimate
        lds_est = (gemm_m_per_block * block_size + gemm_n_per_block * block_size) * bpe
        lds_cap = self._hw["lds_capacity"]
        if pipeline_str.startswith("compv4"):
            lds_cap = 32768
        lds_ratio = lds_est / max(lds_cap, 1)

        # Kernel derived features
        block_tile_ratio_m = gemm_m_per_block / max(block_size, 1)
        block_tile_ratio_n = gemm_n_per_block / max(block_size, 1)
        block_efficiency = min(gemm_m_per_block, gemm_n_per_block) / max(gemm_m_per_block, gemm_n_per_block, 1)
        is_compv3 = float(pipeline_str == "compv3")
        is_compv4 = float(pipeline_str == "compv4")
        is_compv5 = float(pipeline_str == "compv5")

        # Interaction features - Map conv to GEMM dimensions
        # GEMM M: N * Ho * Wo (output spatial)
        # GEMM N: K (output channels)
        # GEMM K: (C/G) * Y * X (input channels per group * filter)
        gemm_m = N * Ho * Wo
        gemm_n = K
        gemm_k = int(channels_per_group * Y * X)

        num_tiles_m = math.ceil(gemm_m / max(gemm_m_per_block, 1))
        num_tiles_n = math.ceil(gemm_n / max(gemm_n_per_block, 1))
        num_tiles_k = math.ceil(gemm_k / max(block_size, 1))
        total_output_tiles = num_tiles_m * num_tiles_n

        rem_m = gemm_m % gemm_m_per_block if gemm_m_per_block > 0 else 0
        tile_eff_m = rem_m / gemm_m_per_block if rem_m > 0 else 1.0
        rem_n = gemm_n % gemm_n_per_block if gemm_n_per_block > 0 else 0
        tile_eff_n = rem_n / gemm_n_per_block if rem_n > 0 else 1.0
        rem_k = gemm_k % block_size if block_size > 0 else 0
        tile_eff_k = rem_k / block_size if rem_k > 0 else 1.0
        overall_eff = tile_eff_m * tile_eff_n * tile_eff_k

        cu_util = total_output_tiles / max(self._hw["num_cus"], 1)

        # Problem-to-tile ratios
        ratio_gemm_m_to_tile_m = gemm_m / max(gemm_m_per_block, 1)
        ratio_gemm_n_to_tile_n = gemm_n / max(gemm_n_per_block, 1)
        ratio_gemm_k_to_tile_k = gemm_k / max(block_size, 1)

        problem_smaller_than_tile_m = float(gemm_m < gemm_m_per_block)
        problem_smaller_than_tile_n = float(gemm_n < gemm_n_per_block)
        problem_smaller_than_tile_k = float(gemm_k < block_size)

        hw = self._hw
        return np.array(
            [
                # Problem features (30)
                N, C, K, G, Hi, Wi, Y, X,
                stride_h, stride_w, pad_h, pad_w,
                Ho, Wo,
                log2_N, log2_C, log2_K, log2_G,
                log2_Hi, log2_Wi, log2_spatial, log2_filter, log2_output,
                ai,
                filter_area, is_1x1_conv, is_3x3_conv,
                channels_per_group, aspect_ratio_hw, aspect_ratio_filter,
                # Kernel features (15)
                block_size, gemm_m_per_block, gemm_n_per_block,
                pipeline_code,
                num_warps, tile_volume, tile_mn,
                lds_est, lds_ratio,
                block_tile_ratio_m, block_tile_ratio_n, block_efficiency,
                is_compv3, is_compv4, is_compv5,
                # Interaction features (18)
                gemm_m, gemm_n, gemm_k,
                num_tiles_m, num_tiles_n, num_tiles_k,
                total_output_tiles,
                tile_eff_m, tile_eff_n, tile_eff_k,
                overall_eff, cu_util,
                ratio_gemm_m_to_tile_m, ratio_gemm_n_to_tile_n, ratio_gemm_k_to_tile_k,
                problem_smaller_than_tile_m, problem_smaller_than_tile_n, problem_smaller_than_tile_k,
                # Hardware features (12)
                hw["num_cus"], hw["simds_per_cu"], hw["total_simds"],
                hw["shader_engines"], hw["max_clock_mhz"],
                hw["max_waves_per_cu"], hw["wavefront_size"],
                hw["lds_capacity"],
                hw["l1_cache_kb"], hw["l2_cache_kb"], hw["l3_cache_kb"],
                hw["num_xcd"],
            ],
            dtype=np.float64,
        )

    def extract_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorized batch extraction -- much faster than row-by-row."""
        n = len(df)
        names = self.get_feature_names()
        result = np.zeros((n, len(names)), dtype=np.float64)

        # Extract problem features
        N = df["N"].values.astype(np.float64)
        C = df["C"].values.astype(np.float64)
        K = df["K"].values.astype(np.float64)
        G = df["G"].values.astype(np.float64)
        Hi = df["Hi"].values.astype(np.float64)
        Wi = df["Wi"].values.astype(np.float64)
        Y = df["Y"].values.astype(np.float64)
        X = df["X"].values.astype(np.float64)
        stride_h = df["stride_h"].values.astype(np.float64)
        stride_w = df["stride_w"].values.astype(np.float64)
        pad_h = df["pad_h"].values.astype(np.float64)
        pad_w = df["pad_w"].values.astype(np.float64)

        # Compute output dimensions
        Ho = (Hi + 2 * pad_h - Y) // stride_h + 1
        Wo = (Wi + 2 * pad_w - X) // stride_w + 1

        # Log features
        log2_N = np.log2(np.maximum(N, 1))
        log2_C = np.log2(np.maximum(C, 1))
        log2_K = np.log2(np.maximum(K, 1))
        log2_G = np.log2(np.maximum(G, 1))
        log2_Hi = np.log2(np.maximum(Hi, 1))
        log2_Wi = np.log2(np.maximum(Wi, 1))
        log2_spatial = np.log2(np.maximum(Hi * Wi, 1))
        log2_filter = np.log2(np.maximum(Y * X, 1))
        log2_output = np.log2(np.maximum(Ho * Wo, 1))

        # Arithmetic intensity
        dtype = df["dtype"].iloc[0] if "dtype" in df.columns else "bf16"
        bpe = DTYPE_BYTES.get(dtype, 2.0)

        flops = N * K * Ho * Wo * (C / np.maximum(G, 1)) * Y * X * 2
        bytes_transferred = (N * C * Hi * Wi + K * (C / np.maximum(G, 1)) * Y * X + N * K * Ho * Wo) * bpe
        ai = flops / np.maximum(bytes_transferred, 1)

        # Derived problem features
        filter_area = Y * X
        is_1x1_conv = ((Y == 1) & (X == 1)).astype(np.float64)
        is_3x3_conv = ((Y == 3) & (X == 3)).astype(np.float64)
        channels_per_group = C / np.maximum(G, 1)
        aspect_ratio_hw = Hi / np.maximum(Wi, 1)
        aspect_ratio_filter = Y / np.maximum(X, 1)

        # Kernel features
        block_size = df["block_size"].values.astype(np.float64)
        gemm_m_per_block = df["gemm_m_per_block"].values.astype(np.float64)
        gemm_n_per_block = df["gemm_n_per_block"].values.astype(np.float64)
        pipeline_code = df["pipeline"].map(PIPELINE_MAP).fillna(0).values.astype(np.float64)

        num_warps = block_size / 4.0
        tile_volume = gemm_m_per_block * gemm_n_per_block * block_size
        tile_mn = gemm_m_per_block * gemm_n_per_block

        # LDS usage
        lds_est = (gemm_m_per_block * block_size + gemm_n_per_block * block_size) * bpe
        lds_cap = np.full(n, self._hw["lds_capacity"], dtype=np.float64)
        is_compv4 = (df["pipeline"] == "compv4").values
        lds_cap[is_compv4] = 32768
        lds_ratio = lds_est / np.maximum(lds_cap, 1)

        # Kernel derived features
        block_tile_ratio_m = gemm_m_per_block / np.maximum(block_size, 1)
        block_tile_ratio_n = gemm_n_per_block / np.maximum(block_size, 1)
        block_efficiency = np.minimum(gemm_m_per_block, gemm_n_per_block) / np.maximum(np.maximum(gemm_m_per_block, gemm_n_per_block), 1)
        is_compv3_arr = (df["pipeline"] == "compv3").values.astype(np.float64)
        is_compv4_arr = (df["pipeline"] == "compv4").values.astype(np.float64)
        is_compv5_arr = (df["pipeline"] == "compv5").values.astype(np.float64)

        # Interaction features
        gemm_m = N * Ho * Wo
        gemm_n = K
        gemm_k = (channels_per_group * Y * X).astype(np.int64)

        num_tiles_m = np.ceil(gemm_m / np.maximum(gemm_m_per_block, 1))
        num_tiles_n = np.ceil(gemm_n / np.maximum(gemm_n_per_block, 1))
        num_tiles_k = np.ceil(gemm_k / np.maximum(block_size, 1))
        total_output_tiles = num_tiles_m * num_tiles_n

        rem_m = np.where(gemm_m_per_block > 0, gemm_m % gemm_m_per_block, 0)
        tile_eff_m = np.where(rem_m > 0, rem_m / gemm_m_per_block, 1.0)
        rem_n = np.where(gemm_n_per_block > 0, gemm_n % gemm_n_per_block, 0)
        tile_eff_n = np.where(rem_n > 0, rem_n / gemm_n_per_block, 1.0)
        rem_k = np.where(block_size > 0, gemm_k % block_size, 0)
        tile_eff_k = np.where(rem_k > 0, rem_k / block_size, 1.0)
        overall_eff = tile_eff_m * tile_eff_n * tile_eff_k

        cu_util = total_output_tiles / max(self._hw["num_cus"], 1)

        # Problem-to-tile ratios
        ratio_gemm_m_to_tile_m = gemm_m / np.maximum(gemm_m_per_block, 1)
        ratio_gemm_n_to_tile_n = gemm_n / np.maximum(gemm_n_per_block, 1)
        ratio_gemm_k_to_tile_k = gemm_k / np.maximum(block_size, 1)

        problem_smaller_than_tile_m = (gemm_m < gemm_m_per_block).astype(np.float64)
        problem_smaller_than_tile_n = (gemm_n < gemm_n_per_block).astype(np.float64)
        problem_smaller_than_tile_k = (gemm_k < block_size).astype(np.float64)

        hw = self._hw

        # Assemble feature matrix column by column
        idx = 0
        result[:, idx] = N; idx += 1
        result[:, idx] = C; idx += 1
        result[:, idx] = K; idx += 1
        result[:, idx] = G; idx += 1
        result[:, idx] = Hi; idx += 1
        result[:, idx] = Wi; idx += 1
        result[:, idx] = Y; idx += 1
        result[:, idx] = X; idx += 1
        result[:, idx] = stride_h; idx += 1
        result[:, idx] = stride_w; idx += 1
        result[:, idx] = pad_h; idx += 1
        result[:, idx] = pad_w; idx += 1
        result[:, idx] = Ho; idx += 1
        result[:, idx] = Wo; idx += 1
        result[:, idx] = log2_N; idx += 1
        result[:, idx] = log2_C; idx += 1
        result[:, idx] = log2_K; idx += 1
        result[:, idx] = log2_G; idx += 1
        result[:, idx] = log2_Hi; idx += 1
        result[:, idx] = log2_Wi; idx += 1
        result[:, idx] = log2_spatial; idx += 1
        result[:, idx] = log2_filter; idx += 1
        result[:, idx] = log2_output; idx += 1
        result[:, idx] = ai; idx += 1
        result[:, idx] = filter_area; idx += 1
        result[:, idx] = is_1x1_conv; idx += 1
        result[:, idx] = is_3x3_conv; idx += 1
        result[:, idx] = channels_per_group; idx += 1
        result[:, idx] = aspect_ratio_hw; idx += 1
        result[:, idx] = aspect_ratio_filter; idx += 1
        result[:, idx] = block_size; idx += 1
        result[:, idx] = gemm_m_per_block; idx += 1
        result[:, idx] = gemm_n_per_block; idx += 1
        result[:, idx] = pipeline_code; idx += 1
        result[:, idx] = num_warps; idx += 1
        result[:, idx] = tile_volume; idx += 1
        result[:, idx] = tile_mn; idx += 1
        result[:, idx] = lds_est; idx += 1
        result[:, idx] = lds_ratio; idx += 1
        result[:, idx] = block_tile_ratio_m; idx += 1
        result[:, idx] = block_tile_ratio_n; idx += 1
        result[:, idx] = block_efficiency; idx += 1
        result[:, idx] = is_compv3_arr; idx += 1
        result[:, idx] = is_compv4_arr; idx += 1
        result[:, idx] = is_compv5_arr; idx += 1
        result[:, idx] = gemm_m; idx += 1
        result[:, idx] = gemm_n; idx += 1
        result[:, idx] = gemm_k; idx += 1
        result[:, idx] = num_tiles_m; idx += 1
        result[:, idx] = num_tiles_n; idx += 1
        result[:, idx] = num_tiles_k; idx += 1
        result[:, idx] = total_output_tiles; idx += 1
        result[:, idx] = tile_eff_m; idx += 1
        result[:, idx] = tile_eff_n; idx += 1
        result[:, idx] = tile_eff_k; idx += 1
        result[:, idx] = overall_eff; idx += 1
        result[:, idx] = cu_util; idx += 1
        result[:, idx] = ratio_gemm_m_to_tile_m; idx += 1
        result[:, idx] = ratio_gemm_n_to_tile_n; idx += 1
        result[:, idx] = ratio_gemm_k_to_tile_k; idx += 1
        result[:, idx] = problem_smaller_than_tile_m; idx += 1
        result[:, idx] = problem_smaller_than_tile_n; idx += 1
        result[:, idx] = problem_smaller_than_tile_k; idx += 1
        result[:, idx] = hw["num_cus"]; idx += 1
        result[:, idx] = hw["simds_per_cu"]; idx += 1
        result[:, idx] = hw["total_simds"]; idx += 1
        result[:, idx] = hw["shader_engines"]; idx += 1
        result[:, idx] = hw["max_clock_mhz"]; idx += 1
        result[:, idx] = hw["max_waves_per_cu"]; idx += 1
        result[:, idx] = hw["wavefront_size"]; idx += 1
        result[:, idx] = hw["lds_capacity"]; idx += 1
        result[:, idx] = hw["l1_cache_kb"]; idx += 1
        result[:, idx] = hw["l2_cache_kb"]; idx += 1
        result[:, idx] = hw["l3_cache_kb"]; idx += 1
        result[:, idx] = hw["num_xcd"]; idx += 1

        return result

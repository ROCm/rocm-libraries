// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/host.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace ck_tile {

enum class BlockMaskPattern
{
    None = 0,
    Random,
    Causal,
    Stripe,
    Checkerboard,
    File
};

struct BlockMaskConfig
{
    BlockMaskPattern pattern       = BlockMaskPattern::None;
    float ratio                    = 0.5f; // sparsity ratio (fraction of blocks to mask out)
    ck_tile::index_t block_size_q  = 0;    // Q block size (kM0), set from kernel
    ck_tile::index_t block_size_kv = 0;    // KV block size (kN0), set from kernel
    std::string file_path;                 // for file-based patterns
};

inline BlockMaskConfig parse_block_mask_config(const std::string& str)
{
    BlockMaskConfig config;
    if(str == "none" || str == "0" || str.empty())
    {
        config.pattern = BlockMaskPattern::None;
        return config;
    }

    // Parse pattern:param format
    auto colon_pos          = str.find(':');
    std::string pattern_str = (colon_pos != std::string::npos) ? str.substr(0, colon_pos) : str;
    std::string param_str   = (colon_pos != std::string::npos) ? str.substr(colon_pos + 1) : "";

    if(pattern_str == "random" || pattern_str == "r")
    {
        config.pattern = BlockMaskPattern::Random;
        config.ratio   = param_str.empty() ? 0.5f : std::stof(param_str);
    }
    else if(pattern_str == "causal" || pattern_str == "c")
    {
        config.pattern = BlockMaskPattern::Causal;
    }
    else if(pattern_str == "stripe" || pattern_str == "s")
    {
        config.pattern = BlockMaskPattern::Stripe;
        config.ratio   = param_str.empty() ? 0.5f : std::stof(param_str);
    }
    else if(pattern_str == "checkerboard" || pattern_str == "cb")
    {
        config.pattern = BlockMaskPattern::Checkerboard;
    }
    else if(pattern_str == "file" || pattern_str == "f")
    {
        config.pattern   = BlockMaskPattern::File;
        config.file_path = param_str;
    }
    else
    {
        std::cerr << "Warning: unknown block_mask pattern '" << str << "', using none" << std::endl;
        config.pattern = BlockMaskPattern::None;
    }

    return config;
}

// Generate random block mask with given sparsity ratio
inline std::vector<int32_t> generate_random_block_mask(ck_tile::index_t num_q_blocks,
                                                       ck_tile::index_t num_kv_blocks,
                                                       float sparsity_ratio,
                                                       uint32_t seed = 42)
{
    std::vector<int32_t> mask(num_q_blocks * num_kv_blocks, 1);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for(ck_tile::index_t i = 0; i < num_q_blocks; i++)
    {
        for(ck_tile::index_t j = 0; j < num_kv_blocks; j++)
        {
            mask[i * num_kv_blocks + j] = (dist(rng) >= sparsity_ratio) ? 1 : 0;
        }
    }
    return mask;
}

// Generate causal-aligned block mask
inline std::vector<int32_t> generate_causal_block_mask(ck_tile::index_t num_q_blocks,
                                                       ck_tile::index_t num_kv_blocks,
                                                       ck_tile::index_t block_size_q,
                                                       ck_tile::index_t block_size_kv)
{
    std::vector<int32_t> mask(num_q_blocks * num_kv_blocks, 0);

    for(ck_tile::index_t qi = 0; qi < num_q_blocks; qi++)
    {
        // The last Q position in this Q block
        ck_tile::index_t q_end = (qi + 1) * block_size_q - 1;
        for(ck_tile::index_t ki = 0; ki < num_kv_blocks; ki++)
        {
            // The first KV position in this KV block
            ck_tile::index_t kv_start = ki * block_size_kv;
            // Block is active if any Q position can attend to any KV position
            // i.e., the last Q position >= first KV position
            mask[qi * num_kv_blocks + ki] = (q_end >= kv_start) ? 1 : 0;
        }
    }
    return mask;
}

// Generate stripe pattern block mask
inline std::vector<int32_t> generate_stripe_block_mask(ck_tile::index_t num_q_blocks,
                                                       ck_tile::index_t num_kv_blocks,
                                                       float ratio)
{
    std::vector<int32_t> mask(num_q_blocks * num_kv_blocks, 0);
    ck_tile::index_t stripe_width = std::max(
        ck_tile::index_t(1), static_cast<ck_tile::index_t>(num_kv_blocks * (1.0f - ratio)));

    for(ck_tile::index_t qi = 0; qi < num_q_blocks; qi++)
    {
        for(ck_tile::index_t ki = 0; ki < num_kv_blocks; ki++)
        {
            // Diagonal stripe: blocks near the diagonal are active
            ck_tile::index_t diag_dist =
                std::abs(qi * num_kv_blocks / std::max(num_q_blocks, ck_tile::index_t(1)) - ki);
            mask[qi * num_kv_blocks + ki] = (diag_dist < stripe_width) ? 1 : 0;
        }
    }
    return mask;
}

// Generate checkerboard pattern block mask (50% sparse)
inline std::vector<int32_t> generate_checkerboard_block_mask(ck_tile::index_t num_q_blocks,
                                                             ck_tile::index_t num_kv_blocks)
{
    std::vector<int32_t> mask(num_q_blocks * num_kv_blocks, 0);

    for(ck_tile::index_t qi = 0; qi < num_q_blocks; qi++)
    {
        for(ck_tile::index_t ki = 0; ki < num_kv_blocks; ki++)
        {
            mask[qi * num_kv_blocks + ki] = ((qi + ki) % 2 == 0) ? 1 : 0;
        }
    }
    return mask;
}

// Main entry point: generate block mask based on config
inline ck_tile::HostTensor<int32_t> generate_block_mask(const BlockMaskConfig& config,
                                                        ck_tile::index_t /*batch*/,
                                                        ck_tile::index_t /*nhead*/,
                                                        ck_tile::index_t max_seqlen_q,
                                                        ck_tile::index_t max_seqlen_k,
                                                        uint32_t seed = 42)
{
    ck_tile::index_t num_q_blocks = (max_seqlen_q + config.block_size_q - 1) / config.block_size_q;
    ck_tile::index_t num_kv_blocks =
        (max_seqlen_k + config.block_size_kv - 1) / config.block_size_kv;

    // For simplicity, we generate a single mask and replicate across batch/head
    // The mask shape is [num_q_blocks, num_kv_blocks]
    std::vector<int32_t> single_mask;

    switch(config.pattern)
    {
    case BlockMaskPattern::None:
        single_mask.assign(num_q_blocks * num_kv_blocks, 1); // all active
        break;
    case BlockMaskPattern::Random:
        single_mask = generate_random_block_mask(num_q_blocks, num_kv_blocks, config.ratio, seed);
        break;
    case BlockMaskPattern::Causal:
        single_mask = generate_causal_block_mask(
            num_q_blocks, num_kv_blocks, config.block_size_q, config.block_size_kv);
        break;
    case BlockMaskPattern::Stripe:
        single_mask = generate_stripe_block_mask(num_q_blocks, num_kv_blocks, config.ratio);
        break;
    case BlockMaskPattern::Checkerboard:
        single_mask = generate_checkerboard_block_mask(num_q_blocks, num_kv_blocks);
        break;
    case BlockMaskPattern::File:
        std::cerr << "File-based block mask not yet implemented" << std::endl;
        single_mask.assign(num_q_blocks * num_kv_blocks, 1); // fallback: all active
        break;
    }

    // Create host tensor [num_q_blocks, num_kv_blocks]
    // Shared across all batches and heads (nhead_stride=0, batch_stride=0)
    ck_tile::HostTensor<int32_t> block_mask_host({num_q_blocks, num_kv_blocks});

    for(ck_tile::index_t qi = 0; qi < num_q_blocks; qi++)
    {
        for(ck_tile::index_t ki = 0; ki < num_kv_blocks; ki++)
        {
            block_mask_host(qi, ki) = single_mask[qi * num_kv_blocks + ki];
        }
    }

    return block_mask_host;
}

inline void print_block_mask_stats(const ck_tile::HostTensor<int32_t>& block_mask,
                                   ck_tile::index_t num_q_blocks,
                                   ck_tile::index_t num_kv_blocks)
{
    ck_tile::index_t total_blocks  = num_q_blocks * num_kv_blocks;
    ck_tile::index_t active_blocks = 0;

    for(ck_tile::index_t qi = 0; qi < num_q_blocks; qi++)
    {
        for(ck_tile::index_t ki = 0; ki < num_kv_blocks; ki++)
        {
            if(block_mask(qi, ki) != 0)
                active_blocks++;
        }
    }

    float sparsity = 1.0f - static_cast<float>(active_blocks) / total_blocks;
    std::cout << "[block_mask] q_blocks=" << num_q_blocks << " kv_blocks=" << num_kv_blocks
              << " active=" << active_blocks << "/" << total_blocks
              << " sparsity=" << (sparsity * 100.0f) << "%" << std::endl;
}

} // namespace ck_tile

#pragma once

#include "config.h"

namespace hipconv::cdna4
{

using direct_l1::Config;

// Maps a wave index to the (p, q) coordinates of the subtile it computes.
template <Config cfg>
class WaveComputeIndex
{
    // Batch-unfold invariant: a q-wave's wave_q16*16 columns must hold whole images.
    //
    // Otherwise images_per_qwave() truncates and lds_col_base()/wave_image_base()
    // compute wrong per-wave bases. Gated to unfold configs (the standard tile never
    // takes the images_per_qwave() path).
    static_assert(cfg.unfold_n == 1 || (cfg.wave_q16 * 16) % cfg.w_unfold() == 0,
                  "batch-unfold requires a q-wave's wave_q16*16 columns to hold a "
                  "whole number of w_unfold()-wide images; otherwise "
                  "images_per_qwave() truncates and the per-wave column/image bases "
                  "are wrong");
    static_assert(cfg.unfold_n == 1 || cfg.images_per_qwave() >= 1,
                  "batch-unfold requires at least one packed image per q-wave "
                  "(wave_q16*16 >= w_unfold())");

public:
    // wave_idx is the wave's index within its wavegroup (0..3).
    __host__ __device__ WaveComputeIndex(int wave_idx) : wave_idx_(wave_idx) {}

    __host__ __device__ int q_idx() const { return wave_idx_ % cfg.waves_q; }

    __host__ __device__ int p_idx() const { return wave_idx_ / cfg.waves_q; }

    // Return the q-coordinate offset divided by 16.
    __host__ __device__ int q16() const { return q_idx() * cfg.wave_q16; }

    // Return the q-coordinate offset of the wave.
    __host__ __device__ int q() const { return q16() * 16; }

    // Return the p-coordinate offset of the wave.
    __host__ __device__ int p() const { return p_idx() * cfg.wave_p; }

    // LDS column base of this wave's input tile (1-element units).
    //
    // Standard tile: q(). Unfold: the q-wave owns images_per_qwave images of
    // w_per_image columns, so the base is q_idx * images_per_qwave * w_per_image
    // (the per-lane split is InputLoaderLds::lane_column()).
    __host__ __device__ int lds_col_base() const
    {
        if constexpr(cfg.unfold_n == 1)
            return q();
        else
            return q_idx() * cfg.images_per_qwave() * (cfg.w_unfold() + cfg.kw - 1);
    }

    // First packed image this q-wave owns (unfold only; writer adds n_local).
    __host__ __device__ int wave_image_base() const { return q_idx() * cfg.images_per_qwave(); }

private:
    int wave_idx_;
};

} // namespace hipconv::cdna4

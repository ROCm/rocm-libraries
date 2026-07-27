// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/features/gemm_embedding_similarity.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace origami::nn::features::gemm_embedding_similarity {
namespace {

constexpr float kEpsilon = 1e-8f;

inline float bucket_dimension(float x)
  {
      // Buckets: [0,16,32,64,128,192,256,512,1024,2048,4096,8192,inf]
      if(x <= 16.0f)
    return 0.0f;
      if(x <= 32.0f)
    return 1.0f;
      if(x <= 64.0f)
    return 2.0f;
      if(x <= 128.0f)
    return 3.0f;
      if(x <= 192.0f)
    return 4.0f;
      if(x <= 256.0f)
    return 5.0f;
      if(x <= 512.0f)
    return 6.0f;
      if(x <= 1024.0f)
    return 7.0f;
      if(x <= 2048.0f)
    return 8.0f;
      if(x <= 4096.0f)
    return 9.0f;
      if(x <= 8192.0f)
    return 10.0f;
      return 11.0f;
  }

inline float bucket_aspect_ratio(float aspect)
  {
      // Buckets: [0, 0.5, 1.5, 2.5, inf]
      if(aspect <= 0.5f)
    return 0.0f;
      if(aspect <= 1.5f)
    return 1.0f;
      if(aspect <= 2.5f)
    return 2.0f;
      return 3.0f;
  }

inline float compute_wastage(float m, float n, float tile)
  {
      float tiles_m = std::ceil(m / tile);
      float tiles_n = std::ceil(n / tile);
      float work    = tiles_m * tile * tiles_n * tile;
      return (work - m * n) / work;
  }

inline float best_fit_tile(float x)
  {
      float tiles[]   = {128.0f, 192.0f, 224.0f, 256.0f};
      float best      = tiles[0];
      float min_waste = std::ceil(x / tiles[0]) * tiles[0] - x;

      for(int i = 1; i < 4; i++)
      {
    float waste = std::ceil(x / tiles[i]) * tiles[i] - x;
    if(waste < min_waste)
    {
        min_waste = waste;
        best      = tiles[i];
    }
      }
      return best;
  }

}  // namespace

void build_features(const problem_t& problem,
                    const esrec::detail::HardwareConstants& hw,
                    bool is_nt,
                    std::vector<float>& features) {
  const bool transA = problem.a_transpose == transpose_t::T;
  const bool transB = problem.b_transpose == transpose_t::T;
  const float m = static_cast<float>(problem.size.m);
  const float n = static_cast<float>(problem.size.n);
  const float k = static_cast<float>(problem.size.k);
  const float batch_count = static_cast<float>(problem.batch);
  const bool is_NT = is_nt;
  const float lda = transA ? m : k;
  const float ldb = transB ? k : n;
  const float ldc = n;
  const float ldd = n;
  const float stride_a = transA ? lda * m : lda * k;
  const float stride_b = transB ? ldb * k : ldb * n;
  const float stride_c = ldc * n;
  const float stride_d = ldd * n;

  // Basic computations
  float flops      = 2.0f * m * n * k * batch_count;
  float bytes_moved          = (m * k + k * n + m * n) * hw.dtype_size;
  float arithmetic_intensity = flops / bytes_moved;
  float output_size          = m * n;

  // Memory and compute characteristics
  float balance_ai = hw.peak_flops / hw.mem_bw;
  float ai_vs_balance = arithmetic_intensity / balance_ai;
  float memory_peak = hw.mem_bw * arithmetic_intensity;
  float compute_peak = hw.peak_flops;
  float is_compute_bound = (memory_peak > compute_peak) ? 1.0f : 0.0f;
  float memory_headroom = memory_peak / compute_peak;
  float memory_headroom_clipped = std::min(std::max(memory_headroom, 0.0f), 2.0f);

  // Cache pressure
  float ws_l1_ratio = bytes_moved / hw.l1_size;
  float ws_l2_ratio = bytes_moved / hw.l2_size;
  float ws_l3_ratio = bytes_moved / hw.l3_size;

  float fits_in_l1 = (bytes_moved <= hw.l1_size) ? 1.0f : 0.0f;
  float fits_in_l2 = (bytes_moved <= hw.l2_size) ? 1.0f : 0.0f;
  float fits_in_l3 = (bytes_moved <= hw.l3_size) ? 1.0f : 0.0f;


  constexpr float SWEET_SPOT_LOWER = 0.5f;
  float in_l2_sweet_spot = (bytes_moved > SWEET_SPOT_LOWER * hw.l2_size && bytes_moved <= hw.l2_size) ? 1.0f : 0.0f;
  float in_l3_sweet_spot = (bytes_moved > SWEET_SPOT_LOWER * hw.l3_size && bytes_moved <= hw.l3_size) ? 1.0f : 0.0f;

  float fits_in_l3_not_l2 = (bytes_moved <= hw.l3_size && bytes_moved > hw.l2_size) ? 1.0f : 0.0f;

  // K-dimension pressure
  float k_underutilizes_wave = (k < hw.wave_size) ? 1.0f : 0.0f;
  float k_saturates_waves    = (k >= 4.0f * hw.wave_size) ? 1.0f : 0.0f;

  // Accumulator pressure
  float acc_bytes = hw.acc_size * output_size;

  // Wave alignment
  float m_wave_misalignment
      = std::fmod(m, hw.wave_size) / hw.wave_size;
  float n_wave_misalignment
      = std::fmod(n, hw.wave_size) / hw.wave_size;
  float wave_misalignment_total = m_wave_misalignment + n_wave_misalignment;
  float m_wave_aligned
      = (static_cast<int>(m) % static_cast<int>(hw.wave_size) == 0) ? 1.0f
                   : 0.0f;
  float n_wave_aligned
      = (static_cast<int>(n) % static_cast<int>(hw.wave_size) == 0) ? 1.0f
                   : 0.0f;
  float both_wave_aligned = m_wave_aligned * n_wave_aligned;

  // Stream-K hints
  float streamk_favorable = ((k > 1024.0f) && (output_size < 4096.0f)) ? 1.0f : 0.0f;

  // Reuse factors
  float low_reuse  = ((n < 64.0f) || (m < 64.0f)) ? 1.0f : 0.0f;
  float high_reuse = ((n >= 256.0f) && (m >= 256.0f)) ? 1.0f : 0.0f;

  // Tile preferences
  float prefer_small_tile = (ws_l1_ratio > 2.0f) ? 1.0f : 0.0f;

  // Aspect ratios
  float aspect_m_n     = m / (n + kEpsilon);
  float sqrt_aspect_nm = std::sqrt(n / (m + kEpsilon));

  // Occupancy
  float est_tiles = std::ceil(m / 256.0f) * std::ceil(n / 256.0f);
  float is_saturating
      = (est_tiles >= static_cast<float>(hw.n_cu)) ? 1.0f : 0.0f;
  float est_waves = est_tiles / static_cast<float>(hw.n_cu);

  // Tile counts
  float tiles_64x48 = std::ceil(m / 64.0f) * std::ceil(n / 48.0f);
  float tiles_64x96 = std::ceil(m / 64.0f) * std::ceil(n / 96.0f);
  float tiles_128 = std::ceil(m / 128.0f) * std::ceil(n / 128.0f);
  float tiles_192 = std::ceil(m / 192.0f) * std::ceil(n / 192.0f);
  float tiles_224 = std::ceil(m / 224.0f) * std::ceil(n / 224.0f);

  // Wastage
  float wastage_32  = compute_wastage(m, n, 32.0f);
  float wastage_64  = compute_wastage(m, n, 64.0f);
  float wastage_128 = compute_wastage(m, n, 128.0f);
  float wastage_192 = compute_wastage(m, n, 192.0f);
  float wastage_224 = compute_wastage(m, n, 224.0f);
  float wastage_256 = compute_wastage(m, n, 256.0f);

  // Best fit
  float best_fit_n = best_fit_tile(n);

  // Edge case features
  float is_tiny_m = (m <= 32.0f) ? 1.0f : 0.0f;
  float is_tiny_n = (n <= 32.0f) ? 1.0f : 0.0f;
  float is_small_m = ((m > 32.0f) && (m <= 128.0f)) ? 1.0f : 0.0f;
  float is_small_n = ((n > 32.0f) && (n <= 128.0f)) ? 1.0f : 0.0f;
  float is_gemv_n = (n == 1.0f) ? 1.0f : 0.0f;
  float is_all_tiny = ((m <= 64.0f) && (n <= 64.0f) && (k <= 64.0f)) ? 1.0f : 0.0f;

  // K-dimension features
  float k_ultra_tiny    = (k <= 8.0f) ? 1.0f : 0.0f;
  float is_tiny_k       = (k <= 16.0f) ? 1.0f : 0.0f;
  float is_very_small_k = (k <= 64.0f) ? 1.0f : 0.0f;
  float is_small_k      = (k < 128.0f) ? 1.0f : 0.0f;
  float k_small_problem = ((k <= 128.0f) && (m < 4096.0f) && (n < 4096.0f)) ? 1.0f : 0.0f;
  float is_large_k = (k > 4096.0f) ? 1.0f : 0.0f;

  // General features
  float n_small_misaligned
      = ((n < 300.0f) && (static_cast<int>(n) % 16 != 0)) ? 1.0f : 0.0f;
  float k_small_misaligned
      = ((k < 300.0f) && (static_cast<int>(k) % 16 != 0)) ? 1.0f : 0.0f;
  float n_small_wastage_ratio = (n < 300.0f) ? (std::fmod(n, 16.0f) / 16.0f) : 0.0f;

  float extreme_aspect_ratio = ((n > 3.0f * m) || (m > 3.0f * n)) ? 1.0f : 0.0f;
  float n_vector = (n <= 2.0f) ? 1.0f : 0.0f;
  float very_extreme_aspect = ((m > 10.0f * n) || (n > 10.0f * m)) ? 1.0f : 0.0f;

  // Output size features
  float is_tiny_output       = (output_size < 1000.0f) ? 1.0f : 0.0f;
  float is_very_tiny_output  = (output_size < 100.0f) ? 1.0f : 0.0f;
  float is_ultra_tiny_output = (output_size < 50.0f) ? 1.0f : 0.0f;

  // Parallelization
  float insufficient_parallelism
      = (output_size < static_cast<float>(hw.n_cu)) ? 1.0f : 0.0f;
  float severe_underutilization
      = (output_size < static_cast<float>(hw.n_cu) / 2.0f) ? 1.0f : 0.0f;

  // K reuse
  float k_reuse_per_output = k / (output_size + 1.0f);

  // K memory
  float k_memory_bytes = k * hw.dtype_size * (m + n);
  float k_memory_vs_l3 = k_memory_bytes / hw.l3_size;

  // Dimension dominance
  float max_dim = std::max({m, n, k});
  float k_is_max_dim = (k == max_dim) ? 1.0f : 0.0f;
  float k_dominates_both = ((k > 10.0f * m) && (k > 10.0f * n)) ? 1.0f : 0.0f;

  // NT-specific features (if IS_NT is true)
  float k_dominates_n_10x = 0.0f, k_dominates_n_100x = 0.0f, k_dominates_n_1000x = 0.0f;
  float m_dominates_n_10x = 0.0f;
  float k_dominates_m_10x = 0.0f, k_dominates_m_1000x = 0.0f;
  float n_dominates_m_10x = 0.0f, n_dominates_m_100x = 0.0f;
  float n_dominates_k_10x = 0.0f, n_dominates_k_100x = 0.0f;
  float m_dominates_k_10x = 0.0f, m_dominates_k_100x = 0.0f;

  float min_dim           = std::min({m, n, k});
  float extreme_dimension_ratio_10x = 0.0f, extreme_dimension_ratio_100x = 0.0f;

  float m_ultra_tiny = 0.0f, n_ultra_tiny = 0.0f, k_ultra_tiny_v2 = 0.0f;
  float any_dim_ultra_tiny = 0.0f, multiple_dims_tiny = 0.0f;

  float large_output_small_k = 0.0f;

  float extreme_ratio_and_tiny_dim = 0.0f, k_dominates_and_small_output = 0.0f;

  float n_div_k_ratio = 0.0f, m_div_k_ratio = 0.0f;
  float n_div_k_very_small = 0.0f, m_div_k_very_small = 0.0f;
  float n_div_k_ultra_small = 0.0f;

  float likely_needs_small_tile = 0.0f;

  float work_elements = m * n * k;
  float is_micro_gemm = 0.0f, is_nano_gemm = 0.0f;

  float m_not_vec4_aligned = 0.0f, n_not_vec4_aligned = 0.0f, k_not_vec4_aligned = 0.0f;
  float m_not_vec8_aligned = 0.0f, n_not_vec8_aligned = 0.0f;

  float pathological_case_type1 = 0.0f;
  float problem_severity_count = 0.0f, multiple_problems = 0.0f;

  if (is_NT) {

      // Aspect ratio patterns
      k_dominates_n_10x   = (k / (n + 1.0f) > 10.0f) ? 1.0f : 0.0f;
      k_dominates_n_100x  = (k / (n + 1.0f) > 100.0f) ? 1.0f : 0.0f;
      k_dominates_n_1000x = (k / (n + 1.0f) > 1000.0f) ? 1.0f : 0.0f;

      m_dominates_n_10x = (m / (n + 1.0f) > 10.0f) ? 1.0f : 0.0f;

      k_dominates_m_10x = (k / (m + 1.0f) > 10.0f) ? 1.0f : 0.0f;
      k_dominates_m_1000x = (k / (m + 1.0f) > 1000.0f) ? 1.0f : 0.0f;

      n_dominates_m_10x  = (n / (m + 1.0f) > 10.0f) ? 1.0f : 0.0f;
      n_dominates_m_100x = (n / (m + 1.0f) > 100.0f) ? 1.0f : 0.0f;

      n_dominates_k_10x  = (n / (k + 1.0f) > 10.0f) ? 1.0f : 0.0f;
      n_dominates_k_100x = (n / (k + 1.0f) > 100.0f) ? 1.0f : 0.0f;

      m_dominates_k_10x  = (m / (k + 1.0f) > 10.0f) ? 1.0f : 0.0f;
      m_dominates_k_100x = (m / (k + 1.0f) > 100.0f) ? 1.0f : 0.0f;

      float extreme_dim_ratio      = max_dim / (min_dim + 1.0f);
      extreme_dimension_ratio_10x  = (extreme_dim_ratio > 10.0f) ? 1.0f : 0.0f;
      extreme_dimension_ratio_100x = (extreme_dim_ratio > 100.0f) ? 1.0f : 0.0f;

      // Ultra tiny dimensions
      m_ultra_tiny    = (m <= 10.0f) ? 1.0f : 0.0f;
      n_ultra_tiny    = (n <= 10.0f) ? 1.0f : 0.0f;
      k_ultra_tiny_v2 = (k <= 10.0f) ? 1.0f : 0.0f;

      any_dim_ultra_tiny = ((m <= 10.0f) || (n <= 10.0f) || (k <= 10.0f)) ? 1.0f : 0.0f;

      int tiny_count = (m <= 32.0f ? 1 : 0) + (n <= 32.0f ? 1 : 0) + (k <= 32.0f ? 1 : 0);
      multiple_dims_tiny = (tiny_count >= 2) ? 1.0f : 0.0f;

      // Problematic configurations
      large_output_small_k = ((output_size > 1000000.0f) && (k < 256.0f)) ? 1.0f : 0.0f;

      // Combined problematic patterns
      extreme_ratio_and_tiny_dim = extreme_dimension_ratio_10x * any_dim_ultra_tiny;
      k_dominates_and_small_output
          = k_dominates_both * ((output_size < 10000.0f) ? 1.0f : 0.0f);

      // Specific ratios
      n_div_k_ratio = n / (k + 1.0f);
      m_div_k_ratio = m / (k + 1.0f);

      n_div_k_very_small = (n_div_k_ratio < 0.1f) ? 1.0f : 0.0f;
      m_div_k_very_small = (m_div_k_ratio < 0.1f) ? 1.0f : 0.0f;
      n_div_k_ultra_small = (n_div_k_ratio < 0.01f) ? 1.0f : 0.0f;

      // Tile needs
      float est_tile_m_16     = (m < 128.0f) ? 1.0f : 0.0f;
      float est_tile_n_16     = (n < 128.0f) ? 1.0f : 0.0f;
      likely_needs_small_tile = (est_tile_m_16 + est_tile_n_16 >= 1.0f) ? 1.0f : 0.0f;
  

      // Work elements
      is_micro_gemm = (work_elements < 100000.0f) ? 1.0f : 0.0f;
      is_nano_gemm  = (work_elements < 10000.0f) ? 1.0f : 0.0f;

      // Vectorization alignment
      m_not_vec4_aligned = (static_cast<int>(m) % 4 != 0) ? 1.0f : 0.0f;
      n_not_vec4_aligned = (static_cast<int>(n) % 4 != 0) ? 1.0f : 0.0f;
      k_not_vec4_aligned = (static_cast<int>(k) % 4 != 0) ? 1.0f : 0.0f;

      m_not_vec8_aligned = (static_cast<int>(m) % 8 != 0) ? 1.0f : 0.0f;
      n_not_vec8_aligned = (static_cast<int>(n) % 8 != 0) ? 1.0f : 0.0f;

      // Pathological cases
      pathological_case_type1 = ((n / (m + 1.0f) > 50.0f) && (k < 100.0f)) ? 1.0f : 0.0f;
 
      // Problem severity
      problem_severity_count = extreme_dimension_ratio_10x + any_dim_ultra_tiny + k_dominates_both;
      multiple_problems = (problem_severity_count >= 2.0f) ? 1.0f : 0.0f;
  }

  // Build feature vector (matching Python order exactly)
  features.reserve(400);

  // Log-transformed inputs (order matching Python after preprocessing)
  features.push_back(std::log1p(m)); 
  features.push_back(std::log1p(n));
  features.push_back(std::log1p(k));
  features.push_back(std::log1p(lda));
  features.push_back(std::log1p(stride_a));
  features.push_back(std::log1p(ldb));
  features.push_back(std::log1p(stride_b));
  features.push_back(std::log1p(ldc));
  features.push_back(std::log1p(stride_c));
  features.push_back(std::log1p(ldd));
  features.push_back(std::log1p(stride_d));
  features.push_back(std::log1p(batch_count));

  // Computed features
  features.push_back(std::log1p(flops)); // log_flops
  features.push_back(std::log1p(bytes_moved)); // log_bytes
  features.push_back(arithmetic_intensity);
  features.push_back(std::log1p(arithmetic_intensity)); // log_ai

  // Roofline model
  features.push_back(is_compute_bound); 
  features.push_back(ai_vs_balance);
  features.push_back(std::log1p(ai_vs_balance));  // log_ai_vs_balance
  features.push_back(memory_headroom_clipped);

  // Cache pressure
  features.push_back(std::log1p(ws_l1_ratio)); // log_ws_l1_ratio
  features.push_back(fits_in_l1);
  features.push_back(std::log1p(ws_l2_ratio)); // log_ws_l2_ratio
  features.push_back(fits_in_l2);
  features.push_back(std::log1p(ws_l3_ratio)); // log_ws_l3_ratio
  features.push_back(fits_in_l3);
  features.push_back(in_l2_sweet_spot);
  features.push_back(in_l3_sweet_spot);
  features.push_back(fits_in_l3_not_l2);

  // K-dimension pressure
  features.push_back(std::log1p((k * hw.dtype_size)
            / hw.l1_size)); // log_k_l1_pressure
  features.push_back(std::log1p(k / hw.wave_size)); // log_k_parallelism
  features.push_back(k_underutilizes_wave);
  features.push_back(k_saturates_waves);

  // Bandwidth pressure
  features.push_back(
      std::log1p(bytes_moved / hw.mem_bw)); // log_bandwidth_pressure

  // Accumulator pressure
  features.push_back(std::log1p(acc_bytes)); // log_acc_bytes
  features.push_back(std::log1p(acc_bytes / hw.l2_size)); // log_acc_pressure
  features.push_back(
      std::log1p(acc_bytes / hw.l3_size)); // log_acc_pressure_l3

  // Wave alignment
  features.push_back(m_wave_misalignment);
  features.push_back(n_wave_misalignment);
  features.push_back(wave_misalignment_total);
  features.push_back(m_wave_aligned);
  features.push_back(n_wave_aligned);
  features.push_back(both_wave_aligned);

  // Stream-K hints
  features.push_back(std::log1p(k / (output_size + kEpsilon))); // log_k_vs_mn
  features.push_back(std::log1p(k / (m + n + kEpsilon))); // log_streamk_imbalance
  features.push_back(streamk_favorable);

  // Reuse factors
  features.push_back(low_reuse);
  features.push_back(high_reuse);

  // Tile preferences
  features.push_back(prefer_small_tile);

  // Problem size buckets (categorical)
  features.push_back(bucket_dimension(m)); // m_bucket
  features.push_back(bucket_dimension(n)); // n_bucket
  features.push_back(bucket_dimension(k)); // k_bucket

  // Aspect ratios
  features.push_back(sqrt_aspect_nm);
  features.push_back(std::log1p(n / (m + kEpsilon))); // log_n_to_m_ratio
  features.push_back(std::log1p(aspect_m_n)); // log_aspect_m_n
  features.push_back(std::log1p(m / (k + kEpsilon))); // log_aspect_m_k
  features.push_back(std::log1p(n / (k + kEpsilon))); // log_aspect_n_k
  features.push_back(bucket_aspect_ratio(aspect_m_n)); // shape_category

  // Memory access patterns
  features.push_back(std::max(1.0f, std::log1p(ldc / (n + kEpsilon)))); // ldc_efficiency

  // Tile alignment (M)
  features.push_back(static_cast<float>(static_cast<int>(m) % 128 == 0));
  features.push_back(static_cast<float>(static_cast<int>(m) % 160 == 0));
  features.push_back(static_cast<float>(static_cast<int>(m) % 192 == 0));
  features.push_back(static_cast<float>(static_cast<int>(m) % 224 == 0));
  features.push_back(static_cast<float>(static_cast<int>(m) % 256 == 0));

  // Tile alignment (N)
  features.push_back(static_cast<float>(static_cast<int>(n) % 128 == 0));
  features.push_back(static_cast<float>(static_cast<int>(n) % 160 == 0));
  features.push_back(static_cast<float>(static_cast<int>(n) % 192 == 0));
  features.push_back(static_cast<float>(static_cast<int>(n) % 224 == 0));
  features.push_back(static_cast<float>(static_cast<int>(n) % 256 == 0));

  // Tile alignment (K)
  features.push_back(static_cast<float>(static_cast<int>(k) % 32 == 0));
  features.push_back(static_cast<float>(static_cast<int>(k) % 128 == 0));

  // Size ratios
  features.push_back(std::log1p(n / 128.0f)); // n_div_tile128
  features.push_back(std::log1p(n / 256.0f)); // n_div_tile256

  // Problem scale
  features.push_back((m >= 8192.0f || n >= 8192.0f || k >= 8192.0f) ? 1.0f
                  : 0.0f); // is_large

  // Shape flags
  features.push_back((m > n) ? 1.0f : 0.0f);  // is_tall
  features.push_back((n > m) ? 1.0f : 0.0f);  // is_wide
  features.push_back(((m > 4.0f * n) && (m > 4.0f * k)) ? 1.0f : 0.0f);  // is_tall_skinny
  features.push_back(((n > 4.0f * m) && (n > 4.0f * k)) ? 1.0f : 0.0f);  // is_short_wide
  features.push_back(((k > 4.0f * m) && (k > 4.0f * n)) ? 1.0f : 0.0f);  // is_deep_k

  // K-dimension features
  features.push_back(k_ultra_tiny);
  features.push_back(is_tiny_k);
  features.push_back(is_very_small_k);
  features.push_back(is_small_k);
  features.push_back(k_small_problem);
  features.push_back(is_large_k);

  features.push_back(std::log1p(k / 32.0f)); // k_div_32
  features.push_back(std::log1p(k / 64.0f)); // k_div_64

  // Occupancy proxy
  features.push_back(std::log1p(est_tiles)); // log_est_tiles
  features.push_back(is_saturating);
  features.push_back(std::log1p(est_waves)); // log_est_waves

  // Modulo features
  features.push_back(static_cast<float>(static_cast<int>(m) % 64 == 0)); // m_mod_64
  features.push_back(static_cast<float>(static_cast<int>(n) % 64 == 0)); // n_mod_64
  features.push_back(static_cast<float>(static_cast<int>(k) % 64 == 0)); // k_mod_64

  // Tile counts (log-transformed for specific sizes)
  features.push_back(std::log1p(tiles_64x48)); // log_tiles_64x48
  features.push_back(std::log1p(tiles_64x96)); // log_tiles_64x96
  features.push_back(std::log1p(tiles_128)); // log_tiles_128x128
  features.push_back(std::log1p(tiles_192)); // log_tiles_192x192
  features.push_back(std::log1p(tiles_224)); // log_tiles_224x224

  // Wastage features
  features.push_back(wastage_32);
  features.push_back(wastage_64);
  features.push_back(wastage_128);
  features.push_back(wastage_192);
  features.push_back(wastage_224);
  features.push_back(wastage_256);

  // Best fit
  features.push_back(best_fit_n);

  // Underfill flags
  features.push_back((m < 256.0f) ? 1.0f : 0.0f); // m_underfills_256
  features.push_back((n < 256.0f) ? 1.0f : 0.0f); // n_underfills_256
  features.push_back((m < 192.0f) ? 1.0f : 0.0f); // m_underfills_192
  features.push_back((n < 192.0f) ? 1.0f : 0.0f); // n_underfills_192

  // Partial tiles (M & N)
  features.push_back(std::fmod(m, 32.0f) / 32.0f);  // m_partial_32
  features.push_back(std::fmod(n, 32.0f) / 32.0f);  // n_partial_32
  features.push_back(std::fmod(m, 64.0f) / 64.0f);  // m_partial_64
  features.push_back(std::fmod(n, 64.0f) / 64.0f);  // n_partial_64

  features.push_back(std::fmod(m, 128.0f) / 128.0f);  // m_partial_128
  features.push_back(std::fmod(n, 128.0f) / 128.0f);  // n_partial_128
  features.push_back(std::fmod(m, 160.0f) / 160.0f);  // m_partial_160
  features.push_back(std::fmod(n, 160.0f) / 160.0f);  // n_partial_160
  features.push_back(std::fmod(m, 192.0f) / 192.0f);  // m_partial_192
  features.push_back(std::fmod(n, 192.0f) / 192.0f);  // n_partial_192
  features.push_back(std::fmod(m, 224.0f) / 224.0f);  // m_partial_224
  features.push_back(std::fmod(n, 224.0f) / 224.0f);  // n_partial_224
  features.push_back(std::fmod(m, 256.0f) / 256.0f);  // m_partial_256
  features.push_back(std::fmod(n, 256.0f) / 256.0f);  // n_partial_256
       
  // Wastage comparisons
  features.push_back(wastage_256 - wastage_224);
  features.push_back(wastage_256 - wastage_192); // wastage_256_vs_192
  features.push_back(wastage_256 - wastage_128);

  // Raw remainders
  features.push_back(std::fmod(m, 224.0f));  // m_mod_224
  features.push_back(std::fmod(n, 224.0f));  // n_mod_224
  features.push_back(std::fmod(n, 256.0f));  // n_mod_256

  // Edge case features
  features.push_back(is_tiny_m);
  features.push_back(is_tiny_n);
  features.push_back(is_small_m);
  features.push_back(is_small_n);
  features.push_back(is_gemv_n);
  features.push_back(is_all_tiny);

  // General features
  features.push_back(n_small_misaligned);
  features.push_back(k_small_misaligned);
  features.push_back(n_small_wastage_ratio);
  features.push_back(extreme_aspect_ratio);
  features.push_back(n_vector);
  features.push_back(very_extreme_aspect);

  // NT-specific features
  if (is_NT) {
      // Output size features
      features.push_back(std::log1p(output_size)); // log_output_size
      features.push_back(is_tiny_output);
      features.push_back(is_very_tiny_output);
      features.push_back(is_ultra_tiny_output);

        
      features.push_back(std::log1p(k / (m + 1.0f)));  // log_k_vs_m
      features.push_back(std::log1p(k / (n + 1.0f)));  // log_k_vs_n

      // Parallelization
      features.push_back(std::log1p(
          output_size / static_cast<float>(hw.n_cu))); // log_output_vs_cu
      features.push_back(insufficient_parallelism);
      features.push_back(severe_underutilization);
      features.push_back(std::log1p(k_reuse_per_output));  // log_k_reuse

      // K memory
      features.push_back(std::log1p(k_memory_bytes));  // log_k_memory
      features.push_back(std::log1p(k_memory_vs_l3)); 

      // Workload distribution
      features.push_back(std::log1p(k * 2.0f));  // log_work_per_output

      // Dimension dominance
      features.push_back(k_is_max_dim);
      features.push_back(k_dominates_both);
      features.push_back(k_dominates_n_10x);
      features.push_back(k_dominates_n_100x);
      features.push_back(k_dominates_n_1000x);
      features.push_back(m_dominates_n_10x);
      features.push_back(k_dominates_m_10x);
      features.push_back(n_dominates_m_10x);
      features.push_back(n_dominates_m_100x);
      features.push_back(n_dominates_k_10x);
      features.push_back(n_dominates_k_100x);
      features.push_back(m_dominates_k_10x); 
      features.push_back(m_dominates_k_100x);

      // Severity
      features.push_back(std::log1p(std::max({
          m / (n + 1.0f), n / (m + 1.0f),
          k / (m + 1.0f), m / (k + 1.0f),
          k / (n + 1.0f), n / (k + 1.0f)
      })));   // log_max_aspect_ratio

      features.push_back(extreme_dimension_ratio_10x);
      features.push_back(extreme_dimension_ratio_100x);

      // Ultra tiny
      features.push_back(m_ultra_tiny);
      features.push_back(n_ultra_tiny);
      features.push_back(k_ultra_tiny_v2);
      features.push_back(any_dim_ultra_tiny);
      features.push_back(multiple_dims_tiny);

      // Problematic configs
      features.push_back(large_output_small_k);

      // Combined patterns
      features.push_back(extreme_ratio_and_tiny_dim);
      features.push_back(k_dominates_and_small_output);

      // Specific ratios
      features.push_back(n_div_k_very_small);
      features.push_back(m_div_k_very_small);
      features.push_back(n_div_k_ultra_small);

      // Tile needs
      features.push_back(likely_needs_small_tile);

      // Work elements
      features.push_back(is_micro_gemm);
      features.push_back(is_nano_gemm);

      // Vectorization
      features.push_back(m_not_vec4_aligned);
      features.push_back(n_not_vec4_aligned);
      features.push_back(k_not_vec4_aligned);
      features.push_back(m_not_vec8_aligned);
      features.push_back(n_not_vec8_aligned);

      // Pathological cases
      features.push_back(pathological_case_type1);
      features.push_back(problem_severity_count);
      features.push_back(multiple_problems);
  }
}

bool build_query(const problem_t& problem,
                 const esrec::detail::HardwareConstants& hw,
                 bool is_nt,
                 float* out,
                 std::size_t out_dim) {
  std::vector<float> features;
  features.reserve(400);
  build_features(problem, hw, is_nt, features);
  if (features.size() != out_dim) {
    return false;
  }
  std::copy(features.begin(), features.end(), out);
  return true;
}

}  // namespace origami::nn::features::gemm_embedding_similarity

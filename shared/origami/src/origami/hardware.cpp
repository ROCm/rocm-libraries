// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/hardware.hpp"
#include "origami/types.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace origami {

hardware_t::hardware_t(architecture_t arch,
                       size_t N_CU,
                       size_t lds_capacity,
                       size_t NUM_XCD,
                       double mem1_perf_ratio,
                       double mem2_perf_ratio,
                       double mem3_perf_ratio,
                       size_t L2_capacity,
                       double compute_clock_ghz,
                       size_t parallel_mi_cu,
                       std::tuple<double, double, double> mem_bw_per_wg_coefficients)
    : arch(arch)
    , N_CU(N_CU)
    , lds_capacity(lds_capacity)
    , mem1_perf_ratio(mem1_perf_ratio)
    , mem2_perf_ratio(mem2_perf_ratio)
    , mem3_perf_ratio(mem3_perf_ratio)
    , L2_capacity(L2_capacity)
    , CU_per_L2(N_CU / NUM_XCD)
    , compute_clock_ghz(compute_clock_ghz)
    , parallel_mi_cu(parallel_mi_cu)
    , mem_bw_per_wg_coefficients(mem_bw_per_wg_coefficients)
    , NUM_XCD(NUM_XCD) {
  init_per_level_bw();
}

hardware_t::hardware_t(architecture_t arch,
                       size_t N_CU,
                       size_t lds_capacity,
                       const architecture_constants& constants,
                       size_t num_xcds,
                       size_t L2_capacity,
                       double compute_clock_ghz,
                       double memory_clock_ghz)
    : hardware_t(
          arch,
          N_CU,
          lds_capacity,
          num_xcds,
          bw_per_cycle(
              constants.mem1_perf_ratio,
              level_clocks(compute_clock_ghz, memory_clock_ghz, constants.mem_clock_ratio).mem1_ghz),
          bw_per_cycle(
              constants.mem2_perf_ratio,
              level_clocks(compute_clock_ghz, memory_clock_ghz, constants.mem_clock_ratio).mem2_ghz),
          bw_per_cycle(
              constants.mem3_perf_ratio,
              level_clocks(compute_clock_ghz, memory_clock_ghz, constants.mem_clock_ratio).mem3_ghz),
          L2_capacity,
          compute_clock_ghz,
          constants.parallel_mi_cu,
          constants.mem_bw_per_wg_coefficients) {}

hardware_t::hardware_t(hipDeviceProp_t properties)
    : hardware_t(get_hardware_for_properties(properties)) {}

hardware_t hardware_t::get_hardware_for_properties(hipDeviceProp_t properties,
                                                   size_t num_xcds_override) {
  auto arch_name = get_before_first_colon(properties.gcnArchName);
  auto arch_enum = arch_name_to_enum(arch_name);
  if (arch_enum == architecture_t::Count) {
    throw std::runtime_error(
        std::string("Attempting to retrieve hardware constants for unsupported architecture: ") +
        std::string(arch_name));
  }
  auto constants = get_arch_constants(arch_enum);
  auto num_xcds  = (num_xcds_override > 0) ? num_xcds_override : get_default_num_xcds(arch_enum);
  return hardware_t(arch_enum,
                    properties.multiProcessorCount,
                    properties.sharedMemPerBlock,
                    constants,
                    num_xcds,
                    properties.l2CacheSize,
                    properties.clockRate / 1.e6,
                    properties.memoryClockRate / 1.e6);
}

hardware_t hardware_t::get_hardware_for_device(int deviceId, hipDeviceProp_t const& prop) {
  size_t num_xcds = 0;
#if HIP_VERSION_MAJOR >= 7
  int queried_xccs = 0;
  if (hipDeviceGetAttribute(&queried_xccs, hipDeviceAttributeNumberOfXccs, deviceId) ==
          hipSuccess &&
      queried_xccs > 0) {
    num_xcds = static_cast<size_t>(queried_xccs);
  }
#endif

  return get_hardware_for_properties(prop, num_xcds);
}

hardware_t hardware_t::get_hardware_for_device(int deviceId) {
  hipDeviceProp_t prop;
  hipError_t e = hipGetDeviceProperties(&prop, deviceId);
  if (e) { throw std::runtime_error(hipGetErrorString(e)); }

  return get_hardware_for_device(deviceId, prop);
}

hardware_t hardware_t::get_hardware_for_arch(architecture_t arch,
                                             size_t N_CU,
                                             size_t lds_capacity,
                                             size_t L2_capacity,
                                             int compute_clock_khz) {
  if (arch == architecture_t::Count) {
    throw std::runtime_error("Attempting to create hardware for unsupported architecture");
  }

  auto constants = get_arch_constants(arch);

  return hardware_t(arch,
                    N_CU,
                    lds_capacity,
                    constants,
                    get_default_num_xcds(arch),
                    L2_capacity,
                    compute_clock_khz / 1.e6,
                    compute_clock_khz / 1.e6 / constants.mem_clock_ratio);
}

bool hardware_t::is_hardware_supported(hipDeviceProp_t properties) {
  auto arch_name = get_before_first_colon(properties.gcnArchName);
  auto arch_enum = arch_name_to_enum(arch_name);
  return arch_enum != architecture_t::Count;
}

size_t hardware_t::get_default_num_xcds(architecture_t arch) {
  // Do NOT add new architectures here — see declaration in hardware.hpp.
  switch (arch) {
    case architecture_t::gfx90a: return 1;
    case architecture_t::gfx942: return 8;
    case architecture_t::gfx950: return 8;
    case architecture_t::gfx1201: return 1;
    case architecture_t::gfx1100: return 1;
    case architecture_t::gfx1150: return 1;
    case architecture_t::gfx1151: return 1;
    case architecture_t::gfx1152: return 1;
    case architecture_t::gfx1153: return 1;
    // TODO: Update this with real value
    case architecture_t::gfx1250: return 1;
    default:
      throw std::runtime_error(
          std::string("No default XCD count for architecture ") +
          std::string(arch_enum_to_name(arch)) +
          ". Use get_hardware_for_device() with a live GPU to query at runtime.");
  }
}

void hardware_t::print() const {
  std::cout << "================== Hardware Configuration ==================\n";
  std::cout << "Number of CUs (N_CU)      : " << N_CU << "\n";
  std::cout << "LDS capacity              : " << lds_capacity << " bytes\n";
  std::cout << "mem1_perf_ratio           : " << mem1_perf_ratio << "\n";
  std::cout << "mem2_perf_ratio           : " << mem2_perf_ratio << "\n";
  std::cout << "mem3_perf_ratio           : " << mem3_perf_ratio << "\n";
  std::cout << "L2 Cache capacity         : " << L2_capacity << " bytes\n";
  std::cout << "CUs per L2 domain         : " << CU_per_L2 << "\n";
  std::cout << "Compute clock (GHz)       : " << compute_clock_ghz << "\n";
  std::cout << "Parallel MI/CU            : " << parallel_mi_cu << "\n";
  std::cout << "Number of XCDs (NUM_XCD)  : " << NUM_XCD << "\n";
  std::cout << "mem_bw_per_wg_coefficients: " << std::get<0>(mem_bw_per_wg_coefficients) << ", "
            << std::get<1>(mem_bw_per_wg_coefficients) << ", "
            << std::get<2>(mem_bw_per_wg_coefficients) << "\n\n";

  std::cout << "------------------ Instruction Map -------------------------\n";
  // Loop over the instruction_map and print each entry
  for (const auto& kv : INSTRUCTION_MAP.at(arch)) {
    const auto& key  = kv.first;
    const auto& L_MI = kv.second;

    std::cout << "Instruction: MI_M=" << key.MI_M << ", MI_N=" << key.MI_N << ", MI_K=" << key.MI_K
              << ", mi_input_type=" << datatype_to_string(key.mi_input_type) << " bytes\n"
              << "  -> Latency (L_MI): " << L_MI << "\n";
  }
  std::cout << "===========================================================\n";
}

size_t hardware_t::get_mi_latency(size_t MI_M,
                                  size_t MI_N,
                                  size_t MI_K,
                                  data_type_t mi_input_type) const {
  const auto& instruction_map = INSTRUCTION_MAP.at(arch);
  auto key                    = matrix_instruction(MI_M, MI_N, MI_K, mi_input_type);

  auto it = instruction_map.find(key);
  if (it != instruction_map.end()) {
    return it->second / parallel_mi_cu;
  } else {
    if (origami::runtime_options::get().debug_enabled)
      std::cerr << "Warning: Latency not found for MI_M=" << MI_M << ", MI_N=" << MI_N
                << ", MI_K=" << MI_K << ", mi_input_type=" << datatype_to_string(mi_input_type)
                << ". Returning latency value of 32 (really slow).\n";
    return 32 / parallel_mi_cu;  // Default latency if instruction is not found
  }
}

bool hardware_t::has_MALL() const {
  switch (arch) {
    case architecture_t::gfx90a:
    case architecture_t::gfx942:
    case architecture_t::gfx950:
    case architecture_t::gfx1201:
    case architecture_t::gfx1100:
    case architecture_t::gfx1151: return true;
    case architecture_t::gfx1150:
    case architecture_t::gfx1152:
    case architecture_t::gfx1153:
    case architecture_t::gfx1250:
    case architecture_t::Count:
      // Count is not a valid architecture, this is to silence compiler warning
      return false;
  }
}

bool hardware_t::has_native_TF32() const {
  switch (arch) {
    case architecture_t::gfx942: return true;
    case architecture_t::gfx90a:
    case architecture_t::gfx950:
    case architecture_t::gfx1201:
    case architecture_t::gfx1100:
    case architecture_t::gfx1150:
    case architecture_t::gfx1151:
    case architecture_t::gfx1152:
    case architecture_t::gfx1153:
    case architecture_t::gfx1250:
    case architecture_t::Count:
      // Count is not a valid architecture, this is to silence compiler warning
      return false;
  }
}

std::string hardware_t::get_before_first_colon(const std::string& input) {
  size_t pos = input.find(':');
  if (pos != std::string::npos) { return input.substr(0, pos); }
  return input;  // Return the whole string if ':' is not found
}

std::vector<dim3_t> hardware_t::get_valid_matrix_instructions(data_type_t mi_input_type) const {
  std::vector<dim3_t> result;

  const auto& instruction_map = INSTRUCTION_MAP.at(arch);

  for (const auto& kv : instruction_map) {
    const matrix_instruction& mi = kv.first;
    if (mi.mi_input_type == mi_input_type) { result.push_back(dim3_t{mi.MI_M, mi.MI_N, mi.MI_K}); }
  }

  return result;
}

dim3_t hardware_t::get_recommended_matrix_instruction(data_type_t mi_input_type) const {
  const auto& instruction_map = INSTRUCTION_MAP.at(arch);

  dim3_t best_dim        = {0, 0, 0};
  double best_throughput = 0.0;

  for (const auto& kv : instruction_map) {
    const matrix_instruction& mi = kv.first;
    if (mi.mi_input_type == mi_input_type) {
      size_t latency = kv.second / parallel_mi_cu;
      if (latency == 0) latency = std::numeric_limits<size_t>::max();  // Avoid division by zero

      // Calculate throughput as M*N*K/latency
      double throughput =
          static_cast<double>(mi.MI_M * mi.MI_N * mi.MI_K) / static_cast<double>(latency);

      // Update if throughput is better, or if equal, prefer instruction where M=16 (tiebreaker)
      bool is_better = throughput > best_throughput;
      bool is_tie_with_m16 =
          (throughput == best_throughput) && (mi.MI_M == 16) && (best_dim.m != 16);

      if (is_better || is_tie_with_m16) {
        best_throughput = throughput;
        best_dim        = dim3_t{mi.MI_M, mi.MI_N, mi.MI_K};
      }
    }
  }

  return best_dim;
}

void hardware_t::init_per_level_bw() {
  // The per-VW absolute-bandwidth model is opt-in (ORIGAMI_GFX950_BW_MODEL=1) and only
  // applies to gfx950. By default every architecture, gfx950 included, uses the legacy
  // per-workgroup-coefficient bandwidth model below.
  if (arch == architecture_t::gfx950 && runtime_options::get().gfx950_bw_model_enabled) {
    uses_absolute_bw = true;
    cache_lines      = {64, 128, 128, 64};
    cache_line_bytes = cache_lines.l2;

    // Measured per-VW peak bandwidth (TB/s) from gfx950 microbenchmarks, monotonic in
    // vector width [Bytes2, Bytes4, Bytes8, Bytes16]. Each level's peaks are converted to
    // the model's working unit (bytes per clock cycle) using the clock domain that drives
    // that level: mem1 the compute clock, mem2 the fabric clock, mem3 the memory clock.
    constexpr size_t NUM_VW         = static_cast<size_t>(mem_vector_width_t::Count);
    constexpr double MEM1_R_PEAK[NUM_VW] = {2.601, 4.991, 8.360, 15.116};
    constexpr double MEM1_W_PEAK[NUM_VW] = {3.726, 6.080, 7.059, 9.191};
    constexpr double MEM2_R_PEAK[NUM_VW] = {5.139, 5.880, 6.310, 6.565};
    constexpr double MEM2_W_PEAK[NUM_VW] = {3.905, 5.129, 5.158, 5.275};
    constexpr double MEM3_R_PEAK[NUM_VW] = {5.106, 6.228, 6.369, 6.373};
    constexpr double MEM3_W_PEAK[NUM_VW] = {4.430, 5.109, 5.153, 5.153};

    const double mem_clock_ratio  = get_arch_constants(arch).mem_clock_ratio;
    const double memory_clock_ghz = compute_clock_ghz / mem_clock_ratio;
    // Same per-level clock mapping the constructor uses (single source of truth in
    // level_clocks). This path has no measured memory clock, so it derives one as
    // compute_clock / mem_clock_ratio; as a result the fabric clock collapses back to
    // the compute clock (mem1 and mem2 share it) while mem3 uses the memory clock.
    const auto clk = level_clocks(compute_clock_ghz, memory_clock_ghz, mem_clock_ratio);
    auto to_mem1   = [&](double t) { return bw_per_cycle(t, clk.mem1_ghz); };
    auto to_mem2   = [&](double t) { return bw_per_cycle(t, clk.mem2_ghz); };
    auto to_mem3   = [&](double t) { return bw_per_cycle(t, clk.mem3_ghz); };

    // Build a per-VW bandwidth array: every vector width reaches its measured peak at full
    // occupancy, following a shared per-level occupancy shape (a*CU^2 + b*CU, normalized to
    // 1.0 at full occupancy). mem1 is XCD-local, driven by the CUs sharing one cache domain
    // (CU_per_L2), so it reaches peak at CU_per_L2 active CUs; mem2 and mem3 are device-wide
    // and reach peak at N_CU. mem3 saturates (quadratic); mem1 and mem2 ramp ~linearly.
    auto build =
        [](double a_norm, double b_norm, const double (&peak)[NUM_VW], auto conv) -> bw_coef_array_t {
      bw_coef_array_t arr{};
      for (size_t i = 0; i < NUM_VW; ++i) {
        const double p = conv(peak[i]);  // peak in bytes per clock cycle
        arr[i]         = std::make_tuple(a_norm * p, b_norm * p, 0.0);
      }
      return arr;
    };
    const double mem1_b = 1.0 / static_cast<double>(CU_per_L2);  // ramps to peak at CU_per_L2
    const double dev_b  = 1.0 / static_cast<double>(N_CU);       // ramps to peak at N_CU
    // mem3 occupancy shape (quadratic), fit from microbenchmark and normalized to 1.0 at N_CU.
    const double mem3_r_a = -1.962e-5, mem3_r_b = 8.931e-3;
    const double mem3_w_a = -2.975e-5, mem3_w_b = 1.152e-2;

    l2_bw_read    = build(0.0, mem1_b, MEM1_R_PEAK, to_mem1);
    l2_bw_write   = build(0.0, mem1_b, MEM1_W_PEAK, to_mem1);
    mall_bw_read  = build(0.0, dev_b, MEM2_R_PEAK, to_mem2);
    mall_bw_write = build(0.0, dev_b, MEM2_W_PEAK, to_mem2);
    hbm_bw_read   = build(mem3_r_a, mem3_r_b, MEM3_R_PEAK, to_mem3);
    hbm_bw_write  = build(mem3_w_a, mem3_w_b, MEM3_W_PEAK, to_mem3);
  } else {
    cache_lines      = {128, 128, 128, 128};
    cache_line_bytes = cache_lines.l2;
    auto fill        = [&]() -> bw_coef_array_t {
      auto c = mem_bw_per_wg_coefficients;
      return {{c, c, c, c}};
    };
    l2_bw_read    = fill();
    l2_bw_write   = fill();
    mall_bw_read  = fill();
    mall_bw_write = fill();
    hbm_bw_read   = fill();
    hbm_bw_write  = fill();
  }
}

}  // namespace origami

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/hardware.hpp"
#include "origami/types.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace origami {

// Resolve the number of CUs to model against.
//   requested_num_cus: caller's CU budget. Signed so non-positive values are
//                      handled without a caller-side clamp: 0 means "use all
//                      CUs" and a negative (invalid) value is treated the same.
//                      A positive value caps the budget.
//   hardware_num_cus:  physical CU count (hardware_t::N_CU), the upper bound.
// Returns the requested budget when it is positive and below the physical
// count; otherwise the full physical count.
std::size_t resolve_num_cus(std::int64_t requested_num_cus, std::size_t hardware_num_cus) {
  if (requested_num_cus > 0
      && static_cast<std::size_t>(requested_num_cus) < hardware_num_cus) {
    return static_cast<std::size_t>(requested_num_cus);
  }
  return hardware_num_cus;
}

hardware_t::hardware_t(architecture_t arch,
                       size_t N_CU,
                       size_t lds_capacity,
                       size_t rf_capacity,
                       size_t NUM_XCD,
                       double mem1_perf_ratio,
                       double mem2_perf_ratio,
                       double mem3_perf_ratio,
                       size_t L2_capacity,
                       double compute_clock_ghz,
                       size_t parallel_mi_cu,
                       std::tuple<double, double, double> mem_bw_per_wg_coefficients,
                       std::optional<int> pci_chip_id)
    : arch(arch)
    , N_CU(N_CU)
    , lds_capacity(lds_capacity)
    , rf_capacity(rf_capacity)
    , mem1_perf_ratio(mem1_perf_ratio)
    , mem2_perf_ratio(mem2_perf_ratio)
    , mem3_perf_ratio(mem3_perf_ratio)
    , L2_capacity(L2_capacity)
    , CU_per_L2(N_CU / NUM_XCD)
    , compute_clock_ghz(compute_clock_ghz)
    , parallel_mi_cu(parallel_mi_cu)
    , mem_bw_per_wg_coefficients(mem_bw_per_wg_coefficients)
    , NUM_XCD(NUM_XCD)
    , pci_chip_id(pci_chip_id) {}

hardware_t::hardware_t(architecture_t arch,
                       size_t N_CU,
                       size_t lds_capacity,
                       size_t rf_capacity,
                       const architecture_constants& constants,
                       size_t num_xcds,
                       size_t L2_capacity,
                       double compute_clock_ghz,
                       double memory_clock_ghz,
                       std::optional<int> pci_chip_id)
   : hardware_t(
          arch,
          N_CU,
          lds_capacity,
          rf_capacity,
          num_xcds,
          1e9 * constants.mem1_perf_ratio / (compute_clock_ghz * 1e6),
          1e9 * constants.mem2_perf_ratio / (memory_clock_ghz * 1e6 * constants.mem_clock_ratio),
          1e9 * constants.mem3_perf_ratio / (memory_clock_ghz * 1e6),
          L2_capacity,
          compute_clock_ghz,
          constants.parallel_mi_cu,
          constants.mem_bw_per_wg_coefficients,
          pci_chip_id) {}

hardware_t::hardware_t(hipDeviceProp_t properties, std::optional<int> pci_chip_id)
    : hardware_t(get_hardware_for_properties(properties, 0, pci_chip_id)) {}

hardware_t::hardware_t(const hardware_t& other)
    : arch(other.arch)
    , N_CU(other.N_CU)
    , lds_capacity(other.lds_capacity)
    , rf_capacity(other.rf_capacity)
    , mem1_perf_ratio(other.mem1_perf_ratio)
    , mem2_perf_ratio(other.mem2_perf_ratio)
    , mem3_perf_ratio(other.mem3_perf_ratio)
    , L2_capacity(other.L2_capacity)
    , CU_per_L2(other.CU_per_L2)
    , compute_clock_ghz(other.compute_clock_ghz)
    , parallel_mi_cu(other.parallel_mi_cu)
    , mem_bw_per_wg_coefficients(other.mem_bw_per_wg_coefficients)
    , NUM_XCD(other.NUM_XCD)
    , pci_chip_id(other.pci_chip_id) {}

namespace {
// On RDNA, HIP runs in WGP (Work Group Processor) mode by default. In that mode CLR halves
// the agent's compute-unit count, so hipDeviceProp_t::multiProcessorCount reports the number
// of WGPs (2 CUs each) rather than physical CUs. Origami reasons in physical CUs, so scale
// the reported count back up on the RDNA architectures. CDNA archs run in CU mode (factor 1).
size_t cus_per_multiProcessorCount(hardware_t::architecture_t arch) {
  switch (arch) {
    case hardware_t::architecture_t::gfx1100:  // RDNA3
    case hardware_t::architecture_t::gfx1150:  // RDNA3.5 (Strix)
    case hardware_t::architecture_t::gfx1151:
    case hardware_t::architecture_t::gfx1152:
    case hardware_t::architecture_t::gfx1153:
    case hardware_t::architecture_t::gfx1200:  // RDNA4
    case hardware_t::architecture_t::gfx1201:  // RDNA4
      return 2;
    default:
      return 1;
  }
}

// ORIGAMI_RDNA_CU_MULT overrides the WGP->CU multiplier used to derive N_CU, for
// StreamK-grid / launch-budget experiments. Unset or <=0 uses the arch default
// (so stock behaviour is unchanged). Read once (env latches in a static).
size_t effective_cu_mult(hardware_t::architecture_t arch) {
  static const size_t override_mult = []() -> size_t {
    if (const char* e = std::getenv("ORIGAMI_RDNA_CU_MULT")) {
      long v = std::atol(e);
      if (v > 0) return static_cast<size_t>(v);
    }
    return 0;  // 0 => use arch default
  }();
  return override_mult ? override_mult : cus_per_multiProcessorCount(arch);
}
}  // namespace

std::size_t effective_num_cus(const problem_t& problem, const hardware_t& hardware) {
  const std::size_t base = resolve_num_cus(problem.num_cus, hardware.N_CU);
  // Opt-in size-dependent gfx1201 StreamK CU multiplier. Default off => stock.
  // Measured on the HHS-TN eval: small GEMMs win with x3 (fills the 32 WGP at
  // higher occupancy), large/compute GEMMs win with x1 (leaner grid, less
  // reduction/fixup overhead). Env latches once in function-local statics.
  static const bool enabled = []() {
    const char* e = std::getenv("ORIGAMI_RDNA_CU_SIZEDEP");
    return e && std::atol(e) != 0;
  }();
  if (!enabled) return base;
  if (hardware.arch != hardware_t::architecture_t::gfx1201) return base;
  // A genuine caller CU cap always wins (do not reshape a capped launch).
  if (problem.num_cus > 0
      && static_cast<std::size_t>(problem.num_cus) < hardware.N_CU) {
    return base;
  }
  static const double threshold = []() {
    const char* e = std::getenv("ORIGAMI_RDNA_CU_SMALL_FLOP");
    return e ? std::atof(e) : 3.16e9;  // 2*M*N*K boundary (from the eval sweep)
  }();
  static const std::size_t small_mult = []() -> std::size_t {
    const char* e = std::getenv("ORIGAMI_RDNA_CU_SMALL_MULT");
    return e ? static_cast<std::size_t>(std::atol(e)) : 3;
  }();
  static const std::size_t big_mult = []() -> std::size_t {
    const char* e = std::getenv("ORIGAMI_RDNA_CU_BIG_MULT");
    return e ? static_cast<std::size_t>(std::atol(e)) : 1;
  }();
  const std::size_t base_mult = effective_cu_mult(hardware.arch);
  const std::size_t wgp = base_mult ? hardware.N_CU / base_mult : hardware.N_CU;
  const double flops = 2.0 * static_cast<double>(problem.size.m)
                       * static_cast<double>(problem.size.n)
                       * static_cast<double>(problem.size.k)
                       * static_cast<double>(problem.batch ? problem.batch : 1);
  const std::size_t mult = (flops < threshold) ? small_mult : big_mult;
  return wgp * mult;
}

hardware_t hardware_t::get_hardware_for_properties(hipDeviceProp_t properties,
                                                   size_t num_xcds_override,
                                                   std::optional<int> pci_chip_id) {
  auto arch_name = get_before_first_colon(properties.gcnArchName);
  auto arch_enum = arch_name_to_enum(arch_name);
  if (arch_enum == architecture_t::Count) {
    throw std::runtime_error(
        std::string("Attempting to retrieve hardware constants for unsupported architecture: ") +
        std::string(arch_name));
  }
  auto constants = get_arch_constants(arch_enum, pci_chip_id);
  auto num_xcds   = (num_xcds_override > 0)
                      ? num_xcds_override
                      : get_default_num_xcds(arch_enum);
  return hardware_t(arch_enum,
                    properties.multiProcessorCount * effective_cu_mult(arch_enum),
                    properties.sharedMemPerBlock,
                    properties.regsPerBlock * 4,  // RF capacity from device (regsPerBlock is in 32-bit registers, convert to bytes)
                    constants,
                    num_xcds,
                    properties.l2CacheSize,
                    properties.clockRate / 1.e6,
                    properties.memoryClockRate / 1.e6,
                    pci_chip_id);
}

hardware_t hardware_t::get_hardware_for_device(int deviceId,
                                               hipDeviceProp_t const& prop,
                                               std::optional<int> pci_chip_id) {
  size_t num_xcds = 0;
#if HIP_VERSION_MAJOR >= 7
  int queried_xccs = 0;
  if (hipDeviceGetAttribute(&queried_xccs, hipDeviceAttributeNumberOfXccs, deviceId) == hipSuccess
      && queried_xccs > 0) {
    num_xcds = static_cast<size_t>(queried_xccs);
  }

  auto arch_name = get_before_first_colon(prop.gcnArchName);
  auto arch_enum = arch_name_to_enum(arch_name);
  if (arch_enum == architecture_t::gfx950 && !pci_chip_id.has_value()) {
      int queried_id = 0;
      if (hipDeviceGetAttribute(&queried_id, hipDeviceAttributePciChipId, deviceId) == hipSuccess)
        pci_chip_id = std::make_optional(queried_id);
  }
#endif

  return get_hardware_for_properties(prop, num_xcds, pci_chip_id);
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
                                             size_t rf_capacity,
                                             size_t L2_capacity,
                                             int compute_clock_khz,
                                             std::optional<int> pci_chip_id) {
  if (arch == architecture_t::Count) {
    throw std::runtime_error("Attempting to create hardware for unsupported architecture");
  }

  auto constants = get_arch_constants(arch, pci_chip_id);

  return hardware_t(arch,
                    N_CU,
                    lds_capacity,
                    rf_capacity,
                    constants,
                    get_default_num_xcds(arch),
                    L2_capacity,
                    compute_clock_khz / 1.e6,
                    compute_clock_khz / 1.e6 / constants.mem_clock_ratio,
                    pci_chip_id);
}

bool hardware_t::is_hardware_supported(hipDeviceProp_t properties) {
  auto arch_name = get_before_first_colon(properties.gcnArchName);
  auto arch_enum = arch_name_to_enum(arch_name);
  return arch_enum != architecture_t::Count;
}

size_t hardware_t::get_default_num_xcds(architecture_t arch) {
  // Do NOT add new architectures here — see declaration in hardware.hpp.
  switch (arch) {
    case architecture_t::gfx90a:  return 1;
    case architecture_t::gfx942:  return 8;
    case architecture_t::gfx950:  return 8;
    case architecture_t::gfx1200: return 1;
    case architecture_t::gfx1201: return 1;
    case architecture_t::gfx1100: return 1;
    case architecture_t::gfx1150: return 1;
    case architecture_t::gfx1151: return 1;
    case architecture_t::gfx1152: return 1;
    case architecture_t::gfx1153: return 1;
    case architecture_t::gfx1250: return 8;
    default:
      throw std::runtime_error(
          std::string("No default XCD count for architecture ") +
          std::string(arch_enum_to_name(arch)) +
          ". Use get_hardware_for_device() with a live GPU to query at runtime.");
  }
}

size_t hardware_t::get_default_cache_line_bytes(architecture_t /*arch*/) {
  // Per-arch L2 cache-line size, currently uniform 128 B across supported archs.
  return 128;
}

void hardware_t::print() const {
  std::cout << "================== Hardware Configuration ==================\n";
  std::cout << "Number of CUs (N_CU)      : " << N_CU << "\n";
  std::cout << "LDS capacity              : " << lds_capacity << " bytes\n";
  std::cout << "RF capacity               : " << rf_capacity << " bytes\n";
  std::cout << "mem1_perf_ratio           : " << mem1_perf_ratio << "\n";
  std::cout << "mem2_perf_ratio           : " << mem2_perf_ratio << "\n";
  std::cout << "mem3_perf_ratio           : " << mem3_perf_ratio << "\n";
  std::cout << "L2 Cache capacity         : " << L2_capacity << " bytes\n";
  std::cout << "CUs per L2 domain         : " << CU_per_L2 << "\n";
  std::cout << "Compute clock (GHz)       : " << compute_clock_ghz << "\n";
  std::cout << "Parallel MI/CU            : " << parallel_mi_cu << "\n";
  std::cout << "Number of XCDs (NUM_XCD)  : " << NUM_XCD << "\n";
  if (pci_chip_id.has_value()) {
    std::cout << "PCI chip ID               : 0x" << std::hex 
              << static_cast<unsigned>(*pci_chip_id) << std::dec 
              << " (" << *pci_chip_id << ")\n";
  } else {
    std::cout << "PCI chip ID               : (not set)\n";
  }
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
    case architecture_t::gfx1200:
    case architecture_t::gfx1201:
    case architecture_t::gfx1100:
    case architecture_t::gfx1151:
      return true;
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
    case architecture_t::gfx942:
      return true;
    case architecture_t::gfx90a:
    case architecture_t::gfx950:
    case architecture_t::gfx1200:
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

}  // namespace origami

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/hardware/hardware.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace origami {

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

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
    , pci_chip_id(pci_chip_id)
    , N_CU(N_CU)
    , NUM_XCD(NUM_XCD)
    , CU_per_L2(N_CU / NUM_XCD)
    , lds_capacity(lds_capacity)
    , rf_capacity(rf_capacity)
    , l1_capacity(get_l1_capacity(arch))
    , L2_capacity(L2_capacity)
    , mem1_perf_ratio(mem1_perf_ratio)
    , mem2_perf_ratio(mem2_perf_ratio)
    , mem3_perf_ratio(mem3_perf_ratio)
    , mem_bw_per_wg_coefficients(mem_bw_per_wg_coefficients)
    , compute_clock_ghz(compute_clock_ghz)
    , parallel_mi_cu(parallel_mi_cu) {}

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
          pci_chip_id) {
  // L2_capacity is one L2's capacity.  CDNA reports one XCD's L2.  On gfx1250 the XCDs of an
  // AID share an L2 and HIP reports a 4 MiB placeholder, so the reported size is split over
  // the L2 domains but at least the measured 96 MiB of one AID's L2 is used.
  if (is_cdna5()) {
    this->L2_capacity = std::max<size_t>(L2_capacity / num_l2_domains(), 96ull << 20);
  }
}

hardware_t::hardware_t(hipDeviceProp_t properties, std::optional<int> pci_chip_id)
    : hardware_t(get_hardware_for_properties(properties, 0, pci_chip_id)) {}

hardware_t::hardware_t(const hardware_t& other) = default;

// ---------------------------------------------------------------------------
// Static factory methods
// ---------------------------------------------------------------------------

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
  // Origami reasons in physical CUs, so scale the count HIP reports back up where it counts
  // WGPs.
  const size_t n_cu =
      properties.multiProcessorCount * traits_of(arch_enum).cus_per_reported_processor;
  auto constants = get_arch_constants(arch_enum, pci_chip_id, n_cu);
  auto num_xcds  = (num_xcds_override > 0) ? num_xcds_override : get_default_num_xcds(arch_enum);
  return hardware_t(
      arch_enum,
      n_cu,
      properties.sharedMemPerBlock,
      properties.regsPerBlock *
          4,  // RF capacity from device (regsPerBlock is in 32-bit registers, convert to bytes)
      constants,
      num_xcds,
      properties.l2CacheSize,
      properties.clockRate / 1.e6,
      properties.memoryClockRate / 1.e6,
      pci_chip_id);
}

hardware_t hardware_t::get_hardware_for_device(int deviceId) {
  hipDeviceProp_t prop;
  hipError_t e = hipGetDeviceProperties(&prop, deviceId);
  if (e) { throw std::runtime_error(hipGetErrorString(e)); }

  return get_hardware_for_device(deviceId, prop);
}

hardware_t hardware_t::get_hardware_for_device(int deviceId,
                                               hipDeviceProp_t const& prop,
                                               std::optional<int> pci_chip_id) {
  size_t num_xcds = 0;
#if HIP_VERSION_MAJOR >= 7
  int queried_xccs = 0;
  if (hipDeviceGetAttribute(&queried_xccs, hipDeviceAttributeNumberOfXccs, deviceId) ==
          hipSuccess &&
      queried_xccs > 0) {
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

  const variant_t variant = resolve_variant(arch, pci_chip_id, N_CU);
  auto constants          = constants_of(arch, variant, N_CU);
  // The 6-XCD gfx1250 part has 32 WGPs per XCD in every partition mode.
  const size_t num_xcds =
      (variant == variant_t::gfx1250_6xcd) ? N_CU / 32 : get_default_num_xcds(arch);

  return hardware_t(arch,
                    N_CU,
                    lds_capacity,
                    rf_capacity,
                    constants,
                    num_xcds,
                    L2_capacity,
                    compute_clock_khz / 1.e6,
                    compute_clock_khz / 1.e6 / constants.mem_clock_ratio,
                    pci_chip_id);
}

// ---------------------------------------------------------------------------
// Static query methods
// ---------------------------------------------------------------------------

size_t hardware_t::get_default_cache_line_bytes(architecture_t arch) {
  return traits_of(arch).cache_line_bytes;
}

size_t hardware_t::get_default_num_xcds(architecture_t arch) {
  const size_t num_xcds = traits_of(arch).default_num_xcds;
  if (num_xcds == 0) {
    throw std::runtime_error(
        std::string("No default XCD count for architecture ") +
        std::string(arch_enum_to_name(arch)) +
        ". Use get_hardware_for_device() with a live GPU to query at runtime.");
  }
  return num_xcds;
}

bool hardware_t::is_hardware_supported(hipDeviceProp_t properties) {
  auto arch_name = get_before_first_colon(properties.gcnArchName);
  auto arch_enum = arch_name_to_enum(arch_name);
  return arch_enum != architecture_t::Count;
}

// ---------------------------------------------------------------------------
// Instance query methods
// ---------------------------------------------------------------------------

bool hardware_t::has_native_TF32() const { return traits_of(arch).has_native_tf32; }

bool hardware_t::has_MALL() const { return traits_of(arch).has_mall; }

size_t hardware_t::get_mi_latency(size_t MI_M,
                                  size_t MI_N,
                                  size_t MI_K,
                                  data_type_t mi_input_type) const {
  const auto& latencies = matrix_instruction_latencies(family());
  auto key              = matrix_instruction(MI_M, MI_N, MI_K, mi_input_type);

  auto it = latencies.find(key);
  if (it != latencies.end()) {
    return it->second / parallel_mi_cu;
  } else {
    if (origami::runtime_options::get().debug_enabled)
      std::cerr << "Warning: Latency not found for MI_M=" << MI_M << ", MI_N=" << MI_N
                << ", MI_K=" << MI_K << ", mi_input_type=" << datatype_to_string(mi_input_type)
                << ". Returning latency value of 32 (really slow).\n";
    return 32 / parallel_mi_cu;  // Default latency if instruction is not found
  }
}

std::vector<dim3_t> hardware_t::get_valid_matrix_instructions(data_type_t mi_input_type) const {
  std::vector<dim3_t> result;

  for (const auto& kv : matrix_instruction_latencies(family())) {
    const matrix_instruction& mi = kv.first;
    if (mi.mi_input_type == mi_input_type) { result.push_back(dim3_t{mi.MI_M, mi.MI_N, mi.MI_K}); }
  }

  return result;
}

dim3_t hardware_t::get_recommended_matrix_instruction(data_type_t mi_input_type) const {
  dim3_t best_dim        = {0, 0, 0};
  double best_throughput = 0.0;

  for (const auto& kv : matrix_instruction_latencies(family())) {
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
    std::cout << "PCI chip ID               : 0x" << std::hex << static_cast<unsigned>(*pci_chip_id)
              << std::dec << " (" << *pci_chip_id << ")\n";
  } else {
    std::cout << "PCI chip ID               : (not set)\n";
  }
  std::cout << "mem_bw_per_wg_coefficients: " << std::get<0>(mem_bw_per_wg_coefficients) << ", "
            << std::get<1>(mem_bw_per_wg_coefficients) << ", "
            << std::get<2>(mem_bw_per_wg_coefficients) << "\n\n";

  std::cout << "------------------ Instruction Map -------------------------\n";
  for (const auto& kv : matrix_instruction_latencies(family())) {
    const auto& key  = kv.first;
    const auto& L_MI = kv.second;

    std::cout << "Instruction: MI_M=" << key.MI_M << ", MI_N=" << key.MI_N << ", MI_K=" << key.MI_K
              << ", mi_input_type=" << datatype_to_string(key.mi_input_type) << " bytes\n"
              << "  -> Latency (L_MI): " << L_MI << "\n";
  }
  std::cout << "===========================================================\n";
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

std::string hardware_t::get_before_first_colon(const std::string& input) {
  size_t pos = input.find(':');
  if (pos != std::string::npos) { return input.substr(0, pos); }
  return input;  // Return the whole string if ':' is not found
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

// A non-positive budget means "use all CUs", so the request only applies when it is positive
// and below the physical count.
std::size_t resolve_num_cus(std::int64_t requested_num_cus, std::size_t hardware_num_cus) {
  if (requested_num_cus > 0 && static_cast<std::size_t>(requested_num_cus) < hardware_num_cus) {
    return static_cast<std::size_t>(requested_num_cus);
  }
  return hardware_num_cus;
}

}  // namespace origami

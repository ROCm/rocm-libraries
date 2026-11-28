// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <hip/hip_runtime.h>

#include "origami/hardware/architecture.hpp"
#include "origami/hardware/constants.hpp"
#include "origami/hardware/matrix_instructions.hpp"
#include "origami/hardware/traits.hpp"
#include "origami/hardware/variant.hpp"
#include "origami/origami_export.h"
#include "origami/types.hpp"

namespace origami {

/**
 * @brief Represents hardware characteristics and capabilities of GPU architectures.
 *
 * What an architecture is comes from the tables of the hardware layer: its identity and
 * family (architecture.hpp), structural traits (traits.hpp), measured constants per chip
 * variant (variant.hpp, constants.hpp) and matrix instructions (matrix_instructions.hpp).
 * A hardware_t is one device: an architecture and variant with its sizes and clocks.
 */
class ORIGAMI_EXPORT hardware_t {
 public:
  // ---------------------------------------------------------------------------
  // Types and constants
  // ---------------------------------------------------------------------------

  /// Supported GPU architectures.
  using architecture_t = origami::architecture_t;

  /// Architecture-specific constants for memory and compute characteristics.
  using architecture_constants = origami::architecture_constants;

  /// MALL value for those architectures that do not support it.
  static constexpr double NO_MALL_AVAILABLE = origami::NO_MALL_AVAILABLE;

  // ---------------------------------------------------------------------------
  // Data members
  // ---------------------------------------------------------------------------

  architecture_t arch;               ///< GPU architecture type
  std::optional<int> pci_chip_id{};  ///< PCI chip ID the variant is resolved from (if set)

  size_t N_CU;       ///< Number of Compute Units
  size_t NUM_XCD;    ///< Number of XCDs (XGMI Complex Die)
  size_t CU_per_L2;  ///< Compute units per L2 cache domain (derived: N_CU / NUM_XCD)

  size_t lds_capacity;  ///< Capacity of Local Data Share (LDS) in bytes
  size_t rf_capacity;   ///< Capacity of Register File (RF) in bytes
  size_t l1_capacity;   ///< Capacity of L1 data cache per CU in bytes (shared across all SIMDs
                        ///< in a CU)
  size_t L2_capacity;   ///< Capacity of one L2 cache (one XCD's on CDNA, one AID's on gfx1250)
                        ///< in bytes

  double mem1_perf_ratio;  ///< L1/shared memory performance ratio (microbenchmarked)
  double mem2_perf_ratio;  ///< L2 cache performance ratio (microbenchmarked)
  double mem3_perf_ratio;  ///< MALL/HBM performance ratio (microbenchmarked)
  std::tuple<double, double, double>
      mem_bw_per_wg_coefficients;  ///< Memory bandwidth coefficients per workgroup
                                   ///< (microbenchmarked)

  double compute_clock_ghz;  ///< Compute clock frequency in GHz
  size_t parallel_mi_cu;     ///< Calibrated parallel matrix-instruction throughput per CU

  // ---------------------------------------------------------------------------
  // Constructors
  // ---------------------------------------------------------------------------

  /**
   * @brief Construct hardware_t with explicit parameters.
   *
   * The variant is resolved from @p arch, @p pci_chip_id and @p N_CU.
   *
   * @param arch GPU architecture type
   * @param N_CU Number of compute units
   * @param lds_capacity LDS capacity in bytes
   * @param rf_capacity RF capacity in bytes
   * @param NUM_XCD Number of XCDs
   * @param mem1_perf_ratio Memory level 1 performance ratio
   * @param mem2_perf_ratio Memory level 2 performance ratio
   * @param mem3_perf_ratio Memory level 3 performance ratio
   * @param L2_capacity L2 cache capacity in bytes
   * @param compute_clock_ghz Compute clock frequency in GHz
   * @param parallel_mi_cu Number of parallel matrix instructions per CU
   * @param mem_bw_per_wg_coefficients Memory bandwidth coefficients per workgroup
   * @param pci_chip_id Optional PCI chip ID stored on this object (default none)
   */
  hardware_t(architecture_t arch,
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
             std::optional<int> pci_chip_id = std::nullopt);

  /**
   * @brief Construct hardware_t using architecture constants and a clock frequency.
   *
   * Computes memory performance ratios from the provided architecture constants
   * and compute clock, then delegates to the full constructor.
   *
   * @param arch GPU architecture type
   * @param N_CU Number of compute units
   * @param lds_capacity LDS capacity in bytes
   * @param rf_capacity RF capacity in bytes
   * @param constants Architecture-specific constants
   * @param num_xcds Number of XCDs — provided separately from constants so that
   *                 it can come from a runtime query or a known-architecture table
   * @param L2_capacity L2 cache capacity in bytes
   * @param compute_clock_ghz Compute clock frequency in GHz
   * @param memory_clock_ghz Memory clock frequency in GHz
   * @param pci_chip_id Optional PCI chip ID stored on this object (default none)
   */
  hardware_t(architecture_t arch,
             size_t N_CU,
             size_t lds_capacity,
             size_t rf_capacity,
             const architecture_constants& constants,
             size_t num_xcds,
             size_t L2_capacity,
             double compute_clock_ghz,
             double memory_clock_ghz,
             std::optional<int> pci_chip_id = std::nullopt);

  /**
   * @brief Construct hardware_t from HIP device properties.
   *
   * Automatically determines architecture and extracts hardware parameters
   * from the provided HIP device properties structure. Delegates to
   * @ref get_hardware_for_properties with no XCD override.
   *
   * This constructor does **not** call HIP to discover PCI identifiers. If you need
   * the gfx950 variant selected from PCI (see @ref resolve_variant), pass @p pci_chip_id
   * explicitly (for example from @c hipDeviceGetAttribute with
   * @c hipDeviceAttributePciChipId). When @p pci_chip_id is @c std::nullopt, gfx950 is the
   * standard variant and the \c pci_chip_id field on this object is unset.
   *
   * @param properties HIP device properties structure
   * @param pci_chip_id Optional PCI chip id used to resolve the variant;
   *                    Origami does not query HIP when omitted.
   */
  hardware_t(hipDeviceProp_t properties, std::optional<int> pci_chip_id = std::nullopt);

  /**
   * @brief Copy constructor.
   *
   * @param other Another hardware_t instance to copy from
   */
  hardware_t(const hardware_t& other);

  // ---------------------------------------------------------------------------
  // Static factory methods (construct hardware_t from device / arch / props)
  // ---------------------------------------------------------------------------

  /**
   * @brief Create hardware_t instance from HIP device properties.
   *
   * @param properties HIP device properties structure
   * @param num_xcds_override If non-zero, use this XCD count instead of
   *                          the hardcoded default. Passed by
   *                          get_hardware_for_device() after a runtime query.
   * @param pci_chip_id Optional PCI chip ID (e.g. from @c hipDeviceAttributePciChipId) used
   *                    to resolve the variant (see @ref resolve_variant). Origami does not
   *                    query HIP for this.
   * @return hardware_t Configured hardware instance
   */
  static hardware_t get_hardware_for_properties(hipDeviceProp_t properties,
                                                size_t num_xcds_override       = 0,
                                                std::optional<int> pci_chip_id = std::nullopt);

  /**
   * @brief Create hardware_t instance for a specific HIP device.
   *
   * Queries the specified HIP device and creates a hardware_t instance
   * with the appropriate architecture and parameters.
   *
   * @param deviceId HIP device ID
   * @return hardware_t Configured hardware instance for the device
   */
  static hardware_t get_hardware_for_device(int deviceId);

  /**
   * @brief Create hardware_t instance for a specific HIP device using
   *        caller-provided properties.
   *
   * Same as @ref get_hardware_for_device(int) but uses the supplied
   * `hipDeviceProp_t` instead of re-querying via `hipGetDeviceProperties`.
   * Callers that have already adjusted fields on `prop` (for example,
   * overriding `multiProcessorCount` with
   * `hipDeviceAttributePhysicalMultiProcessorCount` on multi-XCC
   * architectures) should use this overload so those adjustments are
   * preserved. The runtime XCC query
   * (`hipDeviceAttributeNumberOfXccs` on HIP 7+) is still performed
   * against `deviceId`.
   * Accepts a caller-supplied PCI chip ID (e.g. from
   * `hipDeviceAttributePciChipId` upstream). When @p pci_chip_id is
   * `std::nullopt`, PCI chip ID is queried at runtime on HIP 7+ for gfx950.
   *
   * @param deviceId HIP device ID used to query the XCC count
   * @param prop     Caller-owned device properties to model from
   * @param pci_chip_id Optional PCI chip ID used to resolve the variant
   * @return hardware_t Configured hardware instance for the device
   */
  static hardware_t get_hardware_for_device(int deviceId,
                                            hipDeviceProp_t const& prop,
                                            std::optional<int> pci_chip_id = std::nullopt);

  /**
   * @brief Create hardware_t instance for a specific architecture with specified parameters.
   *
   * Creates a hardware instance using the specified architecture and hardware parameters.
   * Useful for analytical modeling when actual hardware is not available.
   *
   * @param arch Architecture enum value
   * @param N_CU Number of compute units
   * @param lds_capacity LDS capacity in bytes
   * @param rf_capacity rf capacity in bytes
   * @param L2_capacity L2 cache capacity in bytes
   * @param compute_clock_khz Compute clock in KHz
   * @param pci_chip_id Optional PCI chip ID used to resolve the variant; see
   *                    @ref get_hardware_for_properties.
   * @return hardware_t Configured hardware instance
   * @throws std::runtime_error if architecture is not supported
   */
  static hardware_t get_hardware_for_arch(architecture_t arch,
                                          size_t N_CU,
                                          size_t lds_capacity,
                                          size_t rf_capacity,
                                          size_t L2_capacity,
                                          int compute_clock_khz,
                                          std::optional<int> pci_chip_id = std::nullopt);

  // ---------------------------------------------------------------------------
  // Static query methods (arch-keyed lookups, no hardware_t instance needed)
  // ---------------------------------------------------------------------------

  /**
   * @brief Convert architecture name string to architecture_t enum.
   *
   * @param str Architecture name as string (e.g., "gfx90a", "gfx942")
   * @return architecture_t Corresponding enum value, or Count if not recognized
   */
  static constexpr architecture_t arch_name_to_enum(std::string_view str) noexcept {
    return architecture_from_name(str);
  }

  /**
   * @brief Convert architecture_t to string (e.g. for logging).
   *
   * @param a Architecture enum value
   * @return std::string_view Corresponding string value
   */
  static constexpr std::string_view arch_enum_to_name(architecture_t a) noexcept {
    return architecture_name(a);
  }

  /**
   * @brief Threads per wavefront TensileLite builds for an architecture by default: wave32
   *        wherever the ISA supports it (gfx10/11/12), wave64 on CDNA (gfx90a/942/950).
   *
   * A kernel built with another WavefrontSize carries it in tensile_params_t::wavefront_size.
   */
  static constexpr size_t get_wavefront_size(architecture_t arch) noexcept {
    return traits_of(arch).wavefront_size;
  }

  /**
   * @brief Lanes per SIMD: 16 on CDNA (gfx90a/942/950), 32 on RDNA and gfx1250.
   */
  static constexpr size_t get_simd_lanes(architecture_t arch) noexcept {
    return traits_of(arch).simd_lanes;
  }

  /**
   * @brief Return the number of SIMD units per Compute Unit for a given architecture.
   *
   * CDNA (gfx90a/942/950): 4 SIMD16 per CU.
   * RDNA3/4 (gfx11xx/gfx12xx): 2 SIMD32 per CU.
   * gfx1250: 4 SIMD32 per CU.
   */
  static constexpr size_t get_simds_per_cu(architecture_t arch) noexcept {
    return traits_of(arch).simds_per_cu;
  }

  /**
   * @brief Clocks one vector-ALU instruction of a wavefront occupies its SIMD.
   *
   * wave64 on SIMD16 (gfx950) takes 4; wave32 on SIMD32 (gfx1250) takes 1.
   */
  static constexpr size_t get_valu_cycles_per_instruction(architecture_t arch) noexcept {
    return get_wavefront_size(arch) / get_simd_lanes(arch);
  }

  /**
   * @brief Whether matrix results land in separate accumulation registers that the
   *        epilogue has to copy out (CDNA's AccVGPRs).
   *
   * gfx1250 keeps them in its unified VGPR file, so it has no such copy.
   */
  static constexpr bool get_has_accumulation_registers(architecture_t arch) noexcept {
    return traits_of(arch).has_accumulation_registers;
  }

  /**
   * @brief Return the LDS read bandwidth per CU in bytes per clock.
   *
   * 32 four-byte banks (128 B/clk) through CDNA3 and on RDNA; CDNA4 doubles the
   * bank count and read bandwidth to 256 B/clk, and gfx1250 keeps 64 banks.
   */
  static constexpr size_t get_lds_read_bytes_per_clk(architecture_t arch) noexcept {
    return traits_of(arch).lds_read_bytes_per_clk;
  }

  /**
   * @brief Return the L1 data cache capacity per CU for a given architecture.
   */
  static constexpr size_t get_l1_capacity(architecture_t arch) noexcept {
    return traits_of(arch).l1_capacity;
  }

  /**
   * @brief Get the default L2 cache-line size (in bytes) for an architecture.
   *
   * Returns the per-arch L2 cache-line size used for the StreamK per-queue
   * counter stride (currently uniform 128 B across supported archs).
   *
   * @param arch Architecture enum value
   * @return L2 cache-line size in bytes
   */
  static size_t get_default_cache_line_bytes(architecture_t arch);

  /**
   * @brief Get the default (hardcoded) XCD count for a known architecture.
   *
   * Legacy fallback for architectures that predate the runtime XCC
   * query (hipDeviceAttributeNumberOfXccs, HIP 7.0+). New architectures
   * should have no default in the traits table and rely solely on the
   * runtime query via get_hardware_for_device().
   *
   * @param arch Architecture enum value
   * @return Number of XCDs for the architecture
   * @throws std::runtime_error if the architecture has no hardcoded default
   */
  static size_t get_default_num_xcds(architecture_t arch);

  /**
   * @brief Whether a gfx1250 device with @p n_cu WGPs is the 6-XCD part
   *        (variant_t::gfx1250_6xcd; see @ref resolve_variant).
   */
  static constexpr bool gfx1250_wgps_select_6xcd(size_t n_cu) noexcept {
    return resolve_variant(architecture_t::gfx1250, std::nullopt, n_cu) == variant_t::gfx1250_6xcd;
  }

  /**
   * @brief Get architecture-specific constants for a given architecture.
   *
   * Returns the pre-configured constants (memory performance ratios, bandwidth
   * coefficients, etc.) of the variant that @ref resolve_variant derives from the
   * arguments. These values are determined through microbenchmarking.
   *
   * @param arch Architecture enum value
   * @param pci_chip_id For gfx950 only: optional PCI chip id that selects the variant.
   *                    Ignored for all other architectures.
   * @param n_cu For gfx1250 only: the device's WGP count, which selects the variant and
   *             scales the 6-XCD part's bandwidths; 0 selects the 8-XCD part.
   * @return architecture_constants Constants for the specified architecture
   */
  static constexpr architecture_constants get_arch_constants(
      architecture_t arch,
      std::optional<int> pci_chip_id = std::nullopt,
      size_t n_cu                    = 0) noexcept {
    return constants_of(arch, resolve_variant(arch, pci_chip_id, n_cu), n_cu);
  }

  /**
   * @brief gfx950-only architecture constants from optional PCI chip id.
   *
   * @param pci_chip_id Optional PCI chip id that selects the variant
   */
  static constexpr architecture_constants get_gfx950_arch_constants(
      std::optional<int> pci_chip_id = std::nullopt) noexcept {
    return get_arch_constants(architecture_t::gfx950, pci_chip_id);
  }

  /**
   * @brief gfx1250-only architecture constants, by the device's WGP count.
   *
   * @param n_cu The device's WGP count (0 = the 8-XCD part)
   */
  static constexpr architecture_constants get_gfx1250_arch_constants(size_t n_cu = 0) noexcept {
    return get_arch_constants(architecture_t::gfx1250, std::nullopt, n_cu);
  }

  /**
   * @brief Check if the hardware described by properties is supported.
   *
   * Determines whether the GPU architecture represented by the device
   * properties is supported by the analytical model.
   *
   * @param properties HIP device properties structure
   * @return true if the architecture is supported, false otherwise
   */
  static bool is_hardware_supported(hipDeviceProp_t properties);

  // ---------------------------------------------------------------------------
  // Instance query methods (require a constructed hardware_t)
  // ---------------------------------------------------------------------------

  /// Chip variant of this device (@see resolve_variant).
  constexpr variant_t variant() const noexcept { return resolve_variant(arch, pci_chip_id, N_CU); }

  /**
   * @brief Architecture family of this hardware instance.
   */
  constexpr family_t family() const noexcept { return family_of(arch); }

  /// gfx1250 (CDNA5): the XCDs of an AID share one L2, tensor DMA, no MALL.  Every other
  /// architecture keeps the CDNA model.
  bool is_cdna5() const { return family() == family_t::cdna5; }

  /**
   * @brief Default threads per wavefront for this hardware instance's architecture.
   */
  constexpr size_t wavefront_size() const noexcept { return get_wavefront_size(arch); }

  /**
   * @brief Lanes per SIMD for this hardware instance's architecture.
   */
  constexpr size_t simd_lanes() const noexcept { return get_simd_lanes(arch); }

  /**
   * @brief SIMDs whose workgroups a kernel's CUOccupancy counts.
   *
   * TensileLite's occupancy (and hipModuleOccupancyMaxActiveBlocksPerMultiprocessor) is per
   * CU on CDNA and per WGP on RDNA: 4 SIMDs and one LDS pool either way.  N_CU counts
   * physical CUs, and an RDNA WGP is two of them (simds_per_cu() is 2 there).
   */
  static constexpr size_t OCCUPANCY_SIMDS = 4;

  /**
   * @brief SIMD units per Compute Unit for this hardware instance's architecture.
   */
  constexpr size_t simds_per_cu() const noexcept { return get_simds_per_cu(arch); }

  /**
   * @brief Vector-ALU clocks per wavefront instruction for this hardware instance.
   */
  constexpr size_t valu_cycles_per_instruction() const noexcept {
    return get_valu_cycles_per_instruction(arch);
  }

  /**
   * @brief Whether this hardware instance copies matrix results out of accumulation registers.
   */
  constexpr bool has_accumulation_registers() const noexcept {
    return get_has_accumulation_registers(arch);
  }

  /**
   * @brief Check if hardware supports native TF32 matrix instructions.
   *
   * If TF32/XFloat32 is not natively supported, then emulation is required which has some overhead.
   *
   * @return true if the architecture has TF32 matrix instructions
   */
  bool has_native_TF32() const;

  /**
   * @brief Check if the architecture has MALL (Memory Attached Last Level).
   *
   * @return true if the architecture has MALL, false otherwise
   */
  bool has_MALL() const;

  /// Number of L2 domains: on gfx1250 one per AID of up to 4 XCDs, assuming a part with fewer
  /// XCDs spreads them evenly over as few AIDs as hold them; one per XCD on every other
  /// architecture.
  size_t num_l2_domains() const {
    const size_t per_l2 = traits_of(arch).max_xcds_per_l2;
    return std::max<size_t>((NUM_XCD + per_l2 - 1) / per_l2, 1);
  }

  /// XCDs that share one L2 (the last domain may hold fewer).
  size_t xcd_per_l2_domain() const {
    const size_t domains = num_l2_domains();
    return std::max<size_t>((NUM_XCD + domains - 1) / domains, 1);
  }

  /// Capacity of one L2 domain in bytes (L2_capacity is one L2's, on every architecture).
  size_t l2_domain_capacity() const { return L2_capacity; }

  /// XCD that runs the workgroup (or whole cluster) with dispatch id `id`.  Dispatch is
  /// round-robin over the XCDs, starting at XCD 0, or at the last XCD on gfx1250, so id i runs
  /// on XCD (i + NUM_XCD - 1) % NUM_XCD there.
  size_t dispatch_xcd(size_t id) const {
    const size_t n = NUM_XCD > 0 ? NUM_XCD : 1;
    return traits_of(arch).dispatch_starts_at_last_xcd ? (id + n - 1) % n : id % n;
  }

  /**
   * @brief Get matrix instruction latency for given instruction parameters.
   *
   * @param MI_M Matrix instruction M dimension
   * @param MI_N Matrix instruction N dimension
   * @param MI_K Matrix instruction K dimension
   * @param mi_input_type Input data type for the matrix instruction
   * @return size_t Instruction latency in cycles, or that of a slow 32-cycle instruction if
   *         not found
   */
  size_t get_mi_latency(size_t MI_M, size_t MI_N, size_t MI_K, data_type_t mi_input_type) const;

  /**
   * @brief Get valid matrix instruction dimensions for a given datatype.
   *
   * Returns a list of valid matrix instruction dimensions (M, N, K) for
   * the specified datatype on the current hardware architecture. Multiple
   * dimensions may be available for the same datatype.
   *
   * @param mi_input_type Input data type for the matrix instruction
   * @return std::vector<dim3_t> List of valid dimensions for the datatype
   */
  std::vector<dim3_t> get_valid_matrix_instructions(data_type_t mi_input_type) const;

  /**
   * @brief Get recommended matrix instruction dimensions for a given datatype.
   *
   * Returns the single best matrix instruction dimension for the specified datatype
   * based on throughput (M*N*K/latency). If multiple instructions are available,
   * returns the one with the highest throughput.
   *
   * @param mi_input_type Input data type for the matrix instruction
   * @return dim3_t Recommended dimension for the datatype. Returns {0,0,0} if not supported.
   */
  dim3_t get_recommended_matrix_instruction(data_type_t mi_input_type) const;

  /**
   * @brief Print hardware details to stdout.
   */
  void print() const;

 private:
  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  /**
   * @brief Extract substring before the first colon character.
   *
   * Helper function used for parsing architecture names from device
   * property strings (e.g., extracting "gfx90a" from "gfx90a:...").
   *
   * @param input Input string to parse
   * @return std::string Substring before the first colon, or entire string if no colon found
   */
  static std::string get_before_first_colon(const std::string& input);
};

/**
 * @brief Resolve the number of compute units to model against.
 *
 * Central helper for honoring a caller-supplied CU budget (problem.num_cus).
 * Returns @p requested_num_cus when it is a positive cap below the physical
 * count, otherwise the full hardware count. Used across solution selection and
 * mapping so that a budget of <= 0 preserves the "use all CUs" behaviour.
 *
 * @param requested_num_cus Requested CU budget (<= 0 = use all CUs). A signed
 *                          type so that invalid/negative inputs are absorbed
 *                          here rather than wrapping to a huge unsigned value.
 * @param hardware_num_cus Physical number of compute units (hardware_t::N_CU).
 * @return std::size_t Effective number of usable compute units.
 */
ORIGAMI_EXPORT std::size_t resolve_num_cus(std::int64_t requested_num_cus,
                                           std::size_t hardware_num_cus);

}  // namespace origami

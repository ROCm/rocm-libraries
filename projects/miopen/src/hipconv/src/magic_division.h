#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

// Magic-number division by a runtime-but-launch-time-constant divisor.
//
// Adapted from Composable Kernel's MagicDivision / MDiv:
//   https://github.com/ROCm/composable_kernel/blob/develop/include/ck/utility/magic_division.hpp
// (MIT licensed, Copyright (c) Advanced Micro Devices, Inc.)
//
// Precompute {multiplier, shift} on the host once per divisor and pass them
// in kernargs. The device divide is one v_mul_hi_u32 + one v_add + one shift;
// the divmod adds one v_mul_lo + one v_sub on top. Both multiplier and shift
// stay in SGPRs after kernarg preload, so the only VGPR-side cost is the
// vector ops on the dividend itself.
//
// Valid range: divisor in [1, INT32_MAX], dividend in [0, 2^31 - 1]. Compound
// grid-index decompositions sit well inside that range. For full 32-bit
// dividends, the "branchfree" libdivide u32 path is the upgrade.
//
// Algorithm: Granlund-Montgomery "add" form. The multiplier is always 32-bit
// (no 33-bit magic); the cost is the extra (tmp + dividend) before the shift.

struct MagicDiv
{
    uint32_t multiplier;
    uint32_t shift;

    // Host+device so a MagicDiv can be a member of a device-default-constructed
    // aggregate (e.g. direct_l1's RoundInvariantContext, reloaded from LDS).
    __host__ __device__ MagicDiv() : multiplier(0), shift(0) {}

    // Compute {multiplier, shift} for the given runtime-constant divisor.
    // Host-side: call once per divisor at kernel-launch time.
    explicit MagicDiv(uint32_t divisor)
    {
        // Cap at 31: shift reaches ceil(log2(divisor)), which is <= 31 for any
        // divisor in the documented [1, INT32_MAX] range. The bound also keeps
        // the shift well-defined (1u << 32 is undefined behavior) if an
        // out-of-contract divisor > 2^31 is ever passed.
        shift = 0;
        while(shift < 31 && (1u << shift) < divisor)
            ++shift;
        uint64_t one = 1;
        uint64_t m   = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;
        multiplier   = static_cast<uint32_t>(m);
    }

    __device__ uint32_t div(uint32_t dividend) const
    {
        uint32_t tmp = __umulhi(dividend, multiplier);
        return (tmp + dividend) >> shift;
    }

    __device__ void
    divmod(uint32_t dividend, uint32_t divisor, uint32_t& quotient, uint32_t& remainder) const
    {
        quotient  = div(dividend);
        remainder = dividend - quotient * divisor;
    }
};

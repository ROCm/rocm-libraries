// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <stdexcept>

namespace TensileLite
{
    // hipBLASLt passes this as the rocRoller WGM command argument
    // (rocroller/runtime_args_selection.hpp DEFAULT_WGM).
    constexpr int32_t RocRollerDefaultWgm = 2;

    // libdivide branchfree markers. Not Tensile's magicNumber().
    constexpr uint32_t RocRollerLibdivideAddMarker       = 0x40u;
    constexpr uint32_t RocRollerLibdivideNegativeDivisor = 0x80u;
    constexpr uint32_t RocRollerLibdivideShiftMask32     = 31u;

    // Hoisted kernargs for a rocRoller workgroup-mapping custom kernel.
    // Products are int32 wrapping products, matching rocRoller's launch tree.
    struct RocRollerWgmKernargs
    {
        int32_t  quotientTilesByBlock;
        int32_t  magicMultipleWgmMainBlock;
        int32_t  magicMultipleWgm;
        int32_t  workgroupMapping;
        int32_t  magicMultipleWgmTail;
        uint32_t magicShiftAndSignWgm;
        int32_t  magicMultipleNumTilesN;
        uint32_t magicShiftAndSignNumTilesN;
        uint32_t magicShiftAndSignWgmTail;
        uint32_t magicShiftAndSignWgmMainBlock;
        int32_t  quotientTilesMByWgm;
    };

    inline int32_t rocRollerS32(uint32_t value)
    {
        return static_cast<int32_t>(value);
    }

    inline uint32_t rocRollerClz32(uint32_t value)
    {
        return value == 0 ? 32u : static_cast<uint32_t>(__builtin_clz(value));
    }

    inline uint32_t rocRollerClz64(uint64_t value)
    {
        return value == 0 ? 64u : static_cast<uint32_t>(__builtin_clzll(value));
    }

    inline void rocRollerDiv64By32(uint32_t hi, uint32_t lo, uint32_t den, uint32_t* q, uint32_t* rem)
    {
        uint64_t n = (static_cast<uint64_t>(hi) << 32) | lo;
        *q         = static_cast<uint32_t>(n / den);
        *rem       = static_cast<uint32_t>(n - static_cast<uint64_t>(*q) * den);
    }

    inline void rocRollerDiv128By64(uint64_t hi, uint64_t lo, uint64_t den, uint64_t* q, uint64_t* rem)
    {
        unsigned __int128 n = (static_cast<unsigned __int128>(hi) << 64) | lo;
        *q                   = static_cast<uint64_t>(n / den);
        *rem                 = static_cast<uint64_t>(n - static_cast<unsigned __int128>(*q) * den);
    }

    // libdivide_s32_branchfree_gen. Divisor 0 is handled by the callers.
    inline void rocRollerS32Branchfree(int32_t divisor, int32_t* magic, uint32_t* moreOut)
    {
        uint32_t ud   = static_cast<uint32_t>(divisor);
        uint32_t absD = divisor < 0 ? static_cast<uint32_t>(-ud) : ud;
        uint32_t floorLog = 31u - rocRollerClz32(absD);
        if((absD & (absD - 1u)) == 0u)
        {
            *magic   = 0;
            *moreOut = floorLog | (divisor < 0 ? RocRollerLibdivideNegativeDivisor : 0u);
            return;
        }
        uint32_t proposed, rem;
        rocRollerDiv64By32(1u << (floorLog - 1u), 0u, absD, &proposed, &rem);
        proposed += proposed;
        uint32_t twiceRem = rem + rem;
        if(twiceRem >= absD || twiceRem < rem)
            proposed += 1u;
        uint32_t more = floorLog | RocRollerLibdivideAddMarker;
        proposed += 1u;
        *magic = rocRollerS32(proposed);
        if(divisor < 0)
            more |= RocRollerLibdivideNegativeDivisor;
        *moreOut = more;
    }

    // libdivide_u32_branchfree_gen. Divisors 0 and 1 are handled by the callers.
    inline void rocRollerU32Branchfree(uint32_t divisor, uint32_t* magic, uint32_t* shifts)
    {
        uint32_t floorLog = 31u - rocRollerClz32(divisor);
        if((divisor & (divisor - 1u)) == 0u)
        {
            *magic  = 0u;
            *shifts = (floorLog - 1u) & RocRollerLibdivideShiftMask32;
            return;
        }
        uint32_t proposed, rem;
        rocRollerDiv64By32(1u << floorLog, 0u, divisor, &proposed, &rem);
        proposed += proposed;
        uint32_t twiceRem = rem + rem;
        if(twiceRem >= divisor || twiceRem < rem)
            proposed += 1u;
        *magic  = 1u + proposed;
        *shifts = floorLog & RocRollerLibdivideShiftMask32;
    }

    inline void rocRollerS64Branchfree(int64_t divisor, int64_t* magic, uint32_t* moreOut)
    {
        uint64_t ud   = static_cast<uint64_t>(divisor);
        uint64_t absD = divisor < 0 ? static_cast<uint64_t>(-ud) : ud;
        uint32_t floorLog = 63u - rocRollerClz64(absD);
        if((absD & (absD - 1ull)) == 0ull)
        {
            *magic   = 0;
            *moreOut = floorLog | (divisor < 0 ? RocRollerLibdivideNegativeDivisor : 0u);
            return;
        }
        uint64_t proposed, rem;
        rocRollerDiv128By64(1ull << (floorLog - 1u), 0ull, absD, &proposed, &rem);
        proposed += proposed;
        uint64_t twiceRem = rem + rem;
        if(twiceRem >= absD || twiceRem < rem)
            proposed += 1ull;
        uint32_t more = floorLog | RocRollerLibdivideAddMarker;
        proposed += 1ull;
        *magic = static_cast<int64_t>(proposed);
        if(divisor < 0)
            more |= RocRollerLibdivideNegativeDivisor;
        *moreOut = more;
    }

    inline uint32_t rocRollerMagicMultipleU32(uint32_t arg)
    {
        if(arg == 0u)
            return 0xFFFFFFFFu / 2u;
        if(arg == 1u)
            return 0u;
        uint32_t magic, shifts;
        rocRollerU32Branchfree(arg, &magic, &shifts);
        return magic;
    }

    inline uint32_t rocRollerMagicShiftsU32(uint32_t arg)
    {
        if(arg == 0u)
            return 0u;
        if(arg == 1u)
            return 1u << 31;
        uint32_t magic, shifts;
        rocRollerU32Branchfree(arg, &magic, &shifts);
        return shifts & RocRollerLibdivideShiftMask32;
    }

    inline int32_t rocRollerMagicMultipleS32(int32_t arg)
    {
        if(arg == 0)
            return static_cast<int32_t>(2147483647 / 2);
        int32_t  magic;
        uint32_t more;
        rocRollerS32Branchfree(arg, &magic, &more);
        return magic;
    }

    inline uint32_t rocRollerMagicShiftAndSignS32(int32_t arg)
    {
        if(arg == 0)
            return 0u;
        int32_t  magic;
        uint32_t more;
        rocRollerS32Branchfree(arg, &magic, &more);
        return more;
    }

    inline int64_t rocRollerMagicMultipleS64(int64_t arg)
    {
        if(arg == 0)
            return static_cast<int64_t>(9223372036854775807ll / 2);
        int64_t  magic;
        uint32_t more;
        rocRollerS64Branchfree(arg, &magic, &more);
        return magic;
    }

    inline uint32_t rocRollerMagicShiftAndSignS64(int64_t arg)
    {
        if(arg == 0)
            return 0u;
        int64_t  magic;
        uint32_t more;
        rocRollerS64Branchfree(arg, &magic, &more);
        return more;
    }

    inline int32_t rocRollerAddS32(int32_t lhs, int32_t rhs)
    {
        return rocRollerS32(static_cast<uint32_t>(lhs) + static_cast<uint32_t>(rhs));
    }

    inline int32_t rocRollerSubS32(int32_t lhs, int32_t rhs)
    {
        return rocRollerS32(static_cast<uint32_t>(lhs) - static_cast<uint32_t>(rhs));
    }

    inline int32_t rocRollerMulS32(int32_t lhs, int32_t rhs)
    {
        return rocRollerS32(static_cast<uint32_t>(lhs) * static_cast<uint32_t>(rhs));
    }

    inline int32_t rocRollerMulHiS32(int32_t lhs, int32_t rhs)
    {
        return static_cast<int32_t>((static_cast<int64_t>(lhs) * static_cast<int64_t>(rhs)) >> 32);
    }

    inline int32_t rocRollerAsrS32(int32_t value, uint32_t count)
    {
        count &= 31u;
        if(count == 0u)
            return value;
        uint32_t logical = static_cast<uint32_t>(value) >> count;
        if(value < 0)
            logical |= ~uint32_t{0} << (32u - count);
        return rocRollerS32(logical);
    }

    inline int32_t rocRollerShlS32(int32_t value, uint32_t count)
    {
        count &= 31u;
        return rocRollerS32(static_cast<uint32_t>(value) << count);
    }

    // Signed fast division. Divisor 0 follows MagicMultiple's special case.
    inline int32_t rocRollerMagicDivS32(int32_t numerator, int32_t divisor)
    {
        int32_t  magic    = rocRollerMagicMultipleS32(divisor);
        uint32_t bitfield = rocRollerMagicShiftAndSignS32(divisor);
        int32_t  q       = rocRollerAddS32(rocRollerMulHiS32(numerator, magic), numerator);
        int32_t  signOfQ = rocRollerAsrS32(q, 31u);
        uint32_t shifts  = bitfield & RocRollerLibdivideShiftMask32;
        int32_t  oneShifted  = rocRollerShlS32(1, shifts);
        int32_t  magicIsPow2 = magic == 0 ? -1 : 0;
        int32_t  handle  = rocRollerAddS32(q, signOfQ & rocRollerAddS32(oneShifted, magicIsPow2));
        int32_t  shifted = rocRollerAsrS32(handle, shifts);
        int32_t  sign    = rocRollerAsrS32(rocRollerShlS32(rocRollerS32(bitfield), 24u), 31u);
        return rocRollerSubS32(shifted ^ sign, sign);
    }

    inline int32_t rocRollerNumTiles(int64_t size, uint32_t tile)
    {
        if(tile == 0u || (tile & (tile - 1u)) != 0u)
            throw std::runtime_error("rocRoller WGM macrotile must be a positive power of two");
        uint32_t log     = 31u - rocRollerClz32(tile);
        int64_t  shifted = (size + static_cast<int64_t>(tile) - 1) >> log;
        return static_cast<int32_t>(shifted);
    }

    inline RocRollerWgmKernargs evaluateRocRollerWgmKernargs(int64_t  m,
                                                             int64_t  n,
                                                             uint32_t tileM,
                                                             uint32_t tileN,
                                                             int32_t  wgm = RocRollerDefaultWgm)
    {
        int32_t ntM   = rocRollerNumTiles(m, tileM);
        int32_t ntN   = rocRollerNumTiles(n, tileN);
        int32_t w     = wgm;
        int32_t total = rocRollerMulS32(ntM, ntN);
        int32_t block = rocRollerMulS32(w, ntN);
        int32_t qBlock = rocRollerMagicDivS32(total, block);
        int32_t main   = rocRollerMulS32(qBlock, block);
        int32_t qM   = rocRollerMagicDivS32(ntM, w);
        int32_t tail = rocRollerSubS32(ntM, rocRollerMulS32(qM, w));
        RocRollerWgmKernargs out{};
        out.quotientTilesByBlock          = qBlock;
        out.magicMultipleWgmMainBlock     = rocRollerMagicMultipleS32(main);
        out.magicMultipleWgm              = rocRollerMagicMultipleS32(w);
        out.workgroupMapping              = w;
        out.magicMultipleWgmTail          = rocRollerMagicMultipleS32(tail);
        out.magicShiftAndSignWgm          = rocRollerMagicShiftAndSignS32(w);
        out.magicMultipleNumTilesN        = rocRollerMagicMultipleS32(ntN);
        out.magicShiftAndSignNumTilesN    = rocRollerMagicShiftAndSignS32(ntN);
        out.magicShiftAndSignWgmTail      = rocRollerMagicShiftAndSignS32(tail);
        out.magicShiftAndSignWgmMainBlock = rocRollerMagicShiftAndSignS32(main);
        out.quotientTilesMByWgm           = qM;
        return out;
    }
}

#pragma once

// LDS swizzle for the global -> LDS -> matrix-core register path.
//
// Global memory tensor is NHWC with 16-bit elements. Global -> LDS loads use
// uint4 elements (128-bit, 8 channels) and write to swizzled LDS addresses.
//
// =============================================================================
// Usage on wave64 MFMA (e.g. CDNA4 / gfx950) and wave32 WMMA targets
// =============================================================================
// One SwizzleT<C> instance defines a single LDS layout that serves BOTH read
// models. The class is target-agnostic: what differs is the lane mapping the
// caller derives and the ds_read width it issues, NOT the swizzle. The models
// are described below by their access pattern (wave size, ds_read width, lane-
// group size) rather than by product name.
//
// Write (HBM -> LDS), identical on both arches:
//   Each lane owns uint4 slot i = tid (plus block-strided passes). Recover where
//   the slot lives with the inverses and load from that global coordinate:
//       int col = Sw::x(i);    // spatial column
//       int c8  = Sw::c8(i);   // 8-channel band
//   offset_uint4 / x / c8 are a bijection, so every slot is written once.
//
// Read (LDS -> registers), same functions, target-specific argument derivation:
//                    | wave64 MFMA             | wave32 WMMA
//     wave size      | 64                      | 32
//     read width     | uint2/lane (ds_read b64)| uint4/lane (ds_read b128)
//     read call      | Sw::offset_uint2(x, c4) | Sw::offset_uint4(x, c8)
//     lane -> coord  | x = lane%16, c4 = lane/16| x = lane%16, c8 = lane/16
//
// Bank-free status (64 banks * 4 B = 256 B/clock; LDS bank index = (addr/4) %
// 64). A conflict is decided within the set of lanes serviced in one phase, and
// the phase structure depends on the read width:
//   - uint2 via ds_read_b64: 2 phases of 32 consecutive lanes (T0-31, T32-63).
//     Bank-free needs the 32 uint2 offsets in a phase distinct mod 32.
//   - uint4 via ds_read_b128: phases of 16 consecutive lanes (T0-15, T16-31,
//     ...). Bank-free needs the 16 uint4 offsets in a phase distinct mod 16.
// (NB the wave64 tf32 uint4 read, used by SwizzleT_fp32, instead uses 4 phases
// of 16 *scrambled* lanes -- a third model again; see that template.)
// These are DIFFERENT criteria, but the per-row rotation (see X_SHIFT) satisfies
// both: for C in {32, 64, 128, 256} a C bank-free under one is bank-free under
// the other, so SwizzleT serves both read models. See is_bank_conflict_free()
// for the uint4/mod-16 predicate and the model it assumes, SwizzleWmmaTest for
// the uint4/16-lane check, and SwizzleTest::MfmaLoadConflicts for the
// uint2/32-lane check.
// =============================================================================
//
// Which LDS read model a SwizzleT instance is tuned for. The write (HBM -> LDS)
// and the offset<->coordinate bijection are identical across models; only the
// per-row rotation amount (X_SHIFT) differs, because the two reads partition
// lanes into bank-resolution phases differently (see the file-header table):
//
//   Wave32B128 (default): the wave32 WMMA uint4 read (16 CONSECUTIVE lanes) and
//     the wave64 MFMA uint2 read (32 consecutive lanes). Bank-free for C < 128
//     needs the per-row rotation X_SHIFT = max(0, 4 - log2(C8)); for C >= 128 it
//     is 0 and the rotation degenerates to the additive (c8 + x) % C8.
//
//   Wave64B128: the wave64 uint4 read that dispatches in 4 SCRAMBLED 16-lane
//     phases (T0-3,T12-15,T20-23,T24-27, ...).
//     Under those phases the additive rotation (X_SHIFT = 0) is bank-free, while
//     the Wave32B128 rotation reintroduces conflicts for C < 128. This is the
//     read direct_l1 issues (a full uint4 = 8 channels per lane on wave64), so
//     its C(64) input tile selects this model. (For C >= 128 the two models pick
//     the same X_SHIFT = 0, so they coincide there.)
enum class SwizzleReadModel
{
    Wave32B128,
    Wave64B128,
};

// C_ is the number of channels in the tile (BLOCK_C).
template <int C_, SwizzleReadModel Model = SwizzleReadModel::Wave32B128>
struct SwizzleT
{
    static constexpr int C  = C_;
    static constexpr int C8 = C / 8;
    static constexpr int C4 = C / 4;

    // Correctness requirement: C must be a positive number divisible by 8.
    static_assert(C > 0, "SwizzleT requires positive C");
    static_assert(C % 8 == 0, "SwizzleT requires C divisible by 8 (uint4 = 8 channels)");

private:
    static constexpr int ilog2(int n)
    {
        int r = 0;
        while(n > 1)
        {
            n >>= 1;
            ++r;
        }
        return r;
    }

    // Per-row rotation amount: rotate row x's c8 bands by (x >> X_SHIFT).
    //
    // Wave32B128: a read group covers 16 consecutive columns x = 0..15 at one c8
    // band (see the file header for the per-arch read models, and
    // is_bank_conflict_free for the hardware assumptions); bank-free requires
    // their 16 offsets distinct mod 16, i.e. C8 >= 16. For narrower tiles
    // (C8 < 16), 16/C8 lanes share x mod C8 and would land on the same band.
    // Rotating by (x >> X_SHIFT) with X_SHIFT = 4 - log2(C8) separates exactly
    // those lanes. For C8 >= 16 (C >= 128) X_SHIFT is 0 and the rotation is
    // (c8 + x) % C8, byte-identical to the original swizzle.
    //
    // Wave64B128: under the scrambled 4-phase read the additive rotation
    // (X_SHIFT = 0) is what makes direct_l1's C(64) tile bank-free, whereas the
    // Wave32B128 narrowing shift reintroduces conflicts there. So this model pins
    // X_SHIFT = 0. (For C >= 128 both models are 0 anyway, so they only diverge
    // for C < 128. The additive rotation is not bank-free under these phases for
    // every C8 -- e.g. C8 = 4 conflicts -- but it is for the C8 = 8 tile this
    // kernel uses; see the note on is_bank_conflict_free below.)
    static constexpr int X_SHIFT =
        (Model == SwizzleReadModel::Wave64B128) ? 0 : ((4 - ilog2(C8)) > 0 ? (4 - ilog2(C8)) : 0);

public:
    // The public interface is signed int (callers mix these results into signed
    // index math, so a signed return avoids silent int->unsigned conversions in
    // their expressions). Precondition: all coordinate/offset arguments must be
    // non-negative -- they are lane- and loop-derived at every call site. This is
    // a caller contract, not enforced here: a negative argument would wrap when
    // converted to the unsigned locals below and yield a wrong offset.
    //
    // Internally each function uses unsigned arithmetic, which lets the compiler
    // lower the divisions/moduli optimally -- % 2 // 2 become a mask and a shift,
    // and % C8 // C8 become a mask and a shift whenever C8 is a power of two
    // (the C in {32, 64, 128, 256} fprop tiles). On signed int the same expressions
    // emit a sign-correction sequence.

    // Return true if C yields a bank-conflict-free LDS swizzle under the
    // uint4 / consecutive-16-lane ds_read_b128 model:
    //   - 64 LDS banks of 4 B; each lane's uint4 read spans 4 consecutive banks.
    //   - Conflicts are resolved within a group of 16 consecutive lanes (T0-15,
    //     T16-31, ...); that group reads 16 consecutive columns x = 0..15 at one
    //     c8 band.
    // Under this model the read is conflict-free iff the 16 uint4 offsets are
    // distinct mod 16. This predicate answers ONLY that uint4 / consecutive-16
    // question (the wave32 WMMA fp16/bf16 read via SwizzleT).
    //
    // The wave64 fp16/bf16 MFMA read is a different, strictly LOOSER criterion
    // (uint2 / 32-lane group -> distinct mod 32; see the file header and
    // SwizzleTest::MfmaLoadConflicts): some C with C8 >= 16 but not a multiple of
    // 16 (e.g. C=192) are bank-free for that uint2 read yet return false here.
    // So this predicate is sufficient but not necessary for the uint2 path -- a
    // false result does not prove conflicts there. (The wave64 tf32 read is a
    // third model again -- scrambled b128 phases via SwizzleT_fp32 -- not this
    // predicate.) For the shipped C in {32, 64, 128, 256} the uint2 and uint4
    // criteria agree. A read model with a different bank count, lane-group size,
    // or read width needs its own analysis. See SwizzleWmmaTest in
    // test/swizzle_test.cpp, which cross-checks this against the measured pattern.
    //
    // A non-bank-free C still round-trips correctly (offset_uint2/x/c8 stay
    // mutual inverses), it just incurs those conflicts on the read. This lets a
    // caller decide for itself (static_assert, pick a different tile, or accept
    // the conflict) rather than the swizzle forcing the choice.
    //
    // Distinct mod 16 holds in two regimes:
    //   - C8 < 16: X_SHIFT = 4 - log2(C8) must partition the 16/C8 lanes sharing
    //     x mod C8 into equal classes, which needs C8 to be a power of two.
    //   - C8 >= 16: X_SHIFT = 0, so offset mod 16 = (x*C8 + c8 + x) mod 16; this
    //     is distinct across x = 0..15 exactly when C8 is a multiple of 16.
    // Note C8 a multiple of 16 admits non-powers-of-two (e.g. C=384, C8=48), so
    // this is strictly weaker than "C is a power of two".
    //
    // This predicate answers the Wave32B128 model only (its X_SHIFT). The
    // Wave64B128 model's scrambled-phase conflict count has no comparably clean
    // closed form (it is 0 only for C8 in {1, 2, 3, 8}, e.g. direct_l1's C(64)
    // tile), so callers of that model verify bank-freedom by the enumerated
    // phase test rather than this predicate.
    static constexpr bool is_bank_conflict_free()
    {
        static_assert(Model == SwizzleReadModel::Wave32B128,
                      "is_bank_conflict_free() models the Wave32B128 read only");
        return C8 < 16 ? (C8 & (C8 - 1)) == 0 : (C8 % 16 == 0);
    }

    // Map the (x, c4) coordinates to an offset in LDS.
    //
    // The offset is in units of uint2 (64-bit), appropriate for LDS -> register loading.
    static constexpr int offset_uint2(int x_, int c4_)
    {
        unsigned x         = x_;
        unsigned c4        = c4_;
        unsigned c4m2      = c4 % 2;
        unsigned c8        = c4 / 2;
        unsigned offset_x  = x * C4;
        unsigned offset_c8 = (c8 + (x >> X_SHIFT)) % C8;

        return offset_x + offset_c8 * 2 + c4m2;
    }

    // Map (x, c8) coordinates to a uint4 offset in LDS.
    // Inverse of x() / c8(); consistent with offset_uint2(x, c8*2) / 2.
    static constexpr int offset_uint4(int x, int c8)
    {
        return static_cast<unsigned>(offset_uint2(x, c8 * 2)) / 2;
    }

    // Map an offset in LDS to its x-coordinate.
    //
    // The offset is in units of uint4 (128-bit), appropriate for Global -> LDS loading.
    static constexpr int x(int offset_uint4) { return static_cast<unsigned>(offset_uint4) / C8; }

    // Map an offset in LDS to its c8-coordinate.
    //
    // The offset is in units of uint4 (128-bit), appropriate for Global -> LDS loading.
    static constexpr int c8(int offset_uint4_)
    {
        unsigned offset_uint4 = offset_uint4_;
        unsigned x            = SwizzleT::x(offset_uint4);
        unsigned c8           = (offset_uint4 + C8 - (x >> X_SHIFT) % C8) % C8;
        return c8;
    }

private:
    // Compile-time guard: offset_uint2 and the x()/c8() inverse must round-trip
    // for every (x, c4) in the tile. A uint2 slot is 2 * uint4_slot + c4m2, so
    // the inverse recovers c4 as c8(slot) * 2 + (offset & 1). Kernels recover
    // (col, c4) from a linear LDS slot via these methods; if a future edit to
    // offset_uint2 or X_SHIFT breaks the inversion, this fails the build rather
    // than silently desyncing the load and global-memory write sides (the bug
    // that motivated routing the tf32 wgrad kernels through these methods). The
    // c8 term is periodic in x, so iterating x in [0, 16) covers all rows.
    static constexpr bool inverses_round_trip()
    {
        for(int x = 0; x < 16; ++x)
            for(int c4 = 0; c4 < C4; ++c4)
            {
                const int off2 = offset_uint2(x, c4);
                const int off4 = off2 / 2;
                if(SwizzleT::x(off4) != x || c8(off4) * 2 + (off2 & 1) != c4)
                    return false;
            }
        return true;
    }
    static_assert(inverses_round_trip(),
                  "SwizzleT::offset_uint2 and x()/c8() are not mutual inverses for this C");
};

// FP32 (4-byte element) variant of the swizzle.
//
// In FP32 mode every uint4 holds 4 channels (vs 8 for BF16/FP16), and every
// MFMA operand of width 4 occupies 16 bytes (vs 8). All LDS access in the TF32
// kernels collapses to uint4 granularity, so this template only needs
// offset_uint4 / x / c4 — there is no offset_uint2 analog.
//
// Logical layout: a row of width "1 c-block" (= 4 channels = 1 uint4) maps to
//   addr_uint4(x, c4) = x * C4 + ((4*x + c4) % C4)
// where C4 = C / 4 is the number of c-blocks per row.
//
// This template assumes a DIFFERENT read model from SwizzleT (see that class's
// file-header usage block): TF32 reads uint4/lane and on gfx950 ds_read_b128
// dispatches in 4 phases of 16 *scrambled* (non-contiguous) lanes -- e.g. phase
// 0 is T0-3,T12-15,T20-23,T24-27 -- so the bank-free criterion is per-phase over
// those scrambled groups, not the consecutive-16-lane rule SwizzleT's uint4 read
// uses. The two are not interchangeable. The exact phase table is enumerated
// in SwizzleFp32Test.
//
// Design choices:
//   - Use uint4 (= 4 fp32): largest ds_read / HBM-load granularity, and a
//     natural match to the TF32 4c group of 4 channels.
//   - The `4*x` coefficient (vs `1*x` in SwizzleT::offset_uint2) is a pure
//     address permutation — no XOR — picked by brute-force search so each
//     phase lands on 16 distinct banks for all C4 in {16, 32} and kw shifts
//     S in {0, 1, 2}.
//   - See SwizzleFp32Test::DsReadB128BankFree for the enumerated check.
//
// C_ is the number of channels in the tile (BLOCK_C).
template <int C_>
struct SwizzleT_fp32
{
    static constexpr int C  = C_;
    static constexpr int C4 = C / 4;
    static constexpr int C2 = C / 2;

    // Correctness requirement: C must be a positive number divisible by 4,
    // because the swizzle arithmetic itself needs C % 4 == 0 (uint4 = 4 fp32 channels).
    static_assert(C > 0, "SwizzleT_fp32 requires positive C");
    static_assert(C % 4 == 0, "SwizzleT_fp32 requires C divisible by 4");

    // Signed interface, unsigned internals (see SwizzleT). Precondition: callers
    // must pass non-negative coordinates/offsets (not enforced; a negative value
    // wraps in the unsigned locals). Unsigned then lets the compiler lower / C4
    // and % C4 to a shift and a mask when C4 is a power of two; signed int would
    // force a sign-correction sequence instead.

    // Map (x, c4) coordinates to a uint4 offset in LDS.
    static constexpr int offset_uint4(int x_, int c4_)
    {
        unsigned x  = x_;
        unsigned c4 = c4_;
        return x * C4 + ((4 * x + c4) % C4);
    }

    // Map a uint4 offset to its x-coordinate.
    static constexpr int x(int offset_uint4) { return static_cast<unsigned>(offset_uint4) / C4; }

    // Map a uint4 offset to its c4-coordinate (4-channel block index).
    static constexpr int c4(int offset_uint4_)
    {
        unsigned offset_uint4 = offset_uint4_;
        unsigned x_v          = SwizzleT_fp32::x(offset_uint4);
        return (offset_uint4 + C4 - (4 * x_v) % C4) % C4;
    }

private:
    // Compile-time guard: offset_uint4 and the x()/c4() inverse must round-trip
    // for every (x, c4) in the tile, so a future edit to the swizzle cannot
    // silently desync a kernel's load and global-memory write sides. The c4 term
    // is periodic in x, so iterating x in [0, 16) covers all rows.
    static constexpr bool inverses_round_trip()
    {
        for(int x = 0; x < 16; ++x)
            for(int c4_ = 0; c4_ < C4; ++c4_)
            {
                const int off4 = offset_uint4(x, c4_);
                if(SwizzleT_fp32::x(off4) != x || c4(off4) != c4_)
                    return false;
            }
        return true;
    }
    static_assert(inverses_round_trip(),
                  "SwizzleT_fp32::offset_uint4 and x()/c4() are not mutual inverses for this C");
};

// uint2 (64-bit, 4-channel / c4) LDS swizzle for a transpose read.
//
// Structurally this is a c4 / 64-bit swizzle: its only unit is c4 = 4 channels =
// one uint2 = 64 bits (the 4 x 16-bit of a ds_read_tr16_b64), so unlike SwizzleT
// it has no uint4 / c8 concept -- just offset_uint2 and its uint2-unit inverses.
// The "wgrad" name reflects the only current consumer (grouped 4c wgrad), not
// the structure.
//
// It is a separate template from SwizzleT (which also has a uint2/c4 view) not
// because of the I/O width but because of the read access pattern: this swizzle
// targets the transpose read's 4 x × 8 c4 phase partition (the alpha(x) rotation
// below), whereas SwizzleT::offset_uint2 targets fprop's 16 x × 2 c4 partition
// (the x >> X_SHIFT rotation). No single linear-shift uint2 swizzle is bank-free
// under both, so the transpose read uses this dedicated template.
//
// Formula:
//   alpha(x) = 16 * (x % 2) + 8 * ((x / 2) % 2)        // in {0, 8, 16, 24}
//   offset_uint2(x, c4) = x * C4 + (c4 + alpha(x)) % C4
//
// Bank-free argument (32 lanes per phase, 64 LDS banks):
//   - Any 4 consecutive x cover all 4 (x%2, (x/2)%2) combinations, so
//     alpha takes 4 disjoint values {0, 16, 8, 24}.
//   - c4 in [0, 8) shifted by alpha lands in 4 disjoint 8-blocks of [0, 32).
//   - C=64 (C4=16): x*C4 mod 32 = 16*x_0 splits 32 lanes by x_0; within each
//     half, alpha mod 16 = 8*x_1 splits by x_1 into two 8-blocks. All distinct.
//   - C=128 (C4=32): x*C4 mod 32 = 0, alpha alone gives 32 distinct offsets.
//
// Write bijection: c4 -> (c4 + alpha(x)) % C4 is a permutation of [0, C4)
// for each x, so offset_uint2 is 1:1 on [0, BLOCK_W) x [0, C4).
//
// Used by grouped_4c_wgrad_tf32.h. fp16/bf16 4c wgrad still uses SwizzleT
// and has the same inherent conflict (out of scope for this fix).
template <int C_>
struct SwizzleT_wgrad
{
    static constexpr int C  = C_;
    static constexpr int C4 = C / 4;

    // Correctness requirement: C must be a positive number divisible by 4, because
    // the swizzle arithmetic itself needs C % 4 == 0 (uint2 c4 = 4c group).
    static_assert(C > 0, "SwizzleT_wgrad requires positive C");
    static_assert(C % 4 == 0, "SwizzleT_wgrad requires C divisible by 4");

private:
    // Per-row c4 permutation seed. See header comment for derivation.
    static constexpr int alpha(int x_)
    {
        unsigned x = x_;
        return (16 * (x % 2) + 8 * ((x / 2) % 2)) % C4;
    }

public:
    // Signed interface, unsigned internals (see SwizzleT). Precondition: callers
    // must pass non-negative coordinates/offsets (not enforced; a negative value
    // wraps in the unsigned locals). Unsigned then lets the compiler lower / C4
    // and % C4 to a shift and a mask when C4 is a power of two; signed int would
    // force a sign-correction sequence instead. The c4_uint2 reverse subtracts
    // alpha(x) but adds C4 first, and alpha(x) < C4 <= rem + C4, so the running
    // value stays in (0, 2*C4) and the unsigned modulo yields the same residue a
    // signed computation would.

    // (x, c4) -> uint2 LDS offset.
    static constexpr int offset_uint2(int x_, int c4_)
    {
        unsigned x  = x_;
        unsigned c4 = c4_;
        return x * C4 + (c4 + alpha(x)) % C4;
    }

    // Map offset_uint2 to the x-coordinate.
    static constexpr int x_uint2(int offset_uint2)
    {
        return static_cast<unsigned>(offset_uint2) / C4;
    }

    // Map offset_uint2 to the c4-coordinate (c / 4).
    static constexpr int c4_uint2(int offset_uint2_)
    {
        unsigned offset_uint2 = offset_uint2_;
        unsigned x            = SwizzleT_wgrad::x_uint2(offset_uint2);
        unsigned rem          = offset_uint2 % C4;
        return (rem - alpha(x) + C4) % C4;
    }

private:
    // Compile-time guard: offset_uint2 and the x_uint2()/c4_uint2() inverse must
    // round-trip for every (x, c4) in the tile, so a future edit to the alpha(x)
    // rotation cannot silently desync a kernel's load and global-memory write
    // sides. alpha(x) depends only on x mod 4, so iterating x in [0, 16) covers
    // all rows.
    static constexpr bool inverses_round_trip()
    {
        for(int x = 0; x < 16; ++x)
            for(int c4_ = 0; c4_ < C4; ++c4_)
            {
                const int off2 = offset_uint2(x, c4_);
                if(x_uint2(off2) != x || c4_uint2(off2) != c4_)
                    return false;
            }
        return true;
    }
    static_assert(inverses_round_trip(),
                  "SwizzleT_wgrad::offset_uint2 and x_uint2()/c4_uint2() are not mutual inverses "
                  "for this C");
};

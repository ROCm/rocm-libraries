#pragma once

// The parameters one direct_wgrad kernel is compiled for.
//
// docs/algorithms/direct/direct-wgrad.md is the algorithm these configure, and
// docs/algorithms/direct/direct-wgrad-config-table.md the set of them the table holds.

#include "mathutil.h"

namespace hipconv::cdna4::direct_wgrad
{
static constexpr int WAVE_SIZE = 64;

// Waves in a workgroup, which every arrangement has to add up to.
static constexpr int WAVES_PER_WORKGROUP = 8;

// VGPRs a config may spend on the gradient and the operands together.
//
// Past this the symptom is a spill rather than a diagnostic, so config_table.h asserts it
// entry by entry.
static constexpr int TILE_VGPR_BUDGET = 192;

// Channels a swizzled LDS row needs, on either operand: TransposeSwizzle needs C4 >= 8.
static constexpr int MIN_SWIZZLED_CHANS = 32;

// The MFMA the compute phase tiles onto: 16x16x32.
//
// M is the input-channel axis of the gradient tile, N the output-channel axis, and K the Q
// reduction, so one instruction consumes 32 output columns. It sits here rather than with the
// operand loaders because the LDS geometry and the grid both block up that K.
constexpr int MFMA_M = 16;
constexpr int MFMA_N = 16;
constexpr int MFMA_K = 32;

// How a config spreads the eight waves over C, K, the groups and the spatial axis.
struct Arrangement
{
    int waves_c;
    int waves_k;
    int waves_q;
    int waves_g = 1;

    constexpr bool operator==(const Arrangement&) const = default;
};

struct Config
{
    int kh = 3; // filter height, and the depth of the delta register ring
    int kw = 3; // filter width, and the S tiles a compute phase holds at once

    int wave_c16; // input channels per wave / 16
    int wave_k16; // output channels per wave / 16

    int waves_c;     // waves along C
    int waves_k;     // waves along K
    int waves_q = 1; // waves along the spatial axis, one spatial item each
    int waves_g = 1; // groups the workgroup covers at once

    int unfold_n      = 1; // images packed into one column block
    int prefetch_rows = 2; // rows the memory phase runs ahead of its compute phase

    constexpr int waves() const { return waves_c * waves_k * waves_q * waves_g; }
    constexpr int threads() const { return WAVE_SIZE * waves(); }

    // Waves sharing one spatial item, and so the waves that load one of its rows.
    constexpr int waves_per_item() const { return waves_c * waves_k * waves_g; }

    // Channels and filters per wave, per group, and per workgroup block.
    constexpr int wave_c() const { return 16 * wave_c16; }
    constexpr int wave_k() const { return 16 * wave_k16; }
    constexpr int group_c() const { return wave_c() * waves_c; }
    constexpr int group_k() const { return wave_k() * waves_k; }
    constexpr int block_c16() const { return wave_c16 * waves_c * waves_g; }
    constexpr int block_k16() const { return wave_k16 * waves_k * waves_g; }
    constexpr int block_c() const { return 16 * block_c16(); }
    constexpr int block_k() const { return 16 * block_k16(); }

    // Output columns one packed image contributes to a column block, and the LDS columns it
    // occupies: its own plus the kw - 1 halo the filter shifts read past them.
    constexpr int w_unfold() const { return MFMA_K / unfold_n; }
    constexpr int s_cols_per_image() const { return w_unfold() + kw - 1; }

    // VGPRs one wave spends on the gradient, and on the two operands.
    constexpr int acc_vgprs() const { return kh * kw * wave_c16 * wave_k16 * 4; }
    constexpr int operand_vgprs() const { return 4 * (kw * wave_c16 + kh * wave_k16); }

    // LDS row buffers per ring: one front buffer plus the rows in flight.
    //
    // Each is its own __shared__ declaration, never an array of arrays: the waitcnt pass keeps
    // a partial vmcnt on a transpose read only where it sees an LDS object distinct from the
    // in-flight DMA's, and one allocation carved into slots is a single object.
    constexpr int row_buffers() const { return prefetch_rows + 1; }

    // Prologue-only buffers holding the rows that prime the delta register ring.
    constexpr int scratch_rows() const { return kh - 1; }

    // Main-loop unroll that folds every ring slot to a compile-time constant.
    //
    // Slots are keyed on the iteration, so a slot index folds once the unroll is a multiple of
    // the ring depth; the register ring adds a factor of kh.
    constexpr int unroll() const { return lcm(row_buffers(), kh); }

    // Whether two table entries name the same kernel.
    //
    // Every field, because the spread alone does not identify an entry where a filter size
    // carries two wave tiles. See direct-wgrad-config-table.md.
    constexpr bool operator==(const Config&) const = default;
};

// Which slice of the workgroup's work a wave owns.
//
// A wave loads the item it computes on, which confines its buffer loads to one image: the row
// loader deals rounds by load_wave, and item selects both the LDS sub-buffer the wave writes
// and the one it reads back.
struct WaveSlot
{
    int c_base; // channel origin in the workgroup's block, which the LDS reads use
    int k_base;
    int item;
    int load_wave;
    int group;
    int c_local; // the same origin inside the wave's group, dW being contiguous over one group
};

// The wave index decomposes with the item fastest, then the group, then K, then C.
//
// Item fastest puts the waves sharing a channel tile next to each other, which is the run the
// epilogue reduces over; the group above it keeps that run contiguous under several groups.
constexpr WaveSlot wave_slot(const Config& cfg, int wave)
{
    const int rest  = wave / cfg.waves_q;
    const int group = rest % cfg.waves_g;
    const int ck    = rest / cfg.waves_g;

    const int c_local = ck / cfg.waves_k * cfg.wave_c();
    const int k_local = ck % cfg.waves_k * cfg.wave_k();

    return {group * cfg.group_c() + c_local,
            group * cfg.group_k() + k_local,
            wave % cfg.waves_q,
            rest,
            group,
            c_local};
}

} // namespace hipconv::cdna4::direct_wgrad

#pragma once

// Global-to-LDS loader for one tensor row of the direct_wgrad main loop.
//
// One class serves both operands: S rows (W x C, NHWC) and delta rows (Q x K, NPQK). The tile
// shape is the only difference between them, and Layout carries it.

#include "bunnies.hpp"
#include "bunnies_cdna4.hpp"
#include "lds_layout.h"

#include <hip/hip_runtime.h>

#include <array>

namespace hipconv::cdna4::direct_wgrad
{

// Shape of one channels-last source tensor.
//
// S is (images, rows = H, cols = W, chans = C); delta is (images, rows = P, cols = Q,
// chans = K). The caller owns the row schedule, so the loader needs nothing else.
struct RowTensorPars
{
    int images;
    int rows;
    int cols;
    int chans;
};

// Loads one tensor row into one slot of the LDS ring.
//
// Padding is implicit: a row or column outside the image gets a voffset past the buffer window
// and reads zero. That supplies the zero rows the prologue's delta register ring expects, and
// the left and right halo of every S row.
//
// Address split, following direct_l1's InputLoader: the origin of the item's first image folds
// into the 64-bit base pointer, the row term rides in soffset (wave-uniform, so an SGPR), and
// the per-lane column, channel and packed-image terms sit in voffset. The buffer window stops
// at the end of the images the item names, so the eight-channel over-read at the last column
// of the last row stays inside the item and reads zero.
//
// The rounds of an item's row are dealt out over the NumWaves waves that run that item, so a
// wave's loads never leave the one image its item names, and the resource, window and over-read
// bound are the same whatever Items is. Dealing them by position in the concatenated buffer would
// let a wave straddle two items, and so two images, needing a resource spanning the run with the
// item origin in soffset.
//
// Channels past the tensor's own C or K are unguarded. A lane whose eight-channel group starts
// inside the tensor and runs past the real channel count reads the next pixel's leading channels,
// and under the unfold it can reach the next packed image. The data is finite and lands in
// accumulator columns past the group's end, which the epilogue drops; guarding costs a compare
// on every load.
//
// LaneBytes is the per-lane width of one load, 4 or 16.
// It sets the round size, and so how much of a row buffer is padding.
template <typename Layout, typename datatype_t, int NumWaves, int Items = 1, int LaneBytes = 16>
class RowLoader
{
    static constexpr int rsrc_data_format = 1 << 15;

public:
    static_assert(sizeof(datatype_t) == 2);

    using Swizzle = typename Layout::Swizzle;

    static constexpr int wave_size  = arch::wave_size;
    static constexpr int elem_bytes = static_cast<int>(sizeof(datatype_t));

    // Elements in one item's tile row, and the rounds they split into.
    //
    // A round is one wave's buffer load: wave_size * lane_bytes of the row, landing at
    // lane * lane_bytes from the load's LDS base. The row pads to a whole round, giving 1 KiB
    // of granularity at 16 bytes per lane and 256 B at 4.
    static constexpr int tile_elems  = Layout::size_elems;
    static constexpr int lane_bytes  = LaneBytes;
    static constexpr int lane_elems  = lane_bytes / elem_bytes;
    static constexpr int round_elems = wave_size * lane_bytes / elem_bytes;
    static constexpr int item_rounds = tile_elems / round_elems;

    // Rounds the whole ring slot carries, one item's row after another.
    static constexpr int items        = Items;
    static constexpr int total_rounds = items * item_rounds;

    static_assert(tile_elems % round_elems == 0,
                  "one item's row must divide into whole load rounds; pad the layout's Cols");

    // Rounds that carry live columns.
    // The layout pads to a whole round and no further, so this is every round of the item.
    static constexpr int live_rounds =
        (Layout::live_cols * Layout::chans + round_elems - 1) / round_elems;
    static_assert(live_rounds == item_rounds,
                  "an item's row must be padded to a whole load round and no further; the "
                  "rounds past it belong to the buffer's shared drain, not to the item");

    // Buffer loads every wave issues per row, for the main loop's s_waitcnt vmcnt.
    //
    // The same count for every wave whatever the row's length, so one s_waitcnt immediate
    // serves the whole workgroup. The rounds need not divide over the waves; the surplus lands
    // on the drain. Predicating a wave out of a real round costs far more, because the waitcnt
    // pass cannot tell whether a guarded load issued, assumes it did not, and tightens every
    // later wait to the count for a wave that loaded nothing.
    static constexpr int loads_per_row = (item_rounds + NumWaves - 1) / NumWaves;
    static constexpr bool has_drain    = loads_per_row * NumWaves != item_rounds;

    // Elements in one ring slot: every item's row, then the drain if there is surplus.
    // Nothing reads the drain, so every item's waves pile their surplus onto the one round.
    static constexpr int buffer_elems = (total_rounds + (has_drain ? 1 : 0)) * round_elems;

    using load_inst = arch::buffer_load_lds<lane_bytes>;

    // `item` names both the wave's spatial partition and the sub-buffer its rows land in.
    //
    // `valid` is false for a partition whose item ran past the end of the segment; `image` is
    // the first of the Layout::n_unfold images the item packs.
    __device__ RowLoader(const RowTensorPars& pars,
                         const datatype_t* tensor,
                         int item,
                         bool valid,
                         int image,
                         int col0,
                         int chan0)
        : pars_(pars)
        // An idle partition aims its columns at the end of the image.
        //
        // The bounds test the halo already carries then forces every read out of bounds, at no
        // cost in the loop. Clamping to a valid item would add that item's gradient twice, and
        // a validity test would add a compare per load.
        , col0_(valid ? col0 : pars.cols)
        , chan0_(chan0)
        , item_base_(item * item_rounds)
    {
        const int img_elems = pars.rows * pars.cols * pars.chans;

        // Fold the first image's origin into the 64-bit base, and stop the window at the last.
        //
        // The buffer offsets then span the item's own images, so the eight-channel over-read at
        // the last column stays inside them and reads zero. An idle partition takes image 0 and
        // no window.
        //
        // The window is also the whole batch-tail guard. A batch the packing does not divide
        // leaves the last item naming images past the end, whose addresses run past this bound
        // and read zero exactly as an out-of-image column does. Testing for them in the loop
        // would put a live count and a compare on every load.
        //
        // Without the unfold an item owns exactly the one image it names, so the clamp is a
        // constant and drops out.
        int window_images = 1;
        if constexpr(Layout::n_unfold > 1)
        {
            const int left = valid ? pars.images - image : 0;
            window_images  = left < Layout::n_unfold ? left : Layout::n_unfold;
        }

        const datatype_t* base = tensor + static_cast<size_t>(valid ? image : 0) * img_elems;
        const int window_bytes = window_images * img_elems * elem_bytes;
        rsrc_                  = __builtin_amdgcn_make_buffer_rsrc(
            const_cast<datatype_t*>(base), 0, window_bytes, rsrc_data_format);
        oob_bytes_ = window_bytes;

        img_stride_bytes_ = img_elems * elem_bytes;
        row_stride_bytes_ = pars.cols * pars.chans * elem_bytes;
    }

    // Load tensor row `row` of this loader's item into the LDS row buffer at `dest`.
    //
    // `row` is a global row index and may fall outside the image, in which case the whole
    // buffer reads zero. `load_wave` is the wave's index within this item's NumWaves waves.
    // Every wave must call this, and each issues loads_per_row buffer loads. `dest` is a whole
    // LDS object; the caller selects the ring slot by passing a different one.
    __device__ void load(int load_wave, int row, datatype_t* dest) const
    {
        const bool row_in_image = (0 <= row) && (row < pars_.rows);

        // soffset bypasses the buffer range check, so an out-of-image row zeroes it and leans
        // on the out-of-bounds voffset.
        const int soffset =
            row_in_image ? __builtin_amdgcn_readfirstlane(row * row_stride_bytes_) : 0;

        const auto vsoffset = [&](const SourceElem& src) -> std::array<int, 2> {
            const int col = col0_ + src.col;
            // A slot past the row's live columns is padding, forced out of bounds to read zero.
            const bool in_bounds = row_in_image && src.live && (0 <= col) && (col < pars_.cols);
            const int voffset    = in_bounds ? src.image * img_stride_bytes_ +
                                                (col * pars_.chans + chan0_ + src.chan) * elem_bytes
                                             : oob_bytes_;
            return {voffset, soffset};
        };

        // Each wave takes a contiguous block of its item's rounds.
        //
        // `load_wave` is wave-uniform already, and a redundant readfirstlane on it measured
        // 1.07x slower on n128_c512_k512_h32_w32_kh3_kw3: it lands a v_readfirstlane the
        // scheduler cannot move off the chain the loads issue from, mid memory phase.
        const int lane  = bunnies::lane_id();
        const int first = loads_per_row * load_wave;

#pragma unroll
        for(int i = 0; i < loads_per_row; ++i)
        {
            // A round past the row reads zero and drains into the shared slot.
            //
            // Such a round keeps its own index, so its columns fall past live_cols; only its
            // destination is clamped. Clamping keeps every load naming one LDS object. A drain
            // buffer of its own takes a branch, and merging the two paths' counts tightened the
            // waits ahead of the transpose reads from vmcnt(2) to vmcnt(1), a row of prefetch,
            // and measured 1.10x slower.
            const int round = first + i;
            const int slot  = has_drain && round >= item_rounds ? total_rounds : item_base_ + round;

            // A lane's 16 B is two adjacent uint2 slots.
            //
            // The rotation XORs only bits at or above 2 of the channel group, so the pair
            // always decodes to eight contiguous channels starting on an eight-channel
            // boundary, and a narrower lane stays inside one of those groups. See
            // docs/algorithms/direct/direct-wgrad-cdna4-lds-swizzle.md.
            const auto [v_off, s_off] = vsoffset(decode(round * round_elems + lane * lane_elems));
            load_inst::load(rsrc_, dest + slot * round_elems, v_off, s_off);
        }
    }

private:
    // The source element an LDS slot reads: which packed image, which column of it, and
    // which channel. `live` is false for a slot past the row's live columns.
    struct SourceElem
    {
        int image;
        int col;
        int chan;
        bool live;
    };

    // Which source element belongs in the tile's element slot `elem`.
    //
    // The swizzle permutes uint2 groups, so a 4 B half-group keeps its position within the
    // group: recover the group, then the half. See direct-wgrad-cdna4-lds-swizzle.md.
    //
    // The one division by the per-image stride lives here, and its quotient serves three
    // callers at once: the image the slot reads from, the column within that image, and the
    // packed column the rotation keys on. Without an unfold there is no division at all.
    __device__ __host__ static constexpr SourceElem decode(int elem)
    {
        const int uint2_slot = elem / 4;
        const int half       = elem / 2 % 2;
        const int x          = Swizzle::x(uint2_slot);

        if constexpr(Layout::n_unfold == 1)
        {
            const int chan = 4 * Swizzle::c4_at(uint2_slot, Swizzle::pack(x)) + 2 * half;
            return {0, x, chan, x < Layout::live_cols};
        }
        else
        {
            const int image = x / Layout::w_per_image;
            const int col   = x - image * Layout::w_per_image;
            // pack(), without the division it has already been given.
            const int packed = image * Layout::w_unfold + col;
            const int chan   = 4 * Swizzle::c4_at(uint2_slot, packed) + 2 * half;
            // The same test as x < live_cols, every image occupying w_per_image columns.
            return {image, col, chan, image < Layout::n_unfold};
        }
    }

    RowTensorPars pars_;
    __amdgpu_buffer_rsrc_t rsrc_;
    int col0_;
    int chan0_;
    int item_base_; // first round of this item's row within the ring slot
    int oob_bytes_;
    int img_stride_bytes_;
    int row_stride_bytes_;
};

} // namespace hipconv::cdna4::direct_wgrad

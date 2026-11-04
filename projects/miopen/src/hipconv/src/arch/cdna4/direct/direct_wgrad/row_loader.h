#pragma once

// Global-to-LDS loader for one tensor row of the direct_wgrad main loop.
//
// One class serves both operands: S rows (W x C, NHWC) and delta rows (Q x K, NPQK). The tile
// shape is the only difference between them, and Layout carries it.

#include <type_traits>
#include "bunnies.hpp"
#include "bunnies_cdna4.hpp"
#include "lds_layout.h"
#include "mathutil.h"
#include "packed_ops.h"
#include "types.h"

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
// Tiled folds the tile's first row into the 64-bit base so the 32-bit soffset spans one tile
// rather than one image, which is what lets an image past 2 GiB be addressed at all. It is a
// template parameter rather than a runtime origin of zero so the untiled path keeps the
// addressing it already had, byte for byte. See RowSchedule's row-tile section.
//
// Tf32 selects the second of the two transports this class carries:
//
//   16-bit (Tf32 false): buffer_load_lds DMAs straight into LDS. Lane L lands at slot L, so the
//     per-lane fetch address is the swizzle inverse of that slot and the row arrives already
//     swizzled, with no register traffic and no ds_write at all.
//
//   tf32 (Tf32 true): buffer_load_b128 lands fp32 in registers, which `publish` splits into the
//     (big, small) bf16 pair and ds_write_b64s into the two LDS planes. The DMA cannot convert,
//     so the data has to pass through VGPRs; the swizzle then moves to the write address, which
//     the DMA could not have chosen. The caller stages a row and converts it one iteration
//     later, so the split overlaps a compute phase rather than stalling on the fetch.
//
// Both take the same calls; on the DMA path `publish` and `publish_fence` are no-ops.
template <typename Layout,
          typename datatype_t,
          int NumWaves,
          int Items     = 1,
          int LaneBytes = 16,
          bool Tiled    = false,
          bool Tf32     = false>
class RowLoader
{
    static constexpr int rsrc_data_format = 1 << 15;

public:
    // datatype_t is the LDS element, which stays 16-bit for tf32: what it holds is one plane of
    // the split pair.
    static constexpr bool is_tf32 = Tf32;
    static_assert(sizeof(datatype_t) == 2);
    static_assert(Layout::planes == (is_tf32 ? 2 : 1),
                  "the row layout's plane count has to follow the transport: tf32 writes two");

    using Swizzle = typename Layout::Swizzle;

    // Element type in global memory, which is what the buffer loads read and address in.
    using src_t = std::conditional_t<is_tf32, fp32_t, datatype_t>;

    static constexpr int wave_size      = arch::wave_size;
    static constexpr int src_elem_bytes = static_cast<int>(sizeof(src_t));

    // Elements in one plane of an item's tile row, and the rounds they split into.
    //
    // A round is one wave's buffer load of wave_size * lane_bytes of source. At 16 bits that is
    // the same count of LDS elements; at tf32's 32 bits it is half as many per plane, so a row
    // takes twice the rounds and the column granularity halves with it. The row pads to a whole
    // round.
    static constexpr int tile_elems  = Layout::size_elems;
    static constexpr int lane_bytes  = LaneBytes;
    static constexpr int lane_elems  = lane_bytes / src_elem_bytes;
    static constexpr int round_elems = wave_size * lane_bytes / src_elem_bytes;
    static constexpr int item_rounds = tile_elems / round_elems;

    // A tf32 lane's source is four fp32 of one column, which is one uint2 of each plane, so a
    // round enumerates wave_size consecutive uint2 of the unswizzled row. Everything the write
    // side does rests on that, so it is checked rather than assumed.
    static_assert(!is_tf32 || lane_elems == 4,
                  "the tf32 write side places one uint2 per lane, which needs a 16-byte load of "
                  "4-byte elements");

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

    // The drain belongs to the DMA path alone.
    //
    // There a surplus round still has to name an LDS destination, so it names a round nothing
    // reads. A tf32 load has no LDS destination to name -- it lands in a register -- and its
    // surplus rounds simply skip the write, so they need no room.
    static constexpr bool needs_drain = has_drain && !is_tf32;

    // Elements in one ring slot: every item's planes and rows, then the drain if there is one.
    static constexpr int buffer_elems =
        Layout::planes * total_rounds * round_elems + (needs_drain ? round_elems : 0);

    using load_inst = arch::buffer_load_lds<lane_bytes>;

    // Registers one row's loads occupy while they travel to LDS.
    //
    // tf32 only; the DMA path needs none and carries a single dead slot, a zero-length array
    // being ill-formed, which the compiler drops because nothing on that path touches it.
    static constexpr int stage_loads = is_tf32 ? loads_per_row : 1;
    using Stage                      = uint4[stage_loads];

    // Staging for Rows rows, row R in slot R % Rows; one dead slot on the DMA path.
    template <int Rows>
    struct StageRing
    {
        static constexpr int slots = is_tf32 ? maximum(1, Rows) : 1;
        Stage rows[slots];

        template <int Slot>
        __device__ Stage& slot()
        {
            return rows[slots == 1 ? 0 : Slot % slots];
        }
    };

    // `item` names both the wave's spatial partition and the sub-buffer its rows land in.
    //
    // `valid` is false for a partition whose item ran past the end of the segment; `image` is
    // the first of the Layout::n_unfold images the item packs.
    __device__ RowLoader(const RowTensorPars& pars,
                         const src_t* tensor,
                         int item,
                         bool valid,
                         int image,
                         int col0,
                         int chan0,
                         int row_origin  = 0,
                         int window_rows = 0)
        : pars_(pars)
        // An idle partition aims its columns at the end of the image.
        //
        // The bounds test the halo already carries then forces every read out of bounds, at no
        // cost in the loop. Clamping to a valid item would add that item's gradient twice, and
        // a validity test would add a compare per load.
        , col0_(valid ? col0 : pars.cols)
        , chan0_(chan0)
        , item_base_(item * item_rounds)
        , item_uint2_base_(item * Layout::planes * Layout::size_uint2)
    {
        if constexpr(Tiled)
        {
            row_stride_bytes_ = pars.cols * pars.chans * src_elem_bytes;
            // Both origins fold into the 64-bit base, the window covering the tile's rows plus
            // the lookahead its last iterations issue, so a load past the tile still resolves
            // here and the boundary needs no handling.
            //
            // img_stride_bytes_ stays 0: a tiled config runs at unfold_n 1, so src.image is
            // always zero and the int that term would overflow leaves the address.
            const size_t row_elems   = static_cast<size_t>(pars.cols) * pars.chans;
            const size_t img_elems64 = static_cast<size_t>(pars.rows) * row_elems;
            const src_t* image_base = tensor + static_cast<size_t>(valid ? image : 0) * img_elems64;
            tile_                   = {row_origin, image_base, valid};
            const src_t* base       = image_base + static_cast<size_t>(row_origin) * row_elems;
            const int window_bytes  = valid ? tile_window_bytes(pars, row_origin, window_rows) : 0;
            rsrc_                   = __builtin_amdgcn_make_buffer_rsrc(
                const_cast<src_t*>(base), 0, window_bytes, rsrc_data_format);
            oob_bytes_        = window_bytes;
            img_stride_bytes_ = 0;
            return;
        }

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

        const src_t* base      = tensor + static_cast<size_t>(valid ? image : 0) * img_elems;
        const int window_bytes = window_images * img_elems * src_elem_bytes;
        rsrc_                  = __builtin_amdgcn_make_buffer_rsrc(
            const_cast<src_t*>(base), 0, window_bytes, rsrc_data_format);
        oob_bytes_ = window_bytes;

        img_stride_bytes_ = img_elems * src_elem_bytes;
        row_stride_bytes_ = pars.cols * pars.chans * src_elem_bytes;
    }

    // The same loader with its origin moved to `row_origin`, for the next row tile.
    //
    // Only the base and the window move; the column, channel and item state is the tile's own
    // and carries over. Tiled only -- an untiled loader spans the whole image already.
    __device__ RowLoader rebased(int row_origin, int window_rows) const
    {
        static_assert(Tiled, "an untiled loader spans the whole image and has no tile to rebase");
        RowLoader next         = *this;
        next.tile_.row_origin  = row_origin;
        const size_t row_elems = static_cast<size_t>(pars_.cols) * pars_.chans;
        const src_t* base      = tile_.image_base + static_cast<size_t>(row_origin) * row_elems;
        const int window_bytes =
            tile_.valid ? tile_window_bytes(pars_, row_origin, window_rows) : 0;
        next.rsrc_ = __builtin_amdgcn_make_buffer_rsrc(
            const_cast<src_t*>(base), 0, window_bytes, rsrc_data_format);
        next.oob_bytes_ = window_bytes;
        return next;
    }

    // Bytes a tile's window spans, stopping at the end of the image.
    //
    // The clamp is not tidiness. The window is the only thing bounding the over-read a wave
    // makes past a row when its channel origin runs past the tensor's channel count, which a
    // padded block does on every layer. Untiled, the window is the image and catches it; a
    // tile window reaching past the last row would let that read touch unmapped memory, which
    // faults rather than returning zero.
    __device__ int tile_window_bytes(const RowTensorPars& pars, int row_origin, int rows) const
    {
        const int left = pars.rows - row_origin;
        return (rows < left ? rows : left) * row_stride_bytes_;
    }

    // The row index this loader's base measures from: the tensor row when untiled, its offset
    // into the tile when not.
    __device__ int rebase_row(int row) const
    {
        if constexpr(Tiled)
            return row - tile_.row_origin;
        else
            return row;
    }

    // Issue tensor row `row` of this loader's item.
    //
    // `row` is a global row index and may fall outside the image, in which case the whole
    // buffer reads zero. `load_wave` is the wave's index within this item's NumWaves waves.
    // Every wave must call this, and each issues loads_per_row buffer loads.
    //
    // 16-bit: the loads land directly in the LDS row buffer `dest`, which is a whole LDS object;
    //         the caller selects the ring slot by passing a different one. `stage` is the dead
    //         one-slot array and goes untouched.
    // tf32:   they land in `stage`, which the caller hands to publish one iteration later, and
    //         `dest` goes untouched.
    __device__ void load(int load_wave, int row, datatype_t* dest, Stage& stage) const
    {
        const bool row_in_image = (0 <= row) && (row < pars_.rows);

        // soffset bypasses the buffer range check, so an out-of-image row zeroes it and leans
        // on the out-of-bounds voffset.
        const int soffset =
            row_in_image ? __builtin_amdgcn_readfirstlane(rebase_row(row) * row_stride_bytes_) : 0;

        const auto vsoffset = [&](const SourceElem& src) -> std::array<int, 2> {
            const int col = col0_ + src.col;
            // A slot past the row's live columns is padding, forced out of bounds to read zero.
            const bool in_bounds = row_in_image && src.live && (0 <= col) && (col < pars_.cols);
            const int voffset    = in_bounds
                                       ? src.image * img_stride_bytes_ +
                                          (col * pars_.chans + chan0_ + src.chan) * src_elem_bytes
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
            const int round = first + i;

            if constexpr(is_tf32)
            {
                // A surplus round reads out of bounds and is dropped on the write side.
                //
                // It still issues, because the vmcnt immediate the main loop waits on is one
                // count for the whole workgroup, and it needs no destination of its own the way
                // the DMA path's does: nothing has been written yet.
                const auto [v_off, s_off] = vsoffset(stage_slot(round, lane).src);
                stage[i]                  = __builtin_bit_cast(
                    uint4, __builtin_amdgcn_raw_buffer_load_b128(rsrc_, v_off, s_off, 0));
            }
            else
            {
                // A round past the row reads zero and drains into the shared slot.
                //
                // Such a round keeps its own index, so its columns fall past live_cols; only its
                // destination is clamped. Clamping keeps every load naming one LDS object. A
                // drain buffer of its own takes a branch, and merging the two paths' counts
                // tightened the waits ahead of the transpose reads from vmcnt(2) to vmcnt(1), a
                // row of prefetch, and measured 1.10x slower.
                const int slot =
                    has_drain && round >= item_rounds ? total_rounds : item_base_ + round;

                // A lane's 16 B is two adjacent uint2 slots.
                //
                // The rotation XORs only bits at or above 2 of the channel group, so the pair
                // always decodes to eight contiguous channels starting on an eight-channel
                // boundary, and a narrower lane stays inside one of those groups. See
                // docs/algorithms/direct/direct-wgrad-cdna4-lds-swizzle.md.
                const auto [v_off, s_off] =
                    vsoffset(decode(round * round_elems + lane * lane_elems));
                load_inst::load(rsrc_, dest + slot * round_elems, v_off, s_off);
            }
        }
    }

    // The DMA path's call, whose loads need no staging.
    __device__ void load(int load_wave, int row, datatype_t* dest) const
    {
        static_assert(!is_tf32,
                      "the tf32 path stages its loads; call load(wave, row, dest, stage)");
        Stage unused;
        load(load_wave, row, dest, unused);
    }

    // Split a staged row into its (big, small) bf16 pair and write both LDS planes of `dest`.
    //
    // A no-op on the DMA path. The caller must have drained this row's loads, which the main
    // loop's own vmcnt already does, so the split costs no wait of its own. `dest` is the ring
    // slot, as for load().
    //
    // Only live slots are written. A padding column is never read back -- the layout pads the
    // row to a whole round and the compute phase stops at live_cols -- and a surplus round has
    // no slot at all, so neither needs the zero the DMA path happens to leave there.
    __device__ void publish(int load_wave, const Stage& stage, datatype_t* dest) const
    {
        if constexpr(is_tf32)
        {
            // uint2 view: a lane's four fp32 are one uint2 in each plane.
            auto* lds_u2    = reinterpret_cast<bf16x4_t*>(dest);
            const int lane  = bunnies::lane_id();
            const int first = loads_per_row * load_wave;

#pragma unroll
            for(int i = 0; i < loads_per_row; ++i)
            {
                const StageSlot slot = stage_slot(first + i, lane);
                if(!slot.src.live)
                    continue;

                const auto pair =
                    fp32xN_to_bf16_pair<4>(__builtin_bit_cast(packed_vec_t<fp32_t, 4>, stage[i]));

                const int off                    = item_uint2_base_ + slot.write_uint2;
                lds_u2[off]                      = pair.big;
                lds_u2[off + Layout::size_uint2] = pair.small;
            }
        }
        else
        {
            (void)load_wave;
            (void)stage;
            (void)dest;
        }
    }

    // Drain this wave's `publish` writes ahead of the barrier, which drains nothing itself. One
    // call covers every loader's writes; a no-op on the DMA path.
    __device__ static void publish_fence()
    {
        if constexpr(is_tf32)
            arch::s_wait_lgkmcnt<0>();
    }

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

    // Where a tf32 lane's load of `round` comes from, and the uint2 of each plane it goes to.
    struct StageSlot
    {
        SourceElem src;
        int write_uint2; // within one plane of this item's row
    };

    // The tf32 path's decode, which runs the opposite way from decode()'s.
    //
    // The DMA lands lane L at slot L whatever it holds, so decode() inverts the swizzle to find
    // the source. A tf32 lane picks its own destination instead, so it takes the row in plain
    // (column, channel) order and applies the swizzle to place it: a lane's four fp32 are one
    // uint2 of each plane, so a round is wave_size consecutive uint2 of the unswizzled row, and
    // slot u is column u / C4 and channel group u % C4.
    //
    // A surplus round runs off the end of the row, which puts its column past the live ones:
    // the fetch reads out of bounds and the write is dropped, exactly as a padding column's is.
    __device__ __host__ static constexpr StageSlot stage_slot(int round, int lane)
    {
        constexpr int C4 = Swizzle::C4;

        const int u  = round * wave_size + lane;
        const int x  = u / C4;
        const int c4 = u - x * C4;

        const int chan = 4 * c4;
        const int off  = Swizzle::offset_uint2(x, c4);

        if constexpr(Layout::n_unfold == 1)
            return {{0, x, chan, x < Layout::live_cols}, off};
        else
        {
            const int image = x / Layout::w_per_image;
            const int col   = x - image * Layout::w_per_image;
            return {{image, col, chan, image < Layout::n_unfold}, off};
        }
    }

private:
    RowTensorPars pars_;
    __amdgpu_buffer_rsrc_t rsrc_;
    int col0_;
    int chan0_;
    int item_base_;       // first round of this item's row within the ring slot, for the DMA path
    int item_uint2_base_; // the same origin in uint2, planes included, for the tf32 write
    int oob_bytes_;
    int img_stride_bytes_;
    int row_stride_bytes_;
    // The tile origin this loader's base sits on, subtracted from every row it loads, plus what
    // rebased() needs to build the next tile's descriptor.
    //
    // Empty and zero-sized when untiled: carrying it there grew the object enough to move the
    // register allocation of 4x4's C(128) x K(32) tile, which already spills.
    struct TileState
    {
        int row_origin          = 0;
        const src_t* image_base = nullptr;
        bool valid              = false;
    };
    struct NoTileState
    {
    };
    [[no_unique_address]] std::conditional_t<Tiled, TileState, NoTileState> tile_;
};

} // namespace hipconv::cdna4::direct_wgrad

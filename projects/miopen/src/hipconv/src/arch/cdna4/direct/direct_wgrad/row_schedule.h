#pragma once

// Which tensor rows the prologue and the main loop touch, and where each one lands.
//
// Integer arithmetic on the layer's shape with no device code, so the host test can check the
// schedule without running a kernel. See docs/algorithms/direct/direct-wgrad.md, "Prologue and
// Epilogue", for the padding offset between the S and delta streams and the zero row counts.

#include "mathutil.h"

namespace hipconv::cdna4::direct_wgrad
{

struct RowSchedule
{
    // Filter height, and the depth of the delta register ring.
    int kh;

    // Rows the memory phase runs ahead of the row its compute phase consumes.
    int prefetch_rows;

    int pad_h;
    int s_rows;
    int row_buffers;

    // ---- prologue ----

    // Delta row that register-ring slot j takes, for j in [0, kh - 1).
    //
    // Negative for the slots above the top of the image, which the prologue zeroes instead of
    // loading. The main loop fills the ring's remaining slot itself.
    constexpr int prologue_delta_row(int j) const { return pad_h - kh + 1 + j; }

    // Split of those kh - 1 slots into zeroed and loaded.
    // kh bounds both counts: once pad_h reaches kh - 1 the ring needs no zero rows.
    constexpr int zero_slots() const { return maximum(0, kh - 1 - pad_h); }
    constexpr int read_slots() const { return minimum(kh - 1, pad_h); }

    // First delta row any load touches.
    // Rows below it see only padding and contribute nothing to the gradient.
    constexpr int delta_first() const { return maximum(0, pad_h - kh + 1); }

    // Delta rows the prologue issues to LDS, starting at delta_first().
    //
    // Covers the rows the register ring reads plus the prefetch the first memory phase expects
    // to find already in flight.
    constexpr int delta_prologue_rows() const { return read_slots() + prefetch_rows; }

    // ---- main loop ----

    // Trip count: one row of S per iteration.
    // A further iteration would read below the bottom of the image and spend its MFMAs on zeros.
    constexpr int iterations() const { return s_rows; }

    // Rows iteration `iter` consumes, and the rows it issues prefetch_rows ahead.
    //
    // Delta leads S by pad_h rows: S row h pairs with the kh delta rows ending at h + pad_h.
    constexpr int s_row(int iter) const { return iter; }
    constexpr int delta_row(int iter) const { return iter + pad_h; }
    constexpr int s_issue_row(int iter) const { return iter + prefetch_rows; }
    constexpr int delta_issue_row(int iter) const { return iter + pad_h + prefetch_rows; }

    // ---- LDS rings ----
    //
    // Slots are keyed on the iteration, so the unroll folds every slot index to a compile-time
    // constant. Keying on the tensor row would not fold: the delta row is iter + pad_h, and
    // pad_h is a runtime value.

    constexpr int lds_slot(int iter) const { return iter % row_buffers; }
    constexpr int lds_issue_slot(int iter) const { return (iter + prefetch_rows) % row_buffers; }

    // ---- delta register ring ----
    //
    // Slots are numbered relative to the iteration, so unrolling the main loop by kh makes every
    // slot index a compile-time constant and the register addressing folds away. The prologue
    // fills slots 0 .. kh-2 in order.

    // Slot iteration `iter` writes its freshly read row into.
    constexpr int reg_slot_written(int iter) const { return (iter + kh - 1) % kh; }

    // Slot holding window position j of iteration `iter`, oldest row first.
    // Position j carries delta row delta_row(iter) - (kh - 1) + j, pairing with filter row
    // kh - 1 - j.
    constexpr int reg_slot(int iter, int j) const { return (iter + j) % kh; }

    // ---- row tiles ----
    //
    // A loader addresses its rows with a 32-bit soffset of row * row_stride, which overflows
    // once an image passes 2 GiB. A tiled config folds the tile's origin into the 64-bit base
    // instead. Zero is untiled, which every config that does not need this keeps.
    //
    // Both loaders take the same origin, so neither carries a negative offset: delta leads S by
    // pad_h, and the prologue runs only in tile 0.
    int rows_per_tile = 0;

    constexpr bool tiled() const { return rows_per_tile > 0; }

    // Tiles the main loop runs, and the iteration and origin each starts at.
    constexpr int tiles() const
    {
        return tiled() ? (iterations() + rows_per_tile - 1) / rows_per_tile : 1;
    }
    constexpr int tile_first_iter(int tile) const { return tiled() ? tile * rows_per_tile : 0; }
    constexpr int tile_origin(int tile) const { return tile_first_iter(tile); }

    // Iterations in a tile, which the last one may cut short.
    constexpr int tile_iterations(int tile) const
    {
        if(!tiled())
            return iterations();
        const int left = iterations() - tile_first_iter(tile);
        return left < rows_per_tile ? left : rows_per_tile;
    }

    // Rows each loader reaches from the tile origin, which is what its window has to span.
    //
    // A tile issues prefetch_rows past its last iteration, and those loads resolve through the
    // tile's own base rather than the next one's, so the window carries the lookahead. Delta
    // additionally leads S by pad_h.
    constexpr int s_tile_rows() const { return rows_per_tile + prefetch_rows; }
    constexpr int delta_tile_rows() const { return rows_per_tile + pad_h + prefetch_rows; }
};

} // namespace hipconv::cdna4::direct_wgrad

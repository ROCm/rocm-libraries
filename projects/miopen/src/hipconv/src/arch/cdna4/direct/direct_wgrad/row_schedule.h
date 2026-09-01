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
};

} // namespace hipconv::cdna4::direct_wgrad

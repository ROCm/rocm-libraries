#pragma once

// The parameters one patch_embed kernel is compiled for.
//
// The family serves non-overlapping patch convolutions (stride == filter, no padding),
// which are a GEMM over a permuted input and no data duplication. See
// docs/algorithms/direct/patch-embed-cdna4.md.

namespace hipconv::cdna4::patch_embed
{
static constexpr int WAVE_SIZE = 64;

// The MFMA the compute phase tiles onto: 16x16x32.
//
// M is the spatial axis (n, p, q flattened), N the output-channel axis, and K the
// filter-times-channel reduction kh*kw*c.
constexpr int MFMA_M = 16;
constexpr int MFMA_N = 16;
constexpr int MFMA_K = 32;

// Elements a 16-byte load moves, which is the granule both LDS tiles are swizzled on.
constexpr int GRANULE = 8;

struct Config
{
    int m_tile16; // spatial rows per workgroup / 16
    int n_tile16; // output channels per workgroup / 16
    int k_chunk;  // K elements staged in LDS per pass
    int waves_m;  // waves along M
    int waves_n;  // waves along N
    // Whether the K axis is padded out to a whole granule per filter row.
    //
    // A filter row is kw*c elements and the staging granule is eight, so a patch whose row
    // is not a multiple of eight -- 14x14 over 3 channels gives 42 -- has one granule per
    // row straddling into the next row, which is not contiguous on the A side. Padding the
    // row to 48 keeps every granule whole and costs the MFMA the difference.
    bool row_pad = false;

    constexpr int m_tile() const { return MFMA_M * m_tile16; }
    constexpr int n_tile() const { return MFMA_N * n_tile16; }
    constexpr int waves() const { return waves_m * waves_n; }
    constexpr int threads() const { return WAVE_SIZE * waves(); }

    // The MFMA tiles one wave owns.
    constexpr int wave_m16() const { return m_tile16 / waves_m; }
    constexpr int wave_n16() const { return n_tile16 / waves_n; }

    // 16-byte granules per LDS row, and the XOR swizzle's period.
    constexpr int granules() const { return k_chunk / GRANULE; }

    // The K span one filter row occupies in LDS, given the shape's own kw*c.
    constexpr int padded_row(int patch_row) const
    {
        return row_pad ? (patch_row + GRANULE - 1) / GRANULE * GRANULE : patch_row;
    }

    // MFMA K-steps one staged chunk feeds.
    constexpr int k_steps() const { return k_chunk / MFMA_K; }

    // The epilogue stages the output through the larger of the two operand stages, which
    // the widest tiles outgrow; drains() is the number of parts it goes out in.
    constexpr int out_row() const { return n_tile() + GRANULE; }
    constexpr int stage_cap() const
    {
        return (n_tile() > m_tile() ? n_tile() : m_tile()) * k_chunk;
    }
    constexpr int drains() const
    {
        // A part is a whole number of a wave's MFMA row blocks, so the split has to divide
        // wave_m16; 0 means no split does and the config is not buildable.
        for(int d = 1; d <= wave_m16(); ++d)
            if(wave_m16() % d == 0 && (m_tile() / d) * out_row() <= stage_cap())
                return d;
        return 0;
    }

    constexpr int acc_vgprs() const { return wave_m16() * wave_n16() * 4; }
    // Both tiles are double-buffered; see the staging-depth note in kernel.h.
    constexpr int lds_bytes() const { return 2 * (m_tile() + n_tile()) * k_chunk * 2; }

    // 16-byte pieces each tile needs staged, and the passes that takes.
    constexpr int a_pieces() const { return m_tile() * granules(); }
    constexpr int b_pieces() const { return n_tile() * granules(); }
    constexpr int a_rounds() const { return a_pieces() / threads(); }
    constexpr int b_rounds() const { return b_pieces() / threads(); }

    // Whether a tile `width` elements wide can be read with ds_read_b64_tr_b16.
    //
    // The rotation that keeps that read off one bank needs the width to divide into a
    // power-of-two number of four-element groups, and into at least eight of them.
    static constexpr bool transposable(int width)
    {
        const int groups = width / 4;
        return groups >= 8 && (groups & (groups - 1)) == 0;
    }

    // Dgrad reads its filter tile transposed; wgrad reads both tiles so.
    constexpr bool serves_dgrad() const { return transposable(n_tile()); }
    constexpr bool serves_wgrad() const { return transposable(m_tile()) && transposable(n_tile()); }

    constexpr bool operator==(const Config&) const = default;
};

} // namespace hipconv::cdna4::patch_embed

#pragma once

#include "matrix_layout.h"

#include <array>
#include <iomanip>
#include <iostream>

/// Print a grid showing (lane, element_idx) for each element of an M x K matrix.
/// element_idx is the flat index within a lane: reg_idx * items_per_register + sub.
template <int M, int K, int B, typename T>
void print_layout_grid()
{
    using Layout           = MatrixLayout<M, K, B, T>;
    constexpr int IPR      = Layout::items_per_register();
    constexpr int NUM_REGS = (M * K) / (64 * IPR);

    // Reverse map: grid[row][col] = {lane, flat element index within lane}
    std::array<std::array<std::pair<int, int>, K>, M> grid{};

    for(int lane = 0; lane < 64; ++lane)
    {
        for(int idx = 0; idx < NUM_REGS; ++idx)
        {
            int outer_idx  = Layout::outer(lane);
            int inner_base = Layout::inner(lane, idx);
            for(int sub = 0; sub < IPR; ++sub)
            {
                int col              = inner_base + sub;
                grid[outer_idx][col] = {lane, idx * IPR + sub};
            }
        }
    }

    // Field widths
    constexpr int lane_width = 2;
    constexpr int elem_width = 1;
    // Each cell: " (" + lane + "," + elem + ")" = 4 + lane_width + elem_width
    constexpr int cell_width = 4 + lane_width + elem_width;
    // Row label: "i=" + row + " |" (row is lane_width wide)
    constexpr int label_width = 2 + lane_width + 2;

    // Column header
    std::cout << std::string(label_width, ' ');
    for(int col = 0; col < K; ++col)
    {
        std::string hdr = "k=" + std::to_string(col);
        int trail       = cell_width - 1 - static_cast<int>(hdr.size());
        std::cout << " " << hdr << std::string(trail, ' ');
    }
    std::cout << "\n";

    for(int row = 0; row < M; ++row)
    {
        std::cout << "i=" << std::setw(lane_width) << row << " |";
        for(int col = 0; col < K; ++col)
        {
            auto [lane, elem_idx] = grid[row][col];
            std::cout << " (" << std::setw(lane_width) << lane << "," << elem_idx << ")";
        }
        std::cout << "\n";
    }
}

#pragma once

// The delta stream, folded into the step of the wave that drives it. That wave
// carries an accumulator like any other; this is what nobody else runs.

#include <hip/hip_runtime.h>

#include "config_table.h"
#include "lds.h"

namespace hipconv::cdna5::depthwise_wgrad_hankel
{

using namespace hipconv;

// Rows are one stride apart and this wave asks for them in order, so the address
// is carried in the role and stepped rather than linearised at each issue.
template <Config cfg, class ArmDeltaFn, class FireDeltaFn, class BatchDeltaFn>
struct DeltaIssuer
{
    static constexpr StageGeometry GEO = stage_geometry(cfg);
    static constexpr int DELTA_PF      = GEO.delta_pf;
    static constexpr int POOL          = GEO.delta_pool_rows;
    static_assert(DELTA_PF > 1, "a depth of one would make a step's retirement conditional");

    static constexpr int ROWS = GEO.delta_rows;

    ArmDeltaFn& arm_delta;
    FireDeltaFn& fire_delta;
    BatchDeltaFn& batch_delta;
    int y_begin;
    int y_end;
    int py;
    int ho;
    unsigned long long base;
    unsigned long long row_bytes;
    unsigned long long addr;
    int cur_row;
    bool armed = false;

    // The walk starts on the run's first existing row: where the window opens
    // above the image that is row 0, the backs crossing zero exactly.
    __device__ void open(unsigned long long base_addr, unsigned long long stride_bytes)
    {
        base            = base_addr;
        row_bytes       = stride_bytes;
        const int back0 = y_begin - (ROWS - 1) + py;
        cur_row         = back0 >= 0 ? back0 / cfg.stride : 0;
        addr            = base + (unsigned long long)cur_row * stride_bytes;
    }

    // The descriptor filled in for one row and nothing started. p is negative
    // where the image has no row above, and >= ho where it has none below.
    __device__ void arm(int p, int slot)
    {
        if(p >= 0 && p < ho)
        {
            if(p != cur_row)
            {
                addr += row_bytes;
                cur_row = p;
            }
            arm_delta(addr, slot);
        }
        else
        {
            arm_delta.template operator()<false>(base, slot);
        }
    }

    // One row out, where nothing separates the two halves.
    __device__ void row(int p, int slot)
    {
        arm(p, slot);
        fire_delta();
    }

    // Ages walk the pool upwards and wrap exactly once, on the youngest: the run
    // before the wrap is contiguous in LDS and, at stride 1, in the image too,
    // which is the shape iterate mode batches. Stride 2 has two ages share a row,
    // so its global increment is not fixed and it stays on the per-row path.
    static constexpr bool BATCH = cfg.stride == 1 && POOL >= ROWS && ROWS >= 3;

    // The initial window, every slot of it. An age's row is the row the step that
    // far back would have fetched, which at stride 2 two ages share.
    __device__ void window()
    {
        if constexpr(BATCH)
        {
            constexpr int RUN = ROWS - 1;
            constexpr int S0  = POOL - ROWS + 1;

            // Where the run meets the image: rows above it first, then the body,
            // then rows below. A batch has one existence flag for all its passes,
            // so the two edges are where it has to be cut.
            const int back0 = y_begin - RUN + py;
            const int lo    = min(max(-back0, 0), RUN);
            const int hi_   = min(max(ho - back0, 0), RUN);

            if(lo > 0)
            {
                batch_delta(0ull, (unsigned)lo);
                arm_delta.template operator()<false>(base, S0);
                fire_delta();
            }
            if(hi_ > lo)
            {
                // open() left the walk on row back0 + lo, the run's first real one.
                batch_delta(row_bytes, (unsigned)(hi_ - lo));
                arm_delta(addr, S0 + lo);
                fire_delta();
            }
            if(RUN > hi_)
            {
                batch_delta(0ull, (unsigned)(RUN - hi_));
                arm_delta.template operator()<false>(base, S0 + hi_);
                fire_delta();
            }
            batch_delta(0ull, 1u);

            // The wrap, and the row the steady walk carries on from.
            const int p_last = y_begin + py;
            if(p_last < ho)
            {
                cur_row = p_last;
                addr    = base + (unsigned long long)p_last * row_bytes;
                arm_delta(addr, 0);
            }
            else
            {
                arm_delta.template operator()<false>(base, 0);
            }
            fire_delta();
        }
        else
        {
            static_for<ROWS>([&]<int I>() {
                constexpr int AGE  = ROWS - 1 - I;
                constexpr int SLOT = (POOL - AGE) % POOL;
                const int back     = y_begin - AGE + py;
                row(back >= 0 ? back / cfg.stride : -1, SLOT);
            });
        }
        __builtin_amdgcn_s_wait_tensorcnt(0);
    }

    // The rows between the window and the first send. They go out once prime_b
    // has emptied the slots they land in, which is why window() has them not.
    __device__ void open_fetch()
    {
        static_for<DELTA_PF - 1>([&]<int I>() {
            const int y = y_begin + 1 + I;
            if(y < y_end)
                row((y + py) / cfg.stride, (y - y_begin) % POOL);
        });
    }

    // Before the step's barrier: this step's row is down. DELTA_PF - 1 sit behind
    // it at every steady step, which is why the bound is a constant.
    __device__ void wait_landed(int y) const
    {
        if(y + DELTA_PF - 1 < y_end)
            __builtin_amdgcn_s_wait_tensorcnt(DELTA_PF - 1);
        else
            __builtin_amdgcn_s_wait_tensorcnt(0);
    }

    // Inside the barrier, in the window its signal has already opened: which row
    // goes out and where it lands needs no pool row to be free yet.
    __device__ void prepare(int y)
    {
        const int fetch_y = y + DELTA_PF;
        armed             = fetch_y < y_end;
        if(armed)
        {
            arm((fetch_y + py) / cfg.stride, (fetch_y - y_begin) % POOL);
        }
    }

    // Below the barrier: the pool row this lands in is the one the step before
    // read, and the barrier is what says every wave is done with it.
    __device__ void send(int)
    {
        if(armed)
            fire_delta();
    }

    __device__ void drain() const { __builtin_amdgcn_s_wait_tensorcnt(0); }
};

} // namespace hipconv::cdna5::depthwise_wgrad_hankel

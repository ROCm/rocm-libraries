#pragma once

// The input stream, folded into the step of the wave that drives it. That wave
// carries an accumulator like any other; this is what nobody else runs.

#include <hip/hip_runtime.h>

#include "config_table.h"
#include "detail.h"
#include "lds.h"

namespace hipconv::cdna5::depthwise_wgrad_hankel
{

using namespace hipconv;

// Rows are one stride apart and this wave asks for them in order, so the address
// is carried in the role and stepped rather than linearised at each issue.
template <Config cfg, class ArmRowFn, class FireRowFn>
struct InputIssuer
{
    static constexpr int PF = cfg.prefetch_depth;

    ArmRowFn& arm_row;
    FireRowFn& fire_row;
    int y_begin;
    int y_end;
    unsigned long long addr;
    unsigned long long row_bytes;
    bool armed = false;

    // Called on the issuing wave alone, which is the whole point of it being here.
    __device__ void open(unsigned long long base, unsigned long long stride_bytes)
    {
        addr      = base;
        row_bytes = stride_bytes;
    }

    // One row out, and the walk on to the next.
    __device__ void row(int slot)
    {
        arm_row(addr, slot);
        fire_row();
        addr += row_bytes;
    }

    // The rows the loop has no earlier step to issue from: one fewer than the ring
    // is deep, the spare slot being what the first step issues into.
    __device__ void prime()
    {
        static_for<PF - 1>([&]<int I>() {
            if(y_begin + I < y_end)
                row(I);
        });
    }

    // Before the step's barrier, what it establishes being for everyone: this
    // step's row is down. PF - 2 sit behind it at every steady step.
    __device__ void wait_landed(int y) const
    {
        if(y + PF - 2 < y_end)
            __builtin_amdgcn_s_wait_tensorcnt(PF - 2);
        else
            __builtin_amdgcn_s_wait_tensorcnt(0);
    }

    // Inside the barrier, in the window its signal has already opened: which row
    // goes out and where it lands needs no slot to be free yet.
    __device__ void prepare(int y)
    {
        const int r = y + PF - 1;
        armed       = r < y_end;
        if(armed)
        {
            arm_row(addr, (r - y_begin) % PF);
            addr += row_bytes;
        }
    }

    // Below the barrier: the slot it writes is the one the step before read, and
    // the barrier is what says that read has retired.
    __device__ void send(int)
    {
        if(armed)
            fire_row();
    }

    // The engine has to be given back before the wave goes, or its last row lands
    // in LDS a later workgroup already owns.
    __device__ void drain() const { __builtin_amdgcn_s_wait_tensorcnt(0); }
};

} // namespace hipconv::cdna5::depthwise_wgrad_hankel

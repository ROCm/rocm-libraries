#pragma once

// A half-tile's load stream and its walk over an X tile's filter positions (Ping-pong schedule).
// Parenthesized section names refer to docs/algorithms/direct/direct-flex-cdna4.md.

#include "kernel_args.h"
#include "layer_pars.h"

#include <hip/hip_runtime.h>

namespace hipconv::cdna4::direct_flex
{

// A half-tile's load stream; loads past an X tile go on to the next, keeping vmcnt fixed.
struct Stream
{
    int soffset;         // chunk offset in bytes; for weights also the filter position's
    int pos;             // index into LayerPars::seq
    LayerPars::Seq next; // the entry of the position this stream starts next
    int in_row;          // the current position's in_row
    int voff[2];
    int flip; // TAB_BYTES where the X tile its loads target takes the other table buffers
};

// A walk over an X tile's positions seq[seq0, seq0 + taps), then the first again.
template <class T>
class StreamWalk
{
    using Args = KernelArgs<T::PHASED>;

public:
    __device__ __forceinline__ StreamWalk(const Args& args, const LayerPars& pars)
        : pars_(pars)
        , seq0_(T::PHASED ? args.seg().seq0 : 0)
        , taps_(T::PHASED ? args.seg().taps : pars.kh * pars.kw)
        , seq_end_(seq0_ + taps_)
        // With an odd number of filter tiles, consecutive X tiles start in alternate table buffers.
        , tile_flip_(((T::PHASED ? args.seg().n_ftiles : pars.n_ftiles) & 1) * TAB_BYTES)
    {
    }

    __device__ __forceinline__ int taps() const { return taps_; }

    __device__ __forceinline__ Stream start() const
    {
        return {0, seq0_ - 1, pars_.seq[seq0_], 0, {0, 0}, 0};
    }

    // The table buffer flip of the position st starts next, which may lie in the next X tile.
    __device__ __forceinline__ int next_flip(const Stream& st) const
    {
        return st.pos + 1 == seq_end_ ? st.flip ^ tile_flip_ : st.flip;
    }

    // Advances a stream a position, crossing to the next X tile past the last.
    __device__ __forceinline__ LayerPars::Seq advance(Stream& st, bool& crossed) const
    {
        const LayerPars::Seq e = st.next;
        crossed                = ++st.pos == seq_end_;
        if(crossed)
        {
            st.pos = seq0_;
            st.flip ^= tile_flip_;
        }
        st.next = pars_.seq[st.pos + 1];
        return e;
    }

private:
    const LayerPars& pars_;
    const int seq0_;
    const int taps_;
    const int seq_end_;
    const int tile_flip_;
};

} // namespace hipconv::cdna4::direct_flex

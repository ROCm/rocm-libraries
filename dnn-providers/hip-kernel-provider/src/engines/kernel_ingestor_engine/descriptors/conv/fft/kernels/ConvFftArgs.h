// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The argument block of the FFT convolution kernels (ConvFft.cpp). Shared verbatim by the host
// pack, which fills one block per launch, and the kernels, which HIPRTC compiles with this file
// as an embedded include: builtin types only, no standard headers.
#pragma once

/// One complex value of the frequency-domain workspace.
struct ConvFftComplex
{
    float x;
    float y;
};

/// Every launch of one FFT convolution reads this block; each launch uses its own fields.
///
/// The convolution runs as a cross-correlation of tiles: an output tile of oh x ow points reads
/// an input window of th x tw points (the compile-time transform size, th = 1 for 1-D), and
/// oh = th - spanH + 1 with spanH = (r - 1) * dilH + 1 (likewise w). Tiles cover the output
/// extent; tile p of the P = n * tilesH * tilesW tiles is (image p / (tilesH * tilesW), tile row,
/// tile column). Its window starts at (row * oh - padH, column * ow - padW) of the source; the
/// window reads only its first validH x validW points (th x tw for an activation window, oh x ow
/// for the zero-padded dy window of backward weights), zero outside the source.
///
/// The frequency-domain workspace, F = th * (tw / 2 + 1) bins, P tiles cut into chunks of `pc`
/// starting at `p0`:
///   data      [group][F][channel][pc]   (wsOut of a data transform)
///   filter    [group][F][a][b]          (fwd: a = input channel, b = output channel;
///                                        bwd: a = output channel, b = input channel)
///   product   [group][F][b][pc]         (forward / backward data)
///   wgrad     [group][F][kg][cg]        (backward weights, summed over every chunk)
struct ConvFftArgs
{
    const void* src; ///< data transform: the tensor whose windows are read
    const void* filt; ///< filter transform: w (k, c/g, r, s)
    void* dst; ///< inverse: y, dx or dw
    ConvFftComplex* wsOut; ///< transforms: the spectra written
    const ConvFftComplex* wsA; ///< product: the data spectra (contracted channel a)
    const ConvFftComplex* wsB; ///< product: filter spectra; wgrad: the dy spectra
    ConvFftComplex* wsY; ///< product / wgrad result; the inverse reads it
    long long srcStr[4]; ///< element strides of src: n, channel, h, w
    long long dstStr[4]; ///< element strides of dst: n, channel, h, w (dw: k, c, r, s)
    long long filtStr[4]; ///< element strides of w: k, c, r, s
    int n;
    int groups;
    int chans; ///< data transform: channels a group of src (the spectra's channel extent)
    int ca; ///< product: contracted channels a group; wgrad: cg (x channels a group)
    int cb; ///< product: output channels a group; wgrad: kg (dy channels a group)
    int srcH;
    int srcW;
    int dstH; ///< extent the tiles cover (output of the forward / backward-data conv, dy for wgrad)
    int dstW;
    int padH; ///< window origin offset
    int padW;
    int validH;
    int validW;
    int r;
    int s;
    int dilH;
    int dilW;
    int flip; ///< filter transform / wgrad inverse: reflect the taps
    int filtMode; ///< 0: forward (a = c, b = k); 1: backward data (a = k, b = c)
    int oh;
    int ow;
    int tilesH;
    int tilesW;
    int p0; ///< first tile of this chunk
    int pc; ///< tiles in this chunk (the spectra's tile extent)
    int pcGrid; ///< tiles of a full chunk: the launch grids are sized for it (the last chunk may
        ///< hold fewer, pc; blocks past pc do nothing)
    int accumulate; ///< wgrad: add this chunk's sums to wsY (0 for the first chunk)
    int parts; ///< depthwise wgrad: partitions of the tiles
    int tilesPerPart; ///< depthwise wgrad: tiles a partition (a multiple of the block's tiles)
    int src2H; ///< depthwise wgrad: dy extent (src2, read without padding)
    int src2W;
    int valid2H; ///< depthwise wgrad: points of a dy window read (the output tile)
    int valid2W;
    const void* src2; ///< depthwise wgrad: dy
    long long src2Str[4]; ///< element strides of src2: n, channel, h, w
};

/* **************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include <algorithm>
#include <vector>

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

/*
 * ===========================================================================
 *    BDSQR_ROTLOG records the operations that the host QR iteration of the
 *    hybrid BDSQR applies to the singular vectors on the device (sequences of
 *    rotations of adjacent rows or columns, as LASR; single rotations, sign
 *    changes and swaps) and applies them in chunks. Consecutive sequences on
 *    the same matrix and in the same direction are grouped (at most
 *    BDSQR_ROT_K of them), and each group is cut into windows of BDSQR_ROT_B
 *    pairs, staggered by one pair per sequence with the earlier sequences
 *    leading, so that the rotations of a window can be applied after those of
 *    the previous one. The product of the rotations of each window is
 *    accumulated into a small matrix (all the windows of a chunk in one
 *    launch, as they only depend on the rotations) and applied with a GEMM:
 *    U(:, w) = U(:, w) Q for the columns of U, and VT(w, :) = Q' VT(w, :) for
 *    the rows of VT and C. The results are those of the rotations applied one
 *    by one, up to rounding.
 *
 *    All the operations act on pairs of indices (j, j+1) of the rows of VT and
 *    C or of the columns of U, with x_j' = c x_j + s x_(j+1) and
 *    x_(j+1)' = c x_(j+1) - s x_j (the convention of LASR and ROT).
 * ===========================================================================
 */

#ifndef BDSQR_ROT_K
#define BDSQR_ROT_K 128
#endif
#ifndef BDSQR_ROT_B
#define BDSQR_ROT_B 128
#endif
// rotations recorded before a chunk is applied
#ifndef BDSQR_ROT_CHUNK
#define BDSQR_ROT_CHUNK (1 << 22)
#endif

// (errors are thrown, as in the hybrid storage)
#define BDSQR_ROTLOG_HIP(...)                                              \
    {                                                                      \
        hipError_t _e = (__VA_ARGS__);                                     \
        if(_e != hipSuccess)                                               \
            THROW_IF_ROCBLAS_ERROR(get_rocblas_status_for_hip_status(_e)); \
    }

struct bdsqr_rot_seq
{
    int j0, cnt;
    long off;
};
struct bdsqr_rot_grp
{
    int fwd, K, seq0, P0, b, T, w;
    long qoff;
};

/** BDSQR_ROT_ACCUM accumulates the product of the rotations of each window (one thread-block per
    window; a rotation mixes two entries of the same row, so each thread updates its own rows
    without synchronization). In the processing coordinate p (p = j for a forward group,
    p = n-2-j for a backward one), window t of a group takes the pairs of its sequence k in
    [P0 + t b + K-1-k, P0 + (t+1) b + K-1-k), and acts on the indices of
    [P0 + t b, P0 + t b + w), w = b + K (stored in increasing index order). **/
template <typename T, typename S>
ROCSOLVER_KERNEL void bdsqr_rot_accum(const int n,
                                      const bdsqr_rot_grp* grps,
                                      const int* win_grp,
                                      const int* win_t,
                                      const bdsqr_rot_seq* seqs,
                                      const S* cs,
                                      const S* sn,
                                      T* Q)
{
    const int wid = hipBlockIdx_x;
    const bdsqr_rot_grp gp = grps[win_grp[wid]];
    const int t = win_t[wid];
    const int K = gp.K, b = gp.b, w = gp.w, fwd = gp.fwd;
    T* Qt = Q + gp.qoff + size_t(t) * w * w;
    for(int idx = hipThreadIdx_x; idx < w * w; idx += hipBlockDim_x)
        Qt[idx] = (idx % w == idx / w) ? T(1) : T(0);
    __syncthreads();
    const int pw0 = gp.P0 + t * b;
    const int base = fwd ? pw0 : n - 1 - (pw0 + w - 1); // first index of the window
    for(int k = 0; k < K; k++)
    {
        const bdsqr_rot_seq q = seqs[gp.seq0 + k];
        const int plo = fwd ? q.j0 : n - 2 - (q.j0 + q.cnt - 1);
        const int phi = fwd ? q.j0 + q.cnt - 1 : n - 2 - q.j0;
        const int a = max(plo, pw0 + (K - 1 - k));
        const int z = min(phi, pw0 + b + (K - 1 - k) - 1);
        for(int p = a; p <= z; p++)
        {
            const int j = fwd ? p : n - 2 - p;
            const S c = cs[q.off + (j - q.j0)];
            const S s = sn[q.off + (j - q.j0)];
            const int lj = j - base;
            for(int r = hipThreadIdx_x; r < w; r += hipBlockDim_x)
            {
                const T xa = Qt[r + size_t(lj) * w];
                const T xb = Qt[r + size_t(lj + 1) * w];
                Qt[r + size_t(lj) * w] = c * xa + s * xb;
                Qt[r + size_t(lj + 1) * w] = c * xb - s * xa;
            }
        }
    }
}

/** BDSQR_ROT_APPLY_SEQ applies one sequence of rotations (pairs j0..j0+cnt-1, forward or backward) to
    the rows (ROWS) or columns of X; one thread per column (row) of X, L of them. **/
template <bool ROWS, typename T, typename S>
ROCSOLVER_KERNEL void bdsqr_rot_apply_seq(const int L,
                                          T* X,
                                          const int ldx,
                                          const int j0,
                                          const int cnt,
                                          const int fwd,
                                          const S* cs,
                                          const S* sn)
{
    const int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(i >= L)
        return;
    // element j of the line of X that this thread updates
    auto x = [&](int j) -> T& { return ROWS ? X[j + size_t(i) * ldx] : X[i + size_t(j) * ldx]; };
    if(fwd)
    {
        T a = x(j0);
        for(int k = 0; k < cnt; k++)
        {
            const T bb = x(j0 + k + 1);
            x(j0 + k) = cs[k] * a + sn[k] * bb;
            a = cs[k] * bb - sn[k] * a;
        }
        x(j0 + cnt) = a;
    }
    else
    {
        T bb = x(j0 + cnt);
        for(int k = cnt - 1; k >= 0; k--)
        {
            const T a = x(j0 + k);
            x(j0 + k + 1) = cs[k] * bb - sn[k] * a;
            bb = cs[k] * a + sn[k] * bb;
        }
        x(j0) = bb;
    }
}

/** BDSQR_ROT_NEGSWAP negates line j (k < 0) or swaps lines j and k of X. **/
template <bool ROWS, typename T>
ROCSOLVER_KERNEL void bdsqr_rot_negswap(const int L, T* X, const int ldx, const int j, const int k)
{
    const int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(i >= L)
        return;
    auto x = [&](int jj) -> T& { return ROWS ? X[jj + size_t(i) * ldx] : X[i + size_t(jj) * ldx]; };
    if(k < 0)
        x(j) = -x(j);
    else
    {
        const T t = x(j);
        x(j) = x(k);
        x(k) = t;
    }
}

template <typename S, typename T, typename I>
class bdsqr_rotlog
{
    // matrices: 0 = VT (rows), 1 = U (columns), 2 = C (rows)
    struct Op
    {
        char type; // 'Q' sequence, 'R' one rotation, 'N' negation, 'W' swap
        int mat, fwd, j0, cnt;
        long off;
    };
    rocblas_handle handle;
    hipStream_t stream;
    int n = 0;
    T* X[3] = {nullptr, nullptr, nullptr};
    int ld[3] = {0, 0, 0};
    int L[3] = {0, 0, 0};
    std::vector<Op> ops;
    long nrec = 0, cap = 0;
    // two sets of pinned buffers for the rotations: the host fills one while the copy of the other
    // may still be pending (ev[i] is recorded after the copies of set i)
    S *hcb[2] = {nullptr, nullptr}, *hsb[2] = {nullptr, nullptr};
    hipEvent_t ev[2] = {nullptr, nullptr};
    int cur = 0;
    S *hc = nullptr, *hs = nullptr; // current set
    // the plan of the last flush of each set (kept until its copies are done)
    std::vector<bdsqr_rot_grp> pgrps[2];
    std::vector<bdsqr_rot_seq> pseqs[2];
    std::vector<int> pwg[2], pwt[2];
    S *dc = nullptr, *ds = nullptr;
    T *dQ = nullptr, *dT = nullptr;
    size_t qcap = 0, tcap = 0;
    bdsqr_rot_grp* dgrp = nullptr;
    bdsqr_rot_seq* dseq = nullptr;
    int *dwg = nullptr, *dwt = nullptr;
    size_t gcap = 0, scap = 0, wcap = 0, tcap2 = 0;

    template <typename P>
    static void grow(P*& p, size_t& capacity, size_t need)
    {
        if(need <= capacity)
            return;
        if(p)
            (void)hipFree(p);
        capacity = std::max(need, capacity * 2);
        if(hipMalloc(&p, sizeof(P) * capacity) != hipSuccess)
            THROW_IF_ROCBLAS_ERROR(rocblas_status_memory_error);
    }

    // matrix and index of a reference to an element of the first column of VT or C (rows = true),
    // or of the first row of U
    void locate(const T* p, bool rows, int& mat, int& j) const
    {
        for(int m = 0; m < 3; m++)
        {
            if(!X[m] || L[m] == 0 || rows != (m != 1))
                continue;
            const ptrdiff_t off = p - X[m];
            // (an element of the first column of VT or C, or of the first row of U)
            if(m != 1 ? (off >= 0 && off < n) : (off >= 0 && off % ld[m] == 0 && off / ld[m] < n))
            {
                mat = m;
                j = int(m == 1 ? off / ld[m] : off % ld[m]);
                return;
            }
        }
        THROW_IF_ROCBLAS_ERROR(rocblas_status_internal_error);
    }

public:
    bdsqr_rotlog(rocblas_handle h, hipStream_t s, I nmax)
        : handle(h)
        , stream(s)
    {
        cap = std::max(long(BDSQR_ROT_CHUNK), long(2) * nmax);
        for(int i = 0; i < 2; i++)
            if(hipHostMalloc(&hcb[i], sizeof(S) * cap) != hipSuccess
               || hipHostMalloc(&hsb[i], sizeof(S) * cap) != hipSuccess
               || hipEventCreateWithFlags(&ev[i], hipEventDisableTiming) != hipSuccess)
                THROW_IF_ROCBLAS_ERROR(rocblas_status_memory_error);
        hc = hcb[0];
        hs = hsb[0];
        if(hipMalloc(&dc, sizeof(S) * cap) != hipSuccess
           || hipMalloc(&ds, sizeof(S) * cap) != hipSuccess)
            THROW_IF_ROCBLAS_ERROR(rocblas_status_memory_error);
    }
    ~bdsqr_rotlog()
    {
        (void)hipStreamSynchronize(stream);
        for(int i = 0; i < 2; i++)
        {
            (void)hipHostFree(hcb[i]);
            (void)hipHostFree(hsb[i]);
            (void)hipEventDestroy(ev[i]);
        }
        (void)hipFree(dc);
        (void)hipFree(ds);
        (void)hipFree(dQ);
        (void)hipFree(dT);
        (void)hipFree(dgrp);
        (void)hipFree(dseq);
        (void)hipFree(dwg);
        (void)hipFree(dwt);
    }

    // the matrices of one problem (n = order of the bidiagonal matrix)
    void set_matrices(I n_, T* vt, I ldvt, I ncvt, T* u, I ldu, I nru, T* c, I ldc, I ncc)
    {
        n = n_;
        X[0] = vt;
        ld[0] = ldvt;
        L[0] = ncvt;
        X[1] = u;
        ld[1] = ldu;
        L[1] = nru;
        X[2] = c;
        ld[2] = ldc;
        L[2] = ncc;
        grow(dT, tcap, size_t(std::max({L[0], L[1], L[2], 1})) * (BDSQR_ROT_B + BDSQR_ROT_K));
    }

    // LASR with pivot = variable: left side on rows of VT or C (m rows), right side on columns of
    // U (n columns); A is the first element of the submatrix
    void lasr(rocblas_side side, rocblas_direct direct, I m, I nn, const S* c, const S* s, T& A)
    {
        const int cnt = int((side == rocblas_side_left ? m : nn) - 1);
        if(cnt <= 0)
            return;
        if(nrec + cnt > cap)
            flush();
        Op o;
        o.type = 'Q';
        int j;
        locate(&A, side == rocblas_side_left, o.mat, j);
        o.fwd = (direct == rocblas_forward_direction);
        o.j0 = j;
        o.cnt = cnt;
        o.off = nrec;
        std::copy(c, c + cnt, hc + nrec);
        std::copy(s, s + cnt, hs + nrec);
        nrec += cnt;
        ops.push_back(o);
    }
    // ROT of the lines x and y = x + 1 (inc != 1: rows of VT or C; inc = 1: columns of U)
    void rot(T& x, I inc, S c, S s)
    {
        if(nrec + 1 > cap)
            flush();
        Op o;
        o.type = 'R';
        int j;
        locate(&x, inc != 1, o.mat, j);
        o.fwd = 1;
        o.j0 = j;
        o.cnt = 1;
        o.off = nrec;
        hc[nrec] = c;
        hs[nrec] = s;
        nrec++;
        ops.push_back(o);
    }
    void negate(T& x, I inc)
    {
        Op o;
        o.type = 'N';
        int j;
        locate(&x, inc != 1, o.mat, j);
        o.fwd = 1;
        o.j0 = j;
        o.cnt = -1;
        o.off = 0;
        ops.push_back(o);
    }
    void swap(T& x, T& y, I inc)
    {
        Op o;
        o.type = 'W';
        int j, k, m2;
        locate(&x, inc != 1, o.mat, j);
        locate(&y, inc != 1, m2, k);
        o.fwd = 1;
        o.j0 = j;
        o.cnt = k;
        o.off = 0;
        ops.push_back(o);
    }

    // enqueue the recorded operations (without waiting for them)
    void flush()
    {
        if(ops.empty())
            return;
        const int K = BDSQR_ROT_K, b = BDSQR_ROT_B;
        struct Item
        {
            int grp; // group, or -1 for the operation op
            int op;
        };
        std::vector<Item> plan[3];
        std::vector<bdsqr_rot_grp>& grps = pgrps[cur];
        std::vector<bdsqr_rot_seq>& seqs = pseqs[cur];
        std::vector<int>& win_grp = pwg[cur];
        std::vector<int>& win_t = pwt[cur];
        grps.clear();
        seqs.clear();
        win_grp.clear();
        win_t.clear();
        size_t qtot = 0;
        for(int mat = 0; mat < 3; mat++)
        {
            std::vector<bdsqr_rot_seq> cur;
            int cfwd = 1;
            auto close = [&]() {
                if(cur.empty())
                    return;
                const int Kg = int(cur.size());
                int plo = 1 << 30, phi = -1;
                for(auto& q : cur)
                {
                    const int a = cfwd ? q.j0 : n - 2 - (q.j0 + q.cnt - 1);
                    const int z = cfwd ? q.j0 + q.cnt - 1 : n - 2 - q.j0;
                    plo = std::min(plo, a);
                    phi = std::max(phi, z);
                }
                bdsqr_rot_grp gp;
                gp.fwd = cfwd;
                gp.K = Kg;
                gp.seq0 = int(seqs.size());
                gp.P0 = plo - (Kg - 1);
                gp.b = b;
                gp.T = (phi - gp.P0) / b + 1;
                gp.w = b + Kg;
                gp.qoff = long(qtot);
                qtot += size_t(gp.T) * gp.w * gp.w;
                for(int t = 0; t < gp.T; t++)
                {
                    win_grp.push_back(int(grps.size()));
                    win_t.push_back(t);
                }
                seqs.insert(seqs.end(), cur.begin(), cur.end());
                plan[mat].push_back({int(grps.size()), -1});
                grps.push_back(gp);
                cur.clear();
            };
            for(int i = 0; i < int(ops.size()); i++)
            {
                const Op& o = ops[i];
                if(o.mat != mat)
                    continue;
                if(o.type == 'Q')
                {
                    if(!cur.empty() && (o.fwd != cfwd || int(cur.size()) == K))
                        close();
                    cfwd = o.fwd;
                    cur.push_back({o.j0, o.cnt, o.off});
                    continue;
                }
                close();
                plan[mat].push_back({-1, i});
            }
            close();
        }

        // rotations, and the accumulation of all the windows
        BDSQR_ROTLOG_HIP(hipMemcpyAsync(dc, hc, sizeof(S) * nrec, hipMemcpyHostToDevice, stream));
        BDSQR_ROTLOG_HIP(hipMemcpyAsync(ds, hs, sizeof(S) * nrec, hipMemcpyHostToDevice, stream));
        if(!grps.empty())
        {
            grow(dgrp, gcap, grps.size());
            grow(dseq, scap, seqs.size());
            grow(dwg, wcap, win_grp.size());
            grow(dwt, tcap2, win_t.size());
            grow(dQ, qcap, qtot);
            // (pageable copies are staged synchronously, so the vectors can go out of scope)
            BDSQR_ROTLOG_HIP(hipMemcpyAsync(dgrp, grps.data(), sizeof(bdsqr_rot_grp) * grps.size(),
                                            hipMemcpyHostToDevice, stream));
            BDSQR_ROTLOG_HIP(hipMemcpyAsync(dseq, seqs.data(), sizeof(bdsqr_rot_seq) * seqs.size(),
                                            hipMemcpyHostToDevice, stream));
            BDSQR_ROTLOG_HIP(hipMemcpyAsync(dwg, win_grp.data(), sizeof(int) * win_grp.size(),
                                            hipMemcpyHostToDevice, stream));
            BDSQR_ROTLOG_HIP(hipMemcpyAsync(dwt, win_t.data(), sizeof(int) * win_t.size(),
                                            hipMemcpyHostToDevice, stream));
            ROCSOLVER_LAUNCH_KERNEL((bdsqr_rot_accum<T, S>), dim3(int(win_grp.size())), dim3(256),
                                    0, stream, n, dgrp, dwg, dwt, dseq, dc, ds, dQ);
        }
        BDSQR_ROTLOG_HIP(hipEventRecord(ev[cur], stream));

        // the GEMMs and the other operations, in order, for each matrix
        rocblas_pointer_mode_saver saver(handle, rocblas_pointer_mode_host);
        const T one = 1, zero = 0;
        for(int mat = 0; mat < 3; mat++)
        {
            const bool rows = (mat != 1);
            const int Lm = L[mat];
            if(Lm == 0)
                continue;
            T* Xm = X[mat];
            const int ldm = ld[mat];
            const dim3 gr((Lm - 1) / 256 + 1), bl(256);
            for(const Item& it : plan[mat])
            {
                if(it.grp < 0)
                {
                    const Op& o = ops[it.op];
                    if(o.type == 'R')
                    {
                        if(rows)
                            ROCSOLVER_LAUNCH_KERNEL((bdsqr_rot_apply_seq<true, T, S>), gr, bl, 0,
                                                    stream, Lm, Xm, ldm, o.j0, 1, 1, dc + o.off,
                                                    ds + o.off);
                        else
                            ROCSOLVER_LAUNCH_KERNEL((bdsqr_rot_apply_seq<false, T, S>), gr, bl, 0,
                                                    stream, Lm, Xm, ldm, o.j0, 1, 1, dc + o.off,
                                                    ds + o.off);
                    }
                    else
                    {
                        const int k = (o.type == 'N') ? -1 : o.cnt;
                        if(rows)
                            ROCSOLVER_LAUNCH_KERNEL((bdsqr_rot_negswap<true, T>), gr, bl, 0, stream,
                                                    Lm, Xm, ldm, o.j0, k);
                        else
                            ROCSOLVER_LAUNCH_KERNEL((bdsqr_rot_negswap<false, T>), gr, bl, 0,
                                                    stream, Lm, Xm, ldm, o.j0, k);
                    }
                    continue;
                }
                const bdsqr_rot_grp& gp = grps[it.grp];
                if(gp.K == 1)
                {
                    // a single sequence: one launch
                    const bdsqr_rot_seq& q = seqs[gp.seq0];
                    if(rows)
                        ROCSOLVER_LAUNCH_KERNEL((bdsqr_rot_apply_seq<true, T, S>), gr, bl, 0,
                                                stream, Lm, Xm, ldm, q.j0, q.cnt, gp.fwd,
                                                dc + q.off, ds + q.off);
                    else
                        ROCSOLVER_LAUNCH_KERNEL((bdsqr_rot_apply_seq<false, T, S>), gr, bl, 0,
                                                stream, Lm, Xm, ldm, q.j0, q.cnt, gp.fwd,
                                                dc + q.off, ds + q.off);
                    continue;
                }
                for(int t = 0; t < gp.T; t++)
                {
                    const int pw0 = gp.P0 + t * gp.b;
                    const int c0 = gp.fwd ? pw0 : n - 1 - (pw0 + gp.w - 1);
                    const int lo = std::max(c0, 0), hi = std::min(c0 + gp.w - 1, n - 1);
                    if(lo > hi)
                        continue;
                    const int ww = hi - lo + 1;
                    const rocblas_stride qoff
                        = gp.qoff + rocblas_stride(t) * gp.w * gp.w + idx2D(lo - c0, lo - c0, gp.w);
                    if(rows)
                    {
                        // X(lo:hi, :) = Q' X(lo:hi, :)
                        (rocblasCall_gemm(handle, rocblas_operation_transpose,
                                          rocblas_operation_none, ww, Lm, ww, &one, (const T*)dQ,
                                          qoff, gp.w, 0, (const T*)Xm, rocblas_stride(lo), ldm, 0,
                                          &zero, dT, 0, ww, 0, 1, (T**)nullptr));
                        BDSQR_ROTLOG_HIP(hipMemcpy2DAsync(Xm + lo, sizeof(T) * ldm, dT,
                                                          sizeof(T) * ww, sizeof(T) * ww, Lm,
                                                          hipMemcpyDeviceToDevice, stream));
                    }
                    else
                    {
                        // X(:, lo:hi) = X(:, lo:hi) Q
                        (rocblasCall_gemm(handle, rocblas_operation_none, rocblas_operation_none,
                                          Lm, ww, ww, &one, (const T*)Xm, idx2D(0, lo, ldm), ldm, 0,
                                          (const T*)dQ, qoff, gp.w, 0, &zero, dT, 0, Lm, 0, 1,
                                          (T**)nullptr));
                        BDSQR_ROTLOG_HIP(hipMemcpy2DAsync(Xm + idx2D(0, lo, ldm), sizeof(T) * ldm,
                                                          dT, sizeof(T) * Lm, sizeof(T) * Lm, ww,
                                                          hipMemcpyDeviceToDevice, stream));
                    }
                }
            }
        }
        ops.clear();
        nrec = 0;
        // switch to the other set of buffers, once its last copies are done
        cur = 1 - cur;
        BDSQR_ROTLOG_HIP(hipEventSynchronize(ev[cur]));
        hc = hcb[cur];
        hs = hsb[cur];
    }

    // apply the remaining operations and wait for all of them
    void finish()
    {
        flush();
        BDSQR_ROTLOG_HIP(hipStreamSynchronize(stream));
    }
};

ROCSOLVER_END_NAMESPACE
#undef BDSQR_ROTLOG_HIP

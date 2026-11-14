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
#include <memory>
#include <type_traits>
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

// a sequence of rotations of the pairs j0..j0+cnt-1, at level lvl of its group (the sequences of a
// level act on disjoint indices, and are applied after those of the previous levels)
struct bdsqr_rot_seq
{
    int j0, cnt, lvl;
    long off;
};
// a group of K levels of sequences in the same direction (nseq sequences from seq0, sorted by
// level), cut into T windows; its accumulated windows are those of win_grp/win_t from win0
struct bdsqr_rot_grp
{
    int fwd, K, seq0, nseq, P0, b, T, w, win0, nwin;
    long qoff;
};
// block j0..j1 of the bidiagonal matrix, rotated in sweep slot (backward if dir < 0)
struct bdsqr_rot_desc
{
    int slot, dir, j0, j1;
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
    T* Qt = Q + gp.qoff + size_t(wid - gp.win0) * w * w;
    for(int idx = hipThreadIdx_x; idx < w * w; idx += hipBlockDim_x)
        Qt[idx] = (idx % w == idx / w) ? T(1) : T(0);
    __syncthreads();
    const int pw0 = gp.P0 + t * b;
    const int base = fwd ? pw0 : n - 1 - (pw0 + w - 1); // first index of the window
    for(int qi = 0; qi < gp.nseq; qi++)
    {
        const bdsqr_rot_seq q = seqs[gp.seq0 + qi];
        const int k = q.lvl;
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

/** BDSQR_ROT_GEMM applies the accumulated window t of group gp (at offset qoff of Q) to the rows
    (X(lo:hi, :) = Q' X(lo:hi, :)) or the columns (X(:, lo:hi) = X(:, lo:hi) Q) of X, through the
    temporary dT. The pointer mode of the handle must be host. **/
template <typename T>
void bdsqr_rot_gemm(rocblas_handle handle,
                    const bool rows,
                    const int n,
                    const bdsqr_rot_grp& gp,
                    const int t,
                    const rocblas_stride qoff_t,
                    const T* dQ,
                    T* Xm,
                    const int ldm,
                    const int Lm,
                    T* dT)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    const T one = 1, zero = 0;
    const int pw0 = gp.P0 + t * gp.b;
    const int c0 = gp.fwd ? pw0 : n - 1 - (pw0 + gp.w - 1);
    const int lo = std::max(c0, 0), hi = std::min(c0 + gp.w - 1, n - 1);
    if(lo > hi)
        return;
    const int ww = hi - lo + 1;
    const rocblas_stride qoff = qoff_t + idx2D(lo - c0, lo - c0, gp.w);
    if(rows)
    {
        (rocblasCall_gemm(handle, rocblas_operation_transpose, rocblas_operation_none, ww, Lm, ww,
                          &one, dQ, qoff, gp.w, 0, (const T*)Xm, rocblas_stride(lo), ldm, 0, &zero,
                          dT, 0, ww, 0, 1, (T**)nullptr));
        BDSQR_ROTLOG_HIP(hipMemcpy2DAsync(Xm + lo, sizeof(T) * ldm, dT, sizeof(T) * ww,
                                          sizeof(T) * ww, Lm, hipMemcpyDeviceToDevice, stream));
    }
    else
    {
        (rocblasCall_gemm(handle, rocblas_operation_none, rocblas_operation_none, Lm, ww, ww, &one,
                          (const T*)Xm, idx2D(0, lo, ldm), ldm, 0, dQ, qoff, gp.w, 0, &zero, dT, 0,
                          Lm, 0, 1, (T**)nullptr));
        BDSQR_ROTLOG_HIP(hipMemcpy2DAsync(Xm + idx2D(0, lo, ldm), sizeof(T) * ldm, dT, sizeof(T) * Lm,
                                          sizeof(T) * Lm, ww, hipMemcpyDeviceToDevice, stream));
    }
}

/** BDSQR_ROT_GROW grows the device buffer p to at least need entries (at least doubling it). The
    old buffer is freed first; if the allocation fails, p is null and capacity 0 (so that the buffer
    is not freed twice) and an exception is thrown. **/
template <typename P>
void bdsqr_rot_grow(P*& p, size_t& capacity, size_t need)
{
    if(need <= capacity)
        return;
    const size_t newcap = std::max(need, capacity * 2);
    if(p)
        (void)hipFree(p);
    p = nullptr;
    capacity = 0;
    if(hipMalloc(&p, sizeof(P) * newcap) != hipSuccess)
    {
        p = nullptr;
        THROW_IF_ROCBLAS_ERROR(rocblas_status_memory_error);
    }
    capacity = newcap;
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
        bdsqr_rot_grow(p, capacity, need);
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
        // rotations recorded before a chunk is applied: enough for several groups of BDSQR_ROT_K
        // sweeps (at most BDSQR_ROT_CHUNK), and at least two sweeps
        cap = std::max(long(2) * nmax, std::min(long(BDSQR_ROT_CHUNK), long(8) * BDSQR_ROT_K * nmax));
        bool ok = true;
        for(int i = 0; i < 2 && ok; i++)
            ok = hipHostMalloc(&hcb[i], sizeof(S) * cap) == hipSuccess
                && hipHostMalloc(&hsb[i], sizeof(S) * cap) == hipSuccess
                && hipEventCreateWithFlags(&ev[i], hipEventDisableTiming) == hipSuccess;
        ok = ok && hipMalloc(&dc, sizeof(S) * cap) == hipSuccess
            && hipMalloc(&ds, sizeof(S) * cap) == hipSuccess;
        if(!ok)
        {
            // (the destructor does not run when the constructor throws)
            release();
            THROW_IF_ROCBLAS_ERROR(rocblas_status_memory_error);
        }
        hc = hcb[0];
        hs = hsb[0];
    }
    ~bdsqr_rotlog()
    {
        (void)hipStreamSynchronize(stream);
        release();
    }
    // free all the buffers (null pointers are skipped)
    void release()
    {
        for(int i = 0; i < 2; i++)
        {
            if(hcb[i])
                (void)hipHostFree(hcb[i]);
            if(hsb[i])
                (void)hipHostFree(hsb[i]);
            if(ev[i])
                (void)hipEventDestroy(ev[i]);
            hcb[i] = hsb[i] = nullptr;
            ev[i] = nullptr;
        }
        for(S** p : {&dc, &ds})
        {
            if(*p)
                (void)hipFree(*p);
            *p = nullptr;
        }
        for(T** p : {&dQ, &dT})
        {
            if(*p)
                (void)hipFree(*p);
            *p = nullptr;
        }
        if(dgrp)
            (void)hipFree(dgrp);
        if(dseq)
            (void)hipFree(dseq);
        if(dwg)
            (void)hipFree(dwg);
        if(dwt)
            (void)hipFree(dwt);
        dgrp = nullptr;
        dseq = nullptr;
        dwg = dwt = nullptr;
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
                gp.nseq = Kg;
                gp.win0 = int(win_grp.size());
                gp.P0 = plo - (Kg - 1);
                gp.b = b;
                gp.T = (phi - gp.P0) / b + 1;
                gp.w = b + Kg;
                gp.nwin = gp.T;
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
                    cur.push_back({o.j0, o.cnt, int(cur.size()), o.off});
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
                    bdsqr_rot_gemm(handle, rows, n, gp, t,
                                   gp.qoff + rocblas_stride(t) * gp.w * gp.w, dQ, Xm, ldm, Lm, dT);
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

/*
 * ===========================================================================
 *    BDSQR_GPULOG does the same for the QR iteration of BDSQR on the device
 *    (one problem): after each sweep, BDSQR_ROT_SNAPSHOT copies the rotations
 *    of the blocks that were rotated into slot s of a log (one slot per sweep,
 *    indexed by pair as the matrix), with the convention above, and appends a
 *    descriptor of each block. Every BDSQR_ROT_SWEEPS sweeps, flush reads the
 *    descriptors, groups the blocks of consecutive sweeps by direction (the
 *    blocks of a sweep, disjoint, share a level of their group; a group in one
 *    direction is closed when a block in the other direction overlaps it) and
 *    applies the groups with accumulated windows and GEMMs: the rotations of
 *    the right (VT) and of the left (U and C) give two sets of windows.
 * ===========================================================================
 */

#ifndef BDSQR_ROT_SWEEPS
#define BDSQR_ROT_SWEEPS 64
#endif

/** BDSQR_ROT_SNAPSHOT copies the rotations of the blocks rotated in the last sweep (as stored by
    BDSQR_COMPUTE in work, read as BDSQR_ROTATE does) into slot slot of the log. Rotations of the
    right in lA_c/lA_s (if nv), of the left in lB_c/lB_s (if nuc); the sines of forward sweeps
    change sign. **/
template <typename S>
ROCSOLVER_KERNEL void bdsqr_rot_snapshot(const int n,
                                         const int nv,
                                         const int nuc,
                                         const int maxiter,
                                         const int slot,
                                         const int* splits,
                                         const S* work,
                                         const int incW,
                                         S* lA_c,
                                         S* lA_s,
                                         S* lB_c,
                                         S* lB_s,
                                         bdsqr_rot_desc* desc,
                                         int* ndesc,
                                         const int* completed)
{
    if(completed[2])
        return;
    const int num_splits = int(work[2]);
    for(int sid = hipBlockIdx_y; sid < num_splits; sid += hipGridDim_y)
    {
        const int dir = splits[4 * sid], k0 = splits[4 * sid + 1], k1 = splits[4 * sid + 2];
        const int iter = splits[4 * sid + 3];
        if(k0 >= k1 || iter >= maxiter || dir == 0)
            continue;
        if(hipThreadIdx_x == 0)
        {
            const int d = atomicAdd(ndesc, 1);
            desc[d] = {slot, dir, k0, k1};
        }
        const S* rots = work + 4 + incW * k0;
        const int nn = k1 - k0 + 1;
        const int nr = nv ? 2 * nn : 0;
        const S sg = dir > 0 ? S(-1) : S(1);
        const size_t o = size_t(slot) * n + k0;
        for(int r = hipThreadIdx_x; r < nn - 1; r += hipBlockDim_x)
        {
            if(nv)
            {
                lA_c[o + r] = rots[r];
                lA_s[o + r] = sg * rots[r + nn];
            }
            if(nuc)
            {
                lB_c[o + r] = rots[nr + r];
                lB_s[o + r] = sg * rots[nr + r + nn];
            }
        }
    }
}

/** BDSQR_HOST_PTR returns the address of the first matrix of A (shifted), as a host value **/
template <typename T, typename W>
T* bdsqr_host_ptr(W A, const rocblas_stride shift, hipStream_t stream)
{
    if(!A)
        return nullptr;
    if constexpr(std::is_pointer_v<std::remove_pointer_t<W>>)
    {
        T* p;
        if(hipMemcpyAsync(&p, A, sizeof(T*), hipMemcpyDeviceToHost, stream) != hipSuccess
           || hipStreamSynchronize(stream) != hipSuccess)
            THROW_IF_ROCBLAS_ERROR(rocblas_status_internal_error);
        return p + shift;
    }
    else
        return A + shift;
}

template <typename S, typename T>
class bdsqr_gpulog
{
    rocblas_handle handle;
    hipStream_t stream;
    int n, ns;
    // matrices: 0 = VT (rows), 1 = U (columns), 2 = C (rows)
    T* X[3];
    int ld[3], L[3];
    S *lA_c = nullptr, *lA_s = nullptr, *lB_c = nullptr, *lB_s = nullptr;
    bdsqr_rot_desc* ddesc = nullptr;
    int* dnd = nullptr;
    T *dQA = nullptr, *dQB = nullptr, *dT = nullptr;
    size_t qacap = 0, qbcap = 0;
    bdsqr_rot_grp* dgrp = nullptr;
    bdsqr_rot_seq* dseq = nullptr;
    int *dwg = nullptr, *dwt = nullptr;
    size_t gcap = 0, scap = 0, wcap = 0, tcap2 = 0;

    template <typename P>
    static void grow(P*& p, size_t& capacity, size_t need)
    {
        bdsqr_rot_grow(p, capacity, need);
    }
    // (an allocation in the constructor: on failure, everything is released before throwing, as the
    // destructor does not run)
    template <typename P>
    void alloc(P*& p, size_t count)
    {
        if(hipMalloc(&p, sizeof(P) * std::max(count, size_t(1))) != hipSuccess)
        {
            p = nullptr;
            release();
            THROW_IF_ROCBLAS_ERROR(rocblas_status_memory_error);
        }
    }

public:
    bdsqr_gpulog(rocblas_handle h,
                 int n_,
                 T* vt,
                 int ldvt,
                 int nv,
                 T* u,
                 int ldu,
                 int nu,
                 T* c,
                 int ldc,
                 int nc)
        : handle(h)
        , n(n_)
        , ns(BDSQR_ROT_SWEEPS)
        , X{vt, u, c}
        , ld{ldvt, ldu, ldc}
        , L{vt ? nv : 0, u ? nu : 0, c ? nc : 0}
    {
        rocblas_get_stream(handle, &stream);
        const size_t nlog = size_t(ns) * n;
        if(L[0])
        {
            alloc(lA_c, nlog);
            alloc(lA_s, nlog);
        }
        if(L[1] || L[2])
        {
            alloc(lB_c, nlog);
            alloc(lB_s, nlog);
        }
        alloc(ddesc, size_t(ns) * (n / 2 + 1));
        alloc(dnd, 1);
        alloc(dT, size_t(std::max({L[0], L[1], L[2], 1})) * (BDSQR_ROT_B + BDSQR_ROT_K));
        BDSQR_ROTLOG_HIP(hipMemsetAsync(dnd, 0, sizeof(int), stream));
    }
    ~bdsqr_gpulog()
    {
        (void)hipStreamSynchronize(stream);
        release();
    }
    // free all the buffers (null pointers are skipped)
    void release()
    {
        for(S** p : {&lA_c, &lA_s, &lB_c, &lB_s})
        {
            if(*p)
                (void)hipFree(*p);
            *p = nullptr;
        }
        for(T** p : {&dQA, &dQB, &dT})
        {
            if(*p)
                (void)hipFree(*p);
            *p = nullptr;
        }
        if(ddesc)
            (void)hipFree(ddesc);
        if(dnd)
            (void)hipFree(dnd);
        if(dgrp)
            (void)hipFree(dgrp);
        if(dseq)
            (void)hipFree(dseq);
        if(dwg)
            (void)hipFree(dwg);
        if(dwt)
            (void)hipFree(dwt);
        ddesc = nullptr;
        dnd = nullptr;
        dgrp = nullptr;
        dseq = nullptr;
        dwg = dwt = nullptr;
    }

    int sweeps() const
    {
        return ns;
    }

    // the log of the rotations of the left (U and C) and the descriptors, for a QR iteration
    // that records its sweeps itself (as STEQR)
    S* log_c() const
    {
        return lB_c;
    }
    S* log_s() const
    {
        return lB_s;
    }
    bdsqr_rot_desc* desc() const
    {
        return ddesc;
    }
    int* ndesc() const
    {
        return dnd;
    }

    // record the blocks rotated in the last sweep (slot = index of the sweep since the last flush)
    void snapshot(int slot,
                  int nsplits,
                  int maxiter,
                  const int* splits,
                  const S* work,
                  int incW,
                  const int* completed)
    {
        ROCSOLVER_LAUNCH_KERNEL((bdsqr_rot_snapshot<S>), dim3(1, std::max(nsplits, 1)), dim3(64), 0,
                                stream, n, L[0], L[1] + L[2], maxiter, slot, splits, work, incW,
                                lA_c, lA_s, lB_c, lB_s, ddesc, dnd, completed);
    }

    // apply the recorded sweeps (and reset the log)
    void flush()
    {
        int nd = 0;
        BDSQR_ROTLOG_HIP(hipMemcpyAsync(&nd, dnd, sizeof(int), hipMemcpyDeviceToHost, stream));
        BDSQR_ROTLOG_HIP(hipStreamSynchronize(stream));
        if(nd == 0)
            return;
        std::vector<bdsqr_rot_desc> desc(nd);
        BDSQR_ROTLOG_HIP(hipMemcpyAsync(desc.data(), ddesc, sizeof(bdsqr_rot_desc) * nd,
                                        hipMemcpyDeviceToHost, stream));
        BDSQR_ROTLOG_HIP(hipMemsetAsync(dnd, 0, sizeof(int), stream));
        BDSQR_ROTLOG_HIP(hipStreamSynchronize(stream));
        std::stable_sort(
            desc.begin(), desc.end(),
            [](const bdsqr_rot_desc& a, const bdsqr_rot_desc& b) { return a.slot < b.slot; });

        // plan the groups (in the order they must be applied)
        const int K = BDSQR_ROT_K, b = BDSQR_ROT_B;
        std::vector<bdsqr_rot_grp> grps;
        std::vector<bdsqr_rot_seq> seqs;
        std::vector<int> win_grp, win_t;
        size_t qtot = 0;
        struct Pend
        {
            std::vector<bdsqr_rot_seq> q;
            int levels = 0, lastslot = -1;
        } pend[2]; // 0 = backward, 1 = forward
        std::vector<char> owner(n, 0); // 1 + direction of the pending group with the index
        std::vector<char> hit;
        auto emit = [&](int fwd) {
            Pend& g = pend[fwd];
            if(g.q.empty())
                return;
            const int Kg = g.levels;
            int plo = 1 << 30, phi = -1;
            for(auto& q : g.q)
            {
                plo = std::min(plo, fwd ? q.j0 : n - 2 - (q.j0 + q.cnt - 1));
                phi = std::max(phi, fwd ? q.j0 + q.cnt - 1 : n - 2 - q.j0);
                for(int i = q.j0; i <= q.j0 + q.cnt; i++)
                    owner[i] = 0;
            }
            bdsqr_rot_grp gp;
            gp.fwd = fwd;
            gp.K = Kg;
            gp.seq0 = int(seqs.size());
            gp.nseq = int(g.q.size());
            gp.P0 = plo - (Kg - 1);
            gp.b = b;
            gp.T = (phi - gp.P0) / b + 1;
            gp.w = b + Kg;
            gp.win0 = int(win_grp.size());
            // (only the windows with rotations)
            hit.assign(gp.T, 0);
            for(auto& q : g.q)
            {
                const int a = fwd ? q.j0 : n - 2 - (q.j0 + q.cnt - 1);
                const int z = fwd ? q.j0 + q.cnt - 1 : n - 2 - q.j0;
                const int sh = gp.P0 + (Kg - 1 - q.lvl);
                const int t0 = std::max((a - sh) / b, 0), t1 = std::min((z - sh) / b, gp.T - 1);
                for(int t = t0; t <= t1; t++)
                    hit[t] = 1;
            }
            for(int t = 0; t < gp.T; t++)
                if(hit[t])
                {
                    win_grp.push_back(int(grps.size()));
                    win_t.push_back(t);
                }
            gp.nwin = int(win_grp.size()) - gp.win0;
            gp.qoff = long(qtot);
            if(Kg > 1)
                qtot += size_t(gp.nwin) * gp.w * gp.w;
            else
                win_grp.resize(gp.win0), win_t.resize(gp.win0), gp.nwin = 0;
            seqs.insert(seqs.end(), g.q.begin(), g.q.end());
            grps.push_back(gp);
            g = Pend();
        };
        for(const bdsqr_rot_desc& d : desc)
        {
            const int fwd = d.dir > 0 ? 1 : 0;
            bool clash = false;
            for(int i = d.j0; i <= d.j1 && !clash; i++)
                clash = (owner[i] == 1 + (1 - fwd));
            if(clash)
                emit(1 - fwd);
            Pend& g = pend[fwd];
            if(g.lastslot != d.slot)
            {
                if(g.levels == K)
                    emit(fwd);
                g.levels++;
                g.lastslot = d.slot;
            }
            g.q.push_back({d.j0, d.j1 - d.j0, g.levels - 1, long(d.slot) * n + d.j0});
            for(int i = d.j0; i <= d.j1; i++)
                owner[i] = char(1 + fwd);
        }
        emit(0);
        emit(1);

        // accumulate the windows (one launch per set of rotations)
        grow(dgrp, gcap, grps.size());
        grow(dseq, scap, seqs.size());
        BDSQR_ROTLOG_HIP(hipMemcpyAsync(dgrp, grps.data(), sizeof(bdsqr_rot_grp) * grps.size(),
                                        hipMemcpyHostToDevice, stream));
        BDSQR_ROTLOG_HIP(hipMemcpyAsync(dseq, seqs.data(), sizeof(bdsqr_rot_seq) * seqs.size(),
                                        hipMemcpyHostToDevice, stream));
        if(!win_grp.empty())
        {
            grow(dwg, wcap, win_grp.size());
            grow(dwt, tcap2, win_t.size());
            BDSQR_ROTLOG_HIP(hipMemcpyAsync(dwg, win_grp.data(), sizeof(int) * win_grp.size(),
                                            hipMemcpyHostToDevice, stream));
            BDSQR_ROTLOG_HIP(hipMemcpyAsync(dwt, win_t.data(), sizeof(int) * win_t.size(),
                                            hipMemcpyHostToDevice, stream));
            if(L[0])
            {
                grow(dQA, qacap, qtot);
                ROCSOLVER_LAUNCH_KERNEL((bdsqr_rot_accum<T, S>), dim3(int(win_grp.size())), dim3(256),
                                        0, stream, n, dgrp, dwg, dwt, dseq, lA_c, lA_s, dQA);
            }
            if(L[1] || L[2])
            {
                grow(dQB, qbcap, qtot);
                ROCSOLVER_LAUNCH_KERNEL((bdsqr_rot_accum<T, S>), dim3(int(win_grp.size())), dim3(256),
                                        0, stream, n, dgrp, dwg, dwt, dseq, lB_c, lB_s, dQB);
            }
        }

        // the GEMMs (or the sequences of the groups with one level), in order, for each matrix
        rocblas_pointer_mode_saver saver(handle, rocblas_pointer_mode_host);
        for(int mat = 0; mat < 3; mat++)
        {
            const int Lm = L[mat];
            if(Lm == 0)
                continue;
            const bool rows = (mat != 1);
            T* Xm = X[mat];
            const int ldm = ld[mat];
            const S* lc = (mat == 0 ? lA_c : lB_c);
            const S* ls = (mat == 0 ? lA_s : lB_s);
            const T* dQ = (mat == 0 ? dQA : dQB);
            const dim3 gr((Lm - 1) / 256 + 1), bl(256);
            for(const bdsqr_rot_grp& gp : grps)
            {
                if(gp.K == 1)
                {
                    for(int qi = 0; qi < gp.nseq; qi++)
                    {
                        const bdsqr_rot_seq& q = seqs[gp.seq0 + qi];
                        if(rows)
                            ROCSOLVER_LAUNCH_KERNEL((bdsqr_rot_apply_seq<true, T, S>), gr, bl, 0,
                                                    stream, Lm, Xm, ldm, q.j0, q.cnt, gp.fwd,
                                                    lc + q.off, ls + q.off);
                        else
                            ROCSOLVER_LAUNCH_KERNEL((bdsqr_rot_apply_seq<false, T, S>), gr, bl, 0,
                                                    stream, Lm, Xm, ldm, q.j0, q.cnt, gp.fwd,
                                                    lc + q.off, ls + q.off);
                    }
                    continue;
                }
                for(int i = 0; i < gp.nwin; i++)
                    bdsqr_rot_gemm(handle, rows, n, gp, win_t[gp.win0 + i],
                                   gp.qoff + rocblas_stride(i) * gp.w * gp.w, dQ, Xm, ldm, Lm, dT);
            }
        }
    }
};

ROCSOLVER_END_NAMESPACE
#undef BDSQR_ROTLOG_HIP

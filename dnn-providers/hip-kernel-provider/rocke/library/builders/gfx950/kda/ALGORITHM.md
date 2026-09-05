# Chunkwise KDA algorithm

KDA is a gated delta-rule linear-attention recurrence. For one head, let
`S` have shape `DK x DV`, let `q` and `k` be `DK`-vectors, and let `v` be a
`DV`-vector. For each token:

```text
S <- Diag(exp(g)) S
u <- beta (v - k^T S)
S <- S + k u^T
o <- scale q^T S
```

The token recurrence is serial, but tokens can be grouped into chunks of `C`.
Within a chunk, define the cumulative per-channel decay
`Gamma_i = exp(sum(j <= i, g_j))` and the whole-chunk decay
`gamma_C = Gamma_(C-1)`. The chunk body is factored into six tiles:

```text
A    = (I + StrictTril(Diag(beta) Akk))^-1 Diag(beta)
       Akk_ij = k_i . (k_j * Gamma_i / Gamma_j)
GK   = K * Gamma
GQ   = Q * Gamma * scale
Aqk  = Tril(GQ (K / Gamma)^T)
Kt   = (K * gamma_C / Gamma)^T
dec  = gamma_C
```

The state-dependent part then becomes:

```text
Vt = A (V - GK S)
O  = GQ S + Aqk Vt
S  = Diag(dec) S + Kt Vt
```

`A`, `GK`, `GQ`, `Aqk`, `Kt`, and `dec` depend only on the chunk inputs.
Their construction is parallel over chunks. The state update remains serial
over chunks for each `(batch, head)` pair, while matrix products are partitioned
over workgroup waves.

## Transposed scan partition

The implementation carries `S^T` in fp32 MFMA accumulators. Each wave owns one
`scan_atom.m`-row band of `S^T` and the complete `DK` extent. The recurrence is
therefore emitted as:

```text
Z^T  = S^T GK^T
R^T  = V^T - Z^T
Vt^T = R^T A^T
O    = GQ S + Aqk Vt
S^T  = S^T Diag(dec) + Vt^T Kt^T
```

Every product remains in `A B^T` form with its contraction on the fastest
operand dimension. The state never needs a cross-wave reduction. It is
published to a bf16 LDS mirror each chunk because the MFMA accumulator and
operand fragment layouts are transposes of one another.

For both supported scan atoms, every accumulator slot owned by one lane has
the same state-column index. The scan consequently loads one `dec` value per
lane and `DK` atom tile and reuses it across the lane's whole accumulator
fragment.

## Stable decay factorization

The ratio `Gamma_i / Gamma_j` can overflow when formed directly. The emitter
uses a midpoint row `CREF = C // 2`:

```text
K * exp(Gc - Gref)   and   K * exp(Gref - Gc)
```

Their product reconstructs the required ratio while keeping each exponential
within a bounded half-chunk range. The cumulative gate is maintained in the
base-2 exponent domain, and exponent inputs are clamped to the finite hardware
range.

## Two compositions

The split composition uses one workgroup per chunk for tile construction. It
writes the six tiles to global memory, then uses one or more disjoint
value-band workgroups per `(batch, head)` for the ordered state scan. This
separates the tile and scan occupancy requirements.

The fused composition uses one workgroup per `(batch, head)` and keeps the
tiles in LDS while walking the chunks. It has less global-memory traffic but a
larger live LDS footprint. The two paths share the scan body and are checked for
bitwise equality where their layouts and accumulation order match.

Both compositions support a supplied initial state `h0` and an optional final
state `ht`. The numeric tests compare this state-carrying path with the
un-chunked token-serial reference above.

## Standalone scan software pipeline

Immediate HBM-to-LDS copies expose their memory dependency at the beginning of
every chunk. The optimized standalone scan overlaps that traffic with useful
work while retaining a single LDS tile set:

1. Stage chunk zero and rendezvous before entering the recurrence loop.
2. Issue the current chunk's V loads before publishing the state mirror. The
   mirror stores, rendezvous, and `Z^T` MFMAs cover the load latency.
3. Issue the next materialized tile set while the current chunk computes.
   SA32 issues at the top of the body; SA16 issues after V is consumed so its
   larger load burst cannot delay the residual.
4. Complete `Vt`, `O`, and the state update, then rendezvous to retire every
   read of the current LDS contents.
5. Commit the register-held next tiles into those same LDS allocations. The
   next iteration's existing state-publish rendezvous makes the writes visible.

The final iteration clamps its prefetch address to the current tile. This
performs one redundant valid prefetch instead of introducing a divergent tail
branch into every recurrence stream. C16 and single-wave schedules do not use
the tile pipeline.

## Value-band geometry and dispatch

Splitting `DV` creates independent scan workgroups because each band owns
disjoint rows of `S^T`, output channels, and final-state channels. No reduction
or atomic operation is required. The scan grid is:

```text
(batch * heads * value_splits, 1, 1)
```

For the Kimi K3 `bf16`, `DK=DV=128`, `C=32` contract, the gfx950 selector uses
four bands through 96 recurrence streams, two bands through 192 streams, and
one band above that. The corresponding scan schedules are B128/SA16,
B256/SA16, and B256/SA32. These cutovers are implemented by
`tuned_kda_chunk_scan_spec` and are shared by the benchmark and dispatcher.

The prep phase remains B256 and does not inherit scan-only `block_size` or
`scan_atom_m`. Its flat global tile format depends on the logical dimensions
and padding, so it remains layout-compatible with every selected scan
geometry.

`dispatch_kda(..., algorithm="chunk_scan")` returns this tuned scan spec,
builder, multiplied grid, block size, and ABI signature. The split composition
still requires a preceding `chunk_prep` dispatch on the same stream; `auto`
continues to select the self-contained fused kernel.

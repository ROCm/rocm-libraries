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
S  = Diag(dec) S + Kt^T Vt
```

`A`, `GK`, `GQ`, `Aqk`, `Kt`, and `dec` depend only on the chunk inputs.
Their construction is parallel over chunks. The state update remains serial
over chunks for each `(batch, head)` pair, while matrix products are partitioned
over workgroup waves.

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
writes the six tiles to global memory, then uses one workgroup per `(batch, head)`
for the ordered state scan. This separates the tile and scan occupancy
requirements.

The fused composition uses one workgroup per `(batch, head)` and keeps the
tiles in LDS while walking the chunks. It has less global-memory traffic but a
larger live LDS footprint. The two paths share the scan body and are checked for
bitwise equality where their layouts and accumulation order match.

Both compositions support a supplied initial state `h0` and an optional final
state `ht`. The numeric tests compare this state-carrying path with the
un-chunked token-serial reference above.

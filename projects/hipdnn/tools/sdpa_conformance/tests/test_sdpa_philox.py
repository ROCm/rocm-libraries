# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""``philox_mask`` against independent scalar implementations (CPU only).

The scalar reference below is deliberately not the vectorized code path: Python
big integers, one element at a time, straight from the formulas in
``flydsl/kernels_src/kernels/attention_bwd_cdna/philox.py`` and the counter map
of ``attention_bwd_rdna/fmha_bwd_dkdv_gfx11.py``.
"""

import math

import pytest
import torch

from hipdnn_sdpa import philox_mask
from hipdnn_sdpa._philox import dropout_threshold, philox4x32_10

M32 = 0xFFFFFFFF

# Random123 known-answer vectors for philox4x32 with 10 rounds
# (Random123 distribution, `tests/kat_vectors`, lines "philox4x32 10 ..."):
# (counter words, key words, expected output words).
KAT = [
    (
        (0x0, 0x0, 0x0, 0x0),
        (0x0, 0x0),
        (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8),
    ),
    (
        (M32, M32, M32, M32),
        (M32, M32),
        (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
    ),
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
    ),
]


def scalar_philox(counter, key):
    """Philox4x32-10 on Python ints (Triton's round, as in philox.py ``philox_4x``)."""
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for _ in range(10):
        p0 = 0xD2511F53 * c0
        p1 = 0xCD9E8D57 * c2
        c0, c1, c2, c3 = (
            ((p1 >> 32) ^ c1 ^ k0) & M32,
            p1 & M32,
            ((p0 >> 32) ^ c3 ^ k1) & M32,
            p0 & M32,
        )
        k0 = (k0 + 0x9E3779B9) & M32
        k1 = (k1 + 0xBB67AE85) & M32
    return c0, c1, c2, c3


def scalar_keep(z, h, q, kv, *, Hq, max_sq, max_sk, p, seed, offset):
    """Keep decision of element (q, kv) of sequence z, query head h (plan §3 map)."""
    row = -(-max_sk // 4)
    counter = (
        offset + (z * Hq + h) * max_sq * row + q * row + kv // 4
    ) & 0xFFFFFFFFFFFFFFFF
    words = scalar_philox(
        (counter & M32, counter >> 32, 0, 0), (seed & M32, (seed >> 32) & M32)
    )
    u32 = words[kv % 4]
    signed = u32 - (1 << 32) if u32 >= (1 << 31) else u32
    return signed > dropout_threshold(p)


@pytest.mark.parametrize("counter,key,expected", KAT)
def test_philox4x32_10_known_answers(counter, key, expected):
    words = [torch.tensor([w], dtype=torch.int64) for w in counter]
    out = philox4x32_10(*words, *key)
    assert tuple(int(w[0]) for w in out) == expected
    assert scalar_philox(counter, key) == expected


def test_philox_vectorized_matches_scalar_on_random_counters():
    gen = torch.Generator().manual_seed(7)
    ctr = torch.randint(0, 1 << 32, (4, 257), generator=gen, dtype=torch.int64)
    key = (0xDEADBEEF, 0x01234567)
    out = philox4x32_10(*ctr, *key)
    for i in range(ctr.shape[1]):
        want = scalar_philox(tuple(int(c) for c in ctr[:, i]), key)
        assert tuple(int(w[i]) for w in out) == want


@pytest.mark.parametrize(
    "B,Hq,max_sq,max_sk,p,seed,offset",
    [
        # GQA-style head count, max_sk not a multiple of 4, offset whose counters cross 2**32.
        (2, 6, 5, 7, 0.3, 0x123456789ABCDEF0, (1 << 32) - 5),
        # max_sk a multiple of 4, high offset words set, small p.
        (3, 2, 4, 8, 0.05, 0x00000001FFFFFFFF, 0x0000_0002_FFFF_FFF0),
        # single row per plane
        (1, 3, 1, 1, 0.5, 42, 0),
    ],
)
def test_mask_counter_map_matches_scalar_formula(
    B, Hq, max_sq, max_sk, p, seed, offset
):
    mask = philox_mask(B, Hq, max_sq, max_sk, p, seed, offset, "cpu")
    assert mask.shape == (B, Hq, max_sq, max_sk) and mask.dtype == torch.bool
    want = torch.tensor(
        [
            [
                [
                    [
                        scalar_keep(
                            z,
                            h,
                            q,
                            kv,
                            Hq=Hq,
                            max_sq=max_sq,
                            max_sk=max_sk,
                            p=p,
                            seed=seed,
                            offset=offset,
                        )
                        for kv in range(max_sk)
                    ]
                    for q in range(max_sq)
                ]
                for h in range(Hq)
            ]
            for z in range(B)
        ]
    )
    assert torch.equal(mask, want)


def test_mask_chunk_boundary_is_invisible(monkeypatch):
    import hipdnn_sdpa._philox as ph

    args = (2, 3, 17, 13, 0.4, 99, (1 << 32) - 3, "cpu")
    whole = philox_mask(*args)
    monkeypatch.setattr(ph, "_CHUNK_COUNTERS", 7)
    assert torch.equal(ph.philox_mask(*args), whole)


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_keep_rate_within_six_sigma(p):
    B, Hq, sq, sk = 2, 4, 64, 256
    mask = philox_mask(B, Hq, sq, sk, p, seed=2024, offset=0, device="cpu")
    n = mask.numel()
    sigma = math.sqrt(n * p * (1 - p))
    assert abs(int(mask.sum()) - n * (1 - p)) < 6 * sigma


def test_threshold_edge_cases():
    assert dropout_threshold(0.0) == -(2**31) + 1
    assert dropout_threshold(1.0) == 2**31 - 1
    assert dropout_threshold(0.5) == 0
    with pytest.raises(ValueError):
        dropout_threshold(1.5)
    with pytest.raises(ValueError):
        dropout_threshold(-0.1)

    # p -> 1 keeps nothing: no int32 exceeds 2**31 - 1.
    assert not philox_mask(1, 2, 8, 33, 1.0, 5, 0, "cpu").any()
    # p -> 0 keeps everything except the single word 0x80000000 (never drawn at this size).
    assert philox_mask(1, 2, 8, 33, 0.0, 5, 0, "cpu").all()


def test_offset_and_seed_change_the_stream():
    base = philox_mask(1, 2, 8, 32, 0.5, 1, 0, "cpu")
    assert not torch.equal(base, philox_mask(1, 2, 8, 32, 0.5, 2, 0, "cpu"))
    assert not torch.equal(base, philox_mask(1, 2, 8, 32, 0.5, 1, 1000, "cpu"))
    # An offset of one counter shifts the stream by four elements (one counter = four slots).
    shifted = philox_mask(1, 1, 1, 32, 0.5, 1, 1, "cpu")
    assert torch.equal(
        shifted[..., :28], philox_mask(1, 1, 1, 32, 0.5, 1, 0, "cpu")[..., 4:]
    )

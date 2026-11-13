# ########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ########################################################################
"""
Performance model for hipblas-bench results: floating point operations and
bytes moved per call, arithmetic intensity, and roofline ceilings.

Byte counts assume every operand is read once and every output is written once
(an output that is also an input, like y in axpy or A in ger, is read and
written). FLOP counts follow clients/include/flops.hpp. Complex FLOPs are real
FLOPs (6 per complex multiply, 2 per complex add), so complex functions are
compared against the real peak of the same width.
"""

# compute_type -> (prefix, element size in bytes, is_complex, peak key)
PRECISIONS = {
    'f16_r':  ('h',  2, False, 'f16'),
    'bf16_r': ('bf', 2, False, 'bf16'),
    'f32_r':  ('s',  4, False, 'f32'),
    'f64_r':  ('d',  8, False, 'f64'),
    'f32_c':  ('c',  8, True,  'f32'),
    'f64_c':  ('z', 16, True,  'f64'),
}

PEAK_KEYS = ('f16', 'bf16', 'f32', 'f64')


def tri(n):
    return n * (n + 1) / 2


def _band(dim, k):
    """Elements in k off-diagonals of a dim x dim matrix."""
    k1 = min(k, dim)
    return k1 * dim - k1 * (k1 + 1) / 2


def _get(params, key, default=None):
    """Integer argument from a CSV row (column names as printed by hipblas-bench)."""
    for k in (key, key.lower(), key.upper()):
        if k in params and str(params[k]).strip() != '':
            return int(float(params[k]))
    if default is None:
        raise KeyError(key)
    return default


def _trans_is_n(params):
    return str(params.get('transA', 'N')).strip().upper() == 'N'


def _side_is_l(params):
    return str(params.get('side', 'L')).strip().upper() == 'L'


# Each model returns (flops, elements). elements x element size = bytes.
# c selects the complex FLOP count.

def _axpy(p, c):
    n = _get(p, 'N')
    return (8 if c else 2) * n, 3 * n

def _dot(p, c):
    n = _get(p, 'N')
    return (8 if c else 2) * n, 2 * n

def _dotc(p, c):
    n = _get(p, 'N')
    return (9 if c else 2) * n, 2 * n

def _scal(p, c):
    n = _get(p, 'N')
    return (6 if c else 1) * n, 2 * n

def _nrm2(p, c):
    n = _get(p, 'N')
    return (8 if c else 2) * n, n

def _iamax(p, c):
    n = _get(p, 'N')
    return n, n

def _asum(p, c):
    n = _get(p, 'N')
    return (4 if c else 2) * n, n

def _copy(p, c):
    n = _get(p, 'N')
    return n, 2 * n

def _swap(p, c):
    n = _get(p, 'N')
    return n, 4 * n

def _gemv(p, c):
    m, n = _get(p, 'M'), _get(p, 'N')
    x, y = (n, m) if _trans_is_n(p) else (m, n)
    flops = (8 * m * n + 6 * y) if c else (2 * m * n + 2 * y)
    return flops, m * n + x + 2 * y

def _gbmv(p, c):
    m, n = _get(p, 'M'), _get(p, 'N')
    kl, ku = _get(p, 'KL'), _get(p, 'KU')
    x, y = (n, m) if _trans_is_n(p) else (m, n)
    band = _band(x, kl) + _band(x, ku) + x
    return (8 if c else 2) * band, band + x + 2 * y

def _symv(p, c):
    n = _get(p, 'N')
    return (8 if c else 2) * (n * n + n), tri(n) + 3 * n

def _sbmv(p, c):
    n, k = _get(p, 'N'), _get(p, 'K')
    band = 2 * _band(n, k) + n
    return (8 if c else 2) * band, _band(n, k) + n + 3 * n

def _trmv(p, c):
    n = _get(p, 'M', _get(p, 'N', 0))
    return (4 if c else 1) * n * n, tri(n) + 2 * n

def _tbmv(p, c):
    n, k = _get(p, 'M', _get(p, 'N', 0)), _get(p, 'K')
    band = 2 * _band(n, k) + n
    return (4 if c else 1) * band, _band(n, k) + n + 2 * n

def _trsv(p, c):
    n = _get(p, 'N', _get(p, 'M', 0))
    return (4 if c else 1) * n * n, tri(n) + 2 * n

def _tbsv(p, c):
    n, k = _get(p, 'N', _get(p, 'M', 0)), _get(p, 'K')
    band = 2 * _band(n, k) + n
    return (4 if c else 1) * band, _band(n, k) + n + 2 * n

def _ger(p, c):
    m, n = _get(p, 'M'), _get(p, 'N')
    return (8 if c else 2) * m * n, 2 * m * n + m + n

def _syr(p, c):
    n = _get(p, 'N')
    return (4 if c else 1) * (n * n + 2 * n), 2 * tri(n) + n

def _syr2(p, c):
    n = _get(p, 'N')
    return (4 if c else 1) * (2 * n * n + 4 * n), 2 * tri(n) + 2 * n

def _her(p, c):
    n = _get(p, 'N')
    return 4 * n * n, 2 * tri(n) + n

def _her2(p, c):
    n = _get(p, 'N')
    return 8 * (n + 1) * n, 2 * tri(n) + 2 * n

def _gemm(p, c):
    m, n, k = _get(p, 'M'), _get(p, 'N'), _get(p, 'K')
    return (8 if c else 2) * m * n * k, m * k + k * n + 2 * m * n

def _symm(p, c):
    m, n = _get(p, 'M'), _get(p, 'N')
    ka = m if _side_is_l(p) else n
    return (8 if c else 2) * ka * m * n, tri(ka) + 3 * m * n

def _trmm(p, c):
    m, n = _get(p, 'M'), _get(p, 'N')
    ka = m if _side_is_l(p) else n
    return (4 if c else 1) * ka * m * n, tri(ka) + 2 * m * n

def _syrk(p, c):
    n, k = _get(p, 'N'), _get(p, 'K')
    return (4 if c else 1) * 2 * tri(n) * k, n * k + 2 * tri(n)

def _syrkx(p, c):
    n, k = _get(p, 'N'), _get(p, 'K')
    return (4 if c else 1) * 2 * tri(n) * k, 2 * n * k + 2 * tri(n)

def _syr2k(p, c):
    n, k = _get(p, 'N'), _get(p, 'K')
    return (4 if c else 1) * 4 * tri(n) * k, 2 * n * k + 2 * tri(n)


MODELS = {
    # BLAS1
    'axpy': _axpy, 'dot': _dot, 'dotu': _dot, 'dotc': _dotc, 'scal': _scal,
    'nrm2': _nrm2, 'iamax': _iamax, 'iamin': _iamax, 'asum': _asum,
    'copy': _copy, 'swap': _swap,
    # BLAS2
    'gemv': _gemv, 'gbmv': _gbmv,
    'symv': _symv, 'spmv': _symv, 'hemv': _symv, 'hpmv': _symv,
    'sbmv': _sbmv, 'hbmv': _sbmv,
    'trmv': _trmv, 'tpmv': _trmv, 'tbmv': _tbmv,
    'trsv': _trsv, 'tpsv': _trsv, 'tbsv': _tbsv,
    'ger': _ger, 'geru': _ger, 'gerc': _ger,
    'syr': _syr, 'spr': _syr, 'syr2': _syr2, 'spr2': _syr2,
    'her': _her, 'hpr': _her, 'her2': _her2, 'hpr2': _her2,
    # BLAS3
    'gemm': _gemm, 'symm': _symm, 'hemm': _symm, 'trmm': _trmm, 'trsm': _trmm,
    'syrk': _syrk, 'herk': _syrk, 'syr2k': _syr2k, 'her2k': _syr2k,
    'syrkx': _syrkx, 'herkx': _syrkx,
}


def has_model(func):
    return func in MODELS


def flops_bytes(func, compute_type, params):
    """(FLOPs, bytes) for one call, or None when the function or precision is not modeled."""
    if func not in MODELS or compute_type not in PRECISIONS:
        return None
    _, size, is_complex, _ = PRECISIONS[compute_type]
    try:
        flops, elements = MODELS[func](params, is_complex)
    except (KeyError, ValueError):
        return None
    return float(flops), float(elements) * size


def intensity(func, compute_type, params):
    """Arithmetic intensity in FLOP/byte, or None."""
    fb = flops_bytes(func, compute_type, params)
    if fb is None or fb[1] <= 0:
        return None
    return fb[0] / fb[1]


def ceiling_gflops(func, compute_type, params, gbps=None, peak_gflops=None):
    """
    Roofline ceiling in GFLOP/s: min(compute peak, intensity x bandwidth).
    Either term is dropped when its input is missing. Returns None when neither applies.
    """
    terms = []
    ai = intensity(func, compute_type, params)
    if gbps and ai is not None:
        terms.append(ai * gbps)
    if peak_gflops and compute_type in PRECISIONS:
        peak = peak_gflops.get(PRECISIONS[compute_type][3])
        if peak:
            terms.append(peak)
    return min(terms) if terms else None


def footprint_bytes(func, compute_type, params):
    fb = flops_bytes(func, compute_type, params)
    return None if fb is None else fb[1]


def parse_peak_gflops(text):
    """'f32=100000,f64=2000' -> {'f32': 100000.0, 'f64': 2000.0}"""
    peaks = {}
    if not text:
        return peaks
    for item in text.split(','):
        key, _, value = item.partition('=')
        key = key.strip()
        if key not in PEAK_KEYS:
            raise ValueError("unknown peak key '{}', expected one of {}".format(key, PEAK_KEYS))
        peaks[key] = float(value)
    return peaks

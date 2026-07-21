# Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import unittest
import math
import numpy as np

from rocrand import *


def make_test(BaseClass, subname, **kwargs):
    class Class(BaseClass):
        def setUp(self):
            super(Class, self).setUp()

    name = BaseClass.__name__ + subname
    globals()[name] = type(name, (Class,), kwargs)

class TestVersion(unittest.TestCase):
    def test_version(self):
        self.assertGreater(get_version(), 0)

class TestCtor(unittest.TestCase):
    def test_ctor(self):
        with self.assertRaises(RocRandError):
            PRNG(1234)
        with self.assertRaises(RocRandError):
            QRNG(1234)

class TestRNGBase(unittest.TestCase):
    rngtype = None

    def setUp(self):
        if not self.rngtype:
            self.skipTest("rngtype is not set")

class TestCtorPRNG(TestRNGBase):
    def test_ctor(self):
        PRNG(self.rngtype)
        PRNG(self.rngtype, seed=123456)
        PRNG(self.rngtype, offset=987654)
        PRNG(self.rngtype, seed=2345678, offset=7654)

make_test(TestCtorPRNG, "DEFAULT",         rngtype=PRNG.DEFAULT)
make_test(TestCtorPRNG, "LFSR113",         rngtype=PRNG.LFSR113)
make_test(TestCtorPRNG, "XORWOW",          rngtype=PRNG.XORWOW)
make_test(TestCtorPRNG, "MRG31K3P",        rngtype=PRNG.MRG31K3P)
make_test(TestCtorPRNG, "MRG32K3A",        rngtype=PRNG.MRG32K3A)
make_test(TestCtorPRNG, "PHILOX4_32_10",   rngtype=PRNG.PHILOX4_32_10)
make_test(TestCtorPRNG, "THREEFRY2_32_20", rngtype=PRNG.THREEFRY2_32_20)
make_test(TestCtorPRNG, "THREEFRY2_64_20", rngtype=PRNG.THREEFRY2_64_20)
make_test(TestCtorPRNG, "THREEFRY4_32_20", rngtype=PRNG.THREEFRY4_32_20)
make_test(TestCtorPRNG, "THREEFRY4_64_20", rngtype=PRNG.THREEFRY4_64_20)

class TestCtorPRNGMT(TestRNGBase):
    def test_ctor(self):
        PRNG(self.rngtype)
        PRNG(self.rngtype, seed=123456)
        with self.assertRaises(RocRandError):
            PRNG(self.rngtype, offset=987654)
        with self.assertRaises(RocRandError):
            PRNG(self.rngtype, seed=2345678, offset=7654)

make_test(TestCtorPRNGMT, "MTGP32",  rngtype=PRNG.MTGP32)
make_test(TestCtorPRNGMT, "MT19937", rngtype=PRNG.MT19937)

class TestCtorQRNG(TestRNGBase):
    def test_ctor(self):
        QRNG(self.rngtype)
        QRNG(self.rngtype, ndim=10)
        QRNG(self.rngtype, offset=987654)
        QRNG(self.rngtype, ndim=123, offset=7654)

make_test(TestCtorQRNG, "DEFAULT", rngtype=QRNG.DEFAULT)
make_test(TestCtorQRNG, "SOBOL32", rngtype=QRNG.SOBOL32)
make_test(TestCtorQRNG, "SCRAMBLED_SOBOL32", rngtype=QRNG.SCRAMBLED_SOBOL32)
make_test(TestCtorQRNG, "SOBOL64", rngtype=QRNG.SOBOL64)
make_test(TestCtorQRNG, "SCRAMBLED_SOBOL64", rngtype=QRNG.SCRAMBLED_SOBOL64)

class TestParamsPRNG(TestRNGBase):
    def setUp(self):
        super(TestParamsPRNG, self).setUp()
        self.rng = PRNG(self.rngtype)

    def tearDown(self):
        del self.rng

    def test_seed(self):
        self.assertIsNone(self.rng.seed)
        self.rng.seed = 0
        self.assertEqual(self.rng.seed, 0)
        self.rng.seed = 54654634456365
        self.assertEqual(self.rng.seed, 54654634456365)

    def test_offset(self):
        self.assertEqual(self.rng.offset, 0)
        self.rng.offset = 0
        self.assertEqual(self.rng.offset, 0)
        self.rng.offset = 2323423
        self.assertEqual(self.rng.offset, 2323423)

make_test(TestParamsPRNG, "DEFAULT",         rngtype=PRNG.DEFAULT)
make_test(TestParamsPRNG, "LFSR113",         rngtype=PRNG.LFSR113)
make_test(TestParamsPRNG, "XORWOW",          rngtype=PRNG.XORWOW)
make_test(TestParamsPRNG, "MRG31K3P",        rngtype=PRNG.MRG31K3P)
make_test(TestParamsPRNG, "MRG32K3A",        rngtype=PRNG.MRG32K3A)
make_test(TestParamsPRNG, "PHILOX4_32_10",   rngtype=PRNG.PHILOX4_32_10)
make_test(TestParamsPRNG, "THREEFRY2_32_20", rngtype=PRNG.THREEFRY2_32_20)
make_test(TestParamsPRNG, "THREEFRY2_64_20", rngtype=PRNG.THREEFRY2_64_20)
make_test(TestParamsPRNG, "THREEFRY4_32_20", rngtype=PRNG.THREEFRY4_32_20)
make_test(TestParamsPRNG, "THREEFRY4_64_20", rngtype=PRNG.THREEFRY4_64_20)

class TestParamsPRNGMT(TestRNGBase):
    def setUp(self):
        super(TestParamsPRNGMT, self).setUp()
        self.rng = PRNG(self.rngtype)

    def tearDown(self):
        del self.rng

    def test_seed(self):
        self.assertIsNone(self.rng.seed)
        self.rng.seed = 0
        self.assertEqual(self.rng.seed, 0)
        self.rng.seed = 54654634456365
        self.assertEqual(self.rng.seed, 54654634456365)

    def test_offset(self):
        self.assertEqual(self.rng.offset, 0)
        with self.assertRaises(RocRandError):
            self.rng.offset = 2323423

make_test(TestParamsPRNGMT, "MTGP32",  rngtype=PRNG.MTGP32)
make_test(TestParamsPRNGMT, "MT19937", rngtype=PRNG.MT19937)

class TestParamsQRNG(TestRNGBase):
    def setUp(self):
        super(TestParamsQRNG, self).setUp()
        self.rng = QRNG(self.rngtype)

    def tearDown(self):
        del self.rng

    def test_ndim(self):
        self.assertEqual(self.rng.ndim, 1)
        self.rng.ndim = 10
        self.assertEqual(self.rng.ndim, 10)
        self.rng.ndim = 123
        self.assertEqual(self.rng.ndim, 123)
        with self.assertRaises(RocRandError):
            self.rng.ndim = 0
        self.assertEqual(self.rng.ndim, 123)
        with self.assertRaises(RocRandError):
            self.rng.ndim = 30000
        self.assertEqual(self.rng.ndim, 123)

    def test_offset(self):
        self.assertEqual(self.rng.offset, 0)
        self.rng.offset = 0
        self.assertEqual(self.rng.offset, 0)
        self.rng.offset = 2323423
        self.assertEqual(self.rng.offset, 2323423)

make_test(TestParamsQRNG, "DEFAULT", rngtype=QRNG.DEFAULT)
make_test(TestParamsQRNG, "SOBOL32", rngtype=QRNG.SOBOL32)
make_test(TestParamsQRNG, "SCRAMBLED_SOBOL32", rngtype=QRNG.SCRAMBLED_SOBOL32)
make_test(TestParamsQRNG, "SOBOL64", rngtype=QRNG.SOBOL64)
make_test(TestParamsQRNG, "SCRAMBLED_SOBOL64", rngtype=QRNG.SCRAMBLED_SOBOL64)

OUTPUT_SIZE = 8192

class TestGenerate(TestRNGBase):
    def setUp(self):
        super(TestGenerate, self).setUp()
        self.rng = self.klass(self.rngtype, is_host=self.is_host)

    def tearDown(self):
        del self.rng

    def test_types(self):
        with self.assertRaises(TypeError):
            self.rng.uniform(np.empty(100, np.int8))
        with self.assertRaises(TypeError):
            self.rng.normal(np.empty(100, np.int8), 0.0, 1.0)
        with self.assertRaises(TypeError):
            self.rng.lognormal(np.empty(100, np.int8), 0.0, 1.0)
        with self.assertRaises(TypeError):
            self.rng.poisson(np.empty(100, np.float32), 100.0)

        self.rng.generate(np.empty(100, np.uint32))
        self.rng.generate(np.empty((10, 100), np.uint32))

        this_empty = np.empty if self.is_host else empty
        self.rng.uniform(this_empty(100, np.float32))
        self.rng.uniform(this_empty((10, 100), np.float32))

    def test_generate_uint32(self):
        output = np.empty(OUTPUT_SIZE, np.uint32)
        self.rng.generate(output)

        output = output.astype(np.float64)
        output /= 4294967295.0

        self.assertAlmostEqual(output.mean(), 0.5, delta=0.2)
        self.assertAlmostEqual(output.std(), pow(1 / 12.0, 0.5), delta=0.2 * pow(1 / 12.0, 0.5))

    def test_generate_int32(self):
        output = np.empty(OUTPUT_SIZE, np.int32)
        self.rng.generate(output)

        output = output.astype(np.float64)
        output /= 4294967295.0

        self.assertAlmostEqual(output.mean(), 0.0, delta=0.2)
        self.assertAlmostEqual(output.std(), pow(1 / 12.0, 0.5), delta=0.2 * pow(1 / 12.0, 0.5))

    def _test_uniform(self, dtype):
        output = np.empty(OUTPUT_SIZE, dtype)
        self.rng.uniform(output)

        self.assertAlmostEqual(output.mean(), 0.5, delta=0.2)
        self.assertAlmostEqual(output.std(), math.sqrt(1 / 12.0), delta=0.2 * math.sqrt(1 / 12.0))

    def test_uniform_float(self):
        self._test_uniform(np.float32)

    def test_uniform_double(self):
        self._test_uniform(np.float64)

    def _test_normal_real(self, dtype):
        output = np.empty(OUTPUT_SIZE, dtype)
        self.rng.normal(output, 0.0, 1.0)

        self.assertAlmostEqual(output.mean(), 0.0, delta=0.2)
        self.assertAlmostEqual(output.std(), 1.0, delta=0.2)

    def test_normal_float(self):
        self._test_normal_real(np.float32)

    def test_normal_double(self):
        self._test_normal_real(np.float64)

    def _test_lognormal_real(self, dtype):
        output = np.empty(OUTPUT_SIZE, dtype)
        self.rng.lognormal(output, 1.6, 0.25)

        mean = output.mean()
        stddev = output.std()
        logmean = math.log(mean * mean / math.sqrt(stddev + mean * mean))
        logstd = math.sqrt(math.log(1.0 + stddev / (mean * mean)))

        self.assertAlmostEqual(logmean, 1.6, delta=1.6 * 0.2)
        self.assertAlmostEqual(logstd, 0.25, delta=0.25 * 0.2)

    def test_lognormal_float(self):
        self._test_lognormal_real(np.float32)

    def test_lognormal_double(self):
        self._test_lognormal_real(np.float64)

    def test_poisson(self):
        if self.rngtype == PRNG.THREEFRY2_64_20 or self.rngtype == PRNG.THREEFRY4_64_20:
            self.skipTest("64-bits Threefry cannot generate 32-bits Poisson numbers")
        for lambda_value in [1.0, 5.5, 20.0, 100.0, 1234.5, 5000.0]:
            output = np.empty(OUTPUT_SIZE, np.uint32)
            self.rng.poisson(output, lambda_value)

            self.assertAlmostEqual(output.mean(), lambda_value, delta=max(1.0, lambda_value * 1e-1))
            self.assertAlmostEqual(output.var(), lambda_value, delta=max(1.0, lambda_value * 1e-1))

    def test_size(self):
        output = np.full(OUTPUT_SIZE * 2, 10.0, dtype=np.float64)
        self.rng.uniform(output, size=OUTPUT_SIZE)

        self.assertTrue((output[:OUTPUT_SIZE] <= 1.0).all())
        self.assertTrue((output[OUTPUT_SIZE:] == 10.0).all())

make_test(TestGenerate, "PRNG" + "DEFAULT",                klass=PRNG, rngtype=PRNG.DEFAULT,            is_host=False)
make_test(TestGenerate, "PRNG" + "XORWOW",                 klass=PRNG, rngtype=PRNG.XORWOW,             is_host=False)
make_test(TestGenerate, "PRNG" + "MRG31K3P",               klass=PRNG, rngtype=PRNG.MRG31K3P,           is_host=False)
make_test(TestGenerate, "PRNG" + "MRG31K3P_host",          klass=PRNG, rngtype=PRNG.MRG31K3P,           is_host=True)
make_test(TestGenerate, "PRNG" + "MRG32K3A",               klass=PRNG, rngtype=PRNG.MRG32K3A,           is_host=False)
make_test(TestGenerate, "PRNG" + "MTGP32",                 klass=PRNG, rngtype=PRNG.MTGP32,             is_host=False)
make_test(TestGenerate, "PRNG" + "MT19937",                klass=PRNG, rngtype=PRNG.MT19937,            is_host=False)
make_test(TestGenerate, "PRNG" + "PHILOX4_32_10",          klass=PRNG, rngtype=PRNG.PHILOX4_32_10,      is_host=False)
make_test(TestGenerate, "PRNG" + "PHILOX4_32_10_host",     klass=PRNG, rngtype=PRNG.PHILOX4_32_10,      is_host=True)
make_test(TestGenerate, "QRNG" + "DEFAULT",                klass=QRNG, rngtype=QRNG.DEFAULT,            is_host=False)
make_test(TestGenerate, "QRNG" + "SOBOL32",                klass=QRNG, rngtype=QRNG.SOBOL32,            is_host=False)
make_test(TestGenerate, "QRNG" + "SOBOL32_host",           klass=QRNG, rngtype=QRNG.SOBOL32,            is_host=True)
make_test(TestGenerate, "QRNG" + "SCRAMBLED_SOBOL32",      klass=QRNG, rngtype=QRNG.SCRAMBLED_SOBOL32,  is_host=False)
make_test(TestGenerate, "QRNG" + "SCRAMBLED_SOBOL32_host", klass=QRNG, rngtype=QRNG.SCRAMBLED_SOBOL32,  is_host=True)
make_test(TestGenerate, "QRNG" + "SOBOL64",                klass=QRNG, rngtype=QRNG.SOBOL64,            is_host=False)
make_test(TestGenerate, "QRNG" + "SOBOL64_host",           klass=QRNG, rngtype=QRNG.SOBOL64,            is_host=True)
make_test(TestGenerate, "QRNG" + "SCRAMBLED_SOBOL64",      klass=QRNG, rngtype=QRNG.SCRAMBLED_SOBOL64,  is_host=False)
make_test(TestGenerate, "QRNG" + "SCRAMBLED_SOBOL64_host", klass=QRNG, rngtype=QRNG.SCRAMBLED_SOBOL64,  is_host=True)
make_test(TestGenerate, "PRNG" + "LFSR113",                klass=PRNG, rngtype=PRNG.LFSR113,            is_host=False)
make_test(TestGenerate, "PRNG" + "THREEFRY2_32_20",        klass=PRNG, rngtype=PRNG.THREEFRY2_32_20,    is_host=False)
make_test(TestGenerate, "PRNG" + "THREEFRY2_64_20",        klass=PRNG, rngtype=PRNG.THREEFRY2_64_20,    is_host=False)
make_test(TestGenerate, "PRNG" + "THREEFRY4_32_20",        klass=PRNG, rngtype=PRNG.THREEFRY4_32_20,    is_host=False)
make_test(TestGenerate, "PRNG" + "THREEFRY4_64_20",        klass=PRNG, rngtype=PRNG.THREEFRY4_64_20,    is_host=False)


if __name__ == "__main__":
    unittest.main()

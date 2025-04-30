# ##########################################################################
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
# ##########################################################################

import collections
import os
import re
import shlex
import unittest
from subprocess import Popen, PIPE

def call_rocsolver_bench(*args):
    cmd = ['./rocsolver-bench']
    for arg in args:
        if isinstance(arg, str):
            cmd.extend(shlex.split(arg, False, False))
        elif isinstance(arg, collections.Sequence):
            cmd.extend(arg)
        else:
            cmd.push(str(arg))
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    return (str(stdout, encoding='utf-8', errors='surrogateescape'),
            str(stderr, encoding='utf-8', errors='surrogateescape'),
            process.returncode)

class TestRocsolverBench(unittest.TestCase):
    def parse_arguments(self, bench_output):
        m = re.search(r"\n=+\nArguments:\s*\n=+\n(?P<arg_names>.*)\n(?P<arg_values>.*)\n",
                bench_output, re.MULTILINE)
        self.assertTrue(m)
        arg_names = m.group('arg_names').split()
        arg_values = m.group('arg_values').split()
        self.assertEqual(len(arg_names), len(arg_values))
        return dict(zip(arg_names, arg_values))

    def parse_results(self, bench_output):
        m = re.search(r"\n=+\nResults:\s*\n=+\n(?P<arg_names>.*)\n(?P<arg_values>.*)\n",
                bench_output, re.MULTILINE)
        self.assertTrue(m)
        arg_names = m.group('arg_names').split()
        arg_values = m.group('arg_values').split()
        self.assertEqual(len(arg_names), len(arg_values))
        return dict(zip(arg_names, arg_values))

    def test_help(self):
        out, err, exitcode = call_rocsolver_bench('--help')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)
        self.assertNotEqual(out, '')

    def test_validate_precision(self):
        for precision in 'sdcz':
            with self.subTest(precision=precision):
                out, err, exitcode = call_rocsolver_bench(f'-f gels -r {precision} -n 10 -m 15')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f gels -r 0 -n 10 -m 15')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_operation(self):
        for trans in 'NTC':
            with self.subTest(trans=trans):
                out, err, exitcode = call_rocsolver_bench(f'-f gels --trans {trans} -n 10 -m 15')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f gels --trans 0 -n 10 -m 15')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_side(self):
        for side in 'LRB':
            with self.subTest(side=side):
                out, err, exitcode = call_rocsolver_bench(f'-f larf --side {side} -n 10 -m 15')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f larf --side 0 -n 10 -m 15')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_fill(self):
        for fill in 'ULF':
            with self.subTest(fill=fill):
                out, err, exitcode = call_rocsolver_bench(f'-f bdsqr --uplo {fill} -n 15 --nu 10 --nv 10')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f bdsqr --uplo 0 -n 15 --nu 10 --nv 10')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_diagonal(self):
        for diag in 'NU':
            with self.subTest(diag=diag):
                out, err, exitcode = call_rocsolver_bench(f'-f trtri --diag {diag} -n 10')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f trtri --diag 0 -n 10')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_direct(self):
        for direct in 'FB':
            with self.subTest(direct=direct):
                out, err, exitcode = call_rocsolver_bench(f'-f larfb --direct {direct} --side L --storev C -n 10 -m 15 -k 1')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f larfb --direct 0 --side L --storev C -k 1 -n 10 -m 15 -k 1')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_storev(self):
        for storev in 'CR':
            with self.subTest(storev=storev):
                out, err, exitcode = call_rocsolver_bench(f'-f larft --storev {storev} -n 10 -k 1')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f larft --storev 0 -n 10 -k 1')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_left_svect(self):
        for svect in 'ASON':
            with self.subTest(svect=svect):
                out, err, exitcode = call_rocsolver_bench(f'-f gesvd --left_svect {svect} -n 10 -m 15')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f gesvd --left_svect 0 -n 10 -m 15')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_right_svect(self):
        for svect in 'ASON':
            with self.subTest(svect=svect):
                out, err, exitcode = call_rocsolver_bench(f'-f gesvd --right_svect {svect} -n 10 -m 15')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f gesvd --right_svect 0 -n 10 -m 15')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_workmode(self):
        for workmode in 'IO':
            with self.subTest(workmode=workmode):
                out, err, exitcode = call_rocsolver_bench(f'-f gesvd --fast_alg {workmode} -n 10 -m 15')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f gesvd --fast_alg 0 -n 10 -m 15')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_evect(self):
        for evect in 'VIN':
            with self.subTest(evect=evect):
                out, err, exitcode = call_rocsolver_bench(f'-f syev --evect {evect} -n 10')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f syev --evect 0 -n 10')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_erange(self):
        for erange in 'AVI':
            with self.subTest(erange=erange):
                out, err, exitcode = call_rocsolver_bench(f'-f stebz --erange {erange} -n 10')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f stebz --erange 0 -n 10')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_eorder(self):
        for eorder in 'BE':
            with self.subTest(eorder=eorder):
                out, err, exitcode = call_rocsolver_bench(f'-f stebz --eorder {eorder} -n 10')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f stebz --eorder 0 -n 10')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_itype(self):
        for itype in '123':
            with self.subTest(itype=itype):
                out, err, exitcode = call_rocsolver_bench(f'-f sygv --itype {itype} -n 10')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f sygv --itype 0 -n 10')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_unused_arg(self):
        out, err, exitcode = call_rocsolver_bench('-f gels --itype 1 -n 10 -m 15')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_perf_returns_number(self):
        out, err, exitcode = call_rocsolver_bench('-f gels -n 10 -m 15 --perf 1')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)
        self.assertGreaterEqual(float(out), 0)

def generate_parameterized_test(command_options, expected_args):
    def test_function_output(self):
        out, err, exitcode = call_rocsolver_bench(command_options)
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time_us']), 0)
        self.assertGreaterEqual(float(results['gpu_time_us']), 0)
    return test_function_output

parameters = [
    (
        'laswp',
        '-f laswp -n 10 --k1 15 --k2 20',
        {
            'n': '10',
            'lda': '20',
            'k1': '15',
            'k2': '20',
            'inc': '1',
        }
    ),
    (
        'larfg',
        '-f larfg -n 10',
        {
            'n': '10',
            'inc': '1',
        }
    ),
    (
        'larf',
        '-f larf -m 10 --side L',
        {
            'side': 'L',
            'm': '10',
            'n': '10',
            'inc': '1',
            'lda': '10',
        }
    ),
    (
        'larft',
        '-f larft -n 10 --storev C -k 5',
        {
            'direct': 'F',
            'storev': 'C',
            'n': '10',
            'k': '5',
            'ldv': '10',
            'ldt': '5',
        }
    ),
    (
        'larfb',
        '-f larfb -n 10 -m 15 --direct F --side L --storev C -k 5',
        {
            'side': 'L',
            'trans': 'N',
            'direct': 'F',
            'storev': 'C',
            'm': '15',
            'n': '10',
            'k': '5',
            'ldv': '15',
            'ldt': '5',
            'lda': '15',
        }
    ),
    (
        'latrd',
        '-f latrd -n 10 -k 5',
        {
            'uplo': 'U',
            'n': '10',
            'k': '5',
            'lda': '10',
            'ldw': '10',
        }
    ),
    (
        'labrd',
        '-f labrd -n 10 -m 15',
        {
            'm': '15',
            'n': '10',
            'nb': '10',
            'lda': '15',
            'ldx': '15',
            'ldy': '10',
        }
    ),
    (
        'bdsqr',
        '-f bdsqr --uplo U -n 15 --nu 10 --nv 10',
        {
            'uplo': 'U',
            'n': '15',
            'nv': '10',
            'nu': '10',
            'nc': '0',
            'ldv': '15',
            'ldu': '10',
            'ldc': '1',
        }
    ),
    (
        'steqr',
        '-f steqr -n 15',
        {
            'evect': 'N',
            'n': '15',
            'ldc': '15',
        }
    ),
    (
        'stedc',
        '-f stedc -n 15',
        {
            'evect': 'N',
            'n': '15',
            'ldc': '15',
        }
    ),
    (
        'stein',
        '-f stein -n 15',
        {
            'n': '15',
            'nev': '5',
            'ldz': '15',
        }
    ),
    (
        'lasyf',
        '-f lasyf -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'nb': '10',
            'lda': '10',
        }
    ),
    (
        'potf2',
        '-f potf2 -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
        }
    ),
    (
        'potf2_batched',
        '-f potf2_batched -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
            'batch_c': '1',
        }
    ),
    (
        'potf2_strided_batched',
        '-f potf2_strided_batched -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'batch_c': '1',
        }
    ),
    (
        'potrf',
        '-f potrf -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
        }
    ),
    (
        'potrf_batched',
        '-f potrf_batched -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
            'batch_c': '1',
        }
    ),
    (
        'potrf_strided_batched',
        '-f potrf_strided_batched -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'batch_c': '1',
        }
    ),
    (
        'potrs',
        '-f potrs -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
        }
    ),
    (
        'potrs_batched',
        '-f potrs_batched -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
            'batch_c': '1',
        }
    ),
    (
        'potrs_strided_batched',
        '-f potrs_strided_batched -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
            'strideA': '100',
            'strideB': '100',
            'batch_c': '1',
        }
    ),
    (
        'posv',
        '-f posv -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
        }
    ),
    (
        'posv_batched',
        '-f posv_batched -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
            'batch_c': '1',
        }
    ),
    (
        'posv_strided_batched',
        '-f posv_strided_batched -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
            'strideA': '100',
            'strideB': '100',
            'batch_c': '1',
        }
    ),
    (
        'potri',
        '-f potri -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
        }
    ),
    (
        'potri_batched',
        '-f potri_batched -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
            'batch_c': '1',
        }
    ),
    (
        'potri_strided_batched',
        '-f potri_strided_batched -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'batch_c': '1',
        }
    ),
    (
        'getf2_npvt',
        '-f getf2_npvt -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
        }
    ),
    (
        'getf2_npvt_batched',
        '-f getf2_npvt_batched -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
            'batch_c': '1',
        }
    ),
    (
        'getf2_npvt_strided_batched',
        '-f getf2_npvt_strided_batched -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'batch_c': '1',
        }
    ),
    (
        'getrf_npvt',
        '-f getrf_npvt -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
        }
    ),
    (
        'getrf_npvt_batched',
        '-f getrf_npvt_batched -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
            'batch_c': '1',
        }
    ),
    (
        'getrf_npvt_strided_batched',
        '-f getrf_npvt_strided_batched -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'batch_c': '1',
        }
    ),
    (
        'getrf',
        '-f getrf -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
        }
    ),
    (
        'getrf_batched',
        '-f getrf_batched -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'getrf_strided_batched',
        '-f getrf_strided_batched -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'getf2',
        '-f getf2 -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
        }
    ),
    (
        'getf2_batched',
        '-f getf2_batched -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'getf2_strided_batched',
        '-f getf2_strided_batched -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'geqr2',
        '-f geqr2 -n 10 -m 15',
        {
            'm': '15',
            'n': '10',
            'lda': '15',
        }
    ),
    (
        'geqr2_batched',
        '-f geqr2_batched -n 10 -m 15',
        {
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'geqr2_strided_batched',
        '-f geqr2_strided_batched -n 10 -m 15',
        {
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideA': '150',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'geqrf',
        '-f geqrf -n 10 -m 15',
        {
            'm': '15',
            'n': '10',
            'lda': '15',
        }
    ),
    (
        'geqrf_batched',
        '-f geqrf_batched -n 10 -m 15',
        {
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'geqrf_strided_batched',
        '-f geqrf_strided_batched -n 10 -m 15',
        {
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideA': '150',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'geqrf_ptr_batched',
        '-f geqrf_ptr_batched -n 10 -m 15',
        {
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'gerq2',
        '-f gerq2 -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
        }
    ),
    (
        'gerq2_batched',
        '-f gerq2_batched -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'gerq2_strided_batched',
        '-f gerq2_strided_batched -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'gerqf',
        '-f gerqf -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
        }
    ),
    (
        'gerqf_batched',
        '-f gerqf_batched -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'gerqf_strided_batched',
        '-f gerqf_strided_batched -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'geql2',
        '-f geql2 -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
        }
    ),
    (
        'geql2_batched',
        '-f geql2_batched -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'geql2_strided_batched',
        '-f geql2_strided_batched -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'geqlf',
        '-f geqlf -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
        }
    ),
    (
        'geqlf_batched',
        '-f geqlf_batched -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'geqlf_strided_batched',
        '-f geqlf_strided_batched -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'gelq2',
        '-f gelq2 -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
        }
    ),
    (
        'gelq2_batched',
        '-f gelq2_batched -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'gelq2_strided_batched',
        '-f gelq2_strided_batched -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'gelqf',
        '-f gelqf -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
        }
    ),
    (
        'gelqf_batched',
        '-f gelqf_batched -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'gelqf_strided_batched',
        '-f gelqf_strided_batched -m 10',
        {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'getrs',
        '-f getrs -n 10',
        {
            'trans': 'N',
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
        }
    ),
    (
        'getrs_batched',
        '-f getrs_batched -n 10',
        {
            'trans': 'N',
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'getrs_strided_batched',
        '-f getrs_strided_batched -n 10',
        {
            'trans': 'N',
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
            'strideA': '100',
            'strideP': '10',
            'strideB': '100',
            'batch_c': '1',
        }
    ),
    (
        'gesv',
        '-f gesv -n 10',
        {
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
        }
    ),
    (
        'gesv_batched',
        '-f gesv_batched -n 10',
        {
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'gesv_strided_batched',
        '-f gesv_strided_batched -n 10',
        {
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
            'strideA': '100',
            'strideP': '10',
            'strideB': '100',
            'batch_c': '1',
        }
    ),
    (
        'gesvd',
        '-f gesvd -n 10 -m 15',
        {
            'left_svect': 'N',
            'right_svect': 'N',
            'm': '15',
            'n': '10',
            'lda': '15',
            'ldu': '15',
            'ldv': '10',
        }
    ),
    (
        'gesvd_batched',
        '-f gesvd_batched -n 10 -m 15',
        {
            'left_svect': 'N',
            'right_svect': 'N',
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideS': '10',
            'ldu': '15',
            'strideU': '225',
            'ldv': '10',
            'strideV': '100',
            'strideE': '9',
            'batch_c': '1',
        }
    ),
    (
        'gesvd_strided_batched',
        '-f gesvd_strided_batched -n 10 -m 15',
        {
            'left_svect': 'N',
            'right_svect': 'N',
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideA': '150',
            'strideS': '10',
            'ldu': '15',
            'strideU': '225',
            'ldv': '10',
            'strideV': '100',
            'strideE': '9',
            'batch_c': '1',
        }
    ),
    (
        'trtri',
        '-f trtri -n 10',
        {
            'uplo': 'U',
            'diag': 'N',
            'n': '10',
            'lda': '10',
        }
    ),
    (
        'trtri_batched',
        '-f trtri_batched -n 10',
        {
            'uplo': 'U',
            'diag': 'N',
            'n': '10',
            'lda': '10',
            'batch_c': '1',
        }
    ),
    (
        'trtri_strided_batched',
        '-f trtri_strided_batched -n 10',
        {
            'uplo': 'U',
            'diag': 'N',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'batch_c': '1',
        }
    ),
    (
        'getri',
        '-f getri -n 10',
        {
            'n': '10',
            'lda': '10',
        }
    ),
    (
        'getri_batched',
        '-f getri_batched -n 10',
        {
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'getri_strided_batched',
        '-f getri_strided_batched -n 10',
        {
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'getri_npvt',
        '-f getri_npvt -n 10',
        {
            'n': '10',
            'lda': '10',
        }
    ),
    (
        'getri_npvt_batched',
        '-f getri_npvt_batched -n 10',
        {
            'n': '10',
            'lda': '10',
            'batch_c': '1',
        }
    ),
    (
        'getri_npvt_strided_batched',
        '-f getri_npvt_strided_batched -n 10',
        {
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'batch_c': '1',
        }
    ),
    (
        'getri_outofplace',
        '-f getri_outofplace -n 10',
        {
            'n': '10',
            'lda': '10',
            'ldc': '10',
        }
    ),
    (
        'getri_outofplace_batched',
        '-f getri_outofplace_batched -n 10',
        {
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'ldc': '10',
            'batch_c': '1',
        }
    ),
    (
        'getri_outofplace_strided_batched',
        '-f getri_outofplace_strided_batched -n 10',
        {
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'ldc': '10',
            'strideC': '100',
            'batch_c': '1',
        }
    ),
    (
        'getri_npvt_outofplace',
        '-f getri_npvt_outofplace -n 10',
        {
            'n': '10',
            'lda': '10',
            'ldc': '10',
        }
    ),
    (
        'getri_npvt_outofplace_batched',
        '-f getri_npvt_outofplace_batched -n 10',
        {
            'n': '10',
            'lda': '10',
            'ldc': '10',
            'batch_c': '1',
        }
    ),
    (
        'getri_npvt_outofplace_strided_batched',
        '-f getri_npvt_outofplace_strided_batched -n 10',
        {
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'ldc': '10',
            'strideC': '100',
            'batch_c': '1',
        }
    ),
    (
        'gels',
        '-f gels -n 10 -m 15',
        {
            'trans': 'N',
            'm': '15',
            'n': '10',
            'nrhs': '10',
            'lda': '15',
            'ldb': '15',
        }
    ),
    (
        'gels_batched',
        '-f gels_batched -n 10 -m 15',
        {
            'trans': 'N',
            'm': '15',
            'n': '10',
            'nrhs': '10',
            'lda': '15',
            'ldb': '15',
            'batch_c': '1',
        }
    ),
    (
        'gels_strided_batched',
        '-f gels_strided_batched -n 10 -m 15',
        {
            'trans': 'N',
            'm': '15',
            'n': '10',
            'nrhs': '10',
            'lda': '15',
            'ldb': '15',
            'strideA': '150',
            'strideB': '150',
            'batch_c': '1',
        }
    ),
    (
        'gebd2',
        '-f gebd2 -n 10 -m 15',
        {
            'm': '15',
            'n': '10',
            'lda': '15',
        }
    ),
    (
        'gebd2_batched',
        '-f gebd2_batched -n 10 -m 15',
        {
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'gebd2_strided_batched',
        '-f gebd2_strided_batched -n 10 -m 15',
        {
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideA': '150',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'gebrd',
        '-f gebrd -n 10 -m 15',
        {
            'm': '15',
            'n': '10',
            'lda': '15',
        }
    ),
    (
        'gebrd_batched',
        '-f gebrd_batched -n 10 -m 15',
        {
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'gebrd_strided_batched',
        '-f gebrd_strided_batched -n 10 -m 15',
        {
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideA': '150',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'sytf2',
        '-f sytf2 -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
        }
    ),
    (
        'sytf2_batched',
        '-f sytf2_batched -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'sytf2_strided_batched',
        '-f sytf2_strided_batched -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'sytrf',
        '-f sytrf -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
        }
    ),
    (
        'sytrf_batched',
        '-f sytrf_batched -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
    ),
    (
        'sytrf_strided_batched',
        '-f sytrf_strided_batched -n 10',
        {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
    )
]

if __name__ == '__main__':
    for name, command_options, expected_args in parameters:
        test = generate_parameterized_test(command_options, expected_args)
        setattr(TestRocsolverBench, f'test_{name}', test)
    unittest.main()

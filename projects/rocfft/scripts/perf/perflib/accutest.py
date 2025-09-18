# Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
"""rocFFT accuracy tests launch utils."""

import logging
import pathlib
import subprocess


def get_active_tests_tokens(accutest):
    """Run rocfft-test and fetch all active accuracy tests tokens"""

    cmd = [pathlib.Path(accutest).resolve(), '--gtest_list_tests']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    sub_str_1 = b'/accuracy_test.'
    sub_str_2 = b'DISABLED'
    sub_str_3 = b'vs_fftw/'
    sub_str_4 = b'#'

    test_tokens = set()

    get_token = False
    for line in proc.stdout:
        sub_str_1_found = True if line.find(sub_str_1) != -1 else False
        sub_str_2_found = True if line.find(sub_str_2) != -1 else False
        sub_str_3_found = True if line.find(sub_str_3) != -1 else False

        if sub_str_1_found and not sub_str_2_found:
            get_token = True
            continue
        elif sub_str_3_found and get_token:
            token = line.split(sub_str_3)
            token = token[1].split(sub_str_4)
            token = token[0].strip()

            test_tokens.add(token)
        else:
            get_token = False

    proc.stdout.close()
    ret_code = proc.wait()
    if (ret_code != 0):
        logging.warn(f'Unable to run accuracy tests. Return code: ' +
                     str(ret_code))

    return test_tokens

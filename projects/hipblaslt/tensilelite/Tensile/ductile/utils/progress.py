################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#
################################################################################
import sys


class _FallbackProgressBar:
    """Minimal manual progress bar for when tqdm is not installed.

    Supports the same interface as tqdm.tqdm(total=N):
        pbar = tqdm(total=100)
        pbar.update(10)
        pbar.close()

    Also works as a context manager.
    """
    def __init__(self, total=0):
        self.total = total
        self.n = 0

    def update(self, n=1):
        self.n += n
        pct = (self.n * 100 // self.total) if self.total else 0
        ticks = pct * 40 // 100
        sys.stdout.write(f"\r[{'#' * ticks}{' ' * (40 - ticks)}] {pct}%")
        sys.stdout.flush()

    def close(self):
        sys.stdout.write("\n")
        sys.stdout.flush()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


try:
    from tqdm import tqdm
except ImportError:
    tqdm = _FallbackProgressBar

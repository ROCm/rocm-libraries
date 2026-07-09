# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
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

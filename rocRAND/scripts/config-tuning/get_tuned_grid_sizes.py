#! /usr/bin/env python3

import re
import subprocess


def get_grid_sizes(rocminfo_output: str) -> str:
    match = re.search(r'^\s*Name:\s*gfx\d+.*?^\s*Compute Unit:\s*(\d+)',
                      rocminfo_output, flags=re.MULTILINE | re.DOTALL)
    if not match:
        raise Exception('Could not find Compute Unit info in rocminfo output')
    num_compute_units = int(match.group(1))
    compute_unit_multipliers = [4, 5, 8, 10, 16, 32]
    min_grid_size = 128
    max_grid_size = 4096

    grid_sizes = [128, 256, 512, 1024, 2048]
    for cu_mul in compute_unit_multipliers:
        new_grid_size = cu_mul * num_compute_units
        if new_grid_size >= min_grid_size and new_grid_size <= max_grid_size:
            grid_sizes.append(new_grid_size)
    grid_sizes = list(set(grid_sizes))  # Unique
    grid_sizes.sort()
    return ', '.join(str(i) for i in grid_sizes)


if __name__ == '__main__':
    rocminfo_out = subprocess.check_output('rocminfo', encoding='utf-8')
    print(get_grid_sizes(rocminfo_out))

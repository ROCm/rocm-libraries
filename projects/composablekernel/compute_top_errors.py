#!/usr/bin/env python3

import re
from collections import defaultdict

input_file = 'build/testres.txt'
output_file = 'build/error_stats_top10_py.txt'

max_errors = []
avg_errors = []
avg_errors_sum = 0.0

# regex patterns
errors_re = re.compile(r'Max error:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?),\s*Average error:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)')

with open(input_file) as f:
    current_test = 0
    for line in f:
        m = errors_re.search(line)
        if m:
            max_err = float(m.group(1))
            avg_err = float(m.group(2))
            # store/overwrite
            max_errors.append(max_err)
            avg_errors.append(avg_err)
            current_test += 1
            avg_errors_sum += avg_err

# sort and take top 10
max_errors.sort()
avg_errors.sort()

with open(output_file, 'w') as out:
    out.write('Top 10 tests by Max Error:\n')
    for err in max_errors[-10:]:
        out.write(f'{err}\n')
    out.write('\n')
    out.write('Top 10 tests by Average Error:\n')
    for err in avg_errors[-10:]:
        out.write(f'{err}\n')

    out.write('\n')
    out.write(f'Average of Average Errors across all tests: {avg_errors_sum / current_test}\n')

print(f'Results written to {output_file}')

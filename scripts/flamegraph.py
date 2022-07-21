#!/usr/bin/env python3
"""Run an executable and generate a flame graph.

Usage:

  flamegraph.py [-F frequency] [-o output] command args...

Default frequency is 99.  Default output is 'kernel.svg'.

"""

import argparse
import pathlib
import subprocess
import tempfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate flame graph.')
    parser.add_argument('-x', '--xdg-open', default=False, action='store_true', help='Open with xdg-open after generating')
    parser.add_argument('-o', '--output', default="kernel.svg", help='Output file name')
    parser.add_argument('-F', '--frequency', type=int, default=99, help='Sampling frequency')
    parser.add_argument('command', nargs=argparse.REMAINDER)

    args = parser.parse_args()
    svg = pathlib.Path(args.output)

    with tempfile.TemporaryDirectory() as dname:
        tmp = pathlib.Path(dname)
        data = tmp / 'perf.data'
        script = tmp / 'out.perf'
        folded = tmp / 'out.folded'

        subprocess.run(['/usr/bin/perf', 'record', '-F', str(args.frequency), '-g', '-o', str(data)] + args.command)
        with script.open('w') as f:
            subprocess.run(['/usr/bin/perf', 'script', '-i', str(data)], stdout=f)
        with folded.open('w') as f:
            subprocess.run(['stackcollapse-perf.pl', str(script)], stdout=f)
        with svg.open('w') as f:
            subprocess.run(['flamegraph.pl', str(folded)], stdout=f)

    if args.xdg_open:
        subprocess.call(['xdg-open', str(svg)])

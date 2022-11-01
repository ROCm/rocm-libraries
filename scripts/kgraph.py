#!/usr/bin/env python3
"""Load kernel graph from the meta data of an assembly file and render it.

Usage:

  kgraph.py path/to/assembly.s

this creates a PDF file at path/to/assembly.pdf.  To automatically
open it (via xdg-open), pass '-x'.

"""

import argparse
import pathlib
import subprocess
import yaml


def extract_dot(path):
    """Extract .kernel_graph meta data from assembly file."""
    source = path.read_text()
    start, end = source.find('---\n') + 4, source.find('...')
    meta = yaml.safe_load(source[start:end])
    kernel = meta['amdhsa.kernels'][0]
    return kernel['.name'], kernel['.kernel_graph']


def write_dot(dot, fname):
    out_fname = fname.with_suffix(".dot")
    out_fname.write_text(dot)
    return out_fname


def render_dot(fname, dot):
    """Render graph."""
    with fname.open('w') as out:
        subprocess.run(['dot', '-Tpdf'], input=dot.encode(), stdout=out)


def open_dot(fname):
    subprocess.call(['xdg-open', str(fname)])


def open_code(fname):
    subprocess.call(['code', str(fname)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and render kernel graph.')
    parser.add_argument('fname', type=str, help='Assembly file name')
    parser.add_argument('-d', '--dot-only', default=False, action="store_true")
    parser.add_argument('-x', '--xdg-open', default=False, action='store_true', help='Open with xdg-open after generating')
    parser.add_argument('-c', '--code-open', default=False, action='store_true', help='Open .dot with VSCode after generating')

    args = parser.parse_args()
    path = pathlib.Path(args.fname)

    _, dot = extract_dot(path)
    dotfile = write_dot(dot, path)

    if args.code_open:
        open_code(dotfile)

    if not args.dot_only:
        rendered_fname = path.with_suffix(".pdf")
        render_dot(rendered_fname, dot)
        if args.xdg_open:
            open_dot(rendered_fname)

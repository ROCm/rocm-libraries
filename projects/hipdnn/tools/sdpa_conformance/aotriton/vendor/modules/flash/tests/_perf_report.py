# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Headless fallback for `triton.testing.Mark.run`.

`Mark._run` does `import matplotlib.pyplot` at the top of the function, before
it looks at `save_path` -- so a benchmark cannot be run at all without
matplotlib, even when no plot is wanted. That makes performance_forward.py and
performance_backward.py unrunnable on a machine that only has the build venv,
which is most of the machines these get run on.

`run_report` prefers the real thing (so `SAVE_PLOT=.` still draws the plot and
writes the CSV exactly as before) and falls back to walking the same
`Mark.benchmarks` structure and printing the same numbers when the import
fails. The fallback deliberately reuses `mark.fn`, `bench.x_vals`,
`bench.line_vals` and `bench.line_arg` rather than reimplementing the sweep, so
the two paths cannot disagree about WHAT is measured -- only about whether a
picture comes out of it.
"""


def _print_table(mark):
    for bench in mark.benchmarks:
        print(f'\n### {bench.plot_name}')
        x_name = bench.x_names[0]
        widths = [max(len(x_name), 8)] + [max(len(n), 12) for n in bench.line_names]
        header = [x_name] + list(bench.line_names)
        print('  '.join(f'{h:>{w}}' for h, w in zip(header, widths)))
        for x in bench.x_vals:
            cells = [f'{x:>{widths[0]}}']
            for i, y in enumerate(bench.line_vals):
                ret = mark.fn(**{x_name: x}, **{bench.line_arg: y}, **bench.args)
                # Mark._run accepts a scalar or a (mean, min, max) triple.
                value = ret[0] if isinstance(ret, (tuple, list)) else ret
                cells.append(f'{value:>{widths[i + 1]}.4f}')
            print('  '.join(cells), flush=True)


def run_report(mark, save_path=None, print_data=True):
    """`mark.run(...)` when matplotlib is importable, a printed table otherwise."""
    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError:
        if save_path:
            print('_perf_report: matplotlib is not installed, so no plot will be '
                  f'written to {save_path!r}; printing the table instead.')
        _print_table(mark)
        return None
    return mark.run(save_path=save_path, print_data=print_data)

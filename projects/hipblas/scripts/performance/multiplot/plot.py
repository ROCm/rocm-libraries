#!/usr/bin/env python3

import argparse
import collections
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker
import numpy as np
import yaml
import sys
from pathlib import Path
from typing import List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import perf_model

# A result whose bytes moved are below this many times the L2 size is flagged as cache-resident.
CACHE_RESIDENT_FACTOR = 4


def machine_ceilings(machine_spec_dict, row=None):
    """
    (bandwidth GB/s, {peak key: GFLOP/s}) after clock and partition scaling. With per-result clocks
    (benchmark.py --auto-clock-scale) the clocks reported with row scale the published values;
    otherwise, or for a row without clocks, the run-wide scales in machine_spec.yaml apply.
    """
    gbps = machine_spec_dict.get('GBps_effective', machine_spec_dict.get('GBps'))
    peaks = machine_spec_dict.get('peak_gflops_effective') or None
    if row is None:
        return gbps, peaks
    partitions = machine_spec_dict.get('partitions') or 1
    sclk, mclk = perf_model.result_clocks(row)
    sclk_peak = machine_spec_dict.get('sclk_peak_mhz')
    if machine_spec_dict.get('per_result_sclk') and sclk and sclk_peak and machine_spec_dict.get('peak_gflops'):
        peaks = {k: v * sclk / sclk_peak / partitions for k, v in machine_spec_dict['peak_gflops'].items()}
    mclk_max = machine_spec_dict.get('mclk_max_mhz')
    if machine_spec_dict.get('per_result_mclk') and mclk and mclk_max and machine_spec_dict.get('GBps'):
        gbps = machine_spec_dict['GBps'] * mclk / mclk_max / partitions
    return gbps, peaks


def point_ceiling(funcname, prec, function_label, row, machine_spec_dict):
    """Roofline ceiling for one result; falls back to the per-function constant in machine_spec.yaml."""
    gbps, peaks = machine_ceilings(machine_spec_dict, row)
    if row is not None:
        ceiling = perf_model.ceiling_gflops(funcname, prec, row, gbps, peaks)
        if ceiling:
            return ceiling
    return machine_spec_dict.get(function_label)


def is_cache_resident(funcname, prec, row, machine_spec_dict):
    """
    True for a memory-bound result whose bytes moved fit in cache, so it measured cache rather than
    memory bandwidth. Compute-bound results are never flagged: their ceiling is the compute peak, and
    reusing data in cache is how they reach it.
    """
    l2_mb = machine_spec_dict.get('l2_mb')
    if not l2_mb or row is None:
        return False
    gbps, peaks = machine_ceilings(machine_spec_dict, row)
    ai = perf_model.intensity(funcname, prec, row)
    if not gbps or ai is None:
        return False
    peak = (peaks or {}).get(perf_model.PRECISIONS[prec][3])
    if peak and ai * gbps >= peak:
        return False
    footprint = perf_model.footprint_bytes(funcname, prec, row)
    return footprint is not None and footprint < CACHE_RESIDENT_FACTOR * l2_mb * 1024 * 1024

MARKERS = ('o', 's', '^', 'v', 'D', 'P', 'X', '*', '<', '>')

# Charts are saved as 1920 x 1080 images (Full HD), taller only when the legend needs it, with fonts sized
# to stay readable at that resolution.
FIGURE_DPI = 120
matplotlib.rcParams.update({'font.size': 12, 'axes.titlesize': 15, 'axes.labelsize': 13,
                            'legend.fontsize': 11})


def series_styles(count):
    """Distinct (color, marker) pairs for `count` plotted series."""
    if count <= 10:
        colors = plt.get_cmap('tab10').colors[:count]
    elif count <= 20:
        # tab20 comes in dark/light pairs of one hue; use all dark shades before the light ones
        tab20  = plt.get_cmap('tab20').colors
        colors = (tab20[0::2] + tab20[1::2])[:count]
    else:
        colors = cm.rainbow(np.linspace(0, 1, count))
    return [(colors[i], MARKERS[i % len(MARKERS)]) for i in range(count)]


def chart_size(count):
    """
    Figure size in inches: 16 x 9 (1920 x 1080 at FIGURE_DPI), taller when `count` legend entries
    would not fit.
    """
    return (16, max(9.0, 2.0 + 0.25 * count))


def new_chart(count):
    """
    Figure and axes for a chart with `count` legend entries. Constrained layout shrinks the plot to fit
    the legend inside the figure, so every chart keeps the size chart_size gives it.
    """
    return plt.subplots(figsize=chart_size(count), layout='constrained')


def save_chart(figure, path):
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)


def finish_chart(axes):
    """Major and minor grid on both axes, and the legend to the right of the plot."""
    axes.minorticks_on()
    axes.grid(True, which='major', linewidth=0.7, alpha=0.7)
    axes.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    axes.set_axisbelow(True)
    axes.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)


def ceiling_range_label(ceilings):
    """'23,300' for a theoretical maximum, or '4,402-4,611' when per-result clocks make it vary by over 1%."""
    low, high = min(ceilings), max(ceilings)
    if high <= low * 1.01:
        return f"{high:,.0f}"
    return f"{low:,.0f}-{high:,.0f}"


def size_tick_label(value, _position=None):
    """Problem-size tick: 50M from a million up, 20,480 below, so the axis needs no 1e8 offset."""
    if abs(value) >= 1e6:
        return f"{value / 1e6:g}M"
    return f"{value:,.0f}"


SIZE_DIMENSIONS = ('M', 'N', 'K')


def size_axis_label(rows_dicts, function_names, fallback):
    """
    Problem-size axis label naming the dimensions that equal the plotted size, for example
    'problem size: M = N = K', or 'problem size: N (M = N for gemv, ger)' when functions differ.
    rows_dicts holds one {size: row} dict per plotted series, function_names the series' functions.
    """
    functions_by_dims = {}
    for rows, name in zip(rows_dicts, function_names):
        dims = tuple(d for d in SIZE_DIMENSIONS
                     if rows and all(d in row and int(row[d]) == size for size, row in rows.items()))
        if dims:
            names = functions_by_dims.setdefault(dims, [])
            if name not in names:
                names.append(name)
    if not functions_by_dims:
        return fallback
    # the dimensions most functions share name the axis; the others are listed by function
    ordered = sorted(functions_by_dims.items(), key=lambda item: -len(item[1]))
    label = "problem size: " + " = ".join(ordered[0][0])
    if len(ordered) > 1:
        label += " (" + "; ".join(f"{' = '.join(dims)} for {', '.join(names)}" for dims, names in ordered[1:]) + ")"
    return label


def plot_data(title, gflops_dicts1, gflops_dicts2, const_args_dicts, funcname_list, machine_spec_dict, savedir, theo_max, perf_vs_perf, size_arg = 'N', filename_stem=None, rows_dicts1=None, filename_suffix=''):
    """
    plots gflops data from dictionaries, one plot for each common precision present in all dictionaries.

    Parameters:
        title (string): title for plots
        gflops_dicts1 (list[dict{string: list[(int, float)]}]): data for one machine as given by :func:`get_data_from_directories`.
        gflops_dicts2 (list[dict{string: list[(int, float)]}]): data for another machine as given by :func:`get_data_from_directories`.
        const_args_dicts (list[dict{string: string}]): arguments that remain constant
        funcname_list (list[string]): a list of funcname for each data set to be plotted and used as a savefile name.
        machine_spec_dict (dict{string: string}): specification and peak performance for machine
        savedir (string): directory where resulting plots will be saved.
        theo_max (string): true for plotting performance versus theoretical maximum performance
        perf_vs_perf (string): true for plotting relative performance of one machine versus another machine
        size_arg (string): x axis title on plot when the rows do not say which dimensions the size is.
        rows_dicts1 (list[dict{string: dict{int: dict}}]): full CSV rows per precision and size, as given by
            :func:`get_rows_from_file`; used for per-point theoretical maximum and cache-residency markers.
        filename_suffix (string): added to the file name before the precision, to tell sweeps apart.

    Plain GFLOP/s charts also draw each series' theoretical maximum as a dashed line in the series' color
    when machine_spec_dict has a bandwidth or compute peaks.
    """
    if len(gflops_dicts1) == 0:
        return
    if rows_dicts1 is None:
        rows_dicts1 = [{} for _ in gflops_dicts1]

    # every precision that any of the functions has (Hermitian functions, for example, are complex only)
    for prec in sorted(set().union(*(d.keys() for d in gflops_dicts1))):
        # style only the series this precision actually has
        drawn  = sum(1 for d1, d2 in zip(gflops_dicts1, gflops_dicts2) if prec in d1 and prec in d2)
        styles = iter(series_styles(drawn))
        # one more legend entry for the theoretical maximum key
        figure, axes = new_chart(drawn + 1)

        ratio_max    = 1.0
        any_resident = False
        any_ceiling  = False
        drawn_rows   = []
        drawn_functions = []
        for gflops_dict1, gflops_dict2, funcname, const_args_dict, rows_dict1 in zip(gflops_dicts1, gflops_dicts2, funcname_list, const_args_dicts, rows_dicts1):
            if prec not in gflops_dict1 or prec not in gflops_dict2:
                continue
            cur_color, marker = next(styles)
            gflops1 = gflops_dict1[prec]
            gflops2 = gflops_dict2[prec]
            map1 = {int(t[0]): float(t[1]) for t in gflops1}
            map2 = {int(t[0]): float(t[1]) for t in gflops2}
            common_sizes = sorted(set(map1) & set(map2))
            if not common_sizes:
                print(f"skip {funcname} {prec}: no common problem sizes between datasets")
                continue

            sorted_sizes1 = common_sizes
            sorted_gflops1 = [map1[s] for s in common_sizes]
            sorted_gflops2 = [map2[s] for s in common_sizes]

            if perf_vs_perf != True:
                sorted_sizes1 = [0] + sorted_sizes1
                sorted_gflops1 = [0.0] + sorted_gflops1

            if (perf_vs_perf == True):
                for i in range(len(sorted_gflops1)):
                    if sorted_gflops2[i] != 0:
                        sorted_gflops1[i] /= sorted_gflops2[i]
                    else:
                        sorted_gflops1[i] = float("nan")

            function_label = prec + "_" + funcname
            if(prec == "f32_r"):
                function_label = "s" + funcname
            elif(prec == "f64_r"):
                function_label = "d" + funcname
            elif(prec == "f32_c"):
                function_label = "c" + funcname
            elif(prec == "f64_c"):
                function_label = "z" + funcname

            resident = [False] * len(sorted_sizes1)
            rows = rows_dict1.get(prec, {})
            drawn_rows.append({size: rows[size] for size in common_sizes if size in rows})
            drawn_functions.append(funcname)
            ceiling_note = ''
            if theo_max != True and perf_vs_perf != True:
                sizes = [size for size in sorted_sizes1 if size != 0]
                ceilings = [point_ceiling(funcname, prec, function_label, rows.get(size), machine_spec_dict)
                            for size in sizes]
                if sizes and all(ceilings):
                    any_ceiling = True
                    axes.plot(sizes, ceilings, '--', color=cur_color, linewidth=1.3, alpha=0.8)
                    # functions with the same ceiling share one dashed line, so the legend gives the value
                    ceiling_note = f"  (max {ceiling_range_label(ceilings)})"
            if(theo_max == True):
                ratios = []
                for size, gf in zip(sorted_sizes1, sorted_gflops1):
                    if size == 0:
                        ratios.append(0.0)
                        continue
                    ceiling = point_ceiling(funcname, prec, function_label, rows.get(size), machine_spec_dict)
                    if not ceiling:
                        ratios = None
                        break
                    ratios.append(gf / ceiling)
                if ratios is None:
                    print(f"skip {function_label}: no theoretical maximum (missing --gbps / --peak-gflops, or unmodeled function)")
                    continue
                sorted_gflops1[:] = ratios
                resident = [is_cache_resident(funcname, prec, rows.get(size), machine_spec_dict) for size in sorted_sizes1]
                ratio_max = max([ratio_max] + ratios)

            function_label = function_label + " :  " + const_args_dict[prec] + ceiling_note

            axes.plot(sorted_sizes1, sorted_gflops1, '-', color=cur_color, marker=marker,
                      label=function_label)
            if any(resident):
                any_resident = True
                # draw cache-resident points hollow on top of the line's filled markers
                axes.scatter([x for x, r in zip(sorted_sizes1, resident) if r],
                             [y for y, r in zip(sorted_gflops1, resident) if r],
                             facecolors='white', edgecolors=cur_color, marker=marker, zorder=3)

        if not axes.has_data():
            plt.close(figure)
            continue

        if theo_max == True:
            axes.set_ylim(0, 1.05 * ratio_max)
            axes.axhline(1.0, color='gray', linewidth=1, linestyle='-', zorder=0)
            axes.set_ylabel('gflops / theoretical_maximum_gflops')
            if any_resident:
                axes.annotate(f"hollow: bytes moved < {CACHE_RESIDENT_FACTOR} x L2 (cache-resident)",
                              xy=(0.01, 0.01), xycoords='axes fraction', fontsize=10)
        elif perf_vs_perf == True:
            axes.axhline(1.0, color='gray', linewidth=1, linestyle='-', zorder=0)
            axes.set_ylabel('gflops / gflops')
            ylo, yhi = axes.get_ylim()
            ylo = min(ylo, 1.0)
            yhi = max(yhi, 1.0)
            pad = 0.08 * (yhi - ylo) if yhi > ylo else 0.1
            axes.set_ylim(ylo - pad, yhi + pad)
        else:
            axes.set_ylabel('gflops')
            if any_ceiling:
                # one legend key for every dashed line instead of one entry per function
                axes.plot([], [], '--', color='gray', linewidth=1.3,
                          label='theoretical maximum (dashed, in the color of its function)')

        if isinstance(size_arg, (list, tuple)):
            fallback = " equal ".join(str(s) for s in size_arg)
        else:
            fallback = str(size_arg)
        axes.set_xlabel(size_axis_label(drawn_rows, drawn_functions, fallback))
        axes.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(size_tick_label))

        finish_chart(axes)
        axes.set_title(title)

        filename = ''
        if filename_stem:
            filename = filename_stem
        else:
            for funcname in funcname_list:
                if filename != '':
                    filename += '_'
                filename += funcname
        filename += filename_suffix + '_' + prec
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        out_dir = savedir if os.path.isabs(savedir) else os.path.join(os.getcwd(), savedir)
        save_chart(figure, os.path.join(out_dir, filename))


def plot_roofline(dir1, function_names, machine_spec_dict, savedir, title, achievable_bw_frac=None):
    """
    One log-log roofline per compute_type for the functions in dir1: arithmetic intensity from
    perf_model on x, measured GFLOP/s on y, under the bandwidth and compute ceilings in
    machine_spec_dict. Also writes roofline.csv with every plotted point.
    """
    gbps, peaks = machine_ceilings(machine_spec_dict)
    gbps_published = machine_spec_dict.get('GBps')
    peaks_published = machine_spec_dict.get('peak_gflops') or {}
    if not gbps and not peaks:
        raise SystemExit("plot.py --roofline: machine_spec.yaml has neither GBps nor peak_gflops; "
                         "rerun benchmark.py with --gbps and/or --peak-gflops")

    points = {}
    for function_name in function_names:
        csv_file = os.path.join(dir1, function_name + ".csv")
        if not _csv_has_bench_rows(csv_file):
            print(f"skip {function_name}: no benchmark rows in {csv_file}")
            continue
        funcname = get_function_name(csv_file)
        if not perf_model.has_model(funcname):
            print(f"skip {funcname}: no performance model")
            continue
        for row in get_all_rows(csv_file):
            prec = row['compute_type']
            ai = perf_model.intensity(funcname, prec, row)
            if ai is None:
                continue
            gflops = float(row['hipblas-Gflops'])
            sclk, mclk = perf_model.result_clocks(row)
            points.setdefault(prec, []).append({
                'function': funcname,
                'transA': row.get('transA', ''), 'transB': row.get('transB', ''),
                'compute_type': prec,
                'M': row.get('M', ''), 'N': row.get('N', ''), 'K': row.get('K', ''),
                'ai': ai,
                'gflops': gflops,
                'ceiling': perf_model.ceiling_gflops(funcname, prec, row, *machine_ceilings(machine_spec_dict, row)),
                'footprint_bytes': perf_model.footprint_bytes(funcname, prec, row),
                'cache_resident': is_cache_resident(funcname, prec, row, machine_spec_dict),
                'sclk_mhz': sclk or '', 'mclk_mhz': mclk or '',
            })

    if not points:
        print("plot.py --roofline: no modeled results to plot")
        return

    # One series per function, or per transpose combination when a function's results have several
    # (gemm): the transposes do not change the arithmetic intensity, so they would share a column.
    for pts in points.values():
        combos = collections.defaultdict(set)
        for pt in pts:
            combos[pt['function']].add(pt['transA'] + pt['transB'])
        for pt in pts:
            several = len(combos[pt['function']]) > 1
            pt['series'] = pt['function'] + (' ' + pt['transA'] + pt['transB'] if several else '')

    Path(savedir).mkdir(parents=True, exist_ok=True)
    stem = "_".join(function_names)
    with open(os.path.join(savedir, f"roofline_{stem}.csv"), 'w') as csv_out:
        columns = ['function', 'compute_type', 'transA', 'transB', 'M', 'N', 'K', 'ai', 'gflops', 'ceiling',
                   'pct_of_ceiling', 'footprint_bytes', 'cache_resident', 'sclk_mhz', 'mclk_mhz']
        csv_out.write(",".join(columns) + "\n")
        for prec in sorted(points):
            for pt in points[prec]:
                pct = 100.0 * pt['gflops'] / pt['ceiling'] if pt['ceiling'] else float('nan')
                values = dict(pt, pct_of_ceiling=f"{pct:.2f}")
                csv_out.write(",".join(str(values[c]) for c in columns) + "\n")

    for prec in sorted(points):
        pts = points[prec]
        peak_key = perf_model.PRECISIONS[prec][3]
        peak = (peaks or {}).get(peak_key)
        peak_published = peaks_published.get(peak_key)

        ai_values = [pt['ai'] for pt in pts]
        x_lo, x_hi = min(ai_values) / 4, max(ai_values) * 4
        if gbps and peak:
            ridge = peak / gbps
            x_lo, x_hi = min(x_lo, ridge / 4), max(x_hi, ridge * 4)
        x = np.logspace(np.log10(x_lo), np.log10(x_hi), 200)

        funcs = sorted({pt['series'] for pt in pts})
        figure, axes = new_chart(len(funcs) + 3)

        def roof(bw, pk):
            y = np.full_like(x, np.inf)
            if bw:
                y = np.minimum(y, bw * x)
            if pk:
                y = np.minimum(y, pk)
            return y

        per_result = machine_spec_dict.get('per_result_sclk') or machine_spec_dict.get('per_result_mclk')
        axes.plot(x, roof(gbps, peak), color='black', linewidth=2,
                  label=f"ceiling{' at median clocks' if per_result else ''} ({gbps or '-'} GB/s, {peak or '-'} GFLOP/s)")
        if (gbps_published and gbps_published != gbps) or (peak_published and peak_published != peak):
            axes.plot(x, roof(gbps_published, peak_published), color='gray', linestyle='--', linewidth=1,
                      label="published ceiling (before clock / partition scaling)")
        if achievable_bw_frac and gbps:
            axes.plot(x, roof(gbps * achievable_bw_frac, peak), color='gray', linestyle=':', linewidth=1,
                      label=f"{achievable_bw_frac:.0%} of bandwidth")

        for funcname, (color, marker) in zip(funcs, series_styles(len(funcs))):
            fpts = [pt for pt in pts if pt['series'] == funcname]
            label = perf_model.PRECISIONS[prec][0] + funcname
            solid = [pt for pt in fpts if not pt['cache_resident']]
            hollow = [pt for pt in fpts if pt['cache_resident']]
            if solid:
                axes.scatter([pt['ai'] for pt in solid], [pt['gflops'] for pt in solid], color=color,
                             marker=marker, label=label)
            if hollow:
                axes.scatter([pt['ai'] for pt in hollow], [pt['gflops'] for pt in hollow],
                             facecolors='none', edgecolors=color, marker=marker,
                             label=label if not solid else None)

        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.set_xlabel('arithmetic intensity (FLOP / byte)')
        axes.set_ylabel('GFLOP/s')
        if any(pt['cache_resident'] for pt in pts):
            axes.annotate(f"hollow: bytes moved < {CACHE_RESIDENT_FACTOR} x L2 (cache-resident)",
                          xy=(0.01, 0.01), xycoords='axes fraction', fontsize=10)
        finish_chart(axes)
        axes.set_title(f"{title}  {prec}")
        save_chart(figure, os.path.join(savedir, f"roofline_{stem}_{prec}"))


def get_function_name(filename):
    function_str = "function"
    if os.path.exists(filename):
        lines = open(filename, 'r').readlines()
    else:
        print(filename + " does not exist")
    for i in range(0, len(lines)):
        if(function_str in lines[i]):
            arg_line = lines[i].split(",")
            data_line = re.split(r',\s*(?![^()]*\))', lines[i+1])
            function_idx = arg_line.index(function_str)
            return data_line[function_idx]

def _row_trans_matches(arg_line, data_line, trans_a, trans_b):
    if trans_a is None and trans_b is None:
        return True
    if "transA" not in arg_line or "transB" not in arg_line:
        return False
    ia = arg_line.index("transA")
    ib = arg_line.index("transB")
    return data_line[ia].strip() == trans_a and data_line[ib].strip() == trans_b


def get_data_from_file(
    filename,
    output_param='hipblas-Gflops',
    xaxis_str1='N',
    xaxis_str2='M',
    yaxis_str='hipblas-Gflops',
    *,
    trans_a=None,
    trans_b=None,
):

    precision_str = "compute_type"
    if not os.path.exists(filename):
        return {}
    lines = open(filename, 'r').readlines()

    cur_dict = {}
    for i in range(0, len(lines)):
        if(output_param in lines[i]):
            arg_line = lines[i].split(",")
            data_line = re.split(r',\s*(?![^()]*\))', lines[i+1])
            if not _row_trans_matches(arg_line, data_line, trans_a, trans_b):
                continue
            if xaxis_str1 in arg_line:
                xaxis_idx = arg_line.index(xaxis_str1)
            elif xaxis_str2 in arg_line:
                xaxis_idx = arg_line.index(xaxis_str2)
            else:
                continue
            yaxis_idx = arg_line.index(yaxis_str)
            size_perf_tuple = (int(data_line[xaxis_idx]), float(data_line[yaxis_idx]))

            precision_idx = arg_line.index(precision_str)
            precision = data_line[precision_idx]
            if precision in cur_dict:
                cur_dict[precision].append(size_perf_tuple)
            else:
                cur_dict[precision] = [size_perf_tuple]

    return cur_dict

def get_rows_from_file(filename, xaxis_str1='N', xaxis_str2='M', *, trans_a=None, trans_b=None):
    """Full CSV rows as {compute_type: {size: {column: value}}}, keyed like :func:`get_data_from_file`."""
    rows = {}
    for row in get_all_rows(filename, trans_a=trans_a, trans_b=trans_b):
        if xaxis_str1 in row:
            size = int(row[xaxis_str1])
        elif xaxis_str2 in row:
            size = int(row[xaxis_str2])
        else:
            continue
        rows.setdefault(row['compute_type'], {})[size] = row
    return rows


def get_all_rows(filename, output_param='hipblas-Gflops', *, trans_a=None, trans_b=None):
    """Every benchmark row in a hipblas-bench CSV as a {column: value} dict."""
    if not os.path.exists(filename):
        return []
    lines = open(filename, 'r').readlines()
    rows = []
    for i in range(0, len(lines) - 1):
        if output_param in lines[i]:
            arg_line = lines[i].split(",")
            data_line = re.split(r',\s*(?![^()]*\))', lines[i+1])
            if not _row_trans_matches(arg_line, data_line, trans_a, trans_b):
                continue
            row = {key.strip(): value.strip() for key, value in zip(arg_line, data_line) if key.strip()}
            if 'compute_type' in row and output_param in row:
                rows.append(row)
    return rows


tracked_param_list = [ 'transA', 'transB', 'uplo', 'diag', 'side', 'M', 'N', 'K', 'KL', 'KU', 'alpha', 'alphai', 'beta', 'betai',
                       'incx', 'incy', 'lda', 'ldb', 'ldd', 'stride_x', 'stride_y', 'stride_a', 'stride_b', 'stride_c', 'stride_d',
                       'batch_count']

# return string of arguments that remain constant. For example, transA, transB, alpha, beta, incx may remain
# constant. By contrast, M, N, K, lda, ldb, ldc may change
#def get_const_args_str(filename, output_param='hipblas-Gflops'):
def get_const_args_dict(filename, output_param='hipblas-Gflops', *, trans_a=None, trans_b=None):

    if not os.path.exists(filename):
        return {}
    lines = open(filename, 'r').readlines()

    precision_str = "compute_type"
    precisions = []
    for i in range(0, len(lines)):
        if(output_param in lines[i]):
            arg_line = lines[i].split(",")
            data_line = re.split(r',\s*(?![^()]*\))', lines[i+1])
            if not _row_trans_matches(arg_line, data_line, trans_a, trans_b):
                continue

            precision_idx = arg_line.index(precision_str)
            precision = data_line[precision_idx]
            if precision not in precisions:
                precisions.append(precision)

    const_args_dict = {}

    for precision in precisions:

        function_param_list = tracked_param_list

        arg_line_index_dict = {}
        arg_line_value_dict = {}
        for i in range(0, len(lines)):
            if((output_param in lines[i]) and (precision in lines[i+1])):
                arg_line = lines[i].split(",")
                data_line = re.split(r',\s*(?![^()]*\))', lines[i+1])
                if not _row_trans_matches(arg_line, data_line, trans_a, trans_b):
                    continue

                if not arg_line_index_dict:
                    for arg in arg_line :
                        if(arg in function_param_list):
                            index = arg_line.index(arg)
                            value = data_line[index]
                            arg_line_index_dict[arg]=index
                            arg_line_value_dict[arg]=value
                else:
                    for arg in arg_line :
                        if(arg in function_param_list):
                            index = arg_line.index(arg)
                            value = data_line[index]
                            previous_value = arg_line_value_dict[arg]
                            if(value != previous_value):
                                function_param_list.remove(arg)
                                del arg_line_value_dict[arg]

        const_args_str = _format_const_args_legend(arg_line_value_dict)

        const_args_dict[precision] = const_args_str
    return const_args_dict


def _format_const_args_legend(arg_line_value_dict: dict) -> str:
    """Legend fragment: transA/transB first, then other params in tracked_param_list order."""
    ordered_keys: List[str] = []
    for key in tracked_param_list:
        if key in arg_line_value_dict and key not in ordered_keys:
            ordered_keys.append(key)
    for key in sorted(arg_line_value_dict.keys()):
        if key not in ordered_keys:
            ordered_keys.append(key)
    parts = [f"{key}={arg_line_value_dict[key]}" for key in ordered_keys]
    return ", ".join(parts)


def _csv_has_bench_rows(filename: str, output_param: str = "hipblas-Gflops") -> bool:
    if not os.path.isfile(filename):
        return False
    lines = open(filename, "r", encoding="utf-8", errors="replace").readlines()
    for i in range(len(lines)):
        if output_param in lines[i] and i + 1 < len(lines):
            return True
    return False


GEMM_TRANS_PAIRS = (("N", "N"), ("N", "T"), ("T", "N"), ("T", "T"))


def plot_gemm_by_trans(
    csv_a,
    csv_b,
    *,
    title,
    savedir,
    machine_spec_dict,
    theo_max,
    perf_vs_perf,
    x_axis_label,
    label1,
    label2,
    filename_suffix='',
):
    """One PNG per compute_type with a series per (transA, transB): gemm_f32_r.png, etc."""
    gflops_dicts1, gflops_dicts2, const_args_dicts, rows_dicts1 = [], [], [], []
    for trans_a, trans_b in GEMM_TRANS_PAIRS:
        gflops_dict1 = get_data_from_file(
            csv_a, "hipblas-Gflops", label1, label2, "hipblas-Gflops",
            trans_a=trans_a, trans_b=trans_b,
        )
        gflops_dict2 = get_data_from_file(
            csv_b, "hipblas-Gflops", label1, label2, "hipblas-Gflops",
            trans_a=trans_a, trans_b=trans_b,
        )
        if not gflops_dict1 or not gflops_dict2:
            continue
        const_args_dict = get_const_args_dict(
            csv_a, "hipblas-Gflops", trans_a=trans_a, trans_b=trans_b,
        )
        for prec in list(const_args_dict.keys()):
            trans_prefix = f"transA={trans_a}, transB={trans_b}"
            rest = const_args_dict[prec]
            if rest and "transA=" not in rest:
                const_args_dict[prec] = f"{trans_prefix}, {rest}"
            elif not rest:
                const_args_dict[prec] = trans_prefix
        gflops_dicts1.append(gflops_dict1)
        gflops_dicts2.append(gflops_dict2)
        const_args_dicts.append(const_args_dict)
        rows_dicts1.append(get_rows_from_file(csv_a, label1, label2, trans_a=trans_a, trans_b=trans_b))
    if not gflops_dicts1:
        print("skip gemm: no transA/transB combination with data on both sides")
        return
    plot_data(
        title,
        gflops_dicts1,
        gflops_dicts2,
        const_args_dicts,
        ["gemm"] * len(gflops_dicts1),
        machine_spec_dict,
        savedir,
        theo_max,
        perf_vs_perf,
        x_axis_label,
        filename_stem="gemm",
        rows_dicts1=rows_dicts1,
        filename_suffix=filename_suffix,
    )


SWEEPS_FILE = "sweeps.yaml"


def sweep_groups(data_dir, function_names):
    """
    [(filename suffix, title description, function names)]: the functions grouped by the sweep that
    produced their results, as benchmark.py records it in sweeps.yaml. A single group with no suffix when
    every function comes from one sweep, or when the results predate sweeps.yaml.
    """
    path = os.path.join(data_dir, SWEEPS_FILE)
    sweeps = yaml.safe_load(Path(path).read_text()) if os.path.isfile(path) else None
    groups = {}
    for name in function_names:
        sweep = (sweeps or {}).get(name)
        key = (sweep.get('sizes', 'default'), sweep.get('flush_mb') or 0) if sweep else None
        groups.setdefault(key, []).append(name)
    if len(groups) <= 1:
        return [("", "", list(function_names))]
    result = []
    for key, names in groups.items():
        if key is None:
            result.append(("_unrecorded", "sweep not recorded", names))
            continue
        sizes, flush_mb = key
        suffix = f"_{sizes}" + (f"_flush{flush_mb:g}" if flush_mb else "")
        description = f"{sizes} sizes" + (f", --flush-mb {flush_mb:g}" if flush_mb else "")
        result.append((suffix, description, names))
    return result


def _data_dir(level, tag, *, tag_parent=False, blas_subdir=None):
    """CSV directory for one column (matches benchmark_report.py layout options)."""
    if tag_parent or blas_subdir:
        path = os.path.join(tag, level)
        if blas_subdir:
            path = os.path.join(path, blas_subdir)
        return path
    return os.path.join(level, tag)


def _normalize_blas_subdir(base_dir, level, root, tag, blas_subdir):
    if not blas_subdir:
        return None
    level_dir = os.path.join(base_dir, root, level)
    if os.path.isdir(os.path.join(level_dir, blas_subdir)):
        return blas_subdir
    if blas_subdir in (root, tag) and os.path.isdir(level_dir):
        if any(f.endswith(".csv") for f in os.listdir(level_dir)):
            return None
    if blas_subdir == tag:
        for alt in ("gfx950", "gfx90a"):
            if os.path.isdir(os.path.join(level_dir, alt)):
                return alt
    return blas_subdir


def _dir_for_explicit_tree(base_dir, level, tree_tag):
    """Join base, tree, and level only (tree holds blas1, blas2, blas3)."""
    candidates = [tree_tag]
    if len(tree_tag) > 1 and tree_tag.endswith("X"):
        candidates.append(tree_tag[:-1])
    for root in candidates:
        path = os.path.join(base_dir, root, level)
        if os.path.isdir(path):
            return path
    return os.path.join(base_dir, tree_tag, level)


def _resolve_filesystem_tree_tag(
    base_dir,
    level,
    tag,
    *,
    tree_tag,
    tag_parent,
    blas_subdir,
):
    if not (tag_parent or blas_subdir):
        return tree_tag or tag

    def _level_ok(root):
        norm = _normalize_blas_subdir(base_dir, level, root, tag, blas_subdir)
        path = os.path.join(
            base_dir,
            _data_dir(level, root, tag_parent=tag_parent or bool(norm), blas_subdir=norm),
        )
        return os.path.isdir(path)

    if _level_ok(tag):
        return tag
    if len(tag) > 1 and tag.endswith("X"):
        alt = tag[:-1]
        if _level_ok(alt):
            return alt
    return tag


def _plot_pair_folder(tag1, tag2, *, plot_tag2=None, b_blas_subdir=None, plot_tag1=None, a_blas_subdir=None, label_a=None, label_b=None):
    """Subfolder under blasL/ for perf-vs-perf plots (tag1_tag2)."""
    first = plot_tag1 or label_a or a_blas_subdir or tag1
    second = plot_tag2 or label_b or b_blas_subdir or tag2
    return f"{first}_{second}"


if __name__ =='__main__':

    parser = argparse.ArgumentParser(
            description='plot hipblas-bench results for multiple csv files',
            epilog='Example usage: python3 plot_benchmarks.py ' +
                    '-l blas1 -t gfx906  -f scal -f axpy  --label1 "N" --label2 "M"')

    parser.add_argument('-l', '--level',          help='BLAS level',          dest='level',          default='blas1')
    parser.add_argument('-t',  '--tag1',          help='tag1',                dest='tag1',           default='gfx906')
    parser.add_argument(       '--tag2',          help='tag2',                dest='tag2',           default='ref')
    parser.add_argument(     '--label1',          help='label1',              dest='label1',         default='N')
    parser.add_argument(     '--label2',          help='label2',              dest='label2',         default='M')
    parser.add_argument('-f'           ,          help='function name',       dest='function_names', required=True, action='append')
    parser.add_argument(     '--theo_max',        help="perf vs theo_max",    dest='theo_max', default="false", action='store_true')
    parser.add_argument(     '--no_theo_max',     help="no perf vs theo_max", dest='theo_max', action='store_false')
    parser.add_argument(     '--perf_vs_perf',    help="perf vs perf",        dest='perf_vs_perf', default="false", action='store_true')
    parser.add_argument(     '--no_perf_vs_perf', help="no perf vs perf",     dest='perf_vs_perf', action='store_false')
    parser.add_argument(     '--roofline',        help="roofline: GFLOP/s vs arithmetic intensity under the machine_spec.yaml ceilings",
                                                                              dest='roofline', action='store_true')
    parser.add_argument(     '--achievable-bw-frac', help="roofline: also draw a bandwidth line at this fraction of the ceiling (e.g. a measured achievable fraction)",
                                                                              dest='achievable_bw_frac', type=float, default=None)
    parser.add_argument(
        '--tag-parent',
        action='store_true',
        help='Column A (--tag1) CSVs under {tag1}/{level}/ instead of {level}/{tag1}/.',
    )
    parser.add_argument(
        '--a-blas-subdir',
        default=None,
        metavar='NAME',
        help='Column A (--tag1) CSVs under {tag1}/{level}/NAME/ (tag-first layout).',
    )
    parser.add_argument(
        '--b-blas-subdir',
        default=None,
        metavar='NAME',
        help='Column B (--tag2) CSVs under {tag2}/{level}/NAME/ (e.g. gfx950 under MI355X).',
    )
    parser.add_argument(
        '--a-tree',
        default=None,
        metavar='DIR',
        help='Column A CSVs under {tree}/{level}/ only (no arch subfolder under level).',
    )
    parser.add_argument(
        '--b-tree',
        default=None,
        metavar='DIR',
        help='Column B CSVs under {tree}/{level}/ only (no arch subfolder under level).',
    )
    parser.add_argument(
        '--plot-tag2',
        default=None,
        metavar='NAME',
        help='Name in output folder {tag1}_NAME (default: --b-blas-subdir or --tag2).',
    )
    parser.add_argument(
        '--plot-tag1',
        default=None,
        metavar='NAME',
        help='First part of output folder NAME_tag2 (default: --label-a or --tag1).',
    )
    parser.add_argument(
        '--label-a',
        default=None,
        metavar='NAME',
        help='Display/arch label for column A (plot output folder when set).',
    )
    parser.add_argument(
        '--label-b',
        default=None,
        metavar='NAME',
        help='Display/arch label for column B (plot output folder when set).',
    )
    parser.add_argument(
        '--base',
        default='.',
        metavar='DIR',
        help='Root for CSV trees and perf-vs-perf plot output (default: cwd).',
    )
    parser.add_argument(
        '--combine-gemm-trans',
        action='store_true',
        help='Plot gemm as a single series mixing every transA/transB pair (legacy); the default draws one series per pair.',
    )

    args = parser.parse_args()

    if sum([args.theo_max == True, args.perf_vs_perf == True, args.roofline]) > 1:
        parser.error("select only one of --theo_max, --perf_vs_perf, --roofline")

    funcname_list = []

    gflops_dicts1 = []
    gflops_dicts2 = []
    const_args_dicts = []

    const_args_list = []

    data_root = os.path.abspath(args.base)
    if args.a_tree:
        dir1 = _dir_for_explicit_tree(data_root, args.level, args.a_tree)
    else:
        root1 = _resolve_filesystem_tree_tag(
            data_root,
            args.level,
            args.tag1,
            tree_tag=None,
            tag_parent=args.tag_parent or bool(args.a_blas_subdir),
            blas_subdir=args.a_blas_subdir,
        )
        dir1 = os.path.join(
            data_root,
            _data_dir(
                args.level,
                root1,
                tag_parent=args.tag_parent or bool(_normalize_blas_subdir(data_root, args.level, root1, args.tag1, args.a_blas_subdir)),
                blas_subdir=_normalize_blas_subdir(data_root, args.level, root1, args.tag1, args.a_blas_subdir),
            ),
        )
    if args.b_tree:
        dir2 = _dir_for_explicit_tree(data_root, args.level, args.b_tree)
    else:
        root2 = _resolve_filesystem_tree_tag(
            data_root,
            args.level,
            args.tag2,
            tree_tag=None,
            tag_parent=False,
            blas_subdir=args.b_blas_subdir,
        )
        dir2 = os.path.join(
            data_root,
            _data_dir(
                args.level,
                root2,
                tag_parent=bool(_normalize_blas_subdir(data_root, args.level, root2, args.tag2, args.b_blas_subdir)),
                blas_subdir=_normalize_blas_subdir(data_root, args.level, root2, args.tag2, args.b_blas_subdir),
            ),
        )
    plot_tag2 = args.plot_tag2 or args.b_blas_subdir or args.tag2
    plot_pair = _plot_pair_folder(
        args.tag1,
        args.tag2,
        plot_tag2=args.plot_tag2,
        b_blas_subdir=args.b_blas_subdir,
        a_blas_subdir=args.a_blas_subdir,
        plot_tag1=args.plot_tag1,
        label_a=args.label_a,
        label_b=args.label_b,
    )

    if args.roofline:
        savedir = os.path.join(dir1, "plots_roofline")
        title = "Roofline " + args.tag1
    elif (args.theo_max == True):
        savedir = os.path.join(dir1, "plots_vs_theo_max")
        title = args.tag1 +  "(  performance / theoretical_maximum_performance )"
    elif (args.perf_vs_perf == True):
        savedir = os.path.join(data_root, plot_pair, args.level, "plots_perf_vs_perf")
        title = "Relative Performance ( " + args.tag1 + " /  " + plot_tag2 + " )"
    else:
        savedir = os.path.join(dir1, "plots_gflops")
        title = "Performance " + args.tag1

    machine_spec_yaml_file = os.path.join(dir1, "machine_spec.yaml")

    if not os.path.isfile(machine_spec_yaml_file):
        raise SystemExit(f"plot.py: missing {machine_spec_yaml_file}; run benchmark.py first")
    machine_spec_dict = yaml.safe_load(Path(machine_spec_yaml_file).read_text())

    if args.theo_max == True and not (machine_spec_dict.get('GBps') or machine_spec_dict.get('peak_gflops')):
        raise SystemExit("plot.py --theo_max: machine_spec.yaml has no theoretical maximum; "
                         "rerun benchmark.py with --gbps and/or --peak-gflops")

    if args.roofline:
        plot_roofline(dir1, list(args.function_names), machine_spec_dict, savedir, title,
                      achievable_bw_frac=args.achievable_bw_frac)
        raise SystemExit(0)

    if args.level == "blas1":
        x_axis_label = args.label1
    else:
        x_axis_label = f"{args.label1} equal {args.label2}"

    # Separate benchmark.py calls into the same folder (for example the large dense sweeps and the banded
    # sweeps with --flush-mb) get separate charts.
    plotted_any = False
    for filename_suffix, sweep_description, sweep_function_names in sweep_groups(dir1, args.function_names):
        sweep_title = title + (f"  ({sweep_description})" if sweep_description else "")
        function_names = list(sweep_function_names)
        if "gemm" in function_names and not args.combine_gemm_trans:
            csv_a = os.path.join(dir1, "gemm.csv")
            csv_b = os.path.join(dir2, "gemm.csv")
            if os.path.isfile(csv_a) and os.path.isfile(csv_b):
                plot_gemm_by_trans(
                    csv_a,
                    csv_b,
                    title=sweep_title,
                    savedir=savedir,
                    machine_spec_dict=machine_spec_dict,
                    theo_max=args.theo_max,
                    perf_vs_perf=args.perf_vs_perf,
                    x_axis_label=x_axis_label,
                    label1=args.label1,
                    label2=args.label2,
                    filename_suffix=filename_suffix,
                )
            else:
                print("skip gemm by trans: missing gemm.csv on A or B side")
            function_names = [f for f in function_names if f != "gemm"]

        gflops_dicts1 = []
        gflops_dicts2 = []
        const_args_dicts = []
        funcname_list = []
        rows_dicts1 = []

        for function_name in function_names:

            output_filename1 = os.path.join(dir1, function_name+".csv")
            output_filename2   = os.path.join(dir2, function_name+".csv")

            if not os.path.isfile(output_filename1) or not os.path.isfile(output_filename2):
                print(f"skip {function_name}: missing csv on A or B side")
                continue
            if not _csv_has_bench_rows(output_filename1) or not _csv_has_bench_rows(output_filename2):
                print(f"skip {function_name}: no benchmark rows on one or both sides")
                continue

            gflops_dict1 = get_data_from_file(output_filename1, "hipblas-Gflops", args.label1, args.label2, "hipblas-Gflops")
            gflops_dict2 = get_data_from_file(output_filename2, "hipblas-Gflops", args.label1, args.label2, "hipblas-Gflops")

            if not gflops_dict1 or not gflops_dict2:
                print(f"skip {function_name}: empty gflops data")
                continue

            gflops_dicts1.append(gflops_dict1)
            gflops_dicts2.append(gflops_dict2)
            rows_dicts1.append(get_rows_from_file(output_filename1, args.label1, args.label2))

            const_args_dict = get_const_args_dict(output_filename1, "hipblas-Gflops")

            const_args_dicts.append(const_args_dict)

            fn = get_function_name(output_filename1)
            if not fn:
                print(f"skip {function_name}: could not parse function name")
                gflops_dicts1.pop()
                gflops_dicts2.pop()
                const_args_dicts.pop()
                rows_dicts1.pop()
                continue
            funcname_list.append(fn)

        if not funcname_list:
            continue
        plotted_any = True

        print("plotting for: ", funcname_list)
        plot_data(sweep_title, gflops_dicts1, gflops_dicts2, const_args_dicts, funcname_list, machine_spec_dict, savedir, args.theo_max, args.perf_vs_perf, x_axis_label, rows_dicts1=rows_dicts1, filename_suffix=filename_suffix)

    if not plotted_any:
        if "gemm" in args.function_names and not args.combine_gemm_trans:
            raise SystemExit(0)
        print("plot.py: no functions to plot after filtering csvs")
        raise SystemExit(1)

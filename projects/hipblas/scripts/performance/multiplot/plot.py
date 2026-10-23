#!/usr/bin/env python3

import argparse
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import yaml
from pathlib import Path
from typing import List

def plot_data(title, gflops_dicts1, gflops_dicts2, const_args_dicts, funcname_list, machine_spec_dict, savedir, theo_max, perf_vs_perf, size_arg = 'N', filename_stem=None):
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
        size_arg (string): x axis title on plot.
    """
    if len(gflops_dicts1) == 0:
        return

    gflops_dict0 = gflops_dicts1[0]
    for prec in sorted(gflops_dict0.keys()):
        colors=iter(cm.rainbow(np.linspace(0,1,len(gflops_dicts1))))
        figure, axes = plt.subplots(figsize=(7,7))

        for gflops_dict1, gflops_dict2, funcname, const_args_dict in zip(gflops_dicts1, gflops_dicts2, funcname_list, const_args_dicts):
            cur_color = next(colors)
            if prec not in gflops_dict1 or prec not in gflops_dict2:
                continue
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

            if(prec == "f32_r"):
                function_label = "s" + funcname
            elif(prec == "f64_r"):
                function_label = "d" + funcname
            elif(prec == "f32_c"):
                function_label = "c" + funcname
            elif(prec == "f64_c"):
                function_label = "z" + funcname

            if(theo_max == True):
                theo_max_value = machine_spec_dict[function_label]
                sorted_gflops1[:] = [gf / theo_max_value for gf in sorted_gflops1]

            function_label = function_label + " :  " + const_args_dict[prec]

            axes.scatter(sorted_sizes1, sorted_gflops1, color=cur_color, label=function_label)
            axes.plot(sorted_sizes1, sorted_gflops1, '-o', color=cur_color)

        if not axes.has_data():
            plt.close(figure)
            continue

        if theo_max == True:
            axes.set_ylim(0, 1)
            axes.axhline(1.0, color='gray', linewidth=1, linestyle='-', zorder=0)
            axes.set_ylabel('gflops / theoretical_maximum_gflops')
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

        if isinstance(size_arg, (list, tuple)):
            axes.set_xlabel(" equal ".join(str(s) for s in size_arg))
        else:
            axes.set_xlabel(str(size_arg))

        # magic numbers from performancereport.py to make plots look nice
        axes.legend(fontsize=10, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                    mode='expand', borderaxespad=0.)
        figure.tight_layout(rect=(0,0.05,1.0,0.94))

        figure.suptitle(title, y=0.96)

        filename = ''
        if filename_stem:
            filename = filename_stem
        else:
            for funcname in funcname_list:
                if filename != '':
                    filename += '_'
                filename += funcname
        filename += '_' + prec
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        out_dir = savedir if os.path.isabs(savedir) else os.path.join(os.getcwd(), savedir)
        figure.savefig(os.path.join(out_dir, filename), dpi=100)
        plt.close(figure)

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
):
    """One PNG per (transA, transB) and compute_type: gemm_NN_f32_r.png, etc."""
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
            print(f"skip gemm {trans_a}{trans_b}: no data on one or both sides")
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
        stem = f"gemm_{trans_a}{trans_b}"
        plot_data(
            title + f"  transA={trans_a} transB={trans_b}",
            [gflops_dict1],
            [gflops_dict2],
            [const_args_dict],
            ["gemm"],
            machine_spec_dict,
            savedir,
            theo_max,
            perf_vs_perf,
            x_axis_label,
            filename_stem=stem,
        )


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
        help='Plot all GEMM transA/transB pairs on one chart per compute_type (legacy).',
    )

    args = parser.parse_args()

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

    if (args.theo_max == True):
        savedir = os.path.join(dir1, "plots_vs_theo_max")
        title = args.tag1 +  "(  performance / theoretical_maximum_performance )"
    elif (args.perf_vs_perf == True):
        savedir = os.path.join(data_root, plot_pair, args.level, "plots_perf_vs_perf")
        title = "Relative Performance ( " + args.tag1 + " /  " + plot_tag2 + " )"
    else:
        savedir = os.path.join(dir1, "plots_gflops")
        title = "Performance " + args.tag1

    machine_spec_yaml_file = os.path.join(dir1, "machine_spec.yaml")

    machine_spec_dict = yaml.safe_load(Path(machine_spec_yaml_file).read_text())

    if args.level == "blas1":
        x_axis_label = args.label1
    else:
        x_axis_label = f"{args.label1} equal {args.label2}"

    function_names = list(args.function_names)
    if "gemm" in function_names and not args.combine_gemm_trans:
        csv_a = os.path.join(dir1, "gemm.csv")
        csv_b = os.path.join(dir2, "gemm.csv")
        if os.path.isfile(csv_a) and os.path.isfile(csv_b):
            plot_gemm_by_trans(
                csv_a,
                csv_b,
                title=title,
                savedir=savedir,
                machine_spec_dict=machine_spec_dict,
                theo_max=args.theo_max,
                perf_vs_perf=args.perf_vs_perf,
                x_axis_label=x_axis_label,
                label1=args.label1,
                label2=args.label2,
            )
        else:
            print("skip gemm by trans: missing gemm.csv on A or B side")
        function_names = [f for f in function_names if f != "gemm"]

    gflops_dicts1 = []
    gflops_dicts2 = []
    const_args_dicts = []
    funcname_list = []

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

        const_args_dict = get_const_args_dict(output_filename1, "hipblas-Gflops")

        const_args_dicts.append(const_args_dict)

        fn = get_function_name(output_filename1)
        if not fn:
            print(f"skip {function_name}: could not parse function name")
            gflops_dicts1.pop()
            gflops_dicts2.pop()
            const_args_dicts.pop()
            continue
        funcname_list.append(fn)

    if not funcname_list:
        if "gemm" in args.function_names and not args.combine_gemm_trans:
            raise SystemExit(0)
        print("plot.py: no functions to plot after filtering csvs")
        raise SystemExit(1)

    print("plotting for: ", funcname_list)
    plot_data(title, gflops_dicts1, gflops_dicts2, const_args_dicts, funcname_list, machine_spec_dict, savedir, args.theo_max, args.perf_vs_perf, x_axis_label)

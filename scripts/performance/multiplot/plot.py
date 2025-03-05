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
import sys
from pathlib import Path

script_dir = os.path.dirname( __file__ )
getspec_module_dir = os.path.join(script_dir, '..', 'blas')
sys.path.append(getspec_module_dir)
import getspecs

theo_max_dict = {}

#   L1 functions
theo_max_dict['saxpy'] = ( 1 /  6 )
theo_max_dict['daxpy'] = ( 1 / 12 )
theo_max_dict['caxpy'] = ( 1 /  3 )
theo_max_dict['zaxpy'] = ( 1 /  6 )

theo_max_dict['sdot'] = ( 1 / 4 )
theo_max_dict['ddot'] = ( 1 / 8 )
theo_max_dict['cdot'] = ( 1 / 2 )
theo_max_dict['zdot'] = ( 1 / 4 )

theo_max_dict['sscal'] = ( 1 /  8 )
theo_max_dict['dscal'] = ( 1 / 16 )
theo_max_dict['cscal'] = ( 3 /  8 )
theo_max_dict['zscal'] = ( 3 / 16 )

theo_max_dict['scopy'] = 1
theo_max_dict['dcopy'] = 1
theo_max_dict['ccopy'] = 1
theo_max_dict['zcopy'] = 1

theo_max_dict['sswap'] = 1
theo_max_dict['dswap'] = 1
theo_max_dict['cswap'] = 1
theo_max_dict['zswap'] = 1

#   L2 functions
# matrix vector multiplication
theo_max_dict['sgemv'] = ( 1 / 2 )
theo_max_dict['dgemv'] = ( 1 / 4 )
theo_max_dict['cgemv'] = ( 1     )
theo_max_dict['zgemv'] = ( 1 / 2 )

#   theo_max_dict['sgbmv'] = ( 1 / 2 )
#   theo_max_dict['dgbmv'] = ( 1 / 4 )
#   theo_max_dict['cgbmv'] = ( 1     )
#   theo_max_dict['zgbmv'] = ( 1 / 2 )

theo_max_dict['strmv'] = ( 1 / 2 )
theo_max_dict['dtrmv'] = ( 1 / 4 )
theo_max_dict['ctrmv'] = ( 1     )
theo_max_dict['ztrmv'] = ( 1 / 2 )

#   theo_max_dict['stbmv'] = ( 1 / 2 )
#   theo_max_dict['dtbmv'] = ( 1 / 4 )
#   theo_max_dict['ctbmv'] = ( 1     )
#   theo_max_dict['ztbmv'] = ( 1 / 2 )

theo_max_dict['stpmv'] = ( 1 / 2 )
theo_max_dict['dtpmv'] = ( 1 / 4 )
theo_max_dict['ctpmv'] = ( 1     )
theo_max_dict['ztpmv'] = ( 1 / 2 )

# Symmetric matrix vector multiplication
theo_max_dict['ssymv'] = ( 1     )
theo_max_dict['dsymv'] = ( 1 / 2 )

#   theo_max_dict['ssbmv'] = ( 1     )
#   theo_max_dict['dsbmv'] = ( 1 / 2 )

theo_max_dict['sspmv'] = ( 1     )
theo_max_dict['dspmv'] = ( 1 / 2 )

# Hermitian matrix vector multiplication
theo_max_dict['chemv'] = ( 2     )
theo_max_dict['zhemv'] = ( 1     )

theo_max_dict['chpmv'] = ( 2     )
theo_max_dict['zhpmv'] = ( 1     )

#   theo_max_dict['chbmv'] = ( 2     )
#   theo_max_dict['zhbmv'] = ( 1     )

# rank 1 updates
theo_max_dict['sger'] = ( 1 / 4 )
theo_max_dict['dger'] = ( 1 / 8 )

theo_max_dict['ssyr'] = ( 1 / 4 )
theo_max_dict['dsyr'] = ( 1 / 8 )

theo_max_dict['sspr'] = ( 1 / 4 )
theo_max_dict['dspr'] = ( 1 / 8 )

theo_max_dict['ssyr2'] = ( 1 / 2 )
theo_max_dict['dsyr2'] = ( 1 / 4 )

theo_max_dict['sspr2'] = ( 1 / 2 )
theo_max_dict['dspr2'] = ( 1 / 4 )

# Hermitian rank 1 updates
theo_max_dict['cher'] = ( 1 / 2 )
theo_max_dict['zher'] = ( 1 / 4 )

theo_max_dict['chpr'] = ( 1 / 2 )
theo_max_dict['zhpr'] = ( 1 / 4 )

theo_max_dict['cher2'] = ( 1     )
theo_max_dict['zher2'] = ( 1 / 2 )

theo_max_dict['chpr2'] = ( 1     )
theo_max_dict['zhpr2'] = ( 1 / 2 )

# triangle solves
theo_max_dict['strsv'] = ( 1 / 2 )
theo_max_dict['dtrsv'] = ( 1 / 4 )
theo_max_dict['ctrsv'] = ( 1     )
theo_max_dict['ztrsv'] = ( 1 / 2 )

#   theo_max_dict['stbsv'] = ( 1 / 2 )
#   theo_max_dict['dtbsv'] = ( 1 / 4 )
#   theo_max_dict['ctbsv'] = ( 1     )
#   theo_max_dict['ztbsv'] = ( 1 / 2 )

theo_max_dict['stpsv'] = ( 1 / 2 )
theo_max_dict['dtpsv'] = ( 1 / 4 )
theo_max_dict['ctpsv'] = ( 1     )
theo_max_dict['ztpsv'] = ( 1 / 2 )


def plot_data(gflops_dicts, const_args_dicts, funcname_list, savedir, arch, theo_max, memory_bandwidth, size_arg = 'N'):
    """
    plots gflops data from dictionaries, one plot for each common precision present in all dictionaries.

    Parameters:
        gflops_dicts (list[dict{string: list[(int, float)]}]): data as given by :func:`get_data_from_directories`.
        funcname_list (list[string]): a list of funcname for each data set to be plotted and used as a savefile name.
        savedir (string): directory where resulting plots will be saved.
        size_arg (string): x axis title on plot.
    """
    if len(gflops_dicts) == 0:
        return

    gflops_dict0 = gflops_dicts[0]
    for prec, _ in gflops_dict0.items():
        colors=iter(cm.rainbow(np.linspace(0,1,len(gflops_dicts))))
        figure, axes = plt.subplots(figsize=(7,7))

        for gflops_dict, funcname, const_args_dict in zip(gflops_dicts, funcname_list, const_args_dicts):
            cur_color = next(colors)
            if prec not in gflops_dict:
                continue
            gflops = gflops_dict[prec]
            gflops.append((0, 0)) # I prefer having a 0 at the bottom so the performance looks more accurate
            sorted_tuples = sorted(gflops)
            sorted_sizes = [x[0] for x in sorted_tuples]
            sorted_gflops = [x[1] for x in sorted_tuples]

            if(prec == "f32_r"):
                function_label = "s" + funcname
            elif(prec == "f64_r"):
                function_label = "d" + funcname
            elif(prec == "f32_c"):
                function_label = "c" + funcname
            elif(prec == "f64_c"):
                function_label = "z" + funcname

            if(theo_max == True):
                theo_max_value = theo_max_dict[function_label] * memory_bandwidth

                sorted_gflops[:] = [gf / theo_max_value for gf in sorted_gflops]

            function_label = function_label + " :  " + const_args_dict[prec]

            axes.scatter(sorted_sizes, sorted_gflops, color=cur_color, label=function_label)
            axes.plot(sorted_sizes, sorted_gflops, '-o', color=cur_color)

        if(theo_max == True):
            axes.set_ylim(0, 1)
            if ("copy" in funcname_list) or ("swap" in funcname_list):
                axes.set_ylabel('GB/s / theoretical_maximum_GB/s')
            else:
                axes.set_ylabel('gflops / theoretical_maximum_gflops')
        else:
            if ("copy" in funcname_list) or ("swap" in funcname_list):
                axes.set_ylabel('GB/sec')
            else:
                axes.set_ylabel('gflops')

        axes.set_xlabel('='.join(size_arg)) # in case we add multiple params

        # magic numbers from performancereport.py to make plots look nice
        axes.legend(fontsize=10, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                    mode='expand', borderaxespad=0.)
        figure.tight_layout(rect=(0,0.05,1.0,0.94))

        figure.suptitle(arch, y=0.96)

        filename = ''
        for funcname in funcname_list:
            if filename != '':
                filename += '_'
            filename += funcname
        filename += '_' + prec
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        figure.savefig(os.path.join(os.getcwd(), savedir, filename))


def get_function_name(filename):
    function_str = "function"
    if os.path.exists(filename):
        lines = open(filename, 'r').readlines()
    else:
        print(filename + "does not exist")
    for i in range(0, len(lines)):
        if(function_str in lines[i]):
            arg_line = lines[i].split(",")
            data_line = re.split(r',\s*(?![^()]*\))', lines[i+1])
            function_idx = arg_line.index(function_str)
            return data_line[function_idx]

def get_mem_bandwidth_from_file(filename, arch):
    if os.path.exists(filename):
        file = open(filename, 'r')
        file.seek(0)
        for line in file:
            match = re.search('MCLK (\d+)', line)
            if match:
                MCLK = match.group(1)
        file.seek(0)
        for line in file:
            match = re.search('memoryBusWidth (\d+)', line)
            if match:
                memoryBusWidth = match.group(1)

        print("MCLK, memoryBusWidth = ", MCLK, ", ", memoryBusWidth)
        if(arch == 'gfx906'):
            return int(MCLK) / 1000 * int(memoryBusWidth) * 2
        if(arch == 'gfx90a'):
            return int(MCLK) / 1000 * int(memoryBusWidth) * 2
        if(arch == 'gfx942'):
            return int(MCLK) / 1000 * int(memoryBusWidth) * 4
        else:
            print("using memory bandwidth = memoryBusWidth * memoryClockRate * 2")
            return int(MCLK) / 1000 * int(memoryBusWidth) * 2


def get_data_from_file(filename, output_param='rocblas-Gflops', xaxis_str1='N', xaxis_str2='M', yaxis_str='rocblas-Gflops'):

    precision_str = "compute_type"
    if os.path.exists(filename):
        lines = open(filename, 'r').readlines()

    function_name = get_function_name(filename)

    cur_dict = {}
    for i in range(0, len(lines)):
        if(output_param in lines[i]):
            arg_line = lines[i].split(",")
            data_line = re.split(r',\s*(?![^()]*\))', lines[i+1])
            if(xaxis_str1 in arg_line):
                xaxis_idx = arg_line.index(xaxis_str1)
            if(xaxis_str2 in arg_line):
                xaxis_idx = arg_line.index(xaxis_str2)
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
#def get_const_args_str(filename, output_param='rocblas-Gflops'):
def get_const_args_dict(filename, output_param='rocblas-Gflops'):

    if os.path.exists(filename):
        lines = open(filename, 'r').readlines()


    precision_str = "compute_type"
    precisions = []
    for i in range(0, len(lines)):
        if(output_param in lines[i]):
            arg_line = lines[i].split(",")
            data_line = re.split(r',\s*(?![^()]*\))', lines[i+1])

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

        const_args_str = ""
        for key, value in arg_line_value_dict.items():
            if(const_args_str == ""):
                const_args_str += key + "=" + value
            else:
                const_args_str += ", " + key + "=" + value

        const_args_dict[precision] = const_args_str
    return const_args_dict

if __name__ =='__main__':

    parser = argparse.ArgumentParser(
            description='plot rocblas-bench results for multiple csv files',
            epilog='Example usage: python3 plot_benchmarks.py ' +
                    '-l blas1 -t gfx906  -f scal -f axpy  --label1 "N" --label2 "M"')

    parser.add_argument('-l', '--level',      help='BLAS level',          dest='level',          default='blas1')
    parser.add_argument('-t',   '--tag',      help='tag',                 dest='tag',            default='gfx906')
    parser.add_argument(     '--label1',      help='label1',              dest='label1',         default='N')
    parser.add_argument(     '--label2',      help='label2',              dest='label2',         default='M')
    parser.add_argument('-f'           ,      help='function name',       dest='function_names', required=True, action='append')
    parser.add_argument(     '--theo_max',    help="perf vs theo_max",    dest='theo_max', default="false", action='store_true')
    parser.add_argument(     '--no_theo_max', help="no perf vs theo_max", dest='theo_max', action='store_false')
    parser.set_defaults(theo_max=False)

    args = parser.parse_args()

    funcname_list = []

    res_dicts = []
    const_args_dicts = []

    const_args_list = []

    output_dir = os.path.join(args.level,args.tag)

    if (args.theo_max == True):
        output_dir = os.path.join(output_dir, "plots_vs_theo_max")
    else:
        output_dir = os.path.join(output_dir, "plots_gflops")

    device_number = 1
    cuda = False
    arch = getspecs.getgfx(device_number, cuda)

    for function_name in args.function_names:

        output_filename = os.path.join(args.level, args.tag, function_name+".csv")
        memory_bandwidth = get_mem_bandwidth_from_file(output_filename, arch)

        if (function_name == 'copy') or (function_name == 'swap'):
            cur_dict = get_data_from_file(output_filename, "rocblas-GB/s", args.label1, args.label2, "rocblas-GB/s")
        else:
            cur_dict = get_data_from_file(output_filename, "rocblas-Gflops", args.label1, args.label2, "rocblas-Gflops")

        res_dicts.append(cur_dict)

        if (function_name == 'copy') or (function_name == 'swap'):
            const_args_dict = get_const_args_dict(output_filename, "rocblas-GB/s")
        else:
            const_args_dict = get_const_args_dict(output_filename, "rocblas-Gflops")

        const_args_dicts.append(const_args_dict)

        function_name = get_function_name(output_filename)
        funcname_list.append(function_name)

    print("plotting for: ", funcname_list)
    plot_data(res_dicts, const_args_dicts, funcname_list, output_dir, arch, args.theo_max, memory_bandwidth, args.label1)

"""Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
   ies of the Software, and to permit persons to whom the Software is furnished
   to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
   PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
   FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
   IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
   CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import sys
from os import path
sys.path.append(path.join(path.dirname( path.dirname(path.abspath(__file__))), 'blas'))
import getspecs
import yaml
import subprocess
import math
import os
from collections import defaultdict
import re
from pathlib import Path
from git_info import create_github_file

# Parameters for output csv file
trackedParamList = ['function', 'precision', 'bench_command', 'a_type', 'b_type', 'c_type', 'd_type', 'compute_type', 'input_type', 'output_type',
                    'transA', 'transB', 'uplo', 'diag', 'side', 'M', 'N', 'K', 'KL', 'KU', 'alpha', 'alphai', 'beta', 'betai',
                    'incx', 'incy', 'lda', 'ldb', 'ldd', 'stride_x', 'stride_y', 'stride_a', 'stride_b', 'stride_c', 'stride_d',
                    'batch_count', 'algo', 'solution_index', 'flags', 'iters', 'cold_iters', 'pointer_mode', 'num_perfs', 'sample_num']
                    # Add output (eg. gflops) in code dependent on what we want to record (gflops vs. gbytes)

# maps output of rocblas-bench to input to rocblas-bench
benchInputMap = {'function': 'function', 'precision': 'precision',
                 'a_type': 'a_type', 'b_type': 'b_type', 'c_type': 'c_type', 'd_type': 'd_type', 'compute_type': 'compute_type',
                 'transA': 'transposeA', 'transB': 'transposeB',
                 'uplo': 'uplo', 'diag': 'diag', 'side': 'side',
                 'M': 'sizem', 'N': 'sizen', 'K': 'sizek',
                 'KL': 'kl', 'KU': 'ku',
                 'alpha': 'alpha', 'alphai': 'alphai', 'beta': 'beta', 'betai': 'betai',
                 'incx': 'incx', 'incy': 'incy', 'lda': 'lda', 'ldb': 'ldb', 'ldd': 'ldd',
                 'stride_x': 'stride_x', 'stride_y': 'stride_y', 'stride_a': 'stride_a', 'stride_b': 'stride_b', 'stride_c': 'stride_c', 'stride_d': 'stride_d',
                 'batch_count': 'batch_count', 'algo': 'algo', 'solution_index': 'solution_index', 'flags': 'flags', 'iters': 'iters', 'cold_iters': 'cold_iters'
                }

outputIndex = trackedParamList.index('num_perfs')

exInputs = {'gemm': 'a_type',
            'trsm': 'a_type',
            'axpy': 'a_type'}

exOutputs = {'gemm': 'c_type',
             'trsm': 'a_type',
             'axpy': 'c_type'}

def args2Dict(args):
    if type(args) == dict:
        return args
    elif type(args) == str:
        # Extend to account for arguments with '-' in the middle
        arguments = args.split(' ')
        res =[(key.replace('-', ''),
                      value if key.startswith('-') and not value.startswith('-') else '')
                      for key, value in zip(arguments, arguments[1:])]
        return dict(res)



def getFunctionTypes(yamlArgs):
    print(yamlArgs)
    if '_ex' in yamlArgs['function']:
        function = yamlArgs['function'].split('_')
        return {'input_type': yamlArgs[exInputs[function[0]]],
                'output_type': yamlArgs[exOutputs[function[0]]],
                'compute_type': yamlArgs['compute_type']}
    else:
        return {'input_type': yamlArgs['precision'],
                'output_type': yamlArgs['precision'],
                'compute_type': yamlArgs['precision']}

def updateProblemList(problemList, testType, commitHash, problemDesc, efficiency):
    for test in problemList:
        if test['TestType']   == testType and \
           test['Parameters'] == problemDesc:
            if commitHash in test['Samples'].keys():
                test['Samples'][commitHash] += [efficiency]
            else:
                test['Samples'][commitHash] = [efficiency]
            break
    else:
        problemList += [{'TestType': testType,
                        'Parameters': problemDesc,
                        'Samples': {commitHash: [efficiency]}}]

def getEnvironmentInfo(device_num):
    rv = {}
    host_info = {}
    host_info['hostname'] = getspecs.gethostname()
    host_info['cpu info'] = getspecs.getcpu()
    host_info['ram'] = getspecs.getram()
    host_info['distro'] = getspecs.getdistro()
    host_info['kernel version'] = getspecs.getkernel()
    host_info['rocm version'] = getspecs.getrocmversion()
    rv['Host info'] = host_info

    device_info = {}
    device_info['device'] = getspecs.getdeviceinfo(device_num, False)
    device_info['vbios version'] = getspecs.getvbios(device_num, False)
    device_info['vram'] = getspecs.getvram(device_num, False)
    device_info['system clock'] = getspecs.getsclk(device_num, False)
    device_info['memory clock'] = getspecs.getmclk(device_num, False)
    rv['Device info'] = device_info

    return rv

#d is default dict or contains a superset of trackedParams
def extractTrackedParams(d):
    return [d[p] for p in trackedParamList]

def getBenchCommand(keys, vals):
    cmd = 'rocblas-bench '
    for key in keys:
        if key in benchInputMap:
            cmd += '--' + benchInputMap[key] + ' ' + vals[key] + ' '
    return cmd

def parseRocblasBenchOutput(output, yamlArgs):
    csvKeys = ''

    lineLists = []

    captureNext = False
    yamlArgs = yamlArgs.split('\n')
    for line in output.split('\n'):
        if captureNext:
            #print("CAPTURE=",line)
            #dd = defaultdict(str, args2Dict(yamlArgs[len(lineLists)]).items())
            #dd.update(getFunctionTypes(args2Dict(yamlArgs[len(lineLists)])))
            dd_output = defaultdict(str, zip(csvKeys, line.split(',')))
            benchCommand1 = getBenchCommand(csvKeys, dd_output)
            #dd.update(dd_output)
            dd = dd_output
            dd['bench_command'] = benchCommand1
            lineLists += [extractTrackedParams(dd)]

            captureNext = False
        elif line.startswith('function'):
            csvKeys = line.split(',')
            #print("KEYS=",csvKeys)
            captureNext = True
    return lineLists

def addSample(csvLists, newSample, perf_queries):
    newList = []
    matched = False
    for row in csvLists:
        # if new sample has same input params as row
        if row[:outputIndex] == newSample[:outputIndex] and not matched:
            cur_num_samples = row[trackedParamList.index('sample_num')]
            for i in range(len(perf_queries)):
                # outputIndex + 2 puts us just past end of trackedParamList
                # add 2 * len(perf_queries) to get past mean/median results
                # add cur_samples * (i + 1) + i to insert at end of this perf_query output
                idx = outputIndex + 2 + 2 * len(perf_queries) + (cur_num_samples) * (i + 1) + i
                if newSample[trackedParamList.index(perf_queries[i])].strip() != '':
                    row.insert(idx, newSample[trackedParamList.index(perf_queries[i])])
                else:
                    row.insert(idx, '0')
            row[trackedParamList.index('sample_num')] = row[trackedParamList.index('sample_num')] + 1
            matched = True
        newList += [row]

    # if this is the first sample with this combination of input params, create new row
    if not matched:
        num_perfs = 0
        for perf in perf_queries:
            if newSample[trackedParamList.index(perf)].strip() == '':
                newSample[trackedParamList.index(perf)] = '0'
        newSample[trackedParamList.index('num_perfs')] = len(perf_queries)
        newSample[trackedParamList.index('sample_num')] = 1
        newList += [newSample]

    return newList

def splitComplexScalars(problem):
    # Complex alpha/beta are output as "(real: complex)". PTS is expecting a single num for alpha and a single num for alphai
    # to represent real/complex parts of scalars. This function parses the current rocBLAS output and updates problems
    # to the representation PTS expectes.
    # pattern: "(" + [whitespace?] + [number] + [whitespace?] + ":" + [whitespace?] + [number] + [whitespace?] + ")"
    # note groups are used to group real and complex values if matched
    num_pattern = r'[+-]?\d*\.?\d*([eE][+-]?\d+)?'
    complex_scalar_pattern = r'\(\s*(' + num_pattern + r')\s*\:\s*(' + num_pattern + r')\s*\)'
    alpha_match = re.match(complex_scalar_pattern, problem[trackedParamList.index('alpha')])
    if alpha_match is not None:
        # groups are 1 and 3, subgroup to deal with exponent is skipped
        problem[trackedParamList.index('alpha')] = alpha_match.group(1)
        problem[trackedParamList.index('alphai')] = alpha_match.group(3)

    beta_match = re.match(complex_scalar_pattern, problem[trackedParamList.index('beta')])
    if beta_match is not None:
        problem[trackedParamList.index('beta')] = beta_match.group(1)
        problem[trackedParamList.index('betai')] = beta_match.group(3)

def calculateMean(row, param_idx = 0):
    # if we're recording gflops and gbytes for 3 samples,
    # the data looks as follows:
    # [gflops1, gflops2, gflops3, gbytes1, gbytes2, gbytes3] for example
    # the indexing here should support any # of samples, and any # of perf_queries recorded

    # outputIndex + 2 puts us just past end of trackedParamList
    # add 2 * len(perf_queries) to get past mean/median results
    # add cur_samples * i + i to get beginning of results for this perf_query output
    num_perfs = row[trackedParamList.index('num_perfs')]
    num_samples = int(row[trackedParamList.index('sample_num')])
    idx = outputIndex + 2 + 2 * num_perfs + (param_idx) * num_samples + param_idx
    l = [float(v) for v in row[idx:idx + num_samples]]
    return sum(l)/float(len(l))

def calculateMedian(row, param_idx = 0):
    num_perfs = row[trackedParamList.index('num_perfs')]
    num_samples = int(row[trackedParamList.index('sample_num')])
    idx = outputIndex + 2 + 2 * num_perfs + (param_idx) * num_samples + param_idx
    l = [float(v) for v in row[idx:idx + num_samples]]
    l.sort()
    return l[math.floor(len(l)/2)]


def saveRocblasBenchResults(rocblasBenchCommand, problemsYaml, samples, outputFile, perf_queries, res_queries):

    with open(problemsYaml, 'r') as file:
        problems = file.read()

    csvLists = [trackedParamList]

    for i in range(0, int(samples)):
        print('Sample ({}/{})'.format(i+1, samples))
        output = subprocess.check_output([rocblasBenchCommand,
                                          '--log_function_name',
                                          '--log_datatype',
                                          '--yaml', problemsYaml])

        output = output.decode('utf-8')

        print(output)

        benchResults = parseRocblasBenchOutput(output, problems)

        for sample in benchResults:
            csvLists = addSample(csvLists, sample, perf_queries)

    for problem in csvLists[1:]:
        # fix complex alpha/beta to expected alpha/alphai output
        splitComplexScalars(problem)
        for i in range(len(res_queries)):
            problem[trackedParamList.index('mean_' + res_queries[i])] = calculateMean(problem, i)
            problem[trackedParamList.index('median_' + res_queries[i])] = calculateMedian(problem, i)

    content = ''
    for line in csvLists:
        content += ','.join([str(e) for e in line])+'\n'

    with open(outputFile, 'w') as f:
        print("Writing results to {}".format(outputFile))
        f.write(content)

def main(args):
    if len(args) < 4:
        print('Usage:\n\tpython3 write_pts_report.py bench_executable path tag benchfile1 benchfile2...')
        return 0

    rocblasBenchCommand = args[0]
    dirName             = args[1]
    tag                 = args[2]
    filenames           = args[3:]

    # Can alter this to query different performance results output by rocblas-bench
    # perf_queries: what we read from rocblas-bench output
    # res_queries: what we write to .csv file.
    # Using "gflops" over rocblas-Gflops for backwards-compatibility reasons
    perf_queries = ['rocblas-Gflops', 'rocblas-GB/s']
    res_queries = ['gflops', 'rocblas-GB/s']

    # append to trackedParamList our performance output of interest
    global trackedParamList
    for perf in res_queries:
        trackedParamList += ['mean_' + perf]
        trackedParamList += ['median_' + perf]
    for perf in perf_queries:
        trackedParamList += [perf]

    for filename in filenames:
        print("==================================\n|| Running benchmarks from {}\n==================================".format(filename))
        filePrefix = Path(filename).stem
        subDirectory = os.path.join(dirName, "rocBLAS_PTS_Benchmarks_"+filePrefix, tag)
        Path(subDirectory).mkdir(parents=True, exist_ok=True)

        outputName, _ = os.path.splitext(os.path.basename(filename))

        saveRocblasBenchResults(rocblasBenchCommand,
                                filename,
                                5,
                                os.path.join(subDirectory, outputName+'_benchmark.csv'),
                                perf_queries, res_queries)

        environmentInfo = getEnvironmentInfo(0)

        with open(os.path.join(subDirectory, 'specs.txt'), 'w') as f:
            for category, params in environmentInfo.items():
                f.write(category+':\n')
                for key, value in params.items():
                    f.write('    '+str(key)+': '+str(value)+'\n')

        # Will only be correct if script is run from directory of the git repo associated
        # with the rocblas-bench executable
        create_github_file(os.path.join(subDirectory, 'rocBLAS-commit-hash.txt'))

if __name__=='__main__':
    main(sys.argv[1:])

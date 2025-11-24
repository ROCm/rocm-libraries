# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import origami
import math
import yaml
from pathlib import Path

KERNEL_COLUMNS = ['MT_M', 'MT_N', 'MT_K', 'MI_M', 'MI_N', 'MI_K', 'Occupancy', 'WGM', 'NTA', 'NTB']

def parseArguments():
    parser = argparse.ArgumentParser(description="Test Solution Ranking.")
    parser.add_argument("--device", type=int, default=0, help="Device ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--print", action="store_true", help="Print hardware info")
    parser.add_argument(
        "--yaml", type=str, required=True, help="Yaml file containing sizes to benchmark")
    
    args = parser.parse_args()
    return args

def build_tile_list(library_name):
    tilelist = []
    with open("./tiles_library/" + library_name, "rt") as libinput:
        solutions = yaml.load(libinput, Loader=yaml.SafeLoader)
    for item in solutions:
        tilelist.append(tuple([item[col] for col in KERNEL_COLUMNS]))
    return tilelist

def read_configs_list(library_name):
    configs_list = []
    with open("./tiles_library/" + library_name, "rt") as libinput:
        solutions = yaml.load(libinput, Loader=yaml.SafeLoader)
    for item in solutions:
        config = origami.config_t()
        config.mt = origami.dim3_t(item['MT_M'], item['MT_N'], item['MT_K'])
        config.mi = origami.dim3_t(item['MI_M'], item['MI_N'], item['MI_K'])
        config.occupancy = item['Occupancy']
        config.workgroup_mapping = item['WGM']
        config.cache_hints_a = item['NTA']
        config.cache_hints_b = item['NTB']
        configs_list.append(config)
    return configs_list

def main():
    args = parseArguments()
    
    benchyaml = args.yaml
    if not(Path(benchyaml).is_file()):
        print(f"Yaml file {args.yaml} not found.")
        exit(1)

    hardware = origami.get_hardware_for_device(args.device)

    if args.print:
        hardware.print()

    with open(args.yaml, "rt") as finput:
        benchmarks = yaml.load(finput, Loader=yaml.SafeLoader)
    
    print(f"Loaded yaml with {len(benchmarks)} size(s).")
    resultsfile = Path(benchyaml).stem + ".out"

    # Load Baseline
    with open("./baseline/results/" + resultsfile, "rt") as finput:
        base_rankings = yaml.load(finput, Loader=yaml.SafeLoader)

    prefix = 'gfx950_Cijk_'
    transA_coord = {'N': 'Ailk', 'T': 'Alik'}
    transB_coord = {'N': 'Bljk', 'T': 'Bjlk'}
    datadict = {"bf16": "BBS", "f32": "SB", "xf32": "S_MX"}
    hblt_to_origami = {"bf16_r": "bf16", "f32_r": "f32"}
    
    lib_cache = {}
    sizes_list = []
    for idx, bench_item in enumerate(benchmarks):
        solutionlib = prefix + transA_coord[bench_item['transA']]
        solutionlib += "_" + transB_coord[bench_item['transB']]
    
        if bench_item['a_type'] == 'bf16_r':
            solutionlib += "_" + datadict['bf16']
        elif bench_item['compute_type'] == 'c_xf32_r':
            solutionlib += "_" + datadict['xf32']
        else:
            solutionlib += "_" + datadict['f32']
        solutionlib += "_tiles.yaml"
        
        if solutionlib in lib_cache:
            configs_list = lib_cache[solutionlib]
        else:
            configs_list = read_configs_list(solutionlib)
            lib_cache[solutionlib] = configs_list
        
        problem = origami.problem_t()
        problem.size = origami.dim3_t(bench_item['M'], bench_item['N'], bench_item['K'])
        problem.batch = bench_item['batch_count']
        problem.transpose_a = origami.transpose_t.T if bench_item['transA']=='T' else origami.transpose_t.N
        problem.transpose_b = origami.transpose_t.T if bench_item['transB']=='T' else origami.transpose_t.N
        problem.a_dtype = origami.string_to_datatype(hblt_to_origami[bench_item['a_type']])
        problem.b_dtype = origami.string_to_datatype(hblt_to_origami[bench_item['b_type']]) 
        problem.d_dtype = origami.string_to_datatype(hblt_to_origami[bench_item['d_type']])
        problem.c_dtype = problem.d_dtype
        problem.mi_dtype = problem.a_dtype
        problem.a_mx_block_size = 0
        problem.b_mx_block_size = 0

        rv = origami.select_ranked_configs(problem, hardware, configs_list)
        results_list = []
        for rank_item in rv:
            if rank_item.latency < 1e12:
                results_list.append({'Latency': round(rank_item.latency), \
                                     'MT_M': rank_item.config.mt.m, \
                                     'MT_N': rank_item.config.mt.n, \
                                     'MT_K': rank_item.config.mt.k, \
                                     'MI_M': rank_item.config.mi.m, \
                                     'MI_N': rank_item.config.mi.n, \
                                     'MI_K': rank_item.config.mi.k, \
                                     'Occupancy': rank_item.config.occupancy, \
                                     'WGM': rank_item.config.workgroup_mapping, \
                                     'NTA': rank_item.config.cache_hints_a, \
                                     'NTB': rank_item.config.cache_hints_b})
        sizes_list.append(results_list)

        # count = len(base_rankings[idx])
        # for (base_item, new_item) in zip(base_rankings[idx], rv[:count]):
            # assert base_item['Latency']==round(new_item.latency) and \
            #        base_item['MT_M']==new_item.config.mt.m and \
            #        base_item['MT_N']==new_item.config.mt.n and \
            #        base_item['MT_K']==new_item.config.mt.k and \
            #        base_item['MI_M']==new_item.config.mi.m and \
            #        base_item['MI_N']==new_item.config.mi.n and \
            #        base_item['MI_K']==new_item.config.mi.k and \
            #        base_item['Occupancy']==new_item.config.occupancy and \
            #        base_item['WGM']==new_item.config.workgroup_mapping and \
            #        base_item['NTA']==new_item.config.cache_hints_a and \
            #        base_item['NTB']==new_item.config.cache_hints_b, "Solution ranking not identical"
            # print(idx)
            # if base_item['Latency']!=round(new_item.latency):
            #     print(base_item['Latency'], round(new_item.latency))
            # if base_item['MT_M']!=new_item.config.mt.m:
            #     print(base_item['MT_M'], new_item.config.mt.m)
            # if base_item['MT_N']!=new_item.config.mt.n:
            #     print(base_item['MT_N'], new_item.config.mt.n)
            # if base_item['MT_K']!=new_item.config.mt.k:
            #     print(base_item['MT_K'], new_item.config.mt.k)
            # if base_item['MI_M']!=new_item.config.mi.m:
            #     print(base_item['MI_M'], new_item.config.mi.m)
            # if base_item['MI_N']!=new_item.config.mi.n:
            #     print(base_item['MI_N'], new_item.config.mi.n)
            # if base_item['MI_K']!=new_item.config.mi.k:
            #     print(base_item['MI_K'], new_item.config.mi.k)
            # if base_item['Occupancy']!=new_item.config.occupancy:
            #     print(base_item['Occupancy'], new_item.config.occupancy)
            # if base_item['WGM']!=new_item.config.workgroup_mapping:
            #     print(base_item['WGM'], new_item.config.workgroup_mapping)
            # if base_item['NTA']!=new_item.config.cache_hints_a:
            #     print(base_item['NTA'], new_item.config.cache_hints_a)
            # if base_item['NTB']!=new_item.config.cache_hints_b:
            #     print(base_item['NTB'], new_item.config.cache_hints_b)
        # print(f"Benchmark {idx} matches")
    yaml_args = {"default_flow_style": None, "sort_keys": False}
    with open("./bench_sizes/results/" + resultsfile, "wt") as foutput:
        yaml.dump(sizes_list, foutput, **yaml_args)

if __name__ == "__main__":
    exit(main())
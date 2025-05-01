#!/usr/bin/env -S python3

from datetime import datetime
import json
import json5
import os
import re
import pandas
from pandas import DataFrame
import seaborn as sns
from matplotlib import pyplot as plt
import argparse
from jinja2 import Environment, PackageLoader

env = Environment(
    loader=PackageLoader("select_best_config"),
    lstrip_blocks=True,
    trim_blocks=True
)


def load_default_configs_json(in_dir: str) -> dict:
    """
    Reads the config_defaults.json object for rocRAND's generators and returns them as dict.
    """

    with open(os.path.join(in_dir, "config_defaults.json")) as f:
        json_data = json5.load(f)

    return json_data

def load_benchmark_result_json(path: str) -> DataFrame:
    """
    Reads and processes a JSON file produced by rocRAND's tuning benchmark.
    """
    def guess_gcn_architecture(path: str, raw_json_data: dict):
        arch_regex = re.compile(r'gfx[0-9a-f]+')
        arch_name = raw_json_data['context']['hdp_gcn_arch_name']
        _, filename = os.path.split(path)
        for name in (arch_name, filename):
            gfx_match = re.search(arch_regex, name)
            if gfx_match:
                return gfx_match.group(0)
        return 'unknown'

    with open(path) as f:
        raw_json_data = json.load(f)
    benchmark_data = DataFrame(raw_json_data['benchmarks'])

    gb_per_s_series = benchmark_data['bytes_per_second'] / 1000 / 1000 / 1000
    gb_per_s_series.name = 'gb_per_s'

    # Extract the groups of the regex match to a DataFrame
    
    name_regex = r'^(?P<generator>\S+?)_(?P<distribution>uniform|normal|log_normal|poisson)_(?P<value_type>(unsigned_)?(int|short|char|long_long|float|half|double))_t(?P<block_size>\d+)_b(?P<grid_size>\d+)'

    extracted_data = benchmark_data['name'].str.extract(name_regex)
    extracted_data['block_size'] = extracted_data['block_size'].astype(int)
    extracted_data['grid_size']  = extracted_data['grid_size'].astype(int)

    # Merge the regex matches and the gb_per_s series
    benchmark_data = pandas.concat([extracted_data, gb_per_s_series], axis=1)

    # Figure out architecture
    benchmark_data['arch'] = guess_gcn_architecture(path, raw_json_data)
    return benchmark_data


def get_best_config_for_arch(benchmark_data: DataFrame, default_configs: dict):
    """
    Calculates the best config for each architecture, that on average provides the highest
    performance across generated types and distributions.
    """
    def config_to_goodness(subf: DataFrame) -> pandas.Series:
        relative_perfs = subf['normalized_perf'].reset_index(
            drop=True)
        # Averages the performance over the different distribution/value_types
        # to get a single average performance per configuration of the generator
        return relative_perfs.sum() / relative_perfs.count()

    best_configs = {}
    for gen, generator_df in benchmark_data.groupby('generator'):
        best_config_for_generator = {}
        default_block_size = default_configs[gen]['block_size']
        default_grid_size  = default_configs[gen]['grid_size']

        for arch, arch_df in generator_df.groupby('arch'):

            temp_df = DataFrame()
            # Normalize other configurations wrt default config per distribution/value_type
            for _, perf_df in arch_df.groupby(['distribution', 'value_type']):
                default_config_perf = perf_df.loc[
                    (perf_df['block_size'] == default_block_size) &
                    (perf_df['grid_size']  == default_grid_size)
                ].gb_per_s.iloc[0]
                perf_df['normalized_perf'] = perf_df.gb_per_s / default_config_perf
                temp_df = pandas.concat([temp_df, perf_df])

            arch_df = temp_df
            config_goodness = arch_df.groupby(
                ['block_size', 'grid_size']).apply(config_to_goodness)
            config_goodness = config_goodness.sort_values(ascending=False)
            config_goodness = config_goodness.to_frame(name='normalized_perf').reset_index()

            # config_goodness is sorted by normalized_perf, so just take the first element
            best_block_size = config_goodness.iloc[0]['block_size'].astype(int)
            best_grid_size  = config_goodness.iloc[0]['grid_size'].astype(int)
            best_perf       = config_goodness.iloc[0]['normalized_perf']

            # check for performance regressions and possible alternative configurations
            detailed_performance = arch_df[(arch_df['block_size'] == best_block_size) &
                                           (arch_df['grid_size']  == best_grid_size)]
            if (detailed_performance['normalized_perf'] < 0.98).any():
                print("WARNING: configuration with best average performance has performance regressions for some distributions:")
                print("Average speedup: ", best_perf)
                print(detailed_performance.to_string())
                # set index to config for easier looping
                config_goodness = config_goodness.set_index(['block_size', 'grid_size'])
                other_configs = config_goodness[(config_goodness['normalized_perf'] > 1.0) &
                                                (config_goodness['normalized_perf'] != best_perf)]
                if len(other_configs.index) == 0:
                    print("\nNo other configs available that provide an average speedup over the default config!\n")
                # check other configs that have an average speedup for regressions
                found_config_without_regressions = False
                for config, row in other_configs.iterrows():
                    block_size, grid_size = config
                    cur_detailed_perf = arch_df[(arch_df['block_size'] == block_size) &
                                                (arch_df['grid_size']  == grid_size)]
                    if (cur_detailed_perf['normalized_perf'] > 1.0).all():
                        print("Next best config that provides an average speedup over the default config without regressions:")
                        print("Average speedup: ", row['normalized_perf'])
                        print(cur_detailed_perf.to_string())
                        print("\n")
                        found_config_without_regressions = True
                        break
                if not found_config_without_regressions:
                    print("No other config provides a speedup over the default config without any regressions for certain distributions!\n")

            best_config_for_generator[arch] = {
                'block_size': best_block_size, 'grid_size': best_grid_size}
            generator_df.loc[(generator_df['generator'] == gen) & (generator_df['arch'] == arch)] = arch_df
        best_configs[gen] = best_config_for_generator
    return best_configs


def plot_benchmark_data_for_all_arches(benchmark_data: DataFrame, out_path: str):
    """
    Plots comparative figures of the benchmark results for each architecture.
    """
    def plot_benchmark_results(benchmark_data: DataFrame, arch: str, gen: str, out_path: str):
        def draw_heatmap(*args, **kwargs):
            data: DataFrame = kwargs.pop('data')
            d = data.pivot_table(
                index=args[1], columns=args[0], values=args[2])
            sns.heatmap(d, **kwargs)

        grid = sns.FacetGrid(benchmark_data, col='benchmark', col_wrap=3)
        grid.map_dataframe(draw_heatmap, 'grid_size_', 'block_size_', 'gb_per_s',
                           cbar=False, square=True, annot=True, fmt='.2f', cmap='coolwarm', linewidth=1)
        fig = plt.gcf()
        fig.suptitle(f'rocRAND benchmarks ({gen} on {arch}, GB/s)')
        num_grid_sizes = benchmark_data['grid_size'].unique().astype(
            bool).sum()
        fig.set_size_inches(16/6*num_grid_sizes, 18)
        fig.savefig(out_path)
        plt.close()

    benchmark_data['block_size_'] = benchmark_data['block_size'].astype(str).str.pad(
        4, 'left', ' ')
    benchmark_data['grid_size_'] = benchmark_data['grid_size'].astype(str).str.pad(
        4, 'left', ' ')
    benchmark_data['benchmark'] = benchmark_data['value_type'].astype(
        str) + ',' + benchmark_data['distribution'].astype(str)
    base_path, full_filename = os.path.split(out_path)
    filename, ext = os.path.splitext(full_filename)
    for arch in benchmark_data['arch'].unique():
        arch_benchmark_data = benchmark_data[benchmark_data['arch'] == arch]
        for gen in arch_benchmark_data['generator'].unique():
            gen_benchmark_data = arch_benchmark_data[arch_benchmark_data['generator'] == gen]
            plot_benchmark_results(gen_benchmark_data, arch, gen, os.path.join(base_path, f'{filename}_{arch}_{gen}{ext}'))


def generate_config_files(out_dir: str, best_data: dict):
    def generator_type(generator_name: str) -> str:
        if 'sobol' in generator_name.lower():
            return 'quasi'
        return 'pseudo'

    year = datetime.now().year
    algorithm_template = env.get_template("config_template")
    for generator, best_configs in best_data.items():
        with open(os.path.join(out_dir,  f"{generator}_config.hpp"), "w") as text_file:
            text_file.write(algorithm_template.render(
                generator=generator, generator_type=generator_type(generator), configs=best_configs, year=year))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', nargs='+')
    parser.add_argument('--plot-out', help="Output directory for plots. Plots will only be generated, when this argument is provided. Generating these can take quite some while.")
    parser.add_argument('--out-dir')
    parser.add_argument('--default-config-dir',
                        default=os.path.dirname(os.path.realpath(__file__)),
                        help="Path to directory of 'config_defaults.json'")
    args = parser.parse_args()

    default_configs = load_default_configs_json(args.default_config_dir)
    benchmark = pandas.concat(
        [load_benchmark_result_json(j) for j in args.json], ignore_index=True)
    best_configs = get_best_config_for_arch(benchmark, default_configs)
    if args.plot_out:
        plot_benchmark_data_for_all_arches(benchmark, args.plot_out)
    if args.out_dir:
        generate_config_files(args.out_dir, best_configs)

# pip install pandas
# python -m pip install -U matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys
parser = argparse.ArgumentParser(description='Generate Benchmark plots for the given csv file')
parser.add_argument('--csv_fp', help='path to the csv file')
parser.add_argument('--plots_loc', default=max, help='location to write the plots to')
args = parser.parse_args()

if(len(sys.argv) < 2):
  print("Please provide all arguments, csv_fp - path to the csv file and plots_loc - location to store the plots")
  exit(0)
df = pd.read_csv(args.csv_fp, sep="\t")
lngth = len(df.columns)
col = list(df.columns)
skipcol = col[lngth - 1]
df = df[df[skipcol].notna()]
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 3000)

df.columns = df.columns.str.strip()

df = df[df[skipcol].str.contains("SKIPPED")==False]
grouped = df.groupby('Ti_To_Tc')

for key, item in grouped:
    df1 = grouped.get_group(key)
    df1 = df1.sort_values(['BlkM', 'BlkK', 'LytA_LytB_LytC_LytD', 'GFlops'])

    groupMxN = df1.groupby(['BlkM','BlkN'])
    for key1, item1 in groupMxN:
        df2 = groupMxN.get_group(key1)

        groupK = df2.groupby('BlkK')
        firstK = groupK.get_group(groupK.groups.keys()[0])

        groupLayout = firstK.groupby('LytA_LytB_LytC_LytD')
        ng = groupLayout.ngroups

        m, n = key1
        fig, axs = plt.subplots(1, ng, sharex=True, sharey=True)
        fig.suptitle(str(key) + "-" + str(m) + "x" + str(n), fontsize=15)

        for key2, item2 in groupK:
            dfK = groupK.get_group(key2)
            groupLayout = dfK.groupby('LytA_LytB_LytC_LytD')

            targets = zip(groupLayout.groups.keys(), axs.flatten())

            for (key3, ax) in targets:
                dfLayout = groupLayout.get_group(key3)

                dfLayout["GFlops"] = dfLayout["GFlops"].astype(np.float32)
                dfLayout["Efficiency"] = dfLayout["Efficiency"].astype(np.float32)
                dfLayout = dfLayout.sort_values('GFlops')

                x = dfLayout["GFlops"]
                y = dfLayout["Efficiency"]

                ax.plot(np.arange(len(x)), y, label=key2)
                ax.set_xticklabels(x, rotation=30, horizontalalignment='right', fontsize='x-small')
                ax.set_title(key3, loc='center',fontsize='small')

        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        fig.legend(*zip(*unique),  title="BlkK",loc='best')

        fname = (str(key) + "-" + str(m) + "x" + str(n) + ".png").strip()
        plt.savefig(args.plots_loc+"/"+fname)
        plt.close()

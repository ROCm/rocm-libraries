#!/usr/bin/env python3
import argparse
import re
import csv
from collections import defaultdict
import math

def parse_log(path):
    recs = defaultdict(list)
    header = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('[0]:'):
                header = [h.strip() for h in line.split(':',1)[1].split(',')]
            elif header and re.match(r'^[A-Za-z]', line):
                vals = [v.strip() for v in line.split(',')]
                if len(vals) == len(header):
                    d = dict(zip(header, vals))
                    try:
                        m = int(d['m'])
                        n = int(d['n'])
                        k = int(d['k'])
                        a_t = d.get('a_type', 'unknown')
                        b_t = d.get('b_type', 'unknown')
                        c_t = d.get('c_type', 'unknown')
                        d_t = d.get('d_type', 'unknown')
                        gflops = float(d['hipblaslt-Gflops'])
                        recs[(m, n, k, a_t, b_t, c_t, d_t)].append(gflops)
                    except (KeyError, ValueError):
                        pass
                header = None
    return recs

def main():
    p = argparse.ArgumentParser(
        description="Produce CSV of pointwise Origami vs Other performance and print average percent"
    )
    p.add_argument('--origami', required=True, help="origami log file")
    p.add_argument('--other',   required=True, help="other log file")
    p.add_argument('--out',     default='comparison.csv',
                   help="output CSV path (default: comparison.csv)")
    args = p.parse_args()

    orig = parse_log(args.origami)
    oth  = parse_log(args.other)

    ratios = []
    count_origami_faster = 0
    count_other_faster   = 0

    with open(args.out, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'm','n','k',
            'a_type','b_type','c_type','d_type',
            'origami_gflops','other_gflops','perf_pct'
        ])
        for key in sorted(set(orig) & set(oth)):
            m, n, k, a_t, b_t, c_t, d_t = key
            ga = orig[key][0]
            gb = oth[key][0]
            if gb != 0 and not math.isnan(ga) and not math.isnan(gb):
                pct = (ga/gb)*100
                ratios.append(pct)
                pct_str = f"{pct:.1f}"
                if ga > gb:
                    count_origami_faster += 1
                elif gb > ga:
                    count_other_faster += 1
            else:
                pct_str = ""
            writer.writerow([
                m, n, k,
                a_t, b_t, c_t, d_t,
                f"{ga:.3f}", f"{gb:.3f}", pct_str
            ])

    print(f"Wrote comparison for {len(ratios)} entries to {args.out}")

    suffix = ""


    if ratios:
        avg_pct = sum(ratios) / len(ratios)
        if avg_pct < 100:
            suffix = "It's so over"
        else:
            suffix = "We are so back"
        print(f"Average performance of Origami: {avg_pct:.1f}% - {suffix}")
        
        if count_other_faster >= count_origami_faster:
            suffix = "It's so over"
        else:
            suffix = "We are so back"

        print(f"Number of problems where Origami was faster: {count_origami_faster} - {suffix}")
        print(f"Number of problems where Other   was faster: {count_other_faster}")
    else:
        print("No valid entries to compute average performance.")

if __name__ == "__main__":
    main()

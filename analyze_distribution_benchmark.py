#!/usr/bin/env python3
"""
Analyze and visualize GEMM benchmark results across different data distributions
"""

import json
import csv
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def load_json_results(filepath: Path) -> List[Dict]:
    """Load benchmark results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_csv_results(filepath: Path) -> List[Dict]:
    """Load benchmark results from CSV file"""
    results = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'distribution': row['Distribution'],
                'init_method': int(row['Init_Method']),
                'latency_ms': float(row['Latency_ms']),
                'tflops': float(row['TFlops']),
                'bandwidth_gbps': float(row['Bandwidth_GBps'])
            })
    return results

def analyze_results(results: List[Dict]) -> Dict:
    """Analyze benchmark results and compute statistics"""
    if not results:
        return {}

    analysis = {
        'count': len(results),
        'metrics': {}
    }

    # Extract metrics
    for metric in ['latency_ms', 'tflops', 'bandwidth_gbps']:
        values = [r[metric] for r in results if metric in r]
        if not values:
            values = [r['performance'][metric.replace('_ms', '').replace('_gbps', '').replace('_', '_')]
                     for r in results if 'performance' in r]

        if values:
            analysis['metrics'][metric] = {
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'range': max(values) - min(values),
                'relative_range_pct': ((max(values) - min(values)) / min(values) * 100) if min(values) > 0 else 0
            }

    # Find best and worst for each metric
    analysis['best'] = {}
    analysis['worst'] = {}

    # For latency, lower is better
    latency_sorted = sorted(results, key=lambda x: x.get('latency_ms', x.get('performance', {}).get('latency_ms', float('inf'))))
    if latency_sorted:
        analysis['best']['latency'] = latency_sorted[0].get('distribution', 'Unknown')
        analysis['worst']['latency'] = latency_sorted[-1].get('distribution', 'Unknown')

    # For tflops and bandwidth, higher is better
    tflops_sorted = sorted(results, key=lambda x: x.get('tflops', x.get('performance', {}).get('tflops', 0)), reverse=True)
    if tflops_sorted:
        analysis['best']['tflops'] = tflops_sorted[0].get('distribution', 'Unknown')
        analysis['worst']['tflops'] = tflops_sorted[-1].get('distribution', 'Unknown')

    bandwidth_sorted = sorted(results, key=lambda x: x.get('bandwidth_gbps', x.get('performance', {}).get('bandwidth_gbps', 0)), reverse=True)
    if bandwidth_sorted:
        analysis['best']['bandwidth'] = bandwidth_sorted[0].get('distribution', 'Unknown')
        analysis['worst']['bandwidth'] = bandwidth_sorted[-1].get('distribution', 'Unknown')

    return analysis

def print_analysis(analysis: Dict):
    """Print analysis results in a formatted way"""
    print("\n" + "="*70)
    print("BENCHMARK ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nTotal distributions tested: {analysis['count']}")

    print("\n" + "-"*70)
    print("PERFORMANCE METRICS STATISTICS")
    print("-"*70)

    for metric_name, stats in analysis['metrics'].items():
        display_name = metric_name.replace('_', ' ').title()
        unit = {
            'Latency Ms': 'ms',
            'Tflops': 'TFlops',
            'Bandwidth Gbps': 'GB/s'
        }.get(display_name, '')

        print(f"\n{display_name}:")
        print(f"  Min:            {stats['min']:.2f} {unit}")
        print(f"  Max:            {stats['max']:.2f} {unit}")
        print(f"  Mean:           {stats['mean']:.2f} {unit}")
        print(f"  Range:          {stats['range']:.2f} {unit}")
        print(f"  Variation:      {stats['relative_range_pct']:.2f}%")

    print("\n" + "-"*70)
    print("BEST & WORST PERFORMERS")
    print("-"*70)

    print(f"\nLowest Latency:     {analysis['best']['latency']}")
    print(f"Highest Latency:    {analysis['worst']['latency']}")
    print(f"\nHighest TFlops:     {analysis['best']['tflops']}")
    print(f"Lowest TFlops:      {analysis['worst']['tflops']}")
    print(f"\nHighest Bandwidth:  {analysis['best']['bandwidth']}")
    print(f"Lowest Bandwidth:   {analysis['worst']['bandwidth']}")

    print("\n" + "="*70)

def generate_markdown_report(results: List[Dict], analysis: Dict, output_file: Path):
    """Generate a Markdown report"""
    with open(output_file, 'w') as f:
        f.write("# GEMM Data Distribution Benchmark Results\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Total Distributions Tested:** {analysis['count']}\n")
        f.write(f"- **Matrix Size:** 8192x8192x8192\n\n")

        f.write("## Performance Metrics\n\n")
        f.write("| Metric | Min | Max | Mean | Variation |\n")
        f.write("|--------|-----|-----|------|----------|\n")

        for metric_name, stats in analysis['metrics'].items():
            display_name = metric_name.replace('_', ' ').title()
            f.write(f"| {display_name} | {stats['min']:.2f} | {stats['max']:.2f} | {stats['mean']:.2f} | {stats['relative_range_pct']:.1f}% |\n")

        f.write("\n## Detailed Results\n\n")
        f.write("| Distribution | Latency (ms) | TFlops | Bandwidth (GB/s) |\n")
        f.write("|--------------|--------------|--------|------------------|\n")

        for r in results:
            dist = r.get('distribution', 'Unknown')
            lat = r.get('latency_ms', r.get('performance', {}).get('latency_ms', 0))
            tf = r.get('tflops', r.get('performance', {}).get('tflops', 0))
            bw = r.get('bandwidth_gbps', r.get('performance', {}).get('bandwidth_gbps', 0))
            f.write(f"| {dist} | {lat:.2f} | {tf:.2f} | {bw:.2f} |\n")

        f.write("\n## Key Findings\n\n")
        f.write(f"- **Fastest Distribution:** {analysis['best']['latency']} (lowest latency)\n")
        f.write(f"- **Highest Throughput:** {analysis['best']['tflops']} (highest TFlops)\n")
        f.write(f"- **Best Memory Performance:** {analysis['best']['bandwidth']} (highest bandwidth)\n")

        # Check if performance is data-dependent
        lat_variation = analysis['metrics'].get('latency_ms', {}).get('relative_range_pct', 0)
        if lat_variation > 5:
            f.write(f"\n### Data Distribution Impact\n\n")
            f.write(f"Performance shows **{lat_variation:.1f}% variation** across different data distributions, ")
            f.write(f"indicating that the kernel performance **IS dependent on data patterns**.\n")
        else:
            f.write(f"\n### Data Distribution Impact\n\n")
            f.write(f"Performance shows only **{lat_variation:.1f}% variation** across different data distributions, ")
            f.write(f"indicating that the kernel performance is **largely independent of data patterns**.\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze GEMM benchmark results across data distributions')
    parser.add_argument('input_file', type=Path, help='Input file (JSON or CSV)')
    parser.add_argument('--output', '-o', type=Path, help='Output Markdown report file')
    parser.add_argument('--format', '-f', choices=['json', 'csv', 'auto'], default='auto',
                       help='Input file format (default: auto-detect)')

    args = parser.parse_args()

    # Validate input file
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # Detect format
    if args.format == 'auto':
        args.format = 'json' if args.input_file.suffix == '.json' else 'csv'

    # Load results
    print(f"Loading results from {args.input_file} ({args.format})...")
    if args.format == 'json':
        results = load_json_results(args.input_file)
    else:
        results = load_csv_results(args.input_file)

    if not results:
        print("Error: No results found in input file", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(results)} benchmark results")

    # Analyze results
    analysis = analyze_results(results)

    # Print analysis to console
    print_analysis(analysis)

    # Generate Markdown report if requested
    if args.output:
        print(f"\nGenerating Markdown report: {args.output}")
        generate_markdown_report(results, analysis, args.output)
        print(f"Report saved to {args.output}")

if __name__ == '__main__':
    main()

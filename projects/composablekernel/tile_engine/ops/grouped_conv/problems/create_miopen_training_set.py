#!/usr/bin/env python3
"""
Extract diverse training problems from MIOpen ALL_CONFIGS_FULL.txt

This script samples the most diverse shapes from MIOpen data to supplement
our training dataset with real-world production shapes.
"""

import re
import random
from pathlib import Path
from collections import defaultdict

def parse_miopen_line(line):
    """Parse MIOpenDriver command line to extract problem parameters."""
    N = re.search(r'--batchsize (\d+)', line)
    C = re.search(r'--in_channels (\d+)', line)
    K = re.search(r'--out_channels (\d+)', line)
    H = re.search(r'--in_h (\d+)', line)
    W = re.search(r'--in_w (\d+)', line)
    Y = re.search(r'--fil_h (\d+)', line)
    X = re.search(r'--fil_w (\d+)', line)
    G = re.search(r'--group_count (\d+)', line)
    stride_h = re.search(r'--conv_stride_h (\d+)', line)
    stride_w = re.search(r'--conv_stride_w (\d+)', line)
    pad_h = re.search(r'--pad_h (\d+)', line)
    pad_w = re.search(r'--pad_w (\d+)', line)

    if not all([N, C, K, H, W, Y, X, G, stride_h, stride_w, pad_h, pad_w]):
        return None

    return {
        'N': int(N.group(1)),
        'C': int(C.group(1)),
        'K': int(K.group(1)),
        'G': int(G.group(1)),
        'Hi': int(H.group(1)),
        'Wi': int(W.group(1)),
        'Y': int(Y.group(1)),
        'X': int(X.group(1)),
        'stride_h': int(stride_h.group(1)),
        'stride_w': int(stride_w.group(1)),
        'pad_h': int(pad_h.group(1)),
        'pad_w': int(pad_w.group(1)),
    }

def is_valid_shape(prob):
    """Check if shape meets our requirements."""
    # Channel alignment and minimum requirements
    if prob['C'] < 64 or prob['C'] % 16 != 0:
        return False
    if prob['K'] < 64 or prob['K'] % 16 != 0:
        return False

    # Limit batch size to avoid GPU memory issues
    if prob['N'] > 16:
        return False

    # Only forward convolution (group_count constraint)
    # For now, support both grouped and standard convolution
    if prob['G'] > prob['C'] or prob['G'] > prob['K']:
        return False

    # Reasonable spatial dimensions
    if prob['Hi'] < 1 or prob['Wi'] < 1:
        return False

    # Reasonable filter sizes
    if prob['Y'] > 11 or prob['X'] > 11:
        return False

    return True

def get_shape_signature(prob):
    """Create a signature for diversity bucketing."""
    # Bucket by key characteristics
    c_bucket = (prob['C'] // 64) * 64  # Bucket channels by 64
    k_bucket = (prob['K'] // 64) * 64
    spatial_bucket = f"{prob['Hi']}x{prob['Wi']}"
    filter_bucket = f"{prob['Y']}x{prob['X']}"
    stride_bucket = f"s{prob['stride_h']}x{prob['stride_w']}"

    return (c_bucket, k_bucket, spatial_bucket, filter_bucket, stride_bucket, prob['G'])

def main():
    all_configs = Path(__file__).parent / "ALL_CONFIGS_FULL.txt"

    # Parse all valid shapes from MIOpen data
    print("Parsing MIOpen configurations...")
    valid_shapes = []
    bucket_to_shapes = defaultdict(list)

    with open(all_configs, 'r') as f:
        for i, line in enumerate(f):
            if i % 50000 == 0:
                print(f"  Processed {i} lines, found {len(valid_shapes)} valid shapes...")

            prob = parse_miopen_line(line)
            if prob and is_valid_shape(prob):
                valid_shapes.append(prob)
                signature = get_shape_signature(prob)
                bucket_to_shapes[signature].append(prob)

    print(f"\nFound {len(valid_shapes)} valid shapes across {len(bucket_to_shapes)} unique buckets")

    # Sample diverse shapes
    # Strategy: Take 1-2 samples from each bucket to maximize diversity
    sampled_shapes = []
    for bucket, shapes in bucket_to_shapes.items():
        # Take up to 2 samples from each bucket
        sample_count = min(2, len(shapes))
        samples = random.sample(shapes, sample_count)
        sampled_shapes.extend(samples)

    # Limit to reasonable number and shuffle
    target_count = min(300, len(sampled_shapes))
    random.shuffle(sampled_shapes)
    sampled_shapes = sampled_shapes[:target_count]

    print(f"\nSampled {len(sampled_shapes)} diverse shapes")

    # Analyze diversity
    print("\nDiversity analysis:")
    channels_c = sorted(set(s['C'] for s in sampled_shapes))
    channels_k = sorted(set(s['K'] for s in sampled_shapes))
    batch_sizes = sorted(set(s['N'] for s in sampled_shapes))
    spatial_sizes = len(set((s['Hi'], s['Wi']) for s in sampled_shapes))
    filter_sizes = len(set((s['Y'], s['X']) for s in sampled_shapes))

    print(f"  Unique C values: {len(channels_c)} (range: {min(channels_c)} to {max(channels_c)})")
    print(f"  Unique K values: {len(channels_k)} (range: {min(channels_k)} to {max(channels_k)})")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Unique spatial sizes: {spatial_sizes}")
    print(f"  Unique filter sizes: {filter_sizes}")

    # Generate Python module
    output_file = Path(__file__).parent / "forward_training_miopen.py"

    with open(output_file, 'w') as f:
        f.write('#!/usr/bin/env python3\n\n')
        f.write('# Training problem set from MIOpen production shapes\n')
        f.write(f'# {len(sampled_shapes)} diverse problems sampled from MIOpen ALL_CONFIGS_FULL.txt\n')
        f.write('# Filtered for C >= 64, K >= 64, all channels aligned to 16, N <= 16\n\n')
        f.write('import sys\n')
        f.write('from pathlib import Path\n\n')
        f.write('# Add dispatcher/python to path for grouped_conv_utils import\n')
        f.write('dispatcher_python = Path(__file__).resolve().parents[4] / "dispatcher" / "python"\n')
        f.write('sys.path.insert(0, str(dispatcher_python))\n\n')
        f.write('from grouped_conv_utils import GroupedConvProblem\n\n')
        f.write('TRAINING_PROBLEMS_FORWARD_MIOPEN = [\n')

        for prob in sampled_shapes:
            f.write(f'    GroupedConvProblem(')
            f.write(f'N={prob["N"]}, ')
            f.write(f'C={prob["C"]}, ')
            f.write(f'K={prob["K"]}, ')
            f.write(f'G={prob["G"]}, ')
            f.write(f'Hi={prob["Hi"]}, ')
            f.write(f'Wi={prob["Wi"]}, ')
            f.write(f'Y={prob["Y"]}, ')
            f.write(f'X={prob["X"]}, ')
            f.write(f'stride_h={prob["stride_h"]}, ')
            f.write(f'stride_w={prob["stride_w"]}, ')
            f.write(f'pad_h={prob["pad_h"]}, ')
            f.write(f'pad_w={prob["pad_w"]}, ')
            f.write(f'direction="forward"),\n')

        f.write(']\n\n')
        f.write(f'assert len(TRAINING_PROBLEMS_FORWARD_MIOPEN) == {len(sampled_shapes)}, ')
        f.write(f'"Expected {len(sampled_shapes)} problems, got {{len(TRAINING_PROBLEMS_FORWARD_MIOPEN)}}"\n')

    print(f"\n✓ Generated {output_file}")
    print(f"  Total problems: {len(sampled_shapes)}")

if __name__ == '__main__':
    random.seed(42)  # For reproducibility
    main()

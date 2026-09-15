#!/usr/bin/env python3
"""
Tone Mapping: RPP vs OpenCV Performance Comparison

Usage:
    python tone_map_compare.py [--image path.jpg] [--runs 100] [--gamma 2.2]

Requires: opencv-python (pip install opencv-python)
"""

import argparse
import time
import subprocess
import os
import sys
import numpy as np

try:
    import cv2
except ImportError:
    print("Error: opencv-python not installed. Run: pip install opencv-python")
    sys.exit(1)


def reinhard_reference(img_f32, gamma=2.2):
    """Pure NumPy Reinhard tone mapping (scalar reference)."""
    inv_gamma = 1.0 / gamma if gamma > 0 else 1.0
    R, G, B = img_f32[:, :, 0], img_f32[:, :, 1], img_f32[:, :, 2]
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    scale = 1.0 / (1.0 + L)
    out = img_f32 * scale[:, :, np.newaxis]
    if inv_gamma != 1.0:
        out = np.power(np.clip(out, 0, None), inv_gamma)
    return np.clip(out, 0, 1).astype(np.float32)


def benchmark_opencv(img_bgr, gamma, num_runs):
    """Benchmark OpenCV TonemapReinhard."""
    img_f32 = img_bgr.astype(np.float32) / 255.0
    tonemap = cv2.createTonemapReinhard(gamma=gamma)

    # Warmup
    tonemap.process(img_f32)

    start = time.perf_counter()
    for _ in range(num_runs):
        result = tonemap.process(img_f32)
    elapsed = (time.perf_counter() - start) / num_runs * 1000  # ms
    return elapsed, result


def benchmark_numpy_reference(img_bgr, gamma, num_runs):
    """Benchmark NumPy Reinhard reference."""
    img_f32 = img_bgr.astype(np.float32) / 255.0
    # Convert BGR to RGB for correct luminance
    img_rgb = img_f32[:, :, ::-1].copy()

    # Warmup
    reinhard_reference(img_rgb, gamma)

    start = time.perf_counter()
    for _ in range(num_runs):
        result = reinhard_reference(img_rgb, gamma)
    elapsed = (time.perf_counter() - start) / num_runs * 1000  # ms
    return elapsed, result


def benchmark_rpp(test_binary, test_images_dir, script_dir, gamma, num_runs):
    """Benchmark RPP rppt_tone_map via the test binary."""
    output_dir = "/tmp/rpp_tone_map_output"
    os.makedirs(output_dir, exist_ok=True)

    src_dir = os.path.join(test_images_dir, "TEST_IMAGES", "three_images_mixed_src1")

    # Run as performance test (testType=1) to get timing
    # Args: src1 src2 dst bitDepth toggle case additionalParam runs testType layout verbosity qaFlag decoder batchSize roiX roiY roiW roiH scriptPath
    cmd = [
        test_binary,
        src_dir,
        src_dir,
        output_dir,
        "0",  # bitDepth = U8
        "0",  # outputFormatToggle
        "106",  # testCase = TONE_MAP
        "1",  # additionalParam
        str(num_runs),  # numRuns
        "1",  # testType = performance
        "0",  # layoutType = PKD3
        "0",  # verbosity
        "0",  # qaFlag
        "0",  # decoderType
        "1",  # batchSize
        "0",
        "0",
        "0",
        "0",  # ROI
        script_dir,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr

        # Parse wall time from output
        # RPP prints: "max,min,avg wall times in ms/batch = 1.12,0.04,0.18"
        import re

        for line in output.split("\n"):
            if "wall time" in line.lower():
                # Format: max,min,avg wall times in ms/batch = X,Y,Z
                match = re.search(r"=\s*([\d.]+),([\d.]+),([\d.]+)", line)
                if match:
                    avg_ms = float(match.group(3))  # avg is the third value
                    return avg_ms, output

        # If we can't parse, return the raw output for debugging
        return None, output
    except FileNotFoundError:
        return None, f"Error: Binary not found: {test_binary}"
    except subprocess.TimeoutExpired:
        return None, "Error: RPP test timed out"


def main():
    parser = argparse.ArgumentParser(description="RPP vs OpenCV Tone Mapping Benchmark")
    parser.add_argument(
        "--image",
        default=None,
        help="Path to input image (default: generate synthetic)",
    )
    parser.add_argument(
        "--runs", type=int, default=100, help="Number of iterations (default: 100)"
    )
    parser.add_argument(
        "--gamma", type=float, default=2.2, help="Gamma value (default: 2.2)"
    )
    parser.add_argument(
        "--save", action="store_true", help="Save output images for visual comparison"
    )
    args = parser.parse_args()

    # Locate RPP test binary and directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_suite_dir = os.path.dirname(script_dir)
    rpp_dir = os.path.dirname(os.path.dirname(test_suite_dir))
    build_dir = os.path.join(rpp_dir, "build")
    test_binary = os.path.join(build_dir, "bin", "Tensor_image_host")

    # Load or generate image
    if args.image and os.path.exists(args.image):
        img_bgr = cv2.imread(args.image)
        if img_bgr is None:
            print(f"Error: Cannot load image: {args.image}")
            sys.exit(1)
        img_name = os.path.basename(args.image)
    else:
        # Generate synthetic 1920x1080 gradient
        h, w = 1080, 1920
        img_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        img_bgr[:, :, 0] = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
        img_bgr[:, :, 1] = np.tile(
            np.linspace(0, 255, h, dtype=np.uint8).reshape(-1, 1), (1, w)
        )
        img_bgr[:, :, 2] = 128
        img_name = "synthetic_1920x1080"
        if args.image:
            print(f"Warning: Image not found at {args.image}, using synthetic image")

    h, w = img_bgr.shape[:2]

    print("=" * 60)
    print("  Tone Mapping Benchmark: RPP vs OpenCV vs NumPy")
    print("=" * 60)
    print(f"  Image:  {img_name} ({w}x{h}, 3ch U8)")
    print(f"  Gamma:  {args.gamma}")
    print(f"  Runs:   {args.runs}")
    print("=" * 60)
    print()

    # --- OpenCV benchmark ---
    print("Running OpenCV TonemapReinhard...", end=" ", flush=True)
    ocv_ms, ocv_result = benchmark_opencv(img_bgr, args.gamma, args.runs)
    print(f"{ocv_ms:.3f} ms")

    # --- NumPy reference benchmark ---
    print("Running NumPy reference...", end=" ", flush=True)
    np_ms, np_result = benchmark_numpy_reference(img_bgr, args.gamma, args.runs)
    print(f"{np_ms:.3f} ms")

    # --- RPP benchmark ---
    rpp_ms = None
    if os.path.exists(test_binary):
        print("Running RPP rppt_tone_map...", end=" ", flush=True)
        rpp_ms, rpp_output = benchmark_rpp(
            test_binary, test_suite_dir, test_suite_dir, args.gamma, args.runs
        )
        if rpp_ms is not None:
            print(f"{rpp_ms:.3f} ms")
        else:
            print("(could not parse timing)")
            print(f"  RPP output:\n{rpp_output[:500]}")
    else:
        print(f"RPP binary not found at: {test_binary}")
        print("  Build it with: cd rpp/build && make -j$(nproc) Tensor_image_host")

    # --- Results ---
    print()
    print(f"{'Method':<30} {'Avg (ms)':>10} {'vs OpenCV':>12}")
    print("-" * 54)
    print(f"{'OpenCV TonemapReinhard':<30} {ocv_ms:>10.3f} {'1.00x':>12}")
    print(f"{'NumPy Reinhard (reference)':<30} {np_ms:>10.3f} {ocv_ms/np_ms:>11.2f}x")
    if rpp_ms is not None:
        print(f"{'RPP rppt_tone_map':<30} {rpp_ms:>10.3f} {ocv_ms/rpp_ms:>11.2f}x")
    print()

    # Note about algorithmic differences
    print("Note: OpenCV's TonemapReinhard uses log-average luminance adaptation")
    print("      (global key estimation), while RPP uses the simple per-pixel")
    print("      Reinhard operator L/(1+L). Output values will differ.")

    # --- Save outputs ---
    if args.save:
        ocv_u8 = np.clip(ocv_result * 255, 0, 255).astype(np.uint8)
        np_u8 = np.clip(np_result * 255, 0, 255).astype(np.uint8)
        # NumPy result is RGB, convert to BGR for saving
        np_bgr = np_u8[:, :, ::-1]

        cv2.imwrite("tonemap_opencv.png", ocv_u8)
        cv2.imwrite("tonemap_numpy_ref.png", np_bgr)
        cv2.imwrite("tonemap_input.png", img_bgr)
        print("\nSaved: tonemap_input.png, tonemap_opencv.png, tonemap_numpy_ref.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MIT License

Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Generate synthetic test images for OpenCV benchmarking.
Creates 128 images at 1080p resolution with various patterns and colors.
"""

import os
from PIL import Image, ImageDraw
import random
import math
import numpy as np

# Configuration
OUTPUT_DIR = "1080p_128images_dataset"
WIDTH = 1920
HEIGHT = 1080
NUM_IMAGES = 128


def create_output_dir():
    """Create output directory if it doesn't exist."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(output_path, exist_ok=True)
    return output_path


def generate_gradient_image(index, output_path):
    """Generate a gradient image."""
    img = Image.new("RGB", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)

    # Random gradient direction
    for y in range(HEIGHT):
        r = int((y / HEIGHT) * 255)
        g = int(((HEIGHT - y) / HEIGHT) * 255)
        b = int((index * 2) % 255)
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))

    filename = os.path.join(output_path, f"gradient_{index:03d}.jpg")
    img.save(filename, "JPEG", quality=95)
    return filename


def generate_checkerboard_image(index, output_path):
    """Generate a checkerboard pattern."""
    img = Image.new("RGB", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)

    tile_size = 40 + (index * 2) % 60
    color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    for y in range(0, HEIGHT, tile_size):
        for x in range(0, WIDTH, tile_size):
            if ((x // tile_size) + (y // tile_size)) % 2 == 0:
                draw.rectangle([x, y, x + tile_size, y + tile_size], fill=color1)
            else:
                draw.rectangle([x, y, x + tile_size, y + tile_size], fill=color2)

    filename = os.path.join(output_path, f"checkerboard_{index:03d}.png")
    img.save(filename, "PNG")
    return filename


def generate_circle_image(index, output_path):
    """Generate random circles."""
    bg_color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
    img = Image.new("RGB", (WIDTH, HEIGHT), color=bg_color)
    draw = ImageDraw.Draw(img)

    num_circles = 10 + (index % 20)
    for _ in range(num_circles):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        radius = random.randint(20, 200)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius], fill=color, outline=color
        )

    filename = os.path.join(output_path, f"circles_{index:03d}.jpg")
    img.save(filename, "JPEG", quality=95)
    return filename


def generate_noise_image(index, output_path):
    """Generate random noise."""
    # Vectorized numpy array creation is 100-1000x faster than pixel loops
    noise_array = np.random.randint(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    img = Image.fromarray(noise_array, "RGB")

    filename = os.path.join(output_path, f"noise_{index:03d}.png")
    img.save(filename, "PNG")
    return filename


def generate_stripes_image(index, output_path):
    """Generate striped pattern."""
    img = Image.new("RGB", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)

    stripe_width = 10 + (index * 3) % 50
    horizontal = index % 2 == 0

    color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    if horizontal:
        for y in range(0, HEIGHT, stripe_width * 2):
            draw.rectangle([0, y, WIDTH, y + stripe_width], fill=color1)
    else:
        for x in range(0, WIDTH, stripe_width * 2):
            draw.rectangle([x, 0, x + stripe_width, HEIGHT], fill=color2)

    filename = os.path.join(output_path, f"stripes_{index:03d}.jpg")
    img.save(filename, "JPEG", quality=95)
    return filename


def generate_solid_color_image(index, output_path):
    """Generate solid color image."""
    r = (index * 17) % 256
    g = (index * 31) % 256
    b = (index * 47) % 256

    img = Image.new("RGB", (WIDTH, HEIGHT), color=(r, g, b))

    filename = os.path.join(output_path, f"solid_{index:03d}.jpg")
    img.save(filename, "JPEG", quality=95)
    return filename


def generate_radial_pattern_image(index, output_path):
    """Generate radial pattern from center."""
    cx, cy = WIDTH // 2, HEIGHT // 2
    max_dist = math.sqrt(cx**2 + cy**2)

    # Vectorized computation using numpy meshgrid
    x = np.arange(WIDTH)
    y = np.arange(HEIGHT)
    X, Y = np.meshgrid(x, y)

    # Calculate distance from center for all pixels at once
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    intensity = (dist / max_dist * 255).astype(np.uint8)

    # Calculate RGB channels
    r = (intensity + index * 10) % 256
    g = (255 - intensity) % 256
    b = (intensity * 2) % 256

    # Stack channels to create RGB image
    img_array = np.stack([r, g, b], axis=-1).astype(np.uint8)
    img = Image.fromarray(img_array, "RGB")

    filename = os.path.join(output_path, f"radial_{index:03d}.png")
    img.save(filename, "PNG")
    return filename


def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("OpenCV Benchmark Test Dataset Generator")
    print("=" * 60)
    print(f"\nGenerating {NUM_IMAGES} images at {WIDTH}x{HEIGHT} resolution...")
    print(f"Output directory: {OUTPUT_DIR}/\n")

    output_path = create_output_dir()

    # Distribution of image types
    generators = [
        generate_gradient_image,
        generate_checkerboard_image,
        generate_circle_image,
        generate_noise_image,
        generate_stripes_image,
        generate_solid_color_image,
        generate_radial_pattern_image,
    ]

    for i in range(NUM_IMAGES):
        # Cycle through different generators
        generator = generators[i % len(generators)]
        filename = generator(i, output_path)

        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{NUM_IMAGES} images...")

    print(f"\n✓ Successfully generated {NUM_IMAGES} images!")
    print(f"✓ Location: {output_path}")
    print(f"\nDataset is ready for benchmarking!")
    print("\nTo run the benchmark:")
    print("  ./run_benchmarking.sh")


if __name__ == "__main__":
    main()

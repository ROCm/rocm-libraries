import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Summarize noise from CSV files")
    parser.add_argument(
        "csv_dir",
        type=Path,
        help="Path to directory containing CSV files",
    )
    args = parser.parse_args()

    csv_dir = args.csv_dir

    if not csv_dir.is_dir():
        raise ValueError(f"Not a directory: {csv_dir}")

    results = []

    for filepath in csv_dir.iterdir():
        if filepath.suffix != ".csv":
            continue

        noise = 0

        with filepath.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row["noise_timeout"]):
                    noise = max(noise, float(row["noise_percent"]))

        results.append({"noise": noise, "name": filepath.name})

    results.sort(key=lambda x: x["noise"], reverse=True)

    for result in results:
        if result["noise"] == 0:
            continue
        print(f"{result['noise']:.1f}%: {result['name']}")

    print("")


if __name__ == "__main__":
    main()
import sys
import json
import subprocess


def run_gtest(gtest_executable: str, gtest_filter_json: str):
    with open(gtest_filter_json, "r") as f:
        json_data = json.load(f)
    gtest_filter = json_data["gtest_filter"]
    print(f"Running {gtest_executable} with filter: {gtest_filter}", flush=True)
    subprocess.run([gtest_executable, f"--gtest_filter={gtest_filter}"], check=True)


def main():
    gtest_executable = sys.argv[1]
    gtest_filter_json = sys.argv[2]
    run_gtest(gtest_executable, gtest_filter_json)


if __name__ == "__main__":
    main()

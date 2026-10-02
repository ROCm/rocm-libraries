import argparse
from collections import namedtuple
from datetime import datetime
import json
import os
import re
import stat
import subprocess
from urllib.parse import urljoin
import urllib.request


BenchmarkContext = namedtuple('BenchmarkContext', ['run_datetime', 'version', 'gpu_name', 'benchmark_dir', 'benchmark_filename_regex', 'benchmark_filter_regex'])
ApiContext = namedtuple('ApiContext', ['endpoint', 'folder_id', 'auth_token'])


def run_benchmarks(benchmark_context):
    def is_benchmark_executable(filename):
        if not re.match(benchmark_context.benchmark_filename_regex, filename):
            return False
        path = os.path.join(benchmark_context.benchmark_dir, filename)
        st_mode = os.stat(path).st_mode

        # we are not interested in permissions, just whether there is any execution flag set
        # and it is a regular file (S_IFREG)
        return (st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)) and (st_mode & stat.S_IFREG)

    success = True
    benchmark_names = [name for name in os.listdir(benchmark_context.benchmark_dir) if is_benchmark_executable(name)]
    json_paths = []
    for benchmark_name in benchmark_names:
        results_json_name = f'{benchmark_name}_{benchmark_context.version}_{benchmark_context.gpu_name}.json'

        benchmark_path = os.path.join(benchmark_context.benchmark_dir, benchmark_name)
        results_json_path = os.path.join(benchmark_context.benchmark_dir, results_json_name)
        args = [
            benchmark_path,
            '--benchmark_out_format=json',
            f'--benchmark_out={results_json_path}',
            f'--benchmark_filter={benchmark_context.benchmark_filter_regex}'
        ]
        try:
            subprocess.call(args)
            json_paths.append(results_json_path)
        except OSError as error:
            print(f'Could not run benchmark at {benchmark_path}. Error: "{error}"')
            success = False
    return success, json_paths


def write_system_info():
    def try_running_info(executable_name):
        out_filename = f'{executable_name}.txt'
        try:
            run_result = subprocess.run(executable_name, stdout=subprocess.PIPE)
            if run_result.returncode == 0:
                with open(out_filename, 'wb') as file:
                    file.write(run_result.stdout)
                return out_filename
        except OSError:
            # Expected, when the executable is not available on the system
            pass


    rocminfo_filename = try_running_info('rocminfo')
    if rocminfo_filename:
        return rocminfo_filename
    else:
        return try_running_info('deviceQuery')


def create_benchmark_folder(benchmark_context, api_context):
    formatted_datetime = datetime.strftime(benchmark_context.run_datetime, '%Y%m%d_%H%M%S')
    new_folder_name = f'{formatted_datetime}_{benchmark_context.version}_{benchmark_context.gpu_name}'
    create_folder_url = urljoin(api_context.endpoint, f'files/folder/{api_context.folder_id}')
    create_folder_payload = json.dumps({ 'title': new_folder_name }).encode('utf-8')
    create_folder_headers = { 'Content-Type': 'application/json', 'Authorization': api_context.auth_token }

    create_folder_request = urllib.request.Request(
        url=create_folder_url,
        data=create_folder_payload,
        headers=create_folder_headers,
        method='POST')
    try:
        with urllib.request.urlopen(create_folder_request) as response:
            response_data = json.loads(response.read())
            new_folder_id = response_data['response']['id']
            print(f"Created new folder with id {new_folder_id}")
            return new_folder_id
    except Exception as ex:
        print(f'Could not create folder "{new_folder_name}". Error: {ex}')
        return None


def upload_results(folder_id, api_context, paths_to_upload):
    success = True
    upload_file_url = urljoin(api_context.endpoint, f'files/{folder_id}/upload')
    for path in paths_to_upload:
        with open(path) as file:
            body_bytes = file.read().encode('utf-8')
        filename = os.path.basename(path)
        upload_file_headers = {
            'Content-Type': 'text/plain',
            'Content-Disposition': f'attachment; filename="{filename}"',
            'Authorization': api_context.auth_token
        }
        upload_file_request = urllib.request.Request(url=upload_file_url, data=body_bytes, headers=upload_file_headers, method='POST')
        try:
            with urllib.request.urlopen(upload_file_request):
                pass
            print(f'Uploaded {path}')
        except Exception as ex:
            print(f'Could not upload file "{path}". Error: {ex}')
            success = False
    return success


def parse_date(date_str):
    """
    Parses the date format provided by GitLab's builtin variable CI_PIPELINE_CREATED_AT
    """
    return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_endpoint',
        help='URL that specifies the file storage API endpoint. For example: https://website.com/api/2.0/',
        required=True)
    parser.add_argument('--api_base_folder_id',
        help='The ID of the remote folder to which the benchmark results are uploaded',
        required=True)
    parser.add_argument('--api_auth_token',
        help='The authentication token string required by the remote API',
        required=True)
    parser.add_argument('--benchmark_dir',
        help='The local directory that contains the benchmark executables',
        required=True)
    parser.add_argument('--benchmark_datetime',
        help='The datetime string that specifies the creation date of the benchmarks. For example: "2022-03-28T13:16:09Z"',
        required=True)
    parser.add_argument('--benchmark_version',
        help='The identifier of the source control version of the benchmarked source code. For example a commit hash.',
        required=True)
    parser.add_argument('--benchmark_gpu_name',
        help='The name of the currently enabled GPU',
        required=True)
    parser.add_argument('--benchmark_filename_regex',
        help='Regular expression that controls the list of benchmark executables to run',
        default=r'^benchmark',
        required=False)
    parser.add_argument('--benchmark_filter_regex',
        help='Regular expression that controls the list of benchmarks to run in each benchmark executable',
        default='',
        required=False)
    parser.add_argument('--no_upload',
                        help='Only run the benchmarks, do not upload them',
                        default=False,
                        action='store_true',
                        required=False)

    args = parser.parse_args()

    api_context = ApiContext(args.api_endpoint, args.api_base_folder_id, args.api_auth_token)
    benchmark_context = BenchmarkContext(
        parse_date(args.benchmark_datetime),
        args.benchmark_version,
        args.benchmark_gpu_name,
        args.benchmark_dir,
        args.benchmark_filename_regex,
        args.benchmark_filter_regex)

    status = True

    benchmark_run_successful, to_upload_paths = run_benchmarks(benchmark_context)
    status = status and benchmark_run_successful
    sysinfo_path = write_system_info()
    if sysinfo_path:
        # not required to be successful.
        # Not all rocm/nvidia images have rocminfo/deviceQuery in their path
        to_upload_paths.append(sysinfo_path)

    if not args.no_upload:
        upload_successful = False
        folder_id = create_benchmark_folder(benchmark_context, api_context)
        if folder_id is not None:
            upload_successful = upload_results(folder_id, api_context, to_upload_paths)
        status = status and upload_successful

    return status


if __name__ == '__main__':
    success = main()
    if success:
        exit(0)
    else:
        exit(1)

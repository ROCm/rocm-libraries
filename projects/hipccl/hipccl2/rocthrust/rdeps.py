#!/usr/bin/python3
# Copyright 2021 Advanced Micro Devices, Inc.

"""Manage dependency installation"""

import os
import sys
import subprocess
import argparse
import pathlib
import shutil
import platform

def install_conan(**kwargs):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'conan'])

def configure_conan(profile, **kwargs):
    subprocess.call(['conan', 'profile', 'new', profile, '--detect'])
    if platform.system() == 'Linux':
        subprocess.check_call(['conan', 'profile', 'update',
            'settings.compiler.libcxx=libstdc++11', profile])

def conan_install_deps(install_dir, profile='default', **kwargs):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    subprocess.check_call(['conan', 'install', source_dir, '-o', 'clients=True',
        '--install-folder={}'.format(install_dir), '--build=missing', '--profile', profile])

def parse_args():
    parser = argparse.ArgumentParser(description='Manage dependency installation')
    parser.add_argument(      '--install_dir', type=str, required=False, default="build\\deps",
                        help='The install directory path (optional, default: build\\deps)')
    # parser.add_argument('-v', '--verbose', required=False, default=False, action='store_true',
    #                     help='Verbose install')
    return parser.parse_args()

def os_detect():
    info = {}
    try:
        with open('/etc/os-release') as f:
            for line in f:
                if '=' in line:
                    k,v = line.strip().split('=', maxsplit=1)
                    info[k] = v.replace('"', '')
    except OSError as e:
        info["ID"] = platform.system()
    info["NUM_PROC"] = os.cpu_count()
    return info

def install_deps(**kwargs):
    install_conan(**kwargs)
    configure_conan(**kwargs)
    conan_install_deps(**kwargs)

def main():
    args = parse_args()
    print(os_detect())
    install_deps(install_dir=args.install_dir, profile='rocm')

if __name__ == '__main__':
    main()

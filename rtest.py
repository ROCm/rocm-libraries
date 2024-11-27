#!/usr/bin/python3
"""Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
Run tests on build"""

import os
import sys
import subprocess
import argparse
from xml.dom import minidom
import platform as pf

#TODO Implement time outs when its added to the xml
#TODO Implement VRAM limit when its added to the xml

class TestRunner():
    def __get_vram__(self):
        if self.OS_info['System'] == 'Linux':
            process = subprocess.run('rocm-smi --showmeminfo vram', shell=True, stdout=subprocess.PIPE)
            gpu_id = os.getenv('HIP_VISIBLE_DEVICES', '0')

            for l in process.stdout.decode().splitlines():
                if 'Total Memory' in l and f'GPU[{gpu_id}]' in l:
                    self.OS_info['VRAM'] = float(l.split()[-1]) / (1024 ** 3)
                    break

    def __parse_args__(self):
        self.parser.add_argument('-e', '--emulation', required=False, default='',
                            help='Test set to run from rtest.xml (optional, eg.smoke). At least one but not both of -e or -t must be set')
        self.parser.add_argument('-t', '--test', required=False, default='', 
                            help='Test set to run from rtest.xml (optional, e.g. osdb). At least one but not both of -e or -t must be set')
        self.parser.add_argument('-g', '--debug', required=False, default=False,  action='store_true',
                            help='Test Debug build (optional, default: false)')
        self.parser.add_argument('-o', '--output', type=str, required=False, default=None, 
                            help='Test output file (optional, default: None [output to stdout])')
        self.parser.add_argument(      '--install_dir', type=str, required=False, default="build", 
                            help='Installation directory where build or release folders are (optional, default: build)')
        self.parser.add_argument(      '--test_dir', type=str, required=False, default=None,
                            help='Test directory where rocprim tests are (optinal, default=None)')
        self.parser.add_argument(      '--fail_test', default=False, required=False, action='store_true',
                            help='Return as if test failed (optional, default: false)')
        self.args = self.parser.parse_args()

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="""
        Checks build arguments
        """)

        self.__parse_args__()
        
        if (self.args.emulation != '') ^ (self.args.test != ''):
            if self.args.emulation != '':
                self.test_choice = self.args.emulation
            else:
                self.test_choice = self.args.test
        else:
            raise ValueError('At least one but not both of -e/--emulation or -t/--test must be set')
        
        sysInfo = pf.uname()

        # getting os information
        self.OS_info = {
            "Machine" : sysInfo.machine,
            "Node Name" : sysInfo.node,
            "Num Processor" : os.cpu_count(),
            "Processor" : sysInfo.processor,
            "Release" : sysInfo.release,
            "System" : sysInfo.system,
            "Version" : sysInfo.version,
        }

        self.__get_vram__()

        m = ' System Information '

        print()
        print(f'{m:-^100}')
        
        for k in self.OS_info:
            print(f'\t {k}: {self.OS_info[k]}')
        print()

        self.lib_dir = os.path.dirname(os.path.abspath(__file__)) 
        self.xml_path = os.path.join(self.lib_dir, r'rtest.xml')

        if self.args.test_dir:
            if not os.path.isdir(self.args.test_dir):
                raise ValueError(f'Value {self.args.test_dir} is not a directory!')

            self.test_dir = self.args.test_dir
        else:
            # find the test dir with default install dir
            if self.args.install_dir == 'build':
                # if its debug mode
                if self.args.debug: 
                    self.test_dir = os.path.join(self.lib_dir, f'build/debug/test')
                # if its release mode
                elif os.path.isdir(os.path.join(self.lib_dir, f'build/release/test')): 
                    self.test_dir = os.path.join(self.lib_dir, f'build/release/test')
                elif os.path.isdir(os.path.join(self.lib_dir, f'build/test')):
                    self.test_dir = os.path.join(self.lib_dir, f'build/test')
                else:
                    raise FileNotFoundError(f'Did not find test directory')
            else:
                # if its an actual directory AND it has test directory
                if os.path.isdir(os.path.join(self.args.install_dir, f'test')):
                    self.test_dir = os.path.join(self.args.install_dir, f'test')
                else:
                    raise ValueError(f'{self.args.install_dir} is not a valid install directory!')
                
        if self.args.output:
            self.output = open(os.path.abspath(self.args.output), 'w')
            self.output_path = os.path.abspath(self.args.output)
        else:
            self.output = None
            self.output_path = None
        
        if self.OS_info['System'] == 'Windows':
            self.lib_dir = self.lib_dir.replace('\\', '/')
            self.xml_path = self.xml_path.replace('\\', '/')
            self.test_dir = self.test_dir.replace('\\', '/')

            if self.output_path:
                self.output_path = self.output_path.replace('\\', '/')

                self.output = open(os.path.abspath(self.output_path), 'w')

        m = ' Current Paths'
        print(f'{m:-^100}')
        print(f'Working Directory: {self.lib_dir}')
        print(f'rtest.xml:         {self.xml_path}')
        print(f'Test Directory:    {self.test_dir}')
        print(f'Output File:       {self.output_path}')
        
        print()

    def __call__(self):
        xml_file = minidom.parse(self.xml_path)

        curr_dir = os.curdir

        os.chdir(self.test_dir)

        cmd_values = {}
        for var in xml_file.getElementsByTagName('var'):
            name, val = var.getAttribute('name'), var.getAttribute('value')
            cmd_values[name] = val

        noMatch = True
        for test in xml_file.getElementsByTagName('test'):
            sets = test.getAttribute('sets')
            if self.test_choice == sets:

                for run in test.getElementsByTagName('run'):
                    temp = run.firstChild.data
                    temp = temp.replace('{', '')
                    temp = temp.replace('}', '')
                    cmd_list = temp.split()

                    cmd_str = ''
                    for var in cmd_list:
                        cmd_str += cmd_values[var]

                m = 'Final Command'
                print(f'{m:-^100}')
                print(cmd_str)
                print()

                subprocess.run(cmd_str, shell=True, stdout=self.output)
                noMatch = False
                break

        os.chdir(curr_dir)
        if noMatch:
            raise ValueError(f'Test value passed in: "{self.test_choice}" does not match any known test suite')  

        if self.args.fail_test:
            sys.exit(1)              

if __name__ == '__main__':
    runner = TestRunner()
    runner()
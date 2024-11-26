#!/usr/bin/python3
"""Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
Run tests on build"""

from multiprocessing import process
import re
import os
import sys
import subprocess
import shlex
import argparse
import pathlib
import platform
from genericpath import exists
from fnmatch import fnmatchcase
from xml.dom import minidom
import multiprocessing
import time
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

        if self.OS_info['System'] == 'Linux':
            
            # find the test dir with default install dir
            if self.args.install_dir == 'build':
                
                # if its debug mode
                if self.args.debug: 
                    self.test_dir = os.path.join(self.lib_dir, f'build/debug/test')
                # if its release mode
                elif os.path.isdir(os.path.join(self.lib_dir, f'build/release/test')): 
                    self.test_dir = os.path.join(self.lib_dir, f'build/release/test')
                else:
                    self.test_dir = os.path.join(self.lib_dir, f'build/test')


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



# def batch(script, xml):
#     global OS_info
#     global args
#     # 
#     fail = False
#     for i in range(len(script)):
#         cmdline = script[i]
#         xcmd = cmdline.replace('%IDIR%', test_dir)
#         cmd = xcmd.replace('%ODIR%', args.output)
#         if cmd.startswith('tdir '):
#             if pathlib.Path(cmd[5:]).exists():
#                 return 0 # all further cmds skipped
#             else:
#                 continue
#         error = False
#         if cmd.startswith('%XML%'):
#             # run the matching tests listed in the xml test file
#             var_subs = {}
#             for var in xml.getElementsByTagName('var'):
#                 name = var.getAttribute('name')
#                 val = var.getAttribute('value')
#                 var_subs[name] = val
#             for test in xml.getElementsByTagName('test'):
#                 sets = test.getAttribute('sets')
#                 runset = sets.split(',')

#                 A, B = args.test != '', args.emulation != ''
#                 if not (A ^ B):
#                     raise ValueError('At least one but not both of -e/--emulation or -t/--test must be set')

#                 if args.test in runset:
#                     for run in test.getElementsByTagName('run'):
#                         name = run.getAttribute('name')
#                         vram_limit = run.getAttribute('vram_min')
#                         if vram_limit:
#                             if OS_info["VRAM"] < float(vram_limit):
#                                 print( f'***\n*** Skipped: {name} due to VRAM req.\n***')
#                                 continue
#                         if name:
#                             print( f'***\n*** Running: {name}\n***')
#                         time_limit = run.getAttribute('time_max')
#                         if time_limit:
#                             timeout = float(time_limit)
#                         else:
#                             timeout = 0

#                         raw_cmd = run.firstChild.data
#                         var_cmd = raw_cmd.format_map(var_subs)
#                         error = run_cmd(var_cmd, True, timeout)
#                         if (error == 2):
#                             print( f'***\n*** Timed out when running: {name}\n***')
                
#                 if args.emulation in runset:
#                     for run in test.getElementsByTagName('run'):
#                         name = run.getAttribute('name')
#                         vram_limit = run.getAttribute('vram_min')
#                         if vram_limit:
#                             if OS_info["VRAM"] < float(vram_limit):
#                                 print( f'***\n*** Skipped: {name} due to VRAM req.\n***')
#                                 continue
#                         if name:
#                             print( f'***\n*** Running: {name}\n***')
#                         time_limit = run.getAttribute('time_max')
#                         if time_limit:
#                             timeout = float(time_limit)
#                         else:
#                             timeout = 0

#                         raw_cmd = run.firstChild.data
#                         var_cmd = raw_cmd.format_map(var_subs)
#                         error = run_cmd(var_cmd, True, timeout)
#                         if (error == 2):
#                             print( f'***\n*** Timed out when running: {name}\n***')
#         else:
#             error = run_cmd(cmd)
#         fail = fail or error
#     if (fail):
#         if (cmd == "%XML%"):
#             print(f"FAILED xml test suite!")
#         else:
#             print(f"ERROR running: {cmd}")
#         if (os.curdir != cwd):
#             os.chdir( cwd )
#         return 1
#     if (os.curdir != cwd):
#         os.chdir( cwd )
    
#     return 0



# timeout = False
# test_proc = None
# stop = 0

# test_script = [ 'cd %IDIR%', '%XML%' ]

# def vram_detect():
#     global OS_info
#     OS_info["VRAM"] = 0
#     if os.name == "nt":
#         cmd = "hipinfo.exe"
#         process = subprocess.run([cmd], stdout=subprocess.PIPE)
#         for line_in in process.stdout.decode().splitlines():
#             if 'totalGlobalMem' in line_in:
#                 OS_info["VRAM"] = float(line_in.split()[1])
#                 break

# class TimerProcess(multiprocessing.Process):

#     def __init__(self, start, stop, kill_pid):
#         multiprocessing.Process.__init__(self)
#         self.quit = multiprocessing.Event()
#         self.timed_out = multiprocessing.Event()
#         self.start_time = start
#         self.max_time = stop
#         self.kill_pid = kill_pid

#     def run(self):
#         while not self.quit.is_set():
#             #print( f'time_stop {self.start_time} limit {self.max_time}')
#             if (self.max_time == 0):
#                 return
#             t = time.monotonic()
#             if ( t - self.start_time > self.max_time ):
#                 print( f'killing {self.kill_pid} t {t}')
#                 if os.name == "nt":
#                     cmd = ['TASKKILL', '/F', '/T', '/PID', str(self.kill_pid)]
#                     proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
#                 else:
#                     os.kill(self.kill_pid, signal.SIGKILL)  
#                 self.timed_out.set()
#                 self.stop()
#             pass

#     def stop(self):
#         self.quit.set()
    
#     def stopped(self):
#         return self.timed_out.is_set()


# def run_cmd(cmd, test = False, time_limit = 0):
#     global args
#     global test_proc, timer_thread
#     global stop
#     if (cmd.startswith('cd ')):
#         return os.chdir(cmd[3:])
#     if (cmd.startswith('mkdir ')):
#         return create_dir(cmd[6:])
#     cmdline = f"{cmd}"
#     print(cmdline)
#     try:
#         if not test:
#             proc = subprocess.run(cmdline, check=True, stderr=subprocess.STDOUT, shell=True)
#             status = proc.returncode
#         else:
#             error = False
#             timeout = False
#             test_proc = subprocess.Popen(cmdline, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
#             if time_limit > 0:
#                 start = time.monotonic()
#                 #p = multiprocessing.Process(target=time_stop, args=(start, test_proc.pid))
#                 p = TimerProcess(start, time_limit, test_proc.pid)
#                 p.start()
#             while True:
#                 output = test_proc.stdout.readline()
#                 if output == '' and test_proc.poll() is not None:
#                     break
#                 elif output:
#                     outstring = output.strip()
#                     print (outstring)
#                     error = error or re.search(r'FAILED', outstring)
#             status = test_proc.poll()
#             if time_limit > 0:
#                 p.stop()
#                 p.join()
#                 timeout = p.stopped()
#                 print(f"timeout {timeout}")
#             if error: 
#                 status = 1
#             elif timeout:
#                 status = 2
#             else:
#                 status = test_proc.returncode    
#     except:
#         import traceback
#         exc = traceback.format_exc()
#         print( "Python Exception: {0}".format(exc) )
#         status = 3
#     return status


# def run_tests():
#     global test_script
#     global xmlDoc

#     # install
#     cwd = os.curdir

#     xmlPath = os.path.join( cwd, 'rtest.xml')
#     xmlDoc = minidom.parse( xmlPath )

#     scripts = []
#     scripts.append( test_script )
#     for i in scripts:
#         if (batch(i, xmlDoc)):
#             #print("Failure in script. ABORTING")
#             if (os.curdir != cwd):
#                 os.chdir( cwd )
#             return 1       
#     if (os.curdir != cwd):
#         os.chdir( cwd )
#     return 0

# def main():
#     global args
#     global timer_thread

#     os_detect()
#     args = parse_args()

#     status = run_tests()

#     if args.fail_test: 
#         status = 1
#     if (status):
#         sys.exit(status)

if __name__ == '__main__':
    runner = TestRunner()
    runner()
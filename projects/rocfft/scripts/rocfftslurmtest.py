#!/usr/bin/env python3

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import pathlib
import datetime
import sys
import argparse
import configparser

from pathlib import Path

top = Path(__file__).resolve().parent
sys.path.append(str(top))
import rocslurm


def main():

    conf_parser = argparse.ArgumentParser(
        prog='rocfftslurm',
        epilog="For a detailed usage overview, run: %(prog)s overview",
        add_help=False)

    conf_parser.add_argument("--config_file", metavar="FILE")

    args, remaining_args = conf_parser.parse_known_args()
    config = {}
    # configparser can't handle lists, so we have to deal with this separately:
    listvars = {'modules': [], 'exports': []}
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = configparser.ConfigParser()
            config.read([args.config_file])
            for k in listvars:
                if config.has_option('Defaults', '--' + k):
                    listvars[k] = config.get('Defaults', '--' + k).split(" ")

    parser = argparse.ArgumentParser(
        prog='rocfftslurm',
        epilog="For a detailed usage overview, run: %(prog)s overview",
        parents=[conf_parser])

    # NB: confgparser requires a value for boolean arguments, so action='store_true' isn't really an
    # option for boolean argparse arguments which are also handled by configparser.

    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--logdir',
                        type=str,
                        default=None,
                        help='log base directory')
    parser.add_argument('--builddir',
                        type=str,
                        default=None,
                        help='build directory')
    parser.add_argument('--build',
                        type=lambda x: bool(x.lower() in
                                            ("yes", "true", "t", "1")),
                        default=False)
    parser.add_argument('--ccache',
                        type=lambda x: bool(x.lower() in
                                            ("yes", "true", "t", "1")),
                        default=False)
    parser.add_argument('--buildcraympi',
                        type=lambda x: bool(x.lower() in
                                            ("yes", "true", "t", "1")),
                        default=False)
    parser.add_argument('--launcher',
                        type=str,
                        default=None,
                        help='mpi launcher')
    parser.add_argument('--partition',
                        type=str,
                        default=None,
                        help='slurm partition')
    parser.add_argument('--acct', type=str, default=None, help='slurm acct')
    parser.add_argument('--modules',
                        nargs='+',
                        default=[],
                        help='Modules to load')
    parser.add_argument('--exports',
                        nargs='+',
                        default=[],
                        help='Exports for test')
    parser.add_argument('--gpuspernode',
                        type=int,
                        help='Gpus per node on cluster',
                        default=1)
    parser.add_argument(
        '--gpuidvar',
        type=str,
        help=
        'Node-local environment variable for computing ROCR_VISIBLE_DEVICES')
    parser.add_argument('--maxnodes',
                        type=int,
                        default=1,
                        help='Maximum number of nodes to use')

    if args.config_file:
        for k, v in config.items("Defaults"):
            parser.parse_args([str(k), str(v)], args)
        for k in listvars:
            if (len(listvars.get(k)) > 0):
                vars(args)[k] = listvars.get(k)

    args = parser.parse_args(remaining_args, args)

    print("logdir:", args.logdir)
    if args.logdir is None:
        print("logdir is required")
        sys.exit(1)
    if not os.path.exists(args.logdir):
        try:
            os.makedirs(args.logdir)
        except:
            print("unable to create log dir")
            sys.exit(1)

    print("build dir:", args.builddir)
    if args.builddir is None:
        print("build dir is required")
        sys.exit(1)
    if args.build:
        if not os.path.exists(args.builddir):
            print("creating build dir")
            try:
                os.makedirs(args.builddir)
            except:
                print("unable to create build dir")
                sys.exit(1)
    if not os.path.exists(args.builddir):
        print("Working directory", args.builddir, "does not exist")
        sys.exit(1)

    print("launcher:", args.launcher)
    print("acct:", args.acct)
    print("partition:", args.partition)
    print("gpus per node", args.gpuspernode)
    print("modules:", args.modules)
    print("exports", args.exports)
    print("build library?", args.build)
    print("buildcraympi library?", args.buildcraympi)

    # Main job script:

    buildjob = None
    if args.build:
        # Run a compile job first.
        buildcmd = "cmake"
        buildcmd += " -DCMAKE_CXX_COMPILER=amdclang++"
        buildcmd += " -DCMAKE_C_COMPILER=amdclang"
        if args.ccache:
            buildcmd += " -DCMAKE_C_COMPILER_LAUNCHER=ccache"
            buildcmd += " -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
        buildcmd += " -DBUILD_CLIENTS=on"
        buildcmd += " -DROCFFT_KERNEL_CACHE_ENABLE=off"
        buildcmd += " -DROCFFT_MPI_ENABLE=on "
        if args.buildcraympi:
            buildcmd += " -DROCFFT_CRAY_MPI_ENABLE=on"
        buildcmd += " .."
        buildcmd += " && make"

        # Set up the basic slurm parameters:
        buildparams = rocslurm.sbatchparams()
        if args.partition != None:
            buildparams.partition = args.partition
        buildparams.nnodes = 1
        if args.acct != None:
            buildparams.acct = args.acct
        buildparams.modules = args.modules
        buildparams.exports = args.exports
        buildparams.ntaskspernode = 1
        buildparams.gpuspernode = 1
        buildparams.timelimit = datetime.timedelta(hours=2)
        buildparams.ntaskspernode = 1

        buildjob = rocslurm.sbatch("build",
                                   buildparams,
                                   args.logdir,
                                   args.builddir,
                                   buildcmd,
                                   verbose=args.verbose)

        buildid = buildjob.jobid

    # Set up the basic slurm parameters:
    jobparams = rocslurm.sbatchparams()
    if args.partition != None:
        jobparams.partition = args.partition
    jobparams.nnodes = 1
    if args.acct != None:
        jobparams.acct = args.acct
    jobparams.modules = args.modules
    jobparams.exports = args.exports
    jobparams.ntaskspernode = 1
    jobparams.gpuspernode = args.gpuspernode
    jobparams.timelimit = datetime.timedelta(hours=2)
    jobparams.ntaskspernode = jobparams.gpuspernode
    if buildjob != None:
        jobparams.afterok = [buildjob.jobid]

    jobs = []

    if buildjob != None:
        jobs.append(buildjob)

    for nodes in range(1, args.maxnodes + 1):
        jobparams.nnodes = nodes
        mpigpucmd = "export LD_LIBRARY_PATH=" + args.builddir + "/library/src/:${LD_LIBRARY_PATH}\n"
        mpigpucmd += str(
            Path(__file__).resolve().parent / "rocfft_mpi_test.py")
        mpigpucmd += " --worker " + args.builddir + "/clients/staging/rocfft_mpi_worker"
        mpigpucmd += " --rocffttest " + args.builddir + "/clients/staging/rocfft-test"
        mpigpucmd += " --launcher srun"
        if args.gpuidvar != None:
            mpigpucmd += " --gpuidvar " + args.gpuidvar
        mpigpucmd += " --nranks " + str(
            jobparams.ntaskspernode * jobparams.nnodes)
        mpigpucmd += " --gpusperrank " + str(1)  #args.gpuspernode

        jobparams.ntaskspernode = 8
        mpijob = rocslurm.sbatch("mpi" + str(nodes),
                                 jobparams,
                                 args.logdir,
                                 args.builddir,
                                 mpigpucmd,
                                 verbose=args.verbose)
        jobs.append(mpijob)

    # Report on job status
    jobparams.ntaskspernode = 1
    jobparams.nnodes = 1
    rocslurm.reportonjobs(jobparams, args.logdir, jobs, verbose=args.verbose)


if __name__ == '__main__':
    main()

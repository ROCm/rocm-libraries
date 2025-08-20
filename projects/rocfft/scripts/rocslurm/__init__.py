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

import subprocess, os, sys, datetime


class sbatchparams:

    def __init__(self):
        self.partition = None
        self.acct = None
        self.nnodes = 1
        self.modules = []
        self.exports = []
        self.ntaskspernode = None
        self.gpuspernode = None
        self.afterok = []
        self.afterany = []
        self.timelimit = None


class sbatchjob:

    def __init__(self, jobid, outfilename, errfilename):
        self.jobid = jobid
        self.outfilename = outfilename
        self.errfilename = errfilename


def sbatch(jobname, params, logdir, workdir, jobcmd, verbose=0):
    sbatchcmd = []
    sbatchcmd.append("sbatch")

    batchscript = "#!/bin/bash\n"

    batchscript += "#SBATCH --job-name=" + jobname + "\n"
    outfilename = os.path.join(logdir, "%j", jobname + ".out")
    batchscript += "#SBATCH --output=" + outfilename + "\n"
    errfilename = os.path.join(logdir, "%j", jobname + ".err")
    batchscript += "#SBATCH --error=" + errfilename + "\n"
    if params.partition != None:
        batchscript += "#SBATCH --partition=" + params.partition + "\n"
    batchscript += "#SBATCH --nodes=" + str(params.nnodes) + "\n"
    if params.ntaskspernode != None:
        batchscript += "#SBATCH --ntasks-per-node=" + str(
            params.ntaskspernode) + "\n"
    if params.gpuspernode != None:
        batchscript += "#SBATCH --gpus-per-node=" + str(
            params.gpuspernode) + "\n"
    if params.timelimit != None:
        batchscript += "#SBATCH --time=" + str(params.timelimit) + "\n"
    if params.acct != None:
        batchscript += "#SBATCH --account=" + params.acct + "\n"
    if len(params.afterany) > 0:
        batchscript += "#SBATCH --dependency=afterany:" + ",".join(
            str(x) for x in params.afterany) + "\n"
    if len(params.afterok) > 0:
        batchscript += "#SBATCH --dependency=afterok:" + ",".join(
            str(x) for x in params.afterok) + "\n"

    for module in params.modules:
        batchscript += "module load " + module + "\n"
    for export in params.exports:
        batchscript += "export " + export + "\n"

    batchscript += "cd " + workdir + "\n"

    batchscript += jobcmd

    if verbose > 0:
        print(verbose)
        print(batchscript)

    p = subprocess.Popen(sbatchcmd,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    for line in batchscript.splitlines():
        byteline = str.encode(line + "\n")
        p.stdin.write(byteline)
    p.stdin.close()

    p.wait()

    if p.returncode == 0:
        outwords = p.stdout.read().decode("utf-8")
        jobid = int(outwords.split()[-1])
        print("queued", jobname, jobid)
        newjob = sbatchjob(jobid, outfilename.replace("%j", str(jobid)),
                           errfilename.replace("%j", str(jobid)))
        return newjob
    else:
        print("sbatch failed")
        print(batchscript)
        print(p.stdout.read().decode("utf-8"))
        print(p.stderr.read().decode("utf-8"))
        sys.exit(1)

    return None


def scancel(job):
    scancelcmd = ["scancel", str(job.jobid)]
    p = subprocess.Popen(scancelcmd)
    p.wait()
    if p.returncode == 0:
        print("cancelled", job.jobid)
    else:
        print("scancel failed")
        sys.exit(1)


def reportonjobs(params, logdir, jobs, verbose=0):
    jobids = []
    for job in jobs:
        jobids.append(job.jobid)

    batchscript = "#!/bin/bash\n"

    jobname = "report"
    batchscript += "#SBATCH --job-name=" + jobname + "\n"
    batchscript += "#SBATCH --nodes=1\n"
    outfilename = os.path.join(logdir, "%j", jobname + ".out")
    batchscript += "#SBATCH --output=" + outfilename + "\n"
    errfilename = os.path.join(logdir, "%j", jobname + ".err")
    batchscript += "#SBATCH --error=" + errfilename + "\n"
    if params.acct != None:
        batchscript += "#SBATCH --account=" + params.acct + "\n"
    if params.partition != None:
        batchscript += "#SBATCH --partition=" + params.partition + "\n"
    if params.timelimit != None:
        batchscript += "#SBATCH --time=0:5:00\n"
    # TODO: copy out and err log files to the report working dir.
    # TODO: out and err files, and job name.
    if len(jobids) > 0:
        batchscript += "#SBATCH --dependency=afterany:" + ",".join(
            str(x) for x in jobids) + "\n"

    reportcmd = ""

    for job in jobs:
        joblogdir = os.path.dirname(os.path.realpath(job.outfilename))
        #print(joblogdir)
        reportcmd += "cp -r " + joblogdir + " " + logdir + "/$SLURM_JOB_ID\n"


    reportcmd += "sacct -X -j " + ",".join(str(x) for x in jobids) \
        + " -o JobName,JobID,state,Elapsed,ExitCode\n"

    batchscript += reportcmd + "\n"

    if verbose > 0:
        print(batchscript)

    sbatchcmd = []
    sbatchcmd.append("sbatch")
    p = subprocess.Popen(sbatchcmd,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    for line in batchscript.splitlines():
        byteline = str.encode(line + "\n")
        p.stdin.write(byteline)
    p.stdin.close()

    #sacct -X -j 9628127,9628128 -o JobID,state,ExitCode

    #reportjob = sbatch("finalreport", params, logdir, workdir, jobcmd, verbose=False):

    p.wait()

    if p.returncode == 0:
        outwords = p.stdout.read().decode("utf-8")
        jobid = int(outwords.split()[-1])
        print("queued report", jobid)
        newjob = sbatchjob(jobid, outfilename.replace("%j", str(jobid)),
                           errfilename.replace("%j", str(jobid)))
        return newjob
    else:
        print("sbatch failed")
        print(batchscript)
        print(p.stdout.read().decode("utf-8"))
        print(p.stderr.read().decode("utf-8"))
        sys.exit(1)

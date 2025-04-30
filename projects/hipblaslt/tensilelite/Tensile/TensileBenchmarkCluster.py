################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import shlex, subprocess
import sys
import os
import argparse

from .BenchmarkSplitter import BenchmarkSplitter
from .Configuration import ProjectConfig
from .TensileBenchmarkClusterScripts import ScriptWriter
from Tensile.Utilities.merge import mergePartialLogics

try:
    import mgzip as gzip
except ImportError:
    print("Package mgzip not found: docker zipping will be slow. Install pip3 install mgzip to improve performance.")

    # Fallback package import
    import gzip

class BenchmarkImplSLURM(object):

    # baseImage: remote URL to rocm build
    # dockerFilePath: path to docker file for slurm tensile build
    # tag: name:TAG of new image
    # outDir: where to export the new image
    # tensile<Fork/Branch/Commit>: tensile code to use on github
    @staticmethod
    def __createTensileBenchmarkContainer(baseImage, dockerFilePath, tag, outDir, logDir, tensileFork, tensileBranch, tensileCommit):
        """
        Build a docker container with a specific
        ROCm image, Tensile branch and tag. Docker
        will pre-build the Tensile Client and
        configure the container to run the benchmark.
        """
        # Save stdout and stderr to file
        with open(os.path.join(logDir, "dockerBuildLog.log"), 'w') as logFile:

            # Docker build command
            buildCmd = str("\
                docker build \
                -t {0} \
                --pull -f {1} \
                --build-arg user_uid=$UID \
                --build-arg base_image={2} \
                --build-arg tensile_fork={3} \
                --build-arg tensile_branch={4} \
                --build-arg tensile_commit={5} \
                . ").format(tag, dockerFilePath, baseImage, tensileFork, tensileBranch, tensileCommit)

            # Build container and save output streams
            print("Building docker image: {0} ...".format(tag))
            print(buildCmd)
            subprocess.check_call(shlex.split(buildCmd), stdout=logFile, stderr=logFile)
            print("Done building docker image!")

            # Docker save command
            imageBaseName = tag.split(':')[0]
            archivePath = os.path.join(outDir, imageBaseName + str(".tar.gz"))
            saveCmd = str("docker save {0}").format(tag)

            # Docker will save .tar binary to stdout as long as it's not attached to console.
            # Pipe stdout into gzip to get smaller .tar.gz archive
            print("Saving docker image: {0}  to {1} ...".format(tag, archivePath))

            with gzip.open(archivePath, 'wb') as zipFile:
                with subprocess.Popen(shlex.split(saveCmd), stdout=subprocess.PIPE, stderr=logFile) as proc:
                    zipFile.write(proc.stdout.read())

            print("Done saving docker image!")

    @staticmethod
    def __createClusterBenchmarkScripts(baseDir, tasksDir, taskScriptName, jobScriptName):
        """
        Build SLURM specific scripts.
        Entrypoint is the 'sbatch' command, which will request resources and
        enqueue several taskScripts.
        TaskScripts each enqueue a benchmark run to
        the cluster with 'srun'.
        NodeScripts are the actual host code on each node to run the benchmark.
        In this case, they will load the docker image and mount directories
        that contain the .yaml file, and the results.
        """
        # Entrypoint - SLURM specific
        ScriptWriter.writeBenchmarkJobScript(jobScriptName, baseDir)

        # Job submission script - SLURM specific
        ScriptWriter.writeBenchmarkTaskScript(taskScriptName, baseDir)

        # Move each individual config into it's own subfolder and create the node host script
        # These scripts are NOT SLURM specific but rely on the implementation of the executable.
        configFiles = [f for f in os.listdir(tasksDir) if os.path.isfile(os.path.join(tasksDir, f))]
        for f in configFiles:
            (baseFileName, _) = os.path.splitext(f)
            configSubdir = os.path.join(tasksDir, baseFileName)
            try:
                os.makedirs(configSubdir)
            except:
                pass
            os.rename(os.path.join(tasksDir, f), os.path.join(configSubdir, f))
            ScriptWriter.writeBenchmarkNodeScript(baseFileName + ".sh", configSubdir)

    @classmethod
    def initializeConfig(cls, config):
        """
        Store SLURM backend-specific configurations.
        These can all be overridden via commandline
        """
        # Dirs
        rootTensileDir = config["RootTensileDir"]

        # SLURM config
        baseSection = config.createSection("SLURM")

        # Script names
        section = baseSection.createSection("SCRIPTS")
        section.createValue("JobScriptName", "runBenchmark.sh")
        section.createValue("TaskScriptName", "enqueueTask.sh")

        # Docker build
        section = baseSection.createSection("DOCKER")
        section.createValue("DockerBaseImage", "compute-artifactory.amd.com:5000/rocm-plus-docker/compute-rocm-dkms-amd-feature-targetid:3004-STG1")
        section.createValue("DockerBuildFile", os.path.join(rootTensileDir, "docker", "dockerfile-tensile-tuning-slurm"))
        section.createValue("DockerImageName", "tensile-tuning-cluster-executable")
        section.createValue("DockerImageTag", "TEST")
        section.createValue("TensileFork", "ROCmSoftwarePlatform")
        section.createValue("TensileBranch", "develop")
        section.createValue("TensileCommit", "HEAD")

    @classmethod
    def generateBenchmark(cls, config):
        """
        Build executables used in SLURM backend,
        based on configuration parameters
        """
        # Dirs
        (baseDir, tasksDir, imageDir, logsDir) = \
            (config["BenchmarkBaseDir"], \
            config["BenchmarkTasksDir"], \
            config["BenchmarkImageDir"], \
            config["BenchmarkLogsDir"])

        # Create the base image for task executable in image dir
        sConfig = config["SLURM"]["DOCKER"]
        cls.__createTensileBenchmarkContainer( \
            sConfig["DockerBaseImage"], \
            sConfig["DockerBuildFile"], \
            "{0}:{1}".format(sConfig["DockerImageName"], sConfig["DockerImageTag"]), \
            imageDir, \
            logsDir, \
            sConfig["TensileFork"], \
            sConfig["TensileBranch"], \
            sConfig["TensileCommit"])

        # Create the scripts that will be used to invoke the benchmark in base dir
        # Create the scripts that node hosts will run in the tasks dir
        sConfig = config["SLURM"]["SCRIPTS"]
        cls.__createClusterBenchmarkScripts( \
            baseDir, \
            tasksDir, \
            sConfig["TaskScriptName"], \
            sConfig["JobScriptName"])

    @classmethod
    def preInvokeBenchmark(cls, config):
        pass

    @classmethod
    def invokeBenchmark(cls, config):

        # Dirs
        (baseDir, tasksDir, imageDir, resultsDir, logsDir) = \
            (config["BenchmarkBaseDir"], \
            config["BenchmarkTasksDir"], \
            config["BenchmarkImageDir"], \
            config["BenchmarkResultsDir"], \
            config["BenchmarkLogsDir"])

        # Entry point script
        scriptName = config["SLURM"]["SCRIPTS"]["JobScriptName"]
        runScriptPath = os.path.join(baseDir, scriptName)

        # Enqueue script
        enqueueScriptName = config["SLURM"]["SCRIPTS"]["TaskScriptName"]
        enqueueScriptPath = os.path.join(baseDir, enqueueScriptName)

        # Log file
        (name, ext) = os.path.splitext(scriptName)
        logFilePath = os.path.join(logsDir, name + ".log")

        invokeCmd = str("\
                {0} \
                -i {1} \
                -l {2} \
                -r {3} \
                -s {4} \
                -t {5}").format(runScriptPath, imageDir, logsDir, resultsDir, enqueueScriptPath, tasksDir)

        with open(logFilePath, "wt") as logFile:
            subprocess.check_call(shlex.split(invokeCmd), stdout=logFile, stderr=logFile)

    @classmethod
    def postInvokeBenchmark(cls, benchmarkObj):
        pass


class TensileBenchmarkCluster(object):
    """
    This is the main driver class for building and running
    benchmarks with a cluster-configured backend.

    The general workflow (main):
    - Deploy benchmark
    - Invoke benchmark
    - Merge results

    Expected backend interface:
    - initializeConfig
    - generateBenchmark
    - preInvokeBenchmark
    - invokeBenchmark
    - postInvokeBenchmark

    Other notes:
    - The command line provides full control over
    workflow and configuration.
    - You can choose different backends as they
    become available
    - Workflow steps may be selected individually
    - Any part of configuration may be overidden
    by command line args.
    """

    def __init__(self, cmdlineArgs):
        self._config = ProjectConfig()
        self.__initializeConfig(cmdlineArgs)

    def __initializeConfig(self, cmdlineArgs):

        """
        Initialize configuration parameters needed
        to build and run the benchmark.
        """

        # 1. Parse command line
        args = self.__parseArgs(cmdlineArgs)

        # 2. Setup default configurations that depend on inputs
        # Directories
        self._config.createValue("RootTensileDir", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._config.createValue("BenchmarkLogicPath", args.BenchmarkLogicPath)
        self._config.createValue("BenchmarkBaseDir", args.DeploymentPath)
        self._config.createValue("BenchmarkTasksDir", os.path.join(args.DeploymentPath, "Tasks"))
        self._config.createValue("BenchmarkImageDir", os.path.join(args.DeploymentPath, "Image"))
        self._config.createValue("BenchmarkResultsDir", os.path.join(args.DeploymentPath, "Results"))
        self._config.createValue("BenchmarkFinalLogicDir", os.path.join(args.DeploymentPath, "Results", "Final"))
        self._config.createValue("BenchmarkLogsDir", os.path.join(args.DeploymentPath, "Logs"))

        # Benchmarking
        self._config.createValue("BenchmarkTaskSize", 10) # Sizes per benchmark file
        self._config.addConstraint("BenchmarkTaskSize > 0")

        self._config.createValue("RunDeployStep", True and (not args.RunOnly and not args.ResultsOnly and not args.RunAndResultsOnly))
        self._config.createValue("RunBenchmarkStep", True and (not args.DeployOnly and not args.ResultsOnly))
        self._config.createValue("RunResultsStep", True and (not args.DeployOnly and not args.RunOnly))
        self._config.addConstraint("RunDeployStep or RunBenchmarkStep or RunResultsStep")

        # Results merging settings
        self._config.createValue("FinalLogicForceMerge", False)
        self._config.createValue("FinalLogicTrim", True)

        # Initialize the backend implementation
        if args.ClusterBackend.lower() == "slurm":
            self._backendImpl = BenchmarkImplSLURM
        else:
            raise NotImplementedError("Cluster backend not recognized")

        # 3. Setup backend configuration
        self._backendImpl.initializeConfig(self._config)

        # 4. Override requested config parameters
        overrideArgs = {}
        for key, value in args.benchmark_parameters:
            overrideArgs[key] = value
        self.__overrideConfig(overrideArgs)

        self._config.checkConstraints()

    @staticmethod
    def __parseArgs(cmdlineArgs):
        """
        Parse input arguments to call script
        """

        def splitExtraParameters(par):
            """
            Allows the --benchmark-parameters option to specify any parameters from the command line.
            """
            (key, value) = par.split("=")
            value = eval(value)
            return (key, value)

        # Parse incoming args
        argParser = argparse.ArgumentParser()
        argParser.add_argument("BenchmarkLogicPath",     help="Path to benchmark config .yaml files.")
        argParser.add_argument("DeploymentPath",         help="Where to deploy benchmarking files. Should target a directory on shared nfs mount of cluster.")
        argParser.add_argument("--cluster-backend",      dest="ClusterBackend", type=str, default="slurm", help="Choose backend plugin to run benchmark")
        argParser.add_argument("--deploy-only",          dest="DeployOnly", action="store_true", default=False, help="Deploy benchmarking files only without running or reducing results")
        argParser.add_argument("--run-only",             dest="RunOnly", action="store_true", default=False, help="Run benchmark without deploying or reducing results")
        argParser.add_argument("--results-only",         dest="ResultsOnly", action="store_true", default=False, help="Reduce results without deploying or running")
        argParser.add_argument("--run-and-results-only", dest="RunAndResultsOnly", action="store_true", default=False, help="Run benchmark and reduce results without deploying")
        argParser.add_argument("--benchmark-parameters", nargs="+", type=splitExtraParameters, default=[], help="Pairs of X=Y assignments. Note: if Y is a string, then it must have escaped quotes \\\"Y\\\"")
        return argParser.parse_args()

    def __overrideConfig(self, args):
        """
        Assign Benchmark Parameters
        Each parameter has a default parameter, and the user
        can override them, those overridings happen here
        """
        for key in args:
            value = args[key]
            if key not in self._config:
                print("Warning: Benchmark parameter {0} = {1} unrecognised.".format( key, value ))
            else:
                self._config[key] = value

    def __generateClusterBenchmark(self):
        """
        Generate everything for the benchmark to run
        """
        # Ensure that we have all directories
        self.ensurePath(self.baseDir())
        self.ensurePath(self.tasksDir())
        self.ensurePath(self.imageDir())
        self.ensurePath(self.resultsDir())
        self.ensurePath(self.finalLogicDir())
        self.ensurePath(self.logsDir())

        # Split master config into smaller task-sized configs
        # These are stored under the tasks dir
        BenchmarkSplitter.splitBenchmarkBySizes( \
            self._config["BenchmarkLogicPath"], \
            self.tasksDir(), \
            self._config["BenchmarkTaskSize"], \
            suffixFormat="{:04d}") # Support lots of jobs up to 9999

        # Delegate to the backend implementation to generate everything it needs for the benchmark run
        self._backendImpl.generateBenchmark(self._config)

    def __runClusterBenchmark(self):
        """
        Invoke backend benchmark
        """
        # Delegate to the backend implementation to invoke
        self._backendImpl.preInvokeBenchmark(self._config)
        self._backendImpl.invokeBenchmark(self._config)
        self._backendImpl.postInvokeBenchmark(self._config)

    def __combineClusterBenchmarkResults(self):
        """
        Combine all partial results into the final
        kernel logic.
        Each partial result is expected to be under
        the base/ResultsDir/<PART>/3_LibraryLogic
        directory.
        """
        (resultsDir, finalLogicDir) = (self.resultsDir(), self.finalLogicDir())

        # Find the partial logic .yaml files
        # These are in each result directory under 3_LibraryLogic
        resultsDirs = [os.path.join(resultsDir, d, "3_LibraryLogic") for d in os.listdir(resultsDir) if os.path.isdir(os.path.join(resultsDir, d, "3_LibraryLogic"))]

        resultsFiles = []
        for d in resultsDirs:
            resultsFiles += [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) ]

        if len(resultsDirs) != len(resultsFiles):
            print("Warning: inconsistent number of expected results. Check that results are complete.")

        mergePartialLogics(
            resultsFiles, \
            finalLogicDir, \
            self._config["FinalLogicForceMerge"], \
            self._config["FinalLogicTrim"])

    def workflowSteps(self):
        """
        Helper function to easily receive a tuple of important
        workflow step indicators
        """
        return( \
            self._config["RunDeployStep"], \
            self._config["RunBenchmarkStep"], \
            self._config["RunResultsStep"])

    def config(self):
        return self._config

    def rootTensileDir(self):
        return self.config()["RootTensileDir"]

    def baseDir(self):
        return self.config()["BenchmarkBaseDir"]

    def tasksDir(self):
        return self.config()["BenchmarkTasksDir"]

    def imageDir(self):
        return self.config()["BenchmarkImageDir"]

    def resultsDir(self):
        return self.config()["BenchmarkResultsDir"]

    def finalLogicDir(self):
        return self.config()["BenchmarkFinalLogicDir"]

    def logsDir(self):
        return self.config()["BenchmarkLogsDir"]

    @staticmethod
    def ensurePath(path):
        """
        Helper function to create path if necessary
        """
        try:
            os.makedirs(path)
        except OSError:
            pass
        return path

    def main(self):

        """
        Main driver for benchmark:
        - Deploy
        - Run
        - Results
        """
        (doDeploy, doBenchmark, doResults) = self.workflowSteps()

        # Deploy
        if doDeploy is True:
            print("Preparing benchmarking files...")
            self.__generateClusterBenchmark()
            print("Finished preparing benchmarking files")

        # Benchmark invoke
        if doBenchmark is True:
            print("Running benchmark tasks (this might take a while)...")
            self.__runClusterBenchmark()
            print("Finished benchmark tasks")

        # Combining results
        if doResults is True:
            print("Combining benchmark results...")
            self.__combineClusterBenchmarkResults()
            print("Final logic file saved to: {0}".format(self.finalLogicDir()))

        print("Finished")

def main():
    entryPoint = TensileBenchmarkCluster(sys.argv[1:])
    entryPoint.main()

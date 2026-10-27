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

import os

class ScriptHelper(object):
    """
    Helper class to facilitate formatting when
    writing bash scripts.
    """

    @classmethod
    def genLine(cls, text):
        return str("{0}\n").format(text)

    @classmethod
    def genComment(cls, text):
        return cls.genLine(str("#") + str(text))

    @classmethod
    def genEcho(cls, text):
        return cls.genLine(str("echo \"{0}\";").format(str(text)))

    @classmethod
    def makeExecutable(cls, path):
        mode = os.stat(path).st_mode
        mode |= (mode & 0o444) >> 2    # copy R bits to X
        os.chmod(path, mode)

class ScriptWriter(object):
    """
    A script writing utility class that holds a string
    buffer and wraps writing to the buffer.
    Has factory functions to build scripts with regular
    workflow interface.
    """
    def __init__(self):
        self._buffer = ""

    def writeLine(self, text):
        self._buffer += ScriptHelper.genLine(text)

    def writeComment(self, text):
        self._buffer += ScriptHelper.genComment(text)

    def writeEcho(self, text):
        self._buffer += ScriptHelper.genEcho(text)

    def dumpToFile(self, fileName, outputDir = "."):
        scriptPath = os.path.join(outputDir, fileName)
        with open(scriptPath, "w") as f:
            f.write(self._buffer)
        ScriptHelper.makeExecutable(scriptPath)

    def __str__(self):
        return self._buffer

    @classmethod
    def writeScript(bufferClass, writerClass, fileName, outputDir = "."):
        buffer = bufferClass()
        writerClass.writeHeader(buffer)
        writerClass.writeBody(buffer)
        writerClass.writeEpilogue(buffer)
        buffer.dumpToFile(fileName, outputDir)

    @classmethod
    def previewScript(bufferClass, writerClass):
        buffer = bufferClass()
        writerClass.writeHeader(buffer)
        writerClass.writeBody(buffer)
        writerClass.writeEpilogue(buffer)
        print(buffer)

    @classmethod
    def writeBenchmarkNodeScript(cls, fileName, outputDir="."):
        cls.writeScript(BenchmarkNodeWriter, fileName, outputDir)

    @classmethod
    def writeBenchmarkTaskScript(cls, fileName, outputDir="."):
        cls.writeScript(BenchmarkTaskWriter, fileName, outputDir)

    @classmethod
    def writeBenchmarkJobScript(cls, fileName, outputDir="."):
        cls.writeScript(BenchmarkJobWriter, fileName, outputDir)


class BenchmarkNodeWriter(object):
    """
    Node host run script.
    Creates an instance of the docker image and
    mounts the logs, results and tasks directories
    and finally invokes the benchmark on the
    supplied yaml.
    """
    @staticmethod
    def __initHeader(sBuffer):
        sBuffer.writeComment("!/bin/bash")
        sBuffer.writeComment("Benchmark node script")
        sBuffer.writeComment("Actual host task to be run by a SLURM worker node")
        sBuffer.writeLine("")

    @staticmethod
    def __initializeFuncs(sBuffer):
        sBuffer.writeComment("Funcs")
        sBuffer.writeLine("usage() { echo \"Usage: $0 [-i <path_to_docker_image>] [-l <log_dir>] [-r <result_dir>] [-t <task_dir>]\" 1>&2; exit 1; }")
        sBuffer.writeLine("failAndExit() { echo \"FAILED: $1\" 1>&2; exit 1; }")
        sBuffer.writeLine("")

    @staticmethod
    def __parseArgs(sBuffer):
        sBuffer.writeComment("Parse arguments")
        sBuffer.writeLine("while getopts i:l:r:t: flag")
        sBuffer.writeLine("do")
        sBuffer.writeLine("    case \"${flag}\" in")
        sBuffer.writeLine("        i) dockerImagePath=${OPTARG};;")
        sBuffer.writeLine("        l) logDir=${OPTARG};;")
        sBuffer.writeLine("        r) resultDir=${OPTARG};;")
        sBuffer.writeLine("        t) taskDir=${OPTARG};;")
        sBuffer.writeLine("        *) usage;;")
        sBuffer.writeLine("    esac")
        sBuffer.writeLine("done")
        sBuffer.writeLine("[ -z \"${dockerImagePath}\" ] || [ -z \"${logDir}\" ] || [ -z \"${resultDir}\" ]  || [ -z \"${taskDir}\" ] && usage;")
        sBuffer.writeLine("[ -d \"${logDir}\" ] && [ -d \"${resultDir}\" ] && [ -d \"${taskDir}\" ] || failAndExit \"Directory not found\";")
        sBuffer.writeLine("[ -f \"${dockerImagePath}\" ] || failAndExit \"Docker image not found: $dockerImagePath\";")
        sBuffer.writeEcho("Docker image path: $dockerImagePath")
        sBuffer.writeEcho("Path to log: $logDir")
        sBuffer.writeEcho("Path to task: $taskDir")
        sBuffer.writeEcho("Path to result: $resultDir")
        sBuffer.writeLine("")

    @staticmethod
    def __loadDockerImage(sBuffer):
        sBuffer.writeComment("Import docker image")
        sBuffer.writeComment("Successful output is \"Loaded image: imageName:TAG\"")
        sBuffer.writeEcho("Loading docker image...")
        sBuffer.writeLine("dockerLoadOutput=`docker load < $dockerImagePath || failAndExit \"Importing docker image\"`")
        sBuffer.writeLine("")
        sBuffer.writeEcho("Docker Load Result: $dockerLoadOutput")
        sBuffer.writeLine("[ -z \"${dockerLoadOutput}\" ] && failAndExit \"Loading docker image $dockerImagePath\"")
        sBuffer.writeLine("")
        sBuffer.writeComment("Split the output on ':' or spaces")
        sBuffer.writeComment("Capture the docker image name and tag")
        sBuffer.writeLine("outputSplit=(${dockerLoadOutput//:/ })")
        sBuffer.writeLine("dockerName=${outputSplit[2]}")
        sBuffer.writeLine("dockerTag=${outputSplit[3]}")
        sBuffer.writeLine("")
        sBuffer.writeComment("Find docker image record with correct name and tag")
        sBuffer.writeLine("dockerImageQuery=`docker images | grep -i \"$dockerName\" | grep -i \"$dockerTag\"`")
        sBuffer.writeLine("[ -z \"${dockerImageQuery}\" ] && failAndExit \"Finding docker image record for $dockerName:$dockerTag\"")
        sBuffer.writeLine("")
        sBuffer.writeComment("Get docker image ID")
        sBuffer.writeLine("outputSplit=(${dockerImageQuery//   / })")
        sBuffer.writeLine("dockerImageId=${outputSplit[2]}")
        sBuffer.writeLine("[ -z \"${dockerImageId}\" ] && failAndExit \"Finding docker image id for $dockerName:$dockerTag\"")
        sBuffer.writeEcho("Success: Loaded Image ID: ${dockerImageId}")
        sBuffer.writeLine("")

    @staticmethod
    def __runTask(sBuffer):
        sBuffer.writeComment("Check the task dir exists")
        sBuffer.writeLine("")
        sBuffer.writeComment("Run container with mounted taskDir")
        sBuffer.writeEcho("Running container... (this might take some time):")
        sBuffer.writeLine("runCmd=\"docker run --rm --network=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v \"$taskDir\":\"/TaskDir\" -v \"$resultDir\":\"/ResultDir\" -v \"$logDir\":\"/LogDir\" $dockerImageId\"")
        sBuffer.writeEcho("$runCmd")
        sBuffer.writeLine("$runCmd")
        sBuffer.writeLine("returnCode=$?")

    @staticmethod
    def __exit(sBuffer):
        sBuffer.writeEcho("Done!")
        sBuffer.writeLine("exit $returnCode")

    @classmethod
    def writeHeader(cls, sBuffer):
        cls.__initHeader(sBuffer)

    @classmethod
    def writeBody(cls, sBuffer):
        cls.__initializeFuncs(sBuffer)
        cls.__parseArgs(sBuffer)
        cls.__loadDockerImage(sBuffer)
        cls.__runTask(sBuffer)

    @classmethod
    def writeEpilogue(cls, sBuffer):
        cls.__exit(sBuffer)

class BenchmarkTaskWriter(object):
    """
    SLURM specific script.
    Invokes the 'srun' command to queue a task.
    SLURM environment variables tell us which
    indexed task to run as part of the array.
    The final task submitted is the node script
    which will run on one of the hosts.
    """
    @staticmethod
    def __initHeader(sBuffer):
        sBuffer.writeComment("!/bin/bash")
        sBuffer.writeComment("Benchmark task enqueue script")
        sBuffer.writeComment("This script will configure and commit a single task to the cluster.")
        sBuffer.writeComment("Intended to be the run target of SLURM SBATCH multiplexer")
        sBuffer.writeLine("")

    @staticmethod
    def __initializeFuncs(sBuffer):
        sBuffer.writeComment("Funcs")
        sBuffer.writeLine("usage() { echo \"Usage: $0 [-i <image_dir>] [-l <logs_dir>] [-r <results_dir>] [-t <tasks_dir>] \" 1>&2; exit 1; }")
        sBuffer.writeLine("failAndExit() { echo \"FAILED: $1\" 1>&2; exit 1; }")
        sBuffer.writeLine("")

    @staticmethod
    def __parseArgs(sBuffer):
        sBuffer.writeComment("Parse arguments")
        sBuffer.writeLine("while getopts i:l:r:t: flag")
        sBuffer.writeLine("do")
        sBuffer.writeLine("    case \"${flag}\" in")
        sBuffer.writeLine("        i) imageDir=${OPTARG};; ")
        sBuffer.writeLine("        l) logsDir=${OPTARG};;")
        sBuffer.writeLine("        r) resultsDir=${OPTARG};;")
        sBuffer.writeLine("        t) tasksDir=${OPTARG};;")
        sBuffer.writeLine("        *) usage;;")
        sBuffer.writeLine("    esac")
        sBuffer.writeLine("done")
        sBuffer.writeLine("[ -z \"${imageDir}\" ] || [ -z \"${logsDir}\" ] || [ -z \"${resultsDir}\" ] || [ -z \"${tasksDir}\" ] && usage;")
        sBuffer.writeLine("[ -d \"${imageDir}\" ] && [ -d \"${logsDir}\" ] && [ -d \"${resultsDir}\" ] && [ -d \"${tasksDir}\" ] || failAndExit \"Directory not found\";")
        sBuffer.writeEcho("Image dir: $imageDir")
        sBuffer.writeEcho("Logs dir: $logsDir")
        sBuffer.writeEcho("Tasks dir: $tasksDir")
        sBuffer.writeEcho("Results dir: $resultsDir")
        sBuffer.writeLine("")

    @staticmethod
    def __configureTask(sBuffer):
        sBuffer.writeComment("Setup task parameters")
        sBuffer.writeLine("pushd $tasksDir")
        sBuffer.writeLine("subDirs=(*)")
        sBuffer.writeLine("taskName=${subDirs[$SLURM_ARRAY_TASK_ID]}")
        sBuffer.writeLine("taskDir=$tasksDir/$taskName")
        sBuffer.writeLine("taskResultDir=$resultsDir/$taskName")
        sBuffer.writeLine("mkdir -p $taskResultDir")
        sBuffer.writeLine("taskLogDir=$logsDir/$taskName")
        sBuffer.writeLine("mkdir -p $taskLogDir")
        sBuffer.writeLine("task=($taskDir/*.sh)")
        sBuffer.writeLine("image=($imageDir/*.tar.gz)")
        sBuffer.writeEcho("Task ID: $SLURM_ARRAY_TASK_ID / $SLURM_TASK_MAX")
        sBuffer.writeEcho("Task Name: $taskName")
        sBuffer.writeEcho("Host Name: `hostname`")
        sBuffer.writeLine("")

    @staticmethod
    def __submitTask(sBuffer):
        sBuffer.writeComment("Enqueue node task")
        sBuffer.writeLine("srun -N 1 \"$task\" -i \"$image\" -l \"$taskLogDir\" -r \"$taskResultDir\" -t \"$taskDir\"")
        sBuffer.writeLine("returnCode=$?")

    @staticmethod
    def __exit(sBuffer):
        sBuffer.writeEcho("Done!")
        sBuffer.writeLine("exit $returnCode")

    @classmethod
    def writeHeader(cls, sBuffer):
        cls.__initHeader(sBuffer)

    @classmethod
    def writeBody(cls, sBuffer):
        cls.__initializeFuncs(sBuffer)
        cls.__parseArgs(sBuffer)
        cls.__configureTask(sBuffer)
        cls.__submitTask(sBuffer)

    @classmethod
    def writeEpilogue(cls, sBuffer):
        cls.__exit(sBuffer)

class BenchmarkJobWriter(object):
    """
    SLURM specific script.
    Invokes the 'sbatch' and allocates resources.
    Will create an array of N tasks that will each
    run one instance of the task script.
    N = number of sub-folders inside task folder
    Assumes one node per task.
    Will wait until the entire batch is complete
    before exiting.
    NOTE: the wait feature requires that all task
    scripts have a return code, or will hang forever!
    """
    @staticmethod
    def __initHeader(sBuffer):
        sBuffer.writeComment("!/bin/bash")
        sBuffer.writeComment("Benchmark batch job script")
        sBuffer.writeComment("This script will configure and submit a batch job to the cluster")
        sBuffer.writeComment("This is the SLURM cluster benchmark entry-point")
        sBuffer.writeLine("")

    @staticmethod
    def __initializeFuncs(sBuffer):
        sBuffer.writeComment("Funcs")
        sBuffer.writeLine("usage() { echo \"Usage: $0 [-i <image_dir>] [-l <logs_dir>] [-r <results_dir>] [-s <path_to_task_script>] [-t <tasks_dir>] \" 1>&2; exit 1; }")
        sBuffer.writeLine("failAndExit() { echo \"FAILED: $1\" 1>&2; exit 1; }")
        sBuffer.writeLine("")

    @staticmethod
    def __parseArgs(sBuffer):
        sBuffer.writeComment("Parse arguments")
        sBuffer.writeLine("while getopts i:l:r:s:t: flag")
        sBuffer.writeLine("do")
        sBuffer.writeLine("    case \"${flag}\" in")
        sBuffer.writeLine("        i) imageDir=${OPTARG};;")
        sBuffer.writeLine("        l) logsDir=${OPTARG};;")
        sBuffer.writeLine("        r) resultsDir=${OPTARG};;")
        sBuffer.writeLine("        s) taskScript=${OPTARG};;")
        sBuffer.writeLine("        t) tasksDir=${OPTARG};;")
        sBuffer.writeLine("        *) usage;;")
        sBuffer.writeLine("    esac")
        sBuffer.writeLine("done")
        sBuffer.writeLine("[ -z \"${imageDir}\" ] || [ -z \"${logsDir}\" ] || [ -z \"${resultsDir}\" ] || [ -z \"${tasksDir}\" ]  && usage;")
        sBuffer.writeLine("[ -d \"${imageDir}\" ] && [ -d \"${logsDir}\" ] && [ -d \"${resultsDir}\" ] && [ -d \"${tasksDir}\" ] || failAndExit \"Directory not found\";")
        sBuffer.writeLine("[ -f \"${taskScript}\" ] || failAndExit \"Task script not found\";")
        sBuffer.writeEcho("Image dir: $imageDir")
        sBuffer.writeEcho("Logs dir: $logsDir")
        sBuffer.writeEcho("Tasks dir: $tasksDir")
        sBuffer.writeEcho("Results dir: $resultsDir")
        sBuffer.writeEcho("Enqueue script: $taskScript")
        sBuffer.writeLine("")

    @staticmethod
    def __configureJob(sBuffer):
        sBuffer.writeComment("Setup local environment")
        sBuffer.writeComment("Get in the directory where this script lives")
        sBuffer.writeLine("pushd $( dirname `greadlink -f ${BASH_SOURCE[0]} || readlink -f ${BASH_SOURCE[0]}` )")
        sBuffer.writeComment("Enumerate the tasks and set up batch params based on task count")
        sBuffer.writeLine("subDirs=($tasksDir/*)")
        sBuffer.writeLine("let arraySize=${#subDirs[@]}")
        sBuffer.writeLine("[ -z \"${arraySize}\" ] && failAndExit \"No task count\";")
        sBuffer.writeLine("let arrayStart=0")
        sBuffer.writeLine("let arrayEnd=$arraySize-1")
        sBuffer.writeComment("Create logs directory for SLURM")
        sBuffer.writeLine("slurmLogsDir=$logsDir/SLURM")
        sBuffer.writeLine("mkdir -p $slurmLogsDir")
        sBuffer.writeLine("")

    @staticmethod
    def __submitJob(sBuffer):
        sBuffer.writeComment("Invoke batch request to SLURM")
        sBuffer.writeComment("Will wait until the task completes before exiting")
        sBuffer.writeLine("runCmd=\"sbatch --nodes=1 --array=$arrayStart-$arrayEnd -o $slurmLogsDir/slurm-%A_%a.out --wait $taskScript -i $imageDir -l $logsDir -r $resultsDir -t $tasksDir\"")
        sBuffer.writeEcho("Submitting batch job to SLURM. This may take a while...")
        sBuffer.writeEcho("$runCmd")
        sBuffer.writeLine("$runCmd")
        sBuffer.writeLine("returnCode=$?")
        sBuffer.writeLine("popd")

    @staticmethod
    def __exit(sBuffer):
        sBuffer.writeEcho("Done!")
        sBuffer.writeLine("exit $returnCode")

    @classmethod
    def writeHeader(cls, sBuffer):
        cls.__initHeader(sBuffer)

    @classmethod
    def writeBody(cls, sBuffer):
        cls.__initializeFuncs(sBuffer)
        cls.__parseArgs(sBuffer)
        cls.__configureJob(sBuffer)
        cls.__submitJob(sBuffer)

    @classmethod
    def writeEpilogue(cls, sBuffer):
        cls.__exit(sBuffer)

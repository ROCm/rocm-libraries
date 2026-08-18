#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCI =
{
    nodeDetails, jobName, buildCommand, label ->

    def prj = new rocProject('rocALUTION', 'PreCheckin')
    // customize for project
    prj.paths.build_command = buildCommand
    prj.compiler.compiler_name = 'c++'
    prj.compiler.compiler_path = 'c++'
    prj.libraryDependencies = ['rocPRIM', 'hipBLAS-common', 'hipBLASLt', 'rocBLAS', 'rocSPARSE', 'rocRAND']

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    def commonGroovy

    boolean formatCheck = false

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        commonGroovy.runCompileCommand(platform, project)
    }

    def testCommand =
    {
        platform, project->

        def gfilter = "*checkin*"
        commonGroovy.runTestCommand(platform, project, gfilter)
    }

    def packageCommand =
    {
        platform, project->

        commonGroovy.runPackageCommand(platform, project, jobName, label)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
}

def setupCI(urlJobName, jobNameList, buildCommand, runCI, label)
{
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    jobNameList.each
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            stage(label + ' ' + jobName) {
                runCI(nodeDetails, jobName, buildCommand, label)
            }
    }

    // For url job names that are not listed by the jobNameList i.e. compute-rocm-dkms-no-npi-1901
    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * *')])]))
        stage(label + ' ' + urlJobName) {
            runCI([ubuntu18:['gfx906']], urlJobName, buildCommand, label)
        }
    }
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = ["compute-rocm-dkms-no-npi":[pipelineTriggers([cron('0 1 * * 0')])],
                        "rocm-docker":[]]

    propertyList = auxiliary.appendPropertyList(propertyList)

    propertyList.each
    {
        jobName, property->
        if (urlJobName == jobName)
            properties(auxiliary.addCommonProperties(property))
    }

    def defaultJobNameList = ["compute-rocm-dkms-no-npi-hipclang":([ubuntu18:['gfx900'],centos7:['gfx906'],centos8:['any'],sles15sp1:['gfx908']]),
                              "rocm-docker":([ubuntu18:['gfx900'],centos7:['gfx906'],sles15sp1:['gfx908']])]

    def hostJobNameList = ["compute-rocm-dkms-no-npi":([ubuntu18:['gfx900']]),
                           "rocm-docker":([ubuntu18:['gfx900']])]

    def mpiJobNameList = ["compute-rocm-dkms-no-npi":([ubuntu18:['gfx900']]),
                          "rocm-docker":([ubuntu18:['gfx900']])]

    String defaultBuildCommand = './install.sh -c'
    String hostBuildCommand = './install.sh -c --host'
    String mpiBuildCommand = './install.sh -c --host --mpi=on --no-openmp'

    setupCI(urlJobName, defaultJobNameList, defaultBuildCommand, runCI, '')
    setupCI(urlJobName, hostJobNameList, hostBuildCommand, runCI, 'Host')
    setupCI(urlJobName, mpiJobNameList, mpiBuildCommand, runCI, 'MPI')
}

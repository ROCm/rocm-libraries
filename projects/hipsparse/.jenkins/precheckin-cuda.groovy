#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path;

def runCI =
{
    nodeDetails, jobName->

    def prj = new rocProject('hipSPARSE', 'PreCheckin-Cuda')

    prj.paths.build_command = './install.sh -c --cuda'
    prj.compiler.compiler_name = 'c++'
    prj.compiler.compiler_path = 'c++'
    prj.defaults.ccache = false

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = false

    def commonGroovy

    def compileCommand =
    {
        platform, project->

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        commonGroovy.runCompileCommand(platform, project)
    }

    def packageCommand =
    {
        platform, project->

        commonGroovy.runPackageCommand(platform, project)
    }

    def testCommand =
    {
        platform, project->

        def gfilter = "*checkin*csrmv*"

        commonGroovy.runTestCommand(platform, project, gfilter)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = [:]
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = [:]
    jobNameList = auxiliary.appendJobNameList(jobNameList, 'hipSPARSE')

    propertyList.each
    {
        jobName, property->
        if (urlJobName == jobName)
        {
            properties(auxiliary.addCommonProperties(property))
        }
    }

    jobNameList.each
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            runCI(nodeDetails, jobName)
    }

    // For url job names that are not listed by the jobNameList i.e. compute-rocm-dkms-no-npi-1901
    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * 6')])]))
        runCI(['ubuntu22-cuda12':['anycuda']], urlJobName)
    }
}

#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCm/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCI =
{
    nodeDetails, jobName->
    def prj  = new rocProject('hipSPARSE', 'CodeCov')

    // customize for project
    prj.paths.build_command = './install.sh -kc --codecoverage'
    prj.compiler.compiler_name = 'c++'
    prj.compiler.compiler_path = 'c++'
    prj.libraryDependencies = ['hipBLAS-common', 'hipBLASLt', 'rocBLAS', 'rocSPARSE', 'rocPRIM']
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

    def testCommand =
    {
        platform, project->

        def gfilter = "**"
        commonGroovy.runCoverageCommand(platform, project, gfilter, "release-debug")
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, testCommand, null)
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = [:]
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = [:]
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    propertyList.each
    {
        jobName, property->
        if (urlJobName == jobName)
            properties(auxiliary.addCommonProperties(property))
    }

    Set seenJobNames = []
    jobNameList.each
    {
        jobName, nodeDetails->
        seenJobNames.add(jobName)
        if (urlJobName == jobName)
            runCI(nodeDetails, jobName)
    }

    // Set standardJobNameSet = ["compute-rocm-dkms-no-npi", "compute-rocm-dkms-no-npi-hipclang", "rocm-docker"]
    // For url job names that are outside of the standardJobNameSet i.e. compute-rocm-dkms-no-npi-1901
    if(!seenJobNames.contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * *')])]))
        runCI([], urlJobName)
    }
}

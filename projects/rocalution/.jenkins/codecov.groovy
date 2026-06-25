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
    nodeDetails, jobName->
    def prj  = new rocProject('rocALUTION', 'CodeCov')

    // customize for project
    prj.paths.build_command = './install.sh -cg --codecoverage'
    prj.compiler.compiler_name = 'c++'
    prj.compiler.compiler_path = 'c++'
    prj.libraryDependencies = ['rocPRIM', 'hipBLAS-common', 'hipBLASLt', 'rocBLAS', 'rocSPARSE', 'rocRAND']
    prj.defaults.ccache = false

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

        def gfilter = "**"
        commonGroovy.runCoverageCommand(platform, project, gfilter, "debug")
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, testCommand, null)
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = []
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = []
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    propertyList.each
    {
        jobName, property->
        if (urlJobName == jobName)
            properties(auxiliary.addCommonProperties(property))
    }

    jobNameList.each
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            stage(jobName) {
                runCI(nodeDetails, jobName)
            }
    }

    // For url job names that are not listed by the jobNameList i.e. compute-rocm-dkms-no-npi-1901
    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * *')])]))
        stage(urlJobName) {
            runCI([], urlJobName)
        }
    }
}

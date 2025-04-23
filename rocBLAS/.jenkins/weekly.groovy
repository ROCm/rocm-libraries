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

    def settings = [formatCheck: false,
                    addressSanitizer: false,
                    gfilter: "*stress*:*HMM*:"]

    def prj = new rocProject('rocBLAS', 'weekly')

    // customize for project
    prj.paths.build_command = './install.sh -c'

    def noHipblasLT = env.BRANCH_NAME ==~ /PR-\d+/ && pullRequest.labels.contains("noHipblasLT")
    if (!noHipblasLT)
    {
        prj.libraryDependencies = ['hipBLAS-common', 'hipBLASLt']
    }

    prj.defaults.ccache = false
    prj.timeout.compile = 480
    prj.timeout.test = 600

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    def compileCommand =
    {
        platform, project->

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        commonGroovy.runCompileCommand(platform, project, jobName, settings)
    }

    def testCommand =
    {
        platform, project->

        commonGroovy.runTestCommand(platform, project, settings)
    }

    def packageCommand =
    {
        platform, project->

        commonGroovy.runPackageCommand(platform, project)
    }

    buildProject(prj, settings.formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)

}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = ["main":[pipelineTriggers([cron('0 1 * * 6')])]]
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = ["main":([ubuntu20:['gfx90a']])]

    jobNameList = auxiliary.appendJobNameList(jobNameList, 'rocBLAS')

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
            runCI([ubuntu20:['gfx90a']], urlJobName)
        }
    }
}

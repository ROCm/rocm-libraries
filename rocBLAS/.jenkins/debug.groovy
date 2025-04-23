#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for AMD Continuous Integration use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path


def runCI =
{
    nodeDetails, jobName->

    def settings = [formatCheck: false,
                    addressSanitizer: false,
                    gfilter: "*quick*:*pre_checkin*"]

    def prj = new rocProject('rocBLAS', 'Debug')

    // customize for project
    prj.paths.build_command = './install.sh -c -g'

    def noHipblasLT = env.BRANCH_NAME ==~ /PR-\d+/ && pullRequest.labels.contains("noHipblasLT")
    if (!noHipblasLT)
    {
        prj.libraryDependencies = ['hipBLAS-common', 'hipBLASLt']
    }

    prj.defaults.ccache = false

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)
    prj.timeout.compile = 480

    def compileCommand =
    {
        platform, project->

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        commonGroovy.runCompileCommand(platform, project, jobName, settings)
    }

    def packageCommand =
    {
        platform, project->

        commonGroovy.runPackageCommand(platform, project, debug=true)
    }

    buildProject(prj, settings.formatCheck, nodes.dockerArray, compileCommand, null, packageCommand)

}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = ["compute-rocm-dkms-no-npi":[pipelineTriggers([cron('0 1 * * 0')])],
                        "compute-rocm-dkms-no-npi-hipclang":[pipelineTriggers([cron('0 1 * * 0')])],
                        "rocm-docker":[]]
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = ["compute-rocm-dkms-no-npi":([ubuntu18:['gfx900']]),
                       "compute-rocm-dkms-no-npi-hipclang":([ubuntu18:['gfx900']]),
                       "rocm-docker":([ubuntu18:['gfx900']])]
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
            runCI([ubuntu18:['gfx900']], urlJobName)
        }
    }
}

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

    def prj = new rocProject('rocSPARSE', 'Extended')

    // customize for project
    if (env.BRANCH_NAME ==~ /PR-\d+/){
        prj.paths.build_command = './install.sh --matrices-dir-install ${JENKINS_HOME_DIR}/rocsparse_matrices && ./install.sh -c -a \$gfx_arch --matrices-dir ${JENKINS_HOME_DIR}/rocsparse_matrices'
    }
    else{
    prj.paths.build_command = './install.sh --matrices-dir-install ${JENKINS_HOME_DIR}/rocsparse_matrices && ./install.sh -c --matrices-dir ${JENKINS_HOME_DIR}/rocsparse_matrices'
    }
    prj.libraryDependencies = ['rocPRIM', 'hipBLAS-common', 'hipBLASLt', 'rocBLAS']
    prj.timeout.test = 600
    prj.defaults.ccache = false

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    def commonGroovy

    boolean formatCheck = false

    def compileCommand =
    {
        platform, project->

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        commonGroovy.runCompileCommand(platform, project, jobName, false)
    }

    def testCommand =
    {
        platform, project->

        def gfilter = "*nightly*"
        commonGroovy.runTestCommand(platform, project, gfilter)
    }

    def packageCommand =
    {
        platform, project->

        commonGroovy.runPackageCommand(platform, project)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = ["compute-rocm-dkms-no-npi":[pipelineTriggers([cron('0 1 * * 6')])],
                        "compute-rocm-dkms-no-npi-hipclang":[pipelineTriggers([cron('0 1 * * 6')])],
                        "rocm-docker":[]]
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = ["compute-rocm-dkms-no-npi":([ubuntu18:['gfx900'],centos7:['gfx908'],sles15sp1:['gfx906']]),
                       "compute-rocm-dkms-no-npi-hipclang":([ubuntu18:['gfx900'],centos7:['gfx908'],centos8:['any'],sles15sp1:['gfx906']]),
                       "rocm-docker":([ubuntu18:['gfx900'],centos7:['gfx906'],sles15sp1:['gfx908']])]
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
            runCI(nodeDetails, jobName)
    }

    // For url job names that are not listed by the jobNameList i.e. compute-rocm-dkms-no-npi-1901
    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * *')])]))
        runCI([ubuntu18:['gfx900', 'gfx906']], urlJobName)
    }
}

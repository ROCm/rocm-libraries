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
    def prj  = new rocProject('rocSPARSE', 'CodeCov')

    // customize for project
    if (env.BRANCH_NAME ==~ /PR-\d+/){
        prj.paths.build_command = './install.sh --matrices-dir-install ${JENKINS_HOME_DIR}/rocsparse_matrices && ./install.sh -kc -a \$gfx_arch --no-rocblas --codecoverage --matrices-dir ${JENKINS_HOME_DIR}/rocsparse_matrices'
    }
    else {
        prj.paths.build_command = './install.sh --matrices-dir-install ${JENKINS_HOME_DIR}/rocsparse_matrices && ./install.sh -kc --no-rocblas --codecoverage --matrices-dir ${JENKINS_HOME_DIR}/rocsparse_matrices'
    }
    prj.libraryDependencies = ['rocPRIM']
    prj.defaults.ccache = false

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    def commonGroovy

    boolean formatCheck = false

    def compileCommand =
    {
        platform, project->

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        commonGroovy.runCompileCommand(platform, project, jobName)
    }

    def testCommand =
    {
        platform, project->

        def gfilter = "*pre_checkin*"
        commonGroovy.runCoverageCommand(platform, project, gfilter, "release-debug")
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
            runCI(nodeDetails, jobName)
    }
}

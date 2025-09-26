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

    def prj = new rocProject('rocSPARSE', 'Debug')

    // customize for project
    if (env.BRANCH_NAME ==~ /PR-\d+/){
        prj.paths.build_command = './install.sh --matrices-dir-install ${JENKINS_HOME_DIR}/rocsparse_matrices && ./install.sh -c -a \$gfx_arch -g --matrices-dir ${JENKINS_HOME_DIR}/rocsparse_matrices'
    }
    else {
    prj.paths.build_command = './install.sh --matrices-dir-install ${JENKINS_HOME_DIR}/rocsparse_matrices && ./install.sh -c -g --matrices-dir ${JENKINS_HOME_DIR}/rocsparse_matrices'
    }
    prj.libraryDependencies = ['rocPRIM', 'hipBLAS-common', 'hipBLASLt', 'rocBLAS']
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

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, null, null)
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = ["compute-rocm-dkms-no-npi-hipclang":[pipelineTriggers([cron('0 1 * * 6')])]]
    propertyList = auxiliary.appendPropertyList(propertyList)
    def jobNameList = ["compute-rocm-dkms-no-npi-hipclang":([ubuntu18:['any']])]
    jobNameList = auxiliary.appendJobNameList(jobNameList)

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

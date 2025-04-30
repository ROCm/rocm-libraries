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

    def prj  = new rocProject('rocFFT-internal', 'multigpu')

    prj.defaults.ccache = true
    prj.timeout.compile = 600
    prj.timeout.test = 600
    prj.libraryDependencies = ['rocRAND','hipRAND']

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = false

    def commonGroovy

    def compileCommand =
    {
        platform, project->

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
	    // build with MPI enabled
        commonGroovy.runCompileCommand(platform, project, jobName, false, false, true)
        commonGroovy.runCompileClientCommand(platform, project, jobName, false)
    }

    def testCommand =
    {
        platform, project->

	    //run single-process multi-GPU tests
        commonGroovy.runTestCommand(platform, project, false, "*multi_gpu*")
	    // run MPI tests across 4 ranks
        commonGroovy.runTestCommand(platform, project, false, "*multi_gpu*", '--mp_lib mpi --mp_ranks 4 --mp_launch "/usr/local/openmpi/bin/mpirun --np 4 ./rocfft_mpi_worker"')
    }

    def packageCommand =
    {
        platform, project->

	    // don't package anything - we're not distributing MPI-enabled rocFFT so we don't want to expose any MPI-enabled packages anywhere that other builds can mistakenly pick up
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
}

ci: { 
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = ["main":[pipelineTriggers([cron('0 1 * * 0')])]]
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = ["main":([ubuntu20:['8gfx90a']])]
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
            runCI([ubuntu20:['8gfx90a']], urlJobName)
        }
    }
}

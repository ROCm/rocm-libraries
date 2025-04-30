#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runBitwiseReproTest (platform, project, boolean debug=false, gfilter='', reprodb='', int repeat=1)
{
    String testBinaryName = 'rocfft-test'
    String directory = debug ? 'debug' : 'release'

    String gfilterArg = ''
    if (gfilter)
    {
        gfilterArg = "--gtest_filter=${gfilter}"
    }

    String reproDbArg = ''
    if (reprodb)
    {
        reproDbArg = "--repro-db=${reprodb}"
    }    

    String repeatArg = ''
    if (repeat > 1)
    {
        repeatArg = "--gtest_repeat=${repeat}"
    }

    def command = """#!/usr/bin/env bash
                set -ex
                cd ${project.paths.project_build_prefix}/build/${directory}/clients/staging
                ROCM_PATH=/opt/rocm GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./${testBinaryName} --precompile=rocfft-test-precompile.db ${gfilterArg} ${reproDbArg} ${repeatArg} --gtest_color=yes --R 80 --nrand 10
            """
    platform.runCommand(this, command)
}

def runCI = 
{
    nodeDetails, jobName->

    def prj  = new rocProject('rocFFT-internal', 'Extended')

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
        commonGroovy.runCompileCommand(platform, project, jobName)
        commonGroovy.runCompileClientCommand(platform, project, jobName, false)
    }

    def testCommand =
    {
        platform, project->          

        runBitwiseReproTest(platform, project, false, "*pow2_1D/bitwise_repro_test*", 'bitwise_repro.db', 2)
    }

    def packageCommand =
    {
        platform, project->
        
        commonGroovy.runPackageCommand(platform, project, jobName)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
}

ci: { 
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = ["compute-rocm-dkms-no-npi-hipclang":[pipelineTriggers([cron('0 1 * * 0')])]]
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = ["compute-rocm-dkms-no-npi-hipclang":([ubuntu18:['gfx900'],centos7:['gfx906'],centos8:['gfx906'],sles15sp1:['gfx908']])]
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
            runCI([ubuntu18:['gfx906']], urlJobName)
        }
    }
}

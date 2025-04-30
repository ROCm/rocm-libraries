#!/usr/bin/env groovy
@Library('rocJenkins@pong') _

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCI =
{
    nodeDetails, jobName, buildCommand, label, runTest ->

    def prj = new rocProject('hipFFT', 'multigpu')
    // customize for project
    prj.paths.build_command = buildCommand
    prj.libraryDependencies = ['rocRAND', 'rocFFT', 'hipRAND']
    prj.timeout.test = 360

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    def commonGroovy

    boolean formatCheck = false

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        commonGroovy.runCompileCommand(platform, project,jobName)
    }

    def testCommand =
    {
        platform, project->

        def gfilter = "*multi_gpu*"
        commonGroovy.runTestCommand(platform, project, gfilter)
    }

    def packageCommand =
    {
        platform, project->

        commonGroovy.runPackageCommand(platform, project, jobName, label)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, runTest ? testCommand : null, packageCommand)
}

def setupCI(urlJobName, jobNameList, buildCommand, runCI, label, runTest)
{
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    jobNameList.each
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            stage(label + ' ' + jobName) {
                runCI(nodeDetails, jobName, buildCommand, label, runTest)
            }
    }

    // For url job names that are not listed by the jobNameList i.e. compute-rocm-dkms-no-npi-1901
    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * *')])]))
        stage(label + ' ' + urlJobName) {
            runCI([ubuntu18:['gfx906']], urlJobName, buildCommand, label)
        }
    }
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = ["main":[pipelineTriggers([cron('0 6 * * 0')])]]

    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = ["main":([ubuntu20:['8gfx90a']])]
    jobNameList = auxiliary.appendJobNameList(jobNameList)
    
    propertyList.each
    {
        jobName, property->
        if (urlJobName == jobName)
            properties(auxiliary.addCommonProperties(property))
    }
    
    String hipClangBuildCommand = '-DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_SAMPLES=ON -L ../..'

    setupCI(urlJobName, jobNameList, hipClangBuildCommand, runCI, 'hip-clang', true)
}

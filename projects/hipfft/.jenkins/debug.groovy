#!/usr/bin/env groovy
@Library('rocJenkins@pong') _

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCI =
{
    nodeDetails, jobName, buildCommand ->

    def prj = new rocProject('hipFFT', 'Debug')
    // customize for project
    prj.paths.build_command = buildCommand
    prj.libraryDependencies = ['rocRAND', 'rocFFT', 'hipRAND']

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    def commonGroovy

    boolean formatCheck = false

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        commonGroovy.runCompileCommand(platform, project, jobName)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, null, null)
}

def setupCI(urlJobName, jobNameList, buildCommand, runCI, label)
{
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    jobNameList.each
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            stage(label + ' ' + jobName) {
                runCI(nodeDetails, jobName, buildCommand, label)
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

    def propertyList = ["compute-rocm-dkms-no-npi":[pipelineTriggers([cron('0 1 * * 0')])]]

    propertyList = auxiliary.appendPropertyList(propertyList)

    propertyList.each
    {
        jobName, property->
        if (urlJobName == jobName)
            properties(auxiliary.addCommonProperties(property))
    }

    def hostJobNameList = ["compute-rocm-dkms-no-npi":([ubuntu18:['gfx900'],centos7:['gfx906'],sles15sp1:['gfx906']])]

    def hipClangJobNameList = ["compute-rocm-dkms-no-npi":([ubuntu18:['gfx900'],centos7:['gfx906'],sles15sp1:['gfx906']])]

    String hostBuildCommand = '-DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Debug -L ../..'
    String hipClangBuildCommand = '-DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ -DCMAKE_BUILD_TYPE=Debug -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_SAMPLES=ON -L ../..'

    setupCI(urlJobName, hostJobNameList, hostBuildCommand, runCI, 'g++')
    setupCI(urlJobName, hipClangJobNameList, hipClangBuildCommand, runCI, 'hip-clang')
}

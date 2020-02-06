#!/usr/bin/env groovy
@Library('rocJenkins@pong') _
import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path;

def runCI = 
{
    nodeDetails, jobName->

    def prj = new rocProject('rocPRIM', 'PreCheckin')

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

        commonGroovy.runTestCommand(platform, project)
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

    def propertyList = ["compute-rocm-dkms-no-npi":[pipelineTriggers([cron('0 1 * * 0')])], 
                        "compute-rocm-dkms-no-npi-hipclang":[pipelineTriggers([cron('0 1 * * 0')])],
                        "rocm-docker":[]]
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = ["compute-rocm-dkms-no-npi":([ubuntu16:['gfx900'],centos7:['gfx906'],sles15sp1:['gfx908']]), 
                       "compute-rocm-dkms-no-npi-hipclang":([ubuntu16:['gfx900'],centos7:['gfx906'],sles15sp1:['gfx908']]), 
                       "rocm-docker":([ubuntu16:['gfx900'],centos7:['gfx906'],sles15sp1:['gfx908']])]
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

    // For url job names that are outside of the standardJobNameSet i.e. compute-rocm-dkms-no-npi-1901
    Set standardJobNameSet = ["compute-rocm-dkms-no-npi", "compute-rocm-dkms-no-npi-hipclang", "rocm-docker"]
    if(!standardJobNameSet.contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * *')])]))
        stage(jobName) {
            runCI([ubuntu16:['gfx906']], urlJobName)
        }
    }
}

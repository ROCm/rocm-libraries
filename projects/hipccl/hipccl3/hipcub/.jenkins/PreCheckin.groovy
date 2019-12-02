#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins') _

// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path;

hipCUBCI:
{
    def hipcub = new rocProject('hipCUB', 'PreCheckin')

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(['ubuntu && gfx803', 'sles && gfx906', 'gfx900 && centos7', 'gfx906 && centos7'], hipcub)

    boolean formatCheck = false

    def commonGroovy

    def compileCommand =
    {
        platform, project->

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/Common.groovy"
        commonGroovy.runCompileCommand(platform, project)
    }

    def testCommand =
    {
        platform, project->

        commonGroovy.runTestCommand(platform, project)
    }

    def packageCommand =
    {
        platform, project->
        
        commonGroovy.runPackageCommand(platform, project)
    }

    buildProject(hipcub, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
}


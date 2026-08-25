// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.
import static groovy.io.FileType.FILES

def runCompileCommand(platform, project, boolean sameOrg=false)
{
    project.paths.construct_build_prefix()
    String centos7devtoolset = platform.jenkinsLabel.contains('centos7') ? 'source /etc/profile.d/modules.sh && source scl_source enable devtoolset-7 && module load mpi/openmpi-x86_64' : ''

    def getDependenciesCommand = ""
    if (project.installLibraryDependenciesFromCI)
    {
        project.libraryDependencies.each
        { libraryName ->
            getDependenciesCommand += auxiliary.getLibrary(libraryName, platform.jenkinsLabel, 'develop', sameOrg)
        }
    }

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                ${getDependenciesCommand}
                ${centos7devtoolset}
                ${project.paths.build_command}
            """

    platform.runCommand(this, command)
}

def runTestCommand (platform, project, gfilter)
{
    def hmmTestCommand= ''
    if (platform.jenkinsLabel.contains('gfx90a'))
    {
        hmmTestCommand = """
                            HSA_XNACK=0 GTEST_LISTENER=NO_PASS_LINE_IN_LOG  ./rocalution-test --gtest_output=xml:test_detail_hmm_xnack_off.xml --gtest_color=yes --gtest_filter=**
                            HSA_XNACK=1 GTEST_LISTENER=NO_PASS_LINE_IN_LOG  ./rocalution-test --gtest_output=xml:test_detail_hmm_xnack_on.xml --gtest_color=yes --gtest_filter=**
                         """
    }

    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release/clients/staging
                    ${sudo} LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocalution-test --gtest_output=xml --gtest_color=yes #--gtest_filter=${gfilter}-*known_bug*
                    ${hmmTestCommand}
                """

    platform.runCommand(this, command)
}

def runCoverageCommand (platform, project, gfilter, String dirmode = "release")
{
    String commitSha
    String repoUrl
    (commitSha, repoUrl) = util.getGitHubCommitInformation(project.paths.project_src_prefix)

    withCredentials([string(credentialsId: "mathlibs-codecov-token-rocalution", variable: 'CODECOV_TOKEN')])
    {
        def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/${dirmode}
                    export LD_LIBRARY_PATH=/opt/rocm/lib/
                    export ROCALUTION_CODE_COVERAGE=1
                    GTEST_LISTENER=NO_PASS_LINE_IN_LOG make coverage_cleanup coverage GTEST_FILTER=${gfilter}-*known_bug*
                    curl -Os https://uploader.codecov.io/latest/linux/codecov
                    chmod +x codecov
                    ./codecov -v -U \$http_proxy -t ${CODECOV_TOKEN} --file lcoverage/main_coverage.info --name rocALUTION --sha ${commitSha}
                """

        platform.runCommand(this, command)
    }

    publishHTML([allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: false,
                reportDir: "${project.paths.project_build_prefix}/build/${dirmode}/lcoverage",
                reportFiles: "index.html",
                reportName: "Code coverage report",
                reportTitles: "Code coverage report"])
}

def runPackageCommand(platform, project, jobName, label='')
{
    def command

    label = label != '' ? '-' + label.toLowerCase() : ''

    if(platform.jenkinsLabel.contains('centos') || platform.jenkinsLabel.contains('sles'))
    {
        command = """
                set -x
                cd ${project.paths.project_build_prefix}/build/release
                make package
                mkdir -p package
                if [ ! -z "$label" ]
                then
                    for f in rocalution*.rpm
                    do
                        echo f
                        mv "\$f" "rocalution${label}-\${f#*-}"
                        ls
                    done
                fi
                mv *.rpm package/
                rpm -qlp package/*.rpm
            """

        platform.runCommand(this, command)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.rpm""")
    }
    else
    {
        command = """
                set -x
                cd ${project.paths.project_build_prefix}/build/release
                make package
                mkdir -p package
                if [ ! -z "$label" ]
                then
                    for f in rocalution*.deb
                    do
                        echo f
                        mv "\$f" "rocalution${label}-\${f#*-}"
                        ls
                    done
                fi
                mv *.deb package/
                for pkg in package/*.deb; do
                    dpkg -c \$pkg
                done
            """

        platform.runCommand(this, command)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.deb""")
    }
}


return this

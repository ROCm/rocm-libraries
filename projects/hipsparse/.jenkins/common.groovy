// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, boolean sameOrg=false)
{
    project.paths.construct_build_prefix()

    def command
    def getDependenciesCommand = ""
    if (project.installLibraryDependenciesFromCI)
    {
        project.libraryDependencies.each
        { libraryName ->
            getDependenciesCommand += auxiliary.getLibrary(libraryName, platform.jenkinsLabel, 'develop', sameOrg)
        }
    }
    String centos7 = platform.jenkinsLabel.contains('centos7') ? 'source scl_source enable devtoolset-7' : ':'

    command = """#!/usr/bin/env bash
                set -x
                ${centos7}
                cd ${project.paths.project_build_prefix}
                ${getDependenciesCommand}
                ${pullMatricesCommand()}
                CXX=${project.compiler.compiler_path} ${project.paths.build_command}
            """

    platform.runCommand(this, command)
}

// Fetches the DVC-tracked SuiteSparse archives in clients/matrices, which the client tests require
def pullMatricesCommand()
{
    // The sparse checkout may omit the .dvc directory at the repository root.
    // Jenkins runs the container as a UID without a passwd entry, so dvc needs LOGNAME
    // (https://docs.python.org/3/library/getpass.html#getpass.getuser).
    return """
                (
                    set -e
                    repo_root=\$(git rev-parse --show-toplevel)
                    if [ ! -f "\${repo_root}/.dvc/config" ]; then
                        mkdir -p "\${repo_root}/.dvc"
                        git -C "\${repo_root}" show HEAD:.dvc/config > "\${repo_root}/.dvc/config"
                    fi
                    dvc_cmd=dvc
                    if ! command -v dvc > /dev/null; then
                        dvc_venv="\${PWD}/build/dvc-venv"
                        python3 -m venv "\${dvc_venv}"
                        "\${dvc_venv}/bin/python" -m pip install --quiet 'dvc[s3]'
                        dvc_cmd="\${dvc_venv}/bin/dvc"
                    fi
                    cd "\${repo_root}"
                    LOGNAME=jenkins DVC_NO_ANALYTICS=1 \${dvc_cmd} pull -R projects/hipsparse/clients/matrices
                ) || exit 1
            """
}

def runTestCommand (platform, project, gfilter)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release/clients/staging
                    ${sudo} GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./hipsparse-test --gtest_also_run_disabled_tests --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}-*known_bug*
                """

    platform.runCommand(this, command)
}

def runCoverageCommand (platform, project, gfilter, String dirmode = "release")
{
    String commitSha
    String repoUrl
    (commitSha, repoUrl) = util.getGitHubCommitInformation(project.paths.project_src_prefix)

    withCredentials([string(credentialsId: "mathlibs-codecov-token-rocm-libraries", variable: 'CODECOV_TOKEN')])
    {
        def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/${dirmode}
                    export LD_LIBRARY_PATH=/opt/rocm/lib/
                    GTEST_LISTENER=NO_PASS_LINE_IN_LOG make coverage_cleanup coverage GTEST_FILTER=${gfilter}-*known_bug*
                    curl -Os https://uploader.codecov.io/latest/linux/codecov
                    chmod +x codecov
                    ./codecov -v -U \$http_proxy -t ${CODECOV_TOKEN} --file lcoverage/main_coverage.info --name rocm-libraries --flags hipSPARSE --sha ${commitSha}
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

def runPackageCommand(platform, project)
{
    def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/release")
    platform.runCommand(this, packageHelper[0])
    platform.archiveArtifacts(this, packageHelper[1])
}

return this

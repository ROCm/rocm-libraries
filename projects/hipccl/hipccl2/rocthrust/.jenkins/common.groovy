// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName)
{
    project.paths.construct_build_prefix()

    String compiler = jobName.contains('hipclang') ? 'hipcc' : 'hcc'
    String hipClangArgs = jobName.contains('hipclang') ? '--hip-clang' : ''

    command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                LD_LIBRARY_PATH=/opt/rocm/hcc/lib ${project.paths.build_command} ${hipClangArgs} 
            """

    
    platform.runCommand(this, command)
}

def runTestCommand (platform, project)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)

	def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release
                    make -j\$(nproc)
                    ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib ctest --output-on-failure
                """

    platform.runCommand(this, command)
}

def runPackageCommand(platform, project, jobName)
{
    def command
        
    if(jobName.contains('hipclang'))
    {
        packageCommand = null
    }
    else if(platform.jenkinsLabel.contains('ubuntu'))
    {
        command = """
                set -x
                cd ${project.paths.project_build_prefix}/build/release
                make package
                rm -rf package && mkdir -p package
                mv *.deb package/
                dpkg -c package/*.deb
                """        
        
        platform.runCommand(this, command)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.deb""")
    }
	else
	{
        command = """
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release
                    make package
                    rm -rf package && mkdir -p package
                    mv *.rpm package/
                    rpm -qlp package/*.rpm
                    """
        
        platform.runCommand(this, command)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.rpm""")
    }
}

return this


# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

param(
    [Parameter(Mandatory = $true)]
    [string]$ClangFormat,

    [Parameter(Mandatory = $true)]
    [string]$SourceDir,

    [Parameter(Mandatory = $true)]
    [ValidateSet("check", "format")]
    [string]$Mode,

    [Parameter(Mandatory = $true)]
    [int]$FilesPerInvocation,

    [Parameter(Mandatory = $true)]
    [int]$Jobs
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

if($FilesPerInvocation -lt 1)
{
    throw "FilesPerInvocation must be greater than 0"
}

if($Jobs -lt 0)
{
    throw "Jobs must be greater than or equal to 0"
}

if($Jobs -eq 0)
{
    $Jobs = [Environment]::ProcessorCount
}
if($Jobs -lt 1)
{
    $Jobs = 1
}

if($Mode -eq "check")
{
    $ModeArgs = @("--dry-run", "--Werror")
}
else
{
    $ModeArgs = @("--verbose", "-i")
}

$SourceFullPath = [System.IO.Path]::GetFullPath($SourceDir)
if(-not $SourceFullPath.EndsWith([System.IO.Path]::DirectorySeparatorChar.ToString()))
{
    $SourceFullPath += [System.IO.Path]::DirectorySeparatorChar
}

$ExcludedRelativePrefixes = @(
    "build/",
    "flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/data_objects/"
)
$IncludedExtensions = @(".cpp", ".hpp", ".c", ".h")

function Get-RelativeFormatPath
{
    param([Parameter(Mandatory = $true)][string]$FilePath)

    $FileFullPath = [System.IO.Path]::GetFullPath($FilePath)
    if($FileFullPath.StartsWith($SourceFullPath, [System.StringComparison]::OrdinalIgnoreCase))
    {
        $RelativePath = $FileFullPath.Substring($SourceFullPath.Length)
    }
    else
    {
        $RelativePath = $FileFullPath
    }

    return $RelativePath.Replace([System.IO.Path]::DirectorySeparatorChar, '/')
}

function Test-IsExcludedPath
{
    param([Parameter(Mandatory = $true)][string]$FilePath)

    $RelativePath = Get-RelativeFormatPath -FilePath $FilePath
    foreach($Prefix in $ExcludedRelativePrefixes)
    {
        if($RelativePath.StartsWith($Prefix, [System.StringComparison]::OrdinalIgnoreCase))
        {
            return $true
        }
    }

    return $false
}

$FormatFiles = Get-ChildItem -LiteralPath $SourceDir -Recurse -File |
    Where-Object {
        $IncludedExtensions -contains $_.Extension.ToLowerInvariant() -and
        -not (Test-IsExcludedPath -FilePath $_.FullName)
    } |
    Sort-Object FullName

if(-not $FormatFiles)
{
    Write-Host "No source files found for clang-format"
    exit 0
}

$Chunks = @()
$CurrentChunk = New-Object 'System.Collections.Generic.List[string]'
foreach($File in $FormatFiles)
{
    $CurrentChunk.Add($File.FullName)
    if($CurrentChunk.Count -ge $FilesPerInvocation)
    {
        $Chunks += ,@($CurrentChunk.ToArray())
        $CurrentChunk.Clear()
    }
}
if($CurrentChunk.Count -gt 0)
{
    $Chunks += ,@($CurrentChunk.ToArray())
}

$RunningJobs = @()
$HadFailure = $false

function Receive-FormatJob
{
    param([Parameter(Mandatory = $true)]$Job)

    $Results = Receive-Job -Job $Job
    Remove-Job -Job $Job

    foreach($Result in $Results)
    {
        foreach($Line in $Result.Output)
        {
            $OutputLine = $Line.ToString()
            if(-not [string]::IsNullOrWhiteSpace($OutputLine))
            {
                Write-Output $OutputLine
            }
        }

        if($Result.ExitCode -ne 0)
        {
            $script:HadFailure = $true
        }
    }
}

foreach($Chunk in $Chunks)
{
    while($RunningJobs.Count -ge $Jobs)
    {
        $CompletedJob = Wait-Job -Job $RunningJobs -Any
        Receive-FormatJob -Job $CompletedJob
        $RunningJobs = @($RunningJobs | Where-Object { $_.Id -ne $CompletedJob.Id })
    }

    $Payload = [pscustomobject]@{
        Executable = $ClangFormat
        Arguments = $ModeArgs
        Files = $Chunk
    }

    $RunningJobs += Start-Job -ScriptBlock {
        param($Payload)

        try
        {
            $Output = & $Payload.Executable @($Payload.Arguments) @($Payload.Files) 2>&1 |
                ForEach-Object { $_.ToString() }
            $ExitCode = $LASTEXITCODE
            if($null -eq $ExitCode)
            {
                $ExitCode = 1
            }
        }
        catch
        {
            $Output = $_
            $ExitCode = 1
        }

        [pscustomobject]@{
            ExitCode = $ExitCode
            Files = $Payload.Files
            Output = $Output
        }
    } -ArgumentList $Payload
}

while($RunningJobs.Count -gt 0)
{
    $CompletedJob = Wait-Job -Job $RunningJobs -Any
    Receive-FormatJob -Job $CompletedJob
    $RunningJobs = @($RunningJobs | Where-Object { $_.Id -ne $CompletedJob.Id })
}

if($HadFailure)
{
    exit 1
}
